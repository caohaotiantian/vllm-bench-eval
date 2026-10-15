"""Protect persisted runs and preserve evidence during explicit recovery."""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import uuid


class LifecycleError(ValueError):
    """The persisted run cannot support the requested operation."""


@contextmanager
def run_lock(path: Path):
    """Hold an operating system lock without deleting its stable lock file."""
    path = path.resolve()
    with path.with_suffix(path.suffix + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LifecycleError("another process already owns this run") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_events(path: Path, *, repair=False) -> list[dict]:
    """Read complete events and preserve the original file before tail repair."""
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    events = []
    trimmed = False
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index != len(lines) - 1 or line.endswith((b"\n", b"\r")):
                raise LifecycleError("event file contains invalid data") from exc
            if not repair:
                raise LifecycleError("event file has a partial final record; use the recovery command") from exc
            trimmed = True
            break
        if not isinstance(event, dict):
            raise LifecycleError("event record must be an object")
        events.append(event)
    if repair and (trimmed or (raw and not raw.endswith(b"\n"))):
        backup = path.with_name(path.name + ".backup-" + uuid.uuid4().hex)
        with backup.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        content = b"".join(lines[:len(events)])
        if content and not content.endswith(b"\n"):
            content += b"\n"
        temporary = path.with_name(path.name + ".repair-" + uuid.uuid4().hex)
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return events


def settle_attempts(result: dict, events: list[dict]) -> None:
    """Estimate unfinished attempt durations from the last persisted timestamp."""
    for window in result.get("attempts", []):
        if "duration_s" in window:
            continue
        matching = [event for event in events if event.get("run_attempt_id") == window["attempt_id"]]
        end = max((event["timestamp"] for event in matching), default=window["start_time"])
        window.update(end_time=end,
                      duration_s=max(0, (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(window["start_time"])).total_seconds()),
                      duration_estimated=True, duration_estimation="last_persisted_event_may_underestimate_duration")


def snapshot_identity(result, events, destination):
    """Separate immutable result identity from the delivery destination."""
    import hashlib
    content = {key: value for key, value in result.items() if not key.startswith("_") and key != "events_path"}
    snapshot = hashlib.sha256(json.dumps({"result": content, "events": events}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    receipt = hashlib.sha256(json.dumps({"snapshot": snapshot, "destination": destination}, sort_keys=True).encode()).hexdigest()
    return snapshot, receipt


@contextmanager
def capture_stop_signals(controller):
    """Record pending signal intent without locks, events, or file writes."""
    import signal
    import threading
    previous = {}
    def handler(signum, frame):
        controller.pending_signal = signum
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


class MeasurementRecorder:
    """Record load and request boundaries against one attempt monotonic clock."""
    def __init__(self, config, attempt_id, started, target, mode, request_stop):
        self.config = config.model_dump() if hasattr(config, "model_dump") else dict(config)
        self.enabled = self.config["enabled"]
        self.attempt_id, self.started = attempt_id, started
        self.target, self.mode, self.request_stop = target, mode, request_stop
        self.pending = self.active = self.inflight = 0
        self.reached = False
        self.journal = None

    @property
    def deadline(self):
        duration = self.config["duration_sec"]
        if self.enabled and duration is not None:
            return self.started + self.config["warmup_sec"] + duration
        return None

    def load(self):
        return {"pending": self.pending, "active_conversations" if self.mode == "conversation" else "active_tasks": self.active,
                "inflight_requests": self.inflight, "target_active": self.target, "target_reached": self.reached}

    def record_event(self, event):
        if not self.enabled:
            return
        import time
        if event["event"] != "load" or "attempt_offset_s" not in event:
            event["attempt_offset_s"] = max(0, time.monotonic() - self.started)
        if event["event"] == "llm_start":
            self.inflight += 1
        elif event["event"] == "llm_end":
            self.inflight = max(0, self.inflight - 1)
        if event["event"] in {"load", "llm_start", "llm_end"}:
            event["load"] = self.load()

    def change_load(self, *, pending=0, active=0, reason):
        if not self.enabled:
            return
        import time
        with self.journal.lock:
            self.pending += pending
            self.active += active
            self.reached = self.reached or self.active >= self.target
            cutoff = time.monotonic()
            self.journal.emit({"event": "load", "reason": reason, "attempt_offset_s": max(0, cutoff - self.started)})
            drop = self.config["stop_on_load_drop"] and self.reached and self.pending == 0 and self.active < self.target
        if drop:
            self.request_stop("load_drop", at=cutoff)

    def record_stop(self, reason, offset, signum=None):
        if self.journal is not None:
            self.journal.emit({"event": "run_stop", "reason": reason, "requested_offset_s": offset, "signal": signum})


def measure_attempt(config, attempt_id, events, elapsed, stop_offset=None, stop_reason=None, target=1, mode="conversation"):
    """Select complete successful requests without changing the raw events."""
    from .agent_metrics import aggregate
    config = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    if not config["enabled"]:
        return None
    start = config["warmup_sec"]
    candidates = [(elapsed, "execution_end")]
    if config["duration_sec"] is not None:
        candidates.append((start + config["duration_sec"], "measurement_duration"))
    if stop_offset is not None:
        candidates.append((stop_offset, stop_reason or "stop"))
    cutoff, reason = min(candidates, key=lambda item: item[0])
    duration = max(0, cutoff - start)
    scoped = [event for event in events if event.get("run_attempt_id") == attempt_id]
    starts = {event["span_id"]: event for event in scoped if event["event"] == "llm_start"}
    ends = {event["span_id"]: event for event in scoped if event["event"] == "llm_end"}
    selected = []
    counts = {"started_in_window": 0, "included_successful": 0, "failed_started_in_window": 0,
              "cancelled_started_in_window": 0, "cross_boundary_successful_excluded": 0, "missing_boundary_evidence": 0}
    for span, request in starts.items():
        end = ends.get(span)
        before, after = request.get("attempt_offset_s"), (end or {}).get("attempt_offset_s")
        if before is None:
            counts["missing_boundary_evidence"] += 1
            continue
        in_start = start <= before < cutoff
        counts["started_in_window"] += int(in_start)
        if after is None or after < before:
            counts["missing_boundary_evidence"] += 1
            continue
        in_end = start <= after < cutoff
        success = end is not None and "output" in end and not end.get("error")
        if success and in_start and in_end:
            selected.append(span)
            counts["included_successful"] += 1
        elif success and before < cutoff and after >= start:
            counts["cross_boundary_successful_excluded"] += 1
        elif in_start and end is not None:
            counts["cancelled_started_in_window" if end.get("status") == "cancelled" else "failed_started_in_window"] += 1
    selected_events = [event for event in scoped if event.get("span_id") in selected and event["event"] in {"llm_start", "llm_end"}]
    metrics = aggregate(selected_events, duration)
    _null_empty_samples(metrics)
    trajectory = [{"offset_s": event["attempt_offset_s"], "reason": event.get("reason", event["event"]), **event["load"]}
                  for event in scoped if "load" in event and "attempt_offset_s" in event]
    reached = any(point["target_reached"] for point in trajectory)
    return {"attempt_id": attempt_id, "config": config, "window": {"start_offset_s": start, "cutoff_offset_s": cutoff,
            "duration_s": duration, "boundary": "[start,end)", "cutoff_reason": reason}, "selected_request_ids": selected,
            "selection_counts": counts, "metrics": metrics, "load_trajectory": trajectory, "target_active": target,
            "target_reached": reached, "load_scope": "conversations" if mode == "conversation" else "tasks_without_tool_saturation_claim",
            "definition": "complete_successful_requests_with_start_and_end_inside_window",
            "limitation": "Cross-boundary requests are excluded. This creates selection bias and does not measure exact token production."}


def _null_empty_samples(metrics):
    for group, count in metrics["sample_counts"].items():
        if count == 0:
            metrics[group] = {key: None for key in ("mean", "p50", "p90", "p99")}
    metrics["scope"] = "complete_request_measurement_windows"


def combine_measurements(result, events):
    """Combine selected numerators and effective durations across attempts."""
    from .agent_metrics import aggregate
    measurements = [attempt["measurement"] for attempt in result.get("attempts", []) if attempt.get("measurement")]
    if not measurements:
        return
    selected = {span for item in measurements for span in item["selected_request_ids"]}
    duration = sum(item["window"]["duration_s"] for item in measurements)
    metrics = aggregate([event for event in events if event.get("span_id") in selected and event["event"] in {"llm_start", "llm_end"}], duration)
    _null_empty_samples(metrics)
    keys = measurements[0]["selection_counts"]
    result["measurement"] = {"attempts": measurements, "metrics": metrics,
                             "selection_counts": {key: sum(item["selection_counts"].get(key, 0) for item in measurements) for key in keys},
                             "aggregation": "sum_selected_tokens_divided_by_sum_effective_window_seconds",
                             "definition": measurements[0]["definition"], "limitation": measurements[0]["limitation"]}


class ThreadStopController:
    """Separate dispatch stop from cooperative cancellation of active work."""
    def __init__(self, limits, started, cancel, stop):
        import threading
        self.limits, self.started, self.cancel, self.stop = limits, started, cancel, stop
        self.dispatch = threading.Event()
        self.lock = threading.Lock()
        self.stopped_at = None
        self.measurement = None
        self.pending_signal = self.handled_signal = None
        self.stop.update(policy=limits.stop_policy, grace_sec=limits.stop_grace_sec)

    def request(self, reason, signum=None, *, at=None):
        import time
        at = time.monotonic() if at is None else at
        deadlines = [(at, reason)]
        if self.limits.duration_sec is not None:
            deadlines.append((self.started + self.limits.duration_sec, "duration"))
        if self.measurement is not None and self.measurement.deadline is not None:
            deadlines.append((self.measurement.deadline, "measurement_duration"))
        at, reason = min(deadlines)
        with self.lock:
            changed = False
            if self.stopped_at is None:
                changed = True
                self.stopped_at = at
                self.stop.update(reason=reason, requested_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                                 requested_offset_s=max(0, at - self.started))
            elif at < self.stopped_at:
                changed = True
                self.stopped_at = at
                self.stop.update(reason=reason, requested_offset_s=max(0, at - self.started))
            if signum is not None and self.stop.get("signal") is None:
                changed = True
                self.stop["signal"] = signum
            if changed and self.measurement is not None:
                self.measurement.record_stop(self.stop["reason"], self.stop["requested_offset_s"], self.stop.get("signal"))
            self.dispatch.set()
            if self.limits.stop_policy == "cancel":
                self.cancel.set()

    def check(self):
        import time
        signum = self.pending_signal
        if signum is not None and signum != self.handled_signal:
            self.handled_signal = signum
            self.request("signal", signum)
        now = time.monotonic()
        deadlines = []
        if self.limits.duration_sec is not None:
            deadlines.append((self.started + self.limits.duration_sec, "duration"))
        if self.measurement is not None and self.measurement.deadline is not None:
            deadlines.append((self.measurement.deadline, "measurement_duration"))
        if deadlines:
            deadline, reason = min(deadlines)
            if now >= deadline:
                self.request(reason, at=deadline)
        if self.stopped_at is not None and now >= self.stopped_at + self.limits.stop_grace_sec:
            self.cancel.set()


def recover_measurements(result, events, mode):
    """Recover missing windows from their own configuration and persisted offsets."""
    for attempt in result.get("attempts", []):
        config = attempt.get("measurement_config")
        if not config or not config["enabled"] or attempt.get("measurement"):
            continue
        matching = [event for event in events if event.get("run_attempt_id") == attempt["attempt_id"]]
        offsets = [event["attempt_offset_s"] for event in matching if "attempt_offset_s" in event]
        if not offsets:
            continue
        elapsed = max(offsets) if attempt.get("duration_estimated") else attempt["duration_s"]
        stop = attempt.get("stop") or {}
        cutoffs = [(stop["requested_offset_s"], stop.get("reason") or "stop")] if stop.get("requested_offset_s") is not None else []
        cutoffs.extend((event["requested_offset_s"], event.get("reason") or "stop")
                       for event in matching if event.get("event") == "run_stop" and event.get("requested_offset_s") is not None)
        if config.get("stop_on_load_drop"):
            active_key = "active_conversations" if mode == "conversation" else "active_tasks"
            cutoffs.extend((event["attempt_offset_s"], "load_drop") for event in matching
                           if event.get("load", {}).get("target_reached") and event["load"].get("pending") == 0
                           and active_key in event["load"] and event["load"][active_key] < attempt.get("measurement_target", 1)
                           and "attempt_offset_s" in event)
        stop_offset, stop_reason = min(cutoffs) if cutoffs else (None, None)
        measured = measure_attempt(config, attempt["attempt_id"], events, elapsed, stop_offset,
                                   stop_reason, attempt.get("measurement_target", 1), mode)
        estimated = bool(attempt.get("duration_estimated"))
        measured["window"]["duration_estimated"] = estimated
        measured["window"]["stop_boundary_evidence"] = "persisted_stop_or_load_drop" if cutoffs else "missing"
        if estimated and not cutoffs and measured["window"]["cutoff_reason"] == "execution_end":
            measured["window"]["cutoff_reason"] = "last_persisted_offset_unknown_stop"
        attempt["measurement"] = measured
    combine_measurements(result, events)


def sync_measurement_trace(result, cfg, client, snapshot):
    """Send the same complete-request window metrics on a separate trace."""
    measurement = result.get("measurement")
    if not measurement:
        return
    from .agent_metrics import summary_scores
    from .agent_platform import platform_id
    from . import metric_names
    metrics = measurement["metrics"]
    completion_label = metric_names.summary_score("completed_ratio")[0]
    scores = [score for score in summary_scores(metrics) if score["name"] != completion_label and score["value"] is not None]
    attempts = result.get("attempts") or []
    start = attempts[0]["start_time"] if attempts else result["start_time"]
    end = result.get("end_time") or attempts[-1]["end_time"]
    client.trace(id=platform_id(f"{result['run_id']}:measurement:{snapshot}", start, "trace"),
                 project_name=cfg.benchmark_platform.project_name, name="测量窗口汇总",
                 start_time=dt.datetime.fromisoformat(start), end_time=dt.datetime.fromisoformat(end),
                 input={"attempt_measurement_configs": [item.get("measurement_config") for item in attempts]},
                 output=metrics, metadata={"run_id": result["run_id"], "snapshot_id": snapshot, "measurement": measurement,
                     "throughput_scope": "complete_request_measurement_windows", "completion_ratio_scope": "see_selection_counts"},
                 tags=cfg.benchmark_platform.tags + ["measurement"], feedback_scores=scores)
