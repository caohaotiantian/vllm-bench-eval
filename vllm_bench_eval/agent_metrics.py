"""Separate model request performance from complete agent workload throughput."""
from __future__ import annotations

import copy
import datetime as dt
import math

from . import metric_names

METRIC_DEFINITION_VERSION = 2
METRIC_DEFINITIONS = {
    "metric_definition_version": METRIC_DEFINITION_VERSION,
    "tpot_ms": "(e2e_ms - ttft_ms) / (completion_tokens - 1)",
    "chunk_tpot_ms": "decode_ms / (completion_tokens - 1); client chunk estimate",
    "request_tps": "successful request tokens / request e2e seconds",
    "workload_tps": "successful request tokens / complete active agent windows",
    "itl": "client chunk estimate; not exact token intervals",
}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def normalize_timing(timing: dict | None, usage: dict | None) -> dict:
    """Return current metric definitions without changing the stored timing."""
    out = copy.deepcopy(timing or {})
    tokens = (usage or {}).get("completion_tokens")
    tokens_valid = isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
    ttft, e2e, decode = out.get("ttft_ms"), out.get("e2e_ms"), out.get("decode_ms")
    if out.get("metric_definition_version", 1) < METRIC_DEFINITION_VERSION and "chunk_tpot_ms" not in out:
        out["chunk_tpot_ms"] = out.get("tpot_ms")
    reason = None
    if out.get("stream") is False:
        reason = "nonstream"
    elif not _number(ttft):
        reason = out.get("ttft_null_reason") or "no_output"
    elif not tokens_valid:
        reason = "missing_completion_tokens"
    elif tokens < 2:
        reason = "fewer_than_two_tokens"
    elif out.get("tpot_null_reason") == "incomplete_request":
        reason = "incomplete_request"
    elif not _number(e2e) or e2e < ttft:
        reason = "missing_or_invalid_e2e"
    out["tpot_ms"] = (e2e - ttft) / (tokens - 1) if reason is None else None
    out["tpot_null_reason"] = reason
    chunk_valid = _number(ttft) and tokens_valid and tokens > 1 and out.get("stream") is not False and out.get("tpot_null_reason") != "incomplete_request"
    if chunk_valid and _number(decode):
        out["chunk_tpot_ms"] = decode / (tokens - 1)
    elif not chunk_valid or not _number(out.get("chunk_tpot_ms")):
        out["chunk_tpot_ms"] = None
    out["metric_definition_version"] = METRIC_DEFINITION_VERSION
    out["tpot_definition"] = METRIC_DEFINITIONS["tpot_ms"]
    out["chunk_tpot_definition"] = METRIC_DEFINITIONS["chunk_tpot_ms"]
    return out


def normalize_events(events: list[dict]) -> list[dict]:
    out = copy.deepcopy(events)
    for event in out:
        if event.get("event") == "llm_end":
            event["timing"] = normalize_timing(event.get("timing"), event.get("usage"))
    return out


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    result = {"mean": sum(values) / len(values)}
    for percent in (50, 90, 99):
        index = (len(ordered) - 1) * percent / 100
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        result[f"p{percent}"] = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return result


def _tool_metrics(events: list[dict]) -> dict:
    starts = {event["span_id"]: event for event in events if event["event"] in ("tool_start", "verification_start")}
    ends = {event["span_id"]: event for event in events if event["event"] in ("tool_end", "verification_end")}
    out = {}
    for prefix, kind in (("", "tool_start"), ("verification_", "verification_start")):
        selected = [event for event in starts.values() if event["event"] == kind]
        failures, incomplete = 0, 0
        for start in selected:
            end = ends.get(start["span_id"])
            if end is None:
                incomplete += 1
                continue
            output = end.get("output") or {}
            failures += int(bool(end.get("error") or output.get("error") or output.get("timed_out") or output.get("exit_code", 0) != 0))
        out.update({prefix + "calls": len(selected), prefix + "failed": failures, prefix + "incomplete": incomplete})
    return out


def aggregate(events: list[dict], duration_s: float) -> dict:
    events = normalize_events(events)
    starts = [event for event in events if event["event"] == "llm_start"]
    ends = [event for event in events if event["event"] == "llm_end" and "output" in event and not event.get("error")]
    def token_total(key: str):
        values = [(event.get("usage") or {}).get(key) for event in ends]
        return sum(values) if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values) else None
    known_consumed = [(event.get("usage") or {}).get("total_tokens") for event in events if event["event"] == "llm_end"]
    input_tokens = token_total("prompt_tokens")
    output_tokens = token_total("completion_tokens")
    total_tokens = token_total("total_tokens")
    metrics = {"metric_definition_version": METRIC_DEFINITION_VERSION, "scope": "agent_workload_active_windows",
               "counts": {"num_prompts": len(starts), "completed": len(ends), "failed": len(starts) - len(ends),
                          "total_input_tokens": input_tokens, "total_output_tokens": output_tokens,
                          "total_tokens": total_tokens, "known_consumed_tokens": sum(value for value in known_consumed if isinstance(value, int) and not isinstance(value, bool) and value >= 0),
                          "duration_s": duration_s},
               "throughput": {"request_rps": len(ends) / duration_s if duration_s > 0 else None,
                              "output_tps": output_tokens / duration_s if duration_s > 0 and output_tokens is not None else None,
                              "total_tps": total_tokens / duration_s if duration_s > 0 and total_tokens is not None else None},
               "tools": _tool_metrics(events), "sample_counts": {}}
    for key, group in (("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms"), ("e2e_ms", "e2el_ms")):
        values = [event["timing"][key] for event in ends if _number((event.get("timing") or {}).get(key))]
        metrics[group] = stats(values)
        metrics["sample_counts"][group] = len(values)
    for token_key, group in (("completion_tokens", "request_output_tps"), ("total_tokens", "request_total_tps")):
        values = []
        for event in ends:
            e2e = event["timing"].get("e2e_ms")
            tokens = (event.get("usage") or {}).get(token_key)
            if _number(e2e) and e2e > 0 and isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                values.append(tokens / (e2e / 1000))
        metrics[group] = stats(values)
        metrics["sample_counts"][group] = len(values)
    return metrics


def call_scores(timing: dict, usage: dict, success: bool) -> list[dict]:
    timing = normalize_timing(timing, usage)
    scores = [{"name": metric_names.request_score("success")[0], "value": int(success)}]
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        if _number(timing.get(key)):
            scores.append({"name": metric_names.request_score(key)[0], "value": timing[key]})
    seconds = timing["e2e_ms"] / 1000 if _number(timing.get("e2e_ms")) else 0
    if seconds > 0:
        for token_key, label in (("completion_tokens", metric_names.request_score("output_tokens_per_s")[0]),
                                 ("total_tokens", metric_names.summary_score("total_token_throughput_tps")[0])):
            if isinstance(usage.get(token_key), int) and not isinstance(usage[token_key], bool) and usage[token_key] >= 0:
                scores.append({"name": label, "value": usage[token_key] / seconds})
    return scores


def _performance_scores(metrics: dict) -> list[dict]:
    scores = []
    for group, metric in (("ttft_ms", "ttft"), ("tpot_ms", "tpot"), ("e2el_ms", "e2el")):
        for stat, value in metrics.get(group, {}).items():
            scores.append({"name": metric_names.percentile_score_name(stat, metric), "value": value})
    for group in metric_names.AGENT_TPS_LABELS:
        for stat, value in metrics.get(group, {}).items():
            scores.append({"name": metric_names.agent_tps_score_name(stat, group), "value": value})
    counts = metrics["counts"]
    if counts["num_prompts"]:
        scores.append({"name": metric_names.summary_score("completed_ratio")[0], "value": counts["completed"] / counts["num_prompts"]})
    return scores


def task_scores(metrics: dict) -> list[dict]:
    scores = _performance_scores(metrics)
    for field, label in metric_names.AGENT_WORKLOAD_SCORE_NAMES.items():
        value = metrics["throughput"].get(field)
        if value is not None:
            scores.append({"name": label, "value": value})
    scores.append({"name": metric_names.AGENT_TOOL_SCORE_NAMES["failed"], "value": metrics["tools"]["failed"]})
    return scores


def summary_scores(metrics: dict) -> list[dict]:
    scores = _performance_scores(metrics)
    for key, field in (("output_throughput_tps", "output_tps"), ("total_token_throughput_tps", "total_tps"),
                       ("request_throughput_rps", "request_rps")):
        value = metrics["throughput"].get(field)
        if value is not None:
            scores.append({"name": metric_names.summary_score(key)[0], "value": value})
    return scores


def normalize_result(result: dict, events: list[dict] | None = None) -> dict:
    """Return normalized telemetry while preserving persisted IDs and source data."""
    out = copy.deepcopy(result)
    for task in out.get("tasks", []):
        for call in task.get("llm_calls", []):
            call.update(normalize_timing(call, call.get("usage")))
    if events is not None:
        windows = out.get("attempts") or []
        known_windows = [window.get("duration_s") for window in windows]
        if known_windows and all(_number(value) for value in known_windows):
            duration = sum(known_windows)
        else:
            duration = ((out.get("metrics") or {}).get("counts") or {}).get("duration_s")
            if not _number(duration):
                starts = [dt.datetime.fromisoformat(task["start_time"]) for task in out.get("tasks", []) if task.get("start_time")]
                ends = [dt.datetime.fromisoformat(task["end_time"]) for task in out.get("tasks", []) if task.get("end_time")]
                duration = max(0, (max(ends) - min(starts)).total_seconds()) if starts and ends else 0
        out["metrics"] = aggregate(events, duration)
        out["attempt_metrics"] = {window["attempt_id"]: aggregate([event for event in events if event.get("run_attempt_id") == window["attempt_id"]], window.get("duration_s", 0)) for window in windows if _number(window.get("duration_s"))}
        for task in out.get("tasks", []):
            if "duration_s" in task:
                task["metrics"] = aggregate([event for event in events if event["task_id"] == task["task"]["task_id"]], task["duration_s"])
    terminal = [task for task in out.get("tasks", []) if task.get("status") not in {"pending", "interrupted"} and task.get("success") is not None]
    passed = sum(bool(task["success"]) for task in terminal)
    out["task_quality"] = {"terminal_tasks": len(terminal), "passed_tasks": passed,
                           "task_success_pct": passed / len(terminal) * 100 if terminal else None}
    out["metric_definition_version"] = METRIC_DEFINITION_VERSION
    return out
