"""Measure sequential ShareGPT conversations with accumulated message history."""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, contextmanager
import copy
import datetime as dt
import hashlib
import json
import signal
import threading
from pathlib import Path
import time
import uuid

import httpx

from . import metric_names
from .agent_metrics import METRIC_DEFINITIONS, aggregate, call_scores, normalize_events, summary_scores, task_scores
from .agent_platform import checked_delivery, checked_flush, platform_id
from .agent_transport import async_chat_request
from .coding_agent import Journal, _atomic_json, _events, _known_tokens, _now
from .config import AppConfig, ConversationConfig
from .platform_sync import build_platform_client
from .lifecycle import LifecycleError, read_events, run_lock, settle_attempts, snapshot_identity, MeasurementRecorder, measure_attempt, combine_measurements, recover_measurements, sync_measurement_trace


class ConversationError(ValueError):
    """The conversation input or response violates the supported format."""


def load_conversations(path: Path) -> list[dict]:
    """Validate ShareGPT order and remove blank source turns with recorded provenance."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ConversationError("ShareGPT dataset must be a nonempty JSON array")
    conversations, ids = [], set()
    roles = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ConversationError(f"conversation {index}: expected an object")
        source_id = record.get("id", f"conversation-{index}")
        if isinstance(source_id, bool) or not isinstance(source_id, (str, int)) or not str(source_id).strip():
            raise ConversationError(f"conversation {index}: id must be a nonempty string or integer")
        source_id = str(source_id)
        if source_id in ids:
            raise ConversationError(f"duplicate conversation id: {source_id}")
        ids.add(source_id)
        raw = record.get("conversations")
        if not isinstance(raw, list):
            raise ConversationError(f"conversation {source_id}: conversations must be an array")
        messages = []
        expected = "user"
        for position, message in enumerate(raw):
            if not isinstance(message, dict):
                raise ConversationError(f"conversation {source_id}, message {position}: expected an object")
            role = roles.get(message.get("from")) if isinstance(message.get("from"), str) else None
            content = message.get("value")
            if role is None or not isinstance(content, str):
                raise ConversationError(f"conversation {source_id}, message {position}: unsupported role or non-string text")
            if set(message) - {"from", "value"}:
                raise ConversationError(f"conversation {source_id}, message {position}: unsupported message fields")
            if role == "system":
                if position != 0:
                    raise ConversationError(f"conversation {source_id}: system message must be first")
            elif role != expected:
                raise ConversationError(f"conversation {source_id}, message {position}: expected {expected}")
            else:
                expected = "assistant" if role == "user" else "user"
            messages.append({"role": role, "content": content})
        kept, original_indices, cleanup = [], [], []
        position = 0
        if messages and messages[0]["role"] == "system":
            if messages[0]["content"].strip():
                kept.append(messages[0])
                original_indices.append(0)
            else:
                cleanup.append({"message_indices": [0], "reason": "blank_system"})
            position = 1
        while position < len(messages):
            user = messages[position]
            assistant_index = position + 1 if position + 1 < len(messages) else None
            assistant = messages[assistant_index] if assistant_index is not None else None
            pair_indices = [position] + ([assistant_index] if assistant is not None else [])
            if not user["content"].strip():
                cleanup.append({"message_indices": pair_indices, "reason": "blank_user_pair"})
            elif assistant is not None and not assistant["content"].strip():
                if assistant_index == len(messages) - 1:
                    kept.append(user)
                    original_indices.append(position)
                    cleanup.append({"message_indices": [assistant_index], "reason": "blank_final_assistant"})
                else:
                    cleanup.append({"message_indices": pair_indices, "reason": "blank_assistant_pair"})
            else:
                kept.append(user)
                original_indices.append(position)
                if assistant is not None:
                    kept.append(assistant)
                    original_indices.append(assistant_index)
            position += 2 if assistant is not None else 1
        conversations.append({"conversation_id": source_id, "messages": kept,
                              "source": {"record_index": index, "original_id": record.get("id"),
                                         "id_generated": "id" not in record, "original_messages": copy.deepcopy(raw),
                                         "message_indices": original_indices,
                                         "raw_user_turns": sum(message["role"] == "user" for message in messages),
                                         "cleanup": cleanup,
                                         "removed_messages": sum(len(item["message_indices"]) for item in cleanup)}})
    return conversations


async def _execute(source: dict, cfg: AppConfig, client: httpx.AsyncClient, journal: Journal, run_id: str, stop, recovered=None) -> dict:
    limits = cfg.conversation
    started, start_time = time.monotonic(), _now()
    identity = source["conversation_id"]
    trace_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbe-conversation:{run_id}:{identity}"))
    result = {"conversation": source, "trace_id": trace_id, "start_time": start_time,
              "status": "completed", "execution_ok": True, "turns": [], "planned_turns": sum(m["role"] == "user" for m in source["messages"])}
    if result["planned_turns"] == 0:
        result.update(status="skipped", execution_ok=None, end_time=_now(),
                      duration_s=time.monotonic() - started, known_consumed_tokens=0,
                      skip_reason="no_usable_user_turns", metrics=aggregate([], 0))
        return result
    prior_elapsed = recovered.get("duration_s", 0) if recovered else 0
    if recovered:
        result.update(trace_id=recovered["trace_id"], start_time=recovered["start_time"],
                      turns=copy.deepcopy(recovered["turns"]))
    consumed = recovered.get("known_consumed_tokens", 0) if recovered else 0
    confirmed = len(result["turns"])
    history, local_events = [], []
    def emit(event):
        event = {"task_id": identity, "conversation_id": identity,
                 "elapsed_s": prior_elapsed + time.monotonic() - started, **event}
        journal.emit(event)
        local_events.append(event)
    emit({"event": "conversation_start", "prior_elapsed_s": prior_elapsed})
    source_turn = 0
    try:
        if result["planned_turns"] > limits.max_turns:
            raise ConversationError(f"conversation has {result['planned_turns']} turns, above max_turns={limits.max_turns}")
        for message_index, message in enumerate(source["messages"]):
            if message["role"] == "system":
                history.append(copy.deepcopy(message))
                continue
            if message["role"] == "assistant":
                if limits.history_policy == "dataset":
                    history[-1] = copy.deepcopy(message)
                continue
            source_turn += 1
            if source_turn <= confirmed:
                history.append(copy.deepcopy(message))
                history.append({"role": "assistant", "content": result["turns"][source_turn - 1]["output"]["content"]})
                continue
            stop.check_duration()
            if stop.requested:
                raise asyncio.CancelledError()
            history.append(copy.deepcopy(message))
            turn = len(result["turns"]) + 1
            context = {"message_count": len(history), "context_chars": sum(len(m["content"]) for m in history),
                       "history_policy": limits.history_policy, "prompt_tokens": None,
                       "source_message_index": source["source"]["message_indices"][message_index]}
            if context["context_chars"] > limits.max_context_chars:
                raise ConversationError(f"turn {turn}: context exceeds max_context_chars={limits.max_context_chars}")
            remaining = limits.conversation_timeout_sec - prior_elapsed - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("conversation timeout exceeded")
            if consumed >= limits.token_budget:
                raise ConversationError("conversation token_budget exhausted")
            payload = {**limits.extra_body, "model": cfg.server.model, "messages": copy.deepcopy(history),
                       "stream": limits.stream, "max_tokens": min(limits.max_tokens_per_turn, limits.token_budget - consumed), "temperature": 0}
            if limits.stream:
                payload["stream_options"] = {"include_usage": True}
            span_id = str(uuid.uuid4())
            logical_turn_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{trace_id}:{turn}"))
            emit({"event": "llm_start", "span_id": span_id, "logical_turn_id": logical_turn_id, "turn": turn, "input": payload, "context": context})
            request_started = time.monotonic()
            timeout = min(limits.request_timeout_sec, remaining)
            received_usage, received_timing, response_summary = {}, None, None
            try:
                body, timing = await async_chat_request(client, cfg.server.base_url + limits.endpoint, payload, timeout)
                received_timing = timing
                raw_usage = body.get("usage")
                if isinstance(raw_usage, dict):
                    received_usage = {key: value for key, value in raw_usage.items()
                                      if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                                      and isinstance(value, int) and not isinstance(value, bool) and value >= 0}
                    context["prompt_tokens"] = received_usage.get("prompt_tokens")
                if time.monotonic() - request_started > timeout:
                    raise TimeoutError("request or conversation timeout exceeded")
                choices = body.get("choices")
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise ConversationError("response requires exactly one choice")
                choice = choices[0]
                output = choice.get("message")
                response_summary = {"finish_reason": choice.get("finish_reason")}
                if not isinstance(output, dict):
                    raise ConversationError("response message must be an object")
                content = output.get("content")
                response_summary.update(role=output.get("role"), content_type=type(content).__name__,
                                        content_chars=len(content) if isinstance(content, str) else None,
                                        reasoning_present=bool(output.get("reasoning_content") or output.get("reasoning") or output.get("_reasoning_present")),
                                        tool_calls_present=bool(output.get("tool_calls") or output.get("function_call")))
                detail = json.dumps(response_summary, ensure_ascii=False)
                if output.get("role") != "assistant":
                    raise ConversationError(f"response requires assistant role: {detail}")
                if response_summary["tool_calls_present"]:
                    raise ConversationError(f"response contains unsupported tool calls: {detail}")
                if not isinstance(content, str) or not content.strip():
                    raise ConversationError(f"response requires nonblank assistant text: {detail}")
                if choice.get("finish_reason") != "stop":
                    raise ConversationError(f"response did not finish normally: {detail}")
                usage = body.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ConversationError("response usage must be an object")
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                        raise ConversationError(f"response usage.{key} must be a nonnegative integer")
                context["prompt_tokens"] = usage.get("prompt_tokens")
                emit({"event": "llm_end", "span_id": span_id, "turn": turn, "output": output,
                      "usage": usage, "timing": timing, "finish_reason": choice.get("finish_reason"), "context": context})
                result["turns"].append({"turn": turn, "request_id": span_id, "output": output, "usage": usage, "timing": timing, "context": context})
                history.append({"role": "assistant", "content": output["content"]})
                tokens = _known_tokens(usage)
                if tokens is None:
                    raise ConversationError("response total_tokens is missing, cannot enforce token_budget")
                consumed += tokens
                if consumed > limits.token_budget:
                    raise ConversationError("conversation token_budget exceeded")
            except (Exception, asyncio.CancelledError) as exc:
                # Completed requests remain successful when a workload limit stops later turns.
                if not local_events or local_events[-1]["event"] != "llm_end":
                    failed_usage = received_usage or getattr(exc, "agent_usage", {})
                    known_tokens = _known_tokens(failed_usage)
                    if known_tokens is not None:
                        consumed += known_tokens
                    emit({"event": "llm_end", "span_id": span_id, "turn": turn, "error": str(exc) or "request cancelled",
                          **({"status": "cancelled"} if isinstance(exc, asyncio.CancelledError) else {}),
                          "timing": received_timing or getattr(exc, "agent_timing", {"stream": limits.stream, "e2e_ms": (time.monotonic() - request_started) * 1000,
                                    "tpot_null_reason": "incomplete_request"}), "usage": failed_usage,
                          "response_summary": response_summary, "context": context})
                raise
    except asyncio.CancelledError:
        result.update(status="interrupted", execution_ok=None)
    except Exception as exc:
        result.update(status="failed", execution_ok=False, error=f"{type(exc).__name__}: {exc}")
    result.update(end_time=_now(), duration_s=prior_elapsed + time.monotonic() - started, known_consumed_tokens=consumed)
    emit({"event": "conversation_end", "status": result["status"], "error": result.get("error")})
    result["metrics"] = aggregate(local_events, result["duration_s"])
    return result


class _StopController:
    def __init__(self, limits):
        self.limits = limits
        self.started = time.monotonic()
        self.measurement = None
        self.requested = False
        self.reason = self.signum = self.requested_at = self.monotonic = None
        self.pending_signal = self.handled_signal = None

    def request(self, reason, signum=None, *, at=None):
        at = time.monotonic() if at is None else at
        deadlines = [(at, reason)]
        if self.limits.duration_sec is not None:
            deadlines.append((self.started + self.limits.duration_sec, "duration"))
        if self.measurement is not None and self.measurement.deadline is not None:
            deadlines.append((self.measurement.deadline, "measurement_duration"))
        at, reason = min(deadlines)
        changed = False
        if not self.requested:
            changed = True
            self.reason, self.signum = reason, signum
            self.requested_at, self.monotonic = _now(), at
            self.requested = True
        elif at < self.monotonic:
            changed = True
            self.reason, self.monotonic = reason, at
        if signum is not None and self.signum is None:
            changed = True
            self.signum = signum
        if changed and self.measurement is not None:
            self.measurement.record_stop(self.reason, max(0, self.monotonic - self.started), self.signum)

    def check_duration(self):
        signum = self.pending_signal
        if signum is not None and signum != self.handled_signal:
            self.handled_signal = signum
            self.request("signal", signum)
        deadlines = []
        if self.limits.duration_sec is not None:
            deadlines.append((self.started + self.limits.duration_sec, "duration"))
        if self.measurement is not None and self.measurement.deadline is not None:
            deadlines.append((self.measurement.deadline, "measurement_duration"))
        if deadlines:
            deadline, reason = min(deadlines)
            if time.monotonic() >= deadline:
                self.request(reason, at=deadline)

    def signal(self, signum, frame):
        self.pending_signal = signum


@contextmanager
def _capture_signals(stop):
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop.signal)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _unstarted(source, run_id):
    planned = sum(m["role"] == "user" for m in source["messages"])
    now = _now()
    return {"conversation": source,
            "trace_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbe-conversation:{run_id}:{source['conversation_id']}")),
            "start_time": now, "end_time": now, "duration_s": 0, "planned_turns": planned,
            "turns": [], "known_consumed_tokens": 0, "metrics": aggregate([], 0), "execution_ok": None,
            "status": "not_started" if planned else "skipped",
            **({"skip_reason": "no_usable_user_turns"} if not planned else {})}


async def _run_workload(cfg, conversations, client, journal, run_id, result, path, stop, measurement):
    async with AsyncExitStack() as stack:
        if client is None:
            headers = {"Authorization": f"Bearer {cfg.server.api_key}"} if cfg.server.api_key else {}
            client = await stack.enter_async_context(httpx.AsyncClient(headers=headers))
        snapshots = result["conversations"]
        result["conversations"] = snapshots
        _persist_result(path, result)
        next_index = 0
        eligible = sum(item["status"] not in {"completed", "skipped"} and len(item["turns"]) < item["planned_turns"] for item in snapshots)
        measurement.change_load(pending=eligible, reason="queue_initialized")
        async def worker():
            nonlocal next_index
            while next_index < len(conversations):
                stop.check_duration()
                if stop.requested:
                    break
                index = next_index
                next_index += 1
                if snapshots[index]["status"] in {"completed", "skipped"} or len(snapshots[index]["turns"]) == snapshots[index]["planned_turns"]:
                    continue
                measurement.change_load(pending=-1, active=1, reason="conversation_started")
                try:
                    snapshots[index] = await _execute(conversations[index], cfg, client, journal, run_id, stop,
                                                      recovered=snapshots[index] if snapshots[index]["status"] != "not_started" else None)
                finally:
                    measurement.change_load(active=-1, reason="conversation_finished")
                _persist_result(path, result)
        workers = [asyncio.create_task(worker()) for _ in range(min(len(conversations), cfg.conversation.max_concurrency))]
        try:
            while any(not task.done() for task in workers):
                now = time.monotonic()
                stop.check_duration()
                if stop.requested and (cfg.conversation.stop_policy == "cancel" or now - stop.monotonic >= cfg.conversation.stop_grace_sec):
                    for task in workers:
                        if not task.done():
                            task.cancel()
                    break
                await asyncio.sleep(.01)
            outcomes = await asyncio.gather(*workers, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not (stop.requested and isinstance(outcome, asyncio.CancelledError)):
                    raise outcome
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)


def run_conversations(cfg: AppConfig) -> Path:
    """Run conversations through a private event loop and return a saved result."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_run_conversations(cfg))
    raise ConversationError("use async_run_conversations from an active event loop")


def _persist_result(path, result):
    _atomic_json(path, {key: value for key, value in result.items() if key != "_result_path"})


def _recover_snapshot(result, events):
    if not result.get("conversations"):
        raise ConversationError("recovery requires persisted conversation sources")
    sources = {item["conversation"]["conversation_id"]: item for item in result["conversations"]}
    if len(sources) != len(result["conversations"]):
        raise ConversationError("recovery contains duplicate conversation identities")
    starts, ends, offsets = {}, {}, {}
    for index, event in enumerate(events):
        if event.get("run_id") != result["run_id"]:
            raise ConversationError("event run identity differs from the result")
        if result.get("lifecycle_version", 0) >= 2 and event.get("run_attempt_id") not in {window["attempt_id"] for window in result.get("attempts", [])}:
            raise ConversationError("event attempt identity is missing from the result")
        if event.get("event") in {"load", "run_stop"} and event.get("conversation_id") is None:
            if event.get("task_id") is not None:
                raise ConversationError("global lifecycle event contains an unknown task identity")
            continue
        if event.get("conversation_id") not in sources:
            raise ConversationError("event conversation identity is missing")
        if event.get("event") not in {"llm_start", "llm_end"}:
            continue
        span = event.get("span_id")
        if not isinstance(span, str) or not span:
            raise ConversationError("request identity is missing")
        target = starts if event["event"] == "llm_start" else ends
        if span in target:
            raise ConversationError("duplicate physical request event")
        target[span] = event
        offsets[(event["event"], span)] = index
    for span, end in ends.items():
        start = starts.get(span)
        if start is None or any(end.get(key) != start.get(key) for key in ("conversation_id", "turn", "run_attempt_id")):
            raise ConversationError("request end identity differs from its start")
        if offsets[("llm_end", span)] <= offsets[("llm_start", span)]:
            raise ConversationError("request end precedes its start")
    recovered_items = []
    for identity, item in sources.items():
        source = item["conversation"]
        users = [(index, message) for index, message in enumerate(source["messages"]) if message["role"] == "user"]
        matching = [event for event in events if event.get("conversation_id") == identity]
        successful = {start["turn"]: (span, ends[span]) for span, start in starts.items()
                      if start["conversation_id"] == identity and span in ends and "output" in ends[span] and not ends[span].get("error")}
        successes = [span for span, start in starts.items() if start["conversation_id"] == identity
                     and span in ends and "output" in ends[span] and not ends[span].get("error")]
        if len(successes) != len(successful) or sorted(successful) != list(range(1, len(successful) + 1)):
            raise ConversationError("confirmed conversation turns are duplicated or incomplete")
        history = []
        if source["messages"] and source["messages"][0]["role"] == "system":
            history.append(copy.deepcopy(source["messages"][0]))
        turns = []
        for turn, (message_index, message) in enumerate(users, 1):
            history.append(copy.deepcopy(message))
            requests = [(span, start) for span, start in starts.items()
                        if start["conversation_id"] == identity and start.get("turn") == turn]
            for span, start in requests:
                payload = start.get("input") or {}
                context = start.get("context") or {}
                if (turn > len(successful) + 1 or payload.get("messages") != history
                    or payload.get("model") != result["model"]
                    or context.get("source_message_index") != source["source"]["message_indices"][message_index]):
                    raise ConversationError("request source or history evidence differs")
                if turn > 1:
                    previous_span = successful[turn - 1][0]
                    if offsets[("llm_end", previous_span)] >= offsets[("llm_start", span)]:
                        raise ConversationError("request starts before its previous turn was confirmed")
            if turn not in successful:
                if any(start["conversation_id"] == identity and start.get("turn", 0) > turn for start in starts.values()):
                    raise ConversationError("request starts after an unconfirmed turn")
                break
            span, end = successful[turn]
            output = end["output"]
            if end.get("finish_reason") != "stop":
                raise ConversationError("confirmed response did not finish normally")
            if not isinstance(output, dict) or output.get("role") != "assistant" or not isinstance(output.get("content"), str) or not output["content"].strip():
                raise ConversationError("confirmed assistant output is invalid")
            turns.append({"turn": turn, "request_id": span, "output": copy.deepcopy(output),
                          "usage": end.get("usage") or {}, "timing": end.get("timing") or {}, "context": end.get("context", starts[span]["context"])})
            if result["run_config"]["history_policy"] == "generated":
                history.append({"role": "assistant", "content": output["content"]})
            elif message_index + 1 < len(source["messages"]):
                history.append(copy.deepcopy(source["messages"][message_index + 1]))
        for start in starts.values():
            if start["conversation_id"] == identity and (isinstance(start.get("turn"), bool) or not isinstance(start.get("turn"), int) or not 1 <= start["turn"] <= len(users)):
                raise ConversationError("request turn is outside the source")
        consumed = sum(_known_tokens(event.get("usage")) or 0 for span, event in ends.items()
                       if starts[span]["conversation_id"] == identity)
        unknown = any(_known_tokens(turn["usage"]) is None for turn in turns)
        latest_start = max((index for index, event in enumerate(matching) if event["event"] == "conversation_start"), default=0)
        current_events = matching[latest_start:]
        conversation_ends = [event for event in current_events if event["event"] == "conversation_end"]
        request_ends = [event for event in current_events if event["event"] == "llm_end"]
        current_error = conversation_ends[-1].get("error") if conversation_ends else None
        if conversation_ends:
            failed = conversation_ends[-1].get("status") == "failed"
        elif request_ends:
            last_request = request_ends[-1]
            failed = bool(last_request.get("error") and last_request.get("status") != "cancelled")
            current_error = last_request.get("error") if failed else None
        else:
            failed = not matching and item.get("status") == "failed"
        limits = result["run_config"]
        budget_exceeded = consumed > limits["token_budget"] or (len(turns) < len(users) and consumed >= limits["token_budget"])
        if budget_exceeded:
            current_error = "conversation token_budget exhausted during recorded execution"
        failed = failed or unknown or budget_exceeded
        status = "skipped" if not users else "failed" if failed else "completed" if len(turns) == len(users) else "interrupted" if matching else "not_started"
        recovered = {**item, "turns": turns, "known_consumed_tokens": consumed,
                     "unknown_consumed_tokens": unknown, "planned_turns": len(users), "status": status,
                     "execution_ok": True if status == "completed" else False if status == "failed" else None,
                     "end_time": max((event["timestamp"] for event in matching), default=item.get("end_time", result["start_time"])),
                     "duration_s": max([item.get("duration_s", 0)] + [event.get("elapsed_s", 0) for event in matching])}
        if current_error:
            recovered["error"] = current_error
        if status != "failed":
            recovered.pop("error", None)
        recovered_items.append(recovered)
    # Complete abandoned physical requests only after all recorded evidence passed validation.
    for span, start in starts.items():
        if span not in ends:
            event = {"event": "llm_end", "run_id": result["run_id"], "run_attempt_id": start.get("run_attempt_id"),
                     "task_id": start["conversation_id"], "conversation_id": start["conversation_id"], "span_id": span,
                     "turn": start["turn"], "timestamp": max((item["timestamp"] for item in events if item.get("run_attempt_id") == start.get("run_attempt_id")), default=start["timestamp"]),
                     "status": "cancelled", "error": "request interrupted before recovery", "recovered": True,
                     **({"attempt_offset_s": max(item.get("attempt_offset_s", 0) for item in events if item.get("run_attempt_id") == start.get("run_attempt_id"))}
                        if any("attempt_offset_s" in item for item in events if item.get("run_attempt_id") == start.get("run_attempt_id")) else {}),
                     "usage": {}, "timing": {"stream": result["run_config"]["stream"], "tpot_null_reason": "incomplete_request"}}
            Journal(Path(result["events_path"])).emit(event)
            events.append(event)
    result["conversations"] = recovered_items
    if not result.get("attempts"):
        end = result.get("end_time") or max((event["timestamp"] for event in events), default=result["start_time"])
        result["attempts"] = [{"attempt_id": "legacy", "legacy": True, "start_time": result["start_time"], "end_time": end,
                               "duration_s": result.get("duration_s", max(0, (dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(result["start_time"])).total_seconds())),
                               "duration_estimated": not bool(result.get("end_time"))}]
    settle_attempts(result, events)
    _settle_metrics(result, events)
    recover_measurements(result, events, "conversation")


def recover_conversations(path: Path) -> Path:
    """Settle persisted evidence without any model request."""
    path = path.resolve()
    with run_lock(path):
        result = _read_result(path, require_complete=False)
        events = read_events(Path(result["events_path"]), repair=True)
        _recover_snapshot(result, events)
        result.update(status="recovered", end_time=result["attempts"][-1]["end_time"],
                      stop={"reason": "recovery", "signal": None, "requested_at": None, "policy": "offline"})
        _persist_result(path, result)
    return path


def _resume_identity(cfg):
    excluded = {"measurement", "resume_result_path", "result_dir", "dataset_path", "max_concurrency", "duration_sec", "stop_policy", "stop_grace_sec"}
    return {"model": cfg.server.model, "base_url": cfg.server.base_url,
            "settings": cfg.conversation.model_dump(exclude=excluded)}


def _attempt_events(events, window):
    return [event for event in events if event.get("run_attempt_id") == window["attempt_id"]
            or (window.get("legacy") and not event.get("run_attempt_id"))]


def _settle_metrics(result, events):
    result["duration_s"] = sum(window["duration_s"] for window in result["attempts"])
    result["metrics"] = aggregate(events, result["duration_s"])
    result["attempt_metrics"] = {window["attempt_id"]: aggregate(_attempt_events(events, window), window["duration_s"])
                                 for window in result["attempts"]}
    for conversation in result["conversations"]:
        matching = [event for event in events if event.get("conversation_id") == conversation["conversation"]["conversation_id"]]
        conversation["metrics"] = aggregate(matching, conversation["duration_s"])
    result["execution"] = {
        "planned_conversations": sum(c["status"] != "skipped" for c in result["conversations"]),
        "completed_conversations": sum(c["execution_ok"] is True for c in result["conversations"]),
        "planned_turns": sum(c["planned_turns"] for c in result["conversations"]),
        "completed_turns": result["metrics"]["counts"]["completed"],
        "failed_conversations": sum(c["status"] == "failed" for c in result["conversations"]),
        "interrupted_conversations": sum(c["status"] == "interrupted" for c in result["conversations"]),
        "not_started_conversations": sum(c["status"] == "not_started" for c in result["conversations"]),
        "cancelled_turns": result["metrics"]["counts"].get("cancelled", 0),
        "not_started_turns": sum(c["planned_turns"] for c in result["conversations"]) - len({
            (event.get("conversation_id"), event.get("turn")) for event in events if event.get("event") == "llm_start"})}


async def async_run_conversations(cfg: AppConfig, *, client: httpx.AsyncClient | None = None) -> Path:
    """Run conversations in the caller's loop.

    The caller owns an injected AsyncClient and must close it in this loop.
    """
    if client is not None and not isinstance(client, httpx.AsyncClient):
        raise TypeError("client must be an httpx.AsyncClient")
    cfg.conversation = ConversationConfig.model_validate(cfg.conversation.model_dump())
    if not cfg.server.model.strip():
        raise ConversationError("server.model is required")
    source_path = cfg.path(cfg.conversation.dataset_path).resolve()
    conversations = load_conversations(source_path)
    dataset_conversations = len(conversations)
    if cfg.conversation.num_conversations is not None:
        conversations = conversations[:cfg.conversation.num_conversations]
    if cfg.conversation.resume_result_path:
        path = cfg.path(cfg.conversation.resume_result_path).resolve()
        if not path.is_file():
            raise ConversationError("resume result does not exist")
    else:
        directory = cfg.path(cfg.conversation.result_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"conversation-{uuid.uuid4().hex}.json"
    with run_lock(path):
        if cfg.conversation.resume_result_path:
            result = _read_result(path, require_complete=False)
            if result.get("lifecycle_version", 0) < 2:
                raise ConversationError("resume requires lifecycle version 2 identity and attempt evidence")
            if result.get("resume_identity") != _resume_identity(cfg):
                raise ConversationError("resume configuration differs from the original run")
            if result.get("dataset_sha256") != hashlib.sha256(source_path.read_bytes()).hexdigest():
                raise ConversationError("resume dataset differs from the original run")
            if [item["conversation"] for item in result["conversations"]] != conversations:
                raise ConversationError("resume source evidence differs from the selected dataset")
            events = read_events(Path(result["events_path"]), repair=True)
            _recover_snapshot(result, events)
            for item in result["conversations"]:
                if item["status"] not in {"completed", "skipped"} and item.get("unknown_consumed_tokens"):
                    raise ConversationError("resume cannot enforce token_budget because confirmed usage is missing")
        else:
            run_id = path.stem.removeprefix("conversation-")
            events_path = path.with_suffix(".events.jsonl")
            events_path.touch(exist_ok=False)
            result = {"schema_version": 1, "lifecycle_version": 2, "kind": "sharegpt_conversation", "run_id": run_id,
                      "model": cfg.server.model, "run_config": cfg.conversation.model_dump(),
                      "resume_identity": _resume_identity(cfg), "dataset_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      "events_path": str(events_path), "start_time": _now(),
                      "conversations": [_unstarted(source, run_id) for source in conversations], "attempts": []}
        run_id = result["run_id"]
        events_path = Path(result["events_path"])
        attempt = {"attempt_id": uuid.uuid4().hex, "start_time": _now(),
                   "measurement_config": cfg.conversation.measurement.model_dump(),
                   "measurement_target": cfg.conversation.max_concurrency}
        result["attempts"].append(attempt)
        result["status"] = "running"
        result.pop("end_time", None)
        _persist_result(path, result)
        started = time.monotonic()
        stop = _StopController(cfg.conversation)
        stop.started = started
        measurement = MeasurementRecorder(cfg.conversation.measurement, attempt["attempt_id"], started,
                                          cfg.conversation.max_concurrency, "conversation", stop.request)
        stop.measurement = measurement
        journal = Journal(events_path, run_id=run_id, run_attempt_id=attempt["attempt_id"], measurement=measurement)
        with _capture_signals(stop):
            await _run_workload(cfg, conversations, client, journal, run_id, result, path, stop, measurement)
            stop.check_duration()
            attempt.update(end_time=_now(), duration_s=time.monotonic() - started,
                           stop={"reason": stop.reason, "signal": stop.signum, "requested_at": stop.requested_at,
                                 "policy": cfg.conversation.stop_policy, "grace_sec": cfg.conversation.stop_grace_sec,
                                 "requested_offset_s": stop.monotonic - started if stop.requested else None})
            result.update(status="stopped" if stop.requested else "completed", stop=attempt["stop"], end_time=attempt["end_time"])
            events = read_events(events_path)
            measured = measure_attempt(attempt["measurement_config"], attempt["attempt_id"], events, attempt["duration_s"],
                                       stop.monotonic - started if stop.requested else None, stop.reason,
                                       cfg.conversation.max_concurrency)
            if measured is not None:
                attempt["measurement"] = measured
            _settle_metrics(result, events)
            combine_measurements(result, events)
            result["source_cleanup"] = {"dataset_conversations": dataset_conversations,
                "selected_conversations": len(conversations), "eligible_conversations": result["execution"]["planned_conversations"],
                "skipped_conversations": sum(c["status"] == "skipped" for c in result["conversations"]),
                "raw_user_turns": sum(c["source"]["raw_user_turns"] for c in conversations),
                "eligible_user_turns": result["execution"]["planned_turns"],
                "removed_messages": sum(c["source"]["removed_messages"] for c in conversations)}
            _persist_result(path, result)
        return path


def _read_result(path: Path, *, require_complete=True) -> dict:
    """Read the result while the caller holds its run lock."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("kind") != "sharegpt_conversation" or result.get("schema_version") != 1:
        raise ConversationError("unsupported conversation result")
    if require_complete and (not result.get("end_time") or "metrics" not in result):
        raise ConversationError("conversation result is incomplete")
    events = path.with_suffix(".events.jsonl")
    if not events.exists():
        recorded = Path(result["events_path"])
        events = recorded if recorded.is_absolute() else path.parent / recorded
    if not events.is_file():
        raise ConversationError("conversation event file is missing")
    result["events_path"] = str(events.resolve())
    result["_result_path"] = str(path.resolve())
    return result


def load_conversation_result(path: Path) -> dict:
    """Load a settled result under its run lock."""
    with run_lock(path):
        return _read_result(path)


def sync_conversations(result: dict, cfg: AppConfig, *, client=None, experiment_name: str | None = None) -> dict:
    """Send one consistent result snapshot and record successful delivery."""
    path_value = result.get("_result_path")
    if not path_value and result.get("lifecycle_version", 0) < 2:
        path_value = str(Path(result["events_path"]).with_suffix("").with_suffix(".json"))
    if not path_value:
        raise ConversationError("sync requires a result from load_conversation_result")
    path = Path(path_value)
    with run_lock(path):
        if result.get("lifecycle_version", 0) < 2:
            return _sync_conversations(result, cfg, client=client, experiment_name=experiment_name)
        current = _read_result(path)
        events = read_events(Path(current["events_path"]))
        destination = {"url": cfg.benchmark_platform.url, "workspace": cfg.benchmark_platform.workspace,
                       "project": cfg.benchmark_platform.project_name, "dataset": cfg.benchmark_platform.dataset_name,
                       "experiment_name": experiment_name}
        digest, receipt_key = snapshot_identity(current, events, destination)
        receipt_path = path.with_suffix(".sync-receipt")
        receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if receipt_key in receipts:
            return {**receipts[receipt_key], "already_synced": True}
        current["_snapshot_id"] = digest
        report = _sync_conversations(current, cfg, client=client, experiment_name=experiment_name)
        report.update(snapshot_id=digest, delivery_semantics="at_least_once_until_successful_local_receipt")
        receipts[receipt_key] = report
        _atomic_json(receipt_path, receipts)
        return report


def _sync_conversations(result: dict, cfg: AppConfig, *, client=None, experiment_name: str | None = None) -> dict:
    """Send conversation traces, per-turn spans, and performance scores to Opik."""
    if not result["execution"]["planned_conversations"]:
        return {"run_id": result["run_id"], "experiment_name": None, "conversations": 0, "spans": 0,
                "skipped_conversations": len(result["conversations"]), "reason": "no_usable_user_turns"}
    from opik.api_objects.experiment.experiment_item import ExperimentItemReferences
    with checked_delivery():
        client = client or build_platform_client(cfg.benchmark_platform)
        platform = cfg.benchmark_platform
        events = normalize_events(_events(Path(result["events_path"])))
        metrics = aggregate(events, result["duration_s"])
        dataset = client.get_or_create_dataset(name=platform.dataset_name, description="ShareGPT multi-turn performance conversations")
        indexed = {item.get("conversation_key"): item["id"] for item in dataset.get_items() if item.get("conversation_key")}
        keys = {}
        for conversation in result["conversations"]:
            source = conversation["conversation"]
            key = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            keys[source["conversation_id"]] = key
            if key not in indexed:
                dataset_item = {field: value for field, value in source.items() if field != "source"}
                dataset_item.update(dataset_source=source.get("source"), conversation_key=key)
                dataset.insert([dataset_item])
        indexed = {item.get("conversation_key"): item["id"] for item in dataset.get_items() if item.get("conversation_key")}
        attempt_id = result.get("attempts", [{}])[-1].get("attempt_id", "legacy")
        snapshot = result.get("_snapshot_id")
        base_name = experiment_name or f"conversation-{result['model']}-{result['run_id']}"
        name = f"{base_name}-attempt-{attempt_id}-snapshot-{snapshot[:16]}" if snapshot else base_name
        snapshot_state = "final" if all(item["status"] in {"completed", "failed", "skipped"} for item in result["conversations"]) else "partial"
        experiment = client.create_experiment(dataset_name=platform.dataset_name, name=name, experiment_config={
            "run_id": result["run_id"], "model": result["model"], "run": result["run_config"], "metrics": metrics,
            "metric_definitions": METRIC_DEFINITIONS, "execution": result["execution"],
            "source_cleanup": result.get("source_cleanup"), "stop": result.get("stop"),
            "status": result.get("status"), "attempts": result.get("attempts"), "attempt_metrics": result.get("attempt_metrics"),
            "snapshot_id": snapshot, "snapshot_state": snapshot_state, "score_scope": "immutable_cumulative_result_snapshot", "quality_evaluation": "none"})
        refs, count = [], 0
        for conversation in result["conversations"]:
            identity = conversation["conversation"]["conversation_id"]
            trace_id = platform_id(f"{conversation['trace_id']}:{snapshot}" if snapshot else conversation["trace_id"], conversation["start_time"], "trace")
            matching = [event for event in events if event.get("conversation_id") == identity]
            scoped_metrics = aggregate(matching, conversation["duration_s"])
            client.trace(id=trace_id, project_name=platform.project_name, name=f"conversation:{identity}",
                         start_time=dt.datetime.fromisoformat(conversation["start_time"]), end_time=dt.datetime.fromisoformat(conversation["end_time"]),
                         input=conversation["conversation"], output=conversation, tags=platform.tags,
                         metadata={"run_id": result["run_id"], "logical_conversation_id": conversation["trace_id"],
                                   "snapshot_id": snapshot, "snapshot_state": snapshot_state, "history_policy": result["run_config"]["history_policy"], "metrics": scoped_metrics,
                                   "metric_definitions": METRIC_DEFINITIONS, "quality_evaluation": "none", "error": conversation.get("error"),
                                   "source_cleanup": conversation["conversation"].get("source"),
                                   "status": conversation["status"], "skip_reason": conversation.get("skip_reason")},
                         feedback_scores=([{ "name": metric_names.CONVERSATION_DURATION_SCORE, "value": conversation["duration_s"]}] + task_scores(scoped_metrics)
                                          if conversation["status"] not in {"skipped", "not_started"} else []))
            ends = {event["span_id"]: event for event in matching if event["event"] == "llm_end"}
            for start in [event for event in matching if event["event"] == "llm_start"]:
                end = ends.get(start["span_id"], {"timestamp": conversation["end_time"], "error": "request interrupted"})
                client.span(id=platform_id(f"{start['span_id']}:{snapshot}" if snapshot else start["span_id"], start["timestamp"], "span"), trace_id=trace_id, project_name=platform.project_name,
                            name=f"turn:{start['turn']}", type="llm", model=result["model"], provider="openai",
                            start_time=dt.datetime.fromisoformat(start["timestamp"]), end_time=dt.datetime.fromisoformat(end["timestamp"]),
                            input=start["input"], output=end.get("output"), usage=end.get("usage"),
                            metadata={"turn": start["turn"], "physical_request_id": start["span_id"],
                                      "logical_turn_id": start.get("logical_turn_id"), "run_attempt_id": start.get("run_attempt_id"),
                                      "snapshot_id": snapshot, "context": end.get("context", start["context"]), "timing": end.get("timing"),
                                      "metric_definitions": METRIC_DEFINITIONS, "error": end.get("error"), "finish_reason": end.get("finish_reason"), "status": end.get("status")},
                            feedback_scores=[] if end.get("status") == "cancelled" else call_scores(end.get("timing") or {}, end.get("usage") or {}, not end.get("error") and "output" in end))
                count += 1
            refs.append(ExperimentItemReferences(dataset_item_id=indexed[keys[identity]], trace_id=trace_id))
        experiment.insert(refs)
        client.trace(id=platform_id(f"{result['run_id']}:{snapshot}" if snapshot else result["run_id"], result["start_time"], "summary"), project_name=platform.project_name,
                     name=metric_names.SUMMARY_TRACE_NAME, start_time=dt.datetime.fromisoformat(result["start_time"]), end_time=dt.datetime.fromisoformat(result["end_time"]),
                     input={"run": result["run_config"]}, output=metric_names.translate_grouped({k: v for k, v in metrics.items() if k not in {"sample_counts", "scope", "metric_definition_version"}}),
                     metadata={"run_id": result["run_id"], "execution": result["execution"], "metric_definitions": METRIC_DEFINITIONS,
                               "sample_counts": metrics["sample_counts"], "source_cleanup": result.get("source_cleanup"), "stop": result.get("stop"),
                               "status": result.get("status"), "snapshot_id": snapshot, "snapshot_state": snapshot_state,
                               "attempts": result.get("attempts"), "attempt_metrics": result.get("attempt_metrics"),
                               "score_scope": "immutable_cumulative_result_snapshot", "throughput_scope": "sum_active_attempt_windows", "quality_evaluation": "none"},
                     tags=platform.tags + ["summary"], feedback_scores=summary_scores(metrics))
        sync_measurement_trace(result, cfg, client, snapshot)
        checked_flush(client)
    return {"run_id": result["run_id"], "experiment_name": name, "conversations": len(refs), "spans": count}
