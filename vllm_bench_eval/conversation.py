"""Measure sequential ShareGPT conversations with accumulated message history."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx

from . import metric_names
from .agent_metrics import METRIC_DEFINITIONS, aggregate, call_scores, normalize_events, summary_scores, task_scores
from .agent_platform import checked_delivery, checked_flush, platform_id
from .agent_transport import chat_request
from .coding_agent import Journal, _atomic_json, _events, _known_tokens, _now
from .config import AppConfig, ConversationConfig
from .platform_sync import build_platform_client


class ConversationError(ValueError):
    """The conversation input or response violates the supported format."""


def load_conversations(path: Path) -> list[dict]:
    """Read a ShareGPT JSON array without removing turns or changing content."""
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
        if not isinstance(raw, list) or not raw:
            raise ConversationError(f"conversation {source_id}: conversations must be a nonempty array")
        messages = []
        expected = "user"
        for position, message in enumerate(raw):
            if not isinstance(message, dict):
                raise ConversationError(f"conversation {source_id}, message {position}: expected an object")
            role = roles.get(message.get("from")) if isinstance(message.get("from"), str) else None
            content = message.get("value")
            if role is None or not isinstance(content, str) or not content.strip():
                raise ConversationError(f"conversation {source_id}, message {position}: unsupported role or empty text")
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
        if not any(message["role"] == "user" for message in messages):
            raise ConversationError(f"conversation {source_id}: no user turn")
        conversations.append({"conversation_id": source_id, "messages": messages})
    return conversations


def _execute(source: dict, cfg: AppConfig, client: httpx.Client, journal: Journal, run_id: str) -> dict:
    limits = cfg.conversation
    started, start_time = time.monotonic(), _now()
    identity = source["conversation_id"]
    trace_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbe-conversation:{run_id}:{identity}"))
    result = {"conversation": source, "trace_id": trace_id, "start_time": start_time,
              "status": "completed", "execution_ok": True, "turns": [], "planned_turns": sum(m["role"] == "user" for m in source["messages"])}
    history, consumed, local_events = [], 0, []
    def emit(event):
        event = {"task_id": identity, "conversation_id": identity, **event}
        journal.emit(event)
        local_events.append(event)
    try:
        if result["planned_turns"] > limits.max_turns:
            raise ConversationError(f"conversation has {result['planned_turns']} turns, above max_turns={limits.max_turns}")
        for message in source["messages"]:
            if message["role"] == "system":
                history.append(copy.deepcopy(message))
                continue
            if message["role"] == "assistant":
                if limits.history_policy == "dataset":
                    history[-1] = copy.deepcopy(message)
                continue
            history.append(copy.deepcopy(message))
            turn = len(result["turns"]) + 1
            context = {"message_count": len(history), "context_chars": sum(len(m["content"]) for m in history),
                       "history_policy": limits.history_policy, "prompt_tokens": None}
            if context["context_chars"] > limits.max_context_chars:
                raise ConversationError(f"turn {turn}: context exceeds max_context_chars={limits.max_context_chars}")
            remaining = limits.conversation_timeout_sec - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("conversation timeout exceeded")
            if consumed >= limits.token_budget:
                raise ConversationError("conversation token_budget exhausted")
            payload = {**limits.extra_body, "model": cfg.server.model, "messages": copy.deepcopy(history),
                       "stream": limits.stream, "max_tokens": min(limits.max_tokens_per_turn, limits.token_budget - consumed), "temperature": 0}
            if limits.stream:
                payload["stream_options"] = {"include_usage": True}
            span_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{trace_id}:{turn}"))
            emit({"event": "llm_start", "span_id": span_id, "turn": turn, "input": payload, "context": context})
            request_started = time.monotonic()
            timeout = min(limits.request_timeout_sec, remaining)
            try:
                body, timing = chat_request(client, cfg.server.base_url + limits.endpoint, payload, timeout)
                if time.monotonic() - request_started > timeout:
                    raise TimeoutError("request or conversation timeout exceeded")
                choices = body.get("choices")
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise ConversationError("response requires exactly one choice")
                choice = choices[0]
                output = choice.get("message")
                if (not isinstance(output, dict) or output.get("role") != "assistant" or
                        not isinstance(output.get("content"), str) or not output["content"].strip() or output.get("tool_calls")):
                    raise ConversationError("response requires assistant text without tool calls")
                if choice.get("finish_reason") != "stop":
                    raise ConversationError(f"response did not finish normally: {choice.get('finish_reason')}")
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
                result["turns"].append({"turn": turn, "output": output, "usage": usage, "timing": timing, "context": context})
                history.append({"role": "assistant", "content": output["content"]})
                tokens = _known_tokens(usage)
                if tokens is None:
                    raise ConversationError("response total_tokens is missing, cannot enforce token_budget")
                consumed += tokens
                if consumed > limits.token_budget:
                    raise ConversationError("conversation token_budget exceeded")
            except Exception as exc:
                # Completed requests remain successful when a workload limit stops later turns.
                if not local_events or local_events[-1]["event"] != "llm_end":
                    emit({"event": "llm_end", "span_id": span_id, "turn": turn, "error": str(exc),
                          "timing": getattr(exc, "agent_timing", {"stream": limits.stream, "e2e_ms": (time.monotonic() - request_started) * 1000,
                                    "tpot_null_reason": "incomplete_request"}), "usage": getattr(exc, "agent_usage", {}), "context": context})
                raise
    except Exception as exc:
        result.update(status="failed", execution_ok=False, error=f"{type(exc).__name__}: {exc}")
    result.update(end_time=_now(), duration_s=time.monotonic() - started, known_consumed_tokens=consumed)
    result["metrics"] = aggregate(local_events, result["duration_s"])
    return result


def run_conversations(cfg: AppConfig, *, client: httpx.Client | None = None) -> Path:
    """Execute each conversation sequentially and independent conversations concurrently."""
    cfg.conversation = ConversationConfig.model_validate(cfg.conversation.model_dump())
    if not cfg.server.model.strip():
        raise ConversationError("server.model is required")
    source_path = cfg.path(cfg.conversation.dataset_path).resolve()
    conversations = load_conversations(source_path)
    if cfg.conversation.num_conversations is not None:
        conversations = conversations[:cfg.conversation.num_conversations]
    run_id = uuid.uuid4().hex
    directory = cfg.path(cfg.conversation.result_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"conversation-{run_id}.json"
    events_path = path.with_suffix(".events.jsonl")
    events_path.touch(exist_ok=False)
    journal = Journal(events_path, run_id=run_id)
    result = {"schema_version": 1, "kind": "sharegpt_conversation", "run_id": run_id, "model": cfg.server.model,
              "run_config": cfg.conversation.model_dump(), "dataset_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "events_path": str(events_path), "start_time": _now(), "conversations": []}
    started = time.monotonic()
    _atomic_json(path, result)
    with ExitStack() as stack:
        if client is None:
            headers = {"Authorization": f"Bearer {cfg.server.api_key}"} if cfg.server.api_key else {}
            client = stack.enter_context(httpx.Client(headers=headers))
        with ThreadPoolExecutor(max_workers=cfg.conversation.max_concurrency) as pool:
            futures = [pool.submit(_execute, source, cfg, client, journal, run_id) for source in conversations]
            for future in futures:
                result["conversations"].append(future.result())
                _atomic_json(path, result)
    result.update(end_time=_now(), duration_s=time.monotonic() - started)
    result["metrics"] = aggregate(_events(events_path), result["duration_s"])
    result["execution"] = {"planned_conversations": len(conversations), "completed_conversations": sum(c["execution_ok"] for c in result["conversations"]),
                           "planned_turns": sum(c["planned_turns"] for c in result["conversations"]), "completed_turns": result["metrics"]["counts"]["completed"]}
    _atomic_json(path, result)
    return path


def load_conversation_result(path: Path) -> dict:
    """Load a completed result and resolve its event file beside the result."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("kind") != "sharegpt_conversation" or result.get("schema_version") != 1:
        raise ConversationError("unsupported conversation result")
    if not result.get("end_time") or "metrics" not in result:
        raise ConversationError("conversation result is incomplete")
    events = path.with_suffix(".events.jsonl")
    if not events.exists():
        recorded = Path(result["events_path"])
        events = recorded if recorded.is_absolute() else path.parent / recorded
    if not events.is_file():
        raise ConversationError("conversation event file is missing")
    result["events_path"] = str(events.resolve())
    return result


def sync_conversations(result: dict, cfg: AppConfig, *, client=None, experiment_name: str | None = None) -> dict:
    """Send conversation traces, per-turn spans, and performance scores to Opik."""
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
                dataset.insert([{**source, "conversation_key": key}])
        indexed = {item.get("conversation_key"): item["id"] for item in dataset.get_items() if item.get("conversation_key")}
        name = experiment_name or f"conversation-{result['model']}-{result['run_id']}"
        experiment = client.create_experiment(dataset_name=platform.dataset_name, name=name, experiment_config={
            "run_id": result["run_id"], "model": result["model"], "run": result["run_config"], "metrics": metrics,
            "metric_definitions": METRIC_DEFINITIONS, "execution": result["execution"], "quality_evaluation": "none"})
        refs, count = [], 0
        for conversation in result["conversations"]:
            identity = conversation["conversation"]["conversation_id"]
            trace_id = platform_id(conversation["trace_id"], conversation["start_time"], "trace")
            matching = [event for event in events if event["conversation_id"] == identity]
            scoped_metrics = aggregate(matching, conversation["duration_s"])
            client.trace(id=trace_id, project_name=platform.project_name, name=f"conversation:{identity}",
                         start_time=dt.datetime.fromisoformat(conversation["start_time"]), end_time=dt.datetime.fromisoformat(conversation["end_time"]),
                         input=conversation["conversation"], output=conversation, tags=platform.tags,
                         metadata={"run_id": result["run_id"], "history_policy": result["run_config"]["history_policy"], "metrics": scoped_metrics,
                                   "metric_definitions": METRIC_DEFINITIONS, "quality_evaluation": "none", "error": conversation.get("error")},
                         feedback_scores=[{"name": "conversation_execution_ok", "value": int(conversation["execution_ok"])},
                                          {"name": "conversation_duration_s", "value": conversation["duration_s"]}] + task_scores(scoped_metrics))
            ends = {event["span_id"]: event for event in matching if event["event"] == "llm_end"}
            for start in [event for event in matching if event["event"] == "llm_start"]:
                end = ends.get(start["span_id"], {"timestamp": conversation["end_time"], "error": "request interrupted"})
                client.span(id=platform_id(start["span_id"], start["timestamp"], "span"), trace_id=trace_id, project_name=platform.project_name,
                            name=f"turn:{start['turn']}", type="llm", model=result["model"], provider="openai",
                            start_time=dt.datetime.fromisoformat(start["timestamp"]), end_time=dt.datetime.fromisoformat(end["timestamp"]),
                            input=start["input"], output=end.get("output"), usage=end.get("usage"),
                            metadata={"turn": start["turn"], "context": end.get("context", start["context"]), "timing": end.get("timing"),
                                      "metric_definitions": METRIC_DEFINITIONS, "error": end.get("error"), "finish_reason": end.get("finish_reason")},
                            feedback_scores=call_scores(end.get("timing") or {}, end.get("usage") or {}, not end.get("error") and "output" in end))
                count += 1
            refs.append(ExperimentItemReferences(dataset_item_id=indexed[keys[identity]], trace_id=trace_id))
        experiment.insert(refs)
        client.trace(id=platform_id(result["run_id"], result["start_time"], "summary"), project_name=platform.project_name,
                     name=metric_names.SUMMARY_TRACE_NAME, start_time=dt.datetime.fromisoformat(result["start_time"]), end_time=dt.datetime.fromisoformat(result["end_time"]),
                     input={"run": result["run_config"]}, output=metric_names.translate_grouped({k: v for k, v in metrics.items() if k not in {"sample_counts", "scope", "metric_definition_version"}}),
                     metadata={"run_id": result["run_id"], "execution": result["execution"], "metric_definitions": METRIC_DEFINITIONS,
                               "sample_counts": metrics["sample_counts"], "throughput_scope": "complete_conversation_workload_window", "quality_evaluation": "none"},
                     tags=platform.tags + ["summary"], feedback_scores=summary_scores(metrics))
        checked_flush(client)
    return {"run_id": result["run_id"], "experiment_name": name, "conversations": len(refs), "spans": count}
