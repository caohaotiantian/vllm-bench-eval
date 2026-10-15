"""Execute sequential coding tasks and preserve their complete event tree."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
from contextlib import contextmanager, ExitStack
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .agent_transport import chat_request
from .agent_metrics import (aggregate, call_scores, summary_scores, task_scores, normalize_events, normalize_result, METRIC_DEFINITIONS)
from .agent_platform import platform_id, checked_delivery, checked_flush
from . import metric_names
from .config import AppConfig
from .platform_sync import build_platform_client
from .lifecycle import (LifecycleError, capture_stop_signals, read_events, run_lock, settle_attempts, snapshot_identity,
                        MeasurementRecorder, measure_attempt, combine_measurements, recover_measurements,
                        ThreadStopController, sync_measurement_trace)


class AgentError(RuntimeError):
    """The task input or execution cannot continue."""


class CodingTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    repository: str
    verify_command: list[str] = Field(min_length=1)
    verification_files: list[str] = Field(default_factory=list)
    required_tools: list[Literal["read_file", "write_file", "run_command"]] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


def load_tasks(path: Path) -> list[CodingTask]:
    tasks = []
    ids = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            task = CodingTask.model_validate_json(line)
            if task.task_id in ids:
                raise ValueError(f"duplicate task_id: {task.task_id}")
            repository = Path(task.repository).expanduser()
            repository = (path.parent / repository).resolve() if not repository.is_absolute() else repository.resolve()
            if not repository.is_dir():
                raise ValueError(f"repository does not exist: {repository}")
            task.repository = str(repository)
            ids.add(task.task_id)
            tasks.append(task)
        except Exception as exc:
            raise AgentError(f"{path}:{line_no}: {exc}") from exc
    if not tasks:
        raise AgentError("task dataset is empty")
    return tasks


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _id(run_id: str, key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbe-agent:{run_id}:{key}"))


def _typed_id(run_id: str, kind: str, *parts: Any) -> str:
    key = json.dumps([run_id, kind, *parts], ensure_ascii=False, separators=(",", ":"))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vbe-agent-v2:{key}"))


def _known_tokens(usage: dict | None) -> int | None:
    value = (usage or {}).get("total_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class Journal:
    def __init__(self, path: Path, *, measurement=None, **context):
        self.measurement = measurement
        if measurement is not None:
            measurement.journal = self
        self.context = context
        self.path = path
        self.lock = threading.RLock()

    def emit(self, event: dict) -> None:
        with self.lock:
            event = {"timestamp": _now(), **self.context, **event}
            if self.measurement is not None:
                self.measurement.record_event(event)
            event = json.loads(json.dumps(event, ensure_ascii=False))
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


def _file(workspace: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AgentError("path must be a nonempty relative string")
    path = (workspace / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(workspace.resolve()):
        raise AgentError("path escapes the task workspace")
    return path


def _verification_snapshot(task: CodingTask, workspace: Path) -> dict[str, str]:
    paths = set(task.verification_files)
    for argument in task.verify_command:
        try:
            candidate = _file(workspace, argument)
        except AgentError:
            continue
        if candidate.is_file():
            paths.add(argument)
    return {relative: hashlib.sha256(_file(workspace, relative).read_bytes()).hexdigest() for relative in paths}


def _verify_unchanged(workspace: Path, snapshots: dict[str, str]) -> None:
    for relative, expected in snapshots.items():
        try:
            actual = hashlib.sha256(_file(workspace, relative).read_bytes()).hexdigest()
        except (OSError, AgentError) as exc:
            raise AgentError(f"verification files tampered: {relative}") from exc
        if actual != expected:
            raise AgentError(f"verification files tampered: {relative}")


def _command(argv: list[str], workspace: Path, timeout: float, limit: int, cancel=None) -> dict:
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise AgentError("argv must contain nonempty strings")
    # A temporary file prevents large process output from filling memory.
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(argv, cwd=workspace, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("evaluation cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                try:
                    process.wait(timeout=min(.1, remaining))
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        output.seek(0)
        text = output.read(limit * 4 + 1).decode("utf-8", errors="replace")
        return {"exit_code": process.returncode, "timed_out": timed_out,
                "output": text[:limit], "truncated": len(text) > limit}


TOOLS = [
    {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}
    for name, description, properties, required in [
        ("read_file", "Read a UTF-8 file in the task workspace.", {"path": {"type": "string"}}, ["path"]),
        ("write_file", "Write a UTF-8 file in the task workspace.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        ("run_command", "Execute an allowed argument vector in the task workspace.", {"argv": {"type": "array", "items": {"type": "string"}}}, ["argv"]),
    ]
]


def _tools(cfg: AppConfig) -> list[dict]:
    tools = json.loads(json.dumps(TOOLS))
    options = cfg.coding_agent
    description = "Command execution is disabled."
    if options.allow_commands:
        description = "Execute any valid argument vector in the task workspace." if options.allow_any_command else "Execute the exact verification argv or an argv that matches a configured command prefix."
    tools[-1]["function"]["description"] = description
    return tools


def _tool(name: str, args: dict, workspace: Path, cfg: AppConfig, remaining: float, cancel=None, verify_command=None) -> dict:
    options = cfg.coding_agent
    if name == "read_file" and set(args) == {"path"}:
        path = _file(workspace, args["path"])
        with path.open(encoding="utf-8") as stream:
            content = stream.read(options.max_output_chars + 1)
        return {"content": content[:options.max_output_chars], "truncated": len(content) > options.max_output_chars}
    if name == "write_file" and set(args) == {"path", "content"}:
        if not isinstance(args["content"], str):
            raise AgentError("content must be a string")
        path = _file(workspace, args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return {"written": True}
    if name == "run_command" and set(args) == {"argv"}:
        argv = args["argv"]
        if not options.allow_commands:
            raise AgentError("command tools are disabled")
        if not options.allow_any_command and (not isinstance(argv, list) or (argv != verify_command and not any(argv[:len(prefix)] == prefix for prefix in options.command_allowlist if prefix))):
            raise AgentError("command is outside command_allowlist")
        return _command(argv, workspace, min(remaining, options.command_timeout_sec), options.max_output_chars, cancel)
    raise AgentError(f"unknown tool or invalid arguments: {name}")


def _execute(task: CodingTask, workspace: Path, cfg: AppConfig, client: httpx.Client, journal: Journal, run_id: str, snapshots: dict[str, str], *, checkpoint=None, recovered=None, cancel=None, dispatch_stop=None) -> dict:
    options = cfg.coding_agent
    started = time.monotonic()
    prior_elapsed = recovered.get("elapsed_s", 0) if recovered else 0
    deadline = started + options.task_timeout_sec - prior_elapsed
    task_trace = recovered["result"]["trace_id"] if recovered else _typed_id(run_id, "task", task.task_id)
    result = {"task": task.model_dump(), "trace_id": task_trace, "workspace": str(workspace),
              "start_time": _now(), "status": "error", "success": False, "turns": 0,
              "tool_calls": 0, "tool_failures": 0, "total_tokens": 0, "final_answer": "",
              "verification_hashes": snapshots, "llm_calls": [], "tool_types": [],
              "effective_required_tools": options.required_tools if task.required_tools is None else task.required_tools}
    messages = [{"role": "system", "content": options.system_prompt}, {"role": "user", "content": task.prompt}]
    if recovered:
        result = recovered["result"]
        result.pop("error", None)
        result["status"] = "error"
        result["success"] = False
        messages = recovered["messages"]
    result.setdefault("tool_types", [])
    result.setdefault("effective_required_tools", [])
    required_tools = result["effective_required_tools"]
    if not recovered:
        messages[0]["content"] += "\nUse each required tool at least once: " + ", ".join(required_tools) + "."
        messages[0]["content"] += "\nThe verification command argv is: " + json.dumps(task.verify_command)
        if not options.allow_commands:
            messages[0]["content"] += "\nCommand execution is disabled."
        elif options.allow_any_command:
            messages[0]["content"] += "\nAny valid command argv is allowed in the task workspace."
        else:
            messages[0]["content"] += "\nOnly the exact verification argv or configured command prefixes are allowed: " + json.dumps(options.command_allowlist)
    next_turn = recovered["next_turn"] if recovered else 1
    attempt = recovered.get("attempt", 0) + 1 if recovered else 1
    def save(phase: str, next_index: int) -> None:
        if checkpoint:
            checkpoint({"phase": phase, "run_id": run_id, "task_id": task.task_id, "messages": messages, "result": result, "next_turn": next_index,
                        "elapsed_s": time.monotonic() - started + prior_elapsed, "attempt": attempt,
                        "event_offset": journal.path.stat().st_size if journal.path.exists() else 0})
    def remaining() -> float:
        if dispatch_stop is not None:
            dispatch_stop.check()
            if dispatch_stop.dispatch.is_set():
                raise InterruptedError("evaluation cancelled")
        if cancel is not None and cancel.is_set():
            raise InterruptedError("evaluation cancelled")
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise AgentError("task timeout exceeded")
        return seconds
    def event(kind: str, **data: Any) -> None:
        journal.emit({"run_id": run_id, "task_id": task.task_id, "event": kind,
                      "elapsed_s": time.monotonic() - started + prior_elapsed, **data})
    event("task_resume" if recovered else "task_start", trace_id=task_trace, workspace=str(workspace), verification_hashes=snapshots, attempt=attempt)
    try:
        _verify_unchanged(workspace, snapshots)
        if not recovered:
            save("ready", 1)
        if result["total_tokens"] >= options.token_budget:
            raise AgentError("token budget exhausted")
        for turn in range(next_turn, options.max_turns + 1):
            turn_id = _typed_id(run_id, "turn", task.task_id, turn, attempt)
            llm_id = _typed_id(run_id, "llm", task.task_id, turn, attempt)
            remaining()
            save("running", turn)
            result["turns"] = turn
            event("turn_start", span_id=turn_id, parent_span_id=None, turn=turn, attempt=attempt)
            payload = {**options.extra_body, "model": cfg.server.model, "messages": messages, "tools": _tools(cfg),
                       "max_tokens": min(options.max_tokens_per_turn, options.token_budget - result["total_tokens"]),
                       "temperature": 0, "stream": options.stream}
            if options.stream:
                payload["stream_options"] = {"include_usage": True}
            llm_started = time.monotonic()
            event("llm_start", span_id=llm_id, parent_span_id=turn_id, input={"messages": messages}, request=payload, turn=turn, attempt=attempt)
            timing = usage = finish = None
            usage_accounted = False
            try:
                body, timing = chat_request(client, cfg.server.base_url + options.endpoint, payload,
                                            min(remaining(), options.request_timeout_sec), cancel)
                usage = body.get("usage") or {}
                finish = body.get("choices", [{}])[0].get("finish_reason") if body.get("choices") else None
                # Missing usage prevents enforcement of the configured token budget.
                tokens = usage.get("total_tokens")
                if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                    raise AgentError("response requires usage.total_tokens for the token budget")
                result["total_tokens"] += tokens
                usage_accounted = True
                message = body["choices"][0]["message"]
                if message.get("role") != "assistant":
                    raise AgentError("response message must have assistant role")
                if body["choices"][0].get("finish_reason") not in ("stop", "tool_calls"):
                    raise AgentError("model did not finish normally")
                event("llm_end", span_id=llm_id, output=message, usage=usage, timing=timing, finish_reason=finish)
                result["llm_calls"].append({"span_id": llm_id, "turn": turn, "success": True, "usage": usage, "finish_reason": finish, **timing})
            except (Exception, KeyboardInterrupt, SystemExit) as exc:
                failed_timing = timing if timing is not None else getattr(exc, "agent_timing", {"e2e_ms": (time.monotonic() - llm_started) * 1000, "ttft_ms": None, "tpot_ms": None})
                failed_usage = usage if usage is not None else getattr(exc, "agent_usage", None)
                known_tokens = _known_tokens(failed_usage)
                if not usage_accounted and known_tokens is not None:
                    result["total_tokens"] += known_tokens
                event("llm_end", span_id=llm_id, error=str(exc) or type(exc).__name__, timing=failed_timing, usage=failed_usage, finish_reason=finish,
                      **({"status": "cancelled"} if isinstance(exc, (KeyboardInterrupt, SystemExit, InterruptedError)) else {}))
                result["llm_calls"].append({"span_id": llm_id, "turn": turn, "success": False, "error": str(exc),
                                           "usage": failed_usage, "finish_reason": finish, **failed_timing})
                raise
            remaining()
            if result["total_tokens"] > options.token_budget:
                raise AgentError("token budget exceeded")
            messages.append({key: message[key] for key in ("role", "content", "tool_calls") if key in message})
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise AgentError("tool_calls must be a list")
            if not calls:
                if body["choices"][0].get("finish_reason") != "stop":
                    raise AgentError("model did not finish normally")
                result["final_answer"] = message.get("content") or ""
                event("turn_end", span_id=turn_id)
                verification_id = _typed_id(run_id, "verification", task.task_id, attempt)
                event("verification_start", span_id=verification_id, input={"argv": task.verify_command})
                try:
                    _verify_unchanged(workspace, snapshots)
                    verification = _command(task.verify_command, workspace, min(remaining(), options.command_timeout_sec), options.max_output_chars, cancel)
                    _verify_unchanged(workspace, snapshots)
                    event("verification_end", span_id=verification_id, output=verification)
                except Exception as exc:
                    event("verification_end", span_id=verification_id, error=str(exc))
                    raise
                remaining()
                result["verification"] = verification
                missing_tools = sorted(set(required_tools) - set(result["tool_types"]))
                result["missing_required_tools"] = missing_tools
                result["success"] = verification["exit_code"] == 0 and not verification["timed_out"] and not missing_tools
                result["status"] = "passed" if result["success"] else "required_tools_missing" if missing_tools else "verification_failed"
                break
            call_ids = set()
            for index, call in enumerate(calls):
                remaining()
                if result["tool_calls"] >= options.max_tool_calls:
                    raise AgentError("tool call limit exceeded")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in call_ids:
                    raise AgentError("tool call IDs must be unique nonempty strings")
                call_ids.add(call_id)
                result["tool_calls"] += 1
                tool_id = _typed_id(run_id, "tool", task.task_id, turn, index, attempt)
                function = call["function"]
                event("tool_start", span_id=tool_id, parent_span_id=turn_id, name=function["name"], input={"arguments": function.get("arguments")}, turn=turn, attempt=attempt)
                try:
                    args = json.loads(function["arguments"])
                    output = _tool(function["name"], args, workspace, cfg, remaining(), cancel, task.verify_command)
                    if function["name"] not in result["tool_types"]:
                        result["tool_types"].append(function["name"])
                    if output.get("exit_code", 0) != 0 or output.get("timed_out"):
                        result["tool_failures"] += 1
                except Exception as exc:
                    output = {"error": str(exc)}
                    result["tool_failures"] += 1
                event("tool_end", span_id=tool_id, output=output)
                _verify_unchanged(workspace, snapshots)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(output, ensure_ascii=False)})
            event("turn_end", span_id=turn_id)
            save("ready", turn + 1)
            if result["total_tokens"] >= options.token_budget:
                raise AgentError("token budget exhausted")
        else:
            raise AgentError("turn limit exceeded")
    except BaseException as exc:
        result["error"] = str(exc) or type(exc).__name__
        result["status"] = "verification_tampered" if "verification files tampered" in str(exc) else "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit, InterruptedError)) else "limit_exceeded" if any(word in str(exc) for word in ("limit", "budget", "timeout")) else "error"
    result["end_time"] = _now()
    result["duration_s"] = time.monotonic() - started + prior_elapsed
    event("task_end", result=result)
    if result["status"] != "interrupted":
        save("complete", result["turns"] + 1)
    return result


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_config(cfg: AppConfig) -> dict:
    settings = cfg.coding_agent.model_dump(exclude={"duration_sec", "stop_policy", "stop_grace_sec", "measurement", "mode", "max_concurrency", "resume_result_path", "replay_result_path", "result_dir", "workspace_dir", "tasks_path"})
    return {"model": cfg.server.model, "settings": settings}


class RunStore:
    """Serialize manifest and checkpoint replacement across task workers."""
    def __init__(self, path: Path, payload: dict):
        self.path, self.payload = path, payload
        self.lock = threading.Lock()

    def save(self) -> None:
        with self.lock:
            _atomic_json(self.path, self.payload)

    def checkpoint(self, index: int, state: dict) -> None:
        with self.lock:
            path = Path(self.payload["checkpoints"][index])
            _atomic_json(path, state)

    def result(self, index: int, result: dict) -> None:
        with self.lock:
            self.payload["tasks"][index] = result
            _atomic_json(self.path, self.payload)


def _events(path: Path) -> list[dict]:
    events = []
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise AgentError("event file contains invalid data")
            raise AgentError("event file has a partial final record; preserve it before recovery")
    return events


def _recovery(result: dict, cfg: AppConfig, tasks: list[CodingTask]) -> list[dict | None]:
    if result["schema_version"] != 2 or result.get("mode") != "live":
        raise AgentError("resume requires a version 2 live result")
    original_config = json.loads(json.dumps(result["run_config"]))
    original_config["settings"].setdefault("allow_any_command", False)
    original_config["settings"].setdefault("extra_body", {})
    original_config["settings"].setdefault("required_tools", cfg.coding_agent.required_tools)
    if original_config != _run_config(cfg):
        raise AgentError("resume configuration differs from the original run")
    if [task.model_dump() for task in tasks] != [CodingTask.model_validate(item["task"]).model_dump() for item in result["tasks"]]:
        raise AgentError("resume task dataset differs from the original run")
    events = _events(Path(result["events_path"]))
    states = []
    for index, item in enumerate(result["tasks"]):
        checkpoint_path = Path(result["checkpoints"][index])
        if not checkpoint_path.exists():
            if item["status"] != "pending":
                raise AgentError("resume checkpoint is missing")
            if Path(item["workspace"]).exists():
                raise AgentError("unsafe resume: uncheckpointed workspace exists")
            states.append(None)
            continue
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state.get("run_id", result["run_id"]) != result["run_id"] or state.get("task_id", item["task"]["task_id"]) != item["task"]["task_id"]:
            raise AgentError("resume checkpoint identity differs")
        if CodingTask.model_validate(state["result"]["task"]).model_dump() != CodingTask.model_validate(item["task"]).model_dump() or state["result"]["workspace"] != item["workspace"]:
            raise AgentError("resume workspace or task identity differs")
        if state["result"]["verification_hashes"] != item["verification_hashes"]:
            raise AgentError("resume verification hashes differ")
        if state["phase"] not in ("complete", "ready", "running"):
            raise AgentError("resume checkpoint phase is invalid")
        if state["phase"] == "complete":
            states.append(state)
            continue
        if not Path(item["workspace"]).is_dir():
            raise AgentError("resume workspace is missing")
        _verify_unchanged(Path(item["workspace"]), item["verification_hashes"])
        newer = []
        offset = 0
        # Checkpoint offsets refer to bytes in the append-only journal.
        with Path(result["events_path"]).open("rb") as stream:
            for line in stream:
                if offset >= state["event_offset"]:
                    event = json.loads(line)
                    if event.get("task_id") == item["task"]["task_id"]:
                        newer.append(event)
                offset += len(line)
        if any(event["event"] in ("tool_start", "verification_start") for event in newer):
            raise AgentError(f"unsafe resume: task {item['task']['task_id']} has tool or verification activity after its checkpoint")
        state["elapsed_s"] = max([state["elapsed_s"], item.get("duration_s", 0)] + [event.get("elapsed_s", 0) for event in newer])
        for event in newer:
            if event["event"] == "llm_end" and event.get("usage"):
                state["result"]["total_tokens"] += event["usage"].get("total_tokens", 0)
        states.append(state)
    return states


@contextmanager
def _run_lock(path: Path):
    try:
        with run_lock(path):
            yield
    except LifecycleError as exc:
        raise AgentError(str(exc)) from exc


def run_agent(cfg: AppConfig, *, client: httpx.Client | None = None) -> Path:
    cfg.coding_agent = type(cfg.coding_agent).model_validate(cfg.coding_agent.model_dump())
    with ExitStack() as guard:
        if cfg.coding_agent.resume_result_path:
            path = cfg.path(cfg.coding_agent.resume_result_path).resolve()
            if not path.exists():
                raise AgentError("resume result does not exist")
            guard.enter_context(_run_lock(path))
        return _run_agent(cfg, client=client, guard=guard)


def _run_agent(cfg: AppConfig, *, client: httpx.Client | None = None, guard) -> Path:
    if cfg.coding_agent.mode == "replay" or cfg.coding_agent.replay_result_path:
        if cfg.coding_agent.resume_result_path:
            raise AgentError("replay and resume cannot be combined")
        return replay_agent(cfg, client=client)
    if not cfg.server.model:
        raise AgentError("server.model is required")
    if not cfg.coding_agent.allow_commands:
        raise AgentError("coding_agent.allow_commands must be true for trusted verification commands")
    tasks = load_tasks(cfg.path(cfg.coding_agent.tasks_path))
    resuming = bool(cfg.coding_agent.resume_result_path)
    if resuming:
        result_path = cfg.path(cfg.coding_agent.resume_result_path).resolve()
        payload = _load_agent_result(result_path)
        payload.pop("_result_path", None)
        states = _recovery(payload, cfg, tasks)
        run_id = payload["run_id"]
        events_path = Path(payload["events_path"])
        old_events = _events(events_path)
        settle_attempts(payload, old_events)
        recover_measurements(payload, old_events, "agent")
    else:
        run_id = uuid.uuid4().hex
        result_dir = cfg.path(cfg.coding_agent.result_dir).resolve()
        root = cfg.path(cfg.coding_agent.workspace_dir).resolve() / run_id
        for task in tasks:
            if root.is_relative_to(Path(task.repository)):
                raise AgentError("workspace_dir must be outside every task repository")
            if result_dir.is_relative_to(Path(task.repository)):
                raise AgentError("result_dir must be outside every task repository")
            for relative in task.verification_files:
                if not _file(Path(task.repository), relative).is_file():
                    raise AgentError(f"verification file does not exist: {relative}")
            for directory, dirs, files in os.walk(task.repository, followlinks=False):
                dirs[:] = [name for name in dirs if name not in (".git", ".venv", "__pycache__")]
                for name in dirs + files:
                    if (Path(directory) / name).is_symlink():
                        raise AgentError(f"repository contains a symbolic link: {Path(directory) / name}")
        hashes = {task.task_id: _verification_snapshot(task, Path(task.repository)) for task in tasks}
        result_dir.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=False)
        result_path = result_dir / f"agent-{run_id}.json"
        guard.enter_context(_run_lock(result_path))
        events_path = result_path.with_suffix(".events.jsonl")
        events_path.touch()
        payload = {"schema_version": 2, "lifecycle_version": 2, "kind": "coding_agent", "mode": "live", "run_id": run_id,
                   "model": cfg.server.model, "run_config": _run_config(cfg), "events_path": str(events_path),
                   "checkpoints": [str(root / f"{index}.checkpoint.json") for index in range(len(tasks))],
                   "tasks": [{"task": task.model_dump(), "trace_id": _typed_id(run_id, "task", task.task_id),
                              "workspace": str(root / str(index)), "status": "pending", "success": False,
                              "verification_hashes": hashes[task.task_id]} for index, task in enumerate(tasks)]}
        states = [None] * len(tasks)
    attempt_id = uuid.uuid4().hex
    attempt_window = {"attempt_id": attempt_id, "start_time": _now(),
                      "measurement_config": cfg.coding_agent.measurement.model_dump(),
                      "measurement_target": cfg.coding_agent.max_concurrency}
    payload.setdefault("attempts", []).append(attempt_window)
    store = RunStore(result_path, payload)
    store.save()
    attempt_started = time.monotonic()
    owns_client = client is None
    if owns_client:
        headers = {"Authorization": f"Bearer {cfg.server.api_key}"} if cfg.server.api_key else {}
        client = httpx.Client(headers=headers)
    cancel = threading.Event()
    stop = {"reason": None, "signal": None, "requested_at": None, "policy": "cooperative",
            "bounded_http_cancellation": False}
    controller = ThreadStopController(cfg.coding_agent, attempt_started, cancel, stop)
    measurement = MeasurementRecorder(cfg.coding_agent.measurement, attempt_id, attempt_started,
                                      cfg.coding_agent.max_concurrency, "agent", controller.request)
    controller.measurement = measurement
    journal = Journal(events_path, run_id=run_id, run_attempt_id=attempt_id, measurement=measurement)
    def _worker(index: int) -> None:
        state = states[index]
        if state and state["phase"] == "complete":
            store.result(index, state["result"])
            return
        if controller.dispatch.is_set():
            return
        task = tasks[index]
        item = payload["tasks"][index]
        workspace = Path(item["workspace"])
        if state is None:
            if workspace.exists():
                raise AgentError(f"unsafe resume: uncheckpointed workspace exists for {task.task_id}")
            try:
                shutil.copytree(task.repository, workspace, symlinks=False, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
            except Exception as exc:
                timestamp = _now()
                failed = {**item, "start_time": timestamp, "end_time": timestamp, "status": "error", "success": False,
                          "duration_s": 0, "turns": 0, "tool_calls": 0, "tool_failures": 0,
                          "total_tokens": 0, "final_answer": "", "error": str(exc)}
                journal.emit({"run_id": run_id, "task_id": task.task_id, "event": "task_end", "result": failed})
                store.checkpoint(index, {"phase": "complete", "result": failed})
                store.result(index, failed)
                return
        result = _execute(task, workspace, cfg, client, journal, run_id, item["verification_hashes"],
                          checkpoint=lambda state: store.checkpoint(index, state), recovered=state, cancel=cancel, dispatch_stop=controller)
        store.result(index, result)
        if result["status"] == "interrupted":
            controller.request("interrupted", 2 if result.get("error") == "KeyboardInterrupt" else None)
    eligible = set([index for index, state in enumerate(states) if not (state and state["phase"] == "complete")])
    measurement.change_load(pending=len(eligible), reason="queue_initialized")
    def worker(index):
        controller.check()
        if index not in eligible:
            return _worker(index)
        if controller.dispatch.is_set():
            return
        measurement.change_load(pending=-1, active=1, reason="task_started")
        try:
            return _worker(index)
        finally:
            measurement.change_load(active=-1, reason="task_finished")
    with capture_stop_signals(controller):
        executor = ThreadPoolExecutor(max_workers=cfg.coding_agent.max_concurrency)
        try:
            futures = [executor.submit(worker, index) for index in range(len(tasks))]
            pending_futures = set(futures)
            while pending_futures:
                controller.check()
                done, pending_futures = wait(pending_futures, timeout=.01, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        except BaseException as exc:
            cancel.set()
            if not isinstance(exc, (KeyboardInterrupt, SystemExit, InterruptedError)):
                raise
            stop.update(reason="interrupted", signal=2 if isinstance(exc, KeyboardInterrupt) else None, requested_at=_now())
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            controller.check()
            if owns_client:
                client.close()
            attempt_window["end_time"] = _now()
            attempt_window["duration_s"] = time.monotonic() - attempt_started
            payload["metrics"] = aggregate(_events(events_path), sum(window.get("duration_s", 0) for window in payload["attempts"]))
            payload["attempt_metrics"] = {window["attempt_id"]: aggregate([event for event in _events(events_path) if event.get("run_attempt_id") == window["attempt_id"]], window.get("duration_s", 0)) for window in payload["attempts"]}
            if cancel.is_set() and stop["reason"] is None:
                stop.update(reason="interrupted", requested_at=_now())
            attempt_window["stop"] = dict(stop)
            payload.update(status="stopped" if controller.dispatch.is_set() or cancel.is_set() else "completed", stop=dict(stop),
                           end_time=attempt_window["end_time"], lifecycle_version=2)
            events = _events(events_path)
            measured = measure_attempt(attempt_window["measurement_config"], attempt_id, events,
                                       attempt_window["duration_s"], stop.get("requested_offset_s"), stop.get("reason"),
                                       cfg.coding_agent.max_concurrency, "agent")
            if measured is not None:
                attempt_window["measurement"] = measured
            combine_measurements(payload, events)
            store.save()
    return result_path


def replay_agent(cfg: AppConfig, *, client: httpx.Client | None = None) -> Path:
    """Replay recorded requests without new tool or verification execution."""
    if not cfg.coding_agent.replay_result_path:
        raise AgentError("coding_agent.replay_result_path is required")
    if not cfg.server.model:
        raise AgentError("server.model is required")
    source = load_agent_result(cfg.path(cfg.coding_agent.replay_result_path))
    if source["schema_version"] != 2:
        raise AgentError("replay requires version 2 recorded request parameters")
    source_events = _events(Path(source["events_path"]))
    completed = {event["span_id"] for event in source_events
                 if event["event"] == "llm_end" and "output" in event and not event.get("error")}
    requests = {}
    for item in source["tasks"]:
        task_id = item["task"]["task_id"]
        starts = [event for event in source_events if event.get("task_id") == task_id and event["event"] == "llm_start"
                  and event["span_id"] in completed]
        if not starts or any("request" not in event for event in starts):
            raise AgentError(f"replay requires complete recorded requests: {task_id}")
        requests[task_id] = starts
    run_id = uuid.uuid4().hex
    result_dir = cfg.path(cfg.coding_agent.result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / f"agent-{run_id}.json"
    with _run_lock(path):
        events_path = path.with_suffix(".events.jsonl")
        events_path.touch()
        payload = {"schema_version": 2, "lifecycle_version": 2, "kind": "coding_agent", "mode": "replay", "run_id": run_id,
                   "source_run_id": source["run_id"], "model": cfg.server.model, "run_config": _run_config(cfg),
                   "events_path": str(events_path), "tasks": [{"task": item["task"], "trace_id": _typed_id(run_id, "task", item["task"]["task_id"]),
                       "status": "pending", "success": None, "execution_ok": False} for item in source["tasks"]]}
        attempt_id = uuid.uuid4().hex
        attempt_window = {"attempt_id": attempt_id, "start_time": _now(),
                          "measurement_config": cfg.coding_agent.measurement.model_dump(),
                          "measurement_target": cfg.coding_agent.max_concurrency}
        payload["attempts"] = [attempt_window]
        store = RunStore(path, payload)
        store.save()
        attempt_started = time.monotonic()
        owns_client = client is None
        if owns_client:
            headers = {"Authorization": f"Bearer {cfg.server.api_key}"} if cfg.server.api_key else {}
            client = httpx.Client(headers=headers)
        cancel = threading.Event()
        stop = {"reason": None, "signal": None, "requested_at": None, "policy": "cooperative",
                "bounded_http_cancellation": False}
        controller = ThreadStopController(cfg.coding_agent, attempt_started, cancel, stop)
        measurement = MeasurementRecorder(cfg.coding_agent.measurement, attempt_id, attempt_started,
                                          cfg.coding_agent.max_concurrency, "agent", controller.request)
        controller.measurement = measurement
        journal = Journal(events_path, run_id=run_id, run_attempt_id=attempt_id, measurement=measurement)
        def _worker(index: int) -> None:
            if controller.dispatch.is_set():
                return
            item = payload["tasks"][index]
            task_id = item["task"]["task_id"]
            started = time.monotonic()
            result = {**item, "start_time": _now(), "turns": 0, "tool_calls": 0, "tool_failures": 0,
                      "total_tokens": 0, "llm_calls": [], "final_answer": "", "status": "error"}
            def emit(kind, **data):
                journal.emit({"run_id": run_id, "task_id": task_id, "event": kind, **data})
            emit("task_start", trace_id=item["trace_id"], source_run_id=source["run_id"])
            try:
                if len(requests[task_id]) > cfg.coding_agent.max_turns:
                    raise AgentError("replay turn limit exceeded")
                for turn, recorded in enumerate(requests[task_id], 1):
                    controller.check()
                    if controller.dispatch.is_set():
                        raise InterruptedError("evaluation cancelled")
                    remaining = cfg.coding_agent.task_timeout_sec - (time.monotonic() - started)
                    if remaining <= 0:
                        raise AgentError("task timeout exceeded")
                    if result["total_tokens"] >= cfg.coding_agent.token_budget:
                        raise AgentError("token budget exhausted")
                    request = json.loads(json.dumps(recorded["request"]))
                    request["model"] = cfg.server.model
                    request["stream"] = cfg.coding_agent.stream
                    if request["stream"]:
                        request["stream_options"] = {"include_usage": True}
                    else:
                        request.pop("stream_options", None)
                    turn_id = _typed_id(run_id, "replay-turn", task_id, turn)
                    span_id = _typed_id(run_id, "replay-llm", task_id, turn)
                    result["turns"] = turn
                    emit("turn_start", span_id=turn_id, turn=turn)
                    emit("llm_start", span_id=span_id, parent_span_id=turn_id, input={"messages": request["messages"]},
                         request=request, source_span_id=recorded["span_id"], turn=turn)
                    timing = usage = finish = None
                    usage_accounted = False
                    request_started = time.monotonic()
                    try:
                        body, timing = chat_request(client, cfg.server.base_url + cfg.coding_agent.endpoint, request,
                                                   min(remaining, cfg.coding_agent.request_timeout_sec), cancel)
                        usage = body.get("usage") or {}
                        finish = body.get("choices", [{}])[0].get("finish_reason") if body.get("choices") else None
                        tokens = usage.get("total_tokens")
                        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                            raise AgentError("response requires usage.total_tokens for the token budget")
                        result["total_tokens"] += tokens
                        usage_accounted = True
                        message = body["choices"][0]["message"]
                        if message.get("role") != "assistant":
                            raise AgentError("response message must have assistant role")
                        if body["choices"][0].get("finish_reason") not in ("stop", "tool_calls"):
                            raise AgentError("model did not finish normally")
                        emit("llm_end", span_id=span_id, output=message, usage=usage, timing=timing, finish_reason=finish)
                    except (Exception, KeyboardInterrupt, SystemExit) as exc:
                        failed_timing = timing if timing is not None else getattr(exc, "agent_timing", {"e2e_ms": (time.monotonic() - request_started) * 1000, "ttft_ms": None, "tpot_ms": None})
                        failed_usage = usage if usage is not None else getattr(exc, "agent_usage", None)
                        known_tokens = _known_tokens(failed_usage)
                        if not usage_accounted and known_tokens is not None:
                            result["total_tokens"] += known_tokens
                        emit("llm_end", span_id=span_id, error=str(exc) or type(exc).__name__, timing=failed_timing, usage=failed_usage, finish_reason=finish,
                             **({"status": "cancelled"} if isinstance(exc, (KeyboardInterrupt, SystemExit, InterruptedError)) else {}))
                        result["llm_calls"].append({"span_id": span_id, "turn": turn, "success": False, "error": str(exc),
                                                   "usage": failed_usage, "finish_reason": finish, **failed_timing})
                        raise
                    result["llm_calls"].append({"span_id": span_id, "turn": turn, "success": True, "usage": usage, "finish_reason": finish, **timing})
                    result["final_answer"] = message.get("content") or ""
                    emit("turn_end", span_id=turn_id)
                    if time.monotonic() - started > cfg.coding_agent.task_timeout_sec:
                        raise AgentError("task timeout exceeded")
                    if result["total_tokens"] > cfg.coding_agent.token_budget:
                        raise AgentError("token budget exceeded")
                result["status"] = "replayed"
                result["execution_ok"] = True
            except BaseException as exc:
                result["error"] = str(exc) or type(exc).__name__
                result["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit, InterruptedError)) else "error"
                if result["status"] == "interrupted":
                    controller.request("interrupted", 2 if result.get("error") == "KeyboardInterrupt" else None)
            result["end_time"] = _now()
            result["duration_s"] = time.monotonic() - started
            emit("task_end", result=result)
            store.result(index, result)
        eligible = set(list(range(len(payload["tasks"]))))
        measurement.change_load(pending=len(eligible), reason="queue_initialized")
        def worker(index):
            controller.check()
            if index not in eligible:
                return _worker(index)
            if controller.dispatch.is_set():
                return
            measurement.change_load(pending=-1, active=1, reason="task_started")
            try:
                return _worker(index)
            finally:
                measurement.change_load(active=-1, reason="task_finished")
        with capture_stop_signals(controller):
            executor = ThreadPoolExecutor(max_workers=cfg.coding_agent.max_concurrency)
            try:
                futures = [executor.submit(worker, index) for index in range(len(payload["tasks"]))]
                pending_futures = set(futures)
                while pending_futures:
                    controller.check()
                    done, pending_futures = wait(pending_futures, timeout=.01, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
            except BaseException as exc:
                cancel.set()
                if not isinstance(exc, (KeyboardInterrupt, SystemExit, InterruptedError)):
                    raise
                stop.update(reason="interrupted", signal=2 if isinstance(exc, KeyboardInterrupt) else None, requested_at=_now())
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
                controller.check()
                if owns_client:
                    client.close()
                attempt_window["end_time"] = _now()
                attempt_window["duration_s"] = time.monotonic() - attempt_started
                payload["metrics"] = aggregate(_events(events_path), attempt_window["duration_s"])
                payload["attempt_metrics"] = {attempt_id: payload["metrics"]}
                if cancel.is_set() and stop["reason"] is None:
                    stop.update(reason="interrupted", requested_at=_now())
                attempt_window["stop"] = dict(stop)
                payload.update(status="stopped" if controller.dispatch.is_set() or cancel.is_set() else "completed", stop=dict(stop),
                               end_time=attempt_window["end_time"], lifecycle_version=2)
                events = _events(events_path)
                measured = measure_attempt(attempt_window["measurement_config"], attempt_id, events,
                                           attempt_window["duration_s"], stop.get("requested_offset_s"), stop.get("reason"),
                                           cfg.coding_agent.max_concurrency, "agent")
                if measured is not None:
                    attempt_window["measurement"] = measured
                combine_measurements(payload, events)
                store.save()
        return path


def recover_agent(path: Path) -> Path:
    """Settle local events without model, tool, or verification execution."""
    path = path.resolve()
    with _run_lock(path):
        result = json.loads(path.read_text())
        if result.get("kind") != "coding_agent" or result.get("schema_version") not in (1, 2):
            raise AgentError("unsupported coding agent result")
        if not result.get("attempts"):
            raise AgentError("recovery requires persisted attempt windows")
        event_path = path.with_suffix(".events.jsonl")
        if not event_path.is_file():
            event_path = Path(result["events_path"])
        events = read_events(event_path, repair=True)
        identities = {item["task"]["task_id"] for item in result["tasks"]}
        if any(event.get("run_id") != result["run_id"] or (event.get("task_id") not in identities and not (event.get("event") in {"load", "run_stop"} and event.get("task_id") is None)) for event in events):
            raise AgentError("event run or task identity differs from the result")
        starts = {event["span_id"]: event for event in events if event.get("event") == "llm_start"}
        ends = {event["span_id"]: event for event in events if event.get("event") == "llm_end"}
        if len(starts) != sum(event.get("event") == "llm_start" for event in events) or len(ends) != sum(event.get("event") == "llm_end" for event in events):
            raise AgentError("duplicate physical request event")
        for span, end in ends.items():
            if span not in starts or any(end.get(key) != starts[span].get(key) for key in ("task_id", "run_attempt_id")):
                raise AgentError("request end identity differs from its start")
        for span, start in starts.items():
            if span not in ends:
                end = {"run_id": result["run_id"], "run_attempt_id": start.get("run_attempt_id"),
                       "task_id": start["task_id"], "event": "llm_end", "span_id": span,
                       "timestamp": max((event["timestamp"] for event in events if event.get("run_attempt_id") == start.get("run_attempt_id")), default=start["timestamp"]),
                       "status": "cancelled", "error": "request interrupted before recovery", "recovered": True,
                       **({"attempt_offset_s": max(event.get("attempt_offset_s", 0) for event in events if event.get("run_attempt_id") == start.get("run_attempt_id"))}
                          if any("attempt_offset_s" in event for event in events if event.get("run_attempt_id") == start.get("run_attempt_id")) else {}),
                       "usage": {}, "timing": {"tpot_null_reason": "incomplete_request"}}
                Journal(event_path).emit(end)
                events.append(end)
                ends[span] = end
        for index, old in enumerate(result["tasks"]):
            identity = old["task"]["task_id"]
            matching = [event for event in events if event.get("task_id") == identity]
            completed = [event for event in matching if event["event"] == "task_end"]
            if completed:
                item = copy.deepcopy(completed[-1]["result"])
                if item.get("task") != old["task"] or item.get("workspace") != old.get("workspace"):
                    raise AgentError("task end source identity differs from the result")
            elif matching:
                item = {**old, "status": "interrupted", "success": None,
                        "start_time": old.get("start_time", matching[0]["timestamp"]), "end_time": matching[-1]["timestamp"],
                        "duration_s": max([old.get("duration_s", 0)] + [event.get("elapsed_s", 0) for event in matching]),
                        "duration_estimated": True, "duration_estimation": "last_persisted_event_may_underestimate_duration"}
            else:
                item = {**old, "status": "pending", "success": None}
            last_terminal = max((position for position, event in enumerate(matching) if event["event"] == "task_end"), default=-1)
            later_activity = [event for event in matching[last_terminal + 1:] if not event.get("recovered")]
            if later_activity:
                item.update(status="interrupted", success=None,
                            end_time=max(event["timestamp"] for event in matching),
                            duration_estimated=True,
                            duration_estimation="last_persisted_event_may_underestimate_duration")
            if item["status"] == "interrupted":
                item["success"] = None
            if matching:
                item["duration_s"] = max([item.get("duration_s", 0)] + [event.get("elapsed_s", 0) for event in matching])
            calls = []
            for span, start in starts.items():
                if start["task_id"] != identity:
                    continue
                end = ends[span]
                calls.append({"span_id": span, "turn": start.get("turn"), "success": "output" in end and not end.get("error"),
                              "usage": end.get("usage") or {}, "finish_reason": end.get("finish_reason"),
                              **(end.get("timing") or {}), **({"error": end["error"]} if end.get("error") else {})})
            item["llm_calls"] = calls
            item["total_tokens"] = sum(_known_tokens(end.get("usage")) or 0 for span, end in ends.items() if starts[span]["task_id"] == identity)
            result["tasks"][index] = item
        settle_attempts(result, events)
        result["events_path"] = str(event_path.resolve())
        result.update(status="recovered", lifecycle_version=2, end_time=result["attempts"][-1]["end_time"],
                      stop={"reason": "recovery", "signal": None, "requested_at": None, "policy": "offline", "bounded_http_cancellation": False})
        result = normalize_result(result, events)
        recover_measurements(result, events, "agent")
        _atomic_json(path, result)
    return path


def _load_agent_result(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("kind") != "coding_agent" or result.get("schema_version") not in (1, 2):
        raise AgentError("unsupported coding agent result")
    events_path = Path(result["events_path"]) if result.get("events_path") else None
    sibling = path.with_suffix(".events.jsonl")
    if sibling.is_file():
        events_path = sibling
        result["events_path"] = str(sibling.resolve())
    result["_result_path"] = str(path.resolve())
    return normalize_result(result, _events(events_path) if events_path and events_path.exists() else None)


def load_agent_result(path: Path) -> dict:
    with _run_lock(path):
        return _load_agent_result(path)


def sync_agent(result: dict, cfg: AppConfig, *, client=None, experiment_name: str | None = None) -> dict:
    """Send a locked result snapshot and preserve a successful delivery receipt."""
    path_value = result.get("_result_path")
    if not path_value and (result.get("lifecycle_version", 0) < 2 or result.get("schema_version", 0) < 2):
        path_value = str(Path(result["events_path"]).with_suffix("").with_suffix(".json"))
    if not path_value:
        raise AgentError("sync requires a result from load_agent_result")
    path = Path(path_value)
    with _run_lock(path):
        if result.get("lifecycle_version", 0) < 2 or result["schema_version"] < 2:
            with checked_delivery():
                return _sync_agent(result, cfg, client=client, experiment_name=experiment_name)
        current = _load_agent_result(path)
        events = read_events(Path(current["events_path"]))
        platform = cfg.benchmark_platform
        destination = {"url": platform.url, "workspace": platform.workspace, "project": platform.project_name,
                       "dataset": platform.dataset_name, "experiment_name": experiment_name}
        snapshot, receipt_key = snapshot_identity(current, events, destination)
        receipt_path = path.with_suffix(".sync-receipt")
        receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if receipt_key in receipts:
            return {**receipts[receipt_key], "already_synced": True}
        current["_snapshot_id"] = snapshot
        with checked_delivery():
            report = _sync_agent(current, cfg, client=client, experiment_name=experiment_name)
        report.update(snapshot_id=snapshot, delivery_semantics="at_least_once_until_successful_local_receipt")
        receipts[receipt_key] = report
        _atomic_json(receipt_path, receipts)
        return report


def _sync_agent(result: dict, cfg: AppConfig, *, client=None, experiment_name: str | None = None) -> dict:
    """Create task traces and their child spans from persisted events."""
    from opik.api_objects.experiment.experiment_item import ExperimentItemReferences
    platform = cfg.benchmark_platform
    client = client or build_platform_client(platform)
    events_path = Path(result["events_path"])
    events = normalize_events(_events(events_path))
    result = normalize_result(result, events)
    terminal = [task for task in result["tasks"] if task.get("status") not in {"pending", "interrupted"} and task.get("success") is not None]
    quality = {"terminal_tasks": len(terminal), "passed_tasks": sum(bool(task["success"]) for task in terminal)}
    quality["task_success_pct"] = quality["passed_tasks"] / quality["terminal_tasks"] * 100 if quality["terminal_tasks"] else None
    quality_scores = [{"name": metric_names.AGENT_TASK_SUCCESS_SCORE, "value": quality["task_success_pct"]}] if quality["task_success_pct"] is not None else []
    name = experiment_name or f"coding-agent-{result['model']}-{result['run_id']}"
    snapshot = result.get("_snapshot_id")
    snapshot_state = "partial" if any(item.get("status") in {"pending", "interrupted"} for item in result["tasks"]) else "final"
    if snapshot:
        attempt_id = result.get("attempts", [{}])[-1].get("attempt_id", "legacy")
        name += f"-attempt-{attempt_id}-snapshot-{snapshot[:16]}"
    dataset = client.get_or_create_dataset(name=platform.dataset_name, description="Coding agent tasks")
    items = dataset.get_items()
    indexed = {item.get("agent_task_key"): item["id"] for item in items if item.get("agent_task_key")}
    task_keys = {}
    for task_result in result["tasks"]:
        task = task_result["task"]
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(task, sort_keys=True)))
        task_keys[task["task_id"]] = key
        if key not in indexed:
            dataset.insert([{**task, "agent_task_key": key}])
    indexed = {item.get("agent_task_key"): item["id"] for item in dataset.get_items() if item.get("agent_task_key")}
    experiment = client.create_experiment(dataset_name=platform.dataset_name, name=name, experiment_config={
        "run_id": result["run_id"], "mode": result.get("mode", "live"), "model": result["model"],
        "run": result.get("run_config"), "metrics": result.get("metrics"), "metric_definitions": METRIC_DEFINITIONS, "task_quality": quality,
        "stop": result.get("stop"), "snapshot_id": snapshot, "snapshot_state": snapshot_state, "score_scope": "immutable_cumulative_result_snapshot" if snapshot else "legacy"})
    refs = []
    spans = 0
    for task_result in result["tasks"]:
        if "start_time" not in task_result:
            continue
        task_id = task_result["task"]["task_id"]
        source_trace_id = task_result["trace_id"]
        trace_id = platform_id(f"{source_trace_id}:{snapshot}" if snapshot else source_trace_id, task_result["start_time"], "trace")
        task_metrics = task_result["metrics"]
        client.trace(id=trace_id, project_name=platform.project_name, name=f"coding-task:{task_id}",
            start_time=dt.datetime.fromisoformat(task_result["start_time"]), end_time=dt.datetime.fromisoformat(task_result["end_time"]),
            input=task_result["task"], output=task_result, metadata={"run_id": result["run_id"], "task_id": task_id, "mode": result.get("mode", "live"), "effective_required_tools": task_result.get("effective_required_tools", []), "tool_types": task_result.get("tool_types", []),
                      "source_run_id": result.get("source_run_id"), "source_trace_id": source_trace_id,
                      "snapshot_id": snapshot, "snapshot_state": snapshot_state, "stop": result.get("stop"),
                      "metrics": task_metrics, "metric_definitions": METRIC_DEFINITIONS,
                      "performance_score_scope": "aggregated_successful_llm_requests", "throughput_scope": "complete_task_active_window"}, tags=platform.tags,
            feedback_scores=([{ "name": metric_names.AGENT_TASK_SUCCESS_SCORE, "value": int(task_result["success"]) * 100}]
                             if task_result.get("success") is not None and task_result.get("status") not in {"pending", "interrupted"} else []) +
                             [{"name": metric_names.AGENT_TASK_DURATION_SCORE, "value": task_result["duration_s"]}] + task_scores(task_metrics))
        task_events = [event for event in events if event.get("task_id") == task_id]
        starts = {event["span_id"]: event for event in task_events if event["event"].endswith("_start") and "span_id" in event}
        ends = {event["span_id"]: event for event in task_events if event["event"].endswith("_end") and "span_id" in event}
        span_ids = {source_id: platform_id(f"{source_id}:{snapshot}" if snapshot else source_id, start["timestamp"], "span") for source_id, start in starts.items()}
        for span_id, start in starts.items():
            end = ends.get(span_id, {"timestamp": task_result["end_time"], "error": "span interrupted"})
            kind = start["event"].removesuffix("_start")
            kwargs = dict(id=span_ids[span_id], trace_id=trace_id, project_name=platform.project_name,
                name=start.get("name", kind), type="llm" if kind == "llm" else "tool" if kind in ("tool", "verification") else "general",
                start_time=dt.datetime.fromisoformat(start["timestamp"]), end_time=dt.datetime.fromisoformat(end["timestamp"]),
                input=start.get("input"), output=end.get("output"), metadata={"kind": kind, "error": end.get("error"), "timing": end.get("timing"), "metric_definitions": METRIC_DEFINITIONS, "finish_reason": end.get("finish_reason"),
                          "run_id": result["run_id"], "task_id": task_id, "run_attempt_id": start.get("run_attempt_id"),
                          "internal_span_id": span_id, "source_trace_id": source_trace_id, "internal_parent_span_id": start.get("parent_span_id"),
                          "turn": start.get("turn"), "attempt": start.get("attempt"), "source_span_id": start.get("source_span_id"),
                          "status": "cancelled" if end.get("status") == "cancelled" else "error" if end.get("error") or (end.get("output") or {}).get("error") or (end.get("output") or {}).get("timed_out") or (end.get("output") or {}).get("exit_code", 0) != 0 else "completed"})
            if start.get("parent_span_id"):
                if start["parent_span_id"] not in span_ids:
                    raise AgentError("span parent is missing from the recorded event tree")
                kwargs["parent_span_id"] = span_ids[start["parent_span_id"]]
            if kind == "llm":
                kwargs.update(model=result["model"], provider="openai", usage=end.get("usage"),
                              feedback_scores=[] if end.get("status") == "cancelled" else call_scores(end.get("timing") or {}, end.get("usage") or {}, "output" in end and not end.get("error")))
            client.span(**kwargs)
            spans += 1
        refs.append(ExperimentItemReferences(dataset_item_id=indexed[task_keys[task_id]], trace_id=trace_id))
    experiment.insert(refs)
    if result.get("metrics"):
        windows = result.get("attempts") or []
        summary_source_id = _typed_id(result["run_id"], "summary")
        summary_start = windows[0]["start_time"] if windows else min(task["start_time"] for task in result["tasks"] if "start_time" in task)
        client.trace(id=platform_id(f"{summary_source_id}:{snapshot}" if snapshot else summary_source_id, summary_start, "trace"), project_name=platform.project_name,
                     name=metric_names.SUMMARY_TRACE_NAME,
                     start_time=dt.datetime.fromisoformat(windows[0]["start_time"]) if windows else None,
                     end_time=dt.datetime.fromisoformat(windows[-1]["end_time"]) if windows and windows[-1].get("end_time") else None,
                     input={"run": result.get("run_config"), "mode": result.get("mode")},
                     output=metric_names.translate_grouped({key: value for key, value in result["metrics"].items() if key not in {"sample_counts", "scope", "metric_definition_version"}}),
                     metadata={"run_id": result["run_id"], "source_trace_id": summary_source_id, "snapshot_id": snapshot, "snapshot_state": snapshot_state,
                               "stop": result.get("stop"), "score_scope": "immutable_cumulative_result_snapshot" if snapshot else "legacy", "attempts": windows, "attempt_metrics": result.get("attempt_metrics"),
                               "metric_definitions": METRIC_DEFINITIONS, "throughput_scope": "agent_workload_active_windows",
                               "throughput_window": "sum_active_attempt_windows", "sample_counts": result["metrics"].get("sample_counts"), "task_quality": quality}, tags=platform.tags + ["summary"],
                     feedback_scores=summary_scores(result["metrics"]) + quality_scores)
    sync_measurement_trace(result, cfg, client, snapshot)
    checked_flush(client)
    return {"run_id": result["run_id"], "experiment_name": name, "tasks": len(refs), "spans": spans}
