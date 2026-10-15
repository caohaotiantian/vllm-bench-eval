"""Streaming, fixed request replay, recovery, and parallel task evidence."""
import json
from pathlib import Path
import threading
import time

import httpx
import pytest

from vllm_bench_eval.agent_transport import chat_request
from vllm_bench_eval.coding_agent import AgentError, load_agent_result, run_agent, sync_agent
from test_coding_agent import agent_cfg, call, completion, execute


class DelayedStream(httpx.SyncByteStream):
    def __init__(self, frames):
        self.frames = frames
    def __iter__(self):
        for delay, frame in self.frames:
            time.sleep(delay)
            yield frame.encode()


def frame(delta=None, finish=None, usage=None):
    choices = [] if delta is None and finish is None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]
    return "data: " + json.dumps({"choices": choices, "usage": usage}, ensure_ascii=False) + "\r\n\r\n"


def stream_request(frames):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=DelayedStream(frames)))) as client:
        return chat_request(client, "http://localhost/v1/chat/completions", {"stream": True}, 2)


def test_stream_timing_excludes_role_id_and_late_usage():
    body, timing = stream_request([
        (.01, ": heartbeat\r\n\r\n" + frame({"role": "assistant"})),
        (.01, frame({"tool_calls": [{"index": 0, "id": "call-1", "type": "function"}]})),
        (.02, frame({"tool_calls": [{"index": 0, "function": {"name": "read_", "arguments": "{"}}]})),
        (.02, frame({"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": '"path":"value.txt"}'}}]})),
        (0, frame(finish="tool_calls")),
        (.03, frame(usage={"total_tokens": 10, "completion_tokens": 3})),
        (0, "data: [DONE]\r\n\r\n")])
    message = body["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"path": "value.txt"}
    assert timing["ttft_ms"] >= 35
    assert timing["decode_ms"] >= 15
    assert timing["e2e_ms"] - timing["ttft_ms"] - timing["decode_ms"] >= 25
    assert timing["tpot_ms"] == pytest.approx((timing["e2e_ms"] - timing["ttft_ms"]) / 2)
    assert timing["chunk_tpot_ms"] == pytest.approx(timing["decode_ms"] / 2)
    assert timing["tpot_ms"] > timing["chunk_tpot_ms"]
    assert body["usage"]["total_tokens"] == 10


@pytest.mark.parametrize("completion_tokens,expected", [(None, "missing_completion_tokens"), (0, "fewer_than_two_tokens"), (1, "fewer_than_two_tokens")])
def test_tpot_null_semantics(completion_tokens, expected):
    usage = {"total_tokens": 3}
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    body, timing = stream_request([(0, frame({"content": "答案"})), (0, frame(finish="stop")),
                                   (0, frame(usage=usage)), (0, "data: [DONE]\n\n")])
    assert body["choices"][0]["message"]["content"] == "答案"
    assert timing["ttft_ms"] is not None and timing["tpot_ms"] is None
    assert timing["tpot_null_reason"] == expected


def test_no_output_timing_is_null():
    _, timing = stream_request([(0, frame(finish="stop")), (0, frame(usage={"total_tokens": 3, "completion_tokens": 2})), (0, "data: [DONE]\n\n")])
    assert timing["ttft_ms"] is None and timing["tpot_ms"] is None
    assert timing["ttft_null_reason"] == timing["tpot_null_reason"] == "no_output"


@pytest.mark.parametrize("tail", ["", "data: [DONE]\n\n", "data: bad-json\n\n", 'data: {"error":{"message":"failed"}}\n\n'])
def test_incomplete_stream_rejected(tail):
    with pytest.raises(ValueError):
        stream_request([(0, frame({"content": "partial"}) + tail)])


def test_stream_live_executes_only_complete_tool_arguments(agent_cfg):
    agent_cfg.coding_agent.stream = True
    frames = [(0, frame({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "write_file", "arguments": '{"path":"value.txt","content":"new"}'}}]})),
              (0, frame(finish="tool_calls")), (0, frame(usage={"total_tokens": 5, "completion_tokens": 2})), (0, "data: [DONE]\n\n")]
    final = [(0, frame({"content": "Done"})), (0, frame(finish="stop")), (0, frame(usage={"total_tokens": 5, "completion_tokens": 1})), (0, "data: [DONE]\n\n")]
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, stream=DelayedStream(frames if len(requests) == 1 else final))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = load_agent_result(run_agent(agent_cfg, client=client))
    task = result["tasks"][0]
    assert task["success"] and len(task["llm_calls"]) == 2
    assert task["llm_calls"][0]["ttft_ms"] is not None
    assert task["llm_calls"][1]["tpot_ms"] is None
    assert requests[1]["messages"][-1]["role"] == "tool"


def test_replay_preserves_recorded_history_and_never_executes_tools(agent_cfg, monkeypatch):
    result, original_requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("write_file", {"path": "value.txt", "content": "new"})]}, "tool_calls"),
                                                  completion({"role": "assistant", "content": "Done"})])
    source_path = Path(result["events_path"]).with_suffix("").with_suffix(".json")
    agent_cfg.coding_agent.replay_result_path = str(source_path)
    agent_cfg.coding_agent.allow_commands = False
    agent_cfg.server.model = "other-model"
    import vllm_bench_eval.coding_agent as module
    monkeypatch.setattr(module, "_tool", lambda *args, **kwargs: pytest.fail("replay executed a tool"))
    monkeypatch.setattr(module, "_command", lambda *args, **kwargs: pytest.fail("replay executed verification"))
    replay, requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("run_command", {"argv": ["unsafe"]})]}, "tool_calls"),
                                         completion({"role": "assistant", "content": "Different answer"})])
    assert requests[1]["messages"] == original_requests[1]["messages"]
    assert requests[0]["model"] == "other-model"
    assert replay["run_id"] != result["run_id"] and replay["mode"] == "replay"
    assert replay["tasks"][0]["status"] == "replayed" and replay["tasks"][0]["success"] is None
    assert replay["tasks"][0]["execution_ok"]
    assert not any(event["event"].startswith("tool_") for event in map(json.loads, Path(replay["events_path"]).read_text().splitlines()))


def result_path(result):
    return Path(result["events_path"]).with_suffix("").with_suffix(".json")


def test_resume_turn_boundary_preserves_history_and_workspace(agent_cfg):
    first, requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("write_file", {"path": "value.txt", "content": "new"})]}, "tool_calls"), KeyboardInterrupt()])
    assert first["tasks"][0]["status"] == "interrupted"
    agent_cfg.coding_agent.resume_result_path = str(result_path(first))
    resumed, next_requests = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert resumed["tasks"][0]["success"]
    assert resumed["tasks"][0]["workspace"] == first["tasks"][0]["workspace"]
    assert next_requests[0]["messages"] == requests[1]["messages"]
    assert resumed["tasks"][0]["total_tokens"] == 10
    assert len(resumed["tasks"][0]["llm_calls"]) == 2
    events = list(map(json.loads, Path(resumed["events_path"]).read_text().splitlines()))
    assert len([event for event in events if event["event"] == "tool_start"]) == 1
    llm_ids = [event["span_id"] for event in events if event["event"] == "llm_start"]
    assert len(llm_ids) == len(set(llm_ids))
    assert resumed["tasks"][0]["duration_s"] >= first["tasks"][0]["duration_s"]


def test_resume_completed_tasks_skips_model_calls(agent_cfg):
    source, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    resumed, requests = execute(agent_cfg, [])
    assert requests == [] and resumed["tasks"] == source["tasks"]


def test_resume_refuses_uncheckpointed_side_effect(agent_cfg):
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    events_path = Path(source["events_path"])
    with events_path.open("a") as stream:
        stream.write(json.dumps({"task_id": "fix", "event": "tool_start", "span_id": "uncertain"}) + "\n")
    before = events_path.read_bytes()
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    with pytest.raises(AgentError, match="unsafe resume"):
        execute(agent_cfg, [])
    assert events_path.read_bytes() == before


@pytest.mark.parametrize("change", ["config", "missing_workspace", "task_data", "hash"])
def test_resume_identity_errors(agent_cfg, change):
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    if change == "config":
        agent_cfg.coding_agent.token_budget += 1
    elif change == "missing_workspace":
        import shutil
        shutil.rmtree(source["tasks"][0]["workspace"])
    elif change == "task_data":
        path = agent_cfg.root_dir / "tasks.jsonl"
        task = json.loads(path.read_text())
        task["prompt"] = "different"
        path.write_text(json.dumps(task) + "\n")
    else:
        checkpoint = Path(source["checkpoints"][0])
        state = json.loads(checkpoint.read_text())
        state["result"]["verification_hashes"] = {"different": "hash"}
        checkpoint.write_text(json.dumps(state))
    with pytest.raises(AgentError):
        execute(agent_cfg, [])


def test_concurrent_tasks_overlap_and_preserve_each_result(agent_cfg):
    path = agent_cfg.root_dir / "tasks.jsonl"
    task = json.loads(path.read_text())
    task["task_id"] = "second"
    task["prompt"] = "Second task"
    path.write_text(path.read_text() + json.dumps(task) + "\n")
    agent_cfg.coding_agent.max_concurrency = 2
    barrier = threading.Barrier(2)
    active = set()
    lock = threading.Lock()
    def handler(request):
        body = json.loads(request.content)
        with lock:
            active.add(body["messages"][-1]["content"])
        barrier.wait(timeout=2)
        if body["messages"][-1]["content"] == "Second task":
            return httpx.Response(503)
        return completion({"role": "assistant", "content": "Done"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = load_agent_result(run_agent(agent_cfg, client=client))
    assert len(active) == 2 and len(result["tasks"]) == 2
    assert {task["status"] for task in result["tasks"]} == {"error", "verification_failed"}
    assert all(Path(path).exists() for path in result["checkpoints"])
    assert len([event for event in map(json.loads, Path(result["events_path"]).read_text().splitlines()) if event["event"] == "task_end"]) == 2


def test_v1_still_readable_but_not_resumable_or_replayable(agent_cfg, tmp_path):
    source, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    source["schema_version"] = 1
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(source))
    assert load_agent_result(path)["schema_version"] == 1
    agent_cfg.coding_agent.replay_result_path = str(path)
    with pytest.raises(AgentError, match="version 2"):
        execute(agent_cfg, [])
    agent_cfg.coding_agent.replay_result_path = None
    agent_cfg.coding_agent.resume_result_path = str(path)
    with pytest.raises(AgentError, match="version 2"):
        execute(agent_cfg, [])


def test_run_metrics_use_concurrent_window_and_exclude_null(agent_cfg):
    from vllm_bench_eval.agent_metrics import aggregate
    metrics = aggregate([
        {"event": "llm_start"}, {"event": "llm_start"}, {"event": "llm_start"},
        {"event": "llm_end", "output": {}, "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
         "timing": {"ttft_ms": 10, "tpot_ms": None, "e2e_ms": 100}},
        {"event": "llm_end", "output": {}, "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
         "timing": {"ttft_ms": None, "tpot_ms": 20, "e2e_ms": 200}},
        {"event": "llm_end", "error": "failed", "timing": {"ttft_ms": 500}}], .5)
    assert metrics["counts"]["completed"] == 2 and metrics["counts"]["failed"] == 1
    assert metrics["ttft_ms"]["mean"] == 10 and metrics["tpot_ms"]["mean"] == 90
    assert metrics["throughput"]["output_tps"] == 8
    assert metrics["throughput"]["total_tps"] == 20


def test_resume_preserves_token_budget_and_active_attempt_windows(agent_cfg):
    source, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("read_file", {"path": "value.txt"})]}, "tool_calls"), KeyboardInterrupt()])
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    resumed, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert resumed["tasks"][0]["total_tokens"] == 10
    assert len(resumed["attempts"]) == 2
    duration = sum(window["duration_s"] for window in resumed["attempts"])
    assert resumed["metrics"]["counts"]["duration_s"] == duration
    assert resumed["metrics"]["throughput"]["total_tps"] == pytest.approx(10 / duration)
    assert len(resumed["attempt_metrics"]) == 2


def test_resume_counts_usage_from_uncheckpointed_completed_llm(agent_cfg):
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    with Path(source["events_path"]).open("a") as stream:
        stream.write(json.dumps({"task_id": "fix", "event": "llm_end", "span_id": "old",
                                "usage": {"total_tokens": 7}, "elapsed_s": .1}) + "\n")
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    resumed, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert resumed["tasks"][0]["total_tokens"] == 12
    assert resumed["tasks"][0]["duration_s"] >= .1


def test_failed_stream_preserves_partial_timing(agent_cfg):
    agent_cfg.coding_agent.stream = True
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=DelayedStream([
        (.01, frame({"content": "Partial"}))])))) as client:
        result = load_agent_result(run_agent(agent_cfg, client=client))
    task = result["tasks"][0]
    assert not task["success"] and task["llm_calls"][0]["ttft_ms"] is not None
    assert task["llm_calls"][0]["tpot_ms"] is None
    assert result["metrics"]["ttft_ms"] == {}


def test_cancellation_terminates_command(tmp_path):
    import sys
    from vllm_bench_eval.coding_agent import _command
    cancel = threading.Event()
    timer = threading.Timer(.05, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError):
            _command([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, 20, 100, cancel)
    finally:
        timer.join()
    assert time.monotonic() - started < 2


def test_sync_usage_once_and_replay_not_quality_scored(agent_cfg):
    from types import SimpleNamespace
    from vllm_bench_eval import metric_names
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    class Client:
        def __init__(self):
            self.items, self.traces, self.spans, self.experiments = [], [], [], []
        def get_or_create_dataset(self, **kwargs):
            return self
        def get_items(self):
            return self.items
        def insert(self, items):
            self.items.extend({"id": str(index), **item} for index, item in enumerate(items))
        def create_experiment(self, **kwargs):
            self.experiments.append(kwargs)
            return SimpleNamespace(insert=lambda refs: None)
        def trace(self, **kwargs):
            self.traces.append(kwargs)
        def span(self, **kwargs):
            self.spans.append(kwargs)
        def flush(self):
            pass
    client = Client()
    sync_agent(result, agent_cfg, client=client)
    assert all("usage" not in trace for trace in client.traces)
    assert len([span for span in client.spans if span.get("usage")]) == 1
    llm = next(span for span in client.spans if span["type"] == "llm")
    names = {score["name"] for score in llm["feedback_scores"]}
    assert metric_names.request_score("e2e_ms")[0] in names
    assert metric_names.request_score("success")[0] in names
    assert any(trace["name"] == metric_names.SUMMARY_TRACE_NAME for trace in client.traces)
    assert "metrics" in client.experiments[0]["experiment_config"]
    result["mode"] = "replay"
    result["tasks"][0]["success"] = None
    result_path(result).write_text(json.dumps(result))
    client = Client()
    sync_agent(result, agent_cfg, client=client)
    root = next(trace for trace in client.traces if trace["name"].startswith("coding-task:"))
    assert "task_success" not in {score["name"] for score in root["feedback_scores"]}
    failed, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Partial"}, "length")])
    client = Client()
    sync_agent(failed, agent_cfg, client=client)
    failed_llm = next(span for span in client.spans if span["type"] == "llm")
    assert failed_llm["usage"]["total_tokens"] == 5
    assert failed_llm["metadata"]["finish_reason"] == "length"
    assert next(score["value"] for score in failed_llm["feedback_scores"] if score["name"] == metric_names.request_score("success")[0]) == 0


def test_second_resume_process_is_rejected(agent_cfg):
    from vllm_bench_eval.coding_agent import _run_lock
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    with _run_lock(result_path(source)):
        with pytest.raises(AgentError, match="another process"):
            execute(agent_cfg, [])


def test_missing_token_counts_do_not_make_zero_throughput():
    from vllm_bench_eval.agent_metrics import aggregate
    metrics = aggregate([{"event": "llm_start"}, {"event": "llm_end", "output": {}, "usage": {"total_tokens": 5}}], 1)
    assert metrics["counts"]["total_output_tokens"] is None
    assert metrics["throughput"]["output_tps"] is None
    assert metrics["throughput"]["total_tps"] == 5


@pytest.mark.parametrize("mode", ["live", "replay"])
@pytest.mark.parametrize("failure", ["length", "content_filter", "role", "missing_total"])
def test_semantic_failures_keep_known_usage_and_timing(agent_cfg, mode, failure):
    if mode == "replay":
        source, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
        agent_cfg.coding_agent.replay_result_path = str(result_path(source))
    agent_cfg.coding_agent.stream = failure != "role"
    usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    if failure == "missing_total":
        usage.pop("total_tokens")
    finish = failure if failure in ("length", "content_filter") else "stop"
    if failure == "role":
        handler = lambda request: httpx.Response(200, json={"choices": [{"message": {"role": "user", "content": "bad"}, "finish_reason": finish}], "usage": usage})
    else:
        handler = lambda request: httpx.Response(200, stream=DelayedStream([
            (.005, frame({"content": "Part"})), (.005, frame({"content": "two"})),
            (0, frame(finish=finish)), (0, frame(usage=usage)), (0, "data: [DONE]\n\n")]))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = load_agent_result(run_agent(agent_cfg, client=client))
    task = result["tasks"][0]
    assert not task.get("execution_ok") and not task["success"]
    assert task["total_tokens"] == (0 if failure == "missing_total" else 5)
    call_result = task["llm_calls"][0]
    assert not call_result["success"] and call_result["usage"] == usage
    assert call_result["finish_reason"] == finish and call_result["e2e_ms"] > 0
    if failure != "role":
        assert call_result["ttft_ms"] is not None and call_result["tpot_ms"] is not None
    event = next(event for event in map(json.loads, Path(result["events_path"]).read_text().splitlines()) if event["event"] == "llm_end")
    assert event["usage"] == usage and event["finish_reason"] == finish
    assert event["timing"]["e2e_ms"] == call_result["e2e_ms"]
    assert result["metrics"]["counts"]["completed"] == 0
    assert result["metrics"]["ttft_ms"] == {} and result["metrics"]["sample_counts"]["ttft_ms"] == 0


def test_summary_named_task_has_disjoint_trace_id_and_preserves_legacy(agent_cfg):
    from types import SimpleNamespace
    from vllm_bench_eval.coding_agent import _id
    path = agent_cfg.root_dir / "tasks.jsonl"
    task = json.loads(path.read_text())
    task["task_id"] = "summary"
    path.write_text(json.dumps(task) + "\n")
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    class Client:
        def __init__(self):
            self.items, self.traces, self.refs = [], [], []
        def get_or_create_dataset(self, **kwargs):
            return self
        def get_items(self):
            return self.items
        def insert(self, items):
            self.items.extend({"id": str(index), **item} for index, item in enumerate(items))
        def create_experiment(self, **kwargs):
            return SimpleNamespace(insert=lambda refs: self.refs.extend(refs))
        def trace(self, **kwargs):
            self.traces.append(kwargs)
        def span(self, **kwargs):
            pass
        def flush(self):
            pass
    client = Client()
    sync_agent(result, agent_cfg, client=client)
    assert len(client.traces) == 2 and len({trace["id"] for trace in client.traces}) == 2
    result["tasks"][0]["trace_id"] = _id(result["run_id"], "summary")
    result_path(result).write_text(json.dumps(result))
    client = Client()
    sync_agent(result, agent_cfg, client=client)
    assert client.traces[0]["metadata"]["source_trace_id"] == result["tasks"][0]["trace_id"]
    assert len({trace["id"] for trace in client.traces}) == 2
    assert client.refs[0].trace_id == client.traces[0]["id"]


def test_resume_retains_legacy_root_trace_identity(agent_cfg):
    from vllm_bench_eval.coding_agent import _id
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    legacy = _id(source["run_id"], "fix")
    source["tasks"][0]["trace_id"] = legacy
    result_path(source).write_text(json.dumps(source))
    checkpoint = Path(source["checkpoints"][0])
    state = json.loads(checkpoint.read_text())
    state["result"]["trace_id"] = legacy
    checkpoint.write_text(json.dumps(state))
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    resumed, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert resumed["tasks"][0]["trace_id"] == legacy
    resume_event = next(event for event in map(json.loads, Path(resumed["events_path"]).read_text().splitlines()) if event["event"] == "task_resume")
    assert resume_event["trace_id"] == legacy


def test_provider_parameters_live_recorded_and_replayed(agent_cfg):
    agent_cfg.coding_agent.extra_body = {"thinking": {"type": "disabled"}}
    source, requests = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert source["run_config"]["settings"]["extra_body"] == agent_cfg.coding_agent.extra_body
    event = next(event for event in map(json.loads, Path(source["events_path"]).read_text().splitlines()) if event["event"] == "llm_start")
    assert event["request"]["thinking"] == {"type": "disabled"}
    agent_cfg.coding_agent.replay_result_path = str(result_path(source))
    agent_cfg.coding_agent.extra_body = {"thinking": {"type": "enabled"}}
    replay, replay_requests = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert replay_requests[0]["thinking"] == {"type": "disabled"}
    assert replay["tasks"][0]["execution_ok"]


@pytest.mark.parametrize("field", ["model", "messages", "tools", "stream", "stream_options", "max_tokens", "temperature", "n", "api_key"])
def test_provider_parameters_cannot_override_controlled_fields(field):
    from pydantic import ValidationError
    from vllm_bench_eval.config import AppConfig
    with pytest.raises(ValidationError, match="reserved fields"):
        AppConfig(coding_agent={"extra_body": {field: "override"}})


def test_provider_parameters_reject_nonfinite_json():
    from pydantic import ValidationError
    from vllm_bench_eval.config import AppConfig
    with pytest.raises(ValidationError, match="finite JSON"):
        AppConfig(coding_agent={"extra_body": {"provider_setting": float("nan")}})


def test_deepseek_example_paths_and_request_settings():
    from vllm_bench_eval.config import load_config
    path = Path(__file__).parents[1] / "examples/coding-agent/deepseek-public.example.yaml"
    cfg = load_config(path, env={})
    assert cfg.server.base_url == "https://api.deepseek.com"
    assert cfg.coding_agent.endpoint == "/chat/completions"
    assert cfg.coding_agent.extra_body == {"thinking": {"type": "disabled"}}
    assert cfg.benchmark_platform.enabled and cfg.benchmark_platform.url == "http://localhost:5174/api/"
    assert cfg.path(cfg.coding_agent.tasks_path).is_file()
    assert cfg.server.api_key is None


def test_stream_usage_on_final_finish_chunk():
    body, timing = stream_request([(0, frame({"content": "Done"})),
        (0, frame(delta={}, finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})),
        (0, "data: [DONE]\n\n")])
    assert body["choices"][0]["finish_reason"] == "stop" and body["usage"]["total_tokens"] == 5
    assert timing["completion_tokens"] == 2 and timing["ttft_ms"] is not None


def test_provider_reserved_fields_rejected_after_config_mutation(agent_cfg):
    from pydantic import ValidationError
    agent_cfg.coding_agent.extra_body = {"n": 2}
    with pytest.raises(ValidationError, match="reserved fields"):
        execute(agent_cfg, [])
    assert not (agent_cfg.root_dir / "results").exists()


def test_resume_old_v2_without_provider_options(agent_cfg):
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    source["run_config"]["settings"].pop("extra_body")
    result_path(source).write_text(json.dumps(source))
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    resumed, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert resumed["tasks"][0]["status"] == "verification_failed"


def test_provider_option_change_rejected_for_resume(agent_cfg):
    agent_cfg.coding_agent.extra_body = {"thinking": {"type": "disabled"}}
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    agent_cfg.coding_agent.resume_result_path = str(result_path(source))
    agent_cfg.coding_agent.extra_body = {"thinking": {"type": "enabled"}}
    with pytest.raises(AgentError, match="configuration differs"):
        execute(agent_cfg, [])


def test_agent_interrupted_lifecycle_recovery_and_sync_receipt(agent_cfg):
    from vllm_bench_eval.coding_agent import recover_agent
    from test_agent_platform import StrictPlatform
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    path = result_path(source)
    assert source["status"] == "stopped" and source["stop"]["signal"] == 2
    assert source["stop"]["bounded_http_cancellation"] is False
    assert source["metrics"]["counts"]["cancelled"] == 1
    assert source["metrics"]["counts"]["failed"] == 0
    platform = StrictPlatform()
    first = sync_agent(source, agent_cfg, client=platform)
    count = len(platform.traces), len(platform.spans), len(platform.references)
    second = sync_agent(source, agent_cfg, client=platform)
    assert second["already_synced"] is True
    assert count == (len(platform.traces), len(platform.spans), len(platform.references))
    llm = next(span for span in platform.spans if span["type"] == "llm")
    assert llm["feedback_scores"] == [] and llm["metadata"]["status"] == "cancelled"
    events_path = Path(source["events_path"])
    original = events_path.read_bytes() + b'{"event":"partial'
    events_path.write_bytes(original)
    before = {file: file.read_bytes() for file in Path(source["tasks"][0]["workspace"]).rglob("*") if file.is_file()}
    recover_agent(path)
    recovered = load_agent_result(path)
    assert recovered["status"] == "recovered"
    assert recovered["tasks"][0]["status"] == "interrupted"
    assert recovered["tasks"][0]["success"] is None
    assert before == {file: file.read_bytes() for file in before}
    backups = list(path.parent.glob("*.backup-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    agent_cfg.coding_agent.resume_result_path = str(path)
    final, requests = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert len(requests) == 1
    report = sync_agent(final, agent_cfg, client=platform)
    assert report["snapshot_id"] != first["snapshot_id"]
    from vllm_bench_eval.metric_names import SUMMARY_TRACE_NAME, summary_score
    summaries = [trace for trace in platform.traces if trace["name"] == SUMMARY_TRACE_NAME]
    assert len(summaries) == 2
    assert summaries[0]["metadata"]["snapshot_state"] == "partial"
    assert summaries[1]["metadata"]["snapshot_state"] == "final"
    scores = {score["name"]: score["value"] for score in summaries[1]["feedback_scores"]}
    assert scores[summary_score("completed_ratio")[0]] == .5


def test_agent_recover_does_not_relax_tool_checkpoint_protection(agent_cfg):
    from vllm_bench_eval.coding_agent import recover_agent
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    path = result_path(source)
    checkpoint = Path(source["checkpoints"][0])
    original = checkpoint.read_bytes()
    with Path(source["events_path"]).open("a") as stream:
        stream.write(json.dumps({"run_id": source["run_id"], "run_attempt_id": source["attempts"][-1]["attempt_id"],
            "task_id": source["tasks"][0]["task"]["task_id"], "event": "tool_start", "span_id": "unsafe",
            "timestamp": source["attempts"][-1]["end_time"]}) + "\n")
    recover_agent(path)
    assert checkpoint.read_bytes() == original
    agent_cfg.coding_agent.resume_result_path = str(path)
    with pytest.raises(AgentError, match="unsafe resume"):
        execute(agent_cfg, [])


def test_agent_recover_sync_and_resume_share_run_lock(agent_cfg):
    from vllm_bench_eval.coding_agent import _run_lock, recover_agent
    from test_agent_platform import StrictPlatform
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    path = result_path(source)
    agent_cfg.coding_agent.resume_result_path = str(path)
    with _run_lock(path):
        for operation in (lambda: recover_agent(path), lambda: sync_agent(source, agent_cfg, client=StrictPlatform()),
                          lambda: execute(agent_cfg, [])):
            with pytest.raises(AgentError, match="already owns"):
                operation()


@pytest.mark.parametrize("signum", [2, 15])
def test_agent_real_signal_saves_result_and_stops_next_dispatch(agent_cfg, signum):
    import asyncio
    import sys
    async def exercise():
        requested, release = asyncio.Event(), asyncio.Event()
        handlers = set()
        calls = []
        async def handle(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                size = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:"))
                calls.append(json.loads(await reader.readexactly(size)))
                requested.set()
                await release.wait()
                body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "Done"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        agent_cfg.server.base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        task = json.loads(Path(agent_cfg.coding_agent.tasks_path).read_text())
        Path(agent_cfg.coding_agent.tasks_path).write_text(json.dumps(task) + "\n" + json.dumps({**task, "task_id": "queued"}) + "\n")
        config_path = agent_cfg.root_dir / "signal-config.json"
        config_path.write_text(json.dumps(agent_cfg.model_dump(mode="json")))
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", "vllm_bench_eval.cli", "agent-run",
            "--config", str(config_path), "--no-sync", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            await asyncio.wait_for(requested.wait(), 10)
            process.send_signal(signum)
            await asyncio.sleep(.03)
            release.set()
            stdout, stderr = await asyncio.wait_for(process.communicate(), 3)
            assert process.returncode == 128 + signum, (stdout, stderr)
        finally:
            release.set()
            if process.returncode is None:
                process.kill()
                await process.wait()
            server.close()
            await server.wait_closed()
            for task in list(handlers):
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
        assert len(calls) == 1
    asyncio.run(exercise())
    path = next(agent_cfg.path(agent_cfg.coding_agent.result_dir).glob("agent-*.json"))
    result = load_agent_result(path)
    assert result["stop"]["signal"] == signum
    assert result["status"] == "stopped"
    assert [item["status"] for item in result["tasks"]] == ["interrupted", "pending"]
    assert Path(result["tasks"][0]["workspace"], "value.txt").read_text() == "old"


@pytest.mark.parametrize("signum", [2, 15])
def test_agent_cli_preserves_signal_after_sync_error(agent_cfg, monkeypatch, signum):
    from typer.testing import CliRunner
    from vllm_bench_eval import cli, coding_agent
    from vllm_bench_eval.agent_platform import PlatformDeliveryError
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    path = result_path(source)
    source["stop"]["signal"] = signum
    path.write_text(json.dumps(source))
    agent_cfg.benchmark_platform.enabled = True
    monkeypatch.setattr(cli, "_load", lambda config: agent_cfg)
    monkeypatch.setattr(coding_agent, "run_agent", lambda config: path)
    def fail(*args, **kwargs):
        raise PlatformDeliveryError("simulated upload failure")
    monkeypatch.setattr(coding_agent, "sync_agent", fail)
    output = CliRunner().invoke(cli.app, ["agent-run"])
    assert output.exit_code == 128 + signum
    assert "simulated upload failure" in output.stdout
    assert load_agent_result(path)["stop"]["signal"] == signum


def test_agent_recover_replaces_old_task_end_after_new_attempt_activity(agent_cfg):
    from vllm_bench_eval.coding_agent import recover_agent
    from test_agent_platform import StrictPlatform
    first, _ = execute(agent_cfg, [KeyboardInterrupt()])
    path = result_path(first)
    agent_cfg.coding_agent.resume_result_path = str(path)
    current, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    events_path = Path(current["events_path"])
    events = list(map(json.loads, events_path.read_text().splitlines()))
    latest_attempt = current["attempts"][-1]["attempt_id"]
    events = [event for event in events if not (event["event"] == "task_end" and event["run_attempt_id"] == latest_attempt)]
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    current["tasks"] = first["tasks"]
    path.write_text(json.dumps(current))
    recover_agent(path)
    recovered = load_agent_result(path)
    item = recovered["tasks"][0]
    assert item["status"] == "interrupted" and item["success"] is None
    assert item["duration_estimated"] is True
    assert item["duration_s"] > first["tasks"][0]["duration_s"]
    latest_time = max(event["timestamp"] for event in events if event.get("task_id") == item["task"]["task_id"])
    assert item["end_time"] == latest_time
    platform = StrictPlatform()
    sync_agent(recovered, agent_cfg, client=platform)
    root = next(trace for trace in platform.traces if trace["name"].startswith("coding-task:"))
    assert all(span["end_time"] <= root["end_time"] for span in platform.spans)


@pytest.mark.parametrize("policy,grace,completed,cancelled", [
    ("cancel", .1, 0, 1), ("drain", .1, 1, 0), ("drain", .001, 0, 1),
])
def test_agent_duration_stops_dispatch_and_drain_only_keeps_current_http(agent_cfg, policy, grace, completed, cancelled):
    task = json.loads(Path(agent_cfg.coding_agent.tasks_path).read_text())
    Path(agent_cfg.coding_agent.tasks_path).write_text(json.dumps(task) + "\n" + json.dumps({**task, "task_id": "queued"}) + "\n")
    agent_cfg.coding_agent.stream = True
    agent_cfg.coding_agent.duration_sec = .01
    agent_cfg.coding_agent.stop_policy = policy
    agent_cfg.coding_agent.stop_grace_sec = grace
    agent_cfg.coding_agent.measurement.enabled = True
    frames = [(0, frame({"role": "assistant"})), (.04, frame({"content": "Done"}, finish="stop")),
              (0, frame(usage={"total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2})), (0, "data: [DONE]\n\n")]
    result, requests = execute(agent_cfg, [httpx.Response(200, stream=DelayedStream(frames))])
    assert len(requests) == 1
    assert result["stop"]["reason"] == "duration"
    assert result["stop"]["policy"] == policy
    assert result["stop"]["bounded_http_cancellation"] is False
    assert [item["status"] for item in result["tasks"]] == ["interrupted", "pending"]
    assert result["metrics"]["counts"]["completed"] == completed
    assert result["metrics"]["counts"].get("cancelled", 0) == cancelled
    assert result["metrics"]["counts"]["failed"] == 0
    assert not Path(result["tasks"][1]["workspace"]).exists()
    assert Path(result["tasks"][0]["workspace"], "value.txt").read_text() == "old"
    events = list(map(json.loads, Path(result["events_path"]).read_text().splitlines()))
    assert not any(event["event"] in {"tool_start", "verification_start"} for event in events)
    measured = result["measurement"]["attempts"][0]
    assert measured["window"]["duration_s"] == pytest.approx(.01)
    assert measured["selection_counts"]["started_in_window"] == 1
    assert measured["selection_counts"]["included_successful"] == 0
    if cancelled:
        assert measured["selection_counts"]["cancelled_started_in_window"] == 1
    else:
        assert measured["selection_counts"]["cross_boundary_successful_excluded"] == 1


def test_agent_measurement_sync_matches_local_and_cli_duration_success(agent_cfg, monkeypatch):
    from typer.testing import CliRunner
    from vllm_bench_eval import cli, coding_agent
    from test_agent_platform import StrictPlatform
    agent_cfg.coding_agent.measurement.enabled = True
    agent_cfg.coding_agent.measurement.warmup_sec = .02
    agent_cfg.coding_agent.measurement.duration_sec = .01
    agent_cfg.coding_agent.stop_policy = "drain"
    agent_cfg.coding_agent.stop_grace_sec = .1
    def handler(request):
        time.sleep(.04)
        return completion({"role": "assistant", "content": "Done"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path = run_agent(agent_cfg, client=client)
    result = load_agent_result(path)
    assert result["stop"]["reason"] == "measurement_duration"
    assert result["measurement"]["attempts"][0]["window"]["duration_s"] == pytest.approx(.01)
    platform = StrictPlatform()
    sync_agent(result, agent_cfg, client=platform)
    trace = next(trace for trace in platform.traces if trace["name"] == "测量窗口汇总")
    assert trace["output"] == result["measurement"]["metrics"]
    assert trace["metadata"]["measurement"]["attempts"][0]["load_scope"] == "tasks_without_tool_saturation_claim"
    monkeypatch.setattr(cli, "_load", lambda config: agent_cfg)
    monkeypatch.setattr(coding_agent, "run_agent", lambda config: path)
    output = CliRunner().invoke(cli.app, ["agent-run", "--duration-sec", "1", "--stop-policy", "drain", "--stop-grace-sec", ".1", "--no-sync"])
    assert output.exit_code == 0
    assert agent_cfg.coding_agent.duration_sec == 1


@pytest.mark.parametrize("signum", [2, 15])
@pytest.mark.parametrize("held_lock", ["controller", "dispatch", "cancel"])
def test_real_signal_only_records_intent_while_stop_locks_are_held(tmp_path, signum, held_lock):
    import subprocess
    import sys
    evidence = tmp_path / "settled.json"
    script = '''
import json
import os
from pathlib import Path
import sys
import threading
import time
from vllm_bench_eval.config import CodingAgentConfig
from vllm_bench_eval.coding_agent import _atomic_json
from vllm_bench_eval.lifecycle import ThreadStopController, capture_stop_signals
cancel = threading.Event()
stop = {"signal": None}
controller = ThreadStopController(CodingAgentConfig(), time.monotonic(), cancel, stop)
locks = {"controller": controller.lock, "dispatch": controller.dispatch._cond, "cancel": cancel._cond}
with capture_stop_signals(controller):
    with locks[sys.argv[2]]:
        os.kill(os.getpid(), int(sys.argv[1]))
        assert controller.pending_signal == int(sys.argv[1])
        assert stop["signal"] is None
    controller.check()
    assert controller.dispatch.is_set() and cancel.is_set()
    _atomic_json(Path(sys.argv[3]), stop)
print("settled")
'''
    process = subprocess.run([sys.executable, "-c", script, str(signum), held_lock, str(evidence)], capture_output=True, text=True, timeout=3)
    assert process.returncode == 0, process.stderr
    assert "settled" in process.stdout
    assert json.loads(evidence.read_text())["signal"] == signum


def test_agent_measurement_recovery_keeps_persisted_stop_before_drain(agent_cfg):
    from vllm_bench_eval.coding_agent import recover_agent
    agent_cfg.coding_agent.duration_sec = .05
    agent_cfg.coding_agent.stop_policy = "drain"
    agent_cfg.coding_agent.stop_grace_sec = .2
    agent_cfg.coding_agent.measurement.enabled = True
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return completion({"role": "assistant", "tool_calls": [call("read_file", {"path": "value.txt"})]}, "tool_calls")
        time.sleep(.08)
        return completion({"role": "assistant", "content": "Done"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        path = run_agent(agent_cfg, client=client)
    normal = load_agent_result(path)
    measured = normal["attempts"][0]["measurement"]
    assert len(requests) == 2 and len(measured["selected_request_ids"]) == 1
    assert normal["metrics"]["counts"]["completed"] == 2
    damaged = json.loads(path.read_text())
    for field in ("stop", "measurement", "end_time", "duration_s"):
        damaged["attempts"][0].pop(field)
    damaged.pop("stop")
    damaged.pop("end_time")
    damaged.pop("measurement")
    path.write_text(json.dumps(damaged))
    recover_agent(path)
    recovered = load_agent_result(path)["attempts"][0]["measurement"]
    assert recovered["window"]["cutoff_offset_s"] == measured["window"]["cutoff_offset_s"]
    assert recovered["selected_request_ids"] == measured["selected_request_ids"]
    assert recovered["metrics"]["counts"]["total_tokens"] == measured["metrics"]["counts"]["total_tokens"]


def test_direct_resume_recovers_previous_measurement_without_recover_command(agent_cfg):
    agent_cfg.coding_agent.measurement.enabled = True
    source, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [
        call("read_file", {"path": "value.txt"})]}, "tool_calls"), KeyboardInterrupt()])
    path = result_path(source)
    expected = source["attempts"][0]["measurement"]
    assert expected["metrics"]["counts"]["total_tokens"] == 5
    damaged = json.loads(path.read_text())
    for field in ("measurement", "end_time", "duration_s", "stop"):
        damaged["attempts"][0].pop(field)
    damaged.pop("measurement")
    path.write_text(json.dumps(damaged))
    agent_cfg.coding_agent.resume_result_path = str(path)
    agent_cfg.coding_agent.measurement.duration_sec = 1
    resumed, requests = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    assert len(requests) == 1
    windows = resumed["measurement"]["attempts"]
    assert len(windows) == 2
    assert windows[0]["config"]["duration_sec"] is None
    assert windows[1]["config"]["duration_sec"] == 1
    assert windows[0]["selected_request_ids"] == expected["selected_request_ids"]
    assert windows[0]["window"]["duration_estimated"] is True
    assert windows[0]["window"]["stop_boundary_evidence"] == "persisted_stop_or_load_drop"
    assert windows[0]["window"]["duration_s"] == expected["window"]["duration_s"]
    assert [item["metrics"]["counts"]["total_tokens"] for item in windows] == [5, 5]
    metrics = resumed["measurement"]["metrics"]
    assert resumed["metrics"]["counts"]["total_tokens"] == metrics["counts"]["total_tokens"] == 10
    duration = sum(item["window"]["duration_s"] for item in windows)
    assert metrics["throughput"]["total_tps"] == pytest.approx(10 / duration)
