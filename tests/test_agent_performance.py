"""Task-root model metrics, legacy TPOT compatibility, and workload scopes."""
import copy
import json
from pathlib import Path

import pytest

from test_agent_platform import StrictPlatform
from test_coding_agent import agent_cfg, completion, execute
from vllm_bench_eval.agent_metrics import aggregate, normalize_events, normalize_timing
from vllm_bench_eval.coding_agent import load_agent_result, sync_agent


@pytest.mark.parametrize("reason,tokens,expected", [(None, 4, 1000 / 3), (None, 1, None), (None, None, None),
                                                   ("incomplete_request", 4, None), ("nonstream", 4, None)])
def test_normalization_idempotent_and_null_semantics(reason, tokens, expected):
    source = {"stream": reason != "nonstream", "ttft_ms": 100, "e2e_ms": 1100, "decode_ms": 600, "tpot_ms": 200,
              "tpot_null_reason": reason}
    usage = {"completion_tokens": tokens}
    before = copy.deepcopy(source)
    normalized = normalize_timing(source, usage)
    assert normalized["tpot_ms"] == pytest.approx(expected) if expected is not None else normalized["tpot_ms"] is None
    assert normalized["metric_definition_version"] == 2
    assert normalize_timing(normalized, usage) == normalized
    assert source == before
    if expected is not None:
        assert normalized["chunk_tpot_ms"] == 200
    else:
        assert normalized["chunk_tpot_ms"] is None


def test_no_output_and_invalid_usage_do_not_create_tpot():
    for tokens in (None, True, -1, float("nan"), "4"):
        normalized = normalize_timing({"ttft_ms": 10, "e2e_ms": 100, "decode_ms": 20}, {"completion_tokens": tokens})
        assert normalized["tpot_ms"] is None
    assert normalize_timing({"ttft_ms": None, "e2e_ms": 100}, {"completion_tokens": 4})["tpot_ms"] is None


def test_old_result_root_scores_and_experiment_see_current_metrics_without_mutation(agent_cfg):
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    event_path = Path(result["events_path"])
    result_path = event_path.with_suffix("").with_suffix(".json")
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    legacy_timing = {"stream": True, "ttft_ms": 100, "e2e_ms": 1100, "decode_ms": 600, "tpot_ms": 200}
    for event in events:
        if event["event"] == "llm_end":
            event["timing"] = legacy_timing
            event["usage"] = {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
    event_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result["tasks"][0]["llm_calls"][0].update(legacy_timing)
    result["tasks"][0]["llm_calls"][0].pop("metric_definition_version", None)
    result["tasks"][0]["llm_calls"][0]["usage"] = {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
    result["metrics"]["tpot_ms"] = {"mean": 200}
    result_path.write_text(json.dumps(result))
    originals = {path: path.read_bytes() for path in (event_path, result_path)}
    class Client(StrictPlatform):
        def create_experiment(self, **kwargs):
            self.experiment_config = kwargs["experiment_config"]
            return super().create_experiment(**kwargs)
    client = Client()
    sync_agent(result, agent_cfg, client=client)
    root = next(trace for trace in client.traces if trace["name"].startswith("coding-task:"))
    scores = {score["name"]: score["value"] for score in root["feedback_scores"]}
    assert scores["任务成功率(%)"] == 0  # The verifier failed, while inference completed normally.
    assert scores["请求完成率"] == 1
    assert scores["TTFT均值(ms)"] == 100
    assert scores["TPOT均值(ms)"] == pytest.approx(1000 / 3)
    assert scores["TPOT P50(ms)"] == pytest.approx(1000 / 3)
    assert scores["端到端延迟 P99(ms)"] == 1100
    assert scores["单请求输出吞吐均值(tokens/s)"] == pytest.approx(4 / 1.1)
    assert scores["单请求总吞吐 P90(tokens/s)"] == pytest.approx(10 / 1.1)
    assert "任务工作负载输出吞吐(tokens/s)" in scores
    assert client.references[0].trace_id == root["id"]
    assert "usage" not in root
    llm = next(span for span in client.spans if span["type"] == "llm")
    assert llm["metadata"]["timing"]["chunk_tpot_ms"] == 200
    assert llm["metadata"]["timing"]["tpot_ms"] == pytest.approx(1000 / 3)
    assert client.experiment_config["metrics"]["tpot_ms"]["mean"] == pytest.approx(1000 / 3)
    summary = next(trace for trace in client.traces if trace["name"] == "压测汇总")
    assert next(score["value"] for score in summary["feedback_scores"] if score["name"] == "TPOT均值(ms)") == pytest.approx(1000 / 3)
    assert summary["metadata"]["throughput_scope"] == "agent_workload_active_windows"
    assert summary["metadata"]["metric_definitions"]["metric_definition_version"] == 2
    assert "单请求输出吞吐(tokens/s)" in summary["output"]
    assert "单请求总吞吐(tokens/s)" in summary["output"]
    assert "工具与验证" in summary["output"]
    assert "验证总耗时(s)" not in summary["output"]["工具与验证"]
    assert "工具总耗时(s)" not in summary["output"]["工具与验证"]
    assert "已知消耗token总数" in summary["output"]["计数"]
    assert not {"scope", "metric_definition_version", "tools", "request_output_tps", "request_total_tps", "sample_counts"} & summary["output"].keys()
    assert summary["metadata"]["sample_counts"]["tpot_ms"] == 1
    assert not any("COUNT(ms)" in score["name"] for score in summary["feedback_scores"])
    loaded = load_agent_result(result_path)
    assert loaded["metrics"]["tpot_ms"]["mean"] == pytest.approx(1000 / 3)
    assert loaded["tasks"][0]["llm_calls"][0]["tpot_ms"] == pytest.approx(1000 / 3)
    assert all(path.read_bytes() == before for path, before in originals.items())


def test_tool_time_separate_from_successful_inference_stats():
    events = [
        {"event": "llm_start", "span_id": "llm"},
        {"event": "llm_end", "span_id": "llm", "output": {}, "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
         "timing": {"stream": True, "ttft_ms": 100, "e2e_ms": 1100, "decode_ms": 600, "tpot_ms": 200}},
        {"event": "llm_start", "span_id": "failed"},
        {"event": "llm_end", "span_id": "failed", "error": "length", "usage": {"completion_tokens": 4, "total_tokens": 10},
         "timing": {"stream": True, "ttft_ms": 1000, "e2e_ms": 2000, "decode_ms": 500}},
        {"event": "tool_start", "span_id": "tool", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"event": "tool_end", "span_id": "tool", "timestamp": "2026-01-01T00:00:02+00:00", "output": {"exit_code": 1}},
        {"event": "tool_start", "span_id": "unknown", "timestamp": "2026-01-01T00:00:03+00:00"},
        {"event": "verification_start", "span_id": "verify", "timestamp": "2026-01-01T00:00:04+00:00"},
        {"event": "verification_end", "span_id": "verify", "timestamp": "2026-01-01T00:00:05+00:00", "output": {"exit_code": 0}}]
    original = copy.deepcopy(events)
    metrics = aggregate(events, 5)
    assert metrics["tpot_ms"]["mean"] == pytest.approx(1000 / 3)
    assert metrics["counts"]["failed"] == 1 and metrics["sample_counts"]["tpot_ms"] == 1
    assert metrics["request_output_tps"]["mean"] == pytest.approx(4 / 1.1)
    assert metrics["throughput"]["output_tps"] == .8
    assert metrics["tools"]["calls"] == 2 and metrics["tools"]["failed"] == 1
    assert metrics["tools"]["incomplete"] == 1
    assert "duration_s" not in metrics["tools"] and "verification_duration_s" not in metrics["tools"]
    assert metrics["tools"]["verification_calls"] == 1
    normalized = normalize_events(events)
    assert normalized[3]["timing"]["tpot_ms"] == pytest.approx(1000 / 3)
    assert events == original


def test_live_resume_and_replay_share_primary_tpot_definition(agent_cfg):
    import httpx
    from test_agent_advanced import DelayedStream, frame
    from vllm_bench_eval.coding_agent import run_agent
    agent_cfg.coding_agent.stream = True
    count = 0
    def tool_then_interrupt(request):
        nonlocal count
        count += 1
        if count == 2:
            raise KeyboardInterrupt()
        return httpx.Response(200, stream=DelayedStream([
            (.002, frame({"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "read_file", "arguments": '{"path":"value.txt"}'}}]})),
            (0, frame(finish="tool_calls", usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})),
            (.002, "data: [DONE]\n\n")]))
    with httpx.Client(transport=httpx.MockTransport(tool_then_interrupt)) as client:
        path = run_agent(agent_cfg, client=client)
    interrupted = load_agent_result(path)
    agent_cfg.coding_agent.resume_result_path = str(path)
    def final(request):
        return httpx.Response(200, stream=DelayedStream([
            (.002, frame({"content": "Done"})), (.002, frame({"content": "."})),
            (0, frame(finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})),
            (.002, "data: [DONE]\n\n")]))
    with httpx.Client(transport=httpx.MockTransport(final)) as client:
        resumed = load_agent_result(run_agent(agent_cfg, client=client))
    assert resumed["metrics"]["sample_counts"]["tpot_ms"] == 2
    assert resumed["metrics"]["metric_definition_version"] == 2
    assert resumed["tasks"][0]["llm_calls"][0]["tpot_ms"] == interrupted["tasks"][0]["llm_calls"][0]["tpot_ms"]
    agent_cfg.coding_agent.resume_result_path = None
    agent_cfg.coding_agent.replay_result_path = str(path)
    with httpx.Client(transport=httpx.MockTransport(final)) as client:
        replay = load_agent_result(run_agent(agent_cfg, client=client))
    assert replay["tasks"][0]["execution_ok"] and replay["tasks"][0]["success"] is None
    assert replay["metrics"]["sample_counts"]["tpot_ms"] == 2
    for result in (resumed, replay):
        for call in result["tasks"][0]["llm_calls"]:
            if call.get("success"):
                assert call["metric_definition_version"] == 2
                assert call["tpot_ms"] == pytest.approx(call["e2e_ms"] - call["ttft_ms"])
                assert call["tpot_ms"] >= call["chunk_tpot_ms"]
