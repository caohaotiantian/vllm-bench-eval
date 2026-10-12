"""Evidence for ShareGPT history, shared metrics, limits, and Opik delivery."""
import copy
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from test_agent_platform import StrictPlatform
from vllm_bench_eval.agent_platform import PlatformDeliveryError
from vllm_bench_eval.cli import app
from vllm_bench_eval.config import AppConfig, load_config
from vllm_bench_eval.conversation import ConversationError, load_conversations, load_conversation_result, run_conversations, sync_conversations


@pytest.fixture
def cfg(tmp_path):
    records = [{"id": "one", "conversations": [{"from": "system", "value": "system"}, {"from": "human", "value": "first"},
                {"from": "gpt", "value": "recorded answer"}, {"from": "human", "value": "second"}, {"from": "gpt", "value": "recorded final"}]}]
    (tmp_path / "sharegpt.json").write_text(json.dumps(records))
    return AppConfig(root_dir=tmp_path, server={"model": "test"}, conversation={"dataset_path": "sharegpt.json", "result_dir": "results"}, benchmark_platform={"enabled": False})


def response(text="generated answer", usage=None, finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13} if usage is None else usage}


def execute(cfg, handler=None):
    payloads = []
    def dispatch(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if handler:
            return handler(payload)
        body = response()
        if payload["stream"]:
            chunks = [{"choices": [{"delta": {"content": "generated "}}]}, {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
                      {"choices": [], "usage": body["usage"]}]
            return httpx.Response(200, text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n")
        return httpx.Response(200, json=body)
    with httpx.Client(transport=httpx.MockTransport(dispatch)) as client:
        path = run_conversations(cfg, client=client)
    return load_conversation_result(path), payloads, path


@pytest.mark.parametrize("policy,answer", [("dataset", "recorded answer"), ("generated", "generated answer")])
def test_history_growth_shared_streaming_metrics_and_persisted_context(cfg, policy, answer):
    cfg.conversation.history_policy = policy
    result, payloads, path = execute(cfg)
    assert [len(p["messages"]) for p in payloads] == [2, 4]
    assert payloads[1]["messages"][2] == {"role": "assistant", "content": answer}
    assert all(p["stream_options"] == {"include_usage": True} and "tools" not in p for p in payloads)
    turns = result["conversations"][0]["turns"]
    assert [t["context"]["message_count"] for t in turns] == [2, 4]
    assert turns[1]["context"]["context_chars"] > turns[0]["context"]["context_chars"]
    assert turns[1]["context"]["prompt_tokens"] == 10
    timing = turns[0]["timing"]
    assert timing["tpot_ms"] == pytest.approx((timing["e2e_ms"] - timing["ttft_ms"]) / 2)
    assert len(timing["chunk_itl_ms"]) == 1
    assert result["metrics"]["counts"]["completed"] == 2
    assert result["metrics"]["sample_counts"]["ttft_ms"] == 2
    assert result["execution"] == {"planned_conversations": 1, "completed_conversations": 1, "planned_turns": 2, "completed_turns": 2}
    moved = path.parent / "moved.json"
    moved.write_bytes(path.read_bytes())
    moved.with_suffix(".events.jsonl").write_bytes(Path(result["events_path"]).read_bytes())
    assert load_conversation_result(moved)["events_path"] == str(moved.with_suffix(".events.jsonl"))


@pytest.mark.parametrize("messages", [[], [{"from": "gpt", "value": "a"}], [{"from": "human", "value": ""}],
    [{"from": "human", "value": "a"}, {"from": "human", "value": "b"}], [{"from": "human", "value": "a"}, {"from": "system", "value": "b"}],
    [{"from": "tool", "value": "x"}], [{"from": "human", "value": "a", "tool_calls": []}], [{"from": [], "value": "a"}]])
def test_invalid_messages_fail_before_requests(tmp_path, messages):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"id": "x", "conversations": messages}]))
    with pytest.raises(ConversationError):
        load_conversations(path)


def test_missing_ids_aliases_final_user_and_duplicate_ids(tmp_path):
    path = tmp_path / "data.json"
    record = {"conversations": [{"from": "user", "value": "a"}, {"from": "assistant", "value": "b"}, {"from": "user", "value": "c"}]}
    path.write_text(json.dumps([record]))
    assert load_conversations(path)[0]["conversation_id"] == "conversation-0"
    path.write_text(json.dumps([{**record, "id": 1}, {**record, "id": "1"}]))
    with pytest.raises(ConversationError, match="duplicate"):
        load_conversations(path)


@pytest.mark.parametrize("field,value,requests,error", [("max_turns", 1, 0, "max_turns"), ("max_context_chars", 12, 1, "max_context_chars"),
                                                        ("token_budget", 13, 1, "token_budget")])
def test_limits_fail_explicitly_without_truncation(cfg, field, value, requests, error):
    setattr(cfg.conversation, field, value)
    result, payloads, _ = execute(cfg)
    assert len(payloads) == requests
    assert result["conversations"][0]["execution_ok"] is False
    assert error in result["conversations"][0]["error"]
    assert result["execution"]["planned_turns"] == 2
    assert result["metrics"]["counts"]["completed"] == requests


@pytest.mark.parametrize("body,error", [(response(finish="length"), "finish normally"), (response(text=""), "assistant text"),
                                        (response(usage={}), "total_tokens"), (response(usage={"total_tokens": -1}), "nonnegative")])
def test_response_failures_and_missing_budget_usage(cfg, body, error):
    cfg.conversation.stream = False
    result, payloads, _ = execute(cfg, lambda payload: httpx.Response(200, json=body))
    assert len(payloads) == 1 and not result["conversations"][0]["execution_ok"]
    assert error in result["conversations"][0]["error"]
    assert result["metrics"]["counts"]["failed"] == (0 if body["usage"] == {} else 1)


def test_http_and_incomplete_stream_failures(cfg):
    for response_obj in [httpx.Response(400, json={"error": "context exceeded"}), httpx.Response(200, text='data: {"choices": [{"delta": {"content": "partial"}}]}\n\n')]:
        result, payloads, _ = execute(cfg, lambda payload: response_obj)
        assert len(payloads) == 1
        assert result["metrics"]["counts"]["failed"] == 1
        assert result["metrics"]["sample_counts"]["tpot_ms"] == 0


def test_nonstream_and_timeout(cfg):
    cfg.conversation.stream = False
    result, _, _ = execute(cfg)
    assert result["metrics"]["sample_counts"]["ttft_ms"] == 0
    cfg.conversation.conversation_timeout_sec = 0.001
    def delayed(payload):
        time.sleep(0.01)
        return httpx.Response(200, json=response())
    result, _, _ = execute(cfg, delayed)
    assert result["metrics"]["counts"]["failed"] == 1
    assert "timeout" in result["conversations"][0]["error"]


def test_independent_conversations_run_concurrently(cfg):
    records = json.loads(cfg.path(cfg.conversation.dataset_path).read_text())
    records.append({**copy.deepcopy(records[0]), "id": "two"})
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps(records))
    cfg.conversation.max_concurrency = 2
    cfg.conversation.stream = False
    barrier = threading.Barrier(2)
    def concurrent(payload):
        barrier.wait(timeout=2)
        return httpx.Response(200, json=response())
    result, payloads, _ = execute(cfg, concurrent)
    assert len(payloads) == 4
    assert result["execution"]["completed_conversations"] == 2
    assert all(c["metrics"]["counts"]["completed"] == 2 for c in result["conversations"])


def test_strict_opik_datasets_references_ids_scores_and_context(cfg):
    result, _, _ = execute(cfg)
    platform = StrictPlatform()
    original = copy.deepcopy(result)
    report = sync_conversations(result, cfg, client=platform)
    assert report["conversations"] == 1 and report["spans"] == 2
    assert len(platform.references) == 1 and platform.references[0].trace_id == platform.traces[0]["id"]
    assert len(platform.items) == 1 and platform.items[0]["messages"][2]["content"] == "recorded answer"
    assert all(span["trace_id"] == platform.traces[0]["id"] for span in platform.spans)
    assert [s["metadata"]["context"]["message_count"] for s in platform.spans] == [2, 4]
    assert platform.traces[0]["metadata"]["quality_evaluation"] == "none"
    from vllm_bench_eval import metric_names
    names = {s["name"] for t in platform.traces for s in t["feedback_scores"]}
    assert metric_names.AGENT_TASK_SUCCESS_SCORE not in names
    assert {s["name"] for s in platform.spans[0]["feedback_scores"]} >= {metric_names.request_score("ttft_ms")[0], metric_names.request_score("tpot_ms")[0]}
    sync_conversations(result, cfg, client=platform)
    assert len(platform.items) == 1 and result == original


@pytest.mark.parametrize("failure", ["reject", "timeout", "connection"])
def test_delivery_failure_propagates(cfg, failure):
    result, _, _ = execute(cfg)
    with pytest.raises(PlatformDeliveryError):
        sync_conversations(result, cfg, client=StrictPlatform(failure))


def test_cli_run_sync_failures_and_config_validation(cfg, monkeypatch):
    from vllm_bench_eval import cli, conversation
    result, _, path = execute(cfg)
    monkeypatch.setattr(cli, "_load", lambda config: cfg)
    monkeypatch.setattr(conversation, "run_conversations", lambda config: path)
    cfg.benchmark_platform.enabled = True
    platform = StrictPlatform()
    monkeypatch.setattr(conversation, "build_platform_client", lambda platform_settings: platform)
    output = CliRunner().invoke(app, ["conversation-run"])
    assert output.exit_code == 0 and "conversation result" in output.stdout
    assert len(platform.spans) == 2
    output = CliRunner().invoke(app, ["conversation-sync", str(path)])
    assert output.exit_code == 0
    monkeypatch.setattr(conversation, "build_platform_client", lambda platform: StrictPlatform("reject"))
    output = CliRunner().invoke(app, ["conversation-sync", str(path)])
    assert output.exit_code == 1 and "background delivery" in output.stdout
    config = load_config(env={"VBE_CONVERSATION__HISTORY_POLICY": "generated", "VBE_CONVERSATION__STREAM": "false"})
    assert config.conversation.history_policy == "generated" and config.conversation.stream is False
    with pytest.raises(ValueError):
        AppConfig(conversation={"history_policy": "bad"})
    with pytest.raises(ValueError, match=r"conversation\.extra_body contains reserved fields"):
        AppConfig(conversation={"extra_body": {"messages": []}})
    with pytest.raises(ValueError, match=r"conversation\.extra_body must contain finite JSON values"):
        AppConfig(conversation={"extra_body": {"custom": float("nan")}})
