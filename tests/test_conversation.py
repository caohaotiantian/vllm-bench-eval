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


@pytest.mark.parametrize("messages", [[{"from": "gpt", "value": "a"}], [{"from": "human", "value": None}],
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
    from opik.api_objects.dataset.dataset_item import DatasetItem
    class SchemaPlatform(StrictPlatform):
        def insert(self, items):
            for item in items:
                DatasetItem.model_validate(item)
            super().insert(items)
    platform = SchemaPlatform()
    original = copy.deepcopy(result)
    report = sync_conversations(result, cfg, client=platform)
    assert report["conversations"] == 1 and report["spans"] == 2
    assert len(platform.references) == 1 and platform.references[0].trace_id == platform.traces[0]["id"]
    assert len(platform.items) == 1 and platform.items[0]["messages"][2]["content"] == "recorded answer"
    assert "source" not in platform.items[0]
    assert platform.items[0]["dataset_source"] == result["conversations"][0]["conversation"]["source"]
    assert all(span["trace_id"] == platform.traces[0]["id"] for span in platform.spans)
    assert [s["metadata"]["context"]["message_count"] for s in platform.spans] == [2, 4]
    assert platform.traces[0]["metadata"]["quality_evaluation"] == "none"
    from vllm_bench_eval import metric_names
    names = {s["name"] for t in platform.traces for s in t["feedback_scores"]}
    assert metric_names.AGENT_TASK_SUCCESS_SCORE not in names
    assert "conversation_execution_ok" not in names
    assert "conversation_duration_s" not in names
    scores = {score["name"]: score["value"] for score in platform.traces[0]["feedback_scores"]}
    assert scores[metric_names.CONVERSATION_DURATION_SCORE] == result["conversations"][0]["duration_s"]
    assert scores[metric_names.summary_score("completed_ratio")[0]] == 1
    assert platform.traces[0]["output"]["execution_ok"] is True
    assert {s["name"] for s in platform.spans[0]["feedback_scores"]} >= {metric_names.request_score("ttft_ms")[0], metric_names.request_score("tpot_ms")[0]}
    sync_conversations(result, cfg, client=platform)
    assert len(platform.items) == 1 and result == original


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("message,finish,error,reasoning", [
    ({"role": "assistant", "content": None, "reasoning_content": "private reasoning"}, "length", "nonblank assistant text", True),
    ({"role": "assistant", "content": " \n"}, "stop", "nonblank assistant text", False),
    ({"role": "user", "content": "answer"}, "stop", "assistant role", False),
    ({"role": "assistant", "content": "answer", "tool_calls": [{"index": 0, "id": "call", "function": {"name": "tool", "arguments": "{}"}}]}, "tool_calls", "unsupported tool calls", False),
    ({"role": "assistant", "content": "answer", "function_call": {"name": "tool", "arguments": "{}"}}, "function_call", "unsupported tool calls", False),
    ({"role": "assistant", "content": "answer"}, "length", "finish normally", False),
])
def test_response_diagnostics_preserve_failed_usage_without_success_metrics(cfg, stream, message, finish, error, reasoning):
    cfg.conversation.stream = stream
    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    def handler(payload):
        if not stream:
            return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}], "usage": usage})
        chunks = [{"choices": [{"delta": message, "finish_reason": finish}]}, {"choices": [], "usage": usage}]
        return httpx.Response(200, text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n")
    result, payloads, path = execute(cfg, handler)
    assert len(payloads) == 1
    conversation = result["conversations"][0]
    assert conversation["execution_ok"] is False and error in conversation["error"]
    events = [json.loads(line) for line in path.with_suffix(".events.jsonl").read_text().splitlines()]
    end = next(event for event in events if event["event"] == "llm_end")
    assert end["usage"] == usage
    assert end["response_summary"]["finish_reason"] == finish
    assert end["response_summary"]["reasoning_present"] is reasoning
    assert "private reasoning" not in json.dumps(end)
    assert end["timing"]["e2e_ms"] >= 0
    assert conversation["known_consumed_tokens"] == 13
    assert result["metrics"]["counts"]["known_consumed_tokens"] == 13
    assert result["metrics"]["counts"]["completed"] == 0
    assert result["metrics"]["counts"]["failed"] == 1
    assert all(count == 0 for count in result["metrics"]["sample_counts"].values())
    platform = StrictPlatform()
    sync_conversations(result, cfg, client=platform)
    assert platform.spans[0]["usage"] == usage
    assert platform.spans[0]["metadata"]["error"] == end["error"]


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


@pytest.mark.parametrize("policy", ["dataset", "generated"])
def test_blank_source_cleanup_preserves_pairs_provenance_and_history(cfg, policy):
    cfg.conversation.history_policy = policy
    raw = [{"from": "system", "value": " \t"}, {"from": "human", "value": "first valid"},
           {"from": "gpt", "value": "recorded first"}, {"from": "human", "value": ""},
           {"from": "gpt", "value": "removed with blank user"}, {"from": "human", "value": "removed with blank reply"},
           {"from": "gpt", "value": "\n"}, {"from": "human", "value": "last valid"}, {"from": "gpt", "value": " "}]
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps([{"conversations": raw}]))
    result, payloads, _ = execute(cfg)
    assert [len(payload["messages"]) for payload in payloads] == [1, 3]
    assert payloads[1]["messages"][1]["content"] == ("recorded first" if policy == "dataset" else "generated answer")
    source = result["conversations"][0]["conversation"]["source"]
    assert source["original_messages"] == raw and source["message_indices"] == [1, 2, 7]
    assert source["record_index"] == 0 and source["id_generated"] is True
    assert source["removed_messages"] == 6
    assert [item["reason"] for item in source["cleanup"]] == ["blank_system", "blank_user_pair", "blank_assistant_pair", "blank_final_assistant"]
    assert [turn["context"]["source_message_index"] for turn in result["conversations"][0]["turns"]] == [1, 7]
    assert result["source_cleanup"]["raw_user_turns"] == 4
    assert result["source_cleanup"]["eligible_user_turns"] == 2
    platform = StrictPlatform()
    sync_conversations(result, cfg, client=platform)
    assert platform.traces[0]["metadata"]["source_cleanup"] == source
    assert platform.spans[1]["metadata"]["context"]["source_message_index"] == 7


def test_trailing_blank_user_and_fully_blank_conversation_do_not_fail_engine(cfg, monkeypatch):
    records = [{"conversations": [{"from": "human", "value": "valid"}, {"from": "gpt", "value": "recorded"}, {"from": "human", "value": " "}]},
               {"conversations": [{"from": "system", "value": ""}, {"from": "human", "value": " "}, {"from": "gpt", "value": ""}]}]
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps(records))
    result, payloads, _ = execute(cfg)
    assert len(payloads) == 1
    assert [item["status"] for item in result["conversations"]] == ["completed", "skipped"]
    assert result["conversations"][1]["execution_ok"] is None
    assert result["execution"] == {"planned_conversations": 1, "completed_conversations": 1, "planned_turns": 1, "completed_turns": 1}
    assert result["source_cleanup"] == {"dataset_conversations": 2, "selected_conversations": 2, "eligible_conversations": 1,
                                       "skipped_conversations": 1, "raw_user_turns": 3, "eligible_user_turns": 1, "removed_messages": 4}
    assert result["metrics"]["counts"]["failed"] == 0
    platform = StrictPlatform()
    sync_conversations(result, cfg, client=platform)
    skipped = next(trace for trace in platform.traces if trace["metadata"].get("status") == "skipped")
    assert skipped["feedback_scores"] == []
    assert skipped["metadata"]["skip_reason"] == "no_usable_user_turns"
    assert len(platform.spans) == 1
    from vllm_bench_eval import cli, conversation
    monkeypatch.setattr(cli, "_load", lambda config: cfg)
    monkeypatch.setattr(conversation, "run_conversations", lambda config: Path(result["events_path"]).with_suffix("").with_suffix(".json"))
    output = CliRunner().invoke(app, ["conversation-run", "--no-sync"])
    assert output.exit_code == 0 and "数据清理" in output.stdout


def test_all_blank_result_cli_success_and_no_platform_calls(cfg, monkeypatch):
    from vllm_bench_eval import cli, conversation
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps([{"conversations": [{"from": "human", "value": " \n"}]}]))
    result, payloads, path = execute(cfg)
    assert payloads == []
    assert result["execution"] == {"planned_conversations": 0, "completed_conversations": 0, "planned_turns": 0, "completed_turns": 0}
    assert result["metrics"]["counts"]["completed"] == result["metrics"]["counts"]["failed"] == 0
    def forbidden(*args, **kwargs):
        pytest.fail("all-blank input must not call Opik")
    monkeypatch.setattr(conversation, "build_platform_client", forbidden)
    report = sync_conversations(result, cfg)
    assert report["reason"] == "no_usable_user_turns" and report["experiment_name"] is None
    monkeypatch.setattr(cli, "_load", lambda config: cfg)
    monkeypatch.setattr(conversation, "run_conversations", lambda config: path)
    cfg.benchmark_platform.enabled = True
    output = CliRunner().invoke(app, ["conversation-run"])
    assert output.exit_code == 0
    assert "数据清理" in output.stdout and "no_usable_user_turns" in output.stdout
    output = CliRunner().invoke(app, ["conversation-sync", str(path)])
    assert output.exit_code == 0


def test_blank_cleanup_cannot_hide_role_type_or_order_errors(cfg):
    for raw in [[{"from": "gpt", "value": ""}], [{"from": "human", "value": ""}, {"from": "human", "value": "valid"}],
                [{"from": "human", "value": ""}, {"from": "tool", "value": ""}], [{"from": "human", "value": ""}, {"from": "gpt", "value": None}]]:
        cfg.path(cfg.conversation.dataset_path).write_text(json.dumps([{"conversations": raw}]))
        with pytest.raises(ConversationError):
            load_conversations(cfg.path(cfg.conversation.dataset_path))


@pytest.mark.parametrize("messages", [[], [{"from": "system", "value": ""}], [{"from": "system", "value": "nonempty system"}]])
def test_empty_array_and_system_only_conversations_are_skipped(cfg, messages):
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps([{"conversations": messages}]))
    result, payloads, _ = execute(cfg)
    assert not payloads
    assert result["conversations"][0]["status"] == "skipped"
    assert result["source_cleanup"]["skipped_conversations"] == 1
    assert result["execution"]["planned_conversations"] == 0


@pytest.mark.parametrize("records", [[], [{"conversations": None}], [{}], [{"conversations": {}}]])
def test_invalid_dataset_structure_remains_rejected(cfg, records):
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps(records))
    with pytest.raises(ConversationError):
        load_conversations(cfg.path(cfg.conversation.dataset_path))
