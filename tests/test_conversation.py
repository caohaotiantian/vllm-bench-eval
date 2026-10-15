"""Evidence for ShareGPT history, shared metrics, limits, and Opik delivery."""
import asyncio
import copy
import json
import time
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from test_agent_platform import StrictPlatform
from vllm_bench_eval.agent_platform import PlatformDeliveryError
from vllm_bench_eval.cli import app
from vllm_bench_eval.config import AppConfig, load_config
from vllm_bench_eval.conversation import async_run_conversations, ConversationError, load_conversations, load_conversation_result, run_conversations, sync_conversations


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
    async def dispatch(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        if handler:
            result = handler(payload)
            return await result if asyncio.iscoroutine(result) else result
        body = response()
        if payload["stream"]:
            chunks = [{"choices": [{"delta": {"content": "generated "}}]}, {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
                      {"choices": [], "usage": body["usage"]}]
            return httpx.Response(200, text="".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n")
        return httpx.Response(200, json=body)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as client:
            return await async_run_conversations(cfg, client=client)
    path = asyncio.run(run())
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
    assert result["execution"] == {"planned_conversations": 1, "completed_conversations": 1, "planned_turns": 2, "completed_turns": 2,
                                  "failed_conversations": 0, "interrupted_conversations": 0,
                                  "not_started_conversations": 0, "cancelled_turns": 0, "not_started_turns": 0}
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
    arrived = 0
    barrier = asyncio.Event()
    async def concurrent(payload):
        nonlocal arrived, barrier
        generation = barrier
        arrived += 1
        if arrived % 2 == 0:
            barrier = asyncio.Event()
            generation.set()
        await asyncio.wait_for(generation.wait(), 2)
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
    output = CliRunner().invoke(app, ["conversation-sync", str(path), "--experiment-name", "failure-retry"])
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
    assert result["execution"] == {"planned_conversations": 1, "completed_conversations": 1, "planned_turns": 1, "completed_turns": 1,
                                  "failed_conversations": 0, "interrupted_conversations": 0,
                                  "not_started_conversations": 0, "cancelled_turns": 0, "not_started_turns": 0}
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
    assert result["execution"] == {"planned_conversations": 0, "completed_conversations": 0, "planned_turns": 0, "completed_turns": 0,
                                  "failed_conversations": 0, "interrupted_conversations": 0,
                                  "not_started_conversations": 0, "cancelled_turns": 0, "not_started_turns": 0}
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


@pytest.mark.parametrize("policy,delay,completed,cancelled", [
    ("cancel", .2, 0, 1), ("drain", .04, 1, 0), ("drain", .2, 0, 1),
])
def test_duration_stop_preserves_partial_results_and_sync(cfg, policy, delay, completed, cancelled):
    records = json.loads(cfg.path(cfg.conversation.dataset_path).read_text())
    records.append({**copy.deepcopy(records[0]), "id": "pending"})
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps(records))
    cfg.conversation.stream = False
    cfg.conversation.duration_sec = .02
    cfg.conversation.stop_policy = policy
    cfg.conversation.stop_grace_sec = .06
    async def delayed(payload):
        await asyncio.sleep(delay)
        return httpx.Response(200, json=response())
    started = time.monotonic()
    result, payloads, path = execute(cfg, delayed)
    assert time.monotonic() - started < .5
    assert len(payloads) == 1
    assert result["status"] == "stopped" and result["stop"]["reason"] == "duration"
    assert result["execution"]["completed_turns"] == completed
    assert result["execution"]["cancelled_turns"] == cancelled
    assert result["execution"]["not_started_turns"] == 3
    assert result["execution"]["not_started_conversations"] == 1
    assert [item["status"] for item in result["conversations"]] == ["interrupted", "not_started"]
    assert result["metrics"]["counts"]["failed"] == 0
    platform = StrictPlatform()
    report = sync_conversations(result, cfg, client=platform)
    assert report["spans"] == 1
    from vllm_bench_eval.metric_names import SUMMARY_TRACE_NAME
    summary = next(trace for trace in platform.traces if trace["name"] == SUMMARY_TRACE_NAME)
    assert summary["metadata"]["stop"] == result["stop"]
    pending = next(trace for trace in platform.traces if trace["metadata"].get("status") == "not_started")
    assert pending["feedback_scores"] == []
    assert pending["output"]["turns"] == []
    assert len(result["conversations"][0]["turns"]) == completed
    if cancelled:
        assert platform.spans[0]["metadata"]["status"] == "cancelled"
        assert platform.spans[0]["feedback_scores"] == []
    snapshot = path.read_bytes(), path.with_suffix(".events.jsonl").read_bytes()
    time.sleep(.03)
    assert snapshot == (path.read_bytes(), path.with_suffix(".events.jsonl").read_bytes())


@pytest.mark.parametrize("signum", [2, 15])
@pytest.mark.parametrize("stall", ["headers", "sse", "body"])
def test_real_signal_cancels_blocked_http_and_saves_completed_turns(cfg, signum, stall):
    import signal
    import sys
    records = json.loads(cfg.path(cfg.conversation.dataset_path).read_text())
    records[0]["id"] = "fast"
    slow = copy.deepcopy(records[0])
    slow["id"] = "slow"
    slow["conversations"][1]["value"] = "slow"
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps([records[0], slow]))
    cfg.conversation.max_concurrency = 2
    cfg.conversation.stream = stall == "sse"
    async def exercise():
        slow_started, fast_finished, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
        handlers = set()
        requests = []
        async def handle(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            try:
                headers = await reader.readuntil(b"\r\n\r\n")
                length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n") if line.lower().startswith(b"content-length:"))
                payload = json.loads(await reader.readexactly(length))
                requests.append(payload)
                if payload["messages"][-1]["content"] != "slow":
                    body = response()
                    if payload["stream"]:
                        content = ('data: ' + json.dumps({"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]})
                                   + '\n\ndata: ' + json.dumps({"choices": [], "usage": body["usage"]}) + '\n\ndata: [DONE]\n\n').encode()
                    else:
                        content = json.dumps(body).encode()
                    writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: " + str(len(content)).encode() + b"\r\n\r\n" + content)
                    await writer.drain()
                    if len(payload["messages"]) == 4:
                        fast_finished.set()
                    return
                if stall == "sse":
                    content = ('data: ' + json.dumps({"choices": [{"delta": {"content": "partial"}}]})
                               + '\n\ndata: ' + json.dumps({"choices": [], "usage": {"total_tokens": 4}}) + '\n\n').encode()
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" + content)
                elif stall == "body":
                    writer.write(b'HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n{"choices":')
                await writer.drain()
                slow_started.set()
                await reader.read()
                disconnected.set()
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        cfg.server.base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        config_path = cfg.root_dir / "signal-config.json"
        config_path.write_text(json.dumps(cfg.model_dump(mode="json")))
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", "vllm_bench_eval.cli", "conversation-run",
            "--config", str(config_path), "--no-sync", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            await asyncio.wait_for(asyncio.gather(slow_started.wait(), fast_finished.wait()), 10)
            started = time.monotonic()
            process.send_signal(signal.Signals(signum))
            stdout, stderr = await asyncio.wait_for(process.communicate(), 3)
            assert process.returncode == 128 + signum, (stdout, stderr)
            assert time.monotonic() - started < 3
            await asyncio.wait_for(disconnected.wait(), 1)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            server.close()
            await server.wait_closed()
            for task in list(handlers):
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
        return requests
    requests = asyncio.run(exercise())
    assert len(requests) == 3
    path = next(cfg.path(cfg.conversation.result_dir).glob("conversation-*.json"))
    result = load_conversation_result(path)
    assert result["stop"]["signal"] == signum
    assert [item["status"] for item in result["conversations"]] == ["completed", "interrupted"]
    assert result["execution"]["completed_turns"] == 2
    assert result["execution"]["cancelled_turns"] == 1
    assert result["metrics"]["counts"]["failed"] == 0
    if stall == "sse":
        assert result["metrics"]["counts"]["known_consumed_tokens"] == 30
    saved = path.read_bytes(), path.with_suffix(".events.jsonl").read_bytes()
    time.sleep(.03)
    assert saved == (path.read_bytes(), path.with_suffix(".events.jsonl").read_bytes())


@pytest.mark.parametrize("signum", [2, 15])
def test_signal_exit_code_survives_sync_failure(cfg, monkeypatch, signum):
    from vllm_bench_eval import cli, conversation
    result, _, path = execute(cfg)
    result["stop"]["signal"] = signum
    path.write_text(json.dumps(result))
    cfg.benchmark_platform.enabled = True
    monkeypatch.setattr(cli, "_load", lambda config: cfg)
    monkeypatch.setattr(conversation, "run_conversations", lambda config: path)
    def fail(*args, **kwargs):
        raise PlatformDeliveryError("simulated upload failure")
    monkeypatch.setattr(conversation, "sync_conversations", fail)
    output = CliRunner().invoke(app, ["conversation-run"])
    assert output.exit_code == 128 + signum
    assert "simulated upload failure" in output.stdout
    assert load_conversation_result(path)["stop"]["signal"] == signum


def test_async_client_ownership_and_legacy_result_compatibility(cfg):
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response()))) as client:
            cfg.conversation.stream = False
            first = await async_run_conversations(cfg, client=client)
            assert not client.is_closed
            second = await async_run_conversations(cfg, client=client)
            with pytest.raises(ConversationError, match="active event loop"):
                run_conversations(cfg)
            return first, second
    first, second = asyncio.run(exercise())
    assert first != second
    legacy = json.loads(first.read_text())
    for field in ("lifecycle_version", "status", "stop"):
        legacy.pop(field)
    first.write_text(json.dumps(legacy))
    assert load_conversation_result(first)["metrics"]["counts"]["completed"] == 2


@pytest.mark.parametrize("field,value", [("duration_sec", 0), ("duration_sec", float("inf")),
    ("stop_policy", "wait"), ("stop_grace_sec", -1), ("stop_grace_sec", float("nan"))])
def test_stop_configuration_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        AppConfig(conversation={field: value})


def test_duration_cli_overrides_and_partial_retry(cfg, monkeypatch):
    from vllm_bench_eval import cli, conversation
    cfg.conversation.stream = False
    cfg.conversation.duration_sec = .01
    async def stalled(payload):
        await asyncio.sleep(1)
        return httpx.Response(200, json=response())
    result, _, path = execute(cfg, stalled)
    cfg.benchmark_platform.enabled = True
    monkeypatch.setattr(cli, "_load", lambda config: cfg)
    platform = StrictPlatform()
    def saved(config):
        assert config.conversation.duration_sec == 12
        assert config.conversation.stop_policy == "drain"
        assert config.conversation.stop_grace_sec == .5
        return path
    monkeypatch.setattr(conversation, "run_conversations", saved)
    monkeypatch.setattr(conversation, "build_platform_client", lambda config: platform)
    output = CliRunner().invoke(app, ["conversation-run", "--no-sync", "--duration-sec", "12",
        "--stop-policy", "drain", "--stop-grace-sec", ".5"])
    assert output.exit_code == 0
    assert not platform.spans
    original = path.read_bytes(), path.with_suffix(".events.jsonl").read_bytes()
    output = CliRunner().invoke(app, ["conversation-sync", str(path)])
    assert output.exit_code == 0
    assert len(platform.spans) == 1 and platform.spans[0]["metadata"]["status"] == "cancelled"
    assert original == (path.read_bytes(), path.with_suffix(".events.jsonl").read_bytes())


def test_async_transport_matches_sync_parser_and_preserves_cancelled_usage():
    from vllm_bench_eval.agent_transport import async_chat_request, chat_request
    content = ('data: ' + json.dumps({"choices": [{"delta": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]})
               + '\n\ndata: ' + json.dumps({"choices": [], "usage": {"total_tokens": 4, "completion_tokens": 2}}) + '\n\ndata: [DONE]\n\n')
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=content))) as client:
        expected, _ = chat_request(client, "http://test/chat", {"stream": True}, 2)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=content))) as client:
            actual, timing = await async_chat_request(client, "http://test/chat", {"stream": True}, 2)
            assert actual == expected and timing["tpot_null_reason"] is None
        usage_seen, closed = asyncio.Event(), asyncio.Event()
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[],"usage":{"total_tokens":7}}\n\n'
                usage_seen.set()
                await asyncio.Event().wait()
            async def aclose(self):
                closed.set()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))) as client:
            task = asyncio.create_task(async_chat_request(client, "http://test/chat", {"stream": True}, 2))
            await usage_seen.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.agent_usage == {"total_tokens": 7}
            assert caught.value.agent_timing["tpot_null_reason"] == "no_output"
            assert closed.is_set()
    asyncio.run(exercise())


def test_duration_before_first_dispatch_saves_not_started_result(cfg):
    cfg.conversation.duration_sec = .000001
    result, payloads, path = execute(cfg)
    assert payloads == []
    assert result["stop"]["reason"] == "duration"
    assert result["conversations"][0]["status"] == "not_started"
    assert result["execution"]["not_started_turns"] == 2
    assert load_conversation_result(path)["metrics"]["counts"]["failed"] == 0


def test_async_request_deadline_has_explicit_error():
    from vllm_bench_eval.agent_transport import async_chat_request
    async def exercise():
        async def stalled(request):
            await asyncio.Event().wait()
        async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as client:
            with pytest.raises((asyncio.TimeoutError, TimeoutError), match="request timeout exceeded") as caught:
                await async_chat_request(client, "http://test/chat", {"stream": False}, .001)
            assert caught.value.agent_timing["e2e_ms"] > 0
    asyncio.run(exercise())


def interrupted_conversation(cfg, *, policy="generated"):
    cfg.conversation.history_policy = policy
    cfg.conversation.stream = False
    cfg.conversation.duration_sec = .02
    calls = 0
    async def partial(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            body = response("persisted generated text")
            body["choices"][0]["message"]["_reasoning_present"] = True
            return httpx.Response(200, json=body)
        await asyncio.sleep(1)
        return httpx.Response(200, json=response())
    return execute(cfg, partial)


def test_resume_uses_confirmed_generated_history_and_active_attempts(cfg):
    first, _, path = interrupted_conversation(cfg)
    time.sleep(.03)
    cfg.conversation.resume_result_path = str(path)
    cfg.conversation.duration_sec = None
    resumed, requests, again = execute(cfg)
    assert again == path and resumed["run_id"] == first["run_id"]
    assert len(requests) == 1
    assert requests[0]["messages"][2] == {"role": "assistant", "content": "persisted generated text"}
    assert len(resumed["conversations"][0]["turns"]) == 2
    assert resumed["conversations"][0]["known_consumed_tokens"] == 26
    assert len(resumed["attempts"]) == 2
    assert resumed["duration_s"] == sum(window["duration_s"] for window in resumed["attempts"])
    assert resumed["duration_s"] < .05
    events = list(map(json.loads, path.with_suffix(".events.jsonl").read_text().splitlines()))
    starts = [event for event in events if event["event"] == "llm_start"]
    assert len(starts) == len({event["span_id"] for event in starts}) == 3
    assert starts[1]["logical_turn_id"] == starts[2]["logical_turn_id"]
    assert resumed["metrics"]["counts"]["completed"] == 2
    assert resumed["metrics"]["counts"]["failed"] == 0
    assert resumed["metrics"]["counts"]["cancelled"] == 1
    assert len(resumed["attempt_metrics"]) == 2
    _, calls, _ = execute(cfg)
    assert calls == []


@pytest.mark.parametrize("tail", [b"", b'{"event":"truncated', b'\xe4\xb8'])
def test_recover_events_before_snapshot_and_tail_backup(cfg, tail, monkeypatch):
    from vllm_bench_eval import conversation
    result, _, path = execute(cfg)
    events_path = path.with_suffix(".events.jsonl")
    original = events_path.read_bytes().rstrip(b"\n") + (b"\n" + tail if tail else b"")
    events_path.write_bytes(original)
    result["conversations"][0].update(turns=[], status="not_started", execution_ok=None, known_consumed_tokens=0)
    result.pop("end_time")
    result.pop("metrics")
    path.write_text(json.dumps(result))
    def forbidden(*args, **kwargs):
        pytest.fail("offline recovery must not create an HTTP client")
    monkeypatch.setattr(conversation.httpx, "AsyncClient", forbidden)
    conversation.recover_conversations(path)
    recovered = load_conversation_result(path)
    assert len(recovered["conversations"][0]["turns"]) == 2
    assert recovered["conversations"][0]["status"] == "completed"
    assert recovered["conversations"][0]["known_consumed_tokens"] == 26
    backups = list(path.parent.glob("*.backup-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert events_path.read_bytes().endswith(b"\n")
    conversation.recover_conversations(path)
    assert len(list(path.parent.glob("*.backup-*"))) == 1


def test_recovery_rejects_internal_corruption_without_rewrite(cfg):
    from vllm_bench_eval.conversation import recover_conversations
    result, _, path = execute(cfg)
    events_path = path.with_suffix(".events.jsonl")
    original = b"broken\n" + events_path.read_bytes()
    events_path.write_bytes(original)
    with pytest.raises(ValueError, match="invalid data"):
        recover_conversations(path)
    assert events_path.read_bytes() == original
    assert not list(path.parent.glob("*.backup-*"))


@pytest.mark.parametrize("change", ["model", "dataset", "policy", "source"])
def test_resume_rejects_identity_differences_before_requests(cfg, change):
    first, _, path = interrupted_conversation(cfg)
    cfg.conversation.resume_result_path = str(path)
    if change == "model":
        cfg.server.model = "other"
    elif change == "dataset":
        source = cfg.path(cfg.conversation.dataset_path)
        source.write_text(source.read_text() + " ")
    elif change == "policy":
        cfg.conversation.history_policy = "dataset"
    else:
        first["conversations"][0]["conversation"]["messages"][1]["content"] = "changed"
        path.write_text(json.dumps(first))
    with pytest.raises(ConversationError, match="differs"):
        execute(cfg, lambda payload: pytest.fail("identity rejection must precede HTTP"))


def test_resume_keeps_failed_request_usage_in_token_budget(cfg):
    cfg.conversation.stream = False
    cfg.conversation.token_budget = 15
    first, _, path = execute(cfg, lambda payload: httpx.Response(200, json=response(finish="length")))
    assert first["conversations"][0]["known_consumed_tokens"] == 13
    cfg.conversation.resume_result_path = str(path)
    resumed, payloads, _ = execute(cfg)
    assert payloads[0]["max_tokens"] == 2
    assert resumed["conversations"][0]["known_consumed_tokens"] == 26
    assert resumed["conversations"][0]["status"] == "failed"


def test_resume_keeps_conversation_timeout_budget(cfg):
    cfg.conversation.conversation_timeout_sec = .08
    first, _, path = interrupted_conversation(cfg)
    cfg.conversation.resume_result_path = str(path)
    cfg.conversation.duration_sec = None
    async def slow(payload):
        await asyncio.sleep(.2)
        return httpx.Response(200, json=response())
    resumed, requests, _ = execute(cfg, slow)
    item = resumed["conversations"][0]
    assert len(requests) == 1 and item["status"] == "failed"
    assert "timeout" in item["error"]
    assert .07 <= item["duration_s"] < .15


def test_recover_and_resume_lock_conflicts_and_legacy_rejection(cfg):
    from vllm_bench_eval.conversation import recover_conversations
    from vllm_bench_eval.lifecycle import run_lock
    result, _, path = interrupted_conversation(cfg)
    cfg.conversation.resume_result_path = str(path)
    with run_lock(path):
        for operation in (lambda: load_conversation_result(path), lambda: recover_conversations(path),
                          lambda: execute(cfg), lambda: sync_conversations(result, cfg, client=StrictPlatform())):
            with pytest.raises(ValueError, match="already owns"):
                operation()
    result["lifecycle_version"] = 1
    path.write_text(json.dumps(result))
    with pytest.raises(ConversationError, match="version 2"):
        execute(cfg)


def test_partial_resume_sync_snapshot_receipts_and_final_scores(cfg):
    from vllm_bench_eval.metric_names import SUMMARY_TRACE_NAME, summary_score
    platform = StrictPlatform()
    first, _, path = interrupted_conversation(cfg)
    partial = sync_conversations(first, cfg, client=platform)
    objects = len(platform.items), len(platform.traces), len(platform.spans), len(platform.references)
    repeated = sync_conversations(first, cfg, client=platform)
    assert repeated["already_synced"] is True
    assert objects == (len(platform.items), len(platform.traces), len(platform.spans), len(platform.references))
    cfg.conversation.resume_result_path = str(path)
    cfg.conversation.duration_sec = None
    final, _, _ = execute(cfg)
    completed = sync_conversations(final, cfg, client=platform)
    assert partial["snapshot_id"] != completed["snapshot_id"]
    assert partial["experiment_name"] != completed["experiment_name"]
    assert len(platform.items) == 1 and len(platform.traces) == 4 and len(platform.spans) == 5
    assert len(platform.references) == 2
    partial_summary, final_summary = [trace for trace in platform.traces if trace["name"] == SUMMARY_TRACE_NAME]
    assert partial_summary["metadata"]["snapshot_state"] == "partial"
    assert final_summary["metadata"]["snapshot_state"] == "final"
    scores = {score["name"]: score["value"] for score in final_summary["feedback_scores"]}
    assert scores[summary_score("completed_ratio")[0]] == pytest.approx(2 / 3)
    assert scores[summary_score("total_token_throughput_tps")[0]] == pytest.approx(26 / final["duration_s"])
    assert platform.traces[0]["metadata"]["logical_conversation_id"] == platform.traces[2]["metadata"]["logical_conversation_id"]
    assert platform.traces[0]["id"] != platform.traces[2]["id"]
    assert len({span["id"] for span in platform.spans}) == 5


def test_offline_recovery_closes_abandoned_request_and_estimates_attempt(cfg):
    from vllm_bench_eval.conversation import recover_conversations
    result, _, path = interrupted_conversation(cfg)
    events_path = path.with_suffix(".events.jsonl")
    events = list(map(json.loads, events_path.read_text().splitlines()))
    last_start = max(index for index, event in enumerate(events) if event["event"] == "llm_start")
    events = events[:last_start + 1]
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    result["conversations"][0].update(turns=[], status="not_started", duration_s=0)
    for field in ("end_time", "duration_s"):
        result["attempts"][0].pop(field)
    result.pop("end_time")
    path.write_text(json.dumps(result))
    recover_conversations(path)
    recovered = load_conversation_result(path)
    assert len(recovered["conversations"][0]["turns"]) == 1
    assert recovered["conversations"][0]["status"] == "interrupted"
    assert recovered["attempts"][0]["duration_estimated"] is True
    assert recovered["metrics"]["counts"]["cancelled"] == 1
    assert recovered["metrics"]["counts"]["failed"] == 0
    cfg.conversation.resume_result_path = str(path)
    cfg.conversation.duration_sec = None
    final, calls, _ = execute(cfg)
    assert len(calls) == 1 and len(final["conversations"][0]["turns"]) == 2


@pytest.mark.parametrize("budget,expected", [(1000000, "completed"), (30, "failed")])
def test_recovery_uses_latest_attempt_confirmation_after_prior_failure(cfg, budget, expected):
    from vllm_bench_eval.conversation import recover_conversations
    cfg.conversation.stream = False
    cfg.conversation.token_budget = budget
    first, _, path = execute(cfg, lambda payload: httpx.Response(200, json=response(finish="length")))
    assert first["conversations"][0]["status"] == "failed"
    cfg.conversation.resume_result_path = str(path)
    resumed, requests, _ = execute(cfg)
    assert len(requests) == 2
    events_path = path.with_suffix(".events.jsonl")
    events = list(map(json.loads, events_path.read_text().splitlines()))
    latest_attempt = resumed["attempts"][-1]["attempt_id"]
    events = [event for event in events if not (event["event"] == "conversation_end" and event["run_attempt_id"] == latest_attempt)]
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    resumed["conversations"] = first["conversations"]
    path.write_text(json.dumps(resumed))
    recover_conversations(path)
    recovered = load_conversation_result(path)
    item = recovered["conversations"][0]
    assert item["status"] == expected
    assert len(item["turns"]) == 2 and item["known_consumed_tokens"] == 39
    assert item["execution_ok"] is (expected == "completed")
    if expected == "completed":
        assert "error" not in item
    else:
        assert "token_budget" in item["error"]


def test_symlink_alias_uses_the_original_run_lock(cfg):
    from vllm_bench_eval.lifecycle import run_lock
    from vllm_bench_eval.conversation import recover_conversations
    result, _, path = execute(cfg)
    alias = path.with_name("alias.json")
    alias.symlink_to(path)
    with run_lock(path):
        for operation in (lambda: load_conversation_result(alias), lambda: recover_conversations(alias)):
            with pytest.raises(ValueError, match="already owns"):
                operation()
    with run_lock(alias):
        with pytest.raises(ValueError, match="already owns"):
            load_conversation_result(path)


def test_measurement_half_open_boundaries_and_diagnostic_counts():
    from vllm_bench_eval.lifecycle import measure_attempt
    events = []
    def request(span, start, end, status="completed"):
        events.append({"event": "llm_start", "run_attempt_id": "attempt", "span_id": span, "attempt_offset_s": start})
        if end is None:
            return
        result = {"event": "llm_end", "run_attempt_id": "attempt", "span_id": span, "attempt_offset_s": end,
                  "usage": {"total_tokens": 13, "prompt_tokens": 10, "completion_tokens": 3}, "timing": {"e2e_ms": 100}}
        result.update({"output": {"content": "answer"}} if status == "completed" else {"status": status, "error": status})
        events.append(result)
    request("inside", 1, 2)
    request("left", .5, 1.5)
    request("right", 2.5, 3)
    request("outside", .1, .2)
    request("after", 3, 3.5)
    request("failed", 1.2, 1.3, "failed")
    request("cancelled", 1.4, 3.2, "cancelled")
    request("missing", 2, None)
    measured = measure_attempt({"enabled": True, "warmup_sec": 1, "duration_sec": 2, "stop_on_load_drop": False}, "attempt", events, 5)
    assert measured["selected_request_ids"] == ["inside"]
    assert measured["window"]["duration_s"] == 2
    assert measured["selection_counts"] == {"started_in_window": 5, "included_successful": 1,
        "failed_started_in_window": 1, "cancelled_started_in_window": 1,
        "cross_boundary_successful_excluded": 2, "missing_boundary_evidence": 1}
    assert measured["metrics"]["throughput"]["total_tps"] == 6.5


def test_measurement_warmup_duration_and_cross_boundary_drain(cfg):
    cfg.conversation.stream = False
    cfg.conversation.stop_policy = "drain"
    cfg.conversation.stop_grace_sec = 1
    cfg.conversation.measurement.enabled = True
    cfg.conversation.measurement.warmup_sec = .03
    cfg.conversation.measurement.duration_sec = .04
    async def delayed(payload):
        await asyncio.sleep(.04)
        return httpx.Response(200, json=response())
    result, payloads, _ = execute(cfg, delayed)
    measured = result["measurement"]["attempts"][0]
    assert len(payloads) == 2 and result["metrics"]["counts"]["completed"] == 2
    assert result["stop"]["reason"] == "measurement_duration"
    assert measured["window"]["start_offset_s"] == .03
    assert measured["window"]["cutoff_offset_s"] == pytest.approx(.07)
    assert measured["window"]["duration_s"] == pytest.approx(.04)
    assert measured["selection_counts"]["cross_boundary_successful_excluded"] == 2
    assert measured["selection_counts"]["included_successful"] == 0
    assert measured["metrics"]["e2el_ms"]["mean"] is None
    assert measured["load_trajectory"][-1]["active_conversations"] == 0
    assert max(point["inflight_requests"] for point in measured["load_trajectory"]) == 1


def test_measurement_stop_before_warmup_is_empty_and_sync_matches(cfg):
    cfg.conversation.stream = False
    cfg.conversation.duration_sec = .02
    cfg.conversation.measurement.enabled = True
    cfg.conversation.measurement.warmup_sec = 1
    async def delayed(payload):
        await asyncio.sleep(1)
        return httpx.Response(200, json=response())
    result, _, path = execute(cfg, delayed)
    assert result["measurement"]["metrics"]["throughput"]["total_tps"] is None
    assert result["measurement"]["metrics"]["counts"]["duration_s"] == 0
    platform = StrictPlatform()
    sync_conversations(result, cfg, client=platform)
    trace = next(trace for trace in platform.traces if trace["name"] == "测量窗口汇总")
    assert trace["output"] == result["measurement"]["metrics"]
    assert all(score["value"] is not None for score in trace["feedback_scores"])
    assert load_conversation_result(path)["measurement"] == result["measurement"]


@pytest.mark.parametrize("target,reached", [(2, True), (3, False)])
def test_measurement_load_drop_requires_reached_target(cfg, target, reached):
    records = json.loads(cfg.path(cfg.conversation.dataset_path).read_text())
    slow = copy.deepcopy(records[0])
    slow["id"] = "slow"
    slow["conversations"][1]["value"] = "slow"
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps([records[0], slow]))
    cfg.conversation.max_concurrency = target
    cfg.conversation.stream = False
    cfg.conversation.measurement.enabled = True
    cfg.conversation.measurement.stop_on_load_drop = True
    async def mixed(payload):
        if payload["messages"][1]["content"] == "slow":
            await asyncio.sleep(.04)
        else:
            await asyncio.sleep(.003)
        return httpx.Response(200, json=response())
    result, payloads, _ = execute(cfg, mixed)
    measured = result["measurement"]["attempts"][0]
    assert measured["target_reached"] is reached
    if reached:
        assert result["stop"]["reason"] == "load_drop"
        assert len(payloads) == 3
        assert result["execution"]["cancelled_turns"] == 1
    else:
        assert result["stop"]["reason"] is None
        assert len(payloads) == 4


def test_measurement_resume_keeps_prior_config_and_sums_active_windows(cfg):
    cfg.conversation.measurement.enabled = True
    first, _, path = interrupted_conversation(cfg)
    prior = copy.deepcopy(first["attempts"][0]["measurement"])
    cfg.conversation.resume_result_path = str(path)
    cfg.conversation.duration_sec = None
    cfg.conversation.measurement.warmup_sec = .002
    async def delayed(payload):
        await asyncio.sleep(.003)
        return httpx.Response(200, json=response())
    final, _, _ = execute(cfg, delayed)
    assert final["attempts"][0]["measurement"] == prior
    assert final["attempts"][0]["measurement_config"]["warmup_sec"] == 0
    assert final["attempts"][1]["measurement_config"]["warmup_sec"] == .002
    duration = sum(attempt["measurement"]["window"]["duration_s"] for attempt in final["attempts"])
    assert final["measurement"]["metrics"]["counts"]["duration_s"] == duration
    tokens = sum(attempt["measurement"]["metrics"]["counts"]["total_tokens"] for attempt in final["attempts"])
    assert final["measurement"]["metrics"]["throughput"]["total_tps"] == pytest.approx(tokens / duration)


def test_disabled_measurement_keeps_default_execution_and_no_load_events(cfg):
    cfg.conversation.measurement.warmup_sec = 10
    cfg.conversation.measurement.duration_sec = .000001
    cfg.conversation.measurement.stop_on_load_drop = True
    result, requests, path = execute(cfg)
    assert len(requests) == 2 and result["stop"]["reason"] is None
    assert "measurement" not in result
    events = list(map(json.loads, path.with_suffix(".events.jsonl").read_text().splitlines()))
    assert not any(event["event"] == "load" or "attempt_offset_s" in event for event in events)


def test_recovery_uses_persisted_attempt_offsets_without_new_clock(cfg):
    from vllm_bench_eval.conversation import recover_conversations
    cfg.conversation.measurement.enabled = True
    result, _, path = interrupted_conversation(cfg)
    events_path = path.with_suffix(".events.jsonl")
    events = list(map(json.loads, events_path.read_text().splitlines()))
    last_start = max(index for index, event in enumerate(events) if event["event"] == "llm_start")
    events = events[:last_start + 1]
    offset = max(event["attempt_offset_s"] for event in events)
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    for field in ("duration_s", "end_time", "measurement", "stop"):
        result["attempts"][0].pop(field)
    result.pop("measurement")
    result.pop("end_time")
    path.write_text(json.dumps(result))
    time.sleep(.02)
    recover_conversations(path)
    recovered_events = list(map(json.loads, events_path.read_text().splitlines()))
    assert recovered_events[-1]["recovered"] is True
    assert recovered_events[-1]["attempt_offset_s"] == offset
    recovered = load_conversation_result(path)
    measured = recovered["measurement"]["attempts"][0]
    assert measured["window"]["cutoff_offset_s"] == offset
    assert measured["window"]["duration_estimated"] is True
    assert measured["window"]["cutoff_reason"] == "last_persisted_offset_unknown_stop"
    assert measured["window"]["stop_boundary_evidence"] == "missing"


def test_late_stop_observation_keeps_earliest_configured_deadline(cfg):
    import threading
    from vllm_bench_eval.conversation import _StopController
    from vllm_bench_eval.lifecycle import ThreadStopController
    cfg.conversation.duration_sec = 1
    conversation = _StopController(cfg.conversation)
    conversation.request("load_drop", at=conversation.started + 2)
    assert conversation.reason == "duration"
    assert conversation.monotonic == conversation.started + 1
    cfg.coding_agent.duration_sec = 1
    agent = ThreadStopController(cfg.coding_agent, conversation.started, threading.Event(), {})
    agent.request("load_drop", at=conversation.started + 2)
    assert agent.stop["reason"] == "duration"
    assert agent.stop["requested_offset_s"] == 1


def test_successful_measurement_samples_sync_without_false_completion_ratio(cfg):
    from vllm_bench_eval.metric_names import summary_score
    cfg.conversation.measurement.enabled = True
    result, _, _ = execute(cfg)
    metrics = result["measurement"]["metrics"]
    assert metrics["counts"]["completed"] == 2
    assert metrics["sample_counts"]["e2el_ms"] == 2
    assert metrics["throughput"]["total_tps"] > 0
    platform = StrictPlatform()
    sync_conversations(result, cfg, client=platform)
    measured = next(trace for trace in platform.traces if trace["name"] == "测量窗口汇总")
    assert measured["output"] == metrics
    scores = {score["name"]: score["value"] for score in measured["feedback_scores"]}
    assert summary_score("completed_ratio")[0] not in scores
    assert scores[summary_score("total_token_throughput_tps")[0]] == metrics["throughput"]["total_tps"]


@pytest.mark.parametrize("cause", ["duration", "load_drop"])
def test_recovery_keeps_persisted_cutoff_before_drain_without_final_snapshot(cfg, cause):
    from vllm_bench_eval.conversation import recover_conversations
    records = json.loads(cfg.path(cfg.conversation.dataset_path).read_text())
    slow = copy.deepcopy(records[0])
    slow["id"] = "slow"
    slow["conversations"][1]["value"] = "slow"
    cfg.path(cfg.conversation.dataset_path).write_text(json.dumps([records[0], slow]))
    cfg.conversation.stream = False
    cfg.conversation.max_concurrency = 2
    cfg.conversation.stop_policy = "drain"
    cfg.conversation.stop_grace_sec = 1
    cfg.conversation.measurement.enabled = True
    cfg.conversation.measurement.stop_on_load_drop = cause == "load_drop"
    cfg.conversation.duration_sec = .025 if cause == "duration" else None
    async def mixed(payload):
        await asyncio.sleep(.06 if payload["messages"][1]["content"] == "slow" else .003)
        return httpx.Response(200, json=response())
    normal, payloads, path = execute(cfg, mixed)
    measured = normal["attempts"][0]["measurement"]
    assert len(payloads) == 3 and len(measured["selected_request_ids"]) == 2
    damaged = json.loads(path.read_text())
    for field in ("stop", "measurement", "end_time", "duration_s"):
        damaged["attempts"][0].pop(field)
    damaged.pop("stop")
    damaged.pop("end_time")
    damaged.pop("measurement")
    path.write_text(json.dumps(damaged))
    recover_conversations(path)
    recovered = load_conversation_result(path)["attempts"][0]["measurement"]
    assert recovered["window"]["cutoff_offset_s"] == measured["window"]["cutoff_offset_s"]
    assert recovered["selected_request_ids"] == measured["selected_request_ids"]
    assert recovered["metrics"]["counts"]["total_tokens"] == 26
    assert recovered["window"]["stop_boundary_evidence"] == "persisted_stop_or_load_drop"
