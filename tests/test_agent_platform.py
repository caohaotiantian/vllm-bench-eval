"""Platform UUIDv7 IDs and background delivery failure evidence."""
import copy
import datetime as dt
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
from typer.testing import CliRunner

from test_coding_agent import agent_cfg, call, completion, execute
from vllm_bench_eval.agent_platform import PlatformDeliveryError, checked_delivery, platform_id
from vllm_bench_eval.coding_agent import sync_agent
from vllm_bench_eval.cli import app


class StrictPlatform:
    def __init__(self, failure=None):
        self.items, self.traces, self.spans, self.references = [], [], [], []
        self.failure = failure
    def get_or_create_dataset(self, **kwargs):
        return self
    def get_items(self):
        return self.items
    def insert(self, items):
        self.items.extend({"id": str(uuid.uuid4()), **item} for item in items)
    def create_experiment(self, **kwargs):
        return SimpleNamespace(insert=self.insert_references)
    @staticmethod
    def valid(id):
        assert uuid.UUID(id).version == 7, "platform requires UUIDv7"
    def insert_references(self, refs):
        for ref in refs:
            self.valid(ref.trace_id)
        self.references.extend(refs)
    def trace(self, **kwargs):
        self.valid(kwargs["id"])
        self.traces.append(copy.deepcopy(kwargs))
    def span(self, **kwargs):
        self.valid(kwargs["id"])
        self.valid(kwargs["trace_id"])
        if kwargs.get("parent_span_id"):
            self.valid(kwargs["parent_span_id"])
        self.spans.append(copy.deepcopy(kwargs))
    def flush(self):
        if self.failure == "reject":
            logging.getLogger("opik.message_processing.processors.online_message_processor").error("Rejected trace batch")
        elif self.failure == "connection":
            logging.getLogger("opik.message_processing.processors.online_message_processor").warning("Failed to process message due to connection error")
        return self.failure != "timeout"


def test_uuid7_mapping_timestamp_variant_and_passthrough():
    source = str(uuid.uuid5(uuid.NAMESPACE_URL, "source"))
    timestamp = "2026-01-02T03:04:05.678+08:00"
    mapped = platform_id(source, timestamp, "trace")
    assert mapped == platform_id(source, timestamp, "trace")
    parsed = uuid.UUID(mapped)
    assert parsed.version == 7 and parsed.variant == uuid.RFC_4122
    assert parsed.int >> 80 == int(dt.datetime.fromisoformat(timestamp).timestamp() * 1000)
    assert platform_id(mapped, "2027-01-01T00:00:00+00:00", "trace") == mapped
    assert platform_id(source, timestamp, "span") != mapped


def test_strict_platform_accepts_ids_parents_refs_and_repeat(agent_cfg):
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("write_file", {"path": "value.txt", "content": "new"})]}, "tool_calls"),
                                   completion({"role": "assistant", "content": "Done"})])
    original_result = copy.deepcopy(result)
    original_events = Path(result["events_path"]).read_bytes()
    client = StrictPlatform()
    with pytest.raises(AssertionError, match="UUIDv7"):
        client.valid(result["tasks"][0]["trace_id"])
    sync_agent(result, agent_cfg, client=client)
    trace_ids = {trace["id"] for trace in client.traces}
    span_ids = {span["id"] for span in client.spans}
    assert all(span["trace_id"] in trace_ids for span in client.spans)
    assert all(span["parent_span_id"] in span_ids for span in client.spans if span.get("parent_span_id"))
    assert client.references[0].trace_id == client.traces[0]["id"]
    assert client.traces[0]["metadata"]["source_trace_id"] == result["tasks"][0]["trace_id"]
    assert all("usage" not in trace for trace in client.traces)
    assert len([span for span in client.spans if span.get("usage")]) == 2
    second = StrictPlatform()
    sync_agent(result, agent_cfg, client=second)
    assert [trace["id"] for trace in client.traces] == [trace["id"] for trace in second.traces]
    assert [span["id"] for span in client.spans] == [span["id"] for span in second.spans]
    assert result == original_result and Path(result["events_path"]).read_bytes() == original_events
    result["schema_version"] = 1
    legacy = StrictPlatform()
    sync_agent(result, agent_cfg, client=legacy)
    assert [trace["id"] for trace in legacy.traces] == [trace["id"] for trace in client.traces]


@pytest.mark.parametrize("failure", ["reject", "timeout", "connection"])
def test_background_failures_cannot_report_sync_success(agent_cfg, failure):
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    logger = logging.getLogger("opik.message_processing")
    before = list(logger.handlers)
    with pytest.raises(PlatformDeliveryError):
        sync_agent(result, agent_cfg, client=StrictPlatform(failure))
    assert logger.handlers == before


def test_delivery_observer_removed_when_sync_raises():
    logger = logging.getLogger("opik.message_processing")
    before = list(logger.handlers)
    with pytest.raises(ValueError):
        with checked_delivery():
            raise ValueError("input failed")
    assert logger.handlers == before


def test_cli_sync_returns_nonzero_on_background_failure(agent_cfg, monkeypatch):
    from vllm_bench_eval import cli, coding_agent
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    path = Path(result["events_path"]).with_suffix("").with_suffix(".json")
    agent_cfg.benchmark_platform.enabled = True
    monkeypatch.setattr(cli, "_load", lambda config: agent_cfg)
    monkeypatch.setattr(coding_agent, "build_platform_client", lambda platform: StrictPlatform("reject"))
    response = CliRunner().invoke(app, ["agent-sync", str(path)])
    assert response.exit_code == 1 and "background delivery error" in response.stdout
