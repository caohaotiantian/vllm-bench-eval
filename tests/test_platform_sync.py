"""Syncer tests against an in-memory fake of the Benchmark 平台 client."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from vllm_bench_eval.config import PlatformSettings
from vllm_bench_eval.platform_sync import (
    SUMMARY_TRACE_NAME,
    SyncError,
    build_experiment_config,
    dataset_exists,
    index_dataset_items,
    render_experiment_name,
    sync_result,
)
from vllm_bench_eval.results import parse_result_file
from vllm_bench_eval.samples import align_requests_to_samples, load_samples


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


@dataclass
class FakeTrace:
    id: str
    name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    tags: List[str]
    feedback_scores: List[Dict[str, Any]]
    project_name: str
    start_time: Any = None
    end_time: Any = None


@dataclass
class FakeSpan:
    id: str
    trace_id: str
    name: str
    type: str
    start_time: Any
    end_time: Any
    input: Dict[str, Any]
    output: Dict[str, Any]
    metadata: Dict[str, Any]
    usage: Any
    model: Any
    provider: Any
    project_name: Any


@dataclass
class FakeDataset:
    name: str
    description: Optional[str] = None
    items: List[Dict[str, Any]] = field(default_factory=list)
    insert_calls: int = 0

    def insert(self, items):
        self.insert_calls += 1
        for item in items:
            self.items.append({"id": f"ds-{len(self.items)}", **item})

    def get_items(self, nb_samples=None):
        return list(self.items)


@dataclass
class FakeExperiment:
    id: str
    name: str
    dataset_name: str
    experiment_config: Dict[str, Any]
    references: List[Any] = field(default_factory=list)

    def insert(self, experiment_items_references):
        self.references.extend(experiment_items_references)


class NotFound(Exception):
    status_code = 404


class FakePlatformClient:
    def __init__(self):
        self.datasets: Dict[str, FakeDataset] = {}
        self.experiments: List[FakeExperiment] = []
        self.traces: List[FakeTrace] = []
        self.spans: List[FakeSpan] = []
        self.flushed = 0

    def get_dataset(self, name):
        if name not in self.datasets:
            raise NotFound(name)
        return self.datasets[name]

    def get_or_create_dataset(self, name, description=None):
        return self.datasets.setdefault(name, FakeDataset(name=name, description=description))

    def create_experiment(self, dataset_name, name=None, experiment_config=None, **kwargs):
        exp = FakeExperiment(
            id=f"exp-{len(self.experiments)}",
            name=name,
            dataset_name=dataset_name,
            experiment_config=experiment_config or {},
        )
        self.experiments.append(exp)
        return exp

    def trace(self, name=None, input=None, output=None, metadata=None, tags=None,
              feedback_scores=None, project_name=None, start_time=None,
              end_time=None, **kwargs):
        t = FakeTrace(
            id=f"trace-{len(self.traces)}",
            name=name,
            input=input or {},
            output=output or {},
            metadata=metadata or {},
            tags=list(tags or []),
            feedback_scores=list(feedback_scores or []),
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
        )
        self.traces.append(t)
        return t

    def span(self, trace_id=None, name=None, type="general", start_time=None,
             end_time=None, input=None, output=None, metadata=None, usage=None,
             model=None, provider=None, project_name=None, **kwargs):
        sp = FakeSpan(
            id=f"span-{len(self.spans)}", trace_id=trace_id, name=name, type=type,
            start_time=start_time, end_time=end_time, input=input or {},
            output=output or {}, metadata=metadata or {}, usage=usage,
            model=model, provider=provider, project_name=project_name,
        )
        self.spans.append(sp)
        return sp

    def flush(self):
        self.flushed += 1


# --------------------------------------------------------------------------


@pytest.fixture
def platform_cfg():
    return PlatformSettings(
        url="http://localhost:5174/api/",
        workspace="default",
        project_name="proj",
        dataset_name="ds",
        experiment_name="exp-{model}-{ts}",
        tags=["vllm-bench"],
    )


RUN_CFG = {
    "endpoint": "/v1/completions",
    "base_url": "http://h:1",
    "dataset_name": "custom",
    "tool_version": "0.1.0",
    "provider": "openai",
}


@pytest.fixture
def synced(result_file, samples_file, platform_cfg, captured):
    from vllm_bench_eval.samples import align_by_prompt

    result = parse_result_file(result_file)
    samples = load_samples(samples_file)
    matched, extra = align_by_prompt(captured, samples)
    assert matched == len(captured) and extra == []
    client = FakePlatformClient()
    report = sync_result(
        result, samples, platform_cfg, client=client, source_label="sonnet",
        captured=captured, run_cfg=RUN_CFG, run_id="run123",
    )
    return client, report, result, samples


def test_dataset_items_are_uploaded(synced):
    client, report, _, samples = synced
    ds = client.datasets["ds"]
    assert len(ds.items) == len(samples)
    assert {i["prompt"] for i in ds.items} == {s.prompt for s in samples}
    assert ds.items[0]["dataset_source"] == "sonnet"
    # "source" is reserved by the platform and must not be set by us
    assert "source" not in ds.items[0]
    assert report.dataset_items_inserted == 3
    assert report.dataset_items_total == 3


def test_reruns_do_not_duplicate_dataset_items(result_file, samples_file, platform_cfg, captured):
    result = parse_result_file(result_file)
    samples = load_samples(samples_file)
    client = FakePlatformClient()
    sync_result(result, samples, platform_cfg, client=client, captured=captured)
    second = sync_result(result, samples, platform_cfg, client=client, captured=captured)
    assert second.dataset_items_inserted == 0
    assert len(client.datasets["ds"].items) == 3
    assert len(client.experiments) == 2


def test_one_trace_per_request_plus_summary(synced):
    client, report, result, _ = synced
    assert len(client.traces) == 4          # 3 requests + summary
    assert report.traces_created == 3
    assert client.traces[-1].name == SUMMARY_TRACE_NAME
    assert report.summary_trace_id == client.traces[-1].id
    assert all(t.project_name == "proj" for t in client.traces)
    assert [t.name for t in client.traces[:3]] == ["req-0000", "req-0001", "req-0002"]


def test_traces_carry_prompt_output_and_scores(synced):
    client, _, _, _ = synced
    trace = client.traces[0]
    assert trace.input["prompt"] == "prompt-0"
    assert trace.input["max_tokens"] == 128
    assert trace.output["generated_text"] == "alpha out"
    assert trace.output["success"] is True
    scores = {s["name"]: s["value"] for s in trace.feedback_scores}
    assert scores["TTFT(ms)"] == pytest.approx(100.0)
    assert scores["端到端延迟(ms)"] == pytest.approx(180.0)
    assert scores["请求成功"] == 1.0
    # token counts moved out of feedback scores
    assert "input_tokens" not in scores and "output_tokens" not in scores


def test_summary_trace_is_lean_and_grouped(synced):
    client, _, _, _ = synced
    summary = client.traces[-1]
    scores = {s["name"]: s["value"] for s in summary.feedback_scores}
    # headline set only, Chinese display names
    assert scores["TTFT均值(ms)"] == 150.0
    assert scores["TTFT P99(ms)"] == 190.0
    assert scores["TPOT均值(ms)"] == 25.0
    assert scores["输出吞吐(tokens/s)"] == 3.0
    assert scores["总吞吐(tokens/s)"] == 10.5
    assert scores["端到端延迟均值(ms)"] == 300.0
    assert scores["请求完成率"] == 1.0
    # counters are no longer feedback scores
    assert "输入token总数" not in scores and "总耗时(s)" not in scores
    assert len(scores) <= 14
    # grouped output
    out = summary.output
    assert out["TTFT(ms)"]["均值"] == 150.0
    assert out["端到端延迟(ms)"]["p99"] == 400.0
    assert out["吞吐"] == {"请求(req/s)": 0.75, "输出(tokens/s)": 3.0, "总计(tokens/s)": 10.5}
    assert out["计数"]["完成请求数"] == 3 and out["计数"]["失败请求数"] == 0
    # input is the run descriptor only, not the whole raw result
    assert set(summary.input) <= {
        "model", "tokenizer", "backend", "num_prompts", "request_rate",
        "max_concurrency", "endpoint", "base_url", "dataset_name",
        "dataset_path", "tool_version",
    }
    assert "ttfts" not in summary.metadata
    assert summary.start_time is not None and summary.end_time is not None
    assert "summary" in summary.tags


def test_experiment_links_every_trace_to_its_dataset_item(synced):
    client, report, result, samples = synced
    exp = client.experiments[0]
    assert exp.dataset_name == "ds"
    assert len(exp.references) == 3
    assert report.experiment_items_created == 3

    by_id = {item["id"]: item for item in client.datasets["ds"].items}
    traces_by_id = {t.id: t for t in client.traces}
    for ref in exp.references:
        item = by_id[ref.dataset_item_id]
        trace = traces_by_id[ref.trace_id]
        # the dataset item linked to a trace must hold the prompt that was sent
        assert trace.input["prompt"] == item["prompt"]


def test_experiment_config_contains_metrics_and_run_config(synced):
    client, _, _, _ = synced
    cfg = client.experiments[0].experiment_config
    assert cfg["metrics"]["mean_ttft_ms"] == 150.0
    assert cfg["metrics"]["output_throughput_tps"] == 3.0
    assert cfg["metrics"]["total_token_throughput_tps"] == 10.5
    assert cfg["run"]["model_id"] == "demo-model"
    assert cfg["run"]["max_concurrency"] == 2
    assert "ttfts" not in cfg["run"]


def test_client_is_flushed(synced):
    client, _, _, _ = synced
    assert client.flushed == 1


def test_unaligned_requests_are_reported(result_file, samples_file, platform_cfg, captured):
    result = parse_result_file(result_file)
    samples = load_samples(samples_file)
    # deliberately skip alignment
    client = FakePlatformClient()
    report = sync_result(result, samples, platform_cfg, client=client, captured=captured)
    assert report.unaligned_requests == len(captured)
    assert report.experiment_items_created == 0
    assert report.warnings


def test_experiment_name_template():
    class R:
        model_id = "m"
        backend = "openai"
        date = "20260917-120000"
        requests: list = []

        class aggregate:
            num_prompts = 3

    name = render_experiment_name("run-{model}-{backend}-{ts}", R())
    assert name == "run-m-openai-20260917-120000"
    # unknown placeholders fall back to the raw template instead of raising
    assert render_experiment_name("run-{nope}", R()) == "run-{nope}"


def test_build_experiment_config_merges_extra(result_file):
    result = parse_result_file(result_file)
    cfg = build_experiment_config(result, {"runner_mode": "docker"})
    assert cfg["runner_mode"] == "docker"
    assert cfg["tool"] == "vllm-bench-eval"


# --- review fixes ----------------------------------------------------------


def test_dataset_exists_maps_404_to_false():
    client = FakePlatformClient()
    assert dataset_exists(client, "nope") is False
    client.get_or_create_dataset("yes")
    assert dataset_exists(client, "yes") is True


def test_dataset_exists_propagates_other_errors():
    class Boom(FakePlatformClient):
        def get_dataset(self, name):
            raise RuntimeError("500 server error")

    with pytest.raises(RuntimeError, match="500"):
        dataset_exists(Boom(), "x")


def test_duplicate_sample_ids_are_refused():
    items = [
        {"id": "a", "sample_id": "s0000-deadbeef"},
        {"id": "b", "sample_id": "s0000-deadbeef"},
    ]
    with pytest.raises(SyncError, match="duplicate sample_id"):
        index_dataset_items(items)


def test_index_dataset_items_skips_foreign_items():
    items = [{"id": "a", "sample_id": "x"}, {"id": "b"}, {"id": "c", "sample_id": "y"}]
    assert index_dataset_items(items) == {"x": "a", "y": "c"}


def test_sync_refuses_to_create_an_empty_dataset(result_file, platform_cfg):
    """Non-custom dataset + unknown platform dataset would sync zero benchmark cases."""
    result = parse_result_file(result_file)
    client = FakePlatformClient()
    with pytest.raises(SyncError, match="no benchmark samples"):
        sync_result(result, [], platform_cfg, client=client)
    assert client.datasets == {}
    assert client.experiments == []


def test_sync_without_samples_is_allowed_against_an_existing_dataset(result_file, platform_cfg, captured):
    result = parse_result_file(result_file)
    client = FakePlatformClient()
    client.get_or_create_dataset("ds")
    report = sync_result(result, [], platform_cfg, client=client, captured=captured)
    assert report.traces_created == len(captured)
    assert report.experiment_items_created == 0


def test_failed_request_trace_has_no_zero_latency_scores(
    result_file, samples_file, platform_cfg, captured
):
    result = parse_result_file(result_file)
    captured[1].success = False
    captured[1].error = "boom"
    samples = load_samples(samples_file)
    client = FakePlatformClient()
    sync_result(result, samples, platform_cfg, client=client, captured=captured)

    failed = next(t for t in client.traces if t.metadata.get("error") == "boom")
    names = {s["name"] for s in failed.feedback_scores}
    assert names.isdisjoint({"TTFT(ms)", "端到端延迟(ms)", "TPOT(ms)", "输出吞吐(tokens/s)"})
    assert {"name": "请求成功", "value": 0.0, "reason": "1 = 请求成功完成"} in failed.feedback_scores
    assert "error" in failed.tags


def test_sdk_console_log_prefix_is_rebranded():
    """The SDK logs as "OPIK: ..."; users must see the platform name instead."""
    import logging

    from vllm_bench_eval.platform_sync import rebrand_sdk_logging

    sdk_logger = logging.getLogger("opik")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("OPIK: %(message)s"))
    sdk_logger.addHandler(handler)
    try:
        assert rebrand_sdk_logging("Benchmark 平台") is True
        assert handler.formatter._fmt == "Benchmark 平台: %(message)s"
        # idempotent: a second call finds nothing left to change
        assert rebrand_sdk_logging("Benchmark 平台") is False
    finally:
        sdk_logger.removeHandler(handler)


def test_rebrand_never_raises(monkeypatch):
    """Cosmetic only: a broken logging setup must not fail a sync."""
    import logging

    from vllm_bench_eval import platform_sync as mod

    real_get_logger = logging.getLogger

    def boom(name=None):
        if name == "opik":
            raise RuntimeError("logging is broken")
        return real_get_logger(name)

    monkeypatch.setattr(logging, "getLogger", boom)
    assert mod.rebrand_sdk_logging() is False
