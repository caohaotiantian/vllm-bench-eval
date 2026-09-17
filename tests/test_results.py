import math

import pytest

from vllm_bench_platform.results import parse_result_dict, parse_result_file


def test_aggregate_metrics(result_file):
    r = parse_result_file(result_file)
    a = r.aggregate
    assert a.duration_s == 4.0
    assert a.completed == 3
    assert a.num_prompts == 3
    assert a.output_throughput == 3.0
    assert a.total_token_throughput == 10.5
    assert a.get("ttft", "mean") == 150.0
    assert a.get("ttft", "p99") == 190.0
    assert a.get("tpot", "mean") == 25.0
    assert a.get("e2el", "mean") == 300.0
    assert a.get("e2el", "p99") == 400.0
    assert r.model_id == "demo-model"
    assert r.backend == "openai"
    assert r.tokenizer_id == "Qwen/Qwen3-8B"


def test_percentile_parsing_ignores_unrelated_keys(result_dict):
    result_dict["some_other_ms"] = 1.0
    result_dict["mean_unknown_ms"] = 2.0
    r = parse_result_dict(result_dict)
    assert set(r.aggregate.percentiles) == {"ttft", "tpot", "itl", "e2el"}


def test_per_request_metrics(result_file):
    r = parse_result_file(result_file)
    assert len(r.requests) == 3
    first = r.requests[0]
    assert first.index == 0
    assert first.input_len == 10
    assert first.output_len == 5
    assert first.ttft_ms == pytest.approx(100.0)
    # E2E == ttft + sum(itl); 0.10 + 4 * 0.02 = 0.18 s
    assert first.e2e_ms == pytest.approx(180.0)
    # TPOT == (e2e - ttft) / (output_len - 1)
    assert first.tpot_ms == pytest.approx(80.0 / 4)
    assert first.generated_text == "alpha out"
    assert first.error is None
    assert first.succeeded
    assert first.output_tps == pytest.approx(5 / 0.18)


def test_request_without_detailed_arrays(result_dict):
    for key in ("input_lens", "output_lens", "ttfts", "itls", "generated_texts", "errors"):
        result_dict.pop(key)
    r = parse_result_dict(result_dict)
    assert r.requests == []
    assert r.has_detailed is False
    # aggregate metrics still parse
    assert r.aggregate.output_throughput == 3.0


def test_failed_request_is_marked(result_dict):
    result_dict["errors"] = ["", "boom", ""]
    result_dict["output_lens"] = [5, 0, 3]
    r = parse_result_dict(result_dict)
    failed = r.requests[1]
    assert failed.error == "boom"
    assert not failed.succeeded
    assert {"name": "success", "value": 0.0, "reason": "1 = request completed"} in failed.feedback_scores()


def test_failed_request_publishes_no_zero_latencies(result_dict):
    """vLLM reports ttft=0 / itl=[] for a failed request; 0 ms must not be synced."""
    result_dict["errors"] = ["", "connection refused", ""]
    result_dict["output_lens"] = [5, 0, 3]
    result_dict["ttfts"] = [0.10, 0.0, 0.15]
    result_dict["itls"] = [[0.02] * 4, [], [0.01, 0.01]]
    r = parse_result_dict(result_dict)
    names = {s["name"] for s in r.requests[1].feedback_scores()}
    assert names.isdisjoint({"ttft_ms", "tpot_ms", "e2e_ms", "output_tokens_per_s"})
    assert "success" in names and "input_tokens" in names
    # successful siblings are unaffected
    assert "ttft_ms" in {s["name"] for s in r.requests[0].feedback_scores()}


def test_custom_percentiles_are_all_published(result_dict):
    """--metric-percentiles 75,99.9 must reach the platform, not just a hard-coded list."""
    for key in list(result_dict):
        if key.startswith(("p50_", "p90_", "p99_")):
            result_dict.pop(key)
    result_dict["p75_ttft_ms"] = 160.0
    result_dict["p99.9_e2el_ms"] = 999.0
    r = parse_result_dict(result_dict)
    names = {s["name"] for s in r.aggregate.feedback_scores()}
    assert "p75_ttft_ms" in names
    assert "p99.9_e2el_ms" in names


def test_single_token_output_has_no_tpot(result_dict):
    result_dict["output_lens"] = [1, 4, 3]
    r = parse_result_dict(result_dict)
    assert r.requests[0].tpot_ms is None
    names = {s["name"] for s in r.requests[0].feedback_scores()}
    assert "tpot_ms" not in names
    assert "ttft_ms" in names


def test_non_finite_values_are_dropped(result_dict):
    result_dict["mean_ttft_ms"] = float("nan")
    result_dict["output_throughput"] = float("inf")
    r = parse_result_dict(result_dict)
    assert r.aggregate.get("ttft", "mean") is None
    assert r.aggregate.output_throughput is None
    assert all(math.isfinite(s["value"]) for s in r.aggregate.feedback_scores())


def test_request_feedback_scores(result_file):
    r = parse_result_file(result_file)
    scores = {s["name"]: s["value"] for s in r.requests[0].feedback_scores()}
    assert set(scores) == {
        "ttft_ms", "tpot_ms", "e2e_ms", "output_tokens_per_s",
        "output_tokens", "input_tokens", "success",
    }
    assert scores["ttft_ms"] == pytest.approx(100.0)


def test_aggregate_feedback_scores_cover_the_target_metrics(result_file):
    r = parse_result_file(result_file)
    names = {s["name"] for s in r.aggregate.feedback_scores()}
    for required in (
        "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms",
        "output_throughput_tps", "total_token_throughput_tps",
        "mean_e2el_ms", "duration_s", "request_throughput_rps",
    ):
        assert required in names


def test_run_config_excludes_arrays(result_file):
    r = parse_result_file(result_file)
    cfg = r.run_config()
    assert "ttfts" not in cfg and "generated_texts" not in cfg
    assert cfg["model_id"] == "demo-model"
    assert cfg["max_concurrency"] == 2


def test_as_dict_flattens_percentiles(result_file):
    d = parse_result_file(result_file).aggregate.as_dict()
    assert d["mean_ttft_ms"] == 150.0
    assert d["p99_e2el_ms"] == 400.0
    assert d["output_throughput_tps"] == 3.0
