"""The per-request capture sidecar: parsing, ordering, warm-up exclusion."""

import json

import pytest

from vllm_bench_eval.capture import (
    MAX_ITL_VALUES,
    benchmark_window,
    itl_stats,
    load_sidecar,
    merge_with_result,
    sidecar_path_for,
)
from vllm_bench_eval.results import parse_result_file


def test_sidecar_path_sits_next_to_the_result():
    assert str(sidecar_path_for("results/run.json")) == "results/run.json.requests.jsonl"


def test_warmup_request_is_excluded(sidecar_file):
    """vLLM issues an un-measured warm-up request before the real loop."""
    measured = load_sidecar(sidecar_file)
    assert len(measured) == 3
    assert all(r.request_id is not None for r in measured)
    assert all(r.generated_text != "warm" for r in measured)

    with_warmup = load_sidecar(sidecar_file, include_warmup=True)
    assert len(with_warmup) == 4


def test_requests_are_ordered_by_request_id_not_capture_order(sidecar_file):
    """The sidecar is appended as requests *finish*; issue order is the id."""
    reqs = load_sidecar(sidecar_file)
    assert [r.request_id for r in reqs] == [
        "benchmark-serving0", "benchmark-serving1", "benchmark-serving2"
    ]
    assert [r.index for r in reqs] == [0, 1, 2]
    assert [r.prompt for r in reqs] == ["prompt-0", "prompt-2", "prompt-1"]


def test_metrics_are_derived_per_request(sidecar_file):
    first = load_sidecar(sidecar_file)[0]
    assert first.ttft_ms == pytest.approx(100.0)
    assert first.e2e_ms == pytest.approx(180.0)
    assert first.itl_ms == pytest.approx([20.0, 20.0, 20.0, 20.0])
    assert first.input_len == 10
    assert first.extra_body == {"temperature": 0.0}
    assert first.success


def test_wall_clock_window_is_real(sidecar_file):
    reqs = load_sidecar(sidecar_file)
    assert all(r.start_dt is not None and r.end_dt is not None for r in reqs)
    # distinct start times -> a meaningful timeline
    assert len({r.start_dt for r in reqs}) > 1
    for r in reqs:
        assert r.end_dt > r.start_dt
        assert r.first_token_dt is not None
        assert r.start_dt <= r.first_token_dt <= r.end_dt
    start, end = benchmark_window(reqs)
    assert start is not None and end is not None and end > start


def test_end_time_tracks_measured_latency_not_scheduling(sidecar_file):
    """wall_duration includes semaphore wait; end_time must follow `latency`."""
    first = load_sidecar(sidecar_file)[0]
    assert (first.end_wall - first.start_wall) == pytest.approx(0.18)


def test_merge_fills_token_counts_from_the_result(sidecar_file, result_file):
    """Streaming endpoints often omit `usage`, leaving output_tokens at 0."""
    reqs = load_sidecar(sidecar_file)
    assert all(not r.output_len for r in reqs)
    merged = merge_with_result(reqs, parse_result_file(result_file))
    assert [r.output_len for r in merged] == [5, 4, 3]
    assert merged[0].tpot_ms == pytest.approx(20.0)


def test_merge_falls_back_to_itl_count(sidecar_file):
    class NoRows:
        requests = []

    merged = merge_with_result(load_sidecar(sidecar_file), NoRows())
    assert [r.output_len for r in merged] == [5, 4, 3]  # len(itl) + 1


def test_feedback_scores_are_latency_only(captured):
    """Names are the Chinese display labels; TTFT/TPOT stay English."""
    names = {s["name"] for s in captured[0].feedback_scores()}
    assert names == {
        "TTFT(ms)", "TPOT(ms)", "端到端延迟(ms)", "输出吞吐(tokens/s)", "请求成功",
    }


def test_feedback_score_reasons_are_chinese(captured):
    reasons = {s["name"]: s["reason"] for s in captured[0].feedback_scores()}
    assert reasons["TTFT(ms)"] == "首 token 时间"
    assert reasons["TPOT(ms)"] == "每输出 token 时间（不含首 token）"


def test_failed_request_reports_only_success(sidecar_file):
    req = load_sidecar(sidecar_file)[0]
    req.success = False
    req.error = "boom"
    assert [s["name"] for s in req.feedback_scores()] == ["请求成功"]


def test_itl_stats():
    assert itl_stats([]) == {"count": 0}
    stats = itl_stats([10.0, 20.0, 30.0, 40.0])
    assert stats["count"] == 4
    assert stats["mean"] == 25.0
    assert stats["p50"] == 25.0
    assert stats["max"] == 40.0
    assert itl_stats([7.0])["p99"] == 7.0


def test_malformed_lines_are_skipped(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(
        '{"request_id": "r0", "is_warmup": false, "prompt": "a", "ttft": 0.1, "itl": [], "success": true}\n'
        "not json at all\n"
        "\n"
        "[1, 2, 3]\n",
        encoding="utf-8",
    )
    assert len(load_sidecar(p)) == 1


def test_itl_list_is_capped_in_trace_metadata(sidecar_file):
    from vllm_bench_eval.platform_sync import build_request_trace_payload
    from vllm_bench_eval.results import parse_result_dict

    req = load_sidecar(sidecar_file)[0]
    req.itl_ms = [1.0] * (MAX_ITL_VALUES + 50)
    payload = build_request_trace_payload(
        req,
        run_id="r", experiment_name="e",
        result=parse_result_dict({"model_id": "m", "completed": 1}),
        run_cfg={}, tags=[],
    )
    assert len(payload["metadata"]["itl_ms"]) == MAX_ITL_VALUES
    assert payload["metadata"]["itl_truncated"] is True
    assert payload["metadata"]["itl_stats"]["count"] == MAX_ITL_VALUES + 50
