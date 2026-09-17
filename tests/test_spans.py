"""Span construction and prompt-based alignment."""

import pytest

from vllm_bench_platform.capture import load_sidecar
from vllm_bench_platform.platform_sync import (
    SPAN_DECODE,
    SPAN_PREFILL,
    build_request_spans,
    build_request_trace_payload,
)
from vllm_bench_platform.results import parse_result_file
from vllm_bench_platform.samples import align_by_prompt, load_samples


# --- prompt alignment -------------------------------------------------------


def test_align_by_prompt_matches_exact_text(captured, samples_file):
    samples = load_samples(samples_file)
    matched, extra = align_by_prompt(captured, samples)
    assert matched == 3
    assert extra == []
    by_prompt = {s.prompt: s.sample_id for s in samples}
    for req in captured:
        assert req.sample_id == by_prompt[req.prompt]


def test_align_by_prompt_needs_no_seed_replay(captured, samples_file):
    """Issue order is a shuffle of file order; prompt matching handles it."""
    samples = load_samples(samples_file)
    align_by_prompt(captured, samples)
    assert [r.prompt for r in captured] == ["prompt-0", "prompt-2", "prompt-1"]
    assert [r.sample_id for r in captured] == [
        samples[0].sample_id, samples[2].sample_id, samples[1].sample_id
    ]


def test_align_by_prompt_creates_items_for_unknown_prompts(captured):
    """sharegpt / random runs have no local JSONL, but the capture has prompts."""
    matched, extra = align_by_prompt(captured, [])
    assert matched == 3
    assert len(extra) == 3
    assert {s.prompt for s in extra} == {"prompt-0", "prompt-1", "prompt-2"}
    assert len({s.sample_id for s in extra}) == 3
    assert all(r.sample_id is not None for r in captured)


def test_align_by_prompt_dedupes_repeated_prompts(captured):
    captured[1].prompt = captured[0].prompt          # oversampled duplicate
    matched, extra = align_by_prompt(captured, [])
    assert matched == 3
    assert len(extra) == 2                           # deduped by content
    assert captured[0].sample_id == captured[1].sample_id


# --- spans ------------------------------------------------------------------


@pytest.fixture
def spans(captured):
    return build_request_spans(
        captured[0], trace_id="tr-1", model="demo-model",
        provider="openai", project_name="proj",
    )


def test_two_spans_split_at_the_first_token(spans, captured):
    assert [s["name"] for s in spans] == [SPAN_PREFILL, SPAN_DECODE]
    req = captured[0]
    prefill, decode = spans
    assert prefill["start_time"] == req.start_dt
    assert prefill["end_time"] == req.first_token_dt == decode["start_time"]
    assert decode["end_time"] == req.end_dt
    # prefill duration == TTFT
    assert (prefill["end_time"] - prefill["start_time"]).total_seconds() == pytest.approx(
        req.ttft_ms / 1000.0, abs=1e-6
    )


def test_spans_are_llm_typed_with_usage(spans):
    prefill, decode = spans
    assert prefill["type"] == "llm" and decode["type"] == "llm"
    assert decode["usage"] == {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15
    }
    assert decode["model"] == "demo-model" and decode["provider"] == "openai"
    assert prefill["trace_id"] == decode["trace_id"] == "tr-1"
    assert prefill["project_name"] == "proj"


def test_span_metadata_carries_the_phase_detail(spans):
    prefill, decode = spans
    assert prefill["metadata"]["phase"] == "prefill"
    assert prefill["metadata"]["ttft_ms"] == pytest.approx(100.0)
    assert decode["metadata"]["phase"] == "decode"
    assert decode["metadata"]["itl_stats"]["count"] == 4
    assert decode["metadata"]["decode_ms"] == pytest.approx(80.0)


def test_no_spans_without_a_timeline(captured):
    req = captured[0]
    req.start_wall = None
    assert build_request_spans(
        req, trace_id="t", model="m", provider="p", project_name="proj"
    ) == []


def test_no_spans_without_a_first_token(captured):
    req = captured[0]
    req.ttft_ms = None
    assert build_request_spans(
        req, trace_id="t", model="m", provider="p", project_name="proj"
    ) == []


# --- trace payload ----------------------------------------------------------


def test_trace_payload_metadata_keys(captured, result_file, samples_file):
    samples = load_samples(samples_file)
    align_by_prompt(captured, samples)
    result = parse_result_file(result_file)
    payload = build_request_trace_payload(
        captured[0], run_id="run123", experiment_name="exp-1", result=result,
        run_cfg={"endpoint": "/v1/completions", "base_url": "http://h:1"},
        tags=["vllm-bench", "demo-model"],
    )
    md = payload["metadata"]
    for key in (
        "run_id", "experiment_name", "model_id", "tokenizer_id", "backend",
        "endpoint", "base_url", "request_rate", "max_concurrency",
        "request_index", "sample_id", "input_tokens", "output_tokens",
        "ttft_ms", "tpot_ms", "e2e_ms", "itl_stats", "itl_ms",
    ):
        assert key in md, key
    assert md["run_id"] == "run123"
    assert md["endpoint"] == "/v1/completions"
    assert md["max_concurrency"] == 2
    assert md["sampling_params"] == {"temperature": 0.0}
    assert payload["name"] == "req-0000"
    assert payload["start_time"] is not None and payload["end_time"] is not None
