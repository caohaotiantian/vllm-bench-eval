import json
from pathlib import Path

import pytest

RESULT_FIXTURE = {
    "date": "20260917-120000",
    "endpoint_type": "openai",
    "backend": "openai",
    "label": None,
    "model_id": "demo-model",
    "tokenizer_id": "Qwen/Qwen3-8B",
    "num_prompts": 3,
    "request_rate": "inf",
    "burstiness": 1.0,
    "max_concurrency": 2,
    "duration": 4.0,
    "completed": 3,
    "total_input_tokens": 30,
    "total_output_tokens": 12,
    "request_throughput": 0.75,
    "request_goodput": None,
    "output_throughput": 3.0,
    "total_token_throughput": 10.5,
    "max_output_tokens_per_s": 6.0,
    "max_concurrent_requests": 2.0,
    "mean_ttft_ms": 150.0,
    "median_ttft_ms": 140.0,
    "std_ttft_ms": 20.0,
    "p50_ttft_ms": 140.0,
    "p99_ttft_ms": 190.0,
    "mean_tpot_ms": 25.0,
    "p99_tpot_ms": 30.0,
    "mean_itl_ms": 24.0,
    "p99_itl_ms": 31.0,
    "mean_e2el_ms": 300.0,
    "p99_e2el_ms": 400.0,
    # --save-detailed arrays, aligned by index with the issued requests
    "input_lens": [10, 11, 9],
    "output_lens": [5, 4, 3],
    "ttfts": [0.10, 0.20, 0.15],
    "itls": [[0.02, 0.02, 0.02, 0.02], [0.03, 0.03, 0.03], [0.01, 0.01]],
    "generated_texts": ["alpha out", "beta out", "gamma out"],
    "errors": ["", "", ""],
}


@pytest.fixture
def result_dict():
    return json.loads(json.dumps(RESULT_FIXTURE))


@pytest.fixture
def result_file(tmp_path: Path, result_dict) -> Path:
    p = tmp_path / "result.json"
    p.write_text(json.dumps(result_dict), encoding="utf-8")
    return p


@pytest.fixture
def samples_file(tmp_path: Path) -> Path:
    p = tmp_path / "samples.jsonl"
    p.write_text(
        "\n".join(json.dumps({"prompt": f"prompt-{i}"}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    return p


# --- per-request capture sidecar --------------------------------------------

SIDECAR_ROWS = [
    # warm-up: vLLM builds it without a request_id
    {
        "schema": 1, "capture_index": 0, "request_id": None, "is_warmup": True,
        "prompt": "prompt-0", "model": "demo-model", "api_url": "http://h:1/v1/completions",
        "prompt_len": 10, "output_len": 128, "ignore_eos": False,
        "extra_body": {"temperature": 0.0},
        "start_wall": 1_700_000_000.0, "end_wall": 1_700_000_001.0,
        "success": True, "latency": 1.0, "ttft": 0.5, "itl": [0.1],
        "generated_text": "warm", "output_tokens": 0, "error": "",
    },
    # measured requests, deliberately out of file order in the sidecar
    {
        "schema": 1, "capture_index": 2, "request_id": "benchmark-serving1", "is_warmup": False,
        "prompt": "prompt-2", "model": "demo-model", "api_url": "http://h:1/v1/completions",
        "prompt_len": 11, "output_len": 128, "ignore_eos": False,
        "extra_body": {"temperature": 0.0},
        "start_wall": 1_700_000_010.0, "end_wall": 1_700_000_010.5,
        "success": True, "latency": 0.28, "ttft": 0.2, "itl": [0.03, 0.03, 0.02],
        "generated_text": "beta out", "output_tokens": 0, "error": "",
    },
    {
        "schema": 1, "capture_index": 1, "request_id": "benchmark-serving0", "is_warmup": False,
        "prompt": "prompt-0", "model": "demo-model", "api_url": "http://h:1/v1/completions",
        "prompt_len": 10, "output_len": 128, "ignore_eos": False,
        "extra_body": {"temperature": 0.0},
        "start_wall": 1_700_000_010.0, "end_wall": 1_700_000_010.4,
        "success": True, "latency": 0.18, "ttft": 0.1, "itl": [0.02, 0.02, 0.02, 0.02],
        "generated_text": "alpha out", "output_tokens": 0, "error": "",
    },
    {
        "schema": 1, "capture_index": 3, "request_id": "benchmark-serving2", "is_warmup": False,
        "prompt": "prompt-1", "model": "demo-model", "api_url": "http://h:1/v1/completions",
        "prompt_len": 9, "output_len": 128, "ignore_eos": False,
        "extra_body": {"temperature": 0.0},
        "start_wall": 1_700_000_011.0, "end_wall": 1_700_000_011.2,
        "success": True, "latency": 0.17, "ttft": 0.15, "itl": [0.01, 0.01],
        "generated_text": "gamma out", "output_tokens": 0, "error": "",
    },
]


@pytest.fixture
def sidecar_file(tmp_path: Path) -> Path:
    p = tmp_path / "result.json.requests.jsonl"
    p.write_text(
        "\n".join(json.dumps(row) for row in SIDECAR_ROWS) + "\n", encoding="utf-8"
    )
    return p


@pytest.fixture
def captured(sidecar_file, result_file):
    """Sidecar requests merged with the result arrays, as the CLI does."""
    from vllm_bench_eval.capture import load_sidecar, merge_with_result
    from vllm_bench_eval.results import parse_result_file

    return merge_with_result(load_sidecar(sidecar_file), parse_result_file(result_file))
