#!/usr/bin/env python3
"""GPU-free launcher for `vllm bench serve`, with per-request capture.

Two jobs:

1. **Run the benchmark without a GPU.** The stock ``vllm`` CLI builds the
   argument parser for *every* subcommand, including ``vllm serve``, whose
   parser instantiates ``VllmConfig`` and therefore needs a usable accelerator
   ("Failed to infer device type" on a CPU-only host). The serving *benchmark*
   is a pure HTTP client, so this module calls ``vllm.benchmarks.serve``
   directly with the same CLI args and the same ``main()``.

2. **Capture per-request detail that the result JSON throws away.**
   ``--save-detailed`` gives us aligned arrays of ttft/itl/output_lens, but no
   prompts, no wall-clock timestamps and no request ids — so a trace timeline
   cannot be reconstructed. With ``--vbp-capture <path>`` every function in
   ``vllm.benchmarks.lib.endpoint_request_func.ASYNC_REQUEST_FUNCS`` is wrapped
   so that each call appends one JSON line describing the request and its
   result, including wall-clock start/end.

Notes on the capture:

* ``ASYNC_REQUEST_FUNCS`` is a module-level dict that ``serve.py`` imported by
  reference and indexes *inside* ``benchmark()``, so mutating it in place is
  enough — no monkeypatching of ``serve`` itself.
* ``RequestFuncOutput.start_time`` is a ``time.perf_counter()`` value, which is
  not wall-clock. We therefore record ``time.time()`` ourselves on entry/exit.
* ``serve.benchmark()`` issues a warm-up request (and optionally a profiler
  request) *before* the measured loop. Those are built without a ``request_id``
  while every measured request carries one, so ``request_id is None`` is a
  precise, version-stable discriminator. Entries are flagged rather than
  dropped, so the consumer can assert the count itself.

This module deliberately imports nothing beyond the standard library and vLLM,
so it can be executed inside a slim benchmark container that has no pydantic /
typer / pyyaml installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

CAPTURE_FLAG = "--vbp-capture"
SCHEMA_VERSION = 1


class _Recorder:
    """Appends one JSON line per request to a sidecar file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._seq = 0
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        # Truncate: one sidecar per benchmark run.
        with open(self.path, "w", encoding="utf-8"):
            pass

    def next_index(self) -> int:
        with self._lock:
            i = self._seq
            self._seq += 1
            return i

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def _as_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(v) for v in value]
    return str(value)


def _describe_input(req_in: Any) -> Dict[str, Any]:
    """Pull the interesting fields off a RequestFuncInput."""
    out: Dict[str, Any] = {}
    for field in (
        "prompt", "api_url", "prompt_len", "output_len", "model", "model_name",
        "logprobs", "ignore_eos", "language", "request_id",
    ):
        out[field] = _as_jsonable(getattr(req_in, field, None))
    out["extra_body"] = _as_jsonable(getattr(req_in, "extra_body", None))
    mm = getattr(req_in, "multi_modal_content", None)
    out["has_multi_modal"] = mm is not None
    return out


def _describe_output(req_out: Any) -> Dict[str, Any]:
    itl = list(getattr(req_out, "itl", None) or [])
    return {
        "success": bool(getattr(req_out, "success", False)),
        "latency": _as_jsonable(getattr(req_out, "latency", None)),
        "ttft": _as_jsonable(getattr(req_out, "ttft", None)),
        "tpot": _as_jsonable(getattr(req_out, "tpot", None)),
        "itl": [float(x) for x in itl],
        "generated_text": getattr(req_out, "generated_text", "") or "",
        "output_tokens": _as_jsonable(getattr(req_out, "output_tokens", None)),
        "prompt_len": _as_jsonable(getattr(req_out, "prompt_len", None)),
        "error": getattr(req_out, "error", "") or "",
        "perf_start_time": _as_jsonable(getattr(req_out, "start_time", None)),
    }


def install_capture(capture_path: str) -> _Recorder:
    """Wrap every entry of ASYNC_REQUEST_FUNCS so calls are recorded."""
    from vllm.benchmarks.lib import endpoint_request_func as erf

    recorder = _Recorder(capture_path)

    def make_wrapper(original):
        async def wrapper(*args, **kwargs):
            req_in = kwargs.get("request_func_input")
            if req_in is None and args:
                req_in = args[0]
            index = recorder.next_index()
            start_wall = time.time()
            start_perf = time.perf_counter()
            record: Dict[str, Any] = {
                "schema": SCHEMA_VERSION,
                "capture_index": index,
                "start_wall": start_wall,
            }
            try:
                record.update(_describe_input(req_in))
            except Exception as exc:  # never break the benchmark
                record["capture_error"] = f"input: {exc}"

            try:
                result = await original(*args, **kwargs)
            except Exception as exc:
                record["end_wall"] = time.time()
                record["wall_duration"] = time.time() - start_wall
                record["perf_duration"] = time.perf_counter() - start_perf
                record["success"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["is_warmup"] = record.get("request_id") is None
                recorder.write(record)
                raise

            end_wall = time.time()
            record["end_wall"] = end_wall
            record["wall_duration"] = end_wall - start_wall
            record["perf_duration"] = time.perf_counter() - start_perf
            try:
                record.update(_describe_output(result))
            except Exception as exc:
                record["capture_error"] = f"output: {exc}"
            # Warm-up / profiler requests are built without a request_id;
            # every measured request carries one.
            record["is_warmup"] = record.get("request_id") is None
            recorder.write(record)
            return result

        return wrapper

    # Mutate in place: serve.py holds a reference to this very dict.
    for key, func in list(erf.ASYNC_REQUEST_FUNCS.items()):
        erf.ASYNC_REQUEST_FUNCS[key] = make_wrapper(func)
    return recorder


def split_argv(argv: List[str]) -> tuple[Optional[str], List[str]]:
    """Pull our own ``--vbp-capture PATH`` out of the vLLM argument vector."""
    capture: Optional[str] = None
    rest: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == CAPTURE_FLAG:
            if i + 1 >= len(argv):
                raise SystemExit(f"{CAPTURE_FLAG} requires a path argument")
            capture = argv[i + 1]
            i += 2
            continue
        if arg.startswith(CAPTURE_FLAG + "="):
            capture = arg.split("=", 1)[1]
            i += 1
            continue
        # Leading `bench` / `serve` tokens are accepted and ignored so the same
        # argument vector works with or without this launcher.
        if not rest and arg in ("bench", "serve"):
            i += 1
            continue
        rest.append(arg)
        i += 1
    return capture, rest


def cli(argv: Optional[List[str]] = None) -> int:
    capture_path, args_list = split_argv(list(sys.argv[1:] if argv is None else argv))

    from vllm.benchmarks.serve import add_cli_args, main

    if capture_path:
        install_capture(capture_path)

    parser = argparse.ArgumentParser(
        prog="vllm bench serve",
        description="Benchmark an online OpenAI-compatible serving endpoint.",
    )
    add_cli_args(parser)
    args = parser.parse_args(args_list)
    main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
