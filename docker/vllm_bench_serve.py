#!/usr/bin/env python3
"""GPU-free entrypoint for `vllm bench serve`.

The stock ``vllm`` CLI builds the argument parser for *every* subcommand,
including ``vllm serve``, whose parser instantiates ``VllmConfig`` and therefore
needs a usable accelerator ("Failed to infer device type" on a CPU-only host).
The serving *benchmark* is a pure HTTP client and needs none of that, so this
wrapper calls ``vllm.benchmarks.serve`` directly with exactly the same CLI
arguments and the same ``main()`` that ``vllm bench serve`` would call.

Leading ``bench`` / ``serve`` tokens are accepted and ignored so the same
argument vector works with or without the wrapper.
"""

from __future__ import annotations

import argparse
import sys


def cli() -> int:
    from vllm.benchmarks.serve import add_cli_args, main

    argv = list(sys.argv[1:])
    while argv and argv[0] in ("bench", "serve"):
        argv.pop(0)

    parser = argparse.ArgumentParser(
        prog="vllm bench serve",
        description="Benchmark an online OpenAI-compatible serving endpoint.",
    )
    add_cli_args(parser)
    args = parser.parse_args(argv)
    main(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
