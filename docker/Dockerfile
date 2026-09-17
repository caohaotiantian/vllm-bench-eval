# Benchmark *client* image for `vllm bench serve`.
#
# vLLM ships manylinux2014_aarch64 and manylinux1_x86_64 wheels, so this image
# builds on Apple Silicon (linux/arm64) as well as on x86_64 hosts. No GPU and
# no model weights are required: only vLLM's client-side benchmarking code runs
# here, against a remote OpenAI-compatible server.
#
#   docker build -t vllm-bench-client:0.11.0 -f docker/Dockerfile docker/
#
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ARG VLLM_VERSION=0.11.0
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLLM_LOGGING_LEVEL=WARNING \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/hf-cache

# `[bench]` pulls pandas/datasets, which vllm.benchmarks.datasets needs.
#
# transformers is pinned below v5: vLLM 0.11.0 only declares
# `transformers>=4.55.2`, but still uses `all_special_tokens_extended`, which
# Transformers v5 removed ("Qwen2Tokenizer has no attribute
# all_special_tokens_extended" when loading the benchmark tokenizer).
RUN pip install "vllm[bench]==${VLLM_VERSION}" "transformers<5"

# libgomp1 is required by torch/torchaudio which transformers imports eagerly.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY vllm_bench_serve.py /usr/local/bin/vllm-bench-serve
RUN chmod +x /usr/local/bin/vllm-bench-serve

WORKDIR /work
ENTRYPOINT []
CMD ["vllm-bench-serve", "--help"]
