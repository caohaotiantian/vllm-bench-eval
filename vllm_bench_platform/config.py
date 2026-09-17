"""Configuration model: a single YAML file plus environment overrides.

The YAML file is the *only* thing a user has to edit to run this tool against a
different inference server / Benchmark 平台 deployment.

Every field can also be overridden through an environment variable using the
pattern ``VBP_<SECTION>__<FIELD>`` (double underscore between section and
field), e.g. ``VBP_SERVER__MODEL=my-model`` or
``VBP_BENCHMARK_PLATFORM__URL=http://x/api/``. The underlying SDK's own
environment variables are honoured as fallbacks too (see ``_ALIASES`` below).

Unknown keys are rejected (``extra="forbid"``) so that a typo such as
``modle: foo`` fails loudly instead of silently falling back to the default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

ENV_PREFIX = "VBP_"

# Values that mean "unset" when they arrive through an environment variable.
_ENV_NULLS = {"", "null", "none", "~"}


class _Section(BaseModel):
    """Base for every config section: reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


class ServerConfig(_Section):
    """The OpenAI-compatible inference server under test."""

    base_url: str = "http://127.0.0.1:8000"
    endpoint: str = "/v1/completions"
    backend: str = "openai"  # openai | openai-chat | openai-audio | ...
    model: str = ""
    # HuggingFace tokenizer id, or a local directory holding tokenizer files
    # (useful for air-gapped servers). vLLM needs it to count tokens.
    tokenizer: str = ""
    api_key: Optional[str] = None
    # `--ready-check-timeout-sec` only exists since vLLM 0.10.2. Set to null to
    # omit the flag entirely when benchmarking with an older vLLM.
    ready_check_timeout_sec: Optional[int] = 60

    @field_validator("base_url")
    @classmethod
    def _normalise_base_url(cls, v: str) -> str:
        """`--base-url` must be the server root: the /v1 prefix lives in `endpoint`."""
        v = v.strip().rstrip("/")
        while v.endswith("/v1"):
            v = v[: -len("/v1")].rstrip("/")
        return v


class BenchmarkConfig(_Section):
    """`vllm bench serve` knobs."""

    dataset_name: str = "custom"
    dataset_path: str = "data/sonnet_sample.jsonl"
    num_prompts: int = 8
    # Requests per second, or "inf" to issue everything at once. A YAML number
    # (`request_rate: 10`) is accepted and stringified.
    request_rate: str = "inf"
    max_concurrency: Optional[int] = 1
    # Passed to `vllm bench serve --seed`. NOTE: it does *not* influence the
    # order in which a `custom` dataset is sampled (see samples.py).
    seed: int = 0
    custom_output_len: int = 128
    custom_skip_chat_template: bool = True
    ignore_eos: bool = False
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,99"
    result_dir: str = "results"
    # Extra raw CLI flags appended verbatim, e.g. ["--temperature", "0.0"].
    extra_args: List[str] = Field(default_factory=list)
    # Extra `--metadata K=V` entries recorded inside the result JSON. Non-string
    # YAML scalars are stringified.
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator("request_rate", mode="before")
    @classmethod
    def _stringify_rate(cls, v: Any) -> Any:
        if v is None:
            return "inf"
        if isinstance(v, bool):
            raise ValueError("benchmark.request_rate must be a number or 'inf'")
        if isinstance(v, (int, float)):
            return repr(float(v)) if not float(v).is_integer() else str(int(v))
        return v

    @field_validator("request_rate")
    @classmethod
    def _check_rate(cls, v: str) -> str:
        if v.strip().lower() == "inf":
            return "inf"
        try:
            rate = float(v)
        except ValueError:
            raise ValueError(f"benchmark.request_rate must be a number or 'inf', got {v!r}")
        if rate <= 0:
            raise ValueError("benchmark.request_rate must be > 0 (or 'inf')")
        return v

    @field_validator("metadata", mode="before")
    @classmethod
    def _stringify_metadata(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return {str(k): ("" if val is None else str(val)) for k, val in v.items()}
        return v

    @field_validator("percentile_metrics")
    @classmethod
    def _require_e2el(cls, v: str) -> str:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        for required in ("ttft", "tpot", "e2el"):
            if required not in parts:
                parts.append(required)
        return ",".join(parts)


class RunnerConfig(_Section):
    """How to execute `vllm bench serve`."""

    # `native` uses the vLLM installed on this machine (the portable default);
    # `docker` runs the benchmark client in a container (no local vLLM needed).
    mode: str = "native"
    # Capture per-request detail (prompts + wall-clock timestamps) into a
    # `<result>.requests.jsonl` sidecar. Required for traces with a real
    # timeline and prefill/decode spans. Set to false to fall back to a plain
    # `vllm bench serve` invocation (traces without timeline).
    capture: bool = True
    # native
    vllm_bin: str = "vllm"
    # Interpreter used for the capture launcher in native mode. It must be able
    # to `import vllm`. Defaults to the interpreter running this tool; point it
    # at the vLLM environment's python if they differ.
    python: Optional[str] = None
    # docker
    docker_image: str = "vllm-bench-client:0.11.0"
    docker_platform: Optional[str] = None      # e.g. "linux/arm64" on Apple Silicon
    docker_extra_args: List[str] = Field(default_factory=list)
    # Run the container as this uid:gid so that results/ and the HF cache are
    # not left root-owned. "auto" = current user on Linux, ignored on macOS
    # (Docker Desktop / OrbStack already map ownership back to the host user).
    docker_user: Optional[str] = "auto"
    # Host name the container uses to reach the host machine. Any
    # localhost/127.0.0.1 in server.base_url is rewritten to this in docker mode.
    host_gateway_alias: str = "host.docker.internal"
    # Persisted HuggingFace cache so the tokenizer is downloaded only once.
    hf_cache_dir: str = ".cache/huggingface"
    hf_token: Optional[str] = None
    timeout_sec: int = 3600

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("native", "docker"):
            raise ValueError("runner.mode must be 'native' or 'docker'")
        return v


class PlatformSettings(_Section):
    """Benchmark 平台 destination."""

    enabled: bool = True
    # The platform API address. A local deployment usually serves it on port
    # 5173; the trailing `/api/` is required.
    url: str = "http://localhost:5173/api/"
    workspace: str = "default"
    api_key: Optional[str] = None
    project_name: str = "vllm-bench"
    dataset_name: str = "vllm-bench-samples"
    # `{model}`, `{ts}`, `{backend}`, `{num_prompts}` are substituted.
    experiment_name: str = "vllm-bench-{model}-{ts}"
    tags: List[str] = Field(default_factory=lambda: ["vllm-bench"])


class PrepareConfig(_Section):
    """`prepare-dataset` defaults."""

    source: str = "sonnet"  # sonnet | sharegpt | local
    source_url: Optional[str] = None
    num_samples: int = 8
    seed: int = 1234
    output_path: str = "data/sonnet_sample.jsonl"
    # sonnet: how many consecutive lines form one prompt.
    lines_per_prompt: int = 8


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    benchmark_platform: PlatformSettings = Field(default_factory=PlatformSettings)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)

    # Absolute directory the relative paths above resolve against (the directory
    # containing the config file). Not settable from YAML.
    root_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    def path(self, relative: str) -> Path:
        p = Path(relative).expanduser()
        return p if p.is_absolute() else (self.root_dir / p)


_MODELS: Dict[str, type[BaseModel]] = {
    "server": ServerConfig,
    "benchmark": BenchmarkConfig,
    "runner": RunnerConfig,
    "benchmark_platform": PlatformSettings,
    "prepare": PrepareConfig,
}
_SECTIONS = tuple(_MODELS)

# The platform is built on the `opik` SDK, so its stock environment variables
# are accepted as fallbacks for the `benchmark_platform` section.
_ALIASES = {
    ("benchmark_platform", "url"): ["OPIK_URL_OVERRIDE"],
    ("benchmark_platform", "workspace"): ["OPIK_WORKSPACE"],
    ("benchmark_platform", "api_key"): ["OPIK_API_KEY"],
    ("benchmark_platform", "project_name"): ["OPIK_PROJECT_NAME"],
    ("runner", "hf_token"): ["HF_TOKEN"],
}


def _coerce(model_cls: type[BaseModel], field: str, raw: str) -> Any:
    """Turn an environment string into something pydantic will accept."""
    info = model_cls.model_fields.get(field)
    if info is None:
        return raw
    ann = str(info.annotation)
    optional = "Optional" in ann or "None" in ann
    if optional and raw.strip().lower() in _ENV_NULLS:
        return None
    if "bool" in ann:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if "List" in ann or "list" in ann or "Dict" in ann or "dict" in ann:
        try:
            return yaml.safe_load(raw)
        except Exception:
            return raw
    return raw


def _apply_env_overrides(data: Dict[str, Any], env: Dict[str, str]) -> Dict[str, Any]:
    for section in _SECTIONS:
        data.setdefault(section, {})

    # VBP_SECTION__FIELD=value
    for key, value in env.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        section, _, field = key[len(ENV_PREFIX):].partition("__")
        section = section.lower()
        field = field.lower()
        if section in _MODELS:
            data[section][field] = _coerce(_MODELS[section], field, value)

    # Well-known aliases (lower precedence than VBP_*).
    for (section, field), names in _ALIASES.items():
        if f"{ENV_PREFIX}{section.upper()}__{field.upper()}" in env:
            continue
        for name in names:
            if env.get(name):
                data[section][field] = _coerce(_MODELS[section], field, env[name])
                break
    return data


def load_config(
    path: Optional[str | Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> AppConfig:
    """Load the YAML config, apply environment overrides, return an AppConfig."""
    env = dict(os.environ if env is None else env)
    data: Dict[str, Any] = {}
    root = Path.cwd()

    if path is not None:
        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"config file not found: {cfg_path}")
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file must contain a YAML mapping: {cfg_path}")
        unknown = sorted(set(loaded) - set(_SECTIONS))
        if unknown:
            raise ValueError(
                f"unknown top-level key(s) in {cfg_path}: {', '.join(unknown)}; "
                f"expected one of: {', '.join(_SECTIONS)}"
            )
        data = loaded
        root = cfg_path.parent

    data = _apply_env_overrides(data, env)
    cfg = AppConfig(**data)
    cfg.root_dir = root
    return cfg


def find_default_config(start: Optional[Path] = None) -> Optional[Path]:
    """Look for config.yaml (then config.example.yaml) next to the CWD."""
    start = start or Path.cwd()
    for name in ("config.yaml", "config.yml", "config.example.yaml"):
        candidate = start / name
        if candidate.exists():
            return candidate
    return None
