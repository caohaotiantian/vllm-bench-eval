"""Build and execute the `vllm bench serve` command (natively or in Docker)."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from .capture import sidecar_path_for
from .config import AppConfig

# Where the tool's own package is mounted inside the benchmark container, so
# that `python -m vllm_bench_platform.vllm_entry` works there too. The package
# root goes on PYTHONPATH; vllm_entry.py imports only stdlib + vllm.
CONTAINER_PKG_ROOT = "/opt/vbp"

# ---------------------------------------------------------------------------
# vLLM capability detection
#
# `vllm bench serve` grew flags over time, so hard-coding the 0.11.0 set makes
# the tool crash on an older install with e.g.
#   vllm bench serve: error: unrecognized arguments: --custom-skip-chat-template
# We therefore ask the target vLLM once what it accepts and drop what it does
# not know, warning about the behaviour that changes.
#
# Known introduction points (verified against the PyPI sdists):
#   --save-detailed              0.9.1
#   --custom-output-len          0.9.2
#   --custom-skip-chat-template  0.9.2
#   --ready-check-timeout-sec    0.10.1
#   --backend                    0.9.2  (older releases only have --endpoint-type)
# ---------------------------------------------------------------------------

# Flags that were renamed: try the alternatives before giving up on a flag.
FLAG_ALIASES = {
    # `--backend` was introduced in 0.9.2; 0.9.0/0.9.1 only accept the older
    # `--endpoint-type` spelling for the very same value.
    "--backend": ("--endpoint-type",),
}

# Present since at least 0.9.0; without these the benchmark cannot run at all.
REQUIRED_FLAGS = frozenset({
    "--backend", "--base-url", "--endpoint", "--model",
    "--dataset-name", "--dataset-path", "--num-prompts",
    "--save-result", "--result-dir", "--result-filename",
})

# Flag -> what the user loses when the installed vLLM does not support it.
OPTIONAL_FLAG_CONSEQUENCES = {
    "--save-detailed": (
        "the result JSON will have no per-request arrays; per-request metrics "
        "come from the capture sidecar only"
    ),
    "--custom-skip-chat-template": (
        "the tokenizer's chat template will be applied to every prompt, so the "
        "text sent to the server differs from the dataset prompt"
    ),
    "--custom-output-len": "the dataset's default output length will be used",
    "--ready-check-timeout-sec": "the endpoint ready-check timeout falls back to the vLLM default",
    "--percentile-metrics": "E2E (e2el) aggregates will be missing from the result",
    "--metric-percentiles": "the default percentiles will be reported",
    "--ignore-eos": "generation may stop early at EOS",
    "--max-concurrency": "concurrency will be unbounded",
    "--request-rate": "all requests are issued at once",
    "--metadata": "extra metadata will not be embedded in the result JSON",
    "--disable-tqdm": "a progress bar will be printed",
    "--seed": "sampling is not seeded",
    "--tokenizer": "vLLM will use the model id as the tokenizer",
}

_FLAG_RE = re.compile(r"(--[a-zA-Z0-9][a-zA-Z0-9-]*)")
# `--dataset-name {sharegpt,burstgpt,sonnet,random,hf}` in an argparse help dump.
_DATASET_CHOICES_RE = re.compile(r"--dataset-name\s*\{([^}]*)\}")


@dataclass(frozen=True)
class VllmCapabilities:
    """What the target vLLM's `bench serve` accepts."""

    flags: frozenset = frozenset()
    version: Optional[str] = None
    # Accepted `--dataset-name` values; empty means "unknown, do not check".
    dataset_choices: frozenset = frozenset()
    # False => detection failed; callers then emit every flag (old behaviour).
    detected: bool = False
    note: Optional[str] = None

    def supports(self, flag: str) -> bool:
        # Undetected capabilities must not silently strip flags.
        return True if not self.detected else flag in self.flags

    def supports_dataset(self, name: str) -> bool:
        if not self.detected or not self.dataset_choices:
            return True
        return name in self.dataset_choices


def parse_supported_flags(help_text: str) -> frozenset:
    """Pull the `--flag` tokens out of an argparse help dump."""
    return frozenset(_FLAG_RE.findall(help_text or ""))


def parse_dataset_choices(help_text: str) -> frozenset:
    """Pull the accepted `--dataset-name` values out of an argparse help dump."""
    match = _DATASET_CHOICES_RE.search((help_text or "").replace("\n", " "))
    if not match:
        return frozenset()
    return frozenset(part.strip() for part in match.group(1).split(",") if part.strip())


def _last_json_object(text: str) -> Optional[Dict]:
    """vLLM prints warnings around our output; take the last JSON line."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# Process-wide cache: probing costs a container start.
_CAPABILITY_CACHE: Dict[tuple, VllmCapabilities] = {}


def _capability_cache_key(cfg: AppConfig) -> tuple:
    return (
        cfg.runner.mode,
        cfg.runner.capture,
        cfg.runner.docker_image,
        cfg.runner.docker_platform,
        cfg.runner.python or sys.executable,
        cfg.runner.vllm_bin,
    )


def _probe_argv(cfg: AppConfig) -> List[str]:
    """Argv that makes the target vLLM report its flags (and version)."""
    if cfg.runner.mode == "native":
        if cfg.runner.capture:
            python = cfg.runner.python or sys.executable
            return [python, "-m", "vllm_bench_platform.vllm_entry", "--vbp-probe"]
        return [cfg.runner.vllm_bin, "bench", "serve", "--help"]

    docker: List[str] = ["docker", "run", "--rm"]
    if cfg.runner.docker_platform:
        docker += ["--platform", cfg.runner.docker_platform]
    if cfg.runner.capture:
        pkg_dir = Path(__file__).resolve().parent
        docker += [
            "-v", f"{pkg_dir}:{CONTAINER_PKG_ROOT}/{pkg_dir.name}:ro",
            "-e", f"PYTHONPATH={CONTAINER_PKG_ROOT}",
            cfg.runner.docker_image,
            "python", "-m", "vllm_bench_platform.vllm_entry", "--vbp-probe",
        ]
    else:
        docker += [cfg.runner.docker_image, "vllm-bench-serve", "--help"]
    return docker


def detect_capabilities(
    cfg: AppConfig,
    use_cache: bool = True,
    timeout: int = 300,
) -> VllmCapabilities:
    """Ask the target vLLM which `bench serve` flags it supports.

    Falls back to ``detected=False`` (emit everything, warn) when the probe
    cannot be run, so a probe failure never blocks a benchmark.
    """
    key = _capability_cache_key(cfg)
    if use_cache and key in _CAPABILITY_CACHE:
        return _CAPABILITY_CACHE[key]

    argv = _probe_argv(cfg)
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, env=build_env(cfg)
        )
        merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
        payload = _last_json_object(proc.stdout or "")
        if payload and payload.get("flags"):
            flags = frozenset(payload["flags"])
            version = payload.get("vllm_version")
            datasets = frozenset(payload.get("dataset_choices") or ())
        else:
            flags = parse_supported_flags(merged)
            version = _native_version(cfg)
            datasets = parse_dataset_choices(merged)

        # Only trust the probe when it produced something that actually looks
        # like `bench serve --help`. A failed `docker run` prints its own usage
        # text, and scraping `--flags` out of that would strip every real flag.
        missing_core = REQUIRED_FLAGS - flags
        if flags and not missing_core:
            caps = VllmCapabilities(
                flags=flags, version=version, dataset_choices=datasets, detected=True
            )
        else:
            reason = (
                f"probe exited {proc.returncode} without core flags "
                f"(missing {sorted(missing_core)[:3]})"
                if flags
                else f"probe produced no flags (exit {proc.returncode})"
            )
            caps = VllmCapabilities(detected=False, note=reason)
    except Exception as exc:
        caps = VllmCapabilities(detected=False, note=f"{type(exc).__name__}: {exc}")

    if use_cache:
        _CAPABILITY_CACHE[key] = caps
    return caps


def _native_version(cfg: AppConfig) -> Optional[str]:
    """Best-effort `vllm.__version__` for the non-capture native path."""
    if cfg.runner.mode != "native":
        return None
    python = cfg.runner.python or sys.executable
    try:
        proc = subprocess.run(
            [python, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True, timeout=120,
        )
        value = (proc.stdout or "").strip().splitlines()
        return value[-1] if value else None
    except Exception:
        return None

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}

# Environment variables whose *values* must never be printed.
SECRET_ENV_NAMES = ("HF_TOKEN", "OPENAI_API_KEY")
REDACTED = "<redacted>"


class RunnerError(RuntimeError):
    pass


@dataclass
class RunOutcome:
    command: List[str]
    returncode: int
    # stdout and stderr of the child are merged into `output` so that the live
    # stream keeps its original interleaving.
    output: str
    result_path: Path
    duration_s: float
    container_name: Optional[str] = None
    capture_path: Optional[Path] = None
    vllm_version: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class BenchCommand:
    """Everything needed to launch one benchmark."""

    argv: List[str]
    result_path: Path
    env: Dict[str, str] = field(default_factory=dict)
    container_name: Optional[str] = None
    # Host path of the per-request capture sidecar (None when capture is off).
    capture_path: Optional[Path] = None
    capabilities: Optional["VllmCapabilities"] = None
    warnings: List[str] = field(default_factory=list)

    def display(self) -> str:
        return " ".join(shlex.quote(part) for part in redact_command(self.argv))


def redact_command(command: List[str]) -> List[str]:
    """Replace secret values in an argv before showing it to a human.

    Handles both `-e NAME=value` (docker) and a bare `NAME=value` token.
    """
    out: List[str] = []
    for part in command:
        redacted = part
        for name in SECRET_ENV_NAMES:
            prefix = f"{name}="
            if part == name:
                break
            if part.startswith(prefix) and len(part) > len(prefix):
                redacted = prefix + REDACTED
                break
        out.append(redacted)
    return out


def rewrite_for_container(base_url: str, alias: str) -> str:
    """Point a localhost URL at the Docker host gateway."""
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    if host not in LOCAL_HOSTS:
        return base_url
    netloc = alias
    if parts.port:
        netloc = f"{alias}:{parts.port}"
    if parts.username:
        cred = parts.username + (f":{parts.password}" if parts.password else "")
        netloc = f"{cred}@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def default_result_filename(cfg: AppConfig, timestamp: Optional[str] = None) -> str:
    ts = timestamp or time.strftime("%Y%m%d-%H%M%S")
    model = (cfg.server.model or "model").replace("/", "_")
    return f"vllm-bench-{model}-{ts}.json"


def resolve_docker_user(setting: Optional[str]) -> Optional[str]:
    """`--user` value for `docker run`, or None to leave it to the image.

    On Linux a root container would leave root-owned files in the bind-mounted
    results/ and HF cache directories. macOS (Docker Desktop / OrbStack) maps
    ownership back to the calling user already, so "auto" is a no-op there.
    """
    if not setting:
        return None
    if setting != "auto":
        return setting
    if platform.system() != "Linux":
        return None
    return f"{os.getuid()}:{os.getgid()}"


def build_bench_args(
    cfg: AppConfig,
    base_url: str,
    dataset_path: str,
    result_dir: str,
    result_filename: str,
    caps: Optional[VllmCapabilities] = None,
    warnings: Optional[List[str]] = None,
) -> List[str]:
    """The `vllm bench serve` flags, identical for native and docker mode.

    Flags the detected vLLM does not accept are dropped (with a warning naming
    what changes); a *required* flag that is missing is an error, because the
    benchmark would not produce a usable result without it.
    """
    caps = caps or VllmCapabilities()
    warnings = warnings if warnings is not None else []
    b = cfg.benchmark
    args: List[str] = []

    def emit(flag: str, *values: str) -> bool:
        if caps.supports(flag):
            args.append(flag)
            args.extend(values)
            return True
        # Renamed flag? Use the spelling this vLLM does know.
        for alias in FLAG_ALIASES.get(flag, ()):
            if caps.supports(alias):
                warnings.append(f"{flag} not supported by this vLLM -> using {alias} instead")
                args.append(alias)
                args.extend(values)
                return True
        if flag in REQUIRED_FLAGS:
            raise RunnerError(
                f"the installed vLLM"
                + (f" ({caps.version})" if caps.version else "")
                + f" does not support the required flag {flag}; "
                f"upgrade vLLM (>= 0.9.2 recommended) to run this benchmark"
            )
        consequence = OPTIONAL_FLAG_CONSEQUENCES.get(flag, "that option is ignored")
        warnings.append(f"{flag} not supported by this vLLM -> {consequence}")
        return False

    emit("--backend", cfg.server.backend)
    emit("--base-url", base_url)
    emit("--endpoint", cfg.server.endpoint)
    emit("--model", cfg.server.model)
    if not caps.supports_dataset(b.dataset_name):
        raise RunnerError(
            f"the installed vLLM"
            + (f" ({caps.version})" if caps.version else "")
            + f" does not support benchmark.dataset_name={b.dataset_name!r}; "
            f"it accepts {sorted(caps.dataset_choices)}. "
            f"The `custom` dataset (needed to benchmark your own prompts) was "
            f"added in vLLM 0.9.2 — upgrade, or pick one of the supported names."
        )
    emit("--dataset-name", b.dataset_name)
    emit("--dataset-path", dataset_path)
    emit("--num-prompts", str(b.num_prompts))
    emit("--seed", str(b.seed))
    emit("--percentile-metrics", b.percentile_metrics)
    emit("--metric-percentiles", b.metric_percentiles)
    emit("--save-result")
    emit("--save-detailed")
    emit("--result-dir", result_dir)
    emit("--result-filename", result_filename)
    emit("--disable-tqdm")

    # `--ready-check-timeout-sec` was added in vLLM 0.10.1; null omits it.
    if cfg.server.ready_check_timeout_sec is not None:
        emit("--ready-check-timeout-sec", str(cfg.server.ready_check_timeout_sec))
    if cfg.server.tokenizer:
        emit("--tokenizer", cfg.server.tokenizer)
    if b.request_rate and str(b.request_rate) != "inf":
        emit("--request-rate", str(b.request_rate))
    if b.max_concurrency:
        emit("--max-concurrency", str(b.max_concurrency))
    if b.dataset_name == "custom":
        emit("--custom-output-len", str(b.custom_output_len))
        if b.custom_skip_chat_template:
            emit("--custom-skip-chat-template")
    if b.ignore_eos:
        emit("--ignore-eos")
    for key, value in b.metadata.items():
        emit("--metadata", f"{key}={value}")

    # Raw pass-through flags are the user's responsibility, but we still warn.
    for extra in b.extra_args:
        if extra.startswith("--") and not caps.supports(extra.split("=", 1)[0]):
            warnings.append(
                f"{extra} (from benchmark.extra_args) is not supported by this vLLM"
            )
        args.append(extra)
    return args


def build_env(cfg: AppConfig, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for the child process, in both native and docker mode.

    Secrets live here (never on the argv) and are forwarded into the container
    by name only.
    """
    env = dict(os.environ if base_env is None else base_env)
    # HF cache + token matter in native mode too: vLLM downloads the tokenizer.
    env["HF_HOME"] = (
        "/hf-cache" if cfg.runner.mode == "docker" else str(cfg.path(cfg.runner.hf_cache_dir).resolve())
    )
    token = cfg.runner.hf_token or env.get("HF_TOKEN")
    if token:
        env["HF_TOKEN"] = token
    else:
        env.pop("HF_TOKEN", None)
    # Unconditional: a configured key must win over whatever is exported.
    if cfg.server.api_key:
        env["OPENAI_API_KEY"] = cfg.server.api_key
    return env


def build_command(
    cfg: AppConfig,
    result_filename: Optional[str] = None,
    container_name: Optional[str] = None,
    caps: Optional[VllmCapabilities] = None,
    detect: bool = True,
) -> BenchCommand:
    """Return the argv to execute, its environment, and the host result path."""
    if not cfg.server.model:
        raise RunnerError("server.model is required (must match the id served by the endpoint)")
    if not cfg.server.tokenizer:
        raise RunnerError(
            "server.tokenizer is required: vLLM needs a HuggingFace tokenizer to count tokens"
        )

    dataset_host_path = cfg.path(cfg.benchmark.dataset_path).resolve()
    if not dataset_host_path.exists() and cfg.benchmark.dataset_name == "custom":
        raise RunnerError(f"dataset file not found: {dataset_host_path}")

    result_dir_host = cfg.path(cfg.benchmark.result_dir).resolve()
    result_dir_host.mkdir(parents=True, exist_ok=True)
    filename = result_filename or default_result_filename(cfg)
    result_path = result_dir_host / filename
    env = build_env(cfg)

    capture_path = sidecar_path_for(result_path) if cfg.runner.capture else None
    if caps is None:
        caps = detect_capabilities(cfg) if detect else VllmCapabilities()
    warnings: List[str] = []
    if not caps.detected and detect:
        warnings.append(
            "could not detect which `vllm bench serve` flags this vLLM supports"
            + (f" ({caps.note})" if caps.note else "")
            + "; emitting all of them"
        )

    if cfg.runner.mode == "native":
        hf_cache_host = cfg.path(cfg.runner.hf_cache_dir).resolve()
        hf_cache_host.mkdir(parents=True, exist_ok=True)
        args = build_bench_args(
            cfg,
            base_url=cfg.server.base_url,
            dataset_path=str(dataset_host_path),
            result_dir=str(result_dir_host),
            result_filename=filename,
            caps=caps,
            warnings=warnings,
        )
        if cfg.runner.capture:
            # The launcher needs an interpreter that can `import vllm`.
            python = cfg.runner.python or sys.executable
            if shutil.which(python) is None and not Path(python).exists():
                raise RunnerError(
                    f"runner.python={python!r} not found. Point it at a python "
                    f"that can `import vllm`, or set runner.capture: false."
                )
            argv = [
                python, "-m", "vllm_bench_platform.vllm_entry",
                "--vbp-capture", str(capture_path),
                *args,
            ]
        else:
            binary = cfg.runner.vllm_bin
            if shutil.which(binary) is None:
                raise RunnerError(
                    f"`{binary}` not found on PATH. Install vLLM (`uv pip install vllm`) "
                    f"or switch runner.mode to 'docker'."
                )
            argv = [binary, "bench", "serve", *args]
        return BenchCommand(
            argv=argv,
            result_path=result_path,
            env=env,
            capture_path=capture_path,
            capabilities=caps,
            warnings=warnings,
        )

    # ---- docker mode -------------------------------------------------------
    if shutil.which("docker") is None:
        raise RunnerError("`docker` not found on PATH but runner.mode is 'docker'")

    base_url = rewrite_for_container(cfg.server.base_url, cfg.runner.host_gateway_alias)
    hf_cache_host = cfg.path(cfg.runner.hf_cache_dir).resolve()
    hf_cache_host.mkdir(parents=True, exist_ok=True)
    name = container_name or f"vllm-bench-platform-{os.getpid()}-{int(time.time())}"

    docker: List[str] = ["docker", "run", "--rm", "--name", name]
    if cfg.runner.docker_platform:
        docker += ["--platform", cfg.runner.docker_platform]
    user = resolve_docker_user(cfg.runner.docker_user)
    if user:
        docker += ["--user", user]
    docker += [
        "--add-host", f"{cfg.runner.host_gateway_alias}:host-gateway",
        "-v", f"{dataset_host_path.parent}:/work/data:ro",
        "-v", f"{result_dir_host}:/work/results",
        "-v", f"{hf_cache_host}:/hf-cache",
        "-e", "HF_HOME=/hf-cache",
    ]
    # Forward secrets by NAME only: the value stays in the process environment
    # and never reaches the argv (which we echo, and which `ps` can read).
    if env.get("HF_TOKEN"):
        docker += ["-e", "HF_TOKEN"]
    if env.get("OPENAI_API_KEY"):
        docker += ["-e", "OPENAI_API_KEY"]
    if cfg.runner.capture:
        # Mount this package read-only so the container runs the very same
        # launcher as native mode (no duplicated capture logic in the image).
        pkg_dir = Path(__file__).resolve().parent
        docker += [
            "-v", f"{pkg_dir}:{CONTAINER_PKG_ROOT}/{pkg_dir.name}:ro",
            "-e", f"PYTHONPATH={CONTAINER_PKG_ROOT}",
        ]
    docker += list(cfg.runner.docker_extra_args)
    docker += [cfg.runner.docker_image]

    if cfg.runner.capture:
        entry = [
            "python", "-m", "vllm_bench_platform.vllm_entry",
            "--vbp-capture", f"/work/results/{capture_path.name}",
        ]
    else:
        entry = ["vllm-bench-serve"]
    docker += entry

    args = build_bench_args(
        cfg,
        base_url=base_url,
        dataset_path=f"/work/data/{dataset_host_path.name}",
        result_dir="/work/results",
        result_filename=filename,
        caps=caps,
        warnings=warnings,
    )
    return BenchCommand(
        argv=docker + args,
        result_path=result_path,
        env=env,
        container_name=name,
        capture_path=capture_path,
        capabilities=caps,
        warnings=warnings,
    )


def _kill_container(name: str) -> None:
    try:
        subprocess.run(
            ["docker", "kill", name],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:  # pragma: no cover - best effort cleanup
        pass


def run_benchmark(
    cfg: AppConfig,
    result_filename: Optional[str] = None,
    echo: bool = True,
) -> RunOutcome:
    """Launch the benchmark, streaming its output live, and parse the outcome."""
    plan = build_command(cfg, result_filename=result_filename)
    if echo:
        caps = plan.capabilities
        if caps is not None and caps.detected:
            print(f"detected vLLM {caps.version or '?'} ({len(caps.flags)} bench flags)", flush=True)
        for warning in plan.warnings:
            print(f"  WARN  {warning}", flush=True)
        print("$ " + plan.display(), flush=True)

    chunks: List[str] = []
    timed_out = threading.Event()

    proc = subprocess.Popen(
        plan.argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=plan.env,
    )

    def _on_timeout() -> None:
        timed_out.set()
        if plan.container_name:
            # `docker run` forwards no signal to the container; kill it by name
            # so it does not keep hammering the server after we give up.
            _kill_container(plan.container_name)
        proc.kill()

    watchdog = threading.Timer(cfg.runner.timeout_sec, _on_timeout)
    watchdog.daemon = True
    watchdog.start()

    started = time.time()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            if echo:
                print(line, end="", flush=True)
        returncode = proc.wait()
    finally:
        watchdog.cancel()
        if proc.stdout is not None:
            proc.stdout.close()
    duration = time.time() - started
    output = "".join(chunks)

    if timed_out.is_set():
        raise RunnerError(
            f"`vllm bench serve` exceeded runner.timeout_sec={cfg.runner.timeout_sec}s "
            f"and was killed"
            + (f" (container {plan.container_name} killed)" if plan.container_name else "")
            + f".\n--- last output ---\n{output[-4000:]}"
        )
    if returncode != 0:
        raise RunnerError(
            f"`vllm bench serve` exited with code {returncode}.\n"
            f"--- output ---\n{output[-6000:]}"
        )
    if not plan.result_path.exists():
        raise RunnerError(
            f"benchmark finished but no result file was written at {plan.result_path}"
        )
    return RunOutcome(
        command=plan.argv,
        returncode=returncode,
        output=output,
        result_path=plan.result_path,
        duration_s=duration,
        container_name=plan.container_name,
        capture_path=plan.capture_path if (plan.capture_path and plan.capture_path.exists()) else None,
        vllm_version=plan.capabilities.version if plan.capabilities else None,
        warnings=list(plan.warnings),
    )
