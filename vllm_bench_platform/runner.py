"""Build and execute the `vllm bench serve` command (natively or in Docker)."""

from __future__ import annotations

import os
import platform
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


@dataclass
class BenchCommand:
    """Everything needed to launch one benchmark."""

    argv: List[str]
    result_path: Path
    env: Dict[str, str] = field(default_factory=dict)
    container_name: Optional[str] = None
    # Host path of the per-request capture sidecar (None when capture is off).
    capture_path: Optional[Path] = None

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
) -> List[str]:
    """The `vllm bench serve` flags, identical for native and docker mode."""
    b = cfg.benchmark
    args: List[str] = [
        "--backend", cfg.server.backend,
        "--base-url", base_url,
        "--endpoint", cfg.server.endpoint,
        "--model", cfg.server.model,
        "--dataset-name", b.dataset_name,
        "--dataset-path", dataset_path,
        "--num-prompts", str(b.num_prompts),
        "--seed", str(b.seed),
        "--percentile-metrics", b.percentile_metrics,
        "--metric-percentiles", b.metric_percentiles,
        "--save-result",
        "--save-detailed",
        "--result-dir", result_dir,
        "--result-filename", result_filename,
        "--disable-tqdm",
    ]
    # `--ready-check-timeout-sec` was added in vLLM 0.10.2; null omits it.
    if cfg.server.ready_check_timeout_sec is not None:
        args += ["--ready-check-timeout-sec", str(cfg.server.ready_check_timeout_sec)]
    if cfg.server.tokenizer:
        args += ["--tokenizer", cfg.server.tokenizer]
    if b.request_rate and str(b.request_rate) != "inf":
        args += ["--request-rate", str(b.request_rate)]
    if b.max_concurrency:
        args += ["--max-concurrency", str(b.max_concurrency)]
    if b.dataset_name == "custom":
        args += ["--custom-output-len", str(b.custom_output_len)]
        if b.custom_skip_chat_template:
            args += ["--custom-skip-chat-template"]
    if b.ignore_eos:
        args += ["--ignore-eos"]
    for key, value in b.metadata.items():
        args += ["--metadata", f"{key}={value}"]
    args += list(b.extra_args)
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

    if cfg.runner.mode == "native":
        hf_cache_host = cfg.path(cfg.runner.hf_cache_dir).resolve()
        hf_cache_host.mkdir(parents=True, exist_ok=True)
        args = build_bench_args(
            cfg,
            base_url=cfg.server.base_url,
            dataset_path=str(dataset_host_path),
            result_dir=str(result_dir_host),
            result_filename=filename,
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
    )
    return BenchCommand(
        argv=docker + args,
        result_path=result_path,
        env=env,
        container_name=name,
        capture_path=capture_path,
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
    )
