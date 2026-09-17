"""Command line interface: check / prepare-dataset / run / sync."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

from . import __version__
from . import __version__ as _tool_version
from .capture import load_sidecar, merge_with_result, sidecar_path_for
from .config import AppConfig, find_default_config, load_config
from .platform_sync import SyncError, build_platform_client, render_experiment_name, sync_result
from .results import BenchmarkResult, parse_result_file
from .runner import RunnerError, build_command, rewrite_for_container, run_benchmark
from .samples import (
    align_by_prompt,
    align_requests_to_samples,
    load_samples,
    prepare_dataset,
)

app = typer.Typer(
    add_completion=False,
    help="Run `vllm bench serve` against an OpenAI-compatible server and sync "
         "the results to the Benchmark 平台 (dataset + experiment + traces + "
         "feedback scores).",
    no_args_is_help=True,
)

ConfigOpt = typer.Option(
    None, "--config", "-c",
    help="Path to the YAML config (default: ./config.yaml, then ./config.example.yaml).",
)


def _load(config: Optional[str]) -> AppConfig:
    path = Path(config) if config else find_default_config()
    if path is None:
        typer.secho(
            "no config file found; create config.yaml (copy config.example.yaml)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    try:
        cfg = load_config(path)
    except Exception as exc:
        typer.secho(f"failed to load config {path}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    typer.echo(f"config: {path}")
    return cfg


def _ok(msg: str) -> None:
    typer.secho(f"  OK    {msg}", fg=typer.colors.GREEN)


def _fail(msg: str) -> None:
    typer.secho(f"  FAIL  {msg}", fg=typer.colors.RED)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# ---------------------------------------------------------------------------


@app.command()
def check(config: Optional[str] = ConfigOpt) -> None:
    """Verify connectivity to the inference server, the Benchmark 平台, and the runner."""
    import httpx

    from .platform_sync import dataset_exists, platform_sdk_version

    cfg = _load(config)
    failures = 0

    typer.echo("\n[1/4] inference server")
    models_url = f"{cfg.server.base_url}/v1/models"
    try:
        headers = {}
        if cfg.server.api_key:
            headers["Authorization"] = f"Bearer {cfg.server.api_key}"
        resp = httpx.get(models_url, timeout=15.0, headers=headers)
        resp.raise_for_status()
        ids = [m.get("id") for m in resp.json().get("data", [])]
        if not ids:
            _fail(f"{models_url} returned 0 models; is a model loaded?")
            failures += 1
        else:
            _ok(f"{models_url} -> {len(ids)} model(s)")
        if cfg.server.model and cfg.server.model not in ids:
            _fail(f"server.model={cfg.server.model!r} not in {ids}")
            failures += 1
        elif cfg.server.model:
            _ok(f"server.model={cfg.server.model!r} is served")
    except Exception as exc:
        _fail(f"{models_url}: {exc}")
        failures += 1

    typer.echo("\n[2/4] benchmark dataset")
    ds = cfg.path(cfg.benchmark.dataset_path)
    if cfg.benchmark.dataset_name != "custom":
        _ok(f"dataset_name={cfg.benchmark.dataset_name} (managed by vLLM)")
    elif ds.exists():
        try:
            samples = load_samples(ds)
            _ok(f"{ds} -> {len(samples)} prompt(s)")
            if cfg.benchmark.num_prompts > len(samples):
                typer.secho(
                    f"  WARN  num_prompts={cfg.benchmark.num_prompts} > {len(samples)} samples; "
                    "vLLM will oversample (duplicates)",
                    fg=typer.colors.YELLOW,
                )
        except Exception as exc:
            _fail(f"{ds}: {exc}")
            failures += 1
    else:
        _fail(f"{ds} does not exist (run `prepare-dataset`)")
        failures += 1

    typer.echo("\n[3/4] benchmark runner")
    try:
        plan = build_command(cfg)
        _ok(f"mode={cfg.runner.mode}")
        typer.echo(f"        $ {plan.display()}")
        typer.echo(f"        result -> {plan.result_path}")
        if cfg.runner.mode == "docker":
            container_url = rewrite_for_container(
                cfg.server.base_url, cfg.runner.host_gateway_alias
            )
            _ok(f"container will call {container_url}")
    except RunnerError as exc:
        _fail(str(exc))
        failures += 1

    typer.echo("\n[4/4] Benchmark 平台")
    if not cfg.benchmark_platform.enabled:
        typer.echo("  SKIP  benchmark_platform.enabled = false")
    else:
        url = cfg.benchmark_platform.url.rstrip("/")
        try:
            headers = {"Comet-Workspace": cfg.benchmark_platform.workspace}
            if cfg.benchmark_platform.api_key:
                headers["Authorization"] = cfg.benchmark_platform.api_key
            resp = httpx.get(
                f"{url}/v1/private/projects",
                timeout=15.0,
                headers=headers,
            )
            resp.raise_for_status()
            _ok(f"{url} reachable (workspace={cfg.benchmark_platform.workspace})")
        except Exception as exc:
            _fail(f"{url}: {exc}")
            failures += 1
        try:
            # Read-only: `check` must not create anything as a side effect.
            client = build_platform_client(cfg.benchmark_platform)
            exists = dataset_exists(client, cfg.benchmark_platform.dataset_name)
            _ok(f"platform SDK ok (v{platform_sdk_version()})")
            if exists:
                n = len(client.get_dataset(cfg.benchmark_platform.dataset_name).get_items())
                _ok(f"dataset {cfg.benchmark_platform.dataset_name!r} exists ({n} item(s))")
            else:
                typer.secho(
                    f"  INFO  dataset {cfg.benchmark_platform.dataset_name!r} does not exist yet; "
                    "`prepare-dataset` or `run` will create it",
                    fg=typer.colors.BLUE,
                )
        except Exception as exc:
            _fail(f"Benchmark 平台 SDK: {exc}")
            failures += 1

    typer.echo("")
    if failures:
        typer.secho(f"{failures} check(s) failed", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho("all checks passed", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------


@app.command("prepare-dataset")
def prepare_dataset_cmd(
    config: Optional[str] = ConfigOpt,
    num_samples: Optional[int] = typer.Option(None, help="Override prepare.num_samples."),
    source: Optional[str] = typer.Option(None, help="sonnet | sharegpt | local."),
    output: Optional[str] = typer.Option(None, help="Override prepare.output_path."),
    upload: bool = typer.Option(
        True, help="Also upload the samples to the Benchmark 平台 dataset."
    ),
) -> None:
    """Download a slice of an official vLLM benchmark dataset into a `custom` JSONL file."""
    cfg = _load(config)
    p = cfg.prepare
    out_path = cfg.path(output or p.output_path)
    src = source or p.source
    n = num_samples if num_samples is not None else p.num_samples

    typer.echo(f"preparing {n} sample(s) from {src!r} -> {out_path}")
    samples = prepare_dataset(
        source=src,
        output_path=out_path,
        num_samples=n,
        seed=p.seed,
        lines_per_prompt=p.lines_per_prompt,
        source_url=p.source_url,
    )
    _ok(f"wrote {len(samples)} prompt(s) to {out_path}")
    preview = samples[0].prompt.replace("\n", " ")[:100]
    typer.echo(f"        first prompt: {preview}...")

    if upload and cfg.benchmark_platform.enabled:
        client = build_platform_client(cfg.benchmark_platform)
        dataset = client.get_or_create_dataset(
            name=cfg.benchmark_platform.dataset_name,
            description=f"vllm bench serve samples ({src})",
        )
        dataset.insert([s.as_dataset_item(source=src) for s in samples])
        client.flush()
        total = len(dataset.get_items())
        _ok(
            f"Benchmark 平台 dataset {cfg.benchmark_platform.dataset_name!r} "
            f"now has {total} item(s)"
        )


# ---------------------------------------------------------------------------


def _as_captured(result: BenchmarkResult):
    """Fallback shim: build CapturedRequest objects from the result arrays only.

    No prompts and no wall-clock, so these traces get no timeline and no spans
    — exactly the old behaviour, kept for `runner.capture: false`.
    """
    from .capture import CapturedRequest

    out = []
    for req in result.requests:
        out.append(
            CapturedRequest(
                index=req.index,
                prompt=req.prompt,
                sample_id=req.sample_id,
                input_len=req.input_len,
                output_len=req.output_len,
                success=req.succeeded,
                ttft_ms=req.ttft_ms,
                e2e_ms=req.e2e_ms,
                tpot_ms=req.tpot_ms,
                itl_ms=list(req.itl_ms),
                generated_text=req.generated_text or "",
                error=req.error,
            )
        )
    return out


def _run_cfg(cfg: AppConfig) -> dict:
    """The "how was this run" block copied into traces and the summary."""
    return {
        "endpoint": cfg.server.endpoint,
        "base_url": cfg.server.base_url,
        "dataset_name": cfg.benchmark.dataset_name,
        "dataset_path": cfg.benchmark.dataset_path,
        "runner_mode": cfg.runner.mode,
        "tool_version": _tool_version,
        "provider": cfg.server.backend,
    }


def _load_capture(cfg: AppConfig, result: BenchmarkResult, explicit: Optional[Path]):
    """Locate and parse the per-request sidecar, if one was produced."""
    candidates = [explicit] if explicit else []
    if result.source_path is not None:
        candidates.append(sidecar_path_for(result.source_path))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            captured = load_sidecar(candidate)
            return merge_with_result(captured, result), Path(candidate)
    return [], None


def _sync(
    cfg: AppConfig,
    result: BenchmarkResult,
    experiment_name: Optional[str],
    capture_path: Optional[Path] = None,
) -> None:
    samples = []
    ds_path = cfg.path(cfg.benchmark.dataset_path)
    if cfg.benchmark.dataset_name == "custom" and ds_path.exists():
        samples = load_samples(ds_path)

    captured, used_capture = _load_capture(cfg, result, capture_path)

    if captured:
        _ok(f"per-request capture: {len(captured)} request(s) from {used_capture.name}")
        expected = cfg.benchmark.num_prompts or len(captured)
        if len(captured) != expected:
            typer.secho(
                f"  WARN  capture has {len(captured)} request(s) but num_prompts={expected}",
                fg=typer.colors.YELLOW,
            )
        # Prompt-text alignment: exact, and works for any dataset type.
        matched, extra = align_by_prompt(captured, samples)
        if extra:
            samples = list(samples) + extra
            _ok(f"added {len(extra)} prompt(s) from the capture to the dataset")
        _ok(f"aligned {matched}/{len(captured)} request(s) to dataset items by prompt text")
    else:
        # Fallback: no sidecar, so replay vLLM's shuffle to guess the mapping.
        typer.secho(
            "  WARN  no per-request capture found; traces will have no timeline "
            "or spans (set runner.capture: true and re-run)",
            fg=typer.colors.YELLOW,
        )
        aligned = align_requests_to_samples(
            result.requests, samples, dataset_name=cfg.benchmark.dataset_name
        )
        if aligned:
            _ok(f"aligned {len(result.requests)} request(s) by seed replay (fallback)")
        else:
            typer.secho(
                "  WARN  could not align requests to dataset items; "
                "traces will have no dataset link",
                fg=typer.colors.YELLOW,
            )
        captured = _as_captured(result)

    try:
        report = sync_result(
            result,
            samples,
            cfg.benchmark_platform,
            experiment_name=experiment_name
            or render_experiment_name(cfg.benchmark_platform.experiment_name, result),
            source_label=cfg.prepare.source,
            captured=captured,
            run_cfg=_run_cfg(cfg),
            extra_experiment_config={
                "runner_mode": cfg.runner.mode,
                "capture": bool(used_capture),
                "result_file": result.source_path.name if result.source_path else None,
            },
        )
    except SyncError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo("")
    typer.secho("Benchmark 平台 sync", fg=typer.colors.CYAN, bold=True)
    for key, value in report.as_dict().items():
        typer.echo(f"  {key:<24} {value}")
    typer.echo("")
    typer.echo(f"  UI: {cfg.benchmark_platform.url.rstrip('/').removesuffix('/api')}")


def failure_summary(result: BenchmarkResult) -> Optional[str]:
    """Describe why a run should be considered failed, or None when it is clean."""
    problems: List[str] = []
    a = result.aggregate
    if a.num_prompts is not None and a.completed is not None and a.completed < a.num_prompts:
        problems.append(f"only {a.completed}/{a.num_prompts} requests completed")

    errors = [r.error for r in result.requests if r.error]
    if errors:
        distinct: List[str] = []
        for err in errors:
            if err not in distinct:
                distinct.append(err)
        shown = "; ".join(e[:200] for e in distinct[:3])
        more = f" (+{len(distinct) - 3} more distinct)" if len(distinct) > 3 else ""
        problems.append(f"{len(errors)} request error(s): {shown}{more}")

    return "; ".join(problems) if problems else None


@app.command()
def run(
    config: Optional[str] = ConfigOpt,
    experiment_name: Optional[str] = typer.Option(None, help="Override the experiment name."),
    result_filename: Optional[str] = typer.Option(None, help="Override the result JSON filename."),
    no_sync: bool = typer.Option(
        False, "--no-sync", help="Only benchmark, do not touch the Benchmark 平台."
    ),
    allow_failures: bool = typer.Option(
        False, "--allow-failures",
        help="Exit 0 even if some requests failed (results are synced either way).",
    ),
) -> None:
    """Run `vllm bench serve` and sync the results to the Benchmark 平台."""
    cfg = _load(config)
    try:
        outcome = run_benchmark(cfg, result_filename=result_filename)
    except RunnerError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1)

    _ok(f"result written to {outcome.result_path} ({outcome.duration_s:.1f}s)")
    result = parse_result_file(outcome.result_path)
    _print_summary(result)

    problems = failure_summary(result)
    if problems:
        typer.secho(f"  WARN  benchmark had failures: {problems}", fg=typer.colors.YELLOW)

    if no_sync or not cfg.benchmark_platform.enabled:
        typer.echo("\nskipping Benchmark 平台 sync")
    else:
        _sync(cfg, result, experiment_name, capture_path=outcome.capture_path)

    if problems and not allow_failures:
        typer.secho(
            "exiting non-zero because the benchmark did not complete cleanly "
            "(pass --allow-failures to override)",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)


@app.command()
def sync(
    result_file: str = typer.Argument(..., help="Path to a `vllm bench serve` result JSON."),
    config: Optional[str] = ConfigOpt,
    experiment_name: Optional[str] = typer.Option(None, help="Override the experiment name."),
) -> None:
    """Sync an existing `vllm bench serve` result JSON to the Benchmark 平台."""
    cfg = _load(config)
    path = Path(result_file)
    if not path.exists():
        typer.secho(f"no such file: {path}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    result = parse_result_file(path)
    _print_summary(result)
    problems = failure_summary(result)
    if problems:
        typer.secho(f"  WARN  benchmark had failures: {problems}", fg=typer.colors.YELLOW)
    if not result.has_detailed:
        typer.secho(
            "  WARN  the result file has no per-request arrays "
            "(re-run with --save-detailed); only the summary trace will be created",
            fg=typer.colors.YELLOW,
        )
    if not cfg.benchmark_platform.enabled:
        typer.secho(
            "benchmark_platform.enabled = false; nothing to do", fg=typer.colors.YELLOW
        )
        raise typer.Exit(code=1)
    _sync(cfg, result, experiment_name)


@app.command("show")
def show(result_file: str = typer.Argument(..., help="Result JSON to summarise.")) -> None:
    """Print the parsed metrics of a result file (the platform is not contacted)."""
    result = parse_result_file(result_file)
    _print_summary(result)
    typer.echo(json.dumps(result.aggregate.as_dict(), indent=2))


def _print_summary(result: BenchmarkResult) -> None:
    a = result.aggregate
    typer.echo("")
    typer.secho(
        f"benchmark: {result.model_id} via {result.backend} "
        f"({a.completed}/{a.num_prompts} completed)",
        fg=typer.colors.CYAN,
        bold=True,
    )

    def line(label: str, value, unit: str = "") -> None:
        if value is None:
            return
        typer.echo(f"  {label:<34} {value:>12.2f} {unit}")

    line("Benchmark duration (E2E 总时间)", a.duration_s, "s")
    line("Mean TTFT", a.get("ttft", "mean"), "ms")
    line("P99 TTFT", a.get("ttft", "p99"), "ms")
    line("Mean TPOT", a.get("tpot", "mean"), "ms")
    line("P99 TPOT", a.get("tpot", "p99"), "ms")
    line("Mean E2EL (单请求 E2E)", a.get("e2el", "mean"), "ms")
    line("P99 E2EL", a.get("e2el", "p99"), "ms")
    line("Output throughput (输出 TPS)", a.output_throughput, "tok/s")
    line("Total throughput (总 TPS)", a.total_token_throughput, "tok/s")
    line("Request throughput", a.request_throughput, "req/s")
    typer.echo("")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
