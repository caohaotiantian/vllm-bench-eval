"""Push a parsed `vllm bench serve` run into the Benchmark 平台.

Data model
----------
* **Dataset** (``benchmark_platform.dataset_name``) – one item per benchmark
  prompt: ``{"sample_id", "line_index", "prompt", "dataset_source"}``
  (``source`` is a reserved platform field, hence ``dataset_source``).
  Re-running is safe: the platform de-duplicates items by content hash, so only
  new prompts are inserted.
* **Experiment** (``benchmark_platform.experiment_name``) – one per benchmark
  run, attached to that dataset. ``experiment_config`` carries the full
  aggregate metrics plus the run configuration (model, backend, concurrency,
  tokenizer ...).
* **Traces** (project ``benchmark_platform.project_name``) – one per request,
  input = the prompt, output = the generated text, with feedback scores
  ``ttft_ms`` / ``tpot_ms`` / ``e2e_ms`` / ``output_tokens_per_s`` / ... Each
  trace is linked to its dataset item through an experiment item.
* **Summary trace** – one extra trace named ``benchmark-summary`` carrying the
  aggregate feedback scores (mean/p99 TTFT, TPOT, output TPS, total TPS,
  mean E2E latency, duration, request throughput).
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import metric_names
from .capture import MAX_ITL_VALUES, CapturedRequest, benchmark_window
from .results import BenchmarkResult
from .samples import Sample

# Display names live in metric_names; re-exported here for convenience.
SUMMARY_TRACE_NAME = metric_names.SUMMARY_TRACE_NAME
REQUEST_TRACE_PREFIX = metric_names.REQUEST_TRACE_PREFIX
SPAN_PREFILL = metric_names.SPAN_PREFILL
SPAN_DECODE = metric_names.SPAN_DECODE


class SyncError(RuntimeError):
    """Raised when the run cannot be represented faithfully on the platform."""


@dataclass
class SyncReport:
    project_name: str
    dataset_name: str
    experiment_name: str
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    dataset_items_total: int = 0
    dataset_items_inserted: int = 0
    traces_created: int = 0
    spans_created: int = 0
    experiment_items_created: int = 0
    summary_trace_id: Optional[str] = None
    unaligned_requests: int = 0
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "dataset_name": self.dataset_name,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "dataset_items_total": self.dataset_items_total,
            "dataset_items_inserted": self.dataset_items_inserted,
            "traces_created": self.traces_created,
            "spans_created": self.spans_created,
            "experiment_items_created": self.experiment_items_created,
            "summary_trace_id": self.summary_trace_id,
            "unaligned_requests": self.unaligned_requests,
            "warnings": self.warnings,
        }


# The platform SDK ships as the `opik` package and installs a console handler
# formatted as "OPIK: <message>". That prefix is the upstream project name, not
# the product name our users know, so we rebrand the formatter in place.
CONSOLE_LOG_PREFIX = "Benchmark 平台"


def rebrand_sdk_logging(prefix: str = CONSOLE_LOG_PREFIX) -> bool:
    """Relabel the platform SDK's console log prefix. Returns True if changed."""
    import logging

    changed = False
    try:
        sdk_logger = logging.getLogger("opik")
        for handler in sdk_logger.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            fmt = getattr(handler.formatter, "_fmt", "") or ""
            if "OPIK: " in fmt:
                handler.setFormatter(
                    logging.Formatter(fmt.replace("OPIK: ", f"{prefix}: "))
                )
                changed = True
    except Exception:  # pragma: no cover - cosmetic only, never fail a sync
        return False
    return changed


def platform_sdk_version() -> str:
    """Version of the installed platform SDK (shipped as the `opik` package)."""
    import opik

    return str(getattr(opik, "__version__", "?"))


def build_platform_client(platform_cfg, project_name: Optional[str] = None):
    """Create a Benchmark 平台 client from the config (imported lazily).

    The platform's Python SDK is distributed as the `opik` package.
    """
    import opik

    rebrand_sdk_logging()
    return opik.Opik(
        project_name=project_name or platform_cfg.project_name,
        workspace=platform_cfg.workspace,
        host=platform_cfg.url,
        api_key=platform_cfg.api_key,
        _show_misconfiguration_message=False,
    )


def dataset_exists(client, name: str) -> bool:
    """True when the platform workspace already holds a dataset with this name."""
    try:
        client.get_dataset(name)
        return True
    except Exception as exc:  # the SDK raises ApiError(404) for "not found"
        if getattr(exc, "status_code", None) == 404:
            return False
        raise


def index_dataset_items(items) -> Dict[str, str]:
    """Map ``sample_id -> dataset_item_id``, refusing ambiguous duplicates."""
    mapping: Dict[str, str] = {}
    duplicates: List[str] = []
    for item in items:
        sample_id = item.get("sample_id")
        if not sample_id:
            continue
        if sample_id in mapping:
            duplicates.append(sample_id)
            continue
        mapping[sample_id] = item.get("id")
    if duplicates:
        raise SyncError(
            "the Benchmark 平台 dataset contains duplicate sample_id value(s): "
            f"{sorted(set(duplicates))}. Linking traces to dataset items would be "
            "ambiguous. Use a fresh benchmark_platform.dataset_name, or delete "
            "the stale items."
        )
    return mapping


def render_experiment_name(template: str, result: BenchmarkResult, cfg=None) -> str:
    ts = result.date or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    values = {
        "model": result.model_id,
        "ts": ts,
        "date": ts,
        "backend": result.backend,
        "num_prompts": result.aggregate.num_prompts or len(result.requests),
    }
    try:
        return template.format(**values)
    except (KeyError, IndexError):
        return template


def build_experiment_config(
    result: BenchmarkResult,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """`experiment_config` = aggregate metrics + the run configuration."""
    config: Dict[str, Any] = {
        "metrics": result.aggregate.as_dict(),
        "run": result.run_config(),
        "tool": "vllm-bench-eval",
    }
    if extra:
        config.update(extra)
    return config


def run_descriptor(result: BenchmarkResult, run_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The small "what was run" block shared by traces and the summary input."""
    raw = result.raw
    desc: Dict[str, Any] = {
        "model": result.model_id,
        "tokenizer": result.tokenizer_id,
        "backend": result.backend,
        "num_prompts": result.aggregate.num_prompts,
        "request_rate": raw.get("request_rate"),
        "max_concurrency": raw.get("max_concurrency"),
    }
    if run_cfg:
        for key in ("endpoint", "base_url", "dataset_name", "dataset_path",
                    "tool_version", "vllm_version"):
            if run_cfg.get(key) is not None:
                desc[key] = run_cfg[key]
    return {k: v for k, v in desc.items() if v is not None}


def build_request_trace_payload(
    req: CapturedRequest,
    *,
    run_id: str,
    experiment_name: str,
    result: BenchmarkResult,
    run_cfg: Dict[str, Any],
    tags: Sequence[str],
) -> Dict[str, Any]:
    """Everything `client.trace(...)` needs for one benchmarked request."""
    itl_summary = req.itl_summary()
    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "model_id": result.model_id,
        "tokenizer_id": result.tokenizer_id,
        "backend": result.backend,
        "vllm_version": run_cfg.get("vllm_version"),
        "endpoint": run_cfg.get("endpoint"),
        "base_url": run_cfg.get("base_url"),
        "request_rate": result.raw.get("request_rate"),
        "max_concurrency": result.raw.get("max_concurrency"),
        "request_index": req.index,
        "request_id": req.request_id,
        "sample_id": req.sample_id,
        "input_tokens": req.input_len,
        "output_tokens": req.output_len,
        "max_output_tokens": req.max_output_len,
        "ttft_ms": _round(req.ttft_ms),
        "tpot_ms": _round(req.tpot_ms),
        "e2e_ms": _round(req.e2e_ms),
        "itl_stats": itl_summary,
        # Capped: a long generation would otherwise bloat every trace.
        "itl_ms": [round(v, 4) for v in req.itl_ms[:MAX_ITL_VALUES]],
        "itl_truncated": len(req.itl_ms) > MAX_ITL_VALUES,
        "error": req.error,
    }
    if req.extra_body:
        metadata["sampling_params"] = req.extra_body
    if req.ignore_eos is not None:
        metadata["ignore_eos"] = req.ignore_eos

    return {
        "name": f"{REQUEST_TRACE_PREFIX}{req.index:04d}",
        "start_time": req.start_dt,
        "end_time": req.end_dt,
        "input": {
            "prompt": req.prompt,
            "max_tokens": req.max_output_len,
            "model": req.model or result.model_id,
        },
        "output": {
            "generated_text": req.generated_text,
            "success": req.success,
            "output_tokens": req.output_len,
        },
        "tags": list(tags) + (["error"] if req.error else []),
        "feedback_scores": req.feedback_scores(),
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def build_request_spans(
    req: CapturedRequest,
    *,
    trace_id: str,
    model: str,
    provider: str,
    project_name: str,
) -> List[Dict[str, Any]]:
    """Split the request into a prefill span and a decode span.

    The split point is the first token: prefill == TTFT, decode == everything
    after it. Without this the trace is a single opaque bar and TTFT is only
    readable as a number.
    """
    start, first_token, end = req.start_dt, req.first_token_dt, req.end_dt
    if start is None or end is None or first_token is None:
        return []

    prompt_tokens = int(req.input_len or 0)
    completion_tokens = int(req.output_len or 0)
    spans: List[Dict[str, Any]] = [
        {
            "trace_id": trace_id,
            "name": SPAN_PREFILL,
            "type": "llm",
            "start_time": start,
            "end_time": first_token,
            "input": {"prompt": req.prompt},
            "output": {"first_token_received": True},
            "metadata": {
                "ttft_ms": _round(req.ttft_ms),
                "input_tokens": req.input_len,
                "phase": "prefill",
            },
            "model": model,
            "provider": provider,
            "project_name": project_name,
        },
        {
            "trace_id": trace_id,
            "name": SPAN_DECODE,
            "type": "llm",
            "start_time": first_token,
            "end_time": end,
            "input": {"prompt": req.prompt},
            "output": {"generated_text": req.generated_text},
            "metadata": {
                "phase": "decode",
                "tpot_ms": _round(req.tpot_ms),
                "output_tokens": req.output_len,
                "itl_stats": req.itl_summary(),
                "decode_ms": _round(
                    (req.e2e_ms - req.ttft_ms)
                    if (req.e2e_ms is not None and req.ttft_ms is not None)
                    else None
                ),
            },
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model": model,
            "provider": provider,
            "project_name": project_name,
        },
    ]
    return spans


def build_summary_payload(
    result: BenchmarkResult,
    requests: Sequence[CapturedRequest],
    *,
    run_id: str,
    experiment_name: str,
    run_cfg: Dict[str, Any],
    tags: Sequence[str],
) -> Dict[str, Any]:
    """The one lean `benchmark-summary` trace for the whole run."""
    start, end = benchmark_window(requests)
    if start is None:
        start = _parse_result_date(result.date)
        if start is not None and result.aggregate.duration_s:
            end = start + _dt.timedelta(seconds=result.aggregate.duration_s)

    return {
        "name": SUMMARY_TRACE_NAME,
        "start_time": start,
        "end_time": end,
        "input": run_descriptor(result, run_cfg),
        "output": result.aggregate.grouped(),
        # Full raw result, minus the big per-request arrays.
        "metadata": {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "vllm_version": run_cfg.get("vllm_version"),
            **result.run_config(),
        },
        "tags": list(tags) + ["summary"],
        "feedback_scores": result.aggregate.headline_feedback_scores(),
    }


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _parse_result_date(date: Optional[str]) -> Optional[_dt.datetime]:
    """vLLM writes `date` as `%Y%m%d-%H%M%S` in local time."""
    if not date:
        return None
    try:
        return _dt.datetime.strptime(date, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def sync_result(
    result: BenchmarkResult,
    samples: Sequence[Sample],
    platform_cfg,
    *,
    client=None,
    experiment_name: Optional[str] = None,
    dataset_description: Optional[str] = None,
    extra_experiment_config: Optional[Dict[str, Any]] = None,
    source_label: Optional[str] = None,
    captured: Optional[Sequence[CapturedRequest]] = None,
    run_cfg: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> SyncReport:
    """Upload dataset items, request traces with spans, and the experiment."""
    from opik.api_objects.experiment import experiment_item as platform_experiment_item

    owns_client = client is None
    if owns_client:
        client = build_platform_client(platform_cfg)

    exp_name = experiment_name or render_experiment_name(platform_cfg.experiment_name, result)
    run_id = run_id or uuid.uuid4().hex[:12]
    run_cfg = dict(run_cfg or {})
    requests: List[CapturedRequest] = list(captured or [])

    report = SyncReport(
        project_name=platform_cfg.project_name,
        dataset_name=platform_cfg.dataset_name,
        experiment_name=exp_name,
        run_id=run_id,
    )

    samples = list(samples)
    if not samples and not requests and not dataset_exists(client, platform_cfg.dataset_name):
        raise SyncError(
            f"no benchmark samples to upload and Benchmark 平台 dataset "
            f"{platform_cfg.dataset_name!r} does not exist. Use "
            f"benchmark.dataset_name: custom with a JSONL file so the prompts "
            f"can be synced, point benchmark_platform.dataset_name at an "
            f"existing dataset, or run with --no-sync / "
            f"benchmark_platform.enabled: false."
        )

    try:
        # ---------------- dataset ------------------------------------------
        dataset = client.get_or_create_dataset(
            name=platform_cfg.dataset_name,
            description=dataset_description
            or "vllm bench serve prompt samples synced by vllm-bench-eval",
        )
        existing = dataset.get_items()
        existing_by_sample_id = index_dataset_items(existing)
        to_insert = [
            sample.as_dataset_item(source=source_label)
            for sample in samples
            if sample.sample_id not in existing_by_sample_id
        ]
        if to_insert:
            dataset.insert(to_insert)
            existing = dataset.get_items()
            existing_by_sample_id = index_dataset_items(existing)
        report.dataset_items_inserted = len(to_insert)
        report.dataset_items_total = len(existing)

        # ---------------- experiment ---------------------------------------
        experiment = client.create_experiment(
            dataset_name=platform_cfg.dataset_name,
            name=exp_name,
            experiment_config=build_experiment_config(
                result, {"run_id": run_id, **(extra_experiment_config or {})}
            ),
        )
        report.experiment_id = getattr(experiment, "id", None)

        tags = [t for t in (
            *platform_cfg.tags,
            result.model_id,
            result.backend,
            run_cfg.get("dataset_name"),
        ) if t]
        provider = run_cfg.get("provider") or result.backend or "openai"

        # ---------------- per-request traces --------------------------------
        references = []
        for req in requests:
            if req.sample_id is None:
                report.unaligned_requests += 1
            payload = build_request_trace_payload(
                req,
                run_id=run_id,
                experiment_name=exp_name,
                result=result,
                run_cfg=run_cfg,
                tags=tags,
            )
            trace = client.trace(project_name=platform_cfg.project_name, **payload)
            report.traces_created += 1

            for span_kwargs in build_request_spans(
                req,
                trace_id=trace.id,
                model=result.model_id,
                provider=provider,
                project_name=platform_cfg.project_name,
            ):
                client.span(**span_kwargs)
                report.spans_created += 1

            dataset_item_id = (
                existing_by_sample_id.get(req.sample_id) if req.sample_id else None
            )
            if dataset_item_id:
                references.append(
                    platform_experiment_item.ExperimentItemReferences(
                        dataset_item_id=dataset_item_id,
                        trace_id=trace.id,
                    )
                )

        if references:
            experiment.insert(references)
            report.experiment_items_created = len(references)
        elif requests:
            report.warnings.append(
                "no request could be linked to a dataset item; "
                "the experiment has no items"
            )

        # ---------------- summary trace -------------------------------------
        summary_trace = client.trace(
            project_name=platform_cfg.project_name,
            **build_summary_payload(
                result,
                requests,
                run_id=run_id,
                experiment_name=exp_name,
                run_cfg=run_cfg,
                tags=tags,
            ),
        )
        report.summary_trace_id = summary_trace.id
    finally:
        try:
            client.flush()
        except Exception as exc:  # pragma: no cover - best effort
            report.warnings.append(f"flush failed: {exc}")

    return report
