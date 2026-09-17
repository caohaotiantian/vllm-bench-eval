"""Parse the JSON produced by `vllm bench serve --save-result --save-detailed`.

Key facts about that file (verified against vllm 0.11.0 source,
``vllm/benchmarks/serve.py`` and ``vllm/benchmarks/datasets.py``):

* Aggregate keys: ``duration``, ``completed``, ``total_input_tokens``,
  ``total_output_tokens``, ``request_throughput``, ``output_throughput``,
  ``total_token_throughput`` and, for every metric listed in
  ``--percentile-metrics``, ``mean_/median_/std_/p{P}_{metric}_ms``.
* ``--save-detailed`` adds the per-request arrays ``input_lens``,
  ``output_lens``, ``ttfts``, ``itls``, ``generated_texts`` and ``errors``.
  They are produced by ``asyncio.gather(*tasks)`` where the tasks were created
  in the order of the sampled requests, so index *i* of every array refers to
  the same request.
* Per-request end-to-end latency is *not* stored, but in
  ``vllm/benchmarks/lib/endpoint_request_func.py`` ``output.latency`` is
  ``most_recent_timestamp - st`` and ``output.ttft`` is
  ``first_token_timestamp - st`` while ``itl`` holds every subsequent
  inter-token delta, therefore ``latency == ttft + sum(itl)`` exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Metrics whose percentile block we expose.
PERCENTILE_METRICS = ("ttft", "tpot", "itl", "e2el")


def _stat_sort_key(stat: str) -> tuple:
    """Order stats as mean, median, std, then percentiles ascending."""
    fixed = {"mean": 0, "median": 1, "std": 2}
    if stat in fixed:
        return (0, fixed[stat], 0.0)
    try:
        return (1, 0, float(stat.lstrip("p")))
    except ValueError:
        return (2, 0, 0.0)


def _num(value: Any) -> Optional[float]:
    """Return a finite float, or None for missing/NaN/inf values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


@dataclass
class RequestMetrics:
    """One benchmarked request."""

    index: int
    input_len: Optional[int] = None
    output_len: Optional[int] = None
    ttft_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    itl_ms: List[float] = field(default_factory=list)
    generated_text: Optional[str] = None
    error: Optional[str] = None
    # Filled in by align_requests_to_samples().
    prompt: Optional[str] = None
    sample_id: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return not self.error and (self.output_len or 0) > 0

    @property
    def output_tps(self) -> Optional[float]:
        """Output tokens per second for this single request (tokens / E2E)."""
        if not self.e2e_ms or not self.output_len:
            return None
        return self.output_len / (self.e2e_ms / 1000.0)

    def feedback_scores(self) -> List[Dict[str, Any]]:
        """Per-request platform feedback scores (only finite values are emitted).

        A failed request carries ``ttft = 0`` and an empty ``itl`` list in
        vLLM's output. Publishing those as 0 ms would drag every platform average
        towards zero and make a broken run look fast, so latency/throughput
        scores are emitted only for requests that actually completed. The
        ``success`` score is always emitted, so failures stay visible.
        """
        candidates: List[tuple[str, Optional[float], str]] = []
        if self.succeeded:
            candidates += [
                ("ttft_ms", self.ttft_ms, "Time to first token (ms)"),
                ("tpot_ms", self.tpot_ms, "Time per output token, excl. first (ms)"),
                ("e2e_ms", self.e2e_ms, "End-to-end request latency (ms)"),
                ("output_tokens_per_s", self.output_tps, "Output tokens/s for this request"),
            ]
        candidates += [
            ("output_tokens", float(self.output_len) if self.output_len is not None else None,
             "Generated tokens"),
            ("input_tokens", float(self.input_len) if self.input_len is not None else None,
             "Prompt tokens"),
            ("success", 1.0 if self.succeeded else 0.0, "1 = request completed"),
        ]
        scores: List[Dict[str, Any]] = []
        for name, value, reason in candidates:
            v = _num(value)
            if v is None:
                continue
            scores.append({"name": name, "value": v, "reason": reason})
        return scores


@dataclass
class AggregateMetrics:
    """Run-level metrics."""

    duration_s: Optional[float] = None
    completed: Optional[int] = None
    num_prompts: Optional[int] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    request_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    total_token_throughput: Optional[float] = None
    max_output_tokens_per_s: Optional[float] = None
    max_concurrent_requests: Optional[float] = None
    # metric -> stat -> value, e.g. percentiles["ttft"]["mean"] / ["p99"]
    percentiles: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get(self, metric: str, stat: str) -> Optional[float]:
        return self.percentiles.get(metric, {}).get(stat)

    def feedback_scores(self) -> List[Dict[str, Any]]:
        """Aggregate platform feedback scores for the summary trace."""
        candidates: List[tuple[str, Optional[float], str]] = [
            ("duration_s", self.duration_s, "Wall-clock benchmark duration"),
            ("completed_requests", float(self.completed) if self.completed is not None else None,
             "Successfully completed requests"),
            ("request_throughput_rps", self.request_throughput, "Requests per second"),
            ("output_throughput_tps", self.output_throughput,
             "有效输出吞吐: generated tokens / s"),
            ("total_token_throughput_tps", self.total_token_throughput,
             "总吞吐: (prompt + generated) tokens / s"),
            ("total_input_tokens",
             float(self.total_input_tokens) if self.total_input_tokens is not None else None,
             "Total prompt tokens"),
            ("total_output_tokens",
             float(self.total_output_tokens) if self.total_output_tokens is not None else None,
             "Total generated tokens"),
        ]
        labels = {
            "ttft": "TTFT",
            "tpot": "TPOT",
            "itl": "ITL",
            "e2el": "E2E latency",
        }
        # Emit whatever vLLM actually produced, so a custom
        # `--metric-percentiles 75,99.9` shows up on the platform too.
        for metric in sorted(self.percentiles):
            label = labels.get(metric, metric.upper())
            for stat in sorted(self.percentiles[metric], key=_stat_sort_key):
                candidates.append(
                    (
                        f"{stat}_{metric}_ms",
                        self.percentiles[metric][stat],
                        f"{label} {stat} (ms)",
                    )
                )
        scores: List[Dict[str, Any]] = []
        for name, value, reason in candidates:
            v = _num(value)
            if v is None:
                continue
            scores.append({"name": name, "value": v, "reason": reason})
        return scores

    def grouped(self) -> Dict[str, Any]:
        """Compact, grouped view for the summary trace output.

        A flat wall of 30+ keys is unreadable in the UI; grouping by metric
        keeps TTFT / TPOT / ITL / E2EL / throughput / counts legible.
        """
        def block(metric: str) -> Dict[str, float]:
            stats = self.percentiles.get(metric, {})
            return {
                k: round(v, 3)
                for k, v in stats.items()
                if k in ("mean", "median", "p50", "p90", "p95", "p99")
            }

        failed = None
        if self.num_prompts is not None and self.completed is not None:
            failed = max(self.num_prompts - self.completed, 0)

        out: Dict[str, Any] = {
            "ttft_ms": block("ttft"),
            "tpot_ms": block("tpot"),
            "itl_ms": block("itl"),
            "e2el_ms": block("e2el"),
            "throughput": {
                k: round(v, 3)
                for k, v in (
                    ("request_rps", self.request_throughput),
                    ("output_tps", self.output_throughput),
                    ("total_tps", self.total_token_throughput),
                )
                if v is not None
            },
            "counts": {
                k: v
                for k, v in (
                    ("completed", self.completed),
                    ("num_prompts", self.num_prompts),
                    ("failed", failed),
                    ("total_input_tokens", self.total_input_tokens),
                    ("total_output_tokens", self.total_output_tokens),
                    ("duration_s", round(self.duration_s, 3) if self.duration_s else None),
                )
                if v is not None
            },
        }
        return {k: v for k, v in out.items() if v}

    def headline_feedback_scores(self) -> List[Dict[str, Any]]:
        """The small set worth charting; everything else lives in metadata."""
        candidates: List[tuple[str, Optional[float], str]] = []
        labels = {"ttft": "TTFT", "tpot": "TPOT", "e2el": "E2E latency"}
        for metric, label in labels.items():
            stats = self.percentiles.get(metric, {})
            for stat in ("mean", "p50", "p99"):
                value = stats.get(stat)
                if value is None and stat == "p50":
                    value = stats.get("median")
                if value is not None:
                    candidates.append((f"{stat}_{metric}_ms", value, f"{label} {stat} (ms)"))
        candidates += [
            ("output_throughput_tps", self.output_throughput, "有效输出吞吐: generated tokens / s"),
            ("total_token_throughput_tps", self.total_token_throughput,
             "总吞吐: (prompt + generated) tokens / s"),
            ("request_throughput_rps", self.request_throughput, "Requests per second"),
        ]
        if self.completed is not None and self.num_prompts:
            candidates.append(
                ("completed_ratio", self.completed / self.num_prompts,
                 "completed / num_prompts")
            )
        scores: List[Dict[str, Any]] = []
        for name, value, reason in candidates:
            v = _num(value)
            if v is None:
                continue
            scores.append({"name": name, "value": v, "reason": reason})
        return scores

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "duration_s": self.duration_s,
            "completed": self.completed,
            "num_prompts": self.num_prompts,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "request_throughput_rps": self.request_throughput,
            "output_throughput_tps": self.output_throughput,
            "total_token_throughput_tps": self.total_token_throughput,
            "max_output_tokens_per_s": self.max_output_tokens_per_s,
            "max_concurrent_requests": self.max_concurrent_requests,
        }
        for metric, stats in self.percentiles.items():
            for stat, value in stats.items():
                out[f"{stat}_{metric}_ms"] = value
        return {k: v for k, v in out.items() if v is not None}


@dataclass
class BenchmarkResult:
    """A parsed `vllm bench serve` result file."""

    raw: Dict[str, Any]
    aggregate: AggregateMetrics
    requests: List[RequestMetrics]
    source_path: Optional[Path] = None

    # Convenience accessors for run identification / metadata.
    @property
    def model_id(self) -> str:
        return str(self.raw.get("model_id") or "unknown-model")

    @property
    def tokenizer_id(self) -> Optional[str]:
        v = self.raw.get("tokenizer_id")
        return str(v) if v else None

    @property
    def backend(self) -> str:
        return str(self.raw.get("backend") or self.raw.get("endpoint_type") or "openai")

    @property
    def date(self) -> Optional[str]:
        v = self.raw.get("date")
        return str(v) if v else None

    @property
    def has_detailed(self) -> bool:
        return bool(self.requests)

    def run_config(self) -> Dict[str, Any]:
        """Everything in the result file that is *not* a metric array."""
        skip = {
            "input_lens", "output_lens", "ttfts", "itls",
            "generated_texts", "errors", "rps_change_events",
        }
        return {k: v for k, v in self.raw.items() if k not in skip}


def _parse_percentiles(raw: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Collect ``{mean,median,std,pNN}_{metric}_ms`` keys into a nested dict."""
    out: Dict[str, Dict[str, float]] = {}
    for key, value in raw.items():
        if not key.endswith("_ms"):
            continue
        body = key[: -len("_ms")]
        stat, _, metric = body.partition("_")
        if not metric or metric not in PERCENTILE_METRICS:
            continue
        if stat not in ("mean", "median", "std") and not (
            stat.startswith("p") and stat[1:].replace(".", "", 1).isdigit()
        ):
            continue
        v = _num(value)
        if v is None:
            continue
        out.setdefault(metric, {})[stat] = v
    return out


def parse_result_file(path: str | Path) -> BenchmarkResult:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    result = parse_result_dict(raw)
    result.source_path = p
    return result


def parse_result_dict(raw: Dict[str, Any]) -> BenchmarkResult:
    if not isinstance(raw, dict):
        raise ValueError("vllm bench result must be a JSON object")

    aggregate = AggregateMetrics(
        duration_s=_num(raw.get("duration")),
        completed=int(raw["completed"]) if _num(raw.get("completed")) is not None else None,
        num_prompts=int(raw["num_prompts"]) if _num(raw.get("num_prompts")) is not None else None,
        total_input_tokens=(
            int(raw["total_input_tokens"]) if _num(raw.get("total_input_tokens")) is not None else None
        ),
        total_output_tokens=(
            int(raw["total_output_tokens"]) if _num(raw.get("total_output_tokens")) is not None else None
        ),
        request_throughput=_num(raw.get("request_throughput")),
        output_throughput=_num(raw.get("output_throughput")),
        total_token_throughput=_num(raw.get("total_token_throughput")),
        max_output_tokens_per_s=_num(raw.get("max_output_tokens_per_s")),
        max_concurrent_requests=_num(raw.get("max_concurrent_requests")),
        percentiles=_parse_percentiles(raw),
    )

    requests = _parse_requests(raw)
    return BenchmarkResult(raw=raw, aggregate=aggregate, requests=requests)


def _parse_requests(raw: Dict[str, Any]) -> List[RequestMetrics]:
    arrays = {
        name: raw.get(name)
        for name in ("input_lens", "output_lens", "ttfts", "itls", "generated_texts", "errors")
    }
    lengths = [len(v) for v in arrays.values() if isinstance(v, list)]
    if not lengths:
        return []  # the file was written without --save-detailed
    n = max(lengths)

    def at(name: str, i: int) -> Any:
        seq = arrays.get(name)
        if isinstance(seq, list) and i < len(seq):
            return seq[i]
        return None

    requests: List[RequestMetrics] = []
    for i in range(n):
        ttft_s = _num(at("ttfts", i))
        itls_raw = at("itls", i)
        itls_s = [x for x in (_num(v) for v in itls_raw or []) if x is not None]

        e2e_s: Optional[float] = None
        if ttft_s is not None:
            # latency == ttft + sum(itl); see module docstring.
            e2e_s = ttft_s + sum(itls_s)

        tpot_ms: Optional[float] = None
        out_len = at("output_lens", i)
        out_len_i = int(out_len) if _num(out_len) is not None else None
        if e2e_s is not None and ttft_s is not None and out_len_i and out_len_i > 1:
            tpot_ms = (e2e_s - ttft_s) / (out_len_i - 1) * 1000.0

        err = at("errors", i)
        requests.append(
            RequestMetrics(
                index=i,
                input_len=int(at("input_lens", i)) if _num(at("input_lens", i)) is not None else None,
                output_len=out_len_i,
                ttft_ms=ttft_s * 1000.0 if ttft_s is not None else None,
                e2e_ms=e2e_s * 1000.0 if e2e_s is not None else None,
                tpot_ms=tpot_ms,
                itl_ms=[x * 1000.0 for x in itls_s],
                generated_text=at("generated_texts", i),
                error=str(err) if err else None,
            )
        )
    return requests
