"""Read the per-request sidecar written by :mod:`vllm_bench_platform.vllm_entry`.

The sidecar (``<result>.requests.jsonl``) is what makes a *useful* trace
possible: the plain `vllm bench serve` result JSON has no prompts, no
request ids and no wall-clock timestamps, so every trace would land on the
sync timestamp with no duration and no prefill/decode split.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import metric_names

SIDECAR_SUFFIX = ".requests.jsonl"
# Hard cap on how many inter-token latencies we copy into trace metadata.
MAX_ITL_VALUES = 512


def sidecar_path_for(result_path: str | Path) -> Path:
    """`results/run.json` -> `results/run.json.requests.jsonl`."""
    return Path(str(result_path) + SIDECAR_SUFFIX)


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def itl_stats(itl_ms: Sequence[float]) -> Dict[str, float]:
    """Compact distribution summary for the inter-token latencies."""
    values = sorted(float(v) for v in itl_ms)
    if not values:
        return {"count": 0}

    def pct(p: float) -> float:
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * (p / 100.0)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 4),
        "p50": round(pct(50), 4),
        "p90": round(pct(90), 4),
        "p99": round(pct(99), 4),
        "max": round(values[-1], 4),
    }


@dataclass
class CapturedRequest:
    """One measured request, reconstructed from the sidecar."""

    index: int
    request_id: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None
    api_url: Optional[str] = None
    input_len: Optional[int] = None
    max_output_len: Optional[int] = None
    ignore_eos: Optional[bool] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)

    start_wall: Optional[float] = None
    end_wall: Optional[float] = None

    success: bool = False
    ttft_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    itl_ms: List[float] = field(default_factory=list)
    generated_text: str = ""
    output_len: Optional[int] = None
    error: Optional[str] = None

    # Filled in by the dataset alignment step.
    sample_id: Optional[str] = None

    # ---- derived -------------------------------------------------------
    @property
    def start_dt(self) -> Optional[_dt.datetime]:
        return _to_dt(self.start_wall)

    @property
    def end_dt(self) -> Optional[_dt.datetime]:
        return _to_dt(self.end_wall)

    @property
    def first_token_dt(self) -> Optional[_dt.datetime]:
        """Wall-clock instant the first token arrived (end of prefill)."""
        if self.start_wall is None or self.ttft_ms is None:
            return None
        return _to_dt(self.start_wall + self.ttft_ms / 1000.0)

    @property
    def output_tps(self) -> Optional[float]:
        if not self.e2e_ms or not self.output_len:
            return None
        return self.output_len / (self.e2e_ms / 1000.0)

    def itl_summary(self) -> Dict[str, float]:
        return itl_stats(self.itl_ms)

    def feedback_scores(self) -> List[Dict[str, Any]]:
        """Latency scores only; token counts belong in metadata/usage.

        Names come from :mod:`vllm_bench_platform.metric_names` — the keys here
        stay English, only what the platform displays is translated.
        """
        scores: List[Dict[str, Any]] = []
        if self.success:
            for key, value in (
                ("ttft_ms", self.ttft_ms),
                ("tpot_ms", self.tpot_ms),
                ("e2e_ms", self.e2e_ms),
                ("output_tokens_per_s", self.output_tps),
            ):
                v = _num(value)
                if v is not None:
                    name, reason = metric_names.request_score(key)
                    scores.append({"name": name, "value": v, "reason": reason})
        name, reason = metric_names.request_score("success")
        scores.append(
            {"name": name, "value": 1.0 if self.success else 0.0, "reason": reason}
        )
        return scores


def _to_dt(wall: Optional[float]) -> Optional[_dt.datetime]:
    if wall is None:
        return None
    try:
        return _dt.datetime.fromtimestamp(float(wall))
    except (OverflowError, OSError, ValueError):
        return None


def _request_sort_key(record: Dict[str, Any]) -> tuple:
    """Order by the numeric suffix of the request id, else by capture order.

    vLLM builds ids as ``<prefix><i>`` where *i* is the index of the request in
    the sampled list, i.e. the order in which requests were issued.
    """
    rid = record.get("request_id")
    if isinstance(rid, str):
        digits = ""
        for ch in reversed(rid):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        if digits:
            return (0, int(digits))
    return (1, record.get("capture_index", 0))


def load_sidecar(
    path: str | Path,
    include_warmup: bool = False,
    expected: Optional[int] = None,
) -> List[CapturedRequest]:
    """Parse the sidecar JSONL into measured requests, in issue order.

    ``expected`` (usually ``benchmark.num_prompts``) guards the warm-up
    heuristic: on vLLM < 0.10.2 there is no ``request_id`` to key on, so the
    launcher flags the *first* call as the warm-up. If that guess would leave
    the wrong number of requests while keeping everything gives exactly
    ``expected``, the guess is undone.
    """
    p = Path(path)
    all_records: List[Dict[str, Any]] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            all_records.append(record)

    records = (
        list(all_records)
        if include_warmup
        else [r for r in all_records if not r.get("is_warmup")]
    )
    if (
        not include_warmup
        and expected is not None
        and len(records) != expected
        and len(all_records) == expected
    ):
        # The heuristic mis-fired (e.g. the ready-check was skipped, so there
        # was no warm-up at all). Keep every captured request.
        records = list(all_records)

    records.sort(key=_request_sort_key)

    out: List[CapturedRequest] = []
    for idx, record in enumerate(records):
        itl_s = [x for x in (_num(v) for v in record.get("itl") or []) if x is not None]
        ttft_s = _num(record.get("ttft"))
        latency_s = _num(record.get("latency"))
        if latency_s is None and ttft_s is not None:
            latency_s = ttft_s + sum(itl_s)

        output_len = record.get("output_tokens")
        output_len = int(output_len) if _num(output_len) else None

        tpot_ms = None
        if latency_s is not None and ttft_s is not None and output_len and output_len > 1:
            tpot_ms = (latency_s - ttft_s) / (output_len - 1) * 1000.0

        error = record.get("error") or None
        success = bool(record.get("success")) and not error

        start_wall = _num(record.get("start_wall"))
        end_wall = _num(record.get("end_wall"))
        # Prefer the measured latency over the wrapper's wall-clock delta: the
        # latter includes the semaphore wait and coroutine scheduling.
        if start_wall is not None and latency_s is not None and end_wall is not None:
            end_wall = start_wall + latency_s

        extra = record.get("extra_body")
        out.append(
            CapturedRequest(
                index=idx,
                request_id=record.get("request_id"),
                prompt=record.get("prompt"),
                model=record.get("model_name") or record.get("model"),
                api_url=record.get("api_url"),
                input_len=int(record["prompt_len"]) if _num(record.get("prompt_len")) else None,
                max_output_len=int(record["output_len"]) if _num(record.get("output_len")) else None,
                ignore_eos=record.get("ignore_eos"),
                extra_body=extra if isinstance(extra, dict) else {},
                start_wall=start_wall,
                end_wall=end_wall,
                success=success,
                ttft_ms=ttft_s * 1000.0 if ttft_s is not None else None,
                e2e_ms=latency_s * 1000.0 if latency_s is not None else None,
                tpot_ms=tpot_ms,
                itl_ms=[x * 1000.0 for x in itl_s],
                generated_text=record.get("generated_text") or "",
                output_len=output_len,
                error=error,
            )
        )
    return out


def merge_with_result(requests: Sequence[CapturedRequest], result) -> List[CapturedRequest]:
    """Fill gaps in the capture from the `vllm bench serve` result arrays.

    Streaming endpoints often omit a `usage` block, leaving
    ``RequestFuncOutput.output_tokens`` at 0. vLLM recomputes the real count by
    re-tokenising the generated text, so `output_lens[i]` is authoritative.
    The sidecar (sorted by request id) and the detailed arrays share the same
    issue order, which :func:`load_sidecar` guarantees and the tests assert.
    """
    parsed = list(getattr(result, "requests", []) or [])
    for i, req in enumerate(requests):
        if i >= len(parsed):
            break
        ref = parsed[i]
        if not req.output_len and ref.output_len:
            req.output_len = ref.output_len
            if req.e2e_ms is not None and req.ttft_ms is not None and ref.output_len > 1:
                req.tpot_ms = (req.e2e_ms - req.ttft_ms) / (ref.output_len - 1)
        if req.input_len is None and ref.input_len is not None:
            req.input_len = ref.input_len
    # Last resort: the streamed chunk count is one more than the ITL count.
    for req in requests:
        if not req.output_len and req.itl_ms:
            req.output_len = len(req.itl_ms) + 1
    return list(requests)


def benchmark_window(requests: Sequence[CapturedRequest]) -> tuple[Optional[_dt.datetime], Optional[_dt.datetime]]:
    """Real wall-clock window: first request start -> last request end."""
    starts = [r.start_wall for r in requests if r.start_wall is not None]
    ends = [r.end_wall for r in requests if r.end_wall is not None]
    return (_to_dt(min(starts)) if starts else None, _to_dt(max(ends)) if ends else None)
