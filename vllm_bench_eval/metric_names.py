"""Display names for everything the Benchmark 平台 shows to a human.

**This module is the only place display names are defined.** Per-request
feedback scores, summary feedback scores, the summary trace's grouped `output`,
the span names and the summary trace name all resolve through it.

Naming rule
-----------
Labels are Chinese, except for industry-standard terms that are universally
written in English and would only get harder to read if translated:
**TTFT, TPOT, ITL, P50/P90/P95/P99, tokens/s, req/s, ms, s**.

What stays English
------------------
Only *display* strings are translated. Machine-readable fields keep their
English keys so that dashboards, diffs and scripts stay stable:

* the raw `vllm bench serve` result JSON (untouched, it is vLLM's own format)
* `experiment_config` (`metrics` / `run`)
* every trace and span `metadata` key (`ttft_ms`, `itl_stats`, `run_id` ...)
* the capture sidecar
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# --- trace / span names -----------------------------------------------------

SUMMARY_TRACE_NAME = "压测汇总"
REQUEST_TRACE_PREFIX = "req-"          # req-0000: an index, not prose
SPAN_PREFILL = "预填充(TTFT)"
SPAN_DECODE = "解码"

# --- metric labels ----------------------------------------------------------

#: vLLM metric id -> display label.
METRIC_LABELS = {
    "ttft": "TTFT",
    "tpot": "TPOT",
    "itl": "ITL",
    "e2el": "端到端延迟",
}

#: Statistic id -> display label. Percentiles keep their P50/P90/P99 spelling.
STAT_LABELS = {
    "mean": "均值",
    "median": "中位数",
    "std": "标准差",
}

# --- per-request feedback scores --------------------------------------------

REQUEST_SCORE_NAMES = {
    "ttft_ms": "TTFT(ms)",
    "tpot_ms": "TPOT(ms)",
    "e2e_ms": "端到端延迟(ms)",
    "output_tokens_per_s": "输出吞吐(tokens/s)",
    "success": "请求成功",
}

REQUEST_SCORE_REASONS = {
    "ttft_ms": "首 token 时间",
    "tpot_ms": "每输出 token 时间（不含首 token）",
    "e2e_ms": "该请求端到端耗时",
    "output_tokens_per_s": "该请求的输出吞吐",
    "success": "1 = 请求成功完成",
}

# --- summary feedback scores ------------------------------------------------

SUMMARY_SCORE_NAMES = {
    "output_throughput_tps": "输出吞吐(tokens/s)",
    "total_token_throughput_tps": "总吞吐(tokens/s)",
    "request_throughput_rps": "请求吞吐(req/s)",
    "completed_ratio": "请求完成率",
}

SUMMARY_SCORE_REASONS = {
    "output_throughput_tps": "有效输出吞吐：生成 token / 秒",
    "total_token_throughput_tps": "总吞吐：(输入 + 生成) token / 秒",
    "request_throughput_rps": "每秒完成的请求数",
    "completed_ratio": "完成请求数 / 请求总数",
}

# --- summary trace `output` (the grouped view) ------------------------------

GROUP_NAMES = {
    "ttft_ms": "TTFT(ms)",
    "tpot_ms": "TPOT(ms)",
    "itl_ms": "ITL(ms)",
    "e2el_ms": "端到端延迟(ms)",
    "throughput": "吞吐",
    "counts": "计数",
    "request_output_tps": "单请求输出吞吐(tokens/s)",
    "request_total_tps": "单请求总吞吐(tokens/s)",
    "tools": "工具与验证",
}

THROUGHPUT_NAMES = {
    "request_rps": "请求(req/s)",
    "output_tps": "输出(tokens/s)",
    "total_tps": "总计(tokens/s)",
}

COUNT_NAMES = {
    "completed": "完成请求数",
    "num_prompts": "请求总数",
    "failed": "失败请求数",
    "total_input_tokens": "输入token总数",
    "total_output_tokens": "输出token总数",
    "duration_s": "总耗时(s)",
    "total_tokens": "成功请求token总数",
    "known_consumed_tokens": "已知消耗token总数",
}

TOOL_NAMES = {
    "calls": "工具调用数",
    "failed": "工具失败数",
    "incomplete": "未结束工具数",

    "verification_calls": "验证次数",
    "verification_failed": "验证失败数",
    "verification_incomplete": "未结束验证数",

}

_INNER_NAMES = {
    "throughput": THROUGHPUT_NAMES,
    "counts": COUNT_NAMES,
    "tools": TOOL_NAMES,
}


# ---------------------------------------------------------------------------


def percentile_score_name(stat: str, metric: str) -> str:
    """e.g. ("mean", "ttft") -> "TTFT均值(ms)"; ("p99", "e2el") -> "端到端延迟 P99(ms)"."""
    label = METRIC_LABELS.get(metric, metric.upper())
    if stat in STAT_LABELS:
        return f"{label}{STAT_LABELS[stat]}(ms)"
    return f"{label} {stat.upper()}(ms)"


def percentile_score_reason(stat: str, metric: str) -> str:
    label = METRIC_LABELS.get(metric, metric.upper())
    if stat in STAT_LABELS:
        return f"{label} {STAT_LABELS[stat]}"
    return f"{label} {stat.upper()} 分位"


def request_score(key: str) -> tuple[str, str]:
    """Display name + reason for a per-request score key."""
    return REQUEST_SCORE_NAMES.get(key, key), REQUEST_SCORE_REASONS.get(key, "")


def summary_score(key: str) -> tuple[str, str]:
    """Display name + reason for a run-level score key."""
    return SUMMARY_SCORE_NAMES.get(key, key), SUMMARY_SCORE_REASONS.get(key, "")


def stat_label(stat: str) -> str:
    """Inner key of a grouped metric block: `mean` -> `均值`, `p90` stays `p90`."""
    return STAT_LABELS.get(stat, stat)


def translate_grouped(grouped: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the summary `output` dict, keys and inner keys alike."""
    out: Dict[str, Any] = {}
    for group, values in grouped.items():
        name = GROUP_NAMES.get(group, group)
        if not isinstance(values, dict):
            out[name] = values
            continue
        inner = _INNER_NAMES.get(group)
        if inner is not None:
            out[name] = {inner.get(k, k): ({stat_label(stat): value for stat, value in v.items()} if isinstance(v, dict) else v)
                         for k, v in values.items() if group != "tools" or k in TOOL_NAMES}
        else:
            # a metric block: mean/median/std -> Chinese, percentiles unchanged
            out[name] = {stat_label(k): v for k, v in values.items() if group != "tools" or k in TOOL_NAMES}
    return out

# Agent aggregates keep request performance separate from workload throughput.
AGENT_TPS_LABELS = {
    "request_output_tps": "单请求输出吞吐",
    "request_total_tps": "单请求总吞吐",
}
AGENT_WORKLOAD_SCORE_NAMES = {
    "output_tps": "任务工作负载输出吞吐(tokens/s)",
    "total_tps": "任务工作负载总吞吐(tokens/s)",
    "request_rps": "任务工作负载请求吞吐(req/s)",
}
AGENT_TOOL_SCORE_NAMES = {
    "failed": "工具失败数",
}


def agent_tps_score_name(stat: str, group: str) -> str:
    label = AGENT_TPS_LABELS[group]
    suffix = STAT_LABELS.get(stat, f" {stat.upper()}")
    return f"{label}{suffix}(tokens/s)"

AGENT_TASK_SUCCESS_SCORE = "任务成功率(%)"

CONVERSATION_DURATION_SCORE = "会话总耗时(s)"
AGENT_TASK_DURATION_SCORE = "任务总耗时(s)"
