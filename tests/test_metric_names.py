"""Display names: Chinese labels, English for industry-standard terms."""

import pytest

from vllm_bench_platform import metric_names as mn
from vllm_bench_platform.platform_sync import (
    SPAN_DECODE,
    SPAN_PREFILL,
    SUMMARY_TRACE_NAME,
)

# Terms that must stay English wherever they appear.
KEEP_ENGLISH = ("TTFT", "TPOT", "ITL", "P50", "P90", "P99", "tokens/s", "req/s", "ms")


def test_per_request_score_names():
    assert mn.REQUEST_SCORE_NAMES == {
        "ttft_ms": "TTFT(ms)",
        "tpot_ms": "TPOT(ms)",
        "e2e_ms": "端到端延迟(ms)",
        "output_tokens_per_s": "输出吞吐(tokens/s)",
        "success": "请求成功",
    }


def test_per_request_reasons_are_chinese():
    assert mn.request_score("ttft_ms")[1] == "首 token 时间"
    assert mn.request_score("tpot_ms")[1] == "每输出 token 时间（不含首 token）"
    assert all(mn.request_score(k)[1] for k in mn.REQUEST_SCORE_NAMES)


@pytest.mark.parametrize(
    "stat,metric,expected",
    [
        ("mean", "ttft", "TTFT均值(ms)"),
        ("p50", "ttft", "TTFT P50(ms)"),
        ("p99", "ttft", "TTFT P99(ms)"),
        ("mean", "tpot", "TPOT均值(ms)"),
        ("p50", "tpot", "TPOT P50(ms)"),
        ("p99", "tpot", "TPOT P99(ms)"),
        ("mean", "e2el", "端到端延迟均值(ms)"),
        ("p50", "e2el", "端到端延迟 P50(ms)"),
        ("p99", "e2el", "端到端延迟 P99(ms)"),
        ("median", "itl", "ITL中位数(ms)"),
        ("p95", "itl", "ITL P95(ms)"),
    ],
)
def test_summary_percentile_names(stat, metric, expected):
    assert mn.percentile_score_name(stat, metric) == expected


def test_summary_throughput_names():
    assert mn.summary_score("output_throughput_tps")[0] == "输出吞吐(tokens/s)"
    assert mn.summary_score("total_token_throughput_tps")[0] == "总吞吐(tokens/s)"
    assert mn.summary_score("request_throughput_rps")[0] == "请求吞吐(req/s)"
    assert mn.summary_score("completed_ratio")[0] == "请求完成率"


def test_translate_grouped():
    raw = {
        "ttft_ms": {"mean": 1.0, "median": 2.0, "p50": 2.0, "p90": 3.0, "p99": 4.0},
        "itl_ms": {"mean": 1.0},
        "e2el_ms": {"mean": 5.0},
        "throughput": {"request_rps": 1.0, "output_tps": 2.0, "total_tps": 3.0},
        "counts": {"completed": 3, "num_prompts": 3, "failed": 0,
                   "total_input_tokens": 10, "total_output_tokens": 20, "duration_s": 1.5},
    }
    out = mn.translate_grouped(raw)
    assert set(out) == {"TTFT(ms)", "ITL(ms)", "端到端延迟(ms)", "吞吐", "计数"}
    # mean/median translated, percentiles left alone
    assert out["TTFT(ms)"] == {"均值": 1.0, "中位数": 2.0, "p50": 2.0, "p90": 3.0, "p99": 4.0}
    assert out["吞吐"] == {"请求(req/s)": 1.0, "输出(tokens/s)": 2.0, "总计(tokens/s)": 3.0}
    assert out["计数"] == {
        "完成请求数": 3, "请求总数": 3, "失败请求数": 0,
        "输入token总数": 10, "输出token总数": 20, "总耗时(s)": 1.5,
    }


def test_translate_grouped_passes_through_unknown_keys():
    assert mn.translate_grouped({"weird": {"x": 1}}) == {"weird": {"x": 1}}
    assert mn.translate_grouped({"counts": 5}) == {"计数": 5}


def test_trace_and_span_names():
    assert SUMMARY_TRACE_NAME == "压测汇总"
    assert SPAN_PREFILL == "预填充(TTFT)"
    assert SPAN_DECODE == "解码"
    # request traces stay `req-NNNN`: an index, not prose
    assert mn.REQUEST_TRACE_PREFIX == "req-"


def test_industry_terms_are_not_translated():
    everything = (
        list(mn.REQUEST_SCORE_NAMES.values())
        + list(mn.SUMMARY_SCORE_NAMES.values())
        + list(mn.GROUP_NAMES.values())
        + list(mn.THROUGHPUT_NAMES.values())
        + [mn.percentile_score_name(s, m)
           for s in ("mean", "p50", "p90", "p99") for m in mn.METRIC_LABELS]
    )
    blob = " ".join(everything)
    for term in KEEP_ENGLISH:
        assert term in blob, f"{term} should appear untranslated"
    # ...and none of them got a Chinese replacement
    assert "首字延迟" not in blob and "每令牌" not in blob


def test_names_are_plain_text_and_json_safe():
    """Platform score names must survive a JSON round-trip unescaped."""
    import json

    names = list(mn.REQUEST_SCORE_NAMES.values()) + list(mn.SUMMARY_SCORE_NAMES.values())
    for name in names:
        assert json.loads(json.dumps(name, ensure_ascii=False)) == name
        assert "\n" not in name and "\t" not in name
        assert name.strip() == name and name
