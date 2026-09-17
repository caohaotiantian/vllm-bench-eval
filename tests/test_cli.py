"""CLI-level behaviour that is easy to get wrong (exit codes, warnings)."""

from typer.testing import CliRunner

from vllm_bench_platform.cli import app, failure_summary
from vllm_bench_platform.results import parse_result_dict, parse_result_file

runner = CliRunner()


def test_clean_run_has_no_failure_summary(result_file):
    assert failure_summary(parse_result_file(result_file)) is None


def test_incomplete_run_is_reported(result_dict):
    result_dict["completed"] = 2          # num_prompts is 3
    assert "2/3" in failure_summary(parse_result_dict(result_dict))


def test_request_errors_are_reported_with_distinct_messages(result_dict):
    result_dict["errors"] = ["", "connection refused", "connection refused", ]
    msg = failure_summary(parse_result_dict(result_dict))
    assert "2 request error(s)" in msg
    assert msg.count("connection refused") == 1   # distinct, not repeated


def test_many_distinct_errors_are_truncated(result_dict):
    result_dict["errors"] = ["e1", "e2", "e3"]
    result_dict["input_lens"] = [1, 2, 3]
    result_dict["output_lens"] = [1, 2, 3]
    result_dict["ttfts"] = [0.1, 0.1, 0.1]
    result_dict["itls"] = [[], [], []]
    result_dict["generated_texts"] = ["", "", ""]
    msg = failure_summary(parse_result_dict(result_dict))
    assert "e1" in msg and "e2" in msg and "e3" in msg


def test_show_command_prints_the_target_metrics(result_file):
    res = runner.invoke(app, ["show", str(result_file)])
    assert res.exit_code == 0
    for token in ("TTFT", "TPOT", "E2EL", "TPS"):
        assert token in res.stdout


def test_help_lists_every_command():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for cmd in ("check", "prepare-dataset", "run", "sync", "show"):
        assert cmd in res.stdout


def test_version():
    res = runner.invoke(app, ["--version"])
    assert res.exit_code == 0
    assert res.stdout.strip()


def test_sync_rejects_a_missing_file(tmp_path):
    res = runner.invoke(app, ["sync", str(tmp_path / "nope.json")])
    assert res.exit_code == 2


def _failing_run(monkeypatch, tmp_path, result_dict, cfg_file):
    """Wire `run` up to a benchmark whose requests failed."""
    import json

    from vllm_bench_platform import cli as cli_mod

    result_dict["completed"] = 1
    result_dict["errors"] = ["", "connection reset by peer", ""]
    result_dict["output_lens"] = [5, 0, 3]
    result_dict["ttfts"] = [0.10, 0.0, 0.15]
    result_dict["itls"] = [[0.02] * 4, [], [0.01, 0.01]]
    out = tmp_path / "r.json"
    out.write_text(json.dumps(result_dict), encoding="utf-8")

    class Outcome:
        result_path = out
        duration_s = 1.0

    monkeypatch.setattr(cli_mod, "run_benchmark", lambda cfg, result_filename=None: Outcome())
    return cfg_file


def _minimal_cfg(tmp_path, samples_file):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
server: {{model: m, tokenizer: t}}
benchmark: {{dataset_path: {samples_file}, result_dir: {tmp_path / "res"}}}
benchmark_platform: {{enabled: false}}
""",
        encoding="utf-8",
    )
    return cfg_file


def test_run_exits_nonzero_when_requests_failed(monkeypatch, tmp_path, result_dict, samples_file):
    cfg_file = _failing_run(monkeypatch, tmp_path, result_dict, _minimal_cfg(tmp_path, samples_file))
    res = runner.invoke(app, ["run", "--config", str(cfg_file)])
    assert res.exit_code == 1
    assert "connection reset by peer" in res.stdout
    assert "1/3" in res.stdout


def test_allow_failures_restores_exit_zero(monkeypatch, tmp_path, result_dict, samples_file):
    cfg_file = _failing_run(monkeypatch, tmp_path, result_dict, _minimal_cfg(tmp_path, samples_file))
    res = runner.invoke(app, ["run", "--config", str(cfg_file), "--allow-failures"])
    assert res.exit_code == 0
    assert "WARN" in res.stdout


def test_clean_run_exits_zero(monkeypatch, tmp_path, result_dict, samples_file):
    import json

    from vllm_bench_platform import cli as cli_mod

    out = tmp_path / "ok.json"
    out.write_text(json.dumps(result_dict), encoding="utf-8")

    class Outcome:
        result_path = out
        duration_s = 1.0

    monkeypatch.setattr(cli_mod, "run_benchmark", lambda cfg, result_filename=None: Outcome())
    res = runner.invoke(app, ["run", "--config", str(_minimal_cfg(tmp_path, samples_file))])
    assert res.exit_code == 0
