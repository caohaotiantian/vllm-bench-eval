import pytest

from vllm_bench_platform.config import load_config
from vllm_bench_platform.runner import (
    RunnerError,
    build_command,
    build_env,
    redact_command,
    resolve_docker_user,
    rewrite_for_container,
)


@pytest.fixture
def cfg(tmp_path, samples_file):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
server:
  base_url: http://127.0.0.1:1234
  endpoint: /v1/completions
  backend: openai
  model: demo-model
  tokenizer: Qwen/Qwen3-8B
benchmark:
  dataset_name: custom
  dataset_path: {samples_file}
  num_prompts: 3
  max_concurrency: 2
  result_dir: {tmp_path / "results"}
runner:
  mode: docker
  docker_image: img:tag
  docker_platform: linux/arm64
  hf_cache_dir: {tmp_path / "hf"}
""",
        encoding="utf-8",
    )
    return load_config(cfg_file)


def test_rewrite_for_container():
    assert rewrite_for_container("http://127.0.0.1:1234", "host.docker.internal") == \
        "http://host.docker.internal:1234"
    assert rewrite_for_container("http://localhost:8000", "hg") == "http://hg:8000"
    assert rewrite_for_container("https://api.example.com/v1", "hg") == "https://api.example.com/v1"


def test_docker_command_contains_required_flags(cfg):
    plan = build_command(cfg, result_filename="r.json")
    command, result_path = plan.argv, plan.result_path
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--platform" in command and "linux/arm64" in command
    assert "img:tag" in command
    # capture mode runs the shared launcher inside the container
    assert "vllm_bench_platform.vllm_entry" in command
    assert "--vbp-capture" in command
    # the container must talk to the host gateway, not to its own localhost
    assert "http://host.docker.internal:1234" in command
    # the metrics we care about
    i = command.index("--percentile-metrics")
    assert command[i + 1] == "ttft,tpot,itl,e2el"
    for flag in ("--save-result", "--save-detailed", "--custom-skip-chat-template"):
        assert flag in command
    assert result_path.name == "r.json"


def test_percentile_metrics_always_include_e2el(tmp_path, samples_file):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        f"""
server: {{model: m, tokenizer: t}}
benchmark:
  dataset_path: {samples_file}
  percentile_metrics: "ttft,itl"
  result_dir: {tmp_path / "res"}
runner: {{mode: docker}}
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.benchmark.percentile_metrics == "ttft,itl,tpot,e2el"


def test_missing_tokenizer_is_rejected(cfg):
    cfg.server.tokenizer = ""
    with pytest.raises(RunnerError, match="tokenizer"):
        build_command(cfg)


def test_missing_model_is_rejected(cfg):
    cfg.server.model = ""
    with pytest.raises(RunnerError, match="model"):
        build_command(cfg)


def test_native_mode_requires_vllm_on_path(cfg):
    cfg.runner.mode = "native"
    cfg.runner.capture = False          # the plain `vllm bench serve` path
    cfg.runner.vllm_bin = "definitely-not-a-real-binary-xyz"
    with pytest.raises(RunnerError, match="not found on PATH"):
        build_command(cfg)


# --- review fixes ----------------------------------------------------------


def test_docker_run_is_named_so_it_can_be_killed(cfg):
    plan = build_command(cfg, result_filename="r.json")
    assert "--name" in plan.argv
    assert plan.container_name
    assert plan.argv[plan.argv.index("--name") + 1] == plan.container_name


def test_api_key_never_reaches_the_argv(cfg):
    cfg.server.api_key = "sk-super-secret"
    plan = build_command(cfg, result_filename="r.json")
    assert all("sk-super-secret" not in part for part in plan.argv)
    # forwarded by name only ...
    i = [n for n, part in enumerate(plan.argv) if part == "-e"]
    assert "OPENAI_API_KEY" in [plan.argv[n + 1] for n in i]
    # ... with the value in the child environment
    assert plan.env["OPENAI_API_KEY"] == "sk-super-secret"


def test_configured_api_key_beats_a_preexported_one(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    cfg.server.api_key = "from-config"
    assert build_env(cfg)["OPENAI_API_KEY"] == "from-config"


def test_native_mode_also_gets_hf_env(cfg, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg.runner.mode = "native"
    cfg.runner.hf_token = "hf_tok"
    env = build_env(cfg)
    assert env["HF_TOKEN"] == "hf_tok"
    assert env["HF_HOME"].endswith("hf")          # runner.hf_cache_dir, absolute
    assert "/hf-cache" not in env["HF_HOME"]


def test_docker_mode_points_hf_home_at_the_mount(cfg):
    env = build_env(cfg)
    assert env["HF_HOME"] == "/hf-cache"


def test_redact_command_hides_secret_values():
    argv = ["docker", "run", "-e", "OPENAI_API_KEY=sk-abc", "-e", "HF_TOKEN", "img"]
    out = redact_command(argv)
    assert "OPENAI_API_KEY=<redacted>" in out
    assert "sk-abc" not in " ".join(out)
    # a bare name (the form we actually use) is left alone
    assert "HF_TOKEN" in out


def test_display_is_redacted(cfg):
    cfg.server.api_key = "sk-nope"
    assert "sk-nope" not in build_command(cfg, result_filename="r.json").display()


def test_docker_user_auto_only_applies_on_linux(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("os.getuid", lambda: 1000, raising=False)
    monkeypatch.setattr("os.getgid", lambda: 1000, raising=False)
    assert resolve_docker_user("auto") == "1000:1000"
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert resolve_docker_user("auto") is None
    assert resolve_docker_user("1:1") == "1:1"
    assert resolve_docker_user(None) is None


def test_docker_user_appears_on_linux(cfg, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("os.getuid", lambda: 501, raising=False)
    monkeypatch.setattr("os.getgid", lambda: 20, raising=False)
    argv = build_command(cfg, result_filename="r.json").argv
    assert "--user" in argv and "501:20" in argv


def test_ready_check_flag_can_be_omitted_for_old_vllm(cfg):
    assert "--ready-check-timeout-sec" in build_command(cfg, result_filename="r.json").argv
    cfg.server.ready_check_timeout_sec = None
    assert "--ready-check-timeout-sec" not in build_command(cfg, result_filename="r.json").argv


def test_timeout_is_reported_as_runner_error(cfg, monkeypatch):
    """A hung benchmark must surface a RunnerError, not a raw TimeoutExpired."""
    from vllm_bench_platform import runner as runner_mod

    killed = []
    monkeypatch.setattr(runner_mod, "_kill_container", lambda name: killed.append(name))
    cfg.runner.timeout_sec = 0.3
    # `sleep` stands in for a benchmark that never returns
    monkeypatch.setattr(
        runner_mod, "build_command",
        lambda c, result_filename=None, container_name=None: runner_mod.BenchCommand(
            argv=["sleep", "30"],
            result_path=c.path(c.benchmark.result_dir) / "x.json",
            env={}, container_name="fake-container",
        ),
    )
    with pytest.raises(RunnerError, match="timeout_sec"):
        runner_mod.run_benchmark(cfg, echo=False)
    assert killed == ["fake-container"]


# --- capture mode ----------------------------------------------------------


def test_capture_off_falls_back_to_plain_entrypoint(cfg):
    cfg.runner.capture = False
    plan = build_command(cfg, result_filename="r.json")
    assert "vllm-bench-serve" in plan.argv
    assert "--vbp-capture" not in plan.argv
    assert plan.capture_path is None


def test_capture_sidecar_sits_next_to_the_result(cfg):
    plan = build_command(cfg, result_filename="r.json")
    assert plan.capture_path.name == "r.json.requests.jsonl"
    assert plan.capture_path.parent == plan.result_path.parent


def test_docker_capture_mounts_the_package(cfg):
    argv = build_command(cfg, result_filename="r.json").argv
    joined = " ".join(argv)
    assert "/opt/vbp/vllm_bench_platform:ro" in joined
    assert "PYTHONPATH=/opt/vbp" in joined
    # the container writes the sidecar into the mounted results dir
    i = argv.index("--vbp-capture")
    assert argv[i + 1] == "/work/results/r.json.requests.jsonl"


def test_native_capture_uses_a_python_that_has_vllm(cfg):
    import sys as _sys

    cfg.runner.mode = "native"
    argv = build_command(cfg, result_filename="r.json").argv
    assert argv[0] == _sys.executable
    assert argv[1:3] == ["-m", "vllm_bench_platform.vllm_entry"]

    cfg.runner.python = "definitely-not-a-real-python-xyz"
    with pytest.raises(RunnerError, match="runner.python"):
        build_command(cfg, result_filename="r.json")
