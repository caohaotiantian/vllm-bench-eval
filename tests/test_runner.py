import pytest

from vllm_bench_eval.config import load_config
from vllm_bench_eval.runner import (
    RunnerError,
    VllmCapabilities,
    build_command as _build_command,
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


def build_command(cfg, **kwargs):
    """build_command without shelling out to probe the target vLLM."""
    kwargs.setdefault("detect", False)
    return _build_command(cfg, **kwargs)


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
    assert "vllm_bench_eval.vllm_entry" in command
    assert "--vbe-capture" in command
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
    from vllm_bench_eval import runner as runner_mod

    killed = []
    def inspect_owned(plan, timeout, cleanup, identity=None):
        return ("running", "owned-container-id") if not killed else ("absent", None)
    monkeypatch.setattr(runner_mod, "_inspect_owned", inspect_owned)
    monkeypatch.setattr(runner_mod, "_docker_call", lambda args, timeout, cleanup: killed.append(args))
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
    assert any(args[:3] == ["kill", "--signal", "KILL"] and args[-1] == "owned-container-id" for args in killed)


# --- capture mode ----------------------------------------------------------


def test_capture_off_falls_back_to_plain_entrypoint(cfg):
    cfg.runner.capture = False
    plan = build_command(cfg, result_filename="r.json")
    assert "vllm-bench-serve" in plan.argv
    assert "--vbe-capture" not in plan.argv
    assert plan.capture_path is None


def test_capture_sidecar_sits_next_to_the_result(cfg):
    plan = build_command(cfg, result_filename="r.json")
    assert plan.capture_path.name == "r.json.requests.jsonl"
    assert plan.capture_path.parent == plan.result_path.parent


def test_docker_capture_mounts_the_package(cfg):
    argv = build_command(cfg, result_filename="r.json").argv
    joined = " ".join(argv)
    assert "/opt/vbe/vllm_bench_eval:ro" in joined
    assert "PYTHONPATH=/opt/vbe" in joined
    # the container writes the sidecar into the mounted results dir
    i = argv.index("--vbe-capture")
    assert argv[i + 1] == "/work/results/r.json.requests.jsonl"


def test_native_capture_uses_a_python_that_has_vllm(cfg):
    import sys as _sys

    cfg.runner.mode = "native"
    argv = build_command(cfg, result_filename="r.json").argv
    assert argv[0] == _sys.executable
    assert argv[1:3] == ["-m", "vllm_bench_eval.vllm_entry"]

    cfg.runner.python = "definitely-not-a-real-python-xyz"
    with pytest.raises(RunnerError, match="runner.python"):
        build_command(cfg, result_filename="r.json")


def stub_command(cfg, monkeypatch, script, *, name="stub.json", container=None):
    import os
    import sys
    from vllm_bench_eval import runner as runner_mod
    path = cfg.path(cfg.benchmark.result_dir) / name
    plan = runner_mod.BenchCommand(argv=[sys.executable, "-u", "-c", script], result_path=path,
        env=dict(os.environ), container_name=container, capture_path=path.with_name(path.name + ".requests.jsonl"),
        capabilities=runner_mod.VllmCapabilities(version="stub", detected=False))
    monkeypatch.setattr(runner_mod, "build_command", lambda config, result_filename=None: plan)
    cfg.runner.mode = "docker" if container else "native"
    cfg.runner.stop_grace_sec = .03
    cfg.runner.cleanup_timeout_sec = .2
    cfg.runner.timeout_sec = 2
    return plan


def result_writer_script(path, result_dict, *, wait_for_signal=False, code=0):
    import json
    return f'''
import json
from pathlib import Path
import signal
import sys
import time
path = Path({str(path)!r})
data = json.loads({json.dumps(result_dict)!r})
def finish(signum=None, frame=None):
    path.write_text(json.dumps(data))
    sys.exit({code})
signal.signal(signal.SIGINT, finish)
signal.signal(signal.SIGTERM, finish)
{('while True: time.sleep(.01)' if wait_for_signal else 'finish()')}
'''


def test_duration_accepts_valid_nonzero_result_and_keeps_execution_state(cfg, monkeypatch, result_dict):
    import json
    from vllm_bench_eval.runner import run_benchmark
    path = cfg.path(cfg.benchmark.result_dir) / "stub.json"
    plan = stub_command(cfg, monkeypatch, result_writer_script(path, result_dict, wait_for_signal=True, code=3))
    cfg.runner.duration_sec = .12
    outcome = run_benchmark(cfg, echo=False)
    assert outcome.returncode == 3 and outcome.stop["reason"] == "duration"
    assert outcome.result_data == result_dict and outcome.execution_ok is False
    assert outcome.cleanup_confirmed is True
    metadata = json.loads(outcome.metadata_path.read_text())
    assert metadata["result_valid"] is True and metadata["execution_ok"] is False
    assert metadata["capture"]["array_index_merge_allowed"] is False
    assert plan.result_path.read_text() == json.dumps(result_dict)


def test_existing_result_cannot_be_mistaken_for_this_execution(cfg, monkeypatch, result_dict):
    import json
    from vllm_bench_eval.runner import run_benchmark
    plan = stub_command(cfg, monkeypatch, "raise SystemExit(0)")
    plan.result_path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(result_dict)
    plan.result_path.write_text(original)
    with pytest.raises(RunnerError, match="existing result"):
        run_benchmark(cfg, echo=False)
    assert plan.result_path.read_text() == original


def test_duration_without_result_is_explicitly_unsettled(cfg, monkeypatch):
    import json
    from vllm_bench_eval.runner import run_benchmark, runner_metadata_path
    plan = stub_command(cfg, monkeypatch, "import time; time.sleep(30)")
    cfg.runner.duration_sec = .05
    with pytest.raises(RunnerError, match="cannot settle"):
        run_benchmark(cfg, echo=False)
    metadata = json.loads(runner_metadata_path(plan.result_path).read_text())
    assert metadata["stop"]["reason"] == "duration" and metadata["result_valid"] is False


def test_watchdog_or_uncontrolled_nonzero_never_accepts_valid_result(cfg, monkeypatch, result_dict):
    from vllm_bench_eval.runner import run_benchmark
    path = cfg.path(cfg.benchmark.result_dir) / "stub.json"
    script = result_writer_script(path, result_dict, code=4)
    stub_command(cfg, monkeypatch, script)
    with pytest.raises(RunnerError, match="exited with code 4") as caught:
        run_benchmark(cfg, echo=False)
    assert caught.value.metadata["result_valid"] is True
    path.unlink()
    script = result_writer_script(path, result_dict).replace("finish()", "path.write_text(json.dumps(data)); time.sleep(30)")
    stub_command(cfg, monkeypatch, script)
    cfg.runner.timeout_sec = .08
    with pytest.raises(RunnerError, match="timeout_sec") as caught:
        run_benchmark(cfg, echo=False)
    assert caught.value.metadata["result_valid"] is True


def test_output_decoder_preserves_split_multibyte_text(cfg, monkeypatch, result_dict):
    from vllm_bench_eval.runner import run_benchmark
    path = cfg.path(cfg.benchmark.result_dir) / "stub.json"
    script = result_writer_script(path, result_dict).replace("finish()", "import os\ntext = '你好\\n'.encode()\nos.write(1, text[:2]); time.sleep(.02); os.write(1, text[2:]); finish()")
    stub_command(cfg, monkeypatch, script)
    outcome = run_benchmark(cfg, echo=False)
    assert outcome.output == "你好\n"


def test_ignored_signal_descendant_with_inherited_stdout_cannot_block_runner(cfg, monkeypatch, result_dict, tmp_path):
    import json
    import os
    import signal
    import subprocess
    import time
    from vllm_bench_eval.runner import run_benchmark
    path = cfg.path(cfg.benchmark.result_dir) / "stub.json"
    child_file = tmp_path / "child.pid"
    child = f"import os, signal, time; from pathlib import Path; signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN); Path({str(child_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    script = result_writer_script(path, result_dict, wait_for_signal=True, code=3)
    script = script.replace("while True: time.sleep(.01)", f"import subprocess\nsubprocess.Popen([sys.executable, '-u', '-c', {child!r}])\nwhile True: time.sleep(.01)")
    stub_command(cfg, monkeypatch, script)
    cfg.runner.duration_sec = .15
    started = time.monotonic()
    outcome = run_benchmark(cfg, echo=False)
    assert time.monotonic() - started < 2
    child_pid = int(child_file.read_text())
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True)
    assert not status.stdout.strip() or status.stdout.strip().startswith("Z")
    assert outcome.result_data == result_dict


@pytest.mark.parametrize("extra", [["--name=x"], ["--name", "x"], ["--cidfile=x"], ["--detach=false"],
    ["-d"], ["-dit"], ["-itd"], ["--label", "vbe.run_id=other"], ["--label=vbe.run_id=other"], ["-lvbe.run_id=other"]])
def test_docker_owned_identity_cannot_be_overridden(cfg, extra):
    cfg.runner.docker_extra_args = extra
    with pytest.raises(RunnerError, match="cannot override"):
        build_command(cfg)


def test_docker_normal_extra_labels_and_attached_volume_are_preserved(cfg):
    cfg.runner.docker_extra_args = ["--label", "team=example", "-v/data:/data", "-eDEBUG=1"]
    plan = build_command(cfg)
    assert all(arg in plan.argv for arg in cfg.runner.docker_extra_args)
    assert any(part.startswith("vbe.run_id=") for part in plan.argv)


@pytest.mark.parametrize("failure", [None, "timeout", "daemon", "foreign", "cleanup"])
def test_docker_control_is_bounded_and_never_mutates_unverified_identity(cfg, monkeypatch, result_dict, failure):
    import json
    import subprocess
    import time
    from vllm_bench_eval import runner as runner_mod
    path = cfg.path(cfg.benchmark.result_dir) / "stub.json"
    plan = stub_command(cfg, monkeypatch, result_writer_script(path, result_dict, wait_for_signal=True), container="owned-name")
    cfg.runner.duration_sec = .12
    calls = []
    removed = False
    def docker_run(argv, **kwargs):
        nonlocal removed
        assert argv[0] == "docker"
        assert kwargs["timeout"] == cfg.runner.cleanup_timeout_sec
        calls.append(argv[1:])
        if failure == "timeout" or (failure == "cleanup" and argv[1] != "inspect"):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if failure == "daemon":
            return subprocess.CompletedProcess(argv, 1, "", "daemon unavailable")
        if argv[1] == "inspect":
            if removed:
                return subprocess.CompletedProcess(argv, 1, "", "No such container")
            item = {"Id": "immutable-id", "Name": "/owned-name", "Config": {"Labels": {"vbe.run_id": "foreign" if failure == "foreign" else plan.run_id}}, "State": {"Running": True}}
            return subprocess.CompletedProcess(argv, 0, json.dumps([item]), "")
        if argv[1] == "rm":
            removed = True
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(runner_mod.subprocess, "run", docker_run)
    started = time.monotonic()
    outcome = runner_mod.run_benchmark(cfg, echo=False)
    assert time.monotonic() - started < 2
    assert outcome.result_data == result_dict
    assert outcome.cleanup_confirmed is (failure is None)
    metadata = json.loads(outcome.metadata_path.read_text())
    assert metadata["capture"]["read_allowed"] is (failure is None)
    mutations = [call for call in calls if call[0] != "inspect"]
    if failure in {"timeout", "daemon", "foreign"}:
        assert mutations == []
    else:
        assert mutations and all(call[-1] == "immutable-id" for call in mutations)
        assert ["kill", "--signal", "KILL", "immutable-id"] in mutations
        assert ["rm", "--force", "immutable-id"] in mutations
    if failure:
        assert metadata["cleanup"]["errors"]


def test_docker_label_file_cannot_replace_owner(cfg, tmp_path):
    labels = tmp_path / "labels"
    labels.write_text("team=example\nvbe.run_id=foreign\n")
    cfg.runner.docker_extra_args = ["--label-file", str(labels)]
    with pytest.raises(RunnerError, match="owner label"):
        build_command(cfg)
    labels.write_text("team=example\n")
    assert "--label-file" in build_command(cfg).argv


@pytest.mark.parametrize("signum", [2, 15])
def test_real_cli_signal_saves_result_and_returns_signal_status(cfg, tmp_path, result_dict, signum):
    import json
    import os
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path
    path = cfg.path(cfg.benchmark.result_dir) / "signal.json"
    ready = tmp_path / "ready"
    config_file = tmp_path / "config.yaml"
    bench = result_writer_script(path, result_dict, wait_for_signal=True)
    bench = bench.replace("while True: time.sleep(.01)", f"Path({str(ready)!r}).write_text(str(__import__('os').getpid()))\nwhile True: time.sleep(.01)")
    driver = f'''
import os, sys
from pathlib import Path
from vllm_bench_eval import runner as r
from vllm_bench_eval.cli import app
plan = r.BenchCommand(argv=[sys.executable, '-u', '-c', {bench!r}], result_path=Path({str(path)!r}), env=dict(os.environ), capabilities=r.VllmCapabilities(version='stub'))
r.build_command = lambda cfg, result_filename=None: plan
app(args=['run', '--config', {str(config_file)!r}, '--no-sync', '--stop-grace-sec', '0.1'])
'''
    proc = subprocess.Popen([sys.executable, "-u", "-c", driver], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), proc.communicate(timeout=2)[0].decode()
        proc.send_signal(signum)
        output = proc.communicate(timeout=5)[0].decode()
        assert proc.returncode == 128 + signum, output
        assert json.loads(path.read_text()) == result_dict
        metadata = json.loads(Path(str(path) + ".run-meta").read_text())
        assert metadata["stop"]["signal"] == signum
        assert metadata["cleanup"]["confirmed"] is True
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_watchdog_remains_active_during_duration_grace(cfg, monkeypatch, result_dict):
    import time
    from vllm_bench_eval.runner import run_benchmark
    path = cfg.path(cfg.benchmark.result_dir) / "stub.json"
    script = result_writer_script(path, result_dict).replace("finish()", "signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN); path.write_text(json.dumps(data)); time.sleep(30)")
    stub_command(cfg, monkeypatch, script)
    cfg.runner.duration_sec = .06
    cfg.runner.timeout_sec = .12
    cfg.runner.stop_grace_sec = .35
    started = time.monotonic()
    with pytest.raises(RunnerError, match="timeout_sec") as caught:
        run_benchmark(cfg, echo=False)
    assert time.monotonic() - started < .35
    assert caught.value.metadata["stop"]["reason"] == "watchdog"
    assert caught.value.metadata["stop"]["controlled_reason"] == "duration"
    assert caught.value.metadata["result_valid"] is True


@pytest.mark.parametrize("bad", [{"itls": [1]}, {"total_output_tokens": 10**400}])
def test_unparseable_result_is_invalid_and_still_saves_metadata(cfg, monkeypatch, result_dict, bad):
    import json
    from vllm_bench_eval.runner import run_benchmark, runner_metadata_path
    result_dict.update(bad)
    path = cfg.path(cfg.benchmark.result_dir) / "stub.json"
    plan = stub_command(cfg, monkeypatch, result_writer_script(path, result_dict, wait_for_signal=True))
    cfg.runner.duration_sec = .12
    with pytest.raises(RunnerError, match="cannot settle") as caught:
        run_benchmark(cfg, echo=False)
    assert caught.value.metadata["result_valid"] is False
    metadata = json.loads(runner_metadata_path(plan.result_path).read_text())
    assert metadata["result_valid"] is False and metadata["result_error"]
