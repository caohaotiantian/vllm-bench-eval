"""vLLM version compatibility: flag detection, degradation, missing metrics.

Real flag sets (verified against the PyPI sdists):
  0.9.0  no --save-detailed / --custom-* / --ready-check-timeout-sec
  0.9.1  adds --save-detailed
  0.9.2  adds --custom-output-len, --custom-skip-chat-template
  0.10.1 adds --ready-check-timeout-sec
"""

import pytest

from vllm_bench_platform.config import load_config
from vllm_bench_platform.runner import (
    OPTIONAL_FLAG_CONSEQUENCES,
    REQUIRED_FLAGS,
    RunnerError,
    VllmCapabilities,
    build_command,
    detect_capabilities,
    parse_supported_flags,
)

CORE = set(REQUIRED_FLAGS)
V_0_9_0 = frozenset(CORE | {
    "--seed", "--percentile-metrics", "--metric-percentiles", "--disable-tqdm",
    "--tokenizer", "--request-rate", "--max-concurrency", "--ignore-eos", "--metadata",
})
V_0_9_1 = frozenset(V_0_9_0 | {"--save-detailed"})
V_0_9_2 = frozenset(V_0_9_1 | {"--custom-output-len", "--custom-skip-chat-template"})
V_0_10_1 = frozenset(V_0_9_2 | {"--ready-check-timeout-sec"})

HELP_TEXT = """usage: vllm bench serve [-h] [--backend BACKEND] [--base-url BASE_URL]
                        [--endpoint ENDPOINT] [--model MODEL]
                        [--dataset-name {sharegpt,custom}]
                        [--dataset-path DATASET_PATH]
                        [--num-prompts NUM_PROMPTS] [--save-result]
                        [--result-dir RESULT_DIR]
                        [--result-filename RESULT_FILENAME]
                        [--save-detailed]
"""


@pytest.fixture
def cfg(tmp_path, samples_file):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        f"""
server:
  base_url: http://127.0.0.1:1234
  model: demo-model
  tokenizer: Qwen/Qwen3-8B
  ready_check_timeout_sec: 120
benchmark:
  dataset_name: custom
  dataset_path: {samples_file}
  num_prompts: 3
  max_concurrency: 2
  custom_skip_chat_template: true
  result_dir: {tmp_path / "results"}
runner:
  mode: docker
  docker_image: img:tag
""",
        encoding="utf-8",
    )
    return load_config(cfg_file)


def caps_for(flags, version):
    return VllmCapabilities(flags=flags, version=version, detected=True)


# --- help parsing -----------------------------------------------------------


def test_parse_supported_flags():
    flags = parse_supported_flags(HELP_TEXT)
    assert "--save-detailed" in flags
    assert "--dataset-name" in flags
    assert "--custom-skip-chat-template" not in flags
    assert REQUIRED_FLAGS <= flags


def test_parse_supported_flags_on_empty_text():
    assert parse_supported_flags("") == frozenset()
    assert parse_supported_flags(None) == frozenset()


# --- degradation ------------------------------------------------------------


def test_old_vllm_drops_the_flag_that_crashed_the_user(cfg):
    """0.9.1 is the user's case: `unrecognized arguments: --custom-skip-chat-template`."""
    plan = build_command(cfg, result_filename="r.json", caps=caps_for(V_0_9_1, "0.9.1"))
    assert "--custom-skip-chat-template" not in plan.argv
    assert "--custom-output-len" not in plan.argv
    assert "--ready-check-timeout-sec" not in plan.argv
    # still a runnable benchmark
    assert "--save-detailed" in plan.argv
    for flag in REQUIRED_FLAGS:
        assert flag in plan.argv


def test_each_dropped_flag_is_warned_with_its_consequence(cfg):
    plan = build_command(cfg, result_filename="r.json", caps=caps_for(V_0_9_0, "0.9.0"))
    dropped = {"--save-detailed", "--custom-output-len",
               "--custom-skip-chat-template", "--ready-check-timeout-sec"}
    warned = {w.split()[0] for w in plan.warnings}
    assert dropped <= warned
    joined = " ".join(plan.warnings)
    assert OPTIONAL_FLAG_CONSEQUENCES["--custom-skip-chat-template"] in joined
    assert OPTIONAL_FLAG_CONSEQUENCES["--save-detailed"] in joined


@pytest.mark.parametrize(
    "flags,version,expected_present,expected_absent",
    [
        (V_0_9_0, "0.9.0", [], ["--save-detailed", "--custom-skip-chat-template",
                                "--custom-output-len", "--ready-check-timeout-sec"]),
        (V_0_9_1, "0.9.1", ["--save-detailed"], ["--custom-skip-chat-template",
                                                 "--ready-check-timeout-sec"]),
        (V_0_9_2, "0.9.2", ["--save-detailed", "--custom-skip-chat-template",
                            "--custom-output-len"], ["--ready-check-timeout-sec"]),
        (V_0_10_1, "0.10.1", ["--save-detailed", "--custom-skip-chat-template",
                              "--ready-check-timeout-sec"], []),
    ],
)
def test_flag_matrix_per_version(cfg, flags, version, expected_present, expected_absent):
    argv = build_command(cfg, result_filename="r.json", caps=caps_for(flags, version)).argv
    for flag in expected_present:
        assert flag in argv, f"{version} should emit {flag}"
    for flag in expected_absent:
        assert flag not in argv, f"{version} should drop {flag}"


def test_missing_required_flag_is_an_error_naming_the_version(cfg):
    broken = caps_for(frozenset(V_0_9_2 - {"--dataset-path"}), "0.8.0")
    with pytest.raises(RunnerError, match=r"0\.8\.0.*--dataset-path"):
        build_command(cfg, result_filename="r.json", caps=broken)


def test_undetected_capabilities_emit_everything(cfg):
    """A failed probe must not silently strip flags — it warns and emits all."""
    plan = build_command(cfg, result_filename="r.json", caps=VllmCapabilities())
    assert "--custom-skip-chat-template" in plan.argv
    assert "--ready-check-timeout-sec" in plan.argv
    assert any("could not detect" in w for w in plan.warnings)
    # no per-flag warnings, because nothing was dropped
    assert not any(w.startswith("--") for w in plan.warnings)


def test_probe_failure_is_reported_and_degrades_to_emit_all(cfg, monkeypatch):
    from vllm_bench_platform import runner as mod

    monkeypatch.setattr(
        mod, "detect_capabilities",
        lambda c, **kw: VllmCapabilities(detected=False, note="boom"),
    )
    plan = build_command(cfg, result_filename="r.json", detect=True)
    assert "--custom-skip-chat-template" in plan.argv
    assert any("could not detect" in w and "boom" in w for w in plan.warnings)


def test_unsupported_extra_arg_is_warned_but_still_passed(cfg):
    cfg.benchmark.extra_args = ["--top-p", "0.9"]
    plan = build_command(cfg, result_filename="r.json", caps=caps_for(V_0_9_1, "0.9.1"))
    assert "--top-p" in plan.argv          # the user asked for it explicitly
    assert any("--top-p" in w and "extra_args" in w for w in plan.warnings)


# --- probe plumbing ---------------------------------------------------------


def _fake_proc(stdout="", stderr="", returncode=0):
    class P:
        pass

    p = P()
    p.stdout, p.stderr, p.returncode = stdout, stderr, returncode
    return p


def test_detect_reads_the_json_probe(cfg, monkeypatch):
    import json as _json

    from vllm_bench_platform import runner as mod

    payload = _json.dumps({"vllm_version": "0.9.1", "flags": sorted(V_0_9_1)})
    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda *a, **k: _fake_proc(stdout=f"WARNING noise\n{payload}\n"),
    )
    caps = detect_capabilities(cfg, use_cache=False)
    assert caps.detected and caps.version == "0.9.1"
    assert "--custom-skip-chat-template" not in caps.flags


def test_detect_falls_back_to_help_scraping(cfg, monkeypatch):
    from vllm_bench_platform import runner as mod

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _fake_proc(stdout=HELP_TEXT))
    monkeypatch.setattr(mod, "_native_version", lambda c: None)
    caps = detect_capabilities(cfg, use_cache=False)
    assert caps.detected
    assert "--save-detailed" in caps.flags


def test_detect_rejects_output_that_is_not_a_help_dump(cfg, monkeypatch):
    """A failed `docker run` prints its own usage; scraping it would strip everything."""
    from vllm_bench_platform import runner as mod

    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda *a, **k: _fake_proc(
            stderr="Unable to find image 'img:tag'\nSee 'docker run --help'.\n",
            returncode=125,
        ),
    )
    caps = detect_capabilities(cfg, use_cache=False)
    assert caps.detected is False
    assert caps.note and "core flags" in caps.note or "no flags" in caps.note


def test_detect_survives_a_crashing_probe(cfg, monkeypatch):
    from vllm_bench_platform import runner as mod

    def boom(*a, **k):
        raise OSError("docker not found")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    caps = detect_capabilities(cfg, use_cache=False)
    assert caps.detected is False
    assert "docker not found" in caps.note


def test_detection_is_cached_per_process(cfg, monkeypatch):
    import json as _json

    from vllm_bench_platform import runner as mod

    mod._CAPABILITY_CACHE.clear()
    calls = []
    payload = _json.dumps({"vllm_version": "0.11.0", "flags": sorted(V_0_10_1)})
    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda *a, **k: (calls.append(1), _fake_proc(stdout=payload))[1],
    )
    detect_capabilities(cfg)
    detect_capabilities(cfg)
    assert len(calls) == 1
    mod._CAPABILITY_CACHE.clear()


# --- dataset-name is a *value* incompatibility, not just a flag --------------

# Real `--dataset-name` choices from the sdists.
DS_0_9_0 = frozenset({"random"})
DS_0_9_1 = frozenset({"sharegpt", "burstgpt", "sonnet", "random", "hf"})
DS_0_9_2 = frozenset(DS_0_9_1 | {"custom"})


def test_custom_dataset_is_rejected_before_vllm_0_9_2(cfg):
    """`custom` (benchmarking your own prompts) only exists from 0.9.2."""
    caps = VllmCapabilities(
        flags=V_0_9_1, version="0.9.1", dataset_choices=DS_0_9_1, detected=True
    )
    with pytest.raises(RunnerError, match=r"0\.9\.1.*dataset_name='custom'"):
        build_command(cfg, result_filename="r.json", caps=caps)


def test_dataset_error_lists_what_is_supported(cfg):
    caps = VllmCapabilities(
        flags=V_0_9_0, version="0.9.0", dataset_choices=DS_0_9_0, detected=True
    )
    with pytest.raises(RunnerError) as exc:
        build_command(cfg, result_filename="r.json", caps=caps)
    assert "['random']" in str(exc.value)
    assert "0.9.2" in str(exc.value)


def test_supported_dataset_passes(cfg):
    caps = VllmCapabilities(
        flags=V_0_9_2, version="0.9.2", dataset_choices=DS_0_9_2, detected=True
    )
    assert "--dataset-name" in build_command(cfg, result_filename="r.json", caps=caps).argv


def test_unknown_dataset_choices_are_not_enforced(cfg):
    """Help scraping may not expose the choices; never block on missing info."""
    caps = VllmCapabilities(flags=V_0_9_2, version="0.9.2", detected=True)
    assert caps.supports_dataset("custom") is True
    build_command(cfg, result_filename="r.json", caps=caps)


def test_backend_falls_back_to_endpoint_type(cfg):
    """`--backend` is 0.9.2+; older releases spell it `--endpoint-type`."""
    flags = frozenset((V_0_9_1 - {"--backend"}) | {"--endpoint-type"})
    caps = VllmCapabilities(
        flags=flags, version="0.9.1", dataset_choices=DS_0_9_2, detected=True
    )
    plan = build_command(cfg, result_filename="r.json", caps=caps)
    assert "--endpoint-type" in plan.argv
    assert "--backend" not in plan.argv
    i = plan.argv.index("--endpoint-type")
    assert plan.argv[i + 1] == "openai"
    assert any("--backend" in w and "--endpoint-type" in w for w in plan.warnings)


def test_parse_dataset_choices_from_help():
    from vllm_bench_platform.runner import parse_dataset_choices

    text = "  --dataset-name {sharegpt,burstgpt,sonnet,random,hf}\n                        the dataset"
    assert parse_dataset_choices(text) == DS_0_9_1
    assert parse_dataset_choices("no choices here") == frozenset()
