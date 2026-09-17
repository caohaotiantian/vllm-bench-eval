import pytest

from vllm_bench_platform.config import load_config


def test_defaults_when_file_is_minimal(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("server: {model: m}\n", encoding="utf-8")
    cfg = load_config(p, env={})
    assert cfg.server.model == "m"
    assert cfg.benchmark_platform.workspace == "default"
    # portable default: use the local vLLM, not this machine's docker image
    assert cfg.runner.mode == "native"
    assert cfg.runner.docker_platform is None
    assert cfg.root_dir == tmp_path


def test_env_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("server: {model: m}\nbenchmark_platform: {project_name: a}\n", encoding="utf-8")
    cfg = load_config(
        p,
        env={
            "VBP_SERVER__MODEL": "other",
            "VBP_BENCHMARK_PLATFORM__PROJECT_NAME": "b",
            "VBP_BENCHMARK__NUM_PROMPTS": "42",
            "VBP_BENCHMARK__CUSTOM_SKIP_CHAT_TEMPLATE": "false",
            "VBP_BENCHMARK__EXTRA_ARGS": '["--temperature", "0"]',
        },
    )
    assert cfg.server.model == "other"
    assert cfg.benchmark_platform.project_name == "b"
    assert cfg.benchmark.num_prompts == 42
    assert cfg.benchmark.custom_skip_chat_template is False
    assert cfg.benchmark.extra_args == ["--temperature", "0"]


def test_platform_sdk_env_aliases(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmark_platform: {url: 'http://a/api/'}\n", encoding="utf-8")
    cfg = load_config(p, env={"OPIK_URL_OVERRIDE": "http://b/api/", "OPIK_WORKSPACE": "ws"})
    assert cfg.benchmark_platform.url == "http://b/api/"
    assert cfg.benchmark_platform.workspace == "ws"


def test_vbp_prefix_wins_over_alias(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmark_platform: {url: 'http://a/api/'}\n", encoding="utf-8")
    cfg = load_config(
        p, env={"OPIK_URL_OVERRIDE": "http://b/api/", "VBP_BENCHMARK_PLATFORM__URL": "http://c/api/"}
    )
    assert cfg.benchmark_platform.url == "http://c/api/"


def test_relative_paths_resolve_against_config_dir(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmark: {dataset_path: data/x.jsonl}\n", encoding="utf-8")
    cfg = load_config(p, env={})
    assert cfg.path(cfg.benchmark.dataset_path) == tmp_path / "data" / "x.jsonl"


def test_bad_runner_mode(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("runner: {mode: kubernetes}\n", encoding="utf-8")
    with pytest.raises(Exception, match="native"):
        load_config(p, env={})


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", env={})


# --- review fixes ----------------------------------------------------------


def test_numeric_request_rate_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmark: {request_rate: 10}\n", encoding="utf-8")
    assert load_config(p, env={}).benchmark.request_rate == "10"
    p.write_text("benchmark: {request_rate: 2.5}\n", encoding="utf-8")
    assert float(load_config(p, env={}).benchmark.request_rate) == 2.5
    p.write_text("benchmark: {request_rate: inf}\n", encoding="utf-8")
    assert load_config(p, env={}).benchmark.request_rate == "inf"


def test_bad_request_rate_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmark: {request_rate: fast}\n", encoding="utf-8")
    with pytest.raises(Exception, match="request_rate"):
        load_config(p, env={})
    p.write_text("benchmark: {request_rate: 0}\n", encoding="utf-8")
    with pytest.raises(Exception, match="request_rate"):
        load_config(p, env={})


def test_non_string_metadata_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmark: {metadata: {gpus: 8, tp: 2.0, note: null}}\n", encoding="utf-8")
    md = load_config(p, env={}).benchmark.metadata
    assert md == {"gpus": "8", "tp": "2.0", "note": ""}


def test_typo_in_a_section_field_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("server: {modle: m}\n", encoding="utf-8")
    with pytest.raises(Exception, match="modle"):
        load_config(p, env={})


def test_unknown_top_level_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("serverr: {model: m}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_config(p, env={})


def test_base_url_with_v1_suffix_is_stripped(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("server: {base_url: 'http://h:1234/v1/'}\n", encoding="utf-8")
    assert load_config(p, env={}).server.base_url == "http://h:1234"


def test_env_null_clears_an_optional_field(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("benchmark: {max_concurrency: 4}\nserver: {api_key: abc}\n", encoding="utf-8")
    cfg = load_config(
        p,
        env={
            "VBP_BENCHMARK__MAX_CONCURRENCY": "",
            "VBP_SERVER__API_KEY": "null",
            "VBP_SERVER__READY_CHECK_TIMEOUT_SEC": "none",
        },
    )
    assert cfg.benchmark.max_concurrency is None
    assert cfg.server.api_key is None
    assert cfg.server.ready_check_timeout_sec is None


def test_example_config_is_valid_and_portable():
    import pathlib as _p

    import yaml as _yaml

    root = _p.Path(__file__).resolve().parent.parent
    cfg = load_config(root / "config.example.yaml", env={})
    assert cfg.runner.mode == "native"
    assert cfg.runner.docker_platform is None
    raw = _yaml.safe_load((root / "config.example.yaml").read_text(encoding="utf-8"))
    assert set(raw) <= {"server", "benchmark", "runner", "benchmark_platform", "prepare"}
