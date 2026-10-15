"""Small reproducible HumanEval workloads and required tool coverage."""
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from test_agent_platform import StrictPlatform
from test_coding_agent import agent_cfg, call, completion, execute
from vllm_bench_eval.agent_samples import prepare_agent_dataset
from vllm_bench_eval.coding_agent import AgentError, load_agent_result, load_tasks, run_agent, sync_agent
from vllm_bench_eval.config import AppConfig
from vllm_bench_eval.cli import app


@pytest.fixture
def humaneval_source(tmp_path):
    rows = [{"task_id": f"HumanEval/{index}", "prompt": f"def identity_{index}(value):\n    \"\"\"Return value unchanged.\"\"\"\n",
             "entry_point": f"identity_{index}", "test": "def check(candidate):\n    assert candidate(4) == 4\n    assert candidate('a') == 'a'\n",
             "canonical_solution": "    return value  # CANONICAL_MARKER\n"} for index in range(3)]
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
    return path


def test_prepare_two_tasks_original_tests_provenance_no_answer(agent_cfg, humaneval_source):
    path = prepare_agent_dataset(agent_cfg, source_file=humaneval_source)
    tasks = load_tasks(path)
    assert [task.task_id for task in tasks] == ["HumanEval/0", "HumanEval/1"]
    for task in tasks:
        assert task.required_tools == ["read_file", "write_file", "run_command"]
        assert task.verify_command == ["python3", "verify.py"]
        assert task.provenance["source_sha256"] and task.provenance["sample_sha256"]
        repository = Path(task.repository)
        assert "def check(candidate):" in (repository / "verify.py").read_text()
        assert "check(identity_" in (repository / "verify.py").read_text()
        assert "CANONICAL_MARKER" not in (repository / "solution.py").read_text()
        assert "canonical_solution" not in path.read_text()
        assert (repository / "LICENSE.humaneval").is_file()
    assert (path.parent / "provenance.json").is_file()
    with pytest.raises(AgentError, match="already exists"):
        prepare_agent_dataset(agent_cfg, source_file=humaneval_source)


def test_seed_reproducibility_and_source_validation(agent_cfg, humaneval_source):
    first = prepare_agent_dataset(agent_cfg, source_file=humaneval_source, output_dir=agent_cfg.root_dir / "a", seed=7)
    second = prepare_agent_dataset(agent_cfg, source_file=humaneval_source, output_dir=agent_cfg.root_dir / "b", seed=7)
    assert [task.task_id for task in load_tasks(first)] == [task.task_id for task in load_tasks(second)]
    with pytest.raises(AgentError, match="exceeds"):
        prepare_agent_dataset(agent_cfg, source_file=humaneval_source, output_dir=agent_cfg.root_dir / "c", num_samples=4)
    assert not (agent_cfg.root_dir / "c").exists()


def test_default_tool_audit_and_exact_verifier_command(agent_cfg):
    defaults = AppConfig().coding_agent
    assert defaults.allow_commands and defaults.required_tools == ["read_file", "write_file", "run_command"]
    agent_cfg.coding_agent.required_tools = defaults.required_tools
    task = json.loads((agent_cfg.root_dir / "tasks.jsonl").read_text())
    result, requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [
        call("read_file", {"path": "value.txt"}, "r"),
        call("write_file", {"path": "value.txt", "content": "new"}, "w"),
        call("run_command", {"argv": task["verify_command"]}, "c")]}, "tool_calls"),
        completion({"role": "assistant", "content": "Done"})])
    assert result["tasks"][0]["success"]
    assert result["tasks"][0]["tool_types"] == defaults.required_tools
    assert all(name in requests[0]["messages"][0]["content"] for name in defaults.required_tools)
    assert {tool["function"]["name"] for tool in requests[0]["tools"]} == set(defaults.required_tools)
    # Added arguments do not satisfy the exact implicit verifier permission.
    result, requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("run_command", {"argv": task["verify_command"] + ["unexpected"]})]}, "tool_calls"),
        completion({"role": "assistant", "content": "Done"})])
    assert result["tasks"][0]["tool_types"] == []
    assert "error" in json.loads(requests[1]["messages"][-1]["content"])


def test_missing_required_tool_cannot_pass_and_nonzero_command_counts(agent_cfg):
    agent_cfg.coding_agent.required_tools = ["read_file", "write_file", "run_command"]
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("write_file", {"path": "value.txt", "content": "new"})]}, "tool_calls"),
                                   completion({"role": "assistant", "content": "Done"})])
    task = result["tasks"][0]
    assert task["verification"]["exit_code"] == 0 and not task["success"]
    assert task["status"] == "required_tools_missing"
    assert task["missing_required_tools"] == ["read_file", "run_command"]
    input_task = json.loads((agent_cfg.root_dir / "tasks.jsonl").read_text())
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("run_command", {"argv": input_task["verify_command"]})]}, "tool_calls"),
                                   completion({"role": "assistant", "content": "Done"})])
    assert result["tasks"][0]["tool_types"] == ["run_command"]
    assert result["tasks"][0]["tool_failures"] == 1


def test_task_success_percentage_replay_and_pending(agent_cfg):
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    passed = json.loads(json.dumps(result["tasks"][0]))
    passed["task"]["task_id"] = "passed"
    passed["success"], passed["status"] = True, "passed"
    passed["trace_id"] += "-different"
    pending = {"task": {**passed["task"], "task_id": "pending"}, "status": "pending", "success": False}
    result["tasks"] += [passed, pending]
    Path(result["_result_path"]).write_text(json.dumps(result))
    class Client(StrictPlatform):
        def create_experiment(self, **kwargs):
            self.config = kwargs["experiment_config"]
            return super().create_experiment(**kwargs)
    client = Client()
    sync_agent(result, agent_cfg, client=client)
    roots = [trace for trace in client.traces if trace["name"].startswith("coding-task:")]
    assert [next(score["value"] for score in trace["feedback_scores"] if score["name"] == "任务成功率(%)") for trace in roots] == [0, 100]
    summary = next(trace for trace in client.traces if trace["name"] == "压测汇总")
    assert next(score["value"] for score in summary["feedback_scores"] if score["name"] == "任务成功率(%)") == 50
    assert client.config["task_quality"]["terminal_tasks"] == 2
    assert client.config["task_quality"]["task_success_pct"] == 50
    assert all(not {"task_success", "工具总耗时(s)", "验证总耗时(s)"} & {score["name"] for score in trace["feedback_scores"]} for trace in client.traces)
    result["mode"] = "replay"
    for task in result["tasks"]:
        task["success"] = None
    Path(result["_result_path"]).write_text(json.dumps(result))
    client = Client()
    sync_agent(result, agent_cfg, client=client)
    assert all("任务成功率(%)" not in {score["name"] for score in trace["feedback_scores"]} for trace in client.traces)


def test_prepare_run_sync_cli_full_workload(agent_cfg, humaneval_source, monkeypatch):
    from vllm_bench_eval import cli, coding_agent
    agent_cfg.coding_agent.required_tools = ["read_file", "write_file", "run_command"]
    monkeypatch.setattr(cli, "_load", lambda config: agent_cfg)
    runner = CliRunner()
    prepared = runner.invoke(app, ["prepare-agent-dataset", "--source-file", str(humaneval_source)])
    assert prepared.exit_code == 0
    path = agent_cfg.path(agent_cfg.prepare_agent.output_dir) / "tasks.jsonl"
    tasks = load_tasks(path)
    agent_cfg.coding_agent.tasks_path = str(path)
    client_class = httpx.Client
    call_count = 0
    def handler(request):
        nonlocal call_count
        task_index = call_count // 2
        first = call_count % 2 == 0
        call_count += 1
        if not first:
            return completion({"role": "assistant", "content": "Done"})
        entry = "identity_" + tasks[task_index].task_id.rsplit("/", 1)[1]
        return completion({"role": "assistant", "tool_calls": [
            call("read_file", {"path": "solution.py"}, "r"),
            call("write_file", {"path": "solution.py", "content": f"def {entry}(value):\n    return value\n"}, "w"),
            call("run_command", {"argv": ["python3", "verify.py"]}, "c")]}, "tool_calls")
    monkeypatch.setattr(coding_agent.httpx, "Client", lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs))
    platform = StrictPlatform()
    monkeypatch.setattr(coding_agent, "build_platform_client", lambda config: platform)
    agent_cfg.benchmark_platform.enabled = True
    run = runner.invoke(app, ["agent-run"])
    assert run.exit_code == 0 and "2/2 passed" in run.stdout
    assert call_count == 4 and len(platform.references) == 2
    result_file = next(agent_cfg.path(agent_cfg.coding_agent.result_dir).glob("agent-*.json"))
    result = load_agent_result(result_file)
    assert all(set(task["tool_types"]) == {"read_file", "write_file", "run_command"} for task in result["tasks"])
    assert all(task["success"] for task in result["tasks"])


def test_old_checkpoint_keeps_original_audit_disabled_semantics(agent_cfg):
    from vllm_bench_eval.config import CodingAgentConfig
    (agent_cfg.root_dir / 'source/value.txt').write_text('new')
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    path = Path(source['events_path']).with_suffix('').with_suffix('.json')
    source['run_config']['settings'].pop('required_tools')
    for key in ('tool_types', 'effective_required_tools'):
        source['tasks'][0].pop(key, None)
    for key in ('required_tools', 'provenance'):
        source['tasks'][0]['task'].pop(key, None)
    path.write_text(json.dumps(source))
    checkpoint = Path(source['checkpoints'][0])
    state = json.loads(checkpoint.read_text())
    for key in ('tool_types', 'effective_required_tools'):
        state['result'].pop(key, None)
    for key in ('required_tools', 'provenance'):
        state['result']['task'].pop(key, None)
    checkpoint.write_text(json.dumps(state))
    agent_cfg.coding_agent.required_tools = CodingAgentConfig().required_tools
    agent_cfg.coding_agent.resume_result_path = str(path)
    resumed, _ = execute(agent_cfg, [completion({'role':'assistant', 'content':'Done'})])
    assert resumed['tasks'][0]['success']
    assert resumed['tasks'][0]['effective_required_tools'] == []


def test_public_humaneval_configs_complete_paths_and_defaults():
    from vllm_bench_eval.config import load_config
    base = Path(__file__).parents[1] / 'examples/coding-agent'
    for filename in ('vllm-humaneval.example.yaml', 'deepseek-humaneval.example.yaml'):
        cfg = load_config(base / filename, env={})
        assert cfg.coding_agent.allow_commands and len(cfg.coding_agent.required_tools) == 3
        assert cfg.prepare_agent.num_samples == 2 and cfg.prepare_agent.seed is None
        assert cfg.path(cfg.coding_agent.tasks_path) == cfg.path(cfg.prepare_agent.output_dir) / 'tasks.jsonl'
        assert cfg.server.api_key is None
