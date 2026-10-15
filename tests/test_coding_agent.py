"""Coding task execution with a local HTTP transport and temporary repositories."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest
from typer.testing import CliRunner

from vllm_bench_eval.cli import app
from vllm_bench_eval.coding_agent import AgentError, _command, _file, load_agent_result, run_agent, sync_agent
from vllm_bench_eval.config import AppConfig


@pytest.fixture
def agent_cfg(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "value.txt").write_text("old")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": "fix", "prompt": "Set value.txt to new.", "repository": str(repo),
        "verify_command": [sys.executable, "-c", "from pathlib import Path; assert Path('value.txt').read_text() == 'new'"]}) + "\n")
    return AppConfig(root_dir=tmp_path, server={"model": "mock"}, benchmark_platform={"enabled": False},
        coding_agent={"tasks_path": str(tasks), "allow_commands": True, "required_tools": [], "max_turns": 3})


def completion(message, finish="stop", tokens=5):
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish}],
                                     "usage": {"total_tokens": tokens, "prompt_tokens": 3, "completion_tokens": 2}})


def call(name, arguments, id="call-1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def execute(cfg, responses):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        response = responses[len(requests) - 1]
        if isinstance(response, BaseException):
            raise response
        return response
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        path = run_agent(cfg, client=client)
    return load_agent_result(path), requests


def test_closed_loop_copies_source_and_preserves_history(agent_cfg):
    result, requests = execute(agent_cfg, [
        completion({"role": "assistant", "content": None, "tool_calls": [call("write_file", {"path": "value.txt", "content": "new"})]}, "tool_calls"),
        completion({"role": "assistant", "content": "Fixed."})])
    task = result["tasks"][0]
    assert task["success"] and task["turns"] == 2 and task["total_tokens"] == 10
    assert requests[1]["messages"][-1]["tool_call_id"] == "call-1"
    assert Path(task["workspace"], "value.txt").read_text() == "new"
    assert Path(agent_cfg.root_dir, "source/value.txt").read_text() == "old"
    events = [json.loads(line) for line in Path(result["events_path"]).read_text().splitlines()]
    assert events[0]["event"] == "task_start" and events[-1]["event"] == "task_end"
    assert {"turn_start", "llm_start", "tool_start", "verification_start"} <= {e["event"] for e in events}


@pytest.mark.parametrize("case", ["http", "budget", "turns", "verification", "missing_usage", "length"])
def test_failures_are_persisted(agent_cfg, case):
    message = {"role": "assistant", "content": "Done"}
    responses = [completion(message)]
    if case == "http":
        responses = [httpx.Response(503)]
    elif case == "budget":
        agent_cfg.coding_agent.token_budget = 1
    elif case == "turns":
        agent_cfg.coding_agent.max_turns = 1
        responses = [completion({"role": "assistant", "tool_calls": [call("read_file", {"path": "value.txt"})]}, "tool_calls")]
    elif case == "missing_usage":
        responses = [httpx.Response(200, json={"choices": [{"message": message}]})]
    elif case == "length":
        responses = [completion(message, "length")]
    result, _ = execute(agent_cfg, responses)
    task = result["tasks"][0]
    assert not task["success"]
    assert task["status"] != "passed"
    assert json.loads(Path(result["events_path"]).read_text().splitlines()[-1])["event"] == "task_end"


def test_tool_path_escape_and_command_denial_enter_history(agent_cfg):
    result, requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [
        call("write_file", {"path": "../escape", "content": "bad"}),
        call("run_command", {"argv": [sys.executable, "-c", "print(1)"]}, "call-2")]}, "tool_calls"),
        completion({"role": "assistant", "content": "Done"})])
    assert result["tasks"][0]["tool_failures"] == 2
    assert all("error" in json.loads(message["content"]) for message in requests[1]["messages"][-2:])
    assert not Path(result["tasks"][0]["workspace"]).parent.joinpath("escape").exists()


def test_allowed_command(agent_cfg):
    agent_cfg.coding_agent.command_allowlist = [[sys.executable, "-c"]]
    result, requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("run_command", {
        "argv": [sys.executable, "-c", "from pathlib import Path; Path('value.txt').write_text('new')"]})]}, "tool_calls"),
        completion({"role": "assistant", "content": "Done"})])
    assert result["tasks"][0]["success"]
    assert json.loads(requests[1]["messages"][-1]["content"])["exit_code"] == 0


def test_reject_symlink_and_recursive_copy(agent_cfg):
    Path(agent_cfg.root_dir, "source/link").symlink_to(agent_cfg.root_dir / "tasks.jsonl")
    with pytest.raises(AgentError, match="symbolic link"):
        run_agent(agent_cfg)
    Path(agent_cfg.root_dir, "source/link").unlink()
    agent_cfg.coding_agent.workspace_dir = "source/workspaces"
    with pytest.raises(AgentError, match="outside"):
        run_agent(agent_cfg)


def test_verification_requires_explicit_command_permission(agent_cfg):
    agent_cfg.coding_agent.allow_commands = False
    with pytest.raises(AgentError, match="allow_commands"):
        run_agent(agent_cfg)


def test_command_timeout_and_truncation(tmp_path):
    output = _command([sys.executable, "-c", "print('x' * 100)"], tmp_path, 3, 10)
    assert output["truncated"] and len(output["output"]) == 10
    output = _command([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, .05, 10)
    assert output["timed_out"] and output["exit_code"] != 0


def test_file_tool_rejects_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link").symlink_to(tmp_path)
    with pytest.raises(AgentError, match="escapes"):
        _file(workspace, "link/secret")


def test_sync_hierarchy_and_stable_ids(agent_cfg):
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("write_file", {"path": "value.txt", "content": "new"})]}, "tool_calls"),
                                  completion({"role": "assistant", "content": "Done"})])
    class Client:
        def __init__(self):
            self.items, self.traces, self.spans, self.refs = [], [], [], []
        def get_or_create_dataset(self, **kwargs):
            return self
        def get_items(self):
            return self.items
        def insert(self, items):
            self.items.extend({"id": str(index), **item} for index, item in enumerate(items))
        def create_experiment(self, **kwargs):
            return SimpleNamespace(insert=lambda refs: self.refs.extend(refs))
        def trace(self, **kwargs):
            self.traces.append(kwargs)
        def span(self, **kwargs):
            self.spans.append(kwargs)
        def flush(self):
            pass
    client = Client()
    report = sync_agent(result, agent_cfg, client=client)
    assert report["tasks"] == 1 and report["spans"] == 6
    assert len(client.refs) == 1
    parents = {span["id"] for span in client.spans if span["name"] == "turn"}
    assert all(span["parent_span_id"] in parents for span in client.spans if "parent_span_id" in span)
    assert client.traces[0]["feedback_scores"][0]["value"] == 100
    sync_agent(result, agent_cfg, client=client, experiment_name="second-identity-check")
    assert len(client.items) == 1
    task_traces = [trace for trace in client.traces if trace["name"].startswith("coding-task:")]
    assert task_traces[0]["id"] == task_traces[1]["id"]
    assert len([trace for trace in client.traces if trace["name"] == "压测汇总"]) == 2


def test_cli_failed_task_exit(agent_cfg, monkeypatch, tmp_path):
    import vllm_bench_eval.coding_agent as module
    from vllm_bench_eval import cli
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"})])
    path = tmp_path / "saved.json"
    path.write_text(json.dumps(result))
    monkeypatch.setattr(cli, "_load", lambda config: agent_cfg)
    monkeypatch.setattr(module, "run_agent", lambda cfg: path)
    runner = CliRunner()
    response = runner.invoke(app, ["agent-run", "--no-sync"])
    assert response.exit_code == 1 and "0/1 passed" in response.stdout
    assert runner.invoke(app, ["agent-run", "--no-sync", "--allow-failures"]).exit_code == 0


def test_multiple_tasks_do_not_share_context(agent_cfg):
    path = agent_cfg.root_dir / "tasks.jsonl"
    task = json.loads(path.read_text())
    task["task_id"] = "second"
    path.write_text(path.read_text() + json.dumps(task) + "\n")
    result, requests = execute(agent_cfg, [completion({"role": "assistant", "content": "Done"}),
                                          completion({"role": "assistant", "content": "Done"})])
    assert len(result["tasks"]) == 2
    assert len(requests[0]["messages"]) == len(requests[1]["messages"]) == 2
    assert result["tasks"][0]["workspace"] != result["tasks"][1]["workspace"]
    assert result["tasks"][0]["trace_id"] != result["tasks"][1]["trace_id"]


def test_tool_call_limit(agent_cfg):
    agent_cfg.coding_agent.max_tool_calls = 1
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [
        call("read_file", {"path": "value.txt"}), call("read_file", {"path": "value.txt"}, "call-2")]}, "tool_calls")])
    assert result["tasks"][0]["status"] == "limit_exceeded"
    assert result["tasks"][0]["tool_calls"] == 1


def test_interruption_has_terminal_evidence(agent_cfg):
    result, _ = execute(agent_cfg, [KeyboardInterrupt()])
    task = result["tasks"][0]
    assert task["status"] == "interrupted" and not task["success"]
    events = [json.loads(line) for line in Path(result["events_path"]).read_text().splitlines()]
    terminal = [event for event in events if event["event"] == "task_end" and event["task_id"] == task["task"]["task_id"]]
    assert terminal[-1]["result"]["status"] == "interrupted"
    assert any(event["event"] == "run_stop" and event.get("task_id") is None and event["reason"] == "interrupted" and event["signal"] == 2 for event in events)


def test_task_deadline_includes_model_call(agent_cfg):
    import time
    agent_cfg.coding_agent.task_timeout_sec = .01
    def delayed(request):
        time.sleep(.02)
        return completion({"role": "assistant", "content": "Done"})
    with httpx.Client(transport=httpx.MockTransport(delayed)) as client:
        result = load_agent_result(run_agent(agent_cfg, client=client))
    assert result["tasks"][0]["status"] == "limit_exceeded"


def test_nonfinite_timeout_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AppConfig(coding_agent={"task_timeout_sec": float("inf")})


@pytest.mark.parametrize("tool", ["write_file", "run_command"])
def test_verifier_tamper_cannot_pass(agent_cfg, tool):
    repository = agent_cfg.root_dir / "source"
    (repository / "verify.py").write_text("from pathlib import Path\nassert Path('value.txt').read_text() == 'new'\n")
    tasks_path = agent_cfg.root_dir / "tasks.jsonl"
    task = json.loads(tasks_path.read_text())
    task["verify_command"] = [sys.executable, "verify.py"]
    tasks_path.write_text(json.dumps(task) + "\n")
    if tool == "write_file":
        arguments = {"path": "verify.py", "content": "# verifier removed\n"}
    else:
        agent_cfg.coding_agent.command_allowlist = [[sys.executable, "-c"]]
        arguments = {"argv": [sys.executable, "-c", "from pathlib import Path; Path('verify.py').write_text('# removed')"]}
    result, requests = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call(tool, arguments)]}, "tool_calls"),
                                          completion({"role": "assistant", "content": "Done"})])
    executed = result["tasks"][0]
    assert not executed["success"] and executed["status"] == "verification_tampered"
    assert Path(executed["workspace"], "value.txt").read_text() == "old"
    assert len(requests) == 1


def test_explicit_verification_dependency_is_protected(agent_cfg):
    tasks_path = agent_cfg.root_dir / "tasks.jsonl"
    task = json.loads(tasks_path.read_text())
    task["verification_files"] = ["value.txt"]
    tasks_path.write_text(json.dumps(task) + "\n")
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("write_file", {"path": "value.txt", "content": "new"})]}, "tool_calls")])
    assert result["tasks"][0]["status"] == "verification_tampered"


def test_rejected_output_path_does_not_modify_source(agent_cfg):
    agent_cfg.coding_agent.result_dir = "source/new-results"
    with pytest.raises(AgentError, match="result_dir"):
        run_agent(agent_cfg)
    assert not (agent_cfg.root_dir / "source/new-results").exists()


@pytest.mark.parametrize("replacement", ["delete", "symlink"])
def test_checker_delete_and_symlink_fail(agent_cfg, replacement):
    repository = agent_cfg.root_dir / "source"
    (repository / "verify.py").write_text("assert False\n")
    tasks_path = agent_cfg.root_dir / "tasks.jsonl"
    task = json.loads(tasks_path.read_text())
    task["verify_command"] = [sys.executable, "verify.py"]
    tasks_path.write_text(json.dumps(task) + "\n")
    agent_cfg.coding_agent.command_allowlist = [[sys.executable, "-c"]]
    code = "from pathlib import Path; Path('verify.py').unlink()"
    if replacement == "symlink":
        code += "; Path('verify.py').symlink_to('/dev/null')"
    result, _ = execute(agent_cfg, [completion({"role": "assistant", "tool_calls": [call("run_command", {"argv": [sys.executable, "-c", code]})]}, "tool_calls")])
    assert result["tasks"][0]["status"] == "verification_tampered"
    assert "verify.py" in result["tasks"][0]["verification_hashes"]


def test_missing_explicit_checker_fails_before_execution(agent_cfg):
    tasks_path = agent_cfg.root_dir / "tasks.jsonl"
    task = json.loads(tasks_path.read_text())
    task["verification_files"] = ["missing.py"]
    tasks_path.write_text(json.dumps(task) + "\n")
    with pytest.raises(AgentError, match="verification file does not exist"):
        run_agent(agent_cfg)
    assert not (agent_cfg.root_dir / "results").exists()
