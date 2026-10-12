"""Explicit command permission and safe resume compatibility."""
import json
from pathlib import Path
import sys

import pytest

from test_coding_agent import agent_cfg, call, completion, execute
from vllm_bench_eval.coding_agent import AgentError, _tool
from vllm_bench_eval.config import AppConfig, load_config


def test_arbitrary_argv_requires_explicit_permission(agent_cfg):
    commands = [[sys.executable, '-c', 'print("first")'], [sys.executable, '-c', 'print("second")']]
    outputs = []
    for enabled in (False, True):
        agent_cfg.coding_agent.allow_any_command = enabled
        result, requests = execute(agent_cfg, [completion({'role': 'assistant', 'tool_calls': [
            call('run_command', {'argv': argv}, f'call-{index}') for index, argv in enumerate(commands)]}, 'tool_calls'),
            completion({'role': 'assistant', 'content': 'Done'})])
        outputs = [json.loads(message['content']) for message in requests[1]['messages'][-2:]]
        system = requests[0]['messages'][0]['content']
        description = requests[0]['tools'][-1]['function']['description']
        if enabled:
            assert [output['exit_code'] for output in outputs] == [0, 0]
            assert outputs[0]['output'].strip() == 'first' and outputs[1]['output'].strip() == 'second'
            assert 'Any valid command argv' in system and 'any valid argument vector' in description
            assert result['run_config']['settings']['allow_any_command']
        else:
            assert all('error' in output for output in outputs)
            assert 'Only the exact verification argv' in system and 'configured command prefix' in description


def test_master_command_switch_has_priority(agent_cfg):
    agent_cfg.coding_agent.allow_commands = False
    agent_cfg.coding_agent.allow_any_command = True
    with pytest.raises(AgentError, match='disabled'):
        _tool('run_command', {'argv': [sys.executable, '-c', 'print(1)']}, agent_cfg.root_dir / 'source', agent_cfg, 3)


@pytest.mark.parametrize('argv', [None, '', [], [''], [1], [sys.executable, None]])
def test_unrestricted_still_rejects_invalid_argv(agent_cfg, argv):
    agent_cfg.coding_agent.allow_any_command = True
    with pytest.raises(AgentError, match='argv must'):
        _tool('run_command', {'argv': argv}, agent_cfg.root_dir / 'source', agent_cfg, 3)


@pytest.mark.parametrize('new_permission', [False, True])
def test_old_resume_defaults_false_and_rejects_permission_expansion(agent_cfg, new_permission):
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    path = Path(source['events_path']).with_suffix('').with_suffix('.json')
    source['run_config']['settings'].pop('allow_any_command')
    path.write_text(json.dumps(source))
    agent_cfg.coding_agent.resume_result_path = str(path)
    agent_cfg.coding_agent.allow_any_command = new_permission
    if new_permission:
        before = Path(source['events_path']).read_bytes()
        with pytest.raises(AgentError, match='configuration differs'):
            execute(agent_cfg, [])
        assert Path(source['events_path']).read_bytes() == before
    else:
        resumed, _ = execute(agent_cfg, [completion({'role': 'assistant', 'content': 'Done'})])
        assert resumed['tasks'][0]['status'] == 'verification_failed'


def test_environment_override_and_default(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('{}')
    assert not AppConfig().coding_agent.allow_any_command
    assert load_config(path, env={'VBE_CODING_AGENT__ALLOW_ANY_COMMAND': 'true'}).coding_agent.allow_any_command


def test_new_resume_rejects_permission_reduction(agent_cfg):
    agent_cfg.coding_agent.allow_any_command = True
    source, _ = execute(agent_cfg, [KeyboardInterrupt()])
    agent_cfg.coding_agent.resume_result_path = str(Path(source['events_path']).with_suffix('').with_suffix('.json'))
    agent_cfg.coding_agent.allow_any_command = False
    with pytest.raises(AgentError, match='configuration differs'):
        execute(agent_cfg, [])
