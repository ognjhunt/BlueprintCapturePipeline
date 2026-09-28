# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_cli.py
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_plan.py
"""The real module CLI acquires context and invokes the bounded planner."""
import json
import subprocess
import sys

import pytest

from tests.test_scene_lifecycle_plan import context_fixture


@pytest.mark.slow
def test_actual_module_cli_returns_real_history_keep_plan(tmp_path):
    context, intent_id = context_fixture(tmp_path, completed=True)
    file = tmp_path / 'context.json'
    file.write_text(json.dumps(context))
    result = subprocess.run([sys.executable, '-m', 'blueprint_pipeline.task_evaluation_scene_lifecycle_plan',
        '--intent-id', intent_id, '--context-file', str(file.resolve()), '--now', '900000'],
        capture_output=True, text=True, timeout=20)
    report = json.loads(result.stdout)
    assert result.returncode == 0 and result.stderr == ''
    assert report['finished_observation']['status'] == 'completed'
    assert report['context_acquisition']['anchor_coalesced'] is True
    assert report['context_acquisition']['metadata_only'] is True
    assert report['mutations'] == 0 and report['action'] == 'KEEP'
    assert report['cleanup_authorized'] is False


@pytest.mark.slow
@pytest.mark.parametrize('args', [[], ['--unknown', 'secret-value'], ['--now', 'not-a-number'], ['--apply']])
def test_actual_cli_parse_failures_are_fixed_json_without_argv_echo(args):
    result = subprocess.run([sys.executable, '-m', 'blueprint_pipeline.task_evaluation_scene_lifecycle_plan', *args],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and result.stderr == ''
    assert len(result.stdout.encode()) < 4096 and 'secret-value' not in result.stdout
    assert json.loads(result.stdout)['action'] == 'KEEP'


def test_context_and_planner_construct_one_budget_and_recheck_context(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    context, intent_id = context_fixture(tmp_path, completed=True)
    file = tmp_path / 'context.json'
    file.write_text(json.dumps(context))
    original, created = cli.ReferenceCollectionBudget, []
    def constructor(**kwargs):
        value = original(**kwargs)
        created.append(value)
        return value
    monkeypatch.setattr(cli, 'ReferenceCollectionBudget', constructor)
    assert cli.main(['--intent-id', intent_id, '--context-file', str(file.resolve()), '--now', '900000'],
                    monotonic=lambda: 0) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(created) == 1 and created[0].closed
    assert report['context_acquisition']['anchor_coalesced'] is True
    assert report['planner_acquired_raw_bytes'] >= file.stat().st_size
