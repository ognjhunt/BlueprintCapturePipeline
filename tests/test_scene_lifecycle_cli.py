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
    original, created = cli.ReferenceCollectionBudget._for_scene_lifecycle_plan, []
    def constructor(**kwargs):
        value = original(**kwargs)
        created.append(value)
        return value
    monkeypatch.setattr(cli.ReferenceCollectionBudget, '_for_scene_lifecycle_plan', staticmethod(constructor))
    assert cli.main(['--intent-id', intent_id, '--context-file', str(file.resolve()), '--now', '900000'],
                    monotonic=lambda: 0) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(created) == 1 and created[0].closed
    assert report['context_acquisition']['anchor_coalesced'] is True
    assert report['planner_acquired_raw_bytes'] >= file.stat().st_size


@pytest.mark.parametrize('raw', [b'{"roots":{},"roots":{}}', b'{"roots":NaN}', b'{"roots":', b'[]'])
def test_context_malformed_json_is_fixed_refusal_without_acquiring_planner_roots(tmp_path, raw, capsys):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    file = tmp_path/'context.json'
    file.write_bytes(raw)
    assert cli.main(['--intent-id', 'intent-1', '--context-file', str(file.resolve()), '--now', '1'],
                    monotonic=lambda: 0) == 2
    output = capsys.readouterr()
    assert output.err == '' and len(output.out.encode()) < 4096
    report = json.loads(output.out)
    assert report['mutations'] == 0 and report['cleanup_authorized'] is False
    assert 'historical_lineage' not in report


def test_symlink_context_never_opens_target_payload(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    from blueprint_pipeline import task_evaluation_scene_lifecycle_acquisition as acquisition
    payload = tmp_path/'payload.bin'
    payload.write_bytes(b'secret')
    file = tmp_path/'context.json'
    file.symlink_to(payload)
    original, opened = acquisition.os.read, []
    def read(fd, amount):
        opened.append(fd)
        return original(fd, amount)
    monkeypatch.setattr(acquisition.os, 'read', read)
    assert cli.main(['--intent-id', 'intent-1', '--context-file', str(file.absolute()), '--now', '1'],
                    monotonic=lambda: 0) == 2
    assert opened == [] and 'secret' not in capsys.readouterr().out


def test_context_raw_allowance_refuses_before_read_not_after_encoding(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    from blueprint_pipeline import task_evaluation_scene_lifecycle_acquisition as acquisition
    file = tmp_path/'context.json'
    file.write_text('{}')
    original = cli.ReferenceCollectionBudget._for_scene_lifecycle_plan
    def budget(**kwargs):
        shared = original(**kwargs)
        shared.charge('raw_bytes', shared.limits['raw_bytes'])
        return shared
    monkeypatch.setattr(cli.ReferenceCollectionBudget, '_for_scene_lifecycle_plan', staticmethod(budget))
    monkeypatch.setattr(acquisition.os, 'read', lambda *a: pytest.fail('read past exhausted shared allowance'))
    assert cli.main(['--intent-id', 'intent-1', '--context-file', str(file.resolve()), '--now', '1'],
                    monotonic=lambda: 0) == 2
    assert json.loads(capsys.readouterr().out)['blockers'] == ['reference_raw_bytes_limit']


def test_context_replacement_after_acquisition_keeps_only_historical_report(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    from blueprint_pipeline import task_evaluation_scene_lifecycle_plan as planner
    context, intent = context_fixture(tmp_path)
    file = tmp_path/'context.json'
    file.write_text(json.dumps(context))
    original = planner.native._join
    def replacing(*a, **kw):
        result = original(*a, **kw)
        file.write_text(json.dumps(context)+' ')
        return result
    monkeypatch.setattr(planner.native, '_join', replacing)
    assert cli.main(['--intent-id', intent, '--context-file', str(file.resolve()), '--now', '1'],
                    monotonic=lambda: 0) == 0
    output = json.loads(capsys.readouterr().out)
    assert 'metadata_changed_after_observation' in output['blockers']
    assert output['action'] == 'KEEP' and output['cleanup_authorized'] is False


def test_oversized_os_argv_refuses_before_copy(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    class NoSlice(list):
        def __getitem__(self, key):
            if isinstance(key, slice):
                pytest.fail('unbounded argv cloned before count guard')
            return super().__getitem__(key)
    monkeypatch.setattr(cli.sys, 'argv', NoSlice(['program']+['--unrecognized']*100))
    assert cli.main(monotonic=lambda: 0) == 2


def test_real_cli_released_pin_path_array_has_fixed_refusal(tmp_path, capsys):
    from blueprint_pipeline.control_plane_storage_pins import write_storage_pin, release_storage_pin
    from tests.test_scene_lifecycle_plan import context_fixture
    from blueprint_pipeline import task_evaluation_scene_lifecycle_cli as cli
    context, intent_id = context_fixture(tmp_path)
    write_storage_pin(pins_root=context['pins_root'], kind='preparation', owner_id='pin-owner',
                      paths=['/retained/secrets.json'], now=lambda: 1000, ttl_seconds=1000)
    release_storage_pin(pins_root=context['pins_root'], kind='preparation', owner_id='pin-owner', now=lambda: 1100)
    path = tmp_path/'context.json'
    path.write_text(json.dumps(context))
    assert cli.main(['--intent-id', intent_id, '--context-file', str(path), '--now', '900000'], monotonic=lambda: 0) == 2
    observed = capsys.readouterr()
    assert '/retained/secrets.json' not in observed.out and not observed.err
    result = json.loads(observed.out)
    assert 'historical_lineage' not in result and result['references_clear'] is False
    assert len(capsys.readouterr().out.encode()) < 4096
