"""ADP-009D/day28: actual finite root route, default-off and no root override."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
import json
import os
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_registered_experiment_retirement_flow import retirement_installation  # noqa: F401


def call(root, capsys, arguments):
    status = root.main(arguments)
    raw = capsys.readouterr().out
    assert len(raw.encode()) <= 8192
    return status, json.loads(raw)


def test_fixed_root_cli_keeps_expired_scratch_without_producer_completion(retirement_installation, monkeypatch, capsys):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    config, _, _, _ = retirement_installation
    monkeypatch.setattr(root, 'INSTALLED_CONFIG_PATH', config)
    monkeypatch.setattr(root.time, 'time', lambda: 1000)
    status, issued = call(root, capsys, ['issue-create', '--principal', 'operator', '--owner', 'owner',
        '--root', 'work', '--reference', 'run-cli', '--ttl', '1800', '--profile', 'local_root_disposable.v1'])
    assert status == 0 and issued['decision'] == 'completed'
    grant = issued['result']
    selector = ['--sha256', grant['intent']['sha256'], '--size-bytes', str(grant['intent']['size_bytes'])]
    status, born = call(root, capsys, ['create', grant['intent_id'], *selector])
    assert status == 0
    target = Path(born['result']['path'])
    (target / 'temporary.txt').write_text('tiny scratch')
    monkeypatch.setattr(root.time, 'time', lambda: 2900)
    status, action = call(root, capsys, ['issue-action', grant['intent_id'], '--principal', 'operator',
        '--owner', 'owner', '--action', 'delete', '--expires-at', '3500'])
    assert status == 2 and action == {
        'decision': 'refused', 'reason': 'experiment_producer_completion_missing'}
    assert (target / 'temporary.txt').read_text() == 'tiny scratch'


@pytest.mark.parametrize('arguments', [
    ['create', '0' * 32, '--sha256', 'sha256:' + '0' * 64, '--size-bytes', '1', '--config', '/tmp/other'],
    ['apply', '0' * 32, '--sha256', 'sha256:' + '0' * 64, '--size-bytes', '1', '--pins-root', '/tmp/empty'],
    ['run', '0' * 32, '--sha256', 'sha256:' + '0' * 64, '--size-bytes', '1', '--command', 'true'],
])
def test_fixed_cli_refuses_authority_or_executable_overrides(arguments, capsys, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    monkeypatch.setattr(os, 'geteuid', lambda: 0)
    status, result = call(root, capsys, arguments)
    assert status == 2 and result == {'decision': 'refused', 'reason': 'experiment_cli_arguments_invalid'}


def test_fixed_cli_refuses_nonroot_before_dispatch(monkeypatch, capsys):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    monkeypatch.setattr(os, 'geteuid', lambda: 1234)
    monkeypatch.setattr(root, 'issue_experiment_creation_intent', lambda **kwargs: pytest.fail('issuer entered'))
    status, result = call(root, capsys, ['issue-create', '--principal', 'operator', '--owner', 'owner',
        '--root', 'work', '--reference', 'run-cli', '--ttl', '1800', '--profile', 'local_root_disposable.v1'])
    assert status == 2 and result['reason'] == 'experiment_issuer_required'
