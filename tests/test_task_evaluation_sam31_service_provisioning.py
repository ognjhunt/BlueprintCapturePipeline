"""Both SAM preparation processes receive one profile; only execution gets its key."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_service_provisioning as module
from blueprint_pipeline.task_evaluation_sam31_preparation_profile import materialize_sam31_preparation_profile
from tests.test_task_evaluation_sam31_preparation_profile import inputs as inputs


@pytest.fixture
def configuration(inputs, tmp_path):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(materialize_sam31_preparation_profile(**inputs)))
    key = tmp_path / 'inference.secret'
    key.write_text('never-read-or-record-this-key')
    key.chmod(0o600)
    return dict(profile_path=profile, expected_source_commit=inputs['source_commit'],
                openai_api_key_file=key, openai_api_key_id='key-test',
                environment_root=tmp_path/'etc', systemd_unit_root=tmp_path/'units')


def test_real_profile_binds_both_services_without_reading_secret(configuration, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path != configuration['openai_api_key_file']
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    receipt = module.provision_sam31_service_environment(**configuration, allow_live_agents_sdk=True)
    envs = [Path(row['path']) for row in receipt['environment_files']]
    intake, executor = [p.read_text() for p in envs]
    assert intake == f"{module.PROFILE_ENV}={configuration['profile_path']}\n"
    assert intake in executor
    assert 'BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=1\n' in executor
    assert f"OPENAI_API_KEY_FILE={configuration['openai_api_key_file']}\n" in executor
    assert 'OPENAI_API_KEY=\n' in executor
    for row, env in zip(receipt['drop_ins'], envs, strict=True):
        assert Path(row['path']).read_text() == f'[Service]\nEnvironmentFile={env}\n'
    assert receipt['systemd_reloaded'] is False
    assert receipt['services_started'] is False
    assert 'never-read-or-record-this-key' not in json.dumps(receipt)


def test_default_disables_live_sdk_and_successor_preserves_previous_env(configuration):
    first = module.provision_sam31_service_environment(**configuration)
    old_env = Path(first['environment_files'][1]['path'])
    old_bytes = old_env.read_bytes()
    assert b'BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS=0\n' in old_bytes
    second = module.provision_sam31_service_environment(**configuration, allow_live_agents_sdk=True)
    assert old_env.read_bytes() == old_bytes
    assert first['binding_digest'] != second['binding_digest']
    assert first['environment_files'] != second['environment_files']


def test_registry_is_bound_once_for_both_workers(configuration, tmp_path):
    from blueprint_pipeline.task_evaluation_sam31_profile_registry import REGISTRY_ENV
    receipt = module.provision_sam31_service_environment(
        **configuration, profile_registry_root=tmp_path / 'registry')
    assert Path(receipt['profile_registry']['path']).is_file()
    for row in receipt['environment_files']:
        assert f"{REGISTRY_ENV}={tmp_path / 'registry'}\n" in Path(row['path']).read_text()


@pytest.mark.parametrize('mutation, blocker', [
    ('key_id', 'inference_key_id_mismatch'),
    ('commit', 'profile_identity_invalid'),
    ('changed_profile', 'profile_identity_invalid'),
    ('secret_mode', 'inference_key_invalid'),
    ('symlink', 'path_unsafe'),
])
def test_invalid_binding_fails_before_configuration_write(configuration, mutation, blocker, tmp_path):
    if mutation == 'key_id':
        configuration['openai_api_key_id'] = 'wrong-key'
    elif mutation == 'commit':
        configuration['expected_source_commit'] = 'f'*40
    elif mutation == 'changed_profile':
        p = configuration['profile_path']
        data = json.loads(p.read_text())
        data['source_commit'] = 'f'*40
        p.write_text(json.dumps(data))
    elif mutation == 'secret_mode':
        configuration['openai_api_key_file'].chmod(0o644)
    else:
        link = tmp_path/'linked-etc'
        link.symlink_to(tmp_path, target_is_directory=True)
        configuration['environment_root'] = link
    with pytest.raises(ValueError, match=blocker):
        module.provision_sam31_service_environment(**configuration)
    assert not configuration['systemd_unit_root'].exists()


def test_reload_is_only_requested_command_after_both_drop_ins_exist(configuration, monkeypatch):
    from types import SimpleNamespace
    original = module.subprocess.run
    calls = []
    def run(command, **kwargs):
        if command[0] != 'systemctl':
            return original(command, **kwargs)
        calls.append(command)
        assert len(list(configuration['systemd_unit_root'].glob('*.service.d/*.conf'))) == 2
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    receipt = module.provision_sam31_service_environment(**configuration, reload_systemd=True)
    assert calls == [['systemctl', 'daemon-reload']]
    assert receipt['systemd_reloaded'] is True
    assert receipt['services_started'] is False


def test_changed_profile_dependency_is_reopened_before_install(configuration):
    profile = json.loads(configuration['profile_path'].read_text())
    provider = Path(profile['artifact_references']['sam31_provider_profile']['path'])
    provider.write_text(provider.read_text() + '\n')
    with pytest.raises(ValueError, match='profile_evidence_changed'):
        module.provision_sam31_service_environment(**configuration)
    assert not configuration['systemd_unit_root'].exists()


def test_cli_installs_both_bindings_and_writes_receipt(configuration, tmp_path, capsys):
    out = tmp_path/'receipt.json'
    assert module.main([
        '--profile', str(configuration['profile_path']),
        '--expected-source-commit', configuration['expected_source_commit'],
        '--openai-api-key-file', str(configuration['openai_api_key_file']),
        '--openai-api-key-id', configuration['openai_api_key_id'],
        '--environment-root', str(configuration['environment_root']),
        '--systemd-unit-root', str(configuration['systemd_unit_root']),
        '--allow-live-agents-sdk', '--receipt-out', str(out),
    ]) == 0
    assert json.loads(out.read_text()) == json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text())['allow_live_agents_sdk'] is True


def test_exact_git_trust_handles_other_owner_and_restores_environment(configuration, monkeypatch):
    import os
    from blueprint_pipeline import adp_gaussian_excision_vast as excision

    profile = json.loads(configuration['profile_path'].read_text())
    repo = Path(profile['repo_root'])
    flash = Path(profile['released_dependencies']['flashsplat_root'])
    expected = {str(repo), str(flash), *(str(flash/name) for name in excision.EXPECTED_SUBMODULES)}
    # Real Git refuses even this same-user hermetic checkout under its explicit
    # ownership-test mode unless the provisioning scope trusts the exact path.
    monkeypatch.setenv('GIT_TEST_ASSUME_DIFFERENT_OWNER', '1')
    # Hosted runner trust settings must not pre-admit this temporary checkout.
    # Disable external config only inside this ownership regression.
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    for name in ('GIT_CONFIG_PARAMETERS', 'GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('GIT_CONFIG_COUNT', '1')
    monkeypatch.setenv('GIT_CONFIG_KEY_0', 'safe.directory')
    monkeypatch.setenv('GIT_CONFIG_VALUE_0', '/unrelated-preserved-root')
    with pytest.raises(ValueError, match='source_checkout_unverified'):
        module._git(repo, 'rev-parse', 'HEAD')
    original = excision._git
    observed = []
    def checked(path, *args):
        actual = {os.environ[f'GIT_CONFIG_VALUE_{i}']
                  for i in range(1, int(os.environ['GIT_CONFIG_COUNT']))}
        assert actual == expected
        assert all(os.environ[f'GIT_CONFIG_KEY_{i}'] == 'safe.directory'
                   for i in range(1, int(os.environ['GIT_CONFIG_COUNT'])))
        observed.append(path)
        return original(path, *args)
    monkeypatch.setattr(excision, '_git', checked)
    before = dict(os.environ)
    git_config_before = (repo/'.git/config').read_bytes()
    receipt = module.provision_sam31_service_environment(**configuration)
    assert receipt['status'] == 'installed'
    assert set(observed) == {flash, *(flash/name for name in excision.EXPECTED_SUBMODULES)}
    assert dict(os.environ) == before
    assert (repo/'.git/config').read_bytes() == git_config_before


def test_git_scope_restored_when_dependency_validation_fails(configuration, monkeypatch):
    import os
    from blueprint_pipeline import adp_gaussian_excision_vast as excision

    def broken(*_args):
        raise ValueError('retained_source_invalid')
    monkeypatch.setattr(excision, '_source_identity', broken)
    before = dict(os.environ)
    with pytest.raises(ValueError, match='flashsplat_identity_invalid'):
        module.provision_sam31_service_environment(**configuration)
    assert dict(os.environ) == before
    assert not configuration['systemd_unit_root'].exists()
