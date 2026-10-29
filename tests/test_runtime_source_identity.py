# Covers (for impacted-test selection):
#   scripts/install_scene_retirement_runtime.py
#   src/blueprint_pipeline/runtime_source_identity.py
"""Real protected installer snapshots; local fixture provenance, no live proof."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import pytest

from blueprint_pipeline import runtime_source_identity as reader
from blueprint_pipeline.task_evaluation_release_identity import running_release_commit
from tests.test_scene_retirement_runtime_installation import (
    _sdk_wheel_fixture, fixture, tmp_path as tmp_path,
)


def _git(source, *args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(source), *args], stderr=subprocess.DEVNULL).decode().strip()


def _installed(tmp_path, monkeypatch):
    installer, source, _ = fixture(tmp_path, monkeypatch)
    wheel = _sdk_wheel_fixture(source, tmp_path / 'wheelhouse')
    _git(source, 'init', '-q')
    _git(source, 'add', 'src', 'scripts', 'deploy', 'uv.lock')
    _git(source, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'source one')
    commit = _git(source, 'rev-parse', 'HEAD')
    installer.prepare_deployment(source, source_commit=commit, wheelhouse=wheel.parent)
    monkeypatch.setattr(reader, '_OWNER', os.getuid())
    monkeypatch.setattr(reader, '_RUNTIME_ROOT', installer._RUNTIME_ROOT)
    monkeypatch.setattr(reader, '_BOOT_ROOT', installer._BOOT_ROOT)
    return installer, source, wheel, commit


def _module(root):
    return Path(root) / 'src/blueprint_pipeline/__init__.py'


def _no_external_calls(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail('Source identity must not run Git, load credentials or make a network call')
    monkeypatch.setattr(socket.socket, 'connect', refused)
    monkeypatch.setattr(subprocess, 'Popen', refused)


def test_actual_initial_installation_has_admitted_identity_without_git_or_environment(tmp_path, monkeypatch):
    installer, source, _, commit = _installed(tmp_path, monkeypatch)
    initial = json.loads((installer._BOOT_ROOT / 'installation.json').read_bytes())
    assert initial['schema'] == 'scene-retirement-runtime-install.v2'
    assert initial['source_identity']['commit'] == commit
    monkeypatch.setenv('BLUEPRINT_RUNNING_COMMIT', 'f' * 40)
    monkeypatch.setenv('EXPECTED_PRODUCTION_COMMIT', 'e' * 40)
    _no_external_calls(monkeypatch)
    proof = reader.running_source_identity(_module(installer._RUNTIME_ROOT))
    assert proof['state'] == 'proven' and proof['source_commit'] == commit
    assert proof['basis'] == 'protected_admitted_source_snapshot'
    assert str(tmp_path) not in json.dumps(proof)
    assert reader.running_source_identity(_module(source))['state'] == 'unproven'


def test_same_source_bytes_in_different_commits_get_distinct_proven_generations(tmp_path, monkeypatch):
    installer, source, wheel, first = _installed(tmp_path, monkeypatch)
    generation_one = Path(installer.prepare_deployment(source, source_commit=first, wheelhouse=wheel.parent)['runtime_root'])
    _git(source, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'source two same rows')
    second = _git(source, 'rev-parse', 'HEAD')
    generation_two = Path(installer.prepare_deployment(source, source_commit=second, wheelhouse=wheel.parent)['runtime_root'])
    assert first != second and generation_one != generation_two
    manifest_one = json.loads((installer._RUNTIME_ROOT / 'manifests' / f'source-{generation_one.name}.json').read_bytes())
    manifest_two = json.loads((installer._RUNTIME_ROOT / 'manifests' / f'source-{generation_two.name}.json').read_bytes())
    assert manifest_one['rows'] == manifest_two['rows']
    _no_external_calls(monkeypatch)
    # Neither a changed CURRENT nor the second commit relabels retained code.
    assert reader.running_source_identity(_module(generation_one))['source_commit'] == first
    assert reader.running_source_identity(_module(generation_two))['source_commit'] == second
    assert reader.running_source_identity(_module(installer._RUNTIME_ROOT))['source_commit'] == first


@pytest.mark.parametrize('fault', ['changed_source', 'changed_claim', 'extra_source'])
def test_signed_claim_cannot_be_joined_to_different_snapshot_rows(tmp_path, monkeypatch, fault):
    installer, source, _, commit = _installed(tmp_path, monkeypatch)
    protected = installer._sdk_root() / 'release-inputs' / commit
    if fault == 'changed_source':
        _module(protected).write_bytes(b'changed source\n')
    elif fault == 'extra_source':
        (_module(protected).parent / 'unexpected.py').write_bytes(b'unknown\n')
    else:
        (protected.parent / (commit + '.claim.json')).write_text('{}')
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        installer._runtime_source_release(protected, commit, None, time.monotonic() + 10, checkout=source)


@pytest.mark.parametrize('fault', ['bytes', 'symlink', 'hardlink', 'writable_parent', 'claim', 'record_duplicate',
    'record_changed', 'negative_size', 'resealed_rows', 'resealed_inventory'])
def test_optional_identity_is_unproven_for_changed_or_unsafe_evidence(tmp_path, monkeypatch, fault):
    installer, _, _, commit = _installed(tmp_path, monkeypatch)
    module = _module(installer._RUNTIME_ROOT)
    record = installer._BOOT_ROOT / 'installation.json'
    if fault == 'bytes':
        module.write_bytes(b'dirty bytes\n')
    elif fault == 'symlink':
        target = module.with_name('other.py')
        target.write_bytes(module.read_bytes())
        module.unlink()
        module.symlink_to(target)
    elif fault == 'hardlink':
        os.link(module, module.with_name('alias.py'))
    elif fault == 'writable_parent':
        module.parent.chmod(0o777)
    elif fault == 'claim':
        claim = installer._sdk_root() / 'release-inputs' / (commit + '.claim.json')
        claim.write_text('{}')
    elif fault == 'record_duplicate':
        record.write_bytes(b'{"schema":"duplicate",' + record.read_bytes()[1:])
    else:
        value = json.loads(record.read_bytes())
        if fault == 'record_changed':
            value['source_identity']['commit'] = 'e' * 40
        elif fault == 'negative_size':
            value['rows']['src/blueprint_pipeline/__init__.py']['size'] = -1
        else:
            module.write_bytes(b'foreign but internally resealed source\n')
            row = value['rows']['src/blueprint_pipeline/__init__.py']
            row.update(size=module.stat().st_size, sha256=hashlib.sha256(module.read_bytes()).hexdigest())
            if fault == 'resealed_inventory':
                identity = value['source_identity']
                entry = next(row for row in identity['signed_entries'] if row[2] == 'src/blueprint_pipeline/__init__.py')
                entry[3:] = [row['size'], row['sha256']]
                identity['signed_tree_sha256'] = hashlib.sha256(reader._encoded(identity['signed_entries'])).hexdigest()
        record.write_bytes(reader._encoded(value))
    _no_external_calls(monkeypatch)
    try:
        assert reader.running_source_identity(module) == {'schema_version': 'runtime_source_identity.v1',
            'state': 'unproven', 'reason': 'runtime_source_identity_unproven', 'source_commit': None}
    finally:
        if fault == 'writable_parent':
            module.parent.chmod(0o755)


@pytest.mark.parametrize('target', ['module', 'record', 'claim'])
def test_identity_rechecks_evidence_after_the_module_read(tmp_path, monkeypatch, target):
    installer, _, _, commit = _installed(tmp_path, monkeypatch)
    module = _module(installer._RUNTIME_ROOT)
    selected = {'module': module, 'record': installer._BOOT_ROOT / 'installation.json',
        'claim': installer._sdk_root() / 'release-inputs' / (commit + '.claim.json')}[target]
    original = reader._read
    changed = False
    def interleaved(path, maximum):
        nonlocal changed
        observed = original(path, maximum)
        if path == module and not changed:
            changed = True
            selected.write_bytes(selected.read_bytes() + b' ')
        return observed
    monkeypatch.setattr(reader, '_read', interleaved)
    _no_external_calls(monkeypatch)
    assert reader.running_source_identity(module)['state'] == 'unproven'
    assert changed


def test_legacy_and_detached_head_metadata_do_not_gain_stronger_proof(tmp_path, monkeypatch):
    installer, source, dependencies = fixture(tmp_path, monkeypatch)
    installer.prepare(source, dependencies)
    monkeypatch.setattr(reader, '_OWNER', os.getuid())
    monkeypatch.setattr(reader, '_RUNTIME_ROOT', installer._RUNTIME_ROOT)
    monkeypatch.setattr(reader, '_BOOT_ROOT', installer._BOOT_ROOT)
    (source / '.git').mkdir()
    (source / '.git/HEAD').write_text('a' * 40 + '\n')
    assert running_release_commit(_module(source)) == 'a' * 40
    _no_external_calls(monkeypatch)
    assert reader.running_source_identity(_module(installer._RUNTIME_ROOT))['state'] == 'unproven'
    assert reader.running_source_identity(_module(source))['state'] == 'unproven'


def test_initial_copy_interruption_resumes_its_admitted_identity(tmp_path, monkeypatch):
    installer, source, wheel, commit = _installed(tmp_path, monkeypatch)
    # Exercise interruption with a distinct fresh installer root, not a reset of
    # retained snapshots. The same already admitted input cache is reused.
    monkeypatch.setattr(installer, '_RUNTIME_ROOT', tmp_path / 'next-installed')
    monkeypatch.setattr(installer, '_BOOT_ROOT', tmp_path / 'next-bootstrap')
    original = installer._copy
    def interrupted(path, destination, expected, deadline):
        if destination.name == '__init__.py':
            raise OSError('fixture initial copy interrupted')
        return original(path, destination, expected, deadline)
    monkeypatch.setattr(installer, '_copy', interrupted)
    with pytest.raises(ValueError, match='scene_retirement_runtime_unproven'):
        installer.prepare_deployment(source, source_commit=commit, wheelhouse=wheel.parent)
    assert not (installer._BOOT_ROOT / 'continuous_bootstrap.py').exists()
    monkeypatch.setattr(installer, '_copy', original)
    resumed = installer.prepare_deployment(source, source_commit=commit, wheelhouse=wheel.parent)
    monkeypatch.setattr(reader, '_RUNTIME_ROOT', installer._RUNTIME_ROOT)
    monkeypatch.setattr(reader, '_BOOT_ROOT', installer._BOOT_ROOT)
    _no_external_calls(monkeypatch)
    assert reader.running_source_identity(_module(resumed['runtime_root']))['source_commit'] == commit
    assert json.loads((installer._BOOT_ROOT / 'installation.json').read_bytes())['source_identity']['commit'] == commit
