"""No-spend profile admission catches identity, rights and dependency drift."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import adp_gaussian_excision_vast as excision
from blueprint_pipeline import task_evaluation_sam31_preparation_profile as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_sam31_plan import (
    build_sam31_preparation_plan,
    validate_sam31_preparation_plan,
)
from blueprint_pipeline.task_evaluation_supervisor.openai_cost_authority import (
    derive_operator_scope_attestation,
)
from tests.test_sam31_source_track_provider import _request


def _write(path: Path, value: dict, field: str | None = None) -> Path:
    if field:
        value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo, data, flash, wheels = [tmp_path / n for n in ('repo', 'data', 'flash', 'wheels')]
    for p in (repo, data, flash, wheels):
        p.mkdir()
    runtime = data / 'runtime'
    runtime.mkdir()
    for args in [('init', '-q'), ('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                                 'commit', '-qm', 'fixture', '--allow-empty')]:
        subprocess.run(['/usr/bin/git', '-C', str(repo), *args], check=True, capture_output=True)
    commit = subprocess.check_output(['/usr/bin/git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    monkeypatch.setattr(excision, '_git', lambda path, *args: (
        '' if args == ('status', '--short') else excision.SOURCE_TREE
        if args == ('rev-parse', 'HEAD^{tree}') else excision.SOURCE_COMMIT
        if path == flash else excision.EXPECTED_SUBMODULES[str(path.relative_to(flash))]))
    for p in excision.EXPECTED_SUBMODULES:
        (flash / p).mkdir(parents=True)
    rows = []
    for name, version in sorted(excision.DEPENDENCY_REQUIREMENTS.items()):
        p = wheels / f'{name}-{version}-py3-none-any.whl'
        p.write_bytes(b'hermetic dependency ' + name.encode())
        rows.append({'filename': p.name, 'size_bytes': p.stat().st_size,
                     'sha256': module.sha(p)})
    manifest = _write(tmp_path / 'dependencies.json', {
        'schema_version': excision.DEPENDENCY_WHEELHOUSE_SCHEMA, 'status': 'ready',
        'container_image': excision.DEFAULT_IMAGE, 'python_version': excision.DEPENDENCY_PYTHON_VERSION,
        'requirements': [{'distribution': n, 'version': v}
                         for n, v in sorted(excision.DEPENDENCY_REQUIREMENTS.items())],
        'provider_network_install_required': False, 'sdists_allowed': False, 'wheels': rows,
    }, 'manifest_digest')
    provider = _write(tmp_path / 'provider.json', _request()['provider_profile'])
    rights = _write(tmp_path / 'rights.json', {
        'schema_version': module.AI_RIGHTS_SCHEMA_VERSION,
        'status': 'accepted_for_private_derived_visual_review',
        'declared_use_scope': module.AI_REVIEW_DECLARED_USE,
        'provider_id': 'openai', 'runtime': 'openai_agents_sdk', 'model': module.AI_REVIEW_MODEL,
        'max_inference_spend_usd': module.AI_REVIEW_MAX_COST_USD,
        'derived_overlay_pngs_only': True, 'raw_source_splat_or_dataset_bytes_included': False,
        'frame_publication_authorized': False, 'frame_redistribution_authorized': False,
        'issued_by_agent': False, 'agent_accepted_terms': False,
    }, 'attestation_digest')
    now = datetime.now(timezone.utc)
    cost = _write(tmp_path / 'cost.json', derive_operator_scope_attestation(
        provider_id='openai', paid_resource_class=module.OPENAI_REVIEW_RESOURCE_CLASS,
        project_id='project-test', api_key_id='key-test', operator_id='test-operator',
        exclusive_from=now, exclusive_until=now + timedelta(hours=1)))
    hf, admin, ffmpeg = [tmp_path / n for n in ('hf.secret', 'admin.secret', 'ffmpeg')]
    for p in (hf, admin):
        p.write_text('secret-must-never-be-read')
        p.chmod(0o600)
    ffmpeg.write_text('hermetic executable placeholder')
    ffmpeg.chmod(0o755)
    return dict(source_commit=commit, repo_root=repo, server_data_root=data, runtime_root=runtime,
                sam31_provider_profile_path=provider, sam31_review_rights_attestation_path=rights,
                sam31_review_cost_scope_attestation_path=cost, sam31_hf_token_file=hf,
                openai_admin_api_key_file=admin, openai_project_id='project-test',
                openai_api_key_id='key-test', flashsplat_root=flash,
                dependency_wheelhouse_path=wheels, dependency_manifest_path=manifest,
                approved_roots=[tmp_path], ffmpeg_executable=ffmpeg)


def test_materializer_binds_evidence_without_reading_secrets(inputs, monkeypatch, tmp_path):
    secrets = {inputs['sam31_hf_token_file'], inputs['openai_admin_api_key_file']}
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path not in secrets, 'materializer tried to read a secret value'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    result = module.materialize_sam31_preparation_profile(**inputs)
    assert result['profile_digest'] == canonical_digest(result, digest_field='profile_digest')
    assert result['provider_mutations_performed'] == 0
    assert result['authority_boundary']['profile_grants_paid_execution_authority'] is False
    assert result['authority_boundary']['paid_stage_must_reopen_task_request'] is True
    assert 'secret-must-never-be-read' not in json.dumps(result)
    for name, cap, ttl in [('sam31_tracking', 1., 1800), ('contribution_sweep', 1.5, 3600)]:
        row = result['paid_stages'][name]
        assert (row['max_spend_usd'], row['hard_ttl_seconds'], row['retry_cap']) == (cap, ttl, 0)
        assert row['allowed_active_instance_ids'] == []
    profile = _write(tmp_path / 'profile.json', result)
    host = {n: inputs['sam31_review_rights_attestation_path'] for n in
            ('task_request', 'installation_receipt', 'publisher_intake',
             'source_preparation_receipt', 'interiorgs_terms')}
    from tests.test_sam31_camera_geometry import geometry_fixture

    geometry = geometry_fixture(tmp_path / "camera-geometry")
    lower, upper = geometry.pop("source_min"), geometry.pop("source_max")
    plan = build_sam31_preparation_plan(
        source_commit=inputs['source_commit'], task={'task_identity': {'id': 'task'},
            'scene_identity': {'id': 'scene'}, 'publisher_scene_id': '841757'},
        host_inputs=host, source_min=lower, source_max=upper,
        server_profile_path=profile, camera_geometry=geometry)
    assert validate_sam31_preparation_plan(plan, source_commit=inputs['source_commit'],
                                         approved_roots=(tmp_path,)) == plan
    assert plan['human_review_required'] is False
    assert plan['review_model'] == 'gpt-5.6-terra'
    tampered = json.loads(json.dumps(plan))
    tampered["camera_policy"]["views"][0]["position_offset_m"][0] += 0.1
    tampered["plan_digest"] = canonical_digest(tampered, digest_field="plan_digest")
    with pytest.raises(ValueError, match="camera_geometry_policy_mismatch"):
        validate_sam31_preparation_plan(tampered, source_commit=inputs['source_commit'],
                                       approved_roots=(tmp_path,))
    escaped = json.loads(json.dumps(plan))
    escaped["camera_policy"]["geometry_screen"]["source_files"]["labels"]["path"] = "/unapproved/labels.json"
    escaped["plan_digest"] = canonical_digest(escaped, digest_field="plan_digest")
    with pytest.raises(ValueError, match="camera_geometry_reference_outside_roots"):
        validate_sam31_preparation_plan(escaped, source_commit=inputs['source_commit'],
                                       approved_roots=(tmp_path,))


@pytest.mark.parametrize('defect,code', [
    ('wrong_commit', 'source_commit_mismatch'), ('dirty', 'source_checkout_dirty'),
    ('wheel', 'dependency_wheelhouse_invalid'), ('manifest', 'dependency_wheelhouse_invalid'),
    ('provider', 'sam31_provider_profile_invalid'), ('rights', 'review_rights_attestation_invalid'),
    ('cost', 'review_cost_scope_attestation_invalid'), ('hf_mode', 'sam31_hf_token_file_invalid'),
    ('admin_mode', 'openai_admin_api_key_file_invalid'), ('symlink', 'sam31_provider_profile_invalid'),
])
def test_profile_reopens_and_rejects_drift(inputs, defect, code, tmp_path):
    module.materialize_sam31_preparation_profile(**inputs)
    if defect == 'wrong_commit':
        inputs['source_commit'] = 'f' * 40
    elif defect == 'dirty':
        (inputs['repo_root'] / 'untracked').write_text('dirty')
    elif defect == 'wheel':
        next(inputs['dependency_wheelhouse_path'].glob('*.whl')).write_bytes(b'changed')
    elif defect == 'manifest':
        inputs['dependency_manifest_path'].write_text('{}')
    elif defect in ('provider', 'rights', 'cost'):
        key = {'provider': 'sam31_provider_profile_path', 'rights': 'sam31_review_rights_attestation_path',
               'cost': 'sam31_review_cost_scope_attestation_path'}[defect]
        v = json.loads(inputs[key].read_text())
        v['status'] = 'unauthorized-drift'
        _write(inputs[key], v)
    elif defect in ('hf_mode', 'admin_mode'):
        inputs['sam31_hf_token_file' if defect == 'hf_mode' else 'openai_admin_api_key_file'].chmod(0o666)
    else:
        p = tmp_path / 'linked.json'
        p.symlink_to(inputs['sam31_provider_profile_path'])
        inputs['sam31_provider_profile_path'] = p
    with pytest.raises(module.Sam31PreparationProfileError, match=code):
        module.materialize_sam31_preparation_profile(**inputs)


@pytest.mark.parametrize('field,value', [('issued_by_agent', True), ('agent_accepted_terms', True),
    ('raw_source_splat_or_dataset_bytes_included', True), ('frame_redistribution_authorized', True),
    ('model', 'gpt-5.6-luna'), ('max_inference_spend_usd', 2.)])
def test_resigned_rights_cannot_expand_scope(inputs, field, value):
    p = inputs['sam31_review_rights_attestation_path']
    v = json.loads(p.read_text())
    v[field] = value
    _write(p, v, 'attestation_digest')
    with pytest.raises(module.Sam31PreparationProfileError, match='review_rights_attestation_invalid'):
        module.materialize_sam31_preparation_profile(**inputs)


def test_resigned_cost_scope_cannot_reuse_another_stage(inputs):
    p = inputs['sam31_review_cost_scope_attestation_path']
    v = json.loads(p.read_text())
    v['paid_resource_class'] = 'another-stage'
    _write(p, v, 'scope_attestation_digest')
    with pytest.raises(module.Sam31PreparationProfileError, match='review_cost_scope_attestation_invalid'):
        module.materialize_sam31_preparation_profile(**inputs)


def test_cli_exclusive_write_keeps_existing_profile(inputs, tmp_path, capsys):
    options = {'sam31_provider_profile_path': 'sam31-provider-profile',
               'sam31_review_rights_attestation_path': 'sam31-review-rights-attestation',
               'sam31_review_cost_scope_attestation_path': 'sam31-review-cost-scope-attestation',
               'dependency_wheelhouse_path': 'dependency-wheelhouse',
               'dependency_manifest_path': 'dependency-manifest'}
    argv = []
    for key, value in inputs.items():
        if key == 'approved_roots':
            argv.extend(['--approved-root', str(value[0])])
        else:
            argv.extend(['--' + options.get(key, key.replace('_', '-')), str(value)])
    p = tmp_path / 'profile.json'
    argv.extend(['--output', str(p)])
    assert module.main(argv) == 0
    original = p.read_bytes()
    assert json.loads(capsys.readouterr().out)['status'] == 'materialized_no_spend'
    assert module.main(argv) == 2
    assert p.read_bytes() == original
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'


def test_released_source_or_submodule_drift_is_rejected(inputs, monkeypatch):
    original = excision._git

    def changed(path, *args):
        if path != inputs['flashsplat_root'] and args == ('rev-parse', 'HEAD'):
            return '0' * 40
        return original(path, *args)

    monkeypatch.setattr(excision, '_git', changed)
    with pytest.raises(module.Sam31PreparationProfileError, match='flashsplat_identity_invalid'):
        module.materialize_sam31_preparation_profile(**inputs)
