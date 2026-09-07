import hashlib
import json

import pytest

import blueprint_pipeline.task_evaluation_public_scene_machinery_refresh as subject
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_controls_autoprovision import payload_digest
from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import record


def _seal(path, value, field):
    value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return value[field]


def _setup(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    asset = runtime / 'asset'
    asset.write_bytes(b'observed asset')
    ref = {'path': str(asset), 'digest': 'sha256:' + hashlib.sha256(asset.read_bytes()).hexdigest()}
    catalog = tmp_path / 'catalog.json'
    value = {'schema_version': 'task_evaluation_controls_robot_content_catalog.v1', 'bindings': {
        'franka': {'robot_asset_usd': ref, 'embodiment_camera_template': ref,
                   'runtime_source_payload_dir': str(runtime), 'runtime_digest': payload_digest(runtime),
                   'phase_hard_cap_usd': 2}}}
    _seal(catalog, value, 'catalog_digest')
    machinery = tmp_path / 'machinery.json'
    digest = _seal(machinery, {'schema_version': 'task_evaluation_public_scene_machinery.v1',
        'robot_catalog': record(catalog), 'source': 'preserved', 'maximum_preparation_spend_usd': 4.5}, 'machinery_digest')
    old = machinery.read_bytes()
    value['bindings']['franka']['phase_hard_cap_usd'] = 1.75
    _seal(catalog, value, 'catalog_digest')
    return dict(machinery_path=machinery, catalog_path=catalog, expected_machinery_digest=digest,
                source_commit='a' * 40), old


def test_preview_then_refresh_preserves_exact_history_and_only_catalog(tmp_path):
    args, old = _setup(tmp_path)
    assert subject.refresh(**args)['status'] == 'refresh_required'
    assert args['machinery_path'].read_bytes() == old
    result = subject.refresh(**args, apply=True)
    assert result['status'] == 'refreshed'
    assert result['provider_mutation_performed'] is False
    from pathlib import Path
    assert Path(result['archive']['path']).read_bytes() == old
    before, after = json.loads(old), json.loads(args['machinery_path'].read_bytes())
    assert after.pop('robot_catalog') == record(args['catalog_path'])
    before.pop('robot_catalog')
    before.pop('machinery_digest')
    after.pop('machinery_digest')
    assert before == after
    args['expected_machinery_digest'] = result['new_machinery_digest']
    assert subject.refresh(**args, apply=True)['status'] == 'current'


@pytest.mark.parametrize('failure', ['expected_digest', 'catalog_seal', 'runtime_asset', 'catalog_path'])
def test_bad_reference_cannot_change_machinery(tmp_path, failure):
    args, old = _setup(tmp_path)
    if failure == 'expected_digest':
        args['expected_machinery_digest'] = 'sha256:' + '0' * 64
    elif failure == 'catalog_seal':
        args['catalog_path'].write_text('{}')
    elif failure == 'runtime_asset':
        (tmp_path / 'runtime' / 'asset').write_bytes(b'changed')
    else:
        replacement = tmp_path / 'other.json'
        replacement.write_bytes(args['catalog_path'].read_bytes())
        args['catalog_path'] = replacement
    with pytest.raises(ValueError):
        subject.refresh(**args, apply=True)
    assert args['machinery_path'].read_bytes() == old
