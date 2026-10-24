"""Retained mesh adoption binds source, camera request, operation and output bytes."""

from copy import deepcopy
import json

import pytest

from blueprint_pipeline.atlas_result_bundle import retained_mesh_for_splat, find_retained_mesh_root
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


@pytest.fixture
def retained(tmp_path):
    splat_root = tmp_path / 'background_splat'
    mesh_root = tmp_path / 'earlier-export-name'
    splat_root.mkdir()
    (mesh_root / 'atlas_pose').mkdir(parents=True)
    (mesh_root / 'registered').mkdir()
    source = splat_root / 'source.spz'
    source.write_bytes(b'retained source splat')
    spec = {'source_video_digest': 'video', 'request': {
        'splatAsset': {'assetId': 'upload:splat'}, 'cameras': [{'camera': 'retained'}],
        'textureMode': 'image_texture'},
        'uploads': {'splat': {'path': str(source), 'sha256': _sha256_file(source)}}}
    splat = {'gaussian_count': 4, 'outputs': [], 'pose_result_digest': 'pose'}
    splat['digest'] = canonical_digest(splat, digest_field='digest')
    write_json(splat_root / 'splats2mesh_task_spec.json', spec)
    write_json(splat_root / 'atlas_depth_splat_manifest.json', splat)
    output = mesh_root / 'mesh.glb'
    output.write_bytes(b'glTF retained mesh')
    supported = mesh_root / 'registered/supported.glb'
    supported.write_bytes(b'glTF registered mesh')
    mesh = {'task': 'splats2Mesh', 'status': 'completed', 'operation': {'id': 'paid-once'},
            'binding': {'task': 'splats2Mesh', 'source_video_digest': 'video',
                        'task_context_digest': 'context', 'request_template': deepcopy(spec['request']),
                        'uploads': deepcopy(spec['uploads'])},
            'outputs': [{'path': output.name, 'bytes': output.stat().st_size,
                         'sha256': _sha256_file(output), 'role': 'mesh'}]}
    registration = {'splat_manifest_digest': splat['digest'], 'source_glb_sha256': _sha256_file(output),
                    'outputs': [{'path': supported.name, 'sha256': _sha256_file(supported)}]}

    def seal():
        mesh['request_digest'] = canonical_digest(mesh['binding'])
        mesh['digest'] = canonical_digest(mesh, digest_field='digest')
        write_json(mesh_root / 'result.json', mesh)
        write_json(mesh_root / 'atlas_pose/submission.json', {'status': 'completed',
            'operation_id': mesh['operation']['id'], 'binding': mesh['binding'],
            'request_digest': mesh['request_digest'], 'result_digest': mesh['digest']})
        registration['digest'] = canonical_digest(registration, digest_field='digest')
        write_json(mesh_root / 'registered/atlas_mesh_registration.json', registration)

    seal()
    return {'splat_root': splat_root, 'mesh_root': mesh_root, 'mesh': mesh,
            'registration': registration, 'spec': spec, 'seal': seal}


def adopt(case, **options):
    return retained_mesh_for_splat(mesh_root=case['mesh_root'], splat_root=case['splat_root'],
                                   source_video_digest='video', task_context_digest=options.get('context', 'context'))


def test_completed_mesh_keeps_original_attempt_and_receipts(retained):
    paths = list(retained['mesh_root'].rglob('*'))
    before = {str(p): p.read_bytes() for p in paths if p.is_file()}
    assert adopt(retained)['operation']['id'] == 'paid-once'
    assert before == {str(p): p.read_bytes() for p in paths if p.is_file()}
    with pytest.raises(ValueError, match='binding_changed'):
        adopt(retained, context='different task')


def test_automatic_discovery_matches_the_current_scene_and_task(retained):
    args = {'output_root': retained['mesh_root'].parent, 'splat_root': retained['splat_root'],
            'source_video_digest': 'video', 'task_context_digest': 'context'}
    assert find_retained_mesh_root(**args) == retained['mesh_root']
    assert find_retained_mesh_root(**{**args, 'task_context_digest': 'other task'}) is None
    retained['registration']['splat_manifest_digest'] = 'previous reconstruction'
    retained['seal']()
    assert find_retained_mesh_root(**args) is None


@pytest.mark.parametrize('field', ['cameras', 'textureMode', 'source_video_digest', 'task'])
def test_resealed_receipt_for_other_input_is_not_adopted(retained, field):
    binding = retained['mesh']['binding']
    if field in ('cameras', 'textureMode'):
        binding['request_template'][field] = 'different'
    else:
        binding[field] = 'different'
    retained['seal']()
    with pytest.raises(ValueError, match='invalid|changed'):
        adopt(retained)


@pytest.mark.parametrize('which', ['source', 'upload', 'raw_output', 'registered_output', 'registration'])
def test_changed_bytes_or_registration_refuse_adoption(retained, which):
    if which == 'source':
        path = retained['splat_root'] / 'source.spz'
    elif which == 'upload':
        path = retained['mesh_root'] / 'other.spz'
        path.write_bytes(b'another splat')
        retained['mesh']['binding']['uploads']['splat']['path'] = str(path)
        retained['seal']()
    elif which == 'raw_output':
        path = retained['mesh_root'] / 'mesh.glb'
    elif which == 'registered_output':
        path = retained['mesh_root'] / 'registered/supported.glb'
    else:
        retained['registration']['splat_manifest_digest'] = 'other scene'
        retained['seal']()
        path = None
    if path is not None:
        path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        adopt(retained)


def test_verified_mesh_only_reduction_is_reused(retained):
    path = retained['mesh_root'] / 'reduced/input.spz'
    path.parent.mkdir()
    path.write_bytes(b'mesh-only reduced splat')
    upload = {'path': str(path), 'sha256': _sha256_file(path)}
    retained['mesh']['binding']['uploads']['splat'] = upload
    retained['seal']()
    with pytest.raises(ValueError, match='reduction_missing'):
        adopt(retained)
    receipt = {'schema_version': 'atlas_splats2mesh_input.v1',
               'source_sha256': retained['spec']['uploads']['splat']['sha256'],
               'source_gaussian_count': 4, **upload}
    write_json(path.parent / 'mesh_input.json', receipt)
    assert adopt(retained)['operation']['id'] == 'paid-once'
    receipt['source_sha256'] = 'other source'
    write_json(path.parent / 'mesh_input.json', receipt)
    with pytest.raises(ValueError, match='reduction_changed'):
        adopt(retained)


def test_partial_or_different_operation_cannot_be_adopted(retained):
    path = retained['mesh_root'] / 'atlas_pose/submission.json'
    state = json.loads(path.read_text())
    state['operation_id'] = 'another-paid-operation'
    write_json(path, state)
    with pytest.raises(ValueError, match='binding_changed'):
        adopt(retained)
    state['status'] = 'processing'
    write_json(path, state)
    with pytest.raises(ValueError, match='receipt_invalid'):
        adopt(retained)
