"""Replay CPU-to-provider frame relocation through the real downstream consumer."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_content_agents_driver import (
    _reference_frames, TaskEvaluationSceneConfigurationContentAgentsError,
)
from blueprint_pipeline.task_evaluation_scene_configuration_disclosure import MATERIALIZED_STATUS
from blueprint_pipeline.task_evaluation_scene_configuration_render_handoff import (
    materialize_capsule_render_handoff, materialize_provider_render_handoff,
    validate_provider_render_handoff, TaskEvaluationSceneConfigurationRenderHandoffError,
)


def seal(value):
    value['result_digest'] = canonical_digest(value, digest_field='result_digest')
    return value


def inputs(tmp_path):
    cpu = tmp_path / 'cpu.png'
    cpu.write_bytes(b'exact retained camera frame')
    gpu = tmp_path / 'gpu.png'
    gpu.write_bytes(cpu.read_bytes())
    value = seal({'status': MATERIALIZED_STATUS, 'derived_frame_count': 1,
                  'derived_frames': [{'camera_id': 'source-01', 'path': str(cpu),
                                      'digest': 'sha256:' + hashlib.sha256(cpu.read_bytes()).hexdigest(),
                                      'size_bytes': cpu.stat().st_size}],
                  'render_completed_on_provider': False, 'result_digest': ''})
    value['control_plane_result_digest'] = value['result_digest']
    prepared = seal(value)
    current = deepcopy(prepared)
    current['derived_frames'][0]['path'] = str(gpu)
    return prepared, seal(current)


def test_relocated_capsule_handoff_reaches_actual_content_agents_consumer(tmp_path):
    prepared, current = inputs(tmp_path)
    assert prepared['result_digest'] != current['result_digest']
    stage_input = {'construction_envelope': {'render_inputs_result': current}}
    (tmp_path / 'old').mkdir()
    (tmp_path / 'current').mkdir()
    old = materialize_provider_render_handoff(render_inputs=prepared, output_root=tmp_path / 'old')
    with pytest.raises(TaskEvaluationSceneConfigurationContentAgentsError,
                       match='handoff_render_result_digest'):
        _reference_frames(stage_input, [{'output_artifacts': [old]}])
    record = materialize_capsule_render_handoff(
        prepared_render_inputs=prepared, current_render_inputs=current,
        output_root=tmp_path / 'current')
    manifest, frames = validate_provider_render_handoff(record['path'])
    assert manifest['control_plane_render_result_digest'] == prepared['control_plane_result_digest']
    assert manifest['source_render_result_digest'] == current['result_digest']
    assert _reference_frames(stage_input, [{'output_artifacts': [record]}]) == list(frames)
    assert frames[0].read_bytes() == Path(current['derived_frames'][0]['path']).read_bytes()


@pytest.mark.parametrize('mutation', ['origin', 'camera', 'bytes', 'duplicate_camera'])
def test_relocation_cannot_admit_changed_scientific_inputs(tmp_path, mutation):
    prepared, current = inputs(tmp_path)
    if mutation == 'origin':
        current['control_plane_result_digest'] = 'sha256:' + '0' * 64
    elif mutation == 'camera':
        current['derived_frames'][0]['camera_id'] = 'different-camera'
    elif mutation == 'bytes':
        current['derived_frames'][0]['digest'] = 'sha256:' + '0' * 64
    else:
        current['derived_frames'].append(deepcopy(current['derived_frames'][0]))
        current['derived_frame_count'] = 2
    seal(current)
    with pytest.raises(TaskEvaluationSceneConfigurationRenderHandoffError,
                       match='capsule_render_'):
        materialize_capsule_render_handoff(prepared_render_inputs=prepared,
            current_render_inputs=current, output_root=tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()


def test_deferred_provider_render_keeps_completed_capsule_frames(tmp_path):
    prepared, _ = inputs(tmp_path)
    pending = {'status': 'pending_provider_render',
               'result_digest': prepared['control_plane_result_digest']}
    (tmp_path / 'deferred').mkdir()
    record = materialize_capsule_render_handoff(prepared_render_inputs=prepared,
        current_render_inputs=pending, output_root=tmp_path / 'deferred')
    manifest, _ = validate_provider_render_handoff(record['path'])
    assert manifest['source_render_result_digest'] == prepared['result_digest']
