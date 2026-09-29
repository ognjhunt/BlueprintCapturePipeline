"""Completed construction needs its actual published revision before retirement."""

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_construction_queue import (
    finalize_scene_construction, stage_scene_construction,
)
from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance
from blueprint_pipeline.task_evaluation_scene_retirement_reference_proofs import TerminalProofs


def _raw(path: Path):
    body = path.read_bytes()
    return (str(path), 'sha256:' + hashlib.sha256(body).hexdigest(), len(body))


@pytest.mark.parametrize('drift', [None, 'publication_bytes', 'revision_bytes', 'revision_not_archived'])
def test_completed_construction_handoff_requires_selected_published_revision(tmp_path, monkeypatch, drift):
    from tests import test_task_evaluation_scene_configuration_publication as producer_test

    # Exercise the existing no-provider publication test's actual producer and
    # retain its exact request, revision, readback result and bytes.
    publisher_root = tmp_path / 'owned-launch-run'
    publisher_root.mkdir()
    produced = {}
    publish = producer_test.publish_configured_scene_revision

    def capture_publication(**kwargs):
        result = publish(**kwargs)
        if Path(kwargs['output_root']).name == 'publication':
            produced['envelope'] = kwargs['envelope']
            produced['result'] = result
        return result

    monkeypatch.setattr(producer_test, 'publish_configured_scene_revision', capture_publication)
    producer_test.test_control_plane_publishes_reads_back_and_seals_robot_neutral_revision(
        publisher_root, False)
    request = produced['envelope']['request']
    publication = produced['result']
    publication_path = publisher_root / 'publication' / 'task_evaluation_scene_configuration_publication.v1.json'
    publication_path.write_text(canonical_json(publication) + '\n')
    revision_path = Path(publication['configured_scene_revision']['path'])
    revision = json.loads(revision_path.read_bytes())

    pre = {'schema_version': 'task_evaluation_launch_preparation_result.v1',
           'status': 'inputs_materialized_awaiting_construction_adapter',
           'preparation_id': request['preparation_id'], 'run_id': request['run_id'],
           'team_namespace': request['team_namespace'],
           'source_commit': request['expected_production_commit'],
           'reference_count': 0, 'unique_object_count': 0,
           'content_addressed_reuse_count': 0, 'references': [],
           'full_byte_service_account_readback_passed': True,
           'service_account': 'fixture-service', 'service_account_uid': 1,
           'provider_mutation_performed': False, 'catalog_mutation_performed': False,
           'paid_execution_requested': False, 'observed_at_iso': '2026-09-29T00:00:00Z',
           'result_digest': ''}
    pre['result_digest'] = canonical_digest(pre, digest_field='result_digest')
    recipe = dict(produced['envelope']['recipe'], output_identity=request['construction']['output_identity'],
                  stage_sequence=[], recipe_digest='')
    recipe['recipe_digest'] = canonical_digest(recipe, digest_field='recipe_digest')
    render = {'schema_version': 'task_evaluation_scene_configuration_render_inputs.v1',
              'status': 'derived_method_inputs_materialized', 'run_id': request['run_id'],
              'raw_interiorgs_bytes_in_provider_packet': False,
              'provider_mutation_performed': False, 'paid_execution_requested': False,
              'result_digest': ''}
    render['result_digest'] = canonical_digest(render, digest_field='result_digest')
    root = tmp_path / 'construction-queue'
    receipt = stage_scene_construction(request=request, preparation_result=pre, recipe=recipe,
        recipe_configuration_references=[], render_inputs_result=render, queue_root=root)
    queued = json.loads(Path(receipt['queue_path']).read_bytes())
    final = finalize_scene_construction(queue_root=root,
        envelope={**queued, 'control_plane_envelope_digest': queued['envelope_digest']},
        terminal_result={'status': 'completed', 'run_id': request['run_id'],
            'source_commit': request['expected_production_commit'],
            'configuration_completed': True, 'configured_scene_published': True,
            'configured_scene_revision_digest': revision['revision_digest'],
            'publication_result_digest': publication['result_digest'],
            'full_byte_service_account_readback_passed': True,
            'continuing_spend_from_this_run': False, 'blockers': []})
    assert final['status'] == 'completed'
    result = {**pre, 'status': 'queued_for_production_scene_configuration',
              'construction_recipe_digest': recipe['recipe_digest'],
              'construction_orchestration_id': request['preparation_id'],
              'construction_queue_envelope_digest': queued['envelope_digest'],
              'construction_queue_receipt_digest': receipt['receipt_digest']}
    proofs = object.__new__(TerminalProofs)
    proofs.fresh = {'finished_observation': {'status': 'completed'},
                    'planner_context': {'scene_construction_queue_root': str(root)}}
    proofs.allowance = ActionAllowance(expires_at=999, now=lambda: 200, monotonic=lambda: 0)
    proofs.has_inventory = True
    proofs.member_roots = [publisher_root]
    proofs.physical = {str(path): _raw(path) for path in (publication_path, revision_path)}
    proofs.canonical = {}
    assert revision_path == publication_path.parent / 'configured_scene_revision.v1.json'
    assert publication['configured_scene_revision']['digest'] == _raw(revision_path)[1]
    assert publication['configured_scene_revision']['size_bytes'] == _raw(revision_path)[2]
    if drift == 'publication_bytes':
        publication_path.write_bytes(b'{}')
    elif drift == 'revision_bytes':
        revision_path.write_bytes(b'{}')
    elif drift == 'revision_not_archived':
        proofs.physical.pop(str(revision_path))

    if drift is not None:
        with pytest.raises(ValueError, match='scene_retirement_reference_'):
            proofs._index_scene_construction_handoff(result, request, render)
        return

    assert proofs._index_scene_construction_handoff(result, request, render) == _raw(
        root / 'completed' / Path(receipt['queue_path']).name)
    assert proofs.canonical[revision['revision_digest']] == {_raw(revision_path)}
    assert proofs.canonical[publication['result_digest']] == {_raw(publication_path)}
