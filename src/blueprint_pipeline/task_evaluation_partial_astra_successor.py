"""Admit one original-root, partial Astra continuation across explicit run lineage."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Mapping
import zipfile

from .decision_evidence_contracts import canonical_digest
from .task_object_astra_authoring import AssetAuthoringError, AppearanceReview, appearance_passed, file_record

SCHEMA_VERSION = 'task_evaluation_partial_astra_successor_adoption.v1'
IDENTITY_KEYS = ('source_run_id', 'successor_run_id', 'owner_id', 'stable_intent_id', 'stable_intent_digest')


def semantic_request(request: Mapping) -> dict:
    """Only attempt identity and code version may differ; all scientific inputs remain."""
    return {key: value for key, value in request.items()
            if key not in {'run_id', 'request_digest', 'expected_production_commit'}}


def _verify_descriptor(value, *, request_value, original_root, verified_lineage):
    if (value.get('schema_version') != SCHEMA_VERSION
            or value.get('adoption_digest') != canonical_digest(value, digest_field='adoption_digest')
            or value.get('original_runtime_root') != str(original_root)
            or not original_root.is_absolute()
            or value.get('successor_run_id') != request_value['run_id']
            or value.get('source_run_id') == request_value['run_id']
            or value.get('semantic_request_digest') != canonical_digest(semantic_request(request_value))):
        raise AssetAuthoringError('astra_partial_successor_descriptor_invalid')
    claimed = value.get('owner_intent_lineage')
    if (not isinstance(claimed, dict) or not isinstance(verified_lineage, Mapping)
            or any(not isinstance(claimed.get(key), str) or not claimed[key]
                   or claimed[key] != verified_lineage.get(key) for key in IDENTITY_KEYS)
            or any(claimed[key] != value[key] for key in ('source_run_id', 'successor_run_id'))):
        raise AssetAuthoringError('astra_partial_successor_owner_intent_unverified')


def restore_partial_astra(*, value: Mapping, request_value: dict, original_root: Path,
                          verified_lineage: Mapping, archive_path: Path) -> None:
    """Restore exact archived bytes only at their original absolute root, before reuse."""
    _verify_descriptor(value, request_value=request_value, original_root=original_root,
                       verified_lineage=verified_lineage)
    record = value['retained_runtime_archive']
    archive = archive_path
    actual = file_record(archive)
    if (any(actual[key] != record.get(key) for key in ('sha256', 'size_bytes'))
            or archive.stat().st_size > 1024**3):
        raise AssetAuthoringError('astra_partial_successor_archive_invalid')
    inventory = value.get('retained_files')
    if not isinstance(inventory, list) or not inventory:
        raise AssetAuthoringError('astra_partial_successor_inventory_missing')
    expected = {}
    for row in inventory:
        relative = PurePosixPath(row['relative_path'])
        if (relative.is_absolute() or '..' in relative.parts or str(relative) != row['relative_path']
                or not relative.parts or str(relative) in expected or type(row['size_bytes']) is not int
                or row['size_bytes'] < 0):
            raise AssetAuthoringError('astra_partial_successor_inventory_invalid')
        expected[str(relative)] = row
    if sum(row['size_bytes'] for row in inventory) > 2 * 1024**3:
        raise AssetAuthoringError('astra_partial_successor_archive_too_large')
    if original_root.exists():
        from .task_evaluation_scene_configuration_astra_phase_adoption import _inventory
        if original_root.is_symlink() or _inventory(original_root) != inventory:
            raise AssetAuthoringError('astra_partial_successor_original_root_changed')
        return
    original_root.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='.partial-astra-', dir=original_root.parent))
    try:
        with zipfile.ZipFile(archive) as bundle:
            entries = [entry for entry in bundle.infolist() if not entry.is_dir()]
            if len(entries) != len(expected) or {entry.filename for entry in entries} != set(expected):
                raise AssetAuthoringError('astra_partial_successor_archive_inventory_mismatch')
            for entry in entries:
                row = expected[entry.filename]
                if (stat.S_IFMT(entry.external_attr >> 16) not in {0, stat.S_IFREG}
                        or entry.file_size != row['size_bytes']):
                    raise AssetAuthoringError('astra_partial_successor_archive_entry_invalid')
                target = scratch / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with bundle.open(entry) as source, target.open('xb') as output:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
                if 'sha256:' + digest.hexdigest() != row['sha256']:
                    raise AssetAuthoringError('astra_partial_successor_archive_entry_changed')
        from .task_evaluation_scene_configuration_astra_phase_adoption import _inventory
        if _inventory(scratch) != inventory:
            raise AssetAuthoringError('astra_partial_successor_unrecognized_inventory')
        os.rename(scratch, original_root)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def prepare_partial_astra_successor(*, value: Mapping, request_value: dict, source_binding: dict,
                                    verified_lineage: Mapping, package: Path, budget_root: Path) -> dict:
    """Verify original outputs under original identities, then resume only final review."""
    from .task_evaluation_scene_configuration_astra_phase_adoption import (
        _inventory, materialize_automatic_phase_adoption, prepare_phase_adoption,
    )
    from .task_object_astra_inherited_inference import inherit_completed_inference
    from .task_object_astra_retained_artifacts import completed_blender, completed_cad, completed_visual_review
    from .task_object_astra_authoring import BlenderProgram, validate_request

    prior = Path(value['original_runtime_root'])
    _verify_descriptor(value, request_value=request_value, original_root=prior, verified_lineage=verified_lineage)
    if _inventory(prior) != value['retained_files']:
        raise AssetAuthoringError('astra_partial_successor_retained_bytes_changed')
    previous = json.loads((prior / 'authoring/request.json').read_text())
    old_request = validate_request(previous)
    if (previous['run_id'] != value['source_run_id'] or previous['request_digest'] != value['source_request_digest']
            or semantic_request(previous) != semantic_request(request_value)):
        raise AssetAuthoringError('astra_partial_successor_scientific_inputs_changed')
    old_binding = json.loads((prior / 'stage_source_binding.json').read_text())
    if (old_binding.get('binding_digest') != value['source_stage_binding_digest']
            or old_binding.get('binding_digest') != canonical_digest(old_binding, digest_field='binding_digest')
            or old_binding.get('run_id') != previous['run_id']
            or old_binding.get('authoring_input_digest') != canonical_digest({
                k: v for k, v in previous.items() if k not in {'request_digest', 'expected_production_commit'}})
            or any(old_binding.get(key) != source_binding.get(key)
                   for key in ('configuration_sha256', 'source_candidate', 'rights_admission'))):
        raise AssetAuthoringError('astra_partial_successor_source_rights_or_configuration_changed')
    automatic = materialize_automatic_phase_adoption(prior_runtime=prior)
    if (automatic['phases'] != ['source_analysis', 'physical_review', 'blender_program']
            or automatic['completed_artifacts'] != ['cad_result', 'blender_execution']
            or automatic['blender_round'] != 1):
        raise AssetAuthoringError('astra_partial_successor_requires_pending_second_review')
    # Preserve and authenticate the rejected first review, including its exact images.
    cad, _ = completed_cad(prior / 'authoring', old_request)
    first_program = BlenderProgram.model_validate(json.loads(
        (prior / 'authoring/appearance-00/blender_author_0.json').read_text())['output'])
    first_execution = completed_blender(prior / 'authoring', old_request, round_index=0, program=first_program, cad=cad)
    first_review, _ = completed_visual_review(prior / 'authoring', previous, prior / 'inference',
                                            round_index=0, execution=first_execution)
    if appearance_passed(AppearanceReview.model_validate(first_review)):
        raise AssetAuthoringError('astra_partial_successor_first_review_not_rejected')
    validation_root = budget_root.parent / 'retained_astra_phases/source-validation'
    prepared = prepare_phase_adoption(value=automatic, request_value=previous, package=package,
                                      budget_root=validation_root / 'inference')
    balance = inherit_completed_inference(source=validation_root / 'inference', destination=budget_root,
                                         source_run_id=previous['run_id'], successor_run_id=request_value['run_id'])
    if balance['cost_usd'] != prepared['retained_inference_cost_usd'] or balance['call_count'] != prepared['prior_call_count']:
        raise AssetAuthoringError('astra_partial_successor_inference_balance_changed')
    record = prepared['authoring_kwargs']['adoption_record']
    record['new_request_digest'] = request_value['request_digest']
    prepared.update(adoption_digest=value['adoption_digest'], retained_inference_cost_usd=balance['cost_usd'],
                    prior_call_count=balance['call_count'])
    lineage = {'schema_version': 'astra_partial_successor_lineage.v1',
               'source_run_id': previous['run_id'], 'successor_run_id': request_value['run_id'],
               'source_request_digest': previous['request_digest'], 'successor_request_digest': request_value['request_digest'],
               'source_authoring_root': str(prior / 'authoring'), 'adoption_digest': value['adoption_digest'],
               'owner_intent_lineage': dict(verified_lineage), 'prior_call_count': balance['call_count'],
               'inherited_token_pricing_cost_usd': balance['cost_usd'], 'official_billing_proven': False,
               'pending_phase': 'independent_visual_review_1'}
    lineage['lineage_digest'] = canonical_digest(lineage, digest_field='lineage_digest')
    (budget_root.parent / 'partial_successor_lineage.json').write_text(json.dumps(lineage, indent=2) + '\n')
    return prepared
