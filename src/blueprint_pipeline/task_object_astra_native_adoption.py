"""Rebind two authored assets into a retained diagnostic packet, offline.

This builds a new construction packet; it does not provide native import,
collision-cooking, settling, placement, policy, or physical qualification.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .native_marked_area_rehearsal import direct_policy_request
from .native_task_arena_packet import materialize_native_task_arena_packet, validate_native_task_arena_packet_request
from .task_evaluation_rigid_destination_geometry import derive_rigid_destination_geometry
from .task_evaluation_rigid_owner_contract import _derive_configured_owner_success_contract
from .task_object_astra_authoring import AssetAuthoringError, file_record, save_json

BOOK_DIMENSIONS_M = (0.295304002, 0.397696028, 0.0211374)
TRAY_DIMENSIONS_M = (0.33, 0.48, 0.035)
TRAY_INTERIOR = {'minimum': [-0.16, -0.235, 0.005], 'maximum': [0.16, 0.235, 0.035]}
NATIVE_PENDING = ('native_import', 'collision_cooking', 'initial_support_settling',
                  'book_and_tray_camera_visibility', 'native_placement')


def _read(path: Path) -> dict[str, Any]:
    file_record(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise AssetAuthoringError('astra_native_adoption_document_invalid')
    return value


def _verify_record(value: Mapping[str, Any]) -> Path:
    path = Path(str(value.get('path') or ''))
    actual = file_record(path)
    if any(actual.get(key) != value.get(key) for key in ('sha256', 'size_bytes')):
        raise AssetAuthoringError('astra_native_adoption_asset_digest_mismatch')
    return path


def _asset_evidence(packaging_path: Path, static_path: Path, dimensions):
    packaging, static = _read(packaging_path), _read(static_path)
    asset = _verify_record(packaging.get('asset', {}))
    completion = packaging.get('physics_completion', {})
    if (packaging.get('claim_ceiling') != 'development_only' or packaging.get('native_qualified') is not False
            or completion.get('completion_digest') != canonical_digest(completion, digest_field='completion_digest')
            or completion.get('collision_source_matches_final_visual_mesh') is not True
            or completion.get('native_collision_cooking_qualified') is not False
            or static.get('schema_version') != 'task_evaluation_rigid_replacement_static_qualification.v1'
            or static.get('status') != 'authored_structure_statically_qualified'
            or static.get('result_digest') != canonical_digest(static, digest_field='result_digest')
            or static.get('claim_boundary', {}).get('native_simulator_import_qualified') is not False
            or static.get('replacement_usd', {}).get('sha256') != file_record(asset)['sha256']
            or static.get('replacement_usd', {}).get('size_bytes') != asset.stat().st_size):
        raise AssetAuthoringError('astra_native_adoption_static_evidence_invalid')
    bounds = static['observed_structure']['collision_bounds_body_frame_m']
    extent = [bounds['maximum'][axis] - bounds['minimum'][axis] for axis in range(3)]
    if (any(not math.isfinite(value) or abs(value - expected) > 0.00001
            for value, expected in zip(extent, dimensions, strict=True))
            or abs(bounds['minimum'][2]) > 0.00001
            or any(abs(bounds['minimum'][axis] + bounds['maximum'][axis]) > 0.00002 for axis in (0, 1))):
        raise AssetAuthoringError('astra_native_adoption_exact_geometry_or_frame_mismatch')
    return {'asset': asset, 'packaging': packaging, 'static': static,
            'packaging_record': file_record(packaging_path), 'static_record': file_record(static_path)}


def _geometry_task_spec(source_spec, *, book_pose, tray_pose, geometry, support_top_z_m, authority_digest):
    """Replace geometry-dependent fields while retaining every behavioral limit."""
    spec = copy.deepcopy(source_spec)
    half_height = BOOK_DIMENSIONS_M[2] / 2
    start = [*book_pose['position_world_m'], *book_pose['orientation_xyzw']]
    start[2] += half_height
    # Plan the scoring-frame center on the tray floor, not halfway up its volume.
    target = [*tray_pose['position_world_m']]
    target[2] += TRAY_INTERIOR['minimum'][2] + half_height
    old_target = source_spec['configured_success_criteria']['target_center_xyz_m']
    old_bounds = source_spec['destination_position_bounds_world_m']
    bounds = {side: [target[axis] + old_bounds[side][axis] - old_target[axis]
                     for axis in range(3)] for side in ('minimum', 'maximum')}
    spec.update(start_pose_world=start, target_position_world_m=target,
        destination_pose_world=[*tray_pose['position_world_m'], *tray_pose['orientation_xyzw']],
        destination_position_bounds_world_m=bounds,
        destination_position_bounds_destination_frame_m=geometry['destination_position_bounds_destination_frame_m'],
        subject_collision_bounds_scoring_frame_m=geometry['subject_collision_bounds_scoring_frame_m'],
        destination_interior_bounds_body_frame_m=geometry['destination_interior_bounds_body_frame_m'],
        support_height_interval_m=geometry['support_height_interval_m'],
        asset_adoption_authority_digest=authority_digest)
    for name in ('configured_success_criteria', 'success_criteria'):
        if isinstance(spec.get(name), dict):
            spec[name]['target_center_xyz_m'] = list(target)
            spec[name]['per_cell_controls_required'] = False
    affordance = spec['interaction_affordance']
    affordance['asset_root_from_scoring_frame'] = {'position_m': [0.0, 0.0, half_height],
                                                  'orientation_xyzw': [0.0, 0.0, 0.0, 1.0]}
    affordance['contact_point_scoring_frame_m'] = [0.0, -BOOK_DIMENSIONS_M[1] / 2, 0.0]
    affordance['intended_support_prim_paths'] = list(geometry['intended_support_prim_paths'])
    affordance['support_alignment'] = {
        'initial_support_penetration_permitted': False, 'replacement_minimum_root_z_m': 0.0,
        'source_minimum_scoring_z_m': -half_height,
        'support_aligned_root_z_m': support_top_z_m, 'support_top_z_m': support_top_z_m,
        'initial_clearance_m': book_pose['position_world_m'][2] - support_top_z_m,
        'native_settling_qualified': False}
    affordance['affordance_digest'] = canonical_digest(affordance, digest_field='affordance_digest')
    spec.pop('task_success_contract', None)
    spec.pop('task_success_contract_digest', None)
    # This digest refers to old source/native-import geometry and must not label
    # the new task. The untouched source packet retains its historical lineage.
    spec.pop('configured_task_source_documents_digest', None)
    return spec


def rebind_book_tray_request(*, source_request, book, tray, tray_simready, authority,
                            support_top_z_m: float, qualification_limits,
                            native_settle_spec: Mapping[str, Any]) -> tuple[dict, dict]:
    """Build a request only after real packaging/static evidence is verified."""
    request = validate_native_task_arena_packet_request(source_request)
    if (authority.get('schema_version') != 'astra_native_asset_adoption_authority.v1'
            or authority.get('authority_digest') != canonical_digest(authority, digest_field='authority_digest')
            or not authority.get('authorized_by') or not authority.get('authorization_reference')
            or authority.get('source_request_digest') != source_request['request_digest']
            or authority.get('book_asset_sha256') != file_record(book['asset'])['sha256']
            or authority.get('tray_asset_sha256') != file_record(tray['asset'])['sha256']
            or authority.get('run_kind') != 'internal_policy_canary'
            or authority.get('claim_ceiling') != 'diagnostic_policy_execution'
            or not math.isfinite(support_top_z_m)):
        raise AssetAuthoringError('astra_native_adoption_authority_invalid')
    assets = {row['semantic_role']: row for row in request['assets']}
    if set(assets) != {'scene_appearance', 'scene_collision', 'task_object', 'task_support'}:
        raise AssetAuthoringError('astra_native_adoption_asset_roles_invalid')
    for role in ('task_object', 'task_support'):
        if assets[role]['pose_world']['orientation_xyzw'] != [0, 0, 0, 1]:
            raise AssetAuthoringError('astra_native_adoption_nonidentity_asset_pose_unsupported')
        pose = assets[role]['pose_world']
        pose['position_world_m'][2] = support_top_z_m + 0.002
        assets[role]['reset_state']['root_pose_world'] = copy.deepcopy(pose)
    book_pose, tray_pose = assets['task_object']['pose_world'], assets['task_support']['pose_world']
    if (tray_simready.get('schema_version') != 'task_evaluation_passive_destination_simready.v1'
            or tray_simready.get('result_digest') != canonical_digest(tray_simready, digest_field='result_digest')
            or tray_simready.get('destination_identity') != tray['static']['replacement_identity']
            or tray_simready.get('asset', {}).get('sha256') != file_record(tray['asset'])['sha256']
            or tray_simready.get('static_result_digest') != tray['static']['result_digest']
            or tray_simready.get('interior_bounds_body_frame_m') != TRAY_INTERIOR):
        raise AssetAuthoringError('astra_native_adoption_tray_interior_unbound')
    geometry = derive_rigid_destination_geometry(
        subject_identity=book['static']['replacement_identity'], destination_identity=tray['static']['replacement_identity'],
        relation='inside', pose_world=tray_pose,
        subject_static_qualification=book['static'], subject_static_qualification_digest=book['static_record']['sha256'],
        subject_scoring_transform={'position_m': [0, 0, BOOK_DIMENSIONS_M[2] / 2], 'orientation_xyzw': [0, 0, 0, 1]},
        destination_static_qualification=tray['static'], destination_static_qualification_digest=tray['static_record']['sha256'],
        destination_simready_result=tray_simready, qualification_limits=qualification_limits)
    spec = _geometry_task_spec(request['task_spec'], book_pose=book_pose, tray_pose=tray_pose,
        geometry=geometry, support_top_z_m=support_top_z_m, authority_digest=authority['authority_digest'])
    support_binding = {'schema_version': 'astra_collision_support_binding.v1',
        'scene_collision_digest': native_settle_spec['scene_collision_digest'],
        'support_measurement_digest': native_settle_spec['support_measurement_digest'],
        'scene_prim_paths': spec['initial_source_support']['scene_prim_paths'],
        'top_z_m': support_top_z_m, 'native_settling_qualified': False}
    support_binding['support_plane_digest'] = canonical_digest(support_binding, digest_field='support_plane_digest')
    spec['initial_source_support']['support_plane_digest'] = support_binding['support_plane_digest']
    source_adapter = request.pop('configured_task_template_adapter', None)
    spec['source_subject_identity'] = book['static']['replacement_identity']['id']
    owner = spec['configured_owner_authority']
    contract = _derive_configured_owner_success_contract(spec, site_id=request['scene_id'], task_id=request['task_id'],
        team_namespace=owner.get('confirmed_by_team_id'))
    if contract is None:
        raise AssetAuthoringError('astra_native_adoption_owner_contract_missing')
    spec['task_success_contract'], spec['task_success_contract_digest'] = contract, contract['contract_digest']
    request['task_spec'] = spec
    document = request['scenario']['context_document']
    for prefix, values in (('object_start', spec['start_pose_world'][:3]), ('target', spec['target_position_world_m'])):
        for axis, value in zip('xyz', values, strict=True):
            document['resolved_parameters'][f'{prefix}_{axis}_m'] = value
    document.pop('configured_task_source_documents_digest', None)
    document['asset_adoption_authority_digest'] = authority['authority_digest']
    document['instance_digest'] = canonical_digest(document, digest_field='instance_digest')
    request['scenario']['instance_digest'] = document['instance_digest']
    # Keep every camera/reset byte; only its task-contract binding changes.
    camera = request.get('policy_canary_camera_start_configuration')
    if camera is not None:
        camera['task_success_contract_digest'] = contract['contract_digest']
        camera['configuration_digest'] = canonical_digest(camera, digest_field='configuration_digest')
    request['astra_asset_adoption'] = {
        'schema_version': 'astra_native_asset_adoption.v1', 'authority': dict(authority),
        'source_request_digest': source_request['request_digest'],
        'historical_adapter_digest': source_adapter.get('adapter_digest') if source_adapter else None,
        'native_qualification_status': 'pending', 'required_prepolicy_native_checks': list(NATIVE_PENDING),
        'native_application_claimed': False, 'policy_execution_authorized_by_this_record': False,
        'physical_equivalence_proven': False, 'destination_geometry_digest': geometry['geometry_digest'],
        'support_binding': support_binding, 'native_settle_spec': copy.deepcopy(dict(native_settle_spec))}
    request['astra_asset_adoption']['adoption_digest'] = canonical_digest(
        request['astra_asset_adoption'], digest_field='adoption_digest')
    spec['astra_asset_adoption'] = copy.deepcopy(request['astra_asset_adoption'])
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    omission = source_request['diagnostic_control_omission_authority']
    request = direct_policy_request(source_request=request, authorized_by=omission['authorized_by'],
                                    authorization_reference=omission['authorization_reference'])
    return request, geometry


def materialize_astra_native_adoption(*, source_packet_root: Path, book_packaging_path: Path,
        book_static_path: Path, tray_packaging_path: Path, tray_static_path: Path,
        tray_simready_path: Path, authority: Mapping[str, Any], support_measurement_path: Path,
        settle_reference_path: Path, output_root: Path, qualification_limits: Mapping[str, Any]) -> dict[str, Any]:
    """Stage exact retained scene bytes and compile a new unqualified packet."""
    source_packet_root = source_packet_root.resolve(strict=True)
    source_request = _read(source_packet_root / 'native_task_arena_packet_request.v1.json')
    source_receipt = _read(source_packet_root / 'native_task_arena_packet_receipt.v1.json')
    if (source_receipt.get('receipt_digest') != canonical_digest(source_receipt, digest_field='receipt_digest')
            or source_receipt.get('request_digest') != source_request['request_digest']):
        raise AssetAuthoringError('astra_native_adoption_source_packet_invalid')
    book = _asset_evidence(book_packaging_path, book_static_path, BOOK_DIMENSIONS_M)
    tray = _asset_evidence(tray_packaging_path, tray_static_path, TRAY_DIMENSIONS_M)
    support = _read(support_measurement_path)
    values = [support[role][key] for role in ('book', 'tray') for key in ('collision_top_min', 'collision_top_max')]
    if (any(not math.isfinite(value) or abs(value - 0.27650001) > 0.000001 for value in values)
            or max(values) - min(values) > 0.000001
            or any(support[role]['collision_surface_samples'] < 81 for role in ('book', 'tray'))):
        raise AssetAuthoringError('astra_native_adoption_support_observation_invalid')
    source_bindings = {row['semantic_role']: row for row in source_receipt['source_bindings']}
    collision_binding = source_bindings['scene_collision']
    collision_path = source_packet_root / collision_binding['staged_relative_path']
    collision_record = file_record(collision_path)
    if collision_record['sha256'] != collision_binding['staged_sha256']:
        raise AssetAuthoringError('astra_native_adoption_retained_scene_changed')
    support_paths = source_request['task_spec']['initial_source_support']['scene_prim_paths']
    if not support_paths or any(not path.startswith('/Root/') for path in support_paths):
        raise AssetAuthoringError('astra_native_adoption_support_path_unsupported')
    source_assets = {row['semantic_role']: row for row in source_request['assets']}
    settle_reference = _read(settle_reference_path)
    gravity_seconds = settle_reference.get('qualification_limits', {}).get('gravity_settle_seconds')
    if isinstance(gravity_seconds, bool) or not isinstance(gravity_seconds, (int, float)) or gravity_seconds != 3.0:
        raise AssetAuthoringError('astra_native_adoption_settle_limit_source_invalid')
    settle_spec = {'schema_version': 'task_object_native_settle_spec.v1', 'required': True,
        'source_packet_digest': source_receipt['receipt_digest'],
        'scene_collision_digest': collision_record['sha256'],
        'support_measurement_digest': file_record(support_measurement_path)['sha256'],
        'gravity_settle_seconds': gravity_seconds,
        'settle_window_samples': source_request['task_spec']['settle_window_samples'],
        'control_hz': source_request['task_spec']['control_frequency_hz'],
        'physics_hz': source_request['physics_frequency_hz'],
        'qualification_limits': dict(qualification_limits),
        'objects': [{'role': role, 'asset_id': source_assets[role]['asset_id'],
            'asset_version': evidence['static']['replacement_identity']['version'],
            'asset_sha256': file_record(evidence['asset'])['sha256'],
            'collision_bounds_body_frame_m': evidence['static']['observed_structure']['collision_bounds_body_frame_m'],
            'support_source_prim_paths': list(support_paths),
            'support_native_prim_paths': ['{ENV_REGEX_NS}/scene_collision' + path.removeprefix('/Root') for path in support_paths],
            'support_top_z_m': max(values)}
            for role, evidence in (('task_object', book), ('task_support', tray))]}
    settle_spec['spec_digest'] = canonical_digest(settle_spec, digest_field='spec_digest')
    request, geometry = rebind_book_tray_request(source_request=source_request, book=book, tray=tray,
        tray_simready=_read(tray_simready_path), authority=authority, support_top_z_m=max(values),
        qualification_limits=qualification_limits, native_settle_spec=settle_spec)
    if output_root.exists():
        raise AssetAuthoringError('astra_native_adoption_output_exists')
    output_root.mkdir(parents=True)
    evidence = output_root / 'inputs'
    evidence.mkdir()
    lineage = output_root / 'lineage'
    lineage.mkdir()
    for name in ('native_task_arena_packet_request.v1.json', 'native_task_arena_packet_receipt.v1.json',
                 'native_task_arena_scene_plan.v1.json', 'native_task_runtime_contract.v1.json'):
        shutil.copyfile(source_packet_root / name, lineage / name)
    bindings = {row['semantic_role']: row for row in source_receipt['source_bindings']}
    frozen_scene = {}
    for row in request['assets']:
        role = row['semantic_role']
        if role in ('scene_appearance', 'scene_collision'):
            prior = bindings[role]
            path = source_packet_root / prior['staged_relative_path']
            actual = file_record(path)
            if (actual['sha256'] != prior['staged_sha256'] or actual['size_bytes'] != prior['staged_size_bytes']):
                raise AssetAuthoringError('astra_native_adoption_retained_scene_changed')
            frozen_scene[role] = actual
        else:
            path = book['asset'] if role == 'task_object' else tray['asset']
        staged = evidence / row['filename']
        try:
            os.link(path, staged, follow_symlinks=False)
        except OSError:
            shutil.copyfile(path, staged)
        actual = file_record(staged)
        row['source'] = {'root': 'evidence', 'relative_path': staged.name,
                         'sha256': actual['sha256'], 'size_bytes': actual['size_bytes']}
    request['request_digest'] = canonical_digest(request, digest_field='request_digest')
    save_json(output_root / 'asset_change_authority.json', authority)
    save_json(output_root / 'destination_geometry.json', geometry)
    receipt = materialize_native_task_arena_packet(request=request, evidence_root=evidence,
        output_dir=output_root / 'packet', link_sources_within=output_root)
    for role, original in frozen_scene.items():
        bound = next(row for row in receipt['source_bindings'] if row['semantic_role'] == role)
        if bound['staged_sha256'] != original['sha256'] or file_record(Path(original['path'])) != original:
            raise AssetAuthoringError('astra_native_adoption_scene_bytes_not_preserved')
    result = {'schema_version': 'astra_native_asset_adoption_result.v1',
        'status': 'construction_packet_ready_native_qualification_pending',
        'run_kind': 'internal_policy_canary', 'claim_ceiling': 'diagnostic_policy_execution',
        'packet_receipt': file_record(output_root / 'packet/native_task_arena_packet_receipt.v1.json'),
        'source_packet_receipt': file_record(source_packet_root / 'native_task_arena_packet_receipt.v1.json'),
        'book_packaging': book['packaging_record'], 'book_static_qualification': book['static_record'],
        'tray_packaging': tray['packaging_record'], 'tray_static_qualification': tray['static_record'],
        'support_measurements': file_record(support_measurement_path),
        'settle_limit_source': file_record(settle_reference_path),
        'scene_bytes_preserved': True, 'controls_skipped': True,
        'native_qualified': False, 'policy_queried': False, 'provider_allocations': 0,
        'pending_checks': list(NATIVE_PENDING)}
    result['result_digest'] = canonical_digest(result)
    save_json(output_root / 'adoption_result.json', result)
    return result
