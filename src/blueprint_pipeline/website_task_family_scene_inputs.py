"""Capture-bound family scene inputs for the existing beta screening registry."""
from __future__ import annotations

import json
from pathlib import Path

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_attempt_evidence import record
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
from .task_evaluation_scene_intent_contracts import _read as read_intent_record
from .task_evaluation_scene_progression_contracts import require, safe_path
from .website_task_family_packet import family_source_references

SCHEMA = 'blueprint.task_family_scene_inputs.v1'
FILENAME = 'family_scene_inputs.v1.json'


def _operator_runtime_source(preparation_path):
    """The SDK source receipt is selected only by the protected registry."""
    from .team_candidate_quick10_registry import load_registry, profile_for_capture, _derived_settings
    path = Path(preparation_path)
    if path.parent.name != 'website_scene_preparation' or path.parent.parent.name != 'pipeline':
        return None
    registry = load_registry()
    if registry is None:
        return None
    profile = profile_for_capture(path.parents[2])
    settings = _derived_settings(registry)
    return ((profile or {}).get('native_runtime_source_receipt') or
            ((settings or {}).get('defaults') or {}).get('native_runtime_source_receipt'))


def write_family_scene_inputs(*, packet, preparation, preparation_reference, task_context_reference,
                            runtime_inputs_reference, intent_reference):
    """Publish scene facts only; the operator registry supplies launch choices."""
    context = read(checked_file(task_context_reference['path'], task_context_reference), digest_field='context_digest')
    intent = read_intent_record(checked_file(intent_reference['path'], intent_reference), 'intent_digest')
    plan = read(checked_file(packet['scene_plan']['path'], packet['scene_plan']), digest_field='plan_digest')
    inputs = preparation['task_evaluation_inputs']
    runtime_source = _operator_runtime_source(preparation_reference['path'])
    supplied_runtime = inputs.get('runtime_source')
    runtime_gap = ('family_runtime_source_operator_binding_mismatch'
        if supplied_runtime is not None and supplied_runtime != runtime_source else None)
    if runtime_gap:
        runtime_source = None
    source = {'scene_intent_id': intent['intent_id'], 'scene_intent_digest': intent['intent_digest'],
        'capture_id': context['capture_id'], 'task_id': inputs['scope']['task_id'],
        'task_context_digest': context['context_digest'], 'scene_revision_digest': plan['plan_digest']}
    value = {'schema_version': SCHEMA, 'claim_ceiling': 'diagnostic_policy_execution',
        'source': source, 'scene_id': context['scene_id'], 'scene_revision_digest': plan['plan_digest'],
        'configuration_digest': canonical_digest({'native_runtime_contract_digest': plan['runtime_contract_digest'],
            'task_family_runtime_selection': plan['task_family_runtime_selection']}),
        'plan_digest': plan['plan_digest'], 'task_success_contract': inputs['task_success_contract'],
        'task_success_contract_digest': inputs['task_success_contract']['contract_digest'],
        'base_native_packet': packet['packet_receipt'], 'construction_result': packet['packet_receipt'],
        'runtime_source': runtime_source, 'runtime_source_gap': runtime_gap, 'packet_root': packet['packet_root'],
        **({'policy_selection': inputs['policy_selection']} if inputs.get('policy_selection') is not None else {}),
        'execution_authority': {'maximum_provider_allocations': 1, 'retry_cap': 0,
            'single_warm_provider_session_required': True, 'caller_surviving_watchdog_required': True,
            'billing_teardown_provider_zero_required': True},
        'control_diagnostic': {'mode': 'required_before_policy', 'strategy': 'native_task_family_prepolicy.v1'},
        'references': {'preparation': preparation_reference, 'task_context': task_context_reference,
            'runtime_inputs': runtime_inputs_reference, 'scene_intent': intent_reference},
        'native_application_claimed': False, 'policy_episode_claimed': False,
        'provider_mutation_performed': False, 'paid_authority_granted': False}
    value['inputs_digest'] = canonical_digest(value, digest_field='inputs_digest')
    path = safe_path(packet['packet_root']).parent / FILENAME
    write_json(path, value)
    # Retained scene inputs are also useful while the execution source is still
    # missing. Registering those facts cannot supply the absent runtime bytes.
    preparation_path = Path(preparation_reference['path'])
    profile = {'status': 'family_capture_layout_unavailable'}
    if preparation_path.parent.name == 'website_scene_preparation' and preparation_path.parent.parent.name == 'pipeline':
        from .team_candidate_quick10_registry import write_derived_profile
        profile = write_derived_profile({'capture_root': str(preparation_path.parents[2].resolve()),
            'scene_id': context['scene_id'], 'task_id': inputs['scope']['task_id'],
            'scenario_id': inputs['native_runtime_contract']['scenario_cell_id'],
            'robot_free_probe_root': str(path.parent.resolve()), 'scene_revision_digest': plan['plan_digest'],
            'family_scene_inputs_path': str(path.resolve())})
    return {'family_scene_inputs': record(path), 'family_profile': profile}


def read_family_scene_inputs(path):
    """Reopen exact packet/capture sources without granting execution authority."""
    value = read(safe_path(path), digest_field='inputs_digest')
    require(value.get('schema_version') == SCHEMA and value.get('claim_ceiling') == 'diagnostic_policy_execution'
            and value.get('native_application_claimed') is False and value.get('policy_episode_claimed') is False
            and value.get('provider_mutation_performed') is False and value.get('paid_authority_granted') is False
            and value.get('construction_result') == value.get('base_native_packet'), 'family_scene_inputs_invalid')
    refs = value['references']
    for reference in refs.values():
        checked_file(reference['path'], reference)
    preparation = read(refs['preparation']['path'], digest_field='digest')
    context = read(refs['task_context']['path'], digest_field='context_digest')
    intent = read_intent_record(Path(refs['scene_intent']['path']), 'intent_digest')
    family_source_references(preparation=preparation, runtime_inputs_path=refs['runtime_inputs']['path'])
    from .native_task_arena_bundle import verify_native_task_arena_packet
    packet_root, receipt, _ = verify_native_task_arena_packet(value['packet_root'])
    checked_file(value['base_native_packet']['path'], value['base_native_packet'])
    plan = read(packet_root / 'native_task_arena_scene_plan.v1.json', digest_field='plan_digest')
    native = preparation['task_evaluation_inputs']['native_runtime_contract']
    expected_source = {'scene_intent_id': intent['intent_id'], 'scene_intent_digest': intent['intent_digest'],
        'capture_id': context['capture_id'], 'task_id': native['task_id'],
        'task_context_digest': context['context_digest'], 'scene_revision_digest': plan['plan_digest']}
    require(value.get('source') == expected_source and value.get('scene_id') == context['scene_id']
            and value.get('scene_revision_digest') == plan['plan_digest']
            and value.get('plan_digest') == plan['plan_digest']
            and receipt.get('arena_scene_plan_digest') == plan['plan_digest']
            and plan.get('native_runtime_contract') == native
            and value.get('policy_selection') == preparation['task_evaluation_inputs'].get('policy_selection')
            and value.get('policy_selection') == plan.get('policy_selection')
            and intent['request'] == preparation['intake_request']
            and value.get('task_success_contract') == native['task_spec']['task_success_contract']
            and value.get('task_success_contract_digest') == value['task_success_contract']['contract_digest'],
            'family_scene_inputs_source_binding_changed')
    runtime = value.get('runtime_source')
    trusted = _operator_runtime_source(refs['preparation']['path'])
    supplied_runtime = preparation['task_evaluation_inputs'].get('runtime_source')
    require(supplied_runtime is None or supplied_runtime == trusted,
            'family_runtime_source_operator_binding_mismatch')
    require(runtime is None or runtime == trusted, 'family_runtime_source_operator_binding_changed')
    require(isinstance(trusted, dict) and set(trusted) == {'path', 'sha256', 'size_bytes'},
            'family_scene_runtime_source_required')
    checked_file(trusted['path'], trusted)
    if runtime is None:
        # A newly configured operator source may fill a preparation gap. Return
        # an explicitly resealed projection; retained source bytes stay intact.
        value['source_inputs_digest'] = value['inputs_digest']
        value['runtime_source'] = trusted
        value['runtime_source_gap'] = None
        value['inputs_digest'] = canonical_digest(value, digest_field='inputs_digest')
    # Detach caller data after every retained byte was checked.
    return json.loads(json.dumps(value, allow_nan=False))
