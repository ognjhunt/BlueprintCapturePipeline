"""Prepare a paired runtime diagnostic from an unchanged retained task and policies."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_runtime_comparison import (
    BASE_ARENA_COMMIT, BASE_COMMIT, BASE_IMAGE, GA_ARENA_COMMIT, GA_COMMIT, GA_IMAGE,
    retained_workcell_binding, validate_comparison_request,
)


def prepare_retained_policy_comparison(
    *, packet_dir: str | Path, runtime_source_receipt: str | Path,
    retained_spec_paths: Mapping[str, str | Path], readiness_path: str | Path,
    construction_path: str | Path, control_path: str | Path,
    implementation_commit: str, variant: str, confirmed_by: str,
    authorization_reference: str, output_dir: str | Path,
    allowed_machine_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Translate existing freezes; never create a Website activation or qualify controls.

    The caller must name the human authorizing reuse of the unchanged task. No
    thresholds, prompts, model bytes, policy adapters or query limits are chosen
    here. Both runtimes use the existing Quick-10 diagnostic harness.
    """
    from .adp_articulated_task_success_contract import seal_task_success_contract
    from .native_task_arena_bundle import verify_native_task_arena_packet
    from .native_task_arena_policy_bundle import _candidate_runtime_binding
    from .native_task_arena_policy_canary_session import (
        CANDIDATE_IDS, CLAIM_CEILING, RUN_KIND, build_session_authority,
        validate_runtime_input_manifest,
    )
    from .task_evaluation_policy_canary_scene_setup import _quick_cells, _record

    if variant not in {'baseline', 'isaaclab_3_ga'} or not confirmed_by.strip() or not authorization_reference.strip():
        raise ValueError('retained_runtime_comparison_authority_missing')
    if set(retained_spec_paths) != set(CANDIDATE_IDS):
        raise ValueError('retained_runtime_comparison_candidate_set_changed')
    root, _, _ = verify_native_task_arena_packet(packet_dir)
    plan = json.loads((root / 'native_task_arena_scene_plan.v1.json').read_text())
    readiness = json.loads(Path(readiness_path).read_text())
    control = json.loads(Path(control_path).read_text())
    if (readiness.get('readiness_digest') != canonical_digest(readiness, digest_field='readiness_digest')
            or control.get('result_digest') != canonical_digest(control, digest_field='result_digest')
            or control.get('scene_plan_digest') != plan['plan_digest']):
        raise ValueError('retained_runtime_comparison_evidence_changed')
    candidates = {row['candidate_id']: row for row in readiness['candidates']}
    contract = seal_task_success_contract(
        task_kind=plan['task_kind'], task_spec=plan['task_spec'], site_id=plan['scene_id'],
        task_id=plan['task_id'], author_source='compatibility_default',
        author_id='retained-task:' + canonical_digest({'value': plan['task_spec']}),
        confirmation_status='confirmed',
        confirmed_by_team_id=confirmed_by if plan['task_kind'] == 'articulated_open_close' else None,
    )
    ga = variant == 'isaaclab_3_ga'
    identity = {'container_image': GA_IMAGE if ga else BASE_IMAGE,
                'isaaclab_commit': GA_COMMIT if ga else BASE_COMMIT,
                'arena_commit': GA_ARENA_COMMIT if ga else BASE_ARENA_COMMIT,
                'physics_backend': 'isaacsim_physx', 'renderer': 'isaac_rtx'}
    specs = {}
    for candidate_id in CANDIDATE_IDS:
        old = json.loads(Path(retained_spec_paths[candidate_id]).read_text())
        policy, endpoint, policy_identity = _candidate_runtime_binding(candidate_id)
        candidate = candidates[candidate_id]
        checkpoint = candidate['checkpoint']
        if (old.get('execution_spec_digest') != canonical_digest(old, digest_field='execution_spec_digest')
                or old.get('candidate_id') != candidate_id
                or old.get('scene_plan_digest') != plan['plan_digest']
                or old.get('prompt') != plan['task_spec'].get('prompt')
                or old.get('policy_spec') != asdict(policy)
                or old.get('open_loop_horizon') != policy.open_loop_horizon
                or candidate.get('rights_ready') is not True
                or old.get('candidate_rights_binding', {}).get('checkpoint_identity', {}).get('inventory_digest')
                != checkpoint['inventory_digest']):
            raise ValueError('retained_runtime_comparison_policy_inputs_changed:' + candidate_id)
        runtime_identity = {
            'source_commit': implementation_commit, 'container_image': identity['container_image'],
            'candidate_source': candidate['source'], 'policy_identity': policy_identity,
            'checkpoint_inventory_digest': checkpoint['inventory_digest'],
            'observation_adapter': candidate['policy_input_schema'], 'action_adapter': candidate['action_adapter'],
        }
        spec = {
            'schema_version': 'native_task_arena_policy_canary_execution_spec.v1',
            'candidate_id': candidate_id, 'execution_authority': 'internal_policy_canary_unqualified',
            'claim_ceiling': CLAIM_CEILING, 'scene_id': plan['scene_id'], 'task_id': plan['task_id'],
            'scene_plan_digest': plan['plan_digest'], 'prompt': old['prompt'],
            'policy_endpoint': endpoint, 'policy_spec': old['policy_spec'],
            'candidate_rights_binding': old['candidate_rights_binding'],
            'checkpoint_digest': checkpoint['inventory_digest'], 'runtime_identity': runtime_identity,
            'runtime_identity_digest': canonical_digest(runtime_identity),
            'worker_identity_requirement': 'groot_droid_runtime_measurement'
                if policy_identity.get('status') == 'runtime_measurement_required' else 'none',
            'require_observed_eef_support': candidate_id == 'groot_n17_droid',
            'max_policy_queries': old['max_policy_queries'], 'open_loop_horizon': old['open_loop_horizon'],
            'ranking_permitted': False, 'qualification_permitted': False, 'scene_promotion_permitted': False,
            'scoring_authority': 'deterministic_simulator_state', 'task_success_contract': contract,
            'task_success_contract_digest': contract['contract_digest'],
            'retained_execution_spec': _record(retained_spec_paths[candidate_id]),
        }
        spec['execution_spec_digest'] = canonical_digest(spec, digest_field='execution_spec_digest')
        specs[candidate_id] = spec
    cells = _quick_cells(plan['plan_digest'], scene_id=plan['scene_id'].removeprefix('interiorgs-'))
    for cell in cells:
        cell['resolved_scenario_digest'] = canonical_digest(cell['resolved_scenario'])
        cell['control_diagnostic'] = {'mode': 'nonblocking_diagnostic_bound',
            'policy_execution_blocked': False, 'receipt': _record(control_path)}
        cell['retained_runtime_comparison_prompt'] = {'base_plan_digest': plan['plan_digest'],
            'task_digest': canonical_digest({'value': plan['task_spec']}), 'prompt': plan['task_spec']['prompt']}
    configuration_digest = canonical_digest({'task': contract, 'cells': cells, 'plan_digest': plan['plan_digest']})
    activation = {'schema_version': 'retained_runtime_comparison_activation.v1',
        'run_id': 'retained-runtime-comparison-' + plan['task_id'] + '-' + variant,
        'run_kind': RUN_KIND, 'claim_ceiling': CLAIM_CEILING, 'candidate_ids': list(CANDIDATE_IDS),
        'campaign_unit_count': len(cells), 'task_success_contract': contract,
        'task_success_contract_digest': contract['contract_digest'], 'authorized_by': confirmed_by,
        'authorization_reference': authorization_reference, 'task_criteria_changed': False,
        'controls_reused_as_unqualified_diagnostics': True, 'configuration_digest': configuration_digest}
    activation['activation_digest'] = canonical_digest(activation, digest_field='activation_digest')
    inputs = {'schema_version': 'task_evaluation_policy_canary_runtime_inputs.v1',
        'run_id': activation['run_id'], 'run_kind': RUN_KIND, 'claim_ceiling': CLAIM_CEILING,
        'scene_revision_digest': plan['plan_digest'], 'matrix_digest': canonical_digest({'cells': cells}),
        'configuration_digest': configuration_digest, 'plan_digest': plan['plan_digest'],
        'activation_digest': activation['activation_digest'], 'task_success_contract': contract,
        'task_success_contract_digest': contract['contract_digest'],
        'base_native_packet': _record(root / 'native_task_arena_packet_receipt.v1.json'),
        'runtime_source': _record(runtime_source_receipt), 'construction_result': _record(construction_path),
        'policy_readiness': _record(readiness_path), 'candidate_ids': list(CANDIDATE_IDS), 'cells': cells,
        'execution_authority': {'maximum_provider_allocations': 1, 'retry_cap': 0,
            'single_warm_provider_session_required': True, 'caller_surviving_watchdog_required': True,
            'billing_teardown_provider_zero_required': True}}
    inputs['runtime_inputs_digest'] = canonical_digest(inputs, digest_field='runtime_inputs_digest')
    validate_runtime_input_manifest(inputs)
    comparison = {'schema_version': 'native_task_runtime_comparison_request.v1', 'variant': variant,
        'scope': 'retained_policy_diagnostic', 'runtime_identity': identity,
        'binding': retained_workcell_binding(inputs=inputs, plan=plan, specs=specs)}
    comparison['request_digest'] = canonical_digest(comparison, digest_field='request_digest')
    validate_comparison_request(comparison, inputs=inputs, plan=plan, specs=specs)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'activation.json', activation)
    write_json(output / 'runtime-inputs.json', inputs)
    for candidate_id, spec in specs.items():
        write_json(output / (candidate_id + '.json'), spec)
    write_json(output / 'comparison-request.json', comparison)
    authority = build_session_authority(activation_manifest=activation, activation_record=_record(output / 'activation.json'),
        runtime_inputs=inputs, runtime_input_record=_record(output / 'runtime-inputs.json'),
        resource_name='blueprint-native-task-policy-canary-runtime-' + uuid4().hex, hard_cap_usd=2, hard_ttl_seconds=7200,
        allowed_machine_ids=allowed_machine_ids)
    from .vast_independent_watchdog_control import validate_independent_vast_watchdog_names
    validate_independent_vast_watchdog_names(pod_name_prefix='blueprint-native-task-policy-canary-',
                                           resource_name_exact=authority['resource_name'])
    write_json(output / 'session-authority.json', authority)
    from .native_task_arena_policy_canary_worker import preflight_policy_canary_static_inputs
    preflight = preflight_policy_canary_static_inputs(inputs=inputs, authority=authority, base_scene_plan=plan,
        construction=json.loads(Path(construction_path).read_text()),
        packet_request=json.loads((root / 'native_task_arena_packet_request.v1.json').read_text()), specs=specs)
    write_json(output / 'static-preflight.json', preflight)
    return {'status': preflight['status'], 'blockers': preflight['blockers'], 'output_dir': str(output),
            'binding_digest': comparison['binding']['binding_digest'], 'provider_mutation_performed': False}
