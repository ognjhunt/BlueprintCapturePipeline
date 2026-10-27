"""Family source queue preparation retains real bytes without granting execution."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_bundle import verify_native_task_arena_packet
from blueprint_pipeline.task_evaluation_scene_attempt_evidence import record
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.website_task_family_packet import (
    family_source_references, materialize_website_task_family_packet,
)
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_task_family_preparation import _arguments_for_family, _evaluation_inputs


def family_preparation(tmp_path, *, policy_selection=None, runtime_selection=None):
    args = _arguments_for_family(tmp_path)
    args['source_geometry']['geometry_backend'] = 'fixture-native-depth'
    args['source_geometry']['digest'] = canonical_digest(args['source_geometry'], digest_field='digest')
    args['task_masks']['source_geometry_digest'] = args['source_geometry']['digest']
    args['task_masks']['digest'] = canonical_digest(args['task_masks'], digest_field='digest')
    now = time.time()
    args['now'] = now
    args['spend'] = copy.deepcopy(args['spend'])
    args['spend']['expires_at_epoch'] = now + 3600
    args['spend']['consent']['accepted_at_epoch'] = now - 1
    evaluation = _evaluation_inputs(args)
    evaluation['task_family_runtime_selection'] = runtime_selection or {
        'runtime_id': 'qualified_button_runtime', 'runtime_catalog_digest': 'sha256:' + 'b' * 64}
    if policy_selection is not None:
        evaluation['policy_selection'] = policy_selection
    evaluation['asset_sources'] = {}
    for asset in evaluation['native_runtime_inputs']['assets']:
        role = asset.get('semantic_role')
        if role in {'scene_appearance', 'scene_collision'}:
            key = 'splat_path' if role == 'scene_appearance' else 'collision_mesh_path'
            source = Path(args['base_scene'][key])
        else:
            source = tmp_path / (asset['asset_id'] + '.bin')
            source.write_bytes(b'qualified supplied button assembly')
            asset['content_digest'] = record(source)['sha256']
        asset['bundle_relative_path'] = 'assets/' + source.name
        evaluation['asset_sources'][asset['asset_id']] = record(source)
    args['evaluation_inputs'] = evaluation
    preparation = compile_website_scene_preparation(**args)
    assert preparation['status'] == 'intake_ready', preparation
    runtime_path = tmp_path / 'native_runtime_contract.json'
    write_json(runtime_path, preparation['task_evaluation_inputs']['native_runtime_contract'])
    return args, preparation, runtime_path


def test_supplied_family_packet_is_verified_by_native_worker_packet_reader(tmp_path):
    _args, preparation, runtime_path = family_preparation(tmp_path)
    result = materialize_website_task_family_packet(preparation=preparation, runtime_inputs_path=runtime_path,
        output_root=tmp_path / 'packet', intent_reference=record(runtime_path),
        attempt_reference=record(runtime_path), release_reference=record(runtime_path))
    root, receipt, _ = verify_native_task_arena_packet(result['packet_root'])
    plan = json.loads((root / 'native_task_arena_scene_plan.v1.json').read_text())
    assert plan['native_runtime_contract'] == preparation['task_evaluation_inputs']['native_runtime_contract']
    assert plan['task_kind'] == 'button_press'
    assert receipt['native_application_claimed'] is False and receipt['policy_episode_claimed'] is False
    assert receipt['scenario_instance_digest'] == plan['native_runtime_contract']['scenario_instance_digest']
    assert not (root / 'native_task_arena_policy_execution_spec.v1.json').exists()


@pytest.mark.parametrize('corruption', ['source_bytes', 'scope', 'asset_path'])
def test_family_source_registration_reopens_exact_assets_and_binding(tmp_path, corruption):
    args, preparation, runtime_path = family_preparation(tmp_path)
    if corruption == 'source_bytes':
        Path(args['base_scene']['splat_path']).write_bytes(b'changed source')
        error = 'bytes_mismatch'
    elif corruption == 'scope':
        preparation['intake_request']['task']['native_runtime_contract_digest'] = 'sha256:' + 'f' * 64
        error = 'request_binding_changed'
    else:
        native = preparation['task_evaluation_inputs']['native_runtime_contract']
        native['assets'][0]['bundle_relative_path'] = '../escape'
        native['contract_digest'] = canonical_digest(native, digest_field='contract_digest')
        preparation['intake_request']['task']['native_runtime_contract_digest'] = native['contract_digest']
        write_json(runtime_path, native)
        error = 'asset_bundle_path_invalid'
    with pytest.raises(ValueError, match=error):
        family_source_references(preparation=preparation, runtime_inputs_path=runtime_path)


def test_registered_family_progresses_to_concrete_native_packet_under_existing_hold_gates(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_progression as progression
    from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import RELEASE_SCHEMA
    from blueprint_pipeline.task_evaluation_scene_intake import _read
    from tests.test_task_evaluation_scene_configuration_submission import production_fixture, SHA

    args, preparation, runtime_path = family_preparation(tmp_path)
    # Start at the production capture handoff. Only the completed Atlas/mask
    # seams and authenticated WebApp transport are replaced with retained local
    # bytes; source registration, intake, progression and packet consumers run.
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest, cross_runtime_canonical_json
    registration_root, intent_root = tmp_path / 'bindings', tmp_path / 'intents'
    monkeypatch.setenv('BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT', str(registration_root))
    monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT', str(intent_root))
    pipeline = tmp_path / 'pipeline'
    pipeline.mkdir()
    removal_path = pipeline / 'removal.json'
    write_json(removal_path, args['removal_manifest'])
    atlas = {'status': 'ready', 'base_scene': args['base_scene'], 'source_geometry': args['source_geometry'],
        'source_pose_digest': 'sha256:' + 'b' * 64, 'background_pose_digest': 'sha256:' + 'c' * 64,
        'alignment': 'anchored'}
    monkeypatch.setattr('blueprint_pipeline.website_atlas_scene_source.atlas_scene_source', lambda **_: atlas)
    monkeypatch.setattr('blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry', lambda **kw: kw['task_masks'])
    accepted = {}
    def prepared_transport(*, capture_id, operation, payload):
        assert operation == 'prepared-scene' and capture_id == args['task_context']['capture_id']
        receipt = stage_scene_intent(value=json.loads(cross_runtime_canonical_json(payload['request'])), queue_root=intent_root,
            authenticated_client='blueprint-webapp', trusted_clients={'blueprint-webapp'}, now=args['now'])
        accepted.update(receipt)
        return {'id': receipt['intent_id'], 'state': 'forward_pending',
                'request_digest': cross_runtime_canonical_digest(payload['request'])}
    monkeypatch.setattr('blueprint_pipeline.website_task_context.website_webapp_request', prepared_transport)
    authority = {**args['spend'], 'schema_version': 'website_scene_sponsorship.v1'}
    result = prepare_website_scene_handoff(descriptor={'scene_id': args['task_context']['scene_id'],
        'capture_id': args['task_context']['capture_id'], 'metadata': {'site_task_context': args['task_context'],
            'website_scene_execution_authority': authority, 'native_task_evaluation': args['evaluation_inputs']}},
        clean_plate={'privacy_verified': True, 'status': 'objects_removed',
            'removal_manifest_path': str(removal_path), 'task_masks': args['task_masks']},
        capture_root=tmp_path, now=args['now'])
    assert result['status'] == 'intake_ready', result
    assert result['source_registration'] and result['website_intake_outbox']['id'] == accepted['intent_id']
    assert result['simulator_ready'] is False
    directory = intent_root / accepted['intent_id']
    intent = _read(directory / 'intent.json', 'intent_digest')
    from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA
    operator_registry = {'schema_version': REGISTRY_SCHEMA, 'profiles': [],
        'derived_profiles': {'root': str(tmp_path / 'derived-profiles'), 'defaults': {
            'allowed_founder_approvers': ['owner-1'], 'max_hourly_rate_usd': '1.00', 'open_loop_horizon': 1}}}
    operator_registry['registry_digest'] = canonical_digest(operator_registry, digest_field='registry_digest')
    registry_path = tmp_path / 'operator-registry.json'
    write_json(registry_path, operator_registry)
    registry_path.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENV, str(registry_path))
    fixture = production_fixture(tmp_path / 'release-fixture')
    machinery = {'schema_version': 'task_evaluation_website_scene_machinery.v1',
                 'maximum_preparation_spend_usd': 0, 'provider': 'vast'}
    machinery['machinery_digest'] = canonical_digest(machinery, digest_field='machinery_digest')
    machinery_path = tmp_path / 'machinery.json'
    write_json(machinery_path, machinery)
    release = {'schema_version': RELEASE_SCHEMA, 'source_commit': SHA,
        'runtime_digest': 'sha256:' + 'f' * 64, 'repo_root': str(tmp_path / 'repo'),
        'runtime_publication_root': str(fixture['runtime_publication_root']),
        'namespace_timestamp': '20261027T120000Z', 'release_admission_mode': 'promoted',
        **{key: record(fixture[key]) for key in ('deploy_receipt', 'release_provenance', 'release_environment')}}
    release['release_digest'] = canonical_digest(release, digest_field='release_digest')
    config = {'intent_root': str(intent_root), 'website_source_binding_root': str(registration_root),
        'website_source_machinery_path': str(machinery_path), 'factory_output_root': str(tmp_path / 'factories')}

    def advance(config_value):
        return progression._advance_intent(directory, intent, config_value, release,
            resolver=None, publisher=lambda **_: pytest.fail('No rigid publication'),
            submitter=lambda **_: pytest.fail('No rigid submission'), status_reader=None,
            activation_provisioner=lambda **_: pytest.fail('No paid activation'), now=args['now'])

    held = advance({**config, 'paused_intent_ids': [accepted['intent_id']]})
    assert held['phase'] == 'paused' and not list(directory.glob('preparation-attempts/*.json'))
    ready = advance(config)
    assert ready['status'] == 'completed' and ready['phase'] == 'family_native_prepared', ready
    packet = ready['state']['native_task_family_packet']
    assert packet['execution_authority_granted'] is False
    from blueprint_pipeline.task_evaluation_scene_intake import scene_intent_status
    public_status = scene_intent_status(queue_root=intent_root, intent_id=intent['intent_id'], now=args['now'])
    assert public_status['status'] == 'completed' and public_status['phase'] == 'family_native_prepared'
    reference = public_status['result_reference']
    assert reference['schema_version'] == 'website_task_family_preparation_result.v1'
    assert reference['digest'] == ready['state']['family_preparation_completion']['digest']
    assert reference['native_application_claimed'] is False and reference['policy_episode_claimed'] is False
    assert reference['pre_policy_site_qualification_required'] is True
    retained_result = json.loads(Path(reference['path']).read_text())
    assert retained_result['robot_evaluation_performed'] is False
    assert retained_result['execution_authority_granted'] is False
    packet_root, _, _ = verify_native_task_arena_packet(packet['packet_root'])
    plan = json.loads((packet_root / 'native_task_arena_scene_plan.v1.json').read_text())
    from blueprint_pipeline.task_family_policy_execution import task_family_policy_execution_binding
    assert task_family_policy_execution_binding(plan) == {
        'native_runtime_contract_digest': preparation['task_evaluation_inputs']['native_runtime_contract']['contract_digest'],
        'task_family_runtime_selection': args['evaluation_inputs']['task_family_runtime_selection']}
    assert plan['task_kind'] == 'button_press' and 'objects' not in plan
    assert reference['digest'] == plan['plan_digest']
    attempts = list(directory.glob('preparation-attempts/*.json'))
    assert len(attempts) == 1 and _read(attempts[0], 'attempt_digest')['maximum_spend_usd'] == 0
    assert not (directory / 'attempts').exists()
    assert advance(config)['state']['native_task_family_packet'] == packet
    from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
    from blueprint_pipeline.beta_policy_screening_family_sources import enqueue_registered_family_scenes
    profile = profile_for_capture(tmp_path)
    assert profile['family_scene_inputs_path']
    scene_inputs = json.loads(Path(profile['family_scene_inputs_path']).read_text())
    assert scene_inputs['control_diagnostic'] == {
        'mode': 'required_before_policy', 'strategy': 'native_task_family_prepolicy.v1'}
    assert scene_inputs['native_application_claimed'] is False
    # The existing beta source scanner sees this capture. A missing operator
    # runtime-source receipt is a specific, retryable preparation requirement.
    adopted = enqueue_registered_family_scenes(queue_root=tmp_path / 'screening')
    assert adopted['sources'][0]['status'] == 'blocked'
    assert adopted['sources'][0]['blockers'] == ['scene_progression_family_scene_runtime_source_required']
    write_json(directory / 'revoked.json', {'revoked': True})
    revoked = advance(config)
    # CPU preparation is retained as completed evidence. Revocation still
    # blocks every new evaluation admission from the same source.
    assert revoked['phase'] == 'family_native_prepared' and revoked['status'] == 'completed'
    denied = enqueue_registered_family_scenes(queue_root=tmp_path / 'screening')
    assert denied['sources'][0]['blockers'] == ['scene_configuration_submission_scene_owner_authority_revoked']


def test_family_scene_plan_resolves_only_explicit_native_scenario_bindings(tmp_path):
    from blueprint_pipeline.policy_canary_scene_resolution import resolved_scene_plan
    from blueprint_pipeline.task_family_policy_execution import task_family_policy_execution_binding
    _args, preparation, runtime_path = family_preparation(tmp_path)
    packet = materialize_website_task_family_packet(preparation=preparation, runtime_inputs_path=runtime_path,
        output_root=tmp_path / 'packet', intent_reference=record(runtime_path),
        attempt_reference=record(runtime_path), release_reference=record(runtime_path))
    base = json.loads(Path(packet['scene_plan']['path']).read_text())
    scenario = {'family': 'canonical', 'parameters': {}}
    task = base['task_spec']['task_success_contract']
    cell = {'cell_id': 'canonical-seed-7', 'seed': 7, 'resolved_scenario': scenario,
            'resolved_scenario_digest': canonical_digest(scenario), 'task_success_contract': task,
            'task_success_contract_digest': task['contract_digest']}
    resolved = resolved_scene_plan(base, cell)
    assert resolved['task_kind'] == 'button_press' and 'objects' not in resolved
    assert resolved['scenario']['cell_id'] == 'canonical-seed-7'
    assert resolved['native_runtime_contract']['seed'] == 7
    assert resolved['source_native_runtime_contract'] == base['native_runtime_contract']
    assert resolved['source_native_runtime_contract_digest'] == base['runtime_contract_digest']
    assert resolved['cadence'] == base['cadence']
    assert resolved['plan_digest'] == canonical_digest(resolved, digest_field='plan_digest')
    assert task_family_policy_execution_binding(resolved)['native_runtime_contract_digest'] == base['runtime_contract_digest']
    shifted = copy.deepcopy(cell)
    shifted['resolved_scenario']['parameters'] = {'object_start_y_delta_m': .02}
    shifted['resolved_scenario_digest'] = canonical_digest(shifted['resolved_scenario'])
    with pytest.raises(RuntimeError, match='task_family_scenario_parameters_not_implemented:object_start_y_delta_m'):
        resolved_scene_plan(base, shifted)
    assert base['native_runtime_contract']['seed'] == 0


def test_family_scene_plan_reopens_every_portable_asset_byte(tmp_path):
    from blueprint_pipeline.native_task_arena_scene_plan import (
        NativeTaskArenaScenePlanError, materialize_native_task_arena_scene_plan,
    )
    args, preparation, runtime_path = family_preparation(tmp_path)
    packet = materialize_website_task_family_packet(preparation=preparation, runtime_inputs_path=runtime_path,
        output_root=tmp_path / 'packet', intent_reference=record(runtime_path),
        attempt_reference=record(runtime_path), release_reference=record(runtime_path))
    native = preparation['task_evaluation_inputs']['native_runtime_contract']
    (Path(packet['packet_root']) / native['assets'][0]['bundle_relative_path']).write_bytes(b'changed asset')
    with pytest.raises(NativeTaskArenaScenePlanError, match='native_task_family_asset_digest_mismatch'):
        materialize_native_task_arena_scene_plan(runtime_contract=native,
            provider_asset_directory=packet['packet_root'], physics_frequency_hz=120,
            task_family_runtime_selection=args['evaluation_inputs']['task_family_runtime_selection'])


@pytest.mark.parametrize("corruption", ["native_application_claimed", "policy_episode_claimed",
    "pre_policy_site_qualification_required", "task_success_contract_digest", "runtime_catalog", "task_family", "phase"])
def test_family_cpu_preparation_reference_cannot_claim_native_execution(corruption):
    from blueprint_pipeline.task_family_dispatch_contracts import validate_family_preparation_result_reference
    sha = "sha256:" + "a" * 64
    request = {"task": {"strategy": "button_press", "task_success_contract_digest": sha,
        "native_runtime_contract_digest": sha, "task_family_runtime_selection": {
            "runtime_id": "registered_engine", "runtime_catalog_digest": sha}},
        "execution": {"purpose": "scene_preparation"}}
    reference = {"schema_version": "website_task_family_preparation_result.v1",
        "path": "/retained/family_native_preparation_result.v1.json", "sha256": sha, "size_bytes": 100,
        "digest": sha, "task_family": "button_press", "native_application_claimed": False,
        "policy_episode_claimed": False, "pre_policy_site_qualification_required": True,
        **{key: copy.deepcopy(request["task"][key]) for key in ("task_success_contract_digest",
            "native_runtime_contract_digest", "task_family_runtime_selection")}}
    assert validate_family_preparation_result_reference(reference, request=request,
        status="completed", phase="family_native_prepared") == reference
    phase = "family_native_prepared"
    if corruption in {"native_application_claimed", "policy_episode_claimed", "pre_policy_site_qualification_required"}:
        reference[corruption] = not reference[corruption]
    elif corruption == "runtime_catalog":
        reference["task_family_runtime_selection"]["runtime_catalog_digest"] = "sha256:" + "f" * 64
    elif corruption == "task_success_contract_digest":
        reference[corruption] = "sha256:" + "f" * 64
    elif corruption == "task_family":
        reference[corruption] = "pouring"
    else:
        phase = "scene_prepared"
    with pytest.raises(ValueError, match="family_preparation_result_reference_invalid"):
        validate_family_preparation_result_reference(reference, request=request, status="completed", phase=phase)
