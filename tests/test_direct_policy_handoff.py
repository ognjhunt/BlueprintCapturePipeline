"""Website policy admission uses compiled scene evidence, not scripted success."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_control_omission_handoff import prepared
from tests import test_task_evaluation_policy_canary_handoff as rehearsal
from tests.test_task_evaluation_launch_activation_contract import request, ref
from blueprint_pipeline.task_evaluation_launch_activation_contract import validate_launch_activation_request


def source_closeout(tmp_path):
    from tests.test_scene_preparation_completion import _source
    root, _ = _source(tmp_path / 'source-closure')
    old = root / 'scene-839873-qualifying'
    target = tmp_path / 'launch-runs' / rehearsal.SOURCE_LAUNCH_ID
    profile = json.loads((target / 'launch_profile.json').read_bytes())
    receipt = json.loads((old / 'launch_receipt.json').read_bytes())
    receipt.update(launch_id=rehearsal.SOURCE_LAUNCH_ID, launch_profile_digest=profile['profile_digest'])
    receipt['receipt_digest'] = canonical_digest(receipt, digest_field='receipt_digest')
    (target / 'launch_receipt.json').write_text(json.dumps(receipt))
    for name, digest in [('webapp_sync_succeeded.json', 'sync_result_digest'),
                         ('post_teardown_provider_zero_receipt.json', 'provider_zero_receipt_digest')]:
        row = json.loads((old / name).read_bytes())
        for key in ('launch_id', 'launch_profile_digest', 'receipt_digest'):
            if key in row:
                row[key] = receipt[key]
        if 'response' in row:
            row['response'].update(launch_id=receipt['launch_id'], receipt_digest=receipt['receipt_digest'])
        row[digest] = canonical_digest(row, digest_field=digest)
        (target / name).write_text(json.dumps(row))


def direct_case(tmp_path, monkeypatch):
    state, directory, directive, contract = prepared(tmp_path, monkeypatch)
    directive.unlink()  # Platform default, not a hand-created per-site opt-in.
    (state / 'construction_launch_progression.json').unlink()
    source_closeout(tmp_path)
    plan_path = tmp_path / 'compiled-episodes' / rehearsal.PREPARATION_ID / 'native-task-packet/native_task_arena_scene_plan.v1.json'
    plan = json.loads(plan_path.read_bytes())
    from tests.test_camera_start_construction_materializer import inputs
    plan['robot']['joint_reset_positions_rad'] = inputs()['source_binding']['joint_reset_positions_rad']
    plan['plan_digest'] = canonical_digest(plan, digest_field='plan_digest')
    plan_path.write_text(json.dumps(plan))
    packet_path = plan_path.with_name('native_task_arena_packet_receipt.v1.json')
    packet = json.loads(packet_path.read_bytes())
    packet['arena_scene_plan_digest'] = plan['plan_digest']
    packet_path.write_text(json.dumps(packet))
    return state, directory, {'kind': 'initial_project', 'project_spend_reconciliation': ref(21), 'initial_provider_zero': ref(22)}


def test_direct_handoff_submits_policy_without_scripted_run(tmp_path, monkeypatch):
    state, directory, lineage = direct_case(tmp_path, monkeypatch)
    before = (directory / 'intent.json').read_bytes()
    webapp, publisher = rehearsal._WebApp(), rehearsal._Publisher()
    result = rehearsal._advance(tmp_path, state=state, webapp=webapp, publisher=publisher,
        profile_calls=[], diagnostic_initial_lineage=lineage)
    assert result['status'] == 'canary_launch_submitted'
    assert len(webapp.calls) == 1
    params = json.loads((state / 'policy-canary-inputs/presubmission_parameters.json').read_bytes())
    assert params['activation_lineage']['kind'] == 'initial_project'
    construction = json.loads(publisher.published[params['activation_lineage']['construction_result']['uri']])
    assert construction['schema_version'] == 'task_evaluation_episode_compilation_result.v1'
    assert 'construction_gate_qualified' not in construction
    binding = params['diagnostic_control_omission_authority']['policy_canary_camera_start_configuration']
    assert binding['native_qualification_claimed'] is False
    assert 'construction_result_digest' not in binding
    assert (directory / 'intent.json').read_bytes() == before
    assert rehearsal._advance(tmp_path, state=state, webapp=webapp, publisher=publisher,
        profile_calls=[], diagnostic_initial_lineage=lineage)['status'] == 'canary_launch_submitted'
    assert len(webapp.calls) == 1


def test_website_owner_authorizes_direct_policy_without_legacy_scene_registration(tmp_path, monkeypatch):
    state, directory, lineage = direct_case(tmp_path, monkeypatch)
    for path in (tmp_path / 'activation-intents').glob('*.json'):
        path.unlink()
    webapp = rehearsal._WebApp()
    result = rehearsal._advance(tmp_path, state=state, webapp=webapp,
        publisher=rehearsal._Publisher(), profile_calls=[], diagnostic_initial_lineage=lineage)
    assert result['status'] == 'canary_launch_submitted'
    owner = json.loads((directory / 'intent.json').read_bytes())
    params = json.loads((state / 'policy-canary-inputs/presubmission_parameters.json').read_bytes())
    authority = params['activation_authorization']
    assert authority['reference'] == 'scene-intent:' + owner['intent_digest']
    assert authority['authorized_by'] == owner['request']['owner']['user_id']
    assert 300 <= authority['valid_for_seconds'] <= 86400
    (directory / 'revoked.json').write_text('{}')
    with pytest.raises(ValueError, match='revoked'):
        rehearsal._advance(tmp_path, state=state, webapp=webapp,
            publisher=rehearsal._Publisher(), profile_calls=[], diagnostic_initial_lineage=lineage)
    assert len(webapp.calls) == 1


@pytest.mark.parametrize('missing', ['launch_receipt.json', 'webapp_sync_succeeded.json', 'post_teardown_provider_zero_receipt.json'])
def test_direct_handoff_still_requires_scene_preparation_closeout(tmp_path, monkeypatch, missing):
    state, _, lineage = direct_case(tmp_path, monkeypatch)
    (tmp_path / 'launch-runs' / rehearsal.SOURCE_LAUNCH_ID / missing).unlink()
    webapp = rehearsal._WebApp()
    with pytest.raises((ValueError, RuntimeError)):
        rehearsal._advance(tmp_path, state=state, webapp=webapp, publisher=rehearsal._Publisher(),
            profile_calls=[], diagnostic_initial_lineage=lineage)
    assert not webapp.calls


def test_initial_policy_lineage_is_diagnostic_only_and_budget_bound():
    value = request(lane='native_task_arena_policy_evaluation')
    value.update(run_kind='internal_policy_canary', capture_session_id='source-scene', intake_id='configured-task')
    value['lineage'] = {'kind': 'initial_project', 'project_spend_reconciliation': ref(21),
                        'initial_provider_zero': ref(22), 'construction_result': ref(23)}
    assert validate_launch_activation_request(value) == value
    for key in ('run_kind',):
        bad = deepcopy(value)
        bad.pop(key)
        with pytest.raises(ValueError):
            validate_launch_activation_request(bad)
    for key in ('project_spend_reconciliation', 'initial_provider_zero', 'construction_result'):
        bad = deepcopy(value)
        bad['lineage'].pop(key)
        with pytest.raises(ValueError):
            validate_launch_activation_request(bad)


@pytest.mark.parametrize('defect', [None, 'target', 'marker', 'support', 'bounds', 'tilt'])
def test_strict_task_accepts_exact_circular_target_without_requiring_container_or_yaw(tmp_path, defect):
    from tests.test_task_evaluation_policy_canary_scene_setup import _strict_owner_setup_inputs
    from tests.test_task_evaluation_surface_target import target_fixture
    from blueprint_pipeline.task_evaluation_surface_target import marker_for_surface_target
    from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import _require_strict_owner_success_contract
    kwargs, contract = _strict_owner_setup_inputs(tmp_path)
    spec = json.loads(Path(kwargs['scene_plan_path']).read_bytes())['task_spec']
    surface = target_fixture()
    spec.update(surface_target=surface, visible_target_marker=marker_for_surface_target(surface))
    spec['configured_success_criteria']['surface_target'] = deepcopy(surface)
    spec['interaction_affordance']['intended_support_prim_paths'] = [surface['support_prim_path']]
    for field in ('destination_relation', 'destination_support_asset_id', 'destination_interior_bounds_body_frame_m',
                  'destination_position_bounds_destination_frame_m'):
        spec.pop(field, None)
    contract['criteria']['surface_target'] = deepcopy(surface)
    contract['criteria']['orientation']['mode'] = 'ignored'
    if defect == 'target':
        contract['criteria']['surface_target']['radius_m'] *= 2
    elif defect == 'marker':
        spec['visible_target_marker'] = {}
    elif defect == 'support':
        spec['interaction_affordance']['intended_support_prim_paths'] = ['/Other']
    elif defect == 'bounds':
        spec['subject_collision_bounds_scoring_frame_m'] = {}
    elif defect == 'tilt':
        spec['surface_target']['maximum_tilt_rad'] *= 2
    if defect:
        with pytest.raises(ValueError):
            _require_strict_owner_success_contract(task_spec=spec, contract=contract)
    else:
        _require_strict_owner_success_contract(task_spec=spec, contract=contract)


def test_pause_binds_selected_team_request_not_robot_free_site_intent(tmp_path, monkeypatch):
    from tests.test_team_evaluation_authority import _case, _evaluation
    from blueprint_pipeline.task_evaluation_scene_control_omission import load_for_run
    root, source, profile, now = _case(tmp_path, monkeypatch)
    authority, selected = _evaluation(root, source, profile, now)
    launch = tmp_path / 'launch-runs/source-launch'
    launch.mkdir(parents=True)
    (launch / 'launch_profile.json').write_text(json.dumps(profile))
    before = {p: p.read_bytes() for p in root.glob('scene-*/intent.json')}
    directive = load_for_run(launch_state_root=launch.parent, source_launch_id='source-launch',
        now=now, evaluation_authority=authority)
    assert directive['intent_digest'] == selected['intent_digest']
    assert directive['intent_digest'] != profile['scene_intent_digest']
    assert directive['policy_candidates']
    assert all(p.read_bytes() == raw for p, raw in before.items())
    (root / selected['intent_id'] / 'revoked.json').write_text('{}')
    with pytest.raises(ValueError, match='owner_revoked'):
        load_for_run(launch_state_root=launch.parent, source_launch_id='source-launch',
            now=now, evaluation_authority=authority)
