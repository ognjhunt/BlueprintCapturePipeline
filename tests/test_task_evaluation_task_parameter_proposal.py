"""Task authoring stays source-bound, budgeted and proposal-only."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_task_parameter_proposal as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_public_scene_removal_selection import _source_fixture
from blueprint_pipeline.task_evaluation_supervisor.openai_cost_authority import derive_operator_scope_attestation
from tests.test_task_evaluation_scene_configuration_submission import SHA


# Exact retained structured Sol v3 output; no source dataset bytes.
SOL_V3_CONTACT_OMISSION = {'source_proposal_digest': 'sha256:dc660e9df64f40f01137da32b4d3f58309e3a68aea51c6d3d7fa6bf81ce8c837',
 'proposal': {'assumptions': ['Native validation checks all eight transformed object corners '
                              'against the destination interior after release and settling.',
                              'Any uncontrolled loss before commanded release fails the no-drop '
                              'rule, independently of the 10 mm fall-detection threshold.',
                              'Completion requires gripper separation meeting the retreat '
                              'clearance and no residual grasp.',
                              'Missing native evidence includes the destination world transform, '
                              'verified object and tray collision geometry, robot base pose and '
                              'limits, mass, friction, compliance, force-sensor calibration, '
                              'contact-class semantics, settle criteria, and trajectory collision '
                              'data.',
                              'Because source geometry is not physical task truth and no candidate '
                              'policy was queried, deterministic/native validation must reject '
                              'this proposal if those data are unavailable or any bound is '
                              'infeasible.'],
              'confidence': 0.18,
              'rationale': 'Conservative proposal preserving the specified fixed-arm robot, 15 Hz '
                           'cadence, zero retries/regrasps, no-drop rule, and '
                           'acquisition-through-retreat sequence. The 4 mm planar tolerance is '
                           'below the smallest nominal lateral margin inferred from the supplied '
                           'envelopes, but full eight-corner containment must be checked directly '
                           'using native geometry after settling. Robot-background contact is '
                           'forbidden; the other available classes cannot be categorically '
                           'forbidden because they include grasp, support, placement, or static '
                           'destination contacts. The workspace box is a provisional evidence '
                           'envelope, not a reachability or collision-safety claim.',
              'success': {'collision_failure_minimum_force_n': 1.0,
                          'control_frequency_hz': 15,
                          'drop_minimum_fall_m': 0.01,
                          'forbidden_contact_classes': ['robot_background'],
                          'maximum_episode_seconds': 60.0,
                          'maximum_final_planar_target_error_m': 0.004,
                          'maximum_regrasps': 0,
                          'maximum_retries': 0,
                          'maximum_task_contact_force_n': 15.0,
                          'minimum_lift_m': 0.05,
                          'minimum_planar_displacement_m': 0.1,
                          'pregrasp_clearance_m': 0.08,
                          'retreat_clearance_m': 0.1,
                          'robot_workspace_position_bounds_world_m': {'maximum': [-1.841778487,
                                                                                  0.932239608,
                                                                                  0.75],
                                                                      'minimum': [-2.216778713,
                                                                                  -4.236320408,
                                                                                  0.25]}},
              'uncertainty': 'High: target location, dynamics, native geometry, reachability, '
                             'collision behavior, and force observability are not established by '
                             'the supplied evidence.'}}


def _write(p, value, digest=None):
    if digest:
        value[digest] = canonical_digest(value, digest_field=digest)
    p.write_text(json.dumps(value))
    return p


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_verified_checkout_head', lambda: SHA)
    monkeypatch.setattr(module, '_git', lambda root, *args: '')
    f = _source_fixture(tmp_path)
    task = json.loads(f['task_request'].read_text())
    task['success']['forbidden_contact_classes'] = ['robot_background']
    task['human_authority'].update(task_parameter_proposal_authorized=True,
                                  max_task_parameter_proposal_cost_usd=.25,
                                  task_parameter_confirmation_delegated_to_sdk=True)
    _write(f['task_request'], task)
    request = tmp_path/'proposer-request.json'
    module.materialize_task_parameter_request(task_request_path=f['task_request'],
        installation_receipt_path=f['installation_receipt'], publisher_intake_path=f['publisher_intake'],
        source_preparation_receipt_path=f['source_preparation'], destination_simready_path=f['destination_simready'],
        expected_source_commit=SHA, available_contact_classes=['robot_background', 'object_background'],
        output_path=request)
    now = datetime.now(timezone.utc)
    scope = _write(tmp_path/'scope.json', derive_operator_scope_attestation(provider_id='openai',
        paid_resource_class=module.RESOURCE_CLASS, project_id='project', api_key_id='key',
        operator_id='operator', exclusive_from=now-timedelta(hours=1),
        exclusive_until=module._required_scope_end(now)+timedelta(days=1)))
    admin, key = tmp_path/'admin.secret', tmp_path/'inference.secret'
    for path in (admin, key):
        path.write_text('fake-secret-must-not-be-read-by-profile')
        path.chmod(0o600)
    monkeypatch.setenv('OPENAI_API_KEY_FILE', str(key))
    profile = tmp_path/'profile.json'
    module.materialize_task_parameter_profile(expected_source_commit=SHA,
        cost_scope_attestation_path=scope, openai_admin_api_key_file=admin, openai_api_key_file=key,
        openai_project_id='project', openai_api_key_id='key', output_path=profile)
    return f, request, profile


def _proposal():
    return {'success': {'control_frequency_hz': 15, 'maximum_retries': 0, 'maximum_regrasps': 0,
        'maximum_episode_seconds': 30., 'minimum_lift_m': .05, 'pregrasp_clearance_m': .1,
        'minimum_planar_displacement_m': .1, 'maximum_final_planar_target_error_m': .03,
        'retreat_clearance_m': .06, 'drop_minimum_fall_m': .02, 'maximum_task_contact_force_n': 10.,
        'forbidden_contact_classes': ['robot_background'],
        'robot_workspace_position_bounds_world_m': {'minimum': [-1., -1., 0.], 'maximum': [1., 1., 1.]},
        'collision_failure_minimum_force_n': 2.},
        'rationale': 'Synthetic proposals for a hermetic test.', 'assumptions': [],
        'uncertainty': 'Native fit and physical properties are unverified.', 'confidence': .5}


class Gate:
    def __init__(self, events):
        self.events = events

    def reserve(self):
        self.events.append('official_reservation')
        return {'status': 'reserved'}

    def complete(self, **kwargs):
        self.events.append(('official_completion', kwargs))
        return {'status': 'pending_official_settlement'}


class Invoker:
    def __init__(self, events, output=None, cost=.1, failure=None):
        self.events, self.output, self.cost, self.failure = events, output, cost, failure

    def configure_reservation_audit(self, **kwargs):
        self.events.append('audit_configured')

    def invoke(self, spec, text):
        self.events.append('model_call')
        assert spec.model == 'gpt-5.6-sol'
        assert spec.max_turns == 1 and not spec.tool_bindings
        assert spec.reasoning_effort == 'high'
        payload = json.loads(text)
        assert 'source_file' not in text and 'secret' not in text
        assert payload['robot_preset']['robot_preset_id'] == 'droid_franka_panda_robotiq_2f85_v1'
        if self.failure:
            raise self.failure
        return SimpleNamespace(output=self.output or _proposal(), model=module.MODEL, provider='openai',
            usage={'input_tokens': 100, 'output_tokens': 200}, cost_usd=self.cost, cost_status='estimated',
            sdk_version='test', latency_seconds=.1)


def _execute(inputs, tmp_path, events, **kwargs):
    def factory(**options):
        assert options['paid_resource_class'] == module.RESOURCE_CLASS
        assert options['max_cost_usd'] == .25
        return Gate(events)
    return module.execute_task_parameter_proposal(request_path=inputs[1], profile_path=inputs[2],
        output_root=tmp_path/'execution', invoker=Invoker(events, **kwargs), cost_gate_factory=factory)


def _required_contacts_request(inputs, tmp_path, required, *, path_evidence=None):
    fixture = inputs[0]
    task = json.loads(fixture['task_request'].read_text())
    task['success']['forbidden_contact_classes'] = required
    if path_evidence is not None:
        task['task_path_evidence'] = path_evidence
    _write(fixture['task_request'], task)
    request = tmp_path/'required-contacts-request.json'
    module.materialize_task_parameter_request(task_request_path=fixture['task_request'],
        installation_receipt_path=fixture['installation_receipt'], publisher_intake_path=fixture['publisher_intake'],
        source_preparation_receipt_path=fixture['source_preparation'], destination_simready_path=fixture['destination_simready'],
        expected_source_commit=SHA, available_contact_classes=list(module.CONTACT_CHANNELS), output_path=request)
    return fixture, request, inputs[2]


def test_required_contact_payload_binds_filtered_semantics_and_configured_grasp_path(inputs, tmp_path):
    evidence = {'status': 'proposed_not_native_qualified',
                'reset_grasp_frame_position_world_m': [-2., 0., .8],
                'trajectory_waypoints_world_m': [[-2., 0., .4], [-2., .2, .5]]}
    bound = _required_contacts_request(inputs, tmp_path, list(module.CONTACT_CHANNELS), path_evidence=evidence)
    payload = json.loads(bound[1].read_text())['payload']
    assert payload['required_forbidden_contact_classes'] == list(module.CONTACT_CHANNELS)
    assert payload['configured_task_path_evidence'] == evidence
    assert 'grasp midpoint' in payload['robot_workspace_measurement_semantics']
    assert 'excluding admitted gripper' in payload['contact_class_semantics']['robot_object']['meaning']
    assert 'excluding its exact qualified placement support' in payload['contact_class_semantics']['destination_background']['meaning']


def test_actual_sol_v3_cannot_remove_three_configured_contact_requirements(inputs, tmp_path):
    bound = _required_contacts_request(inputs, tmp_path, list(module.CONTACT_CHANNELS))
    actual = SOL_V3_CONTACT_OMISSION
    assert actual['source_proposal_digest'] == 'sha256:dc660e9df64f40f01137da32b4d3f58309e3a68aea51c6d3d7fa6bf81ce8c837'
    events = []
    with pytest.raises(module.TaskParameterProposalError, match='required_contact_classes_omitted'):
        _execute(bound, tmp_path, events, output=actual['proposal'])
    assert events.count('model_call') == 1
    assert events[-1][0] == 'official_completion'
    assert json.loads((tmp_path/'execution/returned_proposal.json').read_text())['output'] == actual['proposal']
    assert not (tmp_path/'execution/task_evaluation_task_parameter_proposal.v1.json').exists()


def test_successor_revalidates_required_contacts_even_for_resigned_proposal(inputs, tmp_path):
    bound = _required_contacts_request(inputs, tmp_path, list(module.CONTACT_CHANNELS))
    valid = _proposal()
    valid['success']['forbidden_contact_classes'] = list(module.CONTACT_CHANNELS)
    _execute(bound, tmp_path, [], output=valid)
    path = tmp_path/'execution/task_evaluation_task_parameter_proposal.v1.json'
    result = json.loads(path.read_text())
    result['success']['forbidden_contact_classes'] = ['robot_background']
    result['proposal']['success']['forbidden_contact_classes'] = ['robot_background']
    _write(path, result, 'proposal_digest')
    with pytest.raises(module.TaskParameterProposalError, match='required_contact_classes_omitted'):
        module.materialize_task_parameter_successor(proposal_path=path,
            task_request_path=bound[0]['task_request'], output_path=tmp_path/'rejected-successor.json')
    assert not (tmp_path/'rejected-successor.json').exists()


@pytest.mark.parametrize('required', [None, [], ['not-a-sensor'], ['robot_background', 'robot_background']])
def test_missing_or_unavailable_owner_contacts_fail_before_request_creation(inputs, tmp_path, required):
    with pytest.raises(module.TaskParameterProposalError, match='required_contact_classes_missing_or_unavailable'):
        _required_contacts_request(inputs, tmp_path, required)
    assert not (tmp_path/'required-contacts-request.json').exists()


def test_missing_sdk_environment_permission_fails_before_reservation_or_output(inputs, tmp_path, monkeypatch):
    monkeypatch.delenv(module.LIVE_AGENTS_SDK_ENV, raising=False)
    events = []
    with pytest.raises(module.TaskParameterProposalError, match='live_sdk_environment_not_authorized'):
        module.execute_task_parameter_proposal(request_path=inputs[1], profile_path=inputs[2],
            output_root=tmp_path/'no-invocation', cost_gate_factory=lambda **kwargs: events.append(kwargs))
    assert events == []
    assert not (tmp_path/'no-invocation').exists()


def test_source_bound_sdk_proposal_retains_cost_and_never_confirms(inputs, tmp_path):
    events = []
    result = _execute(inputs, tmp_path, events)
    assert events[:3] == ['official_reservation', 'audit_configured', 'model_call']
    assert result['status'] == 'proposal_only'
    assert all(value is False for value in result['claim_boundary'].values())
    assert result['proposal_digest'] == canonical_digest(result, digest_field='proposal_digest')
    assert result['official_cost_completion']['status'] == 'pending_official_settlement'
    assert result['official_posted_cost_confirmed'] is False
    assert (tmp_path/'execution/returned_proposal.json').exists()
    with pytest.raises(module.TaskParameterProposalError, match='retry_forbidden'):
        _execute(inputs, tmp_path, events)
    assert events.count('model_call') == 1


@pytest.mark.parametrize('kind', ['score', 'authority', 'model_change', 'nan', 'unknown_contact', 'retry'])
def test_invalid_model_output_is_retained_and_billing_closed(inputs, tmp_path, kind):
    proposal = _proposal()
    if kind in ('score', 'authority', 'model_change'):
        proposal[kind] = 'forbidden'
    elif kind == 'nan':
        proposal['success']['minimum_lift_m'] = float('nan')
    elif kind == 'unknown_contact':
        proposal['success']['forbidden_contact_classes'] = ['not-a-sensor-channel']
    else:
        proposal['success']['maximum_retries'] = 1
    events = []
    with pytest.raises(ValueError):
        _execute(inputs, tmp_path, events, output=proposal)
    assert events.count('model_call') == 1
    assert events[-1][0] == 'official_completion'
    assert events[-1][1]['runtime_exception_type']
    assert (tmp_path/'execution/failure.json').exists()
    assert not (tmp_path/'execution/task_evaluation_task_parameter_proposal.v1.json').exists()


@pytest.mark.parametrize('cost', [.26, None])
def test_unknown_or_excess_cost_cannot_be_accepted(inputs, tmp_path, cost):
    events = []
    with pytest.raises(module.TaskParameterProposalError, match='cost_unknown_or_exceeded'):
        _execute(inputs, tmp_path, events, cost=cost)
    assert events[-1][0] == 'official_completion'


def test_sdk_failure_retains_no_retry_and_closes_official_cost(inputs, tmp_path):
    events = []
    with pytest.raises(RuntimeError):
        _execute(inputs, tmp_path, events, failure=RuntimeError('synthetic'))
    assert events.count('model_call') == 1
    assert events[-1][0] == 'official_completion'


@pytest.mark.parametrize('change', ['task', 'source', 'profile'])
def test_changed_input_bytes_fail_before_cost_reservation(inputs, tmp_path, change):
    fixture, _, profile = inputs
    path = {'task': fixture['task_request'], 'source': fixture['source_preparation'], 'profile': profile}[change]
    path.write_text('{}')
    events = []
    with pytest.raises(ValueError):
        _execute(inputs, tmp_path, events)
    assert events == []


def test_model_cannot_replace_the_cad_or_task_actor(inputs, tmp_path):
    f, _, _ = inputs
    task = json.loads(f['task_request'].read_text())
    del task['human_authority']['task_parameter_proposal_authorized']
    _write(f['task_request'], task)
    with pytest.raises(module.TaskParameterProposalError, match='delegated_authoring_scope_missing'):
        module.materialize_task_parameter_request(task_request_path=f['task_request'],
            installation_receipt_path=f['installation_receipt'], publisher_intake_path=f['publisher_intake'],
            source_preparation_receipt_path=f['source_preparation'], destination_simready_path=f['destination_simready'],
            expected_source_commit=SHA, available_contact_classes=['robot_background'],
            output_path=tmp_path/'rejected-request.json')


@pytest.mark.parametrize('field,value', [('model', 'gpt-5.6-terra'), ('maximum_cost_usd', .5),
                                       ('automatic_retries', 1), ('automatic_retries', False)])
def test_resigned_profile_cannot_expand_invocation_scope(inputs, tmp_path, field, value):
    p = inputs[2]
    v = json.loads(p.read_text())
    v[field] = value
    _write(p, v, 'profile_digest')
    events = []
    with pytest.raises(module.TaskParameterProposalError, match='profile_invalid'):
        _execute(inputs, tmp_path, events)
    assert events == []


def test_cli_preserves_request_and_refuses_overwrite(inputs, tmp_path, capsys):
    f, _, _ = inputs
    out = tmp_path/'cli-request.json'
    argv = ['request', '--task-request', str(f['task_request']), '--installation-receipt',
            str(f['installation_receipt']), '--publisher-intake', str(f['publisher_intake']),
            '--source-preparation-receipt', str(f['source_preparation']), '--destination-simready',
            str(f['destination_simready']), '--expected-source-commit', SHA,
            '--contact-class', 'robot_background', '--output', str(out)]
    assert module.main(argv) == 0
    original = out.read_bytes()
    capsys.readouterr()
    assert module.main(argv) == 2
    assert out.read_bytes() == original
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'


def test_live_entrypoint_rejects_execution_commit_drift_before_cost_gate(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_verified_checkout_head', lambda: 'f' * 40)
    with pytest.raises(module.TaskParameterProposalError, match='execution_commit_mismatch'):
        module.execute_task_parameter_proposal(request_path=inputs[1], profile_path=inputs[2],
            output_root=tmp_path/'live-must-not-start')
    assert not (tmp_path/'live-must-not-start').exists()


def test_successor_preserves_natural_task_and_records_actual_sdk_proposal(inputs, tmp_path):
    original = json.loads(inputs[0]['task_request'].read_text())
    result = _execute(inputs, tmp_path, [])
    proposal = tmp_path/'execution/task_evaluation_task_parameter_proposal.v1.json'
    successor = module.materialize_task_parameter_successor(proposal_path=proposal,
        task_request_path=inputs[0]['task_request'], output_path=tmp_path/'successor.json')
    for key in original:
        if key != 'success':
            assert successor[key] == original[key]
    assert successor['success'] == result['success']
    authority = successor['success_contract_authority']
    assert authority['author_source'] == 'agent_proposal'
    assert authority['author_id'] == 'openai_agents_sdk:gpt-5.6-sol'
    assert authority['proposal_digest'] == result['proposal_digest']
    assert authority['confirmed_by_team_id'] == original['team_namespace']
    assert authority['agent_proposal'] == result
    assert successor['task_parameter_provenance']['native_qualification_granted'] is False


def test_unwired_numeric_fields_are_rejected_instead_of_silently_discarded(inputs, tmp_path):
    proposed = _proposal()
    proposed['success']['settle_window_seconds'] = 2.
    with pytest.raises(ValueError):
        _execute(inputs, tmp_path, [], output=proposed)


def test_successor_cannot_confirm_without_retained_delegation(inputs, tmp_path):
    task = json.loads(inputs[0]['task_request'].read_text())
    task['human_authority'].pop('task_parameter_confirmation_delegated_to_sdk')
    _write(inputs[0]['task_request'], task)
    f = inputs[0]
    request = tmp_path/'unconfirmed-request.json'
    module.materialize_task_parameter_request(task_request_path=f['task_request'],
        installation_receipt_path=f['installation_receipt'], publisher_intake_path=f['publisher_intake'],
        source_preparation_receipt_path=f['source_preparation'], destination_simready_path=f['destination_simready'],
        expected_source_commit=SHA, available_contact_classes=['robot_background', 'object_background'],
        output_path=request)
    _execute((f, request, inputs[2]), tmp_path, [])
    with pytest.raises(module.TaskParameterProposalError, match='successor_delegation_missing'):
        module.materialize_task_parameter_successor(
            proposal_path=tmp_path/'execution/task_evaluation_task_parameter_proposal.v1.json',
            task_request_path=f['task_request'], output_path=tmp_path/'must-not-exist.json')
    assert not (tmp_path/'must-not-exist.json').exists()


def test_successor_cli_and_submission_retain_every_supported_sdk_field(inputs, tmp_path, capsys):
    from tests.test_task_evaluation_scene_configuration_submission import _materialize
    result = _execute(inputs, tmp_path, [])
    successor_path = tmp_path/'cli-successor.json'
    assert module.main(['accept', '--proposal', str(tmp_path/'execution/task_evaluation_task_parameter_proposal.v1.json'),
        '--task-request', str(inputs[0]['task_request']), '--output', str(successor_path)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'successor_materialized_under_delegation'
    f = inputs[0]
    _materialize(f, task_request_path=successor_path)
    template = json.loads((f['staging_root']/'configuration/task_template.v1.json').read_text())
    execution = json.loads((f['staging_root']/'configuration/task_execution_spec.v1.json').read_text())
    for field, value in result['success'].items():
        if field in template['success']:
            assert template['success'][field] == value
        elif field in template:
            assert template[field] == value
        elif field in template['interaction_affordance']:
            assert template['interaction_affordance'][field] == value
        else:
            assert execution[field] == value
    authority = template['owner_success_contract_authority']
    assert authority['agent_proposal'] == result
    assert authority['proposal_digest'] == result['proposal_digest']
    assert authority['confirmed_by_team_id'] == json.loads(successor_path.read_text())['team_namespace']


@pytest.mark.parametrize('defect', ['commit', 'dirty'])
def test_injected_invoker_cannot_bypass_actual_execution_identity(inputs, tmp_path, monkeypatch, defect):
    if defect == 'commit':
        monkeypatch.setattr(module, '_verified_checkout_head', lambda: 'e' * 40)
    else:
        monkeypatch.setattr(module, '_git', lambda root, *args: ' M changed-runtime.py')
    events = []
    with pytest.raises(module.TaskParameterProposalError, match='execution_'):
        _execute(inputs, tmp_path, events)
    assert events == []
    assert not (tmp_path/'execution').exists()


def test_profile_cli_reopens_scope_without_reading_secrets(inputs, tmp_path, monkeypatch, capsys):
    profile = json.loads(inputs[2].read_text())
    secret_paths = {Path(profile[key]) for key in ('openai_api_key_file', 'openai_admin_api_key_file')}
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path not in secret_paths, 'profile materializer read a secret'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    out = tmp_path/'cli-profile.json'
    argv = ['profile', '--expected-source-commit', SHA,
        '--cost-scope-attestation', profile['cost_scope_attestation_path'],
        '--openai-admin-api-key-file', profile['openai_admin_api_key_file'],
        '--openai-api-key-file', profile['openai_api_key_file'],
        '--openai-project-id', 'project', '--openai-api-key-id', 'key', '--output', str(out)]
    assert module.main(argv) == 0
    result = json.loads(out.read_text())
    assert result['raw_secret_values_read'] is False
    assert result['execution_identity']['identity_source'] == 'actual_running_checkout_git_readback'
    assert result['execution_identity']['source_commit'] == SHA
    assert result['paid_execution_started'] is False
    original = out.read_bytes()
    capsys.readouterr()
    assert module.main(argv) == 2
    assert out.read_bytes() == original
    assert json.loads(capsys.readouterr().out)['status'] == 'blocked'


@pytest.mark.parametrize('change', ['expired', 'class', 'secret_mode', 'scope_bytes', 'key_binding'])
def test_profile_scope_and_secret_binding_fail_before_invocation(inputs, tmp_path, monkeypatch, change):
    profile = json.loads(inputs[2].read_text())
    scope_path = Path(profile['cost_scope_attestation_path'])
    scope = json.loads(scope_path.read_text())
    if change == 'expired':
        scope['exclusive_from'] = '2020-01-01T00:00:00+00:00'
        scope['exclusive_until'] = '2020-01-02T00:00:00+00:00'
    elif change == 'class':
        scope['paid_resource_class'] = 'some-other-lane'
    elif change == 'secret_mode':
        Path(profile['openai_api_key_file']).chmod(0o666)
    elif change == 'key_binding':
        monkeypatch.setenv('OPENAI_API_KEY_FILE', str(tmp_path/'wrong.secret'))
    else:
        scope_path.write_text('{}')
    if change in ('expired', 'class'):
        _write(scope_path, scope, 'scope_attestation_digest')
        profile['cost_scope_attestation_reference'] = module._record(scope_path)
        profile['scope_attestation_digest'] = scope['scope_attestation_digest']
        _write(inputs[2], profile, 'profile_digest')
    events = []
    with pytest.raises(ValueError):
        _execute(inputs, tmp_path, events)
    assert events == []


def test_execution_receipt_retains_observed_checkout_identity(inputs, tmp_path):
    result = _execute(inputs, tmp_path, [])
    assert result['execution_identity']['source_commit'] == SHA
    assert result['execution_identity']['checkout_clean'] is True
    started = json.loads((tmp_path/'execution/invocation_started.json').read_text())
    assert started['execution_identity'] == result['execution_identity']


@pytest.mark.parametrize('hour', [10, 23])
@pytest.mark.parametrize('short', [True, False])
@pytest.mark.parametrize('stage', ['profile', 'execution'])
def test_scope_must_cover_official_attribution_before_creating_output(
        inputs, tmp_path, monkeypatch, hour, short, stage):
    moment = datetime(2026, 9, 5, hour, 30, tzinfo=timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment.astimezone(tz)

    monkeypatch.setattr(module, 'datetime', Clock)
    # The authority's one-hour window must be covered through the following
    # UTC midnight, including when that hour itself crosses midnight.
    required_end = datetime(2026, 9, 6 if hour == 10 else 7, tzinfo=timezone.utc)
    profile = json.loads(inputs[2].read_text())
    scope_path = Path(profile['cost_scope_attestation_path'])
    scope = derive_operator_scope_attestation(provider_id='openai',
        paid_resource_class=module.RESOURCE_CLASS, project_id='project', api_key_id='key',
        operator_id='operator', exclusive_from=moment-timedelta(hours=1),
        exclusive_until=required_end-timedelta(seconds=1) if short else required_end)
    _write(scope_path, scope)
    profile['cost_scope_attestation_reference'] = module._record(scope_path)
    profile['scope_attestation_digest'] = scope['scope_attestation_digest']
    _write(inputs[2], profile, 'profile_digest')
    output = tmp_path/('new-profile.json' if stage == 'profile' else 'execution')
    events = []

    def run():
        if stage == 'execution':
            return _execute(inputs, tmp_path, events)
        return module.materialize_task_parameter_profile(expected_source_commit=SHA,
            cost_scope_attestation_path=scope_path,
            openai_admin_api_key_file=profile['openai_admin_api_key_file'],
            openai_api_key_file=profile['openai_api_key_file'],
            openai_project_id='project', openai_api_key_id='key', output_path=output)

    if short:
        with pytest.raises(module.TaskParameterProposalError,
                           match='cost_scope_attribution_window_insufficient'):
            run()
        assert not output.exists()
        assert events == []
    else:
        run()
        assert output.exists()
        if stage == 'execution':
            assert events.count('official_reservation') == events.count('model_call') == 1
