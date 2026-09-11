"""Replay exact V27 native-state and vendor-action failures without Isaac or a model."""
from pathlib import Path
import json

import numpy as np
import pytest

from blueprint_pipeline.adp009d_droid_action_execution import ACTION_SPACE_JOINT_POSITION, DroidActionExecutionError, validate_candidate_action_bounds
from blueprint_pipeline.adp009d_policy_episode import NativeJointStateBoundsError, validate_native_joint_state
from blueprint_pipeline.openpi_droid_policy_runtime import normalize_openpi_inference_response
from blueprint_pipeline.policy_canary_worker_evidence import _write_episode_failure_gap
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_adp009d_policy_episode import _Environment, _Policy, _run

CASES=json.loads((Path(__file__).parent/'fixtures/pi05_v27_action_failures/cases.json').read_text())['cases']


@pytest.mark.parametrize('case',CASES,ids=lambda x:x['cell'])
def test_retained_vendor_responses_preserve_numbers_and_reproduce_exact_bounds_refusal(case):
    response=case['raw_vendor_action_response']
    assert canonical_digest({'raw_vendor_action_response':response})==case['raw_vendor_action_response_digest']
    actions=normalize_openpi_inference_response(response)
    assert actions is response['actions']
    with pytest.raises(DroidActionExecutionError) as caught:
        validate_candidate_action_bounds(np.asarray(actions)[:8],action_space=ACTION_SPACE_JOINT_POSITION,
            joint_limits=case['joint_limits_rad'],candidate_id='pi05_droid')
    assert str(caught.value)==case['failure_message']
    if case['cell']=='cell01':
        with pytest.raises(NativeJointStateBoundsError):
            validate_native_joint_state(case['policy_received_joint_positions_rad'],case['joint_limits_rad'],phase='before_policy_query')
    else:
        validate_native_joint_state(case['policy_received_joint_positions_rad'],case['joint_limits_rad'],phase='before_policy_query')


def test_first_native_violation_is_retained_and_stops_actions_before_next_policy_query(tmp_path):
    case=next(x for x in CASES if x['cell']=='cell01')
    transition=case['first_native_limit_violation_transition']
    class Environment(_Environment):
        def reset(self):
            super().reset()
            self._joints=list(transition['observed_before_rad'])
        def joint_limits(self):
            return case['joint_limits_rad']
        def step(self,action):
            super().step(action)
            self._joints=list(transition['observed_after_rad'])
    class Policy(_Policy):
        action_space = ACTION_SPACE_JOINT_POSITION
        def infer(self,observation):
            self.observations.append(observation)
            return np.tile(transition['clipped_droid_action'],(15,1))
    environment,policy,progress=Environment(),Policy(),{}
    with pytest.raises(NativeJointStateBoundsError) as caught:
        _run(environment=environment,policy=policy,progress=progress)
    assert len(policy.observations)==1 and len(environment.steps)==1
    assert progress['commanded_actions'][0]['joint_position_target_rad']==transition['joint_position_target_rad']
    assert progress['commanded_actions'][0]['observed_after_rad']==transition['observed_after_rad']
    assert progress['commanded_actions'][0]['joint_state_after_validated'] is False
    assert progress['candidate_joint_state_validated'] is False
    assert caught.value.readback['violations'][0]['joint_index']==4
    assert caught.value.readback['violations'][0]['observed_rad']==-4.004210948944092
    assert caught.value.readback['observed_state_clamped'] is False
    path=_write_episode_failure_gap(output_root=tmp_path,run_id='retained-native-violation',
        context={'cell_id':'cell01','candidate_id':'pi05_droid','seed':1},failure=caught.value,progress=progress)
    failure=json.loads(path.read_text())
    assert failure['failure_type']=='NativeJointStateBoundsError'
    assert failure['native_joint_state_violation']==caught.value.readback
    assert failure['episode']['native_joint_state_violation']==caught.value.readback
    assert failure['episode']['score']['status']=='not_scored'


def test_invalid_input_state_is_retained_before_any_policy_call():
    class Environment(_Environment):
        def read_policy_inputs(self):
            return {**super().read_policy_inputs(),'joint_position':[0.,0.,0.,0.,-4.004210948944092,0.,0.]}
    environment,policy,progress=Environment(),_Policy(),{}
    with pytest.raises(NativeJointStateBoundsError) as caught:
        _run(environment=environment,policy=policy,progress=progress)
    assert policy.observations==[] and environment.steps==[]
    assert caught.value.readback['phase']=='before_policy_query'
    assert progress['native_joint_state_violation']==caught.value.readback


def test_native_limits_remain_exact_and_do_not_clamp_even_one_ulp():
    case=CASES[0]
    valid=np.asarray(case['policy_received_joint_positions_rad'])
    valid[4]=case['joint_limits_rad'][4][0]
    validate_native_joint_state(valid,case['joint_limits_rad'],phase='after_action')
    valid[4]=np.nextafter(valid[4],-np.inf)
    original=valid.copy()
    with pytest.raises(NativeJointStateBoundsError):
        validate_native_joint_state(valid,case['joint_limits_rad'],phase='after_action')
    assert np.array_equal(original,valid)


def test_prestart_resets_remain_in_readiness_but_do_not_become_episode_retries(tmp_path):
    from blueprint_pipeline.native_rigid_episode_telemetry import NativeRigidEpisodeTelemetry
    from tests.test_adp009d_policy_episode import _LifecycleEnvironment, _LifecyclePolicy
    class Environment(_LifecycleEnvironment):
        def __init__(self):
            super().__init__()
            self.telemetry=NativeRigidEpisodeTelemetry({})
        def begin_episode(self):
            self.telemetry.begin_episode()
        def reset(self):
            super().reset()
            self.telemetry.reset_executed()
        def read_object_sample(self):
            return {**super().read_object_sample(),'retry_count':self.telemetry.retries}
    environment=Environment()
    receipt=_run(environment=environment,policy=_LifecyclePolicy(),max_policy_queries=1,settle_window_samples=1,
        media_output_dir=tmp_path,episode_id='native-episode-counter-boundary',
        require_complete_multicamera_media=True,require_prestart_readiness=True)
    assert receipt['prestart_readiness']['restored_task_sample']['retry_count']==1
    assert all(sample['retry_count']==0 for sample in receipt['state_trace']['task_state_samples'])
    assert environment.reset_count==3


def test_actual_mid_episode_reset_still_counts_as_retry():
    from blueprint_pipeline.native_rigid_episode_telemetry import NativeRigidEpisodeTelemetry
    class Environment(_Environment):
        def __init__(self):
            super().__init__()
            self.telemetry=NativeRigidEpisodeTelemetry({})
            self.reset_during_actions=False
        def begin_episode(self):
            self.telemetry.begin_episode()
        def reset(self):
            super().reset()
            self.telemetry.reset_executed()
        def step(self,action):
            super().step(action)
            if self._t==4 and not self.reset_during_actions:
                self.reset_during_actions=True
                self.reset()
        def read_object_sample(self):
            return {**super().read_object_sample(),'retry_count':self.telemetry.retries}
    environment=Environment()
    receipt=_run(environment=environment,max_policy_queries=1,settle_window_samples=1)
    counts=[sample['retry_count'] for sample in receipt['state_trace']['task_state_samples']]
    assert counts[:4]==[0,0,0,0] and all(count==1 for count in counts[4:])
    assert environment.reset_count==2


def test_terminal_settle_violation_retains_applied_state_and_prevents_scientific_completion(monkeypatch):
    from blueprint_pipeline import adp009d_policy_episode as episode
    class Environment(_Environment):
        def step(self,action):
            super().step(action)
            if len(self.steps)==9:  # Eight policy actions, then the first release/settle step.
                self._joints[4]=-4.004210948944092
    original_score=episode.score_task_episode_from_spec
    score_sizes=[]
    def forbidden_score(**kwargs):
        score_sizes.append(len(kwargs['samples']))
        if len(kwargs['samples'])==1:
            return original_score(**kwargs)  # Initial reset admission remains active.
        raise AssertionError('invalid terminal native state must never be scored')
    monkeypatch.setattr(episode,'score_task_episode_from_spec',forbidden_score)
    environment,policy,progress=Environment(),_Policy(),{}
    with pytest.raises(NativeJointStateBoundsError) as caught:
        _run(environment=environment,policy=policy,max_policy_queries=1,settle_window_samples=3,progress=progress)
    assert len(policy.observations)==1 and len(environment.steps)==9
    violation=progress['native_joint_state_violation']
    assert violation==caught.value.readback
    assert violation['phase']=='terminal_settle' and violation['step_index']==9
    assert violation['isaac_action']==environment.steps[-1]
    assert violation['environment_step_applied'] is True
    assert violation['observed_joint_positions_rad'][4]==-4.004210948944092
    assert progress['candidate_joint_state_validated'] is False
    assert score_sizes==[1]
