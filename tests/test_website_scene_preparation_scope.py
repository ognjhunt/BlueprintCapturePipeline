"""Site preparation owns scene construction; a robot team owns later evaluation."""
import pytest

from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.task_evaluation_scene_policy_capability import policy_capability_blockers
from tests.test_task_evaluation_scene_intake import request
from tests.test_task_evaluation_controls_autoprovision import setup
from blueprint_pipeline import task_evaluation_controls_autoprovision as controls


def preparation_request():
    value = request()
    value['execution'].update(purpose='scene_preparation', policy_candidates=[])
    return value


def test_scene_preparation_needs_no_robot_or_policy():
    value = preparation_request()
    assert intake.validate_request(value, now=100) == value
    assert policy_capability_blockers(value) == []


@pytest.mark.parametrize('change', ['policies', 'robot', 'unknown_purpose'])
def test_scene_preparation_cannot_sneak_in_robot_evaluation(change):
    value = preparation_request()
    if change == 'policies':
        value['execution']['policy_candidates'] = request()['execution']['policy_candidates']
    elif change == 'robot':
        value['task']['robot_binding_id'] = 'franka-droid'
    else:
        value['execution']['purpose'] = 'anything'
    with pytest.raises(intake.SceneIntakeError):
        intake.validate_request(value, now=100)


def test_evaluation_still_requires_its_explicit_candidates():
    value = request()
    value['execution']['policy_candidates'] = []
    with pytest.raises(intake.SceneIntakeError, match='two_policies_required'):
        intake.validate_request(value, now=100)
    assert policy_capability_blockers(value)


def test_preparation_never_reserves_or_installs_robot_controls(tmp_path):
    kwargs = setup(tmp_path, robot_binding_id=None, purpose='scene_preparation')
    def forbidden(**kw):
        pytest.fail('site preparation must not provision robot controls')
    kwargs.update(provisioner=forbidden, installer=forbidden)
    result = controls.provision_link(**kwargs)
    assert result['status'] == 'scene_preparation_only'
    assert not list((kwargs['link_path'].parent / 'attempts').glob('*.json'))
    assert not kwargs['controls_root'].exists()


def test_preparation_worker_does_not_need_a_robot_catalog(tmp_path, monkeypatch):
    import json
    kwargs = setup(tmp_path, robot_binding_id=None, purpose='scene_preparation')
    config = tmp_path / 'controls.json'
    config.write_text(json.dumps({
        'scene_root': str(kwargs['scene_root']), 'trusted_clients': ['webapp'],
        'preparation_queue_root': str(kwargs['preparation_queue_root']),
    }))
    monkeypatch.setattr(controls, '_registered_terminal_adoption', lambda **kw: pytest.fail('no robot controls'))
    result = controls.process_config(config, expected_production_commit=kwargs['expected_production_commit'])
    assert len(result) == 1 and result[0]['status'] == 'scene_preparation_only'
