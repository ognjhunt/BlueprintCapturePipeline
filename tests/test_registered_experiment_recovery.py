"""Real root expiry interruption resumes only the original durable operation."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_actions.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
import json

import pytest

from tests.test_registered_experiment_retirement_flow import (
    retirement_installation, root_metadata, installation, _born_scratch,  # noqa: F401
    _issue_action, _gc, _current_entry, _payload_snapshot,
)  # noqa: F401


@pytest.mark.parametrize('boundary', ['first_removed', 'retired_receipt'])
def test_actual_timer_resumes_durable_removal_or_final_receipt(retirement_installation, monkeypatch, boundary):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_actions as code
    grant, _, target = _born_scratch(retirement_installation)
    action = _issue_action(retirement_installation, grant)
    real_event = code._event
    stopped = []
    def event(*args, **kwargs):
        selected = real_event(*args, **kwargs)
        kind = args[3]
        if not stopped and kind == ('member_removed' if boundary == 'first_removed' else 'retired'):
            stopped.append(selected)
            raise RuntimeError('development_only_interrupted_after_durable_event')
        return selected
    monkeypatch.setattr(code, '_event', event)
    pins = retirement_installation[0].parent / 'pins'
    pins.mkdir()
    with pytest.raises(RuntimeError, match='interrupted'):
        code.run_action(action['action_id'], expected_action_intent=action['action_intent'],
            installed_config_path=retirement_installation[0], now=lambda: 2900, _pins_root=pins)
    assert _current_entry(retirement_installation, grant['intent_id'])['state'] == 'retiring'
    first_events = {p.name: p.read_bytes() for p in
                    (retirement_installation[2] / 'operations' / action['action_id']).glob('e-*.json')}
    monkeypatch.setattr(code, '_event', real_event)
    outcome = _gc(retirement_installation)['registered_experiments']['outcomes'][0]
    assert outcome['decision'] == 'retired', outcome
    assert outcome['removed_logical_bytes'] == len(b'tiny disposable result') + len(b'local log\n')
    assert _current_entry(retirement_installation, grant['intent_id'])['state'] == 'retired'
    assert not (target / 'nested').exists() and not (target / 'intermediate.bin').exists()
    assert all((retirement_installation[2] / 'operations' / action['action_id'] / name).read_bytes() == raw
               for name, raw in first_events.items())


def test_resume_refuses_changed_reference_configuration_before_more_unlinks(retirement_installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_actions as code
    grant, _, target = _born_scratch(retirement_installation)
    action = _issue_action(retirement_installation, grant)
    real_event = code._event
    def event(*args, **kwargs):
        result = real_event(*args, **kwargs)
        if args[3] == 'member_removed':
            raise RuntimeError('development_only_stop')
        return result
    monkeypatch.setattr(code, '_event', event)
    pins = retirement_installation[0].parent / 'pins'
    pins.mkdir()
    with pytest.raises(RuntimeError):
        code.run_action(action['action_id'], expected_action_intent=action['action_intent'],
            installed_config_path=retirement_installation[0], now=lambda: 2900, _pins_root=pins)
    monkeypatch.setattr(code, '_event', real_event)
    before = _payload_snapshot(target)
    environment = retirement_installation[0].parent / 'gc.env'
    environment.write_text(environment.read_text() + '# changed retained configuration\n')
    outcome = _gc(retirement_installation)['registered_experiments']['outcomes'][0]
    assert outcome['decision'] == 'kept' and outcome['reason'] == 'experiment_reference_authority_changed'
    assert _payload_snapshot(target) == before


def test_resume_rejects_tampered_durable_progress_before_more_unlinks(retirement_installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_actions as code
    grant, _, target = _born_scratch(retirement_installation)
    action = _issue_action(retirement_installation, grant)
    real_event = code._event
    def event(*args, **kwargs):
        result = real_event(*args, **kwargs)
        if args[3] == 'member_removed':
            raise RuntimeError('development_only_stop')
        return result
    monkeypatch.setattr(code, '_event', event)
    pins = retirement_installation[0].parent / 'pins'
    pins.mkdir()
    with pytest.raises(RuntimeError):
        code.run_action(action['action_id'], expected_action_intent=action['action_intent'],
            installed_config_path=retirement_installation[0], now=lambda: 2900, _pins_root=pins)
    monkeypatch.setattr(code, '_event', real_event)
    event_path = retirement_installation[2] / 'operations' / action['action_id'] / 'e-00001.json'
    value = json.loads(event_path.read_bytes())
    value['body']['logical_bytes'] += 1
    event_path.write_text(json.dumps(value))
    before = _payload_snapshot(target)
    outcome = _gc(retirement_installation)['registered_experiments']['outcomes'][0]
    assert outcome['decision'] == 'kept' and outcome['reason'] == 'experiment_operation_invalid'
    assert _payload_snapshot(target) == before


def test_unlink_before_durable_parent_proof_keeps_uncertain_state_without_counting_bytes(
        retirement_installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_actions as code
    grant, _, target = _born_scratch(retirement_installation)
    action = _issue_action(retirement_installation, grant)
    real_event = code._event
    def event(*args, **kwargs):
        if args[3] == 'member_removed':
            raise RuntimeError('development_only_stop_before_record')
        return real_event(*args, **kwargs)
    monkeypatch.setattr(code, '_event', event)
    pins = retirement_installation[0].parent / 'pins'
    pins.mkdir()
    with pytest.raises(RuntimeError):
        code.run_action(action['action_id'], expected_action_intent=action['action_intent'],
            installed_config_path=retirement_installation[0], now=lambda: 2900, _pins_root=pins)
    assert not (target / 'nested' / 'log.txt').exists()
    assert not (retirement_installation[2] / 'operations' / action['action_id'] / 'e-00001.json').exists()
    before = _payload_snapshot(target)
    monkeypatch.setattr(code, '_event', real_event)
    outcome = _gc(retirement_installation)['registered_experiments']['outcomes'][0]
    assert outcome['decision'] == 'kept' and outcome['removed_logical_bytes'] == 0
    assert outcome['removed_allocated_bytes'] == 0 and outcome['receipt'] is None
    assert _payload_snapshot(target) == before
    assert _current_entry(retirement_installation, grant['intent_id'])['state'] == 'retiring'


def test_terminal_replay_rejects_modified_receipt_instead_of_reporting_freed_bytes(
        retirement_installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_actions as code
    grant, _, _ = _born_scratch(retirement_installation)
    action = _issue_action(retirement_installation, grant)
    outcome = _gc(retirement_installation)['registered_experiments']['outcomes'][0]
    assert outcome['decision'] == 'retired'
    op = retirement_installation[2] / 'operations' / action['action_id']
    event_path = sorted(op.glob('e-*.json'))[-1]
    value = json.loads(event_path.read_bytes())
    value['body']['removed_logical_bytes'] += 200000000
    event_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='experiment_operation_invalid'):
        code.run_action(action['action_id'], expected_action_intent=action['action_intent'],
            installed_config_path=retirement_installation[0], now=lambda: 2900, _pins_root=None)
