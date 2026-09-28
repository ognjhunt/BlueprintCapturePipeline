"""The CPU rehearsal cannot reserve paid placement or report full completion."""
import pytest
import json

from scripts import rehearse_selected_evaluation_cpu as replay


@pytest.fixture(autouse=True)
def local_preflight(tmp_path, monkeypatch):
    intent=tmp_path/'intent.json'
    intent.write_text(json.dumps({'placement':{'official_cost_authority':{}}}))
    monkeypatch.setattr(replay, 'validate_placement_openai_environment', lambda **_: None)
    monkeypatch.setattr(replay, 'configured_controls_robot_placement_openai_gate', lambda **_: None)
    return str(intent)


def test_rehearsal_stops_before_paid_reservation_in_fresh_output(tmp_path, monkeypatch, local_preflight):
    root = tmp_path/'isolated'
    root.mkdir()
    monkeypatch.setattr(replay.tempfile, 'mkdtemp', lambda **_: str(root))
    def run(**kwargs):
        assert kwargs['progression_root'] == root/'progression'
        assert kwargs['plan_root'] == root/'plans'
        with kwargs['openai_scope_lock']():
            kwargs['openai_gate_builder']()
        pytest.fail('paid reservation must be unreachable')
    monkeypatch.setattr(replay, 'materialize_configured_controls_autostart', run)
    result = replay.rehearse(source_launch_id='scene', launch_state_root='/retained', intent=local_preflight)
    assert result == {'status':'cpu_paid_boundary_reached','rehearsal_root':str(root),
                     'model_called':False,'provider_allocated':False,'production_evidence':False}


def test_an_early_return_is_not_a_pass(tmp_path, monkeypatch, local_preflight):
    monkeypatch.setattr(replay.tempfile, 'mkdtemp', lambda **_: str(tmp_path))
    monkeypatch.setattr(replay, 'materialize_configured_controls_autostart', lambda **_: {})
    with pytest.raises(RuntimeError, match='paid_boundary_not_reached'):
        replay.rehearse(source_launch_id='scene', launch_state_root='/retained', intent=local_preflight)


def test_bad_credentials_fail_before_expensive_geometry(tmp_path, monkeypatch, local_preflight):
    def invalid(**_):
        raise ValueError("credential_mismatch")
    monkeypatch.setattr(replay, 'validate_placement_openai_environment', invalid)
    monkeypatch.setattr(replay, 'materialize_configured_controls_autostart', lambda **_: pytest.fail("geometry ran"))
    with pytest.raises(ValueError, match="credential_mismatch"):
        replay.rehearse(source_launch_id='scene', launch_state_root='/retained', intent=local_preflight)


def test_rehearsal_constructs_real_gate_before_stopping(monkeypatch):
    calls=[]
    monkeypatch.setattr(replay, 'configured_controls_robot_placement_openai_gate', lambda **kw:calls.append(kw))
    with pytest.raises(replay.PaidBoundaryReached):
        replay._stop_before_paid_gate(run_id='retained')
    assert calls == [{'run_id':'retained'}]
