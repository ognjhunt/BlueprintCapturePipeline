"""The CPU rehearsal cannot reserve paid placement or report full completion."""
import pytest

from scripts import rehearse_selected_evaluation_cpu as replay


def test_rehearsal_stops_before_paid_reservation_in_fresh_output(tmp_path, monkeypatch):
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
    result = replay.rehearse(source_launch_id='scene', launch_state_root='/retained', intent='/retained/intent.json')
    assert result == {'status':'cpu_paid_boundary_reached','rehearsal_root':str(root),
                     'model_called':False,'provider_allocated':False,'production_evidence':False}


def test_an_early_return_is_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.tempfile, 'mkdtemp', lambda **_: str(tmp_path))
    monkeypatch.setattr(replay, 'materialize_configured_controls_autostart', lambda **_: {})
    with pytest.raises(RuntimeError, match='paid_boundary_not_reached'):
        replay.rehearse(source_launch_id='scene', launch_state_root='/retained', intent='/retained/intent.json')
