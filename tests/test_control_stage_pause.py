"""Paused controls cannot enter paid admission or native control execution."""
import json
from pathlib import Path

import pytest
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.native_policy_canary_control_gate import execute_native_controls
from blueprint_pipeline.task_evaluation_robot_placement_warm_executor import WarmNativeConstructionFeedbackExecutor


def test_allocator_refuses_controls_before_bundle_or_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: pytest.fail("controls reached launch admission"))
    output = tmp_path / "result.json"
    result = allocator.main(["gpu-canary", "--probe-kind", "native-task-arena-controls",
                             "--adapter-output", str(output)])
    assert result == 2
    value = json.loads(output.read_bytes())
    assert value["blockers"] == ["task_evaluation_controls_paused_by_owner"]
    assert value["provider_allocations_performed"] == 0
    assert value["continuing_spend_from_this_run"] is False


def test_warm_controls_refused_before_runtime_access():
    executor = object.__new__(WarmNativeConstructionFeedbackExecutor)
    with pytest.raises(ValueError, match="controls_paused_by_owner"):
        executor.continue_to_controls({})


def test_per_cell_controls_refused_before_simulator_access(tmp_path):
    with pytest.raises(ValueError, match="controls_paused_by_owner"):
        execute_native_controls(cell_runtime=None, built=None, scene_plan={}, gate={}, output_root=Path(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('closed', [True, False])
def test_omitted_warm_controls_require_real_teardown(tmp_path, monkeypatch, closed):
    from blueprint_pipeline import native_task_arena_feedback_allocator_adapter as adapter
    packet = tmp_path / 'packet'
    packet.mkdir()
    (packet / 'native_task_arena_packet_request.v1.json').write_text(json.dumps({'native_construction_feedback': {}}))
    native = {'status': 'completed', 'construction_gate_qualified': True, 'result_digest': 'sha256:' + 'a' * 64}
    controller = {
        'status': 'construction_completed_controls_omitted',
        'history': [{'execution': {'native_result': native}}],
        'continuing_spend_from_this_run': not closed,
        'warm_session_closeout': {'provider_instance_absent': closed},
    }
    monkeypatch.setattr(adapter, 'run_retained_native_construction_feedback', lambda **kwargs: controller)
    cold = {'status': 'blocked', 'blockers': ['prior_grasp_failed'], 'warm_session': {'instance_id': 1},
            'native_control_result_path': str(tmp_path / 'old.json'), 'continuing_spend_from_this_run': True}
    result = adapter.continue_retained_feedback_if_requested(execute=True, construction_requested=True,
        retain_warm_session=True, result=cold, packet_dir=packet, runtime_source_packet_receipt_path=tmp_path / 'runtime.json',
        prepared_bundle={'implementation_commit': 'a' * 40}, native_authority={}, job_dir=tmp_path / 'job',
        max_hourly_rate_usd=0.8, hard_cap_usd=0.45, hard_ttl_seconds=2025)
    assert result['status'] == ('completed' if closed else 'blocked')
    assert result['continuing_spend_from_this_run'] is (not closed)
    assert 'native_controls_result_path' not in result
    if closed:
        assert json.loads(Path(result['native_control_result_path']).read_bytes()) == native
        assert result['controls_qualified'] is result['qualified_comparison_permitted'] is False
        assert result['warm_session'] is None


def test_old_strict_canary_bundle_is_refused_before_provider_startup(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    from blueprint_pipeline import native_task_arena_policy_canary_bundle as bundle
    from blueprint_pipeline.native_policy_canary_control_gate import CONTROL_IDS
    path = tmp_path / "strict.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("provider_runtime/adp_arena_provider_manifest.json", "{}")
        archive.writestr("provider_runtime/runtime_inputs/policy_canary_runtime_inputs.json", json.dumps({
            "task_success_contract": {"criteria": {"controls": {
                "mode": "required_per_cell", "control_ids": list(CONTROL_IDS)}}}}))
    monkeypatch.setattr(bundle.subprocess, "run", lambda *a, **kw: pytest.fail("started strict worker"))
    result = bundle.preflight_sealed_policy_canary_bundle({"bundle_path": str(path),
        "bundle_size_bytes": path.stat().st_size, "bundle_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()})
    assert result["status"] == "blocked"
    assert result["blockers"] == ["task_evaluation_controls_paused_by_owner"]
    assert result["provider_mutation_performed"] is False
