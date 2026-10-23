"""The canonical construction dispatch must retain the requested GPU host fence."""
import pytest
from blueprint_pipeline.native_task_arena_allocator_dispatch import native_task_arena_machine_selection_kwargs
from blueprint_pipeline import native_task_arena_vast as transport


def test_construction_machine_restriction_reaches_shared_provider_transport(monkeypatch, tmp_path):
    observed = {}
    def fake(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}
    monkeypatch.setattr(transport, "run_arena_native_control_vast", fake)
    selection = native_task_arena_machine_selection_kwargs("construction_canary", [44762, 21899, 44762])
    result = transport.run_native_task_arena_vast(
        job_dir=tmp_path, prepared_bundle={"schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "construction_canary", "policy_candidate_id": None,
        "candidate_policy_queried": False, "expected_output_filename": "native_task_arena_construction_result.v1.json",
        "container_image": "fixture@sha256:" + "a"*64}, paid_resource_admission_grant=None,
        execute=False, **selection)
    assert result["status"] == "dry_run_ready"
    assert observed["allowed_machine_ids"] == (21899, 44762)
    assert observed["require_independent_watchdog"] is True
    assert observed["execute"] is False


@pytest.mark.parametrize("mode", ["construction_canary", "controlled_policy"])
@pytest.mark.parametrize("value", [0, -1, True, "42"])
def test_invalid_host_identity_cannot_reach_provider(mode, value):
    with pytest.raises(ValueError, match="machine_id_invalid"):
        native_task_arena_machine_selection_kwargs(mode, [value])


def test_ga_image_uses_root_for_vast_ssh_bootstrap():
    from blueprint_pipeline.native_task_isaaclab_ga import IMAGE
    from blueprint_pipeline.vast_provider_adapter import _create_payload
    args = dict(label='test', launch_mode='ssh_direct', probe_script='true', disk_gb=200)
    assert _create_payload(image=IMAGE, **args)['user'] == 'root'
    assert 'user' not in _create_payload(image='nvcr.io/nvidia/isaac-sim:6.0.1', **args)
    assert 'user' not in _create_payload(image='nvcr.io/nvidia/isaac-lab:3.0.0@sha256:unknown', **args)
    assert 'user' not in _create_payload(image=IMAGE, **{**args, 'launch_mode': 'args'})


def test_provider_startup_status_message_is_retained():
    from blueprint_pipeline.vast_provider_adapter import _instance_liveness_from_payload
    observed = _instance_liveness_from_payload(
        {'instances': {'id': 123, 'actual_status': 'stopped', 'status_msg': 'bootstrap permission denied'}},
        instance_id=123)
    assert observed['status_message'] == 'bootstrap permission denied'
