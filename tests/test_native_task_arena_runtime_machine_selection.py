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
