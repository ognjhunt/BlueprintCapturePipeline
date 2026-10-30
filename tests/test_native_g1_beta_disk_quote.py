"""Forward the verified G1 bundle's disk quote through the actual launch wrapper."""
from types import SimpleNamespace

import pytest

from blueprint_pipeline import adp_isaac_lab_arena_vast as arena

KIND = "native_g1_beta_screening"


def _bundle(tmp_path, disk):
    source = tmp_path / "development-only.zip"
    source.write_bytes(b"development-only hermetic wrapper fixture")
    return {"status": "ready", "bundle_path": str(source), "bundle_sha256": arena._file_sha256(source),
        "protocol_digest": "sha256:" + "a" * 64, "provider_disk_gb": disk}


@pytest.mark.parametrize("kind,disk,expected", [(KIND, 347, 347), ("native_task_arena", 347, 200)])
def test_actual_wrapper_passes_quote_to_provider_without_changing_legacy(tmp_path, monkeypatch, kind, disk, expected):
    bundle = _bundle(tmp_path, disk)
    observed = []
    def stage(*, job_dir, **kwargs):
        job_dir.mkdir(parents=True)
        for name in ("provider_bundle_url.txt", "provider_output_put_url.txt", "provider_output_get_url.txt"):
            (job_dir / name).write_text("https://example.invalid/fixture")
        return {"status": "completed"}
    def adapter(**kwargs):
        observed.append(kwargs)
        # Deliberately stop at the adapter boundary: no API, allocation or native result.
        raise RuntimeError("development_only_no_provider_execution")
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("BLUEPRINT_WATCHDOG_CALLER_EXIT_SURVIVAL", raising=False)
    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", stage)
    monkeypatch.setattr(arena, "cleanup_staged_wam_provider_objects", lambda path: {"all_objects_absent": True})
    monkeypatch.setattr(arena, "run_vast_provider_adapter", adapter)
    monkeypatch.setattr(arena, "require_pre_spend_preflight", lambda **kw: {"status": "PASS", "blockers": []})
    monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **kw: 60)
    handle = SimpleNamespace(pod_name_prefix="blueprint-native-task-policy-canary-" + "a" * 32,
        started_instance_id_path=tmp_path / "started-instance.txt")
    monkeypatch.setattr(arena, "arm_independent_vast_watchdog", lambda **kw: ({"status": "armed"}, handle))
    monkeypatch.setattr(arena, "close_independent_vast_watchdog", lambda **kw: {"status": "provider_terminal"})
    result = arena.run_arena_native_control_vast(approval_path=tmp_path / "unused.json", job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(), execute=True, prepared_bundle=bundle,
        provider_bundle_kind=kind, hard_cap_usd=1.0, hard_ttl_seconds=3600,
        require_independent_watchdog=True, instance_label_exact=handle.pod_name_prefix)
    assert len(observed) == 1
    assert observed[0]["disk_gb"] == expected
    assert observed[0]["hard_cap_usd"] == 1.0
    assert observed[0]["session_max_live_minutes"] == 60
    assert observed[0]["heartbeat_no_progress_seconds"] == 1800
    assert result["status"] == "blocked"
    assert "development_only_no_provider_execution" in " ".join(result["blockers"])


@pytest.mark.parametrize("disk", [None, True, 0, -1, 200.0, "200"])
def test_invalid_disk_quote_is_rejected_before_staging_or_provider(tmp_path, monkeypatch, disk):
    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", lambda **kw: pytest.fail("staging"))
    monkeypatch.setattr(arena, "run_vast_provider_adapter", lambda **kw: pytest.fail("provider"))
    with pytest.raises(ValueError, match="g1_beta_provider_disk_quote_invalid"):
        arena.run_arena_native_control_vast(approval_path=tmp_path / "unused.json", job_dir=tmp_path / "job",
            paid_resource_admission_grant=None, execute=False,
            prepared_bundle=_bundle(tmp_path, disk), provider_bundle_kind=KIND)
