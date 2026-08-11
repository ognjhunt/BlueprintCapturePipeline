from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import adp_isaac_lab_arena_vast as arena
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.adp_founder_sim_protocol import (
    build_founder_approval_receipt,
    expected_founder_approval_statement,
)
from blueprint_pipeline.adp_isaac_lab_arena_vast import (
    DEFAULT_IMAGE,
    PROBE_KIND,
    _next_attempt_root,
    _remaining_session_live_minutes,
    build_arena_native_control_bundle,
)
from blueprint_pipeline.common import write_json
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
    _resolve_probe_image,
)


def _approval_path(tmp_path: Path) -> Path:
    receipt = build_founder_approval_receipt(
        statement=expected_founder_approval_statement(),
        evidence_ref="codex-task://test/user-message",
    )
    path = tmp_path / "approval.json"
    write_json(path, receipt)
    return path


def test_bundle_is_deterministic_approved_and_native_control_only(tmp_path: Path) -> None:
    approval = _approval_path(tmp_path)
    first = build_arena_native_control_bundle(
        approval_path=approval, job_dir=tmp_path / "first", generated_at="fixed"
    )
    second = build_arena_native_control_bundle(
        approval_path=approval, job_dir=tmp_path / "second", generated_at="fixed"
    )

    assert first["status"] == "ready"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["candidate_policy_queried"] is False
    assert first["candidate_outcomes_accessed"] is False
    assert first["container_image"] == DEFAULT_IMAGE
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        names = set(archive.namelist())
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
        runner = archive.read("provider_runtime/adp_arena_provider_runner.py").decode()
    assert "provider_runtime/founder_sim_approval_receipt.json" in names
    assert "provider_runtime/arena_worker_request.json" in names
    assert "GR00T-N1.6-DROID" not in runner
    assert "pi05_droid_jointpos_polaris" not in runner
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="adp_arena",
            entrypoint_text=entrypoint,
            runner_text=runner,
        )
        == []
    )
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp_arena",
        bundle_path=Path(first["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_arena_bundle_uses_isaac_image_terms_and_ssh_bundle_path(tmp_path: Path) -> None:
    assert (
        _resolve_probe_image(
            public_image="public",
            isaac_image=DEFAULT_IMAGE,
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_arena",
        )
        == DEFAULT_IMAGE
    )
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=True,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_arena",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp_arena",
        forward_hf_token=False,
    )
    assert env["ACCEPT_EULA"] == "Y"
    assert env["PRIVACY_CONSENT"] == "Y"
    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_arena",
    )
    assert "run_adp_arena_provider_runtime.sh" in script
    assert "adp_arena_provider_runtime_output.zip" in script
    assert "BLUEPRINT_VAST_CUDA_RUNTIME_DEFERRED_TO_ISAAC_SIMULATION_APP" in script
    assert "wp.get_devices()" in script
    assert "isaac_simulation_app_warp" in script
    assert script.rindex("BLUEPRINT_VAST_GPU_SANITY_OK") > script.index(
        "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED"
    )


def test_paid_attempt_roots_are_fresh_and_preserve_prior_evidence(tmp_path: Path) -> None:
    write_json(tmp_path / "adp_arena_vast_session_budget.json", {"attempt_count": 1})
    prior = tmp_path / "attempts" / "attempt_002"
    prior.mkdir(parents=True)
    (prior / "prior_evidence.txt").write_text("preserve", encoding="utf-8")

    number, root = _next_attempt_root(tmp_path)

    assert number == 3
    assert root == tmp_path / "attempts" / "attempt_003"
    assert (prior / "prior_evidence.txt").read_text(encoding="utf-8") == "preserve"


def test_successor_ttl_reserves_only_remaining_cumulative_budget(tmp_path: Path) -> None:
    write_json(
        tmp_path / "adp_arena_vast_session_budget.json",
        {
            "attempts": [
                {
                    "actual_live_runtime_seconds_observed_by_adapter": 560.9,
                    "estimated_cost_usd": 0.083795,
                }
            ]
        },
    )

    assert (
        _remaining_session_live_minutes(
            job=tmp_path,
            hard_cap_usd=4.0,
            hard_ttl_seconds=14_400,
            max_hourly_rate_usd=1.0,
        )
        == 230
    )


def test_live_transport_emits_allocator_artifact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"bundle")
    prepared_bundle = {
        "status": "ready",
        "bundle_path": str(bundle_path),
        "bundle_sha256": arena._file_sha256(bundle_path),
        "protocol_digest": "sha256:" + "b" * 64,
    }

    def fake_stage(*, job_dir, **_kwargs):
        staging = Path(job_dir)
        staging.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text("https://example.invalid/object\n")
        return {"status": "completed"}

    observed_adapter: dict = {}
    watchdog_events: list[str] = []
    watchdog_handle = SimpleNamespace(
        pod_name_prefix="blueprint-adp009d-watchdog-",
        started_instance_id_path=tmp_path / "started_vast_instance_id.txt",
    )

    def fake_arm(**kwargs):
        watchdog_events.append("armed")
        assert kwargs["pod_name_prefix_base"] == "blueprint-adp-arena-"
        return {"status": "armed", "blockers": []}, watchdog_handle

    def fake_adapter(*, job_dir, **kwargs):
        watchdog_events.append("adapter")
        observed_adapter.update(kwargs)
        provider = Path(job_dir)
        provider.mkdir(parents=True, exist_ok=True)
        write_json(provider / "vast_provider_adapter_result.json", {"status": "completed"})
        write_json(
            provider / "vast_teardown_manifest.json",
            {"continuing_spend_from_this_run": False},
        )
        with zipfile.ZipFile(provider / "vast_provider_runtime_output.zip", "w") as archive:
            archive.writestr(
                "adp_arena_native_canary.json",
                json.dumps(
                    {
                        "status": "completed",
                        "candidate_policy_queried": False,
                        "blockers": [],
                    }
                ),
            )
            archive.writestr("lossless_frames/frame_000001.png", b"lossless")
        return {
            "status": "completed",
            "blockers": [],
            "estimated_cost_usd": 0.1,
            "vast_instance_ids": [123],
            "continuing_spend_from_this_run": False,
            "provider_create_attempted": True,
        }

    def fake_close(**kwargs):
        watchdog_events.append("closed")
        assert kwargs["instance_ids"] == [123]
        assert kwargs["provider_teardown_completed"] is True
        return {"status": "provider_terminal"}

    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(
        arena, "cleanup_staged_wam_provider_objects", lambda _path: {"all_objects_absent": True}
    )
    monkeypatch.setattr(arena, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(arena, "arm_independent_vast_watchdog", fake_arm)
    monkeypatch.setattr(arena, "close_independent_vast_watchdog", fake_close)
    monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 60)

    result = arena.run_arena_native_control_vast(
        approval_path=tmp_path / "unused.json",
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=prepared_bundle,
        hard_cap_usd=1.0,
        hard_ttl_seconds=3600,
    )

    assert result["status"] == "completed"
    assert watchdog_events == ["armed", "adapter", "closed"]
    assert observed_adapter["instance_label_prefix"] == watchdog_handle.pod_name_prefix
    assert (
        observed_adapter["started_instance_id_path"]
        == watchdog_handle.started_instance_id_path
    )
    assert observed_adapter["retention_watchdog_handoff"]["status"] == "armed"
    manifest_path = Path(result["artifact_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "completed"
    assert manifest["binding"]["bundle_sha256"] == prepared_bundle["bundle_sha256"]
    assert set(manifest["observed_roles"]) == {
        "allocator_adapter_result",
        "provider_runtime_evidence",
        "teardown_manifest",
    }
    assert "immutable_execution/lossless_frames/frame_000001.png" in {
        row["relative_path"] for row in manifest["files"]
    }


def test_live_transport_blocks_before_storage_or_compute_when_watchdog_is_not_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"bundle")
    prepared_bundle = {
        "status": "ready",
        "bundle_path": str(bundle_path),
        "bundle_sha256": arena._file_sha256(bundle_path),
    }
    stage_called = False
    adapter_called = False

    def fake_stage(**_kwargs):
        nonlocal stage_called
        stage_called = True
        return {"status": "completed"}

    def fake_adapter(**_kwargs):
        nonlocal adapter_called
        adapter_called = True
        return {"status": "completed"}

    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(arena, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 60)
    monkeypatch.setattr(
        arena,
        "arm_independent_vast_watchdog",
        lambda **_kwargs: (
            {
                "status": "blocked",
                "blockers": ["independent_vast_watchdog_not_armed"],
            },
            None,
        ),
    )

    result = arena.run_arena_native_control_vast(
        approval_path=tmp_path / "unused.json",
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=prepared_bundle,
        hard_cap_usd=1.0,
        hard_ttl_seconds=3600,
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == ["independent_vast_watchdog_not_armed"]
    assert stage_called is False
    assert adapter_called is False


def _allocator_args(tmp_path: Path, approval: Path, *, execute: bool) -> list[str]:
    values = [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp-arena",
        "--adp-arena-approval",
        str(approval),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "4.0",
        "--adp-hard-ttl-seconds",
        "14400",
    ]
    if execute:
        values.append("--execute")
    return values


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_arena_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    approval = _approval_path(tmp_path)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_arena_native_control_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, approval, execute=execute)) == 0
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["candidate_policy_queried"] is False
    assert admission["hard_cap_usd"] == 4.0
    assert admission["hard_ttl_seconds"] == 14400
