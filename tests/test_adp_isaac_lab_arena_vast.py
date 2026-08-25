from __future__ import annotations

import json
import os
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
    _bounded_spend_gate_open,
    _candidate_policy_query_blocker,
    _next_attempt_root,
    _remaining_session_live_minutes,
    _vast_authority_environment,
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


def test_policy_query_blocker_does_not_invent_query_when_receipt_is_missing() -> None:
    assert (
        _candidate_policy_query_blocker({}, blocker_prefix="adp009d")
        == "adp009d_candidate_policy_query_status_missing"
    )
    assert (
        _candidate_policy_query_blocker(
            {"candidate_policy_queried": True}, blocker_prefix="adp009d"
        )
        == "adp009d_candidate_policy_queried"
    )
    assert (
        _candidate_policy_query_blocker(
            {"candidate_policy_queried": False}, blocker_prefix="adp009d"
        )
        is None
    )


def test_policy_query_blocker_requires_query_for_an_evaluation_transport() -> None:
    assert (
        _candidate_policy_query_blocker(
            {"candidate_policy_queried": True},
            blocker_prefix="adp009d",
            query_expected=True,
        )
        is None
    )
    assert (
        _candidate_policy_query_blocker(
            {"candidate_policy_queried": False},
            blocker_prefix="adp009d",
            query_expected=True,
        )
        == "adp009d_candidate_policy_query_required"
    )
    assert (
        _candidate_policy_query_blocker(
            {}, blocker_prefix="adp009d", query_expected=True
        )
        == "adp009d_candidate_policy_query_status_missing"
    )


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


def test_arena_bundle_uses_isaac_image_terms_and_ssh_bundle_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    monkeypatch.setenv(
        "BLUEPRINT_ADP009D_GATED_BACKBONE_AUTHORIZED", "true"
    )
    authorized_env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp009d_isaac",
        forward_hf_token=False,
    )
    assert authorized_env["BLUEPRINT_ADP009D_GATED_BACKBONE_AUTHORIZED"] == "true"
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


def test_gated_backbone_authority_is_scoped_to_one_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "BLUEPRINT_ADP009D_GATED_BACKBONE_AUTHORIZED"
    monkeypatch.delenv(name, raising=False)
    with _vast_authority_environment(gated_backbone_authorized=True):
        assert os.environ[name] == "true"
    assert name not in os.environ


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


def test_spend_gate_compares_projected_cost_not_hourly_rate_to_total_cap() -> None:
    assert _bounded_spend_gate_open(
        max_hourly_rate_usd=0.64,
        hard_cap_usd=0.50,
        remaining_live_minutes=46,
    )
    assert not _bounded_spend_gate_open(
        max_hourly_rate_usd=0.64,
        hard_cap_usd=0.50,
        remaining_live_minutes=47,
    )
    assert not _bounded_spend_gate_open(
        max_hourly_rate_usd=0.64,
        hard_cap_usd=0.50,
        remaining_live_minutes=29,
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
        assert kwargs["pod_name_prefix"] == "blueprint-adp-arena-"
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
    monkeypatch.setattr(
        arena,
        "require_pre_spend_preflight",
        lambda **_kwargs: {"status": "PASS", "blockers": []},
    )
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
        require_independent_watchdog=True,
        max_compute_cap=0,
    )

    assert result["status"] == "completed"
    assert watchdog_events == ["armed", "adapter", "closed"]
    assert observed_adapter["instance_label_prefix"] == watchdog_handle.pod_name_prefix
    assert (
        observed_adapter["started_instance_id_path"]
        == watchdog_handle.started_instance_id_path
    )
    assert observed_adapter["retention_watchdog_handoff"]["status"] == "armed"
    assert observed_adapter["max_compute_cap"] == 0
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


@pytest.mark.parametrize("provider_create_attempted", [False, True])
def test_policy_transport_seals_typed_media_gap_before_first_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_create_attempted: bool,
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

    def fake_adapter(*, job_dir, **_kwargs):
        provider = Path(job_dir)
        adapter = {
            "status": "blocked",
            "blockers": ["provider_bundle_readiness_parse_failed"],
            "estimated_cost_usd": 0.00014 if provider_create_attempted else 0.0,
            "vast_instance_ids": [48650527] if provider_create_attempted else [],
            "continuing_spend_from_this_run": False,
            "provider_create_attempted": provider_create_attempted,
        }
        write_json(provider / "vast_provider_adapter_result.json", adapter)
        write_json(
            provider / "vast_teardown_manifest.json",
            {
                "status": (
                    "destroyed"
                    if provider_create_attempted
                    else "not_required_blueprint_bundle_preflight_blocked"
                ),
                "vast_instance_ids": (
                    [48650527] if provider_create_attempted else []
                ),
                "continuing_spend_from_this_run": False,
            },
        )
        return adapter

    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(
        arena,
        "cleanup_staged_wam_provider_objects",
        lambda _path: {"all_objects_absent": True},
    )
    monkeypatch.setattr(arena, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        arena,
        "require_pre_spend_preflight",
        lambda **_kwargs: {"status": "PASS", "blockers": []},
    )
    monkeypatch.setattr(
        arena,
        "arm_independent_vast_watchdog",
        lambda **_kwargs: (
            {"status": "armed", "blockers": []},
            SimpleNamespace(
                pod_name_prefix="blueprint-native-task-policy-diagnostic-",
                started_instance_id_path=tmp_path / "started_vast_instance_id.txt",
            ),
        ),
    )
    monkeypatch.setattr(
        arena,
        "close_independent_vast_watchdog",
        lambda **kwargs: (
            {
                "status": (
                    "provider_terminal"
                    if provider_create_attempted
                    else "cancelled_no_allocation"
                )
            }
            if kwargs["provider_allocation_impossible"] is not provider_create_attempted
            else pytest.fail("watchdog close did not bind allocation truth")
        ),
    )
    monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 60)

    result = arena.run_arena_native_control_vast(
        approval_path=tmp_path / "unused.json",
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=prepared_bundle,
        hard_cap_usd=0.5,
        hard_ttl_seconds=9_000,
        require_independent_watchdog=True,
        candidate_policy_query_expected=True,
        blocker_prefix="native_task_arena_policy_diagnostic",
    )

    assert result["status"] == "blocked"
    assert result["candidate_policy_queried"] is False
    assert result["first_observation_reached"] is False
    assert result["scientific_attempt_started"] is False
    assert result["visual_evidence"] == {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": "provider_bundle_readiness_parse_failed",
        },
    }
    persisted = json.loads(
        (tmp_path / "job/adp_arena_vast_result.json").read_text(encoding="utf-8")
    )
    assert persisted["visual_evidence"] == result["visual_evidence"]


def test_live_transport_seals_guarded_warm_session_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"bundle")
    prepared_bundle = {
        "status": "ready",
        "bundle_path": str(bundle_path),
        "bundle_sha256": arena._file_sha256(bundle_path),
        "protocol_digest": "sha256:" + "b" * 64,
        "runtime_source_packet": {
            "packet_sha256": "sha256:" + "c" * 64,
            "packet_size_bytes": 4_400_000_000,
        },
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

    def fake_adapter(*, job_dir, **kwargs):
        assert kwargs["retain_native_task_arena_warm_session"] is True
        provider = Path(job_dir)
        provider.mkdir(parents=True, exist_ok=True)
        write_json(provider / "vast_provider_adapter_result.json", {"status": "completed"})
        write_json(
            provider / "vast_teardown_manifest.json",
            {"status": "retained_owned", "continuing_spend_from_this_run": True},
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
        return {
            "status": "completed",
            "blockers": [],
            "estimated_cost_usd": 0.1,
            "vast_instance_ids": [123],
            "continuing_spend_from_this_run": True,
            "provider_create_attempted": True,
            "retained_owned": True,
            "retention_decision": {
                "watchdog_pid": 456,
                "watchdog_deadline_epoch": 2_000_000_000,
                "warm_worker_evidence": {
                    "runtime_dependency_cache_ready": True,
                    "ssh_host": "ssh.example",
                    "ssh_port": 12345,
                },
            },
        }

    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(
        arena, "cleanup_staged_wam_provider_objects", lambda _path: {"all_objects_absent": True}
    )
    monkeypatch.setattr(arena, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        arena,
        "require_pre_spend_preflight",
        lambda **_kwargs: {"status": "PASS", "blockers": []},
    )
    monkeypatch.setattr(
        arena,
        "arm_independent_vast_watchdog",
        lambda **_kwargs: (
            {
                "status": "armed",
                "blockers": [],
                "watchdog_out_dir": str(tmp_path / "watchdog"),
                "pod_name_prefix": "blueprint-native-task-controls-",
            },
            SimpleNamespace(
                pod_name_prefix="blueprint-native-task-controls-",
                started_instance_id_path=tmp_path / "started_vast_instance_id.txt",
            ),
        ),
    )
    monkeypatch.setattr(
        arena,
        "close_independent_vast_watchdog",
        lambda **_kwargs: {"status": "retained_until_hard_ttl"},
    )
    monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 60)

    result = arena.run_arena_native_control_vast(
        approval_path=tmp_path / "unused.json",
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=prepared_bundle,
        hard_cap_usd=1.0,
        hard_ttl_seconds=3600,
        require_independent_watchdog=True,
        retain_warm_instance=True,
    )

    assert result["status"] == "completed"
    assert result["continuing_spend_from_this_run"] is True
    session = result["warm_session"]
    assert session["status"] == "ready"
    assert session["instance_id"] == 123
    assert session["runtime_dependency_cache_ready"] is True
    assert session["ssh_host"] == "ssh.example"
    assert session["ssh_port"] == 12345
    assert session["session_digest"].startswith("sha256:")


def test_live_transport_without_watchdog_does_not_close_a_missing_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"bundle")
    prepared_bundle = {
        "status": "ready",
        "bundle_path": str(bundle_path),
        "bundle_sha256": arena._file_sha256(bundle_path),
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

    def fake_adapter(*, job_dir, **_kwargs):
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
        return {
            "status": "completed",
            "blockers": [],
            "estimated_cost_usd": 0.1,
            "vast_instance_ids": [123],
            "continuing_spend_from_this_run": False,
            "provider_create_attempted": True,
        }

    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(
        arena,
        "cleanup_staged_wam_provider_objects",
        lambda _path: {"all_objects_absent": True},
    )
    monkeypatch.setattr(arena, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        arena,
        "require_pre_spend_preflight",
        lambda **_kwargs: {"status": "PASS", "blockers": []},
    )
    monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 60)
    monkeypatch.setattr(
        arena,
        "close_independent_vast_watchdog",
        lambda **_kwargs: pytest.fail("watchdog close called without an armed handle"),
    )

    result = arena.run_arena_native_control_vast(
        approval_path=tmp_path / "unused.json",
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=prepared_bundle,
        hard_cap_usd=1.0,
        hard_ttl_seconds=3600,
        require_independent_watchdog=False,
    )

    assert result["status"] == "completed"
    assert result["independent_watchdog"] == {"status": "not_required"}


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

    def fake_stage(*, job_dir, **_kwargs):
        nonlocal stage_called
        stage_called = True
        staging = Path(job_dir)
        staging.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text("https://example.invalid/object\n")
        return {"status": "completed"}

    def fake_adapter(**_kwargs):
        nonlocal adapter_called
        adapter_called = True
        return {"status": "completed"}

    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(
        arena,
        "cleanup_staged_wam_provider_objects",
        lambda _path: {"all_objects_absent": True},
    )
    monkeypatch.setattr(arena, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        arena,
        "require_pre_spend_preflight",
        lambda **_kwargs: {"status": "PASS", "blockers": []},
    )
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
        require_independent_watchdog=True,
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == ["adp_arena_independent_watchdog_not_armed"]
    assert stage_called is True
    assert adapter_called is False


def test_live_transport_blocks_before_watchdog_storage_or_compute_on_spend_lock_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"bundle")
    prepared_bundle = {
        "status": "ready",
        "bundle_path": str(bundle_path),
        "bundle_sha256": arena._file_sha256(bundle_path),
    }
    calls: list[str] = []
    failed_preflight = {
        "status": "FAIL",
        "blockers": ["spend_admission:spend_admission_lock_missing_or_symlink"],
    }

    def fail_pre_spend(**_kwargs):
        raise arena.PreSpendPreflightBlocked(failed_preflight)

    monkeypatch.setattr(arena, "require_pre_spend_preflight", fail_pre_spend)
    monkeypatch.setattr(
        arena,
        "arm_independent_vast_watchdog",
        lambda **_kwargs: calls.append("watchdog"),
    )
    monkeypatch.setattr(
        arena,
        "stage_wam_provider_bundle_object_store",
        lambda **_kwargs: calls.append("storage"),
    )
    monkeypatch.setattr(
        arena,
        "run_vast_provider_adapter",
        lambda **_kwargs: calls.append("compute"),
    )
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

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert calls == []
    assert result["blockers"] == [
        "adp_arena_pre_spend_preflight_not_passed",
        "spend_admission:spend_admission_lock_missing_or_symlink",
    ]
    assert result["candidate_policy_queried"] is False
    assert result["first_observation_reached"] is False
    assert result["scientific_attempt_started"] is False
    assert result["continuing_spend_from_this_run"] is False
    assert result["visual_evidence"] == {
        "status": "unavailable_before_first_observation",
        "media_gap": {
            "type": "before_first_observation",
            "reason": "spend_admission:spend_admission_lock_missing_or_symlink",
        },
    }
    assert Path(result["teardown_manifest_path"]).is_file()
    assert Path(result["artifact_manifest_path"]).is_file()
    teardown = json.loads(Path(result["teardown_manifest_path"]).read_text())
    assert teardown["continuing_spend_from_this_run"] is False
    manifest = json.loads(Path(result["artifact_manifest_path"]).read_text())
    assert manifest["status"] == "completed"


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
    assert observed["require_independent_watchdog"] is True
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["candidate_policy_queried"] is False
    assert admission["hard_cap_usd"] == 4.0
    assert admission["hard_ttl_seconds"] == 14400


def test_avoidlist_already_in_the_job_directory_does_not_abort_the_launch(
    tmp_path: Path,
) -> None:
    """Staging must be idempotent: the avoidlist may already live in the job dir.

    A live launch passed an avoidlist that was already at its staging
    destination.  shutil.copy2 raised SameFileError and aborted the run after
    the paid admission receipt had been written.
    """

    from blueprint_pipeline.adp_isaac_lab_arena_vast import _stage_machine_avoidlist

    job = tmp_path / "job"
    job.mkdir()
    staged = job / "adp_arena_vast_machine_avoidlist.json"
    write_json(staged, {"machine_ids": [1234]})

    # Source is literally the staging destination.
    result = _stage_machine_avoidlist(job, staged)
    assert result == staged
    assert json.loads(staged.read_text()) == {"machine_ids": [1234]}

    # A symlink pointing at the destination resolves to the same file.
    link = tmp_path / "avoidlist_link.json"
    link.symlink_to(staged)
    assert _stage_machine_avoidlist(job, link) == staged
    assert json.loads(staged.read_text()) == {"machine_ids": [1234]}

    # A genuinely external avoidlist is still copied in, overwriting the stale one.
    external = tmp_path / "external.json"
    write_json(external, {"machine_ids": [5678]})
    assert _stage_machine_avoidlist(job, external) == staged
    assert json.loads(staged.read_text()) == {"machine_ids": [5678]}

    # No avoidlist configured still yields the staging path without touching it.
    staged.unlink()
    assert _stage_machine_avoidlist(job, None) == staged
    assert not staged.exists()
