import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    canonical_digest,
)
from blueprint_pipeline.task_evaluation_launch_reconciler import (
    main as reconcile_main,
    reconcile_launches,
)


LAUNCH_ID = "sam31-840920-task-a-c755b31c-web-20260814T184531Z"
PROFILE_ID = (
    "adp-sam31-source-tracks-live-"
    "c755b31c5ee92229ca3c41760bbd97aca59947de-scene840920-task-a-v4"
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _profile(*, reconciliation: bool, provider_argv: list[str] | None = None) -> dict:
    profile = {
        "profile_id": PROFILE_ID,
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": provider_argv if provider_argv is not None else ["--provider", "vast"],
        },
    }
    if reconciliation:
        profile["reconciliation"] = {
            "required_providers": ["vast"],
            "max_guard_age_seconds": 300,
        }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    return profile


def _zero_guard(*, observed_at: datetime, live_count: int = 0) -> dict:
    zero = live_count == 0
    return {
        "schema_version": "gpu_spend_guard.v1",
        "generated_at": observed_at.isoformat(),
        "reap_mode": True,
        "provider_zero_verified": zero,
        "live_instance_count": live_count,
        "total_burn_per_hour_usd": 0.0 if zero else 0.5,
        "reap_candidate_ids": [],
        "reap_results": [],
        "inventory_results": [{
            "provider": "vast",
            "status": "succeeded",
            "row_count": live_count,
            "required": True,
        }],
        "provider_zero": {
            "status": "verified" if zero else "unverified",
            "required_provider_ids": ["vast"],
            "global_live_instance_count": live_count,
            "global_total_burn_per_hour_usd": 0.0 if zero else 0.5,
        },
    }


def _processing_run(
    tmp_path: Path,
    *,
    profile: dict,
    started_at: datetime,
    ttl_seconds: int = 60,
) -> tuple[Path, Path, Path]:
    request = {
        "launch_id": LAUNCH_ID,
        "run_id": LAUNCH_ID,
        "launch_profile_id": PROFILE_ID,
        "launch_profile_digest": profile["profile_digest"],
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    queue_root = tmp_path / "queue"
    processing = queue_root / "processing" / f"{LAUNCH_ID}-production-shaped.json"
    _write(processing, request)
    profile_dir = tmp_path / "profiles"
    _write(profile_dir / f"{PROFILE_ID}.json", profile)
    run_root = tmp_path / "state" / LAUNCH_ID
    _write(run_root / "launch_started.json", {
        "started_at": started_at.isoformat(),
        "hard_ttl_seconds": ttl_seconds,
    })
    assert not (run_root / "launch_profile.json").exists()
    return processing, profile_dir, run_root


@pytest.mark.parametrize(
    ("reconciliation", "scope_source"),
    [(True, "profile_reconciliation"), (False, "canonical_allocator_argument")],
    ids=["exact-production-profile", "legacy-cli-scope"],
)
def test_expired_sam_record_is_contained_without_making_service_unhealthy(
    tmp_path: Path,
    reconciliation: bool,
    scope_source: str,
) -> None:
    observed_at = datetime.now(timezone.utc)
    processing, profile_dir, run_root = _processing_run(
        tmp_path,
        profile=_profile(reconciliation=reconciliation),
        started_at=observed_at - timedelta(days=3),
    )
    guard_path = tmp_path / "guard.json"
    _write(guard_path, _zero_guard(observed_at=observed_at))
    report_path = tmp_path / "report.json"

    exit_code = reconcile_main([
        "--queue-root", str(tmp_path / "queue"),
        "--state-root", str(tmp_path / "state"),
        "--profile-dir", str(profile_dir),
        "--guard-report", str(guard_path),
        "--report-out", str(report_path),
    ])
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert report["status"] == "passed"
    assert report["launches"][0]["profile_record_source"] == "published_profile"
    assert report["launches"][0]["provider_scope_source"] == scope_source
    assert not processing.exists()
    assert (tmp_path / "queue" / "blocked" / processing.name).is_file()
    recovery = json.loads((run_root / "orphan_recovery_receipt.json").read_text())
    assert recovery["required_providers"] == ["vast"]
    assert recovery["provider_zero_confirmed"] is True
    assert recovery["allocator_invoked"] is False
    assert recovery["provider_scope_source"] == scope_source
    assert recovery["profile_record_source"] == "published_profile"


def test_live_sam_record_is_not_terminalized(tmp_path: Path) -> None:
    observed_at = datetime.now(timezone.utc)
    processing, profile_dir, run_root = _processing_run(
        tmp_path,
        profile=_profile(reconciliation=True),
        started_at=observed_at - timedelta(seconds=5),
    )
    guard_path = tmp_path / "guard.json"
    _write(guard_path, _zero_guard(observed_at=observed_at))

    report = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        profile_dir=profile_dir,
        guard_report_path=guard_path,
        now=observed_at,
        publish_progress=False,
    )

    assert report["launches"][0]["status"] == "processing_within_ttl"
    assert processing.is_file()
    assert not (run_root / "orphan_recovery_receipt.json").exists()


@pytest.mark.parametrize(
    "provider_argv",
    [[], ["--provider"], ["--provider", "vast", "--provider", "runpod"]],
    ids=["missing", "value-missing", "conflicting"],
)
def test_ambiguous_legacy_provider_scope_stays_fail_closed(
    tmp_path: Path,
    provider_argv: list[str],
) -> None:
    observed_at = datetime.now(timezone.utc)
    processing, profile_dir, run_root = _processing_run(
        tmp_path,
        profile=_profile(reconciliation=False, provider_argv=provider_argv),
        started_at=observed_at - timedelta(days=3),
    )
    guard_path = tmp_path / "guard.json"
    _write(guard_path, _zero_guard(observed_at=observed_at))

    report = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        profile_dir=profile_dir,
        guard_report_path=guard_path,
        now=observed_at,
    )

    assert report["status"] == "blocked"
    assert "gpu_required_provider_scope_missing" in report["launches"][0]["blockers"]
    assert processing.is_file()
    assert not (run_root / "orphan_recovery_receipt.json").exists()


def test_live_provider_inventory_prevents_stale_containment(tmp_path: Path) -> None:
    observed_at = datetime.now(timezone.utc)
    processing, profile_dir, run_root = _processing_run(
        tmp_path,
        profile=_profile(reconciliation=False),
        started_at=observed_at - timedelta(days=3),
    )
    guard_path = tmp_path / "guard.json"
    _write(guard_path, _zero_guard(observed_at=observed_at, live_count=1))

    report = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        profile_dir=profile_dir,
        guard_report_path=guard_path,
        now=observed_at,
    )

    assert report["status"] == "blocked"
    assert "gpu_provider_nonzero" in report["launches"][0]["blockers"]
    assert processing.is_file()
    assert not (run_root / "orphan_recovery_receipt.json").exists()


def test_unbound_published_profile_cannot_supply_provider_scope(tmp_path: Path) -> None:
    observed_at = datetime.now(timezone.utc)
    processing, profile_dir, run_root = _processing_run(
        tmp_path,
        profile=_profile(reconciliation=False),
        started_at=observed_at - timedelta(days=3),
    )
    request = json.loads(processing.read_text(encoding="utf-8"))
    request["launch_profile_digest"] = "sha256:" + "0" * 64
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    _write(processing, request)
    guard_path = tmp_path / "guard.json"
    _write(guard_path, _zero_guard(observed_at=observed_at))

    report = reconcile_launches(
        queue_root=tmp_path / "queue",
        state_root=tmp_path / "state",
        profile_dir=profile_dir,
        guard_report_path=guard_path,
        now=observed_at,
    )

    assert report["status"] == "blocked"
    assert report["launches"][0]["status"] == "reconciliation_blocked"
    assert processing.is_file()
    assert not (run_root / "orphan_recovery_receipt.json").exists()


def test_reconciler_service_binds_the_published_profile_directory() -> None:
    service = (
        Path(__file__).resolve().parents[1]
        / "deploy/systemd/blueprint-task-evaluation-launch-reconciler.service"
    ).read_text(encoding="utf-8")

    assert "BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_DIR" in service
    assert "--profile-dir" in service
