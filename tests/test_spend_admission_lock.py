from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline import paid_lane_guard
from blueprint_pipeline.paid_lane_guard import (
    PreSpendPreflightBlocked,
    require_pre_spend_preflight,
)
from blueprint_pipeline.spend_admission_lock import (
    HARD_STOP_USD,
    build_spend_admission_lock,
    validate_spend_admission_lock,
)


NOW = datetime(2026, 7, 9, 18, 0, tzinfo=timezone.utc)


def _billing(
    total: float, *, status: str = "reconciled", now: datetime = NOW
) -> dict[str, object]:
    return {
        "status": status,
        "required": True,
        "billing_export_schema_version": "blueprint.provider_billing_export.v1",
        "billing_export_sha256": f"sha256:{'a' * 64}",
        "billing_export_mode_octal": "0600",
        # Anchor billing freshness to the lock's own clock: a fixed timestamp
        # here turned into a 24h time bomb (MAX_BILLING_AGE_SECONDS) that
        # broke every CI run after 2026-07-10T18:00Z.
        "generated_at": now.isoformat(),
        "currency": "USD",
        "scope": "blueprint_beta_100_user_cohort",
        "provider_totals_usd": {
            "runpod": total,
            "vast": 0.0,
            "digitalocean": 0.0,
        },
        "blockers": [] if status == "reconciled" else ["billing_stale"],
    }


def _inventory() -> list[dict[str, object]]:
    return [
        {
            "provider": provider,
            "status": "succeeded",
            "required": True,
            "credential_present": True,
            "row_count": 0,
            "blockers": [],
        }
        for provider in ("runpod", "vast", "digitalocean")
    ]


def _fleet(total: float, *, blocked_for_total: bool = False) -> dict[str, object]:
    return {
        "status": "blocked" if blocked_for_total else "passed",
        "total_spend_usd": total,
        "blockers": ["fleet_total_spend_limit_exceeded"]
        if blocked_for_total
        else [],
    }


def _lock(
    total: float,
    *,
    now: datetime = NOW,
    override_path: Path | None = None,
    instances: list[dict[str, object]] | None = None,
    blocked_for_total: bool = False,
) -> dict[str, object]:
    return build_spend_admission_lock(
        fleet_budget=_fleet(total, blocked_for_total=blocked_for_total),
        billing_reconciliation=_billing(total, now=now),
        instances=instances or [],
        reap_results=[],
        inventory_results=_inventory(),
        override_path=override_path,
        now=now,
    )


def _write_override(
    path: Path,
    *,
    now: datetime = NOW,
    expires_delta: timedelta = timedelta(hours=2),
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "blueprint.paid_spend_override.v1",
                "status": "approved",
                "scope": "paid_spend_hard_stop",
                "override_id": "override-20260709-001",
                "hard_stop_usd": HARD_STOP_USD,
                "allow_new_paid_work": True,
                "requested_by": "oncall-operator",
                "approved_by": "finance-approver",
                "reason": "Time-bounded customer recovery approved after cost review.",
                "ticket_uri": "https://tickets.example.invalid/INC-1234",
                "issued_at": now.isoformat(),
                "expires_at": (now + expires_delta).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)


def _preflight_kwargs() -> dict[str, object]:
    return {
        "lane": "robot_eval_provider_launcher",
        "provider": "runpod",
        "credential_present": True,
        "capacity_evidence": {"available": True, "detail": "capacity"},
        "image_contract": {"image_ref": "repo/image:v1", "pinned": True},
        "runtime_contract": {
            "startup_marker": "started",
            "progress_marker": "result",
            "startup_timeout_seconds": 60,
            "no_progress_timeout_seconds": 60,
        },
        "spend_gate_open": True,
    }


def test_below_threshold_opens_admission_without_page() -> None:
    evidence = _lock(HARD_STOP_USD - 0.01)

    assert evidence["status"] == "open"
    assert evidence["admission_allowed"] is True
    assert evidence["threshold_crossed"] is False
    assert evidence["page_event"]["required"] is False  # type: ignore[index]
    assert validate_spend_admission_lock(evidence, now=NOW) == []


@pytest.mark.parametrize("total", [HARD_STOP_USD, HARD_STOP_USD + 0.01])
def test_threshold_crossing_stops_new_work_pages_and_drains(total: float) -> None:
    evidence = _lock(
        total,
        blocked_for_total=total > HARD_STOP_USD,
        instances=[
            {
                "provider": "runpod",
                "id": "pod-live",
                "live": True,
                "reap_candidate": False,
            }
        ],
    )

    assert evidence["status"] == "blocked"
    assert evidence["admission_allowed"] is False
    assert evidence["threshold_crossed"] is True
    assert "cohort_hard_stop_reached" in evidence["blockers"]
    assert evidence["page_event"]["required"] is True  # type: ignore[index]
    drain = evidence["controlled_drain"]
    assert drain["status"] == "draining"  # type: ignore[index]
    assert drain["new_paid_work_stopped"] is True  # type: ignore[index]
    assert drain["teardown_evidence_complete"] is False  # type: ignore[index]


def test_current_billing_reconciliation_is_mandatory_even_below_threshold() -> None:
    evidence = build_spend_admission_lock(
        fleet_budget=_fleet(10.0),
        billing_reconciliation=_billing(10.0, status="blocked"),
        instances=[],
        reap_results=[],
        inventory_results=_inventory(),
        override_path=None,
        now=NOW,
    )

    assert evidence["admission_allowed"] is False
    assert "billing_reconciliation_not_current" in evidence["blockers"]
    assert evidence["page_event"]["required"] is True  # type: ignore[index]


def test_complete_three_provider_inventory_is_mandatory_for_admission() -> None:
    evidence = build_spend_admission_lock(
        fleet_budget=_fleet(10.0),
        billing_reconciliation=_billing(10.0),
        instances=[],
        reap_results=[],
        inventory_results=_inventory()[:-1],
        override_path=None,
        now=NOW,
    )

    assert evidence["admission_allowed"] is False
    assert "provider_inventory_coverage_incomplete" in evidence["blockers"]
    assert evidence["controlled_drain"]["status"] == "inventory_unknown"  # type: ignore[index]


def test_chokepoint_recomputes_billing_inventory_and_claim_contracts() -> None:
    evidence = _lock(10.0)
    evidence["billing_reconciliation"]["generated_at"] = (  # type: ignore[index]
        NOW - timedelta(days=2)
    ).isoformat()
    evidence["provider_inventory"][0]["status"] = "failed"  # type: ignore[index]
    evidence["claim_boundary"][  # type: ignore[index]
        "billing_export_is_external_input_not_live_api_proof"
    ] = False

    blockers = validate_spend_admission_lock(evidence, now=NOW)

    assert "spend_admission_lock_billing_stale_or_invalid_time" in blockers
    assert any("provider_inventory_not_succeeded:runpod" in item for item in blockers)
    assert "spend_admission_lock_claim_boundary_invalid" in blockers


def test_two_person_expiring_override_is_audited_and_cannot_go_stale(
    tmp_path: Path,
) -> None:
    override = tmp_path / "override.json"
    _write_override(override)

    opened = _lock(
        HARD_STOP_USD,
        override_path=override,
        blocked_for_total=True,
    )

    assert opened["status"] == "override_open"
    assert opened["admission_allowed"] is True
    assert opened["page_event"]["required"] is True  # type: ignore[index]
    assert opened["page_event"]["delivery_status"] == "external_pending"  # type: ignore[index]
    assert opened["override"]["status"] == "valid"  # type: ignore[index]
    assert opened["override"]["source_artifact_digest"].startswith("sha256:")  # type: ignore[index,union-attr]
    assert validate_spend_admission_lock(opened, now=NOW) == []

    expired = _lock(
        HARD_STOP_USD,
        now=NOW + timedelta(hours=3),
        override_path=override,
        blocked_for_total=True,
    )
    assert expired["admission_allowed"] is False
    assert any(
        blocker.endswith("spend_override_expired")
        for blocker in expired["blockers"]
    )

    opened["override"]["approved_by"] = "oncall-operator"  # type: ignore[index]
    assert any(
        "spend_override_approver_must_differ" in blocker
        for blocker in validate_spend_admission_lock(opened, now=NOW)
    )


def test_group_writable_override_is_rejected(tmp_path: Path) -> None:
    override = tmp_path / "override.json"
    _write_override(override)
    override.chmod(0o660)

    evidence = _lock(
        HARD_STOP_USD,
        override_path=override,
        blocked_for_total=True,
    )

    assert evidence["admission_allowed"] is False
    assert any(
        blocker.endswith("spend_override_writable_by_group_or_world")
        for blocker in evidence["blockers"]
    )


def test_oversized_override_is_rejected_without_digest(tmp_path: Path) -> None:
    override = tmp_path / "override.json"
    override.write_bytes(b"{" + b" " * (64 * 1024) + b"}")
    override.chmod(0o600)

    evidence = _lock(
        HARD_STOP_USD,
        override_path=override,
        blocked_for_total=True,
    )

    assert evidence["admission_allowed"] is False
    assert evidence["override"]["source_artifact_digest"] is None  # type: ignore[index]
    assert any(
        blocker.endswith("spend_override_too_large")
        for blocker in evidence["blockers"]
    )


def test_production_chokepoint_requires_current_admission_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_LAUNCH_PROOF_MODE", "production")
    with pytest.raises(PreSpendPreflightBlocked) as missing:
        require_pre_spend_preflight(**_preflight_kwargs())
    assert any(
        blocker.startswith("spend_admission:")
        for blocker in missing.value.preflight["blockers"]
    )

    now = datetime.now(timezone.utc)
    evidence = _lock(42.0, now=now)
    passed = require_pre_spend_preflight(
        **_preflight_kwargs(),
        spend_admission_lock=evidence,
    )
    assert passed["status"] == "PASS"
    assert passed["spend_admission_lock"]["required"] is True
    assert passed["spend_admission_lock"]["effective_spend_usd"] == 42.0


def test_expired_lock_blocks_every_paid_lane_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(paid_lane_guard.REQUIRE_SPEND_ADMISSION_LOCK_ENV, "true")
    stale = _lock(10.0, now=NOW - timedelta(hours=1))
    blockers: list[tuple[str, ...]] = []
    for lane in (
        "isaac_particlefield_render",
        "runpod_wam_async",
        "robot_eval_provider_launcher",
    ):
        kwargs = _preflight_kwargs()
        kwargs["lane"] = lane
        with pytest.raises(PreSpendPreflightBlocked) as exc_info:
            require_pre_spend_preflight(
                **kwargs,
                spend_admission_lock=stale,
            )
        blockers.append(tuple(exc_info.value.preflight["blockers"]))
    assert len(set(blockers)) == 1
    assert any("spend_admission_lock_stale" in blocker for blocker in blockers[0])


def test_chokepoint_loads_only_permission_safe_runtime_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "paid-spend-admission.json"
    lock_path.write_text(
        json.dumps(_lock(17.0, now=datetime.now(timezone.utc))),
        encoding="utf-8",
    )
    lock_path.chmod(0o600)
    monkeypatch.setenv(
        paid_lane_guard.SPEND_ADMISSION_LOCK_PATH_ENV,
        str(lock_path),
    )
    monkeypatch.setenv(paid_lane_guard.REQUIRE_SPEND_ADMISSION_LOCK_ENV, "true")

    passed = require_pre_spend_preflight(**_preflight_kwargs())
    assert passed["status"] == "PASS"

    lock_path.chmod(0o660)
    with pytest.raises(PreSpendPreflightBlocked) as unsafe:
        require_pre_spend_preflight(**_preflight_kwargs())
    assert "spend_admission:spend_admission_lock_permission_unsafe" in (
        unsafe.value.preflight["blockers"]
    )
