from __future__ import annotations

import fcntl
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_activation_queue import (
    ENVELOPE_SCHEMA_VERSION,
    IDENTITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from blueprint_pipeline.task_evaluation_launch_activation_worker import (
    _acquire_processing_lease,
)
from blueprint_pipeline.task_evaluation_stale_activation_reconciliation import (
    APPLY_ACKNOWLEDGEMENT,
    StaleActivationReconciliationError,
    apply_stale_activation_reconciliation_plan,
    build_stale_activation_reconciliation_plan,
)


COMMIT = "a" * 40
PREPARATION_REQUEST_DIGEST = "sha256:" + "b" * 64
PREPARATION_RESULT_DIGEST = "sha256:" + "c" * 64


def _write_sealed(path: Path, value: dict, *, digest_field: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _envelope(*, activation_id: str, submitted_at: datetime) -> dict:
    request_digest = "sha256:" + (
        "1" * 64 if activation_id.endswith("-a") else "2" * 64
    )
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "request_digest": request_digest,
        "request": {
            "activation_id": activation_id,
            "team_namespace": "blueprint-adp",
            "lane": "task_evaluation_scene_configuration",
            "expected_production_commit": COMMIT,
            "preparation": {
                "preparation_id": "scene-839873-preparation",
                "request_digest": PREPARATION_REQUEST_DIGEST,
                "result_digest": PREPARATION_RESULT_DIGEST,
            },
        },
        "submitted_by": "webapp",
        "submitted_at_iso": submitted_at.isoformat(),
        "provider_mutation_performed_inside_intake": False,
        "catalog_mutation_performed_inside_intake": False,
        "standing_authorization_published_inside_intake": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }


def _provider_zero(path: Path, *, now: datetime) -> None:
    value = {
        "schema_version": "gpu_spend_guard.v1",
        "status": "passed",
        "generated_at": (now - timedelta(seconds=5)).isoformat(),
        "blockers": [],
        "provider_zero_verified": True,
        "live_instance_count": 0,
        "total_burn_per_hour_usd": 0,
        "provider_zero": {
            "status": "verified",
            "blockers": [],
            "global_live_instance_count": 0,
            "global_total_burn_per_hour_usd": 0,
            "required_provider_ids": ["aws", "digitalocean", "runpod", "vast"],
        },
        "inventory_results": [
            {
                "provider": provider,
                "required": True,
                "status": "succeeded",
                "row_count": 0,
                "blockers": [],
            }
            for provider in ("aws", "digitalocean", "runpod", "vast")
        ],
    }
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict:
    now = datetime(2026, 8, 28, 23, 30, tzinfo=timezone.utc)
    queue = tmp_path / "state/task-evaluation-launch-activations"
    for child in (
        "pending",
        "processing",
        "prepared",
        "blocked",
        "identities",
        "results",
        "reconciliations",
    ):
        (queue / child).mkdir(parents=True, exist_ok=True)
    activation_base = tmp_path / "inputs/launch-activations"
    activation_base.mkdir(parents=True)
    references = tmp_path / "references"
    references.mkdir()
    target_id = "scene-839873-activation-a"
    sibling_id = "scene-839873-activation-b"
    target_value = _envelope(
        activation_id=target_id, submitted_at=now - timedelta(hours=4)
    )
    target_name = (
        f"{target_id}-{target_value['request_digest'].removeprefix('sha256:')}.json"
    )
    target = queue / "processing" / target_name
    target_value = _write_sealed(target, target_value, digest_field="envelope_digest")
    sibling_value = _envelope(
        activation_id=sibling_id, submitted_at=now - timedelta(hours=3)
    )
    sibling_name = (
        f"{sibling_id}-{sibling_value['request_digest'].removeprefix('sha256:')}.json"
    )
    sibling = queue / "blocked" / sibling_name
    sibling_value = _write_sealed(
        sibling, sibling_value, digest_field="envelope_digest"
    )
    sibling_result = _write_sealed(
        queue / "results" / sibling_name,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "blocked",
            "activation_id": sibling_id,
            "blockers": ["launch_activation_preparation_graph_blocked"],
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "observed_at_iso": (now - timedelta(hours=2)).isoformat(),
            "result_digest": "",
        },
        digest_field="result_digest",
    )
    identity = _write_sealed(
        queue / "identities" / f"{target_id}.json",
        {
            "schema_version": IDENTITY_SCHEMA_VERSION,
            "activation_id": target_id,
            "request_digest": target_value["request_digest"],
            "identity_digest": "",
        },
        digest_field="identity_digest",
    )
    provider_zero = tmp_path / "provider-zero.json"
    _provider_zero(provider_zero, now=now)
    return {
        "now": now,
        "queue": queue,
        "activation_base": activation_base,
        "references": references,
        "target": target,
        "target_value": target_value,
        "sibling": sibling,
        "sibling_value": sibling_value,
        "sibling_result": sibling_result,
        "identity": identity,
        "provider_zero": provider_zero,
    }


def _plan(fixture: dict) -> dict:
    return build_stale_activation_reconciliation_plan(
        target_envelope=fixture["target"],
        terminal_sibling_envelope=fixture["sibling"],
        activation_queue_root=fixture["queue"],
        activation_base_root=fixture["activation_base"],
        reference_roots=[fixture["references"]],
        provider_zero_report=fixture["provider_zero"],
        now=fixture["now"],
    )


def test_reconciles_exact_crash_stranded_processing_envelope_without_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    plan = _plan(fixture)
    assert plan["status"] == "ready_to_reconcile"
    assert plan["target"]["envelope_digest"] == fixture["target_value"][
        "envelope_digest"
    ]
    assert plan["terminal_sibling"]["result"]["result_digest"] == fixture[
        "sibling_result"
    ]["result_digest"]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    target_bytes = fixture["target"].read_bytes()
    _FrozenDateTime._value = fixture["now"]
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_stale_activation_reconciliation.datetime",
        _FrozenDateTime,
    )
    receipt = apply_stale_activation_reconciliation_plan(
        dry_run_plan_path=plan_path,
        acknowledgement=APPLY_ACKNOWLEDGEMENT,
        receipt_out=fixture["queue"] / "reconciliations" / fixture["target"].name,
    )

    blocked = fixture["queue"] / "blocked" / fixture["target"].name
    result_path = fixture["queue"] / "results" / fixture["target"].name
    assert blocked.read_bytes() == target_bytes
    assert not fixture["target"].exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert result["provider_mutation_performed"] is False
    assert result["paid_execution_requested"] is False
    assert receipt["status"] == "reconciled_terminal_blocked"
    assert receipt["evidence_deletion_performed"] is False
    assert fixture["sibling"].is_file()
    assert (fixture["queue"] / "results" / fixture["sibling"].name).is_file()
    assert (fixture["queue"] / "identities" / "scene-839873-activation-a.json").is_file()


class _FrozenDateTime(datetime):
    _value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls._value if tz is not None else cls._value.replace(tzinfo=None)


def test_refuses_live_worker_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    plan = _plan(fixture)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    _FrozenDateTime._value = fixture["now"]
    monkeypatch.setattr(
        "blueprint_pipeline.task_evaluation_stale_activation_reconciliation.datetime",
        _FrozenDateTime,
    )
    with fixture["target"].open("rb") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            StaleActivationReconciliationError,
            match="stale_activation_reconciliation_worker_lease_active",
        ):
            apply_stale_activation_reconciliation_plan(
                dry_run_plan_path=plan_path,
                acknowledgement=APPLY_ACKNOWLEDGEMENT,
                receipt_out=(
                    fixture["queue"] / "reconciliations" / fixture["target"].name
                ),
            )


def test_activation_worker_processing_lease_excludes_reconciler(
    tmp_path: Path,
) -> None:
    claimed = tmp_path / "processing.json"
    claimed.write_text("{}\n", encoding="utf-8")
    lease = _acquire_processing_lease(claimed)
    try:
        with claimed.open("rb") as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
    finally:
        lease.close()


def test_refuses_sibling_for_different_preparation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    sibling = json.loads(fixture["sibling"].read_text(encoding="utf-8"))
    sibling["request"]["preparation"]["result_digest"] = "sha256:" + "d" * 64
    _write_sealed(fixture["sibling"], sibling, digest_field="envelope_digest")
    with pytest.raises(
        StaleActivationReconciliationError,
        match="stale_activation_reconciliation_sibling_binding_mismatch",
    ):
        _plan(fixture)


def test_refuses_downstream_reference_and_nonzero_provider(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    (fixture["references"] / "live.json").write_text(
        json.dumps({"activation_id": "scene-839873-activation-a"}),
        encoding="utf-8",
    )
    with pytest.raises(
        StaleActivationReconciliationError,
        match="stale_activation_reconciliation_downstream_reference_present",
    ):
        _plan(fixture)
    (fixture["references"] / "live.json").unlink()
    zero = json.loads(fixture["provider_zero"].read_text(encoding="utf-8"))
    zero["live_instance_count"] = 1
    fixture["provider_zero"].write_text(
        json.dumps(zero, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(
        StaleActivationReconciliationError,
        match="stale_activation_reconciliation_provider_zero_invalid",
    ):
        _plan(fixture)
