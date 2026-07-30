from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.task_evaluation_supervisor.vast_recovery_adapter import (
    VastRecoveryAdapterError,
    VastWAMRecoveryAdapter,
)
from blueprint_pipeline.task_evaluation_supervisor.phase2_artifacts import (
    authorization_receipt,
    authorization_request,
)
from blueprint_pipeline.task_evaluation_supervisor.recovery import (
    PreauthorizedRecoveryController,
    PreauthorizedRecoveryPolicy,
    RecoveryControlError,
)


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _grant():
    return require_paid_resource_admission(
        build_paid_lane_admission(resource_class="vast_provider_adapter"),
        resource_class="vast_provider_adapter",
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )


def _adapter(tmp_path: Path, runner, *, digest: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"frozen-bundle")
    paths = {}
    for name in ("bundle-url", "put-url", "get-url"):
        path = tmp_path / f"{name}.txt"
        path.write_text("https://objects.example/secret", encoding="utf-8")
        paths[name] = path
    return VastWAMRecoveryAdapter(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        immutable_commit_sha="a" * 40,
        immutable_input_digests=(digest or _digest(bundle),),
        paid_resource_admission_grant=_grant(),
        provider_bundle_url_file=paths["bundle-url"],
        provider_output_put_url_file=paths["put-url"],
        provider_output_get_url_file=paths["get-url"],
        session_budget_ledger=tmp_path / "budget.json",
        runner=runner,
    )


def _controller(adapter: VastWAMRecoveryAdapter, *, wall_clock) -> PreauthorizedRecoveryController:
    request = authorization_request(
        run_id="vast-supervisor-run",
        tool_id="execute_preauthorized_recovery",
        reason="Permit one bounded Vast recovery.",
        requested_max_cost_usd=0.2,
        requested_ttl_seconds=120,
        immutable_input_digests=adapter.immutable_input_digests,
        requested_retry_count=1,
        requested_provider_ids=[adapter.provider_id],
        requested_action_ids=["bounded_provider_retry"],
    )
    receipt = authorization_receipt(
        request=request,
        operator_id="runtime-owner",
        approved=True,
        granted_max_cost_usd=0.2,
        granted_ttl_seconds=120,
        granted_retry_count=1,
        issued_at="2026-07-30T10:00:00Z",
        expires_at="2026-07-30T10:02:00Z",
        granted_provider_ids=[adapter.provider_id],
        granted_action_ids=["bounded_provider_retry"],
    )
    policy = PreauthorizedRecoveryPolicy(
        run_id="vast-supervisor-run",
        authorization_receipt=receipt,
        immutable_commit_sha=adapter.immutable_commit_sha,
        immutable_input_digests=adapter.immutable_input_digests,
        allowed_provider_ids=(adapter.provider_id,),
        allowed_action_ids=("bounded_provider_retry",),
        watchdog_seconds=120,
    )
    return PreauthorizedRecoveryController(
        policy,
        [adapter],
        wall_clock=wall_clock,
    )


def test_vast_recovery_adapter_reuses_authorized_runner_and_proves_zero(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def runner(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        job_dir = Path(kwargs["job_dir"])
        job_dir.mkdir(parents=True)
        (job_dir / "vast_provider_adapter_result.json").write_text(
            json.dumps(
                {
                    "schema_version": "vast_provider_adapter_result.v1",
                    "status": "completed",
                    "estimated_cost_usd": 0.12,
                }
            ),
            encoding="utf-8",
        )
        (job_dir / "vast_teardown_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "vast_teardown_manifest.v1",
                    "status": "completed",
                    "runner_gpu_teardown_completed": True,
                    "continuing_spend_from_this_run": False,
                    "retention_authorized": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "blockers": [],
            "paid_launch_attempted": True,
            "independent_watchdog_close": {
                "status": "provider_terminal",
                "provider_absence_confirmed": True,
            },
        }

    adapter = _adapter(tmp_path, runner)
    result = adapter.execute(
        action_id="bounded_provider_retry",
        immutable_commit_sha="a" * 40,
        immutable_input_digests=adapter.immutable_input_digests,
        timeout_seconds=119,
        max_cost_usd=0.5,
    )

    assert result["status"] == "completed"
    assert result["cost_usd"] == pytest.approx(0.12)
    assert result["scientific_validity_proven"] is False
    assert result["physical_success_proven"] is False
    assert captured["allow_paid_vast_launch"] is True
    assert captured["require_independent_watchdog"] is True
    assert captured["retain_instance_on_runtime_failure"] is False
    assert captured["target_spend_usd"] == pytest.approx(0.5)
    assert captured["hard_cap_usd"] == pytest.approx(0.5)
    assert captured["max_live_minutes"] == 1
    assert captured["paid_resource_admission_grant"] is adapter.paid_resource_admission_grant
    assert adapter.teardown() == {
        "status": "completed",
        "provider_zero": True,
        "teardown_manifest_path": str(
            tmp_path / "job" / "attempt-0001" / "vast_teardown_manifest.json"
        ),
        "watchdog_status": "provider_terminal",
        "teardown_contract_valid": True,
        "continuing_spend_from_this_run": False,
    }


def test_vast_recovery_adapter_treats_preallocation_block_as_provider_zero(
    tmp_path: Path,
) -> None:
    adapter = _adapter(
        tmp_path,
        lambda **_kwargs: {
            "status": "blocked",
            "blockers": ["paid_vast_launch_preflight_blocked"],
            "paid_launch_attempted": False,
            "independent_watchdog_close": {"status": "not_required"},
        },
    )
    result = adapter.execute(
        action_id="bounded_provider_retry",
        immutable_commit_sha="a" * 40,
        immutable_input_digests=adapter.immutable_input_digests,
        timeout_seconds=60,
        max_cost_usd=0.2,
    )
    assert result["status"] == "failed"
    assert result["failure_type"] == "infrastructure_admission"
    assert result["cost_usd"] == 0.0
    assert adapter.teardown()["provider_zero"] is True


def test_vast_recovery_adapter_rejects_unbound_bundle_and_action_before_runner(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {}

    with pytest.raises(VastRecoveryAdapterError, match="bundle_digest_not_bound"):
        _adapter(tmp_path, runner, digest="sha256:" + "0" * 64)

    adapter = _adapter(tmp_path / "valid", runner)
    with pytest.raises(VastRecoveryAdapterError, match="action_not_bound"):
        adapter.execute(
            action_id="signed_code_overlay",
            immutable_commit_sha="a" * 40,
            immutable_input_digests=adapter.immutable_input_digests,
            timeout_seconds=60,
            max_cost_usd=0.2,
        )
    assert calls == []


def test_vast_recovery_adapter_refuses_false_teardown_proof(tmp_path: Path) -> None:
    def runner(**kwargs: Any) -> dict[str, Any]:
        job_dir = Path(kwargs["job_dir"])
        job_dir.mkdir(parents=True)
        (job_dir / "vast_provider_adapter_result.json").write_text(
            json.dumps({"status": "blocked", "estimated_cost_usd": 0.05}),
            encoding="utf-8",
        )
        (job_dir / "vast_teardown_manifest.json").write_text(
            json.dumps({"status": "blocked", "continuing_spend_from_this_run": True}),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "blockers": ["vast_instance_destroy_failed"],
            "paid_launch_attempted": True,
            "independent_watchdog_close": {
                "status": "blocked",
                "provider_absence_confirmed": False,
            },
        }

    adapter = _adapter(tmp_path, runner)
    adapter.execute(
        action_id="bounded_provider_retry",
        immutable_commit_sha="a" * 40,
        immutable_input_digests=adapter.immutable_input_digests,
        timeout_seconds=60,
        max_cost_usd=0.2,
    )
    teardown = adapter.teardown()
    assert teardown["status"] == "failed"
    assert teardown["provider_zero"] is False
    assert teardown["teardown_contract_valid"] is False
    assert teardown["continuing_spend_from_this_run"] is True


def test_vast_recovery_adapter_refuses_launch_when_authority_window_is_under_minimum(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    adapter = _adapter(tmp_path, lambda **kwargs: calls.append(kwargs) or {})

    result = adapter.execute(
        action_id="bounded_provider_retry",
        immutable_commit_sha="a" * 40,
        immutable_input_digests=adapter.immutable_input_digests,
        timeout_seconds=59.9,
        max_cost_usd=0.2,
    )

    assert result == {
        "status": "failed",
        "failure_type": "authorization_window_too_short",
        "retryable": False,
        "provider_id": "vast_wam_recovery",
        "provider_execution_started": False,
        "cost_usd": 0.0,
        "immutable_commit_sha": "a" * 40,
        "immutable_input_digests": list(adapter.immutable_input_digests),
        "scientific_validity_proven": False,
        "physical_success_proven": False,
    }
    assert calls == []
    assert adapter.teardown() == {
        "status": "completed",
        "provider_zero": True,
        "reason": "vast_recovery_blocked_before_allocation",
    }


def test_vast_recovery_adapter_requires_canonical_commit_inputs_and_actions(
    tmp_path: Path,
) -> None:
    def runner(**_kwargs: Any) -> dict[str, Any]:
        return {}

    with pytest.raises(VastRecoveryAdapterError, match="commit_invalid"):
        adapter = _adapter(tmp_path / "commit", runner)
        adapter.immutable_commit_sha = "main"
        adapter.__post_init__()

    valid = _adapter(tmp_path / "actions", runner)
    with pytest.raises(VastRecoveryAdapterError, match="action_allowlist_invalid"):
        valid.allowed_action_ids = ("bounded_provider_retry", "bounded_provider_retry")
        valid.__post_init__()


def test_vast_recovery_adapter_rejects_unversioned_success_artifacts(
    tmp_path: Path,
) -> None:
    def runner(**kwargs: Any) -> dict[str, Any]:
        job_dir = Path(kwargs["job_dir"])
        job_dir.mkdir(parents=True)
        (job_dir / "vast_provider_adapter_result.json").write_text(
            json.dumps({"status": "completed", "estimated_cost_usd": 0.1}),
            encoding="utf-8",
        )
        (job_dir / "vast_teardown_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "runner_gpu_teardown_completed": True,
                    "continuing_spend_from_this_run": False,
                    "retention_authorized": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "blockers": [],
            "paid_launch_attempted": True,
            "independent_watchdog_close": {
                "status": "provider_terminal",
                "provider_absence_confirmed": True,
            },
        }

    adapter = _adapter(tmp_path, runner)
    result = adapter.execute(
        action_id="bounded_provider_retry",
        immutable_commit_sha="a" * 40,
        immutable_input_digests=adapter.immutable_input_digests,
        timeout_seconds=60,
        max_cost_usd=0.2,
    )

    assert result["status"] == "failed"
    assert result["provider_transport_completed"] is False
    assert result["provider_result_schema_valid"] is False
    teardown = adapter.teardown()
    assert teardown["status"] == "failed"
    assert teardown["provider_zero"] is False
    assert teardown["teardown_contract_valid"] is False


def test_vast_recovery_adapter_runs_through_preauthorized_controller(
    tmp_path: Path,
) -> None:
    def runner(**kwargs: Any) -> dict[str, Any]:
        job_dir = Path(kwargs["job_dir"])
        job_dir.mkdir(parents=True)
        (job_dir / "vast_provider_adapter_result.json").write_text(
            json.dumps(
                {
                    "schema_version": "vast_provider_adapter_result.v1",
                    "status": "completed",
                    "estimated_cost_usd": 0.1,
                }
            ),
            encoding="utf-8",
        )
        (job_dir / "vast_teardown_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "vast_teardown_manifest.v1",
                    "status": "completed",
                    "runner_gpu_teardown_completed": True,
                    "continuing_spend_from_this_run": False,
                    "retention_authorized": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "blockers": [],
            "paid_launch_attempted": True,
            "independent_watchdog_close": {
                "status": "provider_terminal",
                "provider_absence_confirmed": True,
            },
        }

    adapter = _adapter(tmp_path, runner)
    controller = _controller(
        adapter,
        wall_clock=lambda: datetime(2026, 7, 30, 10, 1, tzinfo=timezone.utc),
    )
    result = controller.execute(
        {
            "action_id": "bounded_provider_retry",
            "provider_id": adapter.provider_id,
            "immutable_commit_sha": adapter.immutable_commit_sha,
            "input_digests": list(adapter.immutable_input_digests),
            "projected_cost_usd": 0.2,
            "failure_type": "provider_capacity",
        }
    )

    assert result["status"] == "completed"
    assert result["actual_cost_usd"] == pytest.approx(0.1)
    assert result["provider_zero_proven"] is True
    assert result["shared_paid_resource_admission_validated"] is True
    assert result["proof_effect"] == "none"


def test_vast_recovery_short_remaining_authority_is_terminal_without_launch(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    adapter = _adapter(tmp_path, lambda **kwargs: calls.append(kwargs) or {})
    controller = _controller(
        adapter,
        wall_clock=lambda: datetime(
            2026,
            7,
            30,
            10,
            1,
            30,
            500_000,
            tzinfo=timezone.utc,
        ),
    )
    arguments = {
        "action_id": "bounded_provider_retry",
        "provider_id": adapter.provider_id,
        "immutable_commit_sha": adapter.immutable_commit_sha,
        "input_digests": list(adapter.immutable_input_digests),
        "projected_cost_usd": 0.2,
        "failure_type": "provider_capacity",
    }

    result = controller.execute(arguments)

    assert result["status"] == "failed"
    assert result["typed_failure"] == {
        "failure_type": "authorization_window_too_short",
        "retryable": False,
    }
    assert result["actual_cost_usd"] == 0.0
    assert result["provider_zero_proven"] is True
    assert calls == []
    with pytest.raises(RecoveryControlError, match="terminal_failure_requires_stop"):
        controller.execute(arguments)
