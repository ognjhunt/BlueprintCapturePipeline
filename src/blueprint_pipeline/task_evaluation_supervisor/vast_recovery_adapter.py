"""Vast implementation of the Task Evaluation Supervisor recovery protocol.

This module deliberately wraps Blueprint's existing authorized Vast WAM runner.
It is not a provider launcher: the wrapped runner retains admission, staging,
spend, watchdog, and provider-zero teardown authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from ..paid_resource_admission import PaidResourceAdmissionGrant
from ..vast_wam_authorized_runner import (
    DEFAULT_WAM_PUBLIC_IMAGE,
    run_vast_wam_authorized_runner,
)
from ..vast_provider_adapter import (
    VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
    VAST_TEARDOWN_SCHEMA_VERSION,
)


VAST_RECOVERY_ADAPTER_ID = "vast_wam_recovery"
VAST_RECOVERY_RESOURCE_CLASS = "vast_provider_adapter"
VAST_RECOVERY_ACTIONS = frozenset({"bounded_provider_retry"})
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class VastRecoveryAdapterError(ValueError):
    """Raised before Vast mutation when the frozen recovery binding is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _failure_type(blockers: Sequence[Any]) -> str:
    text = " ".join(str(item).lower() for item in blockers)
    if "capacity" in text or "offer" in text or "inventory" in text:
        return "provider_capacity"
    if "heartbeat" in text or "container" in text or "startup" in text:
        return "container_startup"
    if "timeout" in text or "ttl" in text:
        return "timeout"
    if (
        "admission" in text
        or "authorized" in text
        or "gate" in text
        or "preflight" in text
    ):
        return "infrastructure_admission"
    return "provider_failure"


@dataclass
class VastWAMRecoveryAdapter:
    """Bind one pre-authorized supervisor recovery action to the Vast runner."""

    job_dir: Path
    bundle_path: Path
    immutable_commit_sha: str
    immutable_input_digests: tuple[str, ...]
    paid_resource_admission_grant: PaidResourceAdmissionGrant | None
    provider_bundle_url_file: Path
    provider_output_put_url_file: Path
    provider_output_get_url_file: Path
    session_budget_ledger: Path
    public_image: str = DEFAULT_WAM_PUBLIC_IMAGE
    max_hourly_rate: float = 1.0
    allowed_action_ids: tuple[str, ...] = ("bounded_provider_retry",)
    runner: Callable[..., Mapping[str, Any]] = run_vast_wam_authorized_runner
    provider_id: str = VAST_RECOVERY_ADAPTER_ID
    paid_resource_class: str = VAST_RECOVERY_RESOURCE_CLASS
    _last_runner_result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _last_attempt_dir: Path | None = field(default=None, init=False, repr=False)
    _attempt_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.job_dir = Path(self.job_dir).expanduser().resolve()
        self.bundle_path = Path(self.bundle_path).expanduser().resolve()
        self.provider_bundle_url_file = Path(self.provider_bundle_url_file).expanduser().resolve()
        self.provider_output_put_url_file = Path(
            self.provider_output_put_url_file
        ).expanduser().resolve()
        self.provider_output_get_url_file = Path(
            self.provider_output_get_url_file
        ).expanduser().resolve()
        self.session_budget_ledger = Path(self.session_budget_ledger).expanduser().resolve()
        if not self.bundle_path.is_file():
            raise VastRecoveryAdapterError("vast_recovery_bundle_missing")
        if not _COMMIT_SHA.fullmatch(self.immutable_commit_sha):
            raise VastRecoveryAdapterError("vast_recovery_commit_invalid")
        normalized = tuple(sorted(set(self.immutable_input_digests)))
        if (
            not normalized
            or normalized != self.immutable_input_digests
            or any(not _SHA256_DIGEST.fullmatch(row) for row in normalized)
        ):
            raise VastRecoveryAdapterError("vast_recovery_input_digests_not_canonical")
        bundle_digest = _sha256_file(self.bundle_path)
        if bundle_digest not in normalized:
            raise VastRecoveryAdapterError("vast_recovery_bundle_digest_not_bound")
        normalized_actions = tuple(sorted(set(self.allowed_action_ids)))
        if (
            not normalized_actions
            or normalized_actions != self.allowed_action_ids
            or not set(normalized_actions).issubset(VAST_RECOVERY_ACTIONS)
        ):
            raise VastRecoveryAdapterError("vast_recovery_action_allowlist_invalid")
        if not math.isfinite(self.max_hourly_rate) or self.max_hourly_rate <= 0:
            raise VastRecoveryAdapterError("vast_recovery_hourly_rate_invalid")

    def execute(
        self,
        *,
        action_id: str,
        immutable_commit_sha: str,
        immutable_input_digests: Sequence[str],
        timeout_seconds: float,
        max_cost_usd: float,
    ) -> Mapping[str, Any]:
        if action_id not in self.allowed_action_ids:
            raise VastRecoveryAdapterError("vast_recovery_action_not_bound")
        if immutable_commit_sha != self.immutable_commit_sha:
            raise VastRecoveryAdapterError("vast_recovery_commit_not_bound")
        if tuple(sorted(immutable_input_digests)) != self.immutable_input_digests:
            raise VastRecoveryAdapterError("vast_recovery_inputs_not_bound")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise VastRecoveryAdapterError("vast_recovery_timeout_invalid")
        if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
            raise VastRecoveryAdapterError("vast_recovery_cost_invalid")

        self._attempt_index += 1
        attempt_dir = self.job_dir / f"attempt-{self._attempt_index:04d}"
        if attempt_dir.exists():
            raise VastRecoveryAdapterError("vast_recovery_attempt_dir_exists")
        if timeout_seconds < 60:
            self._last_runner_result = {
                "status": "blocked",
                "paid_launch_attempted": False,
                "independent_watchdog_close": {"status": "cancelled_no_allocation"},
            }
            self._last_attempt_dir = attempt_dir
            return {
                "status": "failed",
                "failure_type": "authorization_window_too_short",
                "retryable": False,
                "provider_id": self.provider_id,
                "provider_execution_started": False,
                "cost_usd": 0.0,
                "immutable_commit_sha": immutable_commit_sha,
                "immutable_input_digests": list(self.immutable_input_digests),
                "scientific_validity_proven": False,
                "physical_success_proven": False,
            }
        max_live_minutes = max(1, int(timeout_seconds // 60))
        result = dict(
            self.runner(
                job_dir=attempt_dir,
                bundle_path=self.bundle_path,
                provider_bundle_url_file=self.provider_bundle_url_file,
                provider_output_put_url_file=self.provider_output_put_url_file,
                provider_output_get_url_file=self.provider_output_get_url_file,
                session_budget_ledger=self.session_budget_ledger,
                allow_paid_vast_launch=True,
                max_hourly_rate=self.max_hourly_rate,
                target_spend_usd=max_cost_usd,
                hard_cap_usd=max_cost_usd,
                allow_target_spend_overrun=False,
                max_live_minutes=max_live_minutes,
                session_max_live_minutes=max_live_minutes,
                startup_timeout_seconds=max(1, int(timeout_seconds)),
                public_image=self.public_image,
                paid_resource_admission_grant=self.paid_resource_admission_grant,
                require_independent_watchdog=True,
                retain_instance_on_runtime_failure=False,
                provider_bundle_kind="wam",
            )
        )
        self._last_runner_result = result
        self._last_attempt_dir = attempt_dir
        adapter_result = _read_mapping(attempt_dir / "vast_provider_adapter_result.json")
        adapter_result_schema_valid = (
            adapter_result.get("schema_version")
            == VAST_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION
        )
        transport_completed = (
            result.get("status") == "completed"
            and adapter_result_schema_valid
            and adapter_result.get("status") == "completed"
        )
        raw: dict[str, Any] = {
            "status": "completed" if transport_completed else "failed",
            "failure_type": _failure_type(result.get("blockers") or []),
            "runner_manifest_path": str(
                attempt_dir / "vast_wam_authorized_runner_manifest.json"
            ),
            "provider_result_path": str(
                attempt_dir / "vast_provider_adapter_result.json"
            ),
            "provider_id": self.provider_id,
            "immutable_commit_sha": immutable_commit_sha,
            "immutable_input_digests": list(self.immutable_input_digests),
            "provider_execution_started": result.get("paid_launch_attempted") is True,
            "provider_transport_completed": transport_completed,
            "provider_result_schema_valid": adapter_result_schema_valid,
            "scientific_validity_proven": False,
            "physical_success_proven": False,
        }
        if "estimated_cost_usd" in adapter_result:
            raw["cost_usd"] = adapter_result["estimated_cost_usd"]
            raw["cost_basis"] = "vast_runtime_estimate"
        elif result.get("paid_launch_attempted") is False:
            raw["cost_usd"] = 0.0
        return raw

    def teardown(self) -> Mapping[str, Any]:
        if self._last_runner_result is None or self._last_attempt_dir is None:
            return {
                "status": "failed",
                "provider_zero": False,
                "reason": "vast_recovery_not_executed",
            }
        result = self._last_runner_result
        if result.get("paid_launch_attempted") is False:
            watchdog_status = dict(result.get("independent_watchdog_close") or {}).get(
                "status"
            )
            provider_zero = watchdog_status in {"not_required", "cancelled_no_allocation"}
            return {
                "status": "completed" if provider_zero else "failed",
                "provider_zero": provider_zero,
                "reason": "vast_recovery_blocked_before_allocation",
            }
        attempt_dir = self._last_attempt_dir
        teardown = _read_mapping(attempt_dir / "vast_teardown_manifest.json")
        adapter_result = _read_mapping(attempt_dir / "vast_provider_adapter_result.json")
        watchdog_close = dict(result.get("independent_watchdog_close") or {})
        teardown_contract_valid = (
            teardown.get("schema_version") == VAST_TEARDOWN_SCHEMA_VERSION
            and teardown.get("status") == "completed"
            and teardown.get("runner_gpu_teardown_completed") is True
            and teardown.get("retention_authorized") is False
        )
        zero_spend = teardown.get("continuing_spend_from_this_run") is False
        watchdog_terminal = (
            watchdog_close.get("status") == "provider_terminal"
            and watchdog_close.get("provider_absence_confirmed") is True
        ) or (
            watchdog_close.get("status") == "cancelled_no_allocation"
            and adapter_result.get("provider_create_attempted") is False
        )
        provider_zero = teardown_contract_valid and zero_spend and watchdog_terminal
        return {
            "status": "completed" if provider_zero else "failed",
            "provider_zero": provider_zero,
            "teardown_manifest_path": str(attempt_dir / "vast_teardown_manifest.json"),
            "watchdog_status": watchdog_close.get("status"),
            "teardown_contract_valid": teardown_contract_valid,
            "continuing_spend_from_this_run": teardown.get(
                "continuing_spend_from_this_run"
            ),
        }


__all__ = [
    "VAST_RECOVERY_ACTIONS",
    "VAST_RECOVERY_ADAPTER_ID",
    "VAST_RECOVERY_RESOURCE_CLASS",
    "VastRecoveryAdapterError",
    "VastWAMRecoveryAdapter",
]
