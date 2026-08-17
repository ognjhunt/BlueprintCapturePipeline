"""Refuse OpenAI spend that could never be officially attributed.

``OpenAIProjectCandidateCostAuthority`` can only produce an official per-run
cost if it reserves *before* the money moves, against an exclusively-scoped
project and API key whose attribution window is still zero::

    if float(baseline["total_cost_usd"]) != 0.0:
        raise OpenAICostAuthorityError("openai_cost_scope_baseline_not_zero")

Once a run has spent without that reservation, no later query can isolate its
charge, and the only honest closure left is the conservative full-cap reserve
in ``openai_unattributable_spend``.  That closure exists because this gate did
not.  This module is the forward fix: a lane asks first, and does not spend if
the answer is no.

The production entry point builds one ``OpenAIOfficialCostRunGate``.  Both
Scene 840920 OpenAI call paths use that object directly, so this is not a
standalone readiness shim that could drift away from the spend path.

The scope attestation is deliberately an operator artifact -- the upstream
validator requires ``issued_by_agent`` to be exactly ``False`` -- so an agent
cannot self-authorize attribution.  Provisioning the dedicated project, key,
and admin key file is a human step this module can check but never perform.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.openai_cost_authority import (
    OpenAICostAuthorityError,
    OpenAIOrganizationCostsClient,
    OpenAIProjectCandidateCostAuthority,
)


RUN_RESERVATION_SCHEMA_VERSION = "openai_official_cost_run_reservation.v1"
RUN_COMPLETION_SCHEMA_VERSION = "openai_official_cost_run_completion.v1"
RUN_SETTLEMENT_SCHEMA_VERSION = "openai_official_cost_run_settlement.v1"


class OpenAIOfficialCostGateError(RuntimeError):
    """Fail-closed refusal to spend OpenAI money that cannot be attributed."""


def _digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 71 or not text.startswith("sha256:"):
        raise OpenAIOfficialCostGateError(f"{field}:invalid_digest")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise OpenAIOfficialCostGateError(f"{field}:invalid_digest") from exc
    return text


def _read_json_file(path: Path, *, code: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise OpenAIOfficialCostGateError(code)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenAIOfficialCostGateError(code) from exc
    if not isinstance(value, dict):
        raise OpenAIOfficialCostGateError(code)
    return value


@dataclass
class OpenAIOfficialCostRunGate:
    """One exact pre-call reservation and post-call official-cost sequence.

    The immediate completion snapshot is official provider evidence, but it is
    deliberately non-terminal because the Costs endpoint may lag.  Only
    :meth:`settle` after the authority's reporting window can set
    ``strict_official_billing_satisfied`` to true.
    """

    authority: OpenAIProjectCandidateCostAuthority
    lane_id: str
    run_id: str
    request_digest: str
    candidate_digest: str
    authorization_receipt_digest: str
    max_cost_usd: float
    output_root: Path

    def __post_init__(self) -> None:
        self.lane_id = str(self.lane_id or "").strip()
        self.run_id = str(self.run_id or "").strip()
        if not self.lane_id or not self.run_id:
            raise OpenAIOfficialCostGateError("openai_official_cost_run_identity_invalid")
        self.request_digest = _digest(self.request_digest, field="request_digest")
        self.candidate_digest = _digest(self.candidate_digest, field="candidate_digest")
        self.authorization_receipt_digest = _digest(
            self.authorization_receipt_digest,
            field="authorization_receipt_digest",
        )
        if isinstance(self.max_cost_usd, bool) or float(self.max_cost_usd) <= 0:
            raise OpenAIOfficialCostGateError("openai_official_cost_run_cap_invalid")
        self.output_root = self.output_root.expanduser().resolve()
        if self.output_root.is_symlink():
            raise OpenAIOfficialCostGateError("openai_official_cost_output_invalid")
        self.output_root.mkdir(parents=True, exist_ok=True)

    @property
    def reservation_path(self) -> Path:
        return self.output_root / "openai_official_cost_run_reservation.v1.json"

    @property
    def completion_path(self) -> Path:
        return self.output_root / "openai_official_cost_run_completion.v1.json"

    @property
    def settlement_path(self) -> Path:
        return self.output_root / "openai_official_cost_run_settlement.v1.json"

    def reserve(self) -> dict[str, Any]:
        """Write the zero-baseline reservation before any OpenAI call."""

        if self.reservation_path.exists() or self.reservation_path.is_symlink():
            raise OpenAIOfficialCostGateError("openai_official_cost_reservation_exists")
        try:
            provider_reservation = dict(
                self.authority.reserve(
                    candidate_id=self.run_id,
                    candidate_evaluation_suite_digest=self.candidate_digest,
                    authorization_receipt_digest=self.authorization_receipt_digest,
                    max_cost_usd=float(self.max_cost_usd),
                )
            )
        except OpenAICostAuthorityError as exc:
            raise OpenAIOfficialCostGateError(
                f"openai_official_cost_reservation_failed:{exc}"
            ) from exc
        value: dict[str, Any] = {
            "schema_version": RUN_RESERVATION_SCHEMA_VERSION,
            "status": "reserved_before_openai_call",
            "lane_id": self.lane_id,
            "run_id": self.run_id,
            "request_digest": self.request_digest,
            "candidate_digest": self.candidate_digest,
            "authorization_receipt_digest": self.authorization_receipt_digest,
            "maximum_cost_usd": float(self.max_cost_usd),
            "provider_reservation": provider_reservation,
            "zero_cost_baseline_confirmed": (
                float(
                    provider_reservation["baseline_cost_snapshot"][
                        "total_cost_usd"
                    ]
                )
                == 0.0
            ),
            "exclusive_scope_attestation_digest": (
                self.authority.scope_attestation_digest
            ),
            "provider_call_performed": False,
            "strict_official_billing_satisfied": False,
            "proof_effect": "none",
            "reservation_receipt_digest": "",
        }
        value["reservation_receipt_digest"] = canonical_digest(
            value, digest_field="reservation_receipt_digest"
        )
        write_json(self.reservation_path, value)
        return value

    def _validated_reservation(self) -> dict[str, Any]:
        value = _read_json_file(
            self.reservation_path,
            code="openai_official_cost_reservation_invalid",
        )
        provider = value.get("provider_reservation")
        if (
            value.get("schema_version") != RUN_RESERVATION_SCHEMA_VERSION
            or value.get("status") != "reserved_before_openai_call"
            or value.get("reservation_receipt_digest")
            != canonical_digest(value, digest_field="reservation_receipt_digest")
            or value.get("lane_id") != self.lane_id
            or value.get("run_id") != self.run_id
            or value.get("request_digest") != self.request_digest
            or value.get("candidate_digest") != self.candidate_digest
            or value.get("authorization_receipt_digest")
            != self.authorization_receipt_digest
            or value.get("zero_cost_baseline_confirmed") is not True
            or value.get("strict_official_billing_satisfied") is not False
            or not isinstance(provider, Mapping)
            or provider.get("candidate_id") != self.run_id
            or provider.get("candidate_evaluation_suite_digest")
            != self.candidate_digest
            or provider.get("authorization_receipt_digest")
            != self.authorization_receipt_digest
        ):
            raise OpenAIOfficialCostGateError(
                "openai_official_cost_reservation_invalid"
            )
        return value

    def complete(
        self,
        *,
        provider_call_performed: bool,
        runtime_result_digest: str | None,
        runtime_exception_type: str | None,
    ) -> dict[str, Any]:
        """Capture an official snapshot immediately after the call terminates."""

        if self.completion_path.exists() or self.completion_path.is_symlink():
            raise OpenAIOfficialCostGateError("openai_official_cost_completion_exists")
        reservation = self._validated_reservation()
        if runtime_result_digest is not None:
            runtime_result_digest = _digest(
                runtime_result_digest, field="runtime_result_digest"
            )
        if not isinstance(provider_call_performed, bool):
            raise OpenAIOfficialCostGateError(
                "openai_official_cost_call_status_invalid"
            )
        provider_reservation = reservation["provider_reservation"]
        try:
            snapshot = self.authority.client.snapshot(
                start_time=int(provider_reservation["attribution_start_time"]),
                end_time=int(provider_reservation["attribution_end_time"]),
            )
        except OpenAICostAuthorityError as exc:
            raise OpenAIOfficialCostGateError(
                f"openai_official_cost_completion_snapshot_failed:{exc}"
            ) from exc
        baseline = float(
            provider_reservation["baseline_cost_snapshot"]["total_cost_usd"]
        )
        observed = float(snapshot["total_cost_usd"]) - baseline
        if observed < 0:
            raise OpenAIOfficialCostGateError(
                "openai_official_cost_completion_delta_invalid"
            )
        value: dict[str, Any] = {
            "schema_version": RUN_COMPLETION_SCHEMA_VERSION,
            "status": "official_cost_reporting_pending",
            "lane_id": self.lane_id,
            "run_id": self.run_id,
            "request_digest": self.request_digest,
            "candidate_digest": self.candidate_digest,
            "authorization_receipt_digest": self.authorization_receipt_digest,
            "reservation_receipt_digest": reservation[
                "reservation_receipt_digest"
            ],
            "cost_reservation_digest": provider_reservation[
                "cost_reservation_digest"
            ],
            "provider_call_performed": provider_call_performed,
            "runtime_result_digest": runtime_result_digest,
            "runtime_exception_type": (
                str(runtime_exception_type) if runtime_exception_type else None
            ),
            "official_completion_snapshot": snapshot,
            "provider_observed_cost_usd": round(observed, 8),
            "cost_is_final": False,
            "strict_official_billing_satisfied": False,
            "reconciliation_not_before": provider_reservation[
                "reconciliation_not_before"
            ],
            "candidate_reported_cost_accepted": False,
            "raw_admin_key_recorded": False,
            "proof_effect": "none",
            "completed_at": utc_now_iso(),
            "completion_receipt_digest": "",
        }
        value["completion_receipt_digest"] = canonical_digest(
            value, digest_field="completion_receipt_digest"
        )
        write_json(self.completion_path, value)
        return value

    def settle(self) -> dict[str, Any]:
        """Query the closed reporting window and write the only terminal receipt."""

        reservation = self._validated_reservation()
        completion = _read_json_file(
            self.completion_path,
            code="openai_official_cost_completion_invalid",
        )
        if (
            completion.get("schema_version") != RUN_COMPLETION_SCHEMA_VERSION
            or completion.get("completion_receipt_digest")
            != canonical_digest(completion, digest_field="completion_receipt_digest")
            or completion.get("reservation_receipt_digest")
            != reservation.get("reservation_receipt_digest")
            or completion.get("lane_id") != self.lane_id
            or completion.get("run_id") != self.run_id
            or completion.get("request_digest") != self.request_digest
            or completion.get("candidate_digest") != self.candidate_digest
        ):
            raise OpenAIOfficialCostGateError(
                "openai_official_cost_completion_invalid"
            )
        provider_settlement = dict(
            self.authority.settle(
                reservation=reservation["provider_reservation"],
                runtime_result=(
                    {"result_digest": completion["runtime_result_digest"]}
                    if completion.get("runtime_result_digest")
                    else None
                ),
                runtime_exception_type=completion.get("runtime_exception_type"),
            )
        )
        final = bool(
            provider_settlement.get("status") == "reconciled"
            and provider_settlement.get("cost_is_final") is True
        )
        value: dict[str, Any] = {
            "schema_version": RUN_SETTLEMENT_SCHEMA_VERSION,
            "status": "reconciled" if final else "reconciliation_required",
            "lane_id": self.lane_id,
            "run_id": self.run_id,
            "request_digest": self.request_digest,
            "candidate_digest": self.candidate_digest,
            "authorization_receipt_digest": self.authorization_receipt_digest,
            "reservation_receipt_digest": reservation[
                "reservation_receipt_digest"
            ],
            "completion_receipt_digest": completion[
                "completion_receipt_digest"
            ],
            "provider_settlement": provider_settlement,
            "actual_cost_usd": provider_settlement.get("actual_cost_usd"),
            "cost_is_final": final,
            "strict_official_billing_satisfied": final,
            "candidate_reported_cost_accepted": False,
            "proof_effect": "none",
            "settled_at": utc_now_iso(),
            "settlement_receipt_digest": "",
        }
        value["settlement_receipt_digest"] = canonical_digest(
            value, digest_field="settlement_receipt_digest"
        )
        write_json(self.settlement_path, value)
        return value


def _read_attestation(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise OpenAIOfficialCostGateError("openai_cost_scope_attestation_missing")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenAIOfficialCostGateError(
            "openai_cost_scope_attestation_unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise OpenAIOfficialCostGateError("openai_cost_scope_attestation_unreadable")
    return value


def _authority(
    *,
    scope_attestation_path: str | Path,
    admin_api_key_file: str | Path,
    project_id: str,
    api_key_id: str,
    provider_id: str,
    paid_resource_class: str,
    transport: Callable[..., Mapping[str, Any]] | None,
    wall_clock: Callable[[], datetime],
) -> OpenAIProjectCandidateCostAuthority:
    client = OpenAIOrganizationCostsClient(
        project_id=project_id,
        api_key_id=api_key_id,
        admin_api_key_file=Path(admin_api_key_file),
        transport=transport,
        wall_clock=wall_clock,
    )
    return OpenAIProjectCandidateCostAuthority(
        client=client,
        scope_attestation=_read_attestation(scope_attestation_path),
        provider_id=provider_id,
        paid_resource_class=paid_resource_class,
        wall_clock=wall_clock,
    )


def build_openai_official_cost_run_gate(
    *,
    scope_attestation_path: str | Path,
    admin_api_key_file: str | Path,
    project_id: str,
    api_key_id: str,
    lane_id: str,
    run_id: str,
    request_digest: str,
    candidate_digest: str,
    authorization_receipt_digest: str,
    max_cost_usd: float,
    output_root: str | Path,
    provider_id: str,
    paid_resource_class: str,
    transport: Callable[..., Mapping[str, Any]] | None = None,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> OpenAIOfficialCostRunGate:
    """Build the shared gate used by the two Scene 840920 OpenAI paths."""

    try:
        authority = _authority(
            scope_attestation_path=scope_attestation_path,
            admin_api_key_file=admin_api_key_file,
            project_id=project_id,
            api_key_id=api_key_id,
            provider_id=provider_id,
            paid_resource_class=paid_resource_class,
            transport=transport,
            wall_clock=wall_clock,
        )
        return OpenAIOfficialCostRunGate(
            authority=authority,
            lane_id=lane_id,
            run_id=run_id,
            request_digest=request_digest,
            candidate_digest=candidate_digest,
            authorization_receipt_digest=authorization_receipt_digest,
            max_cost_usd=max_cost_usd,
            output_root=Path(output_root),
        )
    except (OpenAIOfficialCostGateError, OpenAICostAuthorityError) as exc:
        raise OpenAIOfficialCostGateError(
            f"openai_official_cost_attribution_unavailable:{lane_id}:{exc}"
        ) from exc


def reconcile_openai_official_cost_run(
    *,
    output_root: str | Path,
    scope_attestation_path: str | Path,
    admin_api_key_file: str | Path,
    project_id: str,
    api_key_id: str,
    provider_id: str,
    paid_resource_class: str,
    transport: Callable[..., Mapping[str, Any]] | None = None,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Reopen one completed run and query its now-closed reporting window."""

    root = Path(output_root).expanduser().resolve()
    reservation = _read_json_file(
        root / "openai_official_cost_run_reservation.v1.json",
        code="openai_official_cost_reservation_invalid",
    )
    gate = build_openai_official_cost_run_gate(
        scope_attestation_path=scope_attestation_path,
        admin_api_key_file=admin_api_key_file,
        project_id=project_id,
        api_key_id=api_key_id,
        lane_id=str(reservation.get("lane_id") or ""),
        run_id=str(reservation.get("run_id") or ""),
        request_digest=str(reservation.get("request_digest") or ""),
        candidate_digest=str(reservation.get("candidate_digest") or ""),
        authorization_receipt_digest=str(
            reservation.get("authorization_receipt_digest") or ""
        ),
        max_cost_usd=float(reservation.get("maximum_cost_usd") or 0),
        output_root=root,
        provider_id=provider_id,
        paid_resource_class=paid_resource_class,
        transport=transport,
        wall_clock=wall_clock,
    )
    return gate.settle()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scope-attestation", required=True)
    parser.add_argument("--admin-api-key-file", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--api-key-id", required=True)
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--paid-resource-class", required=True)
    args = parser.parse_args(argv)
    settlement = reconcile_openai_official_cost_run(
        output_root=args.output_root,
        scope_attestation_path=args.scope_attestation,
        admin_api_key_file=args.admin_api_key_file,
        project_id=args.project_id,
        api_key_id=args.api_key_id,
        provider_id=args.provider_id,
        paid_resource_class=args.paid_resource_class,
    )
    print(json.dumps(settlement, indent=2, sort_keys=True))
    return 0 if settlement.get("strict_official_billing_satisfied") is True else 2


__all__ = [
    "RUN_COMPLETION_SCHEMA_VERSION",
    "RUN_RESERVATION_SCHEMA_VERSION",
    "RUN_SETTLEMENT_SCHEMA_VERSION",
    "OpenAIOfficialCostGateError",
    "OpenAIOfficialCostRunGate",
    "build_openai_official_cost_run_gate",
    "reconcile_openai_official_cost_run",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
