"""Promotion gates for the SkyPilot pilot lane — all fail-closed.

SkyPilot may own commodity provisioning for the disposable Vast smoke lane
only. Before any additional traffic shifts (RunPod, warm reuse, retained
sessions, spot recovery, the parallel readiness race), every gate below must
be proven with recorded evidence. Sequential cost-ordered failover is a
different policy than `provider_race`'s concurrent readiness race — passing
these gates widens SkyPilot's lane ownership, it does not retire the racer.
"""

from __future__ import annotations

from typing import Any, Mapping

SKYPILOT_PROMOTION_GATE_SCHEMA_VERSION = "skypilot_promotion_gate.v1"

SKYPILOT_PROMOTION_GATES = (
    "identical_image_digest_and_launch_constraints",
    "hourly_price_cap_enforcement",
    "blueprint_readiness_marker_observed",
    "interruption_cleanup",
    "ambiguous_create_reconciliation",
    "exact_provider_native_instance_identification",
    "target_scoped_and_global_inventory",
    "pending_teardown_v1_closure",
    "provider_api_teardown_proof_and_provider_zero",
    "orphan_recovery_after_skypilot_state_loss",
    "warm_worker_latency_no_regression",
)


def evaluate_skypilot_promotion(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Every gate needs ``{"proven": True, "evidence_path": <non-empty>}``."""

    blockers: list[str] = []
    gates: dict[str, dict[str, Any]] = {}
    for gate in SKYPILOT_PROMOTION_GATES:
        row = evidence.get(gate)
        proven = isinstance(row, Mapping) and row.get("proven") is True
        evidence_path = (
            str(row.get("evidence_path") or "").strip()
            if isinstance(row, Mapping)
            else ""
        )
        if not proven or not evidence_path:
            blockers.append(f"skypilot_promotion_gate_unproven:{gate}")
        gates[gate] = {
            "proven": bool(proven and evidence_path),
            "evidence_path": evidence_path or None,
        }
    return {
        "schema_version": SKYPILOT_PROMOTION_GATE_SCHEMA_VERSION,
        "status": "promotable" if not blockers else "not_promotable",
        "blockers": blockers,
        "gates": gates,
        # Even a full pass only authorizes the pilot lane; each further lane
        # (RunPod, warm reuse, retained sessions, race replacement) needs its
        # own promotion evidence.
        "promoted_scope_allowed": ["vast_disposable_smoke"],
    }


__all__ = [
    "SKYPILOT_PROMOTION_GATE_SCHEMA_VERSION",
    "SKYPILOT_PROMOTION_GATES",
    "evaluate_skypilot_promotion",
]
