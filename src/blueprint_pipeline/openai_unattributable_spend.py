"""Close an OpenAI-backed family whose per-run cost can never be attributed.

``OpenAIProjectCandidateCostAuthority`` is the only path to an official OpenAI
charge, and it reserves against a *pre-run* zero baseline::

    baseline = self.client.snapshot(start_time=..., end_time=...)
    if float(baseline["total_cost_usd"]) != 0.0:
        raise OpenAICostAuthorityError("openai_cost_scope_baseline_not_zero")

A run that already executed without such a reservation therefore cannot be
given an official per-run cost afterwards: by the time anyone asks, the window
is no longer zero.  This is unlike the Vast pending-billing case in
``semantic_teacher_pending_spend``, where the charge is merely late and will
arrive.  Here it will not arrive at all.

The only honest closure is to reserve the full authority cap against the next
authority, name the structural cause, and refuse to present the result as a
final cost.  This module never estimates, never accepts a candidate-reported
figure, and never sets ``cost_is_final``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


RESERVATION_SCHEMA_VERSION = "openai_unattributable_spend_reservation.v1"
MANIFEST_SCHEMA_VERSION = "inference_reservation_manifest.v1"

# The two structural causes. Both mean "no official per-run cost exists", but
# they have different remediations, so a reader must not have to guess which.
ATTRIBUTION_UNAVAILABLE_REASONS = frozenset(
    {
        # No reservation preceded the spend, so the attribution window was
        # already non-zero. Remediation: reserve before spending next time.
        "no_pre_run_zero_baseline",
        # The project/API key was shared with other work in the same window, so
        # even a retrospective query cannot isolate this run's charge.
        # Remediation: give the lane its own exclusively-scoped key.
        "shared_api_key_scope",
    }
)

_REMEDIATION = {
    "no_pre_run_zero_baseline": (
        "Reserve through OpenAIProjectCandidateCostAuthority before the spend, "
        "so the attribution window starts at zero."
    ),
    "shared_api_key_scope": (
        "Give this lane an exclusively-scoped OpenAI project and API key so a "
        "windowed organization-costs query isolates exactly this run."
    ),
}

_EXPLANATION = (
    "No official per-run OpenAI cost exists for this family. Official "
    "attribution requires a pre-run zero-baseline reservation against an "
    "exclusively-scoped project and API key; this run did not have one, so the "
    "charge cannot be isolated after the fact. The full authority cap is "
    "reserved instead. This is a conservative upper bound, not a measured cost."
)


class OpenAIUnattributableSpendError(ValueError):
    """Fail-closed refusal to close an OpenAI family dishonestly."""


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _read_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_manifest_missing"
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_manifest_unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_manifest_unreadable"
        )
    return source, value


def materialize_openai_unattributable_spend(
    *,
    family_id: str,
    run_id: str,
    reservation_manifest_path: str | Path,
    authority_cap_usd: float,
    model_id: str,
    attribution_unavailable_reason: str,
    output_path: str | Path,
    reserved_spend_usd: float | None = None,
) -> dict[str, Any]:
    """Reserve the full cap for a run whose official cost cannot be obtained.

    ``reserved_spend_usd`` exists only so a caller that tries to supply a
    smaller, estimated figure is refused rather than silently obeyed.
    """

    if attribution_unavailable_reason not in ATTRIBUTION_UNAVAILABLE_REASONS:
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_reason_invalid"
        )

    manifest_file, manifest = _read_manifest(reservation_manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_manifest_schema_invalid"
        )
    if manifest.get("inference_reservation_manifest_digest") != canonical_digest(
        manifest, digest_field="inference_reservation_manifest_digest"
    ):
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_manifest_digest_invalid"
        )
    if str(manifest.get("run_id") or "") != str(run_id):
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_run_mismatch"
        )

    in_flight = manifest.get("in_flight_unknown_count")
    if not isinstance(in_flight, int) or isinstance(in_flight, bool) or in_flight != 0:
        # An outstanding call may still be accruing. No number is defensible.
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_in_flight_reservation"
        )

    reserved = _finite_positive(manifest.get("reserved_max_cost_usd"))
    cap = _finite_positive(authority_cap_usd)
    if reserved is None or cap is None:
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_cap_invalid"
        )
    if reserved > cap:
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_exceeds_authority"
        )
    if reserved_spend_usd is not None and float(reserved_spend_usd) != reserved:
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_estimate_forbidden"
        )
    if not str(family_id).strip() or not str(model_id).strip():
        raise OpenAIUnattributableSpendError(
            "openai_unattributable_spend_identity_invalid"
        )

    receipt: dict[str, Any] = {
        "schema_version": RESERVATION_SCHEMA_VERSION,
        "status": "official_attribution_unavailable_conservative_reserve",
        "family_id": str(family_id),
        "run_id": str(run_id),
        "model_id": str(model_id),
        "provider_id": "openai",
        "reserved_spend_usd": reserved,
        "authority_cap_usd": cap,
        "cost_is_final": False,
        "official_per_run_cost_available": False,
        "candidate_reported_cost_accepted": False,
        "attribution_unavailable_reason": str(attribution_unavailable_reason),
        "explanation": _EXPLANATION,
        "remediation": _REMEDIATION[str(attribution_unavailable_reason)],
        "reservation_manifest": {
            "path": str(manifest_file),
            "size_bytes": manifest_file.stat().st_size,
            "manifest_digest": str(
                manifest["inference_reservation_manifest_digest"]
            ),
            "reservation_count": manifest.get("reservation_count"),
        },
        "reserved_at": utc_now_iso(),
        "proof_effect": "none",
    }
    receipt["reservation_digest"] = canonical_digest(
        receipt, digest_field="reservation_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reservation-manifest", required=True)
    parser.add_argument("--authority-cap-usd", required=True, type=float)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--attribution-unavailable-reason",
        required=True,
        choices=sorted(ATTRIBUTION_UNAVAILABLE_REASONS),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = materialize_openai_unattributable_spend(
        family_id=args.family_id,
        run_id=args.run_id,
        reservation_manifest_path=args.reservation_manifest,
        authority_cap_usd=args.authority_cap_usd,
        model_id=args.model_id,
        attribution_unavailable_reason=args.attribution_unavailable_reason,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ATTRIBUTION_UNAVAILABLE_REASONS",
    "MANIFEST_SCHEMA_VERSION",
    "RESERVATION_SCHEMA_VERSION",
    "OpenAIUnattributableSpendError",
    "materialize_openai_unattributable_spend",
]


if __name__ == "__main__":
    raise SystemExit(main())
