"""Fail-closed admission contract for action-conditioned WAM backends.

Motion in a generated clip does not prove that the supplied action caused that
motion.  This contract requires a backend to demonstrate, on a hash-pinned
vendor-native reference asset, that the inference worker parsed and applied the
exact requested actions and that active actions separate from valid no-motion
and shuffled controls across multiple seeds.  A pass admits a later domain
qualification experiment only; it does not prove policy ranking or task success.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .policy_ranking_thesis import canonical_sha256


EVIDENCE_SCHEMA_VERSION = "wam_conditioning_fidelity_evidence.v1"
CERTIFICATE_SCHEMA_VERSION = "wam_conditioning_fidelity_certificate.v1"
REQUIRED_CONDITIONS = ("recorded", "no_motion", "shuffled")
ACCEPTED_ATTESTATION_LOCATIONS = frozenset(
    {
        "inside_inference_worker_after_parse",
        "inside_model_preprocess",
    }
)


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class ConditioningFidelityThresholds:
    """Prospectively frozen minimum evidence for backend admission."""

    minimum_seed_count: int = 4
    minimum_view_count: int = 1
    minimum_action_effect_to_seed_noise_ratio: float = 1.0
    minimum_passing_seed_fraction_per_view: float = 0.75

    def validate(self) -> None:
        if self.minimum_seed_count < 2:
            raise ValueError("conditioning_minimum_seed_count_below_two")
        if self.minimum_view_count < 1:
            raise ValueError("conditioning_minimum_view_count_below_one")
        if (
            not math.isfinite(self.minimum_action_effect_to_seed_noise_ratio)
            or self.minimum_action_effect_to_seed_noise_ratio <= 0.0
        ):
            raise ValueError("conditioning_action_effect_ratio_invalid")
        if not 0.0 < self.minimum_passing_seed_fraction_per_view <= 1.0:
            raise ValueError("conditioning_passing_seed_fraction_invalid")


def _ratio(effect: float, noise: float) -> float:
    if effect < 0.0 or noise < 0.0 or not math.isfinite(effect) or not math.isfinite(noise):
        raise ValueError("conditioning_causal_metric_invalid")
    if noise == 0.0:
        return math.inf if effect > 0.0 else 0.0
    return effect / noise


def assess_wam_conditioning_fidelity(
    evidence: Mapping[str, Any],
    *,
    thresholds: ConditioningFidelityThresholds,
) -> dict[str, Any]:
    """Assess vendor-native transport and causal evidence without outcome labels."""

    thresholds.validate()
    blockers: list[str] = []
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        blockers.append("conditioning_evidence_schema_invalid")

    backend = evidence.get("backend")
    vendor = evidence.get("vendor_reference")
    action_contract = evidence.get("action_contract")
    if not isinstance(backend, Mapping) or not str(backend.get("backend_id") or ""):
        blockers.append("conditioning_backend_identity_missing")
    if not isinstance(backend, Mapping) or not _sha256(backend.get("source_revision")):
        blockers.append("conditioning_backend_source_revision_missing")
    if not isinstance(backend, Mapping) or not _sha256(backend.get("model_revision")):
        blockers.append("conditioning_model_revision_missing")
    if not isinstance(vendor, Mapping) or not str(vendor.get("asset_id") or ""):
        blockers.append("conditioning_vendor_reference_identity_missing")
    if not isinstance(vendor, Mapping) or not _sha256(vendor.get("asset_sha256")):
        blockers.append("conditioning_vendor_reference_hash_missing")
    if not isinstance(vendor, Mapping) or not str(vendor.get("license") or ""):
        blockers.append("conditioning_vendor_reference_license_missing")

    expected_shape: list[int] = []
    effective_parameters_sha256 = ""
    if isinstance(action_contract, Mapping):
        shape = action_contract.get("shape")
        if (
            isinstance(shape, Sequence)
            and not isinstance(shape, (str, bytes, bytearray))
            and len(shape) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape)
        ):
            expected_shape = [int(value) for value in shape]
        effective_parameters_sha256 = str(
            action_contract.get("effective_parameters_sha256") or ""
        )
    if not expected_shape:
        blockers.append("conditioning_action_shape_missing")
    if not _sha256(effective_parameters_sha256):
        blockers.append("conditioning_effective_parameters_hash_missing")

    controls = evidence.get("controls")
    if not isinstance(controls, Mapping) or set(controls) != set(REQUIRED_CONDITIONS):
        blockers.append("conditioning_required_controls_missing")
    else:
        recorded = controls["recorded"]
        no_motion = controls["no_motion"]
        shuffled = controls["shuffled"]
        if not all(isinstance(row, Mapping) and _sha256(row.get("action_sha256")) for row in controls.values()):
            blockers.append("conditioning_control_action_hash_missing")
        else:
            hashes = {str(row["action_sha256"]) for row in controls.values()}
            if len(hashes) != len(REQUIRED_CONDITIONS):
                blockers.append("conditioning_control_actions_not_distinct")
        if not isinstance(no_motion, Mapping) or no_motion.get("valid_identity_rotation") is not True:
            blockers.append("conditioning_no_motion_identity_rotation_invalid")
        if not isinstance(no_motion, Mapping) or no_motion.get("explicit_gripper_hold") is not True:
            blockers.append("conditioning_no_motion_gripper_hold_missing")
        if not isinstance(shuffled, Mapping) or shuffled.get("real_action_permutation") is not True:
            blockers.append("conditioning_shuffled_control_not_real_permutation")
        if not isinstance(recorded, Mapping) or recorded.get("vendor_native_action") is not True:
            blockers.append("conditioning_recorded_action_not_vendor_native")

    attestations = evidence.get("server_action_attestations")
    seed_values: set[int] = set()
    attestation_conditions: dict[int, set[str]] = {}
    if not isinstance(attestations, Sequence) or isinstance(
        attestations, (str, bytes, bytearray)
    ):
        blockers.append("conditioning_server_action_attestations_missing")
    else:
        for row in attestations:
            if not isinstance(row, Mapping):
                blockers.append("conditioning_server_action_attestation_invalid")
                continue
            seed = row.get("seed")
            condition = str(row.get("condition") or "")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                blockers.append("conditioning_attestation_seed_invalid")
                continue
            seed_values.add(seed)
            attestation_conditions.setdefault(seed, set()).add(condition)
            requested = str(row.get("requested_action_sha256") or "")
            parsed = str(row.get("parsed_action_sha256") or "")
            applied = str(row.get("applied_action_sha256") or "")
            if not (_sha256(requested) and requested == parsed == applied):
                blockers.append("conditioning_server_action_hash_mismatch")
            if list(row.get("parsed_action_shape") or []) != expected_shape:
                blockers.append("conditioning_server_action_shape_mismatch")
            if row.get("attestation_location") not in ACCEPTED_ATTESTATION_LOCATIONS:
                blockers.append("conditioning_attestation_not_inside_inference_path")
            if str(row.get("effective_parameters_sha256") or "") != effective_parameters_sha256:
                blockers.append("conditioning_effective_parameters_mismatch")
            if not _sha256(row.get("output_sha256")):
                blockers.append("conditioning_output_hash_missing")
        if len(seed_values) < thresholds.minimum_seed_count:
            blockers.append("conditioning_seed_count_below_threshold")
        if any(attestation_conditions.get(seed) != set(REQUIRED_CONDITIONS) for seed in seed_values):
            blockers.append("conditioning_attestation_condition_matrix_incomplete")

    causal_views = evidence.get("causal_views")
    view_results: list[dict[str, Any]] = []
    if not isinstance(causal_views, Sequence) or isinstance(
        causal_views, (str, bytes, bytearray)
    ):
        blockers.append("conditioning_causal_views_missing")
    else:
        seen_views: set[str] = set()
        for view in causal_views:
            if not isinstance(view, Mapping):
                blockers.append("conditioning_causal_view_invalid")
                continue
            view_id = str(view.get("view_id") or "")
            if not view_id or view_id in seen_views:
                blockers.append("conditioning_causal_view_identity_invalid")
                continue
            seen_views.add(view_id)
            comparisons = view.get("seed_comparisons")
            if not isinstance(comparisons, Sequence) or isinstance(
                comparisons, (str, bytes, bytearray)
            ):
                blockers.append("conditioning_causal_seed_comparisons_missing")
                continue
            passing = 0
            observed_seeds: set[int] = set()
            comparison_rows: list[dict[str, Any]] = []
            for comparison in comparisons:
                if not isinstance(comparison, Mapping):
                    blockers.append("conditioning_causal_seed_comparison_invalid")
                    continue
                seed = comparison.get("seed")
                if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                    blockers.append("conditioning_causal_seed_invalid")
                    continue
                observed_seeds.add(seed)
                try:
                    noise = float(comparison.get("cross_seed_noise"))
                    active_null = float(comparison.get("active_vs_no_motion_distance"))
                    active_shuffled = float(comparison.get("active_vs_shuffled_distance"))
                    null_ratio = _ratio(active_null, noise)
                    shuffled_ratio = _ratio(active_shuffled, noise)
                except (TypeError, ValueError):
                    blockers.append("conditioning_causal_metric_invalid")
                    continue
                passed = bool(
                    null_ratio >= thresholds.minimum_action_effect_to_seed_noise_ratio
                    and shuffled_ratio >= thresholds.minimum_action_effect_to_seed_noise_ratio
                )
                passing += int(passed)
                comparison_rows.append(
                    {
                        "seed": seed,
                        "active_vs_no_motion_to_noise_ratio": null_ratio,
                        "active_vs_shuffled_to_noise_ratio": shuffled_ratio,
                        "passed": passed,
                    }
                )
            if observed_seeds != seed_values:
                blockers.append("conditioning_causal_and_attestation_seeds_mismatch")
            fraction = passing / len(comparison_rows) if comparison_rows else 0.0
            view_passed = bool(
                len(comparison_rows) >= thresholds.minimum_seed_count
                and fraction >= thresholds.minimum_passing_seed_fraction_per_view
            )
            if not view_passed:
                blockers.append(f"conditioning_causal_view_failed:{view_id}")
            view_results.append(
                {
                    "view_id": view_id,
                    "seed_count": len(comparison_rows),
                    "passing_seed_count": passing,
                    "passing_seed_fraction": fraction,
                    "passed": view_passed,
                    "seed_comparisons": comparison_rows,
                }
            )
        if len(view_results) < thresholds.minimum_view_count:
            blockers.append("conditioning_view_count_below_threshold")

    blockers = sorted(set(blockers))
    result: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "status": "passed" if not blockers else "failed",
        "backend_id": str(backend.get("backend_id") or "") if isinstance(backend, Mapping) else "",
        "evidence_sha256": canonical_sha256(dict(evidence)),
        "thresholds": asdict(thresholds),
        "thresholds_sha256": canonical_sha256(asdict(thresholds)),
        "seed_count": len(seed_values),
        "view_results": view_results,
        "server_side_action_attestation_passed": not any(
            blocker.startswith("conditioning_server_action")
            or blocker.startswith("conditioning_attestation")
            or blocker.startswith("conditioning_effective_parameters")
            for blocker in blockers
        ),
        "blockers": blockers,
        "claim_boundary": (
            "vendor-native serving and action-conditioning admission only; not domain "
            "qualification, closed-loop validity, policy ranking, or physical success"
        ),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def validate_conditioning_fidelity_certificate(
    certificate: Mapping[str, Any] | None,
    *,
    backend_id: str,
) -> dict[str, Any]:
    """Validate an immutable pass certificate for a selected WAM arm."""

    if not isinstance(certificate, Mapping):
        raise ValueError("conditioning_fidelity_certificate_required")
    payload = dict(certificate)
    recorded = str(payload.pop("manifest_sha256", ""))
    if not _sha256(recorded) or recorded != canonical_sha256(payload):
        raise ValueError("conditioning_fidelity_certificate_digest_invalid")
    if certificate.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
        raise ValueError("conditioning_fidelity_certificate_schema_invalid")
    if certificate.get("status") != "passed" or certificate.get("blockers") != []:
        raise ValueError("conditioning_fidelity_certificate_not_passed")
    if str(certificate.get("backend_id") or "") != str(backend_id):
        raise ValueError("conditioning_fidelity_certificate_backend_mismatch")
    if certificate.get("server_side_action_attestation_passed") is not True:
        raise ValueError("conditioning_fidelity_server_attestation_not_passed")
    return dict(certificate)


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION",
    "ConditioningFidelityThresholds",
    "EVIDENCE_SCHEMA_VERSION",
    "assess_wam_conditioning_fidelity",
    "validate_conditioning_fidelity_certificate",
]
