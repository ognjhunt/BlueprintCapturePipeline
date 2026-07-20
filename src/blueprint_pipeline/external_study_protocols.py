"""Benchmark-specific admission profiles for frozen external studies.

The profiles intentionally keep SC3, OSCAR/RoboArena, and generic evaluator
studies separate. Structural validation never self-authorizes a public
correlation or real-world policy-ordering claim.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC = "success_rate_difference_pp"

EXTERNAL_STUDY_PROTOCOL_PROFILES = {
    "sc3_eval_v3": {
        "benchmark_family": "sc3_eval",
        "minimum_policy_checkpoint_count": 7,
        "requires_three_view_checkpoint": True,
        "requires_calibrated_inverse_threshold": True,
        "required_metrics": (
            "pearson_success_rate_correlation",
            "spearman_rank_correlation",
            "mean_maximum_rank_violation",
            "abstention_rate",
        ),
    },
    "oscar_roboarena_v2": {
        "benchmark_family": "oscar_roboarena",
        "minimum_policy_checkpoint_count": 7,
        "requires_three_view_checkpoint": False,
        "requires_calibrated_inverse_threshold": False,
        "required_metrics": (
            "pearson_success_rate_correlation",
            "spearman_rank_correlation",
            "mean_maximum_rank_violation",
            OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC,
            "abstention_rate",
        ),
    },
    "generic_evaluator_bounded_v1": {
        "benchmark_family": "generic_evaluator_bounded",
        "minimum_policy_checkpoint_count": 7,
        "requires_three_view_checkpoint": False,
        "requires_calibrated_inverse_threshold": False,
        "required_metrics": ("abstention_rate",),
    },
}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256(value: Any) -> bool:
    text = _string(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact_ref(ref: Mapping[str, Any], *, prefix: str) -> list[str]:
    blockers: list[str] = []
    path_text = _string(ref.get("path"))
    digest = _string(ref.get("sha256")).lower()
    if not path_text:
        blockers.append(f"{prefix}_path_missing")
        return blockers
    path = Path(path_text).expanduser()
    if not path.is_file():
        blockers.append(f"{prefix}_file_missing")
    if not _sha256(digest):
        blockers.append(f"{prefix}_sha256_invalid")
    elif path.is_file() and _file_sha256(path) != digest:
        blockers.append(f"{prefix}_sha256_mismatch")
    return blockers


def _external_study_protocol_profile(
    study: Mapping[str, Any], explicit_profile: str | None
) -> tuple[str, dict[str, Any] | None]:
    profile_id = _string(explicit_profile or study.get("protocol_profile")).strip()
    if not profile_id:
        profile_id = "sc3_eval_v3"
    profile = EXTERNAL_STUDY_PROTOCOL_PROFILES.get(profile_id)
    return profile_id, dict(profile) if profile else None


def validate_external_study(
    study: Mapping[str, Any], *, protocol_profile: str | None = None
) -> dict[str, Any]:
    """Screen a frozen study under one benchmark-specific protocol profile."""

    blockers: list[str] = []
    profile_id, profile = _external_study_protocol_profile(study, protocol_profile)
    if profile is None:
        return {
            "schema_version": "external_study_profile_validation.v2",
            "protocol_profile": profile_id,
            "status": "external_proof_required",
            "external_manual_proof": True,
            "blockers": ["external_study_protocol_profile_unsupported"],
        }
    sc3_profile = profile_id == "sc3_eval_v3"
    blocker_prefix = "external_sc3_study" if sc3_profile else "external_study"
    if study.get("status") != "accepted_frozen_study":
        blockers.append(f"{blocker_prefix}_not_accepted")
    checkpoint_count = study.get("independent_policy_checkpoint_count")
    if (
        isinstance(checkpoint_count, bool)
        or not isinstance(checkpoint_count, int)
        or checkpoint_count < int(profile["minimum_policy_checkpoint_count"])
    ):
        blockers.append(
            f"{blocker_prefix}_policy_checkpoint_count_lt_"
            f"{int(profile['minimum_policy_checkpoint_count'])}"
        )
    accepted_anchor_count = study.get("accepted_anchor_count")
    if (
        isinstance(accepted_anchor_count, bool)
        or not isinstance(accepted_anchor_count, int)
        or accepted_anchor_count <= 0
    ):
        blockers.append(f"{blocker_prefix}_has_no_accepted_anchors")
    independent_policy_ids = [
        _string(item) for item in study.get("independent_policy_checkpoint_ids", []) or []
    ]
    if (
        any(not item for item in independent_policy_ids)
        or len(set(independent_policy_ids)) < 7
        or (
            isinstance(checkpoint_count, int)
            and not isinstance(checkpoint_count, bool)
            and len(set(independent_policy_ids)) != checkpoint_count
        )
    ):
        blockers.append(f"{blocker_prefix}_independent_policy_ids_invalid")
    if (
        profile["requires_three_view_checkpoint"]
        and study.get("three_view_sc3_checkpoint_run_proven") is not True
    ):
        blockers.append("external_sc3_study_three_view_checkpoint_run_missing")
    if (
        profile["requires_calibrated_inverse_threshold"]
        and study.get("calibrated_inverse_threshold_proven") is not True
    ):
        blockers.append("external_sc3_study_inverse_threshold_missing")
    if _string(study.get("benchmark_family")) != _string(profile["benchmark_family"]):
        blockers.append(f"{blocker_prefix}_benchmark_family_mismatch")
    for key in (
        "study_registration_sha256",
        "raw_per_cell_outputs_sha256",
        "code_sha256",
        "model_checkpoint_manifest_sha256",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "environment_manifest_sha256",
    ):
        if not _sha256(study.get(key)):
            blockers.append(f"{blocker_prefix}_{key}_missing_or_invalid")
    for ref_key, digest_key in (
        ("study_registration_artifact", "study_registration_sha256"),
        ("raw_per_cell_outputs_artifact", "raw_per_cell_outputs_sha256"),
        ("code_artifact", "code_sha256"),
        ("model_checkpoint_manifest_artifact", "model_checkpoint_manifest_sha256"),
        ("dataset_manifest_artifact", "dataset_manifest_sha256"),
        ("split_manifest_artifact", "split_manifest_sha256"),
        ("environment_manifest_artifact", "environment_manifest_sha256"),
        ("independent_reproduction_artifact", None),
        ("human_label_protocol_artifact", None),
    ):
        ref = _mapping(study.get(ref_key))
        blockers.extend(_validate_artifact_ref(ref, prefix=f"{blocker_prefix}_{ref_key}"))
        if (
            digest_key
            and ref
            and _string(ref.get("sha256")).lower() != _string(study.get(digest_key)).lower()
        ):
            blockers.append(f"{blocker_prefix}_{ref_key}_digest_mismatch")
    metrics = _mapping(study.get("metrics"))
    pearson = _number(metrics.get("pearson_success_rate_correlation"))
    spearman = _number(metrics.get("spearman_rank_correlation"))
    mmrv = _number(metrics.get("mean_maximum_rank_violation"))
    abstention = _number(metrics.get("abstention_rate"))
    required_metrics = set(profile["required_metrics"])
    if "pearson_success_rate_correlation" in required_metrics and (
        pearson is None or not -1.0 <= pearson <= 1.0
    ):
        blockers.append(f"{blocker_prefix}_pearson_missing_or_invalid")
    if "spearman_rank_correlation" in required_metrics and (
        spearman is None or not -1.0 <= spearman <= 1.0
    ):
        blockers.append(f"{blocker_prefix}_spearman_missing_or_invalid")
    if "mean_maximum_rank_violation" in required_metrics and (mmrv is None or mmrv < 0.0):
        blockers.append(f"{blocker_prefix}_mmrv_missing_or_invalid")
    if abstention is None or not 0.0 <= abstention <= 1.0:
        blockers.append(f"{blocker_prefix}_abstention_missing_or_invalid")
    success_rate_difference = _number(metrics.get(OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC))
    if OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC in required_metrics and success_rate_difference is None:
        blockers.append(f"{blocker_prefix}_success_rate_difference_pp_missing_or_invalid")
    interval_specs = [("abstention_95_ci", abstention, 0.0, 1.0)]
    if "pearson_success_rate_correlation" in required_metrics:
        interval_specs.append(("pearson_95_ci", pearson, -1.0, 1.0))
    if "spearman_rank_correlation" in required_metrics:
        interval_specs.append(("spearman_95_ci", spearman, -1.0, 1.0))
    if "mean_maximum_rank_violation" in required_metrics:
        interval_specs.append(("mmrv_95_ci", mmrv, 0.0, math.inf))
    if OSCAR_SUCCESS_RATE_DIFFERENCE_METRIC in required_metrics:
        interval_specs.append(
            ("success_rate_difference_pp_95_ci", success_rate_difference, -100.0, 100.0)
        )
    for interval_name, estimate, minimum, maximum in interval_specs:
        interval = metrics.get(interval_name)
        bounds = (
            [_number(value) for value in interval]
            if isinstance(interval, Sequence) and not isinstance(interval, (str, bytes, bytearray))
            else []
        )
        if not (
            len(bounds) == 2
            and all(value is not None for value in bounds)
            and minimum <= bounds[0] <= bounds[1] <= maximum
            and estimate is not None
            and bounds[0] <= estimate <= bounds[1]
        ):
            blockers.append(f"{blocker_prefix}_{interval_name}_invalid")
    design = _mapping(study.get("registered_design"))
    for field in (
        "matched_conditions_and_replicates",
        "grouped_train_dev_locked_test",
        "locked_ind_test",
        "locked_ood_test",
        "hierarchical_cluster_uncertainty",
        "abstention_and_coverage_registered",
        "failure_case_reporting_registered",
    ):
        if design.get(field) is not True:
            blockers.append(f"{blocker_prefix}_design_{field}_not_proven")
    rights = _mapping(study.get("rights_and_provenance"))
    for field in (
        "source_provenance_verified",
        "license_inventory_complete",
        "commercial_use_scope_verified",
        "raw_outcomes_imported_not_headline_numbers",
        "final_evaluation_split_locked_before_tuning",
    ):
        if rights.get(field) is not True:
            blockers.append(f"{blocker_prefix}_rights_{field}_not_proven")
    blockers.extend(
        _validate_artifact_ref(
            _mapping(rights.get("license_inventory_artifact")),
            prefix=f"{blocker_prefix}_rights_license_inventory_artifact",
        )
    )
    raw_cell_count = study.get("raw_per_cell_count")
    failure_case_count = study.get("failure_case_count")
    coverage_rate = _number(study.get("coverage_rate"))
    if (
        isinstance(raw_cell_count, bool)
        or not isinstance(raw_cell_count, int)
        or raw_cell_count <= 0
    ):
        blockers.append(f"{blocker_prefix}_raw_per_cell_count_invalid")
    if (
        isinstance(failure_case_count, bool)
        or not isinstance(failure_case_count, int)
        or failure_case_count < 0
    ):
        blockers.append(f"{blocker_prefix}_failure_case_count_invalid")
    if coverage_rate is None or not 0.0 <= coverage_rate <= 1.0:
        blockers.append(f"{blocker_prefix}_coverage_rate_invalid")
    if _mapping(study.get("independent_reproduction")).get("status") != "passed":
        blockers.append(f"{blocker_prefix}_independent_reproduction_not_passed")
    human_protocol = _mapping(study.get("human_label_protocol"))
    agreement = _number(human_protocol.get("inter_rater_agreement"))
    if not (
        human_protocol.get("status") == "accepted"
        and agreement is not None
        and 0.0 <= agreement <= 1.0
        and human_protocol.get("adjudication_completed") is True
    ):
        blockers.append(f"{blocker_prefix}_human_label_protocol_incomplete")
    study_signature = _mapping(study.get("study_signature"))
    if not (
        study_signature.get("signature_verified") is True
        and _string(study_signature.get("signer_key_id"))
        and _string(study_signature.get("verifier_id"))
    ):
        blockers.append(f"{blocker_prefix}_signature_not_verified")
    blockers.extend(
        _validate_artifact_ref(
            _mapping(study_signature.get("verification_report_artifact")),
            prefix=f"{blocker_prefix}_signature_verification_report",
        )
    )
    blockers.append(f"{blocker_prefix}_requires_independent_manual_acceptance")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "external_study_profile_validation.v2",
        "protocol_profile": profile_id,
        "benchmark_family": profile["benchmark_family"],
        "profile_requirements": profile,
        "status": "external_proof_required",
        "external_manual_proof": True,
        "blockers": blockers,
    }
