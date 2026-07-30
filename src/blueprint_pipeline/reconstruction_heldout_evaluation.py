"""Independent real-view evaluation for reconstruction appearance candidates."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest


HELDOUT_APPEARANCE_REQUEST_SCHEMA_VERSION = "heldout_appearance_evaluation_request.v1"
HELDOUT_APPEARANCE_REPORT_SCHEMA_VERSION = "visual_heldout_evaluation_report.v1"


class HeldoutAppearanceEvaluationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_relative_path(root: Path, value: Any, *, label: str) -> Path:
    raw = str(value or "").strip().replace("\\", "/")
    relative = PurePosixPath(raw)
    if (
        not raw
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise HeldoutAppearanceEvaluationError([f"{label}_path_invalid"])
    lexical = root / Path(*relative.parts)
    if lexical.is_symlink():
        raise HeldoutAppearanceEvaluationError([f"{label}_symlink_forbidden"])
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HeldoutAppearanceEvaluationError([f"{label}_path_escape"]) from exc
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise HeldoutAppearanceEvaluationError([f"{label}_missing"])
    return resolved


def build_heldout_appearance_evaluation_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    request = dict(value)
    errors: list[str] = []
    if request.get("schema_version") != HELDOUT_APPEARANCE_REQUEST_SCHEMA_VERSION:
        errors.append("heldout_request_schema_invalid")
    for key in (
        "source_capture_digest",
        "reconstruction_dataset_digest",
        "frozen_split_digest",
        "candidate_reconstruction_result_digest",
        "evaluator_implementation_digest",
        "source_commit_sha",
    ):
        if key == "source_commit_sha":
            commit = str(request.get(key) or "")
            if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
                errors.append(f"heldout_request_{key}_invalid")
        elif not _is_digest(request.get(key)):
            errors.append(f"heldout_request_{key}_invalid")
    for key in (
        "stable_run_identity",
        "source_capture_identity",
        "candidate_method_id",
        "candidate_provider_identity",
        "evaluator_identity",
        "evaluator_provider_identity",
        "candidate_root",
        "evaluator_root",
        "timestamp",
    ):
        if not str(request.get(key) or "").strip():
            errors.append(f"heldout_request_{key}_missing")
    if not isinstance(request.get("coordinate_frame_declaration"), Mapping):
        errors.append("heldout_request_coordinate_frame_declaration_invalid")
    if not isinstance(request.get("authority_used"), Mapping):
        errors.append("heldout_request_authority_used_invalid")
    if request.get("split_frozen_before_training") is not True:
        errors.append("heldout_request_split_not_precommitted")
    if request.get("candidate_had_hidden_access") is not False:
        errors.append("heldout_request_candidate_hidden_access_not_false")
    if request.get("candidate_selected_heldout") is not False:
        errors.append("heldout_request_candidate_selected_heldout_not_false")
    if request.get("candidate_self_grading") is not False:
        errors.append("heldout_request_candidate_self_grading_not_false")
    if request.get("thresholds_frozen_before_evaluation") is not True:
        errors.append("heldout_request_thresholds_not_frozen")
    if request.get("candidate_provider_identity") == request.get(
        "evaluator_provider_identity"
    ):
        errors.append("heldout_request_evaluator_not_independent")
    thresholds = request.get("thresholds")
    if not isinstance(thresholds, Mapping):
        errors.append("heldout_request_thresholds_invalid")
    else:
        expected_thresholds = {
            "minimum_mean_psnr_db",
            "minimum_mean_global_ssim",
            "maximum_mean_absolute_error",
        }
        if set(thresholds) != expected_thresholds:
            errors.append("heldout_request_threshold_fields_invalid")
        for key in expected_thresholds:
            number = thresholds.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(
                float(number)
            ):
                errors.append(f"heldout_request_threshold_invalid:{key}")
    pairs = request.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        errors.append("heldout_request_pairs_missing")
    else:
        seen: set[str] = set()
        for index, raw_pair in enumerate(pairs):
            if not isinstance(raw_pair, Mapping):
                errors.append(f"heldout_request_pair_invalid:{index}")
                continue
            pair = dict(raw_pair)
            view_id = str(pair.get("view_id") or "").strip()
            if not view_id or view_id in seen:
                errors.append(f"heldout_request_view_id_invalid:{index}")
            seen.add(view_id)
            if pair.get("split") != "held_out" or pair.get("excluded_from_training") is not True:
                errors.append(f"heldout_request_pair_not_heldout:{view_id or index}")
            for key in ("real_view_digest", "candidate_render_digest"):
                if not _is_digest(pair.get(key)):
                    errors.append(f"heldout_request_pair_digest_invalid:{view_id or index}:{key}")
            for key in ("real_view_relative_path", "candidate_render_relative_path"):
                path = PurePosixPath(str(pair.get(key) or "").replace("\\", "/"))
                if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
                    errors.append(f"heldout_request_pair_path_invalid:{view_id or index}:{key}")
    supplied_digest = request.pop("heldout_appearance_evaluation_request_digest", None)
    request["heldout_appearance_evaluation_request_digest"] = canonical_digest(
        request, digest_field="heldout_appearance_evaluation_request_digest"
    )
    if supplied_digest is not None and supplied_digest != request[
        "heldout_appearance_evaluation_request_digest"
    ]:
        errors.append("heldout_request_digest_mismatch")
    if errors:
        raise HeldoutAppearanceEvaluationError(errors)
    return request


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def _global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))
    left_variance = float(np.mean(np.square(left - left_mean)))
    right_variance = float(np.mean(np.square(right - right_mean)))
    covariance = float(np.mean((left - left_mean) * (right - right_mean)))
    c1 = 0.01**2
    c2 = 0.03**2
    denominator = (left_mean**2 + right_mean**2 + c1) * (
        left_variance + right_variance + c2
    )
    if denominator == 0.0:
        return 1.0
    return float(
        ((2.0 * left_mean * right_mean + c1) * (2.0 * covariance + c2))
        / denominator
    )


def evaluate_heldout_appearance(
    *, source_artifact: Mapping[str, Any], output_root: str | Path
) -> dict[str, Any]:
    """Evaluate candidate renders without exposing held-out paths to the candidate."""

    del output_root  # The supervisor owns artifact persistence.
    request = build_heldout_appearance_evaluation_request(source_artifact)
    candidate_root = Path(request["candidate_root"]).expanduser().resolve()
    evaluator_root = Path(request["evaluator_root"]).expanduser().resolve()
    if candidate_root == evaluator_root:
        raise HeldoutAppearanceEvaluationError(["heldout_roots_not_isolated"])
    rows: list[dict[str, Any]] = []
    for pair in request["pairs"]:
        view_id = str(pair["view_id"])
        real_path = _safe_relative_path(
            evaluator_root, pair["real_view_relative_path"], label="heldout_real_view"
        )
        candidate_path = _safe_relative_path(
            candidate_root,
            pair["candidate_render_relative_path"],
            label="candidate_render",
        )
        if _sha256_file(real_path) != pair["real_view_digest"]:
            raise HeldoutAppearanceEvaluationError([f"heldout_real_digest_mismatch:{view_id}"])
        if _sha256_file(candidate_path) != pair["candidate_render_digest"]:
            raise HeldoutAppearanceEvaluationError(
                [f"candidate_render_digest_mismatch:{view_id}"]
            )
        real = _rgb(real_path)
        candidate = _rgb(candidate_path)
        if real.shape != candidate.shape:
            raise HeldoutAppearanceEvaluationError([f"heldout_view_shape_mismatch:{view_id}"])
        difference = real - candidate
        mse = float(np.mean(np.square(difference)))
        mae = float(np.mean(np.abs(difference)))
        psnr = float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse))
        rows.append(
            {
                "view_id": view_id,
                "projection_form": str(pair.get("projection_form") or "perspective_rgb"),
                "real_view_digest": pair["real_view_digest"],
                "candidate_render_digest": pair["candidate_render_digest"],
                "psnr_db": "infinity" if math.isinf(psnr) else round(psnr, 6),
                "global_ssim": round(_global_ssim(real, candidate), 8),
                "mean_absolute_error": round(mae, 8),
            }
        )
    finite_psnr = [float(row["psnr_db"]) for row in rows if row["psnr_db"] != "infinity"]
    mean_psnr = (
        float("inf")
        if len(finite_psnr) < len(rows)
        else float(np.mean(finite_psnr))
    )
    mean_ssim = float(np.mean([row["global_ssim"] for row in rows]))
    mean_mae = float(np.mean([row["mean_absolute_error"] for row in rows]))
    thresholds = request["thresholds"]
    passed = bool(
        mean_psnr >= float(thresholds["minimum_mean_psnr_db"])
        and mean_ssim >= float(thresholds["minimum_mean_global_ssim"])
        and mean_mae <= float(thresholds["maximum_mean_absolute_error"])
    )
    report = {
        "schema_version": HELDOUT_APPEARANCE_REPORT_SCHEMA_VERSION,
        "stable_run_identity": request["stable_run_identity"],
        "source_capture_identity": request["source_capture_identity"],
        "source_capture_digest": request["source_capture_digest"],
        "reconstruction_dataset_digest": request["reconstruction_dataset_digest"],
        "frozen_split_digest": request["frozen_split_digest"],
        "candidate_reconstruction_result_digest": request[
            "candidate_reconstruction_result_digest"
        ],
        "evaluation_request_digest": request[
            "heldout_appearance_evaluation_request_digest"
        ],
        "candidate_method_id": request["candidate_method_id"],
        "candidate_provider_identity": request["candidate_provider_identity"],
        "evaluator_identity": request["evaluator_identity"],
        "evaluator_provider_identity": request["evaluator_provider_identity"],
        "evaluator_implementation_digest": request["evaluator_implementation_digest"],
        "source_commit_sha": request["source_commit_sha"],
        "coordinate_frame_declaration": request["coordinate_frame_declaration"],
        "rows": rows,
        "aggregate": {
            "mean_psnr_db": "infinity" if math.isinf(mean_psnr) else round(mean_psnr, 6),
            "mean_global_ssim": round(mean_ssim, 8),
            "mean_absolute_error": round(mean_mae, 8),
            "thresholds": dict(thresholds),
            "thresholds_passed": passed,
        },
        "status": "passed_appearance_only" if passed else "rejected_appearance_quality",
        "heldout_observation_count": len(rows),
        "candidate_had_hidden_access": False,
        "candidate_selected_heldout": False,
        "candidate_self_graded": False,
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": dict(request.get("authority_used") or {}),
        "warnings": ["global_ssim_is_repository_deterministic_equivalent_not_windowed_ssim"],
        "blockers": [] if passed else ["heldout_appearance_thresholds_not_met"],
        "proof_effect": "independent_heldout_appearance_evaluation_only",
        "claim_ceiling": "appearance_reconstruction",
        "metric_scale_proven": False,
        "metric_geometry_proven": False,
        "collision_geometry_proven": False,
        "physics_readiness_proven": False,
        "physical_task_success_proven": False,
        "deployment_readiness_proven": False,
        "parent_artifact_or_event": {
            "candidate_reconstruction_result_digest": request[
                "candidate_reconstruction_result_digest"
            ],
            "frozen_split_digest": request["frozen_split_digest"],
        },
        "timestamp": request["timestamp"],
    }
    report["visual_heldout_evaluation_report_digest"] = canonical_digest(
        report, digest_field="visual_heldout_evaluation_report_digest"
    )
    return report


def build_visual_heldout_evaluation_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a recorded independent evaluator result for replay."""

    report = dict(value)
    errors: list[str] = []
    if report.get("schema_version") != HELDOUT_APPEARANCE_REPORT_SCHEMA_VERSION:
        errors.append("heldout_report_schema_invalid")
    for key in (
        "source_capture_digest",
        "reconstruction_dataset_digest",
        "frozen_split_digest",
        "candidate_reconstruction_result_digest",
        "evaluation_request_digest",
        "evaluator_implementation_digest",
    ):
        if not _is_digest(report.get(key)):
            errors.append(f"heldout_report_{key}_invalid")
    if report.get("status") not in {
        "passed_appearance_only",
        "rejected_appearance_quality",
    }:
        errors.append("heldout_report_status_invalid")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("heldout_report_rows_missing")
    if report.get("heldout_observation_count") != len(rows or []):
        errors.append("heldout_report_count_mismatch")
    seen: set[str] = set()
    for index, raw_row in enumerate(rows or []):
        if not isinstance(raw_row, Mapping):
            errors.append(f"heldout_report_row_invalid:{index}")
            continue
        view_id = str(raw_row.get("view_id") or "").strip()
        if not view_id or view_id in seen:
            errors.append(f"heldout_report_view_id_invalid:{index}")
        seen.add(view_id)
        if not _is_digest(raw_row.get("real_view_digest")) or not _is_digest(
            raw_row.get("candidate_render_digest")
        ):
            errors.append(f"heldout_report_row_digest_invalid:{view_id or index}")
        for key in ("global_ssim", "mean_absolute_error"):
            number = raw_row.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(
                float(number)
            ):
                errors.append(f"heldout_report_row_metric_invalid:{view_id or index}:{key}")
        psnr = raw_row.get("psnr_db")
        if psnr != "infinity" and (
            isinstance(psnr, bool)
            or not isinstance(psnr, (int, float))
            or not math.isfinite(float(psnr))
        ):
            errors.append(f"heldout_report_row_metric_invalid:{view_id or index}:psnr_db")
    aggregate = report.get("aggregate")
    thresholds_passed = (
        aggregate.get("thresholds_passed") if isinstance(aggregate, Mapping) else None
    )
    if not isinstance(aggregate, Mapping) or not isinstance(thresholds_passed, bool):
        errors.append("heldout_report_aggregate_invalid")
    expected_status = (
        "passed_appearance_only" if thresholds_passed is True else "rejected_appearance_quality"
    )
    expected_blockers = [] if thresholds_passed is True else [
        "heldout_appearance_thresholds_not_met"
    ]
    if report.get("status") != expected_status or report.get("blockers") != expected_blockers:
        errors.append("heldout_report_status_threshold_mismatch")
    if (
        not str(report.get("candidate_provider_identity") or "").strip()
        or not str(report.get("evaluator_provider_identity") or "").strip()
        or report.get("candidate_provider_identity")
        == report.get("evaluator_provider_identity")
    ):
        errors.append("heldout_report_evaluator_not_independent")
    if report.get("cost_usd") != 0.0:
        errors.append("heldout_report_cost_invalid")
    for key in (
        "candidate_had_hidden_access",
        "candidate_selected_heldout",
        "candidate_self_graded",
        "metric_scale_proven",
        "metric_geometry_proven",
        "collision_geometry_proven",
        "physics_readiness_proven",
        "physical_task_success_proven",
        "deployment_readiness_proven",
    ):
        if report.get(key) is not False:
            errors.append(f"heldout_report_forbidden_claim:{key}")
    if (
        report.get("proof_effect")
        != "independent_heldout_appearance_evaluation_only"
        or report.get("claim_ceiling") != "appearance_reconstruction"
    ):
        errors.append("heldout_report_claim_boundary_invalid")
    expected_digest = canonical_digest(
        report, digest_field="visual_heldout_evaluation_report_digest"
    )
    if report.get("visual_heldout_evaluation_report_digest") != expected_digest:
        errors.append("heldout_report_digest_mismatch")
    if errors:
        raise HeldoutAppearanceEvaluationError(errors)
    return report


__all__ = [
    "HELDOUT_APPEARANCE_REPORT_SCHEMA_VERSION",
    "HELDOUT_APPEARANCE_REQUEST_SCHEMA_VERSION",
    "HeldoutAppearanceEvaluationError",
    "build_heldout_appearance_evaluation_request",
    "build_visual_heldout_evaluation_report",
    "evaluate_heldout_appearance",
]
