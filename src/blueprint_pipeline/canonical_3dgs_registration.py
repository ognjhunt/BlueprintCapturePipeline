"""Bind one qualified native 3DGS candidate to its captured ARKit site frame.

Registration is an appearance/frame claim only.  This module independently
decodes the standard 3DGS PLY and computes registration residuals, but it never
promotes the observed depth surface into collision, physics, Isaac, or physical
task truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import read_standard_3dgs_ply


REGISTRATION_MEASUREMENT_SCHEMA = "canonical_3dgs_registration_measurement.v1"
CANONICAL_REGISTERED_APPEARANCE_SCHEMA = "canonical_registered_appearance.v1"


class Canonical3DGSRegistrationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finite_nonnegative(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Canonical3DGSRegistrationError(["registration_threshold_invalid"])
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise Canonical3DGSRegistrationError(["registration_threshold_invalid"])
    return result


def _matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise Canonical3DGSRegistrationError(["registration_transform_invalid"])
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise Canonical3DGSRegistrationError(["registration_transform_invalid"])
    return matrix


def build_canonical_3dgs_registration_measurement(
    *,
    source_capture_digest: str,
    appearance_asset_digest: str,
    world_frame: str,
    metric_scale_status: str,
    transform_appearance_to_site: Sequence[Sequence[float]],
    correspondences: Sequence[Mapping[str, Any]],
    thresholds_m: Mapping[str, Any],
    method_id: str,
    threshold_frozen_before_measurement: bool,
    timestamp: str,
) -> dict[str, Any]:
    """Compute, rather than accept, the residual summary for one transform."""

    if not _digest(source_capture_digest) or not _digest(appearance_asset_digest):
        raise Canonical3DGSRegistrationError(["registration_source_binding_invalid"])
    if not str(world_frame or "").strip() or not str(method_id or "").strip():
        raise Canonical3DGSRegistrationError(["registration_frame_or_method_missing"])
    if metric_scale_status not in {
        "sensor_metric_unvalidated",
        "independently_validated_metric",
        "not_established",
        "unknown",
    }:
        raise Canonical3DGSRegistrationError(["registration_metric_scale_status_invalid"])
    if threshold_frozen_before_measurement is not True:
        raise Canonical3DGSRegistrationError(["registration_threshold_not_frozen"])
    threshold_values = {
        key: _finite_nonnegative(thresholds_m.get(key))
        for key in ("maximum_rmse_m", "maximum_p95_m", "maximum_residual_m")
    }
    matrix = _matrix(transform_appearance_to_site)
    linear = matrix[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    if np.any(scales <= 0) or float(np.max(scales) - np.min(scales)) > 1e-6:
        raise Canonical3DGSRegistrationError(["registration_transform_non_similarity"])
    rotation = linear / float(np.mean(scales))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(
        rotation
    ) <= 0:
        raise Canonical3DGSRegistrationError(["registration_transform_non_similarity"])
    if len(correspondences) < 3:
        raise Canonical3DGSRegistrationError(["registration_correspondences_insufficient"])
    residuals: list[float] = []
    ids: set[str] = set()
    normalized_correspondences: list[dict[str, Any]] = []
    for raw in correspondences:
        identifier = str(raw.get("correspondence_id") or "")
        source = np.asarray(raw.get("appearance_point"), dtype=np.float64)
        target = np.asarray(raw.get("site_point"), dtype=np.float64)
        if (
            not identifier
            or identifier in ids
            or source.shape != (3,)
            or target.shape != (3,)
            or not np.isfinite(source).all()
            or not np.isfinite(target).all()
        ):
            raise Canonical3DGSRegistrationError(["registration_correspondence_invalid"])
        transformed = (matrix @ np.append(source, 1.0))[:3]
        residual = float(np.linalg.norm(transformed - target))
        residuals.append(residual)
        ids.add(identifier)
        normalized_correspondences.append(
            {
                "correspondence_id": identifier,
                "appearance_point": [float(value) for value in source],
                "site_point": [float(value) for value in target],
                "residual_m": residual,
            }
        )
    values = np.asarray(residuals, dtype=np.float64)
    summary = {
        "rmse_m": float(np.sqrt(np.mean(values**2))),
        "p95_m": float(np.quantile(values, 0.95)),
        "maximum_residual_m": float(np.max(values)),
        "minimum_residual_m": float(np.min(values)),
    }
    passed = (
        summary["rmse_m"] <= threshold_values["maximum_rmse_m"]
        and summary["p95_m"] <= threshold_values["maximum_p95_m"]
        and summary["maximum_residual_m"] <= threshold_values["maximum_residual_m"]
    )
    result = {
        "schema_version": REGISTRATION_MEASUREMENT_SCHEMA,
        "status": "qualified" if passed else "failed_residual_gate",
        "source_capture_digest": source_capture_digest,
        "appearance_asset_digest": appearance_asset_digest,
        "world_frame": world_frame,
        "metric_scale_status": metric_scale_status,
        "metric_scale_independently_validated": (
            metric_scale_status == "independently_validated_metric"
        ),
        "method_id": method_id,
        "transform_appearance_to_site": [
            [float(value) for value in row] for row in matrix
        ],
        "scale_factor": float(np.mean(scales)),
        "correspondence_count": len(normalized_correspondences),
        "correspondences": normalized_correspondences,
        "residual_summary": summary,
        "thresholds_m": threshold_values,
        "threshold_frozen_before_measurement": True,
        "registration_gate_passed": passed,
        "collision_geometry_evaluated": False,
        "proof_effect": "appearance_to_site_frame_registration_measurement_only",
        "timestamp": timestamp,
    }
    result["canonical_3dgs_registration_measurement_digest"] = canonical_digest(
        result, digest_field="canonical_3dgs_registration_measurement_digest"
    )
    return result


def build_canonical_registered_appearance(
    *,
    source_admission: Mapping[str, Any],
    campaign_result: Mapping[str, Any],
    appearance_asset_path: str | Path,
    appearance_asset_reference: str,
    geometry_asset_digest: str,
    quality_comparison: Mapping[str, Any] | None = None,
    registration_measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a candidate or qualified registered reconstruction without collision claims."""

    source = json.loads(canonical_json(dict(source_admission)))
    campaign = json.loads(canonical_json(dict(campaign_result)))
    errors: list[str] = []
    source_digest = source.get("canonical_3dgs_source_admission_digest")
    if (
        source.get("schema_version") != "canonical_3dgs_source_admission.v1"
        or source.get("status") != "admitted_candidate_training_source"
        or source_digest
        != canonical_digest(source, digest_field="canonical_3dgs_source_admission_digest")
    ):
        errors.append("registered_reconstruction_source_admission_invalid")
    if (
        campaign.get("schema_version") != "canonical_3dgs_campaign_result.v1"
        or campaign.get("canonical_3dgs_campaign_result_digest")
        != canonical_digest(campaign, digest_field="canonical_3dgs_campaign_result_digest")
        or campaign.get("source_capture_digest") != source.get("source_capture_digest")
        or campaign.get("canonical_3dgs_source_admission_digest") != source_digest
        or campaign.get("world_frame") != source.get("world_frame")
        or campaign.get("metric_scale_status") != source.get("metric_scale_status")
    ):
        errors.append("registered_reconstruction_campaign_binding_invalid")
    path = Path(appearance_asset_path).expanduser().resolve()
    if not path.is_file() or path.is_symlink() or not appearance_asset_reference.strip():
        errors.append("registered_reconstruction_appearance_asset_invalid")
        asset_digest = ""
        splat = None
    else:
        asset_digest = _sha256(path)
        try:
            splat = read_standard_3dgs_ply(path)
        except (OSError, TypeError, ValueError):
            splat = None
            errors.append("registered_reconstruction_standard_3dgs_decode_failed")
    bindings = campaign.get("appearance_fidelity_candidate_bindings")
    matching = [
        row
        for row in bindings or []
        if isinstance(row, Mapping) and row.get("asset_digest") == asset_digest
    ]
    if len(matching) != 1 or splat is None:
        errors.append("registered_reconstruction_appearance_campaign_binding_invalid")
        binding: Mapping[str, Any] = {}
    else:
        binding = matching[0]
        if (
            int(binding.get("splat_count") or 0) != splat.count
            or binding.get("representation") != "standard_3dgs_ply"
            or binding.get("global_decimation_applied") is not False
        ):
            errors.append("registered_reconstruction_appearance_profile_mismatch")
    source_artifact_digests = {
        str(row.get("digest"))
        for row in source.get("input_artifacts") or []
        if isinstance(row, Mapping)
    }
    if not _digest(geometry_asset_digest) or geometry_asset_digest not in source_artifact_digests:
        errors.append("registered_reconstruction_geometry_source_binding_invalid")
    if errors:
        raise Canonical3DGSRegistrationError(errors)

    heldout_status = "required_not_executed"
    quality_digest = None
    selected_arm = str(binding.get("candidate_arm_id") or "")
    if quality_comparison is not None:
        quality = json.loads(canonical_json(dict(quality_comparison)))
        quality_digest = quality.get("canonical_3dgs_quality_comparison_digest")
        reports = quality.get("candidate_reports")
        selected_reports = [
            row
            for row in reports or []
            if isinstance(row, Mapping) and row.get("arm_id") == selected_arm
        ]
        if (
            quality.get("schema_version") != "canonical_3dgs_quality_comparison.v1"
            or quality_digest
            != canonical_digest(
                quality, digest_field="canonical_3dgs_quality_comparison_digest"
            )
            or quality.get("canonical_3dgs_campaign_result_digest")
            != campaign.get("canonical_3dgs_campaign_result_digest")
            or quality.get("candidate_hidden_pixel_access") is not False
            or quality.get("quality_winner") != selected_arm
            or len(selected_reports) != 1
            or selected_reports[0].get("appearance_fidelity_status") != "qualified"
        ):
            raise Canonical3DGSRegistrationError(
                ["registered_reconstruction_quality_binding_invalid"]
            )
        heldout_status = "qualified"

    registration_status = "required_not_executed"
    registration_digest = None
    residual_summary = None
    transform = None
    if registration_measurement is not None:
        measurement = json.loads(canonical_json(dict(registration_measurement)))
        registration_digest = measurement.get(
            "canonical_3dgs_registration_measurement_digest"
        )
        if (
            measurement.get("schema_version") != REGISTRATION_MEASUREMENT_SCHEMA
            or registration_digest
            != canonical_digest(
                measurement,
                digest_field="canonical_3dgs_registration_measurement_digest",
            )
            or measurement.get("source_capture_digest") != source.get("source_capture_digest")
            or measurement.get("appearance_asset_digest") != asset_digest
            or measurement.get("world_frame") != source.get("world_frame")
            or measurement.get("metric_scale_status") != source.get("metric_scale_status")
        ):
            raise Canonical3DGSRegistrationError(
                ["registered_reconstruction_registration_binding_invalid"]
            )
        registration_status = (
            "qualified"
            if measurement.get("registration_gate_passed") is True
            and measurement.get("status") == "qualified"
            else "failed_residual_gate"
        )
        residual_summary = dict(measurement.get("residual_summary") or {})
        transform = measurement.get("transform_appearance_to_site")

    qualified = heldout_status == "qualified" and registration_status == "qualified"
    blockers = []
    if heldout_status != "qualified":
        blockers.append("heldout_appearance_qualification_required")
    if registration_status != "qualified":
        blockers.append("appearance_to_site_registration_qualification_required")
    result = {
        "schema_version": CANONICAL_REGISTERED_APPEARANCE_SCHEMA,
        "status": "qualified" if qualified else "candidate_only",
        "source_profile_digest": source_digest,
        "source_capture_digest": source["source_capture_digest"],
        "canonical_3dgs_campaign_result_digest": campaign[
            "canonical_3dgs_campaign_result_digest"
        ],
        "appearance_format": "native_3dgs",
        "appearance_asset_reference": appearance_asset_reference,
        "appearance_asset_digest": asset_digest,
        "appearance_splat_count": int(binding["splat_count"]),
        "appearance_sh_degree": int(binding["sh_degree"]),
        "appearance_coordinate_basis_digest": binding["coordinate_basis_digest"],
        "geometry_asset_digest": geometry_asset_digest,
        "source_scene_digest": geometry_asset_digest,
        "scene_registration_digest": registration_digest,
        "registration_status": registration_status,
        "registration_transform_appearance_to_site": transform,
        "registration_residual_summary": residual_summary,
        "heldout_appearance_status": heldout_status,
        "heldout_quality_comparison_digest": quality_digest,
        "world_frame": source["world_frame"],
        "coordinate_frame_declaration": source["coordinate_frame_declaration"],
        "metric_scale_status": source["metric_scale_status"],
        "metric_scale_proven": source["metric_scale_independently_validated"],
        "full_resolution_appearance_preserved": True,
        "presentation_output_used_as_evaluation_evidence": False,
        "metric_geometry_proven": False,
        "collision_geometry_validated": False,
        "isaac_compatibility_proven": False,
        "physical_success_proven": False,
        "candidate_may_self_authorize": False,
        "blockers": blockers,
        "proof_effect": (
            "registered_appearance_reconstruction"
            if qualified
            else "registered_appearance_candidate_only"
        ),
        "claim_ceiling": "registered_appearance_only",
        "timestamp": campaign["timestamp"],
    }
    result["canonical_registered_appearance_digest"] = canonical_digest(
        result, digest_field="canonical_registered_appearance_digest"
    )
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise Canonical3DGSRegistrationError(["registered_reconstruction_immutable_conflict"])
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-admission", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--appearance-ply", required=True)
    parser.add_argument("--appearance-reference", required=True)
    parser.add_argument("--geometry-asset-digest", required=True)
    parser.add_argument("--quality-comparison")
    parser.add_argument("--registration-measurement")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)

    def load(path: str | None) -> dict[str, Any] | None:
        if path is None:
            return None
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Canonical3DGSRegistrationError(
                ["registered_reconstruction_input_invalid"]
            ) from exc
        if not isinstance(value, dict):
            raise Canonical3DGSRegistrationError(["registered_reconstruction_input_invalid"])
        return value

    result = build_canonical_registered_appearance(
        source_admission=load(arguments.source_admission) or {},
        campaign_result=load(arguments.campaign) or {},
        appearance_asset_path=arguments.appearance_ply,
        appearance_asset_reference=arguments.appearance_reference,
        geometry_asset_digest=arguments.geometry_asset_digest,
        quality_comparison=load(arguments.quality_comparison),
        registration_measurement=load(arguments.registration_measurement),
    )
    _write_immutable(Path(arguments.output).expanduser().resolve(), result)
    print(canonical_json(result))
    return 0 if result["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Canonical3DGSRegistrationError",
    "CANONICAL_REGISTERED_APPEARANCE_SCHEMA",
    "REGISTRATION_MEASUREMENT_SCHEMA",
    "build_canonical_3dgs_registration_measurement",
    "build_canonical_registered_appearance",
]
