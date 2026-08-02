"""Provider-neutral appearance-fidelity qualification and render routing.

The full-resolution splat remains appearance truth.  A render input may remove
only digest-bound, independently qualified nonfinite, low-opacity, or robust
spatial-outlier Gaussians; global decimation can never qualify.  Appearance
rendering, robot dynamics, and compositing are selected independently so a
simulator is not made the visual authority merely because it runs physics.

DiFix, Artifixer, and similar enhancement outputs are presentation derivatives.
They are explicitly forbidden from policy observations, evaluation evidence,
target binding, geometry, collision, or routing qualification.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


FIDELITY_SCHEMA = "appearance_fidelity_qualification.v1"
ROUTE_SCHEMA = "appearance_render_route.v1"
COMPOSITE_SCHEMA = "robot_appearance_composite_contract.v1"
PRESENTATION_SCHEMA = "appearance_presentation_derivative.v1"

_REPRESENTATIONS = {"standard_3dgs_ply", "spz", "ksplat", "splat"}
_REMOVAL_REASONS = {"nonfinite", "low_opacity", "robust_spatial_outlier"}
_PRESENTATION_METHODS = {
    "artifixer",
    "artifixer3d",
    "artifixer3d_plus",
    "difix",
    "difix3d",
    "difix3d_plus",
    "fixer",
    "harmonizer",
    "other_declared_enhancer",
}


class AppearanceFidelityContractError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise AppearanceFidelityContractError(["appearance_fidelity_not_json"]) from exc


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite_number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return (
        math.isfinite(number)
        and (minimum is None or number >= minimum)
        and (maximum is None or number <= maximum)
    )


def _valid_bounds(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    minimum = value.get("aabb_min")
    maximum = value.get("aabb_max")
    robust_minimum = value.get("robust_min")
    robust_maximum = value.get("robust_max")
    rows = (minimum, maximum, robust_minimum, robust_maximum)
    if any(
        not isinstance(row, list)
        or len(row) != 3
        or any(not _finite_number(item) for item in row)
        for row in rows
    ):
        return False
    if any(float(minimum[index]) > float(maximum[index]) for index in range(3)):
        return False
    if any(float(robust_minimum[index]) > float(robust_maximum[index]) for index in range(3)):
        return False
    return _finite_number(value.get("robust_percentile"), minimum=0.9, maximum=1.0)


def build_appearance_fidelity_qualification(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate measurements and deterministically qualify or reject one render input."""

    artifact = _clone(dict(value))
    supplied = artifact.pop("appearance_fidelity_qualification_digest", None)
    blockers: list[str] = []
    if artifact.get("schema_version") not in {None, FIDELITY_SCHEMA}:
        raise AppearanceFidelityContractError(["appearance_fidelity_schema_invalid"])
    artifact["schema_version"] = FIDELITY_SCHEMA
    source = artifact.get("source_appearance")
    render_input = artifact.get("render_input")
    renderer = artifact.get("renderer")
    comparison = artifact.get("reference_frame_comparison")
    policy = artifact.get("qualification_policy")
    if not all(isinstance(item, Mapping) for item in (source, render_input, renderer, comparison, policy)):
        raise AppearanceFidelityContractError(["appearance_fidelity_sections_missing"])

    for label, row in (("source", source), ("render_input", render_input)):
        if not _digest(row.get("asset_digest")):
            raise AppearanceFidelityContractError([f"appearance_fidelity_{label}_digest_invalid"])
        if not _digest(row.get("coordinate_basis_digest")):
            raise AppearanceFidelityContractError(
                [f"appearance_fidelity_{label}_coordinate_basis_digest_invalid"]
            )
        if row.get("representation") not in _REPRESENTATIONS:
            raise AppearanceFidelityContractError(
                [f"appearance_fidelity_{label}_representation_invalid"]
            )
        if (
            isinstance(row.get("splat_count"), bool)
            or not isinstance(row.get("splat_count"), int)
            or int(row["splat_count"]) < 1
        ):
            raise AppearanceFidelityContractError([f"appearance_fidelity_{label}_count_invalid"])
        if (
            isinstance(row.get("sh_degree"), bool)
            or not isinstance(row.get("sh_degree"), int)
            or not 0 <= int(row["sh_degree"]) <= 3
        ):
            raise AppearanceFidelityContractError([f"appearance_fidelity_{label}_sh_invalid"])
        if not _valid_bounds(row.get("bounds")):
            raise AppearanceFidelityContractError([f"appearance_fidelity_{label}_bounds_invalid"])

    source_count = int(source["splat_count"])
    retained_count = int(render_input["splat_count"])
    if retained_count > source_count:
        raise AppearanceFidelityContractError(["appearance_fidelity_retained_count_exceeds_source"])
    retained_fraction = retained_count / source_count
    removed_count = source_count - retained_count
    removal_reasons = render_input.get("removal_reasons")
    if not isinstance(removal_reasons, list):
        raise AppearanceFidelityContractError(["appearance_fidelity_removal_reasons_invalid"])
    seen_reasons: set[str] = set()
    qualified_removed_count = 0
    for row in removal_reasons:
        if not isinstance(row, Mapping) or row.get("reason") not in _REMOVAL_REASONS:
            raise AppearanceFidelityContractError(["appearance_fidelity_removal_reason_invalid"])
        reason = str(row["reason"])
        if reason in seen_reasons:
            raise AppearanceFidelityContractError(["appearance_fidelity_removal_reason_duplicate"])
        seen_reasons.add(reason)
        count = row.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise AppearanceFidelityContractError(["appearance_fidelity_removal_count_invalid"])
        qualified_removed_count += count
        if count and (
            row.get("independently_qualified") is not True
            or not _digest(row.get("qualification_digest"))
            or not _digest(row.get("policy_digest"))
        ):
            blockers.append(f"unqualified_splat_removal:{reason}")
    if qualified_removed_count != removed_count:
        blockers.append("splat_removal_accounting_mismatch")
    if render_input.get("global_decimation_applied") is not False:
        blockers.append("global_splat_decimation_forbidden")
    minimum_retained = policy.get("minimum_retained_fraction")
    if not _finite_number(minimum_retained, minimum=0.0, maximum=1.0):
        raise AppearanceFidelityContractError(["appearance_fidelity_retained_threshold_invalid"])
    if retained_fraction < float(minimum_retained):
        blockers.append("retained_splat_fraction_below_threshold")
    if int(render_input["sh_degree"]) != int(source["sh_degree"]):
        blockers.append("spherical_harmonics_degree_not_preserved")
    source_basis = source["coordinate_basis_digest"]
    render_basis = render_input["coordinate_basis_digest"]
    if source_basis != render_basis and (
        render_input.get("basis_conversion_exact") is not True
        or not _digest(render_input.get("basis_conversion_receipt_digest"))
    ):
        blockers.append("coordinate_basis_conversion_unqualified")

    for key in ("renderer_id",):
        if not isinstance(renderer.get(key), str) or not str(renderer[key]).strip():
            raise AppearanceFidelityContractError([f"appearance_fidelity_{key}_invalid"])
    for key in ("implementation_digest", "runtime_digest"):
        if not _digest(renderer.get(key)):
            raise AppearanceFidelityContractError([f"appearance_fidelity_renderer_{key}_invalid"])
    if renderer.get("native_3dgs") is not True:
        blockers.append("native_3dgs_renderer_required")
    if renderer.get("full_anisotropic_gaussians") is not True:
        blockers.append("full_anisotropic_gaussian_rendering_required")
    if int(renderer.get("maximum_sh_degree", -1)) < int(source["sh_degree"]):
        blockers.append("renderer_spherical_harmonics_support_insufficient")

    for key in (
        "source_frame_digest",
        "rendered_frame_digest",
        "camera_spec_digest",
        "camera_basis_digest",
    ):
        if not _digest(comparison.get(key)):
            raise AppearanceFidelityContractError([f"appearance_fidelity_{key}_invalid"])
    if comparison["camera_basis_digest"] != render_basis:
        blockers.append("camera_coordinate_basis_mismatch")
    metrics = comparison.get("metrics")
    if not isinstance(metrics, Mapping):
        raise AppearanceFidelityContractError(["appearance_fidelity_comparison_metrics_invalid"])
    threshold_pairs = (
        ("ssim", "minimum_ssim", lambda observed, threshold: observed >= threshold),
        ("psnr_db", "minimum_psnr_db", lambda observed, threshold: observed >= threshold),
        ("lpips", "maximum_lpips", lambda observed, threshold: observed <= threshold),
    )
    for metric_key, policy_key, predicate in threshold_pairs:
        if not _finite_number(metrics.get(metric_key), minimum=0.0) or not _finite_number(
            policy.get(policy_key), minimum=0.0
        ):
            raise AppearanceFidelityContractError(
                [f"appearance_fidelity_{metric_key}_measurement_invalid"]
            )
        if not predicate(float(metrics[metric_key]), float(policy[policy_key])):
            blockers.append(f"reference_frame_{metric_key}_threshold_not_met")
    if comparison.get("status") != "completed":
        blockers.append("reference_frame_comparison_not_completed")

    artifact["source_appearance"]["immutable_appearance_truth"] = True
    artifact["retained_splat_fraction"] = round(retained_fraction, 12)
    artifact["removed_splat_count"] = removed_count
    artifact["status"] = "qualified" if not blockers else "blocked"
    artifact["blockers"] = sorted(set(blockers))
    artifact["evaluation_render_authorized"] = not blockers
    artifact["appearance_is_metric_or_collision_truth"] = False
    artifact["presentation_derivative_used"] = False
    artifact["claim_ceiling"] = "qualified_appearance_render" if not blockers else "none"
    expected = canonical_digest(
        artifact, digest_field="appearance_fidelity_qualification_digest"
    )
    if supplied is not None and supplied != expected:
        raise AppearanceFidelityContractError(["appearance_fidelity_digest_mismatch"])
    artifact["appearance_fidelity_qualification_digest"] = expected
    return artifact


def select_best_fidelity_render_route(
    *,
    source_appearance_digest: str,
    site_id: str,
    task_family: str,
    appearance_candidates: Sequence[Mapping[str, Any]],
    dynamics_engine: Mapping[str, Any],
) -> dict[str, Any]:
    """Select best qualified appearance independently from the dynamics engine."""

    if not _digest(source_appearance_digest) or not site_id.strip() or not task_family.strip():
        raise AppearanceFidelityContractError(["appearance_render_route_binding_invalid"])
    dynamics = _clone(dict(dynamics_engine))
    if (
        not isinstance(dynamics.get("engine_id"), str)
        or not _digest(dynamics.get("qualification_digest"))
        or dynamics.get("status") != "qualified"
    ):
        raise AppearanceFidelityContractError(["appearance_render_dynamics_engine_invalid"])
    qualified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in appearance_candidates:
        candidate = build_appearance_fidelity_qualification(raw)
        exact_source = candidate["source_appearance"]["asset_digest"] == source_appearance_digest
        if candidate["status"] == "qualified" and exact_source:
            qualified.append(candidate)
        else:
            rejected.append(
                {
                    "appearance_fidelity_qualification_digest": candidate[
                        "appearance_fidelity_qualification_digest"
                    ],
                    "reasons": sorted(
                        set(candidate["blockers"] + ([] if exact_source else ["source_digest_mismatch"]))
                    ),
                }
            )
    qualified.sort(
        key=lambda row: (
            -float(row["reference_frame_comparison"]["metrics"]["ssim"]),
            float(row["reference_frame_comparison"]["metrics"]["lpips"]),
            -float(row["reference_frame_comparison"]["metrics"]["psnr_db"]),
            -float(row["retained_splat_fraction"]),
            -int(row["render_input"]["sh_degree"]),
            str(row["renderer"]["renderer_id"]),
            str(row["appearance_fidelity_qualification_digest"]),
        )
    )
    selected = qualified[0] if qualified else None
    route = {
        "schema_version": ROUTE_SCHEMA,
        "site_id": site_id,
        "task_family": task_family,
        "source_appearance_digest": source_appearance_digest,
        "status": "qualified_route_selected" if selected else "abstained",
        "selection_policy": "best_qualified_fidelity_then_deterministic_tie_break",
        "selected_appearance": (
            {
                "renderer_id": selected["renderer"]["renderer_id"],
                "appearance_fidelity_qualification_digest": selected[
                    "appearance_fidelity_qualification_digest"
                ],
                "render_input_digest": selected["render_input"]["asset_digest"],
                "retained_splat_fraction": selected["retained_splat_fraction"],
            }
            if selected
            else None
        ),
        "rejected_appearance_candidates": sorted(
            rejected, key=lambda row: row["appearance_fidelity_qualification_digest"]
        ),
        "dynamics_engine": dynamics,
        "appearance_renderer_selected_independently_from_dynamics": True,
        "compositor_requirements": {
            "exact_camera_binding": True,
            "depth_aware_occlusion": True,
            "official_robot_asset": True,
            "presentation_derivative_forbidden": True,
        },
        "evaluation_execution_authorized": bool(selected),
        "next_required_measurement": None if selected else "qualify_native_3dgs_reference_render",
        "claim_ceiling": "render_route_only",
    }
    route["appearance_render_route_digest"] = canonical_digest(
        route, digest_field="appearance_render_route_digest"
    )
    return route


def build_robot_appearance_composite_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    supplied = artifact.pop("robot_appearance_composite_contract_digest", None)
    artifact["schema_version"] = COMPOSITE_SCHEMA
    required_digests = (
        "appearance_fidelity_qualification_digest",
        "appearance_frame_digest",
        "robot_rgba_frame_digest",
        "robot_depth_frame_digest",
        "camera_spec_digest",
        "robot_asset_digest",
        "dynamics_runtime_result_digest",
        "compositor_implementation_digest",
        "output_frame_digest",
    )
    errors = [f"{key}_invalid" for key in required_digests if not _digest(artifact.get(key))]
    if not isinstance(artifact.get("robot_id"), str) or not artifact["robot_id"].strip():
        errors.append("robot_id_invalid")
    if artifact.get("appearance_fidelity_status") != "qualified":
        errors.append("appearance_fidelity_not_qualified")
    if artifact.get("robot_asset_source") != "official_simulator_asset":
        errors.append("official_robot_asset_required")
    if artifact.get("exact_camera_binding") is not True:
        errors.append("exact_camera_binding_required")
    if artifact.get("depth_aware_occlusion") is not True:
        errors.append("depth_aware_occlusion_required")
    if artifact.get("presentation_derivative_used") is not False:
        errors.append("presentation_derivative_forbidden_in_composite")
    if errors:
        raise AppearanceFidelityContractError(errors)
    artifact.update(
        {
            "status": "authorized_composite",
            "evaluation_input_authorized": True,
            "robot_dynamics_proven_by_composite": False,
            "physical_success_proven": False,
            "claim_ceiling": "qualified_scene_robot_visual_composite",
        }
    )
    expected = canonical_digest(
        artifact, digest_field="robot_appearance_composite_contract_digest"
    )
    if supplied is not None and supplied != expected:
        raise AppearanceFidelityContractError(["robot_appearance_composite_digest_mismatch"])
    artifact["robot_appearance_composite_contract_digest"] = expected
    return artifact


def build_presentation_derivative_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _clone(dict(value))
    supplied = artifact.pop("appearance_presentation_derivative_digest", None)
    artifact["schema_version"] = PRESENTATION_SCHEMA
    errors: list[str] = []
    for key in ("source_frame_digest", "output_frame_digest", "implementation_digest"):
        if not _digest(artifact.get(key)):
            errors.append(f"{key}_invalid")
    if artifact.get("method_id") not in _PRESENTATION_METHODS:
        errors.append("presentation_method_invalid")
    required_false = (
        "evaluation_input_allowed",
        "policy_observation_allowed",
        "target_binding_allowed",
        "metric_geometry_allowed",
        "collision_geometry_allowed",
        "qualification_routing_allowed",
    )
    if any(artifact.get(key) is not False for key in required_false):
        errors.append("presentation_derivative_authority_invalid")
    if artifact.get("presentation_only") is not True:
        errors.append("presentation_only_required")
    if errors:
        raise AppearanceFidelityContractError(errors)
    artifact["claim_ceiling"] = "presentation_only"
    expected = canonical_digest(
        artifact, digest_field="appearance_presentation_derivative_digest"
    )
    if supplied is not None and supplied != expected:
        raise AppearanceFidelityContractError(["presentation_derivative_digest_mismatch"])
    artifact["appearance_presentation_derivative_digest"] = expected
    return artifact


__all__ = [
    "COMPOSITE_SCHEMA",
    "FIDELITY_SCHEMA",
    "PRESENTATION_SCHEMA",
    "ROUTE_SCHEMA",
    "AppearanceFidelityContractError",
    "build_appearance_fidelity_qualification",
    "build_presentation_derivative_contract",
    "build_robot_appearance_composite_contract",
    "select_best_fidelity_render_route",
]
