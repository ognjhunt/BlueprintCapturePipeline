"""Immutable SAM precursor plans for one production construction submission."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import (
    checked_file, read, require, sha,
)

SCHEMA = "task_evaluation_sam31_preparation_plan.v1"
PROFILE_SCHEMA = "task_evaluation_sam31_preparation_profile.v1"
PROFILE_ENV = "BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE"
HOST_ROOTS = (Path("/var/lib/blueprint/task-evaluation-inputs"),)
PHASES = (
    "source_selections", "standard_splat_conversion", "calibrated_views",
    "sam31_inputs", "sam31_tracking", "sam31_review", "calibrated_masks",
    "removal_freezes", "contribution_sweep", "segment_cutout",
)


def file_record(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {"path": str(source), "sha256": sha(source), "size_bytes": source.stat().st_size}


def camera_policy(source_min: list[float], source_max: list[float]) -> dict[str, Any]:
    """Sixteen distinct metric proposals; the renderer must verify actual visibility."""
    extent = max(high - low for low, high in zip(source_min, source_max, strict=True))
    require(math.isfinite(extent) and extent > 0, "sam31_source_bounds_invalid")
    views = []
    for index in range(16):
        angle = 2.0 * math.pi * index / 16.0
        radius = extent * (1.5, 1.8, 2.1, 2.4)[index % 4]
        height = extent * (0.65, 0.95, 1.3, 1.7)[index % 4]
        views.append({
            "camera_id": f"source-{index + 1:02d}",
            "position_offset_m": [round(radius * math.cos(angle), 9),
                                  round(radius * math.sin(angle), 9), round(height, 9)],
            "target_offset_m": [0.0, 0.0, 0.0],
        })
    return {"generator": "translated_target_coverage_v1",
            "orbit_only_forbidden": True, "views": views}


def build_sam31_preparation_plan(
    *, source_commit: str, task: dict[str, Any], host_inputs: dict[str, Path],
    source_min: list[float], source_max: list[float], server_profile_path: Path,
) -> dict[str, Any]:
    profile = read(server_profile_path, digest_field="profile_digest")
    require(profile.get("schema_version") == PROFILE_SCHEMA and
            profile.get("source_commit") == source_commit, "sam31_profile_commit_mismatch")
    require(profile.get("review_model") == "gpt-5.6-terra" and
            profile.get("review_maximum_cost_usd") == 1.0 and
            profile.get("candidate_policy_queried") is False,
            "sam31_profile_scope_invalid")
    plan = {
        "schema_version": SCHEMA, "source_commit": source_commit,
        "task_identity": task["task_identity"], "scene_identity": task["scene_identity"],
        "publisher_scene_id": task["publisher_scene_id"],
        "phase_sequence": list(PHASES), "review_kind": "ai",
        "review_model": "gpt-5.6-terra", "human_review_required": False,
        "host_inputs": {name: file_record(path) for name, path in host_inputs.items()},
        # Only a digest crosses publication. Operator secret/configuration paths
        # are resolved from the service environment, never from client parameters.
        "server_profile_sha256": sha(server_profile_path),
        "camera_policy": camera_policy(source_min, source_max),
        "rendering": {"renderer": "reference_spark_renderer_exact_camera",
                      "graphics_backend": ("egl" if profile.get("calibrated_views", {}).get("hardware_required") is True else "swiftshader"),
                      "width": 1280, "height": 1280,
                      "vertical_fov_deg": 55.0},
        "mask_policy": {"authority": "publisher_target_obb_plus_contained_gaussians",
                        "minimum_contained_gaussians": 16, "dilation_pixels": 8,
                        "maximum_image_fraction": 0.85,
                        "visual_contribution_threshold_8bit": 8,
                        "minimum_visible_target_fraction": 0.01},
        "claim_boundary": {"evaluation_authorized": False,
                           "robot_reachability_established": False,
                           "candidate_policy_queried": False,
                           "raw_source_upload_allowed": False},
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def validate_sam31_preparation_plan(
    plan: dict[str, Any], *, source_commit: str,
    approved_roots: tuple[Path, ...] = HOST_ROOTS,
) -> dict[str, Any]:
    require(plan.get("schema_version") == SCHEMA and plan.get("source_commit") == source_commit
            and plan.get("phase_sequence") == list(PHASES)
            and plan.get("review_kind") == "ai" and plan.get("human_review_required") is False
            and plan.get("review_model") == "gpt-5.6-terra"
            and plan.get("plan_digest") == canonical_digest(plan, digest_field="plan_digest"),
            "sam31_plan_invalid")
    boundary = plan.get("claim_boundary", {})
    require(all(boundary.get(key) is False for key in (
        "evaluation_authorized", "robot_reachability_established",
        "candidate_policy_queried", "raw_source_upload_allowed")), "sam31_plan_authority_invalid")
    inputs = plan.get("host_inputs", {})
    require(set(inputs) == {"task_request", "installation_receipt", "publisher_intake",
                           "source_preparation_receipt", "interiorgs_terms"},
            "sam31_plan_inputs_invalid")
    for row in inputs.values():
        require(isinstance(row, dict) and set(row) == {"path", "sha256", "size_bytes"},
                "sam31_plan_reference_invalid")
        path = Path(row["path"])
        require(path.is_absolute() and any(
            path.resolve().is_relative_to(root.resolve()) for root in approved_roots),
            "sam31_plan_reference_outside_roots")
        checked_file(path, row)
    return plan
