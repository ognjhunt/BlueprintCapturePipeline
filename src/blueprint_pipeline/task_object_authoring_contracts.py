"""Neutral source/evidence translation shared by construction backends.

No agent, CAD planner, property estimator, or generated-code executor belongs here.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_object_authoring_backend import ARTISAN_BACKEND, frozen_authoring_cap


class AuthoringContractError(ValueError):
    pass


class AssetAuthoringError(RuntimeError):
    """Shared retained authoring lifecycle error, including historical readers."""


def require(condition: Any, reason: str) -> None:
    if not condition:
        raise AuthoringContractError(reason)


def read_json(path: Path) -> dict:
    require(not path.is_symlink() and path.is_file(), "authoring_record_missing_or_symlink")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "authoring_record_invalid")
    return value


def write_json(path: Path, value: Mapping) -> None:
    require(not path.is_symlink(), "authoring_output_symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict:
    require(not path.is_symlink() and path.is_file(), "authoring_file_missing_or_symlink")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": "sha256:" + digest.hexdigest(),
            "size_bytes": path.stat().st_size}


def verified_file(record: Mapping, *, root: Path | None = None) -> Path:
    path = Path(record["path"])
    if root is not None:
        require(path.resolve().is_relative_to(root.resolve()), "authoring_artifact_outside_workspace")
    observed = file_record(path)
    require(observed["sha256"] == record.get("sha256", record.get("digest"))
            and observed["size_bytes"] == record.get("size_bytes"), "authoring_source_bytes_changed")
    return path.resolve()


def dependency_artifact(dependencies: list, role: str) -> tuple[dict, Path]:
    rows = [row for dep in dependencies for row in dep.get("output_artifacts", [])
            if row.get("role") == role]
    require(len(rows) == 1, "authoring_dependency_missing_or_ambiguous:" + role)
    return rows[0], verified_file(rows[0])


def source_frames(stage_input: Mapping, dependencies: list) -> tuple[dict, list[Path]]:
    configuration = stage_input["configuration"]
    if configuration.get("source_observation_kind") == "website_capture_frames":
        from .website_object_observations import REFERENCE_ROLE, validate_observation_handoff
        _, path = dependency_artifact(dependencies, REFERENCE_ROLE)
        manifest, frames = validate_observation_handoff(path, configuration=configuration)
    else:
        from .task_evaluation_scene_configuration_render_handoff import (
            ARTIFACT_ROLE, validate_provider_render_handoff,
        )
        _, path = dependency_artifact(dependencies, ARTIFACT_ROLE)
        manifest, frames = validate_provider_render_handoff(path)
        render = stage_input["construction_envelope"]["render_inputs_result"]
        require(manifest.get("control_plane_render_result_digest")
                == (render.get("control_plane_result_digest") or render.get("result_digest")),
                "authoring_render_handoff_binding_changed")
    require(bool(frames), "authoring_source_views_missing")
    # Validate real bytes, with no framework-wide reference-count truncation.
    from PIL import Image
    for frame in frames:
        file_record(frame)
        with Image.open(frame) as image:
            image.verify()
    return dict(manifest), list(frames)


def build_construction_request(stage_input: Mapping, dependencies: list,
                               frames: list[Path], frame_manifest: Mapping) -> dict:
    configuration = stage_input["configuration"]
    envelope = stage_input["construction_envelope"]
    require(configuration.get("authoring_backend") == ARTISAN_BACKEND, "artisan_backend_required")
    require(configuration.get("physics_authority_granted_by_authoring") is False,
            "authoring_cannot_grant_physical_authority")
    source, _ = dependency_artifact(dependencies, "source_object_candidate_mesh")
    bounds = configuration["metric_envelope"]
    low, high = bounds["minimum_xyz_m"], bounds["maximum_xyz_m"]
    require(len(low) == len(high) == 3 and all(isinstance(v, (int, float))
            and not isinstance(v, bool) and math.isfinite(v) for v in [*low, *high])
            and all(a < b for a, b in zip(low, high, strict=True)), "authoring_dimensions_invalid")
    dimensions = [float(b - a) for a, b in zip(low, high, strict=True)]
    body = configuration.get("body_extent_m")
    if body is not None:
        require(body.get("frame") == "assembly_front_+X_up_Z", "authoring_body_dimension_frame_invalid")
        dimensions = [float(body[axis]) for axis in ("depth", "width", "height")]
        require(all(math.isfinite(d) and d > 0 for d in dimensions), "authoring_dimensions_invalid")
    tolerance = configuration.get("maximum_export_error_m", 0.001)
    require(isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
            and 0 < tolerance <= 0.001, "authoring_export_tolerance_invalid")
    authority = configuration.get("dimension_authority") or "source_geometry"
    uncertainty = configuration.get("dimension_uncertainty_m")
    if uncertainty is None:
        uncertainty = bounds.get("dimension_uncertainty_m")
    if uncertainty is None:
        uncertainty = [d * float(bounds.get("maximum_dimension_relative_error", 0)) for d in dimensions]
    require(len(uncertainty) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and 0 <= v < d for v, d in zip(uncertainty, dimensions, strict=True)),
            "authoring_dimension_uncertainty_invalid")
    evidence = verified_physical_evidence(configuration, envelope)
    from PIL import Image
    for frame in frames:
        with Image.open(frame) as image:
            image.verify()
    local_to_scene = configuration.get("local_to_scene")
    if local_to_scene is None:
        front = (configuration.get("mechanism") or {}).get("estimated_front_normal_world")
        yaw = math.atan2(float(front[1]), float(front[0])) if front else 0.0
        local_to_scene = {"translation_m": [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]],
            "orientation_xyzw": [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)],
            "authority": "retained_estimated_front_normal" if front else "construction_metric_envelope_frame",
            "independent_scene_registration_still_required": True}
    # Preserve the complete issued configuration and observed-view metadata.
    result = {"schema_version": "task_object_construction_request.v1",
        "backend": ARTISAN_BACKEND, "run_id": stage_input["run_id"],
        "scene_id": configuration["scene_id"],
        "replacement_identity": configuration["replacement_identity"],
        "source_object_identity": configuration["source_object_identity"],
        "source_candidate": source, "configuration_sha256": stage_input["configuration_sha256"],
        "source_commit": stage_input["source_commit"],
        "dimensions_m": dimensions, "dimension_authority": authority,
        "dimension_source_digest": configuration.get("dimension_source_digest") or canonical_digest({
            "metric_envelope": bounds, "body_extent_m": body, "authority": authority}),
        "dimension_uncertainty_m": uncertainty, "maximum_export_error_m": float(tolerance),
        "coordinate_frame": {"up_axis": "Z", "meters_per_unit": 1.0,
                             "origin": "xy_center_bottom_z_zero"},
        "local_to_scene": local_to_scene,
        "authoring_budget": configuration["authoring_budget"],
        "authoring_models": configuration["authoring_models"],
        "physical_evidence": evidence,
        "source_frames": [{**file_record(path), "role": "observed_capture_view" if
            configuration.get("source_observation_kind") == "website_capture_frames" else "derived_source_render"} for path in frames],
        "source_frame_manifest": dict(frame_manifest),
        "source_frame_manifest_digest": canonical_digest(frame_manifest),
        "view_selection": {"policy": "all_admitted_handoff_views", "omitted": []},
        "source_envelope_digest": canonical_digest(envelope),
        "configuration": dict(configuration),
        "construction_constraints": configuration.get("construction_constraints") or {},
        "joint_constraints": source_joint_constraints(configuration),
        "support_and_placement": envelope.get("task") or envelope.get("recipe"),
        "claim_ceiling": "development_only", "generated_design_board_is_observed_truth": False}
    frozen_authoring_cap(configuration)
    result["request_digest"] = canonical_digest(result, digest_field="request_digest")
    return result


def source_joint_constraints(configuration: Mapping) -> dict:
    mechanism = configuration.get("mechanism") or {}
    kind = mechanism.get("joint_type")
    axis = mechanism.get("joint_axis_asset_frame")
    basis = "retained_joint_axis_constraint"
    if axis is None and kind == "prismatic":
        axis = mechanism.get("opening_axis_asset_frame")
        basis = "retained_slide_direction"
    if axis is None and kind == "revolute":
        # Opening/pull direction is +X; a hinge rotates about Y or Z. Preserve
        # the retained edge's outward-opening sign without a family CAD plan.
        edge = configuration.get("hinge_edge") or mechanism.get("hinge_edge")
        axis = {"bottom": [0., 1., 0.], "top": [0., -1., 0.],
                "left": [0., 0., -1.], "right": [0., 0., 1.]}.get(edge)
        basis = "retained_hinge_edge_construction_constraint" if axis else "framework_candidate_axis"
    if axis is not None:
        require(len(axis) == 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in axis)
                and math.isclose(sum(v*v for v in axis), 1., abs_tol=1e-6), "authoring_joint_axis_invalid")
    return {"physical_joint_axis_asset_frame": axis, "axis_basis": basis,
            "pull_direction_asset_frame": mechanism.get("opening_axis_asset_frame"),
            "pivot_asset_m": mechanism.get("pivot_asset_m"), "travel_authority": mechanism.get("travel_authority")}


def verify_construction_request(request: Mapping) -> None:
    require(request.get("request_digest") == canonical_digest(request, digest_field="request_digest"),
            "authoring_request_changed")
    for row in request["source_frames"]:
        verified_file(row)
    verified_file(request["source_candidate"])
    for row in request.get("physical_evidence", []):
        verified_file(row["retained_bytes"])


def verified_physical_evidence(configuration: Mapping, envelope: Mapping) -> list[dict]:
    from .task_object_physical_property_review import EvidenceReference, PropertyValue
    rows = []
    for value in configuration.get("physical_evidence") or []:
        ref = EvidenceReference.model_validate(value)
        parsed = urlparse(ref.uri)
        if parsed.scheme in ("", "file") and parsed.netloc in ("", "localhost"):
            path = Path(unquote(parsed.path))
        else:
            bindings = [r for r in envelope.get("materialized_references", [])
                        if r.get("digest") == "sha256:" + ref.sha256
                        and r.get("full_byte_service_account_readback_passed") is True]
            require(len(bindings) == 1, "authoring_physical_evidence_retained_bytes_required")
            path = Path(bindings[0]["materialized_path"])
        record = file_record(path)
        require(path.is_absolute() and record["sha256"] == "sha256:" + ref.sha256,
                "authoring_physical_evidence_digest_changed")
        rows.append({**ref.model_dump(mode="json"), "retained_bytes": record})
    ids = {row["evidence_id"] for row in rows}
    require(len(ids) == len(rows), "authoring_physical_evidence_duplicate")
    measured = configuration.get("measured_physical_properties") or (configuration.get("physical_review_input") or {}).get("measured") or {}
    for name, value in measured.items():
        if value is not None:
            PropertyValue.model_validate(value)
            require(value.get("basis") == "measured" and value.get("evidence_ids")
                    and set(value["evidence_ids"]) <= ids, "authoring_measured_property_evidence_required:" + name)
            bounds = (configuration.get("required_output") or {}).get(name + "_bounds")
            require(bounds and float(bounds[0]) <= float(value["value"]) <= float(bounds[1]),
                    "authoring_measured_property_outside_admitted_range:" + name)
    return rows


def artisan_prompt(request: Mapping) -> str:
    verify_construction_request(request)
    return ("Construct this one complete task object using ARTISAN's Design, Coding, Review, "
            "simulation and repair loop. Rigid objects have exactly one body and no joints. "
            "Build an articulated assembly as one model, preserving the admitted passive and "
            "coupled joints, hollow interiors, handle access and full required travel. "
            "All dimensions below are in metres. Match the nominal dimensions within "
            "maximum_export_error_m; uncertainty is source uncertainty, not permission to "
            "change dimensions. Use the supplied origin and axes. Measurement authority, "
            "measured properties, axes, pivots and travel cannot be rewritten. "
            "For a hinge, opening_axis_asset_frame is the pull direction, not the rotation axis; "
            "use joint_constraints.physical_joint_axis_asset_frame when supplied. Keep required "
            "part IDs and target_joint_id as model names. Name the graspable handle visual "
            "task_handle, so its cooked collision pieces can be checked after export. "
            "Label unobserved geometry as generated. "
            "For a hollow articulated object also write assets/blueprint_candidate_annotations.v1.json "
            "with schema_version=artisan_candidate_annotations.v1, request_digest copied below, "
            "and interior_cavities: [{cavity_id, link_id, opening_center_link_m:[x,y,z], "
            "opening_width_m, opening_height_m, inner_depth_m, basis:generated_candidate}]. "
            "The openings face +X in the link frame. Use retained cavity constraints verbatim "
            "when supplied; these probe regions are candidates, not measurements. "
            "Generated design-board imagery is a proposal and never a capture observation. "
            "Preserve mass/friction/restitution measurements and document estimated properties "
            "and limitations. Export a reviewed candidate with USD as well as URDF/MJCF. "
            "Do not introduce a fake joint for a rigid object.\n\n"
            + canonical_json(request) + "\n")
