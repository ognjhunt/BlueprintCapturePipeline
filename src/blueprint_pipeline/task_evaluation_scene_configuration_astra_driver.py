"""Stage-3 Astra CAD/Blender candidate authoring behind existing parent admission."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
from functools import wraps
import importlib.util
import hashlib
import inspect
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse

from .agent_operator_runtime import LIVE_AGENTS_SDK_ENV
from .asset_authoring_sandbox import SandboxedAssetRunner
from .astra_cad_skill_runtime import execute_mac_candidate, verify_cad_sources
from .decision_evidence_contracts import canonical_digest, canonical_json
from .production_blender_runtime import validate_runtime
from .task_evaluation_scene_configuration_builtin_producers import TOOLCHAIN_ROOT_ENV, _validate_toolchain
from .task_evaluation_scene_configuration_content_agents_driver import (
    _ADAPTER_ID, _DEPENDENCIES_ENV, _INPUT_ENV, _OUTPUT_ENV, _PACKAGE_ENV, _RESULT_ENV,
    _dependency_candidate, _file_record, _materialize_cad_skill_runtime, _metric_envelope_spec,
    _physics_bounds, _read, _reference_frames, _required_path, _sha256,
    _validate_metric_envelope_dimensions,
)
from .task_evaluation_scene_configuration_openai_gate import (
    scene_configuration_openai_stage_gate, scene_configuration_openai_stage_scope,
)
from .task_evaluation_scene_configuration_astra_phase_adoption import (
    materialize_automatic_phase_adoption, prepare_phase_adoption,
)
from .task_evaluation_scene_configuration_render_inputs import _materialized
from .task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION, _validate_dependencies, _validate_input,
)
from .task_object_articulated_packaging import (
    AUTHORING_RESULT_SCHEMA_VERSION as ARTICULATED_AUTHORING_RESULT_SCHEMA_VERSION,
    COMPLETION_SCHEMA_VERSION as ARTICULATED_COMPLETION_SCHEMA_VERSION,
    DRAWER_FAMILY, HANDLE_PROTRUSION_M, articulation_graph_from_plan, assembly_contract, assembly_family,
    package_astra_articulated_candidate, plan_articulated_assembly,
)
from .task_object_astra_authoring import (
    AssetAuthoringError,
    AuthoringRequest, budgeted_invoker, execute_asset_authoring, file_record, validate_request,
)
from .task_object_physical_property_review import EvidenceReference

BACKEND = "astra_cad_blender_v1"
ARTICULATED_AUTHORING_SCHEMA_VERSION = "articulated_replacement_authoring_configuration.v1"
ARTICULATED_GRAPH_SCHEMA_VERSION = "task_evaluation_articulated_replacement_graph.v1"
ARTICULATED_RECEIPT_SCHEMA_VERSION = "task_evaluation_articulated_replacement_authoring_result.v1"
BLENDER_ROOT_ENV = "BLUEPRINT_BLENDER_RUNTIME_ROOT"
_CAD_PACKAGE_FILES = ("text_to_cad_skills_source.zip", "multi_agent_cad_source.zip",
                      "cad_skill_source_receipt.json", "multi_agent_cad_skill.md")


class AstraStageError(RuntimeError):
    """The parent stage's evidence, runtime, or inference admission refused."""


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _stage_source_binding(request, stage_input, source_record, rights_path):
    semantic = request.model_dump(mode="json")
    for key in ("request_digest", "expected_production_commit"):
        semantic.pop(key)
    value = {"schema_version": "astra_same_run_source_binding.v1", "run_id": request.run_id,
        "authoring_input_digest": canonical_digest(semantic), "source_candidate": dict(source_record),
        "rights_admission": file_record(rights_path), "configuration_sha256": stage_input["configuration_sha256"]}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    return value


def _verify_physical_evidence(values: Any, envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise AstraStageError("astra_physical_evidence_invalid")
    verified = []
    for value in values:
        reference = EvidenceReference.model_validate(value)
        parsed = urlparse(reference.uri)
        if parsed.scheme in ("", "file") and parsed.netloc in ("", "localhost"):
            path = Path(unquote(parsed.path))
        else:
            rows = [row for row in envelope.get("materialized_references", [])
                    if row.get("digest") == "sha256:" + reference.sha256
                    and row.get("full_byte_service_account_readback_passed") is True]
            if len(rows) != 1:
                raise AstraStageError("astra_physical_evidence_requires_retained_bytes")
            path = Path(str(rows[0].get("materialized_path") or ""))
        if not path.is_absolute() or file_record(path)["sha256"] != "sha256:" + reference.sha256:
            raise AstraStageError("astra_physical_evidence_digest_mismatch")
        verified.append(reference.model_dump(mode="json"))
    return verified


def build_authoring_request(stage_input: Mapping[str, Any], source_record: Mapping[str, Any],
                            references: list[Path], rights: Mapping[str, Any]) -> AuthoringRequest:
    """Translate retained data without inventing physical measurements or rounding geometry."""
    configuration = stage_input["configuration"]
    # Fail closed before any inference: this builder briefs one rigid solid.
    # An articulated assembly configuration needs the articulated brief; a
    # rigid brief would spend CAD budget on a single body that packaging
    # and static qualification then refuse.
    if configuration.get("schema_version") != "rigid_replacement_authoring_configuration.v1":
        raise AstraStageError("astra_authoring_configuration_kind_unsupported:"
                              + str(configuration.get("schema_version") or ""))
    disclosure = configuration.get("provider_disclosure") or {}
    website_capture = configuration.get("source_observation_kind") == "website_capture_frames"
    if website_capture:
        from .website_native_inputs import validate_website_authoring_disclosure
        validate_website_authoring_disclosure(envelope=stage_input["construction_envelope"],
            configuration=configuration, rights=rights)
    else:
        if (rights.get("status") != "admitted_for_internal_development"
                or rights.get("private_provider_processing_allowed") is not True
                or rights.get("provider_training_allowed") is not False
                or rights.get("public_redistribution_allowed") is not False):
            raise AstraStageError("astra_derived_disclosure_not_admitted")
    if (configuration.get("authoring_backend") != BACKEND
            or disclosure.get("derived_views_and_metric_envelope") is not True
            or disclosure.get("provider_training") is not False
            or disclosure.get("public_redistribution") is not False):
        raise AstraStageError("astra_derived_disclosure_not_admitted")
    envelope = _metric_envelope_spec(configuration)
    dimensions = envelope["expected_dimensions_m"]
    uncertainty = configuration.get("dimension_uncertainty_m")
    uncertainty_note = "Supplied source-geometry uncertainty; not physical measurement."
    if uncertainty is None:
        uncertainty = configuration["metric_envelope"].get("dimension_uncertainty_m")
    if uncertainty is None:
        uncertainty = [d * envelope["maximum_dimension_relative_error"] for d in dimensions]
        uncertainty_note = ("Source-envelope relative tolerance used as an explicit uncertainty proxy; "
                            "it is not measured accuracy or a permission to round nominal dimensions.")
    if (not isinstance(uncertainty, (tuple, list)) or len(uncertainty) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 < v < dimensions[i] for i, v in enumerate(uncertainty))):
        raise AstraStageError("astra_source_geometry_uncertainty_missing_or_invalid")
    identity = configuration["replacement_identity"]
    owner = str(configuration.get("authoring_target") or "").strip()
    source_identity = configuration.get("source_object_identity")
    if not owner or not source_identity or not references:
        raise AstraStageError("astra_owner_identity_or_reference_missing")
    geometry_id = "retained_source_geometry"
    evidence = [{"evidence_id": geometry_id, "uri": str(source_record["path"]),
                 "sha256": str(source_record["digest"]).removeprefix("sha256:"), "kind": "source_geometry",
                 "excerpt": "Retained source object identity and metric envelope: " + canonical_json({
                     "source_object_identity": source_identity, "metric_envelope": envelope})}]
    frames = []
    if website_capture and configuration.get("dimension_authority") != "estimated":
        raise AstraStageError("astra_website_dimensions_must_remain_estimated")
    for index, reference in enumerate(references):
        record = file_record(reference)
        frames.append({"path": record["path"], "sha256": record["sha256"], "role": "observed_source",
                       "description": (f"Original website capture frame {index}; retain observed appearance. "
                                       "Geometry and scale inferred from it remain estimates." if website_capture else
                                       f"Digest-bound stage-1 appearance view {index}; derived source render, not physical truth.")})
        evidence.append({"evidence_id": f"retained_source_view_{index}", "uri": record["path"],
                         "sha256": record["sha256"].removeprefix("sha256:"), "kind": "material_observation",
                         "excerpt": ("Task-selected object in the original website capture frame." if website_capture else
                                     "Owner-described object shown in the retained source-derived appearance view.")})
    evidence.extend(_verify_physical_evidence(configuration.get("physical_evidence", []),
                                              stage_input["construction_envelope"]))
    if len({row["evidence_id"] for row in evidence}) != len(evidence):
        raise AstraStageError("astra_duplicate_physical_evidence")
    material = str(configuration.get("material_description") or
                   f"Infer material conservatively from the retained owner description and reference views: {owner}")
    appearance = configuration.get("appearance", "unknown")
    physical = {
        "object_id": identity["id"], "object_description": owner, "material_description": material,
        "appearance": appearance, "dimensions": {},
        "measured": configuration.get("measured_physical_properties") or dict.fromkeys(
            ("mass_kg", "static_friction", "dynamic_friction", "restitution")),
        "proposed": None,
        "optical_material": {"name": material, "transmission": 0.0, "opacity": 1.0},
        "admitted_restitution": dict(zip(("lower", "upper"), _physics_bounds(configuration)["restitution"])),
        "evidence": evidence,
    }
    for index, axis in enumerate(("x_m", "y_m", "z_m")):
        physical["dimensions"][axis] = {"value": dimensions[index], "basis": "estimated",
            "interval": {"lower": dimensions[index] - uncertainty[index], "upper": dimensions[index] + uncertainty[index]},
            "rationale": "Exact nominal source-geometry construction constraint.",
            "uncertainty": uncertainty_note, "evidence_ids": [geometry_id]}
    value = {
        "schema_version": "task_object_astra_authoring_request.v1", "run_id": stage_input["run_id"],
        "object_id": identity["id"], "owner_description": owner,
        "role": configuration.get("role", "task_object"), "dimensions_m": dimensions,
        "dimension_authority": "estimated" if website_capture else "source_geometry",
        "dimension_source_digest": source_record["digest"],
        "dimension_uncertainty_m": uncertainty, "coordinate_frame": "object_center_xy_bottom_z_z_up_meters",
        "maximum_export_error_m": configuration.get("maximum_export_error_m", 0.00001),
        "source_frames": frames, "physical_review_input": physical,
        "construction_constraints": canonical_json({"owner_description": owner,
            "source_object_identity": source_identity, "replacement_identity": identity,
            "exact_nominal_dimensions_m": dimensions, "source_uncertainty_note": uncertainty_note,
            "required_output": configuration["required_output"],
            "additional_constraints": configuration.get("construction_constraints", "")}),
        "private_provider_processing_allowed": True, "provider_training_allowed": False,
        "public_redistribution_allowed": False, "expected_production_commit": stage_input["source_commit"],
        "request_digest": "",
    }
    value["request_digest"] = "sha256:" + "0" * 64
    value = AuthoringRequest.model_validate(value).model_dump(mode="json")
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    request = validate_request(value)
    if request.maximum_export_error_m * 1000 > 0.1:
        raise AstraStageError("astra_cad_export_tolerance_unsupported")
    evidence_by_id = {row.evidence_id: row for row in request.physical_review_input.evidence}
    for measured in request.physical_review_input.measured:
        prop = measured[1]
        if prop is not None and (prop.basis != "measured" or any(
            evidence_id not in evidence_by_id for evidence_id in prop.evidence_ids
        ) or not any(evidence_by_id[evidence_id].kind in {"physical_measurement", "capture_measurement"}
                     for evidence_id in prop.evidence_ids)):
            raise AstraStageError("astra_measured_property_evidence_invalid")
    return request


def _articulated_physics_bounds(configuration: Mapping[str, Any],
                                plan: Mapping[str, Any] | None = None) -> dict[str, dict[str, list[float]]]:
    """Per-part admitted bounds: the body carries the assembly mass, the moving part its own.

    Fixed interior parts (racks, baskets) need their own preregistered
    ``fixed_part_mass_kg_bounds``; none is borrowed from another part.
    """
    shared = _physics_bounds(configuration)
    required = configuration.get("required_output") or {}

    def interval(key: str) -> list[float]:
        value = required.get(key)
        if (not isinstance(value, list) or len(value) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
                or not 0 < value[0] <= value[1]):
            raise AstraStageError(f"astra_articulated_{key.removesuffix('_mass_kg_bounds')}_mass_bounds_invalid")
        return [float(value[0]), float(value[1])]

    task_part = interval("task_part_mass_kg_bounds")
    if plan is None or plan.get("family", DRAWER_FAMILY) == DRAWER_FAMILY:
        return {"carcass": {**shared}, "drawer": {**shared, "mass_kg": task_part}}
    return {part_id: {**shared} if spec["link_role"] == "body" else
            {**shared, "mass_kg": task_part if spec["link_role"] == "task_part" else interval("fixed_part_mass_kg_bounds")}
            for part_id, spec in plan["parts"].items()}


def articulated_frame_descriptions(configuration: Mapping[str, Any], references: list[Path]) -> list[dict[str, Any]]:
    """Source frames for one object's part briefs, captioned from its own ``reference_frames``.

    Each retained frame is described by what the contract says it shows
    (visible parts, part state, view, why it was chosen). A legacy drawer
    configuration without reference frames keeps its historical caption; an
    object created from a description has no frames and claims no observation.
    """
    family = assembly_family(configuration)
    contract = assembly_contract(configuration, family)
    website_capture = configuration.get("source_observation_kind") == "website_capture_frames"
    records = [file_record(path) for path in references]
    if not contract["captured"]:
        if records:
            raise AstraStageError("astra_articulated_created_object_frames_invalid")
        return []
    rows = contract["reference_frames"]
    if not rows:
        return [{"path": record["path"], "sha256": record["sha256"], "role": "observed_source",
                 "description": (f"Original website capture frame {index}: the whole assembly, closed. Retain observed "
                                 "front appearance (wood-grain fronts, silver bar handles, grey carcass edges); the "
                                 "interior is unobserved." if website_capture else
                                 f"Digest-bound stage-1 appearance view {index}; derived source render, not physical truth.")}
                for index, record in enumerate(records)]
    by_digest = {row["sha256"]: row for row in rows}
    if sorted(record["sha256"] for record in records) != sorted(by_digest):
        raise AstraStageError("astra_articulated_reference_frames_disagree_with_retained_frames")
    # The retained frames go inline into every provider request; refuse any
    # that the configured provider's request bound would not admit.
    from .authoring_frame_budget import FrameBudgetError, check_transmitted_frames
    try:
        check_transmitted_frames(references, provider=str(configuration.get("authoring_model_provider") or "openai"))
    except FrameBudgetError as exc:
        raise AstraStageError("astra_" + str(exc)) from exc
    labels = {row["part_id"]: row["label"] for row in contract["required_parts"]}
    task_label = str(configuration["mechanism"]["task_part_label"])
    states = {"closed": "closed", "partially_open": "partially open", "open": "open",
              "not_visible": "not visible in this frame"}
    frames = []
    for record in records:
        row = by_digest[record["sha256"]]
        visible = ", ".join(labels.get(part, part.replace("_", " ")) for part in row["visible_parts"])
        frames.append({"path": record["path"], "sha256": record["sha256"], "role": "observed_source",
                       "description": (
                           f"Original capture frame {row['frame_id']} at {float(row['timestamp_seconds']):.2f} s, "
                           f"{row['view'].strip()} view; the {task_label} is {states[row['part_state']]}. "
                           f"Visible parts: {visible or 'none of the required parts'}. "
                           f"Chosen for: {row['reason'].strip()}. Retain observed appearance only where it is visible; "
                           "geometry and scale inferred from it remain estimates.")})
    return frames


def build_articulated_authoring_requests(stage_input: Mapping[str, Any], source_record: Mapping[str, Any],
                                         references: list[Path], rights: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, AuthoringRequest]]:
    """One exact per-part brief per assembly part; the assembly plan binds their rest poses and the task joint."""
    configuration = stage_input["configuration"]
    if configuration.get("schema_version") != ARTICULATED_AUTHORING_SCHEMA_VERSION:
        raise AstraStageError("astra_authoring_configuration_kind_unsupported:" + str(configuration.get("schema_version") or ""))
    disclosure = configuration.get("provider_disclosure") or {}
    website_capture = configuration.get("source_observation_kind") == "website_capture_frames"
    if website_capture:
        from .website_native_inputs import validate_website_authoring_disclosure
        validate_website_authoring_disclosure(envelope=stage_input["construction_envelope"],
            configuration=configuration, rights=rights)
    elif (rights.get("status") != "admitted_for_internal_development"
            or rights.get("private_provider_processing_allowed") is not True
            or rights.get("provider_training_allowed") is not False
            or rights.get("public_redistribution_allowed") is not False):
        raise AstraStageError("astra_derived_disclosure_not_admitted")
    if (configuration.get("authoring_backend") != BACKEND
            or disclosure.get("derived_views_and_metric_envelope") is not True
            or disclosure.get("provider_training") is not False
            or disclosure.get("public_redistribution") is not False):
        raise AstraStageError("astra_derived_disclosure_not_admitted")
    if website_capture and configuration.get("dimension_authority") != "estimated":
        raise AstraStageError("astra_website_dimensions_must_remain_estimated")
    envelope = _metric_envelope_spec(configuration)
    tolerance = float(envelope["maximum_dimension_relative_error"])
    try:
        plan = plan_articulated_assembly(configuration)
    except AssetAuthoringError as exc:
        raise AstraStageError("astra_" + str(exc)) from exc
    hypothesis = plan.get("development_geometry_hypothesis")
    if hypothesis is not None:
        retained_frame_digests = {file_record(path)["sha256"] for path in references}
        if not set(hypothesis["evidence_frame_sha256s"]).issubset(retained_frame_digests):
            raise AstraStageError("astra_articulated_depth_hypothesis_frame_mismatch")
    plan["source_geometry_receipt"] = {
        "source_candidate_digest": source_record["digest"],
        "construction_envelope_digest": stage_input["construction_envelope"].get("envelope_digest"),
        "configuration_digest": stage_input.get("configuration_sha256"),
        "source_aabb_min_xyz_m": list(configuration["metric_envelope"]["minimum_xyz_m"]),
        "source_aabb_max_xyz_m": list(configuration["metric_envelope"]["maximum_xyz_m"]),
    }
    identity = configuration["replacement_identity"]
    owner = str(configuration.get("authoring_target") or "").strip()
    source_identity = configuration.get("source_object_identity")
    family = plan.get("family", DRAWER_FAMILY)
    contract = assembly_contract(configuration, family)
    if not owner or not source_identity or (contract["captured"] and not references):
        raise AstraStageError("astra_owner_identity_or_reference_missing")
    bounds = _articulated_physics_bounds(configuration, plan)
    uncertainty_note = ("Assembly envelope relative tolerance used as an explicit uncertainty proxy for each part; "
                        "part dimensions derive from the estimated envelope projected on the estimated front normal "
                        "and object-prior construction assumptions. None is measured.")
    if hypothesis is not None:
        uncertainty_note = ("Cabinet depth uses a development-only estimate with an explicit interval that disagrees "
                            "with the retained source AABB; other axes use source-envelope tolerance proxies. "
                            "No part dimension is physically measured.")
    elif family != DRAWER_FAMILY:
        uncertainty_note = (f"Body depth basis {plan['body_depth']['basis']}; width and height from the closed front; "
                            "part dimensions add recorded construction priors. Envelope relative tolerance is an "
                            "explicit uncertainty proxy. None is measured.")
    frames = articulated_frame_descriptions(configuration, references)
    if not frames:  # Created-from-description objects need a generated-specification brief first.
        raise AstraStageError("astra_articulated_created_object_authoring_unsupported")
    base_evidence = [{"evidence_id": "retained_source_geometry", "uri": str(source_record["path"]),
                      "sha256": str(source_record["digest"]).removeprefix("sha256:"), "kind": "source_geometry",
                      "excerpt": "Retained source object identity and metric envelope: " + canonical_json({
                          "source_object_identity": source_identity, "metric_envelope": envelope})}]
    for index, record in enumerate(frames):
        base_evidence.append({"evidence_id": f"retained_source_view_{index}", "uri": record["path"],
                              "sha256": record["sha256"].removeprefix("sha256:"), "kind": "material_observation",
                              "excerpt": "Task-selected assembly in the original website capture frame." if website_capture
                              else "Owner-described assembly shown in the retained source-derived appearance view."})
    base_evidence.extend(_verify_physical_evidence(configuration.get("physical_evidence", []), stage_input["construction_envelope"]))
    if len({row["evidence_id"] for row in base_evidence}) != len(base_evidence):
        raise AstraStageError("astra_duplicate_physical_evidence")
    requests: dict[str, AuthoringRequest] = {}
    for part_id, spec in plan["parts"].items():
        dimensions = [float(v) for v in spec["dimensions_m"]]
        uncertainty = [d * tolerance for d in dimensions]
        if hypothesis is not None:
            depth_interval = hypothesis["depth_interval_m"]
            uncertainty[0] = max(dimensions[0] - float(depth_interval[0]),
                                 float(depth_interval[1]) - dimensions[0])
        part_object_id = f"{identity['id']}__{part_id}"
        link_ids = {row["link_id"] for row in plan["links"] if row["part_id"] == part_id}
        carried = [row for row in plan.get("required_parts") or [] if row["link_id"] in link_ids]
        if family == DRAWER_FAMILY and not contract["reference_frames"]:  # historical drawer brief
            material = (f"Infer material conservatively from the retained assembly description and reference views: {owner}. "
                        + ("Grey painted steel or laminate carcass." if part_id == "carcass" else
                           "Wood-grain laminate drawer front with a brushed silver metal bar handle; plain box behind."))
        else:
            material = (f"Infer material conservatively from the retained assembly description and reference views: {owner}. "
                        f"Part: {spec['link_role']}"
                        + (f" carrying {', '.join(row['label'] for row in carried)}" if carried else "")
                        + ". Use only finishes a reference view shows; do not assume any other.")
        physical = {
            "object_id": part_object_id, "object_description": f"{spec['link_role']} of {owner}: {spec['description']}",
            "material_description": material, "appearance": configuration.get("appearance", "unknown"), "dimensions": {},
            "measured": dict.fromkeys(("mass_kg", "static_friction", "dynamic_friction", "restitution")),
            "proposed": None, "optical_material": {"name": material, "transmission": 0.0, "opacity": 1.0},
            "admitted_restitution": dict(zip(("lower", "upper"), bounds[part_id]["restitution"])),
            "evidence": [dict(row) for row in base_evidence],
        }
        for index, axis in enumerate(("x_m", "y_m", "z_m")):
            interval = ({"lower": float(hypothesis["depth_interval_m"][0]),
                         "upper": float(hypothesis["depth_interval_m"][1])}
                        if hypothesis is not None and part_id == "carcass" and index == 0 else
                        {"lower": dimensions[index] - uncertainty[index],
                         "upper": dimensions[index] + uncertainty[index]})
            physical["dimensions"][axis] = {"value": dimensions[index], "basis": "estimated",
                "interval": interval,
                "rationale": ("Development-only cabinet-depth hypothesis; retained source depth disagreement recorded in assembly plan."
                              if hypothesis is not None and index == 0 else
                              "Part envelope derived from the estimated assembly envelope and construction assumptions."),
                "uncertainty": uncertainty_note, "evidence_ids": ["retained_source_geometry"]}
        family_constraints = ({"bay_count": plan["bay_count"], "task_bay_index": plan["task_bay_index"]}
                              if family == DRAWER_FAMILY else
                              {"assembly_family": family, "hinge_edge": plan["hinge_edge"],
                               "task_joint": {key: plan["task_joint"][key] for key in (
                                   "joint_type", "axis_asset_frame", "anchor_asset_frame_m", "limits_rad")},
                               "body_depth": plan["body_depth"]})
        cavities = [row for row in plan.get("interior_cavities") or [] if row["link_id"] in link_ids]
        constraints = {"owner_description": owner, "assembly_part_id": part_id, "link_role": spec["link_role"],
            "part_description": spec["description"], "assembly_frame": plan["assembly_frame"],
            "assembly_dimensions_m": plan["assembly_dimensions_m"], **family_constraints,
            **({"required_parts_on_this_part": [{key: row[key] for key in ("part_id", "label", "feature",
                                                                          "observed_frame_ids")} for row in carried]}
               if carried else {}),
            **({"part_features_part_frame_m": {k: v for k, v in spec["features"].items() if isinstance(v, Mapping)}}
               if family != DRAWER_FAMILY else {}),
            **({"interior_cavities_must_stay_hollow_and_open_front": cavities} if cavities else {}),
            "exact_nominal_dimensions_m": dimensions,
            "part_frame": "center_XY_bottom_Z_with_the_front_face_at_+X",
            "source_uncertainty_note": uncertainty_note, "construction_assumptions": plan["construction_assumptions"],
            "source_geometry_receipt": plan["source_geometry_receipt"],
            **({"mechanism_opening_effort_reference": plan["task_joint"]["opening_effort_reference"]}
               if "opening_effort_reference" in plan["task_joint"] else {}),
            **({"development_geometry_hypothesis": hypothesis} if hypothesis is not None else {}),
            "required_output": configuration["required_output"],
            **({"handle": spec["handle"]} if "handle" in spec else {}),
            "additional_constraints": configuration.get("construction_constraints", "")}
        value = {
            "schema_version": "task_object_astra_authoring_request.v1", "run_id": stage_input["run_id"],
            "object_id": part_object_id, "owner_description": f"{spec['link_role']} of {owner}",
            "role": "task_object", "dimensions_m": dimensions,
            "dimension_authority": "estimated" if website_capture or family != DRAWER_FAMILY else "source_geometry",
            "dimension_source_digest": source_record["digest"],
            "dimension_uncertainty_m": uncertainty, "coordinate_frame": "object_center_xy_bottom_z_z_up_meters",
            "maximum_export_error_m": configuration.get("maximum_export_error_m", 0.00001),
            "source_frames": frames, "physical_review_input": physical,
            "construction_constraints": canonical_json(constraints),
            "private_provider_processing_allowed": True, "provider_training_allowed": False,
            "public_redistribution_allowed": False, "expected_production_commit": stage_input["source_commit"],
            "request_digest": "sha256:" + "0" * 64,
        }
        value = AuthoringRequest.model_validate(value).model_dump(mode="json")
        value["request_digest"] = canonical_digest(value, digest_field="request_digest")
        request = validate_request(value)
        if request.maximum_export_error_m * 1000 > 0.1:
            raise AstraStageError("astra_cad_export_tolerance_unsupported")
        requests[part_id] = request
    return plan, requests


def _articulated_stage_source_binding(part_requests: Mapping[str, AuthoringRequest], stage_input, source_record, rights_path):
    semantics = {}
    for part_id, request in part_requests.items():
        semantic = request.model_dump(mode="json")
        for key in ("request_digest", "expected_production_commit"):
            semantic.pop(key)
        semantics[part_id] = semantic
    run_id = next(iter(part_requests.values())).run_id
    value = {"schema_version": "astra_same_run_source_binding.v1", "run_id": run_id,
        "authoring_input_digest": canonical_digest(semantics), "source_candidate": dict(source_record),
        "rights_admission": file_record(rights_path), "configuration_sha256": stage_input["configuration_sha256"],
        "assembly_parts": sorted(part_requests)}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    return value


def _reuse_prior_part_result(prior_roots: list[Path], part_id: str, request: AuthoringRequest) -> dict[str, Any] | None:
    """Adopt a completed part from the latest same-run attempt when its exact inputs and bytes still hold."""
    for prior in reversed(prior_roots):
        path = prior / "authoring" / "parts" / part_id / "result.json"
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (result.get("request_digest") != request.request_digest
                or result.get("status") != "candidate_authored_pending_native_qualification"
                or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")):
            return None
        for key in ("asset", "final_visual_mesh", "final_visual_mesh_receipt", "physical_review",
                    "physical_review_input", "geometry_readback"):
            record = result.get(key)
            if not isinstance(record, Mapping) or not Path(str(record.get("path") or "")).is_file():
                return None
            actual = file_record(Path(record["path"]))
            if any(actual.get(field) != record.get(field) for field in ("sha256", "size_bytes")):
                return None
        return result
    return None


def _author_articulated_parts(*, plan, part_requests, authored_root, runtime, prior_roots, invoker, sandbox, blender,
                              cad_root, verified_sources, authoring_instructions, configuration, authoring_executor,
                              mac_executor) -> dict[str, Any]:
    """Author each part in its own bounded session; completed parts are checkpoints for a retry."""
    parts: dict[str, Any] = {}
    reused: list[str] = []
    for part_id, part_request in part_requests.items():
        part_root = authored_root / "parts" / part_id
        prior = _reuse_prior_part_result(prior_roots, part_id, part_request)
        if prior is not None:
            part_root.mkdir(parents=True, exist_ok=True)
            _write(part_root / "request.json", part_request.model_dump(mode="json"))
            _write(part_root / "result.json", prior)
            _write(part_root / "part_adoption.json", {"status": "completed_part_adopted_from_same_run",
                "result_digest": prior["result_digest"], "new_provider_calls": 0})
            parts[part_id] = prior
            reused.append(part_id)
            continue
        arguments = dict(request_value=part_request.model_dump(mode="json"), output_root=part_root, invoker=invoker,
                         blender_runner=sandbox, blender_executable=blender["executable"],
                         authoring_instructions=authoring_instructions)
        if authoring_executor is execute_asset_authoring and configuration.get("source_observation_kind") == "website_capture_frames":
            from functools import partial
            from .task_object_agent_cad import execute_cad_program
            from .task_object_agent_session import execute_agent_authoring
            parts[part_id] = execute_agent_authoring(**arguments, budget_root=runtime / "inference" / "parts" / part_id,
                cad_executor=partial(execute_cad_program, cad_root=cad_root / "text-to-cad",
                    mac_root=cad_root / "Multi-Agent-CAD", sandbox=sandbox, verified_sources=verified_sources))
        else:
            parts[part_id] = authoring_executor(**arguments, mac_executor=mac_executor)
    authored = {"schema_version": ARTICULATED_AUTHORING_RESULT_SCHEMA_VERSION,
                "status": "parts_authored_pending_native_qualification", "model": next(iter(parts.values()))["model"],
                "plan": dict(plan), "parts": parts, "reused_part_ids": reused,
                "part_models": {part_id: part["model"] for part_id, part in parts.items()},
                "part_request_digests": {part_id: request.request_digest for part_id, request in part_requests.items()},
                "claim_ceiling": "development_only", "native_import_qualified": False,
                "scene_placement_qualified": False, "physical_equivalence_proven": False}
    authored["result_digest"] = canonical_digest(authored, digest_field="result_digest")
    _write(authored_root / "result.json", authored)
    return authored


def _managed_authoring_receipt(path: Path, authored: Mapping[str, Any]) -> dict[str, Any]:
    value = _read(path, code="agents_api_stage_receipt_missing")
    if (value.get("schema_version") != "task_asset_agents_api_stage_receipt.v1"
            or value.get("provider") != "openai" or value.get("model") != "gpt-6-sol"
            or value.get("runtime") != "openai_agents_api"
            or value.get("session_cleanup") != "deleted"
            or value.get("result_digest") != authored.get("result_digest")
            or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")):
        raise AstraStageError("agents_api_stage_receipt_invalid")
    return value


def _finish_articulated_component(*, plan, part_requests, authored, output, physics_bounds, configuration,
                                  source_record, stage_input, rights_record, cad_runtime, blender, authored_root,
                                  result_path, package_candidate, adoption_lineage=None):
    output.mkdir(parents=True, exist_ok=True)
    packaged = package_candidate(requests=part_requests, authoring_results=authored["parts"], plan=plan,
                                 output_root=output, physics_bounds=physics_bounds)
    asset = Path(packaged["asset"]["path"])
    asset_record = {"path": packaged["asset"]["path"],
                    "digest": packaged["asset"].get("digest") or packaged["asset"].get("sha256"),
                    "size_bytes": packaged["asset"]["size_bytes"]}
    if asset.is_symlink() or not asset.resolve().is_relative_to(output) or _file_record(asset) != asset_record:
        raise AstraStageError("astra_packaged_asset_binding_invalid")
    completion = packaged["physics_completion"]
    if completion.get("schema_version") != ARTICULATED_COMPLETION_SCHEMA_VERSION:
        raise AstraStageError("astra_articulated_completion_schema_invalid")
    # Every contract required part must have been planned and carried into the asset.
    planned = {row["part_id"] for row in plan.get("required_parts") or []}
    carried = {part for rows in (completion.get("required_parts_by_link") or {}).values() for part in rows}
    for row in configuration.get("required_parts") or []:
        part_id = row.get("part_id") if isinstance(row, Mapping) else None
        if part_id not in planned or part_id not in carried:
            raise AstraStageError(f"astra_articulated_required_part_unplanned:{part_id}")
    dims = plan["assembly_dimensions_m"]
    # The family's closed envelope (a door appliance's body depth includes the
    # door) with the handle protrusion; a thin slab body cannot match it.
    expected = [float(v) for v in plan.get("closed_collision_dimensions_m")
                or [dims["depth_x"] + HANDLE_PROTRUSION_M, dims["width_y"], dims["height_z"]]]
    observed = completion["collision_dimensions_m"]
    tolerance = float(configuration["metric_envelope"]["maximum_dimension_relative_error"])
    errors = [abs(observed[i] - expected[i]) / expected[i] for i in range(3)]
    if any(error > tolerance for error in errors):
        raise AstraStageError("astra_articulated_assembly_envelope_mismatch")
    completion["metric_envelope_validation"] = {
        "status": "within_development_geometry_hypothesis" if plan.get("development_geometry_hypothesis")
                  else "within_preregistered_metric_envelope",
        "frame": "assembly_frame_closed_plus_handle_protrusion",
        "expected_dimensions_m": expected, "observed_collision_dimensions_m": observed,
        "dimension_relative_errors": errors, "maximum_dimension_relative_error": tolerance}
    if plan.get("development_geometry_hypothesis"):
        completion["metric_envelope_validation"]["source_aabb_disagreement"] = {
            "source_projected_depth_m": plan["source_geometry"]["projected_depth_m"],
            "estimated_depth_m": dims["depth_x"],
            "depth_disagreement_m": plan["development_geometry_hypothesis"]["depth_disagreement_m"],
            "physical_measurement_proven": False}
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    identity = configuration["replacement_identity"]
    graph = {"schema_version": ARTICULATED_GRAPH_SCHEMA_VERSION, "asset_id": identity["id"],
        "asset_version": identity["version"], "articulation_graph": articulation_graph_from_plan(plan),
        "task_joint_prim_path": completion["task_joint_prim_path"], "task_link_prim_path": completion["task_link_prim_path"],
        "fixed_base_body_prim_path": completion["fixed_base_body_prim_path"],
        "handle_prim_paths": completion["handle_prim_paths"],
        "handle_grasp_point_link_m": completion["handle_grasp_point_link_m"],
        "link_prim_paths": {row["link_id"]: row["prim_path"] for row in completion["links"]},
        "assembly_plan": dict(plan), "physics_bounds": physics_bounds, "physics_authority_granted": False,
        "authoring_backend": BACKEND}
    graph_path = output / "replacement_graph_spec.v1.json"
    _write(graph_path, graph)
    managed_receipts = {}
    if configuration.get("authoring_agent_runtime") == "openai_agents_api":
        for part_id in part_requests:
            path = authored_root.parent / "inference" / "agents_api" / "parts" / part_id / "agents_api_stage_receipt.json"
            managed_receipts[part_id] = _managed_authoring_receipt(path, authored["parts"][part_id])
    receipt = {"schema_version": ARTICULATED_RECEIPT_SCHEMA_VERSION,
        "status": "authored_candidate_pending_qualification", "authoring_backend": BACKEND, "model": authored["model"],
        **({"provider": "anthropic"} if authored["model"] == "claude-opus-5-5" else {}),
        **({"provider": "openai", "agent_runtime": "openai_agents_api",
            "managed_agent_execution_receipts": managed_receipts} if managed_receipts else {}),
        "part_models": authored["part_models"],
        **({"completed_articulated_adoption": adoption_lineage} if adoption_lineage is not None else {}),
        "asset_kind": "articulated_assembly", "replacement_identity": identity,
        "source_candidate_digest": source_record["digest"],
        "source_candidate_claim": "source_geometry_not_observed_truth_or_physics_authority",
        "source_commit": stage_input["source_commit"], "toolchain_digest": stage_input["toolchain_digest"],
        "source_rights_admission": dict(rights_record), "cad_skill_runtime": cad_runtime, "blender_runtime": blender,
        "astra_authoring_result": _file_record(authored_root / "result.json"),
        "part_authoring_results": {part_id: _file_record(authored_root / "parts" / part_id / "result.json")
                                   for part_id in part_requests},
        "output_usd": {"sha256": _sha256(asset), "size_bytes": asset.stat().st_size},
        "candidate_physics_completion": completion, "physics_authority_granted": False, "result_digest": ""}
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    receipt_path = output / "replacement_authoring_receipt.v1.json"
    _write(receipt_path, receipt)
    result = {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, "status": "completed", "adapter_id": _ADAPTER_ID,
        "stage_id": stage_input["stage"]["stage_id"], "provider_mutations_performed": 0,
        "nested_paid_execution_requested": False, "authoring_backend": BACKEND, "model": authored["model"],
        **({"provider": "anthropic"} if authored["model"] == "claude-opus-5-5" else {}),
        **({"provider": "openai", "agent_runtime": "openai_agents_api"} if managed_receipts else {}),
        "asset_kind": "articulated_assembly",
        "artifacts": [{"role": role, **_file_record(path)} for role, path in (
            ("replacement_asset", asset), ("replacement_authoring_receipt", receipt_path), ("replacement_graph_spec", graph_path))],
        "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _write(result_path, result)
    return result


#: The geometric architect emits every section sketch and build step in ONE reply,
#: and reasoning tokens are drawn from the same budget. Scene 840938 object 219
#: (2026-09-15) planned 13 loft sections; the reply was cut mid-string inside the
#: LAST step's notes and failed to parse, so the coder node never ran and the
#: graph exported no STEP/STL -- `cad_graph_missing_exports`, after a full GPU
#: rental. It was within roughly a thousand tokens of finishing.
#:
#: The ceiling is bounded on BOTH sides. The runtime refuses a budget over 32000,
#: and the stage reserves `projected_max_cost = input + tokens * $0.00005` per
#: request against `..._CONTENT_AGENTS_MAX_COST_USD`. That run reserved three
#: requests with $0.827/$0.540/$0.316 of input against a $5.00 cap, so the cap
#: allows about `(5.00 - 1.682) / 3 / 0.00005` ~= 22000 tokens. 20000 clears the
#: observed shortfall many times over and still leaves the reservation total near
#: $4.68 -- raise the stage cap before raising this much further.
CAD_MAX_OUTPUT_TOKENS = 20000


class _StageInvoker:
    def __init__(self, invoker, run_id: str, maximum_calls: int, prior_calls: int = 0):
        self.invoker, self.run_id, self.maximum_calls, self.calls = invoker, run_id, maximum_calls, 0
        self.prior_calls = prior_calls

    def invoke(self, spec, input_value):
        if (self.calls + self.prior_calls >= self.maximum_calls or spec.run_id != self.run_id or spec.model != "gpt-6-astra"
                or spec.max_turns != 1 or spec.tool_bindings
                or spec.max_output_tokens > CAD_MAX_OUTPUT_TOKENS
                or spec.max_input_tokens is None or spec.max_input_tokens > 80000
                or spec.reasoning_effort not in {"medium", "high"}):
            raise AstraStageError("astra_stage_inference_boundary_refused")
        self.calls += 1
        return self.invoker.invoke(spec, input_value)


@contextmanager
def _stage_sdk_environment(key_path: Path):
    from agents import set_default_openai_client
    from agents.models import _openai_shared
    from openai import AsyncOpenAI
    names = ("OPENAI_API_KEY", "OPENAI_API_KEY_FILE", LIVE_AGENTS_SDK_ENV)
    previous = {name: os.environ.get(name) for name in names}
    previous_client = _openai_shared.get_default_openai_client()
    try:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["OPENAI_API_KEY_FILE"] = str(key_path)
        os.environ[LIVE_AGENTS_SDK_ENV] = "1"
        set_default_openai_client(AsyncOpenAI(api_key=key_path.read_text().strip(),
            base_url='https://api.openai.com/v1', max_retries=0, timeout=600), use_for_tracing=False)
        yield
    finally:
        _openai_shared.set_default_openai_client(previous_client)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _locked_component(function):
    @wraps(function)
    def execute(*args, **kwargs):
        values = os.environ if kwargs.get("environment") is None else kwargs["environment"]
        root = _required_path(values, _OUTPUT_ENV)
        path = root / ".astra-component.lock"
        if path.is_symlink():
            raise AstraStageError("astra_component_lock_unsafe")
        with path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AstraStageError("astra_component_already_running") from exc
            return function(*args, **kwargs)
    return execute


def prepare_astra_execution_runtime(*, runtime, package, authored_root, values,
                                    runner=subprocess.run, sandbox_factory=SandboxedAssetRunner,
                                    blender_validator=validate_runtime):
    """The real no-inference CAD/Blender boundary, shared by preflight and stage3."""
    cad_runtime = _materialize_cad_skill_runtime(runtime)
    cad_root = Path(cad_runtime["root"])
    verified_sources = {}
    for name, folder, source_id in (("mac", "Multi-Agent-CAD", "multi-agent-cad"),
                                    ("cad", "text-to-cad", "text-to-cad")):
        source_root = cad_root / folder
        archive = runtime / ("multi_agent_cad_source.zip" if name == "mac" else "text_to_cad_skills_source.zip")
        verified_sources[name] = {"root": str(source_root), "commit": cad_runtime["source_commits"][source_id],
            "tracked_file_sha256": {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in source_root.rglob("*") if path.is_file()},
            "source_diff": "", "source_receipt_digest": cad_runtime["receipt_digest"],
            "source_receipt_path": str(runtime / "cad_skill_source_receipt.json"),
            "archive_path": str(archive), "archive_sha256": _sha256(archive)}
    if "verified_sources" not in inspect.signature(execute_mac_candidate).parameters:
        raise AstraStageError("astra_mac_archive_admission_unavailable")
    verify_cad_sources(cad_root / "Multi-Agent-CAD", cad_root / "text-to-cad", verified_sources)
    if str(values.get(BLENDER_ROOT_ENV) or "").strip():
        blender_root = _required_path(values, BLENDER_ROOT_ENV)
        blender = blender_validator(blender_root, runner=runner)
    else:
        from .task_evaluation_scene_configuration_astra_runtime import materialize_packaged_blender_runtime
        blender_root = runtime / "packaged_blender"
        blender = materialize_packaged_blender_runtime(package, blender_root, runner=runner)
    for dependency in ("build123d", "langgraph", "agents"):
        if importlib.util.find_spec(dependency) is None:
            raise AstraStageError(f"astra_runtime_dependency_missing:{dependency}")
    runtime_loader = [Path(value).resolve() for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                      if value and Path(value).is_dir()]
    library_environment = {}
    library_roots = []
    for name in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        paths = [Path(value).resolve() for value in os.environ.get(name, '').split(os.pathsep)
                 if value and Path(value).is_absolute() and Path(value).is_dir()]
        if paths:
            library_environment[name] = os.pathsep.join(map(str, dict.fromkeys(paths)))
            library_roots.extend(paths)
    # Kit's Python can rely on a launcher-provided library path rather than
    # ELF RUNPATH. Also recognize its actual libpython directory when the
    # caller has already scrubbed the environment after interpreter startup.
    for parent in (Path(sys.base_prefix) / 'lib', Path(sys.base_prefix).parent):
        if any(parent.glob(f'libpython{sys.version_info.major}.{sys.version_info.minor}.so*')):
            paths = [*filter(None, library_environment.get('LD_LIBRARY_PATH', '').split(os.pathsep)), str(parent.resolve())]
            library_environment['LD_LIBRARY_PATH'] = os.pathsep.join(dict.fromkeys(paths))
            library_roots.append(parent.resolve())
    roots = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), cad_root,
             blender_root, Path(__file__).resolve().parent.parent, *runtime_loader, *library_roots]
    roots = list(dict.fromkeys(roots))
    from .asset_runtime_permissions import prepare_runtime_code_access
    _write(runtime / 'runtime_code_access.json', prepare_runtime_code_access(roots))
    sandbox = sandbox_factory(read_roots=roots, write_root=authored_root,
        executable_roots=[Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), blender_root],
        library_environment=library_environment, library_executables=[Path(sys.executable)])
    sandbox.preflight()
    probe_root = authored_root / 'tmp' / 'runtime-probe'
    probe_root.mkdir(parents=True, exist_ok=True)
    cad_probe_timeout_seconds = 180
    try:
        cad_probe = sandbox([sys.executable, "-c", "import build123d; from langgraph.graph import StateGraph; "
                             "from cadpy.generation import run_script_generator; "
                             "assert abs(build123d.Box(1,2,3).volume - 6) < 1e-8"],
            cwd=probe_root, env={"HOME": str(probe_root), "PYTHONPATH": os.pathsep.join(dict.fromkeys([
                str(cad_root / "Multi-Agent-CAD/packages/cadpy/src"),
                str(cad_root / "text-to-cad/packages/cadpy/src"), *map(str, runtime_loader)]))},
            capture_output=True, text=True, check=False, timeout=cad_probe_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _write(runtime / "cad_runtime_preflight_failure.json", {
            "status": "timed_out", "timeout_seconds": cad_probe_timeout_seconds,
            "sandboxed_execution": True,
        })
        raise AstraStageError("astra_sandboxed_cad_runtime_preflight_timeout") from exc
    if cad_probe.returncode:
        _write(runtime / "cad_runtime_preflight_failure.json", {"returncode": cad_probe.returncode,
            "stdout": cad_probe.stdout[-4000:], "stderr": cad_probe.stderr[-4000:]})
        raise AstraStageError("astra_sandboxed_cad_runtime_preflight_failed")
    # Exercise the real writer/renderer under the chosen kernel boundary before
    # training, not just Blender --version outside the sandbox.
    probe_blend = probe_root / "runtime_probe.blend"
    probe_png = probe_root / "runtime_probe.png"
    expression = (
        "import bpy; "
        f"bpy.ops.wm.save_as_mainfile(filepath={str(probe_blend)!r}); "
        "s=bpy.context.scene; s.render.engine='CYCLES'; s.cycles.device='CPU'; "
        "s.cycles.samples=1; s.render.resolution_x=32; s.render.resolution_y=32; "
        "s.render.resolution_percentage=100; "
        f"s.render.filepath={str(probe_png)!r}; bpy.ops.render.render(write_still=True)"
    )
    blender_probe = sandbox([blender["executable"], "--background", "--factory-startup",
        "--threads", "2", "--python-exit-code", "1", "--python-expr", expression],
        cwd=probe_root, env={"HOME": str(probe_root)},
        capture_output=True, text=True, check=False, timeout=90)
    if (blender_probe.returncode or not probe_blend.is_file() or probe_blend.is_symlink()
            or not probe_png.is_file() or probe_png.is_symlink()
            or probe_png.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"):
        _write(runtime / "blender_sandbox_preflight_failure.json", {
            "returncode": blender_probe.returncode, "stdout": blender_probe.stdout[-4000:],
            "stderr": blender_probe.stderr[-4000:], "synthetic_runtime_probe_only": True})
        raise AstraStageError("astra_sandboxed_blender_runtime_preflight_failed")
    _write(runtime / "blender_sandbox_preflight.json", {
        "status": "passed", "render_sha256": _sha256(probe_png),
        "synthetic_runtime_probe_only": True})
    shutil.rmtree(probe_root)  # Trusted cleanup; the real authoring root stays fresh.
    return cad_runtime, cad_root, verified_sources, blender, sandbox


def preflight_astra_execution_runtime(*, package, output_root, environment=None):
    """Discover runtime refusals before paying for ArtiFixer training."""
    output_root.mkdir(parents=True, exist_ok=False)
    authored = output_root / "authoring"
    authored.mkdir()
    for name in _CAD_PACKAGE_FILES:
        source = package / name
        if source.is_symlink() or not source.is_file():
            raise AstraStageError("astra_cad_package_incomplete")
        shutil.copyfile(source, output_root / name)
    prepare_astra_execution_runtime(runtime=output_root, package=package,
                                   authored_root=authored, values=dict(os.environ if environment is None else environment))
    _write(output_root / "runtime_preflight.json", {
        "status": "passed", "model_calls_performed": 0, "provider_allocations_performed": 0})


def _claude_stage_authority(*, values, rights, stage_input, request):
    """Bind a future website scene's signed disclosure to its paid child cap."""
    from .claude_opus_authoring_invoker import ClaudeAuthoringBlocked

    execution = rights.get("execution_authority") or {}
    consent = rights.get("consent") or {}
    terms = rights.get("anthropic_provider_terms_reference")
    authority_digest = values.get("BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST")
    if (stage_input["configuration"].get("authoring_model_provider") != "anthropic"
            or stage_input["configuration"].get("source_observation_kind") != "website_capture_frames"
            or values.get("BLUEPRINT_SCENE_CONFIGURATION_AUTHORING_PROVIDER") != "anthropic"
            or rights.get("schema_version") != "website_native_rights_admission.v1"
            or rights.get("digest") != canonical_digest(rights, digest_field="digest")
            or "anthropic" not in execution.get("allowed_providers", [])
            or rights.get("private_provider_processing_allowed") is not True
            or rights.get("provider_training_allowed") is not False
            or not isinstance(terms, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", terms) is None
            or not isinstance(consent.get("provider_terms_reference"), str)
            or not consent["provider_terms_reference"]
            or not isinstance(authority_digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", authority_digest) is None
            or not request.run_id == stage_input["run_id"]):
        raise ClaudeAuthoringBlocked("claude_stage_signed_provider_authority_missing")
    try:
        maximum_cost = float(values["BLUEPRINT_SCENE_CONFIGURATION_ANTHROPIC_MAX_COST_USD"])
        maximum_calls = int(values["BLUEPRINT_SCENE_CONFIGURATION_ANTHROPIC_MAX_REQUESTS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClaudeAuthoringBlocked("claude_stage_paid_cap_missing") from exc
    if not math.isfinite(maximum_cost) or not 5.0 <= maximum_cost <= 7.0 or not 1 <= maximum_calls <= 32:
        raise ClaudeAuthoringBlocked("claude_stage_paid_cap_invalid")

    def verify(run_id, input_digest):
        if run_id != request.run_id or re.fullmatch(r"sha256:[0-9a-f]{64}", input_digest) is None:
            raise ClaudeAuthoringBlocked("claude_stage_request_identity_changed")
        return {"run_id": run_id, "allowed_providers": ["anthropic"],
                "private_provider_processing_allowed": True,
                "provider_training_allowed": False,
                "authority_digest": canonical_digest({
                    "paid_authority_digest": authority_digest, "rights_digest": rights["digest"],
                    "stage_input_digest": _sha256(_required_path(values, _INPUT_ENV)),
                    "input_digest": input_digest}),
                "provider_terms_digest": canonical_digest({"anthropic_terms_reference": terms,
                    "shared_terms_reference": consent["provider_terms_reference"]})}
    return maximum_cost, maximum_calls, verify


def _execute_claude_stage(*, values, stage_input, rights, request, articulated, plan, part_requests,
                          runtime, authored_root, output, physics_bounds, configuration, source_record,
                          rights_record, cad_runtime, cad_root, verified_sources, blender, sandbox,
                          result_path, package_candidate):
    from functools import partial
    from .claude_opus_authoring_invoker import ClaudeAuthoringConfig, ClaudeOpusAuthoringInvoker
    from .claude_opus_sdk_session import (
        execute_claude_sdk_agent_authoring, inspect_completed_claude_sdk_authoring,
    )
    from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
    from .task_object_agent_cad import execute_cad_program

    maximum_cost, maximum_calls, verify = _claude_stage_authority(
        values=values, rights=rights, stage_input=stage_input, request=request)
    budget_root = runtime / "inference"
    audit = InferenceReservationAudit(run_root=budget_root, run_id=request.run_id)
    invoker = ClaudeOpusAuthoringInvoker(ClaudeAuthoringConfig(
        run_id=request.run_id, maximum_cost_usd=maximum_cost,
        maximum_calls=maximum_calls, allow_live_invocation=True),
        audit=audit, verify_authority=verify)
    cad_executor = partial(execute_cad_program, cad_root=cad_root / "text-to-cad",
        mac_root=cad_root / "Multi-Agent-CAD", sandbox=sandbox, verified_sources=verified_sources)
    instructions = (cad_root / "text-to-cad/skills/cad/SKILL.md").read_text()
    try:
        if articulated:
            parts = {}
            for part_id, part_request in part_requests.items():
                part_root = authored_root / "parts" / part_id
                session_root = budget_root / "parts" / part_id / "asset_session"
                parts[part_id] = execute_claude_sdk_agent_authoring(
                    request_value=part_request.model_dump(mode="json"),
                    output_root=part_root,
                    budget_root=budget_root,
                    session_root=session_root,
                    invoker=invoker, cad_executor=cad_executor, blender_runner=sandbox,
                    blender_executable=blender["executable"], authoring_instructions=instructions)
                if inspect_completed_claude_sdk_authoring(output_root=part_root, budget_root=budget_root,
                        session_root=session_root,
                        request_value=part_request.model_dump(mode="json")) != parts[part_id]:
                    raise AstraStageError("claude_part_retained_validation_changed")
            authored = {"schema_version": ARTICULATED_AUTHORING_RESULT_SCHEMA_VERSION,
                "status": "parts_authored_pending_native_qualification", "model": "claude-opus-5-5",
                "provider": "anthropic", "plan": dict(plan), "parts": parts, "reused_part_ids": [],
                "part_models": {part_id: part["model"] for part_id, part in parts.items()},
                "part_request_digests": {part_id: item.request_digest for part_id, item in part_requests.items()},
                "claim_ceiling": "development_only", "native_import_qualified": False,
                "scene_placement_qualified": False, "physical_equivalence_proven": False}
            authored["result_digest"] = canonical_digest(authored, digest_field="result_digest")
            _write(authored_root / "result.json", authored)
        else:
            authored = execute_claude_sdk_agent_authoring(
                request_value=request.model_dump(mode="json"), output_root=authored_root,
                budget_root=budget_root, invoker=invoker, cad_executor=cad_executor,
                blender_runner=sandbox, blender_executable=blender["executable"],
                authoring_instructions=instructions)
            if inspect_completed_claude_sdk_authoring(output_root=authored_root,
                    budget_root=budget_root, request_value=request.model_dump(mode="json")) != authored:
                raise AstraStageError("claude_retained_validation_changed")
    finally:
        _write(runtime / "inference_audit.json", audit.manifest())
    if articulated:
        return _finish_articulated_component(plan=plan, part_requests=part_requests, authored=authored,
            output=output, physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, stage_input=stage_input, rights_record=rights_record,
            cad_runtime=cad_runtime, blender=blender, authored_root=authored_root,
            result_path=result_path, package_candidate=package_candidate)
    return _finish_component(request=request, authored=authored, package_candidate=package_candidate,
        output=output, physics_bounds=physics_bounds, configuration=configuration,
        source_record=source_record, stage_input=stage_input, rights_record=rights_record,
        cad_runtime=cad_runtime, blender=blender, authored_root=authored_root, result_path=result_path)


def _execute_agents_api_stage(*, values, stage_input, rights, request, articulated, plan, part_requests,
                              runtime, authored_root, output, physics_bounds, configuration, source_record,
                              rights_record, cad_runtime, cad_root, verified_sources, blender, sandbox,
                              result_path, package_candidate, cost_gate_factory):
    """Only a separately signed Sol/managed selection can enter this CPU lane."""
    from functools import partial
    from .task_object_agent_cad import execute_cad_program
    from .task_object_agents_api_stage import run_managed_asset_authoring

    execution = rights.get("execution_authority") or {}
    consent = rights.get("consent") or {}
    policy = configuration.get("agents_api_policy") or {}
    authority_digest = values.get("BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST")
    if (configuration.get("authoring_agent_runtime") != "openai_agents_api"
            or configuration.get("authoring_model_provider") != "openai"
            or configuration.get("authoring_model") != "gpt-6-sol"
            or configuration.get("source_observation_kind") != "website_capture_frames"
            or values.get("BLUEPRINT_SCENE_CONFIGURATION_AUTHORING_RUNTIME") != "openai_agents_api"
            or rights.get("schema_version") != "website_native_rights_admission.v1"
            or rights.get("digest") != canonical_digest(rights, digest_field="digest")
            or "openai" not in execution.get("allowed_providers", [])
            or rights.get("private_provider_processing_allowed") is not True
            or rights.get("provider_training_allowed") is not False
            or not consent.get("provider_terms_reference")
            or not isinstance(authority_digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", authority_digest) is None
            or request.run_id != stage_input["run_id"]):
        raise AstraStageError("agents_api_stage_signed_authority_missing")
    scope = scene_configuration_openai_stage_scope(values, stage="content_agents")
    key_path = Path(scope["api_key_file"]).expanduser()
    guard_file = Path(str(values.get("BLUEPRINT_SCENE_CONFIGURATION_AGENTS_API_PROJECT_GUARD_FILE") or ""))
    try:
        maximum_cost = min(15.0,
            float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD"]),
            float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"]))
        maximum_calls = min(32, int(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise AstraStageError("agents_api_stage_paid_cap_missing") from exc
    if (not math.isfinite(maximum_cost) or maximum_cost <= 0 or maximum_calls < 1
            or not guard_file.is_absolute() or key_path.is_symlink() or not key_path.is_file()
            or key_path.stat().st_mode & 0o077 or not key_path.read_text().strip()):
        raise AstraStageError("agents_api_stage_paid_cap_or_secret_invalid")
    project_id = str(values.get("OPENAI_PROJECT_ID") or "")
    if not project_id or not scope.get("api_key_id"):
        raise AstraStageError("agents_api_stage_project_scope_missing")
    budget_root = runtime / "inference"
    base_invoker, audit = budgeted_invoker(root=budget_root / "review",
        run_id=request.run_id, maximum_cost_usd=maximum_cost)
    reviewer = _StageInvoker(base_invoker, request.run_id, maximum_calls, 0)
    cad_executor = partial(execute_cad_program, cad_root=cad_root / "text-to-cad",
        mac_root=cad_root / "Multi-Agent-CAD", sandbox=sandbox, verified_sources=verified_sources)
    instructions = (cad_root / "text-to-cad/skills/cad/SKILL.md").read_text()
    gate = cost_gate_factory(environment=values, stage="content_agents", run_id=request.run_id,
        request_digest=_sha256(_required_path(values, _INPUT_ENV)), candidate_digest=source_record["digest"],
        output_root=runtime / "official_openai_cost", max_cost_usd=maximum_cost)
    gate.reserve()
    authored, failure, provider_attempted = None, None, False
    try:
        with _stage_sdk_environment(key_path.resolve()):
            if articulated:
                parts = {}
                for part_id, part_request in part_requests.items():
                    provider_attempted = True
                    parts[part_id] = run_managed_asset_authoring(
                        request_value=part_request.model_dump(mode="json"),
                        output_root=authored_root / "parts" / part_id,
                        budget_root=budget_root / "agents_api" / "parts" / part_id,
                        cad_executor=cad_executor, blender_runner=sandbox,
                        blender_executable=blender["executable"], review_invoker=reviewer,
                        policy=policy, authority_digest=authority_digest,
                        source_commit=stage_input["source_commit"], project_id=project_id,
                        credential_id=scope["api_key_id"], guard_file=guard_file,
                        maximum_cost_usd=maximum_cost, key_file=key_path.resolve(),
                        authoring_instructions=instructions)
                authored = {"schema_version": ARTICULATED_AUTHORING_RESULT_SCHEMA_VERSION,
                    "status": "parts_authored_pending_native_qualification", "model": "gpt-6-sol",
                    "provider": "openai", "agent_runtime": "openai_agents_api",
                    "plan": dict(plan), "parts": parts, "reused_part_ids": [],
                    "part_models": {part_id: part["model"] for part_id, part in parts.items()},
                    "part_request_digests": {part_id: item.request_digest for part_id, item in part_requests.items()},
                    "claim_ceiling": "development_only", "native_import_qualified": False,
                    "scene_placement_qualified": False, "physical_equivalence_proven": False}
                authored["result_digest"] = canonical_digest(authored, digest_field="result_digest")
                _write(authored_root / "result.json", authored)
            else:
                provider_attempted = True
                authored = run_managed_asset_authoring(
                    request_value=request.model_dump(mode="json"), output_root=authored_root,
                    budget_root=budget_root / "agents_api", cad_executor=cad_executor,
                    blender_runner=sandbox, blender_executable=blender["executable"],
                    review_invoker=reviewer, policy=policy,
                    authority_digest=authority_digest, source_commit=stage_input["source_commit"],
                    project_id=project_id, credential_id=scope["api_key_id"], guard_file=guard_file,
                    maximum_cost_usd=maximum_cost, key_file=key_path.resolve(),
                    authoring_instructions=instructions)
    except Exception as exc:
        failure = type(exc).__name__
        raise
    finally:
        _write(runtime / "inference_audit.json", audit.manifest())
        gate.complete(provider_call_performed=provider_attempted,
            runtime_result_digest=(authored or {}).get("result_digest"),
            runtime_exception_type=failure)
    if articulated:
        return _finish_articulated_component(plan=plan, part_requests=part_requests, authored=authored,
            output=output, physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, stage_input=stage_input, rights_record=rights_record,
            cad_runtime=cad_runtime, blender=blender, authored_root=authored_root,
            result_path=result_path, package_candidate=package_candidate)
    return _finish_component(request=request, authored=authored, package_candidate=package_candidate,
        output=output, physics_bounds=physics_bounds, configuration=configuration,
        source_record=source_record, stage_input=stage_input, rights_record=rights_record,
        cad_runtime=cad_runtime, blender=blender, authored_root=authored_root, result_path=result_path)


@_locked_component
def execute_astra_component(*, environment=None, runner=subprocess.run,
                            cost_gate_factory=scene_configuration_openai_stage_gate,
                            authoring_executor=execute_asset_authoring, invoker_factory=budgeted_invoker,
                            sandbox_factory=SandboxedAssetRunner, blender_validator=validate_runtime,
                            package_candidate=None, no_cost_replay=False, retained_runtime=None) -> dict[str, Any]:
    values = dict(os.environ if environment is None else environment)
    input_path = _required_path(values, _INPUT_ENV)
    stage_input = _validate_input(_read(input_path, code="astra_stage_input_invalid"),
        adapter_id=_ADAPTER_ID, diagnostic_only=values.get("BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY") == "1")
    dependencies = _validate_dependencies(json.loads(_required_path(values, _DEPENDENCIES_ENV).read_text()))
    toolchain_root = _required_path(values, TOOLCHAIN_ROOT_ENV)
    manifest, _ = _validate_toolchain(root=toolchain_root, expected_source_commit=stage_input["source_commit"])
    if manifest["toolchain_digest"] != stage_input["toolchain_digest"]:
        raise AstraStageError("astra_parent_toolchain_digest_mismatch")
    configuration, envelope = stage_input["configuration"], stage_input["construction_envelope"]
    authoring_provider = configuration.get("authoring_model_provider", "openai")
    authoring_runtime = configuration.get("authoring_agent_runtime")
    if authoring_provider not in {"openai", "anthropic"}:
        raise AstraStageError("astra_authoring_provider_invalid")
    if authoring_runtime not in (None, "openai_agents_api"):
        raise AstraStageError("astra_authoring_runtime_invalid")
    source_record, _ = _dependency_candidate(dependencies)
    references = _reference_frames(stage_input, dependencies)
    rights_record, rights_path = _materialized(envelope, contract_path="scene.rights.admission")
    articulated = configuration.get("schema_version") == ARTICULATED_AUTHORING_SCHEMA_VERSION
    plan = part_requests = None
    if articulated:
        plan, part_requests = build_articulated_authoring_requests(
            stage_input, source_record, references, _read(rights_path, code="astra_rights_invalid"))
        request = next(iter(part_requests.values()))
        physics_bounds = _articulated_physics_bounds(configuration, plan)
    else:
        request = build_authoring_request(stage_input, source_record, references,
                                         _read(rights_path, code="astra_rights_invalid"))
        physics_bounds = _physics_bounds(configuration)
    output = _required_path(values, _OUTPUT_ENV)
    result_path = _required_path(values, _RESULT_ENV)
    package = _required_path(values, _PACKAGE_ENV)
    if not package.is_relative_to(toolchain_root) or not result_path.is_relative_to(output) or result_path.exists():
        raise AstraStageError("astra_component_path_invalid")
    primary = output / "astra_cad_blender_runtime"
    attempts = output / "astra_resume_attempts"
    partial_envelope = envelope.get("partial_astra_successor")
    partial_descriptor = None
    verified_lineage = None
    if articulated and (configuration.get("astra_phase_adoption") is not None or retained_runtime is not None):
        # Assembly parts are adopted per part from the same run root; the rigid
        # phase-adoption descriptors describe one solid and do not apply.
        raise AstraStageError("astra_articulated_phase_adoption_unsupported")
    if partial_envelope is not None:
        from .task_evaluation_partial_astra_successor import restore_partial_astra
        if configuration.get("astra_phase_adoption") is not None or retained_runtime is not None:
            raise AstraStageError("astra_partial_successor_conflicting_adoption")
        descriptor_record = partial_envelope["descriptor"]
        descriptor_path = Path(descriptor_record["materialized_path"])
        actual = file_record(descriptor_path)
        if (actual["sha256"] != descriptor_record.get("sha256", descriptor_record.get("digest"))
                or actual["size_bytes"] != descriptor_record.get("size_bytes")):
            raise AstraStageError("astra_partial_successor_descriptor_file_changed")
        partial_descriptor = _read(descriptor_path, code="astra_partial_successor_descriptor_invalid")
        verified_lineage = partial_envelope["verified_lineage"]
        archive_record = partial_envelope["runtime_archive"]
        archive_path = Path(archive_record["materialized_path"])
        archive_actual = file_record(archive_path)
        if (archive_actual["sha256"] != archive_record.get("sha256", archive_record.get("digest"))
                or archive_actual["size_bytes"] != archive_record.get("size_bytes")):
            raise AstraStageError("astra_partial_successor_archive_file_changed")
        restored_root = Path(partial_descriptor["original_runtime_root"])
        if restored_root != primary and not (restored_root.parent == attempts
                and re.fullmatch(r"attempt-[0-9]{4}", restored_root.name)):
            raise AstraStageError("astra_partial_successor_runtime_root_invalid")
        if articulated and partial_descriptor.get("adoption_kind") != "completed_articulated_agents_api":
            raise AstraStageError("astra_articulated_partial_successor_kind_invalid")
        restore_request = ({"run_id": request.run_id,
                            "part_requests": {part: row.model_dump(mode="json") for part, row in part_requests.items()}}
                           if articulated else request.model_dump(mode="json"))
        restore_partial_astra(value=partial_descriptor, request_value=restore_request,
            original_root=restored_root, verified_lineage=verified_lineage, archive_path=archive_path)
    prior_roots = ([primary] if primary.exists() else []) + sorted(attempts.glob("attempt-????"))
    completed_articulated = (articulated and partial_descriptor is not None
                             and partial_descriptor.get("adoption_kind") == "completed_articulated_agents_api"
                             and len(prior_roots) == 1)
    if authoring_runtime == "openai_agents_api" and (prior_roots or partial_descriptor is not None
            or configuration.get("astra_phase_adoption") is not None
            or no_cost_replay or retained_runtime is not None) and not completed_articulated:
        raise AstraStageError("agents_api_cross_attempt_adoption_not_qualified")
    if authoring_provider == "anthropic" and (prior_roots or partial_descriptor is not None
            or configuration.get("astra_phase_adoption") is not None
            or no_cost_replay or retained_runtime is not None):
        raise AstraStageError("claude_cross_attempt_adoption_not_qualified")
    cross_run = partial_descriptor is not None and len(prior_roots) == 1
    descriptor = configuration.get("astra_phase_adoption")
    if retained_runtime is not None:
        if not no_cost_replay or descriptor is not None:
            raise AstraStageError("astra_operational_replay_scope_invalid")
        descriptor = materialize_automatic_phase_adoption(prior_runtime=Path(retained_runtime))
    if prior_roots and not cross_run:
        if descriptor is not None and Path(descriptor["prior_runtime"]) != prior_roots[-1]:
            raise AstraStageError("astra_resume_cannot_skip_latest_budget_journal")
        descriptor = descriptor or (None if articulated else materialize_automatic_phase_adoption(prior_runtime=prior_roots[-1]))
    if no_cost_replay and (descriptor is None or "authoring_result" not in descriptor.get("completed_artifacts", [])):
        raise AstraStageError("astra_no_cost_replay_requires_completed_authoring")
    source_binding = (_articulated_stage_source_binding(part_requests, stage_input, source_record, rights_path)
                      if articulated else _stage_source_binding(request, stage_input, source_record, rights_path))
    if descriptor is not None and descriptor.get("schema_version") == "task_evaluation_retained_astra_artifacts.v1":
        prior_binding = Path(descriptor["prior_runtime"]) / "stage_source_binding.json"
        if not prior_binding.is_file() or _read(prior_binding, code="astra_retained_source_binding_invalid") != source_binding:
            raise AstraStageError("astra_retained_source_or_rights_binding_changed")
    next_attempt = 1 + max((int(p.name.removeprefix("attempt-")) for p in prior_roots if p.parent == attempts), default=0)
    if len(prior_roots) >= 16 or next_attempt >= 16:
        raise AstraStageError("astra_same_run_resume_limit_reached")
    runtime = primary if not prior_roots else attempts / f"attempt-{next_attempt:04d}"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(mode=0o700)
    _write(runtime / "stage_source_binding.json", source_binding)
    delivery_output = output if not prior_roots else runtime / "delivery"
    for name in _CAD_PACKAGE_FILES:
        source = package / name
        if source.is_symlink() or not source.is_file():
            raise AstraStageError("astra_cad_package_incomplete")
        shutil.copyfile(source, runtime / name)
    if completed_articulated:
        from .task_evaluation_partial_astra_successor import prepare_completed_articulated_successor
        authored_root = runtime / "authoring"
        authored_root.mkdir()
        prepared = prepare_completed_articulated_successor(
            value=partial_descriptor, part_requests=part_requests, plan=plan,
            source_binding=source_binding, verified_lineage=verified_lineage, runtime=runtime,
            successor_envelope_digest=envelope["envelope_digest"])
        if package_candidate is None:
            package_candidate = package_astra_articulated_candidate
        retained = {"status": "retained_completed_articulated_parts",
                    "adoption_digest": partial_descriptor["adoption_digest"],
                    "runtime_execution_repeated": False}
        return _finish_articulated_component(
            plan=plan, part_requests=prepared["source_part_requests"],
            authored=prepared["authored"], output=delivery_output,
            physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, stage_input=stage_input, rights_record=rights_record,
            cad_runtime=retained, blender=retained, authored_root=authored_root,
            result_path=result_path, package_candidate=package_candidate,
            adoption_lineage=prepared["lineage"])
    if cross_run:
        from .task_evaluation_partial_astra_successor import prepare_partial_astra_successor
        adoption = prepare_partial_astra_successor(value=partial_descriptor,
            request_value=request.model_dump(mode="json"), source_binding=source_binding,
            verified_lineage=verified_lineage, package=package, budget_root=runtime / "inference")
        _write(runtime / "partial_successor_descriptor.json", partial_descriptor)
    else:
        adoption = prepare_phase_adoption(value=descriptor, request_value=request.model_dump(mode="json"),
            package=package, budget_root=runtime / "inference")
    if descriptor is not None:
        _write(runtime / "retained_artifact_contract.json", descriptor)
    if package_candidate is None:
        from .task_object_simready_packaging import package_astra_candidate
        package_candidate = package_astra_articulated_candidate if articulated else package_astra_candidate
    authored_root = runtime / "authoring"
    authored_root.mkdir()
    if adoption.get("completed_authoring_result") is not None:
        authored = adoption["completed_authoring_result"]
        _write(authored_root / "request.json", request.model_dump(mode="json"))
        _write(authored_root / "result.json", authored)
        _write(runtime / "no_cost_authoring_adoption.json", {"status": "completed_authoring_adopted",
            "adoption_digest": adoption["adoption_digest"], "retained_inference_cost_usd": adoption["retained_inference_cost_usd"],
            "prior_call_count": adoption["prior_call_count"], "new_provider_calls": 0,
            "cad_execution_repeated": False, "blender_execution_repeated": False})
        retained_runtime = {"status": "retained_completed_artifacts", "adoption_digest": adoption["adoption_digest"],
                            "runtime_execution_repeated": False}
        return _finish_component(request=request, authored=authored, package_candidate=package_candidate,
            output=delivery_output, physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, stage_input=stage_input, rights_record=rights_record,
            cad_runtime=retained_runtime, blender=retained_runtime, authored_root=authored_root, result_path=result_path)
    cad_runtime, cad_root, verified_sources, blender, sandbox = prepare_astra_execution_runtime(
        runtime=runtime, package=package, authored_root=authored_root, values=values,
        runner=runner, sandbox_factory=sandbox_factory, blender_validator=blender_validator)
    if authoring_provider == "anthropic":
        return _execute_claude_stage(values=values, stage_input=stage_input,
            rights=_read(rights_path, code="astra_rights_invalid"), request=request,
            articulated=articulated, plan=plan, part_requests=part_requests,
            runtime=runtime, authored_root=authored_root, output=delivery_output,
            physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, rights_record=rights_record,
            cad_runtime=cad_runtime, cad_root=cad_root, verified_sources=verified_sources,
            blender=blender, sandbox=sandbox, result_path=result_path,
            package_candidate=package_candidate)
    if authoring_runtime == "openai_agents_api":
        return _execute_agents_api_stage(values=values, stage_input=stage_input,
            rights=_read(rights_path, code="astra_rights_invalid"), request=request,
            articulated=articulated, plan=plan, part_requests=part_requests,
            runtime=runtime, authored_root=authored_root, output=delivery_output,
            physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, rights_record=rights_record,
            cad_runtime=cad_runtime, cad_root=cad_root, verified_sources=verified_sources,
            blender=blender, sandbox=sandbox, result_path=result_path,
            package_candidate=package_candidate, cost_gate_factory=cost_gate_factory)
    scope = scene_configuration_openai_stage_scope(values, stage="content_agents")
    key_path = Path(scope["api_key_file"]).expanduser()
    if key_path.is_symlink() or not key_path.is_file() or key_path.stat().st_mode & 0o077 or not key_path.read_text().strip():
        raise AstraStageError("astra_stage_key_file_invalid")
    try:
        stage_cap = float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD"])
        total_cap = float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"])
        maximum_cost = min(15.0, stage_cap, total_cap)
        maximum_calls = min(32, int(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise AstraStageError("astra_parent_budget_invalid") from exc
    if any(not math.isfinite(v) or v <= 0 for v in (stage_cap, total_cap)) or maximum_calls <= 0:
        raise AstraStageError("astra_parent_budget_invalid")
    if adoption.get("retained_inference_cost_usd", 0) > maximum_cost:
        raise AstraStageError("astra_phase_adoption_budget_exhausted")
    base_invoker, audit = invoker_factory(root=runtime / "inference", run_id=request.run_id, maximum_cost_usd=maximum_cost)
    invoker = _StageInvoker(base_invoker, request.run_id, maximum_calls, adoption["prior_call_count"])
    if adoption.get("adoption_digest"):
        _write(runtime / "phase_adoption.json", {key: item for key, item in adoption.items()
                                               if key not in {"authoring_kwargs", "cad_kwargs"}})

    def mac_executor(*, brief, output_root, dimensions_m):
        result = execute_mac_candidate(brief, output_root, cad_root / "Multi-Agent-CAD", cad_root / "text-to-cad",
            invoker, expected_dimensions_mm=tuple(value * 1000 for value in dimensions_m),
            subprocess_runner=sandbox, run_id=request.run_id, object_label=request.object_id,
            max_input_tokens=80000, max_output_tokens=CAD_MAX_OUTPUT_TOKENS,
            max_calls=maximum_calls, repair_budget=2,
            dimension_tolerance_mm=request.maximum_export_error_m * 1000,
            verified_sources=verified_sources, **adoption["cad_kwargs"])
        return {**result, "stl": file_record(Path(result["stl_path"])),
                "step": file_record(Path(result["step_path"])),
                "measured_dimensions_m": [v / 1000 for v in result["readback"]["measured_dimensions_mm"]]}

    gate = cost_gate_factory(environment=values, stage="content_agents", run_id=request.run_id,
        request_digest=_sha256(input_path), candidate_digest=source_record["digest"],
        output_root=runtime / "official_openai_cost", max_cost_usd=maximum_cost)
    gate.reserve()
    authored, failure = None, None
    try:
        with _stage_sdk_environment(key_path.resolve()):
            arguments = dict(request_value=request.model_dump(mode="json"), output_root=authored_root,
                invoker=invoker, blender_runner=sandbox, blender_executable=blender["executable"],
                authoring_instructions=(cad_root / "text-to-cad/skills/cad/SKILL.md").read_text())
            if articulated:
                authored = _author_articulated_parts(plan=plan, part_requests=part_requests, authored_root=authored_root,
                    runtime=runtime, prior_roots=prior_roots, invoker=invoker, sandbox=sandbox, blender=blender,
                    cad_root=cad_root, verified_sources=verified_sources,
                    authoring_instructions=arguments["authoring_instructions"], configuration=configuration,
                    authoring_executor=authoring_executor, mac_executor=mac_executor)
            elif (authoring_executor is execute_asset_authoring
                    and configuration.get("source_observation_kind") == "website_capture_frames"
                    and (not adoption.get("adoption_digest") or "adopted_agent_root" in adoption["authoring_kwargs"])):
                from .task_object_agent_cad import execute_cad_program
                from .task_object_agent_session import execute_agent_authoring
                from functools import partial
                authored = execute_agent_authoring(**arguments, budget_root=runtime / "inference",
                    **adoption["authoring_kwargs"],
                    cad_executor=partial(execute_cad_program, cad_root=cad_root / "text-to-cad",
                        mac_root=cad_root / "Multi-Agent-CAD", sandbox=sandbox, verified_sources=verified_sources))
            else:
                authored = authoring_executor(**arguments, mac_executor=mac_executor, **adoption["authoring_kwargs"])
    except Exception as exc:
        failure = type(exc).__name__
        raise
    finally:
        try:
            _write(runtime / "inference_audit.json", audit.manifest())
        finally:
            gate.complete(provider_call_performed=invoker.calls > 0,
                          runtime_result_digest=(authored or {}).get("result_digest"), runtime_exception_type=failure)
    if articulated:
        return _finish_articulated_component(plan=plan, part_requests=part_requests, authored=authored,
            output=delivery_output, physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, stage_input=stage_input, rights_record=rights_record,
            cad_runtime=cad_runtime, blender=blender, authored_root=authored_root, result_path=result_path,
            package_candidate=package_candidate)
    return _finish_component(request=request, authored=authored, package_candidate=package_candidate,
        output=delivery_output, physics_bounds=physics_bounds, configuration=configuration,
        source_record=source_record, stage_input=stage_input, rights_record=rights_record,
        cad_runtime=cad_runtime, blender=blender, authored_root=authored_root, result_path=result_path)


def _finish_component(*, request, authored, package_candidate, output, physics_bounds, configuration,
                      source_record, stage_input, rights_record, cad_runtime, blender, authored_root, result_path):
    output.mkdir(parents=True, exist_ok=True)
    packaged = package_candidate(request=request, authoring_result=authored,
                                  output_root=output, physics_bounds=physics_bounds)
    asset = Path(packaged["asset"]["path"])
    asset_record = {"path": packaged["asset"]["path"],
                    "digest": packaged["asset"].get("digest") or packaged["asset"].get("sha256"),
                    "size_bytes": packaged["asset"]["size_bytes"]}
    if asset.is_symlink() or not asset.resolve().is_relative_to(output) or _file_record(asset) != asset_record:
        raise AstraStageError("astra_packaged_asset_binding_invalid")
    completion = packaged["physics_completion"]
    completion["metric_envelope_validation"] = _validate_metric_envelope_dimensions(
        envelope=_metric_envelope_spec(configuration), observed_dimensions=completion["collision_dimensions_m"])
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    identity = configuration["replacement_identity"]
    graph = {"schema_version": "task_evaluation_rigid_replacement_graph.v1", "asset_id": identity["id"],
        "asset_version": identity["version"], "articulation_graph": {"joints": []}, "single_rigid_candidate": True,
        "physics_bounds": physics_bounds, "physics_authority_granted": False, "authoring_backend": BACKEND}
    graph_path = output / "replacement_graph_spec.v1.json"
    _write(graph_path, graph)
    managed_receipts = {}
    if configuration.get("authoring_agent_runtime") == "openai_agents_api":
        path = authored_root.parent / "inference" / "agents_api" / "agents_api_stage_receipt.json"
        managed_receipts["object"] = _managed_authoring_receipt(path, authored)
    receipt = {"schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification", "authoring_backend": BACKEND, "model": authored["model"],
        **({"provider": "anthropic"} if authored["model"] == "claude-opus-5-5" else {}),
        **({"provider": "openai", "agent_runtime": "openai_agents_api",
            "managed_agent_execution_receipts": managed_receipts} if managed_receipts else {}),
        "replacement_identity": identity, "source_candidate_digest": source_record["digest"],
        "source_candidate_claim": "source_geometry_not_observed_truth_or_physics_authority",
        "source_commit": stage_input["source_commit"], "toolchain_digest": stage_input["toolchain_digest"],
        "source_rights_admission": dict(rights_record), "cad_skill_runtime": cad_runtime, "blender_runtime": blender,
        "astra_authoring_result": _file_record(authored_root / "result.json"),
        "output_usd": {"sha256": _sha256(asset), "size_bytes": asset.stat().st_size},
        "candidate_physics_completion": completion, "physics_authority_granted": False, "result_digest": ""}
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    receipt_path = output / "replacement_authoring_receipt.v1.json"
    _write(receipt_path, receipt)
    result = {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, "status": "completed", "adapter_id": _ADAPTER_ID,
        "stage_id": stage_input["stage"]["stage_id"], "provider_mutations_performed": 0,
        "nested_paid_execution_requested": False, "authoring_backend": BACKEND, "model": authored["model"],
        **({"provider": "anthropic"} if authored["model"] == "claude-opus-5-5" else {}),
        **({"provider": "openai", "agent_runtime": "openai_agents_api"} if managed_receipts else {}),
        "artifacts": [{"role": role, **_file_record(path)} for role, path in (
            ("replacement_asset", asset), ("replacement_authoring_receipt", receipt_path), ("replacement_graph_spec", graph_path))],
        "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _write(result_path, result)
    return result
