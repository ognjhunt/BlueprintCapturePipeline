"""Prepare reconstructed background geometry for native composition, ADP-030/day 28.

The manipulated subject is authored separately from retained source footage.
Do not run source-object excision on a background reconstructed after removal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .task_evaluation_completed_scene_geometry import normalize_completed_mesh


def prepare_website_runtime_inputs(*, preparation: Mapping[str, Any], base_scene: Mapping[str, Any],
                                   source_geometry: Mapping[str, Any], task_masks: Mapping[str, Any],
                                   output_root: Path) -> dict[str, Any]:
    if preparation.get("digest") != canonical_digest(preparation, digest_field="digest"):
        raise ValueError("website_native_preparation_changed")
    if (preparation.get("claim_ceiling") != "development_only"
            or preparation.get("subject", {}).get("geometry_origin") != "removed_before_reconstruction"):
        raise ValueError("website_native_preparation_kind_invalid")
    if "website_scene_processing_rights_required" in preparation.get("blockers", []):
        raise ValueError("website_scene_processing_rights_required")
    bindings = preparation["binding"]
    for key in ("collision_mesh", "splat"):
        path = Path(base_scene[key + "_path"])
        if _sha256_file(path) != base_scene[key + "_digest"] or base_scene[key + "_digest"] != bindings[key + "_digest"]:
            raise ValueError("website_native_background_changed")
    frames = preparation["authoring_inputs"]["source_frames"]
    if not frames:
        raise ValueError("website_native_observed_object_references_required")
    for frame in frames:
        if frame.get("role") != "observed_source" or _sha256_file(Path(frame["path"])) != frame["sha256"]:
            raise ValueError("website_native_object_reference_changed")
    coordinates = preparation["coordinate_frame"]
    if coordinates.get("physical_scale_measured") is not False:
        raise ValueError("website_native_scale_authority_invalid")
    frame = {"meters_per_unit": coordinates["declared_meters_per_unit"],
             "up_axis": coordinates["declared_up_axis"]}
    if frame["up_axis"] != base_scene["up_axis"] or frame["up_axis"] not in {"Y", "Z"}:
        raise ValueError("website_native_coordinate_frame_mismatch")
    expected = np.eye(4)
    expected[:3, :3] = frame["meters_per_unit"] * (
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) if frame["up_axis"] == "Y" else np.eye(3))
    retained = np.asarray(coordinates.get("runtime_to_simulator"), dtype=float)
    if retained.shape != (4, 4) or not np.allclose(retained, expected, rtol=0, atol=1e-10):
        raise ValueError("website_native_coordinate_frame_mismatch")
    # Include the frame in the cache key: the same provider mesh with a revised
    # MapAnything estimate must never reuse a prior coordinate conversion.
    cache_key = canonical_digest({"mesh": bindings["collision_mesh_digest"], "frame": frame})[7:]
    collision_root = output_root.resolve() / "collision" / cache_key
    source = Path(base_scene["collision_mesh_path"])
    normalized = normalize_completed_mesh(source=source, original_filename=source.name,
                                         coordinate_frame=frame, output_root=collision_root)
    collision = collision_root / normalized["output"]["relative_path"]
    from .website_object_observations import materialize_object_observations, REFERENCE_ROLE
    observations = materialize_object_observations(preparation=preparation, source_geometry=source_geometry,
        task_masks=task_masks, output_root=output_root / "object_observations")
    value = {
        "schema_version": "website_scene_runtime_inputs.v1", "status": "background_collision_prepared",
        "preparation_digest": preparation["digest"], "claim_ceiling": "development_only",
        "collision": {"path": str(collision), "digest": normalized["output"]["sha256"],
                      "size_bytes": normalized["output"]["size_bytes"],
                      "normalization_path": str(collision_root / "mesh_normalization.v1.json"),
                      "normalization_digest": normalized["normalization_digest"],
                      "object_mapping": normalized["object_mapping"]},
        "appearance": {"path": base_scene["splat_path"], "digest": bindings["splat_digest"],
                       "status": "awaiting_splat_frame_binding", "unchanged": True},
        "object_authoring": {**preparation["authoring_inputs"], "configuration": observations["configuration"],
                             "source_candidate": observations["candidate"],
                             "observation_manifest": observations["manifest"]},
        "authoring_dependency_artifacts": [
            {"role": "source_object_candidate_mesh", **observations["candidate"]},
            {"role": REFERENCE_ROLE, **observations["manifest"]}],
        "subject": dict(preparation["subject"]), "destination": preparation["destination"],
        "coordinate_frame": {"up_axis": "Z", "unit": "estimated_meters", "physical_scale_measured": False},
        "appearance_removal_required": False, "collision_excision_required": False,
        "reconstruction_performed": False, "provider_mutation_performed": False,
        "simulator_ready": False, "physics_qualified": False,
    }
    from .website_native_appearance import prepare_native_appearance
    try:
        appearance = prepare_native_appearance(preparation=preparation, base_scene=base_scene,
                                               output_root=output_root / "appearance")
        value["appearance"] = {**appearance["artifact"], "status": appearance["status"],
                               "receipt": appearance, "renderer_qualified": False}
    except (ValueError, OSError, ImportError) as exc:
        value["appearance"]["blockers"] = [str(exc)]
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(output_root / "runtime_inputs.json", value)
    return value
