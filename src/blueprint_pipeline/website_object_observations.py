"""Retain website object observations for the existing native authoring stage.

ADP-030/day 28: the subject is absent from the Marble background. Its input
mesh therefore comes from masked source depth, never from a fabricated prim in
that background. These open surface patches remain estimates, not a SimReady
replacement or evidence of unseen surfaces.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .website_task_masks import decode_track_mask

SCHEMA = "website_object_observations.v1"
REFERENCE_ROLE = "website_object_observation_references"


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    name = path.resolve().relative_to(relative_to.resolve()).as_posix() if relative_to else str(path.resolve())
    return {"path": name, "digest": _sha256_file(path), "size_bytes": path.stat().st_size}


def _checked(record: Mapping[str, Any], *, root: Path) -> Path:
    name = Path(record["path"])
    path = root / name
    if (name.is_absolute() or ".." in name.parts or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
            or _record(path, relative_to=root) != dict(record)):
        raise ValueError("website_object_observation_artifact_changed")
    return path


def validate_observation_handoff(path: Path, *, configuration: Mapping[str, Any]) -> tuple[dict, list[Path]]:
    value = json.loads(path.read_text())
    if (value.get("schema_version") != SCHEMA
            or value.get("digest") != canonical_digest(value, digest_field="digest")
            or value.get("preparation_digest") != configuration.get("website_preparation_digest")
            or value.get("source_geometry_digest") != configuration.get("website_source_geometry_digest")
            or value.get("target_id") != configuration.get("source_object_identity")
            or value.get("dimension_authority") != "estimated"
            or value.get("physical_measurement_proven") is not False
            or value.get("complete_object_geometry") is not False
            or value.get("claim_ceiling") != "development_only"):
        raise ValueError("website_object_observation_binding_invalid")
    rows = value.get("frames")
    if not isinstance(rows, list) or not rows or len({row["frame_id"] for row in rows}) != len(rows):
        raise ValueError("website_object_observation_frames_invalid")
    frames = [_checked(row["image"], root=path.parent) for row in rows]
    _checked(value["candidate"], root=path.parent)
    return value, frames


def _reference_unchanged(row: Mapping[str, Any]) -> bool:
    """The transmitted derivative and, when there is one, its retained original."""
    original = row.get("transmission")
    return (_sha256_file(Path(row["path"])) == row["sha256"]
            and (original is None or _sha256_file(Path(original["source_path"])) == original["source_sha256"]))


def _upright(frame: Mapping[str, Any], source: Path, target: Path) -> float:
    """Originals are decoded without autorotation; send them as the camera displayed them."""
    rotation = frame.get("display_rotation_degrees")
    if (isinstance(rotation, bool) or not isinstance(rotation, (int, float))
            or not math.isfinite(rotation) or rotation % 90):
        raise ValueError("website_source_rotation_not_supported")
    if rotation % 360 == 0:
        shutil.copyfile(source, target)
    else:
        # Same convention as the geometry and removal inputs: PIL's
        # counter-clockwise rotate by the display rotation, canvas expanded.
        with Image.open(source) as image:
            image.convert("RGB").rotate(float(rotation), expand=True).save(target, format="PNG")
    return float(rotation)


def materialize_object_observations(*, preparation: Mapping[str, Any], source_geometry: Mapping[str, Any],
                                    task_masks: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    for value in (preparation, source_geometry, task_masks):
        if value.get("digest") != canonical_digest(value, digest_field="digest"):
            raise ValueError("website_object_observation_input_changed")
    bindings = preparation["binding"]
    if (bindings["source_geometry_digest"] != source_geometry["digest"]
            or bindings["task_masks_digest"] != task_masks["digest"]
            or task_masks.get("source_geometry_digest") != source_geometry["digest"]
            or source_geometry.get("metric_measurement_proven") is not False):
        raise ValueError("website_object_observation_source_mismatch")
    config = preparation["authoring_inputs"]["configuration"]
    target_id = config["source_object_identity"]
    targets = [row for row in task_masks["targets"] if row["target_id"] == target_id and row["task_effect"] == "manipulated"]
    if len(targets) != 1:
        raise ValueError("website_object_observation_target_missing")
    frames = {row["frame_id"]: row for row in source_geometry["frames"]}
    if len(frames) != len(source_geometry["frames"]):
        raise ValueError("website_object_observation_duplicate_frame")
    bound_config = {**config, "website_preparation_digest": preparation["digest"],
                    "website_source_geometry_digest": source_geometry["digest"]}
    root = output_root.resolve() / preparation["digest"][7:]
    path = root / "observations.json"
    if path.exists():
        value, _ = validate_observation_handoff(path, configuration=bound_config)
        # Check source bytes too, rather than adopting stale copied inputs.
        references = {row["frame_id"]: row for row in config.get("reference_frames") or []}
        for row in value["frames"]:
            if row.get("image_basis") == "upright_coverage_frame":
                reference = references[row["frame_id"]]
                if reference["sha256"] != row["source_image_digest"] or not _reference_unchanged(reference):
                    raise ValueError("website_object_observation_source_changed")
                continue
            frame = frames[row["frame_id"]]
            if (_sha256_file(Path(frame["source_image_path"] if row.get("image_basis") == "original_capture" else frame["image_path"])) != row["source_image_digest"]
                    or _sha256_file(Path(frame["geometry_path"])) != row["source_geometry_digest"]):
                raise ValueError("website_object_observation_source_changed")
        return {"manifest": _record(path), "candidate": _record(_checked(value["candidate"], root=root)),
                "configuration": bound_config}
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / "source_object_candidate.usda"
    stage = Usd.Stage.CreateNew(str(candidate))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Root").GetPrim())
    UsdGeom.Xform.Define(stage, "/Root/SourceObjectCandidate")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    source_to_runtime = np.asarray(preparation["registration"]["source_to_runtime"], dtype=float)
    runtime_to_sim = np.asarray(preparation["coordinate_frame"]["runtime_to_simulator"], dtype=float)
    snap = np.eye(4)
    up = {"Y": 1, "-Y": 1, "Z": 2}[preparation["coordinate_frame"]["declared_up_axis"]]
    snap[up, 3] = preparation["compose_back"]["pose_world"]["support_snap_runtime_units"]
    transform = runtime_to_sim @ snap @ source_to_runtime
    # An assembly with whole-object coverage is authored from the upright views
    # chosen to show every part and state, not from the depth-sampling frames.
    references = list(config.get("reference_frames") or [])
    retained, surfaces = [], 0
    for index, observation in enumerate(targets[0]["track"]["observations"]):
        frame = frames[observation["source_frame_id"]]
        if (_sha256_file(Path(frame["geometry_path"])) != frame["geometry_digest"]
                or _sha256_file(Path(frame["image_path"])) != frame["image_digest"]):
            raise ValueError("website_object_observation_source_changed")
        mask = decode_track_mask(observation)
        with np.load(frame["geometry_path"], allow_pickle=False) as geometry:
            depth, valid = geometry["depth_m"], geometry["valid_mask"]
        if mask.shape != depth.shape or depth.shape != valid.shape:
            raise ValueError("website_object_observation_mask_shape_invalid")
        mask &= valid.astype(bool) & np.isfinite(depth) & (depth > 0)
        # Adjacent masked pixels only: no convex hull or invented back faces.
        h, w = mask.shape
        yy, xx = np.indices((h, w))
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3)
        camera = (pixels @ np.linalg.inv(np.asarray(frame["intrinsics"])).T) * depth.reshape(-1, 1)
        pose = transform @ np.asarray(frame["world_from_camera"])
        points = camera @ pose[:3, :3].T + pose[:3, 3]
        a = np.arange(h * w).reshape(h, w)[:-1, :-1].ravel()
        triangles = np.concatenate([np.stack([a, a + 1, a + w], axis=1),
                                    np.stack([a + 1, a + w + 1, a + w], axis=1)])
        triangles = triangles[mask.ravel()[triangles].all(axis=1)]
        if not len(triangles):
            continue
        used, indices = np.unique(triangles, return_inverse=True)
        if not np.isfinite(points[used]).all():
            raise ValueError("website_object_observation_nonfinite_geometry")
        mesh = UsdGeom.Mesh.Define(stage, f"/Root/SourceObjectCandidate/ObservedFrame{index:04d}")
        mesh.CreatePointsAttr(points[used].astype(np.float32).tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
        mesh.CreateFaceVertexIndicesAttr(indices.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        surfaces += 1
        if references:
            continue
        image = root / f"source-{index:04d}.png"
        original = "source_image_path" in frame
        image_path = Path(frame["source_image_path"] if original else frame["image_path"])
        image_digest = frame["source_image_digest"] if original else frame["image_digest"]
        if _sha256_file(image_path) != image_digest:
            raise ValueError("website_object_observation_source_changed")
        rotation = _upright(frame, image_path, image) if original else 0.0
        if not original:
            shutil.copyfile(image_path, image)
        retained.append({"frame_id": frame["frame_id"], "timestamp_seconds": frame["timestamp_seconds"],
                         "image": _record(image, relative_to=root), "source_image_digest": image_digest,
                         "image_basis": "original_capture" if original else "geometry_input",
                         "display_rotation_applied_degrees": rotation,
                         "source_geometry_digest": frame["geometry_digest"], "triangle_count": len(triangles),
                         "mask_digest": canonical_digest(observation)})
    if not surfaces:
        raise ValueError("website_object_observation_surface_missing")
    for index, row in enumerate(references):
        image = root / f"reference-{index:02d}.png"
        if not _reference_unchanged(row):
            raise ValueError("website_object_observation_source_changed")
        shutil.copyfile(row["path"], image)
        retained.append({"frame_id": row["frame_id"], "timestamp_seconds": row["timestamp_seconds"],
                         "image": _record(image, relative_to=root), "source_image_digest": row["sha256"],
                         "image_basis": "upright_coverage_frame",
                         # The transmitted derivative's retained full-resolution original.
                         **({"retained_original_sha256": row["transmission"]["source_sha256"]}
                            if "transmission" in row else {}),
                         **{key: row[key] for key in ("visible_parts", "part_state", "view", "reason")}})
    stage.GetRootLayer().Save()
    value = {"schema_version": SCHEMA, "preparation_digest": preparation["digest"],
             "source_geometry_digest": source_geometry["digest"], "task_masks_digest": task_masks["digest"],
             "target_id": target_id, "frames": retained, "candidate": _record(candidate, relative_to=root),
             "source_to_simulator": transform.tolist(), "dimension_authority": "estimated",
             "complete_object_geometry": False, "physical_measurement_proven": False,
             "claim_ceiling": "development_only", "background_modified": False}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(path, value)
    return {"manifest": _record(path), "candidate": _record(candidate), "configuration": bound_config}
