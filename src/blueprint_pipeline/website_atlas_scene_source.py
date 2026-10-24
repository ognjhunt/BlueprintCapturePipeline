"""A website capture's Atlas job as the scene source for scene preparation.

Beta journey step: concurrent scene preparation. The capture's Atlas job
(``website_atlas_preview``, under ``pipeline/atlas``) is the only paid
reconstruction lane. Scene preparation buys nothing; it reads what that job
retained:

- the object-removed background splat and its supported mesh, as the scene,
  copied level on the floor (or counters) they observed;
- the selected original views' own pose, as source geometry for lifting task
  masks, scaled by the job's reviewed reference estimate;
- the cameras both poses share, which align the source to the scene exactly,
  or, when views cleaned on an earlier crop moved those cameras, the depth both
  poses returned for the views' unedited pixels.

Every value stays an estimate: the scale comes from named reference
assumptions, never a measurement, and nothing here qualifies a scene.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

PROVIDER = "world_labs_atlas"
SCALE_AUTHORITY = "atlas_reference_estimate"
#: Shared cameras must agree on one similarity: orientations to this many
#: degrees, centres to this fraction of their spread.
ALIGNMENT_MAX_ROTATION_DEGREES = 5.0
ALIGNMENT_MAX_CENTRE_RESIDUAL = 0.05


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError("website_atlas_scene_artifact_missing")
    return resolved


def _artifact(root: Path, row: Mapping[str, Any]) -> tuple[Path, str]:
    path = _inside(root / row["path"], root)
    if _sha256_file(path) != row["sha256"]:
        raise ValueError("website_atlas_scene_artifact_changed")
    return path, row["sha256"]


def _cameras(pose: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """OpenCV camera-to-world per frame id, in the pose's own Atlas world units."""
    from .atlas_source_geometry import opencv_world_from_camera
    rows = pose["operation_receipt"]["input_usage"]["frames"]
    return {row["frame_id"]: opencv_world_from_camera(frame["camera"]["extrinsics"])
            for row, frame in zip(rows, pose["operation"]["response"]["frames"], strict=True)}


def world_alignment(source: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]) -> dict[str, Any] | None:
    """The similarity taking one Atlas world to another, from cameras both poses share.

    Rotation comes from the cameras' orientations, so a near-straight walk still
    fixes it; scale and translation then come from their centres. Returns None
    when the shared cameras are too few or disagree.
    """
    shared = [key for key in source if key in target]
    if len(shared) < 2:
        return None
    pairs = [(source[key], target[key]) for key in shared]
    u, _, vt = np.linalg.svd(sum(b[:3, :3] @ a[:3, :3].T for a, b in pairs))
    rotation = u @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))]) @ vt
    a_centres = np.array([a[:3, 3] for a, _ in pairs]) @ rotation.T
    b_centres = np.array([b[:3, 3] for _, b in pairs])
    a_mean, b_mean = a_centres.mean(0), b_centres.mean(0)
    spread = float(((a_centres - a_mean) ** 2).sum())
    if spread <= 1e-12:
        return None
    scale = float(((a_centres - a_mean) * (b_centres - b_mean)).sum() / spread)
    if not math.isfinite(scale) or scale <= 0:
        return None
    translation = b_mean - scale * a_mean
    angles = [math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(
        (rotation @ a[:3, :3]).T @ b[:3, :3])) - 1) / 2)))) for a, b in pairs]
    residual = float(np.sqrt(((scale * a_centres + translation - b_centres) ** 2).sum(1).mean()))
    extent = float(np.sqrt(((b_centres - b_mean) ** 2).sum(1).mean()))
    if max(angles) > ALIGNMENT_MAX_ROTATION_DEGREES or residual > ALIGNMENT_MAX_CENTRE_RESIDUAL * extent:
        return None
    return {"scale": scale, "rotation": rotation, "translation": translation, "shared_frame_count": len(shared),
            "max_rotation_residual_degrees": max(angles), "centre_residual_fraction": residual / extent}


def _shared_view_alignment(*, job_dir: Path, source_root: Path, source: Mapping[str, Any],
                           background_root: Path, background: Mapping[str, Any],
                           bundle_root: Path | None, bundle: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The background world from the depth both poses returned for their shared views.

    A retained background may keep its pose's outputs only in the download
    bundle; each is found by its receipt digest. None when either pose's outputs
    are unavailable or the views fix no similarity.
    """
    from .atlas_depth_splat import posed_rgbd_frames
    from .atlas_shared_view_alignment import shared_view_alignment

    kept = {row["sha256"]: bundle_root / row["path"] for row in (bundle or {}).get("assets") or []
            if bundle_root is not None}

    def locator(root: Path):
        def locate(output: Mapping[str, Any]) -> Path | None:
            for path in (root / output["path"], kept.get(output["sha256"])):
                if (path is not None and not path.is_symlink() and path.is_file()
                        and path.resolve().is_relative_to(job_dir.resolve())):
                    return path
            return None
        return locate
    try:
        frames = (posed_rgbd_frames(source, locator(source_root)),
                  posed_rgbd_frames(background, locator(background_root)))
    except (ValueError, KeyError):
        return None
    return shared_view_alignment(*frames)


def _survey_rotation(job_dir: Path, inputs: Mapping[str, Any]) -> float:
    selection = _read(job_dir / "selection_binding.json")
    survey = _read(_inside(job_dir / selection.get("survey_relative_path", "missing"), job_dir))
    if (survey.get("digest") != canonical_digest(survey, digest_field="digest")
            or survey.get("digest") != inputs["binding"].get("survey_digest")):
        raise ValueError("website_atlas_scene_survey_changed")
    return float(inputs["binding"].get("source_image_rotation_degrees", survey["display_rotation_degrees"]))


def _retained_pose_alignment(*, job_dir: Path, source_root: Path, source: Mapping[str, Any],
                             background_root: Path, background: Mapping[str, Any],
                             bundle_root: Path | None, bundle: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Bridge one expanded source pose through a retained pose of the same video.

    Adding object-state views can change Atlas's depth estimates enough that a
    direct fit to cleaned views fails. An earlier source pose can still connect
    both. Each leg must independently pass the existing shared-pixel depth fit;
    no guessed transform, camera-only baseline or relaxed tolerance is used.
    """
    video = (source.get("binding") or {}).get("source_video_digest")
    if not video:
        return None
    for path in sorted((job_dir / "pose").rglob("atlas_posed_rgbd_result.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(job_dir.resolve()):
            continue
        prior = _read(path)
        if (prior.get("digest") in {source.get("digest"), background.get("digest")}
                or prior.get("digest") != canonical_digest(prior, digest_field="digest")
                or (prior.get("binding") or {}).get("source_video_digest") != video):
            continue
        first = _shared_view_alignment(job_dir=job_dir, source_root=source_root, source=source,
            background_root=path.parent, background=prior, bundle_root=None, bundle=None)
        if first is None:
            continue
        second = _shared_view_alignment(job_dir=job_dir, source_root=path.parent, source=prior,
            background_root=background_root, background=background, bundle_root=bundle_root, bundle=bundle)
        if second is None:
            continue
        return {"scale": second["scale"] * first["scale"],
                "rotation": second["rotation"] @ first["rotation"],
                "translation": second["scale"] * second["rotation"] @ first["translation"] + second["translation"],
                "intermediate_pose_digest": prior["digest"],
                "alignment_legs": [{key: value for key, value in leg.items()
                                    if key not in ("rotation", "translation")} for leg in (first, second)]}
    return None


def atlas_scene_source(*, capture_root: Path, output_root: Path,
                       retained_result_path: Path | None = None,
                       retained_result_digest: str | None = None) -> dict[str, Any]:
    """``ready`` with a base scene and source geometry, ``pending``, or ``needs_input`` with a reason."""
    from .atlas_source_geometry import atlas_posed_view_geometry
    from .atlas_video_inputs import GEOMETRY_INPUTS_FILE

    job_dir = capture_root / "pipeline" / "atlas"
    job = _read(job_dir / "preview_job.json")
    if not job:
        return {"status": "needs_input", "reason": "website_atlas_job_required"}
    if job.get("state") in {None, "processing"}:
        return {"status": "pending", "reason": "website_atlas_job_processing", "stage": job.get("stage")}
    if job.get("state") != "ready":
        return {"status": "needs_input", "reason": f"website_atlas_job_{job.get('state')}",
                "job_reason": job.get("reason")}
    result = _read(job_dir / "result.json")
    if result.get("digest") != canonical_digest(result, digest_field="digest") or result["digest"] != job.get("result_digest"):
        raise ValueError("website_atlas_scene_result_changed")
    if retained_result_path is not None or retained_result_digest is not None:
        if retained_result_path is None or not retained_result_digest:
            raise ValueError("website_atlas_scene_retained_selection_incomplete")
        retained = _read(_inside(retained_result_path, job_dir))
        if (retained.get("schema_version") != "website_atlas_pipeline_result.v1"
                or retained.get("digest") != retained_result_digest
                or retained.get("digest") != canonical_digest(retained, digest_field="digest")
                or any(not result.get(key) or retained.get(key) != result[key]
                       for key in ("source_video_digest", "task_context_digest", "source_pose"))):
            raise ValueError("website_atlas_scene_retained_result_changed")
        # Read a completed background revision without changing the capture's
        # controller state, original observations, scale evidence or publication.
        result = retained
    background = (result.get("automation") or {}).get("background") or {}
    if (result.get("automation") or {}).get("status") != "completed" or background.get("status") != "completed":
        return {"status": "needs_input", "reason": "website_atlas_background_unavailable",
                "job_reason": background.get("reason")}
    if not background.get("mesh_available"):
        return {"status": "needs_input", "reason": "website_atlas_background_mesh_unavailable"}
    source_pose = result.get("source_pose") or {}
    if source_pose.get("status") != "completed":
        return {"status": "needs_input", "reason": "website_atlas_source_pose_unavailable",
                "job_reason": source_pose.get("reason")}

    pose_root = _inside(job_dir / source_pose.get("pose_relative_path", "pose") / GEOMETRY_INPUTS_FILE, job_dir).parent
    measurements_path = _inside(pose_root / source_pose.get('measurements_relative_path', 'scene_measurements.json'), job_dir)
    measurements = _read(measurements_path)
    mpu = measurements.get("meters_per_unit_estimate")
    if (measurements.get("digest") != canonical_digest(measurements, digest_field="digest")
            or measurements["digest"] != source_pose.get("scene_measurements_digest")):
        raise ValueError("website_atlas_scene_measurements_changed")
    if (measurements.get("status") != "estimated" or isinstance(mpu, bool) or not isinstance(mpu, (int, float))
            or not math.isfinite(mpu) or mpu <= 0):
        # Known-size references that disagree say the scale is unknown, not small.
        disagree = (measurements.get("reference_agreement") or {}).get("basis") == "no_majority"
        return {"status": "needs_input", "reason": "website_atlas_scale_references_disagree" if disagree
                else "website_atlas_scale_estimate_unavailable"}
    scale = {"meters_per_unit": float(mpu), "basis": "reference_estimate", "evidence_digest": measurements["digest"],
             "evidence_reference": str(measurements_path.relative_to(capture_root.resolve()))}
    inputs = _read(_inside(pose_root / GEOMETRY_INPUTS_FILE, job_dir))
    geometry = atlas_posed_view_geometry(pose_root=pose_root, rotation=_survey_rotation(job_dir, inputs),
                                         scale=scale, output_root=output_root / "source_geometry")

    automation_root = job_dir / "automation"
    bundle = bundle_root = None
    download = background.get("download") or {}
    if download.get("directory"):
        bundle_path = _inside(Path(download["directory"]) / "manifest.json", job_dir)
        bundle = _read(bundle_path)
        if (bundle.get("schema_version") != "atlas_download_bundle.v1"
                or bundle.get("digest") != canonical_digest(bundle, digest_field="digest")
                or bundle["digest"] != download.get("manifest_digest")
                or bundle.get("splat_manifest_digest") != background.get("splat_manifest_digest")
                or not bundle.get("splat_available") or not bundle.get("mesh_available")):
            raise ValueError("website_atlas_scene_download_changed")
        # The final portable bundle is stable across reconstruction revisions;
        # generated completion uses different working directories from baseline.
        bundle_root = bundle_path.parent
        splat_root, registered_root = bundle_root / "splat", bundle_root / "mesh/registered"
    else:
        splat_root, registered_root = automation_root / "background_splat", automation_root / "background_mesh/registered"
    splat_manifest = _read(_inside(splat_root / "atlas_depth_splat_manifest.json", job_dir))
    registration = _read(_inside(registered_root / "atlas_mesh_registration.json", job_dir))
    if (splat_manifest.get("digest") != canonical_digest(splat_manifest, digest_field="digest")
            or splat_manifest["digest"] != background.get("splat_manifest_digest")
            or registration.get("digest") != canonical_digest(registration, digest_field="digest")
            or registration.get("splat_manifest_digest") != splat_manifest["digest"]
            or (bundle is not None and registration.get("digest") != bundle.get("mesh_registration_digest"))):
        raise ValueError("website_atlas_scene_background_changed")
    splat_path, splat_digest = _artifact(splat_root, next(
        row for row in splat_manifest["outputs"] if row["role"] == "splat_ply"))
    mesh_path, mesh_digest = _artifact(registered_root, next(
        row for row in registration["outputs"] if row["role"] == "supported_mesh_glb"))

    # The background was fused on whichever pose posed its views: the original
    # views' own, or a fresh pose of the reviewed edited views.
    source = json.loads((pose_root / "atlas_posed_rgbd_result.json").read_text())
    background_root = pose_root
    if splat_manifest["pose_result_digest"] == source["digest"]:
        background_pose = source
    else:
        candidates = [(path.parent, _read(path))
                      for path in sorted(automation_root.glob("fallback_pose*/atlas_posed_rgbd_result.json"))]
        background_root, background_pose = next(((root, row) for root, row in candidates
                                                 if row.get("digest") == splat_manifest["pose_result_digest"]
                                                 and row["digest"] == canonical_digest(row, digest_field="digest")),
                                                (None, None))
        if background_pose is None:
            raise ValueError("website_atlas_scene_background_pose_missing")
    alignment = world_alignment(_cameras(source), _cameras(background_pose))
    basis = "same_atlas_pose" if background_pose is source else "shared_atlas_cameras"
    if alignment is None and background_pose is not source:
        alignment = _shared_view_alignment(job_dir=job_dir, source_root=pose_root, source=source,
                                           background_root=background_root, background=background_pose,
                                           bundle_root=bundle_root, bundle=bundle)
        basis = "shared_view_depth"
    if alignment is None and background_pose is not source:
        alignment = _retained_pose_alignment(job_dir=job_dir, source_root=pose_root, source=source,
            background_root=background_root, background=background_pose, bundle_root=bundle_root, bundle=bundle)
        basis = "retained_source_pose_shared_depth"
    anchor, runtime_mpu = None, float(mpu)
    if alignment is not None:
        # Source metres -> background units: (R x / mpu) * s + t. In the
        # registration's terms, metres per background unit is mpu / s and the
        # rigid part carries the translation in those metres.
        runtime_mpu = float(mpu) / alignment["scale"]
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = alignment["rotation"], alignment["translation"] * runtime_mpu
        anchor = {"kind": "source_world_alignment", "source_world_to_provider_m": transform.tolist(),
                  "basis": basis, **{key: value for key, value in alignment.items()
                                     if key not in ("scale", "rotation", "translation")}}
    # Atlas worlds are right-up-back, but their +Y is only roughly up. Compile
    # takes +Y for gravity, so the scene is a copy levelled on what it observed.
    from .atlas_world_levelling import level_atlas_world
    level = level_atlas_world(splat_path=splat_path, splat_digest=splat_digest, mesh_path=mesh_path,
                              mesh_digest=mesh_digest, meters_per_unit=runtime_mpu,
                              camera_centres=np.array([row[:3, 3] for row in _cameras(background_pose).values()]),
                              output_root=output_root / "level")
    if level["status"] != "levelled":
        return {"status": "needs_input", "reason": level["reason"]}
    levelled = {row["role"]: (Path(row["path"]), row["sha256"]) for row in level["outputs"]}
    lift = np.asarray(level["provider_to_levelled"], dtype=np.float64)
    if anchor is not None:
        lift_m = lift.copy()
        lift_m[:3, 3] *= runtime_mpu
        anchor["source_world_to_provider_m"] = (lift_m @ np.asarray(anchor["source_world_to_provider_m"])).tolist()
    provider = {"provider_splat_path": str(splat_path), "provider_splat_digest": splat_digest,
                "provider_collision_mesh_path": str(mesh_path), "provider_collision_mesh_digest": mesh_digest}
    (splat_path, splat_digest), (mesh_path, mesh_digest) = levelled["splat_ply"], levelled["supported_mesh_glb"]
    base = {
        "splat_path": str(splat_path), "splat_digest": splat_digest,
        "splat_binding_id": "website-splat-" + splat_digest[7:39],
        "collision_mesh_path": str(mesh_path), "collision_mesh_digest": mesh_digest,
        "collision_binding_id": "website-collider-" + mesh_digest[7:39],
        "up_axis": "Y", "meters_per_unit": runtime_mpu, "provider": PROVIDER, "scale_authority": SCALE_AUTHORITY,
        "ground_plane_offset_m": level["ground_plane_offset_m"], "anchor": anchor,
        "levelling": {"digest": level["digest"], "up_basis": level["up_basis"], "tilt_degrees": level["tilt_degrees"],
                      "provider_to_levelled": level["provider_to_levelled"], **provider},
        "operation_id": background_pose["operation"]["id"], "atlas_result_digest": result["digest"],
    }
    return {"status": "ready", "base_scene": base, "source_geometry": geometry,
            "source_pose_digest": source["digest"], "background_pose_digest": background_pose["digest"],
            "alignment": "anchored" if anchor else "registration_search"}
