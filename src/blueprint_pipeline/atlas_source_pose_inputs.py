"""Keep the original source field of view when buying task-object geometry.

These original observations are separate from the cleaned environment context.
Padding changes the upload grid, never the source pixels or reference coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .atlas_context_fov import square_pad
from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def task_source_inputs(pose_root: Path, original: dict) -> tuple[Path, dict]:
    """Keep admitted full-video task views out of the static background selection.

    An open/closed assembly needs states outside the background's still interval.
    Reuse the capture's training-only source-frame manifest, at native resolution;
    these views are for observed object depth only and never enter scene generation.
    """
    from .atlas_privacy_screen import blur_faces, screen_identity
    from .website_scene_geometry import _bound_frames

    clean_root = pose_root.parent.parent / "clean_plate"
    manifest_path = clean_root / "clean_plate_stage_manifest.json"
    if not manifest_path.is_file():
        return pose_root, original
    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get("source_frames")
    if sources is None:
        return pose_root, original
    if (sources.get("schema_version") != "website_source_frames.v1"
            or sources.get("digest") != canonical_digest(sources, digest_field="digest")
            or sources.get("binding", {}).get("source_video_digest") != original["binding"]["source_video_digest"]):
        raise ValueError("atlas_task_source_frames_binding_invalid")
    source_root = clean_root / "source_geometry/worker_inputs"
    candidate = json.loads((source_root / "geometry_inputs.json").read_text())
    if (candidate.get("digest") != sources.get("geometry_input_digest")
            or candidate.get("heldout_pixels_included") is not False):
        raise ValueError("atlas_task_source_candidate_split_invalid")
    source_frames = _bound_frames(candidate, source_root, geometry=False)
    existing = {row["frame_id"] for row in original["frames"]}
    additional = [row for row in source_frames if row["frame_id"] not in existing]
    if not additional:
        return pose_root, original
    binding = {**original["binding"], "background_input_digest": original["digest"],
               "task_source_frames_digest": sources["digest"], "source_image_rotation_degrees": 0,
               "object_geometry_only": True}
    binding["maximum_frames"] = len(original["frames"]) + len(additional)
    root = pose_root / ("task-coverage-" + canonical_digest(binding)[7:23])
    path = root / "atlas_geometry_inputs.json"
    if path.is_file():
        retained = json.loads(path.read_text())
        if retained.get("binding") != binding:
            raise ValueError("atlas_task_source_inputs_changed")
        _bound_frames(retained, root, geometry=False)
        return root, retained
    rows, faces = [], {}
    # Existing Atlas images are already upright and screened. Normalize only
    # their retained observation copy; the provider image bytes stay identical.
    for row in _bound_frames(original, pose_root, geometry=False):
        target = root / "views" / (row["frame_id"] + ".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["image_path"], target)
        rows.append({**row, "source_image_path": str(target.resolve()),
                     "source_image_digest": row["image_digest"], "image_path": str(target.resolve()),
                     "original_source_image_digest": row["source_image_digest"]})
    for row in additional:
        with Image.open(row["source_image_path"]) as image:
            upright = image.convert("RGB").rotate(row["display_rotation_degrees"], expand=True)
            screened, boxes = blur_faces(upright)
        target = root / "views" / (row["frame_id"] + ".png")
        screened.save(target)
        digest = _sha256_file(target)
        rows.append({"frame_id": row["frame_id"], "timestamp_seconds": row["timestamp_seconds"],
                     "source_image_path": str(target.resolve()), "source_image_digest": digest,
                     "image_path": str(target.resolve()), "image_digest": digest,
                     "width": screened.width, "height": screened.height,
                     "original_source_image_digest": row["source_image_digest"]})
        faces[row["frame_id"]] = len(boxes)
    result = {**original, "binding": binding, "frames": rows,
              "task_view_privacy_screen": {**screen_identity(), "blurred_faces_by_frame": faces}}
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(path, result)
    return root, result


def prepare_source_pose(pose_root: Path) -> tuple[Path, dict]:
    from .atlas_video_inputs import GEOMETRY_INPUTS_FILE
    from .website_scene_geometry import _bound_frames
    from .website_worldlabs import prepare_atlas_pose_inputs

    original = json.loads((pose_root / GEOMETRY_INPUTS_FILE).read_text())
    # The background retains its frozen inputs; only the separate object pose
    # gains the capture's other admitted task-state observations.
    pose_root, original = task_source_inputs(pose_root, original)
    frames = _bound_frames(original, pose_root, geometry=False)
    root = pose_root / "full-view-source-v1"
    retained = root / GEOMETRY_INPUTS_FILE
    if retained.is_file():
        inputs = json.loads(retained.read_text())
        if inputs["binding"].get("parent_input_digest") != original["digest"]:
            raise ValueError("atlas_source_pose_parent_changed")
        _bound_frames(inputs, root, geometry=False)
        from .atlas_context_fov import full_view_sources
        full_view_sources(inputs, root)
    else:
        rows, sources = [], []
        for index, frame in enumerate(frames):
            paths = {}
            for key, folder in (("source_image_path", "observations"), ("image_path", "approved")):
                source = Path(frame[key])
                target = root / folder / f"{index:06d}{source.suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                paths[key] = str(target.resolve())
            with Image.open(paths["image_path"]) as image:
                padded, transform = square_pad(np.asarray(image.convert("RGB")))
            target = root / "inputs" / f"{index:06d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(padded).save(target)
            sources.append({"frame_id": frame["frame_id"], "image_path": paths["image_path"],
                            "image_digest": frame["image_digest"], "padding": transform})
            rows.append({**frame, "source_image_path": paths["source_image_path"],
                         "image_path": str(target.resolve()), "image_digest": _sha256_file(target)})
        inputs = {**original, "frames": rows, "full_view_sources": sources,
                  "binding": {**original["binding"], "parent_input_digest": original["digest"],
                              "pixel_policy": "square_pad_source_unchanged_v1"}}
        inputs["digest"] = canonical_digest(inputs, digest_field="digest")
        write_json(retained, inputs)
    return root, prepare_atlas_pose_inputs(inputs=inputs, input_root=root, output_root=root)
