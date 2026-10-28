"""Source frames for website task preparation, before pixel editing.

ADP-009B / public_scene_day_14: preserve source placement for a development
replica. The decoder owns source frames; World Labs Atlas poses them
(``website_worldlabs``, ``atlas_source_geometry``). Estimated metres never
establish measured scale or physical proof.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .core.handoff_lease import guarded_handoff_write_json, handoff_effect, handoff_lease_checkpoint
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import LocalDecodedObservationAdapter, _sha256_file

# Retained adapter-v1 inputs name the model they were first prepared for; they still reopen.
LEGACY_MODEL_ID = "facebook/map-anything-apache"
LEGACY_MODEL_CODE_REVISION = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
INPUT_SCHEMA = "website_geometry_inputs.v1"
_FRAME_ARTIFACTS = (("source_image_path", "source_image_digest"),
                    ("image_path", "image_digest"), ("geometry_path", "geometry_digest"))


def source_to_view_pixels(*, width: int, height: int, rotation: float, size: tuple[int, int]) -> list[list[float]]:
    """Source pixel (integer centres) to a view of the upright frame resized to ``size``."""
    if not np.isfinite(rotation) or rotation % 90:
        raise ValueError("website_source_rotation_not_supported")
    transforms = {
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
        180: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        270: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
    }
    upright_width, upright_height = (height, width) if int(rotation) % 180 else (width, height)
    sx, sy = size[0] / upright_width, size[1] / upright_height
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return (resize @ np.array(transforms[int(rotation) % 360])).tolist()


def _prepare_image(source: Path, target: Path, rotation: float) -> dict[str, Any]:
    """Keep an exact source-to-model pixel transform, including phone rotation."""
    if not np.isfinite(rotation) or rotation % 90:
        raise ValueError("website_source_rotation_not_supported")
    with Image.open(source) as original:
        width, height = original.size
        image = original.convert("RGB").rotate(rotation, expand=True)
        scale = 518 / max(image.size)
        size = tuple(max(14, round(dimension * scale / 14) * 14) for dimension in image.size)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.resize(size, Image.Resampling.LANCZOS).save(target)
    return {
        "image_path": str(target), "image_digest": _sha256_file(target),
        "source_width": width, "source_height": height,
        "width": size[0], "height": size[1], "display_rotation_degrees": rotation,
        "source_to_geometry_pixels": source_to_view_pixels(width=width, height=height, rotation=rotation,
                                                           size=size),
    }


def _bound_frames(document: Mapping[str, Any], root: Path, *, geometry: bool) -> list[dict[str, Any]]:
    if document.get("digest") != canonical_digest(document, digest_field="digest"):
        raise ValueError("website_source_geometry_digest_mismatch")
    rows = document.get("frames")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("website_source_frames_invalid")
    limit = document.get("binding", {}).get("maximum_frames")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or limit < 2 or len(rows) > limit):
        raise ValueError("website_source_frames_invalid")
    frames = []
    for row in rows:
        frame = dict(row)
        for path_key, digest_key in _FRAME_ARTIFACTS if geometry else _FRAME_ARTIFACTS[:2]:
            path = Path(frame[path_key])
            path = path if path.is_absolute() else root / path
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("website_source_geometry_artifact_escape")
            if not path.is_file() or _sha256_file(path) != frame[digest_key]:
                raise ValueError("website_source_geometry_artifact_changed")
            frame[path_key] = str(path.resolve())
        frames.append(frame)
    if len({row["frame_id"] for row in frames}) != len(frames):
        raise ValueError("website_source_frame_identity_duplicate")
    return frames


def prepare_website_geometry_inputs(*, source_video: Path, output_root: Path, capture_id: str,
                                    maximum_frames: int | None = None) -> dict[str, Any]:
    """Decode on CPU. The portable worker input directory excludes held-out frames."""
    output_root = output_root.resolve()
    manifest_path = output_root / "worker_inputs" / "geometry_inputs.json"
    if maximum_frames is None:
        try:
            maximum_frames = int(os.getenv("BLUEPRINT_WEBSITE_SOURCE_MAXIMUM_FRAMES", "16"))
        except ValueError as exc:
            raise ValueError("website_source_frame_count_invalid") from exc
    if isinstance(maximum_frames, bool) or not isinstance(maximum_frames, int) or maximum_frames < 2:
        raise ValueError("website_source_frame_count_invalid")
    binding = {"source_video_digest": _sha256_file(source_video), "maximum_frames": maximum_frames,
               "adapter_version": 2}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        legacy_binding = {"source_video_digest": binding["source_video_digest"], "model_id": LEGACY_MODEL_ID,
                          "model_code_revision": LEGACY_MODEL_CODE_REVISION, "adapter_version": 1}
        accepted_bindings = (binding, legacy_binding) if maximum_frames == 16 else (binding,)
        if previous.get("binding") in accepted_bindings and previous.get("schema_version") == INPUT_SCHEMA:
            _bound_frames(previous, manifest_path.parent, geometry=False)
            return previous
        raise ValueError("website_geometry_input_identity_conflict")
    handoff_lease_checkpoint("website_source_frames_before_decode")
    decoded = LocalDecodedObservationAdapter().execute(
        intake_id=capture_id, capture_digest=binding["source_video_digest"],
        capture_authority_profile="monocular_video", capture_root=source_video.parent,
        video_relative_path=source_video.name, output_root=output_root / "source",
        rights_and_retention={"local_processing_authorized": True,
                              "provider_upload_authorized": False, "paid_compute_authorized": False},
        maximum_frames=maximum_frames,
    )
    handoff_lease_checkpoint("website_source_frames_after_decode")
    refs = decoded["asset_references"]
    candidate_path = output_root / "source" / refs["candidate_dataset_manifest"]["relative_path"]
    index_path = output_root / "source" / refs["decoded_observation_index"]["relative_path"]
    candidate, decoded_index = json.loads(candidate_path.read_text()), json.loads(index_path.read_text())
    if candidate.get("heldout_pixels_included") is not False or not candidate["frames"]:
        raise ValueError("website_source_frames_invalid")
    rotation = float(decoded_index["stream_metadata"].get("display_rotation_degrees") or 0)
    prepared = []
    for index, row in enumerate(candidate["frames"]):
        source = (candidate_path.parent / row["candidate_relative_path"]).resolve()
        if not source.is_relative_to(candidate_path.parent) or _sha256_file(source) != row["frame_digest"]:
            raise ValueError("website_source_frame_binding_invalid")
        retained = manifest_path.parent / "observations" / f"{index:06d}{source.suffix}"
        with handoff_effect("website_source_frame_promotion"):
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, retained)
            frame = {
                "frame_id": row["frame_id"], "timestamp_seconds": row["t_video_sec"],
                "source_image_path": str(retained), "source_image_digest": row["frame_digest"],
                **_prepare_image(source, manifest_path.parent / "model_inputs" / f"{index:06d}.png", rotation),
            }
        for key, _ in _FRAME_ARTIFACTS[:2]:
            frame[key] = str(Path(frame[key]).relative_to(manifest_path.parent))
        prepared.append(frame)
    result = {"schema_version": INPUT_SCHEMA, "binding": binding, "frames": prepared,
              "source_sample_count": decoded_index.get("selected_frame_count"),
              "heldout_frame_count": decoded_index.get("hidden_heldout_frame_count"),
              "heldout_pixels_included": False, "claim_ceiling": "development_only"}
    result["digest"] = canonical_digest(result, digest_field="digest")
    guarded_handoff_write_json(manifest_path, result)
    return result


def prepare_website_source_frames(*, source_video: Path, output_root: Path, capture_id: str,
                                maximum_frames: int | None = None) -> dict[str, Any]:
    """CPU-only source views for SAM and image editing, before any GPU job."""
    inputs = prepare_website_geometry_inputs(source_video=source_video, output_root=output_root,
                                           capture_id=capture_id, maximum_frames=maximum_frames)
    value = {"schema_version": "website_source_frames.v1", "binding": inputs["binding"],
             "geometry_input_digest": inputs["digest"], "geometry_available": False,
             "frames": _bound_frames(inputs, output_root / "worker_inputs", geometry=False)}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def run_website_scene_geometry(*, source_video: Path, output_root: Path, capture_id: str,
                               task_context: Mapping[str, Any] | None = None,
                               maximum_frames: int | None = None) -> dict[str, Any]:
    """Prepare source views; provider pose/depth never implies finished scene assets."""
    if task_context is not None:
        if task_context.get("capture_id") != capture_id:
            raise ValueError("website_geometry_task_capture_mismatch")
        if task_context.get("confirmed") is not True:
            if any(not isinstance(task_context.get(key), str) or not task_context[key].strip()
                   for key in ("request_id", "scene_id")):
                raise ValueError("website_task_context_identity_mismatch")
            from .website_task_context_contracts import validate_website_task_context
            validate_website_task_context(task_context,
                request_id=task_context.get("request_id"), scene_id=task_context.get("scene_id"),
                capture_id=capture_id, purpose="scene_preparation")
    if (output_root / "atlas_scene_review.json").is_file():
        from .atlas_video_inputs import prepare_video_inputs, survey_video
        survey_path = output_root / "atlas_video_survey.json"
        if not survey_path.is_file():
            survey_video(source_video=source_video, output_root=output_root, capture_id=capture_id)
        return prepare_video_inputs(survey_path=survey_path,
            review=json.loads((output_root / "atlas_scene_review.json").read_text()),
            output_root=output_root, maximum_frames=maximum_frames)
    inputs = prepare_website_geometry_inputs(source_video=source_video, output_root=output_root,
                                           capture_id=capture_id, maximum_frames=maximum_frames)
    # No API mutation or paid allocation here. Atlas consumes unposed RGB,
    # then its own returned RGB/depth/cameras feed generation unchanged.
    # The existing controller must admit uploads and operation spending.
    from .website_worldlabs import prepare_atlas_pose_inputs
    return prepare_atlas_pose_inputs(inputs=inputs,
        input_root=output_root / "worker_inputs", output_root=output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-video", type=Path, required=True)
    prepare.add_argument("--capture-id", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--maximum-frames", type=int,
                         help="Source sample count (default 16 or BLUEPRINT_WEBSITE_SOURCE_MAXIMUM_FRAMES); provider capacity is separate")
    args = vars(parser.parse_args())
    args.pop("command")
    result = prepare_website_geometry_inputs(**args)
    print(json.dumps({"status": result.get("status", "prepared"), "frame_count": len(result["frames"]),
                      "digest": result["digest"]}))


if __name__ == "__main__":
    main()
