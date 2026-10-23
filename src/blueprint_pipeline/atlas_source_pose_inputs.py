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


def prepare_source_pose(pose_root: Path) -> tuple[Path, dict]:
    from .atlas_video_inputs import GEOMETRY_INPUTS_FILE
    from .website_scene_geometry import _bound_frames
    from .website_worldlabs import prepare_atlas_pose_inputs

    original = json.loads((pose_root / GEOMETRY_INPUTS_FILE).read_text())
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
