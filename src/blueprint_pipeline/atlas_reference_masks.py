"""SAM 3.1 masks of each scale reference's own object, on the posed views it is measured in.

A reference sets the scale only when both endpoints lie on one instance of the
object it names (``atlas_scene_measurements``). One SAM call per job, on the
reference views only; retained responses make a retry free. Everything comes
from the single captured video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

PROMPT_PREFIX = "scale_reference_"


def segment_scale_references(*, frames: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]],
                             root: Path, task_context: Mapping[str, Any],
                             source_digest: str | None) -> list[dict[str, Any]] | None:
    """Per named reference, every SAM 3.1 instance of its object in its own view.

    None when no reference names its object (a retained plan from before the
    names): those references are measured as placed.
    """
    from .meta_sam31 import run_meta_sam31

    named = [(index, row) for index, row in enumerate(references) if (row.get("segmentation_prompt") or "").strip()]
    if not named:
        return None
    if not source_digest:
        raise ValueError("atlas_reference_source_video_digest_missing")
    by_id = {frame["frame_id"]: frame for frame in frames}
    views = sorted({row["frame_id"] for _, row in named if row["frame_id"] in by_id})
    tracks = run_meta_sam31(
        frame_registry=[{"source_frame_id": frame_id, "width": int(by_id[frame_id]["rgb"].shape[1]),
                         "height": int(by_id[frame_id]["rgb"].shape[0])} for frame_id in views],
        frame_artifacts=[{"source_frame_id": frame_id, "path": by_id[frame_id]["image_path"],
                          "sha256": by_id[frame_id]["image_digest"]} for frame_id in views],
        prompts=[{"prompt_id": f"{PROMPT_PREFIX}{index}", "output_label": f"{PROMPT_PREFIX}{index}",
                  "text": row["segmentation_prompt"].strip()} for index, row in named
                 if row["frame_id"] in by_id],
        output_root=root, admission={}, task_context=task_context, source_video_digest=source_digest,
    )["tracks"] if views else []
    return [{"reference_index": index, "frame_id": row["frame_id"],
             "instances": [observation for track in tracks if track["label"] == f"{PROMPT_PREFIX}{index}"
                           for observation in track["observations"]
                           if observation["source_frame_id"] == row["frame_id"]]}
            for index, row in named]
