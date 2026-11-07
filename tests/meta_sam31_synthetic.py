"""Synthetic hosted SAM 3.1 runs for memory and byte-identity tests; no customer data, no provider call.

``install`` replaces the provider (its HTTP opener), the video probe, the mask decoder and the
continuous-video encode with deterministic stand-ins, so the real ``run_website_task_masks`` ->
``run_meta_sam31`` -> ``parse_tracks`` chain runs on large synthetic masks at any frame size:
shelves on both sides and at the end of an aisle, a floor trapezoid that fills the nearest rows,
and optionally a small task object and two people who leave the view.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import numpy as np

SHELVES = ("left_shelf", "right_shelf", "end_shelf", "floor")
MIXED = SHELVES + ("box", "person")


def _jitter(rows, frame, salt, spread):
    return ((rows * 2654435761 + frame * 40503 + salt * 97) % (2 * spread + 1)) - spread


def _band(target: str, frame: int, width: int, height: int) -> tuple[list[int], np.ndarray]:
    sx, sy = width / 3840, height / 2160
    sway = int(40 * sx * math.sin(frame / 30.0))
    if target == "left_shelf":       # column band, ragged edges, gaps between shelf boards
        top, bottom = int(80 * sy), int(2080 * sy)
        rows = np.arange(top, bottom)
        x0 = np.maximum(0, int(60 * sx) + sway + _jitter(rows, frame, 1, 3))
        x1 = int(1180 * sx) + sway + _jitter(rows, frame, 2, 5)
        gap = ((rows // max(1, int(240 * sy))) % 2 == 0) & ((rows % max(1, int(240 * sy))) < int(40 * sy))
        g0, g1 = int(400 * sx) + sway, int(800 * sx) + sway
    elif target == "right_shelf":    # column band touching the right edge
        top, bottom = int(150 * sy), int(2010 * sy)
        rows = np.arange(top, bottom)
        x0 = int(2700 * sx) + sway + _jitter(rows, frame, 3, 5)
        x1 = np.full(rows.shape, width)
        gap = ((rows // max(1, int(200 * sy))) % 3 == 0) & ((rows % max(1, int(200 * sy))) < int(30 * sy))
        g0, g1 = int(3000 * sx) + sway, int(3400 * sx) + sway
    elif target == "end_shelf":      # band of rows between the side shelves
        top, bottom = int(300 * sy) + sway // 4, int(1500 * sy) + sway // 4
        rows = np.arange(top, bottom)
        x0 = int(1250 * sx) + sway + _jitter(rows, frame, 4, 4)
        x1 = int(2620 * sx) + sway + _jitter(rows, frame, 5, 4)
        gap = np.zeros(rows.shape, bool)
        g0 = g1 = 0
    elif target == "floor":          # trapezoid, full width in the nearest rows
        top, bottom = int(1260 * sy) + sway // 4, height
        rows = np.arange(top, bottom)
        t = (rows - top) / max(1, bottom - top)
        x0 = np.maximum(0, (int(1300 * sx) * (1 - 1.6 * t)).astype(int) + sway + _jitter(rows, frame, 6, 2))
        x1 = np.minimum(width, (width - int(1300 * sx) * (1 - 1.6 * t)).astype(int) + sway
                        + _jitter(rows, frame, 7, 2))
        gap = np.zeros(rows.shape, bool)
        g0 = g1 = 0
    else:
        raise ValueError(target)
    left, right = int(max(0, x0.min())), int(min(width, x1.max()))
    cols = np.arange(left, right)[None, :]
    raster = (cols >= x0[:, None]) & (cols < x1[:, None])
    if gap.any():
        raster &= ~(gap[:, None] & (cols >= g0) & (cols < g1))
    return [left, top, right, bottom], raster.astype(np.uint8)


def _blob(cx: float, cy: float, rx: float, ry: float, width: int, height: int) -> tuple[list[int], np.ndarray]:
    left, right = max(0, int(cx - rx)), min(width, int(cx + rx) + 1)
    top, bottom = max(0, int(cy - ry)), min(height, int(cy + ry) + 1)
    ys, xs = np.mgrid[top:bottom, left:right]
    raster = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1
    return [left, top, right, bottom], raster.astype(np.uint8)


def synthetic_masks(target: str, frame: int, width: int, height: int) -> list[tuple[str, list[int], np.ndarray]]:
    """(object id, box, box-local uint8 raster) for every instance of ``target`` in ``frame``."""
    if target in SHELVES:
        box, raster = _band(target, frame, width, height)
        return [("1", box, raster)]
    if target == "box":              # a small task object carried across the aisle
        return [("1", *_blob(width * (0.35 + 0.3 * frame / 437), height * 0.62,
                             width * 0.04, height * 0.05, width, height))]
    if target == "person":           # two people; the second leaves after a while
        found = [("1", *_blob(width * 0.2, height * (0.45 + 0.1 * math.sin(frame / 20)),
                              width * 0.05, height * 0.3, width, height))]
        if frame % 97 < 60:
            found.append(("2", *_blob(width * 0.8, height * 0.5, width * 0.06, height * 0.32, width, height)))
        return found
    raise ValueError(target)


def install(set_attribute: Callable[[Any, str, Any], None], root: Path, *, frames: int, width: int,
            height: int, targets: tuple[str, ...] = SHELVES) -> dict[str, Any]:
    """Patch the provider-facing seams and return the plan and source geometry of the synthetic job."""
    from blueprint_pipeline import meta_sam31 as sam, website_task_masks as masks
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from blueprint_pipeline.paid_resource_admission import (build_paid_lane_admission,
                                                            require_paid_resource_admission)

    root.mkdir(parents=True, exist_ok=True)
    source = root / "source.mp4"
    source.write_bytes(b"synthetic source video stand-in")
    source_digest = _sha256_file(source)

    def prepare(*, source, source_digest, root):
        root.mkdir(parents=True, exist_ok=True)
        video_path = root / "continuous-upright.mp4"
        if not video_path.is_file():
            video_path.write_bytes(b"synthetic continuous video stand-in")
        registry = [{"source_frame_id": f"decoded-{i:09d}", "model_frame_index": i,
                     "decoded_pts_seconds": i / 25, "width": width, "height": height,
                     "retained_video_digest": source_digest} for i in range(frames)]
        video = {"path": str(video_path), "sha256": _sha256_file(video_path),
                 "source_video_digest": source_digest, "encoding": "synthetic_stand_in"}
        receipt = {"registry": registry, "video": video}
        receipt["digest"] = canonical_digest(receipt, digest_field="digest")
        (root / "continuous-video.json").write_text(json.dumps(receipt))
        return registry, video

    def probe(path):
        return {"streams": [{"width": width, "height": height}],
                "frames": [{"best_effort_timestamp_time": str(i / 25)} for i in range(frames)]}

    def parse(text, accept):
        target = text.removeprefix("synthetic:")
        for index in range(frames):
            for object_id, box, raster in synthetic_masks(target, index, width, height):
                accept(object_id, index, box, raster.shape[1], raster.shape[0], raster.tobytes())
        return True

    def opener(request, **kwargs):
        if request.full_url.endswith("/v1/files"):
            return BytesIO(b'{"id":"file-synthetic"}')
        if request.get_method() == "DELETE":
            return BytesIO(b'{"deleted":true}')
        prompt = json.loads(request.data)["input"][0]["content"][0]["text"]
        return BytesIO(json.dumps({"status": "completed", "usage": {"video_frames_processed": frames},
                                   "output": [{"type": "message", "content": [
                                       {"type": "output_text", "text": "synthetic:" + prompt}]}]}).encode())

    hosted = sam.run_meta_sam31

    def run_with_grant(**kw):
        binding = sam.request_binding(frame_registry=kw["frame_registry"], frame_artifacts=kw["frame_artifacts"],
                                      prompts=kw["prompts"], video_artifact=kw.get("video_artifact"))
        authority = {**build_paid_lane_admission(resource_class="evaluator_api"),
                     "allocation_binding_digest": canonical_digest(binding), "maximum_cost_usd": 100.0,
                     "external_disclosure_allowed": True}
        grant = require_paid_resource_admission(authority, resource_class="evaluator_api",
                                                expected_schema_version="paid_lane_admission.v1")
        return hosted(**{**kw, "admission": authority, "admission_grant": grant, "opener": opener})

    set_attribute(sam, "_probe_video", probe)
    set_attribute(sam, "_parse_js_rasters", parse)
    set_attribute(masks, "prepare_continuous_video", prepare)
    set_attribute(masks, "run_meta_sam31", run_with_grant)

    anchor = min(frames - 1, 40)
    plan_targets = []
    for name in targets:
        if name == "person":
            plan_targets.append({"target_id": "person", "semantic_label": "person", "segmentation_prompt": "person",
                                 "target_class": "person", "target_role": "person", "task_effect": "privacy",
                                 "disposition": "remove", "spatial_evidence": []})
            continue
        (_, (left, top, right, bottom), _), *_ = synthetic_masks(name, anchor, width, height)
        moved = name == "box"
        plan_targets.append({"target_id": name, "semantic_label": name.replace("_", " "),
                             "segmentation_prompt": name,
                             "task_effect": "manipulated" if moved else "static_contact",
                             "disposition": "remove" if moved else "keep",
                             "target_role": "task_object" if moved else "support",
                             "spatial_evidence": [{"timestamp_seconds": anchor / 25, "box_xywh_normalized": [
                                 left / width, top / height, (right - left) / width, (bottom - top) / height]}]})
    step = max(1, frames // 16)
    geometry_frames = [{"frame_id": f"decoded-{i:09d}", "timestamp_seconds": i / 25,
                        "width": 518, "height": round(518 * height / width / 14) * 14}
                       for i in range(0, frames, step)][:16]
    source_geometry = {"schema_version": "website_source_frames.v1", "geometry_available": False,
                       "binding": {"source_video_digest": source_digest}, "geometry_input_digest": None,
                       "frames": geometry_frames}
    source_geometry["digest"] = canonical_digest(source_geometry, digest_field="digest")
    return {"plan": {"targets": plan_targets, "task_context_sha256": "synthetic-task"},
            "source_geometry": source_geometry, "source": source}


def persisted_digests(root: Path, *, skip: tuple[str, ...] = ("bench-",)) -> dict[str, str]:
    """sha256 of every JSON file under ``root`` with the run's own root path written as ``<root>``."""
    names = sorted({str(root), str(root.resolve())}, key=len, reverse=True)
    out = {}
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith(skip):
            continue
        data = path.read_bytes()
        for name in names:
            data = data.replace(name.encode(), b"<root>")
        out[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
    return out


def run_clean_plate(set_attribute: Callable[[Any, str, Any], None], root: Path, *, frames: int, width: int,
                    height: int, targets: tuple[str, ...] = SHELVES,
                    before_run: Callable[[], None] | None = None) -> dict[str, Any]:
    """The clean-plate stage's SAM work on the synthetic job: hosted tracks, task masks, stage manifest.

    Kept static surfaces are deferred, as the clean-plate stage defers them; the stage manifest
    holds the returned task masks the way ``run_clean_plate_stage`` writes them.
    """
    from blueprint_pipeline import common, website_task_masks as masks

    setup = install(set_attribute, root, frames=frames, width=width, height=height, targets=targets)
    if before_run is not None:
        before_run()
    manifest = masks.run_website_task_masks(
        plan=setup["plan"], source_geometry=setup["source_geometry"], output_root=root / "clean_plate" / "task_masks",
        source_video=setup["source"], defer_kept_static=True)
    common.write_json(root / "clean_plate" / "clean_plate_stage_manifest.json",
                      {"status": "noop", "source_frames": setup["source_geometry"], "task_masks": manifest})
    return {**setup, "task_masks": manifest}


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def main(argv: list[str]) -> int:
    """``python -m tests.meta_sam31_synthetic ROOT FRAMES WIDTH HEIGHT ADDRESS_SPACE_MIB``.

    Prints the peak resident memory before the run (interpreter, imports, synthetic setup) and
    after it, and the persisted files' digests. The address-space cap makes a memory regression
    fail inside this process with MemoryError instead of exhausting the host.
    """
    root, frames, width, height, cap_mib = Path(argv[0]), *map(int, argv[1:5])
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (cap_mib << 20, cap_mib << 20))
    os.environ.setdefault("META_MODEL_API_KEY", "synthetic-key")
    os.environ["BLUEPRINT_WEBSITE_SAM31_PROVIDER"] = "meta"
    baseline = {}
    run_clean_plate(setattr, root, frames=frames, width=width, height=height,
                    before_run=lambda: baseline.setdefault("rss", _peak_rss_bytes()))
    print(json.dumps({"baseline_peak_rss_bytes": baseline["rss"], "peak_rss_bytes": _peak_rss_bytes(),
                      "persisted": persisted_digests(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
