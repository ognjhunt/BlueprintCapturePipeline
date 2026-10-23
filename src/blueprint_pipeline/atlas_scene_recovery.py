"""Recover reviewed Atlas RGBD omitted by an earlier, narrower gap selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file

RECOVERY_REVISION = "v1"


def reviewed_completion_views(reports):
    """Read completed, visually reviewed native outputs without repurchasing."""
    from .atlas_scene_completion import load_generated_views

    views, seen = [], set()
    for report in reports:
        if (report.get("status") != "passed"
                or report.get("digest") != canonical_digest(report, digest_field="digest")):
            raise ValueError("atlas_recovery_review_not_bound")
        if not report.get("generated_roots") and report.get("review", {}).get("status") == "not_required":
            continue
        if report.get("review", {}).get("status") != "passed":
            raise ValueError("atlas_recovery_review_not_bound")
        admitted = set(report.get("generation", {}).get("native_result_digests", []))
        for root in report.get("generated_roots", []):
            root = Path(root)
            result = json.loads((root / "result.json").read_text())
            if result["digest"] not in admitted:
                raise ValueError("atlas_recovery_native_result_not_reviewed")
            ids = [f"recovered-{result['digest'][7:23]}-{i}" for i in
                   range(len(result["operation"]["response"]["frames"]))]
            native, frames, _ = load_generated_views(root, ids)
            for index, frame in enumerate(frames):
                key = (native["digest"], index)
                if key in seen:
                    continue
                seen.add(key)
                views.append({"frame": frame, "native_result_digest": native["digest"],
                              "native_frame_index": index, "review_digest": report["digest"]})
    return views


def _frame_binding(frame):
    return {"frame_id": frame["frame_id"], "intrinsics": frame["intrinsics"],
            "extrinsics": frame["extrinsics"], **{key: hashlib.sha256(
                np.ascontiguousarray(frame[key]).tobytes()).hexdigest()
                for key in ("rgb", "depth", "confidence")}}


def recover_reviewed_geometry(*, frames, geometry_frames, reviewed_views, root, checkpoint=None):
    """Fill remaining bounded gaps from already reviewed full native views.

    Earlier masks selected where to request generation. They must not permanently
    hide other recoverable holes in those same reviewed outputs. Re-survey each
    camera against all retained geometry and admit only named missing regions.
    Existing RGB, geometry, observed free space, and scene bounds still win.
    """
    from .atlas_scene_chisel import project_source_depth
    from .atlas_scene_gaps import gap_labels, select_gap_mask
    from .atlas_scene_preservation import (
        prepare_completion_views, preserve_generated_view, observed_scene_bounds, _save_image,
    )
    from .atlas_depth_splat import surface_samples

    if any(not view.get("review_digest") for view in reviewed_views):
        raise ValueError("atlas_recovery_review_not_bound")
    binding = {"revision": RECOVERY_REVISION,
               "sources": [_frame_binding(f) for f in frames],
               "geometry": [_frame_binding(f) for f in geometry_frames],
               "native": [{**{k: v for k, v in row.items() if k != "frame"},
                           "frame": _frame_binding(row["frame"])} for row in reviewed_views]}
    digest = canonical_digest(binding)
    root = root / digest[7:23]
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "recovery.json"
    by_id = {row["frame"]["frame_id"]: row for row in reviewed_views}
    if len(by_id) != len(reviewed_views):
        raise ValueError("atlas_recovery_duplicate_frame")
    if saved.is_file():
        report = json.loads(saved.read_text())
        if (report.get("digest") != canonical_digest(report, digest_field="digest")
                or report.get("binding_digest") != digest):
            raise ValueError("atlas_recovery_inputs_changed")
        for row in report["assets"]:
            if Path(row["path"]).is_symlink() or _sha256_file(Path(row["path"])) != row["sha256"]:
                raise ValueError("atlas_recovery_output_changed")
        recovered = []
        for row in report["views"]:
            if not row.get("exportable_samples"):
                continue
            native = by_id[row["frame_id"]]["frame"]
            recovered.append({**native,
                "rgb": np.asarray(Image.open(row["image_path"]).convert("RGB")),
                "confidence": native["confidence"] *
                    (np.asarray(Image.open(row["mask_path"]).convert("L")) > 0)})
        return recovered, report

    extent = observed_scene_bounds(frames)
    candidates = [{"camera_id": row["frame"]["frame_id"],
                   "source_frame_id": min(frames, key=lambda f: np.linalg.norm(
                       f["position"] - row["frame"]["position"]))["frame_id"],
                   "camera": {"intrinsics": row["frame"]["intrinsics"],
                              "extrinsics": row["frame"]["extrinsics"]}}
                  for row in reviewed_views]
    prepared = prepare_completion_views(frames=frames, selected=candidates, root=root / "context")
    recovered, rows, paths = [], [], []
    geometry = list(geometry_frames)
    for index, (view, source) in enumerate(zip(reviewed_views, prepared, strict=True)):
        if checkpoint:
            checkpoint("automation_completion_recover_geometry")
        frame = view["frame"]
        stage = root / f"{index:04d}"
        stage.mkdir(exist_ok=True)
        covered = project_source_depth(geometry, source["camera"])
        known = np.asarray(Image.open(source["keep_mask_path"]).convert("L")) == 255
        labels, regions = gap_labels(covered, known, scene_outline=True)
        # This is a bounded per-region continuation, not permission to fill
        # the entire new camera or extend the capture's world-space bounds.
        eligible = [r for r in regions if r["pixel_count"] <= int(known.sum())]
        row = {"frame_id": frame["frame_id"], "candidate_gap_pixels": int(np.count_nonzero(labels)),
               "eligible_gap_pixels": sum(r["pixel_count"] for r in eligible),
               "native_result_digest": view["native_result_digest"],
               "native_frame_index": view["native_frame_index"], "review_digest": view["review_digest"],
               "exportable_samples": 0, "added_geometry_pixels": 0}
        if eligible:
            label_path, depth_path = stage / "labels.png", stage / "known-depth.png"
            _save_image(label_path, labels)
            _save_image(depth_path, (covered > 0).astype(np.uint8) * 255)
            regions = [{**r, "gap_id": frame["frame_id"] + "-gap-" + str(r["label"])} for r in eligible]
            gaps = {"label_path": str(label_path), "label_sha256": _sha256_file(label_path), "gaps": regions}
            source.update(select_gap_mask(gaps, [r["gap_id"] for r in regions], stage / "selected.png"),
                          source_depth_context_path=str(depth_path),
                          source_depth_context_digest=_sha256_file(depth_path))
            fixed, image, receipt, assets = preserve_generated_view(
                generated=frame, prepared=source, references=geometry, output_root=stage / "preserved",
                native_result_digest=view["native_result_digest"], native_frame_index=view["native_frame_index"],
                extent_bounds=extent)
            samples = len(surface_samples(fixed)["points"])
            mask_path = next(a["path"] for a in receipt["assets"] if a["role"] == "generated_geometry_mask")
            row.update(exportable_samples=samples, added_geometry_pixels=receipt["generated_geometry_pixel_count"],
                       image_path=image["image_path"], mask_path=mask_path,
                       requested_gap_ids=source["completion_gap_ids"])
            paths.extend([label_path, depth_path, Path(source["completion_gap_mask_path"]), *assets])
            if samples:
                recovered.append(fixed)
                geometry.append(fixed)
        rows.append(row)
        paths.extend([Path(source["image_path"]), Path(source["keep_mask_path"])])
    report = {"schema_version": "atlas_geometry_recovery.v1", "binding_digest": digest,
              "revision": RECOVERY_REVISION, "views": rows,
              "new_provider_calls": 0, "source_geometry_modified": False, "known_pixels_changed": 0,
              "added_geometry_pixels": sum(r["added_geometry_pixels"] for r in rows),
              "exportable_samples": sum(r["exportable_samples"] for r in rows),
              "generation_extent": extent, "whole_scene_complete": False,
              "claim_ceiling": "development_only",
              "assets": [{"path": str(p), "sha256": _sha256_file(p)} for p in paths],
              "preservation_paths": [str(p) for p in paths] + [str(saved)]}
    report["digest"] = canonical_digest(report, digest_field="digest")
    write_json(saved, report)
    return recovered, report
