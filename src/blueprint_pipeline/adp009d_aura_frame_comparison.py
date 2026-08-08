"""Compare two retained ADP-009D Aura frames-only probes.

This is an operational admission comparison, not an image-quality judge.  It
verifies that camera, geometry, depth, and semantic evidence are held constant,
then detects the narrow failure where a candidate payload contributes no visible
background appearance.  Without an independently bound truth image it always
leaves ``quality_winner`` null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCHEMA_VERSION = "adp009d_aura_frame_comparison.v1"
PROBE_FILENAME = "adp009d_frames_only_probe.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_probe(root: Path) -> dict[str, Any]:
    probe_path = root / PROBE_FILENAME
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if probe.get("schema_version") != "adp009d_frames_only_probe.v1":
        raise ValueError("frames_only_probe_schema_invalid")
    if probe.get("status") != "completed" or probe.get("mode") != "frames_only":
        raise ValueError("frames_only_probe_not_completed")
    return probe


def _camera_rows(probe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = probe.get("camera_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frames_only_camera_rows_missing")
    indexed = {
        str(row.get("camera_id")): row for row in rows if isinstance(row, dict)
    }
    if len(indexed) != len(rows) or "None" in indexed:
        raise ValueError("frames_only_camera_ids_invalid")
    return indexed


def _load_bound_array(root: Path, binding: dict[str, Any]) -> np.ndarray:
    path = root / str(binding.get("path", ""))
    if not path.is_file() or _sha256(path) != binding.get("sha256"):
        raise ValueError("frames_only_bound_array_digest_mismatch")
    return np.load(path, allow_pickle=False)


def _load_bound_rgb(root: Path, binding: dict[str, Any]) -> np.ndarray:
    path = root / str(binding.get("path", ""))
    if not path.is_file() or _sha256(path) != binding.get("sha256"):
        raise ValueError("frames_only_bound_rgb_digest_mismatch")
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def compare_variants(
    *,
    baseline_root: Path,
    candidate_root: Path,
    baseline_asset: Path | None = None,
    candidate_asset: Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic, evidence-bounded operational comparison."""

    baseline_probe = _load_probe(baseline_root)
    candidate_probe = _load_probe(candidate_root)
    baseline_rows = _camera_rows(baseline_probe)
    candidate_rows = _camera_rows(candidate_probe)
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("frames_only_camera_set_mismatch")

    comparisons: list[dict[str, Any]] = []
    held_constant = True
    candidate_appearance_absent = False
    for camera_id in sorted(baseline_rows):
        baseline = baseline_rows[camera_id]
        candidate = candidate_rows[camera_id]
        constant_fields = (
            "frame_index",
            "intrinsic_matrix",
            "position_world_m",
            "quaternion_world_opengl_xyzw",
            "resolution_hw",
            "sim_time_seconds",
        )
        field_matches = {
            field: baseline.get(field) == candidate.get(field)
            for field in constant_fields
        }
        depth_digest_matches = (
            baseline.get("metric_depth", {}).get("sha256")
            == candidate.get("metric_depth", {}).get("sha256")
        )
        semantic_digest_matches = (
            baseline.get("semantic_segmentation", {}).get("sha256")
            == candidate.get("semantic_segmentation", {}).get("sha256")
        )
        camera_constant = (
            all(field_matches.values())
            and depth_digest_matches
            and semantic_digest_matches
        )
        held_constant = held_constant and camera_constant

        baseline_rgb = _load_bound_rgb(baseline_root, baseline["rgb_png"])
        candidate_rgb = _load_bound_rgb(candidate_root, candidate["rgb_png"])
        baseline_semantic = _load_bound_array(
            baseline_root, baseline["semantic_segmentation"]
        )
        candidate_semantic = _load_bound_array(
            candidate_root, candidate["semantic_segmentation"]
        )
        if baseline_rgb.shape != candidate_rgb.shape:
            raise ValueError("frames_only_rgb_shape_mismatch")
        if baseline_semantic.shape != candidate_semantic.shape:
            raise ValueError("frames_only_semantic_shape_mismatch")
        if not np.array_equal(baseline_semantic, candidate_semantic):
            raise ValueError("frames_only_semantic_content_mismatch")

        background = baseline_semantic == 0
        if background.ndim == 3 and background.shape[-1] == 1:
            background = background[..., 0]
        if background.shape != baseline_rgb.shape[:2] or not np.any(background):
            raise ValueError("frames_only_background_mask_invalid")
        baseline_visible = np.max(baseline_rgb, axis=2) > 1
        candidate_visible = np.max(candidate_rgb, axis=2) > 1
        baseline_fraction = float(np.mean(baseline_visible[background]))
        candidate_fraction = float(np.mean(candidate_visible[background]))
        appearance_absent = baseline_fraction >= 0.1 and candidate_fraction < 0.01
        candidate_appearance_absent = candidate_appearance_absent or appearance_absent
        comparisons.append(
            {
                "camera_id": camera_id,
                "held_constant": camera_constant,
                "constant_field_matches": field_matches,
                "depth_digest_matches": depth_digest_matches,
                "semantic_digest_matches": semantic_digest_matches,
                "baseline_rgb_sha256": baseline["rgb_png"]["sha256"],
                "candidate_rgb_sha256": candidate["rgb_png"]["sha256"],
                "baseline_rgb_mean": float(np.mean(baseline_rgb)),
                "candidate_rgb_mean": float(np.mean(candidate_rgb)),
                "baseline_visible_background_fraction": baseline_fraction,
                "candidate_visible_background_fraction": candidate_fraction,
                "candidate_appearance_absent": appearance_absent,
            }
        )

    blockers = ["independent_quality_reference_missing"]
    if not held_constant:
        blockers.append("frames_only_comparison_inputs_not_held_constant")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if held_constant else "blocked",
        "quality_winner": None,
        "supports_quality_winner_claim": False,
        "operational_decision": (
            "retain_baseline_reject_candidate_as_drop_in"
            if held_constant and candidate_appearance_absent
            else "manual_reference_adjudication_required"
        ),
        "candidate_appearance_absent": candidate_appearance_absent,
        "held_constant": held_constant,
        "camera_comparisons": comparisons,
        "blockers": sorted(blockers),
    }
    if baseline_asset is not None:
        receipt["baseline_asset"] = {
            "path": str(baseline_asset),
            "sha256": _sha256(baseline_asset),
            "size_bytes": baseline_asset.stat().st_size,
        }
    if candidate_asset is not None:
        receipt["candidate_asset"] = {
            "path": str(candidate_asset),
            "sha256": _sha256(candidate_asset),
            "size_bytes": candidate_asset.stat().st_size,
        }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--baseline-asset")
    parser.add_argument("--candidate-asset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = compare_variants(
        baseline_root=Path(args.baseline_root),
        candidate_root=Path(args.candidate_root),
        baseline_asset=Path(args.baseline_asset) if args.baseline_asset else None,
        candidate_asset=Path(args.candidate_asset) if args.candidate_asset else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
