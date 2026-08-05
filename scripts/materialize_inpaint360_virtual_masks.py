#!/usr/bin/env python3
"""Materialize deterministic Inpaint360GS virtual-view masks from model labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


SCHEMA_VERSION = "inpaint360_virtual_mask_handoff.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _require_under(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("inpaint360_virtual_mask_path_outside_runtime_root") from exc
    return resolved


def materialize_virtual_masks(
    *,
    runtime_root: Path,
    predicted_mask_dir: Path,
    output_dir: Path,
    receipt_path: Path,
    target_instance_id: int,
    expected_count: int = 30,
) -> dict[str, object]:
    root = runtime_root.expanduser().resolve()
    source = _require_under(predicted_mask_dir, root)
    output = _require_under(output_dir, root)
    receipt = _require_under(receipt_path, root)
    if not source.is_dir():
        raise ValueError("inpaint360_virtual_predicted_mask_dir_missing")
    if not 1 <= target_instance_id <= 255:
        raise ValueError("inpaint360_virtual_target_instance_id_invalid")
    source_paths = sorted(source.glob("*.png"))
    if len(source_paths) != expected_count:
        raise ValueError("inpaint360_virtual_predicted_mask_count_mismatch")
    if output.exists() and any(output.iterdir()):
        raise ValueError("inpaint360_virtual_output_dir_not_empty")
    output.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)

    source_records: list[dict[str, object]] = []
    output_records: list[dict[str, object]] = []
    expected_shape: tuple[int, int] | None = None
    for source_path in source_paths:
        with Image.open(source_path) as image:
            labels = np.asarray(image)
        if labels.ndim != 2:
            raise ValueError("inpaint360_virtual_predicted_mask_not_single_channel")
        shape = (int(labels.shape[0]), int(labels.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("inpaint360_virtual_predicted_mask_shape_mismatch")
        binary = np.where(labels == target_instance_id, 255, 0).astype(np.uint8)
        foreground_pixels = int(np.count_nonzero(binary))
        if foreground_pixels == 0:
            raise ValueError("inpaint360_virtual_target_missing_from_view")
        output_path = output / source_path.name
        Image.fromarray(binary, mode="L").save(output_path, format="PNG", optimize=False)
        source_records.append(
            {
                "relative_path": source_path.relative_to(root).as_posix(),
                "size_bytes": source_path.stat().st_size,
                "sha256": _sha256(source_path),
            }
        )
        output_records.append(
            {
                "relative_path": output_path.relative_to(root).as_posix(),
                "size_bytes": output_path.stat().st_size,
                "sha256": _sha256(output_path),
                "foreground_pixels": foreground_pixels,
            }
        )

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "target_instance_id": target_instance_id,
        "view_count": len(output_records),
        "image_height": expected_shape[0] if expected_shape else None,
        "image_width": expected_shape[1] if expected_shape else None,
        "source_kind": "inpaint360gs_full_scene_virtual_objects_pred",
        "handoff_kind": "binary_target_mask_without_interactive_refinement",
        "source_masks": source_records,
        "output_masks": output_records,
        "blockers": [],
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--predicted-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-instance-id", type=int, required=True)
    parser.add_argument("--expected-count", type=int, default=30)
    args = parser.parse_args()
    materialize_virtual_masks(
        runtime_root=args.runtime_root,
        predicted_mask_dir=args.predicted_mask_dir,
        output_dir=args.output_dir,
        receipt_path=args.receipt,
        target_instance_id=args.target_instance_id,
        expected_count=args.expected_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
