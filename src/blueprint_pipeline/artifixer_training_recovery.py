"""Durably retain completed optimization before export can refuse its result."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .artifixer_source_geometry_admission import _record
from .decision_evidence_contracts import canonical_digest, canonical_json


def retain_training_checkpoint(
    *, checkpoint: Path, reference: Path, log: Path, request: dict[str, Any], destination: Path
) -> Path:
    if not {"artifixer_output", "artifixer_execution", "artifixer_bundle"}.isdisjoint(
        destination.parts
    ):
        raise ValueError("artifixer_training_recovery_archive_path_invalid")
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, source in (
        ("checkpoint.pt", checkpoint),
        ("reference_gaussians.ply", reference),
        ("training.log", log),
    ):
        if source.is_symlink() or not source.is_file() or not source.stat().st_size:
            raise ValueError("artifixer_training_recovery_source_invalid")
        target = destination / name
        try:
            if name == "training.log":
                shutil.copyfile(source, target)
            else:
                os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        record = _record(target)
        if record["sha256"] != _record(source)["sha256"]:
            raise ValueError("artifixer_training_recovery_copy_invalid")
        files[name] = record
    request_path = destination / "training_request.json"
    request_path.write_text(canonical_json(request) + "\n")
    files[request_path.name] = _record(request_path)
    receipt = {
        "schema_version": "artifixer_training_recovery.v1",
        "status": "optimization_complete_export_unqualified",
        "optimization_complete": True,
        "export_qualified": False,
        "checkpoint_contains_training_config": True,
        "files": files,
        "physical_or_deployment_evidence": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (destination / "training_recovery.json").write_text(canonical_json(receipt) + "\n")
    return destination


def retain_export_outcome(destination: Path, *, exception: Exception | None = None) -> None:
    value = {
        "schema_version": "artifixer_training_export_outcome.v1",
        "status": "export_refused" if exception else "export_completed_requires_review",
        "optimization_complete": True,
        "blockers": [str(exception)] if exception else [],
        "geometry_quality": getattr(exception, "geometry_quality", None),
        "physical_or_deployment_evidence": False,
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    (destination / "export_outcome.json").write_text(canonical_json(value) + "\n")


def retained_training_progress(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        (output_root.parent / "retained_training_evidence").glob("*/training_recovery.json")
    ):
        if path.is_symlink() or path.stat().st_size > 64_000:
            continue
        try:
            receipt = json.loads(path.read_text())
            if receipt.get("schema_version") != "artifixer_training_recovery.v1" or receipt.get(
                "receipt_digest"
            ) != canonical_digest(receipt, digest_field="receipt_digest"):
                continue
            rows.append(
                {
                    "task_id": path.parent.name,
                    "optimization_complete": True,
                    "export_qualified": False,
                    "recovery_receipt": _record(path),
                }
            )
        except (ValueError, OSError, TypeError):
            continue
    return rows


def _retain_bound_file(record: dict[str, Any], destination: Path) -> dict[str, Any]:
    source = Path(record["path"])
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError("artifixer_derivative_recovery_path_invalid")
    actual = _record(source)
    if any(actual[k] != record[k] for k in ("size_bytes", "sha256")):
        raise ValueError("artifixer_derivative_recovery_source_mismatch")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)
    retained = _record(destination)
    if any(retained[k] != actual[k] for k in ("size_bytes", "sha256")):
        raise ValueError("artifixer_derivative_recovery_copy_mismatch")
    return retained


def _derivative_root(destination: Path, name: str) -> Path:
    if not {"artifixer_output", "artifixer_execution", "artifixer_bundle"}.isdisjoint(
        destination.parts
    ):
        raise ValueError("artifixer_training_recovery_archive_path_invalid")
    root = destination / name
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("artifixer_derivative_recovery_path_invalid")
    root.mkdir(parents=True, exist_ok=False)
    return root


def retain_native_exports(destination: Path, native: dict[str, Any]) -> None:
    """Keep completed exports even if rendering or later review refuses."""
    root = _derivative_root(destination, "native_exports")
    records = {}
    for key, name in (
        ("standard_gaussian_ply", "repaired_scene.ply"),
        ("isaac_nurec_usdz", "repaired_scene.usdz"),
    ):
        records[key] = _retain_bound_file(native[key], root / name)
    receipt = {
        "schema_version": "artifixer_native_export_recovery.v1",
        "status": "retained_native_exports_requires_independent_review",
        "original_export": native,
        "retained_files": records,
        "physical_or_deployment_evidence": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (root / "native_export_recovery.json").write_text(canonical_json(receipt) + "\n")


def retain_review_frames(destination: Path, frames: list[dict[str, Any]]) -> None:
    """Preserve the exact PNG bytes that the independent reviewer will receive."""
    if not frames or [row["frame_index"] for row in frames] != list(range(len(frames))):
        raise ValueError("artifixer_review_recovery_frame_order_invalid")
    root = _derivative_root(destination, "review_frames")
    retained = []
    for row in frames:
        retained.append(
            {
                "camera_id": row["camera_id"],
                "frame_index": row["frame_index"],
                "original_frame": row,
                "retained_frame": _retain_bound_file(row, root / f"{row['frame_index']:05d}.png"),
            }
        )
    receipt = {
        "schema_version": "artifixer_review_frame_recovery.v1",
        "status": "retained_raw_frames_requires_independent_review",
        "frames": retained,
        "frame_count": len(retained),
        "physical_or_deployment_evidence": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (root / "review_frame_recovery.json").write_text(canonical_json(receipt) + "\n")
