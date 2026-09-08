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
