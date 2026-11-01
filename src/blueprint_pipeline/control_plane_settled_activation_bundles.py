"""Reclaim the provider bundle zip of an activation whose scene attempt is settled.

Every website scene-configuration activation writes a ~6 GB provider bundle
(``launch-set/bundle/*.zip``): the sealed toolchain plus the scene inputs,
packaged for one launch. The activation directory is pinned as evidence, so
the derived-directory phase keeps it for the evidence hot window (14 days).
That retention is right for its receipts and was never needed for the zip.

On 2026-11-01 capture-447ae4f4's launch was refused before allocation with
``scene_configuration_provider_output_disk_budget_exceeded``: seven activation
directories held ~40 GB of bundles for attempts that were already replaced
and settled, and the work volume had 5.9 GB usable.

Owner decision (2026-11-01, session_01VxbcMNBnpNS4uksXRTqbBj): once an attempt
has a validated terminal settlement, its bundle zip may be removed. Everything
else stays: the bundle receipt (with ``bundle_sha256``), the launch set, the
profile and the settlement. Nothing the settlement, capacity recovery or
profile verification reads is a zip: launch profiles bind the bundle *receipt*.
An attempt that is not settled (a live or recoverable one) keeps its zip, and
so does an activation any queue message still names.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "control_plane_settled_activation_bundle_manifest.v1"
RECEIPT_SCHEMA_VERSION = "control_plane_settled_activation_bundle_receipt.v1"
MARKER_SCHEMA_VERSION = "control_plane_settled_activation_bundle_reaped.v1"
ACK = "reap-settled-activation-bundles"
BUNDLE_RELATIVE = Path("launch-set/bundle/task_evaluation_scene_configuration_provider_bundle.zip")
BUNDLE_RECEIPT_NAME = "task_evaluation_scene_configuration_provider_bundle.v1.receipt.json"
MARKER_NAME = "bundle_reaped.json"
SETTLEMENT_DIRECTORY = "cancelled-unstarted-controls"
SETTLEMENT_SCHEMA = "task_evaluation_terminal_scene_attempt_settlement.v1"
LAUNCH_SUFFIX = "-launch"


class SettledActivationBundleError(ValueError):
    """The settled-bundle manifest could not be built or applied as sealed."""


def _file_record(path: Path) -> dict[str, Any]:
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": "sha256:" + digest}


def _settled_launches(intent_root: Path) -> list[dict[str, Any]]:
    """Every launch id a validated terminal settlement closes, with its evidence."""

    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_retained_controls_evidence import validated_cancellation

    rows: list[dict[str, Any]] = []
    for directory in sorted(intent_root.glob("scene-*")):
        settlements = directory / SETTLEMENT_DIRECTORY
        if directory.is_symlink() or not settlements.is_dir() or settlements.is_symlink():
            continue
        for path in sorted(settlements.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("schema_version") != SETTLEMENT_SCHEMA:
                    continue
                attempt = intake._read(directory / "attempts" / (str(value["attempt_id"]) + ".json"),
                                       "attempt_digest")
                # The intake's own validation: the same proof a spend-cap check accepts.
                if validated_cancellation(directory, attempt) != value:
                    continue
                launch_id = str((value.get("execution_terminal") or {}).get("launch_id") or "")
            except Exception:  # noqa: BLE001 - an unreadable or invalid settlement keeps its bundle
                continue
            if not launch_id.endswith(LAUNCH_SUFFIX) or "/" in launch_id or launch_id.startswith("."):
                continue
            rows.append({"activation_id": launch_id[: -len(LAUNCH_SUFFIX)], "attempt_id": value["attempt_id"],
                         "settlement": _file_record(path)})
    return rows


def _bundle(activation_root: Path, activation_id: str) -> Path | None:
    directory = activation_root / activation_id
    zip_path = directory / BUNDLE_RELATIVE
    parts = [directory, *[directory / Path(*BUNDLE_RELATIVE.parts[: i + 1]) for i in range(len(BUNDLE_RELATIVE.parts))]]
    if any(part.is_symlink() for part in parts) or not zip_path.is_file():
        return None
    return zip_path


def build_settled_activation_bundle_manifest(
    *,
    activation_roots: Sequence[str | Path],
    intent_root: str | Path,
    queue_roots: Sequence[str | Path] = (),
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """List bundle zips of settled attempts' activations, without mutating anything."""

    from .control_plane_storage_references import queue_reference_text

    observed_at = float(now())
    settled = _settled_launches(Path(intent_root))
    queue_text = queue_reference_text(queue_roots)
    candidates: list[dict[str, Any]] = []
    retained = {"no_bundle": 0, "queue_referenced": 0}
    roots = [Path(root) for root in activation_roots if Path(root).is_dir() and not Path(root).is_symlink()]
    for row in settled:
        for root in roots:
            zip_path = _bundle(root, row["activation_id"])
            if zip_path is None:
                continue
            if row["activation_id"] in queue_text:
                retained["queue_referenced"] += 1
                break
            candidates.append({**row, "root": str(root), "bundle_path": str(zip_path),
                               "size_bytes": zip_path.stat().st_size})
            break
        else:
            retained["no_bundle"] += 1
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "observed_at_epoch": observed_at,
        "intent_root": str(intent_root),
        "roots": [str(root) for root in roots],
        "queue_roots": [str(root) for root in queue_roots],
        "settled_launch_count": len(settled),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest


def apply_settled_activation_bundle_manifest(
    manifest: Mapping[str, Any], *, ack: str, now: Callable[[], float] = time.time
) -> dict[str, Any]:
    """Remove each candidate zip whose settlement and bytes are unchanged; leave a sealed marker."""

    from .control_plane_storage_references import queue_reference_text

    if (
        ack != ACK
        or manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("manifest_digest") != canonical_digest(dict(manifest), digest_field="manifest_digest")
    ):
        raise SettledActivationBundleError("settled_activation_bundle_apply_not_authorized")
    settled = {row["activation_id"]: row for row in _settled_launches(Path(str(manifest["intent_root"])))}
    queue_text = queue_reference_text(manifest.get("queue_roots") or ())
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        activation_id = str(row.get("activation_id") or "")
        zip_path = _bundle(Path(str(row.get("root") or "")), activation_id) if activation_id else None
        current = settled.get(activation_id)
        if (
            zip_path is None
            or str(zip_path) != row.get("bundle_path")
            or zip_path.stat().st_size != row.get("size_bytes")
            or current is None
            or current["settlement"] != row.get("settlement")
        ):
            skipped.append({"activation_id": activation_id, "reason": "candidate_changed"})
            continue
        if activation_id in queue_text:
            skipped.append({"activation_id": activation_id, "reason": "queue_referenced"})
            continue
        receipt_sha = None
        try:
            receipt = json.loads((zip_path.parent / BUNDLE_RECEIPT_NAME).read_text(encoding="utf-8"))
            receipt_sha = receipt.get("bundle_sha256")
        except (OSError, ValueError):
            pass
        try:
            os.unlink(zip_path)
        except OSError as exc:
            skipped.append({"activation_id": activation_id, "reason": f"remove_failed:{type(exc).__name__}"})
            continue
        marker = {
            "schema_version": MARKER_SCHEMA_VERSION,
            "activation_id": activation_id,
            "attempt_id": row["attempt_id"],
            "settlement": row["settlement"],
            "bundle_name": zip_path.name,
            "bundle_sha256": receipt_sha,
            "reaped_bytes": row["size_bytes"],
            "reaped_at_epoch": float(now()),
            "source_manifest_digest": manifest["manifest_digest"],
            "receipts_retained": True,
            "marker_digest": "",
        }
        marker["marker_digest"] = canonical_digest(marker, digest_field="marker_digest")
        (zip_path.parent / MARKER_NAME).write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
        removed.append({"activation_id": activation_id, "size_bytes": row["size_bytes"]})
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "candidate_count": manifest["candidate_count"],
        "removed_count": len(removed),
        "removed_bytes": sum(int(row["size_bytes"]) for row in removed),
        "removed": removed,
        "skipped": skipped,
        "evidence_removed": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


__all__ = [
    "ACK",
    "BUNDLE_RELATIVE",
    "MARKER_NAME",
    "SettledActivationBundleError",
    "apply_settled_activation_bundle_manifest",
    "build_settled_activation_bundle_manifest",
]
