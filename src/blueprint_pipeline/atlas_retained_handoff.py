"""Adopt a completed background into its capture job without another provider call.

The portable assets keep their bytes and receipts. Only the controller's paths
change; the original completed result remains beside the adopted result.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("digest") != canonical_digest(value, digest_field="digest"):
        raise ValueError("atlas_retained_handoff_record_changed")
    return value


def _copy(source: Path, target: Path, expected: str) -> None:
    if source.is_symlink() or _sha256_file(source) != expected:
        raise ValueError("atlas_retained_handoff_asset_changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or _sha256_file(target) != expected:
            raise ValueError("atlas_retained_handoff_identity_conflict")
        return
    shutil.copyfile(source, target)
    if _sha256_file(target) != expected:
        raise ValueError("atlas_retained_handoff_copy_changed")


def retain_background(*, automation_root: Path, job_dir: Path, source_video_digest: str,
                      task_context_digest: str) -> dict:
    """Validate the capture/context and retain the exact completed portable bundle."""
    original = _read(automation_root / "result.json")
    prepared = _read(automation_root / "background/prepared_background.json")
    background = original.get("background") or {}
    binding = prepared.get("binding") or {}
    if (original.get("status") != "completed" or background.get("status") != "completed"
            or not background.get("mesh_available")
            or binding.get("source_video_digest") != source_video_digest
            or binding.get("task_context_digest") != task_context_digest):
        raise ValueError("atlas_retained_handoff_capture_mismatch")
    download = background["download"]
    bundle_root = Path(download["directory"]).resolve()
    manifest = _read(bundle_root / "manifest.json")
    if (manifest["digest"] != download["manifest_digest"]
            or manifest.get("source_video_digest") != source_video_digest
            or manifest.get("splat_manifest_digest") != background.get("splat_manifest_digest")
            or not manifest.get("splat_available") or not manifest.get("mesh_available")):
        raise ValueError("atlas_retained_handoff_bundle_mismatch")
    target_root = job_dir / "automation" / "retained" / original["digest"][7:]
    target = target_root / bundle_root.name
    assets = [*manifest["assets"], {"path": "manifest.json", "sha256": _sha256_file(bundle_root / "manifest.json")}]
    # Validate every path and hash before copying anything into the capture job.
    for row in assets:
        relative = Path(row["path"])
        path = bundle_root / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(bundle_root) or _sha256_file(path) != row["sha256"]):
            raise ValueError("atlas_retained_handoff_asset_changed")
    archive = Path(download["zip_path"])
    if archive.is_symlink() or _sha256_file(archive) != download["zip_sha256"]:
        raise ValueError("atlas_retained_handoff_archive_changed")
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != {row["path"] for row in assets} or zipped.testzip() is not None:
            raise ValueError("atlas_retained_handoff_archive_invalid")
    for row in assets:
        _copy(bundle_root / row["path"], target / row["path"], row["sha256"])
    _copy(archive, target.with_suffix(".zip"), download["zip_sha256"])
    # Alignment reads this native receipt only; its digest and frame identities remain intact.
    pose = _read(bundle_root / "evidence/atlas_posed_rgbd_result.json")
    if pose["digest"] != manifest["pose_result_digest"]:
        raise ValueError("atlas_retained_handoff_pose_changed")
    _copy(bundle_root / "evidence/atlas_posed_rgbd_result.json",
          job_dir / "automation" / ("fallback_pose-retained-" + pose["digest"][7:23]) / "atlas_posed_rgbd_result.json",
          _sha256_file(bundle_root / "evidence/atlas_posed_rgbd_result.json"))
    write_json(target_root / "original_result.json", original)
    adopted = deepcopy(original)
    adopted["retained_result_digest"] = original["digest"]
    adopted["background"]["download"] = {**download, "directory": str(target.resolve()),
                                           "zip_path": str(target.with_suffix(".zip").resolve())}
    adopted["digest"] = canonical_digest(adopted, digest_field="digest")
    write_json(target_root / "result.json", adopted)
    return adopted
