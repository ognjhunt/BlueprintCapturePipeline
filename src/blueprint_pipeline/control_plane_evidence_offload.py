"""Move sealed run evidence off the control-plane disk without deleting it.

Run directories under the ``evidence_cold`` roots are append-only evidence and
today exist as exactly one copy on a single root disk.  Once a run has a
terminal receipt and its hot window has passed, this module packs the directory
into one archive, publishes it to the content-addressed artifact store with a
full streaming readback, writes a digest-bound pointer beside where the
directory stood, and only then removes the local directory.  ``restore``
reverses the migration byte-for-byte.  Nothing here ever touches
``evidence_hot`` roots such as the spend guard.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .control_plane_storage_roots import require_storage_class
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_configured_scene_object_store import (
    materialize_configured_scene_artifact,
    publish_configured_scene_artifact,
)


MANIFEST_SCHEMA_VERSION = "control_plane_evidence_offload_manifest.v1"
RECEIPT_SCHEMA_VERSION = "control_plane_evidence_offload_receipt.v1"
POINTER_SCHEMA_VERSION = "control_plane_evidence_offload_pointer.v1"
RESTORE_SCHEMA_VERSION = "control_plane_evidence_restore_receipt.v1"
ARTIFACT_KIND = "control-plane-evidence"
EXECUTE_ACK = "offload-sealed-evidence"
POINTER_SUFFIX = ".offloaded.v1.json"
TERMINAL_RECEIPT_NAMES = ("dispatch_receipt.json", "launch_receipt.json")
DEFAULT_HOT_WINDOW_SECONDS = 14 * 24 * 60 * 60


class ControlPlaneEvidenceOffloadError(RuntimeError):
    """Evidence could not be offloaded or restored safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tree_snapshot(directory: Path) -> tuple[float, int, int]:
    latest = directory.lstat().st_mtime
    size = 0
    count = 0
    for root, directories, files in os.walk(directory):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            path = Path(root) / name
            try:
                metadata = path.lstat()
            except OSError:
                continue
            latest = max(latest, metadata.st_mtime)
            size += metadata.st_size
            count += 1
    return latest, size, count


def _terminal_receipt(directory: Path) -> str | None:
    for name in TERMINAL_RECEIPT_NAMES:
        candidate = directory / name
        if candidate.is_file() and not candidate.is_symlink():
            return name
    return None


def build_evidence_offload_manifest(
    *,
    evidence_roots: Sequence[str | Path],
    hot_window_seconds: int = DEFAULT_HOT_WINDOW_SECONDS,
    now: Callable[[], float] = time.time,
    classifier: Callable[..., Any] = require_storage_class,
) -> dict[str, Any]:
    """List sealed run directories past their hot window, without mutating anything."""

    if (
        not evidence_roots
        or not isinstance(hot_window_seconds, int)
        or isinstance(hot_window_seconds, bool)
        or hot_window_seconds < 0
    ):
        raise ControlPlaneEvidenceOffloadError("control_plane_evidence_offload_input_invalid")
    observed_at = float(now())
    candidates: list[dict[str, Any]] = []
    retained = {"active_or_unsealed": 0, "hot": 0, "already_offloaded": 0, "unsafe": 0}
    roots: list[str] = []
    for raw_root in evidence_roots:
        root = Path(raw_root).expanduser()
        classifier(
            str(root), expected="evidence_cold", code="control_plane_evidence_offload_root_class"
        )
        if root.is_symlink() or not root.is_dir():
            raise ControlPlaneEvidenceOffloadError("control_plane_evidence_offload_root_unsafe")
        roots.append(str(root))
        for child in sorted(root.iterdir()):
            if child.name.startswith(".") or child.name.endswith(POINTER_SUFFIX):
                continue
            if child.is_symlink() or not child.is_dir():
                retained["unsafe"] += 1
                continue
            if (root / f"{child.name}{POINTER_SUFFIX}").exists():
                retained["already_offloaded"] += 1
                continue
            receipt = _terminal_receipt(child)
            if receipt is None:
                retained["active_or_unsealed"] += 1
                continue
            latest, size, count = _tree_snapshot(child)
            if observed_at - latest < hot_window_seconds:
                retained["hot"] += 1
                continue
            candidates.append(
                {
                    "root": str(root),
                    "name": child.name,
                    "terminal_receipt": receipt,
                    "size_bytes": size,
                    "file_count": count,
                    "idle_seconds": int(observed_at - latest),
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "dry_run",
        "hot_window_seconds": hot_window_seconds,
        "roots": roots,
        "candidate_count": len(candidates),
        "candidate_bytes": sum(row["size_bytes"] for row in candidates),
        "candidates": candidates,
        "retained_counts": retained,
        "evidence_hot_roots_scanned": False,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest


def _pack(directory: Path, archive_path: Path) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "w") as archive:
        for root, directories, files in os.walk(directory):
            directories.sort()
            directories[:] = [
                name for name in directories if not (Path(root) / name).is_symlink()
            ]
            for name in sorted(files):
                path = Path(root) / name
                if path.is_symlink() or not path.is_file():
                    continue
                relative = path.relative_to(directory).as_posix()
                info = archive.gettarinfo(str(path), arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
                members.append(
                    {
                        "relative_path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    return members


def apply_evidence_offload(
    manifest: Mapping[str, Any],
    *,
    ack: str,
    publisher: Callable[..., Mapping[str, Any]] = publish_configured_scene_artifact,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Offload every manifest candidate whose state is unchanged; keep the rest."""

    if (
        ack != EXECUTE_ACK
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(dict(manifest), digest_field="manifest_digest")
    ):
        raise ControlPlaneEvidenceOffloadError(
            "control_plane_evidence_offload_apply_not_authorized"
        )
    offloaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in manifest.get("candidates") or []:
        root = Path(str(row.get("root") or ""))
        name = str(row.get("name") or "")
        directory = root / name
        pointer = root / f"{name}{POINTER_SUFFIX}"
        if (
            not name
            or "/" in name
            or name.startswith(".")
            or directory.is_symlink()
            or not directory.is_dir()
            or pointer.exists()
            or _terminal_receipt(directory) != row.get("terminal_receipt")
        ):
            skipped.append({"name": name, "reason": "candidate_changed"})
            continue
        descriptor, archive_name = tempfile.mkstemp(prefix=f".{name}.offload-", suffix=".tar", dir=root)
        os.close(descriptor)
        archive_path = Path(archive_name)
        try:
            members = _pack(directory, archive_path)
            digest = _sha256(archive_path)
            size = archive_path.stat().st_size
            reference = dict(publisher(path=archive_path, artifact_kind=ARTIFACT_KIND))
            if (
                reference.get("digest") != digest
                or reference.get("size_bytes") != size
                or reference.get("full_byte_service_account_readback_passed") is not True
            ):
                raise ControlPlaneEvidenceOffloadError(
                    "control_plane_evidence_offload_publication_mismatch"
                )
            payload: dict[str, Any] = {
                "schema_version": POINTER_SCHEMA_VERSION,
                "status": "offloaded",
                "directory": name,
                "terminal_receipt": row.get("terminal_receipt"),
                "uri": reference["uri"],
                "digest": digest,
                "size_bytes": size,
                "member_count": len(members),
                "members": members,
                "offloaded_at_epoch": float(now()),
                "evidence_deleted": False,
                "pointer_digest": "",
            }
            payload["pointer_digest"] = canonical_digest(payload, digest_field="pointer_digest")
            temporary = root / f".{name}.pointer-{os.getpid()}.tmp"
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.chmod(0o440)
            os.replace(temporary, pointer)
            shutil.rmtree(directory)
        except Exception as exc:  # noqa: BLE001 - every candidate is independent
            skipped.append({"name": name, "reason": f"offload_failed:{type(exc).__name__}"})
            continue
        finally:
            archive_path.unlink(missing_ok=True)
        offloaded.append({"name": name, "uri": reference["uri"], "digest": digest, "size_bytes": size})
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "applied",
        "source_manifest_digest": manifest["manifest_digest"],
        "offloaded_count": len(offloaded),
        "offloaded_bytes": sum(row["size_bytes"] for row in offloaded),
        "offloaded": offloaded,
        "skipped": skipped,
        "evidence_deleted": False,
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def restore_offloaded_evidence(
    *,
    pointer_path: str | Path,
    destination: str | Path,
    materializer: Callable[..., Mapping[str, Any]] = materialize_configured_scene_artifact,
) -> dict[str, Any]:
    """Bring an offloaded run directory back, verifying every member digest."""

    pointer_file = Path(pointer_path).expanduser()
    if pointer_file.is_symlink() or not pointer_file.is_file():
        raise ControlPlaneEvidenceOffloadError("control_plane_evidence_restore_pointer_invalid")
    pointer = json.loads(pointer_file.read_text(encoding="utf-8"))
    if (
        not isinstance(pointer, Mapping)
        or pointer.get("schema_version") != POINTER_SCHEMA_VERSION
        or pointer.get("pointer_digest")
        != canonical_digest(dict(pointer), digest_field="pointer_digest")
    ):
        raise ControlPlaneEvidenceOffloadError("control_plane_evidence_restore_pointer_invalid")
    target = Path(destination).expanduser()
    if target.exists() or target.is_symlink():
        raise ControlPlaneEvidenceOffloadError("control_plane_evidence_restore_destination_exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".restore-", dir=target.parent))
    try:
        archive_path = staging / "evidence.tar"
        materializer(
            reference={
                "uri": pointer["uri"],
                "digest": pointer["digest"],
                "size_bytes": pointer["size_bytes"],
            },
            destination=archive_path,
            maximum_size_bytes=int(pointer["size_bytes"]),
        )
        if _sha256(archive_path) != pointer["digest"]:
            raise ControlPlaneEvidenceOffloadError("control_plane_evidence_restore_digest_mismatch")
        extracted = staging / "tree"
        extracted.mkdir()
        with tarfile.open(archive_path) as archive:
            archive.extractall(extracted, filter="data")
        expected = {row["relative_path"]: row for row in pointer["members"]}
        observed = {
            path.relative_to(extracted).as_posix(): path
            for path in extracted.rglob("*")
            if path.is_file()
        }
        if set(expected) != set(observed) or any(
            _sha256(observed[name]) != expected[name]["sha256"]
            or observed[name].stat().st_size != expected[name]["size_bytes"]
            for name in expected
        ):
            raise ControlPlaneEvidenceOffloadError("control_plane_evidence_restore_member_mismatch")
        os.replace(extracted, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {
        "schema_version": RESTORE_SCHEMA_VERSION,
        "status": "restored",
        "directory": str(target),
        "member_count": len(pointer["members"]),
        "digest": pointer["digest"],
    }


__all__ = [
    "ARTIFACT_KIND",
    "ControlPlaneEvidenceOffloadError",
    "DEFAULT_HOT_WINDOW_SECONDS",
    "EXECUTE_ACK",
    "POINTER_SUFFIX",
    "TERMINAL_RECEIPT_NAMES",
    "apply_evidence_offload",
    "build_evidence_offload_manifest",
    "restore_offloaded_evidence",
]
