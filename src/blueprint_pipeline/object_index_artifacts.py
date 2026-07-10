"""Resolve immutable object-index runs without falling back to stale outputs."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .common import read_json_any


def _mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        payload = read_json_any(path)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_path(root: Path, relative_text: str) -> Optional[Path]:
    if not relative_text or Path(relative_text).is_absolute() or ".." in Path(relative_text).parts:
        return None
    candidate = root / relative_text
    current = root
    for part in Path(relative_text).parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        candidate.resolve(strict=False).relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _object_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return []
    try:
        payload = read_json_any(path)
    except Exception:
        return []
    if isinstance(payload, Mapping):
        raw_rows = payload.get("objects") or payload.get("items") or payload.get("summaries") or []
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raw_rows = []
    return [dict(item) for item in raw_rows if isinstance(item, Mapping)]


def resolve_current_object_index_artifacts(capture_root: str | Path) -> Dict[str, Any]:
    """Return a verified current run, or an explicit capture-raw fallback.

    Once an immutable current pointer exists, an invalid/blocked run never falls
    through to a prior raw or descriptor path.
    """

    root = Path(capture_root).resolve()
    pointer_path = root / "pipeline" / "derived" / "object_index" / "current.json"
    pointer = _mapping(pointer_path)
    if pointer_path.exists():
        blockers: list[str] = []
        if pointer.get("schema_version") != "object_index_current_pointer.v1":
            blockers.append("current_pointer_schema_mismatch")
        run_path = _contained_path(root, str(pointer.get("run_path") or ""))
        if run_path is None or not run_path.is_dir():
            blockers.append("current_run_path_invalid")
            run_path = None
        run_manifest = _mapping(run_path / "run_manifest.json") if run_path else {}
        fingerprint = str(pointer.get("run_fingerprint") or "")
        if (
            run_path is not None
            and (
                run_manifest.get("schema_version") != "object_index_derived_run.v1"
                or run_manifest.get("run_fingerprint") != fingerprint
                or run_path.name != fingerprint
            )
        ):
            blockers.append("current_run_fingerprint_or_schema_mismatch")

        expected_members = run_manifest.get("member_sha256")
        actual_members: Dict[str, str] = {}
        if run_path is not None:
            if any(path.is_symlink() for path in run_path.rglob("*")):
                blockers.append("current_run_symlink_member_forbidden")
            actual_members = {
                path.relative_to(run_path).as_posix(): _sha256_file(path)
                for path in sorted(run_path.rglob("*"))
                if path.is_file() and not path.is_symlink() and path.name != "run_manifest.json"
            }
        if not isinstance(expected_members, Mapping) or dict(expected_members) != actual_members:
            blockers.append("current_run_member_set_or_hash_mismatch")

        object_index_path = run_path / "object_index.json" if run_path else None
        object_index_payload = _mapping(object_index_path) if object_index_path else {}
        if object_index_payload.get("schema_version") != "object_index.v2":
            blockers.append("current_object_index_schema_mismatch")
        report_payload = _mapping(run_path / "object_index_build_report.json") if run_path else {}
        if report_payload.get("schema_version") != "object_index_build_report.v2":
            blockers.append("current_build_report_schema_mismatch")
        objects = _object_rows(object_index_path) if object_index_path else []
        try:
            pointer_count = int(pointer.get("current_usable_object_count"))
            manifest_count = int(run_manifest.get("current_usable_object_count"))
        except (TypeError, ValueError):
            pointer_count = -1
            manifest_count = -1
        if pointer_count != manifest_count or pointer_count != len(objects):
            blockers.append("current_usable_object_count_mismatch")
        expected_status = "ready" if pointer_count > 0 else "blocked_zero_usable_artifacts"
        if pointer.get("status") != expected_status or run_manifest.get("status") != expected_status:
            blockers.append("current_status_mismatch")
        if pointer.get("raw_intake_digest") != run_manifest.get("raw_intake_digest"):
            blockers.append("current_raw_intake_digest_mismatch")

        valid = not blockers
        ready = valid and pointer_count > 0
        return {
            "schema_version": "resolved_object_index_artifacts.v1",
            "status": "ready" if ready else "blocked_zero_usable_artifacts" if valid else "invalid",
            "source": "immutable_derived_run",
            "pointer_path": str(pointer_path),
            "run_path": str(run_path) if run_path else None,
            "run_fingerprint": fingerprint or None,
            "raw_intake_digest": pointer.get("raw_intake_digest"),
            "current_usable_object_count": pointer_count if valid else 0,
            "object_index_path": str(object_index_path) if ready and object_index_path else None,
            "audit_object_index_path": str(object_index_path) if valid and object_index_path else None,
            "build_report_path": str(run_path / "object_index_build_report.json") if valid and run_path else None,
            "grounding_hints_path": str(run_path / "object_grounding_hints.json") if valid and run_path else None,
            "keyframes_path": str(run_path / "object_index_keyframes.json") if valid and run_path else None,
            "artifacts_root": str(run_path / "artifacts") if valid and run_path else None,
            "objects": objects if ready else [],
            "blockers": blockers,
        }

    for raw_path in (
        root / "raw" / "object_index.json",
        root / "raw" / "objects" / "index.json",
        root / "raw" / "arkit" / "objects" / "index.json",
        root / "object_index.json",
    ):
        if not raw_path.is_file() or raw_path.is_symlink():
            continue
        objects = _object_rows(raw_path)
        return {
            "schema_version": "resolved_object_index_artifacts.v1",
            "status": "ready" if objects else "blocked_zero_usable_artifacts",
            "source": "capture_raw",
            "pointer_path": None,
            "run_path": None,
            "run_fingerprint": None,
            "raw_intake_digest": None,
            "current_usable_object_count": len(objects),
            "object_index_path": str(raw_path) if objects else None,
            "audit_object_index_path": str(raw_path),
            "build_report_path": str(root / "raw" / "object_index_build_report.json"),
            "grounding_hints_path": str(root / "raw" / "object_grounding_hints.json"),
            "keyframes_path": str(root / "raw" / "object_index_keyframes.json"),
            "artifacts_root": str(root / "raw" / "object_index_artifacts"),
            "objects": objects,
            "blockers": [],
        }

    return {
        "schema_version": "resolved_object_index_artifacts.v1",
        "status": "missing",
        "source": None,
        "current_usable_object_count": 0,
        "object_index_path": None,
        "audit_object_index_path": None,
        "build_report_path": None,
        "grounding_hints_path": None,
        "keyframes_path": None,
        "artifacts_root": None,
        "objects": [],
        "blockers": ["object_index_missing"],
    }
