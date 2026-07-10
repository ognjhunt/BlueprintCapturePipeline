"""Integrity, retention, freshness, and disclosure controls for city-launch evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_SCHEMA_VERSION = "city-launch-harness-run.v2"
DEFAULT_FRESHNESS_DAYS = 7
DEFAULT_RETENTION_DAYS = 365
EVIDENCE_CLASSIFICATION = "internal_operational_evidence"

_PERSONAL_PATH_PATTERNS = (
    re.compile(r"/Users/[^/\s\"']+"),
    re.compile(r"/home/[^/\s\"']+"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\s\"']+"),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_utc(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def default_output_root() -> Path:
    configured = os.environ.get("BLUEPRINT_CITY_LAUNCH_OUTPUT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    state_root = Path(state_home).expanduser() if state_home else Path.home() / ".local" / "state"
    return state_root / "blueprint" / "city-launch-runs"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def prepare_evidence_root(output_root: Path, *, source_root: Path) -> Path:
    """Create a private evidence root and reject storage inside the source checkout."""

    candidate = output_root.expanduser()
    if candidate.is_symlink():
        raise ValueError("city-launch evidence root must not be a symlink")
    resolved = candidate.resolve()
    source = source_root.resolve()
    if resolved == source or _is_relative_to(resolved, source):
        raise ValueError(
            "city-launch evidence must be stored outside the source checkout; "
            "set BLUEPRINT_CITY_LAUNCH_OUTPUT_ROOT to a private artifact location"
        )
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(resolved, 0o700)
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise ValueError(f"city-launch evidence root must be private (0700); found {mode:04o}")
    return resolved


def prepare_run_root(run_root: Path) -> None:
    run_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(run_root, 0o700)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_inventory(run_root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"city-launch evidence must not contain symlinks: {path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative_path = path.relative_to(run_root).as_posix()
        inventory.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def evidence_policy(*, created_at: datetime | None = None) -> dict[str, Any]:
    created = (created_at or utc_now()).astimezone(timezone.utc)
    return {
        "classification": EVIDENCE_CLASSIFICATION,
        "freshness_days": DEFAULT_FRESHNESS_DAYS,
        "retention_days": DEFAULT_RETENTION_DAYS,
        "delete_after": (created + timedelta(days=DEFAULT_RETENTION_DAYS)).isoformat(),
        "access_control": "private_root_mode_0700",
        "external_disclosure_approved": False,
        "disclosure_note": (
            "Operational evidence can contain local paths and service metadata. "
            "Export only through an independently reviewed redaction process."
        ),
        "claim_boundary": "A manifest proves artifact integrity, not launch readiness.",
    }


def _personal_path_findings(text: str) -> list[str]:
    findings: list[str] = []
    for pattern in _PERSONAL_PATH_PATTERNS:
        findings.extend(match.group(0) for match in pattern.finditer(text))
    return sorted(set(findings))


def validate_run(
    run_root: Path,
    *,
    now: datetime | None = None,
    max_age_days: int | None = DEFAULT_FRESHNESS_DAYS,
    require_disclosure_approval: bool = False,
) -> dict[str, Any]:
    """Validate exact inventory, checksums, retention, freshness, and disclosure state."""

    root = run_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("city-launch manifest.json is missing or unsafe")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("city-launch manifest must be a JSON object")
    if raw.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError(f"unsupported city-launch manifest schema: {raw.get('schema_version')!r}")

    created = parse_utc(raw.get("created_or_updated_at"), field="created_or_updated_at")
    current = (now or utc_now()).astimezone(timezone.utc)
    age = current - created
    if age.total_seconds() < -300:
        raise ValueError("city-launch evidence timestamp is unreasonably in the future")
    if max_age_days is not None and age > timedelta(days=max_age_days):
        raise ValueError(f"city-launch evidence is stale: age exceeds {max_age_days} days")

    policy = raw.get("evidence_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("city-launch manifest evidence_policy is missing")
    if policy.get("classification") != EVIDENCE_CLASSIFICATION:
        raise ValueError("city-launch evidence classification is invalid")
    if policy.get("retention_days") != DEFAULT_RETENTION_DAYS:
        raise ValueError("city-launch retention_days does not match policy")
    delete_after = parse_utc(policy.get("delete_after"), field="evidence_policy.delete_after")
    if delete_after != created + timedelta(days=DEFAULT_RETENTION_DAYS):
        raise ValueError("city-launch delete_after does not match creation time and retention policy")
    if require_disclosure_approval and policy.get("external_disclosure_approved") is not True:
        raise ValueError("city-launch evidence is not approved for external disclosure")

    declared = raw.get("artifact_inventory")
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise ValueError("city-launch artifact_inventory must be an array")
    actual = build_artifact_inventory(root)
    if list(declared) != actual:
        raise ValueError("city-launch artifact inventory, size, or SHA-256 mismatch")

    personal_paths: list[str] = []
    for entry in actual:
        artifact_path = root / str(entry["path"])
        if artifact_path.suffix in {".json", ".jsonl", ".txt", ".md"}:
            personal_paths.extend(
                _personal_path_findings(artifact_path.read_text(encoding="utf-8", errors="replace"))
            )
    personal_paths = sorted(set(personal_paths))
    if require_disclosure_approval and personal_paths:
        raise ValueError("city-launch disclosure contains personal absolute paths")

    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_root": str(root),
        "artifact_count": len(actual),
        "fresh": max_age_days is None or age <= timedelta(days=max_age_days),
        "age_seconds": max(0.0, age.total_seconds()),
        "delete_after": delete_after.isoformat(),
        "personal_absolute_path_count": len(personal_paths),
        "external_disclosure_approved": policy.get("external_disclosure_approved") is True,
        "valid": True,
    }
