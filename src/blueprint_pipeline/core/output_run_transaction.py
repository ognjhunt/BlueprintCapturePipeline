"""Cross-process lease and final commit protocol for generated output trees."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import uuid
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from .common import read_json_any, utc_now_iso, write_json


OUTPUT_RUN_LEASE_SCHEMA_VERSION = "blueprint_output_run_lease.v1"
OUTPUT_RUN_COMMIT_SCHEMA_VERSION = "blueprint_output_run_commit.v1"
OUTPUT_RUN_COMMIT_NAME = "run_commit.json"
_CURRENT_OUTPUT_RUN: ContextVar[dict[str, Any] | None] = ContextVar(
    "blueprint_current_output_run",
    default=None,
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(
                f"output_run_commit_symlink_forbidden:{path.relative_to(output_dir)}"
            )
        if not path.is_file() or path.name == OUTPUT_RUN_COMMIT_NAME:
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            }
        )
    return rows


def _payload_sha256(value: Mapping[str, Any]) -> str:
    import json

    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class OutputRunTransaction:
    """Serialize writers and expose only a final hashed commit as trustworthy."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        lane: str,
        request_fingerprint: str,
        reset_output: bool = True,
        preserve_top_level: Sequence[str] = (),
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.lane = str(lane or "").strip()
        self.request_fingerprint = str(request_fingerprint or "").strip()
        self.reset_output = bool(reset_output)
        self.preserve_top_level = frozenset(
            name
            for name in (str(item).strip() for item in preserve_top_level)
            if name and name not in {".", ".."} and "/" not in name and "\\" not in name
        )
        if not self.lane:
            raise ValueError("output_run_lane_required")
        if not self.request_fingerprint:
            raise ValueError("output_run_request_fingerprint_required")
        self.run_id = f"output-run-{uuid.uuid4().hex}"
        self.lock_path = self.output_dir.parent / f".{self.output_dir.name}.run.lock"
        self.lease_path = self.output_dir.parent / f".{self.output_dir.name}.run-lease.json"
        self.commit_path = self.output_dir / OUTPUT_RUN_COMMIT_NAME
        self._lock_file: BinaryIO | None = None
        self._committed = False
        self._context_token: Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> OutputRunTransaction:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        self._lock_file = self.lock_path.open("r+b")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if self.output_dir.is_symlink():
                raise RuntimeError("output_run_directory_symlink_forbidden")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.reset_output:
                for child in self.output_dir.iterdir():
                    if child.name in self.preserve_top_level:
                        continue
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink(missing_ok=True)
            else:
                self.commit_path.unlink(missing_ok=True)
            write_json(
                self.lease_path,
                {
                    "schema_version": OUTPUT_RUN_LEASE_SCHEMA_VERSION,
                    "status": "active",
                    "run_id": self.run_id,
                    "lane": self.lane,
                    "request_fingerprint": self.request_fingerprint,
                    "acquired_at": utc_now_iso(),
                    "output_dir_name": self.output_dir.name,
                    "final_commit_required": True,
                    "output_reset_before_run": self.reset_output,
                    "preserved_top_level_names": sorted(self.preserve_top_level),
                },
            )
        except BaseException:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
            raise
        self._context_token = _CURRENT_OUTPUT_RUN.set(self.descriptor())
        return self

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": OUTPUT_RUN_LEASE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "lane": self.lane,
            "request_fingerprint": self.request_fingerprint,
            "commit_path": OUTPUT_RUN_COMMIT_NAME,
            "final_commit_required": True,
            "output_reset_before_run": self.reset_output,
            "preserved_top_level_names": sorted(self.preserve_top_level),
        }

    def commit(self) -> dict[str, Any]:
        if self._lock_file is None:
            raise RuntimeError("output_run_transaction_not_active")
        inventory = _inventory(self.output_dir)
        inventory_sha256 = _payload_sha256({"files": inventory})
        commit = {
            "schema_version": OUTPUT_RUN_COMMIT_SCHEMA_VERSION,
            "status": "committed",
            "committed_at": utc_now_iso(),
            "run_id": self.run_id,
            "lane": self.lane,
            "request_fingerprint": self.request_fingerprint,
            "file_count": len(inventory),
            "inventory_sha256": inventory_sha256,
            "files": inventory,
            "atomic_file_writes_required": True,
            "exclusive_run_lease_held_through_commit": True,
        }
        write_json(self.commit_path, commit)
        write_json(
            self.lease_path,
            {
                **self.descriptor(),
                "status": "committed",
                "committed_at": commit["committed_at"],
                "inventory_sha256": inventory_sha256,
            },
        )
        self._committed = True
        return commit

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if not self._committed:
                self.commit_path.unlink(missing_ok=True)
                try:
                    write_json(
                        self.lease_path,
                        {
                            **self.descriptor(),
                            "status": "failed_uncommitted",
                            "failed_at": utc_now_iso(),
                            "error_type": exc_type.__name__ if exc_type else None,
                        },
                    )
                except OSError:
                    pass
        finally:
            if self._lock_file is not None:
                try:
                    os.fsync(self._lock_file.fileno())
                except OSError:
                    pass
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
                self._lock_file.close()
                self._lock_file = None
            if self._context_token is not None:
                _CURRENT_OUTPUT_RUN.reset(self._context_token)
                self._context_token = None


def current_output_run_descriptor() -> dict[str, Any]:
    value = _CURRENT_OUTPUT_RUN.get()
    return dict(value) if isinstance(value, Mapping) else {}


def verify_output_run_commit(
    output_dir: str | Path,
    *,
    expected_request_fingerprint: str | None = None,
) -> dict[str, Any]:
    resolved = Path(output_dir).expanduser().resolve()
    path = resolved / OUTPUT_RUN_COMMIT_NAME
    try:
        payload = read_json_any(path)
    except (OSError, ValueError):
        payload = {}
    commit = dict(payload) if isinstance(payload, Mapping) else {}
    blockers: list[str] = []
    if commit.get("schema_version") != OUTPUT_RUN_COMMIT_SCHEMA_VERSION:
        blockers.append("output_run_commit_schema_invalid")
    if commit.get("status") != "committed":
        blockers.append("output_run_not_committed")
    if expected_request_fingerprint and commit.get(
        "request_fingerprint"
    ) != expected_request_fingerprint:
        blockers.append("output_run_request_fingerprint_mismatch")
    try:
        inventory = _inventory(resolved)
        observed_inventory_sha256 = _payload_sha256({"files": inventory})
    except (OSError, RuntimeError, ValueError):
        inventory = []
        observed_inventory_sha256 = ""
        blockers.append("output_run_inventory_unreadable")
    if commit.get("inventory_sha256") != observed_inventory_sha256:
        blockers.append("output_run_inventory_digest_mismatch")
    if commit.get("file_count") != len(inventory):
        blockers.append("output_run_file_count_mismatch")
    return {
        "status": "passed" if not blockers else "blocked",
        "commit_path": str(path),
        "run_id": commit.get("run_id"),
        "request_fingerprint": commit.get("request_fingerprint"),
        "inventory_sha256": commit.get("inventory_sha256"),
        "observed_inventory_sha256": observed_inventory_sha256,
        "blockers": blockers,
    }
