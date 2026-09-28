"""Compact obsolete input ZIPs without losing any member's bytes.

This is operator-directed retention for a revoked, terminal attempt. It does
not delete observations or runtime outputs. Common members stay in an explicitly
pinned successor ZIP; bounded differing members stay in the durable plan.
Reconstructed members are byte-exact; original ZIP container identity is not
promised. Existing receipts and the original archive digest remain evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_intake import _read

ACK = "reap-revoked-input-zip-preserve-every-member"
SCHEMA = "task_evaluation_revoked_input_bundle_retention.v1"
MAX_UNIQUE_MEMBER_BYTES = 2 * 1024**2
MAX_UNIQUE_BYTES = 32 * 1024**2
BUNDLE_NAME = "task_evaluation_scene_configuration_provider_bundle.zip"


def _sha(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b""):
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("revoked_bundle_path_unsafe")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("revoked_bundle_file_invalid")
    with path.open("rb") as stream:
        digest = _sha(stream)
    after = path.stat()
    if (info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("revoked_bundle_changed_during_read")
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": info.st_size,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
    }


def _not_open(path: Path) -> None:
    proc = Path("/proc")
    if not proc.is_dir():
        result = subprocess.run(
            ["lsof", "-F", "n", "--", str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode != 1 or result.stdout or result.stderr:
            raise ValueError("revoked_bundle_open_reader_or_inspection_unavailable")
        return
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError("revoked_bundle_open_reader_inspection_denied") from exc
        for fd in descriptors:
            try:
                if os.readlink(fd) == str(path):
                    raise ValueError("revoked_bundle_open_reader")
            except FileNotFoundError:
                continue


def _bundle_path(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    if len(relative.parts) != 4 or relative.parts[1:] != ("launch-set", "bundle", BUNDLE_NAME):
        raise ValueError("revoked_bundle_not_input_packaging")


def _members(archive):
    rows = [row for row in archive.infolist() if not row.is_dir()]
    if len(rows) != len({row.filename for row in rows}):
        raise ValueError("revoked_bundle_duplicate_member")
    for row in rows:
        p = PurePosixPath(row.filename)
        if (
            p.is_absolute()
            or ".." in p.parts
            or "\\" in row.filename
            or stat.S_ISLNK(row.external_attr >> 16)
        ):
            raise ValueError("revoked_bundle_member_unsafe")
    return rows


def plan_retention(
    *,
    obsolete: Path,
    retained: Path,
    activation_root: Path,
    revoked_intent_root: Path,
    successor_intent_root: Path,
    terminal_launch_root: Path,
    destination: Path,
) -> dict[str, Any]:
    """Read-only payload inspection; only writes the explicit durable plan."""
    if obsolete == retained or destination.exists():
        raise ValueError("revoked_bundle_retention_destination_invalid")
    _bundle_path(obsolete, activation_root)
    _bundle_path(retained, activation_root)
    intent = _read(revoked_intent_root / "intent.json", "intent_digest")
    successor = _read(successor_intent_root / "intent.json", "intent_digest")
    revoked = _read(revoked_intent_root / "revoked.json", "receipt_digest")
    if (
        revoked.get("status") != "revoked"
        or revoked.get("intent_digest") != intent["intent_digest"]
        or revoked.get("owner") != intent["request"]["owner"]
        or any(intent["request"][k] != successor["request"][k] for k in ("owner", "source", "task"))
        or not obsolete.parent.parent.parent.name.startswith(intent["intent_id"][:26])
        or not terminal_launch_root.name.startswith(intent["intent_id"][:26])
    ):
        raise ValueError("revoked_bundle_owner_or_scope_invalid")
    terminal_path = terminal_launch_root / "launch_receipt.json"
    zero_path = terminal_launch_root / "post_teardown_provider_zero_receipt.json"
    terminal = json.loads(terminal_path.read_text())
    zero = json.loads(zero_path.read_text())
    if (
        terminal.get("status") not in {"completed", "blocked"}
        or zero.get("status") != "provider_zero_confirmed"
        or zero.get("blockers") != []
    ):
        raise ValueError("revoked_bundle_attempt_not_closed")
    records = {
        "obsolete": _record(obsolete),
        "retained": _record(retained),
        "intent": _record(revoked_intent_root / "intent.json"),
        "successor_intent": _record(successor_intent_root / "intent.json"),
        "revocation": _record(revoked_intent_root / "revoked.json"),
        "terminal": _record(terminal_path),
        "provider_zero": _record(zero_path),
    }
    _not_open(obsolete)
    members = []
    unique_bytes = 0
    with zipfile.ZipFile(obsolete) as old, zipfile.ZipFile(retained) as current:
        current_rows = {row.filename: row for row in _members(current)}
        for row in _members(old):
            with old.open(row) as stream:
                digest = _sha(stream)
            shared = False
            if (
                row.filename in current_rows
                and current_rows[row.filename].file_size == row.file_size
            ):
                with current.open(row.filename) as stream:
                    shared = _sha(stream) == digest
            member = {
                "name": row.filename,
                "size_bytes": row.file_size,
                "sha256": digest,
                "retained_identical_member": shared,
            }
            if not shared:
                unique_bytes += row.file_size
                if row.file_size > MAX_UNIQUE_MEMBER_BYTES or unique_bytes > MAX_UNIQUE_BYTES:
                    raise ValueError("revoked_bundle_unique_payload_too_large")
                member["preserved_base64"] = base64.b64encode(old.read(row)).decode("ascii")
            members.append(member)
    for record in records.values():
        if _record(Path(record["path"])) != record:
            raise ValueError("revoked_bundle_input_changed")
    result = {
        "schema_version": SCHEMA,
        "status": "planned",
        "activation_root": str(activation_root),
        "records": records,
        "members": members,
        "unique_member_bytes": unique_bytes,
        "retained_archive_must_remain_pinned": True,
        "original_member_bytes_recoverable": True,
        "original_zip_container_identity_recoverable": False,
        "source_observations_or_model_outputs_removed": False,
    }
    result["plan_digest"] = canonical_digest(result, digest_field="plan_digest")
    with destination.open("x") as stream:
        stream.write(canonical_json(result) + "\n")
    return result


def apply_retention(*, plan_path: Path, receipt_path: Path, acknowledgement: str) -> dict[str, Any]:
    if (
        acknowledgement != ACK
        or receipt_path.exists()
        or receipt_path.is_symlink()
        or not receipt_path.parent.is_dir()
        or any(p.is_symlink() for p in receipt_path.parents)
    ):
        raise ValueError("revoked_bundle_retention_ack_required")
    _record(plan_path)
    plan = json.loads(plan_path.read_text())
    if plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest"):
        raise ValueError("revoked_bundle_plan_digest_invalid")
    if plan.get("schema_version") != SCHEMA or plan.get("status") != "planned":
        raise ValueError("revoked_bundle_plan_invalid")
    for record in plan["records"].values():
        if _record(Path(record["path"])) != record:
            raise ValueError("revoked_bundle_input_changed")
    for member in plan["members"]:
        if not member["retained_identical_member"]:
            data = base64.b64decode(member["preserved_base64"], validate=True)
            if (
                len(data) != member["size_bytes"]
                or "sha256:" + hashlib.sha256(data).hexdigest() != member["sha256"]
            ):
                raise ValueError("revoked_bundle_preserved_member_invalid")
    obsolete = Path(plan["records"]["obsolete"]["path"])
    _bundle_path(obsolete, Path(plan["activation_root"]))
    if obsolete == Path(plan["records"]["retained"]["path"]):
        raise ValueError("revoked_bundle_cannot_reclaim_retained_archive")
    _not_open(obsolete)
    obsolete.unlink()
    result = {
        "schema_version": SCHEMA,
        "status": "reclaimed",
        "plan_digest": plan["plan_digest"],
        "removed": plan["records"]["obsolete"],
        "retained": plan["records"]["retained"],
        "source_observations_or_model_outputs_removed": False,
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    with receipt_path.open("x") as stream:
        stream.write(canonical_json(result) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    plan = commands.add_parser("plan")
    for name in (
        "obsolete",
        "retained",
        "activation_root",
        "revoked_intent_root",
        "successor_intent_root",
        "terminal_launch_root",
        "destination",
    ):
        plan.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    apply = commands.add_parser("apply")
    for name in ("plan_path", "receipt_path"):
        apply.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    apply.add_argument("--acknowledgement", required=True, choices=[ACK])
    args = vars(parser.parse_args())
    operation = args.pop("operation")
    result = plan_retention(**args) if operation == "plan" else apply_retention(**args)
    print(canonical_json({key: result[key] for key in ("schema_version", "status")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
