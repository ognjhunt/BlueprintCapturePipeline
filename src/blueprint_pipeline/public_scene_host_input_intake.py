"""Install rights-bound public-scene source bytes on the production host.

``upload`` streams a deterministic packet over SSH; it never records the
client paths and never uses scp.  ``install`` is the production-host half of
that protocol.  ``stage`` exercises the same installer when source bytes are
already on the host.  No command in this module allocates a provider or grants
rights: an existing human-issued rights receipt is a required input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


REQUEST_SCHEMA = "public_scene_host_input_request.v1"
PACKET_SCHEMA = "public_scene_host_input_packet.v1"
RECEIPT_SCHEMA = "public_scene_host_input_installation_receipt.v1"
RIGHTS_RECEIPT_SCHEMA = "public_scene_rights_authority.v1"
DEFAULT_DESTINATION_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs")
PRODUCTION_ROOTS = (DEFAULT_DESTINATION_ROOT,)
DEFAULT_REMOTE_PYTHON = Path(
    "/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python"
)
DEFAULT_SERVICE_ACCOUNT = "blueprint"
MAX_ARCHIVE_MEMBERS = 16
MAX_ARCHIVE_INPUT_BYTES = 2 * 1024**3
MAX_ARCHIVE_COMPRESSED_BYTES = 2 * 1024**3
MAX_ARCHIVE_MEMBER_BYTES = 256 * 1024**2
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 8 * 1024**3
MAX_ARCHIVE_COMPRESSION_RATIO = 100.0
SERVICE_READBACK_TIMEOUT_SECONDS = 30
SSH_CONNECT_TIMEOUT_SECONDS = 30
HOST_UPLOAD_TIMEOUT_SECONDS = 1800
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,254}\Z")
_SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_APPROVED_RIGHTS_STATES = {
    "accepted_for_declared_local_import_only",
    "approved_for_declared_use",
    "approved_for_internal_use",
}


class PublicSceneHostInputError(ValueError):
    """The requested intake is not immutable, rights-bound, or contained."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicSceneHostInputError(f"input_json_invalid:{path.name}") from exc
    if not isinstance(value, dict):
        raise PublicSceneHostInputError(f"input_json_not_object:{path.name}")
    return value


def _safe_id(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _SAFE_ID.fullmatch(text):
        raise PublicSceneHostInputError(f"{field}_invalid")
    return text


def _expected_digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _DIGEST.fullmatch(text):
        raise PublicSceneHostInputError(f"{field}_invalid")
    return text


def _commit(value: Any) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{40}", text):
        raise PublicSceneHostInputError("source_commit_sha_invalid")
    return text


def _verified_checkout_head() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(  # nosec B603 - absolute executable and fixed argv
            ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=SERVICE_READBACK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicSceneHostInputError("checkout_head_verification_failed") from exc
    if result.returncode != 0:
        raise PublicSceneHostInputError("checkout_head_verification_failed")
    try:
        return _commit(result.stdout.strip())
    except PublicSceneHostInputError as exc:
        raise PublicSceneHostInputError("checkout_head_verification_failed") from exc


def _under(path: Path, roots: Sequence[Path]) -> Path:
    resolved = path.expanduser().resolve()
    allowed = tuple(root.expanduser().resolve() for root in roots)
    if not any(resolved == root or root in resolved.parents for root in allowed):
        raise PublicSceneHostInputError(f"destination_outside_production_root:{resolved}")
    return resolved


def _assert_secret_free(content: bytes, *, name: str) -> None:
    if any(pattern.search(content) for pattern in _SECRET_PATTERNS):
        raise PublicSceneHostInputError(f"secret_like_source_rejected:{name}")


def _rights_state(receipt: Mapping[str, Any]) -> str:
    return str(receipt.get("status") or receipt.get("reviewer_status") or "")


def _preflight_local_members(
    raw_rights: list[Any], raw_files: list[Any]
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if 1 + len(raw_rights) + len(raw_files) > MAX_ARCHIVE_MEMBERS:
        raise PublicSceneHostInputError("local_input_member_count_exceeds_limit")
    observed: list[tuple[Path, int]] = []
    aggregate_bytes = 0
    for raw in [*raw_rights, *raw_files]:
        if not isinstance(raw, Mapping):
            raise PublicSceneHostInputError("local_input_member_record_invalid")
        path = Path(str(raw.get("path") or "")).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise PublicSceneHostInputError("local_input_member_missing_or_unsafe")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise PublicSceneHostInputError("local_input_member_stat_failed") from exc
        if size <= 0 or size > MAX_ARCHIVE_MEMBER_BYTES:
            raise PublicSceneHostInputError("local_input_member_size_exceeds_limit")
        aggregate_bytes += size
        if aggregate_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise PublicSceneHostInputError("local_input_aggregate_size_exceeds_limit")
        observed.append((path, size))
    rights_count = len(raw_rights)
    return observed[:rights_count], observed[rights_count:]


def _load_request(request_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_file = Path(request_path).expanduser().resolve()
    request = _json(request_file)
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise PublicSceneHostInputError("host_input_request_schema_invalid")
    forbidden = {"status", "installed", "service_readable"}.intersection(request)
    if forbidden:
        raise PublicSceneHostInputError("caller_asserted_installation_forbidden")
    scene_id = _safe_id(request.get("scene_id"), field="scene_id")
    packet_id = _safe_id(request.get("packet_id"), field="packet_id")

    raw_rights = request.get("rights_receipts")
    if not isinstance(raw_rights, list) or not raw_rights:
        raise PublicSceneHostInputError("rights_receipts_missing")
    raw_files = request.get("files")
    if not isinstance(raw_files, list):
        raise PublicSceneHostInputError("source_files_missing")
    rights_members, source_members = _preflight_local_members(raw_rights, raw_files)
    rights: dict[str, dict[str, Any]] = {}
    packed: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rights):
        if not isinstance(raw, Mapping):
            raise PublicSceneHostInputError("rights_receipt_record_invalid")
        receipt_id = _safe_id(raw.get("receipt_id"), field="rights_receipt_id")
        if receipt_id in rights:
            raise PublicSceneHostInputError("rights_receipt_id_duplicate")
        path, size = rights_members[index]
        expected = _expected_digest(
            raw.get("sha256"), field=f"rights_receipts_{index}_sha256"
        )
        if _sha256_file(path) != expected:
            raise PublicSceneHostInputError(f"rights_receipt_bytes_mismatch:{receipt_id}")
        value = _json(path)
        if value.get("schema_version") != RIGHTS_RECEIPT_SCHEMA:
            raise PublicSceneHostInputError(f"rights_receipt_schema_invalid:{receipt_id}")
        if _rights_state(value) not in _APPROVED_RIGHTS_STATES:
            raise PublicSceneHostInputError(f"rights_receipt_not_approved:{receipt_id}")
        if value.get("agent_accepted_terms") is True:
            raise PublicSceneHostInputError(f"rights_receipt_agent_acceptance_forbidden:{receipt_id}")
        content = path.read_bytes()
        _assert_secret_free(content, name=path.name)
        authorized = value.get("authorized_source_sha256")
        if (
            not isinstance(authorized, list)
            or not authorized
            or any(not _DIGEST.fullmatch(str(digest)) for digest in authorized)
        ):
            raise PublicSceneHostInputError(
                f"rights_receipt_source_digests_invalid:{receipt_id}"
            )
        rights[receipt_id] = {
            "sha256": expected,
            "value": value,
            "authorized_source_sha256": set(str(digest) for digest in authorized),
        }
        packed.append(
            {
                "kind": "rights_receipt",
                "receipt_id": receipt_id,
                "source": path,
                "archive_path": f"rights/{index:02d}-{path.name}",
                "relative_path": f"rights/{index:02d}-{path.name}",
                "sha256": expected,
                "size_bytes": size,
            }
        )

    core_counts = {"collision_usd": 0, "shared_frame_registration": 0}
    support_count = 0
    destination_names: set[str] = set()
    for index, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise PublicSceneHostInputError("source_file_record_invalid")
        role = str(raw.get("role") or "")
        if role in core_counts:
            core_counts[role] += 1
        elif role == "task_support":
            support_count += 1
            _safe_id(raw.get("task_id"), field="task_id")
        else:
            raise PublicSceneHostInputError(f"source_file_role_invalid:{role}")
        path, size = source_members[index]
        expected = _expected_digest(raw.get("sha256"), field=f"files_{index}_sha256")
        if _sha256_file(path) != expected:
            raise PublicSceneHostInputError(f"source_file_bytes_mismatch:{role}")
        if role == "collision_usd" and path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise PublicSceneHostInputError("collision_usd_extension_invalid")
        if role == "shared_frame_registration" and path.suffix.lower() != ".json":
            raise PublicSceneHostInputError("shared_frame_registration_extension_invalid")
        rights_ids = raw.get("rights_receipt_ids")
        if (
            not isinstance(rights_ids, list)
            or not rights_ids
            or any(str(value) not in rights for value in rights_ids)
        ):
            raise PublicSceneHostInputError(f"source_file_rights_binding_invalid:{role}")
        if not any(
            expected in rights[str(receipt_id)]["authorized_source_sha256"]
            for receipt_id in rights_ids
        ):
            raise PublicSceneHostInputError(f"source_file_not_rights_authorized:{role}")
        destination_name = str(raw.get("destination_name") or path.name)
        if Path(destination_name).name != destination_name or not destination_name:
            raise PublicSceneHostInputError("source_destination_name_invalid")
        if destination_name in destination_names:
            raise PublicSceneHostInputError("source_destination_name_duplicate")
        destination_names.add(destination_name)
        content = path.read_bytes()
        _assert_secret_free(content, name=destination_name)
        relative = f"inputs/{destination_name}"
        packed.append(
            {
                "kind": "source_file",
                "role": role,
                "task_id": str(raw.get("task_id") or "") or None,
                "rights_receipt_ids": sorted(str(value) for value in rights_ids),
                "source": path,
                "archive_path": relative,
                "relative_path": relative,
                "sha256": expected,
                "size_bytes": size,
            }
        )
    if core_counts != {"collision_usd": 1, "shared_frame_registration": 1}:
        raise PublicSceneHostInputError("required_scene_source_files_invalid")
    if support_count > 5:
        raise PublicSceneHostInputError("task_support_file_count_exceeds_five")

    source_commit = _commit(request.get("source_commit_sha"))
    if source_commit != _verified_checkout_head():
        raise PublicSceneHostInputError("source_commit_sha_mismatch")
    metadata = {
        "schema_version": PACKET_SCHEMA,
        "program_id": "arm-decision-proof-v1",
        "adp_item": str(request.get("adp_item") or "ADP-009B"),
        "scene_id": scene_id,
        "packet_id": packet_id,
        "source_commit_sha": source_commit,
        "rights_receipts": [
            {
                key: value
                for key, value in row.items()
                if key in {"receipt_id", "relative_path", "sha256", "size_bytes"}
            }
            for row in packed
            if row["kind"] == "rights_receipt"
        ],
        "files": [
            {
                key: value
                for key, value in row.items()
                if key
                in {
                    "role",
                    "task_id",
                    "rights_receipt_ids",
                    "relative_path",
                    "sha256",
                    "size_bytes",
                }
            }
            for row in packed
            if row["kind"] == "source_file"
        ],
        "claim_ceiling": "rights_bound_public_scene_source_bytes_only",
        "provider_mutation_performed": False,
        "paid_resource_used": False,
    }
    metadata["packet_digest"] = canonical_digest(metadata, digest_field="packet_digest")
    return metadata, packed


def _zip_entry(name: str) -> zipfile.ZipInfo:
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_DEFLATED
    entry.external_attr = 0o100440 << 16
    return entry


def build_packet_archive(request_path: str | Path, output: BinaryIO) -> dict[str, Any]:
    """Write deterministic source bytes and their rights-bound manifest."""

    metadata, rows = _load_request(request_path)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        manifest = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
        archive.writestr(_zip_entry("packet.json"), manifest)
        for row in rows:
            with archive.open(_zip_entry(str(row["archive_path"])), "w") as sink:
                with Path(row["source"]).open("rb") as source:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
    return metadata


def _service_identity(account: str | None) -> tuple[str, int, int]:
    if account is None:
        current = pwd.getpwuid(os.getuid())
        return current.pw_name, current.pw_uid, current.pw_gid
    try:
        value = pwd.getpwnam(account)
    except KeyError as exc:
        raise PublicSceneHostInputError(f"service_account_missing:{account}") from exc
    return account, value.pw_uid, value.pw_gid


def _consumer_digest(path: Path, *, account: str, uid: int) -> str:
    if uid == os.getuid():
        return _sha256_file(path)
    try:
        result = subprocess.run(  # nosec B603 - absolute executable and fixed argv
            ["/usr/sbin/runuser", "-u", account, "--", "/usr/bin/sha256sum", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=SERVICE_READBACK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicSceneHostInputError(f"service_readback_failed:{path.name}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise PublicSceneHostInputError(f"service_readback_failed:{path.name}")
    return "sha256:" + result.stdout.split()[0]


def _validated_archive(archive: zipfile.ZipFile) -> dict[str, Any]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if (
        not names
        or len(names) > MAX_ARCHIVE_MEMBERS
        or len(names) != len(set(names))
        or "packet.json" not in names
    ):
        raise PublicSceneHostInputError("packet_archive_members_invalid")
    uncompressed_bytes = 0
    compressed_bytes = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = (info.external_attr >> 16) & 0xFFFF
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
            or info.is_dir()
            or not stat.S_ISREG(mode)
            or info.flag_bits & 0x1
            or info.compress_type not in {zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED}
        ):
            raise PublicSceneHostInputError("packet_archive_member_unsafe")
        if info.file_size <= 0 or info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise PublicSceneHostInputError("packet_archive_member_size_exceeds_limit")
        if info.compress_size <= 0:
            raise PublicSceneHostInputError("packet_archive_compressed_size_invalid")
        uncompressed_bytes += info.file_size
        compressed_bytes += info.compress_size
        if uncompressed_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise PublicSceneHostInputError(
                "packet_archive_uncompressed_size_exceeds_limit"
            )
        if compressed_bytes > MAX_ARCHIVE_COMPRESSED_BYTES:
            raise PublicSceneHostInputError("packet_archive_compressed_size_exceeds_limit")
        if info.file_size / info.compress_size > MAX_ARCHIVE_COMPRESSION_RATIO:
            raise PublicSceneHostInputError("packet_archive_compression_ratio_exceeds_limit")
    try:
        packet = json.loads(archive.read("packet.json"))
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicSceneHostInputError("packet_manifest_invalid") from exc
    if not isinstance(packet, dict) or packet.get("schema_version") != PACKET_SCHEMA:
        raise PublicSceneHostInputError("packet_manifest_schema_invalid")
    if packet.get("packet_digest") != canonical_digest(packet, digest_field="packet_digest"):
        raise PublicSceneHostInputError("packet_manifest_digest_invalid")
    _safe_id(packet.get("scene_id"), field="scene_id")
    _safe_id(packet.get("packet_id"), field="packet_id")
    source_commit = _commit(packet.get("source_commit_sha"))
    if source_commit != _verified_checkout_head():
        raise PublicSceneHostInputError("source_commit_sha_mismatch")
    expected = {"packet.json"}
    rights_digests: dict[str, set[str]] = {}
    rights_rows = packet.get("rights_receipts")
    if not isinstance(rights_rows, list) or not rights_rows:
        raise PublicSceneHostInputError("packet_manifest_inventory_invalid")
    for row in rights_rows:
        if not isinstance(row, Mapping):
            raise PublicSceneHostInputError("packet_manifest_inventory_invalid")
        receipt_id = _safe_id(row.get("receipt_id"), field="rights_receipt_id")
        relative = str(row.get("relative_path") or "")
        expected.add(relative)
        digest = _expected_digest(row.get("sha256"), field="packet_inventory_sha256")
        content = archive.read(relative)
        if len(content) != row.get("size_bytes") or _sha256_bytes(content) != digest:
            raise PublicSceneHostInputError("packet_archive_rights_bytes_mismatch")
        _assert_secret_free(content, name=PurePosixPath(relative).name)
        try:
            receipt = json.loads(content)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PublicSceneHostInputError("packet_archive_rights_invalid") from exc
        authorized = receipt.get("authorized_source_sha256") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != RIGHTS_RECEIPT_SCHEMA
            or _rights_state(receipt) not in _APPROVED_RIGHTS_STATES
            or receipt.get("agent_accepted_terms") is True
            or not isinstance(authorized, list)
            or not authorized
            or any(not _DIGEST.fullmatch(str(value)) for value in authorized)
        ):
            raise PublicSceneHostInputError("packet_archive_rights_invalid")
        rights_digests[receipt_id] = set(str(value) for value in authorized)

    file_rows = packet.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise PublicSceneHostInputError("packet_manifest_inventory_invalid")
    core_counts = {"collision_usd": 0, "shared_frame_registration": 0}
    support_count = 0
    for row in file_rows:
        if not isinstance(row, Mapping):
            raise PublicSceneHostInputError("packet_manifest_inventory_invalid")
        role = str(row.get("role") or "")
        if role in core_counts:
            core_counts[role] += 1
        elif role == "task_support":
            support_count += 1
            _safe_id(row.get("task_id"), field="task_id")
        else:
            raise PublicSceneHostInputError(f"source_file_role_invalid:{role}")
        relative = str(row.get("relative_path") or "")
        expected.add(relative)
        digest = _expected_digest(row.get("sha256"), field="packet_inventory_sha256")
        rights_ids = row.get("rights_receipt_ids")
        if (
            not isinstance(rights_ids, list)
            or not rights_ids
            or not any(digest in rights_digests.get(str(value), set()) for value in rights_ids)
        ):
            raise PublicSceneHostInputError(f"source_file_not_rights_authorized:{role}")
        info = archive.getinfo(relative)
        if info.file_size != row.get("size_bytes"):
            raise PublicSceneHostInputError("packet_archive_source_size_mismatch")
        _assert_secret_free(archive.read(relative), name=PurePosixPath(relative).name)
    if core_counts != {"collision_usd": 1, "shared_frame_registration": 1}:
        raise PublicSceneHostInputError("required_scene_source_files_invalid")
    if support_count > 5:
        raise PublicSceneHostInputError("task_support_file_count_exceeds_five")
    if set(names) != expected:
        raise PublicSceneHostInputError("packet_archive_inventory_mismatch")
    return packet


def install_packet_archive(
    archive_source: BinaryIO,
    *,
    destination_root: str | Path = DEFAULT_DESTINATION_ROOT,
    service_account: str | None = DEFAULT_SERVICE_ACCOUNT,
    allowed_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Atomically install one packet and prove consumer-readable digests."""

    try:
        offset = archive_source.tell()
        archive_source.seek(0, 2)
        archive_input_bytes = archive_source.tell()
        archive_source.seek(offset)
    except (OSError, AttributeError) as exc:
        raise PublicSceneHostInputError("packet_archive_input_unseekable") from exc
    if archive_input_bytes <= 0 or archive_input_bytes > MAX_ARCHIVE_INPUT_BYTES:
        raise PublicSceneHostInputError("packet_archive_input_size_exceeds_limit")
    root = _under(Path(destination_root), allowed_roots or PRODUCTION_ROOTS)
    root.mkdir(parents=True, exist_ok=True)
    account, uid, gid = _service_identity(service_account)
    with zipfile.ZipFile(archive_source, "r") as archive:
        packet = _validated_archive(archive)
        packet_id = _safe_id(packet.get("packet_id"), field="packet_id")
        target = _under(root / packet_id, (root,))
        if target.exists():
            raise PublicSceneHostInputError(f"destination_already_exists:{packet_id}")
        staging = _under(root / f".{packet_id}.staging-{uuid.uuid4().hex}", (root,))
        staging.mkdir(mode=0o750)
        try:
            records: list[dict[str, Any]] = []
            inventory = [*packet["rights_receipts"], *packet["files"]]
            for row in inventory:
                relative = PurePosixPath(str(row["relative_path"]))
                destination = staging.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(str(relative), "r") as source, destination.open("xb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                if (
                    destination.stat().st_size != row.get("size_bytes")
                    or _sha256_file(destination) != row.get("sha256")
                ):
                    raise PublicSceneHostInputError(
                        f"installed_bytes_mismatch:{destination.name}"
                    )
                records.append(dict(row))
            receipt: dict[str, Any] = {
                "schema_version": RECEIPT_SCHEMA,
                "status": "installed",
                "program_id": packet["program_id"],
                "adp_item": packet["adp_item"],
                "scene_id": packet["scene_id"],
                "packet_id": packet_id,
                "source_commit_sha": packet["source_commit_sha"],
                "packet_digest": packet["packet_digest"],
                "authoritative_request_digest": packet["packet_digest"],
                "request_identity_source": "verified_packet_manifest",
                "destination_root": str(target),
                "service_account": account,
                "service_readable": True,
                "files": list(records),
                "provider_mutation_performed": False,
                "paid_resource_used": False,
                "secret_patterns_scanned": [
                    "openai_api_key_like",
                    "private_key_pem",
                ],
                "secret_pattern_scan_scope": "bounded_patterns_only",
                "raw_secret_values_recorded_in_receipt": False,
                "blockers": [],
            }
            receipt["receipt_digest"] = canonical_digest(
                receipt, digest_field="receipt_digest"
            )
            receipt_path = staging / "public_scene_host_input_installation_receipt.v1.json"
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            records.append(
                {
                    "relative_path": receipt_path.name,
                    "sha256": _sha256_file(receipt_path),
                    "size_bytes": receipt_path.stat().st_size,
                }
            )
            for directory in sorted(
                [path for path in staging.rglob("*") if path.is_dir()], reverse=True
            ):
                os.chown(directory, uid, gid)
                os.chmod(directory, 0o750)  # nosec B103 - group traverse; files 0440
            for path in [path for path in staging.rglob("*") if path.is_file()]:
                os.chown(path, uid, gid)
                os.chmod(path, 0o440)
            os.chown(staging, uid, gid)
            os.chmod(staging, 0o750)  # nosec B103 - group traverse; files 0440
            for row in records:
                path = staging.joinpath(*PurePosixPath(str(row["relative_path"])).parts)
                if _consumer_digest(path, account=account, uid=uid) != row["sha256"]:
                    raise PublicSceneHostInputError(
                        f"service_readback_digest_mismatch:{path.name}"
                    )
            try:
                target.mkdir(mode=0o700)
            except FileExistsError as exc:
                raise PublicSceneHostInputError(
                    f"destination_already_exists:{packet_id}"
                ) from exc
            try:
                staging.rename(target)
            except Exception:
                target.rmdir()
                raise
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return receipt


def _archive_for_request(
    request: Path,
) -> tuple[tempfile.SpooledTemporaryFile[bytes], dict[str, Any]]:
    temporary = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b")
    packet = build_packet_archive(request, temporary)
    temporary.seek(0)
    return temporary, packet


def _copy_bounded_archive_input(source: BinaryIO, destination: BinaryIO) -> None:
    copied = 0
    while True:
        content = source.read(1024 * 1024)
        if not content:
            return
        copied += len(content)
        if copied > MAX_ARCHIVE_INPUT_BYTES:
            raise PublicSceneHostInputError("packet_archive_input_size_exceeds_limit")
        destination.write(content)


def upload_packet(
    *, request_path: str | Path, host: str, destination_root: str | Path
) -> dict[str, Any]:
    """Stream source bytes to the deployed installer without scp or path aliases."""

    if not _HOST.fullmatch(host) or host.startswith("-"):
        raise PublicSceneHostInputError("upload_host_invalid")
    root = _under(Path(destination_root), PRODUCTION_ROOTS)
    command = shlex.join(
        [
            str(DEFAULT_REMOTE_PYTHON),
            "-m",
            "blueprint_pipeline.public_scene_host_input_intake",
            "install",
            "--archive",
            "-",
            "--destination-root",
            str(root),
            "--service-account",
            DEFAULT_SERVICE_ACCOUNT,
        ]
    )
    archive, packet = _archive_for_request(Path(request_path))
    try:
        with archive:
            result = subprocess.run(  # nosec B603 - fixed executable, validated host
                [
                    "/usr/bin/ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
                    host,
                    command,
                ],
                stdin=archive,
                capture_output=True,
                text=False,
                check=False,
                timeout=HOST_UPLOAD_TIMEOUT_SECONDS,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicSceneHostInputError("host_input_upload_failed:transport") from exc
    if result.returncode != 0:
        raise PublicSceneHostInputError(f"host_input_upload_failed:{result.returncode}")
    try:
        receipt = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PublicSceneHostInputError("host_input_upload_receipt_invalid") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("status") != "installed"
    ):
        raise PublicSceneHostInputError("host_input_upload_receipt_invalid")
    expected = {
        "packet_digest": packet["packet_digest"],
        "authoritative_request_digest": packet["packet_digest"],
        "scene_id": packet["scene_id"],
        "packet_id": packet["packet_id"],
        "source_commit_sha": packet["source_commit_sha"],
        "destination_root": str(_under(root / packet["packet_id"], (root,))),
        "service_account": DEFAULT_SERVICE_ACCOUNT,
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise PublicSceneHostInputError("host_input_upload_receipt_binding_mismatch")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="install bytes already accessible locally")
    stage.add_argument("--request", type=Path, required=True)
    stage.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    stage.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    upload = commands.add_parser("upload", help="stream client bytes to the production host")
    upload.add_argument("--request", type=Path, required=True)
    upload.add_argument("--host", required=True)
    upload.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    install = commands.add_parser("install", help="install a streamed immutable packet")
    install.add_argument("--archive", required=True)
    install.add_argument("--destination-root", type=Path, default=DEFAULT_DESTINATION_ROOT)
    install.add_argument("--service-account", default=DEFAULT_SERVICE_ACCOUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "upload":
        receipt = upload_packet(
            request_path=args.request,
            host=args.host,
            destination_root=args.destination_root,
        )
    elif args.command == "stage":
        archive, _packet = _archive_for_request(args.request)
        with archive:
            receipt = install_packet_archive(
                archive,
                destination_root=args.destination_root,
                service_account=args.service_account,
            )
    else:
        if args.archive == "-":
            with tempfile.SpooledTemporaryFile(
                max_size=16 * 1024 * 1024, mode="w+b"
            ) as stream:
                _copy_bounded_archive_input(sys.stdin.buffer, stream)
                stream.seek(0)
                receipt = install_packet_archive(
                    stream,
                    destination_root=args.destination_root,
                    service_account=args.service_account,
                )
        else:
            with Path(args.archive).open("rb") as stream:
                receipt = install_packet_archive(
                    stream,
                    destination_root=args.destination_root,
                    service_account=args.service_account,
                )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
