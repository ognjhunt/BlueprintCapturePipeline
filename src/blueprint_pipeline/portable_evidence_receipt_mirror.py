"""Mirror digest-bound evidence receipts without copying raw evidence bytes.

Portable ADP packages need to remain reviewable when the authoritative evidence
root contains rights-bounded dataset-derived bytes.  This seam deliberately
copies only a caller-selected set of canonical JSON receipts.  It never copies
raw splats, frames, masks, images, media, USD, or arbitrary files.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest


RECEIPT_MIRROR_SCHEMA_VERSION = "adp_portable_evidence_receipt_mirror.v1"
RECEIPT_MIRROR_DIGEST_FIELD = "receipt_mirror_digest"


class PortableEvidenceReceiptMirrorError(ValueError):
    """Raised when a receipt-only mirror cannot be safely materialized."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_json_unreadable:{role}"
        ) from exc
    if not isinstance(value, dict):
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_json_not_mapping:{role}"
        )
    return value


def _inside(root: Path, relative_path: str, *, role: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_path_invalid:{role}"
        )
    unresolved = root / relative_path
    if unresolved.is_symlink():
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_symlink_forbidden:{role}"
        )
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_path_outside_root:{role}"
        ) from exc
    return resolved


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _receipt_row(
    *,
    source_root: Path,
    relative_path: str,
    admitted_schema_digest_fields: Mapping[str, str],
) -> tuple[dict[str, Any], bytes]:
    if not relative_path.endswith(".json"):
        raise PortableEvidenceReceiptMirrorError(
            "portable_receipt_mirror_non_json_forbidden"
        )
    path = _inside(source_root, relative_path, role=relative_path)
    if path.is_symlink() or not path.is_file():
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_receipt_missing:{relative_path}"
        )
    payload = _load_json(path, role=relative_path)
    schema_version = str(payload.get("schema_version") or "")
    digest_field = admitted_schema_digest_fields.get(schema_version)
    if not digest_field:
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_schema_not_admitted:{schema_version or 'missing'}"
        )
    receipt_digest = str(payload.get(digest_field) or "")
    if receipt_digest != canonical_digest(payload, digest_field=digest_field):
        raise PortableEvidenceReceiptMirrorError(
            f"portable_receipt_mirror_digest_invalid:{relative_path}"
        )
    content = path.read_bytes()
    return (
        {
            "relative_path": relative_path,
            "schema_version": schema_version,
            "digest_field": digest_field,
            "receipt_digest": receipt_digest,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        },
        content,
    )


def materialize_portable_evidence_receipt_mirror(
    *,
    source_root: str | Path,
    output_root: str | Path,
    source_root_id: str,
    receipt_relative_paths: Sequence[str],
    admitted_schema_digest_fields: Mapping[str, str],
    output_relative_path: str,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Copy canonical receipts and seal their exact receipt-only inventory.

    ``source_root_id`` is an opaque disclosure-safe identity; it does not make a
    source path or its dataset bytes public.  Callers must pass the explicit
    allowlist of schemas and receipt paths they want mirrored.
    """

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_dir() or not output.is_dir() or not source_root_id:
        raise PortableEvidenceReceiptMirrorError("portable_receipt_mirror_root_invalid")
    if not receipt_relative_paths or not admitted_schema_digest_fields:
        raise PortableEvidenceReceiptMirrorError("portable_receipt_mirror_input_missing")
    if len(set(receipt_relative_paths)) != len(receipt_relative_paths):
        raise PortableEvidenceReceiptMirrorError("portable_receipt_mirror_duplicate_path")
    if not output_relative_path.endswith(".json"):
        raise PortableEvidenceReceiptMirrorError(
            "portable_receipt_mirror_output_path_invalid"
        )

    rows_and_content = [
        _receipt_row(
            source_root=source,
            relative_path=relative_path,
            admitted_schema_digest_fields=admitted_schema_digest_fields,
        )
        for relative_path in receipt_relative_paths
    ]
    output_receipt_paths: list[tuple[Path, bytes]] = []
    for row, content in rows_and_content:
        destination = _inside(output, row["relative_path"], role=row["relative_path"])
        if destination.is_symlink():
            raise PortableEvidenceReceiptMirrorError(
                f"portable_receipt_mirror_output_symlink:{row['relative_path']}"
            )
        if destination.exists() and not replace_existing:
            raise PortableEvidenceReceiptMirrorError(
                f"portable_receipt_mirror_output_exists:{row['relative_path']}"
            )
        output_receipt_paths.append((destination, content))

    manifest_path = _inside(output, output_relative_path, role="manifest")
    if manifest_path.is_symlink():
        raise PortableEvidenceReceiptMirrorError(
            "portable_receipt_mirror_manifest_symlink"
        )
    if manifest_path.exists() and not replace_existing:
        raise PortableEvidenceReceiptMirrorError("portable_receipt_mirror_manifest_exists")

    for destination, content in output_receipt_paths:
        _atomic_write_bytes(destination, content)

    rows = sorted((row for row, _ in rows_and_content), key=lambda row: row["relative_path"])
    manifest: dict[str, Any] = {
        "schema_version": RECEIPT_MIRROR_SCHEMA_VERSION,
        "status": "receipt_only_mirror_materialized",
        "source_root_id": source_root_id,
        "receipt_count": len(rows),
        "receipts": rows,
        "raw_dataset_bytes_copied": False,
        "scene_media_copied": False,
        "claim_ceiling": (
            "digest_bound_receipt_mirror_only; no source-removal, simulator, "
            "physical, or publication-rights claim"
        ),
        RECEIPT_MIRROR_DIGEST_FIELD: "",
    }
    manifest[RECEIPT_MIRROR_DIGEST_FIELD] = canonical_digest(
        manifest, digest_field=RECEIPT_MIRROR_DIGEST_FIELD
    )
    _atomic_write_bytes(
        manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return manifest


__all__ = [
    "RECEIPT_MIRROR_DIGEST_FIELD",
    "RECEIPT_MIRROR_SCHEMA_VERSION",
    "PortableEvidenceReceiptMirrorError",
    "materialize_portable_evidence_receipt_mirror",
]
