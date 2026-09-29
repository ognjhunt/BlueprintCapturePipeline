"""Promotion receipts, the cleanup gate's deletion rule, and staged-object absence proofs.

Pure records: nothing here reads or writes object storage.

Promotion receipt. ``<staging>/provider_output_promotion.v1.json`` (schema
``wam_provider_output_promotion.v1``) is written by
``provider_output_promotion`` and bound to the staging manifest by
``staging_manifest_sha256``. For each staged key it may delete, the receipt
carries ``staged_objects[role]`` = {``key_sha256``, ``state``, ``versions``}:
``role`` is ``output`` or ``paired_witness``; ``state`` is durable
(``promoted``, ``redundant_with_promoted_output``) or not (``absent_confirmed``,
``deferred``, ``failed``); each version is the {``size_bytes``, ``etag``} of
one staged object promotion made durable or proved redundant. A receipt bound
to an earlier version of the same manifest (a rewrite that kept the staged
keys) is carried forward only through ``rebindable_promotion_receipt``.

Deletion rule (``staged_deletion_allowed``). The cleanup gate HEADs a present
output or witness and deletes it only when a receipt bound to this manifest
has a durable state for that exact key and a version whose (size, ETag) is the
object's. An object whose identity is not recorded -- a late upload after
absence was confirmed, a re-upload after promotion, an object seen only by SSH
recovery -- is never deleted: ``staged_output_promotion_receipt_missing`` or
``staged_output_promotion_identity_mismatch``. An absent object needs no receipt.

Absence proof. ``<staging>/staged_object_absence_proof.v1.json`` (schema
``wam_provider_staged_object_absence_proof.v1``) records that every object
the staging manifest names was proven absent by a completed cleanup, in the
pattern of the operator continuation's sealed ``staged_object_cleanup.json``:
it binds the manifest's sha256, lists the exact keys by role and key hash,
embeds the cleanup receipt it was built from (with its canonical digest) and
names the promotion receipt in force. It is written once and never replaced,
so a resume after a deferred cleanup can show closeout and billing that the
staged objects are gone without rewriting the sealed lane result.
``validate_staged_object_absence_proof`` re-derives every binding.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_paired_witness_staging import SUFFIX as PAIRED_WITNESS_SUFFIX

# The staging manifest's name, as wam_provider_object_store writes it; a test
# keeps the two identical (this module imports nothing from the store).
STAGING_MANIFEST_FILENAME = "wam_provider_object_store_staging_manifest.json"
RECEIPT_FILENAME = "provider_output_promotion.v1.json"
RECEIPT_SCHEMA = "wam_provider_output_promotion.v1"
ABSENCE_PROOF_FILENAME = "staged_object_absence_proof.v1.json"
ABSENCE_PROOF_SCHEMA = "wam_provider_staged_object_absence_proof.v1"
CLEANUP_SCHEMA = "wam_provider_object_store_cleanup.v1"
ROLES = ("output", "paired_witness")
DURABLE_STATES = frozenset({"promoted", "redundant_with_promoted_output"})
RECEIPT_STATUSES = frozenset({"promoted", "absent_confirmed", "failed"})
RECEIPT_MISSING = "staged_output_promotion_receipt_missing"
IDENTITY_MISMATCH = "staged_output_promotion_identity_mismatch"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ProviderOutputPromotionRecordError(ValueError):
    """A typed, secret-free refusal; the message is the stable code."""


def key_sha256(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()


def normalized_etag(value: Any) -> str:
    """An ETag without surrounding quotes or a weak prefix, for comparison."""
    text = str(value or "").strip()
    return text.removeprefix("W/").strip('"')


def _digest_or_none(value: Mapping, field: str | None = None) -> str | None:
    try:
        return canonical_digest(value, digest_field=field)
    except (TypeError, ValueError, RecursionError):
        return None


def _regular_json(path: Path) -> dict | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def staging_manifest_sha256(staging_dir: str | Path) -> str | None:
    """The manifest file's sha256 (hex, as the cleanup receipt records it)."""
    path = Path(staging_dir) / STAGING_MANIFEST_FILENAME
    if path.is_symlink() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, value: Mapping) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_promotion_receipt(staging_dir: str | Path, receipt: Mapping) -> dict:
    """Seal ``receipt`` with ``receipt_digest`` and replace the staging receipt atomically."""
    value = dict(receipt)
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    _atomic_write(Path(staging_dir) / RECEIPT_FILENAME, value)
    return value


def _section_valid(section: Any) -> bool:
    return (isinstance(section, Mapping) and _HEX64.fullmatch(str(section.get("key_sha256")))
            and isinstance(section.get("state"), str) and isinstance(section.get("versions"), list)
            and all(isinstance(version, Mapping) and type(version.get("size_bytes")) is int
                    and version["size_bytes"] >= 0 and isinstance(version.get("etag"), str)
                    and normalized_etag(version["etag"]) for version in section["versions"]))


def load_promotion_receipt(staging_dir: str | Path, *, staging_manifest_sha256: str | None) -> dict | None:
    """The staging dir's promotion receipt, or None unless it is sealed and bound here.

    A receipt that is missing, unreadable, a symlink, altered after sealing,
    records a URL, or was written for another manifest allows no deletion.
    """
    value = _regular_json(Path(staging_dir) / RECEIPT_FILENAME)
    objects = value.get("staged_objects") if value else None
    if (value is None or value.get("schema_version") != RECEIPT_SCHEMA
            or value.get("private_url_recorded") is not False
            or value.get("receipt_digest") != _digest_or_none(value, "receipt_digest")
            or not staging_manifest_sha256
            or value.get("staging_manifest_sha256") != staging_manifest_sha256
            or not isinstance(objects, Mapping) or not set(objects) <= set(ROLES)
            or not all(_section_valid(section) for section in objects.values())):
        return None
    return value


def _sealed_receipt(staging_dir: str | Path) -> dict | None:
    """The staging dir's receipt when it is sealed and well formed, whatever manifest it binds."""
    value = _regular_json(Path(staging_dir) / RECEIPT_FILENAME)
    objects = value.get("staged_objects") if value else None
    if (value is None or value.get("schema_version") != RECEIPT_SCHEMA
            or value.get("private_url_recorded") is not False
            or value.get("receipt_digest") != _digest_or_none(value, "receipt_digest")
            or not _HEX64.fullmatch(str(value.get("staging_manifest_sha256")))
            or not isinstance(objects, Mapping) or not set(objects) <= set(ROLES)
            or not all(_section_valid(section) for section in objects.values())):
        return None
    return value


def _same_observation(recorded: Any, current: Mapping | None) -> bool:
    if recorded is None or current is None:
        return recorded is None and current is None
    return (isinstance(recorded, Mapping) and recorded.get("size_bytes") == current.get("size_bytes")
            and normalized_etag(recorded.get("etag")) == normalized_etag(current.get("etag")))


def rebindable_promotion_receipt(staging_dir: str | Path, *, staging_manifest_sha256: str, output_key: str,
                                 witness_key: str | None, observation: Mapping | None) -> dict | None:
    """A durable receipt an earlier version of this staging manifest bound, or None.

    A manifest rewrite that keeps the staged keys (``--refresh-output-get-url``
    adds URL metadata) changes the manifest digest a receipt binds, so the
    receipt no longer loads. It is carried forward only when it is sealed,
    ``promoted``, bound to another digest, names this output key (and this
    witness key, when it has a witness section), and recorded the same observed
    (size, ETag): the same staged object identity. Anything else is None.
    """
    value = _sealed_receipt(staging_dir)
    if (value is None or value.get("status") != "promoted"
            or value.get("staging_manifest_sha256") == staging_manifest_sha256):
        return None
    objects = value["staged_objects"]
    output, witness = objects.get("output"), objects.get("paired_witness")
    if (value.get("output_key_sha256") != key_sha256(output_key) or not isinstance(output, Mapping)
            or output.get("key_sha256") != key_sha256(output_key)
            or (witness is not None and (witness_key is None or witness.get("key_sha256") != key_sha256(witness_key)))
            or not _same_observation(value.get("observation"), observation)):
        return None
    return value


def set_aside_unbound_durable_receipt(staging_dir: str | Path, *, staging_manifest_sha256: str) -> dict | None:
    """Rename a durable receipt bound to another manifest out of the way; never delete it.

    Returns {path, receipt_digest, staging_manifest_sha256} of the renamed
    receipt, or None when there is no such receipt.
    """
    value = _sealed_receipt(staging_dir)
    if (value is None or value.get("status") != "promoted"
            or value.get("staging_manifest_sha256") == staging_manifest_sha256):
        return None
    staging = Path(staging_dir)
    aside = staging / f"{RECEIPT_FILENAME}.superseded-{uuid.uuid4().hex}"
    os.rename(staging / RECEIPT_FILENAME, aside)
    return {"path": aside.name, "receipt_digest": value["receipt_digest"],
            "staging_manifest_sha256": value["staging_manifest_sha256"]}


def staged_deletion_allowed(receipt: Mapping | None, *, role: str, key: str, size_bytes: int,
                            etag: str) -> tuple[bool, str]:
    """Whether the present object (size, ETag) at ``key`` is durable per ``receipt``."""
    objects = receipt.get("staged_objects") if isinstance(receipt, Mapping) else None
    section = objects.get(role) if isinstance(objects, Mapping) else None
    if (not isinstance(section, Mapping) or section.get("key_sha256") != key_sha256(key)
            or section.get("state") not in DURABLE_STATES or not isinstance(section.get("versions"), list)):
        return False, RECEIPT_MISSING
    if any(isinstance(version, Mapping) and version.get("size_bytes") == size_bytes
           and normalized_etag(version.get("etag")) == normalized_etag(etag)
           for version in section["versions"]):
        return True, "promoted_version_matches"
    return False, IDENTITY_MISMATCH


def promotion_gate_decision(receipt: Mapping | None, *, role: str, key: str,
                            present: tuple[int, str] | None) -> dict[str, Any]:
    """The cleanup gate's decision for one output or witness key.

    ``present`` is the (size, ETag) a HEAD saw, or None when it answered 404.
    An absent key needs no receipt; a present one is deleted only under
    ``staged_deletion_allowed``, and otherwise stays with a ``deferred`` row
    whose ``absence_confirmed`` is false and whose reason is the blocker.
    """
    if present is None:
        return {"decision": "absent_no_receipt_required",
                "absence": {"status": "passed", "absence_confirmed": True, "http_status_code": 404,
                            "raw_secret_values_recorded": False}}
    allowed, reason = staged_deletion_allowed(receipt, role=role, key=key, size_bytes=present[0],
                                              etag=present[1])
    if allowed:
        return {"decision": "deleted_after_promotion_receipt"}
    return {"decision": "deferred", "reason": reason,
            "absence": {"status": "deferred", "absence_confirmed": False, "object_still_present": True,
                        "raw_secret_values_recorded": False}}


def staged_object_keys(manifest: Mapping) -> list[tuple[str, str]]:
    """The exact (role, key) set cleanup deletes: the bundle unless retained, the output, the witness."""
    output_key = str(manifest.get("output_key") or "")
    keys = ([] if manifest.get("bundle_object_retained_for_reuse") is True
            else [("bundle", str(manifest.get("bundle_key") or ""))])
    keys.append(("output", output_key))
    witness = manifest.get("paired_witness")
    if isinstance(witness, Mapping) and witness.get("status") == "ready":
        keys.append(("paired_witness", str(witness.get("witness_key") or "")))
    return keys


def _manifest_and_keys(staging_dir: Path) -> tuple[str, dict, list[tuple[str, str]]]:
    manifest = _regular_json(staging_dir / STAGING_MANIFEST_FILENAME)
    digest = staging_manifest_sha256(staging_dir)
    if manifest is None or digest is None:
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_manifest_invalid")
    keys = staged_object_keys(manifest)
    witness_keys = [key for role, key in keys if role == "paired_witness"]
    if (not all(key for _, key in keys) or len({key for _, key in keys}) != len(keys)
            or any(key != str(manifest.get("output_key")) + PAIRED_WITNESS_SUFFIX for key in witness_keys)):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_manifest_invalid")
    return digest, manifest, keys


def _cleanup_proves_absence(cleanup: Any, manifest_sha256: str, keys: list[tuple[str, str]]) -> bool:
    rows = cleanup.get("objects") if isinstance(cleanup, Mapping) else None
    return bool(
        isinstance(cleanup, Mapping) and cleanup.get("schema_version") == CLEANUP_SCHEMA
        and cleanup.get("staging_manifest_sha256") == manifest_sha256
        and cleanup.get("status") == "completed" and cleanup.get("blockers") == []
        and cleanup.get("all_objects_absent") is True and cleanup.get("all_ephemeral_objects_absent") is True
        and cleanup.get("exact_object_count") == len(keys) and isinstance(rows, list)
        and all(isinstance(row, Mapping) and isinstance(row.get("absence"), Mapping) for row in rows)
        and sorted(str(row.get("key_sha256")) for row in rows) == sorted(key_sha256(key) for _, key in keys)
        and all(row["absence"].get("absence_confirmed") is True for row in rows))


def build_staged_object_absence_proof(*, staging_dir: str | Path, cleanup: Mapping,
                                      promotion: Mapping | None) -> dict:
    """A sealed absence proof for this staging dir, from a cleanup that proved every key absent."""
    staging = Path(staging_dir)
    digest, manifest, keys = _manifest_and_keys(staging)
    if not _cleanup_proves_absence(cleanup, digest, keys):
        raise ProviderOutputPromotionRecordError("staged_object_absence_not_proven")
    embedded = json.loads(json.dumps(cleanup))
    proof = {
        "schema_version": ABSENCE_PROOF_SCHEMA,
        "status": "all_staged_objects_absent",
        "staging_manifest_sha256": digest,
        "output_promotion_required": manifest.get("output_promotion_required") is True,
        "objects": sorted(({"role": role, "key_sha256": key_sha256(key)} for role, key in keys),
                          key=lambda row: row["key_sha256"]),
        "exact_object_count": len(keys),
        "cleanup": embedded,
        "cleanup_digest": canonical_digest(embedded),
        "promotion_receipt_digest": (promotion or {}).get("receipt_digest") if promotion else None,
        "promotion_status": (promotion or {}).get("status") if promotion else None,
        "all_staged_objects_absent": True,
        "private_url_recorded": False,
        "raw_secret_values_recorded": False,
    }
    proof["proof_digest"] = canonical_digest(proof, digest_field="proof_digest")
    return proof


def validate_staged_object_absence_proof(proof: Any, *, staging_dir: str | Path) -> dict:
    """Re-derive every binding of an absence proof against this staging dir, or refuse."""
    if not isinstance(proof, Mapping) or proof.get("schema_version") != ABSENCE_PROOF_SCHEMA:
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_invalid")
    if proof.get("proof_digest") != _digest_or_none(proof, "proof_digest"):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_digest_mismatch")
    if (proof.get("status") != "all_staged_objects_absent" or proof.get("all_staged_objects_absent") is not True
            or proof.get("private_url_recorded") is not False
            or proof.get("raw_secret_values_recorded") is not False):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_invalid")
    staging = Path(staging_dir)
    if proof.get("staging_manifest_sha256") != staging_manifest_sha256(staging):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_manifest_changed")
    digest, manifest, keys = _manifest_and_keys(staging)
    expected = sorted(({"role": role, "key_sha256": key_sha256(key)} for role, key in keys),
                      key=lambda row: row["key_sha256"])
    if proof.get("objects") != expected or proof.get("exact_object_count") != len(keys):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_keys_mismatch")
    cleanup = proof.get("cleanup")
    if (not _cleanup_proves_absence(cleanup, digest, keys)
            or proof.get("cleanup_digest") != _digest_or_none(cleanup)):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_cleanup_invalid")
    gated = manifest.get("output_promotion_required") is True
    if (proof.get("output_promotion_required") is not gated
            or (gated and (not _DIGEST.fullmatch(str(proof.get("promotion_receipt_digest")))
                           or proof.get("promotion_status") not in RECEIPT_STATUSES))):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_promotion_missing")
    return dict(proof)


def load_staged_object_absence_proof(staging_dir: str | Path) -> dict | None:
    """The staging dir's absence proof, validated, or None when there is none."""
    path = Path(staging_dir) / ABSENCE_PROOF_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    value = _regular_json(path)
    if value is None:
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_invalid")
    return validate_staged_object_absence_proof(value, staging_dir=staging_dir)


def write_staged_object_absence_proof(*, staging_dir: str | Path, cleanup: Mapping,
                                      promotion: Mapping | None) -> dict:
    """Write the absence proof once; an existing valid proof stands and is returned.

    The file is published by hard link from a fully written temporary, so a
    proof is never replaced and never seen half-written. An existing file that
    does not validate is refused, not overwritten.
    """
    staging = Path(staging_dir)
    existing = load_staged_object_absence_proof(staging)
    if existing is not None:
        return existing
    proof = build_staged_object_absence_proof(staging_dir=staging, cleanup=cleanup, promotion=promotion)
    path = staging / ABSENCE_PROOF_FILENAME
    temporary = staging / f".{ABSENCE_PROOF_FILENAME}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(proof, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return load_staged_object_absence_proof(staging) or proof
    finally:
        temporary.unlink(missing_ok=True)
    return proof


__all__ = [
    "ABSENCE_PROOF_FILENAME",
    "ABSENCE_PROOF_SCHEMA",
    "DURABLE_STATES",
    "IDENTITY_MISMATCH",
    "RECEIPT_FILENAME",
    "RECEIPT_MISSING",
    "RECEIPT_SCHEMA",
    "ProviderOutputPromotionRecordError",
    "build_staged_object_absence_proof",
    "key_sha256",
    "load_promotion_receipt",
    "load_staged_object_absence_proof",
    "normalized_etag",
    "promotion_gate_decision",
    "rebindable_promotion_receipt",
    "set_aside_unbound_durable_receipt",
    "staged_deletion_allowed",
    "staged_object_keys",
    "staging_manifest_sha256",
    "validate_staged_object_absence_proof",
    "write_promotion_receipt",
    "write_staged_object_absence_proof",
]
