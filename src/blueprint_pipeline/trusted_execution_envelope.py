"""Task-neutral, runner-signed execution structure evidence.

The envelope binds immutable execution identities to one exact returned archive.
Verification establishes only that a configured runner key signed the canonical
structure and that the supplied return archive has the signed byte identity.  It
does not interpret allocator artifacts, prove provider-zero, qualify a native
runtime gate, grade an episode, or establish physical truth.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


SCHEMA_VERSION = "trusted_execution_envelope.v1"
VERIFICATION_SCHEMA_VERSION = "trusted_execution_envelope_verification.v1"
TRUSTED_PUBLIC_KEY_SHA256_ENV = "BLUEPRINT_TRUSTED_EXECUTION_ENVELOPE_PUBLIC_KEY_SHA256"
SIGNATURE_DOMAIN = b"blueprint.trusted_execution_envelope.v1\x00"
SIGNATURE_DOMAIN_ID = "blueprint.trusted_execution_envelope.v1"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_NONCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{15,255}$")
_ENVELOPE_FIELDS = frozenset({"schema_version", "payload", "signature"})
_PAYLOAD_FIELDS = frozenset(
    {
        "nonce",
        "run_digest",
        "package_digest",
        "execution_request_digest",
        "worker",
        "instance_id",
        "return_zip",
        "started_at",
        "ended_at",
        "allocator_lifecycle_artifact_digests",
    }
)
_WORKER_FIELDS = frozenset({"entrypoint", "source_tree_digest", "container_digest"})
_RETURN_FIELDS = frozenset({"sha256", "size_bytes"})
_SIGNATURE_FIELDS = frozenset(
    {
        "algorithm",
        "signature_domain",
        "public_key_base64",
        "public_key_sha256",
        "signed_payload_sha256",
        "signature_base64",
    }
)
_MAX_ENVELOPE_BYTES = 128 * 1024
_MAX_RETURN_BYTES = 16 * 1024**3
_MAX_LIFECYCLE_ARTIFACTS = 32
_READ_CHUNK_BYTES = 1024 * 1024


class TrustedExecutionEnvelopeError(ValueError):
    """Stable, sorted failures while materializing an envelope."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _valid_text(value: Any, *, maximum_length: int) -> bool:
    return bool(
        isinstance(value, str)
        and value
        and value == value.strip()
        and len(value) <= maximum_length
        and all(
            ord(character) >= 0x20
            and character != "\x7f"
            and not 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    )


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.utcoffset() is not None else None


def _payload_errors(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["trusted_execution_envelope_payload_not_mapping"]
    errors: list[str] = []
    if set(payload) != _PAYLOAD_FIELDS:
        errors.append("trusted_execution_envelope_payload_fields_invalid")
    nonce = payload.get("nonce")
    if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        errors.append("trusted_execution_envelope_nonce_invalid")
    for field in ("run_digest", "package_digest", "execution_request_digest"):
        if not _is_digest(payload.get(field)):
            errors.append(f"trusted_execution_envelope_{field}_invalid")

    worker = payload.get("worker")
    if not isinstance(worker, Mapping) or set(worker) != _WORKER_FIELDS:
        errors.append("trusted_execution_envelope_worker_invalid")
    else:
        if not _valid_text(worker.get("entrypoint"), maximum_length=512):
            errors.append("trusted_execution_envelope_worker_entrypoint_invalid")
        for field in ("source_tree_digest", "container_digest"):
            if not _is_digest(worker.get(field)):
                errors.append(f"trusted_execution_envelope_worker_{field}_invalid")

    instance_id = payload.get("instance_id")
    if not isinstance(instance_id, str) or not _IDENTIFIER_RE.fullmatch(instance_id):
        errors.append("trusted_execution_envelope_instance_id_invalid")

    returned = payload.get("return_zip")
    if not isinstance(returned, Mapping) or set(returned) != _RETURN_FIELDS:
        errors.append("trusted_execution_envelope_return_zip_invalid")
    else:
        if not _is_digest(returned.get("sha256")):
            errors.append("trusted_execution_envelope_return_zip_sha256_invalid")
        size = returned.get("size_bytes")
        if type(size) is not int or not 0 < size <= _MAX_RETURN_BYTES:
            errors.append("trusted_execution_envelope_return_zip_size_invalid")

    started = _parse_utc_timestamp(payload.get("started_at"))
    ended = _parse_utc_timestamp(payload.get("ended_at"))
    if started is None:
        errors.append("trusted_execution_envelope_started_at_invalid")
    if ended is None:
        errors.append("trusted_execution_envelope_ended_at_invalid")
    if started is not None and ended is not None and ended < started:
        errors.append("trusted_execution_envelope_time_order_invalid")

    lifecycle = payload.get("allocator_lifecycle_artifact_digests")
    if (
        not isinstance(lifecycle, Mapping)
        or not lifecycle
        or len(lifecycle) > _MAX_LIFECYCLE_ARTIFACTS
    ):
        errors.append("trusted_execution_envelope_allocator_lifecycle_digests_invalid")
    else:
        for role, digest in lifecycle.items():
            if not isinstance(role, str) or not _IDENTIFIER_RE.fullmatch(role):
                errors.append("trusted_execution_envelope_allocator_lifecycle_role_invalid")
            if not _is_digest(digest):
                errors.append("trusted_execution_envelope_allocator_lifecycle_digest_invalid")
    return sorted(set(errors))


def _normalized_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = _payload_errors(payload)
    if errors:
        raise TrustedExecutionEnvelopeError(errors)
    return json.loads(_canonical_json(payload))


def materialize_trusted_execution_payload(
    *,
    nonce: str,
    run_digest: str,
    package_digest: str,
    execution_request_digest: str,
    worker_entrypoint: str,
    worker_source_tree_digest: str,
    worker_container_digest: str,
    instance_id: str,
    return_zip_sha256: str,
    return_zip_size_bytes: int,
    started_at: str,
    ended_at: str,
    allocator_lifecycle_artifact_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Create the strict canonical payload a trusted runner must sign."""

    payload = {
        "nonce": nonce,
        "run_digest": run_digest,
        "package_digest": package_digest,
        "execution_request_digest": execution_request_digest,
        "worker": {
            "entrypoint": worker_entrypoint,
            "source_tree_digest": worker_source_tree_digest,
            "container_digest": worker_container_digest,
        },
        "instance_id": instance_id,
        "return_zip": {
            "sha256": return_zip_sha256,
            "size_bytes": return_zip_size_bytes,
        },
        "started_at": started_at,
        "ended_at": ended_at,
        "allocator_lifecycle_artifact_digests": dict(allocator_lifecycle_artifact_digests),
    }
    return _normalized_payload(payload)


def trusted_execution_signature_message(payload: Mapping[str, Any]) -> bytes:
    """Return the domain-separated bytes signed by the trusted runner."""

    normalized = _normalized_payload(payload)
    return SIGNATURE_DOMAIN + _canonical_bytes(normalized)


def _decode_signature(signature: Mapping[str, Any]) -> tuple[bytes, bytes, list[str]]:
    errors: list[str] = []
    if set(signature) != _SIGNATURE_FIELDS:
        errors.append("trusted_execution_envelope_signature_fields_invalid")
    if signature.get("algorithm") != "Ed25519":
        errors.append("trusted_execution_envelope_signature_algorithm_invalid")
    if signature.get("signature_domain") != SIGNATURE_DOMAIN_ID:
        errors.append("trusted_execution_envelope_signature_domain_invalid")
    if not _is_digest(signature.get("public_key_sha256")):
        errors.append("trusted_execution_envelope_public_key_sha256_invalid")
    if not _is_digest(signature.get("signed_payload_sha256")):
        errors.append("trusted_execution_envelope_signed_payload_sha256_invalid")
    try:
        public_key = base64.b64decode(signature.get("public_key_base64", ""), validate=True)
        raw_signature = base64.b64decode(signature.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError, binascii.Error):
        public_key, raw_signature = b"", b""
        errors.append("trusted_execution_envelope_signature_encoding_invalid")
    if isinstance(signature.get("public_key_base64"), str) and base64.b64encode(public_key).decode(
        "ascii"
    ) != signature.get("public_key_base64"):
        errors.append("trusted_execution_envelope_signature_encoding_invalid")
    if isinstance(signature.get("signature_base64"), str) and base64.b64encode(
        raw_signature
    ).decode("ascii") != signature.get("signature_base64"):
        errors.append("trusted_execution_envelope_signature_encoding_invalid")
    if len(public_key) != 32:
        errors.append("trusted_execution_envelope_public_key_length_invalid")
    if len(raw_signature) != 64:
        errors.append("trusted_execution_envelope_signature_length_invalid")
    return public_key, raw_signature, sorted(set(errors))


def _signature_errors(payload: Mapping[str, Any], signature: Any) -> tuple[list[str], str, bool]:
    if not isinstance(signature, Mapping):
        return ["trusted_execution_envelope_signature_not_mapping"], "", False
    public_key, raw_signature, errors = _decode_signature(signature)
    fingerprint = _sha256_bytes(public_key) if len(public_key) == 32 else ""
    if fingerprint and signature.get("public_key_sha256") != fingerprint:
        errors.append("trusted_execution_envelope_public_key_fingerprint_mismatch")
    payload_digest = _sha256_bytes(_canonical_bytes(payload))
    if signature.get("signed_payload_sha256") != payload_digest:
        errors.append("trusted_execution_envelope_signed_payload_sha256_mismatch")
    cryptographically_valid = False
    if len(public_key) == 32 and len(raw_signature) == 64:
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                raw_signature, SIGNATURE_DOMAIN + _canonical_bytes(payload)
            )
            cryptographically_valid = True
        except (InvalidSignature, ValueError):
            errors.append("trusted_execution_envelope_signature_verification_failed")
    return sorted(set(errors)), fingerprint, cryptographically_valid


def materialize_trusted_execution_envelope(
    *,
    payload: Mapping[str, Any],
    public_key_base64: str,
    signature_base64: str,
) -> dict[str, Any]:
    """Materialize a canonical envelope from an externally produced signature."""

    normalized = _normalized_payload(payload)
    try:
        public_key = base64.b64decode(public_key_base64, validate=True)
    except (TypeError, ValueError, binascii.Error):
        public_key = b""
    signature = {
        "algorithm": "Ed25519",
        "signature_domain": SIGNATURE_DOMAIN_ID,
        "public_key_base64": public_key_base64,
        "public_key_sha256": _sha256_bytes(public_key),
        "signed_payload_sha256": _sha256_bytes(_canonical_bytes(normalized)),
        "signature_base64": signature_base64,
    }
    errors, _, _ = _signature_errors(normalized, signature)
    if errors:
        raise TrustedExecutionEnvelopeError(errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "payload": normalized,
        "signature": signature,
    }


def canonical_trusted_execution_envelope_bytes(
    envelope: Mapping[str, Any],
) -> bytes:
    """Serialize a locally validated envelope in its sole admitted encoding."""

    errors, normalized, _, _ = _envelope_errors(envelope)
    if errors:
        raise TrustedExecutionEnvelopeError(errors)
    return _canonical_bytes(normalized) + b"\n"


def _envelope_errors(
    envelope: Any,
) -> tuple[list[str], dict[str, Any], str, bool]:
    if not isinstance(envelope, Mapping):
        return ["trusted_execution_envelope_not_mapping"], {}, "", False
    try:
        normalized = json.loads(_canonical_json(envelope))
    except (TypeError, ValueError, RecursionError):
        return ["trusted_execution_envelope_json_invalid"], {}, "", False
    errors: list[str] = []
    if set(normalized) != _ENVELOPE_FIELDS:
        errors.append("trusted_execution_envelope_fields_invalid")
    if normalized.get("schema_version") != SCHEMA_VERSION:
        errors.append("trusted_execution_envelope_schema_version_invalid")
    payload = normalized.get("payload")
    errors.extend(_payload_errors(payload))
    fingerprint = ""
    cryptographically_valid = False
    if isinstance(payload, Mapping):
        signature_errors, fingerprint, cryptographically_valid = _signature_errors(
            payload, normalized.get("signature")
        )
        errors.extend(signature_errors)
    else:
        errors.append("trusted_execution_envelope_signature_payload_unavailable")
    return sorted(set(errors)), normalized, fingerprint, cryptographically_valid


def _strict_json_object(value: bytes) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = item
        return result

    parsed = json.loads(
        value.decode("utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if not isinstance(parsed, dict):
        raise ValueError("not_mapping")
    return parsed


def _read_once_no_follow(
    path_value: str | Path, *, maximum_size: int, retain_bytes: bool
) -> tuple[dict[str, Any], list[str]]:
    path = Path(path_value).expanduser()
    artifact = {
        "path": str(path),
        "sha256": None,
        "size_bytes": None,
        "opened_once_no_follow": False,
    }
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        return artifact, ["no_follow_unavailable"]
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | no_follow
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return artifact, ["not_regular"]
        if before.st_size <= 0 or before.st_size > maximum_size:
            return artifact, ["size_invalid"]
        digest = hashlib.sha256()
        retained = bytearray() if retain_bytes else None
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_size:
                return artifact, ["size_invalid"]
            digest.update(chunk)
            if retained is not None:
                retained.extend(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or total != before.st_size:
            return artifact, ["changed_while_reading"]
        artifact.update(
            {
                "sha256": "sha256:" + digest.hexdigest(),
                "size_bytes": total,
                "opened_once_no_follow": True,
            }
        )
        if retained is not None:
            artifact["bytes"] = bytes(retained)
        return artifact, []
    except OSError:
        return artifact, ["unreadable"]
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _expected_binding_errors(
    payload: Mapping[str, Any],
    *,
    expected_nonce: str,
    expected_run_digest: str,
    expected_package_digest: str,
    expected_execution_request_digest: str,
    expected_worker_entrypoint: str,
    expected_worker_source_tree_digest: str,
    expected_worker_container_digest: str,
    expected_instance_id: str,
    expected_allocator_lifecycle_artifact_digests: Mapping[str, str],
) -> list[str]:
    errors: list[str] = []
    direct = {
        "nonce": expected_nonce,
        "run_digest": expected_run_digest,
        "package_digest": expected_package_digest,
        "execution_request_digest": expected_execution_request_digest,
        "instance_id": expected_instance_id,
    }
    for field, expected in direct.items():
        if payload.get(field) != expected:
            errors.append(f"trusted_execution_envelope_{field}_mismatch")
    worker = payload.get("worker") if isinstance(payload.get("worker"), Mapping) else {}
    worker_expected = {
        "entrypoint": expected_worker_entrypoint,
        "source_tree_digest": expected_worker_source_tree_digest,
        "container_digest": expected_worker_container_digest,
    }
    for field, expected in worker_expected.items():
        if worker.get(field) != expected:
            errors.append(f"trusted_execution_envelope_worker_{field}_mismatch")
    if not isinstance(expected_allocator_lifecycle_artifact_digests, Mapping):
        return ["trusted_execution_envelope_expected_lifecycle_digests_invalid"]
    expected_lifecycle = dict(expected_allocator_lifecycle_artifact_digests)
    if payload.get("allocator_lifecycle_artifact_digests") != expected_lifecycle:
        errors.append("trusted_execution_envelope_allocator_lifecycle_digests_mismatch")
    return errors


def verify_trusted_execution_envelope(
    envelope_path: str | Path,
    *,
    return_zip_path: str | Path,
    expected_nonce: str,
    expected_run_digest: str,
    expected_package_digest: str,
    expected_execution_request_digest: str,
    expected_worker_entrypoint: str,
    expected_worker_source_tree_digest: str,
    expected_worker_container_digest: str,
    expected_instance_id: str,
    expected_allocator_lifecycle_artifact_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Verify runner signature and exact structural bindings, nothing more."""

    blockers: list[str] = []
    envelope_artifact, read_errors = _read_once_no_follow(
        envelope_path, maximum_size=_MAX_ENVELOPE_BYTES, retain_bytes=True
    )
    blockers.extend(f"trusted_execution_envelope_file_{error}" for error in read_errors)
    envelope: dict[str, Any] = {}
    fingerprint = ""
    cryptographically_valid = False
    envelope_bytes = envelope_artifact.pop("bytes", None)
    if isinstance(envelope_bytes, bytes):
        try:
            envelope = _strict_json_object(envelope_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            blockers.append("trusted_execution_envelope_json_invalid")
        if envelope:
            errors, normalized, fingerprint, cryptographically_valid = _envelope_errors(envelope)
            blockers.extend(errors)
            if envelope_bytes != _canonical_bytes(normalized) + b"\n":
                blockers.append("trusted_execution_envelope_encoding_not_canonical")

    configured_fingerprint = os.getenv(TRUSTED_PUBLIC_KEY_SHA256_ENV)
    if not _is_digest(configured_fingerprint):
        blockers.append("trusted_execution_envelope_trusted_public_key_not_configured")
    elif fingerprint != configured_fingerprint:
        blockers.append("trusted_execution_envelope_public_key_not_authorized")

    payload = envelope.get("payload") if isinstance(envelope.get("payload"), Mapping) else {}
    if payload:
        blockers.extend(
            _expected_binding_errors(
                payload,
                expected_nonce=expected_nonce,
                expected_run_digest=expected_run_digest,
                expected_package_digest=expected_package_digest,
                expected_execution_request_digest=expected_execution_request_digest,
                expected_worker_entrypoint=expected_worker_entrypoint,
                expected_worker_source_tree_digest=expected_worker_source_tree_digest,
                expected_worker_container_digest=expected_worker_container_digest,
                expected_instance_id=expected_instance_id,
                expected_allocator_lifecycle_artifact_digests=(
                    expected_allocator_lifecycle_artifact_digests
                ),
            )
        )

    return_artifact, return_errors = _read_once_no_follow(
        return_zip_path, maximum_size=_MAX_RETURN_BYTES, retain_bytes=False
    )
    blockers.extend(f"trusted_execution_envelope_return_zip_{error}" for error in return_errors)
    returned = payload.get("return_zip") if isinstance(payload.get("return_zip"), Mapping) else {}
    if return_artifact.get("sha256") != returned.get("sha256"):
        blockers.append("trusted_execution_envelope_return_zip_sha256_mismatch")
    if return_artifact.get("size_bytes") != returned.get("size_bytes"):
        blockers.append("trusted_execution_envelope_return_zip_size_mismatch")

    blockers = sorted(set(blockers))
    verified = not blockers
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified" if verified else "blocked",
        "structural_trust_verified": verified,
        "signature_cryptographically_valid": cryptographically_valid,
        "configured_runner_key_matched": bool(
            _is_digest(configured_fingerprint) and fingerprint == configured_fingerprint
        ),
        "envelope_artifact": envelope_artifact,
        "return_zip_artifact": return_artifact,
        "signed_payload_sha256": (
            envelope.get("signature", {}).get("signed_payload_sha256")
            if isinstance(envelope.get("signature"), Mapping)
            else None
        ),
        "presented_public_key_sha256": fingerprint or None,
        "blockers": blockers,
        "claim_scope": "signed_runner_execution_structure_only",
        "does_not_establish": [
            "allocator_lifecycle_semantics",
            "provider_zero",
            "native_simulator_gate_outcomes",
            "task_or_policy_success",
            "physical_truth",
        ],
    }
