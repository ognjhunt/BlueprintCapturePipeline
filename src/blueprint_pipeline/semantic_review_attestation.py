"""Verify task-neutral, authority-signed visual semantic review evidence.

The signed payload authenticates a semantic review decision and its exact
evidence identities.  A separate path-only selection contract freezes which
attestation and authority are admitted by a downstream build.  Verification
uses only the configured public-key fingerprint; callers cannot inject a trust
root.

This module establishes development-only visual semantic authority.  It does
not qualify simulator import, contacts, task success, physical material
equivalence, or source-scene truth beyond the signed assertions.
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
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


ATTESTATION_SCHEMA_VERSION = "semantic_review_attestation.v1"
SELECTION_SCHEMA_VERSION = "semantic_review_authority_selection.v1"
VERIFICATION_SCHEMA_VERSION = "semantic_review_attestation_verification.v1"
TRUSTED_PUBLIC_KEY_SHA256_ENV = "BLUEPRINT_SEMANTIC_REVIEW_AUTHORITY_PUBLIC_KEY_SHA256"
SIGNATURE_DOMAIN = b"blueprint.semantic_review_attestation.v1\x00"
SIGNATURE_DOMAIN_ID = "blueprint.semantic_review_attestation.v1"
CLAIM_SCOPE = "development_only_visual_semantic_review"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_ATTESTATION_FIELDS = frozenset({"schema_version", "payload", "signature", "attestation_digest"})
_PAYLOAD_FIELDS = frozenset(
    {
        "attestation_id",
        "selection_id",
        "authority_id",
        "authority_key_id",
        "scene_id",
        "source_target",
        "evidence",
        "learned_policy_outcomes_inspected",
        "semantic_assertions",
        "claim_scope",
        "claim_boundary",
    }
)
_TARGET_FIELDS = frozenset({"target_id", "source_instance_id", "semantic_role"})
_EVIDENCE_FIELDS = frozenset(
    {
        "visual_review_digest",
        "render_manifest_digest",
        "collision_topology_receipt_digest",
        "cited_frames_digest",
    }
)
_CLAIM_BOUNDARY_FIELDS = frozenset(
    {
        "native_simulator_qualified",
        "physical_equivalence_proven",
        "evaluation_policy_media",
        "source_capture_replaced",
    }
)
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
_SELECTION_FIELDS = frozenset(
    {
        "schema_version",
        "selection_id",
        "attestation_digest",
        "authority",
        "scene_id",
        "source_target",
        "evidence",
        "learned_policy_outcomes_inspected",
        "claim_scope",
        "selection_digest",
    }
)
_AUTHORITY_FIELDS = frozenset({"authority_id", "key_id", "public_key_sha256"})
_FRAME_FIELDS = frozenset({"target_id", "camera_id", "sha256", "size_bytes", "decoded_rgb_sha256"})
_MAX_ATTESTATION_BYTES = 512 * 1024
_MAX_SELECTION_BYTES = 128 * 1024
_MAX_ASSERTIONS = 128
_MAX_FRAME_ROWS = 512


class SemanticReviewAttestationError(ValueError):
    """Stable, sorted failures at the semantic-authority boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({error for error in errors if error}))
        super().__init__(";".join(self.errors))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER_RE.fullmatch(value))


def _is_text(value: Any, *, maximum: int = 1024) -> bool:
    return bool(
        isinstance(value, str)
        and value == value.strip()
        and value
        and len(value) <= maximum
        and all(
            ord(character) >= 0x20
            and character != "\x7f"
            and not 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
    )


def _strict_clone(value: Mapping[str, Any], *, error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(_canonical_json(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise SemanticReviewAttestationError([error]) from exc
    if not isinstance(cloned, dict):
        raise SemanticReviewAttestationError([error])
    return cloned


def _target_errors(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != _TARGET_FIELDS:
        return [f"{prefix}_source_target_invalid"]
    if not all(_is_identifier(value.get(field)) for field in _TARGET_FIELDS):
        return [f"{prefix}_source_target_invalid"]
    return []


def _evidence_errors(value: Any, *, prefix: str) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_FIELDS:
        return [f"{prefix}_evidence_invalid"]
    if not all(_is_digest(value.get(field)) for field in _EVIDENCE_FIELDS):
        return [f"{prefix}_evidence_invalid"]
    return []


def _assertion_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping) or not value or len(value) > _MAX_ASSERTIONS:
        return ["semantic_review_attestation_assertions_invalid"]
    for key, item in value.items():
        if not _is_identifier(key):
            return ["semantic_review_attestation_assertions_invalid"]
        if type(item) is bool:
            continue
        if isinstance(item, str) and _is_text(item):
            continue
        return ["semantic_review_attestation_assertions_invalid"]
    return []


def _payload_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["semantic_review_attestation_payload_invalid"]
    errors: list[str] = []
    if set(value) != _PAYLOAD_FIELDS:
        errors.append("semantic_review_attestation_payload_fields_invalid")
    if not _is_identifier(value.get("attestation_id")):
        errors.append("semantic_review_attestation_id_invalid")
    if not _is_identifier(value.get("selection_id")):
        errors.append("semantic_review_attestation_selection_id_invalid")
    if not _is_identifier(value.get("authority_id")):
        errors.append("semantic_review_attestation_authority_id_invalid")
    if not _is_identifier(value.get("authority_key_id")):
        errors.append("semantic_review_attestation_authority_key_id_invalid")
    if not _is_identifier(value.get("scene_id")):
        errors.append("semantic_review_attestation_scene_id_invalid")
    errors.extend(_target_errors(value.get("source_target"), prefix="semantic_review_attestation"))
    errors.extend(_evidence_errors(value.get("evidence"), prefix="semantic_review_attestation"))
    if value.get("learned_policy_outcomes_inspected") is not False:
        errors.append("semantic_review_attestation_policy_outcomes_inspected")
    errors.extend(_assertion_errors(value.get("semantic_assertions")))
    if value.get("claim_scope") != CLAIM_SCOPE:
        errors.append("semantic_review_attestation_claim_scope_invalid")
    boundary = value.get("claim_boundary")
    if (
        not isinstance(boundary, Mapping)
        or set(boundary) != _CLAIM_BOUNDARY_FIELDS
        or any(boundary.get(field) is not False for field in _CLAIM_BOUNDARY_FIELDS)
    ):
        errors.append("semantic_review_attestation_claim_boundary_invalid")
    return sorted(set(errors))


def _normalized_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    errors = _payload_errors(value)
    if errors:
        raise SemanticReviewAttestationError(errors)
    return _strict_clone(value, error="semantic_review_attestation_payload_invalid")


def materialize_semantic_review_payload(
    *,
    attestation_id: str,
    selection_id: str,
    authority_id: str,
    authority_key_id: str,
    scene_id: str,
    target_id: str,
    source_instance_id: str,
    semantic_role: str,
    visual_review_digest: str,
    render_manifest_digest: str,
    collision_topology_receipt_digest: str,
    cited_frames_digest: str,
    learned_policy_outcomes_inspected: bool,
    semantic_assertions: Mapping[str, bool | str],
) -> dict[str, Any]:
    """Materialize the sole payload admitted for authority signing."""

    return _normalized_payload(
        {
            "attestation_id": attestation_id,
            "selection_id": selection_id,
            "authority_id": authority_id,
            "authority_key_id": authority_key_id,
            "scene_id": scene_id,
            "source_target": {
                "target_id": target_id,
                "source_instance_id": source_instance_id,
                "semantic_role": semantic_role,
            },
            "evidence": {
                "visual_review_digest": visual_review_digest,
                "render_manifest_digest": render_manifest_digest,
                "collision_topology_receipt_digest": (collision_topology_receipt_digest),
                "cited_frames_digest": cited_frames_digest,
            },
            "learned_policy_outcomes_inspected": learned_policy_outcomes_inspected,
            "semantic_assertions": semantic_assertions,
            "claim_scope": CLAIM_SCOPE,
            "claim_boundary": {
                "native_simulator_qualified": False,
                "physical_equivalence_proven": False,
                "evaluation_policy_media": False,
                "source_capture_replaced": False,
            },
        }
    )


def semantic_review_signature_message(payload: Mapping[str, Any]) -> bytes:
    """Return domain-separated canonical bytes for the authority signer."""

    return SIGNATURE_DOMAIN + _canonical_bytes(_normalized_payload(payload))


def _signature_errors(payload: Mapping[str, Any], signature: Any) -> tuple[list[str], str, bool]:
    if not isinstance(signature, Mapping):
        return ["semantic_review_attestation_signature_invalid"], "", False
    errors: list[str] = []
    if set(signature) != _SIGNATURE_FIELDS:
        errors.append("semantic_review_attestation_signature_fields_invalid")
    if signature.get("algorithm") != "Ed25519":
        errors.append("semantic_review_attestation_signature_algorithm_invalid")
    if signature.get("signature_domain") != SIGNATURE_DOMAIN_ID:
        errors.append("semantic_review_attestation_signature_domain_invalid")
    try:
        public_key = base64.b64decode(signature.get("public_key_base64", ""), validate=True)
        raw_signature = base64.b64decode(signature.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError, binascii.Error):
        public_key, raw_signature = b"", b""
        errors.append("semantic_review_attestation_signature_encoding_invalid")
    if isinstance(signature.get("public_key_base64"), str) and base64.b64encode(public_key).decode(
        "ascii"
    ) != signature.get("public_key_base64"):
        errors.append("semantic_review_attestation_signature_encoding_invalid")
    if isinstance(signature.get("signature_base64"), str) and base64.b64encode(
        raw_signature
    ).decode("ascii") != signature.get("signature_base64"):
        errors.append("semantic_review_attestation_signature_encoding_invalid")
    if len(public_key) != 32:
        errors.append("semantic_review_attestation_public_key_length_invalid")
    if len(raw_signature) != 64:
        errors.append("semantic_review_attestation_signature_length_invalid")
    fingerprint = _sha256_bytes(public_key) if len(public_key) == 32 else ""
    if signature.get("public_key_sha256") != fingerprint:
        errors.append("semantic_review_attestation_public_key_fingerprint_mismatch")
    payload_sha256 = _sha256_bytes(_canonical_bytes(payload))
    if signature.get("signed_payload_sha256") != payload_sha256:
        errors.append("semantic_review_attestation_signed_payload_mismatch")
    cryptographically_valid = False
    if len(public_key) == 32 and len(raw_signature) == 64:
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                raw_signature,
                SIGNATURE_DOMAIN + _canonical_bytes(payload),
            )
            cryptographically_valid = True
        except (InvalidSignature, ValueError):
            errors.append("semantic_review_attestation_signature_verification_failed")
    return sorted(set(errors)), fingerprint, cryptographically_valid


def materialize_semantic_review_attestation(
    *,
    payload: Mapping[str, Any],
    public_key_base64: str,
    signature_base64: str,
) -> dict[str, Any]:
    """Materialize an attestation from an externally produced signature."""

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
        raise SemanticReviewAttestationError(errors)
    result: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "payload": normalized,
        "signature": signature,
        "attestation_digest": "",
    }
    result["attestation_digest"] = _sha256_bytes(
        _canonical_bytes(
            {key: value for key, value in result.items() if key != "attestation_digest"}
        )
    )
    return result


def _attestation_errors(
    value: Any,
) -> tuple[list[str], dict[str, Any], str, bool]:
    if not isinstance(value, Mapping):
        return ["semantic_review_attestation_invalid"], {}, "", False
    try:
        normalized = _strict_clone(value, error="semantic_review_attestation_invalid")
    except SemanticReviewAttestationError as exc:
        return list(exc.errors), {}, "", False
    errors: list[str] = []
    if set(normalized) != _ATTESTATION_FIELDS:
        errors.append("semantic_review_attestation_fields_invalid")
    if normalized.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        errors.append("semantic_review_attestation_schema_invalid")
    payload = normalized.get("payload")
    errors.extend(_payload_errors(payload))
    fingerprint = ""
    cryptographically_valid = False
    if isinstance(payload, Mapping):
        signature_errors, fingerprint, cryptographically_valid = _signature_errors(
            payload, normalized.get("signature")
        )
        errors.extend(signature_errors)
    expected_digest = _sha256_bytes(
        _canonical_bytes(
            {key: item for key, item in normalized.items() if key != "attestation_digest"}
        )
    )
    if normalized.get("attestation_digest") != expected_digest:
        errors.append("semantic_review_attestation_digest_invalid")
    return sorted(set(errors)), normalized, fingerprint, cryptographically_valid


def canonical_semantic_review_attestation_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a validated attestation in its only admitted encoding."""

    errors, normalized, _, _ = _attestation_errors(value)
    if errors:
        raise SemanticReviewAttestationError(errors)
    return _canonical_bytes(normalized) + b"\n"


def _selection_errors(value: Any) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(value, Mapping):
        return ["semantic_authority_selection_invalid"], {}
    try:
        normalized = _strict_clone(value, error="semantic_authority_selection_invalid")
    except SemanticReviewAttestationError as exc:
        return list(exc.errors), {}
    errors: list[str] = []
    if set(normalized) != _SELECTION_FIELDS:
        errors.append("semantic_authority_selection_fields_invalid")
    if normalized.get("schema_version") != SELECTION_SCHEMA_VERSION:
        errors.append("semantic_authority_selection_schema_invalid")
    if not _is_identifier(normalized.get("selection_id")):
        errors.append("semantic_authority_selection_id_invalid")
    if not _is_digest(normalized.get("attestation_digest")):
        errors.append("semantic_authority_selection_attestation_digest_invalid")
    authority = normalized.get("authority")
    if (
        not isinstance(authority, Mapping)
        or set(authority) != _AUTHORITY_FIELDS
        or not _is_identifier(authority.get("authority_id"))
        or not _is_identifier(authority.get("key_id"))
        or not _is_digest(authority.get("public_key_sha256"))
    ):
        errors.append("semantic_authority_selection_authority_invalid")
    if not _is_identifier(normalized.get("scene_id")):
        errors.append("semantic_authority_selection_scene_id_invalid")
    errors.extend(
        _target_errors(normalized.get("source_target"), prefix="semantic_authority_selection")
    )
    errors.extend(
        _evidence_errors(normalized.get("evidence"), prefix="semantic_authority_selection")
    )
    if normalized.get("learned_policy_outcomes_inspected") is not False:
        errors.append("semantic_authority_selection_policy_outcomes_inspected")
    if normalized.get("claim_scope") != CLAIM_SCOPE:
        errors.append("semantic_authority_selection_claim_scope_invalid")
    expected_digest = _sha256_bytes(
        _canonical_bytes(
            {key: item for key, item in normalized.items() if key != "selection_digest"}
        )
    )
    if normalized.get("selection_digest") != expected_digest:
        errors.append("semantic_authority_selection_digest_invalid")
    return sorted(set(errors)), normalized


def materialize_semantic_authority_selection(*, attestation: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one exact signed attestation as the admitted semantic authority."""

    errors, normalized_attestation, fingerprint, _ = _attestation_errors(attestation)
    if errors:
        raise SemanticReviewAttestationError(errors)
    payload = normalized_attestation["payload"]
    result: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_id": payload["selection_id"],
        "attestation_digest": normalized_attestation["attestation_digest"],
        "authority": {
            "authority_id": payload["authority_id"],
            "key_id": payload["authority_key_id"],
            "public_key_sha256": fingerprint,
        },
        "scene_id": payload["scene_id"],
        "source_target": payload["source_target"],
        "evidence": payload["evidence"],
        "learned_policy_outcomes_inspected": False,
        "claim_scope": CLAIM_SCOPE,
        "selection_digest": "",
    }
    result["selection_digest"] = _sha256_bytes(
        _canonical_bytes({key: value for key, value in result.items() if key != "selection_digest"})
    )
    selection_errors, normalized = _selection_errors(result)
    if selection_errors:
        raise SemanticReviewAttestationError(selection_errors)
    return normalized


def canonical_semantic_authority_selection_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a validated selection in its only admitted encoding."""

    errors, normalized = _selection_errors(value)
    if errors:
        raise SemanticReviewAttestationError(errors)
    return _canonical_bytes(normalized) + b"\n"


def semantic_frame_evidence_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest the exact normalized frame identities inspected by an authority."""

    if (
        isinstance(rows, (str, bytes, Mapping))
        or not isinstance(rows, Sequence)
        or not rows
        or len(rows) > _MAX_FRAME_ROWS
    ):
        raise SemanticReviewAttestationError(["semantic_review_frame_evidence_invalid"])
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _FRAME_FIELDS:
            raise SemanticReviewAttestationError(["semantic_review_frame_evidence_invalid"])
        target_id = row.get("target_id")
        camera_id = row.get("camera_id")
        size = row.get("size_bytes")
        if (
            not _is_identifier(target_id)
            or not _is_identifier(camera_id)
            or not _is_digest(row.get("sha256"))
            or not _is_digest(row.get("decoded_rgb_sha256"))
            or type(size) is not int
            or size <= 0
            or (target_id, camera_id) in identities
        ):
            raise SemanticReviewAttestationError(["semantic_review_frame_evidence_invalid"])
        identities.add((target_id, camera_id))
        normalized.append(
            {
                "target_id": target_id,
                "camera_id": camera_id,
                "sha256": row["sha256"],
                "size_bytes": size,
                "decoded_rgb_sha256": row["decoded_rgb_sha256"],
            }
        )
    normalized.sort(key=lambda row: (row["target_id"], row["camera_id"]))
    return _sha256_bytes(
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _strict_json_object(data: bytes, *, error: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise SemanticReviewAttestationError([error]) from exc
    if not isinstance(parsed, dict):
        raise SemanticReviewAttestationError([error])
    return parsed


def _read_once_no_follow(
    path: str | os.PathLike[str], *, maximum_size: int, prefix: str
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(path, (str, os.PathLike)):
        raise SemanticReviewAttestationError([f"{prefix}_path_invalid"])
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise SemanticReviewAttestationError([f"{prefix}_no_follow_unavailable"])
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str) or not raw_path:
        raise SemanticReviewAttestationError([f"{prefix}_path_invalid"])
    display_path = os.path.abspath(raw_path)
    try:
        fd = os.open(display_path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise SemanticReviewAttestationError([f"{prefix}_open_failed"]) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise SemanticReviewAttestationError([f"{prefix}_not_regular"])
        if before.st_size <= 0 or before.st_size > maximum_size:
            raise SemanticReviewAttestationError([f"{prefix}_size_invalid"])
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(data) != before.st_size or identity_before != identity_after:
            raise SemanticReviewAttestationError([f"{prefix}_changed_during_read"])
        return (
            {
                "path": display_path,
                "size_bytes": len(data),
                "sha256": _sha256_bytes(data),
                "opened_once_no_follow": True,
            },
            data,
        )
    finally:
        os.close(fd)


def verify_semantic_review_attestation(
    *,
    attestation_path: str | os.PathLike[str],
    selection_contract_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify exact canonical files, signature, configured key, and selection.

    Trust-root and expected-binding overrides are intentionally absent.  All
    admitted bindings come from the two retained path-backed artifacts.
    """

    attestation_artifact, attestation_bytes = _read_once_no_follow(
        attestation_path,
        maximum_size=_MAX_ATTESTATION_BYTES,
        prefix="semantic_review_attestation_file",
    )
    selection_artifact, selection_bytes = _read_once_no_follow(
        selection_contract_path,
        maximum_size=_MAX_SELECTION_BYTES,
        prefix="semantic_authority_selection_file",
    )
    attestation = _strict_json_object(
        attestation_bytes, error="semantic_review_attestation_json_invalid"
    )
    selection = _strict_json_object(
        selection_bytes, error="semantic_authority_selection_json_invalid"
    )
    attestation_errors, normalized_attestation, fingerprint, cryptographically_valid = (
        _attestation_errors(attestation)
    )
    selection_errors, normalized_selection = _selection_errors(selection)
    errors = list(attestation_errors) + list(selection_errors)
    if normalized_attestation and (
        attestation_bytes != _canonical_bytes(normalized_attestation) + b"\n"
    ):
        errors.append("semantic_review_attestation_encoding_not_canonical")
    if normalized_selection and (selection_bytes != _canonical_bytes(normalized_selection) + b"\n"):
        errors.append("semantic_authority_selection_encoding_not_canonical")

    configured_fingerprint = os.getenv(TRUSTED_PUBLIC_KEY_SHA256_ENV)
    if not _is_digest(configured_fingerprint):
        errors.append("semantic_review_attestation_trust_root_not_configured")
    elif fingerprint != configured_fingerprint:
        errors.append("semantic_review_attestation_public_key_not_authorized")

    if normalized_attestation and normalized_selection:
        raw_payload = normalized_attestation.get("payload")
        raw_signature = normalized_attestation.get("signature")
        payload = raw_payload if isinstance(raw_payload, Mapping) else {}
        signature = raw_signature if isinstance(raw_signature, Mapping) else {}
        joins = {
            "selection_id": payload.get("selection_id"),
            "attestation_digest": normalized_attestation.get("attestation_digest"),
            "authority": {
                "authority_id": payload.get("authority_id"),
                "key_id": payload.get("authority_key_id"),
                "public_key_sha256": signature.get("public_key_sha256"),
            },
            "scene_id": payload.get("scene_id"),
            "source_target": payload.get("source_target"),
            "evidence": payload.get("evidence"),
            "learned_policy_outcomes_inspected": payload.get("learned_policy_outcomes_inspected"),
            "claim_scope": payload.get("claim_scope"),
        }
        for field, expected in joins.items():
            if normalized_selection.get(field) != expected:
                errors.append(f"semantic_authority_selection_{field}_mismatch")

    errors = sorted(set(errors))
    if errors:
        raise SemanticReviewAttestationError(errors)
    payload = normalized_attestation["payload"]
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "status": "verified",
        "semantic_authority_verified": True,
        "signature_cryptographically_valid": cryptographically_valid,
        "configured_authority_key_matched": True,
        "authority": normalized_selection["authority"],
        "attestation_digest": normalized_attestation["attestation_digest"],
        "selection_digest": normalized_selection["selection_digest"],
        "scene_id": payload["scene_id"],
        "source_target": payload["source_target"],
        "evidence": payload["evidence"],
        "learned_policy_outcomes_inspected": False,
        "semantic_assertions": payload["semantic_assertions"],
        "signed_attestation": normalized_attestation,
        "frozen_selection_contract": normalized_selection,
        "attestation_artifact": attestation_artifact,
        "selection_contract_artifact": selection_artifact,
        "claim_scope": CLAIM_SCOPE,
        "claim_boundary": payload["claim_boundary"],
        "does_not_establish": [
            "native_simulator_qualification",
            "contact_or_task_success",
            "physical_material_equivalence",
            "real_robot_performance",
        ],
    }


__all__ = [
    "ATTESTATION_SCHEMA_VERSION",
    "CLAIM_SCOPE",
    "SELECTION_SCHEMA_VERSION",
    "TRUSTED_PUBLIC_KEY_SHA256_ENV",
    "VERIFICATION_SCHEMA_VERSION",
    "SemanticReviewAttestationError",
    "canonical_semantic_authority_selection_bytes",
    "canonical_semantic_review_attestation_bytes",
    "materialize_semantic_authority_selection",
    "materialize_semantic_review_attestation",
    "materialize_semantic_review_payload",
    "semantic_frame_evidence_digest",
    "semantic_review_signature_message",
    "verify_semantic_review_attestation",
]
