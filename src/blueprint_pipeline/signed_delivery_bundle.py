"""One integrity contract for every customer delivery root.

The Post-Training Data Package earned real integrity machinery: a
``package_index.json`` naming every member, a ``checksums.json`` giving each
member's digest, and a ``package_root_signature.json`` that signs the whole set
and is itself the one deliberately self-excluded member. A buyer can recompute
every digest independently and detect a single flipped byte.

The other customer-facing roots did not get that. ``build_site_package_manifest``
and its siblings accept ``artifact_uris`` -- a bare mapping of names to URIs.
A URI is a location, not a commitment: it says where bytes were, not which bytes
they were. Anything that can be swapped, truncated, re-uploaded, or silently
regenerated between manifest time and download time is undetectable to the
recipient, and two customers handed "the same" package have no way to establish
that they received the same thing.

This module lifts the PTDP pattern into something any delivery root can use:

* every member carries a digest and a byte size, and a member without a digest
  is a blocker rather than a warning -- an unverifiable member makes the whole
  bundle unverifiable;
* the root digest covers the member set, so adding or removing a member changes
  it, not just editing one;
* the signature is computed over the root digest and excludes itself, because a
  signature cannot cover its own bytes; and
* verification recomputes rather than re-reads, so a tampered ``checksums`` file
  fails instead of being believed.

Signing is optional; a bundle without a signature is still digest-verifiable and
says so. What is not optional is knowing which bytes were promised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json


BUNDLE_SCHEMA_VERSION = "signed_delivery_bundle.v1"
VERIFICATION_SCHEMA_VERSION = "signed_delivery_bundle_verification.v1"

SIGNING_KEY_FILE_ENV = "BLUEPRINT_DELIVERY_BUNDLE_SIGNING_PRIVATE_KEY_FILE"
SIGNING_KEY_ID_ENV = "BLUEPRINT_DELIVERY_BUNDLE_SIGNING_KEY_ID"

# The member that signs the others cannot also be signed by them.
ROOT_SIGNATURE_MEMBER = "delivery_root_signature.json"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _digest_text(value: Any) -> str:
    text = _string(value).lower().removeprefix("sha256:")
    return text if len(text) == 64 and all(c in "0123456789abcdef" for c in text) else ""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return ""
    hasher = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_delivery_bundle(
    *,
    root_id: str,
    root_kind: str,
    members: Sequence[Mapping[str, Any]],
    scene_id: str = "",
    capture_id: str = "",
    sign: bool = False,
) -> dict[str, Any]:
    """Assemble the index + checksums + root digest for one delivery root.

    ``members`` are ``{member_id, uri, sha256?, size_bytes?, local_path?}``.
    A ``local_path`` is digested here; otherwise the caller must supply the
    digest, because a remote URI cannot be verified without fetching it and
    pretending otherwise would be the exact failure this module exists to stop.
    """

    blockers: list[str] = []
    if not _string(root_id):
        blockers.append("delivery_root_id_missing")
    if not _string(root_kind):
        blockers.append("delivery_root_kind_missing")

    indexed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(members):
        member = _mapping(raw)
        member_id = _string(member.get("member_id")) or f"member_{index}"
        if member_id in seen:
            blockers.append(f"delivery_member_duplicate:{member_id}")
            continue
        seen.add(member_id)

        local_path = _string(member.get("local_path"))
        digest = _digest_text(member.get("sha256"))
        size_bytes = member.get("size_bytes")
        if local_path:
            resolved = Path(local_path).expanduser()
            if resolved.is_file():
                computed = file_sha256(resolved)
                if digest and digest != computed:
                    blockers.append(f"delivery_member_digest_mismatch:{member_id}")
                digest = computed
                size_bytes = resolved.stat().st_size
            else:
                blockers.append(f"delivery_member_local_path_missing:{member_id}")

        if not digest:
            # A bare URI is a location, not a commitment.
            blockers.append(f"delivery_member_digest_missing:{member_id}")
        uri = _string(member.get("uri"))
        if not uri and not local_path:
            blockers.append(f"delivery_member_uri_missing:{member_id}")

        indexed.append(
            {
                "member_id": member_id,
                "uri": uri or None,
                "sha256": digest or None,
                "size_bytes": int(size_bytes) if isinstance(size_bytes, int) else None,
                "media_type": _string(member.get("media_type")) or None,
            }
        )

    if not indexed:
        blockers.append("delivery_bundle_has_no_members")

    checksums = {row["member_id"]: row["sha256"] for row in indexed}
    index_core = {
        "root_id": _string(root_id),
        "root_kind": _string(root_kind),
        "scene_id": _string(scene_id),
        "capture_id": _string(capture_id),
        "members": indexed,
    }
    # Covers the member SET, so adding or removing a member changes the root
    # digest rather than only editing one member's bytes.
    root_sha256 = canonical_sha256(index_core)

    signature = None
    if sign:
        signature, signing_blockers = _sign_root(root_sha256)
        blockers.extend(signing_blockers)

    blockers = sorted(set(blockers))
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "sealed" if not blockers else "blocked",
        **index_core,
        "member_count": len(indexed),
        "checksums": checksums,
        "root_sha256": root_sha256,
        "root_signature": signature,
        "signed": bool(signature),
        "root_signature_member": ROOT_SIGNATURE_MEMBER,
        "verification_instructions": [
            "Recompute SHA256 for every member and compare against `checksums`.",
            "Recompute the canonical digest of the index and compare against `root_sha256`.",
            f"`{ROOT_SIGNATURE_MEMBER}` is self-excluded and is not covered by `checksums`.",
        ],
        "blockers": blockers,
        "claim_boundary": {
            "integrity_proves_bytes_not_content_quality": True,
            "a_uri_without_a_digest_is_not_a_commitment": True,
            "unsigned_bundles_are_digest_verifiable_only": not bool(signature),
        },
    }


def _sign_root(root_sha256: str) -> tuple[dict[str, Any] | None, list[str]]:
    key_path = _string(os.getenv(SIGNING_KEY_FILE_ENV))
    if not key_path:
        return None, ["delivery_bundle_signing_key_not_configured"]
    candidate = Path(key_path).expanduser()
    if not candidate.is_file():
        return None, ["delivery_bundle_signing_key_missing"]
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = serialization.load_pem_private_key(candidate.read_bytes(), password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            return None, ["delivery_bundle_signing_key_not_ed25519"]
        signature = private_key.sign(root_sha256.encode("utf-8"))
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    except Exception:  # noqa: BLE001 - a signing failure must not look like success
        return None, ["delivery_bundle_signing_failed"]
    return (
        {
            "algorithm": "ed25519",
            "key_id": _string(os.getenv(SIGNING_KEY_ID_ENV)) or "delivery-bundle-signer",
            "signed_value": "root_sha256",
            "signature_hex": signature.hex(),
            "public_key_sha256": hashlib.sha256(public_bytes).hexdigest(),
        },
        [],
    )


def verify_delivery_bundle(
    bundle: Mapping[str, Any], *, local_root: str | Path | None = None
) -> dict[str, Any]:
    """Recompute a bundle's integrity rather than re-reading its claims."""

    blockers: list[str] = []
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        blockers.append("delivery_bundle_schema_missing_or_unsupported")

    members = [dict(row) for row in bundle.get("members") or [] if isinstance(row, Mapping)]
    index_core = {
        "root_id": _string(bundle.get("root_id")),
        "root_kind": _string(bundle.get("root_kind")),
        "scene_id": _string(bundle.get("scene_id")),
        "capture_id": _string(bundle.get("capture_id")),
        "members": members,
    }
    recomputed_root = canonical_sha256(index_core)
    if recomputed_root != _digest_text(bundle.get("root_sha256")):
        blockers.append("delivery_bundle_root_digest_mismatch")

    declared = _mapping(bundle.get("checksums"))
    for row in members:
        member_id = _string(row.get("member_id"))
        if declared.get(member_id) != row.get("sha256"):
            blockers.append(f"delivery_bundle_checksum_disagrees_with_index:{member_id}")

    verified_locally: list[str] = []
    if local_root is not None:
        root = Path(local_root).expanduser()
        for row in members:
            member_id = _string(row.get("member_id"))
            candidate = root / member_id
            if not candidate.is_file():
                continue
            if file_sha256(candidate) != _digest_text(row.get("sha256")):
                blockers.append(f"delivery_member_bytes_do_not_match_digest:{member_id}")
            else:
                verified_locally.append(member_id)

    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "verified" if not blockers else "blocked",
        "root_id": bundle.get("root_id"),
        "recomputed_root_sha256": recomputed_root,
        "declared_root_sha256": bundle.get("root_sha256"),
        "member_count": len(members),
        "locally_verified_members": sorted(verified_locally),
        "signed": bool(bundle.get("root_signature")),
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "verification_proves_bytes_not_fitness_for_purpose": True,
            "local_verification_covers_only_members_present_on_disk": True,
        },
    }


def attach_delivery_integrity(
    *,
    root_id: str,
    root_kind: str,
    artifact_uris: Mapping[str, Any] | None,
    artifact_digests: Mapping[str, Any] | None = None,
    scene_id: str = "",
    capture_id: str = "",
) -> dict[str, Any]:
    """Turn a bare ``artifact_uris`` mapping into a verifiable bundle.

    The adapter used by manifest builders that historically shipped locations
    without commitments. Missing digests surface as blockers so the gap is
    visible in the manifest rather than invisible to the recipient.
    """

    digests = _mapping(artifact_digests)
    members = [
        {
            "member_id": _string(name),
            "uri": _string(uri) if isinstance(uri, str) else None,
            "sha256": digests.get(_string(name)),
        }
        for name, uri in _mapping(artifact_uris).items()
    ]
    return build_delivery_bundle(
        root_id=root_id,
        root_kind=root_kind,
        members=members,
        scene_id=scene_id,
        capture_id=capture_id,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal or verify a customer delivery bundle")
    sub = parser.add_subparsers(dest="command", required=True)

    seal = sub.add_parser("seal")
    seal.add_argument("--input", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--sign", action="store_true")
    seal.set_defaults(kind="seal")

    verify = sub.add_parser("verify")
    verify.add_argument("--input", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--local-root", default=None)
    verify.set_defaults(kind="verify")

    args = parser.parse_args(argv)
    if args.kind == "seal":
        payload = _mapping(read_json_any(Path(args.input)))
        result = build_delivery_bundle(
            root_id=_string(payload.get("root_id")),
            root_kind=_string(payload.get("root_kind")),
            members=[row for row in payload.get("members") or [] if isinstance(row, Mapping)],
            scene_id=_string(payload.get("scene_id")),
            capture_id=_string(payload.get("capture_id")),
            sign=bool(args.sign),
        )
        ok = result["status"] == "sealed"
    else:
        result = verify_delivery_bundle(
            _mapping(read_json_any(Path(args.input))), local_root=args.local_root
        )
        ok = result["status"] == "verified"

    write_json(Path(args.output), result)
    print(json.dumps({"path": args.output, "status": result["status"]}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
