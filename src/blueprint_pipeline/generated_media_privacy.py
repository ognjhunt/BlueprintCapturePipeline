"""Privacy, consent, and rights contract for world-model generated media.

Blueprint redacts capture walkthrough video: `privacy_processing` runs
segmentation and inpainting over the recorded frames and emits a verification
report. That control protects the *captured* pixels. It does not, and cannot,
protect pixels a generative model invents afterwards.

The gap this closes is specific. A world model conditioned on a site frame
produces new pixels. Those pixels can contain a face, a badge, a name on a
whiteboard, or a proprietary fixture -- either because the conditioning frame
still carried it, because inpainting over a redacted region reconstructed
something recognisable, or because the model simply synthesised a plausible
person into a workplace scene. The generated frame then flowed onward carrying
the source clip's rights metadata, into hosted sessions and customer-visible
artifacts.

That is privacy laundering through generation: the redaction evidence belonged
to a different pixel array, and inheriting it asserts a property nobody checked.

Three principles follow, and this module enforces all three:

1. **Redaction does not survive generation.** A generated artifact's redaction
   status starts as *unverified* regardless of how clean its source was. It is
   never inherited.
2. **Conditioning provenance is part of the control.** Generating from an
   unredacted source contaminates the output at the source, so conditioning
   assets must themselves be redaction-verified and consent-active.
3. **Consent is checked at generation time and again at release time**, and
   generated artifacts carry their source consent identifiers so a later
   revocation can reach them through the existing takedown enumeration.

Everything fails closed. Absent evidence is not permission, and the default
release scope is ``blocked``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import read_json_any, utc_now_iso, write_json
from .consent_normalization import resolve_consent_signals


CONTRACT_SCHEMA_VERSION = "generated_media_privacy_contract.v1"
VERIFICATION_SCHEMA_VERSION = "generated_media_redaction_verification.v1"

# Release scopes, ordered by how far the artifact may travel.
BLOCKED = "blocked"
INTERNAL_REVIEW_ONLY = "internal_review_only"
CUSTOMER_VISIBLE = "customer_visible"
RELEASE_SCOPES = (BLOCKED, INTERNAL_REVIEW_ONLY, CUSTOMER_VISIBLE)

# A conditioning asset is acceptable only when its own redaction pass passed.
ACCEPTED_REDACTION_STATUSES = frozenset({"passed", "verified", "completed"})
# Source kinds that are, by construction, pre-redaction pixels.
UNREDACTED_SOURCE_KINDS = frozenset(
    {"raw_capture", "raw_walkthrough", "unredacted_frame", "original_video"}
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _digest(value: Any) -> str:
    text = _string(value).lower().removeprefix("sha256:")
    return text if len(text) == 64 and all(c in "0123456789abcdef" for c in text) else ""


def file_digest(path: str | Path) -> str:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return ""
    hasher = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_generated_media_redaction_verification(
    *,
    artifact_sha256: str,
    verifier_id: str,
    status: str,
    detected_categories: Sequence[str] = (),
    reviewed_frame_count: int = 0,
    sampled_frame_count: int = 0,
    report_uri: str = "",
) -> dict[str, Any]:
    """Record a redaction pass performed over *generated* pixels.

    This is a distinct artifact from the capture-side verification report. It
    exists so that "these generated frames were checked" is a separate,
    separately-digested claim from "the source clip was checked".
    """

    blockers: list[str] = []
    if not _digest(artifact_sha256):
        blockers.append("generated_redaction_artifact_digest_invalid")
    if not _string(verifier_id):
        blockers.append("generated_redaction_verifier_missing")
    normalized_status = _string(status).lower()
    if normalized_status not in ACCEPTED_REDACTION_STATUSES | {"failed", "blocked"}:
        blockers.append("generated_redaction_status_unrecognized")
    if reviewed_frame_count <= 0:
        blockers.append("generated_redaction_reviewed_no_frames")
    categories = sorted({_string(item) for item in detected_categories if _string(item)})
    passed = normalized_status in ACCEPTED_REDACTION_STATUSES and not categories and not blockers
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "artifact_sha256": _digest(artifact_sha256) or None,
        "verifier_id": _string(verifier_id) or None,
        "status": normalized_status or None,
        "passed": passed,
        "detected_categories": categories,
        "reviewed_frame_count": int(reviewed_frame_count),
        "sampled_frame_count": int(sampled_frame_count or reviewed_frame_count),
        "report_uri": _string(report_uri) or None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "verification_covers_generated_pixels_only": True,
            "verification_is_not_inherited_from_source_capture": True,
        },
    }


def _evaluate_conditioning(assets: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[str]]:
    """Conditioning provenance: generating from unredacted pixels contaminates output."""

    blockers: list[str] = []
    evaluated: list[dict[str, Any]] = []
    if not assets:
        return [], ["generation_conditioning_assets_missing"]
    for index, raw in enumerate(assets):
        asset = _mapping(raw)
        asset_id = _string(asset.get("asset_id")) or f"asset_{index}"
        kind = _string(asset.get("kind")).lower()
        status = _string(asset.get("redaction_verification_status")).lower()
        report_digest = _digest(asset.get("redaction_report_sha256"))
        asset_digest = _digest(asset.get("sha256"))
        row = {
            "asset_id": asset_id,
            "kind": kind or None,
            "sha256": asset_digest or None,
            "redaction_verification_status": status or None,
            "redaction_report_sha256": report_digest or None,
            "accepted": False,
        }
        if not asset_digest:
            blockers.append(f"conditioning_asset_digest_invalid:{asset_id}")
        if kind in UNREDACTED_SOURCE_KINDS:
            blockers.append(f"conditioning_uses_unredacted_source:{asset_id}")
        if status not in ACCEPTED_REDACTION_STATUSES:
            blockers.append(f"conditioning_asset_not_redaction_verified:{asset_id}")
        if not report_digest:
            blockers.append(f"conditioning_asset_redaction_report_missing:{asset_id}")
        row["accepted"] = not any(asset_id in item for item in blockers)
        evaluated.append(row)
    return evaluated, blockers


def build_generated_media_privacy_contract(
    *,
    generation_id: str,
    scene_id: str,
    capture_id: str,
    conditioning_assets: Sequence[Mapping[str, Any]],
    generated_artifacts: Sequence[Mapping[str, Any]],
    consent_payload: Mapping[str, Any] | None = None,
    generated_redaction_verification: Mapping[str, Any] | None = None,
    requested_release_scope: str = INTERNAL_REVIEW_ONLY,
) -> dict[str, Any]:
    """Decide how far a piece of generated media may travel.

    The returned contract is the thing downstream surfaces consult. It never
    reports a scope wider than the evidence supports, and it records the
    source consent identifiers so a later revocation can find this derivative.
    """

    blockers: list[str] = []
    if not _string(generation_id):
        blockers.append("generated_media_generation_id_missing")
    if not _string(scene_id) or not _string(capture_id):
        # Without these the artifact cannot be found by takedown enumeration.
        blockers.append("generated_media_source_identity_missing")

    requested = _string(requested_release_scope).lower() or INTERNAL_REVIEW_ONLY
    if requested not in RELEASE_SCOPES:
        blockers.append("generated_media_requested_scope_invalid")
        requested = BLOCKED

    conditioning_rows, conditioning_blockers = _evaluate_conditioning(conditioning_assets)
    blockers.extend(conditioning_blockers)

    consent = resolve_consent_signals(_mapping(consent_payload))
    consent_state = _string(consent.get("state")).lower()
    consent_revoked = consent.get("consent_revoked")
    # "unknown" is not permission: an absent consent record fails closed here
    # exactly as a revoked one does, differing only in the reason reported.
    consent_active = consent_state == "active"
    if consent_revoked is True or consent_state == "revoked":
        blockers.append("source_consent_revoked")
    elif not consent_active:
        blockers.append(f"source_consent_not_active:{consent_state or 'unknown'}")

    artifacts: list[dict[str, Any]] = []
    for index, raw in enumerate(generated_artifacts):
        artifact = _mapping(raw)
        digest = _digest(artifact.get("sha256"))
        artifact_id = _string(artifact.get("artifact_id")) or f"generated_{index}"
        if not digest:
            blockers.append(f"generated_artifact_digest_invalid:{artifact_id}")
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "sha256": digest or None,
                "media_type": _string(artifact.get("media_type")) or None,
                # Stated explicitly on every row so no consumer can read the
                # source's clean status as applying here.
                "redaction_status_inherited_from_source": False,
            }
        )
    if not artifacts:
        blockers.append("generated_artifacts_missing")

    verification = _mapping(generated_redaction_verification)
    verification_passed = False
    if verification:
        if verification.get("schema_version") != VERIFICATION_SCHEMA_VERSION:
            blockers.append("generated_redaction_verification_schema_invalid")
        else:
            verified_digests = {_digest(verification.get("artifact_sha256"))}
            covered = {row["sha256"] for row in artifacts if row["sha256"]}
            if not covered <= verified_digests:
                # A verification that does not cover these exact bytes proves
                # nothing about them.
                blockers.append("generated_redaction_verification_artifact_mismatch")
            elif verification.get("passed") is True:
                verification_passed = True
            else:
                blockers.append("generated_redaction_verification_did_not_pass")

    # Scope resolution. Customer-visible release additionally requires a
    # redaction pass over the generated pixels themselves.
    if blockers:
        scope = BLOCKED
    elif requested == CUSTOMER_VISIBLE and not verification_passed:
        scope = INTERNAL_REVIEW_ONLY
        blockers.append("customer_visible_requires_generated_redaction_verification")
    else:
        scope = requested

    blockers = sorted(set(blockers))
    if blockers and scope == CUSTOMER_VISIBLE:
        scope = BLOCKED

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "generation_id": _string(generation_id) or None,
        "scene_id": _string(scene_id) or None,
        "capture_id": _string(capture_id) or None,
        "requested_release_scope": requested,
        "release_scope": scope,
        "customer_visible": scope == CUSTOMER_VISIBLE,
        "conditioning_assets": conditioning_rows,
        "generated_artifacts": artifacts,
        "generated_redaction_verification": verification or None,
        "generated_redaction_verified": verification_passed,
        "consent": {
            "consent_active": consent_active,
            "consent_state": consent_state or "unknown",
            "consent_revoked": consent_revoked,
            "consent_status": consent.get("consent_status"),
            "consent_revoked_at": consent.get("consent_revoked_at"),
        },
        # Carried so consent_takedown.enumerate_derived_artifacts can reach this
        # derivative when the source consent is later withdrawn.
        "takedown_keys": {
            "scene_id": _string(scene_id) or None,
            "capture_id": _string(capture_id) or None,
            "generation_id": _string(generation_id) or None,
            "generated_artifact_sha256": [
                row["sha256"] for row in artifacts if row["sha256"]
            ],
        },
        "blockers": blockers,
        "claim_boundary": {
            "redaction_status_is_not_inherited_from_source": True,
            "source_redaction_does_not_cover_generated_pixels": True,
            "generated_media_is_not_capture_truth": True,
            "contract_governs_release_scope_not_generation_quality": True,
            "public_claim_upgrade_allowed": False,
        },
    }


class GeneratedMediaReleaseError(RuntimeError):
    """Raised when generated media is served beyond its authorised scope."""


def assert_release_allowed(
    contract: Mapping[str, Any], *, required_scope: str = CUSTOMER_VISIBLE
) -> None:
    """Fail closed at a serving boundary.

    Callers serving generated media to a customer-visible surface call this
    immediately before handing over bytes. An absent or malformed contract is
    treated exactly like a denial.
    """

    if not isinstance(contract, Mapping) or contract.get("schema_version") != (
        CONTRACT_SCHEMA_VERSION
    ):
        raise GeneratedMediaReleaseError("generated_media_privacy_contract_missing")
    scope = _string(contract.get("release_scope"))
    if required_scope == CUSTOMER_VISIBLE and scope != CUSTOMER_VISIBLE:
        raise GeneratedMediaReleaseError(
            f"generated_media_not_cleared_for_customer_release:{scope or 'unknown'}"
        )
    if required_scope == INTERNAL_REVIEW_ONLY and scope == BLOCKED:
        raise GeneratedMediaReleaseError("generated_media_blocked")


def release_decision(
    contract: Mapping[str, Any], *, required_scope: str = CUSTOMER_VISIBLE
) -> dict[str, Any]:
    """Non-raising form of :func:`assert_release_allowed`."""

    try:
        assert_release_allowed(contract, required_scope=required_scope)
    except GeneratedMediaReleaseError as error:
        return {"allowed": False, "reason": str(error), "required_scope": required_scope}
    return {"allowed": True, "reason": None, "required_scope": required_scope}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a privacy/rights release contract for generated media"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    payload = _mapping(read_json_any(Path(args.input)))
    contract = build_generated_media_privacy_contract(
        generation_id=_string(payload.get("generation_id")),
        scene_id=_string(payload.get("scene_id")),
        capture_id=_string(payload.get("capture_id")),
        conditioning_assets=_rows(payload.get("conditioning_assets")),
        generated_artifacts=_rows(payload.get("generated_artifacts")),
        consent_payload=_mapping(payload.get("consent")),
        generated_redaction_verification=_mapping(
            payload.get("generated_redaction_verification")
        ),
        requested_release_scope=_string(payload.get("requested_release_scope"))
        or INTERNAL_REVIEW_ONLY,
    )
    write_json(Path(args.output), contract)
    print(
        json.dumps(
            {"path": args.output, "release_scope": contract["release_scope"]}, sort_keys=True
        )
    )
    return 0 if contract["release_scope"] != BLOCKED else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
