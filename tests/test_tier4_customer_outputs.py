"""Tests for the Tier 4 customer-output contracts.

Covers the anchor return kit, frozen held-out splits, signed delivery bundles,
per-site difficulty profiles, and the generated-media privacy contract.
"""

from __future__ import annotations

import hashlib

import pytest

from blueprint_pipeline import generated_media_privacy as gmp
from blueprint_pipeline.post_training_holdout_split import (
    build_holdout_split,
    check_package_against_split,
)
from blueprint_pipeline.signed_delivery_bundle import (
    attach_delivery_integrity,
    build_delivery_bundle,
    verify_delivery_bundle,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


# -- 16: anchor return kit ------------------------------------------------


# -- 17: frozen held-out split -------------------------------------------


def _clips(count: int = 40):
    return [
        {"clip_id": f"clip-{index:03d}", "task": "pick" if index % 2 else "place"}
        for index in range(count)
    ]


def test_split_is_disjoint_deterministic_and_order_independent() -> None:
    first = build_holdout_split(split_id="s", clips=_clips(), holdout_fraction=0.2)
    second = build_holdout_split(
        split_id="s", clips=list(reversed(_clips())), holdout_fraction=0.2
    )

    assert first["status"] == "frozen"
    assert first["partitions_disjoint"] is True
    assert not set(first["train_clip_ids"]) & set(first["holdout_clip_ids"])
    # Order-independent, so the same capture always yields the same split.
    assert first["split_sha256"] == second["split_sha256"]


def test_split_id_changes_the_partition() -> None:
    first = build_holdout_split(split_id="a", clips=_clips(), holdout_fraction=0.2)
    other = build_holdout_split(split_id="b", clips=_clips(), holdout_fraction=0.2)
    assert first["holdout_clip_ids"] != other["holdout_clip_ids"]


def test_stratification_preserves_task_balance() -> None:
    split = build_holdout_split(
        split_id="s", clips=_clips(40), holdout_fraction=0.25, stratify_by=["task"]
    )
    holdout = set(split["holdout_clip_ids"])
    picks = sum(1 for index in range(40) if index % 2 and f"clip-{index:03d}" in holdout)
    places = sum(1 for index in range(40) if not index % 2 and f"clip-{index:03d}" in holdout)
    assert picks > 0 and places > 0, "both strata must be represented in the holdout"


def test_training_payload_containing_holdout_clips_fails_closed() -> None:
    """The defect this closes: a buyer evaluating on clips they were sold."""

    split = build_holdout_split(split_id="s", clips=_clips(), holdout_fraction=0.2)
    leaked = check_package_against_split(
        split=split,
        training_clip_ids=split["train_clip_ids"] + split["holdout_clip_ids"][:3],
    )

    assert leaked["status"] == "blocked"
    assert "training_payload_contains_holdout_clips" in leaked["blockers"]
    assert leaked["leaked_clip_count"] == 3

    clean = check_package_against_split(split=split, training_clip_ids=split["train_clip_ids"])
    assert clean["status"] == "clean"


def test_clips_outside_the_frozen_split_are_not_assumed_safe() -> None:
    split = build_holdout_split(split_id="s", clips=_clips(), holdout_fraction=0.2)
    report = check_package_against_split(
        split=split, training_clip_ids=split["train_clip_ids"] + ["clip-999"]
    )
    assert "training_payload_contains_clips_outside_the_frozen_split" in report["blockers"]


def test_tiny_capture_cannot_produce_a_meaningful_holdout() -> None:
    split = build_holdout_split(split_id="s", clips=_clips(4), holdout_fraction=0.2)
    assert split["status"] == "blocked"
    assert any(item.startswith("holdout_partition_below_minimum") for item in split["blockers"])


# -- 18: signed delivery bundle ------------------------------------------


def test_bundle_requires_a_digest_for_every_member() -> None:
    """A URI says where bytes were, not which bytes they were."""

    bundle = build_delivery_bundle(
        root_id="r1",
        root_kind="site_package",
        members=[{"member_id": "scene.usd", "uri": "gs://bucket/scene.usd"}],
    )

    assert bundle["status"] == "blocked"
    assert any(item.startswith("delivery_member_digest_missing") for item in bundle["blockers"])


def test_bundle_seals_and_verifies() -> None:
    bundle = build_delivery_bundle(
        root_id="r1",
        root_kind="site_package",
        members=[
            {"member_id": "scene.usd", "uri": "gs://b/scene.usd", "sha256": _d("scene")},
            {"member_id": "route.json", "uri": "gs://b/route.json", "sha256": _d("route")},
        ],
        scene_id="s1",
        capture_id="c1",
    )

    assert bundle["status"] == "sealed"
    assert bundle["signed"] is False
    assert verify_delivery_bundle(bundle)["status"] == "verified"


def test_root_digest_covers_the_member_set() -> None:
    """Removing a member must change the root digest, not just editing one."""

    members = [
        {"member_id": "a", "uri": "gs://b/a", "sha256": _d("a")},
        {"member_id": "b", "uri": "gs://b/b", "sha256": _d("b")},
    ]
    full = build_delivery_bundle(root_id="r", root_kind="k", members=members)
    fewer = build_delivery_bundle(root_id="r", root_kind="k", members=members[:1])
    assert full["root_sha256"] != fewer["root_sha256"]


def test_verification_detects_a_tampered_checksum() -> None:
    bundle = build_delivery_bundle(
        root_id="r",
        root_kind="k",
        members=[{"member_id": "a", "uri": "gs://b/a", "sha256": _d("a")}],
    )
    tampered = {**bundle, "checksums": {"a": _d("evil")}}
    report = verify_delivery_bundle(tampered)

    assert report["status"] == "blocked"
    assert any("checksum_disagrees" in item for item in report["blockers"])


def test_local_files_are_digested_and_mismatches_rejected(tmp_path) -> None:
    member_file = tmp_path / "payload.bin"
    member_file.write_bytes(b"hello")
    good = build_delivery_bundle(
        root_id="r",
        root_kind="k",
        members=[{"member_id": "payload.bin", "local_path": str(member_file)}],
    )
    assert good["status"] == "sealed"
    assert good["members"][0]["size_bytes"] == 5

    mismatch = build_delivery_bundle(
        root_id="r",
        root_kind="k",
        members=[
            {"member_id": "payload.bin", "local_path": str(member_file), "sha256": _d("other")}
        ],
    )
    assert any("digest_mismatch" in item for item in mismatch["blockers"])


def test_attach_integrity_surfaces_missing_digests_in_the_manifest() -> None:
    bundle = attach_delivery_integrity(
        root_id="site_package:s1:c1",
        root_kind="site_package",
        artifact_uris={"scene": "gs://b/scene.usd", "route": "gs://b/route.json"},
        artifact_digests={"scene": _d("scene")},
    )
    assert bundle["status"] == "blocked"
    assert "delivery_member_digest_missing:route" in bundle["blockers"]


# -- 20: site difficulty profile -----------------------------------------


# -- 21: generated media privacy -----------------------------------------


def _conditioning():
    return [
        {
            "asset_id": "frame-0",
            "kind": "redacted_frame",
            "sha256": _d("frame"),
            "redaction_verification_status": "passed",
            "redaction_report_sha256": _d("report"),
        }
    ]


def _artifacts():
    return [{"artifact_id": "clip", "sha256": _d("clip"), "media_type": "video/mp4"}]


def _contract(**overrides):
    kwargs = {
        "generation_id": "gen-1",
        "scene_id": "s1",
        "capture_id": "c1",
        "conditioning_assets": _conditioning(),
        "generated_artifacts": _artifacts(),
        "consent_payload": {"consent_status": "granted"},
        "requested_release_scope": gmp.CUSTOMER_VISIBLE,
    }
    kwargs.update(overrides)
    return gmp.build_generated_media_privacy_contract(**kwargs)


def _passing_verification():
    return gmp.build_generated_media_redaction_verification(
        artifact_sha256=_d("clip"),
        verifier_id="vip",
        status="passed",
        reviewed_frame_count=60,
    )


def test_generated_media_needs_its_own_redaction_pass() -> None:
    """Source redaction does not cover pixels the model invented."""

    contract = _contract()

    assert contract["release_scope"] == gmp.INTERNAL_REVIEW_ONLY
    assert (
        "customer_visible_requires_generated_redaction_verification" in contract["blockers"]
    )
    assert contract["claim_boundary"]["redaction_status_is_not_inherited_from_source"] is True
    assert contract["generated_artifacts"][0]["redaction_status_inherited_from_source"] is False


def test_verified_generated_media_may_reach_a_customer() -> None:
    contract = _contract(generated_redaction_verification=_passing_verification())

    assert contract["release_scope"] == gmp.CUSTOMER_VISIBLE
    assert contract["blockers"] == []
    gmp.assert_release_allowed(contract)


def test_conditioning_on_unredacted_capture_is_refused() -> None:
    raw = [
        {
            "asset_id": "raw",
            "kind": "raw_capture",
            "sha256": _d("raw"),
            "redaction_verification_status": "passed",
            "redaction_report_sha256": _d("r"),
        }
    ]
    contract = _contract(
        conditioning_assets=raw, generated_redaction_verification=_passing_verification()
    )

    assert contract["release_scope"] == gmp.BLOCKED
    assert "conditioning_uses_unredacted_source:raw" in contract["blockers"]


def test_unverified_conditioning_is_refused() -> None:
    unverified = [
        {
            "asset_id": "frame-0",
            "kind": "redacted_frame",
            "sha256": _d("frame"),
            "redaction_verification_status": "pending",
            "redaction_report_sha256": _d("report"),
        }
    ]
    contract = _contract(conditioning_assets=unverified)
    assert "conditioning_asset_not_redaction_verified:frame-0" in contract["blockers"]


def test_revoked_and_unknown_consent_both_block() -> None:
    revoked = _contract(
        consent_payload={"consent_status": "revoked"},
        generated_redaction_verification=_passing_verification(),
    )
    assert revoked["release_scope"] == gmp.BLOCKED
    assert "source_consent_revoked" in revoked["blockers"]

    # Absent consent is not permission.
    unknown = _contract(
        consent_payload={}, generated_redaction_verification=_passing_verification()
    )
    assert unknown["release_scope"] == gmp.BLOCKED
    assert any(item.startswith("source_consent_not_active") for item in unknown["blockers"])


def test_verification_must_cover_these_exact_bytes() -> None:
    """A pass for a different artifact proves nothing about this one."""

    wrong = gmp.build_generated_media_redaction_verification(
        artifact_sha256=_d("some-other-clip"),
        verifier_id="vip",
        status="passed",
        reviewed_frame_count=60,
    )
    contract = _contract(generated_redaction_verification=wrong)

    assert contract["release_scope"] == gmp.BLOCKED
    assert "generated_redaction_verification_artifact_mismatch" in contract["blockers"]


def test_detected_pii_fails_the_generated_verification() -> None:
    verification = gmp.build_generated_media_redaction_verification(
        artifact_sha256=_d("clip"),
        verifier_id="vip",
        status="passed",
        detected_categories=["face"],
        reviewed_frame_count=60,
    )
    assert verification["passed"] is False
    contract = _contract(generated_redaction_verification=verification)
    assert contract["release_scope"] == gmp.BLOCKED


def test_contract_carries_takedown_keys_for_derivatives() -> None:
    contract = _contract(generated_redaction_verification=_passing_verification())
    keys = contract["takedown_keys"]

    assert keys["scene_id"] == "s1"
    assert keys["capture_id"] == "c1"
    assert _d("clip") in keys["generated_artifact_sha256"]


def test_serving_boundary_fails_closed_without_a_contract() -> None:
    with pytest.raises(gmp.GeneratedMediaReleaseError, match="contract_missing"):
        gmp.assert_release_allowed({})

    blocked = _contract(consent_payload={"consent_status": "revoked"})
    with pytest.raises(gmp.GeneratedMediaReleaseError, match="not_cleared"):
        gmp.assert_release_allowed(blocked)

    assert gmp.release_decision({})["allowed"] is False
