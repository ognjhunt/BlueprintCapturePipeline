"""Cross-surface fail-closed invariants for capture-rights and consent values.

The consent contract flows raw manifest → materialization → qualification →
generated-asset gates → PTDP packaging → buyer readout. These tests assert the
single invariant every surface must share: a malformed, wrong-typed, unknown,
or contradictory consent value can only ever DOWNGRADE a permission or claim,
never upgrade it — and all surfaces agree on the same hostile input.

Local fixture-based only: consent sources are written inline under tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import materialization
from blueprint_pipeline import post_training_data_package as ptdp
from blueprint_pipeline import qualification
from blueprint_pipeline.consent_normalization import (
    CONSENT_ACTIVE_STATUSES,
    CONSENT_REVOKED_STATUSES,
    resolve_consent_signals,
    restrictive_scope_list,
    strict_allow_bool,
)
from blueprint_pipeline.consent_takedown import (
    evaluate_delivery_time_takedown_gate,
    read_consent_state,
)
from blueprint_pipeline.proof_contracts import build_rights_provenance_review


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _capture_root(tmp_path: Path, rights_consent: object) -> Path:
    root = tmp_path / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(root / "raw" / "rights_consent.json", rights_consent)
    return root


# Hostile allow-flag values that must never grant.
DENYING_ALLOW_VALUES = [
    "false",
    "no",
    "denied",
    "0",
    "off",
    "FALSE",
    ["true"],
    {"allowed": True},
    "",
    None,
    0,
    "maybe",
]

# Consent statuses that must never read as active anywhere.
NON_ACTIVE_STATUSES = ["denied", "pending", "refused", "expired", "unknown", "REVOKED "]


# ---------------------------------------------------------------------------
# Normalizer unit invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", DENYING_ALLOW_VALUES)
def test_strict_allow_bool_never_grants_on_hostile_values(value: object) -> None:
    assert strict_allow_bool(value) is False


def test_strict_allow_bool_grants_only_explicit_true() -> None:
    assert strict_allow_bool(True) is True
    assert strict_allow_bool("true") is True
    assert strict_allow_bool("1") is True


@pytest.mark.parametrize(
    "payload",
    [
        # camelCase revocation, snake_case active status
        {"consent_status": "granted", "consentRevoked": True},
        # revocation timestamp in either spelling
        {"consent_status": "granted", "consentRevokedAt": "2026-01-01T00:00:00Z"},
        {"consent_status": "granted", "consent_revoked_at": "2026-01-01T00:00:00Z"},
        # contradictory duplicate spellings resolve to the revoked side
        {"consent_status": "granted", "consentStatus": "revoked"},
        # nested block cannot shadow a top-level revocation
        {"consent_revoked": True, "rights": {"consent_status": "granted"}},
        # top-level revocation with a nested capture_rights block present
        {
            "consentRevoked": "revoked",
            "capture_rights": {"consent_status": "documented"},
        },
        # wrong-typed revocation container fails toward revocation
        {"consent_status": "granted", "consent_revoked": {"value": False}},
    ],
)
def test_resolver_revocation_cannot_be_shadowed(payload: dict) -> None:
    signals = resolve_consent_signals(payload)
    assert signals["consent_revoked"] is True
    assert signals["state"] == "revoked"


@pytest.mark.parametrize("status", NON_ACTIVE_STATUSES)
def test_resolver_unknown_status_never_active(status: str) -> None:
    signals = resolve_consent_signals({"consent_status": status})
    assert signals["state"] != "active"


@pytest.mark.parametrize("value", [{"v": "granted"}, ["granted"], 5, True])
def test_resolver_wrong_typed_status_never_active(value: object) -> None:
    signals = resolve_consent_signals({"consent_status": value})
    assert signals["state"] != "active"


def test_resolver_contradictory_active_statuses_are_not_active() -> None:
    signals = resolve_consent_signals(
        {"consent_status": "granted", "consentStatus": "denied"}
    )
    assert signals["state"] == "unknown"


def test_resolver_malformed_revocation_flag_blocks_active() -> None:
    # "maybe" is unintelligible: it cannot prove revocation, but it must also
    # not leave an otherwise-active status readable as a clean grant.
    signals = resolve_consent_signals(
        {"consent_status": "granted", "consent_revoked": "maybe"}
    )
    assert signals["state"] == "unknown"
    assert signals["malformed_fields"]


def test_restrictive_scope_list_intersects_contradictory_spellings() -> None:
    assert restrictive_scope_list(
        ["model_training", "robot_evaluation"], ["robot_evaluation"]
    ) == ["robot_evaluation"]
    assert restrictive_scope_list({"scope": "model_training"}) == []
    assert restrictive_scope_list(None, ["model_training"]) == ["model_training"]


# ---------------------------------------------------------------------------
# Materialization: raw manifest → materialized capture_rights
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", DENYING_ALLOW_VALUES)
def test_materialization_never_launders_allow_flag(value: object) -> None:
    block = materialization._capture_rights_block(
        {"capture_rights": {"derived_scene_generation_allowed": value}}
    )
    assert block["derived_scene_generation_allowed"] is False
    assert (
        materialization._derived_scene_generation_allowed(
            {"capture_rights": {"derived_scene_generation_allowed": value}}
        )
        is False
    )


def test_materialization_revocation_overrides_allow_flag() -> None:
    manifest = {
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "data_licensing_allowed": True,
            "consent_status": "revoked",
        }
    }
    block = materialization._capture_rights_block(manifest)
    assert block["derived_scene_generation_allowed"] is False
    assert block["data_licensing_allowed"] is False
    assert block["consent_revoked"] is True
    assert materialization._derived_scene_generation_allowed(manifest) is False


def test_materialization_carries_revocation_fields() -> None:
    block = materialization._capture_rights_block(
        {
            "capture_rights": {
                "derived_scene_generation_allowed": True,
                "consentRevokedAt": "2026-01-01T00:00:00Z",
            }
        }
    )
    assert block["consent_revoked"] is True
    assert block["consent_revoked_at"] == "2026-01-01T00:00:00Z"


def test_materialization_downgrade_reason_names_revocation() -> None:
    manifest = {
        "site_identity": {"site_id": "site-1", "site_id_source": "operator"},
        "capture_mode": {"requested_mode": "site_world_candidate"},
        "capture_rights": {
            "derived_scene_generation_allowed": True,
            "consent_status": "revoked",
        },
    }
    reason = materialization._world_model_candidate_downgrade_reason(
        manifest=manifest,
        arkit_poses_uri="gs://b/poses.json",
        arkit_intrinsics_uri="gs://b/intrinsics.json",
        arkit_depth_prefix_uri="gs://b/depth/",
        intake_complete=True,
        capture_source="iphone",
        pose_match_rate=1.0,
        p95_pose_delta_sec=0.001,
        pose_alignment_valid=True,
        geometry_ready=True,
    )
    assert reason == "consent_revoked_takedown_required"


# ---------------------------------------------------------------------------
# Qualification: parsed rights block agrees with materialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", DENYING_ALLOW_VALUES)
def test_qualification_and_materialization_agree_on_hostile_flags(
    value: object,
) -> None:
    metadata = {"capture_rights": {"derived_scene_generation_allowed": value}}
    qual = qualification._capture_rights(metadata)
    mat = materialization._capture_rights_block(metadata)
    assert qual["derived_scene_generation_allowed"] is False
    assert (
        qual["derived_scene_generation_allowed"]
        == mat["derived_scene_generation_allowed"]
    )


def test_qualification_revocation_overrides_allow_flags() -> None:
    rights = qualification._capture_rights(
        {
            "capture_rights": {
                "derived_scene_generation_allowed": True,
                "data_licensing_allowed": True,
                "consentRevoked": True,
            }
        }
    )
    assert rights["derived_scene_generation_allowed"] is False
    assert rights["data_licensing_allowed"] is False
    assert rights["consent_revoked"] is True


# ---------------------------------------------------------------------------
# Rights-provenance review (generated-asset gate input)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rights",
    [
        {"consentRevoked": True, "consent_status": "documented"},
        {"consentRevokedAt": "2026-01-01T00:00:00Z", "consent_status": "documented"},
        {"consentStatus": "revoked", "consent_status": "documented"},
    ],
)
def test_rights_review_blocks_camelcase_revocation(rights: dict) -> None:
    review = build_rights_provenance_review(
        rights_summary={
            "derived_scene_generation_allowed": True,
            "permission_document_uri": "s3://consent/doc.pdf",
            **rights,
        },
        privacy_processing=None,
        provenance_summary=None,
        site_identity=None,
        adjacent_systems=None,
    )
    assert review["rights"]["consent_revoked"] is True
    assert review["rights"]["status"] == "blocked"
    assert review["status"] == "blocked"


def test_rights_review_scope_intersection_blocks_contradictory_spelling() -> None:
    review = build_rights_provenance_review(
        rights_summary={
            "derived_scene_generation_allowed": True,
            "consent_status": "documented",
            "permission_document_uri": "s3://consent/doc.pdf",
            "consent_scope": ["model_training", "robot_evaluation"],
            "consentScope": ["robot_evaluation"],
        },
        privacy_processing=None,
        provenance_summary=None,
        site_identity=None,
        adjacent_systems=None,
        required_use_classes=["model_training"],
    )
    # model_training is granted by only one spelling → not granted.
    assert review["rights"]["status"] == "blocked"


# ---------------------------------------------------------------------------
# Delivery-time consent gate (consent_takedown)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", NON_ACTIVE_STATUSES)
def test_delivery_gate_blocks_unknown_status(tmp_path: Path, status: str) -> None:
    root = _capture_root(tmp_path, {"consent_status": status})
    state = read_consent_state(root)
    assert state["state"] != "active"
    gate = evaluate_delivery_time_takedown_gate(capture_root=root, surface="test")
    assert gate["serve_allowed"] is False


def test_delivery_gate_blocks_consent_file_without_consent_fields(
    tmp_path: Path,
) -> None:
    # A rights_consent.json that exists but proves nothing must not serve.
    root = _capture_root(tmp_path, {"note": "todo"})
    state = read_consent_state(root)
    assert state["state"] == "unknown"
    gate = evaluate_delivery_time_takedown_gate(capture_root=root, surface="test")
    assert gate["serve_allowed"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"consent_status": {"value": "granted"}},
        {"consent_status": ["granted"]},
        {"consent_status": "granted", "consentStatus": "revoked"},
        {"consent_revoked": True, "rights": {"consent_status": "granted"}},
    ],
)
def test_delivery_gate_blocks_malformed_or_contradictory_payloads(
    tmp_path: Path, payload: dict
) -> None:
    root = _capture_root(tmp_path, payload)
    gate = evaluate_delivery_time_takedown_gate(capture_root=root, surface="test")
    assert gate["serve_allowed"] is False


def test_delivery_gate_still_allows_clean_active_consent(tmp_path: Path) -> None:
    root = _capture_root(
        tmp_path,
        {
            "consent_status": "granted",
            "consent_scope": ["model_training", "robot_evaluation"],
        },
    )
    gate = evaluate_delivery_time_takedown_gate(capture_root=root, surface="test")
    assert gate["serve_allowed"] is True


# ---------------------------------------------------------------------------
# PTDP packaging
# ---------------------------------------------------------------------------


def _consent_evidence(tmp_path: Path, rights_consent: object, rights_packet=None):
    root = _capture_root(tmp_path, rights_consent)
    return ptdp._build_consent_evidence_record(
        capture_root=root,
        output_dir=tmp_path / "out",
        rights_packet=rights_packet or {},
        scene_id="scene-1",
        capture_id="capture-1",
        generated_at="2026-07-11T00:00:00+00:00",
    )


def test_ptdp_blocks_top_level_revocation_hidden_by_nested_block(
    tmp_path: Path,
) -> None:
    record = _consent_evidence(
        tmp_path,
        {
            "consent_revoked": True,
            "rights": {
                "consent_status": "granted",
                "consent_scope": ["model_training", "robot_evaluation"],
                "consent_no_expiration": True,
            },
        },
    )
    assert record["consent_revoked"] is True
    assert record["status"] == "blocked_consent_revoked_takedown_required"


def test_ptdp_blocks_contradictory_status_spellings(tmp_path: Path) -> None:
    record = _consent_evidence(
        tmp_path,
        {
            "consent_status": "granted",
            "consentStatus": "denied",
            "consent_scope": ["model_training", "robot_evaluation"],
            "consent_no_expiration": True,
        },
    )
    assert record["consent_evidence_present"] is False
    assert any(
        blocker.startswith("consent_status_unknown:") for blocker in record["blockers"]
    )


def test_ptdp_blocks_malformed_revocation_flag_on_active_status(
    tmp_path: Path,
) -> None:
    record = _consent_evidence(
        tmp_path,
        {
            "consent_status": "granted",
            "consent_revoked": "maybe",
            "consent_scope": ["model_training", "robot_evaluation"],
            "consent_no_expiration": True,
        },
    )
    assert record["consent_evidence_present"] is False


def test_ptdp_per_record_revocation_blocks_package(tmp_path: Path) -> None:
    record = _consent_evidence(
        tmp_path,
        {
            "consent_status": "granted",
            "consent_scope": ["model_training", "robot_evaluation"],
            "consent_no_expiration": True,
        },
        rights_packet={
            "records": [
                {"rights_scope": "model_training", "status": "revoked"},
            ]
        },
    )
    assert record["consent_revoked"] is True
    assert record["status"] == "blocked_consent_revoked_takedown_required"


def test_ptdp_live_closure_gate_passed_requires_strict_true() -> None:
    for hostile in ("false", "no", "0", 1, "true", [True]):
        reference = ptdp._live_closure_gate_reference(
            {"gates": {"rights_privacy_scope": {"passed": hostile, "blockers": []}}},
            "rights_privacy_scope",
        )
        assert reference["passed"] is False, hostile
        assert ptdp._gate_blockers(
            reference, "rights_privacy_scope", "fallback_blocker"
        ), hostile
    reference = ptdp._live_closure_gate_reference(
        {"gates": {"rights_privacy_scope": {"passed": True, "blockers": []}}},
        "rights_privacy_scope",
    )
    assert reference["passed"] is True
    assert (
        ptdp._gate_blockers(reference, "rights_privacy_scope", "fallback_blocker")
        == []
    )


# ---------------------------------------------------------------------------
# Cross-surface agreement matrix
# ---------------------------------------------------------------------------


HOSTILE_CONSENT_SOURCES = [
    pytest.param({"consent_status": "denied"}, id="denied-status"),
    pytest.param({"consent_status": "pending"}, id="pending-status"),
    pytest.param({"consent_status": {"v": "granted"}}, id="wrong-typed-status"),
    pytest.param(
        {"consent_status": "granted", "consentStatus": "revoked"},
        id="contradictory-status",
    ),
    pytest.param(
        {"consent_revoked": True, "rights": {"consent_status": "granted"}},
        id="nested-shadowed-revocation",
    ),
    pytest.param(
        {"consent_status": "granted", "consentRevokedAt": "2026-01-01T00:00:00Z"},
        id="camelcase-revoked-at",
    ),
]


@pytest.mark.parametrize("source", HOSTILE_CONSENT_SOURCES)
def test_every_surface_blocks_the_same_hostile_consent_source(
    tmp_path: Path, source: dict
) -> None:
    """The core invariant: no surface may be more permissive than another.

    For each hostile consent source, the delivery gate must refuse to serve,
    PTDP must refuse consent evidence, and the rights-provenance review over
    the same fields must not clear.
    """
    root = _capture_root(tmp_path, source)

    state = read_consent_state(root)
    assert state["state"] != "active"

    gate = evaluate_delivery_time_takedown_gate(capture_root=root, surface="matrix")
    assert gate["serve_allowed"] is False

    record = ptdp._build_consent_evidence_record(
        capture_root=root,
        output_dir=tmp_path / "out",
        rights_packet={},
        scene_id="scene-1",
        capture_id="capture-1",
        generated_at="2026-07-11T00:00:00+00:00",
    )
    assert record["consent_evidence_present"] is False

    review = build_rights_provenance_review(
        rights_summary={"derived_scene_generation_allowed": True, **source},
        privacy_processing=None,
        provenance_summary=None,
        site_identity=None,
        adjacent_systems=None,
    )
    assert review["rights"]["status"] != "cleared"


def test_active_statuses_and_revoked_statuses_are_disjoint() -> None:
    assert not (CONSENT_ACTIVE_STATUSES & CONSENT_REVOKED_STATUSES)
