"""One effective-catalog implementation: production screening and the intake preview agree.

Every registration here is a development-only fixture. The digests compared are
the catalog's own identity; none of them is evidence that a robot or policy ran.
"""
# Covers: src/blueprint_pipeline/beta_policy_effective_catalog.py
# Covers: src/blueprint_pipeline/beta_policy_screening_g1_catalog.py
# Covers: src/blueprint_pipeline/beta_policy_screening_files.py
# Covers: src/blueprint_pipeline/team_candidate_credential_file.py
# Covers: src/blueprint_pipeline/team_candidate_registry_reader.py
# Covers: src/blueprint_pipeline/team_candidate_rights.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_effective_catalog as effective
from blueprint_pipeline import beta_policy_screening_team as team
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_g1_catalog import CANDIDATES as G1_CANDIDATES
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

UNVALIDATED = "development_only_unvalidated_registration"


def _bytes(catalog):
    return json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _rows(catalog):
    return {row["candidate_id"]: row for row in catalog["candidates"]}


def _g1_registrations():
    return {candidate: {"native_g1": {"scene_packet": {"sha256": "sha256:" + str(index) * 64}}}
            for index, candidate in enumerate(G1_CANDIDATES[:2], start=1)}


def _registration_revision(settings, blockers):
    return {"registration": settings["robot_policy"],
            "settings": {name: value for name, value in settings.items()
                         if name not in {"robot_policy", "credential_file"}},
            "blockers": blockers,
            "credential_reference_digest": canonical_digest({"path": settings.get("credential_file")})}


def _world(kind, tmp_path, monkeypatch):
    """(profile, base catalog, expected revision suffix) from the existing registration fixtures."""
    if kind == "no_registrations":
        return {}, load_default_catalog(), ""
    if kind == "g1_overlay":
        registrations = _g1_registrations()
        native = {key: value["native_g1"] for key, value in registrations.items()}
        return {"screening_candidates": registrations}, load_default_catalog(), ":g1:" + canonical_digest(native)[7:]
    if kind in {"scripted", "local", "g1_overlay_with_scripted"}:
        if kind == "local":
            from tests.test_openwam_franka_runtime import prepared_registration

            _, profile, _ = prepared_registration(tmp_path, monkeypatch)
        else:
            from tests.test_scripted_robot_baseline_bundle import registered_world

            _, profile, _, _ = registered_world(tmp_path, monkeypatch)
        configured = dict(profile["screening_candidates"])
        revisions = {key: _registration_revision(settings, []) for key, settings in configured.items()}
        suffix = ":registrations:" + canonical_digest(revisions)[7:]
        if kind == "g1_overlay_with_scripted":
            g1 = _g1_registrations()
            configured.update(g1)
            native = {key: value["native_g1"] for key, value in g1.items()}
            suffix = ":g1:" + canonical_digest(native)[7:] + suffix
        return {"screening_candidates": configured}, load_default_catalog(), suffix
    from tests.test_beta_policy_screening_team import team_plan

    _, _, registrations = team_plan(tmp_path, monkeypatch, count=3, invalid=kind == "invalid_retained")
    revisions = {key: _registration_revision(settings, []) for key, settings in registrations.items()}
    if kind == "invalid_retained":
        revisions["team_policy_00"] = {"status": "beta_screening_input_bytes_changed"}
        registrations[UNVALIDATED] = {"robot_policy": {"display_name": "never validated"}}
        revisions[UNVALIDATED] = {"status": "beta_screening_candidate_configuration_invalid"}
    return ({"screening_candidates": registrations}, load_default_catalog(),
            ":registrations:" + canonical_digest(revisions)[7:])


def test_existing_entrypoints_reexport_the_single_shared_implementation():
    from blueprint_pipeline import beta_policy_screening_files as files
    from blueprint_pipeline import beta_policy_screening_g1 as g1
    from blueprint_pipeline import beta_policy_screening_g1_catalog as g1_catalog
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from blueprint_pipeline import team_candidate_credential_file as credential
    from blueprint_pipeline import team_candidate_quick10_admission as admission
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from blueprint_pipeline import team_candidate_quick10_vast as vast
    from blueprint_pipeline import team_candidate_registry_reader as reader
    from blueprint_pipeline import team_candidate_rights as rights

    shared = {
        (team, effective): ("catalog_for_profile", "team_policy_interface", "_registration_rights"),
        (g1, g1_catalog): ("CANDIDATES", "catalog_for_profile"),
        (launch, files): ("_read", "_record", "_bound_path"),
        (vast, credential): ("credential_file_blockers",),
        (admission, rights): ("TeamCandidateAdmissionError", "RIGHTS_SCHEMA", "validate_candidate_rights",
                              "validate_provider_disclosure"),
        (registry, reader): ("TeamCandidateRegistryError", "REGISTRY_ENV", "REGISTRY_SCHEMA", "DERIVED_SCHEMA",
                             "DERIVED_FIELDS", "OPERATOR_FIELDS", "load_registry", "_read", "_operator_file",
                             "_derived_settings", "_derived_path", "_site_screening_profile"),
    }
    for (production, extracted), names in shared.items():
        for name in names:
            assert getattr(production, name) is getattr(extracted, name), (production.__name__, name)


@pytest.mark.parametrize("kind", ["no_registrations", "scripted", "local", "hosted", "invalid_retained",
                                  "g1_overlay", "g1_overlay_with_scripted"])
def test_shared_builder_is_digest_identical_to_production_entrypoint(tmp_path, monkeypatch, kind):
    profile, base, suffix = _world(kind, tmp_path, monkeypatch)
    production = team.catalog_for_profile(profile, base)
    shared = effective.catalog_for_profile(profile, base)
    assert _bytes(shared) == _bytes(production)
    assert shared["catalog_digest"] == production["catalog_digest"] == seal_catalog(production)["catalog_digest"]
    # The revision's inputs are pinned: a changed registration digest would replan every site.
    assert production["revision"] == base["revision"] + suffix
    rows, original = _rows(production), _rows(base)
    configured = profile.get("screening_candidates") or {}
    registered = {key for key, value in configured.items() if value.get("robot_policy")}
    # Packaged rows a registration does not touch are carried byte-for-byte.
    assert {key: row for key, row in rows.items() if key not in registered} == {
        key: row for key, row in original.items() if key not in registered}
    if kind == "no_registrations":
        assert production == base
        assert team.catalog_for_profile({}) == effective.catalog_for_profile({}) == load_default_catalog()
    elif kind in {"scripted", "local", "hosted"}:
        expected = {"scripted": "scripted", "local": "checkpoint", "hosted": "hosted"}[kind]
        assert registered and all(rows[key]["policy"]["kind"] == expected for key in registered)
        assert all(rows[key]["eligibility"]["enabled"] and rows[key]["eligibility"]["blockers"] == []
                   for key in registered)
    elif kind == "invalid_retained":
        assert rows["team_policy_00"]["eligibility"]["blockers"] == ["beta_screening_input_bytes_changed"]
        assert rows["team_policy_00"]["eligibility"]["enabled"] is False
        assert rows[UNVALIDATED]["eligibility"] == {**rows[UNVALIDATED]["eligibility"], "enabled": False,
            "commercial_use": "under_review", "blockers": ["beta_screening_candidate_configuration_invalid"]}
        assert all(rows[key]["eligibility"]["enabled"] for key in ("team_policy_01", "team_policy_02"))
    else:
        assert ":g1:" in production["revision"] and production["catalog_digest"] != base["catalog_digest"]


@pytest.mark.parametrize("kind", ["scripted", "local"])
def test_intake_preview_profile_path_matches_production_screening_catalog(tmp_path, monkeypatch, kind):
    from blueprint_pipeline import website_candidate_readiness as preview
    from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture

    if kind == "local":
        from tests.test_openwam_franka_runtime import prepared_registration

        prepared_registration(tmp_path, monkeypatch)
    else:
        from tests.test_scripted_robot_baseline_bundle import registered_world

        registered_world(tmp_path, monkeypatch)
    capture = tmp_path / "capture"
    # The production path a screening plan is revalidated against (validate_team_candidate_catalog).
    production = team.current_catalog_for_scene({"screening_capture_root": str(capture)})
    profile = profile_for_capture(capture)
    configured, snapshot = preview._profile_snapshot(Path(capture), {"scene_id": profile.get("scene_id")})
    assert snapshot["profile_source"] == "capture_profile"
    assert _bytes(effective.catalog_for_profile(configured)) == _bytes(production)
    registered = set(profile["screening_candidates"])
    assert registered and all(_rows(production)[key]["eligibility"]["enabled"] for key in registered)
