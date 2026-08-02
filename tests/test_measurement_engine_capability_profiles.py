from __future__ import annotations

from datetime import date

import pytest

from blueprint_pipeline.measurement_engine_capability_profiles import (
    EngineProfileError,
    engine_capability_profiles,
    engine_profile_by_method_id,
    engine_profile_set_snapshot,
    r1_source_verification_stage_data,
)
from blueprint_pipeline.measurement_method_research_catalog import (
    research_candidate_r0_stage_data,
    research_intake_catalog,
)
from blueprint_pipeline.measurement_research_admission import (
    advance_research_admission,
    create_research_candidate,
)
from blueprint_pipeline.task_site_measurement_routing import (
    PHYSICS_AUTHORITY_CAPABILITIES,
    derive_task_measurement_requirements,
    route_task_site_measurement,
    validate_site_evidence_profile,
)


SHA_A = "sha256:" + "a" * 64

# Rights/operations booleans substantiated by the license entries of the
# identity sources rather than a per-field manifest row.
_LICENSE_BACKED_FIELDS = {
    "source_available",
    "local_offline_supported",
    "api_only",
    "commercial_use_allowed",
    "redistribution_allowed",
    "deletion_right_supported",
    "output_export_supported",
    "provider_training_use_allowed",
}


def _site() -> dict:
    return validate_site_evidence_profile(
        {
            "schema_version": "site_evidence_profile.v1",
            "profile_id": "engine-profile-site",
            "bundle_id": "capture-1",
            "bundle_hash": SHA_A,
            "provenance_record_id": "provenance-fixture",
            "rights": {"commercial_evaluation_allowed": True},
            "privacy": {"external_processing_allowed": False},
            "coordinate_system": {"metric_scale_verified": True},
            "evidence": {
                evidence_id: {
                    "available": True,
                    "validated": True,
                    "record_id": f"record-{evidence_id}",
                }
                for evidence_id in (
                    "metric_scale",
                    "robot_site_registration",
                    "validated_collider",
                    "mass_inertia",
                    "friction_contact",
                    "material_parameters",
                )
            },
            "limitations": {"known_missing_regions": [], "forbidden_claims": []},
        }
    )


def test_every_engine_profile_validates_and_joins_the_research_catalog() -> None:
    profiles = engine_capability_profiles()
    assert len(profiles) == 7
    catalog = {row["candidate_id"]: row for row in research_intake_catalog()}
    for profile in profiles:
        method_id = profile["method_id"]
        assert method_id in catalog, method_id
        assert profile["capabilities"]["method_family"] == catalog[method_id]["method_family"]
        quality = profile["evidence_quality"]
        assert quality["unverified_fields_fail_closed"] is True
        assert quality["public_research_is_qualification"] is False
        assert quality["source_manifest"]
    snapshot = engine_profile_set_snapshot()
    assert snapshot["qualification_record_count"] == 0
    assert snapshot["profiles_are_routable_without_qualification"] is False
    assert snapshot["set_digest"].startswith("sha256:")


def test_every_true_capability_is_traceable_to_a_source_manifest_entry() -> None:
    for profile in engine_capability_profiles():
        manifest_facts = {
            fact for row in profile["evidence_quality"]["source_manifest"] for fact in row["facts"]
        }
        for field, value in profile["capabilities"].items():
            if value is True and field not in _LICENSE_BACKED_FIELDS:
                assert field in manifest_facts, (
                    f"{profile['method_id']}:{field} is True without a source"
                )


def test_verified_features_alone_never_route_anything() -> None:
    requirements = derive_task_measurement_requirements(
        {"claim_id": "rigid-check", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    decision = route_task_site_measurement(
        requirements,
        _site(),
        list(engine_capability_profiles()),
        [],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 2),
    )
    assert decision["status"] == "abstention"
    for row in decision["candidates_considered"]:
        assert "no_exact_verified_qualification" in row["rejection_codes"]
    assert decision["abstention"]["smallest_next_action"]["action_type"] == (
        "qualification_benchmark"
    )


def test_unverified_determinism_fails_a_replay_requirement() -> None:
    requirements = derive_task_measurement_requirements(
        {
            "claim_id": "replay-check",
            "claim_type": "collision_contact",
            "measurement_constraints": {"deterministic_replay_required": True},
        },
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    decision = route_task_site_measurement(
        requirements,
        _site(),
        list(engine_capability_profiles()),
        [],
        catalog_snapshot_hash=SHA_A,
        as_of=date(2026, 8, 2),
    )
    assert decision["status"] == "abstention"
    for row in decision["candidates_considered"]:
        assert "deterministic_replay_not_supported" in row["rejection_codes"]


def test_isaac_rtx_sensor_path_declares_no_physics_capability() -> None:
    profile = engine_profile_by_method_id("isaac-rtx-openusd-sensor-path")
    for field in PHYSICS_AUTHORITY_CAPABILITIES:
        assert profile["capabilities"][field] is False
    with pytest.raises(EngineProfileError, match="engine_profile_unknown"):
        engine_profile_by_method_id("no-such-engine")


def test_chrono_source_discrepancy_is_recorded_not_silently_resolved() -> None:
    profile = engine_profile_by_method_id("project-chrono-10")
    assert "SOURCE DISCREPANCY RECORDED" in profile["evidence_quality"]["notes"]
    assert profile["capabilities"]["release_date"] == "2026-03-27"
    assert profile["capabilities"]["cfd_supported"] is False


def test_r1_stage_data_advances_a_catalog_candidate_to_r1() -> None:
    entry = next(row for row in research_intake_catalog() if row["candidate_id"] == "mujoco-3")

    def approval(role: str) -> dict:
        return {
            "role": role,
            "actor_id": f"fixture-{role}",
            "actor_type": "human",
            "approved": True,
            "signature_id": f"signature-{role}",
        }

    r0 = create_research_candidate(
        candidate_id="admission-mujoco-3",
        method_id="mujoco-3",
        stage_data=research_candidate_r0_stage_data(entry),
        approval=approval("research_analyst"),
    )
    r1 = advance_research_admission(
        r0,
        target_stage="R1",
        stage_data=r1_source_verification_stage_data("mujoco-3"),
        approvals=[approval("research_lead")],
    )
    assert r1["stage"] == "R1"
    assert r1["production_eligible"] is False
    verification = r1["stage_data"]["source_verification"]
    assert verification["unverified_fields_fail_closed"] is True
    assert verification["live_fetch_references"]
