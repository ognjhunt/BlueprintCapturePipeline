from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.measurement_method_research_catalog import (
    QUALIFICATION_PROTOCOLS,
    RESEARCH_CLASSIFICATIONS,
    STANDING_ABSTENTIONS,
    TASK_CLASS_QUALIFICATION_PROTOCOLS,
    ResearchCatalogError,
    priority_investigations,
    qualification_benchmark_blueprints,
    research_candidate_r0_stage_data,
    research_catalog_snapshot,
    research_intake_catalog,
    validate_research_method_candidate,
)
from blueprint_pipeline.measurement_research_admission import (
    admission_supports_production_route,
    create_research_candidate,
)
from blueprint_pipeline.task_site_measurement_routing import (
    CLAIM_LEVELS,
    MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING,
    PHYSICS_AUTHORITY_FORBIDDEN_FAMILIES,
    TASK_CAPABILITIES,
    WORLD_MODEL_ALLOWED_ROLES,
)


def test_every_catalog_entry_is_a_validated_non_production_candidate() -> None:
    entries = research_intake_catalog()
    identifiers = [row["candidate_id"] for row in entries]
    assert len(identifiers) == len(set(identifiers))
    assert len(entries) >= 40
    for row in entries:
        assert row["blueprint_qualified"] is False
        assert row["production_route_eligible"] is False
        assert row["evidence_label"] in {"VF", "EC", "INF", "PQ"}
        assert row["classification"] in RESEARCH_CLASSIFICATIONS
        family_cap = MEASUREMENT_METHOD_FAMILY_CLAIM_CEILING[row["method_family"]]
        assert CLAIM_LEVELS[row["claim_ceiling_after_qualification"]] <= CLAIM_LEVELS[family_cap]
        for protocol in row["required_qualification_protocols"]:
            assert protocol in QUALIFICATION_PROTOCOLS
    snapshot = research_catalog_snapshot()
    assert snapshot["production_route_count"] == 0
    assert snapshot["public_research_is_qualification"] is False
    assert snapshot["entry_count"] == len(entries)
    assert snapshot["catalog_snapshot_digest"].startswith("sha256:")
    assert (
        snapshot["catalog_snapshot_digest"]
        == (research_catalog_snapshot()["catalog_snapshot_digest"])
    )


def test_landscape_verdicts_are_encoded_faithfully() -> None:
    by_id = {row["candidate_id"]: row for row in research_intake_catalog()}
    assert by_id["brax"]["classification"] == "unsuitable_as_new_physics_authority"
    assert by_id["simpler-env"]["classification"] == "methodology_template"
    assert by_id["direct-captured-observations"]["classification"] == (
        "authoritative_evidence_source"
    )
    assert by_id["exact-geometry-stack"]["claim_ceiling_after_qualification"] == "C2"
    for provider_id in ("world-labs-marble", "lightwheel-simready"):
        provider = by_id[provider_id]
        assert provider["classification"] == "provider_requiring_contract_gates"
        assert provider["access"]["api_only"] is True
    flash = by_id["flash"]
    assert flash["access"]["source_available"] is False
    assert any(
        "production_access" in code or "no_production_route" in code
        for code in flash["known_limitations"]
    )
    for engineering_id in ("altair-edem-2026", "ansys-rocky-2026", "abaqus"):
        assert by_id[engineering_id]["classification"] == "engineering_reference_solver"


def test_world_model_and_appearance_entries_carry_forbidden_authorities() -> None:
    for row in research_intake_catalog():
        forbidden = set(row["forbidden_authorities"])
        if row["method_family"] == "learned_world_model":
            assert set(row["world_model_roles"]) <= WORLD_MODEL_ALLOWED_ROLES
            assert {
                "collision_authority",
                "force_authority",
                "safety_authority",
                "physical_success_proof",
            } <= forbidden
        if row["method_family"] in PHYSICS_AUTHORITY_FORBIDDEN_FAMILIES:
            assert {"collision_authority", "force_authority"} <= forbidden


def test_catalog_rejects_bq_labels_and_family_cap_violations() -> None:
    template = copy.deepcopy(research_intake_catalog()[0])
    template.pop("research_candidate_digest")
    self_qualified = copy.deepcopy(template)
    self_qualified["evidence_label"] = "BQ"
    with pytest.raises(
        ResearchCatalogError,
        match="blueprint_qualified_label_requires_signed_qualification_record",
    ):
        validate_research_method_candidate(self_qualified)
    over_cap = copy.deepcopy(template)
    over_cap["claim_ceiling_after_qualification"] = "C6"
    with pytest.raises(ResearchCatalogError, match="exceeds_family_cap"):
        validate_research_method_candidate(over_cap)
    fake_protocol = copy.deepcopy(template)
    fake_protocol["required_qualification_protocols"] = ["Q-VIBES"]
    with pytest.raises(ResearchCatalogError, match="qualification_protocols_invalid"):
        validate_research_method_candidate(fake_protocol)


def test_task_protocol_map_and_standing_abstentions_are_complete() -> None:
    assert set(TASK_CLASS_QUALIFICATION_PROTOCOLS) == set(TASK_CAPABILITIES)
    for protocols in TASK_CLASS_QUALIFICATION_PROTOCOLS.values():
        assert protocols
        assert set(protocols) <= set(QUALIFICATION_PROTOCOLS)
    assert len(STANDING_ABSTENTIONS) == 20
    assert "collision_from_gaussian_splat_alone" in STANDING_ABSTENTIONS
    assert "world_model_physics_or_safety_authority" in STANDING_ABSTENTIONS
    assert "qualification_transfer_without_compatibility_evidence" in STANDING_ABSTENTIONS


def test_benchmark_blueprints_and_priorities_reference_real_candidates() -> None:
    identifiers = {row["candidate_id"] for row in research_intake_catalog()}
    blueprints = qualification_benchmark_blueprints()
    assert [row["benchmark_id"] for row in blueprints] == [
        "capture-to-geometry-and-contact",
        "capture-to-observation",
        "capture-to-deformation",
        "world-model-action-fidelity",
    ]
    for blueprint in blueprints:
        assert set(blueprint["methods_compared"]) <= identifiers
        assert set(blueprint["protocols"]) <= set(QUALIFICATION_PROTOCOLS)
        for lane_members in (blueprint.get("lanes") or {}).values():
            assert set(lane_members) <= identifiers
    priorities = priority_investigations()
    assert [row["priority"] for row in priorities] == list(range(1, 11))
    for row in priorities:
        assert set(row["candidate_ids"]) <= identifiers


def test_catalog_entry_bridges_to_r0_admission_without_eligibility() -> None:
    entry = next(row for row in research_intake_catalog() if row["candidate_id"] == "mujoco-3")
    record = create_research_candidate(
        candidate_id=f"admission-{entry['candidate_id']}",
        method_id=entry["method_id"],
        stage_data=research_candidate_r0_stage_data(entry),
        approval={
            "role": "research_analyst",
            "actor_id": "fixture-analyst",
            "actor_type": "human",
            "approved": True,
            "signature_id": "signature-research-analyst",
        },
    )
    assert record["stage"] == "R0"
    assert record["production_eligible"] is False
    assert admission_supports_production_route(record) is False
