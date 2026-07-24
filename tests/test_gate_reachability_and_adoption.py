"""Tests for gate reachability auditing and architecture-adoption tracks."""

from __future__ import annotations


from blueprint_pipeline import gate_reachability_audit as audit
from blueprint_pipeline import world_model_architecture_adoption as adoption


# -- reachability audit --------------------------------------------------


def test_audit_detects_the_dead_external_study_gate() -> None:
    """`validate_external_study` never returns 'validated', so its gate is dead."""

    report = audit.audit_gate_reachability()
    gate = next(g for g in report["gates"] if g["gate_id"] == "external_study_validated")

    assert gate["classification"] == audit.UNREACHABLE_BY_CONSTRUCTION
    assert gate["probe"]["observed_status"] == "external_proof_required"
    assert gate["probe"]["validated_status_reachable"] is False
    assert gate["probe"]["validated_literal_present_in_source"] is False


def test_gates_depending_on_the_dead_gate_are_also_dead() -> None:
    report = audit.audit_gate_reachability()
    dependents = {
        "sc3_eval_protocol.public_rank_fidelity_claim_eligible",
        "sc3_eval_protocol.claim_ready",
        "sc3_eval_protocol.eligible_preregistered_external_rank_fidelity",
    }
    classified = {
        g["gate_id"]: g["classification"] for g in report["gates"] if g["gate_id"] in dependents
    }

    assert set(classified) == dependents
    assert all(value == audit.UNREACHABLE_BY_CONSTRUCTION for value in classified.values())


def test_audit_finds_hardcoded_false_claim_fields_with_line_numbers() -> None:
    report = audit.audit_gate_reachability()
    closed_loop = next(
        g
        for g in report["gates"]
        if g["gate_id"].endswith("::full_closed_loop_episode_proven")
    )

    assert closed_loop["classification"] == audit.UNREACHABLE_BY_CONSTRUCTION
    # Several distinct emission sites, each a place a caller cannot influence.
    assert closed_loop["probe"]["occurrence_count"] >= 5
    assert all(isinstance(line, int) for line in closed_loop["probe"]["occurrences"])


def test_divergent_ood_registries_are_not_reported_as_unreachable() -> None:
    """They bind different artifacts, so the honest label is divergence."""

    report = audit.audit_gate_reachability()
    gate = next(
        g for g in report["gates"] if g["gate_id"] == "ood_axis_vocabulary_agreement"
    )

    assert gate["classification"] == audit.DIVERGENT_REGISTRY
    assert gate["probe"]["identical"] is False
    assert gate["probe"]["binds_the_same_artifact"] is False
    assert "appearance" in gate["probe"]["decision_axes_absent_from_sc3"]
    assert "viewpoint" in gate["probe"]["decision_axes_absent_from_sc3"]


def test_audit_reports_counts_and_a_status() -> None:
    report = audit.audit_gate_reachability()

    assert report["schema_version"] == "gate_reachability_audit.v1"
    assert report["status"] == "unreachable_gates_present"
    assert report["unreachable_gate_count"] == len(report["unreachable_gate_ids"])
    assert sum(report["counts_by_classification"].values()) == len(report["gates"])
    assert report["claim_boundary"]["a_reachable_gate_is_not_a_satisfied_gate"] is True


def test_blocker_classification_separates_waiting_from_impossible() -> None:
    """The point of the audit: "38 blockers" is not one kind of thing."""

    result = audit.classify_blockers(
        [
            "accepted_anchor_count_zero",
            "public_rank_fidelity_claim_eligible",
            "full_closed_loop_episode_proven",
            "matched_trial_count_below_minimum",
        ]
    )

    by_blocker = {row["blocker"]: row["classification"] for row in result["classified_blockers"]}
    assert by_blocker["accepted_anchor_count_zero"] == audit.SATISFIABLE_NOW
    assert by_blocker["matched_trial_count_below_minimum"] == audit.SATISFIABLE_NOW
    assert (
        by_blocker["public_rank_fidelity_claim_eligible"]
        == audit.UNREACHABLE_BY_CONSTRUCTION
    )
    assert result["unreachable_blocker_count"] == 2
    assert result["total_blocker_count"] == 4


def test_audit_is_a_probe_so_a_repair_would_change_its_verdict() -> None:
    """The audit reads real source and calls real validators, not a static list."""

    report = audit.audit_gate_reachability()
    probed = [g for g in report["gates"] if "probe" in g]
    assert probed, "audit must carry probe evidence"
    external = next(g for g in probed if g["gate_id"] == "external_study_validated")
    assert external["probe"]["probe"] == "validate_external_study"


# -- architecture adoption ----------------------------------------------


def _proposal(**overrides):
    kwargs = {
        "proposal_id": "wan21-action-conditioned-v1",
        "design_elements": [
            "frame_causal_attention",
            "per_frame_action_cross_attention",
        ],
        "base_component_ids": ["wan21_vae"],
        "declared_as_blueprint_authored": True,
    }
    kwargs.update(overrides)
    return adoption.build_architecture_adoption_plan(**kwargs)


def test_adoption_on_already_licensed_components_is_authorized_to_build() -> None:
    """Building on Apache-2.0 components we already pin is not upstream-blocked."""

    plan = _proposal()

    assert plan["status"] == "authorized_to_build", plan["blockers"]
    assert plan["track"] == adoption.ARCHITECTURE_ADOPTION_TRACK
    component = plan["base_components"][0]
    assert component["declared_license"] == "Apache-2.0"
    assert component["revision_pinned"] is True


def test_the_two_tracks_are_reported_independently() -> None:
    plan = _proposal()
    tracks = plan["tracks"]

    assert tracks[adoption.UPSTREAM_ADMISSION_TRACK]["status"] == "awaiting_upstream_release"
    assert (
        tracks[adoption.UPSTREAM_ADMISSION_TRACK]["independent_of_architecture_adoption"]
        is True
    )
    assert tracks[adoption.ARCHITECTURE_ADOPTION_TRACK]["status"] == "authorized_to_build"


def test_authorization_to_build_confers_no_evaluator_standing() -> None:
    plan = _proposal()
    boundary = plan["claim_boundary"]

    assert boundary["authorization_is_to_build_not_to_claim"] is True
    assert boundary["resulting_model_must_pass_ordinary_evaluator_qualification"] is True
    assert boundary["upstream_metrics_are_not_inherited"] is True
    assert boundary["public_claim_upgrade_allowed"] is False
    assert "any rank-fidelity claim" in plan["tracks"][
        adoption.ARCHITECTURE_ADOPTION_TRACK
    ]["does_not_unblock"]


def test_using_upstream_weights_routes_back_to_the_blocked_track() -> None:
    plan = _proposal(uses_upstream_weights_or_code=True)

    assert plan["status"] == "blocked"
    assert (
        "architecture_adoption_may_not_use_upstream_weights_or_code" in plan["blockers"]
    )


def test_inheriting_upstream_metrics_is_refused() -> None:
    plan = _proposal(inherits_upstream_metrics=True)
    assert "architecture_adoption_may_not_inherit_upstream_metrics" in plan["blockers"]


def test_model_must_be_declared_blueprint_authored() -> None:
    plan = _proposal(declared_as_blueprint_authored=False)
    assert "architecture_adoption_must_be_declared_blueprint_authored" in plan["blockers"]


def test_unpinned_or_unknown_components_are_refused() -> None:
    plan = _proposal(base_component_ids=["some_random_checkpoint"])
    assert plan["status"] == "blocked"
    assert any(
        item.startswith("architecture_adoption_component_not_pinned")
        for item in plan["blockers"]
    )


def test_non_permissive_component_is_refused() -> None:
    """The Cosmos text encoder is not Apache-2.0, so it cannot back this track."""

    plan = _proposal(base_component_ids=["cosmos_reason1_text_encoder"])
    assert plan["status"] == "blocked"
    assert any(
        item.startswith("architecture_adoption_component_not_permissively_licensed")
        for item in plan["blockers"]
    )


def test_only_registered_design_elements_may_be_adopted() -> None:
    plan = _proposal(design_elements=["upstream_training_code"])
    assert plan["status"] == "blocked"
    assert any(
        item.startswith("architecture_adoption_element_not_adoptable")
        for item in plan["blockers"]
    )


def test_licensed_inventory_reports_the_pinned_wan_backbone() -> None:
    inventory = {row["component_id"]: row for row in adoption.licensed_component_inventory()}
    wan = inventory["wan21_vae"]

    assert wan["source_id"] == "Wan-AI/Wan2.1-T2V-1.3B"
    assert wan["permissively_licensed"] is True
    assert wan["revision_pinned"] is True


def test_backend_selection_principle_rejects_parameter_scale_ordering() -> None:
    principle = adoption.backend_selection_principle()

    assert principle["principle"] == "order_backends_by_measured_fidelity_not_parameter_scale"
    assert "parameter count" in principle["explicitly_not_a_ranking_input"]
    assert "measured rank fidelity under a frozen Blueprint protocol" in (
        principle["ranking_inputs"]
    )
    assert principle["claim_boundary"]["principle_is_not_a_measurement"] is True
