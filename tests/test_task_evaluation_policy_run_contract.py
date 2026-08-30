from __future__ import annotations

import copy
import json
import os
import pwd

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    DecisionEnvelope,
    EvidencePlan,
    canonical_digest,
)
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    stage_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    canonical_digest as launch_profile_digest,
    public_launch_profile_descriptor,
    validate_public_launch_profile_descriptor,
)
from blueprint_pipeline.task_evaluation_launch_preparation_worker import (
    process_launch_preparation_queue,
)
from blueprint_pipeline.task_evaluation_policy_run_contract import (
    EMBODIMENT_ID,
    FROZEN_CANDIDATE_IDS,
    TaskEvaluationPolicyRunContractError,
    build_policy_campaign_activation_manifest,
    build_policy_run_plan,
    compile_policy_run_configuration,
    expand_policy_run_preparation_request,
    policy_run_result_projection_digest,
    validate_policy_run_configuration,
    validate_policy_run_result_projection,
    validate_policy_run_setup,
)
from blueprint_pipeline.task_evaluation_result_delivery import DELIVERY_SCHEMA_VERSION
from blueprint_pipeline.task_evaluation_run_webapp_sync import (
    build_task_evaluation_run_webapp_publication,
)
from tests.test_task_evaluation_launch_preparation_contract import request
from tests.test_task_evaluation_launch_preparation_worker import (
    fake_adapter,
    fetcher,
    request_with_fetchable_bytes,
)
from tests.test_task_evaluation_launch_dispatcher import _profile as launch_profile


SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name
DIGEST = "sha256:" + "a" * 64


def _ref(index: int) -> dict[str, object]:
    return {
        "uri": f"s3://blueprint-production-inputs/policy-run-{index}.json",
        "digest": f"sha256:{index:064x}",
        "size_bytes": 1000 + index,
    }


def _template() -> dict[str, object]:
    base = request()
    base["controller"] = {
        "identity": {"id": "paired-policy-matrix", "version": "v1"},
        "kind": "policy_container",
        "configuration": _ref(80),
        "model_or_asset_rights": _ref(81),
    }
    template = {
        "schema_version": "task_evaluation_policy_run_preparation_template.v1",
        **{
            field: copy.deepcopy(base[field])
            for field in (
                "scene",
                "construction",
                "robot",
                "controller",
                "task",
                "sensors",
                "runtime",
                "execution_adapter",
                "spend",
            )
        },
        "publication": {"service_account_readback_required": True},
        "template_digest": "",
    }
    template["template_digest"] = canonical_digest(
        template, digest_field="template_digest"
    )
    return template


def setup() -> dict[str, object]:
    families = [
        "canonical_anchor",
        "placement_approach",
        "placement_approach",
        "illumination",
        "camera_sensor",
        "bounded_physics",
        "pairwise",
        "pairwise",
        "held_out",
        "held_out",
    ]
    quick_cells = [
        {
            "cell_id": f"quick-cell-{index}-{family}",
            "family": family,
            "partition": "held_out" if family == "held_out" else "qualification",
            "scored": True,
            "cell_spec_digest": f"sha256:{100 + index:064x}",
        }
        for index, family in enumerate(families)
    ]
    presets = [
        {
            "preset_id": "quick_10",
            "label": "Quick",
            "scenario_count_per_policy": 10,
            "availability": "enabled",
            "default": True,
            "family_counts": {
                "canonical_anchor": 1,
                "placement_approach": 2,
                "illumination": 1,
                "camera_sensor": 1,
                "bounded_physics": 1,
                "pairwise": 2,
                "held_out": 2,
            },
            "scenario_set_digest": canonical_digest({"ordered_cells": quick_cells}),
            "parent_preset_id": None,
            "parent_prefix_count": 0,
            "nesting_proof_digest": "",
            "estimate": {"status": "unavailable"},
            "cells": quick_cells,
        },
        {
            "preset_id": "standard_100",
            "label": "Standard",
            "scenario_count_per_policy": 100,
            "availability": "coming_later",
            "default": False,
            "family_counts": {
                "canonical_anchor": 1,
                "placement_approach": 20,
                "illumination": 14,
                "camera_sensor": 14,
                "bounded_physics": 14,
                "pairwise": 19,
                "held_out": 18,
            },
            "scenario_set_digest": "sha256:" + "c" * 64,
            "parent_preset_id": "quick_10",
            "parent_prefix_count": 10,
            "nesting_proof_digest": "",
            "estimate": {"status": "unavailable"},
        },
        {
            "preset_id": "deep_500",
            "label": "Deep",
            "scenario_count_per_policy": 500,
            "availability": "coming_later",
            "default": False,
            "family_counts": {
                "canonical_anchor": 1,
                "placement_approach": 100,
                "illumination": 70,
                "camera_sensor": 70,
                "bounded_physics": 70,
                "pairwise": 100,
                "held_out": 89,
            },
            "scenario_set_digest": "sha256:" + "d" * 64,
            "parent_preset_id": "standard_100",
            "parent_prefix_count": 100,
            "nesting_proof_digest": "",
            "estimate": {"status": "unavailable"},
        },
    ]
    for preset in presets:
        preset["nesting_proof_digest"] = canonical_digest(
            {
                "preset_id": preset["preset_id"],
                "scenario_set_digest": preset["scenario_set_digest"],
                "parent_preset_id": preset["parent_preset_id"],
                "parent_prefix_count": preset["parent_prefix_count"],
                "selection_rule": "published_ordered_prefix",
            }
        )
    value: dict[str, object] = {
        "schema_version": "task_evaluation_policy_run_setup.v1",
        "source_launch_id": "scene-839873-source-launch",
        "offering_digest": "sha256:" + "b" * 64,
        "embodiment_id": EMBODIMENT_ID,
        "candidate_ids": list(FROZEN_CANDIDATE_IDS),
        "matrix_profile_id": "franka_rigid_relocation_nested_v1",
        "preregistration": _ref(90),
        "scenario_compiler": {
            "compiler_id": "franka_rigid_relocation_nested_prefix",
            "compiler_version": "v1",
            "selection_rule": "published_ordered_prefix",
            "outcome_independent": True,
            "agent_may_select_cells": False,
        },
        "presets": presets,
        "preparation_template": _template(),
        "setup_digest": "",
    }
    value["setup_digest"] = canonical_digest(value, digest_field="setup_digest")
    return value


def selection(setup_value: dict[str, object], *, preset_id: str = "quick_10") -> dict[str, object]:
    return {
        "schema_version": "task_evaluation_policy_run_selection.v1",
        "run_id": "policy-run-1",
        "source_launch_id": setup_value["source_launch_id"],
        "offering_digest": setup_value["offering_digest"],
        "setup_digest": setup_value["setup_digest"],
        "preset_id": preset_id,
    }


def configuration(setup_value: dict[str, object]) -> dict[str, object]:
    return compile_policy_run_configuration(selection(setup_value), setup=setup_value)


def controls_qualification(
    config: dict[str, object], plan: dict[str, object]
) -> dict[str, object]:
    cells = []
    for index, cell in enumerate(config["matrix"]["cells"]):
        pair = {
            "schema_version": "adp_task_control_pair.v1",
            "cell_id": cell["cell_id"],
            "execution_order": [
                "zero_action_negative",
                "deterministic_scripted_positive",
            ],
            "controls": [
                {
                    "control_id": "zero_action_negative",
                    "control_passed": True,
                    "receipt_digest": f"sha256:{200 + index:064x}",
                },
                {
                    "control_id": "deterministic_scripted_positive",
                    "control_passed": True,
                    "receipt_digest": f"sha256:{300 + index:064x}",
                },
            ],
            "cell_admitted_for_policy_execution": True,
            "policy_execution_blockers": [],
            "candidate_policy_queried": False,
            "pair_digest": "",
        }
        pair["pair_digest"] = canonical_digest(pair, digest_field="pair_digest")
        result = {
            "schema_version": "native_task_arena_control_result.v1",
            "status": "completed",
            "controls_qualified": True,
            "blockers": [],
            "candidate_policy_queried": False,
            "control_pair": pair,
            "result_digest": "",
        }
        result["result_digest"] = canonical_digest(
            result, digest_field="result_digest"
        )
        cells.append(
            {
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "controls_result": result,
            }
        )
    value = {
        "schema_version": "task_evaluation_policy_controls_qualification.v1",
        "configuration_digest": config["configuration_digest"],
        "plan_digest": plan["plan_digest"],
        "cells": cells,
        "candidate_policy_queried": False,
        "blockers": [],
        "qualification_digest": "",
    }
    value["qualification_digest"] = canonical_digest(
        value, digest_field="qualification_digest"
    )
    return value


def result_projection(*, state: str = "decided") -> dict[str, object]:
    per_candidate = 10
    family_metrics = {
        family: {
            "attempted": count,
            "succeeded": 0,
            "success_rate": 0.0,
            "degradation_from_canonical": 0.0,
        }
        for family, count in setup()["presets"][0]["family_counts"].items()
    }
    value: dict[str, object] = {
        "schema_version": "task_evaluation_policy_run_result_projection.v1",
        "run_id": "policy-run-1",
        "source_launch_id": "scene-839873-source-launch",
        "offering_digest": "sha256:" + "b" * 64,
        "configuration_digest": "sha256:" + "c" * 64,
        "plan_digest": "sha256:" + "d" * 64,
        "embodiment_id": EMBODIMENT_ID,
        "candidate_ids": list(FROZEN_CANDIDATE_IDS),
        "state": state,
        "matrix": {
            "scored_cell_count": 10,
            "candidate_episode_count": 20,
            "control_episode_count": 20,
            "expected_episode_count": 40,
            "completed_episode_count": 40 if state == "decided" else 20,
            "identical_candidate_cells_and_seeds": True,
            "controls_complete": state == "decided",
        },
        "candidate_results": [
            {
                "candidate_id": candidate,
                "episodes_completed": per_candidate if state == "decided" else 5,
                "family_metrics": copy.deepcopy(family_metrics),
                "failures": [],
                "contacts": {"contact_count": 20, "violation_count": 0},
                "evidence": {
                    "lossless_frame_manifest_count": per_candidate if state == "decided" else 5,
                    "review_video_count": per_candidate if state == "decided" else 5,
                    "typed_media_gap_count": 0,
                },
            }
            for candidate in FROZEN_CANDIDATE_IDS
        ],
        "paired_comparison": {
            "matched_episode_pairs": per_candidate if state == "decided" else 5,
            "decision": "tie" if state == "decided" else "abstain",
            "deterministic_non_policy_scoring": True,
        },
        "result_delivery_digest": "sha256:" + "e" * 64,
        "blockers": [] if state == "decided" else ["incomplete_episode_matrix"],
        "proof_boundary": {
            "simulation_is_physical_success": False,
            "review_video_is_authoritative_evidence": False,
            "policy_can_grade_itself": False,
            "cross_team_leaderboard_authorized": False,
        },
        "projection_digest": "",
    }
    value["projection_digest"] = policy_run_result_projection_digest(value)
    return value


def test_compiles_exact_pair_and_shared_matrix_without_execution() -> None:
    setup_value = setup()
    config = configuration(setup_value)

    assert validate_policy_run_setup(setup_value) == setup_value
    assert validate_policy_run_configuration(config, setup=setup_value) == config
    plan = build_policy_run_plan(config, setup=setup_value)

    assert plan["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert plan["counts"] == {
        "scored_cell_count": 10,
        "scenarios_per_policy": 10,
        "candidate_episode_count": 20,
        "control_episode_count": 20,
        "total_episode_count": 40,
    }
    assert plan["execution_performed"] is False
    assert plan["provider_mutation_performed"] is False
    assert plan["spend_usd"] == 0


def test_activation_materializes_ten_existing_paired_campaign_units() -> None:
    setup_value = setup()
    config = configuration(setup_value)
    plan = build_policy_run_plan(config, setup=setup_value)

    activation = build_policy_campaign_activation_manifest(
        configuration=config,
        plan=plan,
        controls_qualification=controls_qualification(config, plan),
    )

    assert activation["campaign_unit_count"] == 10
    assert all(
        unit["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
        and unit["runtime_contract"] == "native_task_arena_policy_campaign.v1"
        and unit["provider_allocation_authorized"] is False
        for unit in activation["campaign_units"]
    )
    assert activation["paid_execution_requested"] is False


def test_fail_without_fix_activation_rejects_missing_scored_cell_controls() -> None:
    """The former single-controls predecessor could not prove all ten cells."""

    setup_value = setup()
    config = configuration(setup_value)
    plan = build_policy_run_plan(config, setup=setup_value)
    incomplete = controls_qualification(config, plan)
    incomplete["cells"].pop()
    incomplete["qualification_digest"] = canonical_digest(
        incomplete, digest_field="qualification_digest"
    )
    with pytest.raises(
        TaskEvaluationPolicyRunContractError,
        match="policy_controls_qualification_invalid:cells",
    ):
        build_policy_campaign_activation_manifest(
            configuration=config,
            plan=plan,
            controls_qualification=incomplete,
        )

    unqualified = controls_qualification(config, plan)
    controls_result = unqualified["cells"][0]["controls_result"]
    control_pair = controls_result["control_pair"]
    control_pair["controls"][1]["control_passed"] = False
    control_pair["pair_digest"] = canonical_digest(
        control_pair, digest_field="pair_digest"
    )
    controls_result["result_digest"] = canonical_digest(
        controls_result, digest_field="result_digest"
    )
    unqualified["qualification_digest"] = canonical_digest(
        unqualified, digest_field="qualification_digest"
    )
    with pytest.raises(
        TaskEvaluationPolicyRunContractError,
        match="policy_controls_qualification_cell_result_invalid",
    ):
        build_policy_campaign_activation_manifest(
            configuration=config,
            plan=plan,
            controls_qualification=unqualified,
        )


def test_fail_without_fix_rejects_candidate_family_and_seed_drift() -> None:
    """The old generic episode request could represent each of these drifts."""

    setup_value = setup()
    config = configuration(setup_value)

    swapped = copy.deepcopy(config)
    swapped["candidate_ids"].reverse()
    swapped["configuration_digest"] = canonical_digest(
        swapped, digest_field="configuration_digest"
    )
    with pytest.raises(
        TaskEvaluationPolicyRunContractError,
        match="policy_run_configuration_invalid:candidate_ids",
    ):
        validate_policy_run_configuration(swapped, setup=setup_value)

    missing_family = copy.deepcopy(setup_value)
    missing_family["presets"][0]["cells"][-1]["family"] = "pairwise"
    missing_family["setup_digest"] = canonical_digest(
        missing_family, digest_field="setup_digest"
    )
    with pytest.raises(
        TaskEvaluationPolicyRunContractError,
        match="policy_run_setup_preset_cells_invalid",
    ):
        validate_policy_run_setup(missing_family)

    drifted_seed = copy.deepcopy(config)
    drifted_seed["matrix"]["cells"][3]["seed"] += 100
    drifted_seed["configuration_digest"] = canonical_digest(
        drifted_seed, digest_field="configuration_digest"
    )
    with pytest.raises(
        TaskEvaluationPolicyRunContractError,
        match="policy_run_configuration_not_compiler_output",
    ):
        validate_policy_run_configuration(drifted_seed, setup=setup_value)


def test_catalog_template_expands_into_existing_authenticated_preparation() -> None:
    setup_value = setup()
    selected = selection(setup_value)
    expanded = expand_policy_run_preparation_request(
        setup=setup_value,
        selection=selected,
        expected_production_commit="a" * 40,
        team_namespace="authenticated-team",
        run_id=selected["run_id"],
        preparation_id="friend-policy-run-1-preparation",
    )

    assert expanded["run_mode"] == "episode_evaluation"
    assert expanded["controller"]["kind"] == "policy_container"
    assert expanded["team_namespace"] == "authenticated-team"
    assert expanded["policy_run_configuration"] == configuration(setup_value)
    serialized = json.dumps(expanded["policy_run_configuration"], sort_keys=True)
    assert "email" not in serialized
    assert "team_namespace" not in serialized


def test_launch_catalog_projects_validated_setup_for_webapp_server(tmp_path) -> None:
    profile = launch_profile(tmp_path)
    profile["policy_run_setup"] = setup()
    profile["profile_digest"] = launch_profile_digest(
        profile, digest_field="profile_digest"
    )

    descriptor = public_launch_profile_descriptor(profile)

    assert descriptor["policy_run_setup"]["embodiment_id"] == EMBODIMENT_ID
    assert descriptor["policy_run_setup"]["candidate_ids"] == list(
        FROZEN_CANDIDATE_IDS
    )
    assert validate_public_launch_profile_descriptor(descriptor) == []


def test_worker_seals_policy_plan_into_existing_preparation_queue(tmp_path) -> None:
    setup_value = setup()
    selected = selection(setup_value)
    expanded = expand_policy_run_preparation_request(
        setup=setup_value,
        selection=selected,
        expected_production_commit="a" * 40,
        team_namespace="team-a",
        run_id=selected["run_id"],
        preparation_id="policy-run-queue-1-preparation",
    )
    value, payloads = request_with_fetchable_bytes(expanded)
    inline_setup = value["policy_run_setup"]
    for field in (
        "scene",
        "construction",
        "robot",
        "controller",
        "task",
        "sensors",
        "runtime",
        "execution_adapter",
        "spend",
    ):
        inline_setup["preparation_template"][field] = copy.deepcopy(value[field])
    inline_setup["preparation_template"]["template_digest"] = canonical_digest(
        inline_setup["preparation_template"], digest_field="template_digest"
    )
    inline_setup["setup_digest"] = canonical_digest(
        inline_setup, digest_field="setup_digest"
    )
    value["policy_run_selection"]["setup_digest"] = inline_setup["setup_digest"]
    value["policy_run_configuration"] = compile_policy_run_configuration(
        value["policy_run_selection"], setup=inline_setup
    )
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    run = process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=fake_adapter,
        episode_compilation_queue_root=tmp_path / "episode-compilation",
    )

    result = run["results"][0]
    assert result["status"] == "queued_for_production_episode_compilation", result
    assert result["policy_run_plan"]["configuration_digest"] == value[
        "policy_run_configuration"
    ]["configuration_digest"]
    assert result["policy_run_plan"]["provider_mutation_performed"] is False
    assert result["paid_execution_requested"] is False


def test_terminal_projection_requires_complete_lossless_evidence_for_decision() -> None:
    valid = result_projection()
    assert validate_policy_run_result_projection(valid) == valid

    missing_media = copy.deepcopy(valid)
    missing_media["candidate_results"][0]["evidence"][
        "lossless_frame_manifest_count"
    ] -= 1
    missing_media["projection_digest"] = policy_run_result_projection_digest(
        missing_media
    )
    with pytest.raises(
        TaskEvaluationPolicyRunContractError,
        match="policy_run_result_projection_decision_evidence_incomplete",
    ):
        validate_policy_run_result_projection(missing_media)


def test_terminal_projection_digest_uses_cross_runtime_number_semantics() -> None:
    value = {
        "projection_digest": "",
        "success_rate": 0.0,
        "degradation_from_canonical": 1.0,
    }

    assert policy_run_result_projection_digest(value) == (
        "sha256:9502c7aa38b880a7850d548a11ada1a53fc7bc8215e5439542731ca10c35f6c5"
    )
    assert policy_run_result_projection_digest(value) != canonical_digest(
        value, digest_field="projection_digest"
    )


def test_terminal_abstention_retains_typed_blocker() -> None:
    value = result_projection(state="abstained")
    assert validate_policy_run_result_projection(value)["blockers"] == [
        "incomplete_episode_matrix"
    ]


def test_terminal_sync_binds_policy_projection_to_existing_delivery() -> None:
    plan = EvidencePlan.from_mapping(
        {
            "schema_version": "evidence_plan.v1",
            "plan_id": "policy-plan-1",
            "request_id": "policy-request-1",
            "decision_id": "policy-decision-1",
            "request_digest": "sha256:" + "1" * 64,
            "testbed_id": "policy-testbed-1",
            "testbed_version": "1",
            "testbed_digest": "sha256:" + "2" * 64,
            "claim_plans": [{"claim_id": "paired-policy-comparison"}],
            "execution_order": [],
            "stop_conditions": [],
            "escalation_conditions": [],
            "physical_evidence_requests": [],
            "compiled_evaluation_run_specs": [],
            "non_evaluation_run_steps": [],
            "prohibited_claims": ["physical_success"],
            "shared_dependency_warnings": [],
            "budget_status": {"max_cost_usd": 1.0},
        }
    ).to_mapping()
    envelope = DecisionEnvelope.from_mapping(
        {
            "schema_version": "decision_envelope.v1",
            "decision_id": "policy-decision-1",
            "request_id": "policy-request-1",
            "request_digest": plan["request_digest"],
            "plan_digest": plan["plan_digest"],
            "testbed_digest": plan["testbed_digest"],
            "decision_question": "Which frozen policy performed better?",
            "overall_outcome": "decision",
            "per_claim_verdicts": [{"claim_id": "paired-policy-comparison"}],
            "evidence_accepted": [],
            "evidence_rejected": [],
            "unsupported_conditions": [],
            "cross_method_disagreements": [],
            "shared_dependency_warnings": [],
            "physical_evidence_still_required": [],
            "input_run_result_testbed_digests": [],
            "validation_envelope": {"scope": "matched_simulator_cells"},
            "uncertainty": {"kind": "bounded_empirical"},
            "claim_ceiling": {"class": "development_only"},
            "severity_weighted_false_safe_risk": 0.0,
            "evidence_coverage": 1.0,
            "abstention_rate": 0.0,
            "false_reject_estimate": None,
            "decision_rationale": "Deterministic matched-cell evidence complete.",
            "next_cheapest_experiment": "Held-out physical adjudication.",
            "deployment_approval": False,
            "safety_certification": False,
        }
    ).to_mapping()
    delivery = {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "run_id": "policy-run-1",
        "state": "decided",
        "decision_envelope_digest": envelope["decision_envelope_digest"],
        "delivery_digest": "",
    }
    delivery["delivery_digest"] = canonical_digest(
        delivery, digest_field="delivery_digest"
    )
    policy_result = result_projection()
    policy_result["result_delivery_digest"] = delivery["delivery_digest"]
    policy_result["projection_digest"] = policy_run_result_projection_digest(
        policy_result
    )

    publication = build_task_evaluation_run_webapp_publication(
        capture_session_id="capture-policy-run-1",
        intake_id="intake-policy-run-1",
        run_id="policy-run-1",
        state="decided",
        evidence_plan=plan,
        decision_envelope=envelope,
        result_delivery=delivery,
        policy_run_result=policy_result,
    )

    assert publication["schema_version"] == "task_evaluation_run_publication.v3"
    assert publication["policy_run_result"]["result_delivery_digest"] == delivery[
        "delivery_digest"
    ]
