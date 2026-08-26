from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import jsonschema

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.exact_workcell_variation_matrix import (
    DEFAULT_CELL_COUNT,
    ExactWorkcellVariationError,
    build_agent_proposal_brief,
    compile_variation_matrix,
)
from blueprint_pipeline.exact_workcell_variation_inputs import (
    AgentsSDKVariationProposalAgent,
    ExactWorkcellVariationAgentOutput,
    build_variation_request_from_admitted_contracts,
)
from blueprint_pipeline.exact_workcell_variation_runtime import (
    compile_evaluation_schedule,
    compile_isaac_lab_event_plan,
    evaluation_run_task_scenario_pack,
    publish_variation_bundle,
    validate_matrix_and_schedule,
)
from blueprint_pipeline.evaluation_run_contract import validate_evaluation_run_spec
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _dimension(
    dimension_id: str,
    family: str,
    *,
    nominal: float,
    minimum: float,
    maximum: float,
    unit: str,
) -> dict:
    source_contract = {
        "reset_state": "task",
        "placement_approach": "embodiment",
        "camera_sensor": "embodiment",
        "illumination": "scene",
        "bounded_physics": "task",
        "policy_stochasticity": "task",
    }[family]
    return {
        "dimension_id": dimension_id,
        "family": family,
        "value_type": "continuous",
        "nominal": nominal,
        "minimum": minimum,
        "maximum": maximum,
        "decimals": 6,
        "unit": unit,
        "application_target": f"EventManager.reset.{dimension_id}",
        "application_tolerance": 0.000001,
        "source_contract": source_contract,
        "authority_digest": _sha("d"),
        "exact_workcell_invariant": True,
        "changes_object_or_task_identity": False,
    }


def _request(*, task_id: str = "open_washer", embodiment_id: str = "franka") -> dict:
    value = {
        "schema_version": "exact_workcell_variation_request.v1",
        "program_id": "arm-decision-proof-v1",
        "matrix_id": f"site_840920.{task_id}.{embodiment_id}.baseline_100",
        "matrix_kind": "exact_workcell_primary",
        "implementation_commit": "a" * 40,
        "cell_count": DEFAULT_CELL_COUNT,
        "seed_root": 2026082601,
        "scene_binding": {
            "scene_id": "scene_840920",
            "scene_digest": _sha("1"),
            "coordinate_frame_digest": _sha("2"),
            "canonical_object_asset_id": "articulated_washer",
            "canonical_object_asset_digest": _sha("3"),
        },
        "task_binding": {
            "task_id": task_id,
            "task_digest": _sha("4"),
            "reset_contract_digest": _sha("5"),
            "success_contract_digest": _sha("6"),
        },
        "embodiment_binding": {
            "embodiment_id": embodiment_id,
            "embodiment_digest": _sha("7"),
            "joint_limits_digest": _sha("8"),
            "camera_calibration_digest": _sha("9"),
        },
        "controls": {
            "control_ids": [
                "zero_action_negative",
                "deterministic_scripted_positive",
            ],
            "run_on_every_cell": True,
            "same_resolved_cell_required": True,
        },
        "variation_dimensions": [
            _dimension(
                "door_reset_angle_rad",
                "reset_state",
                nominal=0.0,
                minimum=-0.02,
                maximum=0.02,
                unit="rad",
            ),
            _dimension(
                "robot_base_x_m",
                "placement_approach",
                nominal=1.0,
                minimum=0.98,
                maximum=1.02,
                unit="m",
            ),
            _dimension(
                "external_camera_dx_m",
                "camera_sensor",
                nominal=0.0,
                minimum=-0.005,
                maximum=0.005,
                unit="m",
            ),
            _dimension(
                "light_intensity_scale",
                "illumination",
                nominal=1.0,
                minimum=0.8,
                maximum=1.2,
                unit="ratio",
            ),
            _dimension(
                "hinge_dynamic_friction",
                "bounded_physics",
                nominal=0.4,
                minimum=0.35,
                maximum=0.45,
                unit="ratio",
            ),
        ],
        "object_cousins": [],
    }
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


def _schedule_request(matrix: dict) -> dict:
    value = {
        "schema_version": "exact_workcell_evaluation_schedule_request.v1",
        "matrix_digest": matrix["matrix_digest"],
        "candidate_set": {
            "candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "candidate_identity_digests": {
                "pi05_droid": _sha("a"),
                "groot_n17_droid": _sha("b"),
            },
            "frozen_before_schedule_generation": True,
        },
        "controls": {
            "control_ids": [
                "zero_action_negative",
                "deterministic_scripted_positive",
            ],
            "run_on_every_cell": True,
            "same_resolved_cell_required": True,
        },
        "decision_design": {
            "preregistered_experiment_digest": _sha("c"),
            "power_analysis_digest": _sha("d"),
            "minimum_decision_relevant_difference_digest": _sha("e"),
            "planned_cells_per_candidate": matrix["cell_count"],
            "trial_count_justified_by_preregistered_power_analysis": True,
            "preregistered_before_policy_outcomes": True,
        },
    }
    value["schedule_request_digest"] = canonical_digest(
        value, digest_field="schedule_request_digest"
    )
    return value


def _with_agent_proposal(value: dict) -> dict:
    raw_proposal = {
        "dimension_priorities": [
            {"dimension_id": "door_reset_angle_rad", "weight": 3.0},
            {"dimension_id": "external_camera_dx_m", "weight": 2.0},
        ],
        "targeted_interactions": [
            {
                "dimension_ids": [
                    "door_reset_angle_rad",
                    "hinge_dynamic_friction",
                ],
                "rationale": "Reset and hinge friction jointly bound opening effort.",
            }
        ],
        "object_cousins": [],
    }
    brief = build_agent_proposal_brief(value)
    proposal = {
        "schema_version": "exact_workcell_variation_agent_proposal.v1",
        "status": "proposal_only",
        "model_identity": "bounded-task-failure-agent@test",
        "prompt_digest": brief["brief_digest"],
        "response_digest": canonical_digest({"raw_proposal": raw_proposal}),
        "raw_proposal": raw_proposal,
        "outcome_data_accessed": False,
        "may_widen_admitted_bounds": False,
        "may_change_workcell_or_task_identity": False,
        **raw_proposal,
    }
    proposal["proposal_digest"] = canonical_digest(
        proposal, digest_field="proposal_digest"
    )
    value["agent_proposal"] = proposal
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    return value


class _FixtureVariationAgent:
    model_identity = "fixture-variation-agent@1"

    def propose(self, *, brief):
        assert brief["agent_is_proposal_only"] is True
        return {
            "dimension_priorities": [
                {"dimension_id": "door_reset_angle_rad", "weight": 5.0}
            ],
            "targeted_interactions": [
                {
                    "dimension_ids": [
                        "door_reset_angle_rad",
                        "hinge_dynamic_friction",
                    ],
                    "rationale": "Articulation reset and friction interact.",
                }
            ],
            "object_cousins": [],
        }


class _FixtureAgentsSDKInvoker:
    def __init__(self) -> None:
        self.spec = None
        self.input_value = None

    def invoke(self, spec, input_value):
        self.spec = spec
        self.input_value = input_value
        return AgentsSDKInvocationResult(
            output=ExactWorkcellVariationAgentOutput.model_validate(
                {
                    "dimension_priorities": [
                        {"dimension_id": "door_reset_angle_rad", "weight": 4.0}
                    ],
                    "targeted_interactions": [
                        {
                            "dimension_ids": [
                                "door_reset_angle_rad",
                                "hinge_dynamic_friction",
                            ],
                            "rationale": "Reset angle and hinge friction interact.",
                        }
                    ],
                    "object_cousins": [],
                }
            ),
            provider="openai",
            model=spec.model,
            sdk_version="0.19.1",
            latency_seconds=0.01,
            usage={},
            cost_usd=None,
            cost_status="fixture",
        )


def _source_contract(schema_version: str, **values) -> dict:
    contract = {
        "schema_version": schema_version,
        "measurement_authority_digest": _sha("d"),
        "outcome_data_accessed_for_variation_design": False,
        **values,
    }
    contract["contract_digest"] = canonical_digest(
        contract, digest_field="contract_digest"
    )
    return contract


def test_default_compiler_builds_100_exact_workcell_cells_and_400_episode_schedule() -> None:
    request = _with_agent_proposal(_request())

    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    schedule = compile_evaluation_schedule(
        matrix, request=request, schedule_request=schedule_request
    )
    isaac_plan = compile_isaac_lab_event_plan(matrix, request=request)

    assert matrix["cell_count"] == 100
    assert len(matrix["cells"]) == 100
    assert len({row["cell_id"] for row in matrix["cells"]}) == 100
    assert len({row["cell_digest"] for row in matrix["cells"]}) == 100
    assert len({row["exact_workcell_identity_digest"] for row in matrix["cells"]}) == 1
    assert matrix["cells"][0]["phase"] == "canonical_anchor"
    assert matrix["coverage"]["one_factor_cells"] == 10
    assert matrix["coverage"]["targeted_interaction_cells"] == 1
    assert matrix["coverage"]["held_out_composed_cells"] == 20
    assert matrix["claim_boundary"]["object_cousins_in_primary"] is False
    assert all(row["object_cousin"] is False for row in matrix["cells"])

    assert schedule["episode_count"] == 400
    assert schedule["episodes_per_subject"] == 100
    assert schedule["execution_policy"]["retry_cap"] == 0
    assert schedule["execution_policy"]["no_early_success_stop"] is True
    assert schedule["decision_design"] == schedule_request["decision_design"]
    assert len(
        {
            binding["cell_set_digest"]
            for binding in schedule["subject_bindings"].values()
        }
    ) == 1
    assert set(schedule["subject_bindings"]) == {
        "pi05_droid",
        "groot_n17_droid",
        "zero_action_negative",
        "deterministic_scripted_positive",
    }
    assert all(row["complete_planned_duration_required"] for row in schedule["rows"])
    assert isaac_plan["cell_count"] == 100
    assert all(
        term["readback"]["failure_behavior"]
        == "abstain_cell_before_policy_query"
        for cell in isaac_plan["cells"]
        for term in cell["terms"]
    )
    assert all(
        cell["policy_query_allowed_before_all_readbacks_pass"] is False
        for cell in isaac_plan["cells"]
    )

    schema_root = Path(__file__).parents[1] / "docs" / "schemas"
    for payload, name in (
        (request, "exact_workcell_variation_request.v1.schema.json"),
        (
            schedule_request,
            "exact_workcell_evaluation_schedule_request.v1.schema.json",
        ),
        (matrix, "exact_workcell_variation_matrix.v1.schema.json"),
        (isaac_plan, "exact_workcell_isaac_lab_event_plan.v1.schema.json"),
        (schedule, "exact_workcell_evaluation_schedule.v1.schema.json"),
    ):
        schema = json.loads((schema_root / name).read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(payload)


def test_compilation_is_byte_deterministic_and_dynamic_across_task_and_embodiment() -> None:
    first_request = _request()
    second_request = _request(task_id="pick_part", embodiment_id="ur5e_parallel_jaw")

    first = compile_variation_matrix(first_request)
    repeated = compile_variation_matrix(first_request)
    second = compile_variation_matrix(second_request)

    assert first == repeated
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        repeated, sort_keys=True, separators=(",", ":")
    )
    assert first["matrix_digest"] != second["matrix_digest"]
    assert first["cell_count"] == second["cell_count"] == 100
    assert all(
        row["exact_workcell_identity"]["embodiment_id"] == "ur5e_parallel_jaw"
        for row in second["cells"]
    )
    assert all(
        row["exact_workcell_identity"]["task_id"] == "pick_part"
        for row in second["cells"]
    )


def test_matrix_adapts_to_canonical_evaluation_run_for_each_policy() -> None:
    request = _request()
    matrix = compile_variation_matrix(request)
    pack = evaluation_run_task_scenario_pack(
        matrix,
        request=request,
        matrix_uri="https://objects.example/exact-workcell/matrix.json",
    )

    assert pack["adapter_id"] == "exact_workcell_variation_matrix"
    assert len(pack["scenarios"]) == 100
    assert pack["policy_neutral"] is True
    assert pack["object_cousins_in_primary"] is False
    assert {row["condition_digest"] for row in pack["scenarios"]} == {
        row["cell_digest"] for row in matrix["cells"]
    }

    spec = {
        "schema_version": "evaluation_run.v1",
        "run_id": "exact-workcell-policy-a",
        "mode": "evaluate",
        "scene_bundle": {
            "adapter_id": "capture_site_scene_bundle",
            "adapter_version": "1",
            "bundle_id": "site-840920",
            "uri": "gs://scenes/site-840920",
            "entrypoint": "configured_scene.usd",
            "content_digest": _sha("1"),
        },
        "robot_adapter": {
            "adapter_id": "robot_profile_adapter",
            "adapter_version": "1",
            "robot_profile_id": "franka",
            "asset_ref": "robots/franka.usd",
        },
        "task_scenario_pack": pack,
        "policy_adapter": {
            "adapter_id": "robot_eval_policy_package",
            "adapter_version": "1",
            "policy_id": "pi05_droid",
            "observation_schema_ref": "blueprint://schemas/droid_observation.v1",
            "action_schema_ref": "blueprint://schemas/droid_action.v1",
        },
        "runtime_provider_profile": {
            "adapter_id": "robot_eval_runtime_provider",
            "adapter_version": "1",
            "profile_id": "isaac-runtime",
            "providers": ["fixture_local"],
            "simulator": "isaac_lab",
            "max_spend_usd": 0,
        },
        "proof_contract": {
            "adapter_id": "robot_eval_proof_contract",
            "adapter_version": "1",
            "contract_id": "exact-workcell-proof",
            "required_evidence": ["lossless_policy_frames", "episode_receipt"],
            "claim_ceiling": {"development_only": True},
            "prohibited_claims": ["physical_success"],
        },
    }
    validation = validate_evaluation_run_spec(spec)

    assert validation["status"] == "passed"
    assert validation["adapter_resolution"]["task_scenario_pack"]["status"] == (
        "resolved"
    )


def test_agent_brief_exposes_only_admitted_bounds_and_agent_cannot_authorize() -> None:
    brief = build_agent_proposal_brief(_request())

    assert brief["agent_is_proposal_only"] is True
    assert brief["deterministic_compiler_is_authority"] is True
    assert len(brief["allowed_dimensions"]) == 5
    assert "policy" not in json.dumps(brief["allowed_dimensions"]).lower()
    assert any("Do not include object cousins" in row for row in brief["instructions"])
    assert brief["brief_digest"] == canonical_digest(brief, digest_field="brief_digest")


def test_autonomous_builder_merges_scene_task_embodiment_and_seals_agent_output() -> None:
    original = _request()
    dimensions = original["variation_dimensions"]
    scene = _source_contract(
        "exact_workcell_scene_variation_contract.v1",
        **original["scene_binding"],
        variation_dimensions=dimensions[2:4],
    )
    task = _source_contract(
        "exact_workcell_task_variation_contract.v1",
        **original["task_binding"],
        variation_dimensions=[dimensions[0], dimensions[4]],
    )
    embodiment = _source_contract(
        "exact_workcell_embodiment_variation_contract.v1",
        **original["embodiment_binding"],
        variation_dimensions=[dimensions[1]],
    )

    request = build_variation_request_from_admitted_contracts(
        matrix_id="site_840920.open_washer.franka.baseline_100",
        implementation_commit="a" * 40,
        seed_root=2026082601,
        scene_contract=scene,
        task_contract=task,
        embodiment_contract=embodiment,
        agent=_FixtureVariationAgent(),
    )
    matrix = compile_variation_matrix(request)

    assert len(request["variation_dimensions"]) == 5
    assert {row["source_contract"] for row in request["variation_dimensions"]} == {
        "scene",
        "task",
        "embodiment",
    }
    assert request["agent_proposal"]["model_identity"] == (
        "fixture-variation-agent@1"
    )
    assert request["agent_proposal"]["outcome_data_accessed"] is False
    assert matrix["agent_role"]["mode"] == "bounded_proposal"
    assert matrix["cell_count"] == 100


def test_autonomous_builder_uses_canonical_agents_sdk_adapter() -> None:
    original = _request()
    dimensions = original["variation_dimensions"]
    scene = _source_contract(
        "exact_workcell_scene_variation_contract.v1",
        **original["scene_binding"],
        variation_dimensions=dimensions[2:4],
    )
    task = _source_contract(
        "exact_workcell_task_variation_contract.v1",
        **original["task_binding"],
        variation_dimensions=[dimensions[0], dimensions[4]],
    )
    embodiment = _source_contract(
        "exact_workcell_embodiment_variation_contract.v1",
        **original["embodiment_binding"],
        variation_dimensions=[dimensions[1]],
    )
    invoker = _FixtureAgentsSDKInvoker()
    agent = AgentsSDKVariationProposalAgent(
        invoker=invoker,
        run_id="exact-workcell-agent-fixture",
    )

    request = build_variation_request_from_admitted_contracts(
        matrix_id="site_840920.open_washer.franka.agents_sdk_baseline_100",
        implementation_commit="a" * 40,
        seed_root=2026082602,
        scene_contract=scene,
        task_contract=task,
        embodiment_contract=embodiment,
        agent=agent,
    )
    matrix = compile_variation_matrix(request)

    assert invoker.spec.capability == "exact_workcell_variation_proposal"
    assert invoker.spec.output_type is ExactWorkcellVariationAgentOutput
    assert invoker.spec.model == "gpt-5.6-luna"
    assert invoker.spec.reasoning_effort == "max"
    assert '"brief"' in invoker.input_value
    assert request["agent_proposal"]["model_identity"] == (
        "openai-agents-sdk:gpt-5.6-luna:reasoning=max@0.19.1"
    )
    assert matrix["agent_role"]["mode"] == "bounded_proposal"
    assert matrix["cell_count"] == 100


def test_create_only_publication_preserves_full_byte_readback(tmp_path: Path) -> None:
    request = _with_agent_proposal(_request())

    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    receipt = publish_variation_bundle(
        request,
        schedule_request=schedule_request,
        output_dir=tmp_path / "bundle",
    )

    assert receipt["status"] == "published_create_only_full_byte_readback_verified"
    assert len(receipt["artifacts"]) == 6
    assert all(row["full_byte_readback_verified"] for row in receipt["artifacts"])
    persisted = json.loads(
        (
            tmp_path
            / "bundle"
            / "exact_workcell_variation_publication.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted == receipt
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs"
            / "schemas"
            / "exact_workcell_variation_publication.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(receipt)
    with pytest.raises(ExactWorkcellVariationError, match="publication_output_not_empty"):
        publish_variation_bundle(
            request,
            schedule_request=schedule_request,
            output_dir=tmp_path / "bundle",
        )


def test_publication_rejects_symlink_or_file_output_root(tmp_path: Path) -> None:
    request = _request()
    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    output_file = tmp_path / "output.json"
    output_file.write_text("occupied", encoding="utf-8")

    for output_root in (symlink, output_file):
        with pytest.raises(
            ExactWorkcellVariationError, match="publication_output_path_invalid"
        ):
            publish_variation_bundle(
                request,
                schedule_request=schedule_request,
                output_dir=output_root,
            )


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        (
            lambda value: value.update({"object_cousins": [{"cousin_id": "blue_washer"}]}),
            "object_cousins_forbidden_in_exact_workcell_primary",
        ),
        (
            lambda value: value["variation_dimensions"][0].update(
                {"changes_object_or_task_identity": True}
            ),
            "dimension_identity_mutation_forbidden",
        ),
        (
            lambda value: value["variation_dimensions"][0].update({"maximum": float("nan")}),
            "request_not_finite_json",
        ),
    ],
)
def test_primary_matrix_fails_closed_on_identity_or_authority_drift(
    mutation, blocker: str
) -> None:
    request = _request()
    mutation(request)
    if blocker != "request_not_finite_json":
        request["request_digest"] = canonical_digest(
            request, digest_field="request_digest"
        )

    with pytest.raises(ExactWorkcellVariationError, match=blocker):
        compile_variation_matrix(request)


def test_agent_cannot_propose_unknown_dimension_or_cousin() -> None:
    request = _with_agent_proposal(_request())
    request["agent_proposal"]["dimension_priorities"][0]["dimension_id"] = "invented"
    request["agent_proposal"]["object_cousins"] = ["lookalike"]
    request["agent_proposal"]["proposal_digest"] = canonical_digest(
        request["agent_proposal"], digest_field="proposal_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    with pytest.raises(ExactWorkcellVariationError) as caught:
        compile_variation_matrix(request)

    assert "agent_proposal_dimension_priority_invalid" in caught.value.blockers
    assert "agent_proposal_object_cousins_forbidden_in_primary" in caught.value.blockers


def test_agent_projection_must_match_digest_bound_raw_response() -> None:
    request = _with_agent_proposal(_request())
    request["agent_proposal"]["dimension_priorities"] = []
    request["agent_proposal"]["proposal_digest"] = canonical_digest(
        request["agent_proposal"], digest_field="proposal_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    with pytest.raises(
        ExactWorkcellVariationError,
        match="agent_proposal_raw_response_binding_mismatch:dimension_priorities",
    ):
        compile_variation_matrix(request)


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        (
            lambda dimension: dimension.update(
                {"minimum": 0.00001, "nominal": 0.000015, "maximum": 0.00002, "decimals": 3}
            ),
            "dimension_continuous_levels_collapse",
        ),
        (
            lambda dimension: dimension.update(
                {
                    "value_type": "categorical",
                    "nominal": "fixed",
                    "values": ["fixed", "fixed"],
                }
            ),
            "dimension_categorical_values_invalid",
        ),
        (
            lambda dimension: dimension.update(
                {
                    "value_type": "integer",
                    "minimum": 0.5,
                    "nominal": 1.0,
                    "maximum": 2.0,
                }
            ),
            "dimension_integer_levels_invalid",
        ),
        (
            lambda dimension: dimension.update({"source_contract": "agent"}),
            "dimension_source_contract_invalid",
        ),
        (
            lambda dimension: dimension.pop("source_contract"),
            "dimension_source_contract_invalid",
        ),
        (
            lambda dimension: dimension.update({"unit": ""}),
            "dimension_unit_missing",
        ),
    ],
)
def test_dimension_levels_and_authority_fail_closed(mutation, blocker: str) -> None:
    request = _request()
    mutation(request["variation_dimensions"][0])
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    with pytest.raises(ExactWorkcellVariationError, match=blocker):
        compile_variation_matrix(request)


def test_malformed_dimension_collection_returns_stable_blocker() -> None:
    request = _request()
    request["variation_dimensions"] = 7
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    with pytest.raises(
        ExactWorkcellVariationError, match="variation_dimensions_missing_or_invalid"
    ):
        compile_variation_matrix(request)


def test_tampered_matrix_or_schedule_is_rejected() -> None:
    request = _request()
    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    schedule = compile_evaluation_schedule(
        matrix, request=request, schedule_request=schedule_request
    )

    tampered_matrix = copy.deepcopy(matrix)
    tampered_matrix["cells"][1]["resolved_values"]["robot_base_x_m"] = 99.0
    with pytest.raises(ExactWorkcellVariationError, match="matrix_validation_mismatch"):
        validate_matrix_and_schedule(
            request=request, matrix=tampered_matrix, schedule=schedule
            , schedule_request=schedule_request
        )

    tampered_schedule = copy.deepcopy(schedule)
    tampered_schedule["rows"][0]["cell_digest"] = _sha("f")
    with pytest.raises(ExactWorkcellVariationError, match="schedule_validation_mismatch"):
        validate_matrix_and_schedule(
            request=request,
            schedule_request=schedule_request,
            matrix=matrix,
            schedule=tampered_schedule,
        )


def test_candidate_binding_is_later_and_cannot_change_the_matrix() -> None:
    request = _request()
    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    schedule_request["candidate_set"]["candidate_ids"] = ["same", "same"]
    schedule_request["schedule_request_digest"] = canonical_digest(
        schedule_request, digest_field="schedule_request_digest"
    )

    with pytest.raises(
        ExactWorkcellVariationError,
        match="candidate_set_exactly_two_distinct_required",
    ):
        compile_evaluation_schedule(
            matrix, request=request, schedule_request=schedule_request
        )

    request_with_candidates = copy.deepcopy(request)
    request_with_candidates["candidate_set"] = {"candidate_ids": ["a", "b"]}
    request_with_candidates["request_digest"] = canonical_digest(
        request_with_candidates, digest_field="request_digest"
    )
    with pytest.raises(
        ExactWorkcellVariationError,
        match="variation_request_candidate_set_forbidden_policy_neutral_matrix",
    ):
        compile_variation_matrix(request_with_candidates)


def test_candidate_identity_cannot_impersonate_a_control() -> None:
    request = _request()
    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    schedule_request["candidate_set"] = {
        "candidate_ids": ["zero_action_negative", "groot_n17_droid"],
        "candidate_identity_digests": {
            "zero_action_negative": _sha("a"),
            "groot_n17_droid": _sha("b"),
        },
        "frozen_before_schedule_generation": True,
    }
    schedule_request["schedule_request_digest"] = canonical_digest(
        schedule_request, digest_field="schedule_request_digest"
    )

    with pytest.raises(
        ExactWorkcellVariationError,
        match="candidate_id_collides_with_required_control",
    ):
        compile_evaluation_schedule(
            matrix, request=request, schedule_request=schedule_request
        )


def test_schedule_requires_power_justified_trial_count_for_this_matrix() -> None:
    request = _request()
    matrix = compile_variation_matrix(request)
    schedule_request = _schedule_request(matrix)
    schedule_request["decision_design"]["planned_cells_per_candidate"] = 99
    schedule_request["decision_design"][
        "trial_count_justified_by_preregistered_power_analysis"
    ] = False
    schedule_request["schedule_request_digest"] = canonical_digest(
        schedule_request, digest_field="schedule_request_digest"
    )

    with pytest.raises(ExactWorkcellVariationError) as caught:
        compile_evaluation_schedule(
            matrix, request=request, schedule_request=schedule_request
        )

    assert "decision_design_trial_count_matrix_mismatch" in caught.value.blockers
    assert "decision_design_power_justification_missing" in caught.value.blockers


def test_insufficient_100_cell_budget_for_one_factor_contract_fails_closed() -> None:
    request = _request()
    request["variation_dimensions"] = [
        _dimension(
            f"safe_parameter_{index}",
            "bounded_physics",
            nominal=1.0,
            minimum=0.9,
            maximum=1.1,
            unit="ratio",
        )
        for index in range(50)
    ]
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")

    with pytest.raises(
        ExactWorkcellVariationError,
        match="cell_budget_insufficient_for_one_factor_diagnosis",
    ):
        compile_variation_matrix(request)
