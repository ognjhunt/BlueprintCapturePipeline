from __future__ import annotations

from dataclasses import dataclass

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_construction_feedback_controller import (
    AUTHORITY_SCHEMA_VERSION,
    CANDIDATE_SCHEMA_VERSION,
    CONTROLS_CONTINUATION_SCHEMA_VERSION,
    EXECUTION_SCHEMA_VERSION,
    INVENTORY_SCHEMA_VERSION,
    NativeConstructionFeedbackControllerError,
    build_next_native_construction_inventory,
    run_native_construction_feedback_controller,
    summarize_native_construction_feedback,
    validate_native_construction_candidate,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)


def _sealed(value: dict, field: str) -> dict:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _candidate(
    candidate_id: str,
    rank: int,
    *,
    x: float,
    addressed_feedback_codes: list[str] | None = None,
) -> dict:
    reset = _sealed(
        {
            "schema_version": "task_evaluation_native_robot_reset_variant.v1",
            "robot_joint_reset_positions_rad": {
                f"joint_{index}": value
                for index, value in enumerate((0.0, -0.62, 0.1, -1.36, 1.35, 1.75, -0.72))
            },
            "reset_variant_digest": "",
        },
        "reset_variant_digest",
    )
    entry = _sealed(
        {
            "schema_version": "task_evaluation_native_entry_trajectory_variant.v1",
            "joins_authored_phase_id": "precontact",
            "waypoints": [
                {
                    "waypoint_id": "entry-00",
                    "position_world_m": [2.8, -6.7, 1.0],
                    "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                }
            ],
            "entry_trajectory_variant_digest": "",
        },
        "entry_trajectory_variant_digest",
    )
    camera = _sealed(
        {
            "schema_version": "task_evaluation_native_camera_variant.v1",
            "cameras": [
                {
                    "role": "external",
                    "pose_frame": "world",
                    "position_world_m": [2.9, -6.1, 1.8],
                    "target_world_m": [3.0, -6.7, 0.82],
                },
                {
                    "role": "wrist",
                    "pose_frame": "robot_body",
                    "configuration_id": "profile-wrist-default",
                },
                {
                    "role": "overview",
                    "pose_frame": "world",
                    "position_world_m": [3.0, -5.8, 2.2],
                    "target_world_m": [3.0, -6.7, 0.82],
                },
            ],
            "camera_variant_digest": "",
        },
        "camera_variant_digest",
    )
    return _sealed(
        {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "deterministic_rank": rank,
            "robot_base_pose_world": {
                "position_world_m": [x, -6.13, 0.752958],
                "orientation_xyzw": [0.0, 0.0, 0.608761429, -0.79335334],
            },
            "support_surface_id": "/Site/counter",
            "reset_variant": reset,
            "entry_trajectory_variant": entry,
            "camera_variant": camera,
            "maximum_incremental_cost_usd": 0.1,
            "maximum_runtime_seconds": 120.0,
            "addressed_feedback_codes": addressed_feedback_codes or [],
            "candidate_digest": "",
        },
        "candidate_digest",
    )


def _inventory(
    run_id: str,
    round_index: int,
    candidates: list[dict],
    *,
    feedback_digest: str | None = None,
) -> dict:
    return _sealed(
        {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "run_id": run_id,
            "round_index": round_index,
            "source_native_feedback_digest": feedback_digest,
            "model_authored_candidates": False,
            "candidates": candidates,
            "inventory_digest": "",
        },
        "inventory_digest",
    )


def _native(*, passed: bool, collision_force: float = 0.0) -> dict:
    native = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "completed" if passed else "blocked",
        "construction_gate_qualified": passed,
        "blockers": [] if passed else [
            "native_rigid_construction_gate_failed:push_contact_maintained"
        ],
        "initial_readback": {
            "robot_root_pose_world": [2.92, -6.13, 0.752958, 0.0, 0.0, 0.6, -0.8],
            "task_sample": {
                "task_scoring_pose_world": [2.97, -6.76, 0.818, 0.0, 0.0, 0.0, 1.0]
            },
        },
        "phase_results": [
            {
                "phase_id": "precontact",
                "steps": 40,
                "target_reached": True,
                "terminal_position_error_m": 0.001,
                "terminal_orientation_error_rad": 0.01,
                "task_sample": {
                    "task_scoring_pose_world": [
                        3.073,
                        -6.76,
                        0.818,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "task_robot_contact_peak_force_n": 70.2,
                    "task_support_contact_peak_force_n": 4.9,
                    "task_scene_collision_peak_force_n": 0.0,
                    "robot_scene_contact_peak_force_n": 0.0,
                    "robot_task_forbidden_collision_peak_force_n": collision_force,
                    "native_readback": {
                        "contact_sensor_instance_readback": [
                            {
                                "logical_sensor_id": "robot_task_forbidden_collision",
                                "sensor_instance_id": "forbidden__link7",
                            }
                        ]
                    },
                },
            },
            {
                "phase_id": "push_contact",
                "steps": 6,
                "target_reached": passed,
                "terminal_position_error_m": 0.074 if not passed else 0.001,
                "terminal_orientation_error_rad": 0.02,
                "task_sample": {
                    "task_scoring_pose_world": [
                        3.08,
                        -6.78,
                        0.818,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    "task_robot_contact_peak_force_n": 3.2,
                    "task_support_contact_peak_force_n": 4.9,
                    "task_scene_collision_peak_force_n": 0.0,
                    "robot_scene_contact_peak_force_n": 0.0,
                    "robot_task_forbidden_collision_peak_force_n": 0.0,
                },
            },
        ],
        "camera_gates": {
            "external": {
                "passed": passed,
                "best_snapshot_id": "precontact",
                "site_appearance_claimed": passed,
                "best_observability": {
                    "pixel_count": 170,
                    "pixel_fraction": 0.00295,
                    "centroid_xy_fraction": [0.49, 0.45],
                    "blockers": [] if passed else ["site_appearance_missing"],
                    "render_evidence": {
                        "site_rendered": passed,
                        "dominant_rgb_pixel_fraction": 0.62 if not passed else 0.01,
                    },
                },
            }
        },
        "result_digest": "",
    }
    return _sealed(native, "result_digest")


def _authority(run_id: str) -> dict:
    return _sealed(
        {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "run_id": run_id,
            "expected_provider_instance_id": 49322931,
            "warm_session_digest": "sha256:" + "a" * 64,
            "allocator_retry_cap": 0,
            "maximum_rounds": 3,
            "maximum_candidates_per_round": 8,
            "maximum_incremental_cost_usd": 0.5,
            "deadline_unix_s": 2_000.0,
            "authority_digest": "",
        },
        "authority_digest",
    )


@dataclass
class _Invoker:
    selections: list[tuple[str, str | None]]

    def __post_init__(self) -> None:
        self.inputs = []

    def invoke(self, spec, input_value):
        self.inputs.append((spec, input_value))
        candidate_id, feedback_digest = self.selections.pop(0)
        prompt = __import__("json").loads(input_value[0]["content"])
        candidate = next(
            row for row in prompt["candidates"] if row["candidate_id"] == candidate_id
        )
        return AgentsSDKInvocationResult(
            output={
                "inventory_digest": prompt["inventory_digest"],
                "candidate_id": candidate_id,
                "candidate_digest": candidate["candidate_digest"],
                "addressed_feedback_digest": feedback_digest,
                "rationale": "Select the exact deterministic member addressing measured contact.",
            },
            provider="openai",
            model=spec.model,
            sdk_version="0.19.1",
            latency_seconds=0.01,
            usage={"total_tokens": 12},
            cost_usd=None,
            cost_status="test",
            trace_id="trace-test",
        )


def _execution(candidate: dict, inventory_digest: str, *, passed: bool) -> dict:
    return _sealed(
        {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "status": "passed" if passed else "rejected",
            "candidate_id": candidate["candidate_id"],
            "candidate_digest": candidate["candidate_digest"],
            "inventory_digest": inventory_digest,
            "provider_instance_id": 49322931,
            "provider_allocations_performed": 0,
            "incremental_cost_upper_bound_usd": 0.04,
            "native_result": _native(
                passed=passed, collision_force=0.6 if not passed else 0.0
            ),
            "execution_result_digest": "",
        },
        "execution_result_digest",
    )


def test_feedback_names_first_collision_contact_displacement_and_camera() -> None:
    feedback = summarize_native_construction_feedback(
        _native(passed=False, collision_force=0.602)
    )

    assert feedback["passed"] is False
    assert feedback["first_failed_phase"] == "push_contact"
    assert feedback["first_collision"] == {
        "phase_id": "precontact",
        "channel": "robot_task_forbidden_collision",
        "peak_force_n": pytest.approx(0.602),
        "link_or_sensor_id": "forbidden__link7",
        "measurement_only_not_regraded": True,
    }
    precontact = feedback["phase_measurements"][0]
    assert precontact["contacts"]["task_robot_contact_peak_force_n"] == pytest.approx(70.2)
    assert precontact["task_displacement_from_reset_m"] == pytest.approx(0.103)
    assert feedback["camera_measurements"]["external"]["site_rendered"] is False
    assert feedback["feedback_digest"] == canonical_digest(
        feedback, digest_field="feedback_digest"
    )


def test_one_allocation_runs_feedback_rounds_then_automatically_continues_controls() -> None:
    run_id = "scene-839873-construction-feedback"
    first = _candidate("base-reset-a", 0, x=2.92)
    second = _candidate("entry-clearance-b", 0, x=3.04)
    invoker = _Invoker(
        selections=[
            (first["candidate_id"], None),
            # Filled after the first native result by the producer below.
            (second["candidate_id"], "__feedback__"),
        ]
    )
    execution_calls = []
    controls_calls = []

    def execute(candidate, binding):
        execution_calls.append((candidate, binding))
        return _execution(
            candidate,
            binding["inventory_digest"],
            passed=len(execution_calls) == 2,
        )

    def next_inventory(feedback, history, round_index):
        assert round_index == 1
        assert len(history) == 1
        assert feedback["first_collision"]["phase_id"] == "precontact"
        invoker.selections[0] = (second["candidate_id"], feedback["feedback_digest"])
        return _inventory(
            run_id,
            round_index,
            [second],
            feedback_digest=feedback["feedback_digest"],
        )

    def controls(receipt):
        controls_calls.append(receipt)
        return _sealed(
            {
                "schema_version": CONTROLS_CONTINUATION_SCHEMA_VERSION,
                "status": "queued",
                "run_id": run_id,
                "construction_qualification_digest": receipt[
                    "construction_qualification_digest"
                ],
                "qualified_candidate_digest": second["candidate_digest"],
                "provider_instance_id": 49322931,
                "provider_allocations_performed": 0,
                "controls_continuation_digest": "",
            },
            "controls_continuation_digest",
        )

    receipt = run_native_construction_feedback_controller(
        invoker=invoker,
        authority=_authority(run_id),
        initial_inventory=_inventory(run_id, 0, [first]),
        produce_next_inventory=next_inventory,
        execute_candidate=execute,
        continue_to_controls=controls,
        clock=lambda: 1_000.0,
    )

    assert receipt["status"] == "controls_continuation_queued"
    assert receipt["round_count"] == 2
    assert receipt["provider_allocations_performed"] == 0
    assert receipt["allocator_retry_cap"] == 0
    assert receipt["attempted_candidate_digests"] == [
        first["candidate_digest"],
        second["candidate_digest"],
    ]
    assert receipt["incremental_cost_upper_bound_usd"] == pytest.approx(0.08)
    assert len(controls_calls) == 1
    assert all(
        row[1]["expected_provider_instance_id"] == 49322931
        for row in execution_calls
    )
    # The agent received measurements and digests, not authority to edit gates.
    second_prompt = __import__("json").loads(invoker.inputs[1][1][0]["content"])
    assert second_prompt["source_native_feedback"]["first_collision"]["phase_id"] == "precontact"
    assert second_prompt["authority_boundary"]["model_may_not_change_gates_or_thresholds"] is True


def test_candidate_cannot_carry_a_gate_change() -> None:
    candidate = _candidate("bad-threshold", 0, x=2.92)
    candidate["entry_trajectory_variant"]["arrival_tolerance_m"] = 0.5
    candidate["entry_trajectory_variant"]["entry_trajectory_variant_digest"] = canonical_digest(
        candidate["entry_trajectory_variant"],
        digest_field="entry_trajectory_variant_digest",
    )
    candidate["candidate_digest"] = canonical_digest(
        candidate, digest_field="candidate_digest"
    )

    with pytest.raises(
        NativeConstructionFeedbackControllerError,
        match="gate_mutation_forbidden",
    ):
        validate_native_construction_candidate(candidate)


def test_next_inventory_deterministically_prefers_feedback_coverage_and_excludes_attempted() -> None:
    run_id = "scene-839873-deterministic-refresh"
    attempted = _candidate("attempted", 0, x=2.92)
    generic = _candidate("generic", 1, x=3.0)
    collision_clear = _candidate(
        "collision-clear",
        7,
        x=3.04,
        addressed_feedback_codes=[
            "collision:precontact:robot_task_forbidden_collision"
        ],
    )
    feedback = summarize_native_construction_feedback(
        _native(passed=False, collision_force=0.602)
    )

    inventory = build_next_native_construction_inventory(
        run_id=run_id,
        round_index=1,
        source_native_feedback=feedback,
        prior_history=[{"candidate": attempted}],
        candidate_universe=[attempted, generic, collision_clear],
        maximum_candidates=8,
    )

    assert [row["candidate_id"] for row in inventory["candidates"]] == [
        "collision-clear",
        "generic",
    ]
    assert attempted["candidate_digest"] not in {
        row["candidate_digest"] for row in inventory["candidates"]
    }
    assert inventory["source_native_feedback_codes"][0].startswith("camera_")
    assert inventory["inventory_digest"] == canonical_digest(
        inventory, digest_field="inventory_digest"
    )


def test_execution_must_echo_the_exact_candidate_and_allocate_nothing() -> None:
    run_id = "scene-839873-mutation-refusal"
    candidate = _candidate("exact-a", 0, x=2.92)
    invoker = _Invoker(selections=[(candidate["candidate_id"], None)])

    def execute(selected, binding):
        result = _execution(selected, binding["inventory_digest"], passed=True)
        result["candidate_digest"] = "sha256:" + "f" * 64
        result["provider_allocations_performed"] = 1
        result["execution_result_digest"] = canonical_digest(
            result, digest_field="execution_result_digest"
        )
        return result

    with pytest.raises(
        NativeConstructionFeedbackControllerError,
        match="candidate_execution_invalid",
    ):
        run_native_construction_feedback_controller(
            invoker=invoker,
            authority=_authority(run_id),
            initial_inventory=_inventory(run_id, 0, [candidate]),
            produce_next_inventory=lambda *_: pytest.fail("must not refresh"),
            execute_candidate=execute,
            continue_to_controls=lambda _: pytest.fail("must not continue"),
            clock=lambda: 1_000.0,
        )
