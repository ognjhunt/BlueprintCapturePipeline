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
    CompositeCandidateGenerator,
    build_next_native_construction_inventory,
    construction_phase_plan_for_candidate,
    run_native_construction_feedback_controller,
    summarize_native_construction_feedback,
    validate_native_construction_candidate,
)
from blueprint_pipeline.task_evaluation_supervisor.agents_sdk import (
    AgentsSDKInvocationResult,
)
from blueprint_pipeline.task_evaluation_robot_placement_warm_executor import (
    FEEDBACK_EXECUTOR_CONFIG_SCHEMA_VERSION,
    WarmNativeConstructionFeedbackExecutor,
    run_retained_native_construction_feedback,
)
from blueprint_pipeline.task_evaluation_native_construction_terminal_feedback import (
    main as terminal_feedback_main,
    materialize_native_construction_terminal_feedback_adoption,
    validate_native_construction_terminal_feedback_adoption,
)
from blueprint_pipeline.native_task_arena_feedback_bootstrap_runtime import (
    feedback_bootstrap_result,
)
from blueprint_pipeline.native_task_arena_feedback_allocator_adapter import (
    terminal_feedback_bootstrap_blockers,
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
            "runtime_seconds": 10.0,
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
        "sample_index": 0,
        "sample_kind": "terminal",
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


def test_feedback_scans_all_samples_and_maps_native_gate_objectives() -> None:
    native = _native(passed=False, collision_force=0.0)
    precontact = native["phase_results"][0]
    samples = []
    for index in range(44):
        samples.append(
            {
                "task_scoring_pose_world": [
                    2.97,
                    -6.76,
                    0.818,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                "robot_scene_contact_peak_force_n": (
                    59.564 if index == 43 else 0.0
                ),
                "robot_task_forbidden_collision_peak_force_n": 0.0,
                "task_scene_collision_peak_force_n": 0.0,
                "task_robot_contact_peak_force_n": 0.0,
                "task_support_contact_peak_force_n": 4.9,
            }
        )
    precontact["task_samples"] = samples
    precontact["task_sample"]["robot_scene_contact_peak_force_n"] = 0.0
    native["blockers"] = [
        "native_rigid_construction_gate_failed:base_collision_clearance",
        "native_rigid_construction_gate_failed:destination_containment",
        "native_rigid_construction_gate_failed:push_contact_maintained",
        "native_rigid_construction_gate_failed:push_path",
    ]
    native["construction_phase_plan"] = {
        "task_kind": "rigid_pick_place",
        "thresholds": {"task_contact_minimum_force_n": 1.0},
        "phases": [
            {
                "phase_id": "precontact",
                "expected_scoring_position_world_m": [2.97, -6.76, 0.818],
            },
            {
                "phase_id": "push_contact",
                "expected_scoring_position_world_m": [3.0, -6.76, 0.818],
            },
        ],
        "destination_position_world_m": [3.1, -6.76, 0.818],
    }
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")

    feedback = summarize_native_construction_feedback(native)

    assert feedback["first_collision"]["sample_index"] == 43
    assert feedback["first_collision"]["peak_force_n"] == pytest.approx(59.564)
    assert feedback["peak_collision"]["peak_force_n"] == pytest.approx(59.564)
    assert feedback["feedback_codes"] == [
        "gate_failed:base_collision_clearance",
        "gate_failed:destination_containment",
        "gate_failed:push_contact_maintained",
        "gate_failed:push_path",
    ]
    objectives = feedback["physics_objective_measurements"]
    assert objectives["forbidden_robot_scene_collision_peak_force_n"] == pytest.approx(
        59.564
    )
    assert objectives["measurement_digest"] == canonical_digest(
        objectives, digest_field="measurement_digest"
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
    assert receipt["incremental_cost_upper_bound_usd"] == pytest.approx(0.2)
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


def test_composite_generator_records_planner_unavailability_and_uses_cpu_baseline() -> None:
    run_id = "scene-839873-composite-generator"
    baseline_inventory = _inventory(
        run_id, 1, [_candidate("baseline", 0, x=2.92)], feedback_digest=None
    )

    class MissingCurobo:
        def generate(self, **_kwargs):
            raise RuntimeError("runtime unavailable")

    class DeterministicBaseline:
        def generate(self, **_kwargs):
            return baseline_inventory

    generated = CompositeCandidateGenerator(
        generators=(MissingCurobo(),),
        deterministic_fallback=DeterministicBaseline(),
    ).generate(
        source_native_feedback=None,
        prior_history=(),
        round_index=1,
        maximum_candidates=8,
    )

    assert generated["candidate_generator_chain"] == [
        {"generator": "MissingCurobo", "status_code": "unavailable:RuntimeError"},
        {
            "generator": "DeterministicBaseline",
            "status_code": "selected_deterministic_baseline",
        },
    ]
    assert generated["inventory_digest"] == canonical_digest(
        generated, digest_field="inventory_digest"
    )


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


def test_terminal_result_cli_seals_nontrial_feedback_bootstrap(
    tmp_path, capsys
) -> None:
    run_id = "scene-839873-terminal-adoption"
    universe = _inventory(run_id, 0, [_candidate("baseline-family", 0, x=2.92)])
    packet = tmp_path / "packet"
    packet.mkdir()
    request = _sealed(
        {
            "schema_version": "native_task_arena_packet_request.v1",
            "robot_base_pose_world": {
                "position_world_m": [2.92, -6.13, 0.752958],
                "orientation_xyzw": [0.0, 0.0, 0.6, -0.8],
            },
            "robot_joint_reset_positions_rad": {
                f"panda_joint{index}": 0.01 * index for index in range(1, 8)
            },
            "cameras": [{"role": role} for role in ("external", "overview", "wrist")],
            "native_construction_feedback": {
                "selected_placement_candidate_id": "geometry-selected",
                "candidate_universe": universe,
            },
            "request_digest": "",
        },
        "request_digest",
    )
    (packet / "native_task_arena_packet_request.v1.json").write_text(
        __import__("json").dumps(request) + "\n", encoding="utf-8"
    )
    native = _native(passed=False, collision_force=0.6)
    native_path = tmp_path / "native.json"
    native_path.write_text(
        __import__("json").dumps(native) + "\n", encoding="utf-8"
    )
    allocator = {
        "schema_version": "native_task_arena_vast_run.v1",
        "status": "blocked",
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "warm_session": None,
        "warm_session_receipt_path": None,
        "native_control_result_path": str(native_path),
        "native_control_result_digest": native["result_digest"],
        "estimated_cost_usd": 0.095,
        "result_digest": "",
    }
    allocator["result_digest"] = canonical_digest(
        allocator, digest_field="result_digest"
    )
    allocator_path = tmp_path / "allocator.json"
    allocator_path.write_text(
        __import__("json").dumps(allocator) + "\n", encoding="utf-8"
    )
    output = tmp_path / "adoption.json"

    assert terminal_feedback_main(
        [
            "--allocator-result",
            str(allocator_path),
            "--native-result",
            str(native_path),
            "--packet-dir",
            str(packet),
            "--output",
            str(output),
        ]
    ) == 0
    checkpoint = validate_native_construction_terminal_feedback_adoption(
        __import__("json").loads(output.read_text())
    )
    assert checkpoint["initial_native_feedback"]["passed"] is False
    assert checkpoint["prior_attempted_baseline_binding"][
        "optuna_trial_recorded"
    ] is False
    assert checkpoint["prior_attempted_baseline_binding"]["candidate_digest"] is None
    assert checkpoint["prior_attempted_candidate_digests"] == []
    assert checkpoint["baseline_physics_replay_required"] is False
    assert __import__("json").loads(capsys.readouterr().out)["status"] == (
        "accepted_for_feedback_bootstrap"
    )
    runtime = tmp_path / "runtime"
    runtime_input = runtime / "runtime_inputs"
    runtime_input.mkdir(parents=True)
    staged_adoption = (
        runtime_input
        / "native_construction_terminal_feedback_adoption.v1.json"
    )
    staged_adoption.write_bytes(output.read_bytes())
    manifest = {
        "bound_runtime_inputs": [
            {
                "relative_path": "runtime_inputs/" + staged_adoption.name,
                "size_bytes": staged_adoption.stat().st_size,
                "sha256": "sha256:"
                + __import__("hashlib").sha256(staged_adoption.read_bytes()).hexdigest(),
            }
        ]
    }
    bootstrap = feedback_bootstrap_result(
        runtime=runtime, manifest=manifest, packet=packet
    )
    assert bootstrap["feedback_bootstrap_only"] is True
    assert bootstrap["baseline_physics_replayed"] is False
    assert bootstrap["terminal_feedback_adoption_digest"] == checkpoint[
        "checkpoint_digest"
    ]
    adoption_sha = "sha256:" + __import__("hashlib").sha256(
        output.read_bytes()
    ).hexdigest()
    prepared = {
        "bound_runtime_inputs": [
            {
                "relative_path": (
                    "runtime_inputs/"
                    "native_construction_terminal_feedback_adoption.v1.json"
                ),
                "sha256": adoption_sha,
            }
        ]
    }
    assert terminal_feedback_bootstrap_blockers(
        packet_dir=packet,
        prepared_bundle=prepared,
        adoption_path=output,
    ) == []
    prepared["bound_runtime_inputs"][0]["sha256"] = "sha256:" + "0" * 64
    assert terminal_feedback_bootstrap_blockers(
        packet_dir=packet,
        prepared_bundle=prepared,
        adoption_path=output,
    ) == ["native_construction_terminal_feedback_bootstrap_invalid"]

    changed = dict(native)
    changed["blockers"] = ["mutated"]
    with pytest.raises(ValueError, match="terminal_feedback_evidence_invalid"):
        materialize_native_construction_terminal_feedback_adoption(
            allocator_result=allocator,
            native_result=changed,
            packet_dir=packet,
            output_path=tmp_path / "rejected.json",
        )


def test_entry_variant_prepends_motion_without_changing_authored_gates(monkeypatch) -> None:
    import blueprint_pipeline.native_task_construction_plan as plans

    authored = {
        "schema_version": "native_rigid_construction_phase_plan.v1",
        "scene_plan_digest": "sha256:" + "1" * 64,
        "execution_parameters": {
            "stable_samples": 2,
            "maximum_construction_total_steps": 220,
        },
        "thresholds": {"collision_failure_minimum_force_n": 1.0},
        "gate_contract": {"push_contact": "native_contact"},
        "required_gate_ids": ["push_contact"],
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.79, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "gripper_state": "open",
                "gate_ids": ["push_contact"],
            }
        ],
        "phase_count": 1,
        "plan_digest": "",
    }
    authored["plan_digest"] = canonical_digest(
        authored, digest_field="plan_digest"
    )
    monkeypatch.setattr(
        plans,
        "materialize_native_task_construction_phase_plan",
        lambda _scene: __import__("copy").deepcopy(authored),
    )
    scene = {
        "plan_digest": authored["scene_plan_digest"],
        "cadence": {"maximum_action_steps": 240},
    }

    result = construction_phase_plan_for_candidate(
        scene_plan=scene,
        candidate=_candidate("entry-a", 0, x=2.92),
    )

    assert [row["phase_id"] for row in result["phases"]] == [
        "feedback_entry_00_entry-00",
        "precontact",
    ]
    assert result["phases"][0]["gate_ids"] == []
    assert result["phases"][1] == authored["phases"][0]
    assert result["thresholds"] == authored["thresholds"]
    assert result["gate_contract"] == authored["gate_contract"]
    assert result["execution_parameters"]["maximum_construction_total_steps"] == 222
    assert result["authored_gate_contract_unchanged"] is True
    assert result["plan_digest"] == canonical_digest(
        result, digest_field="plan_digest"
    )


def test_curobo_entry_and_approach_joint_paths_are_bound_into_native_plan(
    monkeypatch,
) -> None:
    import blueprint_pipeline.native_task_construction_plan as plans

    authored = _sealed(
        {
            "schema_version": "native_rigid_construction_phase_plan.v1",
            "scene_plan_digest": "sha256:" + "1" * 64,
            "execution_parameters": {
                "stable_samples": 2,
                "maximum_construction_total_steps": 220,
            },
            "thresholds": {"collision_failure_minimum_force_n": 1.0},
            "gate_contract": {"push_contact": "native_contact"},
            "required_gate_ids": ["push_contact"],
            "phases": [
                {
                    "phase_id": "precontact",
                    "position_world_m": [2.79, -6.76, 0.818],
                    "orientation_world_xyzw": [
                        0.0,
                        0.70710678,
                        0.0,
                        0.70710678,
                    ],
                    "gripper_state": "open",
                    "gate_ids": ["push_contact"],
                },
                *[
                    {
                        "phase_id": phase_id,
                        "position_world_m": [x, -6.76, 0.818],
                        "orientation_world_xyzw": [
                            0.0,
                            0.70710678,
                            0.0,
                            0.70710678,
                        ],
                        "gripper_state": "open",
                        "gate_ids": ["push_contact"],
                    }
                    for phase_id, x in (
                        ("push_contact", 2.9),
                        ("push_release", 3.1),
                        ("retreat", 3.2),
                    )
                ],
            ],
            "phase_count": 4,
            "plan_digest": "",
        },
        "plan_digest",
    )
    monkeypatch.setattr(
        plans,
        "materialize_native_task_construction_phase_plan",
        lambda _scene: __import__("copy").deepcopy(authored),
    )
    candidate = _candidate("curobo-a", 0, x=2.92)
    waypoints = []
    for stage_kind, target_x in (("entry", 2.75), ("approach", 2.79)):
        for waypoint_index in range(2):
            waypoints.append(
                {
                    "waypoint_id": f"{stage_kind}-{waypoint_index}",
                    "stage_id": f"curobo-{stage_kind}",
                    "stage_kind": stage_kind,
                    "robot_joint_positions_rad": {
                        f"panda_joint{joint}": 0.01 * (joint + waypoint_index)
                        for joint in range(1, 8)
                    },
                    "target_position_world_m": [target_x, -6.76, 0.818],
                    "target_orientation_world_xyzw": [
                        0.0,
                        0.70710678,
                        0.0,
                        0.70710678,
                    ],
                }
            )
    for stage_kind, phase_id, target_x in (
        ("contact", "push_contact", 2.9),
        ("release", "push_release", 3.1),
        ("retreat", "retreat", 3.2),
    ):
        waypoints.append(
            {
                "waypoint_id": f"{stage_kind}-0",
                "stage_id": f"curobo-{stage_kind}",
                "stage_kind": stage_kind,
                "source_native_phase_id": phase_id,
                "robot_joint_positions_rad": {
                    f"panda_joint{joint}": 0.02 * joint
                    for joint in range(1, 8)
                },
                "target_position_world_m": [target_x, -6.76, 0.818],
                "target_orientation_world_xyzw": [
                    0.0,
                    0.70710678,
                    0.0,
                    0.70710678,
                ],
            }
        )
    candidate["entry_trajectory_variant"] = _sealed(
        {
            "schema_version": "task_evaluation_native_entry_trajectory_variant.v1",
            "joins_authored_phase_id": "precontact",
            "waypoints": waypoints,
            "entry_trajectory_variant_digest": "",
        },
        "entry_trajectory_variant_digest",
    )
    candidate["candidate_digest"] = canonical_digest(
        candidate, digest_field="candidate_digest"
    )

    result = construction_phase_plan_for_candidate(
        scene_plan={
            "plan_digest": authored["scene_plan_digest"],
            "cadence": {"maximum_action_steps": 240},
        },
        candidate=candidate,
    )

    entry_rows = result["phases"][:2]
    assert [row["solver_stage_kind"] for row in entry_rows] == [
        "entry",
        "approach",
    ]
    assert [len(row["solver_joint_waypoint_sequence_rad"]) for row in entry_rows] == [
        2,
        2,
    ]
    assert all(row["solver_path_execution_required"] is True for row in entry_rows)
    assert result["execution_parameters"]["maximum_construction_total_steps"] == 228
    assert result["phases"][2] == authored["phases"][0]
    assert [
        row["solver_stage_kind"] for row in result["phases"][3:]
    ] == ["contact", "release", "retreat"]
    assert all(
        row["solver_path_execution_required"] is True
        for row in result["phases"][3:]
    )


def test_live_warm_executor_callsite_retries_exact_candidate_then_runs_controls(
    tmp_path, monkeypatch
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_warm_executor as warm

    run_id = "scene-839873-live-construction-feedback"
    first = _candidate("first", 0, x=2.92)
    second = _candidate("second", 1, x=3.04)
    base_request = _sealed(
        {
            "schema_version": "native_task_arena_packet_request.v1",
            "request_digest": "",
        },
        "request_digest",
    )
    def _write(path, value):
        path.write_text(
            __import__("json").dumps(value) + "\n", encoding="utf-8"
        )
    request_path = tmp_path / "base-request.json"
    _write(request_path, base_request)
    runtime_path = tmp_path / "runtime-source.json"
    _write(runtime_path, {"schema_version": "fixture.v1"})
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    session = {
        "schema_version": "native_task_arena_warm_session.v1",
        "instance_id": 49322931,
        "session_digest": "sha256:" + "a" * 64,
    }
    session_path = tmp_path / "warm-session.json"
    _write(session_path, session)
    config = {
        "schema_version": FEEDBACK_EXECUTOR_CONFIG_SCHEMA_VERSION,
        "base_packet_request_path": str(request_path),
        "evidence_root": str(evidence_root),
        "runtime_source_packet_receipt_path": str(runtime_path),
        "warm_session_path": str(session_path),
        "implementation_commit": "b" * 40,
        "authorization_reference": "bounded same-allocation feedback controller",
        "authorized_by": "task-evaluation-owner",
        "authorized_on": "2026-08-30T00:00:00Z",
        "max_hourly_rate_usd": 0.8,
        "hard_cap_usd": 2.0,
        "hard_ttl_seconds": 3600,
    }

    def packet(*, request, output_dir, **_kwargs):
        output_dir.mkdir(parents=True)
        _write(
            output_dir / "native_task_arena_scene_plan.v1.json",
            {
                "schema_version": "native_task_arena_scene_plan.v1",
                "plan_digest": "sha256:" + "c" * 64,
            },
        )
        _write(
            output_dir / "native_task_arena_packet_receipt.v1.json",
            {"receipt_digest": request["request_digest"]},
        )
        return {"receipt_digest": request["request_digest"]}

    def bundle(*, job_dir, **_kwargs):
        job_dir.mkdir(parents=True)
        _write(
            job_dir / "native_task_arena_provider_bundle_receipt.v1.json",
            {"receipt_digest": "sha256:" + "d" * 64},
        )
        return {
            "schema_version": "native_task_arena_provider_bundle.v1",
            "execution_mode": "construction_canary",
            "implementation_commit": "b" * 40,
            "bundle_sha256": "sha256:" + "e" * 64,
            "input_digest": "sha256:" + "f" * 64,
        }

    def controls_bundle(*, job_dir, **_kwargs):
        job_dir.mkdir(parents=True)
        _write(
            job_dir / "native_task_arena_provider_bundle_receipt.v1.json",
            {"receipt_digest": "sha256:" + "1" * 64},
        )
        return {
            "schema_version": "native_task_arena_provider_bundle.v1",
            "execution_mode": "controls",
            "implementation_commit": "b" * 40,
            "bundle_sha256": "sha256:" + "2" * 64,
            "input_digest": "sha256:" + "3" * 64,
        }

    monkeypatch.setattr(warm, "materialize_native_task_arena_packet", packet)
    monkeypatch.setattr(
        warm,
        "construction_phase_plan_for_candidate",
        lambda **_kwargs: {
            "schema_version": "native_rigid_construction_phase_plan.v1",
            "plan_digest": "sha256:" + "4" * 64,
        },
    )
    monkeypatch.setattr(warm, "build_native_task_arena_construction_bundle", bundle)
    monkeypatch.setattr(warm, "build_native_task_arena_controls_bundle", controls_bundle)
    monkeypatch.setattr(
        warm,
        "materialize_native_task_arena_warm_attempt_authority",
        lambda *, output_path, **_kwargs: _write(
            output_path, {"schema_version": "fixture-authority.v1"}
        ),
    )
    allocator_calls = []

    def allocator(argv):
        argv = list(argv or [])
        allocator_calls.append(argv)
        adapter = __import__("pathlib").Path(argv[argv.index("--adapter-output") + 1])
        adapter.parent.mkdir(parents=True, exist_ok=True)
        probe = argv[argv.index("--probe-kind") + 1]
        if probe == "native-task-arena-controls":
            control = _sealed(
                {
                    "schema_version": "native_task_arena_control_result.v1",
                    "status": "completed",
                    "controls_qualified": True,
                    "blockers": [],
                    "result_digest": "",
                },
                "result_digest",
            )
            path = adapter.parent / "native-control.json"
            _write(path, control)
            _write(
                adapter,
                {
                    "status": "completed",
                    "provider_instance_id": 49322931,
                    "provider_allocations_performed": 0,
                    "continuing_spend_from_this_run": False,
                    "native_control_result_path": str(path),
                },
            )
            return 0
        passed = sum(
            1
            for call in allocator_calls
            if "native-task-arena-construction" in call
        ) == 2
        native = _native(
            passed=passed, collision_force=0.0 if passed else 0.602
        )
        path = adapter.parent / "native-construction.json"
        _write(path, native)
        _write(
            adapter,
            {
                "status": "completed" if passed else "blocked",
                "provider_instance_id": 49322931,
                "provider_allocations_performed": 0,
                "runtime_seconds": 10.0,
                "incremental_cost_upper_bound_usd": 0.02,
                "native_construction_result_path": str(path),
            },
        )
        return 0 if passed else 2

    executor = WarmNativeConstructionFeedbackExecutor(
        config=config,
        output_root=tmp_path / "warm-rounds",
        allocator_main=allocator,
    )
    first_feedback = summarize_native_construction_feedback(
        _native(passed=False, collision_force=0.602)
    )
    invoker = _Invoker(
        selections=[
            (first["candidate_id"], None),
            (second["candidate_id"], first_feedback["feedback_digest"]),
        ]
    )

    def next_inventory(feedback, _history, round_index):
        assert feedback["feedback_digest"] == first_feedback["feedback_digest"]
        return _inventory(
            run_id,
            round_index,
            [second],
            feedback_digest=feedback["feedback_digest"],
        )

    receipt = run_native_construction_feedback_controller(
        invoker=invoker,
        authority=_authority(run_id),
        initial_inventory=_inventory(run_id, 0, [first]),
        produce_next_inventory=next_inventory,
        execute_candidate=executor,
        continue_to_controls=executor.continue_to_controls,
        clock=lambda: 1_000.0,
    )

    assert receipt["status"] == "controls_completed"
    assert receipt["controls_continuation"][
        "zero_action_and_scripted_positive_qualified"
    ] is True
    assert len(allocator_calls) == 3
    assert [
        call[call.index("--probe-kind") + 1] for call in allocator_calls
    ] == [
        "native-task-arena-construction",
        "native-task-arena-construction",
        "native-task-arena-controls",
    ]
    assert all("49322931" in call for call in allocator_calls)
    assert receipt["allocator_retry_cap"] == 0
    assert receipt["provider_allocations_performed"] == 0


def test_retained_production_callsite_requires_remote_curobo_by_default(
    tmp_path, monkeypatch
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_warm_executor as warm

    run_id = "scene-839873-required-remote-curobo"
    candidate = _candidate("remote-curobo-seed", 0, x=2.92)
    universe = _inventory(run_id, 0, [candidate])
    packet = tmp_path / "packet"
    packet.mkdir()
    request = _sealed(
        {
            "schema_version": "native_task_arena_packet_request.v1",
            "robot_base_pose_world": candidate["robot_base_pose_world"],
            "robot_joint_reset_positions_rad": candidate["reset_variant"][
                "robot_joint_reset_positions_rad"
            ],
            "cameras": candidate["camera_variant"]["cameras"],
            "native_construction_feedback": {
                "selected_placement_candidate_id": "geometry-selected",
                "candidate_universe": universe,
                "candidate_generator_authority": {
                    "generator": "remote_curobo_v2_motion_generation",
                    "package_version": "0.8.0",
                    "source_revision": warm.CUROBO_BACKEND_IDENTITY[
                        "source_revision"
                    ],
                    "required_on_retained_gpu": True,
                    "deterministic_cpu_prefilter_required": True,
                    "silent_fallback_permitted": False,
                },
                "allocator_retry_cap": 0,
                "native_gates_unchanged": True,
            },
            "request_digest": "",
        },
        "request_digest",
    )
    def _write(path, value):
        path.write_text(
            __import__("json").dumps(value) + "\n", encoding="utf-8"
        )
    _write(packet / "native_task_arena_packet_request.v1.json", request)
    _write(
        packet / "native_task_arena_scene_plan.v1.json",
        {"schema_version": "native_task_arena_scene_plan.v1"},
    )
    terminal_native = _native(passed=False, collision_force=0.602)
    terminal_native_path = tmp_path / "terminal-native.json"
    _write(terminal_native_path, terminal_native)
    terminal_allocator = {
        "schema_version": "native_task_arena_vast_run.v1",
        "status": "blocked",
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "warm_session": None,
        "warm_session_receipt_path": None,
        "native_control_result_path": str(terminal_native_path),
        "native_control_result_digest": terminal_native["result_digest"],
        "estimated_cost_usd": 0.08,
        "result_digest": "",
    }
    terminal_allocator["result_digest"] = canonical_digest(
        terminal_allocator, digest_field="result_digest"
    )
    adoption_path = tmp_path / "terminal-adoption.json"
    adoption = materialize_native_construction_terminal_feedback_adoption(
        allocator_result=terminal_allocator,
        native_result=terminal_native,
        packet_dir=packet,
        output_path=adoption_path,
    )
    native_path = tmp_path / "cold-native.json"
    bootstrap_native = {
        "schema_version": "native_task_arena_construction_result.v1",
        "status": "blocked",
        "construction_gate_qualified": False,
        "blockers": ["native_construction_feedback_bootstrap_ready"],
        "feedback_bootstrap_only": True,
        "baseline_physics_replayed": False,
        "terminal_feedback_adoption_digest": adoption["checkpoint_digest"],
        "result_digest": "",
    }
    bootstrap_native["result_digest"] = canonical_digest(
        bootstrap_native, digest_field="result_digest"
    )
    _write(native_path, bootstrap_native)
    warm_session = {
        "schema_version": "native_task_arena_warm_session.v1",
        "status": "ready",
        "provider": "vast",
        "instance_id": 49322931,
        "session_digest": "sha256:" + "a" * 64,
        "watchdog_deadline_epoch": 4_000_000_000.0,
        "continuing_spend": True,
        "remote_work_dir": "/workspace",
    }
    session_path = tmp_path / "warm-session.json"
    _write(session_path, warm_session)
    runtime_path = tmp_path / "runtime.json"
    _write(runtime_path, {"schema_version": "fixture.v1"})
    cold = {
        "native_control_result_path": str(native_path),
        "warm_session_receipt_path": str(session_path),
        "continuing_spend_from_this_run": True,
        "retry_cap": 0,
        "estimated_cost_usd": 0.1,
    }
    observed = {}
    fake_context = object()

    def materialize_context(**kwargs):
        observed["context"] = kwargs
        return fake_context, "/workspace/adp_arena_provider_bundle/provider_runtime"

    class Remote:
        def __init__(self, **kwargs):
            observed["remote"] = kwargs

        def generate(self, **kwargs):
            observed["remote_generate"] = kwargs
            return _inventory(
                run_id,
                kwargs["round_index"],
                [candidate],
                feedback_digest=kwargs["source_native_feedback"][
                    "feedback_digest"
                ],
            )

    class Executor:
        def __init__(self, **kwargs):
            observed["executor"] = kwargs

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("controller mock should not execute")

        def continue_to_controls(self, *_args, **_kwargs):
            raise AssertionError("controller mock should not continue")

    class Ledger:
        def record_adopted_baseline(self, *, baseline_record):
            observed["adopted_baseline"] = baseline_record
            return {"status": "recorded"}

    def controller(**kwargs):
        composite = kwargs["candidate_generator"]
        assert isinstance(composite, warm.CompositeCandidateGenerator)
        assert isinstance(composite._generators[0], Remote)
        assert composite._fallback_on_generator_unavailable is False
        return {"status": "controls_completed"}

    monkeypatch.setattr(warm, "materialize_remote_curobo_context", materialize_context)
    monkeypatch.setattr(warm, "RemoteCuroboCandidateGenerator", Remote)
    monkeypatch.setattr(warm, "WarmNativeConstructionFeedbackExecutor", Executor)
    monkeypatch.setattr(warm, "run_native_construction_feedback_controller", controller)

    result = run_retained_native_construction_feedback(
        cold_allocator_result=cold,
        packet_dir=packet,
        runtime_source_packet_receipt_path=runtime_path,
        implementation_commit="b" * 40,
        output_root=tmp_path / "feedback",
        authorization_reference="user always continue",
        authorized_by="task-evaluation-owner",
        authorized_on="2026-08-30T00:00:00Z",
        max_hourly_rate_usd=0.8,
        hard_cap_usd=1.2,
        hard_ttl_seconds=3600,
        invoker=object(),
        search_ledger=Ledger(),
        terminal_feedback_adoption_path=adoption_path,
    )

    assert result["status"] == "controls_completed"
    assert observed["context"]["warm_session"]["remote_work_dir"] == "/workspace"
    assert observed["remote"]["context"] is fake_context
    assert observed["remote_generate"]["round_index"] == 0
    assert observed["adopted_baseline"]["checkpoint_digest"] == adoption[
        "checkpoint_digest"
    ]
    assert observed["remote_generate"]["source_native_feedback"][
        "feedback_digest"
    ] == adoption["initial_native_feedback"]["feedback_digest"]
    assert observed["remote"]["remote_python_package_root"] == (
        "/workspace/adp_arena_provider_bundle/provider_runtime"
    )
