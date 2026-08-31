from __future__ import annotations

from types import SimpleNamespace

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_control_search_funnel import (
    OUTCOME_SCHEMA_VERSION,
    build_control_search_funnel_plan,
)
from blueprint_pipeline.task_evaluation_isaaclab_control_sweep import (
    build_isaaclab_control_sweep_schedule,
    compile_isaaclab_control_sweep_wave_commands,
    execute_isaaclab_control_sweep,
    validate_isaaclab_control_sweep_schedule,
)


def _candidate(index: int) -> dict:
    value = {
        "candidate_id": f"candidate-{index:03d}",
        "candidate_digest": "",
        "deterministic_rank": index,
    }
    value["candidate_digest"] = canonical_digest(
        value, digest_field="candidate_digest"
    )
    return value


def _inventory(count: int = 10) -> dict:
    value = {
        "schema_version": (
            "task_evaluation_native_construction_candidate_inventory.v1"
        ),
        "run_id": "scene-839873-vector-sweep",
        "round_index": 0,
        "source_native_feedback_digest": "sha256:" + "0" * 64,
        "model_authored_candidates": False,
        "candidates": [_candidate(index) for index in range(count)],
        "inventory_digest": "",
    }
    value["inventory_digest"] = canonical_digest(
        value, digest_field="inventory_digest"
    )
    return value


def _plan(inventory: dict) -> dict:
    return build_control_search_funnel_plan(
        run_id="scene-839873-vector-sweep",
        source_commit="a" * 40,
        packet_request_digest="sha256:" + "1" * 64,
        candidate_inventory=inventory,
        runtime_source_packet_digest="sha256:" + "2" * 64,
        scene_collision_digest="sha256:" + "3" * 64,
        task_object_asset_digest="sha256:" + "4" * 64,
        robot_configuration_digest="sha256:" + "5" * 64,
        task_scoring_digest="sha256:" + "6" * 64,
        requested_vector_env_count=8,
        maximum_vector_env_count=1_024,
        seeds_per_candidate=1,
        shortlist_size=8,
    )


def _outcome(assignment: dict) -> dict:
    value = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "candidate_id": assignment["candidate_id"],
        "candidate_digest": assignment["candidate_digest"],
        "seed_index": assignment["seed_index"],
        "resolved_seed": assignment["resolved_seed"],
        "wave_index": assignment["wave_index"],
        "environment_index": assignment["environment_index"],
        "reset_readback_passed": True,
        "forbidden_collision_peak_force_n": 0.0,
        "required_task_contact_coverage_fraction": 1.0,
        "push_path_tracking_error_m": 0.01,
        "destination_error_m": 0.02,
        "support_stability_error_m": 0.001,
        "task_displacement_m": 0.12,
        "physics_steps": 220,
        "measurement_authority": (
            "isaac_lab_simulator_state_and_contact_sensors"
        ),
        "learned_grader_used": False,
        "outcome_digest": "",
    }
    value["outcome_digest"] = canonical_digest(
        value, digest_field="outcome_digest"
    )
    return value


def test_schedule_assigns_complete_waves_deterministically() -> None:
    inventory = _inventory()
    plan = _plan(inventory)

    schedule = build_isaaclab_control_sweep_schedule(
        plan=plan,
        candidate_inventory=inventory,
        base_seed=839873104,
    )

    assert validate_isaaclab_control_sweep_schedule(schedule, plan=plan) == schedule
    assert schedule["vector_env_count"] == 8
    assert schedule["wave_count"] == 2
    assert schedule["waves"][0]["active_environment_count"] == 8
    assert schedule["waves"][1]["active_environment_count"] == 2
    assert schedule["waves"][1]["assignments"][0]["environment_index"] == 0


def test_executor_boots_once_and_resets_each_wave() -> None:
    inventory = _inventory()
    plan = _plan(inventory)
    schedule = build_isaaclab_control_sweep_schedule(
        plan=plan,
        candidate_inventory=inventory,
        base_seed=839873104,
    )
    builds = []
    waves = []

    def build(scene_plan, **kwargs):
        builds.append((scene_plan, kwargs))
        return SimpleNamespace(env="vector-env")

    def run_wave(*, built, wave, candidate_inventory, plan):
        assert built.env == "vector-env"
        assert candidate_inventory is inventory
        waves.append(wave["wave_index"])
        return {
            "outcomes": [_outcome(row) for row in wave["assignments"]],
            "peak_gpu_memory_bytes": 18_000_000_000 + wave["wave_index"],
        }

    result = execute_isaaclab_control_sweep(
        plan=plan,
        schedule=schedule,
        candidate_inventory=inventory,
        scene_plan={"plan_digest": "sha256:" + "9" * 64},
        bundle_root="/inputs",
        wave_runner=run_wave,
        environment_builder=build,
    )

    assert len(builds) == 1
    assert builds[0][1] == {
        "bundle_root": "/inputs",
        "num_envs": 8,
        "enable_cameras": False,
        "include_scene_appearance": False,
        "render_mode": None,
    }
    assert waves == [0, 1]
    assert result["status"] == "completed_development_only"
    assert result["peak_gpu_memory_bytes"] == 18_000_000_001


def test_wave_compiler_preserves_curobo_joint_targets_per_clone() -> None:
    arm_joint_names = [f"panda_joint{index}" for index in range(1, 8)]
    inventory = _inventory(10)
    for candidate_index, candidate in enumerate(inventory["candidates"]):
        reset = {
            name: 0.01 * (joint_index + candidate_index)
            for joint_index, name in enumerate(arm_joint_names)
        }
        candidate["reset_variant"] = {
            "robot_joint_reset_positions_rad": reset,
        }
        candidate["entry_trajectory_variant"] = {
            "waypoints": [
                {
                    "waypoint_id": "entry-0",
                    "stage_kind": "entry",
                    "robot_joint_positions_rad": reset,
                }
            ]
        }
        candidate["interaction_trajectory_variant"] = {
            "waypoints": [
                {
                    "waypoint_id": "contact-0",
                    "stage_kind": "contact",
                    "robot_joint_positions_rad": {
                        name: value + 0.02 for name, value in reset.items()
                    },
                },
                {
                    "waypoint_id": "release-0",
                    "stage_kind": "release",
                    "robot_joint_positions_rad": reset,
                },
            ]
        }
        candidate["candidate_digest"] = canonical_digest(
            candidate, digest_field="candidate_digest"
        )
    inventory["inventory_digest"] = canonical_digest(
        inventory, digest_field="inventory_digest"
    )
    plan = _plan(inventory)
    schedule = build_isaaclab_control_sweep_schedule(
        plan=plan,
        candidate_inventory=inventory,
        base_seed=839873104,
    )

    commands = compile_isaaclab_control_sweep_wave_commands(
        plan=plan,
        schedule=schedule,
        candidate_inventory=inventory,
        wave_index=0,
        arm_joint_names=arm_joint_names,
    )

    assert commands["active_environment_count"] == 8
    assert commands["maximum_waypoint_count"] == 3
    assert commands["assignments"][0]["waypoints"][0]["gripper_state"] == "open"
    assert commands["assignments"][0]["waypoints"][1]["gripper_state"] == "closed"
    assert commands["assignments"][0]["waypoints"][2]["gripper_state"] == "open"
    assert commands["assignments"][7]["environment_index"] == 7
