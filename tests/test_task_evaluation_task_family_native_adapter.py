"""Native-state execution pins targets, media, cleanup and independent scoring."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from blueprint_pipeline import task_evaluation_task_family_native_adapter as adapter
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_scoring import seal_task_family_contract

SHA = "sha256:" + "a" * 64
POSE = {"position_world_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]}


def _task(kind="button_press", *, criteria=None, reset=None):
    return seal_task_family_contract(task_family=kind,
        scope={"scene_id": "site1", "task_id": "task1", "task_context_digest": SHA},
        stages=[{"stage_id": "stage1", "task_family": kind,
                 "criteria": criteria or {"minimum_button_travel_m": .01, "maximum_contact_force_n": 20},
                 "minimum_hold_seconds": .1 if kind in {"stacking", "cloth_folding"} else 0}],
        reset={"stage1": reset or {"button_travel_m": 0, "button_activated": False, "contact_force_n": 0}},
        runtime_identity_digest=SHA)


def _runtime_contract(task=None, *, geometry=None, roles=None):
    task = task or _task()
    roles = roles or {"button": "button1"}
    return adapter.materialize_task_family_native_runtime_contract(scene_id="site1", task_id="task1",
        task_spec={"task_kind": task["task_family"], "task_success_contract": task},
        task_state_binding={"source": "native_simulator_state", "engine_id": "engine1",
            "runtime_identity_digest": SHA, "stage_bindings": {"stage1": {
                "native_readback_kind": task["task_family"], "asset_ids": list(roles.values()),
                "asset_roles": roles, "measurement_geometry": geometry or {
                    "joint_rest_position_m": 0, "press_direction": 1}}}},
        assets=[{"asset_id": key, "content_digest": SHA, "reset_state": {"reset_seed": 0}} for key in set(roles.values())],
        robot_base_pose_world=POSE, robot_joint_reset_positions_rad={"joint1": 0},
        robot_configuration={"embodiment_id": "humanoid1", "controller": "joint_control",
                             "action_schema_id": "joint_action1", "observation_schema_id": "rgb_proprio1"},
        cameras=[{"camera_id": "camera1", "pose_world": POSE, "intrinsics": [[10, 0, 8], [0, 10, 8], [0, 0, 1]],
                  "resolution": [16, 16]}], scenario_cell_id="cell1", scenario_instance_digest=SHA, seed=0)


class Runtime:
    runtime_identity_digest = SHA
    engine_id = "engine1"

    def __init__(self):
        self.native_runtime_contract = _runtime_contract()
        self.native_runtime_contract_digest = self.native_runtime_contract["contract_digest"]
        self.index = 0
        self.closed = False
        self.actions = []

    def reset(self, *, seed):
        assert seed == 0
        self.index = 0

    def native_state(self):
        return {"source": "native_simulator_state", "simulation_time_seconds": self.index,
                "native_provenance": adapter.native_runtime_provenance(self.native_runtime_contract),
                "safety_violation": False, "asset_states": {"button1": {
                    "joint_position_m": .02 * self.index, "native_button_sensor_active": self.index > 0,
                    "contact_forces_n": [[0, 0, 0]],
                    # These mutable targets must never override sealed geometry.
                    "joint_rest_position_m": 999, "press_direction": -1}}}

    def policy_observation(self):
        return {"rgb": np.full((16, 16, 3), self.index, dtype=np.uint8), "proprioception": [self.index]}

    def step(self, action, *, duration_seconds):
        assert duration_seconds == 1
        self.actions.append(action)
        self.index += 1

    def close(self):
        self.closed = True


class Policy:
    def __init__(self):
        self.observations = []

    def act(self, observation):
        self.observations.append(observation)
        observation["rgb"][:] = 255
        return [1]


def _run(tmp_path, runtime=None, policy=None):
    runtime = runtime or Runtime()
    return adapter.execute_task_family_episode(runtime=runtime, policy=policy or Policy(), contract=_task(),
        maximum_steps=2, control_frequency_hz=1, output_dir=tmp_path, episode_id="episode1")


def test_materialize_command_consumes_input_json_and_writes_valid_contract(tmp_path, capsys):
    expected = _runtime_contract()
    fields = ("scene_id", "task_id", "task_spec", "task_state_binding", "assets", "robot_base_pose_world",
        "robot_joint_reset_positions_rad", "robot_configuration", "cameras", "scenario_cell_id",
        "scenario_instance_digest", "seed", "scenario_context_kind")
    inputs = tmp_path / "native_inputs.json"
    inputs.write_text(json.dumps({key: expected[key] for key in fields}))
    output = tmp_path / "native_contract.json"
    assert adapter.main(["materialize", "--inputs", str(inputs), "--output", str(output)]) == 0
    actual = adapter.validate_task_family_native_runtime_contract(json.loads(output.read_text()))
    assert actual == expected
    assert json.loads(capsys.readouterr().out)["contract_digest"] == actual["contract_digest"]


def test_materialize_command_rejects_invalid_geometry_without_writing_contract(tmp_path):
    expected = _runtime_contract()
    fields = ("scene_id", "task_id", "task_spec", "task_state_binding", "assets", "robot_base_pose_world",
        "robot_joint_reset_positions_rad", "robot_configuration", "cameras", "scenario_cell_id",
        "scenario_instance_digest", "seed", "scenario_context_kind")
    value = {key: expected[key] for key in fields}
    value["task_state_binding"]["stage_bindings"]["stage1"]["measurement_geometry"] = {}
    inputs = tmp_path / "native_inputs.json"
    inputs.write_text(json.dumps(value))
    output = tmp_path / "native_contract.json"
    with pytest.raises(SystemExit) as refused:
        adapter.main(["materialize", "--inputs", str(inputs), "--output", str(output)])
    assert refused.value.code == 2 and not output.exists()


def test_episode_scores_native_state_retains_exact_media_and_closes(tmp_path):
    runtime, policy = Runtime(), Policy()
    receipt = _run(tmp_path, runtime, policy)
    assert receipt["status"] == "completed" and receipt["task_succeeded"]
    assert receipt["runtime_closed"] and runtime.closed and len(runtime.actions) == 2
    assert receipt["scoring"]["scoring_source"] == "deterministic_native_state"
    provenance = adapter.native_runtime_provenance(runtime.native_runtime_contract)
    assert receipt["native_provenance"] == receipt["telemetry"]["native_provenance"] == provenance
    assert all(sample["native_provenance"] == provenance for sample in receipt["telemetry"]["samples"])
    assert receipt["maximum_steps"] == 2 and receipt["control_frequency_hz"] == 1
    assert all(set(row) == {"rgb", "proprioception"} for row in policy.observations)
    manifest = json.loads((tmp_path / "media/episode1/frame_manifest.json").read_text())
    assert manifest["identity"]["native_provenance"] == provenance
    assert manifest["policy_input_frames"][0]["proprioception"] == [0]
    from PIL import Image
    retained = np.asarray(Image.open(tmp_path / manifest["policy_input_frames"][0]["relative_path"]))
    assert np.all(retained == 0)  # A policy cannot rewrite the retained input.
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


@pytest.mark.parametrize("corruption", ["identity", "clock", "reset", "privileged_observation"])
def test_invalid_binding_reset_clock_or_privileged_input_never_succeeds(tmp_path, corruption):
    runtime = Runtime()
    if corruption == "identity":
        runtime.runtime_identity_digest = "sha256:" + "b" * 64
    elif corruption == "clock":
        native = runtime.native_state
        runtime.native_state = lambda: {**native(), "simulation_time_seconds": 42}
    elif corruption == "reset":
        native = runtime.native_state
        def wrong_reset():
            value = native()
            value["asset_states"]["button1"]["joint_position_m"] = 1
            return value
        runtime.native_state = wrong_reset
    else:
        observation = runtime.policy_observation
        runtime.policy_observation = lambda: {**observation(), "privileged_success": True}
    receipt = _run(tmp_path, runtime)
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"] and runtime.closed
    assert runtime.actions == []
    assert receipt["visual_evidence"]["media_gap"]["type"] == "before_first_observation"


def test_video_failure_keeps_frames_and_failed_manifest_and_invalidates_episode(tmp_path, monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("video_encoder_unavailable")
    monkeypatch.setattr(adapter, "finalize_visual_evidence", unavailable)
    receipt = _run(tmp_path)
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"]
    manifest = json.loads((tmp_path / "media/episode1/failed_frame_manifest.json").read_text())
    assert len(manifest["candidate_exact_policy_input_frames"]) == 2
    assert manifest["terminal_observation_invented"] is False
    assert receipt["scoring"]["task_succeeded"]  # Native task success alone cannot hide missing media.


def test_materializer_rejects_undeclared_geometry_assets_and_bad_calibration():
    original = _runtime_contract()
    for field, replacement in (("robot_base_pose_world", {"position_world_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 0]}),
                               ("robot_configuration", {"embodiment_id": "humanoid1"}),
                               ("cameras", [{"camera_id": "camera1"}])):
        value = copy.deepcopy(original)
        value[field] = replacement
        value["contract_digest"] = canonical_digest(value, digest_field="contract_digest")
        with pytest.raises(ValueError):
            adapter.validate_task_family_native_runtime_contract(value)
    value = copy.deepcopy(original)
    value["task_state_binding"]["stage_bindings"]["stage1"]["measurement_geometry"] = {}
    value["contract_digest"] = canonical_digest(value, digest_field="contract_digest")
    with pytest.raises(ValueError, match="button_geometry_invalid"):
        adapter.validate_task_family_native_runtime_contract(value)


def test_pouring_counts_particles_in_moving_bound_containers_not_claimed_volumes():
    task = _task("pouring", criteria={"minimum_transferred_volume_ml": 10, "maximum_spilled_volume_ml": 2,
        "maximum_volume_balance_error_ml": .1, "maximum_final_in_flight_volume_ml": .1}, reset={
            "source_volume_ml": 20, "receiver_volume_ml": 0, "spilled_volume_ml": 0, "in_flight_volume_ml": 0})
    runtime = _runtime_contract(task, geometry={"source_local_bounds_m": [[0, 0, 0], [1, 1, 1]],
        "receiver_local_bounds_m": [[0, 0, 0], [1, 1, 1]],
        "transfer_workspace_bounds_world_m": [[1, 0, 0], [2, 1, 1]]},
        roles={"source": "cup", "receiver": "bowl", "material": "water"})
    state = {"source": "native_simulator_state", "safety_violation": False,
        "native_provenance": adapter.native_runtime_provenance(runtime), "asset_states": {
        "cup": {"pose_world": POSE}, "bowl": {"pose_world": {**POSE, "position_world_m": [2, 0, 0]}},
        "water": {"particle_positions_m": [[.5, .5, .5], [2.5, .5, .5], [4, 0, 0]],
                  "particle_volumes_ml": [9, 10, 1], "receiver_volume_ml": 999}}}
    result = adapter.derive_task_family_measurements(task, state, native_runtime_contract=runtime)
    assert result["stage1"] == {"source_volume_ml": 9, "receiver_volume_ml": 10,
                                "spilled_volume_ml": 1, "in_flight_volume_ml": 0}


def test_insertion_depth_uses_bound_socket_frame_and_asset_pose():
    task = _task("insertion", criteria={"minimum_insertion_depth_m": .1, "maximum_lateral_error_m": .01,
        "maximum_alignment_error_rad": .1, "maximum_contact_force_n": 10},
        reset={"insertion_depth_m": 0, "lateral_error_m": 0, "alignment_error_rad": 0, "contact_force_n": 0})
    runtime = _runtime_contract(task, roles={"peg": "peg1", "socket": "socket1"}, geometry={
        "peg_tip_local_m": [0, 0, 0], "peg_axis_local": [0, 0, 1],
        "socket_entry_local_m": [0, 0, 0], "socket_axis_local": [0, 0, 1]})
    state = {"source": "native_simulator_state", "safety_violation": False,
        "native_provenance": adapter.native_runtime_provenance(runtime), "asset_states": {
        "peg1": {"pose_world": {**POSE, "position_world_m": [0, 0, .2]}, "contact_forces_n": [[0, 3, 4]],
                 "socket_entry_position_m": [0, 0, 999]}, "socket1": {"pose_world": POSE}}}
    row = adapter.derive_task_family_measurements(task, state, native_runtime_contract=runtime)["stage1"]
    assert row == {"insertion_depth_m": .2, "lateral_error_m": 0, "alignment_error_rad": 0, "contact_force_n": 5}


def test_wiping_uses_sealed_cell_areas_and_native_residue():
    task = _task("wiping", criteria={"minimum_cleaned_fraction": .8, "maximum_remaining_dirty_area_m2": .1},
                 reset={"dirty_area_m2": 1})
    runtime = _runtime_contract(task, roles={"surface": "surface1"}, geometry={"cell_areas_m2": [.5, .5]})
    state = {"source": "native_simulator_state", "safety_violation": False,
        "native_provenance": adapter.native_runtime_provenance(runtime), "asset_states": {
        "surface1": {"cell_dirt_fractions": [.1, .2], "cell_areas_m2": [100, 100], "dirty_area_m2": 0}}}
    row = adapter.derive_task_family_measurements(task, state, native_runtime_contract=runtime)["stage1"]
    assert row["dirty_area_m2"] == pytest.approx(.15)


@pytest.mark.parametrize("tilted", [False, True])
def test_stacking_projects_center_into_bound_support_frame_and_retains_robot_contact(tilted):
    task = _task("stacking", criteria={"minimum_support_margin_m": .05,
        "maximum_linear_speed_m_s": .1, "maximum_angular_speed_rad_s": .1},
        reset={"support_contact": False, "com_support_margin_m": -.1,
               "linear_speed_m_s": 0, "angular_speed_rad_s": 0, "robot_contact_active": False})
    runtime = _runtime_contract(task, roles={"object": "object1", "support": "support1"}, geometry={
        "support_polygon_local_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    support_pose = {"position_world_m": [5, 0, 0], "orientation_xyzw": [2**-.5, 0, 0, 2**-.5] if tilted else [0, 0, 0, 1]}
    state = {"source": "native_simulator_state", "safety_violation": False,
        "native_provenance": adapter.native_runtime_provenance(runtime), "asset_states": {
        "support1": {"pose_world": support_pose},
        "object1": {"center_of_mass_m": [5.5, -.1, .5] if tilted else [5.5, .5, .1], "support_normal_forces_n": [1],
                    "linear_velocity_m_s": [0, 0, 0], "angular_velocity_rad_s": [0, 0, 0],
                    "robot_contact_active": True,
                    "support_polygon_xy_m": [[9, 9], [10, 9], [10, 10], [9, 10]]}}}
    row = adapter.derive_task_family_measurements(task, state, native_runtime_contract=runtime)["stage1"]
    assert row["com_support_margin_m"] == pytest.approx(.5) and row["support_contact"]
    assert row["robot_contact_active"] is True


@pytest.mark.parametrize("y_up", [False, True])
def test_cloth_computes_overlap_and_keypoints_from_bound_mesh_indices(y_up):
    task = _task("cloth_folding", criteria={"maximum_fold_keypoint_error_m": .1,
        "minimum_fold_overlap_fraction": .9, "maximum_linear_speed_m_s": .1, "require_release": True},
        reset={"fold_keypoint_error_m": 1, "fold_overlap_fraction": 0,
               "linear_speed_m_s": 0, "robot_contact_active": False})
    runtime = _runtime_contract(task, roles={"cloth": "cloth1"}, geometry={
        "fold_plane_pose_world": {**POSE, "orientation_xyzw": [2**-.5, 0, 0, 2**-.5]} if y_up else POSE,
        "target_keypoint_positions_m": [[0, 0, 0]], "keypoint_vertex_indices": [0],
        "first_fold_vertex_indices": [0, 1, 2, 3], "second_fold_vertex_indices": [4, 5, 6, 7]})
    vertices = ([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]] if y_up else
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]) * 2
    state = {"source": "native_simulator_state", "safety_violation": False,
        "native_provenance": adapter.native_runtime_provenance(runtime), "asset_states": {
        "cloth1": {"vertex_positions_m": vertices, "vertex_velocities_m_s": [[0, 0, 0]] * 8,
                   "robot_contact_active": False, "target_keypoint_positions_m": [[999, 0, 0]]}}}
    row = adapter.derive_task_family_measurements(task, state, native_runtime_contract=runtime)["stage1"]
    assert row == {"fold_keypoint_error_m": 0, "fold_overlap_fraction": 1,
                   "linear_speed_m_s": 0, "robot_contact_active": False}


def test_cloth_requires_explicit_immutable_measurement_plane():
    task = _task("cloth_folding", criteria={"maximum_fold_keypoint_error_m": .1,
        "minimum_fold_overlap_fraction": .9, "maximum_linear_speed_m_s": .1, "require_release": True},
        reset={"fold_keypoint_error_m": 1, "fold_overlap_fraction": 0,
               "linear_speed_m_s": 0, "robot_contact_active": False})
    with pytest.raises(ValueError, match="pose_invalid"):
        _runtime_contract(task, roles={"cloth": "cloth1"}, geometry={
            "target_keypoint_positions_m": [[0, 0, 0]], "keypoint_vertex_indices": [0],
            "first_fold_vertex_indices": [0, 1, 2], "second_fold_vertex_indices": [3, 4, 5]})


@pytest.mark.parametrize("forces", [[], np.empty((0, 3)), np.zeros((2, 3))])
def test_native_numpy_or_empty_contact_readback_is_zero_force(forces):
    runtime = Runtime()
    state = runtime.native_state()
    state["asset_states"]["button1"]["contact_forces_n"] = forces
    result = adapter.derive_task_family_measurements(_task(), state,
        native_runtime_contract=runtime.native_runtime_contract)
    assert result["stage1"]["contact_force_n"] == 0


def test_runtime_contract_must_match_current_applied_scene_before_reset(tmp_path):
    runtime = Runtime()
    runtime.native_runtime_contract_digest = "sha256:" + "b" * 64
    receipt = _run(tmp_path, runtime)
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"]
    assert receipt["failure"]["reason"].endswith("applied_runtime_contract_mismatch")
    assert runtime.actions == [] and runtime.closed


def test_reset_cannot_replace_the_applied_scene_before_policy_execution(tmp_path):
    runtime = Runtime()
    reset = runtime.reset
    def different_scene(*, seed):
        reset(seed=seed)
        runtime.native_runtime_contract_digest = "sha256:" + "b" * 64
    runtime.reset = different_scene
    receipt = _run(tmp_path, runtime)
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"]
    assert receipt["failure"]["reason"].endswith("applied_runtime_contract_mismatch")
    assert runtime.actions == [] and runtime.closed


@pytest.mark.parametrize("field", ["native_runtime_contract_digest", "scenario_cell_id", "scenario_instance_digest", "seed"])
@pytest.mark.parametrize("drift_step", [0, 1])
def test_every_native_state_is_bound_to_exact_scene_and_scenario(tmp_path, field, drift_step):
    runtime = Runtime()
    native = runtime.native_state
    def drifted_state():
        state = native()
        if runtime.index >= drift_step:
            state["native_provenance"][field] = 1 if field == "seed" else "wrong-scene-or-scenario"
        return state
    runtime.native_state = drifted_state
    receipt = _run(tmp_path, runtime)
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"]
    assert receipt["failure"]["reason"].endswith("readback_scene_binding_mismatch")
    assert len(runtime.actions) == drift_step and runtime.closed
    if drift_step:
        assert receipt["visual_evidence"]["retained_policy_input_frame_count"] == 1


def test_final_readback_cannot_change_current_applied_scene(tmp_path):
    runtime = Runtime()
    native = runtime.native_state
    def changed_during_readback():
        state = native()
        if runtime.index == 2:
            runtime.native_runtime_contract_digest = "sha256:" + "b" * 64
        return state
    runtime.native_state = changed_during_readback
    receipt = _run(tmp_path, runtime)
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"] and runtime.closed
    assert receipt["failure"]["reason"].endswith("applied_runtime_contract_mismatch")


def test_boolean_seed_cannot_match_numeric_scenario_seed(tmp_path):
    runtime = Runtime()
    native = runtime.native_state
    def wrong_seed_type():
        state = native()
        state["native_provenance"]["seed"] = False
        return state
    runtime.native_state = wrong_seed_type
    receipt = _run(tmp_path, runtime)
    assert receipt["status"] == "invalid" and runtime.actions == [] and runtime.closed


@pytest.mark.parametrize("unfinished_stream", [False, True])
def test_pouring_in_flight_stream_is_distinct_from_escaped_spill(tmp_path, unfinished_stream):
    task = _task("pouring", criteria={"minimum_transferred_volume_ml": 10, "maximum_spilled_volume_ml": 0,
        "maximum_volume_balance_error_ml": .01, "maximum_final_in_flight_volume_ml": 0}, reset={
            "source_volume_ml": 20, "receiver_volume_ml": 0, "spilled_volume_ml": 0, "in_flight_volume_ml": 0})
    native_contract = _runtime_contract(task, geometry={"source_local_bounds_m": [[0, 0, 0], [1, 1, 1]],
        "receiver_local_bounds_m": [[0, 0, 0], [1, 1, 1]],
        "transfer_workspace_bounds_world_m": [[1, 0, 0], [2, 1, 1]]},
        roles={"source": "cup", "receiver": "bowl", "material": "water"})
    class PouringRuntime(Runtime):
        def native_state(self):
            if self.index == 0:
                particles = [[.5, .5, .5], [.5, .5, .5]]
            elif self.index == 1 or unfinished_stream:
                particles = [[2.5, .5, .5], [1.5, .5, .5]]
            else:
                particles = [[2.5, .5, .5], [2.5, .5, .5]]
            return {"source": "native_simulator_state", "simulation_time_seconds": self.index,
                    "safety_violation": False, "native_provenance": adapter.native_runtime_provenance(self.native_runtime_contract),
                    "asset_states": {"cup": {"pose_world": POSE},
                        "bowl": {"pose_world": {**POSE, "position_world_m": [2, 0, 0]}},
                        "water": {"particle_positions_m": particles, "particle_volumes_ml": [10, 10]}}}
    runtime = PouringRuntime()
    runtime.native_runtime_contract = native_contract
    runtime.native_runtime_contract_digest = native_contract["contract_digest"]
    receipt = adapter.execute_task_family_episode(runtime=runtime, policy=Policy(), contract=task,
        maximum_steps=2, control_frequency_hz=1, output_dir=tmp_path, episode_id="pour1")
    assert receipt["status"] == "completed"
    assert receipt["task_succeeded"] is not unfinished_stream
    intermediate = receipt["telemetry"]["samples"][1]["measurements"]["stage1"]
    assert intermediate["in_flight_volume_ml"] == 10 and intermediate["spilled_volume_ml"] == 0
    assert runtime.closed


def _folding_runtime():
    task = _task("cloth_folding", criteria={"maximum_fold_keypoint_error_m": .1,
        "minimum_fold_overlap_fraction": .9, "maximum_linear_speed_m_s": .1, "require_release": True},
        reset={"fold_keypoint_error_m": 2, "fold_overlap_fraction": 0,
               "linear_speed_m_s": 0, "robot_contact_active": False})
    native_contract = _runtime_contract(task, roles={"cloth": "cloth1"}, geometry={
        "fold_plane_pose_world": POSE, "target_keypoint_positions_m": [[0, 0, 0]],
        "keypoint_vertex_indices": [4], "first_fold_vertex_indices": [0, 1, 2, 3],
        "second_fold_vertex_indices": [4, 5, 6, 7]})
    class FoldingRuntime(Runtime):
        def native_state(self):
            first = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
            if self.index == 0:
                second = [[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]]
            elif self.index == 1:
                # The moving half is perpendicular to the measurement plane.
                second = [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]]
            elif self.index == 2:
                # Cloth may temporarily project to a concave boundary.
                second = [[0, 0, 0], [1, 0, 0], [.5, .25, 0], [0, 1, 0]]
            else:
                second = first
            return {"source": "native_simulator_state", "simulation_time_seconds": self.index,
                    "safety_violation": False, "native_provenance": adapter.native_runtime_provenance(self.native_runtime_contract),
                    "asset_states": {"cloth1": {"vertex_positions_m": first + second,
                        "vertex_velocities_m_s": [[0, 0, 0]] * 8,
                        "robot_contact_active": self.index in {1, 2}}}}
    runtime = FoldingRuntime()
    runtime.native_runtime_contract = native_contract
    runtime.native_runtime_contract_digest = native_contract["contract_digest"]
    return task, runtime


def test_complete_fold_can_pass_through_edge_on_and_concave_projections(tmp_path):
    task, runtime = _folding_runtime()
    receipt = adapter.execute_task_family_episode(runtime=runtime, policy=Policy(), contract=task,
        maximum_steps=4, control_frequency_hz=1, output_dir=tmp_path, episode_id="fold1")
    assert receipt["status"] == "completed" and receipt["task_succeeded"] and runtime.closed
    samples = receipt["telemetry"]["samples"]
    overlaps = [sample["measurements"]["stage1"]["fold_overlap_fraction"] for sample in samples]
    assert overlaps == [0, 0, 0, 1, 1]
    assert len(runtime.actions) == 4


@pytest.mark.parametrize("malformed", ["nonfinite", "wrong_shape"])
def test_malformed_cloth_readback_still_invalidates_fold(tmp_path, malformed):
    task, runtime = _folding_runtime()
    native = runtime.native_state
    def malformed_readback():
        state = native()
        if runtime.index == 1:
            vertices = state["asset_states"]["cloth1"]["vertex_positions_m"]
            if malformed == "nonfinite":
                vertices[7][2] = float("nan")
            else:
                state["asset_states"]["cloth1"]["vertex_positions_m"] = [[0, 0]] * 8
        return state
    runtime.native_state = malformed_readback
    receipt = adapter.execute_task_family_episode(runtime=runtime, policy=Policy(), contract=task,
        maximum_steps=4, control_frequency_hz=1, output_dir=tmp_path, episode_id="fold1")
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"] and runtime.closed
    assert len(runtime.actions) == 1


@pytest.mark.parametrize("polygon", [[[0, 0], [0, 0], [1, 0]],
                                    [[0, 0], [1, 0], [.5, .25], [0, 1]]])
def test_static_support_polygons_remain_strict(polygon):
    task = _task("stacking", criteria={"minimum_support_margin_m": .05,
        "maximum_linear_speed_m_s": .1, "maximum_angular_speed_rad_s": .1},
        reset={"support_contact": False, "com_support_margin_m": -.1,
               "linear_speed_m_s": 0, "angular_speed_rad_s": 0, "robot_contact_active": False})
    with pytest.raises(ValueError, match="polygon_(degenerate|not_convex)"):
        _runtime_contract(task, roles={"object": "object1", "support": "support1"}, geometry={
            "support_polygon_local_xy_m": polygon})
