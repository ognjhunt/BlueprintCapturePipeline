"""Independent numeric and native-fake gates; these are not Isaac execution receipts."""

from __future__ import annotations

import copy
import hashlib
import io
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import adp_task_scoring, episode_visual_evidence, native_task_episode_environment
from blueprint_pipeline.team_arm_kinematics import (
    LiveArmKinematics,
    POSE_CHUNK_SCHEMA,
    pose_servo_step,
    quaternion_xyzw,
    relative_tcp_pose,
    rotation_xyzw,
    validate_pose_chunk,
)
from blueprint_pipeline.team_embodiment import (
    TeamEmbodimentError,
    apply_team_embodiment,
    read_model_structure,
    team_embodiment_binding,
    validate_embodiment_manifest,
    validate_model_dependency_closure,
    verify_model_file,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.team_embodiment_runtime import run_team_episode
from tests.test_team_embodiment import _Camera, _Scoring, _team_loop, bimanual_inputs, station_plan

IDENTITY = [0., 0., 0., 1.]
OFFSET = {"position_m": [0., 0., 0.], "orientation_xyzw": IDENTITY}


def test_tcp_in_its_own_rotated_base_and_rotated_offset():
    half = math.sqrt(0.5)
    base = [2., 3., 0., 0., 0., half, half]
    tool = [2., 4., 0., 0., 0., half, half]
    offset = {"position_m": [0.2, 0., 0.], "orientation_xyzw": [half, 0., 0., half]}
    observed = relative_tcp_pose(base_pose=base, tool_pose=tool, tool_from_tcp=offset)
    assert observed["position_m"] == pytest.approx([1.2, 0., 0.])
    assert rotation_xyzw(observed["orientation_xyzw"]) == pytest.approx(rotation_xyzw(offset["orientation_xyzw"]))
    for q in ([1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], IDENTITY):
        assert rotation_xyzw(quaternion_xyzw(rotation_xyzw(q))) == pytest.approx(rotation_xyzw(q))


def test_servo_uses_real_com_reference_point_and_does_not_claim_achievement():
    # A z revolute joint at the origin: COM at x=.5 has dy/dq=.5;
    # TCP at x=1 has dy/dq=1. The six-row Jacobian also constrains rotation.
    jacobian_com = np.array([[0.], [.5], [0.], [0.], [0.], [1.]])
    target, receipt = pose_servo_step(
        joint_positions=[0.], jacobian_world_com=jacobian_com,
        body_pose_world=[0., 0., 0., *IDENTITY], body_com_position_world=[.5, 0., 0.],
        base_pose_world=[0., 0., 0., *IDENTITY],
        tool_from_tcp={"position_m": [1., 0., 0.], "orientation_xyzw": IDENTITY},
        target_position_m=[1., .1, 0.], target_orientation_xyzw=IDENTITY,
        position_limits=[[-3., 3.]], velocity_limits=[10.], step_seconds=.25)
    expected = .1 / (2 + .05**2)
    assert target == pytest.approx([expected])
    assert receipt["position_error_before_m"] == pytest.approx(.1)
    assert receipt["target_achievement"] == "not_measured_by_this_command"
    assert receipt["jacobian_rank"] == 1


def test_servo_rotates_arm_base_targets_and_records_every_limiting_scale():
    half = math.sqrt(.5)
    base = [2., 3., 0., 0., 0., half, half]
    command, receipt = pose_servo_step(
        joint_positions=[0.]*6, jacobian_world_com=np.eye(6), body_pose_world=base,
        body_com_position_world=base[:3], base_pose_world=base, tool_from_tcp=OFFSET,
        target_position_m=[.2, 0., 0.], target_orientation_xyzw=IDENTITY,
        position_limits=[[-.1, .1]]*6, velocity_limits=[.05]*6, step_seconds=.1)
    assert command[0] == pytest.approx(0., abs=1e-12)
    assert command[1] == pytest.approx(.005)
    assert receipt["velocity_limit_scale"] < 1
    assert receipt["position_limit_scale"] == 1
    assert max(abs(value) for value in command) <= .005 + 1e-12


def pose_chunk(names=("left", "right"), rows=1):
    return {"schema_version": POSE_CHUNK_SCHEMA, "frame": "arm_base", "arm_names": list(names),
            "positions_m": [[[.1, 0., 0.], [-.1, 0., 0.]] for _ in range(rows)],
            "orientations_xyzw": [[IDENTITY, IDENTITY] for _ in range(rows)],
            "gripper_closures": [[1., 0.] for _ in range(rows)]}


@pytest.mark.parametrize("fault", ["names", "units", "quaternion", "nonfinite", "closure", "rows"])
def test_pose_chunk_refuses_convention_identity_and_shape_faults(fault):
    value = pose_chunk()
    if fault == "names":
        value["arm_names"].reverse()
    elif fault == "units":
        value["units"] = "centimetres"
    elif fault == "quaternion":
        value["orientations_xyzw"][0][0] = [0., 0., 0., 2.]
    elif fault == "nonfinite":
        value["positions_m"][0][0][0] = math.nan
    elif fault == "closure":
        value["gripper_closures"][0][0] = 2.
    else:
        value["positions_m"] = []
    with pytest.raises(ValueError):
        validate_pose_chunk(value, arm_names=["left", "right"], maximum_rows=4)


def named_inputs(tmp_path):
    model, value = bimanual_inputs(tmp_path, arm_dof=6)
    value["control_hz"] = 60
    value["arms"] = [{"name": side, "joint_names": [f"{side}_joint_{i}" for i in range(6)],
                      "base_link": "base", "tool_link": f"{side}_tool", "gripper_name": side,
                      "tool_from_tcp": copy.deepcopy(OFFSET)} for side in ("left", "right")]
    return model, value


def test_named_arm_manifest_binds_each_real_link_and_keeps_joint_target_contract(tmp_path):
    model, value = named_inputs(tmp_path)
    declared = validate_embodiment_manifest(value)
    binding = team_embodiment_binding(declared, model_record=verify_model_file(declared, model),
                                     structure=read_model_structure(model), scene_plan=station_plan())
    plan = apply_team_embodiment(station_plan(), binding)
    assert plan["robot"]["arms"] == value["arms"]
    assert plan["cadence"]["task_sample_stride"] == 4
    assert plan["robot"]["joint_names"] == [j["name"] for j in value["joints"]]
    changed = copy.deepcopy(value)
    changed["arms"][0]["base_link"] = "absent"
    manifest = validate_embodiment_manifest(changed)
    with pytest.raises(TeamEmbodimentError, match="missing_or_ambiguous:absent"):
        team_embodiment_binding(manifest, model_record=verify_model_file(manifest, model),
                                structure=read_model_structure(model), scene_plan=station_plan())


@pytest.mark.parametrize("fault", ["joint_overlap", "tool_swap", "offset_norm", "partial", "free", "v1"])
def test_named_arm_manifest_rejects_ambiguous_or_incomplete_bindings(tmp_path, fault):
    _, value = named_inputs(tmp_path)
    if fault == "joint_overlap":
        value["arms"][1]["joint_names"] = list(value["arms"][0]["joint_names"])
    elif fault == "tool_swap":
        value["arms"][0]["gripper_name"] = "right"
    elif fault == "offset_norm":
        value["arms"][0]["tool_from_tcp"]["orientation_xyzw"] = [0., 0., 0., 3.]
    elif fault == "partial":
        value["arms"].pop()
    elif fault == "free":
        value["base"] = {"mode": "free"}
    else:
        value["schema_version"] = "blueprint.robot_embodiment.v1"
    with pytest.raises(TeamEmbodimentError):
        validate_embodiment_manifest(value)


class CartesianRobot:
    """Known independent translational/rotational axes, deliberately shuffled native joints."""

    def __init__(self, row):
        self.row = row
        self.joint_names = list(reversed([*row["joint_names"], *(j["name"] for j in row["gripper_joints"])]))
        self.body_names = ["base", "left_tool", "right_tool"]
        self.is_fixed_base = True
        self.data = self
        self.root_view = self
        self.q = np.array([row["joint_reset_positions_rad"][n] for n in self.joint_names])

    @property
    def joint_pos(self):
        return np.array([self.q])

    @property
    def joint_vel(self):
        return np.zeros((1, len(self.q)))

    @property
    def root_pose_w(self):
        return np.array([[0., 0., 0., *IDENTITY]])

    @property
    def root_vel_w(self):
        return np.zeros((1, 6))

    @property
    def body_pose_w(self):
        poses = [[0., 0., 0., *IDENTITY]]
        for arm in self.row["arms"]:
            values = [self.q[self.joint_names.index(n)] for n in arm["joint_names"]]
            # The integration target uses translation only; all rotations remain zero.
            poses.append([*values[:3], *IDENTITY])
        return np.array([poses])

    @property
    def body_com_pose_w(self):
        return self.body_pose_w

    def get_jacobians(self):
        result = np.zeros((1, 2, 6, len(self.joint_names)))
        for index, arm in enumerate(self.row["arms"]):
            for axis, joint in enumerate(arm["joint_names"]):
                result[0, index, axis, self.joint_names.index(joint)] = 1.
        return result


@pytest.mark.parametrize("real_dream_client", [False, True])
def test_native_fake_executes_pose_rows_at_twenty_hz_with_live_fk_each_tick(tmp_path, monkeypatch, real_dream_client):
    model, value = named_inputs(tmp_path)
    value["action_chunk_rows"] = 6
    declared = validate_embodiment_manifest(value)
    binding = team_embodiment_binding(declared, model_record=verify_model_file(declared, model),
                                     structure=read_model_structure(model), scene_plan=station_plan())
    plan = {**apply_team_embodiment(station_plan(), binding), "scenario": {"seed": 7}}
    plan["cadence"].update(maximum_action_steps=36, settle_window_samples=0)
    row = plan["robot"]

    class Env:
        def __init__(self):
            self.robot, self.steps = CartesianRobot(row), []
            self.unwrapped = SimpleNamespace(scene={"robot": self.robot, **{
                name: _Camera() for name in ("team_camera_0", "team_camera_1", "team_camera_2", "external_camera_2")}}, device="cpu")

        def reset(self, seed):
            self.robot.q = np.array([row["joint_reset_positions_rad"][n] for n in self.robot.joint_names])

        def step(self, action):
            order = [*row["joint_names"], *(j["name"] for j in row["gripper_joints"])]
            self.steps.append(list(action[0]))
            self.robot.q = np.array([action[0][order.index(n)] for n in self.robot.joint_names])

    class Policy:
        policy_id = "native-pose-contract-test"
        require_named_arm_state, action_hz, observation_hz = True, 20, 5

        def __init__(self):
            self.observations, self.history = [], []

        def observe_frame(self, observation, simulation_time_s):
            self.history.append((simulation_time_s, copy.deepcopy(observation["state"])))
            return True

        def infer(self, observation):
            self.observations.append(observation)
            return pose_chunk(rows=6)

    env, policy = Env(), Policy()
    if real_dream_client:
        from dataclasses import replace
        from blueprint_pipeline.dreamlabs_policy_client import DreamLabsPolicyClient
        from tests.test_dreamlabs_policy_client import SPEC, Service, response

        replies = [response(rows=6), response(rows=6, code="DA-0000")]
        for reply in replies:
            for arm, x, grip in (("left", .1, 0.), ("right", -.1, 1.)):
                reply["actions"]["per_key"][f"action.{arm}_eef_xyz"] = [[x, 0., 0.]]*6
                reply["actions"]["per_key"][f"action.{arm}_grip"] = [[grip]]*6
        service = Service(replies=replies)
        camera_names = [camera["name"] for camera in declared["cameras"]]
        spec = replace(SPEC, action_chunk_rows=6, open_loop_horizon=6, camera_names=camera_names,
                       camera_roles={"wrist_l": camera_names[0], "wrist_r": camera_names[1], "overhead": camera_names[2]})
        client = DreamLabsPolicyClient(spec=spec, bearer_token="fixture-only-no-live-credential", transport=service)

        class RecordedDream:
            def __init__(self):
                self.observations, self.history = [], []

            def __getattr__(self, name):
                return getattr(client, name)

            def observe_frame(self, observation, simulation_time_s):
                self.history.append((simulation_time_s, copy.deepcopy(observation["state"])))
                return client.observe_frame(observation, simulation_time_s)

            def infer(self, observation):
                self.observations.append(observation)
                return client.infer(observation)
        policy = RecordedDream()
    built = SimpleNamespace(env=env, camera_scene_names={"team_camera_0": "team_camera_0", "team_camera_1": "team_camera_1",
                            "team_camera_2": "team_camera_2", "overview": "external_camera_2"})
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    if not real_dream_client:
        monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence", lambda **_: ({"status": "complete"}, []))
    sampled = []
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec", lambda **kw: sampled.extend(kw["samples"]) or {"task_succeeded": False})
    result = run_team_episode(built=built, task_readback=None, policy=policy, plan=plan, prompt="Move the object",
                              max_policy_queries=2, open_loop_horizon=6, media_root=tmp_path / "episodes",
                              episode_id="two-real-pose-bindings", progress={}, to_action=lambda targets: [targets])
    assert len(env.steps) == 36
    assert [q["native_steps_executed"] for q in result["queries"]] == [18, 18]
    assert result["policy_cadence"] == {"native_control_hz": 60., "policy_action_hz": 20.,
                                       "native_hold_ticks_per_action": 3, "task_sample_hz": 15., "history_observation_hz": 5.}
    assert [round(t, 8) for t, _ in policy.history] == [0., .2, .4, .6]
    assert policy.observations[1]["state"]["arm_eef_pose"]["left"]["position_m"][0] > 0
    assert policy.observations[1]["state"]["arm_eef_pose"]["right"]["position_m"][0] < 0
    assert policy.observations[0]["state"]["arm_gripper_names"] == {"left": "left", "right": "right"}
    assert [sample["simulation_time_s"] for sample in sampled] == pytest.approx(np.arange(10) / 15.)
    assert all(c["pose_servo"][0]["target_achievement"] == "not_measured_by_this_command" for c in result["commanded_actions"])
    assert env.steps[-1][-2:] == [0., .03]  # Independent closures: left shut, right open.
    assert "task_object_pose_world" not in policy.observations[0]["state"]
    assert "jacobian" not in repr(policy.observations[0]["state"])
    if real_dream_client:
        assert result["policy_request_evidence_complete"] is True
        assert [len(window["frames"]) for window in result["policy_transport_windows"]] == [3, 6]
        assert result["visual_evidence"]["status"] == "complete"
        videos = result["visual_evidence"]["videos"]
        assert all(videos[role]["frame_normalization"]["target_resolution"] == [320, 176]
                   for role in ("team_camera_0", "team_camera_1", "team_camera_2"))
        client.close()
        assert service.calls[-1]["url"].endswith("/end_session")


def test_live_binding_refuses_reference_or_missing_com_jacobians():
    row = {"base_mode": "fixed_at_station", "arms": [{"name": "left", "joint_names": ["joint"],
            "base_link": "base", "tool_link": "tool", "gripper_name": "left", "tool_from_tcp": OFFSET}],
           "joint_names": ["joint"], "joint_position_limits_rad": {"joint": [-1., 1.]},
           "joint_velocity_limits": {"joint": 1.}, "control_hz": 60}
    robot = SimpleNamespace(joint_names=["joint"], data=SimpleNamespace(body_names=["base", "tool"]))
    live = LiveArmKinematics(robot=robot, row=row, as_array=np.asarray)
    with pytest.raises(AttributeError):
        live.joint_targets(positions=[[0., 0., 0.]], orientations=[IDENTITY])


@pytest.mark.parametrize("fault, expected", [("moving_base", "base_not_fixed"), ("coupled_arm", "joint_influence_undeclared")])
def test_live_arm_binding_rejects_moving_bases_and_undeclared_coupling(tmp_path, fault, expected):
    _, manifest = named_inputs(tmp_path)
    row = {"base_mode": "fixed_at_station", "arms": manifest["arms"],
           "joint_names": [joint["name"] for joint in manifest["joints"]],
           "gripper_joints": [j for g in manifest["grippers"] for j in g["joints"]],
           "joint_reset_positions_rad": manifest["reset_joint_positions"]}
    robot = CartesianRobot(row)
    if fault == "moving_base":
        row["arms"][1]["base_link"] = "left_tool"
    else:
        original = robot.get_jacobians
        def coupled():
            jacobian = original()
            jacobian[0, 0, 0, robot.joint_names.index("right_joint_0")] = 1.
            return jacobian
        robot.get_jacobians = coupled
    live = LiveArmKinematics(robot=robot, row=row, as_array=np.asarray)
    with pytest.raises(ValueError, match=expected):
        live.state()


@pytest.mark.parametrize("fault, expected", [
    (None, None), ("source_rgb_sha256", "source_mismatch"), ("source_rgb_shape", "source_mismatch"),
    ("image_sha256", "wire_hash_mismatch"), ("decoded_rgb_sha256", "decoded_hash_mismatch"),
    ("rgb", "wire_pixels_mismatch"), ("preprocessing", "preprocessing_invalid"),
    ("transmission_attempted", "attempt_evidence_missing"),
])
def test_wire_jpeg_pixels_replace_only_transmitted_history_not_native_review_renders(tmp_path, monkeypatch, fault, expected):
    from PIL import Image

    built, plan, _ = _team_loop(tmp_path, monkeypatch)
    plan["cadence"].update(maximum_action_steps=16, settle_window_samples=0)
    pixel_pattern = np.random.default_rng(31).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    for name, camera in built.env.unwrapped.scene.items():
        if name != "robot":
            camera.data.output["rgb"].torch = np.array([pixel_pattern])
    sealed = {}
    monkeypatch.setattr(episode_visual_evidence, "finalize_multicamera_visual_evidence",
                        lambda **kw: (sealed.update(kw) or {"status": "complete"}, []))
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec", lambda **_: {"task_succeeded": False})

    class JpegPolicy:
        observation_hz = 5
        action_hz = 15

        def __init__(self):
            self.buffer, self.sent = [], []

        def observe_frame(self, observation, simulation_time_s):
            for name, image in observation["images"].items():
                stream = io.BytesIO()
                Image.fromarray(image).resize((320, 176), Image.Resampling.BILINEAR).save(stream, format="JPEG", quality=85)
                jpeg = stream.getvalue()
                with Image.open(io.BytesIO(jpeg)) as opened:
                    decoded = np.asarray(opened.convert("RGB"))
                self.buffer.append({"camera_name": name, "provider_role": name,
                                    "simulation_time_s": simulation_time_s, "rgb": decoded,
                                    "transmission_attempted": True, "server_response_received": True,
                                    "image_sha256": hashlib.sha256(jpeg).hexdigest(),
                                    "source_rgb_shape": list(image.shape),
                                    "source_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                    "preprocessing": {"resize_width": 320, "resize_height": 176,
                                                      "resampling": "bilinear", "encoding": "jpeg", "jpeg_quality": 85},
                                    "decoded_rgb_sha256": hashlib.sha256(decoded.tobytes()).hexdigest()})
            return True

        def infer(self, observation):
            self.sent = copy.deepcopy(self.buffer)
            if fault == "preprocessing":
                self.sent[0][fault]["resampling"] = "nearest"
            elif fault == "rgb":
                self.sent[0][fault][0, 0, 0] ^= 1
            elif fault == "source_rgb_shape":
                self.sent[0][fault] = [320, 176, 3]
            elif fault == "transmission_attempted":
                self.sent[0][fault] = False
            elif fault is not None:
                self.sent[0][fault] = "sha256:" + "0"*64
            return [[0., 0., 0., 1.]] * 4

        def last_policy_input_frames(self):
            return self.sent

    policy = JpegPolicy()
    root = tmp_path / "wire-evidence"
    def run():
        return run_team_episode(built=built, task_readback=None, policy=policy, plan=plan, prompt="Move the object",
                                max_policy_queries=2, open_loop_horizon=4, media_root=root,
                                episode_id="wire-pixel-truth", progress={}, to_action=lambda targets: [targets])
    if fault is not None:
        with pytest.raises(ValueError, match=expected):
            run()
        return
    result = run()
    assert [row["simulation_time_s"] for row in sealed["policy_input_observations"]] == pytest.approx([0., .2])
    assert [row["simulation_time_s"] for row in sealed["review_observations"]] == pytest.approx([4/15, .4])
    assert sealed["terminal_observation"]["observation_index"] == 4
    decoded = policy.sent[0]["rgb"]
    actual_frame = sealed["policy_input_observations"][0]["views"]["team_camera_0"]
    with Image.open(root / actual_frame["relative_path"]) as image:
        assert np.array_equal(np.asarray(image), decoded)
        assert not np.array_equal(np.asarray(image), pixel_pattern)
    assert actual_frame["raw_rgb_sha256"] == "sha256:" + hashlib.sha256(decoded.tobytes()).hexdigest()
    native_intrinsic = np.asarray(sealed["review_observations"][0]["views"]["team_camera_0"]["calibration"]["intrinsic_matrix"])
    resize = np.array([[10., 0., 4.5], [0., 176/24, (176/24 - 1)/2], [0., 0., 1.]])
    assert actual_frame["calibration"]["resolution"] == [320, 176]
    assert np.asarray(actual_frame["calibration"]["intrinsic_matrix"]) == pytest.approx(resize @ native_intrinsic)
    assert len(result["policy_transport_windows"]) == 2
    assert [len(window["frames"]) for window in result["policy_transport_windows"]] == [2, 4]
    assert any(artifact["role"] == "native_render_source_manifest" for artifact in result["media_artifacts"])


def closure_for(directory):
    value = {"schema_version": "embodiment_asset_dependency_closure.v1", "model_path": "robot.usda",
             "files": [{"path": p.relative_to(directory).as_posix(), "sha256": "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest(),
                        "size_bytes": p.stat().st_size} for p in sorted(directory.rglob("*")) if p.is_file()],
             "closure_digest": ""}
    value["closure_digest"] = cross_runtime_canonical_digest(value, digest_field="closure_digest")
    return value


@pytest.mark.parametrize("fault", [None, "schema", "claim", "execution", "source_digest", "model_hash", "model_size",
                                   "model_path", "closure", "self_digest"])
def test_complete_source_import_provenance_is_bound_and_resealed_forgeries_refused(tmp_path, fault):
    from blueprint_pipeline.deployment_embodiment_asset_import import SCHEMA as IMPORT_SCHEMA

    original, value = bimanual_inputs(tmp_path)
    directory = tmp_path / "prepared"
    directory.mkdir()
    model = directory / "robot.usda"
    shutil.copyfile(original, model)
    manifest = validate_embodiment_manifest(value)
    record = {**verify_model_file(manifest, model), "dependency_closure": closure_for(directory)}
    root = record["dependency_closure"]["files"][0]
    provenance = {"schema_version": IMPORT_SCHEMA, "claim_ceiling": "source_asset_not_qualified",
                  "embodiment_id": "hermetic_import_fixture", "source_bundle_digest": "sha256:" + "4"*64,
                  "fix_base": True, "files": [copy.deepcopy(root)], "model": copy.deepcopy(root),
                  "dependency_closure": copy.deepcopy(record["dependency_closure"]),
                  "prepared_directory_self_contained": True, "source_bundle_required_for_dependency_resolution": False,
                  "importer": {"name": "hermetic-test-importer"}, "settings": {"retain": "complete-receipt"},
                  "execution_authorized": False}
    if fault == "schema":
        provenance["schema_version"] = "invented_import.v1"
    elif fault == "claim":
        provenance["claim_ceiling"] = "qualified"
    elif fault == "execution":
        provenance["execution_authorized"] = True
    elif fault == "source_digest":
        provenance["source_bundle_digest"] = "unverified"
    elif fault == "model_hash":
        provenance["model"]["sha256"] = "sha256:" + "0"*64
    elif fault == "model_size":
        provenance["model"]["size_bytes"] += 1
    elif fault == "model_path":
        provenance["model"]["path"] = "different.usda"
    elif fault == "closure":
        provenance["dependency_closure"]["closure_digest"] = "sha256:" + "0"*64
    provenance["import_digest"] = cross_runtime_canonical_digest(provenance, digest_field="import_digest")
    if fault == "self_digest":
        provenance["settings"]["retain"] = "changed-after-verification"
    record["source_import_provenance"] = provenance
    def bind():
        return team_embodiment_binding(manifest, model_record=record, structure=read_model_structure(model),
                                       scene_plan=station_plan())
    if fault is not None:
        with pytest.raises(TeamEmbodimentError, match="source_import_provenance_invalid"):
            bind()
        return
    binding = bind()
    assert binding["model"]["source_import_provenance"] == provenance
    plan = apply_team_embodiment(station_plan(), binding)
    assert plan["robot"]["source_import_provenance"] == provenance
    # An outer self-digest cannot promote or substitute the retained receipt.
    binding["model"]["source_import_provenance"]["execution_authorized"] = True
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    with pytest.raises(TeamEmbodimentError, match="source_import_provenance_invalid"):
        apply_team_embodiment(station_plan(), binding)


def test_prepared_model_closure_survives_binding_and_native_transport_verification(tmp_path):
    from pxr import Sdf, Usd
    from blueprint_pipeline.native_task_arena_runtime import NativeTaskArenaRuntimeError, _resolve_portable_robot

    model, value = bimanual_inputs(tmp_path)
    packet = tmp_path / "packet"
    directory = packet / "team_embodiment"
    (directory / "textures").mkdir(parents=True)
    root = directory / "robot.usda"
    shutil.copyfile(model, root)
    texture = directory / "textures" / "pixel.png"
    texture.write_bytes(b"exact texture bytes")
    stage = Usd.Stage.Open(str(root))
    stage.GetDefaultPrim().CreateAttribute("test:texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/pixel.png"))
    stage.GetRootLayer().Save()
    value["model"]["sha256"] = "sha256:" + hashlib.sha256(root.read_bytes()).hexdigest()
    manifest = validate_embodiment_manifest(value)
    record = {**verify_model_file(manifest, root), "dependency_closure": closure_for(directory)}
    sealed = team_embodiment_binding(manifest, model_record=record, structure=read_model_structure(root), scene_plan=station_plan())
    assert sealed["model"]["bundle_path"] == "team_embodiment/robot.usda"
    assert sealed["model"]["dependency_closure"] == record["dependency_closure"]
    plan = apply_team_embodiment(station_plan(), sealed)
    assert plan["robot"]["usd_path"] == "team_embodiment/robot.usda"
    resolved = _resolve_portable_robot(plan, bundle_root=packet)
    assert resolved["usd_path"] == str(root.resolve())
    assert resolved["usd_dependency_bindings"][0]["relative_path"] == "team_embodiment/textures/pixel.png"
    texture.write_bytes(b"tampered")
    with pytest.raises(NativeTaskArenaRuntimeError, match="dependency_identity_mismatch"):
        _resolve_portable_robot(plan, bundle_root=packet)
    texture.write_bytes(b"exact texture bytes")
    # A resealed list omitting an actually referenced file still fails native closure discovery.
    record["dependency_closure"]["files"] = [row for row in record["dependency_closure"]["files"] if row["path"] == "robot.usda"]
    record["dependency_closure"]["closure_digest"] = cross_runtime_canonical_digest(record["dependency_closure"], digest_field="closure_digest")
    sealed = team_embodiment_binding(manifest, model_record=record, structure=read_model_structure(root), scene_plan=station_plan())
    with pytest.raises(NativeTaskArenaRuntimeError, match="closure_inventory_mismatch"):
        _resolve_portable_robot(apply_team_embodiment(station_plan(), sealed), bundle_root=packet)


@pytest.mark.parametrize("fault", ["escape", "absolute", "empty", "duplicate", "model_hash", "bool_size"])
def test_closure_cannot_reseal_unsafe_paths_or_replace_root_bytes(tmp_path, fault):
    (tmp_path / "robot.usda").write_bytes(b"#usda 1.0\n")
    closure = closure_for(tmp_path)
    record = {key: closure["files"][0][key] for key in ("sha256", "size_bytes")}
    if fault == "escape":
        closure["files"][0]["path"] = "../outside.usda"
    elif fault == "absolute":
        closure["files"][0]["path"] = "/outside.usda"
    elif fault == "empty":
        closure["files"][0]["path"] = "."
    elif fault == "duplicate":
        closure["files"].append(dict(closure["files"][0]))
    elif fault == "model_hash":
        closure["files"][0]["sha256"] = "sha256:" + "1"*64
    else:
        closure["files"][0]["size_bytes"] = True
    closure["closure_digest"] = cross_runtime_canonical_digest(closure, digest_field="closure_digest")
    with pytest.raises(TeamEmbodimentError):
        validate_model_dependency_closure(closure, model_record=record)
