from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_policy_visibility import NativePolicyVisibilityReader, rgb_digest
from blueprint_pipeline.policy_observation_episode import (
    ObservationProtocolEnvironment,
    native_initial_protocol_setup,
    require_episode_acquisition_evidence,
)
from blueprint_pipeline.policy_observation_runtime_contract import (
    NativeObservationProtocol,
    apply_native_cell_protocol,
    camera_calibration_digest,
    seal_native_observation_protocol,
    validate_native_cell_protocol,
    validate_search_gate,
)
from blueprint_pipeline.native_task_arena_policy_canary_worker import _resolved_scene_plan
from tests.test_adp009d_policy_episode import (
    _LifecycleEnvironment,
    _LifecyclePolicy,
    _run,
    _runtime_observation_gate,
)
from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import _scene_plan
from tests.test_policy_observation_information import coordinates, information
from tests.test_policy_object_acquisition import protocol

DIGEST = "sha256:" + "a" * 64
IDENTITY = np.eye(4).reshape(-1).tolist()


def source_plan():
    plan = _scene_plan()
    plan["scene_id"] = "fixture_workcell"
    plan["objects"][0]["asset_id"] = "fixture_object"
    plan["cameras"] = [
        {
            "role": role,
            "policy_input": role != "overview",
            "review_only": role == "overview",
            "frame_from_camera_matrix": list(IDENTITY),
            "optical_convention": "opencv",
            "pose_frame": "robot_body" if role == "wrist" else "world",
            "parent_prim_path": "{ENV_REGEX_NS}/Robot/panda_hand"
            if role == "wrist"
            else "{ENV_REGEX_NS}",
            "intrinsics": {
                "fx": 20.0,
                "fy": 20.0,
                "cx": 15.5,
                "cy": 11.5,
                "width": 32,
                "height": 24,
            },
        }
        for role in ("external", "wrist", "overview")
    ]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def native_binding(
    *, cell=None, plan=None, task_success_contract=None, visibility="initially_out_of_view"
):
    from tests.test_task_evaluation_policy_canary_setup import _setup

    scenario = {"family": "camera_sensor", "parameters": {}}
    cell = cell or {
        "cell_id": "search-cell",
        "seed": 41,
        "cell_spec_digest": DIGEST,
        "family": "camera_sensor",
        "resolved_scenario": scenario,
        "resolved_scenario_digest": canonical_digest(scenario),
    }
    plan = plan or source_plan()
    contract = task_success_contract or _setup()["task_success_contract"]
    resolved = _resolved_scene_plan(plan, cell, task_success_contract=contract)
    subject = next(o for o in resolved["objects"] if o.get("task_subject") is True)
    info = information(
        scene_id=resolved["scene_id"],
        task_id=resolved["task_id"],
        setup_object_coordinates=coordinates(
            object_id=subject["asset_id"],
            position_m=subject["pose_world"]["position_world_m"],
            measurement_context_digest=DIGEST,
        ),
    )
    cameras = {c["role"]: c for c in resolved["cameras"] if c["role"] in {"external", "wrist"}}
    acquisition = protocol(
        visibility,
        camera_calibration_digests={
            role: camera_calibration_digest(c) for role, c in cameras.items()
        },
    )
    return seal_native_observation_protocol(
        {
            **{
                name: cell[name]
                for name in ("cell_id", "seed", "cell_spec_digest", "resolved_scenario_digest")
            },
            "reset_digest": DIGEST,
            "task_success_contract_digest": contract["contract_digest"],
            "preregistration_digest": DIGEST,
            "source_plan_digest": DIGEST,
            "native_scene_plan_digest": resolved["plan_digest"],
            "source_configuration_digest": DIGEST,
            "information": info.model_dump(mode="json"),
            "acquisition": acquisition.model_dump(mode="json"),
            "camera_setups": {
                role: {
                    "frame_from_camera_matrix": c["frame_from_camera_matrix"],
                    "reset_world_from_camera_opencv_matrix": c["frame_from_camera_matrix"],
                    "source_intrinsics_digest": canonical_digest(c["intrinsics"]),
                }
                for role, c in cameras.items()
            },
        }
    )


def test_native_binding_cannot_drift_cell_reset_task_or_geometry():
    raw = native_binding()
    binding = NativeObservationProtocol.model_validate(raw)
    cell = {
        name: raw[name]
        for name in ("cell_id", "seed", "cell_spec_digest", "resolved_scenario_digest")
    }
    cell.update(family="camera_sensor", observation_protocol=raw)
    with pytest.raises(ValueError, match="operator_camera_aim_conflict"):
        validate_native_cell_protocol(
            {
                **cell,
                "operator_wrist_camera_aim": {
                    "mode": "point_at_task_object_then_rigidly_follow_wrist"
                },
            },
            task_success_contract_digest=raw["task_success_contract_digest"],
        )
    assert (
        validate_native_cell_protocol(
            cell, task_success_contract_digest=raw["task_success_contract_digest"]
        )
        == binding
    )
    for field, value in (("seed", 42), ("cell_id", "wrong")):
        with pytest.raises(ValueError, match="cell_binding_mismatch"):
            validate_native_cell_protocol(
                {**cell, field: value},
                task_success_contract_digest=raw["task_success_contract_digest"],
            )
    with pytest.raises(ValueError, match="canonical_search_forbidden"):
        validate_native_cell_protocol(
            {**cell, "family": "canonical_anchor"},
            task_success_contract_digest=raw["task_success_contract_digest"],
        )
    drift = copy.deepcopy(raw)
    drift["camera_setups"]["wrist"]["frame_from_camera_matrix"][0] = 2.0
    with pytest.raises(ValueError, match="rigid_matrix_invalid"):
        seal_native_observation_protocol(drift)


def _search_gate(binding):
    gate = _runtime_observation_gate()
    setup = {
        "schema_version": "native_observation_protocol_setup.v1",
        "binding_digest": binding["binding_digest"],
        "initial_visibility": binding["acquisition"]["initial_visibility"],
        "geometry_passed": True,
        "sensor_freshness_passed": True,
        "initial_condition_passed": True,
    }
    setup["receipt_digest"] = canonical_digest(setup)
    gate.update(
        observation_protocol=binding,
        observation_protocol_setup=setup,
        target_semantic_visibility_passed=False,
    )
    gate["gate_digest"] = canonical_digest(gate, digest_field="gate_digest")
    return gate


def test_search_gate_requires_exact_binding_and_never_admits_qualified_scope():
    binding = native_binding()
    gate = _search_gate(binding)
    assert validate_search_gate(gate, expected_binding_digest=binding["binding_digest"])
    assert not validate_search_gate(gate, expected_binding_digest=None)
    gate["claim_ceiling"] = "qualified_evaluation"
    gate["gate_digest"] = canonical_digest(gate, digest_field="gate_digest")
    assert not validate_search_gate(gate, expected_binding_digest=binding["binding_digest"])


def test_search_controls_keep_native_render_and_overview_subject_destination_checks():
    from blueprint_pipeline.native_policy_canary_control_gate import validate_strict_camera_gate

    binding = native_binding()
    gate = _search_gate(binding)
    gate["snapshot"] = {
        "cameras": [
            {
                "role": role,
                "observability": {
                    "render_passed": True,
                    "thresholds": {"effective_minimum_pixels": 8},
                },
                "semantic_label_pixels": {
                    label: {"pixel_count": 20 if role == "overview" else 0}
                    for label in ("task_object", "task_support")
                },
            }
            for role in ("external", "wrist", "overview")
        ]
    }
    gate["gate_digest"] = canonical_digest(gate, digest_field="gate_digest")
    validate_strict_camera_gate(gate, observation_protocol_binding_digest=binding["binding_digest"])
    with pytest.raises(RuntimeError, match="subject_destination_visibility_failed"):
        validate_strict_camera_gate(gate)
    gate["snapshot"]["cameras"][-1]["semantic_label_pixels"]["task_support"]["pixel_count"] = 0
    gate["gate_digest"] = canonical_digest(gate, digest_field="gate_digest")
    with pytest.raises(RuntimeError, match="subject_destination_visibility_failed:overview"):
        validate_strict_camera_gate(
            gate, observation_protocol_binding_digest=binding["binding_digest"]
        )


class FreshEnvironment(_LifecycleEnvironment):
    def read_policy_inputs(self):
        value = super().read_policy_inputs()
        value["sensor_freshness"] = {
            role: {"status": "observed", "control_step_index": self._t, "frame_index": self._t + 1}
            for role in ("external", "wrist")
        }
        return value


class FixtureVisibilityReader:
    def __init__(self, environment, *, visible_from_step=8, stale=False):
        self.environment = environment
        self.visible_from_step = visible_from_step
        self.stale = stale
        self.geometry_reads = 0
        self.closed = False

    def read_geometry(self):
        self.geometry_reads += 1
        return {"geometry_passed": True, "geometry_digest": DIGEST}

    def close(self):
        self.closed = True

    def capture(self, raw_rgb):
        t = self.environment._t
        rows = {}
        for role, rgb in raw_rgb.items():
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            if t >= self.visible_from_step:
                mask[8:16, 8:16] = 1
            rows[role] = {
                "mask": mask,
                "semantic_ids": mask.astype(np.int32),
                "id_to_labels": {"1": {"class": "task_object"}},
                "target_boxes": [{"occlusion_ratio": 0.0}] if mask.any() else [],
                "raw_rgb_digest": rgb_digest(rgb),
                "renderer_frame": 1 if self.stale else t + 1,
            }
        return {"native_time_s": 10.0 + t / 15.0, "physics_step": 100 + t * 8, "cameras": rows}


def test_real_episode_retains_exact_frame_acquisition_before_query_without_xyz(tmp_path):
    binding = native_binding()
    env = FreshEnvironment()
    reader = FixtureVisibilityReader(env)
    wrapped = ObservationProtocolEnvironment(environment=env, binding=binding, reader=reader)
    policy = _LifecyclePolicy()
    receipt = _run(
        environment=wrapped,
        policy=policy,
        max_policy_queries=2,
        settle_window_samples=1,
        require_prestart_readiness=True,
        require_complete_multicamera_media=True,
        observation_protocol=binding,
        media_output_dir=tmp_path,
        episode_id="search-runtime-test",
    )
    acquisition = receipt["object_acquisition"]
    assert acquisition["assessment"]["first_observed_acquisition_seconds"] == 8 / 15.0
    assert acquisition["assessment"]["acquisition_is_task_success"] is False
    assert len(acquisition["sample_artifacts"]) == 2
    assert reader.geometry_reads >= 3
    for row, frame in zip(
        acquisition["sample_artifacts"], receipt["candidate_exact_policy_input_frames"], strict=True
    ):
        evidence_path = tmp_path / row["relative_path"]
        assert row["sha256"] == "sha256:" + hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        evidence = json.loads(evidence_path.read_text())
        assert evidence["exact_policy_frame_manifest_digest"] == frame["frame_manifest_digest"]
        for artifact in evidence["files"]:
            assert (tmp_path / artifact["relative_path"]).is_file()
    assert all(
        set(observation)
        == {
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
            "observation/joint_position",
            "observation/gripper_position",
            "prompt",
        }
        for observation in policy.observations
    )
    require_episode_acquisition_evidence(episode=receipt, binding=binding, output_root=tmp_path)
    first_evidence = json.loads(
        (tmp_path / acquisition["sample_artifacts"][0]["relative_path"]).read_text()
    )
    artifact = tmp_path / first_evidence["files"][0]["relative_path"]
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact_digest_mismatch"):
        require_episode_acquisition_evidence(episode=receipt, binding=binding, output_root=tmp_path)


def test_runtime_rejects_stale_search_frames_before_second_query(tmp_path):
    binding = native_binding()
    env, policy = FreshEnvironment(), _LifecyclePolicy()
    reader = FixtureVisibilityReader(env, stale=True)
    wrapped = ObservationProtocolEnvironment(environment=env, binding=binding, reader=reader)
    with pytest.raises(ValueError, match="sensor_freshness_mismatch"):
        _run(
            environment=wrapped,
            policy=policy,
            max_policy_queries=2,
            settle_window_samples=1,
            require_prestart_readiness=True,
            require_complete_multicamera_media=True,
            observation_protocol=binding,
            media_output_dir=tmp_path,
            episode_id="stale-search",
        )
    assert len(policy.observations) == 1


def test_native_geometry_overlay_is_explicit_and_keeps_base_inputs_unchanged():
    plan = source_plan()
    scenario = {"family": "camera_sensor", "parameters": {}}
    cell = {
        "cell_id": "search-cell",
        "seed": 41,
        "cell_spec_digest": DIGEST,
        "family": "camera_sensor",
        "resolved_scenario": scenario,
        "resolved_scenario_digest": canonical_digest(scenario),
    }
    binding = native_binding(cell=cell, plan=plan)
    from tests.test_task_evaluation_policy_canary_setup import _setup

    resolved = _resolved_scene_plan(
        plan, cell, task_success_contract=_setup()["task_success_contract"]
    )
    before = copy.deepcopy(resolved)
    applied = apply_native_cell_protocol(resolved, {**cell, "observation_protocol": binding})
    assert resolved == before
    assert (
        applied["policy_canary_embodiment_profile"]["preserve_official_policy_camera_calibration"]
        is False
    )
    assert applied["observation_protocol"]["binding_digest"] == binding["binding_digest"]
    assert (
        applied["policy_canary_embodiment_profile"]["preserve_official_policy_camera_intrinsics"]
        is False
    )
    mismatch = copy.deepcopy(resolved)
    mismatch["objects"][0]["pose_world"]["position_world_m"][1] += 0.001
    with pytest.raises(ValueError, match="object_position_mismatch"):
        apply_native_cell_protocol(mismatch, {**cell, "observation_protocol": binding})


def native_reader_fixture(*, occlusion_ratio=0.0, visible=True):
    binding = native_binding(
        visibility="partially_occluded"
        if occlusion_ratio
        else "visible"
        if visible
        else "initially_out_of_view"
    )
    plan = source_plan()
    native = SimpleNamespace(current_time=10.0, current_time_step_index=1200)
    scene = {}
    channels = {}
    frame = FreshEnvironment()._camera_frame(50)
    for role in ("external", "wrist"):
        semantic = np.zeros(frame.shape[:2], dtype=np.int32)
        if visible:
            semantic[8:16, 8:16] = 1
        rgba = np.concatenate([frame, np.full((*frame.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
        camera = SimpleNamespace(
            frame=np.array([1]),
            cfg=SimpleNamespace(
                offset=SimpleNamespace(pos=[0, 0, 0], rot=[0, 0, 0, 1], convention="ros")
            ),
            data=SimpleNamespace(
                output={"rgb": rgba[None], "semantic_segmentation": semantic[None, ..., None]},
                info={"semantic_segmentation": {"idToLabels": {"1": {"class": "task_object"}}}},
                intrinsic_matrices=np.array([[[20.0, 0, 16.0], [0, 20.0, 12.0], [0, 0, 1]]]),
                pos_w=np.zeros((1, 3)),
                quat_w_opengl=np.array([[1.0, 0, 0, 0]]),
            ),
            _render_data=SimpleNamespace(
                render_product=SimpleNamespace(path=f"/render/{role}"),
                spec=SimpleNamespace(camera_prim_paths=(f"/camera/{role}",)),
            ),
            _view=SimpleNamespace(prim_paths=(f"/camera/{role}",)),
        )
        scene[role] = camera
        boxes = (
            [
                {
                    "semanticId": 1,
                    "occlusionRatio": occlusion_ratio,
                    "x_min": 8,
                    "y_min": 8,
                    "x_max": 16,
                    "y_max": 16,
                }
            ]
            if visible
            else []
        )
        channels[f"/render/{role}"] = {
            "rgb": rgba,
            "ReferenceTime": {"referenceTimeNumerator": 100, "referenceTimeDenominator": 10},
            "bounding_box_2d_loose_fast": {
                "data": boxes,
                "info": {"idToLabels": {"1": {"class": "task_object"}}},
            },
        }
    scene["task_object"] = SimpleNamespace(
        data=SimpleNamespace(
            root_pose_w=np.array(
                [
                    [
                        *binding["information"]["setup_object_coordinates"]["position_m"],
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                ]
            )
        )
    )
    built = SimpleNamespace(
        env=SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, sim=native)),
        plan=plan,
        camera_scene_names={"external": "external", "wrist": "wrist"},
        scene_asset_names={"task_object": "task_object"},
    )
    detached = []

    class Annotator:
        def __init__(self, name):
            self.name = name

        def attach(self, paths):
            self.path = paths[0]

        def detach(self, paths):
            detached.append((self.name, paths[0]))

        def get_data(self):
            return channels[self.path][self.name]

    reader = NativePolicyVisibilityReader(
        built=built, binding=binding, annotator_factory=lambda name, **kwargs: Annotator(name)
    )
    return reader, binding, channels, detached


@pytest.mark.parametrize("visible,ratio", [(True, 0.0), (True, 0.4), (False, 0.0)])
def test_native_aovs_validate_all_three_initial_conditions_on_existing_render_products(
    visible, ratio
):
    reader, binding, _, detached = native_reader_fixture(visible=visible, occlusion_ratio=ratio)
    setup = native_initial_protocol_setup(reader=reader, binding=binding)
    assert setup["initial_visibility"] == binding["acquisition"]["initial_visibility"]
    assert setup["geometry_passed"] and setup["sensor_freshness_passed"]
    assert set(setup["candidate_initial_assessments"]) == {"pi05_droid", "groot_n17_droid"}
    reader.close()
    assert len(detached) == 6


def test_native_reader_uses_the_pinned_physics_counter_method():
    reader, binding, _, _ = native_reader_fixture()
    reader.built.env.unwrapped.sim.get_physics_step_count = lambda: 1300
    setup = native_initial_protocol_setup(reader=reader, binding=binding)
    assert setup["initial_condition_passed"] is True
    raw = {
        role: reader.built.env.unwrapped.scene[role].data.output["rgb"][0, ..., :3]
        for role in ("external", "wrist")
    }
    assert reader.capture(raw)["physics_step"] == 1300


def test_native_reader_refuses_a_different_rendered_camera_before_attaching():
    reader, binding, _, _ = native_reader_fixture()
    reader.built.env.unwrapped.scene["external"]._render_data.spec.camera_prim_paths = (
        "/camera/other",
    )
    with pytest.raises(ValueError, match="render_camera_identity_mismatch"):
        NativePolicyVisibilityReader(
            built=reader.built,
            binding=binding,
            annotator_factory=lambda *args, **kwargs: pytest.fail(
                "mismatch must fail before attachment"
            ),
        )


def test_native_intrinsics_use_the_pinned_aperture_center_without_loose_tolerance():
    reader, _, _, _ = native_reader_fixture()
    assert reader.read_geometry()["geometry_passed"]
    # The pinned Camera returns w/2,h/2. The scene-plan convention is (w-1)/2,
    # (h-1)/2; it must not be silently treated as the observed native matrix.
    reader.built.env.unwrapped.scene["external"].data.intrinsic_matrices[0, 0, 2] = 15.5
    with pytest.raises(ValueError, match="camera_intrinsics_mismatch"):
        reader.read_geometry()


@pytest.mark.parametrize(
    "fault,error",
    [
        ("rgb", "policy_rgb_frame_mismatch"),
        ("time", "render_time_stale"),
        ("pose", "reset_world_pose_mismatch"),
        ("bbox", "semantic_bbox_disagreement"),
    ],
)
def test_native_reader_rejects_mismatched_frames_time_pose_and_semantics(fault, error):
    reader, binding, channels, _ = native_reader_fixture()
    if fault == "rgb":
        channels["/render/wrist"]["rgb"] = np.zeros_like(channels["/render/wrist"]["rgb"])
    elif fault == "time":
        channels["/render/wrist"]["ReferenceTime"]["referenceTimeNumerator"] = 90
    elif fault == "pose":
        reader.built.env.unwrapped.scene["wrist"].data.pos_w[0, 0] = 0.5
    else:
        channels["/render/wrist"]["bounding_box_2d_loose_fast"]["data"] = []
    with pytest.raises(ValueError, match=error):
        native_initial_protocol_setup(reader=reader, binding=binding)
