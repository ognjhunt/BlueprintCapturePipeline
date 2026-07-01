from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from blueprint_pipeline import vast_provider_adapter
from blueprint_pipeline import unitree_groot_n17_sonic_vast_persistent_session as session
from blueprint_pipeline import persistent_wam_short_visual_sanity as short_sanity


def _clean_launch_git_evidence() -> dict[str, object]:
    return {
        "status": "available",
        "repo_root": "/repo",
        "git_sha": "0" * 40,
        "dirty": False,
        "dirty_entries_count": 0,
        "dirty_entries": [],
        "dirty_entries_truncated": False,
    }


@pytest.fixture(autouse=True)
def clean_launch_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        session.launch_provenance,
        "git_worktree_evidence",
        _clean_launch_git_evidence,
    )


def _persistent_runner_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {
        "__name__": "_persistent_session_runner_under_test",
        "__file__": "persistent_session_runner_under_test.py",
    }
    exec(session.PERSISTENT_SESSION_RUNNER, namespace)
    return namespace


def test_embedded_persistent_session_runner_compiles() -> None:
    compile(
        session.PERSISTENT_SESSION_RUNNER,
        "<unitree_groot_n17_sonic_persistent_session_runner>",
        "exec",
    )


def test_embedded_persistent_session_runner_carries_review_quality_horizon() -> None:
    namespace = _persistent_runner_namespace()

    assert namespace["REVIEW_QUALITY_MIN_OSCAR_NUM_FRAMES"] == 81


def test_persistent_runner_future_frame_selector_uses_copied_runtime_module() -> None:
    assert "from blueprint_pipeline.wam_generated_video_review import" in (
        session.PERSISTENT_SESSION_RUNNER
    )
    assert "from blueprint_pipeline.oscar_isaac_closed_loop_eval import" not in (
        session.PERSISTENT_SESSION_RUNNER
    )


def test_persistent_runner_generated_frame_visual_gate_blocks_collapsed_feedback(
    tmp_path: Path,
) -> None:
    namespace = _persistent_runner_namespace()
    visual_gate = namespace["_generated_next_observation_visual_gate"]
    rng = np.random.default_rng(17)
    source_frame = tmp_path / "source.jpg"
    generated_frame = tmp_path / "generated.jpg"
    Image.fromarray(
        rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8),
        mode="RGB",
    ).save(source_frame)
    Image.new("RGB", (128, 96), (112, 112, 112)).save(generated_frame)

    result = visual_gate(
        source_frame=source_frame,
        generated_frame=generated_frame,
        materialization={
            "source_kind": "video_future_frame",
            "selection_quality_status": "passed_signal_gate",
        },
    )

    assert result["status"] == "failed_visual_quality_gate"
    assert "wam_generated_frame_edge_structure_drift" in result["blockers"]
    assert "wam_generated_frame_entropy_drift" in result["blockers"]
    assert result["claim_boundary"]["visual_gate_blocks_autoregressive_policy_feedback"] is True


def test_persistent_runner_generated_frame_visual_gate_blocks_first_frame_fallback(
    tmp_path: Path,
) -> None:
    namespace = _persistent_runner_namespace()
    visual_gate = namespace["_generated_next_observation_visual_gate"]
    source_frame = _write_reviewable_frame(tmp_path / "source.jpg")
    generated_frame = _write_reviewable_frame(tmp_path / "generated.jpg")

    result = visual_gate(
        source_frame=source_frame,
        generated_frame=generated_frame,
        materialization={
            "source_kind": "video_first_frame",
            "selection_quality_status": "passed_signal_gate",
        },
    )

    assert result["status"] == "failed_visual_quality_gate"
    assert "wam_generated_next_observation_used_video_first_frame_fallback" in result[
        "blockers"
    ]


def test_persistent_runner_rgb_history_tracks_only_unique_existing_frames(
    tmp_path: Path,
) -> None:
    namespace = _persistent_runner_namespace()
    append_unique = namespace["_frame_history_append_unique"]
    history_window = namespace["_frame_history_window"]
    frame_a = _write_reviewable_frame(tmp_path / "frame_a.jpg")
    frame_b = _write_reviewable_frame(tmp_path / "frame_b.jpg")
    missing = tmp_path / "missing.jpg"
    history: list[str] = []

    append_unique(history, frame_a)
    append_unique(history, frame_a)
    append_unique(history, missing)
    append_unique(history, frame_b)

    assert history == [str(frame_a.resolve()), str(frame_b.resolve())]
    assert history_window(history, max_frames=1) == [
        str(frame_a.resolve()),
        str(frame_b.resolve()),
    ]
    assert history_window(history, max_frames=2) == [
        str(frame_a.resolve()),
        str(frame_b.resolve()),
    ]


def test_persistent_runner_strips_seed_skeleton_until_policy_derived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "BLUEPRINT_ALLOW_SEED_DERIVED_SKELETON_FOR_ACTION_CONDITIONED_WAM",
        raising=False,
    )
    namespace = _persistent_runner_namespace()
    prepare = namespace["_prepare_action_conditioned_wam_inputs"]
    seed_trace = tmp_path / "seed_projected_trace.jsonl"
    seed_trace.write_text(
        json.dumps(
            {
                "claim_boundary": {
                    "projected_skeleton_trace_derived_from_seed_render_geometry": True,
                    "temporal_rows_are_target_conditioning_from_resolved_affordance_projection": True,
                    "not_a_learned_robot_policy_action": True,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observation = {
        "visual_observation": {
            "projected_skeleton_trace_path": str(seed_trace),
            "g1_projected_skeleton_trace_jsonl": str(seed_trace),
        }
    }
    auxiliary = {
        "action_conditioning": {
            "projected_skeleton_trace_path": str(seed_trace),
            "projected_hand_keypoint_trace_path": str(seed_trace),
        }
    }

    sanitized_observation, sanitized_auxiliary, manifest_path, contract = prepare(
        observation=observation,
        auxiliary_observation=auxiliary,
        auxiliary_manifest_path=str(tmp_path / "auxiliary.json"),
        source_policy_action={"action_chunk": [0.1, 0.2]},
    )

    assert manifest_path == ""
    assert "projected_skeleton_trace_path" not in sanitized_observation["visual_observation"]
    assert "g1_projected_skeleton_trace_jsonl" not in sanitized_observation[
        "visual_observation"
    ]
    action_conditioning = sanitized_auxiliary["action_conditioning"]
    assert "projected_skeleton_trace_path" not in action_conditioning
    assert "projected_hand_keypoint_trace_path" not in action_conditioning
    assert action_conditioning["projected_trace_removed_for_policy_ranking_safety"] is True
    assert contract["policy_ranking_claim_safe"] is False
    assert contract["status"] == (
        "stripped_seed_or_target_projected_skeleton_for_policy_action_conditioning"
    )
    assert contract["blockers"] == [
        "policy_action_to_projected_skeleton_decoder_missing_for_ranking_safe_wam"
    ]


def test_persistent_runner_keeps_policy_derived_skeleton_trace(
    tmp_path: Path,
) -> None:
    namespace = _persistent_runner_namespace()
    prepare = namespace["_prepare_action_conditioned_wam_inputs"]
    policy_trace = tmp_path / "policy_projected_trace.jsonl"
    policy_trace.write_text(
        json.dumps(
            {
                "claim_boundary": {
                    "policy_derived_action_conditioning": True,
                    "scene_faithful_isaac_policy_action_projection_bridge_used": True,
                    "blueprint_simulator_only_isaac_action_projection_bridge_used": True,
                    "simulated_state_not_physical_robot_sensor_evidence": True,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observation = {
        "visual_observation": {
            "projected_skeleton_trace_path": str(policy_trace),
        }
    }

    sanitized_observation, _sanitized_auxiliary, manifest_path, contract = prepare(
        observation=observation,
        auxiliary_observation={},
        auxiliary_manifest_path=str(tmp_path / "auxiliary.json"),
        source_policy_action={"action_chunk": [0.1, 0.2]},
    )

    assert manifest_path == str(tmp_path / "auxiliary.json")
    assert sanitized_observation["visual_observation"]["projected_skeleton_trace_path"] == str(
        policy_trace
    )
    assert contract["status"] == "policy_derived_projected_skeleton_trace_available"
    assert contract["policy_ranking_claim_safe"] is True
    assert contract["selected_projected_skeleton_trace_path"] == str(policy_trace)


def test_persistent_runner_accepts_policy_action_projected_skeleton_trace(
    tmp_path: Path,
) -> None:
    namespace = _persistent_runner_namespace()
    prepare = namespace["_prepare_action_conditioned_wam_inputs"]
    policy_trace = tmp_path / "policy_action_projected_trace.jsonl"
    policy_trace.write_text(
        json.dumps(
            {
                "claim_boundary": {
                    "policy_derived_action_conditioning": True,
                    "scene_faithful_isaac_policy_action_projection_bridge_used": True,
                    "blueprint_simulator_only_isaac_action_projection_bridge_used": True,
                    "simulated_state_not_physical_robot_sensor_evidence": True,
                },
                "projected_landmark_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sanitized_observation, _sanitized_auxiliary, manifest_path, contract = prepare(
        observation={"visual_observation": {}},
        auxiliary_observation={},
        auxiliary_manifest_path=str(tmp_path / "auxiliary.json"),
        source_policy_action={
            "action_chunk": [0.1, 0.2],
            "policy_action_projected_skeleton_trace_path": str(policy_trace),
        },
    )

    assert manifest_path == str(tmp_path / "auxiliary.json")
    assert sanitized_observation["visual_observation"] == {}
    assert contract["status"] == "policy_derived_projected_skeleton_trace_available"
    assert contract["policy_derived_projected_skeleton_trace_present"] is True
    assert contract["policy_ranking_claim_safe"] is True
    assert contract["selected_projected_skeleton_trace_path"] == str(policy_trace)


def test_persistent_runner_materializes_nominal_policy_action_trace_without_ranking_claim(
    tmp_path: Path,
) -> None:
    namespace = _persistent_runner_namespace()
    prepare = namespace["_prepare_action_conditioned_wam_inputs"]
    sonic_frame = ([0.1] * 28) + ([0.0] * 36) + ([0.2] * 14)
    source_action = {
        "action_type": "unitree_g1_sonic_latent_action_chunk",
        "action_chunk": [*sonic_frame, *sonic_frame],
    }

    _sanitized_observation, _sanitized_auxiliary, manifest_path, contract = prepare(
        observation={
            "visual_observation": {
                "camera_id": "head_pov",
                "width": 1672,
                "height": 941,
            }
        },
        auxiliary_observation={},
        auxiliary_manifest_path=str(tmp_path / "auxiliary.json"),
        source_policy_action=source_action,
        work_dir=tmp_path / "worker_step",
    )

    trace_path = Path(source_action["policy_action_projected_skeleton_trace_path"])
    assert manifest_path == str(tmp_path / "auxiliary.json")
    assert trace_path.is_file()
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["projected_landmark_count"] == 8
    assert rows[0]["image_width_px"] == 1672
    assert rows[0]["image_height_px"] == 941
    assert rows[0]["source_image_width_px"] == 1672
    assert rows[0]["source_image_height_px"] == 941
    assert rows[0]["coordinate_space"] == "source_policy_observation_pixels"
    assert rows[0]["landmarks"][0]["image_projection"]["image_width_px"] == 1672
    assert rows[0]["landmarks"][0]["image_projection"]["image_height_px"] == 941
    assert rows[0]["landmarks"][0]["image_projection"]["inside_image"] is True
    assert rows[0]["claim_boundary"]["policy_derived_action_conditioning"] is True
    assert (
        rows[0]["claim_boundary"]["nominal_kinematic_projection_without_scene_or_wbc_bridge"]
        is True
    )
    assert contract["status"] == "nominal_policy_action_projected_skeleton_trace_available"
    assert contract["policy_derived_projected_skeleton_trace_present"] is True
    assert contract["ranking_safe_projected_skeleton_trace_present"] is False
    assert contract["policy_ranking_claim_safe"] is False
    assert (
        "nominal_policy_action_projection_without_scene_or_wbc_bridge"
        in contract["blockers"]
    )


def test_persistent_runner_prefers_isaac_geometry_policy_action_trace_for_sim_ranking(
    tmp_path: Path,
) -> None:
    namespace = _persistent_runner_namespace()
    prepare = namespace["_prepare_action_conditioned_wam_inputs"]
    geometry = tmp_path / "manipulation_pov_geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "status": "PASS",
                "frames": [
                    {
                        "status": "PASS",
                        "camera": "robot_pov",
                        "camera_meta": {
                            "camera_eye_xyz": [0.0, 0.0, 0.0],
                            "camera_target_xyz": [1.0, 0.0, 0.0],
                            "camera_vfov_deg": 90.0,
                            "viewport_size_px": [80, 60],
                            "arm_link_points_by_arm_xyz": {
                                "left": {
                                    "shoulder": [1.0, -0.16, -0.22],
                                    "elbow": [1.08, -0.18, -0.30],
                                    "wrist": [1.16, -0.20, -0.36],
                                    "hand": [1.24, -0.22, -0.40],
                                },
                                "right": {
                                    "shoulder": [1.0, 0.16, -0.22],
                                    "elbow": [1.08, 0.18, -0.30],
                                    "wrist": [1.16, 0.20, -0.36],
                                    "hand": [1.24, 0.22, -0.40],
                                },
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sonic_frame = ([0.2] * 28) + ([0.0] * 36) + ([0.1] * 14)
    source_action = {
        "action_type": "unitree_g1_sonic_latent_action_chunk",
        "action_chunk": [*sonic_frame, *sonic_frame],
    }

    _sanitized_observation, _sanitized_auxiliary, manifest_path, contract = prepare(
        observation={
            "manipulation_pov_geometry_path": str(geometry),
            "visual_observation": {
                "camera_id": "robot_pov",
                "manipulation_pov_geometry_path": str(geometry),
            },
        },
        auxiliary_observation={},
        auxiliary_manifest_path=str(tmp_path / "auxiliary.json"),
        source_policy_action=source_action,
        work_dir=tmp_path / "worker_step",
    )

    trace_path = Path(source_action["policy_action_projected_skeleton_trace_path"])
    assert manifest_path == str(tmp_path / "auxiliary.json")
    assert trace_path.name == "policy_action_isaac_geometry_projected_skeleton_trace.jsonl"
    assert trace_path.is_file()
    assert contract["status"] == "policy_derived_projected_skeleton_trace_available"
    assert contract["geometry_anchored_policy_action_projected_skeleton_trace_path"] == str(
        trace_path
    )
    assert contract["nominal_policy_action_projected_skeleton_trace_path"] is None
    assert contract["policy_derived_projected_skeleton_trace_present"] is True
    assert contract["ranking_safe_projected_skeleton_trace_present"] is True
    assert contract["policy_ranking_claim_safe"] is True
    assert contract["blockers"] == []
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["source_geometry_path"] == str(geometry.resolve())
    assert rows[0]["projected_landmark_count"] == 8
    assert rows[0]["claim_boundary"]["policy_action_delta_applied_to_seed_geometry"] is True
    assert (
        rows[0]["claim_boundary"]["projected_skeleton_trace_derived_from_seed_render_geometry"]
        is True
    )
    assert (
        rows[0]["claim_boundary"]["dynamic_scene_coordinates_from_artifact_not_source_code"]
        is True
    )
    assert rows[0]["claim_boundary"]["official_wbc_or_sim_bridge_used"] is False
    assert (
        rows[0]["claim_boundary"]["scene_faithful_isaac_policy_action_projection_bridge_used"]
        is True
    )
    assert (
        rows[0]["claim_boundary"]["uses_isaac_sidecar_link_landmarks_not_hand_drawn_screen_axes"]
        is True
    )
    assert rows[0]["claim_boundary"]["full_g1_urdf_fk_solver_used"] is False
    assert (
        rows[0]["claim_boundary"]["sonic_action_delta_is_heuristic_reach_lift_not_official_wbc"]
        is False
    )
    assert rows[0]["claim_boundary"]["sidecar_kinematic_chain_fk_solver_used"] is True
    assert (
        rows[0]["claim_boundary"]["sonic_action_delta_is_heuristic_joint_delta_not_official_wbc"]
        is True
    )
    assert rows[0]["kinematic_chain"] == {
        "source": "isaac_manipulation_pov_geometry_arm_link_points",
        "projection_method": "isaac_camera_sidecar_pinhole_projection",
        "action_delta_method": "sonic_action_chunk_sidecar_upper_body_fk_joint_deltas",
        "sidecar_kinematic_chain_fk_executed": True,
        "urdf_fk_solver_executed": False,
        "full_g1_urdf_fk_executed": False,
        "official_groot_wholebodycontrol_sim2sim_executed": False,
    }
    assert rows[0]["landmarks"][1]["world_xyz_m"] != rows[0]["landmarks"][1]["seed_world_xyz_m"]
    assert (
        contract["claim_boundary"][
            "geometry_anchored_policy_action_projection_is_wam_conditioning_not_ranking_proof"
        ]
        is False
    )


def test_policy_action_decoding_contract_blocks_latent_without_pose_decoder() -> None:
    contract = session._policy_action_decoding_contract(
        {
            "action_type": "unitree_g1_sonic_latent_action_chunk",
            "action_chunk": [0.1, -0.2, 0.3, 0.0],
            "sonic_latent_action": [[[0.1, -0.2], [0.0, 0.3]]],
            "hand_targets": {
                "left_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                "right_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            },
            "unitree_g1_sonic_control_fields": [
                "left_hand_joints",
                "motion_token",
                "right_hand_joints",
            ],
        },
        generated_at="now",
    )

    assert contract["status"] == "blocked_latent_action_without_pose_decoder"
    assert contract["latent_action_present"] is True
    assert contract["decoded_control_target_present"] is True
    assert contract["decoded_control_target_nonzero"] is False
    assert contract["policy_ranking_claim_safe"] is False
    assert "policy_hand_targets_all_zero" in contract["warnings"]
    assert "policy_action_latent_without_decoded_pose_targets" in contract["blockers"]
    assert contract["tensor_summaries"]["sonic_latent_action"]["shape"] == [1, 2, 2]


def test_policy_action_decoding_contract_reports_bridgeable_sonic_action_chunk() -> None:
    sonic_frame = ([0.1] * 28) + ([0.0] * 36) + ([0.2] * 14)
    contract = session._policy_action_decoding_contract(
        {
            "action_type": "unitree_g1_sonic_latent_action_chunk",
            "action_chunk": [*sonic_frame, *sonic_frame],
            "sonic_latent_action": [[[0.1] * 64, [0.2] * 64]],
            "hand_targets": {
                "left_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
                "right_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            },
        },
        generated_at="now",
    )

    assert contract["status"] == "sonic_action_chunk_available_requires_bridge"
    assert contract["decoded_control_target_nonzero"] is False
    assert contract["bridgeable_sonic_action_chunk"] is True
    assert (
        "policy_action_requires_scene_or_wbc_bridge_for_projected_skeleton"
        in contract["blockers"]
    )
    assert contract["sonic_action_frame_summary"]["frame_count"] == 2
    assert contract["sonic_action_frame_summary"]["hand_control_tail_nonzero_count"] == 28
    assert (
        contract["sonic_action_frame_summary"]["sim2sim_upper_body_slot_nonzero_count"]
        == 56
    )
    assert (
        contract["claim_boundary"][
            "sonic_action_chunk_requires_bridge_before_wam_ranking_claim"
        ]
        is True
    )


def test_bridge_readiness_reports_missing_scene_for_bridgeable_sonic_action_chunk(
    tmp_path: Path,
) -> None:
    readiness = session._write_policy_action_bridge_readiness(
        job=tmp_path,
        extraction_dir=tmp_path / "imported",
        action_contract={
            "latent_action_present": True,
            "decoded_control_target_nonzero": False,
            "bridgeable_sonic_action_chunk": True,
        },
        generated_at="now",
    )

    assert readiness["status"] == "blocked_missing_scene_bridge_for_sonic_action_chunk"
    assert readiness["bridgeable_sonic_action_chunk"] is True
    sim2sim_candidate = next(
        item
        for item in readiness["bridge_candidates"]
        if item["id"] == "unitree_groot_n17_sonic_sim2sim_command"
    )
    assert (
        sim2sim_candidate["requires"][0]
        == "policy_action_40x78_sonic_action_chunk"
    )
    assert sim2sim_candidate["available"] is False
    assert (
        "blocked_missing_scene_faithful_policy_action_projection_bridge"
        in readiness["blockers"]
    )
    assert "blocked_missing_scene_manifest_for_policy_action_bridge" in readiness["blockers"]
    assert readiness["scene_bridge_manifest_path"] is None
    assert readiness["bridge_candidates"][0]["id"] == "isaac_g1_policy_action_projection_bridge"
    assert readiness["bridge_candidates"][0]["available"] is False
    assert (
        readiness["claim_boundary"][
            "mujoco_bridge_is_legacy_action_trace_support_not_isaac_scene_truth"
        ]
        is True
    )


def test_bridge_readiness_does_not_treat_isaac_manifest_as_mujoco_bridge_ready(
    tmp_path: Path,
) -> None:
    (tmp_path / "placement_validation.json").write_text(
        json.dumps({"status": "PASS"}),
        encoding="utf-8",
    )

    readiness = session._write_policy_action_bridge_readiness(
        job=tmp_path,
        extraction_dir=tmp_path / "imported",
        action_contract={
            "latent_action_present": True,
            "decoded_control_target_nonzero": False,
            "bridgeable_sonic_action_chunk": True,
        },
        generated_at="now",
    )

    assert readiness["status"] == "blocked_missing_scene_bridge_for_sonic_action_chunk"
    assert readiness["scene_bridge_manifest_kind"] == "isaac"
    assert readiness["isaac_scene_manifest_path"] == str(tmp_path / "placement_validation.json")
    assert (
        "blocked_missing_isaac_manipulation_pov_geometry_for_action_bridge"
        in readiness["blockers"]
    )
    assert "blocked_no_available_mujoco_sim2sim_manifest_for_legacy_bridge" in readiness[
        "blockers"
    ]
    assert (
        "blocked_missing_scene_manifest_for_policy_action_bridge" not in readiness["blockers"]
    )
    assert readiness["bridge_candidates"][0]["id"] == "isaac_g1_policy_action_projection_bridge"
    assert readiness["bridge_candidates"][0]["implementation_status"] == "implemented"
    assert readiness["bridge_candidates"][0]["available"] is False


def test_bridge_readiness_finds_nested_isaac_scene_context_sidecars(
    tmp_path: Path,
) -> None:
    context_dir = (
        tmp_path
        / "provider_bundle"
        / "provider_runtime"
        / "isaac_scene_context"
    )
    context_dir.mkdir(parents=True)
    placement = context_dir / "placement_validation.json"
    placement.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    geometry = context_dir / "manipulation_pov_geometry.json"
    geometry.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    stance = context_dir / "task_stance_plan.json"
    stance.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")

    readiness = session._write_policy_action_bridge_readiness(
        job=tmp_path,
        extraction_dir=tmp_path / "imported",
        action_contract={
            "latent_action_present": True,
            "decoded_control_target_nonzero": False,
            "bridgeable_sonic_action_chunk": True,
        },
        generated_at="now",
    )

    assert readiness["status"] == "ready_for_isaac_sonic_action_projection_bridge"
    assert readiness["scene_bridge_manifest_kind"] == "isaac"
    assert readiness["isaac_scene_manifest_path"] == str(placement)
    assert readiness["isaac_manipulation_pov_geometry_path"] == str(geometry)
    assert readiness["task_stance_plan_path"] == str(stance)
    assert readiness["blockers"] == []
    assert "blocked_missing_isaac_manipulation_pov_geometry_for_action_bridge" not in readiness["blockers"]
    isaac_candidate = next(
        item
        for item in readiness["bridge_candidates"]
        if item["id"] == "isaac_g1_policy_action_projection_bridge"
    )
    assert "isaac_manipulation_pov_geometry_with_projectable_g1_arm_links" in isaac_candidate["requires"]
    assert isaac_candidate["available"] is True


def _policy_observation(path: Path, frame: Path) -> Path:
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "turn_on_sink_handle",
        "visual_observation": {"camera_frame_path": str(frame)},
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
        "unitree_g1_sonic_state_source": "test_contract_probe",
    }
    path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    return path


def _synthetic_policy_observation(path: Path, frame: Path) -> Path:
    observation = {
        "schema_version": "selected_initial_policy_observation.v1",
        "status": "ready",
        "source_kind": "synthetic_fallback",
        "selection_source_kind": "synthetic_fallback",
        "task_id": "turn_on_sink_handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {
            "available": True,
            "camera_frame_path": str(frame),
            "source_kind": "synthetic_fallback",
            "synthetic_fallback": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
        },
        "provenance": {
            "source_kind": "synthetic_fallback",
            "synthetic_fallback": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
        },
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
        "unitree_g1_sonic_state_source": "synthetic_fallback_contract_probe",
        "claim_boundary": {
            "selected_synthetic_fallback": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
        },
    }
    path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    return path


def _write_reviewable_frame(path: Path, *, size: tuple[int, int] = (640, 480)) -> Path:
    width, height = size
    gradient = np.tile(np.linspace(55, 215, width, dtype=np.uint8), (height, 1))
    frame = np.dstack((gradient, np.roll(gradient, 40, axis=1), np.flipud(gradient)))
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (width // 2 - 70, height // 2 - 50, width // 2 + 70, height // 2 + 50),
        outline=(255, 255, 255),
        width=5,
    )
    draw.ellipse(
        (width // 2 - 22, height // 2 - 22, width // 2 + 22, height // 2 + 22), fill=(235, 80, 50)
    )
    for x in range(0, width, 32):
        draw.line((x, 0, x, height), fill=(20, 20, 20), width=1)
    for y in range(0, height, 32):
        draw.line((0, y, width, y), fill=(235, 235, 235), width=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_episode_consistency_command(tmp_path: Path, *, inverse_consistent: bool) -> Path:
    command = tmp_path / (
        "persistent_consistency_pass.py"
        if inverse_consistent
        else "persistent_consistency_fail.py"
    )
    command.write_text(
        f"""
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_INPUT"]).read_text(encoding="utf-8"))
assert request["schema_version"] == "wam_episode_consistency_request.v1"
assert request["status"] == "ready_for_external_episode_scorer"
assert request["claim_boundary"]["scorer_is_separate_from_wam_execution_and_evaluator"] is True
rollout = request["rollouts"][0]
payload = {{
    "schema_version": "wam_episode_consistency.command.v1",
    "status": "completed",
    "provider": "fake-vlm-episode-consistency",
    "model": "fake-vlm",
    "rollout_checks": [
        {{
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "model_candidate": rollout.get("model_candidate"),
            "forward_consistent": True,
            "inverse_consistent": {inverse_consistent!r},
            "confidence": 0.89,
            "rationale": "Reviewed generated video against the trace summary.",
            "visual_evidence_used": True,
            "action_trace_evidence_used": True,
        }}
    ],
}}
Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    return command


def _write_persistent_postprocess_extraction(root: Path) -> Path:
    extraction_dir = root / "extracted"
    policy_calls_dir = extraction_dir / "policy_calls"
    wam_calls_dir = extraction_dir / "wam_calls"
    generated_dir = extraction_dir / "generated_next_observations"
    step_dir = extraction_dir / "wam_worker_steps" / "step_0001"
    local_materialization = step_dir / "oscar_wam_worker_bundle" / "local_input_materialization"
    local_input = local_materialization / "oscar_input"
    runtime_dir = (
        step_dir
        / "oscar_wam_worker_bundle"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
    )
    runtime_input = runtime_dir / "oscar_input"
    preview_dir = (
        local_materialization
        / "oscar_input_conditioning_visual_review"
        / "generated_rollout_frame_review"
        / "frames"
    )
    policy_calls_dir.mkdir(parents=True)
    wam_calls_dir.mkdir()
    generated_dir.mkdir()
    local_input.mkdir(parents=True)
    runtime_input.mkdir(parents=True)
    preview_dir.mkdir(parents=True)
    _write_reviewable_frame(generated_dir / "wam_generated_next_observation_step_0001.jpg")
    _write_reviewable_frame(runtime_input / "first_frame.png")
    _write_reviewable_frame(preview_dir / "oscar_step_policy_action_conditioning_0001_frame_000.jpg")
    _write_reviewable_frame(preview_dir / "oscar_step_policy_action_conditioning_0001_frame_001.jpg")
    (runtime_input / "rgb_context.mp4").write_bytes(b"mp4")
    (runtime_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"mp4")
    (runtime_input / "wam_auxiliary_observation_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (local_materialization / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "policy_action_to_skeleton_contract": {
                    "status": "stripped_seed_or_target_projected_skeleton_for_policy_action_conditioning",
                    "policy_ranking_claim_safe": False,
                    "blockers": [
                        "policy_action_to_projected_skeleton_decoder_missing_for_ranking_safe_wam"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "wam_rollout_input_manifest.json").write_text(
        json.dumps({"schema_version": "wam_generation_step_input.v1"}),
        encoding="utf-8",
    )
    policy_action = {
        "action_type": "unitree_g1_sonic_latent_action_chunk",
        "action_chunk": [0.1, -0.2, 0.3, 0.0],
        "sonic_latent_action": [[[0.1, -0.2], [0.0, 0.3]]],
        "hand_targets": {
            "left_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            "right_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        },
        "unitree_g1_sonic_control_fields": [
            "left_hand_joints",
            "motion_token",
            "right_hand_joints",
        ],
    }
    (policy_calls_dir / "policy_call_0000.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 0,
                "action": policy_action,
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "wam_generated_next_observations.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "structural_fallback_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wam_calls_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "materialization": {
                    "status": "completed",
                    "source_kind": "video_future_frame",
                    "selected_frame_index": 1,
                    "future_frame_selected": True,
                    "selection_quality_status": "passed_signal_gate",
                    "selected_frame_signal_blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "robot_policy_wam_loop_trace.jsonl").write_text(
        json.dumps({"step_index": 1}) + "\n",
        encoding="utf-8",
    )
    (extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl").write_text(
        json.dumps({"step_index": 1}) + "\n",
        encoding="utf-8",
    )
    return extraction_dir


def _write_dark_frame(path: Path, *, size: tuple[int, int] = (640, 480)) -> Path:
    image = Image.new("RGB", size, (8, 8, 8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] // 2, size[1]), fill=(24, 24, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _write_projected_skeleton_trace(path: Path) -> Path:
    landmarks = []
    for landmark_id, role, u_px, v_px in (
        ("left_wrist_link", "wrist", 176.0, 390.0),
        ("left_hand_link", "hand", 224.0, 330.0),
        ("right_wrist_link", "wrist", 464.0, 390.0),
        ("right_hand_link", "hand", 416.0, 330.0),
    ):
        landmarks.append(
            {
                "landmark_id": landmark_id,
                "link_role": role,
                "image_projection": {
                    "available": True,
                    "u_px": u_px,
                    "v_px": v_px,
                    "depth_m": 0.35,
                },
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "camera": "robot_pov",
                "frame_index": 0,
                "image_size_px": [640, 480],
                "landmarks": landmarks,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_flat_projected_robot_regions_frame(path: Path) -> Path:
    frame = _write_reviewable_frame(path)
    image = Image.open(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y in ((176, 390), (224, 330), (464, 390), (416, 330)):
        draw.rectangle((x - 42, y - 42, x + 42, y + 42), fill=(238, 238, 236))
    image.save(frame)
    return frame


def _write_fake_image_model_remediator(path: Path) -> Path:
    path.write_text(
        """
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

request_path = Path(os.environ["BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_REQUEST_PATH"])
output_dir = Path(os.environ["BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_OUTPUT_DIR"])
response_path = Path(os.environ["BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_RESPONSE_PATH"])
request = json.loads(request_path.read_text(encoding="utf-8"))
width, height = 640, 480
gradient = np.tile(np.linspace(48, 226, width, dtype=np.uint8), (height, 1))
frame = np.dstack((gradient, np.roll(gradient, 72, axis=1), np.flipud(gradient)))
image = Image.fromarray(frame, mode="RGB")
draw = ImageDraw.Draw(image)
draw.rectangle((width // 2 - 82, height // 2 - 58, width // 2 + 82, height // 2 + 58), outline=(255, 255, 255), width=6)
draw.ellipse((width // 2 - 26, height // 2 - 26, width // 2 + 26, height // 2 + 26), fill=(232, 72, 46))
for x in range(0, width, 24):
    draw.line((x, 0, x, height), fill=(24, 24, 24), width=1)
for y in range(0, height, 24):
    draw.line((0, y, width, y), fill=(238, 238, 238), width=1)
enhanced = output_dir / "fake_enhanced.png"
image.save(enhanced)
response_path.write_text(
    json.dumps(
        {
            "status": "completed",
            "provider": "unit_test_fake_image_model",
            "model": request.get("model", "gpt-image-2"),
            "enhanced_image_path": str(enhanced),
        },
        indent=2,
        sort_keys=True,
    )
    + "\\n",
    encoding="utf-8",
)
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_passed_short_sanity_manifest(root: Path, observation_path: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source_qa = root / "source_policy_observation_visual_qa.json"
    report = root / "wam_rollout_visual_quality_report.json"
    contact_sheet = _write_reviewable_frame(root / "wam_rollout_contact_sheet.jpg")
    video_status = root / "video_review_status.json"
    review_video = root / "review.mp4"
    source_qa.write_text(
        json.dumps({"status": "passed_visual_quality_gate"}),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "status": "passed_visual_quality_gate",
                "visual_profile": "review_quality",
                "visual_success": True,
                "profile_contract": {
                    "review_quality_profile": True,
                    "review_quality_minimum_satisfied": True,
                    "smoke_only": False,
                },
            }
        ),
        encoding="utf-8",
    )
    video_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": {
                    "streams": [
                        {
                            "width": 640,
                            "height": 480,
                            "avg_frame_rate": "15/1",
                            "nb_frames": "24",
                        }
                    ],
                    "format": {"duration": "1.6", "size": "1000"},
                },
            }
        ),
        encoding="utf-8",
    )
    review_video.write_bytes(b"mp4")
    manifest = root / "persistent_wam_short_visual_sanity_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION,
                "generated_at": "now",
                "status": "passed_short_visual_sanity",
                "short_visual_sanity_passed": True,
                "policy_observation_path": str(observation_path.resolve()),
                "provider": "runpod",
                "requested_transition_count": 2,
                "requested_loop_step_count": 3,
                "generated_transition_count": 2,
                "visual_profile": "review_quality",
                "source_policy_observation_visual_qa_status": "passed_visual_quality_gate",
                "source_policy_observation_visual_qa_path": str(source_qa),
                "wam_rollout_visual_success": True,
                "wam_rollout_visual_quality_report_path": str(report),
                "wam_rollout_contact_sheet_path": str(contact_sheet),
                "video_review_status_path": str(video_status),
                "review_video_path": str(review_video),
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": {
                    "streams": [
                        {
                            "width": 640,
                            "height": 480,
                            "avg_frame_rate": "15/1",
                            "nb_frames": "24",
                        }
                    ],
                    "format": {"duration": "1.6", "size": "1000"},
                },
                "live_wam_generation_success_count": 2,
                "learned_wam_model_success_count": 2,
                "structural_fallback_used": False,
                "paid_provider": {
                    "provider": "runpod",
                    "used": False,
                    "teardown_status": "not_required_no_paid_provider",
                    "teardown_performed": False,
                    "continuing_spend_from_this_run": False,
                },
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_main_defaults_to_vast_provider(tmp_path: Path, monkeypatch, capsys) -> None:
    observation_path = tmp_path / "observation.json"
    observation_path.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    def fake_vast(**kwargs):
        calls.append("vast")
        return {"status": "completed", "provider": "vast", "received": kwargs}, 0

    def fake_runpod(**kwargs):
        calls.append("runpod")
        return {"status": "blocked", "provider": "runpod", "received": kwargs}, 1

    monkeypatch.setattr(session, "run_persistent_session", fake_vast)
    monkeypatch.setattr(session, "run_persistent_session_runpod", fake_runpod)

    exit_code = session.main(["--policy-observation", str(observation_path)])

    assert exit_code == 0
    assert calls == ["vast"]
    output = json.loads(capsys.readouterr().out)
    assert output["provider"] == "vast"


def test_main_keeps_explicit_runpod_provider(tmp_path: Path, monkeypatch, capsys) -> None:
    observation_path = tmp_path / "observation.json"
    observation_path.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    def fake_vast(**kwargs):
        calls.append("vast")
        return {"status": "blocked", "provider": "vast", "received": kwargs}, 1

    def fake_runpod(**kwargs):
        calls.append("runpod")
        return {"status": "completed", "provider": "runpod", "received": kwargs}, 0

    monkeypatch.setattr(session, "run_persistent_session", fake_vast)
    monkeypatch.setattr(session, "run_persistent_session_runpod", fake_runpod)

    exit_code = session.main(
        ["--policy-observation", str(observation_path), "--provider", "runpod"]
    )

    assert exit_code == 0
    assert calls == ["runpod"]
    output = json.loads(capsys.readouterr().out)
    assert output["provider"] == "runpod"


def _python_heredoc_chunks(script: str) -> list[str]:
    chunks: list[str] = []
    lines = script.splitlines()
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        start = index + 1
        end = start
        while end < len(lines) and lines[end] != "PY":
            end += 1
        chunks.append("\n".join(lines[start:end]) + "\n")
        index = end + 1
    return chunks


def test_persistent_session_bundle_uses_proven_policy_server_rewrite(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["loop_step_count"] == 12
    assert manifest["allow_structural_wam_fallback"] is True
    bundle_path = Path(str(manifest["bundle_path"]))
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        runner = archive.read(
            "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py"
        ).decode()
        runpod_wrapper = archive.read(
            "provider_runtime/run_unitree_groot_n17_sonic_runpod_wrapper.sh"
        ).decode()
        run_script = archive.read(
            "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh"
        ).decode()
        wam_carrier = archive.read("provider_runtime/run_wam_provider_runtime.sh").decode()
        provider_smoke = archive.read(
            "provider_runtime/blueprint_pipeline/unitree_groot_n17_sonic_provider_smoke.py"
        ).decode()
        session_input = json.loads(archive.read("provider_runtime/persistent_session_input.json"))
        runtime_auxiliary = json.loads(
            archive.read(
                "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_manifest.json"
            )
        )

    assert "provider_runtime/run_unitree_groot_n17_sonic_provider_runtime.sh" in names
    assert "provider_runtime/run_unitree_groot_n17_sonic_runpod_wrapper.sh" in names
    assert "provider_runtime/run_wam_provider_runtime.sh" in names
    assert "provider_runtime/unitree_groot_n17_sonic_provider_runner.py" in names
    assert "provider_runtime/blueprint_pipeline/wam_auxiliary_observation.py" in names
    assert "provider_runtime/blueprint_pipeline/oscar_official_release.py" in names
    assert "provider_runtime/policy_input.json" in names
    assert "provider_runtime/input_frame.png" in names
    assert (
        "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_manifest.json"
        in names
    )
    assert (
        "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_claim_boundary.json"
        in names
    )
    assert "provider_instance_reused_for_policy_and_wam_loop" in runner
    assert "bootstrap_venv_policy_server_client_for_persistent_session" in runner
    assert "persistent_policy_worker_command_uses_policy_server_client" in runner
    assert "not self.use_live_wam" in runner
    assert "shlex.quote(str(venv_python))" in runner
    assert "_http_post_json_with_retries" in runner
    assert "loop_step_count = max(1" in runner
    assert "required_wam_transition_count" in runner
    assert "persistent_wam_worker_runtime_stdout_stderr.log" in runner
    assert "persistent_wam_worker_oscar_runtime_timeout" in runner
    assert "persistent_wam_worker_oscar_runtime_started" in runner
    assert "persistent_wam_worker_oscar_runtime_waiting" in runner
    assert "subprocess.Popen" in runner
    assert "start_new_session=True" in runner
    assert "proc.poll()" in runner
    assert "timeout_deadline" in runner
    assert "os.killpg(process_group_id or os.getpgid(proc.pid), signal.SIGTERM)" in runner
    assert "os.killpg(process_group_id or os.getpgid(proc.pid), signal.SIGKILL)" in runner
    assert "process_group_id" in runner
    assert "process_group_terminated" in runner
    assert "process_group_killed" in runner
    assert "stdout_stderr_streamed_to_log" in runner
    assert "_upload_phase_heartbeat(payload)" in runner
    assert "BLUEPRINT_PERSISTENT_SESSION_PHASE_HEARTBEAT_UPLOAD_OK" in runner
    assert "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS" in runner
    assert '"_blueprint_outer_phase_callback": _phase' in session.PERSISTENT_SESSION_RUNNER
    assert "_blueprint_outer_phase_callback" in provider_smoke
    assert "gr00t_model_snapshot_completed" in provider_smoke
    assert "gr00t_policy_server_process_started" in provider_smoke
    assert "BLUEPRINT_GROOT_MODEL_SNAPSHOT_ATTEMPT_FAILED" in provider_smoke
    assert "BLUEPRINT_UNITREE_GROOT_N17_SONIC_MODEL_SNAPSHOT_MAX_WORKERS" in provider_smoke
    assert "BLUEPRINT_UNITREE_GROOT_N17_SONIC_RUN_LOG_HEARTBEAT_SECONDS" in provider_smoke
    assert 'f"{log_path.stem}_running"' in provider_smoke
    assert "log_tail=_tail(log_path)" in provider_smoke
    module_source = Path(str(session.__file__)).read_text(encoding="utf-8")
    assert 'or "wam"' in module_source
    assert (
        "runpod_unitree_groot_sonic_bundle_wrapper_exited_before_runtime_result" in runpod_wrapper
    )
    assert "runpod_unitree_groot_sonic_remote_heartbeat" in runpod_wrapper
    assert (
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_BOOTSTRAP_HEARTBEAT:-true"
        in runpod_wrapper
    )
    assert "run_unitree_groot_n17_sonic_runpod_wrapper.sh" in wam_carrier
    assert "BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR" in wam_carrier
    assert "os.walk(output_dir)" in runpod_wrapper
    assert "dirs[:] = sorted(item for item in dirs if item not in excluded_dirs)" in runpod_wrapper
    assert '"checkpoints"' in runpod_wrapper
    assert "zipfile.is_zipfile(zip_path)" in runpod_wrapper
    assert "invalid_or_empty_runtime_output_zip" in runpod_wrapper
    assert "runpod_runtime_output_zip_creation_failed" in runpod_wrapper
    assert "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS" in runpod_wrapper
    assert "write_unitree_groot_sonic_phase_heartbeat" in runpod_wrapper
    assert "runpod_system_dependency_check_started" in runpod_wrapper
    assert "runpod_entrypoint_subprocess_starting" in runpod_wrapper
    assert "runpod_entrypoint_subprocess_running" in runpod_wrapper
    assert "entrypoint_log_tail" in runpod_wrapper
    assert "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_HEARTBEAT_SECONDS" in runpod_wrapper
    assert "if ! python - <<'PY'" in runpod_wrapper
    assert "BLUEPRINT_RUNPOD_KEEPALIVE_AFTER_SUCCESS" in runpod_wrapper
    assert "BLUEPRINT_RUNPOD_KEEPALIVE_AFTER_SUCCESS_STARTED" in runpod_wrapper
    assert "runpod_keepalive_after_success_status.json" in runpod_wrapper
    assert "blueprint_phase_heartbeat" in run_script
    assert "runpod_entrypoint_dependency_probe_started" in run_script
    assert "runpod_entrypoint_runner_starting" in run_script
    assert 'json.dumps(payload, indent=2, sort_keys=True) + "\\n"' in run_script
    for script_name, script_text in {
        "runpod_wrapper": runpod_wrapper,
        "run_script": run_script,
        "wam_carrier": wam_carrier,
    }.items():
        for index, chunk in enumerate(_python_heredoc_chunks(script_text)):
            compile(chunk, f"<{script_name}:heredoc:{index}>", "exec")
    assert session_input["loop_step_count"] == 12
    assert session_input["use_live_wam"] is False
    assert session_input["allow_structural_wam_fallback"] is True
    assert session_input["wam_auxiliary_observation"]["status"] == "completed"
    assert (
        session_input["initial_observation"]["wam_auxiliary_observation"]["modalities_available"][
            "proprioception"
        ]
        is True
    )
    assert runtime_auxiliary["schema_version"] == "wam_auxiliary_observation_manifest.v1"
    assert runtime_auxiliary["source_image_path"] == "provider_runtime/initial_policy_frame.png"
    assert str(tmp_path) not in json.dumps(runtime_auxiliary)
    assert runtime_auxiliary["claim_boundary"]["collision_truth"] is False
    assert (
        runtime_auxiliary["oscar_conditioning_support"][
            "raw_aux_modalities_consumed_by_public_oscar_entrypoint"
        ]
        is False
    )


def test_persistent_session_bundle_packages_projected_skeleton_trace_for_runtime(
    tmp_path: Path,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    projected_trace = _write_projected_skeleton_trace(
        tmp_path / "g1_projected_skeleton_trace.jsonl"
    )
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "open_refrigerator",
        "target_object_id": "fridge_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {
            "camera_frame_path": str(frame),
            "projected_skeleton_trace_path": str(projected_trace),
        },
        "unitree_g1_sonic_state": {"right_arm": [0.0] * 7},
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps({"observation": observation}), encoding="utf-8")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["semantic_visual_qa_source_paths"]["projected_skeleton_trace"] == str(
        projected_trace.resolve()
    )
    bundle_path = Path(str(manifest["bundle_path"]))
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        session_input = json.loads(archive.read("provider_runtime/persistent_session_input.json"))
        runtime_auxiliary = json.loads(
            archive.read(
                "provider_runtime/wam_auxiliary_observation/wam_auxiliary_observation_manifest.json"
            )
        )
    assert session.RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH in names
    visual = session_input["initial_observation"]["visual_observation"]
    assert (
        visual["projected_skeleton_trace_path"]
        == session.RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
    )
    assert (
        visual["g1_projected_skeleton_trace_jsonl"]
        == session.RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
    )
    action_conditioning = runtime_auxiliary["action_conditioning"]
    assert (
        action_conditioning["projected_skeleton_trace_path"]
        == session.RUNTIME_PROJECTED_SKELETON_TRACE_BUNDLE_PATH
    )
    assert str(tmp_path) not in visual["projected_skeleton_trace_path"]
    assert str(tmp_path) not in visual["g1_projected_skeleton_trace_jsonl"]
    assert str(tmp_path) not in json.dumps(runtime_auxiliary)


def test_synthetic_fallback_cannot_build_live_wam_bundle_without_experimental_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "synthetic.jpg")
    observation_path = _synthetic_policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.delenv(session.SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV, raising=False)

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert (
        "synthetic_fallback_live_or_review_wam_launch_requires_experimental_env"
        in manifest["blockers"]
    )
    gate = manifest["synthetic_fallback_wam_launch_gate"]
    assert gate["synthetic_fallback_initial_observation_used"] is True
    assert gate["experimental_env_enabled"] is False
    assert gate["capture_truth"] is False
    assert gate["geometry_truth"] is False
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert manifest["claim_boundary"]["visually_useful_rollout"] is False
    assert (
        manifest["claim_boundary"]["provider_success_separate_from_visually_useful_rollout"]
        is True
    )
    assert Path(manifest["bundle_path"]).is_file() is False


def test_synthetic_fallback_review_wam_bundle_requires_and_records_experimental_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "synthetic.jpg")
    observation_path = _synthetic_policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv(session.SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV, "true")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=False,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["blockers"] == []
    gate = manifest["synthetic_fallback_wam_launch_gate"]
    assert gate["launch_path_requires_gate"] is True
    assert gate["experimental_env"] == session.SYNTHETIC_FALLBACK_WAM_LAUNCH_EXPERIMENT_ENV
    assert gate["experimental_env_enabled"] is True
    assert gate["capture_truth"] is False
    assert gate["geometry_truth"] is False
    assert manifest["claim_boundary"]["synthetic_fallback_initial_observation_used"] is True
    assert manifest["claim_boundary"]["synthetic_fallback_wam_launch_experiment_enabled"] is True
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert manifest["claim_boundary"]["visually_useful_rollout"] is False
    with zipfile.ZipFile(Path(manifest["bundle_path"])) as archive:
        session_input = json.loads(archive.read("provider_runtime/persistent_session_input.json"))
        policy_input = json.loads(archive.read("provider_runtime/policy_input.json"))
    initial_observation = session_input["initial_observation"]
    assert initial_observation["claim_boundary"]["capture_truth"] is False
    assert initial_observation["claim_boundary"]["geometry_truth"] is False
    assert (
        initial_observation["claim_boundary"][
            "synthetic_fallback_wam_launch_experiment_enabled"
        ]
        is True
    )
    assert policy_input["observation"]["visual_observation"]["capture_truth"] is False
    assert policy_input["observation"]["visual_observation"]["geometry_truth"] is False
    assert (
        session_input["claim_boundary"]["provider_success_separate_from_visually_useful_rollout"]
        is True
    )
    assert session_input["claim_boundary"]["visually_useful_rollout"] is False


def test_persistent_session_runner_phase_heartbeat_helper_is_self_contained(
    tmp_path: Path,
    monkeypatch,
) -> None:
    namespace: dict[str, object] = {
        "__name__": "blueprint_test_persistent_session_runner",
        "__file__": str(tmp_path / "unitree_groot_n17_sonic_wam_persistent_session_runner.py"),
    }
    exec(session.PERSISTENT_SESSION_RUNNER, namespace)
    output_dir = tmp_path / "runtime_output"
    output_path = output_dir / "unitree_groot_n17_sonic_policy_provider_output.json"
    uploads: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b""

    def fake_urlopen(request, timeout: int):  # type: ignore[no-untyped-def]
        uploads["url"] = request.full_url
        uploads["data"] = request.data
        uploads["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(namespace["urllib_request"], "urlopen", fake_urlopen)  # type: ignore[index]
    monkeypatch.setenv("OUTPUT_PUT_URL", "https://upload.example/provider-output.zip")
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("BLUEPRINT_UNITREE_GROOT_N17_SONIC_PROVIDER_OUTPUT", str(output_path))
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_UPLOAD_PHASE_HEARTBEATS", "true")

    namespace["_upload_phase_heartbeat"]({"phase": "bootstrap_policy_server_started"})  # type: ignore[index,operator]

    assert uploads["url"] == "https://upload.example/provider-output.zip"
    assert uploads["timeout"] == 20
    assert output_path.is_file()
    heartbeat = json.loads(output_path.read_text(encoding="utf-8"))
    assert heartbeat["status"] == "running"
    assert heartbeat["runtime_phase"] == "bootstrap_policy_server_started"
    assert zipfile.is_zipfile(tmp_path / "unitree_groot_n17_sonic_provider_phase_heartbeat.zip")


def test_run_persistent_session_imports_reused_worker_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv(
        "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_HEARTBEAT_NO_PROGRESS_SECONDS",
        "123",
    )
    for env_name in session.ALLOWED_MACHINE_ID_ENVS:
        monkeypatch.delenv(env_name, raising=False)
    for env_name in session.EXCLUDED_MACHINE_ID_ENVS:
        monkeypatch.delenv(env_name, raising=False)
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_vast(**kwargs):
        captured.update(kwargs)
        captured["policy_command_env"] = session.os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
        )
        captured["persistent_inner_policy_command_env"] = session.os.environ.get(
            session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV
        )
        captured["vast_inner_policy_command_env"] = session.os.environ.get(
            session.INNER_POLICY_COMMAND_ENV
        )
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr(
                "unitree_groot_n17_sonic_wam_persistent_session_output.json",
                json.dumps(
                    {
                        "schema_version": session.OUTPUT_SCHEMA_VERSION,
                        "status": "completed",
                        "persistent_provider_session_used": True,
                        "provider_instance_reused_for_policy_and_wam_loop": True,
                        "repeated_policy_calls_count": 12,
                        "generated_next_observation_count": 11,
                        "policy_observes_wam_generated_next_observation": True,
                        "wam_evaluator_in_control_loop": True,
                        "live_wam_generation_success_count": 0,
                        "learned_wam_model_success_count": 0,
                        "unitree_groot_n17_sonic_model_executed": True,
                        "unitree_groot_n17_sonic_policy_action_command_ran": True,
                        "unitree_policy_action_command_ran": True,
                        "policy_action_model_command_ran": True,
                        "provider_output_replay_used": False,
                        "blockers": [],
                    }
                ),
            )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.01}

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "run_vast_provider_adapter", fake_vast)

    output, exit_code = session.run_persistent_session(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=12,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
    )

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["persistent_provider_session_used"] is True
    assert output["provider_instance_reused_for_policy_and_wam_loop"] is True
    assert output["repeated_policy_calls_count"] == 12
    assert output["generated_next_observation_count"] == 11
    assert output["policy_observes_wam_generated_next_observation"] is True
    assert output["wam_evaluator_in_control_loop"] is True
    assert output["wam_materialization_summary_path"].endswith(
        "wam_materialization_summary.json"
    )
    assert output["provider_output_replay_used"] is False
    assert captured["provider_bundle_kind"] == "unitree_groot_n17_sonic"
    assert captured["enable_blueprint_bundle"] is True
    assert captured["min_compute_cap"] == 800
    assert captured["heartbeat_no_progress_seconds"] == 123
    assert captured["allowed_machine_ids"] == []
    assert captured["policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND
    assert captured["persistent_inner_policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND
    assert captured["vast_inner_policy_command_env"] == session.DEFAULT_INNER_POLICY_COMMAND


def test_run_persistent_session_blocks_dirty_paid_launch_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    dirty_evidence = {
        **_clean_launch_git_evidence(),
        "dirty": True,
        "dirty_entries_count": 1,
        "dirty_entries": [" M src/blueprint_pipeline/example.py"],
    }
    monkeypatch.setattr(
        session.launch_provenance,
        "git_worktree_evidence",
        lambda: dirty_evidence,
    )

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("paid launch gate should block before provider preparation")

    monkeypatch.setattr(session, "build_persistent_session_provider_bundle", fail_if_called)
    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fail_if_called)
    monkeypatch.setattr(session, "run_vast_provider_adapter", fail_if_called)

    output, exit_code = session.run_persistent_session(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=1,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
    )

    result_path = (
        Path(output["job_dir"]) / "unitree_groot_n17_sonic_vast_persistent_session_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["blockers"] == ["dirty_worktree_paid_launch_blocked"]
    assert output["details"]["provider"] == "vast"
    assert output["details"]["git_evidence"]["dirty"] is True
    assert output["details"]["note"] == session.launch_provenance.DIRTY_WORKTREE_PAID_LAUNCH_NOTE
    assert result["blockers"] == output["blockers"]


def test_runpod_persistent_session_blocks_dirty_paid_launch_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    dirty_evidence = {
        **_clean_launch_git_evidence(),
        "dirty": True,
        "dirty_entries_count": 1,
        "dirty_entries": [" M src/blueprint_pipeline/example.py"],
    }
    monkeypatch.setattr(
        session.launch_provenance,
        "git_worktree_evidence",
        lambda: dirty_evidence,
    )

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("paid launch gate should block before provider preparation")

    monkeypatch.setattr(session, "build_persistent_session_provider_bundle", fail_if_called)
    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fail_if_called)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fail_if_called)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fail_if_called)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=1,
        use_live_wam=False,
        allow_structural_wam_fallback=True,
    )

    result_path = (
        Path(output["job_dir"]) / "unitree_groot_n17_sonic_vast_persistent_session_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["blockers"] == ["dirty_worktree_paid_launch_blocked"]
    assert output["details"]["provider"] == "runpod"
    assert output["details"]["git_evidence"]["dirty"] is True
    assert output["details"]["note"] == session.launch_provenance.DIRTY_WORKTREE_PAID_LAUNCH_NOTE
    assert result["blockers"] == output["blockers"]


def test_postprocess_live_wam_not_task_success_labels_are_consistent(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    extraction_dir = tmp_path / "extracted"
    policy_calls_dir = extraction_dir / "policy_calls"
    wam_calls_dir = extraction_dir / "wam_calls"
    policy_calls_dir.mkdir(parents=True)
    wam_calls_dir.mkdir()
    (policy_calls_dir / "policy_call_0000.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 0,
                "action": {"action": "turn_sink_handle"},
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "wam_generated_next_observations.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "structural_fallback_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wam_calls_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "materialization": {
                    "status": "completed",
                    "source_kind": "video_first_frame",
                    "selected_frame_index": 0,
                    "future_frame_selected": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "robot_policy_wam_loop_trace.jsonl").write_text("", encoding="utf-8")
    (extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    vast_run_dir = tmp_path / "vast-run"
    vast_run_dir.mkdir()

    session._postprocess_imported_persistent_session_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported={
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 2,
            "generated_next_observation_count": 1,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 1,
            "policy_observes_wam_generated_next_observation": True,
            "blockers": [],
        },
        generated_at="now",
        policy_observation_path=tmp_path / "observation.json",
        vast_result={"estimated_cost_usd": 0.01},
        vast_run_dir=vast_run_dir,
    )

    labels = json.loads((job / "failure_labels.json").read_text(encoding="utf-8"))
    assert "live_wam_success_not_task_success_proof" in labels["labels"]
    assert "wam_generation_missing" not in labels["labels"]
    assert "structural_wam_fallback_only" not in labels["labels"]

    judge = json.loads(
        (job / "manipulation_success_evaluator_results.json").read_text(encoding="utf-8")
    )
    assert judge["answer"] == "not_proven"
    assert judge["question"] == "Did the requested manipulation succeed?"
    assert judge["did_target_manipulation_succeed"] is False
    assert judge["manipulation_success_proven"] is False
    assert "sink" not in json.dumps(judge).lower()
    assert judge["live_wam_generation_success_count"] == 1
    assert judge["structural_fallback_used"] is False
    assert "live learned WAM generations" in judge["reason"]
    assert "structural WAM fallback only" not in judge["reason"]
    materialization = json.loads(
        (job / "wam_materialization_summary.json").read_text(encoding="utf-8")
    )
    assert materialization["source_kind_counts"] == {"video_first_frame": 1}
    assert materialization["video_first_frame_materialization_count"] == 1
    assert materialization["materialized_future_frame_count"] == 0
    assert materialization["future_frame_quality_status"] == "failed"
    assert "wam_generated_next_observation_used_video_first_frame_fallback" in materialization[
        "future_frame_quality_blockers"
    ]
    visual_report = json.loads(
        (job / "wam_rollout_visual_quality_report.json").read_text(encoding="utf-8")
    )
    assert visual_report["status"] == "failed_visual_quality_gate"
    assert visual_report["visual_success"] is False
    assert "wam_generated_next_observation_used_video_first_frame_fallback" in visual_report[
        "blockers"
    ]
    assert (
        visual_report["materialization_quality"][
            "future_frame_materialization_required_for_visual_success"
        ]
        if "future_frame_materialization_required_for_visual_success"
        in visual_report.get("materialization_quality", {})
        else visual_report["claim_boundary"][
            "future_frame_materialization_required_for_visual_success"
        ]
    )
    claim_boundary = json.loads((job / "claim_boundary.json").read_text(encoding="utf-8"))
    assert (
        claim_boundary["video_first_frame_materialization_is_not_future_rollout_quality_proof"]
        is True
    )


def test_postprocess_degraded_future_frames_fail_materialization_quality(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    source_frame = _write_reviewable_frame(
        job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    )
    observation_path = _policy_observation(tmp_path / "observation.json", source_frame)
    extraction_dir = tmp_path / "extracted"
    policy_calls_dir = extraction_dir / "policy_calls"
    wam_calls_dir = extraction_dir / "wam_calls"
    generated_dir = extraction_dir / "generated_next_observations"
    step_dir = extraction_dir / "wam_worker_steps" / "step_0001"
    local_materialization = step_dir / "oscar_wam_worker_bundle" / "local_input_materialization"
    runtime_dir = (
        step_dir
        / "oscar_wam_worker_bundle"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
    )
    runtime_input = runtime_dir / "oscar_input"
    preview_dir = (
        local_materialization
        / "oscar_input_conditioning_visual_review"
        / "generated_rollout_frame_review"
        / "frames"
    )
    policy_calls_dir.mkdir(parents=True)
    wam_calls_dir.mkdir()
    generated_dir.mkdir()
    runtime_input.mkdir(parents=True)
    preview_dir.mkdir(parents=True)
    _write_reviewable_frame(generated_dir / "wam_generated_next_observation_step_0001.jpg")
    _write_reviewable_frame(runtime_input / "first_frame.png")
    _write_reviewable_frame(
        preview_dir / "oscar_step_policy_action_conditioning_0001_frame_000.jpg"
    )
    _write_reviewable_frame(
        preview_dir / "oscar_step_policy_action_conditioning_0001_frame_001.jpg"
    )
    (runtime_input / "rgb_context.mp4").write_bytes(b"mp4")
    (runtime_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"mp4")
    (runtime_input / "wam_auxiliary_observation_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (local_materialization / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "policy_action_to_skeleton_contract": {
                    "status": (
                        "stripped_seed_or_target_projected_skeleton_for_policy_action_conditioning"
                    ),
                    "policy_ranking_claim_safe": False,
                    "blockers": [
                        "policy_action_to_projected_skeleton_decoder_missing_for_ranking_safe_wam"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "wam_rollout_input_manifest.json").write_text(
        json.dumps({"schema_version": "wam_generation_step_input.v1"}),
        encoding="utf-8",
    )
    policy_action = {
        "action_type": "unitree_g1_sonic_latent_action_chunk",
        "action_chunk": [0.1, -0.2, 0.3, 0.0],
        "sonic_latent_action": [[[0.1, -0.2], [0.0, 0.3]]],
        "hand_targets": {
            "left_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            "right_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        },
        "unitree_g1_sonic_control_fields": [
            "left_hand_joints",
            "motion_token",
            "right_hand_joints",
        ],
    }
    (policy_calls_dir / "policy_call_0000.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 0,
                "action": policy_action,
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "wam_generated_next_observations.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "structural_fallback_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wam_calls_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "materialization": {
                    "status": "completed",
                    "source_kind": "video_future_frame",
                    "selected_frame_index": 1,
                    "future_frame_selected": True,
                    "selection_quality_status": "degraded_visual_signal",
                    "selected_frame_signal_blockers": [
                        "next_observation_candidate_low_scene_structure"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "robot_policy_wam_loop_trace.jsonl").write_text("", encoding="utf-8")
    (extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    vast_run_dir = tmp_path / "vast-run"
    vast_run_dir.mkdir()

    session._postprocess_imported_persistent_session_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported={
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 2,
            "generated_next_observation_count": 1,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 1,
            "policy_observes_wam_generated_next_observation": True,
            "blockers": [],
        },
        generated_at="now",
        policy_observation_path=observation_path,
        vast_result={"estimated_cost_usd": 0.01},
        vast_run_dir=vast_run_dir,
    )

    materialization = json.loads(
        (job / "wam_materialization_summary.json").read_text(encoding="utf-8")
    )
    assert materialization["source_kind_counts"] == {"video_future_frame": 1}
    assert materialization["materialized_future_frame_count"] == 1
    assert materialization["video_first_frame_materialization_count"] == 0
    assert materialization["degraded_future_frame_count"] == 1
    assert materialization["future_frame_quality_status"] == "failed"
    assert materialization["selection_quality_status_counts"] == {
        "degraded_visual_signal": 1
    }
    assert materialization["selected_frame_signal_blocker_counts"] == {
        "next_observation_candidate_low_scene_structure": 1
    }
    assert "wam_generated_next_observation_future_frame_degraded_visual_signal" in (
        materialization["future_frame_quality_blockers"]
    )

    visual_report = json.loads(
        (job / "wam_rollout_visual_quality_report.json").read_text(encoding="utf-8")
    )
    assert visual_report["status"] == "failed_visual_quality_gate"
    assert visual_report["visual_success"] is False
    assert "wam_generated_next_observation_future_frame_degraded_visual_signal" in (
        visual_report["blockers"]
    )
    assert visual_report["materialization_quality"]["degraded_future_frame_count"] == 1
    assert visual_report["materialization_quality"]["selection_quality_status_counts"] == {
        "degraded_visual_signal": 1
    }
    assert (
        visual_report["claim_boundary"][
            "degraded_future_frame_materialization_is_not_visual_rollout_quality_proof"
        ]
        is True
    )

    claim_boundary = json.loads((job / "claim_boundary.json").read_text(encoding="utf-8"))
    assert claim_boundary["degraded_future_frame_count"] == 1
    assert (
        claim_boundary[
            "degraded_future_frame_materialization_is_not_visual_rollout_quality_proof"
        ]
        is True
    )


def test_postprocess_high_risk_wam_input_contract_fails_visual_quality(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    source_frame = _write_reviewable_frame(
        job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    )
    observation_path = _policy_observation(tmp_path / "observation.json", source_frame)
    extraction_dir = tmp_path / "extracted"
    policy_calls_dir = extraction_dir / "policy_calls"
    wam_calls_dir = extraction_dir / "wam_calls"
    generated_dir = extraction_dir / "generated_next_observations"
    step_dir = extraction_dir / "wam_worker_steps" / "step_0001"
    local_materialization = step_dir / "oscar_wam_worker_bundle" / "local_input_materialization"
    runtime_dir = (
        step_dir
        / "oscar_wam_worker_bundle"
        / "oscar_wam_provider_bundle"
        / "provider_runtime"
    )
    runtime_input = runtime_dir / "oscar_input"
    preview_dir = (
        local_materialization
        / "oscar_input_conditioning_visual_review"
        / "generated_rollout_frame_review"
        / "frames"
    )
    policy_calls_dir.mkdir(parents=True)
    wam_calls_dir.mkdir()
    generated_dir.mkdir()
    runtime_input.mkdir(parents=True)
    preview_dir.mkdir(parents=True)
    _write_reviewable_frame(generated_dir / "wam_generated_next_observation_step_0001.jpg")
    _write_reviewable_frame(runtime_input / "first_frame.png")
    _write_reviewable_frame(
        preview_dir / "oscar_step_policy_action_conditioning_0001_frame_000.jpg"
    )
    _write_reviewable_frame(
        preview_dir / "oscar_step_policy_action_conditioning_0001_frame_001.jpg"
    )
    (runtime_input / "rgb_context.mp4").write_bytes(b"mp4")
    (runtime_input / "blueprint_proxy_skeleton_conditioning.mp4").write_bytes(b"mp4")
    (runtime_input / "wam_auxiliary_observation_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (local_materialization / "oscar_wam_input_package_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "blueprint_oscar_wam_input_package.v1",
                "policy_action_to_skeleton_contract": {
                    "status": (
                        "stripped_seed_or_target_projected_skeleton_for_policy_action_conditioning"
                    ),
                    "policy_ranking_claim_safe": False,
                    "blockers": [
                        "policy_action_to_projected_skeleton_decoder_missing_for_ranking_safe_wam"
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "wam_rollout_input_manifest.json").write_text(
        json.dumps({"schema_version": "wam_generation_step_input.v1"}),
        encoding="utf-8",
    )
    policy_action = {
        "action_type": "unitree_g1_sonic_latent_action_chunk",
        "action_chunk": [0.1, -0.2, 0.3, 0.0],
        "sonic_latent_action": [[[0.1, -0.2], [0.0, 0.3]]],
        "hand_targets": {
            "left_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            "right_hand_joints": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        },
        "unitree_g1_sonic_control_fields": [
            "left_hand_joints",
            "motion_token",
            "right_hand_joints",
        ],
    }
    (policy_calls_dir / "policy_call_0000.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 0,
                "action": policy_action,
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "wam_generated_next_observations.jsonl").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "structural_fallback_used": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (wam_calls_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "materialization": {
                    "status": "completed",
                    "source_kind": "video_future_frame",
                    "selected_frame_index": 1,
                    "future_frame_selected": True,
                    "selection_quality_status": "passed_signal_gate",
                    "selected_frame_signal_blockers": [],
                },
                "live_wam_payload_redacted": {
                    "input_package": {
                        "claim_boundary": {
                            "policy_action_conditioning_proxy_video_used": True
                        },
                        "rgb_video": {"rgb_context_mode": "single_frame_repeat"},
                        "projected_skeleton_trace": {"used_for_conditioning": False},
                        "oscar_input_contract_diagnostic": {
                            "schema_version": "oscar_wam_runtime_input_contract_diagnostic.v1",
                            "status": "warning_high_risk",
                            "skeleton_video": {
                                "conditioning_mode": (
                                    "unitree_sonic_policy_action_proxy_over_scene_frame"
                                ),
                                "policy_action_proxy_used": True,
                            },
                            "projected_skeleton_trace": {"used_for_conditioning": False},
                            "rgb_context": {"rgb_context_mode": "single_frame_repeat"},
                            "warnings": [
                                "oscar_contract_policy_action_proxy_conditioning_without_projected_skeleton"
                            ],
                            "autoregressive_risk_flags": [
                                "policy_action_proxy_without_projected_skeleton_autoregressive_risk"
                            ],
                            "high_risk_flags": [
                                "policy_action_proxy_without_projected_skeleton_high_risk"
                            ],
                            "ranking_risk_flags": [
                                "projected_skeleton_not_policy_derived_action_ranking_risk"
                            ],
                            "autoregressive_risk_level": "high",
                            "policy_ranking_claim_safe": False,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (extraction_dir / "robot_policy_wam_loop_trace.jsonl").write_text("", encoding="utf-8")
    (extraction_dir / "robot_policy_wam_side_by_side_trace.jsonl").write_text(
        "",
        encoding="utf-8",
    )
    vast_run_dir = tmp_path / "vast-run"
    vast_run_dir.mkdir()

    postprocess = session._postprocess_imported_persistent_session_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported={
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 2,
            "generated_next_observation_count": 1,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 1,
            "policy_observes_wam_generated_next_observation": True,
            "blockers": [],
        },
        generated_at="now",
        policy_observation_path=observation_path,
        vast_result={"estimated_cost_usd": 0.01},
        vast_run_dir=vast_run_dir,
    )

    input_contract = json.loads(
        (job / "wam_input_contract_summary.json").read_text(encoding="utf-8")
    )
    action_contract = json.loads(
        (job / "policy_action_decoding_contract.json").read_text(encoding="utf-8")
    )
    bridge_readiness = json.loads(
        (job / "policy_action_bridge_readiness.json").read_text(encoding="utf-8")
    )
    assert postprocess["wam_input_contract_summary"].endswith(
        "wam_input_contract_summary.json"
    )
    assert postprocess["policy_action_decoding_contract"].endswith(
        "policy_action_decoding_contract.json"
    )
    assert postprocess["policy_action_bridge_readiness"].endswith(
        "policy_action_bridge_readiness.json"
    )
    assert action_contract["status"] == "blocked_latent_action_without_pose_decoder"
    assert action_contract["latent_action_present"] is True
    assert action_contract["decoded_control_target_nonzero"] is False
    assert action_contract["tensor_summaries"]["sonic_latent_action"]["shape"] == [1, 2, 2]
    assert bridge_readiness["status"] == "blocked_missing_scene_bridge_for_latent_action"
    assert (
        "blocked_missing_scene_faithful_policy_action_projection_bridge"
        in bridge_readiness["blockers"]
    )
    assert input_contract["status"] == "warning_high_risk"
    assert input_contract["high_risk_input_contract_count"] == 1
    assert input_contract["policy_ranking_risk_input_contract_count"] == 1
    assert input_contract["policy_ranking_claim_safe"] is False
    assert input_contract["contract_ranking_risk_flag_counts"] == {
        "projected_skeleton_not_policy_derived_action_ranking_risk": 1
    }
    assert input_contract["policy_action_proxy_conditioning_count"] == 1
    assert input_contract["projected_skeleton_conditioning_count"] == 0
    assert input_contract["contract_status_counts"] == {"warning_high_risk": 1}
    assert input_contract["rgb_context_mode_counts"] == {"single_frame_repeat": 1}
    assert (
        "wam_input_contract_high_risk_policy_action_proxy_without_projected_skeleton"
        in input_contract["blockers"]
    )

    input_review = json.loads(
        (job / "wam_input_review_manifest.json").read_text(encoding="utf-8")
    )
    assert postprocess["wam_input_review_manifest"].endswith("wam_input_review_manifest.json")
    assert postprocess["wam_input_review_contact_sheet"].endswith(
        "wam_input_review_contact_sheet.jpg"
    )
    assert Path(postprocess["wam_input_review_contact_sheet"]).is_file()
    assert input_review["status"] == "completed"
    assert input_review["wam_step_count"] == 1
    assert input_review["input_media_row_count"] == 1
    assert input_review["rows"][0]["first_frame_path"].endswith("first_frame.png")
    assert input_review["rows"][0]["rgb_context_video_path"].endswith("rgb_context.mp4")
    assert input_review["rows"][0]["action_conditioning_video_path"].endswith(
        "blueprint_proxy_skeleton_conditioning.mp4"
    )
    assert input_review["rows"][0]["policy_ranking_claim_safe"] is False
    assert input_review["contact_sheet"]["status"] == "completed"

    visual_report = json.loads(
        (job / "wam_rollout_visual_quality_report.json").read_text(encoding="utf-8")
    )
    assert visual_report["status"] == "failed_visual_quality_gate"
    assert visual_report["visual_success"] is False
    assert visual_report["frame_visual_success_before_contract_gate"] is True
    assert visual_report["input_contract_gate_failed"] is True
    assert visual_report["materialization_gate_failed"] is False
    assert visual_report["overall_gate_success"] is False
    assert (
        "wam_input_contract_high_risk_policy_action_proxy_without_projected_skeleton"
        in visual_report["blockers"]
    )
    assert visual_report["input_contract_quality"]["high_risk_input_contract_count"] == 1
    assert (
        visual_report["claim_boundary"][
            "high_risk_input_contract_is_not_visual_rollout_quality_proof"
        ]
        is True
    )

    labels = json.loads((job / "failure_labels.json").read_text(encoding="utf-8"))
    assert "wam_input_contract_high_risk" in labels["labels"]


def test_postprocess_labels_nominal_projected_skeleton_risk_without_proxy(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    source_frame = _write_reviewable_frame(
        job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    )
    observation_path = _policy_observation(tmp_path / "observation.json", source_frame)
    extraction_dir = _write_persistent_postprocess_extraction(tmp_path)
    (extraction_dir / "wam_calls" / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "materialization": {
                    "status": "completed",
                    "source_kind": "video_future_frame",
                    "selected_frame_index": 1,
                    "future_frame_selected": True,
                    "selection_quality_status": "passed_signal_gate",
                    "selected_frame_signal_blockers": [],
                },
                "live_wam_payload_redacted": {
                    "input_package": {
                        "skeleton_video": {"conditioning_mode": "projected_g1_skeleton"},
                        "rgb_video": {"rgb_context_mode": "single_frame_repeat"},
                        "projected_skeleton_trace": {"used_for_conditioning": True},
                        "oscar_input_contract_diagnostic": {
                            "schema_version": "oscar_wam_runtime_input_contract_diagnostic.v1",
                            "status": "warning_high_risk",
                            "skeleton_video": {
                                "conditioning_mode": "projected_g1_skeleton",
                                "policy_action_proxy_used": False,
                            },
                            "projected_skeleton_trace": {"used_for_conditioning": True},
                            "rgb_context": {"rgb_context_mode": "single_frame_repeat"},
                            "warnings": [
                                "oscar_contract_projected_skeleton_nominal_action_projection",
                                "oscar_contract_policy_action_to_skeleton_not_ranking_safe",
                            ],
                            "autoregressive_risk_flags": [
                                "rgb_context_single_frame_repeat_autoregressive_risk"
                            ],
                            "high_risk_flags": [
                                "projected_skeleton_nominal_action_projection_high_risk"
                            ],
                            "ranking_risk_flags": [
                                "projected_skeleton_nominal_action_projection_without_scene_or_wbc_bridge",
                                "policy_action_to_skeleton_contract_not_ranking_safe",
                            ],
                            "autoregressive_risk_level": "high",
                            "policy_ranking_claim_safe": False,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    vast_run_dir = tmp_path / "vast-run"
    vast_run_dir.mkdir()

    session._postprocess_imported_persistent_session_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported={
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 2,
            "generated_next_observation_count": 1,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 1,
            "policy_observes_wam_generated_next_observation": True,
            "blockers": [],
        },
        generated_at="now",
        policy_observation_path=observation_path,
        vast_result={"estimated_cost_usd": 0.01},
        vast_run_dir=vast_run_dir,
    )

    input_contract = json.loads(
        (job / "wam_input_contract_summary.json").read_text(encoding="utf-8")
    )
    assert input_contract["projected_skeleton_conditioning_count"] == 1
    assert input_contract["policy_action_proxy_conditioning_count"] == 0
    assert (
        "wam_input_contract_high_risk_projected_skeleton_nominal_action_projection"
        in input_contract["blockers"]
    )
    assert (
        "wam_input_contract_high_risk_policy_action_proxy_without_projected_skeleton"
        not in input_contract["blockers"]
    )
    assert "wam_input_contract_policy_ranking_claim_not_safe" in input_contract["blockers"]


def test_postprocess_preserves_scene_faithful_isaac_bridge_input_contract(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    source_frame = _write_reviewable_frame(
        job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    )
    observation_path = _policy_observation(tmp_path / "observation.json", source_frame)
    extraction_dir = _write_persistent_postprocess_extraction(tmp_path)
    (extraction_dir / "wam_calls" / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "step_index": 1,
                "materialization": {
                    "status": "completed",
                    "source_kind": "video_future_frame",
                    "selected_frame_index": 1,
                    "future_frame_selected": True,
                    "selection_quality_status": "passed_signal_gate",
                    "selected_frame_signal_blockers": [],
                },
                "live_wam_payload_redacted": {
                    "input_package": {
                        "skeleton_video": {"conditioning_mode": "projected_g1_skeleton"},
                        "rgb_video": {
                            "rgb_context_mode": (
                                "omitted_first_frame_plus_skeleton_public_contract"
                            )
                        },
                        "projected_skeleton_trace": {"used_for_conditioning": True},
                        "oscar_input_contract_diagnostic": {
                            "schema_version": "oscar_wam_runtime_input_contract_diagnostic.v1",
                            "status": "ready",
                            "skeleton_video": {
                                "conditioning_mode": "projected_g1_skeleton",
                                "policy_action_proxy_used": False,
                            },
                            "projected_skeleton_trace": {
                                "used_for_conditioning": True,
                                "policy_derived_action_conditioning": True,
                                "official_wbc_or_sim_bridge_used": False,
                                "blueprint_simulator_only_isaac_action_projection_bridge_used": True,
                                "scene_faithful_isaac_policy_action_projection_bridge_used": True,
                                "policy_action_bridge_safe_for_sim_ranking": True,
                            },
                            "rgb_context": {
                                "rgb_context_mode": (
                                    "omitted_first_frame_plus_skeleton_public_contract"
                                )
                            },
                            "warnings": [],
                            "autoregressive_risk_flags": [],
                            "high_risk_flags": [],
                            "ranking_risk_flags": [],
                            "autoregressive_risk_level": "low",
                            "policy_ranking_risk_level": "low",
                            "policy_ranking_claim_safe": True,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    vast_run_dir = tmp_path / "vast-run"
    vast_run_dir.mkdir()

    postprocess = session._postprocess_imported_persistent_session_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported={
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 2,
            "generated_next_observation_count": 1,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 1,
            "policy_observes_wam_generated_next_observation": True,
            "blockers": [],
        },
        generated_at="now",
        policy_observation_path=observation_path,
        vast_result={"estimated_cost_usd": 0.01},
        vast_run_dir=vast_run_dir,
    )

    input_contract = json.loads(
        (job / "wam_input_contract_summary.json").read_text(encoding="utf-8")
    )
    assert input_contract["status"] == "completed"
    assert input_contract["blockers"] == []
    assert input_contract["high_risk_input_contract_count"] == 0
    assert input_contract["policy_ranking_risk_input_contract_count"] == 0
    assert input_contract["policy_ranking_claim_safe"] is True
    assert (
        input_contract["scene_faithful_isaac_policy_action_projection_bridge_count"]
        == 1
    )
    assert input_contract["policy_action_bridge_safe_for_sim_ranking_count"] == 1
    assert input_contract["claim_boundary"]["scene_or_task_specific_pixels_used"] is True

    materialization = json.loads(
        (job / "wam_materialization_summary.json").read_text(encoding="utf-8")
    )
    assert materialization["claim_boundary"]["scene_or_task_specific_pixels_used"] is True

    visual_report = json.loads(
        (job / "wam_rollout_visual_quality_report.json").read_text(encoding="utf-8")
    )
    assert "wam_input_contract_policy_ranking_claim_not_safe" not in visual_report["blockers"]

    calibration = json.loads(
        (job / "rank_fidelity_calibration_requirement.json").read_text(encoding="utf-8")
    )
    anchor_request = json.loads(
        (job / "rank_fidelity_calibration_anchor_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert postprocess["rank_fidelity_calibration_requirement"].endswith(
        "rank_fidelity_calibration_requirement.json"
    )
    assert postprocess["rank_fidelity_calibration_anchor_request"].endswith(
        "rank_fidelity_calibration_anchor_request.json"
    )
    assert postprocess["rank_fidelity_result_proven"] is False
    assert calibration["status"] == "blocked_missing_calibration_anchors"
    assert calibration["calibration_anchor_request"].endswith(
        "rank_fidelity_calibration_anchor_request.json"
    )
    assert calibration["requested_anchor_count"] == 4
    assert calibration["candidate_prediction_record_count"] == 1
    assert calibration["candidate_prediction_records"][0]["actual_status"] == (
        "needs_accepted_anchor_outcome"
    )
    assert calibration["minimum_accepted_anchor_count"] == 4
    assert calibration["minimum_policy_group_count"] == 2
    assert calibration["rank_fidelity_result_proven"] is False
    assert "missing_accepted_calibration_anchor_outcomes" in calibration["blockers"]
    assert anchor_request["status"] == "blocked_awaiting_accepted_anchor_outcomes"
    assert anchor_request["requested_anchor_count"] == 4
    assert anchor_request["accepted_anchor_count"] == 0
    assert anchor_request["anchor_request_rows"][0]["prediction_status"] == "available"
    assert (
        anchor_request["anchor_request_rows"][0]["exact_join_keys_status"]
        == "ready_for_actual_join"
    )
    assert anchor_request["anchor_request_rows"][1]["prediction_status"] == (
        "needs_matching_prediction_record"
    )
    assert (
        anchor_request["claim_boundary"]["anchor_request_rows_are_not_accepted_anchors"]
        is True
    )
    claim_boundary = json.loads((job / "claim_boundary.json").read_text(encoding="utf-8"))
    assert claim_boundary["rank_fidelity_calibration_required"] is True
    assert claim_boundary["rank_fidelity_calibration_anchor_request"].endswith(
        "rank_fidelity_calibration_anchor_request.json"
    )
    assert claim_boundary["visual_review_ranking_is_not_real_world_rank_fidelity"] is True


def test_postprocess_episode_consistency_failure_is_reliability_label_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "job"
    source_frame = _write_reviewable_frame(
        job / "provider_bundle" / "provider_runtime" / "initial_policy_frame.png"
    )
    observation_path = _policy_observation(tmp_path / "observation.json", source_frame)
    extraction_dir = _write_persistent_postprocess_extraction(tmp_path)
    vast_run_dir = tmp_path / "vast-run"
    vast_run_dir.mkdir()
    command = _write_episode_consistency_command(tmp_path, inverse_consistent=False)

    def fake_review_video(**kwargs):
        review_video = job / "review_video" / "persistent_policy_wam_live_rollout_review.mp4"
        review_video.parent.mkdir(parents=True, exist_ok=True)
        review_video.write_bytes(b"fake mp4")
        status = {
            "status": "completed",
            "review_video_path": str(review_video),
            "ffprobe_command_ran": False,
            "blockers": [],
        }
        (job / "video_review_status.json").write_text(
            json.dumps(status),
            encoding="utf-8",
        )
        return status

    def fake_visual_quality(**kwargs):
        report = {
            "schema_version": "persistent_wam_visual_quality_report.v1",
            "generated_at": kwargs["generated_at"],
            "status": "passed_visual_quality_gate",
            "visual_success": True,
            "review_video_path": str(kwargs["review_video_path"]),
            "blockers": [],
            "claim_boundary": {
                "visual_success_does_not_prove_task_success": True,
            },
        }
        (job / "source_policy_observation_visual_qa.json").write_text(
            json.dumps({"status": "passed_visual_quality_gate"}),
            encoding="utf-8",
        )
        (job / "wam_rollout_visual_quality_report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(session, "_write_review_video", fake_review_video)
    monkeypatch.setattr(
        session,
        "write_persistent_wam_visual_quality_artifacts",
        fake_visual_quality,
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    monkeypatch.setenv(
        "BLUEPRINT_WAM_EPISODE_CONSISTENCY_COMMAND",
        f"{sys.executable} {command}",
    )

    postprocess = session._postprocess_imported_persistent_session_artifacts(
        job=job,
        extraction_dir=extraction_dir,
        imported={
            "status": "completed",
            "persistent_provider_session_used": True,
            "provider_instance_reused_for_policy_and_wam_loop": True,
            "repeated_policy_calls_count": 2,
            "generated_next_observation_count": 1,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 1,
            "manipulation_success_evaluator_result": "success",
            "policy_observes_wam_generated_next_observation": True,
            "blockers": [],
        },
        generated_at="now",
        policy_observation_path=observation_path,
        vast_result={"estimated_cost_usd": 0.01},
        vast_run_dir=vast_run_dir,
    )

    assert postprocess["forward_inverse_consistency_proven"] is False
    assert postprocess["external_episode_consistency_scorer_ran"] is True
    assert postprocess["wam_episode_consistency_early_termination_recommended"] is True
    assert "wam_consistency_inverse_not_proven" in postprocess[
        "wam_episode_consistency_blockers"
    ]
    assert "wam_consistency_inverse_not_proven" in postprocess["blockers"]

    request = json.loads(
        Path(postprocess["wam_episode_consistency_request"]).read_text(encoding="utf-8")
    )
    assert request["status"] == "ready_for_external_episode_scorer"
    assert request["generated_rollout_visually_useful_for_success_review"] is True
    consistency = json.loads(
        Path(postprocess["wam_consistency_checks"]).read_text(encoding="utf-8")
    )
    assert consistency["forward_inverse_consistency_proven"] is False
    assert consistency["external_episode_consistency_scorer_id"] == (
        "fake-vlm-episode-consistency"
    )
    assert consistency["claim_boundary"][
        "forward_inverse_consistency_does_not_prove_task_success"
    ] is True

    judge = json.loads(
        (job / "manipulation_success_evaluator_results.json").read_text(encoding="utf-8")
    )
    assert judge["manipulation_success_proven"] is True
    labels = json.loads((job / "failure_labels.json").read_text(encoding="utf-8"))
    assert labels["labels"] == [
        "wam_episode_consistency_early_termination_recommended",
        "forward_inverse_consistency_not_proven",
    ]
    assert labels["task_success_not_failed_by_consistency_label"] is True
    assert "task_success_not_proven" not in labels["labels"]

    claim_boundary = json.loads((job / "claim_boundary.json").read_text(encoding="utf-8"))
    assert claim_boundary["forward_inverse_consistency_proven"] is False
    assert claim_boundary["success_proof_completed"] is True
    assert (
        claim_boundary["forward_inverse_consistency_does_not_prove_task_success"]
        is True
    )


def test_copy_or_extract_wam_frame_prefers_usable_future_video_frame(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "rollout.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    assert writer.isOpened()
    seed_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    seed_frame[:, ::2] = 220
    usable_future = np.zeros((24, 32, 3), dtype=np.uint8)
    usable_future[::2, :] = (235, 235, 235)
    usable_future[:, ::4] = (24, 180, 240)
    collapsed_late = np.full((24, 32, 3), 8, dtype=np.uint8)
    for frame in (seed_frame, usable_future, collapsed_late):
        writer.write(frame)
    writer.release()

    target_frame = tmp_path / "generated" / "next.jpg"
    namespace = _persistent_runner_namespace()
    copy_or_extract = namespace["_copy_or_extract_wam_frame"]
    materialization = copy_or_extract(
        {"rollouts": [{"generated_video_path": str(video)}]},
        target_frame,
    )

    assert materialization["status"] == "completed"
    assert materialization["source_kind"] == "video_future_frame"
    assert materialization["selected_frame_index"] == 1
    assert materialization["future_frame_selected"] is True
    assert (
        materialization["frame_selection_policy"]
        == "prefer_signal_valid_else_earliest_decodable_future_frame"
    )
    assert materialization["selection_quality_status"] == "passed_signal_gate"
    assert materialization["claim_boundary"]["scene_or_task_specific_pixels_used"] is False
    assert target_frame.is_file()
    selection_path = Path(materialization["selection_manifest_path"])
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["status"] == "completed"
    assert selection["selected_frame_index"] == 1


def test_copy_or_extract_wam_frame_uses_warned_future_frame_before_seed_fallback(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "rollout.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (128, 96))
    assert writer.isOpened()
    seed_frame = np.zeros((96, 128, 3), dtype=np.uint8)
    seed_frame[:, :64] = (225, 225, 225)
    seed_frame[:, 64:] = (25, 80, 130)
    smooth_gradient = np.tile(np.linspace(45, 130, 128, dtype=np.uint8), (96, 1))
    weak_future = np.dstack((smooth_gradient, smooth_gradient, smooth_gradient))
    for frame in (seed_frame, weak_future):
        writer.write(frame)
    writer.release()

    target_frame = tmp_path / "generated" / "next.jpg"
    namespace = _persistent_runner_namespace()
    copy_or_extract = namespace["_copy_or_extract_wam_frame"]
    materialization = copy_or_extract(
        {"rollouts": [{"generated_video_path": str(video)}]},
        target_frame,
    )

    assert materialization["status"] == "completed"
    assert materialization["source_kind"] == "video_future_frame"
    assert materialization["selected_frame_index"] == 1
    assert materialization["future_frame_selected"] is True
    assert materialization["selection_quality_status"] == "degraded_visual_signal"
    assert "next_observation_candidate_low_scene_structure" in materialization[
        "selected_frame_signal_blockers"
    ]
    assert materialization["claim_boundary"][
        "selected_frame_is_generated_next_observation_candidate"
    ] is True
    assert target_frame.is_file()


def test_copy_or_extract_wam_frame_marks_video_first_frame_fallback(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "rollout.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    assert writer.isOpened()
    seed_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    seed_frame[:, ::2] = 220
    dark_future = np.full((24, 32, 3), 8, dtype=np.uint8)
    for frame in (seed_frame, dark_future, dark_future):
        writer.write(frame)
    writer.release()

    target_frame = tmp_path / "generated" / "next.jpg"
    namespace = _persistent_runner_namespace()
    copy_or_extract = namespace["_copy_or_extract_wam_frame"]
    materialization = copy_or_extract(
        {"rollouts": [{"generated_video_path": str(video)}]},
        target_frame,
    )

    assert materialization["status"] == "completed"
    assert materialization["source_kind"] == "video_first_frame"
    assert materialization["selected_frame_index"] == 0
    assert materialization["future_frame_selected"] is False
    assert materialization["future_frame_selection_status"] == "blocked"
    assert "no_usable_future_next_observation_frame" in materialization[
        "future_frame_selection_blockers"
    ]
    assert materialization["claim_boundary"][
        "future_frame_rollout_quality_not_proven_by_this_materialization"
    ] is True
    assert target_frame.is_file()


def test_vast_probe_env_forwards_persistent_inner_policy_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV,
        session.DEFAULT_INNER_POLICY_COMMAND,
    )
    monkeypatch.setenv(session.INNER_POLICY_COMMAND_ENV, session.DEFAULT_INNER_POLICY_COMMAND)

    env = vast_provider_adapter._probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
    )

    assert (
        env[session.PERSISTENT_SESSION_INNER_POLICY_COMMAND_ENV]
        == session.DEFAULT_INNER_POLICY_COMMAND
    )
    assert env[session.INNER_POLICY_COMMAND_ENV] == session.DEFAULT_INNER_POLICY_COMMAND


def test_runpod_persistent_session_resumes_completed_output_without_paid_relaunch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    observation_payload = json.loads(observation_path.read_text(encoding="utf-8"))
    observation_payload["observation"]["task_prompt"] = "open the refrigerator"
    observation_payload["observation"]["target_object_id"] = "task_target"
    observation_path.write_text(json.dumps(observation_payload), encoding="utf-8")
    job = tmp_path / "jobs"
    runpod_dir = job / "runpod_persistent_session_run"
    output_zip = runpod_dir / "runpod_provider_runtime_output.zip"
    runpod_dir.mkdir(parents=True)
    (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
        json.dumps({"status": "pod_created", "pod_id": "pod-123"}),
        encoding="utf-8",
    )
    (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pod_id": "pod-123",
                "provider_command_status": "completed",
                "runtime_result_status": "completed",
                "output_zip_present": True,
                "teardown_action": "stop",
                "teardown_performed": True,
                "continuing_spend_from_this_run": False,
                "provider_command_blockers": [],
                "runtime_result_blockers": [],
            }
        ),
        encoding="utf-8",
    )
    (runpod_dir / "runpod_wam_async_stop_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pod_id": "pod-123",
                "continuing_spend_from_this_run": False,
                "stopped_pod_preserved_for_warm_reuse": True,
            }
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "persistent_provider_session_used": True,
                    "provider_instance_reused_for_policy_and_wam_loop": True,
                    "repeated_policy_calls_count": 5,
                    "generated_next_observation_count": 4,
                    "live_wam_generation_success_count": 4,
                    "learned_wam_model_success_count": 4,
                    "policy_observes_wam_generated_next_observation": True,
                    "unitree_groot_n17_sonic_model_executed": True,
                    "unitree_groot_n17_sonic_policy_action_command_ran": True,
                    "blockers": [],
                }
            ),
        )
        for index in range(1, 5):
            archive.write(
                frame,
                f"generated_next_observations/wam_generated_next_observation_step_{index:04d}.jpg",
            )

    def fail_if_called(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("completed RunPod output should be finalized without paid relaunch")

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fail_if_called)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fail_if_called)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fail_if_called)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=job,
        loop_step_count=5,
        timeout_seconds=60,
    )

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["provider_output_resume_used"] is True
    assert output["generated_next_observation_count"] == 4
    assert output["live_wam_generation_success_count"] == 4
    assert output["learned_wam_model_success_count"] == 4
    assert output["continuing_spend_from_this_run"] is False
    assert output["runpod_teardown_manifest_path"].endswith("runpod_wam_async_stop_manifest.json")
    result_path = job / "unitree_groot_n17_sonic_vast_persistent_session_result.json"
    assert result_path.is_file()
    judge = json.loads(
        (job / "manipulation_success_evaluator_results.json").read_text(encoding="utf-8")
    )
    assert judge["task_prompt"] == "open the refrigerator"
    assert judge["question"] == "Did the requested manipulation succeed?"
    assert "sink" not in json.dumps(judge).lower()


def test_runpod_finalizer_surfaces_visual_quality_and_keepalive_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "jobs"
    runpod_dir = job / "runpod_persistent_session_run"
    runpod_dir.mkdir(parents=True)
    output_zip = runpod_dir / "runpod_provider_runtime_output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "status": "completed",
                    "persistent_provider_session_used": True,
                    "provider_instance_reused_for_policy_and_wam_loop": True,
                    "repeated_policy_calls_count": 2,
                    "generated_next_observation_count": 1,
                    "live_wam_generation_success_count": 1,
                    "learned_wam_model_success_count": 1,
                    "unitree_groot_n17_sonic_model_executed": True,
                    "unitree_groot_n17_sonic_policy_action_command_ran": True,
                    "unitree_policy_action_command_ran": True,
                    "policy_action_model_command_ran": True,
                    "blockers": [],
                }
            ),
        )
    visual_report = job / "wam_rollout_visual_quality_report.json"
    visual_report.parent.mkdir(parents=True, exist_ok=True)
    visual_report.write_text(
        json.dumps(
            {
                "status": "failed_visual_quality_gate",
                "visual_success": False,
                "blockers": ["wam_generated_frame_edge_structure_drift"],
            }
        ),
        encoding="utf-8",
    )

    def fake_postprocess(**_kwargs):
        return {
            "wam_rollout_visual_success": False,
            "wam_rollout_visual_quality_report": str(visual_report),
        }

    monkeypatch.setattr(
        session,
        "_postprocess_imported_persistent_session_artifacts",
        fake_postprocess,
    )

    output, exit_code = session._finalize_runpod_persistent_session_output(
        job=job,
        generated_at="now",
        policy_observation_path=tmp_path / "observation.json",
        git_evidence=_clean_launch_git_evidence(),
        poll_manifest={
            "status": "completed",
            "provider_command_status": "completed",
            "output_zip_present": True,
            "teardown_requested": True,
            "teardown_action": "keep_on_success",
            "teardown_performed": False,
            "requested_keep_running_on_success": True,
            "keep_running_on_success": True,
            "keepalive_performed": True,
            "keepalive_manifest_path": str(runpod_dir / "runpod_wam_async_keepalive_manifest.json"),
            "warm_candidate": {"path": str(tmp_path / "warm_candidate.json")},
            "continuing_spend_from_this_run": True,
            "keepalive_runtime_health": {
                "status": "healthy_for_hot_reuse",
                "runtime_healthy_for_hot_reuse": True,
            },
        },
        runpod_dir=runpod_dir,
        output_zip=output_zip,
    )

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["wam_rollout_visual_success"] is False
    assert output["provider_completed_but_visual_quality_failed"] is True
    assert output["policy_evaluation_ranking_ready"] is False
    assert output["policy_evaluation_ranking_status"] == "blocked_wam_visual_quality"
    assert "completed_provider_output_failed_wam_visual_quality_gate" in output[
        "policy_evaluation_ranking_blockers"
    ]
    assert "wam_generated_frame_edge_structure_drift" in output[
        "policy_evaluation_ranking_blockers"
    ]
    assert output["runpod_keepalive"]["keep_running_on_success"] is True
    assert output["runpod_keepalive"]["continuing_spend_from_this_run"] is True
    assert output["claim_boundary"]["provider_completed_but_visual_quality_failed"] is True


def test_runpod_finalizer_classifies_blocked_visual_gate_after_wam_inference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "jobs"
    runpod_dir = job / "runpod_persistent_session_run"
    runpod_dir.mkdir(parents=True)
    output_zip = runpod_dir / "runpod_provider_runtime_output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_wam_persistent_session_output.json",
            json.dumps(
                {
                    "status": "blocked",
                    "generated_next_observation_count": 0,
                    "live_wam_generation_success_count": 1,
                    "learned_wam_model_success_count": 1,
                    "blockers": [
                        "persistent_wam_generated_next_observation_visual_quality_failed",
                        "wam_generated_frame_edge_structure_drift",
                    ],
                }
            ),
        )
    visual_report = job / "wam_rollout_visual_quality_report.json"
    visual_report.write_text(
        json.dumps(
            {
                "status": "failed_visual_quality_gate",
                "visual_success": False,
                "blockers": ["wam_generated_frame_edge_structure_drift"],
            }
        ),
        encoding="utf-8",
    )

    def fake_postprocess(**_kwargs):
        return {
            "wam_rollout_visual_success": False,
            "wam_rollout_visual_quality_report": str(visual_report),
        }

    monkeypatch.setattr(
        session,
        "_postprocess_imported_persistent_session_artifacts",
        fake_postprocess,
    )

    output, exit_code = session._finalize_runpod_persistent_session_output(
        job=job,
        generated_at="now",
        policy_observation_path=tmp_path / "observation.json",
        git_evidence=_clean_launch_git_evidence(),
        poll_manifest={
            "status": "completed",
            "provider_command_status": "completed",
            "output_zip_present": True,
            "continuing_spend_from_this_run": False,
        },
        runpod_dir=runpod_dir,
        output_zip=output_zip,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["provider_completed_but_visual_quality_failed"] is True
    assert output["policy_evaluation_ranking_status"] == "blocked_wam_visual_quality"
    assert "provider_inference_output_failed_wam_visual_quality_gate" in output[
        "policy_evaluation_ranking_blockers"
    ]
    assert "wam_generated_frame_edge_structure_drift" in output[
        "policy_evaluation_ranking_blockers"
    ]
    assert output["claim_boundary"]["provider_completed_but_visual_quality_failed"] is True


def test_runpod_persistent_session_defaults_to_wam_carrier_and_wait_floor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.delenv(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_PROVIDER_BUNDLE_KIND", raising=False
    )
    monkeypatch.setenv(
        session.PERSISTENT_SESSION_PUBLIC_IMAGE_ENV,
        "docker.io/nijelhunt/blueprint-vast-unitree-groot-sonic:20260624-pydeps-vast1",
    )
    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_WAM_PUBLIC_IMAGE",
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
    )
    monkeypatch.setenv(
        "BLUEPRINT_VAST_UNITREE_GROOT_N17_SONIC_PUBLIC_IMAGE",
        "docker.io/nijelhunt/blueprint-vast-unitree-groot-sonic:20260624-pydeps-vast1",
    )
    monkeypatch.setenv(
        "BLUEPRINT_VAST_WAM_PUBLIC_IMAGE",
        "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime",
    )
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ENTRYPOINT_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WRAPPER_WATCHDOG_SECONDS", "180")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_WAIT_BUFFER_SECONDS", "30")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["provider_bundle_kind"] = kwargs["provider_bundle_kind"]
        captured["image_name"] = kwargs["image_name"]
        captured["container_disk_gb"] = kwargs["container_disk_gb"]
        captured["volume_gb"] = kwargs["volume_gb"]
        captured["wam_carrier_enabled"] = session.os.environ.get(
            "BLUEPRINT_RUNPOD_WAM_CARRIER_UNITREE_GROOT_N17_SONIC"
        )
        captured["wam_visual_profile"] = session.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
        )
        captured["wam_num_steps"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS")
        captured["wam_guidance"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_GUIDANCE")
        captured["wam_num_frames"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        captured["wam_height"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT")
        captured["wam_width"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH")
        captured["wam_fps"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_FPS")
        captured["wam_checkpoint_timeout"] = session.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_CHECKPOINT_RESOLUTION_TIMEOUT_SECONDS"
        )
        captured["groot_bootstrap_mode"] = session.os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_BOOTSTRAP_MODE"
        )
        captured["groot_sparse_checkout"] = session.os.environ.get(
            "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SPARSE_CHECKOUT"
        )
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        captured["max_wait_seconds"] = kwargs["max_wait_seconds"]
        runpod_dir = Path(kwargs["job_dir"])
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "output_zip_present": False,
                    "provider_command_status": "blocked",
                    "provider_command_blockers": [
                        "runpod_provider_runtime_output_zip_not_received_locally"
                    ],
                    "teardown_performed": True,
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=2,
        timeout_seconds=60,
        max_wait_seconds=20,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert "runpod_wrapper_or_upload_watchdog_no_valid_provider_artifact" in output["blockers"]
    assert "runpod_provider_runtime_output_zip_not_received_locally" in output["blockers"]
    assert output["details"]["provider_command_blockers"] == [
        "runpod_provider_runtime_output_zip_not_received_locally"
    ]
    assert captured["provider_bundle_kind"] == "wam"
    assert captured["image_name"] == "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
    assert captured["wam_carrier_enabled"] == "true"
    assert captured["wam_visual_profile"] == "smoke"
    assert captured["wam_num_steps"] == "2"
    assert captured["wam_guidance"] == "3.5"
    assert captured["wam_num_frames"] == "9"
    assert captured["wam_height"] == "128"
    assert captured["wam_width"] == "128"
    assert captured["wam_fps"] == "4"
    assert captured["wam_checkpoint_timeout"] == "1200"
    assert captured["groot_bootstrap_mode"] == "system_python_minimal"
    assert captured["groot_sparse_checkout"] == "true"
    assert captured["container_disk_gb"] == 240
    assert captured["volume_gb"] == 120
    assert captured["max_wait_seconds"] == 210


def test_runpod_persistent_session_review_quality_profile_uses_higher_fidelity_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["image_name"] = kwargs["image_name"]
        captured["runpod_teardown_action"] = session.os.environ.get(
            session.RUNPOD_WAM_TEARDOWN_ACTION_ENV
        )
        captured["wam_visual_profile"] = session.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
        )
        captured["wam_num_steps"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS")
        captured["wam_guidance"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_GUIDANCE")
        captured["wam_num_frames"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        captured["wam_height"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT")
        captured["wam_width"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH")
        captured["wam_fps"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_FPS")
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        runpod_dir = Path(kwargs["job_dir"])
        poll_manifest = {
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(poll_manifest),
            encoding="utf-8",
        )
        return poll_manifest

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=2,
        timeout_seconds=60,
        max_wait_seconds=20,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert (
        captured["image_name"]
        == session.DEFAULT_RUNPOD_UNITREE_GROOT_SONIC_WAM_PUBLIC_IMAGE
    )
    assert captured["runpod_teardown_action"] == "keep_on_success"
    assert session.os.environ.get(session.RUNPOD_WAM_TEARDOWN_ACTION_ENV) is None
    assert captured["wam_visual_profile"] == "review_quality"
    assert captured["wam_num_steps"] == "35"
    assert captured["wam_guidance"] == "6.0"
    assert captured["wam_num_frames"] == "81"
    assert captured["wam_height"] == "480"
    assert captured["wam_width"] == "640"
    assert captured["wam_fps"] == "15"


def test_review_quality_profile_rejects_128px_bundle_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "9")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_HEIGHT", "128")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_WIDTH", "128")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_FPS", "4")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "review_quality_profile_width_below_minimum" in manifest["blockers"]
    assert "review_quality_profile_height_below_minimum" in manifest["blockers"]
    assert "review_quality_profile_fps_below_minimum" in manifest["blockers"]
    assert "review_quality_profile_num_frames_below_minimum" in manifest["blockers"]
    assert "review_quality_profile_num_frames_below_oscar_default" in manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_qa["status"] == "passed_visual_quality_gate"


def test_review_quality_profile_rejects_low_oscar_sampling_budget_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_STEPS", "12")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_GUIDANCE", "3.5")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "24")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "low-sampling-budget",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "review_quality_profile_num_frames_below_oscar_default" in manifest["blockers"]
    assert "review_quality_profile_num_steps_below_oscar_default" in manifest["blockers"]
    assert "review_quality_profile_guidance_below_oscar_default" in manifest["blockers"]


def test_review_quality_profile_rejects_bad_source_frame_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "source_policy_observation_visual_qa_failed_for_review_quality" in manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert "source_policy_observation_too_dark_for_review" in source_qa["blockers"]


def test_review_quality_profile_records_projected_skeleton_robot_material_advisory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_flat_projected_robot_regions_frame(tmp_path / "frame.jpg")
    trace = _write_projected_skeleton_trace(tmp_path / "g1_projected_skeleton_trace.jsonl")
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "open_refrigerator",
        "target_object_id": "fridge_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {
            "camera_frame_path": str(frame),
            "projected_skeleton_trace_path": str(trace),
        },
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert "source_policy_observation_visual_qa_failed_for_review_quality" not in manifest[
        "blockers"
    ]
    assert manifest["semantic_visual_qa_source_paths"]["projected_skeleton_trace"] == str(
        trace.resolve()
    )
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_qa["status"] == "passed_visual_quality_gate"
    assert source_qa["blockers"] == []
    assert source_qa["projected_robot_material_quality_enforced"] is False
    material = source_qa["projected_robot_material_quality"]
    assert material["projected_skeleton_trace_path"] == str(trace.resolve())
    assert material["projected_skeleton_trace_used"] is True
    assert "source_policy_observation_projected_robot_material_low_detail" in material[
        "blockers"
    ]


def test_persistent_session_bundle_preserves_isaac_seed_geometry_context(
    tmp_path: Path,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "render" / "frames" / "robot_pov_0000.png")
    geometry = tmp_path / "render" / "manipulation_pov_geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "status": "PASS",
                "frames": [
                    {
                        "schema_version": "manipulation_pov_geometry.v1",
                        "status": "PASS",
                        "camera": "robot_pov",
                        "projected_landmarks": [
                            {
                                "landmark_id": "left_hand_link",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 220.0,
                                    "v_px": 330.0,
                                    "depth_m": 0.3,
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    placement = tmp_path / "render" / "placement_validation.json"
    placement.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    stance = tmp_path / "render" / "task_stance_plan.json"
    stance.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "open_refrigerator",
        "target_object_id": "refrigerator",
        "camera_frame_path": str(frame),
        "manipulation_pov_geometry_path": str(geometry),
        "placement_validation_path": str(placement),
        "task_stance_plan_path": str(stance),
        "visual_observation": {
            "camera_frame_path": str(frame),
            "manipulation_pov_geometry_path": str(geometry),
            "placement_validation_path": str(placement),
        },
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps({"observation": observation}), encoding="utf-8")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["semantic_visual_qa_source_paths"]["manipulation_pov_geometry"] == str(
        geometry.resolve()
    )
    assert manifest["semantic_visual_qa_source_paths"]["placement_validation"] == str(
        placement.resolve()
    )
    assert manifest["semantic_visual_qa_source_paths"]["task_stance_plan"] == str(
        stance.resolve()
    )
    assert manifest["runtime_isaac_scene_context_paths"] == {
        "manipulation_pov_geometry": "provider_runtime/isaac_scene_context/manipulation_pov_geometry.json",
        "placement_validation": "provider_runtime/isaac_scene_context/placement_validation.json",
        "task_stance_plan": "provider_runtime/isaac_scene_context/task_stance_plan.json",
    }
    with zipfile.ZipFile(manifest["bundle_path"]) as archive:
        names = set(archive.namelist())
        runner = archive.read(
            "provider_runtime/unitree_groot_n17_sonic_wam_persistent_session_runner.py"
        ).decode()
    assert "provider_runtime/isaac_scene_context/manipulation_pov_geometry.json" in names
    assert "provider_runtime/isaac_scene_context/placement_validation.json" in names
    assert "provider_runtime/isaac_scene_context/task_stance_plan.json" in names
    assert "isaac_scene_context_output_paths" in runner
    assert 'output_dir / "isaac_scene_context"' in runner
    session_input = json.loads(
        (tmp_path / "bundle" / "provider_runtime" / "persistent_session_input.json").read_text(
            encoding="utf-8"
        )
    )
    assert session_input["isaac_scene_context"]["status"] == "available"
    initial_observation = session_input["initial_observation"]
    assert (
        initial_observation["visual_observation"]["manipulation_pov_geometry_path"]
        == "provider_runtime/isaac_scene_context/manipulation_pov_geometry.json"
    )
    assert (
        initial_observation["visual_observation"]["placement_validation_path"]
        == "provider_runtime/isaac_scene_context/placement_validation.json"
    )
    assert (
        initial_observation["visual_observation"]["task_stance_plan_path"]
        == "provider_runtime/isaac_scene_context/task_stance_plan.json"
    )
    assert (
        session_input["isaac_scene_context"]["claim_boundary"][
            "isaac_scene_context_is_geometry_metadata_not_policy_action_projection"
        ]
        is True
    )


def test_persistent_session_bundle_accepts_explicit_isaac_scene_context_sidecars(
    tmp_path: Path,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.png")
    geometry = tmp_path / "sidecars" / "manipulation_pov_geometry.json"
    geometry.parent.mkdir(parents=True)
    geometry.write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "status": "PASS",
                "frames": [
                    {
                        "schema_version": "manipulation_pov_geometry.v1",
                        "status": "PASS",
                        "camera": "robot_pov",
                        "camera_meta": {
                            "camera_eye_xyz": [0.0, 0.0, 0.0],
                            "camera_target_xyz": [1.0, 0.0, 0.0],
                            "camera_vfov_deg": 90.0,
                            "viewport_size_px": [640, 480],
                            "arm_link_points_by_arm_xyz": {
                                "left": {"shoulder": [1.0, -0.2, -0.2], "hand": [1.2, -0.2, -0.4]},
                                "right": {"shoulder": [1.0, 0.2, -0.2], "hand": [1.2, 0.2, -0.4]},
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    placement = tmp_path / "sidecars" / "placement_validation.json"
    placement.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    stance = tmp_path / "sidecars" / "task_stance_plan.json"
    stance.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(
        json.dumps(
            {
                "observation": {
                    "schema_version": "initial_policy_observation.v1",
                    "task_id": "open_refrigerator",
                    "target_object_id": "refrigerator",
                    "camera_frame_path": str(frame),
                    "visual_observation": {"camera_frame_path": str(frame)},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        manipulation_pov_geometry_path=geometry,
        placement_validation_path=placement,
        task_stance_plan_path=stance,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["explicit_isaac_scene_context"]["status"] == "attached"
    assert manifest["semantic_visual_qa_source_paths"]["manipulation_pov_geometry"] == str(
        geometry.resolve()
    )
    assert manifest["semantic_visual_qa_source_paths"]["placement_validation"] == str(
        placement.resolve()
    )
    assert manifest["semantic_visual_qa_source_paths"]["task_stance_plan"] == str(
        stance.resolve()
    )
    session_input = json.loads(
        (tmp_path / "bundle" / "provider_runtime" / "persistent_session_input.json").read_text(
            encoding="utf-8"
        )
    )
    assert session_input["isaac_scene_context"]["status"] == "available"
    assert session_input["isaac_scene_context"]["explicit_request"]["status"] == "attached"
    initial_observation = session_input["initial_observation"]
    assert (
        initial_observation["manipulation_pov_geometry_path"]
        == "provider_runtime/isaac_scene_context/manipulation_pov_geometry.json"
    )
    assert (
        initial_observation["isaac_scene_manifest_path"]
        == "provider_runtime/isaac_scene_context/placement_validation.json"
    )
    with zipfile.ZipFile(manifest["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/isaac_scene_context/manipulation_pov_geometry.json" in names
    assert "provider_runtime/isaac_scene_context/placement_validation.json" in names
    assert "provider_runtime/isaac_scene_context/task_stance_plan.json" in names


def test_persistent_session_bundle_blocks_missing_explicit_isaac_sidecar(
    tmp_path: Path,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.png")
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(
        json.dumps(
            {
                "observation": {
                    "schema_version": "initial_policy_observation.v1",
                    "task_id": "open_refrigerator",
                    "camera_frame_path": str(frame),
                    "visual_observation": {"camera_frame_path": str(frame)},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        manipulation_pov_geometry_path=tmp_path / "missing_geometry.json",
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "blocked_explicit_isaac_manipulation_pov_geometry_path_missing" in manifest[
        "blockers"
    ]
    assert manifest["explicit_isaac_scene_context"]["status"] == "blocked"


def test_persistent_session_bundle_blocks_offscreen_semantic_initial_observation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    target = {
        "object_id": "Sink054_handle",
        "label": "right sink handle",
        "bbox": {"x": 700, "y": 180, "width": 120, "height": 90},
        "keypoints": {"center": [760, 225]},
        "confidence": 0.95,
        "occlusion": "visible",
    }
    observation = {
        "schema_version": "initial_policy_observation.v1",
        "task_id": "turn_on_sink_handle",
        "target_object_id": "Sink054_handle",
        "camera_frame_path": str(frame),
        "visual_observation": {"camera_frame_path": str(frame)},
        "object_index": {"objects": [target]},
        "eval_ready_task_grounding": {
            "schema_version": "eval_ready_task_grounding.v1",
            "status": "ready_for_learned_wam_rollout_request",
            "selected_task_target": target,
            "readiness": {
                "learned_rollout_request_ready": True,
                "target_crop_available": False,
                "target_mask_or_keypoint_available": True,
                "blockers": [],
            },
        },
        "unitree_g1_sonic_state": {
            "left_leg": [0.0] * 6,
            "right_leg": [0.0] * 6,
            "waist": [0.0] * 3,
            "left_arm": [0.0] * 7,
            "right_arm": [0.0] * 7,
            "left_hand": [0.0] * 7,
            "right_hand": [0.0] * 7,
            "projected_gravity": [0.0, 0.0, -1.0],
        },
    }
    observation_path = tmp_path / "observation.json"
    observation_path.write_text(json.dumps({"observation": observation}), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "source_policy_observation_visual_qa_failed_for_review_quality" in manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    assert source_qa["target_visibility_status"] == "failed_semantic_gate"
    assert "target_object_offscreen_in_source_observation" in source_qa["blockers"]


def test_review_quality_profile_remediates_bad_source_frame_with_image_model_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    command = _write_fake_image_model_remediator(tmp_path / "fake_remediate.py")
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv("BLUEPRINT_ALLOW_IMAGE_MODEL_RENDER_REMEDIATION", "true")
    monkeypatch.setenv(
        "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_COMMAND",
        f"{sys.executable} {command}",
    )

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "bundle_ready"
    assert manifest["image_model_render_remediation_applied"] is True
    assert manifest["image_model_render_remediation_status"] == "completed"
    assert manifest["original_initial_frame_path"] == str(frame.resolve())
    assert manifest["initial_frame_path"].endswith("enhanced_initial_policy_frame.png")
    assert not manifest["blockers"]
    source_qa = json.loads(
        (tmp_path / "bundle" / "source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    original_qa = json.loads(
        (tmp_path / "bundle" / "original_source_policy_observation_visual_qa.json").read_text(
            encoding="utf-8"
        )
    )
    remediation = json.loads(
        Path(manifest["image_model_render_remediation_manifest_path"]).read_text(encoding="utf-8")
    )
    policy_input = json.loads(
        (tmp_path / "bundle" / "provider_runtime" / "policy_input.json").read_text(encoding="utf-8")
    )

    assert source_qa["status"] == "passed_visual_quality_gate"
    assert original_qa["status"] == "failed_visual_quality_gate"
    assert remediation["claim_boundary"]["capture_truth"] is False
    assert remediation["claim_boundary"]["geometry_truth"] is False
    assert remediation["claim_boundary"]["collision_truth"] is False
    assert remediation["claim_boundary"]["near_preserving_image_enhancement_only"] is True
    assert (
        tmp_path / "bundle" / "image_model_render_remediation" / "original_initial_policy_frame.jpg"
    ).is_file()
    assert (
        tmp_path / "bundle" / "image_model_render_remediation" / "enhanced_initial_policy_frame.png"
    ).is_file()
    assert (
        tmp_path
        / "bundle"
        / "provider_runtime"
        / "image_model_render_remediation"
        / "enhanced_initial_policy_frame.png"
    ).is_file()
    with Image.open(tmp_path / "bundle" / "provider_runtime" / "initial_policy_frame.png") as image:
        assert image.size == (640, 480)
        assert np.asarray(image).mean() > 80.0
    observation = policy_input["observation"]
    assert observation["source_kind"] == "image_model_enhanced_3d_render_seed"
    assert observation["claim_boundary"]["capture_truth"] is False
    assert observation["visual_observation"]["original_3d_render_frame_path"] == str(
        frame.resolve()
    )
    with zipfile.ZipFile(Path(manifest["bundle_path"])) as archive:
        names = set(archive.namelist())
    assert (
        "provider_runtime/image_model_render_remediation/enhanced_initial_policy_frame.png" in names
    )


def test_review_quality_profile_records_blocked_image_model_remediation_without_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv("BLUEPRINT_ALLOW_IMAGE_MODEL_RENDER_REMEDIATION", "true")
    monkeypatch.delenv("BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_COMMAND", raising=False)

    manifest = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["image_model_render_remediation_applied"] is False
    assert manifest["image_model_render_remediation_status"] == "blocked"
    assert "image_model_render_remediation_command_not_configured" in manifest["blockers"]
    assert "source_policy_observation_visual_qa_failed_for_review_quality" in manifest["blockers"]
    remediation = json.loads(
        Path(manifest["image_model_render_remediation_manifest_path"]).read_text(encoding="utf-8")
    )
    assert remediation["status"] == "blocked"
    assert "image_model_render_remediation_command_not_configured" in remediation["blockers"]
    assert remediation["claim_boundary"]["capture_truth"] is False
    assert (
        tmp_path / "bundle" / "image_model_render_remediation" / "original_initial_policy_frame.jpg"
    ).is_file()


def test_review_quality_long_rollout_requires_passed_short_visual_sanity_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv(session.PERSISTENT_WAM_LONG_REVIEW_ROLLOUT_ENV, "true")
    monkeypatch.delenv(session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV, raising=False)

    blocked = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "blocked-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert blocked["status"] == "blocked"
    assert "review_quality_long_rollout_requires_passed_short_visual_sanity" in blocked["blockers"]
    assert "short_visual_sanity_manifest_env_missing" in blocked["blockers"]
    assert (
        "review_quality_long_rollout_env_override_requires_short_visual_sanity_manifest"
        in blocked["blockers"]
    )

    sanity_manifest = _write_passed_short_sanity_manifest(
        tmp_path / "short-sanity",
        observation_path,
    )
    structural_fallback_manifest = tmp_path / "short-sanity" / "structural-fallback.json"
    structural_payload = json.loads(sanity_manifest.read_text(encoding="utf-8"))
    structural_payload["structural_fallback_used"] = True
    structural_fallback_manifest.write_text(json.dumps(structural_payload), encoding="utf-8")
    structural_validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        structural_fallback_manifest,
        policy_observation_path=observation_path,
    )
    assert structural_validation["status"] == "blocked"
    assert (
        "short_visual_sanity_structural_fallback_cannot_unlock_long_rollout"
        in structural_validation["blockers"]
    )

    monkeypatch.setenv(
        session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV, str(sanity_manifest)
    )
    blocked_without_strategy = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "blocked-without-strategy-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert blocked_without_strategy["status"] == "blocked"
    assert (
        "review_quality_12_step_paid_rollout_requires_clean_frame_reanchoring_or_drift_blocker"
        in blocked_without_strategy["blockers"]
    )
    quality_gate = json.loads(
        (
            tmp_path
            / "blocked-without-strategy-bundle"
            / "long_review_rollout_quality_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert quality_gate["status"] == "blocked_missing_long_rollout_quality_proof"
    assert quality_gate["paid_rollout_launch_allowed"] is False

    monkeypatch.setenv(session.PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV, "2")
    allowed = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "allowed-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert allowed["status"] == "bundle_ready"
    assert (
        "review_quality_long_rollout_requires_passed_short_visual_sanity" not in allowed["blockers"]
    )
    allowed_gate = json.loads(
        (tmp_path / "allowed-bundle" / "long_review_rollout_quality_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert allowed_gate["status"] == "passed_periodic_clean_frame_reanchoring"
    assert allowed_gate["paid_rollout_launch_allowed"] is True
    assert allowed_gate["clean_frame_reanchoring"]["interval_steps"] == 2
    runtime_input = json.loads(
        (
            tmp_path / "allowed-bundle" / "provider_runtime" / "persistent_session_input.json"
        ).read_text(encoding="utf-8")
    )
    assert runtime_input["clean_frame_reanchoring"]["periodic_clean_frame_reanchoring_proven"] is True
    assert runtime_input["clean_frame_reanchoring"]["expected_reanchor_transition_indices"] == [
        2,
        4,
        6,
        8,
        10,
    ]


def test_review_quality_12_step_rollout_blocks_on_concrete_drift_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    sanity_manifest = _write_passed_short_sanity_manifest(
        tmp_path / "short-sanity",
        observation_path,
    )
    drift_report = tmp_path / "prior_wam_rollout_visual_quality_report.json"
    drift_report.write_text(
        json.dumps(
            {
                "schema_version": "persistent_policy_wam_visual_quality_report.v1",
                "status": "failed_visual_quality_gate",
                "visual_success": False,
                "visual_profile": "review_quality",
                "generated_frame_count": 11,
                "autoregressive_chain_guard": {
                    "autoregressive_chain_used": True,
                    "generated_frame_count": 11,
                    "long_horizon_visual_drift_blocker": True,
                    "long_rollout_should_not_be_overclaimed": True,
                },
                "blockers": [
                    "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout"
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv(
        session.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV,
        str(sanity_manifest),
    )
    monkeypatch.setenv(
        session.PERSISTENT_WAM_AUTOREGRESSIVE_DRIFT_BLOCKER_MANIFEST_ENV,
        str(drift_report),
    )
    monkeypatch.delenv(session.PERSISTENT_WAM_CLEAN_FRAME_REANCHOR_INTERVAL_ENV, raising=False)

    blocked = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "blocked-drift-bundle",
        policy_observation_path=observation_path,
        loop_step_count=12,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert blocked["status"] == "blocked"
    assert (
        "autoregressive_chain_drift_blocker_present_before_12_step_paid_rollout"
        in blocked["blockers"]
    )
    gate = json.loads(
        (tmp_path / "blocked-drift-bundle" / "long_review_rollout_quality_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["status"] == "blocked_autoregressive_drift_confirmed"
    assert gate["concrete_autoregressive_drift_blocker_proven"] is True
    assert gate["paid_rollout_launch_allowed"] is False


def test_review_quality_paid_rollout_blocks_on_materialization_quality_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    materialization_summary = tmp_path / "wam_materialization_summary.json"
    materialization_summary.write_text(
        json.dumps(
            {
                "schema_version": "persistent_wam_materialization_summary.v1",
                "status": "completed",
                "future_frame_quality_status": "failed",
                "future_frame_quality_blockers": [
                    "wam_generated_next_observation_future_frame_degraded_visual_signal"
                ],
                "materialized_future_frame_count": 4,
                "degraded_future_frame_count": 4,
                "video_first_frame_materialization_count": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE", "review_quality")
    monkeypatch.setenv(
        session.PERSISTENT_WAM_MATERIALIZATION_BLOCKER_MANIFEST_ENV,
        str(materialization_summary),
    )

    validation = session.validate_persistent_wam_materialization_quality_blocker(
        materialization_summary
    )
    assert validation["status"] == "confirmed_materialization_quality_blocker"
    assert validation["concrete_materialization_quality_blocker_proven"] is True

    blocked = session.build_persistent_session_provider_bundle(
        job_dir=tmp_path / "blocked-materialization-bundle",
        policy_observation_path=observation_path,
        loop_step_count=2,
        use_live_wam=True,
        allow_structural_wam_fallback=False,
        generated_at="now",
    )

    assert blocked["status"] == "blocked"
    assert (
        "future_frame_materialization_quality_blocker_present_before_paid_rollout"
        in blocked["blockers"]
    )
    gate = json.loads(
        (
            tmp_path
            / "blocked-materialization-bundle"
            / "long_review_rollout_quality_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert gate["status"] == "blocked_materialization_quality_confirmed"
    assert gate["paid_rollout_launch_allowed"] is False
    assert gate["concrete_materialization_quality_blocker_proven"] is True
    assert (
        gate["claim_boundary"][
            "materialization_quality_blocker_prevents_same_config_paid_rollout"
        ]
        is True
    )
    provider_manifest = json.loads(
        (
            tmp_path
            / "blocked-materialization-bundle"
            / "provider_runtime"
            / "unitree_groot_n17_sonic_policy_provider_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert provider_manifest["materialization_quality_blocker_validation"]["status"] == (
        "confirmed_materialization_quality_blocker"
    )


def test_short_visual_sanity_blocks_before_provider_when_source_qa_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_dark_frame(tmp_path / "dark.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    called = False

    def fake_vast(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider runner should not start after source QA failure")

    monkeypatch.setattr(short_sanity, "run_persistent_session", fake_vast)

    manifest, exit_code = short_sanity.run_short_visual_sanity(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "short",
        transition_count=2,
    )

    assert exit_code == 2
    assert called is False
    assert manifest["status"] == "blocked"
    assert manifest["provider"] == "vast"
    assert manifest["paid_provider"]["provider"] == "vast"
    assert manifest["paid_provider"]["used"] is False
    assert manifest["paid_provider"]["teardown_status"] == "not_required_prelaunch_blocked"
    assert manifest["provider_success"] is False
    assert manifest["visually_useful_rollout"] is False
    assert manifest["claim_boundary"]["provider_success_separate_from_visually_useful_rollout"] is True
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert "source_policy_observation_too_dark_for_review" in manifest["blockers"]
    assert Path(manifest["short_visual_sanity_manifest_path"]).is_file()


def test_short_visual_sanity_pass_manifest_records_review_artifacts_and_teardown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    captured: dict[str, object] = {}

    def fake_runpod(**kwargs):
        captured["loop_step_count"] = kwargs["loop_step_count"]
        captured["visual_profile"] = short_sanity.os.environ.get(
            "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE"
        )
        captured["num_frames"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        captured["height"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT")
        captured["width"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH")
        captured["fps"] = short_sanity.os.environ.get("BLUEPRINT_OSCAR_WAM_FPS")
        job = Path(kwargs["job_dir"]) / "short-run"
        job.mkdir(parents=True)
        runpod_dir = job / "runpod_persistent_session_run"
        runpod_dir.mkdir()
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created", "pod_id": "pod-123"}),
            encoding="utf-8",
        )
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "pod_id": "pod-123",
                    "teardown_performed": True,
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        (runpod_dir / "runpod_wam_async_delete_manifest.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "pod_id": "pod-123",
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        source_qa = job / "source_policy_observation_visual_qa.json"
        report = job / "wam_rollout_visual_quality_report.json"
        contact_sheet = _write_reviewable_frame(job / "wam_rollout_contact_sheet.jpg")
        frame_stats = job / "wam_rollout_frame_stats.jsonl"
        video_status = job / "video_review_status.json"
        review_video = job / "review_video" / "persistent_policy_wam_live_rollout_review.mp4"
        review_video.parent.mkdir()
        source_qa.write_text(
            json.dumps({"status": "passed_visual_quality_gate"}),
            encoding="utf-8",
        )
        report.write_text(
            json.dumps(
                {
                    "status": "passed_visual_quality_gate",
                    "visual_profile": "review_quality",
                    "visual_success": True,
                    "profile_contract": {
                        "review_quality_profile": True,
                        "review_quality_minimum_satisfied": True,
                        "smoke_only": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        frame_stats.write_text("{}\n", encoding="utf-8")
        video_status.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "ffprobe_command_ran": True,
                    "ffprobe_returncode": 0,
                    "ffprobe_metadata": {
                        "streams": [
                            {
                                "width": 640,
                                "height": 480,
                                "avg_frame_rate": "15/1",
                                "nb_frames": "24",
                            }
                        ],
                        "format": {"duration": "1.6", "size": "1000"},
                    },
                }
            ),
            encoding="utf-8",
        )
        review_video.write_bytes(b"mp4")
        result = {
            "status": "completed",
            "job_dir": str(job),
            "generated_next_observation_count": 2,
            "live_wam_generation_success_count": 2,
            "learned_wam_model_success_count": 2,
            "runpod_create_manifest_path": str(
                runpod_dir / "runpod_wam_async_create_manifest.json"
            ),
            "runpod_poll_manifest_path": str(runpod_dir / "runpod_wam_async_poll_manifest.json"),
            "runpod_teardown_manifest_path": str(
                runpod_dir / "runpod_wam_async_delete_manifest.json"
            ),
            "postprocess_artifacts": {
                "source_policy_observation_visual_qa": str(source_qa),
                "wam_rollout_visual_quality_report": str(report),
                "wam_rollout_contact_sheet": str(contact_sheet),
                "wam_rollout_frame_stats": str(frame_stats),
                "video_review_status": str(video_status),
                "review_video_path": str(review_video),
            },
            "review_video_path": str(review_video),
            "video_review_status_path": str(video_status),
            "source_policy_observation_visual_qa_path": str(source_qa),
            "wam_rollout_visual_quality_report_path": str(report),
            "wam_rollout_contact_sheet_path": str(contact_sheet),
            "wam_rollout_visual_success": True,
        }
        (job / "unitree_groot_n17_sonic_vast_persistent_session_result.json").write_text(
            json.dumps(result),
            encoding="utf-8",
        )
        return result, 0

    monkeypatch.setattr(short_sanity, "run_persistent_session_runpod", fake_runpod)

    manifest, exit_code = short_sanity.run_short_visual_sanity(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "short",
        provider="runpod",
        transition_count=2,
    )

    assert exit_code == 0
    assert manifest["status"] == "passed_short_visual_sanity"
    assert manifest["short_visual_sanity_passed"] is True
    assert captured["loop_step_count"] == 3
    assert captured["visual_profile"] == "review_quality"
    assert captured["num_frames"] == "81"
    assert captured["height"] == "480"
    assert captured["width"] == "640"
    assert captured["fps"] == "15"
    assert manifest["requested_transition_count"] == 2
    assert manifest["generated_transition_count"] == 2
    assert manifest["ffprobe_metadata"]["streams"][0]["width"] == 640
    assert manifest["provider_success"] is True
    assert manifest["visually_useful_rollout"] is True
    assert manifest["claim_boundary"]["provider_success"] is True
    assert manifest["claim_boundary"]["visually_useful_rollout"] is True
    assert manifest["claim_boundary"]["capture_truth"] is False
    assert manifest["claim_boundary"]["geometry_truth"] is False
    assert Path(manifest["wam_rollout_contact_sheet_path"]).is_file()
    assert manifest["paid_provider"]["used"] is True
    assert manifest["paid_provider"]["teardown_status"] == "completed"
    assert manifest["paid_provider"]["continuing_spend_from_this_run"] is False
    validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        manifest["short_visual_sanity_manifest_path"],
        policy_observation_path=observation_path,
    )
    assert validation["status"] == "passed_short_visual_sanity"


def test_short_visual_sanity_can_wrap_existing_persistent_session_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    existing = tmp_path / "existing-run"
    existing.mkdir()
    runpod_dir = existing / "runpod_persistent_session_run"
    runpod_dir.mkdir()
    (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
        json.dumps({"status": "pod_created", "pod_id": "pod-123"}),
        encoding="utf-8",
    )
    (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pod_id": "pod-123",
                "teardown_performed": True,
                "continuing_spend_from_this_run": False,
            }
        ),
        encoding="utf-8",
    )
    (runpod_dir / "runpod_wam_async_delete_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "pod_id": "pod-123",
                "continuing_spend_from_this_run": False,
            }
        ),
        encoding="utf-8",
    )
    old_source_qa = existing / "old_source_policy_observation_visual_qa.json"
    old_source_qa.write_text(
        json.dumps(
            {
                "status": "passed_visual_quality_gate",
                "visual_profile": "smoke",
                "review_quality_required": False,
            }
        ),
        encoding="utf-8",
    )
    report = existing / "wam_rollout_visual_quality_report.json"
    report.write_text(
        json.dumps({"status": "passed_visual_quality_gate", "visual_success": True}),
        encoding="utf-8",
    )
    contact_sheet = _write_reviewable_frame(existing / "wam_rollout_contact_sheet.jpg")
    frame_stats = existing / "wam_rollout_frame_stats.jsonl"
    frame_stats.write_text("{}\n", encoding="utf-8")
    video_status = existing / "video_review_status.json"
    review_video = existing / "review_video" / "persistent_policy_wam_live_rollout_review.mp4"
    review_video.parent.mkdir()
    review_video.write_bytes(b"mp4")
    video_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": {
                    "streams": [
                        {
                            "width": 320,
                            "height": 256,
                            "avg_frame_rate": "6/1",
                            "nb_frames": "9",
                        }
                    ],
                    "format": {"duration": "1.5", "size": "1000"},
                },
            }
        ),
        encoding="utf-8",
    )
    result = {
        "status": "completed",
        "job_dir": str(existing),
        "generated_next_observation_count": 1,
        "live_wam_generation_success_count": 1,
        "learned_wam_model_success_count": 1,
        "runpod_create_manifest_path": str(runpod_dir / "runpod_wam_async_create_manifest.json"),
        "runpod_poll_manifest_path": str(runpod_dir / "runpod_wam_async_poll_manifest.json"),
        "runpod_teardown_manifest_path": str(runpod_dir / "runpod_wam_async_delete_manifest.json"),
        "postprocess_artifacts": {
            "source_policy_observation_visual_qa": str(old_source_qa),
            "wam_rollout_visual_quality_report": str(report),
            "wam_rollout_contact_sheet": str(contact_sheet),
            "wam_rollout_frame_stats": str(frame_stats),
            "video_review_status": str(video_status),
            "review_video_path": str(review_video),
        },
        "review_video_path": str(review_video),
        "video_review_status_path": str(video_status),
        "source_policy_observation_visual_qa_path": str(old_source_qa),
        "wam_rollout_visual_quality_report_path": str(report),
        "wam_rollout_contact_sheet_path": str(contact_sheet),
        "wam_rollout_visual_success": True,
    }
    result_path = existing / "unitree_groot_n17_sonic_vast_persistent_session_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    called = False

    def fake_runpod(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("imported result should not relaunch provider")

    monkeypatch.setattr(short_sanity, "run_persistent_session_runpod", fake_runpod)

    manifest, exit_code = short_sanity.run_short_visual_sanity(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "short-manifest",
        provider="runpod",
        transition_count=1,
        persistent_session_result_path=result_path,
    )

    assert exit_code == 2
    assert called is False
    assert manifest["status"] == "blocked"
    assert manifest["short_visual_sanity_passed"] is False
    assert manifest["manifest_source"] == "imported_persistent_session_result"
    assert Path(manifest["short_visual_sanity_manifest_path"]).parent == (
        tmp_path / "short-manifest"
    ).resolve()
    assert manifest["persistent_session_result_path"] == str(result_path.resolve())
    assert manifest["requested_transition_count"] == 1
    assert manifest["generated_transition_count"] == 1
    assert manifest["review_media_resolution"] == {
        "width": 320,
        "height": 256,
        "fps": 6.0,
        "frame_count": 9,
        "minimum_width": 320,
        "minimum_height": 256,
        "minimum_fps": 8.0,
        "minimum_num_frames": 12,
        "resolution_passed": True,
        "fps_passed": False,
        "frame_count_passed": False,
        "passed": False,
    }
    assert (
        "short_visual_sanity_review_video_fps_below_review_quality_minimum"
        in manifest["blockers"]
    )
    assert (
        "short_visual_sanity_review_video_frame_count_below_review_quality_minimum"
        in manifest["blockers"]
    )
    assert "short_visual_sanity_quality_report_not_review_quality_profile" in manifest[
        "blockers"
    ]
    source_qa = json.loads(
        Path(manifest["source_policy_observation_visual_qa_path"]).read_text(encoding="utf-8")
    )
    assert source_qa["visual_profile"] == "review_quality"
    assert source_qa["review_quality_required"] is True
    assert manifest["paid_provider"]["used"] is True
    assert manifest["paid_provider"]["teardown_status"] == "completed"
    assert manifest["paid_provider"]["continuing_spend_from_this_run"] is False
    validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        manifest["short_visual_sanity_manifest_path"],
        policy_observation_path=observation_path,
    )
    assert validation["status"] == "blocked"
    assert (
        "short_visual_sanity_video_status_review_video_fps_below_review_quality_minimum"
        in validation["blockers"]
    )
    assert (
        "short_visual_sanity_video_status_review_video_frame_count_below_review_quality_minimum"
        in validation["blockers"]
    )


def test_short_visual_sanity_validation_rechecks_referenced_artifacts(
    tmp_path: Path,
) -> None:
    frame = _write_reviewable_frame(tmp_path / "frame.jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    manifest_path = _write_passed_short_sanity_manifest(
        tmp_path / "short-sanity",
        observation_path,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_qa_path = Path(payload["source_policy_observation_visual_qa_path"])
    visual_report_path = Path(payload["wam_rollout_visual_quality_report_path"])
    video_status_path = Path(payload["video_review_status_path"])
    review_video_path = Path(payload["review_video_path"])
    teardown_path = tmp_path / "short-sanity" / "runpod_wam_async_delete_manifest.json"

    source_qa_path.write_text(
        json.dumps(
            {
                "status": "failed_visual_quality_gate",
                "blockers": ["source_policy_observation_too_dark_for_review"],
            }
        ),
        encoding="utf-8",
    )
    visual_report_path.write_text(
        json.dumps(
            {
                "status": "failed_visual_quality_gate",
                "visual_success": False,
                "structural_fallback_used": True,
                "blockers": [
                    "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout",
                    "wam_generated_frame_too_dark_for_review",
                ],
            }
        ),
        encoding="utf-8",
    )
    video_status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 2,
                "ffprobe_metadata": {},
            }
        ),
        encoding="utf-8",
    )
    review_video_path.write_bytes(b"")
    teardown_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "continuing_spend_from_this_run": True,
            }
        ),
        encoding="utf-8",
    )
    payload["paid_provider"] = {
        "provider": "runpod",
        "used": True,
        "teardown_status": "completed",
        "teardown_performed": True,
        "continuing_spend_from_this_run": False,
        "teardown_manifest_path": str(teardown_path),
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    validation = session.validate_persistent_wam_short_visual_sanity_manifest(
        manifest_path,
        policy_observation_path=observation_path,
    )

    assert validation["status"] == "blocked"
    assert "short_visual_sanity_source_qa_artifact_not_passed" in validation["blockers"]
    assert "source_policy_observation_too_dark_for_review" in validation["blockers"]
    assert "short_visual_sanity_quality_report_visual_success_not_passed" in validation[
        "blockers"
    ]
    assert "short_visual_sanity_quality_report_structural_fallback_used" in validation[
        "blockers"
    ]
    assert "autoregressive_chain_visual_drift_or_quality_blocked_long_rollout" in validation[
        "blockers"
    ]
    assert "wam_generated_frame_too_dark_for_review" in validation["blockers"]
    assert "short_visual_sanity_video_status_ffprobe_returncode_not_zero" in validation[
        "blockers"
    ]
    assert "short_visual_sanity_video_status_ffprobe_metadata_missing" in validation[
        "blockers"
    ]
    assert "short_visual_sanity_review_video_empty" in validation["blockers"]
    assert "short_visual_sanity_paid_provider_teardown_artifact_not_zero_spend" in validation[
        "blockers"
    ]


def test_runpod_persistent_session_clamps_tiny_oscar_frame_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_NUM_FRAMES", "3")
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["wam_num_frames"] = session.os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES")
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        runpod_dir = Path(kwargs["job_dir"])
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "output_zip_present": False,
                    "provider_command_status": "blocked",
                    "provider_command_blockers": [
                        "runpod_provider_runtime_output_zip_not_received_locally"
                    ],
                    "teardown_performed": True,
                    "continuing_spend_from_this_run": False,
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        }

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=2,
        timeout_seconds=60,
        max_wait_seconds=20,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert captured["wam_num_frames"] == "5"
    assert session.os.environ["BLUEPRINT_OSCAR_WAM_NUM_FRAMES"] == "3"


def test_runpod_persistent_session_launches_full_loop_without_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpg")
    observation_path = _policy_observation(tmp_path / "observation.json", frame)
    monkeypatch.delenv(session.RUNPOD_FULL_LOOP_OVERRIDE_ENV, raising=False)
    captured: dict[str, object] = {}

    def fake_stage(**kwargs):
        stage_dir = Path(kwargs["job_dir"])
        stage_dir.mkdir(parents=True)
        (stage_dir / "provider_bundle_url.txt").write_text("https://store.example/bundle.zip")
        (stage_dir / "provider_output_put_url.txt").write_text("https://store.example/out.zip?put")
        (stage_dir / "provider_output_get_url.txt").write_text("https://store.example/out.zip?get")
        return {"status": "completed", "blockers": []}

    def fake_create(**kwargs):
        captured["provider_bundle_kind"] = kwargs["provider_bundle_kind"]
        captured["loop_step_count"] = json.loads(
            (
                Path(kwargs["job_dir"]).parent
                / "provider_bundle"
                / "provider_runtime"
                / "persistent_session_input.json"
            ).read_text(encoding="utf-8")
        )["loop_step_count"]
        runpod_dir = Path(kwargs["job_dir"])
        runpod_dir.mkdir(parents=True, exist_ok=True)
        (runpod_dir / "runpod_wam_async_create_manifest.json").write_text(
            json.dumps({"status": "pod_created"}),
            encoding="utf-8",
        )
        return {"status": "pod_created", "pod_id": "pod-123"}

    def fake_poll(**kwargs):
        runpod_dir = Path(kwargs["job_dir"])
        poll_manifest = {
            "status": "running",
            "output_zip_present": False,
            "provider_command_status": "running",
            "provider_command_blockers": [],
            "teardown_performed": False,
            "continuing_spend_from_this_run": True,
        }
        (runpod_dir / "runpod_wam_async_poll_manifest.json").write_text(
            json.dumps(poll_manifest),
            encoding="utf-8",
        )
        return poll_manifest

    monkeypatch.setattr(session, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(session, "create_runpod_wam_async_run", fake_create)
    monkeypatch.setattr(session, "poll_runpod_wam_async_run", fake_poll)

    output, exit_code = session.run_persistent_session_runpod(
        policy_observation_path=observation_path,
        job_dir=tmp_path / "jobs",
        loop_step_count=12,
        timeout_seconds=60,
    )

    assert exit_code == 2
    assert output["status"] == "blocked"
    assert output["blockers"] == ["runpod_persistent_session_still_running"]
    assert output["details"]["poll_manifest"]["status"] == "running"
    assert captured["provider_bundle_kind"] == "wam"
    assert captured["loop_step_count"] == 12


def test_runpod_live_wam_blocker_classifies_missing_provider_artifact(tmp_path: Path) -> None:
    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_wrapper_or_upload_watchdog_no_valid_provider_artifact"
    )
    assert classification["evidence"]["output_zip_present"] is False
    assert (tmp_path / "runpod_live_wam_blocker_classification.json").is_file()


def test_runpod_live_wam_blocker_classifies_terminal_upload_after_heartbeat(
    tmp_path: Path,
) -> None:
    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(
                    tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
                ),
            },
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_terminal_output_upload_failed_after_remote_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_result_status"] == "running"


def test_runpod_live_wam_blocker_classifies_pod_gone_before_first_heartbeat(
    tmp_path: Path,
) -> None:
    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_before_first_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] is None


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_bootstrap(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "bootstrap_policy_server_started",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_policy_server_bootstrap_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "bootstrap_policy_server_started"
    )


def test_runpod_live_wam_blocker_classifies_pod_disappeared_after_model_snapshot(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_model_snapshot_completed",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_after_gr00t_model_snapshot_before_policy_server_ready"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "gr00t_model_snapshot_completed"
    )


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_policy_server_process(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_policy_server_waiting_for_listen",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_gr00t_policy_server_process_start_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "gr00t_policy_server_waiting_for_listen"
    )


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_uv_sync(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_uv_sync_started",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_gr00t_uv_sync_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == "gr00t_uv_sync_started"


def test_runpod_live_wam_blocker_classifies_pod_disappeared_during_system_python_deps(
    tmp_path: Path,
) -> None:
    nonterminal_zip = tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
    with zipfile.ZipFile(nonterminal_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "unitree_groot_n17_sonic_policy_provider_output.json",
            json.dumps(
                {
                    "status": "running",
                    "runtime_phase": "gr00t_system_python_minimal_deps_install_started",
                    "blockers": [],
                    "raw_secret_values_recorded": False,
                }
            ),
        )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "not_found",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(nonterminal_zip),
            },
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_pod_disappeared_during_gr00t_system_python_minimal_deps_install_after_heartbeat"
    )
    assert classification["evidence"]["last_nonterminal_runtime_phase"] == (
        "gr00t_system_python_minimal_deps_install_started"
    )


def test_runpod_live_wam_blocker_classifies_running_after_heartbeat_until_timeout(
    tmp_path: Path,
) -> None:
    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "blocked",
            "output_zip_present": False,
            "provider_command_status": "blocked",
            "provider_command_blockers": [
                "runpod_provider_runtime_output_zip_not_received_locally"
            ],
            "pod_status": "RUNNING",
            "last_nonterminal_output": {
                "status": "running",
                "runtime_result_status": "running",
                "nonterminal_zip_path": str(
                    tmp_path / "runpod_provider_runtime_output_nonterminal.zip"
                ),
            },
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
    )

    assert classification["status"] == "blocked"
    assert (
        classification["classified_blocker"]
        == "runpod_remote_runtime_still_running_after_heartbeat_until_local_timeout"
    )


def test_runpod_live_wam_blocker_classifies_policy_runtime_bootstrap_timeout(
    tmp_path: Path,
) -> None:
    extraction_dir = tmp_path / "imported_persistent_session_output"
    extraction_dir.mkdir()
    (extraction_dir / "runpod_unitree_groot_sonic_entrypoint_execution.json").write_text(
        json.dumps(
            {
                "status": "timed_out",
                "timed_out": True,
                "timeout_seconds": 240,
                "returncode": -15,
            }
        ),
        encoding="utf-8",
    )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "completed",
            "output_zip_present": True,
            "provider_command_status": "completed",
            "runtime_result_status": "blocked",
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
        extraction_dir=extraction_dir,
        imported={
            "status": "blocked",
            "blockers": ["persistent_session_entrypoint_exited_without_runtime_result"],
        },
    )

    assert classification["status"] == "blocked"
    assert classification["classified_blocker"] == "policy_runtime_bootstrap_timeout"
    assert classification["evidence"]["entrypoint_timed_out"] is True
    assert classification["evidence"]["entrypoint_timeout_seconds"] == 240


def test_runpod_live_wam_blocker_classifies_oscar_temporal_window(
    tmp_path: Path,
) -> None:
    extraction_dir = tmp_path / "imported_persistent_session_output"
    policy_dir = extraction_dir / "policy_calls"
    wam_dir = extraction_dir / "wam_calls"
    runtime_dir = extraction_dir / "wam_worker_steps" / "step_0001" / "oscar_runtime_output"
    policy_dir.mkdir(parents=True)
    wam_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    (policy_dir / "policy_call_0000.json").write_text(
        json.dumps({"status": "completed", "unitree_policy_action_command_ran": True}),
        encoding="utf-8",
    )
    (wam_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "blockers": [
                    "persistent_wam_worker_oscar_runtime_nonzero_exit",
                    "wam_output_missing_materializable_frame_or_video",
                ],
                "materialization": {"status": "blocked"},
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "wam_runtime_result.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "inference_detail": {
                    "stderr_tail_redacted": (
                        "worldsim/_src/tokenizers/wan2pt1.py RuntimeError: "
                        "Kernel size can't be greater than actual input size"
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "completed",
            "output_zip_present": True,
            "provider_command_status": "completed",
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
        extraction_dir=extraction_dir,
        imported={
            "status": "blocked",
            "required_policy_call_count": 2,
            "required_wam_transition_count": 1,
            "repeated_policy_calls_count": 1,
            "generated_next_observation_count": 0,
            "live_wam_generation_success_count": 1,
            "learned_wam_model_success_count": 0,
            "blockers": ["wam_output_missing_materializable_frame_or_video"],
        },
    )

    assert classification["status"] == "blocked"
    assert classification["classified_blocker"] == "oscar_wam_temporal_window_too_short"
    assert classification["evidence"]["oscar_temporal_tokenizer_blocked"] is True


def test_runpod_live_wam_blocker_classifies_frame_materialization(
    tmp_path: Path,
) -> None:
    extraction_dir = tmp_path / "imported_persistent_session_output"
    policy_dir = extraction_dir / "policy_calls"
    wam_dir = extraction_dir / "wam_calls"
    policy_dir.mkdir(parents=True)
    wam_dir.mkdir(parents=True)
    (policy_dir / "policy_call_0000.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "unitree_policy_action_command_ran": True,
                "provider_output_replay_used": False,
            }
        ),
        encoding="utf-8",
    )
    (wam_dir / "wam_call_0001.json").write_text(
        json.dumps(
            {
                "status": "blocked",
                "blockers": ["wam_output_missing_materializable_frame_or_video"],
                "materialization": {"status": "blocked"},
            }
        ),
        encoding="utf-8",
    )

    classification = session._write_runpod_live_wam_blocker_classification(
        job=tmp_path,
        generated_at="now",
        poll_manifest={
            "status": "completed",
            "output_zip_present": True,
            "provider_command_status": "completed",
            "teardown_performed": True,
            "continuing_spend_from_this_run": False,
        },
        extraction_dir=extraction_dir,
        imported={
            "status": "blocked",
            "required_policy_call_count": 2,
            "required_wam_transition_count": 1,
            "repeated_policy_calls_count": 1,
            "generated_next_observation_count": 0,
            "live_wam_generation_success_count": 0,
            "learned_wam_model_success_count": 0,
            "blockers": ["wam_output_missing_materializable_frame_or_video"],
        },
    )

    assert classification["status"] == "blocked"
    assert classification["classified_blocker"] == "wam_frame_materialization_blocked"
    assert classification["evidence"]["policy_call_artifact_count"] == 1
    assert classification["evidence"]["wam_call_artifact_count"] == 1
