from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import new_site_diagnostic_canary_gpu as canary_module
from blueprint_pipeline.droid_oscar_closed_loop_adapter import EXTERIOR_VIEW, WRIST_VIEW
from blueprint_pipeline.droid_policy_bridge import DROID_EXTERIOR_VIEW_2
from blueprint_pipeline.new_site_diagnostic_canary_gpu import (
    MultiViewCanaryReliabilityGate,
    build_canary_input_bundle,
    extract_canary_input_bundle,
    materialize_canary_background,
    run_ctrl_world_canary,
    run_oscar_canary,
    run_skeleton_only_canary,
)
from blueprint_pipeline.new_site_diagnostic_smoke import build_protocol
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256
from blueprint_pipeline.wam_rollout_reliability import (
    FLAG_STATIC_UNDER_COMMAND,
    ReliabilityThresholds,
    RolloutReliabilityReport,
)


ROOT = Path(__file__).resolve().parents[1]


def _active_reliability_actions() -> np.ndarray:
    actions = np.zeros((8, 10), dtype=float)
    actions[:, 0] = 0.1
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    return actions


def _write_wrist_trace_evidence(tmp_path: Path, positions: list[list[float]]) -> dict:
    trace_path = tmp_path / "wrist_skeleton.jsonl"
    rows = []
    for frame_index, position in enumerate(positions):
        row = {
            "schema_version": "franka_droid_skeleton_projection_frame.v1",
            "episode_id": "query_000",
            "view_id": WRIST_VIEW,
            "frame_index": frame_index,
            "source_controller_horizon_frame_index": frame_index,
            "source_width": 640,
            "source_height": 480,
            "landmarks": [
                {
                    "landmark_id": "wrist_action_center",
                    "reference_position_m": position,
                    "camera_position_m": [0.0, 0.0, 0.25],
                    "image_projection": {
                        "available": True,
                        "u_px": 319.5,
                        "v_px": 239.5,
                        "positive_depth": True,
                        "in_image_bounds": True,
                    },
                }
            ],
            "segments": [],
            "projected_landmark_count": 1,
            "physical_future_observation_used": False,
        }
        row["frame_sha256"] = canonical_sha256(row)
        rows.append(row)
    trace_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    evidence = {
        "schema_version": "franka_droid_skeleton_conditioning.v1",
        "trace_evidence": {
            WRIST_VIEW: {
                "trace_path": str(trace_path),
                "trace_sha256": file_sha256(trace_path),
                "frame_count": len(rows),
            }
        },
        "physical_future_observation_used": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return evidence


def _reliability_report(video_path: Path, *, flags: tuple[str, ...]) -> RolloutReliabilityReport:
    return RolloutReliabilityReport(
        video_path=str(video_path),
        n_frames=81,
        n_action_steps=8,
        flags=flags,
        reliable=not flags,
        command_energy_mean=0.2,
        command_energy_std=0.1,
        motion_mean=0.002 if flags else 0.2,
        motion_max=0.008 if flags else 0.3,
        timing_correlation=0.5,
        timing_flag_scope="session",
        spatial_std_mean=20.0,
    )


def _write_protocol(path: Path) -> dict:
    protocol = build_protocol(ROOT, experiment_id="new_site_canary_test_v1")
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return protocol


def test_background_materialization_is_deterministic_and_does_not_crop(tmp_path: Path) -> None:
    source = tmp_path / "task_focus.png"
    Image.new("RGBA", (1920, 1440), color=(12, 34, 56, 255)).save(source)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"

    first_receipt = materialize_canary_background(source_path=source, output_path=first)
    second_receipt = materialize_canary_background(source_path=source, output_path=second)

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["output_sha256"] == second_receipt["output_sha256"]
    assert first_receipt["source_size_px"] == [1920, 1440]
    assert first_receipt["crop_applied"] is False
    assert first_receipt["camera_selection_changed"] is False


def test_canary_input_roundtrip_is_protocol_bound_and_label_free(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(12, 34, 56)).save(background)
    bundle = tmp_path / "canary.zip"

    receipt = build_canary_input_bundle(
        protocol_path=protocol_path,
        background_path=background,
        output_zip=bundle,
        arm_id="skeleton_only",
    )
    extracted = extract_canary_input_bundle(
        bundle_path=bundle,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )

    assert extracted["manifest"]["protocol_sha256"] == protocol["protocol_sha256"]
    assert extracted["manifest"]["label_free"] is True
    assert extracted["manifest"]["policy_id"] == protocol["canary_rule"]["frozen_policy_id"]
    assert Path(extracted["background_path"]).read_bytes() == background.read_bytes()

    with pytest.raises(ValueError, match="input_sha256_mismatch"):
        extract_canary_input_bundle(
            bundle_path=bundle,
            expected_bundle_sha256="0" * 64,
            output_dir=tmp_path / "tampered",
        )


def test_canary_input_roundtrip_carries_passed_native_initial_camera_frames(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(12, 34, 56)).save(background)
    frame_entries = {}
    for view_key, view_id, color in (
        ("external", EXTERIOR_VIEW, (20, 40, 60)),
        ("wrist", WRIST_VIEW, (60, 40, 20)),
    ):
        path = tmp_path / f"native_{view_key}.png"
        Image.new("RGB", (640, 480), color=color).save(path)
        frame_entries[view_key] = {
            "frames": {
                "initial": {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "resolution": [640, 480],
                    "nonblank": True,
                }
            }
        }
    native_result = {
        "status": "passed",
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "assessment": {"views": frame_entries},
    }
    native_result["result_sha256"] = canonical_sha256(native_result)
    native_result_path = tmp_path / "native_result.json"
    native_result_path.write_text(json.dumps(native_result), encoding="utf-8")

    receipt = build_canary_input_bundle(
        protocol_path=protocol_path,
        background_path=background,
        output_zip=tmp_path / "native_canary.zip",
        arm_id="skeleton_only",
        native_camera_canary_result_path=native_result_path,
    )
    extracted = extract_canary_input_bundle(
        bundle_path=tmp_path / "native_canary.zip",
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "native_extracted",
    )

    assert set(extracted["initial_camera_paths"]) == {EXTERIOR_VIEW, WRIST_VIEW}
    assert receipt["manifest"]["initial_observation_source"] == (
        "native_isaac_simready_warehouse_camera_canary"
    )
    for path in extracted["initial_camera_paths"].values():
        assert Image.open(path).size == (640, 480)


def test_ctrl_world_input_roundtrip_requires_and_carries_three_native_views(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(12, 34, 56)).save(background)
    frame_entries = {}
    for view_key, color in (
        ("external", (20, 40, 60)),
        ("external_2", (30, 50, 70)),
        ("wrist", (60, 40, 20)),
    ):
        path = tmp_path / f"native_{view_key}.png"
        Image.new("RGB", (640, 480), color=color).save(path)
        frame_entries[view_key] = {
            "frames": {
                "initial": {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "resolution": [640, 480],
                    "nonblank": True,
                }
            }
        }
    native_result = {
        "status": "passed",
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "assessment": {
            "required_views": ["external", "external_2", "wrist"],
            "views": frame_entries,
        },
    }
    native_result["result_sha256"] = canonical_sha256(native_result)
    native_result_path = tmp_path / "native_result.json"
    native_result_path.write_text(json.dumps(native_result), encoding="utf-8")

    receipt = build_canary_input_bundle(
        protocol_path=protocol_path,
        background_path=background,
        output_zip=tmp_path / "ctrl_world_canary.zip",
        arm_id="ctrl_world",
        native_camera_canary_result_path=native_result_path,
    )
    extracted = extract_canary_input_bundle(
        bundle_path=tmp_path / "ctrl_world_canary.zip",
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "ctrl_world_extracted",
    )

    assert set(extracted["initial_camera_paths"]) == {
        EXTERIOR_VIEW,
        DROID_EXTERIOR_VIEW_2,
        WRIST_VIEW,
    }
    assert set(receipt["manifest"]["native_initial_cameras"]) == {
        EXTERIOR_VIEW,
        DROID_EXTERIOR_VIEW_2,
        WRIST_VIEW,
    }
    assert receipt["manifest"]["wam_seed"] == 23
    assert extracted["manifest"]["wam_seed"] == 23


def test_oscar_input_roundtrip_requires_two_native_views_and_frozen_seed(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(12, 34, 56)).save(background)
    frame_entries = {}
    for view_key, color in (
        ("external", (20, 40, 60)),
        ("wrist", (60, 40, 20)),
    ):
        path = tmp_path / f"native_{view_key}.png"
        Image.new("RGB", (640, 480), color=color).save(path)
        frame_entries[view_key] = {
            "frames": {
                "initial": {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "resolution": [640, 480],
                    "nonblank": True,
                }
            }
        }
    native_result = {
        "status": "passed",
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "assessment": {"required_views": ["external", "wrist"], "views": frame_entries},
    }
    native_result["result_sha256"] = canonical_sha256(native_result)
    native_result_path = tmp_path / "native_result.json"
    native_result_path.write_text(json.dumps(native_result), encoding="utf-8")

    receipt = build_canary_input_bundle(
        protocol_path=protocol_path,
        background_path=background,
        output_zip=tmp_path / "oscar_canary.zip",
        arm_id="oscar",
        native_camera_canary_result_path=native_result_path,
    )
    extracted = extract_canary_input_bundle(
        bundle_path=tmp_path / "oscar_canary.zip",
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "oscar_extracted",
    )

    assert set(extracted["initial_camera_paths"]) == {EXTERIOR_VIEW, WRIST_VIEW}
    assert receipt["manifest"]["wam_seed"] == 42
    assert extracted["manifest"]["wam_seed"] == 42


def test_oscar_input_rejects_missing_native_two_view_result(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224)).save(background)

    with pytest.raises(ValueError, match="native_two_view_result_required"):
        build_canary_input_bundle(
            protocol_path=protocol_path,
            background_path=background,
            output_zip=tmp_path / "oscar_canary.zip",
            arm_id="oscar",
        )


def test_ctrl_world_input_rejects_missing_native_three_view_result(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224)).save(background)

    with pytest.raises(ValueError, match="native_three_view_result_required"):
        build_canary_input_bundle(
            protocol_path=protocol_path,
            background_path=background,
            output_zip=tmp_path / "ctrl_world_canary.zip",
            arm_id="ctrl_world",
        )


def test_native_initial_camera_loader_preserves_distinct_ctrl_world_views(
    tmp_path: Path,
) -> None:
    paths = {}
    expected_means = {}
    for view_index, view_id in enumerate((EXTERIOR_VIEW, DROID_EXTERIOR_VIEW_2, WRIST_VIEW)):
        color = 20 + view_index * 30
        path = tmp_path / f"native_{view_index}.png"
        Image.new("RGB", (640, 480), color=(color,) * 3).save(path)
        paths[view_id] = str(path)
        expected_means[view_id] = color

    loaded = canary_module._load_native_initial_camera_views(paths, np_module=np)

    assert set(loaded) == set(paths)
    assert all(image.shape == (224, 224, 3) for image in loaded.values())
    assert {view_id: int(np.mean(image)) for view_id, image in loaded.items()} == expected_means


def test_canary_input_accepts_portable_native_camera_relative_paths(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.json"
    _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224), color=(12, 34, 56)).save(background)
    frame_entries = {}
    for view_key, view_id, color in (
        ("external", EXTERIOR_VIEW, (20, 40, 60)),
        ("wrist", WRIST_VIEW, (60, 40, 20)),
    ):
        path = tmp_path / "runtime" / f"{view_key}_initial.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (640, 480), color=color).save(path)
        frame_entries[view_key] = {
            "frames": {
                "initial": {
                    "path": f"/provider/ephemeral/{view_key}_initial.png",
                    "relative_path": f"runtime/{view_key}_initial.png",
                    "sha256": file_sha256(path),
                    "resolution": [640, 480],
                    "nonblank": True,
                }
            }
        }
    native_result = {
        "status": "passed",
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "assessment": {"views": frame_entries},
    }
    native_result["result_sha256"] = canonical_sha256(native_result)
    native_result_path = tmp_path / "native_camera_canary_result.json"
    native_result_path.write_text(json.dumps(native_result), encoding="utf-8")

    receipt = build_canary_input_bundle(
        protocol_path=protocol_path,
        background_path=background,
        output_zip=tmp_path / "native_canary.zip",
        arm_id="skeleton_only",
        native_camera_canary_result_path=native_result_path,
    )

    assert receipt["manifest"]["initial_observation_source"] == (
        "native_isaac_simready_warehouse_camera_canary"
    )


def test_canary_input_rejects_mutated_protocol_identity(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path)
    protocol["scene"]["task_instruction"] = "mutated after freeze"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224)).save(background)

    with pytest.raises(ValueError, match="protocol_sha256_invalid"):
        build_canary_input_bundle(
            protocol_path=protocol_path,
            background_path=background,
            output_zip=tmp_path / "invalid.zip",
            arm_id="skeleton_only",
        )


def test_multiview_gate_checks_both_individual_camera_videos(tmp_path: Path) -> None:
    videos = {}
    for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
        path = tmp_path / f"{view_id.rsplit('/', 1)[-1]}.mp4"
        path.write_bytes(b"test-video")
        videos[view_id] = str(path)
    calls = []

    def assessor(video_path, actions, thresholds, **kwargs):
        calls.append((Path(video_path), np.asarray(actions).shape, thresholds, kwargs))
        return RolloutReliabilityReport(
            video_path=str(video_path),
            n_frames=9,
            n_action_steps=8,
            flags=(),
            reliable=True,
            command_energy_mean=0.2,
            command_energy_std=0.1,
            motion_mean=0.2,
            motion_max=0.3,
            timing_correlation=0.5,
            timing_flag_scope="session",
            spatial_std_mean=20.0,
        )

    thresholds = ReliabilityThresholds()
    result = MultiViewCanaryReliabilityGate(thresholds, assessor=assessor).assess(
        previous_observation={},
        prepared_transition={"reliability_actions_10d": np.zeros((8, 10))},
        wam_prediction={"generated_videos_by_view": videos},
        query_index=0,
        output_dir=tmp_path,
    )

    assert result["status"] == "passed"
    assert result["label_free"] is True
    assert set(result["reports_by_view"]) == {EXTERIOR_VIEW, WRIST_VIEW}
    assert {call[0] for call in calls} == {Path(path) for path in videos.values()}


def test_multiview_gate_prefers_exact_generated_frames_over_lossy_video(
    tmp_path: Path,
) -> None:
    videos = {}
    sequences = {}
    for view_index, view_id in enumerate((EXTERIOR_VIEW, WRIST_VIEW)):
        video = tmp_path / f"view_{view_index}.mp4"
        video.write_bytes(b"retained-media")
        videos[view_id] = str(video)
        sequences[view_id] = []
        for frame_index in range(5):
            path = tmp_path / f"view_{view_index}_frame_{frame_index}.png"
            Image.new("RGB", (320, 192), color=(view_index, frame_index, 20)).save(path)
            sequences[view_id].append(str(path))
    calls = []

    def frame_assessor(frame_paths, actions, thresholds, **kwargs):
        calls.append((list(frame_paths), np.asarray(actions).shape, thresholds, kwargs))
        return RolloutReliabilityReport(
            video_path="frame_sequence",
            n_frames=len(frame_paths),
            n_action_steps=8,
            flags=(),
            reliable=True,
            command_energy_mean=0.2,
            command_energy_std=0.1,
            motion_mean=0.2,
            motion_max=0.3,
            timing_correlation=0.5,
            timing_flag_scope="session",
            spatial_std_mean=20.0,
        )

    result = MultiViewCanaryReliabilityGate(
        ReliabilityThresholds(),
        assessor=lambda *_args, **_kwargs: pytest.fail("video assessor must not run"),
        frame_assessor=frame_assessor,
    ).assess(
        previous_observation={},
        prepared_transition={"reliability_actions_10d": np.zeros((8, 10))},
        wam_prediction={
            "generated_videos_by_view": videos,
            "generated_view_frame_sequences": sequences,
        },
        query_index=0,
        output_dir=tmp_path,
    )

    assert result["status"] == "passed"
    assert len(calls) == 2
    assert all(len(call[0]) == 5 for call in calls)
    assert all(
        report["motion_evidence_basis"] == "exact_generated_frame_sequence"
        and report["lossy_video_used_for_motion_measurement"] is False
        for report in result["reports_by_view"].values()
    )


def test_skeleton_wrist_uses_hash_verified_world_motion_not_rigid_camera_pixels(
    tmp_path: Path,
) -> None:
    videos = {}
    for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
        path = tmp_path / f"{view_id.rsplit('/', 1)[-1]}.mp4"
        path.write_bytes(b"test-video")
        videos[view_id] = str(path)

    def assessor(video_path, *_args, **_kwargs):
        path = Path(video_path)
        return _reliability_report(
            path,
            flags=(FLAG_STATIC_UNDER_COMMAND,) if "wrist" in path.name else (),
        )

    evidence = _write_wrist_trace_evidence(
        tmp_path,
        [[0.50, 0.00, 0.10], [0.52, 0.00, 0.10]],
    )
    result = MultiViewCanaryReliabilityGate(ReliabilityThresholds(), assessor=assessor).assess(
        previous_observation={},
        prepared_transition={
            "reliability_actions_10d": _active_reliability_actions(),
            "conditioning_builder_evidence": evidence,
        },
        wam_prediction={
            "generated_videos_by_view": videos,
            "skeleton_only": True,
            "intended_motion_only": True,
        },
        query_index=0,
        output_dir=tmp_path,
    )

    wrist = result["reports_by_view"][WRIST_VIEW]
    assert result["status"] == "passed"
    assert wrist["raw_pixel_flags"] == [FLAG_STATIC_UNDER_COMMAND]
    assert wrist["flags"] == []
    assert wrist["motion_basis"] == "hash_verified_reference_frame_fk_trace"
    assert wrist["kinematic_motion"]["maximum_reference_displacement_m"] == pytest.approx(0.02)


def test_skeleton_wrist_still_fails_when_fk_trace_is_stationary(tmp_path: Path) -> None:
    videos = {}
    for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
        path = tmp_path / f"{view_id.rsplit('/', 1)[-1]}.mp4"
        path.write_bytes(b"test-video")
        videos[view_id] = str(path)

    def assessor(video_path, *_args, **_kwargs):
        path = Path(video_path)
        return _reliability_report(
            path,
            flags=(FLAG_STATIC_UNDER_COMMAND,) if "wrist" in path.name else (),
        )

    evidence = _write_wrist_trace_evidence(
        tmp_path,
        [[0.50, 0.00, 0.10], [0.50, 0.00, 0.10]],
    )
    result = MultiViewCanaryReliabilityGate(ReliabilityThresholds(), assessor=assessor).assess(
        previous_observation={},
        prepared_transition={
            "reliability_actions_10d": _active_reliability_actions(),
            "conditioning_builder_evidence": evidence,
        },
        wam_prediction={
            "generated_videos_by_view": videos,
            "skeleton_only": True,
            "intended_motion_only": True,
        },
        query_index=0,
        output_dir=tmp_path,
    )

    assert result["status"] == "failed"
    assert result["reasons"] == [f"{WRIST_VIEW}:kinematic_static_under_command"]


def test_skeleton_wrist_trace_tampering_fails_closed(tmp_path: Path) -> None:
    videos = {}
    for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
        path = tmp_path / f"{view_id.rsplit('/', 1)[-1]}.mp4"
        path.write_bytes(b"test-video")
        videos[view_id] = str(path)
    evidence = _write_wrist_trace_evidence(
        tmp_path,
        [[0.50, 0.00, 0.10], [0.52, 0.00, 0.10]],
    )
    trace_path = Path(evidence["trace_evidence"][WRIST_VIEW]["trace_path"])
    trace_path.write_text(trace_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="wrist_trace_sha256_mismatch"):
        MultiViewCanaryReliabilityGate(
            ReliabilityThresholds(),
            assessor=lambda video_path, *_args, **_kwargs: _reliability_report(
                Path(video_path), flags=()
            ),
        ).assess(
            previous_observation={},
            prepared_transition={
                "reliability_actions_10d": _active_reliability_actions(),
                "conditioning_builder_evidence": evidence,
            },
            wam_prediction={
                "generated_videos_by_view": videos,
                "skeleton_only": True,
                "intended_motion_only": True,
            },
            query_index=0,
            output_dir=tmp_path,
        )


def test_skeleton_canary_runs_two_policy_wam_transitions_without_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224)).save(background)
    native_external = tmp_path / "native_external.png"
    native_wrist = tmp_path / "native_wrist.png"
    Image.new("RGB", (640, 480), color=(20, 30, 40)).save(native_external)
    Image.new("RGB", (640, 480), color=(50, 60, 70)).save(native_wrist)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = SimpleNamespace(
        checkpoint_uri="s3://example/checkpoint",
        action_chunk_rows=15,
    )
    monkeypatch.setattr(canary_module, "load_policy_spec", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(
        canary_module, "verify_local_checkpoint", lambda **_kwargs: {"status": "verified"}
    )
    monkeypatch.setattr(canary_module, "LocalOpenPIDroidPolicyClient", lambda **_kwargs: object())
    monkeypatch.setattr(canary_module, "prepare_franka_droid_runtime", lambda **_kwargs: {})
    monkeypatch.setattr(
        canary_module,
        "_initial_observation",
        lambda **_kwargs: {
            EXTERIOR_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
            WRIST_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7),
            "observation/gripper_position": np.zeros(1),
            "prompt": protocol["scene"]["task_instruction"],
            "_diagnostic_interaction_pixel_count": 10,
        },
    )

    def fake_loop(**kwargs):
        assert kwargs["config"].max_policy_queries == 2
        initial_sha256 = canary_module.policy_observation_sha256(kwargs["initial_observation"])
        wam_observation_sha256 = "2" * 64
        trace = Path(kwargs["output_dir"]) / "closed_loop_trace.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(
            "".join(
                json.dumps(
                    {
                        "schema_version": "policy_wam_closed_loop_trace.v1",
                        "query_index": index - 1,
                        "status": "completed",
                        "policy_observation_sha256": (
                            initial_sha256 if index == 1 else wam_observation_sha256
                        ),
                        "next_observation_sha256": (
                            wam_observation_sha256 if index == 1 else "3" * 64
                        ),
                        "next_observation_provenance": {"visual_source": "wam_prediction"},
                        "reliability": {"status": "passed", "reasons": []},
                    }
                )
                + "\n"
                for index in (1, 2)
            ),
            encoding="utf-8",
        )
        return {
            "status": "max_horizon",
            "trace_path": str(trace),
            "policy_call_count": 2,
            "wam_call_count": 2,
            "initial_observation_sha256": initial_sha256,
            "trace_sha256": file_sha256(trace),
            "blockers": [],
        }

    monkeypatch.setattr(canary_module, "run_policy_wam_closed_loop", fake_loop)

    result = run_skeleton_only_canary(
        protocol_path=protocol_path,
        background_path=background,
        cohort_path=tmp_path / "cohort.json",
        checkpoint_inventory_path=tmp_path / "inventory.json",
        menagerie_root=tmp_path / "menagerie",
        output_dir=tmp_path / "output",
        initial_camera_paths={
            EXTERIOR_VIEW: native_external,
            WRIST_VIEW: native_wrist,
        },
        checkpoint_downloader=lambda _uri: checkpoint,
        policy_loader=lambda _spec, _checkpoint: object(),
    )

    assert result["status"] == "completed"
    assert result["canary"]["status"] == "passed"
    assert result["loop_manifest"]["policy_call_count"] == 2
    assert result["loop_manifest"]["wam_call_count"] == 2
    assert result["policy_wam_policy_round_trip_passed"] is True
    assert result["first_wam_observation_sha256"] == "2" * 64
    assert result["second_policy_observation_sha256"] == "2" * 64
    assert result["initial_camera_sha256_by_view"] == {
        EXTERIOR_VIEW: file_sha256(native_external),
        WRIST_VIEW: file_sha256(native_wrist),
    }
    assert result["initial_observation_source"] == ("native_isaac_simready_warehouse_camera_canary")
    assert "rankings" not in result
    assert result["claim_boundary"]["physical_success"] is False


def test_ctrl_world_canary_wires_three_view_exact_round_trip_without_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224)).save(background)
    initial_camera_paths = {}
    for view_index, view_id in enumerate((EXTERIOR_VIEW, DROID_EXTERIOR_VIEW_2, WRIST_VIEW)):
        path = tmp_path / f"native_{view_index}.png"
        Image.new("RGB", (640, 480), color=(20 + view_index * 20,) * 3).save(path)
        initial_camera_paths[view_id] = str(path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = SimpleNamespace(checkpoint_uri="s3://example/checkpoint", action_chunk_rows=15)
    monkeypatch.setattr(canary_module, "load_policy_spec", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(
        canary_module, "verify_local_checkpoint", lambda **_kwargs: {"status": "verified"}
    )
    monkeypatch.setattr(canary_module, "LocalOpenPIDroidPolicyClient", lambda **_kwargs: object())
    monkeypatch.setattr(canary_module, "prepare_franka_droid_runtime", lambda **_kwargs: {})
    monkeypatch.setattr(
        canary_module,
        "_initial_observation",
        lambda **_kwargs: {
            EXTERIOR_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
            DROID_EXTERIOR_VIEW_2: np.ones((224, 224, 3), dtype=np.uint8),
            WRIST_VIEW: np.full((224, 224, 3), 2, dtype=np.uint8),
            "observation/joint_position": np.zeros(7),
            "observation/gripper_position": np.zeros(1),
            "prompt": protocol["scene"]["task_instruction"],
            "_diagnostic_interaction_pixel_count": 10,
        },
    )

    def fake_loop(**kwargs):
        assert kwargs["config"].max_policy_queries == 2
        assert kwargs["wam_arm"].seed == 0
        assert kwargs["reliability_gate"].required_views == (
            DROID_EXTERIOR_VIEW_2,
            EXTERIOR_VIEW,
            WRIST_VIEW,
        )
        initial_sha256 = canary_module.policy_observation_sha256(kwargs["initial_observation"])
        wam_observation_sha256 = "2" * 64
        trace = Path(kwargs["output_dir"]) / "closed_loop_trace.jsonl"
        trace.parent.mkdir(parents=True)
        rows = [
            {
                "schema_version": "policy_wam_closed_loop_trace.v1",
                "query_index": 0,
                "status": "completed",
                "policy_observation_sha256": initial_sha256,
                "next_observation_sha256": wam_observation_sha256,
                "next_observation_provenance": {"visual_source": "wam_prediction"},
                "reliability": {"status": "passed", "reasons": []},
            },
            {
                "schema_version": "policy_wam_closed_loop_trace.v1",
                "query_index": 1,
                "status": "completed",
                "policy_observation_sha256": wam_observation_sha256,
                "next_observation_sha256": "3" * 64,
                "next_observation_provenance": {"visual_source": "wam_prediction"},
                "reliability": {"status": "passed", "reasons": []},
            },
        ]
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {
            "status": "max_horizon",
            "trace_path": str(trace),
            "trace_sha256": file_sha256(trace),
            "policy_call_count": 2,
            "wam_call_count": 2,
            "initial_observation_sha256": initial_sha256,
            "blockers": [],
        }

    monkeypatch.setattr(canary_module, "run_policy_wam_closed_loop", fake_loop)

    result = run_ctrl_world_canary(
        protocol_path=protocol_path,
        background_path=background,
        cohort_path=tmp_path / "cohort.json",
        checkpoint_inventory_path=tmp_path / "inventory.json",
        menagerie_root=tmp_path / "menagerie",
        output_dir=tmp_path / "output",
        initial_camera_paths=initial_camera_paths,
        ctrl_world_runner=lambda **_kwargs: pytest.fail("mock loop must not invoke runner"),
        seed=0,
        checkpoint_downloader=lambda _uri: checkpoint,
        policy_loader=lambda _spec, _checkpoint: object(),
    )

    assert result["status"] == "completed"
    assert result["arm_id"] == "ctrl_world"
    assert result["policy_wam_policy_round_trip_passed"] is True
    assert set(result["initial_camera_sha256_by_view"]) == set(initial_camera_paths)
    assert "rankings" not in result
    assert result["claim_boundary"]["physical_success"] is False


def test_oscar_canary_wires_two_view_learned_wam_exact_round_trip_without_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224)).save(background)
    initial_camera_paths = {}
    for view_index, view_id in enumerate((EXTERIOR_VIEW, WRIST_VIEW)):
        path = tmp_path / f"native_{view_index}.png"
        Image.new("RGB", (640, 480), color=(20 + view_index * 20,) * 3).save(path)
        initial_camera_paths[view_id] = str(path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = SimpleNamespace(checkpoint_uri="s3://example/checkpoint", action_chunk_rows=15)
    monkeypatch.setattr(canary_module, "load_policy_spec", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(
        canary_module, "verify_local_checkpoint", lambda **_kwargs: {"status": "verified"}
    )
    monkeypatch.setattr(canary_module, "LocalOpenPIDroidPolicyClient", lambda **_kwargs: object())
    monkeypatch.setattr(canary_module, "prepare_franka_droid_runtime", lambda **_kwargs: {})
    monkeypatch.setattr(
        canary_module,
        "_initial_observation",
        lambda **_kwargs: {
            EXTERIOR_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
            WRIST_VIEW: np.full((224, 224, 3), 2, dtype=np.uint8),
            "observation/joint_position": np.zeros(7),
            "observation/gripper_position": np.zeros(1),
            "prompt": protocol["scene"]["task_instruction"],
            "_diagnostic_interaction_pixel_count": 10,
        },
    )
    observed_seeds: list[int] = []

    def fake_generator(**kwargs):
        observed_seeds.append(kwargs["seed"])
        return {"generated_video_path": tmp_path / "unused.mp4"}

    def fake_loop(**kwargs):
        assert kwargs["config"].max_policy_queries == 2
        assert kwargs["reliability_gate"].required_views == (EXTERIOR_VIEW, WRIST_VIEW)
        assert kwargs["wam_arm"].arm_id == "oscar_purpose_built_wam_multiview"
        kwargs["wam_arm"].generator(
            view_id=EXTERIOR_VIEW,
            view_request={},
            task_prompt="task",
            negative_prompt="negative",
            output_dir=tmp_path,
        )
        initial_sha256 = canary_module.policy_observation_sha256(kwargs["initial_observation"])
        wam_observation_sha256 = "2" * 64
        trace = Path(kwargs["output_dir"]) / "closed_loop_trace.jsonl"
        trace.parent.mkdir(parents=True)
        rows = [
            {
                "schema_version": "policy_wam_closed_loop_trace.v1",
                "query_index": 0,
                "status": "completed",
                "policy_observation_sha256": initial_sha256,
                "next_observation_sha256": wam_observation_sha256,
                "next_observation_provenance": {"visual_source": "wam_prediction"},
                "reliability": {"status": "passed", "reasons": []},
            },
            {
                "schema_version": "policy_wam_closed_loop_trace.v1",
                "query_index": 1,
                "status": "completed",
                "policy_observation_sha256": wam_observation_sha256,
                "next_observation_sha256": "3" * 64,
                "next_observation_provenance": {"visual_source": "wam_prediction"},
                "reliability": {"status": "passed", "reasons": []},
            },
        ]
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {
            "status": "max_horizon",
            "trace_path": str(trace),
            "trace_sha256": file_sha256(trace),
            "policy_call_count": 2,
            "wam_call_count": 2,
            "initial_observation_sha256": initial_sha256,
            "blockers": [],
        }

    monkeypatch.setattr(canary_module, "run_policy_wam_closed_loop", fake_loop)

    result = run_oscar_canary(
        protocol_path=protocol_path,
        background_path=background,
        cohort_path=tmp_path / "cohort.json",
        checkpoint_inventory_path=tmp_path / "inventory.json",
        menagerie_root=tmp_path / "menagerie",
        output_dir=tmp_path / "output",
        initial_camera_paths=initial_camera_paths,
        oscar_generator=fake_generator,
        seed=42,
        checkpoint_downloader=lambda _uri: checkpoint,
        policy_loader=lambda _spec, _checkpoint: object(),
    )

    assert observed_seeds == [42]
    assert result["status"] == "completed"
    assert result["arm_id"] == "oscar"
    assert result["wam_arm_id"] == "oscar_purpose_built_wam_multiview"
    assert result["learned_wam_invoked"] is True
    assert result["policy_wam_policy_round_trip_passed"] is True
    assert set(result["initial_camera_sha256_by_view"]) == set(initial_camera_paths)
    assert "rankings" not in result
    assert result["claim_boundary"]["physical_success"] is False


@pytest.mark.parametrize("fault", ["provenance", "hash_link"])
def test_skeleton_canary_abstains_on_unattributable_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path)
    background = tmp_path / "background.png"
    Image.new("RGB", (224, 224)).save(background)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    spec = SimpleNamespace(checkpoint_uri="s3://example/checkpoint", action_chunk_rows=15)
    monkeypatch.setattr(canary_module, "load_policy_spec", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(
        canary_module, "verify_local_checkpoint", lambda **_kwargs: {"status": "verified"}
    )
    monkeypatch.setattr(canary_module, "LocalOpenPIDroidPolicyClient", lambda **_kwargs: object())
    monkeypatch.setattr(canary_module, "prepare_franka_droid_runtime", lambda **_kwargs: {})
    monkeypatch.setattr(
        canary_module,
        "_initial_observation",
        lambda **_kwargs: {
            EXTERIOR_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
            WRIST_VIEW: np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7),
            "observation/gripper_position": np.zeros(1),
            "prompt": protocol["scene"]["task_instruction"],
            "_diagnostic_interaction_pixel_count": 10,
        },
    )

    def malformed_loop(**kwargs):
        initial_sha256 = canary_module.policy_observation_sha256(kwargs["initial_observation"])
        wam_observation_sha256 = "2" * 64
        trace = Path(kwargs["output_dir"]) / "closed_loop_trace.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(
            "".join(
                json.dumps(
                    {
                        "schema_version": "policy_wam_closed_loop_trace.v1",
                        "query_index": index - 1,
                        "status": "completed",
                        "policy_observation_sha256": (
                            initial_sha256
                            if index == 1
                            else ("4" * 64 if fault == "hash_link" else wam_observation_sha256)
                        ),
                        "next_observation_sha256": (
                            wam_observation_sha256 if index == 1 else "3" * 64
                        ),
                        "next_observation_provenance": (
                            "not-an-object"
                            if index == 1 and fault == "provenance"
                            else {"visual_source": "wam_prediction"}
                        ),
                        "reliability": {"status": "passed", "reasons": []},
                    }
                )
                + "\n"
                for index in (1, 2)
            ),
            encoding="utf-8",
        )
        return {
            "status": "max_horizon",
            "trace_path": str(trace),
            "policy_call_count": 2,
            "wam_call_count": 2,
            "initial_observation_sha256": initial_sha256,
            "trace_sha256": file_sha256(trace),
            "blockers": [],
        }

    monkeypatch.setattr(canary_module, "run_policy_wam_closed_loop", malformed_loop)

    result = run_skeleton_only_canary(
        protocol_path=protocol_path,
        background_path=background,
        cohort_path=tmp_path / "cohort.json",
        checkpoint_inventory_path=tmp_path / "inventory.json",
        menagerie_root=tmp_path / "menagerie",
        output_dir=tmp_path / "output",
        checkpoint_downloader=lambda _uri: checkpoint,
        policy_loader=lambda _spec, _checkpoint: object(),
    )

    assert result["canary"]["status"] != "passed"
    assert result["policy_wam_policy_round_trip_passed"] is False
    assert "policy_wam_policy_round_trip_incomplete" in result["canary"]["blockers"]
