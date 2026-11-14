from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.droid_oscar_closed_loop_adapter import EXTERIOR_VIEW, WRIST_VIEW
from blueprint_pipeline.new_site_diagnostic_canary_gpu import (
    MultiViewCanaryReliabilityGate,
    materialize_canary_background,
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

    first_receipt = materialize_canary_background(
        source_path=source, output_path=first
    )
    second_receipt = materialize_canary_background(
        source_path=source, output_path=second
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_receipt["output_sha256"] == second_receipt["output_sha256"]
    assert first_receipt["source_size_px"] == [1920, 1440]
    assert first_receipt["crop_applied"] is False
    assert first_receipt["camera_selection_changed"] is False


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
    result = MultiViewCanaryReliabilityGate(
        ReliabilityThresholds(), assessor=assessor
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

    wrist = result["reports_by_view"][WRIST_VIEW]
    assert result["status"] == "passed"
    assert wrist["raw_pixel_flags"] == [FLAG_STATIC_UNDER_COMMAND]
    assert wrist["flags"] == []
    assert wrist["motion_basis"] == "hash_verified_reference_frame_fk_trace"
    assert wrist["kinematic_motion"]["maximum_reference_displacement_m"] == pytest.approx(
        0.02
    )


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
    result = MultiViewCanaryReliabilityGate(
        ReliabilityThresholds(), assessor=assessor
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
