from __future__ import annotations

from pathlib import Path

from blueprint_pipeline import oscar_wam_command_adapter as adapter
from blueprint_pipeline import oscar_wam_provider_bundle as bundle


def _pure_projected_skeleton_package(trace_path: Path) -> dict[str, object]:
    return {
        "schema_version": "blueprint_oscar_wam_input_package.v1",
        "conditioning_video_review_validation": {"status": "completed", "blockers": []},
        "conditioning_video_visual_smoke": {
            "status": "failed_visual_quality_smoke",
            "blockers": ["generated_rollout_first_frame_not_scene_like"],
        },
        "conditioning_video_decode_valid_for_review": True,
        "conditioning_video_visually_useful_for_model_input": True,
        "first_frame": {"path": "/tmp/source-first-frame.png"},
        "skeleton_video": {
            "path": "/tmp/source-skeleton.mp4",
            "conditioning_mode": "projected_g1_skeleton",
            "projected_g1_skeleton_rendered": True,
            "skeleton_stream_separate_from_rgb": True,
            "skeleton_stream_texture_free": True,
            "skeleton_stream_image_aligned_to_rgb": True,
            "first_rgb_frame_anchors_scene_and_robot_appearance": True,
            "projected_g1_skeleton_landmark_draw_count": 6,
            "visual_signal": {"status": "completed", "blockers": []},
        },
        "projected_skeleton_trace": {
            "path": str(trace_path),
            "used_for_conditioning": True,
            "row_count": 1,
            "projectable_row_count": 1,
        },
        "oscar_dual_stream_input_contract": {
            "first_rgb_frame_path": "/tmp/source-first-frame.png",
            "skeleton_video_path": "/tmp/source-skeleton.mp4",
            "separate_2d_skeleton_stream": True,
            "skeleton_stream_texture_free": True,
            "skeleton_stream_image_aligned_to_rgb": True,
            "first_rgb_frame_anchors_scene_and_robot_appearance": True,
            "full_rgb_video_required_for_oscar_inference": False,
        },
        "claim_boundary": {
            "projected_g1_skeleton_conditioning_used": True,
            "first_rgb_frame_anchors_scene_and_robot_appearance": True,
            "separate_2d_skeleton_stream_aligned_to_rgb": True,
            "skeleton_stream_is_texture_free": True,
        },
    }


def test_projected_g1_defaults_to_texture_free_dual_stream(
    monkeypatch,
) -> None:
    rows = [{"projected_landmark_count": 2}]

    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE", raising=False)
    assert adapter._configured_conditioning_mode(rows) == "projected_g1_skeleton"

    monkeypatch.setenv(
        "BLUEPRINT_OSCAR_WAM_CONDITIONING_MODE",
        "projected_g1_skeleton_rgb_overlay",
    )
    assert adapter._configured_conditioning_mode(rows) == "projected_g1_skeleton_rgb_overlay"


def test_texture_free_projected_skeleton_uses_signal_not_scene_visual_smoke(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    trace.write_text('{"projected_landmark_count": 2}\n', encoding="utf-8")

    pure_package = _pure_projected_skeleton_package(trace)
    assert bundle._conditioning_video_input_blockers(pure_package) == []

    proxy_package = {
        **pure_package,
        "skeleton_video": {
            "conditioning_mode": "oscar_gripper_scenario_proxy",
            "projected_g1_skeleton_rendered": False,
            "visual_signal": {"status": "completed", "blockers": []},
        },
        "projected_skeleton_trace": {},
        "claim_boundary": {},
    }
    assert (
        "oscar_input_skeleton_conditioning_video_visual_smoke_failed"
        in bundle._conditioning_video_input_blockers(proxy_package)
    )


def test_provider_runtime_preserves_first_frame_plus_separate_skeleton_contract(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    trace.write_text('{"projected_landmark_count": 2}\n', encoding="utf-8")
    package = _pure_projected_skeleton_package(trace)

    runtime = bundle._runtime_input_package_manifest(
        package,
        first_frame_runtime_path="provider_runtime/oscar_input/first_frame.png",
        skeleton_runtime_path=(
            "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"
        ),
        projected_skeleton_runtime_path=(
            "provider_runtime/oscar_input/g1_projected_skeleton_trace.jsonl"
        ),
    )

    assert runtime["oscar_dual_stream_input_contract"]["first_rgb_frame_path"] == (
        "provider_runtime/oscar_input/first_frame.png"
    )
    assert runtime["oscar_dual_stream_input_contract"]["skeleton_video_path"] == (
        "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"
    )
    assert runtime["oscar_dual_stream_input_contract"]["separate_2d_skeleton_stream"] is True
    assert runtime["oscar_dual_stream_input_contract"]["skeleton_stream_texture_free"] is True
    assert runtime["oscar_projected_skeleton_runtime_contract"][
        "separate_2d_skeleton_stream"
    ] is True
    assert runtime["oscar_projected_skeleton_runtime_contract"][
        "first_rgb_frame_anchors_scene_and_robot_appearance"
    ] is True
    assert runtime["claim_boundary"]["separate_2d_skeleton_stream_aligned_to_rgb"] is True
    assert runtime["claim_boundary"]["skeleton_stream_is_texture_free"] is True
