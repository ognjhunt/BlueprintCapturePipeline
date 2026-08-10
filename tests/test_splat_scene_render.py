"""Hermetic fail-closed tests for blueprint_pipeline.splat_scene_render.

The happy path requires node + Spark + ffmpeg and is exercised by the end-to-end
integration run; these tests pin the fail-closed contract (no fabricated passes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.splat_scene_render import (
    RENDERED_BY,
    _decimate_to_standard_ply,
    _derive_bounds_camera_specs,
    _derive_multi_bounds_camera_specs,
    _encode_mp4,
    _normalize_camera_specs,
    render_splat_scene,
)


def test_bounds_camera_specs_are_task_neutral_deterministic_and_receipted() -> None:
    specs, plan, errors = _derive_bounds_camera_specs(
        ((-0.2, -0.3, 0.8), (0.2, 0.3, 1.1)),
        margin=3.0,
        n_azimuths=4,
        elevations_deg=(15.0, 50.0),
        vfov_deg=44.0,
        width=1600,
        height=1200,
        camera_id_prefix="destination_candidate",
    )

    assert errors == []
    assert plan == {
        "planner": "scene_placement.perception_views.view_ring_for_bounds",
        "focus_bounds": {"min": [-0.2, -0.3, 0.8], "max": [0.2, 0.3, 1.1]},
        "margin": 3.0,
        "n_azimuths": 4,
        "elevations_deg": [15.0, 50.0],
        "vfov_deg": 44.0,
        "width": 1600,
        "height": 1200,
        "camera_id_prefix": "destination_candidate",
        "camera_count": 8,
        "claim_boundary": "reconnaissance_camera_plan_not_source_observation_recovery",
    }
    assert [row["id"] for row in specs or []] == [
        "destination_candidate_e00_a00",
        "destination_candidate_e00_a01",
        "destination_candidate_e00_a02",
        "destination_candidate_e00_a03",
        "destination_candidate_e01_a00",
        "destination_candidate_e01_a01",
        "destination_candidate_e01_a02",
        "destination_candidate_e01_a03",
    ]
    assert all(row["spec"]["fov"] == 44.0 for row in specs or [])


def test_invalid_bounds_camera_specs_fail_closed() -> None:
    specs, plan, errors = _derive_bounds_camera_specs(
        ((0.0, 0.0, 1.0), (0.0, 0.5, 1.2)),
        margin=1.0,
        n_azimuths=1,
        elevations_deg=(),
        vfov_deg=180.0,
        width=0,
        height=960,
        camera_id_prefix="../bad",
    )

    assert specs is None
    assert plan is None
    assert errors == ["bounds_view_ring_parameters_invalid", "focus_bounds_invalid"]


def test_multi_entity_bounds_camera_specs_share_one_stable_render_plan() -> None:
    specs, plan, errors = _derive_multi_bounds_camera_specs(
        [
            {
                "target_id": "movable",
                "semantic_role": "movable_deformable",
                "focus_bounds": [[-0.2, -0.1, 0.8], [0.2, 0.1, 1.0]],
                "n_azimuths": 4,
                "elevations_deg": [20.0, 55.0],
            },
            {
                "target_id": "destination",
                "semantic_role": "destination_receptacle",
                "focus_bounds": [[0.5, -0.3, 0.7], [0.9, 0.3, 1.1]],
                "n_azimuths": 4,
                "elevations_deg": [35.0, 65.0],
                "vfov_deg": 45.0,
            },
        ],
        width=1024,
        height=768,
        max_total_cameras=16,
    )

    assert errors == []
    assert len(specs or []) == 16
    assert plan is not None
    assert plan["target_count"] == 2
    assert plan["camera_count"] == 16
    assert plan["max_total_cameras"] == 16
    assert [row["target_id"] for row in plan["targets"]] == ["movable", "destination"]
    assert len({row["id"] for row in specs or []}) == 16


def test_multi_entity_bounds_camera_specs_reject_duplicate_targets() -> None:
    specs, plan, errors = _derive_multi_bounds_camera_specs(
        [
            {"target_id": "same", "focus_bounds": [[0, 0, 0], [1, 1, 1]]},
            {"target_id": "same", "focus_bounds": [[2, 2, 2], [3, 3, 3]]},
        ],
        width=1024,
        height=768,
    )

    assert specs is None
    assert plan is None
    assert errors == ["focus_bounds_request_target_id_invalid:1"]


def test_multi_entity_bounds_camera_specs_fail_before_over_budget_render() -> None:
    specs, plan, errors = _derive_multi_bounds_camera_specs(
        [
            {"target_id": "one", "focus_bounds": [[0, 0, 0], [1, 1, 1]]},
            {"target_id": "two", "focus_bounds": [[2, 2, 2], [3, 3, 3]]},
        ],
        width=1024,
        height=768,
        max_total_cameras=15,
    )

    assert specs is None
    assert plan is None
    assert errors == ["focus_bounds_camera_budget_exceeded"]


def test_valid_standard_ply_needs_no_decoder_when_decimation_disabled(
    tmp_path: Path,
) -> None:
    count = 4
    source = write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=np.zeros((count, 3), dtype=np.float32),
            opacity=np.zeros(count, dtype=np.float32),
            f_dc=np.zeros((count, 3), dtype=np.float32),
            scales=np.zeros((count, 3), dtype=np.float32),
            quats=np.zeros((count, 4), dtype=np.float32),
            properties=(),
        ),
        tmp_path / "source.ply",
    )
    result = _decimate_to_standard_ply(
        source,
        tmp_path / "out" / "scene_standard.ply",
        0,
        repo_root=tmp_path,
        node="node",
        timeout=10,
    )
    assert result["status"] == "completed"
    assert result["decoder"] == "validated_standard_3dgs_copy"
    assert result["vertex_count"] == count


def test_blocked_when_source_missing(tmp_path: Path) -> None:
    m = render_splat_scene(tmp_path / "nope.ply", tmp_path / "out")
    assert m["status"] == "blocked"
    assert "splat_source_missing_or_unsupported" in m["blockers"]
    assert m["rendered_by"] == RENDERED_BY
    assert m["proof_boundary"]["rendered_by_isaac_rtx"] is False


def test_blocked_when_unsupported_suffix(tmp_path: Path) -> None:
    src = tmp_path / "scene.txt"
    src.write_text("not a splat")
    m = render_splat_scene(src, tmp_path / "out")
    assert m["status"] == "blocked"
    assert "splat_source_missing_or_unsupported" in m["blockers"]


def test_blocked_when_graphics_backend_is_unknown(tmp_path: Path) -> None:
    src = tmp_path / "scene.ply"
    src.write_bytes(b"placeholder")
    m = render_splat_scene(src, tmp_path / "out", graphics_backend="cloud_magic")
    assert m["status"] == "blocked"
    assert m["blockers"] == ["unsupported_graphics_backend"]


def test_retained_attempt_output_can_be_guarded_before_decode(
    tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "scene.ply"
    src.write_bytes(b"placeholder")
    out = tmp_path / "retained_attempt"
    out.mkdir()
    (out / "partial.png").write_bytes(b"retained")
    decode_called = False

    def unexpected_decode(*args, **kwargs):
        nonlocal decode_called
        decode_called = True
        raise AssertionError("decode must not run")

    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render._decimate_to_standard_ply",
        unexpected_decode,
    )
    result = render_splat_scene(src, out, require_empty_output=True)

    assert result["status"] == "blocked"
    assert result["blockers"] == ["render_output_directory_not_empty"]
    assert decode_called is False


def test_blocked_when_cli_missing(tmp_path: Path) -> None:
    # valid-looking .ply source but repo_root has no splat-transform CLI installed
    src = tmp_path / "scene.ply"
    src.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    m = render_splat_scene(src, tmp_path / "out", repo_root=tmp_path)
    assert m["status"] == "blocked"
    assert "splat_transform_cli_unavailable" in m["blockers"]
    assert m["proof_boundary"]["captured_scene_displayed"] is False


def test_invalid_focus_point_fails_closed_after_decode(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "scene.ply"
    src.write_bytes(b"placeholder")

    def completed_decode(_source, destination, *args, **kwargs):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"standard")
        return {"status": "completed"}

    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render._decimate_to_standard_ply",
        completed_decode,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render.read_standard_3dgs_ply",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render.analyze_scene",
        lambda *args, **kwargs: type("Geometry", (), {"to_dict": lambda self: {}})(),
    )
    result = render_splat_scene(src, tmp_path / "out", focus_point=[1.0, float("nan"), 2.0])
    assert result["status"] == "blocked"
    assert result["blockers"] == ["invalid_focus_point"]


def test_decoder_success_without_output_fails_closed(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "scene.ply"
    src.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render._decimate_to_standard_ply",
        lambda *args, **kwargs: {"status": "completed"},
    )

    result = render_splat_scene(src, tmp_path / "out")

    assert result["status"] == "blocked"
    assert result["blockers"] == ["standard_ply_missing_after_decode"]


def test_caller_camera_specs_are_normalized_and_digestable() -> None:
    observed, errors = _normalize_camera_specs(
        [
            {
                "id": "room_00_yaw_000",
                "spec": {
                    "pos": [0, 0, 1.35],
                    "target": [2, 0, 1.05],
                    "fov": 70,
                    "up": [0, 0, 1],
                },
            }
        ]
    )

    assert errors == []
    assert observed == [
        {
            "id": "room_00_yaw_000",
            "spec": {
                "pos": [0.0, 0.0, 1.35],
                "target": [2.0, 0.0, 1.05],
                "fov": 70.0,
                "up": [0.0, 0.0, 1.0],
            },
        }
    ]


def test_invalid_caller_camera_specs_fail_before_decode(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "scene.ply"
    src.write_bytes(b"placeholder")
    decode_called = False

    def unexpected_decode(*args, **kwargs):
        nonlocal decode_called
        decode_called = True
        raise AssertionError("decode must not run")

    monkeypatch.setattr(
        "blueprint_pipeline.splat_scene_render._decimate_to_standard_ply",
        unexpected_decode,
    )
    result = render_splat_scene(
        src,
        tmp_path / "out",
        camera_specs=[
            {
                "id": "duplicate",
                "spec": {"pos": [0, 0, 0], "target": [1, 0, 0], "fov": 70},
            },
            {
                "id": "duplicate",
                "spec": {"pos": [0, 0, 0], "target": [1, 0, 0], "fov": 70},
            },
        ],
    )

    assert result["status"] == "blocked"
    assert "camera_spec_id_or_payload_invalid" in result["blockers"]
    assert decode_called is False


def test_encode_mp4_no_frames(tmp_path: Path) -> None:
    result = _encode_mp4([], tmp_path / "out.mp4")
    assert result["status"] == "blocked"
    assert result["blockers"][0] in {"no_frames_to_encode", "ffmpeg_unavailable"}
