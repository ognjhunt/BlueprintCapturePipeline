from __future__ import annotations

import hashlib

from PIL import Image, ImageDraw

from blueprint_pipeline.isaac_review_media import admit_full_ordered_episode


def _frame(path, *, offset: int) -> None:
    image = Image.new("RGB", (96, 64), (180, 180, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10 + offset, 10, 35 + offset, 55), fill=(20, 50, 100))
    image.save(path)


def _semantics(paths, role):
    result = {}
    for path in paths:
        if role == "overview":
            result[str(path)] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "g1_visible": True,
                "target_visible": True,
                "floor_support_visible": True,
                "orientation_visible": True,
                "clearance_visible": True,
                "robot_pixel_occupancy": 0.2,
                "target_pixel_occupancy": 0.1,
            }
        else:
            result[str(path)] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "target_visible": True,
                "active_hand_wrist_chain_visible": True,
            }
    return result


def test_full_ordered_episode_admits_every_frame_and_both_camera_roles(tmp_path) -> None:
    overview = [tmp_path / f"overview_{index:04d}.png" for index in range(3)]
    pov = [tmp_path / f"robot_pov_{index:04d}.png" for index in range(3)]
    for index, path in enumerate(overview):
        _frame(path, offset=index)
    for index, path in enumerate(pov):
        _frame(path, offset=index + 4)
    semantics = {**_semantics(overview, "overview"), **_semantics(pov, "robot_pov")}
    result = admit_full_ordered_episode(
        camera_frames={"overview": overview, "robot_pov": pov},
        frame_semantics=semantics,
        semantic_review={
            "status": "passed",
            "full_ordered_episode_reviewed": True,
            "abstained": False,
            "review_runtime_id": "review-1",
            "review_source": "external_semantic_review_api",
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        },
        expected_frame_count=3,
    )
    assert result["status"] == "passed"
    assert len(result["frame_rows"]) == 6
    assert result["full_ordered_episode_admitted"] is True


def test_full_ordered_episode_rejects_stale_frame_and_missing_semantic_api(tmp_path) -> None:
    overview = [tmp_path / f"overview_{index:04d}.png" for index in range(2)]
    pov = [tmp_path / f"robot_pov_{index:04d}.png" for index in range(2)]
    _frame(overview[0], offset=0)
    overview[1].write_bytes(overview[0].read_bytes())
    _frame(pov[0], offset=2)
    _frame(pov[1], offset=3)
    semantics = {**_semantics(overview, "overview"), **_semantics(pov, "robot_pov")}
    result = admit_full_ordered_episode(
        camera_frames={"overview": overview, "robot_pov": pov},
        frame_semantics=semantics,
        semantic_review=None,
        expected_frame_count=2,
    )
    assert result["status"] == "blocked"
    assert "overview:1:frame_stale_checksum_or_pixels" in result["blockers"]
    assert "full_ordered_episode_semantic_review_missing_or_blocked" in result["blockers"]
