from __future__ import annotations

import hashlib
import json

from PIL import Image, ImageDraw

from blueprint_pipeline.isaac_review_media import (
    admit_collected_scenario_episode,
    admit_full_ordered_episode,
)


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


def _step_bindings(count: int) -> list[dict]:
    return [
        {
            "step_index": index,
            "source_action_sha256": hashlib.sha256(f"action-{index}".encode()).hexdigest(),
            "stage_id": "c" * 64,
            "simulator_session_id": "session-1",
            "before_timestamp": str(1000 + index * 10),
            "after_timestamp": str(1005 + index * 10),
        }
        for index in range(count)
    ]


def _frame_step_bindings(overview, pov) -> dict[str, dict]:
    bindings: dict[str, dict] = {}
    steps = _step_bindings(max(len(overview), len(pov)))
    for role, paths in (("overview", overview), ("robot_pov", pov)):
        for index, path in enumerate(paths):
            bindings[path.name] = {
                "camera_role": role,
                "step_index": index,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                **{key: value for key, value in steps[index].items() if key != "step_index"},
            }
    return bindings


def _review(count: int) -> dict:
    return {
        "status": "passed",
        "full_ordered_episode_reviewed": True,
        "abstained": False,
        "review_runtime_id": "review-1",
        "review_source": "external_semantic_review_api",
        "request_sha256": "a" * 64,
        "response_sha256": "b" * 64,
        "frame_review_count": 2 * count,
    }


def _episode(tmp_path, count: int, *, pov_count: int | None = None):
    overview = [tmp_path / f"overview_{index:04d}.png" for index in range(count)]
    pov = [
        tmp_path / f"robot_pov_{index:04d}.png"
        for index in range(count if pov_count is None else pov_count)
    ]
    for index, path in enumerate(overview):
        _frame(path, offset=index)
    for index, path in enumerate(pov):
        _frame(path, offset=index + 4)
    return overview, pov


def _admit(overview, pov, *, expected: int, review=None, bindings=..., frame_bindings=...):
    semantics = {**_semantics(overview, "overview"), **_semantics(pov, "robot_pov")}
    return admit_full_ordered_episode(
        camera_frames={"overview": overview, "robot_pov": pov},
        frame_semantics=semantics,
        semantic_review=_review(expected) if review is None else review,
        expected_frame_count=expected,
        step_bindings=_step_bindings(expected) if bindings is ... else bindings,
        frame_step_bindings=_frame_step_bindings(overview, pov)
        if frame_bindings is ...
        else frame_bindings,
    )


def test_full_ordered_episode_admits_every_frame_and_both_camera_roles(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 3)
    result = _admit(overview, pov, expected=3)
    assert result["status"] == "passed"
    assert len(result["frame_rows"]) == 6
    assert result["full_ordered_episode_admitted"] is True


def test_frame_sidecar_from_another_action_or_session_blocks(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 2)
    frame_bindings = _frame_step_bindings(overview, pov)
    frame_bindings[overview[1].name]["source_action_sha256"] = "f" * 64
    frame_bindings[pov[0].name]["simulator_session_id"] = "foreign-session"
    result = _admit(
        overview,
        pov,
        expected=2,
        frame_bindings=frame_bindings,
    )
    assert result["status"] == "blocked"
    assert any("source_action_sha256_mismatch" in item for item in result["blockers"])
    assert any("simulator_session_id_mismatch" in item for item in result["blockers"])


def test_full_ordered_episode_rejects_stale_frame_and_missing_semantic_api(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 2)
    overview[1].write_bytes(overview[0].read_bytes())
    result = admit_full_ordered_episode(
        camera_frames={"overview": overview, "robot_pov": pov},
        frame_semantics={
            **_semantics(overview, "overview"),
            **_semantics(pov, "robot_pov"),
        },
        semantic_review=None,
        expected_frame_count=2,
        step_bindings=_step_bindings(2),
        frame_step_bindings=_frame_step_bindings(overview, pov),
    )
    assert result["status"] == "blocked"
    assert "overview:1:frame_stale_checksum_or_pixels" in result["blockers"]
    assert "full_ordered_episode_semantic_review_missing_or_blocked" in result["blockers"]


def test_one_of_n_frames_per_camera_blocks(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 1)
    result = _admit(overview, pov, expected=20, bindings=_step_bindings(20))
    assert result["status"] == "blocked"
    assert any("ordered_frame_count_mismatch" in item for item in result["blockers"])


def test_equal_truncation_of_both_streams_blocks(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 1)
    result = _admit(overview, pov, expected=20, bindings=_step_bindings(20))
    assert result["status"] == "blocked"


def test_n_overview_plus_n_minus_one_pov_blocks(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 3, pov_count=2)
    result = _admit(overview, pov, expected=3)
    assert result["status"] == "blocked"
    assert any(
        item.startswith("robot_pov:") and "ordered_frame_count_mismatch" in item
        for item in result["blockers"]
    )


def test_gap_in_frame_indices_blocks_even_when_count_matches(tmp_path) -> None:
    overview = [tmp_path / "overview_0000.png", tmp_path / "overview_0002.png"]
    pov = [tmp_path / "robot_pov_0000.png", tmp_path / "robot_pov_0001.png"]
    for index, path in enumerate(overview):
        _frame(path, offset=index)
    for index, path in enumerate(pov):
        _frame(path, offset=index + 4)
    result = _admit(overview, pov, expected=2)
    assert result["status"] == "blocked"
    assert any("frame_indices_not_contiguous" in item for item in result["blockers"])


def test_duplicate_hash_at_non_adjacent_steps_blocks(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 3)
    overview[2].write_bytes(overview[0].read_bytes())
    result = _admit(overview, pov, expected=3)
    assert result["status"] == "blocked"
    assert any(
        "frame_duplicate_sha256_across_steps" in item for item in result["blockers"]
    )


def test_out_of_order_step_timestamps_block(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 2)
    bindings = _step_bindings(2)
    bindings[1]["after_timestamp"] = "900"
    bindings[1]["before_timestamp"] = "890"
    result = _admit(overview, pov, expected=2, bindings=bindings)
    assert result["status"] == "blocked"
    assert any(
        "episode_step_bindings_timestamps_not_ordered" in item
        for item in result["blockers"]
    )


def test_missing_step_bindings_block(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 2)
    result = _admit(overview, pov, expected=2, bindings=None)
    assert result["status"] == "blocked"
    assert "episode_step_bindings_missing" in result["blockers"]


def test_frame_from_another_attempt_blocks(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 2)
    frame_bindings = _frame_step_bindings(overview, pov)
    frame_bindings[overview[1].name]["sha256"] = "f" * 64
    result = _admit(overview, pov, expected=2, frame_bindings=frame_bindings)
    assert result["status"] == "blocked"
    assert any(
        "frame_step_binding_sha256_mismatch" in item for item in result["blockers"]
    )


def test_missing_renderer_frame_bindings_block(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 2)
    result = _admit(overview, pov, expected=2, frame_bindings=None)
    assert result["status"] == "blocked"
    assert "episode_frame_step_bindings_missing" in result["blockers"]


def test_semantic_review_coverage_count_must_match_horizon(tmp_path) -> None:
    overview, pov = _episode(tmp_path, 2)
    review = _review(2)
    review["frame_review_count"] = 3
    result = _admit(overview, pov, expected=2, review=review)
    assert result["status"] == "blocked"
    assert any(
        "semantic_review_coverage_count_mismatch" in item for item in result["blockers"]
    )


def test_collected_scenario_episode_requires_renderer_sidecar(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    overview = [frames / f"overview_{index:04d}.png" for index in range(2)]
    pov = [frames / f"robot_pov_{index:04d}.png" for index in range(2)]
    for index, path in enumerate(overview):
        _frame(path, offset=index)
    for index, path in enumerate(pov):
        _frame(path, offset=index + 4)
    semantics = {
        str(path): value
        for path, value in {
            **_semantics(overview, "overview"),
            **_semantics(pov, "robot_pov"),
        }.items()
    }
    (tmp_path / "full_episode_frame_semantics.json").write_text(
        json.dumps({"frames": semantics})
    )
    (tmp_path / "full_episode_semantic_review.json").write_text(json.dumps(_review(2)))
    result = admit_collected_scenario_episode(
        scenario_dir=tmp_path,
        expected_frame_count=2,
        step_bindings=_step_bindings(2),
    )
    assert result["status"] == "blocked"
    assert "episode_frame_step_bindings_missing" in result["blockers"]

    (frames / "frame_step_bindings.json").write_text(
        json.dumps(
            {
                "schema_version": "isaac_review_frame_step_bindings.v1",
                "frames": _frame_step_bindings(overview, pov),
            }
        )
    )
    admitted = admit_collected_scenario_episode(
        scenario_dir=tmp_path,
        expected_frame_count=2,
        step_bindings=_step_bindings(2),
    )
    assert admitted["status"] == "passed"
