from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.slow

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.action_normalization import (  # noqa: E402
    ActionValidationConfig,
    build_action_normalization_manifest,
    compute_normalization_stats,
    normalize_actions,
    validate_action_stream,
    validate_chunk_alignment,
)
from blueprint_pipeline.sc3_eval_protocol import build_sc3_eval_protocol_artifact  # noqa: E402


def _valid_episode(steps: int = 5) -> list[list[float]]:
    return [[0.01 * i, 0.0, 0.005, 0.0, 0.02, 0.0, 0.5] for i in range(steps)]


def test_validate_action_stream_accepts_valid_7d_stream() -> None:
    result = validate_action_stream(_valid_episode(), config=ActionValidationConfig())
    assert result.valid
    assert result.reasons == []


def test_validate_action_stream_rejects_wrong_dimensionality() -> None:
    result = validate_action_stream([[0.0, 0.1, 0.2]], config=ActionValidationConfig())
    assert not result.valid
    assert any(reason.startswith("action_dim_mismatch") for reason in result.reasons)


def test_validate_action_stream_rejects_out_of_bounds_translation() -> None:
    bad = _valid_episode()
    bad[2][0] = 5.0  # 5 m delta in one step is physically implausible
    result = validate_action_stream(bad, config=ActionValidationConfig())
    assert not result.valid
    assert any("translation_delta_out_of_bounds" in reason for reason in result.reasons)


def test_validate_action_stream_rejects_non_finite_and_missing() -> None:
    nan_result = validate_action_stream(
        [[float("nan"), 0, 0, 0, 0, 0, 0]], config=ActionValidationConfig()
    )
    assert not nan_result.valid
    empty_result = validate_action_stream([], config=ActionValidationConfig())
    assert not empty_result.valid
    assert "action_stream_missing_or_non_numeric" in empty_result.reasons


def test_chunk_alignment_flags_misaligned_chunks() -> None:
    config = ActionValidationConfig(chunk_alignment_tolerance_sec=0.05)
    aligned = validate_chunk_alignment(
        chunk_start_times_sec=[0.0, 1.6],
        frame_times_sec=[0.0, 0.8, 1.6, 2.4],
        config=config,
    )
    assert aligned.valid
    misaligned = validate_chunk_alignment(
        chunk_start_times_sec=[0.4],
        frame_times_sec=[0.0, 0.8],
        config=config,
    )
    assert not misaligned.valid


def test_stats_are_per_dimension_and_never_fabricated() -> None:
    stats = compute_normalization_stats({"ep1": _valid_episode(10)})
    assert stats is not None
    assert len(stats["per_dimension"]) == 7
    assert stats["per_dimension"][6]["mean"] == 0.5  # constant gripper column

    assert compute_normalization_stats({}) is None
    assert compute_normalization_stats({"bad": [[1, 2]]}) is None


def test_normalize_actions_returns_copy_and_zero_centers() -> None:
    episode = _valid_episode(10)
    stats = compute_normalization_stats({"ep1": episode})
    assert stats is not None
    normalized = normalize_actions(episode, stats=stats)
    assert normalized is not episode
    column_mean = sum(row[0] for row in normalized) / len(normalized)
    assert abs(column_mean) < 1e-6
    # Raw stream untouched.
    assert episode[0][6] == 0.5


def test_manifest_validates_and_persists_stats(tmp_path: Path) -> None:
    manifest = build_action_normalization_manifest(
        output_dir=tmp_path,
        episodes={
            "good": {"actions": _valid_episode()},
            "bad_dim": {"actions": [[1.0, 2.0]]},
        },
        action_space={"dim": 7},
    )
    assert manifest["status"] == "validated"
    assert manifest["accepted_episode_count"] == 1
    assert manifest["rejected_episode_count"] == 1
    assert manifest["episode_results"]["bad_dim"]["valid"] is False
    stats = json.loads(Path(manifest["action_norm_stats_path"]).read_text(encoding="utf-8"))
    assert len(stats["per_dimension"]) == 7
    assert manifest["raw_actions_untouched"] is True


def test_manifest_blocks_when_no_valid_episode(tmp_path: Path) -> None:
    manifest = build_action_normalization_manifest(
        output_dir=tmp_path,
        episodes={"bad": {"actions": [[1.0]]}},
        action_space={"dim": 7},
    )
    assert manifest["status"] == "blocked"
    assert manifest["action_norm_stats_path"] is None
    assert "no_valid_action_episodes" in manifest["blockers"]


def _sc3_kwargs() -> dict:
    return {
        "generated_at": "now",
        "job_request": {},
        "policy_package_manifest": {"selected_modalities": ["policy_api_endpoint"]},
        "policy_execution_manifest": {},
        "robot_pov_observation_manifest": {"observation_count": 1},
    }


def test_sc3_action_chunks_blocked_without_normalization_manifest() -> None:
    artifact = build_sc3_eval_protocol_artifact(**_sc3_kwargs())
    requirement = artifact["data_requirements"]["action_chunks"]
    assert requirement["status"] == "blocked"
    assert "action_normalization_manifest_missing" in requirement["blockers"]


def test_sc3_action_chunks_reviewable_with_validated_normalization(tmp_path: Path) -> None:
    norm_manifest = build_action_normalization_manifest(
        output_dir=tmp_path,
        episodes={"good": {"actions": _valid_episode()}},
        action_space={"dim": 7},
    )
    artifact = build_sc3_eval_protocol_artifact(
        **_sc3_kwargs(), action_normalization_manifest=norm_manifest
    )
    requirement = artifact["data_requirements"]["action_chunks"]
    assert requirement["status"] == "reviewable"
    assert requirement["action_normalization_status"] == "validated"
    assert requirement["blockers"] == []
