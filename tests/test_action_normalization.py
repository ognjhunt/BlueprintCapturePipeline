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
    build_action_normalization_from_trace,
    build_action_normalization_manifest,
    compute_normalization_stats,
    normalize_actions,
    validate_action_stream,
    validate_chunk_alignment,
)
from blueprint_pipeline.sc3_eval_protocol import build_sc3_eval_protocol_artifact  # noqa: E402


def _valid_episode(steps: int = 5) -> list[list[float]]:
    return [
        [
            0.01 * i,
            -0.005 * i,
            0.002 * i,
            0.01 * i,
            -0.008 * i,
            0.006 * i,
            -0.5 + 0.1 * i,
        ]
        for i in range(steps)
    ]


def _episode_payload(actions: list[list[float]] | None = None) -> dict:
    rows = actions or _valid_episode()
    timestamps = [index / 10.0 for index in range(len(rows))]
    return {
        "actions": rows,
        "chunk_start_times_sec": timestamps,
        "frame_times_sec": timestamps,
        "control_rate_hz": 10.0,
    }


def _action_space() -> dict:
    return {
        "dim": 7,
        "representation": "7d_delta_end_effector_pose",
        "order": [
            "delta_x_m",
            "delta_y_m",
            "delta_z_m",
            "delta_roll_rad",
            "delta_pitch_rad",
            "delta_yaw_rad",
            "gripper_normalized",
        ],
        "units": ["m", "m", "m", "rad", "rad", "rad", "normalized"],
    }


def _provenance() -> dict:
    return {
        "source_trace_path": "/tmp/policy_execution_trace.json",
        "source_trace_sha256": "a" * 64,
        "trace_schema_version": "robot_policy_execution_trace.v1",
        "consumed_by": "unit_test_sc3_evaluator",
    }


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
    assert stats["per_dimension"][6]["std"] > 0.0

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
    assert episode[0][6] == -0.5


def test_manifest_fails_closed_when_any_episode_is_rejected(tmp_path: Path) -> None:
    manifest = build_action_normalization_manifest(
        output_dir=tmp_path,
        episodes={
            "good": _episode_payload(),
            "bad_dim": _episode_payload([[1.0, 2.0]]),
        },
        action_space=_action_space(),
        corpus_provenance=_provenance(),
    )
    assert manifest["status"] == "blocked"
    assert manifest["accepted_episode_count"] == 1
    assert manifest["rejected_episode_count"] == 1
    assert manifest["episode_results"]["bad_dim"]["valid"] is False
    assert "one_or_more_action_episodes_rejected" in manifest["blockers"]
    assert manifest["action_norm_stats_path"] is None


def test_manifest_validates_exact_timed_trace_and_persists_hashed_outputs(
    tmp_path: Path,
) -> None:
    manifest = build_action_normalization_manifest(
        output_dir=tmp_path,
        episodes={"good": _episode_payload()},
        action_space=_action_space(),
        corpus_provenance=_provenance(),
    )
    assert manifest["status"] == "validated"
    stats = json.loads(Path(manifest["action_norm_stats_path"]).read_text(encoding="utf-8"))
    assert len(stats["per_dimension"]) == 7
    assert len(manifest["action_norm_stats_sha256"]) == 64
    assert len(manifest["normalized_action_corpus_sha256"]) == 64
    assert manifest["exact_consumed_trace_bound"] is True
    assert manifest["all_dimensions_nonzero_variance"] is True
    assert manifest["raw_actions_untouched"] is True


def test_manifest_blocks_when_no_valid_episode(tmp_path: Path) -> None:
    manifest = build_action_normalization_manifest(
        output_dir=tmp_path,
        episodes={"bad": _episode_payload([[1.0]])},
        action_space=_action_space(),
        corpus_provenance=_provenance(),
    )
    assert manifest["status"] == "blocked"
    assert manifest["action_norm_stats_path"] is None
    assert "no_valid_action_episodes" in manifest["blockers"]


def test_manifest_blocks_missing_timing_and_zero_variance(tmp_path: Path) -> None:
    missing_time = build_action_normalization_manifest(
        output_dir=tmp_path / "missing-time",
        episodes={"bad": {"actions": _valid_episode()}},
        action_space=_action_space(),
        corpus_provenance=_provenance(),
    )
    assert missing_time["status"] == "blocked"
    assert "chunk_timestamps_missing" in missing_time["episode_results"]["bad"]["reasons"]
    constant = [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5] for _ in range(5)]
    zero_variance = build_action_normalization_manifest(
        output_dir=tmp_path / "zero-variance",
        episodes={"bad": _episode_payload(constant)},
        action_space=_action_space(),
        corpus_provenance=_provenance(),
    )
    assert zero_variance["status"] == "blocked"
    assert any(
        blocker.startswith("normalization_zero_or_invalid_variance")
        for blocker in zero_variance["blockers"]
    )


def test_trace_builder_uses_only_explicit_vectors_and_exact_trace_hash(tmp_path: Path) -> None:
    actions = _valid_episode()
    timestamps = [index / 10.0 for index in range(len(actions))]
    trace = {
        "schema_version": "robot_policy_execution_trace.v1",
        "attempts": [
            {
                "attempt_id": "attempt-1",
                "actions": [
                    {
                        "delta_end_effector_pose_7d": vector,
                        "timestamp_sec": timestamp,
                    }
                    for vector, timestamp in zip(actions, timestamps)
                ],
                "frame_times_sec": timestamps,
                "control_rate_hz": 10.0,
            }
        ],
    }
    trace_path = tmp_path / "policy_execution_trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    manifest = build_action_normalization_from_trace(
        output_dir=tmp_path,
        trace=trace,
        source_trace_path=trace_path,
        consumed_by="unit_test_sc3_evaluator",
        action_space=_action_space(),
    )
    assert manifest["status"] == "validated"
    assert manifest["corpus_provenance"]["source_trace_file_present"] is True
    assert len(manifest["source_trace_sha256"]) == 64

    trace["attempts"][0]["actions"] = [
        {"action_type": "stop", "timestamp_sec": timestamp}
        for timestamp in timestamps
    ]
    blocked = build_action_normalization_from_trace(
        output_dir=tmp_path / "blocked",
        trace=trace,
        source_trace_path=tmp_path / "missing-trace.json",
        consumed_by="unit_test_sc3_evaluator",
        action_space=_action_space(),
    )
    assert blocked["status"] == "blocked"
    assert "no_valid_action_episodes" in blocked["blockers"]


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
        episodes={"good": _episode_payload()},
        action_space=_action_space(),
        corpus_provenance=_provenance(),
    )
    artifact = build_sc3_eval_protocol_artifact(
        **_sc3_kwargs(), action_normalization_manifest=norm_manifest
    )
    requirement = artifact["data_requirements"]["action_chunks"]
    assert requirement["status"] == "reviewable"
    assert requirement["action_normalization_status"] == "validated"
    assert requirement["blockers"] == []
