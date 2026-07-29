from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_pipeline.policy_ranking_powered_droid_analysis import (
    CONDITIONS,
    SEEDS,
    _clustered_window_bootstrap,
    _correlation,
    _load_matrix,
    _load_protocol_thresholds,
    _request_id,
    _scene_distance,
    _session_bootstrap,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "docs/experiments/policy_ranking_roboarena_powered_droid_confirmation_20260729"
    / "protocol_v1.json"
)


def _row() -> dict:
    controls = {
        condition: {"action_sha256": canonical_sha256([[float(index)] for index in range(16)])}
        for condition in CONDITIONS
    }
    return {
        "session_id_internal_only": "session-1",
        "window_index": 0,
        "initial_observation_sha256": "a" * 64,
        "controls": controls,
    }


def test_request_id_matches_runtime_material_contract() -> None:
    row = _row()
    packet_sha = "b" * 64
    revision = "c" * 40
    result = _request_id(packet_sha, revision, row, "recorded", 1)
    assert result == canonical_sha256(
        {
            "packet_sha256": packet_sha,
            "session_id": "session-1",
            "window_index": 0,
            "condition": "recorded",
            "seed": 1,
            "initial_observation_sha256": "a" * 64,
            "action_sha256": row["controls"]["recorded"]["action_sha256"],
            "checkpoint_revision": revision,
        }
    )


def test_load_matrix_binds_every_response_and_video_digest(tmp_path: Path) -> None:
    packet = {"manifest_sha256": "b" * 64, "rows": [_row()]}
    revision = "c" * 40
    response_dir = tmp_path / "responses"
    video_dir = tmp_path / "videos"
    response_dir.mkdir()
    video_dir.mkdir()
    for condition in CONDITIONS:
        for seed in SEEDS:
            request_id = _request_id(
                packet["manifest_sha256"], revision, packet["rows"][0], condition, seed
            )
            video = video_dir / f"{request_id}.mp4"
            video.write_bytes(f"{condition}:{seed}".encode())
            response = {
                "session_id": "session-1",
                "window_index": 0,
                "condition": condition,
                "seed": seed,
                "request_id": request_id,
                "accepted_first_valid": True,
                "response": {"output_sha256": file_sha256(video)},
            }
            (response_dir / f"{request_id}.json").write_text(json.dumps(response), encoding="utf-8")

    videos, records = _load_matrix(packet=packet, output_dir=tmp_path, checkpoint_revision=revision)

    assert len(videos) == len(CONDITIONS) * len(SEEDS)
    assert len(records) == len(videos)


def test_scene_distance_is_normalized_and_temporally_aggregated() -> None:
    black = np.zeros((2, 3, 3), dtype=np.uint8)
    white = np.full((2, 3, 3), 255, dtype=np.uint8)
    gray = np.full((2, 3, 3), 128, dtype=np.uint8)
    assert _scene_distance([black, black], [white, white]) == 1.0
    assert np.isclose(_scene_distance([black, black], [black, gray]), 128.0 / 510.0)


def test_correlation_fails_closed_for_constant_or_short_series() -> None:
    assert _correlation(np.ones(16), np.arange(16)) is None
    assert _correlation(np.arange(3), np.arange(3)) is None
    assert np.isclose(_correlation(np.arange(16), np.arange(16)), 1.0)


def test_bootstraps_use_sessions_as_independent_units() -> None:
    windows = [
        {"session_id": f"session-{session}", "passed": True}
        for session in range(17)
        for _window in range(3)
    ]
    validity = _clustered_window_bootstrap(windows, replicates=200)
    assert validity["independent_session_count"] == 17
    assert validity["window_count"] == 51
    assert validity["estimate"] == validity["lower95"] == validity["upper95"] == 1.0

    sessions = [{"session_id": f"session-{session}", "reliable": True} for session in range(17)]
    reliability = _session_bootstrap(sessions, replicates=200)
    assert reliability["independent_session_count"] == 17
    assert reliability["estimate"] == reliability["lower95"] == 1.0


def test_protocol_thresholds_are_explicitly_loaded_from_frozen_file() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rollout, session, causal = _load_protocol_thresholds(protocol)
    assert rollout.static_motion_max == 0.05
    assert session.minimum_eligible_timing_windows == 3
    assert causal["minimum_original_action_motion_correlation"] == 0.10
