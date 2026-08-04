from __future__ import annotations

import json

import numpy as np
import pytest

from blueprint_pipeline.adp_prospective_design import validate_episode_evidence_contract
from blueprint_pipeline.episode_visual_evidence import (
    finalize_visual_evidence,
    persist_observation_frame,
)


def test_media_seal_retains_lossless_inputs_terminal_manifest_and_review_video(
    tmp_path,
) -> None:
    episode_id = "episode-media-1"
    first = persist_observation_frame(
        np.full((32, 64, 3), 17, dtype=np.uint8),
        output_dir=tmp_path,
        episode_id=episode_id,
        frame_index=0,
        kind="policy-input",
    )
    second = persist_observation_frame(
        np.full((32, 64, 3), 29, dtype=np.uint8),
        output_dir=tmp_path,
        episode_id=episode_id,
        frame_index=1,
        kind="policy-input",
    )
    terminal = persist_observation_frame(
        np.full((32, 64, 3), 43, dtype=np.uint8),
        output_dir=tmp_path,
        episode_id=episode_id,
        frame_index=2,
        kind="terminal-observation",
    )
    visual, artifacts = finalize_visual_evidence(
        output_dir=tmp_path,
        episode_id=episode_id,
        identity={"policy_id": "test-policy"},
        policy_input_frames=[first, second],
        terminal_observation=terminal,
    )
    manifest_artifact = next(
        row for row in artifacts if row["role"] == "observation_frame_manifest"
    )
    manifest = json.loads((tmp_path / manifest_artifact["relative_path"]).read_text())
    video = next(row for row in artifacts if row["role"] == "episode_video")

    assert manifest["frame_manifest_digest"] == visual["frame_manifest_digest"]
    assert len(manifest["policy_input_frames"]) == 2
    assert (tmp_path / video["relative_path"]).read_bytes()[4:8] == b"ftyp"
    episode = {
        "episode_id": episode_id,
        "status": "completed",
        "policy_query_count": 2,
        "visual_evidence": visual,
        "artifacts": artifacts,
        "evaluator": {
            "owner": "environment_not_policy",
            "grader_type": "deterministic_simulator_state",
            "success_source": "frozen_object_state_predicates",
            "policy_self_report_used": False,
        },
        "success_evidence": {
            "grader_type": "deterministic_simulator_state",
            "policy_self_report_used": False,
        },
    }
    admission = validate_episode_evidence_contract(episode)
    assert admission["status"] == "admitted"
    assert admission["completed_media_contract"] is True

    with pytest.raises(FileExistsError, match="overwrite_forbidden"):
        persist_observation_frame(
            np.full((32, 64, 3), 17, dtype=np.uint8),
            output_dir=tmp_path,
            episode_id=episode_id,
            frame_index=0,
            kind="policy-input",
        )
