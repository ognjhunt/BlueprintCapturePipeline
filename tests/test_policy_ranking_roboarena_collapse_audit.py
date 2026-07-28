from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.policy_ranking_roboarena_collapse_audit import _episode_metrics


def test_static_episode_is_retained_and_recommends_abstention(tmp_path: Path) -> None:
    sampled = []
    for index in range(32):
        image = np.full((48, 64), 90, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(encoded.tobytes())
        sampled.append(
            {
                "relative_output_path": path.name,
                "encoded_jpeg_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    result = _episode_metrics(
        {
            "request_id": "r",
            "session_id": "s",
            "policy_id_internal_only": "p",
            "short_episode_source": False,
            "sampled_frames": sampled,
        },
        tmp_path,
    )
    assert "static_or_frozen_future" in result["deterministic_collapse_flags"]
    assert result["safety_abstention_recommended"] is True
    assert result["retained_in_dataset"] is True
