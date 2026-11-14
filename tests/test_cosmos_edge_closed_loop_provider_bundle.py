from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.cosmos_edge_closed_loop_provider_bundle import (
    build_cosmos_edge_policy_canary_bundle,
)


def test_policy_canary_bundle_rejects_incomplete_source_hash_inventory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy_canary_source_hash_views_mismatch"):
        build_cosmos_edge_policy_canary_bundle(
            output_dir=tmp_path / "out",
            policy_snapshot_manifest_path=tmp_path / "not-read.json",
            view_first_frames={
                "observation/wrist_image_left": tmp_path / "not-read.png",
                "observation/exterior_image_1_left": tmp_path / "not-read.png",
                "observation/exterior_image_2_left": tmp_path / "not-read.png",
            },
            source_first_frame_sha256_by_view={"observation/wrist_image_left": "a" * 64},
            joint_position=np.zeros(7),
            gripper_position=np.zeros(1),
            prompt="Pick up the bottle.",
            oscar_fixture_first_frame=tmp_path / "not-read.png",
            oscar_fixture_skeleton=tmp_path / "not-read.mp4",
        )
