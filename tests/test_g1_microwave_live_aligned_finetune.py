from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import g1_microwave_live_aligned_finetune as aligned
from blueprint_pipeline.gear_sonic_joint_order_contract import (
    PROTOCOL_V4_FULL_JOINT_ORDER,
)


def test_canonical_joint_positions_requires_and_preserves_protocol_order() -> None:
    inventory = [
        {
            "normalized_name": name,
            "observed_name": name,
            "observed_index": index,
            "position": index / 10.0,
        }
        for index, name in reversed(list(enumerate(PROTOCOL_V4_FULL_JOINT_ORDER)))
    ]
    values = aligned._canonical_joint_positions(
        {"proprioception_mapping": {"observed_dof_inventory": inventory}}
    )
    assert values == pytest.approx(
        [index / 10.0 for index in range(len(PROTOCOL_V4_FULL_JOINT_ORDER))]
    )


def test_canonical_joint_positions_fails_closed_on_incomplete_inventory() -> None:
    with pytest.raises(ValueError, match="initial_joint_inventory_mismatch"):
        aligned._canonical_joint_positions(
            {
                "proprioception_mapping": {
                    "observed_dof_inventory": [
                        {
                            "normalized_name": PROTOCOL_V4_FULL_JOINT_ORDER[0],
                            "position": 0.0,
                        }
                    ]
                }
            }
        )


def test_numeric_stats_records_directional_distribution() -> None:
    values = np.asarray([[0.0, 4.0], [2.0, 8.0], [4.0, 12.0]])
    result = aligned._numeric_stats(values)
    assert result["mean"] == pytest.approx([2.0, 8.0])
    assert result["min"] == [0.0, 4.0]
    assert result["max"] == [4.0, 12.0]
    assert result["std"] == pytest.approx(
        [np.std(values[:, 0]), np.std(values[:, 1])]
    )


def test_live_aligned_grasp_uses_qualified_palm_down_convention() -> None:
    assert aligned.LIVE_ALIGNED_HAND_AXIS_POLARITY == -1.0
    assert aligned.LIVE_ALIGNED_GRASP_YAW_RAD == 0.0


def test_head_render_is_finalized_before_backend_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = tmp_path / "seed"
    frames = seed / "isaac_head_frames"
    frames.mkdir(parents=True)
    stage = tmp_path / "KitchenRoom.usd"
    stage.write_bytes(b"stage")

    def fake_run(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        assert any(value.endswith("frame_%06d.png") for value in argv)
        Path(argv[-1]).write_bytes(b"encoded-video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(aligned.subprocess, "run", fake_run)
    report = aligned._finalize_isaac_head_render(
        seed=seed,
        frames_dir=frames,
        stage_path=stage,
    )

    assert report["status"] == "exact_isaac_rigid_head_episode_rendered"
    assert report["third_person_used_for_policy"] is False
    assert (seed / "ego_view.mp4").read_bytes() == b"encoded-video"
    assert (seed / "live_aligned_isaac_render_report.json").is_file()
