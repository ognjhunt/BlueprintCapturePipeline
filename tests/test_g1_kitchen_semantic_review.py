from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline.g1_kitchen_semantic_review import (
    run_full_episode_semantic_review,
)


def _frames(root: Path) -> None:
    frames = root / "frames"
    frames.mkdir(parents=True)
    for role in ("overview", "robot_pov"):
        for index in range(2):
            Image.new("RGB", (32, 24), (40 + 20 * index, 80, 120)).save(
                frames / f"{role}_{index:04d}.png"
            )


def _command(tmp_path: Path, *, omit_last: bool = False) -> Path:
    command = tmp_path / "semantic_review.py"
    command.write_text(
        f"""
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ['BLUEPRINT_G1_KITCHEN_SEMANTIC_REVIEW_INPUT']).read_text())
reviews = []
for row in request['frames']{'[:-1]' if omit_last else ''}:
    review = dict(row)
    if row['camera_role'] == 'overview':
        review.update(g1_visible=True, target_visible=True, floor_support_visible=True,
                      orientation_visible=True, clearance_visible=True,
                      robot_pixel_occupancy=0.2, target_pixel_occupancy=0.1)
    else:
        review.update(target_visible=True, active_hand_wrist_chain_visible=True)
    reviews.append(review)
payload = {{
    'status': 'passed', 'abstained': False, 'review_runtime_id': 'api-run-1',
    'provider': 'test-provider', 'model': 'test-model', 'frame_reviews': reviews,
}}
Path(os.environ['BLUEPRINT_G1_KITCHEN_SEMANTIC_REVIEW_OUTPUT']).write_text(json.dumps(payload))
""".strip()
    )
    return command


def test_external_semantic_review_binds_every_ordered_frame(tmp_path: Path) -> None:
    _frames(tmp_path)
    result = run_full_episode_semantic_review(
        scenario_dir=tmp_path,
        expected_frame_count=2,
        command=f"python {str(_command(tmp_path))}",
        allow=True,
    )
    assert result["status"] == "passed"
    assert result["frame_review_count"] == 4
    assert result["review_source"] == "external_semantic_review_api"
    semantics = json.loads((tmp_path / "full_episode_frame_semantics.json").read_text())
    assert len(semantics["frames"]) == 4


def test_external_semantic_review_fails_closed_on_partial_coverage(tmp_path: Path) -> None:
    _frames(tmp_path)
    result = run_full_episode_semantic_review(
        scenario_dir=tmp_path,
        expected_frame_count=2,
        command=f"python {str(_command(tmp_path, omit_last=True))}",
        allow=True,
    )
    assert result["status"] == "blocked"
    assert "semantic_review_frame_coverage_incomplete" in result["blockers"]


def test_external_semantic_review_requires_explicit_allow_and_command(tmp_path: Path) -> None:
    _frames(tmp_path)
    result = run_full_episode_semantic_review(
        scenario_dir=tmp_path,
        expected_frame_count=2,
        command=None,
        allow=False,
    )
    assert result["status"] == "blocked"
    assert any(blocker.startswith("missing_explicit_allow") for blocker in result["blockers"])
    assert any(blocker.startswith("missing_semantic_review_command") for blocker in result["blockers"])
