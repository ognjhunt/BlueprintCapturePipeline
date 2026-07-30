from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline.ctrl_world_joint_position_reference_wam import MODEL_FREEZE
from blueprint_pipeline.ctrl_world_joint_position_runtime_assets import (
    stage_ctrl_world_runtime_assets,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


def test_stage_downloads_public_exact_snapshots_before_policy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixtures = {
        "ctrl_world": ("checkpoint.pt", b"ctrl-world fixture"),
        "stable_video_diffusion": ("model/config.json", b"svd fixture"),
        "clip": ("config.json", b"clip fixture"),
    }
    checkpoint_name, checkpoint_bytes = fixtures["ctrl_world"]
    checkpoint_fixture = tmp_path / "checkpoint-fixture"
    checkpoint_fixture.write_bytes(checkpoint_bytes)
    monkeypatch.setitem(
        MODEL_FREEZE,
        "ctrl_world_checkpoint",
        {
            "repository": "public/ctrl-world",
            "revision": "a" * 40,
            "file": checkpoint_name,
            "size_bytes": len(checkpoint_bytes),
            "sha256": file_sha256(checkpoint_fixture),
        },
    )
    for name in ("stable_video_diffusion", "clip"):
        relative, data = fixtures[name]
        fixture = tmp_path / f"{name}-fixture"
        fixture.write_bytes(data)
        monkeypatch.setitem(
            MODEL_FREEZE,
            name,
            {
                "repository": f"public/{name}",
                "revision": ("b" if name == "stable_video_diffusion" else "c") * 40,
                "required_files": [
                    {
                        "relative_path": relative,
                        "size_bytes": len(data),
                        "sha256": file_sha256(fixture),
                    }
                ],
            },
        )
    calls: list[dict[str, Any]] = []

    def downloader(**kwargs: Any) -> str:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        name = next(
            model_name
            for model_name in fixtures
            if MODEL_FREEZE["ctrl_world_checkpoint" if model_name == "ctrl_world" else model_name][
                "repository"
            ]
            == kwargs["repo_id"]
        )
        relative, data = fixtures[name]
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(target)

    result = stage_ctrl_world_runtime_assets(
        model_root=tmp_path / "models",
        output_dir=tmp_path / "evidence",
        downloader=downloader,
    )

    assert result["status"] == "completed"
    assert result["stage_completed_before_policy_load"] is True
    assert result["rankings_or_policy_outcomes_accessed"] is False
    assert len(calls) == 3
    assert all(call["token"] is False for call in calls)
    assert all(call["max_workers"] == 8 for call in calls)
    for name, snapshot in result["snapshots"].items():
        marker = Path(snapshot["root"]) / ".blueprint_snapshot_identity.json"
        assert json.loads(marker.read_text()) == {
            "repository": snapshot["repository"],
            "revision": snapshot["revision"],
        }
        assert name in fixtures

    clip_file = Path(result["paths"]["clip_model_root"]) / fixtures["clip"][0]
    clip_file.write_bytes(b"drift")
    with pytest.raises(ValueError, match="ctrl_world_asset_stage_file_mismatch:clip"):
        stage_ctrl_world_runtime_assets(
            model_root=tmp_path / "models",
            output_dir=tmp_path / "drift-evidence",
            downloader=lambda **kwargs: str(kwargs["local_dir"]),
        )
