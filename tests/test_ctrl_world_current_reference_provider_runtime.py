from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.ctrl_world_current_reference_provider_runtime import (
    EXPECTED_STATE_STAT_SHA256,
    EXPECTED_WORLD_MODEL_SHA256,
    VIEW_ORDER,
    run_ctrl_world_current_reference_runtime,
    validate_staged_request,
)
from blueprint_pipeline.ctrl_world_current_reference_wam import (
    MODEL_FREEZE,
    stage_ctrl_world_current_reference_request,
)
from blueprint_pipeline.droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


def _request(tmp_path: Path) -> dict[str, Any]:
    histories: dict[str, list[dict[str, str]]] = {}
    for view_index, view_id in enumerate(VIEW_ORDER):
        histories[view_id] = []
        for frame_index in range(6):
            path = tmp_path / "frames" / f"view_{view_index}_{frame_index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 192), (view_index, frame_index, 0)).save(path)
            histories[view_id].append({"path": str(path), "sha256": file_sha256(path)})
    return {
        "schema_version": "blueprint_ctrl_world_current_reference_request.v1",
        "query_index": 0,
        "task_prompt": "Pick up the blue block.",
        "view_order": list(VIEW_ORDER),
        "selected_history_views": histories,
        "current_views": {view_id: dict(rows[-1]) for view_id, rows in histories.items()},
        "selected_history_indices": list(CTRL_WORLD_SELECTED_HISTORY_INDICES),
        "action_conditioning_7d": np.zeros((11, 7), dtype=np.float64),
        "action_conditioning_shape": [11, 7],
        "predicted_frame_count": 5,
        "executed_prefix_steps": 8,
        "executed_prefix_seconds": 8 / 15,
        "physical_future_observation_used": False,
    }


def _staged(tmp_path: Path) -> Path:
    receipt = stage_ctrl_world_current_reference_request(
        _request(tmp_path), output_dir=tmp_path / "request", seed=29
    )
    return Path(receipt["manifest_path"])


def test_runtime_validates_complete_staged_request(tmp_path: Path) -> None:
    validated = validate_staged_request(_staged(tmp_path))

    assert validated["seed"] == 29
    assert validated["action_conditioning_7d"].shape == (11, 7)
    assert set(validated["history_paths"]) == set(VIEW_ORDER)
    assert all(len(paths) == 6 for paths in validated["history_paths"].values())
    assert set(validated["current_paths"]) == set(VIEW_ORDER)


def test_runtime_rejects_manifest_or_staged_byte_drift(tmp_path: Path) -> None:
    manifest_path = _staged(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["task_prompt"] = "Changed after freeze"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="request_digest_mismatch"):
        validate_staged_request(manifest_path)

    manifest_path = _staged(tmp_path / "frame")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = payload["selected_history_views"][VIEW_ORDER[0]][0]
    (manifest_path.parent / row["relative_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="history_hash_mismatch"):
        validate_staged_request(manifest_path)


def test_runtime_contract_with_fake_executor_never_needs_future_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _staged(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "models.py"
    source_file.write_text("# frozen fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    stats = tmp_path / "stat.json"
    stats.write_text('{"state_01":[0,0,0,0,0,0,0],"state_99":[1,1,1,1,1,1,1]}')

    from blueprint_pipeline import ctrl_world_current_reference_provider_runtime as runtime

    monkeypatch.setattr(runtime, "EXPECTED_WORLD_MODEL_SHA256", file_sha256(checkpoint))
    monkeypatch.setattr(runtime, "EXPECTED_STATE_STAT_SHA256", file_sha256(stats))
    monkeypatch.setattr(
        "cv2.imread",
        lambda *_args, **_kwargs: pytest.fail(
            "retained generated PNG decoding must not depend on provider OpenCV/libpng"
        ),
    )

    def executor(
        *, validated_request: dict[str, Any], output_dir: Path, **_: Any
    ) -> dict[str, Any]:
        assert validated_request["manifest"]["physical_future_observation_used"] is False
        sequences: dict[str, list[str]] = {}
        hashes: dict[str, list[str]] = {}
        for view_index, view_id in enumerate(VIEW_ORDER):
            sequences[view_id] = []
            hashes[view_id] = []
            for frame_index in range(5):
                path = output_dir / f"view_{view_index}" / f"frame_{frame_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (320, 192), (29, view_index, frame_index)).save(path)
                sequences[view_id].append(str(path))
                hashes[view_id].append(file_sha256(path))
        return {
            "generated_view_frame_sequences": sequences,
            "generated_view_frame_sha256": hashes,
            "timing": {"inference_and_decode_seconds": 1.0},
            "cuda": {"device_count": 1},
            "randomness": {"seed": 29},
        }

    source_manifest = {
        "repository": MODEL_FREEZE["ctrl_world_source"]["repository"],
        "revision": MODEL_FREEZE["ctrl_world_source"]["revision"],
        "files": [
            {
                "relative_path": "models.py",
                "size_bytes": source_file.stat().st_size,
                "sha256": file_sha256(source_file),
            }
        ],
    }
    result = run_ctrl_world_current_reference_runtime(
        request_manifest_path=manifest_path,
        output_dir=tmp_path / "output",
        source_root=source,
        source_manifest=source_manifest,
        world_model_checkpoint=checkpoint,
        svd_model_root=tmp_path / "svd",
        clip_model_root=tmp_path / "clip",
        state_stat_path=stats,
        executor=executor,
    )

    assert result["status"] == "completed"
    assert result["same_frozen_wam_generated_all_views"] is True
    assert result["physical_future_observation_used"] is False
    assert result["physical_outcome_labels_accessed"] is False
    assert result["recorded_action_trace_used"] is False
    assert result["candidate_policy_loaded_by_wam_runtime"] is False
    assert all(len(paths) == 5 for paths in result["generated_view_frame_sequences"].values())
    assert result["generated_media"]["status"] == "completed"
    assert len(result["generated_media"]["media"]) == 4
    assert result["generated_media"]["physical_pixels_included"] is False
    assert result["artifact_path_mode"] == "result_root_relative"
    assert not Path(result["generated_rollout_video_path"]).is_absolute()
    assert (tmp_path / "output" / result["generated_rollout_video_path"]).is_file()
    assert all(row["frame_count"] == 5 for row in result["generated_media"]["media"])
    assert (tmp_path / "output/wam_runtime_result.json").is_file()


def test_runtime_rejects_generated_png_hash_drift_before_media_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _staged(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    source_file = source / "models.py"
    source_file.write_text("# frozen fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    stats = tmp_path / "stat.json"
    stats.write_text('{"state_01":[0,0,0,0,0,0,0],"state_99":[1,1,1,1,1,1,1]}')
    from blueprint_pipeline import ctrl_world_current_reference_provider_runtime as runtime

    monkeypatch.setattr(runtime, "EXPECTED_WORLD_MODEL_SHA256", file_sha256(checkpoint))
    monkeypatch.setattr(runtime, "EXPECTED_STATE_STAT_SHA256", file_sha256(stats))

    def executor(*, output_dir: Path, **_: Any) -> dict[str, Any]:
        sequences: dict[str, list[str]] = {}
        hashes: dict[str, list[str]] = {}
        for view_index, view_id in enumerate(VIEW_ORDER):
            sequences[view_id] = []
            hashes[view_id] = []
            for frame_index in range(5):
                path = output_dir / f"v{view_index}_{frame_index}.png"
                Image.new("RGB", (320, 192), (view_index, frame_index, 0)).save(path)
                sequences[view_id].append(str(path))
                hashes[view_id].append("0" * 64)
        return {
            "generated_view_frame_sequences": sequences,
            "generated_view_frame_sha256": hashes,
        }

    with pytest.raises(ValueError, match="generated_media_hash_mismatch"):
        run_ctrl_world_current_reference_runtime(
            request_manifest_path=manifest_path,
            output_dir=tmp_path / "output",
            source_root=source,
            source_manifest={
                "repository": MODEL_FREEZE["ctrl_world_source"]["repository"],
                "revision": MODEL_FREEZE["ctrl_world_source"]["revision"],
                "files": [
                    {
                        "relative_path": "models.py",
                        "size_bytes": source_file.stat().st_size,
                        "sha256": file_sha256(source_file),
                    }
                ],
            },
            world_model_checkpoint=checkpoint,
            svd_model_root=tmp_path / "svd",
            clip_model_root=tmp_path / "clip",
            state_stat_path=stats,
            executor=executor,
        )


def test_runtime_frozen_hash_constants_are_full_sha256() -> None:
    assert len(EXPECTED_WORLD_MODEL_SHA256) == 64
    assert len(EXPECTED_STATE_STAT_SHA256) == 64
