from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import ctrl_world_joint_position_reference_runtime as runtime_module
from blueprint_pipeline.ctrl_world_joint_position_reference_runtime import (
    CtrlWorldJointPositionReferenceRuntime,
    CtrlWorldJointPositionSubprocessRuntime,
    validate_staged_joint_position_request,
)
from blueprint_pipeline.ctrl_world_joint_position_reference_wam import (
    ARM_ID,
    MODEL_FREEZE,
    PREDICTED_FRAME_COUNT,
    RUNTIME_RESULT_SCHEMA_VERSION,
    stage_ctrl_world_joint_position_request,
    validate_ctrl_world_joint_position_result,
)
from blueprint_pipeline.droid_ctrl_world_joint_position_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
    REQUEST_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


def _request(tmp_path: Path) -> dict[str, Any]:
    histories: dict[str, list[dict[str, str]]] = {}
    for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
        histories[view_id] = []
        for frame_index in range(6):
            path = tmp_path / "frames" / f"view_{view_index}_{frame_index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 192), (view_index, frame_index, 20)).save(path)
            histories[view_id].append({"path": str(path), "sha256": file_sha256(path)})
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "query_index": 0,
        "task_prompt": "Pick up the spray can and place it inside the marked tray.",
        "view_order": list(CTRL_WORLD_RELEASED_VIEW_ORDER),
        "selected_history_views": histories,
        "current_views": {view_id: dict(rows[-1]) for view_id, rows in histories.items()},
        "selected_history_indices": list(CTRL_WORLD_SELECTED_HISTORY_INDICES),
        "action_conditioning_7d": np.zeros((11, 7), dtype=np.float64),
        "action_conditioning_shape": [11, 7],
        "predicted_frame_count": PREDICTED_FRAME_COUNT,
        "executed_prefix_steps": 8,
        "executed_prefix_seconds": 8 / 15,
        "physical_future_observation_used": False,
    }


def _snapshot(tmp_path: Path, freeze_key: str) -> Path:
    freeze = MODEL_FREEZE[freeze_key]
    root = tmp_path / freeze_key / freeze["revision"]
    root.mkdir(parents=True)
    (root / ".blueprint_snapshot_identity.json").write_text(
        json.dumps({"repository": freeze["repository"], "revision": freeze["revision"]}),
        encoding="utf-8",
    )
    return root


def test_runtime_validates_complete_joint_position_request(tmp_path: Path) -> None:
    receipt = stage_ctrl_world_joint_position_request(
        _request(tmp_path), output_dir=tmp_path / "staged", seed=23
    )

    validated = validate_staged_joint_position_request(receipt["manifest_path"], expected_seed=23)

    assert validated["seed"] == 23
    assert validated["action_conditioning_7d"].shape == (11, 7)
    assert set(validated["history_paths"]) == set(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert all(len(paths) == 6 for paths in validated["history_paths"].values())
    assert set(validated["current_paths"]) == set(CTRL_WORLD_RELEASED_VIEW_ORDER)


def test_runtime_rejects_staged_manifest_and_frame_drift(tmp_path: Path) -> None:
    receipt = stage_ctrl_world_joint_position_request(
        _request(tmp_path), output_dir=tmp_path / "staged", seed=23
    )
    manifest_path = Path(receipt["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_prompt"] = "changed after staging"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="request_digest_mismatch"):
        validate_staged_joint_position_request(manifest_path, expected_seed=23)

    receipt = stage_ctrl_world_joint_position_request(
        _request(tmp_path / "frame"), output_dir=tmp_path / "frame_staged", seed=23
    )
    manifest_path = Path(receipt["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["selected_history_views"][CTRL_WORLD_RELEASED_VIEW_ORDER[0]][0]
    (manifest_path.parent / row["relative_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="history_hash_mismatch"):
        validate_staged_joint_position_request(manifest_path, expected_seed=23)


def test_runtime_rejects_symlinked_staged_manifest(tmp_path: Path) -> None:
    receipt = stage_ctrl_world_joint_position_request(
        _request(tmp_path), output_dir=tmp_path / "staged", seed=23
    )
    link = tmp_path / "request-link.json"
    link.symlink_to(Path(receipt["manifest_path"]))

    with pytest.raises(ValueError, match="request_missing_or_unsafe"):
        validate_staged_joint_position_request(link, expected_seed=23)


def test_configured_runtime_emits_result_accepted_by_wam_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "models.py"
    source_file.write_text("# exact source fixture\n", encoding="utf-8")
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "repository": MODEL_FREEZE["ctrl_world_source"]["repository"],
                "revision": MODEL_FREEZE["ctrl_world_source"]["revision"],
                "files": [
                    {
                        "relative_path": "models.py",
                        "sha256": file_sha256(source_file),
                        "size_bytes": source_file.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        MODEL_FREEZE["ctrl_world_source"],
        "required_files",
        json.loads(source_manifest.read_text(encoding="utf-8"))["files"],
    )
    checkpoint = tmp_path / "checkpoint-10000.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    stats = tmp_path / "stat.json"
    stats.write_text(json.dumps({"state_01": [0] * 7, "state_99": [1] * 7}), encoding="utf-8")
    monkeypatch.setitem(
        MODEL_FREEZE["ctrl_world_checkpoint"], "size_bytes", checkpoint.stat().st_size
    )
    monkeypatch.setitem(MODEL_FREEZE["ctrl_world_checkpoint"], "sha256", file_sha256(checkpoint))
    monkeypatch.setitem(MODEL_FREEZE["ctrl_world_state_stats"], "sha256", file_sha256(stats))
    svd = _snapshot(tmp_path, "stable_video_diffusion")
    clip = _snapshot(tmp_path, "clip")
    for freeze_key, root in (("stable_video_diffusion", svd), ("clip", clip)):
        blob = root / "weights.bin"
        blob.write_bytes(f"{freeze_key} fixture".encode())
        monkeypatch.setitem(
            MODEL_FREEZE[freeze_key],
            "required_blobs",
            [
                {
                    "relative_path": "weights.bin",
                    "size_bytes": blob.stat().st_size,
                    "sha256": file_sha256(blob),
                }
            ],
        )
    receipt = stage_ctrl_world_joint_position_request(
        _request(tmp_path), output_dir=tmp_path / "request", seed=23
    )

    def fake_executor(
        *, validated_request: dict[str, Any], output_dir: Path, **_kwargs: Any
    ) -> dict[str, Any]:
        assert validated_request["manifest"]["physical_future_observation_used"] is False
        sequences: dict[str, list[str]] = {}
        hashes: dict[str, list[str]] = {}
        for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
            sequences[view_id] = []
            hashes[view_id] = []
            for frame_index in range(PREDICTED_FRAME_COUNT):
                path = output_dir / f"view_{view_index}" / f"frame_{frame_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (320, 192), (23, view_index, frame_index)).save(path)
                sequences[view_id].append(str(path))
                hashes[view_id].append(file_sha256(path))
        return {
            "generated_view_frame_sequences": sequences,
            "generated_view_frame_sha256": hashes,
            "timing": {"inference_and_decode_seconds": 1.0},
            "cuda": {"device_count": 1},
            "randomness": {"seed": 23},
        }

    monkeypatch.setattr(
        runtime_module, "execute_generated_only_ctrl_world_joint_position", fake_executor
    )
    runtime = CtrlWorldJointPositionReferenceRuntime(
        source_root=source_root,
        source_manifest_path=source_manifest,
        world_model_checkpoint=checkpoint,
        svd_model_root=svd,
        clip_model_root=clip,
        state_stat_path=stats,
    )

    result = runtime(
        request_manifest_path=Path(receipt["manifest_path"]),
        output_dir=tmp_path / "output",
        seed=23,
    )
    validated = validate_ctrl_world_joint_position_result(result, request_receipt=receipt, seed=23)

    assert result["schema_version"] == RUNTIME_RESULT_SCHEMA_VERSION
    assert result["arm_id"] == ARM_ID
    assert result["runtime_asset_admission_passed"] is True
    assert result["candidate_policy_loaded_by_wam_runtime"] is False
    assert result["physical_future_observation_used"] is False
    assert result["physical_outcome_labels_accessed"] is False
    assert result["recorded_action_trace_used"] is False
    assert result["engineering_provenance"]["confirmation_session_or_result_reused"] is False
    assert all(
        len(paths) == PREDICTED_FRAME_COUNT
        for paths in validated["generated_view_frame_sequences"].values()
    )
    assert (tmp_path / "output/ctrl_world_joint_position_runtime_result.json").is_file()

    source_file.write_text("# drifted source fixture\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ctrl_world_joint_position_runtime_source_file_mismatch"):
        runtime(
            request_manifest_path=Path(receipt["manifest_path"]),
            output_dir=tmp_path / "source-drifted-output",
            seed=23,
        )
    source_file.write_text("# exact source fixture\n", encoding="utf-8")

    (svd / "weights.bin").write_bytes(b"drift after admission")
    with pytest.raises(ValueError, match="ctrl_world_joint_position_runtime_svd_snapshot_mismatch"):
        runtime(
            request_manifest_path=Path(receipt["manifest_path"]),
            output_dir=tmp_path / "drifted-output",
            seed=23,
        )


def test_subprocess_runtime_isolates_environment_and_returns_exact_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL", "secret-url")
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.update({"command": command, **kwargs})
        output = Path(command[command.index("--output-dir") + 1])
        (output / "ctrl_world_joint_position_runtime_result.json").write_text(
            json.dumps({"status": "completed", "sentinel": "exact-child-result"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "child stdout\n", "")

    monkeypatch.setattr(runtime_module.subprocess, "run", fake_run)
    runtime = CtrlWorldJointPositionSubprocessRuntime(
        python_executable=Path(sys.executable),
        source_root=tmp_path / "source",
        source_manifest_path=tmp_path / "source.json",
        world_model_checkpoint=tmp_path / "checkpoint.pt",
        svd_model_root=tmp_path / "svd",
        clip_model_root=tmp_path / "clip",
        state_stat_path=tmp_path / "stat.json",
        timeout_seconds=600,
    )

    result = runtime(
        request_manifest_path=tmp_path / "request.json",
        output_dir=tmp_path / "output",
        seed=23,
    )

    assert result == {"status": "completed", "sentinel": "exact-child-result"}
    assert observed["command"][:4] == [
        str(Path(sys.executable).resolve()),
        "-m",
        "blueprint_pipeline.ctrl_world_joint_position_reference_runtime",
        "run",
    ]
    assert "BLUEPRINT_OPENPI_POLICY_RANKING_INPUT_SECRET_URL" not in observed["env"]
    assert (tmp_path / "output/ctrl_world_subprocess_stdout.log").read_text() == ("child stdout\n")
