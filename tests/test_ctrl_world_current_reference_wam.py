from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.ctrl_world_current_reference_wam import (
    ARM_ID,
    MODEL_FREEZE,
    RUNTIME_RESULT_SCHEMA_VERSION,
    CallableCtrlWorldCurrentReferenceWamArm,
    stage_ctrl_world_current_reference_request,
    validate_ctrl_world_current_reference_request,
)
from blueprint_pipeline.droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _request(tmp_path: Path) -> dict[str, Any]:
    histories: dict[str, list[dict[str, str]]] = {}
    for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
        histories[view_id] = []
        for frame_index in range(6):
            path = tmp_path / "source" / f"view_{view_index}" / f"frame_{frame_index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 192), color=(view_index, frame_index, 0)).save(path)
            histories[view_id].append({"path": str(path), "sha256": file_sha256(path)})
    return {
        "schema_version": "blueprint_ctrl_world_current_reference_request.v1",
        "query_index": 0,
        "task_prompt": "Pick up the blue block.",
        "view_order": list(CTRL_WORLD_RELEASED_VIEW_ORDER),
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


def test_stage_request_copies_exact_three_view_history_and_action(tmp_path: Path) -> None:
    receipt = stage_ctrl_world_current_reference_request(
        _request(tmp_path), output_dir=tmp_path / "staged", seed=17
    )
    manifest = json.loads(Path(receipt["manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["view_order"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert manifest["action_conditioning"]["shape"] == [11, 7]
    assert manifest["seed"] == 17
    assert manifest["model_freeze"] == MODEL_FREEZE
    assert manifest["physical_future_observation_used"] is False
    assert manifest["policy_identity_in_provider_request"] is False
    assert all(len(rows) == 6 for rows in manifest["selected_history_views"].values())
    assert len(manifest["current_views"]) == 3
    assert Path(receipt["manifest_path"]).is_file()


def test_request_validation_rejects_identity_outcome_and_future_pixel_leakage(
    tmp_path: Path,
) -> None:
    for key, value, reason in (
        ("policy_id", "pi05", "request_leakage"),
        ("physical_outcome", True, "request_leakage"),
        ("physical_future_observation_used", True, "physical_future_not_false"),
    ):
        request = _request(tmp_path / key)
        request[key] = value
        with pytest.raises(ValueError, match=reason):
            validate_ctrl_world_current_reference_request(request)


def test_request_validation_rejects_wrong_hash_shape_and_view_order(tmp_path: Path) -> None:
    wrong_hash = _request(tmp_path / "hash")
    wrong_hash["selected_history_views"][CTRL_WORLD_RELEASED_VIEW_ORDER[0]][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="history_hash_mismatch"):
        validate_ctrl_world_current_reference_request(wrong_hash)

    wrong_shape = _request(tmp_path / "shape")
    wrong_shape["action_conditioning_7d"] = np.zeros((10, 7))
    with pytest.raises(ValueError, match="action_conditioning_invalid"):
        validate_ctrl_world_current_reference_request(wrong_shape)

    wrong_order = _request(tmp_path / "order")
    wrong_order["view_order"] = list(reversed(CTRL_WORLD_RELEASED_VIEW_ORDER))
    with pytest.raises(ValueError, match="view_order_invalid"):
        validate_ctrl_world_current_reference_request(wrong_order)

    wrong_geometry = _request(tmp_path / "geometry")
    row = wrong_geometry["selected_history_views"][CTRL_WORLD_RELEASED_VIEW_ORDER[0]][0]
    Image.new("RGB", (224, 224)).save(row["path"])
    row["sha256"] = file_sha256(Path(row["path"]))
    with pytest.raises(ValueError, match="history_geometry_invalid"):
        validate_ctrl_world_current_reference_request(wrong_geometry)


def test_callable_arm_returns_only_hash_bound_generated_three_view_sequences(
    tmp_path: Path,
) -> None:
    def runner(*, request_manifest_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
        manifest = json.loads(request_manifest_path.read_text(encoding="utf-8"))
        sequences: dict[str, list[str]] = {}
        hashes: dict[str, list[str]] = {}
        for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
            sequences[view_id] = []
            hashes[view_id] = []
            for frame_index in range(5):
                path = output_dir / f"view_{view_index}" / f"frame_{frame_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 12), color=(seed, view_index, frame_index)).save(path)
                sequences[view_id].append(str(path))
                hashes[view_id].append(file_sha256(path))
        return {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "arm_id": ARM_ID,
            "request_sha256": manifest["request_sha256"],
            "seed": seed,
            "model_freeze": MODEL_FREEZE,
            "generated_view_frame_sequences": sequences,
            "generated_view_frame_sha256": hashes,
            "same_frozen_wam_generated_all_views": True,
            "physical_future_observation_used": False,
            "physical_outcome_labels_accessed": False,
            "recorded_action_trace_used": False,
            "wam_to_wam_chaining": False,
        }

    arm = CallableCtrlWorldCurrentReferenceWamArm(runner=runner, seed=7)
    result = arm.predict(_request(tmp_path), output_dir=tmp_path / "prediction")

    assert result["status"] == "completed"
    assert set(result["generated_view_frame_sequences"]) == set(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert all(len(paths) == 5 for paths in result["generated_view_frame_sequences"].values())
    assert len(result["result_sha256"]) == 64
    assert result["blueprint_current_reference_not_exact_paper_reproduction"] is True


def test_callable_arm_rejects_unbound_or_unsafe_runtime_result(tmp_path: Path) -> None:
    def runner(**kwargs: Any) -> dict[str, Any]:
        request = json.loads(Path(kwargs["request_manifest_path"]).read_text(encoding="utf-8"))
        return {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "arm_id": ARM_ID,
            "request_sha256": request["request_sha256"],
            "seed": kwargs["seed"],
            "model_freeze": MODEL_FREEZE,
            "same_frozen_wam_generated_all_views": True,
            "physical_future_observation_used": True,
            "physical_outcome_labels_accessed": False,
            "recorded_action_trace_used": False,
            "wam_to_wam_chaining": False,
            "generated_view_frame_sequences": {},
            "generated_view_frame_sha256": {},
        }

    arm = CallableCtrlWorldCurrentReferenceWamArm(runner=runner, seed=3)
    with pytest.raises(ValueError, match="physical_future_observation_used_not_false"):
        arm.predict(_request(tmp_path), output_dir=tmp_path / "prediction")


def test_result_validator_rebases_portable_provider_paths(tmp_path: Path) -> None:
    from blueprint_pipeline.ctrl_world_current_reference_wam import (
        validate_ctrl_world_current_reference_result,
    )

    receipt = stage_ctrl_world_current_reference_request(
        _request(tmp_path), output_dir=tmp_path / "request", seed=41
    )
    root = tmp_path / "extracted"
    sequences: dict[str, list[str]] = {}
    hashes: dict[str, list[str]] = {}
    for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
        sequences[view_id] = []
        hashes[view_id] = []
        for frame_index in range(5):
            relative = Path(f"view_{view_index}/frame_{frame_index}.png")
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 192), (view_index, frame_index, 0)).save(path)
            sequences[view_id].append(relative.as_posix())
            hashes[view_id].append(file_sha256(path))
    media = []
    for role in ("combined_three_view", "view_0", "view_1", "view_2"):
        relative = Path("generated_video") / f"{role}.mp4"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"video:{role}".encode())
        media.append({"role": role, "path": relative.as_posix(), "sha256": file_sha256(path)})
    result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "arm_id": ARM_ID,
        "request_sha256": receipt["request_sha256"],
        "seed": 41,
        "model_freeze": MODEL_FREEZE,
        "artifact_path_mode": "result_root_relative",
        "generated_view_frame_sequences": sequences,
        "generated_view_frame_sha256": hashes,
        "generated_media": {"generated_only": True, "media": media},
        "same_frozen_wam_generated_all_views": True,
        "physical_future_observation_used": False,
        "physical_outcome_labels_accessed": False,
        "recorded_action_trace_used": False,
        "wam_to_wam_chaining": False,
    }
    result["result_sha256"] = canonical_sha256(result)

    validated = validate_ctrl_world_current_reference_result(
        result, request_receipt=receipt, seed=41, result_root=root
    )

    assert all(
        Path(path).is_absolute()
        for paths in validated["generated_view_frame_sequences"].values()
        for path in paths
    )
    assert Path(validated["generated_rollout_video_path"]).is_absolute()


def test_portable_result_requires_explicit_extraction_root(tmp_path: Path) -> None:
    from blueprint_pipeline.ctrl_world_current_reference_wam import (
        validate_ctrl_world_current_reference_result,
    )

    receipt = stage_ctrl_world_current_reference_request(
        _request(tmp_path), output_dir=tmp_path / "request", seed=43
    )
    with pytest.raises(ValueError, match="result_root_required"):
        validate_ctrl_world_current_reference_result(
            {
                "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
                "status": "completed",
                "arm_id": ARM_ID,
                "request_sha256": receipt["request_sha256"],
                "seed": 43,
                "model_freeze": MODEL_FREEZE,
                "artifact_path_mode": "result_root_relative",
                "generated_view_frame_sequences": {
                    view: ["frame.png"] * 5 for view in CTRL_WORLD_RELEASED_VIEW_ORDER
                },
                "generated_view_frame_sha256": {
                    view: ["0" * 64] * 5 for view in CTRL_WORLD_RELEASED_VIEW_ORDER
                },
                "same_frozen_wam_generated_all_views": True,
                "physical_future_observation_used": False,
                "physical_outcome_labels_accessed": False,
                "recorded_action_trace_used": False,
                "wam_to_wam_chaining": False,
                "result_sha256": "0" * 64,
            },
            request_receipt=receipt,
            seed=43,
        )
