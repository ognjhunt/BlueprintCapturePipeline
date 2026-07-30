from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.ctrl_world_joint_position_reference_wam import (
    ARM_ID,
    MODEL_FREEZE,
    RUNTIME_RESULT_SCHEMA_VERSION,
    CallableCtrlWorldJointPositionReferenceWamArm,
    stage_ctrl_world_joint_position_request,
    validate_ctrl_world_joint_position_request,
)
from blueprint_pipeline.droid_ctrl_world_joint_position_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
    REQUEST_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256
from blueprint_pipeline.new_site_diagnostic_canary_gpu import MultiViewCanaryReliabilityGate
from blueprint_pipeline.wam_rollout_reliability import ReliabilityThresholds


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
        "schema_version": REQUEST_SCHEMA_VERSION,
        "query_index": 0,
        "task_prompt": "Pick up the spray can and place it inside the marked tray.",
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


def test_stage_request_binds_three_view_history_action_and_model_freeze(tmp_path: Path) -> None:
    receipt = stage_ctrl_world_joint_position_request(
        _request(tmp_path), output_dir=tmp_path / "staged", seed=17
    )
    manifest = json.loads(Path(receipt["manifest_path"]).read_text(encoding="utf-8"))

    assert manifest["view_order"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert manifest["action_conditioning"]["shape"] == [11, 7]
    assert manifest["seed"] == 17
    assert manifest["model_freeze"] == MODEL_FREEZE
    assert manifest["physical_future_observation_used"] is False
    assert manifest["policy_identity_in_provider_request"] is False
    assert manifest["label_free"] is True
    assert all(len(rows) == 6 for rows in manifest["selected_history_views"].values())
    assert set(manifest["current_views"]) == set(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert all(
        (Path(receipt["request_dir"]) / row["relative_path"]).is_file()
        for row in manifest["current_views"].values()
    )


def test_request_rejects_identity_outcome_future_pixels_and_contract_drift(
    tmp_path: Path,
) -> None:
    for key, value, reason in (
        ("policy_id", "pi05", "request_leakage"),
        ("physical_outcome", True, "request_leakage"),
        ("ranking", [1, 2, 3], "request_leakage"),
        ("physical_future_observation_used", True, "physical_future_not_false"),
    ):
        request = _request(tmp_path / key)
        request[key] = value
        with pytest.raises(ValueError, match=reason):
            validate_ctrl_world_joint_position_request(request)

    wrong_shape = _request(tmp_path / "shape")
    wrong_shape["action_conditioning_7d"] = np.zeros((10, 7))
    with pytest.raises(ValueError, match="action_conditioning_invalid"):
        validate_ctrl_world_joint_position_request(wrong_shape)

    wrong_view = _request(tmp_path / "view")
    wrong_view["view_order"] = list(reversed(CTRL_WORLD_RELEASED_VIEW_ORDER))
    with pytest.raises(ValueError, match="view_order_invalid"):
        validate_ctrl_world_joint_position_request(wrong_view)


def test_callable_arm_returns_hash_bound_frames_and_individual_videos(tmp_path: Path) -> None:
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
                Image.new(
                    "RGB", (320, 192), color=(seed, view_index, frame_index)
                ).save(path)
                sequences[view_id].append(str(path))
                hashes[view_id].append(file_sha256(path))
        return {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "arm_id": ARM_ID,
            "request_sha256": manifest["request_sha256"],
            "seed": seed,
            "model_freeze": MODEL_FREEZE,
            "runtime_asset_admission_passed": True,
            "generated_view_frame_sequences": sequences,
            "generated_view_frame_sha256": hashes,
            "same_frozen_wam_generated_all_views": True,
            "physical_future_observation_used": False,
            "physical_outcome_labels_accessed": False,
            "recorded_action_trace_used": False,
            "wam_to_wam_chaining": False,
        }

    result = CallableCtrlWorldJointPositionReferenceWamArm(runner=runner, seed=7).predict(
        _request(tmp_path), output_dir=tmp_path / "prediction"
    )

    assert result["status"] == "completed"
    assert set(result["generated_view_frame_sequences"]) == set(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert all(len(paths) == 5 for paths in result["generated_view_frame_sequences"].values())
    assert set(result["generated_videos_by_view"]) == set(CTRL_WORLD_RELEASED_VIEW_ORDER)
    assert all(Path(path).stat().st_size > 0 for path in result["generated_videos_by_view"].values())
    assert all(
        evidence["frame_count"] == 5
        for evidence in result["generated_video_evidence_by_view"].values()
    )
    assert result["blueprint_joint_position_reference_not_exact_paper_reproduction"] is True
    assert len(result["result_sha256"]) == 64

    report = SimpleNamespace(flags=(), as_dict=lambda: {"flags": [], "reliable": True})
    gate = MultiViewCanaryReliabilityGate(
        ReliabilityThresholds(),
        required_views=CTRL_WORLD_RELEASED_VIEW_ORDER,
        assessor=lambda *_args, **_kwargs: report,
        gate_id="new_site_ctrl_world_three_view_reliability_v1",
    )
    assessment = gate.assess(
        previous_observation={},
        prepared_transition={"reliability_actions_10d": np.zeros((5, 10))},
        wam_prediction=result,
        query_index=0,
        output_dir=tmp_path / "reliability",
    )
    assert assessment["status"] == "passed"
    assert assessment["required_views"] == list(CTRL_WORLD_RELEASED_VIEW_ORDER)


def test_callable_arm_rejects_unverified_or_future_observation_result(tmp_path: Path) -> None:
    def runner(**kwargs: Any) -> dict[str, Any]:
        request = json.loads(Path(kwargs["request_manifest_path"]).read_text(encoding="utf-8"))
        return {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "arm_id": ARM_ID,
            "request_sha256": request["request_sha256"],
            "seed": kwargs["seed"],
            "model_freeze": MODEL_FREEZE,
            "runtime_asset_admission_passed": True,
            "same_frozen_wam_generated_all_views": True,
            "physical_future_observation_used": True,
            "physical_outcome_labels_accessed": False,
            "recorded_action_trace_used": False,
            "wam_to_wam_chaining": False,
            "generated_view_frame_sequences": {},
            "generated_view_frame_sha256": {},
        }

    arm = CallableCtrlWorldJointPositionReferenceWamArm(runner=runner, seed=3)
    with pytest.raises(ValueError, match="physical_future_observation_used_not_false"):
        arm.predict(_request(tmp_path), output_dir=tmp_path / "prediction")
