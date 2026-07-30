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
)
from blueprint_pipeline.droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
)
from blueprint_pipeline.openpi_current_reference_gpu_bundle import (
    build_current_reference_gpu_input_bundle,
    extract_current_reference_gpu_input_bundle,
)
from blueprint_pipeline.openpi_current_reference_observation import (
    build_generated_current_reference_policy_observation,
    validate_current_reference_policy_observation_manifest,
    write_current_reference_transition_evidence,
)
from blueprint_pipeline.openpi_current_reference_policy_canary import (
    load_current_reference_initial_observation,
)
from blueprint_pipeline.openpi_current_reference_droid_policy_runtime import (
    CURRENT_REFERENCE_INVENTORY_FILES,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wam_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "wam-output"
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
    request_receipt = {
        "status": "completed",
        "request_sha256": "a" * 64,
        "seed": 17,
    }
    request_receipt_path = tmp_path / "wam-request-receipt.json"
    _write_json(request_receipt_path, request_receipt)
    result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "arm_id": ARM_ID,
        "request_sha256": request_receipt["request_sha256"],
        "seed": request_receipt["seed"],
        "model_freeze": MODEL_FREEZE,
        "artifact_path_mode": "result_root_relative",
        "generated_view_frame_sequences": sequences,
        "generated_view_frame_sha256": hashes,
        "same_frozen_wam_generated_all_views": True,
        "physical_future_observation_used": False,
        "physical_outcome_labels_accessed": False,
        "recorded_action_trace_used": False,
        "wam_to_wam_chaining": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    result_path = root / "wam_result.json"
    _write_json(result_path, result)
    return root, result_path, request_receipt_path


def _policy_receipt(tmp_path: Path) -> Path:
    action = tmp_path / "pi05_droid_native_action.npy"
    np.save(action, np.zeros((15, 8), dtype=np.float64), allow_pickle=False)
    receipt = {
        "schema_version": "openpi_current_reference_policy_query_receipt.v1",
        "policy_id": "pi05_droid",
        "policy_identity": {"identity_sha256": "b" * 64},
        "query": {"receipt_sha256": "c" * 64},
        "native_action_path": "pi05_droid_native_action.npy",
        "native_action_file_sha256": file_sha256(action),
        "physical_outcome_accessed": False,
        "wam_called": False,
    }
    receipt["manifest_sha256"] = canonical_sha256(receipt)
    path = tmp_path / "policy-receipt.json"
    _write_json(path, receipt)
    return path


def _transition_evidence(
    tmp_path: Path, *, policy_receipt: Path, request_receipt: Path
) -> Path:
    native_action = tmp_path / "pi05_droid_native_action.npy"
    prepared = {
        "wam_request": {"physical_future_observation_used": False},
        "native_policy_action_path": str(native_action),
        "native_policy_action_sha256": file_sha256(native_action),
        "native_policy_action_shape": [15, 8],
        "action_adapter_evidence": {
            "conditioning_sha256": "e" * 64,
            "physical_future_observation_used": False,
            "task_outcome_accessed": False,
        },
        "next_joint_position": np.arange(7, dtype=np.float64) / 10,
        "next_gripper_position": np.asarray([0.25]),
        "next_cartesian_pose_7d": np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.25]),
        "physical_future_observation_used": False,
    }
    path = tmp_path / "transition-evidence.json"
    write_current_reference_transition_evidence(
        prepared_transition=prepared,
        prior_policy_receipt_path=policy_receipt,
        wam_request_receipt_path=request_receipt,
        output_path=path,
    )
    return path


def test_generated_wam_observation_round_trips_through_existing_policy_bundle(
    tmp_path: Path,
) -> None:
    root, result_path, request_receipt = _wam_evidence(tmp_path)
    policy_receipt = _policy_receipt(tmp_path)
    built = build_generated_current_reference_policy_observation(
        wam_result_path=result_path,
        wam_result_root=root,
        wam_request_receipt_path=request_receipt,
        prior_policy_receipt_path=policy_receipt,
        transition_evidence_path=_transition_evidence(
            tmp_path, policy_receipt=policy_receipt, request_receipt=request_receipt
        ),
        task_prompt="Move the banana to the right",
        query_index=1,
        output_dir=tmp_path / "generated-observation",
    )
    manifest_path = Path(built["manifest_path"])
    image_preprocessor = lambda path: np.asarray(  # noqa: E731
        Image.open(path).convert("RGB").resize((224, 224)), dtype=np.uint8
    )
    observation = load_current_reference_initial_observation(
        manifest_path, image_preprocessor=image_preprocessor
    )

    assert built["status"] == "completed"
    assert observation["prompt"] == "Move the banana to the right"
    assert np.array_equal(observation["observation/joint_position"], np.arange(7) / 10)
    assert all(observation[view_id].shape == (224, 224, 3) for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER)

    source_freeze = tmp_path / "source-freeze.json"
    source_freeze.write_text("{}\n", encoding="utf-8")
    inventories = tmp_path / "inventories"
    inventories.mkdir()
    for name in CURRENT_REFERENCE_INVENTORY_FILES.values():
        (inventories / name).write_text("{}\n", encoding="utf-8")
    bundle_path = tmp_path / "generated-observation-input.zip"
    receipt = build_current_reference_gpu_input_bundle(
        source_freeze_path=source_freeze,
        checkpoint_inventory_dir=inventories,
        initial_observation_manifest_path=manifest_path,
        runtime_source_commit="e" * 40,
        runtime_source_archive_url=(
            "https://codeload.github.com/ognjhunt/BlueprintCapturePipeline/tar.gz/" + "e" * 40
        ),
        runtime_source_archive_sha256="f" * 64,
        image_source_commit="1" * 40,
        output_zip=bundle_path,
    )
    extracted = extract_current_reference_gpu_input_bundle(
        bundle_path=bundle_path,
        expected_bundle_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )
    portable_observation = load_current_reference_initial_observation(
        extracted["initial_observation_manifest_path"],
        image_preprocessor=image_preprocessor,
    )
    assert np.array_equal(
        portable_observation["observation/joint_position"], np.arange(7) / 10
    )


def test_generated_observation_validator_rejects_future_physical_pixels(tmp_path: Path) -> None:
    root, result_path, request_receipt = _wam_evidence(tmp_path)
    policy_receipt = _policy_receipt(tmp_path)
    built = build_generated_current_reference_policy_observation(
        wam_result_path=result_path,
        wam_result_root=root,
        wam_request_receipt_path=request_receipt,
        prior_policy_receipt_path=policy_receipt,
        transition_evidence_path=_transition_evidence(
            tmp_path, policy_receipt=policy_receipt, request_receipt=request_receipt
        ),
        task_prompt="Move the banana to the right",
        query_index=1,
        output_dir=tmp_path / "generated-observation",
    )
    manifest = json.loads(Path(built["manifest_path"]).read_text(encoding="utf-8"))
    manifest["physical_future_rgb_used"] = True
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )

    with pytest.raises(ValueError, match="claim_boundary_invalid"):
        validate_current_reference_policy_observation_manifest(manifest)
