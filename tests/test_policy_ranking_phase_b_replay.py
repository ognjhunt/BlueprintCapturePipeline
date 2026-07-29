from __future__ import annotations

import json
import math
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.policy_ranking_phase_b_cosmos_bundle import (
    build_phase_b_cosmos_canary_bundle,
)
from blueprint_pipeline.policy_ranking_successor_gpu_admission import (
    PHASE_B_PROFILE,
    inspect_successor_bundle,
)
from blueprint_pipeline.policy_ranking_successor_cosmos import (
    SuccessorContractError,
    build_action_controls,
    droid_action_stream,
    validate_smoke_inventory_manifest,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


def _stream(offset: float) -> dict:
    actions = []
    for index in range(16):
        angle = index / 100
        actions.append(
            [
                offset + index / 10000,
                0.0,
                0.0,
                math.cos(angle),
                math.sin(angle),
                0.0,
                -math.sin(angle),
                math.cos(angle),
                0.0,
                float(index % 2),
            ]
        )
    return droid_action_stream(actions)


def test_valid_no_motion_is_identity_rot6d_with_explicit_observation_hold() -> None:
    controls = build_action_controls(
        _stream(0.0),
        _stream(0.01),
        observation_gripper_hold=1.0,
        shuffle_seed=20260728,
    )
    assert all(
        row == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        for row in controls["zero"]["actions"]
    )


def test_no_motion_rejects_nonphysical_gripper_hold() -> None:
    with pytest.raises(
        SuccessorContractError, match="observation_gripper_hold_outside_closed_unit_interval"
    ):
        build_action_controls(
            _stream(0.0),
            _stream(0.01),
            observation_gripper_hold=2.0,
            shuffle_seed=20260728,
        )


def test_native_cosmos_canary_bundle_is_immutable_and_label_blind(tmp_path: Path) -> None:
    controls = build_action_controls(
        _stream(0.0),
        _stream(0.01),
        observation_gripper_hold=1.0,
        shuffle_seed=20260728,
    )
    observation = tmp_path / "observation.png"
    Image.new("RGB", (640, 540), color=(10, 20, 30)).save(observation)
    canary = {
        "schema_version": "policy_ranking_phase_b_replay_canary.v1",
        "status": "passed",
        "controls": controls,
        "initial_observation": {
            "path": str(observation),
            "sha256": file_sha256(observation),
        },
        "label_seal": {"outcome_labels_accessed": False},
    }
    canary["manifest_sha256"] = canonical_sha256(canary)
    canary_path = tmp_path / "canary.json"
    canary_path.write_text(json.dumps(canary), encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    receipt = tmp_path / "receipt.json"
    result = build_phase_b_cosmos_canary_bundle(
        replay_canary_path=canary_path,
        output_bundle=bundle,
        receipt_path=receipt,
        task_instruction="Pick up the bottle and place it in the bin.",
    )
    assert result["status"] == "built"
    assert result["paid_resources_used"] is False
    assert result["initial_observation_sha256"]
    assert result["smoke_inventory_sha256"]
    assert result["action_streams_sha256"]
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
        assert "provider_runtime/cosmos3_input/initial_observation.png" in names
        inventory = json.loads(
            archive.read("provider_runtime/cosmos3_input/smoke_request_inventory.json")
        )
        assert validate_smoke_inventory_manifest(inventory)["status"] == "passed"
        assert inventory["outcome_labels_accessed"] is False
        assert inventory["task_instruction"] == "Pick up the bottle and place it in the bin."
        assert len(inventory["requests"]) == 10
        action_streams = json.loads(
            archive.read("provider_runtime/cosmos3_input/action_streams.json")
        )
        assert set(action_streams["conditions"]) == {
            "recorded",
            "zero",
            "shuffled",
            "reversed",
            "policy_swapped",
        }
    profile = replace(
        PHASE_B_PROFILE,
        expected_bundle_sha256=file_sha256(bundle),
        expected_bundle_size_bytes=bundle.stat().st_size,
        expected_embedded_input_hashes={
            "initial_observation_sha256": result["initial_observation_sha256"],
            "smoke_inventory_sha256": result["smoke_inventory_sha256"],
            "action_streams_sha256": result["action_streams_sha256"],
        },
    )
    inspection = inspect_successor_bundle(
        bundle,
        receipt=result,
        smoke_inventory=inventory,
        profile=profile,
    )
    assert inspection["status"] == "passed"
    assert inspection["blockers"] == []
