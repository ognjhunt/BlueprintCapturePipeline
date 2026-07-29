from __future__ import annotations

import builtins
import json
import math
import shutil
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.policy_ranking_phase_b_cosmos_bundle import (
    build_phase_b_cosmos_canary_bundle,
)
from blueprint_pipeline import policy_ranking_phase_b_cosmos_bundle as cosmos_bundle
from blueprint_pipeline.policy_ranking_phase_b_high_motion_selection import (
    build_high_motion_selection,
    build_powered_window_selection,
)
from blueprint_pipeline.policy_ranking_phase_b_replay import (
    build_powered_replay_packet,
    build_selected_replay_canary,
)
from blueprint_pipeline import policy_ranking_successor_cosmos_bundle as successor_bundle
from blueprint_pipeline.policy_ranking_successor_gpu_admission import (
    PHASE_B_POSITIVE_CONTROL_PROFILE,
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
from blueprint_pipeline.policy_ranking_task_instruction_extraction import (
    extract_task_instruction,
)


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
    controls["shifted"] = _stream(0.02)
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
        assert len(inventory["requests"]) == 12
        action_streams = json.loads(
            archive.read("provider_runtime/cosmos3_input/action_streams.json")
        )
        assert set(action_streams["conditions"]) == {
            "recorded",
            "zero",
            "shuffled",
            "reversed",
            "policy_swapped",
            "shifted",
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

    with pytest.raises(ValueError, match="task_instruction_does_not_match_replay_canary"):
        canary["task_instruction"] = "Put the bottle in the other bin."
        canary["manifest_sha256"] = canonical_sha256(
            {key: value for key, value in canary.items() if key != "manifest_sha256"}
        )
        canary_path.write_text(json.dumps(canary), encoding="utf-8")
        build_phase_b_cosmos_canary_bundle(
            replay_canary_path=canary_path,
            output_bundle=bundle,
            receipt_path=receipt,
            task_instruction="Pick up the bottle and place it in the bin.",
        )


def test_native_cosmos_bundle_embeds_hash_pinned_official_positive_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg unavailable")
    controls = build_action_controls(
        _stream(0.0),
        _stream(0.01),
        observation_gripper_hold=1.0,
        shuffle_seed=20260728,
    )
    controls["shifted"] = _stream(0.02)
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
    first_frame = tmp_path / "first.png"
    Image.new("RGB", (640, 720), color=(20, 30, 40)).save(first_frame)
    action_chunks = tmp_path / "actions.json"
    action_chunks.write_text(
        json.dumps(
            {
                "prompt": "Pickup items in the supermarket",
                "fps": 10,
                "action_chunk_size": 16,
                "domain_name": "agibotworld",
                "image_size": 480,
                "view_point": "concat_view",
                "num_chunks": 4,
                "action_chunks": [[[0.0] * 29 for _ in range(16)] for _ in range(4)],
            }
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "reference.mp4"
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=640x720:r=10:d=6.4",
            "-frames:v",
            "64",
            "-pix_fmt",
            "yuv420p",
            str(reference),
        ],
        check=False,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    monkeypatch.setattr(
        cosmos_bundle,
        "OFFICIAL_POSITIVE_CONTROL_ASSET_SHA256",
        {
            "first_frame": file_sha256(first_frame),
            "action_chunks": file_sha256(action_chunks),
            "reference_output": file_sha256(reference),
        },
    )
    bundle = tmp_path / "positive-control-bundle.zip"
    result = build_phase_b_cosmos_canary_bundle(
        replay_canary_path=canary_path,
        output_bundle=bundle,
        receipt_path=tmp_path / "receipt.json",
        task_instruction="Pick up the bottle and place it in the bin.",
        positive_control_first_frame_path=first_frame,
        positive_control_action_chunks_path=action_chunks,
        positive_control_reference_output_path=reference,
    )

    assert result["positive_control_included"] is True
    assert result["positive_control_manifest_sha256"]
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(
            archive.read("provider_runtime/cosmos3_positive_control/manifest.json")
        )
        runtime_manifest = json.loads(
            archive.read("provider_runtime/wam_provider_runtime_manifest.json")
        )
        assert manifest["frozen_gates"] == cosmos_bundle.POSITIVE_CONTROL_FROZEN_GATES
        assert runtime_manifest["positive_control_request_count"] == 4
        assert runtime_manifest["total_initial_generation_request_count"] == 18
        assert (
            runtime_manifest["request_budget_amendment_sha256"]
            == result["positive_control_manifest_sha256"]
        )
        inventory = json.loads(
            archive.read("provider_runtime/cosmos3_input/smoke_request_inventory.json")
        )
    profile = replace(
        PHASE_B_POSITIVE_CONTROL_PROFILE,
        expected_bundle_sha256=file_sha256(bundle),
        expected_bundle_size_bytes=bundle.stat().st_size,
        expected_embedded_input_hashes={
            "initial_observation_sha256": result["initial_observation_sha256"],
            "smoke_inventory_sha256": result["smoke_inventory_sha256"],
            "action_streams_sha256": result["action_streams_sha256"],
            "positive_control_manifest_sha256": result["positive_control_manifest_sha256"],
        },
        request_budget_amendment_sha256=result["positive_control_manifest_sha256"],
    )
    inspection = inspect_successor_bundle(
        bundle,
        receipt=result,
        smoke_inventory=inventory,
        profile=profile,
    )
    assert inspection["status"] == "passed"
    assert inspection["blockers"] == []


def _replay_npz(path: Path, scale: float) -> None:
    path.parent.mkdir(parents=True)
    rows = []
    position = 0.0
    for index in range(18):
        position += scale * (1.0 + index / 20.0)
        rows.append(
            {
                "cartesian_position": [position, 0.0, 0.0, 0.0, 0.0, index / 1000.0],
                "joint_position": [0.0] * 7,
                "gripper_position": [0.0],
                "action": [0.0] * 7 + [float(index >= 9)],
            }
        )
    np.savez(path, data=np.asarray(rows, dtype=object))


def _replay_video(path: Path, color: tuple[int, int, int]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((48, 64, 3), color, dtype=np.uint8)
        frame[:, : index + 1] = 255
        writer.write(frame)
    writer.release()


def test_selected_replay_canary_binds_motion_prompt_views_and_controls(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    session = root / "evaluation_sessions" / "session-a"
    for prefix, policy, scale in (
        ("A", "policy-one", 0.005),
        ("B", "policy-two", 0.003),
    ):
        directory = session / f"{prefix}_{policy}"
        _replay_npz(directory / f"{policy}_npz_file.npz", scale)
        for view, color in (
            ("left", (10, 20, 30)),
            ("right", (40, 50, 60)),
            ("wrist", (70, 80, 90)),
        ):
            _replay_video(directory / f"{policy}_video_{view}.mp4", color)
    metadata = session / "metadata.yaml"
    metadata.write_text(
        "session_id: session-a\nlanguage_instruction: put the bowl in the plate\nsuccess: true\n",
        encoding="utf-8",
    )
    split = {
        "schema_version": "policy_ranking_disjoint_session_candidate_split_amendment.v2",
        "dataset": {"id": "fixture", "revision": "frozen", "license": "mit"},
        "required_policy_ids": ["policy-one", "policy-two"],
        "selection": {"metadata_yaml_opened": False, "session_ids": ["session-a"]},
    }
    split["manifest_sha256"] = canonical_sha256(split)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    selection = build_high_motion_selection(split_manifest_path=split_path, dataset_root=root)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    task_receipt = extract_task_instruction(metadata, session_id="session-a")
    receipt_path = tmp_path / "task.json"
    receipt_path.write_text(json.dumps(task_receipt), encoding="utf-8")

    result = build_selected_replay_canary(
        high_motion_selection_path=selection_path,
        task_instruction_receipt_path=receipt_path,
        dataset_root=root,
        output_dir=tmp_path / "canary",
    )

    assert result["status"] == "passed"
    assert result["task_instruction"] == "put the bowl in the plate"
    assert result["recorded_policy_id_internal_only"] == "policy-one"
    assert set(result["initial_views"]) == {"left", "right", "wrist"}
    assert Path(result["initial_observation"]["path"]).is_file()
    assert set(result["controls"]) == {
        "recorded",
        "zero",
        "shuffled",
        "reversed",
        "policy_swapped",
        "shifted",
    }
    assert len(set(result["control_action_sha256"].values())) == 6
    assert result["access_contract"]["physical_future_pixels_in_provider_input"] is False
    assert (
        result["conditioning_modes"]["starter_video_supported_by_pinned_native_action_api"] is False
    )


def test_powered_replay_packet_materializes_every_session_window(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    session_ids = ("session-a", "session-b")
    policy_rows = (
        ("A", "policy-one", 0.005),
        ("B", "policy-two", 0.004),
        ("C", "policy-three", 0.003),
        ("D", "policy-four", 0.002),
    )
    for session_id in session_ids:
        session = root / "evaluation_sessions" / session_id
        for prefix, policy, scale in policy_rows:
            directory = session / f"{prefix}_{policy}"
            _replay_npz(directory / f"{policy}_npz_file.npz", scale)
            for view, color in (
                ("left", (10, 20, 30)),
                ("right", (40, 50, 60)),
                ("wrist", (70, 80, 90)),
            ):
                _replay_video(directory / f"{policy}_video_{view}.mp4", color)
    split = {
        "schema_version": "policy_ranking_disjoint_session_candidate_split_amendment.v3",
        "dataset": {"id": "fixture", "revision": "frozen", "license": "mit"},
        "required_policy_ids": [row[1] for row in policy_rows],
        "selection": {"metadata_yaml_opened": False, "session_ids": list(session_ids)},
    }
    split["manifest_sha256"] = canonical_sha256(split)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    selection = build_powered_window_selection(
        split_manifest_path=split_path,
        dataset_root=root,
        windows_per_session=3,
    )
    selection_path = tmp_path / "powered-selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    result = build_powered_replay_packet(
        powered_window_selection_path=selection_path,
        dataset_root=root,
        output_dir=tmp_path / "packet",
    )

    assert result["status"] == "passed"
    assert result["session_count"] == 2
    assert result["window_count"] == 6
    assert result["scientific_request_count"] == 72
    assert all(len(row["controls"]) == 6 for row in result["rows"])
    assert all(Path(row["initial_observation"]["path"]).is_file() for row in result["rows"])
    assert result["label_seal"]["outcome_labels_accessed"] is False


def test_successor_bundle_import_does_not_require_optional_pyarrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def without_pyarrow(name: str, *args: object, **kwargs: object):
        if name == "pyarrow.parquet":
            raise ModuleNotFoundError("optional pyarrow unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pyarrow)
    with pytest.raises(ValueError, match="pyarrow_required_to_build_droid_action_streams"):
        successor_bundle._build_action_streams(tmp_path)
