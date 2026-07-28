from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import policy_ranking_cosmos_causal_screen as screen
from blueprint_pipeline.policy_ranking_cosmos_causal_screen import (
    _correlation,
    action_intensity,
    camera_compensated_motion,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "docs/experiments/policy_ranking_cosmos3_followup_20260728"


def _frozen_inputs() -> tuple[dict[str, object], dict[str, object]]:
    smoke = json.loads((EXPERIMENT / "smoke_request_inventory.json").read_text())
    with zipfile.ZipFile(EXPERIMENT / "cosmos3_followup_provider_bundle.zip") as archive:
        actions = json.loads(
            archive.read("provider_runtime/cosmos3_input/action_streams.json").decode()
        )
    return smoke, actions


def _runtime_matrix(tmp_path: Path, smoke: dict[str, object]) -> Path:
    root = tmp_path / "runtime"
    (root / "responses").mkdir(parents=True)
    (root / "videos").mkdir()
    for index, row in enumerate(smoke["requests"]):
        request_id = row["request_id"]
        video_path = root / "videos" / f"{request_id}.mp4"
        video_path.write_bytes(f"video-{index}".encode())
        response = {
            "experiment_id": smoke["experiment_id"],
            "request_id": request_id,
            "condition": row["condition"],
            "seed": row["seed"],
            "action_sha256": row["action_sha256"],
            "initial_observation_sha256": smoke["initial_observation_sha256"],
            "task_instruction": smoke["task_instruction"],
            "accepted_first_valid": True,
            "generated_media_valid": True,
            "response": {"output_sha256": file_sha256(video_path)},
        }
        (root / "responses" / f"{request_id}.json").write_text(json.dumps(response))
    return root


def test_action_intensity_rejects_wrong_shape_and_preserves_zero_control() -> None:
    assert np.array_equal(action_intensity([[0.0] * 10] * 16), np.zeros(16))
    with pytest.raises(ValueError, match="action_shape_invalid"):
        action_intensity([[0.0] * 10] * 15)


def test_action_intensity_sees_translation_rotation_and_gripper_changes() -> None:
    actions = np.zeros((16, 10), dtype=np.float64)
    actions[:, 3] = 1.0
    actions[:, 7] = 1.0
    actions[2, 0] = 0.25
    actions[7, 4] = 0.2
    actions[11, 9] = 1.0

    signal = action_intensity(actions.tolist())

    assert signal[2] > signal[1]
    assert signal[7] > signal[6]
    assert signal[11] > signal[10]


def test_correlation_fails_closed_for_constant_signals() -> None:
    assert _correlation(np.ones(16), np.arange(16)) == 0.0


def test_camera_compensated_motion_rejects_global_translation() -> None:
    base = np.zeros((136, 160), dtype=np.uint8)
    base[35:100, 45:120] = 180
    frames = []
    for offset in range(17):
        matrix = np.float32([[1, 0, offset], [0, 1, 0]])
        frames.append(__import__("cv2").warpAffine(base, matrix, (160, 136), borderMode=1))

    residual = camera_compensated_motion(np.asarray(frames))

    assert residual.shape == (16,)
    assert float(np.mean(residual)) < 0.25


def test_causal_screen_accepts_only_frozen_response_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, actions = _frozen_inputs()
    runtime_root = _runtime_matrix(tmp_path, smoke)
    monkeypatch.setattr(
        screen,
        "_decode_scene",
        lambda _path: (
            np.zeros((17, 136, 160, 3), dtype=np.float32),
            np.zeros((17, 136, 160), dtype=np.uint8),
        ),
    )

    report = screen.build_causal_screen(
        runtime_output_root=runtime_root,
        action_streams=actions,
        smoke_inventory=smoke,
    )

    assert report["status"] == "completed"
    assert report["blockers"] == []
    assert report["independent_session_count"] == 1
    assert report["confirmatory_power_sufficient"] is False


def test_causal_screen_rejects_mixed_experiment_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, actions = _frozen_inputs()
    runtime_root = _runtime_matrix(tmp_path, smoke)
    first = next((runtime_root / "responses").glob("*.json"))
    response = json.loads(first.read_text())
    response["experiment_id"] = "stale_experiment"
    first.write_text(json.dumps(response))
    monkeypatch.setattr(
        screen,
        "_decode_scene",
        lambda _path: (
            np.zeros((17, 136, 160, 3), dtype=np.float32),
            np.zeros((17, 136, 160), dtype=np.uint8),
        ),
    )

    report = screen.build_causal_screen(
        runtime_output_root=runtime_root,
        action_streams=actions,
        smoke_inventory=smoke,
    )

    assert report["status"] == "blocked"
    assert any("response_frozen_binding_mismatch" in item for item in report["blockers"])
    assert report["evaluator_eligible"] is False


def test_causal_screen_rejects_duplicate_response_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    smoke, actions = _frozen_inputs()
    runtime_root = _runtime_matrix(tmp_path, smoke)
    source = sorted((runtime_root / "responses").glob("*.json"))[0]
    (runtime_root / "responses" / "zz-duplicate.json").write_text(source.read_text())
    monkeypatch.setattr(
        screen,
        "_decode_scene",
        lambda _path: (
            np.zeros((17, 136, 160, 3), dtype=np.float32),
            np.zeros((17, 136, 160), dtype=np.uint8),
        ),
    )

    report = screen.build_causal_screen(
        runtime_output_root=runtime_root,
        action_streams=actions,
        smoke_inventory=smoke,
    )

    assert report["status"] == "blocked"
    assert any("duplicate_request_id" in item for item in report["blockers"])
