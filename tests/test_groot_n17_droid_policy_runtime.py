from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from blueprint_pipeline.groot_n17_droid_policy_runtime import (
    CHECKPOINT_REVISION,
    EMBODIMENT_TAG,
    GROOT_SOURCE_REVISION,
    LANGUAGE_KEY,
    MODEL_ID,
    GrootN17DroidPolicyClient,
    GrootN17DroidPolicySpec,
    droid_eef_9d,
)


@dataclass
class _Modality:
    modality_keys: tuple[str, ...]
    delta_indices: tuple[int, ...]


def _receipt() -> dict:
    return {
        "status": "verified",
        "model_id": MODEL_ID,
        "embodiment_tag": EMBODIMENT_TAG,
        "groot_source_revision": GROOT_SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_files_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
    }


class _FakePolicyClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests = []
        self.reset_calls = 0

    def ping(self) -> bool:
        return True

    def get_modality_config(self) -> dict:
        return {
            "video": _Modality(("exterior_image_1_left", "wrist_image_left"), (-15, 0)),
            "state": _Modality(("eef_9d", "gripper_position", "joint_position"), (0,)),
            "action": _Modality(("eef_9d", "gripper_position", "joint_position"), tuple(range(40))),
            "language": _Modality((LANGUAGE_KEY,), (0,)),
        }

    def get_action(self, request):
        self.requests.append(request)
        joints = np.repeat(np.arange(7, dtype=float)[None, None, :], 40, axis=1)
        gripper = np.linspace(0.0, 1.0, 40, dtype=float)[None, :, None]
        eef = np.zeros((1, 40, 9), dtype=float)
        return {"joint_position": joints, "gripper_position": gripper, "eef_9d": eef}, {}

    def reset(self):
        self.reset_calls += 1
        return {}


def _observation() -> dict:
    return {
        "observation/exterior_image_1_left": np.full((224, 224, 3), 5, dtype=np.uint8),
        "observation/wrist_image_left": np.full((224, 224, 3), 7, dtype=np.uint8),
        "observation/joint_position": np.arange(7, dtype=float),
        "observation/gripper_position": np.asarray([0.25]),
        "observation/eef_9d": np.arange(9, dtype=float),
        "observation_history/exterior_image_1_left_t_minus_15": np.full(
            (224, 224, 3), 3, dtype=np.uint8
        ),
        "observation_history/wrist_image_left_t_minus_15": np.full(
            (224, 224, 3), 4, dtype=np.uint8
        ),
        "prompt": "Pick up the spray can and place it inside the marked tray.",
    }


def test_spec_is_pinned_to_official_droid_checkpoint_and_source() -> None:
    identity = GrootN17DroidPolicySpec().identity()
    assert identity["model_id"] == "nvidia/GR00T-N1.7-DROID"
    assert identity["embodiment_tag"] == "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    assert len(identity["identity_sha256"]) == 64


def test_client_translates_existing_observation_and_returns_joint_chunk() -> None:
    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        client_factory=lambda **kwargs: fake,
    )
    first = client.infer(_observation())
    second = client.infer(_observation())

    assert first.shape == (40, 8)
    assert second.shape == (40, 8)
    assert np.array_equal(first[:, :7], np.repeat(np.arange(7)[None, :], 40, axis=0))
    assert set(first[:, 7]) == {0.0, 1.0}
    request = fake.requests[0]
    assert request["video"]["exterior_image_1_left"].shape == (1, 2, 180, 320, 3)
    assert request["state"]["eef_9d"].shape == (1, 1, 9)
    assert request["language"] == {LANGUAGE_KEY: [[_observation()["prompt"]]]}
    assert client.evidence_summary()["identity_verified"] is True
    native = client.last_inference_evidence()
    assert native["native_action_chunk_shape"] == [40, 17]
    assert len(native["native_action_components"]["eef_9d"]) == 40
    assert len(native["native_action_chunk_sha256"]) == 64
    assert native["execution_projection"].endswith("eef_9d_retained_not_executed")

    client.reset()
    assert fake.reset_calls == 1
    client.infer(_observation())
    reset_request = fake.requests[-1]
    assert reset_request["video"]["exterior_image_1_left"].shape == (
        1,
        2,
        180,
        320,
        3,
    )
    assert not np.array_equal(
        reset_request["video"]["exterior_image_1_left"][:, 0],
        reset_request["video"]["exterior_image_1_left"][:, 1],
    )


def test_client_rejects_unverified_or_mismatched_worker_identity() -> None:
    receipt = _receipt()
    receipt["checkpoint_revision"] = "0" * 40
    with pytest.raises(ValueError, match="worker_receipt_checkpoint_revision_mismatch"):
        GrootN17DroidPolicyClient(
            spec=GrootN17DroidPolicySpec(),
            worker_identity_receipt=receipt,
            host="127.0.0.1",
            client_factory=lambda **kwargs: _FakePolicyClient(),
        )


def test_droid_eef_9d_applies_nvidia_frame_correction() -> None:
    result = droid_eef_9d(position_m=[1.0, 2.0, 3.0], rotation_row_major=np.eye(3).reshape(-1))
    assert result.dtype == np.float32
    assert np.array_equal(result[:3], [1.0, 2.0, 3.0])
    assert np.array_equal(result[3:], [0.0, 0.0, -1.0, -1.0, 0.0, 0.0])
