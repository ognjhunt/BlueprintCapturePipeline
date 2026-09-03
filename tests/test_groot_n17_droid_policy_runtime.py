from __future__ import annotations

import socket
import threading
from dataclasses import dataclass

import numpy as np
import pytest
import zmq

from blueprint_pipeline.adp009d_groot_worker_identity import (
    expected_checkpoint_content_binding,
)
from blueprint_pipeline.groot_n17_droid_policy_runtime import (
    CHECKPOINT_STATISTICS_GIT_BLOB_SHA1,
    CHECKPOINT_STATISTICS_SHA256,
    CHECKPOINT_STATISTICS_SOURCE,
    CHECKPOINT_REVISION,
    DROID_EEF_POSITION_OBSERVED_MAX_M,
    DROID_EEF_POSITION_OBSERVED_MIN_M,
    EMBODIMENT_TAG,
    EEF_FRAME_BODY_NAME,
    EEF_FRAME_BODY_SOURCE,
    EEF_FRAME_PROVENANCE_KEY,
    EEF_FRAME_PROVENANCE_SCHEMA_VERSION,
    EEF_FRAME_STATE_SOURCE,
    FROZEN_VIDEO_DELTA_INDICES,
    GROOT_SOURCE_REVISION,
    LANGUAGE_KEY,
    MODEL_ID,
    GrootN17DroidPolicyClient,
    GrootN17DroidPolicySpec,
    droid_eef_9d,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.groot_n17_wire_client import (
    decode_wire_message,
    encode_wire_message,
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
        "checkpoint_content_manifest_digest": expected_checkpoint_content_binding()[
            "file_manifest_digest"
        ],
        "environment_lock_sha256": "2" * 64,
    }


class _FakePolicyClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.requests = []
        self.reset_calls = 0
        self.close_calls = 0

    def ping(self) -> bool:
        return True

    def get_modality_config(self) -> dict:
        return {
            "video": _Modality(("exterior_image_1_left", "wrist_image_left"), (0,)),
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

    def close(self):
        self.close_calls += 1


def _eef_frame_provenance(position_m: list[float]) -> dict:
    value = {
        "schema_version": EEF_FRAME_PROVENANCE_SCHEMA_VERSION,
        "state_frame": "robot_root",
        "body_name": EEF_FRAME_BODY_NAME,
        "body_source": EEF_FRAME_BODY_SOURCE,
        "state_source": EEF_FRAME_STATE_SOURCE,
        "position_robot_root_m": position_m,
        "body_pose_world_xyzw": [*position_m, 0.0, 0.0, 0.0, 1.0],
        "robot_root_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "provenance_digest": "",
    }
    value["provenance_digest"] = canonical_digest(
        value, digest_field="provenance_digest"
    )
    return value


def _observation(*, position_m: list[float] | None = None) -> dict:
    position = position_m or [0.5, 0.0, 0.3]
    return {
        "observation/exterior_image_1_left": np.full((180, 320, 3), 5, dtype=np.uint8),
        "observation/wrist_image_left": np.full((180, 320, 3), 7, dtype=np.uint8),
        "observation/joint_position": np.arange(7, dtype=float),
        "observation/gripper_position": np.asarray([0.25]),
        "observation/eef_9d": droid_eef_9d(
            position_m=position,
            rotation_row_major=np.eye(3).reshape(-1),
        ),
        EEF_FRAME_PROVENANCE_KEY: _eef_frame_provenance(position),
        "prompt": "Pick up the spray can and place it inside the marked tray.",
    }


def test_spec_is_pinned_to_official_droid_checkpoint_and_source() -> None:
    identity = GrootN17DroidPolicySpec().identity()
    assert identity["model_id"] == "nvidia/GR00T-N1.7-DROID"
    assert identity["embodiment_tag"] == "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    assert len(identity["identity_sha256"]) == 64

    processor_config = next(
        row
        for row in expected_checkpoint_content_binding()["file_manifest"]
        if row["path"] == "processor_config.json"
    )
    assert processor_config == {
        "path": "processor_config.json",
        "size_bytes": 2_833,
        "digest_algorithm": "git_blob_sha1",
        "digest": "55b4d74b3565274662ba33eefe9bdb0ca75df3e9",
    }
    # This exact root file is loaded by AutoProcessor and declares [0].  The
    # checkpoint's legacy experiment config says [-15, 0] but is not served.
    assert FROZEN_VIDEO_DELTA_INDICES == (0,)
    statistics = next(
        row
        for row in expected_checkpoint_content_binding()["file_manifest"]
        if row["path"] == "statistics.json"
    )
    assert statistics == {
        "path": "statistics.json",
        "size_bytes": 144_097,
        "digest_algorithm": "git_blob_sha1",
        "digest": CHECKPOINT_STATISTICS_GIT_BLOB_SHA1,
    }


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
    assert request["video"]["exterior_image_1_left"].shape == (1, 1, 180, 320, 3)
    assert np.array_equal(
        request["video"]["exterior_image_1_left"][0, 0],
        _observation()["observation/exterior_image_1_left"],
    )
    assert np.array_equal(
        request["video"]["wrist_image_left"][0, 0],
        _observation()["observation/wrist_image_left"],
    )
    assert request["state"]["eef_9d"].shape == (1, 1, 9)
    assert request["language"] == {LANGUAGE_KEY: [[_observation()["prompt"]]]}
    assert client.evidence_summary()["identity_verified"] is True
    native = client.last_inference_evidence()
    assert native["native_action_chunk_shape"] == [40, 17]
    assert len(native["native_action_components"]["eef_9d"]) == 40
    assert len(native["native_action_chunk_sha256"]) == 64
    assert native["execution_projection"].endswith("eef_9d_retained_not_executed")
    evidence = client.evidence_summary()
    assert evidence["video_delta_indices"] == [0]
    assert evidence["video_history_source"] == (
        "current_policy_query_observation_only"
    )
    assert evidence["state_delta_indices"] == [0]
    assert evidence["action_delta_indices"] == list(range(40))
    assert evidence["language_delta_indices"] == [0]
    assert evidence["eef_position_observed_support"] == {
        "minimum_m": list(DROID_EEF_POSITION_OBSERVED_MIN_M),
        "maximum_m": list(DROID_EEF_POSITION_OBSERVED_MAX_M),
        "source": CHECKPOINT_STATISTICS_SOURCE,
        "source_sha256": CHECKPOINT_STATISTICS_SHA256,
        "source_git_blob_sha1": CHECKPOINT_STATISTICS_GIT_BLOB_SHA1,
        "enforced_before_policy_query": False,
        "frame_provenance_enforced_before_policy_query": True,
        "frozen_processor_use_percentiles": True,
        "frozen_processor_clip_outliers": True,
    }

    client.reset()
    assert fake.reset_calls == 1
    client.infer(_observation())
    reset_request = fake.requests[-1]
    assert reset_request["video"]["exterior_image_1_left"].shape == (
        1,
        1,
        180,
        320,
        3,
    )
    client.close()
    assert fake.close_calls == 1


def test_groot_preflight_reconfirms_transport_without_inference() -> None:
    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        client_factory=lambda **kwargs: fake,
    )

    readiness = client.preflight_readiness()

    assert readiness["identity_verified"] is True
    assert readiness["candidate_policy_queried"] is False
    assert readiness["candidate_inference_performed"] is False
    assert readiness["policy_state_advanced"] is False
    assert fake.reset_calls == 1
    assert fake.requests == []


def test_groot_preflight_resets_for_next_episode_after_prior_inference() -> None:
    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        client_factory=lambda **kwargs: fake,
    )
    client.infer(_observation())

    readiness = client.preflight_readiness()

    assert readiness["identity_verified"] is True
    assert readiness["candidate_policy_queried"] is False
    assert readiness["candidate_inference_performed"] is False
    assert readiness["prior_candidate_policy_query_observed"] is True
    assert readiness["last_inference_evidence"] is None
    assert client.candidate_policy_queried is True
    assert fake.reset_calls == 1
    assert len(fake.requests) == 1


def test_client_queries_production_reset_outside_empirical_extrema_with_typed_evidence() -> None:
    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        client_factory=lambda **kwargs: fake,
    )
    # Independent FK of the exact Scene 839873 task-aware reset.  NVIDIA's
    # frozen processor clips this z value; statistics extrema are not an API
    # support declaration and must not suppress a diagnostic policy query.
    position = [0.16286441683769226, 0.0867096483707428, 1.0434999465942383]
    result = client.infer(_observation(position_m=position))

    assert result.shape == (40, 8)
    assert len(fake.requests) == 1
    support = client.last_inference_evidence()["eef_position_observed_support"]
    assert support["inside_checkpoint_observed_extrema"] is False
    assert support["above_maximum_by_m"][:2] == [0.0, 0.0]
    assert support["above_maximum_by_m"][2] == pytest.approx(
        position[2] - DROID_EEF_POSITION_OBSERVED_MAX_M[2], abs=1.0e-6
    )
    assert support["frozen_processor_use_percentiles"] is True
    assert support["frozen_processor_clip_outliers"] is True
    assert support["query_blocking"] is False


def test_strict_droid_profile_refuses_out_of_support_reset_before_query() -> None:
    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        require_observed_eef_support=True,
        client_factory=lambda **kwargs: fake,
    )
    position = [0.16286441683769226, 0.0867096483707428, 1.0434999465942383]

    with pytest.raises(
        ValueError, match="groot_droid_eef_outside_checkpoint_observed_support"
    ):
        client.infer(_observation(position_m=position))

    assert fake.requests == []
    assert client.candidate_policy_queried is False
    support = client.last_inference_evidence()["eef_position_observed_support"]
    assert support["inside_checkpoint_observed_extrema"] is False
    assert support["query_blocking"] is True


@pytest.mark.parametrize("mutation", ("missing", "world_position", "digest"))
def test_client_refuses_unproven_eef_frame_before_query(mutation: str) -> None:
    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        client_factory=lambda **kwargs: fake,
    )
    observation = _observation()
    if mutation == "missing":
        observation.pop(EEF_FRAME_PROVENANCE_KEY)
    elif mutation == "world_position":
        observation[EEF_FRAME_PROVENANCE_KEY] = _eef_frame_provenance(
            [3.8094613552093506, 9.223036766052246, 0.5535212159156799]
        )
    else:
        observation[EEF_FRAME_PROVENANCE_KEY]["provenance_digest"] = "sha256:" + "0" * 64

    with pytest.raises(ValueError, match="groot_droid_eef_frame_provenance_invalid"):
        client.infer(observation)

    assert fake.requests == []


def test_client_refuses_nonfinite_state_before_query() -> None:
    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        client_factory=lambda **kwargs: fake,
    )
    observation = _observation()
    observation["observation/joint_position"][0] = np.nan

    with pytest.raises(ValueError, match="groot_droid_observation_state_nonfinite"):
        client.infer(observation)

    assert fake.requests == []


@pytest.mark.parametrize(
    ("modality_name", "replacement", "expected_error"),
    [
        (
            "video",
            _Modality(("exterior_image_1_left", "wrist_image_left"), (-15, 0)),
            "groot_droid_video_delta_indices_mismatch",
        ),
        (
            "state",
            _Modality(("eef_9d", "gripper_position", "joint_position"), (-1, 0)),
            "groot_droid_state_delta_indices_mismatch",
        ),
        (
            "action",
            _Modality(("eef_9d", "gripper_position", "joint_position"), tuple(range(8))),
            "groot_droid_action_delta_indices_mismatch",
        ),
        (
            "language",
            _Modality((LANGUAGE_KEY,), (-1, 0)),
            "groot_droid_language_delta_indices_mismatch",
        ),
    ],
)
def test_client_refuses_any_drift_from_frozen_modality_delta_indices(
    modality_name: str,
    replacement: _Modality,
    expected_error: str,
) -> None:
    class _DriftedModalityClient(_FakePolicyClient):
        def get_modality_config(self) -> dict:
            modality = super().get_modality_config()
            modality[modality_name] = replacement
            return modality

    fake = _DriftedModalityClient()
    with pytest.raises(ValueError, match=expected_error):
        GrootN17DroidPolicyClient(
            spec=GrootN17DroidPolicySpec(),
            worker_identity_receipt=_receipt(),
            host="127.0.0.1",
            client_factory=lambda **kwargs: fake,
        )

    assert fake.close_calls == 1


def test_failed_client_readiness_closes_wire_resources() -> None:
    fake = _FakePolicyClient()
    fake.ping = lambda: False

    with pytest.raises(ValueError, match="groot_policy_server_unreachable"):
        GrootN17DroidPolicyClient(
            spec=GrootN17DroidPolicySpec(),
            worker_identity_receipt=_receipt(),
            host="127.0.0.1",
            client_factory=lambda **kwargs: fake,
        )

    assert fake.close_calls == 1


def test_default_factory_uses_wire_only_client_for_real_loopback_round_trip() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    ready = threading.Event()

    def server() -> None:
        context = zmq.Context()
        reply = context.socket(zmq.REP)
        reply.bind(f"tcp://127.0.0.1:{port}")
        ready.set()
        try:
            for _ in range(3):
                request = decode_wire_message(reply.recv())
                if request["endpoint"] == "ping":
                    response = {"status": "ok"}
                elif request["endpoint"] == "get_modality_config":
                    response = {
                        "video": {
                            "__ModalityConfig__": True,
                            "as_json": {
                                "modality_keys": [
                                    "exterior_image_1_left",
                                    "wrist_image_left",
                                ],
                                "delta_indices": [0],
                            },
                        },
                        "state": {
                            "__ModalityConfig__": True,
                            "as_json": {
                                "modality_keys": [
                                    "eef_9d",
                                    "gripper_position",
                                    "joint_position",
                                ],
                                "delta_indices": [0],
                            },
                        },
                        "action": {
                            "__ModalityConfig__": True,
                            "as_json": {
                                "modality_keys": [
                                    "eef_9d",
                                    "gripper_position",
                                    "joint_position",
                                ],
                                "delta_indices": list(range(40)),
                            },
                        },
                        "language": {
                            "__ModalityConfig__": True,
                            "as_json": {
                                "modality_keys": [LANGUAGE_KEY],
                                "delta_indices": [0],
                            },
                        },
                    }
                else:
                    assert request["endpoint"] == "get_action"
                    response = [
                        {
                            "joint_position": np.zeros((1, 40, 7), dtype=np.float32),
                            "gripper_position": np.zeros((1, 40, 1), dtype=np.float32),
                            "eef_9d": np.zeros((1, 40, 9), dtype=np.float32),
                        },
                        {},
                    ]
                reply.send(encode_wire_message(response))
        finally:
            reply.close(linger=0)
            context.term()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(timeout=2.0)
    client = GrootN17DroidPolicyClient(
        spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(),
        host="127.0.0.1",
        port=port,
    )
    assert client.infer(_observation()).shape == (40, 8)
    client.close()
    thread.join(timeout=2.0)
    assert not thread.is_alive()


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
