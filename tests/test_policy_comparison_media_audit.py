"""Byte-level media and actual adapter/wire regressions; no policy server."""
import hashlib

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.episode_visual_evidence import (
    MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION,
    _verified_retained_rgb_frame,
    persist_observation_frame,
    validate_multicamera_frame_manifest,
)
from tests.test_episode_visual_evidence import _multicamera_observation


def _manifest(tmp_path):
    value = {
        "schema_version": MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION,
        "episode_id": "audit",
        "required_camera_ids": ["external", "wrist"],
        "review_only_camera_ids": [],
        "policy_input_observations": [_multicamera_observation(
            tmp_path, episode_id="audit", index=0, kind="policy-input")],
        "terminal_observation": _multicamera_observation(
            tmp_path, episode_id="audit", index=1, kind="terminal-observation"),
        "policy_input_observation_count": 1, "policy_input_frame_count": 2,
        "review_observation_count": 0, "review_frame_count": 0,
    }
    return _reseal(value)


def _reseal(value):
    for observation in [*value["policy_input_observations"], value["terminal_observation"]]:
        for frame in observation["views"].values():
            frame["frame_digest"] = canonical_digest(frame, digest_field="frame_digest")
        observation["observation_digest"] = canonical_digest(observation, digest_field="observation_digest")
    value["frame_manifest_digest"] = canonical_digest(value, digest_field="frame_manifest_digest")
    return value


@pytest.mark.parametrize("mutation", ["dimensions", "timestamp", "kind", "count", "terminal_kind"])
def test_resealed_manifest_cannot_assert_inconsistent_frame_facts(tmp_path, mutation):
    value = _manifest(tmp_path)
    frame = value["policy_input_observations"][0]["views"]["external"]
    if mutation == "dimensions":
        frame["width"] += 1
    elif mutation == "timestamp":
        frame["timestamp_ns"] += 100
    elif mutation == "kind":
        frame["kind"] = "review-sample"
    elif mutation == "count":
        value["policy_input_frame_count"] = 200
    else:
        value["terminal_observation"]["kind"] = "policy-input"
    with pytest.raises(ValueError):
        validate_multicamera_frame_manifest(_reseal(value), output_dir=tmp_path)


def test_jpeg_with_recomputed_hashes_is_not_lossless_policy_input(tmp_path):
    frame = persist_observation_frame(np.full((8, 8, 3), 20, dtype=np.uint8),
        output_dir=tmp_path, episode_id="jpeg", frame_index=0, kind="policy-input")
    path = tmp_path / frame["relative_path"]
    Image.fromarray(np.full((8, 8, 3), 20, dtype=np.uint8)).save(path, format="JPEG")
    frame["png_sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    frame["size_bytes"] = path.stat().st_size
    with Image.open(path) as decoded:
        frame["raw_rgb_sha256"] = "sha256:" + hashlib.sha256(np.asarray(decoded).tobytes()).hexdigest()
    with pytest.raises(ValueError, match="format"):
        _verified_retained_rgb_frame(frame, output_dir=tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "same_length_bytes", "swapped", "digest", "missing_terminal"])
def test_existing_media_checks_reject_artifact_tampering(tmp_path, mutation):
    value = _manifest(tmp_path)
    observation = value["policy_input_observations"][0]
    frame = observation["views"]["external"]
    path = tmp_path / frame["relative_path"]
    if mutation == "missing":
        path.unlink()
    elif mutation == "same_length_bytes":
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    elif mutation == "swapped":
        path.write_bytes((tmp_path / observation["views"]["wrist"]["relative_path"]).read_bytes())
    elif mutation == "digest":
        value["frame_manifest_digest"] = "sha256:" + "0" * 64
    else:
        value.pop("terminal_observation")
    with pytest.raises((ValueError, OSError)):
        validate_multicamera_frame_manifest(value, output_dir=tmp_path)


def test_groot_final_wire_pixels_equal_retained_candidate_inputs(tmp_path):
    from blueprint_pipeline.groot_n17_droid_policy_runtime import GrootN17DroidPolicyClient, GrootN17DroidPolicySpec
    from blueprint_pipeline.groot_n17_wire_client import encode_wire_message, decode_wire_message
    from tests.test_groot_n17_droid_policy_runtime import _FakePolicyClient, _receipt, _observation

    fake = _FakePolicyClient()
    client = GrootN17DroidPolicyClient(spec=GrootN17DroidPolicySpec(),
        worker_identity_receipt=_receipt(), host="unused", client_factory=lambda **_: fake)
    observation = _observation()
    # Distinct channel ramps catch color/order/transposition mistakes.
    rgb = np.arange(180 * 320 * 3, dtype=np.uint32).reshape(180, 320, 3).astype(np.uint8)
    observation["observation/exterior_image_1_left"] = rgb
    observation["observation/wrist_image_left"] = rgb[:, ::-1].copy()
    composite = np.concatenate([observation["observation/" + key] for key in (
        "exterior_image_1_left", "wrist_image_left")], axis=1)
    frame = persist_observation_frame(composite, output_dir=tmp_path,
        episode_id="wire", frame_index=0, kind="policy-input")
    client.infer(observation)
    wire = decode_wire_message(encode_wire_message(fake.requests[0]))
    with Image.open(tmp_path / frame["relative_path"]) as image:
        retained = np.asarray(image)
    assert np.array_equal(wire["video"]["exterior_image_1_left"][0, 0], retained[:, :320])
    assert np.array_equal(wire["video"]["wrist_image_left"][0, 0], retained[:, 320:])
    assert wire["state"]["joint_position"].dtype == np.float32
    client.close()
