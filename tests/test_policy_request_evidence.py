from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.groot_n17_wire_client import encode_wire_message, decode_wire_message
from blueprint_pipeline.policy_request_evidence import (
    ObservedWebsocket, capture_request, persist_request_evidence, restore_request,
    snapshot_request, validate_request_evidence,
)


def request():
    return {"prompt": " exact prompt ", "image": np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
            "state": np.array([.1, .2, .3], dtype=np.float32)}


def test_exact_values_dtype_order_prompt_and_wire_replay(tmp_path):
    value = request()
    raw = encode_wire_message(value)
    receipt = capture_request(value, transport="groot_zmq_msgpack_numpy", scientific_wire_bytes=raw,
        decoded_wire_request=decode_wire_message(raw))
    assert validate_request_evidence(receipt)["serialization_verified"] is True
    replay = restore_request(receipt["request"])
    assert replay["prompt"] == " exact prompt "
    assert replay["state"].dtype == np.float32
    assert np.array_equal(value["image"], replay["image"])
    artifact = persist_request_evidence(receipt, root=tmp_path, episode_id="pair-a", query_index=0)
    assert artifact["serialization_verified"] is True
    with pytest.raises(FileExistsError):
        persist_request_evidence(receipt, root=tmp_path, episode_id="pair-a", query_index=0)


@pytest.mark.parametrize("mutation", ["channel", "dtype", "prompt", "shape"])
def test_serialization_cannot_change_the_request(mutation):
    value = request()
    changed = deepcopy(value)
    if mutation == "channel":
        changed["image"] = changed["image"][..., ::-1]
    elif mutation == "dtype":
        changed["state"] = changed["state"].astype(np.float64)
    elif mutation == "prompt":
        changed["prompt"] = changed["prompt"].strip()
    else:
        changed["image"] = changed["image"].reshape(2, 8, 3)
    with pytest.raises(ValueError, match="wire_values_mismatch"):
        capture_request(value, transport="test", scientific_wire_bytes=encode_wire_message(changed), decoded_wire_request=changed)


def test_resealed_wire_substitution_is_independently_detected():
    value = request()
    receipt = capture_request(value, transport="test", scientific_wire_bytes=encode_wire_message(value), decoded_wire_request=value)
    changed = request()
    changed["prompt"] = "different"
    substituted = capture_request(changed, transport="test", scientific_wire_bytes=encode_wire_message(changed), decoded_wire_request=changed)
    receipt["scientific_wire_bytes_base64"] = substituted["scientific_wire_bytes_base64"]
    receipt["scientific_wire_sha256"] = substituted["scientific_wire_sha256"]
    receipt["evidence_digest"] = canonical_digest(receipt, digest_field="evidence_digest")
    with pytest.raises(ValueError, match="wire_values_mismatch"):
        validate_request_evidence(receipt)


def test_openpi_witness_blocks_bad_wire_before_send_and_retains_good_request():
    sent, retained = [], []
    socket = SimpleNamespace(send=sent.append)
    proxy = ObservedWebsocket(socket, request=request(), decoder=decode_wire_message, sink=retained.append)
    raw = encode_wire_message(request())
    proxy.send(raw)
    assert sent == [raw]
    assert validate_request_evidence(retained[0])["serialization_verified"] is True
    bad = request()
    bad["prompt"] = "wrong"
    with pytest.raises(ValueError, match="wire_values_mismatch"):
        proxy.send(encode_wire_message(bad))
    assert len(sent) == 1


def test_credentials_and_nonfinite_inputs_never_enter_evidence():
    for value in ({"api_token": "secret"}, {"a": np.array([float("nan")])}, {"a": np.array([object()])}):
        with pytest.raises(ValueError):
            snapshot_request(value)


def test_injected_transport_does_not_claim_serialized_wire():
    assert capture_request(request(), transport="fake")["serialization_verified"] is False


def test_native_sensor_counter_refuses_stale_camera_after_motion():
    from tests.test_adp009d_isaac_episode_adapter import _adapter, _Env
    from blueprint_pipeline.adp009d_isaac_episode_adapter import IsaacEpisodeAdapterError
    env = _Env()
    for camera in env.unwrapped.scene.values():
        camera.frame = np.array([1])
    adapter = _adapter(env)
    adapter.reset()
    assert all(row["status"] == "observed" for row in adapter.read_policy_inputs()["sensor_freshness"].values())
    adapter._control_step_index += 1
    with pytest.raises(IsaacEpisodeAdapterError, match="sensor_frame_stale"):
        adapter.read_policy_inputs()
    for camera in env.unwrapped.scene.values():
        camera.frame += 1
    adapter.read_policy_inputs()


def test_actual_groot_wire_records_science_but_never_transport_credentials():
    import json
    from blueprint_pipeline.groot_n17_wire_client import GrootN17WirePolicyClient
    client = GrootN17WirePolicyClient.__new__(GrootN17WirePolicyClient)
    client._closed = False
    client.api_token = "private-test-transport-token"
    sent, retained = [], []
    client._socket = SimpleNamespace(send=sent.append,
        recv=lambda: encode_wire_message([{"actions": np.zeros((1, 8))}, {}]))
    client.bind_request_evidence_sink(retained.append)
    client.get_action(request())
    assert decode_wire_message(sent[0])["api_token"] == client.api_token
    assert client.api_token not in json.dumps(retained)
    assert validate_request_evidence(retained[0])["serialization_verified"] is True


def test_actual_openpi_client_observes_vendor_serialization_before_inference_result(tmp_path):
    from blueprint_pipeline.openpi_droid_policy_runtime import OpenPIWebsocketDroidPolicyClient, load_policy_spec
    from tests.test_openpi_droid_policy_runtime import _cohort, _runtime_metadata
    spec = load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    sent, retained = [], []
    class Vendor:
        def __init__(self, **kwargs):
            self._ws = SimpleNamespace(send=sent.append, close=lambda: None)
        def get_server_metadata(self):
            return _runtime_metadata(spec)
        def infer(self, observation):
            self._ws.send(encode_wire_message(observation))
            return {"actions": np.zeros((10, 8))}
    client = OpenPIWebsocketDroidPolicyClient(spec=spec, host="127.0.0.1", port=8000,
        client_factory=Vendor, wire_decoder=decode_wire_message)
    client.bind_request_evidence_sink(retained.append)
    client.infer(request())
    assert len(sent) == len(retained) == 1
    assert validate_request_evidence(retained[0])["serialization_verified"] is True
    assert client.candidate_policy_queried is True
