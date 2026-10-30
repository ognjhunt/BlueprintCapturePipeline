"""CPU-only transport/contracts; these fixtures are not native policy evidence."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import trlc_pi05_assets as assets
from blueprint_pipeline.trlc_pi05_policy import (
    ARMS,
    JOINTS,
    GRIPPERS,
    IMAGE_SHAPES,
    WIRE_CAMERAS,
    TRLCPi05Binding,
    TRLCPi05PolicyClient,
    encode_observation,
    decode_actions,
    _json,
)
from blueprint_pipeline.trlc_pi05_policy_server import (
    TRLCPi05Server,
    decode_observation,
    load_exact_state_dict,
)


def binding():
    return TRLCPi05Binding(
        ("overhead", "right_wrist", "left_wrist"),
        "sha256:" + "a" * 64,
        "development_only_initial_simulation_camera_assumption",
    )


def observation():
    names = list(reversed(JOINTS))
    return {
        "prompt": "Move the cup.",
        "images": {
            name: np.full(shape, 10 + i * 40, dtype=np.uint8)
            for i, (name, shape) in enumerate(
                zip(binding().camera_names, IMAGE_SHAPES, strict=True)
            )
        },
        "state": {
            "joint_names": names,
            "joint_position": [JOINTS.index(name) / 10 for name in names],
            "arm_names": list(ARMS),
            "arm_joint_names": {
                side: list(JOINTS[i * 6 : (i + 1) * 6]) for i, side in enumerate(ARMS)
            },
            "gripper_names": list(reversed(GRIPPERS)),
            "gripper_position": [0.8, 0.2],
            "arm_gripper_names": dict(zip(ARMS, GRIPPERS, strict=True)),
        },
    }


class Backend:
    def __init__(self):
        self.calls, self.resets = [], 0
        self.action = np.tile(np.arange(28, dtype=np.float32) / 10, (50, 1))
        self.action[:, 6], self.action[:, 13] = 0.2, 0.8

    def predict(self, value):
        self.calls.append(deepcopy(value))
        return self.action.copy()

    def reset(self):
        self.resets += 1


class Connection:
    def __init__(self, runtime, requests, fault=None):
        self.runtime, self.requests, self.fault = runtime, requests, fault

    def request(self, method, path, body, headers):
        self.requests.append((method, path, body, headers))
        self.path, self.body, self.headers = path, body, headers

    def getresponse(self):
        if self.path == "/act" and self.fault:
            value = self.fault
        else:
            value = self.runtime.handle(
                self.path,
                json.loads(self.body),
                session=self.headers.get("X-Blueprint-Session"),
                sequence=self.headers.get("X-Blueprint-Sequence"),
            )
        raw = _json(value)
        return SimpleNamespace(status=200, read=lambda limit: raw[:limit])

    def close(self):
        pass


def test_state_camera_order_and_joint_prefix_mapping_preserve_full_response():
    value = observation()
    wire = encode_observation(value, binding())
    decoded = decode_observation(wire)
    assert set(wire) == {*WIRE_CAMERAS, "state", "instruction"}
    assert decoded["state"] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 0.8]
    )
    for local, remote in zip(binding().camera_names, WIRE_CAMERAS, strict=True):
        np.testing.assert_array_equal(decoded[remote], value["images"][local])
    actions = Backend().action
    native, evidence = decode_actions(
        actions,
        joint_names=value["state"]["joint_names"],
        gripper_names=value["state"]["gripper_names"],
    )
    assert native.shape == (50, 14)
    assert native[0, :12] == pytest.approx(
        [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    )
    assert native[0, -2:] == pytest.approx([0.8, 0.2])
    assert evidence["eef_diagnostics"][0] == pytest.approx(actions[0, 14:])
    assert evidence["clipped_count"] == 0


def test_projection_reports_closure_saturation_without_changing_joint_bounds():
    actions = Backend().action
    actions[0, 0], actions[0, 6], actions[0, 13] = 100, -0.2, 1.2
    native, evidence = decode_actions(
        actions, joint_names=list(JOINTS), gripper_names=list(GRIPPERS)
    )
    assert native[0, 0] == 100 and native[0, -2:].tolist() == [0.0, 1.0]
    assert evidence["clipped_count"] == 2 and evidence["raw_closures"][0] == pytest.approx(
        [-0.2, 1.2]
    )


@pytest.mark.parametrize("fault", ["joint", "camera_size", "camera_dtype", "nonfinite", "closure"])
def test_invalid_native_observation_fails_before_transport(fault):
    value = observation()
    if fault == "joint":
        value["state"]["joint_names"][0] = "unknown"
    elif fault == "camera_size":
        value["images"]["overhead"] = np.zeros((224, 224, 3), dtype=np.uint8)
    elif fault == "camera_dtype":
        value["images"]["overhead"] = value["images"]["overhead"].astype(float)
    elif fault == "nonfinite":
        value["state"]["joint_position"][0] = float("nan")
    else:
        value["state"]["gripper_position"][0] = 1.2
    with pytest.raises(ValueError):
        encode_observation(value, binding())


def test_prestart_session_and_retained_request_response_are_separate_from_inference():
    backend, requests = Backend(), []
    server = TRLCPi05Server(
        backend=backend,
        binding=binding(),
        checkpoint_receipt={"checkpoint_bytes_verified": True, "strict_checkpoint_loaded": True},
    )
    client = TRLCPi05PolicyClient(
        binding=binding(), connection_factory=lambda *_, **__: Connection(server, requests)
    )
    assert client.preflight_readiness()["policy_queried"] is False
    assert backend.calls == [] and not client.candidate_policy_queried
    result = client.infer(observation())
    assert result.shape == (50, 14) and len(backend.calls) == 1
    assert client.candidate_policy_queried and client.last_inference_evidence()["actions_extracted"]
    assert client.last_inference_evidence()["raw_vendor_action_response"]["actions"]["shape"] == [
        50,
        28,
    ]
    from blueprint_pipeline.policy_request_evidence import validate_request_evidence

    validate_request_evidence(client.last_request_evidence())
    client.reset()
    assert not client.candidate_policy_queried and backend.resets == 2
    assert not client.evidence_summary()["native_execution_validated"]


def test_unloaded_server_cannot_pass_preflight():
    server = TRLCPi05Server(
        backend=Backend(),
        binding=binding(),
        checkpoint_receipt={"checkpoint_bytes_verified": True, "strict_checkpoint_loaded": False},
    )
    client = TRLCPi05PolicyClient(
        binding=binding(), connection_factory=lambda *_, **__: Connection(server, [])
    )
    with pytest.raises(ValueError, match="identity_mismatch"):
        client.preflight_readiness()


def test_bad_response_retains_query_evidence_and_never_retries():
    backend, requests = Backend(), []
    server = TRLCPi05Server(
        backend=backend,
        binding=binding(),
        checkpoint_receipt={"checkpoint_bytes_verified": True, "strict_checkpoint_loaded": True},
    )
    client = TRLCPi05PolicyClient(
        binding=binding(),
        connection_factory=lambda *_, **__: Connection(server, requests, {"actions": "bad"}),
    )
    with pytest.raises(ValueError):
        client.infer(observation())
    assert client.candidate_policy_queried and client.last_inference_evidence()[
        "raw_vendor_action_response"
    ] == {"actions": "bad"}
    with pytest.raises(ValueError, match="reset_required"):
        client.infer(observation())
    assert len([row for row in requests if row[1] == "/act"]) == 1


def test_strict_loader_only_adds_the_published_tied_alias_and_propagates_failure():
    head = "model.paligemma_with_expert.paligemma.lm_head.weight"
    embedding = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    calls = []

    class Model:
        def state_dict(self):
            return {head: np.zeros((2, 3)), embedding: np.zeros((2, 3))}

        def load_state_dict(self, value, *, strict):
            calls.append((value, strict))
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    class Tensor:
        shape = (2, 3)

        def clone(self):
            return self

    assert load_exact_state_dict(Model(), {head: Tensor()})["strict_checkpoint_loaded"] is True
    assert calls[0][1] is True and set(calls[0][0]) == {head, embedding}
    with pytest.raises(ValueError, match="key_identity"):
        load_exact_state_dict(Model(), {head: Tensor(), "unknown": Tensor()})

    class Failed(Model):
        def load_state_dict(self, *args, **kwargs):
            raise RuntimeError("strict loader failed")

    with pytest.raises(RuntimeError, match="strict loader failed"):
        load_exact_state_dict(Failed(), {head: Tensor()})


def fake_tokenizer(root, monkeypatch):
    from blueprint_pipeline import trlc_pi05_bundle as bundle

    root.mkdir(parents=True, exist_ok=True)
    original = assets.tokenizer_inventory()
    rows = []
    for row in original["files"]:
        raw = (
            json.dumps({"fixture": row["path"]}).encode()
            if row["path"].endswith(".json")
            else row["path"].encode()
        )
        (root / row["path"]).write_bytes(raw)
        rows.append(
            {"path": row["path"], "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = {**original, "files": rows}
    monkeypatch.setattr(assets, "tokenizer_inventory", lambda: manifest)
    monkeypatch.setattr(bundle, "tokenizer_inventory", lambda: manifest)
    return root


def test_tokenizer_missing_tampered_override_and_staged_sidecars_are_refused(tmp_path, monkeypatch):
    root = fake_tokenizer(tmp_path / "tokenizer", monkeypatch)
    monkeypatch.setattr(assets, "RETAINED_TOKENIZER_DIR", root)
    assert assets.resolve_tokenizer()[1]["tokenizer_bytes_verified"]
    monkeypatch.setenv(assets.TOKENIZER_DIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        assets.resolve_tokenizer()
    monkeypatch.delenv(assets.TOKENIZER_DIR_ENV)
    (root / "unknown.json").write_text("{}")
    assert assets.resolve_tokenizer()[1]["tokenizer_bytes_verified"]
    with pytest.raises(ValueError, match="closure"):
        assets.verify_tokenizer_directory(root)
    (root / "unknown.json").unlink()
    (root / "tokenizer.json").write_text("corrupt")
    with pytest.raises(ValueError):
        assets.resolve_tokenizer()


def test_real_pinned_source_archive_extracts_and_rejects_added_code(tmp_path):
    from blueprint_pipeline.trlc_pi05_source import (
        stage_source,
        extract_source,
        SOURCE_ARCHIVE,
        SOURCE_MANIFEST,
        verify_source_directory,
    )

    stage_source(tmp_path)
    root = extract_source(
        tmp_path / SOURCE_ARCHIVE, tmp_path / SOURCE_MANIFEST, tmp_path / "source"
    )
    assert (root / "lerobot/src/lerobot/policies/pi05/modeling_pi05.py").is_file()
    (root / "lerobot/src/lerobot/rogue.py").write_text("pass")
    with pytest.raises(ValueError, match="members_changed"):
        verify_source_directory(root, tmp_path / SOURCE_MANIFEST)


def test_binding_horizon_does_not_change_training_or_vendor_claims():
    assert binding().identity()["rigid_task_evaluation"] == "out_of_distribution_diagnostic"
    assert binding().identity()["vendor_policy_claimed"] is False
    with pytest.raises(ValueError):
        replace(binding(), execution_horizon=51).validate()
