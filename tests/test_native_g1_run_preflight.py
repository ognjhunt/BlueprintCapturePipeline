from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_run_preflight as preflight
from blueprint_pipeline import native_g1_official_sonic_target_bridge as sonic


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path, monkeypatch, *, role: str = "manipulation") -> dict:
    pytest.importorskip("pxr")
    scene = tmp_path / "native_task_arena_scene_plan.v1.json"
    plan = {"robot": {"robot_id": "unitree_g1"}, "plan_digest": "sha256:" + "a" * 64}
    scene.write_text(json.dumps(plan))
    monkeypatch.setattr(preflight, "validate_native_task_arena_runtime_plan", lambda value, *, bundle_root: value)
    content = b"model-weights"
    files = [{"path": "model.safetensors", "sha256": _sha(content), "size_bytes": len(content)}]
    inventory = {"schema_version": preflight.INVENTORY_SCHEMA, "candidates": [{
        "candidate_id": "candidate", "subdirectory": "model", "files": files,
        "inventory_digest": "sha256:" + _sha(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()),
        "policy_role": role, "action_interface": "humanoidarena_semantic_v3",
        "input_image_shape_hwc": [480, 640, 3], "input_state_width": 64,
        "output_action_width": 40,
    }]}
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    checkpoint = tmp_path / "checkpoints/model/model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(content)
    server = tmp_path / "server.py"
    server.write_bytes(b"server-source")
    monkeypatch.setattr(preflight, "PINNED_POLICY_SERVER_SHA256", _sha(server.read_bytes()))
    provider = tmp_path / "sonic.py"
    provider.write_bytes(b"sonic-source")
    monkeypatch.setattr(sonic, "PINNED_ACTION_PROVIDER_SHA256", _sha(provider.read_bytes()))
    monkeypatch.setattr(preflight, "PINNED_ACTION_PROVIDER_SHA256", _sha(provider.read_bytes()))
    encoder = tmp_path / "encoder.onnx"
    encoder.write_bytes(b"encoder")
    decoder = tmp_path / "decoder.onnx"
    decoder.write_bytes(b"decoder")
    return {
        "scene_plan_path": scene, "bundle_root": tmp_path,
        "inventory_path": inventory_path, "candidate_id": "candidate",
        "checkpoint_root": tmp_path / "checkpoints", "policy_server_source": server,
        "sonic_provider_source": provider, "sonic_encoder": encoder,
        "sonic_encoder_sha256": "sha256:" + _sha(encoder.read_bytes()),
        "sonic_decoder": decoder,
        "sonic_decoder_sha256": "sha256:" + _sha(decoder.read_bytes()),
    }


@pytest.mark.parametrize("role", ["manipulation", "movement_navigation"])
def test_preflight_binds_scene_candidate_and_exact_runtime_inputs(tmp_path, monkeypatch, role):
    args = _fixture(tmp_path, monkeypatch, role=role)
    receipt = preflight.preflight_g1_shared_scene_run(**args)
    assert receipt["status"] == "staged_inputs_verified"
    assert receipt["scene_plan_digest"] == "sha256:" + "a" * 64
    assert receipt["policy_role"] == role
    assert receipt["checkpoint_files"][0]["sha256"] == "sha256:" + _sha(b"model-weights")
    assert receipt["server_process_verified"] is False
    assert receipt["episode_executed"] is False
    assert receipt["task_scored"] is False


@pytest.mark.parametrize("name,expected", [
    ("checkpoint", "checkpoint_identity_mismatch"),
    ("server", "policy_server_source_identity_mismatch"),
    ("provider", "g1_sonic_source_revision_mismatch"),
    ("encoder", "sonic_encoder_identity_mismatch"),
])
def test_preflight_rejects_changed_inputs(tmp_path, monkeypatch, name, expected):
    args = _fixture(tmp_path, monkeypatch)
    path = {
        "checkpoint": tmp_path / "checkpoints/model/model.safetensors",
        "server": args["policy_server_source"],
        "provider": args["sonic_provider_source"],
        "encoder": args["sonic_encoder"],
    }[name]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match=expected):
        preflight.preflight_g1_shared_scene_run(**args)


def test_preflight_rejects_symlinked_checkpoint_and_inventory_drift(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    checkpoint = tmp_path / "checkpoints/model/model.safetensors"
    checkpoint.unlink()
    target = tmp_path / "target"
    target.write_bytes(b"model-weights")
    checkpoint.symlink_to(target)
    with pytest.raises(ValueError, match="checkpoint_symlink_forbidden"):
        preflight.preflight_g1_shared_scene_run(**args)
    checkpoint.unlink()
    checkpoint.write_bytes(b"model-weights")
    inventory = json.loads(args["inventory_path"].read_text())
    inventory["candidates"][0]["files"][0]["sha256"] = "0" * 64
    args["inventory_path"].write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="candidate_inventory_digest_invalid"):
        preflight.preflight_g1_shared_scene_run(**args)
