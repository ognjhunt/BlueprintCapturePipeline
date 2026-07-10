from __future__ import annotations

import base64
import json
from hashlib import sha256
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import openvla_policy_command_adapter as adapter


def _approved_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    remote_code_required: bool = False,
) -> tuple[Path, Path]:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    config_path.write_text('{"model_type":"openvla"}\n', encoding="utf-8")
    weights_path.write_bytes(b"sealed-model-weights")
    unsigned_manifest = {
        "schema_version": "blueprint_openvla_checkpoint_approval.v1",
        "model_revision": "a" * 40,
        "license": {
            "spdx_id": "Apache-2.0",
            "review_status": "approved",
            "approved_by": "model-governance",
        },
        "remote_code_required": remote_code_required,
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for path in (config_path, weights_path)
        ],
    }
    canonical = json.dumps(
        unsigned_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    private_key = Ed25519PrivateKey.generate()
    signature = private_key.sign(adapter.OPENVLA_APPROVAL_SIGNATURE_DOMAIN + canonical)
    manifest = {
        **unsigned_manifest,
        "approval_signature": {
            "algorithm": "Ed25519",
            "key_id": "test-model-governance",
            "signature": base64.b64encode(signature).decode("ascii"),
        },
    }
    (checkpoint / adapter.OPENVLA_CHECKPOINT_APPROVAL_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    public_key_path = tmp_path / "model-governance-public.pem"
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    monkeypatch.setenv(adapter.OPENVLA_APPROVED_DIGEST_ENV, sha256(canonical).hexdigest())
    monkeypatch.setenv(adapter.OPENVLA_APPROVAL_PUBLIC_KEY_ENV, str(public_key_path))
    return checkpoint, weights_path


class _FakeScalar:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value


class _FakeTensor:
    def __init__(
        self,
        rows: list[list[int]],
        *,
        dtype: str = "long",
        device: str = "cuda:0",
    ) -> None:
        self.rows = rows
        self.dtype = dtype
        self.device = device
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __getitem__(self, key):
        row, col = key
        return _FakeScalar(self.rows[row][col])


class _FakeTorch:
    @staticmethod
    def full(shape, value, *, dtype, device):
        rows, cols = shape
        return _FakeTensor([[int(value) for _ in range(cols)] for _ in range(rows)], dtype=dtype, device=device)

    @staticmethod
    def ones(shape, *, dtype, device):
        rows, cols = shape
        return _FakeTensor([[1 for _ in range(cols)] for _ in range(rows)], dtype=dtype, device=device)

    @staticmethod
    def cat(tensors, dim):
        assert dim == 1
        left, right = tensors
        return _FakeTensor(
            [left_row + right_row for left_row, right_row in zip(left.rows, right.rows)],
            dtype=left.dtype,
            device=left.device,
        )


def test_openvla_policy_adapter_blocks_without_checkpoint_or_frame(tmp_path: Path) -> None:
    response, exit_code = adapter.run_openvla_policy(
        payload={
            "observation": {
                "task_id": "contact_or_push_light_object",
                "task_prompt": "Push the light object forward slightly.",
            }
        },
        checkpoint=tmp_path / "missing-openvla-checkpoint",
        source_root=None,
        device="cpu",
        unnorm_key="bridge_orig",
        allow_cpu=True,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["openvla_policy_action_command_ran"] is False
    assert response["model_ran"] is False
    assert "blocked_openvla_policy_checkpoint_missing" in response["blockers"]
    assert "blocked_missing_policy_visual_observation_frame" in response["blockers"]
    assert response["claim_boundary"]["openvla_model_executed"] is False
    assert response["claim_boundary"]["unitree_g1_dexterous_manipulation_proven"] is False


def test_openvla_checkpoint_requires_signed_digest_and_license_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, weights_path = _approved_checkpoint(tmp_path, monkeypatch)

    approval, blockers = adapter._validate_checkpoint_approval(checkpoint)

    assert blockers == []
    assert approval["status"] == "approved"
    assert approval["model_revision"] == "a" * 40
    assert approval["remote_code_allowed"] is False
    assert approval["network_model_download_allowed"] is False

    weights_path.write_bytes(b"tampered-model-weights")
    tampered, tamper_blockers = adapter._validate_checkpoint_approval(checkpoint)
    assert tampered["status"] == "blocked"
    assert any(
        blocker.startswith("blocked_openvla_checkpoint_digest_mismatch:model.safetensors")
        for blocker in tamper_blockers
    )


def test_openvla_checkpoint_rejects_remote_code_and_external_source_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _weights_path = _approved_checkpoint(
        tmp_path,
        monkeypatch,
        remote_code_required=True,
    )
    approval, blockers = adapter._validate_checkpoint_approval(checkpoint)
    assert approval["status"] == "blocked"
    assert "blocked_openvla_checkpoint_remote_code_required" in blockers

    frame_path = tmp_path / "frame.png"
    frame_path.write_bytes(b"not-decoded-because-preflight-blocks")
    response, exit_code = adapter.run_openvla_policy(
        payload={"observation": {"camera_frame_path": str(frame_path)}},
        checkpoint=checkpoint,
        source_root=tmp_path / "operator-code",
        device="cpu",
        unnorm_key="bridge_orig",
        allow_cpu=True,
    )
    assert exit_code == 2
    assert "blocked_openvla_external_source_root_forbidden" in response["blockers"]


def test_openvla_policy_adapter_decodes_to_blueprint_actions() -> None:
    contact_action = adapter.decode_openvla_action(
        raw_action=[0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 1.0],
        observation={
            "task_id": "contact_or_push_light_object",
            "object_state": {
                "position": [0.36, -0.65, 0.27],
            },
        },
    )
    assert contact_action["action_type"] == "manipulation_contact"
    assert contact_action["target_object_id"] == "blueprint_light_object"
    assert contact_action["waypoint"][0] > 0.5

    inspect_action = adapter.decode_openvla_action(
        raw_action=[],
        observation={"task_id": "inspect_target"},
    )
    assert inspect_action == {"action_type": "inspect_look", "yaw_rate_rad_s": 0.25}

    waypoint_action = adapter.decode_openvla_action(
        raw_action=[0.0],
        observation={
            "task_id": "approach_target",
            "route_task_state": {"target_pose": [0.52, 0.0, 0.79]},
        },
    )
    assert waypoint_action["action_type"] == "waypoint"
    assert waypoint_action["waypoint"] == [0.52, 0.0, 0.79]


def test_openvla_policy_adapter_replays_provider_output_without_fresh_claim(
    tmp_path: Path,
) -> None:
    provider_output = tmp_path / "openvla_policy_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "schema_version": adapter.SCHEMA_VERSION,
                "status": "completed",
                "policy_id": "openvla_policy",
                "model_repo_id": "openvla/openvla-7b",
                "openvla_model_executed": True,
                "openvla_policy_action_command_ran": True,
                "openvla_predict_action_invoked": True,
                "raw_openvla_action_vector": [0.01, -0.02, 0.0],
                "action": {
                    "action_type": "waypoint",
                    "waypoint": [0.36, -0.65, 0.79],
                    "max_speed_mps": 0.08,
                },
            }
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_openvla_policy(
        payload={"observation": {"task_id": "approach_target"}},
        checkpoint=None,
        source_root=None,
        device="cpu",
        unnorm_key="bridge_orig",
        allow_cpu=False,
        provider_output=provider_output,
    )

    assert exit_code == 0
    assert response["status"] == "completed"
    assert response["policy_id"] == "openvla_policy_provider_replay"
    assert response["provider_output_replay_used"] is True
    assert response["fresh_openvla_model_executed_this_invocation"] is False
    assert response["provider_openvla_model_executed"] is True
    assert response["action"]["action_type"] == "waypoint"
    assert response["claim_boundary"]["openvla_model_executed"] is True
    assert (
        response["claim_boundary"][
            "provider_output_replay_is_not_fresh_per_request_model_inference"
        ]
        is True
    )


def test_openvla_policy_adapter_replay_rejects_untrusted_schema(
    tmp_path: Path,
) -> None:
    provider_output = tmp_path / "openvla_policy_provider_output.json"
    provider_output.write_text(
        json.dumps(
            {
                "schema_version": "openvla_policy_provider_output.v1",
                "status": "completed",
                "openvla_model_executed": True,
                "openvla_policy_action_command_ran": True,
                "openvla_predict_action_invoked": True,
                "action": {"action_type": "waypoint", "waypoint": [0.1, 0.0, 0.2]},
            }
        ),
        encoding="utf-8",
    )

    response, exit_code = adapter.run_openvla_policy(
        payload={"observation": {"task_id": "approach_target"}},
        checkpoint=None,
        source_root=None,
        device="cpu",
        unnorm_key="bridge_orig",
        allow_cpu=False,
        provider_output=provider_output,
    )

    assert exit_code == 2
    assert response["status"] == "blocked"
    assert response["provider_output_replay_used"] is True
    assert "blocked_openvla_provider_output_schema_not_trusted" in response["blockers"]
    assert response["openvla_policy_action_command_ran"] is False
    assert response["model_ran"] is False


def test_prepare_openvla_predict_action_inputs_aligns_attention_mask() -> None:
    inputs = {
        "input_ids": _FakeTensor([[1, 2, 3]]),
        "attention_mask": _FakeTensor([[1, 1, 1]]),
        "pixel_values": _FakeTensor([[9, 9]]),
    }

    prepared, diagnostics = adapter._prepare_openvla_predict_action_inputs(inputs, _FakeTorch)

    assert prepared["input_ids"].rows == [[1, 2, 3, adapter.OPENVLA_EMPTY_TOKEN_ID]]
    assert prepared["attention_mask"].rows == [[1, 1, 1, 1]]
    assert diagnostics["input_ids_shape_before"] == [1, 3]
    assert diagnostics["attention_mask_shape_before"] == [1, 3]
    assert diagnostics["input_ids_shape_after"] == [1, 4]
    assert diagnostics["attention_mask_shape_after"] == [1, 4]
    assert diagnostics["openvla_empty_token_appended_before_predict_action"] is True


def test_prepare_openvla_predict_action_inputs_noops_when_token_present() -> None:
    inputs = {
        "input_ids": _FakeTensor([[1, adapter.OPENVLA_EMPTY_TOKEN_ID]]),
        "attention_mask": _FakeTensor([[1, 1]]),
    }

    prepared, diagnostics = adapter._prepare_openvla_predict_action_inputs(inputs, _FakeTorch)

    assert prepared["input_ids"].rows == [[1, adapter.OPENVLA_EMPTY_TOKEN_ID]]
    assert prepared["attention_mask"].rows == [[1, 1]]
    assert diagnostics["input_ids_shape_after"] == [1, 2]
    assert diagnostics["attention_mask_shape_after"] == [1, 2]
    assert diagnostics["openvla_empty_token_appended_before_predict_action"] is False


def test_openvla_policy_adapter_main_writes_blocked_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {
                "observation": {
                    "task_id": "approach_target",
                    "task_prompt": "Approach the target.",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_INPUT", str(input_path))
    monkeypatch.setenv("BLUEPRINT_POLICY_ACTION_OUTPUT", str(output_path))
    exit_code = adapter.main(["--checkpoint", str(tmp_path / "missing")])

    assert exit_code == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"
    assert payload["policy_id"] == adapter.POLICY_ID
    captured = json.loads(capsys.readouterr().out)
    assert captured["raw_credentials_written_to_artifacts"] is False
