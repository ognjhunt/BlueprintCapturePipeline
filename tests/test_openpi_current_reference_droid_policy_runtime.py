from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.openpi_current_reference_droid_policy_runtime import (
    OpenPICurrentReferenceDroidPolicyClient,
    OpenPICurrentReferenceDroidPolicySpec,
    verify_local_current_reference_checkpoint,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


def _spec(**overrides) -> OpenPICurrentReferenceDroidPolicySpec:
    values = {
        "policy_id": "pi0_fast_droid",
        "config_name": "pi0_fast_droid",
        "checkpoint_uri": "gs://openpi-assets/checkpoints/pi0_fast_droid",
        "checkpoint_object_inventory_sha256": "a" * 64,
        "checkpoint_manifest_sha256": "b" * 64,
        "checkpoint_inventory_file_sha256": "c" * 64,
        "checkpoint_object_count": 2,
        "checkpoint_size_bytes": 10,
        "action_chunk_rows": 10,
    }
    values.update(overrides)
    return OpenPICurrentReferenceDroidPolicySpec(**values)


def _observation() -> dict:
    return {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.zeros(1),
        "prompt": "pick up the block",
    }


def test_spec_accepts_only_current_official_policy_contracts() -> None:
    _spec().validate()
    _spec(
        policy_id="pi05_droid",
        config_name="pi05_droid",
        checkpoint_uri="gs://openpi-assets/checkpoints/pi05_droid",
        action_chunk_rows=15,
    ).validate()
    with pytest.raises(ValueError, match="action_rows_invalid"):
        _spec(action_chunk_rows=15).validate()
    with pytest.raises(ValueError, match="checkpoint_uri_invalid"):
        _spec(checkpoint_uri="gs://openpi-assets/checkpoints/polaris/fake").validate()


def test_policy_client_preserves_identity_and_native_output_receipts() -> None:
    spec = _spec()

    class Policy:
        def infer(self, observation):
            assert set(observation) == {
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
                "observation/joint_position",
                "observation/gripper_position",
                "prompt",
            }
            return {"actions": np.zeros((10, 8), dtype=np.float32)}

    client = OpenPICurrentReferenceDroidPolicyClient(
        spec=spec,
        policy=Policy(),
        local_verification={
            "local_checkpoint_verified": True,
            "local_checkpoint_verification_sha256": "d" * 64,
            "local_checkpoint_object_count": 2,
            "local_checkpoint_size_bytes": 10,
        },
    )
    response = client.infer(_observation())
    assert response["actions"].shape == (10, 8)
    receipt = response["policy_request_receipt"]
    assert receipt["native_action_shape"] == [10, 8]
    assert len(receipt["request_sha256"]) == 64
    assert len(receipt["native_action_sha256"]) == 64
    assert receipt["physical_outcome_accessed"] is False
    assert client.evidence_summary()["request_count"] == 1


def test_policy_client_rejects_wrong_native_shape() -> None:
    class Policy:
        def infer(self, observation):
            return {"actions": np.zeros((15, 8))}

    client = OpenPICurrentReferenceDroidPolicyClient(
        spec=_spec(),
        policy=Policy(),
        local_verification={
            "local_checkpoint_verified": True,
            "local_checkpoint_object_count": 2,
            "local_checkpoint_size_bytes": 10,
        },
    )
    with pytest.raises(ValueError, match="action_invalid"):
        client.infer(_observation())


def test_local_checkpoint_verification_binds_every_public_gcs_object(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    files = {"params/a": b"weights", "params/b": b"metadata"}
    objects = []
    for relative, contents in files.items():
        path = checkpoint / relative
        path.write_bytes(contents)
        objects.append(
            {
                "name": f"checkpoints/pi0_fast_droid/{relative}",
                "size_bytes": len(contents),
                "generation": "1",
                "metageneration": "1",
                "md5_base64": base64.b64encode(
                    hashlib.md5(contents, usedforsecurity=False).digest()
                ).decode("ascii"),
                "crc32c_base64": "fixture",
                "updated": "2026-07-30T00:00:00+00:00",
            }
        )
    objects.sort(key=lambda row: row["name"])
    inventory = {
        "schema_version": "public_gcs_checkpoint_inventory.v1",
        "observed_at": "2026-07-30T00:00:00+00:00",
        "source_uri": "gs://openpi-assets/checkpoints/pi0_fast_droid",
        "object_count": len(objects),
        "total_bytes": sum(len(contents) for contents in files.values()),
        "latest_updated": "2026-07-30T00:00:00+00:00",
        "objects": objects,
        "raw_secret_values_recorded": False,
        "object_inventory_sha256": canonical_sha256(objects),
    }
    inventory["manifest_sha256"] = canonical_sha256(inventory)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    spec = _spec(
        checkpoint_object_inventory_sha256=inventory["object_inventory_sha256"],
        checkpoint_manifest_sha256=inventory["manifest_sha256"],
        checkpoint_inventory_file_sha256=file_sha256(inventory_path),
        checkpoint_object_count=len(objects),
        checkpoint_size_bytes=inventory["total_bytes"],
    )
    result = verify_local_current_reference_checkpoint(
        spec=spec,
        checkpoint_dir=checkpoint,
        checkpoint_inventory_path=inventory_path,
    )
    assert result["local_checkpoint_verified"] is True
    assert result["local_checkpoint_object_count"] == 2

    (checkpoint / "params/a").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size_mismatch"):
        verify_local_current_reference_checkpoint(
            spec=spec,
            checkpoint_dir=checkpoint,
            checkpoint_inventory_path=inventory_path,
        )


def test_policy_client_requires_complete_local_verification() -> None:
    with pytest.raises(ValueError, match="local_checkpoint_count_mismatch"):
        OpenPICurrentReferenceDroidPolicyClient(
            spec=_spec(),
            policy=object(),
            local_verification={
                "local_checkpoint_verified": True,
                "local_checkpoint_object_count": 1,
                "local_checkpoint_size_bytes": 10,
            },
        )
