from __future__ import annotations

import json
import base64
import hashlib
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.openpi_droid_policy_runtime import (
    OpenPIWebsocketDroidPolicyClient,
    load_policy_spec,
    serve_identity_bound_policy,
    validate_server_metadata,
    verify_local_checkpoint,
)


def _cohort(tmp_path: Path) -> Path:
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "policy_ranking_warehouse_policy_cohort.v2",
                "openpi_revision": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
                "checkpoint_inventory": {
                    "inventory_sha256": "a" * 64,
                },
                "action_contract": {
                    "space": "absolute_joint_position_plus_gripper_position",
                    "executed_open_loop_horizon": 8,
                },
                "primary_cohort": [
                    {
                        "policy_id": "pi0_fast_droid_jointpos_polaris",
                        "checkpoint": "gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris",
                        "checkpoint_object_count": 36,
                        "checkpoint_size_bytes": 10843569155,
                        "public_object_manifest_sha256": "4f6bc8271938d85a72c89cd76b6cc2e80a153c41ba6bb124302ce318e8b74154",
                        "generation_manifest_sha256": "b" * 64,
                        "action_horizon": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_policy_spec_binds_frozen_identity(tmp_path: Path) -> None:
    spec = load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    assert spec.action_space == "joint_position"
    assert spec.action_chunk_rows == 10
    assert spec.open_loop_horizon == 8
    assert len(spec.server_metadata()["identity_sha256"]) == 64


def test_server_metadata_mismatch_fails_closed(tmp_path: Path) -> None:
    spec = load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    metadata = spec.server_metadata()
    metadata["policy_id"] = "wrong-policy"
    with pytest.raises(ValueError, match="policy_server_identity_mismatch:policy_id"):
        validate_server_metadata(metadata, expected=spec)


def _runtime_metadata(spec):
    return {
        **spec.server_metadata(),
        "local_checkpoint_verified": True,
        "local_checkpoint_verification_sha256": "c" * 64,
        "local_checkpoint_object_count": spec.checkpoint_object_count,
        "local_checkpoint_size_bytes": spec.checkpoint_size_bytes,
    }


def test_server_metadata_requires_local_verification_and_rejects_extras(tmp_path: Path) -> None:
    spec = load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")
    with pytest.raises(ValueError, match="policy_server_local_checkpoint_not_verified"):
        validate_server_metadata(spec.server_metadata(), expected=spec)
    metadata = _runtime_metadata(spec)
    metadata["unfrozen"] = True
    with pytest.raises(ValueError, match="unexpected:unfrozen"):
        validate_server_metadata(metadata, expected=spec)


def test_websocket_client_verifies_before_inference(tmp_path: Path) -> None:
    spec = load_policy_spec(_cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris")

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {"host": "127.0.0.1", "port": 8000, "api_key": None}

        def get_server_metadata(self):
            return _runtime_metadata(spec)

        def infer(self, observation):
            assert observation["prompt"] == "pick"
            return {"actions": np.zeros((10, 8))}

    client = OpenPIWebsocketDroidPolicyClient(
        spec=spec,
        host="127.0.0.1",
        port=8000,
        client_factory=FakeClient,
    )
    response = client.infer({"prompt": "pick"})
    assert response["actions"].shape == (10, 8)
    assert client.evidence_summary()["identity_verified"] is True


def test_verify_local_checkpoint_binds_every_object(tmp_path: Path) -> None:
    cohort_path = _cohort(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    files = {
        "params/model.bin": b"weights",
        "assets/stats.json": b"{}",
    }
    objects = []
    prefix = "checkpoints/polaris/pi0_fast_droid_jointpos_polaris/"
    for relative, contents in files.items():
        local = checkpoint / relative
        local.write_bytes(contents)
        objects.append(
            {
                "name": prefix + relative,
                "size": str(len(contents)),
                "md5Hash": base64.b64encode(
                    hashlib.md5(contents, usedforsecurity=False).digest()
                ).decode("ascii"),
                "crc32c": "unused",
                "generation": "1",
                "metageneration": "1",
                "updated": "2026-07-26T00:00:00Z",
            }
        )
    from blueprint_pipeline.openpi_checkpoint_inventory import (
        generation_manifest_sha256,
        legacy_object_manifest_sha256,
    )
    from blueprint_pipeline.policy_ranking_thesis import canonical_sha256

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    row = cohort["primary_cohort"][0]
    row["checkpoint_object_count"] = len(objects)
    row["checkpoint_size_bytes"] = sum(len(value) for value in files.values())
    row["public_object_manifest_sha256"] = legacy_object_manifest_sha256(objects)
    row["generation_manifest_sha256"] = generation_manifest_sha256(objects)
    inventory = {
        "schema_version": "openpi_checkpoint_inventory.v1",
        "status": "frozen",
        "queried_at_utc": "2026-07-26T00:00:00+00:00",
        "source": "fixture",
        "cohort_path": str(cohort_path),
        "openpi_revision": cohort["openpi_revision"],
        "entries": [
            {
                "policy_id": row["policy_id"],
                "checkpoint_uri": row["checkpoint"],
                "object_count": len(objects),
                "size_bytes": row["checkpoint_size_bytes"],
                "legacy_object_manifest_sha256": row["public_object_manifest_sha256"],
                "generation_manifest_sha256": row["generation_manifest_sha256"],
                "objects": objects,
            }
        ],
        "blockers": [],
        "claim_boundary": {},
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    cohort["checkpoint_inventory"]["inventory_sha256"] = inventory["inventory_sha256"]
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    spec = load_policy_spec(cohort_path, policy_id=row["policy_id"])
    verification = verify_local_checkpoint(
        spec=spec,
        checkpoint_dir=checkpoint,
        checkpoint_inventory_path=inventory_path,
    )
    assert verification["local_checkpoint_verified"] is True
    assert verification["local_checkpoint_object_count"] == 2

    (checkpoint / "params/model.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint_object_size_mismatch"):
        verify_local_checkpoint(
            spec=spec,
            checkpoint_dir=checkpoint,
            checkpoint_inventory_path=inventory_path,
        )


def test_unknown_policy_and_bad_checkpoint_identity_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="policy_id_not_unique_in_cohort"):
        load_policy_spec(_cohort(tmp_path), policy_id="unknown")
    payload = json.loads(_cohort(tmp_path).read_text(encoding="utf-8"))
    payload["primary_cohort"][0]["public_object_manifest_sha256"] = "bad"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_checkpoint_object_manifest_sha256"):
        load_policy_spec(path, policy_id="pi0_fast_droid_jointpos_polaris")


def test_policy_server_rejects_non_loopback_bind_before_checkpoint_io(
    tmp_path: Path,
) -> None:
    spec = load_policy_spec(
        _cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris"
    )
    with pytest.raises(ValueError, match="openpi_policy_server_must_be_loopback_only"):
        serve_identity_bound_policy(
            spec=spec,
            checkpoint_dir=tmp_path / "missing",
            checkpoint_inventory_path=tmp_path / "missing-inventory.json",
            host="0.0.0.0",
            port=8000,
        )
