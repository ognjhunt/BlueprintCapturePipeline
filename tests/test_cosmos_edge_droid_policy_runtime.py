from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.cosmos_edge_droid_policy_runtime import (
    CosmosEdgeDroidPolicyClient,
    CosmosEdgeDroidPolicySpec,
    validate_server_metadata,
    verify_local_policy_snapshot,
)
from blueprint_pipeline.droid_policy_bridge import DROID_ROBOARENA_CONCAT_VIEWS
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


def _spec(manifest_sha256: str = "a" * 64) -> CosmosEdgeDroidPolicySpec:
    return CosmosEdgeDroidPolicySpec(snapshot_manifest_sha256=manifest_sha256)


def _metadata(spec: CosmosEdgeDroidPolicySpec) -> dict:
    return {
        **spec.server_metadata(),
        "local_snapshot_verified": True,
        "local_snapshot_manifest_sha256": spec.snapshot_manifest_sha256,
        "local_snapshot_file_count": 7,
        "local_snapshot_size_bytes": 100,
        "local_snapshot_verification_sha256": "b" * 64,
    }


def _observation() -> dict:
    return {
        **{view: np.zeros((224, 224, 3), dtype=np.uint8) for view in DROID_ROBOARENA_CONCAT_VIEWS},
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.zeros(1),
        "prompt": "Pick up the bottle.",
    }


def test_client_verifies_identity_three_views_and_action_shape() -> None:
    spec = _spec()

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            assert kwargs == {"host": "127.0.0.1", "port": 8000, "api_key": None}

        def get_server_metadata(self):
            return _metadata(spec)

        def infer(self, observation):
            assert set(DROID_ROBOARENA_CONCAT_VIEWS).issubset(observation)
            assert "blueprint/wam_source_view_paths" not in observation
            native = np.arange(32 * 8, dtype=np.float64).reshape(32, 8)
            native[16:, 0] = 9.0
            return {"action": native}

    client = CosmosEdgeDroidPolicyClient(
        spec=spec,
        host="127.0.0.1",
        port=8000,
        client_factory=FakeClient,
    )
    observation = _observation()
    observation["blueprint/wam_source_view_paths"] = {"local": "only"}
    response = client.infer(observation)

    assert response["action"].shape == (16, 8)
    assert response["native_action"].shape == (32, 8)
    assert response["executed_action"].shape == (8, 8)
    assert np.array_equal(response["commanded_next_joint_position"], response["action"][7, :7])
    assert np.array_equal(response["commanded_next_gripper_position"], response["action"][7, 7:])
    assert response["policy_request_receipt"]["native_action_shape"] == [32, 8]
    assert response["policy_request_receipt"]["wam_prefix_action_shape"] == [16, 8]
    assert response["policy_request_receipt"]["executed_prefix_steps"] == 8
    assert len(response["policy_request_receipt"]["receipt_sha256"]) == 64
    assert client.evidence_summary()["request_count"] == 1


def test_client_fails_before_transport_on_missing_third_view() -> None:
    spec = _spec()

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_server_metadata(self):
            return _metadata(spec)

        def infer(self, _observation):
            raise AssertionError("transport must not be called")

    client = CosmosEdgeDroidPolicyClient(
        spec=spec, host="127.0.0.1", port=8000, client_factory=FakeClient
    )
    observation = _observation()
    del observation[DROID_ROBOARENA_CONCAT_VIEWS[-1]]
    with pytest.raises(ValueError, match="cosmos_edge_policy_observation_invalid"):
        client.infer(observation)


def test_server_metadata_rejects_empty_stock_metadata() -> None:
    with pytest.raises(ValueError, match="cosmos_edge_policy_server_identity_mismatch"):
        validate_server_metadata({}, expected=_spec())


def test_local_snapshot_verification_hashes_every_frozen_file(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = snapshot / "config.json"
    weights = snapshot / "weights.safetensors"
    config.write_bytes(b"config")
    weights.write_bytes(b"weights")
    from blueprint_pipeline.policy_ranking_thesis import file_sha256

    manifest = {
        "schema_version": "cosmos_edge_droid_policy_snapshot_manifest.v1",
        "model_id": "nvidia/Cosmos3-Edge-Policy-DROID",
        "model_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777",
        "required_files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in (config, weights)
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    receipt = verify_local_policy_snapshot(
        spec=_spec(manifest["manifest_sha256"]),
        snapshot_dir=snapshot,
        snapshot_manifest_path=manifest_path,
    )

    assert receipt["local_snapshot_verified"] is True
    assert receipt["local_snapshot_file_count"] == 2
    assert receipt["local_snapshot_size_bytes"] == len(b"configweights")


def test_committed_protocol_and_snapshot_manifest_digests_are_frozen() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "experiments"
        / "policy_ranking_cosmos3_edge_closed_loop_20260729"
    )
    checks = (
        ("policy_snapshot_manifest_v1.json", "manifest_sha256"),
        ("source_freeze_v1.json", "manifest_sha256"),
        ("source_freeze_v2.json", "manifest_sha256"),
        ("source_freeze_amendment_v2.json", "amendment_sha256"),
        ("source_freeze_v3.json", "manifest_sha256"),
        ("source_freeze_amendment_v3.json", "amendment_sha256"),
        ("source_freeze_v4.json", "manifest_sha256"),
        ("source_freeze_amendment_v4.json", "amendment_sha256"),
        ("source_freeze_v5.json", "manifest_sha256"),
        ("source_freeze_amendment_v5.json", "amendment_sha256"),
        ("source_freeze_v6.json", "manifest_sha256"),
        ("source_freeze_amendment_v6.json", "amendment_sha256"),
        ("protocol_amendment_v2.json", "amendment_sha256"),
        ("protocol_amendment_v3.json", "amendment_sha256"),
        ("protocol_amendment_v4.json", "amendment_sha256"),
        ("policy_canary_bundle_overwrite_incident_v1.json", "record_sha256"),
        ("policy_canary_bundle_concurrent_supersession_v1.json", "record_sha256"),
        ("allocation_4_pre_provider_gate_v1.json", "record_sha256"),
        ("policy_canary_bundle_freeze_v2.json", "freeze_sha256"),
        ("policy_canary_bundle_freeze_v4.json", "freeze_sha256"),
        ("policy_canary_bundle_freeze_v5.json", "freeze_sha256"),
        ("confirmation_cohort_v1.json", "cohort_sha256"),
        ("offline_oscar_projection_validation_v1.json", "record_sha256"),
        ("diagnostic_session_unseal_v1.json", "record_sha256"),
        ("calibration_availability_v1.json", "record_sha256"),
        ("protocol_v1.json", "protocol_sha256"),
    )
    payloads = {}
    for filename, digest_field in checks:
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        recorded = payload.pop(digest_field)
        assert recorded == canonical_sha256(payload)
        payloads[filename] = {**payload, digest_field: recorded}

    CosmosEdgeDroidPolicySpec(
        snapshot_manifest_sha256=payloads["policy_snapshot_manifest_v1.json"]["manifest_sha256"]
    ).validate()
    assert payloads["protocol_v1.json"]["paid_execution_admitted"] is False
    assert payloads["protocol_v1.json"]["provider_called"] is False
    amendment = payloads["source_freeze_amendment_v2.json"]
    assert (
        amendment["former_manifest_sha256"] == payloads["source_freeze_v1.json"]["manifest_sha256"]
    )
    assert (
        amendment["successor_manifest_sha256"]
        == payloads["source_freeze_v2.json"]["manifest_sha256"]
    )
    assert amendment["paid_execution_admitted"] is False
    assert amendment["provider_called"] is False
    amendment_v3 = payloads["source_freeze_amendment_v3.json"]
    assert (
        amendment_v3["former_manifest_sha256"]
        == payloads["source_freeze_v2.json"]["manifest_sha256"]
    )
    assert (
        amendment_v3["successor_manifest_sha256"]
        == payloads["source_freeze_v3.json"]["manifest_sha256"]
    )
    assert payloads["confirmation_cohort_v1.json"]["physical_outcome_labels_accessed"] is False
    assert payloads["source_freeze_v4.json"]["native_policy_action_shape"] == [32, 8]
    assert payloads["protocol_amendment_v2.json"]["wam_prefix_action_shape"] == [16, 8]
    assert (
        payloads["protocol_amendment_v3.json"]["prospective_runtime_change"]["successor_value"]
        is False
    )
    assert payloads["policy_canary_bundle_overwrite_incident_v1.json"]["provider_called"] is False
    assert payloads["policy_canary_bundle_freeze_v2.json"]["bundle_version"] == 4
    assert payloads["policy_canary_bundle_freeze_v4.json"]["bundle_version"] == 8
    assert payloads["policy_canary_bundle_freeze_v5.json"]["bundle_version"] == 9
    assert (
        payloads["source_freeze_v6.json"]["openpi_dependency_contract"]["openpi_client_version"]
        == "0.1.2"
    )
    assert (
        payloads["protocol_amendment_v4.json"]["prospective_runtime_change"][
            "successor_client_runtime"
        ]
        == "pinned Cosmos framework virtual environment"
    )
    assert (
        payloads["policy_canary_bundle_concurrent_supersession_v1.json"][
            "selected_successor_bundle_version"
        ]
        == 8
    )
    assert (
        payloads["diagnostic_session_unseal_v1.json"][
            "outcome_labels_accessed_before_predictions_were_frozen"
        ]
        is True
    )
    assert (
        payloads["calibration_availability_v1.json"]["public_session_files"]["camera_intrinsics"]
        is False
    )

    former_authorization = json.loads(
        (root / "compute_authorization_allocation_4.json").read_text(encoding="utf-8")
    )
    successor_authorization = json.loads(
        (root / "compute_authorization_allocation_4_v2.json").read_text(encoding="utf-8")
    )
    assert successor_authorization[
        "supersedes_unconsumed_authorization_sha256"
    ] == canonical_sha256(former_authorization)
    assert successor_authorization["bundle_version"] == 8
    assert successor_authorization["source_commit"] == "74132bb03423746fa6b8a88b7c552f64fac9bc45"
    assert successor_authorization["paid_mutation_authorized"] is False

    allocation_5 = json.loads(
        (root / "compute_authorization_allocation_5.json").read_text(encoding="utf-8")
    )
    assert allocation_5["allocation_index"] == 5
    assert allocation_5["bundle_version"] == 9
    assert allocation_5["source_commit"] == "79911f43c1033e9981dc946ae7d254dc94a37a6f"
    assert allocation_5["paid_mutation_authorized"] is True
