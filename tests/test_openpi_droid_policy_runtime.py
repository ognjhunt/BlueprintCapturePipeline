from __future__ import annotations

import json
import base64
import hashlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.openpi_droid_policy_runtime import (
    OpenPIWebsocketDroidPolicyClient,
    load_policy_spec,
    normalize_openpi_inference_response,
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
            return {
                "actions": np.zeros((10, 8)),
                "policy_timing": {"infer_ms": 30.0},
                "server_timing": {"infer_ms": 31.25},
            }

    client = OpenPIWebsocketDroidPolicyClient(
        spec=spec,
        host="127.0.0.1",
        port=8000,
        client_factory=FakeClient,
    )
    response = client.infer({"prompt": "pick"})
    assert response.shape == (10, 8)
    evidence = client.last_inference_evidence()
    assert evidence == {
        "server_response_received": True,
        "wire_response_type": "dict",
        "wire_response_keys": ["actions", "policy_timing", "server_timing"],
        "raw_vendor_action_response": {
            "actions": [[0.0] * 8] * 10,
            "policy_timing": {"infer_ms": 30.0},
            "server_timing": {"infer_ms": 31.25},
        },
        "raw_vendor_action_response_digest": evidence[
            "raw_vendor_action_response_digest"
        ],
        "raw_vendor_action_response_role": (
            "genuine_decoded_vendor_wire_response_before_candidate_normalization"
        ),
        "action_payload_returned": True,
        "actions_extracted": True,
        "action_chunk_shape": [10, 8],
    }
    assert evidence["raw_vendor_action_response_digest"] == (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {"raw_vendor_action_response": evidence["raw_vendor_action_response"]},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert client.evidence_summary()["identity_verified"] is True


def test_websocket_client_records_completed_query_before_response_refusal(
    tmp_path: Path,
) -> None:
    spec = load_policy_spec(
        _cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris"
    )

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def get_server_metadata(self):
            return _runtime_metadata(spec)

        def infer(self, observation):
            del observation
            return {"action": np.zeros((10, 8))}

    client = OpenPIWebsocketDroidPolicyClient(
        spec=spec,
        host="127.0.0.1",
        port=8000,
        client_factory=FakeClient,
    )

    with pytest.raises(
        ValueError, match="openpi_inference_response_unexpected_keys:action"
    ):
        client.infer({"prompt": "pick"})
    assert client.candidate_policy_queried is True
    evidence = client.last_inference_evidence()
    assert evidence == {
        "server_response_received": True,
        "wire_response_type": "dict",
        "wire_response_keys": ["action"],
        "raw_vendor_action_response": {"action": [[0.0] * 8] * 10},
        "raw_vendor_action_response_digest": evidence[
            "raw_vendor_action_response_digest"
        ],
        "raw_vendor_action_response_role": (
            "genuine_decoded_vendor_wire_response_before_candidate_normalization"
        ),
        "action_payload_returned": True,
        "actions_extracted": False,
    }


def test_websocket_client_retains_malformed_nonfinite_ndarray_envelope(
    tmp_path: Path,
) -> None:
    spec = load_policy_spec(
        _cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris"
    )
    malformed = np.zeros((10, 8), dtype=float)
    malformed[0, 0] = np.nan

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def get_server_metadata(self):
            return _runtime_metadata(spec)

        def infer(self, observation):
            del observation
            return malformed

    client = OpenPIWebsocketDroidPolicyClient(
        spec=spec,
        host="127.0.0.1",
        port=8000,
        client_factory=FakeClient,
    )

    with pytest.raises(ValueError, match="openpi_inference_response_not_object"):
        client.infer({"prompt": "pick"})
    evidence = client.last_inference_evidence()
    assert evidence["action_payload_returned"] is True
    assert evidence["actions_extracted"] is False
    assert evidence["raw_vendor_action_response"][0][0] == {
        "nonfinite_float": "nan"
    }
    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (np.zeros((10, 8)), "openpi_inference_response_not_object"),
        ({"server_timing": {}}, "openpi_inference_response_actions_missing"),
        (
            {"actions": [], "action": []},
            "openpi_inference_response_unexpected_keys:action",
        ),
        (
            {"actions": [], "action_chunk": []},
            "openpi_inference_response_unexpected_keys:action_chunk",
        ),
        (
            {"actions": [], "policy_timing": 1.0},
            "openpi_inference_response_policy_timing_not_object",
        ),
        (
            {"actions": [], "server_timing": 1.0},
            "openpi_inference_response_server_timing_not_object",
        ),
    ],
)
def test_openpi_response_normalization_fails_closed(response, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_openpi_inference_response(response)


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


def test_identity_bound_server_uses_verified_local_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A materialized checkpoint must not be re-resolved through public GCS."""

    @dataclass(frozen=True)
    class FakeAssets:
        assets_dir: str
        asset_id: str = "droid"

    @dataclass(frozen=True)
    class FakeData:
        assets: FakeAssets

    @dataclass(frozen=True)
    class FakeModel:
        action_horizon: int = 10

    @dataclass(frozen=True)
    class FakeConfig:
        data: FakeData
        model: FakeModel

    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "assets" / "droid").mkdir(parents=True)
    spec = load_policy_spec(
        _cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris"
    )
    captured: dict[str, object] = {}
    policy_config = types.ModuleType("openpi.policies.policy_config")

    def create_trained_policy(config, checkpoint_path):
        captured["assets_dir"] = config.data.assets.assets_dir
        captured["checkpoint"] = checkpoint_path
        return object()

    policy_config.create_trained_policy = create_trained_policy
    websocket = types.ModuleType("openpi.serving.websocket_policy_server")

    class FakeServer:
        def __init__(self, **kwargs):
            captured["server"] = kwargs

        def serve_forever(self):
            captured["served"] = True

    websocket.WebsocketPolicyServer = FakeServer
    training = types.ModuleType("openpi.training.config")
    training.get_config = lambda name: FakeConfig(
        data=FakeData(
            assets=FakeAssets(
                "gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris/assets"
            )
        ),
        model=FakeModel(),
    )
    modules = {
        "openpi": types.ModuleType("openpi"),
        "openpi.policies": types.ModuleType("openpi.policies"),
        "openpi.policies.policy_config": policy_config,
        "openpi.serving": types.ModuleType("openpi.serving"),
        "openpi.serving.websocket_policy_server": websocket,
        "openpi.training": types.ModuleType("openpi.training"),
        "openpi.training.config": training,
    }
    modules["openpi.policies"].policy_config = policy_config
    modules["openpi.serving"].websocket_policy_server = websocket
    modules["openpi.training"].config = training
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        "blueprint_pipeline.openpi_droid_policy_runtime.verify_local_checkpoint",
        lambda **kwargs: {
            "local_checkpoint_verified": True,
            "local_checkpoint_verification_sha256": "c" * 64,
            "local_checkpoint_object_count": spec.checkpoint_object_count,
            "local_checkpoint_size_bytes": spec.checkpoint_size_bytes,
        },
    )

    serve_identity_bound_policy(
        spec=spec,
        checkpoint_dir=checkpoint,
        checkpoint_inventory_path=tmp_path / "inventory.json",
        host="127.0.0.1",
        port=8000,
    )

    assert captured["assets_dir"] == str((checkpoint / "assets").resolve())
    assert captured["checkpoint"] == checkpoint.resolve()
    assert captured["served"] is True


def _arena_execution_spec(
    tmp_path: Path,
    *,
    candidate_id: str = "pi05_droid",
    policy_id: str = "pi05_droid_jointpos_polaris",
) -> Path:
    """The sealed artifact the arena policy bundle stages as a runtime input."""

    spec = load_policy_spec(
        _cohort(tmp_path), policy_id="pi0_fast_droid_jointpos_polaris"
    )
    policy_spec = {
        "policy_id": policy_id,
        "config_name": policy_id,
        "checkpoint_uri": spec.checkpoint_uri,
        "checkpoint_object_manifest_sha256": spec.checkpoint_object_manifest_sha256,
        "checkpoint_generation_manifest_sha256": (
            spec.checkpoint_generation_manifest_sha256
        ),
        "checkpoint_inventory_sha256": spec.checkpoint_inventory_sha256,
        "checkpoint_object_count": spec.checkpoint_object_count,
        "checkpoint_size_bytes": spec.checkpoint_size_bytes,
        "action_space": spec.action_space,
        "action_chunk_rows": spec.action_chunk_rows,
        "open_loop_horizon": spec.open_loop_horizon,
        "openpi_revision": spec.openpi_revision,
    }
    path = tmp_path / "execution_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "native_task_arena_policy_execution_spec.v1",
                "candidate_id": candidate_id,
                "policy_spec": policy_spec,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_server_identity_comes_from_the_spec_the_client_validates(
    tmp_path: Path,
) -> None:
    """Server and client must read one artifact, not two copies of an identity.

    The arena client validates the server's metadata against the `policy_spec`
    carried by the sealed execution spec. Serving from a separate cohort file
    is how the two come to disagree while each looks correct alone.
    """

    from blueprint_pipeline.openpi_droid_policy_runtime import (
        load_policy_spec_from_execution_spec,
    )

    served = load_policy_spec_from_execution_spec(_arena_execution_spec(tmp_path))

    assert served.policy_id == "pi05_droid_jointpos_polaris"
    # The identity the wrapper publishes satisfies the client's own validator.
    metadata = {
        **served.server_metadata(),
        "local_checkpoint_verified": True,
        "local_checkpoint_verification_sha256": "c" * 64,
        "local_checkpoint_object_count": served.checkpoint_object_count,
        "local_checkpoint_size_bytes": served.checkpoint_size_bytes,
    }
    assert validate_server_metadata(metadata, expected=served) == metadata


def test_execution_spec_candidate_and_policy_id_must_agree(tmp_path: Path) -> None:
    """A spec whose candidate slot disagrees with its policy identity fails closed."""

    from blueprint_pipeline.openpi_droid_policy_runtime import (
        load_policy_spec_from_execution_spec,
    )

    path = _arena_execution_spec(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidate_id"] = "some_other_candidate"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="policy_execution_spec_candidate_mismatch"):
        load_policy_spec_from_execution_spec(path)


def test_execution_spec_refuses_candidate_alias_as_upstream_identity(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.openpi_droid_policy_runtime import (
        load_policy_spec_from_execution_spec,
    )

    path = _arena_execution_spec(tmp_path, policy_id="pi05_droid")

    with pytest.raises(ValueError, match="policy_execution_spec_candidate_mismatch"):
        load_policy_spec_from_execution_spec(path)


def test_identity_bound_server_runs_from_the_flat_provider_runtime() -> None:
    """It is shipped flat and executed as a script, where relative imports fail.

    `from .droid_policy_bridge import ...` raises ImportError -- *not*
    ModuleNotFoundError -- when there is no parent package, so a narrower
    except clause could never catch it.
    """

    import shutil
    import subprocess
    import sys
    import tempfile

    import blueprint_pipeline.openpi_droid_policy_runtime as module

    source_dir = Path(module.__file__).resolve().parent
    with tempfile.TemporaryDirectory() as raw:
        flat = Path(raw)
        for name in ("openpi_droid_policy_runtime.py", "droid_policy_bridge.py"):
            shutil.copy2(source_dir / name, flat / name)
        completed = subprocess.run(
            [sys.executable, str(flat / "openpi_droid_policy_runtime.py"), "--help"],
            capture_output=True,
            text=True,
            cwd=raw,
        )

    assert completed.returncode == 0, completed.stderr
    assert "--policy-spec" in completed.stdout
