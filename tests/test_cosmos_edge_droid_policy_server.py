from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline.cosmos_edge_droid_policy_runtime import CosmosEdgeDroidPolicySpec
from blueprint_pipeline.cosmos_edge_droid_policy_server import (
    NativeActionShapeGuard,
    _disable_policy_guardrails,
    serve_identity_bound_policy,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


def _snapshot(tmp_path: Path) -> tuple[Path, Path]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = snapshot / "model.safetensors"
    config = snapshot / "config.json"
    model.write_bytes(b"weights")
    config.write_bytes(b"config")
    manifest = {
        "schema_version": "cosmos_edge_droid_policy_snapshot_manifest.v1",
        "model_id": "nvidia/Cosmos3-Edge-Policy-DROID",
        "model_revision": "3ea407af3e156c0af3b4bb6edd85842cc9a58777",
        "required_files": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in (config, model)
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot, manifest_path


def test_native_shape_guard_rejects_non_native_horizon() -> None:
    class BadService:
        def infer(self, _observation: dict[str, Any]) -> dict[str, Any]:
            return {"action": np.zeros((16, 8))}

    with pytest.raises(ValueError, match="native_action_invalid"):
        NativeActionShapeGuard(BadService()).infer({})


def test_policy_setup_disables_optional_generated_media_guardrails() -> None:
    class Setup:
        guardrails = True

        def model_copy(self, *, update: dict[str, Any]) -> Any:
            copied = Setup()
            copied.guardrails = update["guardrails"]
            return copied

    updated = _disable_policy_guardrails(Setup())

    assert updated.guardrails is False


def test_server_binds_verified_identity_and_native_shape(tmp_path: Path) -> None:
    snapshot, manifest_path = _snapshot(tmp_path)
    captured = {}

    class Service:
        def infer(self, _observation: dict[str, Any]) -> dict[str, Any]:
            return {"action": np.zeros((32, 8))}

    class Server:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def serve_forever(self) -> None:
            captured["response"] = captured["policy"].infer({})

    startup = serve_identity_bound_policy(
        checkpoint_path=snapshot,
        snapshot_manifest_path=manifest_path,
        host="127.0.0.1",
        port=8000,
        output_dir=tmp_path / "out",
        service_factory=Service,
        server_factory=Server,
    )

    expected = CosmosEdgeDroidPolicySpec(
        snapshot_manifest_sha256=json.loads(manifest_path.read_text())["manifest_sha256"]
    )
    assert captured["metadata"]["identity_sha256"] == expected.server_metadata()["identity_sha256"]
    assert captured["metadata"]["local_snapshot_verified"] is True
    assert captured["response"]["action"].shape == (32, 8)
    assert startup["native_action_shape"] == [32, 8]
    assert startup["nvidia_guardrails_enabled"] is False
    assert startup["blueprint_action_and_abstention_gates_remain_enabled"] is True
    written = json.loads((tmp_path / "out" / "policy_server_startup.json").read_text())
    assert written["raw_credentials_written"] is False
