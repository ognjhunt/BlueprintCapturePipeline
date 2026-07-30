from __future__ import annotations

from pathlib import Path

import numpy as np

from blueprint_pipeline import openpi_current_reference_policy_canary as canary_module
from blueprint_pipeline.openpi_current_reference_policy_canary import (
    run_current_reference_policy_canary,
)


def test_policy_canary_runs_exactly_one_query_each_and_preserves_native_output(
    tmp_path: Path, monkeypatch
) -> None:
    source_freeze = tmp_path / "source_freeze.json"
    source_freeze.write_text("{}", encoding="utf-8")
    initial = tmp_path / "initial.json"
    initial.write_text("{}", encoding="utf-8")
    inventory_dir = tmp_path / "inventories"
    inventory_dir.mkdir()

    class Spec:
        def __init__(self, policy_id: str, rows: int) -> None:
            self.policy_id = policy_id
            self.checkpoint_uri = f"gs://fixture/{policy_id}"
            self.checkpoint_object_count = 1
            self.checkpoint_size_bytes = 1
            self.action_chunk_rows = rows
            self.executed_prefix_steps = 8

        def server_metadata(self):
            return {"policy_id": self.policy_id, "identity_sha256": "a" * 64}

    rows = {"pi0_droid": 10, "pi0_fast_droid": 10, "pi05_droid": 15}
    specs = {policy_id: Spec(policy_id, count) for policy_id, count in rows.items()}
    monkeypatch.setattr(canary_module, "load_current_reference_policy_specs", lambda **_: specs)
    monkeypatch.setattr(
        canary_module,
        "verify_local_current_reference_checkpoint",
        lambda **_: {
            "local_checkpoint_verified": True,
            "local_checkpoint_verification_sha256": "b" * 64,
            "local_checkpoint_object_count": 1,
            "local_checkpoint_size_bytes": 1,
        },
    )

    class Client:
        def __init__(self, *, spec, policy, local_verification) -> None:
            self.spec = spec
            self.policy = policy

        def infer(self, observation):
            return {
                "actions": self.policy.infer(observation)["actions"],
                "policy_request_receipt": {
                    "receipt_sha256": "c" * 64,
                    "native_action_shape": [self.spec.action_chunk_rows, 8],
                },
            }

    monkeypatch.setattr(canary_module, "OpenPICurrentReferenceDroidPolicyClient", Client)
    calls: list[str] = []

    class Policy:
        def __init__(self, spec) -> None:
            self.spec = spec

        def infer(self, observation):
            calls.append(self.spec.policy_id)
            return {"actions": np.zeros((self.spec.action_chunk_rows, 8))}

    result = run_current_reference_policy_canary(
        source_freeze_path=source_freeze,
        checkpoint_inventory_dir=inventory_dir,
        initial_observation_manifest_path=initial,
        output_dir=tmp_path / "output",
        checkpoint_downloader=lambda _: tmp_path,
        policy_loader=lambda spec, _: Policy(spec),
        initial_observation_loader=lambda _: {"prompt": "fixture"},
        gpu_evidence_collector=lambda: {"gpu_device_present": True},
    )
    assert result["status"] == "completed"
    assert calls == ["pi0_droid", "pi0_fast_droid", "pi05_droid"]
    assert [row["native_action_shape"] for row in result["policy_results"]] == [
        [10, 8],
        [10, 8],
        [15, 8],
    ]
    assert result["wam_called"] is False
    assert result["judge_called"] is False

    attempted_loads: list[str] = []

    def load_with_one_policy_failure(spec, _checkpoint):
        attempted_loads.append(spec.policy_id)
        if spec.policy_id == "pi0_droid":
            raise RuntimeError("fixture_policy_specific_failure")
        return Policy(spec)

    partial = run_current_reference_policy_canary(
        source_freeze_path=source_freeze,
        checkpoint_inventory_dir=inventory_dir,
        initial_observation_manifest_path=initial,
        output_dir=tmp_path / "partial-output",
        checkpoint_downloader=lambda _: tmp_path,
        policy_loader=load_with_one_policy_failure,
        initial_observation_loader=lambda _: {"prompt": "fixture"},
        gpu_evidence_collector=lambda: {"gpu_device_present": True},
    )
    assert partial["status"] == "blocked"
    assert attempted_loads == ["pi0_droid", "pi0_fast_droid", "pi05_droid"]
    assert [row["status"] for row in partial["policy_results"]] == [
        "blocked",
        "completed",
        "completed",
    ]


def test_policy_canary_fails_before_download_without_gpu(tmp_path: Path, monkeypatch) -> None:
    source_freeze = tmp_path / "source_freeze.json"
    source_freeze.write_text("{}", encoding="utf-8")
    initial = tmp_path / "initial.json"
    initial.write_text("{}", encoding="utf-8")
    inventory_dir = tmp_path / "inventories"
    inventory_dir.mkdir()
    monkeypatch.setattr(
        canary_module,
        "load_current_reference_policy_specs",
        lambda **_: {policy_id: object() for policy_id in canary_module.FROZEN_POLICY_ORDER},
    )
    downloads: list[str] = []
    result = run_current_reference_policy_canary(
        source_freeze_path=source_freeze,
        checkpoint_inventory_dir=inventory_dir,
        initial_observation_manifest_path=initial,
        output_dir=tmp_path / "output",
        checkpoint_downloader=lambda uri: downloads.append(uri),
        initial_observation_loader=lambda _: {},
        gpu_evidence_collector=lambda: {"gpu_device_present": False},
    )
    assert result["status"] == "blocked"
    assert downloads == []
    assert result["blockers"] == ["openpi_current_reference_jax_gpu_not_present"]
