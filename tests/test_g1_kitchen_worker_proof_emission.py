from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline.g1_kitchen_leaf_evidence import ROLE_PRIVATE_KEY_ENVS
from blueprint_pipeline.g1_kitchen_proof_row_validation import (
    ATTESTATION_PINS_SCHEMA_VERSION,
    WORKER_PROOF_ROW_SPECS,
    validate_worker_proof_rows,
)
from blueprint_pipeline.g1_kitchen_startup_proof import SPECS, sign_startup_proof_rows
from blueprint_pipeline.g1_kitchen_worker_proof_emission import emit_worker_proof_rows
from blueprint_pipeline.task_episode_baseline import build_task_episode_baseline


def _attempt(path: Path) -> dict[str, str]:
    digest = "a" * 64
    identity = {
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
        "source_commit": "b" * 40,
        "source_dirty_patch_sha256": "c" * 64,
        "image_digest": digest,
        "bundle_digest": digest,
        "kitchen_asset_digest": digest,
        "active_selection_sha256": digest,
        "task_contract_sha256": digest,
        "provider_allocation_id": "12345",
    }
    path.write_text(
        json.dumps(
            {
                **{key: identity[key] for key in (
                    "run_id", "attempt_id", "launch_nonce", "source_commit",
                    "source_dirty_patch_sha256", "image_digest",
                )},
                "artifacts": {
                    "bundle": {"sha256": digest},
                    "kitchen_inventory": {"sha256": digest},
                    "selection": {"sha256": digest},
                    "task_success_contract": {"sha256": digest},
                },
            }
        ),
        encoding="utf-8",
    )
    return identity


def _keys(tmp_path: Path, monkeypatch) -> dict:
    public_keys = {}
    roles = {}
    for role, env_name in ROLE_PRIVATE_KEY_ENVS.items():
        key = Ed25519PrivateKey.generate()
        key_path = tmp_path / f"{role}.pem"
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        monkeypatch.setenv(env_name, str(key_path))
        raw = key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        fingerprint = hashlib.sha256(raw).hexdigest()
        public_keys[fingerprint] = base64.b64encode(raw).decode()
        roles[role] = [fingerprint]
    return {
        "schema_version": ATTESTATION_PINS_SCHEMA_VERSION,
        "algorithm": "ed25519",
        "public_keys": public_keys,
        "roles": roles,
    }


def test_emitter_output_is_directly_accepted_by_host_validator(
    tmp_path: Path, monkeypatch
) -> None:
    pins = _keys(tmp_path, monkeypatch)
    monkeypatch.setenv("BLUEPRINT_PROVIDER_ALLOCATION_ID", "12345")
    attempt_path = tmp_path / "attempt_input_manifest.json"
    identity = _attempt(attempt_path)
    out = tmp_path / "closed_loop_out"
    startup = out / "startup_gates"
    for _, (relative, schema) in SPECS.items():
        path = startup / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        status = "completed" if schema == "kitchen_asset_startup_gate.v1" else "passed"
        path.write_text(json.dumps({"schema_version": schema, "status": status}))
    sign_startup_proof_rows(
        startup_dir=startup,
        attempt_input_manifest=attempt_path,
    )
    action_sha = hashlib.sha256(b"action-0").hexdigest()
    controller = out / "controller_fk_skeleton" / "step_0001" / "controller_fk_output.json"
    controller.parent.mkdir(parents=True)
    controller.write_text(
        json.dumps(
            {
                "schema_version": "gear_sonic_controller_fk_execution.v1",
                "status": "completed",
                "source_action_sha256": action_sha,
                "official_controller_action_applied": True,
            }
        )
    )
    baseline = build_task_episode_baseline(
        episode_initial_value=0.0,
        attempt_id=identity["attempt_id"],
        launch_nonce=identity["launch_nonce"],
        simulator_session_id="session-1",
        stage_id="d" * 64,
        articulation_prim_path="/root/Microwave017/Door",
        task_contract_sha256=identity["task_contract_sha256"],
        criterion_id="microwave_door_open",
        unit="rad",
        captured_timestamp="900",
    )
    measurement = {
        "schema_version": "task_transition_measurement.v1",
        "source_step_index": 1,
        "source_action_sha256": action_sha,
        "simulator_session_id": "session-1",
        "stage_id": "d" * 64,
        "before_timestamp": "1000",
        "after_timestamp": "1005",
        "articulation_prim_path": "/root/Microwave017/Door",
        "episode_baseline_digest": baseline["baseline_digest"],
        "episode_initial_value": 0.0,
        "episode_baseline": baseline,
        "episode_baseline_attestation": {"signature_verified": True},
    }
    rows = emit_worker_proof_rows(
        output_dir=out,
        attempt_input_manifest=attempt_path,
        task_completion_results=[measurement],
        controller_result_paths=[controller],
        consistency_results=[
            {
                "forward_dynamics_consistency_proven": True,
                "inverse_dynamics_consistency_proven": True,
            }
        ],
        manipulation_success_judge={
            "manipulation_success_proven": True,
            "did_target_manipulation_succeed": True,
        },
        action_sha256s=[action_sha],
        planned_max_steps=8,
        termination_reason="task_completed",
        task_completed=True,
        scenario_count=1,
        geometry_results={
            "stance": {
                "schema_version": "g1_kitchen_live_stance_validation.v1",
                "stance_valid": True,
                "reach_valid": True,
                "facing_valid": True,
            },
            "collision": {
                "schema_version": "g1_kitchen_live_collision_validation.v1",
                "collision_free": True,
                "clearance_valid": True,
            },
        },
    )
    manifest = out / "oscar_isaac_closed_loop_manifest.json"
    manifest.write_text(json.dumps({"g1_kitchen_proof_rows": rows}))
    result = validate_worker_proof_rows(
        worker_rows=rows,
        worker_manifest_path=manifest,
        collected_root=tmp_path,
        identity=identity,
        attestation_pins=pins,
    )
    assert set(result["rows"]) == set(WORKER_PROOF_ROW_SPECS)
    assert result["blockers"] == []
    assert {row["status"] for row in result["rows"].values()} == {"passed"}
