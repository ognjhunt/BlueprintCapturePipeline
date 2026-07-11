from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline.g1_kitchen_proof_row_validation import (
    WORKER_PROOF_ROW_SPECS,
    validate_worker_proof_rows,
)
from blueprint_pipeline.task_episode_baseline import build_task_episode_baseline


def _identity() -> dict[str, str]:
    digest = "a" * 64
    return {
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
        "source_commit": "b" * 40,
        "source_dirty_patch_sha256": "d" * 64,
        "image_digest": digest,
        "bundle_digest": digest,
        "kitchen_asset_digest": digest,
        "active_selection_sha256": digest,
        "task_contract_sha256": digest,
        "provider_allocation_id": "do-1",
    }


def _fingerprint(key: Ed25519PrivateKey) -> str:
    from cryptography.hazmat.primitives import serialization

    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _pins(key: Ed25519PrivateKey) -> dict[str, Any]:
    from cryptography.hazmat.primitives import serialization

    raw = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    fingerprint = _fingerprint(key)
    roles = sorted(
        {
            role
            for spec in WORKER_PROOF_ROW_SPECS.values()
            for role in (
                [spec["attestation_role"]]
                if spec.get("attestation_role")
                else list(spec["attestation_roles_by_schema"].values())
            )
        }
    )
    return {
        "schema_version": "g1_kitchen_attestation_public_key_pins.v1",
        "algorithm": "ed25519",
        "public_keys": {fingerprint: base64.b64encode(raw).decode()},
        "roles": {role: [fingerprint] for role in roles},
    }


def _write_leaf(
    *,
    collected_root: Path,
    relative: str,
    payload: dict[str, Any],
    key: Ed25519PrivateKey,
    role: str,
) -> dict[str, Any]:
    path = collected_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode()
    path.write_bytes(data)
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "schema_version": payload["schema_version"],
        "attestation": {
            "algorithm": "ed25519",
            "role": role,
            "public_key_fingerprint": _fingerprint(key),
            "signature_b64": base64.b64encode(key.sign(data)).decode(),
        },
    }


def _measurement_payload(
    identity: dict[str, str], *, step: int, action_sha: str
) -> dict[str, Any]:
    baseline = build_task_episode_baseline(
        episode_initial_value=0.0,
        attempt_id=identity["attempt_id"],
        launch_nonce=identity["launch_nonce"],
        simulator_session_id="session-1",
        stage_id="c" * 64,
        articulation_prim_path="/root/Microwave017/Microwave017_Door",
        task_contract_sha256=identity["task_contract_sha256"],
        criterion_id="microwave_door_open",
        unit="rad",
        captured_timestamp="900",
    )
    return {
        "schema_version": "task_transition_measurement.v1",
        "criterion_id": "microwave_door_open",
        "before_value": 0.0 + step * 0.2,
        "after_value": 0.2 + step * 0.2,
        "source_step_index": step,
        "source_action_sha256": action_sha,
        "articulation_prim_path": "/root/Microwave017/Microwave017_Door",
        "simulator_session_id": "session-1",
        "stage_id": "c" * 64,
        "before_timestamp": str(1000 + step * 10),
        "after_timestamp": str(1005 + step * 10),
        "episode_baseline_digest": baseline["baseline_digest"],
        "episode_initial_value": 0.0,
        "episode_baseline": baseline,
        "episode_baseline_attestation": {"signature_verified": True},
        "identity_binding": dict(identity),
    }


def _signed_worker_rows(
    collected_root: Path, identity: dict[str, str], key: Ed25519PrivateKey
) -> dict[str, dict[str, Any]]:
    action_shas = [hashlib.sha256(f"action-{i}".encode()).hexdigest() for i in range(2)]
    measurement_leafs = [
        _write_leaf(
            collected_root=collected_root,
            relative=f"closed_loop_out/task_measurement_{i:04d}.json",
            payload=_measurement_payload(identity, step=i, action_sha=action_shas[i]),
            key=key,
            role="task_transition",
        )
        for i in range(2)
    ]
    judge_leaf = _write_leaf(
        collected_root=collected_root,
        relative="closed_loop_out/manipulation_success_evaluator_results.json",
        payload={
            "schema_version": "isaac_manipulation_success_evaluator_results.v1",
            "manipulation_success_proven": True,
            "did_target_manipulation_succeed": True,
            "identity_binding": dict(identity),
        },
        key=key,
        role="task_transition",
    )
    controller_leafs = [
        _write_leaf(
            collected_root=collected_root,
            relative=f"closed_loop_out/controller_fk_{i:04d}.json",
            payload={
                "schema_version": "gear_sonic_controller_fk_execution.v1",
                "status": "completed",
                "source_action_sha256": action_shas[i],
                "official_controller_action_applied": True,
                "identity_binding": dict(identity),
            },
            key=key,
            role="controller",
        )
        for i in range(2)
    ]
    policy_leaf = _write_leaf(
        collected_root=collected_root,
        relative="closed_loop_out/policy_action_sequence.json",
        payload={
            "schema_version": "g1_kitchen_policy_action_sequence.v1",
            "source_action_sha256s": action_shas,
            "identity_binding": dict(identity),
        },
        key=key,
        role="policy",
    )
    scorer_leaf = _write_leaf(
        collected_root=collected_root,
        relative="closed_loop_out/strict_consistency_results.json",
        payload={
            "schema_version": "strict_action_aware_consistency_contract.v1",
            "forward_consistency_proven": True,
            "inverse_consistency_proven": True,
            "source_action_sha256s": action_shas,
            "identity_binding": dict(identity),
        },
        key=key,
        role="scorer",
    )
    stance_leaf = _write_leaf(
        collected_root=collected_root,
        relative="closed_loop_out/live_stance_validation.json",
        payload={
            "schema_version": "g1_kitchen_live_stance_validation.v1",
            "stance_valid": True,
            "reach_valid": True,
            "facing_valid": True,
            "identity_binding": dict(identity),
        },
        key=key,
        role="geometry",
    )
    collision_leaf = _write_leaf(
        collected_root=collected_root,
        relative="closed_loop_out/live_collision_validation.json",
        payload={
            "schema_version": "g1_kitchen_live_collision_validation.v1",
            "collision_free": True,
            "clearance_valid": True,
            "identity_binding": dict(identity),
        },
        key=key,
        role="geometry",
    )
    startup_leafs = {}
    for row_id, schema, status in (
        ("startup", "groot_oscar_same_allocation_startup_gates.v1", "passed"),
        ("fast_canary", "isaac_worker_runtime_preflight.v1", "passed"),
        ("review_canary", "isaac_review_renderer_canary.v1", "passed"),
        ("asset_gate", "kitchen_asset_startup_gate.v1", "completed"),
    ):
        startup_leafs[row_id] = _write_leaf(
            collected_root=collected_root,
            relative=f"closed_loop_out/startup_{row_id}.json",
            payload={
                "schema_version": schema,
                "status": status,
                "identity_binding": dict(identity),
            },
            key=key,
            role="startup",
        )
    horizon_leaf = _write_leaf(
        collected_root=collected_root,
        relative="closed_loop_out/terminal_horizon.json",
        payload={
            "schema_version": "g1_kitchen_terminal_horizon.v1",
            "planned_max_steps": 8,
            "executed_step_count": 2,
            "terminal_step_index": 1,
            "termination_reason": "task_completed",
            "task_completed": True,
            "scenario_count": 1,
            "source_action_sha256s": action_shas,
            "simulator_session_id": "session-1",
            "stage_id": "c" * 64,
            "identity_binding": dict(identity),
        },
        key=key,
        role="task_transition",
    )
    binding = dict(identity)
    return {
        **{
            row_id: {
                "status": "passed",
                "identity_binding": binding,
                "leaf_artifacts": [leaf],
            }
            for row_id, leaf in startup_leafs.items()
        },
        "stance": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": [stance_leaf],
        },
        "collision": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": [collision_leaf],
        },
        "scene_load": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": measurement_leafs,
        },
        "target": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": measurement_leafs,
        },
        "controller_fk": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": [policy_leaf, *controller_leafs],
        },
        "persistent_simulator_transition": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": [*measurement_leafs, judge_leaf, horizon_leaf],
        },
        "forward_consistency": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": [scorer_leaf],
        },
        "inverse_consistency": {
            "status": "passed",
            "identity_binding": binding,
            "leaf_artifacts": [scorer_leaf],
        },
    }


def _write_worker_manifest(
    collected_root: Path, rows: dict[str, dict[str, Any]]
) -> Path:
    path = collected_root / "closed_loop_out" / "oscar_isaac_closed_loop_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "completed", "g1_kitchen_proof_rows": rows}, indent=2),
        encoding="utf-8",
    )
    return path


def _validate(tmp_path: Path, rows, key, identity=None, pins=...):
    identity = identity or _identity()
    manifest_path = _write_worker_manifest(tmp_path, rows)
    return validate_worker_proof_rows(
        worker_rows=rows,
        worker_manifest_path=manifest_path,
        collected_root=tmp_path,
        identity=identity,
        attestation_pins=_pins(key) if pins is ... else pins,
    )


def test_complete_signed_fixture_passes_and_preserves_digests(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity()
    rows = _signed_worker_rows(tmp_path, identity, key)
    result = _validate(tmp_path, rows, key, identity=identity)

    assert result["blockers"] == []
    assert {row["status"] for row in result["rows"].values()} == {"passed"}
    manifest_path = (
        tmp_path / "closed_loop_out" / "oscar_isaac_closed_loop_manifest.json"
    )
    assert result["worker_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert result["worker_manifest_path"] == str(manifest_path)
    verified = result["rows"]["persistent_simulator_transition"]["evidence"][
        "verified_leaf_artifacts"
    ]
    assert all(len(item["sha256"]) == 64 for item in verified)


def test_forged_passed_row_with_no_leaf_artifacts_blocks(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    rows = _signed_worker_rows(tmp_path, _identity(), key)
    rows["persistent_simulator_transition"] = {
        "status": "passed",
        "identity_binding": dict(_identity()),
        "leaf_artifacts": [],
    }
    result = _validate(tmp_path, rows, key)
    row = result["rows"]["persistent_simulator_transition"]
    assert row["status"] == "blocked"
    assert any("leaf_artifacts_missing" in item for item in row["blockers"])


def test_worker_status_boolean_is_never_the_verdict(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity()
    rows = _signed_worker_rows(tmp_path, identity, key)
    judge = _write_leaf(
        collected_root=tmp_path,
        relative="closed_loop_out/manipulation_success_evaluator_results.json",
        payload={
            "schema_version": "isaac_manipulation_success_evaluator_results.v1",
            "manipulation_success_proven": False,
            "did_target_manipulation_succeed": False,
            "identity_binding": dict(identity),
        },
        key=key,
        role="task_transition",
    )
    leafs = rows["persistent_simulator_transition"]["leaf_artifacts"]
    rows["persistent_simulator_transition"]["leaf_artifacts"] = [*leafs[:-1], judge]
    result = _validate(tmp_path, rows, key, identity=identity)
    row = result["rows"]["persistent_simulator_transition"]
    assert row["status"] == "blocked"


def test_tampered_artifact_bytes_block(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    rows = _signed_worker_rows(tmp_path, _identity(), key)
    target = tmp_path / "closed_loop_out" / "task_measurement_0000.json"
    payload = json.loads(target.read_text())
    payload["after_value"] = 99.0
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    result = _validate(tmp_path, rows, key)
    row = result["rows"]["persistent_simulator_transition"]
    assert row["status"] == "blocked"
    assert any("sha256_mismatch" in item for item in row["blockers"])


def test_invalid_signature_blocks(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    forged_key = Ed25519PrivateKey.generate()
    identity = _identity()
    rows = _signed_worker_rows(tmp_path, identity, key)
    forged = _write_leaf(
        collected_root=tmp_path,
        relative="closed_loop_out/strict_consistency_results.json",
        payload={
            "schema_version": "strict_action_aware_consistency_contract.v1",
            "forward_consistency_proven": True,
            "inverse_consistency_proven": True,
            "identity_binding": dict(identity),
        },
        key=forged_key,
        role="scorer",
    )
    forged["attestation"]["public_key_fingerprint"] = _fingerprint(key)
    rows["forward_consistency"]["leaf_artifacts"] = [forged]
    result = _validate(tmp_path, rows, key, identity=identity)
    row = result["rows"]["forward_consistency"]
    assert row["status"] == "blocked"
    assert any("attestation_invalid" in item for item in row["blockers"])


def test_unpinned_fingerprint_and_missing_pins_block(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    rows = _signed_worker_rows(tmp_path, _identity(), key)

    no_pins = _validate(tmp_path, rows, key, pins=None)
    assert all(row["status"] == "blocked" for row in no_pins["rows"].values())
    assert any(
        "attestation_public_key_pins_missing" in item
        for row in no_pins["rows"].values()
        for item in row["blockers"]
    )

    other = Ed25519PrivateKey.generate()
    wrong_pins = _validate(tmp_path, rows, key, pins=_pins(other))
    assert all(row["status"] == "blocked" for row in wrong_pins["rows"].values())


def test_unknown_schema_blocks(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity()
    rows = _signed_worker_rows(tmp_path, identity, key)
    alien = _write_leaf(
        collected_root=tmp_path,
        relative="closed_loop_out/alien.json",
        payload={
            "schema_version": "totally_unknown.v9",
            "forward_consistency_proven": True,
            "identity_binding": dict(identity),
        },
        key=key,
        role="scorer",
    )
    rows["forward_consistency"]["leaf_artifacts"] = [alien]
    result = _validate(tmp_path, rows, key, identity=identity)
    row = result["rows"]["forward_consistency"]
    assert row["status"] == "blocked"
    assert any("schema" in item for item in row["blockers"])


def test_cross_attempt_replay_blocks(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity()
    foreign = {**identity, "attempt_id": "attempt-9", "launch_nonce": "nonce-9"}
    rows = _signed_worker_rows(tmp_path, identity, key)
    replayed = _write_leaf(
        collected_root=tmp_path,
        relative="closed_loop_out/strict_consistency_results.json",
        payload={
            "schema_version": "strict_action_aware_consistency_contract.v1",
            "forward_consistency_proven": True,
            "inverse_consistency_proven": True,
            "identity_binding": foreign,
        },
        key=key,
        role="scorer",
    )
    rows["forward_consistency"]["leaf_artifacts"] = [replayed]
    result = _validate(tmp_path, rows, key, identity=identity)
    row = result["rows"]["forward_consistency"]
    assert row["status"] == "blocked"
    assert any("leaf_identity_mismatch" in item for item in row["blockers"])


def test_worker_row_without_identity_is_not_repaired(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    rows = _signed_worker_rows(tmp_path, _identity(), key)
    del rows["controller_fk"]["identity_binding"]
    result = _validate(tmp_path, rows, key)
    row = result["rows"]["controller_fk"]
    assert row["status"] == "blocked"
    assert row["identity_binding"] == {}
    assert any(
        "worker_identity_binding_missing" in item for item in row["blockers"]
    )


def test_mismatched_worker_identity_blocks_and_is_not_overwritten(
    tmp_path: Path,
) -> None:
    key = Ed25519PrivateKey.generate()
    rows = _signed_worker_rows(tmp_path, _identity(), key)
    stale = dict(_identity())
    stale["launch_nonce"] = "stale-nonce"
    rows["controller_fk"]["identity_binding"] = stale
    result = _validate(tmp_path, rows, key)
    row = result["rows"]["controller_fk"]
    assert row["status"] == "blocked"
    assert row["identity_binding"]["launch_nonce"] == "stale-nonce"
    assert any(
        "worker_identity_binding_mismatch:launch_nonce" in item
        for item in row["blockers"]
    )


def test_missing_collected_worker_manifest_blocks(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    rows = _signed_worker_rows(tmp_path, _identity(), key)
    result = validate_worker_proof_rows(
        worker_rows=rows,
        worker_manifest_path=tmp_path / "closed_loop_out" / "missing_manifest.json",
        collected_root=tmp_path,
        identity=_identity(),
        attestation_pins=_pins(key),
    )
    assert "collected_worker_manifest_missing" in result["blockers"]
    assert result["worker_manifest_sha256"] is None


def test_broken_action_chronology_blocks(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity()
    rows = _signed_worker_rows(tmp_path, identity, key)
    gap = _write_leaf(
        collected_root=tmp_path,
        relative="closed_loop_out/task_measurement_0001.json",
        payload=_measurement_payload(
            identity, step=5, action_sha=hashlib.sha256(b"action-1").hexdigest()
        ),
        key=key,
        role="task_transition",
    )
    leafs = rows["persistent_simulator_transition"]["leaf_artifacts"]
    rows["persistent_simulator_transition"]["leaf_artifacts"] = [
        leafs[0],
        gap,
        leafs[2],
    ]
    result = _validate(tmp_path, rows, key, identity=identity)
    row = result["rows"]["persistent_simulator_transition"]
    assert row["status"] == "blocked"
    assert any("chronology" in item for item in row["blockers"])


def test_leaf_path_escape_is_rejected(tmp_path: Path) -> None:
    key = Ed25519PrivateKey.generate()
    identity = _identity()
    rows = _signed_worker_rows(tmp_path, identity, key)
    outside = tmp_path.parent / "outside_leaf.json"
    payload = {
        "schema_version": "strict_action_aware_consistency_contract.v1",
        "forward_consistency_proven": True,
        "inverse_consistency_proven": True,
        "identity_binding": dict(identity),
    }
    data = json.dumps(payload, sort_keys=True).encode()
    outside.write_bytes(data)
    rows["forward_consistency"]["leaf_artifacts"] = [
        {
            "path": "../outside_leaf.json",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "schema_version": payload["schema_version"],
            "attestation": {
                "algorithm": "ed25519",
                "role": "scorer",
                "public_key_fingerprint": _fingerprint(key),
                "signature_b64": base64.b64encode(key.sign(data)).decode(),
            },
        }
    ]
    result = _validate(tmp_path, rows, key, identity=identity)
    row = result["rows"]["forward_consistency"]
    assert row["status"] == "blocked"
    assert any("leaf_artifact_path_escape" in item for item in row["blockers"])
