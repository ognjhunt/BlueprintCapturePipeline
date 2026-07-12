from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline import g1_kitchen_digitalocean_closure as closure


def _identity() -> dict[str, str]:
    digest = "a" * 64
    return {
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
        "source_commit": "b" * 40,
        "source_dirty_patch_sha256": digest,
        "image_digest": digest,
        "bundle_digest": digest,
        "kitchen_asset_digest": digest,
        "active_selection_sha256": digest,
        "task_contract_sha256": digest,
        "provider_allocation_id": "do-1",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collected_media_rows_use_manifest_horizon_not_discovered_files(
    tmp_path: Path, monkeypatch
) -> None:
    identity = _identity()
    missing = closure._collected_media_rows(
        collected_root=tmp_path,
        identity=identity,
        expected_frame_count=20,
        expected_scenario_count=1,
        step_bindings=None,
        attestation_pins=None,
    )
    assert missing["robot_pov"]["status"] == "blocked"
    assert "full_ordered_episode_media_not_collected" in missing["semantic_review"][
        "blockers"
    ]

    frames = tmp_path / "closed_loop_out" / "scenario-1" / "frames"
    frames.mkdir(parents=True)
    for role in ("overview", "robot_pov"):
        (frames / f"{role}_0000.png").write_bytes(b"frame")
    calls: list[tuple[Path, int, object]] = []

    def fake_admit(
        *, scenario_dir, expected_frame_count, step_bindings, attestation_pins,
        identity_binding
    ):
        assert attestation_pins is None
        assert identity_binding == identity
        calls.append((Path(scenario_dir), expected_frame_count, step_bindings))
        return {"status": "passed", "blockers": [], "full_ordered_episode_admitted": True}

    monkeypatch.setattr(closure, "admit_collected_scenario_episode", fake_admit)
    bindings = [{"step_index": 0}] * 20
    admitted = closure._collected_media_rows(
        collected_root=tmp_path,
        identity=identity,
        expected_frame_count=20,
        expected_scenario_count=1,
        step_bindings=bindings,
        attestation_pins=None,
    )

    assert calls == [(frames.parent, 20, bindings)]
    assert admitted["robot_pov"]["status"] == "passed"

    extra = tmp_path / "closed_loop_out" / "scenario-2" / "frames"
    extra.mkdir(parents=True)
    (extra / "overview_0000.png").write_bytes(b"frame")
    surplus = closure._collected_media_rows(
        collected_root=tmp_path,
        identity=identity,
        expected_frame_count=20,
        expected_scenario_count=1,
        step_bindings=bindings,
        attestation_pins=None,
    )
    assert surplus["robot_pov"]["status"] == "blocked"
    assert any(
        "scenario_count_mismatch" in item
        for item in surplus["robot_pov"]["blockers"]
    )


class _Provider:
    def billable_inventory(self, *, name_prefix: str) -> dict:
        return {"api_confirmed": True, "live_resource_count": 0}


def _finalize(tmp_path: Path, *, watch_rows: dict, collected_rows: dict) -> dict:
    identity = _identity()
    contract = tmp_path / "task_success_contract.json"
    contract.write_text(json.dumps({"task_id": "microwave_door"}), encoding="utf-8")
    archive = tmp_path / "kitchen_assets.tar.gz"
    archive.write_bytes(b"assets")
    attempt_input = {
        "schema_version": "g1_kitchen_attempt_input_manifest.v1",
        "run_id": identity["run_id"],
        "attempt_id": identity["attempt_id"],
        "launch_nonce": identity["launch_nonce"],
        "source_commit": identity["source_commit"],
        "source_dirty_patch_sha256": identity["source_dirty_patch_sha256"],
        "image_digest": identity["image_digest"],
        "artifacts": {
            "bundle": {"sha256": identity["bundle_digest"]},
            "kitchen_inventory": {"sha256": identity["kitchen_asset_digest"]},
            "selection": {"sha256": identity["active_selection_sha256"]},
            "task_success_contract": {"sha256": identity["task_contract_sha256"]},
        },
    }
    manifest_file = tmp_path / "attempt_input_manifest.json"
    manifest_file.write_text(json.dumps(attempt_input), encoding="utf-8")
    collected_manifest = (
        tmp_path
        / "closed_loop_output"
        / "closed_loop_out"
        / "oscar_isaac_closed_loop_manifest.json"
    )
    _write_json(collected_manifest, {"g1_kitchen_proof_rows": collected_rows})
    return closure.finalize_digitalocean_attempt_closure(
        provider=_Provider(),
        output_dir=tmp_path,
        image_ref=f"registry/image@sha256:{identity['image_digest']}",
        attempt_input_manifest_file=manifest_file,
        task_success_contract_file=contract,
        kitchen_asset_archive_file=archive,
        launch={"status": "launched", "instance_id": identity["provider_allocation_id"]},
        watch={"runner_result": {"closed_loop_manifest": {"g1_kitchen_proof_rows": watch_rows}}},
        teardown_proof={"status": "PASS", "provider_terminal_status": "terminated"},
        expected_episode_steps=2,
        expected_scenario_count=1,
    )


def test_finalize_never_repairs_worker_identity_and_hashes_collected_bytes(
    tmp_path: Path,
) -> None:
    forged_in_memory = {
        "controller_fk": {"status": "passed", "evidence": {"forged": True}}
    }
    collected = {
        "controller_fk": {"status": "passed", "leaf_artifacts": []}
    }
    result = _finalize(tmp_path, watch_rows=forged_in_memory, collected_rows=collected)
    closure_doc = result["closure"]
    assert closure_doc["status"] == "blocked"
    rows = {row["row_id"]: row for row in closure_doc["proof_rows"]}
    controller = rows["controller_fk"]
    assert controller["status"] == "blocked"
    assert controller["identity_binding"] == {}
    assert controller["evidence"].get("forged") is None
    collected_manifest = (
        tmp_path
        / "closed_loop_output"
        / "closed_loop_out"
        / "oscar_isaac_closed_loop_manifest.json"
    )
    assert controller["evidence"]["worker_manifest_sha256"] == hashlib.sha256(
        collected_manifest.read_bytes()
    ).hexdigest()
    assert any(
        "worker_identity_binding_missing" in item for item in controller["blockers"]
    )


def test_finalize_blocks_when_collected_worker_manifest_is_absent(
    tmp_path: Path,
) -> None:
    identity = _identity()
    contract = tmp_path / "task_success_contract.json"
    contract.write_text(json.dumps({"task_id": "microwave_door"}), encoding="utf-8")
    archive = tmp_path / "kitchen_assets.tar.gz"
    archive.write_bytes(b"assets")
    manifest_file = tmp_path / "attempt_input_manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "run_id": identity["run_id"],
                "attempt_id": identity["attempt_id"],
                "launch_nonce": identity["launch_nonce"],
                "source_commit": identity["source_commit"],
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    result = closure.finalize_digitalocean_attempt_closure(
        provider=_Provider(),
        output_dir=tmp_path,
        image_ref=f"registry/image@sha256:{identity['image_digest']}",
        attempt_input_manifest_file=manifest_file,
        task_success_contract_file=contract,
        kitchen_asset_archive_file=archive,
        launch={"status": "launched", "instance_id": "do-1"},
        watch={},
        teardown_proof={"status": "PASS", "provider_terminal_status": "terminated"},
        expected_episode_steps=2,
        expected_scenario_count=1,
    )
    closure_doc = result["closure"]
    assert closure_doc["status"] == "blocked"
    assert any(
        "collected_worker_manifest_missing" in item
        for item in closure_doc["blockers"]
    )


def test_finalize_loads_attempt_bound_worker_public_key_pins(
    tmp_path: Path, monkeypatch
) -> None:
    identity = _identity()
    pin_file = tmp_path / "closed_loop_output" / "runtime_ephemeral_trust.json"
    _write_json(
        pin_file,
        {
            "schema_version": "g1_kitchen_attestation_public_key_pins.v1",
            "algorithm": "ed25519",
            "identity_binding": identity,
            "public_keys": {},
            "roles": {},
        },
    )
    observed: dict = {}

    def fake_validate(**kwargs):
        observed.update(kwargs["attestation_pins"] or {})
        return {"rows": {}, "blockers": []}

    monkeypatch.setattr(closure, "validate_worker_proof_rows", fake_validate)
    _finalize(tmp_path, watch_rows={}, collected_rows={})

    assert observed["identity_binding"] == identity
    assert observed["schema_version"] == "g1_kitchen_attestation_public_key_pins.v1"


def test_finalize_rejects_worker_public_key_pins_from_another_attempt(
    tmp_path: Path, monkeypatch
) -> None:
    identity = _identity()
    pin_file = tmp_path / "closed_loop_output" / "runtime_ephemeral_trust.json"
    _write_json(
        pin_file,
        {
            "schema_version": "g1_kitchen_attestation_public_key_pins.v1",
            "algorithm": "ed25519",
            "identity_binding": identity | {"attempt_id": "replayed-attempt"},
            "public_keys": {},
            "roles": {},
        },
    )
    observed: list[object] = []

    def fake_validate(**kwargs):
        observed.append(kwargs["attestation_pins"])
        return {"rows": {}, "blockers": []}

    monkeypatch.setattr(closure, "validate_worker_proof_rows", fake_validate)
    _finalize(tmp_path, watch_rows={}, collected_rows={})

    assert observed == [None]
