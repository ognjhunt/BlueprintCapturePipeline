from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from blueprint_pipeline.groot_oscar_digitalocean_job_inputs import (
    _write_input_bundle,
    _write_payload_bundle,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_attempt_manifest_hashes_non_circular_payload_inside_transport_envelope(
    tmp_path,
) -> None:
    seed = tmp_path / "seed.png"
    seed.write_bytes(b"png")
    task = tmp_path / "task.json"
    task.write_text('{"task_id":"microwave_door"}')
    kitchen = tmp_path / "kitchen.zip"
    with zipfile.ZipFile(kitchen, "w") as archive:
        archive.writestr("KitchenRoom.usd", "usd")
    plan = {"schema_version": "plan.v1", "sealed_active": True}
    route = {"route_points": [[0, 0, 0], [0, 0, 0]]}
    payload = _write_payload_bundle(
        payload_zip=tmp_path / "payload.zip",
        plan=plan,
        route_payload=route,
        seed_path=seed,
        task_prompt="open microwave",
        seed_provenance={"source": "test"},
        task_success_contract_path=task,
        kitchen_asset_archive_path=kitchen,
    )
    worker_evidence = tmp_path / "worker_image_runtime_evidence.json"
    worker_evidence.write_text('{"status":"passed"}')
    attempt = tmp_path / "attempt.json"
    attempt.write_text(
        json.dumps(
            {
                "artifacts": {
                    "bundle": {"path": str(payload), "sha256": _sha(payload)},
                    "worker_image_runtime_evidence": {
                        "path": str(worker_evidence),
                        "sha256": _sha(worker_evidence),
                    },
                }
            }
        )
    )
    envelope = _write_input_bundle(
        bundle_zip=tmp_path / "envelope.zip",
        plan=plan,
        route_payload=route,
        seed_path=seed,
        task_prompt="open microwave",
        seed_provenance={"source": "test"},
        task_success_contract_path=task,
        attempt_input_manifest_path=attempt,
        kitchen_asset_archive_path=kitchen,
    )
    with zipfile.ZipFile(envelope) as archive:
        assert "immutable_payload_bundle.zip" in archive.namelist()
        assert "attempt_input_manifest.json" in archive.namelist()
        assert "worker_image_runtime_evidence.json" in archive.namelist()
        inventory = json.loads(
            archive.read("kitchen_asset_inventory_checksums.json")
        )
        transport = json.loads(archive.read("transport_envelope_manifest.json"))
    assert inventory["schema_version"] == "kitchen_asset_inventory_checksums.v1"
    assert inventory["main_usd"] == "KitchenRoom.usd"
    assert inventory["files"][0]["path"] == "KitchenRoom.usd"
    assert transport["payload_bundle_sha256"] == _sha(payload)
    assert transport["payload_bundle_is_attempt_manifest_bundle_identity"] is True


def test_transport_envelope_rejects_attempt_bundle_digest_mismatch(tmp_path) -> None:
    attempt = tmp_path / "attempt.json"
    attempt.write_text(
        json.dumps(
            {"artifacts": {"bundle": {"path": str(tmp_path / "none"), "sha256": "0" * 64}}}
        )
    )
    with pytest.raises(ValueError, match="payload_bundle_digest_mismatch"):
        _write_input_bundle(
            bundle_zip=tmp_path / "envelope.zip",
            plan={},
            route_payload={},
            seed_path=tmp_path / "seed",
            task_prompt="task",
            attempt_input_manifest_path=attempt,
        )
