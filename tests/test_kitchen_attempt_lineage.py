from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.kitchen_attempt_lineage import (
    activate_selection_generation,
    allocate_attempt_id,
    build_attempt_input_manifest,
    sha256_file,
    write_attempt_input_manifest,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_selection_supersession_preserves_prior_and_checksum_binds_pointer(tmp_path) -> None:
    first_path = _write(
        tmp_path / "selections" / "sink" / "random_task_selection.json",
        {"schema_version": "kitchen_random_task_selection.v1", "selected_task_id": "sink"},
    )
    first = {
        "selection_id": "sink",
        "selection_path": str(first_path),
        "selection_sha256": sha256_file(first_path),
    }
    pointer1 = activate_selection_generation(
        run_dir=tmp_path,
        generation=first,
        active_from_attempt_id="run-attempt-000001",
    )

    invalidation = _write(
        tmp_path / "invalidate-sink.json",
        {"status": "invalidated", "reason": "fresh scene target missing"},
    )
    microwave_path = _write(
        tmp_path / "selections" / "microwave" / "random_task_selection.json",
        {"schema_version": "kitchen_random_task_selection.v1", "selected_task_id": "microwave"},
    )
    microwave = {
        "selection_id": "microwave",
        "selection_path": str(microwave_path),
        "selection_sha256": sha256_file(microwave_path),
    }
    pointer2 = activate_selection_generation(
        run_dir=tmp_path,
        generation=microwave,
        active_from_attempt_id="run-attempt-000002",
        prior_pointer_path=pointer1["pointer_path"],
        invalidation_path=invalidation,
    )
    events = [json.loads(line) for line in (tmp_path / "selection_supersession.jsonl").read_text().splitlines()]
    assert len(events) == 2
    assert events[1]["prior_selection_sha256"] == first["selection_sha256"]
    assert events[1]["replacement_selection_sha256"] == microwave["selection_sha256"]
    assert pointer2["selection_sha256"] == microwave["selection_sha256"]
    assert pointer2["evidence_policy"] == "pointer_requires_attempt_input_manifest"


def test_replacement_requires_invalidation(tmp_path) -> None:
    path = _write(tmp_path / "selection.json", {"selected_task_id": "sink"})
    generation = {
        "selection_id": "sink",
        "selection_path": str(path),
        "selection_sha256": sha256_file(path),
    }
    pointer = activate_selection_generation(
        run_dir=tmp_path,
        generation=generation,
        active_from_attempt_id="a-1",
    )
    with pytest.raises(ValueError, match="replacement requires"):
        activate_selection_generation(
            run_dir=tmp_path,
            generation=generation,
            active_from_attempt_id="a-2",
            prior_pointer_path=pointer["pointer_path"],
        )


def test_attempt_ids_are_atomic_and_unique(tmp_path) -> None:
    first = allocate_attempt_id(run_dir=tmp_path, run_id="run")
    second = allocate_attempt_id(run_dir=tmp_path, run_id="run")
    assert first["attempt_id"] == "run-attempt-000001"
    assert second["attempt_id"] == "run-attempt-000002"
    assert first["attempt_dir"] != second["attempt_dir"]


def test_attempt_input_rejects_stale_task_contract_before_spend(tmp_path) -> None:
    selection = _write(
        tmp_path / "selection.json",
        {"schema_version": "kitchen_random_task_selection.v1", "selected_task_id": "microwave"},
    )
    selection_sha = sha256_file(selection)
    scenario = _write(
        tmp_path / "scenario.json",
        {"source_selection_sha256": selection_sha},
    )
    route = _write(tmp_path / "route.json", {"source_selection_sha256": selection_sha})
    stale = _write(
        tmp_path / "task.json",
        {"task_id": "sink", "source_selection_sha256": selection_sha},
    )
    inventory = _write(tmp_path / "inventory.json", {"inventory_sha256": "a" * 64})
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")
    worker_evidence = _write(
        tmp_path / "worker_image_runtime_evidence.json",
        {
            "schema_version": "g1_kitchen_worker_image_runtime_evidence.v1",
            "status": "passed",
            "image_digest": "sha256:" + "a" * 64,
            "source_commit": "d1220f788",
            "source_dirty_patch_sha256": "b" * 64,
            "runtime_metadata": {
                "image_family": "isaac-eval-worker",
                "simulator_family": "isaac_sim",
                "simulator_major_version": 6,
                "source_commit": "d1220f788",
                "source_dirty_patch_sha256": "b" * 64,
                "blueprint_pipeline_imported": True,
                "configured_g1_asset_binding_valid": True,
                "configured_g1_usd_exists": False,
                "g1_asset_resolution_deferred_to_runtime": True,
                "build_time_healthcheck_passed": True,
            },
            "fast_canary": {
                "status": "passed",
                "image_digest": "sha256:" + "a" * 64,
                "provider_allocation_id": "pod-1",
                "launch_nonce": "nonce",
            },
            "review_canary": {
                "status": "passed",
                "image_digest": "sha256:" + "a" * 64,
                "provider_allocation_id": "pod-1",
                "launch_nonce": "nonce",
                "width": 640,
                "height": 480,
            },
            "teardown": {"api_confirmed": True, "terminal_state": "not_found"},
            "final_inventory": {"api_confirmed": True, "live_resource_count": 0},
        },
    )
    artifacts = {
        "selection": selection,
        "scenario": scenario,
        "route": route,
        "task_success_contract": stale,
        "kitchen_inventory": inventory,
        "bundle": bundle,
        "worker_image_runtime_evidence": worker_evidence,
    }
    with pytest.raises(ValueError, match="does not match active selection"):
        build_attempt_input_manifest(
            run_id="run",
            attempt_id="run-attempt-000001",
            launch_nonce="nonce",
            provider="runpod",
            artifacts=artifacts,
            image_digest="sha256:" + "a" * 64,
            source_commit="d1220f788",
            source_dirty_patch_sha256="b" * 64,
        )

    stale.write_text(
        json.dumps({"task_id": "microwave", "source_selection_sha256": selection_sha}),
        encoding="utf-8",
    )
    route.write_text(json.dumps({"source_selection_sha256": "f" * 64}))
    with pytest.raises(ValueError, match="route selection checksum mismatch"):
        build_attempt_input_manifest(
            run_id="run",
            attempt_id="run-attempt-000001",
            launch_nonce="nonce",
            provider="runpod",
            artifacts=artifacts,
            image_digest="sha256:" + "a" * 64,
            source_commit="d1220f788",
            source_dirty_patch_sha256="b" * 64,
        )
    route.write_text(json.dumps({"source_selection_sha256": selection_sha}))

    manifest = build_attempt_input_manifest(
        run_id="run",
        attempt_id="run-attempt-000001",
        launch_nonce="nonce",
        provider="runpod",
        artifacts=artifacts,
        image_digest="sha256:" + "a" * 64,
        source_commit="d1220f788",
        source_dirty_patch_sha256="b" * 64,
    )
    target = write_attempt_input_manifest(attempt_dir=tmp_path / "attempt", manifest=manifest)
    assert json.loads(target.read_text())["selected_task_id"] == "microwave"
    with pytest.raises(FileExistsError):
        write_attempt_input_manifest(attempt_dir=tmp_path / "attempt", manifest=manifest)
