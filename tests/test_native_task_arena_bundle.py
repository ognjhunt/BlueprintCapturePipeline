from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_bundle import (
    NativeTaskArenaBundleError,
    build_native_task_arena_bundle,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(root: Path, *, scene_id: str) -> Path:
    packet = root / f"packet-{scene_id}"
    assets = packet / "assets"
    assets.mkdir(parents=True)
    source_bindings = []
    for role in ("scene_collision", "scene_appearance", "task_object"):
        path = assets / f"{role}.usd"
        path.write_text(f"exact:{scene_id}:{role}\n", encoding="utf-8")
        source_bindings.append(
            {
                "semantic_role": role,
                "source": {"root": "evidence", "relative_path": path.name},
                "staged_relative_path": f"assets/{path.name}",
                "staged_size_bytes": path.stat().st_size,
                "staged_sha256": _sha(path),
            }
        )
    documents = {
        "native_task_arena_packet_request.v1.json": {"scene_id": scene_id},
        "native_task_runtime_contract.v1.json": {"contract_digest": "sha256:" + "c" * 64},
        "native_task_arena_scene_plan.v1.json": {"plan_digest": "sha256:" + "p" * 64},
    }
    artifacts = []
    for role, (name, value) in zip(
        ("packet_request", "runtime_contract", "arena_scene_plan"),
        documents.items(),
        strict=True,
    ):
        path = packet / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append(
            {
                "role": role,
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    receipt = {
        "schema_version": "native_task_arena_packet_receipt.v1",
        "status": "construction_packet_completed",
        "scene_id": scene_id,
        "task_id": f"task-{scene_id}",
        "request_digest": "sha256:" + "a" * 64,
        "runtime_contract_digest": "sha256:" + "c" * 64,
        "arena_scene_plan_digest": "sha256:" + "b" * 64,
        "scenario_instance_digest": "sha256:" + "d" * 64,
        "source_bindings": source_bindings,
        "artifacts": artifacts,
        "source_bytes_mutated": False,
        "native_application_claimed": False,
        "policy_episode_claimed": False,
        "simulator_execution_is_not_physical_truth": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    (packet / "native_task_arena_packet_receipt.v1.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return packet


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_rigid_and_articulated_packets_use_the_same_bundle_contract(
    tmp_path: Path, scene_id: str
) -> None:
    packet = _packet(tmp_path, scene_id=scene_id)
    worker = tmp_path / f"worker-{scene_id}.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")
    module = tmp_path / "runtime_helper.py"
    module.write_text("HELPER = 1\n", encoding="utf-8")

    receipt = build_native_task_arena_bundle(
        job_dir=tmp_path / f"job-{scene_id}",
        packet_dir=packet,
        worker_source=worker,
        runtime_module_sources=[module],
        implementation_commit="a" * 40,
        generated_at="fixed",
    )

    assert receipt["status"] == "ready"
    assert receipt["scene_reconstructed_by_bundle"] is False
    assert receipt["packet_receipt_digest"].startswith("sha256:")
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert (
            "provider_runtime/native_task_packet/assets/task_object.usd" in names
        )
        assert "provider_runtime/blueprint_pipeline/runtime_helper.py" in names
        assert archive.read(
            "provider_runtime/adp_arena_provider_runner.py"
        ) == worker.read_bytes()


def test_bundle_is_deterministic_for_one_sealed_packet(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")
    kwargs = {
        "packet_dir": packet,
        "worker_source": worker,
        "runtime_module_sources": [],
        "implementation_commit": "b" * 40,
        "generated_at": "fixed",
    }
    first = build_native_task_arena_bundle(job_dir=tmp_path / "first", **kwargs)
    second = build_native_task_arena_bundle(job_dir=tmp_path / "second", **kwargs)

    assert first["bundle_sha256"] == second["bundle_sha256"]


def test_packet_asset_tamper_fails_before_bundle_creation(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    (packet / "assets/task_object.usd").write_text("tampered\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(NativeTaskArenaBundleError) as excinfo:
        build_native_task_arena_bundle(
            job_dir=tmp_path / "job",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="c" * 40,
        )

    assert any(
        error.startswith("native_task_arena_bundle_packet_asset_identity_mismatch")
        for error in excinfo.value.errors
    )


def test_policy_mode_requires_an_exact_candidate_binding(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(NativeTaskArenaBundleError) as excinfo:
        build_native_task_arena_bundle(
            job_dir=tmp_path / "job",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="d" * 40,
            execution_mode="policy",
        )

    assert excinfo.value.errors == (
        "native_task_arena_bundle_policy_binding_invalid",
    )
