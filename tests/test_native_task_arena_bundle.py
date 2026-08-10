from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.adp009d_native_microcheck_bundle import (
    DEFAULT_IMAGE as QUALIFIED_ADP_IMAGE,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_bundle import (
    NativeTaskArenaBundleError,
    build_native_task_arena_bundle,
)
from blueprint_pipeline.native_task_arena_construction_bundle import (
    CONSTRUCTION_RUNTIME_MODULE_NAMES,
    PROBE_KIND,
    build_native_task_arena_construction_bundle,
    load_verified_native_task_arena_construction_bundle,
)
from blueprint_pipeline.native_task_arena_vast import run_native_task_arena_vast
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
    _resolve_probe_image,
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


@pytest.mark.parametrize("scene_id", ["840313", "840796"])
def test_construction_bundle_has_one_scene_neutral_import_closure(
    tmp_path: Path, scene_id: str
) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / f"construction-{scene_id}",
        packet_dir=_packet(tmp_path, scene_id=scene_id),
        implementation_commit="e" * 40,
        generated_at="fixed",
    )
    assert receipt["container_image"] == QUALIFIED_ADP_IMAGE
    extracted = tmp_path / f"extracted-{scene_id}"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        archive.extractall(extracted)
    package = extracted / "provider_runtime/blueprint_pipeline"
    expected = {
        f"provider_runtime/blueprint_pipeline/{name}"
        for name in CONSTRUCTION_RUNTIME_MODULE_NAMES
    }
    assert expected.issubset(names)
    assert "provider_runtime/blueprint_pipeline/native_task_arena_scene_plan.py" not in names
    assert "provider_runtime/blueprint_pipeline/adp009d_approach_capture.py" not in names
    assert not any(
        name.startswith("provider_runtime/blueprint_pipeline/adp009d")
        for name in names
    )

    modules = [Path(name).stem for name in CONSTRUCTION_RUNTIME_MODULE_NAMES]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib,sys;"
                f"sys.path.insert(0,{str(package.parent)!r});"
                f"[importlib.import_module('blueprint_pipeline.'+name) for name in {modules!r}]"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    worker = (extracted / "provider_runtime/adp_arena_provider_runner.py").read_text()
    assert "840313" not in worker
    assert "840796" not in worker
    assert "BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena" in worker


def test_construction_bundle_passes_native_vast_static_preflight(tmp_path: Path) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "bundle",
        packet_dir=_packet(tmp_path, scene_id="840796"),
        implementation_commit="f" * 40,
        generated_at="fixed",
    )
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    assert (
        _resolve_probe_image(
            public_image="public",
            isaac_image="isaac",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="native_task_arena",
        )
        == "isaac"
    )
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=True,
            enable_blueprint_bundle=True,
            provider_bundle_kind="native_task_arena",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena",
        forward_hf_token=False,
    )
    assert env["ACCEPT_EULA"] == "Y"
    assert "run_adp_arena_provider_runtime.sh" in _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="native_task_arena",
    )

    dry_run = run_native_task_arena_vast(
        job_dir=tmp_path / "dry-run",
        prepared_bundle=receipt,
        paid_resource_admission_grant=None,
        execute=False,
    )
    assert dry_run["status"] == "dry_run_ready"
    assert dry_run["provider_mutations_performed"] == 0


def test_bundle_rejects_an_unpinned_runtime_image(tmp_path: Path) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    worker = tmp_path / "worker.py"
    worker.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(NativeTaskArenaBundleError) as excinfo:
        build_native_task_arena_bundle(
            job_dir=tmp_path / "job",
            packet_dir=packet,
            worker_source=worker,
            runtime_module_sources=[],
            implementation_commit="f" * 40,
            container_image="nvcr.io/nvidia/isaac-sim:latest",
        )
    assert excinfo.value.errors == (
        "native_task_arena_bundle_container_image_not_digest_pinned",
    )


def test_dry_run_bundle_receipt_reloads_exact_bytes_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    receipt = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "bundle",
        packet_dir=_packet(tmp_path, scene_id="840796"),
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    receipt_path = tmp_path / "bundle/native_task_arena_provider_bundle_receipt.v1.json"
    loaded = load_verified_native_task_arena_construction_bundle(
        receipt_path,
        expected_implementation_commit="a" * 40,
        expected_packet_receipt_digest=receipt["packet_receipt_digest"],
    )
    assert loaded["bundle_sha256"] == receipt["bundle_sha256"]

    Path(receipt["bundle_path"]).write_bytes(
        Path(receipt["bundle_path"]).read_bytes() + b"tamper"
    )
    with pytest.raises(ValueError, match="native_task_arena_bundle_bytes_identity_mismatch"):
        load_verified_native_task_arena_construction_bundle(
            receipt_path,
            expected_implementation_commit="a" * 40,
        )


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_routes_sealed_native_task_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    packet = _packet(tmp_path, scene_id="840796")
    frozen_bundle = build_native_task_arena_construction_bundle(
        job_dir=tmp_path / "frozen-bundle",
        packet_dir=packet,
        implementation_commit="a" * 40,
        generated_at="fixed",
    )
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_native_task_arena_vast", fake_run)
    args = [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "native-task-arena",
        "--native-task-arena-packet",
        str(packet),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "0.8",
        "--adp-max-spend-usd",
        "1.0",
        "--adp-hard-ttl-seconds",
        "5400",
    ]
    if execute:
        args.extend(
            [
                "--native-task-arena-bundle-receipt",
                str(
                    tmp_path
                    / "frozen-bundle/native_task_arena_provider_bundle_receipt.v1.json"
                ),
                "--execute",
            ]
        )

    assert allocator.main(args) == 0
    assert observed["execute"] is execute
    assert isinstance(
        observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant
    ) is execute
    if execute:
        assert (
            observed["prepared_bundle"]["bundle_sha256"]
            == frozen_bundle["bundle_sha256"]
        )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["private_data_uploaded"] is True
    assert admission["raw_dataset_bytes_uploaded"] is False
    assert admission["retry_cap"] == 0
    assert admission["allocation_binding"]["packet_receipt_digest"].startswith(
        "sha256:"
    )
