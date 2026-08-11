from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import native_deformable_asset_vast as deformable_vast
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.native_deformable_asset_preparation import (
    PACKAGE_RECEIPT_FILENAME,
    PLAN_FILENAME,
    build_native_deformable_asset_source_package,
    materialize_native_deformable_asset_preparation_plan,
)
from blueprint_pipeline.native_deformable_asset_provider_bundle import (
    NativeDeformableAssetProviderBundleError,
    build_native_deformable_asset_provider_bundle,
    load_verified_native_deformable_asset_provider_bundle,
)
from blueprint_pipeline.native_deformable_asset_vast import (
    run_native_deformable_asset_vast,
)
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    from tests.test_native_deformable_asset_preparation import _physics, _source_fixture

    source, textures, inspection_receipt_path, inspection = _source_fixture(tmp_path)
    plan = materialize_native_deformable_asset_preparation_plan(
        preparation_id="fixture",
        inspection_receipt_path=inspection_receipt_path,
        expected_inspection_receipt_digest=inspection["receipt_digest"],
        source_usd_path=source,
        source_texture_root=textures,
        target_metric_dimensions_m=inspection["observation"]["dimensions_m"],
        physics_configuration=_physics(),
    )
    package = tmp_path / "source_package"
    build_native_deformable_asset_source_package(
        output_dir=package,
        plan=plan,
        expected_plan_digest=plan["plan_digest"],
    )
    receipt_path = package / PACKAGE_RECEIPT_FILENAME
    packet = tmp_path / "runtime_sources.zip"
    packet.write_bytes(b"runtime-source-packet")
    runtime_receipt_path = tmp_path / "native_task_runtime_source_packet.v1.json"
    runtime_receipt_path.write_text("{}\n")
    runtime = {
        "receipt_digest": "sha256:" + "3" * 64,
        "packet_sha256": _sha(packet),
        "verified_packet_path": str(packet),
        "redistribution_permitted": True,
        "runtime_dependency_wheels": [{"package": "fixture", "sha256": _sha(packet)}],
    }
    return receipt_path, runtime_receipt_path, runtime


def test_builds_replayable_bundle_with_exact_native_runtime_contract(
    tmp_path: Path,
) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    receipt = build_native_deformable_asset_provider_bundle(
        job_dir=tmp_path / "bundle",
        source_package_receipt_path=source,
        runtime_source_packet_receipt_path=runtime_path,
        implementation_commit="a" * 40,
        package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
        container_image="registry.example/isaac@sha256:" + "b" * 64,
        runtime_source_packet_verifier=lambda _: runtime,
    )
    assert receipt["status"] == "ready"
    assert receipt["candidate_policy_queried"] is False
    assert receipt["native_cook_qualified"] is False
    assert receipt["one_instance_at_a_time"] is True
    assert receipt["maximum_concurrent_paid_instances"] == 1
    assert receipt["allowed_active_vast_instance_ids"] == []
    assert receipt["provider_zero_scope"] == "this_run_owned_instances"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        assert "provider_runtime/input_package/source/asset.usd" in names
        assert "provider_runtime/run_adp_arena_provider_runtime.sh" in names
        assert "provider_runtime/blueprint_pipeline/core/__init__.py" in names
        assert "provider_runtime/blueprint_pipeline/core/common.py" in names
        entrypoint = archive.read(
            "provider_runtime/run_native_deformable_asset_provider_runtime.sh"
        ).decode()
        runner = archive.read(
            "provider_runtime/blueprint_pipeline/native_deformable_asset_preparation_worker.py"
        ).decode()
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="native_deformable_asset",
            entrypoint_text=entrypoint,
            runner_text=runner,
        )
        == []
    )
    assert (
        '. "$RUNTIME_DIR/provisioned_runtime_sources/native_task_runtime_environment.sh"'
        in entrypoint
    )
    assert "BLUEPRINT_WAM_RUNTIME_PHASE:native_deformable_asset:worker:started" in entrypoint
    assert entrypoint.index("native_task_runtime_environment.sh") < entrypoint.index(
        "native_deformable_asset_preparation_worker"
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(extracted)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import pathlib,sys;"
                f"sys.path.insert(0,{str(extracted / 'provider_runtime')!r});"
                "import blueprint_pipeline.common as common;"
                "assert common.utc_now_iso"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    replay = load_verified_native_deformable_asset_provider_bundle(
        Path(receipt["bundle_path"]).parent
        / "native_deformable_asset_provider_bundle_receipt.v1.json",
        expected_implementation_commit="a" * 40,
        expected_source_package_receipt_digest=receipt["source_package_receipt_digest"],
        expected_runtime_source_packet_receipt_digest=runtime["receipt_digest"],
    )
    assert replay == receipt
    dry = run_native_deformable_asset_vast(
        job_dir=tmp_path / "vast_dry",
        prepared_bundle=receipt,
        paid_resource_admission_grant=None,
        execute=False,
    )
    assert dry["status"] == "dry_run_ready"
    assert dry["provider_mutations_performed"] == 0


def test_bundle_binds_one_exact_concurrent_sibling_before_paid_execution(
    tmp_path: Path,
) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    receipt = build_native_deformable_asset_provider_bundle(
        job_dir=tmp_path / "concurrent-bundle",
        source_package_receipt_path=source,
        runtime_source_packet_receipt_path=runtime_path,
        implementation_commit="a" * 40,
        package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
        container_image="registry.example/isaac@sha256:" + "b" * 64,
        allowed_active_instance_ids=(47373597,),
        runtime_source_packet_verifier=lambda _: runtime,
    )

    assert receipt["one_instance_at_a_time"] is False
    assert receipt["maximum_concurrent_paid_instances"] == 2
    assert receipt["allowed_active_vast_instance_ids"] == [47373597]
    replay = load_verified_native_deformable_asset_provider_bundle(
        Path(receipt["bundle_path"]).parent
        / "native_deformable_asset_provider_bundle_receipt.v1.json",
        expected_implementation_commit="a" * 40,
        expected_source_package_receipt_digest=receipt["source_package_receipt_digest"],
        expected_runtime_source_packet_receipt_digest=runtime["receipt_digest"],
        expected_allowed_active_instance_ids=(47373597,),
    )
    assert replay == receipt
    with pytest.raises(
        NativeDeformableAssetProviderBundleError,
        match="native_deformable_provider_bundle_receipt_invalid",
    ):
        load_verified_native_deformable_asset_provider_bundle(
            Path(receipt["bundle_path"]).parent
            / "native_deformable_asset_provider_bundle_receipt.v1.json",
            expected_implementation_commit="a" * 40,
            expected_source_package_receipt_digest=receipt["source_package_receipt_digest"],
            expected_runtime_source_packet_receipt_digest=runtime["receipt_digest"],
        )


def test_native_vast_transport_requires_the_frozen_sibling_to_still_be_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    receipt = build_native_deformable_asset_provider_bundle(
        job_dir=tmp_path / "concurrent-bundle",
        source_package_receipt_path=source,
        runtime_source_packet_receipt_path=runtime_path,
        implementation_commit="a" * 40,
        package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
        container_image="registry.example/isaac@sha256:" + "b" * 64,
        allowed_active_instance_ids=(47373597,),
        runtime_source_packet_verifier=lambda _: runtime,
    )
    observed: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(deformable_vast, "run_arena_native_control_vast", fake_run)
    assert (
        run_native_deformable_asset_vast(
            job_dir=tmp_path / "vast",
            prepared_bundle=receipt,
            paid_resource_admission_grant=None,
            execute=False,
            allowed_active_instance_ids=(47373597,),
        )["status"]
        == "dry_run_ready"
    )
    assert observed["allowed_active_instance_ids"] == (47373597,)
    assert observed["required_active_instance_ids"] == (47373597,)
    assert "vast_launch_lock_file" not in observed
    with pytest.raises(
        ValueError, match="native_deformable_asset_prepared_bundle_contract_invalid"
    ):
        run_native_deformable_asset_vast(
            job_dir=tmp_path / "mismatch",
            prepared_bundle=receipt,
            paid_resource_admission_grant=None,
            execute=False,
        )


def test_embedded_runtime_result_writer_is_valid_python(tmp_path: Path) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    receipt = build_native_deformable_asset_provider_bundle(
        job_dir=tmp_path / "bundle",
        source_package_receipt_path=source,
        runtime_source_packet_receipt_path=runtime_path,
        implementation_commit="a" * 40,
        package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
        container_image="registry.example/isaac@sha256:" + "b" * 64,
        runtime_source_packet_verifier=lambda _: runtime,
    )
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_native_deformable_asset_provider_runtime.sh"
        ).decode()

    embedded_python = entrypoint.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    ast.parse(embedded_python)
    assert '+ "\\n")' in embedded_python


def test_embedded_runtime_result_writer_retains_worker_failure_logs(
    tmp_path: Path,
) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    receipt = build_native_deformable_asset_provider_bundle(
        job_dir=tmp_path / "bundle",
        source_package_receipt_path=source,
        runtime_source_packet_receipt_path=runtime_path,
        implementation_commit="a" * 40,
        package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
        container_image="registry.example/isaac@sha256:" + "b" * 64,
        runtime_source_packet_verifier=lambda _: runtime,
    )
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_native_deformable_asset_provider_runtime.sh"
        ).decode()

    embedded_python = entrypoint.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    out = tmp_path / "runtime_output"
    out.mkdir()
    (out / "native_deformable_asset_preparation_worker_stdout.log").write_text(
        "worker stdout before crash\n"
    )
    (out / "native_deformable_asset_preparation_worker_stderr.log").write_text(
        "Traceback: fixture native failure\n"
    )
    completed = subprocess.run(
        [sys.executable, "-", str(out), "0", "1"],
        input=embedded_python,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    terminal = json.loads(
        (out / "native_deformable_asset_preparation_worker_terminal.v1.json").read_text()
    )
    assert terminal["status"] == "blocked"
    assert terminal["errors"] == ["native_deformable_asset_worker_failed_without_terminal_receipt"]
    assert terminal["worker_logs"]["stderr"]["present"] is True
    assert "fixture native failure" in terminal["worker_logs"]["stderr"]["tail"]
    result = json.loads((out / "native_deformable_asset_vast_execution.v1.json").read_text())
    assert result["status"] == "blocked"
    assert result["blockers"] == ["native_deformable_asset_worker_failed_without_terminal_receipt"]
    assert result["worker_returncode"] == 1
    assert result["worker_logs"]["stdout"]["sha256"].startswith("sha256:")


def test_source_package_tamper_fails_before_bundle_materialization(tmp_path: Path) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    (source.parent / "source" / "asset.usd").write_bytes(b"tampered")
    with pytest.raises(
        NativeDeformableAssetProviderBundleError,
        match="source_file_identity_mismatch",
    ):
        build_native_deformable_asset_provider_bundle(
            job_dir=tmp_path / "bundle",
            source_package_receipt_path=source,
            runtime_source_packet_receipt_path=runtime_path,
            implementation_commit="a" * 40,
            package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
            container_image="registry.example/isaac@sha256:" + "b" * 64,
            runtime_source_packet_verifier=lambda _: runtime,
        )


def test_dependency_free_runtime_source_requires_matching_complete_runtime_image(
    tmp_path: Path,
) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    runtime = {
        **runtime,
        "runtime_dependency_wheels": [],
        "paired_stack": {
            "simulator_runtime_image": "registry.example/complete-isaac-lab@sha256:" + "c" * 64
        },
    }

    with pytest.raises(
        NativeDeformableAssetProviderBundleError,
        match="native_deformable_provider_runtime_source_container_mismatch",
    ):
        build_native_deformable_asset_provider_bundle(
            job_dir=tmp_path / "bundle",
            source_package_receipt_path=source,
            runtime_source_packet_receipt_path=runtime_path,
            implementation_commit="a" * 40,
            package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
            container_image="registry.example/isaac-sim@sha256:" + "b" * 64,
            runtime_source_packet_verifier=lambda _: runtime,
        )


def test_rehashed_bundle_receipt_cannot_override_embedded_manifest(tmp_path: Path) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    receipt = build_native_deformable_asset_provider_bundle(
        job_dir=tmp_path / "bundle",
        source_package_receipt_path=source,
        runtime_source_packet_receipt_path=runtime_path,
        implementation_commit="a" * 40,
        package_source_root=Path(__file__).parents[1] / "src" / "blueprint_pipeline",
        container_image="registry.example/isaac@sha256:" + "b" * 64,
        runtime_source_packet_verifier=lambda _: runtime,
    )
    receipt_path = (
        Path(receipt["bundle_path"]).parent
        / "native_deformable_asset_provider_bundle_receipt.v1.json"
    )
    forged = json.loads(receipt_path.read_text())
    forged["source_package_receipt_digest"] = "sha256:" + "9" * 64
    forged["input_digest"] = canonical_digest(
        {
            "source_package_receipt_digest": forged["source_package_receipt_digest"],
            "runtime_source_packet_receipt_digest": runtime["receipt_digest"],
            "implementation_commit": "a" * 40,
            "container_image": forged["container_image"],
        }
    )
    forged["receipt_digest"] = canonical_digest(forged, digest_field="receipt_digest")
    receipt_path.write_text(canonical_json(forged) + "\n")
    with pytest.raises(
        NativeDeformableAssetProviderBundleError,
        match="native_deformable_provider_bundle_manifest_invalid",
    ):
        load_verified_native_deformable_asset_provider_bundle(
            receipt_path,
            expected_implementation_commit="a" * 40,
            expected_source_package_receipt_digest=forged["source_package_receipt_digest"],
            expected_runtime_source_packet_receipt_digest=runtime["receipt_digest"],
        )


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
def test_canonical_allocator_routes_owner_authorized_deformable_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool, concurrent: bool
) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    plan = json.loads(
        (source.parent / PLAN_FILENAME).read_text()
    )
    rights = {
        "schema_version": "task_evaluation_prelaunch_abstention_supersession.v1",
        "asset_archive_sha256": plan["source_asset"]["source_archive_sha256"],
        "private_upload_to_vast_permitted": True,
        "public_redistribution_permitted": False,
        "rights_gate_passed_for_private_vast_canary": True,
        "receipt_digest": "",
    }
    rights["receipt_digest"] = canonical_digest(rights, digest_field="receipt_digest")
    rights_path = tmp_path / "rights.json"
    rights_path.write_text(canonical_json(rights) + "\n")
    prepared = {
        "schema_version": "native_deformable_asset_provider_bundle.v2",
        "status": "ready",
        "provider_bundle_kind": "native_deformable_asset",
        "execution_mode": "asset_preparation_canary",
        "expected_output_filename": "native_deformable_asset_vast_execution.v1.json",
        "candidate_policy_queried": False,
        "native_cook_qualified": False,
        "one_instance_at_a_time": not concurrent,
        "maximum_concurrent_paid_instances": 2 if concurrent else 1,
        "allowed_active_vast_instance_ids": [47373597] if concurrent else [],
        "provider_zero_scope": "this_run_owned_instances",
        "container_image": "registry.example/isaac@sha256:" + "b" * 64,
        "bundle_sha256": "sha256:" + "5" * 64,
        "input_digest": "sha256:" + "6" * 64,
    }
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(allocator, "verify_native_task_runtime_source_packet", lambda _: runtime)
    monkeypatch.setattr(
        allocator,
        "build_native_deformable_asset_provider_bundle",
        lambda **_: prepared,
    )
    monkeypatch.setattr(
        allocator,
        "load_verified_native_deformable_asset_provider_bundle",
        lambda *_, **__: prepared,
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_native_deformable_asset_vast", fake_run)
    args = [
        "gpu-canary",
        "--probe-kind",
        "native-deformable-asset-preparation",
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
        str(tmp_path / "bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "native-deformable-asset",
        "--native-deformable-source-package-receipt",
        str(source),
        "--native-task-arena-runtime-source-packet",
        str(runtime_path),
        "--native-deformable-rights-supersession",
        str(rights_path),
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
        bundle_receipt = tmp_path / "bundle-receipt.json"
        bundle_receipt.write_text("{}\n")
        args.extend(["--native-deformable-bundle-receipt", str(bundle_receipt), "--execute"])
    if concurrent:
        args.extend(["--adp-allowed-active-vast-instance-id", "47373597"])
    assert allocator.main(args) == 0
    assert observed["execute"] is execute
    assert (
        isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["authority"] == ("direct_asset_owner_private_vast_processing_authority")
    assert admission["raw_dataset_bytes_uploaded"] is False
    assert admission["retry_cap"] == 0
    assert admission["explicit_concurrent_gpu_authority_bound"] is concurrent
    assert observed["allowed_active_instance_ids"] == ((47373597,) if concurrent else ())


def test_canonical_allocator_reports_typed_deformable_rights_binding_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, runtime_path, runtime = _fixture(tmp_path)
    bad_rights = {
        "schema_version": "external_asset_owner_processing_authority.v1",
        "receipt_digest": "sha256:" + "a" * 64,
    }
    rights_path = tmp_path / "bad-rights.json"
    rights_path.write_text(canonical_json(bad_rights) + "\n")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(allocator, "verify_native_task_runtime_source_packet", lambda _: runtime)

    assert (
        allocator.main(
            [
                "gpu-canary",
                "--probe-kind",
                "native-deformable-asset-preparation",
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
                str(tmp_path / "bound.json"),
                "--adapter-output",
                str(tmp_path / "adapter.json"),
                "--pod-name",
                "native-deformable-asset",
                "--native-deformable-source-package-receipt",
                str(source),
                "--native-task-arena-runtime-source-packet",
                str(runtime_path),
                "--native-deformable-rights-supersession",
                str(rights_path),
                "--adp-job-dir",
                str(tmp_path / "job"),
            ]
        )
        == 2
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["blockers"] == ["native_deformable_rights_binding_invalid"]
