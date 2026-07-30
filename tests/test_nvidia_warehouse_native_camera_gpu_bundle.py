from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_admission import (
    validate_native_camera_gpu_output_archive,
)
from blueprint_pipeline.nvidia_warehouse_native_camera_gpu_bundle import (
    INPUT_SECRET_URL_ENV,
    INPUT_SHA256_ENV,
    OUTPUT_SECRET_PUT_URL_ENV,
    build_native_camera_gpu_bundle,
    extract_native_camera_gpu_bundle,
    require_clean_bundle_source_checkout,
    run_native_camera_gpu_worker,
)
from blueprint_pipeline.nvidia_warehouse_workcell import (
    CANARY_SPEC_SCHEMA_VERSION,
    DATASET_REVISION,
    SCHEMA_VERSION as MATERIALIZATION_SCHEMA_VERSION,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256, file_sha256


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    assets = tmp_path / "assets"
    asset = assets / "Props" / "workcell.usd"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"usd-fixture")
    materialization = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "status": "completed",
        "dataset_revision": DATASET_REVISION,
        "dataset_local_dependency_closure_complete": True,
        "files": [
            {
                "relative_path": "Props/workcell.usd",
                "path": str(asset),
                "sha256": file_sha256(asset),
                "size_bytes": asset.stat().st_size,
            }
        ],
    }
    materialization["manifest_sha256"] = canonical_sha256(materialization)
    materialization_path = assets / "materialization_manifest.json"
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
    spec = {
        "schema_version": CANARY_SPEC_SCHEMA_VERSION,
        "dataset_revision": DATASET_REVISION,
        "materialization_manifest_sha256": materialization["manifest_sha256"],
        "label_free": True,
        "rankings_or_policy_outcomes_accessed": False,
        "paid_gpu_execution_admitted": False,
        "cameras": {
            "external": {"resolution": [640, 480]},
            "wrist": {"resolution": [640, 480]},
        },
    }
    spec["spec_sha256"] = canonical_sha256(spec)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return assets, spec_path


def test_gpu_bundle_cli_source_identity_requires_exact_clean_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "tracked.txt").write_text("frozen", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Blueprint Test",
            "-c",
            "user.email=blueprint@example.invalid",
            "commit",
            "-q",
            "-m",
            "freeze",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert require_clean_bundle_source_checkout(source_commit=head, repo_root=repo) == head
    with pytest.raises(ValueError, match="source_commit_not_checkout_head"):
        require_clean_bundle_source_checkout(source_commit="a" * 40, repo_root=repo)
    (repo / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="source_checkout_not_clean"):
        require_clean_bundle_source_checkout(source_commit=head, repo_root=repo)


def test_gpu_bundle_round_trip_binds_source_spec_manifest_and_assets(tmp_path: Path) -> None:
    assets, spec = _inputs(tmp_path)
    bundle = tmp_path / "camera.zip"
    receipt_path = tmp_path / "receipt.json"
    receipt = build_native_camera_gpu_bundle(
        assets_root=assets,
        spec_path=spec,
        source_commit="a" * 40,
        output_zip=bundle,
        receipt_path=receipt_path,
    )

    extracted = extract_native_camera_gpu_bundle(
        bundle_path=bundle,
        expected_sha256=receipt["bundle_sha256"],
        output_dir=tmp_path / "extracted",
    )

    assert receipt["manifest"]["source_commit"] == "a" * 40
    assert receipt["manifest"]["asset_count"] == 1
    assert Path(extracted["assets_root"], "Props/workcell.usd").read_bytes() == b"usd-fixture"
    assert receipt_path.is_file()


def test_gpu_bundle_rejects_asset_mutation_before_packaging(tmp_path: Path) -> None:
    assets, spec = _inputs(tmp_path)
    (assets / "Props" / "workcell.usd").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="asset_invalid"):
        build_native_camera_gpu_bundle(
            assets_root=assets,
            spec_path=spec,
            source_commit="a" * 40,
            output_zip=tmp_path / "camera.zip",
        )


def test_gpu_bundle_extraction_rejects_member_tampering(tmp_path: Path) -> None:
    assets, spec = _inputs(tmp_path)
    bundle = tmp_path / "camera.zip"
    build_native_camera_gpu_bundle(
        assets_root=assets,
        spec_path=spec,
        source_commit="a" * 40,
        output_zip=bundle,
    )
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(tampered, "w") as target:
        for info in source.infolist():
            data = source.read(info)
            if info.filename == "assets/Props/workcell.usd":
                data = b"tampered"
            target.writestr(info.filename, data)

    with pytest.raises(ValueError, match="asset_sha_invalid"):
        extract_native_camera_gpu_bundle(
            bundle_path=tampered,
            expected_sha256=file_sha256(tampered),
            output_dir=tmp_path / "extracted",
        )


def test_gpu_worker_downloads_runs_and_uploads_without_recording_urls(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"hash-bound-input")
    expected = file_sha256(source)
    uploaded: dict[str, bytes] = {}

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(source.read_bytes())

    def runner(**kwargs):
        output = Path(kwargs["workspace"]) / "output"
        output.mkdir(parents=True)
        (output / "native_camera_canary_result.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )
        (output / "external_initial.png").write_bytes(b"png-evidence")
        return {"status": "passed"}

    def uploader(url: str, path: Path) -> None:
        uploaded[url] = path.read_bytes()

    input_url = "https://storage.example/input?secret=value"
    output_url = "https://storage.example/output?secret=value"
    receipt = run_native_camera_gpu_worker(
        workspace=tmp_path / "worker",
        environment={
            INPUT_SECRET_URL_ENV: input_url,
            INPUT_SHA256_ENV: expected,
            OUTPUT_SECRET_PUT_URL_ENV: output_url,
        },
        downloader=downloader,
        uploader=uploader,
        bundle_runner=runner,
    )

    assert receipt["status"] == "completed"
    assert receipt["input_url_recorded"] is False
    assert receipt["output_url_recorded"] is False
    assert output_url in uploaded
    assert input_url not in json.dumps(receipt)
    assert output_url not in json.dumps(receipt)


def test_gpu_worker_uploads_hash_bound_failure_media_and_exits_transport_cleanly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"hash-bound-input")
    expected = file_sha256(source)
    uploaded: dict[str, bytes] = {}

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(source.read_bytes())

    def runner(**_kwargs):
        raise TypeError("live Isaac SimulationApp shim was not callable")

    def uploader(url: str, path: Path) -> None:
        uploaded[url] = path.read_bytes()

    output_url = "https://storage.example/output?secret=value"
    receipt = run_native_camera_gpu_worker(
        workspace=tmp_path / "worker",
        environment={
            INPUT_SECRET_URL_ENV: "https://storage.example/input?secret=value",
            INPUT_SHA256_ENV: expected,
            OUTPUT_SECRET_PUT_URL_ENV: output_url,
        },
        downloader=downloader,
        uploader=uploader,
        bundle_runner=runner,
    )

    validation = validate_native_camera_gpu_output_archive(uploaded[output_url])
    assert receipt["status"] == "completed"
    assert receipt["canary_status"] == "failed"
    assert validation["status"] == "completed"
    assert validation["canary_status"] == "failed"
    assert validation["canary_result"]["failure_evidence"]["error_type"] == "TypeError"
    assert validation["canary_result"]["claim_boundary"]["policy_wam_loop_proven"] is False


def test_gpu_worker_records_safe_missing_module_without_exception_message(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"hash-bound-input")
    uploaded: dict[str, bytes] = {}

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(source.read_bytes())

    def runner(**_kwargs):
        raise ModuleNotFoundError("signed-url-secret-must-not-leak", name="isaacsim.core.api")

    def uploader(url: str, path: Path) -> None:
        uploaded[url] = path.read_bytes()

    output_url = "https://storage.example/output?secret=value"
    run_native_camera_gpu_worker(
        workspace=tmp_path / "worker",
        environment={
            INPUT_SECRET_URL_ENV: "https://storage.example/input?secret=value",
            INPUT_SHA256_ENV: file_sha256(source),
            OUTPUT_SECRET_PUT_URL_ENV: output_url,
        },
        downloader=downloader,
        uploader=uploader,
        bundle_runner=runner,
    )
    validation = validate_native_camera_gpu_output_archive(uploaded[output_url])
    evidence = validation["canary_result"]["failure_evidence"]
    assert evidence["error_type"] == "ModuleNotFoundError"
    assert evidence["missing_module"] == "isaacsim.core.api"
    assert "signed-url-secret" not in json.dumps(validation)


def test_gpu_worker_rejects_non_https_signed_urls_before_download(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_url_invalid"):
        run_native_camera_gpu_worker(
            workspace=tmp_path / "worker",
            environment={
                INPUT_SECRET_URL_ENV: "file:///tmp/input.zip",
                INPUT_SHA256_ENV: "a" * 64,
                OUTPUT_SECRET_PUT_URL_ENV: "https://storage.example/output",
            },
        )
