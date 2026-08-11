from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import adp_aura_author_smoke_vast as aura
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
)
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


def _load_provider_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/adp_aura_author_smoke_provider_runner.py"
    )
    spec = importlib.util.spec_from_file_location("adp_aura_provider_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    real_root = Path(__file__).resolve().parents[1]
    for name in (
        "run_adp_aura_author_smoke_provider_runtime.sh",
        "adp_aura_author_smoke_provider_runner.py",
    ):
        (scripts / name).write_bytes((real_root / "scripts" / name).read_bytes())
    source = tmp_path / "AuraFusion360_official"
    sam2 = tmp_path / "sam2"
    wonderworld = tmp_path / "WonderWorld"
    (source / "submodules/diff-surfel-rasterization/third_party/glm").mkdir(parents=True)
    (source / "submodules/simple-knn").mkdir(parents=True)
    (source / "README.md").write_text("author source", encoding="utf-8")
    (source / "submodules/diff-surfel-rasterization/LICENSE.md").write_text(
        "license", encoding="utf-8"
    )
    (source / "submodules/diff-surfel-rasterization/third_party/glm/copying.txt").write_text(
        "license", encoding="utf-8"
    )
    (source / "submodules/simple-knn/LICENSE.md").write_text("license", encoding="utf-8")
    sam2.mkdir()
    (sam2 / "LICENSE").write_text("sam2 license", encoding="utf-8")
    (wonderworld / "marigold_module").mkdir(parents=True)
    (wonderworld / "marigold_module/LICENSE.txt").write_text(
        "marigold license", encoding="utf-8"
    )
    for source_path in aura.WONDERWORLD_MARIGOLD_RUNTIME_FILES.values():
        path = wonderworld / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    author_data = tmp_path / "author-data"
    author_file = author_data / "360-USID/sunflower/input.txt"
    author_file.parent.mkdir(parents=True)
    author_file.write_text("publisher author scene", encoding="utf-8")
    author_publisher_files = [
        {
            "path": "360-USID/sunflower/input.txt",
            "size_bytes": author_file.stat().st_size,
            "lfs_sha256": aura._sha256(author_file).removeprefix("sha256:"),
            "git_blob_id": "c" * 40,
        }
    ]
    author_snapshot_digest = canonical_digest({"files": author_publisher_files})
    monkeypatch.setattr(
        aura,
        "_AUTHOR_DATA",
        {
            **aura._AUTHOR_DATA,
            "snapshot_digest": author_snapshot_digest,
        },
    )

    def fake_git(path: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "HEAD^{tree}"):
            if path == sam2:
                return aura.SAM2_SOURCE_TREE
            if path == wonderworld:
                return aura.WONDERWORLD_SOURCE_TREE
            return aura.SOURCE_TREE
        if args == ("rev-parse", "HEAD"):
            if path == sam2:
                return aura.SAM2_SOURCE_COMMIT
            if path == wonderworld:
                return aura.WONDERWORLD_SOURCE_COMMIT
            for relative, revision in aura.SUBMODULES.items():
                if path == source / relative:
                    return revision
            return aura.SOURCE_COMMIT
        raise AssertionError((path, args))

    monkeypatch.setattr(aura, "_git", fake_git)
    monkeypatch.setattr(
        aura,
        "_source_files",
        lambda _source: [
            ("README.md", source / "README.md"),
            (
                "submodules/diff-surfel-rasterization/LICENSE.md",
                source / "submodules/diff-surfel-rasterization/LICENSE.md",
            ),
            (
                "submodules/diff-surfel-rasterization/third_party/glm/copying.txt",
                source / "submodules/diff-surfel-rasterization/third_party/glm/copying.txt",
            ),
            ("submodules/simple-knn/LICENSE.md", source / "submodules/simple-knn/LICENSE.md"),
        ],
    )
    monkeypatch.setattr(aura, "SAM2_LICENSE_SHA256", aura._sha256(sam2 / "LICENSE"))
    monkeypatch.setattr(
        aura,
        "WONDERWORLD_MARIGOLD_LICENSE_SHA256",
        aura._sha256(wonderworld / "marigold_module/LICENSE.txt"),
    )
    original_tracked_files = aura._tracked_files
    monkeypatch.setattr(
        aura,
        "_tracked_files",
        lambda repo, prefix="": (
            [("LICENSE", sam2 / "LICENSE")]
            if repo == sam2
            else original_tracked_files(repo, prefix)
        ),
    )
    snapshots = [
        {"artifact_id": artifact_id, "rights_established": True}
        for artifact_id in (
            "aurafusion360_sunflower_author_scene",
            "aurafusion360_sunflower_expected_output",
            "aurafusion360_sam2_hiera_large",
            "aurafusion360_marigold_depth_v1_0",
            "aurafusion360_marigold_agdd_v1_0",
            "aurafusion360_sd2_inpainting_exact_checkpoint",
        )
    ]
    snapshots[0]["publisher"] = {
        "repository": aura._AUTHOR_DATA["repository"],
        "revision": aura._AUTHOR_DATA["revision"],
        "path_prefix": aura._AUTHOR_DATA["path_prefix"],
        "snapshot_digest": author_snapshot_digest,
    }
    snapshots[-1]["publisher"] = {
        "repository": aura._SD2["repository"],
        "revision": aura._SD2["revision"],
        "path_prefix": aura._SD2["path"],
        "single_file_identity": {
            "path": aura._SD2["path"],
            "size_bytes": 5_214_921_607,
            "lfs_sha256": "sha256:" + "a" * 64,
            "git_blob_id": "b" * 40,
        },
    }
    receipt: dict[str, object] = {
        "methods": {
            "aurafusion360_quality_challenger": {
                "checkpoint_rights_established": True,
                "author_data_rights_established": True,
                "remote_snapshots": snapshots,
            }
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    monkeypatch.setattr(aura, "PREREQUISITE_RECEIPT_DIGEST", receipt["receipt_digest"])
    prerequisite = tmp_path / "prerequisite.json"
    _write_json(prerequisite, receipt)
    author_receipt: dict[str, object] = {
        "schema_version": aura.AUTHOR_DATA_RECEIPT_SCHEMA_VERSION,
        "generated_at": "2026-08-04T00:00:00+00:00",
        "status": "completed",
        "publisher": aura._AUTHOR_DATA,
        "prerequisite_receipt_digest": receipt["receipt_digest"],
        "files": [
            {
                **author_publisher_files[0],
                "sha256": aura._sha256(author_file),
            }
        ],
        "file_count": 1,
        "total_size_bytes": author_file.stat().st_size,
        "rights_established": True,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    author_receipt["receipt_digest"] = canonical_digest(
        author_receipt, digest_field="receipt_digest"
    )
    author_receipt_path = tmp_path / "author-data-receipt.json"
    _write_json(author_receipt_path, author_receipt)
    expected_output_ply = tmp_path / "published-expected-point-cloud.ply"
    expected_output_ply.write_bytes(b"publisher expected output")
    monkeypatch.setattr(
        aura,
        "_EXPECTED_OUTPUT",
        {
            **aura._EXPECTED_OUTPUT,
            "expected_ply_size_bytes": expected_output_ply.stat().st_size,
            "expected_ply_sha256": aura._sha256(expected_output_ply),
        },
    )
    return {
        "repo": repo,
        "source": source,
        "sam2": sam2,
        "wonderworld": wonderworld,
        "prerequisite": prerequisite,
        "author_data": author_data,
        "author_data_receipt": author_receipt_path,
        "expected_output_ply": expected_output_ply,
        "job": tmp_path / "job",
    }


def test_bundle_derives_source_and_rights_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        wonderworld_root=paths["wonderworld"],
        prerequisite_receipt_path=paths["prerequisite"],
        author_data_root=paths["author_data"],
        author_data_receipt_path=paths["author_data_receipt"],
        expected_output_ply_path=paths["expected_output_ply"],
        job_dir=paths["job"],
        generated_at="2026-08-04T00:00:00+00:00",
    )
    assert receipt["status"] == "ready"
    assert receipt["retry_cap"] == 0
    assert receipt["depth_anything3_used"] is False
    assert receipt["smoke_scope"] == "unchanged_author_workflow_through_inpaint_init"
    assert receipt["sd2_checkpoint_identity"]["size_bytes"] == 5_214_921_607
    assert receipt["sd2_checkpoint_identity"]["sha256"] == "sha256:" + "a" * 64
    assert receipt["author_data_file_count"] == 1
    assert receipt["author_data_total_size_bytes"] > 0
    assert receipt["wonderworld_source_commit"] == aura.WONDERWORLD_SOURCE_COMMIT
    assert receipt["wonderworld_marigold_runtime_license"] == "Apache-2.0"
    assert Path(receipt["bundle_path"]).is_file()
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive_names = archive.namelist()
        assert "provider_runtime/author_data.zip" in archive_names
        assert "provider_runtime/wonderworld_marigold_runtime.zip" in archive_names
        dependency_archive = tmp_path / "wonderworld-runtime.zip"
        dependency_archive.write_bytes(
            archive.read("provider_runtime/wonderworld_marigold_runtime.zip")
        )
        runner = archive.read(
            "provider_runtime/adp_aura_author_smoke_provider_runner.py"
        ).decode()
        smoke_spec = json.loads(archive.read("provider_runtime/smoke_spec.json"))
        entrypoint = archive.read(
            "provider_runtime/run_adp_aura_author_smoke_provider_runtime.sh"
        ).decode()
    with zipfile.ZipFile(dependency_archive) as dependency:
        assert set(dependency.namelist()) == set(
            aura.WONDERWORLD_MARIGOLD_RUNTIME_FILES
        )
    assert 'reference = runtime / expected["bundled_path"]' in runner
    assert 'filename=expected["expected_ply_path"]' not in runner
    assert 'allow_patterns=[expected["path_prefix"] + "*"]' not in runner
    assert "provider_runtime/published_expected_point_cloud.ply" in archive_names
    assert smoke_spec["expected_output"]["bundled_path"] == (
        "published_expected_point_cloud.ply"
    )
    marigold = next(
        model
        for model in smoke_spec["runtime_models"]
        if model["repository"] == "prs-eth/marigold-depth-v1-0"
    )
    assert len(marigold["materialized_files"]) == 24
    assert marigold["materialized_total_size_bytes"] == 15_482_514_887
    assert {
        "text_encoder/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    } <= {item["path"] for item in marigold["materialized_files"]}
    marigold_alias = next(
        model
        for model in smoke_spec["runtime_models"]
        if model["repository"] == "prs-eth/marigold-v1-0"
    )
    assert marigold_alias["cache_alias_of"] == "prs-eth/marigold-depth-v1-0"
    assert marigold_alias["revision"] == marigold["revision"]
    assert marigold_alias["snapshot_digest"] == marigold["snapshot_digest"]
    assert marigold_alias["materialized_files"] == marigold["materialized_files"]
    assert "allow_patterns=" in runner
    assert "_verify_runtime_model_snapshot(snapshot, model)" in runner
    assert "HF_HUB_DISABLE_XET=1" in entrypoint
    assert "HF_HUB_DOWNLOAD_TIMEOUT=600" in entrypoint
    assert [command[1] for command in smoke_spec["author_commands"]] == [
        "train.py",
        "render.py",
        "remove.py",
        "utils/sam2_utils.py",
        "inpaint.py",
    ]
    assert "export UV_NATIVE_TLS=true" in entrypoint
    assert 'destination.stat().st_size != sd2["size_bytes"]' in runner
    assert "_extract_author_data(runtime, source, spec)" in runner
    assert "_extract_runtime_dependency(runtime, spec)" in runner
    assert 'os.environ["PYTHONPATH"] = str(runtime_dependencies)' in runner
    assert 'allow_patterns=[data["path_prefix"] + "*"]' not in runner


def test_author_data_materializer_hashes_publisher_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    author_receipt = json.loads(paths["author_data_receipt"].read_text())
    expected = author_receipt["files"][0]
    monkeypatch.setattr(
        aura,
        "_hf_repository_info",
        lambda **_kwargs: {
            "sha": aura._AUTHOR_DATA["revision"],
            "siblings": [
                {
                    "rfilename": expected["path"],
                    "blobId": expected["git_blob_id"],
                    "lfs": {
                        "size": expected["size_bytes"],
                        "sha256": expected["lfs_sha256"],
                    },
                }
            ],
        },
    )

    def fake_download(destination: Path) -> None:
        output = destination / expected["path"]
        output.parent.mkdir(parents=True)
        output.write_bytes((paths["author_data"] / expected["path"]).read_bytes())

    monkeypatch.setattr(aura, "_download_author_snapshot", fake_download)
    receipt = aura.materialize_aura_author_data(
        prerequisite_receipt_path=paths["prerequisite"],
        output_root=tmp_path / "materialized-author-data",
        generated_at="2026-08-04T00:00:00+00:00",
    )
    assert receipt["status"] == "completed"
    assert receipt["file_count"] == 1
    assert receipt["files"][0]["sha256"] == expected["sha256"]


def test_bundle_rejects_changed_materialized_author_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    target = paths["author_data"] / "360-USID/sunflower/input.txt"
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(
        ValueError, match="adp_aura_author_data_materialized_bytes_changed"
    ):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            wonderworld_root=paths["wonderworld"],
            prerequisite_receipt_path=paths["prerequisite"],
            author_data_root=paths["author_data"],
            author_data_receipt_path=paths["author_data_receipt"],
            expected_output_ply_path=paths["expected_output_ply"],
            job_dir=paths["job"],
        )


def test_bundle_rejects_changed_published_expected_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    paths["expected_output_ply"].write_bytes(b"changed publisher output")
    with pytest.raises(
        ValueError, match="adp_aura_expected_output_ply_identity_mismatch"
    ):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            wonderworld_root=paths["wonderworld"],
            prerequisite_receipt_path=paths["prerequisite"],
            author_data_root=paths["author_data"],
            author_data_receipt_path=paths["author_data_receipt"],
            expected_output_ply_path=paths["expected_output_ply"],
            job_dir=paths["job"],
        )


def test_bundle_rejects_caller_rehashed_fake_author_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    target = paths["author_data"] / "360-USID/sunflower/input.txt"
    target.write_text("caller fake", encoding="utf-8")
    receipt = json.loads(paths["author_data_receipt"].read_text())
    receipt["files"][0]["size_bytes"] = target.stat().st_size
    receipt["files"][0]["sha256"] = aura._sha256(target)
    receipt["total_size_bytes"] = target.stat().st_size
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write_json(paths["author_data_receipt"], receipt)
    with pytest.raises(
        ValueError, match="adp_aura_author_data_receipt_lfs_hash_mismatch"
    ):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            wonderworld_root=paths["wonderworld"],
            prerequisite_receipt_path=paths["prerequisite"],
            author_data_root=paths["author_data"],
            author_data_receipt_path=paths["author_data_receipt"],
            expected_output_ply_path=paths["expected_output_ply"],
            job_dir=paths["job"],
        )


def test_bundle_rejects_missing_publisher_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = json.loads(paths["prerequisite"].read_text())
    receipt["methods"]["aurafusion360_quality_challenger"][
        "author_data_rights_established"
    ] = False
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    monkeypatch.setattr(aura, "PREREQUISITE_RECEIPT_DIGEST", receipt["receipt_digest"])
    _write_json(paths["prerequisite"], receipt)
    with pytest.raises(ValueError, match="adp_aura_prerequisite_rights_missing"):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            wonderworld_root=paths["wonderworld"],
            prerequisite_receipt_path=paths["prerequisite"],
            author_data_root=paths["author_data"],
            author_data_receipt_path=paths["author_data_receipt"],
            expected_output_ply_path=paths["expected_output_ply"],
            job_dir=paths["job"],
        )


def test_bundle_rejects_changed_sam2_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    original_git = aura._git

    def changed_git(path: Path, *args: str) -> str:
        if path == paths["sam2"] and args == ("status", "--porcelain"):
            return " M sam2/modeling/sam2_base.py"
        return original_git(path, *args)

    monkeypatch.setattr(aura, "_git", changed_git)
    with pytest.raises(ValueError, match="adp_aura_sam2_source_identity_mismatch"):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            wonderworld_root=paths["wonderworld"],
            prerequisite_receipt_path=paths["prerequisite"],
            author_data_root=paths["author_data"],
            author_data_receipt_path=paths["author_data_receipt"],
            expected_output_ply_path=paths["expected_output_ply"],
            job_dir=paths["job"],
        )


def test_bundle_rejects_changed_wonderworld_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    original_git = aura._git

    def changed_git(path: Path, *args: str) -> str:
        if path == paths["wonderworld"] and args == ("status", "--porcelain"):
            return " M marigold_lcm/util/batchsize.py"
        return original_git(path, *args)

    monkeypatch.setattr(aura, "_git", changed_git)
    with pytest.raises(
        ValueError, match="adp_aura_wonderworld_source_identity_mismatch"
    ):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            wonderworld_root=paths["wonderworld"],
            prerequisite_receipt_path=paths["prerequisite"],
            author_data_root=paths["author_data"],
            author_data_receipt_path=paths["author_data_receipt"],
            expected_output_ply_path=paths["expected_output_ply"],
            job_dir=paths["job"],
        )


def test_dry_run_requires_exact_bundle_without_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        wonderworld_root=paths["wonderworld"],
        prerequisite_receipt_path=paths["prerequisite"],
        author_data_root=paths["author_data"],
        author_data_receipt_path=paths["author_data_receipt"],
        expected_output_ply_path=paths["expected_output_ply"],
        job_dir=paths["job"],
    )
    result = aura.run_aura_author_smoke_vast(
        job_dir=tmp_path / "run",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=receipt,
    )
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0


def test_provider_contract_rejects_generic_import_smoke() -> None:
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="adp_aura_smoke",
        entrypoint_text="adp_aura_smoke_runner_failed_without_runtime_result "
        "blocked_adp_aura_smoke_process_exited_without_result "
        "--no-build-isolation setuptools==80.9.0 torch==2.5.1",
        runner_text="import torch",
    )
    assert blockers == ["provider_runner_missing_adp_aura_smoke_runtime_contract"]


def test_provider_contract_rejects_isolated_native_extension_build() -> None:
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="adp_aura_smoke",
        entrypoint_text=(
            "adp_aura_smoke_runner_failed_without_runtime_result "
            "blocked_adp_aura_smoke_process_exited_without_result torch==2.5.1"
        ),
        runner_text=(
            "adp_aura_author_smoke_result.json inpaint_init_executed "
            "author_source_modified published_expected_output_bound "
            "depth_anything3_used"
        ),
    )
    assert blockers == ["provider_entrypoint_missing_runtime_result_crash_fallback"]


def test_vast_adapter_preflights_dedicated_aura_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        wonderworld_root=paths["wonderworld"],
        prerequisite_receipt_path=paths["prerequisite"],
        author_data_root=paths["author_data"],
        author_data_receipt_path=paths["author_data_receipt"],
        expected_output_ply_path=paths["expected_output_ply"],
        job_dir=paths["job"],
    )
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_aura_smoke",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_aura_smoke",
        forward_hf_token=False,
    )
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    script = _probe_shell_script(
        "https://example.test",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_aura_smoke",
    )
    assert "run_adp_aura_author_smoke_provider_runtime.sh" in script
    assert "curl " in script
    assert "--http1.1" in script
    assert '-fL "$blueprint_download_src"' in script
    assert "adp_aura_provider_runtime_output.zip" in script
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_adp_aura_author_smoke_provider_runtime.sh"
        ).decode()
    assert "--no-build-isolation" in entrypoint
    assert "setuptools==80.9.0" in entrypoint
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_aura_smoke",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.test/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.test/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"


def test_provider_runner_accepts_only_exact_runtime_model_files(tmp_path: Path) -> None:
    runner = _load_provider_runner()
    snapshot = tmp_path / "snapshot"
    source = snapshot / "component/model.safetensors"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"exact-publisher-model")
    model = {
        "materialized_files": [
            {
                "path": "component/model.safetensors",
                "size_bytes": source.stat().st_size,
                "sha256": runner._sha256(source),
            }
        ],
        "materialized_total_size_bytes": source.stat().st_size,
    }

    runner._verify_runtime_model_snapshot(snapshot, model)

    source.write_bytes(b"mutated-publisher-model")
    with pytest.raises(ValueError, match="aurafusion360_runtime_model_file_changed"):
        runner._verify_runtime_model_snapshot(snapshot, model)


def test_provider_runner_rejects_unbound_runtime_model_file(tmp_path: Path) -> None:
    runner = _load_provider_runner()
    snapshot = tmp_path / "snapshot"
    source = snapshot / "model_index.json"
    snapshot.mkdir()
    source.write_bytes(b"{}")
    model = {
        "materialized_files": [
            {
                "path": source.name,
                "size_bytes": source.stat().st_size,
                "sha256": runner._sha256(source),
            }
        ],
        "materialized_total_size_bytes": source.stat().st_size,
    }
    (snapshot / "unbound.bin").write_bytes(b"unexpected")

    with pytest.raises(ValueError, match="aurafusion360_runtime_model_file_set_changed"):
        runner._verify_runtime_model_snapshot(snapshot, model)


def test_provider_runner_materializes_exact_hardlinked_cache_alias(tmp_path: Path) -> None:
    runner = _load_provider_runner()
    cache = tmp_path / "hub"
    source = cache / "models--prs-eth--marigold-depth-v1-0/snapshots/revision"
    source.mkdir(parents=True)
    source_file = source / "model_index.json"
    source_file.write_bytes(b"exact-marigold-snapshot")
    model = {
        "repository": "prs-eth/marigold-v1-0",
        "revision": "revision",
        "cache_alias_of": "prs-eth/marigold-depth-v1-0",
        "materialized_files": [
            {
                "path": source_file.name,
                "size_bytes": source_file.stat().st_size,
                "sha256": runner._sha256(source_file),
            }
        ],
        "materialized_total_size_bytes": source_file.stat().st_size,
    }

    alias = runner._materialize_cache_alias(
        cache=cache,
        source_snapshot=source,
        model=model,
    )

    alias_file = alias / source_file.name
    assert alias_file.read_bytes() == source_file.read_bytes()
    assert alias_file.stat().st_ino == source_file.stat().st_ino
    assert (
        cache / "models--prs-eth--marigold-v1-0/refs/main"
    ).read_text(encoding="utf-8") == "revision"


def test_provider_runner_rejects_cache_alias_with_missing_source_file(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    with pytest.raises(ValueError, match="cache_alias_source_missing"):
        runner._materialize_cache_alias(
            cache=tmp_path / "hub",
            source_snapshot=tmp_path / "missing-source",
            model={
                "repository": "prs-eth/marigold-v1-0",
                "revision": "revision",
                "cache_alias_of": "prs-eth/marigold-depth-v1-0",
                "materialized_files": [
                    {"path": "missing.bin", "size_bytes": 1, "sha256": "sha256:x"}
                ],
                "materialized_total_size_bytes": 1,
            },
        )


def test_provider_runner_retains_same_camera_quality_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    runner = _load_provider_runner()
    monkeypatch.setattr(runner, "QUALITY_FRAME_INDICES", (0,))
    produced = tmp_path / "produced"
    reference = tmp_path / "reference"
    produced.mkdir()
    reference.mkdir()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(produced / "00000.png")
    Image.new("RGB", (4, 3), (12, 20, 30)).save(reference / "00000.png")
    retained = tmp_path / "output/artifacts/quality_frames"

    comparison = runner._compare_quality_frames(
        produced_render_dir=produced,
        reference_render_dir=reference,
        retained_root=retained,
    )

    assert comparison["claim_ceiling"] == (
        "same_camera_similarity_to_publisher_expected_point_cloud"
    )
    assert comparison["physical_or_hidden_surface_truth"] is False
    assert comparison["frame_indices"] == [0]
    frame = comparison["frame_comparisons"][0]
    assert frame["width"] == 4
    assert frame["height"] == 3
    assert frame["mean_absolute_error_8bit"] == pytest.approx(2.0 / 3.0)
    assert frame["psnr_db"] == pytest.approx(46.8814162426)
    assert (tmp_path / "output" / frame["produced"]["relative_path"]).is_file()
    assert (
        tmp_path / "output" / frame["publisher_reference"]["relative_path"]
    ).is_file()


def test_provider_runner_prepares_hardlinked_quality_reference_model(
    tmp_path: Path,
) -> None:
    runner = _load_provider_runner()
    runtime = tmp_path / "runtime"
    working_output = tmp_path / "working-output"
    working_output.mkdir()
    (working_output / "cfg_args").write_text("Namespace(sh_degree=3)", encoding="utf-8")
    expected = tmp_path / "published_expected_point_cloud.ply"
    expected.write_bytes(b"exact-publisher-reference")

    reference_model = runner._prepare_quality_reference_model(
        runtime=runtime,
        working_output=working_output,
        expected_point_cloud=expected,
    )

    retained = (
        reference_model
        / "point_cloud/iteration_object_inpaint_init/point_cloud.ply"
    )
    assert retained.read_bytes() == expected.read_bytes()
    assert retained.stat().st_ino == expected.stat().st_ino
    assert (reference_model / "cfg_args").read_text(encoding="utf-8") == (
        "Namespace(sh_degree=3)"
    )


def _allocator_args(
    tmp_path: Path,
    bundle_receipt: Path,
    *,
    execute: bool,
    machine_avoidlist: Path | None = None,
) -> list[str]:
    args = [
        "gpu-canary",
        "--probe-kind",
        aura.PROBE_KIND,
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
        "adp-aura-smoke",
        "--expected-source-commit",
        "a" * 40,
        "--adp-aura-bundle-receipt",
        str(bundle_receipt),
        "--adp-job-dir",
        str(tmp_path / "run"),
        "--adp-max-hourly-rate-usd",
        "1.50",
        "--adp-max-spend-usd",
        "5.00",
        "--adp-hard-ttl-seconds",
        "10800",
    ]
    if machine_avoidlist is not None:
        args.extend(["--adp-machine-avoidlist", str(machine_avoidlist)])
    if execute:
        args.append("--execute")
    return args


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_aura_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        wonderworld_root=paths["wonderworld"],
        prerequisite_receipt_path=paths["prerequisite"],
        author_data_root=paths["author_data"],
        author_data_receipt_path=paths["author_data_receipt"],
        expected_output_ply_path=paths["expected_output_ply"],
        job_dir=paths["job"],
    )
    receipt_path = paths["job"] / "adp_aura_author_smoke_bundle_receipt.json"
    monkeypatch.setattr(
        allocator,
        "ADP_AURA_PREREQUISITE_RECEIPT_DIGEST",
        receipt["prerequisite_receipt_digest"],
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    observed: dict = {}

    def fake_run(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_aura_author_smoke_vast", fake_run)
    avoidlist = tmp_path / "avoidlist.json"
    _write_json(
        avoidlist,
        {"schema_version": "vast_machine_avoidlist.v1", "machine_ids": [25706]},
    )
    assert (
        allocator.main(
            _allocator_args(
                tmp_path,
                receipt_path,
                execute=execute,
                machine_avoidlist=avoidlist,
            )
        )
        == 0
    )
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["aura_inpaint_init_author_smoke_only"] is True
    assert admission["full_author_workflow_claimed"] is False
    assert admission["hard_cap_usd"] == 5.0
    assert observed["machine_avoidlist_path"] == avoidlist
    assert admission["allocation_binding"]["machine_avoidlist_sha256"].startswith(
        "sha256:"
    )


def test_output_inspector_recognizes_aura_runtime_result(tmp_path: Path) -> None:
    output = tmp_path / "output.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "adp_aura_author_smoke_result.json",
            json.dumps({"status": "blocked", "blockers": ["typed_runtime_failure"]}),
        )
    inspection = inspect_provider_runtime_output_zip(output)
    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_blockers"] == ["typed_runtime_failure"]
