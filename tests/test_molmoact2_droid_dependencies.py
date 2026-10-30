"""Protect the exact Molmo inference dependency closure without downloads/GPU."""
import hashlib
import json

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
import pytest

from blueprint_pipeline import molmoact2_droid_dependencies as dependencies


def test_transitive_requirements_include_every_requested_cuda_toolkit_extra():
    manifest = dependencies.verified_dependency_manifest()
    rows = {canonicalize_name(row["name"]): row for row in manifest["server_wheels"]}
    for row in rows.values():
        if row["requires_python"]:
            assert SpecifierSet(row["requires_python"]).contains("3.12.12")
        for text in row["requires_dist"]:
            requirement = Requirement(text)
            assert requirement.marker is None or any(requirement.marker.evaluate({
                **manifest["server_environment"], "extra": extra,
            }) for extra in ["", *row["requested_extras"]])
            target = rows[canonicalize_name(requirement.name)]
            assert requirement.specifier.contains(target["version"]), (row["name"], text)
            assert requirement.extras <= set(target["requested_extras"])
    toolkit = rows["cuda-toolkit"]
    assert set(toolkit["requested_extras"]) == {
        "cublas", "cudart", "cufft", "cufile", "cupti", "curand", "cusolver",
        "cusparse", "nvjitlink", "nvrtc", "nvtx",
    }
    assert "nvidia-cublas-cu12" in rows and "nvidia-cusolver-cu12" in rows
    assert rows["torch"]["version"] == "2.11.0+cu128"
    assert rows["torchvision"]["version"] == "0.26.0+cu128"
    assert rows["transformers"]["version"] == "4.57.6"
    assert not set(rows) & set(manifest["excluded_optional_packages"])


def test_all_server_wheels_match_exact_interpreter_and_platform():
    manifest = dependencies.verified_dependency_manifest()
    platforms = [f"manylinux_2_{n}_x86_64" for n in range(28, 16, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = set(cpython_tags((3, 12), platforms=platforms))
    tags.update(compatible_tags((3, 12), "cp312", platforms=platforms))
    for row in manifest["server_wheels"]:
        assert parse_wheel_filename(row["filename"])[3] & tags
    assert manifest["thin_client_wheels"] == []
    assert manifest["thin_client_python"] == "native_isaac_python"


def test_source_identity_includes_all_checkpoint_code_and_nonweight_audit_inputs():
    manifest = dependencies.verified_dependency_manifest()
    sources = manifest["checkpoint_import_sources"]
    assert {r["filename"] for r in sources} == {
        "configuration_molmoact2.py", "modeling_molmoact2.py", "inference.py",
        "processing_molmoact2.py", "image_processing_molmoact2.py", "video_processing_molmoact2.py",
    }
    inputs = {r["filename"]: r for r in manifest["checkpoint_audit_inputs"]}
    assert len(inputs) == 14
    assert "tokenizer.json" in inputs and "norm_stats.json" in inputs
    assert all(inputs[r["filename"]]["sha256"] == r["sha256"] for r in sources)
    assert not any(r.endswith(".safetensors") for r in inputs)
    assert all(manifest["model_revision"] in r["url"] for r in sources)


def test_budget_counts_dependencies_once_and_keeps_checkpoint_transfer_separate():
    manifest = dependencies.verified_dependency_manifest()
    rows = [manifest["interpreter"], *manifest["server_wheels"], *manifest["thin_client_wheels"]]
    total = sum({r["url"]: r["size_bytes"] for r in rows}.values())
    assert total == dependencies.TOTAL_DOWNLOAD_BYTES == manifest["total_download_bytes"]
    assert manifest["bundled_wheel_bytes"] == 0
    assert manifest["checkpoint_source_downloads_counted_in"] == "separate_model_inventory_not_dependency_total"
    assert manifest["interpreter"]["observed_expanded_file_bytes"] <= manifest["interpreter"]["maximum_expanded_bytes"]


def test_staging_supports_provider_stdlib_only_validation(tmp_path):
    path = dependencies.stage_dependency_manifest(tmp_path)
    value = dependencies.verified_dependency_manifest(path)
    assert path.name == dependencies.DEPENDENCY_MANIFEST_FILENAME
    assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == dependencies.DEPENDENCY_MANIFEST_SHA256
    assert value["manifest_digest"] == dependencies.DEPENDENCY_MANIFEST_DIGEST


def test_changed_manifest_with_recomputed_self_digest_still_rejected(tmp_path):
    path = dependencies.stage_dependency_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["server_wheels"][0]["url"] = "https://example.com/unverified.whl"
    value.pop("manifest_digest")
    value["manifest_digest"] = "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest_bytes_mismatch"):
        dependencies.verified_dependency_manifest(path)


def test_staging_does_not_write_through_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / dependencies.DEPENDENCY_MANIFEST_FILENAME).symlink_to(outside)
    with pytest.raises(ValueError, match="staging_symlink"):
        dependencies.stage_dependency_manifest(runtime)
    assert outside.read_text() == "preserve"


def test_droid_reuses_exact_wheels_but_has_its_own_source_and_cpu_binding():
    from blueprint_pipeline.molmoact2_yam_dependencies import verified_dependency_manifest
    droid = dependencies.verified_dependency_manifest()
    yam = verified_dependency_manifest()
    assert droid["server_wheels"] == yam["server_wheels"]
    assert droid["interpreter"] == yam["interpreter"]
    assert droid["model_id"] == "allenai/MolmoAct2-DROID" and droid["model_revision"] != yam["model_revision"]
    assert droid["validation"]["cpu_receipt_path"] != yam["validation"]["cpu_receipt_path"]
    assert droid["validation"]["cpu_checkpoint_modules_imported"] == 5
    assert droid["runtime_environment_reuse"]["droid_cpu_proof_separate"] is True
