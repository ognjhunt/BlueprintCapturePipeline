"""Dependency closure and immutable staging without installing a model runtime."""
import hashlib
import json
from pathlib import Path
import zipfile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
import pytest

from blueprint_pipeline import openwam_piper_dependencies as dependencies


def test_exact_inference_closure_has_no_missing_or_conflicting_dependencies():
    manifest = dependencies.verified_dependency_manifest()
    rows = {canonicalize_name(row["name"]): row for row in manifest["server_wheels"]}
    environment = manifest["server_environment"]
    for row in rows.values():
        assert row["filename"].endswith(".whl")
        if row["requires_python"]:
            assert SpecifierSet(row["requires_python"]).contains("3.12.12")
        for text in row["requires_dist"]:
            requirement = Requirement(text)
            assert requirement.marker is None or requirement.marker.evaluate(environment)
            target = rows[canonicalize_name(requirement.name)]
            assert requirement.specifier.contains(target["version"]), (row["name"], text)
    assert not set(manifest["excluded_optional_packages"]) & set(rows)
    assert rows["torch"]["version"] == "2.7.1+cu128"
    assert rows["torchvision"]["version"] == "0.22.1+cu128"
    assert "h5py" in rows and "pyarrow" in rows  # Eager upstream reader imports.


@pytest.mark.parametrize("group,minor", [("server_wheels", 12), ("thin_client_wheels", 11)])
def test_all_wheels_have_real_target_interpreter_platform_tags(group, minor):
    manifest = dependencies.verified_dependency_manifest()
    platforms = [f"manylinux_2_{n}_x86_64" for n in range(28, 16, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = set(cpython_tags((3, minor), platforms=platforms))
    tags.update(compatible_tags((3, minor), f"cp3{minor}", platforms=platforms))
    for row in manifest[group]:
        assert parse_wheel_filename(row["filename"])[3] & tags, row["filename"]
    if group == "thin_client_wheels":
        assert [row["name"] for row in manifest[group]] == ["websockets"]


def test_budget_counts_unique_external_downloads_and_no_bundled_wheel():
    manifest = dependencies.verified_dependency_manifest()
    rows = [manifest["interpreter"], *manifest["server_wheels"], *manifest["thin_client_wheels"]]
    actual = sum({row["url"]: row["size_bytes"] for row in rows if "url" in row}.values())
    assert actual == dependencies.TOTAL_DOWNLOAD_BYTES == manifest["total_download_bytes"]
    assert sum(row["size_bytes"] for row in rows if "bundled_path" in row) == manifest["bundled_wheel_bytes"]
    assert manifest["interpreter"]["observed_expanded_file_bytes"] <= manifest["interpreter"]["maximum_expanded_bytes"]


def test_staging_retains_installable_pure_python_antlr_and_license(tmp_path):
    path = dependencies.stage_dependency_manifest(tmp_path)
    assert path.name == dependencies.DEPENDENCY_MANIFEST_FILENAME
    manifest = dependencies.verified_dependency_manifest(path)
    row = next(row for row in manifest["server_wheels"] if "bundled_path" in row)
    wheel = tmp_path / row["bundled_path"]
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == row["sha256"]
    with zipfile.ZipFile(wheel) as archive:
        assert "antlr4/__init__.py" in archive.namelist()
        assert not any(name.endswith((".so", ".dll")) for name in archive.namelist())
        license_name = next(name for name in archive.namelist() if name.endswith("/LICENSE.txt"))
        assert hashlib.sha256(archive.read(license_name)).hexdigest() == row["license_source"]["sha256"]
        assert b"BSD 3-clause" in archive.read(license_name)


def test_staged_manifest_tampering_is_rejected_even_with_new_self_digest(tmp_path):
    path = dependencies.stage_dependency_manifest(tmp_path)
    value = json.loads(path.read_text())
    value["server_wheels"][0]["url"] = "https://example.com/arbitrary.whl"
    value.pop("manifest_digest")
    value["manifest_digest"] = "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest_bytes_mismatch"):
        dependencies.verified_dependency_manifest(path)


def test_bundled_wheel_tampering_is_not_staged(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    manifest = dependencies.verified_dependency_manifest()
    (vendor / dependencies.DEPENDENCY_MANIFEST_FILENAME).write_bytes(
        (Path(dependencies.__file__).parent / "_vendor" / dependencies.DEPENDENCY_MANIFEST_FILENAME).read_bytes())
    row = next(row for row in manifest["server_wheels"] if "bundled_path" in row)
    (vendor / row["filename"]).write_bytes(b"x" * row["size_bytes"])
    monkeypatch.setattr(dependencies, "_VENDOR", vendor)
    with pytest.raises(ValueError, match="bundled_wheel_digest_mismatch"):
        dependencies.stage_dependency_manifest(tmp_path / "runtime")


def test_stage_cannot_follow_destination_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "openwam_dependencies").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="staging_symlink"):
        dependencies.stage_dependency_manifest(runtime)
    assert list(outside.iterdir()) == []
