"""Offline policy dependency contracts; no model, provider or GPU execution."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from blueprint_pipeline import native_g1_beta_dependency_profile as profile
from blueprint_pipeline.native_task_runtime_source_packet import BASE_RUNTIME_DEPENDENCY_WHEELS
from blueprint_pipeline.native_task_torch_runtime_lock import TORCH_RUNTIME_DEPENDENCY_WHEELS


def test_fixed_manifest_closes_exact_inference_graph_and_quotes_only_new_bytes():
    manifest = profile.artifact_manifest()
    rows = {row["name"]: row for row in manifest["wheels"]}
    assert len(rows) == len(manifest["wheels"])
    assert manifest["source_revision"] == "68479287a784a69be9ce6ad739311d2f11f75ef9"
    assert manifest["core_versions"] == {
        "torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128",
        "transformers": "5.3.0", "diffusers": "0.35.2", "numpy": "2.2.6",
    }
    assert not set(manifest["excluded_unused_dependencies"]) & rows.keys()
    assert {"datasets", "av", "pyarrow", "imageio", "pyserial"} <= rows.keys()
    native = {row["filename"]: row for row in [*BASE_RUNTIME_DEPENDENCY_WHEELS, *TORCH_RUNTIME_DEPENDENCY_WHEELS]}
    environment = manifest["python_environment"]
    platforms = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 11, -1)] + [
        "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64",
    ]
    compatible = set(cpython_tags((3, 12), platforms=platforms)) | set(
        compatible_tags((3, 12), "cp312", platforms=platforms)
    )
    for row in rows.values():
        assert row["license_metadata"] and row["license_provenance"]
        assert row["filename"] == Path(row["filename"]).name
        assert len(row["sha256"]) == 64 and row["size_bytes"] > 0
        if row["requires_python"]:
            assert SpecifierSet(row["requires_python"]).contains(environment["python_full_version"])
        for text in row["requires_dist"]:
            requirement = Requirement(text)
            assert canonicalize_name(requirement.name) in rows
            target = rows[canonicalize_name(requirement.name)]
            assert requirement.specifier.contains(target["version"]), text
        tags = parse_wheel_filename(row["filename"])[3]
        assert tags & compatible
        if row["source"] == "runtime_source_packet":
            retained = native[row["filename"]]
            assert row["version"] == retained["version"]
            assert row["name"] == canonicalize_name(retained["package"])
            assert row["archive_path"] == "runtime_dependencies/wheels/" + row["filename"]
        else:
            assert row["source"] == "external_download"
            assert row["url"].startswith(("https://files.pythonhosted.org/", "https://download-r2.pytorch.org/"))
    assert sum(r["size_bytes"] for r in rows.values() if r["source"] == "external_download") == profile.TOTAL_DOWNLOAD_BYTES
    assert manifest["interpreter"]["additional_download_bytes"] == 0
    assert manifest["installation"]["system_site_packages"] is False


def test_staged_manifest_is_exact_and_refuses_resealed_substitution(tmp_path):
    staged = profile.stage_dependency_manifest(tmp_path)
    assert profile.artifact_manifest(staged) == profile.artifact_manifest()
    changed = json.loads(staged.read_text())
    changed["wheels"][0]["url"] = "https://invalid.example/changed.whl"
    staged.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest_bytes_mismatch"):
        profile.artifact_manifest(staged)


def test_manifest_and_staging_refuse_symlinks(tmp_path):
    staged = profile.stage_dependency_manifest(tmp_path / "real")
    link = tmp_path / "linked"
    link.symlink_to(staged)
    with pytest.raises(ValueError, match="manifest_file_invalid"):
        profile.artifact_manifest(link)
    (tmp_path / profile.DEPENDENCY_MANIFEST_FILENAME).symlink_to(staged)
    with pytest.raises(ValueError, match="staging_symlink"):
        profile.stage_dependency_manifest(tmp_path)


def _wheelhouse(tmp_path, monkeypatch):
    wheel = tmp_path / "development_only-1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("development_only.py", b"# No model evidence\n")
    row = {"filename": wheel.name, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "size_bytes": wheel.stat().st_size}
    monkeypatch.setattr(profile, "artifact_manifest", lambda: {"wheels": [row], "manifest_digest": "development_only"})
    return wheel, {wheel.name: wheel}


def test_wheelhouse_requires_every_exact_retained_or_downloaded_wheel(tmp_path, monkeypatch):
    wheel, paths = _wheelhouse(tmp_path, monkeypatch)
    assert profile.verify_wheelhouse(paths)["wheel_count"] == 1
    with pytest.raises(ValueError, match="wheel_set_mismatch"):
        profile.verify_wheelhouse({})
    with pytest.raises(ValueError, match="wheel_set_mismatch"):
        profile.verify_wheelhouse({**paths, "unquoted.whl": wheel})
    raw = bytearray(wheel.read_bytes())
    raw[-1] ^= 1
    wheel.write_bytes(raw)
    with pytest.raises(ValueError, match="wheel_digest_mismatch"):
        profile.verify_wheelhouse(paths)


def test_wheelhouse_rejects_symlink_and_expanded_budget_overrun(tmp_path, monkeypatch):
    wheel, paths = _wheelhouse(tmp_path, monkeypatch)
    link = tmp_path / "linked.whl"
    link.symlink_to(wheel)
    with pytest.raises(ValueError, match="wheel_file_invalid"):
        profile.verify_wheelhouse({wheel.name: link})
    monkeypatch.setattr(profile, "MAXIMUM_EXPANDED_BYTES", 1)
    with pytest.raises(ValueError, match="expanded_byte_budget_exceeded"):
        profile.verify_wheelhouse(paths)


@pytest.mark.parametrize("member", ["../escape", "/absolute", "C:/absolute", "bad\\path"])
def test_wheelhouse_rejects_unsafe_members_even_with_matching_digest(tmp_path, monkeypatch, member):
    wheel, paths = _wheelhouse(tmp_path, monkeypatch)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, b"development_only")
    manifest = profile.artifact_manifest()
    manifest["wheels"][0].update(size_bytes=wheel.stat().st_size, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    monkeypatch.setattr(profile, "artifact_manifest", lambda: manifest)
    with pytest.raises(ValueError, match="wheel_member_invalid"):
        profile.verify_wheelhouse(paths)


def test_wheelhouse_checks_absolute_deadline_during_hashing(tmp_path, monkeypatch):
    _, paths = _wheelhouse(tmp_path, monkeypatch)
    ticks = iter([1, 2, 4])
    monkeypatch.setattr(profile.time, "time", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="verification_deadline_exceeded"):
        profile.verify_wheelhouse(paths, deadline_epoch=3)


@pytest.mark.parametrize("deadline", [True, "1", float("nan"), float("inf"), -1])
def test_wheelhouse_refuses_invalid_deadline(tmp_path, monkeypatch, deadline):
    _, paths = _wheelhouse(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="deadline_invalid"):
        profile.verify_wheelhouse(paths, deadline_epoch=deadline)



def test_wheelhouse_rejects_archived_symlink(tmp_path, monkeypatch):
    wheel, paths = _wheelhouse(tmp_path, monkeypatch)
    entry = zipfile.ZipInfo("linked.py")
    entry.create_system = 3
    entry.external_attr = 0o120777 << 16
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(entry, "../external")
    manifest = profile.artifact_manifest()
    manifest["wheels"][0].update(size_bytes=wheel.stat().st_size, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
    monkeypatch.setattr(profile, "artifact_manifest", lambda: manifest)
    with pytest.raises(ValueError, match="wheel_member_invalid"):
        profile.verify_wheelhouse(paths)
