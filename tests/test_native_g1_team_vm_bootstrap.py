"""Offline host bootstrap must work before G1's cp312 dependencies can import."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import native_g1_team_vm_bootstrap as bootstrap
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

COMMIT = "a" * 40


def _tar(path, *, fault=None):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in [("python/bin/python3.12", b"runtime"), ("python/LICENSE", b"PSF-2.0")]:
            info = tarfile.TarInfo("../outside" if fault == "traversal" else
                                   "python/bin/python3.12" if fault == "duplicate" else name)
            info.size, info.mode = len(content), 0o755 if "bin" in name else 0o644
            archive.addfile(info, io.BytesIO(content))
        info = tarfile.TarInfo("python/bin/python3")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd" if fault == "link" else "python3" if fault == "cycle" else "python3.12"
        archive.addfile(info)


def _wheel(path, *, package, version, tag, fault=None):
    with zipfile.ZipFile(path, "w") as archive:
        dist = package + "-" + version + ".dist-info/"
        archive.writestr(dist + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: " +
                         ("true" if package == "rfc8785" else "false") + "\nTag: " + tag + "\n")
        archive.writestr(dist + "METADATA", f"Name: {package}\nVersion: {version}\n")
        archive.writestr(dist + "LICENSE", "retained license")
        archive.writestr("../escape" if fault == "traversal" else package + "/__init__.py", "")


@pytest.fixture
def package(tmp_path, monkeypatch):
    paths, assets = {}, {}
    for role, row in bootstrap.ASSETS.items():
        path = tmp_path / row["filename"]
        if role == "python":
            _tar(path)
        else:
            _wheel(path, package=row["package"], version=row["version"], tag=row["wheel_tag"])
        paths[role] = path
        assets[role] = {**row, "size_bytes": path.stat().st_size,
                        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(bootstrap, "ASSETS", assets)
    root = tmp_path / "sealed"
    manifest = bootstrap.seal_g1_vm_host_python(asset_paths=paths, output_root=root,
                                               implementation_commit=COMMIT)
    return root, manifest, paths


def test_exact_assets_seal_reopen_and_preserve_source_bytes(package):
    root, manifest, paths = package
    assert bootstrap.verify_g1_vm_host_python(root, expected_implementation_commit=COMMIT) == manifest
    assert manifest["manifest_digest"] == canonical_digest(manifest, digest_field="manifest_digest")
    assert manifest["provider_mutation_performed"] is False
    assert manifest["rights_authorized"] is False
    for role, row in bootstrap.ASSETS.items():
        assert (root / row["filename"]).read_bytes() == paths[role].read_bytes()


@pytest.mark.parametrize("fault", ["asset", "missing", "catalogue", "commit", "alias", "boolean_type"])
def test_reopen_refuses_altered_or_foreign_inputs(package, tmp_path, fault):
    root, manifest, _ = package
    expected = COMMIT
    if fault in {"asset", "missing"}:
        path = root / bootstrap.ASSETS["numpy"]["filename"]
        path.chmod(0o600)
        if fault == "missing":
            path.unlink()
        else:
            path.write_bytes(b"changed")
    elif fault in {"catalogue", "boolean_type"}:
        if fault == "catalogue":
            manifest["assets"]["numpy"]["version"] = "1.26.4"
            manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
        else:
            manifest["provider_mutation_performed"] = 0  # Must not equal JSON false.
        (root / bootstrap.MANIFEST_NAME).chmod(0o600)
        (root / bootstrap.MANIFEST_NAME).write_text(json.dumps(manifest))
    elif fault == "commit":
        expected = "b" * 40
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    with pytest.raises(ValueError):
        bootstrap.verify_g1_vm_host_python(root, expected_implementation_commit=expected)


@pytest.mark.parametrize("fault", ["traversal", "link", "duplicate", "cycle"])
def test_trusted_python_extraction_rejects_outward_paths_before_writing(tmp_path, fault):
    archive = tmp_path / "python.tar.gz"
    _tar(archive, fault=fault)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError):
        bootstrap.extract_host_python_archive(archive, output)
    assert list(output.iterdir()) == []
    assert not (tmp_path / "outside").exists()


def test_trusted_python_internal_link_stays_inside_new_root(tmp_path):
    archive = tmp_path / "python.tar.gz"
    _tar(archive)
    output = tmp_path / "output"
    output.mkdir()
    bootstrap.extract_host_python_archive(archive, output)
    assert (output / "python/bin/python3").resolve() == output / "python/bin/python3.12"
    assert (output / "python/LICENSE").read_bytes() == b"PSF-2.0"


@pytest.mark.parametrize("fault", ["traversal", "tag"])
def test_wheel_platform_or_path_mismatch_is_rejected(tmp_path, fault):
    path = tmp_path / "numpy.whl"
    row = bootstrap.ASSETS["numpy"]
    _wheel(path, package="numpy", version=row["version"],
           tag="cp310-cp310-manylinux_2_28_x86_64" if fault == "tag" else row["wheel_tag"],
           fault=fault)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError):
        bootstrap.extract_host_dependency_wheel(path, output, row)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("fault", [None, "python", "numpy", "origin", "exit"])
def test_materialization_requires_observed_isolated_runtime(package, tmp_path, monkeypatch, fault):
    root, manifest, _ = package
    destination = tmp_path / "runtime"
    provider = tmp_path / "provider_runtime"
    (provider / "blueprint_pipeline").mkdir(parents=True)
    source = provider / "blueprint_pipeline/__init__.py"
    source.write_text("# sealed test source\n")
    provider_manifest = {"schema_version": "native_g1_team_provider_bundle.v1", "implementation_commit": COMMIT,
        "artifacts": [{"relative_path": "provider_runtime/blueprint_pipeline/__init__.py",
                       "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                       "size_bytes": source.stat().st_size}]}
    provider_manifest["manifest_digest"] = canonical_digest(provider_manifest, digest_field="manifest_digest")
    (provider / "native_g1_team_provider_manifest.json").write_text(json.dumps(provider_manifest))
    calls = []
    def probe(command, **kwargs):
        calls.append((command, kwargs))
        value = {"python_version": "3.10.12" if fault == "python" else "3.12.14",
                 "implementation": "cpython", "platform": "linux", "machine": "x86_64",
                 "numpy_version": "1.26.4" if fault == "numpy" else "2.3.1", "rfc8785_version": "0.1.4",
                 "sealed_module_origins_verified": fault != "origin"}
        return SimpleNamespace(returncode=7 if fault == "exit" else 0, stdout=json.dumps(value), stderr="private")
    monkeypatch.setattr(bootstrap.subprocess, "run", probe)
    if fault is not None:
        with pytest.raises(ValueError):
            bootstrap.materialize_g1_vm_host_python(package_root=root, destination_root=destination,
                provider_runtime_root=provider, expected_implementation_commit=COMMIT,
                expected_provider_manifest_digest=provider_manifest["manifest_digest"])
    else:
        result = bootstrap.materialize_g1_vm_host_python(package_root=root, destination_root=destination,
            provider_runtime_root=provider, expected_implementation_commit=COMMIT,
            expected_provider_manifest_digest=provider_manifest["manifest_digest"])
        assert result["source_manifest_digest"] == manifest["manifest_digest"]
        assert result["provider_mutation_performed"] is False
        assert result["gpu_runtime_qualified"] is False
        assert calls[0][0][0] == str(destination / "python/bin/python3.12")
        assert calls[0][0][1] == "-I"
        assert set(calls[0][1]["env"]) == {"PATH", "LANG", "PYTHONDONTWRITEBYTECODE"}


@pytest.mark.parametrize("fault", ["changed", "unlisted", "digest", "alias"])
def test_provider_source_is_verified_before_extraction_or_execution(package, tmp_path, monkeypatch, fault):
    root, _, _ = package
    provider = tmp_path / "provider_runtime"
    (provider / "blueprint_pipeline").mkdir(parents=True)
    source = provider / "blueprint_pipeline/__init__.py"
    source.write_text("# sealed source\n")
    manifest = {"schema_version": "native_g1_team_provider_bundle.v1", "implementation_commit": COMMIT,
        "artifacts": [{"relative_path": "provider_runtime/blueprint_pipeline/__init__.py",
                       "sha256": "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest(),
                       "size_bytes": source.stat().st_size}]}
    digest = canonical_digest(manifest, digest_field="manifest_digest")
    manifest["manifest_digest"] = digest
    (provider / "native_g1_team_provider_manifest.json").write_text(json.dumps(manifest))
    if fault == "changed":
        source.write_text("# changed source\n")
    elif fault == "unlisted":
        (provider / "numpy.py").write_text("raise AssertionError('foreign code executed')")
    elif fault == "digest":
        digest = "sha256:" + "b" * 64
    else:
        alias = tmp_path / "provider-alias"
        alias.symlink_to(provider, target_is_directory=True)
        provider = alias
    destination = tmp_path / "runtime"
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: pytest.fail("unverified source executed"))
    with pytest.raises(ValueError):
        bootstrap.materialize_g1_vm_host_python(package_root=root, destination_root=destination,
            provider_runtime_root=provider, expected_implementation_commit=COMMIT, expected_provider_manifest_digest=digest)
    assert not destination.exists()


def test_duplicate_manifest_keys_are_rejected_before_asset_use(package):
    root, _, _ = package
    path = root / bootstrap.MANIFEST_NAME
    value = path.read_text()
    path.chmod(0o600)
    path.write_text(value.replace('"python_abi":"cp312"', '"python_abi":"cp310","python_abi":"cp312"'))
    with pytest.raises(ValueError, match="duplicate_json_key"):
        bootstrap.verify_g1_vm_host_python(root, expected_implementation_commit=COMMIT)


@pytest.mark.slow
def test_bootstrap_cli_imports_with_standard_library_only(tmp_path):
    script = Path(bootstrap.__file__)
    result = subprocess.run([sys.executable, "-I", "-S", str(script), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
