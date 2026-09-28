"""Seal and provision offline G1 host Python; stdlib-only before cp312 exists.

No network, provider allocation or rights grant exists in this module. The
canonical controller must admit this prerequisite and the complete VM runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import stat
import subprocess
import tarfile
from typing import Any, Sequence
import zipfile

SCHEMA = "native_g1_team_vm_host_python_package.v1"
MANIFEST_NAME = SCHEMA + ".json"
RUNTIME_SCHEMA = "native_g1_team_vm_host_python_provisioning.v1"
ASSETS = {
    "python": {
        "package": "cpython", "version": "3.12.14",
        "filename": "cpython-3.12.14+20260924-x86_64-unknown-linux-gnu-install_only.tar.gz",
        "sha256": "sha256:5eae8cf79dd47fc2496a4fccc892936be831ce7a84d984b2299dfb1cdb592682",
        "size_bytes": 66890910,
        "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20260924/"
               "cpython-3.12.14%2B20260924-x86_64-unknown-linux-gnu-install_only.tar.gz",
    },
    "numpy": {
        "package": "numpy", "version": "2.3.1",
        "filename": "numpy-2.3.1-cp312-cp312-manylinux_2_28_x86_64.whl",
        "sha256": "sha256:e7cbf5a5eafd8d230a3ce356d892512185230e4781a361229bd902ff403bc660",
        "size_bytes": 16632729, "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
        "url": "https://files.pythonhosted.org/packages/6e/45/c51cb248e679a6c6ab14b7a8e3ead3f4a3fe7425fc7a6f98b3f147bec532/"
               "numpy-2.3.1-cp312-cp312-manylinux_2_28_x86_64.whl",
    },
    "rfc8785": {
        "package": "rfc8785", "version": "0.1.4", "filename": "rfc8785-0.1.4-py3-none-any.whl",
        "sha256": "sha256:520d690b448ecf0703691c76e1a34a24ddcd4fc5bc41d589cb7c58ec651bcd48",
        "size_bytes": 9240, "wheel_tag": "py3-none-any",
        "url": "https://files.pythonhosted.org/packages/4d/78/119878110660b2ad709888c8a1614fce7e2fab39080ab960656dc8605bf6/"
               "rfc8785-0.1.4-py3-none-any.whl",
    },
}
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_MAX_BYTES = 512 * 1024**2


def _canonical(value: Any) -> str:
    # Restricted manifest domain is ASCII strings/bools/null/exact small ints;
    # stdlib serialization equals RFC8785 here without requiring that package.
    def check(item):
        if type(item) in (bool, type(None)):
            return
        if type(item) is int and abs(item) < 2**53:
            return
        if isinstance(item, str) and item.isascii():
            return
        if isinstance(item, list):
            for entry in item:
                check(entry)
            return
        if isinstance(item, dict):
            for key, entry in item.items():
                if not isinstance(key, str) or not key.isascii():
                    raise ValueError("g1_host_python_manifest_domain_invalid")
                check(entry)
            return
        raise ValueError("g1_host_python_manifest_domain_invalid")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value, field="manifest_digest"):
    return "sha256:" + hashlib.sha256(_canonical({k: v for k, v in value.items() if k != field}).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("g1_host_python_duplicate_json_key")
        result[key] = value
    return result


def _path(path: Path, *, directory: bool):
    if (not isinstance(path, Path) or not path.is_absolute() or path.resolve() != path
            or not (path.is_dir() if directory else path.is_file())):
        raise ValueError("g1_host_python_path_invalid")


def _asset(path, row):
    _path(path, directory=False)
    if path.stat().st_size != row["size_bytes"] or _sha(path) != row["sha256"]:
        raise ValueError("g1_host_python_asset_identity_invalid")


def _persist(path, value, field):
    value[field] = _digest(value, field)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_canonical(value) + "\n")
    path.chmod(0o444)
    return value


def seal_g1_vm_host_python(*, asset_paths, output_root: Path, implementation_commit: str):
    if (not _COMMIT.fullmatch(implementation_commit) or set(asset_paths) != set(ASSETS)
            or not output_root.is_absolute() or output_root.resolve() != output_root
            or output_root.exists() or not output_root.parent.is_dir()):
        raise ValueError("g1_host_python_seal_inputs_invalid")
    for role, row in ASSETS.items():
        _asset(asset_paths[role], row)
    output_root.mkdir(mode=0o700)
    for role, row in ASSETS.items():
        destination = output_root / row["filename"]
        shutil.copyfile(asset_paths[role], destination)
        _asset(destination, row)
        destination.chmod(0o444)
    return _persist(output_root / MANIFEST_NAME, {
        "schema_version": SCHEMA, "status": "sealed_runtime_prerequisites",
        "implementation_commit": implementation_commit, "platform": "linux-x86_64", "python_abi": "cp312",
        "assets": json.loads(_canonical(ASSETS)), "provider_network_install_required": False,
        "provider_mutation_performed": False, "rights_authorized": False,
        "gpu_runtime_qualified": False, "claim_ceiling": "development_only",
    }, "manifest_digest")


def verify_g1_vm_host_python(root: Path, *, expected_implementation_commit: str):
    _path(root, directory=True)
    path = root / MANIFEST_NAME
    _path(path, directory=False)
    if path.stat().st_size > 16384:
        raise ValueError("g1_host_python_manifest_invalid")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    expected = {
        "schema_version": SCHEMA, "status": "sealed_runtime_prerequisites",
        "implementation_commit": expected_implementation_commit, "platform": "linux-x86_64", "python_abi": "cp312",
        "assets": ASSETS, "provider_network_install_required": False,
        "provider_mutation_performed": False, "rights_authorized": False,
        "gpu_runtime_qualified": False, "claim_ceiling": "development_only",
    }
    expected["manifest_digest"] = _digest(expected)
    if not _COMMIT.fullmatch(expected_implementation_commit) or _canonical(value) != _canonical(expected):
        raise ValueError("g1_host_python_manifest_invalid")
    if {p.name for p in root.iterdir()} != {MANIFEST_NAME, *(row["filename"] for row in ASSETS.values())}:
        raise ValueError("g1_host_python_inventory_invalid")
    for row in ASSETS.values():
        _asset(root / row["filename"], row)
    return value


def _relative(name):
    relative = PurePosixPath(name.rstrip("/"))
    if (not name or not relative.parts or relative.is_absolute() or str(relative) != name.rstrip("/")
            or any(part in {".", ".."} for part in relative.parts)):
        raise ValueError("g1_host_python_member_path_invalid")
    return relative


def extract_host_python_archive(archive_path: Path, destination: Path):
    _path(destination, directory=True)
    if list(destination.iterdir()):
        raise ValueError("g1_host_python_extraction_not_empty")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 20000:
            raise ValueError("g1_host_python_member_limit")
        total, names, links = 0, set(), []
        for member in members:
            name = _relative(member.name)
            if (name.parts[0] != "python" or str(name) in names or member.size < 0
                    or not (member.isfile() or member.isdir() or member.issym())):
                raise ValueError("g1_host_python_archive_member_invalid")
            names.add(str(name))
            total += member.size
            if total > _MAX_BYTES:
                raise ValueError("g1_host_python_expansion_limit")
            if member.issym():
                target = posixpath.normpath(str(name.parent / member.linkname))
                if (PurePosixPath(member.linkname).is_absolute() or not target.startswith("python/")):
                    raise ValueError("g1_host_python_archive_link_invalid")
                links.append((member, target))
        if any(target not in names for _, target in links):
            raise ValueError("g1_host_python_archive_link_missing")
        link_targets = {member.name: target for member, target in links}
        for member, target in links:
            seen = {member.name}
            while target in link_targets:
                if target in seen:
                    raise ValueError("g1_host_python_archive_link_cycle")
                seen.add(target)
                target = link_targets[target]
        if any(parent.as_posix() in link_targets for name in names for parent in PurePosixPath(name).parents):
            raise ValueError("g1_host_python_archive_link_parent")
        if shutil.disk_usage(destination).free < total + 128 * 1024**2:
            raise ValueError("g1_host_python_capacity_insufficient")
        for member in members:
            target = destination.joinpath(*_relative(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("g1_host_python_archive_member_unreadable")
                with target.open("xb") as output, source:
                    shutil.copyfileobj(source, output)
                target.chmod(0o555 if member.mode & 0o111 else 0o444)
        for member, _ in links:
            target = destination.joinpath(*_relative(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            target.symlink_to(member.linkname)
        for member, _ in links:
            try:
                target = destination.joinpath(*_relative(member.name).parts).resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError("g1_host_python_archive_link_invalid") from exc
            if not target.is_relative_to(destination / "python"):
                raise ValueError("g1_host_python_archive_link_invalid")


def extract_host_dependency_wheel(path: Path, destination: Path, row):
    _path(destination, directory=True)
    with zipfile.ZipFile(path) as archive:
        members, paths, total = archive.infolist(), set(), 0
        if len(members) > 20000:
            raise ValueError("g1_host_python_member_limit")
        for member in members:
            relative = _relative(member.filename)
            mode = member.external_attr >> 16
            if (str(relative) in paths or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or member.file_size < 0
                    or any(part.endswith(".data") for part in relative.parts)):
                raise ValueError("g1_host_python_wheel_member_invalid")
            paths.add(str(relative))
            total += member.file_size
            if total > _MAX_BYTES:
                raise ValueError("g1_host_python_expansion_limit")
        dist = row["package"] + "-" + row["version"] + ".dist-info/"
        wheel = archive.read(dist + "WHEEL").decode("utf-8")
        metadata = archive.read(dist + "METADATA").decode("utf-8")
        pure = "true" if row["package"] == "rfc8785" else "false"
        if (f"Tag: {row['wheel_tag']}" not in wheel or f"Root-Is-Purelib: {pure}" not in wheel
                or f"Name: {row['package']}" not in metadata or f"Version: {row['version']}" not in metadata):
            raise ValueError("g1_host_python_wheel_platform_invalid")
        for member in members:
            target = destination.joinpath(*_relative(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
            else:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                with target.open("xb") as output:
                    output.write(archive.read(member))
                target.chmod(0o444)


_PROBE = '''
import importlib, importlib.abc, importlib.metadata, json, pathlib, platform, sys
assert sys.dont_write_bytecode, 'g1_host_probe_requires_explicit_no_bytecode'
class HostImportBoundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'onnxruntime', 'pxr', 'isaaclab',
                'isaacsim', 'PIL', 'yaml', 'packaging'} or fullname == 'blueprint_pipeline.vast_provider_adapter':
            raise ImportError('g1_host_import_forbidden:' + fullname)
sys.meta_path.insert(0, HostImportBoundary())
runtime = pathlib.Path(sys.argv[1])
python = pathlib.Path(sys.prefix).resolve()
sys.path.insert(0, str(runtime))
import numpy
import rfc8785
for name in ('native_g1_team_vm_host', 'native_g1_team_vm_output'):
    importlib.import_module('blueprint_pipeline.' + name)
for name, module in sys.modules.items():
    origin = getattr(module, '__file__', None)
    if origin and name.startswith('blueprint_pipeline'):
        assert pathlib.Path(origin).resolve().is_relative_to(runtime)
assert pathlib.Path(numpy.__file__).resolve().is_relative_to(python)
assert pathlib.Path(rfc8785.__file__).resolve().is_relative_to(runtime) or pathlib.Path(rfc8785.__file__).resolve().is_relative_to(python)
print(json.dumps({'python_version': platform.python_version(), 'implementation': sys.implementation.name,
    'platform': sys.platform, 'machine': platform.machine(), 'numpy_version': numpy.__version__,
    'rfc8785_version': importlib.metadata.version('rfc8785'), 'sealed_module_origins_verified': True}))
'''


def _verify_provider_sources(root: Path, expected_digest: str, commit: str):
    _path(root, directory=True)
    manifest_path = root / "native_g1_team_provider_manifest.json"
    _path(manifest_path, directory=False)
    if manifest_path.stat().st_size > 8 * 1024**2:
        raise ValueError("g1_host_python_provider_manifest_invalid")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if (manifest.get("manifest_digest") != expected_digest or _digest(manifest) != expected_digest
            or manifest.get("implementation_commit") != commit
            or manifest.get("schema_version") != "native_g1_team_provider_bundle.v1"):
        raise ValueError("g1_host_python_provider_manifest_invalid")
    observed = set()
    for row in manifest.get("artifacts", []):
        relative = _relative(row["relative_path"])
        if relative.parts[0] != "provider_runtime" or str(relative) in observed:
            raise ValueError("g1_host_python_provider_source_invalid")
        observed.add(str(relative))
        _asset(root.parent.joinpath(*relative.parts), row)
    if "provider_runtime/blueprint_pipeline/__init__.py" not in observed:
        raise ValueError("g1_host_python_provider_source_invalid")
    # Do not let a foreign unlisted Python source run during the isolated probe.
    for path in root.rglob("*"):
        if path.suffix in {".py", ".pyc", ".so"} and path.relative_to(root.parent).as_posix() not in observed:
            raise ValueError("g1_host_python_provider_source_invalid")


def materialize_g1_vm_host_python(*, package_root: Path, destination_root: Path,
                                  provider_runtime_root: Path, expected_implementation_commit: str,
                                  expected_provider_manifest_digest: str):
    manifest = verify_g1_vm_host_python(package_root, expected_implementation_commit=expected_implementation_commit)
    _verify_provider_sources(provider_runtime_root, expected_provider_manifest_digest, expected_implementation_commit)
    if (not destination_root.is_absolute() or destination_root.resolve() != destination_root
            or destination_root.exists() or not destination_root.parent.is_dir()):
        raise ValueError("g1_host_python_destination_invalid")
    destination_root.mkdir(mode=0o700)
    extract_host_python_archive(package_root / ASSETS["python"]["filename"], destination_root)
    sites = destination_root / "python/lib/python3.12/site-packages"
    sites.mkdir(parents=True, exist_ok=True)
    for role in ("numpy", "rfc8785"):
        extract_host_dependency_wheel(package_root / ASSETS[role]["filename"], sites, ASSETS[role])
    executable = destination_root / "python/bin/python3.12"
    if not executable.is_file() or executable.resolve() != executable:
        raise ValueError("g1_host_python_executable_invalid")
    observed = subprocess.run([str(executable), "-I", "-B", "-c", _PROBE, str(provider_runtime_root)],
        capture_output=True, text=True, check=False, timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"})
    expected = {"python_version": ASSETS["python"]["version"], "implementation": "cpython",
                "platform": "linux", "machine": "x86_64", "numpy_version": ASSETS["numpy"]["version"],
                "rfc8785_version": ASSETS["rfc8785"]["version"], "sealed_module_origins_verified": True}
    if observed.returncode != 0 or _canonical(json.loads(observed.stdout)) != _canonical(expected):
        raise ValueError("g1_host_python_observed_runtime_invalid")
    _verify_provider_sources(provider_runtime_root, expected_provider_manifest_digest, expected_implementation_commit)
    return _persist(destination_root / (RUNTIME_SCHEMA + ".json"), {
        "schema_version": RUNTIME_SCHEMA, "status": "host_python_materialized_not_spend_admitted",
        "source_manifest_digest": manifest["manifest_digest"], "implementation_commit": expected_implementation_commit,
        "provider_manifest_digest": expected_provider_manifest_digest,
        "python_executable_sha256": _sha(executable), "observed_runtime": expected,
        "provider_network_install_required": False, "provider_mutation_performed": False,
        "gpu_runtime_qualified": False, "rights_authorized": False, "claim_ceiling": "development_only",
    }, "receipt_digest")


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--asset-directory", type=Path)
    parser.add_argument("--destination-root", type=Path)
    parser.add_argument("--provider-runtime-root", type=Path)
    parser.add_argument("--provider-manifest-digest")
    args = parser.parse_args(argv)
    if args.asset_directory is not None:
        result = seal_g1_vm_host_python(asset_paths={role: args.asset_directory / row["filename"]
            for role, row in ASSETS.items()}, output_root=args.package_root, implementation_commit=args.implementation_commit)
    elif (args.destination_root is not None and args.provider_runtime_root is not None
          and args.provider_manifest_digest is not None):
        result = materialize_g1_vm_host_python(package_root=args.package_root, destination_root=args.destination_root,
            provider_runtime_root=args.provider_runtime_root, expected_implementation_commit=args.implementation_commit,
            expected_provider_manifest_digest=args.provider_manifest_digest)
    else:
        result = verify_g1_vm_host_python(args.package_root, expected_implementation_commit=args.implementation_commit)
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
