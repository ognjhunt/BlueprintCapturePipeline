"""Opt-in Isaac Lab 3.0 GA source profile on NVIDIA's immutable full Kit image.

No production default changes. The image supplies compiled dependencies; exact
released source blobs supply Python and Kit files. The runtime still uses Isaac
Sim PhysX and the existing RTX camera path. Standalone/Newton are separate work.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from types import SimpleNamespace
from typing import Any, Mapping
import zipfile

from .decision_evidence_contracts import canonical_digest
from .native_task_ga_runtime_lock import GA_SUPPLEMENTAL_WHEELS, ARENA_LOCK_SHA256

PROFILE = "isaaclab_3_ga"
IMAGE = "nvcr.io/nvidia/isaac-lab:3.0.0@sha256:88f1d5c1b588299897eeddc5c7f6376d11d1bfb8224896ae288ea13bf6e8f1e2"
COMMIT = "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4"
TREE = "2cf60bb84972c569c1a23f3f2ec67cac0ca5cf8c"
ARENA_COMMIT = "b5708b72b21f37f74c8c66b1ee9cedff362a424b"
ARENA_TREE = "f8517294d636668dd8cbafb50912d3de2cff9d6a"
PYTHON = "/opt/isaaclab-venv/bin/python"
LAUNCHER = "/workspace/isaaclab/isaaclab.sh"
PROBES = (
    {"module": "warp", "expected_version": "1.17.0"},
    {"module": "torch", "expected_version": "2.11.0+cu130", "cuda_required": True,
     "expected_cuda_version": "13.0"},
    {"module": "msgpack", "expected_version": "1.1.0"},
)


def materialize_ga_source_packet(*, output_dir: str | Path, isaaclab_repo: str | Path,
                                arena_repo: str | Path, dependency_wheel_dir: str | Path,
                                generated_at: str | None = None) -> dict[str, Any]:
    from .native_task_runtime_source_packet import (
        ISAACLAB_PACKAGE_NAMES, RUNTIME_EXPERIENCE_RELATIVE_PATH,
        SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION, _repository_rows, _sha256_file,
        _utc_now_iso, _write_json, _runtime_dependency_rows,
    )
    root = Path(output_dir).resolve()
    if root.exists():
        raise ValueError("ga_source_packet_output_exists")
    wheel_dir = Path(dependency_wheel_dir).resolve()
    for row in GA_SUPPLEMENTAL_WHEELS:
        path = wheel_dir / row["filename"]
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            raise ValueError("ga_supplemental_wheel_identity_mismatch:" + row["package"])
    dependency_rows, dependency_files = _runtime_dependency_rows(wheel_dir, GA_SUPPLEMENTAL_WHEELS)
    lab, lab_files = _repository_rows(repo=Path(isaaclab_repo).resolve(),
        repository="https://github.com/isaac-sim/IsaacLab.git", commit=COMMIT, expected_tree=TREE,
        license_id="BSD-3-Clause", license_path="LICENSE", archive_namespace="isaaclab",
        prefixes=("LICENSE", "pyproject.toml", "uv.lock", "apps", "source"))
    arena_names = [p.name for p in Path(arena_repo).iterdir() if p.name.startswith("isaaclab_arena") and p.is_dir()]
    arena, arena_files = _repository_rows(repo=Path(arena_repo).resolve(),
        repository="https://github.com/isaac-sim/IsaacLab-Arena.git", commit=ARENA_COMMIT,
        expected_tree=ARENA_TREE, license_id="Apache-2.0", license_path="LICENSE.md",
        archive_namespace="arena", prefixes=("LICENSE.md", "pyproject.toml", "uv.lock", *arena_names))
    if next(row["sha256"] for row in arena["files"] if row["source_path"] == "uv.lock") != ARENA_LOCK_SHA256:
        raise ValueError("ga_arena_dependency_lock_mismatch")
    roots = [f"runtime_sources/isaaclab/source/{name}" for name in (*ISAACLAB_PACKAGE_NAMES, "isaaclab_ppisp")]
    roots.append("runtime_sources/arena")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "generated_at": generated_at or _utc_now_iso(),
        "status": "ready", "runtime_profile": PROFILE, "container_image": IMAGE,
        "provisioning_mode": "immutable_image_dependencies_verified_source_roots",
        "repositories": [lab, arena], "install_roots": roots, "runtime_dependency_wheels": dependency_rows,
        "runtime_dependency_basis": {"package": "warp-lang", "version": "1.17.0", "container_image": IMAGE},
        "torch_runtime_dependency_basis": {"package": "torch", "version": "2.11.0+cu130", "container_image": IMAGE},
        "runtime_experience": {
            "relative_path": RUNTIME_EXPERIENCE_RELATIVE_PATH,
            "repository": lab["repository"], "source_revision": COMMIT, "source_tree": TREE,
            "upstream_fix_revisions": [COMMIT],
            "sha256": next(row["sha256"] for row in lab["files"] if row["archive_path"] == RUNTIME_EXPERIENCE_RELATIVE_PATH)},
        "source_file_count": lab["file_count"] + arena["file_count"], "released_source_only": True,
        "scene_bytes_included": False, "policy_bytes_included": False, "redistribution_permitted": True,
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    root.mkdir(parents=True)
    archive_path = root / "native_task_runtime_sources.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted([("native_task_runtime_source_manifest.v1.json",
                                  (json.dumps(manifest, sort_keys=True, indent=2)+"\n").encode()),
                                 *lab_files, *arena_files]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        for name, path in dependency_files:
            archive.write(path, name)
    receipt = {**manifest, "schema_version": SCHEMA_VERSION,
               "repositories": [{k: row[k] for k in ("repository", "commit", "tree", "license")}
                                for row in (lab, arena)],
               "packet_path": str(archive_path), "packet_size_bytes": archive_path.stat().st_size,
               "packet_sha256": _sha256_file(archive_path)}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_json(root / "native_task_runtime_source_packet.v1.json", receipt)
    return receipt


def provision_ga_sources(*, verified: Mapping[str, Any], packet_path: Path,
                         extraction_dir: Path, output_path: Path,
                         run_command: Any = subprocess.run) -> dict[str, Any]:
    from .native_task_runtime_source_provision import (
        _persist, _sha256, _runtime_import_probe_source, _extract_runtime_dependency_wheels,
    )
    result: dict[str, Any] = {"schema_version": "native_task_runtime_source_provisioning.v1",
        "status": "blocked", "runtime_profile": PROFILE, "container_image": IMAGE,
        "source_receipt_digest": verified["receipt_digest"], "extraction_dir": str(extraction_dir),
        "python_executable": PYTHON, "installation_method": "verified_source_roots_on_pinned_image",
        "candidate_policy_queried": False, "scene_bytes_processed": False, "blockers": []}
    try:
        if (verified.get("container_image") != IMAGE
                or verified.get("provisioning_mode") != "immutable_image_dependencies_verified_source_roots"
                or [(r["commit"], r["tree"]) for r in verified["repositories"]] != [(COMMIT, TREE), (ARENA_COMMIT, ARENA_TREE)]
                or sys.executable != PYTHON):
            raise ValueError("ga_runtime_source_or_interpreter_identity_mismatch")
        # The enclosing source verifier checks every archive member and path.
        extraction_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(packet_path) as archive:
            archive.extractall(extraction_dir)
        expected_wheels = {row["filename"]: row["sha256"] for row in GA_SUPPLEMENTAL_WHEELS}
        if {row["filename"]: row["sha256"] for row in verified["runtime_dependency_wheels"]} != expected_wheels:
            raise ValueError("ga_supplemental_dependency_closure_mismatch")
        from packaging.tags import sys_tags
        dependencies = extraction_dir / "ga_runtime_dependencies"
        result["installed_runtime_dependencies"] = _extract_runtime_dependency_wheels(
            extraction_root=extraction_dir, wheel_rows=verified["runtime_dependency_wheels"],
            destination=dependencies, runtime_python_tag="cp312",
            runtime_platform_tags=tuple({tag.platform for tag in sys_tags()}))
        source_roots = [extraction_dir / row for row in verified["install_roots"]]
        if any(not root.is_dir() for root in source_roots):
            raise ValueError("ga_runtime_source_roots_missing")
        lab = extraction_dir / "runtime_sources/isaaclab"
        link = lab / "_isaac_sim"
        if os.path.lexists(link):
            if not link.is_symlink() or link.resolve() != Path("/isaac-sim"):
                raise ValueError("ga_runtime_simulator_link_conflict")
        else:
            link.symlink_to("/isaac-sim", target_is_directory=True)
        site_path = Path(sysconfig.get_paths()["purelib"]) / "blueprint_native_task_runtime_sources.pth"
        site_path.write_text(f"import sys;sys.path[:0]={[str(p) for p in [*source_roots, dependencies]]!r}\n")
        experience = extraction_dir / verified["runtime_experience"]["relative_path"]
        if _sha256(experience) != verified["runtime_experience"]["sha256"]:
            raise ValueError("ga_runtime_experience_mismatch")
        result["runtime_experience"] = {**verified["runtime_experience"], "path": str(experience)}
        probe = run_command([PYTHON, "-P", "-c", _runtime_import_probe_source(PROBES)],
                            check=False, capture_output=True, text=True)
        result["runtime_import_probe_returncode"] = probe.returncode
        result["runtime_import_probe_stderr"] = probe.stderr[-16000:]
        result["runtime_import_probes"] = json.loads(probe.stdout)
        if probe.returncode != 0:
            raise ValueError("ga_runtime_dependency_probe_failed")
        result.update(status="completed", install_roots=[str(p) for p in source_roots],
                      source_packages_made_importable=True, all_sources_verified_before_install=True,
                      path_file=str(site_path), path_file_sha256=_sha256(site_path))
    except Exception as exc:
        result["blockers"].append(str(exc))
    return _persist(output_path, result)


def verify_ga_launch(provisioning: Mapping[str, Any]) -> dict[str, Any]:
    from .native_task_isaaclab_launch import NativeTaskIsaacLabLaunchError, _sha256
    experience = provisioning.get("runtime_experience") or {}
    rows = provisioning.get("runtime_import_probes") or []
    errors = []
    if (provisioning.get("schema_version") != "native_task_runtime_source_provisioning.v1"
            or provisioning.get("status") != "completed"
            or provisioning.get("receipt_digest") != canonical_digest(provisioning, digest_field="receipt_digest")
            or provisioning.get("container_image") != IMAGE or provisioning.get("python_executable") != PYTHON
            or experience.get("source_revision") != COMMIT or experience.get("source_tree") != TREE
            or provisioning.get("runtime_import_probe_returncode") != 0):
        errors.append("ga_runtime_provisioning_invalid")
    installed = {row.get("package"): row.get("wheel_sha256")
                 for row in provisioning.get("installed_runtime_dependencies", [])}
    if installed != {row["package"]: row["sha256"] for row in GA_SUPPLEMENTAL_WHEELS}:
        errors.append("ga_runtime_supplemental_dependencies_invalid")
    root = Path(str(provisioning.get("extraction_dir") or "")).resolve()
    path = Path(str(experience.get("path") or "")).resolve()
    if (not path.is_relative_to(root) or not path.is_file()
            or _sha256(path) != experience.get("sha256")):
        errors.append("ga_runtime_experience_identity_invalid")
    for contract in PROBES:
        matched = [row for row in rows if row.get("module") == contract["module"]]
        if (len(matched) != 1 or matched[0].get("observed_version") != contract["expected_version"]
                or matched[0].get("version_matches") is not True):
            errors.append("ga_runtime_dependency_identity_invalid:" + contract["module"])
        elif contract.get("cuda_required") and (matched[0].get("cuda_tensor_operation_passed") is not True
                or matched[0].get("observed_cuda_version") != contract["expected_cuda_version"]):
            errors.append("ga_runtime_cuda_unqualified")
    if errors:
        raise NativeTaskIsaacLabLaunchError(errors)
    return {"schema_version": "native_task_isaaclab_launch.v1", "status": "qualified",
            "runtime_profile": PROFILE, "container_image": IMAGE, "experience": dict(experience),
            "runtime_import_probes": rows, "device_coherence_still_requires_native_readback": True}


def ga_app_launcher(**arguments: Any) -> Any:
    from isaaclab.app import launch_simulation
    from isaaclab_physx.physics import PhysxCfg
    context = launch_simulation(PhysxCfg(), {**arguments, "physics": "isaacsim_physx", "require_kit": True})
    context.__enter__()
    import omni.kit.app
    app = omni.kit.app.get_app()

    class App:
        def __getattr__(self, name: str) -> Any:
            return getattr(app, name)

        def close(self, *args: Any, exit_code: int = 0, **kwargs: Any) -> None:
            context.__exit__(SystemExit, SystemExit(exit_code), None)

        def is_exiting(self) -> bool:
            return not app.is_running()

    return SimpleNamespace(app=App())
