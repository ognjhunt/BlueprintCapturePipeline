#!/usr/bin/env python3
"""ADP-009 release artifact writer. Seal upstream + locked installed runtime.

Run the inspected official installer in a separate runtime first. This builder
never calls model APIs and never substitutes Blueprint CAD agents for ARTISAN.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_object_authoring_backend import ARTISAN_BACKEND, ARTISAN_REVISION, ARTISAN_TREE
from blueprint_pipeline.task_object_authoring_contracts import require, file_record
from blueprint_pipeline.task_evaluation_scene_configuration_artisan_runtime import MANIFEST, SCHEMA
from blueprint_pipeline.artisan_vendor_reuse import SITE_IGNORE, vendor_copy_function
from blueprint_pipeline.immutable_file_digest import sha256_file
from blueprint_pipeline.immutable_tree_walk import path_order, walk_tree
from blueprint_pipeline.provision_phase_clock import mark_phase
from scripts.build_task_evaluation_scene_configuration_component_package import build_scene_configuration_component_package

BRIDGE_MODULES = ("asset_authoring_sandbox", "asset_landlock", "task_object_artisan_hooks",
                  "task_object_artisan_sandbox", "task_object_artisan_bridge", "task_object_artisan_metering")


PREFLIGHT_TIMEOUT_SECONDS = 300


def _runtime_preflight(staging: Path) -> None:
    code = '''import importlib,json,pathlib,platform,sys
assert sys.version_info[:2] == (3,12) and platform.system() == 'Linux' and platform.machine() == 'x86_64'
stdlib = [pathlib.Path(p).resolve() for p in sys.path if p]
roots = [pathlib.Path(p).resolve() for p in sys.argv[1:]]
sys.path[:0] = [str(p) for p in roots]
for name in ('sdk','pipeline.compiler','pipeline.agents.design','pipeline.agents.coding','pipeline.agents.review',
             'cadquery','manifold3d','mujoco','coacd','fcl','openai','pxr.Usd'):
    importlib.import_module(name)
from openai.resources.responses.input_tokens import InputTokens
assert callable(InputTokens.count)
for name,module in list(sys.modules.items()):
    origin = getattr(module,'__file__',None)
    if not origin: continue
    path = pathlib.Path(origin).resolve()
    if any(path.is_relative_to(root) for root in roots): continue
    if not {'site-packages','dist-packages'} & set(path.parts) and any(path.is_relative_to(root) for root in stdlib): continue
    raise ImportError('ambient_runtime_import:'+name)
print(json.dumps({'status':'isolated_upstream_imports_passed','python':platform.python_version(),
                  'framework_generation_performed':False}))
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code,
        str(staging / "bridge"), str(staging / "artisan_source"), str(staging / "runtime/site_packages")],
        cwd=staging, env={"PATH": "/usr/bin:/bin"}, check=True, text=True, capture_output=True,
        # A correctness check, not a speed gate: cold imports of a freshly copied
        # runtime without bytecode (-B) took over 60 s on the 4-core control plane
        # under load (2026-10-13); they take about 20-25 s warm.
        timeout=PREFLIGHT_TIMEOUT_SECONDS)
    from json import loads
    require(loads(result.stdout)["status"] == "isolated_upstream_imports_passed", "artisan_import_preflight_invalid")


def missing_root_dependencies(roots, installed) -> list[str]:
    """Pinned roots the installed distributions do not satisfy.

    A root pins a public version. Under PEP 440, ``==2.7.1`` is still satisfied
    by a local label from the package index (torch ``2.7.1+cu128`` from the
    pinned CUDA index); the lock check above compares exact installed versions.
    """
    def name(value):
        return value.strip().lower().replace("_", "-")
    versions = {name(key): str(value) for key, value in installed.items()}
    missing = []
    for item in roots:
        if "==" not in item:
            missing.append(item)
            continue
        pinned_name, pinned_version = item.split("==", 1)
        if versions.get(name(pinned_name), "").split("+", 1)[0] != pinned_version.strip():
            missing.append(item)
    return missing


def _file_record(entry) -> dict:
    """``file_record`` without the path, through the publication's digest cache.

    A vendored file linked from the previous toolchain is an inode its
    readback already read in full; hashing its 10 GB again here was one of
    about ten full passes per deploy.
    """
    require(not entry.is_symlink() and entry.is_file(), "authoring_file_missing_or_symlink")
    return {"sha256": sha256_file(Path(entry.path)), "size_bytes": entry.stat().st_size}


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _seal_staging_files(staging: Path) -> None:
    """Seal owned copies so the package publisher can hard-link their bytes.

    This runs after the isolated import check. Vendor installations are never
    chmodded; the private staging tree is deleted after the package is sealed.
    """
    entries = [entry for _relative, entry in walk_tree(staging)]
    require(not any(entry.is_symlink() for entry in entries), "artisan_staging_symlink_forbidden")
    for entry in entries:
        if entry.is_file():
            current = entry.stat().st_mode
            mode = 0o555 if current & 0o111 else 0o444
            if current & 0o777 != mode:
                os.chmod(entry.path, mode)


def build_artisan_scene_configuration_component(*, repository_root, expected_blueprint_commit,
                                               artisan_root, site_packages_root, blender_root, output_root,
                                               artisan_vendor_reuse_plan=None):
    repository, upstream = Path(repository_root).resolve(), Path(artisan_root).resolve()
    libraries, blender = Path(site_packages_root).resolve(), Path(blender_root).resolve()
    reuse = json.loads(Path(artisan_vendor_reuse_plan).read_text()) if artisan_vendor_reuse_plan else None
    require(_git(repository, "rev-parse", "HEAD") == expected_blueprint_commit
            and not _git(repository, "status", "--porcelain"), "artisan_blueprint_source_not_clean")
    require(_git(upstream, "rev-parse", "HEAD") == ARTISAN_REVISION
            and _git(upstream, "rev-parse", "HEAD^{tree}") == ARTISAN_TREE
            and not _git(upstream, "status", "--porcelain", "--untracked-files=no"),
            "artisan_upstream_not_pinned")
    require(libraries.is_dir() and not libraries.is_symlink(), "artisan_installed_runtime_missing")
    # Validate before metadata discovery or copytree can dereference links.
    for vendor in (libraries, blender):
        require(vendor.is_dir() and all(p.resolve().is_relative_to(vendor)
                for p in vendor.rglob("*") if p.is_symlink()), "artisan_vendor_symlink_escape")
    project = tomllib.loads((upstream / "pyproject.toml").read_text())
    lock = tomllib.loads((upstream / "uv.lock").read_text())
    locked = {(p["name"].replace("_", "-"), p["version"]) for p in lock["package"]}
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version
                 for d in importlib.metadata.distributions(path=[str(libraries)])}
    roots = project["project"]["dependencies"] + project["project"]["optional-dependencies"]["usd"]
    require(all((name, version) in locked for name, version in installed.items()
                if name != "artisan-assets"), "artisan_installed_dependency_not_locked")
    missing = missing_root_dependencies(roots, installed)
    require(not missing, "artisan_installed_dependency_incomplete:" + ",".join(missing[:8]))
    version = subprocess.run([str(blender / "blender"), "--version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    require(version == "Blender 5.0.0", "artisan_blender_version_invalid")
    mark_phase("artisan_vendor_checked")
    with tempfile.TemporaryDirectory(prefix="blueprint-artisan-component-") as raw:
        staging = Path(raw)
        archive = staging / "upstream.zip"
        subprocess.run(["git", "-C", str(upstream), "archive", "--format=zip",
                        "--output=" + str(archive), ARTISAN_REVISION], check=True)
        source = staging / "artisan_source"
        with zipfile.ZipFile(archive) as zipped:
            for row in zipped.infolist():
                relative = Path(row.filename)
                require(not relative.is_absolute() and ".." not in relative.parts,
                        "artisan_source_archive_path_invalid")
            zipped.extractall(source)
        archive.unlink()
        patch = repository / "patches/artisan/0001-confine-generated-code.patch"
        patch_hash = file_record(patch)["sha256"]
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
        shutil.copyfile(patch, staging / "upstream.patch")
        mark_phase("artisan_source_unpacked")
        site_copy = vendor_copy_function(reuse, kind="site_packages", source_root=libraries) if reuse else None
        shutil.copytree(libraries, staging / "runtime/site_packages", ignore=SITE_IGNORE,
                        **({"copy_function": site_copy} if site_copy else {}))
        mark_phase("artisan_site_packages_linked")
        # Reject source/runtime links to private or ambient files; flatten only
        # in-root vendor links, which the immutable component publisher seals.
        blender_copy = vendor_copy_function(reuse, kind="blender", source_root=blender) if reuse else None
        shutil.copytree(blender, staging / "runtime/blender",
                        **({"copy_function": blender_copy} if blender_copy else {}))
        mark_phase("artisan_blender_linked")
        if reuse:
            site_copy.require_complete()
            blender_copy.require_complete()
        bridge = staging / "bridge/blueprint_pipeline"
        bridge.mkdir(parents=True)
        (bridge / "__init__.py").write_text("")
        for module in BRIDGE_MODULES:
            shutil.copyfile(repository / f"src/blueprint_pipeline/{module}.py", bridge / (module + ".py"))
        _runtime_preflight(staging)
        mark_phase("artisan_import_preflight")
        driver = staging / "run"
        shutil.copyfile(repository / "scripts/run_task_evaluation_scene_configuration_content_agents_component.sh", driver)
        driver.chmod(0o755)
        files = [{"relative_path": relative, **_file_record(entry)}
                 for relative, entry in sorted(walk_tree(staging), key=lambda item: path_order(item[0]))
                 if entry.is_file()]
        value = {"schema_version": SCHEMA, "backend": ARTISAN_BACKEND,
                 "upstream_revision": ARTISAN_REVISION, "upstream_tree": ARTISAN_TREE,
                 "lock_sha256": file_record(source / "uv.lock")["sha256"],
                 "patch_sha256": patch_hash, "python": ">=3.12,<3.13",
                 "blender_version": "5.0.0", "installed_dependencies": installed,
                 "files": files, "network_install_required": False,
                 "controller_and_generated_execution_separate": True,
                 "claim_ceiling": "development_only"}
        value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
        (staging / MANIFEST).write_text(canonical_json(value) + "\n")
        _seal_staging_files(staging)
        mark_phase("artisan_manifest_and_seal")
        return build_scene_configuration_component_package(adapter_id="content_agents_rigid_replacement",
            source_root=staging, driver_entrypoint="run", source_repository="https://github.com/ognjhunt/BlueprintCapturePipeline",
            source_commit=expected_blueprint_commit, source_license="Blueprint integration; ARTISAN Apache-2.0; upstream dependency notices",
            output_root=output_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repository-root", "expected-blueprint-commit", "artisan-root", "site-packages-root", "blender-root", "output-root"):
        parser.add_argument("--" + flag, required=True)
    build_artisan_scene_configuration_component(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
