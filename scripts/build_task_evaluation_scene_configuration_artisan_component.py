#!/usr/bin/env python3
"""ADP-009 release artifact writer. Seal upstream + locked installed runtime.

Run the inspected official installer in a separate runtime first. This builder
never calls model APIs and never substitutes Blueprint CAD agents for ARTISAN.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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
from scripts.build_task_evaluation_scene_configuration_component_package import build_scene_configuration_component_package

BRIDGE_MODULES = ("asset_authoring_sandbox", "asset_landlock", "task_object_artisan_hooks",
                  "task_object_artisan_sandbox", "task_object_artisan_bridge", "task_object_artisan_metering")


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
        cwd=staging, env={"PATH": "/usr/bin:/bin"}, check=True, text=True, capture_output=True, timeout=60)
    from json import loads
    require(loads(result.stdout)["status"] == "isolated_upstream_imports_passed", "artisan_import_preflight_invalid")


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def build_artisan_scene_configuration_component(*, repository_root, expected_blueprint_commit,
                                               artisan_root, site_packages_root, blender_root, output_root):
    repository, upstream = Path(repository_root).resolve(), Path(artisan_root).resolve()
    libraries, blender = Path(site_packages_root).resolve(), Path(blender_root).resolve()
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
    require(all("==" in item and installed.get(item.split("==")[0]) == item.split("==")[1]
                for item in roots), "artisan_installed_dependency_incomplete")
    version = subprocess.run([str(blender / "blender"), "--version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    require(version == "Blender 5.0.0", "artisan_blender_version_invalid")
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
        shutil.copytree(libraries, staging / "runtime/site_packages", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pth", "__editable__*", "artisan_assets-*.dist-info"))
        # Reject source/runtime links to private or ambient files; flatten only
        # in-root vendor links, which the immutable component publisher seals.
        shutil.copytree(blender, staging / "runtime/blender")
        bridge = staging / "bridge/blueprint_pipeline"
        bridge.mkdir(parents=True)
        (bridge / "__init__.py").write_text("")
        for module in BRIDGE_MODULES:
            shutil.copyfile(repository / f"src/blueprint_pipeline/{module}.py", bridge / (module + ".py"))
        _runtime_preflight(staging)
        driver = staging / "run"
        shutil.copyfile(repository / "scripts/run_task_evaluation_scene_configuration_content_agents_component.sh", driver)
        driver.chmod(0o755)
        files = [{"relative_path": p.relative_to(staging).as_posix(),
                  **{k: v for k, v in file_record(p).items() if k != "path"}}
                 for p in sorted(staging.rglob("*")) if p.is_file()]
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
