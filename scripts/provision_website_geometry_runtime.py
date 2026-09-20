"""ADP-009B/day14: build the website geometry worker during exact-release deployment."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Callable

from blueprint_pipeline.website_geometry_dispatch import load_profile


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _build_wheel(repository: Path, commit: str, destination: Path) -> Path:
    # Build the committed tree, never dirty files or a previous release's wheel.
    with tempfile.TemporaryDirectory(prefix="website-worker-source-") as temporary:
        root = Path(temporary)
        archive = root / "source.tar"
        subprocess.run(["git", "-C", str(repository), "archive", commit,
                        "--output", str(archive), "--", "pyproject.toml", "README.md", "LICENSE", "src"],
                       check=True, capture_output=True, timeout=60)
        source = root / "source"
        source.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(source, filter="data")
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                        "--no-build-isolation", "--no-index", "--wheel-dir", str(destination),
                        str(source)], check=True, capture_output=True, timeout=300)
    wheels = list(destination.glob("blueprint_capture_pipeline-*.whl"))
    if len(wheels) != 1:
        raise ValueError("website_geometry_release_wheel_missing")
    return wheels[0]


def provision_website_geometry_runtime(*, repository_root: Path, source_commit: str,
        runtime_root: Path, template_path: Path, readback: Callable[[Path], bytes]) -> Path:
    """Publish/reopen one release profile from three operator-pinned runtime inputs.

    This builds CPU deployment artifacts only. It neither launches a GPU nor
    creates scene-specific outputs or grants spending authority.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("website_geometry_release_commit_invalid")
    template = json.loads(template_path.read_text())
    if template.get("schema_version") != "website_mapanything_deployment.v1":
        raise ValueError("website_geometry_deployment_template_invalid")
    rows = template.get("runtime_files", [])
    if (len(rows) != 3 or len({Path(row["path"]).name for row in rows}) != 3
            or sum(Path(row["path"]).suffix == ".whl" for row in rows) != 2
            or any(Path(row["path"]).name.startswith("blueprint_capture_pipeline-") for row in rows)):
        raise ValueError("website_geometry_deployment_inputs_invalid")
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or _digest(path) != row["digest"]:
            raise ValueError("website_geometry_deployment_input_changed")
    template_digest = _digest(template_path)
    parent = runtime_root / "website-mapanything"
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / source_commit
    profile_path = root / "profile.json"
    if root.exists():
        value = load_profile(source_commit=source_commit, profile_path=profile_path)
        if value.get("deployment_template_digest") != template_digest:
            raise ValueError("website_geometry_existing_runtime_configuration_changed")
    else:
        with tempfile.TemporaryDirectory(prefix=".website-runtime-", dir=parent) as temporary:
            staging = Path(temporary)
            wheel = _build_wheel(repository_root, source_commit, staging)
            for row in rows:
                copied = staging / Path(row["path"]).name
                shutil.copyfile(row["path"], copied)
                if _digest(copied) != row["digest"]:
                    raise ValueError("website_geometry_deployment_input_changed")
            files = [wheel, *(staging / Path(row["path"]).name for row in rows)]
            value = {key: template[key] for key in (
                "worker_image_digest", "maximum_cost_usd", "max_hourly_rate_usd",
                "hard_ttl_seconds", "minimum_gpu_ram_mb")}
            value.update(schema_version="website_mapanything_runtime.v1", source_commit=source_commit,
                         deployment_template_digest=template_digest,
                         runtime_files=[{"path": str(path), "digest": _digest(path)} for path in files])
            staged_profile = staging / "profile.json"
            staged_profile.write_text(json.dumps(value, sort_keys=True) + "\n")
            load_profile(source_commit=source_commit, profile_path=staged_profile)
            for row in value["runtime_files"]:
                row["path"] = str(root / Path(row["path"]).name)
            staged_profile.write_text(json.dumps(value, sort_keys=True) + "\n")
            for file in staging.iterdir():
                file.chmod(0o444)
            staging.chmod(0o755)
            staging.rename(root)
    # Read using the service identity, including every sealed worker input.
    for row in value["runtime_files"]:
        if "sha256:" + hashlib.sha256(readback(Path(row["path"]))).hexdigest() != row["digest"]:
            raise ValueError("website_geometry_service_readback_failed")
    if readback(profile_path) != profile_path.read_bytes():
        raise ValueError("website_geometry_profile_readback_failed")
    return profile_path
