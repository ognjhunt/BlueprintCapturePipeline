from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from blueprint_pipeline.groot_oscar_thin_remote_build_packet import (
    REQUIRED_IMAGE_FILES,
    REQUIRED_ROOT_FILES,
    prepare_remote_build_packet,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    image_root = root / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
    image_root.mkdir(parents=True)
    for filename in REQUIRED_ROOT_FILES:
        (root / filename).write_text(filename + "\n", encoding="utf-8")
    for filename in REQUIRED_IMAGE_FILES:
        (image_root / filename).write_text(filename + "\n", encoding="utf-8")
    package = root / "src/blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "thin_release_image_contract.py").write_text(
        "def build_thin_release_contract(*a, **k): return {'status':'passed'}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_packet_binds_minimal_context_and_exact_build_flow(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/blueprint/foundation:20260714",
        release_ref="registry.example/blueprint/release:20260714",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    assert result["status"] == "ready"
    assert result["provider_launch_performed_by_packet"] is False
    assert result["supported_execution_planes"] == {
        "native_linux_amd64_docker_builder": True,
        "runpod_pod": False,
    }
    script = Path(result["run_script_path"]).read_text(encoding="utf-8")
    assert script.count("docker buildx build") == 2
    assert "FOUNDATION_IMAGE=$foundation_exact" in script
    assert "thin_release_image_contract" in script
    assert 'release.get("required_cuda_version")' in script
    assert '"required_cuda_version":"12.6"' not in script
    subprocess.run(
        ["bash", "-n", result["run_script_path"]],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '+"\\n",encoding="utf-8")' in script
    assert "--push" in script
    assert "hf_token" not in script
    assert "snapshot_download" not in script

    packet = Path(result["packet_dir"])
    context_manifest = json.loads(
        (packet / "context_manifest.json").read_text(encoding="utf-8")
    )
    paths = {row["path"] for row in context_manifest["files"]}
    assert "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile" in paths
    assert "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile" in paths
    assert (
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_oscar_foundation.lock"
        in paths
    )
    assert "src/blueprint_pipeline/thin_release_image_contract.py" in paths
    with tarfile.open(result["tarball_path"], "r:gz") as archive:
        names = set(archive.getnames())
    assert (
        "groot_oscar_thin_remote_build/remote_build_groot_oscar_thin_images.sh"
        in names
    )


def test_packet_refuses_dirty_source_and_unstable_refs(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/foundation:latest",
        release_ref="registry.example/release:dev",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=True,
    )
    assert result["status"] == "blocked"
    assert "remote_release_packet_requires_clean_source_worktree" in result["blockers"]
    assert "foundation_image_ref_refuses_unstable_tag" in result["blockers"]
    assert "release_image_ref_refuses_unstable_tag" in result["blockers"]
    assert Path(result["tarball_path"]).is_file()


def test_packet_refuses_commit_that_only_matches_head_prefix(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    head = _head(root)
    mistyped = head[:8] + ("0" if head[8] != "0" else "1") + head[9:]
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/foundation:versioned",
        release_ref="registry.example/release:versioned",
        source_commit=mistyped,
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
    )
    assert result["status"] == "blocked"
    assert "source_commit_not_exact_head" in result["blockers"]


def test_packet_reports_missing_required_context(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "README.md").unlink()
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/foundation:versioned",
        release_ref="registry.example/release:versioned",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
    )
    assert result["status"] == "blocked"
    assert "remote_context_source_missing:README.md" in result["blockers"]
