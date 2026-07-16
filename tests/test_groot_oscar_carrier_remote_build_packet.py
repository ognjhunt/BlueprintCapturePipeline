from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from blueprint_pipeline.groot_oscar_carrier_remote_build_packet import (
    DEFAULT_BASE_IMAGE,
    prepare_remote_build_packet,
)
from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    validate_carrier_image_archive,
)


def test_carrier_packet_is_clean_source_bound_and_allocator_typed(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = prepare_remote_build_packet(
        output_dir=tmp_path,
        repo_root=repo,
        image_ref="docker.io/example/carrier:versioned",
        base_image_ref=DEFAULT_BASE_IMAGE,
        source_commit=head,
        source_worktree_dirty=True,
        generated_at="2026-07-16T12:00:00Z",
    )

    assert result["packet_kind"] == "carrier_image"
    assert result["status"] == "blocked"
    assert "carrier_packet_requires_clean_source_worktree" in result["blockers"]
    assert len(result["carrier_dockerfile_sha256"]) == 64
    with tarfile.open(result["tarball_path"], "r:gz") as archive:
        names = archive.getnames()
    assert names == result["archive_members"]
    assert "groot_oscar_carrier_remote_build/context/Dockerfile" in names
    assert validate_carrier_image_archive(result) == []
    script = Path(result["run_script_path"]).read_text(encoding="utf-8")
    assert "docker buildx build --platform linux/amd64" in script
    assert "--push" in script
    assert "groot_oscar_carrier_remote_build_result.v1" in script
    assert "resolved_digest_ref" in script


def test_carrier_packet_rejects_unpinned_base_and_untagged_output(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = prepare_remote_build_packet(
        output_dir=tmp_path,
        repo_root=repo,
        image_ref="docker.io/example/carrier",
        base_image_ref="pytorch/pytorch:latest",
        source_commit="a" * 40,
        source_worktree_dirty=True,
    )
    assert "carrier_base_image_not_digest_pinned" in result["blockers"]
    assert "carrier_image_ref_not_versioned" in result["blockers"]


def test_carrier_packet_checks_cleanliness_before_writing_inside_repo(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    dockerfile = repo / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Carrier.Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("ARG PYTORCH_CARRIER_BASE\nFROM ${PYTORCH_CARRIER_BASE}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Packet Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "packet-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = prepare_remote_build_packet(
        output_dir=repo / "generated-packet",
        repo_root=repo,
        image_ref="docker.io/example/carrier:versioned",
        base_image_ref=DEFAULT_BASE_IMAGE,
        source_commit=head,
        source_worktree_dirty=False,
    )

    assert result["status"] == "ready"
    assert result["blockers"] == []


def test_carrier_packet_missing_dockerfile_emits_blocked_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Packet Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "packet-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("fixture\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repo, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    result = prepare_remote_build_packet(
        output_dir=tmp_path / "packet",
        repo_root=repo,
        image_ref="docker.io/example/carrier:versioned",
        base_image_ref=DEFAULT_BASE_IMAGE,
        source_commit=head,
        source_worktree_dirty=False,
    )

    assert result["status"] == "blocked"
    assert "carrier_dockerfile_missing" in result["blockers"]
    assert "groot_oscar_carrier_remote_build/context/Dockerfile" not in result[
        "archive_members"
    ]
    assert Path(result["tarball_path"]).is_file()
