from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from blueprint_pipeline.groot_oscar_carrier_remote_build_packet import (
    DEFAULT_BASE_IMAGE,
    prepare_remote_build_packet,
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
    assert "groot_oscar_carrier_remote_build/context/Dockerfile" in names
    script = Path(result["run_script_path"]).read_text(encoding="utf-8")
    assert "docker buildx build --platform linux/amd64" in script
    assert "--push" in script
    assert "groot_oscar_carrier_remote_build_result.v1" in script
    assert "resolved_digest_ref" in script


def test_carrier_packet_rejects_unpinned_base(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    result = prepare_remote_build_packet(
        output_dir=tmp_path,
        repo_root=repo,
        image_ref="docker.io/example/carrier:versioned",
        base_image_ref="pytorch/pytorch:latest",
        source_commit="a" * 40,
        source_worktree_dirty=True,
    )
    assert "carrier_base_image_not_digest_pinned" in result["blockers"]
