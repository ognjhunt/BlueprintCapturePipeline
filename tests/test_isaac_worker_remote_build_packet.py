from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.isaac_worker_remote_build_packet import (
    DEFAULT_BASE_IMAGE_REF,
    DEFAULT_IMAGE_REF,
    prepare_remote_build_packet,
    validate_isaac_worker_archive,
)


def test_isaac_worker_packet_binds_exact_context_and_clean_identity(
    tmp_path: Path,
) -> None:
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
        image_ref=DEFAULT_IMAGE_REF,
        base_image_ref=DEFAULT_BASE_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=True,
        generated_at="2026-07-29T12:00:00Z",
    )

    assert result["packet_kind"] == "isaac_worker_image"
    assert result["status"] == "blocked"
    assert "isaac_worker_packet_requires_clean_source_worktree" in result["blockers"]
    assert result["source_dirty_patch_sha256"] == (
        CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256
    )
    assert validate_isaac_worker_archive(result) == []
    with tarfile.open(result["tarball_path"], "r:gz") as archive:
        names = archive.getnames()
        script = archive.getmember(
            "isaac_worker_remote_build/remote_build_isaac_worker_image.sh"
        )
    assert names == result["archive_members"]
    assert script.mode & 0o111
    assert (
        "isaac_worker_remote_build/context/deploy/docker/robot_eval_worker/isaac/Dockerfile"
        in names
    )
    assert any(
        name.endswith("/src/blueprint_pipeline/nvidia_warehouse_native_camera_canary.py")
        for name in names
    )
    assert not any(".egg-info/" in name for name in names)


