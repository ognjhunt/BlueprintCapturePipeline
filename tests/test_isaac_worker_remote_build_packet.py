from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

from blueprint_pipeline.g1_kitchen_bundle_compatibility import (
    CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
)
from blueprint_pipeline.groot_oscar_digitalocean_builder import build_cloud_init
from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    MIN_BUILD_FREE_BYTES,
    build_build_plane_admission,
)
from blueprint_pipeline.image_build_result_verification import (
    validate_remote_isaac_worker_result,
)
from blueprint_pipeline.isaac_worker_image_manifest import (
    SCHEMA_VERSION as ISAAC_IMAGE_MANIFEST_SCHEMA,
)
from blueprint_pipeline.isaac_worker_remote_build_packet import (
    DEFAULT_BASE_IMAGE_REF,
    DEFAULT_IMAGE_REF,
    RESULT_NAME,
    prepare_remote_build_packet,
    validate_isaac_worker_archive,
)


def _builder(source_commit: str) -> dict:
    return {
        "provider": "digitalocean",
        "purpose": "image_build",
        "platform": "linux/amd64",
        "docker_daemon_verified": True,
        "docker_buildx_verified": True,
        "free_disk_bytes": MIN_BUILD_FREE_BYTES,
        "registry_push_auth_file_verified": True,
        "independent_teardown_watchdog": True,
        "ssh_host_key_sha256": "SHA256:" + "a" * 43,
        "ssh_host_key_independently_verified": True,
        "ssh_host_key_verification_method": "launch_bound_client_generated_key",
        "expected_source_commit": source_commit,
    }


def _spend() -> dict:
    return {
        "paid_mutation_authorized": True,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 7200,
        "one_resource_limit": True,
    }


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


def test_isaac_worker_packet_admission_rejects_tampered_binding(
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
    packet = prepare_remote_build_packet(
        output_dir=tmp_path,
        repo_root=repo,
        image_ref=DEFAULT_IMAGE_REF,
        base_image_ref=DEFAULT_BASE_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=True,
    )
    packet.update(status="ready", blockers=[], source_worktree_dirty=False)

    admitted = build_build_plane_admission(
        packet=packet, builder=_builder(head), spend=_spend()
    )
    assert admitted["status"] == "admitted"
    assert admitted["checks"]["packet_kind"] == "isaac_worker_image"

    packet["context_manifest_sha256"] = "0" * 64
    blocked = build_build_plane_admission(
        packet=packet, builder=_builder(head), spend=_spend()
    )
    assert blocked["status"] == "blocked"
    assert "builder_isaac_archive_script_binding_mismatch" in blocked["blockers"]


def test_isaac_worker_cloud_init_uses_governed_docker_builder() -> None:
    text = build_cloud_init(
        host_private_b64="private",
        host_public_b64="public",
        shutdown_minutes=120,
        packet_kind="isaac_worker_image",
    )
    assert "docker-buildx" in text
    assert "/swapfile" in text
    assert text.index("mkswap") < text.index("touch /root/blueprint-builder-ready")


def test_remote_isaac_result_binds_digest_and_worker_identity(tmp_path: Path) -> None:
    packet = {
        "image_ref": "docker.io/example/isaac-worker:versioned",
        "source_commit": "a" * 40,
        "source_dirty_patch_sha256": CANONICAL_CLEAN_SOURCE_DIRTY_PATCH_SHA256,
    }
    payload = {
        "schema_version": ISAAC_IMAGE_MANIFEST_SCHEMA,
        "status": "completed",
        "blockers": [],
        "resolved_digest_ref": "docker.io/example/isaac-worker@sha256:" + "d" * 64,
        "runnable_platform": "linux/amd64",
        "raw_secret_values_recorded": False,
        "worker_build_identity": {
            "status": "verified",
            "blockers": [],
            "source_commit": packet["source_commit"],
            "source_dirty_patch_sha256": packet["source_dirty_patch_sha256"],
            "worker_image_family": "isaac-eval-worker",
            "isaac_sim_major_version": 6,
        },
    }
    (tmp_path / RESULT_NAME).write_text(json.dumps(payload), encoding="utf-8")

    verified = validate_remote_isaac_worker_result(tmp_path, packet=packet)
    assert verified["status"] == "verified"
    assert verified["resolved_digest_ref"] == payload["resolved_digest_ref"]

    payload["worker_build_identity"]["source_commit"] = "f" * 40
    (tmp_path / RESULT_NAME).write_text(json.dumps(payload), encoding="utf-8")
    blocked = validate_remote_isaac_worker_result(tmp_path, packet=packet)
    assert blocked["status"] == "blocked"
    assert "isaac_remote_build_source_commit_mismatch" in blocked["blockers"]
