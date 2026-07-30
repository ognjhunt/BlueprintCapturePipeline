from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

from blueprint_pipeline.groot_oscar_digitalocean_builder import (
    build_cloud_init,
    validate_remote_openpi_result,
)
from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    MIN_BUILD_FREE_BYTES,
    build_build_plane_admission,
)
from blueprint_pipeline.openpi_policy_ranking_remote_build_packet import (
    CTRL_WORLD_IMAGE_VARIANT,
    CTRL_WORLD_OSCAR_IMAGE_VARIANT,
    DEFAULT_CTRL_WORLD_IMAGE_REF,
    DEFAULT_CTRL_WORLD_OSCAR_IMAGE_REF,
    DEFAULT_IMAGE_REF,
    MENAGERIE_REVISION,
    OPENPI_REVISION,
    prepare_remote_build_packet,
    validate_openpi_policy_ranking_archive,
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
        "max_spend_usd": 2.0,
        "hard_ttl_seconds": 7200,
        "one_resource_limit": True,
    }


def test_openpi_packet_binds_full_context_and_executable(tmp_path: Path) -> None:
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
        source_commit=head,
        source_worktree_dirty=True,
        generated_at="2026-07-26T12:00:00Z",
    )

    assert result["packet_kind"] == "openpi_policy_ranking_image"
    assert result["status"] == "blocked"
    assert "openpi_packet_requires_clean_source_worktree" in result["blockers"]
    assert validate_openpi_policy_ranking_archive(result) == []
    with tarfile.open(result["tarball_path"], "r:gz") as archive:
        names = archive.getnames()
        script = archive.getmember(
            "openpi_policy_ranking_remote_build/remote_build_openpi_policy_ranking_image.sh"
        )
    assert names == result["archive_members"]
    assert script.mode & 0o111
    assert any(
        name.endswith("/src/blueprint_pipeline/openpi_policy_ranking_gpu_bootstrap.py")
        for name in names
    )
    assert not any(".egg-info/" in name for name in names)
    assert "openpi_policy_ranking_remote_build/context/src/AGENTS.md" not in names
    assert result["openpi_revision"] == OPENPI_REVISION
    assert result["menagerie_revision"] == MENAGERIE_REVISION


def test_ctrl_world_variant_binds_combined_dockerfile_and_runtime_lock(tmp_path: Path) -> None:
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
        image_ref=DEFAULT_CTRL_WORLD_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=True,
        image_variant=CTRL_WORLD_IMAGE_VARIANT,
    )

    assert result["status"] == "blocked"
    assert result["image_variant"] == CTRL_WORLD_IMAGE_VARIANT
    assert result["dockerfile_relative_path"] == (
        "deploy/docker/policy_ranking_openpi_ctrl_world/Dockerfile"
    )
    assert validate_openpi_policy_ranking_archive(result) == []
    assert any(
        name.endswith("/policy_ranking_openpi_ctrl_world/requirements.lock")
        for name in result["archive_members"]
    )
    assert any(
        name.endswith("/policy_ranking_openpi_ctrl_world/ctrl_world_source_manifest.json")
        for name in result["archive_members"]
    )


def test_ctrl_world_oscar_variant_binds_three_runtime_dockerfile(tmp_path: Path) -> None:
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
        image_ref=DEFAULT_CTRL_WORLD_OSCAR_IMAGE_REF,
        source_commit=head,
        source_worktree_dirty=True,
        image_variant=CTRL_WORLD_OSCAR_IMAGE_VARIANT,
    )

    assert result["status"] == "blocked"
    assert result["image_variant"] == CTRL_WORLD_OSCAR_IMAGE_VARIANT
    assert result["dockerfile_relative_path"] == (
        "deploy/docker/policy_ranking_openpi_ctrl_world_oscar/Dockerfile"
    )
    assert validate_openpi_policy_ranking_archive(result) == []
    with tarfile.open(result["tarball_path"], "r:gz") as archive:
        dockerfile = archive.extractfile(
            "openpi_policy_ranking_remote_build/context/"
            "deploy/docker/policy_ranking_openpi_ctrl_world_oscar/Dockerfile"
        )
        assert dockerfile is not None
        text = dockerfile.read().decode("utf-8")
        oscar_lock = archive.extractfile(
            "openpi_policy_ranking_remote_build/context/"
            "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
            "requirements_oscar_foundation.lock"
        )
        assert oscar_lock is not None
        oscar_lock_text = oscar_lock.read().decode("utf-8")
    assert "blueprint-groot-oscar-eval@sha256:ab8fbccb" in text
    assert (
        "COPY --from=oscar_runtime /opt/blueprint/oscar_source_provenance.json"
        not in text
    )
    assert "COPY --from=oscar_runtime /opt/oscar-public /opt/OSCAR" not in text
    assert "https://github.com/wuzy2115/oscar-public.git" in text
    assert "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb" in text
    assert 'git -C /opt/OSCAR checkout --detach FETCH_HEAD' in text
    shim = "/opt/oscar-venv/bin/python /tmp/oscar_te_shim.py /opt/OSCAR"
    cleanup = "find /opt/OSCAR -type d -name __pycache__"
    seal = "blueprint_pipeline.oscar_runtime_source_provenance seal"
    assert shim in text
    assert cleanup in text
    assert seal in text
    assert text.index(shim) < text.index(cleanup) < text.index(seal)
    assert "--output /opt/blueprint/oscar_source_provenance.json" in text
    assert '--source-url "${OSCAR_SOURCE_URL}"' in text
    assert '--source-commit "${OSCAR_SOURCE_REVISION}"' in text
    assert "--runtime-source-root /opt/OSCAR" in text
    assert "chmod 0444 /opt/blueprint/oscar_source_provenance.json" in text
    assert "rm -rf /opt/OSCAR/.git /tmp/oscar_te_shim.py" in text
    assert "requirements_oscar_foundation.lock" in text
    assert "loguru==0.7.3" in oscar_lock_text
    assert "opencv-python==5.0.0.93" in oscar_lock_text
    assert "torch==2.10.0+cu128" in oscar_lock_text
    assert "export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5" in text
    assert text.index("export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5") < text.index(
        "uv pip install --python /.ctrl-world-venv/bin/python"
    )
    assert "uv pip install --python /opt/oscar-venv/bin/python" in text
    assert "--require-hashes" in text
    assert "--index-strategy unsafe-best-match" in text
    assert "/opt/oscar-venv/bin/python -m pip check" in text
    assert "oscar_site_packages" not in text
    assert "import torch, cv2, loguru" in text
    assert "token=False" in text
    assert "offline_preflight" in text
    assert any(
        name.endswith("/requirements_oscar_foundation.lock")
        for name in result["archive_members"]
    )


def test_openpi_packet_is_admitted_only_when_clean_source_claim_is_ready(
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
        source_commit=head,
        source_worktree_dirty=True,
    )
    packet.update(status="ready", blockers=[], source_worktree_dirty=False)

    admitted = build_build_plane_admission(packet=packet, builder=_builder(head), spend=_spend())
    assert admitted["status"] == "admitted"
    assert admitted["checks"]["packet_kind"] == "openpi_policy_ranking_image"

    packet["context_manifest_sha256"] = "0" * 64
    blocked = build_build_plane_admission(packet=packet, builder=_builder(head), spend=_spend())
    assert blocked["status"] == "blocked"
    assert "builder_openpi_archive_script_binding_mismatch" in blocked["blockers"]


def test_openpi_cloud_init_uses_governed_docker_builder() -> None:
    text = build_cloud_init(
        host_private_b64="private",
        host_public_b64="public",
        shutdown_minutes=120,
        packet_kind="openpi_policy_ranking_image",
    )
    assert "docker-buildx" in text
    assert "/swapfile" in text
    assert text.index("mkswap") < text.index("touch /root/blueprint-builder-ready")


def test_openpi_runtime_installs_and_verifies_camera_render_thread_library() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[1] / "deploy/docker/policy_ranking_openpi/Dockerfile"
    ).read_text(encoding="utf-8")

    assert "libglib2.0-0" in dockerfile
    assert 'ldconfig -p | grep -F "libgthread-2.0.so.0"' in dockerfile


def test_remote_openpi_result_binds_digest_source_and_frozen_revisions(
    tmp_path: Path,
) -> None:
    packet = {
        "image_ref": "docker.io/example/openpi-ranking:versioned",
        "source_commit": "a" * 40,
        "dockerfile_sha256": "b" * 64,
        "context_manifest_sha256": "c" * 64,
        "openpi_revision": OPENPI_REVISION,
        "menagerie_revision": MENAGERIE_REVISION,
    }
    payload = {
        "schema_version": "openpi_policy_ranking_gpu_release.v1",
        "status": "passed",
        "blockers": [],
        "resolved_digest_ref": "docker.io/example/openpi-ranking@sha256:" + "d" * 64,
        "source_commit": packet["source_commit"],
        "dockerfile_sha256": packet["dockerfile_sha256"],
        "context_manifest_sha256": packet["context_manifest_sha256"],
        "openpi_revision": OPENPI_REVISION,
        "menagerie_revision": MENAGERIE_REVISION,
        "runnable_platform": "linux/amd64",
        "checkpoint_bytes_embedded": 0,
        "interiorgs_assets_embedded": False,
        "raw_secret_values_recorded": False,
    }
    (tmp_path / "openpi_policy_ranking_gpu_release.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    verified = validate_remote_openpi_result(tmp_path, packet=packet)
    assert verified["status"] == "verified"
    assert verified["resolved_digest_ref"] == payload["resolved_digest_ref"]
