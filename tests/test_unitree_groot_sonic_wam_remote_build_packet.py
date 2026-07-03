from __future__ import annotations

import json
import tarfile
from pathlib import Path

from blueprint_pipeline.unitree_groot_sonic_wam_image_remote_build_packet import (
    CONTEXT_FILENAMES,
    prepare_remote_build_packet,
)


def _source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for name in CONTEXT_FILENAMES:
        (source / name).write_text(f"{name}\n", encoding="utf-8")
    return source


def test_remote_build_packet_packages_minimal_context_without_secrets(tmp_path: Path) -> None:
    source = _source_dir(tmp_path)
    manifest = prepare_remote_build_packet(
        output_dir=tmp_path / "packet",
        source_dir=source,
        image_ref="registry.example/blueprint/unitree-groot-sonic-wam:20260703-sealed",
        generated_at="2026-07-03T00:00:00+00:00",
    )

    assert manifest["status"] == "ready"
    assert manifest["raw_secret_values_recorded"] is False
    assert manifest["claim_boundary"]["remote_build_packet_is_not_task_success"] is True
    assert manifest["provider_use"]["provider_launch_performed_by_packet"] is False
    assert {row["path"] for row in manifest["context_files"]} == {
        f"context/{name}" for name in CONTEXT_FILENAMES
    }

    packet_dir = Path(manifest["packet_dir"])
    run_script = Path(manifest["run_script_path"]).read_text(encoding="utf-8")
    assert "--push" in run_script
    assert "docker buildx build" in run_script
    assert "BLUEPRINT_REMOTE_IMAGE_BUILD_DOCKER_LOGIN" in run_script
    assert "hf_token" in run_script
    assert "registry.example/blueprint/unitree-groot-sonic-wam:20260703-sealed" in run_script
    assert "raw_secret_values_recorded" in run_script
    assert (packet_dir / "README.md").is_file()

    tarball = Path(manifest["tarball_path"])
    assert tarball.is_file()
    with tarfile.open(tarball, "r:gz") as tf:
        names = set(tf.getnames())
    assert "unitree_groot_sonic_wam_remote_build/remote_build_unitree_groot_sonic_wam_image.sh" in names
    assert all(
        f"unitree_groot_sonic_wam_remote_build/context/{name}" in names
        for name in CONTEXT_FILENAMES
    )

    persisted = json.loads(
        Path(manifest["manifest_path"]).read_text(encoding="utf-8")
    )
    assert persisted["status"] == "ready"


def test_remote_build_packet_blocks_unstable_image_tag(tmp_path: Path) -> None:
    manifest = prepare_remote_build_packet(
        output_dir=tmp_path / "packet",
        source_dir=_source_dir(tmp_path),
        image_ref="registry.example/blueprint/unitree-groot-sonic-wam:latest",
    )

    assert manifest["status"] == "blocked"
    assert "unitree_groot_sonic_wam_image_ref_refuses_unstable_tag" in manifest["blockers"]
    assert Path(manifest["tarball_path"]).is_file()


def test_remote_build_packet_blocks_missing_context_file(tmp_path: Path) -> None:
    source = _source_dir(tmp_path)
    (source / "Dockerfile").unlink()

    manifest = prepare_remote_build_packet(
        output_dir=tmp_path / "packet",
        source_dir=source,
        image_ref="registry.example/blueprint/unitree-groot-sonic-wam:20260703-sealed",
    )

    assert manifest["status"] == "blocked"
    assert "missing_remote_build_context_file:Dockerfile" in manifest["blockers"]
