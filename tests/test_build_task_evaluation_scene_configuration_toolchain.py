from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
    validate_scene_configuration_toolchain,
)
from blueprint_pipeline.task_evaluation_scene_configuration_stage_producers import (
    ADMITTED_PRODUCER_IDENTITIES,
)
from scripts.build_task_evaluation_scene_configuration_toolchain import (
    build_published_scene_configuration_toolchain,
)


def test_builds_exclusive_read_only_full_byte_readback_toolchain(tmp_path: Path) -> None:
    commit = "a" * 40
    output = tmp_path / "runtime" / commit
    observed: list[Path] = []

    def readback(path: Path) -> bytes:
        observed.append(path)
        return path.read_bytes()

    receipt = build_published_scene_configuration_toolchain(
        source_commit=commit,
        output_root=output,
        readback=readback,
        readback_actor="service-account:test-runner",
    )

    manifest = validate_scene_configuration_toolchain(
        root=output,
        expected_source_commit=commit,
    )
    assert receipt["toolchain_digest"] == manifest["toolchain_digest"]
    assert receipt["full_byte_service_account_readback_passed"] is True
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_allocated"] is False
    assert len(observed) == len(ADMITTED_PRODUCER_IDENTITIES) + 1
    assert not output.stat().st_mode & 0o222
    assert all(not path.stat().st_mode & 0o222 for path in output.rglob("*"))
    for identity in ADMITTED_PRODUCER_IDENTITIES:
        executable = output / "stages" / identity.adapter_id
        assert executable.stat().st_mode & 0o111
        assert (
            "blueprint_pipeline.task_evaluation_scene_configuration_stage_tool"
            in executable.read_text(encoding="utf-8")
        )


def test_toolchain_publication_fails_closed_on_existing_or_bad_readback(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runtime"
    output.mkdir()
    with pytest.raises(ValueError, match="output_exists"):
        build_published_scene_configuration_toolchain(
            source_commit="b" * 40,
            output_root=output,
            readback=lambda path: path.read_bytes(),
            readback_actor="service-account:test-runner",
        )

    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="service_readback_failed"):
        build_published_scene_configuration_toolchain(
            source_commit="b" * 40,
            output_root=failed,
            readback=lambda _path: b"tampered",
            readback_actor="service-account:test-runner",
        )
    assert not failed.exists()
