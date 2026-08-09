from __future__ import annotations

import json
from collections import namedtuple

import pytest

from blueprint_pipeline.provider_runtime_capacity import (
    measure_runtime_disk_headroom,
)


Usage = namedtuple("Usage", "total used free")


@pytest.mark.parametrize(
    ("free_bytes", "status", "blockers"),
    [
        (40 * 1024**3, "passed", []),
        (
            10 * 1024**3,
            "blocked",
            ["joint_agent_runtime_disk_headroom_insufficient"],
        ),
    ],
)
def test_native_disk_headroom_receipt_is_manifest_bound(
    tmp_path, free_bytes: int, status: str, blockers: list[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime_resource_requirements": {
                    "requested_disk_gb": 96,
                    "minimum_free_bytes_before_dependency_install": 32 * 1024**3,
                    "failure_blocker": (
                        "joint_agent_runtime_disk_headroom_insufficient"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "receipt.json"

    receipt = measure_runtime_disk_headroom(
        manifest_path=manifest,
        receipt_path=receipt_path,
        measurement_path=tmp_path,
        disk_usage=lambda _: Usage(96 * 1024**3, 96 * 1024**3 - free_bytes, free_bytes),
    )

    assert receipt["status"] == status
    assert receipt["blockers"] == blockers
    assert receipt["native_readback"] is True
    assert receipt["free_bytes"] == free_bytes
    assert json.loads(receipt_path.read_text()) == receipt


def test_disk_headroom_rejects_caller_asserted_or_missing_capacity(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "runtime_resource_requirements": {
                    "requested_disk_gb": 96,
                    "minimum_free_bytes_before_dependency_install": True,
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="provider_runtime_capacity_requirements_invalid"
    ):
        measure_runtime_disk_headroom(
            manifest_path=manifest,
            receipt_path=tmp_path / "receipt.json",
            measurement_path=tmp_path,
        )
