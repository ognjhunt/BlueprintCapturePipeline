"""Refuse to launch when staging the bundle would fill the disk."""

from __future__ import annotations

import pytest

from blueprint_pipeline.launch_disk_headroom import (
    LaunchDiskHeadroomError,
    require_launch_disk_headroom,
)


def test_a_launch_with_room_proceeds(tmp_path):
    receipt = require_launch_disk_headroom(
        path=tmp_path,
        required_bytes=100 * 1024 * 1024,
        free_bytes_override=5 * 1024 * 1024 * 1024,
    )

    assert receipt["sufficient"] is True


def test_a_launch_that_would_fill_the_disk_is_refused(tmp_path):
    """Twice now the disk filled mid-run and killed a healthy launch.

    Both times the instance was already created, so the cost was not a failed
    launch - it was an orphan billing until someone noticed. Refusing before
    the create is free; dying after it is not.
    """

    with pytest.raises(LaunchDiskHeadroomError) as excinfo:
        require_launch_disk_headroom(
            path=tmp_path,
            required_bytes=200 * 1024 * 1024,
            free_bytes_override=150 * 1024 * 1024,
        )

    joined = ";".join(excinfo.value.errors)
    assert "insufficient_disk" in joined
    # The message must say how much to free, not merely that there is too little.
    assert "needed_bytes" in joined and "free_bytes" in joined


def test_the_margin_covers_more_than_the_bundle_itself(tmp_path):
    """A run writes logs, results and an output zip after the bundle.

    Sizing the check to the bundle alone would pass a launch that then dies
    writing its own receipt - which is the same failure with extra steps.
    """

    with pytest.raises(LaunchDiskHeadroomError):
        require_launch_disk_headroom(
            path=tmp_path,
            required_bytes=1000,
            safety_margin_bytes=500 * 1024 * 1024,
            free_bytes_override=200 * 1024 * 1024,
        )


def test_the_receipt_reports_what_it_measured(tmp_path):
    receipt = require_launch_disk_headroom(
        path=tmp_path,
        required_bytes=1024,
        safety_margin_bytes=0,
        free_bytes_override=4096,
    )

    assert receipt["free_bytes"] == 4096
    assert receipt["needed_bytes"] == 1024
    assert receipt["measured_path"] == str(tmp_path)


def test_a_real_filesystem_is_measured_when_no_override_is_given(tmp_path):
    """The override exists for tests; the default must consult the disk."""

    receipt = require_launch_disk_headroom(
        path=tmp_path, required_bytes=1, safety_margin_bytes=0
    )

    assert receipt["free_bytes"] > 0
    assert receipt["measured_from"] == "statvfs"


def test_the_allocator_blocks_a_launch_with_no_disk(tmp_path, monkeypatch):
    """Wired before the create, so a full disk costs nothing instead of an orphan."""

    from types import SimpleNamespace

    from blueprint_pipeline import paid_resource_allocator as allocator

    asset = tmp_path / "twin.usda"
    asset.write_text("#usda 1.0\n" * 1000, encoding="utf-8")
    args = SimpleNamespace(
        adp_job_dir=str(tmp_path / "job"),
        adp009d_approved_can=str(asset),
        adp009d_sage_collision=None,
        adp009d_harness_manifest=None,
        adp009d_extra_native=None,
    )

    monkeypatch.setattr(
        allocator.os, "statvfs", lambda _p: SimpleNamespace(f_bavail=1, f_frsize=4096)
    )

    with pytest.raises(LaunchDiskHeadroomError):
        allocator._require_launch_disk_headroom(args)
