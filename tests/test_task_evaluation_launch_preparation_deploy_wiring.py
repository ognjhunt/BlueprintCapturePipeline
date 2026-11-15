from __future__ import annotations

import os
from pathlib import Path

import scripts.deploy_control_plane_commit as deploy


ROOT = Path(__file__).resolve().parents[1]


def test_exact_sha_deployer_materializes_discovery_runtime_directories(
    tmp_path: Path, monkeypatch
) -> None:
    owner = (os.getuid(), os.getgid())
    monkeypatch.setattr(deploy, "_service_account_ids", lambda _account: owner)
    directories = (
        str(tmp_path / "queue"),
        str(tmp_path / "queue" / "pending"),
        str(tmp_path / "inputs"),
    )

    receipts = deploy._install_scene_object_discovery_runtime_directories(
        directories=directories,
        account="test-account",
    )

    assert [receipt["path"] for receipt in receipts] == list(directories)
    assert all(Path(path).is_dir() for path in directories)
    assert all(Path(path).stat().st_mode & 0o777 == 0o750 for path in directories)
