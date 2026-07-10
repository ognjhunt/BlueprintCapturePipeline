from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.city_launch_evidence_policy import (
    RUN_SCHEMA_VERSION,
    build_artifact_inventory,
    evidence_policy,
    prepare_evidence_root,
    validate_run,
)
from blueprint_pipeline.common import write_json


def _write_run(root: Path, *, created_at: datetime | None = None) -> Path:
    created = created_at or datetime.now(timezone.utc)
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    write_json(root / "proof.launch-proof.json", {"status": "blocked_external_dependency"})
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": "test-run",
        "city_slug": "test-city",
        "created_or_updated_at": created.isoformat(),
        "evidence_policy": evidence_policy(created_at=created),
        "artifact_inventory": build_artifact_inventory(root),
    }
    write_json(root / "manifest.json", manifest)
    return root


def test_prepare_evidence_root_rejects_source_checkout(tmp_path: Path) -> None:
    source_root = tmp_path / "repo"
    source_root.mkdir()

    with pytest.raises(ValueError, match="outside the source checkout"):
        prepare_evidence_root(source_root / "ops" / "city-launch-runs", source_root=source_root)


def test_prepare_evidence_root_enforces_private_permissions(tmp_path: Path) -> None:
    source_root = tmp_path / "repo"
    source_root.mkdir()
    evidence_root = prepare_evidence_root(tmp_path / "evidence", source_root=source_root)

    assert evidence_root.stat().st_mode & 0o777 == 0o700


def test_validate_run_checks_schema_inventory_retention_and_freshness(tmp_path: Path) -> None:
    run_root = _write_run(tmp_path / "run")

    result = validate_run(run_root)

    assert result["valid"] is True
    assert result["artifact_count"] == 1
    assert result["fresh"] is True


def test_validate_run_rejects_tampered_artifact(tmp_path: Path) -> None:
    run_root = _write_run(tmp_path / "run")
    (run_root / "proof.launch-proof.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory, size, or SHA-256 mismatch"):
        validate_run(run_root)


def test_validate_run_rejects_stale_evidence(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    run_root = _write_run(tmp_path / "run", created_at=now - timedelta(days=8))

    with pytest.raises(ValueError, match="evidence is stale"):
        validate_run(run_root, now=now)


def test_validate_run_rejects_unapproved_external_disclosure(tmp_path: Path) -> None:
    run_root = _write_run(tmp_path / "run")

    with pytest.raises(ValueError, match="not approved for external disclosure"):
        validate_run(run_root, require_disclosure_approval=True)


def test_repository_tracks_no_generated_city_launch_evidence() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "ls-files", "ops/city-launch-runs/**"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = {
        line
        for line in result.stdout.splitlines()
        if line and (repo_root / line).exists()
    }

    assert tracked <= {"ops/city-launch-runs/README.md"}


def test_manifest_json_is_not_used_as_its_own_inventory_member(tmp_path: Path) -> None:
    run_root = _write_run(tmp_path / "run")
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))

    assert {entry["path"] for entry in manifest["artifact_inventory"]} == {
        "proof.launch-proof.json"
    }
