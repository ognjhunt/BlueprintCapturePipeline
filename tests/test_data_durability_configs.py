"""Fast-lane gate for the data-durability program (findings R048 retention + R053 backup/DR).

Wires the standalone validators + their unittest suites (which live in ``scripts/`` to mirror
the ``BlueprintCapture/scripts/validate_storage_lifecycle*`` idiom) into the pipeline's real
``pytest`` gate, so the committed retention + backup configs cannot drift without a red build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
CONFIGS = REPO_ROOT / "configs"

RETENTION_VALIDATOR = SCRIPTS / "validate_data_retention_policy.py"
RETENTION_TESTS = SCRIPTS / "validate_data_retention_policy_tests.py"
BACKUP_VALIDATOR = SCRIPTS / "validate_firestore_backup_config.py"
BACKUP_TESTS = SCRIPTS / "validate_firestore_backup_config_tests.py"
BACKUP_EMITTER = SCRIPTS / "emit_firestore_backup_command.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )


def test_committed_configs_exist() -> None:
    assert (CONFIGS / "data_retention_policy.json").exists()
    assert (CONFIGS / "firestore_backup_schedule.json").exists()


def test_retention_validator_passes_on_committed_policy() -> None:
    result = _run(str(RETENTION_VALIDATOR))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validation passed" in result.stdout


def test_backup_validator_passes_on_committed_config() -> None:
    result = _run(str(BACKUP_VALIDATOR))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "validation passed" in result.stdout


def test_backup_emitter_renders_gcloud_export() -> None:
    result = _run(str(BACKUP_EMITTER), "--timestamp", "20260709T070000Z")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().startswith("gcloud firestore export gs://")


@pytest.mark.parametrize("suite", [RETENTION_TESTS, BACKUP_TESTS], ids=["retention", "backup"])
def test_validator_unittest_suites(suite: Path) -> None:
    result = _run(str(suite))
    assert result.returncode == 0, result.stdout + result.stderr
