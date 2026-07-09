#!/usr/bin/env python3
"""Unit tests for the Firestore + storage backup/DR config validator (finding R053)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_firestore_backup_config.py"
COMMITTED_CONFIG = REPO_ROOT / "configs" / "firestore_backup_schedule.json"

GOOD_CONFIG = {
    "schema_version": "1.0",
    "rpo_hours": 24,
    "rto_hours": 4,
    "firestore_export": {
        "project_id": "blueprint-8c1ca",
        "database": "(default)",
        "destination_bucket": "blueprint-8c1ca-backups",
        "destination_prefix": "firestore-exports",
        "collection_ids": [],
        "schedule_cron": "0 7 * * *",
        "backup_retention_days": 90,
        "gcloud_command_template": (
            "gcloud firestore export gs://{destination_bucket}/{destination_prefix}/{timestamp} "
            "--project={project_id} --database=\"{database}\""
        ),
    },
    "storage_buckets": [
        {
            "bucket": "blueprint-8c1ca.appspot.com",
            "role": "primary_authoritative",
            "object_versioning": True,
            "versioning_retention_days": 2555,
        },
        {
            "bucket": "blueprint-8c1ca-backups",
            "role": "backup_target",
            "object_versioning": True,
            "versioning_retention_days": 90,
        },
    ],
    "restore": {
        "firestore_import_command_template": (
            "gcloud firestore import gs://{destination_bucket}/{destination_prefix}/{export_id} "
            "--project={project_id} --database=\"{database}\""
        )
    },
}


class BackupConfigValidatorTests(unittest.TestCase):
    def run_validator(self, config_path: Path) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, str(VALIDATOR), "--config", str(config_path)],
            env=env,
            capture_output=True,
            text=True,
        )

    def write_config(self, config: object) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        if isinstance(config, str):
            tmp.write(config)
        else:
            json.dump(config, tmp)
        tmp.close()
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.unlink(tmp.name))
        return Path(tmp.name)

    def assert_fails(self, config: object, needle: str) -> None:
        result = self.run_validator(self.write_config(config))
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)

    # ── The committed config must pass, unmodified (default and explicit path). ──

    def test_committed_config_is_valid(self) -> None:
        self.assertTrue(COMMITTED_CONFIG.exists(), "firestore_backup_schedule.json must be committed")
        result = self.run_validator(COMMITTED_CONFIG)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("validation passed", result.stdout)

    def test_committed_config_default_path(self) -> None:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run([sys.executable, str(VALIDATOR)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_synthetic_good_config_passes(self) -> None:
        result = self.run_validator(self.write_config(GOOD_CONFIG))
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    # ── Authoritative-surface coverage. ──

    def test_missing_firestore_export_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config.pop("firestore_export")
        self.assert_fails(config, "firestore_export")

    def test_partial_collection_export_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["firestore_export"]["collection_ids"] = ["creatorPayouts"]
        self.assert_fails(config, "export the WHOLE authoritative")

    def test_missing_primary_bucket_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["storage_buckets"][0]["role"] = "backup_target"
        self.assert_fails(config, "authoritative capture bucket must be covered")

    # ── Durability / isolation guardrails. ──

    def test_backup_in_primary_bucket_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["firestore_export"]["destination_bucket"] = "blueprint-8c1ca.appspot.com"
        self.assert_fails(config, "must differ from the")

    def test_export_bucket_not_matching_backup_target_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["firestore_export"]["destination_bucket"] = "some-other-bucket"
        self.assert_fails(config, "does not match the declared")

    def test_primary_versioning_off_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["storage_buckets"][0]["object_versioning"] = False
        self.assert_fails(config, "object_versioning=true")

    def test_primary_versioning_retention_below_floor_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["storage_buckets"][0]["versioning_retention_days"] = 365
        self.assert_fails(config, "capture-truth floor")

    # ── RPO / RTO / schedule / restore. ──

    def test_rpo_too_large_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["rpo_hours"] = 72
        self.assert_fails(config, "maximum RPO")

    def test_bad_cron_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["firestore_export"]["schedule_cron"] = "daily"
        self.assert_fails(config, "5-field cron")

    def test_bad_export_command_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["firestore_export"]["gcloud_command_template"] = "rsync everything somewhere"
        self.assert_fails(config, "gcloud firestore export")

    def test_missing_restore_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config.pop("restore")
        self.assert_fails(config, "restore")

    def test_bad_import_command_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["restore"]["firestore_import_command_template"] = "copy the files back"
        self.assert_fails(config, "gcloud firestore import")

    # ── Schema / well-formedness. ──

    def test_malformed_json_fails(self) -> None:
        self.assert_fails("{not valid json", "not valid JSON")

    def test_empty_buckets_fails(self) -> None:
        config = copy.deepcopy(GOOD_CONFIG)
        config["storage_buckets"] = []
        self.assert_fails(config, "non-empty array")

    def test_missing_file_fails(self) -> None:
        result = self.run_validator(REPO_ROOT / "configs" / "does_not_exist.json")
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn("is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
