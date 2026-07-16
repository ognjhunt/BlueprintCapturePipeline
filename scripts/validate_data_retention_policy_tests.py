#!/usr/bin/env python3
"""Unit tests for the cross-surface data-retention policy validator (finding R048)."""

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
VALIDATOR = REPO_ROOT / "scripts" / "validate_data_retention_policy.py"
COMMITTED_POLICY = REPO_ROOT / "configs" / "data_retention_policy.json"

# A minimal, well-formed policy used as the base for mutation-based failure cases.
GOOD_POLICY = {
    "schema_version": "1.0",
    "capture_truth_floor_days": 2555,
    "financial_floor_days": 2555,
    "pii_floor_days": 30,
    "surfaces": {
        "storage_prefixes": [
            {
                "bucket": "blueprint-8c1ca.appspot.com",
                "prefix": "scenes/",
                "class": "raw_capture_authoritative",
                "retention_days": 3650,
                "action": "tier_then_delete",
                "enforced_by": "gcs_lifecycle",
                "managed_by": "BlueprintCapture/storage.lifecycle.json",
            },
            {
                "bucket": "blueprint-8c1ca.appspot.com",
                "prefix": "marketplace-artifacts/",
                "class": "derived_hosted_delivery",
                "retention_days": 365,
                "action": "delete",
                "enforced_by": "gcs_lifecycle",
            },
        ],
        "firestore_collections": [
            {
                "collection": "creatorPayouts",
                "class": "financial",
                "retention_days": 2555,
                "action": "review_then_delete",
                "enforced_by": "manual_review",
                "legal_hold": True,
            },
            {
                "collection": "waitlistSubmissions",
                "class": "pii_lead",
                "retention_days": 365,
                "action": "ttl_delete",
                "enforced_by": "firestore_ttl",
                "ttl_field": "expireAt",
            },
        ],
    },
}


class DataRetentionValidatorTests(unittest.TestCase):
    def run_validator(self, policy_path: Path) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, str(VALIDATOR), "--policy", str(policy_path)],
            env=env,
            capture_output=True,
            text=True,
        )

    def write_policy(self, policy: object) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        if isinstance(policy, str):
            tmp.write(policy)
        else:
            json.dump(policy, tmp)
        tmp.close()
        self.addCleanup(lambda: os.path.exists(tmp.name) and os.unlink(tmp.name))
        return Path(tmp.name)

    def assert_fails(self, policy: object, needle: str) -> None:
        result = self.run_validator(self.write_policy(policy))
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn(needle, result.stderr)

    # ── The committed policy must pass, unmodified (default path and explicit path). ──

    def test_committed_policy_is_valid(self) -> None:
        self.assertTrue(COMMITTED_POLICY.exists(), "data_retention_policy.json must be committed")
        result = self.run_validator(COMMITTED_POLICY)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("validation passed", result.stdout)

    def test_committed_policy_default_path(self) -> None:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, str(VALIDATOR)], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_synthetic_good_policy_passes(self) -> None:
        result = self.run_validator(self.write_policy(GOOD_POLICY))
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    # ── Capture-truth guardrails. ──

    def test_raw_below_capture_floor_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"][0]["retention_days"] = 400
        self.assert_fails(policy, "capture-truth floor")

    def test_derived_longer_than_raw_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"][1]["retention_days"] = 4000  # > raw 3650
        self.assert_fails(policy, "exceeds raw capture retention")

    def test_derived_equal_to_raw_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"][1]["retention_days"] = 3650  # == raw
        self.assert_fails(policy, "strictly less than raw")

    def test_missing_raw_entry_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"][0]["class"] = "derived_world_model"
        self.assert_fails(policy, "authoritative")

    def test_raw_not_gcs_lifecycle_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"][0]["enforced_by"] = "scheduled_job"
        self.assert_fails(policy, "gcs_lifecycle")

    def test_raw_without_lifecycle_file_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"][0].pop("managed_by")
        self.assert_fails(policy, "storage.lifecycle.json")

    def test_raw_wrong_prefix_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"][0]["prefix"] = "captures/"
        self.assert_fails(policy, "scenes/")

    # ── Financial / legal-hold guardrails. ──

    def test_financial_auto_delete_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][0]["action"] = "delete"
        self.assert_fails(policy, "never the auto-destructive")

    def test_financial_below_floor_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][0]["retention_days"] = 1000
        self.assert_fails(policy, "financial floor")

    # ── PII guardrails. ──

    def test_pii_below_floor_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][1]["retention_days"] = 5
        self.assert_fails(policy, "PII")

    # ── Firestore TTL consistency. ──

    def test_ttl_without_field_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][1].pop("ttl_field")
        self.assert_fails(policy, "ttl_field")

    def test_ttl_field_without_firestore_ttl_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][0]["ttl_field"] = "expireAt"
        self.assert_fails(policy, "ttl_field")

    def test_ttl_delete_requires_firestore_ttl(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        # action ttl_delete but enforced by a scheduled job (and no ttl_field).
        policy["surfaces"]["firestore_collections"][1]["enforced_by"] = "scheduled_job"
        policy["surfaces"]["firestore_collections"][1].pop("ttl_field")
        self.assert_fails(policy, "ttl_delete")

    # ── Schema / well-formedness. ──

    def test_malformed_json_fails(self) -> None:
        self.assert_fails("{not valid json", "not valid JSON")

    def test_empty_storage_array_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["storage_prefixes"] = []
        self.assert_fails(policy, "storage_prefixes must be a non-empty array")

    def test_unsupported_action_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][1]["action"] = "shred"
        self.assert_fails(policy, "unsupported action")

    def test_unsupported_enforcer_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][1]["enforced_by"] = "vibes"
        self.assert_fails(policy, "unsupported enforced_by")

    def test_missing_capture_floor_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy.pop("capture_truth_floor_days")
        self.assert_fails(policy, "capture_truth_floor_days")

    def test_firestore_claiming_raw_class_fails(self) -> None:
        policy = copy.deepcopy(GOOD_POLICY)
        policy["surfaces"]["firestore_collections"][0]["class"] = "raw_capture_authoritative"
        self.assert_fails(policy, "may not claim")

    def test_missing_file_fails(self) -> None:
        result = self.run_validator(REPO_ROOT / "configs" / "does_not_exist.json")
        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn("is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
