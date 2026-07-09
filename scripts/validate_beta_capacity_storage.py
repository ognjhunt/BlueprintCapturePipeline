#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
EXPECTED_MAX_DURATION_SECONDS = 45 * 60
EXPECTED_INLINE_EXTRACT_BYTES = 1_500_000_000
EXPECTED_TARGET_CONCURRENCY = 25


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a JSON object")
    return payload


def _find_rule(lifecycle: dict[str, Any], action_type: str, age: int, prefixes: set[str]) -> dict[str, Any]:
    for rule in lifecycle.get("rule", []):
        if not isinstance(rule, dict):
            continue
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
        condition = rule.get("condition") if isinstance(rule.get("condition"), dict) else {}
        if action.get("type") != action_type:
            continue
        if condition.get("age") != age:
            continue
        if prefixes.issubset(set(condition.get("matchesPrefix") or [])):
            return rule
    raise AssertionError(f"missing lifecycle rule action={action_type} age={age} prefixes={sorted(prefixes)}")


def _require_text(path: Path, required: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for needle in required:
        if needle not in text:
            raise AssertionError(f"{path} is missing required text: {needle}")


def _validate_capture_swift_constants(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "betaMaxFileSizeBytes" not in text or "20 * 1024 * 1024 * 1024" not in text:
        raise AssertionError(f"{path} does not define the 20 GiB beta upload size cap")
    if "betaMaxDurationSeconds" not in text or "45 * 60" not in text:
        raise AssertionError(f"{path} does not define the 45 minute beta duration cap")
    if not re.search(r"maxFileSizeBytes:\s*Int64", text):
        raise AssertionError(f"{path} must keep an explicit maxFileSizeBytes policy field")
    if not re.search(r"maxDurationSeconds:\s*Double", text):
        raise AssertionError(f"{path} must keep an explicit maxDurationSeconds policy field")


def validate_files(repo_root: Path, capture_swift_policy: Path | None = None) -> dict[str, Any]:
    model_path = repo_root / "docs" / "beta_capacity_cost_storage_model_2026-07-08.json"
    lifecycle_path = repo_root / "deploy" / "storage" / "primary-capture-bucket-lifecycle.json"
    doc_path = repo_root / "docs" / "BETA_CAPACITY_COST_STORAGE_MODEL_2026-07-08.md"
    terraform_path = repo_root / "deploy" / "terraform" / "main.tf"
    terraform_vars_example_path = repo_root / "deploy" / "terraform" / "terraform.tfvars.example"

    model = _load_json(model_path)
    lifecycle = _load_json(lifecycle_path)

    if model.get("schema_version") != "blueprint.beta_capacity_cost_storage_model.v1":
        raise AssertionError("unexpected capacity model schema_version")
    if model.get("beta_target", {}).get("external_users") != 100:
        raise AssertionError("capacity model must target 100 external users")
    if model.get("beta_target", {}).get("modeled_captures_per_month") != 300:
        raise AssertionError("capacity model must explicitly model 300 captures/month")
    if model.get("beta_target", {}).get("target_concurrent_uploaders") != EXPECTED_TARGET_CONCURRENCY:
        raise AssertionError("capacity model must target 25 concurrent uploaders")

    limits = model.get("per_capture_limits", {})
    if limits.get("max_upload_payload_bytes") != EXPECTED_MAX_UPLOAD_BYTES:
        raise AssertionError("capacity model max_upload_payload_bytes must be 20 GiB")
    if limits.get("max_duration_seconds") != EXPECTED_MAX_DURATION_SECONDS:
        raise AssertionError("capacity model max_duration_seconds must be 45 minutes")
    if limits.get("inline_extract_frames_max_video_bytes") != EXPECTED_INLINE_EXTRACT_BYTES:
        raise AssertionError("capacity model inline extract limit must match extractFrames default")

    runtime_capacity = model.get("runtime_capacity", {})
    if runtime_capacity.get("target_concurrent_jobs") != EXPECTED_TARGET_CONCURRENCY:
        raise AssertionError("runtime capacity must target 25 concurrent jobs")
    if runtime_capacity.get("task_queue_max_concurrent_dispatches") != EXPECTED_TARGET_CONCURRENCY:
        raise AssertionError("task queue capacity must target 25 concurrent dispatches")
    if runtime_capacity.get("cloud_run_max_instances_per_privacy_runner") != EXPECTED_TARGET_CONCURRENCY:
        raise AssertionError("Cloud Run privacy runner max instances must target 25")

    lifecycle_ref = model.get("storage_lifecycle", {})
    if lifecycle_ref.get("policy_file") != "deploy/storage/primary-capture-bucket-lifecycle.json":
        raise AssertionError("capacity model must point to the primary capture bucket lifecycle file")
    if lifecycle_ref.get("apply_script") != "scripts/apply_primary_capture_bucket_lifecycle.sh":
        raise AssertionError("capacity model must point to the lifecycle apply script")

    _find_rule(lifecycle, "SetStorageClass", 30, {"scenes/", "targets/"})
    _find_rule(lifecycle, "SetStorageClass", 90, {"scenes/", "targets/"})
    _find_rule(lifecycle, "Delete", 180, {"scenes/", "targets/"})
    _find_rule(lifecycle, "Delete", 14, {"tmp/", "staging/", "debug/"})
    _find_rule(lifecycle, "Delete", 365, {"buyer_delivery/", "marketplace/", "hosted_sessions/", "robot_eval_jobs/"})

    _require_text(
        doc_path,
        [
            "Max capture upload payload: 20 GiB",
            "Max capture duration: 45 minutes",
            "Terraform `max_concurrent_jobs`",
            "scripts/apply_primary_capture_bucket_lifecycle.sh",
            "scripts/run_beta_intake_soak_test.py --dry-run",
        ],
    )

    _require_text(
        terraform_path,
        [
            'variable "max_concurrent_jobs"',
            "default     = 25",
            "var.max_concurrent_jobs >= 25",
            "max_concurrent_dispatches = var.max_concurrent_jobs",
            "max_instance_count = var.max_concurrent_jobs",
        ],
    )
    _require_text(terraform_vars_example_path, ["max_concurrent_jobs = 25"])

    if capture_swift_policy is not None:
        _validate_capture_swift_constants(capture_swift_policy)

    return {
        "status": "passed",
        "model_path": str(model_path),
        "lifecycle_path": str(lifecycle_path),
        "external_users": model["beta_target"]["external_users"],
        "modeled_captures_per_month": model["beta_target"]["modeled_captures_per_month"],
        "target_concurrent_uploaders": model["beta_target"]["target_concurrent_uploaders"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate beta capacity, cost, and storage lifecycle artifacts.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--capture-swift-policy")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    capture_swift_policy = Path(args.capture_swift_policy).resolve() if args.capture_swift_policy else None
    result = validate_files(repo_root, capture_swift_policy)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
