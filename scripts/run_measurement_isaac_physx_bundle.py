#!/usr/bin/env python3
"""Run a digest-bound Isaac/PhysX measurement bundle inside Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import fmean

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.claim_contract_keys import PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY
from blueprint_pipeline.measurement_adapter_execution import (
    run_measurement_adapter_execution,
    validate_measurement_adapter_execution_request,
)
from blueprint_pipeline.measurement_isaac_physx_rigid_adapter import WORKER_SCRIPT
from blueprint_pipeline.measurement_isaac_runtime_release import (
    ISAAC_VERSION,
    RUNTIME_IMAGE,
)


RUNTIME_RESULT_SCHEMA_VERSION = "measurement_isaac_physx_rtx_vast_runtime_result.v2"


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("measurement_isaac_bundle_object_required")
    return value


def _environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"measurement_isaac_environment_missing:{name}")
    return value


def _sha256_json(value: dict) -> str:
    normalized = dict(value)
    normalized.pop("preflight_result_digest", None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_process_tail(value: str) -> str:
    tail = value[-4000:]
    tail = re.sub(r"https?://\S+", "<redacted-url>", tail)
    return tail.replace("\x00", "")


def _run_rtx_preflight(bundle_root: Path) -> tuple[dict, list[str]]:
    output = bundle_root / "rtx_openusd_runtime_preflight.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.isaac_worker_runtime_preflight",
            "--output",
            str(output),
            "--require-nvidia-smi",
            "--require-rtx-render",
            "--smoke-steps",
            "2",
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    process_observation = {
        "exit_code": completed.returncode,
        "stdout_bytes": len(completed.stdout.encode("utf-8", "replace")),
        "stderr_bytes": len(completed.stderr.encode("utf-8", "replace")),
        "stdout_digest": "sha256:"
        + hashlib.sha256(completed.stdout.encode("utf-8", "replace")).hexdigest(),
        "stderr_digest": "sha256:"
        + hashlib.sha256(completed.stderr.encode("utf-8", "replace")).hexdigest(),
        "stdout_tail": _safe_process_tail(completed.stdout),
        "stderr_tail": _safe_process_tail(completed.stderr),
        "raw_secret_values_recorded": False,
    }
    if not output.is_file() or output.is_symlink() or output.stat().st_size > 1024 * 1024:
        result = {
            "schema_version": "isaac_worker_runtime_preflight.v1",
            "status": "blocked",
            "checks": [],
            "blockers": ["rtx_preflight_result_missing_or_unsafe"],
            "subprocess_observation": process_observation,
            "raw_secret_values_recorded": False,
            "proof_boundary": {
                "rtx_pixels_rendered": False,
                "calibrated_sensor_match_proven": False,
                "q_sensor_qualification_created": False,
                PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY: False,
            },
        }
        result["preflight_result_digest"] = _sha256_json(result)
        return result, ["measurement_isaac_rtx_preflight_result_missing_or_unsafe"]
    try:
        result = _read_object(output)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, ["measurement_isaac_rtx_preflight_result_invalid"]
    result["subprocess_observation"] = process_observation
    result["preflight_result_digest"] = _sha256_json(result)
    blockers: list[str] = []
    if completed.returncode != 0:
        blockers.append(f"measurement_isaac_rtx_preflight_exit_nonzero:{completed.returncode}")
    if result.get("status") != "passed" or result.get("blockers") != []:
        blockers.append("measurement_isaac_rtx_preflight_reported_blockers")
    return result, blockers


def run_bundle(bundle_root: Path) -> dict:
    manifest = _read_object(bundle_root / "bundle_manifest.json")
    blockers: list[str] = []
    if manifest.get("schema_version") != "measurement_isaac_physx_rtx_input_bundle.v2":
        blockers.append("measurement_isaac_bundle_manifest_schema_invalid")
    supplied_manifest_digest = manifest.get("bundle_manifest_digest")
    if supplied_manifest_digest != canonical_digest(
        manifest, digest_field="bundle_manifest_digest"
    ):
        blockers.append("measurement_isaac_bundle_manifest_digest_mismatch")
    expected_bindings = {
        "source_commit_sha": _environment("BLUEPRINT_MEASUREMENT_ISAAC_SOURCE_COMMIT"),
        "runtime_image_digest": _environment("BLUEPRINT_MEASUREMENT_ISAAC_RUNTIME_IMAGE"),
        "runtime_release_digest": _environment(
            "BLUEPRINT_MEASUREMENT_ISAAC_RUNTIME_RELEASE_DIGEST"
        ),
        "input_bundle_digest": _environment("BLUEPRINT_MEASUREMENT_ISAAC_INPUT_BUNDLE_DIGEST"),
    }
    for key, expected in expected_bindings.items():
        observed = (
            manifest.get(key)
            if key != "input_bundle_digest"
            else _environment("BLUEPRINT_MEASUREMENT_ISAAC_INPUT_BUNDLE_DIGEST")
        )
        if key != "input_bundle_digest" and observed != expected:
            blockers.append(f"measurement_isaac_bundle_{key}_mismatch")
    if manifest.get("isaac_sim_version") != ISAAC_VERSION:
        blockers.append("measurement_isaac_bundle_version_mismatch")
    if manifest.get("runtime_image_digest") != RUNTIME_IMAGE:
        blockers.append("measurement_isaac_bundle_image_mismatch")
    if (
        manifest.get("rtx_openusd_runtime_preflight_required") is not True
        or manifest.get("rtx_renderer") != "RayTracedLighting"
        or manifest.get("rtx_smoke_resolution") != [64, 64]
    ):
        blockers.append("measurement_isaac_bundle_rtx_contract_invalid")

    requests: list[dict] = []
    request_files = manifest.get("request_files")
    if not isinstance(request_files, list) or len(request_files) != 2:
        blockers.append("measurement_isaac_bundle_request_files_invalid")
    else:
        for entry in request_files:
            if not isinstance(entry, dict):
                blockers.append("measurement_isaac_bundle_request_entry_invalid")
                continue
            relative = str(entry.get("path") or "")
            path = bundle_root / relative
            try:
                request = validate_measurement_adapter_execution_request(_read_object(path))
            except (OSError, ValueError) as exc:
                blockers.append(f"measurement_isaac_bundle_request_invalid:{type(exc).__name__}")
                continue
            if request["execution_request_digest"] != entry.get("execution_request_digest"):
                blockers.append("measurement_isaac_bundle_request_digest_mismatch")
            requests.append(request)

    bundles: list[dict] = []
    rtx_preflight: dict = {}
    if not blockers:
        for request in requests:
            bundles.append(
                run_measurement_adapter_execution(
                    request,
                    command_argv=[sys.executable, str(WORKER_SCRIPT)],
                    execute=True,
                )
            )
    completed = bool(bundles) and all(
        bundle["receipt"]["status"] == "completed" for bundle in bundles
    )
    if bundles and not completed:
        blockers.extend(
            code for bundle in bundles for code in bundle["receipt"].get("failure_codes", [])
        )
    if completed and not blockers:
        rtx_preflight, rtx_blockers = _run_rtx_preflight(bundle_root)
        blockers.extend(rtx_blockers)
    development_execution_completed = bool(completed and rtx_preflight and not blockers)
    metrics = [
        dict(bundle["prediction"]["observed_metrics"])
        for bundle in bundles
        if isinstance(bundle.get("prediction"), dict)
    ]
    aggregate_metrics: dict[str, float | int] = {}
    if completed and len(metrics) == 2:
        penetrations = [float(row["penetration"]) for row in metrics]
        aggregate_metrics = {
            "ground_contact_case_count": sum(
                row.get("contact_sequence") == "ground_contact" for row in metrics
            ),
            "maximum_penetration_m": max(penetrations),
            "mean_penetration_m": fmean(penetrations),
        }
    result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "passed" if development_execution_completed else "failed",
        "source_commit_sha": expected_bindings["source_commit_sha"],
        "runtime_image_digest": expected_bindings["runtime_image_digest"],
        "runtime_release_digest": expected_bindings["runtime_release_digest"],
        "input_bundle_digest": expected_bindings["input_bundle_digest"],
        "bundle_manifest_digest": supplied_manifest_digest,
        "isaac_sim_version": ISAAC_VERSION,
        "execution_bundle_count": len(bundles),
        "execution_bundles": bundles,
        "aggregate_metrics": aggregate_metrics,
        "rtx_openusd_runtime_preflight": rtx_preflight,
        "rtx_runtime_preflight_completed": development_execution_completed,
        "q_sensor_qualification_created": False,
        "blockers": sorted(set(blockers)),
        "development_only": True,
        "synthetic_fixture": True,
        "held_out": False,
        "physical_measurements_included": False,
        "qualification_created": False,
        "r5_evidence": False,
        "r6_decision": False,
        "r7_admission": False,
        "production_route_eligible": False,
        "physical_success_established": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
        "raw_secret_values_recorded": False,
        "proof_effect": (
            "development_execution_only" if development_execution_completed else "none"
        ),
        "claim_ceiling": (
            "isaac_physx_rigid_contact_plus_rtx_startup_development"
            if development_execution_completed
            else "no_execution_evidence"
        ),
    }
    result["runtime_result_digest"] = canonical_digest(result, digest_field="runtime_result_digest")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_bundle(args.bundle_root.resolve())
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "failed",
            "blockers": [f"measurement_isaac_bundle_controller_failure:{type(exc).__name__}"],
            "raw_secret_values_recorded": False,
            "proof_effect": "none",
            "claim_ceiling": "no_execution_evidence",
        }
        result["runtime_result_digest"] = canonical_digest(
            result, digest_field="runtime_result_digest"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
