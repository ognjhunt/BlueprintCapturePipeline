#!/usr/bin/env python3
"""Run a digest-bound DLO-Lab CUDA development bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import fmean

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.measurement_adapter_execution import (
    run_measurement_adapter_execution,
    validate_measurement_adapter_execution_request,
)
from blueprint_pipeline.measurement_dlo_lab_cable_adapter import EXPECTED_SOURCE_COMMIT
from blueprint_pipeline.measurement_dlo_lab_runtime_release import RUNTIME_IMAGE


RUNTIME_RESULT_SCHEMA_VERSION = "measurement_dlo_lab_cuda_vast_runtime_result.v1"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("measurement_dlo_lab_bundle_object_required")
    return value


def _environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"measurement_dlo_lab_environment_missing:{name}")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def run_bundle(bundle_root: Path) -> dict:
    manifest = _read(bundle_root / "bundle_manifest.json")
    blockers: list[str] = []
    if manifest.get("schema_version") != "measurement_dlo_lab_cuda_input_bundle.v1":
        blockers.append("measurement_dlo_lab_bundle_manifest_schema_invalid")
    manifest_digest = manifest.get("bundle_manifest_digest")
    if manifest_digest != canonical_digest(manifest, digest_field="bundle_manifest_digest"):
        blockers.append("measurement_dlo_lab_bundle_manifest_digest_mismatch")
    bindings = {
        "source_commit_sha": _environment("BLUEPRINT_MEASUREMENT_DLO_SOURCE_COMMIT"),
        "runtime_image_digest": _environment("BLUEPRINT_MEASUREMENT_DLO_RUNTIME_IMAGE"),
        "runtime_release_digest": _environment("BLUEPRINT_MEASUREMENT_DLO_RUNTIME_RELEASE_DIGEST"),
        "input_bundle_digest": _environment("BLUEPRINT_MEASUREMENT_DLO_INPUT_BUNDLE_DIGEST"),
    }
    for key in ("source_commit_sha", "runtime_image_digest", "runtime_release_digest"):
        if manifest.get(key) != bindings[key]:
            blockers.append(f"measurement_dlo_lab_bundle_{key}_mismatch")
    if manifest.get("runtime_image_digest") != RUNTIME_IMAGE:
        blockers.append("measurement_dlo_lab_bundle_image_mismatch")
    if (
        manifest.get("dlo_lab_source_commit") != EXPECTED_SOURCE_COMMIT
        or manifest.get("required_backend") != "cuda"
        or manifest.get("replay_count") != 2
    ):
        blockers.append("measurement_dlo_lab_bundle_runtime_contract_invalid")
    for record in manifest.get("source_files") or []:
        if not isinstance(record, dict):
            blockers.append("measurement_dlo_lab_bundle_source_record_invalid")
            continue
        path = bundle_root / str(record.get("path") or "")
        if not path.is_file() or path.is_symlink() or _sha256(path) != record.get("digest"):
            blockers.append(
                f"measurement_dlo_lab_bundle_source_digest_mismatch:{record.get('path')}"
            )
    requests: list[dict] = []
    request_files = manifest.get("request_files")
    if not isinstance(request_files, list) or len(request_files) != 2:
        blockers.append("measurement_dlo_lab_bundle_request_files_invalid")
    else:
        for record in request_files:
            try:
                request = validate_measurement_adapter_execution_request(
                    _read(bundle_root / str(record.get("path") or ""))
                )
            except (OSError, ValueError) as exc:
                blockers.append(f"measurement_dlo_lab_bundle_request_invalid:{type(exc).__name__}")
                continue
            if request["execution_request_digest"] != record.get("execution_request_digest"):
                blockers.append("measurement_dlo_lab_bundle_request_digest_mismatch")
            requests.append(request)
    bundles: list[dict] = []
    if not blockers:
        for request in requests:
            bundles.append(
                run_measurement_adapter_execution(
                    request,
                    command_argv=[
                        sys.executable,
                        "-m",
                        "blueprint_pipeline.measurement_dlo_lab_cable_adapter",
                    ],
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
    observations = [
        dict(bundle["worker_result"]["runtime_observations"])
        for bundle in bundles
        if isinstance(bundle.get("worker_result"), dict)
    ]
    if completed and any(
        row.get("source_commit") != EXPECTED_SOURCE_COMMIT
        or row.get("cuda_available") is not True
        or row.get("cuda_device_count", 0) < 1
        or row.get("cpu_fallback_used") is not False
        or row.get("deterministic_replay_match") is not True
        for row in observations
    ):
        blockers.append("measurement_dlo_lab_runtime_observation_invalid")
    metrics = [
        dict(bundle["prediction"]["observed_metrics"])
        for bundle in bundles
        if isinstance(bundle.get("prediction"), dict)
    ]
    aggregate: dict[str, float | int] = {}
    if completed and len(metrics) == 2:
        displacements = [float(row["state_trajectory"]) for row in metrics]
        aggregate = {
            "case_count": 2,
            "maximum_tip_displacement_m": max(displacements),
            "mean_tip_displacement_m": fmean(displacements),
            "within_envelope_case_count": sum(
                row.get("task_outcome") == "within_deformation_envelope" for row in metrics
            ),
        }
    passed = bool(completed and len(observations) == 2 and not blockers)
    result = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "source_commit_sha": bindings["source_commit_sha"],
        "runtime_image_digest": bindings["runtime_image_digest"],
        "runtime_release_digest": bindings["runtime_release_digest"],
        "input_bundle_digest": bindings["input_bundle_digest"],
        "bundle_manifest_digest": manifest_digest,
        "dlo_lab_source_commit": EXPECTED_SOURCE_COMMIT,
        "execution_bundle_count": len(bundles),
        "execution_bundles": bundles,
        "aggregate_metrics": aggregate,
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
        "proof_effect": "development_execution_only" if passed else "none",
        "claim_ceiling": "dlo_lab_cuda_cable_development" if passed else "no_execution_evidence",
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
            "blockers": [f"measurement_dlo_lab_bundle_controller_failure:{type(exc).__name__}"],
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
