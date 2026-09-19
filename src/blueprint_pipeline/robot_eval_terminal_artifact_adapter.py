"""Read canonical robot-eval artifacts into a strict terminal observation.

This module is read-only.  It does not create a second receipt or infer episode
or spend facts that the existing artifacts did not record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


TERMINAL_MANIFEST_SCHEMA = "robot_eval_job_run_manifest.v1"
COST_LEDGER_SCHEMA = "robot_eval_gpu_cost_control_ledger.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number >= 0 else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _read_object(path: Path, blockers: list[str], artifact: str) -> dict[str, Any]:
    if not path.is_file():
        blockers.append(f"{artifact}_missing")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        blockers.append(f"{artifact}_invalid_json")
        return {}
    if not isinstance(value, Mapping):
        blockers.append(f"{artifact}_not_object")
        return {}
    return dict(value)


def _first_string(payload: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> str:
    for path in paths:
        value: Any = payload
        for part in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        text = _string(value)
        if text:
            return text
    return ""


def read_terminal_robot_eval_artifacts(job_dir: str | Path) -> dict[str, Any]:
    """Return observed terminal episode/cost facts, or explicit blockers.

    The canonical job id must agree across every present artifact.  An execution
    admission digest is carried only when the run manifest or preserved request
    provenance records it, and conflicting provenance blocks the projection.
    """

    root = Path(job_dir).resolve()
    blockers: list[str] = []
    manifest = _read_object(root / "job_run_manifest.json", blockers, "job_run_manifest")
    metrics = _read_object(
        root / "simulator_command_batch_metrics.json", blockers, "simulator_command_batch_metrics"
    )
    ledger = _read_object(root / "gpu_cost_control_ledger.json", blockers, "gpu_cost_control_ledger")
    request = {}
    if (root / "job_request.json").is_file():
        request = _read_object(root / "job_request.json", blockers, "job_request")

    if manifest and manifest.get("schema_version") != TERMINAL_MANIFEST_SCHEMA:
        blockers.append("job_run_manifest_schema_invalid")
    if ledger and ledger.get("schema_version") != COST_LEDGER_SCHEMA:
        blockers.append("gpu_cost_control_ledger_schema_invalid")

    ids = {
        name: _string(payload.get("job_id"))
        for name, payload in (("manifest", manifest), ("ledger", ledger), ("request", request))
        if payload and _string(payload.get("job_id"))
    }
    job_id = ids.get("manifest") or ids.get("request") or ids.get("ledger") or root.name
    if not ids.get("manifest"):
        blockers.append("canonical_job_id_missing_from_manifest")
    if any(value != job_id for value in ids.values()):
        blockers.append("canonical_job_id_mismatch")

    claim_boundary = _mapping(manifest.get("claim_boundary")) or _mapping(
        manifest.get("proof_boundary")
    )
    if manifest:
        if manifest.get("status") != "simulator_command_completed":
            blockers.append("job_run_manifest_not_terminal_completed")
        if manifest.get("simulator_service_status") != "completed":
            blockers.append("simulator_service_not_completed")
        if claim_boundary.get("simulator_execution_proven") is not True:
            blockers.append("simulator_execution_not_proven")
        if manifest.get("blockers"):
            blockers.append("job_run_manifest_has_blockers")

    attempts = _integer(metrics.get("attempt_count"))
    successes = _integer(
        metrics.get("passed_attempt_count")
        if "passed_attempt_count" in metrics
        else metrics.get("success_count")
    )
    if attempts is None or attempts <= 0:
        blockers.append("episode_attempt_count_not_observed")
    if successes is None or attempts is None or successes > attempts:
        blockers.append("episode_success_count_invalid")
    if metrics and metrics.get("scenario_eval_run_coverage_complete") is not True:
        blockers.append("episode_coverage_not_complete")

    gpu_time = _mapping(ledger.get("gpu_time"))
    actual_gpu_seconds = _number(gpu_time.get("actual_gpu_seconds"))
    if gpu_time.get("actual_gpu_time_record_present") is not True or actual_gpu_seconds is None:
        blockers.append("actual_gpu_time_not_observed")
    observed_cost_usd = _number(
        ledger.get("actual_cost_usd")
        if "actual_cost_usd" in ledger
        else ledger.get("cost_usd")
    )
    if ledger.get("status") == "fixture_local_no_gpu" and actual_gpu_seconds == 0:
        observed_cost_usd = 0.0
    if observed_cost_usd is None:
        blockers.append("actual_cost_usd_not_observed")
    if ledger.get("blockers"):
        blockers.append("gpu_cost_control_ledger_has_blockers")

    manifest_digest = _first_string(
        manifest,
        (("execution_admission_digest",), ("request_provenance", "execution_admission_digest")),
    )
    request_digest = _first_string(
        request,
        (("execution_admission_digest",), ("source", "execution_admission_digest"), ("provenance", "execution_admission_digest")),
    )
    admission_digest = manifest_digest or request_digest or None
    if manifest_digest and request_digest and manifest_digest != request_digest:
        blockers.append("execution_admission_digest_mismatch")

    unique_blockers = list(dict.fromkeys(blockers))
    return {
        "status": "terminal_observed" if not unique_blockers else "blocked",
        "job_id": job_id,
        "execution_admission_digest": admission_digest,
        "episode_result": {
            "episodes_run": attempts,
            "episodes_succeeded": successes,
            "metrics_path": str(root / "simulator_command_batch_metrics.json"),
        },
        "cost": {
            "provider": ledger.get("provider"),
            "ledger_status": ledger.get("status"),
            "actual_gpu_seconds": actual_gpu_seconds,
            "actual_gpu_time_source": gpu_time.get("actual_gpu_time_source"),
            "observed_cost_usd": observed_cost_usd,
            "ledger_path": str(root / "gpu_cost_control_ledger.json"),
        },
        "blockers": unique_blockers,
    }
