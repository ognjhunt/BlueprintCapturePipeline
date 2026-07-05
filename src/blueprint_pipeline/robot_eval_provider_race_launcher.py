"""No-spend launcher contract for robot-eval provider-race handoffs."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .logging_utils import log_event


PROVIDER_RACE_LAUNCHER_RESULT_SCHEMA_VERSION = (
    "robot_eval_gpu_provider_race_launcher_result.v1"
)
PROVIDER_RACE_HANDOFF_SCHEMA_VERSION = "robot_eval_gpu_provider_race_handoff.v1"
PROVIDER_LAUNCH_REQUEST_SCHEMA_VERSION = "robot_eval_gpu_provider_launch_request.v1"
ALLOW_PROVIDER_RACE_LAUNCH_ENV = "BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH"
logger = logging.getLogger(__name__)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [item for item in (_string(item) for item in value) if item]
    return []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _read_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = read_json_any(path)
    except Exception as exc:  # noqa: BLE001 - launcher artifacts must fail closed
        return {}, type(exc).__name__
    if not isinstance(payload, Mapping):
        return {}, "not_mapping"
    return dict(payload), None


def _provider_race_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    prelaunch_guard = _mapping(request.get("prelaunch_spend_guard"))
    return _mapping(prelaunch_guard.get("provider_race") or request.get("provider_race"))


def _resolve_handoff_path(
    *,
    request_path: Path,
    request: Mapping[str, Any],
    handoff_path: str | Path | None,
) -> Path:
    provider_race = _provider_race_contract(request)
    raw_path = (
        str(handoff_path)
        if handoff_path is not None
        else _string(request.get("provider_race_handoff_path"))
        or _string(provider_race.get("provider_race_handoff_path"))
        or "gpu_provider_race_handoff.json"
    )
    path = Path(raw_path)
    return path if path.is_absolute() else request_path.parent / path


def _candidate_count(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _base_result(
    *,
    request_path: Path,
    handoff_path: Path,
    output_path: Path,
    request: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    provider_race = _provider_race_contract(request)
    return {
        "schema_version": PROVIDER_RACE_LAUNCHER_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider_launch_request_path": str(request_path),
        "provider_race_handoff_path": str(handoff_path),
        "output_path": str(output_path),
        "job_id": _string(request.get("job_id") or handoff.get("job_id")),
        "provider": _string(request.get("provider")) or None,
        "provider_race": provider_race or None,
        "provider_race_required_for_customer_path": bool(
            provider_race.get("race_required_for_customer_path")
            or handoff.get("provider_race_required_for_customer_path")
        ),
        "provider_race_launcher_available": True,
        "live_provider_calls_performed": False,
        "provider_race_execution_performed": False,
        "provider_race_execution_proven": False,
        "remote_cloud_execution_proven": False,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "claim_boundary": {
            "provider_race_launcher_result_is_not_provider_execution": True,
            "live_provider_calls_performed": False,
            "provider_race_execution_proven": False,
            "remote_cloud_execution_proven": False,
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
        },
    }


def _handoff_blockers(
    *,
    request: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if request.get("schema_version") != PROVIDER_LAUNCH_REQUEST_SCHEMA_VERSION:
        blockers.append("invalid_provider_launch_request_schema")
    if handoff.get("schema_version") != PROVIDER_RACE_HANDOFF_SCHEMA_VERSION:
        blockers.append("invalid_provider_race_handoff_schema")
    request_job_id = _string(request.get("job_id"))
    handoff_job_id = _string(handoff.get("job_id"))
    if request_job_id and handoff_job_id and request_job_id != handoff_job_id:
        blockers.append("provider_race_handoff_job_id_mismatch")
    if handoff.get("provider_race_required_for_customer_path") is not True:
        blockers.append("provider_race_handoff_does_not_require_customer_race")
    if handoff.get("live_provider_calls_performed") is True:
        blockers.append("provider_race_handoff_unexpected_live_provider_calls")
    if _candidate_count(handoff.get("race_candidate_count")) < 2:
        blockers.append("provider_race_handoff_requires_two_race_candidates")
    if _candidate_count(handoff.get("runnable_candidate_count")) < 2:
        blockers.append("provider_race_handoff_requires_two_runnable_candidates")
    if handoff.get("provider_race_runtime_launcher_available") is not True:
        blockers.append("provider_race_launcher_command_not_declared")
    if not _string(handoff.get("launcher_command")):
        blockers.append("provider_race_launcher_command_missing")
    return blockers


def _runtime_blockers(handoff: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    handoff_status = _string(handoff.get("status"))
    if handoff_status != "ready_for_customer_provider_race_runtime":
        blockers.append("provider_race_handoff_not_ready")
    if handoff.get("customer_path_provider_failover_runtime_wired") is not True:
        blockers.append("customer_path_provider_failover_runtime_not_wired")
    blockers.extend(_string_list(handoff.get("blockers")))
    blockers.extend(
        _string_list(handoff.get("customer_path_provider_failover_runtime_blockers"))
    )
    blockers.extend(
        _string_list(handoff.get("provider_race_runtime_launcher_blockers"))
    )
    return blockers


def run_robot_eval_provider_race_launcher(
    *,
    provider_launch_request_path: str | Path,
    handoff_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a provider-race handoff and write a no-spend launcher result.

    The command intentionally stops before provider API calls. It proves the
    customer path has a runnable race-launcher artifact contract, not that a live
    multi-provider race, teardown, simulator run, or rank result occurred.
    """

    request_path = Path(provider_launch_request_path).resolve()
    request, request_error = _read_mapping(request_path)
    resolved_handoff_path = _resolve_handoff_path(
        request_path=request_path,
        request=request,
        handoff_path=handoff_path,
    )
    resolved_output = (
        Path(output_path).resolve()
        if output_path
        else request_path.parent / "gpu_provider_race_launcher_result.json"
    )
    ensure_dir(resolved_output.parent)

    handoff, handoff_error = _read_mapping(resolved_handoff_path)
    result = _base_result(
        request_path=request_path,
        handoff_path=resolved_handoff_path,
        output_path=resolved_output,
        request=request,
        handoff=handoff,
    )

    structural_blockers: list[str] = []
    if request_error:
        structural_blockers.append(f"provider_launch_request_{request_error}")
    if handoff_error:
        structural_blockers.append(f"provider_race_handoff_{handoff_error}")
    if not request_error and not handoff_error:
        structural_blockers.extend(
            _handoff_blockers(request=request, handoff=handoff)
        )

    runtime_blockers = [] if structural_blockers else _runtime_blockers(handoff)
    blockers = _dedupe([*structural_blockers, *runtime_blockers])
    ready = not blockers
    result.update(
        {
            "status": "ready_for_live_provider_race" if ready else "blocked",
            "reason": "provider_race_launcher_ready"
            if ready
            else "provider_race_launcher_gate_blocked",
            "blockers": blockers,
            "structural_blockers": _dedupe(structural_blockers),
            "runtime_blockers": _dedupe(runtime_blockers),
            "provider_race_handoff_status": handoff.get("status"),
            "provider_race_handoff_ready": ready,
            "provider_race_runtime_launcher_available": bool(
                handoff.get("provider_race_runtime_launcher_available")
            ),
            "launcher_command": handoff.get("launcher_command"),
            "race_candidate_count": _candidate_count(handoff.get("race_candidate_count")),
            "runnable_candidate_count": _candidate_count(
                handoff.get("runnable_candidate_count")
            ),
            "allow_live_provider_race_env": ALLOW_PROVIDER_RACE_LAUNCH_ENV,
            "allow_live_provider_race_env_present": _env_truthy(
                ALLOW_PROVIDER_RACE_LAUNCH_ENV
            ),
        }
    )
    write_json(resolved_output, result)
    log_event(
        logger,
        logging.INFO if ready else logging.WARNING,
        "robot_eval_provider_race_launcher.ready"
        if ready
        else "robot_eval_provider_race_launcher.blocked",
        output_path=str(resolved_output),
        job_id=result.get("job_id"),
        status=result.get("status"),
        blocker_count=len(blockers),
        blockers=blockers,
        live_provider_calls_performed=False,
    )
    return result


def _request_path_from_args(args: argparse.Namespace) -> Path:
    if args.provider_launch_request:
        return Path(args.provider_launch_request)
    if args.job_dir:
        return Path(args.job_dir) / "gpu_provider_launch_request.json"
    raise ValueError("Provide --provider-launch-request or --job-dir")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a no-spend robot-eval GPU provider-race launcher handoff."
    )
    parser.add_argument("--provider-launch-request")
    parser.add_argument("--job-dir")
    parser.add_argument("--handoff")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)
    try:
        request_path = _request_path_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_robot_eval_provider_race_launcher(
        provider_launch_request_path=request_path,
        handoff_path=args.handoff,
        output_path=args.output_path,
    )
    print(f"[robot-eval-provider-race-launcher] result={result['output_path']}")
    print(f"[robot-eval-provider-race-launcher] status={result['status']}")
    print(f"[robot-eval-provider-race-launcher] job_id={result.get('job_id')}")
    blockers = result.get("blockers")
    if blockers:
        print(
            "[robot-eval-provider-race-launcher] blockers="
            + ",".join(str(item) for item in blockers)
        )
    return 0 if result["status"] == "ready_for_live_provider_race" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
