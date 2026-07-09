"""No-spend launcher contract for robot-eval provider-race handoffs."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .logging_utils import log_event


PROVIDER_RACE_LAUNCHER_RESULT_SCHEMA_VERSION = (
    "robot_eval_gpu_provider_race_launcher_result.v1"
)
PROVIDER_RACE_RUNTIME_RESULT_SCHEMA_VERSION = (
    "robot_eval_gpu_provider_race_runtime_result.v1"
)
PROVIDER_RACE_HANDOFF_SCHEMA_VERSION = "robot_eval_gpu_provider_race_handoff.v1"
PROVIDER_LAUNCH_REQUEST_SCHEMA_VERSION = "robot_eval_gpu_provider_launch_request.v1"
ALLOW_PROVIDER_RACE_LAUNCH_ENV = "BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH"
# Paid-lane name for pending-teardown records opened by the eval race runtime. The
# render path uses its own lane (ISAAC_G1_KITCHEN_PARITY_LANE); the eval race gets a
# distinct lane so orphan-reaping and teardown proofs stay attributable to eval jobs.
PROVIDER_RACE_RUNTIME_LANE = "robot_eval_provider_race_runtime"
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


def race_eval_providers(
    *,
    providers: Sequence,
    request,
    marker_check: Callable[[object, dict], bool],
    job_dir,
    marker_timeout: float = 180.0,
    poll_interval: float = 10.0,
    circuit_breaker=None,
    prelaunch_guard: Mapping[str, Any] | Callable[[object], Mapping[str, Any]] | None = None,
    launch_kwargs: Mapping[str, Any] | Callable[[object], Mapping[str, Any]] | None = None,
    cold: bool = False,
    terminate_losers: bool = True,
    pending_teardown_lane: str | None = PROVIDER_RACE_RUNTIME_LANE,
    pending_teardown_max_age_seconds: int = 7200,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict:
    """Race an eval GPU launch across providers, reusing the render race runtime.

    This is the RUNTIME that was previously "not implemented" for the customer eval
    path: it delegates straight to :func:`blueprint_pipeline.provider_race.race_launch`
    (the same parallel racer + :class:`ProviderCircuitBreaker` the render path uses),
    so a degraded provider can no longer stall an eval job — the first provider to
    show its boot marker wins and every loser is torn down with provider-API proof.

    The racer only touches ``provider.launch``/``terminate``/``stop``/``inspect``; the
    caller supplies live provider objects (or hermetic fakes in tests). No provider
    SDK, cloud dependency, or credential is imported here.
    """
    from .provider_race import ProviderCircuitBreaker, race_launch

    breaker = circuit_breaker if circuit_breaker is not None else ProviderCircuitBreaker()
    clock_kwargs: dict[str, Any] = {}
    if sleep is not None:
        clock_kwargs["sleep"] = sleep
    if monotonic is not None:
        clock_kwargs["monotonic"] = monotonic
    race = race_launch(
        list(providers),
        request,
        marker_check,
        marker_timeout,
        job_dir=job_dir,
        cold=cold,
        poll_interval=poll_interval,
        circuit_breaker=breaker,
        terminate_losers=terminate_losers,
        launch_kwargs=launch_kwargs,
        prelaunch_guard=prelaunch_guard,
        pending_teardown_lane=pending_teardown_lane,
        pending_teardown_max_age_seconds=pending_teardown_max_age_seconds,
        **clock_kwargs,
    )
    race["circuit_breaker"] = breaker.snapshot()
    return race


def _live_race_gate_blockers(allow_live_provider_race: bool) -> list[str]:
    blockers: list[str] = []
    if not _env_truthy(ALLOW_PROVIDER_RACE_LAUNCH_ENV):
        blockers.append(f"missing_env_{ALLOW_PROVIDER_RACE_LAUNCH_ENV}")
    if not allow_live_provider_race:
        blockers.append("missing_cli_allow_live_provider_race")
    return blockers


def run_robot_eval_provider_race_runtime(
    *,
    provider_launch_request_path: str | Path,
    handoff_path: str | Path | None = None,
    output_path: str | Path | None = None,
    providers: Sequence | None = None,
    provider_factory: Callable[..., Sequence] | None = None,
    marker_check: Callable[[object, dict], bool] | None = None,
    request_builder=None,
    job_dir: str | Path | None = None,
    marker_timeout: float = 180.0,
    poll_interval: float = 10.0,
    circuit_breaker=None,
    launch_kwargs: Mapping[str, Any] | Callable[[object], Mapping[str, Any]] | None = None,
    allow_live_provider_race: bool = False,
    live_provider_race: bool = False,
    cold: bool = False,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict:
    """Validate a provider-race handoff and then actually race across providers.

    Flow:

    1. Validate the handoff through :func:`run_robot_eval_provider_race_launcher`
       (the existing no-spend gate). If it is not ready, this returns blocked with the
       same blockers and touches no provider.
    2. If ready and provider objects are supplied (directly or via ``provider_factory``),
       run the live-capable race via :func:`race_eval_providers`.
    3. If ready but no providers are supplied, return a no-spend, dry-run-verifiable
       result that proves the runtime is wired and names exactly what still needs live
       credentials — never a false ``provider_race_execution_performed``.

    ``live_provider_race`` marks whether the supplied providers make real (billable)
    provider API calls. When True, an explicit env + flag gate
    (``BLUEPRINT_ALLOW_GPU_PROVIDER_RACE_LAUNCH`` + ``allow_live_provider_race``) is
    required before any launch. Hermetic fakes leave it False and need no gate.
    """

    request_path = Path(provider_launch_request_path).resolve()
    request, _request_error = _read_mapping(request_path)
    resolved_handoff_path = _resolve_handoff_path(
        request_path=request_path,
        request=request,
        handoff_path=handoff_path,
    )
    resolved_output = (
        Path(output_path).resolve()
        if output_path
        else request_path.parent / "gpu_provider_race_runtime_result.json"
    )
    ensure_dir(resolved_output.parent)

    # 1) Reuse the no-spend handoff gate. It writes the standard launcher-result
    #    artifact the handoff points at; the runtime result is written separately.
    validation = run_robot_eval_provider_race_launcher(
        provider_launch_request_path=request_path,
        handoff_path=resolved_handoff_path,
    )
    validation_blockers = _string_list(validation.get("blockers"))

    result: dict[str, Any] = {
        "schema_version": PROVIDER_RACE_RUNTIME_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider_launch_request_path": str(request_path),
        "provider_race_handoff_path": str(resolved_handoff_path),
        "output_path": str(resolved_output),
        "job_id": _string(request.get("job_id")) or validation.get("job_id"),
        "provider_race_launcher_result_status": validation.get("status"),
        "provider_race_runtime_launcher_implemented": True,
        "provider_race_runtime_wired": bool(
            validation.get("status") == "ready_for_live_provider_race"
        ),
        "provider_race_execution_performed": False,
        "provider_race_execution_proven": False,
        "live_provider_calls_performed": False,
        "provider_race_lane": PROVIDER_RACE_RUNTIME_LANE,
        "simulator_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
        "winner_provider": None,
        "failover_selected": False,
        "contenders": [],
        "skipped": [],
        "terminated_losers": 0,
    }

    def _finish(status: str, *, reason: str, blockers: Sequence[str]) -> dict:
        result["status"] = status
        result["reason"] = reason
        result["blockers"] = _dedupe([str(b) for b in blockers if str(b or "").strip()])
        result["claim_boundary"] = {
            # The runtime launcher exists now — this claim is always False here.
            "provider_race_runtime_launcher_not_implemented": False,
            "provider_race_runtime_launcher_implemented": True,
            "provider_race_execution_is_not_simulator_or_rank_proof": True,
            "live_provider_calls_performed": bool(
                result.get("live_provider_calls_performed")
            ),
            "provider_race_execution_performed": bool(
                result.get("provider_race_execution_performed")
            ),
        }
        write_json(resolved_output, result)
        log_event(
            logger,
            logging.INFO
            if status in {"provider_race_executed", "ready_for_live_provider_race_runtime"}
            else logging.WARNING,
            "robot_eval_provider_race_runtime." + status,
            output_path=str(resolved_output),
            job_id=result.get("job_id"),
            status=status,
            winner_provider=result.get("winner_provider"),
            failover_selected=result.get("failover_selected"),
            blocker_count=len(result["blockers"]),
            blockers=result["blockers"],
            live_provider_calls_performed=result.get("live_provider_calls_performed"),
        )
        return result

    if validation.get("status") != "ready_for_live_provider_race":
        return _finish(
            "blocked",
            reason="provider_race_handoff_not_ready_for_runtime",
            blockers=validation_blockers or ["provider_race_handoff_not_ready"],
        )

    # 2) Resolve providers. Injected objects win; else a factory builds them; else
    #    there is nothing live to race and we return a verifiable dry-run.
    race_providers = list(providers) if providers else None
    if race_providers is None and provider_factory is not None:
        runnable_candidates = [
            _mapping(item)
            for item in _read_mapping(resolved_handoff_path)[0].get("runnable_candidates")
            or []
            if isinstance(item, Mapping)
        ]
        built = provider_factory(
            runnable_candidates=runnable_candidates,
            request=request,
            handoff_path=resolved_handoff_path,
        )
        race_providers = list(built) if built else None

    if not race_providers:
        result["needs_live_provider_credentials"] = True
        return _finish(
            "ready_for_live_provider_race_runtime",
            reason="provider_race_runtime_wired_no_providers_supplied",
            blockers=[],
        )

    if marker_check is None:
        return _finish(
            "blocked",
            reason="provider_race_runtime_marker_check_missing",
            blockers=["provider_race_runtime_marker_check_missing"],
        )

    # Live (billable) races require the explicit env + flag gate. Hermetic fakes
    # (live_provider_race=False) make no real calls and need no gate.
    if live_provider_race:
        gate_blockers = _live_race_gate_blockers(allow_live_provider_race)
        if gate_blockers:
            return _finish(
                "blocked",
                reason="live_provider_race_gate_blocked",
                blockers=gate_blockers,
            )

    resolved_job_dir = Path(job_dir) if job_dir else resolved_output.parent / "provider_race_runtime"
    ensure_dir(resolved_job_dir)
    prelaunch_guard = _mapping(request.get("prelaunch_spend_guard")) or None
    race_request = request_builder if request_builder is not None else request

    race = race_eval_providers(
        providers=race_providers,
        request=race_request,
        marker_check=marker_check,
        job_dir=resolved_job_dir,
        marker_timeout=marker_timeout,
        poll_interval=poll_interval,
        circuit_breaker=circuit_breaker,
        prelaunch_guard=prelaunch_guard,
        launch_kwargs=launch_kwargs,
        cold=cold,
        sleep=sleep,
        monotonic=monotonic,
    )
    winner = _string(race.get("provider")) or None
    first_priority = _string(getattr(race_providers[0], "name", "")) or None
    result.update(
        {
            "provider_race_execution_performed": True,
            "live_provider_calls_performed": bool(live_provider_race),
            "winner_provider": winner,
            "winner_instance_id": race.get("instance_id"),
            "winner_mode": race.get("mode"),
            "failover_selected": bool(winner and first_priority and winner != first_priority),
            "first_priority_provider": first_priority,
            "contenders": race.get("contenders") or [],
            "skipped": race.get("skipped") or [],
            "terminated_losers": race.get("terminated_losers") or 0,
            "circuit_breaker": race.get("circuit_breaker") or {},
            "pending_teardown_record": race.get("pending_teardown_record"),
            "race_result": {k: v for k, v in race.items() if k != "winner_provider"},
        }
    )
    if race.get("status") == "launched":
        return _finish(
            "provider_race_executed",
            reason="provider_race_winner_selected",
            blockers=[],
        )
    return _finish(
        "provider_race_blocked",
        reason=_string(race.get("reason")) or "provider_race_all_providers_dudded",
        blockers=[_string(race.get("reason")) or "provider_race_all_providers_dudded"],
    )


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
    parser.add_argument(
        "--run-provider-race",
        action="store_true",
        help=(
            "Run the provider-race runtime after validation. Without injected provider "
            "objects (only available in-process) this proves the runtime is wired and "
            "reports that live provider credentials are still required; it never spends."
        ),
    )
    args = parser.parse_args(argv)
    try:
        request_path = _request_path_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.run_provider_race:
        result = run_robot_eval_provider_race_runtime(
            provider_launch_request_path=request_path,
            handoff_path=args.handoff,
            output_path=args.output_path,
        )
        print(f"[robot-eval-provider-race-runtime] result={result['output_path']}")
        print(f"[robot-eval-provider-race-runtime] status={result['status']}")
        print(f"[robot-eval-provider-race-runtime] job_id={result.get('job_id')}")
        blockers = result.get("blockers")
        if blockers:
            print(
                "[robot-eval-provider-race-runtime] blockers="
                + ",".join(str(item) for item in blockers)
            )
        return 0 if result["status"] in {
            "provider_race_executed",
            "ready_for_live_provider_race_runtime",
        } else 1
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
