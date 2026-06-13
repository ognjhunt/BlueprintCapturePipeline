"""Collect gated RunPod live execution proof for robot-eval workers."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .runpod_provider_adapter import (
    RUNPOD_API_GATE_ENV,
    RUNPOD_API_KEY_ENV,
    RUNPOD_API_KEY_FILE_ENV,
    RUNPOD_CONFIG_FILE_ENV,
    RUNPOD_REST_API_BASE,
    _read_runpod_api_key,
    _redact_runtime_value,
    _redact_text,
)


RUNPOD_LIVE_EXECUTION_PROOF_SCHEMA_VERSION = "runpod_live_execution_proof.v1"
RUNPOD_GPU_LAUNCH_GATE_ENV = "BLUEPRINT_ALLOW_GPU_PROVIDER_LAUNCH"


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _redact(text: str, api_key: str) -> str:
    return _redact_text(text, api_key)


def _read_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _derive_pod_id(adapter_result: Mapping[str, Any], explicit_pod_id: str | None) -> str:
    if explicit_pod_id:
        return explicit_pod_id
    direct_id = _string(adapter_result.get("id"))
    if direct_id:
        return direct_id
    response = _mapping(adapter_result.get("runpod_response"))
    response_id = _string(response.get("id"))
    if response_id:
        return response_id
    data = _mapping(response.get("data"))
    created = _mapping(data.get("podFindAndDeployOnDemand"))
    return _string(created.get("id"))


def _runtime_manifest_proof(runtime_manifest: Mapping[str, Any]) -> dict[str, Any]:
    runtime_blockers = [
        str(item)
        for item in runtime_manifest.get("blockers", [])
        if isinstance(item, (str, int, float))
    ]
    job_blockers = [
        str(item)
        for item in runtime_manifest.get("job_blockers", [])
        if isinstance(item, (str, int, float))
    ]
    preflight_blockers = [
        str(item)
        for item in runtime_manifest.get("runtime_preflight_blockers", [])
        if isinstance(item, (str, int, float))
    ]
    startup_blockers = [
        str(item)
        for item in runtime_manifest.get("startup_architecture_blockers", [])
        if isinstance(item, (str, int, float))
    ]
    signed_put_upload = _mapping(runtime_manifest.get("signed_put_runtime_manifest_upload"))
    signed_put_completed = signed_put_upload.get("status") == "completed"
    runtime_worker_completed = (
        runtime_manifest.get("schema_version") == "robot_eval_worker_runtime_manifest.v1"
        and runtime_manifest.get("status") == "completed"
        and not runtime_blockers
        and runtime_manifest.get("job_status") == "simulator_command_completed"
        and not job_blockers
        and runtime_manifest.get("runtime_preflight_status") == "passed"
        and not preflight_blockers
        and runtime_manifest.get("startup_architecture_audit_status") == "passed"
        and not startup_blockers
        and runtime_manifest.get("scenario_eval_matrix_status") == "completed"
        and runtime_manifest.get("simulator_service_status") == "completed"
        and runtime_manifest.get("evaluation_status") == "completed"
        and runtime_manifest.get("simulator_execution_proven") is True
        and runtime_manifest.get("robot_readiness_proven") is False
        and runtime_manifest.get("public_claim_upgrade_allowed") is False
        and signed_put_completed
    )
    return {
        "runtime_manifest_status": runtime_manifest.get("status"),
        "runtime_manifest_job_status": runtime_manifest.get("job_status"),
        "runtime_manifest_runtime_preflight_status": runtime_manifest.get(
            "runtime_preflight_status"
        ),
        "runtime_manifest_startup_architecture_audit_status": runtime_manifest.get(
            "startup_architecture_audit_status"
        ),
        "runtime_manifest_signed_put_upload_status": signed_put_upload.get("status"),
        "runtime_manifest_worker_completed": runtime_worker_completed,
        "runtime_manifest_simulator_execution_proven": (
            runtime_manifest.get("simulator_execution_proven") is True
        ),
        "runtime_manifest_robot_readiness_proven": (
            runtime_manifest.get("robot_readiness_proven") is True
        ),
        "runtime_manifest_public_claim_upgrade_allowed": (
            runtime_manifest.get("public_claim_upgrade_allowed") is True
        ),
        "runtime_manifest_blockers": runtime_blockers,
        "runtime_manifest_job_blockers": job_blockers,
        "runtime_manifest_runtime_preflight_blockers": preflight_blockers,
        "runtime_manifest_startup_architecture_blockers": startup_blockers,
    }


def _gate_blockers(*, allow_runpod_api_call: bool, api_key: str) -> list[str]:
    blockers: list[str] = []
    if not _env_truthy(RUNPOD_API_GATE_ENV):
        blockers.append(f"missing_env_{RUNPOD_API_GATE_ENV}")
    if not _env_truthy(RUNPOD_GPU_LAUNCH_GATE_ENV):
        blockers.append(f"missing_env_{RUNPOD_GPU_LAUNCH_GATE_ENV}")
    if not allow_runpod_api_call:
        blockers.append("missing_cli_allow_runpod_api_call")
    if not api_key:
        blockers.append(
            f"missing_env_{RUNPOD_API_KEY_ENV}_or_{RUNPOD_API_KEY_FILE_ENV}_or_{RUNPOD_CONFIG_FILE_ENV}"
        )
    return blockers


def _http_json(
    *,
    url: str,
    payload: Mapping[str, Any] | None,
    method: str,
    api_key: str,
    timeout_seconds: int,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status_code = int(getattr(response, "status", 200))
        response_text = response.read().decode("utf-8", errors="replace")
    if not response_text.strip():
        return status_code, {}
    parsed = json.loads(response_text)
    return status_code, dict(parsed) if isinstance(parsed, Mapping) else {"response": parsed}


def _pods_from_response(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(response.get("items"), list):
        return [dict(pod) for pod in response["items"] if isinstance(pod, Mapping)]
    if isinstance(response.get("pods"), list):
        return [dict(pod) for pod in response["pods"] if isinstance(pod, Mapping)]
    if isinstance(response.get("response"), list):
        return [dict(pod) for pod in response["response"] if isinstance(pod, Mapping)]
    data = _mapping(response.get("data"))
    myself = _mapping(data.get("myself"))
    pods = myself.get("pods")
    if not isinstance(pods, list):
        return []
    return [dict(pod) for pod in pods if isinstance(pod, Mapping)]


def _active_pod_count(pods: Sequence[Mapping[str, Any]]) -> int:
    inactive_statuses = {"EXITED", "STOPPED", "TERMINATED"}
    count = 0
    for pod in pods:
        status = _string(pod.get("desiredStatus") or pod.get("status")).upper()
        if not status or status not in inactive_statuses:
            count += 1
    return count


def collect_runpod_live_execution_proof(
    *,
    provider_launch_request_path: str | Path,
    adapter_result_path: str | Path | None = None,
    runtime_manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
    pod_id: str | None = None,
    stop_pod: bool = False,
    allow_runpod_api_call: bool = False,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    request_path = Path(provider_launch_request_path).expanduser().resolve()
    adapter_path = Path(adapter_result_path).expanduser().resolve() if adapter_result_path else None
    runtime_path = (
        Path(runtime_manifest_path).expanduser().resolve() if runtime_manifest_path else None
    )
    resolved_output = (
        Path(output_path).expanduser().resolve()
        if output_path
        else request_path.parent / "runpod_live_execution_proof.json"
    )
    ensure_dir(resolved_output.parent)
    provider_request = _read_mapping(request_path)
    adapter_result = _read_mapping(adapter_path)
    runtime_manifest = _read_mapping(runtime_path)
    api_key, api_key_meta = _read_runpod_api_key()
    resolved_pod_id = _derive_pod_id(adapter_result, pod_id)
    runtime_proof = _runtime_manifest_proof(runtime_manifest) if runtime_manifest else {}
    result: dict[str, Any] = {
        "schema_version": RUNPOD_LIVE_EXECUTION_PROOF_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider_launch_request_path": str(request_path),
        "adapter_result_path": str(adapter_path) if adapter_path else None,
        "runtime_manifest_path": str(runtime_path) if runtime_path else None,
        "output_path": str(resolved_output),
        "job_id": _string(provider_request.get("job_id")) or _string(adapter_result.get("job_id")),
        "pod_id": resolved_pod_id or None,
        "stop_pod_requested": stop_pod,
        "api_call_performed": False,
        "runpod_side_effects_may_have_occurred": False,
        "secret_values_in_artifact": False,
        "raw_api_key_stored": False,
        "active_pod_count_before": None,
        "active_pod_count_after": None,
        "pod_stop_performed": False,
        "shutdown_or_termination_proof": False,
        "production_runpod_worker_execution_proven": False,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "public_claim_upgrade_allowed": False,
        **runtime_proof,
        **api_key_meta,
    }
    gate_blockers = _gate_blockers(
        allow_runpod_api_call=allow_runpod_api_call,
        api_key=api_key,
    )
    if gate_blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "runpod_live_proof_gate_blocked",
                "blockers": gate_blockers,
                **api_key_meta,
            }
        )
        write_json(resolved_output, result)
        return result
    try:
        before_status, before_response = _http_json(
            url=f"{RUNPOD_REST_API_BASE}/pods",
            payload=None,
            method="GET",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        before_pods = _pods_from_response(before_response)
        result.update(
            {
                "api_call_performed": True,
                "runpod_side_effects_may_have_occurred": False,
                "before_http_status_code": before_status,
                "pods_before": _redact_runtime_value(before_pods),
                "active_pod_count_before": _active_pod_count(before_pods),
            }
        )
        stop_response: dict[str, Any] | None = None
        if stop_pod:
            if not resolved_pod_id:
                result.setdefault("blockers", []).append("missing_pod_id_for_stop")
            else:
                stop_status, stop_response = _http_json(
                    url=f"{RUNPOD_REST_API_BASE}/pods/{resolved_pod_id}/stop",
                    payload=None,
                    method="POST",
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
                result.update(
                    {
                        "runpod_side_effects_may_have_occurred": True,
                        "pod_stop_performed": True,
                        "stop_http_status_code": stop_status,
                        "stop_response": _redact_runtime_value(stop_response),
                    }
                )
        after_status, after_response = _http_json(
            url=f"{RUNPOD_REST_API_BASE}/pods",
            payload=None,
            method="GET",
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        after_pods = _pods_from_response(after_response)
        result.update(
            {
                "after_http_status_code": after_status,
                "pods_after": _redact_runtime_value(after_pods),
                "active_pod_count_after": _active_pod_count(after_pods),
            }
        )
        blockers = list(result.get("blockers") or [])
        if runtime_path and not runtime_manifest:
            blockers.append("runtime_manifest_missing")
        if result["active_pod_count_before"] is None or result["active_pod_count_after"] is None:
            blockers.append("active_pod_counts_not_verified")
        if stop_pod and not result.get("pod_stop_performed"):
            blockers.append("pod_stop_not_performed")
        if stop_pod and result.get("active_pod_count_after", 0) > result.get(
            "active_pod_count_before", 0
        ):
            blockers.append("active_pod_count_increased_after_stop")
        shutdown_proof = bool(stop_pod and result.get("pod_stop_performed") and not blockers)
        adapter_submitted_pod = (
            adapter_result.get("status") == "submitted"
            and adapter_result.get("api_call_performed") is True
            and adapter_result.get("provider_job_submitted") is True
        )
        runtime_worker_completed = result.get("runtime_manifest_worker_completed") is True
        production_worker_execution_proven = bool(
            adapter_submitted_pod and runtime_worker_completed and shutdown_proof
        )
        result.update(
            {
                "status": "runpod_live_proof_collected" if not blockers else "blocked",
                "reason": "runpod_live_api_calls_completed",
                "blockers": blockers,
                "shutdown_or_termination_proof": shutdown_proof,
                "production_runpod_worker_execution_proven": production_worker_execution_proven,
                "simulator_execution_proven": bool(
                    production_worker_execution_proven
                    and result.get("runtime_manifest_simulator_execution_proven") is True
                ),
                "robot_readiness_proven": False,
                "public_claim_upgrade_allowed": False,
            }
        )
    except urllib.error.HTTPError as exc:
        error_body = _redact(exc.read().decode("utf-8", errors="replace"), api_key)
        result.update(
            {
                "status": "failed",
                "reason": "runpod_live_proof_http_error",
                "blockers": ["runpod_live_proof_http_error"],
                "api_call_performed": True,
                "runpod_side_effects_may_have_occurred": True,
                "http_status_code": exc.code,
                "runpod_error": error_body,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive around provider/network failures
        result.update(
            {
                "status": "failed",
                "reason": "runpod_live_proof_call_failed",
                "blockers": ["runpod_live_proof_call_failed"],
                "api_call_performed": True,
                "runpod_side_effects_may_have_occurred": True,
                "error_type": type(exc).__name__,
                "error": _redact(str(exc), api_key),
            }
        )
    write_json(resolved_output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-launch-request", required=True)
    parser.add_argument("--adapter-result")
    parser.add_argument("--runtime-manifest")
    parser.add_argument("--output-path")
    parser.add_argument("--pod-id")
    parser.add_argument("--stop-pod", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--allow-runpod-api-call",
        action="store_true",
        help=(
            f"Required with {RUNPOD_API_GATE_ENV}=true and "
            f"{RUNPOD_GPU_LAUNCH_GATE_ENV}=true for live RunPod API calls."
        ),
    )
    args = parser.parse_args(argv)
    result = collect_runpod_live_execution_proof(
        provider_launch_request_path=args.provider_launch_request,
        adapter_result_path=args.adapter_result,
        runtime_manifest_path=args.runtime_manifest,
        output_path=args.output_path,
        pod_id=args.pod_id,
        stop_pod=args.stop_pod,
        allow_runpod_api_call=args.allow_runpod_api_call,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[runpod-live-execution-proof] result={result['output_path']}")
    print(f"[runpod-live-execution-proof] status={result['status']}")
    blockers = result.get("blockers")
    if blockers:
        print("[runpod-live-execution-proof] blockers=" + ",".join(blockers))
    return 0 if result["status"] == "runpod_live_proof_collected" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
