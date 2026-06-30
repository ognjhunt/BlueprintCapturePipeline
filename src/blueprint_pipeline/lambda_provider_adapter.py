"""Lambda Cloud adapter (stub) for prepared robot-eval GPU provider launch requests.

This is a wired-in stub: it validates a ``robot_eval_gpu_provider_launch_request.v1``
payload, resolves the Lambda Cloud API key using the same env/file convention as the
RunPod and Vast adapters (``LAMBDA_API_KEY`` / ``LAMBDA_API_KEY_FILE`` →
``~/.blueprint-secrets/lambda_api_key``), and emits a standard provider adapter result
artifact so the launcher can consume ``lambda_cloud`` like any other provider.

It deliberately never performs a live Lambda Cloud API call. Dry-run mode validates the
request and reports readiness; any live allocation mode is reported as ``blocked`` with a
``lambda_cloud_live_launch_not_implemented`` blocker until the allocation/teardown path is
built. This keeps the adapter honest: it proves request consumability, not GPU allocation.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import urlparse

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .logging_utils import log_event


LAMBDA_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION = "lambda_provider_adapter_result.v1"
LAMBDA_PROVIDER_NAME = "lambda_cloud"
LAMBDA_API_KEY_ENV = "LAMBDA_API_KEY"
LAMBDA_API_KEY_FILE_ENV = "LAMBDA_API_KEY_FILE"
DEFAULT_LAMBDA_API_KEY_FILE = "~/.blueprint-secrets/lambda_api_key"
LAMBDA_API_GATE_ENV = "BLUEPRINT_ALLOW_LAMBDA_API_CALLS"
PROVIDER_LAUNCH_REQUEST_ENV = "BLUEPRINT_GPU_PROVIDER_LAUNCH_REQUEST"
PROVIDER_ADAPTER_OUTPUT_ENV = "BLUEPRINT_GPU_PROVIDER_ADAPTER_OUTPUT"
LIVE_LAUNCH_NOT_IMPLEMENTED_BLOCKER = "lambda_cloud_live_launch_not_implemented"
LIVE_MODES = {"auto", "allocate"}
logger = logging.getLogger(__name__)


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _read_lambda_api_key() -> tuple[str, dict[str, Any]]:
    """Resolve the Lambda Cloud API key from env, then the file pointer.

    Mirrors the RunPod/Vast convention: ``LAMBDA_API_KEY`` wins, otherwise read the file
    named by ``LAMBDA_API_KEY_FILE`` (defaulting to ``~/.blueprint-secrets/lambda_api_key``).
    The raw key is never returned to the caller for persistence — only readiness metadata.
    """
    env_value = _string(os.getenv(LAMBDA_API_KEY_ENV))
    if env_value:
        return env_value, {
            "api_key_configured": True,
            "api_key_source": LAMBDA_API_KEY_ENV,
            "api_key_file_configured": False,
        }
    key_file = _string(os.getenv(LAMBDA_API_KEY_FILE_ENV)) or DEFAULT_LAMBDA_API_KEY_FILE
    path = Path(key_file).expanduser()
    if not path.is_file():
        return "", {
            "api_key_configured": False,
            "api_key_source": None,
            "api_key_file_configured": False,
            "api_key_file": str(path),
        }
    try:
        key = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return "", {
            "api_key_configured": False,
            "api_key_source": LAMBDA_API_KEY_FILE_ENV,
            "api_key_file_configured": True,
            "api_key_file": str(path),
            "api_key_file_read_error": type(exc).__name__,
        }
    return key, {
        "api_key_configured": bool(key),
        "api_key_source": LAMBDA_API_KEY_FILE_ENV if key else None,
        "api_key_file_configured": True,
        "api_key_file": str(path),
    }


def _provider_shape(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(request.get("provider_request_shape"))


def _inputs(request: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping(_provider_shape(request).get("inputs"))


def _request_summary(request: Mapping[str, Any]) -> Dict[str, Any]:
    inputs = _inputs(request)
    image = _mapping(_provider_shape(request).get("image"))
    return {
        "job_id": _string(request.get("job_id")),
        "provider": _string(request.get("provider")),
        "provider_launch_request_status": _string(request.get("status")),
        "operation": _string(request.get("operation"))
        or _string(_provider_shape(request).get("operation")),
        "worker_image_ref_present": bool(_string(image.get("configured_image_ref"))),
        "manifest_uri_present": bool(_string(inputs.get("manifest_uri"))),
        "capture_root_bundle_uri_present": bool(
            _string(inputs.get("capture_root_bundle_uri"))
        ),
        "artifact_output_uri_present": bool(_string(inputs.get("artifact_output_uri"))),
    }


def _request_blockers(request: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if request.get("schema_version") != "robot_eval_gpu_provider_launch_request.v1":
        blockers.append("invalid_provider_launch_request_schema")
    if _string(request.get("provider")) != LAMBDA_PROVIDER_NAME:
        blockers.append("provider_launch_request_not_lambda_cloud")
    if request.get("status") != "request_manifest_ready":
        blockers.append("provider_launch_request_not_ready")
    provider_input_setup = request.get("provider_input_setup")
    if isinstance(provider_input_setup, Mapping):
        setup_blockers = [
            _string(item)
            for item in provider_input_setup.get("blockers", [])
            if _string(item)
        ]
        if setup_blockers:
            blockers.append("provider_input_setup_blocked")
            blockers.extend(setup_blockers)
    inputs = _inputs(request)
    if not _string(inputs.get("manifest_uri")):
        blockers.append("missing_provider_worker_manifest_uri")
    if not _string(inputs.get("capture_root_bundle_uri")):
        blockers.append("missing_provider_capture_root_bundle_uri")
    artifact_output_uri = _string(inputs.get("artifact_output_uri"))
    artifact_output_required = _bool(inputs.get("artifact_output_uri_required"))
    if not artifact_output_uri and artifact_output_required is not False:
        blockers.append("missing_provider_artifact_output_uri")
    return _dedupe(blockers)


def _adapter_event_name(status: str) -> str:
    if status in {"blocked", "failed"}:
        return f"lambda_provider_adapter.{status}"
    return "lambda_provider_adapter.completed"


def _persist_result(output_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    write_json(output_path, result)
    blockers = [b for b in result.get("blockers", []) if isinstance(b, str)]
    status = _string(result.get("status"))
    log_event(
        logger,
        logging.WARNING if status in {"blocked", "failed"} else logging.INFO,
        _adapter_event_name(status),
        output_path=str(output_path),
        provider_launch_request_path=result.get("provider_launch_request_path"),
        job_id=result.get("job_id"),
        provider=result.get("provider"),
        mode=result.get("mode"),
        status=status,
        reason=result.get("reason"),
        blocker_count=len(blockers),
        blockers=blockers,
        api_call_performed=result.get("api_call_performed"),
    )
    return result


def run_lambda_provider_adapter(
    *,
    provider_launch_request_path: str | Path,
    output_path: str | Path | None = None,
    mode: str = "dry-run",
    allow_lambda_api_call: bool = False,
) -> Dict[str, Any]:
    request_path = Path(provider_launch_request_path).resolve()
    resolved_output = (
        Path(output_path).resolve()
        if output_path
        else Path(
            os.getenv(PROVIDER_ADAPTER_OUTPUT_ENV)
            or request_path.parent / "lambda_provider_adapter_result.json"
        ).resolve()
    )
    ensure_dir(resolved_output.parent)
    payload = read_json_any(request_path)
    request = dict(payload) if isinstance(payload, Mapping) else {}
    _, api_key_meta = _read_lambda_api_key()
    artifact_output_uri = _string(_inputs(request).get("artifact_output_uri"))
    result: Dict[str, Any] = {
        "schema_version": LAMBDA_PROVIDER_ADAPTER_RESULT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "provider_launch_request_path": str(request_path),
        "output_path": str(resolved_output),
        "mode": mode,
        "job_id": _string(request.get("job_id")),
        "provider": _string(request.get("provider")) or LAMBDA_PROVIDER_NAME,
        "adapter_implementation_status": "stub_request_validation_only",
        "live_launch_supported": False,
        "api_call_performed": False,
        "lambda_side_effects_may_have_occurred": False,
        "provider_allocation_proven": False,
        "provider_job_submitted": False,
        "simulator_execution_proven": False,
        "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
        "api_key_readiness": api_key_meta,
        "artifact_output_uri_scheme": urlparse(artifact_output_uri).scheme or None,
        "request_summary": _request_summary(request),
        "proof_boundary": (
            "This stub validates a Lambda Cloud provider launch request and resolves the "
            "Lambda API key. It does not allocate GPUs, submit a job, or prove simulator "
            "execution. Live allocation and teardown are not yet implemented."
        ),
    }

    log_event(
        logger,
        logging.INFO,
        "lambda_provider_adapter.started",
        provider_launch_request_path=str(request_path),
        output_path=str(resolved_output),
        job_id=request.get("job_id"),
        provider=request.get("provider"),
        mode=mode,
        allow_lambda_api_call=allow_lambda_api_call,
    )

    if not request:
        result.update(
            {
                "status": "blocked",
                "reason": "invalid_provider_launch_request_json",
                "blockers": ["invalid_provider_launch_request_json"],
            }
        )
        return _persist_result(resolved_output, result)

    blockers = _request_blockers(request)
    if blockers:
        result.update(
            {
                "status": "blocked",
                "reason": "provider_launch_request_not_consumable",
                "blockers": blockers,
            }
        )
        return _persist_result(resolved_output, result)

    if mode in LIVE_MODES or allow_lambda_api_call:
        result.update(
            {
                "status": "blocked",
                "reason": LIVE_LAUNCH_NOT_IMPLEMENTED_BLOCKER,
                "blockers": [LIVE_LAUNCH_NOT_IMPLEMENTED_BLOCKER],
            }
        )
        return _persist_result(resolved_output, result)

    result.update(
        {
            "status": "dry_run_ready",
            "reason": "lambda_cloud_request_validated_dry_run",
            "blockers": [],
        }
    )
    return _persist_result(resolved_output, result)


def _request_path_from_args(args: argparse.Namespace) -> Path:
    if args.provider_launch_request:
        return Path(args.provider_launch_request)
    env_path = _string(os.getenv(PROVIDER_LAUNCH_REQUEST_ENV))
    if env_path:
        return Path(env_path)
    raise ValueError(
        f"Provide --provider-launch-request or {PROVIDER_LAUNCH_REQUEST_ENV}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a gated Lambda Cloud robot-eval provider launch request (stub: "
            "no live allocation)."
        )
    )
    parser.add_argument("--provider-launch-request")
    parser.add_argument("--output-path")
    parser.add_argument(
        "--mode",
        choices=["dry-run", "auto", "allocate"],
        default="dry-run",
    )
    parser.add_argument(
        "--allow-lambda-api-call",
        action="store_true",
        help=(
            f"Reserved for {LAMBDA_API_GATE_ENV}=true once live launch is implemented; "
            "currently reports the live path as not implemented."
        ),
    )
    args = parser.parse_args(argv)
    try:
        request_path = _request_path_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    result = run_lambda_provider_adapter(
        provider_launch_request_path=request_path,
        output_path=args.output_path,
        mode=args.mode,
        allow_lambda_api_call=args.allow_lambda_api_call,
    )
    print(f"[lambda-provider-adapter] result={result['output_path']}")
    print(f"[lambda-provider-adapter] status={result['status']}")
    print(f"[lambda-provider-adapter] mode={result.get('mode')}")
    blockers = result.get("blockers")
    if blockers:
        print("[lambda-provider-adapter] blockers=" + ",".join(blockers))
    return 0 if result["status"] in {"dry_run_ready", "submitted"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
