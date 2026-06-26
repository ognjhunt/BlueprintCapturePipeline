"""Provider-neutral endpoint/discovery manifest for GPU policy workers."""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .provider_worker_contract import (
    HEALTHZ_PATH,
    INFER_PATH,
    PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
    READYZ_PATH,
    SHUTDOWN_PATH,
)


PROVIDER_WORKER_ENDPOINT_MANIFEST_SCHEMA_VERSION = "provider_worker_endpoint_manifest.v1"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _redact_url(url: str | None) -> str | None:
    text = _string(url)
    if not text:
        return None
    try:
        parsed = urllib.parse.urlsplit(text)
    except ValueError:
        return "<invalid_url>"
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path or "/", "", ""))


def _provider_mode_kind(provider: str, mode: str) -> dict[str, Any]:
    provider_text = provider.strip().lower()
    mode_text = mode.strip().lower()
    if provider_text == "runpod" and mode_text == "serverless-run":
        return {
            "worker_invocation_grain": "evaluation_job_provider_submission",
            "direct_http_worker_endpoint_expected": False,
            "direct_policy_infer_from_local_loop_allowed": False,
            "reason": "runpod_serverless_run_is_job_submission_not_direct_infer_endpoint",
        }
    if provider_text == "runpod" and mode_text in {"on-demand-pod", "auto", "dry-run"}:
        return {
            "worker_invocation_grain": "direct_http_worker_after_provider_endpoint_discovery",
            "direct_http_worker_endpoint_expected": mode_text != "serverless-run",
            "direct_policy_infer_from_local_loop_allowed": mode_text == "on-demand-pod",
            "reason": "on_demand_worker_must_publish_or_report_provider_worker_urls",
        }
    if provider_text == "vast":
        return {
            "worker_invocation_grain": "direct_http_worker_after_provider_endpoint_discovery",
            "direct_http_worker_endpoint_expected": True,
            "direct_policy_infer_from_local_loop_allowed": mode_text == "live-startup-probe",
            "reason": "vast_instance_must_report_reachable_worker_urls_before_local_loop_reuse",
        }
    return {
        "worker_invocation_grain": "provider_neutral_worker_session",
        "direct_http_worker_endpoint_expected": True,
        "direct_policy_infer_from_local_loop_allowed": False,
        "reason": "provider_mode_requires_endpoint_discovery_contract",
    }


def build_provider_worker_endpoint_manifest(
    *,
    provider: str,
    mode: str,
    job_id: str | None = None,
    generated_at: str | None = None,
    worker_url: str | None = None,
    ready_url: str | None = None,
    shutdown_url: str | None = None,
    serverless_endpoint_id: str | None = None,
    provider_instance_id: str | int | None = None,
    provider_request_shape: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    generated = generated_at or utc_now_iso()
    mode_policy = _provider_mode_kind(provider, mode)
    worker_present = bool(_string(worker_url))
    ready_present = bool(_string(ready_url))
    status = (
        "endpoint_ready_for_policy_commands"
        if worker_present and ready_present
        else "endpoint_discovery_pending_provider_runtime"
    )
    direct_loop_allowed = (
        mode_policy.get("direct_policy_infer_from_local_loop_allowed") is True
        and worker_present
        and ready_present
    )
    provider_shape = _mapping(provider_request_shape)
    return {
        "schema_version": PROVIDER_WORKER_ENDPOINT_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated,
        "status": status,
        "provider_worker_contract_schema_version": PROVIDER_WORKER_CONTRACT_SCHEMA_VERSION,
        "provider": provider,
        "provider_mode": mode,
        "job_id": job_id,
        **mode_policy,
        "direct_policy_infer_from_local_loop_allowed": direct_loop_allowed,
        "known_endpoint": {
            "worker_url_present": worker_present,
            "worker_url_redacted": _redact_url(worker_url),
            "ready_url_present": ready_present,
            "ready_url_redacted": _redact_url(ready_url),
            "shutdown_url_present": bool(_string(shutdown_url)),
            "shutdown_url_redacted": _redact_url(shutdown_url),
            "serverless_endpoint_id_present": bool(_string(serverless_endpoint_id)),
            "serverless_endpoint_id_redacted": "<configured>"
            if _string(serverless_endpoint_id)
            else None,
            "provider_instance_id_present": provider_instance_id is not None,
            "provider_instance_id_redacted": "<configured>"
            if provider_instance_id is not None
            else None,
        },
        "http_contract": {
            "healthz": {"method": "GET", "path": HEALTHZ_PATH},
            "readyz": {"method": "GET", "path": READYZ_PATH},
            "infer": {"method": "POST", "path": INFER_PATH},
            "shutdown": {"method": "POST", "path": SHUTDOWN_PATH},
        },
        "consumer_env_contract": {
            "worker_url_env": "BLUEPRINT_PROVIDER_POLICY_WORKER_URL",
            "ready_url_env": "BLUEPRINT_PROVIDER_POLICY_WORKER_READY_URL",
            "auth_token_file_env": "BLUEPRINT_PROVIDER_POLICY_WORKER_AUTH_TOKEN_FILE",
            "policy_command_adapter": "blueprint-provider-worker-policy-command-adapter",
            "session_runner": "blueprint-run-provider-worker-session",
        },
        "provider_runtime_must_write": [
            "provider_worker_endpoint_manifest.json",
            "worker_readyz_probe.json",
            "provider_worker_session_run.json",
            "provider_teardown_manifest.json",
        ],
        "provider_request_summary": {
            "operation": provider_shape.get("operation"),
            "command_configured": bool(_string(provider_shape.get("command"))),
            "image_configured": bool(
                _string(_mapping(provider_shape.get("image")).get("configured_image_ref"))
            ),
            "manifest_uri_present": bool(
                _string(_mapping(provider_shape.get("inputs")).get("manifest_uri"))
            ),
        },
        "blockers": []
        if worker_present and ready_present
        else ["provider_worker_endpoint_not_discovered_yet"],
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
        "claim_boundary": {
            "endpoint_manifest_is_not_provider_allocation_proof": True,
            "endpoint_manifest_is_not_worker_ready_proof": True,
            "readyz_probe_required_before_customer_eval": True,
            "shutdown_response_is_not_provider_teardown_or_cost_proof": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
            "non_ranking_operational_claim_proven": False,
        },
    }


def write_provider_worker_endpoint_manifest(
    *,
    output_dir: str | Path,
    provider: str,
    mode: str,
    job_id: str | None = None,
    generated_at: str | None = None,
    worker_url: str | None = None,
    ready_url: str | None = None,
    shutdown_url: str | None = None,
    serverless_endpoint_id: str | None = None,
    provider_instance_id: str | int | None = None,
    provider_request_shape: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser()
    ensure_dir(output)
    manifest = build_provider_worker_endpoint_manifest(
        provider=provider,
        mode=mode,
        job_id=job_id,
        generated_at=generated_at,
        worker_url=worker_url,
        ready_url=ready_url,
        shutdown_url=shutdown_url,
        serverless_endpoint_id=serverless_endpoint_id,
        provider_instance_id=provider_instance_id,
        provider_request_shape=provider_request_shape,
    )
    write_json(output / "provider_worker_endpoint_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--worker-url")
    parser.add_argument("--ready-url")
    parser.add_argument("--shutdown-url")
    parser.add_argument("--serverless-endpoint-id")
    args = parser.parse_args(argv)
    manifest = write_provider_worker_endpoint_manifest(
        output_dir=args.output_dir,
        provider=args.provider,
        mode=args.mode,
        job_id=args.job_id,
        worker_url=args.worker_url,
        ready_url=args.ready_url,
        shutdown_url=args.shutdown_url,
        serverless_endpoint_id=args.serverless_endpoint_id,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
