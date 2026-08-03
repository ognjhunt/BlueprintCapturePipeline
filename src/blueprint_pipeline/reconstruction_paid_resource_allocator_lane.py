"""Canonical allocator sub-lane for the Vast reconstruction worker smoke."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .common import ensure_dir, write_json
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def run_reconstruction_paid_resource_allocator_lane(
    args: Any,
    *,
    checkout_commit: str,
    prepare: Callable[..., dict[str, Any]],
    read_sensitive_url: Callable[..., tuple[str, dict[str, Any]]],
    provider_factory: Callable[[str], Any],
    execute_smoke: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    admission = prepare(
        request_path=args.provider_launch_request,
        preflight_path=args.preflight_bundle,
        admission_out=args.admission_out,
        bound_request_out=args.bound_request_out,
        adapter_output=args.adapter_output,
        provider=args.provider,
        expected_source_commit=args.expected_source_commit or "",
        checkout_source_commit=checkout_commit,
        checkout_clean=True,
        max_spend_usd=args.reconstruction_max_spend_usd,
        hard_ttl_seconds=args.reconstruction_hard_ttl_seconds,
        retry_cap=args.reconstruction_retry_cap,
        authority_id=args.reconstruction_authority_id,
        execute=args.execute,
        execution_adapter_id=(
            "reconstruction_vast_operation_v1" if args.execute else None
        ),
    )
    if not args.execute or admission.get("status") != "execute_ready":
        return admission
    blockers: list[str] = []
    urls: dict[str, str] = {}
    for label, path_value in (
        ("provider_output_put_url", args.provider_output_put_url_file),
        ("provider_output_get_url", args.provider_output_get_url_file),
    ):
        value, metadata = read_sensitive_url(str(path_value or ""), label=label)
        if not value:
            blockers.append(f"reconstruction_{label}_missing")
        elif not value.startswith("https://"):
            blockers.append(f"reconstruction_{label}_not_https")
        elif metadata.get("mode_is_0600") is not True:
            blockers.append(f"reconstruction_{label}_file_permissions_not_0600")
        urls[label] = value
    paid_admission = build_paid_lane_admission(
        resource_class="gpu_render",
        blockers=[*list(admission.get("blockers") or []), *blockers],
    )
    adapter_path = Path(args.adapter_output).expanduser().resolve()
    ensure_dir(adapter_path.parent)
    write_json(adapter_path.parent / "reconstruction_paid_lane_admission.json", paid_admission)
    try:
        grant = require_paid_resource_admission(
            paid_admission,
            resource_class="gpu_render",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked as exc:
        result = {
            "schema_version": "reconstruction_gpu_canary_adapter_result.v1",
            "status": "blocked",
            "blockers": sorted(set(exc.blockers + blockers)),
            "provider_mutations_performed": 0,
            "cost_usd": 0.0,
            "scientific_qualification_inferred": False,
            "proof_effect": "none",
            "claim_ceiling": "no_execution_evidence",
        }
        write_json(adapter_path, result)
        return result
    result = execute_smoke(
        bound_request=_load(args.bound_request_out),
        preflight=_load(args.preflight_bundle),
        job_dir=adapter_path.parent / "reconstruction_vast_worker_smoke",
        output_put_url=urls["provider_output_put_url"],
        output_get_url=urls["provider_output_get_url"],
        provider=provider_factory(args.provider),
        paid_resource_admission_grant=grant,
    )
    write_json(adapter_path, result)
    return result


__all__ = ["run_reconstruction_paid_resource_allocator_lane"]
