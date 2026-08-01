"""Canonical allocator sub-lane for the SAM 3.1 Vast source-track canary."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any, Callable

from .common import ensure_dir, write_json
from .gpu_render_providers import get_render_provider
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .sam31_gpu_admission import prepare_sam31_gpu_canary
from .sam31_vast_source_track_canary import run_sam31_vast_source_track_canary
from .wam_async_runner_common import read_sensitive_url_file


def _load_object(path: str | Path) -> dict[str, Any]:
    import json

    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _read_private_secret(path_value: str | Path | None) -> tuple[str, list[str]]:
    path = Path(str(path_value or "")).expanduser()
    blockers: list[str] = []
    if not str(path_value or "").strip() or path.is_symlink() or not path.is_file():
        return "", ["sam31_hf_token_file_missing_or_unsafe"]
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return "", ["sam31_hf_token_file_unreadable"]
    if mode != 0o600:
        blockers.append("sam31_hf_token_file_permissions_not_0600")
    if not value or len(value) > 4096 or "\n" in value or "\r" in value:
        blockers.append("sam31_hf_token_invalid")
    return value, blockers


def run_sam31_paid_resource_allocator_lane(
    args: Any,
    *,
    checkout_commit: str,
    prepare: Callable[..., dict[str, Any]] = prepare_sam31_gpu_canary,
    provider_factory: Callable[[str], Any] = get_render_provider,
    execute_canary: Callable[..., dict[str, Any]] = run_sam31_vast_source_track_canary,
) -> dict[str, Any]:
    """Admit and optionally execute one Vast-first semantic track canary."""

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
        max_spend_usd=args.sam31_max_spend_usd,
        hard_ttl_seconds=args.sam31_hard_ttl_seconds,
        retry_cap=args.sam31_retry_cap,
        authority_id=args.sam31_authority_id,
        execute=args.execute,
        execution_adapter_qualified=args.execute,
    )
    if not args.execute or admission.get("status") != "execute_ready":
        return admission

    blockers: list[str] = []
    urls: dict[str, str] = {}
    for label, path_value in (
        ("input_bundle_get_url", args.provider_bundle_url_file),
        ("output_put_url", args.provider_output_put_url_file),
        ("output_get_url", args.provider_output_get_url_file),
    ):
        value, metadata = read_sensitive_url_file(str(path_value or ""), label=label)
        if not value:
            blockers.append(f"sam31_{label}_missing")
        elif not value.startswith("https://"):
            blockers.append(f"sam31_{label}_not_https")
        elif metadata.get("mode_is_0600") is not True:
            blockers.append(f"sam31_{label}_file_permissions_not_0600")
        urls[label] = value
    hf_token, token_blockers = _read_private_secret(args.sam31_hf_token_file)
    blockers.extend(token_blockers)

    paid_admission = build_paid_lane_admission(
        resource_class="gpu_render",
        blockers=[*list(admission.get("blockers") or []), *blockers],
    )
    adapter_path = Path(args.adapter_output).expanduser().resolve()
    ensure_dir(adapter_path.parent)
    write_json(adapter_path.parent / "sam31_paid_lane_admission.json", paid_admission)
    try:
        grant = require_paid_resource_admission(
            paid_admission,
            resource_class="gpu_render",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
    except PaidResourceAdmissionBlocked as exc:
        result = {
            "schema_version": "semantic_sam31_gpu_canary_adapter_result.v1",
            "status": "blocked",
            "blockers": sorted(set(exc.blockers + blockers)),
            "provider_mutations_performed": 0,
            "cost_usd": 0.0,
            "raw_secret_values_recorded": False,
            "scientific_qualification_inferred": False,
            "proof_effect": "none",
            "claim_ceiling": "no_execution_evidence",
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        }
        write_json(adapter_path, result)
        return result

    result = execute_canary(
        bound_request=_load_object(args.bound_request_out),
        preflight=_load_object(args.preflight_bundle),
        job_dir=adapter_path.parent / "sam31_vast_source_track_canary",
        input_bundle_get_url=urls["input_bundle_get_url"],
        output_put_url=urls["output_put_url"],
        output_get_url=urls["output_get_url"],
        hf_token=hf_token,
        provider=provider_factory(args.provider),
        paid_resource_admission_grant=grant,
    )
    write_json(adapter_path, result)
    return result


__all__ = ["run_sam31_paid_resource_allocator_lane"]
