#!/usr/bin/env python3
"""Build a host-resident live profile for one exact SAM 3.1 source-track run.

This is a read-only profile builder. It binds the exact request, deterministic
input bundle and receipt, single-use paid authority, source commit, zero-retry
budget, and the canonical secret-file location. It performs no upload or
provider mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.host_resident_launch_inputs import launch_profile_residency_blockers
from blueprint_pipeline.sam31_gpu_admission import PROBE_KIND, REQUEST_SCHEMA_VERSION
from blueprint_pipeline.sam31_paid_attempt_authority import (
    AUTHORITY_SCHEMA_VERSION,
    validate_sam31_paid_attempt_authority,
)
from blueprint_pipeline.sam31_source_track_canary_worker import BUNDLE_RECEIPT_SCHEMA_VERSION
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    TaskEvaluationLaunchError,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)


RUN_ROOT = "{launch_run_root}"
SECRET_PROFILE_ID = "canonical-vast-adp"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_input_not_object:{path.name}")
    return dict(value)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_sam31_source_tracks_live_profile(
    *,
    request_path: str | Path,
    input_bundle_path: str | Path,
    input_bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    hf_token_file: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str | None = None,
) -> dict[str, Any]:
    """Derive a publishable, zero-retry profile from immutable SAM inputs."""

    request_file = Path(request_path).expanduser().resolve()
    bundle_file = Path(input_bundle_path).expanduser().resolve()
    receipt_file = Path(input_bundle_receipt_path).expanduser().resolve()
    authority_file = Path(attempt_authority_path).expanduser().resolve()
    token_file = Path(hf_token_file).expanduser().resolve()
    blockers: list[str] = []
    for label, path in (
        ("request", request_file),
        ("input_bundle", bundle_file),
        ("input_bundle_receipt", receipt_file),
        ("attempt_authority", authority_file),
        ("hf_token_file", token_file),
    ):
        if path.is_symlink() or not path.is_file():
            blockers.append(f"sam31_live_profile_{label}_missing_or_unsafe")
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))
    request = _read(request_file)
    receipt = _read(receipt_file)
    authority = _read(authority_file)
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("sam31_live_profile_request_schema_invalid")
    if receipt.get("schema_version") != BUNDLE_RECEIPT_SCHEMA_VERSION:
        blockers.append("sam31_live_profile_bundle_receipt_schema_invalid")
    if authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        blockers.append("sam31_live_profile_authority_schema_invalid")
    if request.get("source_commit_sha") != source_commit:
        blockers.append("sam31_live_profile_commit_mismatch")
    try:
        validate_sam31_paid_attempt_authority(
            authority,
            request=request,
            bundle_path=bundle_file,
            bundle_receipt=receipt,
            blueprint_commit=source_commit,
            max_hourly_rate_usd=authority.get("maximum_hourly_rate_usd"),
            hard_cap_usd=authority.get("hard_attempt_spend_cap_usd"),
            hard_ttl_seconds=authority.get("maximum_single_resource_ttl_seconds"),
            allowed_active_instance_ids=(authority.get("active_instance_allowlist") or {}).get(
                "external_provider_owned", []
            ),
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if token_file.stat().st_mode & 0o077:
        blockers.append("sam31_live_profile_hf_token_permissions_not_0600")
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    rate = float(authority["maximum_hourly_rate_usd"])
    cap = float(authority["hard_attempt_spend_cap_usd"])
    ttl = int(authority["maximum_single_resource_ttl_seconds"])
    profile_id = f"adp-sam31-source-tracks-live-{source_commit}"
    if revision:
        profile_id = f"{profile_id}-{revision}"
    request_digest = str(request["request_digest"])
    allowlist = list(
        (authority.get("active_instance_allowlist") or {}).get(
            "external_provider_owned", []
        )
    )
    profile: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "claim_ceiling": "development_only",
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--admission-out",
                f"{RUN_ROOT}/allocator/admission.json",
                "--bound-request-out",
                f"{RUN_ROOT}/allocator/bound-request.json",
                "--adapter-output",
                f"{RUN_ROOT}/allocator/result.json",
                "--pod-name",
                profile_id,
                "--expected-source-commit",
                source_commit,
                "--provider",
                "vast",
                "--probe-kind",
                PROBE_KIND,
                "--provider-launch-request",
                str(request_file),
                "--preflight-bundle",
                f"{RUN_ROOT}/allocator/sam31-execution-preflight.json",
                "--sam31-input-bundle",
                str(bundle_file),
                "--sam31-input-bundle-receipt",
                str(receipt_file),
                "--sam31-attempt-authority",
                str(authority_file),
                "--sam31-hf-token-file",
                str(token_file),
                "--sam31-max-hourly-rate-usd",
                str(rate),
                "--sam31-max-spend-usd",
                str(cap),
                "--sam31-hard-ttl-seconds",
                str(ttl),
                "--sam31-retry-cap",
                "0",
                "--sam31-authority-id",
                str(request["authority_id"]),
                *[
                    argument
                    for instance_id in allowlist
                    for argument in (
                        "--sam31-allowed-active-vast-instance-id",
                        str(instance_id),
                    )
                ],
            ],
            "max_spend_usd": cap,
            "hard_ttl_seconds": ttl,
            "retry_cap": 0,
        },
        "execution_admission": {
            "live_enabled": True,
            "blockers": [],
            "readiness_receipt": {"uri": raw_manifest_uri, "digest": request_digest},
        },
        "evaluation_run_spec": {"uri": raw_manifest_uri, "digest": request_digest},
        "source_bundle": {
            "bundle_id": f"sam31-source-tracks-{source_commit[:12]}",
            "source_kind": "interiorgs_sage",
            "uri": raw_manifest_uri,
            "digest": request_digest,
        },
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(request_file),
                "digest": _digest(request_file),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(receipt_file),
                "digest": _digest(receipt_file),
            },
            {
                "name": "sam31_input_bundle",
                "path": str(bundle_file),
                "digest": _digest(bundle_file),
            },
            {
                "name": "sam31_paid_attempt_authority",
                "path": str(authority_file),
                "digest": _digest(authority_file),
            },
        ],
        "reconciliation": {"max_guard_age_seconds": 300, "required_providers": ["vast"]},
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "secret_profile_id": SECRET_PROFILE_ID,
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "runtime_environment": {},
        "terminal_contract": {
            "result_path": f"{RUN_ROOT}/allocator/result.json",
            "success_statuses": ["completed"],
            "required_values": {
                "continuing_spend_from_this_run": False,
                "retry_cap": 0,
            },
            "required_path_fields": [
                "teardown_manifest_path",
                "artifact_manifest_path",
                "source_track_import_result_path",
            ],
        },
        "webapp_sync": {"max_attempts": 20},
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    validation = [
        *validate_launch_profile(profile),
        *verify_profile_immutable_inputs(profile),
        *launch_profile_residency_blockers(profile),
    ]
    if validation:
        raise TaskEvaluationLaunchError(",".join(sorted(set(validation))))
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--input-bundle", required=True)
    parser.add_argument("--input-bundle-receipt", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--hf-token-file", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--raw-manifest-uri", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        profile = build_sam31_source_tracks_live_profile(
            request_path=args.request,
            input_bundle_path=args.input_bundle,
            input_bundle_receipt_path=args.input_bundle_receipt,
            attempt_authority_path=args.attempt_authority,
            hf_token_file=args.hf_token_file,
            source_commit=args.source_commit,
            raw_manifest_uri=args.raw_manifest_uri,
            revision=args.revision,
        )
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)]}, sort_keys=True))
        return 2
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "built",
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "provider_mutation_performed": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
