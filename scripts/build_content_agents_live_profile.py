#!/usr/bin/env python3
"""Build a live launch profile for one Content Agents CAD candidate.

The four content-agents bundles were built, rehearsed, and left at
`blocked_before_paid_execution` because no profile existed to run them: the
lane had a bundle builder, a provider runner, and an allocator branch, and no
way for the website to reach any of it.

Everything a profile needs is already stated by the receipts the run will use,
so this derives it rather than asking an operator to keep four documents
consistent by hand -- the same reasoning as the retained-scene builder, and the
same failure mode avoided: a profile that names an authoring path, or a spend
ceiling that disagrees with the authority it runs under.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.adp_content_agents_vast import PROBE_KIND
from blueprint_pipeline.host_resident_launch_inputs import (
    HostResidentInputError,
    launch_profile_residency_blockers,
    resolve_host_resident_bundle_receipt,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    TaskEvaluationLaunchError,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)

SECRET_PROFILE_ID = "canonical-vast-adp"
RUN_ROOT = "{launch_run_root}"
# The allocator refuses a TTL outside this band for this probe.
MIN_TTL_SECONDS = 2_700
MAX_TTL_SECONDS = 14_400


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_input_not_object:{path.name}")
    return dict(value)


def build_content_agents_live_profile(
    *,
    bundle_receipt_path: str | Path,
    config_preflight_path: str | Path,
    attempt_authority_path: str | Path,
    source_commit: str,
    candidate_id: str,
    raw_manifest_uri: str,
    max_hourly_rate_usd: float = 1.0,
    max_spend_usd: float = 3.0,
    hard_ttl_seconds: int = 7_200,
) -> dict[str, Any]:
    """Derive a live profile from the receipts it will run."""

    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    preflight_path = Path(config_preflight_path).expanduser().resolve()
    authority_path = Path(attempt_authority_path).expanduser().resolve()

    blockers: list[str] = []
    try:
        resolution = resolve_host_resident_bundle_receipt(receipt_path)
    except HostResidentInputError as exc:
        raise TaskEvaluationLaunchError(str(exc)) from exc
    blockers.extend(resolution["blockers"])
    receipt = resolution["receipt"]

    if receipt.get("status") != "ready":
        blockers.append(f"bundle_receipt_not_ready:{receipt.get('status')}")
    if not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS:
        blockers.append(f"hard_ttl_out_of_band:{hard_ttl_seconds}")
    if not 0 < max_hourly_rate_usd <= max_spend_usd:
        blockers.append("budget_invalid")
    # The ceiling this profile publishes has to be the one the attempt authority
    # was issued against, or the allocator refuses at the paid boundary having
    # already been handed a provider.
    if not authority_path.is_file():
        blockers.append("attempt_authority_missing")
    else:
        authority = _read(authority_path)
        if authority.get("hard_attempt_spend_cap_usd") != max_spend_usd:
            blockers.append("attempt_authority_spend_cap_mismatch")
        if authority.get("maximum_hourly_rate_usd") != max_hourly_rate_usd:
            blockers.append("attempt_authority_hourly_rate_mismatch")
        if authority.get("maximum_single_resource_ttl_seconds") != hard_ttl_seconds:
            blockers.append("attempt_authority_ttl_mismatch")
        if authority.get("bundle_sha256") != receipt.get("bundle_sha256"):
            blockers.append("attempt_authority_bundle_mismatch")
    if not preflight_path.is_file():
        blockers.append("config_preflight_missing")
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    profile_id = f"adp-content-agents-live-{candidate_id}-{source_commit}"
    bundle_digest = str(receipt.get("bundle_sha256") or "")
    profile: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "claim_ceiling": "development_only",
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--admission-out", f"{RUN_ROOT}/allocator/admission.json",
                "--bound-request-out", f"{RUN_ROOT}/allocator/bound-request.json",
                "--adapter-output", f"{RUN_ROOT}/allocator/result.json",
                "--pod-name", profile_id,
                "--expected-source-commit", source_commit,
                "--provider", "vast",
                "--probe-kind", PROBE_KIND,
                "--adp-content-agents-bundle-receipt", str(receipt_path),
                "--adp-content-agents-config-preflight-receipt", str(preflight_path),
                "--adp-content-agents-attempt-authority", str(authority_path),
                "--adp-job-dir", f"{RUN_ROOT}/allocator/content-agents-job",
                "--adp-max-hourly-rate-usd", str(max_hourly_rate_usd),
                "--adp-max-spend-usd", str(max_spend_usd),
                "--adp-hard-ttl-seconds", str(hard_ttl_seconds),
            ],
            "max_spend_usd": float(max_spend_usd),
            "hard_ttl_seconds": hard_ttl_seconds,
            "retry_cap": 0,
        },
        "execution_admission": {
            "live_enabled": True,
            "blockers": [],
            "readiness_receipt": {"uri": raw_manifest_uri, "digest": bundle_digest},
        },
        "evaluation_run_spec": {"uri": raw_manifest_uri, "digest": bundle_digest},
        "source_bundle": {
            "bundle_id": f"content-agents-{candidate_id}",
            "source_kind": "interiorgs_sage",
            "uri": raw_manifest_uri,
            "digest": bundle_digest,
        },
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(receipt_path),
                "digest": _digest(receipt_path),
            },
            {
                "name": "evaluation_run_spec",
                "path": str(preflight_path),
                "digest": _digest(preflight_path),
            },
            {
                "name": "content_agents_bundle",
                # The path the receipt resolved to here, not the one it was
                # built at.
                "path": str(resolution["resolutions"]["bundle"]["path"]),
                "digest": bundle_digest,
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
            "required_values": {"continuing_spend_from_this_run": False, "retry_cap": 0},
            "required_path_fields": ["teardown_manifest_path", "artifact_manifest_path"],
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
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--config-preflight", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Which CAD candidate this profile runs, e.g. a-earthtojake-text-to-cad.",
    )
    parser.add_argument("--raw-manifest-uri", required=True)
    parser.add_argument("--max-hourly-rate-usd", type=float, default=1.0)
    parser.add_argument("--max-spend-usd", type=float, default=3.0)
    parser.add_argument("--hard-ttl-seconds", type=int, default=7_200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        profile = build_content_agents_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            config_preflight_path=args.config_preflight,
            attempt_authority_path=args.attempt_authority,
            source_commit=args.source_commit,
            candidate_id=args.candidate_id,
            raw_manifest_uri=args.raw_manifest_uri,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            max_spend_usd=args.max_spend_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
        )
    except (OSError, json.JSONDecodeError, TaskEvaluationLaunchError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "task_evaluation_launch_profile.v1",
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
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
                "max_spend_usd": profile["allocator"]["max_spend_usd"],
                "output": str(output),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
