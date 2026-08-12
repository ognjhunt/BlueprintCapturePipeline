#!/usr/bin/env python3
"""Build a live launch profile for the retained-scene GPU render probe.

Every profile in the production catalog used probe kind
``task-evaluation-profile-preflight``, which is dry-only by design: it never
calls a provider and rejects ``execute=True`` unconditionally. That is correct
for a dry validator, but it meant the catalog could not express a paid run at
all, so the website path could never reach the allocator's provider boundary.

The first live profile was assembled by hand, and each hand-assembly cost a
round trip through a fail-closed gate:

* the source-bundle manifest was recorded at the authoring machine's path, so
  the control plane refused it with ``immutable_input_missing``;
* the allocator's ``--adp-retained-scene-render-attempt-authority`` argument is
  required only under ``--execute``, so its absence surfaced as a paid-attempt
  blocker rather than a build error;
* the instance allowlist passed on the command line must equal the bundle
  authority's exactly, and the two are edited in different files.

Each of those is derivable from the bundle receipt and its execution authority,
so this derives them instead of asking an operator to keep four documents
consistent by hand.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)

PROBE_KIND = "adp-retained-scene-gpu-render"
CANONICAL_ALLOCATOR = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
SECRET_PROFILE_ID = "canonical-vast-adp"
RUN_ROOT = "{launch_run_root}"
# The allocator refuses a TTL outside this band for this probe.
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 10_800


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_input_not_object:{path.name}")
    return dict(value)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_retained_scene_render_live_profile(
    *,
    bundle_receipt_path: str | Path,
    request_manifest_path: str | Path,
    attempt_authority_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    max_hourly_rate_usd: float = 2.0,
    hard_ttl_seconds: int = MAX_TTL_SECONDS,
    revision: str | None = None,
) -> dict[str, Any]:
    """Derive a live profile from the bundle receipt it will run."""

    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    request_path = Path(request_manifest_path).expanduser().resolve()
    authority_path = Path(attempt_authority_path).expanduser().resolve()
    receipt = _read(receipt_path)

    blockers: list[str] = []
    if receipt.get("status") != "ready":
        blockers.append(f"bundle_receipt_not_ready:{receipt.get('status')}")
    if receipt.get("probe_kind") != PROBE_KIND:
        blockers.append(f"bundle_probe_kind_mismatch:{receipt.get('probe_kind')}")
    # A profile pinned to a commit the control plane is not running is refused
    # by the allocator, so catch it here rather than at the paid boundary.
    if receipt.get("blueprint_commit") != source_commit:
        blockers.append("bundle_commit_not_source_commit")
    if not MIN_TTL_SECONDS <= hard_ttl_seconds <= MAX_TTL_SECONDS:
        blockers.append(f"hard_ttl_out_of_band:{hard_ttl_seconds}")

    # The spend cap belongs to the bundle, not to this builder: the allocator
    # reads it off the receipt and rejects any rate/TTL pair whose worst case
    # exceeds it.
    cap = receipt.get("hard_total_spend_cap_usd")
    if not isinstance(cap, (int, float)) or isinstance(cap, bool):
        blockers.append("bundle_spend_cap_missing")
        cap = 0.0
    worst_case = max_hourly_rate_usd * hard_ttl_seconds / 3600.0
    if worst_case > float(cap):
        blockers.append(f"worst_case_spend_exceeds_bundle_cap:{worst_case}>{cap}")

    # The allowlist must equal the bundle authority's exactly; deriving it
    # removes the chance of the two drifting apart in separate files.
    authority_record = receipt.get("execution_authority")
    allowlist: list[int] = []
    if not isinstance(authority_record, Mapping):
        blockers.append("bundle_execution_authority_missing")
    else:
        authority_source = Path(str(authority_record.get("path") or "")).expanduser()
        if not authority_source.is_file():
            blockers.append("bundle_execution_authority_unreadable")
        else:
            paid = _read(authority_source).get("paid_compute")
            if not isinstance(paid, Mapping):
                blockers.append("bundle_execution_authority_paid_compute_missing")
            else:
                allowlist = sorted(
                    {int(item) for item in paid.get("external_instance_allowlist") or []}
                )

    if not authority_path.is_file():
        blockers.append("attempt_authority_missing")
    if not request_path.is_file():
        blockers.append("request_manifest_missing")
    if blockers:
        raise TaskEvaluationLaunchError(",".join(sorted(set(blockers))))

    profile_id = f"adp-retained-scene-render-live-{source_commit}"
    if revision:
        # Published profiles are immutable, so a changed profile needs its own
        # id rather than a conflicting rewrite of an existing one.
        profile_id = f"{profile_id}-{revision}"

    request_digest = _digest(request_path)
    profile: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "claim_ceiling": "development_only",
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR,
            "subcommand": "gpu-canary",
            "argv": [
                "--admission-out", f"{RUN_ROOT}/allocator/admission.json",
                "--bound-request-out", f"{RUN_ROOT}/allocator/bound-request.json",
                "--adapter-output", f"{RUN_ROOT}/allocator/result.json",
                "--pod-name", profile_id,
                "--expected-source-commit", source_commit,
                "--provider", "vast",
                "--probe-kind", PROBE_KIND,
                "--adp-retained-scene-render-bundle-receipt", str(receipt_path),
                "--adp-retained-scene-render-attempt-authority", str(authority_path),
                "--adp-retained-scene-render-job-dir",
                f"{RUN_ROOT}/allocator/retained-scene-job",
                "--adp-retained-scene-render-max-hourly-rate-usd", str(max_hourly_rate_usd),
                "--adp-retained-scene-render-hard-ttl-seconds", str(hard_ttl_seconds),
                *[
                    argument
                    for instance_id in allowlist
                    for argument in (
                        "--adp-allowed-active-vast-instance-id",
                        str(instance_id),
                    )
                ],
            ],
            "max_spend_usd": float(cap),
            "hard_ttl_seconds": hard_ttl_seconds,
            "retry_cap": 0,
        },
        "execution_admission": {
            "live_enabled": True,
            "blockers": [],
            "readiness_receipt": {"uri": raw_manifest_uri, "digest": request_digest},
        },
        "evaluation_run_spec": {"uri": raw_manifest_uri, "digest": request_digest},
        "source_bundle": {
            "bundle_id": f"retained-scene-render-{source_commit[:12]}",
            "source_kind": "interiorgs_sage",
            "uri": raw_manifest_uri,
            "digest": request_digest,
        },
        # The request manifest is this probe's source-bundle manifest; the
        # bundle receipt is what actually pins the run.
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": str(request_path),
                "digest": request_digest,
            },
            {
                "name": "evaluation_run_spec",
                "path": str(receipt_path),
                "digest": _digest(receipt_path),
            },
            {
                "name": "retained_scene_render_bundle",
                "path": str(receipt.get("bundle_path") or ""),
                "digest": str(receipt.get("bundle_sha256") or ""),
            },
        ],
        "reconciliation": {"max_guard_age_seconds": 300, "required_providers": ["vast"]},
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR,
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

    validation = [*validate_launch_profile(profile), *verify_profile_immutable_inputs(profile)]
    if validation:
        raise TaskEvaluationLaunchError(",".join(sorted(set(validation))))
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--request-manifest", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--raw-manifest-uri", required=True)
    parser.add_argument("--max-hourly-rate-usd", type=float, default=2.0)
    parser.add_argument("--hard-ttl-seconds", type=int, default=MAX_TTL_SECONDS)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        profile = build_retained_scene_render_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            request_manifest_path=args.request_manifest,
            attempt_authority_path=args.attempt_authority,
            source_commit=args.source_commit,
            raw_manifest_uri=args.raw_manifest_uri,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            revision=args.revision,
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
