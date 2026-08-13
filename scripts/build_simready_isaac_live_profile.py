#!/usr/bin/env python3
"""Build a live launch profile for the ADP-009B exact SimReady import probe.

ADP-009 names "one exact SimReady USD" in as many words, the bundle has read
`status: ready` for some time, and the allocator has had a branch for this probe
throughout. What was missing was a launch profile -- the one thing that carries
a lane across the website boundary -- so nothing could reach it.

This is the first lane written against the shared skeleton in
`task_evaluation_live_profile` rather than as another near-copy of an existing
builder. Everything below is only what makes this lane different: its probe
kind, its TTL band, the arguments the allocator branch expects, and the receipts
it pins. Residency, spend binding, terminal contract, and validation are shared,
which is the point -- those are exactly the checks a hurried copy drops.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.public_scene_simready_isaac_vast import PROBE_KIND
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

# The allocator refuses a TTL outside this band for this probe
# (`simready_isaac_hard_ttl_invalid`).
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 14_400


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    receipt = context.receipt
    # Mirror the allocator's own binding checks so a profile that could never be
    # admitted is refused here, before an attempt authority is consumed.
    if receipt.get("source_commit_sha") != context.source_commit:
        blockers.append("bundle_commit_not_source_commit")
    if not receipt.get("probe_spec_sha256"):
        blockers.append("bundle_probe_spec_digest_missing")
    if receipt.get("retry_cap") != 0:
        blockers.append(f"bundle_retry_cap_not_zero:{receipt.get('retry_cap')}")
    if not 0 < context.max_hourly_rate_usd <= context.max_spend_usd:
        blockers.append("budget_invalid")

    # The ceiling this profile publishes has to be the one the attempt authority
    # was issued against, or the allocator refuses at the paid boundary having
    # already been handed a provider.
    authority_path = context.extra_paths.get("attempt_authority")
    if authority_path is None or not authority_path.is_file():
        blockers.append("attempt_authority_missing")
    else:
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        if authority.get("hard_attempt_spend_cap_usd") != context.max_spend_usd:
            blockers.append("attempt_authority_spend_cap_mismatch")
        if authority.get("maximum_hourly_rate_usd") != context.max_hourly_rate_usd:
            blockers.append("attempt_authority_hourly_rate_mismatch")
        if authority.get("maximum_single_resource_ttl_seconds") != context.hard_ttl_seconds:
            blockers.append("attempt_authority_ttl_mismatch")
        if authority.get("bundle_sha256") != receipt.get("bundle_sha256"):
            blockers.append("attempt_authority_bundle_mismatch")
        if authority.get("probe_spec_sha256") != receipt.get("probe_spec_sha256"):
            blockers.append("attempt_authority_probe_spec_mismatch")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    return [
        "--adp-simready-isaac-bundle-receipt", str(context.receipt_path),
        "--adp-simready-isaac-attempt-authority",
        str(context.extra_paths["attempt_authority"]),
        "--adp-job-dir", context.job_dir("simready-isaac-job"),
        "--adp-max-hourly-rate-usd", str(context.max_hourly_rate_usd),
        "--adp-max-spend-usd", str(context.max_spend_usd),
        "--adp-hard-ttl-seconds", str(context.hard_ttl_seconds),
    ]


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    return [
        {
            "name": "source_bundle_manifest",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "evaluation_run_spec",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "simready_isaac_bundle",
            # The path the receipt resolved to here, not the one it was built at.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
    ]


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp009b-simready-isaac-live",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: f"simready-isaac-{context.source_commit[:12]}",
    # The probe's stage is built over `native/scene/assets/840313_collision.usd`
    # -- scene 840313, the InteriorGS/SAGE pair. "SimReady" describes what the
    # asset became, not where the scene came from, and this field records the
    # latter.
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    extra_path_names=("attempt_authority",),
)


def build_simready_isaac_live_profile(
    *,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str | None = None,
    max_hourly_rate_usd: float = 1.0,
    max_spend_usd: float = 3.0,
    hard_ttl_seconds: int = 7_200,
) -> dict[str, Any]:
    """Derive a live profile from the bundle receipt it will run."""

    return build_lane_live_profile(
        SPEC,
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision,
        max_spend_usd=max_spend_usd,
        extra_paths={"attempt_authority": attempt_authority_path},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--raw-manifest-uri", required=True)
    parser.add_argument(
        "--revision",
        help="Distinguish a rebuilt profile whose inputs changed at the same commit.",
    )
    parser.add_argument("--max-hourly-rate-usd", type=float, default=1.0)
    parser.add_argument("--max-spend-usd", type=float, default=3.0)
    parser.add_argument("--hard-ttl-seconds", type=int, default=7_200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        profile = build_simready_isaac_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            attempt_authority_path=args.attempt_authority,
            source_commit=args.source_commit,
            raw_manifest_uri=args.raw_manifest_uri,
            revision=args.revision,
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
