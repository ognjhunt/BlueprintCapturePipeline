#!/usr/bin/env python3
"""Build a live launch profile for the USD Joint Agent enrichment probe.

The bundle has read `status: ready` for some time and the allocator has always
had a branch for the probe. What was missing was a launch profile, which is the
one thing that carries a lane across the website boundary.

This lane uses the shared standing website authorization as its one-use paid
attempt authority.  The profile requires a one-launch authorization, and the
dispatcher atomically consumes it before the allocator may stage bytes or
allocate a provider. The TTL band is 5400 to 14400 seconds.

Its claim ceiling is narrow and the bundle says so itself: this is optional
construction enrichment, it is not SimReady authority, and its failure blocks
neither deterministic asset construction nor native simulator qualification.
Those are carried into the profile rather than restated, so a passing run cannot
later be read as more than it is.

The skeleton is shared with every other paid lane in
`task_evaluation_live_profile`.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.adp_joint_agent_vast import (
    DEFAULT_IMAGE as JOINT_AGENT_IMAGE,
    SOURCE_TREE as JOINT_AGENT_SOURCE_TREE,
)

from blueprint_pipeline.adp_joint_agent_vast import PROBE_KIND
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

# The allocator refuses a TTL outside this band for this probe.
MIN_TTL_SECONDS = 5_400
MAX_TTL_SECONDS = 14_400


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    receipt = context.receipt
    if not 0 < context.max_hourly_rate_usd <= context.max_spend_usd:
        blockers.append("budget_invalid")
    if receipt.get("provider_bundle_kind") != "adp_joint_agent":
        blockers.append(f"bundle_kind_mismatch:{receipt.get('provider_bundle_kind')}")
    if not receipt.get("freeze_digest"):
        blockers.append("bundle_freeze_digest_missing")
    if receipt.get("completion_retries") != 0:
        blockers.append(f"bundle_completion_retries_not_zero:{receipt.get('completion_retries')}")
    if receipt.get("automatic_paid_retry_allowed") is not False:
        blockers.append("bundle_permits_automatic_paid_retry")
    # The claim boundary is the bundle's, not this builder's. A bundle that
    # stopped saying so would be publishing a wider claim than the lane earns.
    if receipt.get("agent_output_is_simready_authority") is not False:
        blockers.append("bundle_claims_simready_authority")
    if receipt.get("provider_zero_required_after_return") is not True:
        blockers.append("bundle_does_not_require_provider_zero")
    # Mirror the receipt-readable half of the allocator's own binding check,
    # the way the SimReady builder already does. The dispatcher consumes this
    # lane's one-launch standing authorization *before* it invokes the
    # allocator, and `record_launch` is exclusive-create with no release path,
    # so any allocator-side refusal burns the authorization for zero provider
    # work and leaves the lane unlaunchable until someone hand-writes a new
    # authorization file on the control-plane host. A bundle built at another
    # commit is the recurring case: every merge moves main past the deployed
    # release.
    source = receipt.get("blueprint_source")
    source = source if isinstance(source, dict) else {}
    if receipt.get("status") != "ready":
        blockers.append(f"bundle_status_not_ready:{receipt.get('status')}")
    if source.get("commit") != context.source_commit:
        blockers.append("bundle_commit_not_source_commit")
    if source.get("dirty") is not False:
        blockers.append("bundle_source_tree_dirty")
    if receipt.get("container_image") != JOINT_AGENT_IMAGE:
        blockers.append("bundle_container_image_mismatch")
    if receipt.get("source_tree") != JOINT_AGENT_SOURCE_TREE:
        blockers.append("bundle_source_tree_mismatch")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    return [
        "--adp-joint-agent-bundle-receipt", str(context.receipt_path),
        "--adp-job-dir", context.job_dir("joint-agent-job"),
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
            "name": "joint_agent_bundle",
            # The path the receipt resolved to here, not the one it was built at.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
    ]


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp-joint-agent-live",
    profile_builder="build_joint_agent_live_profile.py",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: f"joint-agent-{context.source_commit[:12]}",
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    extra_path_names=(),
    one_use_standing_authority_required=True,
)


def build_joint_agent_live_profile(
    *,
    bundle_receipt_path: str | Path,
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
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--raw-manifest-uri",
        required=True,
        help="Local digest-bound content-addressed publication receipt for this run spec.",
    )
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
        profile = build_joint_agent_live_profile(
            bundle_receipt_path=args.bundle_receipt,
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
