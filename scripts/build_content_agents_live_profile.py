#!/usr/bin/env python3
"""Build a live launch profile for one Content Agents CAD candidate.

The four content-agents bundles were built, rehearsed, and left at
`blocked_before_paid_execution` because no profile existed to run them: the
lane had a bundle builder, a provider runner, and an allocator branch, and no
way for the website to reach any of it.

Everything a profile needs is already stated by the receipts the run will use,
so this derives it rather than asking an operator to keep four documents
consistent by hand -- and the same failure mode is avoided: a profile that names
an authoring path, or a spend ceiling that disagrees with the authority it runs
under.

The skeleton is shared with every other paid lane in
`task_evaluation_live_profile`. What is here is only what makes this lane
different.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.adp_content_agents_vast import PROBE_KIND
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

# The allocator refuses a TTL outside this band for this probe.
MIN_TTL_SECONDS = 2_700
MAX_TTL_SECONDS = 14_400


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_input_not_object:{path.name}")
    return dict(value)


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    if not 0 < context.max_hourly_rate_usd <= context.max_spend_usd:
        blockers.append("budget_invalid")
    # The ceiling this profile publishes has to be the one the attempt authority
    # was issued against, or the allocator refuses at the paid boundary having
    # already been handed a provider.
    authority_path = context.extra_paths.get("attempt_authority")
    authority: dict[str, Any] | None = None
    if authority_path is None or not authority_path.is_file():
        blockers.append("attempt_authority_missing")
    else:
        authority = _read(authority_path)
        if authority.get("hard_attempt_spend_cap_usd") != context.max_spend_usd:
            blockers.append("attempt_authority_spend_cap_mismatch")
        if authority.get("maximum_hourly_rate_usd") != context.max_hourly_rate_usd:
            blockers.append("attempt_authority_hourly_rate_mismatch")
        if authority.get("maximum_single_resource_ttl_seconds") != context.hard_ttl_seconds:
            blockers.append("attempt_authority_ttl_mismatch")
        if authority.get("bundle_sha256") != context.receipt.get("bundle_sha256"):
            blockers.append("attempt_authority_bundle_mismatch")
    preflight = context.extra_paths.get("config_preflight")
    if preflight is None or not preflight.is_file():
        blockers.append("config_preflight_missing")
    elif authority is not None:
        preflight_value = _read(preflight)
        if authority.get("config_preflight_receipt_sha256") != file_digest(preflight):
            blockers.append("attempt_authority_preflight_sha256_mismatch")
        if authority.get("config_preflight_receipt_digest") != preflight_value.get(
            "receipt_digest"
        ):
            blockers.append("attempt_authority_preflight_digest_mismatch")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    return [
        "--adp-content-agents-bundle-receipt", str(context.receipt_path),
        "--adp-content-agents-config-preflight-receipt",
        str(context.extra_paths["config_preflight"]),
        "--adp-content-agents-attempt-authority",
        str(context.extra_paths["attempt_authority"]),
        "--adp-job-dir", context.job_dir("content-agents-job"),
        "--adp-max-hourly-rate-usd", str(context.max_hourly_rate_usd),
        "--adp-max-spend-usd", str(context.max_spend_usd),
        "--adp-hard-ttl-seconds", str(context.hard_ttl_seconds),
    ]


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    preflight = context.extra_paths["config_preflight"]
    return [
        {
            "name": "source_bundle_manifest",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "evaluation_run_spec",
            "path": str(preflight),
            "digest": file_digest(preflight),
        },
        {
            "name": "content_agents_bundle",
            # The path the receipt resolved to here, not the one it was built at.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
        {
            "name": "paid_attempt_authority",
            "path": str(context.extra_paths["attempt_authority"]),
            "digest": file_digest(context.extra_paths["attempt_authority"]),
        },
    ]


def _spec(candidate_id: str) -> LaneLiveProfileSpec:
    return LaneLiveProfileSpec(
        profile_id_prefix=f"adp-content-agents-live-{candidate_id}",
        profile_builder="build_content_agents_live_profile.py",
        probe_kind=PROBE_KIND,
        min_ttl_seconds=MIN_TTL_SECONDS,
        max_ttl_seconds=MAX_TTL_SECONDS,
        source_bundle_id=lambda context: f"content-agents-{candidate_id}",
        source_kind="interiorgs_sage",
        lane_argv=_lane_argv,
        immutable_inputs=_immutable_inputs,
        lane_blockers=_lane_blockers,
        extra_path_names=("config_preflight", "attempt_authority"),
    )


def build_content_agents_live_profile(
    *,
    bundle_receipt_path: str | Path,
    config_preflight_path: str | Path,
    attempt_authority_path: str | Path,
    source_commit: str,
    candidate_id: str,
    raw_manifest_uri: str,
    revision: str | None = None,
    max_hourly_rate_usd: float = 1.0,
    max_spend_usd: float = 3.0,
    hard_ttl_seconds: int = 7_200,
) -> dict[str, Any]:
    """Derive a live profile from the receipts it will run."""

    return build_lane_live_profile(
        _spec(candidate_id),
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision,
        max_spend_usd=max_spend_usd,
        extra_paths={
            "config_preflight": config_preflight_path,
            "attempt_authority": attempt_authority_path,
        },
    )


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
    parser.add_argument(
        "--raw-manifest-uri",
        required=True,
        help="Local digest-bound GCS publication receipt for this run spec.",
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
        profile = build_content_agents_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            config_preflight_path=args.config_preflight,
            attempt_authority_path=args.attempt_authority,
            source_commit=args.source_commit,
            candidate_id=args.candidate_id,
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
