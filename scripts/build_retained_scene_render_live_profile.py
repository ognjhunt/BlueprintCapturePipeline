#!/usr/bin/env python3
"""Build a live launch profile for the retained-scene GPU render probe.

The first live profile for this probe was written by hand and each mistake was
caught by a different fail-closed gate, one paid round trip at a time: a
source-bundle manifest recorded at the authoring machine's path, a missing
attempt-authority argument that only matters under ``--execute``, and an
instance allowlist that has to match a value living in another file.

Everything a profile needs is already stated by the receipts the run will use,
so this derives it rather than asking an operator to keep several documents
consistent by hand.

The skeleton -- residency, spend binding, terminal contract, validation -- lives
in `task_evaluation_live_profile`, shared with every other paid lane. What is
here is only what makes this lane different.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    allowlist_arguments,
    build_lane_live_profile,
    file_digest,
)

PROBE_KIND = "adp-retained-scene-gpu-render"
# The allocator refuses a TTL outside this band for this probe.
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 10_800


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(f"profile_input_not_object:{path.name}")
    return dict(value)


def _instance_allowlist(context: LaneLiveProfileContext) -> tuple[list[int], list[str]]:
    """The allowlist must equal the bundle authority's exactly.

    Deriving it removes the chance of the two drifting apart in separate files,
    which is the failure that cost a paid round trip.
    """

    if not isinstance(context.receipt.get("execution_authority"), Mapping):
        return [], ["bundle_execution_authority_missing"]
    source = Path(
        str((context.resolutions.get("execution_authority") or {}).get("path") or "")
    ).expanduser()
    if not source.is_file():
        return [], ["bundle_execution_authority_unreadable"]
    paid = _read(source).get("paid_compute")
    if not isinstance(paid, Mapping):
        return [], ["bundle_execution_authority_paid_compute_missing"]
    return (
        sorted({int(item) for item in paid.get("external_instance_allowlist") or []}),
        [],
    )


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    receipt = context.receipt
    if receipt.get("probe_kind") != PROBE_KIND:
        blockers.append(f"bundle_probe_kind_mismatch:{receipt.get('probe_kind')}")
    # A profile pinned to a commit the control plane is not running is refused
    # by the allocator, so catch it here rather than at the paid boundary.
    if receipt.get("blueprint_commit") != context.source_commit:
        blockers.append("bundle_commit_not_source_commit")

    # The spend cap belongs to the bundle, not to this builder: the allocator
    # reads it off the receipt and rejects any rate/TTL pair whose worst case
    # exceeds it.
    cap = receipt.get("hard_total_spend_cap_usd")
    if not isinstance(cap, (int, float)) or isinstance(cap, bool):
        blockers.append("bundle_spend_cap_missing")
        cap = 0.0
    worst_case = context.max_hourly_rate_usd * context.hard_ttl_seconds / 3600.0
    if worst_case > float(cap):
        blockers.append(f"worst_case_spend_exceeds_bundle_cap:{worst_case}>{cap}")

    blockers.extend(_instance_allowlist(context)[1])
    if not context.extra_paths.get("attempt_authority", Path("/nonexistent")).is_file():
        blockers.append("attempt_authority_missing")
    if not context.extra_paths.get("request_manifest", Path("/nonexistent")).is_file():
        blockers.append("request_manifest_missing")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    return [
        "--adp-retained-scene-render-bundle-receipt", str(context.receipt_path),
        "--adp-retained-scene-render-attempt-authority",
        str(context.extra_paths["attempt_authority"]),
        "--adp-retained-scene-render-job-dir",
        context.job_dir("retained-scene-job"),
        "--adp-retained-scene-render-max-hourly-rate-usd",
        str(context.max_hourly_rate_usd),
        "--adp-retained-scene-render-hard-ttl-seconds", str(context.hard_ttl_seconds),
        *allowlist_arguments(_instance_allowlist(context)[0]),
    ]


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    # The request manifest is this probe's source-bundle manifest; the bundle
    # receipt is what actually pins the run.
    request = context.extra_paths["request_manifest"]
    return [
        {
            "name": "source_bundle_manifest",
            "path": str(request),
            "digest": file_digest(request),
        },
        {
            "name": "evaluation_run_spec",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "retained_scene_render_bundle",
            # The path the receipt resolved to here, not the one it was built at.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
    ]


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp-retained-scene-render-live",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: f"retained-scene-render-{context.source_commit[:12]}",
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    # What this profile can actually spend -- rate times TTL -- not the bundle's
    # lifetime ceiling. The two were the same number, which made every standing
    # authorization reserve the whole bundle cap for a single launch. The bundle
    # cap still bounds it: a worst case above it is refused in `_lane_blockers`.
    run_spec_digest=lambda context: file_digest(context.extra_paths["request_manifest"]),
    extra_path_names=("request_manifest", "attempt_authority"),
)


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

    return build_lane_live_profile(
        SPEC,
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision,
        extra_paths={
            "request_manifest": request_manifest_path,
            "attempt_authority": attempt_authority_path,
        },
    )


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
