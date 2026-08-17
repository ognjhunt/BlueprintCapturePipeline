#!/usr/bin/env python3
"""Build a live launch profile for the paired-target native import probe.

This is the appearance path the program now bets on: a whole-frame GPT teacher
paired with the original outside-mask anchors, then paired ArtiFixer3D, then a
paid zero-retry native-Isaac import gate that says whether the registered asset
actually loads in a real simulator.

It arrived with a bundle, a preflight, a render request, a Vast runner, and a
closeout -- and no launch profile, which is the one thing that carries a lane
across the website boundary. Without this it is reachable by hand and not by
the pipeline, which is the state every other lane started in.

The skeleton -- residency, spend binding, terminal contract, validation -- is
shared with every other paid lane in `task_evaluation_live_profile`. What is
here is only what makes this lane different.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.paired_target_native_import_vast import (
    MAX_HARD_CAP_USD,
    PROBE_KIND,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)

# The allocator refuses a TTL outside this band for this probe.
MIN_TTL_SECONDS = 1_800
MAX_TTL_SECONDS = 7_200


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    receipt = context.receipt
    if not 0 < context.max_hourly_rate_usd <= context.max_spend_usd:
        blockers.append("budget_invalid")
    # The allocator compares the bundle's implementation commit against the
    # commit the control plane is running, so a mismatch caught here is one
    # fewer consumed attempt authority.
    if receipt.get("implementation_commit") != context.source_commit:
        blockers.append(
            f"bundle_commit_not_source_commit:{receipt.get('implementation_commit')}"
        )

    authority_path = context.extra_paths.get("attempt_authority")
    if authority_path is None or not authority_path.is_file():
        blockers.append("attempt_authority_missing")
    else:
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        if authority.get("hard_attempt_spend_cap_usd") != context.max_spend_usd:
            blockers.append("attempt_authority_spend_cap_mismatch")
        if authority.get("maximum_single_resource_ttl_seconds") != context.hard_ttl_seconds:
            blockers.append("attempt_authority_ttl_mismatch")
        if authority.get("bundle_sha256") != receipt.get("bundle_sha256"):
            blockers.append("attempt_authority_bundle_mismatch")
        if list(authority.get("excluded_vast_machine_ids") or []) != list(
            context.extra_values.get("excluded_machine_ids") or []
        ):
            blockers.append("attempt_authority_machine_exclusions_mismatch")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    argv = [
        "--paired-target-native-import-bundle-receipt", str(context.receipt_path),
        "--paired-target-native-import-attempt-authority",
        str(context.extra_paths["attempt_authority"]),
        "--adp-job-dir", context.job_dir("paired-target-native-import-job"),
        "--adp-max-hourly-rate-usd", str(context.max_hourly_rate_usd),
        "--adp-max-spend-usd", str(context.max_spend_usd),
        "--adp-hard-ttl-seconds", str(context.hard_ttl_seconds),
    ]
    for machine_id in context.extra_values.get("excluded_machine_ids") or []:
        argv += ["--adp-excluded-vast-machine-id", str(machine_id)]
    return argv


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
            "name": "paired_target_native_import_bundle",
            # The path the receipt resolved to here, not the one it was built at.
            "path": context.bundle_path,
            "digest": context.bundle_sha256,
        },
        {
            "name": "paired_target_native_import_paid_attempt_authority",
            "path": str(context.extra_paths["attempt_authority"]),
            "digest": file_digest(context.extra_paths["attempt_authority"]),
        },
    ]


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp-paired-target-native-import-live",
    profile_builder="build_paired_target_native_import_live_profile.py",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: (
        f"paired-target-native-import-{context.source_commit[:12]}"
    ),
    # The probe's stage is built over the public scene pair, not a new capture:
    # "paired target" describes what is registered into it.
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    extra_path_names=("attempt_authority",),
)


def build_paired_target_native_import_live_profile(
    *,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str | None = None,
    max_hourly_rate_usd: float = 1.0,
    max_spend_usd: float = MAX_HARD_CAP_USD,
    hard_ttl_seconds: int = 7_200,
) -> dict[str, Any]:
    """Derive a live profile from the bundle receipt it will run."""
    authority = json.loads(
        Path(attempt_authority_path).expanduser().resolve().read_text(encoding="utf-8")
    )
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
        extra_values={
            "excluded_machine_ids": list(
                authority.get("excluded_vast_machine_ids") or []
            )
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--attempt-authority", required=True)
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
    parser.add_argument("--max-spend-usd", type=float, default=MAX_HARD_CAP_USD)
    parser.add_argument("--hard-ttl-seconds", type=int, default=7_200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        profile = build_paired_target_native_import_live_profile(
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
