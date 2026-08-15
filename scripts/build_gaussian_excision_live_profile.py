#!/usr/bin/env python3
"""Build a live launch profile for the released-code Gaussian ownership audit.

The bundle has read `status: ready` for some time and the allocator has always
had a branch for the probe. What was missing was a launch profile, which is the
one thing that carries a lane across the website boundary.

This lane's budget is not a band but three fixed points: the allocator refuses
any hourly rate above 0.60, any spend cap that is not exactly 1.50, and any TTL
that is not exactly 3600. Those are stated here so the refusal happens at
authoring time rather than after a provider has been handed over.

The skeleton -- residency, spend binding, terminal contract, validation -- is
shared with every other paid lane in `task_evaluation_live_profile`.

Reads retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace
from typing import Any, Sequence

from blueprint_pipeline.adp_gaussian_excision_vast import PROBE_KIND
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    allowlist_arguments,
    build_lane_live_profile,
    file_digest,
)

#: The allocator compares each of these for exact equality, not a range.
EXACT_HARD_TTL_SECONDS = 3_600
EXACT_MAX_SPEND_USD = 1.50
MAX_HOURLY_RATE_USD = 0.60


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    receipt = context.receipt
    if context.max_spend_usd != EXACT_MAX_SPEND_USD:
        blockers.append(f"gaussian_excision_hard_cap_must_be_exact:{context.max_spend_usd}")
    if context.hard_ttl_seconds != EXACT_HARD_TTL_SECONDS:
        blockers.append(f"gaussian_excision_hard_ttl_must_be_exact:{context.hard_ttl_seconds}")
    if not 0 < context.max_hourly_rate_usd <= MAX_HOURLY_RATE_USD:
        blockers.append(f"gaussian_excision_hourly_rate_out_of_band:{context.max_hourly_rate_usd}")
    # The bundle carries its own ceiling; the attempt authority is validated
    # against it by the allocator, so a disagreement here is caught before the
    # authority is consumed rather than at the paid boundary.
    if receipt.get("hard_cap_usd") != EXACT_MAX_SPEND_USD:
        blockers.append(f"bundle_hard_cap_mismatch:{receipt.get('hard_cap_usd')}")
    if receipt.get("hard_ttl_seconds") != EXACT_HARD_TTL_SECONDS:
        blockers.append(f"bundle_hard_ttl_mismatch:{receipt.get('hard_ttl_seconds')}")
    if receipt.get("blueprint_commit") != context.source_commit:
        blockers.append("bundle_commit_not_source_commit")
    if not receipt.get("freeze_digest"):
        blockers.append("bundle_freeze_digest_missing")

    authority_path = context.extra_paths.get("attempt_authority")
    if authority_path is None or not authority_path.is_file():
        blockers.append("attempt_authority_missing")
    else:
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        if authority.get("freeze_digest") != receipt.get("freeze_digest"):
            blockers.append("attempt_authority_freeze_digest_mismatch")
        if authority.get("bundle_sha256") != receipt.get("bundle_sha256"):
            blockers.append("attempt_authority_bundle_mismatch")
        if authority.get("hard_attempt_spend_cap_usd") != EXACT_MAX_SPEND_USD:
            blockers.append("attempt_authority_spend_cap_mismatch")
        ordinal = authority.get("paid_attempt_ordinal")
        prior = context.extra_paths.get("previous_attempt_receipt")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            blockers.append("attempt_authority_ordinal_invalid")
        if isinstance(ordinal, int) and ordinal > 1 and prior is None:
            blockers.append("previous_attempt_receipt_missing_for_ordinal")
        if isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal > 1 and prior:
            try:
                previous = json.loads(prior.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                blockers.append("previous_attempt_receipt_invalid")
            else:
                previous_digest = previous.get("receipt_digest")
                if previous_digest != canonical_digest(
                    previous, digest_field="receipt_digest"
                ):
                    blockers.append("previous_attempt_receipt_digest_invalid")
                if authority.get("previous_attempt_receipt_digest") != previous_digest:
                    blockers.append("previous_attempt_receipt_authority_mismatch")

    avoidlist = context.extra_paths.get("machine_avoidlist")
    if avoidlist is not None and not avoidlist.is_file():
        blockers.append("machine_avoidlist_missing")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    argv = [
        "--adp-gaussian-excision-bundle-receipt", str(context.receipt_path),
        "--adp-gaussian-excision-attempt-authority",
        str(context.extra_paths["attempt_authority"]),
        "--adp-job-dir", context.job_dir("gaussian-excision-job"),
        "--adp-max-hourly-rate-usd", str(context.max_hourly_rate_usd),
        "--adp-max-spend-usd", str(EXACT_MAX_SPEND_USD),
        "--adp-hard-ttl-seconds", str(EXACT_HARD_TTL_SECONDS),
    ]
    prior = context.extra_paths.get("previous_attempt_receipt")
    if prior is not None:
        # Attempts are ordinal here: a second paid attempt has to be authorized
        # against the sealed evidence of the first.
        argv += ["--adp-gaussian-excision-previous-attempt-receipt", str(prior)]
    avoidlist = context.extra_paths.get("machine_avoidlist")
    if avoidlist is not None:
        argv += ["--adp-machine-avoidlist", str(avoidlist)]
    return argv + allowlist_arguments(())


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    rows = [
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
            "name": "gaussian_excision_bundle",
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
    avoidlist = context.extra_paths.get("machine_avoidlist")
    if avoidlist is not None:
        rows.append(
            {
                "name": "machine_avoidlist",
                "path": str(avoidlist),
                "digest": file_digest(avoidlist),
            }
        )
    prior = context.extra_paths.get("previous_attempt_receipt")
    if prior is not None:
        rows.append(
            {
                "name": "previous_attempt_receipt",
                "path": str(prior),
                "digest": file_digest(prior),
            }
        )
    return rows


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp-gaussian-excision-live",
    profile_builder="build_gaussian_excision_live_profile.py",
    probe_kind=PROBE_KIND,
    # A single admitted value, expressed as a degenerate band.
    min_ttl_seconds=EXACT_HARD_TTL_SECONDS - 1,
    max_ttl_seconds=EXACT_HARD_TTL_SECONDS,
    source_bundle_id=lambda context: f"gaussian-excision-{context.source_commit[:12]}",
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    extra_path_names=("attempt_authority",),
)


def build_gaussian_excision_live_profile(
    *,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str | None = None,
    max_hourly_rate_usd: float = 0.60,
    machine_avoidlist_path: str | Path | None = None,
    previous_attempt_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Derive a live profile from the bundle receipt it will run."""

    authority_path = Path(attempt_authority_path).expanduser().resolve()
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    ordinal = authority.get("paid_attempt_ordinal", 1)
    if (
        previous_attempt_receipt_path is None
        and isinstance(ordinal, int)
        and not isinstance(ordinal, bool)
        and ordinal > 1
    ):
        attempts = authority.get("prior_terminal_attempts") or []
        if attempts and isinstance(attempts[-1], dict):
            previous_attempt_receipt_path = attempts[-1].get("result_path")
    extra: dict[str, Any] = {"attempt_authority": authority_path}
    names = list(SPEC.extra_path_names)
    if machine_avoidlist_path is not None:
        extra["machine_avoidlist"] = machine_avoidlist_path
        names.append("machine_avoidlist")
    if previous_attempt_receipt_path is not None:
        extra["previous_attempt_receipt"] = previous_attempt_receipt_path
        names.append("previous_attempt_receipt")
    spec = SPEC if names == list(SPEC.extra_path_names) else replace(
        SPEC, extra_path_names=tuple(names)
    )

    return build_lane_live_profile(
        spec,
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=EXACT_HARD_TTL_SECONDS,
        revision=revision,
        max_spend_usd=EXACT_MAX_SPEND_USD,
        extra_paths=extra,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--raw-manifest-uri",
        required=True,
        help="Local digest-bound GCS publication receipt for this run spec.",
    )
    parser.add_argument(
        "--revision",
        help="Distinguish a rebuilt profile whose inputs changed at the same commit.",
    )
    parser.add_argument("--max-hourly-rate-usd", type=float, default=0.60)
    parser.add_argument("--machine-avoidlist")
    parser.add_argument(
        "--previous-attempt-receipt",
        help="Sealed receipt of the prior paid attempt; required from the second on.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        profile = build_gaussian_excision_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            attempt_authority_path=args.attempt_authority,
            source_commit=args.source_commit,
            raw_manifest_uri=args.raw_manifest_uri,
            revision=args.revision,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            machine_avoidlist_path=args.machine_avoidlist,
            previous_attempt_receipt_path=args.previous_attempt_receipt,
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
