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

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
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

    scene_id = str(context.extra_values.get("scene_id") or "")
    native_manifest_path = context.extra_paths.get("native_probe_manifest")
    candidate_usd_path = context.extra_paths.get("candidate_usd")
    native_manifest: dict[str, Any] = {}
    if native_manifest_path is None or not native_manifest_path.is_file():
        blockers.append("native_probe_manifest_missing")
    else:
        native_manifest = json.loads(native_manifest_path.read_text(encoding="utf-8"))
        manifest_identity = native_manifest.get(
            "manifest_digest", native_manifest.get("receipt_digest")
        )
        if native_manifest.get("scene_id") != scene_id:
            blockers.append("native_probe_manifest_scene_mismatch")
        if (
            file_digest(native_manifest_path)
            != receipt.get("native_probe_manifest_sha256")
        ):
            blockers.append("native_probe_manifest_digest_mismatch")
        if manifest_identity != receipt.get("native_probe_manifest_digest"):
            blockers.append("native_probe_manifest_identity_mismatch")
        predecessor = native_manifest.get("paired_native_predecessor") or {}
        if (
            not isinstance(predecessor, dict)
            or predecessor.get("binding_digest")
            != canonical_digest(predecessor, digest_field="binding_digest")
            or predecessor.get("scene_id") != scene_id
            or predecessor.get("candidate_usd_sha256")
            != receipt.get("candidate_usd_sha256")
            or predecessor.get("task_id") != receipt.get("task_id")
            or predecessor.get("asset_id") != receipt.get("asset_id")
            or native_manifest.get("validation_mode")
            != receipt.get("validation_mode")
            or predecessor.get("binding_digest")
            != receipt.get("predecessor_binding_digest")
            or receipt.get("paired_native_predecessor") != predecessor
        ):
            blockers.append("paired_native_predecessor_binding_invalid")
        else:
            for role in (
                "bundle_receipt",
                "request",
                "terminal_result",
                "runtime_result",
                "candidate_probe",
            ):
                record = predecessor.get(role) or {}
                path = Path(str(record.get("path") or "")).expanduser().resolve()
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size != record.get("size_bytes")
                    or file_digest(path) != record.get("sha256")
                ):
                    blockers.append(f"paired_native_predecessor_input_invalid:{role}")
    if candidate_usd_path is None or not candidate_usd_path.is_file():
        blockers.append("candidate_usd_missing")
    elif file_digest(candidate_usd_path) != receipt.get("candidate_usd_sha256"):
        blockers.append("candidate_usd_digest_mismatch")
    if receipt.get("scene_id") != scene_id:
        blockers.append("bundle_scene_id_mismatch")

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
        if authority.get("scene_id") != scene_id:
            blockers.append("attempt_authority_scene_mismatch")
        if authority.get("task_id") != receipt.get("task_id"):
            blockers.append("attempt_authority_task_mismatch")
        if authority.get("asset_id") != receipt.get("asset_id"):
            blockers.append("attempt_authority_asset_mismatch")
        if authority.get("validation_mode") != receipt.get("validation_mode"):
            blockers.append("attempt_authority_validation_mode_mismatch")
        if authority.get("candidate_usd_sha256") != receipt.get(
            "candidate_usd_sha256"
        ):
            blockers.append("attempt_authority_candidate_mismatch")
        if authority.get("native_probe_manifest_sha256") != receipt.get(
            "native_probe_manifest_sha256"
        ):
            blockers.append("attempt_authority_native_manifest_mismatch")
        if authority.get("native_probe_manifest_digest") != receipt.get(
            "native_probe_manifest_digest"
        ):
            blockers.append("attempt_authority_native_manifest_identity_mismatch")
        if authority.get("predecessor_binding_digest") != receipt.get(
            "predecessor_binding_digest"
        ):
            blockers.append("attempt_authority_predecessor_mismatch")
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
    inputs = [
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
        {
            "name": "simready_native_probe_manifest",
            "path": str(context.extra_paths["native_probe_manifest"]),
            "digest": file_digest(context.extra_paths["native_probe_manifest"]),
        },
        {
            "name": "simready_candidate_usd",
            "path": str(context.extra_paths["candidate_usd"]),
            "digest": file_digest(context.extra_paths["candidate_usd"]),
        },
        {
            "name": "simready_paid_attempt_authority",
            "path": str(context.extra_paths["attempt_authority"]),
            "digest": file_digest(context.extra_paths["attempt_authority"]),
        },
    ]
    manifest = json.loads(
        context.extra_paths["native_probe_manifest"].read_text(encoding="utf-8")
    )
    predecessor = manifest.get("paired_native_predecessor") or {}
    for role in (
        "bundle_receipt",
        "request",
        "terminal_result",
        "runtime_result",
        "candidate_probe",
    ):
        record = predecessor[role]
        inputs.append(
            {
                "name": f"simready_paired_native_{role}",
                "path": str(Path(record["path"]).expanduser().resolve()),
                "digest": record["sha256"],
            }
        )
    return inputs


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp009b-simready-isaac-live",
    profile_builder="build_simready_isaac_live_profile.py",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: (
        f"simready-isaac-{context.extra_values['scene_id']}-"
        f"{context.receipt['task_id']}-"
        f"{context.receipt['candidate_usd_sha256'].removeprefix('sha256:')[:12]}-"
        f"{context.source_commit[:12]}"
    ),
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    extra_path_names=("attempt_authority", "native_probe_manifest", "candidate_usd"),
)


def build_simready_isaac_live_profile(
    *,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    native_probe_manifest_path: str | Path,
    scene_id: str,
    candidate_usd_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str | None = None,
    max_hourly_rate_usd: float = 1.0,
    max_spend_usd: float = 3.0,
    hard_ttl_seconds: int = 7_200,
) -> dict[str, Any]:
    """Derive a live profile from the bundle receipt it will run."""

    normalized_scene_id = str(scene_id or "").strip()
    candidate = Path(candidate_usd_path).expanduser().resolve()
    if (
        not normalized_scene_id
        or Path(normalized_scene_id).name != normalized_scene_id
        or not candidate.is_file()
    ):
        raise TaskEvaluationLaunchError("simready_live_scene_or_candidate_invalid")
    candidate_revision = file_digest(candidate).removeprefix("sha256:")[:12]
    return build_lane_live_profile(
        SPEC,
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        revision=revision or f"{normalized_scene_id}-{candidate_revision}",
        max_spend_usd=max_spend_usd,
        extra_paths={
            "attempt_authority": attempt_authority_path,
            "native_probe_manifest": native_probe_manifest_path,
            "candidate_usd": candidate,
        },
        extra_values={"scene_id": normalized_scene_id},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--native-probe-manifest", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--candidate-usd", required=True)
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
        profile = build_simready_isaac_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            attempt_authority_path=args.attempt_authority,
            native_probe_manifest_path=args.native_probe_manifest,
            scene_id=args.scene_id,
            candidate_usd_path=args.candidate_usd,
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
