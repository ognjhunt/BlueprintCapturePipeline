#!/usr/bin/env python3
"""Build one immutable Website-reachable scene-configuration profile."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
)
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)
from blueprint_pipeline.task_evaluation_scene_configuration_bundle import (
    PROBE_KIND,
    load_scene_configuration_provider_bundle_receipt,
)
from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import (
    MAX_TTL_SECONDS,
    MIN_TTL_SECONDS,
    validate_scene_configuration_paid_authority,
)


PROFILE_BUILDER = "build_task_evaluation_scene_configuration_live_profile.py"


def _authority(context: LaneLiveProfileContext) -> dict[str, Any]:
    path = context.extra_paths["attempt_authority"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchError(
            "scene_configuration_live_profile_authority_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise TaskEvaluationLaunchError(
            "scene_configuration_live_profile_authority_invalid"
        )
    return validate_scene_configuration_paid_authority(
        value, bundle_receipt=context.receipt
    )


def _blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    try:
        receipt = load_scene_configuration_provider_bundle_receipt(
            context.receipt_path, expected_source_commit=context.source_commit
        )
        authority = _authority(context)
    except (OSError, ValueError):
        return ["scene_configuration_live_profile_binding_invalid"]
    if receipt.get("bundle_sha256") != context.bundle_sha256:
        blockers.append("scene_configuration_live_profile_bundle_mismatch")
    if authority.get("maximum_hourly_rate_usd") != context.max_hourly_rate_usd:
        blockers.append("scene_configuration_live_profile_rate_mismatch")
    if (
        authority.get("maximum_single_resource_ttl_seconds")
        != context.hard_ttl_seconds
    ):
        blockers.append("scene_configuration_live_profile_ttl_mismatch")
    if authority.get("hard_attempt_spend_cap_usd") != context.max_spend_usd:
        blockers.append("scene_configuration_live_profile_spend_mismatch")
    return blockers


def _argv(context: LaneLiveProfileContext) -> list[str]:
    return [
        "--scene-configuration-bundle-receipt",
        str(context.receipt_path),
        "--scene-configuration-attempt-authority",
        str(context.extra_paths["attempt_authority"]),
        "--scene-configuration-job-dir",
        context.job_dir("scene-configuration-job"),
    ]


def _inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    authority = context.extra_paths["attempt_authority"]
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
            "name": "scene_configuration_attempt_authority",
            "path": str(authority),
            "digest": file_digest(authority),
        },
    ]


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="task-evaluation-scene-configuration",
    profile_builder="build_task_evaluation_scene_configuration_live_profile.py",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: str(context.receipt["run_id"]),
    source_kind="interiorgs_sage",
    lane_argv=_argv,
    immutable_inputs=_inputs,
    lane_blockers=_blockers,
    profile_fields=lambda context: {
        "task_evaluation_run": {
            "run_mode": "scene_configuration",
            "team_namespace": context.extra_values["team_namespace"],
            "scene_id": context.extra_values["scene_id"],
            "task_id": context.extra_values["task_id"],
            "configuration_run_id": context.receipt["run_id"],
            "evaluation_episode_executed": False,
        }
    },
    declared_spend=lambda context: context.max_spend_usd,
    claim_ceiling="development_only",
    extra_path_names=("attempt_authority",),
    one_use_standing_authority_required=True,
    additional_terminal_path_fields=("execution_result_path",),
)


def build_scene_configuration_live_profile(
    *,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str,
    max_hourly_rate_usd: float,
    hard_ttl_seconds: int,
    max_spend_usd: float,
    team_namespace: str,
    scene_id: str,
    task_id: str,
    pod_name: str | None = None,
) -> dict[str, Any]:
    return build_lane_live_profile(
        SPEC,
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=max_hourly_rate_usd,
        hard_ttl_seconds=hard_ttl_seconds,
        max_spend_usd=max_spend_usd,
        revision=revision,
        pod_name=pod_name,
        profile_binding_identity=pod_name,
        extra_paths={"attempt_authority": attempt_authority_path},
        extra_values={
            "team_namespace": team_namespace,
            "scene_id": scene_id,
            "task_id": task_id,
        },
    )


def _write_exclusive(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ValueError("scene_configuration_live_profile_output_conflict")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--raw-manifest-uri",
        required=True,
        help="Local digest-bound content-addressed publication receipt for this run spec.",
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--max-spend-usd", required=True, type=float)
    parser.add_argument("--team-namespace", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--pod-name",
        default=None,
        help=(
            "Exact provider resource name the paid attempt authority binds. "
            "The allocator refuses a pod name that differs from the "
            "authority's resource_name, so the launch graph passes the "
            "activation id here. Defaults to the profile id."
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        profile = build_scene_configuration_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            attempt_authority_path=args.attempt_authority,
            source_commit=args.source_commit,
            raw_manifest_uri=args.raw_manifest_uri,
            revision=args.revision,
            max_hourly_rate_usd=args.max_hourly_rate_usd,
            hard_ttl_seconds=args.hard_ttl_seconds,
            max_spend_usd=args.max_spend_usd,
            team_namespace=args.team_namespace,
            scene_id=args.scene_id,
            task_id=args.task_id,
            pod_name=args.pod_name,
        )
    except (OSError, ValueError, TaskEvaluationLaunchError) as exc:
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
    _write_exclusive(Path(args.output).resolve(), profile)
    print(json.dumps(profile, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
