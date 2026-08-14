#!/usr/bin/env python3
"""Build a website-reachable profile for one semantic-teacher edit attempt.

The profile binds a successful dry run, the exact immutable bundle, one
file-backed paid authority, a registry-bound runtime image, and a private token
file. Building it performs no upload, allocation, inference, or token lookup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from blueprint_pipeline.semantic_teacher_image_edit_paid_authority import (
    validate_semantic_teacher_image_edit_paid_authority,
)
from blueprint_pipeline.semantic_teacher_image_edit_paid_lane import (
    DRY_RUN_SCHEMA_VERSION,
)
from blueprint_pipeline.semantic_teacher_image_edit_vast import PROBE_KIND
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    TaskEvaluationLaunchError,
    canonical_digest,
)
from blueprint_pipeline.task_evaluation_live_profile import (
    LaneLiveProfileContext,
    LaneLiveProfileSpec,
    build_lane_live_profile,
    file_digest,
)


MIN_TTL_SECONDS = 1
MAX_TTL_SECONDS = 3_600


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationLaunchError(
            f"semantic_teacher_profile_input_invalid:{path.name}"
        ) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationLaunchError(
            f"semantic_teacher_profile_input_invalid:{path.name}"
        )
    return dict(value)


def _native_receipt(context: LaneLiveProfileContext) -> dict[str, Any]:
    return _read(context.receipt_path)


def _prior_spend_path(authority: Mapping[str, Any]) -> Path | None:
    record = authority.get("prior_spend_reconciliation")
    if record is None:
        return None
    if not isinstance(record, Mapping):
        raise TaskEvaluationLaunchError(
            "semantic_teacher_profile_prior_spend_invalid"
        )
    path = Path(str(record.get("path") or "")).expanduser().resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or file_digest(path) != record.get("sha256")
    ):
        raise TaskEvaluationLaunchError(
            "semantic_teacher_profile_prior_spend_invalid"
        )
    return path


def _dry_run_valid(
    value: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    receipt: Mapping[str, Any],
    source_commit: str,
) -> bool:
    return bool(
        value.get("schema_version") == DRY_RUN_SCHEMA_VERSION
        and value.get("status") == "dry_run_ready"
        and value.get("source_commit_sha") == source_commit
        and value.get("authorization_digest") == authority.get("authorization_digest")
        and value.get("bundle_sha256") == (receipt.get("bundle") or {}).get("sha256")
        and value.get("bundle_size_bytes")
        == (receipt.get("bundle") or {}).get("size_bytes")
        and value.get("backend_entry_digest") == authority.get("backend_entry_digest")
        and value.get("task_count") == authority.get("task_count")
        and value.get("camera_count") == authority.get("camera_count")
        and value.get("maximum_provider_allocations") == 1
        and value.get("automatic_retry_count") == 0
        and value.get("provider_inventory_api_zero") is True
        and value.get("provider_mutations_performed") == 0
        and value.get("dry_run_digest")
        == canonical_digest(value, digest_field="dry_run_digest")
    )


def _lane_blockers(context: LaneLiveProfileContext) -> list[str]:
    blockers: list[str] = []
    receipt = _native_receipt(context)
    authority_path = context.extra_paths["attempt_authority"]
    dry_run_path = context.extra_paths["dry_run_receipt"]
    token_path = context.extra_paths["token_file"]
    if any(
        path.is_symlink() or not path.is_file()
        for path in (authority_path, dry_run_path, token_path)
    ):
        return ["semantic_teacher_profile_input_missing_or_unsafe"]
    token_mode = token_path.stat().st_mode
    if not stat.S_ISREG(token_mode) or token_mode & 0o077:
        blockers.append("semantic_teacher_profile_token_permissions_not_0600")
    authority = _read(authority_path)
    dry_run = _read(dry_run_path)
    try:
        _prior_spend_path(authority)
    except TaskEvaluationLaunchError as exc:
        blockers.append(str(exc))
    try:
        validate_semantic_teacher_image_edit_paid_authority(
            authority,
            bundle_path=context.bundle_path,
            bundle_receipt=receipt,
            source_commit_sha=context.source_commit,
            backend_entry_digest=str(receipt.get("backend_entry_digest") or ""),
            task_count=int(receipt.get("task_count") or 0),
            camera_count=int(receipt.get("camera_count") or 0),
            maximum_hourly_rate_usd=context.max_hourly_rate_usd,
            hard_total_spend_cap_usd=context.max_spend_usd,
            hard_ttl_seconds=context.hard_ttl_seconds,
        )
    except ValueError as exc:
        blockers.append(str(exc))
    if not _dry_run_valid(
        dry_run,
        authority=authority,
        receipt=receipt,
        source_commit=context.source_commit,
    ):
        blockers.append("semantic_teacher_profile_dry_run_unbound")
    return blockers


def _lane_argv(context: LaneLiveProfileContext) -> list[str]:
    authority = _read(context.extra_paths["attempt_authority"])
    return [
        "--semantic-teacher-bundle",
        context.bundle_path,
        "--semantic-teacher-bundle-receipt",
        str(context.receipt_path),
        "--semantic-teacher-attempt-authority",
        str(context.extra_paths["attempt_authority"]),
        "--semantic-teacher-token-file",
        str(context.extra_paths["token_file"]),
        "--semantic-teacher-runtime-image-identity",
        str(authority["runtime_image_identity"]),
        "--semantic-teacher-job-dir",
        context.job_dir("semantic-teacher-job"),
        "--semantic-teacher-dry-run-output",
        context.job_dir("semantic-teacher-dry-run.json"),
        "--semantic-teacher-dry-run-receipt",
        str(context.extra_paths["dry_run_receipt"]),
        "--semantic-teacher-preflight-output",
        context.job_dir("semantic-teacher-preflight.json"),
    ]


def _immutable_inputs(context: LaneLiveProfileContext) -> list[dict[str, Any]]:
    receipt = _native_receipt(context)
    authority = _read(context.extra_paths["attempt_authority"])
    immutable = [
        {
            "name": "source_bundle_manifest",
            "path": str(context.receipt_path),
            "digest": file_digest(context.receipt_path),
        },
        {
            "name": "evaluation_run_spec",
            "path": str(context.extra_paths["dry_run_receipt"]),
            "digest": file_digest(context.extra_paths["dry_run_receipt"]),
        },
        {
            "name": "semantic_teacher_bundle",
            "path": context.bundle_path,
            "digest": str((receipt.get("bundle") or {}).get("sha256") or ""),
        },
        {
            "name": "semantic_teacher_paid_attempt_authority",
            "path": str(context.extra_paths["attempt_authority"]),
            "digest": file_digest(context.extra_paths["attempt_authority"]),
        },
    ]
    prior_spend = _prior_spend_path(authority)
    if prior_spend is not None:
        immutable.append(
            {
                "name": "semantic_teacher_prior_spend_reconciliation",
                "path": str(prior_spend),
                "digest": file_digest(prior_spend),
            }
        )
    return immutable


def _run_spec_digest(context: LaneLiveProfileContext) -> str:
    return str(_read(context.extra_paths["dry_run_receipt"])["dry_run_digest"])


SPEC = LaneLiveProfileSpec(
    profile_id_prefix="adp-semantic-teacher-image-edit-live",
    probe_kind=PROBE_KIND,
    min_ttl_seconds=MIN_TTL_SECONDS,
    max_ttl_seconds=MAX_TTL_SECONDS,
    source_bundle_id=lambda context: f"semantic-teacher-{context.source_commit[:12]}",
    source_kind="interiorgs_sage",
    lane_argv=_lane_argv,
    immutable_inputs=_immutable_inputs,
    lane_blockers=_lane_blockers,
    run_spec_digest=_run_spec_digest,
    extra_path_names=("attempt_authority", "dry_run_receipt", "token_file"),
)


def build_semantic_teacher_image_edit_live_profile(
    *,
    bundle_receipt_path: str | Path,
    attempt_authority_path: str | Path,
    dry_run_receipt_path: str | Path,
    token_file_path: str | Path,
    source_commit: str,
    raw_manifest_uri: str,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build an immutable profile after the unpaid dry run passed."""

    authority = _read(Path(attempt_authority_path).expanduser().resolve())
    return build_lane_live_profile(
        SPEC,
        bundle_receipt_path=bundle_receipt_path,
        source_commit=source_commit,
        raw_manifest_uri=raw_manifest_uri,
        max_hourly_rate_usd=float(authority.get("maximum_hourly_rate_usd") or 0),
        hard_ttl_seconds=int(authority.get("hard_ttl_seconds") or 0),
        revision=revision,
        max_spend_usd=float(authority.get("hard_total_spend_cap_usd") or 0),
        extra_paths={
            "attempt_authority": attempt_authority_path,
            "dry_run_receipt": dry_run_receipt_path,
            "token_file": token_file_path,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--attempt-authority", required=True)
    parser.add_argument("--dry-run-receipt", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--raw-manifest-uri", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        profile = build_semantic_teacher_image_edit_live_profile(
            bundle_receipt_path=args.bundle_receipt,
            attempt_authority_path=args.attempt_authority,
            dry_run_receipt_path=args.dry_run_receipt,
            token_file_path=args.token_file,
            source_commit=args.source_commit,
            raw_manifest_uri=args.raw_manifest_uri,
            revision=args.revision,
        )
    except (OSError, ValueError, TaskEvaluationLaunchError) as exc:
        print(
            json.dumps(
                {
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
    output.write_text(
        json.dumps(profile, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "built",
                "profile_id": profile["profile_id"],
                "profile_digest": profile["profile_digest"],
                "provider_mutation_performed": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
