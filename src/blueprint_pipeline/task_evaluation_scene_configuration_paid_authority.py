"""Fresh single-use spend authority for one Website-started scene configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adp_task_evaluation_abstention import valid_vast_provider_zero_api_call
from .decision_evidence_contracts import canonical_digest
from .project_spend_reconciliation import validate_project_spend_reconciliation
from .task_evaluation_scene_configuration_bundle import (
    load_scene_configuration_provider_bundle_receipt,
)


AUTHORITY_SCHEMA_VERSION = "task_evaluation_scene_configuration_paid_authority.v1"
AGGREGATE_GOAL_SPEND_CAP_USD = 50.0
MAX_ATTEMPT_SPEND_USD = 1.0
MAX_HOURLY_RATE_USD = 0.80
MAX_TTL_SECONDS = 9_000
MIN_TTL_SECONDS = 600
MAX_PROVIDER_ZERO_AGE_SECONDS = 900
_OCI_DIGEST = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}")
_RESOURCE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{15,127}")


class TaskEvaluationSceneConfigurationAuthorityError(ValueError):
    """A configuration authority was incomplete, stale, or not single-use."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAuthorityError(code) from exc
    if path.is_symlink() or not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationAuthorityError(code)
    return dict(value)


def _aware(value: Any, *, code: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise TaskEvaluationSceneConfigurationAuthorityError(code) from exc
    if parsed.tzinfo is None:
        raise TaskEvaluationSceneConfigurationAuthorityError(code)
    return parsed.astimezone(timezone.utc)


def _provider_zero(path: Path) -> dict[str, Any]:
    value = _read(path, code="scene_configuration_provider_zero_invalid")
    if (
        value.get("schema_version") != "adp_paid_provider_zero.v1"
        or value.get("provider") != "vast"
        or value.get("api_confirmed") is not True
        or value.get("global_live_resource_count") != 0
        or value.get("provider_zero") is not True
        or value.get("inventory") != []
        or not valid_vast_provider_zero_api_call(value.get("api_command"))
        or value.get("raw_secret_values_recorded") is not False
        or not isinstance(value.get("stderr_present"), bool)
        or value.get("provider_zero_digest")
        != canonical_digest(value, digest_field="provider_zero_digest")
    ):
        raise TaskEvaluationSceneConfigurationAuthorityError(
            "scene_configuration_provider_zero_invalid"
        )
    return value


def _budget_valid(*, rate: Any, cap: Any, ttl: Any) -> bool:
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or isinstance(ttl, bool)
        or not isinstance(ttl, int)
    ):
        return False
    rate_value = float(rate)
    cap_value = float(cap)
    return (
        math.isfinite(rate_value)
        and math.isfinite(cap_value)
        and 0 < rate_value <= MAX_HOURLY_RATE_USD
        and 0 < cap_value <= MAX_ATTEMPT_SPEND_USD
        and MIN_TTL_SECONDS <= ttl <= MAX_TTL_SECONDS
        and rate_value * ttl / 3600.0 <= cap_value
    )


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    payload = (
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short authority write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o440)
        directory = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.close(descriptor)


def materialize_scene_configuration_paid_authority(
    *,
    bundle_receipt_path: str | Path,
    project_spend_reconciliation_path: str | Path,
    initial_provider_zero_path: str | Path,
    authorization_reference: str,
    authorized_by: str,
    authorized_on: str,
    source_commit: str,
    container_image: str,
    resource_name: str,
    max_hourly_rate_usd: float,
    hard_cap_usd: float,
    hard_ttl_seconds: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal one fresh project-spend-derived authority; retries are impossible."""

    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    receipt = load_scene_configuration_provider_bundle_receipt(
        receipt_path, expected_source_commit=source_commit
    )
    project_path = Path(project_spend_reconciliation_path).expanduser().resolve()
    project, project_record = validate_project_spend_reconciliation(project_path)
    zero_path = Path(initial_provider_zero_path).expanduser().resolve()
    zero = _provider_zero(zero_path)
    authorized_time = _aware(
        authorized_on, code="scene_configuration_authorized_on_invalid"
    )
    zero_time = _aware(
        zero.get("observed_at_utc"), code="scene_configuration_provider_zero_time_invalid"
    )
    project_total = float(project["total_cost_usd"])
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or _OCI_DIGEST.fullmatch(container_image) is None
        or _RESOURCE_NAME.fullmatch(resource_name) is None
        or not _budget_valid(
            rate=max_hourly_rate_usd, cap=hard_cap_usd, ttl=hard_ttl_seconds
        )
        or zero_time > authorized_time
        or (authorized_time - zero_time).total_seconds()
        > MAX_PROVIDER_ZERO_AGE_SECONDS
        or project_total + hard_cap_usd > AGGREGATE_GOAL_SPEND_CAP_USD
    ):
        raise TaskEvaluationSceneConfigurationAuthorityError(
            "scene_configuration_authority_configuration_invalid"
        )
    authority: dict[str, Any] = {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": authorization_reference.strip(),
        "authorized_by": authorized_by.strip(),
        "authorized_on": authorized_time.isoformat().replace("+00:00", "Z"),
        "purpose": "one_shot_task_evaluation_scene_configuration",
        "provider": "vast",
        "paid_compute_authorized": True,
        "maximum_paid_attempts": 1,
        "maximum_provider_allocations": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "retry_cap": 0,
        "bundle_receipt": _record(receipt_path),
        "bundle_sha256": receipt["bundle_sha256"],
        "portable_construction_envelope_digest": receipt[
            "portable_construction_envelope_digest"
        ],
        "toolchain_digest": receipt["toolchain_digest"],
        "run_id": receipt["run_id"],
        "source_commit": source_commit,
        "container_image": container_image,
        "resource_name": resource_name,
        "hard_attempt_spend_cap_usd": hard_cap_usd,
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": project_total,
        "aggregate_goal_spend_cap_usd": AGGREGATE_GOAL_SPEND_CAP_USD,
        "project_spend_reconciliation": project_record,
        "initial_provider_zero": {
            **_record(zero_path),
            "provider_zero_digest": zero["provider_zero_digest"],
        },
        "active_instance_allowlist": {
            "external_provider_owned": [],
            "same_goal_concurrent": [],
        },
        "raw_interiorgs_bytes_authorized_for_provider": False,
        "evaluation_episode_authorized": False,
        "authority_digest": "",
    }
    authority["authority_digest"] = canonical_digest(
        authority, digest_field="authority_digest"
    )
    destination = Path(output_path).expanduser().resolve()
    _write_exclusive(destination, authority)
    return authority


def validate_scene_configuration_paid_authority(
    value: Mapping[str, Any], *, bundle_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Reopen nested spend/zero bytes and validate one exact authority."""

    authority = dict(value)
    errors: list[str] = []
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authority_kind")
        != "explicit_user_direction_in_current_goal"
        or authority.get("purpose")
        != "one_shot_task_evaluation_scene_configuration"
        or authority.get("provider") != "vast"
        or authority.get("paid_compute_authorized") is not True
        or authority.get("maximum_paid_attempts") != 1
        or authority.get("maximum_provider_allocations") != 1
        or authority.get("maximum_automatic_retries") != 0
        or authority.get("automatic_paid_retry_authorized") is not False
        or authority.get("retry_cap") != 0
        or authority.get("bundle_sha256") != bundle_receipt.get("bundle_sha256")
        or authority.get("run_id") != bundle_receipt.get("run_id")
        or authority.get("source_commit") != bundle_receipt.get("source_commit")
        or authority.get("portable_construction_envelope_digest")
        != bundle_receipt.get("portable_construction_envelope_digest")
        or authority.get("toolchain_digest") != bundle_receipt.get("toolchain_digest")
        or _OCI_DIGEST.fullmatch(str(authority.get("container_image") or "")) is None
        or _RESOURCE_NAME.fullmatch(str(authority.get("resource_name") or "")) is None
        or not _budget_valid(
            rate=authority.get("maximum_hourly_rate_usd"),
            cap=authority.get("hard_attempt_spend_cap_usd"),
            ttl=authority.get("maximum_single_resource_ttl_seconds"),
        )
        or authority.get("raw_interiorgs_bytes_authorized_for_provider") is not False
        or authority.get("evaluation_episode_authorized") is not False
        or authority.get("active_instance_allowlist")
        != {"external_provider_owned": [], "same_goal_concurrent": []}
        or authority.get("authority_digest")
        != canonical_digest(authority, digest_field="authority_digest")
    ):
        errors.append("authority_contract_invalid")
    try:
        project, project_record = validate_project_spend_reconciliation(
            str((authority.get("project_spend_reconciliation") or {}).get("path") or ""),
            expected_total_cost_usd=authority.get(
                "aggregate_goal_spend_before_attempt_usd"
            ),
        )
        if project_record != authority.get("project_spend_reconciliation"):
            raise ValueError("record mismatch")
        zero_record = authority.get("initial_provider_zero") or {}
        zero_path = Path(str(zero_record.get("path") or "")).expanduser().resolve()
        zero = _provider_zero(zero_path)
        if (
            _record(zero_path)
            != {key: zero_record.get(key) for key in ("path", "sha256", "size_bytes")}
            or zero_record.get("provider_zero_digest") != zero.get("provider_zero_digest")
        ):
            raise ValueError("record mismatch")
        authorized = _aware(
            authority.get("authorized_on"), code="scene_configuration_authorized_on_invalid"
        )
        observed = _aware(
            zero.get("observed_at_utc"), code="scene_configuration_provider_zero_time_invalid"
        )
        if (
            observed > authorized
            or (authorized - observed).total_seconds() > MAX_PROVIDER_ZERO_AGE_SECONDS
            or float(project["total_cost_usd"])
            + float(authority.get("hard_attempt_spend_cap_usd") or 0)
            > AGGREGATE_GOAL_SPEND_CAP_USD
        ):
            raise ValueError("budget or age mismatch")
    except (OSError, TypeError, ValueError):
        errors.append("authority_lineage_invalid")
    if errors:
        raise TaskEvaluationSceneConfigurationAuthorityError(
            "scene_configuration_paid_authority_invalid:"
            + ",".join(sorted(set(errors)))
        )
    return authority


__all__ = [
    "AUTHORITY_SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationAuthorityError",
    "materialize_scene_configuration_paid_authority",
    "validate_scene_configuration_paid_authority",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-receipt", required=True)
    parser.add_argument("--project-spend-reconciliation", required=True)
    parser.add_argument("--initial-provider-zero", required=True)
    parser.add_argument("--authorization-reference", required=True)
    parser.add_argument("--authorized-by", required=True)
    parser.add_argument("--authorized-on", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--resource-name", required=True)
    parser.add_argument("--max-hourly-rate-usd", required=True, type=float)
    parser.add_argument("--hard-cap-usd", required=True, type=float)
    parser.add_argument("--hard-ttl-seconds", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    authority = materialize_scene_configuration_paid_authority(
        bundle_receipt_path=args.bundle_receipt,
        project_spend_reconciliation_path=args.project_spend_reconciliation,
        initial_provider_zero_path=args.initial_provider_zero,
        authorization_reference=args.authorization_reference,
        authorized_by=args.authorized_by,
        authorized_on=args.authorized_on,
        source_commit=args.source_commit,
        container_image=args.container_image,
        resource_name=args.resource_name,
        max_hourly_rate_usd=args.max_hourly_rate_usd,
        hard_cap_usd=args.hard_cap_usd,
        hard_ttl_seconds=args.hard_ttl_seconds,
        output_path=args.output,
    )
    print(json.dumps(authority, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
