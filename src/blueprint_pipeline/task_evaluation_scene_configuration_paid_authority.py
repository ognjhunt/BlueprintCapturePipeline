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
from .native_task_isaaclab_launch import (
    NATIVE_TASK_ARENA_IMAGE as SCENE_CONFIGURATION_PROVIDER_IMAGE,
)
from .project_spend_reconciliation import validate_project_spend_reconciliation
from .task_evaluation_scene_configuration_bundle import (
    load_scene_configuration_provider_bundle_receipt,
)
from .task_evaluation_scene_configuration_disclosure import renders_on_provider
from .task_evaluation_scene_configuration_diagnostic_mode import (
    FRESH_DIAGNOSTIC_BOOTSTRAP_MODE,
)
from .task_evaluation_scene_configuration_runtime_budget import (
    MAX_ATTEMPT_SPEND_USD,
    MAX_EXTERNAL_SERVICE_SPEND_USD,
    MAX_HOURLY_RATE_USD,
    MAX_PROVIDER_COMPUTE_SPEND_USD,
    MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD,
    MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD,
    MIN_CONTENT_AGENTS_SPEND_USD,
    REQUIRED_PARENT_TTL_SECONDS,
    diagnostic_parent_runtime_budget_blockers,
)


AUTHORITY_SCHEMA_VERSION = "task_evaluation_scene_configuration_paid_authority.v1"
MAX_TTL_SECONDS = REQUIRED_PARENT_TTL_SECONDS
MIN_TTL_SECONDS = 600
MAX_PROVIDER_ZERO_AGE_SECONDS = 900
_RESOURCE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{15,127}")


class TaskEvaluationSceneConfigurationAuthorityError(ValueError):
    """A configuration authority was incomplete, stale, or not single-use."""


def _required_external_stage_minima(
    *,
    diagnostic_only: bool,
    diagnostic_bootstrap_mode: object,
    carried_stage_count: int,
) -> dict[str, float]:
    fresh_diagnostic_bootstrap = (
        diagnostic_only
        and diagnostic_bootstrap_mode == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
    )
    return {
        "artifixer_semantic_teacher": (
            0.0
            if diagnostic_only and not fresh_diagnostic_bootstrap
            else MIN_ARTIFIXER_SEMANTIC_TEACHER_SPEND_USD
        ),
        "artifixer_visual_review": (
            0.0
            if diagnostic_only and carried_stage_count >= 1
            else MIN_ARTIFIXER_VISUAL_REVIEW_SPEND_USD
        ),
        "content_agents": (
            0.0
            if diagnostic_only and carried_stage_count >= 3
            else MIN_CONTENT_AGENTS_SPEND_USD
        ),
    }


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
        and 0 < cap_value <= MAX_PROVIDER_COMPUTE_SPEND_USD
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
    provider_compute_spend_cap_usd: float | None = None,
    openai_max_cost_usd: float = 0.0,
    openai_max_requests: int = 0,
    openai_artifixer_semantic_teacher_max_cost_usd: float = 0.0,
    openai_artifixer_visual_review_max_cost_usd: float = 0.0,
    openai_content_agents_max_cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Seal one fresh project-spend-derived authority; retries are impossible."""

    receipt_path = Path(bundle_receipt_path).expanduser().resolve()
    receipt_value = _read(
        receipt_path, code="scene_configuration_bundle_receipt_invalid"
    )
    diagnostic_only = receipt_value.get("diagnostic_only") is True
    receipt = load_scene_configuration_provider_bundle_receipt(
        receipt_path,
        expected_source_commit=source_commit,
        diagnostic_only=diagnostic_only,
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
    compute_cap = (
        float(provider_compute_spend_cap_usd)
        if provider_compute_spend_cap_usd is not None
        else float(hard_cap_usd)
    )
    external_cap = float(openai_max_cost_usd)
    stage_caps = {
        "artifixer_semantic_teacher": float(
            openai_artifixer_semantic_teacher_max_cost_usd
        ),
        "artifixer_visual_review": float(
            openai_artifixer_visual_review_max_cost_usd
        ),
        "content_agents": float(openai_content_agents_max_cost_usd),
    }
    carried_stage_count = int(receipt.get("carried_completed_stage_count") or 0)
    diagnostic_bootstrap_mode = receipt.get("diagnostic_bootstrap_mode")
    fresh_diagnostic_bootstrap = (
        diagnostic_only
        and diagnostic_bootstrap_mode == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
    )
    required_stage_minima = _required_external_stage_minima(
        diagnostic_only=diagnostic_only,
        diagnostic_bootstrap_mode=diagnostic_bootstrap_mode,
        carried_stage_count=carried_stage_count,
    )
    minimum_external_cap = sum(required_stage_minima.values())
    diagnostic_budget_blockers = (
        diagnostic_parent_runtime_budget_blockers(
            completed_stage_prefix_count=carried_stage_count,
            ttl_seconds=hard_ttl_seconds,
            maximum_hourly_rate_usd=max_hourly_rate_usd,
            provider_compute_spend_cap_usd=compute_cap,
        )
        if diagnostic_only
        else []
    )
    external_contract_valid = (
        math.isfinite(external_cap)
        and minimum_external_cap <= external_cap <= MAX_EXTERNAL_SERVICE_SPEND_USD
        and all(math.isfinite(value) and value >= 0 for value in stage_caps.values())
        and all(
            stage_caps[name] >= required_minimum
            for name, required_minimum in required_stage_minima.items()
        )
        and (
            not diagnostic_only
            or fresh_diagnostic_bootstrap
            or stage_caps["artifixer_semantic_teacher"] == 0
        )
        and (
            not diagnostic_only
            or carried_stage_count < 1
            or stage_caps["artifixer_visual_review"] == 0
        )
        and (
            not diagnostic_only
            or carried_stage_count < 3
            or stage_caps["content_agents"] == 0
        )
        and sum(stage_caps.values()) <= external_cap + 1e-9
        and isinstance(openai_max_requests, int)
        and not isinstance(openai_max_requests, bool)
        and (
            (external_cap == 0 and openai_max_requests == 0)
            or (external_cap > 0 and 1 <= openai_max_requests <= 100)
        )
        and (
            (diagnostic_only and 0 < compute_cap <= MAX_PROVIDER_COMPUTE_SPEND_USD)
            or (
                not diagnostic_only
                and abs(compute_cap - MAX_PROVIDER_COMPUTE_SPEND_USD) <= 1e-9
            )
        )
        and (
            (
                diagnostic_only
                and abs(float(hard_cap_usd) - (compute_cap + external_cap))
                <= 1e-9
            )
            or (
                not diagnostic_only
                and abs(float(hard_cap_usd) - MAX_ATTEMPT_SPEND_USD) <= 1e-9
            )
        )
        and compute_cap + external_cap <= float(hard_cap_usd) + 1e-9
    )
    raw_source_authorized = (
        receipt.get("raw_interiorgs_bytes_in_provider_bundle") is True
    )
    disclosure_decision = receipt.get("disclosure_decision")
    disclosure_decision_digest = (
        disclosure_decision.get("decision_digest")
        if isinstance(disclosure_decision, Mapping)
        else None
    )
    if raw_source_authorized and not renders_on_provider(
        disclosure_decision or {}
    ):
        raise TaskEvaluationSceneConfigurationAuthorityError(
            "scene_configuration_authority_provider_disclosure_invalid"
        )
    if container_image != SCENE_CONFIGURATION_PROVIDER_IMAGE:
        raise TaskEvaluationSceneConfigurationAuthorityError(
            "scene_configuration_authority_container_image_invalid"
        )
    if (
        not authorization_reference.strip()
        or not authorized_by.strip()
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or _RESOURCE_NAME.fullmatch(resource_name) is None
        or not _budget_valid(
            rate=max_hourly_rate_usd, cap=compute_cap, ttl=hard_ttl_seconds
        )
        or diagnostic_budget_blockers
        or not external_contract_valid
        or zero_time > authorized_time
        or (authorized_time - zero_time).total_seconds()
        > MAX_PROVIDER_ZERO_AGE_SECONDS
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
        "purpose": (
            (
                "one_shot_task_evaluation_scene_configuration_diagnostic_fresh_bootstrap"
                if fresh_diagnostic_bootstrap
                else "one_shot_task_evaluation_scene_configuration_diagnostic_resume"
            )
            if diagnostic_only
            else "one_shot_task_evaluation_scene_configuration"
        ),
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
        "provider_compute_spend_cap_usd": compute_cap,
        "external_service_spend_caps": {
            "openai": {
                "maximum_cost_usd": external_cap,
                "maximum_requests": openai_max_requests,
                "stage_max_cost_usd": stage_caps,
                "credentials_via_ephemeral_private_file_only": True,
            }
        },
        "maximum_hourly_rate_usd": max_hourly_rate_usd,
        "maximum_single_resource_ttl_seconds": hard_ttl_seconds,
        "aggregate_goal_spend_before_attempt_usd": project_total,
        "project_spend_reconciliation": project_record,
        "initial_provider_zero": {
            **_record(zero_path),
            "provider_zero_digest": zero["provider_zero_digest"],
        },
        "active_instance_allowlist": {
            "external_provider_owned": [],
            "same_goal_concurrent": [],
        },
        "raw_interiorgs_bytes_authorized_for_provider": raw_source_authorized,
        "provider_disclosure_decision_digest": disclosure_decision_digest,
        "evaluation_episode_authorized": False,
        "authority_digest": "",
    }
    if diagnostic_only:
        authority.update(
            {
                "diagnostic_only": True,
                "qualification_eligible": False,
                "configured_revision_publication_permitted": False,
                "offering_publication_permitted": False,
                "terminal_e2e_completion_permitted": False,
                "source_diagnostic_checkpoint_digest": receipt.get(
                    "source_diagnostic_checkpoint_digest"
                ),
                "carried_completed_stage_count": carried_stage_count,
                "diagnostic_bootstrap_mode": diagnostic_bootstrap_mode,
                "diagnostic_scientific_binding_digest": receipt.get(
                    "diagnostic_scientific_binding_digest"
                ),
                "diagnostic_stage_sequence_ids": list(
                    receipt.get("diagnostic_stage_sequence_ids") or []
                ),
            }
        )
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
    external = (authority.get("external_service_spend_caps") or {}).get("openai")
    compute_cap = authority.get("provider_compute_spend_cap_usd")
    total_cap = authority.get("hard_attempt_spend_cap_usd")
    external_cost = external.get("maximum_cost_usd") if isinstance(external, Mapping) else None
    external_requests = external.get("maximum_requests") if isinstance(external, Mapping) else None
    stage_caps = external.get("stage_max_cost_usd") if isinstance(external, Mapping) else None
    diagnostic_only = bundle_receipt.get("diagnostic_only") is True
    diagnostic_bootstrap_mode = bundle_receipt.get("diagnostic_bootstrap_mode")
    fresh_diagnostic_bootstrap = (
        diagnostic_only
        and diagnostic_bootstrap_mode == FRESH_DIAGNOSTIC_BOOTSTRAP_MODE
    )
    carried_stage_count = int(
        bundle_receipt.get("carried_completed_stage_count") or 0
    )
    required_stage_minima = _required_external_stage_minima(
        diagnostic_only=diagnostic_only,
        diagnostic_bootstrap_mode=diagnostic_bootstrap_mode,
        carried_stage_count=carried_stage_count,
    )
    minimum_external_cap = sum(required_stage_minima.values())
    diagnostic_budget_blockers = (
        diagnostic_parent_runtime_budget_blockers(
            completed_stage_prefix_count=carried_stage_count,
            ttl_seconds=authority.get("maximum_single_resource_ttl_seconds"),
            maximum_hourly_rate_usd=authority.get("maximum_hourly_rate_usd"),
            provider_compute_spend_cap_usd=compute_cap,
        )
        if diagnostic_only
        else []
    )
    external_contract_valid = (
        isinstance(compute_cap, (int, float))
        and not isinstance(compute_cap, bool)
        and math.isfinite(float(compute_cap))
        and isinstance(total_cap, (int, float))
        and not isinstance(total_cap, bool)
        and isinstance(external_cost, (int, float))
        and not isinstance(external_cost, bool)
        and math.isfinite(float(external_cost))
        and minimum_external_cap
        <= float(external_cost)
        <= MAX_EXTERNAL_SERVICE_SPEND_USD
        and isinstance(external_requests, int)
        and not isinstance(external_requests, bool)
        and (
            (float(external_cost) == 0 and external_requests == 0)
            or (float(external_cost) > 0 and 1 <= external_requests <= 100)
        )
        and isinstance(stage_caps, Mapping)
        and set(stage_caps)
        == {
            "artifixer_semantic_teacher",
            "artifixer_visual_review",
            "content_agents",
        }
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
            for value in stage_caps.values()
        )
        and all(
            float(stage_caps[name]) >= required_minimum
            for name, required_minimum in required_stage_minima.items()
        )
        and (
            not diagnostic_only
            or fresh_diagnostic_bootstrap
            or float(stage_caps["artifixer_semantic_teacher"]) == 0
        )
        and (
            not diagnostic_only
            or carried_stage_count < 1
            or float(stage_caps["artifixer_visual_review"]) == 0
        )
        and (
            not diagnostic_only
            or carried_stage_count < 3
            or float(stage_caps["content_agents"]) == 0
        )
        and sum(float(value) for value in stage_caps.values())
        <= float(external_cost) + 1e-9
        and (
            (
                diagnostic_only
                and 0 < float(compute_cap) <= MAX_PROVIDER_COMPUTE_SPEND_USD
            )
            or (
                not diagnostic_only
                and abs(float(compute_cap) - MAX_PROVIDER_COMPUTE_SPEND_USD)
                <= 1e-9
            )
        )
        and (
            (
                diagnostic_only
                and abs(float(total_cap) - (float(compute_cap) + float(external_cost)))
                <= 1e-9
            )
            or (
                not diagnostic_only
                and abs(float(total_cap) - MAX_ATTEMPT_SPEND_USD) <= 1e-9
            )
        )
        and float(compute_cap) + float(external_cost)
        <= float(total_cap) + 1e-9
        and external.get("credentials_via_ephemeral_private_file_only") is True
    )
    expected_raw_source_authorized = (
        bundle_receipt.get("raw_interiorgs_bytes_in_provider_bundle") is True
    )
    disclosure_decision = bundle_receipt.get("disclosure_decision")
    expected_disclosure_decision_digest = (
        disclosure_decision.get("decision_digest")
        if isinstance(disclosure_decision, Mapping)
        else None
    )
    if (
        authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or authority.get("authority_kind")
        != "explicit_user_direction_in_current_goal"
        or authority.get("purpose")
        != (
            (
                "one_shot_task_evaluation_scene_configuration_diagnostic_fresh_bootstrap"
                if fresh_diagnostic_bootstrap
                else "one_shot_task_evaluation_scene_configuration_diagnostic_resume"
            )
            if diagnostic_only
            else "one_shot_task_evaluation_scene_configuration"
        )
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
        or authority.get("container_image") != SCENE_CONFIGURATION_PROVIDER_IMAGE
        or _RESOURCE_NAME.fullmatch(str(authority.get("resource_name") or "")) is None
        or not _budget_valid(
            rate=authority.get("maximum_hourly_rate_usd"),
            cap=compute_cap,
            ttl=authority.get("maximum_single_resource_ttl_seconds"),
        )
        or not external_contract_valid
        or diagnostic_budget_blockers
        or authority.get("raw_interiorgs_bytes_authorized_for_provider")
        is not expected_raw_source_authorized
        or authority.get("provider_disclosure_decision_digest")
        != expected_disclosure_decision_digest
        or (
            expected_raw_source_authorized
            and not renders_on_provider(disclosure_decision or {})
        )
        or authority.get("evaluation_episode_authorized") is not False
        or (
            diagnostic_only
            and (
                authority.get("diagnostic_only") is not True
                or authority.get("qualification_eligible") is not False
                or authority.get("configured_revision_publication_permitted")
                is not False
                or authority.get("offering_publication_permitted") is not False
                or authority.get("terminal_e2e_completion_permitted") is not False
                or authority.get("source_diagnostic_checkpoint_digest")
                != bundle_receipt.get("source_diagnostic_checkpoint_digest")
                or authority.get("carried_completed_stage_count")
                != carried_stage_count
                or authority.get("diagnostic_bootstrap_mode")
                != diagnostic_bootstrap_mode
                or authority.get("diagnostic_scientific_binding_digest")
                != bundle_receipt.get("diagnostic_scientific_binding_digest")
                or authority.get("diagnostic_stage_sequence_ids")
                != bundle_receipt.get("diagnostic_stage_sequence_ids")
            )
        )
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
    "MAX_PROVIDER_COMPUTE_SPEND_USD",
    "SCENE_CONFIGURATION_PROVIDER_IMAGE",
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
    parser.add_argument("--provider-compute-spend-cap-usd", type=float)
    parser.add_argument("--openai-max-cost-usd", type=float, default=0.0)
    parser.add_argument("--openai-max-requests", type=int, default=0)
    parser.add_argument(
        "--openai-artifixer-semantic-teacher-max-cost-usd", type=float, default=0.0
    )
    parser.add_argument(
        "--openai-artifixer-visual-review-max-cost-usd", type=float, default=0.0
    )
    parser.add_argument("--openai-content-agents-max-cost-usd", type=float, default=0.0)
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
        provider_compute_spend_cap_usd=args.provider_compute_spend_cap_usd,
        openai_max_cost_usd=args.openai_max_cost_usd,
        openai_max_requests=args.openai_max_requests,
        openai_artifixer_semantic_teacher_max_cost_usd=(
            args.openai_artifixer_semantic_teacher_max_cost_usd
        ),
        openai_artifixer_visual_review_max_cost_usd=(
            args.openai_artifixer_visual_review_max_cost_usd
        ),
        openai_content_agents_max_cost_usd=args.openai_content_agents_max_cost_usd,
    )
    print(json.dumps(authority, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
