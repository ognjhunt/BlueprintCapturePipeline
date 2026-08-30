"""Officially billed OpenAI placement review for configured-controls autostart.

The deterministic CPU inventory remains the placement authority.  This seam
lets the production OpenAI Agents SDK reviewer select one exact inventory
member while reusing the already-provisioned scene visual-review credential.
It holds every canonical Vast launch slot during the short model call, so a
scene-configuration visual review cannot overlap the same official-cost scope.
"""

from __future__ import annotations

import fcntl
import json
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .openai_official_cost_gate import (
    OpenAIOfficialCostRunGate,
    build_openai_official_cost_run_gate,
)
from .task_evaluation_scene_configuration_openai_gate import (
    read_stage_scope_attestation,
    resolve_stage_scope_attestation,
)


PAID_RESOURCE_CLASS = "task_evaluation_configured_controls_robot_placement"
LANE_ID = PAID_RESOURCE_CLASS
PROVIDER_ID = "openai"
VISUAL_REVIEW_CREDENTIAL_ROLE = "artifixer_visual_review"
MAX_VAST_LAUNCH_SLOTS = 3


class TaskEvaluationConfiguredControlsOpenAIPlacementError(RuntimeError):
    """The live placement reviewer lacked exact spend or exclusion authority."""


def _required(environment: Mapping[str, str], name: str) -> str:
    value = str(environment.get(name) or "").strip()
    if not value:
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            f"configured_controls_openai_environment_missing:{name}"
        )
    return value


def _regular_file(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_required(environment, name)).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            f"configured_controls_openai_file_invalid:{name}"
        )
    return path.resolve()


def _lock_paths(environment: Mapping[str, str]) -> list[Path]:
    base = Path(_required(environment, "VAST_LAUNCH_LOCK_FILE")).expanduser()
    if not base.is_absolute() or base.is_symlink():
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            "configured_controls_openai_vast_lock_path_invalid"
        )
    try:
        slot_count = int(
            str(
                environment.get("BLUEPRINT_VAST_MAX_CONCURRENT_PAID_LAUNCHES")
                or str(MAX_VAST_LAUNCH_SLOTS)
            )
        )
    except ValueError as exc:
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            "configured_controls_openai_vast_slot_count_invalid"
        ) from exc
    if not 1 <= slot_count <= MAX_VAST_LAUNCH_SLOTS:
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            "configured_controls_openai_vast_slot_count_invalid"
        )
    result = [base]
    result.extend(
        base.with_name(f"{base.stem}.slot{index}{base.suffix}")
        for index in range(1, slot_count)
    )
    return result


@contextmanager
def exclusive_visual_review_cost_scope(
    *, environment: Mapping[str, str], output_root: str | Path
) -> Iterator[dict[str, Any]]:
    """Hold every Vast launch slot while the shared scene-review key is used."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    handles: list[Any] = []
    paths = _lock_paths(environment)
    acquired_at = utc_now_iso()
    acquired: dict[str, Any] | None = None
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
                    "configured_controls_openai_vast_lock_path_invalid"
                )
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
                    "configured_controls_openai_visual_review_scope_busy"
                ) from exc
            handles.append(handle)
        acquired = {
            "schema_version": (
                "task_evaluation_configured_controls_openai_scope_lock.v1"
            ),
            "status": "acquired",
            "credential_role": VISUAL_REVIEW_CREDENTIAL_ROLE,
            "paid_resource_class": PAID_RESOURCE_CLASS,
            "vast_launch_slots_held": [str(path.resolve()) for path in paths],
            "all_vast_launch_slots_held": True,
            "acquired_at": acquired_at,
            "raw_secret_values_recorded": False,
            "lock_receipt_digest": "",
        }
        acquired["lock_receipt_digest"] = canonical_digest(
            acquired, digest_field="lock_receipt_digest"
        )
        write_json(root / "openai_scope_lock_acquired.v1.json", acquired)
        yield acquired
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        if handles and acquired is not None:
            released: dict[str, Any] = {
                "schema_version": (
                    "task_evaluation_configured_controls_openai_scope_lock_release.v1"
                ),
                "status": "released",
                "acquisition_receipt_digest": acquired["lock_receipt_digest"],
                "credential_role": VISUAL_REVIEW_CREDENTIAL_ROLE,
                "paid_resource_class": PAID_RESOURCE_CLASS,
                "vast_launch_slots_released": [
                    str(path.resolve()) for path in paths
                ],
                "all_vast_launch_slots_released": True,
                "released_at": utc_now_iso(),
                "raw_secret_values_recorded": False,
                "release_receipt_digest": "",
            }
            released["release_receipt_digest"] = canonical_digest(
                released, digest_field="release_receipt_digest"
            )
            write_json(root / "openai_scope_lock_released.v1.json", released)


def configured_controls_robot_placement_openai_gate(
    *,
    environment: Mapping[str, str],
    placement_authority: Mapping[str, Any],
    run_id: str,
    request_digest: str,
    candidate_digest: str,
    authorization_receipt_digest: str,
    output_root: str | Path,
    transport: Callable[..., Mapping[str, Any]] | None = None,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> OpenAIOfficialCostRunGate:
    """Build the exact existing-key official-cost gate, without reserving."""

    authority = dict(placement_authority)
    project_id = _required(environment, "OPENAI_PROJECT_ID")
    api_key_id = _required(
        environment, "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID"
    )
    api_key_file = _regular_file(environment, "OPENAI_API_KEY_FILE")
    expected_key_file = _regular_file(
        environment, "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE"
    )
    admin_key_file = _regular_file(environment, "OPENAI_ADMIN_API_KEY_FILE")
    if api_key_file != expected_key_file:
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            "configured_controls_openai_credential_role_mismatch"
        )
    try:
        maximum_cost = float(authority.get("maximum_cost_usd"))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            "configured_controls_openai_authority_invalid"
        ) from exc
    if (
        set(authority)
        != {
            "provider_id",
            "credential_role",
            "project_id",
            "api_key_id",
            "paid_resource_class",
            "maximum_cost_usd",
        }
        or authority.get("provider_id") != PROVIDER_ID
        or authority.get("credential_role") != VISUAL_REVIEW_CREDENTIAL_ROLE
        or authority.get("project_id") != project_id
        or authority.get("api_key_id") != api_key_id
        or authority.get("paid_resource_class") != PAID_RESOURCE_CLASS
        or not math.isfinite(maximum_cost)
        or maximum_cost <= 0
    ):
        raise TaskEvaluationConfiguredControlsOpenAIPlacementError(
            "configured_controls_openai_authority_invalid"
        )
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_attestation_path = _regular_file(
        environment,
        "BLUEPRINT_OPENAI_ARTIFIXER_VISUAL_REVIEW_COST_SCOPE_ATTESTATION_FILE",
    )
    resolved_attestation = resolve_stage_scope_attestation(
        attestation=read_stage_scope_attestation(source_attestation_path),
        paid_resource_class=PAID_RESOURCE_CLASS,
        project_id=project_id,
        api_key_id=api_key_id,
        now=wall_clock(),
    )
    attestation_path = root / "openai_cost_scope_attestation_robot_placement.json"
    attestation_path.write_text(
        json.dumps(resolved_attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attestation_path.chmod(0o600)
    return build_openai_official_cost_run_gate(
        scope_attestation_path=attestation_path,
        admin_api_key_file=admin_key_file,
        project_id=project_id,
        api_key_id=api_key_id,
        lane_id=LANE_ID,
        run_id=run_id,
        request_digest=request_digest,
        candidate_digest=candidate_digest,
        authorization_receipt_digest=authorization_receipt_digest,
        max_cost_usd=maximum_cost,
        output_root=root,
        provider_id=PROVIDER_ID,
        paid_resource_class=PAID_RESOURCE_CLASS,
        transport=transport,
        wall_clock=wall_clock,
        require_zero_baseline=False,
    )


__all__ = [
    "LANE_ID",
    "PAID_RESOURCE_CLASS",
    "PROVIDER_ID",
    "VISUAL_REVIEW_CREDENTIAL_ROLE",
    "TaskEvaluationConfiguredControlsOpenAIPlacementError",
    "configured_controls_robot_placement_openai_gate",
    "exclusive_visual_review_cost_scope",
]
