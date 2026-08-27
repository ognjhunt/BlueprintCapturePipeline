"""Immutable production queue for Task Evaluation scene construction phases."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_disclosure import (
    RENDER_INPUT_STATUSES,
    render_inputs_disclosure_is_coherent,
)


ENVELOPE_SCHEMA_VERSION = "task_evaluation_scene_construction_envelope.v1"
FINALIZATION_SCHEMA_VERSION = "task_evaluation_scene_construction_finalization.v1"
QUEUE_STATES = ("pending", "processing", "completed", "blocked")


class TaskEvaluationSceneConstructionQueueError(ValueError):
    """A production scene-construction handoff could not be sealed safely."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def ensure_scene_construction_queue_root(queue_root: str | Path) -> Path:
    root = Path(queue_root).expanduser()
    if root.is_symlink():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_root_unsafe"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    root = root.resolve(strict=True)
    if not stat.S_ISDIR(root.stat().st_mode):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_root_unsafe"
        )
    for state in QUEUE_STATES:
        child = root / state
        if child.is_symlink():
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_state_unsafe"
            )
        child.mkdir(mode=0o750, exist_ok=True)
    return root


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(value)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o440,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable scene-construction queue write")
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
        if descriptor >= 0:
            os.close(descriptor)


def stage_scene_construction(
    *,
    request: Mapping[str, Any],
    preparation_result: Mapping[str, Any],
    recipe: Mapping[str, Any],
    recipe_configuration_references: Sequence[Mapping[str, Any]],
    render_inputs_result: Mapping[str, Any],
    queue_root: str | Path,
) -> dict[str, Any]:
    """Atomically continue one website-started run into production construction."""

    preparation_id = str(request.get("preparation_id") or "")
    run_id = str(request.get("run_id") or "")
    source_commit = str(request.get("expected_production_commit") or "")
    recipe_digest = str(recipe.get("recipe_digest") or "")
    if (
        not preparation_id
        or not run_id
        or len(source_commit) != 40
        or not recipe_digest.startswith("sha256:")
        or preparation_result.get("status")
        != "inputs_materialized_awaiting_construction_adapter"
        or preparation_result.get("full_byte_service_account_readback_passed")
        is not True
        or preparation_result.get("provider_mutation_performed") is not False
        or len(recipe_configuration_references)
        != len(recipe.get("stage_sequence") or [])
        or any(
            row.get("full_byte_service_account_readback_passed") is not True
            for row in recipe_configuration_references
        )
        or render_inputs_result.get("schema_version")
        != "task_evaluation_scene_configuration_render_inputs.v1"
        or render_inputs_result.get("status")
        not in RENDER_INPUT_STATUSES
        or render_inputs_result.get("run_id") != run_id
        or not render_inputs_disclosure_is_coherent(render_inputs_result)
        or render_inputs_result.get("provider_mutation_performed") is not False
        or render_inputs_result.get("paid_execution_requested") is not False
        or render_inputs_result.get("result_digest")
        != canonical_digest(render_inputs_result, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_handoff_binding_invalid"
        )
    envelope: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "orchestration_id": preparation_id,
        "preparation_id": preparation_id,
        "run_id": run_id,
        "team_namespace": request["team_namespace"],
        "expected_production_commit": source_commit,
        "construction_output_identity": recipe["output_identity"],
        "recipe_digest": recipe_digest,
        "preparation_result_digest": preparation_result["result_digest"],
        "request": dict(request),
        "recipe": dict(recipe),
        "materialized_references": [
            dict(row) for row in preparation_result.get("references") or []
        ],
        "stage_configuration_references": [
            dict(row) for row in recipe_configuration_references
        ],
        "render_inputs_result": dict(render_inputs_result),
        "stage_states": [
            {
                "stage_id": stage["stage_id"],
                "capability": stage["capability"],
                "execution_class": stage["execution_class"],
                "status": "pending",
            }
            for stage in recipe["stage_sequence"]
        ],
        "automatic_progression_required": True,
        "canonical_allocator_required_for_gpu_stages": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(
        envelope, digest_field="envelope_digest"
    )
    root = ensure_scene_construction_queue_root(queue_root)
    filename = (
        f"{preparation_id}-{recipe_digest.removeprefix('sha256:')}.json"
    )
    matches = [
        root / state / filename
        for state in QUEUE_STATES
        if (root / state / filename).exists()
    ]
    if matches:
        if len(matches) != 1 or matches[0].is_symlink():
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_identity_ambiguous"
            )
        try:
            existing = json.loads(matches[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_existing_envelope_invalid"
            ) from exc
        if existing != envelope:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_immutable_conflict"
            )
        created = False
        queue_path = matches[0]
    else:
        queue_path = root / "pending" / filename
        try:
            _write_exclusive(queue_path, envelope)
            created = True
        except FileExistsError:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_race_conflict"
            ) from None
    receipt: dict[str, Any] = {
        "schema_version": "task_evaluation_scene_construction_intake_receipt.v1",
        "status": "queued_for_production_construction",
        "orchestration_id": preparation_id,
        "run_id": run_id,
        "recipe_digest": recipe_digest,
        "envelope_digest": envelope["envelope_digest"],
        "queue_path": str(queue_path),
        "created": created,
        "automatic_progression_required": True,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def finalize_scene_construction(
    *,
    queue_root: str | Path,
    envelope: Mapping[str, Any],
    terminal_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Move one exact construction envelope to its immutable terminal state."""

    orchestration_id = str(envelope.get("orchestration_id") or "")
    recipe_digest = str(envelope.get("recipe_digest") or "")
    envelope_digest = str(envelope.get("control_plane_envelope_digest") or "")
    run_id = str(envelope.get("run_id") or "")
    source_commit = str(envelope.get("expected_production_commit") or "")
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", orchestration_id)
        is None
        or not run_id
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", recipe_digest) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", envelope_digest) is None
        or terminal_result.get("run_id") != run_id
        or terminal_result.get("source_commit") != source_commit
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_binding_invalid"
        )
    root = ensure_scene_construction_queue_root(queue_root)
    filename = f"{orchestration_id}-{recipe_digest.removeprefix('sha256:')}.json"
    matches = [
        root / state / filename
        for state in QUEUE_STATES
        if (root / state / filename).exists()
    ]
    if len(matches) != 1 or matches[0].is_symlink():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_identity_ambiguous"
        )
    source = matches[0]
    try:
        queued = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_envelope_invalid"
        ) from exc
    if (
        queued.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or queued.get("orchestration_id") != orchestration_id
        or queued.get("run_id") != run_id
        or queued.get("expected_production_commit") != source_commit
        or queued.get("recipe_digest") != recipe_digest
        or queued.get("envelope_digest") != envelope_digest
        or queued.get("envelope_digest")
        != canonical_digest(queued, digest_field="envelope_digest")
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_binding_invalid"
        )
    completed = (
        terminal_result.get("status") == "completed"
        and terminal_result.get("configuration_completed") is True
        and terminal_result.get("configured_scene_published") is True
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(terminal_result.get("configured_scene_revision_digest") or ""),
        )
        is not None
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(terminal_result.get("publication_result_digest") or ""),
        )
        is not None
        and terminal_result.get("full_byte_service_account_readback_passed") is True
        and terminal_result.get("continuing_spend_from_this_run") is False
        and not terminal_result.get("blockers")
    )
    terminal_state = "completed" if completed else "blocked"
    target = root / terminal_state / filename
    results_root = root / "results"
    if results_root.is_symlink():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_results_unsafe"
        )
    results_root.mkdir(mode=0o750, exist_ok=True)
    result_path = results_root / filename
    finalization: dict[str, Any] = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "status": "completed" if completed else "blocked",
        "queue_state": terminal_state,
        "orchestration_id": orchestration_id,
        "run_id": run_id,
        "source_commit": source_commit,
        "recipe_digest": recipe_digest,
        "construction_envelope_digest": envelope_digest,
        "configuration_completed": terminal_result.get("configuration_completed")
        is True,
        "configured_scene_published": terminal_result.get(
            "configured_scene_published"
        )
        is True,
        "configured_scene_revision_digest": terminal_result.get(
            "configured_scene_revision_digest"
        ),
        "publication_result_digest": terminal_result.get(
            "publication_result_digest"
        ),
        "full_byte_service_account_readback_passed": terminal_result.get(
            "full_byte_service_account_readback_passed"
        )
        is True,
        "continuing_spend_from_this_run": terminal_result.get(
            "continuing_spend_from_this_run"
        ),
        "finalization_performed": True,
        "queue_path": str(target),
        "result_path": str(result_path),
        "blockers": sorted(
            set(str(item) for item in terminal_result.get("blockers") or [] if str(item))
        ),
        "result_digest": "",
    }
    finalization["result_digest"] = canonical_digest(
        finalization, digest_field="result_digest"
    )
    try:
        _write_exclusive(result_path, finalization)
    except FileExistsError:
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_finalization_result_conflict"
            ) from exc
        if existing != finalization:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_finalization_result_conflict"
            )
    if source != target:
        try:
            os.replace(source, target)
        except FileNotFoundError:
            if not target.is_file():
                raise TaskEvaluationSceneConstructionQueueError(
                    "scene_construction_queue_finalization_race"
                ) from None
    return finalization


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "FINALIZATION_SCHEMA_VERSION",
    "TaskEvaluationSceneConstructionQueueError",
    "ensure_scene_construction_queue_root",
    "finalize_scene_construction",
    "stage_scene_construction",
]
