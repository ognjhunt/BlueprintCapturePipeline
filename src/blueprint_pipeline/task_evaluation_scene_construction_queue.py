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
from .task_evaluation_release_reference_lock import release_reference_lock


ENVELOPE_SCHEMA_VERSION = "task_evaluation_scene_construction_envelope.v1"
FINALIZATION_SCHEMA_VERSION = "task_evaluation_scene_construction_finalization.v1"
REVISION_LINEAGE_SCHEMA_VERSION = (
    "task_evaluation_scene_configuration_revision_lineage.v1"
)
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


def _write_exclusive_locked(path: Path, value: Mapping[str, Any]) -> None:
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
        # Preparation and corrective-revision minting may run as root while
        # the production queue consumer runs as the directory's service
        # group.  A plain 0440 create inherits the writer's effective group,
        # which made root-authored envelopes unreadable by the dispatcher.
        # Bind the immutable file to the queue state's already-authoritative
        # group before publishing it.
        parent_gid = path.parent.stat().st_gid
        os.fchown(descriptor, -1, parent_gid)
        os.fchmod(descriptor, 0o440)
        metadata = os.fstat(descriptor)
        if (
            metadata.st_gid != parent_gid
            or stat.S_IMODE(metadata.st_mode) != 0o440
        ):
            raise OSError("scene-construction queue ownership install failed")
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


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with release_reference_lock(path.parents[2], exclusive=False):
        _write_exclusive_locked(path, value)


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


def stage_scene_configuration_revision(
    *,
    queue_root: str | Path,
    source_envelope: Mapping[str, Any],
    expected_production_commit: str,
    revision_id: str,
    semantic_checkpoint_digest: str,
) -> dict[str, Any]:
    """Stage a fresh queue identity derived from one completed construction.

    A corrective configuration is a new immutable attempt, not a replay of the
    queue item that produced an earlier configured revision.  Reusing that
    terminal item would either mutate historical state or make finalization
    collide with its existing result.  This derivation keeps the captured scene,
    recipe, and materialized references byte-bound while giving the new run its
    own pending -> terminal lifecycle and current runtime commit.
    """

    source_orchestration_id = str(source_envelope.get("orchestration_id") or "")
    source_run_id = str(source_envelope.get("run_id") or "")
    source_commit = str(source_envelope.get("expected_production_commit") or "")
    recipe_digest = str(source_envelope.get("recipe_digest") or "")
    source_digest = str(source_envelope.get("envelope_digest") or "")
    if (
        source_envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", source_orchestration_id)
        is None
        or not source_run_id
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", expected_production_commit) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", recipe_digest) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", source_digest) is None
        or source_digest
        != canonical_digest(source_envelope, digest_field="envelope_digest")
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", revision_id)
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", semantic_checkpoint_digest
        )
        is None
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_configuration_revision_source_binding_invalid"
        )

    root = ensure_scene_construction_queue_root(queue_root)
    source_filename = (
        f"{source_orchestration_id}-{recipe_digest.removeprefix('sha256:')}.json"
    )
    source_matches = [
        root / state / source_filename
        for state in QUEUE_STATES
        if (root / state / source_filename).exists()
    ]
    source_path = root / "completed" / source_filename
    source_result_path = root / "results" / source_filename
    try:
        queued_source = json.loads(source_path.read_text(encoding="utf-8"))
        source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_configuration_revision_source_not_completed"
        ) from exc
    if (
        len(source_matches) != 1
        or source_matches[0] != source_path
        or source_path.is_symlink()
        or source_result_path.is_symlink()
        or queued_source != dict(source_envelope)
        or source_result.get("schema_version") != FINALIZATION_SCHEMA_VERSION
        or source_result.get("status") != "completed"
        or source_result.get("queue_state") != "completed"
        or source_result.get("finalization_performed") is not True
        or source_result.get("orchestration_id") != source_orchestration_id
        or source_result.get("run_id") != source_run_id
        or source_result.get("source_commit") != source_commit
        or source_result.get("recipe_digest") != recipe_digest
        or source_result.get("construction_envelope_digest") != source_digest
        or source_result.get("configuration_completed") is not True
        or source_result.get("configured_scene_published") is not True
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(source_result.get("configured_scene_revision_digest") or ""),
        )
        is None
        or source_result.get("full_byte_service_account_readback_passed")
        is not True
        or source_result.get("continuing_spend_from_this_run") is not False
        or bool(source_result.get("blockers"))
        or source_result.get("result_digest")
        != canonical_digest(source_result, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_configuration_revision_source_not_completed"
        )

    identity_binding = {
        "source_construction_envelope_digest": source_digest,
        "source_configuration_result_digest": source_result["result_digest"],
        "expected_production_commit": expected_production_commit,
        "revision_id": revision_id,
        "semantic_checkpoint_digest": semantic_checkpoint_digest,
    }
    identity_digest = canonical_digest(identity_binding)
    suffix = f"-revision-{revision_id}-{identity_digest.removeprefix('sha256:')[:12]}"
    if len(suffix) >= 192:
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_configuration_revision_identity_invalid"
        )
    orchestration_id = source_orchestration_id[: 192 - len(suffix)] + suffix
    run_id = source_run_id[: 240 - len(suffix)] + suffix
    lineage: dict[str, Any] = {
        "schema_version": REVISION_LINEAGE_SCHEMA_VERSION,
        "revision_id": revision_id,
        "source_orchestration_id": source_orchestration_id,
        "source_run_id": source_run_id,
        "source_production_commit": source_commit,
        "source_construction_envelope_digest": source_digest,
        "source_configuration_result_digest": source_result["result_digest"],
        "source_configured_scene_revision_digest": source_result.get(
            "configured_scene_revision_digest"
        ),
        "semantic_checkpoint_digest": semantic_checkpoint_digest,
        "lineage_digest": "",
    }
    lineage["lineage_digest"] = canonical_digest(
        lineage, digest_field="lineage_digest"
    )
    derived = json.loads(json.dumps(source_envelope))
    derived.update(
        {
            "orchestration_id": orchestration_id,
            "preparation_id": orchestration_id,
            "run_id": run_id,
            "expected_production_commit": expected_production_commit,
            "configuration_revision_lineage": lineage,
            "stage_states": [
                {**dict(row), "status": "pending"}
                for row in source_envelope.get("stage_states") or []
            ],
            "provider_mutation_performed": False,
            "paid_execution_requested": False,
            "envelope_digest": "",
        }
    )
    request = derived.get("request")
    if isinstance(request, Mapping):
        derived["request"] = {
            **dict(request),
            "preparation_id": orchestration_id,
            "run_id": run_id,
            "expected_production_commit": expected_production_commit,
        }
    derived.pop("control_plane_envelope_digest", None)
    derived["envelope_digest"] = canonical_digest(
        derived, digest_field="envelope_digest"
    )

    filename = f"{orchestration_id}-{recipe_digest.removeprefix('sha256:')}.json"
    matches = [
        root / state / filename
        for state in QUEUE_STATES
        if (root / state / filename).exists()
    ]
    target = root / "pending" / filename
    created = False
    if matches:
        if len(matches) != 1 or matches[0] != target or target.is_symlink():
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_configuration_revision_identity_already_terminal"
            )
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_configuration_revision_existing_envelope_invalid"
            ) from exc
        if existing != derived:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_configuration_revision_immutable_conflict"
            )
    else:
        try:
            _write_exclusive(target, derived)
            created = True
        except FileExistsError:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_configuration_revision_race_conflict"
            ) from None

    receipt: dict[str, Any] = {
        "schema_version": "task_evaluation_scene_configuration_revision_intake.v1",
        "status": "queued_for_production_configuration_revision",
        "revision_id": revision_id,
        "source_construction_envelope_digest": source_digest,
        "source_configuration_result_digest": source_result["result_digest"],
        "semantic_checkpoint_digest": semantic_checkpoint_digest,
        "run_id": run_id,
        "expected_production_commit": expected_production_commit,
        "envelope_digest": derived["envelope_digest"],
        "queue_path": str(target),
        "created": created,
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


def recover_scene_construction_publication(
    *,
    queue_root: str | Path,
    envelope: Mapping[str, Any],
    terminal_result: Mapping[str, Any],
    prior_finalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote one publication-only failure without rewriting its first result.

    The provider execution and its original blocked finalization remain immutable.
    Recovery is permitted only after configuration completed, spend stopped, and
    publication subsequently produced a fully qualified completed result.
    """

    orchestration_id = str(envelope.get("orchestration_id") or "")
    recipe_digest = str(envelope.get("recipe_digest") or "")
    envelope_digest = str(envelope.get("control_plane_envelope_digest") or "")
    run_id = str(envelope.get("run_id") or "")
    source_commit = str(envelope.get("expected_production_commit") or "")
    prior_digest = str(prior_finalization.get("result_digest") or "")
    prior_blockers = [
        str(item) for item in prior_finalization.get("blockers") or [] if str(item)
    ]
    publication_only_blockers = bool(prior_blockers) and all(
        item == "scene_configuration_configured_revision_not_published"
        or item.startswith(
            "scene_configuration_configured_revision_publication_failed:"
        )
        for item in prior_blockers
    )
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}", orchestration_id)
        is None
        or not run_id
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", recipe_digest) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", envelope_digest) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", prior_digest) is None
        or prior_finalization.get("schema_version") != FINALIZATION_SCHEMA_VERSION
        or prior_finalization.get("status") != "blocked"
        or prior_finalization.get("queue_state") != "blocked"
        or prior_finalization.get("finalization_performed") is not True
        or prior_finalization.get("orchestration_id") != orchestration_id
        or prior_finalization.get("run_id") != run_id
        or prior_finalization.get("source_commit") != source_commit
        or prior_finalization.get("recipe_digest") != recipe_digest
        or prior_finalization.get("construction_envelope_digest") != envelope_digest
        or prior_digest
        != canonical_digest(prior_finalization, digest_field="result_digest")
        or prior_finalization.get("configuration_completed") is not True
        or prior_finalization.get("configured_scene_published") is not False
        or prior_finalization.get("continuing_spend_from_this_run") is not False
        or not publication_only_blockers
        or terminal_result.get("run_id") != run_id
        or terminal_result.get("source_commit") != source_commit
        or terminal_result.get("status") != "completed"
        or terminal_result.get("configuration_completed") is not True
        or terminal_result.get("configured_scene_published") is not True
        or terminal_result.get("full_byte_service_account_readback_passed") is not True
        or terminal_result.get("continuing_spend_from_this_run") is not False
        or terminal_result.get("blockers")
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(terminal_result.get("configured_scene_revision_digest") or ""),
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(terminal_result.get("publication_result_digest") or ""),
        )
        is None
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_publication_recovery_binding_invalid"
        )

    root = ensure_scene_construction_queue_root(queue_root)
    filename = f"{orchestration_id}-{recipe_digest.removeprefix('sha256:')}.json"
    blocked_path = root / "blocked" / filename
    completed_path = root / "completed" / filename
    matches = [path for path in (blocked_path, completed_path) if path.exists()]
    if len(matches) != 1 or matches[0].is_symlink():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_publication_recovery_identity_ambiguous"
        )
    source = matches[0]
    try:
        queued = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_publication_recovery_envelope_invalid"
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
            "scene_construction_publication_recovery_binding_invalid"
        )

    original_result_path = root / "results" / filename
    try:
        original_result = json.loads(original_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_publication_recovery_prior_result_invalid"
        ) from exc
    if original_result != dict(prior_finalization):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_publication_recovery_prior_result_invalid"
        )

    recoveries_root = root / "publication-recoveries"
    if recoveries_root.is_symlink():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_publication_recovery_results_unsafe"
        )
    recoveries_root.mkdir(mode=0o750, exist_ok=True)
    recovery_path = recoveries_root / filename
    finalization: dict[str, Any] = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "status": "completed",
        "queue_state": "completed",
        "orchestration_id": orchestration_id,
        "run_id": run_id,
        "source_commit": source_commit,
        "recipe_digest": recipe_digest,
        "construction_envelope_digest": envelope_digest,
        "configuration_completed": True,
        "configured_scene_published": True,
        "configured_scene_revision_digest": terminal_result[
            "configured_scene_revision_digest"
        ],
        "publication_result_digest": terminal_result["publication_result_digest"],
        "full_byte_service_account_readback_passed": True,
        "continuing_spend_from_this_run": False,
        "finalization_performed": True,
        "queue_path": str(completed_path),
        "result_path": str(recovery_path),
        "blockers": [],
        "publication_recovery": {
            "performed": True,
            "provider_execution_repeated": False,
            "prior_finalization_digest": prior_digest,
            "prior_result_path": str(original_result_path),
        },
        "result_digest": "",
    }
    finalization["result_digest"] = canonical_digest(
        finalization, digest_field="result_digest"
    )
    try:
        _write_exclusive(recovery_path, finalization)
    except FileExistsError:
        try:
            existing = json.loads(recovery_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_publication_recovery_result_conflict"
            ) from exc
        if existing != finalization:
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_publication_recovery_result_conflict"
            )
    if source != completed_path:
        try:
            os.replace(source, completed_path)
        except FileNotFoundError:
            if not completed_path.is_file():
                raise TaskEvaluationSceneConstructionQueueError(
                    "scene_construction_publication_recovery_race"
                ) from None
    return finalization


def preflight_scene_construction_finalization(
    *,
    queue_root: str | Path,
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the exact live queue item can reach either terminal state.

    This check is intentionally read-only.  Preparation owns provisioning the
    queue tree; the paid caller must not discover a missing, ambiguous, or
    unwritable finalization target only after renting a provider.
    """

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
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_binding_invalid"
        )

    candidate_root = Path(queue_root).expanduser()
    if candidate_root.is_symlink():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_root_unsafe"
        )
    try:
        root = candidate_root.resolve(strict=True)
    except OSError as exc:
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_root_unsafe"
        ) from exc
    if not stat.S_ISDIR(root.stat().st_mode):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_root_unsafe"
        )

    state_roots: dict[str, Path] = {}
    for state in QUEUE_STATES:
        child = root / state
        if child.is_symlink() or not child.is_dir():
            raise TaskEvaluationSceneConstructionQueueError(
                "scene_construction_queue_state_unsafe"
            )
        state_roots[state] = child

    filename = f"{orchestration_id}-{recipe_digest.removeprefix('sha256:')}.json"
    matches = [
        state_roots[state] / filename
        for state in QUEUE_STATES
        if (state_roots[state] / filename).exists()
    ]
    if len(matches) != 1 or matches[0].is_symlink():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_identity_ambiguous"
        )
    source = matches[0]
    if source.parent.name not in {"pending", "processing"}:
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_state_invalid"
        )
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

    results_root = root / "results"
    if results_root.is_symlink() or (
        results_root.exists() and not results_root.is_dir()
    ):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_results_unsafe"
        )
    writable_roots = [
        source.parent,
        state_roots["completed"],
        state_roots["blocked"],
        results_root if results_root.exists() else root,
    ]
    if any(not os.access(path, os.W_OK | os.X_OK) for path in writable_roots):
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_destination_unwritable"
        )
    if (results_root / filename).exists():
        raise TaskEvaluationSceneConstructionQueueError(
            "scene_construction_queue_finalization_result_conflict"
        )
    return {
        "status": "ready",
        "run_id": run_id,
        "source_commit": source_commit,
        "construction_envelope_digest": envelope_digest,
        "queue_path": str(source),
        "result_path": str(results_root / filename),
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
    }


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "FINALIZATION_SCHEMA_VERSION",
    "TaskEvaluationSceneConstructionQueueError",
    "ensure_scene_construction_queue_root",
    "finalize_scene_construction",
    "preflight_scene_construction_finalization",
    "stage_scene_configuration_revision",
    "stage_scene_construction",
]
