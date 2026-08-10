"""Seal terminal ADP Task Evaluation abstention before any episode exists."""

from __future__ import annotations

import json
import hashlib
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json

SCHEMA_VERSION = "adp_task_evaluation_run_abstention.v1"
CONSTRUCTION_SCHEMA_VERSION = "articulated_public_scene_construction_run.v2"
FREEZE_SCHEMA_VERSION = "second_scene_scene_task_freeze.v1"
NATIVE_GATE_ABSTENTION_SCHEMA_VERSION = "adp_native_gate_abstention.v1"
PROVIDER_ZERO_SCHEMA_VERSION = "adp_paid_provider_zero.v1"
GAUSSIAN_ATTEMPT_SCHEMA_VERSION = "adp_gaussian_excision_attempt_receipt.v1"
GAUSSIAN_RECOVERY_SCHEMA_VERSION = "adp_gaussian_excision_recovery_readiness.v1"
DUAL_TASK_FREEZE_SCHEMA_VERSION = "dual_task_task_freeze.v1"
GAUSSIAN_AUTHORITY_BLOCKER = (
    "fresh_paid_authority_for_qualified_gaussian_contribution_missing"
)


class TaskEvaluationAbstentionError(ValueError):
    """Fail-closed terminal abstention validation error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _bound_json_file(
    root: Path, relative_path: str, *, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise TaskEvaluationAbstentionError(f"native_gate_{role}_file_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationAbstentionError(
            f"native_gate_{role}_json_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise TaskEvaluationAbstentionError(f"native_gate_{role}_json_invalid")
    return value, {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def collect_vast_provider_zero_receipt(
    *,
    command_runner: Callable[[Sequence[str]], subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Collect read-only, API-derived global Vast inventory evidence."""

    command = ["vastai", "show", "instances", "--raw"]
    runner = command_runner or (
        lambda argv: subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    )
    completed = runner(command)
    try:
        inventory = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationAbstentionError(
            "provider_zero_api_response_invalid"
        ) from exc
    if completed.returncode != 0 or not isinstance(inventory, list):
        raise TaskEvaluationAbstentionError("provider_zero_api_query_failed")
    receipt: dict[str, Any] = {
        "schema_version": PROVIDER_ZERO_SCHEMA_VERSION,
        "provider": "vast",
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "api_command": command,
        "api_confirmed": True,
        "global_live_resource_count": len(inventory),
        "provider_zero": inventory == [],
        "inventory": inventory,
        "stderr_present": bool(completed.stderr.strip()),
        "raw_secret_values_recorded": False,
        "provider_zero_digest": "",
    }
    receipt["provider_zero_digest"] = canonical_digest(
        receipt, digest_field="provider_zero_digest"
    )
    if not receipt["provider_zero"]:
        raise TaskEvaluationAbstentionError("provider_zero_not_observed")
    return receipt


def materialize_native_gate_task_evaluation_abstention(
    *,
    scene_task_freeze: Mapping[str, Any],
    evidence_root: str | Path,
    construction_join_relative_path: str,
    native_adapter_relative_path: str,
    teardown_relative_path: str,
    provider_zero_receipt: Mapping[str, Any],
    output_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Seal completion path B when native execution never reached the asset.

    The blocker is derived from retained runtime and teardown bytes.  A caller
    cannot relabel a policy failure, a failed control, or an unqualified asset
    as infrastructure failure through this seam.
    """

    freeze = _clone(scene_task_freeze, code="task_evaluation_freeze_invalid")
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("freeze_digest")
        != canonical_digest(freeze, digest_field="freeze_digest")
    ):
        raise TaskEvaluationAbstentionError("task_evaluation_freeze_invalid")
    evidence = Path(evidence_root).expanduser().resolve()
    if not evidence.is_dir():
        raise TaskEvaluationAbstentionError("native_gate_evidence_root_missing")
    construction, construction_binding = _bound_json_file(
        evidence, construction_join_relative_path, role="construction_join"
    )
    adapter, adapter_binding = _bound_json_file(
        evidence, native_adapter_relative_path, role="adapter"
    )
    teardown, teardown_binding = _bound_json_file(
        evidence, teardown_relative_path, role="teardown"
    )
    provider_zero = _clone(
        provider_zero_receipt, code="native_gate_provider_zero_invalid"
    )
    errors: list[str] = []
    if (
        construction.get("schema_version") != "articulated_excision_join.v1"
        or construction.get("status") != "join_admitted"
        or construction.get("receipt_digest")
        != canonical_digest(construction, digest_field="receipt_digest")
        or (construction.get("claim_boundary") or {}).get(
            "native_simulator_qualified"
        )
        is not False
    ):
        errors.append("native_gate_construction_join_invalid")
    adapter_blockers = set(adapter.get("blockers") or [])
    if (
        adapter.get("schema_version") != "adp009d_franka_vast_run.v1"
        or adapter.get("status") != "blocked"
        or adapter.get("native_control_result_path") is not None
        or adapter.get("candidate_policy_query_expected") is not False
        or adapter.get("continuing_spend_from_this_run") is not False
        or adapter.get("all_staged_objects_absent") is not True
        or adapter.get("retry_cap") != 0
        or "vast_heartbeat_container_missing" not in adapter_blockers
    ):
        errors.append("native_gate_adapter_not_infrastructure_null")
    try:
        adapter_teardown = Path(str(adapter.get("teardown_manifest_path") or "")).resolve()
        expected_teardown = (evidence / teardown_relative_path).resolve()
        if adapter_teardown != expected_teardown:
            errors.append("native_gate_teardown_path_mismatch")
    except (OSError, RuntimeError, ValueError):
        errors.append("native_gate_teardown_path_invalid")
    if (
        teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("runner_gpu_teardown_completed") is not True
        or teardown.get("continuing_spend_from_this_run") is not False
        or not teardown.get("vast_instance_ids")
        or any(
            action.get("status") != "completed"
            or action.get("action") != "destroy_instance"
            or action.get("http_status_code") != 200
            for action in teardown.get("teardown_actions_performed") or []
        )
    ):
        errors.append("native_gate_teardown_invalid")
    if (
        provider_zero.get("schema_version") != PROVIDER_ZERO_SCHEMA_VERSION
        or provider_zero.get("provider") != "vast"
        or provider_zero.get("api_command")
        != ["vastai", "show", "instances", "--raw"]
        or provider_zero.get("api_confirmed") is not True
        or provider_zero.get("global_live_resource_count") != 0
        or provider_zero.get("provider_zero") is not True
        or provider_zero.get("inventory") != []
        or provider_zero.get("provider_zero_digest")
        != canonical_digest(provider_zero, digest_field="provider_zero_digest")
    ):
        errors.append("native_gate_provider_zero_invalid")
    estimated_cost = adapter.get("estimated_cost_usd")
    hard_cap = adapter.get("hard_cap_usd")
    if (
        not isinstance(estimated_cost, (int, float))
        or not isinstance(hard_cap, (int, float))
        or estimated_cost < 0
        or estimated_cost > hard_cap
    ):
        errors.append("native_gate_cost_invalid")
    if errors:
        raise TaskEvaluationAbstentionError(";".join(sorted(set(errors))))

    scene = freeze.get("scene") or {}
    task = freeze.get("task_spec") or {}
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "terminal_gate_schema_version": NATIVE_GATE_ABSTENTION_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "status": "typed_evidence_backed_abstention",
        "scene_id": scene.get("publisher_scene_id"),
        "target_instance_id": scene.get("target_instance_id"),
        "task_id": task.get("task_id"),
        "task_kind": task.get("task_kind"),
        "freeze_digest": freeze["freeze_digest"],
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": (
            "native_articulated_asset_diagnostic_unobserved:"
            "vast_heartbeat_container_missing"
        ),
        "all_terminal_blockers": sorted(adapter_blockers),
        "construction_join": {
            **construction_binding,
            "receipt_digest": construction["receipt_digest"],
        },
        "native_adapter": adapter_binding,
        "teardown": teardown_binding,
        "provider_zero": provider_zero,
        "paid_attempt": {
            "attempt_number": adapter["attempt_number"],
            "estimated_cost_usd": float(estimated_cost),
            "hard_cap_usd": float(hard_cap),
            "hard_ttl_seconds": adapter["hard_ttl_seconds"],
            "automatic_retry_executed": False,
        },
        "native_asset_opened": False,
        "native_simulator_qualification_observed": False,
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "episode_media_exists": False,
        "comparison_exists": False,
        "automatic_paid_retry_executed": False,
        "claim_ceiling": (
            "public_dataset_construction_rehearsal_only; no partner capture, "
            "real_site_fidelity, deployment readiness, physical performance, "
            "native_asset_qualification, control outcome, or learned_policy_comparison"
        ),
        "next_action": (
            "repair and hermetically test the generic Vast container-heartbeat startup "
            "path, then authorize one new zero-retry native articulated diagnostic"
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if output_path is not None:
        if repo_root is None:
            raise TaskEvaluationAbstentionError("task_evaluation_repo_root_missing")
        repo = Path(repo_root).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        if not output.is_relative_to(repo) or output.is_symlink():
            raise TaskEvaluationAbstentionError(
                "task_evaluation_abstention_output_outside_repo"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def _clone(value: Mapping[str, Any], *, code: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEvaluationAbstentionError(code) from exc
    return result


def materialize_gaussian_contribution_authority_abstention(
    *,
    gaussian_excision_attempt: Mapping[str, Any],
    recovery_readiness: Mapping[str, Any],
    task_freeze: Mapping[str, Any],
    removal_binding: Mapping[str, Any],
    scene_id: str,
    output_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Seal the current gate after a failed attempt has been repaired locally.

    Historical worker failures remain immutable attempt evidence.  Once an exact
    repaired bundle passes its hermetic rehearsal and paid-admission dry-run,
    those failures are no longer the current smallest blocker.  This seam makes
    that distinction explicit and refuses to imply that the unexecuted repair
    produced Gaussian ownership evidence.
    """

    attempt = _clone(
        gaussian_excision_attempt, code="gaussian_authority_attempt_invalid"
    )
    recovery = _clone(
        recovery_readiness, code="gaussian_authority_recovery_invalid"
    )
    freeze = _clone(task_freeze, code="gaussian_authority_task_freeze_invalid")
    binding = _clone(
        removal_binding, code="gaussian_authority_removal_binding_invalid"
    )
    errors: list[str] = []
    historical_blockers = attempt.get("execution_blockers")
    proof_boundaries = attempt.get("proof_boundaries") or {}
    if (
        attempt.get("schema_version") != GAUSSIAN_ATTEMPT_SCHEMA_VERSION
        or attempt.get("receipt_digest")
        != canonical_digest(attempt, digest_field="receipt_digest")
        or attempt.get("status") != "sealed_blocked_attempt"
        or attempt.get("execution_status") != "blocked"
        or not isinstance(historical_blockers, list)
        or not historical_blockers
        or attempt.get("released_code_executed") is not False
        or attempt.get("heldout_cameras_accessed_for_classification") is not False
        or attempt.get("continuing_spend") is not False
        or attempt.get("provider_absence_confirmed") is not True
        or proof_boundaries.get("gaussian_contribution_evidence_completed") is not False
        or proof_boundaries.get("gaussian_ownership_qualified") is not False
        or proof_boundaries.get("source_removal_qualified") is not False
    ):
        errors.append("gaussian_authority_attempt_invalid")
    recovery_boundaries = recovery.get("proof_boundaries") or {}
    if (
        recovery.get("schema_version") != GAUSSIAN_RECOVERY_SCHEMA_VERSION
        or recovery.get("receipt_digest")
        != canonical_digest(recovery, digest_field="receipt_digest")
        or recovery.get("status") != "ready_for_new_authority_not_executed"
        or recovery.get("exact_bundle_rehearsal_passed") is not True
        or recovery.get("canonical_paid_admission_dry_run_passed") is not True
        or recovery.get("provider_mutations_performed") != 0
        or recovery.get("automatic_retry_authorized") is not False
        or recovery_boundaries.get("gpu_runtime_executed") is not False
        or recovery_boundaries.get("gaussian_ownership_qualified") is not False
        or recovery_boundaries.get("new_paid_authority_required") is not True
    ):
        errors.append("gaussian_authority_recovery_invalid")
    if (
        freeze.get("schema_version") != DUAL_TASK_FREEZE_SCHEMA_VERSION
        or freeze.get("task_freeze_digest")
        != canonical_digest(freeze, digest_field="task_freeze_digest")
        or freeze.get("candidate_ids") != ["pi05_droid", "groot_n17_droid"]
        or freeze.get("learned_policy_outcomes_accessed") is not False
    ):
        errors.append("gaussian_authority_task_freeze_invalid")
    freeze_digest = freeze.get("task_freeze_digest")
    excision_freeze_digest = (binding.get("excision_freeze") or {}).get(
        "freeze_digest"
    )
    if (
        not str(binding.get("schema_version") or "")
        or binding.get("binding_digest")
        != canonical_digest(binding, digest_field="binding_digest")
        or binding.get("task_id") != freeze.get("task_id")
        or binding.get("task_freeze_digest") != freeze_digest
        or attempt.get("freeze_digest") != recovery.get("freeze_digest")
        or attempt.get("freeze_digest") != excision_freeze_digest
    ):
        errors.append("gaussian_authority_removal_binding_invalid")
    if not scene_id or not str(freeze.get("task_id") or ""):
        errors.append("gaussian_authority_identity_invalid")
    if errors:
        raise TaskEvaluationAbstentionError(";".join(sorted(set(errors))))

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "terminal_gate_schema_version": "adp_gaussian_contribution_authority_gate.v1",
        "program_id": "arm-decision-proof-v1",
        "status": "typed_evidence_backed_abstention",
        "scene_id": scene_id,
        "task_id": freeze["task_id"],
        "task_freeze_digest": freeze_digest,
        "gaussian_excision_freeze_digest": excision_freeze_digest,
        "removal_binding_digest": binding["binding_digest"],
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": GAUSSIAN_AUTHORITY_BLOCKER,
        "all_terminal_blockers": [GAUSSIAN_AUTHORITY_BLOCKER],
        "historical_attempt_blockers": list(historical_blockers),
        "gaussian_excision_attempt_receipt_digest": attempt["receipt_digest"],
        "recovery_readiness_receipt_digest": recovery["receipt_digest"],
        "repaired_bundle": {
            "blueprint_commit": recovery["blueprint_commit"],
            "bundle_sha256": recovery["bundle_sha256"],
            "container_image": recovery["container_image"],
            "dependency_wheelhouse_manifest_digest": recovery[
                "dependency_wheelhouse_manifest_digest"
            ],
            "exact_bundle_rehearsal_passed": True,
            "canonical_paid_admission_dry_run_passed": True,
            "gpu_runtime_executed": False,
        },
        "gaussian_contribution_evidence_completed": False,
        "gaussian_ownership_qualified": False,
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "episode_media_exists": False,
        "comparison_exists": False,
        "automatic_paid_retry_executed": False,
        "next_action": (
            "obtain fresh explicit one-attempt zero-retry authority for the exact "
            "repaired bundle and require qualified Gaussian contribution output"
        ),
        "claim_ceiling": (
            "public_dataset_simulator_construction_rehearsal_only; no physical, "
            "deployment, customer_value, Gaussian_ownership, or learned_policy claim"
        ),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if output_path is not None:
        if repo_root is None:
            raise TaskEvaluationAbstentionError("task_evaluation_repo_root_missing")
        repo = Path(repo_root).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        if not output.is_relative_to(repo) or output.is_symlink():
            raise TaskEvaluationAbstentionError(
                "task_evaluation_abstention_output_outside_repo"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def materialize_task_evaluation_abstention(
    *,
    construction_run: Mapping[str, Any],
    scene_task_freeze: Mapping[str, Any],
    output_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Turn observed terminal construction receipts into completion path B."""

    run = _clone(construction_run, code="task_evaluation_construction_run_invalid")
    freeze = _clone(scene_task_freeze, code="task_evaluation_freeze_invalid")
    errors: list[str] = []
    if (
        run.get("schema_version") != CONSTRUCTION_SCHEMA_VERSION
        or run.get("run_digest") != canonical_digest(run, digest_field="run_digest")
        or run.get("status") != "typed_abstention_before_simready_build"
    ):
        errors.append("task_evaluation_construction_run_invalid")
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA_VERSION
        or freeze.get("freeze_digest")
        != canonical_digest(freeze, digest_field="freeze_digest")
    ):
        errors.append("task_evaluation_freeze_invalid")
    scene = run.get("scene") or {}
    freeze_scene = freeze.get("scene") or {}
    if (
        scene.get("publisher_scene_id") != freeze_scene.get("publisher_scene_id")
        or scene.get("target_instance_id") != freeze_scene.get("target_instance_id")
        or scene.get("freeze_digest") != freeze.get("freeze_digest")
        or run.get("frozen_candidates") != ["pi05_droid", "groot_n17_droid"]
    ):
        errors.append("task_evaluation_construction_freeze_join_invalid")
    stage_receipts = run.get("stage_receipts") or {}
    stage_status = run.get("stage_status") or {}
    blockers = run.get("blockers")
    if (
        not isinstance(blockers, list)
        or not blockers
        or run.get("smallest_blocker") != blockers[0]
    ):
        errors.append("task_evaluation_terminal_blocker_missing")
    if (
        not stage_receipts.get("aura_execution")
        or stage_status.get("released_code_inpainting_executed") is not True
    ):
        errors.append("task_evaluation_construction_attempt_receipts_incomplete")
    if (
        stage_status.get("simready_replacement_materialized") is not False
        or stage_status.get("native_simulator_qualified") is not False
        or stage_status.get("controls_executed") is not False
        or stage_status.get("learned_candidates_executed") is not False
    ):
        errors.append("task_evaluation_abstention_after_episode_state_invalid")
    if any(
        str(blocker).startswith("joint_agent_topology_execution")
        for blocker in (blockers or [])
    ):
        errors.append("research_preview_agent_cannot_be_terminal_blocker")
    if errors:
        raise TaskEvaluationAbstentionError(";".join(sorted(set(errors))))

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "status": "typed_evidence_backed_abstention",
        "scene_id": scene["publisher_scene_id"],
        "target_instance_id": scene["target_instance_id"],
        "task_id": (freeze.get("task_spec") or {}).get("task_id"),
        "task_kind": scene.get("task_kind"),
        "freeze_digest": freeze["freeze_digest"],
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "smallest_missing_capability": run["smallest_blocker"],
        "all_terminal_construction_blockers": list(blockers),
        "construction_run_digest": run["run_digest"],
        "stage_receipts": dict(stage_receipts),
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "episode_media_exists": False,
        "comparison_exists": False,
        "automatic_paid_retry_executed": False,
        "research_preview_agents_are_nonblocking_enrichment": True,
        "claim_ceiling": (
            "public_dataset_construction_rehearsal_only; no partner capture, "
            "real_site_fidelity, deployment readiness, physical performance, "
            "or learned_policy_comparison"
        ),
        "next_action": run.get("next_action"),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    if output_path is not None:
        if repo_root is None:
            raise TaskEvaluationAbstentionError("task_evaluation_repo_root_missing")
        repo = Path(repo_root).expanduser().resolve()
        output = Path(output_path).expanduser().resolve()
        if not output.is_relative_to(repo) or output.is_symlink():
            raise TaskEvaluationAbstentionError(
                "task_evaluation_abstention_output_outside_repo"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "TaskEvaluationAbstentionError",
    "collect_vast_provider_zero_receipt",
    "materialize_gaussian_contribution_authority_abstention",
    "materialize_task_evaluation_abstention",
    "materialize_native_gate_task_evaluation_abstention",
]
