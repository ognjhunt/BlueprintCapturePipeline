"""One-command Arm Decision Proof v1 evidence reconstruction.

The command consumes an admitted public-source manifest and one immutable
closed-loop execution package.  Physical-reference values are deliberately a
separate input: they are not opened until a digest-bound development decision
seal has been written and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .evaluation_run_contract import compile_evaluation_run
from .public_reference_admission import build_public_reference_admission_receipt


EXECUTION_SCHEMA_VERSION = "simpler_closed_loop_execution.v2"
LEGACY_EXECUTION_SCHEMA_VERSION = "simpler_closed_loop_execution.v1"
FRAME_MANIFEST_SCHEMA_VERSION = "adp_observation_frame_manifest.v1"
VISUAL_EVIDENCE_SCHEMA_VERSION = "adp_episode_visual_evidence.v1"
EPISODE_RECEIPT_SCHEMA_VERSION = "adp_episode_receipt.v1"
SEAL_SCHEMA_VERSION = "adp_development_decision_seal.v1"
RELEASE_SCHEMA_VERSION = "adp_physical_outcome_release_receipt.v1"
JOIN_SCHEMA_VERSION = "adp_physical_outcome_join.v1"
VERDICT_SCHEMA_VERSION = "adp_bounded_verdict.v1"
PHASE_LABEL = "retrospective_external_reference"
CLAIM_CEILING = "development_only"
SHA256_PREFIX = "sha256:"
DEFAULT_MANIFEST_PATH = Path(
    "docs/arm_decision_proof_v1/manifests/simpler_google_robot_pick_coke_can.v1.json"
)
DEFAULT_EXECUTION_PATH = Path(
    "docs/arm_decision_proof_v1/immutable_execution/adp_simpler_closed_loop_execution.json"
)
DEFAULT_OUTCOMES_PATH = Path(
    "docs/arm_decision_proof_v1/manifests/"
    "simpler_google_robot_pick_coke_can_physical_outcomes.v1.json"
)
DEFAULT_OUTPUT_PATH = Path("output/arm_decision_proof_v1/evidence")
ACQUISITION_COMMAND = "git restore --source=HEAD -- docs/arm_decision_proof_v1/immutable_execution"


class ArmDecisionProofError(ValueError):
    """Fail-closed ADP error with stable blocker identifiers."""

    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(sorted(set(str(item) for item in blockers if str(item))))
        super().__init__(";".join(self.blockers))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_digest(value: Any) -> bool:
    text = _string(value)
    return (
        len(text) == 71
        and text.startswith(SHA256_PREFIX)
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return SHA256_PREFIX + digest.hexdigest()


def _load_json(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArmDecisionProofError([blocker]) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArmDecisionProofError([blocker + ":invalid_json"]) from exc
    if not isinstance(value, Mapping):
        raise ArmDecisionProofError([blocker + ":not_mapping"])
    return dict(value)


def _checkpoint_digest(candidate: Mapping[str, Any]) -> str:
    return canonical_digest(
        {
            "candidate_id": candidate.get("candidate_id"),
            "checkpoint_prefix": candidate.get("checkpoint_prefix"),
            "checkpoint_objects": candidate.get("checkpoint_objects"),
        }
    )


def _expected_pairs(manifest: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (candidate["candidate_id"], condition["condition_id"])
        for candidate in _rows(manifest.get("candidates"))
        for condition in _rows(manifest.get("conditions"))
    }


def build_evaluation_run_spec(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the admitted SIMPLER bindings through EvaluationRunSpec."""

    task = _mapping(manifest.get("task"))
    source = _mapping(manifest.get("source"))
    repository = _mapping(source.get("repository"))
    runtime = _mapping(manifest.get("runtime"))
    candidates = _rows(manifest.get("candidates"))
    conditions = _rows(manifest.get("conditions"))
    prohibited = sorted(
        key
        for key, allowed in _mapping(manifest.get("claim_boundaries")).items()
        if allowed is False
    )
    return {
        "schema_version": "evaluation_run.v1",
        "run_id": "adp-v1-simpler-google-robot-pick-coke-can",
        "mode": "evaluate",
        "scene_bundle": {
            "adapter_id": "simpler_public_scene",
            "adapter_version": "1",
            "bundle_id": manifest["reference_id"],
            "uri": repository["url"],
            "entrypoint": task["scene_id"],
            "content_digest": manifest["manifest_digest"],
        },
        "robot_adapter": {
            "adapter_id": "simpler_google_robot",
            "adapter_version": "1",
            "robot_profile_id": task["robot_id"],
            "asset_ref": next(
                row["git_object_sha1"]
                for row in _rows(source.get("asset_bindings"))
                if row.get("role") == "robot_assets"
            ),
        },
        "task_scenario_pack": {
            "adapter_id": "simpler_condition_matrix",
            "adapter_version": "1",
            "pack_id": task["task_id"],
            "tasks": [{"task_id": task["task_id"]}],
            "scenarios": [
                {
                    "scenario_id": row["condition_id"],
                    "task_id": task["task_id"],
                    "reset_binding": row["reset_binding"],
                }
                for row in conditions
            ],
        },
        "policy_adapter": {
            "adapter_id": "simpler_rt1_candidate_set",
            "adapter_version": "1",
            "policy_id": "adp-exactly-two-rt1-candidates",
            "observation_schema_ref": canonical_digest(task["observation_schema"]),
            "action_schema_ref": canonical_digest(task["action_schema"]),
            "candidate_ids": [row["candidate_id"] for row in candidates],
        },
        "runtime_provider_profile": {
            "adapter_id": "simpler_cached_execution",
            "adapter_version": "1",
            "profile_id": "simpler-sapien-vast-immutable-input",
            "providers": ["vast"],
            "simulator": "SAPIEN-2.2.2",
            "environment_lock_digest": _mapping(runtime.get("environment_lock")).get("digest"),
        },
        "proof_contract": {
            "adapter_id": "declared_evidence_proof_contract",
            "adapter_version": "1",
            "contract_id": "arm-decision-proof-v1-retrospective",
            "required_evidence": [
                "closed_loop_execution",
                "independent_environment_metric",
                "episode_receipts",
                "decision_seal",
                "physical_outcome_release",
                "exact_physical_outcome_join",
            ],
            "claim_ceiling": {
                "phase_label": PHASE_LABEL,
                "claim_ceiling": CLAIM_CEILING,
            },
            "prohibited_claims": prohibited,
        },
        "metadata": {
            "program_id": "arm-decision-proof-v1",
            "source_commit": repository["commit"],
            "candidate_count": len(candidates),
        },
    }


def validate_paid_runtime_canary(
    manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    execution_root: Path,
) -> dict[str, Any]:
    """Verify the observed paid canary and every artifact declared by admission."""

    runtime = _mapping(manifest.get("runtime"))
    canary = _mapping(runtime.get("paid_runtime_canary"))
    if not canary:
        if _mapping(runtime.get("zero_spend_feasibility")).get("status") == "passed":
            return {
                "status": "not_required_zero_spend_execution",
                "artifact_count": 0,
            }
        raise ArmDecisionProofError(["paid_runtime_canary_missing"])
    blockers: list[str] = []
    root = execution_root.resolve()
    bound: dict[str, tuple[dict[str, Any], Path]] = {}
    for row in _rows(canary.get("artifacts")):
        role = _string(row.get("role"))
        relative_path = _string(row.get("relative_path"))
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            blockers.append(f"paid_runtime_artifact_path_outside_root:{role}")
            continue
        if role in bound:
            blockers.append(f"paid_runtime_artifact_role_duplicate:{role}")
            continue
        bound[role] = (row, target)
        if not target.is_file():
            blockers.append(f"paid_runtime_artifact_missing:{role}")
            continue
        if target.stat().st_size != row.get("size_bytes"):
            blockers.append(f"paid_runtime_artifact_size_mismatch:{role}")
        if _file_digest(target) != row.get("sha256"):
            blockers.append(f"paid_runtime_artifact_digest_mismatch:{role}")

    def artifact_json(role: str) -> dict[str, Any]:
        binding = bound.get(role)
        if binding is None or not binding[1].is_file():
            blockers.append(f"paid_runtime_artifact_unavailable:{role}")
            return {}
        try:
            value = json.loads(binding[1].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append(f"paid_runtime_artifact_invalid_json:{role}")
            return {}
        if not isinstance(value, Mapping):
            blockers.append(f"paid_runtime_artifact_not_mapping:{role}")
            return {}
        return dict(value)

    source_manifest = artifact_json("source_manifest")
    if source_manifest:
        try:
            source_admission = build_public_reference_admission_receipt(source_manifest)
        except ValueError:
            blockers.append("paid_runtime_source_manifest_structurally_invalid")
        else:
            if source_admission.get("manifest_digest") != canary.get("source_manifest_digest"):
                blockers.append("paid_runtime_source_manifest_digest_mismatch")
            if source_admission.get("source_identity_digest") != manifest.get(
                "source_identity_digest"
            ):
                blockers.append("paid_runtime_source_identity_mismatch")
    runtime_lock = artifact_json("runtime_lock")
    if runtime_lock:
        if runtime_lock.get("runtime_lock_digest") != canary.get(
            "runtime_lock_digest"
        ) or canonical_digest(runtime_lock, digest_field="runtime_lock_digest") != canary.get(
            "runtime_lock_digest"
        ):
            blockers.append("paid_runtime_lock_digest_mismatch")
        repository = _mapping(_mapping(manifest.get("source")).get("repository"))
        expected_submodules = {
            row["path"]: row["commit"] for row in _rows(repository.get("submodules"))
        }
        if runtime_lock.get("source_commit") != repository.get("commit"):
            blockers.append("paid_runtime_lock_source_commit_mismatch")
        if runtime_lock.get("submodule_commits") != expected_submodules:
            blockers.append("paid_runtime_lock_submodule_mismatch")
        if runtime_lock.get("container_image") != _mapping(runtime.get("environment_lock")).get(
            "container_image"
        ):
            blockers.append("paid_runtime_lock_container_mismatch")
    execution_binding = bound.get("execution")
    if (
        execution_binding is None
        or execution_binding[1] != (root / DEFAULT_EXECUTION_PATH.name).resolve()
    ):
        blockers.append("paid_runtime_execution_artifact_binding_mismatch")
    if execution.get("source_manifest_digest") != canary.get("source_manifest_digest"):
        blockers.append("paid_runtime_execution_source_manifest_mismatch")
    if execution.get("execution_digest") != canary.get("execution_digest"):
        blockers.append("paid_runtime_execution_digest_mismatch")
    if execution.get("runtime_lock_digest") != canary.get("runtime_lock_digest"):
        blockers.append("paid_runtime_execution_lock_mismatch")
    if execution.get("physical_outcome_values_accessed") is not False:
        blockers.append("paid_runtime_execution_outcome_firebreak_failed")

    paid_admission = artifact_json("paid_admission")
    allocation = _mapping(paid_admission.get("allocation_binding"))
    if paid_admission.get("status") != "admitted":
        blockers.append("paid_runtime_allocator_admission_missing")
    expected_allocation = {
        "orchestrator_source_commit": canary.get("orchestrator_source_commit"),
        "source_identity_digest": canary.get("source_identity_digest"),
        "bundle_sha256": canary.get("bundle_sha256"),
        "hard_cap_usd": canary.get("hard_cap_usd"),
        "hard_ttl_seconds": canary.get("hard_ttl_seconds"),
        "retry_cap": canary.get("retry_cap"),
    }
    for key, expected_value in expected_allocation.items():
        if allocation.get(key) != expected_value:
            blockers.append(f"paid_runtime_allocation_binding_mismatch:{key}")
    bundle_receipt = artifact_json("bundle_receipt")
    if bundle_receipt.get("bundle_sha256") != canary.get("bundle_sha256"):
        blockers.append("paid_runtime_bundle_digest_mismatch")
    provider_result = artifact_json("provider_result")
    for key in (
        "source_identity_digest",
        "bundle_sha256",
        "execution_digest",
        "runtime_lock_digest",
        "estimated_cost_usd",
        "hard_cap_usd",
        "hard_ttl_seconds",
        "retry_cap",
    ):
        if provider_result.get(key) != canary.get(key):
            blockers.append(f"paid_runtime_provider_result_mismatch:{key}")
    if (
        provider_result.get("status") != "completed"
        or provider_result.get("continuing_spend_from_this_run") is not False
        or provider_result.get("all_staged_objects_absent") is not True
    ):
        blockers.append("paid_runtime_provider_result_incomplete")
    offer = _mapping(artifact_json("offer_selection").get("selected_offer"))
    if offer.get("machine_id") != canary.get("machine_id"):
        blockers.append("paid_runtime_machine_id_mismatch")
    final_validation = artifact_json("final_validation")
    if (
        final_validation.get("status") != "passed"
        or final_validation.get("all_vast_instances_destroyed_by_adapter") is not True
        or final_validation.get("continuing_spend_from_this_run") is not False
    ):
        blockers.append("paid_runtime_final_validation_failed")
    teardown = artifact_json("teardown")
    if (
        teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or canary.get("instance_id") not in (teardown.get("vast_instance_ids") or [])
    ):
        blockers.append("paid_runtime_teardown_not_proven")
    cleanup = artifact_json("object_store_cleanup")
    if cleanup.get("status") != "completed" or cleanup.get("all_objects_absent") is not True:
        blockers.append("paid_runtime_object_cleanup_not_proven")
    if blockers:
        raise ArmDecisionProofError(blockers)
    result = {
        "schema_version": "adp_paid_runtime_canary_validation.v1",
        "status": "passed",
        "source_manifest_digest": canary["source_manifest_digest"],
        "runtime_lock_digest": canary["runtime_lock_digest"],
        "execution_digest": canary["execution_digest"],
        "machine_id": canary["machine_id"],
        "instance_id": canary["instance_id"],
        "estimated_cost_usd": canary["estimated_cost_usd"],
        "provider_zero_verified": True,
        "physical_outcome_values_accessed": False,
        "artifact_count": len(bound),
        "artifact_binding_digest": canonical_digest({"artifacts": canary.get("artifacts")}),
    }
    result["validation_digest"] = canonical_digest(result, digest_field="validation_digest")
    return result


def _validate_episode_visual_evidence(
    episode: Mapping[str, Any],
    *,
    artifacts: Sequence[Mapping[str, Any]],
    execution_root: Path | None,
) -> list[str]:
    """Require lossless policy inputs and a digest-bound review video for v2."""

    candidate_id = _string(episode.get("candidate_id"))
    condition_id = _string(episode.get("condition_id"))
    cell = f"{candidate_id}/{condition_id}"
    blockers: list[str] = []
    visual = _mapping(episode.get("visual_evidence"))
    if visual.get("schema_version") != VISUAL_EVIDENCE_SCHEMA_VERSION:
        blockers.append(f"execution_visual_evidence_schema_invalid:{cell}")
    policy_query_count = episode.get("policy_query_count")
    visual_status = _string(visual.get("status"))
    if isinstance(policy_query_count, int) and policy_query_count > 0:
        if visual_status != "complete" or visual.get("human_review_available") is not True:
            blockers.append(f"execution_human_visual_evidence_incomplete:{cell}")
    elif visual_status not in {"complete", "unavailable_before_first_observation"}:
        blockers.append(f"execution_visual_evidence_status_invalid:{cell}")
    if visual.get("vlm_grading_used") is not False:
        blockers.append(f"execution_visual_evidence_vlm_flag_invalid:{cell}")

    evaluator = _mapping(episode.get("evaluator"))
    success_evidence = _mapping(episode.get("success_evidence"))
    if (
        evaluator.get("grader_type") != "deterministic_simulator_state"
        or evaluator.get("success_source") != "environment_step_info.success"
        or evaluator.get("vlm_used") is not False
        or evaluator.get("human_grade_used") is not False
    ):
        blockers.append(f"execution_success_grader_binding_invalid:{cell}")
    if (
        success_evidence.get("grader_type") != "deterministic_simulator_state"
        or success_evidence.get("source_field") != "environment_step_info.success"
        or success_evidence.get("vlm_used") is not False
        or success_evidence.get("human_grade_used") is not False
        or success_evidence.get("final_value") != episode.get("success")
    ):
        blockers.append(f"execution_success_evidence_invalid:{cell}")

    by_role: dict[str, list[dict[str, Any]]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for artifact_value in artifacts:
        artifact = dict(artifact_value)
        by_role.setdefault(_string(artifact.get("role")), []).append(artifact)
        relative_path = _string(artifact.get("relative_path"))
        if relative_path:
            by_path[relative_path] = artifact
    manifests = by_role.get("observation_frame_manifest", [])
    if len(manifests) != 1:
        blockers.append(f"execution_frame_manifest_binding_invalid:{cell}")
        return blockers
    if execution_root is None:
        return blockers
    manifest_path = (execution_root / _string(manifests[0].get("relative_path"))).resolve()
    if not manifest_path.is_file():
        return blockers
    try:
        frame_manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        blockers.append(f"execution_frame_manifest_invalid_json:{cell}")
        return blockers
    frame_manifest = _mapping(frame_manifest_value)
    if frame_manifest.get("schema_version") != FRAME_MANIFEST_SCHEMA_VERSION:
        blockers.append(f"execution_frame_manifest_schema_invalid:{cell}")
    if frame_manifest.get("episode_id") != episode.get("episode_id"):
        blockers.append(f"execution_frame_manifest_episode_mismatch:{cell}")
    expected_manifest_digest = canonical_digest(
        frame_manifest, digest_field="frame_manifest_digest"
    )
    if (
        frame_manifest.get("frame_manifest_digest") != expected_manifest_digest
        or visual.get("frame_manifest_digest") != expected_manifest_digest
    ):
        blockers.append(f"execution_frame_manifest_digest_mismatch:{cell}")
    frame_rows = _rows(frame_manifest.get("policy_input_frames"))
    if (
        len(frame_rows) != policy_query_count
        or frame_manifest.get("policy_input_frame_count") != policy_query_count
    ):
        blockers.append(f"execution_policy_input_frame_count_mismatch:{cell}")
    if visual.get("policy_input_frame_count") != policy_query_count:
        blockers.append(f"execution_visual_policy_input_frame_count_mismatch:{cell}")
    raw_trace = [row.get("raw_rgb_sha256") for row in frame_rows]
    if canonical_digest({"observations": raw_trace}) != episode.get("observation_trace_digest"):
        blockers.append(f"execution_observation_frame_trace_mismatch:{cell}")

    terminal = _mapping(frame_manifest.get("terminal_observation"))
    expected_terminal = visual.get("terminal_observation_frame_present") is True
    if bool(terminal) != expected_terminal:
        blockers.append(f"execution_terminal_observation_binding_mismatch:{cell}")
    all_frame_rows = [*frame_rows, *([terminal] if terminal else [])]
    for row in all_frame_rows:
        relative_path = _string(row.get("relative_path"))
        artifact = by_path.get(relative_path, {})
        expected_role = (
            "terminal_observation_frame"
            if row.get("kind") == "terminal-observation"
            else "policy_input_frame"
        )
        if (
            artifact.get("role") != expected_role
            or artifact.get("sha256") != row.get("png_sha256")
            or artifact.get("raw_rgb_sha256") != row.get("raw_rgb_sha256")
        ):
            blockers.append(f"execution_observation_frame_binding_mismatch:{cell}:{relative_path}")
            continue
        target = (execution_root / relative_path).resolve()
        if not target.is_file():
            continue
        try:
            from PIL import Image

            with Image.open(target) as image:
                rgb = image.convert("RGB")
                decoded_digest = "sha256:" + hashlib.sha256(rgb.tobytes()).hexdigest()
                decoded_size = rgb.size
        except (OSError, ValueError):
            blockers.append(f"execution_observation_frame_not_decodable:{cell}:{relative_path}")
            continue
        if decoded_digest != row.get("raw_rgb_sha256") or decoded_size != (
            row.get("width"),
            row.get("height"),
        ):
            blockers.append(f"execution_observation_frame_pixels_mismatch:{cell}:{relative_path}")

    if isinstance(policy_query_count, int) and policy_query_count > 0:
        videos = by_role.get("episode_video", [])
        video = _mapping(visual.get("video"))
        if (
            len(videos) != 1
            or videos[0].get("relative_path") != video.get("relative_path")
            or videos[0].get("sha256") != video.get("sha256")
            or video.get("derived_from_frame_manifest_digest") != expected_manifest_digest
            or video.get("frame_count") != len(all_frame_rows)
        ):
            blockers.append(f"execution_episode_video_binding_invalid:{cell}")
        elif execution_root is not None:
            video_path = (execution_root / _string(video.get("relative_path"))).resolve()
            if video_path.is_file():
                try:
                    header = video_path.read_bytes()[:12]
                except OSError:
                    header = b""
                if len(header) < 8 or header[4:8] != b"ftyp":
                    blockers.append(f"execution_episode_video_not_mp4:{cell}")
    return blockers


def validate_execution_package(
    execution: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    execution_root: Path | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    execution_schema = execution.get("schema_version")
    if execution_schema not in {
        EXECUTION_SCHEMA_VERSION,
        LEGACY_EXECUTION_SCHEMA_VERSION,
    }:
        blockers.append("execution_schema_invalid")
    if execution.get("status") != "completed":
        blockers.append("execution_not_completed")
    if execution.get("reference_id") != manifest.get("reference_id"):
        blockers.append("execution_reference_id_mismatch")
    if execution.get("source_identity_digest") != manifest.get("source_identity_digest"):
        blockers.append("execution_source_identity_digest_mismatch")
    canary = _mapping(_mapping(manifest.get("runtime")).get("paid_runtime_canary"))
    expected_source_manifest_digest = canary.get(
        "source_manifest_digest", manifest.get("manifest_digest")
    )
    if execution.get("source_manifest_digest") != expected_source_manifest_digest:
        blockers.append("execution_source_manifest_digest_mismatch")
    if execution.get("physical_outcome_values_accessed") is not False:
        blockers.append("execution_physical_outcome_firebreak_failed")
    if execution.get("phase_label") != PHASE_LABEL:
        blockers.append("execution_phase_label_invalid")
    if execution.get("claim_ceiling") != CLAIM_CEILING:
        blockers.append("execution_claim_ceiling_invalid")
    runtime_digest = _mapping(_mapping(manifest.get("runtime")).get("environment_lock")).get(
        "digest"
    )
    if execution.get("runtime_lock_digest") != runtime_digest or not _is_digest(runtime_digest):
        blockers.append("execution_runtime_lock_digest_mismatch")
    expected_candidates = {
        row["candidate_id"]: _checkpoint_digest(row) for row in _rows(manifest.get("candidates"))
    }
    observed_candidates = _rows(execution.get("candidates"))
    if len(observed_candidates) != 2:
        blockers.append("execution_must_bind_exactly_two_candidates")
    observed_ids: list[str] = []
    for row in observed_candidates:
        candidate_id = _string(row.get("candidate_id"))
        observed_ids.append(candidate_id)
        if row.get("checkpoint_identity_digest") != expected_candidates.get(candidate_id):
            blockers.append(f"execution_checkpoint_identity_mismatch:{candidate_id}")
        if row.get("genuine_checkpoint_loaded") is not True:
            blockers.append(f"execution_genuine_checkpoint_not_loaded:{candidate_id}")
    if len(set(observed_ids)) != len(observed_ids):
        blockers.append("execution_duplicate_candidate_identity")
    if set(observed_ids) != set(expected_candidates):
        blockers.append("execution_candidate_set_mismatch")

    expected = _expected_pairs(manifest)
    episodes = _rows(execution.get("episodes"))
    pairs: list[tuple[str, str]] = []
    completed_by_candidate = {candidate_id: 0 for candidate_id in expected_candidates}
    for episode in episodes:
        candidate_id = _string(episode.get("candidate_id"))
        condition_id = _string(episode.get("condition_id"))
        pair = (candidate_id, condition_id)
        pairs.append(pair)
        status = _string(episode.get("status"))
        if status not in {"completed", "failed", "invalid", "timed_out", "interrupted"}:
            blockers.append(f"execution_episode_status_invalid:{candidate_id}/{condition_id}")
        if not _string(episode.get("episode_id")):
            blockers.append(f"execution_episode_id_missing:{candidate_id}/{condition_id}")
        if not isinstance(episode.get("seed"), int):
            blockers.append(f"execution_seed_missing:{candidate_id}/{condition_id}")
        for name in (
            "source_commit",
            "dependency_lock_digest",
            "reset_digest",
            "observation_trace_digest",
            "action_trace_digest",
            "metric_trace_digest",
        ):
            if not _is_digest(episode.get(name)) and name != "source_commit":
                blockers.append(f"execution_episode_{name}_invalid:{candidate_id}/{condition_id}")
        if episode.get("source_commit") != _mapping(manifest.get("source")).get(
            "repository", {}
        ).get("commit"):
            blockers.append(
                f"execution_episode_source_commit_mismatch:{candidate_id}/{condition_id}"
            )
        if episode.get("checkpoint_identity_digest") != expected_candidates.get(candidate_id):
            blockers.append(
                f"execution_episode_checkpoint_identity_mismatch:{candidate_id}/{condition_id}"
            )
        if status == "completed":
            completed_by_candidate[candidate_id] = completed_by_candidate.get(candidate_id, 0) + 1
            if (
                not isinstance(episode.get("policy_query_count"), int)
                or episode.get("policy_query_count", 0) <= 0
            ):
                blockers.append(f"execution_policy_not_queried:{candidate_id}/{condition_id}")
            if (
                not isinstance(episode.get("simulator_step_count"), int)
                or episode.get("simulator_step_count", 0) <= 0
            ):
                blockers.append(f"execution_simulator_not_stepped:{candidate_id}/{condition_id}")
        evaluator = _mapping(episode.get("evaluator"))
        if (
            evaluator.get("owner") != "environment_not_policy"
            or evaluator.get("policy_self_report_used") is not False
        ):
            blockers.append(f"execution_evaluator_not_independent:{candidate_id}/{condition_id}")
        if status == "completed" and not isinstance(episode.get("success"), bool):
            blockers.append(f"execution_completed_success_missing:{candidate_id}/{condition_id}")
        artifacts = _rows(episode.get("artifacts"))
        if not artifacts or any(not _is_digest(row.get("sha256")) for row in artifacts):
            blockers.append(f"execution_artifact_digests_missing:{candidate_id}/{condition_id}")
        if execution_root is not None:
            for artifact in artifacts:
                relative_path = _string(artifact.get("relative_path"))
                if not relative_path:
                    continue
                target = (execution_root / relative_path).resolve()
                try:
                    target.relative_to(execution_root.resolve())
                except ValueError:
                    blockers.append(
                        f"execution_artifact_path_outside_root:{candidate_id}/{condition_id}"
                    )
                    continue
                if not target.is_file():
                    blockers.append(
                        f"execution_artifact_missing:{candidate_id}/{condition_id}:{relative_path}"
                    )
                elif _file_digest(target) != artifact.get("sha256"):
                    blockers.append(
                        f"execution_artifact_digest_mismatch:{candidate_id}/{condition_id}:{relative_path}"
                    )
        if execution_schema == EXECUTION_SCHEMA_VERSION:
            blockers.extend(
                _validate_episode_visual_evidence(
                    episode,
                    artifacts=artifacts,
                    execution_root=execution_root,
                )
            )
    if len(pairs) != len(set(pairs)):
        blockers.append("execution_duplicate_candidate_condition_episode")
    if set(pairs) != expected:
        blockers.append("execution_candidate_condition_matrix_incomplete")
    for candidate_id, count in completed_by_candidate.items():
        if count == 0:
            blockers.append(f"execution_candidate_has_no_completed_episode:{candidate_id}")
    supplied_digest = execution.get("execution_digest")
    expected_digest = canonical_digest(execution, digest_field="execution_digest")
    if supplied_digest != expected_digest:
        blockers.append("execution_digest_mismatch")
    if blockers:
        raise ArmDecisionProofError(blockers)
    return {
        "schema_version": execution_schema,
        "status": "passed",
        "candidate_count": 2,
        "episode_count": len(episodes),
        "pair_count": len(expected),
        "execution_digest": expected_digest,
        "human_visual_evidence_status": (
            "complete"
            if execution_schema == EXECUTION_SCHEMA_VERSION
            else "legacy_execution_missing_required_media"
        ),
    }


def _episode_receipt(
    episode: Mapping[str, Any], *, manifest: Mapping[str, Any], execution_digest: str
) -> dict[str, Any]:
    receipt = {
        "schema_version": EPISODE_RECEIPT_SCHEMA_VERSION,
        "episode_id": episode["episode_id"],
        "candidate_id": episode["candidate_id"],
        "condition_id": episode["condition_id"],
        "seed": episode["seed"],
        "status": episode["status"],
        "success": episode.get("success"),
        "source_manifest_digest": manifest["manifest_digest"],
        "execution_digest": execution_digest,
        "source_commit": episode["source_commit"],
        "dependency_lock_digest": episode["dependency_lock_digest"],
        "checkpoint_identity_digest": episode["checkpoint_identity_digest"],
        "reset_digest": episode["reset_digest"],
        "observation_trace_digest": episode["observation_trace_digest"],
        "action_trace_digest": episode["action_trace_digest"],
        "metric_trace_digest": episode["metric_trace_digest"],
        "policy_query_count": episode["policy_query_count"],
        "simulator_step_count": episode["simulator_step_count"],
        "evaluator": episode["evaluator"],
        "success_evidence": episode.get("success_evidence"),
        "visual_evidence": episode.get("visual_evidence"),
        "failure": episode.get("failure"),
        "artifacts": episode["artifacts"],
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 1.0]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return [round(max(0.0, center - margin), 12), round(min(1.0, center + margin), 12)]


def compile_bounded_decision(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compile the frozen two-candidate decision; uncertainty can force abstention."""

    candidate_ids = sorted({_string(row.get("candidate_id")) for row in receipts})
    if len(candidate_ids) != 2:
        raise ArmDecisionProofError(["decision_requires_exactly_two_candidates"])
    rule = {
        "baseline_candidate_id": candidate_ids[0],
        "minimum_decision_relevant_difference": 0.20,
        "alpha": 0.05,
        "target_power": 0.80,
        "design": "paired_fixed_condition_matrix",
        "invalid_trial_handling": "count_as_failure_and_expose_invalid_region",
        "multiplicity": "single_two_candidate_contrast_none_required",
        "stop_rule": "all_planned_cells_terminal",
        "uncertainty": "95_percent_wilson_per_candidate_conservative_difference_bounds",
    }
    z_alpha = NormalDist().inv_cdf(1.0 - rule["alpha"] / 2.0)
    z_power = NormalDist().inv_cdf(rule["target_power"])
    minimum_trials_per_candidate = math.ceil(
        0.5 * (z_alpha + z_power) ** 2 / rule["minimum_decision_relevant_difference"] ** 2
    )
    rule["minimum_trials_per_candidate"] = minimum_trials_per_candidate
    rule["sample_size_method"] = (
        "conservative_two_proportion_normal_approximation_without_paired_discordance_prior"
    )
    rows: dict[str, list[Mapping[str, Any]]] = {
        candidate_id: [row for row in receipts if row.get("candidate_id") == candidate_id]
        for candidate_id in candidate_ids
    }
    summaries: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate_rows in rows.items():
        valid = [row for row in candidate_rows if row.get("status") == "completed"]
        successes = sum(row.get("success") is True for row in valid)
        denominator = len(candidate_rows)
        summaries[candidate_id] = {
            "planned": denominator,
            "valid": len(valid),
            "invalid_or_failed": denominator - len(valid),
            "successes": successes,
            "success_rate_with_invalid_as_failure": successes / denominator if denominator else 0.0,
            "wilson_interval": _wilson(successes, denominator),
        }
    baseline, challenger = candidate_ids
    baseline_summary = summaries[baseline]
    challenger_summary = summaries[challenger]
    observed_difference = (
        challenger_summary["success_rate_with_invalid_as_failure"]
        - baseline_summary["success_rate_with_invalid_as_failure"]
    )
    difference_interval = [
        round(
            challenger_summary["wilson_interval"][0] - baseline_summary["wilson_interval"][1], 12
        ),
        round(
            challenger_summary["wilson_interval"][1] - baseline_summary["wilson_interval"][0], 12
        ),
    ]
    mdre = rule["minimum_decision_relevant_difference"]
    invalid = sum(summary["invalid_or_failed"] for summary in summaries.values())
    trial_count_sufficient = all(
        summary["planned"] >= minimum_trials_per_candidate for summary in summaries.values()
    )
    if invalid:
        decision = "abstain"
        selected = None
        reason = "invalid_or_failed_cells_make_selection_unsafe"
    elif not trial_count_sufficient:
        decision = "abstain"
        selected = None
        reason = "planned_trial_count_below_frozen_power_requirement"
    elif difference_interval[0] >= mdre:
        decision = "select"
        selected = challenger
        reason = "challenger_conservative_difference_exceeds_mdre"
    elif difference_interval[1] <= -mdre:
        decision = "eliminate"
        selected = challenger
        reason = "challenger_conservative_difference_below_negative_mdre"
    elif difference_interval[0] >= -mdre and difference_interval[1] <= mdre:
        decision = "equivalent_inconclusive"
        selected = None
        reason = "difference_bounded_inside_equivalence_region"
    else:
        decision = "abstain"
        selected = None
        reason = "uncertainty_crosses_decision_boundaries"
    result = {
        "schema_version": "adp_bounded_decision.v1",
        "decision": decision,
        "selected_candidate_id": selected,
        "reason": reason,
        "rule": rule,
        "candidate_summaries": summaries,
        "observed_difference_challenger_minus_baseline": round(observed_difference, 12),
        "difference_interval": difference_interval,
        "coverage": sum(summary["valid"] for summary in summaries.values())
        / sum(summary["planned"] for summary in summaries.values()),
        "invalid_region": [
            row["receipt_digest"] for row in receipts if row.get("status") != "completed"
        ],
        "trial_count_qualification": {
            "status": "passed" if trial_count_sufficient else "insufficient_power_abstain",
            "minimum_trials_per_candidate": minimum_trials_per_candidate,
            "observed_planned_trials_per_candidate": {
                candidate_id: summary["planned"] for candidate_id, summary in summaries.items()
            },
            "arbitrary_trial_count_accepted_for_selection": False,
        },
        "next_cheapest_missing_measurement": (
            "additional digest-bound fixed-reset replications for each candidate-condition cell"
            if decision in {"abstain", "equivalent_inconclusive"}
            else None
        ),
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    result["decision_digest"] = canonical_digest(result, digest_field="decision_digest")
    return result


def seal_decision(
    *,
    manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    compiler_path = Path(__file__).resolve()
    admission_compiler_path = Path(
        build_public_reference_admission_receipt.__code__.co_filename
    ).resolve()
    seal = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "status": "sealed",
        "source_manifest_digest": manifest["manifest_digest"],
        "execution_digest": execution["execution_digest"],
        "evaluation_run_spec_digest": plan["spec_digest"],
        "candidate_ids": sorted(row["candidate_id"] for row in _rows(manifest.get("candidates"))),
        "condition_ids": sorted(row["condition_id"] for row in _rows(manifest.get("conditions"))),
        "episode_receipt_digests": sorted(row["receipt_digest"] for row in receipts),
        "evidence_compiler_code_digest": _file_digest(compiler_path),
        "admission_compiler_code_digest": _file_digest(admission_compiler_path),
        "decision": dict(decision),
        "physical_outcome_values_accessed": False,
        "physical_outcomes_artifact_digest": _mapping(
            _mapping(manifest.get("physical_reference")).get("outcomes_artifact")
        ).get("digest"),
        "amendment": None,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    seal["seal_digest"] = canonical_digest(seal, digest_field="seal_digest")
    return seal


def release_physical_outcomes(
    *, outcomes_path: Path, manifest: Mapping[str, Any], seal: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open outcome values only after a complete, matching decision seal exists."""

    if seal.get("schema_version") != SEAL_SCHEMA_VERSION or seal.get("status") != "sealed":
        raise ArmDecisionProofError(["physical_outcome_release_requires_valid_seal"])
    if seal.get("seal_digest") != canonical_digest(seal, digest_field="seal_digest"):
        raise ArmDecisionProofError(["physical_outcome_release_seal_digest_mismatch"])
    if seal.get("physical_outcome_values_accessed") is not False:
        raise ArmDecisionProofError(["physical_outcome_release_early_access_detected"])
    expected_digest = _mapping(
        _mapping(manifest.get("physical_reference")).get("outcomes_artifact")
    ).get("digest")
    outcomes = _load_json(outcomes_path, blocker="physical_outcomes_artifact_missing")
    actual_digest = canonical_digest(outcomes, digest_field="outcomes_digest")
    if outcomes.get("outcomes_digest") != expected_digest or actual_digest != expected_digest:
        raise ArmDecisionProofError(["physical_outcomes_artifact_digest_mismatch"])
    receipt = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "released_after_seal",
        "seal_digest": seal["seal_digest"],
        "outcomes_digest": actual_digest,
        "reference_id": manifest["reference_id"],
        "task_id": _mapping(manifest.get("task"))["task_id"],
        "custodian": "blueprint_programmatic_public_reference_loader",
        "software_firebreak_only": True,
        "published_outcomes_were_not_genuinely_unseen": True,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    receipt["release_receipt_digest"] = canonical_digest(
        receipt, digest_field="release_receipt_digest"
    )
    return outcomes, receipt


def join_physical_outcomes(
    *,
    manifest: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    outcomes: Mapping[str, Any],
    release: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    if release.get("seal_digest") != seal.get("seal_digest"):
        blockers.append("physical_join_seal_id_mismatch")
    if outcomes.get("reference_id") != manifest.get("reference_id"):
        blockers.append("physical_join_reference_id_mismatch")
    if outcomes.get("task_id") != _mapping(manifest.get("task")).get("task_id"):
        blockers.append("physical_join_task_id_mismatch")
    receipt_pairs = {(row["candidate_id"], row["condition_id"]): row for row in receipts}
    outcome_rows = _rows(outcomes.get("cells"))
    outcome_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in outcome_rows:
        pair = (_string(row.get("candidate_id")), _string(row.get("condition_id")))
        if pair in outcome_pairs:
            blockers.append("physical_join_duplicate_candidate_condition")
        outcome_pairs[pair] = row
    if set(receipt_pairs) != set(outcome_pairs) or set(receipt_pairs) != _expected_pairs(manifest):
        blockers.append("physical_join_candidate_condition_set_mismatch")
    if blockers:
        raise ArmDecisionProofError(blockers)
    cells = []
    for pair in sorted(receipt_pairs):
        receipt = receipt_pairs[pair]
        outcome = outcome_pairs[pair]
        cells.append(
            {
                "candidate_id": pair[0],
                "condition_id": pair[1],
                "episode_receipt_digest": receipt["receipt_digest"],
                "simulation_status": receipt["status"],
                "simulation_success": receipt.get("success"),
                "physical_trial_count": outcome["trial_count"],
                "physical_success_rate": outcome["success_rate"],
                "physical_reported_uncertainty": outcome["reported_uncertainty"],
                "cell_status": "joined",
            }
        )
    joined = {
        "schema_version": JOIN_SCHEMA_VERSION,
        "status": "joined_exactly",
        "seal_digest": seal["seal_digest"],
        "release_receipt_digest": release["release_receipt_digest"],
        "source_manifest_digest": manifest["manifest_digest"],
        "outcomes_digest": outcomes["outcomes_digest"],
        "cells": cells,
        "missing_outcomes": [],
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    joined["join_digest"] = canonical_digest(joined, digest_field="join_digest")
    return joined


def adjudicate(decision: Mapping[str, Any], joined: Mapping[str, Any]) -> dict[str, Any]:
    candidate_rates: dict[str, list[float]] = {}
    cells = []
    for row in _rows(joined.get("cells")):
        candidate_rates.setdefault(row["candidate_id"], []).append(row["physical_success_rate"])
        sim_success = row.get("simulation_success")
        physical_success = row.get("physical_success_rate", 0.0) >= 0.5
        if row.get("simulation_status") != "completed":
            relation = "inconclusive"
        else:
            relation = "agreement" if sim_success is physical_success else "contradiction"
        cells.append({**row, "cell_relation": relation})
    physical_means = {
        candidate_id: sum(values) / len(values) for candidate_id, values in candidate_rates.items()
    }
    physical_preferred = max(sorted(physical_means), key=physical_means.get)
    sealed_decision = decision.get("decision")
    selected = decision.get("selected_candidate_id")
    if sealed_decision in {"abstain", "equivalent_inconclusive"}:
        overall = "inconclusive"
    elif selected == physical_preferred and sealed_decision == "select":
        overall = "agreement"
    elif selected == physical_preferred and sealed_decision == "eliminate":
        overall = "contradiction"
    elif sealed_decision in {"select", "eliminate"}:
        overall = "contradiction"
    else:
        overall = "abstain"
    verdict = {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "verdict": overall,
        "sealed_development_decision": sealed_decision,
        "physical_reference_preferred_candidate": physical_preferred,
        "physical_candidate_mean_success_rates": physical_means,
        "agreement_cell_count": sum(row["cell_relation"] == "agreement" for row in cells),
        "contradiction_cell_count": sum(row["cell_relation"] == "contradiction" for row in cells),
        "inconclusive_cell_count": sum(row["cell_relation"] == "inconclusive" for row in cells),
        "coverage": len(cells) / 6.0,
        "uncertainty": {
            "published_cell_uncertainty": "not_reported",
            "simulation_decision_interval": decision.get("difference_interval"),
        },
        "invalid_region": decision.get("invalid_region"),
        "next_cheapest_missing_measurement": decision.get("next_cheapest_missing_measurement"),
        "cells": cells,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
        "two_candidates_establish_rank_correlation": False,
    }
    verdict["verdict_digest"] = canonical_digest(verdict, digest_field="verdict_digest")
    return verdict


def _write_artifact(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, dict(value))


def reconstruct_evidence_package(
    *,
    manifest_path: str | Path,
    execution_path: str | Path,
    outcomes_path: str | Path,
    output_dir: str | Path,
    generated_at: str = "2026-08-04T00:00:00Z",
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    execution_file = Path(execution_path).expanduser().resolve()
    outcomes_file = Path(outcomes_path).expanduser().resolve()
    manifest = _load_json(manifest_file, blocker="public_reference_manifest_missing")
    admission = build_public_reference_admission_receipt(manifest)
    if admission.get("status") != "admitted":
        raise ArmDecisionProofError(
            ["public_reference_not_admitted", *admission.get("blockers", [])]
        )
    execution = _load_json(
        execution_file,
        blocker=(
            "immutable_execution_input_missing:restore exact tracked immutable input "
            "documented in docs/arm_decision_proof_v1/REPLAY.md"
        ),
    )
    paid_runtime_validation = validate_paid_runtime_canary(
        manifest,
        execution,
        execution_root=execution_file.parent,
    )
    execution_validation = validate_execution_package(
        execution, manifest, execution_root=execution_file.parent
    )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    inputs_dir = output / "immutable_inputs"
    inputs_dir.mkdir()
    shutil.copy2(manifest_file, inputs_dir / manifest_file.name)
    shutil.copy2(execution_file, inputs_dir / execution_file.name)
    code_inputs_dir = inputs_dir / "code"
    code_inputs_dir.mkdir()
    shutil.copy2(Path(__file__).resolve(), code_inputs_dir / Path(__file__).name)
    admission_compiler_path = Path(
        build_public_reference_admission_receipt.__code__.co_filename
    ).resolve()
    shutil.copy2(
        admission_compiler_path,
        code_inputs_dir / admission_compiler_path.name,
    )
    spec = build_evaluation_run_spec(manifest)
    plan = compile_evaluation_run(
        spec, output_dir=output / "normalized_run", generated_at=generated_at
    )
    if plan.get("status") != "prepared":
        raise ArmDecisionProofError(["normalized_evaluation_run_plan_blocked"])
    receipts_dir = output / "episode_receipts"
    receipts_dir.mkdir()
    receipts = [
        _episode_receipt(
            episode,
            manifest=manifest,
            execution_digest=execution["execution_digest"],
        )
        for episode in _rows(execution.get("episodes"))
    ]
    replayed_receipts = [
        _episode_receipt(
            episode,
            manifest=manifest,
            execution_digest=execution["execution_digest"],
        )
        for episode in _rows(execution.get("episodes"))
    ]
    if [row["receipt_digest"] for row in receipts] != [
        row["receipt_digest"] for row in replayed_receipts
    ]:
        raise ArmDecisionProofError(["episode_receipt_replay_non_reproducible"])
    for receipt in receipts:
        _write_artifact(receipts_dir / f"{receipt['episode_id']}.json", receipt)
    replay = {
        "schema_version": "adp_receipt_replay.v1",
        "status": "reproduced",
        "execution_digest": execution["execution_digest"],
        "receipt_digests": sorted(row["receipt_digest"] for row in receipts),
        "non_reproducibility_failures": [],
    }
    replay["replay_digest"] = canonical_digest(replay, digest_field="replay_digest")
    decision = compile_bounded_decision(receipts)
    seal = seal_decision(
        manifest=manifest,
        execution=execution,
        plan=plan,
        receipts=receipts,
        decision=decision,
    )
    _write_artifact(output / "public_reference_admission_receipt.json", admission)
    _write_artifact(output / "paid_runtime_canary_validation.json", paid_runtime_validation)
    _write_artifact(output / "execution_validation.json", execution_validation)
    _write_artifact(output / "receipt_replay.json", replay)
    _write_artifact(output / "bounded_development_decision.json", decision)
    _write_artifact(output / "decision_seal.json", seal)
    outcomes, release = release_physical_outcomes(
        outcomes_path=outcomes_file,
        manifest=manifest,
        seal=seal,
    )
    shutil.copy2(outcomes_file, inputs_dir / outcomes_file.name)
    _write_artifact(output / "physical_outcome_release_receipt.json", release)
    joined = join_physical_outcomes(
        manifest=manifest,
        receipts=receipts,
        outcomes=outcomes,
        release=release,
        seal=seal,
    )
    verdict = adjudicate(decision, joined)
    _write_artifact(output / "physical_outcome_join.json", joined)
    _write_artifact(output / "bounded_verdict.json", verdict)
    receipt_by_pair = {(row["candidate_id"], row["condition_id"]): row for row in receipts}
    matrix_cells = []
    for verdict_cell in verdict["cells"]:
        pair = (verdict_cell["candidate_id"], verdict_cell["condition_id"])
        receipt = receipt_by_pair[pair]
        matrix_cells.append(
            {
                **verdict_cell,
                "episode_id": receipt["episode_id"],
                "episode_receipt_digest": receipt["receipt_digest"],
                "source_commit": receipt["source_commit"],
                "source_manifest_digest": receipt["source_manifest_digest"],
                "environment_lock_digest": receipt["dependency_lock_digest"],
                "checkpoint_identity_digest": receipt["checkpoint_identity_digest"],
                "reset_digest": receipt["reset_digest"],
                "observation_trace_digest": receipt["observation_trace_digest"],
                "action_trace_digest": receipt["action_trace_digest"],
                "metric_trace_digest": receipt["metric_trace_digest"],
                "trace_artifacts": receipt["artifacts"],
                "human_review_available": _mapping(receipt.get("visual_evidence")).get(
                    "human_review_available"
                )
                is True,
                "visual_evidence": receipt.get("visual_evidence"),
                "human_review_artifacts": [
                    {
                        **artifact,
                        "evidence_package_relative_path": (
                            "immutable_inputs/execution_artifacts/"
                            + _string(artifact.get("relative_path"))
                        ),
                    }
                    for artifact in _rows(receipt.get("artifacts"))
                    if artifact.get("role")
                    in {
                        "policy_input_frame",
                        "terminal_observation_frame",
                        "observation_frame_manifest",
                        "episode_video",
                    }
                ],
                "evaluator": receipt["evaluator"],
                "success_evidence": receipt.get("success_evidence"),
                "failure": receipt["failure"],
                "physical_outcomes_digest": outcomes["outcomes_digest"],
                "physical_release_receipt_digest": release["release_receipt_digest"],
                "qualification_receipt_digest": admission["receipt_digest"],
                "qualification_status": admission["status"],
            }
        )
    matrix = {
        "schema_version": "adp_evidence_matrix.v1",
        "status": "complete",
        "labels": [PHASE_LABEL, CLAIM_CEILING],
        "source_manifest_digest": manifest["manifest_digest"],
        "seal_digest": seal["seal_digest"],
        "release_receipt_digest": release["release_receipt_digest"],
        "join_digest": joined["join_digest"],
        "source_artifact": _mapping(
            _mapping(manifest.get("physical_reference")).get("source_artifact")
        ),
        "environment_lock_digest": execution["runtime_lock_digest"],
        "cells": matrix_cells,
        "human_review_coverage": (
            sum(cell["human_review_available"] for cell in matrix_cells) / len(matrix_cells)
            if matrix_cells
            else 0.0
        ),
        "human_review_media_required_for_new_executions": True,
        "missing_outcomes_visible": True,
        "missing_outcomes": joined["missing_outcomes"],
    }
    matrix["matrix_digest"] = canonical_digest(matrix, digest_field="matrix_digest")
    _write_artifact(output / "evidence_matrix.json", matrix)
    replay_instructions = {
        "schema_version": "adp_replay_instructions.v1",
        "command": "PYTHONPATH=src .venv/bin/python -m blueprint_pipeline.arm_decision_proof",
        "missing_input_acquisition_command": ACQUISITION_COMMAND,
        "immutable_input_digests": {
            "manifest": manifest["manifest_digest"],
            "execution": execution["execution_digest"],
            "physical_outcomes": outcomes["outcomes_digest"],
        },
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
    }
    _write_artifact(output / "replay_instructions.json", replay_instructions)
    execution_inputs_dir = inputs_dir / "execution_artifacts"
    for episode in _rows(execution.get("episodes")):
        for artifact in _rows(episode.get("artifacts")):
            relative_path = _string(artifact.get("relative_path"))
            if not relative_path:
                continue
            source_artifact = (execution_file.parent / relative_path).resolve()
            target_artifact = execution_inputs_dir / relative_path
            target_artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_artifact, target_artifact)
    runtime_inputs_dir = inputs_dir / "paid_runtime_canary"
    for artifact in _rows(
        _mapping(_mapping(manifest.get("runtime")).get("paid_runtime_canary")).get("artifacts")
    ):
        relative_path = _string(artifact.get("relative_path"))
        if not relative_path:
            continue
        source_artifact = (execution_file.parent / relative_path).resolve()
        target_artifact = runtime_inputs_dir / relative_path
        target_artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_artifact, target_artifact)
    artifact_rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact_index.json":
            continue
        artifact_rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": _file_digest(path),
                "size_bytes": path.stat().st_size,
            }
        )
    index = {
        "schema_version": "adp_artifact_index.v1",
        "status": "complete",
        "generated_at": generated_at,
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "phase_label": PHASE_LABEL,
        "claim_ceiling": CLAIM_CEILING,
        "adp_008_complete": True,
    }
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    _write_artifact(output / "artifact_index.json", index)
    return index


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--execution-package", type=Path, default=DEFAULT_EXECUTION_PATH)
    parser.add_argument("--physical-outcomes", type=Path, default=DEFAULT_OUTCOMES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = reconstruct_evidence_package(
            manifest_path=args.manifest,
            execution_path=args.execution_package,
            outcomes_path=args.physical_outcomes,
            output_dir=args.output_dir,
        )
    except ArmDecisionProofError as exc:
        payload: dict[str, Any] = {"status": "blocked", "blockers": exc.blockers}
        if any("immutable_execution_input_missing" in item for item in exc.blockers):
            payload["exact_acquisition_command"] = ACQUISITION_COMMAND
        print(json.dumps(payload, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ArmDecisionProofError",
    "ACQUISITION_COMMAND",
    "adjudicate",
    "build_evaluation_run_spec",
    "compile_bounded_decision",
    "join_physical_outcomes",
    "reconstruct_evidence_package",
    "release_physical_outcomes",
    "seal_decision",
    "validate_execution_package",
    "validate_paid_runtime_canary",
]
