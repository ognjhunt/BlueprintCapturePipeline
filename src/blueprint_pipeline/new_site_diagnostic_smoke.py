"""Prospective freeze and fail-closed gates for a new-site diagnostic smoke lane.

This module deliberately does not launch providers.  It freezes the experiment,
records arm canary attempts, and admits complete episodes only when one
label-free canary passes observation, motion, and collapse checks.  Paid
execution remains owned by :mod:`blueprint_pipeline.paid_resource_allocator`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import ensure_dir, write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


EXPERIMENT_ID = "policy_ranking_new_site_smoke_interiorgs_0787_20260729_v1"
SCHEMA_VERSION = "policy_ranking_new_site_diagnostic_smoke_protocol.v1"
CANARY_SCHEMA_VERSION = "policy_ranking_new_site_diagnostic_canary.v1"
RANKING_SCHEMA_VERSION = "policy_ranking_new_site_diagnostic_ranking.v1"
SCENE_SPEC = Path(
    "docs/experiments/policy_ranking_thesis_20260726/"
    "interiorgs_0787_hybrid_scene_spec.json"
)
POLICY_COHORT = Path(
    "docs/experiments/policy_ranking_thesis_20260726/"
    "warehouse_policy_cohort_v2_joint_position.json"
)
CONFIRMATION_SPLIT = Path(
    "docs/experiments/policy_ranking_roboarena_powered_droid_confirmation_20260729/"
    "disjoint_session_candidate_split_amendment_v3.json"
)
RELIABILITY_FREEZE = Path(
    "docs/experiments/policy_ranking_roboarena_full_stack_calibration_20260728/"
    "phase_b_rollout_reliability_gate_v1.json"
)
VARIANTS = ("center", "left_2cm", "right_2cm")
ARM_IDS = ("skeleton_only", "oscar", "ctrl_world")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _identity(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = canonical_sha256(result)
    return result


def build_protocol(
    repo_root: str | Path,
    local_scene_asset: str | Path | None = None,
    *,
    experiment_id: str = EXPERIMENT_ID,
    parent_protocol: str | Path | Mapping[str, Any] | None = None,
    disclosures: Sequence[str] = (),
    scene_reference_manifest: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    scene_path = root / SCENE_SPEC
    cohort_path = root / POLICY_COHORT
    split_path = root / CONFIRMATION_SPLIT
    reliability_path = root / RELIABILITY_FREEZE
    scene = _read(scene_path)
    cohort = _read(cohort_path)
    split = _read(split_path)
    reliability = _read(reliability_path)
    lineage = _build_lineage(experiment_id, parent_protocol, disclosures)
    scene_reference = _scene_reference(scene_reference_manifest)
    policies = list(cohort.get("primary_cohort") or [])[:3]
    if len(policies) != 3:
        raise ValueError("three_manifest_order_policies_unavailable")
    reserved = list((split.get("selection") or {}).get("session_ids") or [])
    if len(reserved) != 17:
        raise ValueError("confirmation_reserved_session_count_invalid")
    if str(scene.get("scene_id") or "") in reserved:
        raise ValueError("diagnostic_scene_reserved_for_confirmation")
    threshold_block = (reliability.get("window_gate") or {}).get("thresholds")
    if not isinstance(threshold_block, Mapping):
        raise ValueError("reliability_threshold_freeze_missing")
    selected_policies = []
    for policy in policies:
        selected_policies.append(
            {
                key: policy[key]
                for key in (
                    "policy_id",
                    "checkpoint",
                    "checkpoint_object_count",
                    "checkpoint_size_bytes",
                    "public_object_manifest_sha256",
                    "generation_manifest_sha256",
                    "action_horizon",
                )
            }
        )
    scene_input: dict[str, Any] | None = None
    if local_scene_asset is not None:
        scene_asset_path = Path(local_scene_asset).expanduser().resolve()
        if not scene_asset_path.is_file() or scene_asset_path.stat().st_size <= 0:
            raise ValueError("local_scene_asset_unavailable")
        scene_input = {
            "local_path": str(scene_asset_path),
            "representation": "decoded_standard_3dgs_ply",
            "bytes": scene_asset_path.stat().st_size,
            "sha256": file_sha256(scene_asset_path),
            "derived_from_source_sha256": scene["site_visual"]["sha256"],
            "authoritative_source_replacement": False,
        }
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "status": "prospective_offline_freeze_paid_execution_not_admitted",
        "diagnostic_namespace": experiment_id,
        "lineage": lineage,
        "source_commit": _git_head(root),
        "implementation": {
            "files": [
                {
                    "path": "src/blueprint_pipeline/new_site_diagnostic_smoke.py",
                    "sha256": file_sha256(Path(__file__).resolve()),
                },
                {
                    "path": "src/blueprint_pipeline/splat_scene_render.py",
                    "sha256": file_sha256(root / "src/blueprint_pipeline/splat_scene_render.py"),
                },
                {
                    "path": "src/blueprint_pipeline/franka_droid_skeleton_conditioning.py",
                    "sha256": file_sha256(
                        root / "src/blueprint_pipeline/franka_droid_skeleton_conditioning.py"
                    ),
                },
                {
                    "path": "src/blueprint_pipeline/ctrl_world_droid_action_adapter.py",
                    "sha256": file_sha256(
                        root / "src/blueprint_pipeline/ctrl_world_droid_action_adapter.py"
                    ),
                },
                {
                    "path": "src/blueprint_pipeline/new_site_diagnostic_canary_gpu.py",
                    "sha256": file_sha256(
                        root
                        / "src/blueprint_pipeline/new_site_diagnostic_canary_gpu.py"
                    ),
                },
                {
                    "path": "src/blueprint_pipeline/openpi_policy_ranking_gpu_bootstrap.py",
                    "sha256": file_sha256(
                        root
                        / "src/blueprint_pipeline/openpi_policy_ranking_gpu_bootstrap.py"
                    ),
                },
                {
                    "path": "src/blueprint_pipeline/openpi_policy_ranking_gpu_admission.py",
                    "sha256": file_sha256(
                        root
                        / "src/blueprint_pipeline/openpi_policy_ranking_gpu_admission.py"
                    ),
                },
                {
                    "path": "src/blueprint_pipeline/openpi_policy_ranking_runpod.py",
                    "sha256": file_sha256(
                        root / "src/blueprint_pipeline/openpi_policy_ranking_runpod.py"
                    ),
                },
            ],
            "uncommitted_implementation_permitted_for_diagnostic_only": True,
        },
        "scene": {
            "scene_id": scene["scene_id"],
            "task_instruction": scene["task_semantics"]["instruction"],
            "source_revision": scene["site_visual"]["source_revision"],
            "source_sha256": scene["site_visual"]["sha256"],
            "scene_spec_path": SCENE_SPEC.as_posix(),
            "scene_spec_sha256": file_sha256(scene_path),
            "previously_exposed_development_scene": True,
            "independent_confirmation_scene": False,
            "local_scene_input": scene_input,
            "supporting_reference": scene_reference,
        },
        "policy_freeze": {
            "selection_rule": "first_three_primary_cohort_entries_in_manifest_order",
            "selection_used_prior_rankings": False,
            "cohort_path": POLICY_COHORT.as_posix(),
            "cohort_sha256": file_sha256(cohort_path),
            "policies": selected_policies,
            "variants": list(VARIANTS),
        },
        "confirmation_exclusion": {
            "split_path": CONFIRMATION_SPLIT.as_posix(),
            "split_sha256": file_sha256(split_path),
            "reserved_session_ids": reserved,
            "diagnostic_scene_outside_reserved_split": True,
            "confirmation_namespace_write_forbidden": True,
        },
        "arms": {
            "skeleton_only": {
                "requested": True,
                "role": "intended_motion_baseline_only",
                "fresh_label_free_canary_required": True,
                "world_consequence_credit": False,
                "production_conditioning_builder": (
                    "FrankaDroidSkeletonConditioningBuilder"
                ),
                "camera_aligned_external_and_live_wrist": True,
                "task_physics_or_future_rgb_used": False,
            },
            "oscar": {
                "requested": True,
                "role": "purpose_built_wam",
                "fresh_label_free_canary_required": True,
                "camera_aligned_external_and_wrist_required": True,
                "production_conditioning_builder": (
                    "FrankaDroidSkeletonConditioningBuilder"
                ),
                "multiview_generator_wiring": "CallableMultiViewOscarWamArm",
            },
            "ctrl_world": {
                "requested": True,
                "role": "nonprimary_comparator",
                "fresh_label_free_canary_required": True,
                "normalized_7d_cartesian_eef_adapter_validation_required": True,
                "joint_position_adapter": (
                    "FrankaCtrlWorldJointPositionAdapter"
                ),
                "conversion": "deterministic_pinned_franka_forward_kinematics",
                "official_velocity_adapter_misuse_forbidden": True,
                "world_model_checkpoint_admission_separate": True,
            },
            "cosmos": {
                "requested_if_already_valid": True,
                "admitted_arm_id": None,
                "reason": "no_captured_site_polaris_cosmos_arm_currently_qualified",
                "native_cosmos_unchanged_rerun_forbidden": True,
                "generic_terminal_diagnostic_upload_fix_present": True,
            },
        },
        "canary_rule": {
            "one_attempt_per_arm_per_version": True,
            "label_free": True,
            "frozen_policy_id": selected_policies[0]["policy_id"],
            "frozen_variant": "center",
            "same_scene_task_policy_and_center_variant": True,
            "advance_requires_all": [
                "observation_validity_passed",
                "motion_passed",
                "collapse_checks_passed",
            ],
            "failed_arm_complete_episode_execution_forbidden": True,
        },
        "reliability_freeze": {
            "source_path": RELIABILITY_FREEZE.as_posix(),
            "source_sha256": file_sha256(reliability_path),
            "thresholds": dict(threshold_block),
            "immediate_hard_failure_flags": list(
                (reliability.get("window_gate") or {}).get(
                    "immediate_hard_failure_flags", []
                )
            ),
        },
        "ranking_rule": {
            "separate_ranking_per_arm": True,
            "cross_arm_score_fusion_forbidden": True,
            "score": "mean_over_three_frozen_variants",
            "uncertainty": "exact_min_max_over_three_frozen_variants",
            "pairwise_order": "strict_nonoverlapping_intervals_only",
            "otherwise": "abstain",
        },
        "media_contract": {
            "individual_camera_videos_required": ["external", "wrist"],
            "failure_media_retained": True,
            "non_cherry_picked_gallery": True,
            "interiorgs_redistribution_forbidden": True,
        },
        "execution": {
            "offline_preparation_may_run_in_parallel": True,
            "paid_gpu_maximum_concurrency": 1,
            "paid_launcher": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
            "exclusive_campaign_lease_required": True,
            "watchdog_ttl_teardown_and_provider_zero_required": True,
        },
        "no_tuning": {
            "rankings_may_not_change": [
                "prompts",
                "thresholds",
                "adapters",
                "wam_settings",
                "policy_selection",
                "camera_selection",
            ],
            "engineering_fix_requires_generic_regression_test": True,
            "every_rerun_requires_new_version_and_disclosure": True,
            "prior_evidence_overwrite_forbidden": True,
        },
        "claim_boundary": {
            "ranking_accuracy": False,
            "physical_success": False,
            "captured_site_transfer_validation": False,
            "phase_b_confirmation": False,
            "independently_attributable_matching_physical_outcomes_present": False,
            "result_type": "bounded_nonconfirmatory_technical_diagnostic",
        },
        "paid_execution_admitted": False,
        "provider_called": False,
    }
    return _identity(protocol, "protocol_sha256")


def _scene_reference(manifest_path: str | Path | None) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    path = Path(manifest_path).expanduser().resolve()
    manifest = _read(path)
    if manifest.get("status") != "completed" or manifest.get("nonblank_camera_count", 0) < 2:
        raise ValueError("diagnostic_scene_reference_not_completed")
    cameras = []
    for camera in manifest.get("cameras") or []:
        camera_path = Path(str(camera.get("path") or "")).expanduser().resolve()
        if not camera_path.is_file() or camera.get("nonblank") is not True:
            raise ValueError("diagnostic_scene_reference_camera_invalid")
        cameras.append(
            {
                "id": camera["id"],
                "path": str(camera_path),
                "bytes": camera_path.stat().st_size,
                "sha256": file_sha256(camera_path),
            }
        )
    mp4_path = Path(str((manifest.get("mp4") or {}).get("mp4") or "")).expanduser().resolve()
    if not mp4_path.is_file():
        raise ValueError("diagnostic_scene_reference_mp4_invalid")
    return {
        "manifest_path": str(path),
        "manifest_sha256": file_sha256(path),
        "rendered_by": manifest.get("rendered_by"),
        "proof_boundary": dict(manifest.get("proof_boundary") or {}),
        "cameras": cameras,
        "mp4": {
            "path": str(mp4_path),
            "bytes": mp4_path.stat().st_size,
            "sha256": file_sha256(mp4_path),
        },
    }


def _build_lineage(
    experiment_id: str,
    parent_protocol: str | Path | Mapping[str, Any] | None,
    disclosures: Sequence[str],
) -> dict[str, Any]:
    match = re.fullmatch(r"(.+)_v([1-9][0-9]*)", experiment_id)
    if match is None:
        raise ValueError("diagnostic_experiment_version_invalid")
    family, version_text = match.groups()
    version = int(version_text)
    normalized_disclosures = [str(item).strip() for item in disclosures if str(item).strip()]
    if version == 1:
        if parent_protocol is not None or normalized_disclosures:
            raise ValueError("diagnostic_initial_version_must_not_have_parent")
        return {
            "version": 1,
            "parent_experiment_id": None,
            "parent_protocol_sha256": None,
            "disclosures": [],
        }
    if parent_protocol is None or not normalized_disclosures:
        raise ValueError("diagnostic_successor_requires_parent_and_disclosure")
    if isinstance(parent_protocol, Mapping):
        parent = dict(parent_protocol)
        parent_file_sha256 = None
    else:
        parent_path = Path(parent_protocol).expanduser().resolve()
        parent = _read(parent_path)
        parent_file_sha256 = file_sha256(parent_path)
    parent_id = str(parent.get("experiment_id") or "")
    expected_parent_id = f"{family}_v{version - 1}"
    if parent_id != expected_parent_id:
        raise ValueError("diagnostic_parent_version_not_immediate_predecessor")
    parent_sha256 = str(parent.get("protocol_sha256") or "")
    if len(parent_sha256) != 64:
        raise ValueError("diagnostic_parent_protocol_identity_missing")
    return {
        "version": version,
        "parent_experiment_id": parent_id,
        "parent_protocol_sha256": parent_sha256,
        "parent_protocol_file_sha256": parent_file_sha256,
        "disclosures": normalized_disclosures,
    }


def _git_head(root: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assess_canary(protocol: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    arm_id = str(evidence.get("arm_id") or "")
    if arm_id not in ARM_IDS:
        raise ValueError("diagnostic_canary_arm_invalid")
    if evidence.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError("diagnostic_canary_protocol_binding_mismatch")
    if evidence.get("label_free") is not True:
        raise ValueError("diagnostic_canary_not_label_free")
    if evidence.get("ranking_outputs_accessed") is not False:
        raise ValueError("diagnostic_canary_ranking_access_not_false")
    attempt_stage = str(evidence.get("attempt_stage") or "")
    if attempt_stage not in {"paid_admission", "arm_preflight", "rollout"}:
        raise ValueError("diagnostic_canary_attempt_stage_invalid")
    model_invoked = evidence.get("model_invoked")
    if not isinstance(model_invoked, bool):
        raise ValueError("diagnostic_canary_model_invoked_not_boolean")
    expected_bindings = {
        "scene_id": protocol["scene"]["scene_id"],
        "task_instruction": protocol["scene"]["task_instruction"],
        "policy_id": protocol["canary_rule"]["frozen_policy_id"],
        "variant": protocol["canary_rule"]["frozen_variant"],
    }
    if any(evidence.get(key) != value for key, value in expected_bindings.items()):
        raise ValueError("diagnostic_canary_freeze_binding_mismatch")
    checks = {
        "observation_validity_passed": evidence.get("observation_validity_passed") is True,
        "motion_passed": evidence.get("motion_passed") is True,
        "collapse_checks_passed": evidence.get("collapse_checks_passed") is True,
    }
    blockers = [str(item) for item in (evidence.get("blockers") or [])]
    if model_invoked is False and any(checks.values()):
        raise ValueError("diagnostic_canary_check_pass_without_model_invocation")
    passed = all(checks.values()) and not blockers
    if passed and model_invoked is not True:
        raise ValueError("diagnostic_canary_pass_without_model_invocation")
    status = "passed" if passed else "failed"
    if not model_invoked and blockers:
        status = "blocked_before_model"
    result: dict[str, Any] = {
        "schema_version": CANARY_SCHEMA_VERSION,
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "arm_id": arm_id,
        "status": status,
        "label_free": True,
        "attempt_stage": attempt_stage,
        "model_invoked": model_invoked,
        "freeze_bindings": expected_bindings,
        "checks": checks,
        "blockers": blockers,
        "advance_to_complete_episodes": passed,
        "failed_arm_complete_episode_execution_forbidden": not passed,
        "evidence_sha256": canonical_sha256(dict(evidence)),
        "claim_boundary": "technical arm admission only; not ranking or physical evidence",
    }
    return _identity(result, "canary_sha256")


def build_ranking(
    protocol: Mapping[str, Any],
    canary: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]],
    evidence_root: str | Path,
) -> dict[str, Any]:
    if canary.get("protocol_sha256") != protocol.get("protocol_sha256"):
        raise ValueError("ranking_canary_protocol_binding_mismatch")
    if canary.get("advance_to_complete_episodes") is not True:
        raise ValueError("ranking_forbidden_for_failed_canary")
    arm_id = str(canary.get("arm_id") or "")
    frozen_policies = [
        item["policy_id"] for item in protocol["policy_freeze"]["policies"]
    ]
    grouped: dict[str, list[float]] = {policy: [] for policy in frozen_policies}
    seen_variants: dict[str, set[str]] = {policy: set() for policy in frozen_policies}
    missing_media: list[str] = []
    invalid_media: list[str] = []
    root = Path(evidence_root).expanduser().resolve()
    for row in episodes:
        if row.get("arm_id") != arm_id:
            raise ValueError("ranking_cross_arm_episode_forbidden")
        policy_id = str(row.get("policy_id") or "")
        variant = str(row.get("variant") or "")
        score = row.get("score")
        if policy_id not in grouped or variant not in VARIANTS:
            raise ValueError("ranking_episode_outside_freeze")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("ranking_episode_score_invalid")
        if variant in seen_variants[policy_id]:
            raise ValueError("ranking_duplicate_policy_variant")
        seen_variants[policy_id].add(variant)
        grouped[policy_id].append(float(score))
        media = row.get("camera_videos")
        if not isinstance(media, Mapping) or any(
            not media.get(view) for view in ("external", "wrist")
        ):
            missing_media.append(f"{policy_id}:{variant}")
        else:
            for view in ("external", "wrist"):
                ref = media[view]
                if not isinstance(ref, Mapping) or not _valid_artifact_ref(root, ref):
                    invalid_media.append(f"{policy_id}:{variant}:{view}")
        failure_detected = row.get("failure_detected") is True
        failure_media = row.get("failure_media")
        if failure_detected and (
            not isinstance(failure_media, Sequence)
            or isinstance(failure_media, (str, bytes))
            or not failure_media
            or any(
                not isinstance(ref, Mapping) or not _valid_artifact_ref(root, ref)
                for ref in failure_media
            )
        ):
            invalid_media.append(f"{policy_id}:{variant}:failure")
    if len(episodes) != 9 or any(len(values) != 3 for values in grouped.values()):
        raise ValueError("ranking_requires_three_policies_by_three_variants")
    if any(seen_variants[policy] != set(VARIANTS) for policy in frozen_policies):
        raise ValueError("ranking_requires_each_frozen_variant_once")
    summaries = []
    for policy_id, values in grouped.items():
        summaries.append(
            {
                "policy_id": policy_id,
                "mean_score": sum(values) / len(values),
                "uncertainty_min": min(values),
                "uncertainty_max": max(values),
                "variant_count": len(values),
            }
        )
    summaries.sort(key=lambda item: (-item["mean_score"], item["policy_id"]))
    separated = all(
        summaries[index]["uncertainty_min"] > summaries[index + 1]["uncertainty_max"]
        for index in range(len(summaries) - 1)
    )
    abstention_reasons = []
    if not separated:
        abstention_reasons.append("frozen_variant_intervals_overlap")
    if missing_media:
        abstention_reasons.append("individual_camera_media_incomplete")
    if invalid_media:
        abstention_reasons.append("media_integrity_invalid")
    ranking: dict[str, Any] = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "canary_sha256": canary["canary_sha256"],
        "arm_id": arm_id,
        "policy_summaries": summaries,
        "ordered_policy_ids": [item["policy_id"] for item in summaries],
        "strict_total_ranking_emitted": separated and not missing_media and not invalid_media,
        "abstained": not separated or bool(missing_media) or bool(invalid_media),
        "abstention_reasons": abstention_reasons,
        "missing_camera_media": missing_media,
        "invalid_media": invalid_media,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    return _identity(ranking, "ranking_sha256")


def _valid_artifact_ref(root: Path, reference: Mapping[str, Any]) -> bool:
    relative = reference.get("path")
    expected_sha256 = reference.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        return False
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        return False
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path.is_file() and path.stat().st_size > 0 and file_sha256(path) == expected_sha256


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"diagnostic_evidence_overwrite_forbidden:{path}")
    ensure_dir(path.parent)
    write_json(path, dict(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--repo-root", default=".")
    freeze.add_argument("--local-scene-asset")
    freeze.add_argument("--experiment-id", default=EXPERIMENT_ID)
    freeze.add_argument("--parent-protocol")
    freeze.add_argument("--disclosure", action="append", default=[])
    freeze.add_argument("--scene-reference-manifest")
    freeze.add_argument("--output", required=True)
    canary = sub.add_parser("canary")
    canary.add_argument("--protocol", required=True)
    canary.add_argument("--evidence", required=True)
    canary.add_argument("--output", required=True)
    rank = sub.add_parser("rank")
    rank.add_argument("--protocol", required=True)
    rank.add_argument("--canary", required=True)
    rank.add_argument("--episodes", required=True)
    rank.add_argument("--evidence-root", required=True)
    rank.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = build_protocol(
            args.repo_root,
            args.local_scene_asset,
            experiment_id=args.experiment_id,
            parent_protocol=args.parent_protocol,
            disclosures=args.disclosure,
            scene_reference_manifest=args.scene_reference_manifest,
        )
    elif args.command == "canary":
        payload = assess_canary(_read(Path(args.protocol)), _read(Path(args.evidence)))
    else:
        episode_value = json.loads(Path(args.episodes).read_text(encoding="utf-8"))
        if not isinstance(episode_value, list):
            raise ValueError("ranking_episode_list_required")
        payload = build_ranking(
            _read(Path(args.protocol)),
            _read(Path(args.canary)),
            episode_value,
            args.evidence_root,
        )
    _write_new(Path(args.output).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["assess_canary", "build_protocol", "build_ranking"]
