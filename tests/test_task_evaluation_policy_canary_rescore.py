from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.task_evaluation_policy_canary_rescore import (
    PolicyCanaryRescoreError,
    rescore_policy_canary_result,
    validate_policy_canary_score_correction,
)


RUN_ID = (
    "scene839873-artifixer-corrective-68d36be3-r13-web-20260901T031158Z-"
    "policy-canary-abe19c87-5997-4c7c-aedf-6d10fb6abd27"
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _task_spec() -> dict[str, object]:
    return {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "rigid_pick_place",
        "subject_asset_id": "cup",
        "start_pose_world": [1.0, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        "destination_position_bounds_world_m": {
            "minimum": [1.14, 1.99, 0.79],
            "maximum": [1.16, 2.01, 0.81],
        },
        "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "destination_orientation_tolerance_rad": 0.1,
        "support_height_interval_m": [0.79, 0.81],
        "minimum_translation_m": 0.14,
        "minimum_lift_m": 0.02,
        "movement_epsilon_m": 0.001,
        "reset_translation_tolerance_m": 0.001,
        "reset_orientation_tolerance_rad": 0.01,
        "settle_window_samples": 3,
        "settle_position_tolerance_m": 0.002,
        "settle_orientation_tolerance_rad": 0.01,
        "release_required": True,
        "release_gripper_width_min_m": 0.07,
        "task_contact_minimum_force_n": 0.5,
    }


def _sample(step: int, position: list[float]) -> dict[str, object]:
    return {
        "step_index": step,
        "task_object_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
        "gripper_width_m": 0.08 if step >= 3 else 0.04,
        "task_contact_active": step < 3,
        "support_contact_active": step >= 3,
        "containment_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "forbidden_robot_task_collision_failure": False,
        "locked_joint_containment_violation": False,
    }


def _scorer_identity() -> dict[str, object]:
    sources = [{"path": "adp_task_scoring.py", "sha256": "sha256:" + "1" * 64}]
    value: dict[str, object] = {
        "schema_version": "task_evaluation_deterministic_scorer_identity.v1",
        "scorer": "blueprint_pipeline.adp_task_scoring.score_task_episode_from_spec",
        "scorer_commit": "a" * 40,
        "source_files": sources,
        "source_files_digest": canonical_digest({"value": sources}),
        "scoring_version_digest": "",
    }
    value["scoring_version_digest"] = cross_runtime_canonical_digest(
        value, digest_field="scoring_version_digest"
    )
    return value


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    task_spec = _task_spec()
    samples = [
        _sample(0, [1.0, 2.0, 0.8]),
        _sample(1, [1.0, 2.0, 0.83]),
        _sample(2, [1.15, 2.0, 0.83]),
        _sample(3, [1.15, 2.0, 0.8]),
        _sample(4, [1.15, 2.0, 0.8]),
        _sample(5, [1.15, 2.0, 0.8]),
    ]
    new_score = score_task_episode_from_spec(task_spec=task_spec, samples=samples)
    old_score = copy.deepcopy(new_score)
    old_score["task_succeeded"] = False
    old_score["outcome"] = "moved_below_threshold"
    old_score["report_digest"] = canonical_digest(old_score, digest_field="report_digest")
    episodes: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(("pi05_droid", "groot_n17_droid")):
        for cell_index in range(10):
            episode_id = f"episode-{candidate_index}-{cell_index}"
            state_trace = {"task_state_samples": samples}
            state_record = _write_json(evidence / f"{episode_id}.state_trace.json", state_trace)
            state_record["role"] = "state_trace"
            score_record = _write_json(evidence / f"{episode_id}.score_receipt.json", old_score)
            score_record["role"] = "score_receipt"
            inventory.extend((state_record, score_record))
            raw_episode: dict[str, object] = {
                "candidate_id": candidate,
                "episode_id": episode_id,
                "task_spec": task_spec,
                "task_spec_digest": canonical_digest(task_spec),
                "state_trace": state_trace,
                "score": old_score,
                "receipt_digest": "",
            }
            raw_episode["receipt_digest"] = canonical_digest(
                raw_episode, digest_field="receipt_digest"
            )
            parity: dict[str, object] = {
                "schema_version": "droid_policy_canary_embodiment_parity.v1",
                "status": "passed",
                "diagnostic_only": True,
                "receipt_digest": "",
            }
            parity["receipt_digest"] = canonical_digest(parity, digest_field="receipt_digest")
            # The provider worker adds this independently sealed diagnostic
            # after it seals the policy-episode receipt.
            raw_episode["embodiment_parity_diagnostic"] = parity
            episodes.append(
                {
                    "run_kind": "internal_policy_canary",
                    "claim_ceiling": "diagnostic_policy_execution",
                    "ranking_eligible": False,
                    "candidate_id": candidate,
                    "cell_id": f"quick10.{cell_index:02d}",
                    "seed": 1000 + cell_index,
                    "family": "held_out_composition" if cell_index == 9 else "stress",
                    "status": "completed",
                    "candidate_policy_queried": True,
                    "actions_reached_robot": True,
                    "arm_moved": True,
                    "checkpoint_digest": "sha256:" + str(candidate_index + 1) * 64,
                    "runtime_identity_digest": "sha256:" + "3" * 64,
                    "lossless_frame_manifest_digest": "sha256:" + "4" * 64,
                    "review_video_digest": "sha256:" + "5" * 64,
                    "returned_action_sequence_digest": "sha256:" + "6" * 64,
                    "action_delivery_readback_digest": "sha256:" + "7" * 64,
                    "state_trace_digest": canonical_digest({"value": state_trace}),
                    "contact_force_digest": "sha256:" + "8" * 64,
                    "task_object_trajectory_digest": "sha256:" + "9" * 64,
                    "deterministic_score_digest": canonical_digest({"value": old_score}),
                    "scoring_authority": "deterministic_simulator_state",
                    "scoring_version_digest": "sha256:" + "b" * 64,
                    "visual_evidence": {},
                    "evidence_artifacts": {
                        "state_trace": state_record,
                        "score_receipt": score_record,
                    },
                    "episode": raw_episode,
                }
            )
    result: dict[str, object] = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "completed_unqualified",
        "run_id": RUN_ID,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "episodes_per_policy": 10,
        "learned_policy_rollout_count": 20,
        "retry_cap": 0,
        "warm_session_open_count": 1,
        "provider_allocations_observed": 1,
        "scene_promotion_performed": False,
        "official_ranking_performed": False,
        "episodes": episodes,
        "session_closeout": {
            "provider_allocations_observed": 1,
            "teardown_completed": True,
            "provider_zero_confirmed": True,
        },
        "artifact_inventory": inventory,
        "artifact_inventory_digest": canonical_digest({"value": inventory}),
        "result_digest": "",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    source = tmp_path / "policy_canary_terminal_result.json"
    source.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source, evidence, result


def test_rescore_writes_overlay_and_preserves_original_bytes(tmp_path: Path) -> None:
    source, evidence, original = _fixture(tmp_path)
    source_bytes = source.read_bytes()
    original_score_bytes = sorted(evidence.glob("*.score_receipt.json"))[0].read_bytes()

    correction = rescore_policy_canary_result(
        source_result_path=source,
        evidence_root=evidence,
        output_root=tmp_path / "corrections",
        expected_run_id=RUN_ID,
        scorer_identity=_scorer_identity(),
    )

    assert correction["episode_count"] == 20
    assert correction["source_result_digest"] == original["result_digest"]
    assert correction["source_artifact_inventory_digest"] == original["artifact_inventory_digest"]
    assert all(row["new_score"]["task_succeeded"] is True for row in correction["score_updates"])
    assert all(
        row["old_score_digest"] != row["new_score_digest"] for row in correction["score_updates"]
    )
    assert source.read_bytes() == source_bytes
    assert sorted(evidence.glob("*.score_receipt.json"))[0].read_bytes() == original_score_bytes
    correction_root = tmp_path / "corrections" / str(correction["correction_id"])
    assert len(list((correction_root / "episodes").glob("*.json"))) == 20
    assert (correction_root / "score_correction.json").is_file()


def test_rescore_fails_before_output_when_artifact_bytes_changed(tmp_path: Path) -> None:
    source, evidence, _ = _fixture(tmp_path)
    sorted(evidence.glob("*.state_trace.json"))[0].write_text("{}\n", encoding="utf-8")

    with pytest.raises(PolicyCanaryRescoreError, match="artifact_digest_mismatch"):
        rescore_policy_canary_result(
            source_result_path=source,
            evidence_root=evidence,
            output_root=tmp_path / "corrections",
            expected_run_id=RUN_ID,
            scorer_identity=_scorer_identity(),
        )

    assert not (tmp_path / "corrections").exists()


def test_rescore_rejects_old_score_digest_disagreement(tmp_path: Path) -> None:
    source, evidence, result = _fixture(tmp_path)
    row = result["episodes"][0]
    row["deterministic_score_digest"] = "sha256:" + "f" * 64
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    source.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PolicyCanaryRescoreError, match="old_score_digest_mismatch"):
        rescore_policy_canary_result(
            source_result_path=source,
            evidence_root=evidence,
            output_root=tmp_path / "corrections",
            expected_run_id=RUN_ID,
            scorer_identity=_scorer_identity(),
        )


def test_publication_validator_rejects_added_mutation_fields(tmp_path: Path) -> None:
    source, evidence, result = _fixture(tmp_path)
    correction = rescore_policy_canary_result(
        source_result_path=source,
        evidence_root=evidence,
        output_root=tmp_path / "corrections",
        expected_run_id=RUN_ID,
        scorer_identity=_scorer_identity(),
    )
    correction_root = tmp_path / "corrections" / str(correction["correction_id"])
    mutated = copy.deepcopy(correction)
    mutated["score_updates"][0]["candidate_id_replacement"] = "forbidden"
    mutated["correction_digest"] = cross_runtime_canonical_digest(
        mutated, digest_field="correction_digest"
    )

    with pytest.raises(PolicyCanaryRescoreError, match="update_fields_invalid"):
        validate_policy_canary_score_correction(
            source_result=result,
            correction=mutated,
            receipt_root=correction_root,
        )
