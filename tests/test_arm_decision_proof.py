from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.arm_decision_proof import (
    ACQUISITION_COMMAND,
    ArmDecisionProofError,
    join_physical_outcomes,
    main,
    reconstruct_evidence_package,
    release_physical_outcomes,
    validate_execution_package,
    validate_paid_runtime_canary,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.simpler_public_runtime_worker import (
    _finalize_visual_evidence,
    _write_observation_png,
)


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = (
    ROOT
    / "docs"
    / "arm_decision_proof_v1"
    / "manifests"
    / "simpler_google_robot_pick_coke_can.v1.json"
)
OUTCOMES_PATH = MANIFEST_PATH.with_name(
    "simpler_google_robot_pick_coke_can_physical_outcomes.v1.json"
)
IMMUTABLE_EXECUTION_ROOT = MANIFEST_PATH.parents[1] / "immutable_execution"


def _admitted_manifest() -> dict:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    value.pop("manifest_digest")
    value["runtime"]["environment_lock"] = {
        "status": "exact_immutable",
        "digest": "sha256:" + "4" * 64,
        "container_image": "nvidia/cuda@sha256:" + "1" * 64,
    }
    value["runtime"].pop("paid_runtime_canary", None)
    value["runtime"]["zero_spend_feasibility"]["status"] = "passed"
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
    return value


def _execution(manifest: dict) -> dict:
    candidates = []
    checkpoint_digests = {}
    for candidate in manifest["candidates"]:
        digest = canonical_digest(
            {
                "candidate_id": candidate["candidate_id"],
                "checkpoint_prefix": candidate["checkpoint_prefix"],
                "checkpoint_objects": candidate["checkpoint_objects"],
            }
        )
        checkpoint_digests[candidate["candidate_id"]] = digest
        candidates.append(
            {
                "candidate_id": candidate["candidate_id"],
                "checkpoint_identity_digest": digest,
                "genuine_checkpoint_loaded": True,
            }
        )
    episodes = []
    for candidate_index, candidate in enumerate(manifest["candidates"]):
        for condition in manifest["conditions"]:
            candidate_id = candidate["candidate_id"]
            condition_id = condition["condition_id"]
            episode_id = f"episode-{candidate_index}-{condition_id}"
            episodes.append(
                {
                    "episode_id": episode_id,
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "seed": 0,
                    "status": "completed",
                    "success": candidate_index == 1,
                    "source_commit": manifest["source"]["repository"]["commit"],
                    "dependency_lock_digest": manifest["runtime"]["environment_lock"]["digest"],
                    "checkpoint_identity_digest": checkpoint_digests[candidate_id],
                    "reset_digest": "sha256:" + "5" * 64,
                    "observation_trace_digest": "sha256:" + "6" * 64,
                    "action_trace_digest": "sha256:" + "7" * 64,
                    "metric_trace_digest": "sha256:" + "8" * 64,
                    "policy_query_count": 10,
                    "simulator_step_count": 10,
                    "evaluator": {
                        "owner": "environment_not_policy",
                        "policy_self_report_used": False,
                    },
                    "artifacts": [{"role": "trace", "sha256": "sha256:" + "9" * 64}],
                }
            )
    value = {
        "schema_version": "simpler_closed_loop_execution.v1",
        "status": "completed",
        "reference_id": manifest["reference_id"],
        "source_identity_digest": manifest["source_identity_digest"],
        "source_manifest_digest": manifest["manifest_digest"],
        "runtime_lock_digest": manifest["runtime"]["environment_lock"]["digest"],
        "candidates": candidates,
        "episodes": episodes,
        "physical_outcome_values_accessed": False,
        "phase_label": "retrospective_external_reference",
        "claim_ceiling": "development_only",
    }
    value["execution_digest"] = canonical_digest(value, digest_field="execution_digest")
    return value


def _add_required_visual_evidence(execution: dict, output_root: Path) -> None:
    for index, episode in enumerate(execution["episodes"]):
        image = np.full((32, 48, 3), index * 20, dtype=np.uint8)
        policy_frame = _write_observation_png(
            image,
            output_dir=output_root,
            episode_id=episode["episode_id"],
            frame_index=0,
            kind="policy-input",
        )
        terminal_frame = _write_observation_png(
            image,
            output_dir=output_root,
            episode_id=episode["episode_id"],
            frame_index=1,
            kind="terminal-observation",
        )
        visual, artifacts = _finalize_visual_evidence(
            output_dir=output_root,
            episode_id=episode["episode_id"],
            identity={
                "candidate_id": episode["candidate_id"],
                "condition_id": episode["condition_id"],
                "seed": episode["seed"],
            },
            policy_input_frames=[policy_frame],
            terminal_observation=terminal_frame,
            frames_per_second=3.0,
        )
        episode["policy_query_count"] = 1
        episode["simulator_step_count"] = 1
        episode["observation_trace_digest"] = canonical_digest(
            {"observations": [policy_frame["raw_rgb_sha256"]]}
        )
        episode["evaluator"].update(
            {
                "grader_type": "deterministic_simulator_state",
                "success_source": "environment_step_info.success",
                "vlm_used": False,
                "human_grade_used": False,
            }
        )
        episode["success_evidence"] = {
            "grader_type": "deterministic_simulator_state",
            "source_field": "environment_step_info.success",
            "final_value": episode["success"],
            "vlm_used": False,
            "human_grade_used": False,
            "policy_self_report_used": False,
        }
        episode["visual_evidence"] = visual
        episode["artifacts"].extend(artifacts)
    execution["schema_version"] = "simpler_closed_loop_execution.v2"
    execution["execution_digest"] = canonical_digest(execution, digest_field="execution_digest")


def test_full_reconstruction_seals_before_release_and_renders_every_cell(
    tmp_path: Path,
) -> None:
    manifest = _admitted_manifest()
    execution = _execution(manifest)
    manifest_path = tmp_path / "manifest.json"
    execution_path = tmp_path / "execution.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    execution_path.write_text(json.dumps(execution), encoding="utf-8")

    index = reconstruct_evidence_package(
        manifest_path=manifest_path,
        execution_path=execution_path,
        outcomes_path=OUTCOMES_PATH,
        output_dir=tmp_path / "evidence",
    )

    output = tmp_path / "evidence"
    seal = json.loads((output / "decision_seal.json").read_text())
    release = json.loads((output / "physical_outcome_release_receipt.json").read_text())
    matrix = json.loads((output / "evidence_matrix.json").read_text())
    verdict = json.loads((output / "bounded_verdict.json").read_text())
    assert index["adp_008_complete"] is True
    assert seal["physical_outcome_values_accessed"] is False
    assert release["seal_digest"] == seal["seal_digest"]
    assert release["published_outcomes_were_not_genuinely_unseen"] is True
    assert len(matrix["cells"]) == 6
    assert matrix["labels"] == ["retrospective_external_reference", "development_only"]
    assert matrix["cells"][0]["reset_digest"].startswith("sha256:")
    assert (
        matrix["cells"][0]["physical_release_receipt_digest"] == release["release_receipt_digest"]
    )
    assert matrix["cells"][0]["qualification_status"] == "admitted"
    assert matrix["human_review_coverage"] == 0.0
    assert matrix["human_review_media_required_for_new_executions"] is True
    assert verdict["sealed_development_decision"] == "abstain"
    assert verdict["verdict"] == "inconclusive"
    decision = json.loads((output / "bounded_development_decision.json").read_text())
    assert decision["trial_count_qualification"]["status"] == ("insufficient_power_abstain")
    assert (
        decision["trial_count_qualification"]["arbitrary_trial_count_accepted_for_selection"]
        is False
    )


def test_duplicate_candidate_identity_is_never_padding() -> None:
    manifest = _admitted_manifest()
    execution = _execution(manifest)
    execution["candidates"][1] = copy.deepcopy(execution["candidates"][0])
    execution["execution_digest"] = canonical_digest(execution, digest_field="execution_digest")

    with pytest.raises(ArmDecisionProofError) as caught:
        validate_execution_package(execution, manifest)

    assert "execution_duplicate_candidate_identity" in caught.value.blockers
    assert "execution_candidate_set_mismatch" in caught.value.blockers


def test_v2_execution_requires_and_validates_human_visual_evidence(
    tmp_path: Path,
) -> None:
    manifest = _admitted_manifest()
    execution = _execution(manifest)
    _add_required_visual_evidence(execution, tmp_path)

    validation = validate_execution_package(
        execution,
        manifest,
        execution_root=tmp_path,
    )

    assert validation["human_visual_evidence_status"] == "complete"
    first = execution["episodes"][0]
    assert first["visual_evidence"]["human_review_available"] is True
    video_path = tmp_path / first["visual_evidence"]["video"]["relative_path"]
    assert video_path.read_bytes()[4:8] == b"ftyp"


def test_v2_execution_rejects_missing_episode_video(tmp_path: Path) -> None:
    manifest = _admitted_manifest()
    execution = _execution(manifest)
    _add_required_visual_evidence(execution, tmp_path)
    first = execution["episodes"][0]
    first["artifacts"] = [row for row in first["artifacts"] if row.get("role") != "episode_video"]
    execution["execution_digest"] = canonical_digest(execution, digest_field="execution_digest")

    with pytest.raises(ArmDecisionProofError) as caught:
        validate_execution_package(
            execution,
            manifest,
            execution_root=tmp_path,
        )

    assert any(
        blocker.startswith("execution_episode_video_binding_invalid:")
        for blocker in caught.value.blockers
    )


def test_paid_runtime_canary_rejects_changed_provider_receipt(tmp_path: Path) -> None:
    execution_root = tmp_path / "immutable_execution"
    shutil.copytree(IMMUTABLE_EXECUTION_ROOT, execution_root)
    teardown_path = execution_root / "provider_evidence" / "vast_teardown_manifest.json"
    teardown = json.loads(teardown_path.read_text(encoding="utf-8"))
    teardown["continuing_spend_from_this_run"] = True
    teardown_path.write_text(json.dumps(teardown), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    execution = json.loads(
        (execution_root / "adp_simpler_closed_loop_execution.json").read_text(encoding="utf-8")
    )

    with pytest.raises(ArmDecisionProofError) as caught:
        validate_paid_runtime_canary(
            manifest,
            execution,
            execution_root=execution_root,
        )

    assert "paid_runtime_artifact_digest_mismatch:teardown" in caught.value.blockers
    assert "paid_runtime_teardown_not_proven" in caught.value.blockers


def test_outcome_loader_rejects_before_reading_without_a_valid_seal(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArmDecisionProofError, match="physical_outcome_release_requires_valid_seal"):
        release_physical_outcomes(
            outcomes_path=tmp_path / "does-not-exist.json",
            manifest=_admitted_manifest(),
            seal={},
        )


def test_join_rejects_changed_condition_or_seal() -> None:
    manifest = _admitted_manifest()
    execution = _execution(manifest)
    receipts = []
    for episode in execution["episodes"]:
        row = dict(episode)
        row["receipt_digest"] = "sha256:" + "a" * 64
        receipts.append(row)
    outcomes = json.loads(OUTCOMES_PATH.read_text(encoding="utf-8"))
    outcomes["cells"][0]["condition_id"] = "changed-condition"
    seal = {"seal_digest": "sha256:" + "b" * 64}
    release = {
        "seal_digest": seal["seal_digest"],
        "release_receipt_digest": "sha256:" + "c" * 64,
    }

    with pytest.raises(
        ArmDecisionProofError, match="physical_join_candidate_condition_set_mismatch"
    ):
        join_physical_outcomes(
            manifest=manifest,
            receipts=receipts,
            outcomes=outcomes,
            release=release,
            seal=seal,
        )


def test_missing_execution_input_returns_exact_acquisition_instruction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _admitted_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status = main(
        [
            "--manifest",
            str(manifest_path),
            "--execution-package",
            str(tmp_path / "missing.json"),
            "--physical-outcomes",
            str(OUTCOMES_PATH),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert status == 2
    output = capsys.readouterr().out
    assert "restore exact tracked immutable input" in output
    assert ACQUISITION_COMMAND in output
