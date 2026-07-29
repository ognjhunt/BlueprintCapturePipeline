from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline.policy_ranking_disjoint_preflight import (
    build_disjoint_technical_preflight,
    main,
)
from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


EXPERIMENT_DOCS = (
    Path(__file__).resolve().parents[1]
    / "docs/experiments/policy_ranking_roboarena_disjoint_reasoner_successor_20260728"
)


def _npz(path: Path, steps: int = 17) -> None:
    rows = [
        {
            "cartesian_position": [index / 100, 0, 0, 0, 0, 0],
            "joint_position": [0.0] * 7,
            "gripper_position": [0.0],
            "action": [0.0] * 8,
        }
        for index in range(steps)
    ]
    np.savez(path, data=np.asarray(rows, dtype=object))


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(3):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    policy_dir = root / "evaluation_sessions" / "session-1" / "A_policy-1"
    policy_dir.mkdir(parents=True)
    (policy_dir.parent / "metadata.yaml").write_text("outcome: DO_NOT_OPEN\n", encoding="utf-8")
    _npz(policy_dir / "policy-1_npz_file.npz")
    for view in ("left", "right", "wrist"):
        _video(policy_dir / f"policy-1_video_{view}.mp4")
    split = {
        "schema_version": "policy_ranking_disjoint_session_candidate_split.v1",
        "dataset": {"id": "fixture", "revision": "abc", "license": "mit"},
        "required_policy_ids": ["policy-1"],
        "selection": {"metadata_yaml_opened": False, "session_ids": ["session-1"]},
    }
    split["manifest_sha256"] = canonical_sha256(split)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    return split_path, root


def test_disjoint_preflight_passes_without_opening_metadata(tmp_path: Path, monkeypatch) -> None:
    split_path, root = _fixture(tmp_path)
    original_read_text = Path.read_text

    def guarded_read_text(path: Path, *args, **kwargs):
        assert path.name != "metadata.yaml"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    report = build_disjoint_technical_preflight(split_manifest_path=split_path, dataset_root=root)
    assert report["status"] == "passed"
    assert report["passed_row_count"] == 1
    assert report["view_counts"] == {"left": 1, "right": 1, "wrist": 1}
    assert report["label_seal"]["outcome_labels_accessed"] is False
    assert report["claim_ceiling"]["policy_ranking_or_wam_qualification"] is False


def test_disjoint_preflight_fails_closed_on_missing_required_view(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    (root / "evaluation_sessions/session-1/A_policy-1/policy-1_video_right.mp4").unlink()
    report = build_disjoint_technical_preflight(split_manifest_path=split_path, dataset_root=root)
    assert report["status"] == "blocked"
    assert report["passed_row_count"] == 0
    assert any("video_resolution:right:0" in blocker for blocker in report["blockers"])


def test_disjoint_preflight_rejects_tampered_split(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["selection"]["session_ids"].append("session-2")
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        build_disjoint_technical_preflight(split_manifest_path=split_path, dataset_root=root)
    except ValueError as exc:
        assert str(exc) == "split_manifest_sha256_mismatch"
    else:
        raise AssertionError("tampered split was accepted")


def test_disjoint_preflight_cli_writes_output(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    output = tmp_path / "nested" / "report.json"
    assert (
        main(
            [
                "--split-manifest",
                str(split_path),
                "--dataset-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"


def test_disjoint_preflight_accepts_prospective_v2_split_amendment(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "policy_ranking_disjoint_session_candidate_split_amendment.v2"
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_sha256(payload)
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_disjoint_technical_preflight(split_manifest_path=split_path, dataset_root=root)
    assert report["status"] == "passed"


def test_disjoint_preflight_accepts_prospective_v3_split_amendment(tmp_path: Path) -> None:
    split_path, root = _fixture(tmp_path)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "policy_ranking_disjoint_session_candidate_split_amendment.v3"
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = canonical_sha256(payload)
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    report = build_disjoint_technical_preflight(split_manifest_path=split_path, dataset_root=root)
    assert report["status"] == "passed"


def test_committed_disjoint_preflight_artifact_digests_are_canonical() -> None:
    for filename in (
        "disjoint_session_candidate_split_amendment_v2.json",
        "disjoint_technical_preflight_summary_v1.json",
        "phase_b_native_cosmos_environment_v1.json",
        "phase_b_native_cosmos_canary_preparation_v1.json",
        "phase_b_open_loop_replay_protocol_v1.json",
        "phase_b_high_motion_native_cosmos_result_v1.json",
        "phase_b_high_motion_native_cosmos_cost_provider_zero_v1.json",
        "phase_b_high_motion_native_cosmos_gallery_manifest_v1.json",
        "phase_b_positive_control_bisection_amendment_v1.json",
        "phase_b_positive_control_download_transport_amendment_v2.json",
        "phase_b_positive_control_bisection_result_v2.json",
        "phase_b_terminal_admission_v1.json",
        "terminal_verdict_v1.json",
        "terminal_verdict_v2.json",
    ):
        payload = json.loads((EXPERIMENT_DOCS / filename).read_text(encoding="utf-8"))
        recorded = payload.pop("manifest_sha256")
        assert recorded == canonical_sha256(payload)


def test_committed_native_cosmos_result_preserves_failure_and_claim_ceiling() -> None:
    result = json.loads(
        (EXPERIMENT_DOCS / "phase_b_high_motion_native_cosmos_result_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "completed_causal_screen_failed_product_abstention"
    assert result["causal_screen"]["screen_passed"] is False
    assert result["tier_1_rollout_reliability"]["session_reliable"] is False
    assert result["abstention"]["product_abstention_required"] is True
    assert result["abstention"]["native_cosmos_full_episode_matrix_admitted"] is False
    assert result["phase_b_design_achieved"].startswith("fallback_level_3")
    assert result["label_unseal"]["remaining_selected_sessions_still_label_sealed"] == 16


def test_committed_native_cosmos_cost_receipt_proves_zero_without_overclaiming() -> None:
    receipt = json.loads(
        (
            EXPERIMENT_DOCS / "phase_b_high_motion_native_cosmos_cost_provider_zero_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["gpu"]["estimated_cost_usd"] == 0.088401
    assert receipt["gpu"]["continuing_spend_from_this_run"] is False
    assert receipt["provider_zero"]["task_live_instance_count"] == 0
    assert receipt["provider_zero"]["all_live_instance_count"] == 0
    assert (
        receipt["provider_zero"][
            "provider_zero_does_not_prove_invoice_settlement_or_scientific_validity"
        ]
        is True
    )
    assert "not a completed ranking cost" in receipt["claim_boundary"]


def test_terminal_admission_does_not_promote_short_screen_to_phase_b() -> None:
    admission = json.loads(
        (EXPERIMENT_DOCS / "phase_b_terminal_admission_v1.json").read_text(encoding="utf-8")
    )
    assert admission["status"] == "terminal_not_admitted"
    assert admission["paid_execution_admitted"] is False
    assert admission["short_horizon_screen_interpretation"]["generated_frames"] == 17
    assert admission["current_frozen_evidence"]["selected_sessions_still_label_sealed"] == 16
    assert admission["decision"]["additional_gpu_or_evaluator_spend_admitted"] is False
    assert admission["superseded_for_future_execution_by"]["artifact"] == (
        "phase_b_positive_control_bisection_amendment_v1.json"
    )
    assert "No live closed-loop" in admission["claim_ceiling"]


def test_download_transport_amendment_preserves_science_and_minimizes_credentials() -> None:
    amendment = json.loads(
        (
            EXPERIMENT_DOCS / "phase_b_positive_control_download_transport_amendment_v2.json"
        ).read_text(encoding="utf-8")
    )
    failed = amendment["failed_infrastructure_attempt"]
    replacement = amendment["replacement_runtime_changes"]
    assert failed["provider_generation_requests_attempted"] == 0
    assert failed["provider_zero_proven"] is True
    assert failed["scientific_result_produced"] is False
    assert replacement["disable_huggingface_xet_before_model_library_import"] is True
    assert replacement["forward_huggingface_token_to_provider"] is False
    assert amendment["unchanged_scientific_contract"]["outcome_labels_accessed"] is False


def test_terminal_verdict_keeps_all_components_separate() -> None:
    verdict = json.loads((EXPERIMENT_DOCS / "terminal_verdict_v1.json").read_text(encoding="utf-8"))
    assert verdict["overall_verdict"] == "inconclusive"
    assert set(verdict["components"]) == {
        "cosmos_wam_qualification",
        "frozen_benchmark_calibration",
        "captured_site_transfer",
        "economics_and_speed",
    }
    assert verdict["components"]["cosmos_wam_qualification"]["qualified"] is False
    assert verdict["phase_b"]["full_episode_executed"] is False
    assert verdict["phase_b"]["policy_requery_executed"] is False
    assert verdict["phase_b"]["selected_sessions_remaining_label_sealed"] == 16
    assert (
        verdict["abstention"]["native_cosmos_product_abstention_correct_for_frozen_gates"] is True
    )
    assert verdict["cost_and_time"]["completed_useful_digital_ranking"] is False
    assert verdict["provider_zero"]["all_live_instance_count"] == 0


def test_terminal_verdict_v2_preserves_positive_control_and_droid_failure() -> None:
    verdict = json.loads((EXPERIMENT_DOCS / "terminal_verdict_v2.json").read_text(encoding="utf-8"))
    assert verdict["overall_verdict"] == "inconclusive"
    assert set(verdict["components"]) == {
        "cosmos_wam_qualification",
        "frozen_benchmark_calibration",
        "captured_site_transfer",
        "economics_and_speed",
    }
    assert verdict["components"]["cosmos_wam_qualification"]["qualified"] is False
    assert verdict["phase_b"]["policy_requery_executed"] is False
    assert verdict["phase_b"]["ranking_executed"] is False
    assert (
        verdict["abstention"]["native_cosmos_product_abstention_correct_for_frozen_gates"] is True
    )
    assert verdict["cost_and_time"]["completed_useful_digital_ranking"] is False
    assert verdict["provider_zero"]["all_live_instance_count"] == 0
