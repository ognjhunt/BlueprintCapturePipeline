from __future__ import annotations

import json
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
import pytest

from blueprint_pipeline.policy_ranking_droid_reference import (
    CHUNK_STARTS,
    COSMOS3_VAE_SPATIAL_FACTOR,
    UPSTREAM_ASSET_SHA256,
    _decoded_dimension,
    _motion_metrics,
    _no_motion_stream,
    amend_reference_canary_geometry,
)
from blueprint_pipeline.policy_ranking_droid_reference_analysis import (
    _decode_concat_view,
    analyze_droid_reference_pair,
)
from blueprint_pipeline.policy_ranking_successor_cosmos import (
    canonical_sha256,
    validate_droid_action_stream,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256
from blueprint_pipeline.wam_rollout_reliability import (
    FLAG_TIMING_EVIDENCE_INSUFFICIENT,
    ReliabilityThresholds,
    SessionReliabilityThresholds,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = (
    ROOT / "docs/experiments/policy_ranking_roboarena_droid_reference_confirmation_20260729"
)


def test_official_droid_reference_source_freeze_is_complete() -> None:
    assert CHUNK_STARTS == (0, 16, 32, 48, 64)
    assert len(UPSTREAM_ASSET_SHA256) == 7
    assert all(len(value) == 64 for value in UPSTREAM_ASSET_SHA256.values())


def test_motion_metrics_use_full_temporal_window() -> None:
    frames = [np.full((4, 5, 3), index, dtype=np.uint8) for index in range(17)]
    metrics = _motion_metrics(frames)
    assert metrics["temporal_absolute_difference_mean_gray_0_255"] == 1.0
    assert metrics["first_to_last_absolute_difference_mean_gray_0_255"] == 16.0


def test_no_motion_uses_valid_identity_rot6d_and_explicit_gripper_hold() -> None:
    stream = validate_droid_action_stream(_no_motion_stream(gripper_hold=0.25))
    assert stream["shape"] == [16, 10]
    assert all(
        row == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25] for row in stream["actions"]
    )


def test_decoded_geometry_matches_pinned_cosmos3_latent_flooring() -> None:
    assert COSMOS3_VAE_SPATIAL_FACTOR == 16
    assert _decoded_dimension(640) == 640
    assert _decoded_dimension(540) == 528
    assert _decoded_dimension(544) == 544
    with pytest.raises(ValueError, match="requires_positive_integers"):
        _decoded_dimension(0)


def test_geometry_amendment_preserves_inputs_and_supersedes_prior_packet(tmp_path: Path) -> None:
    source = tmp_path / "v1"
    source.mkdir()
    initial = source / "initial_observation.png"
    initial.write_bytes(b"frozen-initial-observation")
    actions = {"recorded": {"actions": []}, "no_motion": {"actions": []}}
    (source / "action_streams.json").write_text(json.dumps(actions), encoding="utf-8")
    prior = {
        "schema_version": "policy_ranking_cosmos3_official_droid_reference_canary.v1",
        "request_contract": {"size": "640x540"},
        "provider_inputs": {
            "initial_observation_sha256": file_sha256(initial),
            "action_streams_sha256": canonical_sha256(actions),
        },
        "frozen_gates": {
            "structured_canary": {
                "output_width": 640,
                "output_height": 540,
                "temporal_absolute_difference_mean_minimum_gray_0_255": 1.0,
            }
        },
        "runtime": {"paid_execution_admitted": False, "provider_called": False},
    }
    prior["manifest_sha256"] = canonical_sha256(prior)
    (source / "canary_manifest.json").write_text(json.dumps(prior), encoding="utf-8")

    amended = amend_reference_canary_geometry(source_dir=source, output_dir=tmp_path / "v2")

    assert amended["schema_version"].endswith(".v2")
    assert amended["supersedes"]["manifest_sha256"] == prior["manifest_sha256"]
    assert amended["supersedes"]["prior_paid_output_reclassified"] is False
    gate = amended["frozen_gates"]["structured_canary"]
    assert (gate["output_width"], gate["output_height"]) == (640, 528)
    assert gate["temporal_absolute_difference_mean_minimum_gray_0_255"] == 1.0
    assert (tmp_path / "v2/initial_observation.png").read_bytes() == initial.read_bytes()


def test_protocol_digest_and_paid_state_are_frozen() -> None:
    protocol = json.loads((EXPERIMENT / "protocol_v1.json").read_text(encoding="utf-8"))
    recorded = protocol.pop("manifest_sha256")
    assert recorded == canonical_sha256(protocol)
    assert protocol["paid_execution_admitted"] is False
    assert protocol["provider_called"] is False
    assert protocol["outcome_labels_accessed"] is False
    assert protocol["stage_1_structured_canary"]["maximum_provider_requests"] == 1


def test_protocol_v2_provider_amendment_is_unpaid_and_scientifically_unchanged() -> None:
    protocol = json.loads((EXPERIMENT / "protocol_v2.json").read_text(encoding="utf-8"))
    recorded = protocol.pop("manifest_sha256")

    assert recorded == canonical_sha256(protocol)
    assert protocol["supersedes"]["protocol_v1_rewritten"] is False
    assert protocol["amendment_timing"]["paid_provider_allocation_created"] is False
    assert protocol["amendment_timing"]["model_provider_called"] is False
    unchanged = protocol["unchanged_scientific_contract"]
    assert unchanged["recorded_action_request_maximum"] == 1
    assert unchanged["no_motion_request_maximum"] == 1
    assert unchanged["untouched_data_not_admitted"] is True
    assert protocol["runpod_preflight"]["provider_mutations_performed"] == 0


def test_protocol_v4_geometry_amendment_is_prospective_and_does_not_reclassify() -> None:
    protocol = json.loads((EXPERIMENT / "protocol_v4.json").read_text(encoding="utf-8"))
    recorded = protocol.pop("manifest_sha256")

    assert recorded == canonical_sha256(protocol)
    assert protocol["supersedes"]["protocol_v3_rewritten"] is False
    timing = protocol["amendment_timing"]
    assert timing["allocation_2_output_accessed"] is True
    assert timing["allocation_2_output_retroactively_reclassified"] is False
    assert timing["replacement_allocation_provider_called"] is False
    changes = {row["field"]: row for row in protocol["changed_fields"]}
    assert changes["structured_canary_decoded_output_height"]["new"] == 528
    unchanged = protocol["unchanged_scientific_contract"]
    assert unchanged["request_size"] == "640x540"
    assert unchanged["motion_thresholds_unchanged"] is True
    assert unchanged["same_output_may_be_retroactively_passed"] is False


def test_protocol_v5_provider_retry_changes_no_scientific_field() -> None:
    protocol = json.loads((EXPERIMENT / "protocol_v5.json").read_text(encoding="utf-8"))
    recorded = protocol.pop("manifest_sha256")

    assert recorded == canonical_sha256(protocol)
    assert protocol["supersedes"]["protocol_v4_rewritten"] is False
    assert protocol["allocation_3_closeout"]["provider_mutations_performed"] == 0
    assert protocol["allocation_3_closeout"]["provider_generation_requests_attempted"] == 0
    unchanged = protocol["unchanged_contract"]
    assert unchanged["protocol_v4_scientific_fields_unchanged"] is True
    assert unchanged["bundle_sha256"] == (
        "d8378dda5c21757c35cb010506615cdb2886c11fbe4c6c9dbd97ff7aef8b044f"
    )
    assert unchanged["motion_thresholds_unchanged"] is True
    assert protocol["replacement_provider_called"] is False


def test_allocation_2_result_preserves_failed_gate_and_provider_zero() -> None:
    result = json.loads(
        (EXPERIMENT / "allocation_2_geometry_gate_result_v1.json").read_text(encoding="utf-8")
    )

    assert result["output"]["motion_gate_passed"] is True
    assert result["output"]["frozen_geometry_gate_passed"] is False
    assert result["adjudication"]["cosmos_wam_qualification_credit"] is False
    assert result["adjudication"]["same_output_may_be_retroactively_reclassified_as_pass"] is False
    assert result["provider_zero"]["task_inventory_live_count"] == 0
    assert result["provider_zero"]["continuing_hourly_burn"] is False


def _write_concat_video(path: Path, *, wrist_step: int, left_step: int, right_step: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (60, 48))
    assert writer.isOpened()
    try:
        for index in range(17):
            frame = np.zeros((48, 60, 3), dtype=np.uint8)
            frame[4:12, (index * wrist_step) % 48 : (index * wrist_step) % 48 + 8] = 255
            frame[36:44, (index * left_step) % 22 : (index * left_step) % 22 + 8] = 180
            offset = 30 + (index * right_step) % 22
            frame[36:44, offset : offset + 8] = 220
            writer.write(frame)
    finally:
        writer.release()


def test_reference_pair_analysis_is_view_attributable_and_fails_closed(tmp_path: Path) -> None:
    recorded = tmp_path / "recorded.mp4"
    no_motion = tmp_path / "no_motion.mp4"
    _write_concat_video(recorded, wrist_step=2, left_step=2, right_step=2)
    _write_concat_video(no_motion, wrist_step=1, left_step=0, right_step=0)
    views = _decode_concat_view(recorded)
    assert {name: frames[0].shape[:2] for name, frames in views.items()} == {
        "wrist": (32, 60),
        "left": (16, 30),
        "right": (16, 30),
    }
    active = np.zeros((16, 10), dtype=np.float64)
    active[:, 3] = active[:, 7] = 1.0
    active[:, 0] = np.linspace(0.0, 0.1, 16)
    null = np.zeros((16, 10), dtype=np.float64)
    null[:, 3] = null[:, 7] = 1.0

    report = analyze_droid_reference_pair(
        recorded_video=recorded,
        no_motion_video=no_motion,
        recorded_actions=active,
        no_motion_actions=null,
        reliability_thresholds=ReliabilityThresholds(),
        session_thresholds=SessionReliabilityThresholds(
            timing_correlation_min=0.15,
            minimum_eligible_timing_windows=3,
        ),
        session_id="fixture-session",
    )

    assert set(report["view_comparison"]) == {"wrist", "left", "right"}
    assert report["abstain"] is True
    assert FLAG_TIMING_EVIDENCE_INSUFFICIENT in report["abstention_reasons"]
    assert report["cosmos_wam_qualification_credit"] is False
    recorded_report = report["recorded_rollout_reliability"]
    assert recorded_report["timing_flag_scope"] == "session"
    assert len(report["analysis_sha256"]) == 64
