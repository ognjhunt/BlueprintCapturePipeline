from __future__ import annotations

import json
from pathlib import Path

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
from blueprint_pipeline.policy_ranking_successor_cosmos import (
    canonical_sha256,
    validate_droid_action_stream,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256


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
