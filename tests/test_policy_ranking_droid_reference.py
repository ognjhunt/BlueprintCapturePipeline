from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from blueprint_pipeline.policy_ranking_droid_reference import (
    CHUNK_STARTS,
    UPSTREAM_ASSET_SHA256,
    _motion_metrics,
    _no_motion_stream,
)
from blueprint_pipeline.policy_ranking_successor_cosmos import (
    canonical_sha256,
    validate_droid_action_stream,
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
