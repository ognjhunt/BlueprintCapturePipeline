from __future__ import annotations

import copy

import numpy as np
import pytest

from blueprint_pipeline.adp009d_live_hybrid_observation import (
    HYBRID_RUNTIME_RECEIPT_SCHEMA_VERSION,
    ISAAC_CAMERA_BACKEND,
    LiveHybridObservationError,
    compose_live_hybrid_observation,
    validate_live_hybrid_runtime_receipt,
)
from blueprint_pipeline.adp009d_aura_renderer_conformance import (
    FROZEN_THRESHOLDS,
    RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.adp009d_aura_native_conformance import (
    RECEIPT_SCHEMA_VERSION as NATIVE_RECEIPT_SCHEMA_VERSION,
)
from blueprint_pipeline.adp009d_aura_native_vast import (
    EXPECTED_AURA_PLY_SHA256,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TREE,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _calibration() -> dict:
    return {
        "camera_model": "pinhole",
        "intrinsic_matrix": [
            [40.0, 0.0, 1.0],
            [0.0, 40.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        "world_from_camera": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "resolution": [2, 2],
    }


def _arrays() -> dict:
    return {
        "aura_rgb": np.full((2, 2, 3), 10, dtype=np.uint8),
        "aura_depth_m": np.full((2, 2), 1.0, dtype=np.float32),
        "dynamic_rgb": np.full((2, 2, 3), 100, dtype=np.uint8),
        "dynamic_depth_m": np.array([[0.5, 1.5], [2.0, 2.0]], dtype=np.float32),
        "dynamic_segmentation": np.array([[7, 7], [0, 0]], dtype=np.int32),
        "dynamic_alpha": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32),
    }


def test_metric_depth_composition_preserves_front_and_static_occlusion() -> None:
    composed, receipt = compose_live_hybrid_observation(
        **_arrays(),
        aura_calibration=_calibration(),
        isaac_calibration=_calibration(),
        timestamp_ns=123,
        simulation_time_s=1.25,
        dynamic_depth_aov="DistanceToCameraSD",
        semantic_labels={7: "robot"},
        semantic_override_layer_digest="sha256:" + "a" * 64,
    )

    assert composed[0, 0].tolist() == [100, 100, 100]
    assert composed[0, 1].tolist() == [10, 10, 10]
    assert receipt["dynamic_front_pixel_count"] == 1
    assert receipt["dynamic_occluded_pixel_count"] == 1
    assert receipt["visual_judgment_used_for_success"] is False
    assert receipt["live_execution_proven_by_this_function"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_metric_depth_composition_accepts_isaac_camera_aov() -> None:
    composed, receipt = compose_live_hybrid_observation(
        **_arrays(),
        aura_calibration=_calibration(),
        isaac_calibration=_calibration(),
        timestamp_ns=123,
        simulation_time_s=1.25,
        dynamic_depth_aov="distance_to_camera",
        semantic_labels={7: "robot"},
        semantic_override_layer_digest="sha256:" + "a" * 64,
    )

    assert composed.shape == (2, 2, 3)
    assert receipt["dynamic_depth_aov"] == "distance_to_camera"


def test_composition_rejects_unitless_depth_or_mismatched_camera() -> None:
    changed = _calibration()
    changed["world_from_camera"][0][3] = 0.01

    with pytest.raises(
        LiveHybridObservationError,
        match="hybrid_camera_calibration_mismatch.*hybrid_metric_dynamic_depth_aov_required",
    ):
        compose_live_hybrid_observation(
            **_arrays(),
            aura_calibration=_calibration(),
            isaac_calibration=changed,
            timestamp_ns=123,
            simulation_time_s=1.25,
            dynamic_depth_aov="DepthSD",
            semantic_labels={7: "robot"},
            semantic_override_layer_digest="sha256:" + "a" * 64,
        )


def test_composition_rejects_visible_semantic_pixel_without_metric_depth() -> None:
    arrays = _arrays()
    arrays["dynamic_depth_m"][0, 0] = np.nan

    with pytest.raises(
        LiveHybridObservationError, match="hybrid_dynamic_metric_depth_missing"
    ):
        compose_live_hybrid_observation(
            **arrays,
            aura_calibration=_calibration(),
            isaac_calibration=_calibration(),
            timestamp_ns=123,
            simulation_time_s=1.25,
            dynamic_depth_aov="DistanceToImagePlaneSD",
            semantic_labels={7: "robot"},
            semantic_override_layer_digest="sha256:" + "a" * 64,
        )


def _runtime_receipt() -> dict:
    conformance = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "passed_exact_camera_conformance",
        "passed": True,
        "thresholds": FROZEN_THRESHOLDS,
        "thresholds_frozen_before_ovrtx_execution": True,
        "ovrtx_repository": "https://github.com/NVIDIA-Omniverse/ovrtx",
        "ovrtx_revision": "4b9a5fe6f8becf6c5ff031e167cd4201054a96ce",
    }
    conformance["receipt_digest"] = canonical_digest(
        conformance, digest_field="receipt_digest"
    )
    value = {
        "schema_version": HYBRID_RUNTIME_RECEIPT_SCHEMA_VERSION,
        "status": "executed_live_renderer_microcheck",
        "backend": "OVRTX",
        "initialization_order": ["OVRTX", "OvPhysX"],
        "render_settings_target": "RenderProduct",
        "metric_depth_aov": "DistanceToCameraSD",
        "unitless_depth_sd_used": False,
        "attached_mode_ordinals_respected": True,
        "write_floors_respected": True,
        "semantic_source_usd_mutated": False,
        "semantic_override_layer_composed": True,
        "camera_or_settings_change_reset": True,
        "dlpack_ownership_explicit": True,
        "map_unmap_balanced": True,
        "device_synchronization_explicit": True,
        "rtpt_warmup_frames": 40,
        "rtpt_warmup_change_reason": None,
        "path_tracing_used": False,
        "path_tracing_samples_per_pixel": None,
        "camera_ids": ["external", "wrist"],
        "observed_frame_count": 4,
        "frame_receipt_digests": ["sha256:" + "b" * 64],
        "policy_frames_retained_losslessly": True,
        "aura_renderer_conformance_receipt": conformance,
        "camera_motion_occlusion_probe_passed": True,
        "static_occlusion_probe_passed": True,
        "moving_occlusion_probe_passed": True,
    }
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    return value


def test_live_runtime_receipt_requires_executed_ovrtx_evidence() -> None:
    receipt = _runtime_receipt()
    assert validate_live_hybrid_runtime_receipt(receipt) == receipt

    mutated = copy.deepcopy(receipt)
    mutated["status"] = "prepared"
    mutated["semantic_source_usd_mutated"] = True
    mutated["rtpt_warmup_frames"] = 12
    mutated["receipt_digest"] = canonical_digest(
        mutated, digest_field="receipt_digest"
    )
    with pytest.raises(
        LiveHybridObservationError,
        match=(
            "hybrid_runtime_rtpt_warmup_below_documented_default.*"
            "hybrid_runtime_sealed_source_mutated.*"
            "sealed_aura_hybrid_policy_observation_renderer_missing"
        ),
    ):
        validate_live_hybrid_runtime_receipt(mutated)


def test_live_runtime_receipt_rejects_missing_aura_renderer_conformance() -> None:
    receipt = _runtime_receipt()
    receipt.pop("aura_renderer_conformance_receipt")
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    with pytest.raises(
        LiveHybridObservationError,
        match="hybrid_runtime_aura_exact_camera_conformance_missing",
    ):
        validate_live_hybrid_runtime_receipt(receipt)


def test_live_runtime_receipt_accepts_selected_isaac_camera_evidence() -> None:
    receipt = _runtime_receipt()
    receipt.update(
        {
            "backend": ISAAC_CAMERA_BACKEND,
            "initialization_order": None,
            "render_settings_target": None,
            "metric_depth_aov": "distance_to_camera",
            "attached_mode_ordinals_respected": None,
            "write_floors_respected": None,
            "dlpack_ownership_explicit": None,
            "map_unmap_balanced": None,
            "rtpt_warmup_frames": None,
            "rtpt_warmup_change_reason": None,
            "camera_data_types": [
                "rgb",
                "distance_to_camera",
                "semantic_segmentation",
            ],
            "camera_calibration_retained": True,
            "camera_timestamps_retained": True,
            "camera_warmup_frames": 40,
        }
    )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    assert validate_live_hybrid_runtime_receipt(receipt) == receipt

    native_conformance = {
        "schema_version": NATIVE_RECEIPT_SCHEMA_VERSION,
        "status": "passed_exact_camera_conformance",
        "passed": True,
        "thresholds": FROZEN_THRESHOLDS,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_modified": False,
        "aura_ply_sha256": EXPECTED_AURA_PLY_SHA256,
        "candidate_policy_queried": False,
    }
    native_conformance["receipt_digest"] = canonical_digest(
        native_conformance, digest_field="receipt_digest"
    )
    receipt["aura_renderer_conformance_receipt"] = native_conformance
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert validate_live_hybrid_runtime_receipt(receipt) == receipt

    mutated = copy.deepcopy(receipt)
    mutated["camera_calibration_retained"] = False
    mutated["camera_timestamps_retained"] = False
    mutated["camera_data_types"] = ["rgb"]
    mutated["receipt_digest"] = canonical_digest(
        mutated, digest_field="receipt_digest"
    )
    with pytest.raises(
        LiveHybridObservationError,
        match=(
            "hybrid_runtime_camera_calibration_missing.*"
            "hybrid_runtime_camera_timestamps_missing.*"
            "hybrid_runtime_isaac_camera_data_types_invalid"
        ),
    ):
        validate_live_hybrid_runtime_receipt(mutated)
