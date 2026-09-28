from copy import deepcopy

import pytest
import numpy as np

from blueprint_pipeline.native_task_arena_policy_canary_worker import (
    _policy_camera_visibility_contract,
)
from blueprint_pipeline.native_task_camera_observability import (
    measure_native_task_semantic_label_pixels,
)


def _snapshot():
    return {"cameras": [
        {"role": role, "semantic_label_pixels": {
             "task_support": measure_native_task_semantic_label_pixels(
                 semantic_ids=np.ones((8, 8), dtype=np.int64),
                 id_to_labels={"1": {"class": "task_support"}},
                 target_label="task_support",
             )},
         "observability": {"passed": True, "pixel_count": 100,
                           "thresholds": {"effective_minimum_pixels": 64},
                           "render_passed": True, "centroid_within_margin": True,
                           "target_semantic_ids": [1]}}
        for role in ("external", "wrist", "overview")
    ]}


@pytest.mark.parametrize("role", ["external", "overview"])
@pytest.mark.parametrize("pixels", [None, 0, 63, True, 64.0])
def test_visible_book_cannot_mask_missing_or_insufficient_tray(role, pixels):
    snapshot = _snapshot()
    camera = next(row for row in snapshot["cameras"] if row["role"] == role)
    camera["semantic_label_pixels"]["task_support"]["pixel_count"] = pixels
    original = deepcopy(snapshot)
    result = _policy_camera_visibility_contract(
        snapshot, preserve_official_droid_calibration=True, require_task_support=True,
    )
    assert snapshot == original
    assert result["passed"] is False
    assert result["camera_visibility"][role] is False
    assert result["role_qualifications"][role]["destination_visible"] is False
    assert f"policy_canary_{role}_destination_visibility_failed" in result["blockers"]


def test_tray_visible_to_external_policy_and_overview_does_not_require_reset_wrist_view():
    snapshot = _snapshot()
    snapshot["cameras"][1]["semantic_label_pixels"].pop("task_support")
    result = _policy_camera_visibility_contract(
        snapshot, preserve_official_droid_calibration=True, require_task_support=True,
    )
    assert result["passed"] is True
    assert result["role_qualifications"]["external"]["destination_visible"] is True
    assert result["role_qualifications"]["external"]["destination_pixel_count"] == 64
    assert result["role_qualifications"]["overview"]["minimum_destination_pixels"] == 64


def test_marker_task_keeps_existing_visibility_contract():
    snapshot = _snapshot()
    for row in snapshot["cameras"]:
        row["semantic_label_pixels"].pop("task_support")
    result = _policy_camera_visibility_contract(
        snapshot, preserve_official_droid_calibration=True,
    )
    assert result["passed"] is True
    assert all("destination_visible" not in row for row in result["role_qualifications"].values())


@pytest.mark.parametrize("field,value", [
    ("target_label", "task_object"),
    ("measurement_authority", "model_label"),
    ("target_semantic_ids", []),
])
def test_destination_pixels_require_native_support_measurement(field, value):
    snapshot = _snapshot()
    snapshot["cameras"][0]["semantic_label_pixels"]["task_support"][field] = value
    result = _policy_camera_visibility_contract(
        snapshot, preserve_official_droid_calibration=True, require_task_support=True,
    )
    assert result["passed"] is False


def test_bare_destination_count_is_not_a_native_measurement():
    snapshot = _snapshot()
    snapshot["cameras"][0]["semantic_label_pixels"]["task_support"] = 14196
    result = _policy_camera_visibility_contract(
        snapshot, preserve_official_droid_calibration=True, require_task_support=True,
    )
    assert result["passed"] is False
