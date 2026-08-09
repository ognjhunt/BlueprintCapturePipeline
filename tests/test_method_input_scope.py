from __future__ import annotations

import pytest

from blueprint_pipeline.method_input_scope import (
    MethodInputScopeError,
    evaluate_multiview_mask_fraction_scope,
)


def _camera(camera_id: str) -> dict:
    return {
        "camera_id": camera_id,
        "intrinsics": {"width": 100, "height": 50},
    }


def test_method_scope_admits_small_masks_and_blocks_large_masks() -> None:
    small = evaluate_multiview_mask_fraction_scope(
        method_id="released_method",
        profile_id="bounded_v1",
        cameras=[_camera("a"), _camera("b")],
        mask_records=[
            {"camera_id": "a", "masked_pixel_count": 100},
            {"camera_id": "b", "masked_pixel_count": 500},
        ],
        maximum_mask_fraction=0.1,
        profile_basis="qualified_anchor",
    )
    assert small["status"] == "admitted"
    assert small["maximum_observed_mask_fraction"] == 0.1

    large = evaluate_multiview_mask_fraction_scope(
        method_id="released_method",
        profile_id="bounded_v1",
        cameras=[_camera("a"), _camera("b")],
        mask_records=[
            {"camera_id": "a", "masked_pixel_count": 501},
            {"camera_id": "b", "masked_pixel_count": 500},
        ],
        maximum_mask_fraction=0.1,
        profile_basis="qualified_anchor",
    )
    assert large["status"] == "blocked"
    assert large["blockers"] == ["released_method_input_exceeds_qualified_mask_scale"]


def test_method_scope_requires_complete_exact_camera_mask_join() -> None:
    with pytest.raises(MethodInputScopeError, match="camera_mask_join_incomplete"):
        evaluate_multiview_mask_fraction_scope(
            method_id="released_method",
            profile_id="bounded_v1",
            cameras=[_camera("a"), _camera("b")],
            mask_records=[{"camera_id": "a", "masked_pixel_count": 100}],
            maximum_mask_fraction=0.1,
            profile_basis="qualified_anchor",
        )
