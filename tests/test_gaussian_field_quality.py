from __future__ import annotations

import numpy as np
import pytest

from blueprint_pipeline.gaussian_field_quality import (
    measure_gaussian_field_quality,
    measure_source_relative_gaussian_drift,
)


def _room(count: int = 10_000, *, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(839873)
    positions = rng.uniform([-4.0, -6.0, 0.0], [4.0, 6.0, 3.0], size=(count, 3))
    scales = rng.uniform(0.004, 0.08, size=(count, 3))
    return positions * scale, scales * scale


def test_healthy_field_qualifies_at_room_and_tabletop_scales() -> None:
    room_positions, room_scales = _room()
    room = measure_gaussian_field_quality(
        positions=room_positions,
        activated_scales=room_scales,
        opacities=np.full(len(room_positions), 0.8),
    )
    tabletop_positions, tabletop_scales = _room(scale=0.01)
    tabletop = measure_gaussian_field_quality(
        positions=tabletop_positions,
        activated_scales=tabletop_scales,
        opacities=np.full(len(tabletop_positions), 0.8),
    )

    assert room["status"] == "qualified"
    assert tabletop["status"] == "qualified"
    assert room["blockers"] == tabletop["blockers"] == []
    assert (
        room["metrics"]["max_scale_to_robust_diagonal"]
        == (tabletop["metrics"]["max_scale_to_robust_diagonal"])
    )


def test_pathological_far_field_and_kernel_scale_are_rejected() -> None:
    positions, scales = _room()
    positions[-1] = [8_000.0, -7_000.0, 900.0]
    scales[-1] = [975.0, 1.0, 1.0]

    result = measure_gaussian_field_quality(
        positions=positions,
        activated_scales=scales,
        opacities=np.full(len(positions), 0.9),
    )

    assert result["status"] == "blocked"
    assert "gaussian_field_scale_to_robust_extent_above_ceiling" in result["blockers"]
    assert "gaussian_field_center_outlier_above_ceiling" in result["blockers"]
    assert result["learned_tensors_mutated"] is False


def test_source_relative_drift_rejects_diverged_training_output() -> None:
    reference_positions, reference_scales = _room()
    candidate_positions = reference_positions.copy()
    candidate_scales = reference_scales.copy()
    candidate_positions[-1] = [8_000.0, -7_000.0, 900.0]
    candidate_scales[-1] = [975.0, 1.0, 1.0]

    result = measure_source_relative_gaussian_drift(
        reference_positions=reference_positions,
        reference_activated_scales=reference_scales,
        candidate_positions=candidate_positions,
        candidate_activated_scales=candidate_scales,
        candidate_opacities=np.full(len(candidate_positions), 0.9),
    )

    assert result["status"] == "blocked"
    assert "gaussian_field_source_relative_max_scale_growth_above_ceiling" in result["blockers"]
    assert "gaussian_field_source_relative_position_drift_above_ceiling" in result["blockers"]
    assert result["measurement_authority"] == ("exact_retained_and_trained_gaussian_tensor_arrays")


def test_source_relative_contract_is_scale_invariant() -> None:
    positions, scales = _room()
    candidate_positions = positions + 0.005
    candidate_scales = scales * 1.1
    room = measure_source_relative_gaussian_drift(
        reference_positions=positions,
        reference_activated_scales=scales,
        candidate_positions=candidate_positions,
        candidate_activated_scales=candidate_scales,
    )
    tabletop = measure_source_relative_gaussian_drift(
        reference_positions=positions * 0.01,
        reference_activated_scales=scales * 0.01,
        candidate_positions=candidate_positions * 0.01,
        candidate_activated_scales=candidate_scales * 0.01,
    )

    assert room["status"] == tabletop["status"] == "qualified"
    assert room["metrics"] == pytest.approx(tabletop["metrics"])


def test_source_relative_contract_refuses_count_changes_without_correspondence() -> None:
    positions, scales = _room()
    result = measure_source_relative_gaussian_drift(
        reference_positions=positions,
        reference_activated_scales=scales,
        candidate_positions=positions[:-1],
        candidate_activated_scales=scales[:-1],
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["gaussian_field_source_relative_count_mismatch"]
