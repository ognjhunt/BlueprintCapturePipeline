"""Mass must describe the body actually shipped, not a superseded hollowness guess."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_object_simready_packaging as packaging
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError


def _review(*, mass_kg, fill, density=(2000.0, 2600.0), dims=(0.08142562, 0.06046583, 0.143437244)):
    interval = SimpleNamespace(lower=mass_kg * 0.9, upper=mass_kg * 1.1)
    return SimpleNamespace(accepted=SimpleNamespace(
        mass_model=SimpleNamespace(method="density_fill",
                                   density_kg_m3=SimpleNamespace(lower=density[0], upper=density[1]),
                                   envelope_fill_fraction=SimpleNamespace(lower=fill[0], upper=fill[1])),
        properties=SimpleNamespace(mass_kg=SimpleNamespace(value=mass_kg, interval=interval)),
        dimensions=SimpleNamespace(x_m=SimpleNamespace(value=dims[0]),
                                   y_m=SimpleNamespace(value=dims[1]),
                                   z_m=SimpleNamespace(value=dims[2]))))


def test_measured_solid_supersedes_the_reviewed_fill_assumption():
    """Scene 840938 object 219: review said 5-11% hollow, the accepted CAD is 37.87% solid.

    Packaging refused a 3.4x mass disagreement. Density is a material property and
    survives; the fill fraction was a guess made from source views before any solid
    existed, so the measured geometry wins and mass is re-derived from it.
    """

    mesh = SimpleNamespace(volume=267458.3106265911 / 1e9)
    review = _review(mass_kg=0.06, fill=(0.05, 0.11))

    result = packaging._final_mass_consistency(review, mesh)

    assert result["review_fill_assumption_superseded_by_final_geometry"] is True
    assert result["final_fill_fraction"] == pytest.approx(0.3787, abs=1e-4)
    assert result["reviewed_envelope_fill_fraction"] == [0.05, 0.11]
    # Mass now comes from reviewed density x measured volume.
    assert result["accepted_mass_kg"] == pytest.approx(0.6151, abs=1e-3)
    assert result["accepted_mass_interval_kg"][0] == pytest.approx(0.5349, abs=1e-3)
    assert result["accepted_mass_interval_kg"][1] == pytest.approx(0.6954, abs=1e-3)
    # The superseded number is recorded, not dropped, and no truth is claimed.
    assert result["superseded_review_mass_kg"] == 0.06
    assert result["physical_truth_claimed"] is False
    assert result["uncertainty_preserved"] is True
    # Density stays consistent with the body actually shipped.
    assert 2000.0 <= result["implied_density_kg_m3"] <= 2600.0


def test_consistent_review_is_left_untouched():
    """A review whose fill assumption matches the final solid keeps its own mass."""

    mesh = SimpleNamespace(volume=267458.3106265911 / 1e9)
    review = _review(mass_kg=0.6151, fill=(0.30, 0.45))

    result = packaging._final_mass_consistency(review, mesh)

    assert "review_fill_assumption_superseded_by_final_geometry" not in result
    assert result["accepted_mass_kg"] == 0.6151


def test_mass_outside_density_times_volume_still_refuses():
    """The real safety property survives: mass must fit density x measured volume."""

    mesh = SimpleNamespace(volume=267458.3106265911 / 1e9)
    review = _review(mass_kg=5.0, fill=(0.30, 0.45))

    with pytest.raises(AssetAuthoringError, match="mass_inconsistent_with_final_geometry"):
        packaging._final_mass_consistency(review, mesh)
