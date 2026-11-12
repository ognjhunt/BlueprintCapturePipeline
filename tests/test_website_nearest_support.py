"""Part 5 (2026-11-12): no surface within the search window under the object is a recorded best guess.

``support_surface_not_found_under_subject`` stopped the job although a surface lay under the object's
footprint, lower than its estimated bottom (a registration residual, or an object filmed held up). Now the
object rests on the nearest surface under its footprint, at most 0.5 m below, with the drop recorded and
the flag ``scene_estimate:support_guessed`` on the result. Nothing under the footprint stays the stop.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_support_geometry.py
#   src/blueprint_pipeline/atlas_scene_estimates.py
#   src/blueprint_pipeline/website_task_preparation.py
import json

import pytest
import trimesh

from blueprint_pipeline.atlas_scene_estimates import SUPPORT_GUESSED, preparation_flags
from blueprint_pipeline.website_support_geometry import nearest_support_under, support_under

# A 1.2 x 0.8 m table top at 0.75 m (Z up, metres); a 0.3 x 0.2 m tray whose estimated bottom floats at 0.95 m.
TABLE = trimesh.creation.box(extents=[1.2, 0.8, 0.05], transform=trimesh.transformations.translation_matrix([0, 0, 0.725]))
LOWER, UPPER = [-0.15, -0.1, 0.95], [0.15, 0.1, 1.0]


def test_the_object_rests_on_the_nearest_surface_under_it_with_the_drop_recorded():
    assert support_under(TABLE, LOWER, UPPER, up=2, meters_per_unit=1.0) is None  # the stop before
    found = nearest_support_under(TABLE, LOWER, UPPER, up=2, meters_per_unit=1.0)
    assert found["top_runtime_units"] == pytest.approx(0.75, abs=1e-6)
    assert found["guess"]["drop_m"] == pytest.approx(0.20, abs=1e-6) and found["guess"]["flag"] == SUPPORT_GUESSED


def test_nothing_under_the_footprint_stays_the_stop():
    aside = [[0.8, 0.6, 0.95], [1.0, 0.8, 1.0]]
    assert nearest_support_under(TABLE, *aside, up=2, meters_per_unit=1.0) is None
    too_far = [[-0.15, -0.1, 1.40], [0.15, 0.1, 1.45]]  # 0.65 m above the top
    assert nearest_support_under(TABLE, *too_far, up=2, meters_per_unit=1.0) is None


def test_the_guess_reaches_the_scene_estimate_flags(tmp_path):
    found = nearest_support_under(TABLE, LOWER, UPPER, up=2, meters_per_unit=1.0)
    path = tmp_path / "preparation.json"
    path.write_text(json.dumps({"support": {"guess": found["guess"]}}, default=float))
    assert SUPPORT_GUESSED in preparation_flags(path)
