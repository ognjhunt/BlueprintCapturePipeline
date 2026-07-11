from __future__ import annotations

import math

import pytest

from blueprint_pipeline.isaac_task_review_renderer import look_at_quaternion


def test_look_at_quaternion_is_finite_and_normalized() -> None:
    quaternion = look_at_quaternion((2.0, -2.0, 1.5), (0.0, 0.0, 1.0))
    assert all(math.isfinite(value) for value in quaternion)
    assert math.sqrt(sum(value * value for value in quaternion)) == pytest.approx(1.0)
