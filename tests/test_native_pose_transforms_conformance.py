"""Differential conformance for the hand-rolled pose math.

`native_pose_transforms` is deliberately pure Python: Isaac imports happen only
when the runtime class is instantiated, so planning stays importable on the
control plane, which has no GPU. That constraint is real and the module should
stay -- but it means the repo carries its own quaternion algebra alongside
`isaaclab.utils.math`, and four separate convention defects landed at those
seams in one day (PR #774, #775, #777, #778).

The repo's own build-on-top doctrine is that commodity code gets wrapped and
qualified by a conformance test rather than trusted. This is that test: our
xyzw algebra is checked against SciPy, which is available in the fast lane and
is also scalar-last, on deterministic pseudo-random inputs. A future refactor
that quietly changes convention fails here, on CPU, instead of on a rented GPU.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
scipy_rotation = pytest.importorskip("scipy.spatial.transform").Rotation

from blueprint_pipeline.native_franka_pose_servo import (  # noqa: E402
    contract_xyzw_to_native_xyzw,
    native_xyzw_to_contract_xyzw,
)
from blueprint_pipeline.native_pose_transforms import (  # noqa: E402
    _multiply,
    _rotate,
    pose_world_to_base,
)


def _quaternions(count: int) -> list[list[float]]:
    """Deterministic unit quaternions, xyzw."""

    generator = np.random.default_rng(20260819)
    raw = generator.normal(size=(count, 4))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return [list(map(float, row)) for row in raw]


def _vectors(count: int) -> list[list[float]]:
    generator = np.random.default_rng(90260819)
    return [list(map(float, row)) for row in generator.normal(size=(count, 3))]


def _same_rotation(left, right) -> bool:
    """q and -q are the same rotation, so compare as rotations, not floats."""

    return bool(
        np.allclose(left, right, atol=1e-9) or np.allclose(left, -np.asarray(right), atol=1e-9)
    )


def test_quaternion_multiply_matches_scipy() -> None:
    """Our _multiply must compose in the same order SciPy does."""

    left = _quaternions(64)
    right = _quaternions(64)[::-1]
    for a, b in zip(left, right):
        ours = list(_multiply(a, b))
        reference = (scipy_rotation.from_quat(a) * scipy_rotation.from_quat(b)).as_quat()
        assert _same_rotation(ours, reference), (a, b, ours, reference)


def test_quaternion_rotate_matches_scipy() -> None:
    """Our _rotate must apply the rotation, not its inverse."""

    for quaternion, vector in zip(_quaternions(64), _vectors(64)):
        ours = _rotate(quaternion, vector)
        reference = scipy_rotation.from_quat(quaternion).apply(vector)
        assert np.allclose(ours, reference, atol=1e-9), (quaternion, vector, ours, reference)


def test_pose_world_to_base_matches_scipy_frame_subtraction() -> None:
    """Expressing a world pose in a base frame is R^-1 (p - p_base)."""

    poses = _quaternions(48)
    bases = _quaternions(48)[::-1]
    points = _vectors(48)
    base_points = _vectors(48)[::-1]

    for quaternion, base_quaternion, point, base_point in zip(
        poses, bases, points, base_points
    ):
        position, orientation = pose_world_to_base(
            position_world=point,
            quaternion_world_xyzw=quaternion,
            base_position_world=base_point,
            base_quaternion_world_xyzw=base_quaternion,
        )
        base_rotation = scipy_rotation.from_quat(base_quaternion)
        expected_position = base_rotation.inv().apply(
            np.asarray(point) - np.asarray(base_point)
        )
        expected_orientation = (base_rotation.inv() * scipy_rotation.from_quat(quaternion)).as_quat()
        assert np.allclose(position, expected_position, atol=1e-9)
        assert _same_rotation(orientation, expected_orientation)


def test_contract_and_native_conventions_round_trip() -> None:
    """Beta2 spawn, articulation data, and DifferentialIK are all XYZW.

    The old seam reordered XYZW as WXYZ on both write and read, making an
    identity spawn into a 180-degree X rotation and then hiding it on readback.
    """

    for quaternion in _quaternions(64):
        native = list(contract_xyzw_to_native_xyzw(quaternion))
        # the helper renormalises, so compare as values not bits
        assert native == pytest.approx(quaternion)
        assert list(native_xyzw_to_contract_xyzw(native)) == pytest.approx(
            quaternion
        )

    identity_xyzw = [0.0, 0.0, 0.0, 1.0]
    assert list(contract_xyzw_to_native_xyzw(identity_xyzw)) == identity_xyzw
    misread = scipy_rotation.from_quat([1.0, 0.0, 0.0, 0.0])
    assert math.isclose(misread.magnitude(), math.pi, abs_tol=1e-9), (
        "the removed reorder turns identity into a 180 degree X rotation"
    )
