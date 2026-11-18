"""Turn a Gaussian splat's view-dependent colour with the splat: real spherical harmonics of degree 1 to 3.

A splat's colour seen from direction ``d`` is ``0.5 + SH_C0 * f_dc + sum_k c_k Y_k(d)``, with ``Y_k`` the
real SH basis in the order and signs the reference 3DGS renderer evaluates (``eval_sh``) and PLY ``f_rest_*``
channel-major (all red coefficients, then green, then blue). Moving the splat by a rotation ``R`` (a point
``p`` goes to ``R p``) must leave every view's colour as it was: ``f'(R d) = f(d)``. Each band ``l`` mixes
only within itself, so its ``(2l+1) x (2l+1)`` matrix is solved exactly from the basis at fixed directions.
Pure numpy; deterministic.
"""
from __future__ import annotations

import numpy as np

C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435)
#: The coefficient range of each band in ``f_rest`` order (band 0 is ``f_dc``).
BANDS = ((0, 3), (3, 8), (8, 15))


def sh_basis(directions: np.ndarray, degree: int) -> np.ndarray:
    """The higher-order basis at unit ``directions`` (M, 3): (M, (degree + 1) ** 2 - 1), as ``eval_sh`` orders it."""
    x, y, z = (np.asarray(directions, np.float64)[:, axis] for axis in range(3))
    columns = [-C1 * y, C1 * z, -C1 * x]
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        columns += [C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy)]
        if degree > 2:
            columns += [C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                        C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]
    return np.stack(columns, axis=1)


def _directions(count: int = 64) -> np.ndarray:
    """A fixed Fibonacci sphere: well spread, more than any band needs."""
    index = np.arange(count) + 0.5
    polar = np.arccos(1 - 2 * index / count)
    azimuth = np.pi * (1 + 5 ** 0.5) * index
    return np.column_stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)])


def sh_degree(sh_rest: np.ndarray) -> int:
    """The degree of channel-major ``f_rest`` columns (N, 3 * ((degree + 1) ** 2 - 1)); refuses any other width."""
    width = np.asarray(sh_rest).shape[1] if np.asarray(sh_rest).ndim == 2 else -1
    degree = next((d for d in (1, 2, 3) if 3 * ((d + 1) ** 2 - 1) == width), None)
    if degree is None:
        raise ValueError("splat_sh_rest_width_invalid")
    return degree


def band_rotations(rotation: np.ndarray, degree: int) -> list[np.ndarray]:
    """Per band, the matrix ``M`` with new coefficients ``c' = M c`` after a point ``p`` moves to ``rotation @ p``."""
    rotation = np.asarray(rotation, np.float64)
    if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("splat_sh_rotation_invalid")
    turned = _directions()
    before = sh_basis(turned @ rotation, degree)  # Y(R^T d') for each new direction d'
    after = sh_basis(turned, degree)
    matrices = []
    for start, stop in BANDS[:degree]:
        # Y_l(d') M = Y_l(R^T d') at every direction, exact since the band is closed under rotation.
        matrices.append(np.linalg.lstsq(after[:, start:stop], before[:, start:stop], rcond=None)[0])
    return matrices


def rotate_sh_rest(sh_rest: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Channel-major ``f_rest`` columns turned by ``rotation`` (a point ``p`` goes to ``rotation @ p``)."""
    rest = np.asarray(sh_rest, np.float64)
    degree = sh_degree(rest)
    count = rest.shape[0]
    dim = (degree + 1) ** 2 - 1
    coefficients = rest.reshape(count, 3, dim).copy()
    for (start, stop), matrix in zip(BANDS[:degree], band_rotations(rotation, degree)):
        coefficients[:, :, start:stop] = coefficients[:, :, start:stop] @ matrix.T
    return coefficients.reshape(count, 3 * dim).astype(np.asarray(sh_rest).dtype)
