"""Protect source appearance outside a declared local repair; freeze geometry and opacity."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

GEOMETRY_MODE = "freeze_declared_appearance_initialization"


def validate_partition(partition, total_count):
    if not isinstance(partition, dict):
        raise ValueError("artifixer_appearance_partition_missing")
    values = [
        partition.get(k) for k in ("frozen_source_count", "generated_support_count", "total_count")
    ]
    if (
        any(type(v) is not int or v <= 0 for v in values)
        or values[0] + values[1] != values[2]
        or values[2] != total_count
        or partition.get("reused_source_vertex_rows_byte_exact") is not True
    ):
        raise ValueError("artifixer_appearance_partition_invalid")
    return values[0]


TRAINING_POLICY = "corrected_only_local_appearance"
LEGACY_TRAINING_POLICY = "masked_original_anchors"


def local_appearance_mask(reference, partition):
    """Recompute the permitted color region from the bound initialization.

    A 10 cm tabletop neighborhood includes contact shadow, but only small,
    surface-adjacent splats can change outside the exact target box. Geometry
    and opacity never change, including those of the neighboring object.
    """
    prefix = validate_partition(partition, reference.count)
    mask = np.zeros(prefix, dtype=bool)
    policy = partition.get("local_appearance_policy")
    if policy is None:
        return mask
    if (not isinstance(policy, dict) or policy.get("mode") != TRAINING_POLICY
            or policy.get("region_rule") != "target_or_registered_tabletop_3sigma_v1"):
        raise ValueError("artifixer_local_appearance_policy_invalid")
    lower = np.asarray(policy.get("target_lower_m"), np.float64)
    upper = np.asarray(policy.get("target_upper_m"), np.float64)
    top = policy.get("support_top_z_m")
    if (lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all() or np.any(upper <= lower)
            or np.max(upper - lower) > 1 or type(top) not in (int, float)
            or not np.isfinite(top) or abs(top - lower[2]) > .01):
        raise ValueError("artifixer_local_appearance_bounds_invalid")
    xyz = np.asarray(reference.xyz[:prefix], np.float64)
    scales = np.exp(np.asarray(reference.scales[:prefix], np.float64))
    if not np.isfinite(xyz).all() or not np.isfinite(scales).all():
        raise ValueError("artifixer_local_appearance_nonfinite")
    target = ((xyz >= lower) & (xyz <= upper)).all(axis=1)
    tabletop = ((xyz[:, :2] >= lower[:2] - .10)
                & (xyz[:, :2] <= upper[:2] + .10)).all(axis=1)
    # Restrict a splat's whole three-sigma vertical extent, accounting for its
    # quaternion, rather than allowing tall neighboring-bottle splats by center.
    q = np.asarray(reference.quats[:prefix], np.float64)
    norm = np.linalg.norm(q, axis=1)
    if not np.isfinite(q).all() or np.any(norm < 1e-10):
        raise ValueError("artifixer_local_appearance_rotation_invalid")
    w, x, y, z = (q / norm[:, None]).T
    vertical = np.c_[2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    sigma_z = np.sqrt(((vertical * scales)**2).sum(axis=1))
    tabletop &= np.abs(xyz[:, 2] - top) + 3*sigma_z <= .01
    mask = (target | tabletop) & (scales.max(axis=1) <= .08)
    return mask


def verify_frozen_appearance(*, model, reference, partition):
    prefix = validate_partition(partition, reference.count)
    editable = local_appearance_mask(reference, partition)
    protected = ~editable
    expected = {
        "density": np.asarray(reference.opacity, np.float32).reshape(-1, 1),
        "features_albedo": np.asarray(reference.f_dc[:prefix], np.float32),
        "features_specular": np.asarray(reference.sh_rest[:prefix], np.float32)
        .reshape(prefix, 3, -1)
        .transpose(0, 2, 1)
        .reshape(prefix, -1),
    }
    for field, value in expected.items():
        actual = np.asarray(getattr(model, field).detach().cpu(), np.float32)
        if field != "density":
            actual = actual[:prefix][protected]
            value = value[protected]
        if actual.shape != value.shape or not np.array_equal(
            actual.view(np.uint32), value.view(np.uint32)
        ):
            raise ValueError("artifixer_frozen_appearance_changed:" + field)
    return {
        "exact_source_appearance_prefix_match": not bool(editable.any()),
        "exact_protected_source_appearance_match": True,
        "local_appearance_policy": partition.get("local_appearance_policy"),
        "editable_source_count": int(editable.sum()),
        "protected_source_count": int(protected.sum()),
        "exact_full_density_tensor_match": True,
        "frozen_source_count": prefix,
        "generated_support_count": partition["generated_support_count"],
    }


@contextmanager
def freeze_source_appearance(model_class, *, reference, partition):
    """Install scoped gradient hooks before the released trainer creates Adam.

    Released ``train_3dgrut`` builds and trains in this process. The class method
    is restored even on failure; the upstream files and model implementation
    remain unchanged. Exact post-training tensor checks are the final safeguard.
    """
    prefix = validate_partition(partition, reference.count)
    protected = ~local_appearance_mask(reference, partition)
    original = model_class.set_optimizable_parameters
    models = []

    def initialize(model):
        original(model)
        if int(model.positions.shape[0]) != reference.count:
            raise ValueError("artifixer_appearance_initialization_count_changed")
        if any(
            getattr(model, field).requires_grad
            for field in ("positions", "rotation", "scale", "density")
        ):
            raise ValueError("artifixer_appearance_geometry_or_opacity_trainable")
        verify_frozen_appearance(model=model, reference=reference, partition=partition)
        for field in ("features_albedo", "features_specular"):
            tensor = getattr(model, field)
            if not tensor.requires_grad:
                raise ValueError("artifixer_generated_appearance_not_trainable")

            import torch
            fixed = torch.as_tensor(protected, device=tensor.device)

            def protect(gradient, fixed=fixed):
                result = gradient.clone()
                result[:prefix][fixed] = 0
                return result

            tensor.register_hook(protect)
        models.append(model)

    model_class.set_optimizable_parameters = initialize
    try:
        yield
        if len(models) != 1:
            raise ValueError("artifixer_appearance_optimizer_initialization_not_exact")
        verify_frozen_appearance(model=models[0], reference=reference, partition=partition)
    finally:
        model_class.set_optimizable_parameters = original
