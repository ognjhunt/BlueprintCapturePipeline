"""Keep original appearance and all geometry/opacity fixed during repair training."""

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


def verify_frozen_appearance(*, model, reference, partition):
    prefix = validate_partition(partition, reference.count)
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
            actual = actual[:prefix]
        if actual.shape != value.shape or not np.array_equal(
            actual.view(np.uint32), value.view(np.uint32)
        ):
            raise ValueError("artifixer_frozen_appearance_changed:" + field)
    return {
        "exact_source_appearance_prefix_match": True,
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

            def protect(gradient):
                result = gradient.clone()
                result[:prefix] = 0
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
