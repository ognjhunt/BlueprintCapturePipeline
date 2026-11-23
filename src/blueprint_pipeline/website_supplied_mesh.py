"""A job's object authored from a supplied mesh, with no provider and no spend: a designed object, or a plain shape the
build plan chose for an observed one (owner direction 2026-11-23).

Compile's stage-3 configuration (``website_task_preparation``) names the generative builder (ARTISAN) by default. A
designed object (``website_synthetic_subject``) and an observed object the plan sourced as a parametric shape
(``website_parametric_object``) are authored instead as supplied geometry by ``provided_mesh_rigid_authoring``: the same
static, native-import and assembly qualification follow and nothing is bought. A parametric shape is sized from the
object's estimated bounds in the scene (never a measurement); a design that cannot be made leaves the builder as it was
and says why.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

#: The plan's parametric object, as the handoff names it on the removal manifest (``website_build_plan``).
PARAMETRIC_SUBJECT = "parametric_subject"


def supplied_mesh_configuration(configuration: dict[str, Any], *, synthetic: Mapping[str, Any] | None,
                                synthetic_authoring: Mapping[str, Any] | None, parametric: Any, subject_id: str,
                                sim_min: Sequence[float], sim_max: Sequence[float]) -> dict[str, Any]:
    """``configuration`` with a designed or parametric object authored as supplied geometry; unchanged otherwise."""
    from .website_synthetic_subject import AUTHORING_METHOD, ORIGIN
    if synthetic is not None:
        # The designed mesh is authored as supplied geometry: no generative
        # backend, so no Astra or ARTISAN spend or runtime contract.
        configuration.pop("authoring_backend", None)
        configuration.update(authoring_method=AUTHORING_METHOD, dimension_authority=ORIGIN,
                             source_observation_kind=ORIGIN, appearance_inputs="synthetic_primitive_design",
                             geometry_support="synthetic_primitive_design", synthetic_design=dict(synthetic),
                             synthetic_authoring=synthetic_authoring)
        return configuration
    if not isinstance(parametric, Mapping) or parametric.get("target_id") != subject_id:
        return configuration
    from .website_parametric_object import SOURCE, design
    extents = [float(b) - float(a) for a, b in zip(sim_min, sim_max, strict=True)]
    try:
        chosen = design(parametric.get("shape"), extents)
        # Authoring checks the asset against the object's estimated bounds (a can on an oblong footprint is not one).
        tolerance = float((configuration.get("metric_envelope") or {}).get("maximum_dimension_relative_error", 0))
        if any(abs(made - seen) > tolerance * seen + 1e-6
               for made, seen in zip(chosen["dimensions_m"], extents, strict=True)):
            raise ValueError("website_parametric_object_outside_estimated_bounds")
    except (TypeError, ValueError) as exc:  # the builder stays as it was; the reason travels with it
        configuration["parametric_not_used"] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return configuration
    configuration.pop("authoring_backend", None)
    configuration.update(authoring_method=AUTHORING_METHOD, appearance_inputs="parametric_shape",
                         geometry_support="parametric_shape_sized_from_estimated_bounds", parametric_design=chosen,
                         authoring_source=SOURCE, authoring_cost_usd=0.0)
    return configuration


__all__ = ["PARAMETRIC_SUBJECT", "supplied_mesh_configuration"]
