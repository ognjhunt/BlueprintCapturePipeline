"""A job's object authored from a supplied mesh, with no provider and no spend: a designed object, a licensed library
model of an observed one, or a plain shape the build plan chose for it (owner direction 2026-11-23).

Compile's stage-3 configuration (``website_task_preparation``) names the generative builder (ARTISAN) by default. A
designed object (``website_synthetic_subject``), a library model (``website_object_library``) and a plain parametric
shape (``website_parametric_object``) are authored instead as supplied geometry by ``provided_mesh_rigid_authoring``:
the same static, native-import and assembly qualification follow and nothing is bought. For the plan's object the
order is the owner's: the library first, then the parametric shape, then the builder as it was. Each is sized from
the object's estimated bounds in the scene (never a measurement); one that cannot be used leaves the next and says
why (``library_not_used``, ``parametric_not_used``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

#: The plan's parametric object, as the handoff names it on the removal manifest (``website_build_plan``).
PARAMETRIC_SUBJECT = "parametric_subject"
#: The library's answer for the plan's object, as the handoff names it on the removal manifest
#: (``website_object_library.attach_library_subject``).
LIBRARY_SUBJECT = "library_subject"


def _library(configuration: dict[str, Any], library: Any, *, subject_id: str, extents: Sequence[float],
             tolerance: float, library_root: Path | None, fetch: Callable[..., Any] | None) -> dict[str, Any] | None:
    """The library model's design for this object, or None with why it is not used recorded on ``configuration``.
    Rights are a hard stop: the deployed catalog must still allow the model under the license it was chosen under
    (``website_library_catalog.rights_now``), whatever the handoff sealed."""
    if not isinstance(library, Mapping) or library.get("target_id") != subject_id:
        return None
    if "mechanism" in configuration:  # supplied geometry is one rigid body: an opening part keeps its builder
        configuration["library_not_used"] = "website_object_library_object_opens"
        return None
    if library.get("not_used") or library_root is None:
        configuration["library_not_used"] = str(library.get("not_used") or "website_object_library_no_output_root")
        return None
    from .website_library_catalog import rights_now
    refused = rights_now(library.get("attribution"))
    if refused:
        configuration["library_not_used"] = refused
        return None
    from .website_library_asset import library_design
    design, why = library_design(library, estimated_m=extents, tolerance=tolerance, output_root=library_root,
                                 fetch=fetch)
    if design is None:
        configuration["library_not_used"] = why
    return design


def supplied_mesh_configuration(configuration: dict[str, Any], *, synthetic: Mapping[str, Any] | None,
                                synthetic_authoring: Mapping[str, Any] | None, parametric: Any, subject_id: str,
                                sim_min: Sequence[float], sim_max: Sequence[float], library: Any = None,
                                library_root: Path | None = None,
                                library_fetch: Callable[..., Any] | None = None) -> dict[str, Any]:
    """``configuration`` with a designed, library or parametric object authored as supplied geometry; unchanged
    otherwise, but for why a library model or a shape named for this object was not used. ``library_fetch``: how a
    library file whose cached bytes changed is fetched again (by default the outbound download)."""
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
    extents = [float(b) - float(a) for a, b in zip(sim_min, sim_max, strict=True)]
    # Authoring checks the asset against the object's estimated bounds with this tolerance; so does every source here.
    try:
        tolerance = float((configuration.get("metric_envelope") or {}).get("maximum_dimension_relative_error", 0))
    except (TypeError, ValueError):
        tolerance = 0.0  # an unreadable tolerance admits only an exact fit
    design = _library(configuration, library, subject_id=subject_id, extents=extents, tolerance=tolerance,
                      library_root=library_root, fetch=library_fetch)
    if design is not None:
        configuration.pop("authoring_backend", None)
        configuration.update(authoring_method=AUTHORING_METHOD, appearance_inputs="library_model",
                             geometry_support="library_model_" + design["sizing"]["status"], library_design=design,
                             authoring_source=design["source"], authoring_cost_usd=0.0)
        return configuration
    if not isinstance(parametric, Mapping) or parametric.get("target_id") != subject_id:
        return configuration
    if "mechanism" in configuration:
        # A parametric shape is one rigid body too: an opening part keeps its builder, whatever the plan's row says
        # (Codex review of #3715: the analysis saw a mechanism the plan's rigid shape would drop).
        configuration["parametric_not_used"] = "website_parametric_object_object_opens"
        return configuration
    from .website_parametric_object import SOURCE, design as parametric_design
    try:
        chosen = parametric_design(parametric.get("shape"), extents)
        # Authoring checks the asset against the object's estimated bounds (a can on an oblong footprint is not one).
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


__all__ = ["LIBRARY_SUBJECT", "PARAMETRIC_SUBJECT", "supplied_mesh_configuration"]
