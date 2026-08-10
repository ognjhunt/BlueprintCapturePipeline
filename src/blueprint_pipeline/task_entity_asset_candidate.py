"""Fail-closed SimReady candidate contracts for task-entity assets.

This module records what was authored before a native simulator is allowed to
upgrade the asset.  It supports the two asset classes needed by the bounded
deformable-transfer rehearsal without encoding a towel or basket execution
path: a closed volumetric FEM deformable and an open rigid receptacle.

A valid contract is still only a candidate.  Native import, contacts, reset,
render alignment, and real-material equivalence remain separate evidence
gates.  Caller-authored qualification booleans are rejected so the existence
of a USD file cannot silently become a simulator or physical claim.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_entity_asset_candidate.v1"
ASSET_CLASSES = ("deformable_volume", "rigid_receptacle")
AUTHORING_METHODS = (
    "released_code_parametric",
    "checked_in_manual_parametric",
    "released_code_geometry_conversion",
)
DEFORMABLE_REPRESENTATION = "closed_tetrahedral_volumetric_fem"
RIGID_COLLISION_REPRESENTATIONS = (
    "static_open_triangle_mesh",
    "multi_part_convex_open_receptacle",
    "rigid_open_triangle_mesh",
)

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_FILE_ROLES = {
    "deformable_volume": frozenset(
        {
            "rest_geometry",
            "material_definition",
            "texture",
            "physics_configuration",
            "runtime_usd",
        }
    ),
    "rigid_receptacle": frozenset(
        {
            "visual_geometry",
            "collision_geometry",
            "material_definition",
            "texture",
            "physics_configuration",
            "runtime_usd",
        }
    ),
}


class TaskEntityAssetCandidateError(ValueError):
    """Stable, sorted candidate-contract failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskEntityAssetCandidateError(
            ["task_entity_asset_candidate_not_json"]
        ) from exc
    if not isinstance(result, dict):
        raise TaskEntityAssetCandidateError(["task_entity_asset_candidate_invalid"])
    return result


def _mapping(
    value: Any, *, field: str, errors: list[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"task_entity_asset_{field}_invalid")
        return {}
    return value


def _identifier(value: Any, *, field: str, errors: list[str]) -> str:
    result = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(result):
        errors.append(f"task_entity_asset_{field}_invalid")
    return result


def _string(value: Any, *, field: str, errors: list[str]) -> str:
    result = str(value or "").strip()
    if not result:
        errors.append(f"task_entity_asset_{field}_invalid")
    return result


def _digest(value: Any, *, field: str, errors: list[str]) -> str:
    result = str(value or "").strip()
    if not _DIGEST.fullmatch(result):
        errors.append(f"task_entity_asset_{field}_invalid")
    return result


def _boolean(value: Any, *, field: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"task_entity_asset_{field}_invalid")
        return False
    return value


def _number(
    value: Any,
    *,
    field: str,
    errors: list[str],
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool):
        errors.append(f"task_entity_asset_{field}_invalid")
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"task_entity_asset_{field}_invalid")
        return 0.0
    if not math.isfinite(result):
        errors.append(f"task_entity_asset_{field}_invalid")
        return 0.0
    if minimum is not None:
        if minimum_inclusive and result < minimum:
            errors.append(f"task_entity_asset_{field}_invalid")
        if not minimum_inclusive and result <= minimum:
            errors.append(f"task_entity_asset_{field}_invalid")
    if maximum is not None and result > maximum:
        errors.append(f"task_entity_asset_{field}_invalid")
    return result


def _integer(
    value: Any, *, field: str, errors: list[str], minimum: int = 1
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.append(f"task_entity_asset_{field}_invalid")
        return 0
    return value


def _vector(
    value: Any,
    *,
    length: int,
    field: str,
    errors: list[str],
    positive: bool = False,
) -> list[float]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        errors.append(f"task_entity_asset_{field}_invalid")
        return []
    result = [
        _number(
            item,
            field=field,
            errors=errors,
            minimum=0.0 if positive else None,
            minimum_inclusive=not positive,
        )
        for item in value
    ]
    return result


def _pose(value: Any, *, field: str, errors: list[str]) -> dict[str, Any]:
    source = _mapping(value, field=field, errors=errors)
    position = _vector(
        source.get("position_world_m"),
        length=3,
        field=field,
        errors=errors,
    )
    orientation = _vector(
        source.get("orientation_xyzw"),
        length=4,
        field=field,
        errors=errors,
    )
    if orientation:
        norm = math.sqrt(sum(component * component for component in orientation))
        if abs(norm - 1.0) > 1.0e-6:
            errors.append(f"task_entity_asset_{field}_invalid")
    return {
        "position_world_m": position,
        "orientation_xyzw": orientation,
    }


def _string_list(value: Any, *, field: str, errors: list[str]) -> list[str]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
    ):
        errors.append(f"task_entity_asset_{field}_invalid")
        return []
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(set(result)) != len(result):
        errors.append(f"task_entity_asset_{field}_invalid")
    return result


def _normalise_observation(
    value: Any, *, asset_class: str, errors: list[str]
) -> dict[str, Any]:
    source = _mapping(value, field="source_observation", errors=errors)
    bounds = _mapping(
        source.get("bounds_world"), field="source_bounds", errors=errors
    )
    minimum = _vector(
        bounds.get("minimum_m"),
        length=3,
        field="source_bounds",
        errors=errors,
    )
    maximum = _vector(
        bounds.get("maximum_m"),
        length=3,
        field="source_bounds",
        errors=errors,
    )
    dimensions = _vector(
        source.get("metric_dimensions_m"),
        length=3,
        field="metric_dimensions",
        errors=errors,
        positive=True,
    )
    if minimum and maximum and dimensions:
        observed_dimensions = [maximum[i] - minimum[i] for i in range(3)]
        if any(
            observed_dimensions[i] <= 0.0
            or abs(observed_dimensions[i] - dimensions[i]) > 1.0e-6
            for i in range(3)
        ):
            errors.append("task_entity_asset_source_bounds_dimension_mismatch")

    coverage = _mapping(
        source.get("coverage"), field="source_coverage", errors=errors
    )
    full_surface_observed = _boolean(
        coverage.get("full_surface_observed"),
        field="source_coverage_full_surface",
        errors=errors,
    )
    unobserved_regions = _string_list(
        coverage.get("unobserved_regions"),
        field="source_coverage_unobserved_regions",
        errors=errors,
    )
    metric_bounds_observed = _boolean(
        coverage.get("metric_bounds_observed"),
        field="source_coverage_metric_bounds",
        errors=errors,
    )
    rest_state_bounded = _boolean(
        coverage.get("rest_state_bounded"),
        field="source_coverage_rest_state",
        errors=errors,
    )
    if full_surface_observed and unobserved_regions:
        errors.append("task_entity_asset_source_coverage_contradictory")
    if not full_surface_observed and not unobserved_regions:
        errors.append("task_entity_asset_unobserved_regions_missing")
    if not metric_bounds_observed or not rest_state_bounded:
        errors.append("task_entity_asset_source_geometry_not_bounded")

    interior_collision_observed = _boolean(
        coverage.get("interior_collision_observed"),
        field="source_coverage_interior_collision",
        errors=errors,
    )
    interior_appearance_observed = _boolean(
        coverage.get("interior_appearance_observed"),
        field="source_coverage_interior_appearance",
        errors=errors,
    )
    engineered_interior_not_factual = _boolean(
        coverage.get("engineered_interior_not_factual"),
        field="source_coverage_engineered_interior",
        errors=errors,
    )
    if asset_class == "rigid_receptacle" and not interior_collision_observed:
        errors.append("task_entity_asset_receptacle_interior_collision_unobserved")
    if (
        asset_class == "rigid_receptacle"
        and not interior_appearance_observed
        and not engineered_interior_not_factual
    ):
        errors.append("task_entity_asset_hidden_interior_factual_claim_forbidden")

    source_size_bytes = source.get("source_size_bytes")
    if (
        isinstance(source_size_bytes, bool)
        or not isinstance(source_size_bytes, int)
        or source_size_bytes <= 0
    ):
        errors.append("task_entity_asset_source_size_invalid")
        source_size_bytes = 0

    return {
        "observation_id": _identifier(
            source.get("observation_id"), field="observation_id", errors=errors
        ),
        "source_reference": _string(
            source.get("source_reference"),
            field="source_reference",
            errors=errors,
        ),
        "source_sha256": _digest(
            source.get("source_sha256"), field="source_digest", errors=errors
        ),
        "source_size_bytes": source_size_bytes,
        "bounds_world": {"minimum_m": minimum, "maximum_m": maximum},
        "metric_dimensions_m": dimensions,
        "coverage": {
            "metric_bounds_observed": metric_bounds_observed,
            "rest_state_bounded": rest_state_bounded,
            "full_surface_observed": full_surface_observed,
            "interior_collision_observed": interior_collision_observed,
            "interior_appearance_observed": interior_appearance_observed,
            "engineered_interior_not_factual": engineered_interior_not_factual,
            "unobserved_regions": unobserved_regions,
        },
    }


def _normalise_rights(value: Any, *, errors: list[str]) -> dict[str, Any]:
    source = _mapping(value, field="rights", errors=errors)
    result = {
        key: _string(source.get(key), field=f"rights_{key}", errors=errors)
        for key in (
            "source_revision",
            "license_id",
            "license_reference",
            "attribution",
            "derived_processing_authority_id",
            "provider_terms_id",
            "output_rights_id",
        )
    }
    result["license_sha256"] = _digest(
        source.get("license_sha256"), field="rights_license_digest", errors=errors
    )
    for key in (
        "raw_source_private_upload_permitted",
        "derived_asset_private_upload_permitted",
        "raw_redistribution_permitted",
        "provider_retention_permitted",
        "provider_training_permitted",
    ):
        result[key] = _boolean(
            source.get(key), field=f"rights_{key}", errors=errors
        )
    if result["provider_training_permitted"]:
        errors.append("task_entity_asset_provider_training_forbidden")
    return result


def _normalise_authoring(value: Any, *, errors: list[str]) -> dict[str, Any]:
    source = _mapping(value, field="authoring", errors=errors)
    method = str(source.get("method") or "")
    if method not in AUTHORING_METHODS:
        errors.append("task_entity_asset_authoring_method_invalid")
    revision = str(source.get("source_revision") or "")
    tree = str(source.get("source_tree") or "")
    if not _REVISION.fullmatch(revision):
        errors.append("task_entity_asset_authoring_revision_invalid")
    if not _REVISION.fullmatch(tree):
        errors.append("task_entity_asset_authoring_tree_invalid")
    return {
        "method": method,
        "source_repository": _string(
            source.get("source_repository"),
            field="authoring_source_repository",
            errors=errors,
        ),
        "source_revision": revision,
        "source_tree": tree,
        "package_name": _string(
            source.get("package_name"),
            field="authoring_package_name",
            errors=errors,
        ),
        "package_version": _string(
            source.get("package_version"),
            field="authoring_package_version",
            errors=errors,
        ),
        "generated_geometry_used": _boolean(
            source.get("generated_geometry_used"),
            field="authoring_generated_geometry_used",
            errors=errors,
        ),
        "generated_physics_used": _boolean(
            source.get("generated_physics_used"),
            field="authoring_generated_physics_used",
            errors=errors,
        ),
    }


def _normalise_files(
    value: Any, *, asset_class: str, errors: list[str]
) -> list[dict[str, Any]]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
    ):
        errors.append("task_entity_asset_files_invalid")
        return []
    rows: list[dict[str, Any]] = []
    roles: set[str] = set()
    paths: set[str] = set()
    for index, raw in enumerate(value):
        source = _mapping(raw, field=f"file_{index}", errors=errors)
        role = str(source.get("role") or "")
        if role not in _FILE_ROLES.get(asset_class, frozenset()) or role in roles:
            errors.append(f"task_entity_asset_file_role_invalid:{role or index}")
        roles.add(role)
        path = str(source.get("path") or "")
        pure = PurePosixPath(path)
        if (
            not path
            or pure.is_absolute()
            or ".." in pure.parts
            or path in paths
        ):
            errors.append(f"task_entity_asset_file_path_invalid:{role or index}")
        paths.add(path)
        size_bytes = source.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            errors.append(f"task_entity_asset_file_size_invalid:{role or index}")
            size_bytes = 0
        rows.append(
            {
                "role": role,
                "path": path,
                "sha256": _digest(
                    source.get("sha256"),
                    field=f"file_digest_{role or index}",
                    errors=errors,
                ),
                "size_bytes": size_bytes,
            }
        )
    missing = _FILE_ROLES.get(asset_class, frozenset()) - roles
    for role in sorted(missing):
        errors.append(f"task_entity_asset_file_missing:{role}")
    return sorted(rows, key=lambda row: (row["role"], row["path"]))


def _normalise_transform(value: Any, *, errors: list[str]) -> dict[str, Any]:
    source = _mapping(value, field="transform", errors=errors)
    scale = _vector(
        source.get("scale_xyz"),
        length=3,
        field="transform_scale",
        errors=errors,
        positive=True,
    )
    return {
        "authored_origin_m": _vector(
            source.get("authored_origin_m"),
            length=3,
            field="transform_authored_origin",
            errors=errors,
        ),
        "pivot_m": _vector(
            source.get("pivot_m"),
            length=3,
            field="transform_pivot",
            errors=errors,
        ),
        "scale_xyz": scale,
        "world_pose": _pose(
            source.get("world_pose"), field="transform_world_pose", errors=errors
        ),
        "meters_per_unit": _number(
            source.get("meters_per_unit"),
            field="transform_meters_per_unit",
            errors=errors,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "up_axis": _string(
            source.get("up_axis"), field="transform_up_axis", errors=errors
        ),
    }


def _normalise_deformable(value: Any, *, errors: list[str]) -> dict[str, Any]:
    source = _mapping(value, field="deformable_configuration", errors=errors)
    representation = str(source.get("representation") or "")
    if representation != DEFORMABLE_REPRESENTATION:
        errors.append("task_entity_asset_deformable_representation_invalid")
    topology = _mapping(
        source.get("rest_topology"), field="deformable_topology", errors=errors
    )
    topology_result = {
        "vertex_count": _integer(
            topology.get("vertex_count"),
            field="deformable_vertex_count",
            errors=errors,
            minimum=4,
        ),
        "tetrahedron_count": _integer(
            topology.get("tetrahedron_count"),
            field="deformable_tetrahedron_count",
            errors=errors,
            minimum=1,
        ),
        "closed_volume": _boolean(
            topology.get("closed_volume"),
            field="deformable_closed_volume",
            errors=errors,
        ),
        "manifold_surface": _boolean(
            topology.get("manifold_surface"),
            field="deformable_manifold_surface",
            errors=errors,
        ),
        "topology_sha256": _digest(
            topology.get("topology_sha256"),
            field="deformable_topology_digest",
            errors=errors,
        ),
    }
    if not topology_result["closed_volume"] or not topology_result["manifold_surface"]:
        errors.append("task_entity_asset_deformable_topology_not_closed_manifold")

    material = _mapping(
        source.get("material"), field="deformable_material", errors=errors
    )
    material_result = {
        "mass_kg": _number(
            material.get("mass_kg"),
            field="deformable_mass",
            errors=errors,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "volume_density_kg_m3": _number(
            material.get("volume_density_kg_m3"),
            field="deformable_density",
            errors=errors,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "effective_thickness_m": _number(
            material.get("effective_thickness_m"),
            field="deformable_thickness",
            errors=errors,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "youngs_modulus_pa": _number(
            material.get("youngs_modulus_pa"),
            field="deformable_youngs_modulus",
            errors=errors,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "poissons_ratio": _number(
            material.get("poissons_ratio"),
            field="deformable_poissons_ratio",
            errors=errors,
            minimum=0.0,
            maximum=0.499999,
        ),
        "elasticity_damping": _number(
            material.get("elasticity_damping"),
            field="deformable_elasticity_damping",
            errors=errors,
            minimum=0.0,
        ),
        "velocity_damping": _number(
            material.get("velocity_damping"),
            field="deformable_velocity_damping",
            errors=errors,
            minimum=0.0,
        ),
        "dynamic_friction": _number(
            material.get("dynamic_friction"),
            field="deformable_dynamic_friction",
            errors=errors,
            minimum=0.0,
        ),
        "independent_bend_parameter_available": _boolean(
            material.get("independent_bend_parameter_available"),
            field="deformable_independent_bend",
            errors=errors,
        ),
        "independent_shear_parameter_available": _boolean(
            material.get("independent_shear_parameter_available"),
            field="deformable_independent_shear",
            errors=errors,
        ),
        "thin_shell_cloth_claimed": _boolean(
            material.get("thin_shell_cloth_claimed"),
            field="deformable_thin_shell_claim",
            errors=errors,
        ),
    }
    if (
        material_result["independent_bend_parameter_available"]
        or material_result["independent_shear_parameter_available"]
        or material_result["thin_shell_cloth_claimed"]
    ):
        errors.append("task_entity_asset_unsupported_cloth_parameter_claim")

    solver = _mapping(
        source.get("solver"), field="deformable_solver", errors=errors
    )
    collision = _mapping(
        source.get("collision"), field="deformable_collision", errors=errors
    )
    reset = _mapping(source.get("reset"), field="deformable_reset", errors=errors)
    solver_result = {
        "mesh_resolution": _integer(
            solver.get("mesh_resolution"),
            field="deformable_mesh_resolution",
            errors=errors,
        ),
        "particle_or_vertex_spacing_m": _number(
            solver.get("particle_or_vertex_spacing_m"),
            field="deformable_particle_spacing",
            errors=errors,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "position_iterations": _integer(
            solver.get("position_iterations"),
            field="deformable_position_iterations",
            errors=errors,
        ),
        "substeps": _integer(
            solver.get("substeps"),
            field="deformable_substeps",
            errors=errors,
        ),
        "maximum_admitted_principal_strain": _number(
            solver.get("maximum_admitted_principal_strain"),
            field="deformable_maximum_strain",
            errors=errors,
            minimum=0.0,
            minimum_inclusive=False,
        ),
    }
    collision_result = {
        "self_collision_enabled": _boolean(
            collision.get("self_collision_enabled"),
            field="deformable_self_collision",
            errors=errors,
        ),
        "contact_offset_m": _number(
            collision.get("contact_offset_m"),
            field="deformable_contact_offset",
            errors=errors,
            minimum=0.0,
        ),
        "rest_offset_m": _number(
            collision.get("rest_offset_m"),
            field="deformable_rest_offset",
            errors=errors,
            minimum=0.0,
        ),
        "requested_grasp_contact_representation": _string(
            collision.get("requested_grasp_contact_representation"),
            field="deformable_grasp_contact_representation",
            errors=errors,
        ),
        "hidden_kinematic_attachment_allowed": _boolean(
            collision.get("hidden_kinematic_attachment_allowed"),
            field="deformable_hidden_attachment",
            errors=errors,
        ),
    }
    if collision_result["hidden_kinematic_attachment_allowed"]:
        errors.append("task_entity_asset_hidden_kinematic_attachment_forbidden")
    reset_result = {
        "reset_kind": _string(
            reset.get("reset_kind"), field="deformable_reset_kind", errors=errors
        ),
        "write_default_nodal_state_before_episode": _boolean(
            reset.get("write_default_nodal_state_before_episode"),
            field="deformable_reset_default_nodal_state",
            errors=errors,
        ),
        "zero_nodal_velocities": _boolean(
            reset.get("zero_nodal_velocities"),
            field="deformable_reset_zero_velocities",
            errors=errors,
        ),
        "free_kinematic_flag_value": _number(
            reset.get("free_kinematic_flag_value"),
            field="deformable_reset_free_flag",
            errors=errors,
        ),
        "native_readback_required": _boolean(
            reset.get("native_readback_required"),
            field="deformable_reset_readback",
            errors=errors,
        ),
        "direct_state_write_after_episode_start_allowed": _boolean(
            reset.get("direct_state_write_after_episode_start_allowed"),
            field="deformable_reset_post_start_write",
            errors=errors,
        ),
    }
    if (
        reset_result["reset_kind"] != "native_default_nodal_state"
        or not reset_result["write_default_nodal_state_before_episode"]
        or not reset_result["zero_nodal_velocities"]
        or abs(reset_result["free_kinematic_flag_value"] - 1.0) > 1.0e-9
        or not reset_result["native_readback_required"]
        or reset_result["direct_state_write_after_episode_start_allowed"]
    ):
        errors.append("task_entity_asset_deformable_reset_invalid")

    return {
        "representation": representation,
        "rest_topology": topology_result,
        "material": material_result,
        "solver": solver_result,
        "collision": collision_result,
        "reset": reset_result,
    }


def _normalise_receptacle(value: Any, *, errors: list[str]) -> dict[str, Any]:
    source = _mapping(value, field="receptacle_configuration", errors=errors)
    geometry = _mapping(
        source.get("geometry"), field="receptacle_geometry", errors=errors
    )
    collision = _mapping(
        source.get("collision"), field="receptacle_collision", errors=errors
    )
    material = _mapping(
        source.get("material"), field="receptacle_material", errors=errors
    )
    anchoring = _mapping(
        source.get("anchoring"), field="receptacle_anchoring", errors=errors
    )
    representation = str(collision.get("representation") or "")
    if representation not in RIGID_COLLISION_REPRESENTATIONS:
        errors.append("task_entity_asset_receptacle_collision_representation_invalid")
    open_interior = _boolean(
        geometry.get("open_interior"),
        field="receptacle_open_interior",
        errors=errors,
    )
    top_cap_present = _boolean(
        geometry.get("top_cap_present"),
        field="receptacle_top_cap",
        errors=errors,
    )
    if not open_interior or top_cap_present:
        errors.append("task_entity_asset_receptacle_not_open")
    static_anchored = _boolean(
        anchoring.get("static_anchored"),
        field="receptacle_static_anchored",
        errors=errors,
    )
    mass = _number(
        anchoring.get("mass_kg"),
        field="receptacle_mass",
        errors=errors,
        minimum=0.0,
    )
    if static_anchored and mass != 0.0:
        errors.append("task_entity_asset_static_receptacle_mass_must_be_zero")
    stable_support_readback_required = _boolean(
        anchoring.get("stable_support_readback_required"),
        field="receptacle_support_readback",
        errors=errors,
    )
    native_collision_readback_required = _boolean(
        anchoring.get("native_collision_readback_required"),
        field="receptacle_collision_readback",
        errors=errors,
    )
    if not stable_support_readback_required or not native_collision_readback_required:
        errors.append("task_entity_asset_receptacle_native_readback_required")
    static_friction = _number(
        material.get("static_friction"),
        field="receptacle_static_friction",
        errors=errors,
        minimum=0.0,
    )
    dynamic_friction = _number(
        material.get("dynamic_friction"),
        field="receptacle_dynamic_friction",
        errors=errors,
        minimum=0.0,
    )
    if dynamic_friction > static_friction:
        errors.append("task_entity_asset_receptacle_friction_invalid")
    return {
        "geometry": {
            "open_interior": open_interior,
            "top_cap_present": top_cap_present,
            "interior_dimensions_m": _vector(
                geometry.get("interior_dimensions_m"),
                length=3,
                field="receptacle_interior_dimensions",
                errors=errors,
                positive=True,
            ),
            "wall_thickness_m": _number(
                geometry.get("wall_thickness_m"),
                field="receptacle_wall_thickness",
                errors=errors,
                minimum=0.0,
                minimum_inclusive=False,
            ),
            "floor_thickness_m": _number(
                geometry.get("floor_thickness_m"),
                field="receptacle_floor_thickness",
                errors=errors,
                minimum=0.0,
                minimum_inclusive=False,
            ),
            "engineered_interior": _boolean(
                geometry.get("engineered_interior"),
                field="receptacle_engineered_interior",
                errors=errors,
            ),
        },
        "collision": {
            "representation": representation,
            "collision_sha256": _digest(
                collision.get("collision_sha256"),
                field="receptacle_collision_digest",
                errors=errors,
            ),
            "contact_offset_m": _number(
                collision.get("contact_offset_m"),
                field="receptacle_contact_offset",
                errors=errors,
                minimum=0.0,
            ),
            "rest_offset_m": _number(
                collision.get("rest_offset_m"),
                field="receptacle_rest_offset",
                errors=errors,
                minimum=0.0,
            ),
        },
        "material": {
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "restitution": _number(
                material.get("restitution"),
                field="receptacle_restitution",
                errors=errors,
                minimum=0.0,
                maximum=1.0,
            ),
            "material_provenance_sha256": _digest(
                material.get("material_provenance_sha256"),
                field="receptacle_material_provenance_digest",
                errors=errors,
            ),
        },
        "anchoring": {
            "static_anchored": static_anchored,
            "mass_kg": mass,
            "inertia_diagonal_kg_m2": _vector(
                anchoring.get("inertia_diagonal_kg_m2"),
                length=3,
                field="receptacle_inertia",
                errors=errors,
            ),
            "stable_support_readback_required": stable_support_readback_required,
            "native_collision_readback_required": native_collision_readback_required,
        },
    }


def _normalise_import_identity(value: Any, *, errors: list[str]) -> dict[str, Any]:
    source = _mapping(value, field="simulator_import", errors=errors)
    revision = str(source.get("source_revision") or "")
    if not _REVISION.fullmatch(revision):
        errors.append("task_entity_asset_simulator_import_revision_invalid")
    return {
        "simulator": _string(
            source.get("simulator"), field="simulator_import_name", errors=errors
        ),
        "simulator_version": _string(
            source.get("simulator_version"),
            field="simulator_import_version",
            errors=errors,
        ),
        "source_repository": _string(
            source.get("source_repository"),
            field="simulator_import_repository",
            errors=errors,
        ),
        "source_revision": revision,
        "importer_module": _string(
            source.get("importer_module"),
            field="simulator_import_module",
            errors=errors,
        ),
        "expected_prim_type": _string(
            source.get("expected_prim_type"),
            field="simulator_import_prim_type",
            errors=errors,
        ),
    }


def materialize_task_entity_asset_candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one candidate without granting native or physical authority."""

    source = _clone(value)
    errors: list[str] = []
    if source.get("schema_version") != SCHEMA_VERSION:
        errors.append("task_entity_asset_schema_invalid")
    forbidden_claim_fields = {
        "claims",
        "native_simulator_qualified",
        "visually_aligned_replacement",
        "physically_equivalent_real_material",
        "execution_authorized",
    }
    for field in sorted(forbidden_claim_fields & set(source)):
        errors.append(f"task_entity_asset_caller_claim_forbidden:{field}")

    asset_class = str(source.get("asset_class") or "")
    if asset_class not in ASSET_CLASSES:
        errors.append("task_entity_asset_class_invalid")
    entity_id = _identifier(source.get("entity_id"), field="entity_id", errors=errors)
    asset_id = _identifier(source.get("asset_id"), field="asset_id", errors=errors)
    observation = _normalise_observation(
        source.get("source_observation"), asset_class=asset_class, errors=errors
    )
    rights = _normalise_rights(source.get("rights"), errors=errors)
    authoring = _normalise_authoring(source.get("authoring"), errors=errors)
    files = _normalise_files(source.get("files"), asset_class=asset_class, errors=errors)
    transform = _normalise_transform(source.get("transform"), errors=errors)
    simulator_import = _normalise_import_identity(
        source.get("simulator_import"), errors=errors
    )

    class_configuration: dict[str, Any]
    if asset_class == "deformable_volume":
        class_configuration = {
            "deformable_configuration": _normalise_deformable(
                source.get("deformable_configuration"), errors=errors
            )
        }
        if "receptacle_configuration" in source:
            errors.append("task_entity_asset_unexpected_receptacle_configuration")
        if authoring["generated_physics_used"]:
            errors.append("task_entity_asset_deformable_generated_physics_forbidden")
    elif asset_class == "rigid_receptacle":
        class_configuration = {
            "receptacle_configuration": _normalise_receptacle(
                source.get("receptacle_configuration"), errors=errors
            )
        }
        if "deformable_configuration" in source:
            errors.append("task_entity_asset_unexpected_deformable_configuration")
        if class_configuration:
            outer = observation["metric_dimensions_m"]
            receptacle = class_configuration["receptacle_configuration"]
            interior = receptacle["geometry"]["interior_dimensions_m"]
            wall = receptacle["geometry"]["wall_thickness_m"]
            floor = receptacle["geometry"]["floor_thickness_m"]
            if (
                len(outer) == 3
                and len(interior) == 3
                and (
                    interior[0] + 2.0 * wall > outer[0] + 1.0e-9
                    or interior[1] + 2.0 * wall > outer[1] + 1.0e-9
                    or interior[2] + floor > outer[2] + 1.0e-9
                )
            ):
                errors.append("task_entity_asset_receptacle_dimensions_inconsistent")
    else:
        class_configuration = {}

    retained_diagnostics = _string_list(
        source.get("retained_diagnostic_requirements"),
        field="retained_diagnostic_requirements",
        errors=errors,
    )
    required_diagnostics = {
        "native_import",
        "stable_support_and_no_initial_penetration",
        "native_contact",
        "native_reset_readback",
        "native_render_coverage",
    }
    if asset_class == "deformable_volume":
        required_diagnostics.update(
            {"native_deformable_settling", "native_strain_and_solver_stability"}
        )
    if not required_diagnostics.issubset(retained_diagnostics):
        errors.append("task_entity_asset_retained_diagnostics_incomplete")

    if errors:
        raise TaskEntityAssetCandidateError(errors)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "entity_id": entity_id,
        "asset_id": asset_id,
        "asset_class": asset_class,
        "source_observation": observation,
        "rights": rights,
        "authoring": authoring,
        "files": files,
        "transform": transform,
        "simulator_import": simulator_import,
        **class_configuration,
        "retained_diagnostic_requirements": sorted(retained_diagnostics),
        "status": "simready_candidate_pending_native_qualification",
        "claims": {
            "generated_candidate": authoring["generated_geometry_used"],
            "simready_candidate": True,
            "native_simulator_qualified": False,
            "visually_aligned_replacement": False,
            "physically_equivalent_real_material": False,
            "execution_authorized": False,
        },
        "pending_gates": [
            "native_import_and_schema_readback",
            "native_contact_support_reset_and_settling",
            "native_render_alignment_and_coverage",
        ],
        "physically_unresolved": [
            "real_material_constitutive_equivalence",
            "real_robot_grasp_performance",
        ],
        "candidate_digest": "",
    }
    if asset_class == "deformable_volume":
        result["physically_unresolved"].extend(
            [
                "thin_shell_cloth_behavior",
                "independent_bend_and_shear_equivalence",
            ]
        )
    result["candidate_digest"] = canonical_digest(
        result, digest_field="candidate_digest"
    )
    return result


__all__ = [
    "ASSET_CLASSES",
    "DEFORMABLE_REPRESENTATION",
    "SCHEMA_VERSION",
    "TaskEntityAssetCandidateError",
    "materialize_task_entity_asset_candidate",
]
