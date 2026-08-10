from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_entity_asset_candidate import (
    TaskEntityAssetCandidateError,
    materialize_task_entity_asset_candidate,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
REV_A = "a" * 40
REV_B = "b" * 40


def _common(asset_class: str) -> dict:
    deformable = asset_class == "deformable_volume"
    files = [
        {
            "role": role,
            "path": f"assets/{role}.bin",
            "sha256": SHA_A if index % 2 == 0 else SHA_B,
            "size_bytes": 100 + index,
        }
        for index, role in enumerate(
            (
                "rest_geometry",
                "material_definition",
                "texture",
                "physics_configuration",
                "runtime_usd",
            )
            if deformable
            else (
                "visual_geometry",
                "collision_geometry",
                "material_definition",
                "texture",
                "physics_configuration",
                "runtime_usd",
            )
        )
    ]
    result = {
        "schema_version": "task_entity_asset_candidate.v1",
        "entity_id": "task_deformable" if deformable else "task_destination",
        "asset_id": "asset_deformable" if deformable else "asset_destination",
        "asset_class": asset_class,
        "source_observation": {
            "observation_id": "840873-label-79" if deformable else "840873-label-87",
            "source_reference": "InteriorGS/SAGE:840873",
            "source_sha256": SHA_A,
            "source_size_bytes": 1024,
            "bounds_world": {
                "minimum_m": [0.0, 0.0, 1.0],
                "maximum_m": [0.3, 0.2, 1.1],
            },
            "metric_dimensions_m": [0.3, 0.2, 0.1],
            "coverage": {
                "metric_bounds_observed": True,
                "rest_state_bounded": True,
                "full_surface_observed": False,
                "interior_collision_observed": not deformable,
                "interior_appearance_observed": False,
                "engineered_interior_not_factual": not deformable,
                "unobserved_regions": ["underside", "occluded interior"],
            },
        },
        "rights": {
            "source_revision": REV_A,
            "license_id": "fixture-license",
            "license_reference": "https://example.invalid/license",
            "license_sha256": SHA_A,
            "attribution": "fixture attribution",
            "derived_processing_authority_id": "fixture-derived-authority",
            "provider_terms_id": "fixture-provider-terms",
            "output_rights_id": "fixture-output-rights",
            "raw_source_private_upload_permitted": False,
            "derived_asset_private_upload_permitted": True,
            "raw_redistribution_permitted": False,
            "provider_retention_permitted": False,
            "provider_training_permitted": False,
        },
        "authoring": {
            "method": "released_code_parametric",
            "source_repository": "https://example.invalid/released-authoring",
            "source_revision": REV_A,
            "source_tree": REV_B,
            "package_name": "released-authoring",
            "package_version": "1.0.0",
            "generated_geometry_used": True,
            "generated_physics_used": False,
        },
        "files": files,
        "transform": {
            "authored_origin_m": [0.0, 0.0, 0.0],
            "pivot_m": [0.0, 0.0, 0.0],
            "scale_xyz": [1.0, 1.0, 1.0],
            "world_pose": {
                "position_world_m": [1.0, 2.0, 1.1],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "meters_per_unit": 1.0,
            "up_axis": "Z",
        },
        "simulator_import": {
            "simulator": "Isaac Sim",
            "simulator_version": "6.0.0-dev2",
            "source_repository": "https://github.com/isaac-sim/IsaacLab",
            "source_revision": REV_A,
            "importer_module": "isaaclab_physx.assets.DeformableObject"
            if deformable
            else "isaaclab.assets.RigidObject",
            "expected_prim_type": "PhysxDeformableBodyAPI"
            if deformable
            else "UsdGeom.Xform+collision",
        },
        "retained_diagnostic_requirements": [
            "native_import",
            "stable_support_and_no_initial_penetration",
            "native_contact",
            "native_reset_readback",
            "native_render_coverage",
            *(
                [
                    "native_deformable_settling",
                    "native_strain_and_solver_stability",
                ]
                if deformable
                else []
            ),
        ],
    }
    return result


def _deformable() -> dict:
    result = _common("deformable_volume")
    result["deformable_configuration"] = {
        "representation": "closed_tetrahedral_volumetric_fem",
        "rest_topology": {
            "vertex_count": 100,
            "tetrahedron_count": 240,
            "closed_volume": True,
            "manifold_surface": True,
            "topology_sha256": SHA_B,
        },
        "material": {
            "mass_kg": 0.25,
            "volume_density_kg_m3": 250.0,
            "effective_thickness_m": 0.02,
            "youngs_modulus_pa": 50_000.0,
            "poissons_ratio": 0.35,
            "elasticity_damping": 0.01,
            "velocity_damping": 0.1,
            "dynamic_friction": 0.6,
            "independent_bend_parameter_available": False,
            "independent_shear_parameter_available": False,
            "thin_shell_cloth_claimed": False,
        },
        "solver": {
            "mesh_resolution": 16,
            "particle_or_vertex_spacing_m": 0.01,
            "position_iterations": 16,
            "substeps": 4,
            "maximum_admitted_principal_strain": 0.5,
        },
        "collision": {
            "self_collision_enabled": True,
            "contact_offset_m": 0.005,
            "rest_offset_m": 0.002,
            "requested_grasp_contact_representation": ("native_finger_to_deformable_collision"),
            "hidden_kinematic_attachment_allowed": False,
        },
        "reset": {
            "reset_kind": "native_default_nodal_state",
            "write_default_nodal_state_before_episode": True,
            "zero_nodal_velocities": True,
            "free_kinematic_flag_value": 1.0,
            "native_readback_required": True,
            "direct_state_write_after_episode_start_allowed": False,
        },
    }
    return result


def _receptacle() -> dict:
    result = _common("rigid_receptacle")
    result["receptacle_configuration"] = {
        "geometry": {
            "open_interior": True,
            "top_cap_present": False,
            "interior_dimensions_m": [0.25, 0.16, 0.08],
            "wall_thickness_m": 0.01,
            "floor_thickness_m": 0.01,
            "engineered_interior": True,
        },
        "collision": {
            "representation": "multi_part_convex_open_receptacle",
            "collision_sha256": SHA_B,
            "contact_offset_m": 0.002,
            "rest_offset_m": 0.0,
        },
        "material": {
            "static_friction": 0.6,
            "dynamic_friction": 0.5,
            "restitution": 0.0,
            "material_provenance_sha256": SHA_A,
        },
        "anchoring": {
            "static_anchored": True,
            "mass_kg": 0.0,
            "inertia_diagonal_kg_m2": [0.0, 0.0, 0.0],
            "stable_support_readback_required": True,
            "native_collision_readback_required": True,
        },
    }
    return result


@pytest.mark.parametrize("source", [_deformable(), _receptacle()])
def test_both_asset_classes_are_candidates_not_qualified_claims(source: dict) -> None:
    result = materialize_task_entity_asset_candidate(source)

    assert result["status"] == "simready_candidate_pending_native_qualification"
    assert result["claims"] == {
        "generated_candidate": True,
        "simready_candidate": True,
        "native_simulator_qualified": False,
        "visually_aligned_replacement": False,
        "physically_equivalent_real_material": False,
        "execution_authorized": False,
    }
    assert result["candidate_digest"] == canonical_digest(result, digest_field="candidate_digest")


def test_surface_mesh_candidate_defers_native_tetrahedral_topology_truthfully() -> None:
    source = _deformable()
    source["deformable_configuration"]["rest_topology"] = {
        "topology_stage": "surface_mesh_pending_native_cook",
        "vertex_count": 98,
        "surface_triangle_count": 192,
        "tetrahedron_count": None,
        "closed_volume": True,
        "manifold_surface": True,
        "topology_sha256": SHA_B,
        "native_simulation_topology_sha256": None,
        "native_topology_readback_required": True,
    }

    result = materialize_task_entity_asset_candidate(source)

    topology = result["deformable_configuration"]["rest_topology"]
    assert topology == {
        "topology_stage": "surface_mesh_pending_native_cook",
        "vertex_count": 98,
        "surface_triangle_count": 192,
        "tetrahedron_count": None,
        "closed_volume": True,
        "manifold_surface": True,
        "topology_sha256": SHA_B,
        "native_simulation_topology_sha256": None,
        "native_topology_readback_required": True,
    }
    assert result["status"] == "geometry_candidate_pending_native_topology_cook"
    assert result["claims"]["generated_candidate"] is True
    assert result["claims"]["simready_candidate"] is False
    assert result["claims"]["native_simulator_qualified"] is False
    assert result["pending_gates"][0] == ("native_deformable_topology_cook_and_readback")


def test_legacy_v1_explicit_tetrahedral_topology_shape_is_unchanged() -> None:
    result = materialize_task_entity_asset_candidate(_deformable())

    assert result["deformable_configuration"]["rest_topology"] == {
        "vertex_count": 100,
        "tetrahedron_count": 240,
        "closed_volume": True,
        "manifold_surface": True,
        "topology_sha256": SHA_B,
    }
    assert result["status"] == "simready_candidate_pending_native_qualification"
    assert result["claims"]["simready_candidate"] is True


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "tetrahedron_count",
            240,
            "task_entity_asset_deformable_native_topology_premature",
        ),
        (
            "native_simulation_topology_sha256",
            SHA_A,
            "task_entity_asset_deformable_native_topology_premature",
        ),
        (
            "native_topology_readback_required",
            False,
            "task_entity_asset_deformable_native_topology_readback_required",
        ),
    ],
)
def test_surface_mesh_candidate_cannot_invent_pre_cook_native_topology(
    field: str, value: object, error: str
) -> None:
    source = _deformable()
    source["deformable_configuration"]["rest_topology"] = {
        "topology_stage": "surface_mesh_pending_native_cook",
        "vertex_count": 98,
        "surface_triangle_count": 192,
        "tetrahedron_count": None,
        "closed_volume": True,
        "manifold_surface": True,
        "topology_sha256": SHA_B,
        "native_simulation_topology_sha256": None,
        "native_topology_readback_required": True,
    }
    source["deformable_configuration"]["rest_topology"][field] = value

    with pytest.raises(TaskEntityAssetCandidateError, match=error):
        materialize_task_entity_asset_candidate(source)


def test_volumetric_surrogate_cannot_claim_thin_shell_or_independent_bend() -> None:
    source = _deformable()
    source["deformable_configuration"]["material"]["thin_shell_cloth_claimed"] = True
    source["deformable_configuration"]["material"]["independent_bend_parameter_available"] = True

    with pytest.raises(
        TaskEntityAssetCandidateError,
        match="task_entity_asset_unsupported_cloth_parameter_claim",
    ):
        materialize_task_entity_asset_candidate(source)


def test_deformable_forbids_hidden_attachment_and_post_start_writes() -> None:
    source = _deformable()
    source["deformable_configuration"]["collision"]["hidden_kinematic_attachment_allowed"] = True
    source["deformable_configuration"]["reset"][
        "direct_state_write_after_episode_start_allowed"
    ] = True

    with pytest.raises(TaskEntityAssetCandidateError) as exc_info:
        materialize_task_entity_asset_candidate(source)

    assert "task_entity_asset_hidden_kinematic_attachment_forbidden" in exc_info.value.errors
    assert "task_entity_asset_deformable_reset_invalid" in exc_info.value.errors


def test_receptacle_cannot_truthify_an_unobserved_engineered_interior() -> None:
    source = _receptacle()
    source["source_observation"]["coverage"]["engineered_interior_not_factual"] = False

    with pytest.raises(
        TaskEntityAssetCandidateError,
        match="task_entity_asset_hidden_interior_factual_claim_forbidden",
    ):
        materialize_task_entity_asset_candidate(source)


def test_caller_cannot_upgrade_candidate_with_boolean_claims() -> None:
    source = _deformable()
    source["native_simulator_qualified"] = True

    with pytest.raises(
        TaskEntityAssetCandidateError,
        match="task_entity_asset_caller_claim_forbidden:native_simulator_qualified",
    ):
        materialize_task_entity_asset_candidate(source)


def test_geometry_file_and_rights_fail_closed() -> None:
    source = copy.deepcopy(_receptacle())
    source["files"][0]["path"] = "../escape.usd"
    source["rights"]["provider_training_permitted"] = True

    with pytest.raises(TaskEntityAssetCandidateError) as exc_info:
        materialize_task_entity_asset_candidate(source)

    assert any("task_entity_asset_file_path_invalid" in error for error in exc_info.value.errors)
    assert "task_entity_asset_provider_training_forbidden" in exc_info.value.errors


def test_deformable_ai_generated_physics_cannot_own_native_contract() -> None:
    source = _deformable()
    source["authoring"]["generated_physics_used"] = True

    with pytest.raises(
        TaskEntityAssetCandidateError,
        match="task_entity_asset_deformable_generated_physics_forbidden",
    ):
        materialize_task_entity_asset_candidate(source)


def test_receptacle_interior_must_fit_outer_observed_bounds() -> None:
    source = _receptacle()
    source["receptacle_configuration"]["geometry"]["interior_dimensions_m"] = [
        0.30,
        0.20,
        0.10,
    ]

    with pytest.raises(
        TaskEntityAssetCandidateError,
        match="task_entity_asset_receptacle_dimensions_inconsistent",
    ):
        materialize_task_entity_asset_candidate(source)


def test_receptacle_preserves_asymmetric_wall_clearances() -> None:
    source = _receptacle()
    source["receptacle_configuration"]["geometry"]["wall_clearances_m"] = {
        "x_min": 0.01,
        "x_max": 0.04,
        "y_min": 0.01,
        "y_max": 0.03,
    }
    source["receptacle_configuration"]["geometry"]["interior_dimensions_m"] = [
        0.25,
        0.16,
        0.08,
    ]

    result = materialize_task_entity_asset_candidate(source)

    assert result["receptacle_configuration"]["geometry"]["wall_clearances_m"] == {
        "x_min": 0.01,
        "x_max": 0.04,
        "y_min": 0.01,
        "y_max": 0.03,
    }


def test_receptacle_scalar_wall_is_minimum_of_per_side_clearances() -> None:
    source = _receptacle()
    source["receptacle_configuration"]["geometry"]["wall_clearances_m"] = {
        "x_min": 0.02,
        "x_max": 0.04,
        "y_min": 0.02,
        "y_max": 0.03,
    }

    with pytest.raises(
        TaskEntityAssetCandidateError,
        match="task_entity_asset_receptacle_minimum_wall_clearance_mismatch",
    ):
        materialize_task_entity_asset_candidate(source)


@pytest.mark.parametrize(
    ("path", "expected_error"),
    [
        (("entity_id",), "task_entity_asset_entity_id_invalid"),
        (("asset_id",), "task_entity_asset_asset_id_invalid"),
        (
            ("simulator_import", "simulator"),
            "task_entity_asset_simulator_import_name_invalid",
        ),
        (
            ("simulator_import", "simulator_version"),
            "task_entity_asset_simulator_import_version_invalid",
        ),
    ],
)
def test_identifiers_and_strings_never_coerce_non_string_values(
    path: tuple[str, ...],
    expected_error: str,
) -> None:
    source = _receptacle()
    target = source
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = 7

    with pytest.raises(TaskEntityAssetCandidateError) as caught:
        materialize_task_entity_asset_candidate(source)
    assert expected_error in caught.value.errors
