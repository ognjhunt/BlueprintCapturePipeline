"""Attach reviewed candidate physics to the exact final evaluated visual mesh.

Original source/CAD evidence is retained; final geometry supplies collision,
constant-density candidate COM/inertia, and a fresh mass-model consistency check.
This package still requires the existing independent static/native/placement
gates; studio rendering alone never admits an object into an evaluation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_object_astra_authoring import (
    AssetAuthoringError, AuthoringRequest, file_record, save_json,
    validate_geometry_readback,
)
from .task_object_physical_property_review import (
    PhysicalPropertyReviewInput, PhysicalPropertyReviewResult, review_physical_properties,
)


def _verified(record: Mapping[str, Any]) -> Path:
    if not isinstance(record, Mapping) or not all(key in record for key in ('path', 'sha256', 'size_bytes')):
        raise AssetAuthoringError('authoring_packaging_input_record_missing')
    path = Path(record['path'])
    actual = file_record(path)
    if any(actual[key] != record[key] for key in ('sha256', 'size_bytes')):
        raise AssetAuthoringError('authoring_packaging_input_digest_mismatch')
    return path



def _final_visual_mesh(*, request: AuthoringRequest, authoring_result: Mapping[str, Any], source: Any):
    """Independently verify mesh/receipt/visual triangle equivalence in metric space."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree
    from pxr import Gf, Usd, UsdGeom

    mesh_path = _verified(authoring_result.get('final_visual_mesh', {}))
    receipt_path = _verified(authoring_result.get('final_visual_mesh_receipt', {}))
    receipt = json.loads(receipt_path.read_text())
    asset = _verified(authoring_result['asset'])
    cad = _verified(authoring_result['cad']['stl'])
    if (receipt.get('schema_version') != 'final_visual_mesh_receipt.v1'
            or receipt.get('claim_ceiling') != 'development_only'
            or receipt.get('receipt_digest') != canonical_digest(receipt, digest_field='receipt_digest')
            or receipt.get('mesh_file') != mesh_path.name
            or receipt.get('mesh_sha256') != file_record(mesh_path)['sha256']
            or receipt.get('candidate_usd_file') != asset.name
            or receipt.get('candidate_usd_sha256') != file_record(asset)['sha256']
            or receipt.get('source_cad_stl_sha256') != file_record(cad)['sha256']
            or receipt.get('physics_authority') != 'packaging_accepted_physical_review_only'
            or receipt.get('exported_rigid_body_count') != 0):
        raise AssetAuthoringError('authoring_packaging_final_mesh_receipt_invalid')
    program_name = str(receipt.get('author_program_file') or '')
    if not program_name or Path(program_name).name != program_name:
        raise AssetAuthoringError('authoring_packaging_final_mesh_program_invalid')
    program = file_record(mesh_path.parent / program_name)
    if program['sha256'] != receipt.get('author_program_sha256'):
        raise AssetAuthoringError('authoring_packaging_final_mesh_program_invalid')
    payload = json.loads(mesh_path.read_text())
    if (payload.get('schema_version') != 'final_visual_mesh.v1' or payload.get('units') != 'metres'
            or payload.get('coordinate_frame') != 'center_XY_bottom_Z'):
        raise AssetAuthoringError('authoring_packaging_final_mesh_frame_invalid')
    vertices = np.asarray(payload.get('vertices_m'), dtype=float)
    faces = np.asarray(payload.get('faces'))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or len(faces) < 4 or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(vertices)):
        raise AssetAuthoringError('authoring_packaging_final_mesh_topology_invalid')
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if (not mesh.is_watertight or not mesh.is_winding_consistent or mesh.body_count != 1
            or not math.isfinite(mesh.volume) or mesh.volume <= 0 or np.any(mesh.area_faces <= 0)):
        raise AssetAuthoringError('authoring_packaging_final_mesh_not_single_watertight_solid')
    if (receipt.get('watertight') is not True or receipt.get('winding_consistent') is not True
            or receipt.get('connected_components') != 1
            or not isinstance(receipt.get('volume_m3'), (int, float))
            or not math.isclose(mesh.volume, receipt['volume_m3'], rel_tol=1e-8, abs_tol=1e-12)
            or len(receipt.get('dimensions_m', [])) != 3
            or not np.allclose(mesh.extents, receipt['dimensions_m'], rtol=0, atol=1e-8)):
        raise AssetAuthoringError('authoring_packaging_final_mesh_metrics_mismatch')
    minimum, maximum = mesh.bounds
    if (not np.allclose(mesh.extents, request.dimensions_m, rtol=0, atol=request.maximum_export_error_m)
            or abs(minimum[2]) > request.maximum_export_error_m
            or np.any(np.abs((minimum[:2] + maximum[:2]) / 2) > request.maximum_export_error_m)):
        raise AssetAuthoringError('authoring_packaging_final_mesh_dimension_or_origin_mismatch')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    if not np.allclose(np.asarray(cache.GetLocalToWorldTransform(source.GetDefaultPrim())), np.eye(4), rtol=0, atol=1e-12):
        raise AssetAuthoringError('authoring_packaging_visual_root_transform_invalid')
    tree, visual_faces = cKDTree(vertices), []
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        visual = UsdGeom.Mesh(prim)
        if visual.ComputeVisibility() == 'invisible' or visual.ComputePurpose() not in ('default', 'render'):
            continue
        counts = list(visual.GetFaceVertexCountsAttr().Get() or [])
        indices = np.asarray(visual.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if not counts or set(counts) != {3} or len(indices) != 3 * len(counts):
            raise AssetAuthoringError('authoring_packaging_visual_mesh_not_triangular')
        transform = cache.GetLocalToWorldTransform(prim)
        world = np.array([transform.Transform(Gf.Vec3d(*point)) for point in visual.GetPointsAttr().Get()])
        distances, ids = tree.query(world)
        if (not np.isfinite(world).all() or distances.max() > 1e-7
                or indices.min() < 0 or indices.max() >= len(world)):
            raise AssetAuthoringError('authoring_packaging_visual_mesh_mismatch')
        visual_faces.extend(ids[indices.reshape(-1, 3)].tolist())
    # Vertex order and mesh names may change on USD export; the triangles may not.
    if not visual_faces or sorted(map(tuple, np.sort(visual_faces, axis=1))) != sorted(map(tuple, np.sort(faces, axis=1))):
        raise AssetAuthoringError('authoring_packaging_visual_mesh_mismatch')
    return mesh, receipt, {'mesh': file_record(mesh_path), 'receipt': file_record(receipt_path),
                           'author_program': program, 'source_visual': file_record(asset), 'source_cad_stl': file_record(cad)}


def _final_mass_consistency(review: PhysicalPropertyReviewResult, mesh: Any) -> dict[str, Any]:
    """Recheck retained mass against final geometry without narrowing its uncertainty."""
    accepted = review.accepted
    assert accepted is not None
    model, mass = accepted.mass_model, accepted.properties.mass_kg
    result = {'final_solid_volume_m3': float(mesh.volume), 'accepted_mass_kg': mass.value,
              'accepted_mass_interval_kg': [mass.interval.lower, mass.interval.upper],
              'implied_density_kg_m3': mass.value / float(mesh.volume),
              'uncertainty_preserved': True, 'physical_truth_claimed': False}
    if model is not None and model.method == 'density_fill':
        density, fill = model.density_kg_m3, model.envelope_fill_fraction
        assert density is not None and fill is not None
        envelope = math.prod(getattr(accepted.dimensions, axis).value for axis in ('x_m', 'y_m', 'z_m'))
        fraction = float(mesh.volume) / envelope
        low, high = density.lower * mesh.volume, density.upper * mesh.volume
        if not fill.lower - 1e-10 <= fraction <= fill.upper + 1e-10 or not low - 1e-10 <= mass.value <= high + 1e-10:
            raise AssetAuthoringError('authoring_packaging_review_mass_inconsistent_with_final_geometry')
        result.update(method='reviewed_density_times_final_solid_volume', final_fill_fraction=fraction,
                      reviewed_density_kg_m3=[density.lower, density.upper],
                      final_geometry_density_mass_interval_kg=[float(low), float(high)])
    else:
        result['method'] = 'measured_or_page_model_mass_preserved_with_verified_dimensions'
    return result


def package_astra_candidate(*, request: AuthoringRequest,
                            authoring_result: Mapping[str, Any], output_root: Path,
                            physics_bounds: Mapping[str, list[float]]) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    import numpy as np
    from .task_evaluation_scene_configuration_content_agents_driver import _complete_candidate_physics

    if authoring_result.get('request_digest') != request.request_digest:
        raise AssetAuthoringError('authoring_packaging_request_mismatch')
    if authoring_result.get('status') != 'candidate_authored_pending_native_qualification':
        raise AssetAuthoringError('authoring_packaging_candidate_not_reviewed')
    if authoring_result.get('result_digest') != canonical_digest(authoring_result, digest_field='result_digest'):
        raise AssetAuthoringError('authoring_packaging_result_digest_mismatch')
    review_path = _verified(authoring_result['physical_review'])
    review = PhysicalPropertyReviewResult.model_validate_json(review_path.read_text())
    review_input = PhysicalPropertyReviewInput.model_validate_json(
        _verified(authoring_result['physical_review_input']).read_text())
    recomputed = review_physical_properties(review_input, review.proposed)
    if recomputed.model_dump(mode='json') != review.model_dump(mode='json'):
        raise AssetAuthoringError('authoring_packaging_physics_review_not_reproducible')
    if review.accepted is None or review.blockers or review.claim_ceiling != 'development_only':
        raise AssetAuthoringError('authoring_packaging_physics_not_accepted')
    if (review_input.object_id != request.object_id
            or any(abs(getattr(review_input.dimensions, axis).value - expected) > 1e-12
                   for axis, expected in zip(('x_m', 'y_m', 'z_m'), request.dimensions_m, strict=True))):
        raise AssetAuthoringError('authoring_packaging_physics_identity_mismatch')
    properties = review.accepted.properties
    expected_names = {'mass_kg', 'static_friction', 'dynamic_friction', 'restitution'}
    if set(physics_bounds) != expected_names:
        raise AssetAuthoringError('authoring_packaging_physics_bounds_invalid')
    for name in expected_names:
        value = getattr(properties, name)
        lower, upper = physics_bounds[name]
        if not lower <= value.interval.lower <= value.value <= value.interval.upper <= upper:
            raise AssetAuthoringError('authoring_packaging_estimate_outside_admitted_bounds:' + name)
    measurement = json.loads(_verified(authoring_result['geometry_readback']).read_text())
    validate_geometry_readback(request, measurement, review_input.appearance)
    source = Usd.Stage.Open(str(_verified(authoring_result['asset'])))
    if source is None or source.GetDefaultPrim().GetPath() != Sdf.Path('/Asset'):
        raise AssetAuthoringError('authoring_packaging_visual_root_invalid')
    if abs(UsdGeom.GetStageMetersPerUnit(source) - 1.) > 1e-12 or UsdGeom.GetStageUpAxis(source) != 'Z':
        raise AssetAuthoringError('authoring_packaging_visual_frame_invalid')
    for prim in source.Traverse():
        if (any(schema.startswith(('Physics', 'Physx')) for schema in prim.GetAppliedSchemas())
                or prim.IsA(UsdPhysics.Joint)
                or any(prop.GetName().startswith(('physics:', 'physx')) for prop in prim.GetProperties())):
            raise AssetAuthoringError('authoring_packaging_unreviewed_physics_in_visual')
    mesh, mesh_receipt, geometry_sources = _final_visual_mesh(
        request=request, authoring_result=authoring_result, source=source)
    consistency = _final_mass_consistency(review, mesh)
    points = np.asarray(mesh.vertices, dtype=float)
    output_root.mkdir(parents=True, exist_ok=True)
    authored = output_root / 'astra_rigid_candidate.usdc'
    if authored.exists():
        raise AssetAuthoringError('authoring_packaging_output_exists')
    # Flatten resolves source texture paths before writing a new root layer.
    source.Flatten().Export(str(authored))
    stage = Usd.Stage.Open(str(authored))
    root = UsdGeom.Xform.Define(stage, '/Asset').GetPrim()
    stage.SetDefaultPrim(root)
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass_kg = properties.mass_kg.value
    mass.CreateMassAttr(mass_kg)
    # Final metric solid geometry supplies candidate COM/inertia; reviewed mass wins.
    tensor = np.asarray(mesh.moment_inertia, dtype=float) * (mass_kg / mesh.mass)
    principal, rotation = np.linalg.eigh(tensor)
    if np.any(principal <= 0) or not np.isfinite(principal).all():
        raise AssetAuthoringError('authoring_packaging_inertia_invalid')
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    orientation = Gf.Matrix4d(Gf.Matrix3d(*rotation.T.reshape(-1).tolist()),
                             Gf.Vec3d(0)).ExtractRotationQuat()
    center = np.asarray(mesh.center_mass)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*center.tolist()))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*principal.tolist()))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(orientation))
    collision = UsdGeom.Mesh.Define(stage, '/Asset/Collision/FinalVisualShape')
    collision.CreatePointsAttr([Gf.Vec3f(*v) for v in points.tolist()])
    collision.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    collision.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    collision.CreateSubdivisionSchemeAttr('none')
    collision.CreatePurposeAttr('guide')
    collision.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    approximation = 'convexDecomposition' if request.role == 'passive_destination' or not mesh.is_convex else 'convexHull'
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(approximation)
    material = UsdShade.Material.Define(stage, '/Asset/Looks/ReviewedPhysicsCandidate')
    contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact.CreateStaticFrictionAttr(properties.static_friction.value)
    contact.CreateDynamicFrictionAttr(properties.dynamic_friction.value)
    contact.CreateRestitutionAttr(properties.restitution.value)
    UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(material,
        UsdShade.Tokens.weakerThanDescendants, 'physics')
    stage.GetRootLayer().Save()
    completion = _complete_candidate_physics(authored, bounds=dict(physics_bounds))
    # The compatibility completer must not silently repair reviewed values.
    if completion.get('modifications'):
        raise AssetAuthoringError('authoring_reviewed_physics_changed_by_completion')
    completion['astra_review'] = file_record(review_path)
    completion['final_visual_solid_volume_m3'] = float(mesh.volume)
    completion['final_visual_mesh_sources'] = geometry_sources
    completion['final_visual_mesh_receipt_digest'] = mesh_receipt['receipt_digest']
    completion['mass_model_final_geometry_consistency'] = consistency
    completion['original_cad_readback'] = authoring_result['cad'].get('readback')
    completion['center_of_mass_authority'] = 'constant_density_final_visual_candidate'
    completion['inertia_authority'] = 'constant_density_final_visual_candidate_scaled_to_reviewed_mass'
    completion['collision_approximation'] = approximation
    completion['collision_source_matches_final_visual_mesh'] = True
    completion['native_collision_cooking_qualified'] = False
    completion['completion_digest'] = canonical_digest(completion, digest_field='completion_digest')
    asset = output_root / 'astra_replacement_candidate.usdz'
    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(authored)), str(asset)):
        raise AssetAuthoringError('authoring_usdz_packaging_failed')
    reopened = Usd.Stage.Open(str(asset))
    if reopened is None or not reopened.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI):
        raise AssetAuthoringError('authoring_usdz_readback_failed')
    receipt = {'asset': file_record(asset), 'physics_completion': completion,
               'request_digest': request.request_digest,
               'authoring_result_digest': authoring_result.get('result_digest'),
               'claim_ceiling': 'development_only', 'native_qualified': False}
    save_json(output_root / 'astra_packaging_receipt.json', receipt)
    return receipt


def validate_box_shell_interior(mesh: Any, interior: Mapping[str, Any]) -> dict[str, Any]:
    """Prove an axis-aligned open box shell has the declared empty interior."""
    import numpy as np
    low, high = np.asarray(interior['minimum'], dtype=float), np.asarray(interior['maximum'], dtype=float)
    outer_low, outer_high = mesh.bounds
    tolerance = 1e-7  # Exported float32 metre coordinates, not physical measurement precision.
    if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
            or np.any(low >= high) or np.any(low[:2] <= outer_low[:2]) or np.any(high[:2] >= outer_high[:2])
            or low[2] <= outer_low[2] or abs(high[2] - outer_high[2]) > tolerance
            or not mesh.is_watertight or not mesh.is_winding_consistent or mesh.body_count != 1):
        raise AssetAuthoringError('astra_destination_interior_invalid')
    expected_volume = float(np.prod(outer_high-outer_low) - np.prod(high-low))
    if not math.isclose(float(mesh.volume), expected_volume, rel_tol=1e-6, abs_tol=1e-10):
        raise AssetAuthoringError('astra_destination_interior_solid_volume_mismatch')
    for triangle in mesh.triangles:
        # Outer sides/bottom, inner sides/floor, or one of the four top rim strips.
        outer_face = any(np.all(np.abs(triangle[:, axis] - bound) <= tolerance)
                         for axis in (0, 1) for bound in (outer_low[axis], outer_high[axis]))
        outer_bottom = np.all(np.abs(triangle[:, 2] - outer_low[2]) <= tolerance)
        within_inner = np.all(triangle >= low-tolerance) and np.all(triangle <= high+tolerance)
        inner_face = within_inner and (
            np.all(np.abs(triangle[:, 2] - low[2]) <= tolerance)
            or any(np.all(np.abs(triangle[:, axis] - bound) <= tolerance)
                   for axis in (0, 1) for bound in (low[axis], high[axis])))
        rim = (np.all(np.abs(triangle[:, 2] - high[2]) <= tolerance)
               and any(np.all(triangle[:, axis] <= low[axis]+tolerance)
                       or np.all(triangle[:, axis] >= high[axis]-tolerance) for axis in (0, 1)))
        if not (outer_face or outer_bottom or inner_face or rim):
            raise AssetAuthoringError('astra_destination_interior_surface_mismatch')
    return {'interior_bounds_body_frame_m': {'minimum': low.tolist(), 'maximum': high.tolist()},
            'interior_dimensions_m': (high-low).tolist(), 'empty_open_box_shell_verified': True,
            'geometry_comparison_tolerance_m': tolerance, 'observed_solid_volume_m3': float(mesh.volume),
            'expected_box_shell_volume_m3': expected_volume}


def qualify_astra_box_shell_interior(*, packaging_receipt: Mapping[str, Any],
                                     static_qualification: Mapping[str, Any],
                                     interior_bounds_body_frame_m: Mapping[str, Any],
                                     output_path: Path) -> dict[str, Any]:
    """Bind actual final-mesh cavity evidence to the statically qualified USDZ."""
    import trimesh
    from pxr import Usd, UsdGeom
    asset = file_record(_verified(packaging_receipt['asset']))
    if (static_qualification.get('status') != 'authored_structure_statically_qualified'
            or static_qualification.get('result_digest') != canonical_digest(static_qualification, digest_field='result_digest')
            or static_qualification.get('replacement_usd', {}).get('sha256') != asset['sha256']):
        raise AssetAuthoringError('astra_destination_static_binding_invalid')
    sources = packaging_receipt['physics_completion']['final_visual_mesh_sources']
    mesh_record = file_record(_verified(sources['mesh']))
    # Measure the packaged collision itself, not merely a supplied mesh sidecar.
    stage = Usd.Stage.Open(asset['path'])
    collider = UsdGeom.Mesh(stage.GetPrimAtPath('/Asset/Collision/FinalVisualShape'))
    counts = list(collider.GetFaceVertexCountsAttr().Get() or [])
    if not counts or set(counts) != {3}:
        raise AssetAuthoringError('astra_destination_collision_mesh_invalid')
    indices = list(collider.GetFaceVertexIndicesAttr().Get())
    mesh = trimesh.Trimesh(vertices=list(collider.GetPointsAttr().Get()),
                           faces=[indices[index:index+3] for index in range(0, len(indices), 3)], process=False)
    geometry = validate_box_shell_interior(mesh, interior_bounds_body_frame_m)
    result = {'schema_version': 'task_evaluation_passive_destination_simready.v1',
              'destination_identity': static_qualification['replacement_identity'], 'asset': asset,
              'static_result_digest': static_qualification['result_digest'],
              **geometry, 'final_visual_mesh': mesh_record,
              'intended_support_prim_paths': ['/Asset'],
              'intended_support_collision_prim_paths': ['/Asset/Collision/FinalVisualShape'],
              'native_collision_cooking_qualified': False, 'claim_ceiling': 'development_only'}
    structure = static_qualification['observed_structure']
    if (result['intended_support_prim_paths'] != structure['rigid_body_paths']
            or result['intended_support_collision_prim_paths'] != structure['collision_prim_paths']):
        raise AssetAuthoringError('astra_destination_support_path_invalid')
    result['result_digest'] = canonical_digest(result, digest_field='result_digest')
    save_json(output_path, result)
    return result
