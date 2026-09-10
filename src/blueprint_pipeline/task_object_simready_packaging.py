"""Attach reviewed candidate physics and CAD collision to an Astra visual asset.

The material/mass review, exact CAD and visual envelopes are prerequisites.
This package still requires the existing independent static/native/placement
gates; studio rendering alone never admits an object into an evaluation.
"""
from __future__ import annotations

import json
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
    path = Path(record['path'])
    actual = file_record(path)
    if any(actual[key] != record[key] for key in ('sha256', 'size_bytes')):
        raise AssetAuthoringError('authoring_packaging_input_digest_mismatch')
    return path


def package_astra_candidate(*, request: AuthoringRequest,
                            authoring_result: Mapping[str, Any], output_root: Path,
                            physics_bounds: Mapping[str, list[float]]) -> dict[str, Any]:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils
    import numpy as np
    import trimesh
    from .task_evaluation_scene_configuration_content_agents_driver import _complete_candidate_physics

    if authoring_result.get('request_digest') != request.request_digest:
        raise AssetAuthoringError('authoring_packaging_request_mismatch')
    if authoring_result.get('status') != 'candidate_authored_pending_native_qualification':
        raise AssetAuthoringError('authoring_packaging_candidate_not_reviewed')
    review_path = _verified(authoring_result['physical_review'])
    review = PhysicalPropertyReviewResult.model_validate_json(review_path.read_text())
    review_input = PhysicalPropertyReviewInput.model_validate_json(
        _verified(authoring_result['physical_review_input']).read_text())
    recomputed = review_physical_properties(review_input, review.proposed)
    if recomputed.model_dump(mode='json') != review.model_dump(mode='json'):
        raise AssetAuthoringError('authoring_packaging_physics_review_not_reproducible')
    if review.accepted is None or review.blockers or review.claim_ceiling != 'development_only':
        raise AssetAuthoringError('authoring_packaging_physics_not_accepted')
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
    stl = _verified(authoring_result['cad']['stl'])
    cad = trimesh.load_mesh(stl, file_type='stl', process=True)
    if not isinstance(cad, trimesh.Trimesh) or not cad.is_watertight or cad.volume <= 0:
        raise AssetAuthoringError('authoring_packaging_cad_not_watertight')
    points = np.asarray(cad.vertices, dtype=float) / 1000.
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    offset = np.array([(minimum[0]+maximum[0])/2, (minimum[1]+maximum[1])/2, minimum[2]])
    points -= offset
    if any(abs(actual-expected) > request.maximum_export_error_m
           for actual, expected in zip(maximum-minimum, request.dimensions_m, strict=True)):
        raise AssetAuthoringError('authoring_packaging_collision_dimension_mismatch')
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
    # Constant-density CAD candidate COM/inertia, scaled to reviewed mass.
    # This uses the hollow CAD shape for a tray, not its enclosing solid box.
    tensor = np.asarray(cad.moment_inertia, dtype=float) * (1e-6 * mass_kg / cad.mass)
    principal, rotation = np.linalg.eigh(tensor)
    if np.any(principal <= 0) or not np.isfinite(principal).all():
        raise AssetAuthoringError('authoring_packaging_inertia_invalid')
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    orientation = Gf.Matrix4d(Gf.Matrix3d(*rotation.T.reshape(-1).tolist()),
                             Gf.Vec3d(0)).ExtractRotationQuat()
    center = np.asarray(cad.center_mass) / 1000. - offset
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*center.tolist()))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*principal.tolist()))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(orientation))
    collision = UsdGeom.Mesh.Define(stage, '/Asset/Collision/CadShape')
    collision.CreatePointsAttr([Gf.Vec3f(*v) for v in points.tolist()])
    collision.CreateFaceVertexCountsAttr([3] * len(cad.faces))
    collision.CreateFaceVertexIndicesAttr(cad.faces.reshape(-1).tolist())
    collision.CreateSubdivisionSchemeAttr('none')
    collision.CreatePurposeAttr('guide')
    collision.CreateVisibilityAttr('invisible')
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    approximation = 'convexDecomposition' if request.role == 'passive_destination' else 'convexHull'
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
    completion['cad_solid_volume_m3'] = float(cad.volume) * 1e-9
    completion['center_of_mass_authority'] = 'constant_density_cad_candidate'
    completion['inertia_authority'] = 'constant_density_cad_candidate_scaled_to_reviewed_mass'
    completion['collision_approximation'] = approximation
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
