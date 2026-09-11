"""Real USD/mesh readback protects final-surface collision, reviewed mass and units."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_astra_authoring import AssetAuthoringError, file_record
from blueprint_pipeline.task_object_simready_packaging import package_astra_candidate
from blueprint_pipeline.task_object_physical_property_review import (
    PhysicalPropertyReviewInput, PhysicalPropertyReviewProposal, review_physical_properties,
)


def _fixture(tmp_path, *, final_mesh=None, mass_kg=None):
    dimensions = [.3, .4, .02]
    mesh = final_mesh if final_mesh is not None else trimesh.creation.box(extents=dimensions)
    if final_mesh is None:
        mesh.apply_translation([0, 0, .01])
    stage = Usd.Stage.CreateNew(str(tmp_path / 'candidate.usdc'))
    root = UsdGeom.Xform.Define(stage, '/Asset')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    visual = UsdGeom.Mesh.Define(stage, '/Asset/Visual')
    visual.CreatePointsAttr([Gf.Vec3f(*p) for p in mesh.vertices])
    visual.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    visual.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    visual.CreateSubdivisionSchemeAttr('none')
    stage.GetRootLayer().Save()
    # Original CAD is deliberately the full box even when final visual shape changes.
    cad = trimesh.creation.box(extents=np.array(dimensions) * 1000)
    cad.export(tmp_path / 'cad.stl')
    (tmp_path/'asset_program.py').write_text('# Retained author program, not executed by packaging.\n')
    (tmp_path/'final_visual_mesh.json').write_text(json.dumps(dict(schema_version='final_visual_mesh.v1',
        units='metres', coordinate_frame='center_XY_bottom_Z', vertices_m=mesh.vertices.tolist(), faces=mesh.faces.tolist())))
    receipt = dict(schema_version='final_visual_mesh_receipt.v1', claim_ceiling='development_only',
        mesh_file='final_visual_mesh.json', mesh_sha256=file_record(tmp_path/'final_visual_mesh.json')['sha256'],
        candidate_usd_file='candidate.usdc', candidate_usd_sha256=file_record(tmp_path/'candidate.usdc')['sha256'],
        author_program_file='asset_program.py', author_program_sha256=file_record(tmp_path/'asset_program.py')['sha256'],
        source_cad_stl_sha256=file_record(tmp_path/'cad.stl')['sha256'], dimensions_m=mesh.extents.tolist(),
        volume_m3=float(mesh.volume), watertight=True, winding_consistent=True, connected_components=1,
        physics_authority='packaging_accepted_physical_review_only', exported_rigid_body_count=0)
    receipt['receipt_digest'] = canonical_digest(receipt, digest_field='receipt_digest')
    (tmp_path/'final_visual_mesh_receipt.json').write_text(json.dumps(receipt))
    def value(v, low, high):
        return dict(value=v, basis='estimated', interval=dict(lower=low, upper=high),
                    rationale='Fixture model', uncertainty='Explicit interval', evidence_ids=['fixture'])
    mass_kg = mass_kg if mass_kg is not None else 800 * mesh.volume
    properties = dict(mass_kg=value(mass_kg, mass_kg*.95, mass_kg*1.04), static_friction=value(.6, .55, .7),
                      dynamic_friction=value(.4, .3, .5), restitution=value(.05, 0., .1))
    proposal = dict(object_id='fixture', dimensions={axis:value(v, v*.95, v*1.05)
        for axis,v in zip(('x_m','y_m','z_m'),dimensions)}, properties=properties,
        optical_material=dict(name='Opaque',transmission=0.,opacity=1.),
        mass_model=dict(method='density_fill',density_kg_m3=dict(lower=750.,upper=850.),
            envelope_fill_fraction=dict(lower=.4,upper=1.),sheet_count=None,sheet_area_m2=None,
            grammage_g_m2=None,cover_mass_kg=None,rationale='Fixture',uncertainty='Range',evidence_ids=['fixture']),
        review_rationale='Independent fixture review')
    review_input = PhysicalPropertyReviewInput.model_validate(dict(object_id='fixture',
        object_description='Paper candidate',material_description='Opaque paper',appearance='opaque',
        dimensions=proposal['dimensions'],measured={k:None for k in properties},proposed=None,
        optical_material=proposal['optical_material'],admitted_restitution=dict(lower=0.,upper=.2),
        evidence=[dict(evidence_id='fixture',uri='retained://fixture',sha256='a'*64,
                       kind='primary_reference',excerpt='Fixture density range')]))
    review = review_physical_properties(review_input,PhysicalPropertyReviewProposal.model_validate(proposal)).model_dump(mode='json')
    assert review['accepted'] is not None
    (tmp_path/'review.json').write_text(json.dumps(review))
    (tmp_path/'review_input.json').write_text(review_input.model_dump_json())
    geometry = dict(dimensions_m=dimensions,minimum_z_m=0.,center_xy_m=[0.,0.],materials=[dict(alpha=1.,transmission=0.)])
    (tmp_path/'geometry.json').write_text(json.dumps(geometry))
    request = SimpleNamespace(request_digest='sha256:'+'a'*64,object_id='fixture',role='task_object',
        dimensions_m=dimensions,maximum_export_error_m=.00001,physical_review_input=review_input)
    result = dict(request_digest=request.request_digest,status='candidate_authored_pending_native_qualification',
        physical_review=file_record(tmp_path/'review.json'),physical_review_input=file_record(tmp_path/'review_input.json'),
        geometry_readback=file_record(tmp_path/'geometry.json'),asset=file_record(tmp_path/'candidate.usdc'),
        cad={'stl':file_record(tmp_path/'cad.stl'),'readback':{'volume_mm3':float(cad.volume)}},
        final_visual_mesh=file_record(tmp_path/'final_visual_mesh.json'),
        final_visual_mesh_receipt=file_record(tmp_path/'final_visual_mesh_receipt.json'))
    result['result_digest'] = canonical_digest(result)
    return request, result, mesh


def _package(tmp_path, request, result):
    return package_astra_candidate(request=request,authoring_result=result,output_root=tmp_path/'packaged',
        physics_bounds={'mass_kg':[.05,2.],'static_friction':[.2,1.],'dynamic_friction':[.15,1.],'restitution':[0.,.2]})


def _reseal(tmp_path, result, **receipt_changes):
    path = tmp_path/'final_visual_mesh_receipt.json'
    receipt = json.loads(path.read_text())
    receipt.update(receipt_changes)
    receipt['receipt_digest'] = canonical_digest(receipt, digest_field='receipt_digest')
    path.write_text(json.dumps(receipt))
    result.update(final_visual_mesh_receipt=file_record(path),asset=file_record(tmp_path/'candidate.usdc'),
                  final_visual_mesh=file_record(tmp_path/'final_visual_mesh.json'))
    result['result_digest'] = canonical_digest(result,digest_field='result_digest')


def _dented_mesh():
    mesh = trimesh.creation.box(extents=[.3,.4,.02])
    mesh.apply_translation([0,0,.01])
    top = np.isclose(mesh.vertices[:,2],.02)
    vertices = np.vstack([mesh.vertices,[.025,.035,.005]])
    ring = np.flatnonzero(top)
    ring = ring[np.argsort(np.arctan2(vertices[ring,1],vertices[ring,0]))]
    faces = mesh.faces[~np.all(top[mesh.faces],axis=1)].tolist()
    faces.extend([[int(ring[i]),int(ring[(i+1)%4]),8] for i in range(4)])
    return trimesh.Trimesh(vertices=vertices,faces=faces,process=False)


def test_packaged_metric_units_reviewed_mass_and_inertia_survive_usdz_readback(tmp_path):
    request,result,mesh = _fixture(tmp_path)
    packaged = _package(tmp_path,request,result)
    reopened = Usd.Stage.Open(packaged['asset']['path'])
    body = UsdPhysics.MassAPI(reopened.GetDefaultPrim())
    assert body.GetMassAttr().Get() == pytest.approx(1.92)
    assert list(body.GetCenterOfMassAttr().Get()) == pytest.approx([0.,0.,.01])
    assert sorted(body.GetDiagonalInertiaAttr().Get()) == pytest.approx(sorted(
        [1.92*(.4**2+.02**2)/12,1.92*(.3**2+.02**2)/12,1.92*(.3**2+.4**2)/12]))
    assert packaged['physics_completion']['collision_dimensions_m'] == pytest.approx(request.dimensions_m)
    assert packaged['physics_completion']['modifications'] == []
    assert packaged['native_qualified'] is False


def test_curved_final_visual_drives_collision_center_inertia_and_decomposition(tmp_path):
    request,result,mesh = _fixture(tmp_path,final_mesh=_dented_mesh())
    packaged = _package(tmp_path,request,result)
    completion = packaged['physics_completion']
    assert completion['final_visual_solid_volume_m3'] == pytest.approx(mesh.volume)
    assert completion['center_of_mass_m'] == pytest.approx(mesh.center_mass)
    assert not np.allclose(mesh.center_mass,[0,0,.01])
    assert completion['collision_approximation'] == 'convexDecomposition'
    assert completion['native_collision_cooking_qualified'] is False
    reopened = Usd.Stage.Open(packaged['asset']['path'])
    collision = UsdGeom.Mesh(reopened.GetPrimAtPath('/Asset/Collision/FinalVisualShape'))
    assert np.asarray(collision.GetPointsAttr().Get()) == pytest.approx(mesh.vertices,abs=1e-7)
    assert list(collision.GetFaceVertexIndicesAttr().Get()) == mesh.faces.reshape(-1).tolist()
    body = UsdPhysics.MassAPI(reopened.GetDefaultPrim())
    basis = np.asarray(Gf.Matrix3d().SetRotate(Gf.Quatd(body.GetPrincipalAxesAttr().Get())))
    recovered = basis.T @ np.diag(body.GetDiagonalInertiaAttr().Get()) @ basis
    assert recovered == pytest.approx(mesh.moment_inertia * (body.GetMassAttr().Get()/mesh.mass),abs=1e-8)


def test_cad_based_mass_cannot_be_silently_reused_for_different_final_volume(tmp_path):
    request,result,_ = _fixture(tmp_path,final_mesh=_dented_mesh(),mass_kg=1.92)
    with pytest.raises(AssetAuthoringError,match='mass_inconsistent_with_final_geometry'):
        _package(tmp_path,request,result)
    assert not (tmp_path/'packaged').exists()


def test_author_physics_cannot_override_accepted_review(tmp_path):
    request,result,_ = _fixture(tmp_path)
    stage = Usd.Stage.Open(str(tmp_path/'candidate.usdc'))
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath('/Asset/Visual')).CreateMassAttr(1.2)
    stage.GetRootLayer().Save()
    _reseal(tmp_path,result,candidate_usd_sha256=file_record(tmp_path/'candidate.usdc')['sha256'])
    with pytest.raises(AssetAuthoringError,match='unreviewed_physics_in_visual'):
        _package(tmp_path,request,result)


def test_same_envelope_but_different_visual_triangles_are_rejected(tmp_path):
    request,result,_ = _fixture(tmp_path,final_mesh=_dented_mesh())
    stage = Usd.Stage.Open(str(tmp_path/'candidate.usdc'))
    visual = UsdGeom.Mesh(stage.GetPrimAtPath('/Asset/Visual'))
    points = visual.GetPointsAttr().Get()
    points[8] = Gf.Vec3f(.025,.035,.012)
    visual.GetPointsAttr().Set(points)
    stage.GetRootLayer().Save()
    _reseal(tmp_path,result,candidate_usd_sha256=file_record(tmp_path/'candidate.usdc')['sha256'])
    with pytest.raises(AssetAuthoringError,match='visual_mesh_mismatch'):
        _package(tmp_path,request,result)


def test_disconnected_mesh_cannot_pass_a_forged_watertight_receipt(tmp_path):
    request,result,_ = _fixture(tmp_path)
    path = tmp_path/'final_visual_mesh.json'
    split = trimesh.util.concatenate([trimesh.creation.box(),trimesh.creation.box(transform=trimesh.transformations.translation_matrix([3,0,0]))])
    data = json.loads(path.read_text())
    data.update(vertices_m=split.vertices.tolist(),faces=split.faces.tolist())
    path.write_text(json.dumps(data))
    _reseal(tmp_path,result,mesh_sha256=file_record(path)['sha256'])
    with pytest.raises(AssetAuthoringError,match='not_single_watertight_solid'):
        _package(tmp_path,request,result)


def test_tampered_final_mesh_receipt_and_missing_final_mesh_are_rejected(tmp_path):
    request,result,_ = _fixture(tmp_path)
    path = tmp_path/'final_visual_mesh_receipt.json'
    receipt = json.loads(path.read_text())
    receipt['volume_m3'] *= 2
    path.write_text(json.dumps(receipt))
    result['final_visual_mesh_receipt'] = file_record(path)
    result['result_digest'] = canonical_digest(result,digest_field='result_digest')
    with pytest.raises(AssetAuthoringError,match='final_mesh_receipt_invalid'):
        _package(tmp_path,request,result)
    del result['final_visual_mesh']
    result['result_digest'] = canonical_digest(result,digest_field='result_digest')
    with pytest.raises(AssetAuthoringError,match='input_record_missing'):
        _package(tmp_path,request,result)


def test_box_shell_interior_is_measured_not_inferred_from_outer_extents():
    from blueprint_pipeline.task_object_simready_packaging import validate_box_shell_interior
    xy_outer = [(-.165,-.24),(.165,-.24),(.165,.24),(-.165,.24)]
    xy_inner = [(-.16,-.235),(.16,-.235),(.16,.235),(-.16,.235)]
    vertices = [[x,y,z] for ring,z in [(xy_outer,0.),(xy_outer,.035),(xy_inner,.035),(xy_inner,.005)] for x,y in ring]
    quads = [[3,2,1,0],[12,13,14,15]]
    for i in range(4):
        j = (i+1)%4
        quads.extend([[i,j,j+4,i+4],[i+4,j+4,j+8,i+8],[i+12,i+8,j+8,j+12]])
    faces = [[q[0],q[1],q[2]] for q in quads] + [[q[0],q[2],q[3]] for q in quads]
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    interior = {'minimum':[-.16,-.235,.005],'maximum':[.16,.235,.035]}
    result = validate_box_shell_interior(mesh,interior)
    assert result['empty_open_box_shell_verified'] is True
    assert result['interior_dimensions_m'] == pytest.approx([.32,.47,.03])
    assert result['observed_solid_volume_m3'] == pytest.approx(.001032)
    filled = trimesh.creation.box(extents=[.33,.48,.035])
    filled.apply_translation([0,0,.0175])
    with pytest.raises(AssetAuthoringError,match='solid_volume_mismatch'):
        validate_box_shell_interior(filled,interior)
    with pytest.raises(AssetAuthoringError,match='solid_volume_mismatch'):
        validate_box_shell_interior(mesh,{'minimum':[-.16,-.235,.008],'maximum':[.16,.235,.035]})


def test_asset_environment_lights_are_excluded_without_changing_geometry_or_physics(tmp_path):
    from pxr import UsdLux
    request, result, _ = _fixture(tmp_path)
    source = Usd.Stage.Open(str(tmp_path/'candidate.usdc'))
    UsdLux.DomeLight.Define(source, '/Asset/env_light').CreateIntensityAttr(1.)
    source.GetRootLayer().Save()
    _reseal(tmp_path, result, candidate_usd_sha256=file_record(tmp_path/'candidate.usdc')['sha256'])
    original = file_record(tmp_path/'candidate.usdc')
    packaged = _package(tmp_path, request, result)
    stage = Usd.Stage.Open(packaged['asset']['path'])
    assert not [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]
    assert Usd.Stage.Open(original['path']).GetPrimAtPath('/Asset/env_light')
    assert file_record(tmp_path/'candidate.usdc') == original
    assert packaged['physics_completion']['excluded_asset_lighting'][0]['path'] == '/Asset/env_light'
    assert packaged['physics_completion']['illumination_authority'] == 'site_scene_only'
    assert UsdPhysics.MassAPI(stage.GetDefaultPrim()).GetMassAttr().Get() == pytest.approx(1.92)
