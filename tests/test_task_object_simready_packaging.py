"""Real OpenUSD/STEP-derived collision protects units, mass and USDZ closure."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from blueprint_pipeline.task_object_astra_authoring import file_record
from blueprint_pipeline.task_object_simready_packaging import package_astra_candidate
from blueprint_pipeline.task_object_physical_property_review import (
    PhysicalPropertyReviewInput, PhysicalPropertyReviewProposal, review_physical_properties,
)


def test_packaged_cad_units_mass_and_inertia_survive_usdz_readback(tmp_path):
    dimensions = [.3, .4, .02]
    stage = Usd.Stage.CreateNew(str(tmp_path / 'visual.usdc'))
    root = UsdGeom.Xform.Define(stage, '/Asset')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    cube = UsdGeom.Cube.Define(stage, '/Asset/Visual')
    cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, .01))
    cube.AddScaleOp().Set(Gf.Vec3d(.15, .2, .01))
    stage.GetRootLayer().Save()
    mesh = trimesh.creation.box(extents=np.array(dimensions) * 1000)
    mesh.export(tmp_path / 'cad.stl')
    def value(v, low, high):
        return dict(value=v, basis='estimated', interval=dict(lower=low, upper=high),
                    rationale='Fixture model', uncertainty='Explicit interval', evidence_ids=['fixture'])
    properties = dict(mass_kg=value(.96, .9, 1.), static_friction=value(.6, .55, .7),
                      dynamic_friction=value(.4, .3, .5), restitution=value(.05, 0., .1))
    proposal = dict(object_id='fixture', dimensions={axis:value(v, v*.95, v*1.05)
        for axis,v in zip(('x_m','y_m','z_m'),dimensions)}, properties=properties,
        optical_material=dict(name='Opaque',transmission=0.,opacity=1.),
        mass_model=dict(method='density_fill',density_kg_m3=dict(lower=750.,upper=850.),
            envelope_fill_fraction=dict(lower=.45,upper=.65),sheet_count=None,sheet_area_m2=None,
            grammage_g_m2=None,cover_mass_kg=None,rationale='Fixture',uncertainty='Range',evidence_ids=['fixture']),
        review_rationale='Independent fixture review')
    review_input = PhysicalPropertyReviewInput.model_validate(dict(object_id='fixture',
        object_description='Paper candidate',material_description='Opaque paper',appearance='opaque',
        dimensions=proposal['dimensions'],measured={k:None for k in properties},proposed=None,
        optical_material=proposal['optical_material'],admitted_restitution=dict(lower=0.,upper=.2),
        evidence=[dict(evidence_id='fixture',uri='retained://fixture',sha256='a'*64,
                       kind='primary_reference',excerpt='Fixture density range')]))
    review = review_physical_properties(review_input,PhysicalPropertyReviewProposal.model_validate(proposal)).model_dump(mode='json')
    (tmp_path / 'review.json').write_text(json.dumps(review))
    (tmp_path / 'review_input.json').write_text(review_input.model_dump_json())
    geometry = dict(dimensions_m=dimensions,minimum_z_m=0.,center_xy_m=[0.,0.],
                    materials=[dict(alpha=1.,transmission=0.)])
    (tmp_path / 'geometry.json').write_text(json.dumps(geometry))
    request = SimpleNamespace(request_digest='sha256:'+'a'*64,role='task_object',
        dimensions_m=dimensions, maximum_export_error_m=.00001,
        physical_review_input=SimpleNamespace(appearance='opaque'))
    result = dict(request_digest=request.request_digest,
        status='candidate_authored_pending_native_qualification',
        physical_review=file_record(tmp_path/'review.json'),
        physical_review_input=file_record(tmp_path/'review_input.json'),
        geometry_readback=file_record(tmp_path/'geometry.json'),
        asset=file_record(tmp_path/'visual.usdc'), cad={'stl':file_record(tmp_path/'cad.stl')})
    packaged = package_astra_candidate(request=request,authoring_result=result,
        output_root=tmp_path/'packaged',physics_bounds={'mass_kg':[.05,2.],
        'static_friction':[.2,1.],'dynamic_friction':[.15,1.],'restitution':[0.,.2]})
    reopened = Usd.Stage.Open(packaged['asset']['path'])
    body = UsdPhysics.MassAPI(reopened.GetDefaultPrim())
    assert body.GetMassAttr().Get() == pytest.approx(.96)
    assert list(body.GetCenterOfMassAttr().Get()) == pytest.approx([0.,0.,.01])
    assert sorted(body.GetDiagonalInertiaAttr().Get()) == pytest.approx(sorted(
        [.96*(.4**2+.02**2)/12,.96*(.3**2+.02**2)/12,.96*(.3**2+.4**2)/12]))
    assert packaged['physics_completion']['collision_dimensions_m'] == pytest.approx(dimensions)
    assert packaged['physics_completion']['modifications'] == []
    assert packaged['native_qualified'] is False
