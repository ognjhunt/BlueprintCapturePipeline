from types import SimpleNamespace

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from blueprint_pipeline import native_rigid_friction_scenario as friction


def source_asset(path, *, material_count=1, articulated=False):
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/Object').GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.13)
    if articulated:
        UsdPhysics.ArticulationRootAPI.Apply(root)
    for index in range(material_count):
        material = UsdShade.Material.Define(stage, f'/Object/materials/physics{index}')
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateDynamicFrictionAttr(0.25)
        physics.CreateStaticFrictionAttr(0.7)
        physics.CreateRestitutionAttr(0.1)
        cube = UsdGeom.Cube.Define(stage, f'/Object/shape{index}').GetPrim()
        UsdPhysics.CollisionAPI.Apply(cube)
        UsdShade.MaterialBindingAPI.Apply(cube).Bind(material, materialPurpose='physics')
    stage.GetRootLayer().Save()


def prepare(path):
    objects = [{'task_subject': True, 'usd_path': str(path)}]
    applications = [{'readback_kind': friction.KIND, 'parameter_id': 'dynamic_friction',
                     'runtime_name': 'task_object', 'expected_native_value': 0.45}]
    return objects, friction.prepare(objects, applications)


def test_changes_bound_material_before_import_without_editing_source(tmp_path):
    path = tmp_path/'source.usda'
    source_asset(path)
    original = path.read_bytes()
    objects, result = prepare(path)
    try:
        stage = Usd.Stage.Open(objects[0]['usd_path'])
        material = UsdPhysics.MaterialAPI(stage.GetPrimAtPath('/Object/materials/physics0'))
        assert material.GetDynamicFrictionAttr().Get() == pytest.approx(0.45)
        assert material.GetStaticFrictionAttr().Get() == pytest.approx(0.7)
        assert material.GetRestitutionAttr().Get() == pytest.approx(0.1)
        assert UsdPhysics.MassAPI(stage.GetDefaultPrim()).GetMassAttr().Get() == pytest.approx(0.13)
        assert path.read_bytes() == original
        assert result['dynamic_friction']['source_dynamic_friction'] == 0.25
    finally:
        from pathlib import Path
        Path(objects[0]['usd_path']).unlink()


@pytest.mark.parametrize('kwargs,reason', [
    ({'material_count': 2}, 'material_not_unique'),
    ({'material_count': 0}, 'material_not_unique'),
    ({'articulated': True}, 'requires_rigid_subject'),
])
def test_refuses_to_guess_which_material_or_articulation_to_change(tmp_path, kwargs, reason):
    path = tmp_path/'source.usda'
    source_asset(path, **kwargs)
    with pytest.raises(ValueError, match=reason):
        prepare(path)


def test_native_readback_requires_every_physx_shape_to_receive_the_override():
    materials = np.array([[[0.7, 0.45, 0.1], [0.7, 0.45, 0.1]]], dtype=np.float32)
    subject = SimpleNamespace(root_physx_view=SimpleNamespace(get_material_properties=lambda: materials))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={'task_object': subject}))
    override = {'dynamic_friction': {'runtime_name': 'task_object', 'expected_dynamic_friction': 0.45}}
    result = friction.verify(env, override)
    assert result['dynamic_friction']['source'] == 'physx_material_properties'
    assert result['dynamic_friction']['observed_dynamic_friction'] == pytest.approx(0.45)
    materials[0, 1, 1] = 0.25
    with pytest.raises(ValueError, match='native_readback_mismatch'):
        friction.verify(env, override)


def test_scenarios_without_friction_do_not_change_assets():
    assert friction.prepare([{'task_subject': True, 'usd_path': 'unused.usd'}], []) == {}
    assert friction.verify(None, {}) == {}


def test_articulated_subject_keeps_explicit_link_binding_gap():
    applications, gaps = [], []
    friction.add_application({'dynamic_friction': 0.45}, {'object_type': 'ARTICULATION'}, applications, gaps)
    assert applications == []
    assert gaps == [{'family': 'bounded_physics', 'reason': 'runtime_material_link_binding_unavailable',
                     'fallback': 'canonical_task_material'}]
