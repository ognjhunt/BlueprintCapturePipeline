"""Pinned public XML with synthetic meshes/SDK: software contracts, not native proof."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as XML

import numpy as np
import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_industrial_composition as composition
from blueprint_pipeline import deployment_embodiment_gofa_tool_model as model
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline import deployment_embodiment_runtime_compatibility as compatibility
from blueprint_pipeline.deployment_embodiment_station import _origin_matrix
from tests.test_deployment_embodiment_gofa import _stl, _sdk
from tests.test_deployment_embodiment_tool_composition import composed  # noqa: F401

FIXTURE = Path(__file__).parent / "fixtures/industrial_robotiq"


@pytest.fixture
def public_sources(tmp_path):
    inventory = json.loads((FIXTURE / "inventory.json").read_text())
    sets = {}
    for name, component in inventory.items():
        files = {}
        for row in component["files"]:
            if row["synthetic_geometry_in_tests"]:
                payload = b'<COLLADA/>' if row["path"].lower().endswith('.dae') else _stl()
            else:
                payload = (FIXTURE / name / row["path"]).read_bytes()
                assert len(payload) == row["size_bytes"]
                assert 'sha256:' + hashlib.sha256(payload).hexdigest() == row["sha256"]
            files[row["path"]] = payload
        # Several exact variants share a revision/repository and common paths.
        sets.setdefault((component['repository_url'], component['revision']), {}).update(files)

    def get(url, *, maximum):
        for (repo, revision), files in sets.items():
            if revision not in url or repo.removeprefix('https://github.com/') not in url:
                continue
            if '/git/trees/' in url:
                return json.dumps({'truncated': False, 'tree': [
                    {'path': name, 'type': 'blob', 'mode': '100644', 'size': len(data), 'sha': assets._git_blob_sha(data)}
                    for name, data in files.items()]}).encode()
            return files[url.split('/' + revision + '/', 1)[1]]
        pytest.fail('Unexpected source URL: ' + url)

    def materialize(identity):
        folder = tmp_path / (identity + '-source')
        assets.materialize_sources(identity, folder, get=get)
        return folder
    return materialize


@pytest.mark.parametrize('identity', composition.VARIANTS)
def test_exact_source_chain_mount_tool_closure_and_fake_sdk_binding(public_sources, tmp_path, monkeypatch, identity):
    bundle = public_sources(identity)
    before = {p: p.read_bytes() for p in (bundle / 'source').rglob('*') if p.is_file()}
    source = assets.verify_source_bundle(bundle)
    assert len(source['licenses']) == 2 and source['policy_compatibility_established'] is False
    config = tmp_path / 'configuration'
    record = runtime.prepare_runtime_configuration(bundle, config)
    assert runtime.verify_runtime_configuration(config) == record
    fields = record['manifest_fields']
    assert len(fields['joints']) == 6 and len(fields['grippers']) == 1
    assert [a['name'] for a in fields['arms']] == ['arm']
    assert len(fields['grippers'][0]['joints']) == 6
    assert [c['name'] for c in fields['cameras']] == ['overhead', 'arm_wrist']
    assert fields['arms'][0]['tool_from_tcp']['position_m'] == pytest.approx([0, 0, .134])
    root = XML.parse(config / 'robot.urdf').getroot()
    mount = next(j for j in root.findall('joint') if j.find('child').get('link') == 'robotiq_85_base_link')
    assert mount.find('parent').get('link') == composition.mount_descriptor(identity)['parent_link']
    assert np.array(_origin_matrix(mount)) == pytest.approx(np.eye(4))
    assert root.find("link[@name='gripper_mount_link']") is None
    flange_child = 'fanuc_flange' if identity.startswith('fanuc_') else 'tool0'
    flange = next(j for j in root.findall('joint') if j.find('child').get('link') == flange_child)
    expected_rpy = [np.pi, -np.pi/2, 0] if identity.startswith('fanuc_') else [0, np.pi/2, 0]
    assert [float(v) for v in flange.find('origin').get('rpy').split()] == pytest.approx(expected_rpy)
    assert _origin_matrix(flange)[0][2] == pytest.approx(1)
    assert all(p.read_bytes() == data for p, data in before.items())
    _sdk(monkeypatch)
    prepared = tmp_path / 'imported'
    imported = importer.import_source_asset(config, prepared, fix_base=True)
    assert imported['settings']['merge_fixed_joints'] is False
    assert importer.verify_prepared_asset_directory(prepared) == imported
    binding = runtime.bind_prepared_configuration(config, prepared, 'https://models.example/development-only.usda')
    assert binding['robot_embodiment']['embodiment_id'] == identity
    assert binding['execution_authorized'] is False
    if identity in model.VARIANTS:
        report = record['simulation_model']
        assert imported['simulation_model'] == report
        assert report['arm_model']['modeled_robot_mass_total_kg'] == pytest.approx(28 if '_5_95_' in identity else 48)
        assert report['tool_source_dynamics']['mass_total_kg'] == pytest.approx(.92086427304)
        assert report['combined_mass_total_kg'] == pytest.approx((28 if '_5_95_' in identity else 48)+.92086427304)
        observations = model.usd_values(prepared/'robot.usda', report)
        assert len(observations['arm_model']['links']) == 7
        assert len(observations['tool_source_dynamics']['links']) == 9
        assert len(observations['tool_source_dynamics']['joints']) == 6
        assert all(v['effort_limit_nm'] == 50 for v in observations['tool_source_dynamics']['joints'])


@pytest.mark.parametrize('fault', ['arm', 'tool', 'mount', 'claim'])
def test_unreviewed_source_mount_and_hardware_claim_cannot_enter_fixed_recipe(fault):
    recipe = deepcopy(assets.recipe_for(next(iter(composition.VARIANTS))))
    if fault == 'arm':
        recipe['source']['revision'] = 'a'*40
    elif fault == 'tool':
        recipe['composition']['tool_source']['revision'] = 'a'*40
    elif fault == 'mount':
        recipe['composition']['mount']['translation_m'][2] = .011
    else:
        recipe['composition']['publisher_hardware_match_established'] = True
    with pytest.raises(assets.AssetPreparationError):
        composition.component_recipe(recipe)


@pytest.mark.parametrize('fault', ['mass', 'tensor', 'drive', 'merged'])
def test_composite_usd_readback_refuses_tool_tampering(public_sources, tmp_path, monkeypatch, fault):
    from pxr import Gf, Usd, UsdPhysics
    identity = next(iter(model.VARIANTS))
    source = public_sources(identity)
    _sdk(monkeypatch)
    prepared = tmp_path/'imported'
    receipt = importer.import_source_asset(source, prepared, fix_base=True)
    path = prepared/'robot.usda'
    prepared.chmod(0o755)
    path.chmod(0o644)
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath('/robot/robotiq_85_base_link')
    if fault == 'mass':
        UsdPhysics.MassAPI(prim).GetMassAttr().Set(0.1)
    elif fault == 'tensor':
        UsdPhysics.MassAPI(prim).GetPrincipalAxesAttr().Set(Gf.Quatf(1, 0, 0, 0))
    elif fault == 'drive':
        UsdPhysics.DriveAPI(stage.GetPrimAtPath('/robot/robotiq_85_left_knuckle_joint'), 'angular').GetTypeAttr().Set('acceleration')
    else:
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    stage.GetRootLayer().Save()
    with pytest.raises(assets.AssetPreparationError, match='usd_tool_'):
        model.usd_values(path, receipt['simulation_model'])


def predecessor():
    path = Path(__file__).parent/'fixtures/runtime_configuration_537a62.py'
    assert 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest() == compatibility.CURRENT_23_GENERATOR
    spec = importlib.util.spec_from_file_location('blueprint_pipeline._previous23', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module._PROFILES) == compatibility.CURRENT_23_PROFILES
    assert all(runtime._PROFILES[key] == value for key, value in module._PROFILES.items())
    return module


def test_original23_configuration_reuse_and_metadata_reconstruction(public_sources, tmp_path, monkeypatch):
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal
    old = predecessor()
    source = public_sources('abb_gofa_crb15000_5_95')
    folder = tmp_path/'old-configuration'
    record = old.prepare_runtime_configuration(source, folder)
    original = (folder/runtime.RECEIPT).read_bytes()
    monkeypatch.setattr(runtime, '_expand_xacro', lambda *a, **k: pytest.fail('No historical reexpansion'))
    assert runtime.verify_runtime_configuration(folder) == record
    assert (folder/runtime.RECEIPT).read_bytes() == original
    changed = deepcopy(record)
    changed['manifest_fields']['cameras'][0]['position_m'][0] += .01
    _reseal(folder, changed)
    with pytest.raises(assets.AssetPreparationError, match='predecessor_metadata_changed'):
        runtime.verify_runtime_configuration(folder)


def test_original23_profiles_29_recipes_104_rows_and13_policy_sources_are_unchanged():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    previous = json.loads((FIXTURE/'predecessor_identities.json').read_text())
    package = Path(assets.__file__).parent
    root = package.parents[1]
    recipes = {r['embodiment_id']: r for r in assets.asset_recipes()['recipes']}
    catalog = {r['candidate_id']: r for r in json.loads((root/'config/beta_policy_candidates.v1.json').read_text())['candidates']}
    assert len(recipes) == 49 and len(catalog) == 136 and len(runtime._PROFILES) == 42
    assert len(previous['recipes']) == 29 and len(previous['catalog']) == 104
    for name, expected in previous['recipes'].items():
        assert canonical_digest(recipes[name]) == expected
    for name, expected in previous['catalog'].items():
        assert canonical_digest(catalog[name]) == expected
    for name, expected in previous['modules'].items():
        assert 'sha256:'+hashlib.sha256((package/name).read_bytes()).hexdigest() == expected
    current = json.loads((root/'config/policy_runtimes/scripted_robot_baseline_registration_profile.json').read_text())
    assert len(current['bodies']) == 24 and len(previous['registration_profile']['bodies']) == 13
    for key, expected in previous['registration_profile'].items():
        if key == 'bodies':
            assert {body: current[key][body] for body in expected} == expected
        else:
            assert current[key] == expected
    predecessor()


def _poses(root, angles):
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}
    cache = {}
    def pose(link):
        if link not in cache:
            if link not in joints:
                cache[link] = np.eye(4)
            else:
                joint = joints[link]
                local = np.array(_origin_matrix(joint))
                q = angles.get(joint.get('name'), 0.)
                if joint.get('type') == 'revolute':
                    axis = np.array([float(v) for v in joint.find('axis').get('xyz').split()])
                    axis /= np.linalg.norm(axis)
                    x, y, z = axis
                    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    rotation = np.eye(4)
                    rotation[:3, :3] = np.eye(3)+np.sin(q)*cross+(1-np.cos(q))*(cross@cross)
                    local = local@rotation
                cache[link] = pose(joint.find('parent').get('link'))@local
        return cache[link]
    return {link.get('name'): pose(link.get('name')) for link in root.findall('link')}


@pytest.mark.parametrize('identity', composition.VARIANTS)
def test_source_arm_fk_is_preserved_through_full_flange_and_ideal_mount(public_sources, tmp_path, identity):
    bare, combined = tmp_path/'bare', tmp_path/'combined'
    runtime.prepare_runtime_configuration(public_sources(composition.VARIANTS[identity]), bare)
    runtime.prepare_runtime_configuration(public_sources(identity), combined)
    a, b = XML.parse(bare/'robot.urdf').getroot(), XML.parse(combined/'robot.urdf').getroot()
    arm_names = [j.get('name') for j in a.findall('joint') if j.get('type') == 'revolute']
    assert len(arm_names) == 6
    for scalar in (0., .05, -.1):
        angles = {name: scalar*(index+1) for index, name in enumerate(arm_names)}
        ap, bp = _poses(a, angles), _poses(b, angles)
        for link in ap:
            if link in ('world', 'ee_link'):
                continue
            assert np.linalg.inv(ap['base_link'])@ap[link] == pytest.approx(np.linalg.inv(bp['base_link'])@bp[link], abs=1e-12)
        parent = composition.mount_descriptor(identity)['parent_link']
        assert np.linalg.inv(bp[parent])@bp['robotiq_85_base_link'] == pytest.approx(np.eye(4), abs=1e-12)
        assert np.linalg.inv(bp['robotiq_85_base_link'])@bp['grasp_link'] == pytest.approx(
            np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, .134], [0, 0, 0, 1]]), abs=1e-12)
