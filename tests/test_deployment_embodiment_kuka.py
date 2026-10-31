"""Exact source XML with synthetic meshes/SDK; these are not native/GPU proofs."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as XML

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline import deployment_embodiment_kuka_bridge as bridge
from blueprint_pipeline import deployment_embodiment_kuka_model as model
from tests.test_deployment_embodiment_gofa import _stl, _sdk
from tests.test_deployment_embodiment_industrial import _poses
from tests.scripted_controller_supersession import current as with_reviewed_revisions

FIXTURE = Path(__file__).parent / "fixtures/kuka_robotiq"


@pytest.fixture
def public_sources(tmp_path):
    record = json.loads((FIXTURE / "inventory.json").read_text())
    files = {}
    for row in record["files"]:
        path = row["path"]
        if path.endswith(".obj"):
            data = b"v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n"
        elif path.endswith(".png"):
            data = b"development-only-thumbnail-not-rendered"
        else:
            data = (FIXTURE / "arm" / path).read_bytes()
            assert "sha256:" + hashlib.sha256(data).hexdigest() == row["sha256"]
        files[path] = data
    industrial = FIXTURE.parent / "industrial_robotiq"
    tool = json.loads((industrial / "inventory.json").read_text())["tool"]
    tool_files = {
        row["path"]: (
            (b"<COLLADA/>" if row["path"].lower().endswith(".dae") else _stl())
            if row["synthetic_geometry_in_tests"]
            else (industrial / "tool" / row["path"]).read_bytes()
        )
        for row in tool["files"]
    }
    sets = [(record, files), (tool, tool_files)]

    def get(url, *, maximum):
        for source, payloads in sets:
            if source["revision"] not in url:
                continue
            if "/git/trees/" in url:
                return json.dumps(
                    {
                        "truncated": False,
                        "tree": [
                            {
                                "path": p,
                                "type": "blob",
                                "mode": "100644",
                                "size": len(data),
                                "sha": assets._git_blob_sha(data),
                            }
                            for p, data in payloads.items()
                        ],
                    }
                ).encode()
            return payloads[url.split("/" + source["revision"] + "/", 1)[1]]
        pytest.fail("Unexpected public source URL: " + url)

    def prepare(identity=bridge.IDENTITY):
        path = tmp_path / (identity + "-source")
        assets.materialize_sources(identity, path, get=get)
        return path

    return prepare


def fake_physx_sdk(monkeypatch):
    """Explicit schema double: CPU USD serialization, no installed Physx claim."""
    import pxr

    def schema(name, attr, kind):
        class API:
            @classmethod
            def Apply(cls, prim):
                prim.AddAppliedSchema(name)
                return cls(prim)

            def __init__(self, prim):
                self.prim = prim

            def create(self, value):
                a = self.prim.CreateAttribute(attr, kind)
                a.Set(value)
                return a

        if name == "PhysxJointAPI":
            API.CreateMaxJointVelocityAttr = API.create
        else:
            API.CreateEnabledSelfCollisionsAttr = API.create
        return API

    stub = types.SimpleNamespace(
        PhysxJointAPI=schema(
            "PhysxJointAPI", "physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float
        ),
        PhysxArticulationAPI=schema(
            "PhysxArticulationAPI",
            "physxArticulation:enabledSelfCollisions",
            Sdf.ValueTypeNames.Bool,
        ),
    )
    monkeypatch.setattr(pxr, "PhysxSchema", stub, raising=False)
    monkeypatch.setitem(sys.modules, "pxr.PhysxSchema", stub)
    _sdk(monkeypatch)
    original = importer._converter

    def convert(*args, **kwargs):
        path, settings, identity = original(*args, **kwargs)
        stage = Usd.Stage.Open(str(path))
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())
        stage.GetRootLayer().Save()
        return path, settings, identity

    monkeypatch.setattr(importer, "_converter", convert)


def test_exact_source_bridge_configuration_and_fake_sdk_readback(
    public_sources, tmp_path, monkeypatch
):
    source = public_sources()
    before = {p: p.read_bytes() for p in (source / "source").rglob("*") if p.is_file()}
    config = tmp_path / "configuration"
    record = runtime.prepare_runtime_configuration(source, config)
    assert runtime.verify_runtime_configuration(config) == record
    report = record["simulation_model"]
    assert report["combined_mass_total_kg"] == pytest.approx(31.53086427304)
    assert len(report["links"]) == 17 and len(report["joints"]) == 13
    assert len(report["source_bridge"]["collision_spheres"]) == 46
    assert len(report["source_bridge"]["visuals"]) == 15
    assert len(report["source_bridge"]["source_contact_exclusions"]) == 7
    fields = record["manifest_fields"]
    assert len(fields["joints"]) == 7 and len(fields["grippers"][0]["joints"]) == 6
    assert [fields["reset_joint_positions"]["joint" + str(i)] for i in range(1, 8)] == [
        0,
        0.785398,
        0,
        -1.5708,
        0,
        0,
        0,
    ]
    assert all(
        j["max_velocity"] == 1 and j["stiffness"] == 2000 and j["damping"] == 200
        for j in fields["joints"]
    )
    assert [c["name"] for c in fields["cameras"]] == ["overhead", "arm_wrist"]
    assert (
        report["estimated_inertias_added"] is False and report["manufacturer_motor_limits"] is False
    )
    assert all(p.read_bytes() == data for p, data in before.items())
    fake_physx_sdk(monkeypatch)
    prepared = tmp_path / "prepared"
    imported = importer.import_source_asset(config, prepared, fix_base=True)
    assert importer.verify_prepared_asset_directory(prepared) == imported
    readback = imported["simulation_model_readback"]
    assert readback["enabled_self_collisions"] is True
    assert (
        len(readback["source_collision_spheres"]) == 46
        and len(readback["source_contact_exclusions"]) == 7
    )
    assert [row["velocity_limit_rad_s"] for row in readback["joints"][:7]] == pytest.approx([1] * 7)
    assert [row["usd_velocity_degrees_s"] for row in readback["joints"][:7]] == pytest.approx(
        [180 / np.pi] * 7
    )
    bound = runtime.bind_prepared_configuration(
        config, prepared, "https://models.example/development-only.usda"
    )
    assert (
        bound["robot_embodiment"]["embodiment_id"] == bridge.IDENTITY
        and bound["execution_authorized"] is False
    )

    from blueprint_pipeline.team_embodiment import team_embodiment_binding, apply_team_embodiment
    from tests.test_deployment_embodiment_runtime_configuration import station_plan

    plan = station_plan()
    plan["task_spec"]["start_pose_world"] = [0, 0, 0, 0, 0, 0, 1]
    plan["task_spec"]["destination_position_bounds_world_m"] = {
        "minimum": [0, 0, 0],
        "maximum": [0.1, 0.1, 0.1],
    }
    plan["robot"]["base_pose_world"]["position_world_m"] = [0, 0, 0]
    binding = team_embodiment_binding(
        bound["robot_embodiment"],
        model_record=bound["model_record"],
        structure=bound["structure"],
        scene_plan=plan,
    )
    native = apply_team_embodiment(plan, binding)
    assert native["robot"]["enabled_self_collisions"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "mass",
        "force",
        "velocity",
        "sphere",
        "exclusion",
        "self_collision",
        "extra_filter",
        "extra_collider",
        "units",
    ],
)
def test_imported_physics_cannot_silently_lose_source_or_declared_limits(
    public_sources, tmp_path, monkeypatch, fault
):
    source = public_sources()
    fake_physx_sdk(monkeypatch)
    folder = tmp_path / "prepared"
    imported = importer.import_source_asset(source, folder, fix_base=True)
    folder.chmod(0o755)
    path = folder / "robot.usda"
    path.chmod(0o644)
    stage = Usd.Stage.Open(str(path))
    if fault == "mass":
        UsdPhysics.MassAPI(stage.GetPrimAtPath("/robot/link2")).GetPrincipalAxesAttr().Set(
            __import__("pxr", fromlist=["Gf"]).Gf.Quatf(1, 0, 0, 0)
        )
    elif fault == "force":
        UsdPhysics.DriveAPI(stage.GetPrimAtPath("/robot/joint1"), "angular").GetTypeAttr().Set(
            "acceleration"
        )
    elif fault == "velocity":
        stage.GetPrimAtPath("/robot/joint1").GetAttribute("physxJoint:maxJointVelocity").Set(1.0)
    elif fault == "sphere":
        sphere = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Sphere))
        UsdGeom.Sphere(sphere).GetRadiusAttr().Set(0.5)
    elif fault == "exclusion":
        UsdPhysics.FilteredPairsAPI(
            stage.GetPrimAtPath("/robot/base")
        ).GetFilteredPairsRel().ClearTargets(True)
    elif fault == "extra_filter":
        sphere = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Sphere))
        UsdPhysics.FilteredPairsAPI.Apply(sphere).CreateFilteredPairsRel().SetTargets(
            ["/robot/link7"]
        )
    elif fault == "extra_collider":
        sphere = UsdGeom.Sphere.Define(stage, "/robot/link1/extra")
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    elif fault == "units":
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    else:
        stage.GetDefaultPrim().GetAttribute("physxArticulation:enabledSelfCollisions").Set(False)
    stage.GetRootLayer().Save()
    with pytest.raises(assets.AssetPreparationError, match="usd_"):
        model.usd_values(path, imported["simulation_model"])


@pytest.mark.parametrize(
    "q", [[0] * 7, [0, 0.785398, 0, -1.5708, 0, 0, 0], [0.15, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]]
)
def test_source_mujoco_fk_inertia_frames_and_spheres_survive_bridge(public_sources, tmp_path, q):
    mujoco = pytest.importorskip("mujoco")
    bundle = public_sources()
    configured = tmp_path / "configuration"
    report = runtime.prepare_runtime_configuration(bundle, configured)["simulation_model"]
    source = bundle / "source/arm/source" / bridge.SOURCE_ENTRY
    original = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(original)
    data.qpos[:] = q
    mujoco.mj_forward(original, data)
    converted = XML.parse(configured / "robot.urdf").getroot()
    poses = _poses(converted, dict(zip(["joint" + str(i) for i in range(1, 8)], q, strict=True)))
    for body in report["source_bridge"]["source_kinematics"]["bodies"]:
        name = body["name"]
        index = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_BODY, name)
        assert poses[name][:3, 3] == pytest.approx(data.xpos[index], abs=1e-12)
        assert poses[name][:3, :3] == pytest.approx(data.xmat[index].reshape(3, 3), abs=1e-12)
        row = next(row for row in report["links"] if row["link"] == name)
        frame = data.ximat[index].reshape(3, 3)
        tensor = frame @ np.diag(original.body_inertia[index]) @ frame.T
        generated = poses[name][:3, :3] @ row["inertia_body_frame_kg_m2"] @ poses[name][:3, :3].T
        assert generated == pytest.approx(tensor, abs=1e-12)
        assert (poses[name] @ [*row["center_of_mass_m"], 1])[:3] == pytest.approx(
            data.xipos[index], abs=1e-12
        )
    source_spheres = []
    for i in range(original.ngeom):
        if original.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
            source_spheres.append([*data.geom_xpos[i], original.geom_size[i, 0]])
    generated_spheres = [
        [*((poses[r["link"]] @ [*r["position_m"], 1])[:3]), r["radius_m"]]
        for r in report["source_bridge"]["collision_spheres"]
    ]
    assert np.array(generated_spheres) == pytest.approx(np.array(source_spheres), abs=1e-12)
    # Tool-parent transform retains the authored45mm site plus nominal134mm grasp.
    expected = np.diag([-1.0, -1.0, 1.0, 1.0])
    expected[2, 3] = 0.179
    assert np.linalg.inv(poses["link7"]) @ poses["grasp_link"] == pytest.approx(expected, abs=1e-12)
    assert not original.actuator_forcelimited.any()
    assert [row["effort_limit_nm"] for row in report["joints"][:7]] == pytest.approx(
        [316.1600224, 210.4772022, 136.1158312, 74.3139517, 37.6520140, 10.9337099, 5.8749933],
        abs=1e-6,
    )
    assert original.opt.gravity.tolist() == [0, 0, -9.81]
    assert report["native_fixed_base_gravity_disabled"] is True


def test_original30_profiles_36_recipes_111_rows_20_policy_identities_unchanged():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline import deployment_embodiment_runtime_compatibility as compatibility
    import importlib.util

    previous = with_reviewed_revisions(json.loads((FIXTURE / "predecessor_identities.json").read_text()))
    package = Path(assets.__file__).parent
    root = package.parents[1]
    recipes = {row["embodiment_id"]: row for row in assets.asset_recipes()["recipes"]}
    catalog = {
        row["candidate_id"]: row
        for row in json.loads((root / "config/beta_policy_candidates.v1.json").read_text())[
            "candidates"
        ]
    }
    assert (len(recipes), len(catalog), len(runtime._PROFILES)) == (49, 136, 42)
    for name, expected in previous["recipes"].items():
        assert canonical_digest(recipes[name]) == expected
    for name, expected in previous["catalog"].items():
        assert canonical_digest(catalog[name]) == expected
    for name, expected in previous["modules"].items():
        assert "sha256:" + hashlib.sha256((package / name).read_bytes()).hexdigest() == expected
    current = json.loads(
        (
            root / "config/policy_runtimes/scripted_robot_baseline_registration_profile.json"
        ).read_text()
    )
    for key, expected in previous["registration_profile"].items():
        assert (
            {b: current[key][b] for b in expected} if key == "bodies" else current[key]
        ) == expected
    assert len(current["bodies"]) == 24
    path = FIXTURE / "runtime_configuration_c596.py"
    assert (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        == compatibility.CURRENT_30_GENERATOR
    )
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._previous30", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert set(old._PROFILES) == compatibility.CURRENT_30_PROFILES
    assert all(runtime._PROFILES[key] == value for key, value in old._PROFILES.items())


def test_previous30_receipt_reuse_preserves_complete_metadata(
    public_sources, tmp_path, monkeypatch
):
    from blueprint_pipeline import deployment_embodiment_runtime_compatibility as compatibility
    from tests.test_deployment_embodiment_runtime_compatibility import _reseal
    import importlib.util

    path = FIXTURE / "runtime_configuration_c596.py"
    spec = importlib.util.spec_from_file_location("blueprint_pipeline._previous30_test", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    folder = tmp_path / "old"
    record = old.prepare_runtime_configuration(public_sources("kuka_lbr_iiwa_14"), folder)
    before = (folder / runtime.RECEIPT).read_bytes()
    monkeypatch.setattr(
        runtime, "_expand_xacro", lambda *a, **k: pytest.fail("No historical re-expansion")
    )
    assert runtime.verify_runtime_configuration(folder) == record
    assert (folder / runtime.RECEIPT).read_bytes() == before
    changed = deepcopy(record)
    changed["manifest_fields"]["cameras"][0]["position_m"][0] += 0.01
    _reseal(folder, changed)
    with pytest.raises(assets.AssetPreparationError, match="predecessor_metadata_changed"):
        runtime.verify_runtime_configuration(folder)
    assert not compatibility.is_exact_predecessor(
        {
            "embodiment_id": bridge.IDENTITY,
            "generator_module_sha256": compatibility.CURRENT_30_GENERATOR,
        }
    )
    assert not compatibility.is_exact_predecessor(
        {"embodiment_id": "kuka_lbr_iiwa_14", "generator_module_sha256": "sha256:" + "a" * 64}
    )


@pytest.mark.parametrize("fault", ["arm", "tool", "mount", "policy_claim"])
def test_changed_fixed_source_recipe_is_not_silently_composed(fault):
    from blueprint_pipeline.deployment_embodiment_kuka_composition import component_recipe

    recipe = deepcopy(assets.recipe_for(bridge.IDENTITY))
    if fault == "arm":
        recipe["source"]["revision"] = "a" * 40
    elif fault == "tool":
        recipe["composition"]["tool_source"]["revision"] = "a" * 40
    elif fault == "mount":
        recipe["composition"]["mount"]["translation_m"][2] = 0.011
    else:
        recipe["composition"]["policy_compatibility_established"] = True
    with pytest.raises(assets.AssetPreparationError):
        component_recipe(recipe)
