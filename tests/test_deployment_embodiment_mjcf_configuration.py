"""Development-only MJCF source binding; converter output is a CPU USD fixture."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as XML

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest


def fixture_xml(*, angle="radian"):
    bodies = '<body name="base" childclass="iiwa"><inertial mass="5" pos="0 0 .1" diaginertia=".05 .06 .03"/>'
    for n in range(1, 8):
        joint_class = "joint3" if n == 7 else "joint2" if n % 2 == 0 else "joint1"
        bodies += (f'<body name="link{n}" pos="0 0 .1" quat="0 0 1 1">'
                   f'<inertial mass="{n}" pos="0 0 .02" diaginertia=".01 .02 .03" quat="1 1 0 0"/>'
                   f'<joint name="joint{n}" class="{joint_class}"/>')
    bodies += '<site name="attachment_site" pos="0 0 .045"/>' + '</body>' * 8
    actuators = ''.join(f'<general name="actuator{n}" joint="joint{n}" class="joint3"/>' for n in range(1, 8))
    return (f'<mujoco model="development_only"><compiler angle="{angle}" autolimits="true"/>'
            '<default><default class="iiwa"><joint axis="0 0 2"/>'
            '<general gaintype="fixed" biastype="affine" gainprm="2000" biasprm="0 -2000 -200"/>'
            '<default class="joint1"><joint range="-3 3"/>'
            '<default class="joint2"><joint range="-2 2"/></default></default>'
            '<default class="joint3"><joint range="-3.1 3.1"/></default></default></default>'
            f'<worldbody>{bodies}</worldbody><actuator>{actuators}</actuator>'
            '<keyframe><key name="home" qpos="0 .01 0 -.02 0 0 0"/></keyframe></mujoco>').encode()


@pytest.fixture
def source_case(tmp_path):
    recipe = assets.recipe_for("kuka_lbr_iiwa_14")

    def create(content=None):
        # Exercise real source retention with an injected development-only
        # transport, never with a live request or a production import claim.
        files = {"kuka_iiwa_14/iiwa14.xml": content or fixture_xml(),
                 "kuka_iiwa_14/LICENSE": b"Development-only fixture; not licensed model evidence."}
        tree = {"truncated": False, "tree": [{"path": name, "type": "blob", "mode": "100644", "size": len(data),
                "sha": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
                for name, data in files.items()]}

        def get(url, *, maximum):
            return json.dumps(tree).encode() if "/git/trees/" in url else files[url.split(recipe["source"]["revision"] + "/")[1]]

        bundle = tmp_path / "bundle"
        assets.materialize_sources("kuka_lbr_iiwa_14", bundle, get=get)
        return bundle
    return create


def test_nested_defaults_native_axes_authored_dynamics_and_home_remain_source_facts(source_case, tmp_path):
    bundle = source_case()
    output = tmp_path / "configuration"
    result = runtime.prepare_runtime_configuration(bundle, output)
    fields, source = result["manifest_fields"], result["source_kinematics"]
    assert runtime.verify_runtime_configuration(output) == result
    assert runtime.prepare_runtime_configuration(bundle, output) == result
    assert not (output / "robot.urdf").exists() and "source_urdf_sha256" not in result
    assert result["source_model"]["format"] == "mjcf"
    assert (output / result["source_model"]["path"]).read_bytes() == fixture_xml()
    assert [joint["upper"] for joint in fields["joints"]] == [3, 2, 3, 2, 3, 2, 3.1]
    assert all(j["max_velocity"] == 1 and j["stiffness"] == 2000 and j["damping"] == 200 for j in fields["joints"])
    assert all(j["axis"] == [0, 0, 1] for j in source["joints"])
    assert source["bodies"][1]["authored_quaternion_wxyz"] == [0, 0, 1, 1]
    assert source["bodies"][1]["orientation_wxyz"] == pytest.approx([0, 0, math.sqrt(.5), math.sqrt(.5)])
    assert source["sites"][0]["position_m"] == [0, 0, .045]
    assert fields["reset_joint_positions"]["joint2"] == .01
    assert source["reset"]["kind"] == "source_keyframe"
    assert result["source_dynamics"]["moving_links_without_positive_authored_mass_and_inertia"] == []
    assert result["source_dynamics"]["estimated_inertias_added"] is False
    assert fields["grippers"] == [] and "arms" not in fields and fields["tool_link"] == "link7"
    assert "bare_arm_without_application_tool_or_grasp_capability" in result["simulation_assumptions"]
    assert result["policy_compatibility_established"] is result["execution_authorized"] is False
    assert [camera["mount_link"] for camera in fields["cameras"]] == [None, "link7"]
    validate_embodiment_manifest({**fields, "model": {"format": "usd", "url": "https://fixture.example/development.usda",
                                                     "sha256": "sha256:" + "a" * 64}})


def test_compiler_degree_converts_hinges_but_not_native_keyframe_qpos(source_case, tmp_path):
    result = runtime.prepare_runtime_configuration(source_case(fixture_xml(angle="degree")), tmp_path / "configuration")
    assert result["manifest_fields"]["joints"][0]["upper"] == pytest.approx(math.radians(3))
    assert result["manifest_fields"]["reset_joint_positions"]["joint2"] == .01


@pytest.mark.parametrize("defect", ["source", "model_path", "metadata_module", "extra_generated"])
def test_bound_mjcf_receipts_detect_source_metadata_and_inventory_tampering(source_case, tmp_path, defect):
    output = tmp_path / "configuration"
    result = runtime.prepare_runtime_configuration(source_case(), output)
    if defect == "source":
        (output / result["source_model"]["path"]).write_bytes(b"changed")
    elif defect == "extra_generated":
        output.chmod(0o755)  # Deliberate corruption of an immutable test publication.
        (output / "unexpected.xml").write_bytes(b"changed")
    else:
        if defect == "model_path":
            result["source_model"]["path"] = "../unbound.xml"
        else:
            result["metadata_generator_module_sha256"] = "sha256:" + "0" * 64
        result["configuration_digest"] = cross_runtime_canonical_digest(result, digest_field="configuration_digest")
        (output / runtime.RECEIPT).write_text(json.dumps(result))
    with pytest.raises(assets.AssetPreparationError):
        runtime.verify_runtime_configuration(output)


def test_missing_authored_inertia_and_reset_are_explicit_without_fabricated_physics(source_case, tmp_path):
    root = XML.fromstring(fixture_xml())
    body = root.find(".//body[@name='link3']")
    body.remove(body.find("inertial"))
    root.remove(root.find("keyframe"))
    result = runtime.prepare_runtime_configuration(source_case(XML.tostring(root)), tmp_path / "configuration")
    assert result["source_dynamics"]["moving_links_without_positive_authored_mass_and_inertia"] == ["link3"]
    assert result["source_dynamics"]["estimated_inertias_added"] is False
    assert result["source_kinematics"]["reset"]["kind"] == "simulation_zero_clamped_to_source_limits"
    assert "simulation_zero_reset_clamped_to_source_limits" in result["simulation_assumptions"]


def test_original_mjcf_is_converter_input_and_real_usd_readback_binds_joint_only_manifest(source_case, tmp_path, monkeypatch):
    output = tmp_path / "configuration"
    record = runtime.prepare_runtime_configuration(source_case(), output)
    seen = []

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class MjcfFixtureConverter:
        def __init__(self, config):
            from pxr import Usd, UsdPhysics
            seen.append(config)
            assert Path(config.asset_path).read_bytes() == fixture_xml()
            path = Path(config.usd_dir) / "development_only.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            stage.SetDefaultPrim(stage.DefinePrim("/robot", "Xform"))
            for body in record["source_kinematics"]["bodies"]:
                UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim("/robot/" + body["name"], "Xform"))
            for joint in record["manifest_fields"]["joints"]:
                UsdPhysics.RevoluteJoint.Define(stage, "/robot/" + joint["name"])
            UsdPhysics.FixedJoint.Define(stage, "/robot/anchor").CreateBody1Rel().SetTargets(["/robot/base"])
            stage.GetRootLayer().Save()
            self.usd_path = str(path)

    sdk = SimpleNamespace(MjcfConverterCfg=Config, MjcfConverter=MjcfFixtureConverter)
    real_import = importer.importlib.import_module
    monkeypatch.setattr(importer.importlib, "import_module", lambda name, *args:
        sdk if name == "isaaclab.sim.converters" else real_import(name, *args))
    prepared = tmp_path / "prepared"
    imported = importer.import_source_asset(output, prepared, fix_base=True)
    bound = runtime.bind_prepared_configuration(output, prepared, "https://fixture.example/development.usda")
    assert len(seen) == 1 and seen[0].import_physics_scene is False and seen[0].override_gain_prm is None
    assert bound["robot_embodiment"]["model"]["sha256"] == imported["model"]["sha256"]
    assert bound["robot_embodiment"]["grippers"] == [] and "arms" not in bound["robot_embodiment"]
    assert bound["claim_ceiling"] == "configured_not_executed" and bound["execution_authorized"] is False
