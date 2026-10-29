"""Development-only source-to-runtime binding tests; no GPU or policy calls."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as XML

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from blueprint_pipeline import deployment_embodiment_station as station
from blueprint_pipeline.team_embodiment import apply_team_embodiment, team_embodiment_binding
from tests.test_deployment_embodiment_asset_import import _source
from tests.test_team_embodiment import station_plan


SOURCE = b'''<robot name="development_only">
<link name="base"/><link name="tool"/><link name="tcp"/>
<link name="finger-a"/><link name="finger-b"/>
<joint name="joint" type="revolute"><parent link="base"/><child link="tool"/>
  <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
  <limit lower="-1.0" upper="1.0" velocity="2" effort="10"/></joint>
<joint name="tcp_mount" type="fixed"><parent link="tool"/><child link="tcp"/>
  <origin xyz="0 0 0.1"/></joint>
<joint name="finger" type="prismatic"><parent link="tool"/><child link="finger-a"/>
  <axis xyz="0 1 0"/><limit lower="0" upper="0.04" velocity="0.1" effort="10"/></joint>
<joint name="slave" type="prismatic"><parent link="tool"/><child link="finger-b"/>
  <axis xyz="0 1 0"/><limit lower="-0.04" upper="0" velocity="0.1" effort="10"/>
  <mimic joint="finger" multiplier="-1" offset="0"/></joint></robot>'''


@pytest.fixture
def configured(tmp_path, monkeypatch):
    def prepare(content=SOURCE):
        source = _source(tmp_path, monkeypatch, content=content)
        recipe = assets.recipe_for("fixture_dual_option")
        monkeypatch.setattr(runtime, "recipe_for", lambda identity: recipe)
        monkeypatch.setattr(station, "recipe_for", lambda identity: recipe)
        monkeypatch.setattr(runtime, "_PROFILES", {"fixture_dual_option": [runtime._arm(
            "base", "tool", 1, tcp="tcp", fingers={"finger": [.04, 0.]},
            contacts=["finger-a", "finger-b"], width=.08)]})
        output = tmp_path / "configured"
        record = runtime.prepare_runtime_configuration(source, output)
        return SimpleNamespace(source=source, output=output, record=record, recipe=recipe)
    return prepare


def test_two_real_source_instances_preserve_independent_tools_mimic_limits_and_tcp(configured):
    case = configured()
    record, fields = case.record, case.record["manifest_fields"]
    assert record["claim_ceiling"] == "configuration_only" and record["execution_authorized"] is False
    assert record["policy_compatibility_established"] is False and "model" not in fields
    assert record["source_dynamics"]["estimated_inertias_added"] is False
    assert record["source_dynamics"]["moving_links_without_positive_authored_mass_and_inertia"] == [
        "left_finger-a", "left_finger-b", "left_tool", "right_finger-a", "right_finger-b", "right_tool"]
    assert any(item.startswith("missing_positive_authored_moving_link_inertias:") for item in record["source_limitations"])
    assert [j["name"] for j in fields["joints"]] == ["left_joint", "right_joint"]
    assert all(j["max_velocity"] == 1.0 and j["lower"] == -1 for j in fields["joints"])
    for side, arm, gripper in zip(("left", "right"), fields["arms"], fields["grippers"], strict=True):
        assert arm["name"] == side and arm["tool_from_tcp"]["position_m"] == pytest.approx([0, 0, .1])
        assert gripper["contact_links"] == [side + "_finger_a", side + "_finger_b"]
        assert gripper["joints"] == [{"name": side + "_finger", "open": .04, "closed": 0.},
                                     {"name": side + "_slave", "open": -.04, "closed": 0.}]
    source = XML.parse(case.output / "robot.urdf").getroot()
    assert source.find("joint[@name='left_slave']/mimic").get("joint") == "left_finger"
    assert source.find("joint[@name='right_slave']/mimic").get("joint") == "right_finger"
    assert runtime.prepare_runtime_configuration(case.source, case.output) == record
    assert runtime.verify_runtime_configuration(case.output) == record


@pytest.mark.parametrize("defect", ["robot", "extra_generated", "extra_named_receipt", "source"])
def test_changed_source_and_generated_files_never_reuse_a_configuration(configured, defect):
    case = configured()
    target = {"robot": "robot.urdf", "source": "source/robot.urdf",
              "extra_generated": "expanded_source/orphan", "extra_named_receipt": "expanded_source/source-bundle.json"}[defect]
    case.output.chmod(0o755)  # The test deliberately tampers with its immutable output.
    path = case.output / target
    path.parent.mkdir(exist_ok=True)
    path.write_text("changed")
    with pytest.raises(assets.AssetPreparationError):
        runtime.prepare_runtime_configuration(case.source, case.output)


@pytest.mark.parametrize("defect,reason", [
    ("uncontrolled", "uncontrolled_source_joint"), ("mimic", "finger_limits_mismatch"),
    ("missing_limit", "joint_limits_missing"), ("moving_tcp", "tcp_frame_not_rigid")])
def test_source_controller_mismatches_report_precise_failure_without_output(configured, defect, reason):
    root = XML.fromstring(SOURCE)
    if defect == "uncontrolled":
        XML.SubElement(root, "link", name="uncontrolled")
        joint = XML.SubElement(root, "joint", name="uncontrolled", type="revolute")
        XML.SubElement(joint, "parent", link="base")
        XML.SubElement(joint, "child", link="uncontrolled")
    elif defect == "mimic":
        root.find("joint[@name='slave']/mimic").set("multiplier", "2")
    elif defect == "missing_limit":
        joint = root.find("joint[@name='joint']")
        joint.remove(joint.find("limit"))
    else:
        root.find("joint[@name='tcp_mount']").set("type", "revolute")
    with pytest.raises(assets.AssetPreparationError, match=reason):
        configured(XML.tostring(root))


def _fake_sdk_from_urdf(monkeypatch):
    """Real USD structure from fixture URDF, never an actual simulator import."""
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Converter:
        def __init__(self, config):
            from pxr import Usd, UsdPhysics
            root = XML.parse(config.asset_path).getroot()
            path = Path(config.usd_dir) / "development_only.usda"
            path.parent.mkdir(parents=True, exist_ok=True)
            usd = Usd.Stage.CreateNew(str(path))
            usd.SetDefaultPrim(usd.DefinePrim("/robot", "Xform"))
            for link in root.findall("link"):
                UsdPhysics.RigidBodyAPI.Apply(usd.DefinePrim("/robot/" + runtime._name(link.get("name")), "Xform"))
            for joint in root.findall("joint"):
                if joint.get("type") != "fixed":
                    UsdPhysics.RevoluteJoint.Define(usd, "/robot/" + runtime._name(joint.get("name")))
            UsdPhysics.FixedJoint.Define(usd, "/robot/anchor").CreateBody1Rel().SetTargets(["/robot/blueprint_station"])
            usd.GetRootLayer().Save()
            self.usd_path = str(path)
    sdk = SimpleNamespace(UrdfConverterCfg=Config, UrdfConverter=Converter)
    real_import = importer.importlib.import_module
    monkeypatch.setattr(importer.importlib, "import_module", lambda name, *args:
                        sdk if name == "isaaclab.sim.converters" else real_import(name, *args))


def test_actual_import_digest_and_structure_bind_into_generic_plan_without_policy_claim(configured, monkeypatch, tmp_path):
    case = configured()
    _fake_sdk_from_urdf(monkeypatch)
    imported = tmp_path / "imported"
    receipt = importer.import_source_asset(case.output, imported, fix_base=True,
                                            station_parameters=case.record["station_parameters"])
    inputs = runtime.bind_prepared_configuration(case.output, imported, "https://assets.example/fixture.usda",
                                                 runtime_settings={"control_hz": 30, "action_chunk_rows": 8})
    assert inputs["robot_embodiment"]["model"]["sha256"] == receipt["model"]["sha256"]
    assert inputs["runtime_settings"] == {"control_hz": 30, "action_chunk_rows": 8}
    assert inputs["claim_ceiling"] == "configured_not_executed" and inputs["execution_authorized"] is False
    plan = station_plan()
    plan["task_spec"]["start_pose_world"] = [0, 0, 0, 0, 0, 0, 1]
    plan["task_spec"]["destination_position_bounds_world_m"] = {"minimum": [0, 0, 0], "maximum": [.1, .1, .1]}
    plan["robot"]["base_pose_world"]["position_world_m"] = [0, 0, 0]
    binding = team_embodiment_binding(inputs["robot_embodiment"], model_record=inputs["model_record"],
                                      structure=inputs["structure"], scene_plan=plan)
    bound = apply_team_embodiment(plan, binding)
    assert bound["robot"]["control_hz"] == 30
    assert [c["name"] for c in bound["robot"]["gripper_channels"]] == ["left_gripper", "right_gripper"]
    with pytest.raises(ValueError, match="control_hz"):
        runtime.bind_prepared_configuration(case.output, imported, "https://assets.example/fixture.usda",
                                            runtime_settings={"control_hz": 7})
    (imported / "robot.usda").write_text("changed actual model")
    with pytest.raises(assets.AssetPreparationError, match="output_changed"):
        runtime.bind_prepared_configuration(case.output, imported, "https://assets.example/fixture.usda")


def test_batch_preserves_each_individual_success_failure_unavailable_and_existing_result(configured, monkeypatch, tmp_path):
    case = configured()
    bad = deepcopy(case.recipe)
    bad["embodiment_id"] = "missing_source"
    profiles = deepcopy(runtime._PROFILES)
    profiles["missing_source"] = profiles["fixture_dual_option"]
    monkeypatch.setattr(runtime, "_PROFILES", profiles)
    monkeypatch.setattr(runtime, "asset_recipes", lambda: {"recipes": [case.recipe, bad,
        {"embodiment_id": "unknown", "source": None, "status": "source_unavailable", "reason": "missing license"},
        {"embodiment_id": "existing", "source": None, "status": "existing_runtime", "reason": "retain existing"}]})
    root = tmp_path / "catalog"
    root.mkdir()
    case.source.chmod(0o755)
    case.source.rename(root / "fixture_dual_option")
    result = runtime.prepare_available_runtime_configurations(root, tmp_path / "batch")
    assert [row["status"] for row in result["results"]] == ["source_configured", "configuration_failed",
                                                             "configuration_unavailable", "existing_runtime"]
    assert result["results"][2]["reason"] == "missing license"


@pytest.mark.parametrize("url", ["http://assets.example/a.usd", "https://user:secret@assets.example/a.usd",
                                "https://assets.example/a.usd?token=secret"])
def test_import_cli_rejects_unpublishable_url_before_configuration_or_isaac(monkeypatch, tmp_path, url):
    monkeypatch.setattr(runtime, "prepare_runtime_configuration", lambda *_: pytest.fail("unexpected preparation"))
    with pytest.raises(SystemExit):
        runtime.main(["--source-bundle", str(tmp_path / "source"), "--output", str(tmp_path / "output"),
                      "--import-prepared", str(tmp_path / "imported"), "--model-url", url, "--initialize-isaac"])
