"""Explicit simulation fixtures: source kinematics are composed, never invented."""

from copy import deepcopy
import json
import math
import xml.etree.ElementTree as ET

import pytest

from blueprint_pipeline import deployment_embodiment_station as station
from blueprint_pipeline.deployment_embodiment_assets import AssetPreparationError


def _pose(xyz, q=None):
    return {"translation_m": xyz, "quaternion_wxyz": q or [1, 0, 0, 0]}


def _parameters():
    return {"schema_version": station.PARAMETERS_SCHEMA, "embodiment_id": "fixture_dual_option",
            "composition_mode": "two_instances", "station_root_link": "fixture_station",
            "arms": [{"name": name, "prefix": prefix, "source_base_link": "base",
                      "source_mount_joint": None, "station_from_base": _pose(xyz)}
                     for name, prefix, xyz in (("left", "L_", [0.1, 0.2, 0.3]),
                                               ("right", "R_", [0.4, -0.2, 0.3]))]}


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "tool.stl").write_bytes(b"development fixture mesh; not a qualified robot")
    path = source / "arm.urdf"
    path.write_text('''<robot name="fixture">
      <material name="gray"><color rgba="0.3 0.3 0.3 1"/></material>
      <link name="base"><inertial><mass value="2"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <link name="tool"><visual><geometry><mesh filename="tool.stl"/></geometry><material name="gray"/></visual><collision><origin xyz="0 0 0.1"/><geometry><box size=".1 .2 .3"/></geometry></collision></link>
      <link name="finger1"/><link name="finger2"/>
      <joint name="arm_joint" type="revolute"><parent link="base"/><child link="tool"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="5" velocity="2"/></joint>
      <joint name="finger_joint" type="prismatic"><parent link="tool"/><child link="finger1"/><axis xyz="0 1 0"/><limit lower="0" upper=".04" effort="2" velocity=".1"/></joint>
      <joint name="finger_mimic" type="prismatic"><parent link="tool"/><child link="finger2"/><axis xyz="0 -1 0"/><limit lower="0" upper=".04" effort="2" velocity=".1"/><mimic joint="finger_joint" multiplier="1" offset="0"/></joint>
      <transmission name="transmission"><joint name="arm_joint"/><actuator name="motor"/></transmission>
      <gazebo reference="tool"><material>fixture</material></gazebo>
    </robot>''')
    return path


def _compose(path, parameters):
    root = path.parent.parent
    return station.gen_urdf(path, parameters, output_anchor=root, packages={},
                            allowed_files={p.resolve() for p in root.rglob("*") if p.is_file()})


def test_two_instances_preserve_geometry_inertials_and_all_named_references(tmp_path):
    source = _source(tmp_path)
    before = source.read_bytes()
    parameters = _parameters()
    parameters["arms"][1]["station_from_base"] = _pose([.4, -.2, .3], [math.sqrt(.5), 0, 0, math.sqrt(.5)])
    text, receipt = _compose(source, parameters)
    root = ET.fromstring(text)
    assert source.read_bytes() == before
    assert station._tree(root)[0] == "fixture_station"
    assert len(root.findall("joint")) == 8
    for prefix in ("L_", "R_"):
        assert root.find(f"joint[@name='{prefix}finger_mimic']/mimic").get("joint") == prefix + "finger_joint"
        assert root.find(f"transmission[@name='{prefix}transmission']/joint").get("name") == prefix + "arm_joint"
        assert root.find(f"transmission[@name='{prefix}transmission']/actuator").get("name") == prefix + "motor"
        assert root.find(f"gazebo[@reference='{prefix}tool']") is not None
        assert root.find(f"link[@name='{prefix}tool']/visual/material").get("name") == prefix + "gray"
        assert root.find(f"link[@name='{prefix}tool']/visual/geometry/mesh").get("filename") == "source/tool.stl"
        original = ET.fromstring(before)
        assert ET.tostring(root.find(f"link[@name='{prefix}base']/inertial")) == ET.tostring(original.find("link[@name='base']/inertial"))
        assert ET.tostring(root.find(f"link[@name='{prefix}tool']/collision")) == ET.tostring(original.find("link[@name='tool']/collision"))
    rotation = [float(x) for x in root.find("joint[@name='right_station_mount']/origin").get("rpy").split()]
    assert rotation == pytest.approx([0, 0, math.pi/2])
    assert receipt["manufacturer_geometry_changed"] is False
    assert receipt["mount_parameter_authority"] == "sim_authored"
    assert receipt["parameters"] == parameters


def test_remount_publisher_pair_changes_only_explicit_fixed_mounts(tmp_path):
    path = tmp_path / "pair.urdf"
    path.write_text('''<robot name="fixture_pair"><link name="left_base"/><link name="left_tool"/>
      <link name="right_base"/><link name="right_tool"/><link name="camera"/>
      <joint name="left_joint" type="revolute"><parent link="left_base"/><child link="left_tool"/></joint>
      <joint name="right_base_joint" type="fixed"><parent link="left_base"/><child link="right_base"/><origin xyz="0 -.6 0" rpy="0 0 0"/></joint>
      <joint name="right_joint" type="revolute"><parent link="right_base"/><child link="right_tool"/></joint>
      <joint name="camera_joint" type="fixed"><parent link="left_base"/><child link="camera"/><origin xyz=".2 0 1"/></joint></robot>''')
    p = {"schema_version": station.PARAMETERS_SCHEMA, "embodiment_id": "fixture_dual_option",
         "composition_mode": "remount_existing_pair", "station_root_link": "station",
         "station_from_source_root": _pose([.1, .2, .3]), "arms": [
             {"name": "left", "prefix": None, "source_base_link": "left_base", "source_mount_joint": None,
              "station_from_base": _pose([.1, .2, .3])},
             {"name": "right", "prefix": None, "source_base_link": "right_base", "source_mount_joint": "right_base_joint",
              "station_from_base": _pose([.6, -.4, .3])}]}
    text, _ = station.gen_urdf(path, p, output_anchor=tmp_path, packages={}, allowed_files={path})
    root = ET.fromstring(text)
    assert set(j.get("name") for j in root.findall("joint")) == {
        "left_joint", "right_joint", "right_base_joint", "camera_joint", "source_root_station_mount"}
    assert [float(x) for x in root.find("joint[@name='right_base_joint']/origin").get("xyz").split()] == pytest.approx([.5, -.6, 0])
    assert root.find("joint[@name='camera_joint']/origin").get("xyz") == ".2 0 1"
    assert station._tree(root)[0] == "station"
    bad = deepcopy(p)
    bad["arms"][1]["source_mount_joint"] = "right_joint"
    with pytest.raises(AssetPreparationError, match="actual_fixed_mount_required"):
        station.gen_urdf(path, bad, output_anchor=tmp_path, packages={}, allowed_files={path})


@pytest.mark.parametrize("defect", ["missing_mount", "duplicate_prefix", "missing_root", "not_unit", "nan", "joint_collision", "extra_key"])
def test_no_mount_defaults_or_silent_name_inference(tmp_path, defect):
    path = _source(tmp_path)
    p = _parameters()
    if defect == "missing_mount":
        del p["arms"][0]["station_from_base"]
    elif defect == "duplicate_prefix":
        p["arms"][1]["prefix"] = "L_"
    elif defect == "missing_root":
        p["arms"][0]["source_base_link"] = "guessed_base"
    elif defect == "not_unit":
        p["arms"][0]["station_from_base"]["quaternion_wxyz"] = [1, 1, 0, 0]
    elif defect == "nan":
        p["arms"][0]["station_from_base"]["translation_m"][0] = float("nan")
    elif defect == "joint_collision":
        p["station_root_link"] = "L_base"
    else:
        p["invent_gripper"] = True
    with pytest.raises(AssetPreparationError, match="station_composition_"):
        _compose(path, p)


def test_station_source_cli_uses_exact_parameters_and_does_not_boot_isaac(tmp_path, monkeypatch, capsys):
    calls = []
    params = tmp_path / "params.json"
    params.write_text(json.dumps(_parameters()))
    monkeypatch.setattr(station, "prepare_station_source", lambda source, output, p: calls.append((source, output, p)) or {"execution_authorized": False})
    assert station.main(["--source-bundle", str(tmp_path / "bundle"), "--parameters", str(params),
                         "--output", str(tmp_path / "new-station")]) == 0
    assert len(calls) == 1 and calls[0][2] == _parameters()
    assert json.loads(capsys.readouterr().out)["execution_authorized"] is False
