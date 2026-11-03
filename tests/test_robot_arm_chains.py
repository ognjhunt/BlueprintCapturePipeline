"""Each catalog arm's kinematic chain, read from its pinned public description: the reader, the shipped
catalog and the solver's view of it.

The shipped chains are read from the pipeline's own prepared URDFs (``tests/fixtures/body_kinematics``,
regenerated from the recipes' pinned sources), so every test here is hermetic. ``robot_kinematics`` reads
the same URDFs independently and is the reference.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import team_body_kinematics as kinematics
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.robot_arm_chains import (
    CATALOG_PATH,
    RobotArmChainError,
    arm_chain,
    load_robot_arm_chains,
    main,
    read_urdf_arm,
    refresh_robot_arm_chains,
    validate_robot_arm_chains,
)
from blueprint_pipeline.robot_kinematics import _axis_rotation, _parse_urdf
from blueprint_pipeline.robot_spec_catalog import RECIPE_CATALOG_PATH, load_robot_spec_catalog

FIXTURES = Path(__file__).parent / "fixtures" / "body_kinematics"
TOOL_LENGTH_M = 0.1628

TOY = """<robot name="toy">
  <link name="world"/><link name="base"/><link name="upper"/><link name="lower"/><link name="wrist"/>
  <link name="flange"/><link name="camera"/>
  <joint name="mount" type="fixed"><parent link="world"/><child link="base"/><origin xyz="1 0 0"/></joint>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.3" rpy="0 0 0.5"/><axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="elbow" type="continuous"><parent link="upper"/><child link="lower"/>
    <origin xyz="0.4 0 0"/><axis xyz="0 1 0"/></joint>
  <joint name="slide" type="prismatic"><parent link="lower"/><child link="wrist"/>
    <limit lower="0" upper="0.1" effort="1" velocity="1"/></joint>
  <joint name="tip" type="fixed"><parent link="wrist"/><child link="flange"/>
    <origin xyz="0.2 0 0" rpy="0 1.5707963267948966 0"/></joint>
  <joint name="cam" type="fixed"><parent link="upper"/><child link="camera"/><origin xyz="0 0.1 0"/></joint>
</robot>
"""


def _write(tmp_path, text=TOY):
    path = tmp_path / "toy.urdf"
    path.write_text(text)
    return path


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _shipped() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _recipes() -> dict:
    return {row["embodiment_id"]: row for row in json.loads(RECIPE_CATALOG_PATH.read_text())["recipes"]}


# --- the reader ---------------------------------------------------------------------------------------

def test_the_reader_walks_the_one_route_from_the_base_to_the_flange(tmp_path):
    joints = read_urdf_arm(_write(tmp_path), base_link="base", flange_link="flange")
    # The mount above the base and the camera branch are not on the route; URDF defaults fill the rest.
    assert joints == [
        {"name": "shoulder", "type": "revolute", "parent": "base", "child": "upper", "xyz": [0.0, 0.0, 0.3],
         "rpy": [0.0, 0.0, 0.5], "axis": [0.0, 0.0, 1.0], "lower": -2.0, "upper": 2.0},
        {"name": "elbow", "type": "continuous", "parent": "upper", "child": "lower", "xyz": [0.4, 0.0, 0.0],
         "rpy": [0.0, 0.0, 0.0], "axis": [0.0, 1.0, 0.0]},
        {"name": "slide", "type": "prismatic", "parent": "lower", "child": "wrist", "xyz": [0.0, 0.0, 0.0],
         "rpy": [0.0, 0.0, 0.0], "axis": [1.0, 0.0, 0.0], "lower": 0.0, "upper": 0.1},
        {"name": "tip", "type": "fixed", "parent": "wrist", "child": "flange", "xyz": [0.2, 0.0, 0.0],
         "rpy": [0.0, 1.5707963267948966, 0.0]},
    ]


@pytest.mark.parametrize("old, new, links, code", [
    ('<axis xyz="0 1 0"/></joint>', '<axis xyz="0 1 0"/><mimic joint="shoulder"/></joint>', None,
     "robot_arm_urdf_mimic_joint:elbow"),
    ('name="elbow" type="continuous"', 'name="elbow" type="floating"', None,
     "robot_arm_urdf_joint_type_unsupported:elbow:floating"),
    ('<limit lower="-2" upper="2" effort="1" velocity="1"/>', "", None, "robot_arm_urdf_limit_invalid:shoulder"),
    ('lower="-2" upper="2"', 'lower="2" upper="-2"', None, "robot_arm_urdf_limit_invalid:shoulder"),
    ('<axis xyz="0 1 0"/>', '<axis xyz="0 0 0"/>', None, "robot_arm_urdf_axis_invalid:elbow"),
    ('xyz="0 0 0.3"', 'xyz="0 0 nan"', None, "robot_arm_urdf_origin_invalid:shoulder"),
    ('xyz="0.4 0 0"', 'xyz="0.4 0"', None, "robot_arm_urdf_origin_invalid:elbow"),
    ("", "", ("flange", "base"), "robot_arm_urdf_flange_not_below_base:base"),
    ("", "", ("base", "hand"), "robot_arm_urdf_link_missing:hand"),
    ('<joint name="cam"', '<joint name="twin" type="fixed"><parent link="base"/><child link="lower"/></joint>'
     '<joint name="cam"', None, "robot_arm_urdf_link_has_two_parents:lower"),
    ('<robot name="toy">', '<sdf name="toy">', None, "robot_arm_urdf_invalid"),
    ("</robot>", "", None, "robot_arm_urdf_invalid"),
    ("", "", ("wrist", "flange"), "robot_arm_urdf_no_movable_joint"),
])
def test_the_reader_refuses_what_it_cannot_model(tmp_path, old, new, links, code):
    assert TOY.count(old) == 1 or not old
    text = TOY.replace(old, new) if old else TOY
    if old == '<robot name="toy">':
        text = text.replace("</robot>", "</sdf>")
    base, flange = links or ("base", "flange")
    with pytest.raises(RobotArmChainError) as caught:
        read_urdf_arm(_write(tmp_path, text), base_link=base, flange_link=flange)
    assert caught.value.errors == (code,)


# --- the shipped catalog --------------------------------------------------------------------------------

def test_the_shipped_chains_are_read_from_the_pinned_prepared_urdfs():
    shipped = _shipped()
    # Reading every arm again from its URDF reproduces the shipped file exactly.
    assert refresh_robot_arm_chains(shipped, urdf_root=FIXTURES) == shipped
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    recipes = _recipes()
    for arm in shipped["arms"]:
        body = arm["read_from"]["body"]
        assert arm["read_from"]["urdf_sha256"] == provenance["bodies"][body]["robot_urdf_sha256"] == _sha256(
            FIXTURES / body / "robot.urdf")
        source = recipes[arm["recipe_id"]]["source"]
        assert arm["source"] == {key: source[key] for key in ("repository_url", "revision", "entry_path", "license_id")}
        # The prepared body was built from that same arm source.
        assert {(row["repository_url"], row["revision"]) for row in provenance["bodies"][body]["sources"]} >= {
            (source["repository_url"], source["revision"])}


def test_every_fixed_arm_with_a_pinned_source_has_a_chain():
    recipes = _recipes()
    expected = {model["recipe_id"] for model in load_robot_spec_catalog()["models"]
                if model["body_type"] == "fixed_arm" and model.get("recipe_id")
                and recipes[model["recipe_id"]]["status"] == "source_available"}
    assert {arm["recipe_id"] for arm in load_robot_arm_chains()["arms"]} == expected
    assert len(expected) == 13


def _reference_frames(urdf: Path, values: dict) -> dict:
    """Every link's pose from ``robot_kinematics``' own URDF reading."""
    model = _parse_urdf(urdf)
    poses = {model["base_frame"]: np.eye(4)}
    for joint in model["joints"]:
        motion = np.eye(4)
        value = float(values.get(joint["name"], 0.0))
        if joint["type"] in {"revolute", "continuous"}:
            motion = _axis_rotation(joint["axis"], value)
        elif joint["type"] == "prismatic":
            motion[:3, 3] = np.asarray(joint["axis"]) / np.linalg.norm(joint["axis"]) * value
        poses[joint["child"]] = poses[joint["parent"]] @ joint["origin"] @ motion
    return poses


def test_a_chain_reproduces_its_urdf_forward_kinematics():
    generator = np.random.default_rng(11)
    worst = 0.0
    for arm in load_robot_arm_chains()["arms"]:
        chain = arm_chain(arm, tool_length_m=TOOL_LENGTH_M)
        urdf = FIXTURES / arm["read_from"]["body"] / "robot.urdf"
        assert chain.joint_names == tuple(row["name"] for row in arm["joints"] if row["type"] != "fixed")
        low = np.where(np.isfinite(chain.lower), chain.lower, -math.pi)
        high = np.where(np.isfinite(chain.upper), chain.upper, math.pi)
        for q in low + (high - low) * generator.random((25, len(low))):
            reference = _reference_frames(urdf, dict(zip(chain.joint_names, q)))
            base_from = np.linalg.inv(reference[arm["base_link"]])
            flange = base_from @ reference[arm["flange_link"]]
            tcp = flange.copy()
            tcp[:3, 3] += TOOL_LENGTH_M * flange[:3, 2]
            pose = kinematics.forward(chain, q)
            # Whole frames, rotation and position, in the base link's frame.
            worst = max(worst, float(np.abs(pose.tool - flange).max()), float(np.abs(pose.tcp - tcp).max()))
        # The reach bound is measured from the first moving joint and covers the tool.
        assert chain.reach_bound_m > TOOL_LENGTH_M
    assert worst < 1e-12, worst


def test_each_flange_points_out_toward_the_gripper_fitted_to_it():
    """The +z of every flange link is out of the flange: the gripper the pipeline composes on each body
    holds its TCP straight along it."""
    for arm in load_robot_arm_chains()["arms"]:
        body = FIXTURES / arm["read_from"]["body"]
        manifest = json.loads((body / "runtime-configuration.json").read_text())["manifest_fields"]
        moving = {row["name"] for row in arm["joints"] if row["type"] != "fixed"}
        fitted = next(row for row in manifest["arms"] if set(row["joint_names"]) == moving)
        reference = _reference_frames(body / "robot.urdf", {})
        tcp = kinematics.pose_matrix(fitted["tool_from_tcp"]["position_m"], fitted["tool_from_tcp"]["orientation_xyzw"])
        offset = (np.linalg.inv(reference[arm["flange_link"]]) @ reference[fitted["tool_link"]] @ tcp)[:3, 3]
        assert abs(offset[0]) < 1e-6 and abs(offset[1]) < 1e-6 and offset[2] > 0.1, (arm["recipe_id"], offset)


def test_the_catalog_is_refused_with_every_reason():
    value = _shipped()
    first, second, third, fourth, fifth, sixth = (arm["recipe_id"] for arm in value["arms"][:6])
    value["schema_version"] = "robot_arm_chains.v0"
    value["arms"][0]["source"]["revision"] = "0" * 40
    value["arms"][1]["joints"][1]["type"] = "floating"
    value["arms"][2]["recipe_id"] = "not_a_recipe"
    value["arms"][3]["joints"][0]["parent"] = "elsewhere"
    value["arms"][4]["joints"] = [row for row in value["arms"][4]["joints"] if row["type"] == "fixed"]
    value["arms"][5]["read_from"]["urdf_sha256"] = "md5:0"
    value["arms"].append(copy.deepcopy(value["arms"][6]))
    with pytest.raises(RobotArmChainError) as caught:
        validate_robot_arm_chains(value)
    assert set(caught.value.errors) == {
        "robot_arm_chains_schema_invalid", f"robot_arm_chains_source_mismatch:{first}",
        f"robot_arm_chains_joint_invalid:{second}:1", "robot_arm_chains_recipe_unknown:not_a_recipe",
        f"robot_arm_chains_route_broken:{fourth}", f"robot_arm_chains_no_movable_joint:{fifth}",
        f"robot_arm_chains_route_broken:{fifth}", f"robot_arm_chains_read_from_invalid:{sixth}",
        "robot_arm_chains_duplicate_recipe:" + value["arms"][6]["recipe_id"]}
    assert third  # the unknown recipe is reported by its new name


@pytest.mark.parametrize("change, code", [
    (lambda row: row.update(lower=1.0, upper=-1.0), "robot_arm_chains_joint_invalid:{recipe}:{index}"),
    (lambda row: row.update(axis=[0.0, 0.0, 0.0]), "robot_arm_chains_joint_invalid:{recipe}:{index}"),
    (lambda row: row.update(xyz=[0.0, float("nan"), 0.0]), "robot_arm_chains_not_json"),
    (lambda row: row.update(extra=1), "robot_arm_chains_joint_invalid:{recipe}:{index}"),
    (lambda row: row.pop("upper"), "robot_arm_chains_joint_invalid:{recipe}:{index}"),
])
def test_a_joint_row_is_checked_field_by_field(change, code):
    value = _shipped()
    arm = value["arms"][0]
    index = next(i for i, row in enumerate(arm["joints"]) if row["type"] == "revolute")
    change(arm["joints"][index])
    with pytest.raises(RobotArmChainError) as caught:
        validate_robot_arm_chains(value)
    assert caught.value.errors == (code.format(recipe=arm["recipe_id"], index=index),)


def test_the_loader_seals_the_file_and_refuses_a_repeated_key(tmp_path):
    catalog = load_robot_arm_chains()
    assert catalog["catalog_digest"] == canonical_digest(catalog, digest_field="catalog_digest")
    assert load_robot_arm_chains()["catalog_digest"] == catalog["catalog_digest"]
    text = CATALOG_PATH.read_text(encoding="utf-8")
    repeated = tmp_path / "repeated.json"
    repeated.write_text(text.replace('"schema_version": "robot_arm_chains.v1"',
                                     '"schema_version": "robot_arm_chains.v1", "schema_version": "x"', 1))
    with pytest.raises(RobotArmChainError) as caught:
        load_robot_arm_chains(repeated)
    assert caught.value.errors == ("robot_arm_chains_duplicate_json_key:schema_version",)
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(RobotArmChainError) as caught:
        load_robot_arm_chains(tmp_path / "broken.json")
    assert caught.value.errors == ("robot_arm_chains_not_json",)


def test_the_refresh_command_rewrites_the_file_from_the_urdfs(tmp_path):
    stale = _shipped()
    stale["arms"][0]["joints"][-1]["xyz"] = [9.0, 9.0, 9.0]
    stale["arms"][0]["read_from"]["urdf_sha256"] = "sha256:" + "0" * 64
    path = tmp_path / "robot_arm_chains.v1.json"
    path.write_text(json.dumps(stale))
    assert main(["--catalog", str(path), "--urdf-root", str(FIXTURES)]) == 0
    assert json.loads(path.read_text()) == _shipped()
    assert path.read_text() == CATALOG_PATH.read_text(encoding="utf-8")
