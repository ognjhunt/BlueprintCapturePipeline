"""Development-only grasp-frame contracts; never native or physical proof."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as XML

import numpy as np
import pytest

from blueprint_pipeline import scripted_robot_grasp_frames as frames
from blueprint_pipeline.deployment_embodiment_runtime_configuration import runtime_profiles
from blueprint_pipeline.team_arm_kinematics import pose_matrix
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest

IDS = tuple(frames._IDENTITIES)


def development_manifest(identity="franka_research_3_bimanual", model=None):
    """Synthetic body declaration for software tests; source tool names only.

    ``model`` may name a separately authored development-only USD fixture.
    This helper never represents imported source-model or native execution.
    """
    from blueprint_pipeline.scripted_robot_robotiq_runtime import BODIES as ORIGINAL_BODIES
    from blueprint_pipeline.scripted_robot_industrial_robotiq_runtime import BODIES as INDUSTRIAL_BODIES
    BODIES = ORIGINAL_BODIES | INDUSTRIAL_BODIES
    kind = "robotiq_ur" if identity in BODIES else frames._IDENTITIES[identity]
    frame = frames._FRAMES[kind]
    source = next(row for row in runtime_profiles()["profiles"] if row["embodiment_id"] == identity)
    names = ["arm"] if identity in ({"kinova_gen3", "ufactory_xarm7"} | BODIES) else ["left", "right"]
    tcp = {"yam": ([0., 0., 0.], [0., 0., 0., 1.]), "piper": ([0., 0., 0.], [0., 0., 0., 1.]),
           "axol": ([0., 0., 0.], [0., 0., 0., 1.]), "fr3": ([0., 0., .1034], [0., 0., 0., 1.]),
           "robotiq_ur": ([0., 0., .134], [0., 0., 1., 0.]),
           "kinova": ([0., 0., .12], [0., 0., math.sin(-1.57/2), math.cos(-1.57/2)]),
           "xarm": ([0., 0., .172], [0., 0., 0., 1.]), "trlc": ([.158, 0., 0.], [-.5, .5, -.5, .5])}[kind]
    arms, channels, joints, reset = [], [], [], {}
    for name in names:
        prefix = "" if name == "arm" else name + "_"
        if identity == "ufactory_xarm7_bimanual":
            prefix = {"left": "L_", "right": "R_"}[name]
        tool = prefix + frame["tool"]
        arm_joints = [prefix + "development_joint_" + str(i) for i in range(source["source_bindings"][0]["dof"])]
        joints += [{"name": joint, "control": "position", "lower": -3., "upper": 3., "max_velocity": 1.}
                   for joint in arm_joints]
        reset.update(dict.fromkeys(arm_joints, 0.))
        fingers = [{"name": prefix + joint, "open": q[0], "closed": q[1]} for joint, q in frame["jaws"].items()]
        reset.update({joint["name"]: joint["open"] for joint in fingers})
        channels.append({"name": name + "_gripper", "tool_link": tool, "joints": fingers, "max_width_m": .08,
                         "contact_links": [prefix + contact for contact in frame["contacts"]]})
        arms.append({"name": name, "joint_names": arm_joints, "base_link": prefix + "development_base",
                     "tool_link": tool, "gripper_name": name + "_gripper",
                     "tool_from_tcp": {"position_m": list(tcp[0]), "orientation_xyzw": list(tcp[1])}})
    value = {"schema_version": "blueprint.robot_embodiment.v2", "embodiment_id": identity,
        "form": "fixed_base_arm" if len(arms) == 1 else "bimanual_arm", "base": {"mode": "fixed"},
        "model": {"format": "usd", "url": "https://models.example/development-only.usd",
                  "sha256": "sha256:" + (hashlib.sha256(Path(model).read_bytes()).hexdigest() if model else "1"*64)},
        "joints": joints, "grippers": channels, "arms": arms, "tool_link": arms[0]["tool_link"],
        "reset_joint_positions": reset, "actuator": {"stiffness": 100., "damping": 20.},
        "control_hz": 60, "action_chunk_rows": 48, "reach_m": 1.,
        "cameras": [{"name": "overhead", "mount_link": None, "position_m": [0., 0., 1.],
                     "orientation_xyzw": [1., 0., 0., 0.], "width": 320, "height": 176, "horizontal_fov_deg": 90.}]}
    return validate_embodiment_manifest(value)


def _tool_grasp(manifest, binding, index=0):
    tcp = manifest["arms"][index]["tool_from_tcp"]
    grasp = binding[manifest["arms"][index]["name"]]["tcp_from_grasp"]
    return pose_matrix(tcp["position_m"], tcp["orientation_xyzw"]) @ pose_matrix(grasp["position_m"], grasp["orientation_xyzw"])


@pytest.mark.parametrize("identity", IDS)
def test_all_ten_source_configurations_retain_explicit_noncalibrated_basis(identity):
    manifest = development_manifest(identity)
    result = frames.grasp_bindings(manifest)
    assert list(result) == [arm["name"] for arm in manifest["arms"]]
    for row in result.values():
        assert row["basis_reference"].startswith("https://github.com/")
        assert row["simulation_assumptions"] and row["source_geometry_basis"]
        assert row["publisher_calibration_claimed"] is False
        assert row["physical_contact_point_measured"] is False
        assert row["native_task_success_claimed"] is False
        rotation = pose_matrix(**{  # Check normalised, right-handed SE(3).
            "position": row["tcp_from_grasp"]["position_m"], "orientation_xyzw": row["tcp_from_grasp"]["orientation_xyzw"]})[:3, :3]
        assert np.linalg.det(rotation) == pytest.approx(1.)


@pytest.mark.parametrize("identity,approach,jaw,position", [
    ("i2rt_yam_bimanual", [0, 0, 1], [1, 0, 0], [0, 0, .138017]),
    ("agilex_piper_bimanual", [0, 0, 1], [0, 1, 0], [0, 0, .126073]),
    ("almond_axol", [0, 0, -1], [1, 0, 0], [0, 0, -.14254]),
    ("trlc_dk1_bimanual", [1, 0, 0], [0, 1, 0], [.15285, 0, 0]),
    ("universal_robots_ur5e_bimanual", [0, 0, 1], [1, 0, 0], [0, 0, .134]),
    ("kinova_gen3", [0, 0, 1], [0, 1, 0], [0, 0, .130324]),
    ("ufactory_xarm7", [0, 0, 1], [0, 1, 0], [0, 0, .158331]),
])
def test_task_axes_are_composed_through_each_actual_tcp(identity, approach, jaw, position):
    manifest = development_manifest(identity)
    for index, _ in enumerate(manifest["arms"]):
        transform = _tool_grasp(manifest, frames.grasp_bindings(manifest), index)
        assert transform[:3, 2] == pytest.approx(approach)
        assert transform[:3, 1] == pytest.approx(jaw)
        assert transform[:3, 3] == pytest.approx(position)


def test_calibrated_tcp_change_preserves_source_grasp_and_independent_other_arm():
    manifest = development_manifest("i2rt_yam_bimanual")
    original = frames.grasp_bindings(manifest)
    manifest.pop("embodiment_digest")
    manifest["arms"][0]["tool_from_tcp"] = {"position_m": [.01, -.02, .03], "orientation_xyzw": [.5, .5, .5, .5]}
    changed = frames.grasp_bindings(manifest)
    assert changed["right"] == original["right"] and changed["left"] != original["left"]
    assert _tool_grasp(manifest, changed) == pytest.approx(_tool_grasp(development_manifest("i2rt_yam_bimanual"), original))


@pytest.mark.parametrize("identity", ["almond_axol", "trlc_dk1_bimanual"])
def test_stale_reversed_closure_has_its_own_reason(identity):
    manifest = development_manifest(identity)
    manifest.pop("embodiment_digest")
    for channel in manifest["grippers"]:
        for joint in channel["joints"]:
            joint["open"], joint["closed"] = joint["closed"], joint["open"]
            manifest["reset_joint_positions"][joint["name"]] = joint["open"]
    with pytest.raises(ValueError, match="scripted_grasp_source_closure_mismatch"):
        frames.grasp_bindings(manifest)


def test_relabelled_tool_does_not_inherit_source_frame():
    manifest = development_manifest()
    manifest.pop("embodiment_digest")
    manifest["grippers"][0]["contact_links"] = ["different_pad_a", "different_pad_b"]
    with pytest.raises(ValueError, match="scripted_grasp_source_tool_binding_mismatch"):
        frames.grasp_bindings(manifest)


@pytest.mark.parametrize("identity", ["almond_axol", "trlc_dk1_bimanual"])
def test_mesh_approximation_lies_inside_retained_distal_pad_depth(identity):
    # Independently check the actual source meshes retained for the closure
    # regression, rather than testing the numeric descriptor against itself.
    from tests.test_source_gripper_closure_geometry import ROOT, SOURCES, _origin, _stl_vertices

    source = next(row for row in SOURCES if row["embodiment_id"] == identity)
    root = ROOT / identity
    tree = XML.parse(root / source["entry_path"]).getroot()
    axis, sign = (0, 1) if identity == "trlc_dk1_bimanual" else (2, -1)
    profiles = next(row for row in runtime_profiles()["profiles"] if row["embodiment_id"] == identity)
    manifest = development_manifest(identity)
    grasp = _tool_grasp(manifest, frames.grasp_bindings(manifest))[:3, 3]
    arm = profiles["source_bindings"][0]
    for contact in arm["contact_links"]:
        joint = next(j for j in tree.findall("joint") if j.find("child").get("link") == contact)
        collision = tree.find(f"link[@name='{contact}']/collision")
        path = root / source["entry_path"]
        vertices = _stl_vertices(path.parent / collision.find("geometry/mesh").get("filename"))
        transform = _origin(joint) @ _origin(collision)
        points = vertices @ transform[:3, :3].T + transform[:3, 3]
        depth = sign * points[:, axis]
        assert depth.max() - .020 < sign * grasp[axis] < depth.max()
    assert json.dumps(frames.grasp_bindings(manifest))  # No NumPy-only payload values.
