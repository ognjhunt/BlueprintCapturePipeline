"""Source metadata for the registered bare-arm MJCF profile; geometry stays MJCF.

This reads the pinned model's hinge/default/keyframe/affine-servo subset. It
does not translate geometry, synthesize inertias, or make an application tool.
The official Isaac MJCF converter still receives the original retained model.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from defusedxml import ElementTree as XML

from .deployment_embodiment_assets import AssetPreparationError


def _require(condition, reason):
    if not condition:
        raise AssetPreparationError("runtime_configuration_mjcf_" + reason)


def _values(value, size, reason):
    try:
        result = [float(x) for x in value.split()]
    except (AttributeError, TypeError, ValueError) as exc:
        raise AssetPreparationError("runtime_configuration_mjcf_" + reason) from exc
    _require(len(result) == size and all(math.isfinite(x) for x in result), reason)
    return result


def _pose(attributes):
    _require(not set(attributes) & {"euler", "axisangle", "xyaxes", "zaxis"}, "orientation_encoding_unsupported")
    authored = _values(attributes.get("quat", "1 0 0 0"), 4, "quaternion_invalid")
    norm = math.hypot(*authored)
    _require(norm > 0, "quaternion_invalid")
    return {"position_m": _values(attributes.get("pos", "0 0 0"), 3, "position_invalid"),
            "authored_quaternion_wxyz": authored, "orientation_wxyz": [x / norm for x in authored]}


def configuration_from_mjcf(entry: Path, identity: str, profile: dict, source_receipt: dict) -> dict:
    """Bind source joints to native position control with explicit bare-arm limits."""
    root = XML.fromstring(entry.read_bytes())
    _require(root.tag == "mujoco", "root_invalid")
    compiler = root.find("compiler")
    angle = compiler.get("angle", "degree") if compiler is not None else "degree"
    _require(angle in {"degree", "radian"}, "angle_invalid")
    radians = math.pi / 180 if angle == "degree" else 1.0
    defaults = {"main": {}}

    def visit_defaults(node, inherited):
        values = deepcopy(inherited)
        for child in node:
            if child.tag != "default":
                values[child.tag] = {**values.get(child.tag, {}), **child.attrib}
        name = node.get("class", "main")
        _require(name not in defaults or name == "main", "default_class_duplicate")
        defaults[name] = values
        for child in node.findall("default"):
            visit_defaults(child, values)

    for node in root.findall("default"):
        visit_defaults(node, {})

    def attributes(node, inherited="main"):
        selected = node.get("class", inherited)
        _require(selected in defaults, "default_class_missing")
        return {**defaults[selected].get(node.tag, {}), **node.attrib}

    bodies, joints, sites = {}, [], []

    def visit_body(node, parent, inherited):
        attrs = attributes(node, inherited)
        name = attrs.get("name")
        _require(isinstance(name, str) and name not in bodies, "body_name_invalid")
        child_class = node.get("childclass", inherited)
        body = {"name": name, "parent": parent, **_pose(attrs), "inertial": None}
        bodies[name] = body
        inertial = node.find("inertial")
        if inertial is not None:
            attrs_inertial = dict(inertial.attrib)
            if "mass" in attrs_inertial and "diaginertia" in attrs_inertial:
                body["inertial"] = {"mass_kg": _values(attrs_inertial["mass"], 1, "mass_invalid")[0],
                    "principal_inertia_kg_m2": _values(attrs_inertial["diaginertia"], 3, "inertia_invalid"),
                    **_pose(attrs_inertial)}
        _require(not node.findall("freejoint") and len(node.findall("joint")) <= 1, "body_joint_layout_unsupported")
        for joint in node.findall("joint"):
            attrs_joint = attributes(joint, child_class)
            _require(attrs_joint.get("type", "hinge") == "hinge", "joint_type_unsupported")
            axis = _values(attrs_joint.get("axis", "0 0 1"), 3, "axis_invalid")
            norm = math.hypot(*axis)
            bounds = [x * radians for x in _values(attrs_joint.get("range"), 2, "joint_range_missing")]
            _require(norm > 0 and bounds[0] < bounds[1], "joint_range_or_axis_invalid")
            joints.append({"name": attrs_joint.get("name"), "body": name, "range_rad": bounds,
                "position_m": _values(attrs_joint.get("pos", "0 0 0"), 3, "joint_position_invalid"),
                "axis": [x / norm for x in axis], "source_class": joint.get("class", child_class)})
        for site in node.findall("site"):
            attrs_site = attributes(site, child_class)
            sites.append({"name": attrs_site.get("name"), "body": name, **_pose(attrs_site)})
        for child in node.findall("body"):
            visit_body(child, name, child_class)

    world = root.find("worldbody")
    _require(world is not None, "worldbody_missing")
    for node in world.findall("body"):
        visit_body(node, None, "main")
    _require(not profile["fingers"] and profile["base_link"] in bodies and profile["tool_link"] in bodies,
             "bare_arm_binding_missing")
    chain, current = [], profile["tool_link"]
    while current != profile["base_link"]:
        _require(current in bodies and current not in chain, "arm_chain_unbound")
        chain.append(current)
        current = bodies[current]["parent"]
    chain.reverse()
    names = [joint["name"] for joint in joints]
    _require(all(isinstance(x, str) and x for x in names) and len(set(names)) == len(names)
             and len(joints) == profile["dof"] and [joint["body"] for joint in joints] == chain,
             "controlled_joint_order_invalid")
    actuator_node = root.find("actuator")
    _require(actuator_node is not None, "actuators_missing")
    servos = {}
    for actuator in actuator_node:
        attrs = attributes(actuator)
        _require(actuator.tag == "general" and attrs.get("gaintype", "fixed") == "fixed"
                 and attrs.get("biastype", "none") == "affine", "actuator_type_unsupported")
        gain = [float(x) for x in attrs.get("gainprm", "1").split()]
        bias = [float(x) for x in attrs.get("biasprm", "0 0 0").split()]
        _require(1 <= len(gain) <= 10 and 3 <= len(bias) <= 10
                 and all(math.isfinite(x) for x in gain + bias) and gain[0] > 0
                 and bias[0] == 0 and bias[1] == -gain[0] and bias[2] < 0,
                 "actuator_affine_position_contract_invalid")
        name = attrs.get("joint")
        _require(name in names and name not in servos, "actuator_joint_binding_invalid")
        servos[name] = {"name": attrs.get("name"), "stiffness": gain[0], "damping": -bias[2],
                        "gainprm": gain, "biasprm": bias}
    _require(set(servos) == set(names), "unactuated_joint")
    reset = {j["name"]: max(j["range_rad"][0], min(0.0, j["range_rad"][1])) for j in joints}
    reset_source = {"kind": "simulation_zero_clamped_to_source_limits"}
    home = root.find("keyframe/key[@name='home']")
    if home is not None:
        # MuJoCo keyframe qpos is in native radians even for degree-authored XML.
        positions = _values(home.get("qpos"), len(joints), "home_qpos_invalid")
        _require(all(j["range_rad"][0] <= q <= j["range_rad"][1] for j, q in zip(joints, positions)),
                 "home_outside_limits")
        reset = dict(zip(names, positions))
        reset_source = {"kind": "source_keyframe", "name": "home", "joint_order": names, "qpos_rad": positions}
    missing = [j["body"] for j in joints if bodies[j["body"]]["inertial"] is None or any(x <= 0 for x in
               [bodies[j["body"]]["inertial"]["mass_kg"], *bodies[j["body"]]["inertial"]["principal_inertia_kg_m2"]])]
    fields = {"schema_version": "blueprint.robot_embodiment.v2", "embodiment_id": identity,
        "form": "fixed_base_arm", "base": {"mode": "fixed"}, "grippers": [], "tool_link": profile["tool_link"],
        "joints": [{"name": j["name"], "control": "position", "lower": j["range_rad"][0],
                    "upper": j["range_rad"][1], "max_velocity": 1.0,
                    "stiffness": servos[j["name"]]["stiffness"], "damping": servos[j["name"]]["damping"]} for j in joints],
        "reset_joint_positions": reset, "actuator": {"stiffness": servos[names[0]]["stiffness"],
                                                     "damping": servos[names[0]]["damping"]},
        "cameras": [{"name": "overhead", "mount_link": None, "position_m": [.5, 0., 1.5],
                     "orientation_xyzw": [1., 0., 0., 0.], "width": 320, "height": 176, "horizontal_fov_deg": 90.},
                    {"name": "arm_wrist", "mount_link": profile["tool_link"], "position_m": [0., .08, 0.],
                     "orientation_xyzw": [0., 0., 0., 1.], "width": 320, "height": 176, "horizontal_fov_deg": 90.}],
        "control_hz": 60, "action_chunk_rows": 48,
        "reach_m": min(3., max(.1, sum(math.sqrt(sum(x*x for x in bodies[b]["position_m"])) for b in chain)))}
    return {"schema_version": "embodiment_runtime_configuration.v1", "claim_ceiling": "configuration_only",
        "embodiment_id": identity, "source_bundle_digest": source_receipt["bundle_digest"], "station_parameters": None,
        "manifest_fields": fields, "source_name_mapping": {name: name for name in sorted(set(names) | set(bodies))},
        "source_kinematics": {"format": "mjcf", "compiler_angle": angle, "bodies": list(bodies.values()),
                              "joints": joints, "actuators": servos, "sites": sites, "reset": reset_source},
        "source_dynamics": {"moving_link_count": len(joints),
            "moving_links_without_positive_authored_mass_and_inertia": sorted(missing),
            "estimated_inertias_added": False, "simulator_dynamics_validated": False},
        "simulation_assumptions": ["source_affine_PD_values_mapped_to_native_position_drives_not_dynamics_validated",
            "simulation_velocity_cap_1_rad_per_second_not_source_velocity_limit",
            "90_degree_simulated_camera_rig_not_site_calibration", "reach_is_source_chain_upper_bound",
            "bare_arm_without_application_tool_or_grasp_capability", "tool_link_is_body_frame_not_application_TCP",
            *([] if home is not None else ["simulation_zero_reset_clamped_to_source_limits"])],
        "source_limitations": [*source_receipt["missing"],
            *(["missing_positive_authored_moving_link_inertias:" + ",".join(sorted(missing))] if missing else [])],
        "execution_authorized": False, "policy_compatibility_established": False}
