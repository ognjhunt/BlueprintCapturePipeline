"""Pinned public finger geometry, not imported/native/physical grasp proof."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as XML

import numpy as np
import pytest

from blueprint_pipeline.deployment_embodiment_runtime_configuration import runtime_profiles

ROOT = Path(__file__).parent / "fixtures" / "gripper_closure_sources"
SOURCES = json.loads((ROOT / "sources.json").read_text())["sources"]


def _rotation(rpy):
    r, p, y = rpy
    cx, cy, cz = math.cos(r), math.cos(p), math.cos(y)
    sx, sy, sz = math.sin(r), math.sin(p), math.sin(y)
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])


def _origin(element):
    origin = element.find("origin")
    matrix = np.eye(4)
    if origin is not None:
        matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        matrix[:3, :3] = _rotation(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
    return matrix


def _stl_vertices(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + 50 * count
    return np.array([struct.unpack_from("<9f", data, 84 + index*50 + 12)
                     for index in range(count)]).reshape(-1, 3)


@pytest.mark.parametrize("source", SOURCES, ids=lambda row: row["embodiment_id"])
def test_fixture_preserves_pinned_public_mesh_and_license_bytes(source):
    root = ROOT / source["embodiment_id"]
    for file in source["files"]:
        data = (root / file["path"]).read_bytes()
        assert len(data) == file["size_bytes"]
        assert "sha256:" + hashlib.sha256(data).hexdigest() == file["sha256"]
    assert source["license"]["path"] in {row["path"] for row in source["files"]}


@pytest.mark.parametrize("source", SOURCES, ids=lambda row: row["embodiment_id"])
def test_declared_close_moves_actual_distal_collision_pads_together(source):
    root = ROOT / source["embodiment_id"]
    tree = XML.parse(root / source["entry_path"]).getroot()
    incoming = {joint.find("child").get("link"): joint for joint in tree.findall("joint")}
    links = {link.get("name"): link for link in tree.findall("link")}
    profile = next(row for row in runtime_profiles()["profiles"] if row["embodiment_id"] == source["embodiment_id"])
    # Source fingers extend +X on TRLC and -Z on Axol. Jaw travel is Y / X.
    trlc = source["embodiment_id"] == "trlc_dk1_bimanual"
    distal_axis, distal_sign, jaw_axis = (0, 1, 1) if trlc else (2, -1, 0)
    for arm in profile["source_bindings"]:
        states = []
        for state in (0, 1):  # Declared open, closed; no assumed sign of motion.
            pads = []
            for contact in arm["contact_links"]:
                joint = incoming[contact]
                assert joint.get("type") == "prismatic" and joint.find("parent").get("link") == arm["tool_link"]
                master = joint.find("mimic")
                if master is None:
                    q = arm["fingers"][joint.get("name")][state]
                else:
                    q = (arm["fingers"][master.get("joint")][state] * float(master.get("multiplier", "1"))
                         + float(master.get("offset", "0")))
                joint_pose = _origin(joint)
                joint_pose[:3, 3] += joint_pose[:3, :3] @ (q * np.fromstring(joint.find("axis").get("xyz"), sep=" "))
                collision = links[contact].find("collision")
                mesh = collision.find("geometry/mesh")
                filename = mesh.get("filename").removeprefix("package://trlc_dk1_description/")
                # Source-relative meshes are below the retained original URDF.
                path = root / source["entry_path"]
                vertices = _stl_vertices(path.parent / filename)
                transform = joint_pose @ _origin(collision)
                points = vertices @ transform[:3, :3].T + transform[:3, 3]
                depth = distal_sign * points[:, distal_axis]
                pads.append(points[depth > depth.max() - .020])
            states.append(pads)
        positive, negative = sorted(range(2), key=lambda i: states[0][i][:, jaw_axis].mean(), reverse=True)
        gaps = [pads[positive][:, jaw_axis].min() - pads[negative][:, jaw_axis].max() for pads in states]
        assert gaps[0] > .075, gaps
        assert -.011 < gaps[1] < .005, gaps
        assert gaps[0] - gaps[1] > .075, gaps
        # This checks the authored closing direction. TRLC's full negative
        # joint limit overlaps the distal envelopes by ~10mm; contact forces
        # and the declared nominal width still need native measurement.
