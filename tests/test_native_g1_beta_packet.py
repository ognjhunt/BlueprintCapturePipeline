from copy import deepcopy
import hashlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_beta_packet as packet
from blueprint_pipeline.native_g1_beta_authority import CANDIDATE_DIGESTS
from blueprint_pipeline.native_g1_beta_authority_composer import compose_beta_authority

SHA = "sha256:" + "a" * 64
DP = next(iter(CANDIDATE_DIGESTS))


def seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def cell(seed=7):
    scenario = {"family": "canonical_anchor", "ordinal": 0, "parameters": {}}
    return {"cell_id": "screening.000", "seed": seed, "family": "canonical_anchor", "partition": "screening",
            "resolved_scenario": scenario, "cell_spec_digest": canonical_digest(scenario),
            "resolved_scenario_digest": canonical_digest(scenario)}


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "base"
    root.mkdir()
    scenario = {"context_kind": "canonical", "cell_id": "base", "seed": 1,
                "instance_digest": SHA, "applied_parameters": {"camera": "authored"}}
    task = {"task_success_contract_digest": SHA, "objects": [{"id": "object", "pose": [1, 2, 3]}]}
    common = {"scene_id": "site-current", "task_id": "task"}
    request = seal({**common, "scenario": scenario, "assets": [{"source": "source.usd"}]}, "request_digest")
    contract = seal({**common, "scenario": deepcopy(scenario), "task_spec": task}, "contract_digest")
    scene = seal({**common, "task_kind": "rigid_pick_place", "task_spec": task,
        "scenario": deepcopy(scenario), "runtime_contract_digest": contract["contract_digest"],
        "robot": {"robot_id": "unitree_g1", "camera": {"height": 480}}, "geometry": [1, 2, 3]}, "plan_digest")
    for name, value in ((packet.REQUEST, request), (packet.CONTRACT, contract), (packet.PLAN, scene)):
        packet._write(root / name, value)
    (root / "source.usd").write_text("#usda 1.0\n")
    rows = [packet._file_row(path) for path in root.iterdir()]
    receipt = seal({"schema_version": "native_task_arena_packet_receipt.v1", "request_digest": request["request_digest"],
        "runtime_contract_digest": contract["contract_digest"], "arena_scene_plan_digest": scene["plan_digest"],
        "scenario_instance_digest": SHA, "artifacts": rows, "source_bindings": []}, "receipt_digest")
    packet._write(root / packet.RECEIPT, receipt)
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", lambda *a, **k: None)
    source = {"capture_id": "capture", "task_id": "task", "scene_intent_id": "intent",
              "task_context_digest": SHA, "scene_intent_digest": SHA, "scene_revision_digest": SHA}
    authority = compose_beta_authority(candidate_id=DP, scene_plan_digest=scene["plan_digest"],
        source_packet_receipt_digest=receipt["receipt_digest"], task_success_contract_digest=SHA,
        checkpoint_digest=CANDIDATE_DIGESTS[DP], current_source=source, site_id="site-current", registration_digest=SHA)
    return root, authority


def test_native_packet_derivation_retains_assets_geometry_and_frozen_context(tmp_path, monkeypatch):
    base, authority = fixture(tmp_path, monkeypatch)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
    result = packet.materialize_cell_packet(base_packet=base, output_dir=tmp_path / "cell", cell=cell(), base_authority=authority)
    assert result["rights_authority"]["base_authority"] == authority
    new = json.loads((tmp_path / "cell" / packet.PLAN).read_text())
    old = json.loads((base / packet.PLAN).read_text())
    for key in ("geometry", "task_spec", "robot"):
        assert new[key] == old[key]
    assert new["scenario"]["seed"] == 7
    assert new["scenario"]["applied_parameters"] == old["scenario"]["applied_parameters"]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()} == before
    assert packet.verify_cell_packet(base_packet=base, cell_packet=tmp_path / "cell", cell=cell(), base_authority=authority) == result


@pytest.mark.parametrize("target", ["geometry", "robot", "task_spec"])
def test_resealed_cell_cannot_change_scene(tmp_path, monkeypatch, target):
    base, authority = fixture(tmp_path, monkeypatch)
    root = tmp_path / "cell"
    packet.materialize_cell_packet(base_packet=base, output_dir=root, cell=cell(), base_authority=authority)
    scene = json.loads((root / packet.PLAN).read_text())
    scene[target] = {"invented": True}
    packet._write(root / packet.PLAN, seal(scene, "plan_digest"))
    receipt = json.loads((root / packet.RECEIPT).read_text())
    receipt["arena_scene_plan_digest"] = scene["plan_digest"]
    receipt["artifacts"] = [packet._file_row(root / row["relative_path"]) for row in receipt["artifacts"]]
    packet._write(root / packet.RECEIPT, seal(receipt, "receipt_digest"))
    with pytest.raises(ValueError, match="document_changed"):
        packet.verify_cell_packet(base_packet=base, cell_packet=root, cell=cell(), base_authority=authority)


def test_packet_extra_file_is_rejected(tmp_path, monkeypatch):
    base, authority = fixture(tmp_path, monkeypatch)
    root = tmp_path / "cell"
    packet.materialize_cell_packet(base_packet=base, output_dir=root, cell=cell(), base_authority=authority)
    (root / "unexpected.usd").write_text("unbound")
    with pytest.raises(ValueError, match="closure_changed"):
        packet.verify_cell_packet(base_packet=base, cell_packet=root, cell=cell(), base_authority=authority)


# --- navigation cells move the robot's start, nothing else (docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md, part C) ---

from blueprint_pipeline import native_g1_beta_authority as rights  # noqa: E402

NAV = "humanoidarena_dp_g1_dex3_sonic_vision_navi"
YAW_QUARTER = [0.0, 0.0, 2 ** -0.5, 2 ** -0.5]  # The robot faces +y.


def offset_cell(ordinal=1, seed=11):
    scenario = {"family": rights.START_OFFSET_FAMILY, "ordinal": ordinal,
                "parameters": rights.start_offset_parameters(ordinal)}
    return {"cell_id": f"screening.navigation.{ordinal:03d}", "seed": seed, "family": rights.START_OFFSET_FAMILY,
            "partition": "screening", "resolved_scenario": scenario, "cell_spec_digest": canonical_digest(scenario),
            "resolved_scenario_digest": canonical_digest(scenario)}


def navigation_fixture(tmp_path, monkeypatch):
    root = tmp_path / "base"
    root.mkdir()
    pose = {"position_world_m": [1.0, 2.0, 0.79], "orientation_xyzw": list(YAW_QUARTER)}
    scenario = {"context_kind": "evaluation_cell", "cell_id": "scene.nominal", "seed": 0, "instance_digest": SHA}
    task = {"task_kind": "goal_region_navigation", "task_success_contract_digest": SHA,
            "start_pose_world": [1.0, 2.0, 0.0, *YAW_QUARTER], "g1_navigation_goal": {"center_world_m": [1.0, 5.0, 0.0]}}
    common = {"scene_id": "site-current", "task_id": "task"}
    request = seal({**common, "scenario": scenario, "assets": [{"source": "source.usd"}], "task_spec": task,
                    "robot_base_pose_world": deepcopy(pose), "robot_configuration": {"base_pose_world": deepcopy(pose),
                                                                                     "robot_id": "unitree_g1"}},
                   "request_digest")
    contract = seal({**common, "scenario": deepcopy(scenario), "task_spec": task,
                     "robot": {"robot_id": "unitree_g1", "base_pose_world": deepcopy(pose)}}, "contract_digest")
    scene = seal({**common, "task_kind": "goal_region_navigation", "task_spec": task, "scenario": deepcopy(scenario),
                  "runtime_contract_digest": contract["contract_digest"], "geometry": [1, 2, 3],
                  "robot": {"robot_id": "unitree_g1", "base_pose_world": deepcopy(pose)},
                  "reset": {"robot_base_pose_world": deepcopy(pose), "robot_joint_positions_rad": {"knee": 0.3}}},
                 "plan_digest")
    for name, value in ((packet.REQUEST, request), (packet.CONTRACT, contract), (packet.PLAN, scene)):
        packet._write(root / name, value)
    (root / "source.usd").write_text("#usda 1.0\n")
    rows = [packet._file_row(path) for path in root.iterdir()]
    receipt = seal({"schema_version": "native_task_arena_packet_receipt.v1", "request_digest": request["request_digest"],
        "runtime_contract_digest": contract["contract_digest"], "arena_scene_plan_digest": scene["plan_digest"],
        "scenario_instance_digest": SHA, "artifacts": rows, "source_bindings": []}, "receipt_digest")
    packet._write(root / packet.RECEIPT, receipt)
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", lambda *a, **k: None)
    source = {"capture_id": "capture", "task_id": "task", "scene_intent_id": "intent",
              "task_context_digest": SHA, "scene_intent_digest": SHA, "scene_revision_digest": SHA}
    authority = compose_beta_authority(candidate_id=NAV, scene_plan_digest=scene["plan_digest"],
        source_packet_receipt_digest=receipt["receipt_digest"], task_success_contract_digest=SHA,
        checkpoint_digest=CANDIDATE_DIGESTS[NAV], current_source=source, site_id="site-current", registration_digest=SHA)
    return root, authority, pose


def _documents(root):
    return {name: json.loads((root / filename).read_text())
            for name, filename in (("request", packet.REQUEST), ("contract", packet.CONTRACT), ("scene", packet.PLAN))}


def test_the_start_offsets_are_four_frozen_rows_in_the_robots_start_frame():
    from blueprint_pipeline.native_g1_navigation_goal import NAVIGATION_TASK_KIND

    assert rights.NAVIGATION_TASK_KIND == NAVIGATION_TASK_KIND  # The bundled module's literal copy.
    assert rights.NAVIGATION_START_OFFSETS == (("start", 0.0, 0.0), ("left_15cm", 0.0, 0.15),
                                               ("right_15cm", 0.0, -0.15), ("back_15cm", -0.15, 0.0))
    facing_y = {"position_world_m": [1.0, 2.0, 0.79], "orientation_xyzw": YAW_QUARTER}
    moved = [rights.apply_start_offset(facing_y, rights.start_offset_parameters(i)) for i in range(4)]
    assert moved[0] == facing_y
    for pose, (x, y) in zip(moved[1:], [(0.85, 2.0), (1.15, 2.0), (1.0, 1.85)]):
        assert pose["position_world_m"] == pytest.approx([x, y, 0.79]) and pose["orientation_xyzw"] == YAW_QUARTER


@pytest.mark.parametrize("ordinal", range(4))
def test_a_navigation_cell_moves_only_the_robot_start(tmp_path, monkeypatch, ordinal):
    base, authority, pose = navigation_fixture(tmp_path, monkeypatch)
    cell = offset_cell(ordinal)
    result = packet.materialize_cell_packet(base_packet=base, output_dir=tmp_path / "cell", cell=cell,
                                            base_authority=authority)
    old, new = _documents(base), _documents(tmp_path / "cell")
    moved = rights.apply_start_offset(pose, rights.start_offset_parameters(ordinal))
    assert new["request"]["robot_base_pose_world"] == new["request"]["robot_configuration"]["base_pose_world"] == moved
    assert new["contract"]["robot"]["base_pose_world"] == moved
    assert new["scene"]["robot"]["base_pose_world"] == new["scene"]["reset"]["robot_base_pose_world"] == moved
    # Everything else is the base: the task and its compiled start, the goal, the geometry, the joints.
    for name, digest in (("request", "request_digest"), ("contract", "contract_digest"), ("scene", "plan_digest")):
        before, after = deepcopy(old[name]), deepcopy(new[name])
        for document in (before, after):
            for key in (digest, "scenario", "runtime_contract_digest", "robot_base_pose_world"):
                document.pop(key, None)
            for key in ("robot", "robot_configuration", "reset"):
                if key in document:
                    document[key].pop("base_pose_world", None)
                    document[key].pop("robot_base_pose_world", None)
        assert after == before, name
    assert new["scene"]["task_spec"]["start_pose_world"] == old["scene"]["task_spec"]["start_pose_world"]
    context = json.loads((tmp_path / "cell" / packet.REQUEST).read_text())["scenario"]["context_document"]
    assert context["resolved_parameters"] == rights.start_offset_parameters(ordinal)
    assert packet.verify_cell_packet(base_packet=base, cell_packet=tmp_path / "cell", cell=cell,
                                     base_authority=authority) == result
    # The binding check expects the same offset, applied by the same rule.
    derived = new["scene"]
    assert rights.validate_cell_binding(old["scene"], derived, cell, result["rights_authority"],
                                        base_contract=old["contract"]) == derived


def test_the_binding_refuses_a_navigation_cell_whose_robot_did_not_move(tmp_path, monkeypatch):
    base, authority, _ = navigation_fixture(tmp_path, monkeypatch)
    cell = offset_cell(3)
    result = packet.materialize_cell_packet(base_packet=base, output_dir=tmp_path / "cell", cell=cell,
                                            base_authority=authority)
    old, new = _documents(base), _documents(tmp_path / "cell")
    unmoved = deepcopy(new["scene"])
    unmoved["robot"]["base_pose_world"] = old["scene"]["robot"]["base_pose_world"]
    seal(unmoved, "plan_digest")
    cell_authority = rights.derive_cell_authority(base_authority=authority, cell=cell,
        scene_plan_digest=unmoved["plan_digest"], packet_receipt_digest=result["packet_receipt_digest"])
    with pytest.raises(ValueError, match="^g1_beta_cell_changed_frozen_scene$"):
        rights.validate_cell_binding(old["scene"], unmoved, cell, cell_authority, base_contract=old["contract"])


@pytest.mark.parametrize("mutation", ["parameters", "ordinal", "family", "beyond_table"])
def test_a_start_offset_cell_must_be_a_row_of_the_table(mutation):
    cell = offset_cell(1)
    scenario = cell["resolved_scenario"]
    if mutation == "parameters":
        scenario["parameters"] = {"robot_start_forward_m": 0.0, "robot_start_lateral_m": 0.3}
    elif mutation == "ordinal":
        scenario["ordinal"] = 2  # Row 2 is 15 cm right, not left.
    elif mutation == "family":
        scenario["family"] = "canonical_anchor"
    else:
        scenario.update(ordinal=4, parameters={"robot_start_forward_m": 0.0, "robot_start_lateral_m": 0.0})
    cell["cell_spec_digest"] = cell["resolved_scenario_digest"] = canonical_digest(scenario)
    with pytest.raises(ValueError, match="^g1_beta_frozen_cell_invalid$"):
        rights.validate_frozen_cell(cell)
    assert rights.validate_frozen_cell(offset_cell(1)) == offset_cell(1)


def test_a_cell_family_must_match_its_scene_kind(tmp_path, monkeypatch):
    for name in ("navigation", "rigid"):
        (tmp_path / name).mkdir()
    nav, nav_authority, _ = navigation_fixture(tmp_path / "navigation", monkeypatch)
    with pytest.raises(ValueError, match="^g1_beta_cell_family_task_mismatch$"):
        packet.materialize_cell_packet(base_packet=nav, output_dir=tmp_path / "anchor", cell=cell(),
                                       base_authority=nav_authority)
    rigid, rigid_authority = fixture(tmp_path / "rigid", monkeypatch)
    with pytest.raises(ValueError, match="^g1_beta_cell_family_task_mismatch$"):
        packet.materialize_cell_packet(base_packet=rigid, output_dir=tmp_path / "offset", cell=offset_cell(1),
                                       base_authority=rigid_authority)
