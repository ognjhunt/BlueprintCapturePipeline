from copy import deepcopy
import hashlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_beta_packet as packet
from blueprint_pipeline.native_g1_beta_authority import CANDIDATE_DIGESTS, compose_beta_authority

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
