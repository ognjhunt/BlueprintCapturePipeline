"""A rigid scene's robot-free probe, recorded in a team evaluation's first cell, decides the scene.

Beta journey step: L5 team policy evaluation. Hermetic: the rigid probe's
deterministic stand-in scene (``FlatSupport``) plays the GPU host. No Isaac,
provider, network or spend.
"""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline import agent_execution_offer
from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_readback import _compose_pose_xyzw
from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
from blueprint_pipeline.team_embodiment_runtime import asset_root_for_scoring_pose, probe_status
from blueprint_pipeline.team_robot_free_probe import (
    probe_request,
    retain_recorded_probe,
    scored_task_spec,
)
from tests.test_rigid_robot_free_probe import FlatSupport, _sealed_contract, rigid_case

REVISION = "sha256:" + "4" * 64


def _scored_spec(tmp_path: Path) -> dict:
    """The task a team is scored with: the scene's specification under the job's sealed contract."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    _configured, _references, _docs, spec = rigid_case(tmp_path)
    spec = dict(spec, site_id="site-1", task_id="task-1")
    return scored_task_spec(spec, _sealed_contract(spec))


def test_the_first_cell_records_the_probe_and_its_pass_opens_the_scene(tmp_path, monkeypatch) -> None:
    spec = _scored_spec(tmp_path / "case")
    request = probe_request(task_spec=spec, scene_revision_digest=REVISION)
    assert request is not None and request["binding"]["task_spec_digest"] == canonical_digest(spec)
    # The GPU host executes exactly the control plane's plan; the stand-in scene plays physics.
    trace = record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"]), plan=request["plan"],
                                          binding=request["binding"])
    # The GPU host scores it before any team episode, as closeout will; a tampered request decides nothing.
    assert probe_status(request=request, trace=trace) == "passed"
    assert probe_status(request={**request, "task_spec": {**spec, "task_id": "other"}}, trace=trace) == (
        "probe_request_invalid")
    published: list[Path] = []
    monkeypatch.setattr(agent_execution_offer, "publish_agent_execution_offer",
                        lambda root: published.append(root) or {"status": "published"})
    probe_root, capture = tmp_path / "source-launch", tmp_path / "capture"
    probe_root.mkdir()
    facts = {"scene": {"task_spec": spec, "scene_revision_digest": REVISION}}
    outcome = retain_recorded_probe(profile={"robot_free_probe_root": str(probe_root), "capture_root": str(capture)},
                                    facts=facts, trace=trace)
    assert outcome["receipt_status"] == "passed" and outcome["write"] == "written", outcome
    assert published == [capture] and outcome["offer_status"] == "published"
    # The gate now admits a scored run bound to the exact specification and contract teams are scored with.
    receipt = gate.retained_probe_receipt(probe_root)
    gate.require_robot_free_probe(
        receipt, scene_revision_digest=REVISION, task_spec_digest=canonical_digest(spec),
        task_kind="rigid_pick_place",
        task_success_contract_digest=gate.task_success_contract_binding(
            "rigid_pick_place", spec["task_success_contract"]))
    # A decided receipt is sealed: a later trace never replaces it.
    again = retain_recorded_probe(profile={"robot_free_probe_root": str(probe_root), "capture_root": str(capture)},
                                  facts=facts, trace=trace)
    assert again["write"] in {"unchanged", "kept_existing_decided"}


def test_a_scene_that_holds_no_goal_fails_and_a_failed_recording_stays_undecided(tmp_path, monkeypatch) -> None:
    spec = _scored_spec(tmp_path / "case")
    request = probe_request(task_spec=spec, scene_revision_digest=REVISION)
    monkeypatch.setattr(agent_execution_offer, "publish_agent_execution_offer", lambda root: {"status": "published"})
    # The goal lies beyond the support's edge: released there, the object falls.
    fell = record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"], support_max_xy=(3.05, -1.4)),
                                         plan=request["plan"], binding=request["binding"])
    root = tmp_path / "fell"
    root.mkdir()
    profile = {"robot_free_probe_root": str(root), "capture_root": str(tmp_path / "capture")}
    facts = {"scene": {"task_spec": spec, "scene_revision_digest": REVISION}}
    assert retain_recorded_probe(profile=profile, facts=facts, trace=fell)["receipt_status"] == "failed"
    assert probe_status(request=request, trace=fell) == "failed"  # no team episode runs on this scene
    # A recording that did not complete (the GPU host refused the plan) never decides the scene.
    broken = {**fell, "status": "failed", "error": "RuntimeError:team_robot_free_probe_request_or_rate_invalid",
              "trace_digest": ""}
    broken["trace_digest"] = canonical_digest(broken, digest_field="trace_digest")
    other = tmp_path / "broken"
    other.mkdir()
    outcome = retain_recorded_probe(profile={**profile, "robot_free_probe_root": str(other)}, facts=facts,
                                    trace=broken)
    assert outcome["receipt_status"] == "probe_unavailable"
    assert json.loads((other / gate.RECEIPT_FILENAME).read_text())["learned_policy_evaluation_admitted"] is False


def test_a_task_that_cannot_be_probed_without_a_robot_records_nothing(tmp_path) -> None:
    spec = _scored_spec(tmp_path / "case")
    assert probe_request(task_spec={**spec, "task_kind": "articulated_open_close"}, scene_revision_digest=REVISION) is None
    assert probe_request(task_spec={"task_kind": "rigid_pick_place"}, scene_revision_digest=REVISION) is None


def test_the_object_is_placed_so_its_scoring_frame_lands_on_the_probe_pose() -> None:
    offset = {"position_m": [0.02, -0.01, 0.05], "orientation_xyzw": [0.0, 0.0, 0.3826834, 0.9238795]}
    scoring = [1.0, 2.0, 0.8, 0.0, 0.0, 0.7071068, 0.7071068]
    root = asset_root_for_scoring_pose(scoring, offset)
    landed = _compose_pose_xyzw(root, [*offset["position_m"], *offset["orientation_xyzw"]])
    assert all(abs(a - b) < 1e-6 for a, b in zip(landed, scoring, strict=True))
