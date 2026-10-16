"""The internal beta L5 fixture exercises production staging without customer authority."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import blueprint_pipeline.internal_team_evaluation_diagnostic as diagnostic
from blueprint_pipeline import team_candidate_quick10_launch as launch
from blueprint_pipeline.droid_https_policy_client import INTERFACE_ID, REQUEST_SCHEMA
from blueprint_pipeline.team_candidate_quick10_registry import offer_eligibility
from tests.test_native_task_arena_bundle import _runtime_source_packet
from tests.test_native_task_arena_packet import _request


def test_authored_fixture_binds_the_real_native_rigid_scorer(tmp_path):
    from blueprint_pipeline.native_task_episode_environment import NativeRigidScoringEnvironment

    root = _fixture(tmp_path)
    plan = json.loads((root / "scene/native-task-packet/native_task_arena_scene_plan.v1.json").read_text())
    # This is the actual adapter constructor that refused the retained GPU run.
    NativeRigidScoringEnvironment(environment=object(),
        task_readback=SimpleNamespace(read_task_sample=lambda: {}), task_spec=plan["task_spec"])


def test_review_camera_has_room_for_placement_variations(tmp_path):
    from blueprint_pipeline.native_task_camera_framing_expectation import camera_framing_expectation

    root = _fixture(tmp_path)
    plan = json.loads((root / "scene/native-task-packet/native_task_arena_scene_plan.v1.json").read_text())
    overview = next(camera for camera in plan["cameras"] if camera["role"] == "overview")
    assert overview["policy_input"] is False
    # Retained cells 07/08/12 had only 62-63 interior pixels against 64 required.
    # Keep the runtime guard and give the authored review view a larger target.
    positions = [[.5 + x, y, .78] for x in [-.03, 0., .03] for y in [-.03, .03, .17, .23]]
    framing = camera_framing_expectation(camera=overview, object_extent_m=[.06] * 3,
                                        object_positions_world=positions)
    assert framing["expected_bbox_area_px"] >= 225


def _approval() -> dict:
    return {"schema_version": "blueprint.internal_team_diagnostic_owner_approval.v1",
            "approved_by": "@ognjhunt", "episodes": 200, "cap_usd": 20, "max_attempts": 1,
            "scene_id": diagnostic.SCENE_ID, "candidate_id": diagnostic.CANDIDATE_ID,
            "customer_recommendation_permitted": False, "expires_at_iso": "2099-01-01T00:00:00Z",
            "approval_reference": "test-only-owner-authorization"}


def _fixture(tmp_path: Path) -> Path:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_request(tmp_path / "profile-assets", articulated=False)))
    runtime = _runtime_source_packet(tmp_path)
    root = tmp_path / "isolated"
    root.mkdir()
    diagnostic.materialize_fixture(root / "scene", droid_profile=profile, runtime_source=runtime)
    return root


def test_stages_all_200_real_cells_and_keeps_owner_issued_credential_separate(tmp_path, monkeypatch) -> None:
    root = _fixture(tmp_path)
    approval = tmp_path / "owner.json"
    approval.write_text(json.dumps(_approval()))
    commit = "6" * 40
    monkeypatch.setattr(launch, "_source_commit", lambda: commit)
    monkeypatch.setattr(launch, "_default_provider_zero", lambda: {"provider_zero_verified": True})
    with pytest.raises(ValueError, match="robot_free_probe_receipt_missing"):
        diagnostic.stage(root, policy_url="https://policy.example/internal-diagnostic", owner_approval=approval,
                         source_commit=commit)
    assert not (root / "evaluation").exists() and not (root / "policy-bearer").exists()
    # A hermetic probe fixture is supplied explicitly; production must retain
    # the actual simulator probe and may not manufacture this test receipt.
    _retain_test_probe(root)
    result = diagnostic.stage(root, policy_url="https://policy.example/internal-diagnostic", owner_approval=approval,
                              source_commit=commit)
    job = Path(result["job_dir"])
    admission = json.loads((job / "team_candidate_admission.json").read_text())
    assert admission["internal_diagnostic"] and admission["trained_model"] is False
    assert admission["customer_recommendation_permitted"] is False
    assert admission["credential"]["kms_lease_claimed"] is False
    assert admission["credential"]["authority"] == "owner_issued_local_single_run"
    shards = [launch._current_shard(job, i) for i in range(1)]
    assert sum(row["cap_usd"] for row in shards) == 20
    assert {row["cell_count"] for row in shards} == {200}
    cells = [cell for row in shards for cell in row["cells"]]
    assert len({cell["cell_id"] for cell in cells}) == 200
    assert (job / ".team-credential" / "bearer").stat().st_mode & 0o077 == 0
    assert (root / "policy-bearer").read_bytes() == (job / ".team-credential" / "bearer").read_bytes()
    # Preparing cannot launch or silently issue a second credential.
    assert not list(job.rglob("allocator_process.json"))
    with pytest.raises(FileExistsError):
        diagnostic.stage(root, policy_url=result["policy_url"], owner_approval=approval, source_commit=commit)
    registry = json.loads((root / "registry.json").read_text())
    facts, blockers = offer_eligibility(registry["profiles"][0])
    assert facts is not None and blockers == []
    assert facts["probe"]["receipt_digest"].startswith("sha256:")
    assert admission["robot_free_probe_receipt_digest"] == facts["probe"]["receipt_digest"]
    for change in ({"cap_usd": 21}, {"customer_recommendation_permitted": True}, {"approved_by": "someone-else"}):
        with pytest.raises(ValueError, match="owner_approval_invalid"):
            diagnostic.internal_admission(facts=facts, approval={**_approval(), **change},
                policy_url=result["policy_url"], credential_digest="sha256:" + "4" * 64)
    called = []
    monkeypatch.setattr(launch, "_team_metadata", lambda *_args: diagnostic.metadata())

    def completed(*, job_dir, hooks):
        called.append(job_dir)
        control = json.loads((job_dir / launch.CONTROL_FILENAME).read_text())
        assert hooks["execution_monitor"](control)["ok"] is True
        terminal = {"status": "completed", "episodes_run": 200}
        (job_dir / launch.TERMINAL_FILENAME).write_text(json.dumps(terminal))
        return terminal

    monkeypatch.setattr(launch, "run_staged_evaluation", completed)
    assert diagnostic.run(root)["episodes_run"] == 200
    result = json.loads((root / "internal_diagnostic_result.json").read_text())
    assert result["trained_policy_evaluated"] is False and result["customer_recommendation_permitted"] is False
    # A completed operator command reads its terminal, without renting again or needing its policy service.
    (root / "policy-bearer").unlink()
    assert diagnostic.run(root)["episodes_run"] == 200 and called == [job]


def _retain_test_probe(root: Path) -> None:
    from blueprint_pipeline import robot_free_probe_gate as gate
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.team_robot_free_probe import scored_task_spec

    inputs = json.loads((root / "scene/team_scene_inputs.v1.json").read_text())
    task_spec = json.loads((root / "scene/native-task-packet/native_task_arena_scene_plan.v1.json").read_text())["task_spec"]
    receipt = {"schema_version": gate.RIGID_RECEIPT_SCHEMA, "status": "passed",
        "configured_scene_revision_digest": inputs["scene_revision_digest"],
        "task_spec_digest": canonical_digest(scored_task_spec(task_spec, inputs["task_success_contract"])),
        "learned_policy_evaluation_admitted": True,
        "task_success_contract_digest": inputs["task_success_contract_digest"],
        "checks": dict.fromkeys(gate.REQUIRED_CHECKS["rigid_pick_place"], True),
        "proof_boundary": {"robot_embodiment_used": False}, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (root / "probe").mkdir()
    (root / "probe" / gate.RECEIPT_FILENAME).write_text(json.dumps(receipt))


def test_policy_repeats_only_observed_robot_state_and_never_claims_success() -> None:
    state = [0., -0.6, 0., -2.4, 0., 1.8, 0.]
    request = {"schema_version": REQUEST_SCHEMA, "policy_interface_id": INTERFACE_ID,
               "observation": {"observation/joint_position": state, "observation/gripper_position": [0.9],
                               "prompt": "move the cube", "pretend_success": True}}
    result = diagnostic.hold_position(request)
    assert result == {"actions": [state + [0.9]] * 10}
    for bad in ([float("nan")] * 7, [0.] * 6):
        with pytest.raises(ValueError, match="state_invalid"):
            diagnostic.hold_position({**request, "observation": {**request["observation"],
                                                                 "observation/joint_position": bad}})
