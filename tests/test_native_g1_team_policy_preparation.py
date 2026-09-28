"""Selected intake prepares its exact registered scene without allocating."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_team_policy_preparation as preparation
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from tests.test_native_g1_team_provider_bundle import _inputs, COMMIT


def _selected(tmp_path, monkeypatch):
    inputs, _ = _inputs(tmp_path, monkeypatch, endpoint=True)
    authority = preparation.verify_g1_team_policy_authority(**inputs["authority_arguments"])
    binding = authority["registry_binding"]
    # Retained geometry/runtime validators are hermetic fixtures; preparation
    # must still pass the selected registered directory to the real bundler.
    for source in inputs["scene_packet_root"].iterdir():
        shutil.copy2(source, Path(binding["manipulation_packet_dir"]) / source.name)
    shutil.copytree(inputs["publisher_source"], Path(binding["publisher_source_dir"]), dirs_exist_ok=True)
    shutil.copy2(inputs["runtime_source_receipt"], Path(binding["runtime_source_receipt_path"]))
    return {
        "authority_arguments": inputs["authority_arguments"],
        "work_root": tmp_path / "selected-preparations",
        "sonic_asset_dir": inputs["sonic_asset_dir"],
        "implementation_commit": COMMIT,
    }, authority


def test_real_selected_preparation_seals_registered_scene_and_reuses_exact_bytes(tmp_path, monkeypatch):
    args, authority = _selected(tmp_path, monkeypatch)
    result = preparation.prepare_g1_team_policy(**args)
    assert result["status"] == "bundle_prepared_not_executed"
    assert result["intent_id"] == authority["intent"]["intent_id"]
    assert result["policy_profile_digest"] == authority["intent"]["request"]["policy_profile"]["profile_digest"]
    assert result["objective_id"] == "task_success"
    assert result["provider_mutation_performed"] is False
    assert result["preparation_digest"] == digest(result, digest_field="preparation_digest")
    bundle_path = Path(result["bundle_receipt_path"])
    bundle = json.loads(bundle_path.read_text())
    assert bundle["execution_packet_digest"] == result["execution_packet_digest"]
    assert bundle["scene_id"] == authority["trusted_setup"]["scene_id"]
    assert preparation.prepare_g1_team_policy(**args) == result
    with Path(bundle["bundle_path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="bundle_bytes_invalid"):
        preparation.prepare_g1_team_policy(**args)


def test_cached_preparation_never_silently_rebinds_release_or_approval(tmp_path, monkeypatch):
    args, _ = _selected(tmp_path, monkeypatch)
    preparation.prepare_g1_team_policy(**args)
    with pytest.raises(ValueError, match="preparation_conflict"):
        preparation.prepare_g1_team_policy(**{**args, "implementation_commit": "b" * 40})
    approval = Path(args["authority_arguments"]["approval_path"])
    approval.unlink()
    with pytest.raises(ValueError, match="authority_path_invalid"):
        preparation.prepare_g1_team_policy(**args)


def test_missing_operator_sonic_cache_or_symlink_work_refuses_before_bundle(tmp_path, monkeypatch):
    args, _ = _selected(tmp_path, monkeypatch)
    monkeypatch.setattr(preparation, "build_g1_team_provider_bundle", lambda **kwargs: pytest.fail("unsafe preparation reached builder"))
    with pytest.raises(ValueError, match="operator_directory_invalid"):
        preparation.prepare_g1_team_policy(**{**args, "sonic_asset_dir": tmp_path / "missing"})
    target = tmp_path / "foreign"
    target.mkdir()
    link = tmp_path / "work-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="operator_directory_invalid"):
        preparation.prepare_g1_team_policy(**{**args, "work_root": link})


def test_navigation_choice_seals_registered_movement_packet_not_manipulation(tmp_path, monkeypatch):
    from blueprint_pipeline import native_g1_team_provider_bundle as bundle
    from blueprint_pipeline.native_g1_team_policy_run_intake import stage_g1_team_policy_run
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION
    from tests.test_native_g1_team_policy_approval import _approval
    args, authority = _selected(tmp_path, monkeypatch)
    request = {**authority["intent"]["request"], "run_id": "team-g1-navigation",
               "objective_id": "g1_navigation_goal"}
    request["request_digest"] = digest(request, digest_field="request_digest")
    live = args["authority_arguments"]
    accepted = stage_g1_team_policy_run(
        value=request, registry_path=live["registry_path"], queue_root=tmp_path / "movement-queue",
        authenticated_client="blueprint-webapp", trusted_clients=live["trusted_clients"], now_epoch=live["now_epoch"],
    )
    live["intent_path"] = tmp_path / "movement-queue" / accepted["intent_id"] / "intent.json"
    approval = _approval(authority["trusted_setup"], request["policy_profile"],
                         authority["operator_approval"]["runtime_binding"], objective="g1_navigation_goal")
    Path(live["approval_path"]).write_text(json.dumps(approval))
    binding = authority["registry_binding"]
    movement = Path(binding["movement_packet_dir"])
    manipulation = Path(binding["manipulation_packet_dir"])
    shutil.copy2(manipulation / bundle.REQUEST_FILENAME, movement / bundle.REQUEST_FILENAME)
    plan = json.loads((manipulation / bundle.PLAN_FILENAME).read_text())
    plan["task_spec"].update(task_kind="rigid_pick_place", g1_navigation_goal={
        "schema_version": "native_g1_navigation_goal.v1", "center_world_m": [2.0, 0.0, 0.0],
        "acceptance_radius_m": 0.3, "max_root_height_drift_m": 0.2, "settle_window_samples": 2,
        "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        "visible_target_marker": {"schema_version": "native_task_target_marker.v1",
                                  "shape": "flat_yellow_disc", "non_colliding": True,
                                  "surface_position_world_m": [2.0, 0.0, 0.0], "radius_m": 0.4},
    })
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    (movement / bundle.PLAN_FILENAME).write_text(json.dumps(plan))
    seen = []
    def scene_packet(root):
        seen.append(root)
        assert root == movement
        scene_request = json.loads((root / bundle.REQUEST_FILENAME).read_text())
        receipt = {"receipt_digest": "sha256:" + "f" * 64,
                   "request_digest": scene_request["request_digest"],
                   "arena_scene_plan_digest": plan["plan_digest"]}
        return root, receipt, [{"relative_path": name} for name in (bundle.REQUEST_FILENAME, bundle.PLAN_FILENAME)]
    monkeypatch.setattr(bundle, "verify_native_task_arena_packet", scene_packet)
    result = preparation.prepare_g1_team_policy(**args)
    assert result["objective_id"] == "g1_navigation_goal"
    assert seen == [movement]


def test_authority_is_reopened_after_waiting_for_preparation_lock(tmp_path, monkeypatch):
    args, _ = _selected(tmp_path, monkeypatch)
    original = preparation.verify_g1_team_policy_authority
    calls = []
    def authority(**kwargs):
        calls.append(None)
        if len(calls) == 2:
            path = Path(args["authority_arguments"]["approval_path"])
            approval = json.loads(path.read_text())
            approval["site_observation_exchange_authorized"] = False
            from blueprint_pipeline.decision_evidence_contracts import canonical_digest
            approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
            path.write_text(json.dumps(approval))
        return original(**kwargs)
    monkeypatch.setattr(preparation, "verify_g1_team_policy_authority", authority)
    monkeypatch.setattr(preparation, "build_g1_team_provider_bundle", lambda **kwargs: pytest.fail("revoked preparation built bundle"))
    with pytest.raises(ValueError, match="approval_binding_invalid"):
        preparation.prepare_g1_team_policy(**args)
    assert len(calls) == 2
