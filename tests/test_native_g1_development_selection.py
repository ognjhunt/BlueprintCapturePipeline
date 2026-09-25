from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_development_selection as selection_module
from blueprint_pipeline.native_g1_development_pair import validate_g1_development_pair
from blueprint_pipeline.native_g1_development_worker import PATH_FIELDS, REQUEST_SCHEMA
from blueprint_pipeline.native_g1_navigation_goal import seal_g1_navigation_goal_authority
from blueprint_pipeline.task_evaluation_g1_catalog import G1_PRESET_ID, unavailable_g1_preset
from blueprint_pipeline.task_evaluation_policy_canary_setup import policy_canary_setup_digest
from tests.test_native_g1_development_worker import _request as worker_request
from tests.test_native_g1_navigation_goal import _authority_plan
from tests.test_task_evaluation_policy_canary_setup import _setup


DP = "humanoidarena_dp_g1_dex3_sonic"
PI = "humanoidarena_pi05_g1_dex3_sonic"


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    setup = _setup()
    setup["robot_presets"].append(unavailable_g1_preset())
    setup["setup_digest"] = policy_canary_setup_digest(setup)
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(json.dumps(setup))

    worker = worker_request(tmp_path)
    plan = {
        "scene_id": "scene-a",
        "task_id": setup["task_success_contract"]["scope"]["task_id"],
        "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "unitree_g1"},
        "task_spec": {
            "task_success_contract": setup["task_success_contract"],
            "task_success_contract_digest": setup["task_success_contract_digest"],
        },
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    scene_path = Path(worker["bundle_root"]) / "native_task_arena_scene_plan.v1.json"
    scene_path.write_text(json.dumps(plan))
    monkeypatch.setattr(
        selection_module,
        "_verify_packet",
        lambda _bundle: {
            "arena_scene_plan_digest": plan["plan_digest"],
            "receipt_digest": "sha256:" + "f" * 64,
        },
    )
    selection = {
        "schema_version": selection_module.SCHEMA,
        "setup_digest": setup["setup_digest"],
        "source_launch_id": setup["source_launch_id"],
        "robot_preset_id": G1_PRESET_ID,
        "policy_candidate_ids": [DP, PI],
        "objective_id": "task_success",
        "scene_plan_digest": plan["plan_digest"],
    }
    selection["selection_digest"] = canonical_digest(selection, digest_field="selection_digest")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))

    template = {
        "schema_version": REQUEST_SCHEMA,
        **{field: worker[field] for field in PATH_FIELDS},
        "sonic_encoder_sha256": worker["sonic_encoder_sha256"],
        "sonic_decoder_sha256": worker["sonic_decoder_sha256"],
        "port": worker["port"],
        "max_steps": worker["max_steps"],
        "device": worker["device"],
    }
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(template))
    inventory = Path(worker["inventory_path"])
    inventory.write_text("inventory fixture")
    inventory_sha = selection_module._sha256(inventory)
    rights_paths = {}
    for candidate in (DP, PI):
        rights = copy.deepcopy(worker["rights_review"])
        rights["candidate_id"] = candidate
        rights["scene_plan_digest"] = plan["plan_digest"]
        rights["inventory_file_sha256"] = inventory_sha
        rights["rights_review_digest"] = canonical_digest(
            rights, digest_field="rights_review_digest"
        )
        path = tmp_path / (candidate + "-rights.json")
        path.write_text(json.dumps(rights))
        rights_paths[candidate] = path
    return {
        "setup_path": setup_path,
        "selection_path": selection_path,
        "runtime_template_path": template_path,
        "rights_review_paths": rights_paths,
        "output_dir": tmp_path / "staged",
    }


def test_stages_two_requests_from_shared_catalog_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    result = selection_module.stage_g1_development_selection(**args)
    assert result["status"] == "staged_not_executed"
    assert result["candidate_ids"] == [DP, PI]
    assert result["plan_digest"] == canonical_digest(result, digest_field="plan_digest")
    assert result["ranking_eligible"] is False
    assert (
        json.loads(Path(result["selection_path"]).read_text())["selection_digest"]
        == result["selection_digest"]
    )
    paths = [Path(path) for path in result["request_paths"]]
    assert validate_g1_development_pair(paths)["request_digests"] == result["request_digests"]
    assert [json.loads(path.read_text())["candidate_id"] for path in paths] == [DP, PI]


def test_operator_can_choose_objective_without_handcrafting_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    args["selection_path"] = None
    args["objective_id"] = "task_success"
    result = selection_module.stage_g1_development_selection(**args)
    assert result["candidate_ids"] == [DP, PI]
    assert result["selection_digest"].startswith("sha256:")


def test_stages_navigation_pair_only_with_confirmed_goal_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    scene = _authority_plan()
    setup = json.loads(args["setup_path"].read_text())
    setup["task_success_contract"] = scene["task_spec"]["task_success_contract"]
    setup["task_success_contract_digest"] = setup["task_success_contract"]["contract_digest"]
    setup["setup_digest"] = policy_canary_setup_digest(setup)
    args["setup_path"].write_text(json.dumps(setup))
    template = json.loads(args["runtime_template_path"].read_text())
    Path(template["bundle_root"], "native_task_arena_scene_plan.v1.json").write_text(
        json.dumps(scene)
    )
    monkeypatch.setattr(
        selection_module,
        "_verify_packet",
        lambda _bundle: {
            "arena_scene_plan_digest": scene["plan_digest"],
            "receipt_digest": "sha256:" + "f" * 64,
        },
    )
    args["selection_path"] = None
    args["objective_id"] = "g1_navigation_goal"
    nav_candidates = [DP + "_vision_navi", PI + "_vision_navi"]
    rights_paths = {}
    for previous, candidate in zip((DP, PI), nav_candidates, strict=True):
        path = args["rights_review_paths"][previous]
        rights = json.loads(path.read_text())
        rights["candidate_id"] = candidate
        rights["scene_plan_digest"] = scene["plan_digest"]
        rights["rights_review_digest"] = canonical_digest(
            rights, digest_field="rights_review_digest"
        )
        path.write_text(json.dumps(rights))
        rights_paths[candidate] = path
    args["rights_review_paths"] = rights_paths
    authority = seal_g1_navigation_goal_authority(
        plan=scene, confirmed_by_team_id="team-a", human_reviewer="owner-a"
    )
    authority_path = tmp_path / "navigation-authority.json"
    authority_path.write_text(json.dumps(authority))
    args["navigation_authority_path"] = authority_path
    result = selection_module.stage_g1_development_selection(**args)
    assert result["candidate_ids"] == nav_candidates
    assert result["navigation_goal_authority_digest"] == authority["authority_digest"]
    assert (
        validate_g1_development_pair([Path(path) for path in result["request_paths"]])[
            "objective_id"
        ]
        == "g1_navigation_goal"
    )


def test_rejects_changed_catalog_pair_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    selection = json.loads(args["selection_path"].read_text())
    selection["policy_candidate_ids"] = [DP, DP]
    selection["selection_digest"] = canonical_digest(selection, digest_field="selection_digest")
    args["selection_path"].write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="g1_selection_pair_invalid"):
        selection_module.stage_g1_development_selection(**args)
    assert not args["output_dir"].exists()


def test_rejects_unreviewed_rights_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    path = args["rights_review_paths"][PI]
    rights = json.loads(path.read_text())
    rights["checkpoint_terms_reviewed"] = False
    rights["rights_review_digest"] = canonical_digest(rights, digest_field="rights_review_digest")
    path.write_text(json.dumps(rights))
    with pytest.raises(ValueError, match="g1_worker_rights_review_invalid"):
        selection_module.stage_g1_development_selection(**args)
    assert not args["output_dir"].exists()


def test_rejects_scene_not_bound_to_selected_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    template = json.loads(args["runtime_template_path"].read_text())
    path = Path(template["bundle_root"]) / "native_task_arena_scene_plan.v1.json"
    plan = json.loads(path.read_text())
    plan["task_spec"]["task_success_contract"]["scope"]["site_id"] = "other-site"
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="g1_selection_scene_or_site_mismatch"):
        selection_module.stage_g1_development_selection(**args)
    assert not args["output_dir"].exists()
