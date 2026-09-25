from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_scene_packet_request as module
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION
from blueprint_pipeline.native_task_arena_packet import _asset_source
from blueprint_pipeline.task_evaluation_g1_catalog import G1_PRESET_ID, unavailable_g1_preset
from blueprint_pipeline import task_evaluation_packet_planning_setup as packet_planning
from blueprint_pipeline.task_evaluation_policy_canary_setup import policy_canary_setup_digest
from tests.test_task_evaluation_policy_canary_setup import _setup


def _camera(role: str) -> dict:
    return {
        "role": role,
        "policy_input": role == "head",
        "review_only": role == "overview",
        "scoring_input": False,
        "pose_frame": "robot_body" if role == "head" else "world",
        "parent_prim_path": "{ENV_REGEX_NS}/Robot/torso_link"
        if role == "head"
        else "{ENV_REGEX_NS}",
        "optical_convention": "opencv",
        "frame_from_camera_matrix": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "intrinsics": {
            "fx": 500.0,
            "fy": 500.0,
            "cx": 319.5,
            "cy": 239.5,
            "width": 640,
            "height": 480,
        },
    }


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    setup = _setup()
    setup["robot_presets"].append(unavailable_g1_preset())
    setup["setup_digest"] = policy_canary_setup_digest(setup)
    pair = ["humanoidarena_dp_g1_dex3_sonic", "humanoidarena_pi05_g1_dex3_sonic"]
    choice = {
        "schema_version": "task_evaluation_policy_pair_choice.v1",
        "claim_ceiling": "planning_only",
        "source_launch_id": setup["source_launch_id"],
        "offering_digest": setup["offering_digest"],
        "scene_revision_digest": setup["scene_revision_digest"],
        "setup_digest": setup["setup_digest"],
        "robot_preset_id": G1_PRESET_ID,
        "policy_candidate_ids": pair,
        "objective_id": "task_success",
    }
    choice["choice_digest"] = canonical_digest(choice, digest_field="choice_digest")
    packet = tmp_path / "packet"
    packet.mkdir()
    source = {
        "schema_version": "native_task_arena_packet_request.v1",
        "scene_id": "site-scene-a",
        "task_id": setup["task_success_contract"]["scope"]["task_id"],
        "task_spec": {
            "task_kind": "rigid_pick_place",
            "task_success_contract": setup["task_success_contract"],
            "task_success_contract_digest": setup["task_success_contract_digest"],
            "target_position_world_m": [1.0, 2.0, 0.3],
            "robot_workspace_position_bounds_world_m": {"minimum": [0, 0, 0], "maximum": [1, 1, 1]},
        },
        "assets": [
            {
                "semantic_role": "scene_collision",
                "filename": "collision.usda",
                "source": {"root": "evidence"},
            }
        ],
        "appearance_variant": {"representation": "usd_geometry"},
        "physics_frequency_hz": 120,
        "scenario": {"context_kind": "evaluation_cell"},
        "policy_canary_camera_start_configuration": {"franka_only": True},
    }
    source["request_digest"] = canonical_digest(source, digest_field="request_digest")
    (packet / "native_task_arena_packet_request.v1.json").write_text(json.dumps(source))
    (packet / "assets").mkdir()
    (packet / "assets/collision.usda").write_text("sealed collision")
    receipt = {
        "receipt_digest": "sha256:" + "a" * 64,
        "arena_scene_plan_digest": "sha256:" + "d" * 64,
        "request_digest": source["request_digest"],
        "source_bindings": [
            {
                "semantic_role": "scene_collision",
                "staged_relative_path": "assets/collision.usda",
                "staged_size_bytes": len("sealed collision"),
                "staged_sha256": "sha256:" + hashlib.sha256(b"sealed collision").hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(module, "verify_native_task_arena_packet", lambda _: (packet, receipt, []))
    monkeypatch.setattr(
        module, "_asset_source", lambda *a, **kw: (tmp_path / "asset.usda", "sha256:x", 1)
    )
    monkeypatch.setattr(
        module, "_asset_rigid_body_paths", lambda _: {"{ENV_REGEX_NS}/Robot/torso_link"}
    )
    robot = {
        "robot_id": "unitree_g1",
        "base_pose_world": {"position_world_m": [0.0, 0.0, 0.8], "orientation_xyzw": [0, 0, 0, 1]},
        "joint_reset_positions_rad": {"left_knee_joint": 0.3},
    }
    monkeypatch.setattr(module, "build_pinned_g1_arena_robot_configuration", lambda **_: robot)
    context = {
        "schema_version": "adp009d_scenario_instance.v1",
        "program_id": "arm-decision-proof-v1",
        "partition": "development",
        "cell_id": "g1-dev-1",
        "seed": 1,
        "policy_neutral": True,
        "caller_asserted_success": False,
        "learned_policy_outcomes_consulted": False,
        "resolved_parameters": {},
        "factor_records": [],
    }
    context["instance_digest"] = canonical_digest(context, digest_field="instance_digest")
    scenario = {
        "context_kind": "evaluation_cell",
        "cell_id": context["cell_id"],
        "seed": context["seed"],
        "instance_digest": context["instance_digest"],
        "context_document": context,
    }
    authoring = {
        "schema_version": module.SCHEMA,
        "claim_ceiling": "development_only",
        "source_packet_receipt_digest": receipt["receipt_digest"],
        "pair_choice_digest": choice["choice_digest"],
        "base_pose_world": robot["base_pose_world"],
        "task_hand": "right",
        "cameras": [_camera("head"), _camera("overview")],
        "task_spec": {
            "task_kind": "rigid_pick_place",
            "task_success_contract": setup["task_success_contract"],
            "task_success_contract_digest": setup["task_success_contract_digest"],
            "target_position_world_m": [1.0, 2.0, 0.3],
            "robot_workspace_position_bounds_world_m": {
                "minimum": [-1, -1, 0],
                "maximum": [2, 2, 2],
            },
        },
        "scenario": scenario,
    }
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    return dict(
        source_packet_dir=packet,
        evidence_root=tmp_path,
        g1_usd_path=tmp_path / "g1.usd",
        setup=setup,
        choice=choice,
        authoring=authoring,
    )


def test_authors_g1_request_from_exact_source_scene_without_franka_camera_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    request = module.author_g1_scene_packet_request(**args)
    assert request["robot_configuration"]["robot_id"] == "unitree_g1"
    assert request["scene_id"] == "site-scene-a"
    assert request["task_id"] == args["setup"]["task_success_contract"]["scope"]["task_id"]
    assert request["assets"][0]["semantic_role"] == "scene_collision"
    assert request["assets"][0]["source"]["relative_path"] == "scene/collision.usda"
    assert [row["role"] for row in request["cameras"]] == ["head", "overview"]
    assert "policy_canary_camera_start_configuration" not in request
    assert request["scenario"]["context_document"]["partition"] == "development"
    assert request["request_digest"] == canonical_digest(request, digest_field="request_digest")
    assert (
        request["g1_scene_derivation"]["source_packet_receipt_digest"]
        == (args["authoring"]["source_packet_receipt_digest"])
    )


def test_stages_from_retained_packet_without_original_evidence_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    asset = tmp_path / "official-g1.usd"
    asset.write_text("official asset fixture")
    monkeypatch.setattr(module, "G1_USD_SIZE_BYTES", asset.stat().st_size)
    monkeypatch.setattr(module, "G1_USD_SHA256", hashlib.sha256(asset.read_bytes()).hexdigest())
    result = module.prepare_g1_scene_packet_request(
        source_packet_dir=args["source_packet_dir"],
        g1_usd_path=asset,
        setup=args["setup"],
        choice=args["choice"],
        authoring=args["authoring"],
        output_dir=tmp_path / "g1-staged",
    )
    request = json.loads(Path(result["request_path"]).read_text())
    assert result["status"] == "development_packet_request_staged"
    assert Path(result["evidence_root"], "scene/collision.usda").read_text() == "sealed collision"
    assert (
        Path(result["evidence_root"], "robot/official-g1.usd").read_text()
        == "official asset fixture"
    )
    assert (
        request["assets"][0]["source"]["sha256"]
        == "sha256:" + hashlib.sha256(b"sealed collision").hexdigest()
    )
    _asset_source(request["assets"][0], evidence_root=Path(result["evidence_root"]))


def test_one_command_can_materialize_shared_native_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    asset = tmp_path / "official-g1.usd"
    asset.write_text("official asset fixture")
    monkeypatch.setattr(module, "G1_USD_SIZE_BYTES", asset.stat().st_size)
    monkeypatch.setattr(module, "G1_USD_SHA256", hashlib.sha256(asset.read_bytes()).hexdigest())
    calls = []
    monkeypatch.setattr(
        module,
        "materialize_native_task_arena_packet",
        lambda **kw: calls.append(kw) or {"receipt_digest": "sha256:" + "c" * 64},
    )
    output = tmp_path / "g1-staged"
    result = module.prepare_g1_scene_packet_request(
        source_packet_dir=args["source_packet_dir"],
        g1_usd_path=asset,
        setup=args["setup"],
        choice=args["choice"],
        authoring=args["authoring"],
        output_dir=output,
        materialize_packet=True,
    )
    assert result["status"] == "development_packet_materialized"
    assert result["packet_path"] == str(output / "packet")
    assert result["packet_receipt_digest"] == "sha256:" + "c" * 64
    assert calls[0]["link_sources_within"] == output
    assert calls[0]["request"]["robot_configuration"]["robot_id"] == "unitree_g1"


@pytest.mark.parametrize("field", ["source_packet_receipt_digest", "pair_choice_digest"])
def test_stale_authoring_binding_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    authoring = copy.deepcopy(args["authoring"])
    authoring[field] = "sha256:" + "0" * 64
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    args["authoring"] = authoring
    with pytest.raises(ValueError, match="authoring_binding_invalid"):
        module.author_g1_scene_packet_request(**args)


def test_franka_qualification_scenario_cannot_be_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    authoring = copy.deepcopy(args["authoring"])
    document = authoring["scenario"]["context_document"]
    document["partition"] = "qualification"
    document["instance_digest"] = canonical_digest(document, digest_field="instance_digest")
    authoring["scenario"]["instance_digest"] = document["instance_digest"]
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    args["authoring"] = authoring
    with pytest.raises(ValueError, match="scenario_not_development"):
        module.author_g1_scene_packet_request(**args)


def test_wrong_head_resolution_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _inputs(tmp_path, monkeypatch)
    authoring = copy.deepcopy(args["authoring"])
    authoring["cameras"][0]["intrinsics"]["height"] = 360
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    args["authoring"] = authoring
    with pytest.raises(ValueError, match="head_camera_invalid"):
        module.author_g1_scene_packet_request(**args)


def test_task_target_cannot_change_between_embodiments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    authoring = copy.deepcopy(args["authoring"])
    authoring["task_spec"]["target_position_world_m"] = [3.0, 2.0, 0.3]
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    args["authoring"] = authoring
    with pytest.raises(ValueError, match="task_contract_mismatch"):
        module.author_g1_scene_packet_request(**args)


def test_navigation_choice_requires_authored_goal_in_same_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    choice = copy.deepcopy(args["choice"])
    choice["policy_candidate_ids"] = [
        "humanoidarena_dp_g1_dex3_sonic_vision_navi",
        "humanoidarena_pi05_g1_dex3_sonic_vision_navi",
    ]
    choice["objective_id"] = "g1_navigation_goal"
    choice["choice_digest"] = canonical_digest(choice, digest_field="choice_digest")
    args["choice"] = choice
    authoring = copy.deepcopy(args["authoring"])
    authoring["pair_choice_digest"] = choice["choice_digest"]
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    args["authoring"] = authoring
    with pytest.raises(ValueError, match="task_contract_mismatch"):
        module.author_g1_scene_packet_request(**args)
    authoring["task_spec"]["g1_navigation_goal"] = {
        "schema_version": "native_g1_navigation_goal.v1",
        "center_world_m": [1.0, 2.0, 0.0],
        "acceptance_radius_m": 0.2,
        "max_root_height_drift_m": 0.1,
        "settle_window_samples": 5,
        "task_instruction": PUBLISHED_TASK_INSTRUCTION,
        "visible_target_marker": {
            "schema_version": "native_task_target_marker.v1",
            "shape": "flat_yellow_disc",
            "non_colliding": True,
            "surface_position_world_m": [1.0, 2.0, 0.0],
            "radius_m": 0.2,
        },
    }
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    request = module.author_g1_scene_packet_request(**args)
    assert request["task_spec"]["g1_navigation_goal"]["center_world_m"] == [1.0, 2.0, 0.0]


def test_packet_planning_choice_corrects_only_stale_declared_contract_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    packet = args["source_packet_dir"]
    source_path = packet / "native_task_arena_packet_request.v1.json"
    source = json.loads(source_path.read_text())
    source["scene_id"] = args["setup"]["task_success_contract"]["scope"]["site_id"]
    destination = args["setup"]["task_success_contract"]["criteria"]["destination_containment"]
    source["task_spec"]["target_position_world_m"] = [
        (lower + upper) / 2
        for lower, upper in zip(
            destination["position_bounds_world_m"]["minimum"],
            destination["position_bounds_world_m"]["maximum"],
            strict=True,
        )
    ]
    stale_digest = "sha256:" + "e" * 64
    source["task_spec"]["task_success_contract_digest"] = stale_digest
    source["request_digest"] = canonical_digest(source, digest_field="request_digest")
    source_path.write_text(json.dumps(source))
    receipt = {
        "receipt_digest": "sha256:" + "a" * 64,
        "arena_scene_plan_digest": "sha256:" + "d" * 64,
        "request_digest": source["request_digest"],
        "source_bindings": [
            {
                "semantic_role": "scene_collision",
                "staged_relative_path": "assets/collision.usda",
                "staged_size_bytes": len("sealed collision"),
                "staged_sha256": "sha256:" + hashlib.sha256(b"sealed collision").hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(module, "verify_native_task_arena_packet", lambda _: (packet, receipt, []))
    monkeypatch.setattr(
        packet_planning, "verify_native_task_arena_packet", lambda _: (packet, receipt, [])
    )
    setup = packet_planning.make_packet_planning_setup(source_packet_dir=packet)
    choice = packet_planning.make_packet_policy_pair_choice(
        setup=setup,
        objective_id="task_success",
    )
    assert setup["source_declared_task_success_contract_digest"] == stale_digest
    assert setup["task_success_contract_digest"] == args["setup"]["task_success_contract_digest"]
    assert "offering_digest" not in setup
    authoring = args["authoring"]
    authoring["task_spec"]["target_position_world_m"] = source["task_spec"][
        "target_position_world_m"
    ]
    authoring["pair_choice_digest"] = choice["choice_digest"]
    authoring["authoring_digest"] = canonical_digest(authoring, digest_field="authoring_digest")
    request = module.author_g1_scene_packet_request(
        **{**args, "setup": setup, "choice": choice, "authoring": authoring},
    )
    assert (
        request["task_spec"]["task_success_contract_digest"]
        == setup["task_success_contract_digest"]
    )
    assert (
        request["g1_scene_derivation"]["source_declared_task_success_contract_digest"]
        == stale_digest
    )
    assert (
        request["g1_scene_derivation"][
            "task_contract_digest_corrected_from_embedded_confirmed_contract"
        ]
        is True
    )
    changed = copy.deepcopy(authoring)
    changed["task_spec"]["target_position_world_m"] = [8.0, 2.0, 0.3]
    changed["authoring_digest"] = canonical_digest(changed, digest_field="authoring_digest")
    with pytest.raises(ValueError, match="task_contract_mismatch"):
        module.author_g1_scene_packet_request(
            **{**args, "setup": setup, "choice": choice, "authoring": changed},
        )
