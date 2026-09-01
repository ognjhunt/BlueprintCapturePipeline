from __future__ import annotations

import copy
import json
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    dispatch_launch_request,
    public_launch_profile_descriptor,
    validate_launch_request,
)
from blueprint_pipeline.task_evaluation_policy_run_contract import (
    build_policy_campaign_activation_manifest,
    build_policy_run_plan,
)
from blueprint_pipeline.task_evaluation_policy_canary_setup import (
    policy_canary_setup_digest,
)
from tests.test_task_evaluation_launch_dispatcher import (
    _profile as launch_profile,
    _request as launch_request,
)
from tests.test_task_evaluation_policy_canary_setup import _setup as public_setup
from tests.test_task_evaluation_policy_run_contract import setup as legacy_setup
from scripts.attach_internal_policy_canary_setup import (
    materialize_policy_canary_launch_profile,
)


COMMIT = "a" * 40
CONFIGURED_PROFILE_ID = (
    "task-evaluation-scene-configuration-a65bc2af-r4-binding-dcb54b7eef91"
)
OFFERING_SOURCE_LAUNCH_ID = (
    "adp-new-scene-simple-relocation-839873-a65bc2af-r1-web-paused-ungraded-20260830T102134Z"
)


def _contracts() -> tuple[dict, dict]:
    legacy = legacy_setup()
    public = public_setup()
    legacy["source_launch_id"] = OFFERING_SOURCE_LAUNCH_ID
    quick = legacy["presets"][0]
    for index, cell in enumerate(quick["cells"]):
        scenario = {
            "family": cell["family"],
            "ordinal": index,
            "parameters": {"ordinal_delta": index},
        }
        cell["resolved_scenario"] = scenario
        cell["cell_spec_digest"] = canonical_digest(scenario)
        cell["seed"] = 839_873_000 + index
    quick["scenario_set_digest"] = canonical_digest({"ordered_cells": quick["cells"]})
    quick["nesting_proof_digest"] = canonical_digest(
        {
            "preset_id": "quick_10",
            "scenario_set_digest": quick["scenario_set_digest"],
            "parent_preset_id": None,
            "parent_prefix_count": 0,
            "selection_rule": "published_ordered_prefix",
        }
    )
    legacy["setup_digest"] = canonical_digest(legacy, digest_field="setup_digest")
    visible = public["episode_presets"][0]["matrix"]
    visible["cells"] = [
        {
            "cell_id": cell["cell_id"],
            "family": cell["family"],
            "seed": cell["seed"],
            "partition": (
                "canonical"
                if cell["family"] == "canonical_anchor"
                else "held_out"
                if cell["family"] == "held_out_composition"
                else "stress"
            ),
            "label": f"{cell['family']} {index}",
            "cell_digest": cell["cell_spec_digest"],
        }
        for index, cell in enumerate(quick["cells"])
    ]
    visible["matrix_digest"] = canonical_digest(
        {"ordered_cells": visible["cells"]}
    )
    public["source_launch_id"] = legacy["source_launch_id"]
    public["offering_digest"] = legacy["offering_digest"]
    public["setup_digest"] = policy_canary_setup_digest(public)
    return public, legacy


def _profile_and_request(tmp_path: Path) -> tuple[dict, dict]:
    public, legacy = _contracts()
    base_profile = launch_profile(tmp_path)
    base_profile["profile_id"] = CONFIGURED_PROFILE_ID
    base_profile["profile_digest"] = canonical_digest(
        base_profile, digest_field="profile_digest"
    )
    controller = copy.deepcopy(legacy["preparation_template"]["controller"])
    policy_registry = {
        "uri": "s3://blueprint/policy-canary/policy-registry.json",
        "digest": "sha256:" + "9" * 64,
        "size_bytes": 3_821,
    }
    plan = {
        "schema_version": "task_evaluation_policy_canary_execution_plan.v1",
        "source_commit": COMMIT,
        "configured_source_launch_id": public["source_launch_id"],
        "configured_offering_configuration_run_id": "configuration-run-839873",
        "scene_revision_digest": public["scene_revision_digest"],
        "public_setup_digest": public["setup_digest"],
        "configured_preparation_request_digest": "sha256:" + "3" * 64,
        "policy_controller_configuration": policy_registry,
        "model_rights": controller["model_or_asset_rights"],
        "resolved_scenarios": copy.deepcopy(legacy["presets"][0]["cells"]),
        "legacy_policy_run_setup": legacy,
        "preparation_template": copy.deepcopy(legacy["preparation_template"]),
        "resource_authority": {
            "maximum_hourly_rate_usd": 0.8,
            "hard_cap_usd": 4.0,
            "hard_ttl_seconds": 14_400,
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
        },
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    wrapper = {
        "schema_version": "task_evaluation_policy_canary_profile_materialization_input.v1",
        "profile_id": "scene839873-internal-policy-canary-current",
        "configured_base_profile_id": base_profile["profile_id"],
        "configured_base_profile_digest": base_profile["profile_digest"],
        "configured_source_launch_id": public["source_launch_id"],
        "source_commit": COMMIT,
        "internal_policy_canary_setup": public,
        "internal_policy_canary_execution_plan": plan,
        "materialization_digest": "",
    }
    wrapper["materialization_digest"] = canonical_digest(
        wrapper, digest_field="materialization_digest"
    )
    profile = materialize_policy_canary_launch_profile(
        base_configured_profile=base_profile,
        profile_materialization_input=wrapper,
    )
    assert wrapper["configured_base_profile_id"] == base_profile["profile_id"]
    assert wrapper["configured_source_launch_id"] != base_profile["profile_id"]
    assert wrapper["configured_source_launch_id"] == OFFERING_SOURCE_LAUNCH_ID
    descriptor = public_launch_profile_descriptor(profile)
    assert descriptor["internal_policy_canary_setup"] == public
    assert "internal_policy_canary_execution_plan" not in descriptor
    assert "policy_run_setup" not in descriptor
    request = launch_request(profile)
    matrix = public["episode_presets"][0]["matrix"]
    request.update({
        "source_launch_id": public["source_launch_id"],
        "offering_digest": public["offering_digest"],
        "setup_digest": public["setup_digest"],
        "preset_id": "quick_10",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "scene_revision_digest": public["scene_revision_digest"],
        "scene_controls_status_at_submission": "configured_controls_pending",
        "team_namespace": "blueprint-internal",
        "robot_preset_id": "franka_panda_robotiq_2f85_v1",
        "policy_candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "episode_plan": {
            "preset": "quick_10",
            "episodes_per_policy": 10,
            "policy_count": 2,
            "learned_policy_rollout_count": 20,
            "variation_matrix_digest": matrix["matrix_digest"],
            "resolved_cells": copy.deepcopy(matrix["cells"]),
            "resolved_seeds": [cell["seed"] for cell in matrix["cells"]],
            "coverage_gaps": [],
            "diagnostic_control_rollouts": {
                "zero_action_count": 10,
                "deterministic_scripted_positive_count": 10,
                "total_count": 20,
                "blocking_for_policy_execution": False,
            },
        },
        "notification": {
            "email": "robotics@example.com",
            "notify_on": ["completed", "blocked", "cancelled"],
        },
        "authorization": {
            "actor": {"id": "robotics-member", "role": "team_member"},
            "authorized_at": "2026-08-31T16:30:00Z",
            "spend": {
                "approved": True,
                "currency": "USD",
                "max_spend_usd": 4.0,
                "hard_ttl_seconds": 14_400,
            },
            "execution": {"approved": True},
        },
        "required_controls": {
            **profile["required_controls"],
            "maximum_provider_allocations": 1,
            "retry_cap": 0,
        },
        "controls_qualification_bypassed": False,
        "scene_promotion_permitted": False,
        "official_ranking_permitted": False,
    })
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return profile, request


def test_direct_website_canary_launch_diverts_to_preparation_without_allocator(
    tmp_path: Path, monkeypatch,
) -> None:
    profile, request = _profile_and_request(tmp_path)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / f"{profile['profile_id']}.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    preparation_queue = tmp_path / "preparations"
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT",
        str(preparation_queue),
    )
    assert validate_launch_request(request) == []
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/webapp_internal_policy_canary_launch_request.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert {
        "schema_version": request["schema_version"],
        "run_kind": request["run_kind"],
        "claim_ceiling": request["claim_ceiling"],
        "robot_preset_id": request["robot_preset_id"],
        "policy_candidate_ids": request["policy_candidate_ids"],
        "preset_id": request["preset_id"],
        "episode_plan": {
            "preset": request["episode_plan"]["preset"],
            "episodes_per_policy": request["episode_plan"]["episodes_per_policy"],
            "policy_count": request["episode_plan"]["policy_count"],
            "learned_policy_rollout_count": request["episode_plan"][
                "learned_policy_rollout_count"
            ],
            "resolved_cell_count": len(request["episode_plan"]["resolved_cells"]),
            "resolved_seed_count": len(request["episode_plan"]["resolved_seeds"]),
            "diagnostic_control_rollout_count": request["episode_plan"][
                "diagnostic_control_rollouts"
            ]["total_count"],
            "blocking_for_policy_execution": request["episode_plan"][
                "diagnostic_control_rollouts"
            ]["blocking_for_policy_execution"],
        },
        "notification": {"notify_on": request["notification"]["notify_on"]},
        "authorization": {
            "actor_role": request["authorization"]["actor"]["role"],
            "currency": request["authorization"]["spend"]["currency"],
            "hard_ttl_seconds": request["authorization"]["spend"][
                "hard_ttl_seconds"
            ],
        },
        "required_controls": {
            "maximum_provider_allocations": request["required_controls"][
                "maximum_provider_allocations"
            ],
            "retry_cap": request["required_controls"]["retry_cap"],
        },
        "controls_qualification_bypassed": request[
            "controls_qualification_bypassed"
        ],
        "scene_promotion_permitted": request["scene_promotion_permitted"],
        "official_ranking_permitted": request["official_ranking_permitted"],
    } == fixture

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profiles,
        state_root=tmp_path / "runs",
        execute=True,
        allocator_runner=lambda _argv: (_ for _ in ()).throw(
            AssertionError("allocator must not run before activation")
        ),
    )

    assert receipt["status"] == "queued_for_no_spend_preparation"
    assert receipt["allocator_invoked"] is False
    assert receipt["provider_mutation_attempted"] is False
    envelope_path = next((preparation_queue / "pending").glob("*.json"))
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    preparation = envelope["request"]
    assert preparation["policy_run_selection"]["website_request_digest"] == request[
        "request_digest"
    ]
    assert preparation["policy_run_configuration"]["counts"][
        "learned_policy_rollout_count"
    ] == 20
    assert [
        (cell["cell_id"], cell["seed"])
        for cell in preparation["policy_run_configuration"]["matrix"]["cells"]
    ] == [
        (cell["cell_id"], cell["seed"])
        for cell in request["episode_plan"]["resolved_cells"]
    ]
    assert all(
        "resolved_scenario" in cell
        for cell in preparation["policy_run_configuration"]["matrix"]["cells"]
    )
    policy_plan = build_policy_run_plan(
        preparation["policy_run_configuration"], setup=preparation["policy_run_setup"]
    )
    activation = build_policy_campaign_activation_manifest(
        configuration=preparation["policy_run_configuration"],
        plan=policy_plan,
    )
    assert activation["run_kind"] == "internal_policy_canary"
    assert activation["campaign_unit_count"] == 10
    assert activation["provider_mutation_performed"] is False


def test_visible_cell_tampering_blocks_before_preparation_queue(
    tmp_path: Path, monkeypatch,
) -> None:
    profile, request = _profile_and_request(tmp_path)
    request["episode_plan"]["resolved_cells"][0]["seed"] += 1
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / f"{profile['profile_id']}.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_QUEUE_ROOT",
        str(tmp_path / "preparations"),
    )

    receipt = dispatch_launch_request(
        request_path=request_path,
        profile_dir=profiles,
        state_root=tmp_path / "runs",
        execute=True,
    )

    assert receipt["status"] == "blocked"
    assert receipt["allocator_invoked"] is False
    assert not (tmp_path / "preparations/pending").exists()
