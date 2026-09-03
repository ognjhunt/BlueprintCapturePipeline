from __future__ import annotations

import copy
import json
from pathlib import Path

from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
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
    def activation_ref(name: str, character: str) -> dict:
        return {
            "uri": f"s3://blueprint/policy-canary/activation/{name}.json",
            "digest": "sha256:" + character * 64,
            "size_bytes": 256,
        }
    plan = {
        "schema_version": "task_evaluation_policy_canary_execution_plan.v1",
        "source_commit": COMMIT,
        "configured_source_launch_id": public["source_launch_id"],
        "configured_offering_configuration_run_id": "configuration-run-839873",
        "scene_revision_digest": public["scene_revision_digest"],
        "public_setup_digest": public["setup_digest"],
        "task_success_contract": copy.deepcopy(public["task_success_contract"]),
        "task_success_contract_digest": public["task_success_contract_digest"],
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
        "activation_automation": {
            "mode": "automatic_after_no_spend_compilation",
            "release_window_template": activation_ref("release-window-template", "1"),
            "lineage": {
                "kind": "predecessor",
                "prior_authority": activation_ref("prior-authority", "2"),
                "prior_result": activation_ref("prior-result", "3"),
                "prior_launch_receipt": activation_ref("prior-launch-receipt", "4"),
                "prior_webapp_sync": activation_ref("prior-webapp-sync", "5"),
                "prior_provider_zero": activation_ref("prior-provider-zero", "6"),
                "prior_spend_reconciliation": activation_ref("prior-spend", "7"),
                "construction_result": activation_ref("construction-result", "8"),
            },
            "authorization_template": {
                "reference": "automatic policy-canary activation",
                "authorized_by": "blueprint-policy-lead",
                "profile_revision": "policy-canary-v1",
                "valid_for_seconds": 3600,
            },
            "requested_mutations": {
                "profile_publication": False,
                "catalog_synchronization": False,
                "standing_authorization": False,
                "policy_campaign_queue": True,
            },
        },
        "lineage_aliases": {
            "capture_session_id": public["source_launch_id"],
            "capture_session_id_semantics": (
                "configured_scene_offering_source_launch_id_no_capture_upload_session"
            ),
            "intake_id": "configuration-run-839873",
            "intake_id_semantics": "configured_scene_offering_configuration_run_id",
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
        "task_success_contract": copy.deepcopy(public["task_success_contract"]),
        "task_success_contract_digest": public["task_success_contract_digest"],
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
        "task_success_contract": copy.deepcopy(public["task_success_contract"]),
        "task_success_contract_digest": public["task_success_contract_digest"],
        "team_namespace": "blueprint-internal",
        "robot_preset_id": "droid_franka_panda_robotiq_2f85_v1",
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
    source_rights = {
        "schema_version": "policy_canary_episode_interpretation_source_rights_admission.v1",
        "run_id": request["run_id"],
        "team_namespace": "blueprint-internal",
        "accepted_by": "robotics-member",
        "accepted_on": "2026-08-31T16:30:00Z",
        "external_disclosure_authorized": True,
        "disclosed_artifact_roles": [
            "contact_force_trace",
            "deterministic_score",
            "frame_manifest",
            "lossless_frame",
            "review_video",
            "state_trace",
            "task_success_contract",
        ],
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "admission_digest": "",
    }
    source_rights["admission_digest"] = canonical_digest(
        source_rights, digest_field="admission_digest"
    )
    interpretation_authority = {
        "schema_version": "policy_canary_episode_interpretation_batch_authority.v1",
        "status": "approved",
        "run_id": request["run_id"],
        "interpreter": {
            "interpreter_id": "openai_multimodal_episode_interpreter_v1",
            "principal_kind": "independent_interpreter",
            "provider_id": "openai",
            "execution_site": "external_provider",
            "runtime": "openai_agents_sdk",
            "model": "gpt-5.6-luna",
            "model_version": "gpt-5.6-luna",
        },
        "interpreter_profile_digest": "sha256:eca3944e331b60cc08fdb1548d753be7c1b513b7b703ad5fce8401b09eb83baf",
        "allowed_artifact_roles": source_rights["disclosed_artifact_roles"],
        "external_disclosure_authorized": True,
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "maximum_cost_usd": 1.5,
        "source_rights_admission_digest": source_rights["admission_digest"],
        "accepted_by": "robotics-member",
        "accepted_on": "2026-08-31T16:30:00Z",
        "authority_reference": f"website:{request['run_id']}",
        "authority_digest": "",
    }
    interpretation_authority["authority_digest"] = canonical_digest(
        interpretation_authority, digest_field="authority_digest"
    )
    request["episode_interpretation_source_rights_admission"] = source_rights
    request["episode_interpretation_authority"] = interpretation_authority
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
        "task_success_contract": {
            "schema_version": request["task_success_contract"]["schema_version"],
            "contract_digest": request["task_success_contract"][
                "contract_digest"
            ],
        },
        "task_success_contract_digest": request[
            "task_success_contract_digest"
        ],
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
    assert preparation["policy_run_selection"]["task_success_contract"] == request[
        "task_success_contract"
    ]
    assert preparation["policy_run_configuration"]["task_success_contract"] == request[
        "task_success_contract"
    ]
    assert preparation["policy_run_configuration"][
        "task_success_contract_digest"
    ] == request["task_success_contract_digest"]
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
    assert activation["task_success_contract"] == request["task_success_contract"]
    assert activation["task_success_contract_digest"] == request[
        "task_success_contract_digest"
    ]
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


def test_launch_request_refuses_task_success_contract_tamper(tmp_path: Path) -> None:
    _profile, request = _profile_and_request(tmp_path)
    request["task_success_contract"]["criteria"]["orientation"]["mode"] = (
        "required"
    )
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    assert any(
        "rigid_task_success_contract_digest_mismatch" in blocker
        for blocker in validate_launch_request(request)
    )


def test_launch_request_refuses_unconfirmed_agent_success_contract(
    tmp_path: Path,
) -> None:
    _profile, request = _profile_and_request(tmp_path)
    published = request["task_success_contract"]
    proposal = seal_rigid_task_success_contract(
        task_spec={},
        site_id=published["scope"]["site_id"],
        task_id=published["scope"]["task_id"],
        author_source="agent_proposal",
        author_id="agent:criteria-drafter",
        confirmation_status="proposal_only",
        criteria=published["criteria"],
    )
    request["task_success_contract"] = proposal
    request["task_success_contract_digest"] = proposal["contract_digest"]
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    assert any(
        "rigid_task_success_contract_unconfirmed" in blocker
        for blocker in validate_launch_request(request)
    )


def test_policy_canary_launch_request_rejects_unknown_fields(tmp_path: Path) -> None:
    _profile, request = _profile_and_request(tmp_path)
    request["unreviewed_grading_override"] = True
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )

    assert "policy_canary_launch_request_fields_invalid" in (
        validate_launch_request(request)
    )
