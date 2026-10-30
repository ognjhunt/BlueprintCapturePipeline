"""Hermetic G1 admission/rights/publication boundaries; no inference or GPU."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from blueprint_pipeline import beta_policy_screening_g1 as g1
from blueprint_pipeline import native_g1_beta_authority as rights
from blueprint_pipeline.native_g1_beta_authority_composer import compose_beta_authority
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog, seal_catalog
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan, validate_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_development_worker import _rights_review
from tests.test_beta_policy_screening_plan import admitted, scene, NOW

DP, PI, DP_NAV, PI_NAV = g1.CANDIDATES
SHA = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64


def source():
    return {"capture_id": "capture-current", "task_id": "task-current", "scene_intent_id": "intent-current",
            "task_context_digest": SHA, "scene_intent_digest": OTHER, "scene_revision_digest": SHA}


def authority(monkeypatch, *, candidate=DP):
    calls = []
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
        lambda source, *, site_id: calls.append((source, site_id)))
    value = compose_beta_authority(candidate_id=candidate, scene_plan_digest=SHA,
        source_packet_receipt_digest=SHA, task_success_contract_digest=SHA,
        checkpoint_digest=rights.CANDIDATE_DIGESTS[candidate], current_source=source(),
        site_id="site-current", registration_digest=OTHER)
    assert calls == [(source(), "site-current")]
    return value


def reseal(value, key="authority_digest"):
    value[key] = canonical_digest(value, digest_field=key)
    return value


def test_composed_current_authority_preserves_original_review_and_scope(monkeypatch):
    value = authority(monkeypatch)
    review = rights.reviewed_model_terms(DP)
    assert value["model_review"] == review
    assert review["receipt"]["scene_plan_digest"] != value["scene_plan_digest"]
    assert value["new_human_review_claimed"] is False
    assert value["model_terms_review_reused"] is True
    assert value["private_task_summary_permitted"] is True
    assert value["public_media_publication_authorized"] is False
    assert rights.validate_beta_authority(value, candidate_id=DP, scene_plan_digest=SHA,
        current_source=source(), task_success_contract_digest=SHA) == value
    # A legacy scene approval still cannot be repurposed by passing it directly.
    with pytest.raises(ValueError, match="g1_worker_rights_review_invalid"):
        _rights_review(review["receipt"], preflight={"candidate_id": DP, "scene_plan_digest": SHA,
            "inventory_file_sha256": rights.INVENTORY_SHA256})


@pytest.mark.parametrize("field,value", [
    ("candidate_id", PI), ("checkpoint_digest", OTHER), ("source_revision", "b" * 40),
    ("inventory_file_sha256", OTHER), ("robot_usd_sha256", OTHER),
    ("public_media_publication_authorized", True), ("model_redistribution_authorized", True),
    ("ranking_eligible", True), ("new_human_review_claimed", True), ("model_terms_review_reused", 1),
])
def test_resealed_identity_or_scope_changes_are_rejected(monkeypatch, field, value):
    row = authority(monkeypatch)
    row[field] = value
    reseal(row)
    with pytest.raises(ValueError, match="g1_beta_development_authority_invalid"):
        rights.validate_beta_authority(row, candidate_id=DP, scene_plan_digest=SHA)


def test_source_contract_sonic_and_review_identity_remain_exact(monkeypatch):
    original = authority(monkeypatch)
    for mutate in (
        lambda x: x["sonic"].update(encoder_sha256=OTHER),
        lambda x: x["model_review"]["receipt"].update(scene_plan_digest=SHA),
        lambda x: x["current_task_source"].update(capture_id="different-capture"),
        lambda x: x.update(task_success_contract_digest=OTHER),
    ):
        value = deepcopy(original)
        mutate(value)
        reseal(value)
        with pytest.raises(ValueError, match="g1_beta_development_authority_invalid"):
            rights.validate_beta_authority(value, candidate_id=DP, scene_plan_digest=SHA,
                current_source=source(), task_success_contract_digest=SHA)


def test_revoked_capture_prevents_authority_derivation(monkeypatch):
    def refused(*args, **kwargs):
        raise ValueError("beta_screening_current_capture_consent_required")
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source", refused)
    with pytest.raises(ValueError, match="current_capture_consent_required"):
        compose_beta_authority(candidate_id=DP, scene_plan_digest=SHA,
            source_packet_receipt_digest=SHA, task_success_contract_digest=SHA,
            checkpoint_digest=rights.CANDIDATE_DIGESTS[DP], current_source=source(),
            site_id="site-current", registration_digest=OTHER)


def test_worker_requires_actual_preflight_checkpoint_and_reviewed_sonic(monkeypatch):
    row = authority(monkeypatch)
    preflight = {"candidate_id": DP, "scene_plan_digest": SHA,
        "inventory_file_sha256": rights.INVENTORY_SHA256, "candidate_inventory_digest": rights.CANDIDATE_DIGESTS[DP],
        "sonic_encoder": {"sha256": rights.SONIC["encoder_sha256"]},
        "sonic_decoder": {"sha256": rights.SONIC["decoder_sha256"]}}
    assert _rights_review(row, preflight=preflight) == row
    preflight["sonic_encoder"]["sha256"] = OTHER
    with pytest.raises(ValueError, match="authority_runtime_mismatch"):
        _rights_review(row, preflight=preflight)


def g1_plan(*, maximum="4.00"):
    catalog = load_default_catalog()
    catalog = seal_catalog({**catalog, "candidates": [r for r in catalog["candidates"] if r["candidate_id"] in g1.CANDIDATES]})
    bindings, quotes = {}, {}
    for candidate in catalog["candidates"]:
        binding, quote = admitted(candidate, maximum=maximum)
        binding["backend"] = g1.BACKEND
        binding["execution_spec"]["backend"] = g1.BACKEND
        binding["execution_spec"]["minimum_gpu_memory_gb"] = 48
        binding["admission_digest"] = canonical_digest(binding["execution_spec"])
        bindings[candidate["candidate_id"]] = binding
        quotes[candidate["candidate_id"]] = quote
    return build_screening_plan(catalog, scene(), campaign_id="g1-screening", now=NOW,
        capabilities={"candidate_bindings": bindings}, cost_quotes=quotes)


def test_four_candidates_have_individual_budget_outcomes_and_explicit_frozen_resets():
    plan = g1_plan()
    assert len(plan["candidates"]) == 4
    assert [r["status"] for r in plan["candidates"]] == ["selected"] * 3 + ["deferred"]
    assert plan["maximum_planned_cost_usd"] == "12.00"
    assert plan["candidates"][-1]["blockers"] == ["site_screening_budget_insufficient"]
    for row in plan["candidates"][:3]:
        assert row["execution_evidence_kind"] == "learned_policy"
        assert {c["family"] for c in row["scenario_cells"]} == {"canonical_anchor"}
        assert all(c["resolved_scenario"]["parameters"] == {} for c in row["scenario_cells"])
        assert [c["seed"] for c in row["scenario_cells"]] == [c["seed"] for c in plan["scenario_cells"]]
    assert validate_screening_plan(plan) == plan
    schema = json.loads((g1.REPOSITORY / "docs/schemas/beta_policy_screening_plan.v1.schema.json").read_text())
    Draft202012Validator(schema).validate(plan)


@pytest.mark.parametrize("cell_id", ["screening./../../escape", "screening.\\..\\escape", "screening...", "screening." + "x" * 201])
def test_frozen_cell_cannot_escape_output_namespace(cell_id):
    cell = g1_plan()["candidates"][0]["scenario_cells"][0]
    cell["cell_id"] = cell_id
    with pytest.raises(ValueError, match="frozen_cell_invalid"):
        rights.validate_frozen_cell(cell)


def test_no_registration_retains_each_g1_reason_without_displacing_existing_candidates():
    catalog = load_default_catalog()
    value = build_screening_plan(catalog, scene(), campaign_id="unavailable", now=NOW,
        capabilities={"candidate_blockers": {key: ["g1_beta_scene_packet_missing"] for key in g1.CANDIDATES}})
    rows = {r["candidate_id"]: r for r in value["candidates"]}
    assert set(rows) == {candidate["candidate_id"] for candidate in catalog["candidates"]}
    for candidate in g1.CANDIDATES:
        assert rows[candidate]["status"] == "unavailable"
        assert "g1_beta_scene_packet_missing" in rows[candidate]["blockers"]
    assert {"pi05_droid", "groot_n17_droid"} <= rows.keys()


def test_corrected_g1_registration_changes_campaign_inventory_identity():
    catalog = load_default_catalog()
    first = g1.catalog_for_profile({"screening_candidates": {DP: {"native_g1": {"scene_packet": {"sha256": SHA}}}}}, catalog)
    second = g1.catalog_for_profile({"screening_candidates": {DP: {"native_g1": {"scene_packet": {"sha256": OTHER}}}}}, catalog)
    assert first["catalog_digest"] != second["catalog_digest"]
    assert first["candidates"] == second["candidates"] == catalog["candidates"]


def session_result():
    """Synthetic contract fixture, never retained as a real simulator result."""
    plan = g1_plan()
    candidate = plan["candidates"][0]
    episodes = []
    for cell in candidate["scenario_cells"]:
        bound = {"candidate_id": candidate["candidate_id"], "scene_plan_digest": SHA}
        actions = [0.0] * 40
        actions[3] = actions[6] = 1.0
        query = {"query_index": 0, "returned_action_count": 1}
        loaded = reseal({"schema_version": "native_g1_checkpoint_load_observation.v1", "candidate_id": DP,
            "candidate_inventory_digest": rights.CANDIDATE_DIGESTS[DP], "source_revision": rights.SOURCE_REVISION,
            "policy_type": "diffusion", "loader_completed": True, "full_state_load_required": True,
            "log_sha256": SHA, "inference_observed": False}, "receipt_digest")
        checkpoint = reseal({"schema_version": "native_g1_policy_query_checkpoint.v1", **bound,
            "status": "policy_response_retained", "query": query, "action_chunk": [actions],
            "checkpoint_load_observation": loaded}, "checkpoint_digest")
        trace = reseal({"schema_version": "native_g1_shared_scene_episode_trace.v1", **bound,
            "seed": cell["seed"], "policy_query_count": 1, "scene_step_count": 1,
            "queries": [{**query, "checkpoint_digest": checkpoint["checkpoint_digest"]}],
            "steps": [{"step_index": 1}]}, "trace_digest")
        score = reseal({"schema_version": "adp_rigid_task_scoring.v2", "status": "scored",
            "task_success_contract_digest": candidate["execution_spec"]["task_success_contract_digest"],
            "task_succeeded": False, "outcome": "collision_or_containment_failure"}, "report_digest")
        flags = {"ranking_eligible": False, "physical_outcome_claimed": False}
        episode = reseal({"schema_version": "native_g1_built_scene_policy_episode.v1", **bound, **flags,
            "status": "development_only_scored_episode", "trace_digest": trace["trace_digest"],
            "evaluation_task_kind": "rigid_pick_place", "score": score}, "result_digest")
        lease = reseal({"schema_version": "native_g1_policy_server_lease.v1", **bound,
            "status": "server_ready_process_bound", "source_revision": rights.SOURCE_REVISION,
            "candidate_inventory_digest": rights.CANDIDATE_DIGESTS[DP],
            "checkpoint_bytes_verified": True, "listener_owner_verified": True,
            "loaded_checkpoint_identity_observed": True, "checkpoint_load_observation": loaded}, "receipt_digest")
        supervised = reseal({"schema_version": "native_g1_supervised_built_scene_episode.v1", **bound, **flags,
            "status": "completed_development_only", "server_lease_receipt": lease,
            "server_teardown": {"status": "child_exited"}, "episode_result_digest": episode["result_digest"]}, "result_digest")
        worker = reseal({"schema_version": "native_g1_development_worker_result.v1", **bound, **flags,
            "status": "completed_development_only", "supervised_episode": supervised,
            "teardown": {"environment": "closed", "simulator": "closed"}}, "result_digest")
        episodes.append({**cell, "scene_plan_digest": SHA, "status": "completed", "worker_result": worker,
                         "episode": episode, "trace": trace, "query_checkpoints": [checkpoint]})
    result = {"schema_version": g1.RESULT_SCHEMA, "candidate_id": DP, "plan_digest": plan["plan_digest"],
        "configuration_digest": candidate["configuration_digest"], "admission_digest": candidate["admission_digest"],
        "runtime_identity_digest": candidate["execution_spec"]["runtime_digest"], "source": plan["source"],
        "task_success_contract_digest": candidate["execution_spec"]["task_success_contract_digest"],
        "scenario_plan_digest": plan["scenario_plan_digest"], "claim_ceiling": "development_only",
        "episodes": episodes, "ranking_eligible": False, "qualification_permitted": False,
        "public_media_publication_authorized": False, "model_redistribution_authorized": False,
        "physical_outcome_claimed": False,
        "session_closeout": reseal({"provider_zero_confirmed": True, "staged_objects_deleted": True,
            "watchdog_cancelled": True, "scope": "exact_instance_and_authority_resource_name",
            "global_provider_zero_claimed": False}, "receipt_digest")}
    return plan, candidate, reseal(result, "result_digest")


def test_failed_native_task_is_learned_execution_without_success_or_media_publication():
    from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary, _publication_execution_evidence
    plan, candidate, result = session_result()
    summary = candidate_result_summary(result, plan=plan, candidate=candidate)
    assert summary["episodes_scored"] == summary["episodes_failed"] == 10
    assert summary["episodes_succeeded"] == 0 and summary["success_rate"] == 0
    assert summary["learned_policy_rollout_count"] == 10 and summary["candidate_policy_queried"] is True
    assert _publication_execution_evidence(candidate, summary)["action_source"] == "learned_policy"
    assert not {"trace", "query_checkpoints", "worker_result", "media_artifacts", "task_prompt"} & summary.keys()


@pytest.mark.parametrize("case", ["source", "cell", "runtime", "score", "missing_query", "closed_without_actions", "closeout", "public_media"])
def test_unbound_or_unwitnessed_native_results_never_become_scores(case):
    plan, candidate, result = session_result()
    row = result["episodes"][0]
    if case == "source":
        result["source"]["capture_id"] = "different"
        # A returned result must not mutate the trusted expected plan fixture.
        plan["source"] = deepcopy(source())
    elif case == "cell":
        row["seed"] += 1
    elif case == "runtime":
        result["runtime_identity_digest"] = OTHER
    elif case == "score":
        row["episode"]["score"]["task_succeeded"] = True
    elif case == "missing_query":
        row["query_checkpoints"] = []
    elif case == "closed_without_actions":
        row["query_checkpoints"][0]["action_chunk"] = []
        reseal(row["query_checkpoints"][0], "checkpoint_digest")
    elif case == "closeout":
        result["session_closeout"]["watchdog_cancelled"] = False
        reseal(result["session_closeout"], "receipt_digest")
    else:
        result["public_media_publication_authorized"] = True
    reseal(result, "result_digest")
    with pytest.raises(ValueError):
        g1.summarize_result(result, plan=plan, candidate=candidate)


def test_partial_returned_actions_survive_timeout_as_void_not_task_failure():
    plan, candidate, result = session_result()
    for row in result["episodes"]:
        row["status"] = "blocked"
        row["reason"] = "g1_beta_native_worker_deadline"
        row.pop("worker_result")
        row.pop("episode")
        row.pop("trace")
    reseal(result, "result_digest")
    summary = g1.summarize_result(result, plan=plan, candidate=candidate)
    assert summary["episodes_void"] == 10
    assert summary["episodes_scored"] == 0 and summary["success_rate"] is None
    assert summary["candidate_policy_queried"] is True and summary["learned_policy_rollout_count"] == 10
    assert {row["reason"] for row in summary["episode_outcomes"]} == {"g1_beta_native_worker_deadline"}
    for row in result["episodes"]:
        row["query_checkpoints"] = []
    reseal(result, "result_digest")
    summary = g1.summarize_result(result, plan=plan, candidate=candidate)
    assert summary["candidate_policy_queried"] is False and summary["learned_policy_rollout_count"] == 0


@pytest.mark.parametrize("mutation", ["missing", "wrong_checkpoint", "not_loaded"])
def test_partial_actions_without_exact_loaded_checkpoint_are_not_published(mutation):
    plan, candidate, result = session_result()
    row = result["episodes"][0]
    row["status"] = "blocked"
    retained = row["query_checkpoints"][0]
    if mutation == "missing":
        retained.pop("checkpoint_load_observation")
    else:
        observed = retained["checkpoint_load_observation"]
        if mutation == "wrong_checkpoint":
            observed["candidate_inventory_digest"] = rights.CANDIDATE_DIGESTS[PI]
        else:
            observed["loader_completed"] = False
        reseal(observed, "receipt_digest")
    reseal(retained, "checkpoint_digest")
    reseal(result, "result_digest")
    with pytest.raises(ValueError):
        g1.summarize_result(result, plan=plan, candidate=candidate)


def builder_inputs(tmp_path, monkeypatch):
    """Mock only existing file-closure verification, never task binding checks."""
    from blueprint_pipeline.adp_task_scoring import seal_rigid_task_success_contract
    from blueprint_pipeline.beta_policy_screening_launch import _record
    from tests.test_native_g1_embodiment import robot_plan
    from tests.test_task_evaluation_policy_canary_setup import _setup
    contract = seal_rigid_task_success_contract(task_spec={}, site_id="site-current", task_id="task-current",
        author_source="compatibility_default", author_id="blueprint:manipulation_strategy_defaults.v1",
        confirmation_status="confirmed", criteria=_setup()["task_success_contract"]["criteria"])
    task = {"task_success_contract": contract, "task_success_contract_digest": contract["contract_digest"],
            "control_frequency_hz": 50, "target_position_world_m": [0.1, 0.2, 0.3]}
    request = {"scene_id": "site-current", "task_id": "task-current", "task_joint_bindings": {},
        "task_state_binding": {"object": "actual-site-object"}, "appearance_variant": {},
        "assets": [{"semantic_role": "scene_collision", "source": {"root": "source"}, "filename": "room.usda"}]}
    base, native = tmp_path / "base", tmp_path / "native"
    base.mkdir()
    native.mkdir()
    receipts = {}
    for root in (base, native):
        receipts[root] = {"receipt_digest": SHA if root == base else OTHER, "arena_scene_plan_digest": SHA,
            "source_bindings": [{"semantic_role": "scene_collision", "staged_sha256": SHA, "staged_size_bytes": 20}]}
        (root / "native_task_arena_packet_receipt.v1.json").write_text(json.dumps(receipts[root]))
    (base / "native_task_arena_packet_request.v1.json").write_text(json.dumps(request))
    native_request = {**deepcopy(request), "g1_scene_derivation": {
        "source_packet_receipt_digest": SHA, "source_scene_plan_digest": SHA}}
    (native / "native_task_arena_packet_request.v1.json").write_text(json.dumps(native_request))
    robot = robot_plan(tmp_path)
    robot["usd_sha256"] = rights.ROBOT_USD_SHA256
    native_scene = reseal({"scene_id": "site-current", "task_id": "task-current", "task_kind": "rigid_pick_place",
        "task_spec": task, "robot": robot,
        "cameras": [{"role": "head", "policy_input": True, "intrinsics": {"width": 640, "height": 480}}]}, "plan_digest")
    (native / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(native_scene))
    publisher, runtime = tmp_path / "publisher.json", tmp_path / "runtime.json"
    publisher.write_text("{}")
    runtime.write_text("{}")
    monkeypatch.setattr("blueprint_pipeline.native_task_arena_bundle.verify_native_task_arena_packet",
        lambda root: (root, receipts[root], []))
    monkeypatch.setattr("blueprint_pipeline.native_g1_publisher_source_stage.verify_g1_publisher_source", lambda root: {})
    monkeypatch.setattr("blueprint_pipeline.native_task_runtime_source_packet.verify_native_task_runtime_source_packet",
        lambda path: {"runtime_profile": "unitree_g1", "redistribution_permitted": True})
    calls = []
    monkeypatch.setattr("blueprint_pipeline.beta_policy_screening_worker.require_current_source",
        lambda value, *, site_id: calls.append((value, site_id)))
    kwargs = {"candidate_id": DP, "registration": {"scene_packet": _record(native / "native_task_arena_packet_receipt.v1.json"),
        "publisher_source": _record(publisher), "runtime_source": _record(runtime)},
        "scene_plan": {"task_spec": deepcopy(task)}, "base_packet": _record(base / "native_task_arena_packet_receipt.v1.json"),
        "source": source(), "contract": contract, "site_id": "site-current"}
    return kwargs, native_scene, native_request, receipts[native], calls


def test_native_registration_builds_exact_dp_spec_without_pi_assets(tmp_path, monkeypatch):
    kwargs, _, _, _, calls = builder_inputs(tmp_path, monkeypatch)
    spec = g1.candidate_execution_spec(**kwargs)
    assert g1.validate_execution_spec(spec) == spec
    assert calls == [(source(), "site-current")]
    assert spec["rights_authority"]["auxiliary_assets"] == [rights.DP_AUXILIARY]
    assert set(spec["native_g1"]) == {"scene_packet", "publisher_source", "runtime_source", "model_terms_review"}
    assert spec["runtime_identity"]["interface"]["action_horizon"] == 20


@pytest.mark.parametrize("mutation,reason", [
    ("source_receipt", "source_scene_derivation_mismatch"),
    ("source_asset", "source_scene_assets_mismatch"),
    ("source_task", "current_task_binding_mismatch"),
    ("missing_camera", "head_camera_binding_missing"),
    ("navigation_without_goal", "current_navigation_goal_authority_missing"),
])
def test_registration_cannot_repurpose_an_unrelated_g1_scene(tmp_path, monkeypatch, mutation, reason):
    kwargs, native, request, receipt, calls = builder_inputs(tmp_path, monkeypatch)
    if mutation == "source_receipt":
        request["g1_scene_derivation"]["source_packet_receipt_digest"] = OTHER
    elif mutation == "source_asset":
        receipt["source_bindings"][0]["staged_sha256"] = OTHER
    elif mutation == "source_task":
        native["task_spec"]["target_position_world_m"][0] = 20
    elif mutation == "missing_camera":
        native["cameras"] = []
    else:
        kwargs["candidate_id"] = DP_NAV
    (tmp_path / "native/native_task_arena_packet_request.v1.json").write_text(json.dumps(request))
    (tmp_path / "native/native_task_arena_scene_plan.v1.json").write_text(json.dumps(reseal(native, "plan_digest")))
    with pytest.raises(ValueError, match=reason):
        g1.candidate_execution_spec(**kwargs)
    assert calls == []


def test_cell_derivation_keeps_geometry_contract_and_original_review_frozen(monkeypatch):
    base_authority = authority(monkeypatch)
    contract = reseal({"scenario": {"seed": 1}, "physics": {"rate_hz": 200}}, "contract_digest")
    base = reseal({"scenario": {"seed": 1}, "runtime_contract_digest": contract["contract_digest"],
        "task_spec": {"task_success_contract_digest": SHA}, "geometry": {"target": [1, 2, 3]}}, "plan_digest")
    base_authority["scene_plan_digest"] = base["plan_digest"]
    reseal(base_authority)
    cell = g1_plan()["candidates"][0]["scenario_cells"][0]
    context = {"schema_version": "adp009d_scenario_instance.v1", "cell_id": cell["cell_id"], "seed": cell["seed"],
        "policy_neutral": True, "caller_asserted_success": False, "learned_policy_outcomes_consulted": False,
        "factor_records": [], "resolved_parameters": {}, "beta_cell_spec_digest": cell["cell_spec_digest"],
        "scenario_scope": "native_frozen_reset_seeds"}
    scenario = {"context_kind": "evaluation_cell", "cell_id": cell["cell_id"], "seed": cell["seed"],
        "instance_digest": canonical_digest(context, digest_field="instance_digest")}
    derived_contract = reseal({**contract, "scenario": scenario}, "contract_digest")
    derived = reseal({**deepcopy(base), "scenario": scenario,
        "runtime_contract_digest": derived_contract["contract_digest"]}, "plan_digest")
    cell_authority = rights.derive_cell_authority(base_authority=base_authority, cell=cell,
        scene_plan_digest=derived["plan_digest"], packet_receipt_digest=OTHER)
    assert g1.validate_cell_binding(base, derived, cell, cell_authority, base_contract=contract) == derived
    assert cell_authority["base_authority"]["model_review"] == rights.reviewed_model_terms(DP)
    changed = deepcopy(derived)
    changed["geometry"]["target"][0] += 1
    reseal(changed, "plan_digest")
    cell_authority["scene_plan_digest"] = changed["plan_digest"]
    reseal(cell_authority)
    with pytest.raises(ValueError, match="changed_frozen_scene"):
        g1.validate_cell_binding(base, changed, cell, cell_authority, base_contract=contract)


def test_navigation_reset_derivation_preserves_original_human_goal_decision(monkeypatch):
    from blueprint_pipeline.native_g1_beta_navigation import derive_navigation_authority, validate_navigation_authority
    from blueprint_pipeline.native_g1_navigation_goal import seal_g1_navigation_goal_authority, validate_g1_navigation_goal_authority
    from tests.test_native_g1_navigation_goal import _authority_plan
    base = _authority_plan()
    contract = reseal({"scenario": {"seed": 1}, "physics": {"rate_hz": 200}}, "contract_digest")
    base.update(scenario={"seed": 1}, runtime_contract_digest=contract["contract_digest"])
    reseal(base, "plan_digest")
    goal_authority = seal_g1_navigation_goal_authority(plan=base, confirmed_by_team_id="team-a", human_reviewer="owner-a")
    model_authority = authority(monkeypatch, candidate=DP_NAV)
    model_authority.update(scene_plan_digest=base["plan_digest"],
        task_success_contract_digest=base["task_spec"]["task_success_contract_digest"])
    reseal(model_authority)
    cell = g1_plan()["candidates"][2]["scenario_cells"][0]
    context = {"schema_version": "adp009d_scenario_instance.v1", "cell_id": cell["cell_id"], "seed": cell["seed"],
        "policy_neutral": True, "caller_asserted_success": False, "learned_policy_outcomes_consulted": False,
        "factor_records": [], "resolved_parameters": {}, "beta_cell_spec_digest": cell["cell_spec_digest"],
        "scenario_scope": "native_frozen_reset_seeds"}
    scenario = {"context_kind": "evaluation_cell", "cell_id": cell["cell_id"], "seed": cell["seed"],
        "instance_digest": canonical_digest(context, digest_field="instance_digest")}
    derived_contract = reseal({**contract, "scenario": scenario}, "contract_digest")
    derived = reseal({**deepcopy(base), "scenario": scenario,
        "runtime_contract_digest": derived_contract["contract_digest"]}, "plan_digest")
    cell_authority = rights.derive_cell_authority(base_authority=model_authority, cell=cell,
        scene_plan_digest=derived["plan_digest"], packet_receipt_digest=OTHER)
    wrapped = derive_navigation_authority(goal_authority, base, derived, cell_authority, base_contract=contract)
    assert wrapped["base_authority"] == goal_authority and wrapped["new_human_review_claimed"] is False
    assert validate_navigation_authority(wrapped, plan=derived, cell_authority=cell_authority) == wrapped
    with pytest.raises(ValueError, match="goal_authority_invalid"):
        validate_g1_navigation_goal_authority(goal_authority, plan=derived)
    changed = deepcopy(derived)
    changed["task_spec"]["g1_navigation_goal"]["center_world_m"][0] += 1
    reseal(changed, "plan_digest")
    with pytest.raises(ValueError, match="beta_navigation_authority_invalid"):
        validate_navigation_authority(wrapped, plan=changed, cell_authority=cell_authority)


def test_scored_isaac_close_recovery_requires_retained_preclose_and_observed_zero_exit():
    plan, candidate, result = session_result()
    row = result["episodes"][0]
    worker = row["worker_result"]
    preclose = {k: v for k, v in worker.items() if k != "result_digest"}
    preclose.update(schema_version="native_g1_development_worker_preclose.v1", status="awaiting_simulator_close",
        teardown={"environment": "closed", "simulator": "close_requested"}, blocker=None)
    reseal(preclose, "preclose_digest")
    row["worker_preclose"] = preclose
    worker.update(teardown={"environment": "closed", "simulator": "process_exited_after_close_request"}, blocker=None,
        process_exit_evidence={"returncode": 0, "preclose_digest": preclose["preclose_digest"]})
    reseal(worker, "result_digest")
    reseal(result, "result_digest")
    assert g1.summarize_result(result, plan=plan, candidate=candidate)["episodes_failed"] == 10
    for change in ("wrong_exit", "missing_preclose", "changed_preclose"):
        invalid = deepcopy(result)
        episode = invalid["episodes"][0]
        if change == "wrong_exit":
            episode["worker_result"]["process_exit_evidence"]["returncode"] = 1
            reseal(episode["worker_result"], "result_digest")
        elif change == "missing_preclose":
            episode.pop("worker_preclose")
        else:
            episode["worker_preclose"]["candidate_id"] = PI
            reseal(episode["worker_preclose"], "preclose_digest")
        reseal(invalid, "result_digest")
        with pytest.raises(ValueError):
            g1.summarize_result(invalid, plan=plan, candidate=candidate)
