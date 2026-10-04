# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_inventory_seed.py
#   src/blueprint_pipeline/task_evaluation_scene_preparation_lineage.py
#   src/blueprint_pipeline/task_evaluation_scene_source_attempt_lineage.py
"""Exact retained downstream joins expose gaps without granting authority."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_scene_inventory_history import api, event, fixture as history_fixture, pair, project, ref, refuses, seal

COMMIT, DIGEST = "a" * 40, "sha256:" + "b" * 64


def test_recipe_stage_cache_candidate_requires_same_selected_result_and_typed_parent():
    from blueprint_pipeline.task_evaluation_scene_inventory_seed import _result_references
    roots = {'preparation_input_root': '/inputs', 'content_store_root': '/cache'}
    recipe = {'uri': 's3://inputs/recipe', 'digest': 'sha256:' + 'a' * 64, 'size_bytes': 12}
    stage = {'uri': 's3://inputs/stage', 'digest': 'sha256:' + 'c' * 64, 'size_bytes': 13}
    parent_row = {'contract_path': 'construction.recipe', **recipe,
                  'materialized_path': '/inputs/prep-1/' + 'a' * 64,
                  'content_addressed_reuse': False, 'full_byte_service_account_readback_passed': True}
    stage_row = {'contract_path': 'construction.recipe.stage_sequence.0.configuration', **stage,
                 'materialized_path': '/inputs/prep-1/construction-stage-configurations/' + 'c' * 64,
                 'content_addressed_reuse': False, 'full_byte_service_account_readback_passed': True}
    provenance = {'role': 'preparation_results', 'path': '/queue/results/prep-1.json',
                  'sha256': 'sha256:' + 'e' * 64, 'size_bytes': 1}

    def run(*, typed=True, readback=True, stage_readback=True, stage_path=True):
        parent = {**parent_row, 'full_byte_service_account_readback_passed': readback}
        child = {**stage_row, 'full_byte_service_account_readback_passed': stage_readback}
        if not stage_path:
            child['materialized_path'] = '/inputs/prep-1/unbound/' + 'c' * 64
        context = {'typed': {'construction.recipe': recipe} if typed else {},
                   'link': {'preparation_id': 'prep-1', 'request_digest': 'sha256:' + 'f' * 64},
                   'sources': [provenance]}
        budget = {'result': 0, 'members': {}, 'edges': {}, 'provenances': {}}
        members, shared, deferred, reasons = [], {}, [], set()
        missing = {'count': 0, 'projections': []}
        _result_references({'references': [parent, child]}, provenance, context, roots,
                           reasons, members, shared, deferred, budget, True, missing)
        return shared, deferred, members

    shared, deferred, members = run()
    assert stage['digest'] in shared
    assert shared[stage['digest']]['path'] == '/cache/' + 'c' * 64
    assert any(row['contract_path'] == stage_row['contract_path'] and
               row['reason'] == 'deferred_parent_reference_proof' for row in deferred)
    assert any(row['path'] == stage_row['materialized_path'] and
               row['binding_strength'] == 'recipe_parent_pending' for row in members)
    assert stage['digest'] not in run(typed=False)[0]
    assert stage['digest'] not in run(readback=False)[0]
    assert stage['digest'] not in run(stage_readback=False)[0]
    assert stage['digest'] not in run(stage_path=False)[0]


def fixture(*, activation=False, configuration=False, preparation="prep-1", status="inputs_materialized_awaiting_construction_adapter",
            request_edit=None, result_edit=None, progression_edit=None, activation_edit=None):
    args = history_fixture()
    intent = json.loads(args["records"]["intent"][1])
    roots, records = args["roots"], args["records"]
    request = {"schema_version": "task_evaluation_launch_preparation_request.v1", "run_mode": "scene_configuration",
        "preparation_id": preparation, "team_namespace": "team-1", "expected_production_commit": COMMIT, "run_id": "run-1",
        "scene": {"identity": {"id": "scene-1"}}, "task": {"identity": {"id": "task-1"},
        "artifact": {"uri": "https://example.test/input?q=1", "digest": DIGEST, "size_bytes": 1}},
        "scene_intent_digest": intent["intent_digest"], "execution_adapter": {"runtime_source_bundle": {"digest": DIGEST}}}
    if request_edit:
        request_edit(request)
    digest = canonical_digest(request)
    filename = preparation + "-" + digest[7:] + ".json"
    link = {"schema_version": "task_evaluation_scene_preparation_link.v1", "intent_id": args["intent_id"],
        "intent_digest": intent["intent_digest"], "preparation_id": preparation, "request_digest": digest,
        "expected_production_commit": COMMIT, "team_namespace": "team-1", "scene_id": "scene-1", "task_id": "task-1",
        "result_filename": filename}
    base = roots["intent_root"] + "/" + args["intent_id"]
    records["preparation_links"] = [pair(base + "/preparations/" + digest[7:] + ".json", seal(link, "link_digest"))]
    if activation:
        attempt_id = "scene-configuration-" + digest[7:31]
        attempt = {"schema_version": "task_evaluation_scene_attempt.v1", "attempt_id": attempt_id,
            "intent_id": args["intent_id"], "intent_digest": intent["intent_digest"], "source_commit": COMMIT,
            "input_digest": digest, "runtime_digest": DIGEST, "provider": "vast", "maximum_spend_usd": 1.0}
        records["attempts"] = [pair(base + "/attempts/" + attempt_id + ".json", seal(attempt, "attempt_digest", cross=True))]
        link["scene_configuration_attempt"] = ref(records["attempts"][0])
        records["preparation_links"].append(pair(base + "/preparations/" + digest[7:] + ".activation.json", seal(link, "link_digest")))
    envelope = {"schema_version": "task_evaluation_launch_preparation_envelope.v1", "request": request, "request_digest": digest}
    queue_state = status if status in {"blocked", "awaiting_capacity"} else "completed"
    records["preparation_envelopes"] = [pair(roots["preparation_queue_root"] + "/" + queue_state + "/" + filename, seal(envelope, "envelope_digest"))]
    artifact = request["task"]["artifact"]
    result = {"schema_version": "task_evaluation_launch_preparation_result.v1", "status": status, "source_commit": COMMIT,
        "preparation_id": preparation, "team_namespace": "team-1", "run_id": "run-1", "reference_count": 1,
        "references": [{"contract_path": "task.artifact", **artifact,
            "materialized_path": roots["preparation_input_root"] + "/" + preparation + "/" + artifact["digest"][7:],
            "content_addressed_reuse": False, "full_byte_service_account_readback_passed": True}],
        "provider_mutation_performed": False, "catalog_mutation_performed": False, "paid_execution_requested": False}
    if result_edit:
        result_edit(result)
    result = seal(result, "result_digest")
    records["preparation_results"] = [pair(roots["preparation_queue_root"] + "/results/" + filename, result)]
    if configuration:
        activation_request = {"schema_version": "task_evaluation_launch_activation_request.v1",
            "lane": "task_evaluation_scene_configuration", "activation_id": "activation-1", "team_namespace": "team-1",
            "expected_production_commit": COMMIT, "preparation": {"preparation_id": preparation, "request_digest": digest,
                                                                  "result_digest": result["result_digest"]}}
        if activation_edit:
            activation_edit(activation_request)
        activation_digest = canonical_digest(activation_request)
        readable = activation_request["activation_id"] + "-" + activation_digest[7:] + ".json"
        activation_filename = readable if len(readable.encode()) <= 255 else (
            "activation-" + hashlib.sha256(activation_request["activation_id"].encode()).hexdigest() + "-" + activation_digest[7:] + ".json")
        activation_envelope = {"schema_version": "task_evaluation_launch_activation_envelope.v1", "request": activation_request,
            "request_digest": activation_digest, "provider_mutation_performed_inside_intake": False,
            "catalog_mutation_performed_inside_intake": False, "standing_authorization_published_inside_intake": False,
            "paid_execution_requested": False}
        records["activation_envelopes"] = [pair(roots["activation_queue_root"] + "/pending/" + activation_filename,
                                                  seal(activation_envelope, "envelope_digest"))]
        progression = {"schema_version": "task_evaluation_scene_configuration_activation_progression.v1",
            "status": "scene_configuration_activation_queued", "preparation_id": preparation, "run_id": "run-1",
            "team_namespace": "team-1", "scene_id": "scene-1", "task_id": "task-1", "expected_production_commit": COMMIT,
            "activation_id": activation_request["activation_id"], "intent_digest": "sha256:" + "c" * 64,
            "preparation_result_digest": result["result_digest"], "preparation_request_digest": digest,
            "activation_request_digest": activation_digest, "provider_mutation_performed": False, "paid_execution_requested": False}
        if progression_edit:
            progression_edit(progression)
        records["configuration_progressions"] = [pair(roots["configuration_progression_root"] + "/scene-configuration-activations/" + preparation + "/activation_progression.json",
                                                        seal(progression, "progression_digest"))]
    project(args, event(args, {"preparation_link": ref(records["preparation_links"][0]),
                               "preparation_result": ref(records["preparation_results"][0])}))
    return args


def replace_record(args, role, edit, field, *, position=0, cross=False):
    path, raw = args["records"][role][position]
    value = json.loads(raw)
    edit(value)
    args["records"][role][position] = pair(path, seal(value, field, cross=cross))


def test_exact_preparation_request_result_and_configuration_join_retained_members():
    args = fixture(configuration=True)
    result = api().join_retained_scene_inventory_seed(**args)
    assert result["status"] == "joined_supplied_seed"
    assert {row["kind"] for row in result["members"]} == {
        "preparation_workspace", "preparation_projected_file", "configuration_progression_workspace"}
    projected = next(row for row in result["members"] if row["kind"] == "preparation_projected_file")
    assert projected["binding_strength"] == "request_typed_reference" and projected["contract_paths"] == ["task.artifact"]
    assert projected["receipt_size_bytes"] == 1 and projected["measured_bytes"] is None
    assert result["shared_cache_references"][0]["exclusive_scene_membership"] is False
    assert not result["cleanup_authorized"] and not result["publication_readback_checked"]


@pytest.mark.parametrize("missing,reason", [("preparation_envelopes", "preparation_envelope_missing"),
                                          ("preparation_results", "preparation_result_missing"),
                                          ("activation_envelopes", "configuration_join_unresolved")])
def test_missing_dependency_is_explicit_kept_not_not_created(missing, reason):
    args = fixture(configuration=missing == "activation_envelopes")
    args["records"][missing] = []
    result = api().join_retained_scene_inventory_seed(**args)
    assert reason in result["unresolved_reasons"] and result["status"] == "kept_unresolved"
    if missing == "preparation_envelopes":
        assert result["members"] == []


def test_supplied_envelope_without_link_refuses_instead_of_guessing_owner():
    args = fixture()
    args["records"]["preparation_links"] = []
    refuses(args)


@pytest.mark.parametrize("activation", [False, True])
def test_queue_state_duplicates_refuse_even_identical_bytes(activation):
    args = fixture(configuration=activation)
    role = "activation_envelopes" if activation else "preparation_envelopes"
    path, raw = args["records"][role][0]
    args["records"][role].append((path.replace("/pending/", "/processing/").replace("/completed/", "/processing/"), raw))
    refuses(args)


def test_absent_activation_attempt_keeps_but_malformed_available_edges_refuse():
    args = fixture(activation=True)
    args["records"]["attempts"] = []
    result = api().join_retained_scene_inventory_seed(**args)
    assert "configuration_attempt_missing" in result["unresolved_reasons"] and result["members"] == []
    args = fixture(activation=True)
    args["records"]["preparation_envelopes"] = []
    replace_record(args, "attempts", lambda row: row.update(provider="wrong"), "attempt_digest", cross=True)
    # Rebind the link's raw reference so the intended provider edge is tested.
    replace_record(args, "preparation_links", lambda row: row.update(scene_configuration_attempt=ref(args["records"]["attempts"][0])), "link_digest", position=1)
    refuses(args)


def test_available_bad_result_cannot_hide_behind_missing_envelope():
    args = fixture()
    args["records"]["preparation_envelopes"] = []
    replace_record(args, "preparation_results", lambda row: row.update(preparation_id="foreign"), "result_digest")
    refuses(args)


@pytest.mark.parametrize("field,value", [("uri", "s3://foreign/key"), ("digest", "sha256:" + "c" * 64), ("size_bytes", 2)])
def test_fully_resealed_request_owned_result_tuple_mismatch_refuses(field, value):
    def edit(result):
        result["references"][0][field] = value
        if field == "digest":
            result["references"][0]["materialized_path"] = "/retained/inputs/prep-1/" + value[7:]
    refuses(fixture(configuration=True, result_edit=edit))


@pytest.mark.parametrize("field,value", [("uri", "b2://bucket/key"), ("uri", "r2://bucket/key"),
    ("uri", "https://example.test/a b"), ("digest", "sha256:" + "B" * 64),
    ("size_bytes", 0), ("size_bytes", -1), ("size_bytes", True), ("size_bytes", 1.5)])
def test_request_typed_reference_matches_exact_schema(field, value):
    refuses(fixture(request_edit=lambda request: request["task"]["artifact"].update({field: value})))


def test_transitive_zero_byte_row_is_deferred_receipt_only():
    def edit(result):
        extra = dict(result["references"][0], contract_path="construction.recipe.stage_sequence.0.configuration",
                     uri="s3://bucket/empty", digest="sha256:" + "c" * 64, size_bytes=0,
                     materialized_path="/retained/inputs/prep-1/construction-stage-configurations/" + "c" * 64)
        result["references"].append(extra)
        result["reference_count"] = 2
    result = api().join_retained_scene_inventory_seed(**fixture(result_edit=edit))
    assert len(result["deferred_result_references"]) == 1
    assert result["deferred_result_references"][0]["binding_strength"] == "result_receipt_only"
    assert len(result["shared_cache_references"]) == 1
    assert sum(row["kind"] == "preparation_projected_file" for row in result["members"]) == 1


@pytest.mark.parametrize("path", ["/retained/inputs/" + "b" * 64, "/retained/inputs/prep-2/" + "b" * 64,
                                  "/retained/inputs/prep-1/wrong", "/retained/inputs/prep-1/../" + "b" * 64])
def test_result_materialization_path_cannot_rebind_owned_directory(path):
    refuses(fixture(result_edit=lambda result: result["references"][0].update(materialized_path=path)))


def test_missing_request_projection_is_unknown_without_invented_file():
    result = api().join_retained_scene_inventory_seed(**fixture(result_edit=lambda row: row.update(references=[], reference_count=0)))
    assert "request_projection_missing" in result["unresolved_reasons"]
    assert not result["shared_cache_references"] and not any(row["kind"] == "preparation_projected_file" for row in result["members"])


@pytest.mark.parametrize("status", ["blocked", "awaiting_capacity", "unknown_future_status"])
def test_minimal_nonmaterialized_results_keep_without_download_members(status):
    def edit(row):
        for key in ("references", "reference_count", "team_namespace", "run_id"):
            row.pop(key)
        row["blockers"] = ["capacity_unknown"]
    result = api().join_retained_scene_inventory_seed(**fixture(status=status, result_edit=edit))
    assert "preparation_result_scope_unproven" in result["unresolved_reasons"]
    assert not any(row["kind"] == "preparation_projected_file" for row in result["members"])


def test_exact_other_release_blocked_exception_is_descriptive_only():
    def edit(row):
        row.update(source_commit="d" * 40, blockers=["launch_preparation_worker_source_commit_mismatch"])
    result = api().join_retained_scene_inventory_seed(**fixture(status="blocked", result_edit=edit))
    assert "preparation_result_scope_unproven" in result["unresolved_reasons"]
    refuses(fixture(result_edit=edit))


def test_valid_192_character_activation_id_uses_hashed_filename():
    args = fixture(configuration=True, activation_edit=lambda row: row.update(activation_id="a" * 192))
    assert "/activation-" in args["records"]["activation_envelopes"][0][0]
    result = api().join_retained_scene_inventory_seed(**args)
    assert any(row["kind"] == "configuration_progression_workspace" for row in result["members"])
    refuses(fixture(configuration=True, activation_edit=lambda row: row.update(activation_id="a" * 193)))


@pytest.mark.parametrize("field", ["provider_mutation_performed", "paid_execution_requested"])
@pytest.mark.parametrize("value", [True, 0, None])
def test_configuration_scope_flags_must_be_literal_false(field, value):
    refuses(fixture(configuration=True, progression_edit=lambda row: row.update({field: value})))


@pytest.mark.parametrize("field,value", [("preparation_id", "foreign"), ("preparation_request_digest", "sha256:" + "f" * 64),
    ("preparation_result_digest", "sha256:" + "f" * 64), ("team_namespace", "other"), ("scene_id", "other"),
    ("task_id", "other"), ("expected_production_commit", "e" * 40), ("run_id", "other"), ("activation_id", "other")])
def test_resealed_configuration_tuple_mismatch_refuses(field, value):
    refuses(fixture(configuration=True, progression_edit=lambda row: row.update({field: value})))


def test_mutable_result_versions_preserve_old_bytes_and_do_not_select_latest():
    args = fixture()
    path, raw = args["records"]["preparation_results"][0]
    changed = json.loads(raw)
    changed["status"] = "blocked"
    changed["blockers"] = ["historical_pause"]
    old = pair(path, seal(changed, "result_digest"))
    args["records"]["preparation_results"].append(old)
    event(args, {"preparation_result": ref(old)})
    result = api().join_retained_scene_inventory_seed(**args)
    rows = [row for row in result["obligations"] if row["role"] == "preparation_result"]
    assert len(rows) == 2 and all(row["status"] == "matched_retained_bytes" for row in rows)


def test_complete_groups_reuse_unchanged_preparation_join(monkeypatch):
    args = fixture(activation=True)
    module = api()
    seen = []
    original = module.retained.join_scene_preparation_lineage
    def spy(**kwargs):
        seen.append(original(**kwargs))
        return seen[-1]
    monkeypatch.setattr(module.retained, "join_scene_preparation_lineage", spy)
    result = module.join_retained_scene_inventory_seed(**args)
    assert len(seen) == 1 and seen[0]["preparation_count"] == 1
    workspace = next(row for row in result["members"] if row["kind"] == "preparation_workspace")
    assert workspace["path"] == seen[0]["preparations"][0]["workspace_path"]


def test_typed_reference_index_bounds_and_path_collision_refuse(monkeypatch):
    for limit in ("MAX_NODES", "MAX_DEPTH", "MAX_REFERENCES"):
        with monkeypatch.context() as context:
            context.setattr(api(), limit, 0)
            refuses(fixture())
    args = fixture(request_edit=lambda request: request.update({"task.artifact": copy.deepcopy(request["task"]["artifact"])}))
    refuses(args)


def test_two_materialized_result_versions_keep_both_raw_provenances():
    args = fixture()
    path, raw = args["records"]["preparation_results"][0]
    value = json.loads(raw)
    value["references"][0]["content_addressed_reuse"] = True
    version = pair(path, seal(value, "result_digest"))
    args["records"]["preparation_results"].append(version)
    result = api().join_retained_scene_inventory_seed(**args)
    member = next(row for row in result["members"] if row["kind"] == "preparation_projected_file")
    result_sources = [row for row in member["source_provenance"] if row["role"] == "preparation_results"]
    assert {row["sha256"] for row in result_sources} == {ref((path, raw))["sha256"], ref(version)["sha256"]}


def test_many_contract_edges_deduplicate_members_without_losing_request_edges():
    def request_edit(request):
        request["task"]["aliases"] = [copy.deepcopy(request["task"]["artifact"]) for _ in range(80)]
    def result_edit(result):
        result["references"].extend(dict(result["references"][0], contract_path=f"task.aliases.{number}") for number in range(80))
        result["reference_count"] = 81
    result = api().join_retained_scene_inventory_seed(**fixture(request_edit=request_edit, result_edit=result_edit))
    members = [row for row in result["members"] if row["kind"] == "preparation_projected_file"]
    assert len(members) == len(result["shared_cache_references"]) == 1
    assert len(members[0]["contract_paths"]) == 81


def test_missing_flags_refuse_configuration_even_with_resealed_proof():
    for key in ("provider_mutation_performed", "paid_execution_requested"):
        refuses(fixture(configuration=True, progression_edit=lambda row: row.pop(key)))


def test_absent_envelope_obligation_never_fabricates_raw_identity_or_queue_state():
    args = fixture()
    args["records"]["preparation_envelopes"] = []
    result = api().join_retained_scene_inventory_seed(**args)
    row = next(row for row in result["preparation_join_obligations"] if row["role"] == "preparation_envelope")
    assert row["role"] == "preparation_envelope" and row["reason"] == "preparation_envelope_missing"
    assert row["expected_path"] is None and row["expected_root"] == args["roots"]["preparation_queue_root"]
    assert "sha256" not in row and "size_bytes" not in row and "pending" in row["expected_states"]
    assert row["status"] == "kept_unresolved" and row["source_provenance"]


def test_missing_activation_attempt_retains_original_raw_reference():
    args = fixture(activation=True)
    exact = json.loads(args["records"]["preparation_links"][1][1])["scene_configuration_attempt"]
    args["records"]["attempts"] = []
    result = api().join_retained_scene_inventory_seed(**args)
    row = next(row for row in result["obligations"] if row["role"] == "attempt")
    assert all(row[field] == exact[field] for field in ("path", "sha256", "size_bytes"))
    assert row["status"] == "kept_unresolved"


def test_missing_request_projection_preserves_exact_typed_edge_not_presence():
    result = api().join_retained_scene_inventory_seed(**fixture(result_edit=lambda row: row.update(references=[], reference_count=0)))
    row = result["request_projection_obligations"][0]
    assert row["contract_path"] == "task.artifact" and row["uri"] == "https://example.test/input?q=1"
    assert row["digest"] == DIGEST and row["size_bytes"] == 1
    assert row["reason"] == "request_projection_missing" and row["source_provenance"]
    assert "materialized_path" not in row and row["status"] == "kept_unresolved"


def test_structural_obligations_share_occurrence_cap_before_dedup(monkeypatch):
    args = fixture()
    args["records"]["preparation_envelopes"] = []
    # Two event reference occurrences + one structural envelope obligation.
    monkeypatch.setattr(api(), "MAX_REFERENCES", 2)
    refuses(args, "references_limit")


@pytest.mark.parametrize("mode", ["no_envelope", "no_attempt", "blocked", "unknown"])
def test_extra_receipt_rows_survive_incomplete_and_nonmaterialized_join(mode):
    def edit(result):
        extra = dict(result["references"][0], contract_path="transitive.extra", digest="sha256:" + "c" * 64,
                     materialized_path="/retained/inputs/prep-1/extra/" + "c" * 64)
        result["references"].append(extra)
        result["reference_count"] = 2
        if mode in {"blocked", "unknown"}:
            result["status"] = "blocked" if mode == "blocked" else "future_status"
            result["blockers"] = ["held"]
    args = fixture(activation=mode == "no_attempt", result_edit=edit)
    if mode == "no_envelope":
        args["records"]["preparation_envelopes"] = []
    if mode == "no_attempt":
        args["records"]["attempts"] = []
    result = api().join_retained_scene_inventory_seed(**args)
    row = next(row for row in result["deferred_result_references"] if row["contract_path"] == "transitive.extra")
    assert row["binding_strength"] == "result_receipt_only" and row["reason"] == "deferred_parent_reference_proof"
    assert row["source_provenance"] and result["deferred_branch_count"] >= 1
    assert not any(member["path"] == row["materialized_path"] for member in result["members"])
    assert not any(shared["digest"] == row["digest"] for shared in result["shared_cache_references"])


def test_unreferenced_empty_result_versions_still_have_exact_retained_provenance():
    args = fixture(result_edit=lambda row: row.update(references=[], reference_count=0))
    path, raw = args["records"]["preparation_results"][0]
    older = json.loads(raw)
    older.update(status="blocked", blockers=["historical"])
    version = pair(path, seal(older, "result_digest"))
    args["records"]["preparation_results"].append(version)
    result = api().join_retained_scene_inventory_seed(**args)
    rows = [row for row in result["obligations"] if row["role"] == "preparation_result"]
    assert {row["sha256"] for row in rows} == {ref((path, raw))["sha256"], ref(version)["sha256"]}


def test_same_cache_path_with_conflicting_size_across_preparations_refuses():
    args = fixture()
    second = fixture(preparation="prep-2", request_edit=lambda row: row["task"]["artifact"].update(size_bytes=2, uri="s3://bucket/second"))
    for role in ("preparation_links", "preparation_envelopes", "preparation_results"):
        args["records"][role].extend(second["records"][role])
    refuses(args, "cache_identity_conflict")


def test_same_digest_and_size_across_preparations_deduplicates_shared_cache():
    args = fixture()
    second = fixture(preparation="prep-2", request_edit=lambda row: row["task"]["artifact"].update(uri="s3://bucket/second"))
    for role in ("preparation_links", "preparation_envelopes", "preparation_results"):
        args["records"][role].extend(second["records"][role])
    result = api().join_retained_scene_inventory_seed(**args)
    assert len(result["shared_cache_references"]) == 1
    assert sum(row["kind"] == "preparation_projected_file" for row in result["members"]) == 2


def test_absent_configuration_records_are_unknown_with_exact_expected_projection_path():
    args = fixture()
    result = api().join_retained_scene_inventory_seed(**args)
    assert result["status"] == "kept_unresolved" and "configuration_join_unresolved" in result["unresolved_reasons"]
    row = next(row for row in result["preparation_join_obligations"] if row["role"] == "configuration_progression")
    assert row["expected_path"] == "/retained/progression/scene-configuration-activations/prep-1/activation_progression.json"
    assert "sha256" not in row and "size_bytes" not in row
    assert not any(member["kind"] == "configuration_progression_workspace" for member in result["members"])
