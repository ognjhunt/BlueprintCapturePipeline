"""Robot capability/evaluation boundaries, offline snapshot admission and private-output regressions.

All organizations/pages/hardware here are synthetic; no provider, CRM or public artifact is used.
"""
import copy
import json
from datetime import date

import pytest

from tests.team_universe_fixture import TODAY, company, pipeline, screen_answers, weights
from tools.daily_research import site_screen as ss
from tools.team_universe import qualification as tq
from tools.team_universe import rank as tr
from tools.team_universe import universe as tu

PRODUCT = ("Synthbot Robotics 1 deploys the Synthbot Arm One for stacking cases at customer warehouses. "
           "Our palletizing controller commands the arm joints with RGB camera frames and joint position actions. "
           "The runnable policy controller operates the fixed arm on this rigid case stacking task.")


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", ())
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"})


@pytest.fixture
def retained(tmp_path):
    answers = screen_answers(1, robot_forms="fixed_arm", robot_forms_quote=PRODUCT)
    workspace, provider, reader, records = pipeline(tmp_path, {1: answers})
    states, _ = workspace.states()
    teams, _ = tu.team_list(tu.stage_records(workspace, states, "discover").values())
    record = records[1]
    evidence = tu._evidence(workspace.path("screen", "evidence", record["site_key"]).read_bytes())
    return workspace, provider, reader, teams, record, evidence


def proof(record, evidence, field="robot_forms"):
    url, quote = record["answers"][field + "_url"], record["answers"][field + "_quote"]
    return {"field": field, "url": url, "quote": quote, "quote_sha256": ss._sha256(quote.encode()),
            "level": record["verification"][field]["level"],
            "page_sha256": ss._sha256(evidence["pages"][url]["text"].encode()),
            "tool_result_sha256": evidence["pages"][url]["sha256"]}


def decision(teams, record, evidence, **changes):
    team = next(item for item in teams if item["site_key"] == record["site_key"])
    row = {**tq.binding(team, record), "capability_class": "physical_robot_task", "manual_current_task_fit": True,
           "capability_reason": "Reviewed own-page product explicitly offers arm case stacking.",
           "capability_evidence": [proof(record, evidence)], "identified_hardware": "Synthbot Arm One",
           "physical_task": "stacking cases", "offering_relation": "deploys the Synthbot Arm One",
           "capability_as_of": evidence["checked_on"], "capability_current_basis": "current_offering_page",
           "robot_forms": ["fixed_arm"], "task_families": ["palletizing_depalletizing"],
           "blueprint_current_evaluation_compatibility": "not_verified",
           "published_partner_intent": {"state": "unknown", "kind": "none", "evidence": []},
           "willingness_to_work_with_blueprint": "unknown", "promotion_allowed": False}
    row.update(changes)
    return row


def audit_raw(workspace, teams, screens, rows):
    return json.dumps({"schema_version": tq.AUDIT,
                       "team_list_sha256": ss._sha256(workspace.teams_path().read_bytes()),
                       "decisions_sha256": "b" * 64, "auditor_script_sha256": "c" * 64,
                       "audited_at": "2026-10-05T12:00:00Z", "auditor_reference": "synthetic-independent-review",
                       "scope_sha256": tq.scope_manifest(teams, screens)["sha256"], "teams": rows}).encode()


def outcome(record, row, evidence, today=TODAY):
    qualification = tq.qualify(record, row, evidence, today=today)
    return tr.assess(record, tr.load_config(), tr.load_weights(weights()), qualification)


def test_old_high_scoring_robot_words_cannot_qualify_without_audit(retained):
    _, _, _, _, record, evidence = retained
    assert tr._source_score(record, tr.load_config(), tr.load_weights(weights()))["score"] == 100
    result = outcome(record, None, evidence)
    assert result["tier"] == "pending" and not result["evaluable"]
    assert "offering_not_audited" in result["blockers"]


def test_hardware_phrase_cannot_be_fabricated_by_joining_separate_quotes():
    assert not tq._phrases_supported([{"quote": "Our robot is called Synthbot Arm"},
                                      {"quote": "One controller operates our other product"}], ["Synthbot Arm One"])


@pytest.mark.parametrize("text", [
    "Synthbot Robotics 1 offers a vision detector API for pallet dock analytics and PLC ERP integration.",
    "Synthbot Robotics 1 offers a solid state cleaner with no moving parts for dock and pallet industry facilities.",
    "Synthbot Robotics 1 offers a world model platform and simulation assets for pallet and dock layouts.",
    "Synthbot Robotics 1 offers CAD synthetic data and ONNX camera edge inspection for pallet traceability.",
])
def test_reference_software_and_infrastructure_never_inherit_v1_positive_words(tmp_path, text):
    workspace, _, _, records = pipeline(tmp_path, {1: screen_answers(1, robot_forms="software_only",
                                                                    robot_forms_quote=text)})
    record = records[1]
    states, _ = workspace.states()
    teams, _ = tu.team_list(tu.stage_records(workspace, states, "discover").values())
    evidence = tu._evidence(workspace.path("screen", "evidence", record["site_key"]).read_bytes())
    row = decision(teams, record, evidence, capability_class="adjacent_reference", manual_current_task_fit=False,
                   capability_reason="Reviewed quoted product is adjacent; no physical robot action offering shown.")
    result = outcome(record, row, evidence)
    assert result["tier"] == "reference_only" and not result["evaluable"]


@pytest.mark.parametrize("kind, relation", [
    ("embodied_control_task", "Our palletizing controller commands the arm joints"),
    ("robot_integrator_task", "deploys the Synthbot Arm One"),
    ("physical_robot_task", "deploys the Synthbot Arm One"),
])
def test_robot_product_control_software_and_integrator_remain_capability_prospects(retained, kind, relation):
    _, _, _, teams, record, evidence = retained
    row = decision(teams, record, evidence, capability_class=kind, offering_relation=relation)
    result = outcome(record, row, evidence)
    assert result["tier"] == "capability_prospect" and not result["evaluable"]
    assert "evaluation_compatibility_not_verified" in result["blockers"]


def test_generic_future_orchestration_is_pending_and_partner_invitation_is_independent(retained):
    _, _, _, teams, record, evidence = retained
    row = decision(teams, record, evidence, capability_class="embodied_control_task_pending",
                   manual_current_task_fit=False,
                   published_partner_intent={"state": "explicit_invitation", "kind": "design_partner",
                                             "evidence": [proof(record, evidence, "design_partners")]})
    qualification = tq.qualify(record, row, evidence, today=TODAY)
    assert qualification["offering"] == "unknown"
    assert qualification["partner_intent"] == {"state": "explicit_invitation", "kind": "design_partner"}
    assert qualification["willingness_to_work_with_blueprint"] == "unknown"
    assert not qualification["relationship_implied"] and not qualification["promotion_allowed"]
    assert outcome(record, row, evidence)["tier"] == "pending"


@pytest.mark.parametrize("attributed, owned_relation, expected", [
    (True, True, "capability_prospect"), (False, True, "pending"), (True, False, "pending"),
])
def test_owned_robot_offering_can_use_independent_named_team_hardware_reporting(tmp_path, attributed,
                                                                               owned_relation, expected):
    text = PRODUCT if attributed else PRODUCT.replace("Synthbot Robotics 1", "Another Robotics Team")
    offered = "Synthbot Robotics 1 offers our palletizing controller for stacking cases at customer warehouses."
    answers = screen_answers(1, robot_forms="fixed_arm", robot_forms_url="https://robot-news.example/synthbot-1",
                             robot_forms_quote=text, task_evidence_quote=offered)
    workspace, _, _, records = pipeline(tmp_path, {1: answers})
    record = records[1]
    states, _ = workspace.states()
    teams, _ = tu.team_list(tu.stage_records(workspace, states, "discover").values())
    evidence = tu._evidence(workspace.path("screen", "evidence", record["site_key"]).read_bytes())
    row = decision(teams, record, evidence, offering_relation="offers our palletizing controller" if owned_relation
                   else "deploys the Synthbot Arm One",
                   capability_evidence=[proof(record, evidence), proof(record, evidence, "task_evidence")])
    assert outcome(record, row, evidence)["tier"] == expected


@pytest.mark.parametrize("changes, code", [
    ({"identified_hardware": "unidentified robot hardware"}, "physical_robot_offering_evidence_incomplete"),
    ({"physical_task": "unidentified industrial task"}, "physical_robot_offering_evidence_incomplete"),
    ({"offering_relation": "possible future robot orchestration"}, "physical_robot_offering_evidence_incomplete"),
    ({"robot_forms": ["software_only"]}, "audit_row_invalid"),
    ({"capability_as_of": "2024-01-01"}, "physical_robot_offering_not_current"),
    ({"capability_as_of": "2027-01-01"}, "physical_robot_offering_not_current"),
    ({"manual_current_task_fit": False}, "current_robot_task_fit_not_established"),
])
def test_current_hardware_task_and_actual_offering_are_separate_requirements(retained, changes, code):
    _, _, _, teams, record, evidence = retained
    row = decision(teams, record, evidence, **changes)
    result = outcome(record, row, evidence)
    assert result["tier"] == "pending" and code in result["blockers"]


def test_fresh_funding_cannot_make_old_physical_offering_current(retained):
    _, _, _, teams, record, evidence = retained
    row = decision(teams, record, evidence)
    record["latest_activity"] = "2028-01-01"
    result = outcome(record, row, evidence, today=date(2028, 7, 2))
    assert result["tier"] == "pending" and "physical_robot_offering_not_current" in result["blockers"]


def evaluation(record, evidence):
    return {"profile": tq.PROFILE, "working_embodiment": "Synthbot Arm One", "physical_task": "stacking cases",
            "observation_interface": "RGB camera frames", "action_interface": "joint position actions",
            "runnable_controller": "runnable policy controller", "support_reference": "synthetic-runtime-contract-review",
            "as_of": evidence["checked_on"], "evidence": [proof(record, evidence)]}


def test_vendor_interfaces_and_profile_label_never_prove_blueprint_runtime_support(retained):
    _, _, _, teams, record, evidence = retained
    row = decision(teams, record, evidence, blueprint_current_evaluation_compatibility="supported",
                   evaluation=evaluation(record, evidence))
    result = outcome(record, row, evidence)
    assert result["tier"] == "capability_prospect" and not result["evaluable"]
    assert "blueprint_support_contract_unavailable" in result["blockers"]
    for name, bad in (("action_interface", "unproven API interface"), ("runnable_controller", "generic simulation SDK"),
                      ("working_embodiment", "different Robot Two"), ("profile", "future-humanoid-profile"),
                      ("as_of", "2024-01-01")):
        changed = copy.deepcopy(row)
        changed["evaluation"][name] = bad
        rejected = outcome(record, changed, evidence)
        assert rejected["tier"] == "capability_prospect" and not rejected["evaluable"]
        assert "blueprint_support_contract_unavailable" in rejected["blockers"]


def test_evidence_tamper_or_wrong_field_url_cannot_promote(retained):
    _, _, _, teams, record, evidence = retained
    for mutate in (lambda p: p.update(url="https://synthbot-1.example/other"),
                   lambda p: p.update(page_sha256="f" * 64),
                   lambda p: p.update(quote=PRODUCT + " invented hardware")):
        row = decision(teams, record, evidence)
        mutate(row["capability_evidence"][0])
        row["capability_evidence"][0]["quote_sha256"] = ss._sha256(row["capability_evidence"][0]["quote"].encode())
        result = outcome(record, row, evidence)
        assert result["tier"] == "pending" and "audit_evidence_not_held" in result["blockers"]
    changed = copy.deepcopy(evidence)
    url = record["answers"]["robot_forms_url"]
    changed["pages"][url]["text"] += " changed retained bytes"
    assert outcome(record, decision(teams, record, evidence), changed)["tier"] == "pending"


@pytest.mark.parametrize("mutation", [
    lambda doc: doc["teams"].clear(),
    lambda doc: doc["teams"].append(copy.deepcopy(doc["teams"][0])),
    lambda doc: doc["teams"][0].update(result_sha256="f" * 64),
    lambda doc: doc["teams"][0].update(evidence_sha256="f" * 64),
    lambda doc: doc["teams"][0].update(discovery_sha256="f" * 64),
    lambda doc: doc["teams"][0].update(domain="different.example"),
    lambda doc: doc.update(scope_sha256="f" * 64),
    lambda doc: doc.update(team_list_sha256="f" * 64),
])
def test_audit_must_bind_every_team_exact_discovery_result_and_evidence_snapshot(retained, mutation):
    workspace, _, _, teams, record, evidence = retained
    raw = audit_raw(workspace, teams, {record["site_key"]: record}, [decision(teams, record, evidence)])
    doc = json.loads(raw)
    mutation(doc)
    with pytest.raises(tu.TeamError):
        tr.rank(workspace, weights(), audit_raw=json.dumps(doc).encode(), today=TODAY)
    assert not (workspace.root / tu.RANKED_NAME).exists()


@pytest.mark.parametrize("mutation", [
    lambda doc: doc.update(extra="unrecognized"),
    lambda doc: doc["teams"][0].update(promotion_allowed=True),
    lambda doc: doc["teams"][0].update(manual_current_task_fit="true"),
    lambda doc: doc["teams"][0].update(manual_current_task_fit=1),
    lambda doc: doc["teams"][0].update(willingness_to_work_with_blueprint="yes"),
    lambda doc: doc["teams"][0].update(capability_class="physical_robot"),
    lambda doc: doc["teams"][0]["capability_evidence"][0].update(quote_sha256="f" * 64),
    lambda doc: doc.update(audited_at="2026-10-05T12:00:00+01:00"),
])
def test_malformed_audit_never_becomes_approval(retained, mutation):
    workspace, _, _, teams, record, evidence = retained
    doc = json.loads(audit_raw(workspace, teams, {record["site_key"]: record}, [decision(teams, record, evidence)]))
    mutation(doc)
    with pytest.raises(tu.TeamError, match="^team_universe_audit_invalid$"):
        tr.rank(workspace, weights(), audit_raw=json.dumps(doc).encode(), today=TODAY)


def test_duplicate_json_fields_are_rejected(retained):
    workspace, _, _, teams, record, evidence = retained
    raw = audit_raw(workspace, teams, {record["site_key"]: record}, [decision(teams, record, evidence)])
    raw = raw.replace(b'"promotion_allowed": false', b'"promotion_allowed": true, "promotion_allowed": false')
    with pytest.raises(tu.TeamError, match="^team_universe_audit_invalid$"):
        tr.rank(workspace, weights(), audit_raw=raw, today=TODAY)


def test_rank_is_offline_private_counts_only_and_preserves_v1_and_paid_receipts(retained, monkeypatch):
    workspace, provider, reader, teams, record, evidence = retained
    row = decision(teams, record, evidence)
    raw = audit_raw(workspace, teams, {record["site_key"]: record}, [row])
    old = workspace.root / "ranked.team-rank.v1.json"
    old.write_bytes(b"retained old derived artifact")
    before = {str(path.relative_to(workspace.root)): ss._sha256(path.read_bytes())
              for path in workspace.root.rglob("*") if path.is_file()}
    counts = len(provider.calls), len(reader.requested)
    monkeypatch.setattr(ss.TaskClient, "create", lambda *a, **kw: pytest.fail("provider POST forbidden"))
    result = tr.rank(workspace, weights(), audit_raw=raw, today=TODAY)
    assert result["tiers"] == {tier: int(tier == "capability_prospect") for tier in tr.TIERS}
    assert result["audit_sha256"] == ss._sha256(raw)
    assert (len(provider.calls), len(reader.requested)) == counts
    assert all(ss._sha256((workspace.root / name).read_bytes()) == digest for name, digest in before.items())
    ranked = json.loads((workspace.root / tu.RANKED_NAME).read_text())
    assert ranked["schema_version"] == "blueprint.team-rank.ranked.v2"
    assert ranked["teams"][0]["status"] == "capability_prospect"
    assert ranked["teams"][0]["qualification"]["partner_intent"]["state"] == "unknown"
    assert (workspace.root / tu.RANKED_NAME).stat().st_mode & 0o777 == 0o600
    assert not ranked["claim_boundary"]["evaluation_or_contact_authorized"]
    assert not any(value in ss.canonical(result) for value in ("Synthbot", ".example", PRODUCT))


def test_summary_does_not_reuse_qualification_after_source_change_or_date_change(retained):
    workspace, _, _, teams, record, evidence = retained
    raw = audit_raw(workspace, teams, {record["site_key"]: record}, [decision(teams, record, evidence)])
    tr.rank(workspace, weights(), audit_raw=raw, today=TODAY)
    assert tu.summary(workspace, today=TODAY)["rank"]["tiers"]["capability_prospect"] == 1
    tomorrow = date(2026, 10, 6)
    assert tu.summary(workspace, today=tomorrow)["rank"]["snapshot_state"] == "stale_snapshot"
    assert tu.summary(workspace, today=tomorrow)["rank"]["tiers"] is None
    path = workspace.path("screen", "results", record["site_key"])
    changed = path.read_bytes().replace(b"Synthbot Arm One", b"generic inspection camera")
    path.write_bytes(changed)
    summary = tu.summary(workspace, today=TODAY)
    assert summary["rank"]["snapshot_state"] == "stale_snapshot" and summary["rank"]["tiers"] is None


def test_redacted_retained_page_hash_is_distinct_from_original_tool_provenance(retained):
    _, _, _, teams, record, evidence = retained
    # Seal redacts non-admitted address text after the original page reader computed its hash.
    url = record["answers"]["robot_forms_url"]
    evidence["pages"][url]["sha256"] = "e" * 64
    row = decision(teams, record, evidence)
    assert row["capability_evidence"][0]["page_sha256"] != row["capability_evidence"][0]["tool_result_sha256"]
    assert outcome(record, row, evidence)["tier"] == "capability_prospect"
    row["capability_evidence"][0]["page_sha256"] = "e" * 64
    assert outcome(record, row, evidence)["tier"] == "pending"


def test_unscreened_scope_row_is_explicitly_bound_and_cannot_qualify(tmp_path):
    workspace, _, _, records = pipeline(tmp_path, {1: screen_answers(1, robot_forms="fixed_arm", robot_forms_quote=PRODUCT)},
                                       discovery=[company(1), company(2, source_quote="Unidentified company builds robots.")])
    states, _ = workspace.states()
    teams, _ = tu.team_list(tu.stage_records(workspace, states, "discover").values())
    record = records[1]
    evidence = tu._evidence(workspace.path("screen", "evidence", record["site_key"]).read_bytes())
    pending = next(item for item in teams if item["site_key"] != record["site_key"])
    row = {**tq.binding(pending, None), "capability_class": "unscreened", "manual_current_task_fit": False,
           "capability_reason": "No retained screen.", "capability_evidence": [],
           "blueprint_current_evaluation_compatibility": "not_verified",
           "published_partner_intent": {"state": "unknown", "kind": "none", "evidence": []},
           "willingness_to_work_with_blueprint": "unknown", "promotion_allowed": False}
    result = tr.rank(workspace, weights(), audit_raw=audit_raw(workspace, teams, {record["site_key"]: record},
                                                              [decision(teams, record, evidence), row]), today=TODAY)
    assert result["tiers"]["insufficient"] == 1 and result["tiers"]["capability_prospect"] == 1
