"""Same-session output repair: complete located feedback, revision, revalidation.

Hermetic: fake provider and in-memory Firestore only. The frozen legacy
validator below is the pre-repair strict gate verbatim; the new gate must agree
with it on every corruption so repair never loosens what reaches QA.
"""
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_daily_research_operator_canary as canary_tests
from tests.test_daily_research_adaptive import result
from tests.test_daily_research_consumer import HEADERS, QAAPI
from tests.test_daily_research_knowledge import context, enable_v3, policy_bundle, v2, v3
from tests.test_daily_research_runner import DAY, NOW, output
from tests.test_daily_research_runner import fixture as runner_fixture
from tests.test_daily_research_search import SearchAPI
from tools.daily_research import discovery, freshness, render, repair, search
from tools.daily_research.consumer import Consumer, qa_deadline
from tools.daily_research.contracts import checked_day, lookup
from tools.daily_research.knowledge import (
    LEVELS,
    SnapshotError,
    calendar_date,
    fact_state,
    require,
    shape,
    source_moment,
    text,
    timestamp,
    url,
)
from tools.daily_research.firestore import Bridge, FencedProvider, FirestoreLedger
from tools.daily_research.runner import (
    AGENT,
    SHEET,
    Refusal,
    Runner,
    canonical,
    digest,
    keys,
    observation_seconds,
    output_issues,
    public_url,
    save_json,
    status_summary,
    validate_output,
)

ROOT = Path(__file__).resolve().parents[1]
canary = canary_tests.canary

# --- Frozen pre-repair strict gate (origin/main 5695309ec), verbatim semantics. ---


def legacy_evidence(value, day, context, observed_at=None, *, policy=None):
    if value["role"] in {"capability", "background"}:
        require(value["evidence_level"] in LEVELS, "evidence_level_invalid")
        require(value["evidence_level"] != "unknown", "unsupported_evidence_level")
    else:
        require(value["evidence_level"] is None, "site_evidence_level_must_be_null")
    require(value["checked_date"] == checked_day(value["source_checked_at"]), "evidence_date_integrity_invalid")
    require(calendar_date(value["checked_date"]) <= date.fromisoformat(day), "evidence_date_in_future")
    require(source_moment(value["source_checked_at"]) <= timestamp(context["snapshot_loaded_at"])
            or value["origin"] == "live", "evidence_date_in_future")
    if policy is not None:
        require(value["assertion_scope"] in {"as_of_background", "current_operational", "deployment_critical"}, "evidence_assertion_scope_invalid")
        if value["origin"] == "snapshot":
            require(value["assertion_scope"] == "as_of_background", "cached_operational_assertion_forbidden")
    if value["origin"] == "live":
        require(source_moment(value["source_checked_at"]) <= (observed_at or datetime.now(timezone.utc)), "evidence_date_in_future")
        require(value["checked_date"] == day and value["snapshot_loaded_at"] is None
                and value["snapshot_record_id"] is None and value["snapshot_fact_id"] is None,
                "live_evidence_binding_invalid")
        require(value["revalidated_at"] in {None, value["source_checked_at"]}, "evidence_date_integrity_invalid")
        return
    require(value["origin"] == "snapshot", "evidence_origin_invalid")
    require(value["role"] in ({"capability", "background"} if policy is not None else {"capability"}), "live_task_geography_required")
    fact = lookup(context, value["snapshot_record_id"], value["snapshot_fact_id"])
    if policy is not None:
        freshness.cached_citation(fact, policy, value["snapshot_record_id"], value["role"])
    else:
        require(fact["load_state"] == "usable_background"
                and fact_state(fact, observed_at or datetime.now(timezone.utc)) == "usable_background", "cached_fact_not_usable")
    require(value["claim"] == fact["statement"] and value["evidence_level"] == fact["evidence_level"]
            and value["snapshot_loaded_at"] == context["snapshot_loaded_at"], "cached_fact_binding_invalid")
    source = {"url": value["url"], "publisher": value["publisher"], "publication_date": value["source_date"],
              "source_checked_at": value["source_checked_at"], "revalidated_at": value["revalidated_at"],
              "classification": value["classification"], "quote": value["quote"]}
    require(source in fact["sources"], "cached_source_binding_invalid")
    require(value["claim_kind"] != "hypothesis", "cached_fact_binding_invalid")
    if fact["evidence_level"] == "vendor_claim":
        require(value["claim_kind"] == "vendor_claim", "vendor_claim_presented_as_fact")


def legacy_deltas(values, day, context, observed_at=None, contract_version=2):
    require(isinstance(values, list) and len(values) <= 10, "knowledge_deltas_invalid")
    for delta in values:
        shape(delta, {"record_id", "fact_id", "reason", "proposed_statement", "evidence", "unknowns"})
        age_reason = "refresh_due" if contract_version == 3 else "stale"
        require(delta["reason"] in {"gap", "conflict", age_reason, "unsupported", "discovery", "consequential"}, "knowledge_delta_reason_invalid")
        if delta["reason"] == "discovery":
            require(delta["record_id"] is None and delta["fact_id"] is None, "knowledge_delta_binding_invalid")
        else:
            lookup(context, delta["record_id"], delta["fact_id"])
        text(delta["proposed_statement"])
        require(isinstance(delta["unknowns"], list) and 1 <= len(delta["unknowns"]) <= 20, "knowledge_delta_unknowns_required")
        for unknown in delta["unknowns"]:
            text(unknown)
        require(isinstance(delta["evidence"], list) and 1 <= len(delta["evidence"]) <= 4, "knowledge_delta_evidence_required")
        for item in delta["evidence"]:
            shape(item, {"url", "publisher", "publication_date", "source_checked_at", "classification", "evidence_level", "quote"},
                  {"assertion_scope"} if contract_version == 3 else ())
            if "assertion_scope" in item:
                require(isinstance(item["assertion_scope"], str)
                        and item["assertion_scope"] in {"as_of_background", "current_operational", "deployment_critical"},
                        "knowledge_delta_assertion_scope_invalid")
            url(item["url"])
            text(item["publisher"], 200)
            text(item["quote"])
            require(checked_day(item["source_checked_at"]) == day, "delta_live_evidence_required")
            require(source_moment(item["source_checked_at"]) <= (observed_at or datetime.now(timezone.utc)), "evidence_date_in_future")
            require(item["classification"] in {"operator", "vendor", "independent"} and item["evidence_level"] in LEVELS,
                    "knowledge_delta_evidence_invalid")
            if item["publication_date"] is not None:
                require(calendar_date(item["publication_date"]) <= date.fromisoformat(day), "source_date_in_future")


def legacy_validate_output(output, run_date, known, *, contract_version=1, knowledge_context=None, observed_at=None, refresh_policy=None):
    if not isinstance(output, dict):
        raise Refusal("output_schema_invalid")
    required_output = {"checked_date", "findings", "blockers", "proposed_next_actions", "candidates"}
    if contract_version in {2, 3}:
        required_output |= {"schema_version", "snapshot_content_hash", "proposed_knowledge_deltas"}
        if (output.get("schema_version") != f"blueprint.daily-research.v{contract_version}" or not knowledge_context
                or output.get("snapshot_content_hash") != knowledge_context["content_hash"]):
            raise Refusal("output_version_or_snapshot_binding_invalid")
        if contract_version == 3:
            required_output.add("refresh_policy_hash")
            try:
                require(isinstance(refresh_policy, dict), "refresh_policy_context_missing")
                freshness.validate_context(knowledge_context, refresh_policy)
                require(output.get("refresh_policy_hash") == refresh_policy["policy_hash"], "output_refresh_policy_binding_invalid")
            except SnapshotError as exc:
                raise Refusal(str(exc)) from None
        try:
            legacy_deltas(output.get("proposed_knowledge_deltas"), run_date, knowledge_context, observed_at, contract_version)
        except SnapshotError as exc:
            raise Refusal(str(exc)) from None
    elif contract_version != 1:
        raise Refusal("research_contract_version_unsupported")
    if contract_version == 3 and "coverage" in output:
        required_output.add("coverage")
        try:
            discovery.validate_coverage(output["coverage"], len(output.get("candidates", [])))
        except (ValueError, TypeError) as exc:
            raise Refusal(str(exc) if isinstance(exc, ValueError) else "discovery_coverage_invalid") from None
    if set(output) != required_output:
        raise Refusal("output_schema_invalid")
    limit = discovery.MAX_CANDIDATES if contract_version == 3 else 3
    if output["checked_date"] != run_date or not isinstance(output["candidates"], list) or len(output["candidates"]) > limit:
        raise Refusal("output_date_or_count_invalid")
    for field in ("findings", "blockers", "proposed_next_actions"):
        if (not isinstance(output[field], list)
                or (contract_version in {2, 3} and len(output[field]) > 20)
                or any(not isinstance(x, str) or len(x) > 2000 or (contract_version in {2, 3} and not x.strip()) for x in output[field])):
            raise Refusal("output_summary_invalid")
    accepted, duplicates = [], []
    required = {"organization", "organization_url", "site", "location", "task",
                "potential_robot_match", "qualification_status", "confidence", "unknowns",
                "proposed_next_action", "evidence"}
    for c in output["candidates"]:
        if not isinstance(c, dict) or set(c) != required:
            raise Refusal("candidate_schema_invalid")
        for field in required - {"unknowns", "evidence"}:
            if not isinstance(c[field], str) or not c[field].strip() or len(c[field]) > 2000:
                raise Refusal("candidate_field_invalid")
        if c["confidence"] not in {"low", "medium", "high"} or c["qualification_status"] not in {"unqualified", "needs_review"}:
            raise Refusal("candidate_claim_ceiling_invalid")
        if not isinstance(c["unknowns"], list) or not c["unknowns"] or any(not isinstance(x, str) for x in c["unknowns"]):
            raise Refusal("candidate_unknowns_required")
        if contract_version in {2, 3} and (len(c["unknowns"]) > 20 or any(not x.strip() or len(x) > 2000 for x in c["unknowns"])):
            raise Refusal("candidate_unknowns_required")
        if not isinstance(c["evidence"], list) or not 3 <= len(c["evidence"]) <= 12:
            raise Refusal("candidate_evidence_required")
        roles = set()
        for e in c["evidence"]:
            evidence_fields = {"claim", "url", "publisher", "source_date", "checked_date", "classification", "claim_kind", "role", "quote"}
            if contract_version in {2, 3}:
                evidence_fields |= {"origin", "evidence_level", "source_checked_at", "snapshot_loaded_at", "revalidated_at",
                                    "snapshot_record_id", "snapshot_fact_id"}
                if contract_version == 3:
                    evidence_fields.add("assertion_scope")
            if not isinstance(e, dict) or set(e) != evidence_fields:
                raise Refusal("evidence_schema_invalid")
            public_url(e["url"])
            text_fields = ("claim", "publisher") if contract_version in {2, 3} and e["origin"] == "snapshot" else ("claim", "publisher", "quote")
            if ((contract_version == 1 and e["checked_date"] != run_date) or e["classification"] not in {"operator", "vendor", "independent"}
                    or e["claim_kind"] not in {"fact", "vendor_claim", "hypothesis"}
                    or e["role"] not in ({"task", "capability", "geography", "background"} if contract_version == 3 else {"task", "capability", "geography"})
                    or any(not isinstance(e[x], str) or not e[x].strip() or len(e[x]) > 2000 for x in text_fields)):
                raise Refusal("evidence_field_invalid")
            if e["classification"] == "vendor" and e["claim_kind"] == "fact":
                raise Refusal("vendor_claim_presented_as_fact")
            if e["source_date"] is not None:
                try:
                    published = calendar_date(e["source_date"]) if contract_version in {2, 3} else date.fromisoformat(e["source_date"])
                except SnapshotError as exc:
                    raise Refusal(str(exc)) from None
                if published > date.fromisoformat(run_date):
                    raise Refusal("source_date_in_future")
            if contract_version in {2, 3}:
                try:
                    legacy_evidence(e, run_date, knowledge_context, observed_at, policy=refresh_policy if contract_version == 3 else None)
                except SnapshotError as exc:
                    raise Refusal(str(exc)) from None
            roles.add(e["role"])
        if (not {"task", "capability", "geography"} <= roles if contract_version == 3 else roles != {"task", "capability", "geography"}):
            raise Refusal("task_capability_geography_evidence_required")
        if not any(e["role"] == "task" and e["classification"] == "operator"
                   and public_url(e["url"]) == public_url(c["organization_url"]) for e in c["evidence"]):
            raise Refusal("operator_task_source_domain_mismatch")
        identities = keys(c)
        if identities & known:
            duplicates.append({"organization": c["organization"], "site": c["site"], "reason": "matching_site_task"})
        else:
            accepted.append({**c, "candidate_key": min(identities), "identity_keys": sorted(identities)})
            known.update(identities)
    return accepted, duplicates


# --- Shared synthetic documents ---------------------------------------------------


def delta(level="vendor_claim", classification="vendor", count=1, day=DAY):
    return {"record_id": None, "fact_id": None, "reason": "discovery",
            "proposed_statement": "Synthetic vendor states a tray depositing capability.",
            "unknowns": ["Independent demonstration unknown"],
            "evidence": [{"url": f"https://vendor.example/robot-{index}", "publisher": "Synthetic Vendor",
                          "publication_date": None, "source_checked_at": day, "classification": classification,
                          "evidence_level": level, "quote": "Synthetic depositing claim"} for index in range(count)]}


def base_v1():
    return output(), {"contract_version": 1}


def base_v2():
    ctx = context()
    return v2(ctx), {"contract_version": 2, "knowledge_context": ctx, "observed_at": NOW}


def base_v3():
    document, ctx, policy = result(2)
    document["proposed_knowledge_deltas"] = [delta()]
    return document, {"contract_version": 3, "knowledge_context": ctx, "observed_at": NOW, "refresh_policy": policy}


BASES = {1: base_v1, 2: base_v2, 3: base_v3}
MISSING = object()
REPLACEMENTS = (MISSING, None, "", 7, [], {}, ["x"], "x" * 2001, "2099-01-01", "qualified", "vendor",
                "background", "snapshot", "unknown", "http://10.0.0.1/a", True)


def nodes(value, pointer=""):
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, child in children:
        yield f"{pointer}/{key}"
        yield from nodes(child, f"{pointer}/{key}")


def mutate(document, pointer, replacement):
    result_document = deepcopy(document)
    *parents, last = pointer.split("/")[1:]
    target = result_document
    for part in parents:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = int(last) if isinstance(target, list) else last
    if replacement is MISSING:
        del target[key]
    elif replacement == "__extra__":
        target[key] = {**target[key], "unexpected_field": 1} if isinstance(target[key], dict) else target[key]
    else:
        target[key] = deepcopy(replacement)
    return result_document


def corpus(document):
    for pointer in list(nodes(document)):
        for replacement in (*REPLACEMENTS, "__extra__"):
            yield pointer, replacement, mutate(document, pointer, replacement)


def outcome(function):
    try:
        accepted, duplicates = function()
        return "ok", canonical([accepted, duplicates])
    except Refusal as exc:
        return "refusal", str(exc)
    except Exception as exc:  # noqa: BLE001 - malformed values must fail identically, not merely somehow
        return "error", type(exc).__name__


def compare(document, options):
    legacy = outcome(lambda: legacy_validate_output(deepcopy(document), DAY, set(), **options))
    current = outcome(lambda: validate_output(deepcopy(document), DAY, set(), **options))
    assert current == legacy
    issues = list(output_issues(document, DAY, collect=True, **options))
    if current[0] == "ok":
        assert issues == []
    elif current[0] == "refusal":
        assert issues and issues[0]["code"] == current[1], (current, issues[:3])
    else:
        assert issues  # A malformed value is always reported, never silently passed.
    return current


@pytest.mark.parametrize("version", [1, 2, 3])
def test_strict_gate_is_unchanged_and_diagnosis_agrees_for_every_single_corruption(version):
    document, options = BASES[version]()
    assert compare(document, options)[0] == "ok"
    seen = {"ok": 0, "refusal": 0, "error": 0}
    for _pointer, _replacement, mutated in corpus(document):
        seen[compare(mutated, options)[0]] += 1
    assert seen["refusal"] > 100  # The corpus genuinely exercises the gate.


@pytest.mark.parametrize("version", [2, 3])
def test_strict_gate_is_unchanged_for_compound_corruptions(version):
    document, options = BASES[version]()
    pointers = list(nodes(document))
    rng = random.Random(20261001 + version)
    for _ in range(300):
        mutated = document
        for _step in range(rng.randint(2, 4)):
            available = list(nodes(mutated)) or pointers
            mutated = mutate(mutated, rng.choice(available), rng.choice((*REPLACEMENTS, "__extra__")))
        compare(mutated, options)


def entry(candidate, role, origin="live"):
    return next(e for e in candidate["evidence"] if e["role"] == role and e["origin"] == origin)


def valid_bundle():
    document, ctx, policy = result()
    document["coverage"].update(defined_run_scope=["Synthetic exact site/task industry/region scope"],
                                unresolved_promising_branches=[], completion_state="coverage_complete")
    return document, ctx, policy


def valid_output():
    return valid_bundle()[0]


def defective(document):
    """Fifteen independent defects, each exactly located; mirrors real mixed agent errors."""
    bad = deepcopy(document)
    bad["proposed_knowledge_deltas"] = [delta(level=None, classification="operator", count=3)]
    entry(bad["candidates"][0], "task")["evidence_level"] = "named_deployment"
    geography = entry(bad["candidates"][1], "geography")
    geography.update(classification="vendor", claim_kind="fact")
    entry(bad["candidates"][2], "task")["source_date"] = "2099-01-01"
    entry(bad["candidates"][3], "geography")["source_checked_at"] = "2026-09-29"
    evidence = bad["candidates"][4]["evidence"]
    evidence[evidence.index(entry(bad["candidates"][4], "geography"))] = deepcopy(entry(bad["candidates"][4], "task"))
    bad["candidates"][5]["qualification_status"] = "qualified"
    bad["candidates"][6]["unknowns"] = []
    bad["coverage"]["unresolved_promising_branches"] = ["Synthetic unresolved branch"]
    bad["findings"].append("")
    bad["candidates"][7]["contact_email"] = "invented@example.com"
    entry(bad["candidates"][8], "task")["assertion_scope"] = "forever"
    bad["candidates"][9]["organization_url"] = "https://unrelated.example/"
    return bad


def index_of(document, number, role):
    return document["candidates"][number]["evidence"].index(entry(document["candidates"][number], role))


def expected_issues(document):
    return {
        ("/proposed_knowledge_deltas/0/evidence/0/evidence_level", "knowledge_delta_evidence_invalid"),
        ("/proposed_knowledge_deltas/0/evidence/1/evidence_level", "knowledge_delta_evidence_invalid"),
        ("/proposed_knowledge_deltas/0/evidence/2/evidence_level", "knowledge_delta_evidence_invalid"),
        (f"/candidates/0/evidence/{index_of(document, 0, 'task')}/evidence_level", "site_evidence_level_must_be_null"),
        (f"/candidates/1/evidence/{index_of(document, 1, 'geography')}/claim_kind", "vendor_claim_presented_as_fact"),
        (f"/candidates/2/evidence/{index_of(document, 2, 'task')}/source_date", "source_date_in_future"),
        (f"/candidates/3/evidence/{index_of(document, 3, 'geography')}/checked_date", "evidence_date_integrity_invalid"),
        ("/candidates/4/evidence", "task_capability_geography_evidence_required"),
        ("/candidates/5/qualification_status", "candidate_claim_ceiling_invalid"),
        ("/candidates/6/unknowns", "candidate_unknowns_required"),
        ("/coverage", "discovery_completion_has_unresolved_branches"),
        (f"/findings/{len(document['findings'])}", "output_summary_invalid"),
        ("/candidates/7", "candidate_schema_invalid"),
        (f"/candidates/8/evidence/{index_of(document, 8, 'task')}/assertion_scope", "evidence_assertion_scope_invalid"),
        ("/candidates/9/organization_url", "operator_task_source_domain_mismatch"),
    }


def test_every_problem_is_located_at_once_while_the_gate_still_fails_fast():
    good, ctx, policy = valid_bundle()
    options = {"contract_version": 3, "knowledge_context": ctx, "observed_at": NOW, "refresh_policy": policy}
    validate_output(good, DAY, set(), **options)
    bad = defective(good)
    issues = list(output_issues(bad, DAY, collect=True, **options))
    assert {(i["pointer"], i["code"]) for i in issues} == expected_issues(good)
    assert len(issues) == 15
    with pytest.raises(Refusal, match="^knowledge_delta_evidence_invalid$"):
        validate_output(bad, DAY, set(), **options)


# --- Runner integration ----------------------------------------------------------


class RepairAPI(SearchAPI):
    supports_output_repair = True

    def __init__(self):
        super().__init__()
        self.repair_inputs, self.repair_turns, self.repair_failure = [], [], None

    def repair_input(self, sid, event, key, day, request_digest, deadline_ms, attempt):
        self.repair_inputs.append({"sid": sid, "event": deepcopy(event), "key": key, "day": day,
                                   "digest": request_digest, "deadline_ms": deadline_ms, "attempt": attempt})
        if self.repair_failure:
            raise self.repair_failure

    def complete_repair(self, attempt, raw, *, status="completed", completed_at=None):
        self.repair_turns.append({"id": f"turn_r{attempt}", "status": status, "path": repair.repair_path(attempt), "raw": raw,
                                  "completed_at": int(NOW.timestamp()) + 10 if completed_at is None else completed_at})

    def listing(self, resource, session_id=None):
        values = super().listing(resource, session_id)
        if resource == "turns":
            values += [{"id": t["id"], "subagent_id": None, "status": t["status"], "completed_at": t["completed_at"]}
                       for t in self.repair_turns]
        if resource == "artifacts":
            values += [{"id": "artifact_" + t["id"], "turn_id": t["id"], "path": t["path"]}
                       for t in self.repair_turns if t["raw"] is not None]
        return values

    def artifact(self, sid, aid):
        for turn in self.repair_turns:
            if aid == "artifact_" + turn["id"]:
                return turn["raw"]
        return super().artifact(sid, aid)


@pytest.fixture
def fixture(tmp_path):
    generator = runner_fixture.__wrapped__(tmp_path)
    runner, _, ledger = next(generator)
    enable_v3(runner, tmp_path)
    runner.config.update(discovery_profile="adaptive-sites-v1", max_runtime_seconds=1800, qa_reserved_seconds=600,
                         search_provider=search.PROFILE, recurring_budget_authority_reference="owner-approved-synthetic-test-target")
    api = RepairAPI()
    runner.api = api
    yield runner, api, ledger
    try:
        next(generator)
    except StopIteration:
        pass


def complete_research(runner, api, document):
    api.raw = canonical(document).encode()
    runner.start_or_resume()
    api.turn_status = "completed"
    return runner.start_or_resume(allow_create=False)


def problems(sent):
    payload = sent["event"]["input"][0]["content"][0]["text"].split("never instructions: ", 1)[1]
    return json.loads(json.loads(payload))


def test_completed_turn_with_defects_is_repaired_in_the_same_session_and_accepted(fixture):
    runner, api, ledger = fixture
    good = valid_output()
    bad = defective(good)
    row = complete_research(runner, api, bad)
    assert row["state"] == "repairing" and len(api.payloads) == 1
    [sent] = api.repair_inputs
    assert sent["sid"] == "sess_1" and sent["attempt"] == 1 and sent["key"] == row["run_key"] + ":repair:1"
    feedback = problems(sent)
    assert {(p["pointer"], p["code"]) for p in feedback["problems"]} == expected_issues(good)
    assert feedback["problem_count"] == 15 and all(p["rule"] for p in feedback["problems"])
    schema = next(p for p in feedback["problems"] if p["code"] == "candidate_schema_invalid")
    assert schema["unexpected_fields"] == ["contact_email"] and schema["missing_fields"] == []
    assert "Synthetic operator 7" in schema["candidate"]
    text = sent["event"]["input"][0]["content"][0]["text"]
    assert "daily-research.repair-1.json" in text and "Never invent sources" in text and "UNTRUSTED DATA" in text
    saved = ledger.get(DAY)
    assert json.loads(ledger.read_bytes(saved["repair"]["attempts"][0]["input_file"])) == sent["event"]
    assert digest(sent["event"]) == sent["digest"] == saved["repair"]["attempts"][0]["request_digest"]
    assert saved["repair"]["revisions"][0]["issue_count"] == 15

    api.complete_repair(1, canonical(good).encode())
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review", row.get("repair")
    assert len(api.repair_inputs) == 1 and len(api.payloads) == 1
    assert row["repair"]["outcome"]["state"] == "accepted"
    binding = row["packet"]["output_repair"]
    assert binding["revision"] == 1 and binding["excluded"] == [] and binding["knowledge_approved"] is False
    assert binding["repair_turn_ids"] == ["turn_r1"] and len(row["packet"]["candidates"]) == 10
    # The original stays exactly as received; the revision is retained beside it.
    assert ledger.read_bytes(DAY + "-artifact.json") == canonical(bad).encode()
    assert json.loads(ledger.read_bytes(DAY + "-output.json")) == bad
    assert ledger.read_bytes(DAY + "-repair-1-artifact.json") == canonical(good).encode()
    assert json.loads(ledger.read_bytes(DAY + "-repair-1-validation.json"))["issue_count"] == 0
    assert json.loads(ledger.read_bytes(DAY + "-repair-1-original.json"))["issue_count"] == 15
    assert status_summary(row)["output_repair"]["outcome"]["state"] == "accepted"


def test_progress_gets_a_second_attempt_from_the_revision_not_the_original(fixture):
    runner, api, _ = fixture
    good = valid_output()
    bad = deepcopy(good)
    bad["candidates"][5]["qualification_status"] = "qualified"
    bad["candidates"][6]["unknowns"] = []
    complete_research(runner, api, bad)
    partly = deepcopy(good)
    partly["candidates"][6]["unknowns"] = []
    api.complete_repair(1, canonical(partly).encode())
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "repairing" and len(api.repair_inputs) == 2
    second = problems(api.repair_inputs[1])
    assert [(p["pointer"], p["code"]) for p in second["problems"]] == [("/candidates/6/unknowns", "candidate_unknowns_required")]
    assert row["repair"]["attempts"][1]["source_path"] == repair.repair_path(1)
    api.complete_repair(2, canonical(good).encode())
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and row["packet"]["output_repair"]["revision"] == 2


def test_repeated_identical_failure_stops_and_excludes_only_the_defective_item(fixture):
    runner, api, ledger = fixture
    bad = valid_output()
    entry(bad["candidates"][3], "task")["evidence_level"] = "named_deployment"
    complete_research(runner, api, bad)
    api.complete_repair(1, canonical(bad).encode())
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and len(api.repair_inputs) == 1
    assert row["repair"]["outcome"]["state"] == "accepted_with_exclusions"
    assert row["repair"]["outcome"]["finalize_reason"] == "repeated_failure"
    [excluded] = row["packet"]["output_repair"]["excluded"]
    assert excluded["field"] == "candidates" and excluded["index"] == 3
    assert excluded["identity"]["organization"] == "Synthetic operator 3"
    assert excluded["issues"][0]["code"] == "site_evidence_level_must_be_null"
    assert len(row["packet"]["candidates"]) == 9
    assert "Synthetic operator 3" not in canonical(row["packet"]["candidates"])
    assert ledger.read_bytes(DAY + "-artifact.json") == canonical(bad).encode()


def test_global_defect_that_survives_repair_escalates_with_the_complete_list(fixture):
    runner, api, ledger = fixture
    bad = valid_output()
    bad["schema_version"] = "blueprint.daily-research.v2"
    complete_research(runner, api, bad)
    api.complete_repair(1, canonical(bad).encode())
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "failed" and row["error"] == "output_repair_exhausted"
    outcome_record = row["repair"]["outcome"]
    assert outcome_record["state"] == "escalated" and outcome_record["reason"] == "repeated_failure"
    assert outcome_record["remaining_codes"] == {"output_version_or_snapshot_binding_invalid": 1}
    assert json.loads(ledger.read_bytes(outcome_record["validation_file"]))["issues"][0]["pointer"] == "/schema_version"
    assert row["cleanup_required"] is True and "packet" not in row


def test_exhausted_attempts_keep_the_best_revision_and_exclude_its_leftovers(fixture):
    runner, api, _ = fixture
    good = valid_output()
    bad = deepcopy(good)
    bad["schema_version"] = "wrong"
    entry(bad["candidates"][2], "task")["evidence_level"] = "named_deployment"
    complete_research(runner, api, bad)
    first = deepcopy(good)
    entry(first["candidates"][2], "task")["evidence_level"] = "named_deployment"
    first["candidates"][4]["qualification_status"] = "qualified"
    api.complete_repair(1, canonical(first).encode())
    runner.start_or_resume(allow_create=False)
    second = deepcopy(good)
    entry(second["candidates"][2], "task")["evidence_level"] = "named_deployment"
    api.complete_repair(2, canonical(second).encode())
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and len(api.repair_inputs) == 2
    assert row["repair"]["outcome"]["finalize_reason"] == "attempts_exhausted"
    assert row["packet"]["output_repair"]["revision"] == 2
    assert [e["index"] for e in row["packet"]["output_repair"]["excluded"]] == [2]


def test_an_unacknowledged_input_is_never_resent_and_its_turn_is_still_adopted(fixture):
    runner, api, _ = fixture
    good = valid_output()
    bad = deepcopy(good)
    bad["candidates"][5]["qualification_status"] = "qualified"
    api.repair_failure = TimeoutError()
    row = complete_research(runner, api, bad)
    assert row["state"] == "repairing" and row["repair"]["attempts"][0]["input_reply_unresolved"] is True
    api.repair_failure = None
    for _ in range(3):
        assert runner.start_or_resume(allow_create=False)["state"] == "repairing"
    assert len(api.repair_inputs) == 1
    api.complete_repair(1, canonical(good).encode())
    assert runner.start_or_resume(allow_create=False)["state"] == "awaiting_review"
    assert len(api.repair_inputs) == 1


def test_a_refused_input_claim_is_definite_and_finalizes_without_waiting(fixture):
    runner, api, _ = fixture
    bad = valid_output()
    bad["candidates"][5]["qualification_status"] = "qualified"
    api.repair_failure = Refusal("research_repair_input_not_admitted")
    row = complete_research(runner, api, bad)
    assert row["state"] == "awaiting_review"
    assert row["repair"]["attempts"][0]["error"] == "research_repair_input_not_admitted"
    assert [e["index"] for e in row["packet"]["output_repair"]["excluded"]] == [5]


def test_deadline_cancels_the_repair_turn_once_then_finalizes(fixture):
    runner, api, _ = fixture
    bad = valid_output()
    bad["candidates"][5]["qualification_status"] = "qualified"
    complete_research(runner, api, bad)
    api.complete_repair(1, None, status="in_progress")
    assert runner.start_or_resume(allow_create=False)["state"] == "repairing" and not api.cancellations
    runner.clock = lambda: NOW + timedelta(seconds=repair.ATTEMPT_SECONDS + 1)
    runner.start_or_resume(allow_create=False)
    runner.start_or_resume(allow_create=False)
    assert api.cancellations == [("sess_1", "blueprint-researcher:" + DAY + ":repair:1")]
    api.repair_turns[0]["status"] = "cancelled"
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and row["repair"]["attempts"][0]["error"] == "repair_turn_cancelled"


def test_observer_stop_and_cancel_current_leave_the_durable_attempt_untouched(fixture):
    runner, api, ledger = fixture
    bad = valid_output()
    bad["candidates"][5]["qualification_status"] = "qualified"
    complete_research(runner, api, bad)
    before = ledger.get(DAY)
    assert runner.cancel_current(DAY, "observer_interrupted") == before
    assert not api.cancellations and ledger.get(DAY) == before
    assert observation_seconds(before, runner.config, "research", now=NOW) >= repair.ATTEMPT_SECONDS


def test_system_failures_keep_the_legacy_outcome_and_never_spend(fixture):
    runner, api, _ = fixture
    bad = valid_output()
    bad["candidates"][5]["qualification_status"] = "qualified"
    api.raw = canonical(bad).encode()
    runner.start_or_resume()
    api.turn_status = "completed"
    runner.clock = lambda: NOW + timedelta(hours=27)  # CRM snapshot now stale: Blueprint's problem, not the agent's.
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "failed" and row["error"] == "crm_snapshot_missing_incomplete_or_stale"
    assert "repair" not in row and not api.repair_inputs


def test_provider_without_repair_support_keeps_legacy_failure(fixture):
    runner, api, _ = fixture
    api.supports_output_repair = False
    bad = valid_output()
    bad["candidates"][5]["qualification_status"] = "qualified"
    row = complete_research(runner, api, bad)
    assert row["state"] == "failed" and row["error"] == "candidate_claim_ceiling_invalid" and "repair" not in row


def test_repair_turn_may_reread_sources_only_within_its_own_turn(fixture):
    runner, api, ledger = fixture
    bad = valid_output()
    bad["candidates"][5]["qualification_status"] = "qualified"
    complete_research(runner, api, bad)
    api.complete_repair(1, None, status="in_progress")
    api.actions = [{"type": "function_call", "turn_id": "turn_r1", "call_id": "call_r1", "name": search.READ,
                    "arguments": {"url": "https://operator.example/source"}}]
    assert runner.start_or_resume(allow_create=False)["state"] == "repairing"
    call = ledger.get(DAY)["application_tool_calls"]["call_r1"]
    assert call["phase"] == "repair" and call["repair_attempt"] == 1 and call["result_acknowledged"] is True
    api.actions = [{"type": "function_call", "turn_id": "turn_1", "call_id": "call_bad", "name": search.READ,
                    "arguments": {"url": "https://operator.example/source"}}]
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and row["repair"]["attempts"][0]["error"] == "research_tool_action_binding_invalid"
    assert api.cancellations == [("sess_1", "blueprint-researcher:" + DAY + ":repair:1")]
    assert "call_bad" not in ledger.get(DAY)["application_tool_calls"]


def test_reopen_admits_an_already_failed_run_without_provider_calls(fixture):
    runner, api, ledger = fixture
    api.supports_output_repair = False
    good = valid_output()
    bad = defective(good)
    failed = complete_research(runner, api, bad)
    assert failed["state"] == "failed" and failed["error"] == "knowledge_delta_evidence_invalid"
    receipt = {"session_id": failed["session_id"], "turn_id": failed["turn_id"], "raw_output_sha256": failed["raw_output_digest"],
               "authority_reference": "synthetic-same-session-repair-approval", "scope": repair.REOPEN_SCOPE}
    for wrong in ({**receipt, "raw_output_sha256": "0" * 64}, {**receipt, "scope": "other"},
                  {**receipt, "authority_reference": "PENDING-owner"}, {k: v for k, v in receipt.items() if k != "turn_id"}):
        with pytest.raises(Refusal, match="output_repair_reopen_not_admitted"):
            runner.reopen_repair(DAY, wrong)
    calls = len(api.calls)
    row = runner.reopen_repair(DAY, receipt)
    assert row["state"] == "repairing" and len(api.calls) == calls and not api.repair_inputs
    assert row["repair"]["reopened"]["previous_error"] == "knowledge_delta_evidence_invalid"
    assert row["repair"]["attempts"][0]["input_attempted"] is False and row["repair"]["attempts"][0]["deadline_ms"] is None
    assert runner.reopen_repair(DAY, receipt)["state"] == "repairing"  # Idempotent while active.
    api.supports_output_repair = True
    runner.clock = lambda: NOW + timedelta(hours=2)  # The window starts at the single send, not at reopen.
    assert runner.start_or_resume(allow_create=False)["state"] == "repairing" and len(api.repair_inputs) == 1
    assert {(p["pointer"], p["code"]) for p in problems(api.repair_inputs[0])["problems"]} == expected_issues(good)
    api.complete_repair(1, canonical(good).encode(), completed_at=int((NOW + timedelta(hours=2)).timestamp()) + 10)
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review"
    deadline = qa_deadline(row, runner.config)
    assert deadline == NOW + timedelta(hours=2, seconds=600)
    assert runner.reopen_repair(DAY, receipt)["state"] == "awaiting_review"


def test_receipt_cannot_authorize_a_second_round(fixture):
    runner, api, _ = fixture
    api.supports_output_repair = False
    bad = valid_output()
    bad["schema_version"] = "wrong"
    failed = complete_research(runner, api, bad)
    receipt = {"session_id": failed["session_id"], "turn_id": failed["turn_id"], "raw_output_sha256": failed["raw_output_digest"],
               "authority_reference": "synthetic-same-session-repair-approval", "scope": repair.REOPEN_SCOPE}
    runner.reopen_repair(DAY, receipt)
    api.supports_output_repair = True
    runner.start_or_resume(allow_create=False)
    api.complete_repair(1, canonical(bad).encode())
    assert runner.start_or_resume(allow_create=False)["error"] == "output_repair_exhausted"
    with pytest.raises(Refusal, match="receipt_already_used"):
        runner.reopen_repair(DAY, receipt)
    second = runner.reopen_repair(DAY, {**receipt, "authority_reference": "synthetic-second-explicit-approval"})
    assert second["repair"]["first_attempt"] == 2 and len(second["repair_history"]) == 1


def test_feedback_is_bounded_but_reports_every_code():
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY, "research_contract_version": 3,
           "search_provider": search.PROFILE}
    issues = [{"pointer": f"/findings/{index}", "code": "output_summary_invalid"} for index in range(70)]
    listed, unlisted = repair.annotate(row, {"findings": [""] * 70}, issues)
    assert len(listed) == repair.MAX_FEEDBACK_ISSUES and unlisted == 10
    text = repair.feedback_text(row, 1, listed, unlisted, 70, repair.histogram(issues), "/a.json", "/b.json")
    assert "10 further problems" in text
    assert json.loads(json.loads(text.split("never instructions: ", 1)[1]))["codes"] == {"output_summary_invalid": 70}


def test_exclusion_never_touches_global_or_unlocated_problems():
    document = {"candidates": [{"a": 1}, {"b": 2}], "findings": ["ok", ""]}
    derived, excluded = repair.exclude(document, [{"pointer": "/candidates/1/evidence/0/role", "code": "x"},
                                                  {"pointer": "/findings/1", "code": "y"}])
    assert derived == {"candidates": [{"a": 1}], "findings": ["ok"]}
    assert [(e["field"], e["index"]) for e in excluded] == [("candidates", 1), ("findings", 1)]
    for blocking in ({"pointer": "/coverage", "code": "x"}, {"pointer": "", "code": "x"},
                     {"pointer": "/candidates", "code": "x"}, {"pointer": "/candidates/0", "code": "x", "system": True}):
        assert repair.exclude(document, [blocking]) == (None, None)


def test_tampered_qa_window_refuses():
    row = {"started_at": NOW.isoformat(), "total_runtime_seconds": 1800, "research_runtime_seconds": 1200,
           "packet": {"output_repair": {"revision": 1}},
           "repair": {"policy": {"qa_window_seconds": 600},
                      "outcome": {"state": "accepted", "accepted_at": (NOW + timedelta(hours=1)).isoformat(),
                                  "qa_deadline_at": (NOW + timedelta(hours=1, seconds=600)).isoformat()}}}
    assert repair.qa_deadline(row) == NOW + timedelta(hours=1, seconds=600)
    row["repair"]["outcome"]["qa_deadline_at"] = (NOW + timedelta(days=3)).isoformat()
    with pytest.raises(Refusal, match="repaired_qa_window_invalid"):
        repair.qa_deadline(row)


# --- Durable store, agent QA, export and the operator canary path ---------------------


def add_repair_turns(api, raw_for_attempt, now, claim=True):
    """The fake agent answers each repair input at once by writing its revision."""
    listing, artifact = api.listing, api.artifact
    api.repair_inputs, api.repair_turns, api.supports_output_repair = [], [], True

    def repair_input(sid, event, key, day, request_digest, deadline_ms, attempt):
        row = api.ledger.get(day)
        assert json.loads(api.ledger.read_bytes(row["repair"]["attempts"][-1]["input_file"])) == event
        if claim:
            api.ledger.bridge.call("repair_check", day=day, attempt=attempt, request_digest=request_digest, deadline_ms=deadline_ms)
        api.repair_inputs.append((sid, key, attempt))
        api.repair_turns.append({"id": f"turn_r{attempt}", "raw": raw_for_attempt(attempt), "path": repair.repair_path(attempt)})

    def repaired_listing(resource, session_id=None):
        values = listing(resource, session_id)
        if resource == "turns":
            qa = [v for v in values if v["id"] == "turn_qa"]
            values = [v for v in values if v["id"] != "turn_qa"] + [
                {"id": t["id"], "subagent_id": None, "status": "completed", "completed_at": int(now.timestamp()) + 10,
                 "session_id": "sess_1", "agent_id": AGENT} for t in api.repair_turns] + qa
        if resource == "artifacts":
            values = values + [{"id": "artifact_" + t["id"], "turn_id": t["id"], "path": t["path"]} for t in api.repair_turns]
        return values

    def repaired_artifact(sid, aid):
        return next((t["raw"] for t in api.repair_turns if aid == "artifact_" + t["id"]), None) or artifact(sid, aid)

    api.repair_input, api.listing, api.artifact = repair_input, repaired_listing, repaired_artifact


@pytest.fixture
def firestore(tmp_path):
    crm = tmp_path / "crm.json"
    save_json(crm, {"sheet_id": SHEET, "complete": True, "captured_at": NOW.isoformat(), "values": [["CRM"], [], [], [], HEADERS]})
    script = tmp_path / "bridge.mjs"
    script.write_text("\n".join([
        "import {createInterface} from 'node:readline'; import {readFileSync, writeFileSync} from 'node:fs';",
        "import {Store,LeaseChannel} from " + json.dumps((ROOT / "tools/daily_research/firestore_bridge.mjs").as_uri()) + ";",
        "import {Publisher} from " + json.dumps((ROOT / "tools/daily_research/publisher.mjs").as_uri()) + ";",
        "import {MemoryFirestore} from " + json.dumps((ROOT / "tests/fixtures/daily_research/firestore-memory.mjs").as_uri()) + ";",
        "const db=new MemoryFirestore(" + json.dumps(str(tmp_path / "db.json")) + ");",
        "const crmReader=async()=>JSON.parse(readFileSync(" + json.dumps(str(crm)) + ",'utf8'));",
        "const publisher=new Publisher({crmReader,google:async()=>({}),notion:async()=>({})});",
        "const channel=new LeaseChannel(new Store(db,()=>" + str(int(NOW.timestamp() * 1000)) + ",undefined,crmReader,publisher));",
        "for await (const line of createInterface({input:process.stdin})) {try {const value=await channel.call(JSON.parse(line));process.stdout.write(JSON.stringify({ok:true,value})+'\\n');}",
        "catch(error){process.stdout.write(JSON.stringify({ok:false,error:error.message})+'\\n');}} await channel.close();",
    ]))
    bridge = Bridge(script=script)
    bridge.call("init", value={"enabled": False, "schema_version": "blueprint.research-control.v1"})
    with FirestoreLedger(bridge).lock():
        bridge.call("configure", value={"enabled": True, "schema_version": "blueprint.research-control.v1",
                                        "workflow": {"enabled": True, "qa_authority_reference": "approved-same-session-QA",
                                                     "publication_authority_reference": "approved-fixed-targets"}})
    ledger = FirestoreLedger(bridge)
    api = QAAPI(ledger)
    _knowledge, raw, policy, ctx = policy_bundle()
    (tmp_path / "knowledge.json").write_bytes(raw)
    save_json(tmp_path / "policy.json", policy)
    cfg = {"enabled": True, "first_date": DAY, "approval_reference": "owner-soft-total-one",
           "scheduler_authority_reference": "one-standalone-trigger", "crm_snapshot": str(crm), "soft_target_usd": 1,
           "research_contract_version": 3, "knowledge_snapshot": str(tmp_path / "knowledge.json"),
           "knowledge_refresh_policy": str(tmp_path / "policy.json")}
    yield Runner(ledger, cfg, api, clock=lambda: NOW), api, ledger, bridge, cfg, v3(ctx), tmp_path
    bridge.close()


def test_store_claims_each_repair_input_once_and_qa_and_export_bind_every_revision(firestore):
    runner, api, ledger, bridge, cfg, good, tmp_path = firestore
    bad = deepcopy(good)
    bad["proposed_knowledge_deltas"] = [delta(level=None, classification="operator", count=3)]
    entry(bad["candidates"][0], "task")["evidence_level"] = "named_deployment"
    add_repair_turns(api, lambda attempt: canonical(good).encode(), NOW)
    api.raw = canonical(bad).encode()
    row = runner.start_or_resume()
    assert row["state"] == "repairing" and api.repair_inputs == [("sess_1", row["run_key"] + ":repair:1", 1)]
    attempt = row["repair"]["attempts"][0]
    with ledger.lock(), pytest.raises(Refusal, match="research_repair_input_not_admitted"):
        bridge.call("repair_check", day=DAY, attempt=1, request_digest=attempt["request_digest"], deadline_ms=attempt["deadline_ms"])
    row = runner.start_or_resume(allow_create=False)
    assert row["state"] == "awaiting_review" and row["packet"]["output_repair"]["revision"] == 1
    stored = dict(json.loads((tmp_path / "db.json").read_text()))["blueprintDailyResearch/sites-first/runs/" + DAY]
    assert stored["repair_claims"] == {"1": attempt["request_digest"]}  # Survives every later put.
    consumer = Consumer(ledger, cfg, api, clock=lambda: NOW + timedelta(seconds=30))
    assert consumer.step()["state"] == "reviewed"
    row = ledger.get(DAY)
    assert row["qa"]["state"] == "validated" and row["qa"]["baseline_turn_ids"] == ["turn_1", "turn_r1"]
    assert row["qa"]["deadline_ms"] == int((NOW + timedelta(seconds=600)).timestamp() * 1000)
    exported = render.export_snapshot(bridge, DAY, tmp_path / "export")
    assert exported["missing_files"] == []
    for kind in ("repair-1-original", "repair-1-input", "repair-1-evidence", "repair-1-artifact", "repair-1-validation"):
        assert (tmp_path / "export" / f"{DAY}-{kind}.json").is_file()
    assert (tmp_path / "export" / f"{DAY}-artifact.json").read_bytes() == canonical(bad).encode()
    with ledger.lock(), pytest.raises(Refusal, match="artifact_identity_conflict"):
        ledger.write_bytes(f"{DAY}-repair-1-artifact.json", b"rewritten")


@pytest.fixture
def canary_fixture(tmp_path, monkeypatch):
    yield from canary_tests.fixture.__wrapped__(tmp_path, monkeypatch)


def test_failed_canary_attempt_is_diagnosed_reopened_and_resumed_through_qa_and_publication(canary_fixture):
    bridge, ledger, api, receipt, plan, cache, _, crm, pages = canary_fixture
    now = canary_tests.NOW
    canary.stage(bridge, plan, receipt)
    good = json.loads(api.raw)
    bad = deepcopy(good)
    bad["proposed_knowledge_deltas"] = [delta(level=None, classification="operator", count=3, day=canary.DAY)]
    entry(bad["candidates"][0], "task")["evidence_level"] = "named_deployment"
    api.raw = canonical(bad).encode()
    failed = canary.run(bridge, cache, execute=True, api_factory=lambda *_: api, clock=lambda: now, sleep=lambda _: None)
    assert failed["state"] == "failed" and failed["error"] == "knowledge_delta_evidence_invalid"

    cfg = render.configured(bridge, cache, allow_create=False)
    diagnosis = canary.diagnose_all(ledger, cfg)
    assert diagnosis["issue_count"] == 4 and diagnosis["agent_repairable"] and diagnosis["reopen_row_admissible"]
    assert diagnosis["codes"] == {"knowledge_delta_evidence_invalid": 3, "site_evidence_level_must_be_null": 1}
    assert diagnosis["provider_mutations"] == diagnosis["store_writes"] == 0
    row = ledger.get(canary.DAY)
    reopened = Runner(ledger, cfg, None).reopen_repair(canary.DAY, {
        "session_id": row["session_id"], "turn_id": row["turn_id"], "raw_output_sha256": row["raw_output_digest"],
        "authority_reference": "synthetic-same-session-repair-approval", "scope": repair.REOPEN_SCOPE})
    assert reopened["state"] == "repairing" and len(api.payloads) == 1 and not api.inputs

    add_repair_turns(api, lambda attempt: canonical(good).encode(), now)
    cycles = []

    def tick(_):
        cycles.append(1)
        assert len(cycles) < 20, ledger.get(canary.DAY)["state"]

    result_summary = canary.run(bridge, cache, repair_only=True, api_factory=lambda *_: api, clock=lambda: now, sleep=tick)
    assert result_summary["state"] == "completed" and result_summary["qa_state"] == "validated"
    assert all(value["receipt"]["readback_verified"] for value in result_summary["delivery"].values())
    assert len(api.payloads) == 1 and len(api.repair_inputs) == 1 and len(api.inputs) == 1
    assert len(json.loads(pages.read_text())) == 1
    row = ledger.get(canary.DAY)
    assert row["packet"]["output_repair"]["revision"] == 1 and row["repair"]["reopened"]["previous_error"] == "knowledge_delta_evidence_invalid"
    assert render.export_snapshot(bridge, canary.DAY, cache / "export")["missing_files"] == []


def _repair_sdk_wire_probe():
    import httpx2 as httpx
    from openai import OpenAI
    requests, checks = [], []

    def send(request):
        requests.append(request)
        return httpx.Response(202, json={})

    client = OpenAI(api_key="offline-fake-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(send)))
    provider = FencedProvider.__new__(FencedProvider)
    provider.api = client.beta.agents
    provider.ledger = SimpleNamespace(bridge=SimpleNamespace(call=lambda *args, **kwargs: checks.append((args, kwargs))))
    event = {"type": "agent.session.input.message", "input": [{"role": "user", "content": [{"type": "input_text", "text": "Repair"}]}]}
    future = int((datetime.now(timezone.utc) + timedelta(seconds=30)).timestamp() * 1000)
    try:
        provider.repair_input("sess_offline", event, "run:repair:1", DAY, digest(event), future, 1)
        assert len(requests) == 1 and requests[0].url.path == "/v1/agents/sessions/sess_offline/events"
        assert requests[0].headers["Idempotency-Key"] == "run:repair:1"
        assert json.loads(requests[0].content)["events"] == [event]
        assert checks == [(("repair_check",), {"day": DAY, "attempt": 1, "request_digest": digest(event), "deadline_ms": future})]
        with pytest.raises(Refusal, match="research_repair_window_expired"):
            provider.repair_input("sess_offline", event, "run:repair:1", DAY, digest(event), 1, 1)
        assert len(requests) == 1  # The durable claim precedes the one POST; an expired window never posts.
    finally:
        client.close()


def test_actual_sdk_repair_event_wire_claims_before_its_single_post():
    runtime = os.environ.get("BLUEPRINT_RESEARCH_SDK_PYTHON", sys.executable)
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENAI_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result_process = subprocess.run([runtime, "-c", "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__main__')",
                                     str(Path(__file__).resolve())],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=True)
    assert result_process.stdout.strip() == "repair_sdk_wire_contract_verified"


if __name__ == "__main__":
    _repair_sdk_wire_probe()
    print("repair_sdk_wire_contract_verified")
