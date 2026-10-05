"""Exact located feedback, exclusion instead of blocking, approval as data.

Hermetic: fake provider and in-memory Firestore only. The frozen copy below is
main's strict gate at 73d3be8d6, with the explicit October4 v3 discovery-only
exceptions (optional capability and byte-based inventory retention); the refactored
gate must agree on admission/refusal for every corruption so feedback never
loosens QA input. New discovery audit fields are compared through the frozen
payload projection; full raw retention is covered by the verification tests.
"""
import json
import random
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import pytest

from tests import test_daily_research_consumer as consumer_tests
from tests.test_daily_research_adaptive import result
from tests.test_daily_research_consumer import consumer_setup
from tests.test_daily_research_knowledge import context, v2
from tests.test_daily_research_runner import AGENT, DAY, NOW, output
from tools.daily_research import (
    contracts,
    discovery,
    freshness,
    knowledge,
    recovery,
    render,
)
from tools.daily_research import runner as runner_module
from tools.daily_research.contracts import checked_day, lookup
from tools.daily_research.knowledge import (  # noqa: F401
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
from tools.daily_research.runner import (
    Refusal,
    canonical,
    keys,
    output_issues,
    public_url,
    validate_output,
)

# --- Frozen main gate (73d3be8d6) ------------------------------------------------------

def legacy_evidence(value, day, context, observed_at=None, *, policy=None):
    if value["role"] == "capability":
        require(value["evidence_level"] in LEVELS, "evidence_level_invalid")
        require(value["evidence_level"] != "unknown", "unsupported_evidence_level")
    elif value["role"] == "background":
        # Ordinary company/operator context is not a robot maturity claim.
        # Background never supplies the required positive capability coverage;
        # snapshot context still has its exact reviewed fact binding below.
        require(value["evidence_level"] is None or value["evidence_level"] in LEVELS, "evidence_level_invalid")
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
            # v3's scope describes this proposed claim; it does not approve it
            # or rewrite the reviewed snapshot. Older scope-free deltas remain valid.
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
                knowledge.require(isinstance(refresh_policy, dict), "refresh_policy_context_missing")
                freshness.validate_context(knowledge_context, refresh_policy)
                knowledge.require(output.get("refresh_policy_hash") == refresh_policy["policy_hash"], "output_refresh_policy_binding_invalid")
            except knowledge.SnapshotError as exc:
                raise Refusal(str(exc)) from None
        try:
            legacy_deltas(output.get("proposed_knowledge_deltas"), run_date, knowledge_context, observed_at, contract_version)
        except knowledge.SnapshotError as exc:
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
    limit = None if contract_version == 3 else 3
    if output["checked_date"] != run_date or not isinstance(output["candidates"], list) or limit is not None and len(output["candidates"]) > limit:
        raise Refusal("output_date_or_count_invalid")
    for field in ("findings", "blockers", "proposed_next_actions"):
        if (not isinstance(output[field], list)
                or (contract_version == 2 and len(output[field]) > 20)
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
        if not isinstance(c["evidence"], list) or not (2 if contract_version == 3 else 3) <= len(c["evidence"]) <= 12:
            raise Refusal("candidate_evidence_required")
        roles = set()
        for e in c["evidence"]:
            evidence_fields = {"claim", "url", "publisher", "source_date", "checked_date", "classification", "claim_kind", "role", "quote"}
            if contract_version in {2, 3}:
                evidence_fields |= contracts.EVIDENCE_V2
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
                    published = knowledge.calendar_date(e["source_date"]) if contract_version in {2, 3} else date.fromisoformat(e["source_date"])
                except knowledge.SnapshotError as exc:
                    raise Refusal(str(exc)) from None
                if published > date.fromisoformat(run_date):
                    raise Refusal("source_date_in_future")
            if contract_version in {2, 3}:
                try:
                    legacy_evidence(e, run_date, knowledge_context, observed_at, policy=refresh_policy if contract_version == 3 else None)
                except knowledge.SnapshotError as exc:
                    raise Refusal(str(exc)) from None
            roles.add(e["role"])
        if (not {"task", "geography"} <= roles if contract_version == 3 else roles != {"task", "capability", "geography"}):
            raise Refusal("task_geography_evidence_required" if contract_version == 3 else "task_capability_geography_evidence_required")
        if contract_version == 3 and "capability" not in roles and c["potential_robot_match"] != "unknown":
            raise Refusal("unsupported_robot_match_must_remain_unknown")
        operator_task_sources = [e for e in c["evidence"] if e["role"] == "task" and e["classification"] == "operator"]
        if not operator_task_sources:
            raise Refusal("operator_task_source_required")
        affiliation_review = not any(public_url(e["url"]) == public_url(c["organization_url"])
                                     for e in operator_task_sources)
        identities = keys(c)
        if identities & known:
            duplicates.append({"organization": c["organization"], "site": c["site"], "reason": "matching_site_task"})
        else:
            accepted.append({**c, "candidate_key": min(identities), "identity_keys": sorted(identities),
                             "operator_affiliation_qa_required": affiliation_review})
            known.update(identities)
    return accepted, duplicates


# --- Corruption corpus -------------------------------------------------------------------


def proposal(level="vendor_claim", classification="vendor", count=1, day=DAY):
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
    document["proposed_knowledge_deltas"] = [proposal()]
    background = deepcopy(document["candidates"][0]["evidence"][0])
    background.update(role="background", evidence_level=None)
    document["candidates"][0]["evidence"].append(background)
    return document, {"contract_version": 3, "knowledge_context": ctx, "observed_at": NOW, "refresh_policy": policy}


BASES = {1: base_v1, 2: base_v2, 3: base_v3}
MISSING = object()
REPLACEMENTS = (MISSING, None, "", 7, [], {}, ["x"], "x" * 2001, "2099-01-01", "qualified", "vendor", "operator",
                "background", "task", "snapshot", "unknown", "http://10.0.0.1/a", "https://[", "https://other.example/x", True)


def nodes(value, pointer=""):
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, child in children:
        yield f"{pointer}/{key}"
        yield from nodes(child, f"{pointer}/{key}")


def mutate(document, pointer, replacement):
    changed = deepcopy(document)
    *parents, last = pointer.split("/")[1:]
    target = changed
    for part in parents:
        target = target[int(part)] if isinstance(target, list) else target[part]
    key = int(last) if isinstance(target, list) else last
    if replacement is MISSING:
        del target[key]
    elif replacement == "__extra__":
        if isinstance(target[key], dict):
            target[key] = {**target[key], "unexpected_field": 1}
    else:
        target[key] = deepcopy(replacement)
    return changed


def outcome(function):
    try:
        accepted, duplicates = function()
        accepted = [{k: v for k, v in c.items() if k != "discovery_index"} for c in accepted]
        duplicates = [{k: c[k] for k in ("organization", "site", "reason")} for c in duplicates]
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
    return current[0]


@pytest.fixture
def fixed_policy_context(monkeypatch):
    """The corpus mutates only the report; the trusted context/policy pair is
    constant, so its pure binding check is computed once for BOTH gates."""
    checked, original = set(), freshness.validate_context

    def once(context, policy):
        if (id(context), id(policy)) not in checked:
            original(context, policy)
            checked.add((id(context), id(policy)))
    monkeypatch.setattr(freshness, "validate_context", once)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_gate_is_unchanged_and_feedback_agrees_for_every_single_corruption(version, fixed_policy_context):
    document, options = BASES[version]()
    assert compare(document, options) == "ok"
    seen = {"ok": 0, "refusal": 0, "error": 0}
    for pointer in list(nodes(document)):
        for replacement in (*REPLACEMENTS, "__extra__"):
            seen[compare(mutate(document, pointer, replacement), options)] += 1
    assert seen["refusal"] > 100


@pytest.mark.parametrize("version", [2, 3])
def test_gate_is_unchanged_for_compound_corruptions(version, fixed_policy_context):
    document, options = BASES[version]()
    rng = random.Random(20261002 + version)
    for _ in range(300):
        mutated = document
        for _step in range(rng.randint(2, 4)):
            available = list(nodes(mutated)) or ["/candidates"]
            mutated = mutate(mutated, rng.choice(available), rng.choice((*REPLACEMENTS, "__extra__")))
        compare(mutated, options)


def entry(candidate, role):
    return next(e for e in candidate["evidence"] if e["role"] == role and e["origin"] == "live")


def test_feedback_lists_every_located_failure_at_once_without_cascades():
    document, ctx, policy = result()
    document["coverage"].update(defined_run_scope=["Synthetic scope"], unresolved_promising_branches=[],
                                completion_state="coverage_complete")
    row = {"date": DAY, "research_contract_version": 3, "knowledge_context": ctx, "refresh_policy": policy,
           "discovery_profile": "adaptive-sites-v1", "search_provider": recovery.search.PROFILE}
    assert recovery.validation_feedback(document, row, set(), NOW) == []
    bad = deepcopy(document)
    bad["proposed_knowledge_deltas"] = [proposal(level=None, classification="operator", count=3)]
    entry(bad["candidates"][0], "task")["evidence_level"] = "named_deployment"
    entry(bad["candidates"][1], "geography").update(classification="vendor", claim_kind="fact")
    entry(bad["candidates"][2], "task")["source_date"] = "2099-01-01"
    entry(bad["candidates"][3], "geography")["source_checked_at"] = "2026-09-29"
    entry(bad["candidates"][4], "task")["classification"] = "unknown-kind"
    bad["candidates"][5]["qualification_status"] = "qualified"
    bad["candidates"][6]["unknowns"] = []
    bad["coverage"]["unresolved_promising_branches"] = ["Synthetic unresolved branch"]
    bad["findings"].append("")
    bad["candidates"][7]["contact_email"] = "invented@example.com"
    entry(bad["candidates"][8], "task")["assertion_scope"] = "forever"
    bad["candidates"][9]["organization_url"] = "https://["
    index = {n: {role: bad["candidates"][n]["evidence"].index(entry(bad["candidates"][n], role))
                 for role in ("task", "geography")} for n in range(10) if n != 7}
    feedback = recovery.validation_feedback(bad, row, set(), NOW)
    assert {(i["path"], i["reason"]) for i in feedback} == {
        *((f"/proposed_knowledge_deltas/0/evidence/{n}/evidence_level", "knowledge_delta_evidence_invalid") for n in range(3)),
        (f"/candidates/0/evidence/{index[0]['task']}/evidence_level", "site_evidence_level_must_be_null"),
        (f"/candidates/1/evidence/{index[1]['geography']}/claim_kind", "vendor_claim_presented_as_fact"),
        (f"/candidates/2/evidence/{index[2]['task']}/source_date", "source_date_in_future"),
        (f"/candidates/3/evidence/{index[3]['geography']}/checked_date", "evidence_date_integrity_invalid"),
        (f"/candidates/4/evidence/{index[4]['task']}/classification", "evidence_field_invalid"),
        ("/candidates/5/qualification_status", "candidate_claim_ceiling_invalid"),
        ("/candidates/6/unknowns", "candidate_unknowns_required"),
        ("/coverage", "discovery_completion_has_unresolved_branches"),
        (f"/findings/{len(document['findings'])}", "output_summary_invalid"),
        ("/candidates/7", "candidate_schema_invalid"),
        (f"/candidates/8/evidence/{index[8]['task']}/assertion_scope", "evidence_assertion_scope_invalid"),
        ("/candidates/9/organization_url", "source_url_invalid")}
    assert all(i["allowed_semantics"] for i in feedback)
    flagged = next(i for i in feedback if i["path"].startswith("/candidates/2/evidence/"))
    assert flagged["evidence_reference"]["url"] == entry(bad["candidates"][2], "task")["url"]
    with pytest.raises(Refusal, match="^knowledge_delta_evidence_invalid$"):
        validate_output(bad, DAY, set(), contract_version=3, knowledge_context=ctx, refresh_policy=policy, observed_at=NOW)


def test_exclusion_drops_only_items_whose_every_failure_is_inside_them():
    document = {"candidates": [{"organization": "Kept"}, {"organization": "Bad", "site": "S", "task": "T"}],
                "proposed_knowledge_deltas": [{"reason": "x"}], "findings": ["ok", ""]}
    feedback = [{"path": "/candidates/1/evidence/0/role", "reason": "evidence_field_invalid"},
                {"path": "/proposed_knowledge_deltas/0/evidence/0/evidence_level", "reason": "knowledge_delta_evidence_invalid"},
                {"path": "/findings/1", "reason": "output_summary_invalid"}]
    derived, excluded = recovery.exclude_located_items(document, feedback)
    assert derived == {"candidates": [{"organization": "Kept"}], "proposed_knowledge_deltas": [], "findings": ["ok"]}
    assert [(e["field"], e["index"]) for e in excluded] == [("candidates", 1), ("proposed_knowledge_deltas", 0), ("findings", 1)]
    assert excluded[0]["identity"]["organization"] == "Bad" and document["findings"] == ["ok", ""]
    for blocking in ("/coverage", "/", "/candidates", "/schema_version"):
        assert recovery.exclude_located_items(document, [{"path": blocking, "reason": "x"}]) == (None, None)


def test_canary_authority_is_data_bound_to_the_admitted_baseline():
    # Every admitted row pins total_runtime_seconds; retained baseline rows pinned 1800.
    row = {"session_id": "sess", "turn_id": "turn", "raw_output_digest": "a" * 64, "total_runtime_seconds": 1800,
           "canary": {"baseline": {"baseline_id": "baseline-any", "soft_total_usd": 40, "authority_reference": "budget-approval"}}}
    request = {"scope": "same-session-validation-repair-and-qa-no-outreach", "session_id": "sess", "root_turn_id": "turn",
               "raw_output_sha256": "a" * 64, "authority_reference": "any-newly-recorded-approval", "baseline_id": "baseline-any",
               "soft_total_usd": 40, "budget_authority_reference": "budget-approval"}
    row["validation_repair_authority"] = {"started_at": NOW.isoformat(), "duration_seconds": 1800, "request": request}
    assert recovery.repair_deadline(row) == NOW + timedelta(seconds=1800)
    for change in ({"authority_reference": "PENDING-owner"}, {"authority_reference": " "}, {"authority_reference": None},
                   {"baseline_id": "baseline-20261002"}, {"soft_total_usd": 25}, {"budget_authority_reference": "other"},
                   {"session_id": "other"}, {"raw_output_sha256": "b" * 64}):
        changed = deepcopy(row)
        changed["validation_repair_authority"]["request"].update(change)
        with pytest.raises(Refusal, match="validation_repair_authority_or_binding_invalid"):
            recovery.repair_deadline(changed)
    unbound = deepcopy(row)
    unbound["canary"] = {}
    unbound["validation_repair_authority"]["request"].update(baseline_id=None, soft_total_usd=None, budget_authority_reference=None)
    with pytest.raises(Refusal, match="validation_repair_authority_or_binding_invalid"):
        recovery.repair_deadline(unbound)


def repair_authority_row(total, *, kind="workflow", duration=None, authority_started=NOW):
    row = {"started_at": NOW.isoformat(), "session_id": "sess", "turn_id": "turn", "raw_output_digest": "a" * 64,
           "total_runtime_seconds": total}
    request = {"scope": "same-session-validation-repair-and-qa-no-outreach", "session_id": "sess",
               "root_turn_id": "turn", "raw_output_sha256": "a" * 64, "authority_reference": "recorded-approval"}
    authority = {"started_at": authority_started.isoformat(), "duration_seconds": total if duration is None else duration,
                 "request": request}
    if kind == "workflow":
        authority["kind"] = "workflow"
    else:
        row["canary"] = {"baseline": {"baseline_id": "baseline-any", "soft_total_usd": 40, "authority_reference": "budget"}}
        request.update(baseline_id="baseline-any", soft_total_usd=40, budget_authority_reference="budget")
    row["validation_repair_authority"] = authority
    return row


@pytest.mark.parametrize("kind", ["workflow", "canary"])
def test_repair_authority_lasts_exactly_the_rows_own_admitted_total(kind):
    later = NOW + timedelta(minutes=7)
    started = NOW if kind == "workflow" else later  # The canary window starts at its own authorization time.
    for total in (1800, 3600):  # Existing 30-minute rows and owner-approved 60-minute rows.
        row = repair_authority_row(total, kind=kind, authority_started=started)
        assert recovery.repair_deadline(row) == started + timedelta(seconds=total)
    # The authority cannot claim a window other than the row's admitted total.
    for total, duration in ((3600, 1800), (1800, 3600), (3600, 3601), (3600, None)):
        row = repair_authority_row(total, kind=kind, authority_started=started)
        row["validation_repair_authority"]["duration_seconds"] = duration
        with pytest.raises(Refusal, match="^validation_repair_authority_or_binding_invalid$"):
            recovery.repair_deadline(row)
    # A row outside the owner-approved envelope, or without a pinned total, is refused.
    for total in (3601, 0, None, "3600"):
        with pytest.raises(Refusal, match="^validation_repair_authority_or_binding_invalid$"):
            recovery.repair_deadline(repair_authority_row(total, kind=kind, duration=3600, authority_started=started))
    # Every other binding is still checked.
    for change in ({"session_id": "other"}, {"root_turn_id": "other"}, {"raw_output_sha256": "b" * 64},
                   {"scope": "other"}, {"authority_reference": "PENDING-owner"}):
        row = repair_authority_row(3600, kind=kind, authority_started=started)
        row["validation_repair_authority"]["request"].update(change)
        with pytest.raises(Refusal, match="^validation_repair_authority_or_binding_invalid$"):
            recovery.repair_deadline(row)
    if kind == "workflow":
        moved = repair_authority_row(3600, authority_started=later)
        with pytest.raises(Refusal, match="^validation_repair_authority_or_binding_invalid$"):
            recovery.repair_deadline(moved)


# --- Daily worker: exclusion instead of blocking -----------------------------------------


def correcting_agent(api, ledger, bridge, revision, *, completed_at=None):
    """Each admitted correction input yields one terminal turn writing ``revision``."""
    turns, calls = [], []
    listing, artifact = api.listing, api.artifact

    def repair_input(sid, event, key, day, request_digest, deadline_ms):
        current = ledger.get(day)["validation_repairs"][-1]
        assert json.loads(ledger.read_bytes(current["input_file"])) == event
        bridge.call("repair_check", day=day, request_digest=request_digest, deadline_ms=deadline_ms)
        calls.append(key)
        turns.append({"id": f"turn_correction_{len(turns) + 1}", "session_id": sid, "agent_id": AGENT, "subagent_id": None,
                      "status": "completed", "completed_at": completed_at or int((NOW + timedelta(seconds=25)).timestamp())})

    def values(resource, sid=None):
        found = listing(resource, sid)
        if resource == "turns":
            found.extend(deepcopy(turns))
        if resource == "artifacts":
            found.extend({"id": "artifact_" + t["id"], "turn_id": t["id"], "path": recovery.REPAIR_PATH} for t in turns)
        return found

    api.repair_input, api.listing = repair_input, values
    api.artifact = lambda sid, aid: revision if aid.startswith("artifact_turn_correction") else artifact(sid, aid)
    return calls


def run_daily_worker(consumer, api, bridge, tmp_path, monkeypatch):
    class FixedDatetime:
        @staticmethod
        def now(_zone):
            return consumer.clock()
    monkeypatch.setattr(render, "datetime", FixedDatetime)
    monkeypatch.setattr(render, "configured", lambda *_: consumer.config)
    monkeypatch.setattr(render, "Consumer", lambda *a, **kw: type(consumer)(*a, **kw, clock=consumer.clock))
    monkeypatch.setattr(render.time, "sleep", lambda _: None)
    return render.consume_workflow(bridge, tmp_path, api_factory=lambda *_: api)


def test_sixty_minute_daily_row_repairs_qa_and_publishes_after_the_old_envelope(tmp_path, monkeypatch):
    generator = consumer_setup(tmp_path, failed=True, envelope=(3600, 900))
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        original = ledger.get(DAY)
        assert original["state"] == "failed" and original["error"] == "knowledge_delta_evidence_invalid"
        assert (original["total_runtime_seconds"], original["research_runtime_seconds"]) == (3600, 2700)
        repaired = json.loads(ledger.read_bytes(DAY + "-artifact.json"))
        repaired["proposed_knowledge_deltas"] = []
        # Repair, QA and publication all run after the old 1800-second envelope has closed.
        after_old = NOW + timedelta(seconds=1900)
        consumer.clock = lambda: after_old
        bridge.call("test_clock", now=int(after_old.timestamp() * 1000))
        # Observe on the test clock: this row's own remaining window, not wall time.
        observed = []
        def observation(row, cfg, phase):
            observed.append(runner_module.observation_seconds(row, cfg, phase, consumer.clock()))
            return observed[-1]
        monkeypatch.setattr(render, "observation_seconds", observation)
        calls = correcting_agent(api, ledger, bridge, canonical(repaired).encode(),
                                 completed_at=int((NOW + timedelta(seconds=1950)).timestamp()))
        assert run_daily_worker(consumer, api, bridge, tmp_path, monkeypatch)["state"] == "completed"
        assert observed and observed[0] == 3600 - 1900 + 30
        final = ledger.get(DAY)
        deadline_ms = int((NOW + timedelta(seconds=3600)).timestamp() * 1000)
        authority = final["validation_repair_authority"]
        assert authority["kind"] == "workflow" and authority["duration_seconds"] == final["total_runtime_seconds"] == 3600
        assert authority["started_at"] == final["started_at"] == original["started_at"]
        assert recovery.repair_deadline(final) == NOW + timedelta(seconds=3600)
        assert final["validation_repairs"][-1]["state"] == "validated"
        assert final["validation_repairs"][-1]["deadline_ms"] == final["qa"]["deadline_ms"] == deadline_ms
        assert len(calls) == len(api.inputs) == len(api.payloads) == 1
        assert all(d["receipt"]["readback_verified"] for d in final["delivery"].values())
    finally:
        generator.close()


def test_repeated_item_failure_excludes_only_that_item_then_qa_publishes(tmp_path, monkeypatch):
    generator = consumer_setup(tmp_path, failed=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        raw = ledger.read_bytes(DAY + "-artifact.json")
        calls = correcting_agent(api, ledger, bridge, raw)  # The agent cannot fix it and repeats itself.
        assert run_daily_worker(consumer, api, bridge, tmp_path, monkeypatch)["state"] == "completed"
        final = ledger.get(DAY)
        assert len(calls) == len(api.inputs) == len(api.payloads) == 1
        assert final["validation_repairs"][-1]["state"] == "no_progress"
        decided = final["validation_repair_outcome"]
        assert decided["state"] == "accepted_with_exclusions" and decided["reason"] == "validation_repair_no_progress"
        assert decided["revision"] == 1 and decided["original_artifact_sha256"] == final["raw_output_digest"]
        [excluded] = decided["excluded"]
        assert (excluded["field"], excluded["index"]) == ("proposed_knowledge_deltas", 0)
        assert excluded["failures"][0]["reason"] == "knowledge_delta_evidence_invalid"
        assert final["packet"]["proposed_knowledge_deltas"] == [] and len(final["packet"]["candidates"]) == 1
        assert final["packet"]["research_exclusions"]["excluded"] == decided["excluded"]
        assert all(d["receipt"]["readback_verified"] for d in final["delivery"].values())
        assert ledger.read_bytes(DAY + "-artifact.json") == raw
        assert not render.export_snapshot(bridge, DAY, tmp_path / "export")["missing_files"]
    finally:
        generator.close()


def test_late_correction_is_retained_but_never_used(tmp_path, monkeypatch):
    generator = consumer_setup(tmp_path, failed=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        repaired = json.loads(ledger.read_bytes(DAY + "-artifact.json"))
        repaired["proposed_knowledge_deltas"] = []
        late = int((NOW + timedelta(seconds=181)).timestamp())  # After the run's pinned 180-second window.
        correcting_agent(api, ledger, bridge, canonical(repaired).encode(), completed_at=late)
        assert run_daily_worker(consumer, api, bridge, tmp_path, monkeypatch)["state"] == "completed"
        final = ledger.get(DAY)
        assert final["validation_repairs"][-1]["error"] == "validation_repair_terminal_guard_failed"
        assert ledger.read_bytes(final["validation_repairs"][-1]["artifact_file"]) == canonical(repaired).encode()
        assert final["validation_repair_outcome"]["revision"] == 0  # The in-window original, minus its located failure.
        assert final["validation_repair_outcome"]["excluded"][0]["field"] == "proposed_knowledge_deltas"
    finally:
        generator.close()


def test_global_failure_stays_blocked_with_complete_feedback(tmp_path, monkeypatch):
    original_v3 = consumer_tests.v3
    monkeypatch.setattr(consumer_tests, "v3", lambda ctx: {**original_v3(ctx), "schema_version": "blueprint.daily-research.v2"})
    generator = consumer_setup(tmp_path, failed=True)
    consumer, api, ledger, bridge, _ = next(generator)
    try:
        correcting_agent(api, ledger, bridge, ledger.read_bytes(DAY + "-artifact.json"))
        blocked = run_daily_worker(consumer, api, bridge, tmp_path, monkeypatch)
        assert blocked["state"] == "validation_repair_blocked"
        assert {i["path"] for i in blocked["feedback"]} >= {
            "/schema_version", "/proposed_knowledge_deltas/0/evidence/0/evidence_level"}
        final = ledger.get(DAY)
        assert final["state"] == "failed" and "validation_repair_outcome" not in final and not api.inputs
        assert bridge.call("work_item") is None  # Blocked runs wait for a person; nothing re-picks them.
    finally:
        generator.close()
