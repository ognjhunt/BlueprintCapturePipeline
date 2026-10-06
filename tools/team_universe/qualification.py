"""Offline, reviewed robot-offering qualification; source identity is not eligibility.

The private manual audit is an assessment, not owner approval or permission to run/contact.
It binds every discovered/screened team to its exact retained result and evidence bytes.
No model, provider, page read, CRM mutation or semantic keyword classifier is used here.
"""
import json
from datetime import date, datetime, timezone

from tools.daily_research import site_screen as ss
from tools.team_universe import universe as tu

AUDIT = "blueprint.team-manual-eligibility-audit.v1"
RULE = "blueprint.team-qualification-rule.v1"
POSITIVE = frozenset({"physical_robot_task", "embodied_control_task", "robot_integrator_task"})
CLASSES = POSITIVE | {"physical_robot_task_fit_pending", "embodied_control_task_pending", "adjacent_reference",
                      "insufficient_evidence", "unscreened"}
ROOT_KEYS = frozenset({"schema_version", "team_list_sha256", "decisions_sha256", "auditor_script_sha256",
                       "audited_at", "auditor_reference", "scope_sha256", "teams"})
ROW_KEYS = frozenset({"team_key", "domain", "discovery_sha256", "run_id", "result_sha256", "evidence_sha256", "capability_class",
                      "manual_current_task_fit", "capability_reason", "capability_evidence",
                      "blueprint_current_evaluation_compatibility", "published_partner_intent",
                      "willingness_to_work_with_blueprint", "promotion_allowed"})
DETAIL_KEYS = frozenset({"identified_hardware", "physical_task", "offering_relation", "capability_as_of",
                         "capability_current_basis", "robot_forms", "task_families", "evaluation"})
PROOF_KEYS = frozenset({"field", "url", "quote", "level", "quote_sha256"})
EVALUATION_KEYS = frozenset({"profile", "working_embodiment", "physical_task", "observation_interface",
                             "action_interface", "runnable_controller", "support_reference", "as_of", "evidence"})
PROFILE = "arm-decision-proof-v1"
MAX_TEAMS = 10_000


def _string(value, limit=2000):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _sha(value):
    return isinstance(value, str) and bool(ss.SHA.fullmatch(value))


def _day(value):
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def binding(team, record):
    """Stable private scope item; null screen hashes explicitly mean no retained screen."""
    subject = team or record
    return {"team_key": subject["site_key"], "domain": subject["domain"],
            "discovery_sha256": ss._sha256(ss.canonical(team).encode()) if team else None,
            **{name: record.get(name) if record else None
               for name in ("run_id", "result_sha256", "evidence_sha256")}}


def scope_manifest(teams, screens):
    """Canonical all-team bindings and digest, suitable for a private auditor's frozen overlay."""
    listed = {team["site_key"]: team for team in teams}
    items = [binding(listed.get(key), screens.get(key)) for key in sorted(set(listed) | set(screens))]
    return {"bindings": items, "sha256": ss._sha256(ss.canonical(items).encode())}


def _proof_shape(proof):
    return (isinstance(proof, dict) and PROOF_KEYS <= set(proof) <= PROOF_KEYS | {"page_sha256", "tool_result_sha256"}
            and proof["field"] in tu.PROOFS and _string(proof["url"])
            and ss.url_key(proof["url"]) is not None and _string(proof["quote"], 12000)
            and proof["level"] in ss.PROVEN and _sha(proof["quote_sha256"])
            and proof["quote_sha256"] == ss._sha256(proof["quote"].encode())
            and ("page_sha256" not in proof or _sha(proof["page_sha256"]))
            and ("tool_result_sha256" not in proof or _sha(proof["tool_result_sha256"])))


def _proofs_shape(proofs):
    return isinstance(proofs, list) and len(proofs) <= 32 and all(_proof_shape(item) for item in proofs)


def _intent_shape(intent):
    return (isinstance(intent, dict) and set(intent) == {"state", "kind", "evidence"}
            and intent["state"] in ("explicit_invitation", "not_established", "unknown")
            and intent["kind"] in ("design_partner", "pilot", "customer", "general", "none")
            and _proofs_shape(intent["evidence"])
            and (intent["state"] != "explicit_invitation" or intent["kind"] != "none"))


def _row_shape(row):
    if not (isinstance(row, dict) and ROW_KEYS <= set(row) <= ROW_KEYS | DETAIL_KEYS):
        return False
    if not (_sha(row["team_key"]) and _string(row["domain"])
            and (row["discovery_sha256"] is None or _sha(row["discovery_sha256"]))
            and row["capability_class"] in CLASSES
            and type(row["manual_current_task_fit"]) is bool and _string(row["capability_reason"], 8000)
            and _proofs_shape(row["capability_evidence"]) and _intent_shape(row["published_partner_intent"])
            and row["blueprint_current_evaluation_compatibility"] in ("not_verified", "unsupported", "supported")
            and row["willingness_to_work_with_blueprint"] == "unknown" and row["promotion_allowed"] is False):
        return False
    empty = row["run_id"] is None and row["result_sha256"] is None and row["evidence_sha256"] is None
    bound = (_string(row["run_id"], 128) and bool(ss.RUN_ID.fullmatch(row["run_id"]))
             and _sha(row["result_sha256"]) and _sha(row["evidence_sha256"]))
    if not (empty or bound) or empty != (row["capability_class"] == "unscreened"):
        return False
    for name in ("robot_forms", "task_families"):
        if name in row:
            allowed = set(tu.ROBOT_FORMS) - {"software_only"} if name == "robot_forms" else tu.TASK_FAMILIES
            value = row[name]
            if not (isinstance(value, list) and all(isinstance(item, str) and item in allowed for item in value)
                    and len(set(value)) == len(value)):
                return False
    evaluation = row.get("evaluation")
    return evaluation is None or (isinstance(evaluation, dict) and set(evaluation) == EVALUATION_KEYS
                                  and all(_string(evaluation[name]) for name in EVALUATION_KEYS - {"evidence"})
                                  and bool(_day(evaluation["as_of"])) and _proofs_shape(evaluation["evidence"]))


def load_audit(raw, teams, screens, *, team_list_raw=None, today=None):
    """Reject malformed, duplicate, partial or stale-snapshot overlays before any derived output is written.

    Missing raw is permitted only as a fail-closed pending qualification for every row.
    Metadata digests document the auditor's inputs; they convey no owner or provider-data authority.
    """
    if raw is None:
        return {"sha256": None, "rows": {}, "reference": None}
    today = today or datetime.now(timezone.utc).date()
    try:
        document = json.loads(raw, object_pairs_hook=_object,
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite")))
        timestamp = datetime.fromisoformat(document["audited_at"].replace("Z", "+00:00"))
        valid = (isinstance(document, dict) and set(document) == ROOT_KEYS and document["schema_version"] == AUDIT
                 and all(_sha(document[name]) for name in ("team_list_sha256", "decisions_sha256",
                                                          "auditor_script_sha256", "scope_sha256"))
                 and timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() == 0
                 and timestamp.date() <= today and _string(document["auditor_reference"], 256)
                 and bool(ss.REFERENCE.fullmatch(document["auditor_reference"]))
                 and isinstance(document["teams"], list) and len(document["teams"]) <= MAX_TEAMS
                 and all(_row_shape(row) for row in document["teams"]))
        if not valid:
            raise ValueError("shape")
        rows = {row["team_key"]: row for row in document["teams"]}
        expected = scope_manifest(teams, screens)
        if (team_list_raw is None or document["team_list_sha256"] != ss._sha256(team_list_raw)
                or ss._json(team_list_raw).get("teams") != teams
                or len(rows) != len(document["teams"]) or set(rows) != {item["team_key"] for item in expected["bindings"]}
                or document["scope_sha256"] != expected["sha256"]
                or any({name: rows[item["team_key"]][name] for name in item} != item
                       for item in expected["bindings"])):
            raise tu.TeamError("team_universe_audit_snapshot_mismatch")
        return {"sha256": ss._sha256(bytes(raw)), "rows": rows, "reference": document["auditor_reference"]}
    except tu.TeamError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError):
        raise tu.TeamError("team_universe_audit_invalid") from None


def _held(proofs, record, evidence, *, own=False):
    """Recheck each exact quote against the exact retained field URL and hashed own-page text.

    Citation-only evidence can support a reference assessment, never a positive offering/interface gate.
    Quotes may select more informative text on an already retained field page; no new URL is admitted.
    """
    answers = record["answers"]
    index = ss.evidence_index(evidence, [])
    for proof in proofs:
        stem, url, quote = proof["field"], proof["url"], proof["quote"]
        if url != answers[stem + "_url"]:
            return False
        if own and not tu.own_page(url, record["domain"]):
            return False
        if proof["level"] == "verified_on_page":
            page = (evidence.get("pages") or {}).get(url)
            if not (isinstance(page, dict) and isinstance(page.get("text"), str)
                    and _sha(page.get("sha256"))
                    and proof.get("page_sha256") == ss._sha256(page["text"].encode())
                    and ("tool_result_sha256" not in proof or proof["tool_result_sha256"] == page["sha256"])):
                return False
            level, sha = ss.quote_level(quote, url, index)
            if level != "verified_on_page" or sha != page["sha256"]:
                return False
        elif own or quote != answers[stem + "_quote"] or record["verification"][stem]["level"] != proof["level"]:
            return False
    return bool(proofs)


def _phrases_supported(proofs, phrases):
    """Structural anchoring only: the reviewer, not these phrase checks, judges the offering semantics."""
    texts = [ss.words(proof["quote"]) for proof in proofs]
    return all(_string(phrase) and len(ss.words(phrase).split()) >= 2
               and any(ss.has_phrase(ss.words(phrase), text) for text in texts) for phrase in phrases)


def _capability_held(proofs, record, evidence, relation):
    """Own offering anchor plus attributed independently read physical evidence, including reporting.

    The reviewer still judges whether the quotes describe an actual product/control/deployment offering.
    A third-party physical/body/task quote must name this team, not merely another robot on the same page.
    """
    company = record["identity"].get("company") or record["input"].get("company")
    if not (_held(proofs, record, evidence) and _string(relation)):
        return False
    owned = []
    for proof in proofs:
        if proof["level"] != "verified_on_page":
            return False
        if tu.own_page(proof["url"], record["domain"]):
            owned.append(proof)
        elif not company or not ss.names_operator(proof["quote"], company):
            return False
    return bool(owned) and _phrases_supported(owned, [relation])


def qualify(record, row, evidence, *, today=None):
    """Separate physical offering, current task fit, evaluation compatibility and published invitation axes."""
    today = today or datetime.now(timezone.utc).date()
    result = {"rule_version": RULE, "offering": "unknown", "current_task_fit": False,
              "evaluation_compatibility": "not_verified", "robot_forms": [], "task_families": [],
              "partner_intent": {"state": "unknown", "kind": "none"},
              "willingness_to_work_with_blueprint": "unknown", "relationship_implied": False,
              "promotion_allowed": False, "blockers": []}
    if record is None:
        result["blockers"] = ["not_screened"]
        return result
    if row is None:
        result["blockers"] = ["offering_not_audited", "evaluation_compatibility_not_verified"]
        return result
    if not _row_shape(row):
        result["blockers"] = ["audit_row_invalid"]
        return result
    result["audit_class"] = row["capability_class"]
    proofs = row["capability_evidence"]
    held = _held(proofs, record, evidence)
    if proofs and not held:
        result["blockers"].append("audit_evidence_not_held")
        return result
    intent = row["published_partner_intent"]
    if intent["state"] != "explicit_invitation" or _held(intent["evidence"], record, evidence, own=True):
        result["partner_intent"] = {name: intent[name] for name in ("state", "kind")}
    else:
        result["blockers"].append("partner_invitation_unproven")
    if row["capability_class"] == "adjacent_reference":
        result["offering"] = "reference_only"
        result["blockers"].append("physical_robot_offering_not_established")
        return result
    if row["capability_class"] not in POSITIVE:
        result["blockers"].extend(["physical_robot_offering_pending", "evaluation_compatibility_not_verified"])
        return result
    phrases = [row.get(name) for name in ("identified_hardware", "physical_task", "offering_relation")]
    if not (_capability_held(proofs, record, evidence, row.get("offering_relation"))
            and _phrases_supported(proofs, phrases)
            and row.get("robot_forms") and row.get("task_families")):
        result["blockers"].append("physical_robot_offering_evidence_incomplete")
        return result
    day = _day(row.get("capability_as_of"))
    basis = row.get("capability_current_basis")
    anchored = ((basis == "current_offering_page" and day == _day(evidence.get("checked_on")))
                or (basis == "dated_deployment" and day == tu._day(record["answers"].get("task_evidence_date"))
                    and any(proof["field"] == "task_evidence" for proof in proofs)))
    if not (day and anchored and 0 <= (today - day).days <= ss.FRESH_DAYS):
        result["blockers"].append("physical_robot_offering_not_current")
        return result
    result.update(offering=row["capability_class"], current_task_fit=row["manual_current_task_fit"],
                  robot_forms=row["robot_forms"], task_families=row["task_families"])
    if not result["current_task_fit"]:
        result["blockers"].append("current_robot_task_fit_not_established")
    compatibility = row["blueprint_current_evaluation_compatibility"]
    if compatibility == "supported":
        # A vendor's own interface/controller proof does not establish Blueprint runtime support.
        # No independently pinned current support registry is admitted by this version. A future
        # extension must bind actual runtime/profile/embodiment/interface artifacts, not a label.
        result["blockers"].append("blueprint_support_contract_unavailable")
    elif compatibility == "unsupported":
        result["evaluation_compatibility"] = "unsupported"
        result["blockers"].append("evaluation_compatibility_unsupported")
    else:
        result["blockers"].append("evaluation_compatibility_not_verified")
    return result
