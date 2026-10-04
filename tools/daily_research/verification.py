"""Provider-neutral evidence assessment and promotion gate. No I/O or model calls.

The agent judges source meaning and exact site/task linkage. This harness binds
that judgment to retained evidence, enforces the transition and exposes gaps.
Public research never grants commercial, consent, robot or deployment authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from datetime import datetime
from urllib.parse import urlsplit

VERSION = "blueprint.lead-verification.v1"
RESULT_VERSION = "blueprint.lead-verification-result.v1"
DIAGNOSTIC_RESULT_VERSION = "blueprint.lead-verification-result.v2"
FACTS = ("operator", "physical_site", "site_task", "human_workflow")
CLAIMS = (*FACTS, "plausible_fit")
STATES = {"verified_fact", "inference", "unresolved", "contradicted", "stale", "unreachable"}
SEPARATE_GATES = ["buying_intent", "consent_rights", "commercial_qualification",
                  "robot_compatibility", "deployment_readiness"]


def digest(value):
    # Separate from the retained packet's historical Python JSON hash. Native
    # JSON readers lose 1.0 vs 1 and exponent lexemes; encode finite numbers by
    # their IEEE754 bytes so inert float metadata is portable across runtimes.
    def encode(item):
        if type(item) in {int, float}:
            if not math.isfinite(item) or float(item).is_integer() and abs(item) > 2**53 - 1:
                raise ValueError("number must be finite and exactly portable; retain large IDs as strings")
            return "n:" + struct.pack(">d", float(item)).hex()
        if isinstance(item, list):
            return "[" + ",".join(encode(x) for x in item) + "]"
        if isinstance(item, dict):
            return "{" + ",".join(json.dumps(k, ensure_ascii=True) + ":" + encode(item[k]) for k in sorted(item)) + "}"
        return json.dumps(item, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encode(value).encode()).hexdigest()


def normalized(value):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", value).lower()))


def identity_key(candidate):
    """Conservative raw identity; the agent resolves semantic site aliases.

    Geographic location may be just a city. Include the named site/address so
    separate facilities in that city survive for independent assessment.
    """
    parts = [candidate.get("organization", ""), candidate.get("site") or candidate.get("location", ""),
             candidate.get("location") or candidate.get("site", ""), candidate.get("task", "")]
    if any(not isinstance(x, str) or not normalized(x) for x in parts):
        return None
    return digest([normalized(x) for x in parts])


def moment(value):
    if not isinstance(value, str):
        raise TypeError("timestamp missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp needs offset")
    return parsed


def diagnostic_moment(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})", value):
        raise ValueError("timestamp invalid")
    return moment(value)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def source_usable(source, assessed_at, *, primary=False):
    try:
        parsed = urlsplit(source["url"])
        return (parsed.scheme in {"https", "http"} and bool(parsed.hostname)
                and not parsed.username and not parsed.password
                and text(source.get("id")) and text(source.get("publisher"))
                and text(source.get("quote")) and text(source.get("freshness_reason"))
                and source.get("freshness") == "current"
                and source.get("retrieval") in {"rendered", "static", "operator_document"}
                and source.get("classification") in ({"operator", "primary"} if primary
                                                     else {"operator", "primary", "independent", "vendor"})
                and moment(source.get("checked_at")) <= assessed_at)
    except (AttributeError, KeyError, TypeError, ValueError):
        # A non-string URL (number/boolean) is unusable evidence, never a QA crash.
        return False


def _evaluate_v1(candidate, assessment, now, *, allow_unknown_expiry=False):
    """Missing or repairable evidence yields unresolved feedback, never rejection.

    Extra inert metadata is preserved. Labels alone cannot pass: each decisive
    claim needs retrieved current sources and an explanation of its exact scope.
    """
    reasons, rejected = [], False
    try:
        candidate_digest, assessment_digest = digest(candidate), digest(assessment)
    except (ValueError, TypeError, OverflowError):
        candidate_digest, assessment_digest = None, None
        reasons.append("digest: repair nonportable numbers/types; retain large IDs as strings")
    result = {"version": RESULT_VERSION, "candidate_key": candidate.get("candidate_key"),
              "candidate_digest": candidate_digest, "identity_key": identity_key(candidate),
              "evaluated_at": now.isoformat(), "assessment": assessment,
              "assessment_digest": assessment_digest, "assessment_present": assessment is not None,
              "assessment_valid": False, "separate_gates": SEPARATE_GATES.copy()}
    if not result["identity_key"]:
        reasons.append("identity: resolve named operator, physical site/location and task")
    if not isinstance(assessment, dict):
        reasons.append("assessment: perform source assessment; discovery alone is unverified")
    else:
        try:
            assessed_at = moment(assessment.get("assessed_at"))
            if (assessment.get("version") != VERSION
                    or candidate_digest is None or assessment_digest is None
                    or assessment.get("candidate_digest") != candidate_digest):
                raise ValueError("binding")
            if assessed_at > now or not (allow_unknown_expiry and assessment.get("valid_until") is None) and not now < moment(assessment.get("valid_until")):
                raise ValueError("freshness")
            claims, sources = assessment.get("claims"), assessment.get("sources")
            counter = assessment.get("counterevidence")
            if (not isinstance(claims, dict) or not isinstance(sources, list)
                    or not isinstance(counter, dict)):
                raise TypeError("structure")
            indexed = {s["id"]: s for s in sources if isinstance(s, dict) and text(s.get("id"))}
            if len(indexed) != len(sources):
                raise ValueError("source identity")
            result["assessment_valid"] = True
            for name in CLAIMS:
                claim = claims.get(name)
                if not isinstance(claim, dict) or claim.get("status") not in STATES or not text(claim.get("reason")):
                    reasons.append(f"{name}: assess claim and retain its source-linked reason")
                    result["assessment_valid"] = False
                    continue
                refs = claim.get("source_refs")
                linked = [indexed[r] for r in refs if isinstance(r, str) and r in indexed] if isinstance(refs, list) else []
                usable = (bool(linked) and len(linked) == len(refs)
                          and all(source_usable(s, assessed_at, primary=name in FACTS) for s in linked))
                state = claim["status"]
                if state == "contradicted" and usable:
                    rejected = True
                    reasons.append(f"{name}: contradicted — {claim['reason']}")
                elif state not in ({"verified_fact"} if name in FACTS else {"verified_fact", "inference"}) or not usable:
                    reasons.append(f"{name}: {state}; resolve source, primary support, scope or freshness — {claim['reason']}")
            refs = counter.get("source_refs")
            linked = [indexed[r] for r in refs if isinstance(r, str) and r in indexed] if isinstance(refs, list) else []
            usable = (bool(linked) and len(linked) == len(refs)
                      and all(source_usable(s, assessed_at) and s.get("classification") != "vendor" for s in linked))
            searches = counter.get("searches")
            searched = isinstance(searches, list) and bool(searches) and all(text(s) for s in searches)
            if counter.get("status") == "contradicted" and usable and text(counter.get("reason")):
                rejected = True
                reasons.append("counterevidence: contradicted — " + counter["reason"])
            elif (counter.get("status") != "checked" or not text(counter.get("reason"))
                  or not (searched or usable) or (refs and not usable)):
                reasons.append("counterevidence: resolve automation/contradictions; retain actual searches, sources and limits")
            if counter.get("status") not in {"checked", "unresolved", "contradicted"}:
                result["assessment_valid"] = False
        except (KeyError, TypeError, ValueError):
            reasons.append("assessment: repair candidate binding, dates, source IDs or assessment structure")
    status = "rejected" if rejected else "unresolved" if reasons else "verified"
    result.update(status=status, reasons=reasons or ["Primary sources support the named site/task and human workflow; fit remains a bounded hypothesis"],
                  eligible_for_qualified_promotion=status == "verified")
    return result


def assessment_issues(candidate, assessment):
    """Locate format/binding defects without manufacturing missing evidence.

    Null assessment/expiry and empty unresolved source/search lists are honest
    outcomes. Keep the submitted bytes/digest; a matching schema_version alias
    is lossless, while conflicting or explicit-null version keys fail closed.
    """
    if assessment is None:
        return []
    issues = []
    def issue(path, code, expected):
        issues.append({"path": path, "code": code, "expected": expected})
    if not isinstance(assessment, dict):
        issue("/", "assessment_object_required", "an assessment object or null for unassessed evidence")
        return issues
    versions = [assessment[key] for key in ("version", "schema_version") if key in assessment]
    if not versions or any(value != VERSION for value in versions):
        issue("/version", "assessment_version_invalid", f'"version": "{VERSION}"; a matching schema_version alias is accepted without rewriting evidence')
    try:
        bound = assessment.get("candidate_digest") == digest(candidate)
        digest(assessment)
    except (ValueError, TypeError, OverflowError):
        bound = False
    if not bound:
        issue("/candidate_digest", "assessment_candidate_binding_invalid", "copy the supplied exact candidate_digest; retain portable metadata and retain large IDs as strings")
    for key in ("assessed_at", "valid_until"):
        if key == "valid_until" and key in assessment and assessment[key] is None:
            continue
        try:
            diagnostic_moment(assessment.get(key))
        except (ValueError, TypeError):
            issue("/" + key, "assessment_timestamp_invalid", "actual ISO timestamp with timezone; valid_until may be null when freshness is unknown, never invent an expiry")
    sources = assessment.get("sources")
    ids = set()
    if not isinstance(sources, list):
        issue("/sources", "assessment_sources_invalid", "a list of retained source objects, including inaccessible/unknown evidence")
    else:
        for index, source in enumerate(sources):
            path = f"/sources/{index}"
            if not isinstance(source, dict):
                issue(path, "assessment_source_invalid", "a retained source object")
                continue
            sid = source.get("id")
            if not text(sid) or sid in ids:
                issue(path + "/id", "assessment_source_id_invalid", "a unique nonempty local source ID; do not invent or merge sources")
            else:
                ids.add(sid)
            if source.get("checked_at") is not None:
                try:
                    diagnostic_moment(source["checked_at"])
                except (ValueError, TypeError):
                    issue(path + "/checked_at", "assessment_timestamp_invalid", "retain the actual source check timestamp with timezone; unknown/unreachable sources cannot supply positive support")
    claims = assessment.get("claims")
    if not isinstance(claims, dict):
        issue("/claims", "assessment_claims_invalid", "separate operator, physical_site, site_task, human_workflow and plausible_fit claims")
    else:
        for name in CLAIMS:
            claim, path = claims.get(name), "/claims/" + name
            if not isinstance(claim, dict):
                issue(path, "assessment_claim_invalid", "a claim object with status, source-linked reason and source_refs; missing facts stay unresolved")
                continue
            if not isinstance(claim.get("status"), str) or claim["status"] not in STATES or not text(claim.get("reason")):
                issue(path, "assessment_claim_invalid", "an allowed evidence status and a nonempty reason at the exact operator/site/task scope")
            refs = claim.get("source_refs")
            if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in ids for ref in refs):
                issue(path + "/source_refs", "assessment_source_reference_invalid", "a list of exact retained source IDs; unresolved claims may use an empty list")
    counter = assessment.get("counterevidence")
    if not isinstance(counter, dict):
        issue("/counterevidence", "assessment_counterevidence_invalid", "status, reason, source_refs and actual bounded searches; unknown results remain unresolved")
    else:
        if not isinstance(counter.get("status"), str) or counter["status"] not in {"checked", "unresolved", "contradicted"} or not text(counter.get("reason")):
            issue("/counterevidence", "assessment_counterevidence_invalid", "checked, unresolved or contradicted with the actual scope and limits")
        refs, searches = counter.get("source_refs"), counter.get("searches")
        if not isinstance(refs, list) or any(not isinstance(ref, str) or ref not in ids for ref in refs):
            issue("/counterevidence/source_refs", "assessment_source_reference_invalid", "exact retained source IDs or an empty list for unresolved evidence")
        if not isinstance(searches, list) or any(not (text(query) or isinstance(query, dict) and text(query.get("query"))) for query in searches):
            issue("/counterevidence/searches", "assessment_searches_invalid", "a list of actual query strings or retained query records with a query field; an empty list is valid when no countersearch was performed")
    return issues


def evaluate(candidate, assessment, now, *, result_version=DIAGNOSTIC_RESULT_VERSION):
    if result_version == RESULT_VERSION:
        return _evaluate_v1(candidate, assessment, now)
    if result_version != DIAGNOSTIC_RESULT_VERSION:
        raise ValueError("lead verification result version unsupported")
    # Reuse the proven claim checks after validating the envelope. Read the
    # version alias on a copy; raw assessment, hash and unknown expiry stay intact.
    issues = assessment_issues(candidate, assessment)
    evaluation = dict(assessment) if isinstance(assessment, dict) else assessment
    expiry_unknown = isinstance(assessment, dict) and assessment.get("valid_until") is None and "valid_until" in assessment
    if isinstance(evaluation, dict) and not issues:
        evaluation["version"] = VERSION
        counter = assessment["counterevidence"]
        def query_text(query):
            if isinstance(query, dict):
                try:
                    if query.get("checked_at") is not None and moment(query["checked_at"]) > moment(assessment["assessed_at"]):
                        return None
                except (ValueError, TypeError):
                    return None
                return query["query"]
            return query
        evaluation["counterevidence"] = {**counter, "searches": [query_text(query) for query in counter["searches"]]}
    result = _evaluate_v1(candidate, evaluation, now, allow_unknown_expiry=expiry_unknown and not issues)
    result.update(version=DIAGNOSTIC_RESULT_VERSION, assessment=assessment,
                  assessment_digest=None, validation_errors=issues)
    try:
        result["assessment_digest"] = digest(assessment)
    except (ValueError, TypeError, OverflowError):
        pass
    if issues:
        result.update(assessment_valid=False, status="unresolved", eligible_for_qualified_promotion=False)
        result["reasons"] = [item["path"] + ": " + item["expected"] for item in issues]
    elif isinstance(assessment, dict) and moment(assessment["assessed_at"]) > now:
        result.update(assessment_valid=False, status="unresolved", eligible_for_qualified_promotion=False)
        result["reasons"] = ["/assessed_at: assessment is in the future; retain the actual checked time"]
    elif expiry_unknown:
        retained_reasons = result["reasons"] if result["status"] != "verified" else []
        result.update(status="unresolved", eligible_for_qualified_promotion=False)
        result["reasons"] = ["/valid_until: freshness is unknown; retain unresolved evidence and establish a supported currentness boundary before promotion"] + retained_reasons
    elif isinstance(assessment, dict):
        # Temporal failures are evidence gaps, not an excuse to invent a date.
        assessed_at, valid_until = moment(assessment["assessed_at"]), moment(assessment["valid_until"])
        if assessed_at > now or valid_until <= now:
            result.update(assessment_valid=False, status="unresolved", eligible_for_qualified_promotion=False)
            result["reasons"] = (["/assessed_at: assessment is in the future; retain the actual checked time"] if assessed_at > now else []) + (
                ["/valid_until: assessment has expired; recheck the sources or retain unknown freshness"] if valid_until <= now else [])
    return result


def packet_candidates(packet):
    """New packets preserve full duplicate evidence without duplicating arrays."""
    candidates = packet["candidates"]
    if packet.get("verification_cohort_version") == VERSION:
        candidates = [*candidates, *packet.get("duplicates", [])]
        return sorted(candidates, key=lambda c: c["discovery_index"])
    return candidates


def cohort(candidates, assessments, now, actual_cost_usd=None, duplicate_checks=None, *, result_version=DIAGNOSTIC_RESULT_VERSION):
    """Apply identical criteria to the entire supplied discovery cohort."""
    results = [evaluate(c, assessments.get(c.get("candidate_key")), now, result_version=result_version) for c in candidates]
    duplicate_checks = duplicate_checks or {}
    indexed = {r["candidate_key"]: r for r in results}
    for result in results:
        check = duplicate_checks.get(result["candidate_key"])
        if not isinstance(check, dict) or check.get("duplicate") is not True:
            continue
        result["duplicate_check"] = check
        result["eligible_for_qualified_promotion"] = False
        target = check.get("duplicate_of")
        visited = {result["candidate_key"]}
        while text(check.get("reason")) and isinstance(target, str) and target in indexed and target not in visited:
            visited.add(target)
            next_check = duplicate_checks.get(target, {})
            if not isinstance(next_check, dict):
                target = None
                continue
            if next_check.get("duplicate") is not True:
                result["identity_key"] = indexed[target]["identity_key"]
                result["duplicate_of"] = target
                break
            if not text(next_check.get("reason")):
                target = None
                continue
            target = next_check.get("duplicate_of")
        else:
            result.update(status="unresolved", eligible_for_qualified_promotion=False)
            result["reasons"].append("duplicate: retain referenced original candidate and equivalence reason; unresolved duplicate cannot inflate verified yield")
    groups = {}
    for index, result in enumerate(results):
        # Missing identity cannot collapse unrelated unknown candidates.
        groups.setdefault(result["identity_key"] or f"unresolved:{index}", []).append(result)
    statuses = []
    for group in groups.values():
        group = sorted(group, key=lambda r: bool(r.get("duplicate_check", {}).get("duplicate")))
        # A duplicate with contradictory assessments cannot inflate verified yield.
        conflict = len({r["status"] for r in group}) > 1
        statuses.append("unresolved" if conflict else group[0]["status"])
        for index, result in enumerate(group):
            if conflict:
                result.update(status="unresolved", eligible_for_qualified_promotion=False)
                result["reasons"].append("duplicate assessments conflict: resolve the same operator/site/task before promotion")
            if index:
                result.update(duplicate_of=group[0]["candidate_key"], eligible_for_qualified_promotion=False)
    counts = {name: statuses.count(name) for name in ("verified", "unresolved", "rejected")}
    assessed = sum(r["assessment_valid"] for r in results)
    actual = (type(actual_cost_usd) in {int, float} and math.isfinite(actual_cost_usd) and actual_cost_usd >= 0)
    return {"criteria_version": VERSION, **({"result_version": result_version} if result_version != RESULT_VERSION else {}), "duplicate_checks": duplicate_checks,
            "candidate_count": len(results), "unique_site_task_candidates": len(groups),
            "duplicates": len(results) - len(groups), "verified_unique_site_task_candidates": counts["verified"],
            "unresolved_unique_site_task_candidates": counts["unresolved"], "rejected_unique_site_task_candidates": counts["rejected"],
            "unresolved_count": sum(r["status"] == "unresolved" for r in results),
            "rejected_count": sum(r["status"] == "rejected" for r in results),
            "assessed_count": assessed, "verification_coverage": assessed / len(results) if results else None,
            "actual_cost_usd": actual_cost_usd if actual else None,
            "verified_unique_per_usd": counts["verified"] / actual_cost_usd if actual and actual_cost_usd > 0 else None,
            "results": results}


def compare(runs, now):
    """No winner is inferred from narrative specificity or a selected sample.

    Callers supply the full retained candidate lists, never the reviewed subset.
    Actual costs must cover the same discovery+verification scope for comparison.
    """
    cohorts = {name: cohort(run["candidates"], run.get("assessments", {}), now, run.get("actual_cost_usd"), run.get("duplicate_checks"))
               for name, run in runs.items()}
    scopes = {r.get("comparison_scope") for r in runs.values()}
    comparable = (bool(cohorts) and len(scopes) == 1 and None not in scopes and "" not in scopes
                  and all(c["verified_unique_per_usd"] is not None for c in cohorts.values())
                  and all(r.get("cost_scope") == "discovery_and_verification" for r in runs.values()))
    return {"criteria_version": VERSION, "providers": cohorts, "accuracy_winner": None,
            "accuracy_note": "Report full-cohort verification coverage and unique verified yield. Selected examples and specificity do not establish an accuracy winner.",
            "cost_comparison_available": comparable}
