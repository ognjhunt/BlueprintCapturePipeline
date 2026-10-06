"""Provider-neutral evidence assessment, promotion gate and outreach tier. No model calls.

The agent judges source meaning and exact site/task linkage. This harness binds
that judgment to retained evidence, enforces the transition and exposes gaps.
Public research never grants commercial, consent, robot or deployment authority.
The only I/O is retained_evidence's injected read of digest-checked tool results.
The outreach tier (blueprint.outreach-ready-rule.v1.2) is one pure derivation over
these gates: a hypothesis label for drafting, never verification or send authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
import unicodedata
from datetime import datetime
from urllib.parse import urlsplit

VERSION = "blueprint.lead-verification.v1"
RESULT_VERSION = "blueprint.lead-verification-result.v1"
DIAGNOSTIC_RESULT_VERSION = "blueprint.lead-verification-result.v2"
# v2 plus tier, eligible_for_outreach_ready and the outreach_ready block. status and
# eligible_for_qualified_promotion keep their v2 meaning: the verified path is unchanged.
OUTREACH_RESULT_VERSION = "blueprint.lead-verification-result.v3"
# v1.1 (design section 10): a site link check, one question, facility blocks, automation as a question.
# v1.2: the same tier with the question's wording fixed (question_task, site_phrase, template U). A
# direction must name exactly this version. v1.1 is kept only to re-derive rows published under it.
OUTREACH_RULE_VERSION = "blueprint.outreach-ready-rule.v1.2"
LEGACY_OUTREACH_RULE_VERSION = "blueprint.outreach-ready-rule.v1.1"
OUTREACH_RULE_VERSIONS = (LEGACY_OUTREACH_RULE_VERSION, OUTREACH_RULE_VERSION)
EVIDENCE_VERSION = "blueprint.outreach-ready-evidence.v1"
FACTS = ("operator", "physical_site", "site_task", "human_workflow")
CLAIMS = (*FACTS, "plausible_fit")
STATES = {"verified_fact", "inference", "unresolved", "contradicted", "stale", "unreachable"}
SEPARATE_GATES = ["buying_intent", "consent_rights", "commercial_qualification",
                  "robot_compatibility", "deployment_readiness"]
PROVEN_FACTS = ("operator", "physical_site", "site_task")
# site_task may rest on company-wide evidence (inference): the site link stays open and is the question.
SITE_TASK_STATES = frozenset({"verified_fact", "inference"})
# A shorter quote proves too little: one or two words appear on almost any page. Words are
# whitespace-separated tokens with punctuation stripped, so "U.S. Foods" is two words.
MIN_QUOTE_WORDS = 3
READ_TOOL, SEARCH_TOOL = "blueprint_read_source", "blueprint_search"  # search.READ and search.SEARCH
# Publication-phase reads come after review, so they never change a recomputed tier.
EVIDENCE_PHASES = frozenset({"research", "repair", "qa"})
# Open checks in rule order; the last three are always open (Blueprint-WebApp #855 derives the same list).
OPEN_CHECKS = ("site_link", "manual_workflow", "freshness", "existing_automation", "fit", "interest")
# Exactly one question: the missing fact whose answer would change the decision, by precedence S
# (site link open), then M (manual workflow open), then A (manual workflow verified and partial
# automation evidenced), else U (automation unknown or none shown). Word for word; {task} is
# question_task(task) and {site} is site_phrase(site, location). The other open checks stay unasked.
QUESTION_TEMPLATES = {
    "S": "Is {task} done at {site}, or somewhere else in the company?",
    "M": "Which parts of {task} at {site} still need people, and what has kept them from being automated?",
    "A": "What has kept the rest of {task} at {site} from being automated so far?",
    "U": "Is any of {task} at {site} automated today, or is it all done by hand?"}
# v1.1, word for word, with the candidate's task and site fields verbatim; never used for a new row.
LEGACY_QUESTION_TEMPLATES = {
    "S": "Is {task} done at your {site} site, or somewhere else in the company?",
    "M": "Which parts of {task} at {site} still need people, and what has kept them from being automated?",
    "A": "What has kept the remaining {task} work at {site} from being automated so far?"}
QUESTION_TEMPLATES_BY_RULE = {LEGACY_OUTREACH_RULE_VERSION: LEGACY_QUESTION_TEMPLATES, OUTREACH_RULE_VERSION: QUESTION_TEMPLATES}
# The question's site: a city without its state, ZIP code or country (site_city, city_case). STATE_NAMES is
# shared with site_screen. Each mirror (publisher.mjs, Blueprint-WebApp) copies these tables.
STATE_NAMES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california", "CO": "colorado",
    "CT": "connecticut", "DE": "delaware", "DC": "district of columbia", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland", "MA": "massachusetts", "MI": "michigan",
    "MN": "minnesota", "MS": "mississippi", "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada",
    "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "PR": "puerto rico", "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee",
    "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming"}
COUNTRY_NAMES = frozenset({"us", "usa", "united states", "united states of america", "canada", "mexico", "uk",
                           "united kingdom"})
SPACES = re.compile(r"[ \t\n\r\f\v]+")  # ASCII whitespace only, as every mirror reads it.
TRAILING_MARKS = ".,;:!?\u2026\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u061f\u060c\u061b "
ZIP_TAIL = r"[0-9]{5}(?:-[0-9]{4})?"  # ASCII digits and case only, as every mirror reads them.
STATE_TAIL = re.compile(r"(?:^| )(?:(?:" + "|".join(sorted({*STATE_NAMES, *STATE_NAMES.values()}, key=len, reverse=True))
                        + r")(?: " + ZIP_TAIL + r")?|" + ZIP_TAIL + r")$", re.IGNORECASE | re.ASCII)
SHORT_SITE_NAME = (4, 40)  # At most four words and 40 characters.
# Words kept lower-case inside an ALL CAPS city that city_case title-cases: Isle of Palms, Prairie du Chien.
SMALL_WORDS = frozenset({"and", "de", "del", "du", "la", "le", "of", "or", "the"})
# Optional assessment fields (design v1.1 section 10.2), each {"value", "source_refs", "reason"}. A
# blocking value blocks the tier only when proven: a cited, usable, non-vendor source whose quote is
# found in retained text. Absent or "unknown" changes nothing; a malformed field gives none.
FACILITY_FIELDS = {"facility_type": (frozenset({"operations", "office", "mailing_only", "unknown"}), frozenset({"office", "mailing_only"})),
                   "facility_operator": (frozenset({"company", "contractor", "tenant", "unknown"}), frozenset({"contractor", "tenant"}))}
FACILITY_BLOCKERS = {"facility_type": "facility_office_or_mailing_only", "facility_operator": "facility_operated_by_another_party"}
# A job post names the place of work, so one that names the site ties the task to it.
JOB_PATH_SEGMENTS = frozenset({"careers", "career", "jobs", "job", "job-posting", "job-postings", "openings", "vacancies"})
JOB_HOSTS = ("greenhouse.io", "lever.co", "myworkdayjobs.com", "icims.com", "smartrecruiters.com", "jobvite.com",
             "ashbyhq.com", "workable.com", "bamboohr.com", "taleo.net")
STREET = re.compile(r"\d[^\W_]* [^\W\d_]")  # A normalized address segment: a house number, then a name.
UNAVAILABLE = {"schema_version": EVIDENCE_VERSION, "state": "unavailable"}


class EvidenceBudgetExhausted(Exception):
    """retained_evidence stopped before reading every result: its read or time budget ran out."""


class ReadBudget:
    """At most ``reads`` result reads within ``seconds`` of monotonic time, checked before each read."""

    def __init__(self, reads, seconds, clock=time.monotonic):
        self.reads, self.clock = reads, clock
        self.deadline = clock() + seconds

    def take(self):
        if self.reads <= 0 or self.clock() >= self.deadline:
            raise EvidenceBudgetExhausted()
        self.reads -= 1


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


def evaluate(candidate, assessment, now, *, result_version=DIAGNOSTIC_RESULT_VERSION, evidence=None):
    if result_version == RESULT_VERSION:
        return _evaluate_v1(candidate, assessment, now)
    if result_version == OUTREACH_RESULT_VERSION:
        # One record without cohort context; cohort() adds duplicate and conflict state.
        result = evaluate(candidate, assessment, now)
        result.update(version=OUTREACH_RESULT_VERSION, **_tier(result, candidate, evidence_index(evidence), now, False))
        return result
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


def _json_digest(value):
    # runner.digest; this module cannot import runner (runner imports it).
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _citations(snapshot):
    """(url, excerpt) for every citation in a retained Parallel FindAll snapshot."""
    stack = [snapshot]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get("url"), str) and isinstance(item.get("excerpts"), list):
                yield from ((item["url"], excerpt) for excerpt in item["excerpts"] if isinstance(excerpt, str))
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def host_key(value):
    """The host part of url_key: letter case and www. do not differ; None for an unusable URL."""
    key = url_key(value)
    return key.split("/", 1)[0].split("?", 1)[0] if key else None


def retained_evidence(row, read, *, budget=None):
    """Page text and excerpts from this row's retained tool results, each checked against its digests.

    ``read`` is the ledger's read_bytes, the only I/O. Pages are blueprint_read_source text;
    excerpts are blueprint_search snippets and Parallel FindAll citation excerpts. A result whose
    bytes, digest or shape differ, or whose file is missing, is skipped and counted as refused;
    the others are still used. Page text is credited to the requested URL and its redirect hops
    only while every hop stays on the requested host: a cross-host redirect gets no credit, since
    its text is another host's. ``budget`` (a ReadBudget) is taken before each read and raises
    EvidenceBudgetExhausted when spent. Other store errors propagate: callers record the evidence
    as unavailable.
    """
    pages, excerpts, refused = [], [], 0

    def load(name):
        if budget is not None:
            budget.take()
        try:
            return read(name)
        except FileNotFoundError:
            return None

    calls = row.get("application_tool_calls") if isinstance(row, dict) else None
    calls = calls if isinstance(calls, dict) else {}
    for cid, call in sorted(calls.items()):
        request = call.get("request") if isinstance(call, dict) else None
        name = request.get("name") if isinstance(request, dict) else None
        if (name not in {READ_TOOL, SEARCH_TOOL} or call.get("phase") not in EVIDENCE_PHASES
                or call.get("success") is not True or not isinstance(call.get("result_file"), str)):
            continue
        raw = load(call["result_file"])
        try:
            if raw is None:
                raise ValueError("tool result missing")
            event = json.loads(raw)
            if (hashlib.sha256(raw).hexdigest() != call.get("result_sha256") or _json_digest(event) != call.get("result_digest")
                    or event.get("success") is not True or event.get("call_id") != cid):
                raise ValueError("tool result binding")
            output, sha = json.loads(event["output"]), call["result_sha256"]
            if name == READ_TOOL:
                if not isinstance(output["text"], str):
                    raise TypeError("page text")
                chain = [output["requested_url"], *(item["url"] for item in output.get("redirects") or []), output["url"]]
                host = host_key(chain[0])
                same_host = host is not None and all(host_key(url) == host for url in chain)
                found = ([{"url": url, "text": output["text"], "tool_result_sha256": sha} for url in sorted(set(chain))]
                         if same_host else [])
            else:
                found = [{"url": item["url"], "text": item["snippet"], "tool_result_sha256": sha, "kind": "search_snippet"}
                         for item in output["response"]["results"]]
        except (AttributeError, KeyError, TypeError, ValueError, UnicodeError):
            refused += 1
            continue
        (pages if name == READ_TOOL else excerpts).extend(found)
    reads = row.get("parallel_findall_reads") if isinstance(row, dict) else None
    for cid, receipt in sorted(reads.items()) if isinstance(reads, dict) else ():
        call = calls.get(cid)
        if (not isinstance(call, dict) or call.get("phase") not in EVIDENCE_PHASES or not isinstance(receipt, dict)
                or receipt.get("operation") not in {"status", "result"} or not isinstance(receipt.get("file"), str)):
            continue
        raw = load(receipt["file"])
        try:
            if raw is None or hashlib.sha256(raw).hexdigest() != receipt.get("sha256") or len(raw) != receipt.get("bytes"):
                raise ValueError("snapshot binding")
            found = [{"url": url, "text": text, "tool_result_sha256": receipt["sha256"], "kind": "citation_excerpt"}
                     for url, text in _citations(json.loads(raw))]
        except (TypeError, ValueError, UnicodeError):
            refused += 1
            continue
        excerpts.extend(found)
    return {"schema_version": EVIDENCE_VERSION, "state": "retained", "pages": pages, "excerpts": excerpts, "refused": refused}


def url_key(value):
    """Same-URL comparison: scheme, default port, host case, www., trailing slash and fragment do not differ.

    Non-default ports identify a different source and must remain in the binding.
    """
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").lower().removeprefix("www.")
        if parts.scheme not in {"http", "https"} or not host or parts.username or parts.password:
            return None
        port = parts.port  # Validate malformed and out-of-range ports even when no suffix is needed.
        if ":" in host:
            host = "[" + host + "]"
        if port is not None and port != (443 if parts.scheme == "https" else 80):
            host += ":" + str(port)
        return host + (parts.path.rstrip("/") or "") + ("?" + parts.query if parts.query else "")
    except (AttributeError, TypeError, ValueError):
        return None


def evidence_index(evidence):
    """Normalized text by URL key from retained evidence; malformed records are skipped."""
    index, memo = {"pages": {}, "excerpts": {}}, {}
    if not isinstance(evidence, dict) or evidence.get("state") != "retained":
        return index
    for kind in ("pages", "excerpts"):
        records = evidence.get(kind)
        for record in records if isinstance(records, list) else ():
            try:
                key, raw, sha = url_key(record["url"]), record["text"], record["tool_result_sha256"]
                if key is None or not isinstance(sha, str) or not isinstance(raw, str):
                    continue
                if id(raw) not in memo:
                    memo[id(raw)] = (raw, " " + normalized(raw) + " ")  # Keep raw alive so its id stays unique.
                index[kind].setdefault(key, []).append((memo[id(raw)][1], sha))
            except (KeyError, TypeError):
                continue
    return index


def quote_words(quote):
    """Whitespace-separated tokens that keep a letter or digit once punctuation is stripped."""
    return [token for token in unicodedata.normalize("NFKC", quote).split() if re.search(r"[^\W_]", token)]


def quote_level(quote, url, index):
    """(verified_on_page | in_citation_excerpt, tool_result_sha256) for a quote proven at this URL.

    The normalized quote must appear whole-word in retained page text for the same URL
    (verified_on_page), else in a retained search snippet or citation excerpt for it.
    (None, None) otherwise, including a quote of fewer than MIN_QUOTE_WORDS words (quote_words).
    """
    try:
        needle, key = normalized(quote), url_key(url)
        if key is None or len(quote_words(quote)) < MIN_QUOTE_WORDS:
            return None, None
    except (AttributeError, TypeError):
        return None, None
    needle = " " + needle + " "
    for level, kind in (("verified_on_page", "pages"), ("in_citation_excerpt", "excerpts")):
        for text_value, sha in index[kind].get(key, ()):
            if needle in text_value:
                return level, sha
    return None, None


def site_terms(candidate):
    """(places, names) that tie evidence to this facility, normalized.

    places: the city (the first part of a "City, Region[, Country]" location) and any street
    address (a site or location part that starts with a house number). names: the places and
    the site's own name (its first part), which only a job post may use.
    """
    def parts(value):
        return [normalized(part) for part in value.split(",")] if isinstance(value, str) else []
    site, location = parts(candidate.get("site")), parts(candidate.get("location"))
    places = {part for part in site + location if STREET.match(part)}
    if len([part for part in location if part]) >= 2 and location[0]:
        places.add(location[0])
    names = places | ({site[0]} if site and site[0] else set())
    return frozenset(places), frozenset(names)


def job_post(url):
    """True for a careers or jobs page: a careers/jobs path segment or host, or a hiring-system host."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except (AttributeError, TypeError, ValueError):
        return False
    segments = {segment.lower() for segment in parts.path.split("/") if segment}
    return (bool(segments & JOB_PATH_SEGMENTS) or host.split(".", 1)[0] in {"careers", "jobs"}
            or any(host == name or host.endswith("." + name) for name in JOB_HOSTS))


def names_site(source, index, places, names):
    """The quote or its retained page names the site's city or street, or the source is a job post naming the site."""
    try:
        key = url_key(source.get("url"))
        texts = [" " + normalized(source["quote"]) + " ", *(text_value for kind in ("pages", "excerpts")
                                                             for text_value, _ in index[kind].get(key, ()))]
    except (AttributeError, KeyError, TypeError):
        return False
    def found(terms):
        return any(" " + term + " " in text_value for term in terms for text_value in texts)
    return found(places) or job_post(source.get("url")) and found(names)


def facility_gates(assessment, indexed, index):
    """{field: {"value", "proven"}} for the optional facility fields; "invalid" when one is malformed."""
    value = {"valid": True}
    try:
        assessed_at = moment(assessment.get("assessed_at"))
    except (TypeError, ValueError):
        assessed_at = None
    for field, (allowed, _) in FACILITY_FIELDS.items():
        entry = assessment.get(field)
        if entry is None:
            value[field] = {"value": None, "proven": False}
            continue
        refs = entry.get("source_refs") if isinstance(entry, dict) else None
        if (not isinstance(entry, dict) or entry.get("value") not in allowed or not isinstance(refs, list)
                or any(not isinstance(ref, str) or ref not in indexed for ref in refs)):
            value.update(valid=False)
            value[field] = {"value": None, "proven": False}
            continue
        proven = assessed_at is not None and any(
            source_usable(indexed[ref], assessed_at) and indexed[ref].get("classification") != "vendor"
            and quote_level(indexed[ref].get("quote"), indexed[ref].get("url"), index)[0] for ref in refs)
        value[field] = {"value": entry["value"], "proven": proven}
    return value


def outreach_gates(result, candidate, index, *, conflict=False):
    """The inputs outreach_tier reads, from one evaluated result and the retained evidence."""
    assessment = result.get("assessment") if isinstance(result.get("assessment"), dict) else {}
    claims = assessment.get("claims") if isinstance(assessment.get("claims"), dict) else {}
    sources = assessment.get("sources") if isinstance(assessment.get("sources"), list) else []
    indexed = {s["id"]: s for s in sources if isinstance(s, dict) and text(s.get("id"))}
    counter = assessment.get("counterevidence") if isinstance(assessment.get("counterevidence"), dict) else {}
    states = {name: claims[name].get("status") if isinstance(claims.get(name), dict) else None for name in CLAIMS}
    states["counterevidence"] = counter.get("status")
    places, names = site_terms(candidate)
    facts = {}
    for name in PROVEN_FACTS:
        refs = claims[name].get("source_refs") if isinstance(claims.get(name), dict) else None
        linked = [indexed[r] for r in refs if isinstance(r, str) and r in indexed] if isinstance(refs, list) else []
        try:
            assessed_at = moment(assessment.get("assessed_at"))
            usable = bool(linked) and len(linked) == len(refs) and all(source_usable(s, assessed_at, primary=True) for s in linked)
        except (TypeError, ValueError):
            usable = False
        proofs, site_specific = [], False
        for source in linked:
            level, sha = quote_level(source.get("quote"), source.get("url"), index)
            if level:
                proofs.append({"claim": name, "source_id": source["id"], "url": source["url"],
                               "quote_sha256": hashlib.sha256(source["quote"].encode()).hexdigest(),
                               "level": level, "tool_result_sha256": sha})
                site_specific = site_specific or name == "site_task" and names_site(source, index, places, names)
        facts[name] = {"primary_sources_usable": usable, "proofs": proofs, "site_specific": site_specific}
    check = result.get("duplicate_check")
    return {"eligible_for_qualified_promotion": result.get("eligible_for_qualified_promotion") is True,
            "assessment_valid": result.get("assessment_valid") is True and not result.get("validation_errors"),
            "identity_present": bool(result.get("identity_key")),
            "duplicate": bool(result.get("duplicate_of")) or isinstance(check, dict) and check.get("duplicate") is True,
            "conflict": conflict is True, "valid_until": assessment.get("valid_until"), "states": states, "facts": facts,
            "facility": facility_gates(assessment, indexed, index), "task": candidate.get("task"), "site": candidate.get("site"),
            # v1.2: the question's city, and its automation evidence: a contradicted counterevidence records
            # automation (of part of this task, other tasks or other sites) that does not block.
            "location": candidate.get("location"), "partial_automation": states["counterevidence"] == "contradicted"}


def open_checks(states, valid_until):
    """The rule-order open checks Blueprint-WebApp #855 derives from the same assessment."""
    return (["site_link"] if states.get("site_task") != "verified_fact" else []) + (
        ["manual_workflow"] if states.get("human_workflow") != "verified_fact" else []) + (
        ["freshness"] if valid_until is None else []) + ["existing_automation", "fit", "interest"]


def question_template(checks, partial_automation=False, rule_version=OUTREACH_RULE_VERSION):
    """S while the site link is open, else M while the manual workflow is open, else A when partial automation
    is evidenced and U when it is unknown or none is shown. v1.1 asks A in both of those cases."""
    if "site_link" in checks:
        return "S"
    if "manual_workflow" in checks:
        return "M"
    return "A" if partial_automation is True or rule_version == LEGACY_OUTREACH_RULE_VERSION else "U"


def _collapsed(value):
    return SPACES.sub(" ", value).strip(" ") if isinstance(value, str) else ""


def question_task(task):
    """The task as the question writes it: whitespace collapsed, trailing punctuation dropped, and the first
    letter lower-cased only when the first word is an ordinary capitalised word ("Loading" but not "CNC",
    "SMT", "iPhone" or "McKinney"): an upper-case letter, then at least one more letter, all lower-case."""
    value = _collapsed(task).rstrip(TRAILING_MARKS)
    word = value.split(" ", 1)[0]
    letters = [char for char in word if unicodedata.category(char).startswith("L")]
    if (word and unicodedata.category(word[0]) == "Lu" and len(letters) >= 2
            and all(unicodedata.category(char) == "Ll" for char in letters[1:])):
        value = value[0].lower() + value[1:]
    return value


def _place(value):
    """A location part compared with the state and country names: lower-case, without dots."""
    return _collapsed(value.replace(".", "")).lower()


def site_city(location):
    """The city of a location or address, as the source wrote it (site_phrase applies city_case), or None.

    A "City, Region[, Country]" location gives its first part. An address that ends in a state, ZIP code
    or both ("12 Main St, Springfield, IL 62701", "GRAND PRAIRIE TX 75050") gives the part before them. A
    trailing country is dropped, a note after a semicolon is ignored, and only one state is removed, so
    "New York, NY" keeps New York. A part with a digit is never a city.
    """
    if not isinstance(location, str):
        return None
    parts = [part for part in (_collapsed(item) for item in location.split(";")[0].split(",")) if part]
    while parts and _place(parts[-1]) in COUNTRY_NAMES:
        parts.pop()
    stated = False
    while parts and re.fullmatch(ZIP_TAIL, parts[-1]):
        parts.pop()
        stated = True
    if parts and (_place(parts[-1]).upper() in STATE_NAMES or _place(parts[-1]) in STATE_NAMES.values()):
        parts.pop()
        stated = True
    elif parts and STATE_TAIL.search(parts[-1]):
        parts[-1] = STATE_TAIL.sub("", parts[-1]).rstrip(" ")
        stated = True
    parts = [part for part in (item.rstrip(TRAILING_MARKS) for item in parts) if part]
    city = (parts[-1] if stated else parts[0]) if parts else None
    if not city or re.search("[0-9]", city) or not any(unicodedata.category(char).startswith("L") for char in city):
        return None
    return city


def city_case(city):
    """A city as the question writes it. Mixed or lower casing is the source's own and is kept exactly
    (McKinney, DeSoto). An ALL CAPS city, as government records write it, is title-cased word by word and
    hyphen part by part, Mc and a letter-apostrophe prefix restored (MCKINNEY -> McKinney, O'FALLON ->
    O'Fallon; Mac is left alone, so Macon stays Macon), with SMALL_WORDS lower-case inside the name."""
    categories = {unicodedata.category(char) for char in city}
    if "Lu" not in categories or "Ll" in categories:
        return city
    words = []
    for index, word in enumerate(city.split(" ")):
        parts = []
        for position, part in enumerate(word.split("-")):
            letters = list(part.lower())
            if (index or position) and "".join(letters) in SMALL_WORDS:
                parts.append("".join(letters))
                continue
            if letters:
                letters[0] = letters[0].upper()
            if len(letters) > 2 and letters[0] + letters[1] == "Mc" or len(letters) > 2 and letters[1] in "'\u2019":
                letters[2] = letters[2].upper()
            parts.append("".join(letters))
        words.append("-".join(parts))
    return " ".join(words)


def site_phrase(site, location=None):
    """The question's site: "your <City> site" (city_case of site_city of ``location``), else "your <name>
    site" for a short site name (the first part of ``site``, at most four words and 40 characters, not a
    street address; one ending in "site" is not doubled), else "this site". Daily QA passes the candidate's
    site and location fields, and the site screen the input site name and the address city."""
    city = site_city(location)
    if city:
        return f"your {city_case(city)} site"
    name = _collapsed(site.split(",")[0]).rstrip(TRAILING_MARKS) if isinstance(site, str) else ""
    words, characters = SHORT_SITE_NAME
    if (name and name[0] not in "0123456789" and len(name.split(" ")) <= words and len(name) <= characters
            and any(unicodedata.category(char).startswith("L") for char in name)):
        return f"your {name}" if name.lower() == "site" or name.lower().endswith(" site") else f"your {name} site"
    return "this site"


def outreach_question(checks, task, site, location=None, *, partial_automation=False, rule_version=OUTREACH_RULE_VERSION):
    """(template, question): the one question, word for word, under ``rule_version``. v1.2 writes
    question_task(task) and site_phrase(site, location); v1.1 writes the task and site fields verbatim.
    site_screen and every mirror build the question this way; never copy it."""
    template = question_template(checks, partial_automation, rule_version)
    if rule_version == LEGACY_OUTREACH_RULE_VERSION:
        return template, LEGACY_QUESTION_TEMPLATES[template].format(task=task, site=site)
    if rule_version != OUTREACH_RULE_VERSION:
        raise ValueError("outreach_rule_version_unknown")
    return template, QUESTION_TEMPLATES[template].format(task=question_task(task), site=site_phrase(site, location))


def outreach_tier(gates, now, rule_version=OUTREACH_RULE_VERSION):
    """blueprint.outreach-ready-rule.v1.2 (``rule_version`` v1.1 re-derives a row published under it).

    verified: the unchanged full-proof path. outreach_ready: operator and physical_site are
    verified_fact and site_task is verified_fact or inference, each from usable primary
    sources with a quote at verified_on_page or in_citation_excerpt; a verified_fact
    site_task must be tied to this facility (the quote or its page names the site's city or
    street, or it is a job post naming the site), so company-wide capability text is
    inference and leaves site_link open. The assessment is valid, unexpired (or of unknown
    freshness), not a duplicate or conflict. A contradicted claim blocks: a closed site is a
    contradicted physical_site, and only a contradicted human_workflow (the exact task at
    this site shown fully automated) is an automation block. Other automation evidence,
    including a contradicted counterevidence, changes the question, not the eligibility. A
    proven office or mailing-only address, or a site run by a contractor or tenant rather
    than the named operator, blocks. Exactly one question is asked (outreach_question), by
    precedence S, M, then A or U. Anything else, including any defect here, is none.
    """
    block = {"rule_version": rule_version, "proving_sources": [], "open_checks": [],
             "open_questions": [], "blockers": []}
    try:
        facts, states = gates["facts"], gates["states"]
        block["proving_sources"] = [facts[name]["proofs"][0] for name in PROVEN_FACTS if facts[name]["proofs"]]
        if gates["eligible_for_qualified_promotion"] is True:
            return {"tier": "verified", "eligible_for_outreach_ready": False, "outreach_ready": block}
        blockers = [name + "_contradicted" for name in CLAIMS if states.get(name) == "contradicted"]
        valid_until = gates["valid_until"]
        for failed, code in ((gates["assessment_valid"] is not True, "assessment_invalid"),
                             (gates["identity_present"] is not True, "identity_missing"),
                             (gates["duplicate"] is not False, "duplicate"),
                             (gates["conflict"] is not False, "duplicate_conflict"),
                             (valid_until is not None and not now < moment(valid_until), "assessment_expired")):
            if failed:
                blockers.append(code)
        for name in PROVEN_FACTS:
            if states.get(name) not in (SITE_TASK_STATES if name == "site_task" else {"verified_fact"}):
                blockers.append(name + ("_not_verified_fact_or_inference" if name == "site_task" else "_not_verified_fact"))
            elif facts[name]["primary_sources_usable"] is not True:
                blockers.append(name + "_primary_source_unusable")
            elif not facts[name]["proofs"]:
                blockers.append(name + "_quote_unproven")
            elif name == "site_task" and states[name] == "verified_fact" and facts[name]["site_specific"] is not True:
                blockers.append("site_task_company_level")
        facility = gates["facility"]
        if facility["valid"] is not True:
            blockers.append("facility_invalid")
        for field, (_, blocking) in FACILITY_FIELDS.items():
            if facility[field]["value"] in blocking and facility[field]["proven"] is True:
                blockers.append(FACILITY_BLOCKERS[field])
        task, site = gates["task"], gates["site"]
        checks = open_checks(states, valid_until)
        question = None
        if not text(task) or not text(site) or rule_version != LEGACY_OUTREACH_RULE_VERSION and not text(question_task(task)):
            blockers.append("question_task_or_site_missing")
        else:
            _, question = outreach_question(checks, task, site, gates.get("location"),
                                            partial_automation=gates.get("partial_automation") is True, rule_version=rule_version)
            if question.count("?") != 1:
                blockers.append("question_not_single")
        if blockers:
            return {"tier": "none", "eligible_for_outreach_ready": False,
                    "outreach_ready": {**block, "blockers": list(dict.fromkeys(blockers))}}
        return {"tier": "outreach_ready", "eligible_for_outreach_ready": True,
                "outreach_ready": {**block, "open_checks": checks, "open_questions": [question]}}
    except Exception:  # noqa: BLE001 - any defect in the tier computation yields none; QA and publication continue
        return {"tier": "none", "eligible_for_outreach_ready": False,
                "outreach_ready": {**block, "proving_sources": [], "blockers": ["tier_computation_unavailable"]}}


def _tier(result, candidate, index, now, conflict, rule_version=OUTREACH_RULE_VERSION):
    try:
        return outreach_tier(outreach_gates(result, candidate, index, conflict=conflict), now, rule_version)
    except Exception:  # noqa: BLE001 - a malformed record is tier none, never a QA failure
        return outreach_tier({}, now, rule_version)


def evidence_summary(evidence):
    """Bounded binding of the evidence a v3 cohort used; review recomputes against the same state."""
    if not isinstance(evidence, dict) or evidence.get("state") != "retained":
        return dict(UNAVAILABLE)
    counts = {kind: len(evidence[kind]) if isinstance(evidence.get(kind), list) else 0 for kind in ("pages", "excerpts")}
    shas = sorted({r["tool_result_sha256"] for kind in ("pages", "excerpts") for r in evidence.get(kind) or []
                   if isinstance(r, dict) and isinstance(r.get("tool_result_sha256"), str)})
    refused = evidence.get("refused")
    return {"schema_version": EVIDENCE_VERSION, "state": "retained", **counts,
            "refused": refused if type(refused) is int and refused >= 0 else 0,
            "sources_sha256": _json_digest(shas)}


TIER_FIELDS = ("tier", "eligible_for_outreach_ready", "outreach_ready")
TIER_COHORT_FIELDS = ("outreach_rule_version", "tier_evidence", "outreach_ready_count")


def without_tier(value):
    """A result-v3 cohort as the v2 cohort it extends: the verified path, which the tier never changes.

    Review binds this part exactly even when it cannot re-prove the tier from retained evidence.
    Anything else is returned unchanged.
    """
    if not isinstance(value, dict) or value.get("result_version") != OUTREACH_RESULT_VERSION or not isinstance(value.get("results"), list):
        return value
    results = [{**{key: item for key, item in r.items() if key not in TIER_FIELDS}, "version": DIAGNOSTIC_RESULT_VERSION}
               if isinstance(r, dict) else r for r in value["results"]]
    return {**{key: item for key, item in value.items() if key not in TIER_COHORT_FIELDS},
            "result_version": DIAGNOSTIC_RESULT_VERSION, "results": results}


def packet_candidates(packet):
    """New packets preserve full duplicate evidence without duplicating arrays."""
    candidates = packet["candidates"]
    if packet.get("verification_cohort_version") == VERSION:
        candidates = [*candidates, *packet.get("duplicates", [])]
        return sorted(candidates, key=lambda c: c["discovery_index"])
    return candidates


def cohort(candidates, assessments, now, actual_cost_usd=None, duplicate_checks=None, *, result_version=DIAGNOSTIC_RESULT_VERSION,
           evidence=None, outreach_rule=OUTREACH_RULE_VERSION):
    """Apply identical criteria to the entire supplied discovery cohort.

    v3 adds each candidate's outreach tier after the duplicate and conflict pass, from
    ``evidence`` (retained_evidence output; None or unavailable proves no quote), under
    ``outreach_rule`` (the current rule; v1.1 only re-derives a row published under it).
    """
    tiered = result_version == OUTREACH_RESULT_VERSION
    if tiered and outreach_rule not in OUTREACH_RULE_VERSIONS:
        raise ValueError("outreach_rule_version_unknown")
    base = DIAGNOSTIC_RESULT_VERSION if tiered else result_version
    results = [evaluate(c, assessments.get(c.get("candidate_key")), now, result_version=base) for c in candidates]
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
    statuses, conflicted = [], []
    for group in groups.values():
        group = sorted(group, key=lambda r: bool(r.get("duplicate_check", {}).get("duplicate")))
        # A duplicate with contradictory assessments cannot inflate verified yield.
        conflict = len({r["status"] for r in group}) > 1
        statuses.append("unresolved" if conflict else group[0]["status"])
        for index, result in enumerate(group):
            if conflict:
                conflicted.append(result)
                result.update(status="unresolved", eligible_for_qualified_promotion=False)
                result["reasons"].append("duplicate assessments conflict: resolve the same operator/site/task before promotion")
            if index:
                result.update(duplicate_of=group[0]["candidate_key"], eligible_for_qualified_promotion=False)
    counts = {name: statuses.count(name) for name in ("verified", "unresolved", "rejected")}
    assessed = sum(r["assessment_valid"] for r in results)
    actual = (type(actual_cost_usd) in {int, float} and math.isfinite(actual_cost_usd) and actual_cost_usd >= 0)
    value = {"criteria_version": VERSION, **({"result_version": result_version} if result_version != RESULT_VERSION else {}), "duplicate_checks": duplicate_checks,
             "candidate_count": len(results), "unique_site_task_candidates": len(groups),
             "duplicates": len(results) - len(groups), "verified_unique_site_task_candidates": counts["verified"],
             "unresolved_unique_site_task_candidates": counts["unresolved"], "rejected_unique_site_task_candidates": counts["rejected"],
             "unresolved_count": sum(r["status"] == "unresolved" for r in results),
             "rejected_count": sum(r["status"] == "rejected" for r in results),
             "assessed_count": assessed, "verification_coverage": assessed / len(results) if results else None,
             "actual_cost_usd": actual_cost_usd if actual else None,
             "verified_unique_per_usd": counts["verified"] / actual_cost_usd if actual and actual_cost_usd > 0 else None,
             "results": results}
    if tiered:
        index = evidence_index(evidence)
        for candidate, result in zip(candidates, results):
            result.update(version=OUTREACH_RESULT_VERSION,
                          **_tier(result, candidate, index, now, any(result is item for item in conflicted), outreach_rule))
        value.update(outreach_rule_version=outreach_rule, tier_evidence=evidence_summary(evidence),
                     outreach_ready_count=sum(r["eligible_for_outreach_ready"] for r in results))
    return value


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
