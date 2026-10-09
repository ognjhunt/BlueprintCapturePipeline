"""Host-owned site-screen admission (ADP-010 partner discovery; design section 3). Standard library only.

Owner decisions 2026-10-05 (company GCS, operations/recovery/2026-10-05/owner-decisions/): outreach-ready site-screen
records move into the CRM as rows labelled Hypothesis, deduplicated against the CRM, through this host-owned
admission (owner-decision-screen-to-crm-and-contacts-20261005.json), and a first small batch spread across task
families goes on to WebApp drafting (owner-decision-first-outreach-batch-20261005.json). The founder sends from his
Gmail Drafts himself. Nothing here sends, drafts, or authorizes a send: ``sends_authorized`` is always false.

Build, on the owner's machine (``operators/site-screen.py admit``), from one private site-screen out dir:
- Selection: outreach-ready screen records, at most ``per_focus`` per task family (default 2) or an explicit key
  list, preferring a verified recipient in the owner's order (``recipient_choice``).
- Each selected screen record, and its contact record, is recomputed from the stored results and page reads and
  must equal its derived record file; a record whose values differ is refused.
- Each result carries ``result_digest`` (over its input, answers and checks), the Parallel run id, each proving
  quote with the SHA-256 of the page text or excerpt that holds it, the record's own one question and template
  (from the screen rule's shared question builder in verification, recomputed with the record), and the chosen
  recipient with its source label and provenance.
- The bundle binds the owner direction (``blueprint.outreach-ready-direction.v1`` naming ``site_screen``, pinned by
  generation and SHA-256). ``admission_id`` is the SHA-256 of the bundle's canonical bytes, so it is a digest of the
  manifest, the results and the direction.

Hand-off. The owner uploads the bundle create-only to ``gs://blueprint-8c1ca.appspot.com/operations/research/
screen-admission/<admission_id>/bundle.json`` with his existing gcloud identity and reads it back
(``GcloudObjects``): no credential is created or moved. In the Render worker shell, ``pin`` reads exactly that
generation with the worker identity, validates it with this package's own loader (``load_bundle``) against the live
direction, and pins ``control.screen_admission`` through the fenced compare-and-swap ``screen_admission_set``.

Processing (``step``, from the worker's scheduler), only when no daily row, QA, repair or publication is active and
only under the worker lease: refresh the canonical CRM; drop sites the CRM already holds (``runner.keys`` and the
publisher's structural duplicate rule) or that an earlier admission named; then the bridge op
``screen_admission_publish`` makes a create-only plan, a claim, one Sheets write and a readback (publisher
``planScreenSheets``), each row marked ``[screen:<admission_id>;<result_digest>]``. A claimed write the readback does
not show is uncertain: it is reconciled by reading only and never written again. The acknowledged admission writes
``screenWorkItems/<admission_id>`` for the WebApp, which reads it with ``Store.screenSnapshot``.
"""
import base64
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

from tools.daily_research import contact_lookup, outreach_ready, runner, site_screen, verification
from tools.daily_research.runner import Refusal, canonical

BUNDLE = "blueprint.site-screen-admission.v1"
PAYLOAD = "blueprint.site-screen-sheets-payload.v1"
RECIPIENT = "blueprint.site-screen-recipient.v1"
OPERATION = "blueprint.site-screen-admission-operation.v1"
BUCKET = outreach_ready.BUCKET
OBJECT_PREFIX = "operations/research/screen-admission/"
OBJECT_NAME = "bundle.json"
PATH = "site_screen"  # The outreach_ready direction path this admission needs (outreach_ready.PATHS).
LABEL = outreach_ready.LABEL
MAX_OBJECT_BYTES = 1024 * 1024  # Mirrored by the bridge; 50 results are about 400 KiB.
MAX_RECORDS = outreach_ready.MAX_ROWS_PER_BATCH
DEFAULT_PER_FOCUS = 2
HEX = re.compile(r"[a-f0-9]{64}")
GENERATION = re.compile(r"[1-9][0-9]{0,18}")
TEXT = re.compile(r"[\x20-\x7e]{1,500}")
CODE = re.compile(r"screen_admission_[a-z_]{1,80}")  # No digits: the worker log admits [a-z_] codes only.
ALWAYS_OPEN = ("existing_automation", "fit", "interest")
# The owner's recipient order (owner decisions 2026-10-05: contact sources, provider lookup, provider-sourced person).
ROUTES = ("published_person_email", "quoted_person_looked_up_email", "provider_sourced_corroborated",
          "provider_sourced_uncorroborated", "published_team_inbox", "published_general_inbox")
PERSON_ROUTES = frozenset(ROUTES[:4])
LOOKUP_ROUTES = frozenset(ROUTES[1:4])
LABELS = {"published_person_email": "published person email",
          "quoted_person_looked_up_email": "looked-up email, quoted person",
          "provider_sourced_corroborated": "looked-up email, provider-sourced person, corroborated",
          "provider_sourced_uncorroborated": "looked-up email, provider-sourced person, not corroborated",
          "published_team_inbox": "published team inbox", "published_general_inbox": "published general inbox"}
PUBLISHED = {"person_email": "published_person_email", "team_inbox": "published_team_inbox",
             "general_inbox": "published_general_inbox"}
ROLE_WORDS = site_screen.TEAM_INBOX | site_screen.GENERAL_INBOX | site_screen.REFUSED_INBOX
# A busy worker releases the lease for about 3 s between passes and a lost lease expires after 180 s.
LEASE_WAIT_SECONDS, LEASE_POLL_SECONDS = 200.0, 0.25
PIN_FIELDS = frozenset({"admission_id", "generation", "bytes", "uri", "records", "direction_sha256",
                        "approval_reference"})
RESULT_FIELDS = frozenset({"site_key", "result_digest", "run_id", "result_sha256", "evidence_sha256", "checked_on",
                           "origin", "calibration", "task_focus", "input", "answers", "checks", "candidate",
                           "hypothesis", "proofs", "contact", "recipient"})
CHECK_FIELDS = ("rule_version", "verification", "choices", "gates", "task_scope", "tier", "blockers", "open_checks",
                "question", "question_template")
CANDIDATE_FIELDS = ("organization", "site", "location", "task", "task_url", "checked_on")


class AdmissionError(Refusal):
    """A stable screen_admission_* code; never site text, a person or an address."""


def _refuse(code):
    raise AdmissionError(code)


def uri(admission_id):
    return f"gs://{BUCKET}/{OBJECT_PREFIX}{admission_id}/{OBJECT_NAME}"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value, code):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        _refuse(code)
    return value


def _text(value, limit=500):
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def reference(value):
    """An owner reference: printable ASCII, never PENDING."""
    if not isinstance(value, str) or not TEXT.fullmatch(value) or not value.strip() or value.strip().upper().startswith(
            "PENDING"):
        _refuse("screen_admission_reference_invalid")
    return value.strip()


def direction_binding(sha256, generation):
    """The owner direction a bundle names: its SHA-256, object generation and URI."""
    _hex(sha256, "screen_admission_direction_invalid")
    if not isinstance(generation, str) or not GENERATION.fullmatch(generation):
        _refuse("screen_admission_direction_invalid")
    return {"generation": generation, "sha256": sha256, "uri": outreach_ready.uri(sha256)}


# --- one result -------------------------------------------------------------------------------------------------
def one_question(value):
    """True for exactly one question: a nonblank line that ends with its only question mark."""
    return _text(value, 600) and "\n" not in value and value.count("?") == 1 and value.endswith("?")


def open_checks(record):
    """The record's open checks in the rule's canonical order (verification.OPEN_CHECKS); any unknown check refuses."""
    checks = [check for check in verification.OPEN_CHECKS if check in record["open_checks"]]
    if len(checks) != len(record["open_checks"]) or len(set(checks)) != len(checks) or not set(ALWAYS_OPEN) <= set(checks):
        _refuse("screen_admission_open_checks_invalid")
    return checks


def checks_of(record):
    return {name: record[name] for name in CHECK_FIELDS}


def result_digest(entry):
    """The digest over one result's input, answers and checks."""
    return runner.digest({"input": entry["input"], "answers": entry["answers"], "checks": entry["checks"]})


def candidate_of(record):
    """What the CRM row and the WebApp name: the proven operator, the site, its location, the task and its proof URL,
    and the date our reader checked the pages."""
    given, answers = record["input"], record["answers"]
    place = (record.get("identity") or {}).get("physical_site") or {}
    site = given.get("site_name") or place.get("answer") or answers.get("site_identity") or given.get("location")
    value = {"organization": answers.get("operator_identity"), "site": site, "location": given.get("location") or site,
             "task": answers.get("target_task"), "task_url": answers.get("target_task_url"),
             "checked_on": record.get("checked_on")}
    if not all(_text(value[name]) for name in CANDIDATE_FIELDS) or not site_screen._public_url(value["task_url"]):
        _refuse("screen_admission_candidate_incomplete")
    return value


def proofs_of(record):
    """Each proving quote with its level and the SHA-256 of the page text or excerpt that holds it."""
    proofs = []
    for item in record["proving_sources"]:
        if item.get("source_id") == "government_record":
            proofs.append({"claim": item["claim"], "source_id": "government_record", "level": "government_record",
                           "source_ids": list(item["source_ids"]), "site_id": item["site_id"]})
            continue
        quote = record["answers"].get(str(item.get("source_id")) + "_quote")
        if (item.get("level") not in site_screen.PROVEN or not isinstance(quote, str) or not _text(item.get("url"), 2000)
                or _sha(quote.encode()) != item.get("quote_sha256")
                or not isinstance(item.get("tool_result_sha256"), str) or not HEX.fullmatch(item["tool_result_sha256"])):
            _refuse("screen_admission_proof_invalid")
        proofs.append({"claim": item["claim"], "source_id": item["source_id"], "url": item["url"],
                       "level": item["level"], "quote": quote, "quote_sha256": item["quote_sha256"],
                       "text_sha256": item["tool_result_sha256"]})
    if [proof["claim"] for proof in proofs] != list(verification.PROVEN_FACTS):
        _refuse("screen_admission_proof_invalid")
    return proofs


def _on_domain(address, domain):
    host = address.rpartition("@")[2]
    return host == domain or host.endswith("." + domain)


def _contact_inputs(workspace, key):
    """The stored contact result's answers, our page reads of it and its evidence index."""
    content, basis = site_screen.output_of(site_screen._json(workspace.path("contact", "results", key).read_bytes()))
    evidence = site_screen._evidence(workspace.path("contact", "evidence", key).read_bytes())
    answers = {field: site_screen._string(content.get(field)) for field in site_screen.CONTACT_SCHEMA["properties"]}
    return answers, evidence, site_screen.evidence_index(evidence, basis)


def person_block(contact, answers, index):
    """The contact stage's verified person, with the quote and the SHA-256 of the text that proves it; else None."""
    person = contact["person"]
    if person.get("verified") is not True:
        return None
    level, sha = site_screen.quote_level(answers["person_quote"], person["url"], index)
    if level not in site_screen.PROVEN or not sha:
        return None
    return {"source": "public_quote", "name": person["name"], "title": person["title"], "url": person["url"],
            "quote": answers["person_quote"], "level": level, "text_sha256": sha, "date": person["date"],
            "current": person["current"]}


def domain_proof(domain, proofs):
    """An operator-domain proof with the page that proves it: for basis operator_quote, the proven operator quote's
    page and the SHA-256 of the text that holds it; for basis website, the page site_screen read on that domain."""
    if domain.get("basis") == "operator_quote":
        operator = next(proof for proof in proofs if proof["claim"] == "operator")
        return {"domain": domain["domain"], "basis": "operator_quote", "url": operator["url"],
                "text_sha256": operator["text_sha256"]}
    return {key: domain[key] for key in ("domain", "basis", "url", "text_sha256")}


def published_recipient(contact, record, proofs, workspace, *, today=None):
    """Route 1, 5 or 6: the contact stage's verified published address, with its page and operator-domain proof."""
    if contact_lookup.choose_recipient(contact, today=today).get("source") != "published":
        return None
    email, route = contact["email"], PUBLISHED.get(contact["recipient"]["kind"])
    domain = email.get("operator_domain")
    if route is None or email.get("verified") is not True or not isinstance(domain, dict):
        return None
    answers, evidence, index = _contact_inputs(workspace, record["site_key"])
    person = person_block(contact, answers, index)
    if route == "published_person_email" and (not person or person.get("current") is not True):
        return None
    page = (evidence.get("pages") or {}).get(email["url"]) or {}
    domain = domain_proof(domain, proofs)
    return {"schema_version": RECIPIENT, "route": route, "rank": ROUTES.index(route) + 1, "label": LABELS[route],
            "address": email["address"], "address_source": "published", "operator_domain": domain,
            "published": {"url": email["url"], "quote": answers["email_quote"], "text_sha256": page.get("sha256"),
                          "level": email["level"], "checked_on": contact["checked_on"]},
            "person": person, "provider": None}


def _provider(value):
    fields = {"name", "status", "score", "checked_at", "request_digest"}
    return (isinstance(value, dict) and set(value) == fields | {"verification_status", "lookup_sha256", "record_sha256"} and value["name"] == "fullenrich" and value["status"] == "valid"
            and (value["score"] is None or type(value["score"]) is int and 0 <= value["score"] <= 100) and _text(value["checked_at"], 64)
            and isinstance(value["request_digest"], str) and bool(HEX.fullmatch(value["request_digest"]))
            and value.get("verification_status") == "DELIVERABLE"
            and all(isinstance(value.get(key), str) and HEX.fullmatch(value[key])
                    for key in ("lookup_sha256", "record_sha256")))


def lookup_recipients(record, contact, domains, person_proof=None, *, today=None):
    """The final contact_lookup selector, enriched with the screen's retained public/domain proofs.
    Cached recipients never grant admission: choose_recipient rechecks the current contact and lookup evidence.
    FullEnrich DELIVERABLE is retained alongside verification=valid; an unavailable score stays None."""
    try:
        if (not isinstance(record, dict) or record.get("schema_version") != contact_lookup.RECORD
                or record.get("rule_version") != contact_lookup.RULE or record.get("site_key") != contact["site_key"]):
            return []
        selected = contact_lookup.choose_recipient(contact, record.get("lookups", ()), today=today)
        if selected.get("source") != "provider_lookup":
            return []
        lookup = next(item for item in record["lookups"] if item.get("address") == selected["address"]
                      and contact_lookup.usable_lookup(item))
        address, person, raw = selected["address"], lookup["person"], lookup["provider"]
        domain = next((item for item in domains if _on_domain(address, item["domain"])), None)
        if domain is None or lookup.get("operator_domain") != domain["domain"]:
            return []
        provider = {"name": raw["name"], "status": raw["verification"], "score": raw.get("score"),
                    "verification_status": raw["status"], "checked_at": raw["checked_at"],
                    "request_digest": raw["request_digest"], "lookup_sha256": runner.digest(lookup),
                    "record_sha256": runner.digest(record)}
        if not _provider(provider):
            return []
        if person["sourcing"] == "quoted_person":
            if (contact["person"].get("current") is not True or not person_proof
                    or site_screen.words(person_proof["name"]) != site_screen.words(person["name"])):
                return []
            route, kept = "quoted_person_looked_up_email", dict(person_proof)
        elif person["sourcing"] == "provider_sourced":
            proof = person.get("corroboration") or {}
            corroborated = proof.get("corroborated") is True
            if corroborated and not (proof.get("level") in site_screen.PROVEN and _text(proof.get("quote"), 1200)
                    and site_screen._public_url(proof.get("url")) and not site_screen.never_fetch(proof.get("url"))
                    and isinstance(proof.get("text_sha256"), str) and HEX.fullmatch(proof["text_sha256"])):
                return []
            route = "provider_sourced_corroborated" if corroborated else "provider_sourced_uncorroborated"
            kept = {"source": "provider_sourced", "name": person["name"], "title": person["title"],
                    "corroborated": corroborated, "proof": person["proof"], "corroboration":
                    {key: proof[key] for key in ("url", "quote", "level", "text_sha256")} if corroborated else None}
        else:
            return []
        kept["site_responsibility"] = person.get("site_responsibility") or {
            "site_key": contact["site_key"], "status": "unknown", "route": "corporate_referral",
            "reason": "target_site_responsibility_unproven", "proof": None}
    except (AttributeError, KeyError, TypeError, StopIteration):
        return []
    return [{"schema_version": RECIPIENT, "route": route, "rank": ROUTES.index(route) + 1, "label": LABELS[route],
             "address": address, "address_source": "provider_lookup", "operator_domain": domain, "published": None,
             "person": kept, "provider": provider}]


def recipient_choice(published, looked_up=()):
    """The first available recipient in the owner's order (ROUTES), or None."""
    choices = [item for item in (published, *looked_up) if item]
    return min(choices, key=lambda item: item["rank"]) if choices else None


def recipient_question(candidate, recipient, question):
    """A referral asks who covers this facility; the screen's task hypothesis remains separately intact."""
    responsibility = ((recipient or {}).get("person") or {}).get("site_responsibility") or {}
    if responsibility.get("route") == "corporate_referral":
        return (f"Could you direct me to the person responsible for {candidate['task']} at "
                f"{candidate['site']} ({candidate['location']})?")
    return question


def responsibility_target(entry):
    """The retained physical-site evidence, including a proven street omitted from the CRM display location."""
    given = {"address": site_screen.parse_location(entry["input"].get("location"))}
    found = site_screen.found_address(given, entry["answers"], entry["checks"]["verification"])
    return {"address": {**given["address"], **(found or {})},
            "operator": entry["answers"].get("operator_identity") or entry["input"].get("operator") or "",
            "task_input": {"site_name": entry["input"].get("site_name") or ""}}


def admission_entry(workspace, record, stored, contact, stored_contact, lookup=None, *, today=None):
    """One bundle result from one outreach-ready screen record and its contact record, both recomputed and equal to
    their derived files; any difference, or a defect, refuses with a stable code."""
    if not isinstance(stored, dict) or canonical(stored) != canonical(record):
        _refuse("screen_admission_record_mismatch")
    if record["tier"] != "outreach_ready" or record["rule_version"] != site_screen.SCREEN_RULE or record["blockers"]:
        _refuse("screen_admission_not_outreach_ready")
    checks = open_checks(record)
    candidate = candidate_of(record)
    # The record's own question, recomputed with it under the current screen rule (its question builder is the shared
    # one in verification); a later wording rule needs only verify, never a new paid run.
    template, text = record["question_template"], record["question"]
    if not one_question(text) or not isinstance(template, str) or not re.fullmatch(r"[A-Z]", template):
        _refuse("screen_admission_question_mismatch")
    proofs = proofs_of(record)
    block = recipient = None
    if contact is not None:
        if not isinstance(stored_contact, dict) or canonical(stored_contact) != canonical(contact):
            _refuse("screen_admission_contact_record_mismatch")
        answers, evidence, index = _contact_inputs(workspace, record["site_key"])
        domains = [domain_proof(item, proofs) for item in site_screen.operator_domain_proofs(contact_site(record),
                   site_screen.screen_context(workspace, record["site_key"]), (evidence.get("pages") or {}).items())]
        recipient = recipient_choice(published_recipient(contact, record, proofs, workspace, today=today),
                                     lookup_recipients(lookup, contact, domains, person_block(contact, answers, index), today=today))
        block = {"run_id": contact["run_id"], "result_sha256": contact["result_sha256"],
                 "evidence_sha256": contact["evidence_sha256"], "rule_version": contact["rule_version"],
                 "checked_on": contact["checked_on"], "decision_role": contact["decision_role"],
                 "recipient_kind": contact["recipient"]["kind"], "email_level": contact["email"]["level"]}
    text = recipient_question(candidate, recipient, text)
    if not one_question(text):
        _refuse("screen_admission_question_mismatch")
    entry = {"site_key": record["site_key"], "run_id": record["run_id"], "result_sha256": record["result_sha256"],
             "evidence_sha256": record["evidence_sha256"], "checked_on": record["checked_on"], "origin": record["origin"],
             "calibration": record["calibration"], "task_focus": record["task_focus"], "input": record["input"],
             "answers": record["answers"], "checks": checks_of(record), "candidate": candidate,
             "hypothesis": {"tier": "outreach_ready", "label": LABEL, "rule_version": record["rule_version"],
                            "outreach_rule_version": verification.OUTREACH_RULE_VERSION, "open_checks": checks,
                            "question_template": template, "question": text},
             "proofs": proofs, "contact": block, "recipient": recipient}
    return {**entry, "result_digest": result_digest(entry)}


def contact_site(record):
    """The contact-stage input of a screen record (site_screen.contact_sites), for the operator-domain proofs."""
    return {"site_key": record["site_key"], "operator_domains": record["operator_domains"]}


def _rank(entry):
    return entry["recipient"]["rank"] if entry["recipient"] else len(ROUTES) + 1


def select(entries, *, per_focus, keys, max_records):
    """The admitted results in screen order: an explicit key list exactly, else at most ``per_focus`` per task family,
    taking verified recipients first in the owner's order and then screen order."""
    if keys is not None:
        found = {entry["site_key"]: entry for entry in entries}
        if any(key not in found for key in keys):
            _refuse("screen_admission_key_not_admissible")
        chosen = {key for key in keys}
    else:
        groups, chosen = Counter(), set()
        for entry in sorted(entries, key=_rank):  # Stable: screen order within a rank.
            focus = entry["task_focus"] or "none"
            if groups[focus] < per_focus:
                groups[focus] += 1
                chosen.add(entry["site_key"])
    result = [entry for entry in entries if entry["site_key"] in chosen]
    if not result:
        _refuse("screen_admission_nothing_to_admit")
    if len(result) > max_records:
        _refuse("screen_admission_batch_above_limit")
    return result


def parse_keys(value):
    """--keys: comma-separated site keys, each once."""
    keys = [part.strip() for part in value.split(",")] if isinstance(value, str) else []
    if not keys or any(not HEX.fullmatch(key) for key in keys) or len(set(keys)) != len(keys):
        _refuse("screen_admission_keys_invalid")
    return keys


def build(workspace, *, direction_sha256, direction_generation, per_focus=DEFAULT_PER_FOCUS, keys=None,
          max_records=MAX_RECORDS, lookups=None):
    """(bundle, canonical bytes, counts-only report) for one admission from one out dir. Reads no page and calls no
    provider; holds the out dir lock while it reads."""
    direction = direction_binding(direction_sha256, direction_generation)
    if keys is None and (type(per_focus) is not int or not 1 <= per_focus <= MAX_RECORDS):
        _refuse("screen_admission_per_focus_invalid")
    if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS:
        _refuse("screen_admission_max_records_invalid")
    with workspace.lock():
        states, pin = workspace.states()
        if pin is None:
            _refuse("screen_admission_out_dir_unpinned")
        as_of = contact_lookup.utc_today()
        loaded = contact_lookup.load(workspace, states=states, today=as_of) if lookups is None else None
        lookup_for = (lambda key: loaded.get(key)) if lookups is None else lookups
        screens = site_screen.stage_records(workspace, states, "screen")
        contacts = site_screen.stage_records(workspace, states, "contact")
        stored = {record.get("site_key"): record for record in workspace.records("screen") if isinstance(record, dict)}
        stored_contacts = {record.get("site_key"): record for record in workspace.records("contact")
                           if isinstance(record, dict)}
        entries, refused = [], Counter()
        for key in states["screen"]:
            record = screens.get(key)
            if record is None or record["tier"] != "outreach_ready":
                continue
            try:
                entries.append(admission_entry(workspace, record, stored.get(key), contacts.get(key),
                                               stored_contacts.get(key), lookup_for(key), today=as_of))
            except AdmissionError as error:
                if keys is not None and key in keys:
                    raise
                refused[str(error)] += 1
        chosen = select(entries, per_focus=per_focus, keys=keys, max_records=max_records)
    manifest = {"schema_version": BUNDLE, "label": LABEL, "path": PATH, "sends_authorized": False,
                "rules": {"screen": site_screen.SCREEN_RULE, "contact": site_screen.CONTACT_RULE,
                          "outreach": verification.OUTREACH_RULE_VERSION},
                "owner_ceiling_sha256": runner.digest(pin),
                "selection": {"per_focus": per_focus if keys is None else None,
                              "keys": sorted(keys) if keys is not None else None, "max_records": max_records},
                "records": len(chosen)}
    bundle = {"direction": direction, "manifest": manifest, "results": chosen}
    raw = canonical(bundle).encode()
    if len(raw) > MAX_OBJECT_BYTES:
        _refuse("screen_admission_bundle_too_large")
    admission_id = _sha(raw)
    report = {"schema_version": OPERATION, "command": "admit", "state": "planned", "admission_id": admission_id,
              "uri": uri(admission_id), "bytes": len(raw), "records": len(chosen), "eligible": len(entries),
              "refused": dict(refused), "direction": direction,
              "by_focus": dict(Counter(entry["task_focus"] or "none" for entry in chosen)),
              "by_route": dict(Counter((entry["recipient"] or {}).get("route", "none") for entry in chosen)),
              "calibration": sum(entry["calibration"] is True for entry in chosen), "sends_authorized": False,
              "provider_calls": 0, "object_writes": 0}
    return bundle, raw, report


# --- the package's own loader -----------------------------------------------------------------------------------
def _url(value):
    return site_screen._public_url(value) and not site_screen.never_fetch(value)


def recipient_problem(value, entry):
    """None for a recipient in exactly the shape ``recipient_choice`` makes, else a stable code."""
    code = "screen_admission_recipient_invalid"
    try:
        fields = {"schema_version", "route", "rank", "label", "address", "address_source", "operator_domain",
                  "published", "person", "provider"}
        route = value["route"]
        if (set(value) != fields or value["schema_version"] != RECIPIENT or route not in ROUTES
                or value["rank"] != ROUTES.index(route) + 1 or value["label"] != LABELS[route]):
            return code
        address, domain = site_screen.email_address(value["address"]), value["operator_domain"]
        if (address != value["address"] or not isinstance(domain, dict) or domain.get("basis") not in ("operator_quote", "website")
                or not _text(domain.get("domain"), 253) or not _on_domain(address, domain["domain"])
                or address.rpartition("@")[2] in site_screen.FREE_MAIL or not _url(domain.get("url"))
                or not isinstance(domain.get("text_sha256"), str) or not HEX.fullmatch(domain["text_sha256"])
                or set(domain) != {"domain", "basis", "url", "text_sha256"}):
            return code
        person = value["person"]
        if route in LOOKUP_ROUTES:
            if (value["address_source"] != "provider_lookup" or value["published"] is not None
                    or not _provider(value["provider"]) or not isinstance(person, dict)
                    or set(re.findall(r"[a-z]+", address.partition("@")[0])) & ROLE_WORDS
                    or site_screen.address_role(address, {"verified": True, "name": person.get("name")}) != "person"):
                return code
            wanted = "public_quote" if route == "quoted_person_looked_up_email" else "provider_sourced"
            if person.get("source") != wanted or not _text(person.get("name"), 200) or not _text(person.get("title"), 200):
                return code
            responsibility = person.get("site_responsibility")
            if (not isinstance(responsibility, dict)
                    or set(responsibility) != {"site_key", "status", "route", "reason", "proof"}
                    or responsibility["site_key"] != entry["site_key"]):
                return code
            if responsibility["route"] == "corporate_referral":
                if (responsibility["status"] != "unknown" or responsibility["proof"] is not None
                        or responsibility["reason"] not in
                        ("target_site_responsibility_unproven", "target_site_location_mismatch")):
                    return code
                target = {**responsibility_target(entry), "site_key": entry["site_key"], "kept_pages": [],
                          "person_quote": person.get("quote") or (person.get("corroboration") or {}).get("quote") or ""}
                if (contact_lookup.site_responsibility(target, person)["route"] == "hold"
                        or responsibility["reason"] == "target_site_location_mismatch"
                        and not contact_lookup.corporate_referral_title(person["title"])):
                    return code
            elif responsibility["route"] == "site_contact":
                scope = responsibility["proof"]
                target = responsibility_target(entry)
                if (responsibility["status"] != "verified" or responsibility["reason"] is not None
                        or not isinstance(scope, dict) or scope.get("level") != "verified_on_page"
                        or not _url(scope.get("url")) or not _text(scope.get("quote"), 1200)
                        or not isinstance(scope.get("text_sha256"), str) or not HEX.fullmatch(scope["text_sha256"])
                        or not contact_lookup.site_role_quote(target, person, scope["quote"])):
                    return code
            else:
                return code
            if route == "quoted_person_looked_up_email" and (
                    not _url(person.get("url")) or person.get("level") not in site_screen.PROVEN
                    or not _text(person.get("quote"), 1200) or not isinstance(person.get("text_sha256"), str)
                    or not HEX.fullmatch(person["text_sha256"]) or person.get("current") is not True
                    or not site_screen._fresh(person.get("date"), date.fromisoformat(value["provider"]["checked_at"][:10]))
                    or not site_screen.has_phrase(site_screen.words(person["name"]), site_screen.words(person["quote"]))
                    or not contact_lookup.holds_title(person["quote"], person["title"])):
                return code
            if route == "provider_sourced_corroborated":
                proof = person.get("corroboration")
                if (person.get("corroborated") is not True or not isinstance(proof, dict) or not _url(proof.get("url"))
                        or proof.get("level") not in site_screen.PROVEN or not _text(proof.get("quote"), 1200)
                        or not isinstance(proof.get("text_sha256"), str) or not HEX.fullmatch(proof["text_sha256"])
                        or not site_screen.has_phrase(site_screen.words(person["name"]), site_screen.words(proof["quote"]))
                        or not contact_lookup.holds_title(proof["quote"], person["title"])):
                    return code
            if route in ("provider_sourced_corroborated", "provider_sourced_uncorroborated"):
                proof = person.get("proof") or {}
                employment = proof.get("current_employment") or {}
                if (proof.get("source") != "fullenrich_people_search"
                        or not isinstance(proof.get("request_digest"), str) or not HEX.fullmatch(proof["request_digest"])
                        or employment.get("field") not in ("employment.current.is_current", "employment.current.end_at_absent",
                                                           "employment.all.is_current")
                        or not isinstance(employment.get("company_domain"), str)
                        or not _on_domain("person@" + employment["company_domain"], domain["domain"])):
                    return code
            if route == "provider_sourced_uncorroborated" and (person.get("corroborated") is not False
                                                                or person.get("corroboration") is not None):
                return code
            return None
        published = value["published"]
        if (value["address_source"] != "published" or value["provider"] is not None or not isinstance(published, dict)
                or set(published) != {"url", "quote", "text_sha256", "level", "checked_on"}
                or not _url(published["url"]) or published["level"] != "verified_on_page"
                or not isinstance(published["quote"], str) or address not in site_screen.addresses(published["quote"])
                or not isinstance(published["text_sha256"], str) or not HEX.fullmatch(published["text_sha256"])
                or published["checked_on"] != (entry["contact"] or {}).get("checked_on")):
            return code
        if person is not None and (person.get("source") != "public_quote" or not _url(person.get("url"))
                                   or person.get("level") not in site_screen.PROVEN or not _text(person.get("name"), 200)):
            return code
        expected_role = {"published_person_email": "person", "published_team_inbox": "team", "published_general_inbox": "general"}[route]
        if site_screen.address_role(address, {"verified": bool(person), "name": (person or {}).get("name")}) != expected_role:
            return code
        if route == "published_person_email" and (not person or person.get("current") is not True
                or not site_screen._fresh(person.get("date"), date.fromisoformat(published["checked_on"]))
                or not site_screen.has_phrase(site_screen.words(person.get("name")), site_screen.words(person.get("quote")))
                or not contact_lookup.holds_title(person.get("quote", ""), person.get("title", ""))):
            return code
    except (AttributeError, KeyError, TypeError, ValueError):
        return code
    return None


def entry_problem(entry):
    """None for one well-formed bundle result whose digest, open checks and question hold, else a stable code."""
    try:
        if not isinstance(entry, dict) or set(entry) != RESULT_FIELDS or not HEX.fullmatch(entry["site_key"]):
            return "screen_admission_result_invalid"
        if entry["result_digest"] != result_digest(entry):
            return "screen_admission_result_digest_mismatch"
        checks, candidate, hypothesis = entry["checks"], entry["candidate"], entry["hypothesis"]
        answers, given = entry["answers"], entry["input"]
        if (set(checks) != set(CHECK_FIELDS) or checks["tier"] != "outreach_ready" or checks["blockers"]
                or checks["rule_version"] != site_screen.SCREEN_RULE
                or set(candidate) != set(CANDIDATE_FIELDS) or not all(_text(candidate[name]) for name in CANDIDATE_FIELDS)
                or not site_screen._public_url(candidate["task_url"])
                # The candidate repeats the proven answers and the input exactly (candidate_of).
                or (candidate["organization"], candidate["task"], candidate["task_url"], candidate["checked_on"])
                != (answers.get("operator_identity"), answers.get("target_task"), answers.get("target_task_url"),
                    entry["checked_on"])
                or candidate["location"] != (given.get("location") or candidate["site"])):
            return "screen_admission_result_invalid"
        ordered = [check for check in verification.OPEN_CHECKS if check in checks["open_checks"]]
        if (hypothesis != {"tier": "outreach_ready", "label": LABEL, "rule_version": site_screen.SCREEN_RULE,
                           "outreach_rule_version": verification.OUTREACH_RULE_VERSION, "open_checks": ordered,
                           "question_template": checks["question_template"],
                           "question": recipient_question(candidate, entry["recipient"], checks["question"])}
                or len(ordered) != len(checks["open_checks"]) or not set(ALWAYS_OPEN) <= set(ordered)
                or not one_question(checks["question"])):
            return "screen_admission_question_mismatch"
        if [proof.get("claim") for proof in entry["proofs"]] != list(verification.PROVEN_FACTS) or any(
                proof.get("level") != "government_record" and (
                    _sha(str(proof.get("quote")).encode()) != proof.get("quote_sha256") or not _url(proof.get("url")))
                for proof in entry["proofs"]):
            return "screen_admission_proof_invalid"
        if entry["recipient"] is not None:
            if entry["contact"] is None:
                return "screen_admission_recipient_invalid"
            return recipient_problem(entry["recipient"], entry)
    except (AttributeError, KeyError, TypeError, ValueError):
        return "screen_admission_result_invalid"
    return None


def load_bundle(raw, admission_id=None):
    """The bundle in ``raw``: canonical bytes whose SHA-256 is ``admission_id``, made under this package's rules, at
    most MAX_RECORDS results with unique sites, each with a valid digest, question and recipient. Else a code."""
    if not isinstance(raw, (bytes, bytearray)) or not raw or len(raw) > MAX_OBJECT_BYTES:
        _refuse("screen_admission_bundle_invalid")
    raw = bytes(raw)
    if admission_id is not None and _sha(raw) != admission_id:
        _refuse("screen_admission_digest_mismatch")
    try:
        bundle = json.loads(raw)
    except (UnicodeDecodeError, ValueError, RecursionError):
        _refuse("screen_admission_bundle_invalid")
    if not isinstance(bundle, dict) or canonical(bundle).encode() != raw:
        _refuse("screen_admission_bundle_not_canonical")
    try:
        manifest, results, direction = bundle["manifest"], bundle["results"], bundle["direction"]
        if set(bundle) != {"direction", "manifest", "results"} or manifest.get("schema_version") != BUNDLE:
            _refuse("screen_admission_bundle_invalid")
        if manifest.get("rules") != {"screen": site_screen.SCREEN_RULE, "contact": site_screen.CONTACT_RULE,
                                     "outreach": verification.OUTREACH_RULE_VERSION}:
            _refuse("screen_admission_rule_mismatch")
        if (manifest.get("label") != LABEL or manifest.get("path") != PATH or manifest.get("sends_authorized") is not False
                or not isinstance(results, list) or not 1 <= len(results) <= MAX_RECORDS
                or manifest.get("records") != len(results) or not HEX.fullmatch(str(manifest.get("owner_ceiling_sha256")))
                or direction != direction_binding(direction.get("sha256"), direction.get("generation"))):
            _refuse("screen_admission_bundle_invalid")
    except (AttributeError, KeyError, TypeError):
        _refuse("screen_admission_bundle_invalid")
    if len({entry.get("site_key") if isinstance(entry, dict) else None for entry in results}) != len(results):
        _refuse("screen_admission_bundle_invalid")
    for entry in results:
        code = entry_problem(entry)
        if code:
            _refuse(code)
    return bundle


# --- dedupe and the Sheets payload ------------------------------------------------------------------------------
def identity(candidate):
    """The CRM identity runner.keys gives this candidate."""
    return next(iter(runner.keys(candidate)))


def structural_duplicate(candidate, rows):
    """The publisher's structural duplicate rule (publisher.mjs planSheets): the same operator, site or location and
    task, or the same operator, site and task on a row without a location."""
    norm = verification.normalized
    wanted = [norm(candidate["organization"]), norm(candidate["site"] or candidate["location"]),
              norm(candidate["location"] or candidate["site"]), norm(candidate["task"])]
    for row in rows:
        cells = [str(row[index]) if len(row) > index else "" for index in (1, 3, 17, 14)]
        organization, site, location, task = (norm(cell) for cell in cells)
        if [organization, site or location, location or site, task] == wanted:
            return True
        if not location and (organization, site, task) == (wanted[0], norm(candidate["site"]), wanted[3]):
            return True
    return False


def plan_entries(bundle, snapshot, known, history, admission_id):
    """(entries to write, duplicates) against a fresh CRM snapshot and earlier admissions, in bundle order."""
    rows = [row for row in snapshot["values"][5:] if row and any(str(cell).strip() for cell in row)]
    earlier_sites, earlier_identities = set(), set()
    for item in history if isinstance(history, list) else ():
        if isinstance(item, dict) and item.get("admission_id") != admission_id:
            earlier_sites.update(item.get("site_keys") or [])
            earlier_identities.update(item.get("identities") or [])
    entries, duplicates, seen = [], [], set()
    for result in bundle["results"]:
        candidate, key = result["candidate"], identity(result["candidate"])
        code = ("screen_admission_site_already_admitted" if result["site_key"] in earlier_sites
                else "screen_admission_crm_duplicate" if key in known or structural_duplicate(candidate, rows)
                else "screen_admission_identity_already_admitted" if key in earlier_identities
                else "screen_admission_batch_duplicate" if key in seen else None)
        if code:
            duplicates.append({"site_key": result["site_key"], "code": code})
            continue
        seen.add(key)
        entries.append(result)
    return entries, duplicates


def contact_cells(recipient):
    """Sheets columns E, F and H: the person's name for a person route, the address with its label, and the page
    that publishes the address or proves the person (empty for a provider-sourced person not corroborated)."""
    if not recipient:
        return {"name": "", "details": "", "source_url": ""}
    person = recipient.get("person") or {}
    responsibility = person.get("site_responsibility") or {}
    source = ((recipient.get("published") or {}).get("url") or (person.get("corroboration") or {}).get("url")
              or person.get("url") or "")
    suffix = "; corporate referral, target-site responsibility unknown" if responsibility.get("route") == "corporate_referral" else ""
    return {"name": person.get("name", "") if recipient["route"] in PERSON_ROUTES else "",
            "details": f"{recipient['address']} ({recipient['label']}{suffix})", "source_url": source}


def sheets_payload(admission_id, entries, duplicates):
    """The exact input of publisher planScreenSheets: one row per entry in the existing 19 columns."""
    return {"schema_version": PAYLOAD, "admission_id": admission_id, "sheet_id": runner.SHEET, "tab": "Prospects",
            "entries": [{"site_key": entry["site_key"], "result_digest": entry["result_digest"],
                         "identity": identity(entry["candidate"]),
                         **{name: entry["candidate"][name] for name in ("organization", "site", "location", "task",
                                                                        "task_url", "checked_on")},
                         "question": entry["hypothesis"]["question"], "contact": contact_cells(entry["recipient"])}
                        for entry in entries],
            "duplicates": duplicates}


# --- the pin, the direction gate and the worker step -------------------------------------------------------------
def pin_problem(value):
    """None for a control.screen_admission value in exactly the pinned shape, else a stable code."""
    if not isinstance(value, dict) or set(value) != {"enabled", "current"} or type(value["enabled"]) is not bool:
        return "screen_admission_pin_invalid"
    current = value["current"]
    if (not isinstance(current, dict) or set(current) != PIN_FIELDS
            or not isinstance(current["admission_id"], str) or not HEX.fullmatch(current["admission_id"])
            or not isinstance(current["generation"], str) or not GENERATION.fullmatch(current["generation"])
            or type(current["bytes"]) is not int or not 1 <= current["bytes"] <= MAX_OBJECT_BYTES
            or current["uri"] != uri(current["admission_id"]) or type(current["records"]) is not int
            or not 1 <= current["records"] <= MAX_RECORDS or not isinstance(current["direction_sha256"], str)
            or not HEX.fullmatch(current["direction_sha256"])):
        return "screen_admission_pin_invalid"
    try:
        reference(current["approval_reference"])
    except AdmissionError:
        return "screen_admission_pin_invalid"
    return None


def gate(control, bundle, now):
    """The live owner direction this bundle may be admitted under now: enabled, verified, exactly the bundle's pinned
    generation and SHA-256, naming site_screen, effective and unexpired, and at least as wide as the batch."""
    entry, code = outreach_ready.current(control)
    if code:
        _refuse("screen_admission_direction_disabled" if code == "outreach_ready_disabled"
                else "screen_admission_direction_invalid")
    direction = entry["direction"]
    if (entry["sha256"], entry["generation"]) != (bundle["direction"]["sha256"], bundle["direction"]["generation"]):
        _refuse("screen_admission_direction_changed")
    if PATH not in direction["scope"]["paths"]:
        _refuse("screen_admission_path_not_directed")
    if not outreach_ready.stamp(direction["effective_from"]) <= now < outreach_ready.stamp(direction["expires_at"]):
        _refuse("screen_admission_direction_expired")
    if len(bundle["results"]) > direction["scope"]["max_rows_per_batch"]:
        _refuse("screen_admission_batch_above_direction_limit")
    return entry


def direction_record(entry):
    """The direction an admission was processed under, as the admission state and the WebApp record it."""
    direction, scope = entry["direction"], entry["direction"]["scope"]
    return {"sha256": entry["sha256"], "generation": entry["generation"], "uri": entry["uri"], "version": entry["version"],
            "rule_version": direction["rule_version"], "paths": list(scope["paths"]), "label": scope["label"],
            "max_rows_per_batch": scope["max_rows_per_batch"], "sends_authorized": False,
            "effective_from": direction["effective_from"], "expires_at": direction["expires_at"],
            "approval_reference": direction["approval_reference"]}


def read_object(bridge, admission_id, generation, size=None):
    """Exactly the pinned generation, through the worker identity; the bridge checks size and SHA-256."""
    value = bridge.call("screen_admission_object_get", sha256=admission_id, generation=generation, size=size)
    return base64.b64decode(value["data"], validate=True)


def idle(bridge):
    """True when daily/QA/publication work is idle and the communications lap is explicitly drained."""
    return (not bridge.call("summary").get("unfinished") and not bridge.call("active_qa")
            and not bridge.call("work_item") and bridge.call("communications_lap_idle") is True)


@contextmanager
def lease(bridge, sleep=time.sleep, monotonic=time.monotonic, wait=LEASE_WAIT_SECONDS):
    """The existing fenced lease, held only for the swap; a worker holding it is waited for."""
    deadline = monotonic() + wait
    while True:
        try:
            bridge.call("acquire", scope="research_release")
            break
        except Refusal as exc:
            if str(exc) != "runner_overlap" or monotonic() >= deadline:
                raise
            sleep(LEASE_POLL_SECONDS)
    try:
        yield
    finally:
        bridge.call("release")


# Passes the scheduler repeats every five minutes: daily work active, a claimed write still being read back, a plan made
# again, or a refusal a later pass can clear. Any other refusal waits for a control change (a new pin or direction).
RETRY_STATES = frozenset({"screen_admission_waiting", "screen_admission_write_uncertain",
                          "screen_admission_replan_required"})
TRANSIENT = frozenset({"runner_overlap", "communications_worker_lap_active", "firestore_lease_lost",
                       "firestore_bridge_deadline", "firestore_bridge_unavailable",
                       "canonical_crm_read_unavailable", "crm_snapshot_missing_incomplete_or_stale",
                       "screen_admission_object_unavailable", "screen_admission_crm_changed",
                       "screen_admission_crm_duplicate_changed", "screen_admission_state_changed",
                       "screen_admission_readback_unavailable", "screen_admission_plan_unavailable",
                       "screen_admission_unavailable", "publication_crm_changed_before_write",
                       "publication_target_cells_not_plain_empty"})


def retry(result):
    """True when the scheduler should run another pass in five minutes."""
    return isinstance(result, dict) and (result.get("state") in RETRY_STATES or (
        result.get("state") == "screen_admission_blocked" and result.get("error") in TRANSIENT))


def step(bridge, *, now, root):
    """One worker pass for the pinned admission (render.py scheduler): None when nothing is pinned, else a counts-only
    result. Never raises: a refusal is a stable code, retried on a later pass."""
    try:
        return _step(bridge, now=now, root=Path(root))
    except Refusal as error:
        code = str(error)
        return {"state": "screen_admission_blocked",
                "error": code if re.fullmatch(r"[a-z_]{1,100}", code) else "screen_admission_unavailable"}
    except Exception:  # noqa: BLE001 - an admission never stops the daily worker; fixed codes only
        return {"state": "screen_admission_blocked", "error": "screen_admission_unavailable"}


def _step(bridge, *, now, root):
    # The worker's Firestore adapter, imported on use.
    from tools.daily_research.firestore import FirestoreLedger

    pin = (bridge.call("control") or {}).get("screen_admission")
    if pin is None:
        return None
    code = pin_problem(pin)
    if code:
        _refuse(code)
    if pin["enabled"] is not True:
        return {"state": "screen_admission_disabled"}
    current = pin["current"]
    admission_id = current["admission_id"]
    known = bridge.call("screen_admission_get", admission_id=admission_id)
    if known and known.get("state") == "acknowledged":
        return {"state": "screen_admission_acknowledged", "rows": known.get("rows"), "duplicates": known.get("duplicates")}
    if not idle(bridge):
        return {"state": "screen_admission_waiting", "error": "screen_admission_daily_work_active"}
    ledger = FirestoreLedger(bridge, lease_scope="research_release")
    with ledger.lock():
        if not idle(bridge):
            return {"state": "screen_admission_waiting", "error": "screen_admission_daily_work_active"}
        control = bridge.call("control") or {}
        if control.get("screen_admission") != pin:
            return {"state": "screen_admission_waiting", "error": "screen_admission_pin_changed"}
        raw = read_object(bridge, admission_id, current["generation"], current["bytes"])
        bundle = load_bundle(raw, admission_id)
        request = {"admission_id": admission_id, "generation": current["generation"],
                   "direction": direction_record(gate(control, bundle, now))}
        known = bridge.call("screen_admission_get", admission_id=admission_id)
        if not known or known.get("state") == "replan":
            bridge.call("refresh_crm")
            path = root / "screen-admission-crm.json"
            path.write_bytes(ledger.read_bytes("crm.json"))
            snapshot, keys = runner.crm_snapshot(path, now)
            entries, duplicates = plan_entries(bundle, snapshot, keys, bridge.call("screen_admission_history"),
                                               admission_id)
            request.update(payload=sheets_payload(admission_id, entries, duplicates),
                           crm_values_sha256=runner.digest(snapshot["values"]))
        result = bridge.call("screen_admission_publish", **request)
    state = result.get("state") if isinstance(result, dict) else None
    if not isinstance(state, str) or not re.fullmatch(r"[a-z_]{1,60}", state):
        _refuse("screen_admission_result_invalid")
    return {"state": "screen_admission_" + state, "rows": result.get("rows"), "duplicates": result.get("duplicates")}


def pin_ids(value):
    return dict(value["current"], enabled=value["enabled"]) if isinstance(value, dict) and isinstance(
        value.get("current"), dict) else None


def show(bridge, *, now):
    """The pin, its verified object and direction, the admission's state and whether the worker is idle; reads only."""
    control = bridge.call("control") or {}
    value = control.get("screen_admission")
    result = {"schema_version": OPERATION, "command": "admission-show", "state": "unset", "firestore_writes": 0,
              "object_writes": 0, "provider_calls": 0, "sends_authorized": False}
    if value is None:
        return result
    code = pin_problem(value)
    if code:
        return {**result, "state": "unverified", "code": code}
    current = value["current"]
    result.update(state="enabled" if value["enabled"] else "disabled", pin=pin_ids(value),
                  admission=bridge.call("screen_admission_get", admission_id=current["admission_id"]),
                  worker_idle=idle(bridge))
    try:
        bundle = load_bundle(read_object(bridge, current["admission_id"], current["generation"], current["bytes"]),
                             current["admission_id"])
        gate(control, bundle, now)
        result.update(object_verified=True, direction="ok")
    except Refusal as error:
        result.update(object_verified=False, direction=str(error) if CODE.fullmatch(str(error)) else
                      "screen_admission_unavailable")
    return result


def active_run(bridge):
    return bool(bridge.call("summary").get("unfinished") or bridge.call("active_qa"))


def pin(bridge, *, admission_id, generation, approval_reference, expect=None, supersede_uncertain=False, apply=False,
        during_active_run=False, now, sleep=time.sleep, monotonic=time.monotonic):
    """Pin one uploaded bundle after this package's loader and the live direction accept it (worker shell)."""
    _hex(admission_id, "screen_admission_id_invalid")
    if not isinstance(generation, str) or not GENERATION.fullmatch(generation):
        _refuse("screen_admission_generation_invalid")
    approval = reference(approval_reference)
    control = bridge.call("control") or {}
    current = (control.get("screen_admission") or {}).get("current") if isinstance(
        control.get("screen_admission"), dict) else None
    observed = current.get("admission_id") if isinstance(current, dict) else None
    if expect is not None and expect != (observed or "none"):
        _refuse("screen_admission_control_conflict")
    raw = read_object(bridge, admission_id, generation)
    bundle = load_bundle(raw, admission_id)
    entry = gate(control, bundle, now)
    value = {"enabled": True, "current": {"admission_id": admission_id, "generation": generation, "bytes": len(raw),
                                          "uri": uri(admission_id), "records": len(bundle["results"]),
                                          "direction_sha256": entry["sha256"], "approval_reference": approval}}
    result = {"schema_version": OPERATION, "command": "admission-pin", "apply": apply, "expected_admission_id": observed,
              "pin": pin_ids(value), "records": len(bundle["results"]), "provider_calls": 0, "object_writes": 0,
              "sends_authorized": False}
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0}
    if not during_active_run and active_run(bridge):
        _refuse("screen_admission_run_active_apply_after_run")
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("screen_admission_set", expected_admission_id=observed, value=value,
                              supersede_uncertain=supersede_uncertain is True)
    if (bridge.call("control") or {}).get("screen_admission") != value:
        _refuse("screen_admission_readback_failed")
    return {**result, "state": "pinned", "receipt": receipt, "readback_verified": True, "firestore_writes": 1}


def disable(bridge, *, apply=False, sleep=time.sleep, monotonic=time.monotonic):
    """The brake: from the worker's next pass the pinned admission gets no plan, claim, write, readback or hand-off
    until a pin enables it again; a write already sent cannot be recalled."""
    value = (bridge.call("control") or {}).get("screen_admission")
    result = {"schema_version": OPERATION, "command": "admission-disable", "apply": apply, "provider_calls": 0,
              "object_writes": 0, "sends_authorized": False}
    if not isinstance(value, dict) or value.get("enabled") is not True:
        return {**result, "state": "already_disabled" if value is not None else "unset", "firestore_writes": 0}
    if not apply:
        return {**result, "state": "planned", "firestore_writes": 0, "pin": pin_ids(value)}
    braked = {"enabled": False, "current": value["current"]}
    with lease(bridge, sleep, monotonic):
        receipt = bridge.call("screen_admission_set", expected_admission_id=value["current"]["admission_id"],
                              value=braked, supersede_uncertain=False)
    if (bridge.call("control") or {}).get("screen_admission") != braked:
        _refuse("screen_admission_readback_failed")
    return {**result, "state": "disabled", "receipt": receipt, "readback_verified": True, "firestore_writes": 1}


# --- upload with the owner's existing gcloud identity -----------------------------------------------------------
class GcloudObjects:
    """The owner's existing gcloud login: a create-only upload, the stored generation and a byte readback. No
    credential is created, moved or read here; the command runs only where the owner is already signed in."""

    def __init__(self, run=subprocess.run, gcloud="gcloud", timeout=120):
        self._run, self.gcloud, self.timeout = run, gcloud, timeout

    def _call(self, *args, binary=False):
        try:
            done = self._run([self.gcloud, *args], capture_output=True, timeout=self.timeout, check=False,
                             **({} if binary else {"text": True}))
        except (OSError, subprocess.SubprocessError):
            _refuse("screen_admission_upload_unavailable")
        return done

    def create(self, admission_id, path):
        """(generation, size) of the object; an existing object is kept and read back by the caller."""
        self._call("storage", "cp", "--if-generation-match=0", "--content-type=application/json", str(path),
                   uri(admission_id))
        described = self._call("storage", "objects", "describe", uri(admission_id), "--format=json")
        try:
            meta = json.loads(described.stdout) if described.returncode == 0 else None
            generation, size = str(meta["generation"]), int(meta["size"])
        except (KeyError, TypeError, ValueError):
            _refuse("screen_admission_upload_failed")
        if not GENERATION.fullmatch(generation):
            _refuse("screen_admission_upload_failed")
        return generation, size

    def read(self, admission_id, generation):
        done = self._call("storage", "cat", f"{uri(admission_id)}#{generation}", binary=True)
        if done.returncode != 0 or not isinstance(done.stdout, bytes):
            _refuse("screen_admission_readback_failed")
        return done.stdout


def admit(workspace, *, direction_sha256, direction_generation, per_focus=DEFAULT_PER_FOCUS, keys=None,
          max_records=MAX_RECORDS, apply=False, objects=None, lookups=None):
    """Build one admission bundle (dry run), or with ``apply`` keep it write-once in the out dir, upload it create-only
    and read it back. Counts and digests only; never a site name, person or address."""
    _, raw, report = build(workspace, direction_sha256=direction_sha256, direction_generation=direction_generation,
                           per_focus=per_focus, keys=keys, max_records=max_records, lookups=lookups)
    if not apply:
        return report
    admission_id = report["admission_id"]
    path = workspace.root / "admissions" / f"{admission_id}.json"
    site_screen._write_once(path, raw)
    if path.read_bytes() != raw:
        _refuse("screen_admission_local_copy_mismatch")
    objects = objects or GcloudObjects()
    generation, size = objects.create(admission_id, path)
    if size != len(raw) or _sha(objects.read(admission_id, generation)) != admission_id:
        _refuse("screen_admission_readback_failed")
    return {**report, "state": "uploaded", "generation": generation, "object_writes": 1, "readback_verified": True,
            "next": "site-screen.py admission-pin --admission-id ADMISSION --generation GENERATION "
                    "--approval-reference REF (Render worker shell)"}


def at(value):
    """A timezone-aware datetime from an ISO string (operator input)."""
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        _refuse("screen_admission_time_invalid")
    return moment
