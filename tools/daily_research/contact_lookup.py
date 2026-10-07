"""Contact lookup: a provider-verified work email for a real person at each contact site of a site-screen out dir.
Standard library only. One provider: FullEnrich (API v2, ``app.fullenrich.com``).

Owner decisions 2026-10-05 (company GCS ``operations/recovery/2026-10-05/owner-decisions/``).
``owner-decision-contact-provider-lookup-20261005.json`` amends ``owner-decision-contact-sources-20261005.json``: a paid
contact-data provider may look up the business email of a named, current person in the deciding role. The person still
comes from a reputable public source with a verified quote; LinkedIn is never evidence.
``owner-decision-provider-sourced-person-20261005.json``: a people database may find that person, labelled
``provider_sourced``, and we try to corroborate the role on a page we read ourselves and record whether that worked.
Only an address the provider marks verified, on the operator's own domain or a subdomain, is kept: never free mail, a
personal address or a role inbox. It is labelled looked up, not published. Guessing an address from a name pattern
without provider verification stays forbidden. Nothing here sends, drafts or writes a CRM; the founder sends every
email himself.

For each contact record of the out dir (``site_screen.stage_records``, under the current contact rule), in contact order:

* a site whose contact stage holds a published, verified person email is left alone (``published_person_email``);
* a named person the contact stage proved by a quote (``site_screen.PROVEN``) whose dated source is at most 18 months
  old gets one work-email enrichment (``quoted_person``);
* otherwise, and only when the run names ``PERSON_SEARCH_DECISION``, one people search on the operator's proven domain
  for ``TITLES`` keeps minimal current-role candidates, qualified separately for each target facility. A proven site
  contact takes priority over an explicitly unknown corporate referral; mismatched local managers are held. The selected
  person gets one enrichment (``provider_sourced``). The employment field relied on is recorded, and the pages the site screen
  already read and kept for the site are checked for the person with their title (``corroboration``).

A work email counts only when FullEnrich marks it ``DELIVERABLE`` (``HIGH_PROBABILITY`` is its catch-all estimate, and
``CATCH_ALL`` and ``INVALID`` are not verified either), it is on the operator's domain or a subdomain, never free mail,
and its local part names the person (``site_screen.address_role``; never a role inbox), and the enrichment's own
profile, where it gives one, names the same person at the operator now. A deliverable address never makes up for the
wrong person. Every other address is dropped before anything is written; personal emails and phones are never asked for.

Spend. The first ``--apply`` pins the owner reference, a FullEnrich credit ceiling and a call limit in
``lookup/owner_ceiling.json``, once; a later run may only lower them. Each call is admitted against the pin, and its
``intent`` is fsynced to ``lookup/spend.jsonl`` (whose first line is the pin) before the call is sent: the credits of
answered calls, the most a call may still cost while its outcome or result is unknown, and the new call stay within the
ceiling. A lookup (one people search per domain, one enrichment per person and domain) is never sent twice; an
enrichment that may exist is only read again. Only a refusal is sent again by a later run: an answer every later call
would meet (401, 402, 403, 429, a redirect or no connection), or an enrichment FullEnrich ended unbilled (out of credits,
rate limited or cancelled). Results are read (reads are not billed) for up to ``wait_seconds``, and a later run reads the
rest. Every command first checks the pin against the journal and refuses on damage, so the out dir
must stay durable and never be edited by hand. Records and the counts-only summary are recomputed from the journal
alone, in private files (0600, folders 0700). The key is held only in the client's request headers.
"""
import http.client
import json
import os
import re
import secrets
import ssl
import time
import unicodedata
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import pairwise

from tools.daily_research import site_screen as ss  # Standard library only, like this module.

RECORD = "blueprint.contact-lookup.v1"
RULE = "blueprint.contact-lookup-rule.v2"
RECIPIENT = "blueprint.contact-recipient.v1"
JOURNAL = "blueprint.contact-lookup.journal.v1"
OWNER_CEILING = "blueprint.contact-lookup.owner-ceiling.v1"
SUMMARY = "blueprint.contact-lookup.summary.v1"
LOOKUP_DECISION = "owner-decision-contact-provider-lookup-20261005"  # The decision the owner's pin names.
PERSON_SEARCH_DECISION = "owner-decision-provider-sourced-person-20261005"  # The only reference that allows a search.
FOLDER = "lookup"  # Inside the site-screen out dir; the site screen never reads it.
PROVIDER, API_HOST, KEY_ENV = "fullenrich", "app.fullenrich.com", "FULLENRICH_API_KEY"
SEARCH_PATH, ENRICH_PATH = "/api/v2/people/search", "/api/v2/contact/enrich/bulk"
VALID_STATUSES = frozenset({"DELIVERABLE"})
STATUSES = frozenset({"DELIVERABLE", "HIGH_PROBABILITY", "CATCH_ALL", "INVALID"})  # FullEnrich email statuses.
# The deciding roles a people search asks for. A kept title holds every word of one, and no word of NOT_DECIDING.
TITLES = ("owner", "president", "general manager", "plant manager", "operations manager", "operations director",
          "engineering manager", "automation manager")
NOT_DECIDING = frozenset({"assistant", "associate", "intern", "coordinator", "vice", "product", "sales", "marketing",
                          "account", "accounts", "finance", "financial", "hr", "human", "talent", "recruiting",
                          "recruiter", "legal", "customer", "former", "retired"})
TITLE_FILLER = frozenset({"of", "the", "and", "for", "at", "senior", "sr"})
SEARCH_LIMIT = 5  # People per search; FullEnrich bills 0.25 credit per person returned.
MOST_CREDITS = {"search": Decimal("0.25") * SEARCH_LIMIT, "enrich": Decimal(1)}  # A found work email is 1 credit.
MAX_CREDITS, MAX_CALLS = Decimal(10000), 5000  # Typo guards; the owner's pin is the control.
WAIT_SECONDS, MAX_WAIT_SECONDS, POLL_SECONDS = 120, 1800, 10
REQUEST_TIMEOUT_SECONDS, MAX_RESPONSE_BYTES = 60, 4 * 1024 * 1024
# Answers every later call would meet: they stop the command, and nothing was billed.
STOPPING = {401: "contact_lookup_provider_auth_refused", 402: "contact_lookup_provider_credits_exhausted",
            403: "contact_lookup_provider_access_refused", 429: "contact_lookup_provider_rate_limited"}
ENDED = frozenset({"FINISHED", "CANCELED", "CREDITS_INSUFFICIENT", "RATE_LIMIT", "UNKNOWN"})  # Enrichment statuses.
ENRICHMENT_ID = re.compile(r"[A-Za-z0-9-]{8,64}")
HONORIFICS = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof"})
SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "phd", "md", "pe", "cpa", "mba", "esq"})
CHOICES = ("published_person_email", "looked_up_person_email", "published_team_inbox", "published_general_inbox", "none")


class LookupFailure(ss.ScreenError):
    """A stable contact_lookup_* code; never upstream text, a key, a name or an address."""


class Stop(LookupFailure):
    """Ends the command before its next call: the pin's ceiling or call limit is reached."""


class Refused(Stop):
    """The provider refused, or was never reached. Nothing was billed, so a later run may send the call again."""


class Unknown(Stop):
    """The call may have been billed without a usable answer. It is never sent again."""


def _text(value, limit=300):
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def amount(value):
    """A Decimal as compact text: 2 for 2.00, 0.25 for 0.250."""
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _credits(value):
    """A finite, non-negative credit amount from a provider number or our own text, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    return number if number.is_finite() and number >= 0 else None


# --- provider client ----------------------------------------------------------------------------
def https(method, path, *, headers, body=None, timeout=REQUEST_TIMEOUT_SECONDS):
    """One HTTPS request to FullEnrich; no redirect, retry or proxy. Returns (status, bytes)."""
    connection = http.client.HTTPSConnection(API_HOST, 443, timeout=timeout, context=ssl.create_default_context())
    try:
        try:
            connection.connect()
        except (OSError, http.client.HTTPException):
            raise ss.TransportError("contact_lookup_provider_unreachable", sent=False) from None
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, http.client.HTTPException):
            raise ss.TransportError("contact_lookup_provider_connection_lost", sent=True) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ss.TransportError("contact_lookup_provider_response_too_large", sent=True)
        return response.status, raw
    finally:
        connection.close()


class FullEnrichClient:
    """FullEnrich requests with one key, which only this client's request headers hold."""

    def __init__(self, api_key, *, transport=https, timeout=REQUEST_TIMEOUT_SECONDS):
        if not isinstance(api_key, str) or not ss.API_KEY.fullmatch(api_key):
            raise LookupFailure("contact_lookup_api_key_invalid")
        self._headers = {"Authorization": "Bearer " + api_key, "Accept": "application/json",
                         "User-Agent": "BlueprintContactLookup/1"}
        self._transport, self._timeout = transport, timeout

    def __repr__(self):
        return "FullEnrichClient(<key withheld>)"

    def _send(self, method, path, body=None):
        headers = dict(self._headers, **({"Content-Type": "application/json"} if body is not None else {}))
        return self._transport(method, path, headers=headers, body=body, timeout=self._timeout)

    def post(self, path, body):
        """One billable request: (status, bytes) for a 2xx answer or a 4xx that refuses this request only. Refused for an
        answer every later call would meet; Unknown when the request may have been billed without an answer."""
        try:
            status, raw = self._send("POST", path, ss.canonical(body).encode())
        except ss.TransportError as error:
            if error.sent:
                raise Unknown("contact_lookup_outcome_unknown") from None
            raise Refused("contact_lookup_provider_unreachable") from None
        if status in STOPPING:
            raise Refused(STOPPING[status])
        if 300 <= status < 400:  # Never followed: the key would travel with the redirect.
            raise Refused("contact_lookup_provider_redirect_refused")
        if 200 <= status < 300 or 400 <= status < 500:
            return status, raw
        raise Unknown("contact_lookup_outcome_unknown")

    def result(self, enrichment_id):
        """One enrichment's result as (status, bytes). Reads are not billed; a failed read is a LookupFailure code."""
        if not isinstance(enrichment_id, str) or not ENRICHMENT_ID.fullmatch(enrichment_id):
            raise LookupFailure("contact_lookup_enrichment_id_invalid")
        try:
            status, raw = self._send("GET", f"{ENRICH_PATH}/{enrichment_id}")
        except ss.TransportError:
            raise LookupFailure("contact_lookup_result_unavailable") from None
        if status == 401:
            raise Refused("contact_lookup_provider_auth_refused")
        return status, raw

    def balance(self):
        """One unbilled credit-balance GET. Return only the finite balance; never upstream text."""
        try:
            status, raw = self._send("GET", "/api/v2/account/credits")
        except ss.TransportError:
            raise LookupFailure("contact_lookup_balance_unavailable") from None
        if status in STOPPING:
            raise Refused(STOPPING[status])
        if status != 200:
            raise LookupFailure("contact_lookup_balance_unavailable")
        value = ss._json(raw)
        balance = _credits(value.get("balance")) if isinstance(value, dict) else None
        if balance is None:
            raise LookupFailure("contact_lookup_balance_invalid")
        return {"command": "balance", "state": "complete", "provider": PROVIDER,
                "credits_available": amount(balance), "checked_at": ss._now()}


# --- people, titles and addresses ----------------------------------------------------------------
def name_parts(name):
    """(first, last) of a full name, honorifics and suffixes aside, or None."""
    tokens = [token.strip(".,") for token in _text(name).split()]
    tokens = [token for token in tokens if ss.words(token)]
    while tokens and ss.words(tokens[0]) in HONORIFICS:
        tokens.pop(0)
    while tokens and ss.words(tokens[-1]) in SUFFIXES:
        tokens.pop()
    return (tokens[0], tokens[-1]) if len(tokens) >= 2 else None


def same_person(first, second):
    """True when both names have the same first and last name, as whole words."""
    one, other = name_parts(first), name_parts(second)
    return bool(one and other) and all(ss.words(a) == ss.words(b) for a, b in zip(one, other))


def title_words(title):
    return [word for word in ss.words(title).split() if word not in TITLE_FILLER]


def listed_title(title):
    """True for a title holding every word of one of TITLES and no word of a role that does not decide."""
    have = set(ss.words(title).split())
    return not have & NOT_DECIDING and any(set(title_words(listed)) <= have for listed in TITLES)


def holds_title(text, title):
    wanted = title_words(title)
    return bool(wanted) and set(wanted) <= set(ss.words(text).split())


def on_domain(domain, operator_domains):
    """True for an operator domain or a subdomain of one."""
    domain = _text(domain).lower().rstrip(".")
    return bool(domain) and isinstance(operator_domains, (list, tuple)) and any(
        isinstance(item, str) and item and (domain == item or domain.endswith("." + item)) for item in operator_domains)


def verification(status):
    """FullEnrich's own email status as valid or not_valid; only valid is ever usable."""
    return "valid" if isinstance(status, str) and status in VALID_STATUSES else "not_valid"


def accept_address(address, status, operator_domains, name):
    """(address, None) when a provider-found work email may be used, else (None, code): FullEnrich marked it
    DELIVERABLE, it is one plain address, never free mail, on the operator's domain or a subdomain, and its local part
    names the person (site_screen.address_role), so never a team, general, careers or other role inbox."""
    if verification(status) != "valid":
        return None, "contact_lookup_status_not_valid"
    address = ss.email_address(address) if isinstance(address, str) else None
    if address is None:
        return None, "contact_lookup_address_invalid"
    domain = address.rpartition("@")[2]
    if domain in ss.FREE_MAIL or ss.site_domain("https://" + domain) in ss.FREE_MAIL:
        return None, "contact_lookup_free_mail"
    if not on_domain(domain, operator_domains):
        return None, "contact_lookup_off_operator_domain"
    if ss.address_role(address, {"verified": False}) in ("team", "general", "refused"):
        return None, "contact_lookup_role_inbox"
    role = ss.address_role(address, {"verified": True, "name": name})
    if role != "person":
        return None, "contact_lookup_address_not_personal"
    return address, None


def profile_check(profile, name, operator_domains):
    """None when the enrichment's own profile, where it gives one, names this person and places them at the operator
    now (no end date, not marked past, company on the operator's domain); else the code."""
    if not isinstance(profile, dict):
        return None
    full = _text(profile.get("full_name")) or " ".join(
        part for part in (_text(profile.get("first_name")), _text(profile.get("last_name"))) if part)
    if full and not same_person(full, name):
        return "contact_lookup_person_mismatch"
    employment = profile.get("employment") if isinstance(profile.get("employment"), dict) else {}
    current = employment.get("current")
    if isinstance(current, dict):
        company = current.get("company") if isinstance(current.get("company"), dict) else {}
        if (company.get("domain") and not on_domain(company.get("domain"), operator_domains)
                or current.get("is_current") is not None and current.get("is_current") is not True
                or current.get("end_at")):
            return "contact_lookup_person_not_current_at_operator"
        if not company.get("domain") and isinstance(employment.get("all"), list) and employment["all"]:
            job, _ = current_job(profile, operator_domains)
            if job is None:
                return "contact_lookup_person_not_current_at_operator"
    elif isinstance(employment.get("all"), list) and employment["all"]:
        job, _ = current_job(profile, operator_domains)
        if job is None:
            return "contact_lookup_person_not_current_at_operator"
    return None


def current_job(person, operator_domains):
    """The job a searched person holds at the operator now, and the field that says so: is_current true, or no end
    date where is_current is absent. A past job never counts, although the company filter can match a past employer."""
    employment = person.get("employment") if isinstance(person.get("employment"), dict) else {}
    every = employment.get("all") if isinstance(employment.get("all"), list) else []
    for where, job in [("employment.current", employment.get("current")), *(("employment.all", job) for job in every)]:
        if (not isinstance(job, dict) or job.get("end_at")
                or job.get("is_current") is not None and job.get("is_current") is not True
                or where == "employment.all" and job.get("is_current") is not True):
            continue
        company = job.get("company") if isinstance(job.get("company"), dict) else {}
        if on_domain(company.get("domain"), operator_domains):
            return job, where + (".is_current" if job.get("is_current") is True else ".end_at_absent")
    return None, None


def candidate(person, operator_domains):
    """A searched person kept as provider_sourced, with the employment field relied on, or (None, code)."""
    if not isinstance(person, dict):
        return None, "contact_lookup_person_invalid"
    name = _text(person.get("full_name")) or " ".join(
        part for part in (_text(person.get("first_name")), _text(person.get("last_name"))) if part)
    if name_parts(name) is None:
        return None, "contact_lookup_person_name_incomplete"
    job, field = current_job(person, operator_domains)
    if job is None:
        return None, "contact_lookup_person_not_current_at_operator"
    title = _text(job.get("title"))
    if not listed_title(title):
        return None, "contact_lookup_person_role_not_listed"
    place = person.get("location") if isinstance(person.get("location"), dict) else {}
    location = ", ".join(part for part in (_text(place.get(key)) for key in ("city", "region", "country")) if part)
    return {"name": name, "title": title, "location": location or None,
            "location_fields": {key: _text(place.get(key)) for key in ("city", "region", "country")},
            "current_employment": {"field": field, "company_domain": _text(job["company"].get("domain")).lower(),
                                   "start_at": _text(job.get("start_at")) or None}}, None


# --- the owner's pin and the spend journal -------------------------------------------------------
EVENTS = {"pinned": ("pin",), "intent": ("call", "attempt", "kind", "site_key", "most_credits"),
          "created": ("call", "attempt", "enrichment_id"), "answered": ("call", "attempt", "credits", "observation"),
          "refused": ("call", "attempt", "code"), "uncertain": ("call", "attempt", "code")}
PIN_FIELDS = frozenset({"schema_version", "owner_reference", "max_credits", "max_calls", "created_at"})
# The states each answer may follow: an enrichment that ended unbilled is refused after it was created.
FOLLOWS = {"created": ("unknown",), "answered": ("unknown", "created"), "refused": ("unknown", "created"),
           "uncertain": ("unknown",)}


def parse_reference(value):
    try:
        return ss.parse_reference(value)
    except ss.ScreenError:
        raise LookupFailure("contact_lookup_owner_reference_invalid") from None


def parse_credits(value):
    credits = _credits(value)
    if credits is None or not Decimal(0) < credits <= MAX_CREDITS:
        raise LookupFailure("contact_lookup_credits_invalid")
    return credits


def parse_calls(value):
    if type(value) is not int or not 1 <= value <= MAX_CALLS:
        raise LookupFailure("contact_lookup_calls_invalid")
    return value


def _pin_valid(pin):
    try:
        return (isinstance(pin, dict) and set(pin) == PIN_FIELDS and pin["schema_version"] == OWNER_CEILING
                and parse_reference(pin["owner_reference"]) == pin["owner_reference"]
                and amount(parse_credits(pin["max_credits"])) == pin["max_credits"]
                and parse_calls(pin["max_calls"]) == pin["max_calls"] and isinstance(pin["created_at"], str))
    except LookupFailure:
        return False


def limits(pin, owner_reference, max_credits, max_calls):
    """This run's owner reference, credit ceiling and call limit. Against a pin they may only be lower."""
    reference, credits, calls = parse_reference(owner_reference), parse_credits(max_credits), parse_calls(max_calls)
    if pin is not None:
        if reference != pin["owner_reference"]:
            raise LookupFailure("contact_lookup_owner_reference_mismatch")
        if credits > Decimal(pin["max_credits"]):
            raise LookupFailure("contact_lookup_credits_above_pin")
        if calls > pin["max_calls"]:
            raise LookupFailure("contact_lookup_calls_above_pin")
    return {"owner_reference": reference, "max_credits": credits, "max_calls": calls}


def _event(line):
    """One journal line's event, or None when it is not one. ss.Ledger seals a torn line with its own schema."""
    event = ss._json(line)
    kind = event.get("event") if isinstance(event, dict) else None
    if kind == "sealed":
        valid = (event.get("schema_version") == ss.LEDGER and type(event.get("line")) is int
                 and isinstance(event.get("sha256"), str) and ss.SHA.fullmatch(event["sha256"]))
        return event if valid else None
    if kind not in EVENTS or event.get("schema_version") != JOURNAL or not all(name in event for name in EVENTS[kind]):
        return None
    if kind == "pinned":
        return event if _pin_valid(event["pin"]) else None
    valid = (isinstance(event["call"], str) and ss.SHA.fullmatch(event["call"]) and isinstance(event["attempt"], str)
             and ss.ATTEMPT.fullmatch(event["attempt"]))
    if kind == "intent":
        valid = (valid and event["kind"] in MOST_CREDITS and isinstance(event["site_key"], str)
                 and ss.SHA.fullmatch(event["site_key"]) and _credits(event["most_credits"]) is not None)
    elif kind == "created":
        valid = valid and isinstance(event["enrichment_id"], str) and ENRICHMENT_ID.fullmatch(event["enrichment_id"])
    elif kind == "answered":
        valid = valid and _credits(event["credits"]) is not None and isinstance(event["observation"], dict)
    else:
        valid = valid and isinstance(event["code"], str) and ss.CODE.fullmatch(event["code"])
    return event if valid else None


class Journal(ss.Ledger):
    """The lookup spend journal, ``lookup/spend.jsonl``: ss.Ledger's appends (one write and an fsync per line, a torn
    tail sealed first), holding the pin and every call's events."""

    def __init__(self, path):
        super().__init__(path, None)

    def events(self):
        """Every event in order. A torn final line is skipped; any other damaged line refuses unless the next seals it."""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return []
        lines = raw.split(b"\n")[:-1]
        parsed = [_event(line) for line in lines]
        events = []
        for number, (line, event) in enumerate(zip(lines, parsed), 1):
            seal = parsed[number] if number < len(parsed) else None
            if event is None and not (seal and seal["event"] == "sealed" and seal["line"] == number
                                      and seal["sha256"] == ss._sha256(line)):
                raise LookupFailure("contact_lookup_journal_invalid")
            if event is not None and event["event"] != "sealed":
                events.append(event)
        return events


def fold(events):
    """Each call's state: unknown (an intent with no answer, or an uncertain answer), created (an enrichment exists and
    its result is unread), answered, or refused (nothing was billed; a later run may send it again)."""
    calls = {}
    for event in events:
        kind, key = event["event"], event["call"]
        call = calls.get(key)
        if kind == "intent":
            if call is not None and call["state"] != "refused":
                raise LookupFailure("contact_lookup_journal_invalid")
            calls[key] = {"state": "unknown", "attempt": event["attempt"], "kind": event["kind"],
                          "site_key": event["site_key"], "most_credits": event["most_credits"], "closed": False}
            continue
        if (call is None or call["attempt"] != event["attempt"] or call["closed"] or call["state"] not in FOLLOWS[kind]
                or kind == "created" and call["kind"] != "enrich"):
            raise LookupFailure("contact_lookup_journal_invalid")
        if kind == "created":
            call.update(state="created", enrichment_id=event["enrichment_id"])
        elif kind == "answered":
            call.update(state="answered", closed=True, credits=event["credits"], observation=event["observation"])
        elif kind == "refused":
            call.update(state="refused", closed=True, code=event["code"])
        else:
            call["closed"] = True
    return calls


def committed(calls):
    """(credits, calls) that may be billed: answered calls at their reported cost, and calls whose outcome or result is
    still unknown at their most. A refused call was never billed."""
    credits, count = Decimal(0), 0
    for call in calls.values():
        if call["state"] == "answered":
            credits, count = credits + Decimal(call["credits"]), count + 1
        elif call["state"] in ("unknown", "created"):
            credits, count = credits + Decimal(call["most_credits"]), count + 1
    return credits, count


class Book:
    """A site-screen out dir's lookup folder: the owner's pin and the spend journal, checked against each other. A
    deleted journal, an edited pin or an answer without its intent refuses, so spend is never reset by damage to one
    file; matching edits to both, or the loss of the whole folder, are not caught."""

    def __init__(self, workspace, *, readonly=False):
        self.root = workspace.root / FOLDER
        self.pin_path, self.journal = self.root / "owner_ceiling.json", Journal(self.root / "spend.jsonl")
        events, self.pin = self.journal.events(), self._pin()
        self.calls = {}
        if not events:
            if self.pin is not None:
                raise LookupFailure("contact_lookup_journal_missing")
            return
        head, body = events[0], events[1:]
        if head["event"] != "pinned" or any(event["event"] == "pinned" for event in body):
            raise LookupFailure("contact_lookup_journal_invalid")
        if self.pin is None:
            if body:
                raise LookupFailure("contact_lookup_owner_ceiling_missing")
            if not readonly:
                ss._write_once(self.pin_path, (ss.canonical(head["pin"]) + "\n").encode())  # The pin's own crash window.
            self.pin = head["pin"]
        if self.pin != head["pin"]:
            raise LookupFailure("contact_lookup_owner_ceiling_mismatch")
        self.calls = fold(body)

    def _pin(self):
        try:
            pin = ss._json(self.pin_path.read_bytes())
        except FileNotFoundError:
            return None
        if not _pin_valid(pin):
            raise LookupFailure("contact_lookup_owner_ceiling_invalid")
        return pin

    def append(self, **event):
        self.journal.append({"schema_version": JOURNAL, **event, "recorded_at": ss._now()})

    def create_pin(self, bounds):
        """Pin the owner reference, credit ceiling and call limit: in the journal first, then owner_ceiling.json."""
        self.root.mkdir(mode=0o700, exist_ok=True)
        pin = {"schema_version": OWNER_CEILING, "owner_reference": bounds["owner_reference"],
               "max_credits": amount(bounds["max_credits"]), "max_calls": bounds["max_calls"], "created_at": ss._now()}
        self.append(event="pinned", pin=pin)
        ss._write_once(self.pin_path, (ss.canonical(pin) + "\n").encode())
        self.pin = pin


class Calls:
    """One command's provider calls. ``apply`` admits each against the pin, journals its intent and sends it; ``dry``
    admits and counts it only; ``replay`` sends nothing. A call with any journal state but refused is never sent again."""

    def __init__(self, book, mode, *, client=None, bounds=None, code=None):
        self.book, self.mode, self.client, self.bounds, self.code = book, mode, client, bounds, code
        self.counts = Counter({"made": 0, "would_make": 0, "known": 0})

    @staticmethod
    def key(kind, identity):
        return ss._sha256(ss.canonical([PROVIDER, kind, identity]).encode())

    def __call__(self, kind, identity, site_key, body, seal):
        """(key, the call's state) once it is known or sent, or (key, None) when it is not sent here."""
        key = self.key(kind, identity)
        call = self.book.calls.get(key)
        if call is not None and call["state"] != "refused":
            self.counts["known"] += 1
            return key, call
        if self.mode == "replay":
            return key, None
        most = MOST_CREDITS[kind]
        credits, count = committed(self.book.calls)
        if count + 1 > self.bounds["max_calls"]:
            raise Stop("contact_lookup_max_calls_reached")
        if credits + most > self.bounds["max_credits"]:
            raise Stop("contact_lookup_credit_ceiling_reached")
        state = {"state": "unknown", "kind": kind, "site_key": site_key, "most_credits": amount(most), "closed": False}
        if self.mode == "dry":
            self.book.calls[key] = state  # What this call would commit.
            self.counts["would_make"] += 1
            return key, None
        attempt = secrets.token_hex(8)
        self.book.append(event="intent", call=key, attempt=attempt, kind=kind, site_key=site_key,
                         most_credits=amount(most), owner_reference=self.bounds["owner_reference"], code=self.code)
        call = self.book.calls[key] = {**state, "attempt": attempt}
        try:
            status, raw = self.client.post(SEARCH_PATH if kind == "search" else ENRICH_PATH, body(key))
        except Refused as error:
            self.book.append(event="refused", call=key, attempt=attempt, code=str(error))
            call.update(state="refused", closed=True, code=str(error))
            raise
        except Unknown as error:
            self.book.append(event="uncertain", call=key, attempt=attempt, code=str(error))
            call["closed"] = True
            raise
        self.counts["made"] += 1
        try:
            fields = seal(status, raw)
        except (AttributeError, KeyError, TypeError, ValueError, InvalidOperation):
            self.book.append(event="uncertain", call=key, attempt=attempt, code="contact_lookup_response_invalid")
            call["closed"] = True
            raise Unknown("contact_lookup_response_invalid") from None
        self.book.append(call=key, attempt=attempt, **fields)
        if fields["event"] == "created":
            call.update(state="created", enrichment_id=fields["enrichment_id"])
        else:
            call.update(state="answered", closed=True, credits=fields["credits"], observation=fields["observation"])
        return key, call


def _rejected():
    """A 4xx answer that refuses this request only: not billed, kept, and never sent again."""
    return {"event": "answered", "credits": "0", "observation": {
        "outcome": "rejected", "status": None, "address": None, "reason": "contact_lookup_request_rejected",
        "checked_at": ss._now()}}


def seal_search(site):
    """Keep minimal current-role candidates so a shared company search can be qualified for each target site.
    Never retain rejected people's names, profiles, addresses or phone numbers."""
    def seal(status, raw):
        if status >= 400:
            return _rejected()
        value = json.loads(raw)
        people = value["people"]
        if not isinstance(people, list):
            raise TypeError("people")
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        credits = _credits(metadata.get("credits"))
        candidates, passed = [], Counter()
        for person in people:
            found, code = candidate(person, site["operator_domains"])
            if found is None:
                passed[code] += 1
            elif len(candidates) < SEARCH_LIMIT:
                candidates.append(found)
        return {"event": "answered", "credits": amount(Decimal("0.25") * len(people) if credits is None else credits),
                "observation": {"outcome": "searched", "people": len(people),
                                "candidate": candidates[0] if candidates else None, "candidates": candidates,
                                "passed_over": dict(passed), "checked_at": ss._now()}}
    return seal


def seal_start(status, raw):
    """An enrichment start: its id, or a rejection."""
    if status >= 400:
        return _rejected()
    enrichment = json.loads(raw)["enrichment_id"]
    if not isinstance(enrichment, str) or not ENRICHMENT_ID.fullmatch(enrichment):
        raise ValueError("enrichment_id")
    return {"event": "created", "enrichment_id": enrichment}


def seal_result(status, raw, key, call, person, site):
    """The journal event of an ended enrichment, or None while it is not ready. Only a usable address is kept."""
    if status == 400:  # FullEnrich: "Enrichment not ready, try again in 30 seconds".
        return None
    if status != 200:
        raise LookupFailure("contact_lookup_result_unavailable")
    value = ss._json(raw)
    if not isinstance(value, dict) or value.get("id") != call["enrichment_id"]:
        raise LookupFailure("contact_lookup_result_invalid")
    state = value.get("status")
    if state in ("CREATED", "IN_PROGRESS"):
        return None
    if state not in ENDED:
        raise LookupFailure("contact_lookup_result_invalid")
    if state == "UNKNOWN":
        return None  # The existing enrichment may still cost its maximum. Only GET may be tried again.
    cost = value.get("cost") if isinstance(value.get("cost"), dict) else {}
    credits = _credits(cost.get("credits"))
    observation = {"outcome": state.lower(), "status": None, "address": None, "checked_at": ss._now()}
    if state != "FINISHED":
        if credits == Decimal(0):  # An explicit zero cost proves this ended unbilled.
            return {"event": "refused", "code": "contact_lookup_enrichment_" + state.lower()}
        if credits is None:
            raise LookupFailure("contact_lookup_result_invalid")
        return {"event": "answered", "credits": amount(credits),
                "observation": {**observation, "reason": "contact_lookup_enrichment_" + state.lower()}}
    data = value.get("data")
    if not isinstance(data, list) or len(data) > 1 or data and not isinstance(data[0], dict):
        raise LookupFailure("contact_lookup_result_invalid")  # One contact was sent.
    item = data[0] if data else {}
    custom = item.get("custom") if isinstance(item.get("custom"), dict) else {}
    if custom.get("call") not in (None, key):  # The id binds the result to this call; an echo must agree.
        raise LookupFailure("contact_lookup_result_invalid")
    info = item.get("contact_info") if isinstance(item.get("contact_info"), dict) else {}
    best = info.get("most_probable_work_email")
    if not isinstance(best, dict):
        listed = info.get("work_emails") if isinstance(info.get("work_emails"), list) else []
        best = next((entry for entry in listed if isinstance(entry, dict)), {})
    email, found = best.get("email"), best.get("status") if isinstance(best.get("status"), str) else None
    if credits is None:
        credits = Decimal(1) if email else Decimal(0)
    reason = profile_check(item.get("profile"), person["name"], site["operator_domains"])
    if not email:
        address, reason = None, reason or "contact_lookup_email_not_found"
    elif reason:
        address = None
    else:
        address, reason = accept_address(email, found, site["operator_domains"], person["name"])
    return {"event": "answered", "credits": amount(credits),
            "observation": {**observation, "status": found, "address": address, "reason": reason}}


# --- sites and people -----------------------------------------------------------------------------
def utc_today():
    return date.fromisoformat(ss._now()[:10])


def lookup_site(workspace, contact, screen, *, today=None):
    """What a lookup needs from one site's contact and screen records. The operator's domains are the contact
    record's own when its rule gives them, else the screen's (the domain whose page proved the operator)."""
    domains = contact["operator_domains"] if isinstance(contact.get("operator_domains"), list) else screen["operator_domains"]
    domains = [item.lower() for item in domains if isinstance(item, str) and item and item.lower() not in ss.FREE_MAIL]
    today = today or utc_today()
    content, _ = ss.output_of(ss._json(workspace.path("contact", "results", contact["site_key"]).read_bytes()))
    kept = []
    for stage in ("screen", "contact"):  # Our own earlier reads; LinkedIn was never read.
        path = workspace.path(stage, "evidence", contact["site_key"])
        evidence = ss._json(path.read_bytes()) if path.exists() else None
        pages = evidence.get("pages") if isinstance(evidence, dict) and isinstance(evidence.get("pages"), dict) else {}
        kept += [(url, page["text"], page.get("sha256")) for url, page in sorted(pages.items()) if isinstance(page, dict)
                 and page.get("state") == "ok" and isinstance(page.get("text"), str) and not ss.never_fetch(url)]
    address = screen.get("address") or {}
    found = ss.found_address({"address": address}, screen["answers"], screen["verification"])
    address = {**address, **(found or {})}
    return {"site_key": contact["site_key"], "contact": contact,
            "address": address, "task_input": ss.contact_input(screen),
            "person": contact["person"] if isinstance(contact.get("person"), dict) else {},
            "published_person_email": choose_recipient(contact, today=today)["choice"] == "published_person_email",
            "operator": _text(screen["answers"].get("operator_identity")) or _text(screen["input"].get("operator")),
            "operator_domains": domains, "operator_domain": domains[0] if domains else None, "kept_pages": kept,
            "person_quote": _text(content.get("person_quote"), 1200), "today": today}


def lookup_sites(workspace, states, *, today=None):
    """One lookup input per contact record with a screen record, in contact order, and how many could not be read."""
    contacts = ss.stage_records(workspace, states, "contact")
    screens = ss.stage_records(workspace, states, "screen")
    sites, unreadable = [], 0
    for key in states["contact"]:
        if key in contacts and key in screens:
            try:
                sites.append(lookup_site(workspace, contacts[key], screens[key], today=today))
            except (AttributeError, KeyError, TypeError, ValueError):
                unreadable += 1  # A record this rule cannot read is never looked up.
    return sites, unreadable


def quoted_person(site):
    """The contact stage's named person when a quote proves them and their source is current, else (None, code)."""
    person = site["person"]
    if (person.get("verified") is not True or person.get("level") not in ss.PROVEN
            or name_parts(person.get("name")) is None):
        return None, "no_verified_person"
    if person.get("current") is not True or not ss._fresh(person.get("date"), site.get("today") or utc_today()):
        return None, "person_not_current"
    if not holds_title(site.get("person_quote", ""), person.get("title")):
        return None, "person_role_unproven"
    found = {"name": person["name"], "title": person.get("title"), "location": None, "sourcing": "quoted_person",
            "proof": {"source": "site_contact", "url": person.get("url"), "level": person["level"],
                      "date": person.get("date"), "current": True}, "corroboration": None}
    found["site_responsibility"] = site_responsibility(site, found)
    if found["site_responsibility"]["route"] == "hold":
        return None, "target_site_location_mismatch"
    return found, None


def corroborate(site, person):
    """Whether a page the site screen already read and kept for this site (our own read, never LinkedIn) names the
    operator, and this person with their title in one sentence: that sentence is the quote, with the page text's
    SHA-256. No new read."""
    name = ss.words(person["name"])
    for url, text, text_sha256 in site["kept_pages"]:
        if not ss.names_operator(text, site["operator"]):
            continue
        for sentence in ss.sentences(text):
            if ss.has_phrase(name, ss.words(sentence)) and holds_title(sentence, person["title"]):
                return {"corroborated": True, "url": url, "quote": _text(sentence, 1200), "level": "verified_on_page",
                        "text_sha256": text_sha256 or ss._sha256(text.encode())}
    return {"corroborated": False, "url": None, "quote": None, "level": None, "text_sha256": None}


def search_body(site):
    return {"current_company_domains": [{"value": site["operator_domain"], "exact_match": True}],
            "current_position_titles": [{"value": title} for title in TITLES], "limit": SEARCH_LIMIT, "offset": 0}


def role_scope_suffix(text, known_names, *, allow_news=True):
    """An address's direct entity/facility qualifier must belong to this retained operator or site."""
    remaining = ss.words(text)
    literals = sorted(known_names | {"united states of america", "united states", "usa", "us"}, key=len, reverse=True)
    while remaining:
        literal = next((name for name in literals if name and
                        (remaining == name or remaining.startswith(name + " "))), None)
        if literal:
            remaining = remaining[len(literal):].strip()
            continue
        postal = re.match(r"^\d{5}(?:\s+\d{4})?(?:\s|$)", remaining)
        if postal:
            remaining = remaining[postal.end():].strip()
            continue
        word, _, rest = remaining.partition(" ")
        if word in {"the", "for", "of", "at", "s", "plant", "facility", "site", "factory"}:
            remaining = rest
            continue
        # A separate news predicate does not erase the preceding explicit role complement.
        return allow_news and word in {"said", "says", "announced", "visited", "visits", "discussed"}
    return True


UNIT = re.compile(r"\b(suite|ste|unit|building|bldg|floor|fl)\s+([a-z0-9-]+)\b", re.IGNORECASE)
UNIT_KINDS = {"ste": "suite", "bldg": "building", "fl": "floor"}


def hash_units(text, protected=()):
    text = unicodedata.normalize("NFKC", text)
    spans = [match.span() for name in protected if ss.words(name)
             for match in re.finditer(r"\b" + r"[^a-z0-9]+".join(re.escape(word) for word in ss.words(name).split())
                                     + r"\b", text, re.IGNORECASE)]
    return re.sub(r"#\s*([a-z0-9-]+)\b", lambda match: match.group() if any(
        match.start() < end and match.end() > start for start, end in spans) else "suite " + match.group(1),
        text, flags=re.IGNORECASE)


def address_units(text, protected=()):
    """Unit identity is preserved while placement and standard designator spelling are normalized."""
    spans = [match.span() for name in protected if name
             for match in re.finditer(r"\b" + re.escape(name) + r"\b", text)]
    units = set()

    def strip(match):
        if any(start <= match.start() < end for start, end in spans):
            return match.group()
        kind, value = match.groups()
        kind = UNIT_KINDS.get(kind.lower(), kind.lower())
        units.add((kind, value.lower().replace("-", "")))
        return " "
    return UNIT.sub(strip, text), units


def role_complement(person, sentence):
    normalized = ss.words(sentence)
    name = ss.words(person["name"])
    start = re.search(r"\b" + re.escape(name) + r"\s+(?:(?:is|the|a|serves|as|our|current|now|currently|new|recently)\s+){0,5}", normalized)
    if start:
        description = normalized[start.end():]
        for preposition in re.finditer(r"\b(?:at|for|of)\s+", description):
            role = description[:preposition.start()]
            if (holds_title(role, person["title"])
                    and set(role.split()) <= set(title_words(person["title"])) | TITLE_FILLER):
                offset = len(normalized[:start.end()].split()) + len(description[:preposition.end()].split())
                return description[preposition.end():], offset, preposition.group().strip()
    return "", 0, None


def quoted_role_address(person, sentence):
    tail, offset, preposition = role_complement(person, sentence)
    if preposition is None or set(tail.split()) & {"whose", "which", "headquarters", "hq"}:
        return {}
    raw = unicodedata.normalize("NFKC", sentence)
    tokens = list(re.finditer(r"[^\W_]+", raw))
    if offset >= len(tokens) or ss.words(" ".join(token.group() for token in tokens)) != ss.words(raw):
        return {}
    complement = raw[tokens[offset].start():]
    for boundary in re.finditer(r"\b(?:and|but|while|whereas|for)\b", complement, re.IGNORECASE):
        prefix = complement[:boundary.start()].rstrip(" .;")
        location = ss.parse_location(prefix)
        if (location.get("city") and (location.get("state") or ss.street_anchor(location.get("street")))
                or ss.street_anchor(prefix)):
            complement = prefix
            break
    address = ss.parse_location(complement.rstrip(" .;"))
    city, state = address.get("city"), address.get("state")
    if not state and (ss.street_anchor(city) or UNIT.fullmatch(city or "")):
        return {"street": complement.rstrip(" .;")}
    if (not city or not re.search(ss.city_pattern(city), complement)
            or not (state or ss.street_anchor(address.get("street")))):
        return {}
    return address


def employment_sentence(sentence):
    """Remove temporal/editorial preambles before identifying the employment subject."""
    return re.sub(r"^\s*(?:(?:in|on|as\s+of|during|by)\b[^,;:.!?]{1,80},"
                  r"|(?:update|note|correction|announcement|news)\s*:)\s*", "",
                  unicodedata.normalize("NFKC", sentence), flags=re.IGNORECASE)


def employment_contradiction(person, sentence, known_names):
    """Current-role evidence cannot ignore an explicit same-person employment contradiction."""
    name = ss.words(person["name"])
    sentence = employment_sentence(sentence)
    role_sentence = re.sub(r"^\s*(?:he|she|they)\b", person["name"], sentence, flags=re.IGNORECASE)
    role_employer, _, role_preposition = role_complement(person, role_sentence)
    if role_preposition and not quoted_role_address(person, role_sentence):
        employers = {role_employer, re.sub(r"^(?:the|an?)\s+", "", role_employer)}
        if not any(entity and (value == entity or value.startswith(entity + " "))
                   for entity in known_names for value in employers):
            return True
    protected = [match.span() for entity in known_names | {name} if entity
                 for match in re.finditer(r"\b" + r"[\W_]+".join(re.escape(word) for word in entity.split())
                                         + r"\b", sentence, re.IGNORECASE)]
    employment = re.compile(r"\b(?:(?P<employment>works?|worked|(?:is\s+)?employed|(?:is\s+an?\s+)?employee|serves|served)"
                            r"(?:\s+as\s+[^\n.,;:]{1,120}?)?\s+(?:for|by|at|with)\s+"
                            r"|(?P<transition>join(?:s|ed|ing)?\s+|moved\s+(?:on\s+)?to\s+)"
                            r"|(?P<departure>left|leaves|leaving|depart(?:ed|ing)?|resign(?:ed|s|ing)?|quit(?:s|ting)?"
                            r"|retire(?:d|s|ing)?|fired|dismissed|terminated|laid\s+off|step(?:ped|s|ping)?\s+down|no\s+longer)"
                            r"(?=\s|[.,;:]|$)\s*)", re.IGNORECASE)
    cuts = [0] + [match.end() for match in re.finditer(r"\b(?:and|but|while|whereas)\b", sentence, re.IGNORECASE)
                  if not any(start <= match.start() < end for start, end in protected)] + [len(sentence)]
    for start, end in pairwise(cuts):
        clause = sentence[start:end]
        match = next((found for found in employment.finditer(clause)
                      if not any(left <= start + found.start() < right for left, right in protected)), None)
        if not match:
            continue
        subject = clause[:match.start()].strip()
        reported = re.search(r"\b(?:said|stated|reported|confirmed|announced|noted|explained|told|mentioned|recalled)"
                             r"\s+(?:that\s+)?(.+)$", ss.words(subject))
        if reported and not (ss.has_phrase(name, reported.group(1))
                             or re.match(r"^(?:he|she|they|his|her|their|i|my)\b", reported.group(1))):
            continue
        possessive = re.search(r"\b(?:" + re.escape(name) + r"\s+s|his|her|their)\s+(.+)", ss.words(subject))
        if possessive:
            owned = re.sub(r"^(?:(?:the|current|former|previous|first|last)\s+)+", "", possessive.group(1))
            if owned.split()[0] not in {"employment", "job", "role", "position", "tenure", "appointment", "service", "contract"}:
                continue
        # An explicitly named different subject grants no facts about this person. Lowercase modifiers do not
        # introduce a subject; known personal names/pronouns continue the preceding role statement.
        named = {ss.words(word) for word in re.findall(r"\b[^\W\d_]+\b", subject)
                 if word.isupper() or word[:1].isupper() and word[1:].islower()} - {
            "he", "she", "they", "his", "her", "their", "i", "my", "now", "currently", "still", "also", "recently"}
        named -= set(title_words(person["title"])) | TITLE_FILLER
        # A full subject immediately before its verb binds the claim despite a date/editorial preamble.
        if re.search(r"\b" + re.escape(name) + r"(?:\s+(?:is|was|has|had|have|been|an?|now|currently|recently))*$",
                     ss.words(subject)):
            named = set()
        if named and not named <= set(name.split()):
            continue
        if set(ss.words(subject).split()) & {"company", "operator", "team", "workers", "employees"}:
            continue
        employer = ss.words(clause[match.end():])
        if match.group("departure"):
            object_text = re.sub(r"^(?:(?:the|his|her|their|our|from|as|for)\s+)+", "", employer)
            if (any(entity and (object_text == entity or object_text.startswith(entity + " ")) for entity in known_names)
                    or match.group("departure").lower().startswith(("resign", "quit", "retire"))
                    or match.group("departure").lower() in {"fired", "dismissed", "terminated", "laid off"}
                    or match.group("departure").lower().startswith("step")
                    and (not employer or holds_title(employer, person["title"]))
                    or set(object_text.split()) & {"job", "role", "position", "employment", "corporation", "corp", "inc", "llc"}
                    or re.match(r"^(?:company|employer|business|organization|firm)"
                                r"(?:$|\s+(?:in|on|to|for|after|before|during)\b)", object_text)
                    or re.match(r"^(?:to\s+)?(?:join|work|serve|be\s+employed)\b", employer)
                    or employer.startswith("for ") and not set(object_text.split()) & {
                        "conference", "meeting", "lunch", "vacation", "trip", "training", "workshop", "airport", "home"}
                    or match.group("departure").lower() == "no longer"
                    and re.match(r"^(?:works?|worked|serves|employed)\b", employer)):
                return True
            continue
        ended = re.search(r"\b(?:to|until|through)\s+(?:\w+\s+){0,3}(?:19|20)\d{2}\b", employer)
        past = match.group("employment") in {"worked", "served"} or re.search(r"\bwas(?:\s+an?)?$", ss.words(subject))
        employers = {employer, re.sub(r"^(?:the|an?)\s+", "", employer)}
        known_employer = any(entity and (value == entity or value.startswith(entity + " "))
                             for entity in known_names for value in employers)
        if match.group("transition"):
            event = re.search(r"\b(?:meeting|conference|lunch|vacation|trip|training|workshop)\b"
                              r"(?:$|\s+(?:at|in|with|for)\b)", employer)
            employment_like = bool(set(employer.split()) & {
                "corporation", "corp", "inc", "llc", "ltd", "company", "industries", "works", "manufacturing"}
                or holds_title(employer, person["title"]))
            if event or not known_employer and not employment_like:
                continue
        if ended and past:
            if known_employer:
                return True
            continue  # An explicitly ended rival job does not contradict current target employment.
        if not known_employer:
            return True
    return False


def site_role_quote(site, person, sentence):
    """A role tied to this facility, rather than a person's visit or a company-wide name/title match."""
    given_address = site.get("address") or {}
    names = [site.get("operator") or "", site["task_input"].get("site_name") or ""]
    sentence = hash_units(sentence, names)
    street, target_units = address_units(ss.words(hash_units(given_address.get("street") or "")))
    anchors = ss.site_anchors({**site, "address": {**given_address, "street": street}})
    specific = [item for item in anchors if item["kind"] == "street"] or [
        item for item in anchors if item["kind"] == "site_name_city"]
    normalized = ss.words(sentence)
    cased = re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", sentence)).strip()
    if ss.words(cased) != normalized:
        return False
    name = ss.words(person["name"])
    tail, scope_offset, preposition = role_complement(person, sentence)
    role_scope = preposition is not None
    # A conjunction inside a known entity name is not a clause boundary (including '&' / 'and' spellings).
    known_names = {ss.words(name.replace("&", " and ")) for name in names} | {ss.words(name) for name in names}
    known_names |= {name.replace(" and ", " ") for name in known_names}
    if employment_contradiction(person, sentence, known_names):
        return False
    spans = [match.span() for name in known_names if name
             for match in re.finditer(r"\b" + re.escape(name) + r"\b", tail)]
    for boundary in re.finditer(r"\b(?:and|but|while|whereas)\b|\b" + re.escape(ss.words(person["title"])) + r"\b", tail):
        if not any(start <= boundary.start() < end for start, end in spans):
            tail = tail[:boundary.start()]
            break
    cased_scope = " ".join(cased.split()[scope_offset:scope_offset + len(tail.split())])
    tail, quoted_units = address_units(tail, known_names)
    tail = ss.words(tail)
    if target_units and quoted_units != target_units:
        return False
    address = site.get("address") or {}
    city, state = address.get("city"), address.get("state")
    if not city:
        return False  # A street alone cannot distinguish facilities in different cities.
    if state and not re.search(ss.city_pattern(city) + r"\s+" + ss.state_pattern(state), cased_scope):
        return False
    canonical = ss._canon(tail, ss.CITY_WORDS)
    if city and not ss.has_phrase(ss._canon(city, ss.CITY_WORDS), canonical):
        return False
    if city and state and not re.search(r"\b" + re.escape(ss._canon(city, ss.CITY_WORDS)) + r"\s+(?:"
                                       + re.escape(state.lower()) + "|"
                                       + re.escape(ss.STATE_NAMES.get(state, state).lower()) + r")\b", canonical):
        return False
    # Only the direct role complement can name the facility; a company's HQ or another claim in the sentence cannot.
    allowed = {word for name in known_names for word in name.split()}
    allowed.update(ss.words(city or "").split())
    allowed.update({"the", "at", "of", "for", "in", "on", "s", "plant", "facility", "site", "factory"})
    street_tail = ss._canon(tail, ss.STREET_WORDS)
    scoped = []
    for anchor in specific:
        if anchor["kind"] == "street":
            street = ss._canon(anchor["street"], ss.STREET_WORDS)
            if not ss.has_phrase(street, street_tail):
                continue
            prefix, _, _ = street_tail.partition(street)
            if not set(prefix.split()) <= allowed:
                continue
            # Compare the city immediately following this street, not a suffix of another city's name.
            suffix = ss._canon(" ".join(tail.split()[len(prefix.split()) + len(street.split()):]), ss.CITY_WORDS)
            city_pattern = re.escape(ss._canon(city, ss.CITY_WORDS)) if city else ""
            if state:
                city_pattern += r"\s+(?:" + re.escape(state.lower()) + "|" + re.escape(ss.STATE_NAMES.get(state, state).lower()) + ")"
            location_match = re.match(city_pattern + r"\b", suffix)
            if not location_match or not role_scope_suffix(suffix[location_match.end():], known_names):
                continue
        else:
            given_name = site["task_input"].get("site_name") or ""
            exact_names = {ss.words(given_name), ss.words(given_name.replace("&", " and "))}
            exact_names |= {name.replace(" and ", " ") for name in exact_names}
            before_city = canonical.partition(ss._canon(anchor["city"], ss.CITY_WORDS))[0]
            if not (any(name and tail.startswith((name + " ", "the " + name + " "))
                        for name in exact_names)
                    and set(before_city.split()) <= allowed
                    and ss.has_phrase(ss._canon(anchor["city"], ss.CITY_WORDS), canonical)):
                continue
            location_match = re.search(re.escape(ss._canon(city, ss.CITY_WORDS)) + r"\s+(?:"
                                       + re.escape((state or "").lower()) + "|"
                                       + re.escape(ss.STATE_NAMES.get(state, state or "").lower()) + r")\b", canonical) if state else re.search(
                                           re.escape(ss._canon(city, ss.CITY_WORDS)) + r"\b", canonical)
            if not location_match or not role_scope_suffix(canonical[location_match.end():], known_names):
                continue
        scoped.append(anchor)
    assertion = normalized
    for name in sorted(known_names | {ss.words(person["name"])}, key=len, reverse=True):
        if name:
            assertion = re.sub(r"\b" + re.escape(name) + r"\b", " ", assertion)
    return bool(role_scope and ss.names_site(sentence, scoped) and not sentence.rstrip().endswith("?")
                and not set(assertion.split()) & {"former", "formerly", "retired", "not"})


def site_responsibility(site, person):
    """Location is a mismatch signal, never proof of responsibility. Only a retained role/site quote proves scope.
    Unknown scope stays usable as an explicit corporate referral; off-site local managers are held."""
    title = ss.words(person["title"])
    known_names = [site["task_input"].get("site_name") or "", site.get("operator") or ""]
    known_names = {ss.words(name.replace("&", " and ")) for name in known_names} | {
        ss.words(name) for name in known_names}
    known_names |= {name.replace(" and ", " ") for name in known_names}
    pages = [(url, text, text_sha256, ss.sentences(text)) for url, text, text_sha256 in site["kept_pages"]]
    # Read all retained evidence before accepting any role proof; page order cannot erase a departure.
    for _, _, _, sentences in pages:
        linked = False
        for sentence in sentences:
            named = ss.has_phrase(ss.words(person["name"]), ss.words(sentence))
            pronoun = bool(re.match(r"\s*(?:he|she|they|his|her|their)\b", employment_sentence(sentence), re.IGNORECASE))
            if (named or linked and pronoun) and employment_contradiction(person, sentence, known_names):
                return {"site_key": site["site_key"], "status": "unknown", "route": "hold",
                        "reason": "target_site_responsibility_unproven", "proof": None}
            if named:
                subject = ss.words(employment_sentence(sentence))
                person_name = ss.words(person["name"])
                linked = subject == person_name or subject.startswith(person_name + " ")
            elif not pronoun:
                linked = False
    for url, text, text_sha256, sentences in pages:
        if not ss.names_operator(text, site["operator"]):
            continue
        for sentence in sentences:
            if site_role_quote(site, person, sentence):
                return {"site_key": site["site_key"], "status": "verified", "route": "site_contact",
                        "reason": None, "proof": {"url": url, "quote": _text(sentence, 1200),
                        "level": "verified_on_page", "text_sha256": text_sha256 or ss._sha256(text.encode())}}
    place = person.get("location_fields")
    if not isinstance(place, dict):  # Old sealed journals kept a display location only; no new paid search.
        parsed = ss.parse_location(person.get("location"))
        place = {"city": parsed.get("city"), "region": parsed.get("state")}
    address = site.get("address") or {}
    assigned = quoted_role_address(person, site.get("person_quote") or "")
    assigned_street, assigned_units = address_units(ss.words(hash_units(assigned.get("street") or "")))
    target_street, target_units = address_units(ss.words(hash_units(address.get("street") or "")))
    assigned_street, target_street = ss.street_anchor(assigned_street), ss.street_anchor(target_street)
    named_facility_mismatch = bool(assigned.get("street") and not assigned_street
                                  and site["task_input"].get("site_name")
                                  and not role_scope_suffix(assigned["street"], known_names, allow_news=False))
    role_mismatch = bool(named_facility_mismatch or assigned.get("city") and address.get("city")
                         and ss._canon(assigned["city"], ss.CITY_WORDS) != ss._canon(address["city"], ss.CITY_WORDS)
                         or assigned.get("state") and address.get("state") and assigned["state"] != address["state"]
                         or assigned_street and target_street
                         and ss._canon(assigned_street, ss.STREET_WORDS) != ss._canon(target_street, ss.STREET_WORDS)
                         or assigned_units and target_units and assigned_units != target_units)
    city, target_city = place.get("city"), address.get("city")
    region = _text(place.get("region"))
    state = region.upper() if region.upper() in ss.STATE_NAMES else ss.STATE_CODES.get(ss.normalized(region))
    mismatch = bool(role_mismatch or city and target_city and ss._canon(city, ss.CITY_WORDS) != ss._canon(target_city, ss.CITY_WORDS)
                    or state and address.get("state") and state != address["state"]
                    or place.get("country") and ss.normalized(place["country"]) not in
                    {"us", "usa", "united states", "united states of america"})
    corporate = bool(set(title.split()) & {"owner", "president", "director"})
    return {"site_key": site["site_key"], "status": "unknown",
            "route": "hold" if mismatch and not corporate else "corporate_referral",
            "reason": "target_site_location_mismatch" if mismatch else "target_site_responsibility_unproven",
            "proof": None}


def target(site, calls, search=True):
    """The person this site's enrichment is for, or (None, code). A people search, when allowed and needed, goes
    through ``calls``."""
    if site["published_person_email"]:
        return None, "published_person_email"
    if not site["operator_domain"]:
        return None, "operator_domain_unproven"
    person, code = quoted_person(site)
    if person is not None or not search:
        return person, code
    key, call = calls("search", [site["operator_domain"], list(TITLES)], site["site_key"],
                      lambda _: search_body(site), seal_search(site))
    found = call["observation"] if call is not None and call["state"] == "answered" else None
    if found is None or not found.get("candidate"):
        return None, code if found is None else "no_current_person_found"
    candidates = found.get("candidates") if isinstance(found.get("candidates"), list) else [found["candidate"]]
    referral = None
    for chosen in candidates:
        person = {"name": chosen["name"], "title": chosen["title"], "location": chosen["location"],
                  "location_fields": chosen.get("location_fields"), "sourcing": "provider_sourced",
                  "proof": {"source": "fullenrich_people_search", "request_digest": key,
                            "current_employment": chosen["current_employment"]}}
        person["corroboration"] = corroborate(site, person)
        person["site_responsibility"] = site_responsibility(site, person)
        route = person["site_responsibility"]["route"]
        if route == "site_contact":
            return person, None
        if route == "corporate_referral" and referral is None:
            referral = person
    return (referral, None) if referral is not None else (None, "target_site_location_mismatch")


def enrich_identity(site, person):
    return [ss.words(person["name"]), site["operator_domain"]]


def enrich_body(site, person):
    first, last = name_parts(person["name"])

    def body(key):
        item = {"first_name": first, "last_name": last, "domain": site["operator_domain"],
                "company_name": site["operator"] or None, "enrich_fields": ["contact.work_emails"],
                "custom": {"call": key}}
        return {"name": "blueprint-contact-lookup-" + key[:12],
                "data": [{name: value for name, value in item.items() if value is not None}]}
    return body


def collect(book, client, sites, *, wait_seconds, monotonic, sleep):
    """Read each started enrichment until it ends or ``wait_seconds`` pass, and journal each ended one once. Reads are
    not billed; a failed read is tried again on the next pass or run."""
    replay, pending = Calls(book, "replay"), {}
    for site in sites:
        person, _ = target(site, replay)
        key = Calls.key("enrich", enrich_identity(site, person)) if person is not None else None
        if key is not None and book.calls.get(key, {}).get("state") == "created":
            pending[key] = (book.calls[key], person, site)
    errors, deadline = Counter(), monotonic() + wait_seconds
    while pending:
        for key, (call, person, site) in list(pending.items()):
            try:
                status, raw = client.result(call["enrichment_id"])
                fields = seal_result(status, raw, key, call, person, site)
            except Refused:
                raise
            except LookupFailure as error:
                errors[str(error)] += 1
                continue
            if fields is None:
                continue
            book.append(call=key, attempt=call["attempt"], **fields)
            if fields["event"] == "refused":
                call.update(state="refused", closed=True, code=fields["code"])
            else:
                call.update(state="answered", closed=True, credits=fields["credits"], observation=fields["observation"])
            del pending[key]
        if not pending or monotonic() + POLL_SECONDS > deadline:
            break
        sleep(POLL_SECONDS)
    return len(pending), errors


# --- records and the recipient hand-off ------------------------------------------------------------
def result(site, person, key, call):
    """One looked-up email in the common shape: the provider's own status mapped to valid or not_valid, and the address
    only when it is usable."""
    observation = call["observation"] if call["state"] == "answered" else {}
    status = observation.get("status")
    valid = verification(status) if call["state"] == "answered" else "not_valid"
    address = observation.get("address") if valid == "valid" and not observation.get("reason") else None
    if address is not None:
        address, reason = accept_address(address, status, site["operator_domains"], person["name"])
        if reason:
            observation = {**observation, "reason": reason}
    usable = isinstance(address, str)
    pending = {"created": ("pending", "contact_lookup_result_pending"), "unknown": ("unknown", "contact_lookup_outcome_unknown"),
               "refused": ("refused", "contact_lookup_not_sent")}.get(call["state"], (None, None))
    return {"source": "provider_lookup", "label": "looked_up",
            "provider": {"name": PROVIDER, "status": status, "verification": valid, "score": None,
                         "outcome": observation.get("outcome") or pending[0], "checked_at": observation.get("checked_at"),
                         "request_digest": key, "enrichment_id": call.get("enrichment_id"), "credits": call.get("credits")},
            "usable": usable,
            "reason": None if usable else (observation.get("reason") or call.get("code") or pending[1]
                                           or "contact_lookup_status_not_valid"),
            "address": address if usable else None, "operator_domain": site["operator_domain"], "person": person}


def usable_lookup(lookup, *, operator_domains=None, site_key=None):
    """Recheck the provider status, person label and business address at the admission boundary."""
    if not isinstance(lookup, dict) or lookup.get("source") != "provider_lookup" or lookup.get("usable") is not True:
        return False
    provider, person, address = lookup.get("provider"), lookup.get("person"), lookup.get("address")
    if (not isinstance(provider, dict) or provider.get("name") != PROVIDER
            or provider.get("verification") != "valid" or verification(provider.get("status")) != "valid"
            or not isinstance(person, dict) or person.get("sourcing") not in ("quoted_person", "provider_sourced")
            or name_parts(person.get("name")) is None):
        return False
    responsibility = person.get("site_responsibility")
    if responsibility is not None:
        if (not isinstance(responsibility, dict)
                or set(responsibility) != {"site_key", "status", "route", "reason", "proof"}
                or site_key is not None and responsibility.get("site_key") != site_key):
            return False
        if responsibility["route"] == "corporate_referral":
            if (responsibility["status"] != "unknown" or responsibility["proof"] is not None
                    or responsibility["reason"] not in
                    ("target_site_responsibility_unproven", "target_site_location_mismatch")):
                return False
        elif responsibility["route"] == "site_contact":
            proof = responsibility["proof"]
            if (responsibility["status"] != "verified" or responsibility["reason"] is not None
                    or not isinstance(proof, dict) or proof.get("level") != "verified_on_page"
                    or not ss._public_url(proof.get("url")) or ss.never_fetch(proof.get("url"))
                    or not _text(proof.get("quote"), 1200)
                    or not isinstance(proof.get("text_sha256"), str)
                    or not re.fullmatch(r"[a-f0-9]{64}", proof["text_sha256"])):
                return False
        else:
            return False
    domain = lookup.get("operator_domain")
    if not isinstance(domain, str) or not domain:
        return False
    domains = operator_domains if isinstance(operator_domains, (list, tuple)) else [domain]
    kept, _ = accept_address(address, provider["status"], domains, person["name"])
    return kept is not None and kept == address and on_domain(domain, domains)


def _choice(rank, *, address=None, lookup=None, site_key=None):
    base = {"schema_version": RECIPIENT, "rank": rank, "choice": CHOICES[rank - 1]}
    if lookup is None:
        kind = {1: "person_email", 3: "team_inbox", 4: "general_inbox"}.get(rank, "none")
        return {**base, "kind": kind, "source": "published" if address else "none",
                "labels": ["published"] if address else [], "address": address, "person": None, "provider": None}
    person, provider = lookup["person"], lookup["provider"]
    labels, corroborated = ["looked_up", person["sourcing"]], None
    if person["sourcing"] == "provider_sourced":
        corroboration = person.get("corroboration")
        corroborated = isinstance(corroboration, dict) and corroboration.get("corroborated") is True
        labels.append("corroborated" if corroborated else "uncorroborated")
    return {**base, "kind": "person_email", "source": "provider_lookup", "labels": labels, "address": lookup["address"],
            "person": {"name": person["name"], "title": person.get("title"), "sourcing": person["sourcing"],
                       "corroborated": corroborated, "site_responsibility": person.get("site_responsibility") or {
                           "site_key": site_key, "status": "unknown", "route": "corporate_referral",
                           "reason": "target_site_responsibility_unproven", "proof": None}},
            "provider": {name: provider.get(name) for name in ("name", "status", "checked_at", "request_digest")}}


def choose_recipient(contact_record, lookups=(), *, today=None):
    """The admission hand-off's recipient for one site, in the owner's order: 1 a published, verified person email;
    2 a looked-up person email (``quoted_person`` or ``provider_sourced``, FullEnrich DELIVERABLE); 3 a published team
    inbox; 4 a published general inbox; 5 none. Pure: it reads only its arguments, and a malformed one chooses
    nothing from it. Admission supplies ``today`` to recheck the source date against its current UTC day.
    Every choice carries its source (published, provider_lookup or none) and labels."""
    record = contact_record if isinstance(contact_record, dict) else {}
    published = record.get("recipient") if isinstance(record.get("recipient"), dict) else {}
    email = record.get("email") if isinstance(record.get("email"), dict) else {}
    address = published.get("address") if email.get("verified") is True else None
    address = address if isinstance(address, str) and ss.email_address(address) == address else None
    person = record.get("person") if isinstance(record.get("person"), dict) else {}
    current = person.get("current") is True and (today is None or ss._fresh(person.get("date"), today))
    person = {"verified": person.get("verified") is True and current,
              "name": _text(person.get("name"))}
    role = ss.address_role(address, {"verified": False}) if address else None
    if address and role == "unknown":
        role = ss.address_role(address, person)
    kind = {"person": "person_email", "team": "team_inbox", "general": "general_inbox"}.get(role)
    if email.get("address") != address or email.get("level") != "verified_on_page" or published.get("kind") != kind:
        address, kind = None, None
    if kind == "person_email":
        return _choice(1, address=address)
    for lookup in lookups if isinstance(lookups, (list, tuple)) else ():
        if usable_lookup(lookup, operator_domains=record.get("operator_domains"), site_key=record.get("site_key")):
            return _choice(2, lookup=lookup, site_key=record.get("site_key"))
    if kind in ("team_inbox", "general_inbox"):
        return _choice(3 if kind == "team_inbox" else 4, address=address)
    return _choice(5)


def records(book, sites):
    """Each site's lookup record under RULE, recomputed from the journal alone. No call and no page read."""
    replay, built = Calls(book, "replay"), {}
    for site in sites:
        person, code = target(site, replay)
        lookups = []
        if person is not None:
            key = Calls.key("enrich", enrich_identity(site, person))
            if key in book.calls:
                lookups.append(result(site, person, key, book.calls[key]))
        built[site["site_key"]] = {"schema_version": RECORD, "rule_version": RULE, "site_key": site["site_key"],
                                   "operator_domain": site["operator_domain"], "skipped": code, "lookups": lookups,
                                   "recipient": choose_recipient(site["contact"], lookups, today=site["today"])}
    return built


def write_records(book, sites):
    built, rule = records(book, sites), RULE.removeprefix("blueprint.")
    for key, record in built.items():
        ss._write_derived(book.root / "records" / f"{key}.{rule}.json", record)
    return built


def load(workspace, *, states=None, today=None):
    """Recompute lookup records from the durable journal and current site evidence. The admission caller holds the
    out-dir lock and may pass its already-reconciled states. This function takes no lock, writes no lookup artifacts
    and calls no provider; Workspace.states may reconcile its own crash window. Cached records are not authority."""
    if states is None:
        states, _ = workspace.states()
    sites, _ = lookup_sites(workspace, states, today=today)
    return records(Book(workspace, readonly=True), sites)


# --- commands -------------------------------------------------------------------------------------
def refuse_on_worker(environ=None):
    """The daily worker spends only through paid_resource_admission, which this command does not use."""
    if ss.WORKER_FLAG in (os.environ if environ is None else environ):
        raise LookupFailure("contact_lookup_worker_needs_paid_admission")


def person_search_on(reference):
    """False without a reference; True only for the owner decision that allows a provider-sourced person."""
    if reference is None:
        return False
    if reference != PERSON_SEARCH_DECISION:
        raise LookupFailure("contact_lookup_person_search_reference_invalid")
    return True


def _recipients(built):
    counts = Counter({choice: 0 for choice in CHOICES})
    counts.update(record["recipient"]["choice"] for record in built.values())
    return dict(counts)


def lookup(workspace, *, client, owner_reference, max_credits, max_calls, person_search=None, apply=False,
           wait_seconds=WAIT_SECONDS, monotonic=time.monotonic, sleep=time.sleep, environ=None, today=None):
    """Look up a work email for each contact site that needs one, within the pinned ceilings; a dry run unless
    ``apply``. A dry run admits the same calls and writes and sends nothing; it counts only the searches of sites
    without a person, as their enrichments depend on the answers. Counts only."""
    refuse_on_worker(environ)
    search = person_search_on(person_search)
    if type(wait_seconds) is not int or not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
        raise LookupFailure("contact_lookup_wait_invalid")
    if not isinstance(client, FullEnrichClient):
        raise LookupFailure("contact_lookup_client_missing")
    with workspace.lock():
        states, _ = workspace.states()
        sites, unreadable = lookup_sites(workspace, states, today=today)
        book = Book(workspace)
        bounds = limits(book.pin, owner_reference, max_credits, max_calls)
        pinned = "pinned" if book.pin else "created" if apply else "would_create"
        code = ss.code_state() if apply else None
        if apply and book.pin is None:
            book.create_pin(bounds)
        calls = Calls(book, "apply" if apply else "dry", client=client, bounds=bounds, code=code)
        skipped, stop, pending, errors, recipients = Counter(), None, 0, Counter(), None
        for site in sites:
            try:
                person, why = target(site, calls, search)
                if person is None:
                    skipped[why] += 1
                    continue
                calls("enrich", enrich_identity(site, person), site["site_key"], enrich_body(site, person), seal_start)
            except Stop as error:
                stop = str(error)
                break
        if apply:
            try:
                pending, errors = collect(book, client, sites, wait_seconds=wait_seconds, monotonic=monotonic,
                                          sleep=sleep)
            except Refused as error:
                stop = stop or str(error)
                pending = sum(call["state"] == "created" for call in book.calls.values())
            recipients = _recipients(write_records(book, sites))
        credits, count = committed(book.calls)
    report = {"command": "lookup", "apply": apply, "provider": PROVIDER,
              "state": "stopped" if stop else "pending" if pending else "complete" if apply else "planned", "stop": stop,
              "person_search": search, "sites": len(sites), "unreadable": unreadable, "skipped": dict(skipped),
              "calls": dict(calls.counts), "pending_results": pending, "read_errors": dict(errors),
              "credits": {"committed": amount(credits), "max_credits": amount(bounds["max_credits"])},
              "committed_calls": count, "max_calls": bounds["max_calls"], "code": code,
              "pin": {"state": pinned, "owner_reference": bounds["owner_reference"]}}
    if recipients is not None:
        report["recipients"] = recipients
    return report


def tally(built, book, unreadable):
    """Counts only: never names, addresses or quotes."""
    people, outcomes, status, rejected, skipped = Counter(), Counter(), Counter(), Counter(), Counter()
    valid = usable = 0
    for record in built.values():
        if record["skipped"]:
            skipped[record["skipped"]] += 1
        for item in record["lookups"]:
            person, provider = item["person"], item["provider"]
            people[person["sourcing"]] += 1
            if person["sourcing"] == "provider_sourced":
                people["corroborated" if person["corroboration"]["corroborated"] else "uncorroborated"] += 1
            outcomes[provider["outcome"]] += 1
            status[provider["status"] if provider["status"] in STATUSES else "other" if provider["status"] else "none"] += 1
            valid, usable = valid + (provider["verification"] == "valid"), usable + item["usable"]
            if provider["verification"] == "valid" and not item["usable"]:
                rejected[item["reason"]] += 1
    searches = [call for call in book.calls.values() if call["kind"] == "search" and call["state"] == "answered"]
    used = sum((Decimal(call["credits"]) for call in book.calls.values() if call["state"] == "answered"), Decimal(0))
    credits, count = committed(book.calls)
    return {"schema_version": SUMMARY, "command": "summary", "state": "complete", "rule": RULE, "provider": PROVIDER,
            "sites": len(built), "unreadable": unreadable, "skipped": dict(skipped),
            "people": {name: people[name] for name in ("quoted_person", "provider_sourced", "corroborated",
                                                       "uncorroborated")},
            "searches": {"made": len(searches),
                         "people_returned": sum(call["observation"].get("people", 0) for call in searches),
                         "with_current_person": sum(bool(call["observation"].get("candidate")) for call in searches)},
            "enrichments": {"made": sum(call["kind"] == "enrich" and call["state"] != "refused"
                                        for call in book.calls.values()),
                            "outcomes": dict(outcomes), "status": dict(status), "valid": valid, "usable": usable,
                            "rejected_by_rule": dict(rejected)},
            "recipients": _recipients(built),
            "credits": {"used": amount(used), "committed": amount(credits),
                        "max_credits": book.pin["max_credits"] if book.pin else None, "calls": count,
                        "max_calls": book.pin["max_calls"] if book.pin else None}}


def summary(workspace):
    """Counts of people, searches, enrichments, statuses, recipients and credits, recomputed from the journal alone and
    written with the records once a lookup has pinned. No call and no page read."""
    with workspace.lock():
        states, _ = workspace.states()
        sites, unreadable = lookup_sites(workspace, states)
        book = Book(workspace)
        built = write_records(book, sites) if book.pin else records(book, sites)
        report = tally(built, book, unreadable)
        if book.pin:
            ss._write_derived(book.root / "summary.json", report)
    return report
