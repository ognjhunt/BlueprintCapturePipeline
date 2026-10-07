"""Hermetic contact lookup: a fake FullEnrich transport, fake pages and a synthetic site-screen out dir. No network."""
import ast
import importlib.util
import json
import re
import stat
import sys
from pathlib import Path

import pytest

from tests import daily_research_site_screen_fixture as fixture
from tests.daily_research_site_screen_fixture import (
    KEY,
    OTHER_PERSON,
    OWNER,
    PERSON,
    SITE_STRINGS,
    TODAY,
    FakePages,
    contact_answers,
    inventory_record,
    pages_for,
    screen,
    screen_answers,
)
from tools.daily_research import contact_lookup as cl
from tools.daily_research import site_screen as ss

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("contact_lookup_operator", ROOT / "tools/daily_research/operators/contact-lookup.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)

FULLENRICH_KEY = "synthetic-fullenrich-key-5d2c8e41"  # Never a real key; the tests check it never leaves the client.
LOOKUP_OWNER = "owner-decision-synthetic-lookup-20261005"
NOBODY = {"person_name": "", "person_title": "", "person_url": "", "person_quote": "", "person_date": ""}
UNPUBLISHED = {"email": "", "email_url": "", "email_quote": "", "channel_type": "none"}
ADDRESS = "avery.placeholder@operator-1.example"
FOUND = "jordan.fixture@operator-1.example"


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """pytest's tmp_path is on storage the out-dir guard refuses, a shell may set the worker flag, and Git is slow."""
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", ())
    monkeypatch.delenv(ss.WORKER_FLAG, raising=False)
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"})
    monkeypatch.setattr(cl, "utc_today", lambda: TODAY)


class FakeFullEnrich:
    """FullEnrich's API v2 behind FullEnrichClient's transport seam. ``people`` answers a search by company domain,
    ``emails`` an enrichment by (first name, last name, domain) with (email, status, profile), and ``waiting`` holds
    the answers a result read gets before it is finished. Every request is recorded."""

    def __init__(self):
        self.calls, self.people, self.emails, self.waiting, self.jobs, self.scripted = [], {}, {}, {}, {}, []

    def __call__(self, method, path, *, headers, body, timeout):
        request = json.loads(body) if body is not None else None
        self.calls.append({"method": method, "path": path, "headers": dict(headers), "body": request})
        if self.scripted:
            answer = self.scripted.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer
        if (method, path) == ("POST", "/api/v2/people/search"):
            hits = self.people.get(request["current_company_domains"][0]["value"], [])
            return 200, json.dumps({"people": hits, "metadata": {"total": len(hits), "credits": 0.25 * len(hits),
                                                                  "offset": 0}}).encode()
        if (method, path) == ("POST", "/api/v2/contact/enrich/bulk"):
            job = f"00000000-0000-4000-8000-{len(self.jobs) + 1:012d}"
            self.jobs[job] = request["data"][0]
            return 200, json.dumps({"enrichment_id": job}).encode()
        job = path.removeprefix("/api/v2/contact/enrich/bulk/")
        assert method == "GET" and job in self.jobs, path
        item = self.jobs[job]
        who = (item["first_name"], item["last_name"], item["domain"])
        if self.waiting.get(who):
            answer = self.waiting[who].pop(0)
            return answer if isinstance(answer, tuple) else (200, json.dumps(
                {"id": job, "status": answer, "cost": {"credits": 0}, "data": []}).encode())
        email, status, profile = self.emails.get(who, (None, None, None))
        info = {"most_probable_work_email": {"email": email, "status": status} if email else None,
                "work_emails": [{"email": email, "status": status}] if email else [],
                # Never asked for: the lookup must keep none of it.
                "personal_emails": [{"email": "avery@personal-mail.example", "status": "DELIVERABLE"}],
                "most_probable_phone": {"number": "+1 555-010-0199"}, "phones": [{"number": "+1 555-010-0199"}]}
        return 200, json.dumps({"id": job, "name": "synthetic", "status": "FINISHED", "cost": {"credits": 1 if email else 0},
                                "data": [{"input": {"first_name": item["first_name"], "last_name": item["last_name"],
                                                    "company_domain": item["domain"]},
                                          "custom": item.get("custom", {}), "contact_info": info,
                                          "profile": profile}]}).encode()

    def posts(self, path):
        return [call["body"] for call in self.calls if call["method"] == "POST" and call["path"] == path]


def profile(name, domain="operator-1.example", **job):
    return {"id": "p-1", "full_name": name, "employment": {"current": {
        "title": "Plant Manager", "is_current": True, "start_at": "2022-03-15T00:00:00Z",
        "company": {"id": "c-1", "name": "Synthetic Operator 1", "domain": domain}, **job}}}


def searched(name, title="Plant Manager", domain="operator-1.example", **job):
    """One people search hit: a current job at ``domain`` unless ``job`` changes it."""
    first, last = name.split()
    current = {"title": title, "seniority": "Manager", "is_current": True, "start_at": "2022-03-15T00:00:00Z",
               "company": {"id": "c-1", "name": "Synthetic Operator 1", "domain": domain}, **job}
    return {"id": f"p-{first.lower()}", "full_name": name, "first_name": first, "last_name": last,
            "location": {"country": "United States", "country_code": "US", "city": "Fixture City", "region": "Texas"},
            "social_profiles": {"professional_network": {"url": "https://www.linkedin.com/in/synthetic-profile"}},
            "employment": {"current": current, "all": [current]}}


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def quoted(number, **changes):
    """A contact answer with a named current person proven on their page and no published address."""
    return contact_answers(number, **{**UNPUBLISHED, **changes})


def nobody(number):
    return contact_answers(number, **NOBODY, **UNPUBLISHED)


def site_screen_out(tmp_path, contacts, *, kept=None, screen_changes=None):
    """A site-screen out dir with one outreach-ready site per contact answer, its contact run collected and verified.
    ``kept`` adds text to pages the screen keeps; ``screen_changes`` changes one site's screen answers."""
    records = [inventory_record(number) for number in range(1, len(contacts) + 1)]
    answers = [screen_answers(number, **(screen_changes or {}).get(number, {})) for number in range(1, len(contacts) + 1)]
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    pages.update({url: pages[url] + " " + text for url, text in (kept or {}).items()})
    workspace, provider, _, _ = screen(tmp_path, records, answers, pages)
    keys = [ss.from_inventory(record)["site_key"] for record in records]
    for key, content in zip(keys, contacts):
        provider.contacts[key] = {"content": content, "basis": []}
    client = ss.TaskClient(KEY, transport=provider)
    ss.contact(workspace, client=client, owner_reference=OWNER, ceiling_usd="5", max_runs=100, apply=True)
    reader = FakePages({url: text for answer in contacts for url, text in pages_for(answer).items()})
    ss.collect(workspace, client=client, reader=reader, today=TODAY, wait_seconds=0)
    ss.verify(workspace, reader=reader, today=TODAY)
    return workspace, keys


def lookup(workspace, api, **options):
    clock = options.pop("clock", None) or Clock()
    settings = {"owner_reference": LOOKUP_OWNER, "max_credits": "20", "max_calls": 50, "apply": True, "wait_seconds": 0,
                "monotonic": clock.monotonic, "sleep": clock.sleep, **options}
    return cl.lookup(workspace, client=cl.FullEnrichClient(FULLENRICH_KEY, transport=api), **settings)


def stored(workspace):
    return b"".join(path.read_bytes() for path in sorted((workspace.root / cl.FOLDER).rglob("*")) if path.is_file())


# --- the quoted person ----------------------------------------------------------------------------
@pytest.mark.parametrize("preposition", ["at", "for", "of"])
@pytest.mark.parametrize("assignment", [
    "2 Other Road, Other Fixture City, OH",
    "2 Other Road, Fixture City, TX",
    "2 Other Road",
    "2 Other Road, Suite 4",
    "2 Other Road and he left the conference early",
    "Other Fixture Plant, Other Fixture City, OH",
    "Rival Fixture Works, Fixture City, TX",
    "Said Fixture Works, Fixture City, TX",
    "Rival and Sons, Fixture City, TX",
    "Rival and Sons, Fixture City, TX and he discussed the plant",
])
def test_quoted_manager_at_another_facility_is_held_before_paid_enrichment(tmp_path, assignment, preposition):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1, person_quote=f"{PERSON} is Plant Manager {preposition} {assignment}.")])
    api = FakeFullEnrich()
    result = lookup(workspace, api)
    assert result["calls"]["made"] == 0 and api.calls == []
    record = cl.load(workspace)[key]
    assert record["lookups"] == [] and record["skipped"] == "target_site_location_mismatch"


@pytest.mark.parametrize("assignment", ["Synthetic Works", "the Synthetic Works plant", "Synthetic Operator 1", "1 Example Road"])
@pytest.mark.parametrize("preposition", ["at", "for", "of"])
def test_named_target_or_company_role_keeps_unknown_referral_scope(assignment, preposition):
    site = qualification_site()
    site["person_quote"] = f"Jordan Fixture is Plant Manager {preposition} {assignment}, Fixture City, TX."
    person = {"name": "Jordan Fixture", "title": "Plant Manager", "location": None}
    responsibility = cl.site_responsibility(site, person)
    assert responsibility["route"] == "corporate_referral" and responsibility["status"] == "unknown"


@pytest.mark.parametrize("assignment", ["Synthetic Works and Sons", "Synthetic Works & Sons"])
def test_named_target_with_conjunction_keeps_unknown_referral_scope(assignment):
    site = qualification_site()
    site["task_input"]["site_name"] = "Synthetic Works & Sons"
    site["person_quote"] = f"Jordan Fixture is Plant Manager at {assignment}, Fixture City, TX."
    responsibility = cl.site_responsibility(site, {"name": "Jordan Fixture", "title": "Plant Manager"})
    assert responsibility["route"] == "corporate_referral" and responsibility["status"] == "unknown"


@pytest.mark.parametrize("name", ["José García", "Zoë Fixture"])
def test_unicode_quoted_manager_is_held_before_enrichment(tmp_path, name):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1, person_name=name,
                                       person_quote=f"{name} is Plant Manager at 2 Other Road, Other City, OH.")])
    api = FakeFullEnrich()
    assert lookup(workspace, api)["calls"]["made"] == 0 and api.calls == []
    assert cl.load(workspace)[key]["skipped"] == "target_site_location_mismatch"
    assert cl.site_role_quote(qualification_site(), {"name": name, "title": "Plant Manager"},
                              f"{name} is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1.")


def test_a_quoted_person_gets_one_enrichment_and_a_deliverable_email_is_kept(tmp_path):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    api.emails[("Avery", "Placeholder", "operator-1.example")] = (ADDRESS, "DELIVERABLE", profile(PERSON))
    result = lookup(workspace, api)
    assert (result["state"], result["calls"]["made"], result["credits"]["committed"]) == ("complete", 1, "1")
    start, read = api.calls
    assert (start["method"], start["path"], read["method"]) == ("POST", "/api/v2/contact/enrich/bulk", "GET")
    assert start["headers"]["Authorization"] == "Bearer " + FULLENRICH_KEY
    (item,) = start["body"]["data"]
    assert item == {"first_name": "Avery", "last_name": "Placeholder", "domain": "operator-1.example",
                    "company_name": "Synthetic Operator 1", "enrich_fields": ["contact.work_emails"],
                    "custom": {"call": item["custom"]["call"]}}
    records = cl.load(workspace)
    record = json.loads(json.dumps(records[key]))
    assert (record["schema_version"], record["rule_version"]) == (cl.RECORD, "blueprint.contact-lookup-rule.v2")
    (found,) = record["lookups"]
    responsibility = found["person"].pop("site_responsibility")
    assert responsibility == {"site_key": key, "status": "unknown", "route": "corporate_referral",
                              "reason": "target_site_responsibility_unproven", "proof": None}
    provider = found.pop("provider")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", provider.pop("checked_at"))
    assert provider.pop("request_digest") == item["custom"]["call"] and re.fullmatch(r"[0-9a-f]{64}", item["custom"]["call"])
    assert provider == {"name": "fullenrich", "status": "DELIVERABLE", "verification": "valid", "score": None,
                        "outcome": "finished", "enrichment_id": "00000000-0000-4000-8000-000000000001", "credits": "1"}
    assert found == {"source": "provider_lookup", "label": "looked_up", "usable": True, "reason": None,
                     "address": ADDRESS, "operator_domain": "operator-1.example",
                     "person": {"name": PERSON, "title": "Plant Manager", "location": None, "sourcing": "quoted_person",
                                "proof": {"source": "site_contact", "url": "https://operator-1.example/team",
                                          "level": "verified_on_page", "date": "2026-06-01", "current": True},
                                "corroboration": None}}
    recipient = record["recipient"]
    assert (recipient["rank"], recipient["choice"], recipient["kind"], recipient["source"], recipient["labels"],
            recipient["address"]) == (2, "looked_up_person_email", "person_email", "provider_lookup",
                                      ["looked_up", "quoted_person"], ADDRESS)
    # Never asked for and never kept: personal addresses and phones.
    assert b"personal-mail" not in stored(workspace) and b"555-010" not in stored(workspace)
    again = lookup(workspace, api)
    assert len(api.calls) == 2 and again["calls"]["made"] == 0 and cl.load(workspace) == records


@pytest.mark.parametrize("email, status, found_profile, reason", [
    (ADDRESS, "HIGH_PROBABILITY", profile(PERSON), "contact_lookup_status_not_valid"),  # A catch-all guess.
    (ADDRESS, "CATCH_ALL", profile(PERSON), "contact_lookup_status_not_valid"),
    (ADDRESS, "INVALID", profile(PERSON), "contact_lookup_status_not_valid"),
    ("avery.placeholder@operator-1.example.net", "DELIVERABLE", profile(PERSON), "contact_lookup_off_operator_domain"),
    ("avery.placeholder@gmail.com", "DELIVERABLE", profile(PERSON), "contact_lookup_free_mail"),
    ("info@operator-1.example", "DELIVERABLE", profile(PERSON), "contact_lookup_role_inbox"),
    ("j.smith@operator-1.example", "DELIVERABLE", profile(PERSON), "contact_lookup_address_not_personal"),
    (ADDRESS, "DELIVERABLE", profile(OTHER_PERSON), "contact_lookup_person_mismatch"),
    (ADDRESS, "DELIVERABLE", profile(PERSON, domain="other-operator.example"),
     "contact_lookup_person_not_current_at_operator"),
    (None, None, None, "contact_lookup_email_not_found"),
])
def test_only_a_deliverable_email_for_the_same_person_on_the_operator_domain_is_kept(tmp_path, email, status,
                                                                                     found_profile, reason):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    api.emails[("Avery", "Placeholder", "operator-1.example")] = (email, status, found_profile)
    lookup(workspace, api)
    record = cl.load(workspace)[key]
    (found,) = record["lookups"]
    assert (found["usable"], found["address"], found["reason"], found["provider"]["status"]) == (False, None, reason, status)
    assert record["recipient"]["choice"] == "none"
    if email:
        assert email.encode() not in stored(workspace)  # A rejected address is never written.


def test_only_a_named_current_person_on_a_proven_operator_domain_is_looked_up(tmp_path):
    contacts = [quoted(1), quoted(2, person_date="2024-01-10"), nobody(3), contact_answers(4), quoted(5)]
    workspace, keys = site_screen_out(tmp_path, contacts, screen_changes={
        5: {"operator_identity_url": "https://records.synthetic.gov/facility/5"}})  # A government page proves no domain.
    api = FakeFullEnrich()
    result = lookup(workspace, api)
    assert [body["data"][0]["domain"] for body in api.posts("/api/v2/contact/enrich/bulk")] == ["operator-1.example"]
    assert result["skipped"] == {"person_not_current": 1, "no_verified_person": 1, "published_person_email": 1,
                                 "operator_domain_unproven": 1}
    records = cl.load(workspace)
    published = records[keys[3]]["recipient"]
    assert (published["rank"], published["source"], published["labels"], published["address"]) == (
        1, "published", ["published"], "avery.placeholder@operator-4.example")
    assert records[keys[1]]["lookups"] == [] and records[keys[1]]["skipped"] == "person_not_current"


# --- the provider-sourced person ------------------------------------------------------------------
def test_people_search_stays_off_unless_the_run_names_the_owner_decision(tmp_path):
    workspace, _ = site_screen_out(tmp_path, [nobody(1)])
    api = FakeFullEnrich()
    off = lookup(workspace, api)
    assert api.calls == [] and off["person_search"] is False and off["skipped"] == {"no_verified_person": 1}
    for reference in ("owner-decision-synthetic-20261005", cl.PERSON_SEARCH_DECISION + "-draft", "PENDING"):
        with pytest.raises(ss.ScreenError, match="^contact_lookup_person_search_reference_invalid$"):
            lookup(workspace, api, person_search=reference)
    assert api.calls == [] and cl.PERSON_SEARCH_DECISION == "owner-decision-provider-sourced-person-20261005"


@pytest.mark.parametrize("corroborating", [True, False])
@pytest.mark.parametrize("provider_title", ["Plant Manager", "Production Manager", "Warehouse Manager", "Site Leader",
                                           "Vice President of Operations", "VP Operations", "Chief Operating Officer",
                                           "COO", "Manufacturing Engineering Manager"])
def test_people_search_keeps_only_a_person_working_at_the_operator_now_in_a_listed_role(tmp_path, corroborating,
                                                                                      provider_title):
    sentence = f"{OTHER_PERSON}, {provider_title} of Synthetic Operator 1, opened the new lathe cell this spring."
    kept = {"https://operator-1.example/news": sentence} if corroborating else None
    workspace, (key,) = site_screen_out(tmp_path, [nobody(1)], kept=kept)
    api = FakeFullEnrich()
    api.people["operator-1.example"] = [
        # A past employee whose current job is elsewhere: the company filter can match a previous employer.
        searched("Casey Placeholder", domain="other-operator.example"),
        searched("Riley Fixture", is_current=False, end_at="2024-01-31T00:00:00Z"),  # Ended at the operator.
        searched("Morgan Fixture", title="Vice President of Sales"),  # Not a listed role.
        searched(OTHER_PERSON, title=provider_title),
    ]
    api.emails[("Jordan", "Fixture", "operator-1.example")] = (FOUND, "DELIVERABLE", profile(OTHER_PERSON))
    result = lookup(workspace, api, person_search=cl.PERSON_SEARCH_DECISION)
    (search,) = api.posts("/api/v2/people/search")
    assert search["current_company_domains"] == [{"value": "operator-1.example", "exact_match": True}]
    assert [title["value"] for title in search["current_position_titles"]] == list(cl.TITLES)
    assert all(title["exact_match"] is False for title in search["current_position_titles"])
    assert search["person_locations"] == [{"value": "United States", "exact_match": True}]
    assert all(not cl.listed_title(title) for title in ("VP Sales", "VP Finance", "Human Resources Director",
                                                       "Operations Audit Director", "Retired Plant Manager",
                                                       "Vice President Strategy", "Vice President Procurement",
                                                       "Vice President Quality", "VP Quality"))
    assert search["limit"] == cl.SEARCH_LIMIT
    assert [body["data"][0]["first_name"] for body in api.posts("/api/v2/contact/enrich/bulk")] == ["Jordan"]
    assert (result["calls"]["made"], result["credits"]["committed"]) == (2, "2")  # Four people at 0.25, one email.
    (found,) = cl.load(workspace)[key]["lookups"]
    person = found["person"]
    assert (found["usable"], found["address"], person["sourcing"], person["name"], person["title"], person["location"]) == (
        True, FOUND, "provider_sourced", OTHER_PERSON, provider_title, "Fixture City, Texas, United States")
    assert person["proof"]["source"] == "fullenrich_people_search"
    assert person["proof"]["current_employment"] == {"field": "employment.current.is_current",
                                                     "company_domain": "operator-1.example",
                                                     "start_at": "2022-03-15T00:00:00Z"}
    corroboration = person["corroboration"]
    if corroborating:  # Our own read, kept by the site screen: the page, its sentence and the page text's digest.
        page = json.loads(workspace.path("screen", "evidence", key).read_text())["pages"]["https://operator-1.example/news"]
        assert corroboration == {"corroborated": True, "url": "https://operator-1.example/news", "quote": sentence,
                                 "level": "verified_on_page", "text_sha256": page["sha256"]}
    else:
        assert corroboration == {"corroborated": False, "url": None, "quote": None, "level": None, "text_sha256": None}
    recipient = cl.load(workspace)[key]["recipient"]
    assert (recipient["rank"], recipient["labels"]) == (
        2, ["looked_up", "provider_sourced", "corroborated" if corroborating else "uncorroborated"])
    assert b"linkedin" not in stored(workspace) and b"Casey" not in stored(workspace)


# --- spend: pin, journal, never twice -------------------------------------------------------------
def test_the_first_apply_pins_the_ceilings_and_a_later_run_may_only_lower_them(tmp_path):
    workspace, _ = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    dry = lookup(workspace, api, apply=False)
    assert (dry["state"], dry["calls"]["would_make"], dry["pin"]["state"]) == ("planned", 1, "would_create")
    assert api.calls == [] and not (workspace.root / cl.FOLDER).exists()
    lookup(workspace, api)
    pin = json.loads((workspace.root / cl.FOLDER / "owner_ceiling.json").read_text())
    assert {name: pin[name] for name in ("schema_version", "owner_reference", "max_credits", "max_calls")} == {
        "schema_version": cl.OWNER_CEILING, "owner_reference": LOOKUP_OWNER, "max_credits": "20", "max_calls": 50}
    for changes, code in (({"max_credits": "21"}, "contact_lookup_credits_above_pin"),
                          ({"max_calls": 51}, "contact_lookup_calls_above_pin"),
                          ({"owner_reference": "owner-decision-synthetic-other"}, "contact_lookup_owner_reference_mismatch"),
                          ({"max_credits": "0"}, "contact_lookup_credits_invalid"),
                          ({"max_calls": True}, "contact_lookup_calls_invalid")):
        with pytest.raises(ss.ScreenError, match=f"^{code}$"):
            lookup(workspace, api, **changes)
    assert lookup(workspace, api, max_credits="2", max_calls=5)["pin"]["state"] == "pinned"


def test_every_call_is_admitted_and_journaled_before_it_is_sent(tmp_path):
    workspace, _ = site_screen_out(tmp_path, [quoted(1), quoted(2), quoted(3)])
    api = FakeFullEnrich()
    for number in (1, 2, 3):
        api.emails[("Avery", "Placeholder", f"operator-{number}.example")] = (
            f"avery.placeholder@operator-{number}.example", "DELIVERABLE", None)
    result = lookup(workspace, api, max_credits="2")
    assert (result["state"], result["stop"], result["calls"]["made"]) == ("stopped", "contact_lookup_credit_ceiling_reached", 2)
    events = [json.loads(line) for line in (workspace.root / cl.FOLDER / "spend.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["pinned", "intent", "created", "intent", "created", "answered", "answered"]
    assert {event["owner_reference"] for event in events if event["event"] == "intent"} == {LOOKUP_OWNER}
    assert lookup(workspace, api, max_credits="2", max_calls=2)["stop"] == "contact_lookup_max_calls_reached"
    assert len(api.posts("/api/v2/contact/enrich/bulk")) == 2


@pytest.mark.parametrize("answer, code, retried", [
    ((401, b'{"code": "error.api.key"}'), "contact_lookup_provider_auth_refused", True),
    ((429, b'{"code": "error.rate.limit"}'), "contact_lookup_provider_rate_limited", True),
    ((302, b""), "contact_lookup_provider_redirect_refused", True),
    (ss.TransportError("synthetic", sent=False), "contact_lookup_provider_unreachable", True),
    ((503, b"upstream"), "contact_lookup_outcome_unknown", False),
    (ss.TransportError("synthetic", sent=True), "contact_lookup_outcome_unknown", False),
    ((200, b'{"no": "id"}'), "contact_lookup_response_invalid", False),
])
def test_a_refusal_stops_and_is_tried_later_but_a_lookup_that_may_exist_is_never_sent_again(tmp_path, answer, code,
                                                                                            retried):
    workspace, _ = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    api.scripted = [answer]
    result = lookup(workspace, api)
    assert (result["state"], result["stop"]) == ("stopped", code)
    events = [json.loads(line)["event"] for line in (workspace.root / cl.FOLDER / "spend.jsonl").read_text().splitlines()]
    assert events == ["pinned", "intent", "refused" if retried else "uncertain"]
    again = lookup(workspace, api)
    assert len(api.posts("/api/v2/contact/enrich/bulk")) == (2 if retried else 1)
    assert again["credits"]["committed"] == ("0" if retried else "1")  # An unknown outcome keeps its most committed.


def test_a_started_enrichment_is_read_until_it_finishes_and_never_started_again(tmp_path):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    who = ("Avery", "Placeholder", "operator-1.example")
    api.emails[who] = (ADDRESS, "DELIVERABLE", profile(PERSON))
    api.waiting[who] = [(400, b'{"message": "Enrichment not ready, try again in 30 seconds"}'), "IN_PROGRESS",
                        "IN_PROGRESS"]
    clock = Clock()
    first = lookup(workspace, api, wait_seconds=20, clock=clock)
    assert (first["state"], first["pending_results"], clock.sleeps) == ("pending", 1, [10, 10])
    assert cl.load(workspace)[key]["lookups"][0]["reason"] == "contact_lookup_result_pending"
    assert cl.load(workspace)[key]["recipient"]["choice"] == "none"
    second = lookup(workspace, api)
    assert (second["state"], second["calls"]["made"], second["pending_results"]) == ("complete", 0, 0)
    assert len(api.posts("/api/v2/contact/enrich/bulk")) == 1 and cl.load(workspace)[key]["lookups"][0]["usable"] is True


@pytest.mark.parametrize("ended", ["CREDITS_INSUFFICIENT", "RATE_LIMIT", "CANCELED"])
def test_an_enrichment_that_ends_unbilled_is_sent_again_by_a_later_run(tmp_path, ended):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    who = ("Avery", "Placeholder", "operator-1.example")
    api.emails[who], api.waiting[who] = (ADDRESS, "DELIVERABLE", profile(PERSON)), [ended]
    first = lookup(workspace, api)
    assert (first["state"], first["credits"]["committed"]) == ("complete", "0")
    (found,) = cl.load(workspace)[key]["lookups"]
    assert (found["usable"], found["provider"]["outcome"], found["reason"]) == (
        False, "refused", "contact_lookup_enrichment_" + ended.lower())
    lookup(workspace, api)
    assert len(api.posts("/api/v2/contact/enrich/bulk")) == 2 and cl.load(workspace)[key]["lookups"][0]["usable"] is True


def test_a_damaged_journal_or_pin_refuses(tmp_path):
    workspace, _ = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    lookup(workspace, api)
    folder = workspace.root / cl.FOLDER
    journal, pin = (folder / "spend.jsonl").read_bytes(), (folder / "owner_ceiling.json").read_bytes()
    (folder / "spend.jsonl").unlink()
    with pytest.raises(ss.ScreenError, match="^contact_lookup_journal_missing$"):
        lookup(workspace, api)
    (folder / "spend.jsonl").write_bytes(journal.replace(b'"max_credits":"20"', b'"max_credits":"99"', 1))
    with pytest.raises(ss.ScreenError, match="^contact_lookup_owner_ceiling_mismatch$"):
        lookup(workspace, api)
    (folder / "spend.jsonl").write_bytes(b"\n".join(journal.split(b"\n")[:1] + journal.split(b"\n")[2:]))
    with pytest.raises(ss.ScreenError, match="^contact_lookup_journal_invalid$"):  # An answer without its intent.
        lookup(workspace, api)
    (folder / "spend.jsonl").write_bytes(journal + b'{"torn')  # A crash mid-line is sealed and skipped.
    assert lookup(workspace, api)["calls"]["made"] == 0 and pin == (folder / "owner_ceiling.json").read_bytes()


# --- the recipient hand-off -----------------------------------------------------------------------
def contact_record(kind, address=None):
    return {"email": {"verified": address is not None, "address": address, "level": "verified_on_page"},
            "operator_domains": ["operator-1.example"], "person": {"verified": True, "current": True, "name": PERSON},
            "recipient": {"kind": kind, "address": address}}


def looked_up(sourcing="quoted_person", *, usable=True, corroborated=None):
    person = {"name": OTHER_PERSON, "title": "Plant Manager", "sourcing": sourcing,
              "corroboration": None if corroborated is None else {"corroborated": corroborated}}
    return {"source": "provider_lookup", "label": "looked_up", "usable": usable, "address": FOUND if usable else None,
            "operator_domain": "operator-1.example",
            "person": person, "provider": {"name": "fullenrich", "status": "DELIVERABLE" if usable else "CATCH_ALL",
                                            "verification": "valid" if usable else "not_valid"}}


@pytest.mark.parametrize("record, lookups, rank, choice, labels", [
    (contact_record("person_email", ADDRESS), [looked_up()], 1, "published_person_email", ["published"]),
    (contact_record("team_inbox", "sales@operator-1.example"), [looked_up()], 2, "looked_up_person_email",
     ["looked_up", "quoted_person"]),
    (contact_record("general_inbox", "info@operator-1.example"), [looked_up("provider_sourced", corroborated=False)], 2,
     "looked_up_person_email", ["looked_up", "provider_sourced", "uncorroborated"]),
    (contact_record("team_inbox", "sales@operator-1.example"), [looked_up(usable=False)], 3, "published_team_inbox",
     ["published"]),
    (contact_record("general_inbox", "info@operator-1.example"), [], 4, "published_general_inbox", ["published"]),
    (contact_record("none"), [looked_up(usable=False)], 5, "none", []),
    ({"recipient": "damaged"}, [{"usable": True, "address": FOUND}], 5, "none", []),  # Malformed input chooses nothing.
])
def test_the_recipient_follows_the_owner_order_and_carries_its_labels(record, lookups, rank, choice, labels):
    before = json.dumps([record, lookups], sort_keys=True)
    recipient = cl.choose_recipient(record, lookups)
    assert (recipient["schema_version"], recipient["rank"], recipient["choice"], recipient["labels"]) == (
        cl.RECIPIENT, rank, choice, labels)
    assert recipient["kind"] == {1: "person_email", 2: "person_email", 3: "team_inbox", 4: "general_inbox"}.get(rank, "none")
    assert recipient["source"] == {1: "published", 2: "provider_lookup", 3: "published", 4: "published"}.get(rank, "none")
    assert json.dumps([record, lookups], sort_keys=True) == before  # Pure: the inputs are unchanged.


@pytest.mark.parametrize("changes", [
    {"provider": {"name": "fullenrich", "status": "CATCH_ALL", "verification": "valid"}},
    {"provider": {"name": "another-provider", "status": "DELIVERABLE", "verification": "valid"}},
    {"provider": {"name": "fullenrich", "status": [], "verification": "valid"}},
    {"address": "jordan.fixture@other-operator.example"},
    {"address": "info@operator-1.example"},
    {"address": "jordan.fixture@gmail.com"},
    {"operator_domain": "other-operator.example"},
    {"person": {"name": "Jordan", "sourcing": "quoted_person"}},
])
def test_admission_rechecks_the_looked_up_status_person_and_current_operator(changes):
    recipient = cl.choose_recipient(contact_record("general_inbox", "info@operator-1.example"),
                                    [{**looked_up(), **changes}])
    assert recipient["choice"] == "published_general_inbox"


@pytest.mark.parametrize("changes", [
    {"recipient": {"kind": "person_email", "address": "info@operator-1.example"}},
    {"email": {"verified": True, "address": ADDRESS, "level": "in_citation_excerpt"}},
    {"recipient": {"kind": "person_email", "address": "invalid"}},
    {"email": {"verified": True, "address": FOUND, "level": "verified_on_page"}},
])
def test_admission_does_not_promote_a_malformed_or_mislabelled_published_address(changes):
    assert cl.choose_recipient({**contact_record("person_email", ADDRESS), **changes})["choice"] == "none"


def test_role_inboxes_never_become_person_emails_when_a_name_matches_the_role():
    assert cl.accept_address("sales@operator-1.example", "DELIVERABLE", ["operator-1.example"],
                             "Jordan Sales") == (None, "contact_lookup_role_inbox")
    record = {**contact_record("team_inbox", "sales@operator-1.example"),
              "person": {"verified": True, "name": "Jordan Sales"}}
    recipient = cl.choose_recipient(record)
    assert recipient["choice"] == "published_team_inbox" and recipient["person"] is None


@pytest.mark.parametrize("current", [False, None, "true", 1])
def test_published_person_must_still_be_current_and_can_fall_back_to_a_verified_lookup(current):
    record = {**contact_record("person_email", ADDRESS),
              "person": {"verified": True, "current": current, "name": PERSON}}
    assert cl.choose_recipient(record)["choice"] == "none"
    assert cl.choose_recipient(record, [looked_up()])["choice"] == "looked_up_person_email"


def test_a_stale_published_person_does_not_prevent_authorized_people_search(tmp_path):
    workspace, _ = site_screen_out(tmp_path, [contact_answers(1, person_date="2020-06-01")])
    api = FakeFullEnrich()
    api.people["operator-1.example"] = [searched(OTHER_PERSON)]
    api.emails[("Jordan", "Fixture", "operator-1.example")] = (FOUND, "DELIVERABLE", profile(OTHER_PERSON))
    result = lookup(workspace, api, person_search=cl.PERSON_SEARCH_DECISION)
    assert result["calls"]["made"] == 2 and result["recipients"]["looked_up_person_email"] == 1


def test_published_person_source_is_rechecked_at_the_admission_day(tmp_path):
    workspace, (key,) = site_screen_out(tmp_path, [contact_answers(1, person_date="2025-06-01")])
    states, _ = workspace.states()
    contact = ss.stage_records(workspace, states, "contact")[key]
    assert contact["person"]["current"] is True
    later = TODAY.replace(year=2027)
    assert cl.choose_recipient(contact, today=later)["choice"] == "none"
    api = FakeFullEnrich()
    api.people["operator-1.example"] = [searched(OTHER_PERSON)]
    api.emails[("Jordan", "Fixture", "operator-1.example")] = (FOUND, "DELIVERABLE", profile(OTHER_PERSON))
    report = lookup(workspace, api, person_search=cl.PERSON_SEARCH_DECISION, today=later)
    assert report["calls"]["made"] == 2 and report["recipients"]["looked_up_person_email"] == 1


def test_malformed_person_or_domains_choose_no_unsupported_address():
    record = {**contact_record("person_email", ADDRESS), "person": {}}
    assert cl.choose_recipient(record)["choice"] == "none"
    record = {**contact_record("none"), "operator_domains": [None, 1, {}]}
    assert cl.choose_recipient(record, [looked_up()])["choice"] == "none"


def test_load_recomputes_from_journal_ignores_cached_tampering_and_is_read_only(tmp_path, monkeypatch):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    api.emails[("Avery", "Placeholder", "operator-1.example")] = (ADDRESS, "DELIVERABLE", profile(PERSON))
    lookup(workspace, api)
    record_path = next((workspace.root / cl.FOLDER / "records").glob("*.json"))
    tampered = json.loads(record_path.read_text())
    tampered["lookups"][0]["address"] = "avery.placeholder@other-operator.example"
    record_path.write_text(json.dumps(tampered))
    before, calls = stored(workspace), len(api.calls)
    monkeypatch.setattr(workspace, "lock", lambda: pytest.fail("load must reuse the admission caller's lock"))
    result = cl.load(workspace)
    assert result[key]["recipient"]["address"] == ADDRESS
    assert stored(workspace) == before and len(api.calls) == calls
    (workspace.root / cl.FOLDER / "spend.jsonl").unlink()
    with pytest.raises(cl.LookupFailure, match="^contact_lookup_journal_missing$"):
        cl.load(workspace)


def test_load_recovers_a_pin_crash_in_memory_without_writing(tmp_path):
    workspace, (key,) = site_screen_out(tmp_path, [quoted(1)])
    book = cl.Book(workspace)
    book.create_pin(cl.limits(None, LOOKUP_OWNER, "20", 50))
    book.pin_path.unlink()
    before = stored(workspace)
    assert cl.load(workspace)[key]["recipient"]["choice"] == "none"
    assert not book.pin_path.exists() and stored(workspace) == before


def test_current_employment_history_requires_an_explicit_current_flag():
    person = searched(OTHER_PERSON)
    history = person["employment"].pop("current")
    history.pop("is_current")
    assert cl.candidate(person, ["operator-1.example"])[0] is None
    history["is_current"] = True
    found, reason = cl.candidate(person, ["operator-1.example"])
    assert reason is None and found["current_employment"]["field"] == "employment.all.is_current"


@pytest.mark.parametrize("city,region", [("Other Fixture City", "Texas"), ("Fixture City", "Ohio")])
def test_off_site_local_manager_is_held_before_enrichment(tmp_path, city, region):
    workspace, (key,) = site_screen_out(tmp_path, [nobody(1)])
    api = FakeFullEnrich()
    person = searched(OTHER_PERSON)
    person["location"].update(city=city, region=region)
    api.people["operator-1.example"] = [person]
    api.emails[("Jordan", "Fixture", "operator-1.example")] = (FOUND, "DELIVERABLE", profile(OTHER_PERSON))
    report = lookup(workspace, api, person_search=cl.PERSON_SEARCH_DECISION)
    assert report["calls"]["made"] == 1
    assert api.posts(cl.ENRICH_PATH) == []
    record = cl.load(workspace)[key]
    assert record["skipped"] == "target_site_location_mismatch"
    assert record["lookups"] == [] and record["recipient"]["choice"] == "none"
    before = stored(workspace)
    assert lookup(workspace, api, person_search=cl.PERSON_SEARCH_DECISION)["calls"]["made"] == 0
    assert stored(workspace) == before


def qualification_site(key="site-one", street="1 Example Road", city="Fixture City", pages=()):
    return {"site_key": key, "address": {"street": street, "city": city, "state": "TX"},
            "task_input": {"site_name": "Synthetic Works"}, "operator": "Synthetic Operator 1",
            "operator_domains": ["operator-1.example"], "operator_domain": "operator-1.example",
            "person": {}, "published_person_email": False, "kept_pages": list(pages)}


def replay_search(*people, legacy=False):
    found = [cl.candidate(person, ["operator-1.example"])[0] for person in people]
    if legacy:
        found[0].pop("location_fields")
    observation = {"candidate": found[0]}
    if not legacy:
        observation["candidates"] = found
    return lambda *args: ("c" * 64, {"state": "answered", "observation": observation})


def test_shared_company_search_is_requalified_for_each_target_site_without_calls():
    person = searched(OTHER_PERSON)
    calls = replay_search(person)
    first, reason = cl.target(qualification_site(), calls)
    assert reason is None
    assert first["site_responsibility"]["status"] == "unknown"  # Even a matching city proves no authority.
    assert first["site_responsibility"]["route"] == "corporate_referral"
    second, reason = cl.target(qualification_site("site-two", city="Other Fixture City"), calls)
    assert second is None and reason == "target_site_location_mismatch"


@pytest.mark.parametrize("legacy_title", ["Plant Manager", "Operations Audit Director", "Vice President Strategy"])
def test_legacy_shared_search_is_requalified_without_rebuying_the_search(tmp_path, legacy_title, monkeypatch):
    person, reason = cl.target(qualification_site(city="Other Fixture City"),
                               replay_search(searched(OTHER_PERSON), legacy=True))
    assert person is None and reason == "target_site_location_mismatch"
    workspace, (key,) = site_screen_out(tmp_path, [nobody(1)])
    api = FakeFullEnrich()
    api.people["operator-1.example"] = [searched(OTHER_PERSON, title=legacy_title)]
    api.emails[("Jordan", "Fixture", "operator-1.example")] = (FOUND, "DELIVERABLE", profile(OTHER_PERSON))
    site = cl.lookup_sites(workspace, workspace.states()[0], today=TODAY)[0][0]
    book = cl.Book(workspace)
    bounds = cl.limits(None, LOOKUP_OWNER, "5", 10)
    book.create_pin(bounds)
    calls = cl.Calls(book, "apply", client=cl.FullEnrichClient(FULLENRICH_KEY, transport=api), bounds=bounds)
    with monkeypatch.context() as old_rules:
        old_rules.setattr(cl, "listed_title", lambda title: True)
        calls("search", [site["operator_domain"], list(cl.LEGACY_TITLES)], key,
              lambda _: {"current_company_domains": [{"value": site["operator_domain"], "exact_match": True}],
                         "current_position_titles": [{"value": title} for title in cl.LEGACY_TITLES],
                         "limit": cl.SEARCH_LIMIT, "offset": 0}, cl.seal_search(site))
    report = lookup(workspace, api, person_search=cl.PERSON_SEARCH_DECISION, max_credits="5", max_calls=10)
    assert len(api.posts(cl.SEARCH_PATH)) == 1 and report["calls"]["known"] == 1
    assert len(api.posts(cl.ENRICH_PATH)) == (1 if legacy_title == "Plant Manager" else 0)


@pytest.mark.parametrize("operator_prefix", ["", "Synthetic Operator 1's "])
def test_specific_role_and_site_quote_can_qualify_a_person_with_a_different_home_location(operator_prefix):
    quote = f"Jordan Fixture is Plant Manager at {operator_prefix}1 Example Road, Fixture City, TX for Synthetic Operator 1."
    site = qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)])
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["status"] == "verified"
    assert chosen["site_responsibility"]["proof"]["quote"] == quote
    assert chosen["proof"]["current_employment"]["company_domain"] == "operator-1.example"


@pytest.mark.parametrize("quote", [
    "Jordan Fixture is Plant Manager of Synthetic Operator 1 and visited 1 Example Road, Fixture City, TX.",
    "Jordan Fixture is Plant Manager of Synthetic Operator 1. The plant is at 1 Example Road, Fixture City, TX.",
    "Jordan Fixture is Plant Manager at 2 Example Road, Fixture City, TX for Synthetic Operator 1.",
    "Jordan Fixture is Plant Manager at 2 Example Road and Avery Placeholder is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1.",
    "At 1 Example Road, Fixture City, TX, Jordan Fixture is Plant Manager of the other Synthetic Operator 1 facility.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Other Fixture City, TX for Synthetic Operator 1.",
    "Jordan Fixture is Plant Manager of Synthetic Operator 1 whose headquarters are at 1 Example Road, Fixture City, TX.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, OH with customers in Fixture City, TX for Synthetic Operator 1.",
    "Is Jordan Fixture the Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1?",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX facility of Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX and works for Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but he is employed by Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but now works for Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX and he currently works for Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but He currently works for Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1 but now works at Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX and is employed at Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1 but recently joined Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX and moved to Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1 but left for Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but left Synthetic Operator 1 for Rival Fixture Corporation.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but resigned from Synthetic Operator 1.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but resigned to join Rival Works.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but left for Acme.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but left for acme.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but resigned.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but resigns.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but resigning.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but quit.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but quits.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but quitting.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX but no longer works for Synthetic Operator 1.",
])
def test_company_title_city_and_visits_do_not_prove_responsibility_at_this_plant(quote):
    text = quote + " Synthetic Operator 1 operates the target plant."
    site = qualification_site(pages=[("https://operator-1.example/news", text, "d" * 64)])
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    assert cl.target(site, replay_search(person)) == (None, "target_site_location_mismatch")


@pytest.mark.parametrize("transition", [
    "Jordan Fixture resigned.", "Jordan Fixture now works for Rival Works.", "Jordan Fixture retired.",
    "Jordan Fixture was fired.", "Jordan Fixture was dismissed.", "Jordan Fixture was terminated.",
    "Jordan Fixture was laid off.", "JORDAN FIXTURE retired.",
])
def test_later_same_person_departure_invalidates_page_scope_proof(transition):
    text = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1. "
            + transition)
    site = qualification_site(pages=[("https://operator-1.example/team", text, "d" * 64)])
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    assert cl.target(site, replay_search(person)) == (None, "target_site_location_mismatch")


@pytest.mark.parametrize("transition", ["Jordan Fixture retired.", "Jordan Fixture was terminated."])
@pytest.mark.parametrize("reverse", [False, True])
def test_departure_on_another_retained_page_precedes_all_scope_proofs(transition, reverse):
    quote = "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1."
    pages = [("https://operator-1.example/team", quote, "d" * 64),
             ("https://operator-1.example/update", transition, "e" * 64)]
    person = cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0]
    scope = cl.site_responsibility(qualification_site(pages=list(reversed(pages)) if reverse else pages), person)
    assert scope["route"] == "hold" and scope["proof"] is None


@pytest.mark.parametrize("transition", ["He retired.", "She was terminated.", "They resigned.",
                                       "His employment was terminated.", "Her position was terminated."])
def test_departure_pronoun_linked_to_named_role_invalidates_scope(transition):
    quote = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1. "
             + transition)
    scope = cl.site_responsibility(qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)]),
                                   cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0])
    assert scope["route"] == "hold" and scope["proof"] is None


def test_departure_pronoun_after_a_different_named_person_preserves_scope():
    quote = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1. "
             "Avery Placeholder is Operations Director for Rival Works. She retired.")
    scope = cl.site_responsibility(qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)]),
                                   cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0])
    assert scope["route"] == "site_contact"


@pytest.mark.parametrize("transition", [
    "Jordan Fixture worked for Synthetic Operator 1 from 2018 to 2020.",
    "Jordan Fixture worked for Synthetic Operator 1 until 2020.",
    "Jordan Fixture was employed at Synthetic Operator 1 until January 2020.",
    "Jordan Fixture served for Synthetic Operator 1 from 2018 through 2020.",
    "Jordan Fixture served as Plant Manager at Synthetic Operator 1 until 2020.",
    "Jordan Fixture worked as Plant Manager for Synthetic Operator 1 from 2018 to 2020.",
    "In 2025, Jordan Fixture retired.", "Update: Jordan Fixture retired.",
    "In 2025, he retired.", "Update: she was terminated.",
    "Jordan Fixture won an award. He retired.",
    "Jordan Fixture is Plant Manager at Rival Works.",
    "Jordan Fixture is Plant Manager for Rival Works.",
    "He is Plant Manager at Rival Works.",
    "Jordan Fixture is now Plant Manager at Rival Works.",
    "Jordan Fixture is currently Plant Manager at Rival Works.",
    "Jordan Fixture is the new Plant Manager at Rival Works.",
    "Jordan Fixture is Plant Manager at 2 Other Road, Other Fixture City, OH for Rival Works.",
    "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Rival Works.",
    "He is Plant Manager at 2 Other Road, Other Fixture City, OH for Rival Works.",
    "Jordan Fixture is Plant Manager for Rival Works at 2 Other Road, Other Fixture City, OH.",
    "Jordan Fixture is Plant Manager of Rival Works at 2 Other Road, Other Fixture City, OH.",
    "Jordan Fixture is Plant Manager at Rival Works at 2 Other Road, Other Fixture City, OH.",
    "He is Plant Manager for Rival Works at 2 Other Road, Other Fixture City, OH.",
    "Jordan Fixture is Plant Manager at Rival Works, Other Fixture City, OH.",
    "Jordan Fixture worked for Synthetic Operator 1; now works for Rival Works.",
    "Jordan Fixture worked for Synthetic Operator 1; he now works for Rival Works.",
    "Jordan Fixture worked for Synthetic Operator 1, now works for Rival Works.",
    "Jordan Fixture worked for Synthetic Operator 1, then joined Rival Works.",
    "Jordan Fixture worked for Synthetic Operator 1; subsequently joined Rival Works.",
    "Jordan Fixture worked for Synthetic Operator 1, later joined Rival Works.",
    "Jordan Fixture, the Plant Manager, retired.",
    "Jordan Fixture, the PLANT MANAGER, retired.",
    "Jordan Fixture stepped down as Plant Manager in 2025.",
    "He stepped down as Plant Manager.",
    "Jordan Fixture left the company in 2025.",
    "Jordan Fixture departed the company.",
    "Jordan Fixture left his employer.",
    "Jordan Fixture said he resigned.",
    "Jordan Fixture and Avery Placeholder resigned.",
    "Jordan Fixture and AVERY PLACEHOLDER resigned.",
    "Jordan Fixture and Avery Placeholder and Taylor Fixture retired.",
    "Jordan Fixture is no longer Plant Manager.",
    "He is no longer Plant Manager.",
    "Jordan Fixture, along with Avery Placeholder, resigned.",
    "Jordan Fixture as well as Avery Placeholder retired.",
    "Jordan Fixture is no longer Plant Manager at Synthetic Operator 1.",
])
def test_explicit_ended_employment_and_same_subject_updates_override_old_role(transition):
    quote = "Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1."
    pages = [("https://operator-1.example/team", quote, "d" * 64),
             ("https://operator-1.example/update", quote + " " + transition, "e" * 64)]
    scope = cl.site_responsibility(qualification_site(pages=pages),
                                   cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0])
    assert scope["route"] == "hold" and scope["proof"] is None


@pytest.mark.parametrize("affirmation", [
    "Jordan Fixture works for the Synthetic Operator 1.",
    "Jordan Fixture serves as Plant Manager at the Synthetic Operator 1.",
    "Jordan Fixture is Plant Manager for Synthetic Operator 1 at 1 Example Road, Fixture City, TX.",
    "Jordan Fixture is Plant Manager of Synthetic Operator 1 at 1 Example Road, Fixture City, TX.",
    "Jordan Fixture is Plant Manager at Synthetic Operator 1 at 1 Example Road, Fixture City, TX.",
    "Jordan Fixture is Plant Manager at Synthetic Operator 1, Fixture City, TX.",
    "Jordan Fixture is Plant Manager for 1 Example Road, Fixture City, TX.",
    "Jordan Fixture works for Synthetic Operator 1; Avery Placeholder works for Rival Works.",
    "Jordan Fixture retired the old assembly line.",
    "Jordan Fixture fired a contractor.",
    "Jordan Fixture dismissed an employee.",
    "Jordan Fixture terminated a contractor.",
    "Jordan Fixture resigned from Rival Works in 2015.",
    "Jordan Fixture retired from Rival Works in 2015.",
    "Jordan Fixture left Rival Works in 2015.",
    "Jordan Fixture resigned as Plant Manager at Rival Works in 2015.",
    "Jordan Fixture retired as Plant Manager for Rival Works in 2015.",
    "Jordan Fixture worked for Rival Works, then joined Synthetic Operator 1.",
    "Jordan Fixture joined the safety meeting.",
    "Jordan Fixture moved to Fixture City.",
    "Jordan Fixture worked for Rival Works from 2010 to 2015.",
    "Jordan Fixture's predecessor retired.",
    "Jordan Fixture's assistant resigned.",
    "His assistant resigned.",
    "Jordan Fixture stepped down from the platform.",
    "Jordan Fixture said the manager resigned.",
    "Jordan Fixture confirmed an employee was terminated.",
    "Jordan Fixture left the company meeting.",
    "Jordan Fixture has not resigned.",
    "Jordan Fixture hasn't resigned.",
    "Jordan Fixture never retired.",
    "Jordan Fixture will retire next year.",
    "Did Jordan Fixture retire?",
    "Jordan Fixture plans to retire next year.",
    "Jordan Fixture is expected to resign next year.",
    "Jordan Fixture is scheduled to retire next year.",
    "Jordan Fixture intends to resign next year.",
    "Jordan Fixture no longer works for Rival Works.",
    "Jordan Fixture is no longer Plant Manager at Rival Works.",
])
def test_affirming_target_employment_with_a_determiner_preserves_scope(affirmation):
    quote = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1. "
             + affirmation)
    scope = cl.site_responsibility(qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)]),
                                   cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0])
    assert scope["route"] == "site_contact"


def test_unqualified_multi_predicate_scope_uses_corporate_referral_without_site_authority():
    quote = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1. "
             "Jordan Fixture worked for Synthetic Operator 1 before joining Rival Works.")
    person = cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0]
    person["location_fields"] = {"city": "Fixture City", "region": "Texas", "country": "United States"}
    scope = cl.site_responsibility(qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)]), person)
    assert scope == {"site_key": qualification_site()["site_key"], "status": "unknown", "route": "corporate_referral",
                     "reason": "target_site_responsibility_unproven", "proof": None}


def test_another_subjects_multi_job_sentence_does_not_downgrade_supported_scope():
    quote = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1. "
             "Avery Placeholder worked for Rival Works before joining Other Works and thanked Jordan Fixture.")
    scope = cl.site_responsibility(qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)]),
                                   cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0])
    assert scope["route"] == "site_contact" and scope["proof"] is not None


def test_object_name_mention_does_not_link_another_subjects_departure():
    quote = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1. "
             "Avery Placeholder thanked Jordan Fixture. She retired.")
    scope = cl.site_responsibility(qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)]),
                                   cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0])
    assert scope["route"] == "site_contact"


@pytest.mark.parametrize("modifier", ["now", "currently", "the new"])
def test_current_role_modifiers_preserve_explicit_target_scope(modifier):
    quote = (f"Jordan Fixture is {modifier} Plant Manager at 1 Example Road, Fixture City, TX "
             "for Synthetic Operator 1.")
    scope = cl.site_responsibility(qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)]),
                                   cl.candidate(searched(OTHER_PERSON), ["operator-1.example"])[0])
    assert scope["route"] == "site_contact"


@pytest.mark.parametrize("name", ["Jordan Fixture", "Jordan Quit", "Jordan Retired"])
@pytest.mark.parametrize("other", ["Avery Placeholder", "AVERY PLACEHOLDER"])
@pytest.mark.parametrize("join", [". ", " while "])
def test_another_person_departure_does_not_erase_retained_scope(name, other, join):
    text = (f"{name} is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1"
            + join + f"{other} resigned.")
    site = qualification_site(pages=[("https://operator-1.example/team", text, "d" * 64)])
    person = searched(name)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["route"] == "site_contact"


def test_later_site_proven_candidate_wins_over_a_company_referral():
    quote = "Avery Placeholder is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1."
    site = qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)])
    chosen, _ = cl.target(site, replay_search(searched(OTHER_PERSON), searched(PERSON)))
    assert chosen["name"] == PERSON and chosen["site_responsibility"]["route"] == "site_contact"


@pytest.mark.parametrize("field,name,prefix", [
    ("operator", "Synthetic Smith and Sons", "Synthetic Smith and Sons' "),
    ("operator", "Synthetic Smith & Sons", "Synthetic Smith and Sons' "),
    ("site_name", "Synthetic Works and Foundry", "Synthetic Works and Foundry's "),
])
def test_known_operator_and_facility_names_preserve_their_conjunctions(field, name, prefix):
    site = qualification_site()
    (site if field == "operator" else site["task_input"])[field] = name
    quote = f"Jordan Fixture is Plant Manager at {prefix}1 Example Road, Fixture City, TX for {site['operator']}."
    site["kept_pages"] = [("https://operator-1.example/team", quote, "d" * 64)]
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["route"] == "site_contact"


def test_news_after_an_explicit_site_role_does_not_erase_that_role():
    quote = ("Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1, "
             "and said the plant visited another facility this spring.")
    site = qualification_site(pages=[("https://operator-1.example/news", quote, "d" * 64)])
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["route"] == "site_contact"


def test_a_street_without_target_city_does_not_establish_site_responsibility():
    quote = "Jordan Fixture is Plant Manager at 1 Example Road for Synthetic Operator 1."
    site = qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)])
    site["address"].pop("city")
    chosen, _ = cl.target(site, replay_search(searched(OTHER_PERSON)))
    assert chosen["site_responsibility"]["status"] == "unknown"
    assert chosen["site_responsibility"]["route"] == "corporate_referral"


@pytest.mark.parametrize("provider_title,quoted_title", [
    ("Senior Plant Manager", "Plant Manager"),
    ("Plant Manager", "Senior Plant Manager"),
    ("Plant Manager", "Manager of the Plant"),
])
def test_site_role_proof_uses_the_existing_normalized_title_semantics(provider_title, quoted_title):
    quote = f"Jordan Fixture is {quoted_title} at 1 Example Road, Fixture City, TX for Synthetic Operator 1."
    site = qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)])
    person = searched(OTHER_PERSON, title=provider_title)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["route"] == "site_contact"


@pytest.mark.parametrize("employment", ["works for Synthetic Operator 1", "recently joined Synthetic Operator 1",
                                         "Avery Placeholder recently joined Rival Fixture Corporation",
                                         "Avery Placeholder left for Rival Fixture Corporation",
                                         "left the conference early", "departed the conference early",
                                         "left for the conference early", "left for conference early"])
def test_conjunction_can_restate_the_same_employer_without_losing_site_role(employment):
    quote = f"Jordan Fixture is Plant Manager at 1 Example Road, Fixture City, TX for Synthetic Operator 1 and {employment}."
    site = qualification_site(pages=[("https://operator-1.example/team", quote, "d" * 64)])
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["route"] == "site_contact"


@pytest.mark.parametrize("location,expected", [
    ("Bloomington in the United States", False),
    ("Bloomington IN", True),
    ("Bloomington Indiana", True),
    ("Bloomington in the United States and announced news in Bloomington IN", False),
])
def test_state_abbreviation_requires_case_in_the_direct_role_clause(location, expected):
    site = qualification_site(city="Bloomington")
    site["address"]["state"] = "IN"
    quote = f"Jordan Fixture is Plant Manager at 1 Example Road, {location} for Synthetic Operator 1."
    assert cl.site_role_quote(site, {"name": "Jordan Fixture", "title": "Plant Manager"}, quote) is expected


@pytest.mark.parametrize("target,quoted", [
    ("9 Mill Rd #4", "9 Mill Rd, Suite 4"),
    ("9 Mill Rd, Suite 4", "9 Mill Rd #4"),
    ("9 Mill Rd #4", "9 Mill Rd #4"),
    ("9 Mill Rd, Suite 4", "9 Mill Rd, Suite 4"),
    ("Suite 4, 9 Mill Rd", "9 Mill Rd, Suite 4"),
    ("9 Mill Rd, Suite 4", "Suite 4, 9 Mill Rd"),
    ("9 Mill Rd, Unit 4", "Unit 4, 9 Mill Rd"),
])
def test_site_role_preserves_units_before_or_after_the_street(target, quoted):
    quote = f"Jordan Fixture is Plant Manager at {quoted}, Fixture City, TX for Synthetic Operator 1."
    site = qualification_site(street=target, pages=[("https://operator-1.example/team", quote, "d" * 64)])
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["route"] == "site_contact"


@pytest.mark.parametrize("quoted", ["9 Mill Rd, Suite 5", "9 Mill Rd #5", "9 Mill Rd"])
def test_a_conflicting_or_missing_target_unit_grants_no_site_responsibility(quoted):
    quote = f"Jordan Fixture is Plant Manager at {quoted}, Fixture City, TX for Synthetic Operator 1."
    site = qualification_site(street="9 Mill Rd, Suite 4", pages=[("https://operator-1.example/team", quote, "d" * 64)])
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    assert cl.target(site, replay_search(person)) == (None, "target_site_location_mismatch")


def test_hash_in_a_retained_operator_name_is_not_an_address_unit():
    site = qualification_site()
    site["operator"] = "#1 Manufacturing"
    quote = "Jordan Fixture is Plant Manager at #１ Manufacturing's 1 Example Road, Fixture City, TX."
    site["kept_pages"] = [("https://operator-1.example/team", quote, "d" * 64)]
    person = searched(OTHER_PERSON)
    person["location"]["city"] = "Other Fixture City"
    chosen, reason = cl.target(site, replay_search(person))
    assert reason is None and chosen["site_responsibility"]["route"] == "site_contact"


def test_off_site_corporate_contact_is_explicitly_a_referral_not_local_authority():
    person = searched(OTHER_PERSON, title="Operations Director")
    person["location"]["city"] = "Other Fixture City"
    chosen, _ = cl.target(qualification_site(), replay_search(person))
    assert chosen["site_responsibility"] == {
        "site_key": "site-one", "status": "unknown", "route": "corporate_referral",
        "reason": "target_site_location_mismatch", "proof": None}


def test_enrichment_profile_checks_names_and_employment_history():
    person = profile(OTHER_PERSON)
    person["employment"]["all"] = [{**person["employment"].pop("current"), "is_current": False,
                                     "end_at": "2024-01-31T00:00:00Z"}]
    assert cl.profile_check(person, OTHER_PERSON, ["operator-1.example"]) == "contact_lookup_person_not_current_at_operator"
    person["employment"]["current"] = {}
    assert cl.profile_check(person, OTHER_PERSON, ["operator-1.example"]) == "contact_lookup_person_not_current_at_operator"
    person = {"first_name": "Avery", "last_name": "Placeholder"}
    assert cl.profile_check(person, OTHER_PERSON, ["operator-1.example"]) == "contact_lookup_person_mismatch"


@pytest.mark.parametrize("value", [1, 0, "true", [], {}])
def test_current_employment_indicators_must_be_booleans_when_present(value):
    person = searched(OTHER_PERSON, is_current=value)
    assert cl.candidate(person, ["operator-1.example"])[0] is None
    assert cl.profile_check(person, OTHER_PERSON, ["operator-1.example"]) == "contact_lookup_person_not_current_at_operator"


def test_quoted_person_source_must_still_be_current_at_lookup_time(tmp_path):
    workspace, _ = site_screen_out(tmp_path, [quoted(1, person_date="2025-06-01")])
    api = FakeFullEnrich()
    result = lookup(workspace, api, today=TODAY.replace(year=2027))
    assert result["skipped"] == {"person_not_current": 1} and api.calls == []


def test_quoted_person_title_must_be_supported_by_the_verified_quote(tmp_path):
    workspace, _ = site_screen_out(tmp_path, [quoted(1, person_title="Automation Manager")])
    api = FakeFullEnrich()
    result = lookup(workspace, api)
    assert result["skipped"] == {"person_role_unproven": 1} and api.calls == []


@pytest.mark.parametrize("on_worker", [False, True])
def test_balance_is_one_get_with_counts_only_output(tmp_path, capsys, on_worker):
    key_file = tmp_path / "synthetic-key.env"
    key_file.write_text("FULLENRICH_API_KEY=" + FULLENRICH_KEY)
    requests = []

    def transport(method, path, **kwargs):
        requests.append((method, path))
        return 200, json.dumps({"balance": 49.25, "ignored": FULLENRICH_KEY}).encode()

    args = ["balance"] if on_worker else ["balance", "--key-file", str(key_file)]
    environment = {ss.WORKER_FLAG: "0", cl.KEY_ENV: FULLENRICH_KEY} if on_worker else {}
    report = operator.main(args, transport=transport, environ=environment)
    printed = capsys.readouterr().out
    assert requests == [("GET", "/api/v2/account/credits")]
    assert report["credits_available"] == "49.25" and FULLENRICH_KEY not in printed


@pytest.mark.parametrize("value", [{"balance": True}, {"balance": -1}, {"balance": "Infinity"}, {}, []])
def test_balance_never_guesses_missing_or_invalid_credits(value):
    client = cl.FullEnrichClient(FULLENRICH_KEY, transport=lambda *args, **kwargs: (200, json.dumps(value).encode()))
    with pytest.raises(cl.LookupFailure, match="^contact_lookup_balance_invalid$"):
        client.balance()


@pytest.mark.parametrize("state, credits", [("UNKNOWN", None), ("UNKNOWN", 0), ("CANCELED", None)])
def test_unknown_or_unproven_unbilled_results_never_allow_another_paid_start(tmp_path, state, credits):
    workspace, _ = site_screen_out(tmp_path, [quoted(1)])
    api = FakeFullEnrich()
    # The first result read must bind the real job ID while leaving billing uncertain.
    def uncertain_result(method, path, **kwargs):
        if method == "GET":
            return 200, json.dumps({"id": path.rsplit("/", 1)[-1], "status": state,
                                    "cost": {} if credits is None else {"credits": credits}, "data": []}).encode()
        return api(method, path, **kwargs)
    client = cl.FullEnrichClient(FULLENRICH_KEY, transport=uncertain_result)
    options = {"client": client, "owner_reference": LOOKUP_OWNER, "max_credits": "1", "max_calls": 1,
               "apply": True, "wait_seconds": 0}
    first, again = cl.lookup(workspace, **options), cl.lookup(workspace, **options)
    assert first["credits"]["committed"] == again["credits"]["committed"] == "1"
    assert first["pending_results"] == again["pending_results"] == 1
    assert len(api.posts(cl.ENRICH_PATH)) == 1 and again["calls"]["made"] == 0


# --- the owner command ----------------------------------------------------------------------------
def test_the_command_prints_counts_only_keeps_the_key_private_and_writes_private_files(tmp_path, capsys):
    workspace, _ = site_screen_out(tmp_path, [quoted(1), nobody(2)])
    capsys.readouterr()
    api = FakeFullEnrich()
    api.emails[("Avery", "Placeholder", "operator-1.example")] = (ADDRESS, "DELIVERABLE", profile(PERSON))
    api.people["operator-2.example"] = [searched(OTHER_PERSON, domain="operator-2.example")]
    api.emails[("Jordan", "Fixture", "operator-2.example")] = (
        "jordan.fixture@operator-2.example", "DELIVERABLE", profile(OTHER_PERSON, domain="operator-2.example"))
    key_file = tmp_path / "private.env"
    key_file.write_text(f"# owner-only\nOTHER_KEY=unrelated\nexport FULLENRICH_API_KEY=\"{FULLENRICH_KEY}\"\n")
    spend = ["lookup", "--out", str(workspace.root), "--owner-reference", LOOKUP_OWNER, "--max-credits", "50",
             "--max-calls", "20", "--key-file", str(key_file), "--wait-seconds", "0",
             "--person-search", cl.PERSON_SEARCH_DECISION]
    assert operator.main(spend, transport=api)["state"] == "planned" and api.calls == []
    assert operator.main([*spend, "--apply"], transport=api)["state"] == "complete"
    report = operator.main(["summary", "--out", str(workspace.root)])
    assert report["recipients"]["looked_up_person_email"] == 2 and report["people"] == {
        "quoted_person": 1, "provider_sourced": 1, "corroborated": 0, "uncorroborated": 1}
    assert report["enrichments"]["usable"] == 2 and report["credits"]["used"] == "2.25"
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 3 and all(json.loads(line) for line in output.splitlines())
    assert FULLENRICH_KEY not in output and "@" not in output
    assert not any(value in output for value in (*SITE_STRINGS, OTHER_PERSON, "Fixture", "Placeholder"))
    assert {call["headers"]["Authorization"] for call in api.calls} == {"Bearer " + FULLENRICH_KEY}
    assert FULLENRICH_KEY.encode() not in stored(workspace) and "withheld" in repr(cl.FullEnrichClient(FULLENRICH_KEY))
    folder = workspace.root / cl.FOLDER
    for path in (folder, *folder.rglob("*")):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600), path
    key_file.write_text("# FULLENRICH_API_KEY=commented-out\n")
    with pytest.raises(ss.ScreenError, match="^contact_lookup_api_key_missing$"):
        operator.main(spend, transport=api)
    with pytest.raises(ss.ScreenError, match="^contact_lookup_key_file_unreadable$"):
        operator.main([*spend[:-6], "--key-file", str(tmp_path / "missing.env"), *spend[-4:]], transport=api)
    with pytest.raises(ss.ScreenError, match="^contact_lookup_worker_needs_paid_admission$"):
        operator.main(spend, transport=api, environ={ss.WORKER_FLAG: "1"})


def test_contact_lookup_imports_only_the_standard_library_and_the_site_screen():
    tree = ast.parse((ROOT / "tools/daily_research/contact_lookup.py").read_text())
    modules = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert modules - {"tools.daily_research"} <= set(sys.stdlib_module_names)
    assert {(node.module, alias.name) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for alias in node.names if node.module == "tools.daily_research"} == {("tools.daily_research", "site_screen")}


def test_every_person_and_host_in_these_tests_is_synthetic():
    source = Path(__file__).read_text()
    hosts = set(re.findall(r"https://([A-Za-z0-9.-]+)", source)) | set(re.findall(r"[\w.+-]@([A-Za-z0-9.-]+\.[a-z]+)", source))
    refused = {"www.linkedin.com", "gmail.com", "operator-1.example.net", "records.synthetic.gov"}  # Refusal fixtures.
    assert hosts and all(host.endswith(".example") or host in refused for host in hosts), hosts
    names = set(re.findall(r'searched\("([A-Z][a-z]+ [A-Z][a-z]+)"', source)) | set(fixture.PEOPLE)
    assert names and all(name.split()[-1] in {"Placeholder", "Fixture"} for name in names), names
