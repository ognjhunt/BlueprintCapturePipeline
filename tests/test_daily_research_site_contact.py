"""Hermetic contact stage of the site screen line: fake Parallel Task API and fake pages; synthetic people only."""
import importlib.util
import re
from pathlib import Path

import pytest

from tests import daily_research_site_screen_fixture as fixture
from tests.daily_research_site_screen_fixture import (
    KEY,
    OWNER,
    OTHER_PERSON,
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
from tools.daily_research import site_screen as ss

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_screen_operator", ROOT / "tools/daily_research/operators/site-screen.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
NO_LINKEDIN = ("Do not use LinkedIn as the source; a LinkedIn profile may only point you to a confirming press release, "
               "news story, job post or company page.")
LONG = "For plant tours, supplier visits, quality audits and shipping questions, write to {}"


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """pytest's tmp_path is on storage the out-dir guard refuses, and a shell may set the worker flag."""
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", (), raising=False)
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)


def screened_sites(tmp_path, count, *, unproven=()):
    """Screen ``count`` web-found sites. The site numbers in ``unproven`` keep their task page unpublished, so
    they stay screened."""
    records = [inventory_record(number) for number in range(1, count + 1)]
    answers = [screen_answers(number) for number in range(1, count + 1)]
    pages = {url: text for number, answer in enumerate(answers, 1) for url, text in pages_for(
        answer, [stem for stem in ss.SCREEN_PROOFS if not (number in unproven and stem == "target_task")]).items()}
    workspace, provider, _, _ = screen(tmp_path, records, answers, pages)
    return workspace, provider, [ss.from_inventory(record)["site_key"] for record in records]


def contacted(tmp_path, answers, pages, *, basis=None):
    """Screen one site per contact answer, then run, collect and verify the contact stage with the fakes."""
    workspace, provider, keys = screened_sites(tmp_path, len(answers))
    for key, content in zip(keys, answers):
        provider.contacts[key] = {"content": content, "basis": (basis or {}).get(key, [])}
    client = ss.TaskClient(KEY, transport=provider)
    result = ss.contact(workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    ss.collect(workspace, client=client, wait_seconds=0)
    reader = FakePages(pages)
    ss.verify(workspace, reader=reader, today=TODAY)
    records = {record["site_key"]: record for record in workspace.records("contact")}
    return workspace, reader, [records[key] for key in keys if key in records], result


def site_key(number):
    return ss.from_inventory(inventory_record(number))["site_key"]


def test_the_contact_form_is_versioned_and_keeps_linkedin_out_of_its_sources():
    schema = ss.CONTACT_SCHEMA
    assert set(schema["properties"]) == {
        "decision_role", "person_name", "person_title", "person_url", "person_quote", "person_date", "email",
        "email_url", "email_quote", "channel_type", "channel_url", "notes"}
    assert schema["required"] == list(schema["properties"]) and schema["additionalProperties"] is False
    for field in ("person_name", "person_url", "email", "email_url"):
        assert NO_LINKEDIN in schema["properties"][field]["description"]
    assert "Never guess an address or derive one from a name pattern" in schema["properties"]["email"]["description"]
    site = {"site_key": "b" * 64, "task_input": {"operator": "Synthetic Operator 1"}}
    body = ss.create_body("contact", site, "core")
    assert body["metadata"] == {"site_key": "b" * 64, "form": "blueprint.site-contact.v1"}
    assert body["task_spec"] == {"output_schema": {"type": "json", "json_schema": schema}}
    assert ss.FORMS["contact"]["sha256"] != ss.FORMS["screen"]["sha256"]


def test_contact_runs_only_for_outreach_ready_sites_in_screen_order(tmp_path):
    workspace, provider, keys = screened_sites(tmp_path, 3, unproven=(2,))
    tiers = {record["site_key"]: record["tier"] for record in workspace.records("screen")}
    assert [tiers[key] for key in keys] == ["outreach_ready", "screened", "outreach_ready"]
    result = ss.contact(workspace, client=ss.TaskClient(KEY, transport=provider), owner_reference=OWNER, ceiling_usd="1", max_runs=10,
                        apply=True)
    assert (result["command"], result["form"], result["sites"], result["created"]) == ("contact", ss.CONTACT, 2, 2)
    bodies = provider.creates(ss.CONTACT)
    assert [body["metadata"]["site_key"] for body in bodies] == [keys[0], keys[2]]
    assert bodies[0]["input"] == {"operator": "Synthetic Operator 1", "site_name": "Synthetic Works 1",
                                  "site_address": "1 Example Road, Fixture City, TX", "location": "Fixture City, TX",
                                  "target_task": "CNC machine tending", "website": "https://operator-1.example"}


def test_contact_runs_are_idempotent_and_share_the_screen_ceiling(tmp_path):
    workspace, provider, keys = screened_sites(tmp_path, 3)  # The screen committed $0.075 under a $5 pin.
    client = ss.TaskClient(KEY, transport=provider)
    options = {"client": client, "owner_reference": OWNER}
    dry = ss.contact(workspace, **options, ceiling_usd="0.125", max_runs=10)
    assert (dry["state"], dry["would_create"], dry["stop"]) == ("stopped", 2, "site_screen_spend_ceiling_reached")
    assert provider.creates(ss.CONTACT) == []
    first = ss.contact(workspace, **options, ceiling_usd="0.125", max_runs=10, apply=True)
    assert (first["created"], first["committed_usd"], first["stop"]) == (2, "0.125", "site_screen_spend_ceiling_reached")
    assert (first["runs"], first["stage_runs"], first["stage_committed_usd"]) == (5, 2, "0.050")
    capped = ss.contact(workspace, **options, ceiling_usd="1", max_runs=5, apply=True)
    assert (capped["created"], capped["already_created"], capped["stop"]) == (0, 2, "site_screen_max_runs_reached")
    rest = ss.contact(workspace, **options, ceiling_usd="1", max_runs=10, apply=True)
    assert (rest["created"], rest["already_created"], rest["state"]) == (1, 2, "complete")
    again = ss.contact(workspace, **options, ceiling_usd="1", max_runs=10, apply=True)
    assert (again["created"], again["already_created"]) == (0, 3)
    assert [body["metadata"]["site_key"] for body in provider.creates(ss.CONTACT)] == keys
    assert [event["event"] for event in workspace.ledger("contact").events()].count("intent") == 3
    fresh, other_provider, _ = screened_sites(tmp_path / "fresh", 1)
    below = ss.contact(fresh, client=ss.TaskClient(KEY, transport=other_provider), owner_reference=OWNER,
                       ceiling_usd="0.04", max_runs=10, apply=True)
    assert (below["created"], below["stop"]) == (0, "site_screen_spend_ceiling_reached")
    assert other_provider.creates(ss.CONTACT) == []


def test_a_verified_person_email_is_the_preferred_recipient(tmp_path):
    answers = contact_answers(1)
    _, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers))
    assert record["schema_version"] == ss.CONTACT and record["decision_role"] == "Plant manager"
    assert record["person"] == {"verified": True, "level": "verified_on_page", "name": PERSON, "title": "Plant Manager",
                                "url": "https://operator-1.example/team", "date": "2026-06-01", "current": True}
    assert record["email"] == {"verified": True, "level": "verified_on_page", "discarded": False,
                               "address": "plant.lead@operator-1.example", "url": "https://operator-1.example/team"}
    assert record["recipient"] == {"kind": "person_email", "rank": 0, "address": "plant.lead@operator-1.example"}
    # A title alone never proves remit, so the remit stays an open question.
    assert [question["check"] for question in record["open_questions"]] == ["decision_remit"]


def test_an_email_not_on_its_page_is_discarded_and_recorded_unverified(tmp_path):
    answers = contact_answers(1)
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers, ["person"]))
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True}
    assert record["recipient"] == {"kind": "none", "rank": 3, "address": None}
    assert [question["check"] for question in record["open_questions"]] == ["decision_remit", "recipient"]
    stored = workspace.path("contact", "records", record["site_key"]).read_text()
    assert answers["email"] not in stored  # Discarded: only the raw provider response keeps it.


def test_a_pattern_guessed_email_is_refused(tmp_path):
    guessed = "avery.placeholder@operator-1.example"  # The person's name in a first.last pattern.
    answers = contact_answers(1, email=guessed, email_quote=f"Write to {guessed} for plant questions.")
    pages = {answers["person_url"]: f"{answers['person_quote']} Write to info@operator-1.example for plant questions."}
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages)
    assert record["person"]["verified"] is True
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True}
    assert record["recipient"]["kind"] == "none"
    assert guessed not in workspace.path("contact", "records", record["site_key"]).read_text()


@pytest.mark.parametrize("published, claimed, near", [
    ("sales@operator-1.example.net", "sales@operator-1.example", True),  # The page's address is longer.
    ("sales@operator-1.example", "sales@operator-1.exampl", True),  # The claim cuts the domain.
    ("presales@operator-1.example", "sales@operator-1.example", False),  # The page's local part is longer.
    ("sales@operator-1.example", "s.ales@operator-1.example", False),  # A changed local part.
    ("sales [at] operator-1.example", "sales@operator-1.example", False),  # Obfuscated: not published verbatim.
])
def test_only_an_address_published_verbatim_is_accepted(tmp_path, published, claimed, near):
    quote = LONG.format(claimed)
    answers = contact_answers(1, email=claimed, email_quote=quote, email_url="https://operator-1.example/visit",
                              channel_type="general_inbox")
    page = "Visitors. " + LONG.format(published)
    # The near-exact quote rule alone would accept some of these quotes; the address check still refuses them.
    assert ss.contains(ss.normalize(quote), ss.normalize(page)) is near
    _, _, (record,), _ = contacted(tmp_path, [answers], {**pages_for(answers, ["person"]), answers["email_url"]: page})
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True}
    assert record["recipient"]["kind"] == "none"


def test_the_email_quote_must_contain_the_exact_address(tmp_path):
    answers = contact_answers(1, email_quote="Write to the plant team for plant questions.")
    pages = {answers["person_url"]: f"{answers['person_quote']} {answers['email_quote']} plant.lead@operator-1.example"}
    _, _, (record,), _ = contacted(tmp_path, [answers], pages)
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True,
                               "reason": "site_screen_quote_lacks_address"}


def test_a_citation_excerpt_counts_when_the_page_blocks_our_reader_and_holds_the_exact_address(tmp_path):
    answers = contact_answers(1, email_url="https://operator-1.example/press")
    citation = {"url": answers["email_url"], "excerpts": [answers["email_quote"]]}
    basis = {site_key(1): [{"field": "email", "citations": [citation]}]}
    _, reader, (record,), _ = contacted(tmp_path / "held", [answers], pages_for(answers, ["person"]), basis=basis)
    assert record["email"]["level"] == "in_citation_excerpt" and record["recipient"]["kind"] == "person_email"
    assert answers["email_url"] in reader.requested
    other = {**citation, "excerpts": [answers["email_quote"].replace("plant.lead@", "plantlead@")]}
    _, _, (record,), _ = contacted(tmp_path / "other", [answers], pages_for(answers, ["person"]),
                                   basis={site_key(1): [{"field": "email", "citations": [other]}]})
    assert record["email"] == {"verified": False, "level": "unverified_page_unreachable", "discarded": True,
                               "reason": "source_http_failure"}


def test_the_person_quote_must_contain_the_name_and_the_page_must_show_it(tmp_path):
    nameless = contact_answers(1, person_quote="Our plant manager leads the machining plant.")
    renamed = contact_answers(2, person_name=OTHER_PERSON, person_quote=(
        f"{OTHER_PERSON} leads the machining plant, the twelve lathe cells, the inspection lab, shipping and "
        "receiving, tooling and the second shift as the plant manager."))
    pages = {nameless["person_url"]: f"{PERSON} is our plant manager. {nameless['person_quote']} "
                                     f"{nameless['email_quote']}",
             renamed["person_url"]: renamed["person_quote"].replace(OTHER_PERSON, "Jordan Fixtur") + " "
                                    + renamed["email_quote"]}
    assert ss.contains(ss.normalize(renamed["person_quote"]), ss.normalize(pages[renamed["person_url"]]))
    _, _, (first, second), _ = contacted(tmp_path, [nameless, renamed], pages)
    assert first["person"] == {"verified": False, "level": "unverified", "reason": "site_screen_quote_lacks_name"}
    assert second["person"] == {"verified": False, "level": "unverified"}  # The near-exact quote never vouches for a name.
    for record in (first, second):
        # A verified address alone is not a person's email: the recipient needs a verified person too.
        assert record["email"]["verified"] is True and record["recipient"]["kind"] == "none"
        assert [question["check"] for question in record["open_questions"]] == ["decision_maker", "recipient"]


def test_the_recipient_follows_the_owner_preference_order(tmp_path):
    assert ss.RECIPIENT_PREFERENCE == ("person_email", "team_inbox", "general_inbox")
    assert ss.recipient("person_email", True, True) == "person_email"
    assert ss.recipient("team_inbox", True, False) == "team_inbox"
    assert ss.recipient("general_inbox", True, False) == "general_inbox"
    assert ss.recipient("person_email", True, False) == "none"
    assert {ss.recipient(channel, True, True) for channel in ("contact_form", "phone", "none", "fax")} == {"none"}
    assert {ss.recipient(channel, False, True) for channel in ss.RECIPIENT_PREFERENCE} == {"none"}
    nobody = {"person_name": "", "person_title": "", "person_url": "", "person_quote": "", "person_date": ""}
    team = contact_answers(1, **nobody, email="plant.team@operator-1.example",
                           email_quote="Write to plant.team@operator-1.example for plant questions.",
                           channel_type="Team inbox")
    general = contact_answers(2, **nobody, email="info@operator-2.example",
                              email_quote="General questions go to info@operator-2.example any day.",
                              channel_type="general_inbox")
    form = contact_answers(3, **nobody, email="", email_url="", email_quote="", channel_type="contact_form",
                           channel_url="https://operator-3.example/contact-us")
    person = contact_answers(4)
    answers = [team, general, form, person]
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    workspace, _, records, _ = contacted(tmp_path, answers, pages)
    assert [record["recipient"]["kind"] for record in records] == ["team_inbox", "general_inbox", "none", "person_email"]
    assert [record["recipient"]["kind"] for record in sorted(records, key=lambda r: r["recipient"]["rank"])] == [
        "person_email", "team_inbox", "general_inbox", "none"]
    assert records[2]["channel"] == {"type": "contact_form", "url": "https://operator-3.example/contact-us"}
    assert records[2]["email"] == {"verified": False, "level": "no_email", "discarded": False}
    report = ss.summary(workspace)["contact"]
    assert report["recipients"] == {"person_email": 1, "team_inbox": 1, "general_inbox": 1, "none": 1}
    assert report["emails_discarded"] == 0 and report["channels"]["contact_form"] == 1
    assert (report["estimated_cost_usd"], report["runs"]["completed"]) == ("0.100", 4)


@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/in/synthetic-profile", "https://linkedin.com/company/synthetic-operator-1",
    "https://uk.linkedin.com/in/synthetic-profile", "www.linkedin.com/in/synthetic-profile",
    "https://lnkd.in/synthetic",
])
def test_a_linkedin_source_is_refused_as_evidence_and_never_fetched(tmp_path, url):
    answers = contact_answers(1, person_url=url, email_url=url)
    # Even the provider's own LinkedIn excerpt is no evidence for a person or an address.
    citation = {"url": url, "excerpts": [answers["person_quote"], answers["email_quote"]]}
    basis = {site_key(1): [{"field": "person_quote", "citations": [citation]}, {"field": "email", "citations": [citation]}]}
    pages = {url: answers["person_quote"] + " " + answers["email_quote"]}  # It would verify if it were read.
    _, reader, (record,), _ = contacted(tmp_path, [answers], pages, basis=basis)
    assert record["person"] == {"verified": False, "level": "person_source_not_allowed"}
    assert record["email"] == {"verified": False, "level": "person_source_not_allowed", "discarded": True}
    assert record["recipient"]["kind"] == "none" and reader.requested == []


def test_a_linkedin_excerpt_never_stands_in_for_a_page_that_blocks_our_reader(tmp_path):
    answers = contact_answers(1)
    citation = {"url": "https://www.linkedin.com/in/synthetic-profile",
                "excerpts": [answers["person_quote"], answers["email_quote"]]}
    basis = {site_key(1): [{"field": "person_name", "citations": [citation]},
                           {"field": "email_quote", "citations": [citation]}]}
    _, reader, (record,), _ = contacted(tmp_path, [answers], {}, basis=basis)
    assert record["person"]["level"] == record["email"]["level"] == "unverified_page_unreachable"
    assert reader.requested == [answers["person_url"]]


def test_an_old_or_undated_person_source_leaves_the_role_open(tmp_path):
    stale, undated = contact_answers(1, person_date="2024-01-10"), contact_answers(2, person_date="")
    pages = {url: text for answer in (stale, undated) for url, text in pages_for(answer).items()}
    _, _, records, _ = contacted(tmp_path, [stale, undated], pages)
    for record in records:
        assert record["person"]["verified"] is True and record["person"]["current"] is False
        assert [question["check"] for question in record["open_questions"]] == ["decision_remit", "person_current"]


def test_the_contact_command_prints_counts_only(tmp_path, capsys):
    workspace, provider, keys = screened_sites(tmp_path, 2)
    answers = [contact_answers(1), contact_answers(2)]
    for key, content in zip(keys, answers):
        provider.contacts[key] = {"content": content, "basis": []}
    environ, out = {"PARALLEL_API_KEY": KEY}, str(workspace.root)
    spend = ["contact", "--out", out, "--owner-reference", OWNER, "--ceiling-usd", "1", "--max-runs", "5"]
    assert operator.main(spend, environ=environ, transport=provider)["would_create"] == 2
    assert operator.main([*spend, "--apply"], environ=environ, transport=provider)["created"] == 2
    operator.main(["collect", "--out", out, "--wait-seconds", "0"], environ=environ, transport=provider)
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    assert operator.main(["verify", "--out", out], reader=FakePages(pages), today=TODAY)["contact"] == {
        "written": 2, "kept": 0, "recipient_person_email": 2}
    report = operator.main(["summary", "--out", out])
    assert report["contact"]["recipients"] == {"person_email": 2, "team_inbox": 0, "general_inbox": 0, "none": 0}
    output = capsys.readouterr().out
    assert KEY not in output
    assert not any(value in output for value in (*SITE_STRINGS, "plant.lead", "Plant Manager", "@"))


def test_every_fixture_person_and_host_is_synthetic():
    source = Path(fixture.__file__).read_text()
    hosts = set(re.findall(r"https://([^/\"'\s]+)", source)) | set(re.findall(r"@([A-Za-z0-9.{}-]+)", source))
    assert hosts and all(host.endswith(".example") for host in hosts)
    assert all(name.split()[-1] in {"Placeholder", "Fixture"} for name in fixture.PEOPLE)
