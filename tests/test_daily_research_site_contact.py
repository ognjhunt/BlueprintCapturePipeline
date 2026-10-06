"""Hermetic contact stage of the site screen line: fake Parallel Task API and fake pages; synthetic people only."""
import importlib.util
import json
import re
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
    """pytest's tmp_path is on storage the out-dir guard refuses, a shell may set the worker flag, and Git is slow."""
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", (), raising=False)
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)
    # Each --apply asks Git for the commit; one test below runs the real code_state.
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"},
                        raising=False)


def screened_sites(tmp_path, count, *, unproven=(), changes=None):
    """Screen ``count`` web-found sites, with ``changes`` to every screen answer. The site numbers in
    ``unproven`` keep their task page unpublished, so they stay screened."""
    records = [inventory_record(number) for number in range(1, count + 1)]
    answers = [screen_answers(number, **(changes or {})) for number in range(1, count + 1)]
    pages = {url: text for number, answer in enumerate(answers, 1) for url, text in pages_for(
        answer, [stem for stem in ss.SCREEN_PROOFS if not (number in unproven and stem == "target_task")]).items()}
    workspace, provider, _, _ = screen(tmp_path, records, answers, pages)
    return workspace, provider, [ss.from_inventory(record)["site_key"] for record in records]


def contacted(tmp_path, answers, pages, *, basis=None, screen_changes=None):
    """Screen one site per contact answer, then run, collect and verify the contact stage with the fakes. The
    contact pages are read when a result is collected, so its email is checked before anything is stored."""
    workspace, provider, keys = screened_sites(tmp_path, len(answers), changes=screen_changes)
    for key, content in zip(keys, answers):
        provider.contacts[key] = {"content": content, "basis": (basis or {}).get(key, [])}
    client = ss.TaskClient(KEY, transport=provider)
    result = ss.contact(workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    reader = FakePages(pages)
    ss.collect(workspace, client=client, reader=reader, today=TODAY, wait_seconds=0)
    ss.verify(workspace, reader=reader, today=TODAY)
    records = {record["site_key"]: record for record in workspace.records("contact")}
    return workspace, reader, [records[key] for key in keys if key in records], result


def stored(workspace, key):
    """Every byte the contact stage kept for one site: its raw result, page reads and record."""
    return b"".join(path.read_bytes() for path in (workspace.path("contact", "results", key),
                                                    workspace.path("contact", "evidence", key),
                                                    workspace.record_path("contact", key)))


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
    assert body["metadata"] == {"site_key": "b" * 64, "form": "blueprint.site-contact.v2"}
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
    assert (record["schema_version"], record["rule_version"]) == (ss.CONTACT, "blueprint.site-contact-rule.v3")
    assert record["decision_role"] == "Plant manager"
    assert record["person"] == {"verified": True, "level": "verified_on_page", "name": PERSON, "title": "Plant Manager",
                                "url": "https://operator-1.example/team", "date": "2026-06-01", "current": True}
    assert record["email"] == {"verified": True, "level": "verified_on_page", "discarded": False,
                               "address": "avery.placeholder@operator-1.example", "url": "https://operator-1.example/team",
                               "role": "person", "operator_domain": {"domain": "operator-1.example", "basis": "operator_quote"}}
    assert record["recipient"] == {"kind": "person_email", "rank": 0, "address": "avery.placeholder@operator-1.example"}
    # A title alone never proves remit, so the remit stays an open question.
    assert [question["check"] for question in record["open_questions"]] == ["decision_remit"]


def test_review_c1_an_email_counts_only_on_our_own_read_of_its_page(tmp_path):
    answers = contact_answers(1, email_url="https://operator-1.example/contact")
    page = {answers["email_url"]: "Synthetic page. General questions: info@operator-1.example. Footer."}
    excerpt = {"excerpts": [answers["email_quote"]]}
    for name, cited in (("broker", "https://people-broker.example/avery"), ("same", answers["email_url"])):
        basis = {site_key(1): [{"field": "email", "citations": [{"url": cited, **excerpt}]}]}
        for label, pages in (("read", {**pages_for(answers, ["person"]), **page}), ("blocked", pages_for(answers, ["person"]))):
            workspace, _, (record,), _ = contacted(tmp_path / f"{name}-{label}", [answers], pages, basis=basis)
            assert record["email"]["verified"] is False and record["email"]["discarded"] is True
            assert record["recipient"]["kind"] == "none"
            assert answers["email"].encode() not in stored(workspace, record["site_key"])


@pytest.mark.parametrize("email, published_on, reason", [
    ("avery.placeholder@gmail.com", "https://club-news.example/roster", "site_screen_email_free_mail"),  # Review C2.
    ("avery.placeholder@other-operator.example", "https://operator-1.example/team", "site_screen_email_off_operator_domain"),
    ("avery.placeholder@operator-1.example.net", "https://operator-1.example/team", "site_screen_email_off_operator_domain"),
])
def test_review_c2_an_email_must_be_on_the_operator_domain_and_never_free_mail(tmp_path, email, published_on, reason):
    answers = contact_answers(1, email=email, email_url=published_on, email_quote=f"Contact Avery at {email} about the plant.")
    pages = {**pages_for(answers, ["person"]), published_on: f"Roster. Contact Avery at {email} about the plant."}
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages)
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True, "reason": reason}
    assert record["recipient"]["kind"] == "none" and email.encode() not in stored(workspace, record["site_key"])


def test_review_c6_the_bare_website_answer_never_sets_the_email_domain(tmp_path):
    directory = "https://www.bigdirectory.example/profile/synthetic-operator-1"
    mail = "sales@bigdirectory.example"
    answers = contact_answers(1, email=mail, email_url="https://www.bigdirectory.example/profile/x",
                              email_quote=f"For quotes write to {mail} any business day.")
    pages = {**pages_for(answers, ["person"]), answers["email_url"]: f"Listing. For quotes write to {mail} any business day."}
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages, screen_changes={"website": directory})
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True,
                               "reason": "site_screen_email_off_operator_domain"}
    assert mail.encode() not in stored(workspace, record["site_key"])
    # The operator's domain is the one whose page proves the operator, and the provider is told that one.
    contact_input = next(event["input"] for event in workspace.ledger("contact").events() if event["event"] == "intent")
    assert contact_input["operator_domains"] == ["operator-1.example"]
    assert contact_input["task_input"]["website"] == "https://operator-1.example"


@pytest.mark.parametrize("operator_url", ["https://www.bigdirectory.example/profile/synthetic-operator-1",
                                          "https://records.synthetic.gov/facility/1"])
def test_a_directory_or_government_page_never_gives_the_operator_domain(tmp_path, monkeypatch, operator_url):
    monkeypatch.setattr(ss, "NOT_OPERATOR_DOMAINS", ss.NOT_OPERATOR_DOMAINS | {"bigdirectory.example"})
    host = operator_url.split("/")[2]
    mail = f"plant.team@{host.removeprefix('www.')}"
    answers = contact_answers(1, email=mail, email_url=operator_url, email_quote=f"Write to {mail} for plant questions.")
    pages = {**pages_for(answers, ["person"]), operator_url: f"Profile. Write to {mail} for plant questions."}
    _workspace, _, (record,), _ = contacted(tmp_path, [answers], pages, screen_changes={"operator_identity_url": operator_url})
    # The directory or government host never becomes an operator domain. Under contact rule v3 the website answer's
    # domain is proven by our read of the operator's own about page, so the address on that host is off the domain.
    assert record["email"]["reason"] == "site_screen_email_off_operator_domain" and record["recipient"]["kind"] == "none"
    without_site = {"website": "", "operator_identity_url": operator_url}
    _workspace, _, (record,), _ = contacted(tmp_path / "no-website", [answers], pages, screen_changes=without_site)
    assert record["email"]["reason"] == "site_screen_operator_domain_unproven" and record["recipient"]["kind"] == "none"


def test_an_address_on_a_subdomain_of_the_operator_is_accepted(tmp_path):
    email = "avery.placeholder@plant.operator-1.example"
    answers = contact_answers(1, email=email, email_quote=f"Write to {email} for plant questions.")
    _, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers))
    assert record["email"]["verified"] is True and record["recipient"]["kind"] == "person_email"


@pytest.mark.parametrize("local, label, kind, role", [
    ("info", "person_email", "general_inbox", "general"),  # Review C3: the provider's label is ignored.
    ("press", "team_inbox", "general_inbox", "general"),
    ("sales", "general_inbox", "team_inbox", "team"),
    ("plant.operations", "none", "team_inbox", "team"),
    ("careers", "general_inbox", "none", "refused"),
    ("jsmith", "person_email", "none", "unknown"),  # Neither the person's name nor a role word.
    ("aplaceholder", "team_inbox", "person_email", "person"),
])
def test_review_c3_the_recipient_kind_comes_from_the_address_itself(tmp_path, local, label, kind, role):
    email = f"{local}@operator-1.example"
    answers = contact_answers(1, email=email, email_quote=f"Write to {email} for plant questions.", channel_type=label)
    _, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers))
    assert (record["email"]["verified"], record["email"]["role"], record["recipient"]["kind"]) == (True, role, kind)
    assert record["channel"]["label"] == label


def test_a_person_email_needs_the_verified_person_named_in_the_address(tmp_path):
    answers = contact_answers(1, person_quote="Our plant manager leads the machining plant and its lathes.")
    _, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers))
    assert record["person"]["verified"] is False and record["email"]["verified"] is True
    assert record["recipient"]["kind"] == "none"


def test_the_recipient_follows_the_owner_preference_order(tmp_path):
    assert ss.RECIPIENT_PREFERENCE == ("person_email", "team_inbox", "general_inbox")
    nobody = {"person_name": "", "person_title": "", "person_url": "", "person_quote": "", "person_date": ""}
    team = contact_answers(1, **nobody, email="plant.team@operator-1.example",
                           email_quote="Write to plant.team@operator-1.example for plant questions.",
                           channel_type="general_inbox")
    general = contact_answers(2, **nobody, email="info@operator-2.example",
                              email_quote="General questions go to info@operator-2.example any day.",
                              channel_type="person_email")
    form = contact_answers(3, **nobody, email="", email_url="", email_quote="", channel_type="contact_form",
                           channel_url="https://operator-3.example/contact-us")
    person = contact_answers(4, channel_type="team_inbox")
    answers = [team, general, form, person]
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    workspace, _, records, _ = contacted(tmp_path, answers, pages)
    assert [record["recipient"]["kind"] for record in records] == ["team_inbox", "general_inbox", "none", "person_email"]
    assert [record["recipient"]["kind"] for record in sorted(records, key=lambda r: r["recipient"]["rank"])] == [
        "person_email", "team_inbox", "general_inbox", "none"]
    assert records[2]["channel"] == {"label": "contact_form", "url": "https://operator-3.example/contact-us"}
    assert records[2]["email"] == {"verified": False, "level": "no_email", "discarded": False}
    report = ss.summary(workspace)["contact"]
    assert report["recipients"] == {"person_email": 1, "team_inbox": 1, "general_inbox": 1, "none": 1}
    assert report["emails_discarded"] == 0 and report["channels"]["contact_form"] == 1
    assert (report["estimated_cost_usd"], report["runs"]["completed"]) == ("0.100", 4)


def test_an_email_not_on_its_page_is_discarded_and_never_stored(tmp_path):
    answers = contact_answers(1, notes="An older page also lists avery.p@operator-1.example for the plant.")
    citation = {"url": "https://people-broker.example/avery", "excerpts": ["Reach avery.p@operator-1.example today."]}
    basis = {site_key(1): [{"field": "notes", "citations": [citation]}]}
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers, ["person"]), basis=basis)
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True}
    kept = stored(workspace, record["site_key"])
    assert b"@operator-1.example" not in kept and b"[redacted-email]" in kept
    result = json.loads(workspace.path("contact", "results", record["site_key"]).read_text())
    assert result["output"]["content"]["email"] == "[redacted-email]" and result["run"]["status"] == "completed"


def test_only_the_verified_address_survives_in_stored_results_and_pages(tmp_path):
    answers = contact_answers(1, notes="The site also lists jordan.fixture@operator-1.example and info@operator-1.example.")
    page = {answers["email_url"]: f"{answers['person_quote']} {answers['email_quote']} Billing: billing@operator-1.example."}
    workspace, _, (record,), _ = contacted(tmp_path, [answers], {**pages_for(answers), **page})
    kept = stored(workspace, record["site_key"]).decode()
    assert record["recipient"]["address"] == "avery.placeholder@operator-1.example"
    assert set(ss.EMAIL.findall(kept.lower())) == {"avery.placeholder@operator-1.example"}


def test_a_pattern_guessed_email_is_refused(tmp_path):
    guessed = "avery.placeholder@operator-1.example"  # The person's name in a first.last pattern.
    answers = contact_answers(1, email=guessed, email_quote=f"Write to {guessed} for plant questions.")
    pages = {answers["person_url"]: f"{answers['person_quote']} Write to info@operator-1.example for plant questions."}
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages)
    assert record["person"]["verified"] is True
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True}
    assert record["recipient"]["kind"] == "none" and guessed.encode() not in stored(workspace, record["site_key"])


@pytest.mark.parametrize("published, claimed, quote_on_page", [
    ("sales@operator-1.example.net", "sales@operator-1.example", True),  # The page's address is longer.
    ("sales@operator-1.example", "sales@operator-1.exampl", False),  # The claim cuts the domain.
    ("presales@operator-1.example", "sales@operator-1.example", False),  # The page's local part is longer.
    ("sales@operator-1.example", "s.ales@operator-1.example", False),  # A changed local part.
    ("sales [at] operator-1.example", "sales@operator-1.example", False),  # Obfuscated: not published verbatim.
])
def test_only_an_address_published_verbatim_is_accepted(tmp_path, published, claimed, quote_on_page):
    quote = LONG.format(claimed)
    answers = contact_answers(1, email=claimed, email_quote=quote, email_url="https://operator-1.example/visit")
    page = "Visitors. " + LONG.format(published)
    # Whole-word matching ignores the @, so a quote can stand on a page that publishes a longer address; the
    # address check still refuses it.
    assert ss.has_phrase(ss.words(quote), ss.words(page)) is quote_on_page
    _, _, (record,), _ = contacted(tmp_path, [answers], {**pages_for(answers, ["person"]), answers["email_url"]: page})
    assert {key: record["email"][key] for key in ("verified", "level", "discarded")} == {
        "verified": False, "level": "unverified", "discarded": True}
    assert record["recipient"]["kind"] == "none"


def test_the_email_quote_must_contain_the_exact_address(tmp_path):
    answers = contact_answers(1, email_quote="Write to the plant team for plant questions.")
    pages = {answers["person_url"]: f"{answers['person_quote']} {answers['email_quote']} avery.placeholder@operator-1.example"}
    _, _, (record,), _ = contacted(tmp_path, [answers], pages)
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True,
                               "reason": "site_screen_quote_lacks_address"}


def test_the_person_quote_must_contain_the_name_and_the_page_must_show_it(tmp_path):
    nameless = contact_answers(1, person_quote="Our plant manager leads the machining plant.")
    renamed = contact_answers(2, person_name=OTHER_PERSON, person_quote=(
        f"{OTHER_PERSON} leads the machining plant, the twelve lathe cells, the inspection lab, shipping and "
        "receiving, tooling and the second shift as the plant manager."))
    pages = {nameless["person_url"]: f"{PERSON} is our plant manager. {nameless['person_quote']} "
                                     f"{nameless['email_quote']}",
             renamed["person_url"]: renamed["person_quote"].replace(OTHER_PERSON, "Jordan Fixtur") + " "
                                    + renamed["email_quote"]}
    assert not ss.has_phrase(ss.words(renamed["person_quote"]), ss.words(pages[renamed["person_url"]]))
    _, _, (first, second), _ = contacted(tmp_path, [nameless, renamed], pages)
    assert first["person"] == {"verified": False, "level": "unverified", "reason": "site_screen_quote_lacks_name"}
    assert second["person"] == {"verified": False, "level": "unverified"}
    for record in (first, second):
        # A verified address alone is not a person's email: the recipient needs a verified person too.
        assert record["email"]["verified"] is True and record["recipient"]["kind"] == "none"
        assert [question["check"] for question in record["open_questions"]] == ["decision_maker", "recipient"]


@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/in/synthetic-profile", "https://linkedin.com/company/synthetic-operator-1",
    "https://uk.linkedin.com/in/synthetic-profile", "www.linkedin.com/in/synthetic-profile",
    "https://lnkd.in/synthetic", "https://web.archive.org/web/2025/https://www.linkedin.com/in/avery-placeholder",
])
def test_review_c4_a_linkedin_source_is_refused_and_never_fetched_even_inside_a_wrapper(tmp_path, url):
    answers = contact_answers(1, person_url=url, email_url=url)
    # Even the provider's own LinkedIn excerpt is no evidence for a person or an address.
    citation = {"url": url, "excerpts": [answers["person_quote"], answers["email_quote"]]}
    basis = {site_key(1): [{"field": "person_quote", "citations": [citation]}, {"field": "email", "citations": [citation]}]}
    pages = {url: answers["person_quote"] + " " + answers["email_quote"]}  # It would verify if it were read.
    workspace, reader, (record,), _ = contacted(tmp_path, [answers], pages, basis=basis)
    assert record["person"] == {"verified": False, "level": "person_source_not_allowed"}
    assert record["email"] == {"verified": False, "level": "person_source_not_allowed", "discarded": True}
    assert record["recipient"]["kind"] == "none" and reader.requested == []
    assert answers["email"].encode() not in stored(workspace, record["site_key"])


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
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    operator.main(["collect", "--out", out, "--wait-seconds", "0"], environ=environ, transport=provider,
                  reader=FakePages(pages), today=TODAY)
    assert operator.main(["verify", "--out", out], reader=FakePages(pages), today=TODAY)["contact"] == {
        "records": 2, "pages_kept": 0, "recipient_person_email": 2}
    report = operator.main(["summary", "--out", out])
    assert report["contact"]["recipients"] == {"person_email": 2, "team_inbox": 0, "general_inbox": 0, "none": 0}
    output = capsys.readouterr().out
    assert KEY not in output
    assert not any(value in output for value in (*SITE_STRINGS, "placeholder", "Plant Manager", "@"))


def test_every_fixture_person_and_host_is_synthetic():
    source = Path(fixture.__file__).read_text()
    hosts = set(re.findall(r"https://([^/\"'\s]+)", source)) | set(re.findall(r"@([A-Za-z0-9.{}-]+)", source))
    assert hosts and all(host.endswith(".example") for host in hosts)
    assert all(name.split()[-1] in {"Placeholder", "Fixture"} for name in fixture.PEOPLE)


# --- contact rule v3 (blueprint.site-contact-rule.v3) ---------------------------------------------------------
GOVERNMENT = "https://records.synthetic.gov/facility/1"
DIRECTORY = "https://www.bigdirectory.example/profile/synthetic-operator-1"


def test_rule_v3_a_government_proven_operator_gets_its_website_domain_from_our_read_of_its_own_page(tmp_path):
    answers = contact_answers(1)
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers),
                                           screen_changes={"operator_identity_url": GOVERNMENT})
    assert record["rule_version"] == "blueprint.site-contact-rule.v3"
    # The government page proves the operator but never gives a domain; our read of the operator's own about page,
    # on the website answer's domain, names the operator.
    assert record["email"] == {"verified": True, "level": "verified_on_page", "discarded": False,
                               "address": "avery.placeholder@operator-1.example", "url": "https://operator-1.example/team",
                               "role": "person", "operator_domain": {
                                   "domain": "operator-1.example", "basis": "website", "url": "https://operator-1.example/about",
                                   "text_sha256": record["email"]["operator_domain"]["text_sha256"]}}
    assert ss.SHA.fullmatch(record["email"]["operator_domain"]["text_sha256"])
    assert record["recipient"] == {"kind": "person_email", "rank": 0, "address": "avery.placeholder@operator-1.example"}
    # Under rule v2 (operator quote domains only) the same stored data has no operator domain.
    contact_input = next(event["input"] for event in workspace.ledger("contact").events() if event["event"] == "intent")
    assert contact_input["operator_domains"] == []
    assert ss.operator_domain_proofs(contact_input) == []
    assert ss.summary(workspace)["contact"]["operator_domain_basis"] == {"website": 1}


def test_rule_v3_a_news_or_directory_page_naming_the_operator_never_sets_the_domain(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "NOT_OPERATOR_DOMAINS", ss.NOT_OPERATOR_DOMAINS | {"bigdirectory.example"})
    news = "https://local-news.example/2026/plant-shift"
    unnamed = {"operator_identity_url": DIRECTORY, "operating_now_url": news,
               "operating_now_quote": "Synthetic Operator 1 added a second shift at the plant this spring.",
               "facility_operator_quote": "The machining plant is owned and run for its own customers."}
    answers = contact_answers(1)
    _, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers), screen_changes=unnamed)
    # The directory proves the operator and the news story names it, but no page on the website's own domain does,
    # and the address on that domain spelling the name is not the page naming the operator.
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True,
                               "reason": "site_screen_operator_domain_unproven"}
    # A website answer that is itself a directory never sets the domain, even where its page names the operator.
    mail = "sales@bigdirectory.example"
    listed = contact_answers(1, email=mail, email_url=DIRECTORY, email_quote=f"For quotes write to {mail} any business day.")
    pages = {**pages_for(listed, ["person"]), DIRECTORY: f"Synthetic Operator 1. For quotes write to {mail} any business day."}
    _, _, (record,), _ = contacted(tmp_path / "directory", [listed], pages,
                                   screen_changes={**unnamed, "website": DIRECTORY})
    assert record["email"]["reason"] == "site_screen_operator_domain_unproven" and record["recipient"]["kind"] == "none"


@pytest.mark.parametrize("site_url, verified", [("https://operator-1.example/contact-us", True),
                                                ("https://chamber-members.example/list", False)])
def test_rule_v3_a_short_email_quote_counts_only_on_the_operators_own_page(tmp_path, site_url, verified):
    mail = "info@operator-1.example"
    answers = contact_answers(1, email=mail, email_url=site_url, email_quote=mail)
    assert len(ss.words(mail).split()) < ss.MIN_QUOTE_WORDS
    pages = {**pages_for(answers, ["person"]), site_url: f"Questions? {mail} Monday to Friday."}
    workspace, _, (record,), _ = contacted(tmp_path, [answers], pages)
    if verified:
        assert record["email"]["verified"] is True and record["recipient"] == {
            "kind": "general_inbox", "rank": 2, "address": mail}
    else:
        assert record["email"] == {"verified": False, "level": "quote_too_short", "discarded": True}
        assert mail.encode() not in stored(workspace, record["site_key"])


def test_rule_v3_a_short_quote_on_the_operators_page_must_still_stand_on_our_read(tmp_path):
    mail = "info@operator-1.example"
    answers = contact_answers(1, email=mail, email_url="https://operator-1.example/contact-us", email_quote=mail)
    pages = {**pages_for(answers, ["person"]), answers["email_url"]: "Questions? Use the form below."}
    _, _, (record,), _ = contacted(tmp_path, [answers], pages)
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True}


def test_rule_v3_person_quotes_keep_the_five_word_minimum(tmp_path):
    answers = contact_answers(1, person_quote=f"{PERSON}, plant manager")
    _, _, (record,), _ = contacted(tmp_path, [answers], pages_for(answers))
    assert record["person"] == {"verified": False, "level": "quote_too_short"}
    assert record["email"]["verified"] is True and record["recipient"]["kind"] == "none"


def test_rule_v3_never_recovers_an_address_an_earlier_rule_discarded(tmp_path):
    """seal_contact removes a discarded address from the stored result and page reads, so recomputing under a
    later rule keeps that decision. Recovering one needs the result read again, never a rule change."""
    answers = contact_answers(1)
    workspace, provider, keys = screened_sites(tmp_path, 1, changes={"operator_identity_url": GOVERNMENT})
    provider.contacts[keys[0]] = {"content": answers, "basis": []}
    client = ss.TaskClient(KEY, transport=provider)
    ss.contact(workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    # The page reads and the decision an earlier rule kept before an interruption: discarded, address removed.
    earlier = {"schema_version": ss.EVIDENCE, "stage": "contact", "site_key": keys[0], "checked_on": TODAY.isoformat(),
               "pages": {answers["person_url"]: {"state": "ok", "text": ss.REDACTED, "sha256": "0" * 64}},
               "email": {"level": "unverified", "reason": "site_screen_operator_domain_unproven"}}
    path = workspace.path("contact", "evidence", keys[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(earlier, sort_keys=True) + "\n")
    ss.collect(workspace, client=client, reader=FakePages(pages_for(answers)), today=TODAY, wait_seconds=0)
    ss.verify(workspace, reader=FakePages({}), today=TODAY)
    (record,) = workspace.records("contact")
    assert record["email"] == {"verified": False, "level": "unverified", "discarded": True,
                               "reason": "site_screen_operator_domain_unproven"}
    assert answers["email"].encode() not in stored(workspace, keys[0])
@pytest.mark.parametrize("address,name,expected", [
    ("sales@operator.example", "Ann Sales", "team"),
    ("info@operator.example", "Bob Info", "general"),
    ("support@operator.example", "Chris Support", "refused"),
    ("ann.sales@operator.example", "Ann Sales", "team"),
])
def test_role_inboxes_take_precedence_over_a_matching_person_name(address, name, expected):
    assert ss.address_role(address, {"verified": True, "name": name}) == expected
