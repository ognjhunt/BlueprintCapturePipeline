"""Hermetic robot-team universe: fake Parallel Task API, fake page reader, synthetic teams only. No network."""
import ast
import hashlib
import json
import re
import secrets
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from tests.team_universe_fixture import (
    FIXTURE_STRINGS,
    KEY,
    OWNER,
    SPEND,
    TODAY,
    Clock,
    FakePages,
    FakeProvider,
    company,
    discovered,
    discovery_output,
    discovery_pages,
    one,
    pipeline,
    query,
    query_document,
    query_set,
    screen_answers,
    screen_pages,
    setup,
    weights,
)
from tools.daily_research import site_screen as ss
from tools.team_universe import cli
from tools.team_universe import rank as tr
from tools.team_universe import universe as tu

ROOT = Path(__file__).resolve().parents[1]
REAL_VOLATILE_ROOTS = ss.VOLATILE_ROOTS


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """pytest's tmp_path is on storage the out-dir guard refuses, a shell may set the worker flag, and Git is slow."""
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", ())
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"})


# --- forms ------------------------------------------------------------------------------------
def walk(schema, depth=1):
    """Every object schema in a form with its nesting depth."""
    yield schema, depth
    for item in schema.get("properties", {}).values():
        if item["type"] == "object":
            yield from walk(item, depth + 1)
        elif item["type"] == "array":
            yield from walk(item["items"], depth + 2)


def property_count(schema):
    return sum(len(node["properties"]) for node, _ in walk(schema))


UNSUPPORTED = ("contains", "format", "maxContains", "maxItems", "maxLength", "maxProperties", "maximum", "minContains",
               "minItems", "minLength", "minimum", "minProperties", "multipleOf", "pattern", "patternProperties",
               "propertyNames", "uniqueItems", "unevaluatedItems", "unevaluatedProperties")


@pytest.mark.parametrize("stage", tu.STAGES)
def test_each_form_is_strict_and_inside_the_provider_limits(stage):
    form = tu.FORMS[stage]
    schema = form["json_schema"]
    assert form["sha256"] == hashlib.sha256(ss.canonical(schema).encode()).hexdigest()
    for node, depth in walk(schema):
        assert node["required"] == list(node["properties"]) and node["additionalProperties"] is False
        assert depth <= 5
    assert property_count(schema) <= 100  # The Task API's total across all levels.
    text = ss.canonical(schema)
    assert not any(f'"{word}"' in text for word in UNSUPPORTED)
    assert "linkedin" in text.lower()  # Every form tells the provider never to use it.


def test_the_discovery_form_lists_up_to_25_companies_with_one_quoted_source_each():
    schema = tu.DISCOVERY_SCHEMA
    assert tu.FORMS["discover"]["version"] == "blueprint.team-discovery.v1" and set(schema["properties"]) == {
        "companies", "notes"}
    item = schema["properties"]["companies"]["items"]
    assert list(item["properties"]) == ["name", "website", "robot_form", "task_focus", "source_url", "source_quote",
                                        "source_date"]
    assert all(value == {"type": "string", "description": value["description"]} for value in item["properties"].values())
    assert "25" in schema["description"] and tu.MAX_COMPANIES == 25


def test_the_team_screen_form_is_lean_and_quotes_every_answer():
    schema = tu.SCREEN_SCHEMA
    assert tu.FORMS["screen"]["version"] == "blueprint.team-screen.v1"
    assert all(item["type"] == "string" for item in schema["properties"].values())
    assert 10 <= len(tu.PROOFS) <= 20
    for name in tu.PROOFS:
        assert {name + "_url", name + "_quote"} <= set(schema["properties"]), name
    text = ss.canonical(schema).lower()
    # A published business route only: no person is asked for, and no field holds one.
    assert not any(field.startswith("person") for field in schema["properties"])
    assert "never a person" in text and "never guessed" in text
    for word in ("isaac", "mujoco", "ros"):
        assert word in text
    for family in tu.TASK_FAMILIES:
        assert family in text
    for form in tu.ROBOT_FORMS:
        assert form in text


def test_the_task_families_are_the_site_screen_focus_families():
    assert set(tu.TASK_FAMILIES) == set(ss.FOCUS_HINTS)
    assert tu.ROBOT_FORMS == ("fixed_arm", "mobile_manipulator", "humanoid", "wheeled", "bimanual", "amr_with_arm",
                              "software_only")


def test_create_bodies_name_their_form_and_keep_linkedin_out():
    queries = query_set()
    subject = tu.query_subject(queries, queries["queries"][0])
    assert ss.create_body("discover", subject, "core", tu.FORMS) == {"processor": "core", "input": subject["task_input"],
                              "metadata": {"site_key": subject["site_key"], "form": tu.DISCOVERY},
                              "source_policy": {"exclude_domains": ["linkedin.com", "lnkd.in"]},
                              "task_spec": {"output_schema": {"type": "json", "json_schema": tu.DISCOVERY_SCHEMA}}}
    assert subject["task_input"] == {"query": queries["queries"][0]["query"], "published_since": "2024-01-01",
                                     "max_companies": 25}
    # The provider's limit: the task spec and the input together stay under 25,000 characters.
    for stage, schema in (("discover", tu.DISCOVERY_SCHEMA), ("screen", tu.SCREEN_SCHEMA)):
        assert len(ss.canonical(schema)) + 3000 < 25000, stage


# --- the reviewed query set -------------------------------------------------------------------
def test_the_reviewed_query_set_is_pinned_and_covers_every_family_form_and_task():
    raw, reviewed = tu.reviewed_query_set()
    assert hashlib.sha256(raw).hexdigest() == tu.QUERIES_SHA256 == reviewed["sha256"]
    queries = reviewed["queries"]
    counts = {family: sum(item["family"] == family for item in queries) for family in tu.QUERY_FAMILIES}
    assert counts == {"funding_by_task": 11, "funding_by_form": 7, "stealth": 4, "accelerator": 4, "robot_learning": 4,
                      "integrator": 11, "university_spinout": 3, "trade_show": 4, "open_source": 3}
    assert len(queries) == 51 and reviewed["since"] == "2024-01-01" and reviewed["max_companies"] == 25
    for family in ("funding_by_task", "integrator"):
        assert sorted(item["task_focus"] for item in queries if item["family"] == family) == sorted(tu.TASK_FAMILIES)
    assert sorted(item["robot_form"] for item in queries if item["family"] == "funding_by_form") == sorted(tu.ROBOT_FORMS)
    text = " ".join(item["query"] for item in queries)
    for name in ("Y Combinator", "Techstars", "Automate", "ProMat", "MODEX", "stealth", "job posts", "open-source",
                 "pre-seed", "Series B", "spinout", "foundation model"):
        assert name in text, name
    assert all(re.search(r"\b202[3-6]\b", item["query"]) or item["family"] == "robot_learning" for item in queries)


# Capitalized words the query set may use: programs and events the owner named, funding words and acronyms.
# Any other capitalized word could start a real company list, which never belongs in this public repository.
ALLOWED_NAMES = {"Y", "Combinator", "Techstars", "Automate", "ProMat", "MODEX", "Series", "A", "B", "AI", "CNC"}


def test_the_query_set_names_no_company():
    _, reviewed = tu.reviewed_query_set()
    for item in reviewed["queries"]:
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", item["query"])
        named = {word for word in words[1:] if any(letter.isupper() for letter in word)}
        assert named <= ALLOWED_NAMES, (item["id"], named - ALLOWED_NAMES)


@pytest.mark.parametrize("change, code", [
    ({"schema_version": "blueprint.team-discovery-queries.v0"}, "team_universe_query_set_invalid"),
    ({"form": "blueprint.team-discovery.v0"}, "team_universe_query_set_invalid"),
    ({"extra": True}, "team_universe_query_set_invalid"),
    ({"since": "2024-13-01"}, "team_universe_query_set_invalid"),
    ({"max_companies": 26}, "team_universe_query_set_invalid"),
    ({"queries": []}, "team_universe_query_set_invalid"),
    ({"queries": [query(1), query(1)]}, "team_universe_query_invalid"),
    ({"queries": [query(1), query(2, query=query(1)["query"])]}, "team_universe_query_invalid"),
    ({"queries": [query(1, family="fan_mail")]}, "team_universe_query_invalid"),
    ({"queries": [query(1, task_focus="juggling")]}, "team_universe_query_invalid"),
    ({"queries": [query(1, robot_form="drone")]}, "team_universe_query_invalid"),
    ({"queries": [dict(query(1), extra=1)]}, "team_universe_query_invalid"),
    ({"queries": [query(1, query="short")]}, "team_universe_query_invalid"),
    ({"queries": [query(1, id="Bad Id")]}, "team_universe_query_invalid"),
])
def test_a_malformed_query_set_is_refused(change, code):
    with pytest.raises(ss.ScreenError, match=f"^{code}$"):
        tu.load_queries(json.dumps(query_document(**change)).encode())
    with pytest.raises(ss.ScreenError, match="^team_universe_query_set_invalid$"):
        tu.load_queries(b"not json")


def test_an_edited_query_set_is_refused_until_it_is_reviewed_and_pinned(tmp_path, monkeypatch):
    edited = tmp_path / "queries.v1.json"
    raw, _ = tu.reviewed_query_set()
    edited.write_bytes(raw.replace(b"Techstars", b"Techstarz"))
    monkeypatch.setattr(tu, "QUERIES_PATH", edited)
    with pytest.raises(ss.ScreenError, match="^team_universe_query_set_unreviewed$"):
        tu.reviewed_query_set()


def test_a_query_key_is_stable_and_changes_with_the_query():
    queries = query_set()
    first = tu.query_subject(queries, queries["queries"][0])
    assert ss.SHA.fullmatch(first["site_key"]) and tu.query_subject(queries, queries["queries"][0]) == first
    renamed = query_set(query(1, id="synthetic-renamed-1"), query(2))
    assert tu.query_subject(renamed, renamed["queries"][0])["site_key"] == first["site_key"]  # Same question.
    reworded = query_set(query(1, query="Synthetic robot companies that unload trailers at docks."))
    assert tu.query_subject(reworded, reworded["queries"][0])["site_key"] != first["site_key"]


# --- plan -------------------------------------------------------------------------------------
def test_plan_states_the_worst_case_spend_without_any_call():
    queries = query_set()
    assert tu.plan(queries) == {
        "command": "plan", "state": "planned", "forms": {"discover": tu.DISCOVERY, "screen": tu.SCREEN},
        "query_set": {"id": "synthetic-team-queries", "sha256": queries["sha256"], "queries": 3,
                      "by_family": {"funding_by_task": 1, "stealth": 1, "robot_learning": 1}},
        "processor": "core", "price_usd": "0.025",
        "discover": {"runs": 3, "worst_case_usd": "0.075"},
        "screen": {"teams_upper_bound": 75, "batch": None, "runs": 75, "worst_case_usd": "1.875"},
        "uncapped_worst_case_usd": "1.950", "limits": None, "worst_case_usd": "1.950", "provider_calls": 0}
    capped = tu.plan(queries, ceiling_usd="1", max_runs=60, screen_batch=50)
    assert capped["screen"] == {"teams_upper_bound": 75, "batch": 50, "runs": 50, "worst_case_usd": "1.250"}
    assert capped["limits"] == {"ceiling_usd": "1", "max_runs": 60, "admitted_runs": 40, "screen_runs_left": 37}
    assert (capped["uncapped_worst_case_usd"], capped["worst_case_usd"]) == ("1.325", "1.000")
    with pytest.raises(ss.ScreenError, match="^site_screen_processor_price_unknown$"):
        tu.plan(queries, processor="ultra")
    with pytest.raises(ss.ScreenError, match="^site_screen_ceiling_invalid$"):
        tu.plan(queries, ceiling_usd="101")


def test_plan_on_the_reviewed_query_set_fits_the_owner_ceiling():
    _, reviewed = tu.reviewed_query_set()
    result = tu.plan(reviewed, ceiling_usd="15", max_runs=600)
    assert result["discover"] == {"runs": 51, "worst_case_usd": "1.275"}
    assert result["limits"]["admitted_runs"] == 600 and result["worst_case_usd"] == "15.000"
    assert result["limits"]["screen_runs_left"] == 549


# --- discover: spend, idempotency and the canary ----------------------------------------------
def test_discover_creates_one_run_per_query_once_and_a_canary_runs_only_the_first(tmp_path):
    workspace, provider, client = setup(tmp_path)
    queries = query_set()
    canary = tu.discover(workspace, queries, client=client, apply=True, limit=1, **SPEND)
    assert (canary["state"], canary["created"], canary["subjects"], canary["queries"], canary["limit"]) == (
        "complete", 1, 1, 3, 1)
    assert canary["pin"] == {"state": "created", "ceiling_usd": "5", "max_runs": 200, "owner_reference": OWNER}
    full = tu.discover(workspace, queries, client=client, apply=True, **SPEND)
    assert (full["created"], full["already_created"], full["runs"], full["committed_usd"]) == (2, 1, 3, "0.075")
    again = tu.discover(workspace, queries, client=client, apply=True, **SPEND)
    assert (again["created"], again["already_created"]) == (0, 3) and len(provider.creates()) == 3
    bodies = provider.creates()
    assert [body["metadata"]["form"] for body in bodies] == [tu.DISCOVERY] * 3
    assert [body["input"]["query"] for body in bodies] == [item["query"] for item in queries["queries"]]
    intent = workspace.ledger("discover").events()[0]
    assert intent["input_sha256"] == queries["sha256"] and intent["form_sha256"] == tu.FORMS["discover"]["sha256"]
    assert {call["headers"]["x-api-key"] for call in provider.calls} == {KEY}
    for value in (0, -1, True, "1"):
        with pytest.raises(ss.ScreenError, match="^team_universe_limit_invalid$"):
            tu.discover(workspace, queries, client=client, limit=value, **SPEND)


def test_a_dry_run_admits_like_apply_and_writes_nothing(tmp_path):
    workspace, provider, client = setup(tmp_path)
    result = tu.discover(workspace, query_set(), client=client, owner_reference=OWNER, ceiling_usd="0.05", max_runs=10)
    assert (result["state"], result["would_create"], result["stop"]) == ("stopped", 2, "site_screen_spend_ceiling_reached")
    assert result["pin"]["state"] == "would_create" and provider.calls == []
    assert sorted(path.name for path in workspace.root.iterdir()) == [".lock"]


def test_a_create_that_may_exist_is_never_submitted_again(tmp_path):
    workspace, provider, client = setup(tmp_path)
    provider.create_answers = [(503, b"")]
    queries = query_set()
    first = tu.discover(workspace, queries, client=client, apply=True, **SPEND)
    assert (first["state"], first["outcome_unknown"], first["committed_usd"]) == ("stopped", 1, "0.025")
    second = tu.discover(workspace, queries, client=client, apply=True, **SPEND)
    assert (second["outcome_unknown_kept_out"], second["created"]) == (1, 2) and len(provider.creates()) == 3


def test_discovery_and_screen_share_one_pinned_ceiling_and_run_limit(tmp_path):
    spend = {"owner_reference": OWNER, "ceiling_usd": "0.10", "max_runs": 10}
    workspace, provider, client, _ = discovered(tmp_path, [[company(number) for number in range(1, 7)]], spend=spend)
    for number in range(1, 7):
        provider.outputs[tu.team_key(f"synthbot-{number}.example")] = {"content": screen_answers(number), "basis": []}
    result = tu.screen(workspace, client=client, apply=True, **spend)
    # $0.025 for the one discovery run, then three screens reach the $0.10 ceiling.
    assert (result["created"], result["stop"], result["committed_usd"]) == (3, "site_screen_spend_ceiling_reached", "0.100")
    with pytest.raises(ss.ScreenError, match="^site_screen_ceiling_above_pin$"):
        tu.screen(workspace, client=client, apply=True, owner_reference=OWNER, ceiling_usd="0.20", max_runs=10)
    with pytest.raises(ss.ScreenError, match="^site_screen_owner_reference_mismatch$"):
        tu.discover(workspace, query_set(), client=client, apply=True, owner_reference="owner-decision-other",
                    ceiling_usd="0.10", max_runs=10)
    limited = tu.screen(workspace, client=client, apply=True, owner_reference=OWNER, ceiling_usd="0.10", max_runs=4)
    assert (limited["created"], limited["stop"]) == (0, "site_screen_max_runs_reached")
    assert len(provider.creates()) == 4


def test_review_a_deleted_journal_or_ledger_never_resets_team_spend(tmp_path):
    workspace, provider, client = setup(tmp_path)
    queries = query_set()
    tu.discover(workspace, queries, client=client, apply=True, **SPEND)
    for name, code in (("spend.jsonl", "site_screen_spend_journal_missing"),
                       ("discover/runs.jsonl", "site_screen_spend_journal_mismatch")):
        path = workspace.root / name
        saved = path.read_bytes()
        path.unlink()
        with pytest.raises(ss.ScreenError, match=f"^{code}$"):
            tu.discover(workspace, queries, client=client, apply=True, **SPEND)
        with pytest.raises(ss.ScreenError, match=f"^{code}$"):
            tu.screen(workspace, client=client, apply=True, **SPEND)
        path.write_bytes(saved)
    assert len(provider.creates()) == 3


@pytest.mark.parametrize("value", ["true", ""])
def test_discover_and_screen_refuse_on_the_daily_worker(tmp_path, monkeypatch, value):
    monkeypatch.setenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", value)
    workspace, provider, client = setup(tmp_path)
    for apply in (False, True):
        with pytest.raises(ss.ScreenError, match="^site_screen_worker_needs_paid_admission$"):
            tu.discover(workspace, query_set(), client=client, apply=apply, **SPEND)
        with pytest.raises(ss.ScreenError, match="^site_screen_worker_needs_paid_admission$"):
            tu.screen(workspace, client=client, apply=apply, **SPEND)
    with pytest.raises(ss.ScreenError, match="^site_screen_worker_needs_paid_admission$"):
        cli.main(["discover", "--out", str(tmp_path / "cli"), "--owner-reference", OWNER, "--ceiling-usd", "1",
                  "--max-runs", "1", "--apply"], environ={"PARALLEL_API_KEY": KEY}, transport=provider)
    assert provider.calls == [] and not (tmp_path / "cli").exists()


# --- collect and discovery verification -------------------------------------------------------
def test_collect_stores_each_discovery_result_once_without_any_email_address(tmp_path):
    workspace, provider, client = setup(tmp_path)
    queries = query_set(query(1))
    key = tu.query_subject(queries, queries["queries"][0])["site_key"]
    listed = company(1, source_quote="Synthbot Robotics 1 raised a seed round; write to jo.doe@synthbot-1.example now.")
    provider.outputs[key] = discovery_output([listed], notes="Founder contact: someone@freemail.example.")
    provider.progress[key] = ["running"]
    tu.discover(workspace, queries, client=client, apply=True, **SPEND)
    clock = Clock()
    result = ss.collect(workspace, client=client, wait_seconds=30, poll_seconds=15, monotonic=clock.monotonic,
                        sleep=clock.sleep)
    assert result["observed"] == {"discover_completed": 1} and clock.sleeps == [15]
    stored = workspace.path("discover", "results", key).read_bytes()
    assert b"@" not in stored.replace(b"[redacted-email]", b"") and stored.count(b"[redacted-email]") == 2
    calls = len(provider.calls)
    assert ss.collect(workspace, client=client, wait_seconds=0)["observed"] == {} and len(provider.calls) == calls


def test_a_discovered_company_is_proven_only_by_a_quote_on_its_own_url_that_names_it(tmp_path):
    listed = [company(1),
              company(2, source_quote="A palletizing startup raised a seed round to build its robots."),
              company(3, source_url="https://news.example/blocked/3"),
              company(4, source_url="https://www.linkedin.com/posts/synthbot-4"),
              company(5, source_quote="Synthbot Robotics 5 raised")]
    pages = discovery_pages(listed[:2])
    workspace, *_ = discovered(tmp_path, [listed], pages=pages)
    (record,) = workspace.records("discover")
    assert record["rule_version"] == tu.DISCOVERY_RULE and record["listed"] == 5
    levels = [item["verification"]["level"] for item in record["companies"]]
    assert levels == ["verified_on_page", "verified_on_page", "unverified_page_unreachable", "source_not_allowed",
                      "quote_too_short"]
    assert [item["named"] for item in record["companies"]] == [True, False, True, True, True]
    assert [item["proven"] for item in record["companies"]] == [True, False, False, False, False]
    assert record["companies"][0]["robot_forms"] == ["mobile_manipulator"] and record["companies"][0]["fresh"] is True
    assert {"palletizing_depalletizing", "mobile_manipulator_case_picking"} <= set(record["companies"][0]["families"])


def test_the_linkedin_source_is_never_read(tmp_path):
    listed = [company(4, source_url="https://www.linkedin.com/posts/synthbot-4")]
    *_, reader = discovered(tmp_path, [listed], pages={listed[0]["source_url"]: listed[0]["source_quote"]})
    assert reader.requested == []


def test_more_than_25_companies_are_cut_and_counted(tmp_path):
    listed = [company(number) for number in range(1, 31)]
    workspace, *_ = discovered(tmp_path, [listed])
    (record,) = workspace.records("discover")
    assert (record["listed"], len(record["companies"]), record["beyond_cap"]) == (30, 25, 5)
    assert tu.summary(workspace)["discover"]["companies"]["beyond_cap"] == 5


@pytest.mark.parametrize("website, refusal", [
    ("", "team_universe_website_missing"),
    ("https://www.linkedin.com/company/synthbot-9", "team_universe_website_not_allowed"),
    ("not a url", "team_universe_website_invalid"),
    ("https://www.crunchbase.com/organization/synthbot-9", "team_universe_website_not_own"),
    ("https://synthbot-9.github.io", "team_universe_website_not_own"),
    ("https://synthbot-9.vercel.app/", "team_universe_website_not_own"),
    ("https://www.ycombinator.com/companies/synthbot-9", "team_universe_website_not_own"),
    ("https://robotics.state.example.gov/synthbot", "team_universe_website_not_own"),
    ("https://lab.university.edu/synthbot", "team_universe_website_university"),
    ("https://robots.lab.ac.uk/synthbot", "team_universe_website_university"),
])
def test_a_company_without_its_own_domain_never_becomes_a_team(website, refusal):
    assert tu.team_domain(website) == (None, refusal)


def test_a_team_domain_is_the_registrable_domain_of_its_own_site():
    for website in ("https://synthbot-9.example", "https://www.synthbot-9.example/en/", "http://app.synthbot-9.example"):
        assert tu.team_domain(website) == ("synthbot-9.example", None)
    assert tu.team_domain("https://www.synthbot-9.co.uk/about") == ("synthbot-9.co.uk", None)


def test_teams_are_deduped_by_registrable_domain_and_keep_every_source(tmp_path):
    first = [company(1), company(2)]
    second = [company(1, name="Synthbot Robotics 1 Inc.", website="https://www.synthbot-1.example/",
                      source_url="https://news.example/batch/1",
                      source_quote="Synthbot Robotics 1 Inc. joined the robotics accelerator batch this summer.",
                      source_date="2026-07-01", robot_form="humanoid"),
              company(3, website=""),
              company(4, name="Synthbot Other Name", website="https://synthbot-2.example",
                      source_quote="Synthbot Other Name raised a seed round to build palletizing robots.")]
    workspace, *_ = discovered(tmp_path, [first, second])
    teams = json.loads((workspace.root / "discover" / "teams.team-discovery-rule.v1.json").read_text())
    assert teams["schema_version"] == tu.TEAMS and [team["domain"] for team in teams["teams"]] == ["synthbot-1.example"]
    (team,) = teams["teams"]
    assert team["site_key"] == tu.team_key("synthbot-1.example") and team["origin"] == "team_discovery"
    assert [source["source_url"] for source in team["discovery"]["sources"]] == [
        "https://news.example/funding/1", "https://news.example/batch/1"]
    assert team["discovery"]["proven"] is True and team["discovery"]["mentions"] == 2
    assert team["discovery"]["robot_forms"] == ["humanoid", "mobile_manipulator"]
    assert team["discovery"]["latest"] == "2026-07-01"
    assert team["task_input"]["company"] == "Synthbot Robotics 1" and team["task_input"]["website"] == (
        "https://synthbot-1.example")
    # Two different names on one domain is no team: the domain is a directory or a mistake.
    assert teams["refused"] == {"team_universe_website_missing": 1, "team_universe_domain_names_conflict": 1}


# --- screen: the queue ------------------------------------------------------------------------
def test_screen_takes_proven_teams_in_priority_order_and_never_twice(tmp_path):
    listed = [company(1, task_focus="folding laundry", source_date="2026-09-01"),
              company(2, source_date="2025-01-01"),
              company(3, source_date="2026-06-01"),
              company(4, source_url="https://news.example/blocked/4")]
    pages = discovery_pages(listed[:3])
    workspace, provider, client, _ = discovered(tmp_path, [listed], pages=pages)
    plain = tu.screen(workspace, client=client, **SPEND)
    assert (plain["state"], plain["would_create"], plain["teams"]) == ("planned", 3, {
        "discovered": 4, "proven": 3, "unproven_held": 1, "refused": {}})
    # Without weights: proven first, then the most recent proven source.
    queue = [tu.team_key(f"synthbot-{number}.example") for number in (1, 3, 2)]
    weighted = tu.screen(workspace, client=client, apply=True, batch_size=2,
                         weights=tr.load_weights(weights(palletizing_depalletizing=4)), **SPEND)
    assert weighted["created"] == 2
    # With weights, the palletizing teams lead; the folding team has weight zero.
    assert [body["metadata"]["site_key"] for body in provider.creates(tu.SCREEN)] == [queue[1], queue[2]]
    everything = tu.screen(workspace, client=client, apply=True, include_unproven=True, **SPEND)
    assert (everything["created"], everything["already_created"]) == (2, 2)
    assert tu.screen(workspace, client=client, apply=True, include_unproven=True, **SPEND)["created"] == 0
    assert queue[0] in [body["metadata"]["site_key"] for body in provider.creates(tu.SCREEN)]
    with pytest.raises(ss.ScreenError, match="^team_universe_batch_exceeds_max_runs$"):
        tu.screen(workspace, client=client, batch_size=11, owner_reference=OWNER, ceiling_usd="5", max_runs=10)


def test_a_screen_request_sends_the_team_and_its_discovery_sources(tmp_path):
    workspace, provider, client, _ = discovered(tmp_path, [[company(1)]])
    tu.screen(workspace, client=client, apply=True, **SPEND)
    (body,) = provider.creates(tu.SCREEN)
    assert body["input"] == {"company": "Synthbot Robotics 1", "website": "https://synthbot-1.example",
                             "robot_form_hint": "mobile_manipulator",
                             "task_focus_hint": "palletizing and case picking",
                             "known_source_urls": ["https://news.example/funding/1"]}
    assert body["metadata"] == {"site_key": tu.team_key("synthbot-1.example"), "form": tu.SCREEN}
    assert body["source_policy"] == {"exclude_domains": ["linkedin.com", "lnkd.in"]}


# --- team screen records ----------------------------------------------------------------------
def test_a_fully_evidenced_team_screen_record(tmp_path):
    record = one(tmp_path, screen_answers(1))
    assert (record["schema_version"], record["rule_version"]) == (tu.SCREEN, tu.SCREEN_RULE)
    assert record["identity"] == {"state": "verified_fact", "company": "Synthbot Robotics 1"}
    assert record["robot_forms"] == {"claimed": ["mobile_manipulator"], "proven": ["mobile_manipulator"]}
    assert record["task"] == {"evidence": "pilot", "claimed": ["palletizing_depalletizing"],
                              "proven": ["palletizing_depalletizing"], "date": "2026-05-01", "recent": True}
    assert record["stage"] == {"answer": "seed", "verified": True, "amount": "USD 8 million", "date": "2026-03-01"}
    assert {name for name, signal in record["signals"].items() if signal["proven"]} == {
        "design_partners", "simulation", "learned_policy", "api_sdk", "seeking_partners"}
    assert record["signals"]["shares_policy"] == {"answer": "no", "proven": False}
    assert record["hq"] == {"answer": "Fixture City, United States", "proven": True}
    assert record["geography"] == {"answer": "United States", "proven": True}
    assert record["contact"] == {"route": "role_inbox", "address": "partnerships@synthbot-1.example",
                                 "url": "https://synthbot-1.example/contact", "email_level": "verified_on_page",
                                 "page_level": "verified_on_page", "discarded": False}
    assert record["latest_activity"] == "2026-08-01" and record["checked_on"] == TODAY.isoformat()
    assert {source["claim"] for source in record["proving_sources"]} >= {"company", "task_evidence", "funding"}


def test_each_screen_quote_must_be_on_its_own_url_and_name_its_answer(tmp_path):
    answers = screen_answers(
        1, robot_forms="humanoid",  # The quote shows a mobile manipulator, not a humanoid.
        task_families="bimanual_folding",  # The quote shows palletizing, not folding.
        stage="series_b",  # The quote shows a seed round.
        simulation_quote="We publish a monthly newsletter for every one of our customers.")
    record = one(tmp_path, answers)
    assert record["robot_forms"] == {"claimed": ["humanoid"], "proven": []}
    assert record["task"]["proven"] == [] and record["stage"]["verified"] is False
    assert record["signals"]["simulation"]["proven"] is False
    unpublished = screen_answers(1)
    pages = screen_pages(unpublished, [name for name in tu.PROOFS if name != "task_evidence"])
    assert one(tmp_path / "unpublished", unpublished, pages=pages)["task"]["proven"] == []


def test_a_third_party_page_credits_only_a_quote_that_names_the_team(tmp_path):
    answers = screen_answers(1, design_partners_url="https://news.example/roundup",
                             design_partners_quote="Several startups are accepting design partners for pilot programs.")
    assert one(tmp_path, answers)["signals"]["design_partners"]["proven"] is False
    named = screen_answers(1, design_partners_url="https://news.example/roundup",
                           design_partners_quote="Synthbot Robotics 1 is accepting design partners for pilot programs.")
    assert one(tmp_path / "named", named)["signals"]["design_partners"]["proven"] is True


def test_identity_needs_the_team_domain_and_a_matching_name(tmp_path):
    elsewhere = screen_answers(1, company_url="https://directory.example/synthbot-1")
    assert one(tmp_path / "elsewhere", elsewhere)["identity"]["state"] == "unresolved"
    renamed = screen_answers(1, company="Synthbot Unrelated Holdings",
                             company_quote="Synthbot Unrelated Holdings builds mobile manipulators for warehouses.")
    record = one(tmp_path / "renamed", renamed)
    assert record["identity"] == {"state": "contradicted", "company": "Synthbot Unrelated Holdings"}
    legal = screen_answers(1, company="Synthbot Robotics 1, Inc.",
                           company_quote="Synthbot Robotics 1, Inc. builds mobile manipulators for warehouses.")
    assert one(tmp_path / "legal", legal)["identity"]["state"] == "verified_fact"


def test_an_old_or_undated_task_and_funding_leave_activity_not_recent(tmp_path):
    answers = screen_answers(1, task_evidence_date="2023-01-01", funding_date="", seeking_partners_date="")
    record = one(tmp_path, answers)
    assert record["task"]["recent"] is False and record["latest_activity"] == "2026-03-01"  # The discovery source.


# --- the contact: a published role inbox or contact page on the team's domain -----------------
@pytest.mark.parametrize("email, reason", [
    ("avery.placeholder@synthbot-1.example", "team_universe_email_not_role_inbox"),
    ("careers@synthbot-1.example", "team_universe_email_not_role_inbox"),
    ("partnerships@freemail.example", "team_universe_email_off_team_domain"),
    ("synthbot1@gmail.com", "team_universe_email_free_mail"),
    ("not an address", "team_universe_email_invalid"),
])
def test_only_a_role_inbox_on_the_team_domain_counts(tmp_path, email, reason):
    answers = screen_answers(1, contact_email=email, contact_quote=f"Write to {email} to discuss a pilot at your site.")
    record = one(tmp_path, answers)
    assert (record["contact"]["route"], record["contact"]["email_level"]) == ("contact_page", "unverified")
    assert record["contact"]["email_reason"] == reason and record["contact"]["discarded"] is True
    assert "address" not in record["contact"]


@pytest.mark.parametrize("local", ["partnerships", "sales", "press", "info", "hello", "bd", "contact.us", "sales.eu"])
def test_a_published_role_inbox_is_the_contact_route(tmp_path, local):
    email = f"{local}@synthbot-1.example"
    answers = screen_answers(1, contact_email=email, contact_quote=f"Write to {email} to discuss a pilot at your site.")
    assert one(tmp_path, answers)["contact"]["address"] == email


def test_an_address_not_on_our_read_of_its_page_is_discarded_and_never_stored(tmp_path):
    answers = screen_answers(1)
    pages = screen_pages(answers, [name for name in tu.PROOFS if name != "contact"])
    pages[answers["contact_url"]] = "Contact us. Synthetic page with a form and no address at all on it."
    workspace, _, _, records = pipeline(tmp_path, {1: answers}, pages=pages)
    assert records[1]["contact"]["route"] == "none" and records[1]["contact"]["email_level"] == "unverified"
    for folder in ("results", "evidence", "records"):
        for path in (workspace.root / "screen" / folder).glob("*.json"):
            assert b"partnerships@" not in path.read_bytes(), path


def test_only_the_verified_role_inbox_survives_in_stored_results_and_pages(tmp_path):
    answers = screen_answers(1, notes="A founder wrote from avery.placeholder@synthbot-1.example last year.")
    pages = screen_pages(answers)
    pages[answers["contact_url"]] += " Jobs: jobs@synthbot-1.example. Founder: avery.placeholder@synthbot-1.example."
    workspace, *_ = pipeline(tmp_path, {1: answers}, pages=pages)
    stored = b"".join(path.read_bytes() for folder in ("results", "evidence")
                      for path in (workspace.root / "screen" / folder).glob("*.json"))
    assert b"partnerships@synthbot-1.example" in stored
    assert b"avery.placeholder@" not in stored and b"jobs@" not in stored


def test_a_contact_page_on_the_team_domain_is_the_route_without_an_address(tmp_path):
    answers = screen_answers(1, contact_email="", contact_url="https://synthbot-1.example/contact",
                             contact_quote="Tell us about your site and our pilot team will reply within a week.")
    record = one(tmp_path, answers)
    assert record["contact"] == {"route": "contact_page", "url": "https://synthbot-1.example/contact",
                                 "email_level": "no_email", "page_level": "verified_on_page", "discarded": False}
    offsite = screen_answers(1, contact_email="", contact_url="https://directory.example/synthbot-1",
                             contact_quote="Tell us about your site and our pilot team will reply within a week.")
    assert one(tmp_path / "offsite", offsite)["contact"]["route"] == "none"
    linkedin = screen_answers(1, contact_email="", contact_url="https://www.linkedin.com/company/synthbot-1",
                              contact_quote="Tell us about your site and our pilot team will reply within a week.")
    _, _, reader, records = pipeline(tmp_path / "linkedin", {1: linkedin})
    assert records[1]["contact"]["route"] == "none" and linkedin["contact_url"] not in reader.requested


def test_records_are_recomputed_without_a_read_or_a_run(tmp_path, monkeypatch):
    workspace, provider, reader, records = pipeline(tmp_path, {1: screen_answers(1)})
    key = tu.team_key("synthbot-1.example")
    current = workspace.record_path("screen", key)
    assert current.name == f"{key}.team-screen-rule.v1.json" and current.exists()
    calls, reads = len(provider.calls), len(reader.requested)
    monkeypatch.setattr(tu, "SCREEN_RULE", "blueprint.team-screen-rule.v2")
    monkeypatch.setattr(tu, "DISCOVERY_RULE", "blueprint.team-discovery-rule.v2")
    result = tu.verify(workspace, reader=FakePages({}), today=TODAY)
    assert result["page_reads"] == 0 and result["screen"]["records"] == 1 and result["discover"]["records"] == 1
    again = json.loads(workspace.record_path("screen", key).read_text())
    assert again["rule_version"] == "blueprint.team-screen-rule.v2" and again["contact"] == records[1]["contact"]
    assert (len(provider.calls), len(reader.requested)) == (calls, reads)


# --- summary, privacy and the command ---------------------------------------------------------
def test_summary_counts_fields_tiers_and_cost_without_any_team_data(tmp_path):
    workspace, *_ = pipeline(tmp_path, {1: screen_answers(1), 2: screen_answers(2, contact_email="", contact_url="")})
    tr.rank(workspace, weights())
    report = tu.summary(workspace)
    assert report["schema_version"] == tu.SUMMARY and report["discover"]["runs"]["completed"] == 1
    assert report["teams"] == {"discovered": 2, "proven": 2, "unproven": 0, "refused": {}}
    screen = report["screen"]
    assert screen["records"] == 2 and screen["contact_routes"] == {"role_inbox": 1, "none": 1}
    assert screen["identity"] == {"verified_fact": 2} and screen["task_families"] == {"palletizing_depalletizing": 2}
    assert screen["signals"]["simulation"] == 2 and screen["stages"] == {"seed": 2}
    assert report["rank"]["tiers"] == {"beta_candidate": 1, "prospect": 1, "insufficient": 0}
    assert (report["estimated_cost_usd"], report["committed_usd"], report["ceiling_usd"]) == ("0.075", "0.075", "5")
    assert json.loads((workspace.root / "summary.json").read_text()) == report
    text = ss.canonical(report)
    assert not any(value in text for value in FIXTURE_STRINGS) and "news.example" not in text


def test_the_out_dir_is_private_and_never_inside_a_repository_or_on_pruned_storage(tmp_path, monkeypatch):
    workspace, *_ = setup(tmp_path)
    assert workspace.root.stat().st_mode & 0o777 == 0o700
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(ss.ScreenError, match="^team_universe_out_dir_not_private$"):
        tu.TeamWorkspace(shared, create=True)
    assert shared.stat().st_mode & 0o777 == 0o755  # Never changed: the owner fixes it.
    for path in (ROOT, ROOT / "output" / "team-universe-synthetic"):
        with pytest.raises(ss.ScreenError, match="^site_screen_out_dir_inside_repository$"):
            tu.TeamWorkspace(path, create=True)
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", REAL_VOLATILE_ROOTS)
    pruned = Path("/tmp") / f"team-universe-synthetic-{secrets.token_hex(6)}"
    with pytest.raises(ss.ScreenError, match="^site_screen_out_dir_volatile$"):
        tu.TeamWorkspace(pruned, create=True)
    assert not pruned.exists()


def test_every_written_file_is_owner_only(tmp_path):
    workspace, *_ = pipeline(tmp_path, {1: screen_answers(1)})
    tr.rank(workspace, weights())
    tu.summary(workspace)
    for path in workspace.root.rglob("*"):
        assert path.stat().st_mode & 0o077 == 0, path


def test_the_command_runs_end_to_end_and_prints_counts_only(tmp_path, capsys, monkeypatch):
    queries = query_set(query(1))
    monkeypatch.setattr(tu, "reviewed_query_set", lambda: (b"synthetic", queries))
    out, key_file, weights_file = tmp_path / "out", tmp_path / "private.env", tmp_path / "weights.json"
    key_file.write_text(f"export PARALLEL_API_KEY=\"{KEY}\"\n")
    weights_file.write_bytes(weights())
    provider = FakeProvider()
    listed = [company(1), company(2)]
    provider.outputs[tu.query_subject(queries, queries["queries"][0])["site_key"]] = discovery_output(listed)
    answers = {1: screen_answers(1), 2: screen_answers(2, learned_policy="unknown", simulation="unknown",
                                                         api_sdk="unknown")}
    for number, value in answers.items():
        provider.outputs[tu.team_key(f"synthbot-{number}.example")] = {"content": value, "basis": []}
    pages = {**discovery_pages(listed), **screen_pages(answers[1]), **screen_pages(answers[2])}
    spend = ["--out", str(out), "--owner-reference", OWNER, "--ceiling-usd", "1", "--max-runs", "10",
             "--key-file", str(key_file)]
    options = {"transport": provider, "reader": FakePages(pages), "today": TODAY, "environ": {}}
    assert cli.main(["plan", "--ceiling-usd", "15", "--max-runs", "600"])["worst_case_usd"] == "0.650"
    assert cli.main(["discover", *spend, "--limit", "1"], **options)["state"] == "planned" and provider.calls == []
    assert cli.main(["discover", *spend, "--apply"], **options)["created"] == 1
    assert cli.main(["collect", "--out", str(out), "--wait-seconds", "0", "--key-file", str(key_file)],
                    **options)["state"] == "complete"
    assert cli.main(["verify", "--out", str(out)], **options)["discover"]["teams"] == 2
    assert cli.main(["screen", *spend, "--family-weights", str(weights_file), "--apply"], **options)["created"] == 2
    cli.main(["collect", "--out", str(out), "--wait-seconds", "0", "--key-file", str(key_file)], **options)
    assert cli.main(["verify", "--out", str(out)], **options)["screen"]["records"] == 2
    ranked = cli.main(["rank", "--out", str(out), "--family-weights", str(weights_file)], **options)
    assert ranked["tiers"] == {"beta_candidate": 1, "prospect": 1, "insufficient": 0}
    assert cli.main(["summary", "--out", str(out)])["screen"]["records"] == 2
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 10 and all(json.loads(line) for line in output.splitlines())
    assert KEY not in output and not any(value in output for value in FIXTURE_STRINGS)
    assert not [path for path in out.rglob("*") if path.is_file() and KEY.encode() in path.read_bytes()]
    assert {call["headers"]["x-api-key"] for call in provider.calls} == {KEY}


def test_a_failure_prints_one_stable_code_and_never_the_key(tmp_path, capsys):
    assert cli.run(["screen", "--out", str(tmp_path / "missing"), "--owner-reference", OWNER, "--ceiling-usd", "1",
                    "--max-runs", "1"]) == 1
    assert json.loads(capsys.readouterr().out) == {"state": "blocked", "error": "site_screen_out_dir_missing"}
    assert cli.run(["discover", "--out", str(tmp_path / "new"), "--owner-reference", OWNER, "--ceiling-usd", "1",
                    "--max-runs", "1", "--key-file", str(tmp_path / "missing.env")]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "site_screen_key_file_unreadable"
    assert not (tmp_path / "new").exists()


# --- packaging ----------------------------------------------------------------------------------
def test_the_team_universe_imports_only_the_standard_library_and_the_site_screen():
    allowed = set(sys.stdlib_module_names) | {"tools"}
    for name in ("universe.py", "rank.py", "cli.py", "__main__.py", "__init__.py"):
        tree = ast.parse((ROOT / "tools/team_universe" / name).read_text())
        modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        modules |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert {module.split(".")[0] for module in modules} <= allowed, name
        assert {module for module in modules if module.startswith("tools.")} <= {
            "tools.daily_research", "tools.daily_research.site_screen", "tools.team_universe",
            "tools.team_universe.universe", "tools.team_universe.rank", "tools.team_universe.cli"}, name


def test_every_fixture_company_and_host_is_synthetic():
    text = (ROOT / "tests/team_universe_fixture.py").read_text() + (ROOT / "tests/test_team_universe.py").read_text()
    hosts = set(re.findall(r"https?://([A-Za-z0-9.{}-]+)", text))
    refusal_examples = ("linkedin.com", "crunchbase.com", "ycombinator.com", ".gov", ".edu", ".ac.uk")
    assert hosts and all(host.endswith(".example") or "synthbot" in host or host.endswith(refusal_examples)
                         for host in hosts), hosts
    names = set(re.findall(r'\bname(?:": |=)f?"([^"]+)"', text))
    assert names and all(name.startswith("Synthbot") for name in names), names


def test_the_ranked_teams_are_prospects_and_the_weights_are_never_in_the_repository():
    assert not list((ROOT / "tools/team_universe").glob("*weights*"))
    config = json.loads((ROOT / "tools/team_universe/rank.v1.json").read_text())
    assert config["schema_version"] == "blueprint.team-rank.v1" and "weights" not in config
    assert Decimal(sum(config["points"].values())) == 100
