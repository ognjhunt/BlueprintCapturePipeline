"""Hermetic site screen line: fake Parallel Task API, fake page reader, synthetic sites only. No network."""
import ast
import hashlib
import importlib.util
import json
import secrets
import subprocess
import sys
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pytest

from tests.daily_research_site_screen_fixture import (
    KEY,
    OWNER,
    SITE_STRINGS,
    TODAY,
    Clock,
    FakePages,
    FakeProvider,
    inventory_record,
    pages_for,
    raw_input,
    screen,
    screen_answers,
    universe_row,
)
from tests.test_daily_research_site_universe import build_export
from tools.daily_research import discovery, search
from tools.daily_research import site_screen as ss
from tools.daily_research.standalone import FILES

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("site_screen_operator", ROOT / "tools/daily_research/operators/site-screen.py")
operator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operator)
PILOT_FIELDS = {"website", "operating_now", "target_task", "target_task_found", "manual_today", "existing_automation",
                "notes", *(stem + suffix for stem in ("operating_now", "target_task", "manual_today", "existing_automation")
                           for suffix in ("_url", "_quote", "_date"))}

REAL_VOLATILE_ROOTS = getattr(ss, "VOLATILE_ROOTS", None)
REAL_CODE_STATE = getattr(ss, "code_state", None)


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    """pytest's tmp_path is on storage the out-dir guard refuses, a shell may set the worker flag, and Git is slow."""
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", (), raising=False)
    monkeypatch.delenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", raising=False)
    # Each --apply asks Git for the commit; one test below runs the real code_state.
    monkeypatch.setattr(ss, "code_state", lambda root=None: {"commit": "0" * 40, "dirty": False, "source": "git"},
                        raising=False)


def workspace_and_client(tmp_path, provider=None):
    provider = provider or FakeProvider()
    return ss.Workspace(tmp_path / "out", create=True), provider, ss.TaskClient(KEY, transport=provider)


def kinds(workspace, stage="screen"):
    return [event["event"] for event in workspace.ledger(stage).events()]


# --- form and inputs ------------------------------------------------------------------------
def test_screen_form_v2_is_strict_keeps_the_pilot_fields_and_adds_quoted_answers():
    schema = ss.SCREEN_SCHEMA
    added = {name + suffix for name in ("operator_identity", "site_identity", "facility_type", "facility_operator",
                                        "variability_signals") for suffix in ("", "_url", "_quote")}
    assert ss.SCREEN == "blueprint.site-screen.v2" and set(schema["properties"]) == PILOT_FIELDS | added
    assert schema["required"] == list(schema["properties"]) and schema["additionalProperties"] is False
    assert all(set(item) == {"type", "description"} and item["type"] == "string" for item in schema["properties"].values())
    assert ss.FORMS["screen"]["sha256"] == hashlib.sha256(ss.canonical(schema).encode()).hexdigest()
    site = ss.from_inventory(inventory_record(1))
    body = ss.create_body("screen", site, "core")
    assert body == {"processor": "core", "input": site["task_input"],
                    "metadata": {"site_key": site["site_key"], "form": "blueprint.site-screen.v2"},
                    "source_policy": {"exclude_domains": ["linkedin.com", "lnkd.in"]},
                    "task_spec": {"output_schema": {"type": "json", "json_schema": schema}}}
    # The provider's metadata limits: string keys of at most 16 and values of at most 512 characters.
    assert all(len(key) <= 16 and isinstance(value, str) and len(value) <= 512 for key, value in body["metadata"].items())
    assert ss.INVENTORY_VERSION == discovery.INVENTORY_VERSION
    assert "calibration" not in body["input"] and "address" not in body["input"]


def test_an_inventory_record_is_a_web_found_site_with_a_stable_key():
    site = ss.from_inventory(inventory_record(1))
    assert site == {"schema_version": ss.INPUT, "site_key": site["site_key"], "origin": "discovery_inventory",
                    "calibration": False, "identity": {}, "address": {"city": "Fixture City", "state": "TX"},
                    "task_input": {"site_name": "Synthetic Works 1", "operator": "Synthetic Operator 1",
                                   "location": "Fixture City, TX", "task_hint": "CNC machine tending",
                                   "known_source_urls": ["https://operator-1.example/plant"]}}
    assert ss.SHA.fullmatch(site["site_key"]) and ss.from_inventory(inventory_record(1), calibration=True)["calibration"]
    # Spacing, case and punctuation do not change the key, and another task hypothesis is the same site.
    same = inventory_record(1, operator=" synthetic  OPERATOR 1. ", task_hypothesis="Kitting")
    assert ss.from_inventory(same)["site_key"] == site["site_key"]
    assert ss.from_inventory(inventory_record(2))["site_key"] != site["site_key"]
    # A record about a site universe site shares that site's key.
    assert ss.from_inventory(inventory_record(1, site_universe_id="a" * 64))["site_key"] == "a" * 64


@pytest.mark.parametrize("location, address", [
    ("12 Main St, Springfield, IL 62701", {"street": "12 Main St", "city": "Springfield", "state": "IL"}),
    ("Fixture City, TX", {"city": "Fixture City", "state": "TX"}),
    ("Fixture City, Texas, USA", {"city": "Fixture City", "state": "TX"}),
    ("Fixture City, TX, United States", {"city": "Fixture City", "state": "TX"}),
    ("Port Fixture, New York, United States", {"city": "Port Fixture", "state": "NY"}),
    ("12 Example Road, Fixture City, TX, US", {"street": "12 Example Road", "city": "Fixture City", "state": "TX"}),
    ("Suite 4, 9 Mill Rd, Fixture City, tx 78701-1234", {"street": "Suite 4, 9 Mill Rd", "city": "Fixture City",
                                                          "state": "TX"}),
    # The backlog's form when the state is unknown: the city is kept, and the site name stays in the input.
    ("Fixture City, United States; state not individually established", {"city": "Fixture City"}),
    ("Texas Junction, United States; state not individually established", {"city": "Texas Junction"}),
    ("Fixture City", {"city": "Fixture City"}), ("Texas", {"state": "TX"}), ("United States", {}), ("", {}),
])
def test_a_free_text_location_gives_whatever_street_city_and_state_it_holds(location, address):
    assert ss.parse_location(location) == address


def anchors_for(location, site="Synthetic Works 1"):
    return ss.site_anchors(ss.from_inventory(inventory_record(1, location=location, site=site)))


@pytest.mark.parametrize("text, kind", [
    ("Our plant is at 1 Example Rd in the north end of town.", "street"),  # USPS suffixes match their full words.
    ("Visit the plant in Fixture City, TX for a tour.", "city_state"),
    ("Visit the plant in Fixture City, Texas for a tour.", "city_state"),
    ("Synthetic Works 1 in Fixture City makes parts.", "site_name_city"),
    ("Visit the plant in Fixture City for a tour today.", None),  # A city alone never counts.
    ("Fixture City is a long way from Texas by road.", None),  # The state must follow the city.
    ("Synthetic Works 1 makes parts. Fixture City is far away.", None),  # Not in one sentence.
    ("Our plant is at 10 Example Road in the north end.", None),
    ("visit the plant in fixture city, tx today.", None),  # A place name is capitalized.
])
def test_a_text_names_the_site_by_street_or_city_and_state_or_site_name_and_city(text, kind):
    assert ss.names_site(text, anchors_for("1 Example Road, Fixture City, TX")) == kind


@pytest.mark.parametrize("location, text", [
    ("Mission, TX", "Our mission is precision for every customer in Texas."),
    ("Mission, TX", "We live our mission. TX customers come first."),
    ("Commerce, CA", "We grew our e-commerce, CA sales and our e-commerce team."),
    ("Commerce, CA", "Our e-commerce business serves every Commerce customer."),
    ("Fixture City, IN", "Fixture City in the north has a plant."),  # Lower-case 'in' is not Indiana.
])
def test_review_a_common_word_city_alone_never_ties_a_text_to_the_site(location, text):
    assert ss.names_site(text, anchors_for(location)) is None


def test_a_common_word_city_still_counts_with_its_state():
    assert ss.names_site("The Mission, TX plant runs two shifts.", anchors_for("Mission, TX")) == "city_state"
    assert ss.names_site("Synthetic Works 1 opened in Mission last year.", anchors_for("Mission, TX")) == "site_name_city"


def test_the_backlog_form_without_a_state_anchors_on_the_site_name_and_city():
    anchors = anchors_for("Fixture City, United States; state not individually established")
    assert [anchor["kind"] for anchor in anchors] == ["site_name_city"]
    assert ss.names_site("Synthetic Works 1 in Fixture City hires lathe operators.", anchors) == "site_name_city"
    assert ss.names_site("Our Fixture City plant hires lathe operators.", anchors) is None
    # A generic site name is no anchor, and neither is a street without a house number.
    assert anchors_for("Main St, Fixture City, United States", site="Main Plant") == []


def test_a_site_universe_row_takes_only_the_site_address_from_its_government_record():
    row = universe_row(3)
    site = ss.from_site_universe(row)
    assert site == {"schema_version": ss.INPUT, "site_key": row["site_id"], "origin": "site_universe",
                    "calibration": False, "address": {"street": "3 Example Road", "city": "Fixture City", "state": "TX"},
                    "identity": {"physical_site": {"source": "government_record", "source_ids": ["epa_frs", "osha_ita"],
                                                   "site_id": row["site_id"],
                                                   "answer": "3 Example Road, Fixture City, TX 00003"}},
                    "task_input": {"site_name": "Synthetic Works 3", "operator": "Synthetic Operator 3",
                                   "location": "3 Example Road, Fixture City, TX 00003",
                                   "task_hint": "fixed arm machine tending", "naics": "332710"}}


def test_the_government_record_needs_an_osha_or_epa_source_and_a_full_street_address():
    for sources in (("osm_overpass",), ("fsis_mpi", "osm_overpass")):
        assert ss.from_site_universe(universe_row(3, sources=sources))["identity"] == {}
    for changes in ({"street": None}, {"city": None}, {"state": None}):
        site = ss.from_site_universe(universe_row(3, **changes))
        assert site["identity"] == {} and "operator" not in site["identity"]
    # The row's sources are all kept, so a reader sees that OpenStreetMap may have given the name.
    assert ss.from_site_universe(universe_row(3, sources=("epa_frs", "osm_overpass")))["identity"]["physical_site"][
        "source_ids"] == ["epa_frs", "osm_overpass"]
    with pytest.raises(ss.ScreenError, match="^site_screen_input_record_invalid$"):
        ss.from_site_universe(universe_row(3, sources=[]))


def test_load_sites_reads_an_export_an_inventory_page_or_a_list_and_refuses_repeats():
    rows = [universe_row(1), universe_row(2)]
    export = build_export(rows)
    sites, refused = ss.load_sites(export)
    assert [site["site_key"] for site in sites] == [row["site_id"] for row in rows] and refused == {}
    page = {"version": discovery.INVENTORY_VERSION, "run_key": "synthetic", "start": 0, "end": 3,
            "records": [inventory_record(1), inventory_record(1, task_hypothesis="Kitting"),
                        inventory_record(2, disposition="rejected")]}
    sites, refused = ss.load_sites(json.dumps(page).encode())
    assert len(sites) == 1 and refused == {"site_screen_input_duplicate_site": 1, "site_screen_input_rejected": 1}
    mixed, refused = ss.load_sites(raw_input([rows[0], inventory_record(4), {"unexpected": True}]))
    assert [site["origin"] for site in mixed] == ["site_universe", "discovery_inventory"]
    assert refused == {"site_screen_input_record_invalid": 1}
    with pytest.raises(ss.ScreenError, match="^site_screen_input_export_invalid$"):
        ss.load_sites(export[:-4])
    for raw in (b'{"rows": []}', b"not json"):
        with pytest.raises(ss.ScreenError, match="^site_screen_input_format_unknown$"):
            ss.load_sites(raw)


def test_plan_counts_sites_refusals_and_cost_without_any_call():
    raw = raw_input([inventory_record(1), inventory_record(2), universe_row(3), inventory_record(5, location=None)])
    assert ss.plan(raw) == {
        "command": "plan", "state": "planned", "form": ss.SCREEN, "sites": 3,
        "by_origin": {"discovery_inventory": 2, "site_universe": 1}, "government_record": {"physical_site": 1},
        "site_anchors": {"usable": 3, "none": 0, "street": 1, "city_state": 3, "site_name_city": 3},
        "input_refused": {"site_screen_input_location_missing": 1}, "processor": "core", "price_usd": "0.025",
        "batch": {"size": 3, "calibration": 0, "seed": None}, "estimated_cost_usd": "0.075", "provider_calls": 0}
    with pytest.raises(ss.ScreenError, match="^site_screen_processor_price_unknown$"):
        ss.plan(raw, processor="ultra")


def test_a_batch_keeps_rank_order_and_adds_about_one_seeded_calibration_site_in_ten():
    sites, _ = ss.load_sites(raw_input([inventory_record(number) for number in range(1, 101)]))
    batch = ss.select_batch(sites, 30, "seed-one")
    keys = [site["site_key"] for site in sites]
    ranked = [site for site in batch if not site["calibration"]]
    calibration = [site for site in batch if site["calibration"]]
    assert (len(batch), len(ranked), len(calibration)) == (30, 27, 3)
    assert [site["site_key"] for site in ranked] == keys[:27]
    assert all(keys.index(site["site_key"]) >= 27 for site in calibration)
    # One calibration site follows each nine ranked sites, so a run that stops early keeps the mix.
    assert [index for index, site in enumerate(batch) if site["calibration"]] == [9, 19, 29]
    assert ss.select_batch(sites, 30, "seed-one") == batch
    assert {s["site_key"] for s in ss.select_batch(sites, 30, "seed-two") if s["calibration"]} != {
        s["site_key"] for s in calibration}
    # When every site fits there is no cut, so there is no calibration site.
    assert [s["calibration"] for s in ss.select_batch(sites, 100, "seed-one")] == [False] * 100
    assert sum(s["calibration"] for s in ss.select_batch(sites, 5, "seed-one")) == 0
    plan = ss.plan(raw_input([inventory_record(number) for number in range(1, 101)]), batch_size=30, seed="seed-one")
    assert plan["batch"] == {"size": 30, "calibration": 3, "seed": "seed-one"} and plan["estimated_cost_usd"] == "0.750"
    for size in (0, -1, 5001, True):
        with pytest.raises(ss.ScreenError, match="^site_screen_batch_size_invalid$"):
            ss.select_batch(sites, size, "seed-one")


def test_run_screens_a_batch_within_max_runs_and_records_the_calibration_flag(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in range(1, 41)])
    with pytest.raises(ss.ScreenError, match="^site_screen_batch_exceeds_max_runs$"):
        ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="5", max_runs=19, batch_size=20,
               apply=True)
    result = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="5", max_runs=20, batch_size=20,
                    seed="seed-one", apply=True)
    assert (result["created"], result["calibration"], result["batch"]) == (20, 2, {"size": 20, "seed": "seed-one"})
    flags = [event["input"]["calibration"] for event in workspace.ledger("screen").events() if event["event"] == "intent"]
    assert flags.count(True) == 2 and len(provider.creates()) == 20
    assert all("calibration" not in body["input"] for body in provider.creates())


# --- spend: idempotency, ceiling and max_runs -----------------------------------------------
def test_a_dry_run_admits_like_apply_and_pins_nothing(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in range(1, 6)])
    result = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="0.075", max_runs=10)
    assert result["state"] == "stopped" and result["stop"] == "site_screen_spend_ceiling_reached"
    assert result["would_create"] == 3 and result["created"] == 0 and result["committed_usd"] == "0.075"
    assert result["pin"] == {"state": "would_create", "ceiling_usd": "0.075", "max_runs": 10, "owner_reference": OWNER}
    assert provider.calls == [] and not (workspace.root / "screen").exists()
    assert not (workspace.root / "owner_ceiling.json").exists() and not (workspace.root / "spend.jsonl").exists()


def test_each_site_is_created_once_and_a_rerun_creates_nothing(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(1), universe_row(2)])
    sites, _ = ss.load_sites(raw)
    first = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert (first["state"], first["created"], first["runs"], first["committed_usd"]) == ("complete", 2, 2, "0.050")
    assert provider.creates() == [ss.create_body("screen", site, "core") for site in sites]
    assert {call["headers"]["x-api-key"] for call in provider.calls} == {KEY}
    second = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert (second["created"], second["already_created"]) == (0, 2) and len(provider.creates()) == 2
    assert kinds(workspace) == ["intent", "created", "intent", "created"]
    intent = workspace.ledger("screen").events()[0]
    assert intent["input"] == sites[0] and intent["price_usd"] == "0.025" and intent["ceiling_usd"] == "1"
    assert intent["form_sha256"] == ss.FORMS["screen"]["sha256"]
    assert intent["input_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("answer", [
    ss.TransportError("site_screen_provider_connection_lost", sent=True), (500, b"{}"), (503, b""), (504, b""),
    (202, b'{"status": "queued"}'), (202, b"not json"),
])
def test_a_create_that_may_exist_is_never_submitted_again(tmp_path, answer):
    workspace, provider, client = workspace_and_client(tmp_path)
    provider.create_answers = [answer]
    raw = raw_input([inventory_record(1), inventory_record(2)])
    first = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    # The first unknown outcome stops the run, and its price stays committed.
    assert first["state"] == "stopped" and first["stop"].startswith("site_screen_create_")
    assert (first["outcome_unknown"], first["created"], first["committed_usd"]) == (1, 0, "0.025")
    second = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert (second["outcome_unknown_kept_out"], second["created"]) == (1, 1)
    assert [body["metadata"]["site_key"] for body in provider.creates()] == [
        site["site_key"] for site in ss.load_sites(raw)[0]]
    assert kinds(workspace) == ["intent", "uncertain", "intent", "created"]


def test_an_interrupted_create_counts_as_unknown_and_is_never_submitted_again(tmp_path):
    class Interrupted(BaseException):
        pass

    workspace, provider, client = workspace_and_client(tmp_path)
    provider.create_answers = [Interrupted()]
    raw = raw_input([inventory_record(1), inventory_record(2)])
    with pytest.raises(Interrupted):
        ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert kinds(workspace) == ["intent"]
    again = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert (again["outcome_unknown_kept_out"], again["created"], again["committed_usd"]) == (1, 1, "0.050")
    assert len(provider.creates()) == 2


def test_a_refused_create_costs_nothing_and_is_tried_again_later(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    provider.create_answers = [(422, b'{"detail": "validation"}')]
    raw = raw_input([inventory_record(1)])
    first = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert (first["state"], first["refused"], first["runs"], first["committed_usd"]) == (
        "complete", {"site_screen_create_rejected": 1}, 0, "0")
    second = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert second["created"] == 1 and kinds(workspace) == ["intent", "refused", "intent", "created"]
    assert workspace.ledger("screen").events()[1]["http_status"] == 422


@pytest.mark.parametrize("answer, code", [
    ((401, b""), "site_screen_provider_auth_refused"), ((402, b""), "site_screen_provider_credit_exhausted"),
    ((403, b""), "site_screen_processor_refused"), ((429, b""), "site_screen_provider_rate_limited"),
    ((302, b""), "site_screen_provider_redirect_refused"),
    (ss.TransportError("site_screen_provider_unreachable", sent=False), "site_screen_provider_unreachable"),
])
def test_a_refusal_every_later_create_would_meet_stops_the_run(tmp_path, answer, code):
    workspace, provider, client = workspace_and_client(tmp_path)
    provider.create_answers = [answer]
    result = ss.run(raw_input([inventory_record(number) for number in (1, 2, 3)]), workspace, client=client,
                    owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    assert (result["state"], result["stop"], result["refused"], result["created"]) == ("stopped", code, {code: 1}, 0)
    assert len(provider.calls) == 1 and result["committed_usd"] == "0"


def test_the_ceiling_and_max_runs_refuse_before_any_create(tmp_path):
    raw = raw_input([inventory_record(number) for number in (1, 2, 3)])
    workspace, provider, client = workspace_and_client(tmp_path / "below")
    below = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="0.02", max_runs=10, apply=True)
    assert (below["created"], below["stop"]) == (0, "site_screen_spend_ceiling_reached")
    assert provider.calls == [] and not workspace.ledger("screen").path.exists()
    workspace, provider, client = workspace_and_client(tmp_path / "limit")
    one = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=1, apply=True)
    assert (one["created"], one["stop"]) == (1, "site_screen_max_runs_reached")
    again = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=1, apply=True)
    assert (again["created"], again["already_created"], again["stop"]) == (0, 1, "site_screen_max_runs_reached")
    assert len(provider.creates()) == 1
    workspace, provider, client = workspace_and_client(tmp_path / "exact")
    exact = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="0.075", max_runs=10, apply=True)
    assert (exact["created"], exact["state"], exact["committed_usd"]) == (3, "complete", "0.075")


def test_the_ceiling_counts_runs_in_flight_and_frees_only_failed_ones(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in (1, 2, 3)])
    sites, _ = ss.load_sites(raw)
    options = {"client": client, "owner_reference": OWNER, "ceiling_usd": "0.05", "max_runs": 3, "apply": True}
    first = ss.run(raw, workspace, **options)
    assert (first["created"], first["stop"]) == (2, "site_screen_spend_ceiling_reached")
    provider.final[sites[0]["site_key"]] = "failed"
    provider.final[sites[1]["site_key"]] = "cancelled"  # Not known to be free, so its price stays committed.
    provider.outputs[sites[1]["site_key"]] = {"content": {}, "basis": []}
    ss.collect(workspace, client=client, wait_seconds=0)
    second = ss.run(raw, workspace, **options)
    assert (second["created"], second["committed_usd"]) == (1, "0.050")
    # A failed run is free but still a run: max_runs counts it.
    assert ss.run(raw_input([inventory_record(4)]), workspace, **options)["stop"] == "site_screen_max_runs_reached"


@pytest.mark.parametrize("ceiling", ["0", "-1", "nan", "inf", "100.01", "a dollar", ""])
def test_a_ceiling_outside_the_reviewed_bound_is_refused(tmp_path, ceiling):
    workspace, provider, client = workspace_and_client(tmp_path)
    with pytest.raises(ss.ScreenError, match="^site_screen_ceiling_invalid$"):
        ss.run(raw_input([inventory_record(1)]), workspace, client=client, owner_reference=OWNER, ceiling_usd=ceiling, max_runs=1, apply=True)
    assert provider.calls == []


@pytest.mark.parametrize("max_runs", [0, -1, 5001, True, 1.0])
def test_max_runs_outside_the_reviewed_bound_is_refused(tmp_path, max_runs):
    workspace, provider, client = workspace_and_client(tmp_path)
    with pytest.raises(ss.ScreenError, match="^site_screen_max_runs_invalid$"):
        ss.run(raw_input([inventory_record(1)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=max_runs,
               apply=True)
    assert provider.calls == []


def test_a_second_command_on_the_same_out_dir_refuses(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    with workspace.lock(), pytest.raises(ss.ScreenError, match="^site_screen_out_dir_busy$"):
        ss.run(raw_input([inventory_record(1)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=1, apply=True)
    assert provider.calls == []


def test_a_torn_final_ledger_line_is_sealed_and_other_damage_refuses(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    ss.run(raw_input([inventory_record(1)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=5, apply=True)
    ledger = workspace.ledger("screen")
    with open(ledger.path, "ab") as handle:  # A crash in the middle of a write.
        handle.write(b'{"schema_version":"blueprint.site-screen.led')
    assert kinds(workspace) == ["intent", "created"]
    ss.run(raw_input([inventory_record(2)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=5, apply=True)
    assert kinds(workspace) == ["intent", "created", "intent", "created"]
    assert ledger.path.read_bytes().count(b'"event":"sealed"') == 1
    ledger.path.write_bytes(ledger.path.read_bytes().replace(b'"event":"created"', b'"event":"made"', 1))
    with pytest.raises(ss.ScreenError, match="^site_screen_ledger_invalid$"):
        ss.run(raw_input([inventory_record(3)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=5, apply=True)
    assert len(provider.creates()) == 2


def test_the_first_apply_pins_the_owner_ceiling_once(tmp_path):
    workspace, _provider, client = workspace_and_client(tmp_path)
    ss.run(raw_input([inventory_record(1)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="0.50",
           max_runs=7, apply=True)
    path = workspace.root / "owner_ceiling.json"
    pin = json.loads(path.read_text())
    assert set(pin) == {"schema_version", "ceiling_usd", "max_runs", "created_at", "owner_reference"}
    assert (pin["schema_version"], pin["ceiling_usd"], pin["max_runs"], pin["owner_reference"]) == (
        ss.OWNER_CEILING, "0.50", 7, OWNER)
    assert path.stat().st_mode & 0o777 == 0o600
    journal = workspace.journal().events()
    assert journal[0]["event"] == "pinned" and journal[0]["pin"] == pin
    assert [event["event"] for event in journal[1:]] == ["intent", "created"]
    # A lower ceiling binds one invocation and leaves the pin as it was.
    lower = ss.run(raw_input([inventory_record(number) for number in (2, 3)]), workspace, client=client,
                   owner_reference=OWNER, ceiling_usd="0.05", max_runs=7, apply=True)
    assert (lower["created"], lower["stop"], lower["pin"]["state"]) == (1, "site_screen_spend_ceiling_reached", "pinned")
    assert json.loads(path.read_text()) == pin


@pytest.mark.parametrize("changes, code", [
    ({"ceiling_usd": "0.11"}, "site_screen_ceiling_above_pin"),
    ({"ceiling_usd": "100"}, "site_screen_ceiling_above_pin"),
    ({"max_runs": 5}, "site_screen_max_runs_above_pin"),
    ({"max_runs": 5000}, "site_screen_max_runs_above_pin"),
    ({"owner_reference": "owner-decision-synthetic-other"}, "site_screen_owner_reference_mismatch"),
])
def test_review_s2_a_later_invocation_can_never_raise_the_pin(tmp_path, changes, code):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in range(1, 11)])
    first = ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="0.10", max_runs=4, apply=True)
    assert (first["created"], first["stop"]) == (4, "site_screen_max_runs_reached")
    options = {"owner_reference": OWNER, "ceiling_usd": "0.10", "max_runs": 4, **changes}
    for apply in (False, True):
        with pytest.raises(ss.ScreenError, match=f"^{code}$"):
            ss.run(raw, workspace, client=client, apply=apply, **options)
        with pytest.raises(ss.ScreenError, match=f"^{code}$"):
            ss.contact(workspace, client=client, apply=apply, **options)
    assert len(provider.creates()) == 4


@pytest.mark.parametrize("ceiling, max_runs, contact_created, stop", [
    ("0.25", 20, 2, "site_screen_spend_ceiling_reached"),  # $0.20 screen + 2 x $0.025 contact.
    ("15", 9, 1, "site_screen_max_runs_reached"),  # 8 screen runs + 1 contact run.
])
def test_review_s1_both_stages_share_one_ceiling_and_run_limit(tmp_path, ceiling, max_runs, contact_created, stop):
    records = [inventory_record(number) for number in range(1, 9)]
    answers = [screen_answers(number) for number in range(1, 9)]
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    workspace, provider, client = workspace_and_client(tmp_path)
    for site, content in zip(ss.load_sites(raw_input(records))[0], answers):
        provider.outputs[site["site_key"]] = {"content": content, "basis": []}
        provider.contacts[site["site_key"]] = {"content": {}, "basis": []}
    options = {"client": client, "owner_reference": OWNER, "ceiling_usd": ceiling, "max_runs": max_runs, "apply": True}
    assert ss.run(raw_input(records), workspace, **options)["created"] == 8
    ss.collect(workspace, client=client, wait_seconds=0)
    ss.verify(workspace, reader=FakePages(pages), today=TODAY)
    contact = ss.contact(workspace, **options)
    assert (contact["sites"], contact["created"], contact["stop"]) == (8, contact_created, stop)
    report = ss.summary(workspace)
    assert len(provider.creates()) == 8 + contact_created <= max_runs
    assert Decimal(report["committed_usd"]) == Decimal("0.025") * (8 + contact_created) <= Decimal(ceiling)


def test_review_s3_a_deleted_or_altered_record_never_resets_spend(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in range(1, 4)])
    options = {"client": client, "owner_reference": OWNER, "ceiling_usd": "1", "max_runs": 10, "apply": True}
    ss.run(raw, workspace, **options)
    files = {name: workspace.root / name for name in ("screen/runs.jsonl", "spend.jsonl", "owner_ceiling.json")}
    saved = {name: path.read_bytes() for name, path in files.items()}
    damage = [("screen/runs.jsonl", None, "site_screen_spend_journal_mismatch"),
              ("spend.jsonl", None, "site_screen_spend_journal_missing"),
              ("owner_ceiling.json", None, "site_screen_owner_ceiling_missing"),
              ("owner_ceiling.json", saved["owner_ceiling.json"].replace(b'"1"', b'"100"'),
               "site_screen_owner_ceiling_mismatch")]
    for name, content, code in damage:
        files[name].unlink()
        if content is not None:
            files[name].write_bytes(content)
        for command in (lambda: ss.run(raw, workspace, **options), lambda: ss.contact(workspace, **options),
                        lambda: ss.collect(workspace, client=client, wait_seconds=0),
                        lambda: ss.verify(workspace, reader=FakePages({}), today=TODAY), lambda: ss.summary(workspace)):
            with pytest.raises(ss.ScreenError, match=f"^{code}$"):
                command()
        files[name].write_bytes(saved[name])
    assert len(provider.creates()) == 3
    assert ss.run(raw, workspace, **options)["already_created"] == 3 and len(provider.creates()) == 3


def test_review_s4b_kept_results_whose_ledger_entries_are_gone_refuse(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in range(1, 4)])
    for site in ss.load_sites(raw)[0]:
        provider.outputs[site["site_key"]] = {"content": screen_answers(1), "basis": []}
    options = {"client": client, "owner_reference": OWNER, "ceiling_usd": "1", "max_runs": 10, "apply": True}
    ss.run(raw, workspace, **options)
    ss.collect(workspace, client=client, wait_seconds=0)
    journal = workspace.root / "spend.jsonl"
    journal.write_bytes(journal.read_bytes().split(b"\n")[0] + b"\n")  # Only the pin is left.
    (workspace.root / "screen" / "runs.jsonl").unlink()
    with pytest.raises(ss.ScreenError, match="^site_screen_spend_journal_mismatch$"):
        ss.run(raw, workspace, **options)
    assert len(provider.creates()) == 3


def test_each_intent_records_the_code_commit_and_whether_the_tree_was_dirty(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "code_state", REAL_CODE_STATE)
    workspace, _provider, client = workspace_and_client(tmp_path)
    result = ss.run(raw_input([inventory_record(1)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="1",
                    max_runs=1, apply=True)
    intent = workspace.ledger("screen").events()[0]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert intent["code"]["commit"] == head and intent["code"]["source"] == "git"
    assert type(intent["code"]["dirty"]) is bool and result["code"] == intent["code"]
    release = tmp_path / "release"
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps({"source_commit": "a" * 40}))
    assert ss.code_state(release) == {"commit": "a" * 40, "dirty": None, "source": "release_manifest"}
    assert ss.code_state(tmp_path / "nothing") == {"commit": None, "dirty": None, "source": "unknown"}


def test_a_used_out_dir_without_its_journal_refuses(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(1)])
    options = {"client": client, "owner_reference": OWNER, "ceiling_usd": "1", "max_runs": 10, "apply": True}
    ss.run(raw, workspace, **options)
    provider.outputs[ss.load_sites(raw)[0][0]["site_key"]] = {"content": screen_answers(1), "basis": []}
    ss.collect(workspace, client=client, wait_seconds=0)
    for name in ("spend.jsonl", "owner_ceiling.json", "screen/runs.jsonl"):
        (workspace.root / name).unlink()
    # The stored result shows this out dir has spent before, so it cannot start again from zero.
    with pytest.raises(ss.ScreenError, match="^site_screen_spend_journal_missing$"):
        ss.run(raw, workspace, **options)
    assert len(provider.creates()) == 1


def test_a_crash_between_the_journal_and_the_ledger_is_completed_from_the_journal(tmp_path, monkeypatch):
    class Interrupted(BaseException):
        pass

    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(1), inventory_record(2)])
    options = {"client": client, "owner_reference": OWNER, "ceiling_usd": "1", "max_runs": 10, "apply": True}
    second = ss.load_sites(raw)[0][1]["site_key"]
    append = ss.Ledger.append

    def interrupted(ledger, event):
        if ledger.stage == "screen" and event["event"] == "intent" and event["site_key"] == second:
            raise Interrupted  # The journal copy is on disk; the ledger copy is not.
        return append(ledger, event)

    monkeypatch.setattr(ss.Ledger, "append", interrupted)
    with pytest.raises(Interrupted):
        ss.run(raw, workspace, **options)
    monkeypatch.setattr(ss.Ledger, "append", append)
    again = ss.run(raw, workspace, **options)
    assert (again["created"], again["already_created"], again["outcome_unknown_kept_out"]) == (0, 1, 1)
    assert kinds(workspace) == ["intent", "created", "intent"] and len(provider.creates()) == 1
    assert again["committed_usd"] == "0.050"


def test_a_crash_after_the_journal_pin_completes_the_pin_file(tmp_path, monkeypatch):
    class Interrupted(BaseException):
        pass

    workspace, provider, client = workspace_and_client(tmp_path)
    options = {"client": client, "owner_reference": OWNER, "ceiling_usd": "1", "max_runs": 10, "apply": True}
    write_once = ss._write_once
    monkeypatch.setattr(ss, "_write_once", lambda path, data: (_ for _ in ()).throw(Interrupted())
                        if path.name == "owner_ceiling.json" else write_once(path, data))
    with pytest.raises(Interrupted):
        ss.run(raw_input([inventory_record(1)]), workspace, **options)
    monkeypatch.setattr(ss, "_write_once", write_once)
    assert not (workspace.root / "owner_ceiling.json").exists() and provider.calls == []
    assert ss.run(raw_input([inventory_record(1)]), workspace, **options)["created"] == 1
    pin = json.loads((workspace.root / "owner_ceiling.json").read_text())
    assert pin == workspace.journal().events()[0]["pin"]


@pytest.mark.parametrize("value", ["true", "false", ""])
def test_run_and_contact_refuse_on_the_worker_until_paid_admission_exists(tmp_path, monkeypatch, value):
    monkeypatch.setenv("BLUEPRINT_DAILY_RESEARCH_WORKER_ENABLED", value)
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(1)])
    for apply in (False, True):
        with pytest.raises(ss.ScreenError, match="^site_screen_worker_needs_paid_admission$"):
            ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=1, apply=apply)
        with pytest.raises(ss.ScreenError, match="^site_screen_worker_needs_paid_admission$"):
            ss.contact(workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=1, apply=apply)
    source = tmp_path / "sites.json"
    source.write_bytes(raw)
    with pytest.raises(ss.ScreenError, match="^site_screen_worker_needs_paid_admission$"):
        operator.main(["run", "--input", str(source), "--out", str(tmp_path / "cli"), "--owner-reference", OWNER,
                       "--ceiling-usd", "1", "--max-runs", "1", "--apply"], environ={"PARALLEL_API_KEY": KEY},
                      transport=provider)
    assert provider.calls == [] and not (workspace.root / "owner_ceiling.json").exists()


@pytest.mark.parametrize("root", ["/tmp", "/private/tmp", "/var/tmp", "/var/folders/zz"])
def test_an_out_dir_on_storage_the_system_prunes_is_refused(monkeypatch, root):
    monkeypatch.setattr(ss, "VOLATILE_ROOTS", REAL_VOLATILE_ROOTS)
    path = Path(root) / f"site-screen-synthetic-{secrets.token_hex(6)}"
    with pytest.raises(ss.ScreenError, match="^site_screen_out_dir_volatile$"):
        ss.Workspace(path, create=True)
    assert not path.exists()
    assert ss.guard_out_dir("/Users/Shared/blueprint-private/site-screen-synthetic").name == "site-screen-synthetic"


# --- client: status, result and failed runs -------------------------------------------------
def test_collect_stores_each_terminal_response_once_and_observes_it(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in (1, 2, 3)])
    sites, _ = ss.load_sites(raw)
    keys = [site["site_key"] for site in sites]
    for key in keys:
        provider.outputs[key] = {"content": screen_answers(1), "basis": []}
    provider.progress[keys[1]] = ["queued", "running"]
    provider.final[keys[2]] = "failed"
    ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=10, apply=True)
    clock = Clock()
    result = ss.collect(workspace, client=client, wait_seconds=60, poll_seconds=15, monotonic=clock.monotonic,
                        sleep=clock.sleep)
    assert result == {"command": "collect", "state": "complete", "observed": {"screen_completed": 2, "screen_failed": 1},
                      "still_running": 0, "read_errors": {}}
    assert clock.sleeps == [15, 15]
    events = [event for event in workspace.ledger("screen").events() if event["event"] == "observed"]
    for event in events:
        stored = workspace.path("screen", "results", event["site_key"]).read_bytes()
        assert hashlib.sha256(stored).hexdigest() == event["result_sha256"]
    assert json.loads(workspace.path("screen", "results", keys[2]).read_bytes())["status"] == "failed"
    calls = len(provider.calls)
    assert ss.collect(workspace, client=client, wait_seconds=0)["observed"] == {} and len(provider.calls) == calls


def test_collect_waits_within_its_bound_and_reports_runs_still_in_flight(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(1)])
    key = ss.load_sites(raw)[0][0]["site_key"]
    provider.progress[key] = ["running"] * 10
    ss.run(raw, workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=1, apply=True)
    clock = Clock()
    result = ss.collect(workspace, client=client, wait_seconds=30, poll_seconds=15, monotonic=clock.monotonic,
                        sleep=clock.sleep)
    assert (result["state"], result["still_running"], clock.sleeps) == ("pending", 1, [15, 15])
    with pytest.raises(ss.ScreenError, match="^site_screen_wait_invalid$"):
        ss.collect(workspace, client=client, wait_seconds=7201)


def test_collect_stops_on_an_auth_refusal_and_counts_other_read_errors(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    ss.run(raw_input([inventory_record(1)]), workspace, client=client, owner_reference=OWNER, ceiling_usd="1", max_runs=1, apply=True)
    provider.read_answers = [(500, b"upstream"), ss.TransportError("site_screen_provider_unreachable", sent=False)]
    clock = Clock()
    result = ss.collect(workspace, client=client, wait_seconds=15, poll_seconds=15, monotonic=clock.monotonic,
                        sleep=clock.sleep)
    assert result["read_errors"] == {"site_screen_status_unavailable": 2} and result["still_running"] == 1
    provider.read_answers = [(401, b"")]
    with pytest.raises(ss.ScreenError, match="^site_screen_provider_auth_refused$"):
        ss.collect(workspace, client=client, wait_seconds=0)


@pytest.mark.parametrize("path, answer, code", [
    ("status", (200, b'{"run_id": "trun_9999", "status": "completed"}'), "site_screen_status_invalid"),
    ("status", (200, b'{"run_id": "trun_0001", "status": "paused"}'), "site_screen_status_invalid"),
    ("status", (404, b""), "site_screen_run_not_found"),
    ("result", (200, b'{"run": {"run_id": "trun_9999", "status": "completed"}, "output": {}}'),
     "site_screen_result_invalid"),
    ("result", (200, b'{"run": {"run_id": "trun_0001", "status": "running"}, "output": {}}'),
     "site_screen_result_invalid"),
    ("result", (408, b""), "site_screen_result_unavailable"),
])
def test_status_and_result_reads_bind_to_their_run(path, answer, code):
    provider = FakeProvider()
    provider.read_answers = [answer]
    with pytest.raises(ss.ScreenError, match=f"^{code}$"):
        getattr(ss.TaskClient(KEY, transport=provider), path)("trun_0001")
    with pytest.raises(ss.ScreenError, match="^site_screen_run_id_invalid$"):
        getattr(ss.TaskClient(KEY, transport=provider), path)("../trun_0001")
    assert len(provider.calls) == 1


# --- quote verification ---------------------------------------------------------------------
URL = "https://operator-1.example/capabilities"
QUOTE = "Operators at the Example Road plant unload the molding presses by hand."


def level(page=None, excerpts=(), quote=QUOTE, url=URL, cited=URL):
    """The level of one quote against one page read of ``url`` and excerpts cited for ``cited``."""
    evidence = {"pages": {url: {"state": "ok", "text": page, "sha256": "0" * 64} if page is not None
                          else {"state": "unreachable", "code": "source_http_failure"}}}
    basis = [{"field": "target_task_quote", "citations": [{"url": cited, "excerpts": list(excerpts)}]}]
    return ss.proof(quote, url, ss.evidence_index(evidence, basis), evidence)["level"]


def test_a_quote_matches_whole_words_up_to_case_spacing_and_punctuation_only():
    assert level(f"Header. {QUOTE} Footer.") == "verified_on_page"
    assert level("Header. " + QUOTE.upper().replace(" ", "\u00a0\n ")) == "verified_on_page"
    page = "It\u2019s the \u201cmain\u201d line \u2014 operators load and unload twelve CNC lathes."
    assert level(page, quote="It's the \"main\" line - operators load and unload twelve CNC lathes.") == "verified_on_page"
    # Review P4: no near-exact rule, so a changed last word or a changed lead never matches.
    assert level(QUOTE.replace("by hand", "by robot")) == "unverified"
    assert level("We do not use manual labor; every Example Road press is unloaded by a robot cell today.",
                 quote="Every Example Road press is unloaded by hand by operators on every single shift today.") == "unverified"
    assert level(QUOTE[:-6]) == "unverified" and level(QUOTE[5:]) == "unverified"
    # Review P5: whole words only.
    assert not ss.has_phrase(ss.words("one plant"), ss.words("We are done planting."))
    assert level("We are done planting at the plant today.", quote="one plant at the plant today") == "unverified"


@pytest.mark.parametrize("quote, expected", [
    ("a", "quote_too_short"), ("the", "quote_too_short"), ("the molding presses by", "quote_too_short"),
    ("the molding presses by hand", "verified_on_page"), ("Operators unload the molding presses", "unverified"),
    ("", "no_quote"),
])
def test_review_p1_a_quote_needs_at_least_five_words(quote, expected):
    assert level(f"Header. {QUOTE} Footer.", quote=quote) == expected


def test_review_p9_an_excerpt_counts_only_for_the_answer_url():
    closed = "Synthetic page. Our plant closed in 2024. Synthetic footer."
    assert level(closed, excerpts=[QUOTE], cited="https://directory.example/listing/1") == "unverified"
    assert level(None, excerpts=[QUOTE], cited="https://directory.example/listing/1") == "unverified_page_unreachable"
    # The same URL, up to scheme, host case, www., a trailing slash and the fragment, holds the excerpt.
    for cited in (URL, "http://www.OPERATOR-1.example/capabilities/", URL + "#scripts"):
        assert level("A page that renders the quote only after its scripts run.", excerpts=[QUOTE], cited=cited) == (
            "in_citation_excerpt")
    assert level(None, excerpts=[QUOTE[:30], QUOTE[30:]]) == "unverified_page_unreachable"


@pytest.mark.parametrize("url", [
    "https://www.linkedin.com/company/synthetic-operator-1", "https://uk.linkedin.com/in/synthetic",
    "https://lnkd.in/synthetic", "https://web.archive.org/web/2025/https://www.linkedin.com/in/synthetic",
    "https://translate.example/?u=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fsynthetic",
])
def test_review_p3_a_linkedin_url_never_counts_directly_or_through_a_wrapper(tmp_path, url):
    answers = screen_answers(1, site_identity_url=url)
    pages = {**pages_for(answers, [stem for stem in ss.SCREEN_PROOFS if stem != "site_identity"]),
             url: answers["site_identity_quote"]}  # It would verify if it were read.
    basis = {ss.from_inventory(inventory_record(1))["site_key"]: [
        {"field": "site_identity_quote", "citations": [{"url": url, "excerpts": [answers["site_identity_quote"]]}]}]}
    _, _, reader, records = screen(tmp_path, [inventory_record(1)], [answers], pages, basis=basis)
    (record,) = records.values()
    assert record["verification"]["site_identity"] == {"level": "source_not_allowed"}
    assert record["tier"] == "screened" and "physical_site_not_verified_fact" in record["blockers"]
    assert url not in reader.requested


def test_both_forms_ask_the_provider_to_leave_linkedin_out():
    site = ss.from_inventory(inventory_record(1))
    for stage in ss.STAGES:
        assert ss.create_body(stage, site, "core")["source_policy"] == {"exclude_domains": ["linkedin.com", "lnkd.in"]}


# --- answers, the tier rule and the one question ---------------------------------------------
def screened(tmp_path, answers, *, record=None, pages=None, basis=None, unpublished=()):
    """The screen record of one site whose answers are published on their pages except ``unpublished``."""
    record = record or inventory_record(1)
    if pages is None:
        pages = pages_for(answers, [stem for stem in ss.SCREEN_PROOFS if stem not in unpublished])
    key = ss.load_sites(raw_input([record]))[0][0]["site_key"]
    *_, records = screen(tmp_path, [record], [answers], pages, basis={key: basis} if basis else None)
    return records[key]


def test_a_web_found_site_with_every_proof_is_outreach_ready_with_one_question(tmp_path):
    record = screened(tmp_path, screen_answers(1))
    assert (record["tier"], record["rule_version"], record["schema_version"]) == (
        "outreach_ready", "blueprint.site-screen-rule.v2", "blueprint.site-screen.v2")
    assert record["blockers"] == [] and record["task_scope"] == "site"
    assert record["gates"]["states"] == {"operator": "verified_fact", "physical_site": "verified_fact",
                                         "site_task": "verified_fact", "human_workflow": "verified_fact",
                                         "plausible_fit": "unresolved", "counterevidence": "unresolved"}
    assert record["question_template"] == "A" and record["question"] == (
        "What has kept the remaining CNC machine tending work at Fixture City from being automated so far?")
    assert record["open_checks"] == ["existing_automation", "freshness", "fit", "interest"]
    assert record["variability"] == {"answer": "high-mix batches", "proven": True}
    assert [source["claim"] for source in record["proving_sources"]] == ["operator", "physical_site", "site_task"]


def test_review_p2_each_quote_must_name_its_answer(tmp_path):
    unrelated = screen_answers(1, operator_identity="Unrelated Holdings LLC", site_identity="999 Nowhere Lane, Elsewhere, ZZ",
                               operator_identity_quote="The plant added a second shift this spring.",
                               operator_identity_url="https://operator-1.example/news",
                               site_identity_quote="The plant added a second shift this spring.",
                               site_identity_url="https://operator-1.example/news")
    record = screened(tmp_path / "unrelated", unrelated, pages=pages_for(screen_answers(1)))
    assert record["verification"]["operator_identity"]["level"] == "verified_on_page"  # Found, but it names nobody.
    assert record["gates"]["states"]["physical_site"] == "unresolved" and record["tier"] == "screened"
    assert record["gates"]["states"]["operator"] == "contradicted" and "operator_mismatch" in record["blockers"]
    no_task_word = screen_answers(1, target_task="Kitting",
                                  target_task_quote="Operators at our Fixture City, TX plant work on every single shift.")
    assert screened(tmp_path / "task", no_task_word)["gates"]["states"]["site_task"] == "unresolved"
    # A legal form never has to be quoted; every other word of the name does.
    assert ss.names_operator("Synthetic Operator 1 runs the plant", "Synthetic Operator 1, Inc.")
    assert not ss.names_operator("Synthetic runs the plant at the end of the road", "Synthetic Operator 1")


def test_a_provider_found_address_that_the_site_quote_proves_anchors_the_site(tmp_path):
    record = inventory_record(1, location="Fixture City, United States; state not individually established")
    found = screen_answers(1, site_identity="44 Mill Road, Fixture City, TX 78701",
                           site_identity_quote="Our plant at 44 Mill Road in Fixture City runs two shifts.",
                           target_task_quote="Operators at 44 Mill Rd load and unload twelve CNC lathes every shift.")
    proven = screened(tmp_path / "found", found, record=record)
    assert proven["gates"]["facts"]["physical_site"]["proofs"][0]["source_id"] == "site_identity"
    assert (proven["task_scope"], proven["tier"]) == ("site", "outreach_ready")
    # The address must be in the quote itself, and in the input's city.
    unquoted = screen_answers(1, site_identity="44 Mill Road, Fixture City, TX 78701",
                              site_identity_quote="Our plant in Fixture City runs two shifts every day.")
    elsewhere = screen_answers(1, site_identity="44 Mill Road, Elsewhere, TX 78701",
                               site_identity_quote="Our plant at 44 Mill Road in Elsewhere runs two shifts.")
    for name, answers in (("unquoted", unquoted), ("elsewhere", elsewhere)):
        assert screened(tmp_path / name, answers, record=record)["gates"]["states"]["physical_site"] == "unresolved"


def test_the_government_record_proves_only_the_site_address(tmp_path):
    blank = {name + suffix: "" for name in ("operator_identity", "site_identity") for suffix in ("", "_url", "_quote")}
    government = screened(tmp_path / "blank", screen_answers(2, **blank), record=universe_row(2))
    assert government["gates"]["facts"]["physical_site"]["proofs"][0]["source_id"] == "government_record"
    assert government["gates"]["states"]["operator"] == "unresolved" and government["tier"] == "screened"
    quoted = screened(tmp_path / "quoted", screen_answers(2, **{name + suffix: "" for suffix in ("", "_url", "_quote")
                                                               for name in ("site_identity",)}), record=universe_row(2))
    assert quoted["tier"] == "outreach_ready" and quoted["gates"]["states"]["physical_site"] == "verified_fact"
    mapped = screened(tmp_path / "mapped", screen_answers(3, **blank), record=universe_row(3, sources=("osm_overpass",)))
    assert mapped["gates"]["states"]["physical_site"] == "unresolved" and mapped["tier"] == "screened"


def test_review_p2b_an_operator_the_input_does_not_name_never_qualifies(tmp_path):
    other = screen_answers(1, operator_identity="Unrelated Holdings LLC", operator_identity_url="https://unrelated.example/about",
                           operator_identity_quote="Unrelated Holdings LLC makes precision parts for many industries.")
    record = screened(tmp_path / "other", other)
    assert record["verification"]["operator_identity"]["level"] == "verified_on_page"
    assert record["gates"]["states"]["operator"] == "contradicted" and record["tier"] == "screened"
    assert "operator_mismatch" in record["blockers"]
    # A legal form, or the input's distinctive words inside a longer name, still match.
    for name in ("Synthetic Operator 1, Inc.", "Synthetic Operator 1 Holdings"):
        same = screen_answers(1, operator_identity=name, operator_identity_quote=(
            f"{name} runs the machining plant on 1 Example Road."))
        assert screened(tmp_path / name[-4:].strip(" ,."), same)["tier"] == "outreach_ready"
    # An input without an operator names no one to compare with.
    unnamed = screened(tmp_path / "unnamed", other, record=inventory_record(1, operator=None))
    assert "operator_mismatch" not in unnamed["blockers"] and unnamed["tier"] == "outreach_ready"


def test_a_company_level_task_never_qualifies_and_asks_the_site_question(tmp_path):
    company = screen_answers(1, target_task_quote="Our team loads and unloads CNC lathes for every customer order.")
    record = screened(tmp_path / "company", company)
    assert (record["task_scope"], record["tier"], record["question_template"]) == ("company", "screened", "S")
    assert "company_level_task" in record["blockers"]
    assert record["question"] == "Is CNC machine tending done at your Fixture City site, or somewhere else in the company?"
    # The page or a same-URL excerpt naming the city with its state, or the street, ties the quote to the site.
    page = {company["target_task_url"]: "Fixture City, TX plant careers. " + company["target_task_quote"]}
    assert screened(tmp_path / "page", company, pages={**pages_for(company), **page})["task_scope"] == "site"
    basis = [{"field": "target_task_quote", "citations": [{"url": company["target_task_url"], "excerpts": [
        "Lathe operator, Fixture City, TX. " + company["target_task_quote"]]}]}]
    assert screened(tmp_path / "excerpt", company, basis=basis)["task_scope"] == "site"
    # A site universe row also has a street, which ties the task as well as the city does.
    street = {company["target_task_url"]: "Lathe operator at 1 Example Rd. " + company["target_task_quote"]}
    assert screened(tmp_path / "street", company, record=universe_row(1),
                    pages={**pages_for(company), **street})["task_scope"] == "site"


@pytest.mark.parametrize("changes, blocker", [
    ({"facility_type": "office"}, "physical_site_contradicted"),
    ({"facility_type": "mailing address"}, "physical_site_contradicted"),
    ({"operating_now": "no"}, "physical_site_contradicted"),
    ({"operating_now": "Closed for a retool"}, "physical_site_contradicted"),
    ({"manual_today": "no", "manual_today_quote": "No operator loads the lathes; a gantry loader does every part."},
     "human_workflow_contradicted"),
    ({"existing_automation": "full", "existing_automation_url": "https://operator-1.example/robots",
      "existing_automation_quote": "Robots load and unload every CNC lathe in the Fixture City plant."},
     "counterevidence_contradicted"),
])
def test_a_proven_contradiction_blocks_the_tier(tmp_path, changes, blocker):
    record = screened(tmp_path, screen_answers(1, **changes))
    assert record["tier"] == "screened" and blocker in record["blockers"]


@pytest.mark.parametrize("changes", [
    {"facility_type": "office"},
    {"manual_today": "no", "manual_today_quote": "No operator loads the lathes; a gantry loader does every part."},
    {"existing_automation": "full", "existing_automation_url": "https://operator-1.example/robots",
     "existing_automation_quote": "Robots load and unload every CNC lathe in the Fixture City plant."},
])
def test_review_p8_an_unproven_contradiction_never_blocks(tmp_path, changes):
    stem = next(name for name in ("facility_type", "manual_today", "existing_automation") if name in changes)
    record = screened(tmp_path, screen_answers(1, **changes), unpublished=(stem,))
    assert record["tier"] == "outreach_ready", record["blockers"]


def test_partial_automation_keeps_the_site_and_asks_question_a(tmp_path):
    partial = screen_answers(1, existing_automation="partial", existing_automation_url="https://operator-1.example/robots",
                             existing_automation_quote="A robot cell tends two of the twelve lathes at the plant.")
    record = screened(tmp_path, partial)
    assert (record["tier"], record["question_template"], record["gates"]["states"]["counterevidence"]) == (
        "outreach_ready", "A", "checked")


def test_an_unproven_manual_workflow_asks_question_m(tmp_path):
    record = screened(tmp_path, screen_answers(1, manual_today="unknown", manual_today_url="", manual_today_quote=""))
    assert (record["tier"], record["question_template"]) == ("outreach_ready", "M")
    assert record["question"] == ("Which parts of CNC machine tending at Fixture City still need people, and what has "
                                  "kept them from being automated?")
    assert record["open_checks"] == ["manual_workflow", "existing_automation", "freshness", "fit", "interest"]


@pytest.mark.parametrize("attributed, blocked", [("Synthetic Operator 1", False), ("Unrelated Holdings LLC", True),
                                                 (None, False)])
def test_a_contractor_site_blocks_only_when_the_input_names_another_operator(tmp_path, attributed, blocked):
    contractor = screen_answers(1, facility_operator="contractor", facility_operator_quote=(
        "Synthetic Operator 1 runs the Fixture City plant under contract for a customer."))
    record = screened(tmp_path, contractor, record=inventory_record(1, operator=attributed))
    assert ("operator_contradicted" in record["blockers"]) is blocked
    assert (record["tier"] == "outreach_ready") is not blocked


def test_variability_signals_are_recorded_and_never_required(tmp_path):
    record = screened(tmp_path, screen_answers(1, variability_signals="", variability_signals_url="",
                                               variability_signals_quote=""))
    assert record["tier"] == "outreach_ready" and record["variability"] == {"answer": None, "proven": False}


def test_screen_gates_have_the_lead_verification_shape_and_a_url_keyed_index(tmp_path):
    record = screened(tmp_path, screen_answers(1))
    gates = record["gates"]
    assert set(gates) >= {"eligible_for_qualified_promotion", "assessment_valid", "identity_present", "duplicate",
                          "conflict", "valid_until", "states", "facts", "task", "site"}
    assert set(gates["states"]) == {*ss.CLAIMS, "counterevidence"} and set(gates["facts"]) == set(ss.PROVEN_FACTS)
    proof = gates["facts"]["operator"]["proofs"][0]
    assert set(proof) == {"claim", "source_id", "url", "quote_sha256", "level", "tool_result_sha256"}
    assert proof["quote_sha256"] == hashlib.sha256(screen_answers(1)["operator_identity_quote"].encode()).hexdigest()
    evidence = {"pages": {"https://www.operator-1.example/about/": {"state": "ok", "text": "Text.", "sha256": "1" * 64}}}
    index = ss.evidence_index(evidence, [{"field": "x", "citations": [
        {"url": "https://operator-1.example/about", "excerpts": ["Excerpt."]},
        {"url": "https://www.linkedin.com/company/synthetic", "excerpts": ["Never kept."]}]}])
    assert set(index["pages"]) == set(index["excerpts"]) == {"operator-1.example/about"}
    assert ss.outreach_tier({})["tier"] == "none"  # A defect yields none, never outreach_ready.


def test_each_page_is_read_once_per_verify_and_kept_for_recomputation(tmp_path):
    shared = "https://operator-1.example/plant"
    answers = screen_answers(1, operator_identity_url=shared, site_identity_url=shared)
    pages = pages_for(answers)
    workspace, _, reader, records = screen(tmp_path, [inventory_record(1)], [answers], pages)
    assert sorted(reader.requested) == sorted(set(ss.cited_urls("screen", answers)))
    (key,) = records
    evidence = json.loads(workspace.path("screen", "evidence", key).read_text())
    assert evidence["checked_on"] == TODAY.isoformat() and set(evidence["pages"]) == set(reader.requested)


def test_records_carry_their_rule_and_are_recomputed_without_a_read_or_a_run(tmp_path, monkeypatch):
    answers = screen_answers(1)
    workspace, provider, reader, records = screen(tmp_path, [inventory_record(1)], [answers], pages_for(answers))
    (key,) = records
    current = workspace.path("screen", "records", key).with_name(f"{key}.site-screen-rule.v2.json")
    assert current.exists() and json.loads(current.read_text())["rule_version"] == ss.SCREEN_RULE
    calls, reads = len(provider.calls), len(reader.requested)
    current.unlink()
    assert ss.summary(workspace)["screen"]["tiers"] == {"outreach_ready": 1, "screened": 0}
    monkeypatch.setattr(ss, "SCREEN_RULE", "blueprint.site-screen-rule.v3")
    result = ss.verify(workspace, reader=FakePages({}), today=TODAY)
    assert (result["screen"]["records"], result["screen"]["pages_kept"], result["page_reads"]) == (1, 0, 0)
    assert json.loads(current.with_name(f"{key}.site-screen-rule.v3.json").read_text())["rule_version"] == (
        "blueprint.site-screen-rule.v3")
    assert (len(provider.calls), len(reader.requested)) == (calls, reads)


def test_summary_counts_fields_claims_tiers_questions_and_cost_without_site_data(tmp_path):
    ready, closed = screen_answers(1), screen_answers(2, operating_now="no")
    pages = {**pages_for(ready), **pages_for(closed)}
    workspace, *_ = screen(tmp_path, [inventory_record(1), universe_row(2)], [ready, closed], pages)
    report = ss.summary(workspace)
    screen_report = report["screen"]
    assert screen_report["runs"] == {"sites": 2, "created": 2, "outcome_unknown": 0, "refused_not_created": 0,
                                     "completed": 2, "failed": 0, "cancelled": 0, "in_flight": 0}
    assert screen_report["tiers"] == {"outreach_ready": 1, "screened": 1} and screen_report["pages_not_read"] == 0
    assert screen_report["tiers_by_origin"] == {"discovery_inventory": {"outreach_ready": 1, "screened": 0},
                                                "site_universe": {"outreach_ready": 0, "screened": 1}}
    assert screen_report["tiers_by_calibration"]["ranked"] == {"outreach_ready": 1, "screened": 1}
    assert screen_report["fields"]["operating_now"] == {"answers": {"yes": 1, "no": 1}, "levels": {"verified_on_page": 2}}
    assert screen_report["physical_site_basis"] == {"site_identity": 1, "government_record": 1}
    assert screen_report["claims"]["physical_site"] == {"verified_fact": 1, "contradicted": 1}
    assert screen_report["questions"] == {"A": 1} and screen_report["variability_proven"] == 2
    assert (report["estimated_cost_usd"], report["committed_usd"], report["rules"]["screen"]) == (
        "0.050", "0.050", ss.SCREEN_RULE)
    assert json.loads((workspace.root / "summary.json").read_text()) == report
    assert not any(value in ss.canonical(report) for value in SITE_STRINGS)


# --- operator command, key and out dir ------------------------------------------------------
def test_an_out_dir_inside_a_repository_is_refused(tmp_path, capsys):
    for path in (ROOT, ROOT / "output" / "site-screen-synthetic", ROOT / "tools" / "daily_research"):
        with pytest.raises(ss.ScreenError, match="^site_screen_out_dir_inside_repository$"):
            ss.Workspace(path, create=True)
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    with pytest.raises(ss.ScreenError, match="^site_screen_out_dir_inside_repository$"):
        ss.Workspace(checkout / "private" / "out", create=True)
    source = tmp_path / "sites.json"
    source.write_bytes(raw_input([inventory_record(1)]))
    with pytest.raises(ss.ScreenError, match="^site_screen_out_dir_inside_repository$"):
        operator.main(["run", "--input", str(source), "--out", str(ROOT / "output" / "site-screen-synthetic"),
                       "--owner-reference", OWNER, "--ceiling-usd", "1", "--max-runs", "1", "--apply"], environ={"PARALLEL_API_KEY": KEY},
                      transport=FakeProvider())
    assert not (ROOT / "output" / "site-screen-synthetic").exists() and not (checkout / "private").exists()
    assert ss.Workspace(tmp_path / "private" / "out", create=True).root.is_dir()


def test_the_key_comes_from_the_environment_or_a_key_file(tmp_path):
    assert operator.api_key(environ={"PARALLEL_API_KEY": KEY}) == KEY
    key_file = tmp_path / "private.env"
    key_file.write_text(f"# owner-only\nOTHER_KEY=unrelated\nexport PARALLEL_API_KEY=\"{KEY}\"\n")
    assert operator.api_key(key_file, environ={"PARALLEL_API_KEY": "stale-shell-value"}) == KEY
    with pytest.raises(ss.ScreenError, match="^site_screen_api_key_missing$"):
        operator.api_key(environ={})
    key_file.write_text("# PARALLEL_API_KEY=commented-out\n")
    with pytest.raises(ss.ScreenError, match="^site_screen_api_key_missing$"):
        operator.api_key(key_file, environ={"PARALLEL_API_KEY": KEY})
    with pytest.raises(ss.ScreenError, match="^site_screen_key_file_unreadable$"):
        operator.api_key(tmp_path / "missing.env", environ={})
    with pytest.raises(ss.ScreenError, match="^site_screen_api_key_invalid$") as refusal:
        ss.TaskClient("a key with spaces")
    assert "spaces" not in str(refusal.value)
    source = tmp_path / "sites.json"
    source.write_bytes(raw_input([inventory_record(1)]))
    with pytest.raises(ss.ScreenError, match="^site_screen_api_key_missing$"):
        operator.main(["run", "--input", str(source), "--out", str(tmp_path / "out"), "--owner-reference", OWNER, "--ceiling-usd", "1",
                       "--max-runs", "1"], environ={})
    assert not (tmp_path / "out").exists()


def test_the_command_prints_counts_only_and_the_key_never_leaves_the_client(tmp_path, capsys):
    records = [inventory_record(1), universe_row(2)]
    answers = [screen_answers(1), screen_answers(2)]
    source, out = tmp_path / "sites.json", tmp_path / "out"
    source.write_bytes(raw_input(records))
    provider = FakeProvider()
    for site, content in zip(ss.load_sites(raw_input(records))[0], answers):
        provider.outputs[site["site_key"]] = {"content": content, "basis": []}
    environ = {"PARALLEL_API_KEY": KEY}
    spend = ["--out", str(out), "--owner-reference", OWNER, "--ceiling-usd", "0.05", "--max-runs", "2"]
    assert operator.main(["plan", "--input", str(source)])["sites"] == 2
    assert operator.main(["run", "--input", str(source), *spend], environ=environ, transport=provider)["state"] == "planned"
    assert provider.calls == []
    assert operator.main(["run", "--input", str(source), *spend, "--apply"], environ=environ,
                         transport=provider)["created"] == 2
    assert operator.main(["collect", "--out", str(out), "--wait-seconds", "0"], environ=environ,
                         transport=provider)["state"] == "complete"
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    assert operator.main(["verify", "--out", str(out)], reader=FakePages(pages), today=TODAY)["screen"]["records"] == 2
    assert operator.main(["summary", "--out", str(out)])["screen"]["tiers"] == {"outreach_ready": 2, "screened": 0}
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 6 and all(json.loads(line) for line in output.splitlines())
    assert KEY not in output and not any(value in output for value in SITE_STRINGS)
    assert {call["headers"]["x-api-key"] for call in provider.calls} == {KEY}
    assert not [path for path in out.rglob("*") if path.is_file() and KEY.encode() in path.read_bytes()]
    assert KEY not in repr(ss.TaskClient(KEY))


# --- packaging and the shared reader ----------------------------------------------------------
def test_site_screen_imports_only_the_standard_library_and_its_own_subtree():
    tree = ast.parse((ROOT / "tools/daily_research/site_screen.py").read_text())
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)
            and node.module != "tools.daily_research.verification"}
    assert top <= set(sys.stdlib_module_names)
    later = {(node.module, alias.name) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
             and node not in tree.body for alias in node.names}
    assert later == {("tools.daily_research", "site_universe"), ("tools.daily_research", "search")}
    for name in ("site_universe.py", "search.py", "verification.py"):  # Standard library only at import too.
        imported = ast.parse((ROOT / "tools/daily_research" / name).read_text()).body
        modules = {alias.name.split(".")[0] for node in imported if isinstance(node, ast.Import) for alias in node.names}
        modules |= {node.module.split(".")[0] for node in imported if isinstance(node, ast.ImportFrom)}
        assert modules <= set(sys.stdlib_module_names), name


def test_the_release_packages_the_site_screen_and_its_command():
    for name in ("site_screen.py", "operators/site-screen.py"):
        assert name in FILES and (ROOT / "tools/daily_research" / name).is_file()


def test_the_default_page_reader_is_the_daily_reader_under_its_alarm(monkeypatch):
    seen, alarms = [], []

    @contextmanager
    def bounded(seconds):
        alarms.append(seconds)
        yield

    monkeypatch.setattr(search, "bounded_request", bounded)
    monkeypatch.setattr(search, "source", lambda arguments, **options: seen.append((arguments, options)) or {
        "text": "Synthetic page.", "links": ["/x"], "raw_sha256": "0" * 64})
    read = ss.public_page_reader()
    assert read("https://operator-1.example/plant") == {"text": "Synthetic page."}
    assert seen == [({"url": "https://operator-1.example/plant"}, {"blocked_domains": ss.NEVER_FETCH})]
    assert alarms == [ss.PAGE_READ_SECONDS]
