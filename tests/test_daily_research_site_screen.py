"""Hermetic site screen line: fake Parallel Task API, fake page reader, synthetic sites only. No network."""
import ast
import hashlib
import importlib.util
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from tests.daily_research_site_screen_fixture import (
    KEY,
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


def workspace_and_client(tmp_path, provider=None):
    provider = provider or FakeProvider()
    return ss.Workspace(tmp_path / "out", create=True), provider, ss.TaskClient(KEY, transport=provider)


def kinds(workspace, stage="screen"):
    return [event["event"] for event in workspace.ledger(stage).events()]


# --- form and inputs ------------------------------------------------------------------------
def test_screen_form_is_versioned_strict_and_keeps_the_pilot_fields():
    schema = ss.SCREEN_SCHEMA
    identity = {name + suffix for name in ("operator_identity", "site_identity") for suffix in ("", "_url", "_quote")}
    assert set(schema["properties"]) == PILOT_FIELDS | identity
    assert schema["required"] == list(schema["properties"]) and schema["additionalProperties"] is False
    assert all(set(item) == {"type", "description"} and item["type"] == "string" for item in schema["properties"].values())
    assert ss.FORMS["screen"]["sha256"] == hashlib.sha256(ss.canonical(schema).encode()).hexdigest()
    site = ss.from_inventory(inventory_record(1))
    body = ss.create_body("screen", site, "core")
    assert body == {"processor": "core", "input": site["task_input"],
                    "metadata": {"site_key": site["site_key"], "form": "blueprint.site-screen.v1"},
                    "task_spec": {"output_schema": {"type": "json", "json_schema": schema}}}
    # The provider's metadata limits: string keys of at most 16 and values of at most 512 characters.
    assert all(len(key) <= 16 and isinstance(value, str) and len(value) <= 512 for key, value in body["metadata"].items())
    assert ss.INVENTORY_VERSION == discovery.INVENTORY_VERSION


def test_an_inventory_record_is_a_web_found_site_with_a_stable_key():
    site = ss.from_inventory(inventory_record(1))
    assert site == {"schema_version": ss.INPUT, "site_key": site["site_key"], "origin": "discovery_inventory",
                    "identity": {}, "task_input": {"site_name": "Synthetic Works 1", "operator": "Synthetic Operator 1",
                                                   "location": "Fixture City, TX", "task_hint": "CNC machine tending",
                                                   "known_source_urls": ["https://operator-1.example/plant"]}}
    assert ss.SHA.fullmatch(site["site_key"])
    # Spacing, case and punctuation do not change the key, and another task hypothesis is the same site.
    same = inventory_record(1, operator=" synthetic  OPERATOR 1. ", task_hypothesis="Kitting")
    assert ss.from_inventory(same)["site_key"] == site["site_key"]
    assert ss.from_inventory(inventory_record(2))["site_key"] != site["site_key"]
    # A record about a site universe site shares that site's key.
    assert ss.from_inventory(inventory_record(1, site_universe_id="a" * 64))["site_key"] == "a" * 64


@pytest.mark.parametrize("changes, code", [
    ({"disposition": "rejected"}, "site_screen_input_rejected"),
    ({"disposition": "learning"}, "site_screen_input_learning"),
    ({"disposition": "duplicate"}, "site_screen_input_duplicate"),
    ({"operator": None, "site": None}, "site_screen_input_identity_missing"),
    ({"location": "  "}, "site_screen_input_location_missing"),
    ({"source_urls": ["ftp://operator-1.example/plant"]}, "site_screen_input_source_urls_invalid"),
    ({"source_urls": "https://operator-1.example/plant"}, "site_screen_input_source_urls_invalid"),
    ({"operator": 7}, "site_screen_input_record_invalid"),
    ({"location": ...}, "site_screen_input_record_invalid"),
])
def test_inventory_records_that_cannot_be_screened_are_refused(changes, code):
    record = inventory_record(1, **changes)
    record = {key: value for key, value in record.items() if value is not ...}
    with pytest.raises(ss.ScreenError, match=f"^{code}$"):
        ss.from_inventory(record)


def test_a_site_universe_row_takes_operator_and_site_from_its_government_record():
    row = universe_row(3)
    site = ss.from_site_universe(row)
    record = {"source": "government_record", "source_ids": ["epa_frs", "osha_ita"], "site_id": row["site_id"]}
    assert site == {"schema_version": ss.INPUT, "site_key": row["site_id"], "origin": "site_universe",
                    "identity": {"operator": {**record, "answer": "Synthetic Operator 3"},
                                 "physical_site": {**record, "answer": "3 Example Road, Fixture City, TX 00003"}},
                    "task_input": {"site_name": "Synthetic Works 3", "operator": "Synthetic Operator 3",
                                   "location": "3 Example Road, Fixture City, TX 00003",
                                   "task_hint": "fixed arm machine tending", "naics": "332710"}}


def test_the_government_record_needs_an_osha_or_epa_source_and_a_street_for_the_site():
    for sources in (("osm_overpass",), ("fsis_mpi", "osm_overpass")):
        assert ss.from_site_universe(universe_row(3, sources=sources))["identity"] == {}
    no_street = ss.from_site_universe(universe_row(3, street=None))
    assert set(no_street["identity"]) == {"operator"}
    assert no_street["task_input"]["location"] == "Fixture City, TX 00003"
    # EPA FRS names the facility only, so the facility name stands for the operator; all source ids are kept.
    facility = ss.from_site_universe(universe_row(3, sources=("epa_frs", "osm_overpass"), operator=None))
    assert facility["identity"]["operator"]["answer"] == "Synthetic Works 3"
    assert facility["identity"]["operator"]["source_ids"] == ["epa_frs", "osm_overpass"]
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
        "by_origin": {"discovery_inventory": 2, "site_universe": 1},
        "government_record": {"operator": 1, "physical_site": 1},
        "input_refused": {"site_screen_input_location_missing": 1}, "processor": "core", "price_usd": "0.025",
        "estimated_cost_usd": "0.075", "provider_calls": 0}
    with pytest.raises(ss.ScreenError, match="^site_screen_processor_price_unknown$"):
        ss.plan(raw, processor="ultra")


# --- spend: idempotency, ceiling and max_runs -----------------------------------------------
def test_a_dry_run_admits_like_apply_and_sends_nothing(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in range(1, 6)])
    result = ss.run(raw, workspace, client=client, ceiling_usd="0.075", max_runs=10)
    assert result["state"] == "stopped" and result["stop"] == "site_screen_spend_ceiling_reached"
    assert result["would_create"] == 3 and result["created"] == 0 and result["committed_usd"] == "0.075"
    assert provider.calls == [] and not (workspace.root / "screen").exists()


def test_each_site_is_created_once_and_a_rerun_creates_nothing(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(1), universe_row(2)])
    sites, _ = ss.load_sites(raw)
    first = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
    assert (first["state"], first["created"], first["runs"], first["committed_usd"]) == ("complete", 2, 2, "0.050")
    assert provider.creates() == [ss.create_body("screen", site, "core") for site in sites]
    assert {call["headers"]["x-api-key"] for call in provider.calls} == {KEY}
    second = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
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
    first = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
    # The first unknown outcome stops the run, and its price stays committed.
    assert first["state"] == "stopped" and first["stop"].startswith("site_screen_create_")
    assert (first["outcome_unknown"], first["created"], first["committed_usd"]) == (1, 0, "0.025")
    second = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
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
        ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
    assert kinds(workspace) == ["intent"]
    again = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
    assert (again["outcome_unknown_kept_out"], again["created"], again["committed_usd"]) == (1, 1, "0.050")
    assert len(provider.creates()) == 2


def test_a_refused_create_costs_nothing_and_is_tried_again_later(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    provider.create_answers = [(422, b'{"detail": "validation"}')]
    raw = raw_input([inventory_record(1)])
    first = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
    assert (first["state"], first["refused"], first["runs"], first["committed_usd"]) == (
        "complete", {"site_screen_create_rejected": 1}, 0, "0")
    second = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
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
                    ceiling_usd="1", max_runs=10, apply=True)
    assert (result["state"], result["stop"], result["refused"], result["created"]) == ("stopped", code, {code: 1}, 0)
    assert len(provider.calls) == 1 and result["committed_usd"] == "0"


def test_the_ceiling_and_max_runs_refuse_before_any_create(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in (1, 2, 3)])
    below = ss.run(raw, workspace, client=client, ceiling_usd="0.02", max_runs=10, apply=True)
    assert (below["stop"], below["created"]) == ("site_screen_spend_ceiling_reached", 0)
    assert provider.calls == [] and not workspace.ledger("screen").path.exists()
    one = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=1, apply=True)
    assert (one["created"], one["stop"]) == (1, "site_screen_max_runs_reached")
    again = ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=1, apply=True)
    assert (again["created"], again["already_created"], again["stop"]) == (0, 1, "site_screen_max_runs_reached")
    assert len(provider.creates()) == 1
    exact = ss.run(raw, workspace, client=client, ceiling_usd="0.075", max_runs=10, apply=True)
    assert (exact["created"], exact["state"], exact["committed_usd"]) == (2, "complete", "0.075")


def test_the_ceiling_counts_runs_in_flight_and_frees_only_failed_ones(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    raw = raw_input([inventory_record(number) for number in (1, 2, 3)])
    sites, _ = ss.load_sites(raw)
    first = ss.run(raw, workspace, client=client, ceiling_usd="0.05", max_runs=10, apply=True)
    assert (first["created"], first["stop"]) == (2, "site_screen_spend_ceiling_reached")
    provider.final[sites[0]["site_key"]] = "failed"
    provider.final[sites[1]["site_key"]] = "cancelled"  # Not known to be free, so its price stays committed.
    provider.outputs[sites[1]["site_key"]] = {"content": {}, "basis": []}
    ss.collect(workspace, client=client, wait_seconds=0)
    second = ss.run(raw, workspace, client=client, ceiling_usd="0.05", max_runs=10, apply=True)
    assert (second["created"], second["committed_usd"]) == (1, "0.050")
    # A failed run is free but still a run: max_runs counts it.
    capped = ss.run(raw_input([inventory_record(4)]), workspace, client=client, ceiling_usd="1", max_runs=3,
                    apply=True)
    assert capped["stop"] == "site_screen_max_runs_reached"


@pytest.mark.parametrize("ceiling", ["0", "-1", "nan", "inf", "100.01", "a dollar", ""])
def test_a_ceiling_outside_the_reviewed_bound_is_refused(tmp_path, ceiling):
    workspace, provider, client = workspace_and_client(tmp_path)
    with pytest.raises(ss.ScreenError, match="^site_screen_ceiling_invalid$"):
        ss.run(raw_input([inventory_record(1)]), workspace, client=client, ceiling_usd=ceiling, max_runs=1, apply=True)
    assert provider.calls == []


@pytest.mark.parametrize("max_runs", [0, -1, 5001, True, 1.0])
def test_max_runs_outside_the_reviewed_bound_is_refused(tmp_path, max_runs):
    workspace, provider, client = workspace_and_client(tmp_path)
    with pytest.raises(ss.ScreenError, match="^site_screen_max_runs_invalid$"):
        ss.run(raw_input([inventory_record(1)]), workspace, client=client, ceiling_usd="1", max_runs=max_runs,
               apply=True)
    assert provider.calls == []


def test_a_second_command_on_the_same_out_dir_refuses(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    with workspace.lock():
        with pytest.raises(ss.ScreenError, match="^site_screen_out_dir_busy$"):
            ss.run(raw_input([inventory_record(1)]), workspace, client=client, ceiling_usd="1", max_runs=1, apply=True)
    assert provider.calls == []


def test_a_torn_final_ledger_line_is_sealed_and_other_damage_refuses(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    ss.run(raw_input([inventory_record(1)]), workspace, client=client, ceiling_usd="1", max_runs=5, apply=True)
    ledger = workspace.ledger("screen")
    with open(ledger.path, "ab") as handle:  # A crash in the middle of a write.
        handle.write(b'{"schema_version":"blueprint.site-screen.led')
    assert kinds(workspace) == ["intent", "created"]
    ss.run(raw_input([inventory_record(2)]), workspace, client=client, ceiling_usd="1", max_runs=5, apply=True)
    assert kinds(workspace) == ["intent", "created", "intent", "created"]
    assert ledger.path.read_bytes().count(b'"event":"sealed"') == 1
    ledger.path.write_bytes(ledger.path.read_bytes().replace(b'"event":"created"', b'"event":"made"', 1))
    with pytest.raises(ss.ScreenError, match="^site_screen_ledger_invalid$"):
        ss.run(raw_input([inventory_record(3)]), workspace, client=client, ceiling_usd="1", max_runs=5, apply=True)
    assert len(provider.creates()) == 2


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
    ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=10, apply=True)
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
    ss.run(raw, workspace, client=client, ceiling_usd="1", max_runs=1, apply=True)
    clock = Clock()
    result = ss.collect(workspace, client=client, wait_seconds=30, poll_seconds=15, monotonic=clock.monotonic,
                        sleep=clock.sleep)
    assert (result["state"], result["still_running"], clock.sleeps) == ("pending", 1, [15, 15])
    with pytest.raises(ss.ScreenError, match="^site_screen_wait_invalid$"):
        ss.collect(workspace, client=client, wait_seconds=7201)


def test_collect_stops_on_an_auth_refusal_and_counts_other_read_errors(tmp_path):
    workspace, provider, client = workspace_and_client(tmp_path)
    ss.run(raw_input([inventory_record(1)]), workspace, client=client, ceiling_usd="1", max_runs=1, apply=True)
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
QUOTE = "Operators load and unload twelve CNC lathes on every shift at the plant."
URL = "https://operator-1.example/capabilities"


def level(page=None, notes=(), quote=QUOTE, url=URL):
    pages = ss.Pages(FakePages({} if page is None else {URL: page}))
    return ss.quote_level(url, quote, [ss.normalize(note) for note in notes], pages)


def test_a_quote_on_its_page_is_verified_after_normalization():
    assert level(f"Header. {QUOTE} Footer.") == ("verified_on_page", None)
    # Case, no-break spaces and line breaks do not matter.
    assert level("Header. " + QUOTE.upper().replace(" ", "\u00a0\n ")) == ("verified_on_page", None)
    # Curly quotes and dashes match their plain forms.
    page = "Header. It\u2019s the \u201cmain\u201d line \u2014 operators load and unload twelve CNC lathes."
    quote = "It's the \"main\" line - operators load and unload twelve CNC lathes."
    assert level(page, quote=quote) == ("verified_on_page", None)


def test_near_exact_needs_90_percent_of_a_quote_of_40_or_more_characters():
    span = math.ceil(len(QUOTE) * 0.9)
    assert len(QUOTE) >= 40
    assert level(QUOTE[:span]) == ("verified_on_page", None)
    assert level(QUOTE[-span:]) == ("verified_on_page", None)
    assert level(QUOTE[:span - 1]) == ("unverified", None)
    short = "Operators unload twelve CNC lathes."
    assert len(short) < 40 and level(short[:-1], quote=short) == ("unverified", None)


def test_the_citation_excerpt_level_applies_only_when_our_read_does_not_find_the_quote():
    assert level(None, notes=[f"... {QUOTE} ..."]) == ("in_citation_excerpt", None)
    assert level("A page that shows the quote only after its scripts run.", notes=[QUOTE]) == (
        "in_citation_excerpt", None)
    assert level(f"Header. {QUOTE}", notes=["Unrelated excerpt."]) == ("verified_on_page", None)
    # Each excerpt is matched alone: a quote split across two excerpts is not in one.
    assert level(None, notes=[QUOTE[:30], QUOTE[30:]]) == ("unverified_page_unreachable", "source_http_failure")
    assert level("A page without it.") == ("unverified", None)
    assert level(None) == ("unverified_page_unreachable", "source_http_failure")
    assert level(f"Header. {QUOTE}", quote="") == ("no_quote", None)
    assert level(f"Header. {QUOTE}", url="") == ("no_quote", None)


def test_excerpts_come_only_from_the_basis_entries_of_that_field(tmp_path):
    answers = screen_answers(1)
    pages = pages_for(answers, [stem for stem in ss.SCREEN_PROOFS if stem != "target_task"])
    key = ss.from_inventory(inventory_record(1))["site_key"]
    citation = {"url": answers["target_task_url"], "excerpts": [answers["target_task_quote"]]}
    other = {key: [{"field": "manual_today", "citations": [citation]}]}
    *_, records = screen(tmp_path / "other", [inventory_record(1)], [answers], pages, basis=other)
    assert records[key]["verification"]["target_task"] == {"level": "unverified_page_unreachable",
                                                            "read": "source_http_failure"}
    own = {key: [{"field": "target_task_quote", "citations": [citation], "reasoning": "", "confidence": "high"}]}
    *_, records = screen(tmp_path / "own", [inventory_record(1)], [answers], pages, basis=own)
    assert records[key]["verification"]["target_task"] == {"level": "in_citation_excerpt"}
    assert records[key]["tier"] == "outreach_ready"


def test_each_page_is_read_once_per_verify_and_never_from_linkedin(tmp_path):
    shared = "https://operator-1.example/plant"
    answers = screen_answers(1, operator_identity_url=shared, site_identity_url=shared,
                             manual_today_url="https://www.linkedin.com/jobs/view/synthetic")
    pages = {shared: answers["operator_identity_quote"] + " " + answers["site_identity_quote"],
             **pages_for(answers, ["operating_now", "target_task"])}
    key = ss.from_inventory(inventory_record(1))["site_key"]
    citation = {"url": answers["manual_today_url"], "excerpts": [answers["manual_today_quote"]]}
    _, _, reader, records = screen(tmp_path, [inventory_record(1)], [answers], pages,
                                   basis={key: [{"field": "manual_today", "citations": [citation]}]})
    assert sorted(reader.requested) == sorted({shared, answers["operating_now_url"], answers["target_task_url"]})
    # The screen never reads LinkedIn; the provider's own excerpt can still place the quote.
    assert records[key]["verification"]["manual_today"] == {"level": "in_citation_excerpt"}


# --- the outreach-ready rule and open questions ---------------------------------------------
def test_a_web_found_site_with_every_proof_is_outreach_ready(tmp_path):
    answers = screen_answers(1)
    *_, records = screen(tmp_path, [inventory_record(1)], [answers], pages_for(answers))
    (record,) = records.values()
    assert record["tier"] == "outreach_ready" and record["schema_version"] == ss.SCREEN
    assert record["checks"] == {
        "operator": {"passed": True, "basis": "web_quote", "level": "verified_on_page"},
        "physical_site": {"passed": True, "basis": "web_quote", "level": "verified_on_page"},
        "site_task": {"passed": True, "level": "verified_on_page"},
        "not_closed": {"passed": True, "answer": "yes"},
        "no_existing_automation": {"passed": True, "answer": "unknown"}}
    assert record["input"] == ss.from_inventory(inventory_record(1))["task_input"] and record["answers"] == answers
    assert [question["check"] for question in record["open_questions"]] == ["existing_automation", "fit", "interest"]
    assert all(question["question"].endswith("?") for question in record["open_questions"])


@pytest.mark.parametrize("changes, unpublished, failed", [
    ({}, "operator_identity", "operator"),
    ({"operator_identity": ""}, None, "operator"),
    ({}, "site_identity", "physical_site"),
    ({}, "target_task", "site_task"),
    ({"target_task_found": "unknown"}, None, "site_task"),
    ({"target_task": ""}, None, "site_task"),
    ({"operating_now": "No"}, None, "not_closed"),
    ({"operating_now": "Closed for a retool"}, None, "not_closed"),
    ({"existing_automation": "yes"}, None, "no_existing_automation"),
    ({"existing_automation": "Partly, on one line"}, None, "no_existing_automation"),
])
def test_each_unproven_or_blocking_check_keeps_a_site_screened(tmp_path, changes, unpublished, failed):
    answers = screen_answers(1, **changes)
    pages = pages_for(answers, [stem for stem in ss.SCREEN_PROOFS if stem != unpublished])
    *_, records = screen(tmp_path, [inventory_record(1)], [answers], pages)
    (record,) = records.values()
    assert record["tier"] == "screened"
    assert [name for name, check in record["checks"].items() if not check["passed"]] == [failed]


def test_the_government_record_proves_operator_and_site_on_site_universe_rows(tmp_path):
    blank = {name + suffix: "" for name in ("operator_identity", "site_identity") for suffix in ("", "_url", "_quote")}
    rows = [universe_row(2), universe_row(3, sources=("osm_overpass",))]
    answers = [screen_answers(2, **blank), screen_answers(3, **blank)]
    pages = {**pages_for(answers[0]), **pages_for(answers[1])}
    *_, records = screen(tmp_path, rows, answers, pages)
    government, mapped = records[rows[0]["site_id"]], records[rows[1]["site_id"]]
    assert government["tier"] == "outreach_ready"
    assert government["checks"]["operator"] == government["checks"]["physical_site"] == {
        "passed": True, "basis": "government_record"}
    assert government["identity"]["operator"]["source"] == "government_record"
    # A row without an OSHA or EPA record needs web proof like any web-found site.
    assert mapped["tier"] == "screened" and not mapped["checks"]["operator"]["passed"]
    assert not mapped["checks"]["physical_site"]["passed"]


def test_open_questions_carry_what_the_rule_leaves_unproven(tmp_path):
    proven_no = screen_answers(1, existing_automation="no", existing_automation_url="https://operator-1.example/faq",
                               existing_automation_quote="Every part on the lathes is loaded by our operators.",
                               existing_automation_date="2026-08-01")
    stale = screen_answers(2, manual_today="unknown", target_task_date="2023-01-15")
    undated = screen_answers(3, operating_now_date="")
    records = [inventory_record(number) for number in (1, 2, 3)]
    answers = [proven_no, stale, undated]
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    *_, screened = screen(tmp_path, records, answers, pages)
    questions = {record["input"]["site_name"]: [q["check"] for q in record["open_questions"]]
                 for record in screened.values()}
    assert questions == {"Synthetic Works 1": ["fit", "interest"],
                         "Synthetic Works 2": ["manual_workflow", "existing_automation", "freshness", "fit", "interest"],
                         "Synthetic Works 3": ["existing_automation", "freshness", "fit", "interest"]}
    assert {record["tier"] for record in screened.values()} == {"outreach_ready"}


def test_verify_writes_each_record_once(tmp_path):
    answers = screen_answers(1)
    workspace, provider, reader, records = screen(tmp_path, [inventory_record(1)], [answers], pages_for(answers))
    (path,) = (workspace.root / "screen" / "records").glob("*.json")
    before = path.read_bytes()
    again = ss.verify(workspace, reader=FakePages({}), today=TODAY)
    assert (again["screen"], again["page_reads"]) == ({"written": 0, "kept": 1}, 0)
    assert path.read_bytes() == before


def test_summary_counts_fields_checks_tiers_and_cost_without_site_data(tmp_path):
    ready, closed = screen_answers(1), screen_answers(2, operating_now="no")
    pages = {**pages_for(ready), **pages_for(closed)}
    workspace, *_ = screen(tmp_path, [inventory_record(1), universe_row(2)], [ready, closed], pages)
    report = ss.summary(workspace)
    assert report["screen"]["runs"] == {"sites": 2, "created": 2, "outcome_unknown": 0, "refused_not_created": 0,
                                        "completed": 2, "failed": 0, "cancelled": 0, "in_flight": 0}
    assert report["screen"]["tiers"] == {"outreach_ready": 1, "screened": 1}
    assert report["screen"]["tiers_by_origin"] == {"discovery_inventory": {"outreach_ready": 1, "screened": 0},
                                                   "site_universe": {"outreach_ready": 0, "screened": 1}}
    assert report["screen"]["fields"]["operating_now"] == {"answers": {"yes": 1, "no": 1},
                                                           "levels": {"verified_on_page": 2}}
    assert report["screen"]["fields"]["operator_identity"]["answers"] == {"present": 2}
    assert report["screen"]["identity_basis"] == {"operator": {"web_quote": 1, "government_record": 1},
                                                  "physical_site": {"web_quote": 1, "government_record": 1}}
    assert report["screen"]["checks"]["not_closed"] == 1 and report["screen"]["open_questions"]["fit"] == 2
    assert (report["estimated_cost_usd"], report["committed_usd"]) == ("0.050", "0.050")
    assert report["contact"]["runs"]["sites"] == 0 and report["contact"]["records"] == 0
    assert json.loads((workspace.root / "summary.json").read_text()) == report
    text = ss.canonical(report)
    assert not any(value in text for value in SITE_STRINGS)


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
                       "--ceiling-usd", "1", "--max-runs", "1", "--apply"], environ={"PARALLEL_API_KEY": KEY},
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
        operator.main(["run", "--input", str(source), "--out", str(tmp_path / "out"), "--ceiling-usd", "1",
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
    spend = ["--out", str(out), "--ceiling-usd", "0.05", "--max-runs", "2"]
    assert operator.main(["plan", "--input", str(source)])["sites"] == 2
    assert operator.main(["run", "--input", str(source), *spend], environ=environ, transport=provider)["state"] == "planned"
    assert provider.calls == []
    assert operator.main(["run", "--input", str(source), *spend, "--apply"], environ=environ,
                         transport=provider)["created"] == 2
    assert operator.main(["collect", "--out", str(out), "--wait-seconds", "0"], environ=environ,
                         transport=provider)["state"] == "complete"
    pages = {url: text for answer in answers for url, text in pages_for(answer).items()}
    assert operator.main(["verify", "--out", str(out)], reader=FakePages(pages), today=TODAY)["screen"]["written"] == 2
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
    top |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert top <= set(sys.stdlib_module_names)
    later = {(node.module, alias.name) for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
             and node not in tree.body for alias in node.names}
    assert later == {("tools.daily_research", "site_universe"), ("tools.daily_research", "search")}
    for name in ("site_universe.py", "search.py"):  # Those two are standard library only at import too.
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
