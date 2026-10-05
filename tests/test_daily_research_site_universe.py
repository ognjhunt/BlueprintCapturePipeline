"""Site universe pin, export loader, selection, attachment and outcomes; synthetic data only."""
import base64
import gzip
import hashlib
import json
import random
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from tools.daily_research import recovery
from tools.daily_research import site_universe as su
from tools.daily_research.consumer import qa_text
from tools.daily_research.runner import Refusal, canonical, digest, keys

DAY = "2026-10-06"
WEIGHTS = {"fixed_arm_machine_tending": 1.0, "hospital_logistics": 0.25, "kitting": 0.85, "palletizing": 0.85}
SEEDS = ["fixed_arm_machine_tending", "kitting", "palletizing"]
ANCHOR = "Before searching read /workspace/inputs/blueprint-research-crm-identities.json; exact SHA256 x. "
STARTED = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def site(number, *, rank=None, lead="fixed_arm_machine_tending", group=..., city="Fixture City", postal=None,
         name=None, operator=None, aliases=()):
    return {"site_id": sha(f"synthetic-site-{number}"), "rank": number if rank is None else rank,
            "score": round(90 - number / 4, 3), "lead_capability": lead, "capabilities": [lead],
            "primary_site_type": "synthetic_machine_shop", "category": "manufacturing", "naics": "332710",
            "name": name or f"Synthetic Works {number}", "aliases": list(aliases),
            "operator": operator or f"Synthetic Operator {number}",
            "group_key": f"synthetic operator {number}" if group is ... else group,
            "street": f"{number} Example Road", "city": city, "state": "TX",
            "postal_code": postal or f"{number:05d}", "employees": 120, "sources": ["synthetic_source"],
            "attribution_required": False,
            "fit": "capability_fit +25.0 (synthetic); size_fit +22.0 (synthetic); task_evidence +10.0 (synthetic)"}


def manifest(rows, **changes):
    ranked = max([row["rank"] for row in rows] + [len(rows)])
    leads = {}
    for row in rows:
        leads[row["lead_capability"]] = leads.get(row["lead_capability"], 0) + 1
    leads["fixed_arm_machine_tending"] = leads.get("fixed_arm_machine_tending", 0) + ranked - len(rows)
    value = {"snapshot_id": sha("synthetic-snapshot"), "states": ["TX"],
             "counts": {"sites": ranked + 10, "ranked": ranked, "excluded": 4, "rows": len(rows)},
             "rank_manifest_sha256": sha("rank-manifest"), "ranked_file_sha256": sha("ranked-file"),
             "rank_config_sha256": sha("rank-config"), "rank_config_version": "synthetic-rank-v1",
             "ranker_sha256": sha("ranker"), "taxonomy_sha256": sha("taxonomy"),
             "exclusion_counts": {"input:synthetic_crm": 2, "synthetic_rule": 2}, "lead_capability_counts": leads,
             "capability_weights": dict(WEIGHTS), "selection_policy": {"seed_capabilities": list(SEEDS)},
             "license_union": ["ODbL-1.0", "US-Gov-Work"],
             "attribution": ["Synthetic data contributors"], "distribution": "internal_only",
             "approval_reference": "owner-synthetic-ranking-review", "rows_sha256": su.rows_digest(rows),
             "previous_snapshot_id": None, "new_sites": None}
    value.update(changes)
    return value


def build_export(rows, *, serialize=su.canonical_json, document=None, **changes):
    rows = sorted(deepcopy(rows), key=lambda row: (row["rank"], row["site_id"]))
    value = document or {"schema_version": su.EXPORT, "manifest": manifest(rows, **changes), "rows": rows}
    return gzip.compress(serialize(value).encode(), mtime=0)


def pin_for(raw, *, slice_size=5, reoffer_after_days=90, generation="1001"):
    value = hashlib.sha256(raw).hexdigest()
    return {"enabled": True, "uri": su.object_uri(value), "generation": generation, "sha256": value, "bytes": len(raw),
            "snapshot_id": sha("synthetic-snapshot"), "rank_config_sha256": sha("rank-config"),
            "slice_size": slice_size, "reoffer_after_days": reoffer_after_days,
            "approval_reference": "owner-synthetic-pin"}


def crm(*rows):
    values = [["CRM"], [], [], [], ["Prospect ID", "Organization", "Prospect type", "Site / team"]]
    for number, (organization, place, location) in enumerate(rows):
        values.append([f"SYN-{number}", organization, "site", place, "", "", "", "", "",
                       "https://synthetic.example/task", "", "", "", "", "Synthetic task", "", "", location or ""])
    return values


def prior(day, outcomes=(), candidates=(), *, tamper=False):
    packet = {"candidates": list(candidates),
              "site_universe": {"schema_version": su.OUTCOMES, "state": "attached", "outcomes": list(outcomes)}}
    return {"date": day, "packet": packet, "packet_digest": "0" * 64 if tamper else digest(packet)}


def outcome(number, value="screened"):
    return {"site_id": sha(f"synthetic-site-{number}"), "outcome": value, "inventory_index": 0, "candidate_key": None,
            "event_id": None}


def chosen(rows, *, slice_size=5, history=(), values=None, **changes):
    export = su.load_export(build_export(rows, **changes))
    sites, selection = su.select(export, history=list(history), crm_values=values or crm(), run_date=DAY,
                                 slice_size=slice_size, reoffer_after_days=90)
    return [s["rank"] for s in sites], selection


class Store:
    """Company control and object reads as the Firestore ledger exposes them."""

    def __init__(self, control=None, objects=None):
        self.control, self.objects, self.reads = control, dict(objects or {}), []

    def site_universe_control(self):
        self.reads.append("control")
        if isinstance(self.control, Exception):
            raise self.control
        return deepcopy(self.control)

    def site_universe_object(self, pin):
        self.reads.append("object")
        value = self.objects.get((pin["sha256"], pin["generation"]))
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise Refusal("site_universe_object_missing")
        return value


def store_for(rows, **pin_changes):
    raw = build_export(rows)
    value = pin_for(raw, **pin_changes)
    return Store(value, {(value["sha256"], value["generation"]): raw}), value, raw


def attach(store, *, history=(), values=None, research_seconds=3600, supported=True):
    return su.attach(store, history=list(history), crm_values=values or crm(), day=DAY,
                     research_seconds=research_seconds, supported=supported)


def no_floats(value):
    if isinstance(value, dict):
        return all(no_floats(item) for item in value.values())
    if isinstance(value, list):
        return all(no_floats(item) for item in value)
    return not isinstance(value, float)


# --- pin ------------------------------------------------------------------------------------------


def test_absent_or_disabled_pin_is_off_whatever_else_it_holds():
    assert su.pin(None) is None
    assert su.pin({"enabled": False}) is None
    assert su.pin({"enabled": False, "uri": "anything", "slice_size": 999}) is None
    raw = build_export([site(1)])
    assert su.pin(pin_for(raw)) == pin_for(raw)


@pytest.mark.parametrize("change", [
    {"enabled": "true"}, {"uri": "gs://other-bucket/x"}, {"generation": "0123"}, {"generation": "12a"},
    {"generation": 1001}, {"generation": "1" * 20}, {"bytes": 0}, {"bytes": su.MAX_OBJECT_BYTES + 1}, {"bytes": True},
    {"sha256": "A" * 64}, {"snapshot_id": "x"}, {"rank_config_sha256": None}, {"slice_size": 4}, {"slice_size": 31},
    {"slice_size": 20.0}, {"reoffer_after_days": 0}, {"reoffer_after_days": 366}, {"approval_reference": "PENDING-owner"},
    {"approval_reference": " pending review"}, {"approval_reference": " "}, {"approval_reference": "café"},
    {"extra": True}])
def test_pin_fields_are_exact(change):
    value = {**pin_for(build_export([site(1)])), **change}
    with pytest.raises(su.SiteUniverseError, match="^site_universe_pin_invalid$"):
        su.pin(value)
    missing = pin_for(build_export([site(1)]))
    missing.pop("generation")
    with pytest.raises(su.SiteUniverseError, match="^site_universe_pin_invalid$"):
        su.pin(missing)


def test_slice_must_fit_the_research_window():
    value = pin_for(build_export([site(1)]), slice_size=20)
    assert su.pin(value, 1800)["slice_size"] == 20
    with pytest.raises(su.SiteUniverseError, match="^site_universe_slice_exceeds_research_window$"):
        su.pin(value, 1799)


# --- export loader ----------------------------------------------------------------------------------


def test_loader_reads_the_exact_canonical_export_and_binds_the_pin():
    rows = [site(1), site(2, name="Synthetic Fábrica 2")]  # Non-ASCII stays unescaped in the producer bytes.
    raw = build_export(rows)
    export = su.load_export(raw, pin_for(raw))
    assert [row["rank"] for row in export["rows"]] == [1, 2] and export["sha256"] == hashlib.sha256(raw).hexdigest()
    assert export["manifest"]["rows_sha256"] == hashlib.sha256(su.canonical_json(export["rows"]).encode()).hexdigest()
    assert "Fábrica".encode() in gzip.decompress(raw)
    # runner.canonical escapes non-ASCII; an export serialized that way is not the producer's bytes.
    escaped = build_export(rows, serialize=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    with pytest.raises(su.SiteUniverseError, match="^site_universe_export_invalid$"):
        su.load_export(escaped)
    with pytest.raises(su.SiteUniverseError, match="^site_universe_export_invalid$"):
        su.load_export(build_export(rows, serialize=lambda value: json.dumps(value, ensure_ascii=False, indent=1)))


@pytest.mark.parametrize(("change", "code"), [
    ({"snapshot_id": sha("other-snapshot")}, "site_universe_export_binding_mismatch"),
    ({"rank_config_sha256": sha("other-config")}, "site_universe_export_binding_mismatch"),
    ({"sha256": "0" * 64}, "site_universe_object_digest_mismatch"),
    ({"bytes": 7}, "site_universe_object_digest_mismatch"),
])
def test_loader_binds_digest_size_snapshot_and_rank_config(change, code):
    raw = build_export([site(1)])
    with pytest.raises(su.SiteUniverseError, match=f"^{code}$"):
        su.load_export(raw, {**pin_for(raw), **change})


def corrupt_rows(change):
    rows = [site(1), site(2), site(3)]
    change(rows)
    return build_export(rows, rows_sha256=su.rows_digest(sorted(rows, key=lambda r: (r["rank"], r["site_id"]))))


@pytest.mark.parametrize("raw", [
    b"not gzip", build_export([site(1)]) + b"trailing", build_export([site(1)]) * 2,
    gzip.compress(b" " * (su.MAX_RAW_BYTES + 1), mtime=0), gzip.compress(b'{"schema_version": NaN}', mtime=0),
    build_export([site(1)], rows_sha256="0" * 64),
    build_export([site(1)], distribution="public"), build_export([site(1)], approval_reference="PENDING-owner"),
    build_export([site(1)], previous_snapshot_id=sha("previous"), new_sites=None),
    build_export([site(1)], lead_capability_counts={"fixed_arm_machine_tending": 99}),
    build_export([site(1)], selection_policy={"seed_capabilities": ["unknown_capability"]}),
    build_export([site(1)], extra_field=True),
    corrupt_rows(lambda rows: rows[0].update(lat=30.0)),
    corrupt_rows(lambda rows: rows[0].update(fit="x" * 241)),
    corrupt_rows(lambda rows: rows[0].update(lead_capability="palletizing")),
    corrupt_rows(lambda rows: rows[0].update(capabilities=["unknown_capability"])),
    corrupt_rows(lambda rows: rows[0].update(name="Synthetic\nWorks")),
    corrupt_rows(lambda rows: rows[0].update(score=True)),
    corrupt_rows(lambda rows: rows[1].update(site_id=rows[0]["site_id"])),
    gzip.compress(su.canonical_json({"schema_version": su.EXPORT, "manifest": manifest([site(2), site(1)]),
                                     "rows": [site(2), site(1)]}).encode(), mtime=0),
])
def test_loader_corruption_fails_closed(raw):
    with pytest.raises(su.SiteUniverseError, match="^site_universe_export_invalid$"):
        su.load_export(raw)


@pytest.mark.parametrize("licenses", [["CC-BY-NC-4.0"], ["ODbL-1.0", "proprietary"], ["synthetic-license-a"], ["odbl-1.0"]])
def test_loader_fails_closed_on_a_license_outside_the_producer_allowlist(licenses):
    with pytest.raises(su.SiteUniverseError, match="^site_universe_export_invalid$"):
        su.load_export(build_export([site(1)], license_union=licenses))
    assert su.load_export(build_export([site(1)], license_union=["ODbL-1.0", "US-Gov-Work", "US-PD"]))["rows"]


def test_loader_size_bounds_fail_closed():
    with pytest.raises(su.SiteUniverseError, match="^site_universe_object_too_large$"):
        su.load_export(b"\x1f\x8b" + b"\0" * su.MAX_OBJECT_BYTES)
    rows = [site(number) for number in range(1, su.MAX_ROWS + 2)]
    with pytest.raises(su.SiteUniverseError, match="^site_universe_export_invalid$"):
        su.load_export(build_export(rows))


# --- selection ------------------------------------------------------------------------------------


def test_seeds_caps_and_groups_follow_the_documented_passes():
    rows = [site(1, group="group a"), site(2, group="group a"), site(3), site(4), site(5, lead="palletizing"),
            site(6, lead="kitting"), site(7), site(8, lead="hospital_logistics")]
    ranks, selection = chosen(rows)
    assert ranks == [1, 3, 5, 6, 8]  # Presented in rank order.
    assert selection["passes"] == {"seed": 3, "fill": 2, "relax_capability_cap": 0, "relax_group_cap": 0}
    assert selection["lead_capability_cap"] == 2 and selection["offered_by_lead_capability"] == {
        "fixed_arm_machine_tending": 2, "hospital_logistics": 1, "kitting": 1, "palletizing": 1}
    assert chosen(rows, slice_size=6)[0] == [1, 3, 4, 5, 6, 8]


def test_seeds_bring_in_policy_capabilities_that_rank_fill_would_skip():
    rows = [site(1), site(2), site(3, lead="hospital_logistics"), site(4, lead="hospital_logistics"),
            site(5, lead="palletizing"), site(6, lead="palletizing"), site(7, lead="kitting")]
    assert chosen(rows)[0] == [1, 2, 3, 5, 7]
    assert chosen(rows, selection_policy={"seed_capabilities": []})[0] == [1, 2, 3, 4, 5]


def test_relaxing_passes_fill_the_capability_cap_then_the_group_cap():
    rows = [site(number) for number in range(1, 7)]
    ranks, selection = chosen(rows)
    assert ranks == [1, 2, 3, 4, 5]
    assert selection["passes"] == {"seed": 1, "fill": 1, "relax_capability_cap": 3, "relax_group_cap": 0}
    shared = [site(number, group="one synthetic group") for number in range(1, 7)]
    ranks, selection = chosen(shared)
    assert ranks == [1, 2, 3, 4, 5]
    assert selection["passes"] == {"seed": 1, "fill": 0, "relax_capability_cap": 0, "relax_group_cap": 4}
    # A null group key is the site's own group.
    assert chosen([site(number, group=None) for number in range(1, 7)])[1]["passes"]["relax_group_cap"] == 0


def test_equal_ranks_break_by_site_id():
    first, second = site(1, rank=4), site(2, rank=4)
    ordered = sorted([first, second], key=lambda row: row["site_id"])
    export = su.load_export(build_export([second, first]))
    sites, _ = su.select(export, history=[], crm_values=crm(), run_date=DAY, slice_size=5, reoffer_after_days=90)
    assert [s["site_id"] for s in sites] == [row["site_id"] for row in ordered]


def test_history_outcomes_are_removed_inside_the_reoffer_window_only():
    rows = [site(number) for number in range(1, 9)]
    history = [prior("2026-09-26", [outcome(1), outcome(2, "untouched"), outcome(3, "rejected")]),
               prior("2026-07-08", [outcome(4, "researched_gap")]),  # 90 days before: re-offered.
               prior(DAY, [outcome(5)]), prior("2026-10-07", [outcome(6)])]  # Not prior to the run date.
    ranks, selection = chosen(rows, slice_size=8, history=history)
    assert ranks == [2, 4, 5, 6, 7, 8]
    assert selection["removed"] == {"reoffer_window": 2, "untrusted_history": 0, "crm": 0, "prior_candidate": 0}
    assert selection["history_outcome_rows"] == 1 and selection["eligible"] == 6
    export = su.load_export(build_export(rows))
    sites, _ = su.select(export, history=history, crm_values=crm(), run_date=DAY, slice_size=8, reoffer_after_days=91)
    assert [s["rank"] for s in sites] == [2, 5, 6, 7, 8]


def damaged(day, *, block=..., record=None, outcomes=()):
    """A prior row whose packet no longer matches its packet_digest."""
    value = prior(day, outcomes, tamper=True)
    if block is not ...:
        value["packet"]["site_universe"] = block
    if record is not None:
        value["site_universe"] = record
    return value


def test_a_damaged_history_row_is_skipped_and_every_site_it_names_stays_out_for_the_window():
    rows = [site(number) for number in range(1, 9)]
    record = {"state": "attached", "site_ids": [sha("synthetic-site-2"), sha("synthetic-site-3")]}
    history = [damaged("2026-10-01", record=record, outcomes=[outcome(1), outcome(2, "untouched")])]
    ranks, selection = chosen(rows, slice_size=8, history=history)
    assert ranks == [4, 5, 6, 7, 8]  # 1 from its untrusted outcomes, 2 and 3 from the slice it offered.
    assert selection["removed"] == {"reoffer_window": 0, "untrusted_history": 3, "crm": 0, "prior_candidate": 0}
    assert selection["history_rows_untrusted"] == 1 and selection["history_outcome_rows"] == 0
    assert selection["history_codes"] == ["site_universe_history_binding_invalid"]
    # A bound row whose outcomes were unavailable is untrusted the same way, with its own code.
    unavailable = prior("2026-10-01")
    unavailable["packet"]["site_universe"] = {"schema_version": su.OUTCOMES, "state": "unavailable",
                                              "code": "site_universe_frozen_slice_binding_invalid"}
    unavailable.update(packet_digest=digest(unavailable["packet"]), site_universe=record)
    ranks, selection = chosen(rows, slice_size=8, history=[unavailable])
    assert ranks == [1, 4, 5, 6, 7, 8] and selection["history_codes"] == ["site_universe_frozen_slice_binding_invalid"]
    # A damaged row that attached no slice offered nothing, and one outside the window is ignored.
    assert chosen(rows, slice_size=8, history=[damaged("2026-10-01", record={"state": "refused", "code": "x"})])[0] == list(range(1, 9))
    assert chosen(rows, slice_size=8, history=[damaged("2026-07-08", block="garbage")])[1]["history_rows_untrusted"] == 0


@pytest.mark.parametrize("value", [
    lambda: damaged("2026-10-01", block="garbage"),
    lambda: damaged("2026-10-01", outcomes=[{"site_id": "x", "outcome": "screened"}]),
    lambda: damaged("2026-10-01", record={"state": "attached", "site_ids": "not-a-list"}),
    lambda: {**prior("2026-10-01", [{"site_id": "x", "outcome": "screened"}])},  # bound but malformed
])
def test_a_damaged_row_that_names_no_readable_site_refuses_the_slice(value):
    with pytest.raises(su.SiteUniverseError, match="^site_universe_history_binding_invalid$"):
        chosen([site(1)], history=[value()])


def test_crm_and_prior_candidates_are_prefiltered_by_organization_words_and_place():
    rows = [site(1), site(2), site(3, aliases=["Example Alias Plant"]), site(4), site(5),
            site(6, name="Alpha Works", operator="Beta Holdings"), site(7), site(8)]
    values = crm(("Synthetic Operator 1", "North plant", "Fixture City, TX"),   # operator words + city
                 ("synthetic works 2", "Unit 00002", None),                  # name words + postal code
                 ("Example Alias", "Plant", "Fixture City"),                  # alias words + city
                 ("Synthetic Operator 4", "Plant", "Elsewhere"),              # place does not match
                 ("Alpha Holdings", "Plant", "Fixture City"),                 # words split across fields
                 ("Unrelated Organization", "Plant", "Fixture City"))
    candidates = [{"organization": "Synthetic Operator 7", "site": "Synthetic site", "location": "Fixture City, TX"}]
    ranks, selection = chosen(rows, slice_size=8, values=values, history=[prior("2026-10-01", candidates=candidates)])
    assert ranks == [4, 5, 6, 8]
    assert selection["removed"] == {"reoffer_window": 0, "untrusted_history": 0, "crm": 3, "prior_candidate": 1}
    assert selection["crm_identities"] == 6 and selection["prior_candidates"] == 1


def test_selection_is_deterministic_and_independent_of_history_and_crm_order():
    rows = [site(number, lead=random.Random(number).choice(sorted(WEIGHTS)), group=f"group {number % 4}")
            for number in range(1, 40)]
    history = [prior(f"2026-09-{day:02d}", [outcome(day)]) for day in range(10, 20)]
    values = crm(*[(f"Synthetic Operator {n}", "Plant", "Fixture City") for n in range(20, 26)])
    expected = chosen(rows, slice_size=12, history=history, values=values)
    for seed in range(3):
        shuffled_history, shuffled = list(history), values[:5] + random.Random(seed).sample(values[5:], len(values) - 5)
        random.Random(seed).shuffle(shuffled_history)
        assert chosen(rows, slice_size=12, history=shuffled_history, values=shuffled) == expected


# --- attach ------------------------------------------------------------------------------------------


def test_attach_is_off_without_a_control_reader_or_an_enabled_pin():
    assert attach(object()) is None
    for control in (None, {"enabled": False}, {"enabled": False, "sha256": "garbage"}):
        store = Store(control)
        assert attach(store) is None and store.reads == ["control"]


def test_attach_freezes_a_slice_with_its_pin_export_and_selection():
    rows = [site(number) for number in range(1, 8)]
    store, value, _ = store_for(rows, slice_size=5)
    record, slice_raw = attach(store)
    assert store.reads == ["control", "object"] and record["state"] == "attached"
    assert record["pin"] == value and record["export"]["sha256"] == value["sha256"]
    assert record["slice_sha256"] == hashlib.sha256(slice_raw).hexdigest() and record["slice_bytes"] == len(slice_raw)
    document = json.loads(slice_raw)
    assert slice_raw == canonical(document).encode() and document["distribution"] == "internal_only"
    assert [s["site_id"] for s in document["sites"]] == record["site_ids"] == [row["site_id"] for row in rows[:5]]
    assert set(document["sites"][0]) == set(su.SLICE_FIELDS)  # No group key, score or coordinates.
    assert record["selection"]["run_date"] == DAY and no_floats(record)


@pytest.mark.parametrize(("setup", "code"), [
    (lambda store, value: store.control.update(slice_size=99), "site_universe_pin_invalid"),
    (lambda store, value: store.objects.clear(), "site_universe_object_missing"),
    (lambda store, value: store.objects.update({(value["sha256"], "1001"): Refusal("site_universe_object_generation_mismatch")}),
     "site_universe_object_generation_mismatch"),
    (lambda store, value: store.objects.update({(value["sha256"], "1001"): Refusal("site_universe_object_too_large")}),
     "site_universe_object_too_large"),
    (lambda store, value: store.objects.update({(value["sha256"], "1001"): Refusal("site_universe_object_unavailable")}),
     "site_universe_object_unavailable"),
    (lambda store, value: store.objects.update({(value["sha256"], "1001"): b"different bytes"}),
     "site_universe_object_digest_mismatch"),
    (lambda store, value: store.control.update(snapshot_id=sha("other")), "site_universe_export_binding_mismatch"),
    (lambda store, value: store.objects.update({(value["sha256"], "1001"): RuntimeError("upstream text")}),
     "site_universe_attach_unavailable"),
])
def test_every_slice_failure_is_a_refused_record_without_bytes(setup, code):
    store, value, _ = store_for([site(number) for number in range(1, 8)])
    setup(store, value)
    assert attach(store) == ({"schema_version": su.ATTACHMENT, "state": "refused", "code": code}, None)


def test_profile_window_history_and_exhaustion_are_recorded_codes():
    rows = [site(number) for number in range(1, 8)]
    store, _, _ = store_for(rows)
    assert attach(store, supported=False)[0]["code"] == "site_universe_profile_unsupported" and store.reads == ["control"]
    assert attach(store_for(rows)[0], research_seconds=449)[0]["code"] == "site_universe_slice_exceeds_research_window"
    history = [damaged("2026-10-01", block="garbage")]  # A damaged row that names no readable site.
    assert attach(store_for(rows)[0], history=history)[0]["code"] == "site_universe_history_binding_invalid"
    record, raw = attach(store_for(rows)[0], values=crm(("Synthetic", "Plant", "Fixture City")))
    assert raw is None and record["state"] == "exhausted" and record["code"] == "site_universe_slice_empty"
    assert record["selection"]["removed"]["crm"] == 7 and record["selection"]["offered"] == 0


@pytest.mark.parametrize("where", ["control", "object"])
def test_a_lost_store_or_lease_still_stops_the_run(where):
    store, value, _ = store_for([site(1), site(2), site(3), site(4), site(5)])
    if where == "control":
        store.control = Refusal("firestore_lease_lost")
    else:
        store.objects[(value["sha256"], value["generation"])] = Refusal("firestore_bridge_unavailable")
    with pytest.raises(Refusal, match="^firestore_(lease_lost|bridge_unavailable)$"):
        attach(store)


# --- handoff, frozen slice and outcomes ----------------------------------------------------------------


def attached_row(rows, *, slice_size=5, history=()):
    store, _, _ = store_for(rows, slice_size=slice_size)
    record, raw = attach(store, history=history)
    body = {"environment": {"files": [{"type": "inline", "path": "/workspace/other.json", "data": ""}]},
            "metadata": {"run_key": "blueprint-researcher:" + DAY},
            "input": "Lead text. " + ANCHOR + "Rest. Each record has " + su.DISPOSITIONS_TODAY + "."}
    su.bind(body, record, raw, ANCHOR)
    return {"date": DAY, "run_key": "blueprint-researcher:" + DAY, "started_at": STARTED.isoformat(),
            "remote_completed_at": int(STARTED.timestamp()) + 1500, "site_universe": record, "create_payload": body,
            "metadata": body["metadata"], "application_tool_calls": {}}


def test_bind_adds_the_file_digest_and_one_paragraph_after_the_crm_prefix():
    row = attached_row([site(number) for number in range(1, 8)])
    body, record = row["create_payload"], row["site_universe"]
    assert body["environment"]["files"][-1]["path"] == su.SLICE_PATH
    assert body["metadata"]["site_universe_slice_digest"] == record["slice_sha256"]
    assert body["input"] == ("Lead text. " + ANCHOR + su.paragraph(record) + "Rest. Each record has "
                             + su.DISPOSITIONS_WITH_SLICE + ".")
    assert "screened" not in su.DISPOSITIONS_TODAY and su.DISPOSITIONS_WITH_SLICE.endswith("duplicate or screened)")
    assert su.frozen_ids(row) == frozenset(record["site_ids"])
    assert su.frozen_ids({"site_universe": su.refused("site_universe_object_missing")}) is None and su.frozen_ids({}) is None
    text = su.paragraph(record)
    assert su.SLICE_PATH in text and record["slice_sha256"] in text and "untrusted data" in text
    assert "never copy it" in text and "live evidence" in text and len(text.encode()) < 1500
    assert su.frozen_slice(row)["sites"][0]["site_id"] == record["site_ids"][0]


@pytest.mark.parametrize("tamper", [
    lambda row: row["create_payload"]["environment"]["files"][-1].update(data=base64.b64encode(b"{}").decode()),
    lambda row: row["metadata"].update(site_universe_slice_digest="0" * 64),
    lambda row: row["site_universe"]["site_ids"].reverse(),
    lambda row: row["create_payload"]["environment"]["files"].append(dict(row["create_payload"]["environment"]["files"][-1])),
])
def test_frozen_slice_rechecks_the_sha256_before_any_use(tamper):
    row = attached_row([site(number) for number in range(1, 8)])
    tamper(row)
    with pytest.raises(su.SiteUniverseError, match="^site_universe_frozen_slice_binding_invalid$"):
        su.frozen_slice(row)
    assert su.outcomes(row, {}, [], []) == {"schema_version": su.OUTCOMES, "state": "unavailable",
                                            "code": "site_universe_frozen_slice_binding_invalid"}


def record_for(number, disposition, **changes):
    value = {"operator": f"Synthetic Operator {number}", "site": f"Synthetic site {number}",
             "location": "Fixture City, TX", "task_hypothesis": f"Synthetic task {number}", "source_urls": [],
             "evidence_gap": "Synthetic gap", "disposition": disposition, "site_universe_id": sha(f"synthetic-site-{number}")}
    value.update(changes)
    return value


def formal(record, key):
    identities = keys({"organization": record["operator"], "site": record["site"], "location": record["location"],
                       "task": record["task_hypothesis"]})
    return {"candidate_key": key, "identity_keys": sorted(identities)}


def test_outcomes_map_dispositions_link_candidates_and_list_issues():
    row = attached_row([site(number) for number in range(1, 10)], slice_size=9)
    row["application_tool_calls"] = {
        "a": {"phase": "research", "attempted": True, "result_bytes": 100},
        "b": {"phase": "repair", "attempted": True, "result_bytes": 200},
        "c": {"phase": "research", "attempted": True, "budget_exhausted": True, "result_bytes": 50},
        "d": {"phase": "qa", "attempted": True, "result_bytes": 999}}
    inventory = [record_for(1, "screened"), record_for(2, "unresolved"), record_for(3, "candidate"),
                 record_for(4, "candidate"), record_for(5, "candidate"), record_for(6, "rejected"),
                 record_for(0, "screened", site_universe_id=sha("not-in-slice")), record_for(1, "rejected"),
                 {key: value for key, value in record_for(0, "unresolved").items() if key != "site_universe_id"},
                 record_for(7, "learning"), record_for(8, "duplicate")]
    candidates, duplicates = [formal(inventory[2], "k3")], [formal(inventory[4], "kd5")]
    block = su.outcomes(row, {"discovery_inventory": inventory}, candidates, duplicates)
    found = {item["site_id"]: item for item in block["outcomes"]}
    assert [item["site_id"] for item in block["outcomes"]] == row["site_universe"]["site_ids"]
    expected = {1: ("screened", 0, None), 2: ("researched_gap", 1, None), 3: ("candidate", 2, "k3"),
                4: ("candidate", 3, None), 5: ("duplicate", 4, "kd5"), 6: ("rejected", 5, None),
                7: ("learning", 9, None), 8: ("duplicate", 10, None), 9: ("untouched", None, None)}
    for number, (value, index, key) in expected.items():
        item = found[sha(f"synthetic-site-{number}")]
        assert (item["outcome"], item["inventory_index"], item["candidate_key"]) == (value, index, key)
        assert (item["event_id"] is None) == (value == "untouched")
    assert found[sha("synthetic-site-1")]["event_id"] == digest([su.OUTCOMES, row["run_key"], sha("synthetic-site-1")])
    assert block["link_issues"] == [
        {"code": "site_universe_id_unknown", "inventory_index": 6},
        {"code": "site_universe_id_repeated", "inventory_index": 7, "site_id": sha("synthetic-site-1")},
        {"code": "site_universe_candidate_unlinked", "inventory_index": 3, "site_id": sha("synthetic-site-4")}]
    funnel = block["funnel"]
    assert funnel["agent"] == {"touched": 8, "untouched": 1, "screened": 1, "researched_gap": 1, "inventory_candidates": 3,
                               "formal_candidates_linked": 1, "formal_candidates_unlinked": 1, "rejected": 1,
                               "learning": 1, "duplicate": 2, "agent_added": 1}
    assert funnel["universe"] == {"sites": 19, "ranked": 9, "export_rows": 9, "new_sites": None}
    assert funnel["selection"]["offered"] == 9 and funnel["selection"]["removed"]["crm"] == 0
    assert funnel["run"] == {"elapsed_seconds": 1500, "application_calls": 2, "research_evidence_bytes": 350,
                             "paid_reservations_micros": 0}
    assert funnel["qa"] is None and funnel["not_measured"]["contacted"] is None
    assert funnel["not_measured"]["per_site_cost"] == "not_measured" and no_floats(block)


def test_run_level_time_and_post_review_counts():
    row = attached_row([site(number) for number in range(1, 8)])
    row["remote_completed_at"] = int(STARTED.timestamp()) + 61.6  # Provider timestamps never put a float in the packet.
    inventory = [record_for(1, "candidate"), record_for(2, "candidate")]
    candidates = [formal(inventory[0], "k1"), formal(inventory[1], "k2"), {"candidate_key": "k-agent", "identity_keys": ["x"]}]
    block = su.outcomes(row, {"discovery_inventory": inventory}, candidates, [])
    assert block["funnel"]["run"]["elapsed_seconds"] == 61 and no_floats(block)
    row["packet"] = {"site_universe": block}
    assert su.status(row)["funnel"]["qa"] is None
    row["review"] = {"accepted_keys": ["k1", "k-agent"], "lead_verification": {"results": [
        {"candidate_key": "k1", "eligible_for_qualified_promotion": True},
        {"candidate_key": "k2", "eligible_for_qualified_promotion": False},
        {"candidate_key": "k-agent", "eligible_for_qualified_promotion": True}]}}
    status = su.status(row)
    assert status["funnel"]["qa"] == {"eligible_for_promotion": {"slice": 1, "run": 2}, "accepted": {"slice": 1, "run": 2}}
    assert status["state"] == "attached" and status["offered"] == 5 and status["packet_state"] == "attached"
    assert "site_ids" not in status and "Synthetic" not in canonical(status)


def test_packet_block_stays_inside_its_reserve_at_the_slice_cap():
    rows = [site(number) for number in range(1, 40)]
    row = attached_row(rows, slice_size=su.MAX_SLICE)
    inventory = [record_for(number, "candidate") for number in range(1, 31)]
    inventory += [record_for(0, "screened", site_universe_id=sha(f"unknown-{n}")) for n in range(60)]
    block = su.outcomes(row, {"discovery_inventory": inventory}, [], [])
    assert block["state"] == "attached" and len(block["outcomes"]) == su.MAX_SLICE
    assert len(block["link_issues"]) == su.MAX_LINK_ISSUES and block["link_issue_count"] == 90
    assert len(canonical(block).encode()) <= su.PACKET_RESERVE


def test_refused_and_exhausted_rows_get_a_small_packet_block_and_no_text():
    for record in (su.refused("site_universe_object_missing"),
                   {"schema_version": su.ATTACHMENT, "state": "exhausted", "code": "site_universe_slice_empty"}):
        row = {"site_universe": record}
        assert su.outcomes(row, {}, [], []) == {"schema_version": su.OUTCOMES, "state": record["state"], "code": record["code"]}
        assert su.packet_reserve(row) == 0 and su.frozen_slice(row) is None
        assert su.explain_sentence(row) == su.repair_sentence(row) == su.qa_sentence(row) == ""
        assert su.status(row) == {"state": record["state"], "code": record["code"]}
    assert su.refused("free text from upstream")["code"] == "site_universe_attach_unavailable"


def test_explain_repair_and_qa_text_gain_one_sentence_only_with_a_slice():
    row = attached_row([site(number) for number in range(1, 8)])
    plain = {"date": DAY, "research_contract_version": 3}
    assert recovery.explain(plain, "discovery_inventory_invalid") + su.explain_sentence(row) == recovery.explain(
        {**plain, "site_universe": row["site_universe"]}, "discovery_inventory_invalid")
    assert recovery.explain({**plain, "site_universe": row["site_universe"]}, "output_schema_invalid") == recovery.explain(
        plain, "output_schema_invalid")
    assert su.packet_reserve(row) == su.PACKET_RESERVE
    for sentence in (su.explain_sentence(row), su.repair_sentence(row), su.qa_sentence(row)):
        assert sentence.strip().endswith(".") and sentence.strip()[:-1].count(". ") == 0 and "site-universe slice" in sentence
    qa_row = {"packet": {"candidates": [], "site_universe": {"state": "attached"}}, "packet_digest": "0" * 64,
              "web_tool_activities": 0, "discovery_profile": "adaptive-sites-v1"}
    snapshot = {"values": crm()}
    without = qa_text(qa_row, snapshot, "crm")
    with_slice = qa_text({**qa_row, "site_universe": row["site_universe"]}, snapshot, "crm")
    assert with_slice.replace(su.qa_sentence(row), "", 1) == without and su.qa_sentence(row) in with_slice


def test_publication_text_gains_one_sentence_only_with_a_slice():
    row = attached_row([site(number) for number in range(1, 8)])
    sentence = su.publication_sentence(row)
    assert sentence.strip().endswith(".") and "never copy" in sentence and "site-universe slice" in sentence
    assert su.publication_sentence({"site_universe": su.refused("site_universe_object_missing")}) == ""
    assert su.publication_sentence({}) == ""


class CountingBridge:
    def __init__(self, control):
        self.control, self.calls = control, []

    def call(self, op, **fields):
        self.calls.append(op)
        return deepcopy(self.control) if op == "control" else True


def test_the_pin_reuses_the_control_read_the_run_start_already_made_in_the_same_lease():
    from tools.daily_research.firestore import FirestoreLedger
    bridge = CountingBridge({"learning": {"enabled": True}, "site_universe": {"enabled": False}})
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        assert ledger.company_history_binding() == {"enabled": True}
        assert ledger.site_universe_control() == {"enabled": False}
    assert bridge.calls == ["acquire", "control", "release"]
    with ledger.lock():
        assert ledger.site_universe_control() == {"enabled": False}  # A new lease never reuses an older read.
    assert bridge.calls[3:] == ["acquire", "control", "release"]
    ledger.company_history_binding()
    ledger.site_universe_control()  # Outside a lease nothing is reused.
    assert bridge.calls[6:] == ["control", "control"]


def test_frozen_ids_and_the_short_record_near_the_intent_ceiling():
    row = attached_row([site(number) for number in range(1, 8)])
    assert su.frozen_ids(row) == frozenset(row["site_universe"]["site_ids"])
    assert su.frozen_ids({"site_universe": su.refused("site_universe_object_missing")}) is None and su.frozen_ids({}) is None
    assert su.short(row["site_universe"]) == {"state": "refused", "code": "site_universe_intent_resource_ceiling"}
    assert su.short(su.refused("site_universe_object_missing")) == {"state": "refused", "code": "site_universe_object_missing"}
    assert su.short({"schema_version": su.ATTACHMENT, "state": "exhausted", "code": "site_universe_slice_empty",
                     "selection": {"offered": 0}}) == {"state": "exhausted", "code": "site_universe_slice_empty"}
