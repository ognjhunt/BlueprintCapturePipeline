"""Complete synthetic discovery retention and resumable bounded reads, no I/O providers."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from tests.test_daily_research_adaptive import result
from tests.test_daily_research_runner import DAY, NOW
from tools.daily_research import discovery, verification
from tools.daily_research.runner import Ledger, Refusal, digest, validate_output


def inventory(count):
    return [{"operator": "Invented operator", "site": f"Invented site {n}", "location": "Test city",
             "task_hypothesis": None, "source_urls": [f"https://fixture.example/site/{n}"],
             "evidence_gap": "Unverified physical task. " + "x" * 1500,
             "disposition": "unresolved"} for n in range(count)]


def test_inventory_over_display_and_page_limits_roundtrips_every_original_record(tmp_path):
    entries = inventory(240)
    ledger = Ledger(tmp_path)
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY}
    from hashlib import sha256
    source = {"discovery_inventory": entries}
    ledger.write_json(DAY + "-artifact.json", source)
    row["raw_output_digest"] = sha256(ledger.read_bytes(DAY + "-artifact.json")).hexdigest()
    manifest = discovery.retain_inventory(entries, row, ledger, digest(source))
    assert manifest["complete_retention"] and manifest["record_count"] == 240
    assert manifest["page_count"] > 1 and manifest["eligibility"] == "discovery_only_no_promotion"
    cursor, retained = 0, []
    while cursor is not None:
        page = discovery.read_inventory_page(manifest, ledger, cursor)
        retained.extend(page["records"])
        assert page["start"] == len(retained) - len(page["records"])
        assert manifest["pages"][cursor]["bytes"] <= discovery.INVENTORY_PAGE_BYTES
        cursor = page["next_cursor"]
    assert retained == entries
    assert len(verification.packet_candidates({"candidates": [], "discovery_inventory_manifest": manifest})) == 0
    damaged = Path(tmp_path, manifest["pages"][1]["file"])
    damaged.write_bytes(damaged.read_bytes() + b" ")
    with pytest.raises(ValueError, match="page_binding_invalid"):
        discovery.read_inventory_page(manifest, ledger, 1)


def test_v3_retains_more_than_20_findings_and_100_raw_candidates_without_robot_matching():
    out, ctx, policy = result(125)
    out["findings"] = [f"Synthetic retained site finding {n}" for n in range(41)]
    out["discovery_inventory"] = inventory(125)
    for candidate in out["candidates"]:
        candidate["evidence"] = [e for e in candidate["evidence"] if e["role"] != "capability"]
        candidate["potential_robot_match"] = "unknown"
    schema = json.loads((Path(__file__).parents[1] / "tools/daily_research/daily-research.v3.schema.json").read_text())
    Draft202012Validator(schema).validate(out)
    candidates, duplicates = validate_output(out, DAY, set(), contract_version=3, knowledge_context=ctx,
        refresh_policy=policy, observed_at=NOW)
    assert len(candidates) == 125 and not duplicates and len(out["findings"]) == 41
    assert verification.cohort(candidates, {}, NOW)["verified_unique_site_task_candidates"] == 0
    bad = deepcopy(out)
    bad["candidates"][0]["potential_robot_match"] = "Unsupported invented robot fit"
    with pytest.raises(Refusal, match="unsupported_robot_match_must_remain_unknown"):
        validate_output(bad, DAY, set(), contract_version=3, knowledge_context=ctx,
            refresh_policy=policy, observed_at=NOW)


def test_inventory_missing_site_task_is_retained_without_fabrication():
    entries = inventory(1)
    entries[0].update(site=None, task_hypothesis=None, location=None)
    assert list(discovery.inventory_issues(entries)) == []
    entries[0]["source_urls"] = ["https://user:secret@fixture.example/source"]
    issues = list(discovery.inventory_issues(entries))
    assert issues[0]["pointer"] == "/discovery_inventory/0/source_urls/0"


def test_a_single_valid_unicode_record_cannot_overrun_page_bytes(tmp_path):
    import hashlib
    record = inventory(1)[0]
    for key in ("operator", "site", "location", "task_hypothesis", "evidence_gap"):
        record[key] = "𐀀" * 2000
    assert list(discovery.inventory_issues([record])) == [{"pointer": "/discovery_inventory/0", "code": "discovery_inventory_record_resource_ceiling_raw_retained"}]
    ledger = Ledger(tmp_path)
    source = {"discovery_inventory": [record]}
    ledger.write_json(DAY + "-artifact.json", source)
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY,
           "raw_output_digest": hashlib.sha256(ledger.read_bytes(DAY + "-artifact.json")).hexdigest()}
    with pytest.raises(Refusal, match="discovery_inventory_record_resource_ceiling_raw_retained"):
        discovery.retain_inventory([record], row, ledger, digest(source))
    assert not list(tmp_path.glob('*-inventory-*.json'))
    assert json.loads(ledger.read_bytes(DAY + "-artifact.json")) == source


def test_inventory_cannot_bind_replaced_or_truncated_records_to_an_original_artifact(tmp_path):
    import hashlib
    entries = inventory(3)
    ledger = Ledger(tmp_path)
    ledger.write_json(DAY + "-artifact.json", {"discovery_inventory": entries})
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY,
           "raw_output_digest": hashlib.sha256(ledger.read_bytes(DAY + "-artifact.json")).hexdigest()}
    with pytest.raises(Refusal, match="discovery_inventory_source_binding_invalid"):
        discovery.retain_inventory(entries[:2], row, ledger, digest({"discovery_inventory": entries[:2]}))
    replaced = deepcopy(entries)
    replaced[1]["source_urls"] = ['https://invented.example/other']
    with pytest.raises(Refusal, match="discovery_inventory_source_binding_invalid"):
        discovery.retain_inventory(replaced, row, ledger, digest({"discovery_inventory": replaced}))


def test_exhausted_repair_does_not_exclude_inventory_records_or_claim_false_complete_retention():
    """Independent review S5 (2026-10-04): one malformed optional inventory record must not discard
    the day. The inventory is quarantined whole with a receipt; it is never trimmed to a subset
    presented as complete, and its records stay in the immutable source artifact."""
    from tools.daily_research.recovery import exclude_located_items
    from tools.daily_research.runner import digest as row_digest
    original = {"discovery_inventory": inventory(3), "candidates": []}
    original["discovery_inventory"][1]["disposition"] = "invalid"
    saved = deepcopy(original)
    feedback = [{"path": "/discovery_inventory/1", "reason": "discovery_inventory_invalid"}]
    derived, exclusions = exclude_located_items(original, feedback)
    assert original == saved
    assert "discovery_inventory" not in derived and derived["candidates"] == []
    assert exclusions == [{"field": "discovery_inventory", "quarantined_whole_field": True, "record_count": 3,
                           "failures": feedback, "failure_count": 1, "failures_digest": row_digest(feedback),
                           "item_digest": row_digest(saved["discovery_inventory"])}]
    # A located inventory failure never removes other fields or a partial set of records.
    mixed, _ = exclude_located_items(original, feedback + [{"path": "/coverage", "reason": "discovery_coverage_invalid"}])
    assert mixed is None


def test_inventory_quarantine_receipt_is_bounded_and_counts_toward_the_packet_ceiling(tmp_path):
    """Re-review of S5: thousands of located inventory failures once produced a receipt larger than
    the review packet ceiling, attached after the ceiling check. The receipt now carries a bounded
    sample plus the count and digest, and exclusions join the packet before it is checked."""
    from tools.daily_research import search
    from tools.daily_research.recovery import MAX_QUARANTINE_FAILURES, exclude_located_items
    from tools.daily_research.runner import canonical
    from tools.daily_research.runner import digest as row_digest
    original = {"discovery_inventory": inventory(7000), "candidates": []}
    feedback = [{"path": f"/discovery_inventory/{i}", "reason": "discovery_inventory_invalid"} for i in range(7000)]
    _, exclusions = exclude_located_items(original, feedback)
    receipt = exclusions[0]
    assert len(receipt["failures"]) == MAX_QUARANTINE_FAILURES and receipt["failure_count"] == 7000
    assert receipt["failures_digest"] == row_digest(feedback)
    assert len(canonical(exclusions).encode()) < search.MAX_PACKET // 100


def test_exclusion_receipt_counts_toward_the_review_packet_ceiling(tmp_path):
    """Re-review of S5: exclusions join the packet before its ceiling check, so an oversized
    receipt refuses (raw retained) before the packet or its review file is written."""
    from tests.test_daily_research_search import fixture as search_fixture
    from tools.daily_research import search
    from tools.daily_research.runner import Refusal, canonical
    out = result(40)[0]
    out["coverage"].update(defined_run_scope=["Synthetic exact site/task industry/region scope"],
                           unresolved_promising_branches=[], completion_state="coverage_complete")
    runner, api, ledger = next(search_fixture.__wrapped__(tmp_path))
    api.raw, api.turn_status = canonical(out).encode(), "completed"
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review", row.get("error")
    before, review = deepcopy(row), ledger.read_bytes(DAY + "-review.json")
    oversized = {"revision": 1, "excluded": [{"field": "candidates", "index": i, "failures": [], "item_digest": "0" * 64}
                                             for i in range(search.MAX_PACKET // 60)]}
    with pytest.raises(Refusal, match="research_profile_packet_resource_ceiling_raw_retained"):
        runner.prepare_output(row, out, research_exclusions=oversized)
    assert row == before and ledger.read_bytes(DAY + "-review.json") == review
    runner.prepare_output(row, out, research_exclusions={"revision": 1, "excluded": []})
    assert row["packet"]["research_exclusions"] == {"revision": 1, "excluded": []}


def test_overflowing_number_literal_is_invalid_json_not_a_crash():
    from tools.daily_research.recovery import parse_artifact_json
    with pytest.raises(ValueError, match="nonfinite_json_number"):
        parse_artifact_json(b'{"discovery_inventory": [{"confidence": 1e400}]}')
    with pytest.raises(ValueError, match="nonfinite_json_number"):
        parse_artifact_json(b'```json\n{"x": -1e999}\n```')
    assert parse_artifact_json(b'{"x": 1.5e3}') == ({"x": 1500.0}, None)


def test_oversized_review_packet_is_repairable_feedback_not_a_blocked_run():
    """v3 removed count caps; display content beyond the 500 KB review packet used to
    raise only after validation (no repair route). The paged inventory is excluded."""
    from tools.daily_research import search
    from tools.daily_research.recovery import validation_feedback
    from tools.daily_research.runner import PACKET_OUTPUT_BUDGET, packet_overflow
    out, ctx, policy = result(4)
    assert PACKET_OUTPUT_BUDGET < search.MAX_PACKET and not packet_overflow(out)
    out["findings"] = ["Synthetic retained finding " + "y" * 1900 for _ in range(260)]
    assert packet_overflow(out)
    row = {"date": DAY, "research_contract_version": 3, "knowledge_context": ctx, "refresh_policy": policy,
           "search_provider": search.PROFILE, "discovery_profile": "adaptive-sites-v1"}
    reasons = {issue["reason"] for issue in validation_feedback(out, row, set(), NOW)}
    assert "research_packet_resource_ceiling_use_inventory" in reasons
    moved = deepcopy(out)
    moved["findings"] = moved["findings"][:3]
    moved["discovery_inventory"] = inventory(260)
    assert not packet_overflow(moved)
    other = dict(row, search_provider="native")
    assert "research_packet_resource_ceiling_use_inventory" not in {
        issue["reason"] for issue in validation_feedback(out, other, set(), NOW)}


def test_largest_output_the_packet_guard_admits_still_fits_the_review_packet(tmp_path):
    """Independent review S3: candidate normalization adds bytes after validation, so the
    guard must leave room for it; otherwise a valid output hits the unrepairable ceiling."""
    from tests.test_daily_research_search import fixture as search_fixture
    from tools.daily_research import search
    from tools.daily_research.recovery import validation_feedback
    from tools.daily_research.runner import canonical, packet_overflow
    def build(n):
        out = result(n)[0]
        out["coverage"].update(defined_run_scope=["Synthetic exact site/task industry/region scope"],
                               unresolved_promising_branches=[], completion_state="coverage_complete")
        return out
    n = 1
    while not packet_overflow(build(n + 1)):
        n += 1
    runner, api, _ledger = next(search_fixture.__wrapped__(tmp_path))
    api.raw, api.turn_status = canonical(build(n)).encode(), "completed"
    row = runner.start_or_resume()
    assert row["state"] == "awaiting_review", row.get("error")
    assert len(canonical(row["packet"]).encode()) <= search.MAX_PACKET
    over = build(n + 1)
    reasons = {f["reason"] for f in validation_feedback(over, {**row, "search_provider": search.PROFILE}, set(), NOW)}
    assert "research_packet_resource_ceiling_use_inventory" in reasons


def test_forged_inventory_manifest_with_null_source_refuses_instead_of_crashing(tmp_path):
    """Independent review N5: a repair revision without an artifact must not let a manifest
    with null source fields match it and crash the export with a TypeError."""
    import base64

    from tools.daily_research import render
    repair_input = b'{"synthetic": "repair input"}'
    row = {"date": DAY, "run_key": "blueprint-researcher:" + DAY,
           "validation_repairs": [{"number": 1, "input_file": DAY + "-repair-1-input.json",
                                   "request_digest": digest(json.loads(repair_input)), "state": "invalid"}],
           "packet": {"discovery_inventory_manifest": {"version": discovery.INVENTORY_VERSION, "complete_retention": True,
                      "page_count": 0, "pages": [], "source_artifact_file": None, "source_artifact_sha256": None}}}
    class Bridge:
        def call(self, name, **_):
            assert name == "snapshot"
            return {"row": row, "files": {"repair-1-input": base64.b64encode(repair_input).decode()}}
    with pytest.raises(Refusal, match="discovery_inventory_source_binding_invalid"):
        render.export_snapshot(Bridge(), DAY, tmp_path / "export")
