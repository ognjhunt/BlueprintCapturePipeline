"""Hermetic owner-direction grants and per-run paid expansion arbitration."""
import copy
from datetime import datetime, timedelta, timezone

import pytest

from tools.daily_research import allocation as a

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
COMMIT = "c" * 40
RUN = "blueprint-researcher:2026-10-05"


def direction(**changes):
    value = {"schema_version": a.DIRECTION, "version": 1, "supersedes": None, "per_run_limit_usd": "10.00",
             "sources": ["exa"], "scope": dict(a.SCOPE), "effective_from": "2026-10-04T18:00:00+00:00",
             "expires_at": "2027-01-02T18:00:00+00:00", "approval_reference": "owner-decision-2026-10-04",
             "approved_by": "owner", "issued_at": "2026-10-04T18:00:00+00:00", "reason": "Owner per-run allowance"}
    value.update(changes)
    return value


def entry(value=None):
    value = direction() if value is None else value
    sha = a.digest(value)
    return {"sha256": sha, "version": value["version"], "uri": a.uri(sha), "direction": value}


def control(value=None, *, enabled=True, commit=COMMIT):
    return {"source_commit": commit, "paid_expansion": {"enabled": enabled, "current": entry(value)}}


def row(**changes):
    value = {"run_key": RUN, "started_at": NOW.isoformat(), "research_runtime_seconds": 1200}
    value.update(changes)
    return value


def frozen(limit="10.00"):
    return a.grant(control(direction(per_run_limit_usd=limit)), row(), NOW)


def claim(micros, state="submission_unresolved"):
    return {"source": "exa", "intent_sha256": "e" * 64, "reserved_micros": micros, "state": state}


def test_grant_freezes_exact_direction_limit_release_and_deadline():
    owner = control()
    grant = a.grant(owner, row(), NOW)
    current = owner["paid_expansion"]["current"]
    assert grant == {"schema_version": a.GRANT, "state": "granted", "run_key": RUN, "frozen_at": NOW.isoformat(),
                     "direction_sha256": current["sha256"], "direction_uri": current["uri"], "version": 1,
                     "grant_id": a.grant_id(current["sha256"], RUN), "sources": ["exa"], "limit_micros": 10_000_000,
                     "per_start_max_micros": 5_000_000, "source_commit": COMMIT,
                     "approval_reference": "owner-decision-2026-10-04",
                     "valid_until": (NOW + timedelta(seconds=1200)).isoformat()}
    assert a.granted(grant) is None and a.problem(grant, [], 5_000_000, NOW, control=owner) is None
    # A direction expiring before the research deadline bounds the grant instead.
    early = a.grant(control(direction(expires_at="2026-10-05T12:05:00+00:00")), row(), NOW)
    assert early["valid_until"] == "2026-10-05T12:05:00+00:00"


def test_grant_for_a_sixty_minute_row_lasts_its_forty_five_minute_research_window():
    owner = control()
    grant = a.grant(owner, row(research_runtime_seconds=2700), NOW)
    assert grant["state"] == "granted" and grant["valid_until"] == (NOW + timedelta(seconds=2700)).isoformat()
    # After the old 1200-second research window the grant still admits a start.
    assert a.problem(grant, [], 5_000_000, NOW + timedelta(seconds=1201), control=owner) is None
    assert a.problem(grant, [], 5_000_000, NOW + timedelta(seconds=2700), control=owner) == "paid_expansion_expired"
    # The owner-approved 3600-second envelope bounds every research window.
    assert a.grant(owner, row(research_runtime_seconds=3600), NOW)["state"] == "granted"
    refused = a.grant(owner, row(research_runtime_seconds=3601), NOW)
    assert refused["state"] == "refused" and refused["code"] == "paid_expansion_run_context_invalid"


@pytest.mark.parametrize(("owner", "context", "code"), [
    (None, {}, "paid_expansion_disabled"),
    ({"source_commit": COMMIT}, {}, "paid_expansion_disabled"),
    (control(enabled=False), {}, "paid_expansion_disabled"),
    ({"source_commit": COMMIT, "paid_expansion": {"enabled": "true", "current": entry()}}, {}, "paid_expansion_disabled"),
    ({"source_commit": COMMIT, "paid_expansion": {"enabled": True, "current": entry(), "extra": 1}}, {}, "paid_expansion_direction_invalid"),
    ({"source_commit": COMMIT, "paid_expansion": {"enabled": True}}, {}, "paid_expansion_direction_invalid"),
    (control(direction(schema_version="v0")), {}, "paid_expansion_direction_invalid"),
    (control({**direction(), "extra": True}), {}, "paid_expansion_direction_invalid"),
    (control(direction(version=2)), {}, "paid_expansion_direction_invalid"),
    (control(direction(supersedes="a" * 64)), {}, "paid_expansion_direction_invalid"),
    (control(direction(version=True)), {}, "paid_expansion_direction_invalid"),
    (control(direction(sources=["exa", "exa"])), {}, "paid_expansion_direction_invalid"),
    (control(direction(sources=["findall"])), {}, "paid_expansion_direction_invalid"),
    (control(direction(sources=[])), {}, "paid_expansion_direction_invalid"),
    (control(direction(approval_reference="PENDING-owner")), {}, "paid_expansion_direction_invalid"),
    (control(direction(approval_reference=" ")), {}, "paid_expansion_direction_invalid"),
    (control(direction(reason="café")), {}, "paid_expansion_direction_invalid"),
    (control(direction(issued_at="2026-10-04T18:00:01+00:00")), {}, "paid_expansion_direction_invalid"),
    (control(direction(expires_at="2026-10-04T18:00:00+00:00")), {}, "paid_expansion_direction_invalid"),
    (control(direction(expires_at="2027-10-06T18:00:00+00:00")), {}, "paid_expansion_direction_invalid"),
    (control(direction(effective_from="2026-10-04T18:00:00Z")), {}, "paid_expansion_direction_invalid"),
    (control(direction(effective_from="2026-10-04T18:00:00.5+00:00")), {}, "paid_expansion_direction_invalid"),
    (control(direction(per_run_limit_usd="200.00")), {}, "paid_expansion_limit_invalid"),
    (control(direction(per_run_limit_usd=10)), {}, "paid_expansion_limit_invalid"),
    (control(direction(scope={**a.SCOPE, "agent_id": "agent_other"})), {}, "paid_expansion_scope_mismatch"),
    (control(direction(scope={**a.SCOPE, "firestore_root": "other/root"})), {}, "paid_expansion_scope_mismatch"),
    (control(commit="PENDING-reviewed-package"), {}, "paid_expansion_source_commit_unverified"),
    (control(commit=None), {}, "paid_expansion_source_commit_unverified"),
    (control(direction(effective_from="2026-10-05T12:00:01+00:00")), {}, "paid_expansion_not_yet_effective"),
    (control(direction(expires_at="2026-10-05T12:00:00+00:00")), {}, "paid_expansion_expired"),
    (control(), {"started_at": (NOW - timedelta(seconds=1200)).isoformat()}, "paid_expansion_expired"),
    (control(), {"run_key": "other-researcher:2026-10-05"}, "paid_expansion_scope_mismatch"),
    (control(), {"research_runtime_seconds": 0}, "paid_expansion_run_context_invalid"),
    (control(), {"started_at": "2026-10-05T12:00:00"}, "paid_expansion_run_context_invalid"),
    (control(), {"started_at": None}, "paid_expansion_run_context_invalid"),
])
def test_every_grant_refusal_is_named_recorded_and_never_raises(owner, context, code):
    grant = a.grant(owner, row(**context), NOW)
    assert grant["state"] == "refused" and grant["code"] == code
    assert set(grant) == {"schema_version", "state", "code", "run_key", "frozen_at", "direction_sha256"}
    assert a.granted(grant) == code and a.problem(grant, [], 1_000_000, NOW, control=owner) == code


@pytest.mark.parametrize("change", [{"sha256": "0" * 64}, {"version": 2}, {"version": True},
                                    {"uri": "gs://another-bucket/direction.json"}])
def test_worker_verifies_content_address_without_reading_object_storage(change):
    owner = control()
    owner["paid_expansion"]["current"].update(change)
    grant = a.grant(owner, row(), NOW)
    assert grant["code"] == "paid_expansion_direction_digest_mismatch"


@pytest.mark.parametrize(("amount", "micros"), [
    ("1", 1_000_000), ("1.00", 1_000_000), ("10", 10_000_000), ("10.00", 10_000_000), ("20.05", 20_050_000),
    ("99.99", 99_990_000), ("100", 100_000_000), ("100.00", 100_000_000),
    ("100.01", None), ("200", None), ("999.99", None), ("1000", None), ("10000", None), ("0.99", None), ("0", None),
    ("01.00", None), ("10.0", None), ("10.000", None), ("1.5", None), ("1e1", None), ("+10", None), (" 10", None),
    ("10 ", None), ("$10", None), ("1,000", None), ("1２", None), ("", None), (None, None), (10, None), (10.0, None),
])
def test_owner_amount_format_and_hard_code_ceiling_reject_typos(amount, micros):
    assert a.micros(amount) == micros
    assert a.HARD_CEILING_MICROS == 100_000_000
    if micros is None:
        assert a.direction_problem(direction(per_run_limit_usd=amount)) == "paid_expansion_limit_invalid"


@pytest.mark.parametrize(("limit", "per_start"), [
    ("10.00", 5_000_000), ("20.00", 10_000_000), ("30.00", 15_000_000), ("1.00", 1_000_000), ("1.50", 1_000_000),
    ("3.00", 1_500_000), ("100.00", 50_000_000)])
def test_per_start_maximum_scales_with_the_owner_amount(limit, per_start):
    grant = frozen(limit)
    assert grant["per_start_max_micros"] == per_start == a.per_start_max(a.micros(limit))


def test_remaining_is_limit_minus_every_durable_reservation_and_second_start_is_refused():
    grant, owner = frozen("10.00"), control()
    assert a.headroom(grant, [claim(5_000_000)]) == {"reserved_micros": 5_000_000, "remaining_micros": 5_000_000,
                                                     "max_start_micros": 5_000_000}
    assert a.problem(grant, [claim(5_000_000)], 5_000_000, NOW, control=owner) is None
    used = [claim(5_000_000), claim(5_000_000)]
    assert a.problem(grant, used, 1_000_000, NOW, control=owner) == "paid_expansion_cap_exceeds_remaining"
    assert a.headroom(grant, used)["remaining_micros"] == 0
    assert a.problem(grant, [claim(7_000_000)], 4_000_000, NOW, control=owner) == "paid_expansion_cap_exceeds_remaining"
    assert a.problem(grant, [], 5_000_001, NOW, control=owner) == "paid_expansion_cap_exceeds_per_start_maximum"


@pytest.mark.parametrize("state", ["submission_unresolved", "running", "completed", "failed", "cancelled"])
def test_uncertain_or_terminal_claims_hold_their_whole_reservation(state):
    exa = {"cap_micros": 4_000_000, "intent_sha256": "f" * 64, "state": state, "run_id": None,
           "provider_record": {"costDollars": {"total": 0.25}}}
    found = a.claims({"exa_expansion": exa})
    assert found == [{"source": "exa", "intent_sha256": "f" * 64, "reserved_micros": 4_000_000}]
    assert a.headroom(frozen(), found)["remaining_micros"] == 6_000_000


@pytest.mark.parametrize("reserved", [None, 0, -1, True, 1.5, "1000000"])
def test_unknowable_debits_fail_closed(reserved):
    grant = frozen()
    assert a.headroom(grant, [claim(reserved)])["remaining_micros"] is None
    assert a.problem(grant, [claim(reserved)], 1_000_000, NOW, control=control()) == "paid_expansion_claims_unverified"


@pytest.mark.parametrize("cap", [0, -1, True, 1.0, "1000000", None])
def test_cap_must_be_positive_integer_micros(cap):
    assert a.problem(frozen(), [], cap, NOW, control=control()) == "paid_expansion_cap_invalid"


def test_live_brake_release_and_validity_fences_apply_to_a_frozen_grant():
    grant = frozen()
    assert a.problem(grant, [], 1_000_000, NOW, control=control(enabled=False)) == "paid_expansion_disabled"
    assert a.problem(grant, [], 1_000_000, NOW, control={"source_commit": COMMIT}) == "paid_expansion_disabled"
    assert a.problem(grant, [], 1_000_000, NOW, control=None) == "paid_expansion_disabled"
    assert a.problem(grant, [], 1_000_000, NOW, control=control(commit="d" * 40)) == "paid_expansion_source_commit_changed"
    later = NOW + timedelta(seconds=1200)
    assert a.problem(grant, [], 1_000_000, later, control=control()) == "paid_expansion_expired"
    assert a.problem(grant, [], 1_000_000, NOW, control=control(), source="findall") == "paid_expansion_source_not_directed"


def successor(limit, **changes):
    return direction(per_run_limit_usd=limit, version=2, supersedes=entry()["sha256"], **changes)


def test_a_raised_amount_waits_for_the_next_run_grant():
    grant, raised = frozen("10.00"), control(successor("30.00"))
    assert a.effective_limit(grant, raised) == 10_000_000
    assert a.problem(grant, [], 5_000_000, NOW, control=raised) is None
    assert a.problem(grant, [], 5_000_001, NOW, control=raised) == "paid_expansion_cap_exceeds_per_start_maximum"
    assert a.headroom(grant, [claim(5_000_000)])["remaining_micros"] == 5_000_000
    following = a.grant(raised, row(run_key="blueprint-researcher:2026-10-06"), NOW)
    assert (following["limit_micros"], following["per_start_max_micros"]) == (30_000_000, 15_000_000)


def test_live_control_can_only_tighten_a_frozen_grant():
    grant = frozen("30.00")
    lowered = control(successor("6.00"))
    assert a.effective_limit(grant, lowered) == 6_000_000
    assert a.headroom(grant, [], a.effective_limit(grant, lowered))["max_start_micros"] == 3_000_000
    assert a.problem(grant, [], 3_000_000, NOW, control=lowered) is None
    assert a.problem(grant, [], 3_000_001, NOW, control=lowered) == "paid_expansion_cap_exceeds_per_start_maximum"
    assert a.problem(grant, [claim(4_000_000)], 3_000_000, NOW, control=lowered) == "paid_expansion_cap_exceeds_remaining"
    # Brake, then a corrected lower amount: the correction reaches the run already in progress.
    assert a.problem(grant, [], 1_000_000, NOW, control=control(successor("6.00"), enabled=False)) == "paid_expansion_disabled"
    assert a.problem(grant, [], 15_000_000, NOW, control=lowered) == "paid_expansion_cap_exceeds_remaining"
    assert a.problem(grant, [], 1_000_000, NOW, control=control(successor("6.00", reason="caf\u00e9"))) == (
        "paid_expansion_direction_invalid")


def test_a_current_direction_without_the_source_stops_new_starts():
    owner = control()
    owner["paid_expansion"]["current"]["direction"]["sources"] = []
    assert a.problem(frozen(), [], 1_000_000, NOW, control=owner) == "paid_expansion_direction_invalid"


@pytest.mark.parametrize("changes, code", [
    ({"expires_at": "2026-10-05T12:00:10+00:00"}, "paid_expansion_expired"),
    ({"effective_from": "2026-10-05T12:01:00+00:00"}, "paid_expansion_not_yet_effective"),
])
def test_live_successor_interval_tightens_an_unexpired_frozen_grant(changes, code):
    grant = frozen("30.00")
    assert a.problem(grant, [], 1_000_000, NOW + timedelta(seconds=11),
                     control=control(successor("30.00", **changes))) == code


def test_refused_record_keeps_bindings_and_names_the_store_code():
    record = a.refused(frozen(), "paid_expansion_grant_not_admitted")
    assert record["state"] == "refused" and record["code"] == "paid_expansion_grant_not_admitted"
    assert set(record) == {"schema_version", "state", "code", "run_key", "frozen_at", "direction_sha256"}
    assert a.granted(record) == "paid_expansion_grant_not_admitted"


@pytest.mark.parametrize("change", [
    {"limit_micros": 20_000_000}, {"per_start_max_micros": 10_000_000}, {"limit_micros": 200_000_000, "per_start_max_micros": 50_000_000},
    {"grant_id": "0" * 64}, {"run_key": "blueprint-researcher:2026-10-06"}, {"sources": ["findall"]}, {"state": "maybe"},
    {"valid_until": "not-a-time"}, {"valid_until": "2026-10-05T12:20:00"}, {"schema_version": "v0"}, {"extra": True}])
def test_tampered_or_malformed_grants_are_refused(change):
    grant = {**frozen(), **change}
    assert a.problem(grant, [], 1_000_000, NOW, control=control()) in {"paid_expansion_grant_invalid", "paid_expansion_grant_missing"}


def test_refusal_codes_are_bounded_and_missing_grant_is_named():
    assert a.granted(None) == a.granted({}) == "paid_expansion_grant_missing"
    assert a.granted({"schema_version": a.GRANT, "state": "refused", "code": "free text"}) == "paid_expansion_grant_invalid"


def test_older_package_control_without_paid_expansion_records_no_grant():
    legacy = {"schema_version": "blueprint.research-control.v1", "enabled": True, "source_commit": COMMIT,
              "exa_expansion_allocation": {"limit_micros": 5_000_000}}
    grant = a.grant(legacy, row(), NOW)
    assert grant["state"] == "refused" and grant["code"] == "paid_expansion_disabled"
    assert a.diagnostic({"paid_expansion_grant": grant})["reasons"] == ["paid_expansion_disabled"]
    assert a.diagnostic({})["grant_state"] == "missing" and a.diagnostic({})["remaining_micros"] is None


def test_diagnostic_reports_frozen_amounts_and_live_fence_without_inventing_cost():
    found = {"paid_expansion_grant": frozen(), "exa_expansion": {"cap_micros": 2_000_000, "intent_sha256": "a" * 64}}
    status = a.diagnostic(found)
    assert status["limit_micros"] == status["effective_limit_micros"] == 10_000_000
    assert status["reserved_micros"] == 2_000_000 and status["remaining_micros"] == 8_000_000
    assert status["max_start_micros"] == 5_000_000 and status["reasons"] == []
    braked = a.diagnostic(found, control(enabled=False), now=NOW)
    assert braked["remaining_micros"] == 8_000_000 and braked["max_start_micros"] == 0
    assert braked["reasons"] == ["paid_expansion_disabled"]
    assert braked["claim_created"] is False and braked["provider_started"] is False
    lowered = a.diagnostic(found, control(successor("6.00")), now=NOW)
    assert lowered["effective_limit_micros"] == 6_000_000 and lowered["remaining_micros"] == 4_000_000
    assert lowered["max_start_micros"] == 3_000_000 and lowered["reasons"] == []


def test_direction_digest_is_canonical_and_copy_independent():
    value = direction()
    assert a.digest(value) == a.digest(copy.deepcopy(dict(reversed(list(value.items())))))
    assert a.uri(a.digest(value)).startswith("gs://blueprint-8c1ca.appspot.com/operations/research/paid-expansion/")
    assert a.uri(a.digest(value)).endswith("/direction.json")
