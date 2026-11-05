"""A finished screening attempt settles once, to its own run's measured cost, and never above its cap.

The figures are capture-447ae4f4's pi05_droid run (2026-11-04): Vast instance 59145013 at $0.6074/h
for 1252 s ($0.211272) plus $0.2314 of priced transfer, under a $4 hard cap.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_job_ceiling as job_ceiling
from blueprint_pipeline.beta_policy_screening_closeout import SCHEMA as ABSENCE_SCHEMA, SCOPE as ABSENCE_SCOPE
from blueprint_pipeline.beta_policy_screening_settlement import (
    SETTLEMENT_BASIS, SettlementEvidenceError, attempt_settlement, settled_cost,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_beta_policy_screening_coordinator import Backend, native_result, plan
from tests.test_beta_policy_screening_waitlist import waitlisted_plan

RUNTIME, TRANSFER = 0.211272, 0.2314
STAMP = "2026-11-04T20:00:00+00:00"


def _sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return {"path": str(path), "size_bytes": len(raw), "sha256": _sha(raw)}


def prepared_record(attempt_dir: Path, candidate_id: str) -> dict:
    value = {"schema_version": "beta_policy_screening_prepared.v1", "candidate_id": candidate_id,
             "attempt_dir": str(Path(attempt_dir).resolve()), "run_id": "beta-run-" + candidate_id,
             "prepared_digest": ""}
    value["prepared_digest"] = canonical_digest(value, digest_field="prepared_digest")
    return value


def write_run(attempt_dir: Path, prepared: dict, *, instance_id: int = 59145013, runtime: float = RUNTIME,
              transfer: float = TRANSFER, total: float | None = None, hard_cap: float = 4.0,
              run_root: Path | None = None, edit=None) -> None:
    """One finished Vast run's records, as the canary allocator and the scoped closeout leave them."""
    edit = edit or (lambda name, value: value)
    total = round(runtime + transfer, 6) if total is None else total
    run = Path(run_root or attempt_dir) / "allocator" / "attempts" / "attempt_001" / "vast_provider_run"
    _write(run / "vast_budget_ledger.json", edit("ledger", {
        "schema_version": "vast_budget_ledger.v1", "generated_at": "2026-11-04T18:48:36+00:00",
        "status": "completed", "target_spend_usd": hard_cap, "hard_cap_usd": hard_cap,
        "max_hourly_rate_usd": 0.8, "max_live_runtime_minutes": 240,
        "selected_hourly_rate_usd": 0.6074074074074074, "vast_instance_ids": [instance_id],
        "actual_live_runtime_seconds_observed_by_adapter": 1252.1737463911995,
        "estimated_runtime_cost_usd": runtime, "expected_provider_download_bytes": 51238455276,
        "expected_provider_upload_bytes": 8000000000, "projected_provider_transfer_cost_usd": transfer,
        "provider_transfer_cost_priced": True, "estimated_cost_usd": total, "actual_cost_usd": None,
        "actual_cost_source": "not_available_from_instance_probe_api", "estimated_spend_under_target": True,
        "estimated_spend_under_hard_cap": True, "continuing_spend_from_this_run": False,
        "raw_secret_values_recorded": False}))
    teardown = _write(run / "vast_teardown_manifest.json", edit("teardown", {
        "schema_version": "vast_teardown_manifest.v1", "status": "completed", "vast_instance_ids": [instance_id],
        "teardown_actions_performed": [{"instance_id": instance_id, "action": "destroy_instance",
                                        "http_status_code": 200, "response": {"success": True},
                                        "status": "completed"}],
        "runner_gpu_teardown_completed": True, "continuing_spend_from_this_run": False,
        "retention_authorized": False,
        "zero_continuing_spend_scope": "all Vast instances created by this adapter were destroyed"}))
    adapter = _write(run / "vast_provider_adapter_result.json", edit("adapter", {
        "schema_version": "vast_provider_adapter_result.v1", "status": "completed",
        "vast_instance_ids": [instance_id], "estimated_cost_usd": total, "continuing_spend_from_this_run": False}))
    result = edit("result", {
        "schema_version": "native_task_arena_policy_canary_session_result.v1", "status": "blocked",
        "attempt_root": str(run.parent), "estimated_cost_usd": total, "hard_cap_usd": hard_cap,
        "provider_closeout": {"adapter_result": adapter, "teardown_manifest": teardown, "estimated_cost_usd": total,
                              "provider_zero_confirmed": True, "warm_session_retained": False,
                              "all_staged_objects_absent": True},
        "continuing_spend_from_this_run": False, "all_staged_objects_absent": True})
    _write(Path(attempt_dir) / "allocator_result.json", result)
    receipt = edit("receipt", {
        "schema_version": ABSENCE_SCHEMA, "scope": ABSENCE_SCOPE, "provider": "vast",
        "prepared_digest": prepared["prepared_digest"], "adapter_digest": canonical_digest(result),
        "instance_id": instance_id, "resource_name": "blueprint-native-task-policy-canary-" + prepared["run_id"],
        "collection_started_at_epoch": 1793818820.9584014, "observed_at_epoch": 1793818821.3331785,
        "exact_instance": {"provider": "vast", "status": "absent", "instance_id": str(instance_id),
                           "api_confirmed": True, "provider_absence_confirmed": True},
        "inventory": {"status": "observed", "provider": "vast", "api_confirmed": True, "name_prefix": "",
                      "blockers": [], "resources": [], "live_resource_count": 0,
                      "observed_at_epoch": 1793818821.149574},
        "unrelated_live_resources": [], "raw_provider_response_recorded": False,
        "global_provider_zero_claimed": False, "status": "verified", "blockers": [],
        "provider_absence_confirmed": True, "receipt_digest": ""})
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(Path(attempt_dir) / "provider_absence" / (canonical_digest(receipt)[7:] + ".json"), receipt)


def committed_attempt(tmp_path: Path, candidate: dict, *, maximum: str = "4", **run) -> tuple[Path, dict]:
    attempt_dir = tmp_path / "attempts" / "screen-0123456789abcdef0123456789abcdef"
    attempt_dir.mkdir(parents=True)
    prepared = prepared_record(attempt_dir, candidate["candidate_id"])
    write_run(attempt_dir, prepared, **run)
    return attempt_dir, {"attempt_id": attempt_dir.name, "generation": 0, "maximum_cost_usd": maximum,
                         "reservation_status": "committed", "provider_allocation_observed": True,
                         "prepared": prepared}


def settle(attempt_dir, attempt, candidate):
    return attempt_settlement(attempt_dir=attempt_dir, attempt=attempt, candidate=candidate, settled_at=STAMP)


def test_settles_a_real_shaped_run_at_its_ledger_estimate_rounded_up_to_the_cent(tmp_path):
    candidate = plan()["candidates"][0]
    attempt_dir, attempt = committed_attempt(tmp_path, candidate)
    fields = settle(attempt_dir, attempt, candidate)
    receipt = fields["settlement_receipt"]
    # $0.211272 runtime + $0.2314 transfer = $0.442672, rounded up.
    assert fields["settled_cost_usd"] == "0.45" and fields["settlement_basis"] == SETTLEMENT_BASIS
    assert fields["settled_at"] == STAMP and fields["settlement_receipt_digest"] == receipt["receipt_digest"]
    assert receipt["estimated_runtime_cost_usd"] == "0.211272"
    assert receipt["non_runtime_charges_usd"] == {"projected_provider_transfer_cost_usd": "0.2314"}
    assert receipt["instance_id"] == 59145013 and receipt["attempt_id"] == attempt["attempt_id"]
    assert receipt["actual_cost_usd"] is None and receipt["continuing_spend_from_this_run"] is False
    assert settled_cost({**attempt, **fields}) == Decimal("0.45")


@pytest.mark.parametrize(("runtime", "transfer", "expected"), [
    (0.211272, 0.2314, "0.45"), (0.2, 0.24, "0.44"), (0.200001, 0.0, "0.21"), (0.162328, 1.151049, "1.32"),
])
def test_the_measured_estimate_rounds_up_to_the_cent(tmp_path, runtime, transfer, expected):
    candidate = plan()["candidates"][0]
    attempt_dir, attempt = committed_attempt(tmp_path, candidate, runtime=runtime, transfer=transfer)
    assert settle(attempt_dir, attempt, candidate)["settled_cost_usd"] == expected


def test_a_ledger_total_above_its_parts_settles_at_the_total(tmp_path):
    candidate = plan()["candidates"][0]
    attempt_dir, attempt = committed_attempt(tmp_path, candidate, total=0.442673)
    assert settle(attempt_dir, attempt, candidate)["settled_cost_usd"] == "0.45"


def test_never_settles_above_the_reserved_maximum(tmp_path):
    candidate = plan()["candidates"][0]
    attempt_dir, attempt = committed_attempt(tmp_path, candidate, runtime=3.9, transfer=0.100001)
    with pytest.raises(SettlementEvidenceError, match="above_reservation"):
        settle(attempt_dir, attempt, candidate)


def test_a_hosted_endpoint_adds_its_whole_bound_and_the_total_stays_under_the_maximum(tmp_path):
    candidate = plan(cost="4.00")["candidates"][0]
    candidate["execution_spec"] = {**candidate["execution_spec"], "credential_file": "/operator/token",
                                   "provider_hard_cap_usd": "3.00", "hosted_inference_maximum_usd": "1.00"}
    attempt_dir, attempt = committed_attempt(tmp_path, candidate, hard_cap=3.0)
    assert settle(attempt_dir, attempt, candidate)["settled_cost_usd"] == "1.45"
    # A provider cap the hosted bound does not leave room for is not this attempt's.
    other_dir, other = committed_attempt(tmp_path / "other", candidate, hard_cap=4.0)
    with pytest.raises(SettlementEvidenceError, match="ledger_cap_not_this_attempt"):
        settle(other_dir, other, candidate)


def _set(name, key, value):
    def edit(record, current):
        if record == name:
            current[key] = value
        return current
    return edit


def _closeout(key, value):
    def edit(record, current):
        if record == "result":
            current["provider_closeout"][key] = value
        return current
    return edit


@pytest.mark.parametrize(("edit", "reason"), [
    (_set("result", "continuing_spend_from_this_run", True), "teardown_unproven"),
    (_set("result", "all_staged_objects_absent", False), "teardown_unproven"),
    (_closeout("provider_zero_confirmed", False), "teardown_unproven"),
    (_closeout("warm_session_retained", True), "teardown_unproven"),
    (_set("teardown", "status", "blocked"), "teardown_unproven"),
    (_set("teardown", "continuing_spend_from_this_run", True), "teardown_unproven"),
    (_set("teardown", "retention_authorized", True), "teardown_unproven"),
    (_set("teardown", "vast_instance_ids", [59145013, 59145014]), "teardown_unproven"),
    (_set("receipt", "status", "blocked"), "provider_zero_unproven"),
    (_set("receipt", "provider_absence_confirmed", False), "provider_zero_unproven"),
    (_set("receipt", "prepared_digest", "sha256:" + "0" * 64), "provider_zero_unproven"),
    (_set("receipt", "instance_id", 1), "provider_zero_unproven"),
    (_set("receipt", "adapter_digest", "sha256:" + "0" * 64), "provider_zero_unproven"),
    (_set("receipt", "unrelated_live_resources", [{"instance_id": "59145013", "name": "x"}]),
     "provider_zero_unproven"),
    (_set("ledger", "vast_instance_ids", [59145014]), "ledger_invalid"),
    (_set("ledger", "status", "live_instance_created"), "ledger_invalid"),
    (_set("ledger", "continuing_spend_from_this_run", True), "ledger_invalid"),
    (_set("ledger", "provider_transfer_cost_priced", False), "ledger_invalid"),
    (_set("ledger", "actual_cost_usd", 0.30), "ledger_invalid"),
    (_set("ledger", "estimated_storage_cost_usd", 0.05), "ledger_charge_unknown"),
    (_set("ledger", "estimated_cost_usd", 0.30), "ledger_inconsistent"),
    (_set("result", "estimated_cost_usd", 0.30), "estimate_not_this_run"),
    (_set("adapter", "vast_instance_ids", [59145014]), "adapter_mismatch"),
])
def test_no_settlement_when_teardown_zero_or_the_ledger_binding_is_not_proven(tmp_path, edit, reason):
    candidate = plan()["candidates"][0]
    attempt_dir, attempt = committed_attempt(tmp_path, candidate, edit=edit)
    with pytest.raises(SettlementEvidenceError, match=reason):
        settle(attempt_dir, attempt, candidate)


def test_records_from_another_attempt_or_a_changed_record_do_not_settle(tmp_path):
    candidate = plan()["candidates"][0]
    attempt_dir, attempt = committed_attempt(tmp_path, candidate, run_root=tmp_path / "attempts" / "screen-other")
    with pytest.raises(SettlementEvidenceError, match="record_missing"):
        settle(attempt_dir, attempt, candidate)
    attempt_dir, attempt = committed_attempt(tmp_path / "changed", candidate)
    ledger = attempt_dir / "allocator/attempts/attempt_001/vast_provider_run/vast_teardown_manifest.json"
    ledger.write_text(ledger.read_text().replace('"completed"', '"completed" '))
    with pytest.raises(SettlementEvidenceError, match="reference_changed:teardown"):
        settle(attempt_dir, attempt, candidate)
    # The attempt's own identity binds the records: another directory name or candidate is refused.
    attempt_dir, attempt = committed_attempt(tmp_path / "renamed", candidate)
    with pytest.raises(SettlementEvidenceError, match="attempt_unbound"):
        settle(attempt_dir, {**attempt, "attempt_id": "screen-ffffffffffffffffffffffffffffffff"}, candidate)
    with pytest.raises(SettlementEvidenceError, match="attempt_unbound"):
        settle(attempt_dir, attempt, {**candidate, "candidate_id": "candidate-999"})
    with pytest.raises(SettlementEvidenceError, match="attempt_unbound"):
        settle(attempt_dir, {**attempt, "reservation_status": "reserved"}, candidate)


def test_an_unrecorded_or_altered_settlement_counts_the_whole_maximum(tmp_path):
    candidate = plan()["candidates"][0]
    attempt_dir, attempt = committed_attempt(tmp_path, candidate)
    settled = {**attempt, **settle(attempt_dir, attempt, candidate)}
    assert settled_cost(attempt) is None
    assert settled_cost({**settled, "settled_cost_usd": "0.01"}) is None
    assert settled_cost({**settled, "maximum_cost_usd": "0.40"}) is None
    receipt = {**settled["settlement_receipt"], "settled_cost_usd": "0.01"}
    assert settled_cost({**settled, "settled_cost_usd": "0.01", "settlement_receipt": receipt}) is None


# --- The coordinator: settle once, fund from what was freed, publish what the WebApp accepts. ---


class Measured(Backend):
    """Every launch allocates one Vast instance, tears it down and leaves the allocator's records."""

    def __init__(self, value, *, running_first=False, edit=None):
        super().__init__()
        self.value, self.running_first, self.edit, self.instance = value, running_first, edit, 59145012

    def prepare_candidate(self, *, candidate, attempt_dir, **kwargs):
        self.preparations += 1
        return prepared_record(attempt_dir, candidate["candidate_id"])

    def _finish(self, candidate, attempt_dir, prepared):
        self.instance += 1
        write_run(Path(attempt_dir), dict(prepared), instance_id=self.instance, edit=self.edit)
        return {"status": "completed", "provider_allocation_performed": True, "provider_zero_verified": True,
                "result": native_result(self.value, candidate)}

    def launch_candidate(self, *, candidate, attempt_dir, prepared, **kwargs):
        self.launches += 1
        if self.running_first:
            return {"status": "running", "provider_allocation_performed": True, "handle": {"pid": 1}}
        return self._finish(candidate, attempt_dir, prepared)

    def reconcile_candidate(self, *, candidate, attempt_dir, prepared, **kwargs):
        self.reconciliations += 1
        return self._finish(candidate, attempt_dir, prepared)


def run(root, backend):
    return coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend,
                                               max_launches=10)


def ledger(root):
    return json.loads((Path(root) / coordinator.LEDGER_FILENAME).read_text())


def attempts(root, campaign_id="campaign-1"):
    rows = ledger(root)["campaigns"][campaign_id]["candidates"]
    return {key: row["attempt"] for key, row in sorted(rows.items()) if row.get("attempt")}


def site_budget(root, site_id="site-job-1"):
    return coordinator._budget(ledger(root), site_id)


def reseal(root, change):
    """Rewrite the ledger as the code before settlement left it (a deploy's first pass reads such a one)."""
    value = ledger(root)
    change(value)
    value["ledger_digest"] = canonical_digest(value, digest_field="ledger_digest")
    (Path(root) / coordinator.LEDGER_FILENAME).write_text(json.dumps(value))


def unsettle(value):
    for state in value["campaigns"].values():
        for row in state["candidates"].values():
            attempt = row.get("attempt") or {}
            for key in ("settled_cost_usd", "settlement_basis", "settled_at", "settlement_receipt_digest",
                        "settlement_receipt", "settlement_held"):
                attempt.pop(key, None)


class WebApp:
    """The WebApp's acceptance of each campaign's publications, as ``screeningPublicationTransition`` and the
    publication schema decide it (Blueprint-WebApp ``server/utils/betaPolicyScreeningContract.ts``)."""

    TERMINAL = {"completed", "blocked", "unavailable", "deferred", "infeasible"}
    RANK = {"pending": 0, "preparing": 1, "launching": 2, "running": 3}

    def __init__(self):
        self.accepted: dict[str, dict] = {}
        self.latest: dict | None = None

    def deliver(self, root):
        for value in coordinator.pending_publications(queue_root=root):
            self.accept(value)
            assert coordinator.acknowledge_publication(value["campaign_id"], value["publication_digest"],
                                                       queue_root=root)

    def accept(self, value):
        budget = value["budget"]
        assert set(budget) == {"currency", "limit_usd", "committed_upper_bound_usd", "spent_upper_bound_usd",
                               "reserved_usd"}
        limit, committed, spent, reserved = (Decimal(budget[key]) for key in (
            "limit_usd", "committed_upper_bound_usd", "spent_upper_bound_usd", "reserved_usd"))
        assert limit <= 15 and spent + reserved == committed <= limit, "Screening must retain the $15 ceiling"
        previous = self.accepted.get(value["campaign_id"])
        if previous is not None and previous["publication_digest"] != value["publication_digest"]:
            assert value["revision"] > previous["revision"], "screening_revision_conflict"
            if previous["status"] in {"completed", "blocked"}:
                assert value["status"] == previous["status"] and value["candidates"] == previous["candidates"]
            assert value["generated_at"] >= previous["generated_at"], "screening_progress_regression"
            assert spent >= Decimal(previous["budget"]["spent_upper_bound_usd"]), "screening_progress_regression"
            for old, new in zip(previous["candidates"], value["candidates"]):
                if old["status"] in self.TERMINAL:
                    assert old == new, "screening_terminal_candidate_conflict"
                waiting = new["status"] == "preparing" and "beta_screening_waiting_for_capacity" in new["blockers"]
                if not waiting and old["status"] in self.RANK and new["status"] in self.RANK:
                    assert self.RANK[new["status"]] >= self.RANK[old["status"]], "screening_progress_regression"
        self.accepted[value["campaign_id"]] = value
        if self.latest is None or value["generated_at"] >= self.latest["generated_at"]:
            self.latest = value


def test_a_finished_attempt_settles_in_the_same_write_that_commits_it(tmp_path):
    value = plan(count=2)
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    webapp = WebApp()
    webapp.deliver(tmp_path)
    run(tmp_path, Measured(value))
    webapp.deliver(tmp_path)
    for attempt in attempts(tmp_path).values():
        assert attempt["reservation_status"] == "committed" and attempt["settled_cost_usd"] == "0.45"
        assert attempt["settlement_basis"] == "adapter_budget_ledger_estimate" and attempt["settled_at"]
    assert site_budget(tmp_path)["spent_upper_bound_usd"] == "0.9"
    # No publication ever showed either run at its $4 cap.
    assert webapp.accepted["campaign-1"]["budget"] == {
        "currency": "USD", "limit_usd": "15", "committed_upper_bound_usd": "0.9", "spent_upper_bound_usd": "0.9",
        "reserved_usd": "0"}


def test_unproven_records_keep_the_whole_maximum_and_are_decided_once(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Measured(value, edit=_closeout("provider_zero_confirmed", False)))
    attempt = attempts(tmp_path)["candidate-000"]
    assert "settled_cost_usd" not in attempt
    assert attempt["settlement_held"]["reason"] == "beta_screening_settlement_teardown_unproven"
    assert site_budget(tmp_path)["spent_upper_bound_usd"] == "4"
    before = ledger(tmp_path)
    assert coordinator._settle_finished(tmp_path) == 0
    assert ledger(tmp_path) == before


def test_freed_money_funds_the_waitlist_and_the_site_budget_counts_settled_spend(tmp_path):
    value = waitlisted_plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    webapp = WebApp()
    webapp.deliver(tmp_path)
    outcome = run(tmp_path, Measured(value))
    webapp.deliver(tmp_path)
    # Three $4 runs settled at $0.45 each, so the waitlisted fourth fits ($1.35 + $4 <= $15).
    assert outcome["waitlist_changes"] == 1
    assert [row["status"] for row in webapp.accepted["campaign-1"]["candidates"]] == ["completed"] * 3 + ["pending"]
    run(tmp_path, Measured(value))
    webapp.deliver(tmp_path)
    assert [row["status"] for row in webapp.accepted["campaign-1"]["candidates"]] == ["completed"] * 4
    assert webapp.accepted["campaign-1"]["budget"]["spent_upper_bound_usd"] == "1.8"
    assert site_budget(tmp_path)["committed_upper_bound_usd"] == "1.8"


@pytest.mark.parametrize("last_delivered", [True, False])
def test_retroactive_settlement_frees_the_site_and_every_publication_stays_acceptable(tmp_path, monkeypatch,
                                                                                     last_delivered):
    """capture-447ae4f4: three runs committed at $4 before settlement existed, $12 of $15 published."""
    first = waitlisted_plan()
    coordinator.enqueue_screening_campaign(first, queue_root=tmp_path)
    webapp = WebApp()
    webapp.deliver(tmp_path)
    with monkeypatch.context() as patched:
        patched.setattr(coordinator, "_settle", lambda *args, **kwargs: False)
        run(tmp_path, Measured(first))
    if last_delivered:
        webapp.deliver(tmp_path)
        published = webapp.accepted["campaign-1"]
        assert [row["status"] for row in published["candidates"]] == ["completed"] * 3 + ["deferred"]
        assert published["budget"]["spent_upper_bound_usd"] == "12"
    reseal(tmp_path, unsettle)

    # The first pass after the deploy settles each finished run once.
    outcome = run(tmp_path, Measured(first))
    assert outcome["settlements_decided"] == 3
    assert {attempt["settled_cost_usd"] for attempt in attempts(tmp_path).values()} == {"0.45"}
    assert site_budget(tmp_path)["committed_upper_bound_usd"] == "1.35"
    # A campaign never publishes a lower spent bound than it recorded ($12).
    if last_delivered:
        # The WebApp holds all of it already: it is not sent again for the settlement.
        assert coordinator.pending_publications(queue_root=tmp_path) == []
    else:
        webapp.deliver(tmp_path)
        assert webapp.accepted["campaign-1"]["budget"]["spent_upper_bound_usd"] == "12"
    before = ledger(tmp_path)
    assert run(tmp_path, Measured(first))["settlements_decided"] == 0
    assert ledger(tmp_path)["campaigns"] == before["campaigns"]

    # The job ceiling counts the settled $1.35, never more than the WebApp's stale $12.
    assert job_ceiling.local_screening_holds(tmp_path, site_id="site-job-1", published_usd=Decimal(0)) == Decimal("1.35")
    assert job_ceiling.local_screening_holds(tmp_path, site_id="site-job-1", published_usd=Decimal(12)) == 0

    # A later plan at the site is funded from the freed money: one $11 candidate, still running.
    second = plan(campaign_id="campaign-2", cost="11.00")
    coordinator.enqueue_screening_campaign(second, queue_root=tmp_path)
    webapp.deliver(tmp_path)
    assert webapp.accepted["campaign-2"]["budget"]["spent_upper_bound_usd"] == "1.35"
    run(tmp_path, Measured(second, running_first=True))
    assert site_budget(tmp_path)["reserved_usd"] == "11"
    webapp.deliver(tmp_path)
    assert webapp.accepted["campaign-2"]["budget"] == {
        "currency": "USD", "limit_usd": "15", "committed_upper_bound_usd": "12.35",
        "spent_upper_bound_usd": "1.35", "reserved_usd": "11"}
    # The newest publication never understates the real $12.35 commitment.
    assert webapp.latest["campaign_id"] == "campaign-2"
    # The first campaign, acknowledged in full, keeps its bytes: it is not sent again.
    assert webapp.accepted["campaign-1"]["budget"]["committed_upper_bound_usd"] == "12"
    run(tmp_path, Measured(second, running_first=True))
    webapp.deliver(tmp_path)
    assert webapp.accepted["campaign-2"]["status"] == "completed"
    # The WebApp's job ceiling now reads the settled site: $1.80, not $12 + $11.
    assert webapp.latest["campaign_id"] == "campaign-2"
    assert webapp.latest["budget"]["committed_upper_bound_usd"] == "1.8"
    assert webapp.accepted["campaign-1"]["budget"]["spent_upper_bound_usd"] == "12"
    assert site_budget(tmp_path)["committed_upper_bound_usd"] == "1.8"


def test_a_campaign_with_a_floor_publishes_the_real_total_above_it():
    state = {"site_id": "site-job-1", "site_budget": {"spent_upper_bound_usd": "12"}, "candidates": {}}

    def ledger_with(committed, reserved):
        attempt = {"maximum_cost_usd": committed, "reservation_status": "committed"}
        rows = {"a": {"status": "completed", "attempt": attempt}}
        if reserved:
            rows["b"] = {"status": "running", "attempt": {"maximum_cost_usd": reserved, "reservation_status": "reserved"}}
        return {"campaigns": {"campaign-1": {**state, "candidates": rows}}}

    assert coordinator._publication_budget(ledger_with("1.35", None), state) == {
        "currency": "USD", "limit_usd": "15", "committed_upper_bound_usd": "12", "spent_upper_bound_usd": "12",
        "reserved_usd": "0"}
    assert coordinator._publication_budget(ledger_with("1.35", "13"), state) == {
        "currency": "USD", "limit_usd": "15", "committed_upper_bound_usd": "14.35", "spent_upper_bound_usd": "12",
        "reserved_usd": "2.35"}
    # Spend past the floor publishes as it is.
    assert coordinator._publication_budget(ledger_with("13", None), state)["spent_upper_bound_usd"] == "13"


def test_a_spent_row_is_still_never_planned_again(tmp_path):
    value = plan()
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    run(tmp_path, Measured(value))
    row = ledger(tmp_path)["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert row["attempt"]["settled_cost_usd"] == "0.45" and coordinator._spent(row)
