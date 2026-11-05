"""A paid beta run's measured cells reach the site through the real reconcile, even when it stopped part way.

The provider side is the real worker sealing code (``_aggregate_isolated_cell_results``,
the beta prefix and interruption seals, ``seal_cell_setup_failure``); the host side
is the real ``reconcile_candidate`` (scoped closeout with a read-only provider stub,
the session join and validation), the coordinator, settlement and the publication.
A planned cell the run did not return is unscored, never invented; a run that
returned no cell keeps its blocked, harness-failure handling.

``BLUEPRINT_WRITE_SCREENING_PUBLICATION_FIXTURES=1`` rewrites the publication
fixtures the WebApp's producer-contract test reads; otherwise this checks the
retained fixtures still have the shape the real chain produces.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from jsonschema import Draft202012Validator
import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import gpu_render_providers
from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
from blueprint_pipeline import team_evaluation_worker_retention as retention
from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PROVIDER_BUNDLE_SCHEMA_VERSION, PROVIDER_RESULT_FILENAME, consume_session_authority_once,
    validate_session_authority,
)
from blueprint_pipeline.policy_canary_cell_setup_failure import (
    cell_setup_failure, seal_cell_machine_fault, seal_cell_setup_failure,
)
from tests.test_beta_policy_screening_coordinator import native_result, plan as make_plan, seal
from tests.test_native_task_arena_policy_canary_session import _authority, _record

ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "docs/schemas/beta_policy_screening_publication.v2.schema.json"
FIXTURES = {"full": ROOT / "tests/fixtures/beta_policy_screening_reconciled_publication.v2.json",
            "partial": ROOT / "tests/fixtures/beta_policy_screening_partial_publication.v2.json"}
TIMEOUT = "policy_canary_isolated_cell_process_timeout"
LINEAGE = "diagnostic"


class Provider:
    """Read-only provider observations: this attempt's instance is gone and nothing else is live."""

    def inspect(self, instance_id):
        return {"status": "absent", "provider": "vast", "instance_id": instance_id, "api_confirmed": True,
                "provider_absence_confirmed": True}

    def billable_inventory(self, *, name_prefix):
        return {"status": "observed", "provider": "vast", "name_prefix": name_prefix, "api_confirmed": True,
                "observed_at_epoch": time.time(), "resources": [], "live_resource_count": 0}


@pytest.fixture(autouse=True)
def _no_provider_io(monkeypatch):
    monkeypatch.setattr(gpu_render_providers, "VastRenderProvider", Provider)


def _write(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))
    return _record(path)


def _inputs(value, candidate):
    base = native_result(value, candidate)
    return {"run_kind": BETA_RUN_KIND, "run_id": "beta-" + candidate["candidate_id"],
            "candidate_ids": [candidate["candidate_id"]], "cells": candidate["scenario_cells"],
            "task_success_contract": base["task_success_contract"],
            "task_success_contract_digest": base["task_success_contract_digest"],
            "matrix_digest": value["scenario_plan_digest"], "beta_screening_admission": base["beta_screening_admission"],
            "scene_revision_digest": value["scene_revision_digest"], "plan_digest": value["plan_digest"],
            "configuration_digest": candidate["configuration_digest"], "runtime_inputs_digest": "sha256:" + "c" * 64}


def _flags(flagged):
    """What a cell's session records when its usable-frame gate did not pass integrity."""
    if not flagged:
        return {}
    from blueprint_pipeline.native_task_arena_policy_canary_session import observation_quality_flags
    from tests.test_beta_screening_worker_control import runtime_observation_gate
    gate = runtime_observation_gate(backend_digest="sha256:" + "b" * 64)
    return {"preload_observation_gate": gate, "observation_integrity_flagged": True,
            "observation_quality_flags": observation_quality_flags(gate, run_kind=BETA_RUN_KIND)}


def _children(value, candidate, inputs, successes, flagged=False):
    """One sealed child per cell, as each fresh-interpreter cell returns it."""
    children = []
    for index, row in enumerate(native_result(value, candidate)["episodes"]):
        score = {"status": "scored", "task_succeeded": index < successes}
        row["episode"] = {"score": score}
        row["deterministic_score_digest"] = canonical_digest({"value": score})
        children.append(seal({"schema_version": "native_task_arena_policy_canary_session_result.v1",
                              "selected_cell_index": index,
                              "status": "runtime_selected_cell_completed_pending_aggregation", "episodes": [row],
                              "task_success_contract": inputs["task_success_contract"],
                              "task_success_contract_digest": inputs["task_success_contract_digest"],
                              **_flags(flagged)}, "result_digest"))
    return children


def _gate_refused(inputs, index):
    """The retained pre-policy observation-gate refusal of one cell; the matrix continues."""
    return seal({"selected_cell_index": index, "status": "blocked", "episodes": [], "policy_loads": [],
                 "candidate_policy_queried": False, "preload_observation_gate": None,
                 "session_closeout": {"runtime_closed": True},
                 "telemetry": {"message_count": 0, "channel_message_counts": {"observations": 0, "episode_envelopes": 0}},
                 "session_failure_type": "PolicyObservationGateRefused",
                 "session_failure_source_trace": [{"file": "worker.py", "function": "prepolicy_observation_gate"}],
                 "task_success_contract": inputs["task_success_contract"],
                 "task_success_contract_digest": inputs["task_success_contract_digest"]}, "result_digest")


def _seal_prefix(**kwargs):
    return retention.seal_team_prefix(**kwargs, construction_lineage_mode=LINEAGE,
                                      aggregate=worker._aggregate_isolated_cell_results, seal_result=worker._seal_result)


def provider_result(value, candidate, output_root: Path, *, successes: int, stop: str = "none", at: int = 10,
                    flagged: bool = False) -> dict:
    """The sealed provider session result, written by the real worker sealing code."""
    inputs = _inputs(value, candidate)
    children = _children(value, candidate, inputs, successes, flagged)
    output_root.mkdir(parents=True, exist_ok=True)
    authority = {"run_id": inputs["run_id"], "authority_digest": "sha256:" + "d" * 64}
    if stop == "none":
        worker._seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME,
            result=worker._aggregate_isolated_cell_results(authority={}, inputs=inputs, child_results=children,
                                                           output_root=output_root, construction_lineage_mode=LINEAGE))
    elif stop == "gate":
        children[at] = _gate_refused(inputs, at)
        worker._seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME,
            result=worker._aggregate_isolated_cell_results(authority={}, inputs=inputs, child_results=children,
                                                           output_root=output_root, construction_lineage_mode=LINEAGE))
    elif stop == "timeout":
        child_root = output_root / "cell_runs" / f"{at:02d}"
        child_root.mkdir(parents=True, exist_ok=True)  # The timed-out cell sealed nothing.
        assert retention.seal_beta_interruption(inputs=inputs, authority=authority, children=children[:at],
            child_root=child_root, output_root=output_root, index=at, reason=TIMEOUT,
            seal_result=worker._seal_result, seal_prefix=_seal_prefix)
    elif stop == "setup":
        child_root = output_root / "cell_runs" / f"{at:02d}"
        failed = seal({"schema_version": "native_task_arena_policy_canary_session_result.v1", "selected_cell_index": at,
                       "status": "blocked", "episodes": [], "policy_loads": [], "candidate_policy_queried": False,
                       "session_failure_type": "RuntimeError", "preload_observation_gate": None,
                       "task_success_contract": inputs["task_success_contract"],
                       "task_success_contract_digest": inputs["task_success_contract_digest"]}, "result_digest")
        _write(child_root / PROVIDER_RESULT_FILENAME, failed)
        failure = cell_setup_failure(child=failed, inputs=inputs, index=at, child_root=child_root,
                                     output_root=output_root)
        assert failure is not None
        seal_cell_setup_failure(output_root=output_root, inputs=inputs, authority=authority,
                                children=[*children[:at], failed], failure=failure,
                                construction_lineage_mode=LINEAGE, aggregate=worker._aggregate_isolated_cell_results,
                                seal_result=worker._seal_result)
    elif stop == "machine":
        # The cell's simulator died by a fatal signal while Isaac was starting and wrote nothing.
        child_root = output_root / "cell_runs" / f"{at:02d}"
        child_root.mkdir(parents=True, exist_ok=True)
        (child_root / "cell_progress.log").write_text("".join(
            f"BLUEPRINT_POLICY_CANARY_PROGRESS:cell={at}:stage={stage}\n"
            for stage in ("static_preflight_passed", "isaac_launch_started")))
        (child_root / "worker_console.log").write_text("[Fatal] terminating this process with exit code 139.\n")
        assert seal_cell_machine_fault(exit_code=139, inputs=inputs, index=at, child_root=child_root,
            output_root=output_root, children=children[:at], authority=authority, construction_lineage_mode=LINEAGE,
            aggregate=worker._aggregate_isolated_cell_results, seal_result=worker._seal_result)
    else:
        raise AssertionError(stop)
    return json.loads((output_root / PROVIDER_RESULT_FILENAME).read_text())


def build_attempt(value, candidate, attempt_dir: Path, **run) -> dict:
    """The attempt's sealed authority, bundle, prepared record, provider result, closeout and budget ledger."""
    fixture = attempt_dir / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    inner = provider_result(value, candidate, attempt_dir / "allocator" / "runtime_output", **run)
    authority, _ = _authority(fixture)
    authority.update(run_kind=BETA_RUN_KIND, candidate_ids=[candidate["candidate_id"]], learned_policy_rollout_count=10,
                     beta_screening_admission=inner["beta_screening_admission"], plan_digest=value["plan_digest"],
                     configuration_digest=candidate["configuration_digest"],
                     scene_revision_digest=value["scene_revision_digest"])
    validate_session_authority(seal(authority, "authority_digest"))
    archive = fixture / "bundle.zip"
    archive.write_bytes(b"bundle-" + candidate["candidate_id"].encode())
    archive_record = _record(archive)
    bundle = {"schema_version": PROVIDER_BUNDLE_SCHEMA_VERSION, "status": "ready",
              "execution_mode": "blueprint_beta_screening_session", "run_kind": BETA_RUN_KIND,
              "claim_ceiling": "diagnostic_policy_execution", "candidate_ids": [candidate["candidate_id"]],
              "episodes_per_policy": 10, "learned_policy_rollout_count": 10, "maximum_provider_allocations": 1,
              "retry_cap": 0, "candidate_policy_queried": False, "expected_output_filename": PROVIDER_RESULT_FILENAME,
              **{key: authority[key] for key in ("runtime_inputs_digest", "authority_digest", "task_success_contract_digest")},
              "bundle_path": archive_record["path"], "bundle_size_bytes": archive_record["size_bytes"],
              "bundle_sha256": archive_record["sha256"]}
    prepared = seal({"schema_version": "beta_policy_screening_prepared.v1", "plan_digest": value["plan_digest"],
                     "candidate_id": candidate["candidate_id"], "configuration_digest": candidate["configuration_digest"],
                     "run_id": authority["run_id"], "attempt_dir": str(attempt_dir.resolve()), "argv": [],
                     "authority": _write(fixture / "authority.json", authority),
                     "bundle_receipt": _write(fixture / "bundle_receipt.json", bundle), "prepared_digest": ""},
                    "prepared_digest")
    (attempt_dir / "prepared.json").write_text(json.dumps(prepared, indent=2, sort_keys=True))
    run_dir = attempt_dir / "allocator" / "attempts" / "attempt_001" / "vast_provider_run"
    instance = 7001 + value["candidates"].index(candidate)
    teardown = _write(run_dir / "vast_teardown_manifest.json", {
        "schema_version": "vast_teardown_manifest.v1", "status": "completed", "vast_instance_ids": [instance],
        "runner_gpu_teardown_completed": True, "continuing_spend_from_this_run": False, "retention_authorized": False,
        "teardown_actions_performed": [{"action": "destroy_instance", "status": "completed", "instance_id": instance}]})
    raw_adapter = _write(run_dir / "vast_provider_adapter_result.json",
                         {"vast_instance_ids": [instance], "continuing_spend_from_this_run": False})
    _write(run_dir / "vast_budget_ledger.json", {
        "schema_version": "vast_budget_ledger.v1", "status": "completed", "continuing_spend_from_this_run": False,
        "vast_instance_ids": [instance], "provider_transfer_cost_priced": True, "estimated_spend_under_hard_cap": True,
        "actual_cost_usd": None, "actual_cost_source": "not_available_from_instance_probe_api",
        "estimated_runtime_cost_usd": 0.41, "projected_provider_transfer_cost_usd": 0.03, "estimated_cost_usd": 0.44,
        "hard_cap_usd": 4.0})
    adapter = {"status": "completed", "vast_instance_ids": [instance], "bundle_sha256": bundle["bundle_sha256"],
               "authorization_consumption": consume_session_authority_once(authority, consumption_path=fixture / "consume.json"),
               "continuing_spend_from_this_run": False, "all_staged_objects_absent": True, "estimated_cost_usd": 0.44,
               "native_control_result_path": str(attempt_dir / "allocator" / "runtime_output" / PROVIDER_RESULT_FILENAME),
               "native_control_result_digest": inner["result_digest"],
               "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
                                     "all_staged_objects_absent": True, "estimated_cost_usd": 0.44,
                                     "adapter_result": raw_adapter, "teardown_manifest": teardown},
               "provider_mutations_performed": 2, "provider_create_attempted": True}
    (attempt_dir / "allocator_result.json").write_text(json.dumps(adapter, indent=2, sort_keys=True))
    return prepared


class Executor:
    """Staging and the provider run are stand-ins; reconciliation is the real ``launch.reconcile_candidate``."""

    def __init__(self, runs):
        self.runs = runs
        self.observations = {}

    def prepare_candidate(self, *, plan, candidate, attempt_dir):
        return build_attempt(plan, candidate, Path(attempt_dir), **self.runs[candidate["candidate_id"]])

    def launch_candidate(self, **_):
        return {"status": "running", "provider_allocation_performed": True, "handle": {"pid": 1}}

    def reconcile_candidate(self, **kwargs):
        observed = launch.reconcile_candidate(**kwargs)
        self.observations[kwargs["candidate"]["candidate_id"]] = observed
        return observed


def run_campaign(root: Path, value, runs):
    """Drive the queue as the worker does, delivering and acknowledging every revision."""
    coordinator.enqueue_screening_campaign(value, queue_root=root)
    executor, delivered = Executor(runs), []
    for _ in range(2 * len(value["candidates"]) + 3):
        for refresh in (True, False):
            for publication in coordinator.pending_publications(queue_root=root, refresh_startable=refresh):
                delivered.append(publication)
                coordinator.acknowledge_publication(publication["campaign_id"], publication["publication_digest"],
                                                    queue_root=root)
            if refresh:
                coordinator.process_screening_queue(queue_root=root, executor=executor, max_launches=1)
    ledger = json.loads((root / coordinator.LEDGER_FILENAME).read_text())
    return delivered, ledger["campaigns"][value["campaign_id"]]["candidates"], executor


def _webapp_contract(publication):
    """The WebApp's count and budget rules (``betaPolicyScreeningContract.ts``), restated."""
    Draft202012Validator(json.loads(SCHEMA.read_text())).validate(publication)
    assert publication["publication_digest"] == cross_runtime_canonical_digest(publication, digest_field="publication_digest")
    for row in publication["candidates"]:
        assert row["episodes_succeeded"] + row["episodes_failed"] == row["episodes_scored"]
        assert row["success_rate"] == (row["episodes_succeeded"] / row["episodes_scored"] if row["episodes_scored"] else None)
        if row["status"] == "completed":
            assert row["session_result_digest"] and row["episodes_scored"] + row["episodes_void"] == row["episodes_planned"]
        else:
            assert row["episodes_scored"] == row["episodes_void"] == 0 and row["session_result_digest"] is None


def _shape(value):
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else type(value).__name__


def _retained_fixture(name, publication):
    """Rewrite the WebApp producer fixture on request; otherwise it must still match the real chain."""
    path = FIXTURES[name]
    if os.environ.get("BLUEPRINT_WRITE_SCREENING_PUBLICATION_FIXTURES") == "1":
        path.write_text(json.dumps(publication, indent=2, sort_keys=True) + "\n")
    retained = json.loads(path.read_text())
    _webapp_contract(retained)
    assert _shape(retained) == _shape(publication)
    facts = ("status", "blockers", "episodes_planned", "episodes_scored", "episodes_succeeded", "episodes_failed",
             "episodes_void", "success_rate")
    assert [{key: row[key] for key in facts} for row in retained["candidates"]] == [
        {key: row[key] for key in facts} for row in publication["candidates"]]
    assert (retained["status"], retained["budget"]) == (publication["status"], publication["budget"])


def test_full_ten_cell_runs_reconcile_to_completed_estimates(tmp_path):
    value = make_plan(count=2)
    runs = {value["candidates"][0]["candidate_id"]: {"successes": 7},
            value["candidates"][1]["candidate_id"]: {"successes": 10}}
    delivered, rows, executor = run_campaign(tmp_path / "queue", value, runs)
    assert all(observed["status"] == "completed" for observed in executor.observations.values())
    final = delivered[-1]
    assert final["status"] == "completed"
    assert [(row["status"], row["episodes_scored"], row["episodes_succeeded"], row["episodes_void"])
            for row in final["candidates"]] == [("completed", 10, 7, 0), ("completed", 10, 10, 0)]
    # Settled at each run's own ledger estimate, not the $4 reservation.
    assert final["budget"] == {"currency": "USD", "limit_usd": "30", "committed_upper_bound_usd": "0.88",
                               "spent_upper_bound_usd": "0.88", "reserved_usd": "0"}
    assert all("episodes_not_returned" not in row["summary"] for row in rows.values())
    for publication in delivered:
        _webapp_contract(publication)
    _retained_fixture("full", final)


@pytest.mark.parametrize("stop, at, reason", [
    ("timeout", 6, f"cell_not_returned:{TIMEOUT}"),
    ("setup", 5, "cell_not_returned:policy_canary_session_setup_failed:runtime_open"),
    ("gate", 9, "cell_not_returned:prepolicy_observation_gate_refused"),
])
def test_run_stopped_part_way_counts_its_measured_cells_and_voids_the_rest(tmp_path, stop, at, reason):
    value = make_plan(count=1)
    candidate = value["candidates"][0]
    delivered, rows, executor = run_campaign(tmp_path / "queue", value,
                                             {candidate["candidate_id"]: {"successes": 4, "stop": stop, "at": at}})
    observed = executor.observations[candidate["candidate_id"]]
    assert observed["status"] == "completed" and len(observed["result"]["episodes"]) == (9 if stop == "gate" else at)
    row = rows[candidate["candidate_id"]]
    returned = 9 if stop == "gate" else at
    missing = [cell["cell_id"] for cell in candidate["scenario_cells"]][at:] if stop != "gate" else [
        candidate["scenario_cells"][at]["cell_id"]]
    assert row["status"] == "completed" and row["blockers"] == []
    summary = row["summary"]
    assert summary["episodes_not_returned"] == len(missing) == 10 - returned
    assert (summary["episodes_scored"], summary["episodes_succeeded"], summary["episodes_void"]) == (
        returned, 4, 10 - returned)
    assert summary["success_rate"] == 4 / returned  # The estimate rests on scored episodes only.
    assert [item for item in summary["episode_outcomes"] if item["outcome"] == "void"] == [
        {"cell_id": cell_id, "outcome": "void", "reason": reason} for cell_id in missing]
    assert [item["cell_id"] for item in summary["episode_outcomes"]] == [
        cell["cell_id"] for cell in candidate["scenario_cells"]]
    # Paid, measured and completed: never a harness failure planned again after a deploy.
    assert not coordinator._harness_failed(row)
    assert row["attempt"]["settled_cost_usd"] == "0.44"
    final = delivered[-1]
    assert final["status"] == "completed"
    for publication in delivered:
        _webapp_contract(publication)
    if stop == "timeout":
        _retained_fixture("partial", final)


@pytest.mark.parametrize("stop, at", [("none", 10), ("timeout", 6)])
def test_flagged_observation_gate_reaches_the_ledger_summary_and_stays_out_of_the_publication(tmp_path, stop, at):
    from tests.test_beta_screening_worker_control import REAL_GATE_BLOCKERS
    value = make_plan(count=1)
    candidate = value["candidates"][0]
    delivered, rows, executor = run_campaign(tmp_path / "queue", value, {
        candidate["candidate_id"]: {"successes": 4, "stop": stop, "at": at, "flagged": True}})
    observed = executor.observations[candidate["candidate_id"]]
    flags = sorted(REAL_GATE_BLOCKERS)
    # Reconciled through the real scoped closeout join and session validation.
    assert observed["status"] == "completed" and observed["result"]["observation_quality_flags"] == flags
    row = rows[candidate["candidate_id"]]
    assert row["status"] == "completed" and row["blockers"] == []
    summary = row["summary"]
    assert summary["observation_quality_flags"] == flags
    assert (summary["episodes_scored"], summary["episodes_succeeded"]) == (at, 4)  # The estimate is unchanged.
    # The v2 publication has no field for the flags: it carries the estimate as before, schema-valid.
    for publication in delivered:
        _webapp_contract(publication)
        assert "observation_quality_flags" not in json.dumps(publication)
    plain, plain_rows, _ = run_campaign(tmp_path / "plain", value, {
        candidate["candidate_id"]: {"successes": 4, "stop": stop, "at": at}})
    unflagged = plain[-1]
    assert "observation_quality_flags" not in plain_rows[candidate["candidate_id"]]["summary"]
    facts = ("status", "episodes_scored", "episodes_succeeded", "episodes_failed", "episodes_void", "success_rate")
    assert [{key: item[key] for key in facts} for item in delivered[-1]["candidates"]] == [
        {key: item[key] for key in facts} for item in unflagged["candidates"]]


@pytest.mark.parametrize("stop", ["timeout", "setup"])
def test_run_that_returned_no_cell_keeps_its_harness_failure(tmp_path, stop):
    value = make_plan(count=1)
    candidate = value["candidates"][0]
    if stop == "timeout":
        # Nothing measured: no beta prefix is sealed; the existing interrupted record stands.
        inputs = _inputs(value, candidate)
        child_root = tmp_path / "out" / "cell_runs" / "00"
        child_root.mkdir(parents=True)
        assert not retention.seal_beta_interruption(inputs=inputs, authority={}, children=[], child_root=child_root,
            output_root=tmp_path / "out", index=0, reason=TIMEOUT, seal_result=worker._seal_result,
            seal_prefix=_seal_prefix)
        assert not (tmp_path / "out" / PROVIDER_RESULT_FILENAME).exists()
        return
    _, rows, executor = run_campaign(tmp_path / "queue", value,
                                     {candidate["candidate_id"]: {"successes": 0, "stop": "setup", "at": 0}})
    assert executor.observations[candidate["candidate_id"]]["blockers"] == [
        "policy_canary_session_setup_failed:runtime_open"]
    row = rows[candidate["candidate_id"]]
    assert row["status"] == "blocked" and "summary" not in row
    assert coordinator._harness_failed(row)


def test_a_prefix_of_only_gate_refusals_seals_nothing(tmp_path):
    value = make_plan(count=1)
    candidate = value["candidates"][0]
    inputs = _inputs(value, candidate)
    assert not _seal_prefix(output_root=tmp_path, inputs=inputs, authority={}, children=[_gate_refused(inputs, 0)],
                            reason=TIMEOUT, interrupted_cell_index=1)
    assert not (tmp_path / PROVIDER_RESULT_FILENAME).exists()


@pytest.mark.parametrize("change", ["unplanned_subset", "duplicate_cell", "foreign_cell", "empty"])
def test_partial_summary_requires_exact_planned_cells(tmp_path, change):
    from blueprint_pipeline.native_task_arena_policy_canary_session import PolicyCanarySessionError
    value = make_plan(count=1)
    candidate = value["candidates"][0]
    inner = provider_result(value, candidate, tmp_path / "out", successes=4, stop="timeout", at=6)
    joined = dict(inner, provider_allocations_observed=1, status="completed_unqualified",
                  session_closeout={"provider_allocations_observed": 1, "teardown_completed": True,
                                    "provider_zero_confirmed": True})
    summary = candidate_result_summary(seal(dict(joined), "result_digest"), plan=value, candidate=candidate)
    assert (summary["episodes_scored"], summary["episodes_void"]) == (6, 4)
    if change == "unplanned_subset":
        joined.pop("planned_episodes_per_policy")
    elif change == "duplicate_cell":
        joined["episodes"] = [*joined["episodes"][:5], {**joined["episodes"][0]}]
    elif change == "foreign_cell":
        joined["episodes"][0] = {**joined["episodes"][0], "seed": joined["episodes"][0]["seed"] + 1000}
    else:
        joined.update(episodes=[], episodes_per_policy=0, learned_policy_rollout_count=0)
    with pytest.raises((ValueError, PolicyCanarySessionError)):
        candidate_result_summary(seal(joined, "result_digest"), plan=value, candidate=candidate)


@pytest.mark.parametrize("failure, reason", [
    ("timeout", TIMEOUT),
    ("timeout_after_sealing", TIMEOUT),
    ("process_failed", "policy_canary_isolated_cell_process_failed"),
])
def test_actual_parent_seals_the_measured_prefix_of_a_beta_run_that_stopped(tmp_path, monkeypatch, failure, reason):
    import subprocess
    from tests.test_beta_screening_worker_control import _completed_parent_fixture_child, stage_beta_bundle

    monkeypatch.delenv(worker._HARD_DEADLINE_ENV, raising=False)
    runtime, output, inputs, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    stop_at, spawned = 3, []

    def spawn(**kwargs):
        index = kwargs["index"]
        spawned.append(index)
        child = _completed_parent_fixture_child(inputs, index)
        if index == stop_at:
            if failure == "process_failed":
                raise OSError("cell process could not start")
            if failure == "timeout_after_sealing":  # Isaac hung while closing after it sealed its cell.
                (kwargs["child_root"] / PROVIDER_RESULT_FILENAME).write_text(json.dumps(child))
            raise subprocess.TimeoutExpired("cell", timeout=5)
        (kwargs["child_root"] / PROVIDER_RESULT_FILENAME).write_text(json.dumps(child))
        return 0

    if failure == "process_failed":
        with pytest.raises(OSError):
            worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn)
    else:
        assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn) == 1
    assert spawned == list(range(stop_at + 1))
    retained = json.loads((output / PROVIDER_RESULT_FILENAME).read_text())
    measured = stop_at + (failure == "timeout_after_sealing")
    assert retained["run_kind"] == BETA_RUN_KIND and retained["status"] == "blocked"
    assert retained["candidate_ids"] == inputs["candidate_ids"]
    assert retained["beta_screening_admission"] == inputs["beta_screening_admission"]
    assert retained["plan_digest"] == inputs["plan_digest"]
    assert retained["episodes_per_policy"] == len(retained["episodes"]) == retained["learned_policy_rollout_count"] == measured
    assert [row["cell_id"] for row in retained["episodes"]] == [cell["cell_id"] for cell in inputs["cells"][:measured]]
    assert retained["planned_episodes_per_policy"] == len(inputs["cells"])
    assert retained["blockers"] == [reason] and retained["interrupted_cell_index"] == stop_at
    assert retained["result_digest"] == canonical_digest(retained, digest_field="result_digest")


def test_actual_parent_first_cell_timeout_seals_no_beta_prefix(tmp_path, monkeypatch):
    import subprocess
    from tests.test_beta_screening_worker_control import stage_beta_bundle

    monkeypatch.delenv(worker._HARD_DEADLINE_ENV, raising=False)
    runtime, output, *_ = stage_beta_bundle(tmp_path, monkeypatch)

    def spawn(**kwargs):
        raise subprocess.TimeoutExpired("cell", timeout=5)

    try:
        worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output, run_cell_process=spawn)
    except Exception:  # noqa: BLE001 - today's handling of a run with nothing measured is unchanged
        pass
    path = output / PROVIDER_RESULT_FILENAME
    assert not path.exists() or not json.loads(path.read_text()).get("episodes")


def test_a_void_episode_names_its_typed_failure_in_the_ledger_summary(tmp_path):
    """2026-11-05: ten void episodes, and the archive that said why was gone at settlement.
    The summary keeps each stopped episode's exception class and leading typed code, never
    its message."""
    value = make_plan(count=1)
    candidate = value["candidates"][0]
    inner = provider_result(value, candidate, tmp_path / "out", successes=4)
    joined = dict(inner, provider_allocations_observed=1, status="completed_unqualified",
                  session_closeout={"provider_allocations_observed": 1, "teardown_completed": True,
                                    "provider_zero_confirmed": True})
    stopped = {key: item for key, item in joined["episodes"][0].items()
               if key not in ("episode", "deterministic_score_digest")}
    # The shape the worker's failure record gives an episode that stopped before its first observation.
    stopped.update(status="blocked", failure_type="PolicyEpisodeError", typed_harness_failure="PolicyEpisodeError",
                   failure_message="policy_episode_prestart_readiness_failed:reset_joint_state_out_of_bounds;"
                                   "second error at <path>",
                   episode_failure_stage="before_first_observation", candidate_policy_queried=False,
                   actions_reached_robot=False, policy_outcome_interpretable=False,
                   visual_evidence={"status": "unavailable_before_first_observation", "media_gap": {
                       "type": "before_first_observation", "reason": "policy_canary_episode_runner_failed"}})
    # A session with any stopped episode closes as blocked.
    joined.update(episodes=[stopped, *joined["episodes"][1:]], status="blocked")
    summary = candidate_result_summary(seal(dict(joined), "result_digest"), plan=value, candidate=candidate)
    first, second = summary["episode_outcomes"][:2]
    assert first == {"cell_id": stopped["cell_id"], "outcome": "void", "reason": "simulator_failed_before_policy",
                     "failure_type": "PolicyEpisodeError",
                     "failure_code": "policy_episode_prestart_readiness_failed:reset_joint_state_out_of_bounds"}
    assert set(second) == {"cell_id", "outcome", "reason"}  # a scored episode carries no failure


def test_a_machine_that_crashed_starting_the_simulator_spends_no_harness_retry_and_runs_again(tmp_path):
    """2026-11-05: a pi05 cell segfaulted in the GPU driver 2 s into Isaac start-up (owner: never a harness retry)."""
    value = make_plan(count=2)
    crashed, other = value["candidates"]
    root = tmp_path / "queue"
    _, rows, executor = run_campaign(root, value, {crashed["candidate_id"]: {"successes": 0, "stop": "machine", "at": 0},
                                                   other["candidate_id"]: {"successes": 0, "stop": "machine", "at": 0}})
    for candidate in (crashed, other):
        assert executor.observations[candidate["candidate_id"]]["blockers"] == [
            "policy_canary_session_machine_fault:simulator_start_crash"]
        row = rows[candidate["candidate_id"]]
        # Its money is settled like any closed run; it is not ours and nothing else is held for it.
        assert row["status"] == "blocked" and row["attempt"]["reservation_status"] == "committed"
        assert coordinator._machine_faulted(row) and not coordinator._harness_failed(row)
    ledger = json.loads((root / coordinator.LEDGER_FILENAME).read_text())
    state = ledger["campaigns"][value["campaign_id"]]
    assert coordinator._harness_failures(ledger, state["site_id"], crashed["candidate_id"]) == 0
    # Planned again at the same commit: nothing to deploy, the next offer search finds another machine.
    frozen = crashed["execution_spec"]["implementation_commit"]
    assert sorted(coordinator._replan_scope(ledger, value, frozen)) == sorted(
        candidate["candidate_id"] for candidate in value["candidates"])


def test_machine_faults_are_planned_again_a_bounded_number_of_times():
    row = {"status": "blocked", "blockers": ["policy_canary_session_machine_fault:simulator_start_crash"]}
    ledger = {"campaigns": {}}
    for count in range(1, coordinator.MACHINE_FAULT_RETRY_LIMIT + 2):
        ledger["campaigns"][f"campaign-{count}"] = {"site_id": "site-a", "candidates": {"c": dict(row)}}
        assert coordinator._machine_fault_retry_allowed(ledger, "site-a", "c") is (
            count <= coordinator.MACHINE_FAULT_RETRY_LIMIT)
    # Another blocker beside it is not a machine fault alone.
    assert not coordinator._machine_faulted({**row, "blockers": [*row["blockers"], "beta_screening_other"]})
