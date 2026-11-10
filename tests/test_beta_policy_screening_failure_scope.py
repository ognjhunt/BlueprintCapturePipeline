"""A failed screening run is run again, holds its siblings, or neither, by what its typed codes predict."""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_failure_scope as scope
from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint
from blueprint_pipeline.native_task_arena_policy_canary_session import BETA_RUN_KIND, PROVIDER_RESULT_FILENAME
from tests.test_beta_policy_screening_coordinator import Backend, plan, publication
from tests.test_beta_policy_screening_replan import HARNESS_FAILED, NEW, OLD, successor

PACKAGE_MIRROR = f"{scope.TRANSIENT_PROVISIONING_PREFIX}:package_mirror"
TRANSIENT = {**HARNESS_FAILED, "blockers": [PACKAGE_MIRROR]}
SHARED_STAGE = "policy_canary_session_setup_failed:runtime_open"
POLICY_LOAD = "policy_canary_session_setup_failed:policy_load"
# The allocator result's codes when a bundle started and nothing came back.
NO_OUTPUT = ["provider_bundle_completion_marker_missing", "provider_output_upload_marker_missing",
             "provider_runtime_output_zip_not_received_locally",
             "native_task_arena_policy_canary_session_provider_output_zip_missing"]


def _pass(root, observation):
    backend = Backend(observation)
    coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend)
    return backend


def _rows(root, campaign_id="campaign-1"):
    return [(row["status"], row["blockers"]) for row in publication(root, campaign_id)["candidates"]]


def _ledger(root):
    return coordinator._ledger(coordinator._root(root))


# --- what the run's own codes stand for --------------------------------------------------------


@pytest.mark.parametrize("blockers,break_reason,expected", [
    # The entrypoint could not install ffmpeg from the distribution's archive (2026-11-09, archive.ubuntu.com).
    (["policy_canary_media_toolchain_unavailable"], None, "package_mirror"),
    (["g1_beta_media_toolchain_unavailable", "native_g1_beta_runtime_not_completed"], None, "package_mirror"),
    # The machine could not fetch our bundle.
    (["provider_remote_blocker:download_failed:7", "provider_bundle_download_marker_missing"], None,
     "bundle_download"),
    # The adapter's own judgement of the machine (it also put it on the avoid list).
    (["provider_instance_exited_before_bundle_terminal_marker", *NO_OUTPUT], None, "rented_machine"),
    (["vast_heartbeat_no_log_progress_timeout", "vast_heartbeat_output_missing_success_marker"], None,
     "rented_machine"),
    (["provider_remote_blocker:asset_sandbox_unavailable"], None, "rented_machine"),
    # The bundle started and the provider log went dark (2026-11-09, DP's sixth run).
    (NO_OUTPUT, "no_log_progress_timeout", "provider_log_silent"),
    (NO_OUTPUT, "log_transport_unavailable", "provider_log_silent"),
    # A run that used its whole time, or whose wait ended for a reason nobody recorded, is not transient.
    (NO_OUTPUT, "max_wait_deadline_reached", None),
    (NO_OUTPUT, None, None),
    # Our entrypoint ran to an end and said so: ours, whatever the log did.
    (["policy_canary_entrypoint_failed_without_result"], "no_log_progress_timeout", None),
    (["policy_canary_worker_failed_without_result"], None, None),
    # The candidate's own policy provisioning: its stage is known, its cause is not, so never transient.
    ([scope.POLICY_PROVISIONING_STAGE_CODE], None, None),
    ([], None, None),
])
def test_only_typed_machine_network_and_mirror_codes_are_transient(blockers, break_reason, expected):
    code = scope.transient_provisioning_blocker(blockers, log_break_reason=break_reason)
    assert code == (f"{scope.TRANSIENT_PROVISIONING_PREFIX}:{expected}" if expected else None)
    assert code is None or code in scope.TRANSIENT_PROVISIONING_BLOCKERS


def test_the_log_break_reason_is_read_only_from_the_runs_own_record_inside_the_attempt(tmp_path):
    attempt = tmp_path / "attempt"
    run = attempt / "allocator" / "attempts" / "attempt_001" / "vast_provider_run"
    run.mkdir(parents=True)
    adapter = {"provider_closeout": {"adapter_result": {"path": str(run / "vast_provider_adapter_result.json")}}}
    assert scope.provider_log_break_reason(adapter, attempt_dir=attempt) is None  # no record: unknown
    (run / scope.STARTUP_PROBE_MANIFEST).write_text(json.dumps(
        {"container_log_result": {"break_reason": "no_log_progress_timeout"}}))
    assert scope.provider_log_break_reason(adapter, attempt_dir=attempt) == "no_log_progress_timeout"
    # A record outside the attempt, an unreadable one, or no closeout record at all is unknown.
    assert scope.provider_log_break_reason(adapter, attempt_dir=tmp_path / "another") is None
    (run / scope.STARTUP_PROBE_MANIFEST).write_text("{not json")
    assert scope.provider_log_break_reason(adapter, attempt_dir=attempt) is None
    assert scope.provider_log_break_reason({}, attempt_dir=attempt) is None


# --- transient infrastructure: the same commit, no deploy, no sibling held ---------------------


def test_a_transient_provisioning_failure_holds_no_sibling_and_spends_no_harness_retry(tmp_path):
    """2026-11-05: cosmos3's machine could not finish a download before any policy ran. Counted as a
    harness failure, it held four other candidates unspent until the next deploy."""
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    _pass(tmp_path, TRANSIENT)
    sibling = _pass(tmp_path, {"status": "running"})
    assert sibling.launches == 1
    assert _rows(tmp_path) == [("blocked", [PACKAGE_MIRROR]), ("running", [])]
    ledger = _ledger(tmp_path)
    row = ledger["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert coordinator._machine_faulted(row) and not coordinator._harness_failed(row)
    assert coordinator._harness_failures(ledger, "site-job-1", "candidate-000") == 0
    # It spent money like any closed run: committed, inside its own cap and the site limit.
    assert row["attempt"]["reservation_status"] == "committed"
    assert publication(tmp_path)["budget"]["committed_upper_bound_usd"] == "8"


def test_a_transient_provisioning_failure_is_planned_again_at_the_same_commit_with_no_deploy(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    _pass(tmp_path, TRANSIENT)
    _pass(tmp_path, TRANSIENT)
    assert _rows(tmp_path) == [("blocked", [PACKAGE_MIRROR])] * 2
    # No deploy: the plan's own commit.
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == [
        "candidate-000", "candidate-001"]
    coordinator.supersede_campaign("campaign-1", successor(plan(count=2), commit=OLD), implementation_commit=OLD,
                                   queue_root=tmp_path)
    retried = publication(tmp_path, "campaign-1b")
    assert [row["status"] for row in retried["candidates"]] == ["pending", "pending"]
    # The retry shares the site's allowance with what the failed runs committed.
    assert retried["budget"]["spent_upper_bound_usd"] == "8"


def test_transient_failures_share_the_machine_fault_bound_on_one_counter():
    transient = {"status": "blocked", "blockers": [PACKAGE_MIRROR]}
    machine = {"status": "blocked", "blockers": ["policy_canary_session_machine_fault:simulator_start_crash"]}
    ledger = {"campaigns": {}}
    for count in range(1, coordinator.MACHINE_FAULT_RETRY_LIMIT + 2):
        row = transient if count % 2 else machine
        ledger["campaigns"][f"campaign-{count}"] = {"site_id": "site-a", "candidates": {"c": dict(row)}}
        assert coordinator._machine_fault_retry_allowed(ledger, "site-a", "c") is (
            count <= coordinator.MACHINE_FAULT_RETRY_LIMIT)
    # Another code beside it is not a transient failure alone.
    assert not coordinator._machine_faulted({**transient, "blockers": [PACKAGE_MIRROR, "beta_screening_other"]})


def _void_summary(reason, count=2):
    return {"episodes_scored": 0, "episodes_succeeded": 0, "episodes_failed": 0, "episodes_void": count,
            "success_rate": None, "session_result_digest": "sha256:" + "0" * 64,
            "episode_outcomes": [{"cell_id": f"cell-{index}", "outcome": "void", "reason": reason}
                                 for index in range(count)]}


def test_a_g1_run_void_only_for_its_machines_package_mirror_is_the_machines_not_ours(tmp_path, monkeypatch):
    """2026-11-09, the library job's DP run: every cell void with ``g1_beta_media_toolchain_unavailable``.
    It was DP's fifth harness failure and held pi0.5 unspent."""
    monkeypatch.setattr(coordinator, "candidate_result_summary",
                        lambda *_args, **_kwargs: _void_summary("g1_beta_media_toolchain_unavailable"))
    closed = {"status": "completed", "result": {"session": "closed"},
              "provider_allocation_performed": True, "provider_zero_verified": True}
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    _pass(tmp_path, closed)
    assert _pass(tmp_path, {"status": "running"}).launches == 1  # the sibling is not held
    ledger = _ledger(tmp_path)
    row = ledger["campaigns"]["campaign-1"]["candidates"]["candidate-000"]
    assert coordinator._machine_faulted(row) and not coordinator._harness_failed(row)
    assert coordinator._harness_failures(ledger, "site-job-1", "candidate-000") == 0
    # A run that measured nothing proves no harness, and one void for any other reason is still ours.
    assert not coordinator._machine_faulted({**row, "summary": _void_summary("simulator_failed_before_policy")})
    assert coordinator._harness_failed({**row, "summary": _void_summary("simulator_failed_before_policy")})
    mixed = _void_summary("g1_beta_media_toolchain_unavailable")
    mixed["episode_outcomes"][0]["reason"] = "g1_beta_worker_failed_without_result"
    assert coordinator._harness_failed({**row, "summary": mixed})


# --- the sibling hold reaches as far as the failure predicts ------------------------------------


@pytest.mark.parametrize("code", sorted(scope.CANDIDATE_SPECIFIC_BLOCKERS))
def test_a_candidates_own_policy_failure_holds_only_that_candidate(tmp_path, code):
    assert code in coordinator.HARNESS_FAILURE_BLOCKERS
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    _pass(tmp_path, {**HARNESS_FAILED, "blockers": [code]})
    assert _pass(tmp_path, {"status": "running"}).launches == 1
    assert _rows(tmp_path) == [("blocked", [code]), ("running", [])]
    # Still that candidate's own harness failure: it waits for a deploy, inside its retry bound.
    ledger = _ledger(tmp_path)
    assert coordinator._harness_failures(ledger, "site-job-1", "candidate-000") == 1
    assert coordinator._harness_failures(ledger, "site-job-1", "candidate-001") == 0


def _on_backends(value, backends):
    """The same plan with each candidate on the named runtime backend (a G1 row runs its own cells)."""
    from blueprint_pipeline.beta_policy_screening_g1 import BACKEND as G1_BACKEND, frozen_cells

    for row, backend in zip(value["candidates"], backends, strict=True):
        row["execution_spec"]["backend"] = backend
        row["admission_digest"] = canonical_digest(row["execution_spec"])
        if backend == G1_BACKEND:
            row["scenario_cells"] = frozen_cells(value["scenario_cells"])
            row["episode_count"] = row["cost_quote"]["episode_count"] = len(row["scenario_cells"])
            row["cost_quote"]["quote_digest"] = canonical_digest(row["cost_quote"], digest_field="quote_digest")
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return validate_screening_plan(value)


def test_a_shared_stage_failure_holds_the_pending_siblings_on_its_backend_only(tmp_path):
    value = _on_backends(plan(count=3), ["robolab.v1", "robolab.v1", "native_g1.v1"])
    coordinator.enqueue_screening_campaign(value, queue_root=tmp_path)
    _pass(tmp_path, {**HARNESS_FAILED, "blockers": [SHARED_STAGE]})
    running = _pass(tmp_path, {"status": "running"})
    # The simulator did not start for one RoboLab candidate: the other RoboLab one is held unspent.
    # The candidate on another backend has its own bundle and session, and runs.
    assert _rows(tmp_path) == [("blocked", [SHARED_STAGE]), ("blocked", [coordinator.HARNESS_HOLD_BLOCKER]),
                               ("running", [])]
    assert running.launches == 1
    assert publication(tmp_path)["budget"]["committed_upper_bound_usd"] == "8"


def test_a_run_that_names_no_stage_still_holds_its_backend(tmp_path):
    coordinator.enqueue_screening_campaign(plan(count=2), queue_root=tmp_path)
    _pass(tmp_path, HARNESS_FAILED)
    _pass(tmp_path, {"status": "running"})
    assert _rows(tmp_path) == [("blocked", HARNESS_FAILED["blockers"]),
                               ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])]


def test_a_same_commit_re_plan_leaves_held_rows_for_the_next_deploy(tmp_path):
    """A row held behind a sibling's shared-stage failure would fail the same way at this commit: a
    transient retry beside it never releases it, and the next deploy plans it again from the chain."""
    coordinator.enqueue_screening_campaign(plan(count=3), queue_root=tmp_path)
    _pass(tmp_path, TRANSIENT)                                    # candidate-000: its machine
    _pass(tmp_path, {**HARNESS_FAILED, "blockers": [SHARED_STAGE]})  # candidate-001: the simulator
    _pass(tmp_path, {"status": "running"})                        # candidate-002: held, unspent
    assert _rows(tmp_path) == [("blocked", [PACKAGE_MIRROR]), ("blocked", [SHARED_STAGE]),
                               ("blocked", [coordinator.HARNESS_HOLD_BLOCKER])]
    assert coordinator.replan_scope("campaign-1", implementation_commit=OLD, queue_root=tmp_path) == [
        "candidate-000"]
    with pytest.raises(coordinator.ScreeningCoordinatorError, match="replan_not_admitted"):
        coordinator.supersede_campaign("campaign-1", successor(plan(count=3), commit=OLD), implementation_commit=OLD,
                                       queue_root=tmp_path)
    coordinator.supersede_campaign("campaign-1", successor(plan(count=3), keep={"candidate-000"}, commit=OLD),
                                   implementation_commit=OLD, queue_root=tmp_path)
    # Nothing is lost: the deploy's re-plan takes the harness failure and the held row up again.
    assert coordinator.replan_scope("campaign-1b", implementation_commit=OLD, queue_root=tmp_path) is None
    assert coordinator.replan_scope("campaign-1b", implementation_commit=NEW, queue_root=tmp_path) == [
        "candidate-000", "candidate-001", "candidate-002"]


# --- reconciliation reads the scope from the run's own records ----------------------------------


def _reconciled(tmp_path, *, blockers, fallback=None, break_reason=None):
    """Reconcile one closed attempt whose allocator result carries ``blockers`` and no usable native result."""
    from tests.test_beta_policy_screening_partial_cells import Provider, build_attempt
    from blueprint_pipeline import gpu_render_providers

    value = plan(count=1)
    candidate = value["candidates"][0]
    attempt = tmp_path / "attempts" / "screen-1"
    prepared = build_attempt(value, candidate, attempt, successes=0)
    path = attempt / "allocator_result.json"
    adapter = json.loads(path.read_text())
    native = Path(adapter["native_control_result_path"])
    native.unlink()
    if fallback is not None:
        # What the entrypoint's fallback writes: a typed code, no digest, nothing to score.
        native.write_text(json.dumps({"schema_version": "native_task_arena_policy_canary_session_result.v1",
                                      "status": "blocked", "run_kind": BETA_RUN_KIND, "blockers": [fallback],
                                      "candidate_policy_queried": False, "worker_exit_code": 28}))
    adapter.update(status="blocked", blockers=[*blockers, *([fallback] if fallback else [])],
                   native_control_result_digest=None)
    path.write_text(json.dumps(adapter, indent=2, sort_keys=True))
    if break_reason is not None:
        run = Path(adapter["provider_closeout"]["adapter_result"]["path"]).parent
        (run / scope.STARTUP_PROBE_MANIFEST).write_text(json.dumps(
            {"container_log_result": {"break_reason": break_reason}}))
    original = gpu_render_providers.VastRenderProvider
    gpu_render_providers.VastRenderProvider = Provider
    try:
        return launch.reconcile_candidate(plan=value, candidate=candidate, attempt_dir=attempt, prepared=prepared)
    finally:
        gpu_render_providers.VastRenderProvider = original


@pytest.mark.parametrize("blockers,fallback,break_reason,expected", [
    ([], "policy_canary_media_toolchain_unavailable", None, [PACKAGE_MIRROR]),
    (NO_OUTPUT, None, "no_log_progress_timeout", [f"{scope.TRANSIENT_PROVISIONING_PREFIX}:provider_log_silent"]),
    (["provider_instance_exited_before_bundle_terminal_marker", *NO_OUTPUT], None, None,
     [f"{scope.TRANSIENT_PROVISIONING_PREFIX}:rented_machine"]),
    # The candidate's own policy provisioning: its own harness failure, no sibling held.
    ([], scope.POLICY_PROVISIONING_STAGE_CODE, None, [scope.POLICY_PROVISIONING_BLOCKER]),
    # Nothing typed narrows it: the harness failure it always was.
    ([], "policy_canary_entrypoint_failed_without_result", None, ["beta_screening_native_result_missing_or_invalid"]),
    (NO_OUTPUT, None, "max_wait_deadline_reached", ["beta_screening_native_result_missing_or_invalid"]),
])
def test_reconciliation_names_the_scope_of_a_closed_run_with_no_result(tmp_path, blockers, fallback, break_reason,
                                                                      expected):
    observed = _reconciled(tmp_path, blockers=blockers, fallback=fallback, break_reason=break_reason)
    # The money is the same in every case: allocated, torn down, closed.
    assert (observed["status"], observed["provider_allocation_performed"], observed["provider_zero_verified"]) == (
        "blocked", True, True)
    assert observed["blockers"] == expected


# --- the beta entrypoint names the stage it failed in ------------------------------------------


def _provisioning_step(tmp_path, *, run_kind, provisioning_exit):
    """Run the generated entrypoint's own fallback writer and policy provisioning step, nothing else."""
    script = _entrypoint(local_candidates=("molmoact2_droid",), run_kind=run_kind)
    writer = script[script.index("write_fallback_result() {"):script.index("teardown_servers() {")]
    start = script.index('cp "$RUNTIME_DIR/runtime_inputs/policy_execution_spec.molmoact2_droid.json"')
    step = script[start:script.index("\n\n", start)]
    runtime, out = tmp_path / "runtime", tmp_path / "out"
    (runtime / "runtime_inputs").mkdir(parents=True)
    out.mkdir()
    (runtime / "runtime_inputs" / "policy_execution_spec.molmoact2_droid.json").write_text("{}")
    (runtime / "adp009d_policy_provisioning.molmoact2_droid.sh").write_text(f"exit {provisioning_exit}\n")
    body = (f"set +e\nRUNTIME_DIR={shlex.quote(str(runtime))}\nOUT_DIR={shlex.quote(str(out))}\n"
            + writer + step + "\necho REACHED_THE_WORKER\n").replace("/isaac-sim/python.sh", shlex.quote(sys.executable))
    completed = subprocess.run(["bash", "-c", body], capture_output=True, text=True, timeout=30)
    return completed, out / PROVIDER_RESULT_FILENAME


def test_the_beta_entrypoint_names_its_candidates_failed_policy_provisioning(tmp_path):
    completed, result = _provisioning_step(tmp_path, run_kind=BETA_RUN_KIND, provisioning_exit=28)
    assert completed.returncode == 28 and "REACHED_THE_WORKER" not in completed.stdout, completed.stderr
    written = json.loads(result.read_text())
    assert written["blockers"] == [scope.POLICY_PROVISIONING_STAGE_CODE] and written["worker_exit_code"] == 28
    assert written["candidate_policy_queried"] is False


def test_a_provisioned_candidate_continues_and_other_run_kinds_keep_their_entrypoint(tmp_path):
    completed, result = _provisioning_step(tmp_path, run_kind=BETA_RUN_KIND, provisioning_exit=0)
    assert completed.returncode == 0 and "REACHED_THE_WORKER" in completed.stdout and not result.exists()
    # The ADP pair's entrypoint is untouched.
    other = _entrypoint()
    assert scope.POLICY_PROVISIONING_STAGE_CODE not in other
    assert other.count('.log" 2>&1 || exit $?') == 2
