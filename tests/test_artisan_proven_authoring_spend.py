"""B-04: a terminal ARTISAN attempt keeps its proven spend, not its maximum.

capture-447ae4f4, 2026-11-02: its failed attempts held about $58 against about
$18 spent, because every attempt that entered ARTISAN kept its whole authoring
allowance. The owner approved settling them at proven cost: the metered
authoring spend, once the official cost gate agrees with it, plus the GPU
allowance. Disagreeing, partial or ambiguous evidence keeps the full hold.

Every figure and identifier here is synthetic. The gate records come from the
real gate and the real usage meter, so a change to either shape fails here.
"""
import copy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import pytest

from blueprint_pipeline import task_evaluation_terminal_scene_attempt_settlement as settlement
from blueprint_pipeline import task_evaluation_unentered_authoring_budget as budget
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.openai_official_cost_gate import OpenAIOfficialCostRunGate
from blueprint_pipeline.task_evaluation_retained_controls_evidence import validated_cancellation
from blueprint_pipeline.task_evaluation_sam31_prefix_adoption import record
from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import write_output_archive
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import spend_block
from blueprint_pipeline.task_evaluation_scene_intake import (
    SceneIntakeError, committed_spend_usd, reserve_scene_attempt, stage_scene_intent,
)
from blueprint_pipeline.task_evaluation_scene_reservation_spend_evidence import scene_reservation_spend_record
from blueprint_pipeline.task_object_artisan_metering import Meter
from blueprint_pipeline.website_preparation_revision import shared_reserved_budget
from blueprint_pipeline.website_preparation_revision_identity import FIELD as revision_field
from blueprint_pipeline.website_preparation_revision_identity import PREFIX as revision_prefix
from tests.test_task_evaluation_scene_intake import request as intake_request

COMMIT = "d" * 40
AUTHORITY = "sha256:" + "a1" * 32
CANDIDATE = "sha256:" + "c2" * 32
PROJECT, KEY = "proj_synthetic", "key_synthetic"
STAGE3_FAILURE = ("scene_configuration_provider_failed:TaskEvaluationSceneConfigurationStageProducerError:"
                  "scene_configuration_stage_producer_failed:content_agents_rigid_replacement:1")
PRODUCER = "stages/stage-3/producer/"
RUNTIME = PRODUCER + "artisan_runtime/"
SANDBOX_LOG = ("    preflight_runner.preflight()\n"
               "blueprint_pipeline.asset_authoring_sandbox.AssetSandboxError: asset_sandbox_unavailable\n")
DEADLINE = time.time() + 7 * 86400


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True) + "\n")
    return path


class _Costs:
    """The official Costs endpoint: each snapshot reports the next day total."""

    def __init__(self, totals):
        self.totals = list(totals)

    def snapshot(self, *, start_time, end_time):
        return {"schema_version": "openai_organization_cost_snapshot.v1", "status": "complete",
                "project_id": PROJECT, "api_key_id": KEY, "start_time": start_time, "end_time": end_time,
                "total_cost_usd": self.totals.pop(0)}


class _Authority:
    scope_attestation_digest = "sha256:" + "5" * 64

    def __init__(self, baseline, totals):
        self.baseline, self.client = baseline, _Costs(totals)

    def reserve(self, *, candidate_id, candidate_evaluation_suite_digest, authorization_receipt_digest,
                max_cost_usd):
        value = {"schema_version": "candidate_policy_cost_reservation.v1", "status": "reserved",
                 "provider_id": "openai", "paid_resource_class": "task_evaluation_scene_configuration_content_agents",
                 "candidate_id": candidate_id, "candidate_evaluation_suite_digest": candidate_evaluation_suite_digest,
                 "authorization_receipt_digest": authorization_receipt_digest, "reserved_max_cost_usd": max_cost_usd,
                 "attribution_start_time": 1793577600, "attribution_end_time": 1793664000,
                 "reconciliation_not_before": "2026-11-04T00:00:00+00:00", "project_id": PROJECT, "api_key_id": KEY,
                 "baseline_cost_snapshot": {"total_cost_usd": self.baseline, "project_id": PROJECT,
                                            "api_key_id": KEY},
                 "baseline_cost_usd": self.baseline}
        return _seal(value, "cost_reservation_digest")


def _request(name, *, authoring, total=None):
    return {"run_id": "website-run-" + name, "preparation_id": f"website-{name}-preparation",
            "expected_production_commit": COMMIT,
            "spend": spend_block("artisan_v1", authoring_max_cost_usd=authoring, requires_artifixer=False,
                                 maximum_total_spend_usd=total)}


def _stage_tree(output, request, *, provider_blockers=(STAGE3_FAILURE,)):
    """Stages 1-2 and stage 3's inputs, as the runner leaves them before ARTISAN."""
    for stage in ("stage-1", "stage-2"):
        _write(output / f"stages/{stage}/adapter/result.json", {})
    stage_input = _write(output / PRODUCER / "stage_production_input.v1.json", {
        "run_id": request["run_id"], "source_commit": COMMIT, "construction_envelope": {},
        "configuration": {"authoring_backend": "artisan_v1", "source_observation_kind": "website_capture_frames"}})
    _write(output / PRODUCER / "dependency_results.v1.json", [_seal({
        "stage_id": stage, "status": "completed", "execution_class": "no_spend", "paid_execution_requested": False,
        "provider_mutations_performed": 0}, "stage_result_digest") for stage in ("stage-1", "stage-2")])
    _write(output / "task_evaluation_scene_configuration_provider_result.v1.json", _seal({
        "run_id": request["run_id"], "source_commit": COMMIT, "status": "blocked", "first_stage_started": True,
        "blockers": list(provider_blockers)}, "result_digest"))
    return stage_input


def _artisan_runtime(output, request, *, runs, cap=None, baseline=0.0, continuation=None):
    """Drive the real gate and meter the way the ARTISAN driver does, one controller run at a time.

    ``runs`` lists each controller run as ``(call costs, official day total at
    its snapshot, failed)``. A cost of ``None`` is a call still in flight; a
    total of ``None`` means the run never reached its snapshot (the instance died).
    """
    stage_input = _stage_tree(output, request)
    runtime = output / RUNTIME
    cap = request["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]["content_agents"] \
        if cap is None else cap
    issued = _seal({"run_id": request["run_id"], "backend": "artisan_v1",
                    "source_candidate": {"digest": CANDIDATE}}, "request_digest")
    _write(runtime / "blueprint_request.v1.json", issued)
    usage_file = "/workspace/runtime_output/" + RUNTIME + "blueprint_usage.json"
    _write(runtime / "bridge_context.v1.json", {"maximum_cost_usd": cap, "maximum_requests": 100,
                                                "deadline_epoch": DEADLINE, "usage_file": usage_file})
    if continuation is not None:
        source_calls, carried = continuation
        _write(runtime / "continuation/source/blueprint_usage.json", {
            "schema_version": "artisan_upstream_usage.v1", "calls": source_calls})
        _write(runtime / "continuation/source/adoption.v1.json", _seal({
            "schema_version": "task_evaluation_artisan_continuation_record.v1",
            "adoption_digest": "sha256:" + "ad" * 32, "source_run_id": "website-run-parent",
            "successor_run_id": request["run_id"], "source_request_digest": "sha256:" + "be" * 32,
            "successor_request_digest": issued["request_digest"], "upstream_run_id": "pl_synthetic",
            "cumulative_cap_usd": 25.0, "attempt_allocation_usd": cap, "carried_cost_usd": carried,
            "carried_call_count": len(source_calls), "carried_largest_call_usd": 0.0, "increment_usd": cap,
            "upstream_cap_usd": round(carried + cap, 6), "carried_spend_settled_under_source_attempt": True,
            "new_spend_authorized": False}, "record_digest"))
    totals = [total for _, total, _ in runs if total is not None]
    gate = OpenAIOfficialCostRunGate(
        authority=_Authority(baseline, totals), lane_id="task_evaluation_scene_configuration_content_agents",
        run_id=request["run_id"], request_digest="sha256:" + hashlib.sha256(stage_input.read_bytes()).hexdigest(),
        candidate_digest=CANDIDATE, authorization_receipt_digest=AUTHORITY, max_cost_usd=cap,
        output_root=runtime / "official_openai_cost")
    _write(runtime / "official_openai_cost/openai_cost_scope_attestation_content_agents.json", {})
    gate.reserve()
    terminal = None
    for costs, total, failed in runs:
        meter = Meter(runtime / "blueprint_usage.json", maximum_cost_usd=cap, maximum_requests=100,
                      deadline_epoch=DEADLINE)
        for cost in costs:
            identity = meter.before("create", "synthetic-model")
            if cost is None:
                break  # still in flight when the run ended
            try:
                meter.after(identity, cost)
            except RuntimeError:
                break  # the meter records the charge, then stops the run
        terminal = {"schema_version": "blueprint_artisan_terminal.v1", "returncode": int(failed),
                    "phase": "P5_review", "status": "running", "usage": meter.read()}
        _write(runtime / "terminal.v1.json", json.dumps(terminal))
        if total is None:
            return gate
        snapshot = gate
        if failed:
            attempts = runtime / "official_openai_cost/attempts"
            attempts.mkdir(exist_ok=True)
            snapshot = replace(gate, output_root=attempts / str(len(list(attempts.iterdir()))))
            shutil.copyfile(gate.reservation_path, snapshot.reservation_path)
        snapshot.complete(provider_call_performed=bool(costs), runtime_result_digest=canonical_digest(terminal),
                          runtime_exception_type="AuthoringContractError" if failed else None)
    return gate


def _pre_gate_refusal(output, request):
    """ARTISAN's sandbox preflight refused before its cost gate (capture-447ae4f4, 2026-11-01)."""
    _stage_tree(output, request)
    _write(output / RUNTIME / "blueprint_request.v1.json", {})
    _write(output / RUNTIME / "bridge_context.v1.json", {})
    _write(output / RUNTIME / "sandbox_preflight.sandbox_preflight.json", {"status": "blocked"})
    _write(output / PRODUCER / "stage_producer.log", SANDBOX_LOG)


def _result(run, request, output, *, prestage=False):
    """The lane's result. A failed CPU prestage keeps its own archive and failure record in the job dir."""
    job = run / "allocator/scene-configuration-job" if prestage else run
    archive = job / ("cpu_prestage_output.zip" if prestage else "vast_provider_runtime_output.zip")
    job.mkdir(parents=True, exist_ok=True)
    write_output_archive(output, archive)
    if prestage:
        _write(job / "cpu_prestage_failure.json", {
            "status": "blocked_before_gpu_allocation", "phase": "cpu_prestage", "run_id": request["run_id"],
            "source_commit": COMMIT, "failure": "TaskEvaluationSceneConfigurationCpuPrestageError:prefix_not_completed",
            "gpu_allocation_attempted": False, "raw_secret_values_recorded": False})
    with archive.open("rb") as stream:
        digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
    return _seal({"schema_version": "task_evaluation_scene_configuration_vast_result.v1", "status": "blocked",
                  "run_id": request["run_id"], "source_commit": COMMIT, "authority_digest": AUTHORITY,
                  "retry_cap": 0, "api_pretraining": None, "cpu_prestage": None,
                  "provider_mutations_performed": 0 if prestage else 1,
                  "continuing_spend_from_this_run": False,
                  "blockers": ["provider_result_blocker:" + STAGE3_FAILURE, "scene_configuration_provider_not_completed"],
                  "provider_runtime_output_zip_path": str(archive), "provider_runtime_output_zip_sha256": digest},
                 "result_digest")


def _proof(tmp_path, *, request=None, mutate=None, **runtime):
    request = request or _request("unit", authoring=15.0, total=20.0)
    output = tmp_path / "runtime_output"
    _artisan_runtime(output, request, **runtime)
    if mutate is not None:
        mutate(output)
    result = _result(tmp_path, request, output)
    return result, request


AGREEING = [([1.25, 2.5, 4.75], 7.0, True), ([5.26], 11.16, True)]


def test_agreeing_gate_and_meter_prove_the_larger_figure(tmp_path):
    result, request = _proof(tmp_path, runs=AGREEING)
    proof = budget.artisan_proven_authoring_spend(result, request)
    assert proof["proven_authoring_spend_usd"] == pytest.approx(13.76)
    assert proof["metered_cost_usd"] == pytest.approx(13.76) and proof["metered_call_count"] == 4
    assert proof["official_observed_cost_usd"] == 11.16 and proof["official_cost_final"] is False
    assert proof["policy_version"] == budget.PROVEN_AUTHORING_SPEND_POLICY
    assert proof["authoring_allowance_usd"] == proof["gate_maximum_cost_usd"] == 15.0
    assert proof["output_archive"] == {"path": result["provider_runtime_output_zip_path"],
                                       "sha256": result["provider_runtime_output_zip_sha256"]}
    with zipfile.ZipFile(result["provider_runtime_output_zip_path"]) as archive:
        for row in proof["records"]:
            assert row["sha256"] == "sha256:" + hashlib.sha256(archive.read(row["name"])).hexdigest()
    named = {row["name"] for row in proof["records"]}
    assert {RUNTIME + "official_openai_cost/openai_official_cost_run_reservation.v1.json",
            RUNTIME + "official_openai_cost/attempts/1/openai_official_cost_run_completion.v1.json",
            RUNTIME + "blueprint_usage.json", RUNTIME + "terminal.v1.json"} <= named


def test_an_official_figure_above_an_agreeing_meter_within_noise_is_the_proof(tmp_path):
    result, request = _proof(tmp_path, runs=[([4.0], 4.0000005, False)])
    assert budget.artisan_proven_authoring_spend(result, request)["proven_authoring_spend_usd"] == 4.0000005


def _metered_after_snapshot(output):
    meter = Meter(output / RUNTIME / "blueprint_usage.json", maximum_cost_usd=15.0, maximum_requests=100,
                  deadline_epoch=DEADLINE)
    meter.after(meter.before("create", "synthetic-model"), 0.5)


def _rewrite(name, change, field=None):
    def mutate(output):
        path = output / RUNTIME / name
        value = json.loads(path.read_text())
        change(value)
        _write(path, _seal(value, field) if field else value)
    return mutate


COMPLETION = "official_openai_cost/attempts/1/openai_official_cost_run_completion.v1.json"
RESERVATION = "official_openai_cost/openai_official_cost_run_reservation.v1.json"
FAILING = {
    # The official cost already shows spend the meter never saw.
    "official_above_meter": dict(runs=[([1.25, 2.5], 3.76, True)]),
    # The gate never completed: the instance died during the run.
    "no_completion": dict(runs=[([1.25, 2.5], None, True)]),
    "latest_run_without_snapshot": dict(runs=[([1.25], 1.0, True), ([2.5], None, True)]),
    "snapshot_reservation_without_completion": dict(runs=AGREEING, mutate=lambda output: (
        output / RUNTIME / COMPLETION).unlink()),
    # A call with no price: in flight, or answered without usage.
    "unpriced_call": dict(runs=[([1.25, None], 1.0, True)]),
    "call_without_usage": dict(runs=[([1.25, "unknown"], 1.0, True)]),
    "metered_after_snapshot": dict(runs=AGREEING, mutate=_metered_after_snapshot),
    "terminal_not_bound": dict(runs=AGREEING, mutate=_rewrite("terminal.v1.json",
                                                              lambda value: value.update(phase="P6"))),
    "completion_tampered": dict(runs=AGREEING, mutate=_rewrite(
        COMPLETION, lambda value: value.update(provider_observed_cost_usd=1.0))),
    "completion_resealed_inconsistent": dict(runs=AGREEING, mutate=_rewrite(
        COMPLETION, lambda value: value.update(provider_observed_cost_usd=1.0), "completion_receipt_digest")),
    "reservation_other_run": dict(runs=AGREEING, mutate=_rewrite(
        RESERVATION, lambda value: value.update(run_id="website-run-other"), "reservation_receipt_digest")),
    "unknown_gate_record": dict(runs=AGREEING, mutate=lambda output: _write(
        output / RUNTIME / "official_openai_cost/openai_official_cost_run_settlement.v1.json", {})),
    "meter_cap_differs": dict(runs=AGREEING, mutate=_rewrite(
        "bridge_context.v1.json", lambda value: value.update(maximum_cost_usd=20.0))),
    "missing_meter": dict(runs=AGREEING, mutate=lambda output: (output / RUNTIME / "blueprint_usage.json").unlink()),
    "malformed_meter": dict(runs=AGREEING, mutate=lambda output: _write(
        output / RUNTIME / "blueprint_usage.json", "{not json")),
    "later_stage": dict(runs=AGREEING, mutate=lambda output: _write(output / "stages/stage-4/adapter/result.json", {})),
    "continuation_without_adoption": dict(runs=AGREEING, mutate=lambda output: _write(
        output / RUNTIME / "continuation/source/blueprint_usage.json", {"calls": []})),
    "spend_above_gate_cap": dict(runs=[([9.0, 8.0], 16.0, True)]),
}


@pytest.mark.parametrize("case", sorted(FAILING))
def test_disagreeing_partial_or_ambiguous_evidence_proves_nothing(tmp_path, case):
    result, request = _proof(tmp_path, **FAILING[case])
    assert budget.artisan_proven_authoring_spend(result, request) is None


def test_a_changed_archive_proves_nothing(tmp_path):
    result, request = _proof(tmp_path, runs=AGREEING)
    with open(result["provider_runtime_output_zip_path"], "ab") as stream:
        stream.write(b"changed")
    assert budget.artisan_proven_authoring_spend(result, request) is None


def test_a_signed_allowance_below_the_gate_cap_proves_nothing(tmp_path):
    result, request = _proof(tmp_path, runs=AGREEING)
    narrowed = copy.deepcopy(request)
    narrowed["spend"]["external_service_caps"]["openai"]["stage_max_cost_usd"]["content_agents"] = 14.0
    assert budget.artisan_proven_authoring_spend(result, narrowed) is None


PARENT_CALLS = [{"id": "parent-%d" % index, "status": "priced", "cost_usd": cost}
                for index, cost in enumerate((1.25, 2.5, 4.75, 5.26))]


def test_a_continuation_proves_only_its_own_increment(tmp_path):
    request = _request("successor", authoring=11.24)
    result, request = _proof(tmp_path, request=request, runs=[([1.0, 2.25], 12.0, True)], baseline=12.0,
                             continuation=(PARENT_CALLS, 13.76))
    proof = budget.artisan_proven_authoring_spend(result, request)
    assert proof["proven_authoring_spend_usd"] == 3.25
    assert proof["continuation"] == {"source_run_id": "website-run-parent", "carried_cost_usd": 13.76,
                                     "carried_spend_settled_under_source_attempt": True,
                                     "carried_spend_counted_here": False}


@pytest.mark.parametrize("change", ["carried_call_metered_again", "increment_not_the_gate_cap"])
def test_a_continuation_that_may_count_carried_spend_proves_nothing(tmp_path, change):
    request = _request("successor", authoring=11.24)

    def mutate(output):
        if change == "carried_call_metered_again":
            path = output / RUNTIME / "continuation/source/blueprint_usage.json"
            usage = json.loads(path.read_text())
            successor = json.loads((output / RUNTIME / "blueprint_usage.json").read_text())
            usage["calls"].append(successor["calls"][0])
            _write(path, usage)
        else:
            _rewrite("continuation/source/adoption.v1.json", lambda value: value.update(increment_usd=25.0),
                     "record_digest")(output)

    result, request = _proof(tmp_path, request=request, runs=[([1.0, 2.25], 12.0, True)], baseline=12.0,
                             continuation=(PARENT_CALLS, 13.76), mutate=mutate)
    assert budget.artisan_proven_authoring_spend(result, request) is None


# ---------------------------------------------------------------------------
# The capture-447 shape: one intent, several terminal website attempts, settled
# and then read by reservation, the shared sibling budget and the spend ledger.


def _preparation_request(spend=70.0):
    value = copy.deepcopy(intake_request())
    value["execution"].update({"max_total_spend_usd": spend, "max_paid_attempts": 8, "purpose": "scene_preparation",
                               "allowed_providers": ["vast", "openai"], "policy_candidates": []})
    return value


def _intent(root):
    value = _preparation_request()
    receipt = stage_scene_intent(value=value, queue_root=root, authenticated_client="webapp",
                                 trusted_clients={"webapp"}, now=100)
    return json.loads((root / receipt["intent_id"] / "intent.json").read_text())


@pytest.fixture
def site(tmp_path, monkeypatch):
    root = tmp_path / "intents"
    intent = _intent(root)
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(root))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "webapp")
    queue = tmp_path / "launches"
    for state in ("pending", "processing"):
        (queue / state).mkdir(parents=True)
    return {"tmp": tmp_path, "root": root, "intent": intent, "directory": root / intent["intent_id"],
            "launches": tmp_path / "launch-runs", "queue": queue}


def _terminal_attempt(site, name, request, build, *, prestage=False, after=None):
    """Reserve one website scene attempt, run it to a terminal blocked launch, and settle it.

    ``prestage`` runs it the way a failed CPU prestage ends: no GPU, the
    prestage's archive retained, and a teardown that never had to run.
    ``after(run)`` changes the retained records before the launch is sealed.
    """
    from blueprint_pipeline.task_evaluation_scene_preparation_attempts import create_preparation_attempt
    directory, intent = site["directory"], site["intent"]
    source = create_preparation_attempt(directory=directory, attempt_id="source-" + name, source_commit=COMMIT,
                                        runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
                                        now=101)
    digest = canonical_digest(request)
    row_id = next(iter(settlement.dependent_row_ids(digest)))
    reserve_scene_attempt(queue_root=site["root"], intent_id=intent["intent_id"], attempt_id=row_id,
                          source_commit=COMMIT, runtime_digest="sha256:" + "e" * 64, input_digest=digest,
                          provider="vast", maximum_spend_usd=request["spend"]["hard_cap_usd"], now=102)
    _write(directory / "preparations" / (digest[7:] + ".json"), _seal({
        "schema_version": "task_evaluation_scene_preparation_link.v1", "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"], "expected_production_commit": COMMIT,
        "preparation_id": request["preparation_id"], "request_digest": digest, "result_filename": "r.json",
        "scene_id": "s", "task_id": "t", "team_namespace": "n"}, "link_digest"))
    out = site["tmp"] / "out" / name
    transition = _write(out / "release-transition.json", _seal({
        "schema_version": "task_evaluation_scene_release_transition.v1", "attempt_digest": source["attempt_digest"],
        "observed_at_epoch": 200, "parent_envelope": {}, "parent_state": "materialized",
        "provider_allocation_performed": True}, "failure_digest"))
    ownership = _write(out / "ownership.json", _seal({
        "schema_version": "task_evaluation_scene_attempt_ownership.v1", "attempt_digest": source["attempt_digest"],
        "status": "closed_without_resource", "active_writer_count": 0, "unresolved_create_count": 0,
        "observed_at_epoch": 201, "provider_mutation_performed": False}, "ownership_digest"))
    factory = record(_write(out / "factory.json", _seal({
        "schema_version": "website_scene_attempt_factory.v1", "attempt_digest": source["attempt_digest"],
        "intent_digest": intent["intent_digest"], "source_commit": COMMIT,
        "submission_request": record(_write(out / "request.json", request))}, "factory_digest")))
    launch_id = settlement.launch_id_for_preparation(request["preparation_id"])
    run = site["launches"] / launch_id
    output = out / "runtime_output"
    build(output, request)
    result = _write(run / "result.json", _result(run, request, output, prestage=prestage))
    teardown = _write(run / "teardown.json", {"schema_version": "vast_teardown_manifest.v1",
        "status": "not_required_provider_adapter_never_invoked" if prestage else "completed",
        "vast_instance_ids": [] if prestage else [7], "continuing_spend_from_this_run": False})
    if after is not None:
        after(run)
    _write(run / "launch_receipt.json", {"launch_id": launch_id, "source_commit": COMMIT, "status": "blocked",
        "terminal_evidence": {"result": {**settlement._file(result), "exists": True},
                              "artifacts": {"teardown_manifest_path": {**settlement._file(teardown), "exists": True}}}})
    outcome = settlement.settle_retired_attempt_rows(
        directory=directory, retired_attempt=source, retirement_record={"path": str(transition)},
        ownership_record={"path": str(ownership)}, launch_execution_root=site["launches"],
        launch_queue_root=site["queue"], source_factory=factory)
    assert outcome["rows"] == [{"attempt_id": row_id, "status": "settled"}]
    return directory / "attempts" / (row_id + ".json")


def _hold(site, row_path):
    receipt = validated_cancellation(site["directory"], json.loads(row_path.read_text()))
    # The sealed settlement itself is never rewritten: it still records the full maximum.
    assert receipt["settled_spend"] == {"basis": "terminal_launch_unreconciled", "counts_as_attempt": False,
                                        "retained_spend_usd": json.loads(row_path.read_text())["maximum_spend_usd"]}
    return settlement.budget_retained_hold(receipt)


def _capture_447(site):
    """Five terminal attempts on one $70 intent, in the order capture-447ae4f4 ran them."""
    rows = {}
    rows["pre_gate"] = _terminal_attempt(site, "pregate", _request("pregate", authoring=15.0, total=20.0),
                                         _pre_gate_refusal)
    rows["parent"] = _terminal_attempt(site, "parent", _request("parent", authoring=15.0, total=20.0),
        lambda output, request: _artisan_runtime(output, request, runs=AGREEING))
    rows["resumed"] = _terminal_attempt(site, "resumed", _request("resumed", authoring=15.0, total=20.0),
        lambda output, request: _artisan_runtime(output, request, baseline=11.16,
                                                 runs=[([2.0, 1.5], 13.66, True), ([0.75], 15.16, True)]))
    # Continues the parent: carries its $13.76 and reserves only the rest of the $25 cap.
    rows["continuation"] = _terminal_attempt(site, "successor", _request("successor", authoring=11.24),
        lambda output, request: _artisan_runtime(output, request, baseline=15.16, runs=[([1.0, 2.25], 15.16, True)],
                                                 continuation=(PARENT_CALLS, 13.76)))
    rows["no_completion"] = _terminal_attempt(site, "died", _request("died", authoring=15.0, total=20.0),
        lambda output, request: _artisan_runtime(output, request, baseline=18.41, runs=[([3.0, 4.0], None, True)]))
    return rows


def test_capture_447_holds_total_the_proven_bound(site, monkeypatch):
    rows = _capture_447(site)
    holds = {name: _hold(site, path) for name, path in rows.items()}
    assert {name: (hold["basis"], hold["retained_spend_usd"]) for name, hold in holds.items()} == {
        "pre_gate": ("native_allowance_with_unentered_authoring", 5.0),
        "parent": (settlement.PROVEN_AUTHORING_SPEND_BASIS, 18.76),      # $5 GPU + $13.76 metered
        "resumed": (settlement.PROVEN_AUTHORING_SPEND_BASIS, 9.25),      # $5 GPU + $4.25 over two runs
        "continuation": (settlement.PROVEN_AUTHORING_SPEND_BASIS, 9.25),  # $6 GPU + its own $3.25 only
        "no_completion": ("terminal_launch_unreconciled", 20.0),          # unproven: the full maximum
    }
    assert holds["continuation"]["evidence"]["continuation"]["carried_cost_usd"] == 13.76
    assert holds["continuation"]["evidence"]["kept_allowance_usd"] == 6.0
    proven = Decimal("62.26")
    assert sum(Decimal(str(hold["retained_spend_usd"])) for hold in holds.values()) == proven
    assert committed_spend_usd(site["directory"], site["intent"]) == proven

    # Reservation counts the same bound: what is left of the $70 fits, a cent more does not.
    with pytest.raises(SceneIntakeError, match="spend_cap_exhausted"):
        reserve_scene_attempt(queue_root=site["root"], intent_id=site["intent"]["intent_id"],
                              attempt_id="scene-configuration-next-over", source_commit=COMMIT,
                              runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
                              provider="vast", maximum_spend_usd=7.75, now=300)
    assert reserve_scene_attempt(queue_root=site["root"], intent_id=site["intent"]["intent_id"],
                                 attempt_id="scene-configuration-next", source_commit=COMMIT,
                                 runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
                                 provider="vast", maximum_spend_usd=7.74, now=300)["status"] == "reserved"

    # Without the proofs the same settled rows would hold every maximum but the pre-gate one.
    monkeypatch.setattr(budget, "artisan_proven_authoring_spend", lambda result, request: None)
    assert sum(Decimal(str(_hold(site, path)["retained_spend_usd"])) for path in rows.values()) == Decimal("82.24")


def test_shared_sibling_budget_and_spend_ledger_read_the_proven_holds(site):
    rows = _capture_447(site)
    # A corrected derivative of the same capture is a sibling intent: it counts these holds too.
    corrected = _preparation_request()
    corrected["task"]["subject"][revision_field] = {"capture_id": corrected["submission_id"],
                                                    "corrections_digest": "sha256:" + "9" * 64}
    body = {key: item for key, item in corrected.items() if key != "submission_id"}
    corrected["submission_id"] = revision_prefix + cross_runtime_canonical_digest(body)[7:]
    exposure, attempts = shared_reserved_budget(site["root"], site["root"] / "scene-corrected",
                                                {"request": corrected})
    assert (exposure, attempts) == (Decimal("62.26"), 0)
    _, ledger = scene_reservation_spend_record(rows["parent"])
    assert ledger["hard_attempt_spend_cap_usd"] == 18.76
    assert ledger["settled_spend"]["basis"] == settlement.PROVEN_AUTHORING_SPEND_BASIS
    evidence = ledger["settled_spend"]["evidence"]
    assert evidence["policy_version"] == budget.PROVEN_AUTHORING_SPEND_POLICY
    assert evidence["output_archive"]["sha256"].startswith("sha256:") and evidence["records"]
    _, unproven = scene_reservation_spend_record(rows["no_completion"])
    assert unproven["hard_attempt_spend_cap_usd"] == 20.0 and "settled_spend" not in unproven


@pytest.mark.parametrize("case", ["official_above_meter", "no_completion", "unpriced_call", "metered_after_snapshot"])
def test_settled_rows_with_disagreeing_or_partial_evidence_keep_their_maximum(site, case):
    spec = FAILING[case]

    def build(output, request):
        _artisan_runtime(output, request, runs=spec["runs"])
        if "mutate" in spec:
            spec["mutate"](output)

    row = _terminal_attempt(site, "only", _request("only", authoring=15.0, total=20.0), build)
    assert _hold(site, row) == {"basis": "terminal_launch_unreconciled", "retained_spend_usd": 20.0,
                                "counts_as_attempt": False}
    assert committed_spend_usd(site["directory"], site["intent"]) == Decimal("20.0")


def test_proof_changed_after_settlement_restores_the_maximum(site):
    row = _terminal_attempt(site, "only", _request("only", authoring=15.0, total=20.0),
                            lambda output, request: _artisan_runtime(output, request, runs=AGREEING))
    assert _hold(site, row)["retained_spend_usd"] == 18.76
    receipt = validated_cancellation(site["directory"], json.loads(row.read_text()))
    result = Path(json.loads(Path(receipt["execution_terminal"]["launch_receipt"]["path"]).read_text())
                  ["terminal_evidence"]["result"]["path"])
    with open(json.loads(result.read_text())["provider_runtime_output_zip_path"], "ab") as stream:
        stream.write(b"changed")
    assert _hold(site, row)["retained_spend_usd"] == 20.0


# ---------------------------------------------------------------------------
# The same proof when ARTISAN ran on the CPU prestage and failed before any GPU
# (#3259 sizes the prestage to fit, so capture-447's restart runs stages 1-4 on
# the control plane first). No GPU was rented, so no compute allowance is kept.


def _prestage_attempt(site, *, runs=AGREEING, after=None):
    """A $31 attempt ($25 ARTISAN cumulative cap + $6 GPU) whose prestage failed in ARTISAN."""
    return _terminal_attempt(site, "prestage", _request("prestage", authoring=25.0, total=31.0),
                             lambda output, request: _artisan_runtime(output, request, runs=runs),
                             prestage=True, after=after)


def test_a_failed_cpu_prestage_keeps_only_its_proven_authoring_spend(site, monkeypatch):
    row = _prestage_attempt(site)
    hold = _hold(site, row)
    assert (hold["basis"], hold["retained_spend_usd"]) == (settlement.PROVEN_PRESTAGE_AUTHORING_SPEND_BASIS, 13.76)
    assert hold["evidence"]["compute_allowance_kept"] is False and hold["evidence"]["compute_allowance_usd"] == 6.0
    assert hold["evidence"]["output_archive"]["path"].endswith("/cpu_prestage_output.zip")
    assert committed_spend_usd(site["directory"], site["intent"]) == Decimal("13.76")
    # The restart's next $31 attempt now fits beside it.
    assert reserve_scene_attempt(queue_root=site["root"], intent_id=site["intent"]["intent_id"],
                                 attempt_id="scene-configuration-restart", source_commit=COMMIT,
                                 runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
                                 provider="vast", maximum_spend_usd=31.0, now=300)["status"] == "reserved"
    monkeypatch.setattr(budget, "artisan_proven_authoring_spend", lambda result, request: None)
    assert _hold(site, row)["retained_spend_usd"] == 31.0


def _job(run):
    return run / "allocator/scene-configuration-job"


def _reseal_result(run, **changes):
    path = run / "result.json"
    value = json.loads(path.read_text())
    value.update(changes)
    _write(path, _seal(value, "result_digest"))


def _retouch(name, **changes):
    def change(run):
        path = _job(run) / "cpu_prestage_failure.json" if name == "failure" else run / "teardown.json"
        _write(path, {**json.loads(path.read_text()), **changes})
    return change


PRESTAGE_FAILING = {
    # The records must prove no provider allocation, or the GPU allowance stays too.
    "failure_record_missing": dict(after=lambda run: (_job(run) / "cpu_prestage_failure.json").unlink()),
    "gpu_allocation_attempted": dict(after=_retouch("failure", gpu_allocation_attempted=True)),
    "instance_recorded": dict(after=_retouch("teardown", vast_instance_ids=[7])),
    "teardown_ran": dict(after=_retouch("teardown", status="completed")),
    "provider_mutation": dict(after=lambda run: _reseal_result(run, provider_mutations_performed=1)),
    "prestage_completed": dict(after=lambda run: _reseal_result(run, cpu_prestage={"status": "completed_prefix"})),
    # The authoring proof itself fails closed exactly as on the GPU.
    "official_above_meter": dict(runs=[([1.25, 2.5], 3.76, True)]),
    "no_completion": dict(runs=[([1.25, 2.5], None, True)]),
    "unpriced_call": dict(runs=[([1.25, None], 1.0, True)]),
}


@pytest.mark.parametrize("case", sorted(PRESTAGE_FAILING))
def test_a_failed_cpu_prestage_without_both_proofs_keeps_its_maximum(site, case):
    row = _prestage_attempt(site, **PRESTAGE_FAILING[case])
    assert _hold(site, row) == {"basis": "terminal_launch_unreconciled", "retained_spend_usd": 31.0,
                                "counts_as_attempt": False}


@pytest.mark.parametrize("shape", ["archive_without_stages", "no_archive"])
def test_a_cpu_prestage_that_failed_before_its_first_stage_settles_at_zero(site, shape):
    """Existing proofs: ``prestage_before_first_stage``, or no output and no API preparation at all."""
    early = "scene_configuration_provider_failed:TimeoutExpired"

    def build(output, request):
        _write(output / "task_evaluation_scene_configuration_provider_result.v1.json", _seal({
            "schema_version": "task_evaluation_scene_configuration_provider_result.v1", "status": "blocked",
            "run_id": request["run_id"], "source_commit": COMMIT, "first_stage_started": False,
            "evaluation_episode_executed": False, "candidate_policy_queried": False, "blockers": [early]},
            "result_digest"))

    def after(run):
        if shape == "no_archive":
            # The prestage stopped before it wrote an archive (work dir or disk refused).
            (_job(run) / "cpu_prestage_output.zip").unlink()
            _reseal_result(run, provider_runtime_output_zip_path=None, provider_runtime_output_zip_sha256=None,
                           blockers=["vast_adapter_failed:TaskEvaluationSceneConfigurationCpuPrestageError"])
        else:
            _reseal_result(run, blockers=["provider_result_blocker:" + early])

    row = _terminal_attempt(site, "early", _request("early", authoring=25.0, total=31.0), build,
                            prestage=True, after=after)
    hold = _hold(site, row)
    assert hold["retained_spend_usd"] == 0.0
    assert hold["basis"] == ("preallocation_unentered_authoring" if shape == "archive_without_stages"
                             else "preallocation_api_budget_upper_bound")
