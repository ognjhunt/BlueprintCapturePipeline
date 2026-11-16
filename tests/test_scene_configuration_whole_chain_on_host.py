# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_scene_without_simulator.py
#   src/blueprint_pipeline/website_native_background.py
#   src/blueprint_pipeline/task_evaluation_scene_configuration_vast.py
#   src/blueprint_pipeline/task_evaluation_scene_configuration_cpu_prestage.py
"""A website rigid job's scene: all six stages on this host, no machine rented (owner, 2026-11-15).

End to end without paying, through the lane's real code:

- the real ``run_scene_configuration_vast`` with the repository's own lane doubles (staging,
  watchdog, object stores; ``tests/test_task_evaluation_scene_configuration_output_admission``);
- the real ``prepare_stage_prefix_before_gpu(whole_chain=True)``, whose entrypoint double runs
  the real ``execute_scene_configuration_stage_chain`` over the six stages: the real stage-5
  record (``website_scene_without_simulator``) and the real stage-6 assembly, with the
  repository's role doubles for stages 1-4 (ARTISAN and the website background cannot run here);
- the real provider runner's portable projection, the real output archive, extraction,
  execution binding and publication, and the sealed terminal result.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_cpu_prestage as cpu_prestage
from blueprint_pipeline import task_evaluation_scene_configuration_output_admission as admission
from blueprint_pipeline import task_evaluation_scene_configuration_vast as scene_vast
from blueprint_pipeline import website_native_background
from blueprint_pipeline import website_scene_without_simulator as not_run
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES, SceneConfigurationAdapterRegistry,
)
from blueprint_pipeline.task_evaluation_scene_configuration_builtin_adapters import (
    builtin_scene_configuration_adapter_handlers,
)
from blueprint_pipeline.task_evaluation_scene_configuration_orchestrator import STAGE_RESULT_SCHEMA_VERSION
from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import (
    execute_scene_configuration_stage_chain,
)
from blueprint_pipeline import task_evaluation_scene_configuration_submission_records as records
from tests.test_task_evaluation_scene_configuration_output_admission import (
    GIB, ROLES, _harness, _thumbnail_review,
)
from tests.test_task_evaluation_scene_configuration_provider_runtime import _producers

ROOT = Path(__file__).resolve().parents[1]
RESULT_NAME = "task_evaluation_scene_configuration_provider_result.v1.json"
#: A website rigid job's recipe as ``prepare_construction_stages`` now writes it.
HOST_ADAPTERS = ("website_prepared_appearance", "website_prepared_collision", "content_agents_rigid_replacement",
                 "simready_static_rigid_qualification", not_run.ADAPTER_ID, "native_task_scene_assembly")
STAGE_ROLES = {
    "stage-1": ("configured_appearance_without_source_object", "appearance_removal_receipt",
                "appearance_visual_review_receipt", "configured_task_thumbnail"),
    "stage-2": ("configured_collision_without_source_object", "collision_excision_receipt"),
    "stage-3": (),
}


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _host_stages() -> list[dict]:
    rows = []
    for index, adapter in enumerate(HOST_ADAPTERS, start=1):
        identity = next(row for row in ADMITTED_STAGE_ADAPTER_IDENTITIES if row.adapter_id == adapter)
        rows.append({"stage_id": f"stage-{index}", "capability": identity.capability,
                     "adapter": {"id": adapter, "version": identity.version},
                     "execution_class": identity.execution_class,
                     "depends_on": [] if index == 1 else [f"stage-{index - 1}"]})
    return rows


def _double_result(stage, configuration_path, artifacts):
    result = {"schema_version": STAGE_RESULT_SCHEMA_VERSION, "status": "completed", "stage_id": stage["stage_id"],
              "capability": stage["capability"], "execution_class": stage["execution_class"],
              "configuration_digest": _sha(configuration_path), "retry_cap": 0, "raw_secret_values_recorded": False,
              "canonical_allocator": None, "provider_mutations_performed": 0, "paid_execution_requested": False,
              "executed_inside_parent_configuration_run": True, "output_artifacts": artifacts,
              "stage_result_digest": ""}
    result["stage_result_digest"] = canonical_digest(result, digest_field="stage_result_digest")
    return result


def _registry(identity: dict) -> SceneConfigurationAdapterRegistry:
    """Real stage 5 and 6; role doubles for 1-4 with the bytes the publisher's fixture expects."""
    real = builtin_scene_configuration_adapter_handlers()
    review, thumbnail = _thumbnail_review()
    special = {"appearance_visual_review_receipt": review, "configured_task_thumbnail": thumbnail}

    def double(*, stage, configuration_path, output_root, **_kwargs):
        artifacts = []
        if stage["stage_id"] == "stage-4":
            asset = output_root / "replacement_asset.usdz"
            asset.write_bytes(b"PK fixture usdz bytes of the authored object")
            receipt = {"status": "authored_structure_statically_qualified",
                       "authored_structure_statically_qualified": True, "replacement_identity": identity,
                       "replacement_usd": {"sha256": _sha(asset), "size_bytes": asset.stat().st_size},
                       "claim_boundary": {"native_simulator_import_qualified": False}}
            receipt_path = output_root / "static_qualification_receipt.v1.json"
            receipt_path.write_text(json.dumps(receipt))
            rows = [("statically_qualified_replacement_asset", asset), ("static_qualification_receipt", receipt_path)]
        else:
            rows = []
            for role in STAGE_ROLES[stage["stage_id"]]:
                path = output_root / ROLES[role]
                path.write_bytes(special.get(role, (role + "\n").encode()))
                rows.append((role, path))
        for role, path in rows:
            artifacts.append({"role": role, "path": str(path), "digest": _sha(path), "size_bytes": path.stat().st_size})
        return _double_result(stage, configuration_path, artifacts)

    handlers = {}
    for row in ADMITTED_STAGE_ADAPTER_IDENTITIES:
        if row.adapter_id in {not_run.ADAPTER_ID, "native_task_scene_assembly"}:
            handlers[row] = real[row]
        elif row.adapter_id in HOST_ADAPTERS:
            handlers[row] = double
    return SceneConfigurationAdapterRegistry(handlers)


def _runner_module():
    path = ROOT / "scripts/task_evaluation_scene_configuration_provider_runner.py"
    spec = importlib.util.spec_from_file_location("whole_chain_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain_entrypoint(*, receipt: dict, publication_envelope: dict, seen: dict):
    """The bundle entrypoint as the host runs it: the real chain, then the runner's real result."""

    def run(argv, *, cwd, env, stdout, stderr, check):
        seen.update(env)
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT"])
        recipe = publication_envelope["recipe"]
        envelope = {"run_id": receipt["run_id"], "team_namespace": publication_envelope["team_namespace"],
                    "request": scene_vast._portable_construction_envelope(receipt)["request"],
                    "expected_production_commit": receipt["source_commit"],
                    "recipe": {**recipe, "stage_sequence": _host_stages()}}
        configurations = {}
        values = {"stage-5": not_run.stage_five_configuration(replacement_identity=recipe["subject_identity"]),
                  "stage-6": records.stage_six_configuration(
                      scene_identity=recipe["scene_identity"],
                      support_plane={"sage_prim_path": "/Root", "publisher_instance_id": "website-support",
                                     "top_z_m": 0.8, "bounds_min_xyz_m": [0, 0, 0], "bounds_max_xyz_m": [1, 1, 0.8]},
                      start_center=[0.5, 0.5, 0.85], bottom_z=0.8)}
        for stage in envelope["recipe"]["stage_sequence"]:
            value = values.get(stage["stage_id"], {"stage": stage["stage_id"]})
            path = output.parent / "configuration" / f"{stage['stage_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            configurations[stage["stage_id"]] = (value, path)
        (output / "stages").mkdir(parents=True, exist_ok=True)
        chain = execute_scene_configuration_stage_chain(
            envelope=envelope, configurations=configurations, output_root=output / "stages",
            registry=_registry(recipe["subject_identity"]), producer_registry=_producers(),
            parent_deadline_epoch=time.time() + 5 * 3600,
            stage_limit=env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT"] or None)
        runner = _runner_module()
        portable = runner._portable_stage_chain(chain, output_root=output)
        result = {"schema_version": runner.RESULT_SCHEMA_VERSION, "status": portable["status"],
                  "run_id": receipt["run_id"], "source_commit": receipt["source_commit"],
                  "construction_envelope_digest": receipt["portable_construction_envelope_digest"],
                  "source_construction_envelope_digest": receipt["construction_envelope_source_digest"],
                  "stage_chain": portable, "first_stage_started": True, "evaluation_episode_executed": False,
                  "candidate_policy_queried": False, "provider_zero_required_after_return": True,
                  "blockers": [], "result_digest": ""}
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        (output / RESULT_NAME).write_text(canonical_json(result) + "\n")
        stdout.write(b"whole chain ran on the host\n")
        return type("Completed", (), {"returncode": 0})()

    return run


def _host_lane(tmp_path, monkeypatch, *, configured_limit: str | None = "stage-4", free_bytes: int = 90 * GIB):
    lane = _harness(tmp_path, monkeypatch, free_bytes=free_bytes)
    envelope = scene_vast._portable_construction_envelope(lane.receipt)
    for row, adapter in zip(envelope["recipe"]["stage_sequence"], HOST_ADAPTERS, strict=True):
        identity = next(item for item in ADMITTED_STAGE_ADAPTER_IDENTITIES if item.adapter_id == adapter)
        row.update(adapter={"id": adapter, "version": "v1"}, execution_class=identity.execution_class)
    work = tmp_path / "prestage-work"
    work.mkdir()
    monkeypatch.setenv(cpu_prestage.WORK_DIR_ENV, str(work))
    monkeypatch.setenv(cpu_prestage.STAGE_LIMIT_ENV, configured_limit or "")
    seen: dict = {}
    publication_envelope = scene_vast._publication_envelope(lane.receipt, output_root=tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python3").write_text("")
    real_prepare = cpu_prestage.prepare_stage_prefix_before_gpu
    monkeypatch.setattr(cpu_prestage, "prepare_stage_prefix_before_gpu", functools.partial(
        real_prepare, runner=_chain_entrypoint(receipt=lane.receipt, publication_envelope=publication_envelope,
                                               seen=seen),
        python_bin_dir=tmp_path / "bin", reservation_root=lane.ledger, disk_usage=lane.volume))
    lane.seen = seen
    lane.work = work
    return lane


def test_a_website_rigid_scene_is_built_end_to_end_on_the_host_and_rents_nothing(tmp_path, monkeypatch):
    lane = _host_lane(tmp_path, monkeypatch)

    result = lane.run(cpu_prestage_stage_limit="stage-4")

    assert result["status"] == "completed", result["blockers"]
    assert result["configuration_completed"] is True and result["configured_scene_published"] is True
    # Nothing was rented, staged, armed or called at a provider.
    assert [name for name, _ in lane.events if name in {"adapter", "stage", "watchdog_close"}] == []
    assert result["independent_watchdog"]["status"] == cpu_prestage.WATCHDOG_NOT_ARMED_STATUS
    assert result["provider_mutations_performed"] == 0 and result["continuing_spend_from_this_run"] is False
    assert result["object_store_cleanup"]["status"] == "nothing_staged_every_stage_ran_on_host"
    # F1: these records are the proof a failed publication of this run is republished at $0.
    from blueprint_pipeline.task_evaluation_scene_configuration_publication_recovery import _no_provider_resource
    assert _no_provider_resource(result)
    # The records say plainly that no GPU was allocated.
    job = lane.job
    adapter = json.loads((job / "vast_provider_run/vast_provider_adapter_result.json").read_text())
    teardown = json.loads((job / "vast_provider_run/vast_teardown_manifest.json").read_text())
    assert adapter["status"] == cpu_prestage.NOT_INVOKED_STATUS and adapter["gpu_allocated"] is False
    assert adapter["provider_create_attempted"] is False and adapter["vast_instance_ids"] == []
    assert teardown["status"] == "not_required_provider_adapter_never_invoked" and teardown["gpu_allocated"] is False
    prestage = result["cpu_prestage"]
    assert prestage["status"] == "completed_whole_chain_without_gpu" and prestage["gpu_allocated"] is False
    assert prestage["completed_stage_ids"] == [f"stage-{i}" for i in range(1, 7)]
    # The host's prefix setting (stage-4) did not cut the chain: the entrypoint ran with no limit.
    assert lane.seen["BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT"] == ""
    assert Path(result["provider_runtime_output_zip_path"]).name == "cpu_prestage_output.zip"
    # Stage 5 passed stage 4's object on byte for byte and says no simulator ran.
    execution = json.loads((job / "immutable_execution" / RESULT_NAME).read_text())
    stages = execution["stage_chain"]["stage_results"]
    by_role = {row["role"]: row for stage in stages for row in stage["output_artifacts"]}
    assert by_role["native_qualified_replacement_asset"]["digest"] == by_role[
        "statically_qualified_replacement_asset"]["digest"]
    receipt_path = job / "immutable_execution" / by_role[
        "native_import_qualification_receipt"]["provider_output_relative_path"]
    receipt = json.loads(receipt_path.read_text())
    assert not_run.stage_five_receipt_kind(receipt) == "not_run"
    assert receipt["native_isaac_executed"] is False and receipt["native_simulator_import_qualified"] is False
    manifest = json.loads((job / "immutable_execution" / by_role[
        "configured_scene_bundle_candidate_manifest"]["provider_output_relative_path"]).read_text())
    assert manifest["simulator_run"] is False
    # The configured revision says what the company package must say.
    revision = json.loads(Path(result["configured_scene_revision_path"]).read_text())
    assert revision["replacement"]["physics_authority"] == not_run.PHYSICS_AUTHORITY_NOT_RUN
    # The company package reads stage 5's record through this reference: the published bytes are the record.
    assert revision["replacement"]["native_import_qualification"]["digest"] == _sha(receipt_path)
    _dump(tmp_path, result, revision, receipt_path)


def _dump(tmp_path: Path, result: dict, revision: dict, receipt_path: Path) -> None:
    import os
    root = os.environ.get("WHOLE_CHAIN_DUMP_ROOT")
    if not root:
        return
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lane_result.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    (out / "configured_scene_revision.json").write_text(json.dumps(revision, indent=1, sort_keys=True))
    # The exact bytes the revision's ``replacement.native_import_qualification`` reference names.
    (out / "stage5_not_run_receipt.json").write_bytes(receipt_path.read_bytes())


def test_an_all_host_run_contacts_no_provider_and_resolves_no_network_name(tmp_path, monkeypatch):
    import socket

    from blueprint_pipeline import vast_api_transport

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an all-host run contacted a provider or resolved a network name")

    lane = _host_lane(tmp_path, monkeypatch)
    for module, name in ((socket, "getaddrinfo"), (socket, "create_connection"), (vast_api_transport, "_api_json"),
                         (scene_vast, "arm_independent_vast_watchdog"),
                         (scene_vast, "close_independent_vast_watchdog"),
                         (scene_vast, "close_independent_vast_watchdog_without_allocation")):
        monkeypatch.setattr(module, name, forbidden)
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    assert result["status"] == "completed", result["blockers"]
    watchdog = result["independent_watchdog"]
    assert watchdog["status"] == cpu_prestage.WATCHDOG_NOT_ARMED_STATUS and watchdog["provider_contacted"] is False
    # The deadline is enforced here: the prestage ran under it.
    assert float(lane.seen["BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"]) <= watchdog["watchdog_deadline_epoch"]


def test_a_short_disk_is_a_zero_dollar_refusal_that_capacity_recovery_retries(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_capacity_recovery import (
        nothing_started, preallocation_capacity_failure,
    )
    # Room for the output hold alone, not for the whole chain beside it.
    lane = _host_lane(tmp_path, monkeypatch, free_bytes=8 * GIB + 2 * GIB)

    result = lane.run(cpu_prestage_stage_limit="stage-4")

    assert result["status"] == "blocked"
    assert result["blockers"] == ["scene_configuration_provider_output_disk_budget_exceeded"]
    assert "cpu_prestage_decision" not in result  # never a fallback to a GPU
    assert lane.seen == {}  # no stage ran
    assert [name for name, _ in lane.events if name in {"adapter", "stage", "watchdog_close"}] == []
    assert result["provider_mutations_performed"] == 0
    assert preallocation_capacity_failure(result) and nothing_started(result)
    phases = result["provider_output_disk_capacity"]["sequential_phases"]
    assert [row["phase"] for row in phases] == [admission.CPU_PRESTAGE_PHASE]


def test_any_gpu_stage_keeps_the_rented_path_and_the_host_setting():
    receipt = {"carried_completed_stage_count": 0}
    envelope = {"recipe": {"stage_sequence": _host_stages()}}
    assert cpu_prestage.host_chain_plan(envelope, receipt, "stage-4", diagnostic_only=False) == (True, "stage-6")
    # No host setting at all: still the whole chain here, never a GPU.
    assert cpu_prestage.host_chain_plan(envelope, receipt, None, diagnostic_only=False) == (True, "stage-6")
    for adapter in ("simready_native_import_qualification", "artifixer3d_observed_object_removal", None):
        changed = copy.deepcopy(envelope)
        if adapter is None:  # a host adapter in a class that is not a no-spend or authoring class
            changed["recipe"]["stage_sequence"][3]["execution_class"] = "paid_robot_run"
        else:
            row = next(r for r in changed["recipe"]["stage_sequence"]
                       if r["capability"] == next(i for i in ADMITTED_STAGE_ADAPTER_IDENTITIES
                                                  if i.adapter_id == adapter).capability)
            row["adapter"] = {"id": adapter, "version": "v1"}
        assert cpu_prestage.host_chain_plan(changed, receipt, "stage-4", diagnostic_only=False) == (False, "stage-4")
        assert cpu_prestage.host_chain_plan(changed, receipt, None, diagnostic_only=False) == (False, None)
    assert cpu_prestage.host_chain_plan(envelope, receipt, "stage-4", diagnostic_only=True) == (False, "stage-4")
    carried = {"carried_completed_stage_count": 4}
    assert cpu_prestage.host_chain_plan(envelope, carried, None, diagnostic_only=False) == (False, None)


def test_a_host_stage_six_failure_after_paid_authoring_parks_the_job_after_one_attempt(tmp_path, monkeypatch):
    """S5: the failure names its code, rents nothing, and is none of the classes recovery retries.

    ``observe_failure`` (``task_evaluation_scene_capacity_recovery``) makes a successor attempt only
    for these classes; any other blocked result parks at ``scene_configuration_failed``. So stage 3
    (ARTISAN) is paid once: no automatic second attempt follows.
    """
    from blueprint_pipeline import task_evaluation_authoring_auth_recovery as auth
    from blueprint_pipeline import task_evaluation_scene_capacity_recovery as capacity
    from blueprint_pipeline import task_evaluation_scene_launch_recovery as launch
    from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
        TaskEvaluationSceneConfigurationAdapterError,
    )

    real = builtin_scene_configuration_adapter_handlers

    def failing_assembly():
        handlers = real()

        def fail(**_kwargs):
            raise TaskEvaluationSceneConfigurationAdapterError("assembly_pxr_unavailable")
        return {key: fail if key.adapter_id == "native_task_scene_assembly" else value
                for key, value in handlers.items()}

    monkeypatch.setattr(sys.modules[__name__], "builtin_scene_configuration_adapter_handlers", failing_assembly)
    lane = _host_lane(tmp_path, monkeypatch)
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    assert result["status"] == "blocked"
    assert any(blocker.endswith(":assembly_pxr_unavailable") for blocker in result["blockers"])
    assert [name for name, _ in lane.events if name in {"adapter", "stage"}] == []
    failure = json.loads((lane.job / "cpu_prestage_failure.json").read_text())
    assert failure["gpu_allocation_attempted"] is False and failure["phase"] == "cpu_prestage"
    assert result["provider_mutations_performed"] == 0 and result["continuing_spend_from_this_run"] is False
    assert not capacity.nothing_started(result)  # the paid authoring is on record
    assert not any((capacity.preallocation_capacity_failure(result), capacity.credit_launch_failure(result),
                    auth.initial_authentication_failure(result), capacity.dead_machine_launch_failure(result),
                    launch.transient_launch_failure(result)))


def _not_run_record() -> dict:
    return not_run.not_run_receipt(replacement_identity={"id": "tray", "version": "v1"},
                                   asset_digest="sha256:" + "a" * 64, static_qualification_digest="sha256:" + "b" * 64)


def test_a_refused_output_hold_on_the_host_claims_no_watchdog_fault(tmp_path, monkeypatch):
    """V1: there was no watchdog, so a refused deferred hold says only that the hold was refused."""
    lane = _host_lane(tmp_path, monkeypatch)

    def refused(self):
        self._deferred_hold_refused = True
        self._refuse(admission.BUDGET_EXCEEDED_BLOCKER)
        raise admission.SceneConfigurationOutputHoldRefused(admission.BUDGET_EXCEEDED_BLOCKER)

    monkeypatch.setattr(admission.SceneConfigurationOutputAdmission, "hold_before_allocation", refused)
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    assert result["status"] == "blocked" and "independent_watchdog_not_closed" not in result["blockers"]
    assert result["independent_watchdog"]["status"] == cpu_prestage.WATCHDOG_NOT_ARMED_STATUS


@pytest.mark.parametrize("code", ["cpu_prestage_parent_deadline_exhausted",
                                  "control_plane_disk_budget_exceeded:cpu_prestage:need_bytes=1:available_bytes=0"])
def test_a_zero_dollar_refusal_inside_the_host_chain_is_retried(tmp_path, monkeypatch, code):
    """F2: nothing ran, so the refusal is the $0 transient class capacity recovery retries."""
    from blueprint_pipeline import task_evaluation_scene_capacity_recovery as capacity
    from blueprint_pipeline import task_evaluation_scene_launch_recovery as launch

    lane = _host_lane(tmp_path, monkeypatch)

    def refuse(**_kwargs):
        raise cpu_prestage.CpuPrestageError(code)

    monkeypatch.setattr(cpu_prestage, "prepare_stage_prefix_before_gpu", refuse)
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    assert f"host_chain_failed:CpuPrestageError:{code}" in result["blockers"]
    assert capacity.nothing_started(result) and launch.transient_launch_failure(result)
    teardown = json.loads(Path(result["teardown_manifest_path"]).read_text())
    assert launch.unallocated_teardown(teardown, launch.adapter_result(result))


def test_a_second_job_that_finds_the_workspace_busy_is_retried_and_completes_after_the_first(tmp_path, monkeypatch):
    """F2: every website rigid job holds the one host workspace for all six stages."""
    import fcntl
    import os

    from blueprint_pipeline import task_evaluation_scene_launch_recovery as launch

    first = _host_lane(tmp_path / "first", monkeypatch)
    lock = os.open(first.work / ".cpu-prestage.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock, fcntl.LOCK_EX)  # the first job's chain is running
    try:
        busy = first.run(cpu_prestage_stage_limit="stage-4")
    finally:
        os.close(lock)  # the first job's chain ended
    assert "host_chain_failed:CpuPrestageError:cpu_prestage_workspace_busy" in busy["blockers"]
    assert launch.transient_launch_failure(busy)  # a $0 successor attempt is admitted
    retry = _host_lane(tmp_path / "retry", monkeypatch)
    monkeypatch.setenv(cpu_prestage.WORK_DIR_ENV, str(first.work))  # the same host workspace
    assert retry.run(cpu_prestage_stage_limit="stage-4")["status"] == "completed"


def _website_stage_five(tmp_path: Path, *, static_change: dict | None = None):
    """Stage 4's files and stage 5's real inputs for a rigid website job."""
    from tests.test_website_native_inputs import packet

    envelope, configs = packet(tmp_path / "packet")
    identity = envelope["recipe"]["subject_identity"]
    asset = tmp_path / "stage4" / "replacement_asset.usdz"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"PK authored object bytes")
    static = {"status": "authored_structure_statically_qualified", "authored_structure_statically_qualified": True,
              "replacement_identity": identity, "replacement_usd": {"sha256": _sha(asset)}, **(static_change or {})}
    static_path = asset.parent / "static_qualification_receipt.v1.json"
    static_path.write_text(json.dumps(static))
    rows = [{"role": role, "path": str(path), "digest": _sha(path), "size_bytes": path.stat().st_size}
            for role, path in (("statically_qualified_replacement_asset", asset),
                               ("static_qualification_receipt", static_path))]
    stage = envelope["recipe"]["stage_sequence"][4]
    configuration_path = tmp_path / "stage-5.json"
    configuration_path.write_text(json.dumps(configs["stage-5"]))
    return {"envelope": envelope, "stage": stage, "configuration": configs["stage-5"],
            "configuration_path": configuration_path, "dependency_results": ({"output_artifacts": rows},)}


@pytest.mark.parametrize(("change", "code"), [
    ({"static": {"replacement_usd": {"sha256": "sha256:" + "0" * 64}}}, "static_binding_invalid"),
    ({"not_website": True}, "configuration_invalid:website_job_required"),
], ids=["static_sha", "not_a_website_job"])
def test_stage_five_refuses_what_is_not_a_rigid_website_object(tmp_path, change, code):
    from blueprint_pipeline.task_evaluation_scene_configuration_adapters import (
        TaskEvaluationSceneConfigurationAdapterError,
    )
    inputs = _website_stage_five(tmp_path, static_change=change.get("static"))
    if change.get("not_website"):
        inputs["envelope"]["request"]["scene"].pop("website_native_inputs")
    with pytest.raises(TaskEvaluationSceneConfigurationAdapterError, match=code):
        website_native_background.execute_simulator_not_run(**inputs, output_root=tmp_path / "out")


def _tamper(paths: dict, case) -> None:
    record_path = paths["native_import_qualification_receipt"]
    if isinstance(case, bytes):
        record_path.write_bytes(case)
    elif case == "replacement_bytes":
        paths["native_qualified_replacement_asset"].write_bytes(b"other bytes")
    elif isinstance(case, dict):
        change = dict(case)
        reseal = change.pop("_reseal", True)
        record = {**json.loads(record_path.read_text()), **change}
        if reseal:
            record["result_digest"] = canonical_digest(record, digest_field="result_digest")
        record_path.write_text(json.dumps(record))


INVALID = "simulator_not_run_record_invalid"


@pytest.mark.parametrize(("case", "outcome"), [
    (None, not_run.PHYSICS_AUTHORITY_NOT_RUN),
    # A record that says what is not true (every reader refuses it).
    ({"native_isaac_executed": True}, INVALID), ({"native_simulator_import_qualified": True}, INVALID),
    ({"gpu_execution_performed": True}, INVALID), ({"asset_is_stage_four_bytes": False}, INVALID),
    ({"simulator": "isaac-sim-5.0"}, INVALID), ({"support_contact_observed": True}, INVALID),
    ({"physics_authority": "qualified_replacement_asset"}, INVALID), ({"blockers": ["x"]}, INVALID),
    ({"reason": "edited_after_sealing", "_reseal": False}, INVALID),  # only the record's own seal is broken
    # A record that does not name the files publication holds.
    ({"asset_digest": "sha256:" + "0" * 64}, "simulator_not_run_asset_digest_mismatch"),
    ({"static_qualification_digest": "sha256:" + "0" * 64}, "simulator_not_run_static_qualification_digest_mismatch"),
    ("replacement_bytes", "simulator_not_run_asset_not_stage_four_bytes"),
    # What publication cannot read is never labelled qualified; an Isaac receipt is, as stage 5 checked it.
    (b"{not json", "stage_five_receipt_unreadable"), (b"[]", "stage_five_receipt_schema_unknown"),
    (json.dumps({"schema_version": "some_other_receipt.v1"}).encode(), "stage_five_receipt_schema_unknown"),
    (json.dumps({"schema_version": not_run.QUALIFIED_RESULT_SCHEMA}).encode(), "qualified_replacement_asset"),
], ids=str)
def test_publication_publishes_only_a_true_stage_five_record_bound_to_stage_four(tmp_path, case, outcome):
    from blueprint_pipeline import task_evaluation_scene_configuration_publication as publication

    inputs = _website_stage_five(tmp_path)
    five = website_native_background.execute_simulator_not_run(**inputs, output_root=tmp_path / "five")
    paths = {row["role"]: Path(row["path"]) for result in (*inputs["dependency_results"], five)
             for row in result["output_artifacts"]}
    _tamper(paths, case)
    if outcome in {not_run.PHYSICS_AUTHORITY_NOT_RUN, "qualified_replacement_asset"}:
        assert publication._physics_authority(paths) == outcome
    else:
        with pytest.raises(publication.TaskEvaluationSceneConfigurationPublicationError, match=outcome):
            publication._physics_authority(paths)
    if outcome == INVALID:  # stage 6 reads the record too, and refuses it
        with pytest.raises(not_run.SimulatorNotRunError, match=INVALID):
            not_run.is_not_run_record(paths["native_import_qualification_receipt"])


def test_the_real_website_recipe_runs_whole_on_the_host_and_passes_the_real_preflight(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import (
        TaskEvaluationSceneConfigurationStageConfigurationError, validate_immutable_stage_configurations,
    )
    from tests.test_website_native_inputs import packet

    envelope, configs = packet(tmp_path)
    stage = envelope["recipe"]["stage_sequence"][4]
    assert stage["adapter"] == {"id": not_run.ADAPTER_ID, "version": "v1"} and stage["execution_class"] == "no_spend"
    assert configs["stage-5"]["schema_version"] == not_run.CONFIGURATION_SCHEMA
    validate_immutable_stage_configurations(envelope=envelope, configurations=configs)
    assert [row["adapter"]["id"] for row in envelope["recipe"]["stage_sequence"]] == list(HOST_ADAPTERS)
    assert cpu_prestage.host_chain_plan(envelope, {"carried_completed_stage_count": 0}, "stage-4",
                                        diagnostic_only=False) == (True, "stage-6")
    # Refused: a configuration that asks for a simulator, a supplemental destination, or a job that is
    # not a website job (the reviewer's probe C, inverted).
    with_destination = copy.deepcopy(envelope)
    with_destination["recipe"]["supplemental_destination"] = {"identity": {"id": "tray", "version": "v1"}}
    not_website = copy.deepcopy(envelope)
    not_website["request"]["scene"].pop("website_native_inputs")
    for changed, configurations, code in (
            (envelope, {**configs, "stage-5": {**configs["stage-5"], "simulator_run_requested": True}}, "stage-5"),
            (envelope, {**configs, "stage-5": {**configs["stage-5"], "reason": "other"}}, "stage-5"),
            (with_destination, configs, "supplemental_destination_requires_native_import"),
            (not_website, configs, "website_job_required")):
        with pytest.raises(TaskEvaluationSceneConfigurationStageConfigurationError, match=code):
            validate_immutable_stage_configurations(envelope=changed, configurations=configurations)


def test_an_articulated_website_job_records_that_no_simulator_ran_and_rents_nothing():
    """Its joint is the authored file's (the stage-4 record); stage 5 runs on the host, no Isaac, no GPU."""
    stages = records.stage_sequence()
    identity = {"id": "door", "version": "v1"}
    configurations = [{} for _ in range(4)] + [records.stage_five_configuration(
        replacement_identity=identity, articulated=True), {}]
    website_native_background.select_stage_five(stages, configurations)
    assert stages[4]["adapter"]["id"] == not_run.ADAPTER_ID and stages[4]["execution_class"] == "no_spend"
    assert configurations[4] == not_run.stage_five_configuration(replacement_identity=identity)


def _not_run_case(tmp_path: Path, **change):
    """The rigid adapter fixture's revision with stage 5 replaced by this chain's record."""
    from tests.test_task_evaluation_rigid_relocation_native_adapter import NATIVE_IMPORT, _case, _rewrite

    launch, configured, references, docs = _case(tmp_path)
    record = not_run.not_run_receipt(replacement_identity=docs[NATIVE_IMPORT]["replacement_identity"],
                                     asset_digest=configured["replacement"]["asset"]["digest"],
                                     static_qualification_digest="sha256:" + "b" * 64)
    record.update(change)
    record["result_digest"] = canonical_digest(record, digest_field="result_digest")
    _rewrite(tmp_path=tmp_path, configured=configured, references=references, contract_path=NATIVE_IMPORT,
             document=record)
    launch["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    return launch, configured, references


@pytest.mark.parametrize(("accept", "change", "code"), [
    (False, {}, "source_contract_invalid:scene.configured_revision.replacement.native_import_qualification"),
    (True, {}, None),
    # The reviewer's probe A, inverted: each is refused even when the caller opts in.
    (True, {"asset_digest": "sha256:" + "0" * 64}, "simulator_not_run_asset_unbound"),
    (True, {"asset_is_stage_four_bytes": False}, "executable_geometry_missing"),
    (True, {"status": "qualified", "native_simulator_import_qualified": True}, "executable_geometry_missing"),
], ids=["default_refuses", "opted_in", "asset_digest_names_no_real_file", "not_stage_four_bytes",
        "relabelled_qualified"])
def test_the_rigid_adapter_reads_the_not_run_record_only_for_a_caller_that_opts_in(tmp_path, accept, change, code):
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
        TaskEvaluationRigidRelocationNativeAdapterError, adapt_rigid_relocation_task_template,
    )
    launch, configured, references = _not_run_case(tmp_path, **change)
    call = functools.partial(adapt_rigid_relocation_task_template, request=launch, configured_revision=configured,
                             materialized_references=references, accept_simulator_not_run=accept)
    if code is None:
        assert call()["native_task_kind"] == "rigid_pick_place"
    else:
        with pytest.raises(TaskEvaluationRigidRelocationNativeAdapterError, match=code):
            call()


def _episode_compiler(tmp_path, monkeypatch):
    from tests import test_task_evaluation_native_arena_episode_compiler as compiler_tests

    real = compiler_tests._configured_runtime_documents

    def not_run_documents(configured):
        documents = real(configured)
        documents["scene.configured_revision.replacement.native_import_qualification"] = not_run.not_run_receipt(
            replacement_identity=configured["replacement"]["identity"], asset_digest="sha256:" + "a" * 64,
            static_qualification_digest="sha256:" + "b" * 64)
        return documents

    monkeypatch.setattr(compiler_tests, "_configured_runtime_documents", not_run_documents)
    compiler_tests.test_closed_compiler_joins_revision_and_robot_team_inputs(tmp_path, monkeypatch, False, False, False)


def _deferred_controls(tmp_path, _monkeypatch):
    from blueprint_pipeline import task_evaluation_configured_controls_deferred_inputs as deferred

    _launch, configured, references = _not_run_case(tmp_path)
    deferred.derive_native_trajectory_plan(revision=configured, documents={
        path: Path(row["materialized_path"]) for path, row in references.items()})


def _robot_free_readiness(tmp_path, _monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid

    _launch, configured, references = _not_run_case(tmp_path)
    rigid.rigid_task_for_revision(revision=configured, documents={
        path: Path(row["materialized_path"]) for path, row in references.items()})


def _articulated_not_run_case(tmp_path: Path, **change):
    """The articulated adapter fixture's revision with stage 5 replaced by the website's record."""
    from tests.test_task_evaluation_articulated_open_close_native_adapter import _case
    from tests.test_task_evaluation_rigid_relocation_native_adapter import NATIVE_IMPORT, _rewrite

    case = _case(tmp_path / "case")
    configured, references = case["configured"], case["references"]
    record = not_run.not_run_receipt(replacement_identity=configured["replacement"]["identity"],
                                     asset_digest=configured["replacement"]["asset"]["digest"],
                                     static_qualification_digest="sha256:" + "b" * 64)
    record.update(change)
    record["result_digest"] = canonical_digest(record, digest_field="result_digest")
    (tmp_path / "not-run").mkdir()
    _rewrite(tmp_path=tmp_path / "not-run", configured=configured, references=references,
             contract_path=NATIVE_IMPORT, document=record)
    return configured, references


def _articulated_adapter(tmp_path, _monkeypatch):
    """The articulated adapter as the episode compiler, configured controls and readiness call it."""
    from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
        adapt_articulated_open_close_task_template,
    )
    configured, references = _articulated_not_run_case(tmp_path)
    adapt_articulated_open_close_task_template(configured_revision=configured, materialized_references=references)


@pytest.mark.parametrize(("change", "code"), [
    ({}, None),
    ({"asset_digest": "sha256:" + "0" * 64}, "simulator_not_run_asset_unbound"),
    ({"native_isaac_executed": True}, "executable_geometry_missing"),
], ids=["opted_in", "asset_digest_names_no_real_file", "claims_isaac"])
def test_the_articulated_adapter_reads_the_not_run_record_only_for_a_caller_that_opts_in(tmp_path, change, code):
    from blueprint_pipeline.task_evaluation_articulated_open_close_native_adapter import (
        adapt_articulated_open_close_task_template,
    )
    configured, references = _articulated_not_run_case(tmp_path, **change)
    call = functools.partial(adapt_articulated_open_close_task_template, configured_revision=configured,
                             materialized_references=references, accept_simulator_not_run=True)
    if code is None:  # the joint is the authored one, read from the stage-4 record
        assert call()["native_task_definition"]["task_joint_bindings"]
    else:
        with pytest.raises(ValueError, match=code):
            call()


def _construction_probe(tmp_path, _monkeypatch):
    """``rigid_robot_free_construction_probe`` hands its own documents to the shared derivation."""
    from blueprint_pipeline import task_evaluation_rigid_relocation_native_adapter as rigid

    _launch, configured, references = _not_run_case(tmp_path)
    documents = {path: json.loads(Path(row["materialized_path"]).read_text()) for path, row in references.items()}
    template = documents[rigid.DEFINITION_CONTRACT_PATH]
    rigid._adapt_rigid_relocation_documents(
        documents=documents, bindings=[], replacement_identity=configured["replacement"]["identity"],
        task={"identity": template["task_identity"], "subject": {"identity": template["object_identity"]},
              "strategy": template["strategy"]},
        authority_digest=configured["revision_digest"], diagnostic=False, configured_revision=configured)


@pytest.mark.parametrize(("caller", "code"), [
    (_episode_compiler, "source_contract_invalid:scene.configured_revision.replacement"),
    (_deferred_controls, "source_contract_invalid:scene.configured_revision.replacement"),
    (_robot_free_readiness, "source_contract_invalid:scene.configured_revision.replacement"),
    (_construction_probe, "executable_geometry_missing"),
    (_articulated_adapter, "source_contract_invalid:scene.configured_revision.replacement"),
], ids=["paid_episode_compiler", "configured_controls_inputs", "robot_free_readiness", "construction_probe",
        "articulated_paid_callers"])
def test_every_paid_or_scoring_caller_refuses_a_not_run_scene(tmp_path, monkeypatch, caller, code):
    with pytest.raises(Exception, match=code):
        caller(tmp_path, monkeypatch)


@pytest.mark.parametrize(("change", "code"), [
    ({"status": "completed_prefix"}, "whole_chain_not_completed:completed_prefix:stage-6:no_blocker"),
    ({"status": "blocked", "blockers": ["scene_configuration_provider_failed:assembly_pxr_unavailable"]},
     "whole_chain_not_completed:blocked:stage-6:scene_configuration_provider_failed:assembly_pxr_unavailable"),
    ({"chain": {"whole_run_completed": False}}, "whole_chain_results_invalid"),
    ({"chain": {"stage_limit": "stage-4"}}, "whole_chain_results_invalid"),
    ({"chain": {"stage_results": [{"stage_id": f"stage-{i}"} for i in range(1, 5)]}}, "whole_chain_results_invalid"),
    ({"returncode": 2}, "whole_chain_not_completed"),
])
def test_the_host_never_takes_a_prefix_or_a_failed_chain_for_a_whole_one(tmp_path, change, code):
    for index in range(1, 7):
        (tmp_path / "stages" / f"stage-{index}").mkdir(parents=True)
    bundle = {"run_id": "run", "source_commit": "c" * 40}
    chain = {"status": "completed", "whole_run_completed": True, "stage_limit": None, "stage_count": 6,
             "stage_results": [{"stage_id": f"stage-{i}"} for i in range(1, 7)]}
    result = {"status": "completed", "run_id": "run", "source_commit": "c" * 40, "stage_chain": chain}
    check = functools.partial(cpu_prestage._completed_whole_chain, stage_limit="stage-6", bundle_receipt=bundle,
                              output=tmp_path)
    assert check(result, returncode=0) == [f"stage-{i}" for i in range(1, 7)]
    changed = copy.deepcopy(result)
    changed["stage_chain"].update(change.get("chain", {}))
    changed.update({key: change[key] for key in ("status", "blockers") if key in change})
    if "chain" in change and "stage_results" in change["chain"]:
        changed["stage_chain"]["stage_count"] = 4
    with pytest.raises(cpu_prestage.CpuPrestageError, match=code):
        check(changed, returncode=change.get("returncode", 0))


def _not_run_published_case(tmp_path: Path) -> dict:
    """The rigid fixture's published revision as this chain publishes it: stage 5 is the not-run record."""
    from tests.test_robot_free_readiness_retention import _published
    from tests.test_task_evaluation_rigid_relocation_native_adapter import NATIVE_IMPORT, _rewrite

    tmp_path.mkdir(parents=True)
    case = _published(tmp_path)
    configured = case["configured"]
    record = not_run.not_run_receipt(replacement_identity=case["docs"][NATIVE_IMPORT]["replacement_identity"],
                                     asset_digest=configured["replacement"]["asset"]["digest"],
                                     static_qualification_digest="sha256:" + "b" * 64)
    (tmp_path / "rewrites").mkdir()
    _rewrite(tmp_path=tmp_path / "rewrites", configured=configured, references=case["references"],
             contract_path=NATIVE_IMPORT, document=record)
    configured["replacement"]["physics_authority"] = not_run.PHYSICS_AUTHORITY_NOT_RUN
    configured["revision_digest"] = canonical_digest(configured, digest_field="revision_digest")
    case["store"] = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in case["references"].values()}
    return case


def test_a_not_run_revision_is_decided_not_applicable_and_its_team_half_is_still_enqueued(tmp_path, monkeypatch):
    """M2: readiness says ``simulator_not_run`` (never ``probe_unavailable``), keeps no receipt, and
    preparation completion still queues the half (only a decided ``failed`` holds it)."""
    from blueprint_pipeline import robot_free_readiness_retention as retention
    from blueprint_pipeline import team_scene_packet_worker as worker
    from blueprint_pipeline.task_evaluation_scene_preparation_completion import reconcile_preparation_completion
    from tests.test_robot_free_readiness_retention import _retain
    from tests.test_scene_preparation_completion import _prepare, _website_request

    scored = _retain(_not_run_published_case(tmp_path / "published"), tmp_path / "scored-launch")
    assert scored == {"status": "simulator_not_run", "reason": not_run.PHYSICS_AUTHORITY_NOT_RUN, "retained": False}
    assert not (tmp_path / "scored-launch" / "robot_free_readiness.v1.json").exists()
    args, _run, _result_path = _prepare(tmp_path / "completion", monkeypatch, _website_request())
    # The completion harness publishes placeholder references; score the not-run revision instead.
    monkeypatch.setattr(retention, "retain_robot_free_readiness", lambda **_kwargs: dict(scored))
    queue = tmp_path / "team-scene-packets"
    monkeypatch.setenv(worker.QUEUE_ROOT_ENV, str(queue))
    result = reconcile_preparation_completion(**args)
    assert (result["status"], result["phase"]) == ("completed", "scene_prepared")
    completion = result["state"]["scene_preparation_completion"]
    assert completion["team_scene_request"]["status"] == "enqueued"
    [pending] = (queue / "pending").glob("*.json")
    assert worker.validated_request(json.loads(pending.read_text()))["robot_free_readiness_status"] == "simulator_not_run"
    assert not list(queue.glob("holds/*.json"))


def test_a_packet_whose_stage_five_ran_no_simulator_is_never_queued_for_a_paid_probe(tmp_path, monkeypatch):
    """M2: the one state every probe queue site reads says ``simulator_not_run``, so nothing is queued or paid."""
    from blueprint_pipeline.rigid_robot_free_probe_bundle import current_probe_state
    from blueprint_pipeline.rigid_robot_free_probe_runner import enqueue_rigid_robot_free_probe
    from tests.test_rigid_robot_free_probe_lane import build_world

    world = build_world(tmp_path, monkeypatch)
    state = current_probe_state(world["capture"])
    assert state["status"] == "probe_required"
    request_path = Path(state["facts"]["packet_dir"]) / "native_task_arena_packet_request.v1.json"
    frozen = json.loads(request_path.read_text()) if request_path.is_file() else {}
    adapter = frozen.setdefault("configured_task_template_adapter", {})
    adapter.setdefault("source_documents", {}).setdefault("documents", {})["native_import_qualification"] = (
        _not_run_record())
    request_path.write_text(json.dumps(frozen))
    assert current_probe_state(world["capture"])["status"] == "simulator_not_run"
    queued = enqueue_rigid_robot_free_probe(world["capture"], queue_root=world["queue"])
    assert (queued["status"], queued["queued"]) == ("simulator_not_run", False)
    assert world["allocator_calls"] == []
