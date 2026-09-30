# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_episode_compilation_collector.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_remote.py
#   src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py
#   tests/remote_episode_compilation_support.py
"""ADP-009D/day-28, plan 14 PR 4: the paid unit collects a remote compile and lands only what consumers read."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_job_lease as leases
from blueprint_pipeline import remote_cpu_job_records as records
from blueprint_pipeline import task_evaluation_episode_compilation_collector as collector
from blueprint_pipeline import task_evaluation_episode_compilation_remote as remote
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.remote_cpu_allocator_fakes import B2_BUCKET
from tests.remote_cpu_fakes import FakeGcsError
from tests.remote_episode_compilation_support import (
    HOST_RECORD,
    OUTPUTS,
    CollectorWorld,
    Crash,
    Host,
    digest_of,
    tree_snapshot,
)

# Write URLs outlive the hard deadline (dispatch + 600 + 1800 + 120 s) by the recorded margin: provider-zero
# is provable only after them.
PAST_WRITE_EXPIRY = 600 + 1800 + 120 + 300 + 60


def _done(world: CollectorWorld) -> bool:
    return world.terminal() and not remote.marker_path(world.host.jobs, "authoritative", world.name).exists()


def _complete(world: CollectorWorld, **drive) -> None:
    world.drive(until=lambda: _done(world), **drive)


def _files(root: Path) -> dict[str, int]:
    return {path.relative_to(root).as_posix(): path.stat().st_size for path in sorted(root.rglob("*"))
            if path.is_file()} if root.exists() else {}


def _attempts(world: CollectorWorld) -> list[dict]:
    lease = world.lease()
    return [*lease["prior_attempts"], lease] if lease else []


def _staging_versions(world: CollectorWorld) -> list[str]:
    listed = world.store.list_object_versions(Bucket=B2_BUCKET, Prefix="")
    return [row["Key"] for row in [*listed["Versions"], *listed["DeleteMarkers"]] if "/remote-cpu/staging/" in row["Key"]]


def _assert_torn_down(world: CollectorWorld) -> None:
    """Every started attempt has a sealed provider-zero teardown, and nothing of it remains anywhere."""

    started = [attempt for attempt in _attempts(world) if attempt["dispatch_started"]]
    assert started
    for attempt in started:
        teardown = records.validate_teardown(json.loads(
            (world.host.jobs / "teardowns" / f"{attempt['attempt_id']}.json").read_text(encoding="utf-8")))
        assert teardown["compute_zero_proven"] and teardown["provider_zero_proven"], attempt["attempt_id"]
        assert attempt["provider_zero_proven"] and attempt["teardown_digest"] == teardown["teardown_digest"]
    assert leases.slots_in_use(world.host.jobs) == 0
    assert _staging_versions(world) == [] and world.bucket._objects == {}
    assert all(view["completionTime"] for view in world.executions())
    settled = world.tmp_path / "remote" / "spend-authority" / "remote-cpu-settled"
    assert len(list(settled.glob("*.json"))) == len(started)


def test_remote_stage_writes_only_the_consumer_subset_on_the_host(tmp_path: Path, monkeypatch) -> None:
    world = CollectorWorld(tmp_path, monkeypatch, case=(False, True, True))
    compiled = world.host.outputs
    before = _files(world.host.fs)
    _complete(world)
    assert world.lease()["state"] == "completed" and world.row_state() == "completed"
    compilation = world.plan.compilation_id
    after = _files(world.host.fs)
    added = {path: size for path, size in after.items() if before.get(path) != size}
    landed = {path.removeprefix(f"{OUTPUTS.lstrip('/')}/{compilation}/"): size for path, size in added.items()
              if path.startswith(f"{OUTPUTS.lstrip('/')}/{compilation}/")}
    # Only what launch activation and the canary hand-off read: the adapter tree and the probe request.
    assert landed and all(path.startswith("native-arena-adapter/")
                          or path == "rigid_destination_native_probe_request.v1.json" for path in landed)
    assert "rigid_destination_native_probe_request.v1.json" in landed
    assert not any(path.startswith(("configured-scene/", "native-task-packet/", "native-appearance/", "task-destination/"))
                   or path == "native-task-arena-bundle.zip" for path in landed)
    assert sorted(path.name for path in compiled.iterdir()) == sorted([compilation, f"{compilation}.remote-output.v1.json"])
    # The worker compiled the whole tree; the host holds its consumer subset plus a little metadata.
    worker_tree = next((tmp_path / "workers").iterdir()) / OUTPUTS.lstrip("/") / compilation
    subset = sum(size for path, size in _files(worker_tree).items() if path in landed)
    assert sum(landed.values()) == subset < sum(_files(worker_tree).values())
    metadata = sum(size for path, size in added.items() if not path.startswith(f"{OUTPUTS.lstrip('/')}/{compilation}/"))
    assert metadata <= 1024 * 1024
    pointer = json.loads((compiled / f"{compilation}.remote-output.v1.json").read_text(encoding="utf-8"))
    assert pointer["landed"] == {"subset": "episode_compilation_consumer.v1", "paths": len(landed),
                                 "bytes": sum(landed.values())}
    assert pointer["provider_zero_proven"] and pointer["state"] == "landed"
    assert (compiled / f"{compilation}.remote-output.v1.json").stat().st_mode & 0o777 == 0o440
    # It lists what the result names that stayed remote, the packet, so the owner census can let it stand for it.
    result = json.loads((world.host.queue / "results" / world.name).read_text(encoding="utf-8"))
    packet = Path(result["compiled_episode_packet_path"]).resolve().relative_to((compiled / compilation).resolve())
    assert pointer["raw_references"] == [{"path": f"{OUTPUTS}/{compilation}/{packet.as_posix()}",
                                          "digest": result["compiled_episode_packet_digest"],
                                          "size_bytes": result["compiled_episode_packet_size_bytes"]}]
    assert packet.as_posix() not in landed and not (compiled / compilation / packet).exists()


def test_landed_subset_passes_launch_activation_and_canary_handoff(tmp_path: Path, monkeypatch) -> None:
    """``tests/test_task_evaluation_launch_activation_worker.py``'s production-compiler case, compiled remotely:
    activation and the canary hand-off read the landed adapter tree, packet root and runtime receipt."""

    from blueprint_pipeline import task_evaluation_launch_activation_worker as activation
    from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
    from blueprint_pipeline.task_evaluation_policy_canary_handoff import _compiled_construction
    from tests.test_task_evaluation_launch_activation_worker import _stage_verified_preparation

    preparation, _, _, queue, input_root = _stage_verified_preparation(tmp_path / "preparation")
    prepared = input_root / preparation["preparation_id"]
    host = Host(tmp_path)
    envelope = {"schema_version": "task_evaluation_episode_compilation_envelope.v1",
                "compilation_id": preparation["preparation_id"], "preparation_id": preparation["preparation_id"],
                "run_id": preparation["run_id"], "team_namespace": preparation["team_namespace"],
                "expected_production_commit": preparation["expected_production_commit"],
                "configured_scene_revision_digest": preparation["task"]["configured_scene_revision_digest"],
                "request": preparation, "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    name = f"{envelope['compilation_id']}-{envelope['envelope_digest'][7:]}.json"
    row = host.queue / "processing" / name
    row.write_text(json.dumps(envelope), encoding="utf-8")
    reference = host.inputs / preparation["preparation_id"] / "robot.json"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b'{"robot": "franka"}\n')
    compilation = envelope["compilation_id"]
    plan = remote.RemotePlan(
        queue_row={"queue": remote.QUEUE, "name": name, "envelope_digest": envelope["envelope_digest"]},
        compilation_id=compilation, source_commit=preparation["expected_production_commit"],
        image="gcr.io/blueprint-8c1ca/blueprint-pipeline@sha256:" + "d" * 64,
        environment_digest=HOST_RECORD["environment_digest"], host_environment_digest=HOST_RECORD["environment_digest"],
        closure={"class": "not_applicable", "source_appearance_digest": None}, environment={},
        inputs=tuple(remote._input(role, contract_path, path, host.fs) for role, contract_path, path in (
            ("queue_envelope", "queue_envelope", row), ("materialized_reference", "robot.configuration", reference))),
        output_root=f"{OUTPUTS}/{compilation}", declared_scratch=(f"{OUTPUTS}/content-addressed/",),
        allowed_cpu_classes=(), ephemeral_bytes_required=0)
    world = CollectorWorld(tmp_path, monkeypatch, plan=plan, host=host)

    def fabricated(descriptor: dict, roots) -> dict:
        """What test :685 copies into place, compiled in the worker: the adapter tree, bound to this output."""

        output = roots.local(descriptor["outputs"]["output_root"])
        adapter_root = output / "native-arena-adapter"
        shutil.copytree(prepared / "native-arena-adapter", adapter_root)
        for document in ("native_task_arena_scene_plan.v1.json", "native_task_arena_packet_receipt.v1.json"):
            (adapter_root / "construction-packet" / document).write_text("{}\n", encoding="utf-8")
        adapter_path = adapter_root / "task_evaluation_native_arena_adapter_result.v1.json"
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter.update(packet_root=str(adapter_root / "construction-packet"), runtime_source_receipt=str(
            adapter_root / "runtime-source" / "native_task_runtime_source_packet.v1.json"))
        adapter["result_digest"] = canonical_digest(adapter, digest_field="result_digest")
        adapter_path.chmod(0o640)
        adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
        (output / "configured-scene").mkdir()
        (output / "configured-scene" / "appearance.usdc").write_bytes(b"appearance" * 1000)
        packet = output / "native-task-arena-bundle.zip"
        packet.write_bytes(b"production-owned-episode-packet" * 100)
        result = {"schema_version": "task_evaluation_episode_compilation_result.v1",
                  "status": "compiled_for_production_launch", "compilation_id": compilation,
                  "run_id": envelope["run_id"], "team_namespace": envelope["team_namespace"],
                  "source_commit": descriptor["code"]["source_commit"],
                  "configured_scene_revision_digest": envelope["configured_scene_revision_digest"],
                  "compiled_episode_packet_digest": digest_of(packet.read_bytes()),
                  "compiled_episode_packet_size_bytes": packet.stat().st_size, "compiled_episode_packet_path": str(packet),
                  "adapter_result_path": str(adapter_path), "adapter_result_digest": adapter["result_digest"],
                  "compiler_output_digest": "sha256:" + "c" * 64, "customer_supplied_prebuilt_episode_packet": False,
                  "compiled_by_production": True, "provider_mutation_performed": False,
                  "paid_execution_requested": False, "automatic_progression_required": True, "blockers": [],
                  "result_digest": ""}
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        return {"result": result, "release_path_misses": [], "failures": []}

    world.stage_override = fabricated
    _complete(world)
    assert world.lease()["state"] == "completed" and world.row_state() == "completed"
    assert sorted(path.name for path in (host.outputs / compilation).iterdir()) == ["native-arena-adapter"]

    # Test :685's activation request, pointed at the remotely compiled result.
    result_path = next((queue / "results").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(status="queued_for_production_episode_compilation", episode_compilation_id=compilation,
                  episode_compilation_queue_envelope_digest=envelope["envelope_digest"])
    result.pop("adapter_result_digest")
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.chmod(0o640)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    request = {"preparation": {"preparation_id": preparation["preparation_id"],
                               "request_digest": launch_preparation_request_digest(preparation),
                               "result_digest": result["result_digest"]},
               "team_namespace": preparation["team_namespace"],
               "expected_production_commit": preparation["expected_production_commit"]}
    loaded_request, _, loaded_adapter, _ = activation._load_verified_preparation(
        activation_request=request, preparation_queue_root=queue, preparation_input_root=input_root,
        episode_compilation_queue_root=host.queue, episode_compilation_output_root=host.outputs)
    assert loaded_request["preparation_id"] == compilation
    assert Path(loaded_adapter["packet_root"]).is_dir()
    construction = _compiled_construction(host.queue, preparation_id=compilation,
                                          expected_production_commit=preparation["expected_production_commit"])
    assert construction["scene_plan_path"].is_file() and construction["runtime_source_receipt_path"].is_file()


def test_row_never_moves_before_compute_zero_is_proven(tmp_path: Path, monkeypatch) -> None:
    world = CollectorWorld(tmp_path, monkeypatch)
    steps: list[str] = []
    monkeypatch.setattr(collector, "_after_step", lambda step, attempt_id: steps.append(step))
    refusals = {"left": 4}
    delete = world.bucket.delete

    def unavailable(name, *, generation=None):  # the transport cannot yet be proven gone
        if refusals["left"]:
            refusals["left"] -= 1
            raise FakeGcsError(503, "backendError")
        return delete(name, generation=generation)

    monkeypatch.setattr(world.bucket, "delete", unavailable)
    world.drive(until=lambda: (world.lease() or {}).get("state") == "collecting", step=30)
    for _ in range(3):
        world.collect()
        # The execution has finished and its receipt is up, but its compute-zero is unproven: nothing moves.
        assert world.row_state() == "processing" and steps == []
        assert not (world.host.queue / "results" / world.name).exists()
        assert not (world.host.outputs / world.plan.compilation_id).exists()
        assert not world.lease()["compute_zero_proven"]
    _complete(world)
    # Plan 14 §9's commit order, compute-zero first; a step repeated on a later run is idempotent.
    assert list(dict.fromkeys(steps)) == ["compute_zero", "promotion", "validation", "landing", "pointer", "pin",
                                          "result", "move", "provider_zero"]
    assert world.row_state() == "completed"


def test_second_attempt_and_fallback_wait_for_compute_zero(tmp_path: Path, monkeypatch) -> None:
    world = CollectorWorld(tmp_path, monkeypatch)
    world.remote.jobs.script("crash", "crash")
    delete, refused = world.bucket.delete, {"on": False}

    def gated(name, *, generation=None):
        if refused["on"]:
            raise FakeGcsError(503, "backendError")
        return delete(name, generation=generation)

    monkeypatch.setattr(world.bucket, "delete", gated)
    # Attempt 1 crashes without a receipt: an infrastructure failure, so the lease expires.
    world.drive(until=lambda: (world.lease() or {}).get("state") == "expired", step=30)
    refused["on"] = True
    for _ in range(3):
        world.advance(60)
        world.collect()
        lease = world.lease()
        assert (lease["attempt"], lease["compute_zero_proven"]) == (1, False)
        assert len(world.executions()) == 1  # no second attempt before the first is compute-zero
    refused["on"] = False
    world.drive(until=lambda: world.lease()["attempt"] == 2 and world.lease()["state"] == "expired", step=30)
    assert world.lease()["prior_attempts"][0]["compute_zero_proven"]
    refused["on"] = True
    for _ in range(3):
        world.advance(60)
        world.collect()
        assert not remote.marker_path(world.host.jobs, "fallback", world.name).exists()
    refused["on"] = False
    world.drive(until=lambda: remote.marker_path(world.host.jobs, "fallback", world.name).exists(), step=30)
    assert all(attempt["compute_zero_proven"] for attempt in _attempts(world))
    assert world.row_state() == "processing" and len(world.executions()) == 2
    fallback = remote.read_marker(remote.marker_path(world.host.jobs, "fallback", world.name))
    assert (fallback["reason"], fallback["attempts"]) == ("remote_cpu_receipt_missing", 2)
    _complete(world)
    assert world.lease()["state"] == "fallback_host"
    _assert_torn_down(world)


@pytest.mark.parametrize("scenario", ["success", "blocked", "crash", "timeout", "stale_heartbeat", "lost_response",
                                      "promotion_failure"])
def test_every_attempt_ends_with_a_provider_zero_teardown_receipt(tmp_path: Path, monkeypatch, scenario: str) -> None:
    world = CollectorWorld(tmp_path, monkeypatch)
    expected = {"success": ("completed", "completed"), "blocked": ("blocked", "blocked"),
                "lost_response": ("completed", "completed")}.get(scenario, ("fallback_host", "processing"))
    if scenario == "blocked":
        from tests.remote_cpu_worker_stages import install_compile_stand_ins

        def refuses(**_kwargs):
            from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
                TaskEvaluationNativeArenaEpisodeCompilerError,
            )

            raise TaskEvaluationNativeArenaEpisodeCompilerError("episode_compiler_destination_usd_format_unrecognized")

        assert install_compile_stand_ins() is not None
        world.compiler = refuses
    behaviours = {"crash": ("crash", "crash"), "timeout": ("timeout", "timeout"), "stale_heartbeat": ("hang", "hang"),
                  "lost_response": ("lost_response",)}
    world.remote.jobs.script(*behaviours.get(scenario, ()))
    if scenario == "promotion_failure":
        def fails(**_kwargs):
            raise FakeGcsError(500, "InternalError")

        monkeypatch.setattr(world.store, "copy_object", fails)
    _complete(world, step=300)
    assert (world.lease()["state"], world.row_state()) == expected, world.results[-1]
    _assert_torn_down(world)
    if expected[0] == "fallback_host":
        assert len([attempt for attempt in _attempts(world) if attempt["dispatch_started"]]) == 2
        assert remote.marker_path(world.host.jobs, "fallback", world.name).exists()
    if scenario == "blocked":
        result = json.loads((world.host.queue / "results" / world.name).read_text(encoding="utf-8"))
        assert result["blockers"] == ["episode_compiler_destination_usd_format_unrecognized"]
        assert not (world.host.outputs / world.plan.compilation_id).exists()
    if scenario == "stale_heartbeat":
        assert {attempt["outcome"] for attempt in _attempts(world)} <= {"heartbeat_stale", "start_timeout"}
        assert all(view["cancelledCount"] for view in world.executions())


def test_promotion_failure_retries_from_staging_without_rerunning(tmp_path: Path, monkeypatch) -> None:
    world = CollectorWorld(tmp_path, monkeypatch)
    copy, failures = world.store.copy_object, {"left": 2}

    def flaky(**kwargs):
        if failures["left"] and kwargs["Key"].endswith("/blobs.tar"):
            failures["left"] -= 1
            raise FakeGcsError(500, "InternalError")
        return copy(**kwargs)

    monkeypatch.setattr(world.store, "copy_object", flaky)
    _complete(world)
    assert world.lease()["state"] == "completed" and world.row_state() == "completed"
    assert len(world.executions()) == 1 and world.lease()["attempt"] == 1
    row = json.loads((world.host.jobs / "rows" / "episode_compilation" / world.name).read_text(encoding="utf-8"))
    assert row["failures"] == 2 and row["promoted"]["archive"]["uri"].endswith("/blobs.tar")
    # A readback that does not match discards the promoted object, so the retry copies again.
    promoted = row["promoted"]["archive"]["uri"].removeprefix(f"s3://{B2_BUCKET}/")
    body = world.store.get_object(Bucket=B2_BUCKET, Key=promoted)["Body"].read()
    assert digest_of(body) == row["promoted"]["archive"]["digest"]


def test_partial_landing_then_host_fallback_compiles_cleanly(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline.task_evaluation_episode_compilation_worker import compile_claimed_envelope

    world = CollectorWorld(tmp_path, monkeypatch)
    real = collector._range
    reads = {"count": 0}

    def breaks(c, uri, offset, length):  # the landing's archive reads fail after its first blob
        reads["count"] += 1
        if reads["count"] > 3:
            raise OSError(5, "Input/output error")
        return real(c, uri, offset, length)

    monkeypatch.setattr(collector, "_range", breaks)
    world.drive(until=lambda: remote.marker_path(world.host.jobs, "fallback", world.name).exists(), step=600)
    compilation = world.plan.compilation_id
    assert not (world.host.outputs / compilation).exists()
    assert not list(world.host.outputs.glob(f".{compilation}.landing-*"))
    # The no-spend unit's host compile of the fallback row starts from an owned output that does not exist.
    state, result = compile_claimed_envelope(
        world.host.queue / "processing" / world.name, source_name=world.name, inputs=world.host.inputs.resolve(),
        outputs=world.host.outputs.resolve(), source_commit=world.plan.source_commit, episode_compiler=world.compiler,
        disk_reservation_root=None, storage_pins_root=None)
    assert (state, result["status"]) == ("completed", "compiled_for_production_launch")
    assert (world.host.outputs / compilation / "native-task-arena-bundle.zip").is_file()


def test_abandoned_dispatch_deletes_its_transport_first(tmp_path: Path, monkeypatch) -> None:
    world = CollectorWorld(tmp_path, monkeypatch)
    events: list[tuple[str, str]] = []
    run_job = world.remote.jobs.run_job

    def lost_before_creating(name, **kwargs):  # the request never reached Cloud Run, and nothing ran
        from tests.remote_cpu_fakes import FakeCloudRunError

        if kwargs.get("validate_only"):
            return run_job(name, **kwargs)
        raise FakeCloudRunError(None, "UNAVAILABLE", "connection reset before the request was sent")

    delete = world.bucket.delete

    def recorded(name, *, generation=None):
        events.append(("transport_deleted", name))
        return delete(name, generation=generation)

    transition = leases.transition

    def watched(root, job_id, *, attempt_id, to_state, now, updates=None):
        if to_state == "abandoned_dispatch":
            assert world.bucket._objects == {}, "abandoned before its transport was deleted"
            events.append(("abandoned_dispatch", attempt_id))
        return transition(root, job_id, attempt_id=attempt_id, to_state=to_state, now=now, updates=updates)

    monkeypatch.setattr(world.remote.jobs, "run_job", lost_before_creating)
    monkeypatch.setattr(world.bucket, "delete", recorded)
    monkeypatch.setattr(leases, "transition", watched)
    _complete(world, step=300)
    assert world.lease()["state"] == "abandoned_dispatch" and world.executions() == []
    kinds = [kind for kind, _ in events]
    assert "transport_deleted" in kinds and kinds.index("transport_deleted") < kinds.index("abandoned_dispatch")
    # Nothing ran, so the row goes back to the host.
    assert remote.marker_path(world.host.jobs, "fallback", world.name).exists()
    assert world.row_state() == "processing"


STEPS = ["compute_zero", "promotion", "validation", "landing", "pointer", "pin", "result", "move", "provider_zero"]


@pytest.mark.parametrize("step", STEPS)
def test_collector_resumes_idempotently_after_a_crash_at_each_commit_step(tmp_path: Path, monkeypatch,
                                                                          step: str) -> None:
    # The same host paths twice, so the two runs' outputs are comparable byte for byte: a clean run first.
    root = tmp_path / "run"
    clean = CollectorWorld(root, monkeypatch)
    _complete(clean)
    compilation, name = clean.plan.compilation_id, clean.name
    expected_result = (clean.host.queue / "results" / name).read_bytes()
    expected_tree = tree_snapshot(clean.host.outputs / compilation)
    shutil.rmtree(root)
    world = CollectorWorld(root, monkeypatch)
    crashed: list[str] = []

    def crash_once(name: str, attempt_id: str) -> None:
        if name == step and not crashed:
            crashed.append(name)
            raise Crash(name)

    monkeypatch.setattr(collector, "_after_step", crash_once)
    for _ in range(200):
        try:
            world.collect()
        except Crash:
            pass
        if _done(world):
            break
        world.advance(60)
    assert crashed == [step] and _done(world) and world.name == name
    assert (world.lease()["state"], world.row_state()) == ("completed", "completed")
    assert len(world.executions()) == 1 and world.lease()["attempt"] == 1
    assert (world.host.queue / "results" / name).read_bytes() == expected_result
    assert tree_snapshot(world.host.outputs / compilation) == expected_tree
    pointer = json.loads((world.host.outputs / f"{compilation}.remote-output.v1.json").read_text(encoding="utf-8"))
    assert pointer["provider_zero_proven"] and pointer["teardown_receipt_digest"] == world.lease()["teardown_digest"]
    cas = [key for key in world.store.buckets[B2_BUCKET] if "/remote-cpu-output/" in key]
    assert len(cas) == 2 and all(len(world.store.buckets[B2_BUCKET][key]) == 1 for key in cas)
    _assert_torn_down(world)
    assert not list(world.host.outputs.glob(f".{compilation}.landing-*"))


def test_the_collector_writes_object_storage_only_through_its_approved_promotion() -> None:
    """Plan 14 §9: the collector's CAS writes are the server-side promotion and, after a failed readback, the
    discard of that promoted object; the verifier names ``_promote`` as their only new caller."""

    import importlib.util

    root = Path(collector.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("verify_paid_resource_allocator",
                                                  root / "scripts" / "verify_paid_resource_allocator.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    module = "src/blueprint_pipeline/task_evaluation_episode_compilation_collector.py"
    source = (root / module).read_text(encoding="utf-8")
    assert "discard_remote_cpu_output_object" in verifier.REMOTE_CPU_OBJECT_STORE_WRITERS
    assert verifier._s3_transport_capability_callers(
        {module: source}, verifier.REMOTE_CPU_OBJECT_STORE_WRITERS) == {(module, "_promote")}
    assert {(path, name) for path, name in verifier.APPROVED_REMOTE_CPU_OBJECT_STORE_CALLERS
            if path == module} == {(module, "_promote")}
    # No direct provider mutation of its own: executions start and stop only through the allocator subcommand.
    assert verifier._direct_paid_mutation_signals(source) == set()
    assert not {"require_paid_resource_admission", "build_paid_lane_admission"} & verifier._all_calls(root / module)


def test_shadow_keeps_host_authoritative_and_counts_parity_per_closure_class(tmp_path: Path, monkeypatch) -> None:
    """Plan 14 §1, §13: the host compiled and moved each row first; a shadow attempt compiles again remotely,
    is compared, and is discarded: nothing is promoted or landed, staging is deleted, parity is per class."""

    from tests.remote_episode_compilation_support import nurec_usdz

    monkeypatch.setattr(remote, "MAX_INLINE_NUREC_BYTES", 8192)
    world = CollectorWorld(tmp_path, monkeypatch, mode="cloud_run_shadow", marker="shadow")
    appearance = nurec_usdz(16384)
    shipped = world.add_row(label="shipped", appearance=appearance, appearance_name="appearance.usdz", cache=True)
    drifted = world.add_row(label="drifted")
    assert (world.plan.closure["class"], shipped.closure["class"], drifted.closure["class"]) == (
        "not_applicable", "shipped", "not_applicable")

    def differs(descriptor: dict, roots) -> dict:  # this worker's packet request is not the host's
        from blueprint_pipeline.task_evaluation_episode_compilation_remote import run_episode_compilation_in_worker

        result = run_episode_compilation_in_worker(descriptor, roots, episode_compiler=world.compiler)
        if descriptor["queue_row"]["name"] == drifted.queue_row["name"]:
            request = roots.local(descriptor["outputs"]["output_root"]) / "native-task-packet" / (
                "native_task_arena_packet_request.v1.json")
            request.write_text(request.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return {"result": result, "release_path_misses": [], "failures": []}

    world.stage_override = differs
    host_before = {path: tree_snapshot(path) for path in sorted(world.host.outputs.iterdir())}
    results_before = {path.name: path.read_bytes() for path in (world.host.queue / "results").iterdir()}
    world.drive(until=lambda: not remote.markers(world.host.jobs, "shadow"), step=120)

    # The host stayed authoritative: every row was compiled and moved by the host, and none of it changed.
    assert {path.name: path.read_bytes() for path in (world.host.queue / "results").iterdir()} == results_before
    assert {path: tree_snapshot(path) for path in sorted(world.host.outputs.iterdir())} == host_before
    assert not list(world.host.outputs.glob("*.remote-output.v1.json"))
    assert sorted(path.name for path in (world.host.queue / "completed").iterdir()) == sorted(results_before)
    # Nothing was promoted, and every attempt's staging and transport are gone at provider-zero.
    assert not [key for key in world.store.buckets[B2_BUCKET] if "/remote-cpu-output/" in key]
    assert _staging_versions(world) == [] and world.bucket._objects == {}
    assert leases.slots_in_use(world.host.jobs) == 0 and len(world.executions()) == 3
    for plan in (world.plan, shipped, drifted):
        job_id = contract.job_id_for("episode_compilation", plan.queue_row["name"])
        lease = json.loads((world.host.jobs / "leases" / f"{job_id}.json").read_text(encoding="utf-8"))
        assert lease["state"] == "shadow_compared" and lease["provider_zero_proven"]
    # Parity is counted per closure class, for the image, host environment and CPU class it was measured on.
    parity = {record["queue_row"]["name"]: record for record in (
        json.loads(path.read_text(encoding="utf-8"))
        for path in (world.host.jobs / "parity" / "episode_compilation").glob("*.json"))}
    assert parity[world.name]["parity"] == parity[shipped.queue_row["name"]]["parity"] == "passed"
    assert parity[drifted.queue_row["name"]]["parity"] == "failed"
    assert parity[drifted.queue_row["name"]]["mismatches"] == ["native-task-packet/native_task_arena_packet_request.v1.json"]
    assert world.results[-1]["parity"] == {"not_applicable": {"passed": 1, "failed": 1, "inconclusive": 0},
                                           "shipped": {"passed": 1, "failed": 0, "inconclusive": 0}}
    identity = {"image": world.plan.image, "host_environment_digest": HOST_RECORD["environment_digest"],
                "cpu_class": HOST_RECORD["cpu_class"]}
    # Consecutive passes since the class's last failure: the pass counts only if it was compared after it.
    passed_last = parity[world.name]["compared_at_epoch"] > parity[drifted.queue_row["name"]]["compared_at_epoch"]
    assert remote.shadow_passes(world.host.jobs, closure_class="not_applicable", **identity) == int(passed_last)
    assert remote.shadow_passes(world.host.jobs, closure_class="shipped", **identity) == 1


def _consumed(world: CollectorWorld) -> list[Path]:
    return sorted((world.remote.spend / "consumed").glob("remote-cpu-*.json"))


def test_a_refused_dispatch_hands_the_row_back_and_is_never_dispatched_again(tmp_path: Path, monkeypatch) -> None:
    """Review C1: a dispatch refused before its lease claim writes the fallback, then drops the hand-off, so no
    later run can pay to compile a row the host has already compiled."""

    from tests.remote_cpu_allocator_fakes import standing_authority

    world = CollectorWorld(tmp_path, monkeypatch)
    handoff = remote.marker_path(world.host.jobs, "authoritative", world.name)
    fallback = remote.marker_path(world.host.jobs, "fallback", world.name)
    (world.remote.spend / "authorizations" / "remote-cpu-standing-authorization.v1.json").unlink()
    world.collect()
    assert world.results[-1]["rows"][world.name]["status"] == "dispatch_refused"
    assert fallback.is_file() and not handoff.exists() and world.lease() is None
    # The no-spend unit compiles the handed-back row; then the authority comes back.
    world.host_compile(world.name)
    fallback.unlink()
    world.remote.write_authority(standing_authority())
    for _ in range(3):
        world.advance(120)
        world.collect()
    assert world.executions() == [] and _consumed(world) == [] and world.lease() is None
    assert world.row_state() == "completed"


@pytest.mark.parametrize("left", ["fallback", "gave_up", "row_moved"])
def test_a_hand_off_whose_row_went_back_is_never_dispatched(tmp_path: Path, monkeypatch, left: str) -> None:
    """Review C1: whatever a crash left between the fallback and the hand-off's removal, or wherever the row went,
    a hand-off whose row was given up, handed back or moved on is retired, never dispatched."""

    world = CollectorWorld(tmp_path, monkeypatch)
    handoff = remote.marker_path(world.host.jobs, "authoritative", world.name)
    fallback = remote.marker_path(world.host.jobs, "fallback", world.name)
    c = world.collector()
    if left == "fallback":  # the fallback was written, then the process died before the hand-off went
        remote.write_fallback(world.host.jobs, world.plan.queue_row, reason="remote_cpu_dispatch_refused", attempts=0,
                              now=world.clock.now)
    elif left == "gave_up":  # the give-up was recorded, then the process died before the fallback
        collector._save_row(c, collector._row(c, world.name, world.plan.queue_row),
                            gave_up={"reason": "remote_cpu_dispatch_refused", "at_epoch": world.clock.now})
    else:  # the row is no longer claimed: the host compiled it
        world.host_compile(world.name)
    world.collect()
    assert world.executions() == [] and _consumed(world) == [] and world.lease() is None
    assert not handoff.exists()
    if left == "row_moved":
        assert not fallback.exists() and world.row_state() == "completed"
    else:
        assert fallback.is_file() and world.row_state() == "processing"


def test_a_refused_second_attempt_hands_back_and_keeps_following_the_first_to_provider_zero(
        tmp_path: Path, monkeypatch) -> None:
    """Review C1: the refusal of attempt 2 hands the row back at once; the hand-off stays only until attempt 1 is
    provider-zero, then the lease closes as ``fallback_host`` and attempt 2 never runs."""

    from tests.remote_cpu_allocator_fakes import standing_authority

    world = CollectorWorld(tmp_path, monkeypatch)
    world.remote.jobs.script("crash")
    world.drive(until=lambda: (world.lease() or {}).get("state") == "expired", step=30)
    (world.remote.spend / "authorizations" / "remote-cpu-standing-authorization.v1.json").unlink()
    fallback = remote.marker_path(world.host.jobs, "fallback", world.name)
    world.drive(until=fallback.exists, step=30)
    assert world.row_state() == "processing" and len(world.executions()) == 1
    world.remote.write_authority(standing_authority())
    _complete(world, step=300)
    assert world.lease()["state"] == "fallback_host" and world.lease()["attempt"] == 1
    assert len(world.executions()) == 1 and len(_consumed(world)) == 1
    _assert_torn_down(world)


def test_a_shadow_comparison_of_two_blocked_compiles_is_inconclusive(tmp_path: Path, monkeypatch) -> None:
    """Review I2: a pass needs both sides compiled and their trees compared byte for byte.  Two blocked
    compiles compare nothing, so they neither advance a class toward ``cloud_run`` nor reset it."""

    from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import (
        TaskEvaluationNativeArenaEpisodeCompilerError,
    )

    world = CollectorWorld(tmp_path, monkeypatch, mode="cloud_run_shadow", marker="shadow")
    world.drive(until=lambda: not remote.markers(world.host.jobs, "shadow"), step=120)  # a real pass first

    def refuses(**_kwargs):
        raise TaskEvaluationNativeArenaEpisodeCompilerError("episode_compiler_destination_usd_format_unrecognized")

    world.compiler = refuses  # the host and the worker both refuse the next rows
    blocked = [world.add_row(label=f"refused-{index}") for index in range(3)]
    assert all((world.host.queue / "blocked" / plan.queue_row["name"]).is_file() for plan in blocked)
    world.drive(until=lambda: not remote.markers(world.host.jobs, "shadow"), step=120)

    parity = {record["queue_row"]["name"]: record["parity"] for record in (
        json.loads(path.read_text(encoding="utf-8"))
        for path in (world.host.jobs / "parity" / "episode_compilation").glob("*.json"))}
    assert parity == {world.name: "passed", **{plan.queue_row["name"]: "inconclusive" for plan in blocked}}
    identity = {"image": world.plan.image, "host_environment_digest": HOST_RECORD["environment_digest"],
                "cpu_class": HOST_RECORD["cpu_class"]}
    assert remote.shadow_passes(world.host.jobs, closure_class="not_applicable", **identity) == 1
    assert world.results[-1]["parity"] == {"not_applicable": {"passed": 1, "failed": 0, "inconclusive": 3}}


@pytest.mark.parametrize("trigger", ["result", "pointer"])
def test_a_commit_that_cannot_finish_still_tears_down_and_hands_the_row_back(tmp_path: Path, monkeypatch,
                                                                              trigger: str) -> None:
    """Review I3: an unexpected commit error after compute-zero (a result or a pointer already there) never
    wedges the attempt.  It is torn down to provider-zero and settled, the lease ends ``blocked`` and frees its
    slot, and the uncommitted row goes back to the host; what was already there is left untouched."""

    from blueprint_pipeline.task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive

    world = CollectorWorld(tmp_path, monkeypatch)
    if trigger == "result":
        existing = world.host.queue / "results" / world.name
        write_launch_preparation_record_exclusive(existing, {"schema_version": "task_evaluation_episode_compilation_result.v1",
                                                             "status": "blocked", "blockers": ["written_elsewhere"]})
    else:
        existing = world.host.outputs / f"{world.plan.compilation_id}.remote-output.v1.json"
        existing.write_text("{}\n", encoding="utf-8")
    before = existing.read_bytes()
    _complete(world, step=300)
    lease = world.lease()
    assert lease["state"] == "blocked" and lease["outcome"].startswith("remote_cpu_commit_failed:"), lease["outcome"]
    _assert_torn_down(world)
    assert remote.marker_path(world.host.jobs, "fallback", world.name).is_file()
    assert world.row_state() == "processing" and existing.read_bytes() == before


def test_the_shadow_tree_comparison_hashes_files_as_a_stream(tmp_path: Path, monkeypatch) -> None:
    """Review I3: comparing a host tree never reads a whole output file into memory (``MemoryMax=2G``)."""

    root = tmp_path / "out"
    (root / "native-task-packet").mkdir(parents=True)
    member = root / "native-task-packet" / "bundle.zip"
    member.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    index = {"root_mode": f"{root.stat().st_mode & 0o7777:04o}",
             "directories": [{"path": "native-task-packet", "mode": f"{member.parent.stat().st_mode & 0o7777:04o}"}],
             "entries": [{"path": "native-task-packet/bundle.zip", "blob": digest_of(member.read_bytes()),
                          "size_bytes": member.stat().st_size, "mode": f"{member.stat().st_mode & 0o7777:04o}"}]}

    def whole_file(self):
        raise MemoryError("read the whole file")

    monkeypatch.setattr(Path, "read_bytes", whole_file)
    assert collector._tree_mismatches(root, index) == []


def test_without_a_provider_connection_undispatched_hand_offs_still_go_back(tmp_path: Path, monkeypatch) -> None:
    """Review I5: a rotated or revoked key must not strand every hand-off in processing/.  With no provider
    connection the unit still hands back each hand-off that never dispatched and writes its summary; an attempt
    that may have started keeps its hand-off, since it is never handed back without its teardown."""

    import functools

    from blueprint_pipeline import remote_cpu_job_allocator as allocator
    from blueprint_pipeline.task_evaluation_episode_compilation_worker import OUTPUT_ROOT_ENV, QUEUE_ROOT_ENV

    world = CollectorWorld(tmp_path, monkeypatch)
    world.drive(until=lambda: (world.lease() or {}).get("state") in {"dispatched", "running"}, step=30)
    started = world.lease()
    waiting = world.add_row(label="waiting", marker="authoritative")
    name = waiting.queue_row["name"]
    monkeypatch.setenv(remote.EXECUTION_ENV, "cloud_run")
    monkeypatch.setenv(QUEUE_ROOT_ENV, str(world.host.queue))
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(world.host.outputs))
    monkeypatch.setattr(allocator, "load_remote_cpu_config", lambda path=None: (world.remote.config, []))
    monkeypatch.setattr(allocator, "_connect", lambda runtime, config: ["remote_cpu_dispatcher_unavailable:KeyError"])
    monkeypatch.setattr(allocator, "RemoteCpuRuntime", functools.partial(allocator.RemoteCpuRuntime, clock=world.clock))

    assert collector.main(["run", "--source-commit", world.plan.source_commit, "--jobs-root", str(world.host.jobs)]) == 0
    assert remote.marker_path(world.host.jobs, "fallback", name).is_file()
    assert not remote.marker_path(world.host.jobs, "authoritative", name).exists()
    assert (world.host.queue / "processing" / name).is_file()
    # The started attempt keeps its hand-off and its lease, untouched, until a connected run tears it down.
    assert remote.marker_path(world.host.jobs, "authoritative", world.name).is_file()
    assert not remote.marker_path(world.host.jobs, "fallback", world.name).exists()
    assert world.lease() == started
    summary = json.loads((world.host.jobs / "summary.json").read_text(encoding="utf-8"))
    assert summary["blockers"] == ["remote_cpu_dispatcher_unavailable:KeyError"]
    assert summary["rows"][name]["status"] == "returned_to_host"
    assert summary["rows"][world.name]["status"] == "held_for_provider"


def test_a_failed_readback_discards_only_the_version_this_attempt_created(tmp_path: Path, monkeypatch) -> None:
    """Review minor: a content-addressed key another pointer may share is never emptied by an attempt that did
    not write it.  Here the archive's key already holds bytes that do not match: the readback fails, nothing
    this attempt did not create is deleted, and the attempt fails and the row goes back to the host instead."""

    from blueprint_pipeline.task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX

    world = CollectorWorld(tmp_path, monkeypatch)
    real, planted = collector.copy_remote_cpu_staging_to_cas, []

    def someone_else_promoted_first(**kwargs):
        if kwargs["filename"] == "blobs.tar" and not planted:
            key = f"{LARGE_ARTIFACT_KEY_PREFIX}/remote-cpu-output/sha256/{kwargs['digest'][7:]}/blobs.tar"
            world.store.put_object(Bucket=B2_BUCKET, Key=key, Body=b"\0" * kwargs["size_bytes"],
                                   Metadata={"sha256": kwargs["digest"][7:]})
            planted.append(key)
        return real(**kwargs)

    monkeypatch.setattr(collector, "copy_remote_cpu_staging_to_cas", someone_else_promoted_first)
    _complete(world, step=300)
    assert world.lease()["state"] == "fallback_host" and world.row_state() == "processing"
    [key] = planted
    versions = [row for row in world.store.list_object_versions(Bucket=B2_BUCKET, Prefix=key)["Versions"]
                if row["Key"] == key]
    assert len(versions) == 1  # the other promotion's object is still there, untouched
    _assert_torn_down(world)


@pytest.mark.parametrize("reads", [1, 3])
def test_a_transient_read_during_the_commit_is_retried_not_abandoned(tmp_path: Path, monkeypatch, reads: int) -> None:
    """Review I3 follow-up: only a commit that cannot finish is abandoned.  A CAS read that fails a few times
    (in the readback, the validation or the landing) counts against the collection budget and is retried
    from staging; the compile still lands, from the one execution."""

    world = CollectorWorld(tmp_path, monkeypatch)
    get, failures = world.store.get_object, {"left": reads}

    def flaky(**kwargs):
        if failures["left"] and "/remote-cpu-output/" in kwargs["Key"]:
            failures["left"] -= 1
            raise FakeGcsError(500, "InternalError")
        return get(**kwargs)

    monkeypatch.setattr(world.store, "get_object", flaky)
    _complete(world)
    assert (world.lease()["state"], world.row_state()) == ("completed", "completed")
    assert len(world.executions()) == 1 and failures["left"] == 0
    _assert_torn_down(world)


def test_an_allocator_call_reads_only_its_own_result_and_a_lost_dispatch_is_ambiguous(tmp_path: Path,
                                                                                       monkeypatch) -> None:
    """Review minor: each allocator call gets a fresh ``--out`` path, so a result an earlier call left can never
    be read as this one's; a dispatch that timed out or left no result may have started an execution, so it is
    ambiguous (the hand-off is held for the allocator's reconcile), never a refusal that hands the row back."""

    import subprocess

    world = CollectorWorld(tmp_path, monkeypatch)
    seen: list[str] = []
    c = world.collector(allocate=lambda argv: seen.append(argv[argv.index("--out") + 1]) or {"status": "blocked"})
    collector._allocate(c, "sweep", None, "stage")
    collector._allocate(c, "sweep", None, "stage")
    assert len(set(seen)) == 2 and not any(Path(path).exists() for path in seen)

    out = tmp_path / "allocator" / "rcj.dispatch.json"
    out.parent.mkdir()
    out.write_text(json.dumps({"status": "dispatched", "blockers": []}), encoding="utf-8")  # an earlier call's
    out.chmod(0o640)
    argv = ["remote-cpu-job", "--action", "dispatch", "--stage", "episode_compilation", "--lease", str(tmp_path),
            "--out", str(out), "--execute"]
    monkeypatch.setattr(collector.subprocess, "run",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 1))
    assert collector.subprocess_allocate(argv) == {
        "status": "ambiguous_dispatch_unresolved", "blockers": ["remote_cpu_allocator_result_unreadable:exit_1"]}

    def hangs(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(collector.subprocess, "run", hangs)
    assert collector.subprocess_allocate(argv) == {"status": "ambiguous_dispatch_unresolved",
                                                   "blockers": ["remote_cpu_allocator_timeout"]}
    reconcile = [("reconcile" if item == "dispatch" else item) for item in argv]
    assert collector.subprocess_allocate(reconcile)["status"] == "blocked"
    # Ambiguous is not refused: the hand-off stays for reconcile, and nothing goes back to the host.
    world.collect(allocate=lambda argv: {"status": "ambiguous_dispatch_unresolved",
                                         "blockers": ["remote_cpu_allocator_timeout"]}
                  if "dispatch" in argv else world.allocate(argv))
    assert remote.marker_path(world.host.jobs, "authoritative", world.name).is_file()
    assert not remote.marker_path(world.host.jobs, "fallback", world.name).exists()


@pytest.mark.parametrize("where", ["provider_zero", "settle"])
def test_a_transient_error_after_the_row_committed_resumes_its_teardown(tmp_path: Path, monkeypatch,
                                                                        where: str) -> None:
    """Review N1: once the result is written, an error in the teardown (the provider-zero proof, or the
    settlement after it) is retried by the next run, which finishes ``completed`` with the pointer resealed and
    exactly one settlement; it is never taken for a commit that could not finish."""

    world = CollectorWorld(tmp_path, monkeypatch)
    failures = {"left": 1}
    if where == "provider_zero":
        real_proof = collector.allocator.prove_provider_zero

        def flaky_proof(action, attempt, descriptor):
            if failures["left"] and world.row_state() == "completed":
                failures["left"] -= 1
                raise ConnectionResetError(54, "Connection reset by peer")
            return real_proof(action, attempt, descriptor)

        monkeypatch.setattr(collector.allocator, "prove_provider_zero", flaky_proof)
    else:
        real_settle = collector._settle

        def flaky_settle(c, attempt, descriptor, record):
            if failures["left"]:
                failures["left"] -= 1
                raise ConnectionResetError(54, "Connection reset by peer")
            return real_settle(c, attempt, descriptor, record)

        monkeypatch.setattr(collector, "_settle", flaky_settle)
    _complete(world, step=300)
    lease = world.lease()
    assert failures["left"] == 0
    assert (lease["state"], lease["outcome"], world.row_state()) == (
        "completed", "compiled_for_production_launch", "completed")
    pointer = json.loads((world.host.outputs / f"{world.plan.compilation_id}.remote-output.v1.json").read_text(
        encoding="utf-8"))
    assert pointer["provider_zero_proven"] and pointer["teardown_receipt_digest"] == lease["teardown_digest"]
    assert not remote.marker_path(world.host.jobs, "fallback", world.name).exists()
    _assert_torn_down(world)  # one settlement for the one started attempt


def test_a_transient_write_error_before_the_result_resumes_the_same_attempt(tmp_path: Path, monkeypatch) -> None:
    """Review N1: a full disk while writing the pointer is not a conflict.  The next run resumes the same
    attempt at the step that failed: no fallback, no interruption counted, nothing set aside."""

    import errno

    world = CollectorWorld(tmp_path, monkeypatch)
    real, failures = collector.replace_remote_cpu_record, {"left": 1}

    def full(path, value, **kwargs):
        if failures["left"] and str(path).endswith(".remote-output.v1.json"):
            failures["left"] -= 1
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(path, value, **kwargs)

    monkeypatch.setattr(collector, "replace_remote_cpu_record", full)
    _complete(world)
    assert failures["left"] == 0
    assert (world.lease()["state"], world.row_state()) == ("completed", "completed")
    assert len(world.executions()) == 1 and world.lease()["attempt"] == 1
    assert not remote.marker_path(world.host.jobs, "fallback", world.name).exists()
    assert not list(world.host.outputs.glob(".*.interrupted-*"))
    assert not list((world.host.jobs / "recovery").rglob("*.json"))
    _assert_torn_down(world)


def _denied(*_args, **_kwargs):
    raise PermissionError(13, "Permission denied")


def _lease_of(world: CollectorWorld, plan) -> dict | None:
    path = world.host.jobs / "leases" / f"{contract.job_id_for('episode_compilation', plan.queue_row['name'])}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@pytest.mark.parametrize("site", ["result", "pointer", "shadow_tree"])
def test_a_persistent_host_error_before_the_commit_is_bounded_then_abandoned(tmp_path: Path, monkeypatch,
                                                                             site: str) -> None:
    """Review Q3: an unnamed error that never clears (EACCES writing the result or the pointer, or reading the
    shadow tree) counts against the collection budget.  Past it the attempt is torn down to provider-zero and
    settled, its lease ends ``blocked``, the row goes back, and no second attempt is paid for."""

    shadow = site == "shadow_tree"
    world = CollectorWorld(tmp_path, monkeypatch, **({"mode": "cloud_run_shadow", "marker": "shadow"} if shadow else {}))
    if site == "result":
        monkeypatch.setattr(collector, "write_launch_preparation_record_exclusive", _denied)
    elif site == "pointer":
        real = collector.replace_remote_cpu_record

        def denied_pointer(path, value, **kwargs):
            if str(path).endswith(".remote-output.v1.json"):
                _denied()
            return real(path, value, **kwargs)

        monkeypatch.setattr(collector, "replace_remote_cpu_record", denied_pointer)
    else:
        monkeypatch.setattr(collector, "_file_digest", _denied)
    kind = "shadow" if shadow else "authoritative"
    world.drive(until=lambda: world.terminal() and not remote.marker_path(world.host.jobs, kind, world.name).exists(),
                step=60)
    lease = world.lease()
    assert (lease["state"], lease["attempt"], len(world.executions())) == ("blocked", 1, 1)
    assert lease["outcome"].startswith("remote_cpu_commit_failed:")
    row = json.loads((world.host.jobs / "rows" / "episode_compilation" / world.name).read_text(encoding="utf-8"))
    assert row["failures"] == collector.COLLECTION_RETRIES
    _assert_torn_down(world)  # provider-zero, settled, and its slot free
    if shadow:
        assert not list((world.host.jobs / "parity").rglob("*.json"))
    else:
        assert remote.marker_path(world.host.jobs, "fallback", world.name).is_file()
        assert world.row_state() == "processing"


def test_two_wedged_rows_free_their_slots_for_a_third_hand_off(tmp_path: Path, monkeypatch) -> None:
    """Review Q3: two rows whose commits keep failing held both live-execution slots forever, so a third
    hand-off waited in ``awaiting_capacity``.  Past the budget they tear down, and the third one runs."""

    world = CollectorWorld(tmp_path, monkeypatch)
    wedged = world.add_row(label="aaa-wedged", marker="authoritative")
    waiting = world.add_row(label="zzz-waiting", marker="authoritative")
    real = collector.write_launch_preparation_record_exclusive

    def denied_unless_waiting(path, value):
        if Path(path).name != waiting.queue_row["name"]:
            _denied()
        return real(path, value)

    monkeypatch.setattr(collector, "write_launch_preparation_record_exclusive", denied_unless_waiting)
    seen: set = set()

    def finished() -> bool:
        lease = _lease_of(world, waiting)
        seen.add(None if lease is None else lease["state"])
        return lease is not None and lease["state"] == "completed"

    world.drive(until=finished, step=60)
    assert "awaiting_capacity" in seen
    assert [_lease_of(world, plan)["state"] for plan in (world.plan, wedged)] == ["blocked", "blocked"]
    assert len(world.executions()) == 3 and leases.slots_in_use(world.host.jobs) == 0
