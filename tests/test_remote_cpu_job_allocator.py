# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_job_allocator.py
#   src/blueprint_pipeline/cloud_run_jobs_client.py
#   src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py
#   src/blueprint_pipeline/remote_cpu_job_lease.py
#   tests/remote_cpu_allocator_fakes.py
"""ADP-009D/day-28, plan 14 PR 2: remote CPU jobs are admitted, consumed, leased and torn down by one seam."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import logging
import os
import stat
import sys
import types
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from blueprint_pipeline import remote_cpu_job_allocator as allocator
from blueprint_pipeline import task_evaluation_configured_scene_object_store as object_store
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_job_lease as leases
from blueprint_pipeline import remote_cpu_job_records as records
from blueprint_pipeline.cloud_run_jobs_client import GcsTransportBucket, allocation_binding
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
)
from blueprint_pipeline.task_evaluation_configured_scene_object_store import (
    TaskEvaluationConfiguredSceneObjectStoreError as ObjectStoreError,
)
from tests.remote_cpu_allocator_fakes import (
    B2_BUCKET,
    CAS,
    HOST_RECORD,
    IMAGE,
    JOB,
    OBJECT_PREFIX,
    T0,
    TRANSPORT_BUCKET,
    WORST_CASE_USD,
    CloudRunRest,
    FakeWorker,
    RecordingArtifactStore,
    RemoteCpuWorld,
    environment,
    remote_cpu_config,
    standing_authority,
)
from tests.remote_cpu_fakes import FakeClock, FakeGcsError, env_value


def _consumption(world: RemoteCpuWorld, descriptor: dict) -> Path:
    return world.spend / "consumed" / f"remote-cpu-{hashlib.sha256(descriptor['attempt_id'].encode()).hexdigest()}.json"


def test_dispatch_is_refused_before_any_mutation_without_standing_authority(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch, with_authority=False)
    world.record_environment()
    descriptor = world.descriptor()

    result = world.run("dispatch", descriptor=descriptor)
    assert (result["status"], result["success"]) == ("blocked", False)
    assert "remote_cpu_standing_authority_missing" in result["blockers"]
    assert result["admission"]["status"] == "blocked"
    world.assert_untouched()

    for authority, mode, reason in (
        (standing_authority(expires_at_epoch=T0 - 1), 0o600, "remote_cpu_standing_authority_expired"),
        (standing_authority(stages=["another_stage"]), 0o600, "remote_cpu_standing_authority_stage_not_covered"),
        ({**standing_authority(), "max_total_usd": 1000.0}, 0o600,
         "remote_cpu_standing_authority_invalid:authorization_digest"),
        (standing_authority(), 0o640, "remote_cpu_standing_authority_invalid:unsafe"),
    ):
        world.write_authority(authority, mode=mode)
        result = world.run("dispatch", descriptor=descriptor)
        assert result["status"] == "blocked" and reason in result["blockers"], reason
        assert result["admission"]["status"] == "blocked"
        world.assert_untouched()
    assert world.lease(descriptor) is None

    # The admission is recorded even when it refuses; nothing in it names a URL or a credential.
    assert result["admission"]["allocation_binding"]["descriptor_digest"] == descriptor["descriptor_digest"]
    assert contract.forbidden_record_content(result) == []


def test_admission_binding_is_added_after_build_paid_lane_admission(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    descriptor = world.descriptor()
    built: list[tuple[dict, dict]] = []
    required: list[dict] = []
    build, require = allocator.build_paid_lane_admission, allocator.require_paid_resource_admission

    def spy_build(*args, **kwargs) -> dict:
        assert not args, "build_paid_lane_admission is keyword-only"
        admission = build(**kwargs)
        built.append((dict(kwargs), json.loads(json.dumps(admission))))
        return admission

    def spy_require(admission: dict, **kwargs):
        required.append(json.loads(json.dumps(admission)))
        return require(admission, **kwargs)

    monkeypatch.setattr(allocator, "build_paid_lane_admission", spy_build)
    monkeypatch.setattr(allocator, "require_paid_resource_admission", spy_require)

    dry = world.run("dispatch", descriptor=descriptor, execute=False)
    assert (dry["status"], required) == ("dry_run_ready", [])
    [(arguments, admission)] = built
    assert arguments == {"resource_class": "cloud_run_cpu_job", "blockers": []}
    assert "allocation_binding" not in admission and "allocation_binding_digest" not in admission
    binding = allocation_binding(descriptor_digest=descriptor["descriptor_digest"], attempt_id=descriptor["attempt_id"],
                                 job=JOB, etag=world.jobs.get_job(JOB)["etag"])
    assert dry["admission"] == {
        "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION, "status": "admitted",
        "resource_class": "cloud_run_cpu_job", "blockers": [], "allocation_binding": binding,
        "allocation_binding_digest": canonical_digest(binding),
    }
    world.assert_untouched()

    admission, grant = allocator.admit_remote_cpu_job(blockers=[], binding=binding, execute=True)
    assert grant is not None and grant.allocation_binding_digest == canonical_digest(binding)
    assert required[-1]["allocation_binding_digest"] == canonical_digest(binding)
    refused, none = allocator.admit_remote_cpu_job(blockers=["remote_cpu_x", "remote_cpu_x"], binding=binding,
                                                   execute=True)
    assert none is None and refused["blockers"] == ["remote_cpu_x"]
    assert refused["allocation_binding_digest"] == canonical_digest(binding)

    executed = world.run("dispatch", descriptor=descriptor)
    assert executed["admission"]["status"] == "admitted"
    assert required[-1]["allocation_binding"] == binding


def test_authority_is_consumed_once_per_attempt(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    authority = standing_authority()

    first = world.descriptor(label="prep-1", nonce="1" * 32)
    consumed = allocator.consume_remote_cpu_authority_once(
        descriptor=first, authority=authority, worst_case_usd=WORST_CASE_USD,
        binding_digest="sha256:" + "b" * 64, now=T0)
    assert consumed["status"] == "consumed"
    path = _consumption(world, first)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (record["attempt_id"], record["worst_case_usd"], record["standing_authority_digest"]) == (
        first["attempt_id"], WORST_CASE_USD, authority["authorization_digest"])
    before = path.read_bytes()
    again = allocator.consume_remote_cpu_authority_once(
        descriptor=first, authority=authority, worst_case_usd=WORST_CASE_USD,
        binding_digest="sha256:" + "b" * 64, now=T0 + 1)
    assert again == {"status": "blocked", "blockers": ["remote_cpu_authority_already_consumed"]}
    assert path.read_bytes() == before and world.consumed() == [path]

    # A dispatch of an attempt whose authority is already consumed is refused before any mutation.
    refused = world.run("dispatch", descriptor=first)
    assert refused["status"] == "blocked" and "remote_cpu_authority_already_consumed" in refused["blockers"]
    assert world.rest.mutations == [] and world.bucket._objects == {} and world.store.presigned == []

    # Another attempt of the same row is consumed on its own, once.
    second = world.descriptor(label="prep-1", nonce="2" * 32)
    dispatched = world.run("dispatch", descriptor=second)
    assert dispatched["admission"]["status"] == "admitted"
    assert world.consumed() == sorted([path, _consumption(world, second)])
    replay = world.run("dispatch", descriptor=second)
    assert replay["success"] is False and world.consumed() == sorted([path, _consumption(world, second)])
    assert len(world.rest.runs(validate_only=False)) <= 1


def test_caps_count_unsettled_attempts_at_worst_case(tmp_path: Path, monkeypatch) -> None:
    authority = standing_authority(max_attempt_usd=1.0, max_daily_usd=1.5, max_total_usd=2.0, max_executions=4)
    world = RemoteCpuWorld(tmp_path, monkeypatch, authority=authority)
    world.record_environment()
    limits = allocator.stage_limits(world.config, "episode_compilation")
    worst = allocator.worst_case_usd(limits=limits, rate_table=world.config["rate_table"])
    assert worst == WORST_CASE_USD

    def blockers(now: float, **authority_changes) -> list[str]:
        return allocator.spend_ledger_blockers(
            authority={**authority, **authority_changes}, worst_case_usd=worst, now=now)

    def consume(label: str, now: float) -> dict:
        descriptor = world.descriptor(label=label)
        assert allocator.consume_remote_cpu_authority_once(
            descriptor=descriptor, authority=authority, worst_case_usd=worst,
            binding_digest="sha256:" + "b" * 64, now=now)["status"] == "consumed"
        return descriptor

    assert blockers(T0) == []
    assert blockers(T0, max_attempt_usd=0.5) == ["remote_cpu_attempt_cap_exceeded"]
    first, _second = consume("prep-a", T0), consume("prep-b", T0 + 1)
    # Two unsettled attempts count at their worst case: a third would pass $1.5 in a day and $2 in total.
    assert blockers(T0 + 2) == ["remote_cpu_daily_cap_exceeded", "remote_cpu_total_cap_exceeded"]
    # Re-issuing the authority (here only its date changes) never resets what earlier attempts spent.
    reissued = standing_authority(max_attempt_usd=1.0, max_daily_usd=1.5, max_total_usd=2.0, max_executions=4,
                                  authorized_on="2026-09-29")
    assert allocator.spend_ledger_blockers(authority=reissued, worst_case_usd=worst, now=T0 + 2) == [
        "remote_cpu_daily_cap_exceeded", "remote_cpu_total_cap_exceeded"]
    refused = world.run("dispatch", descriptor=world.descriptor(label="prep-c"))
    assert {"remote_cpu_daily_cap_exceeded", "remote_cpu_total_cap_exceeded"} <= set(refused["blockers"])
    assert world.rest.mutations == [] and world.store.presigned == []

    # Settling the first at its estimate frees the difference.
    allocator.settle_remote_cpu_attempt(descriptor=first, teardown_digest="sha256:" + "d" * 64, settled_usd=0.05,
                                        basis="execution_runtime", now=T0 + 3)
    assert blockers(T0 + 3) == []
    consume("prep-c", T0 + 4)
    assert blockers(T0 + 5) == ["remote_cpu_daily_cap_exceeded", "remote_cpu_total_cap_exceeded"]
    # A day later only the total still counts them; and the execution count is its own cap.
    assert blockers(T0 + 86400 + 10) == ["remote_cpu_total_cap_exceeded"]
    assert blockers(T0 + 86400 + 10, max_total_usd=25.0, max_executions=3) == ["remote_cpu_execution_cap_exceeded"]

    # An unreadable consumption record fails the ledger closed.
    world.consumed()[0].write_text("{", encoding="utf-8")
    assert blockers(T0 + 86400 + 10, max_total_usd=25.0) == ["remote_cpu_spend_ledger_unreadable"]


def test_live_execution_cap_is_a_capacity_wait(tmp_path: Path, monkeypatch) -> None:
    config = remote_cpu_config(max_live_executions=1)
    world = RemoteCpuWorld(tmp_path, monkeypatch, config=config)
    world.record_environment()
    holder = world.descriptor(label="prep-holder")
    leases.claim_handoff(world.root, descriptor=holder, config=config, now=T0)
    transport = {"transport_object": f"gs://{TRANSPORT_BUCKET}/transport/{holder['job_id']}/"
                                     f"{holder['attempt_id']}-{'a' * 32}.json",
                 "transport_generation": 7, "write_urls_expire_at_epoch": T0 + 2520,
                 "read_urls_expire_at_epoch": T0 + 1020}
    leases.transition(world.root, holder["job_id"], attempt_id=holder["attempt_id"], to_state="dispatching", now=T0,
                      updates=transport)
    assert leases.slots_in_use(world.root) == 1

    waiting = world.descriptor(label="prep-waiting")
    result = world.run("dispatch", descriptor=waiting)
    assert (result["status"], result["blockers"], result["success"]) == (
        "awaiting_capacity", ["remote_cpu_live_execution_cap_reached"], False)
    assert world.lease(waiting)["state"] == "awaiting_capacity" and leases.slots_in_use(world.root) == 1
    assert world.consumed() == [] and world.rest.mutations == [] and world.store.presigned == []

    # The holder stops being live long before its slot frees: only its provider zero frees it.
    world.clock.advance(4000)
    assert leases.live_leases(world.root, now=world.clock.now) == []
    assert world.run("dispatch", descriptor=waiting)["status"] == "awaiting_capacity"
    assert world.lease(waiting)["state"] == "awaiting_capacity"

    leases.expire_stale(world.root, now=world.clock.now)
    teardown = records.teardown_record(
        descriptor=holder, worker_identity=None, outcome="start_timeout",
        compute={"execution_completed": False, "running_count": 0, "listing_complete": True, "listing_pages": 1,
                 "executions_for_attempt": 0, "unfinished_executions_for_attempt": 0,
                 "transport_object": transport["transport_object"], "transport_generation": 7,
                 "transport_deleted": True, "transport_absent_at_generation": True},
        provider={"staging_versions_deleted": 0, "staging_versions_remaining": 0, "staging_listing_complete": True,
                  "write_urls_expire_at_epoch": T0 + 2520, "read_urls_expire_at_epoch": T0 + 1020},
        observed_at_epoch=world.clock.now)
    leases.transition(world.root, holder["job_id"], attempt_id=holder["attempt_id"], to_state="abandoned_dispatch",
                      now=world.clock.now, updates={"teardown": teardown})
    assert leases.slots_in_use(world.root) == 0

    admitted = world.run("dispatch", descriptor=waiting)
    assert admitted["admission"]["status"] == "admitted"
    assert world.consumed() == [_consumption(world, waiting)]


def test_non_us_config_or_b2_region_is_refused(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    descriptor = world.descriptor()

    world.write_config(remote_cpu_config(region="europe-west1"))
    assert allocator.load_remote_cpu_config(world.config_path)[1] == ["remote_cpu_config_region_not_us"]
    refused = world.run("dispatch", descriptor=descriptor)
    assert "remote_cpu_config_region_not_us" in refused["blockers"] and refused["success"] is False
    world.assert_untouched()

    world.write_config(remote_cpu_config())
    world.runtime.object_store = (world.store, B2_BUCKET, "eu-central-003")
    refused = world.run("dispatch", descriptor=descriptor)
    assert "remote_cpu_object_store_region_not_us" in refused["blockers"]
    world.assert_untouched()

    # Production reads the region from the dedicated B2 settings, and refuses a non-US one.
    settings = tmp_path / "b2"
    settings.mkdir()
    for name, value in {"ACCESS_KEY_ID": "id", "SECRET_ACCESS_KEY": "secret", "BUCKET": B2_BUCKET,
                        "ENDPOINT_URL": "https://s3.eu-central-003.backblazeb2.com", "REGION": "eu-central-003"}.items():
        (settings / name).write_text(value, encoding="utf-8")
        (settings / name).chmod(0o600)
        monkeypatch.setenv(f"BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_{name}_FILE", str(settings / name))
    world.runtime.object_store = None
    refused = world.run("dispatch", descriptor=descriptor)
    assert "remote_cpu_object_store_region_not_us" in refused["blockers"]
    world.assert_untouched()
    # The region is not self-declared: it must be the one the endpoint serves.
    (settings / "REGION").chmod(0o600)
    (settings / "REGION").write_text("us-west-004", encoding="utf-8")
    world.runtime.object_store = None
    unbound = world.run("dispatch", descriptor=descriptor)
    assert unbound["blockers"] == ["remote_cpu_object_store_region_unbound"]
    world.assert_untouched()


def _transport(world: RemoteCpuWorld, descriptor: dict) -> tuple[str, int, dict]:
    lease = world.lease(descriptor)
    name = lease["transport_object"].removeprefix(f"gs://{TRANSPORT_BUCKET}/")
    generation = lease["transport_generation"]
    return name, generation, json.loads(world.bucket.reader().get(name, generation=generation))


def test_presigned_puts_exist_only_after_admission_and_expire_at_the_hard_deadline(tmp_path: Path,
                                                                                   monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch, with_authority=False)
    world.record_environment()
    descriptor = world.descriptor()
    assert world.run("dispatch", descriptor=descriptor)["status"] == "blocked"
    world.assert_untouched()
    # Without the grant bound to this attempt the mint presigns nothing and writes nothing.
    binding = "sha256:" + "b" * 64
    for grant in (None, allocator.admit_remote_cpu_job(blockers=[], binding={"x": 1}, execute=True)[1]):
        with pytest.raises(PaidResourceAdmissionBlocked):
            allocator.mint_transport(
                grant=grant, binding_digest=binding, descriptor=descriptor, bucket=world.bucket,
                object_store=world.runtime.object_store, clock=world.clock,
                object_uri=f"gs://{TRANSPORT_BUCKET}/transport/{descriptor['job_id']}/{descriptor['attempt_id']}-{'a' * 32}.json")
    world.assert_untouched()

    world.write_authority(standing_authority())
    assert world.run("dispatch", descriptor=descriptor)["admission"]["status"] == "admitted"
    lease = world.lease(descriptor)
    hard, fetch_end = T0 + 600 + 1800 + 120, T0 + 600 + 300 + 120
    # The URLs stop at the hard deadline and the fetch window; the recorded bounds add a 300 s margin.
    assert (lease["write_urls_expire_at_epoch"], lease["read_urls_expire_at_epoch"]) == (hard + 300, fetch_end + 300)
    assert lease["deadlines"]["hard_deadline_epoch"] == hard
    staging = descriptor["outputs"]["staging_prefix"].removeprefix(f"s3://{B2_BUCKET}/")
    puts = [(key, seconds) for method, key, seconds in world.store.presigned if method == "put_object"]
    gets = [(key, seconds) for method, key, seconds in world.store.presigned if method == "get_object"]
    assert sorted(key.removeprefix(staging) for key, _ in puts) == [
        "blobs.tar", "heartbeat.json", "index.json", "receipt.json"]
    # Data GETs (inputs and source) live through the fetch window; the receipt GET, which the worker checks
    # before every receipt it writes so that it never overwrites a committed one, lives as long as the PUTs.
    assert {seconds for _, seconds in puts} == {2520}
    assert sorted(gets) == sorted([
        *((item["uri"].removeprefix(f"s3://{B2_BUCKET}/"), 1020) for item in descriptor["inputs"]),
        (descriptor["code"]["source_archive"]["uri"].removeprefix(f"s3://{B2_BUCKET}/"), 1020),
        (staging + "receipt.json", 2520)])

    _, _, transport = _transport(world, descriptor)
    assert transport["schema_version"] == "remote_cpu_job_transport.v1" and transport["descriptor"] == descriptor
    heartbeat, source = transport["outputs"]["heartbeat.json"], transport["source_archive"]["url"]
    world.clock.now = fetch_end - 1
    assert world.store.request("GET", source).status != 403
    assert world.store.request("PUT", heartbeat, body=b"{}").status == 200
    world.clock.now = fetch_end
    assert world.store.request("GET", source).status == 403
    assert world.store.request("GET", transport["receipt_url"]).status == 404  # still answers: none is up yet
    world.clock.now = hard - 1
    assert world.store.request("PUT", heartbeat, body=b"{}").status == 200
    assert world.store.request("GET", transport["receipt_url"]).status == 404
    world.clock.now = hard
    assert world.store.request("PUT", heartbeat, body=b"{}").status == 403
    assert world.store.request("GET", transport["receipt_url"]).status == 403


def test_transport_is_read_only_at_its_generation(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    descriptor = world.descriptor()
    world.run("dispatch", descriptor=descriptor)
    name, generation, transport = _transport(world, descriptor)
    assert name.startswith(f"transport/{descriptor['job_id']}/{descriptor['attempt_id']}-")
    reader = world.bucket.reader()
    assert not any(hasattr(reader, verb) for verb in ("create", "delete", "list", "exists"))
    assert transport["descriptor"]["descriptor_digest"] == descriptor["descriptor_digest"]
    with pytest.raises(FakeGcsError) as other_generation:
        reader.get(name, generation=generation + 1)
    assert other_generation.value.code == 404
    # Create-if-absent: the same name is never overwritten in place ...
    with pytest.raises(FakeGcsError) as overwrite:
        world.bucket.create(name, b"{}", if_generation_match=0)
    assert overwrite.value.code == 412
    # ... and a replacement takes a new generation, so the worker's pinned read fails.
    world.bucket.delete(name, generation=generation)
    assert world.bucket.create(name, b"{}", if_generation_match=0) != generation
    with pytest.raises(FakeGcsError):
        reader.get(name, generation=generation)
    lease_text = (world.root / "leases" / f"{descriptor['job_id']}.json").read_text(encoding="utf-8")
    assert "X-Amz-" not in lease_text and "https://" not in lease_text

    # The production adapter creates only if absent and reads and deletes only at a generation.
    calls: list[tuple] = []

    class Blob:
        def __init__(self, blob_name: str, generation: int | None) -> None:
            self.name, self.pinned, self.generation = blob_name, generation, None

        def upload_from_string(self, data: bytes, *, content_type: str, if_generation_match: int | None) -> None:
            calls.append(("create", self.name, if_generation_match, content_type))
            self.generation = 1234

        def download_as_bytes(self) -> bytes:
            calls.append(("get", self.name, self.pinned))
            return b"{}"

        def exists(self) -> bool:
            calls.append(("exists", self.name, self.pinned))
            return False

        def delete(self) -> None:
            calls.append(("delete", self.name, self.pinned))

    class Client:
        def __init__(self, *, project: str, credentials: object) -> None:
            calls.append(("client", project))

        def bucket(self, bucket_name: str):
            calls.append(("bucket", bucket_name))
            return type("Bucket", (), {"blob": staticmethod(lambda blob_name, generation=None: Blob(blob_name, generation))})()

    # Hermetic even where google.* cannot be imported: the adapter's lazy import meets these stubs.
    storage = types.ModuleType("google.cloud.storage")
    storage.Client = Client
    cloud = types.ModuleType("google.cloud")
    cloud.storage = storage
    google = types.ModuleType("google")
    google.cloud = cloud
    for name, module in (("google", google), ("google.cloud", cloud), ("google.cloud.storage", storage)):
        monkeypatch.setitem(sys.modules, name, module)
    bucket = GcsTransportBucket(TRANSPORT_BUCKET, credentials=object(), project="blueprint-8c1ca")
    assert bucket.create("transport/x.json", b"{}", if_generation_match=0) == 1234
    assert bucket.get("transport/x.json", generation=1234) == b"{}"
    assert bucket.exists("transport/x.json", generation=1234) is False
    bucket.delete("transport/x.json", generation=1234)
    assert calls == [("client", "blueprint-8c1ca"), ("bucket", TRANSPORT_BUCKET),
                     ("create", "transport/x.json", 0, "application/json"), ("get", "transport/x.json", 1234),
                     ("exists", "transport/x.json", 1234), ("delete", "transport/x.json", 1234)]


def texts_written(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): path.read_bytes().decode("utf-8", "replace")
            for path in sorted(root.rglob("*")) if path.is_file()}


def test_no_url_reaches_host_disk_or_logs(tmp_path: Path, monkeypatch, caplog, capsys) -> None:
    caplog.set_level(logging.DEBUG)
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    descriptor = world.descriptor()
    world.run("dispatch", descriptor=descriptor, execute=False)
    world.run("dispatch", descriptor=descriptor)
    world.run("dispatch", descriptor=descriptor)
    _, _, transport = _transport(world, descriptor)
    assert "X-Amz-Signature=" in json.dumps(transport)  # the authority exists, but only in its GCS object
    # Every other writer too: the probe's descriptor, teardown, environment and settlement, and cancel and sweep.
    FakeWorker(world, environment=environment())
    assert world.run("preflight")["status"] == "completed"
    assert world.run("cancel", descriptor=descriptor)["status"] == "cancelled"
    assert world.run("sweep")["status"] == "swept"
    assert any(name.startswith("remote-cpu-jobs/teardowns/") for name in texts_written(tmp_path))

    captured = capsys.readouterr()
    texts = {"log": caplog.text, "stdout": captured.out, "stderr": captured.err, **texts_written(tmp_path)}
    assert {"remote-cpu-jobs/environment/episode_compilation.json"} <= set(texts)
    assert any(name.startswith("spend-authority/remote-cpu-settled/") for name in texts)
    for name, text in texts.items():
        assert "X-Amz-" not in text and "backblazeb2" not in text and "https://" not in text, name


def test_object_store_writes_name_only_attempt_staging_and_promote_server_side(tmp_path: Path, monkeypatch) -> None:
    store = RecordingArtifactStore(clock=FakeClock(T0), bucket=B2_BUCKET, max_copy_bytes=1024, min_part_bytes=256)
    binding = {"schema_version": "remote_cpu_job_allocation_binding.v1"}
    grant = allocator.admit_remote_cpu_job(blockers=[], binding=binding, execute=True)[1]
    job_id = "rcj-ec-" + "a" * 24
    staging = f"{OBJECT_PREFIX}/remote-cpu/staging/{job_id}/{job_id}-a1-{'b' * 32}/"
    key = staging.removeprefix(f"s3://{B2_BUCKET}/")
    for uri in (f"{CAS}/remote-cpu-output/sha256/{'c' * 64}/blobs.tar", staging.replace(B2_BUCKET, "other-bucket") + "x",
                staging.replace("/staging/", "/stage/") + "x", staging + "nested/x", staging + "..", staging + "x?versionId=1",
                staging.replace(f"{job_id}-a1", "rcj-ec-" + "f" * 24 + "-a1") + "x"):
        with pytest.raises(ObjectStoreError) as refused:
            object_store.presign_remote_cpu_put(grant=grant, binding_digest=canonical_digest(binding), staging_uri=uri,
                                                expires_in_seconds=60, client=store, bucket=B2_BUCKET)
        assert str(refused.value) == "remote_cpu_object_name_invalid"
        with pytest.raises(ObjectStoreError):
            object_store.copy_remote_cpu_staging_to_cas(
                staging_uri=uri, digest="sha256:" + "c" * 64, size_bytes=1, etag='"e"', artifact_kind="remote-cpu-output",
                filename="blobs.tar", client=store, bucket=B2_BUCKET)
    for prefix in (f"{CAS}/", staging + "x", staging.rstrip("/")):
        with pytest.raises(ObjectStoreError):
            object_store.delete_remote_cpu_staging_versions(staging_prefix=prefix, client=store, bucket=B2_BUCKET)
    assert store.presigned == [] and store.operations == []

    # Promotion is a server-side copy guarded by the staging ETag; above the single-copy limit it is
    # UploadPartCopy.  Either way no byte crosses the host.
    promoted = {}
    for name, data in (("index.json", b"i" * 100), ("blobs.tar", bytes(range(256)) * 12)):
        etag = store.put_object(Bucket=B2_BUCKET, Key=key + name, Body=data)["ETag"]
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        arguments = {"staging_uri": staging + name, "digest": digest, "size_bytes": len(data),
                     "artifact_kind": "remote-cpu-output", "filename": name, "client": store, "bucket": B2_BUCKET,
                     "single_copy_limit": 1024, "part_bytes": 1024}
        with pytest.raises(ObjectStoreError) as changed:
            object_store.copy_remote_cpu_staging_to_cas(etag='"stale"', **arguments)
        assert str(changed.value) == "remote_cpu_staging_changed"
        moved = (store.bytes_sent_to_client, store.bytes_received_from_client)
        reference = object_store.copy_remote_cpu_staging_to_cas(etag=etag, **arguments)
        assert (store.bytes_sent_to_client, store.bytes_received_from_client) == moved
        assert (reference["status"], reference["uri"]) == ("copied", f"{CAS}/remote-cpu-output/sha256/{digest[7:]}/{name}")
        cas_key = reference["uri"].removeprefix(f"s3://{B2_BUCKET}/")
        assert store.head_object(Bucket=B2_BUCKET, Key=cas_key)["Metadata"] == {"sha256": digest[7:]}
        assert store.get_object(Bucket=B2_BUCKET, Key=cas_key)["Body"].read() == data
        assert object_store.copy_remote_cpu_staging_to_cas(etag=etag, **arguments)["status"] == "already_present"
        promoted[name] = cas_key
    assert ("CopyObject", B2_BUCKET, promoted["index.json"]) in store.operations
    assert ("CompleteMultipartUpload", B2_BUCKET, promoted["blobs.tar"]) in store.operations

    # A plain delete only hides a B2 object; every version and marker goes, page by page.
    store.delete_object(Bucket=B2_BUCKET, Key=key + "index.json")
    deleted = object_store.delete_remote_cpu_staging_versions(staging_prefix=staging, client=store, bucket=B2_BUCKET,
                                                              page_size=1)
    assert (deleted["versions_deleted"], deleted["versions_remaining"], deleted["listing_complete"]) == (3, 0, True)
    listing = store.list_object_versions(Bucket=B2_BUCKET, Prefix=key)
    assert listing["Versions"] == [] and listing["DeleteMarkers"] == []
    assert all(store.head_object(Bucket=B2_BUCKET, Key=cas_key) for cas_key in promoted.values())

    # The production client sends and validates flexible checksums only when an API requires them.
    settings = tmp_path / "b2"
    settings.mkdir()
    for name, value in {"ACCESS_KEY_ID": "id", "SECRET_ACCESS_KEY": "secret", "BUCKET": B2_BUCKET,
                        "ENDPOINT_URL": "https://s3.us-west-004.backblazeb2.com", "REGION": "us-west-004"}.items():
        (settings / name).write_text(value, encoding="utf-8")
        (settings / name).chmod(0o600)
        monkeypatch.setenv(f"BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_{name}_FILE", str(settings / name))
    client, bucket, region = object_store.remote_cpu_object_store()
    assert (bucket, region) == (B2_BUCKET, "us-west-004")
    assert client.meta.config.request_checksum_calculation == "when_required"
    assert client.meta.config.response_checksum_validation == "when_required"


ATTEMPT_ENV = "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID"


def _run_directly(world: RemoteCpuWorld, attempt: str, behaviour: str = "hang") -> dict:
    """An execution Cloud Run holds that no dispatch of this host created (a foreign or orphaned run)."""

    world.jobs.script(behaviour)
    overrides = {"containerOverrides": [{"env": [{"name": ATTEMPT_ENV, "value": attempt}]}], "taskCount": 1,
                 "timeout": "1800s"}
    return world.jobs.run_job(JOB, etag=None, overrides=overrides)["metadata"]


def _runs(world: RemoteCpuWorld, attempt_id: str) -> list:
    """The :run requests (not validate_only) that named this attempt."""

    return [request for request in world.rest.runs(validate_only=False) if attempt_id.encode() in request.body]


def _attempt_executions(world: RemoteCpuWorld, attempt_id: str) -> list[dict]:
    return [world.jobs.get_execution(row["name"]) for row in world.jobs.executions[JOB]
            if env_value(world.jobs.get_execution(row["name"]), ATTEMPT_ENV) == attempt_id]


def test_ambiguous_run_job_reconciles_across_all_pages_and_never_double_dispatches(tmp_path: Path,
                                                                                  monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch, config=remote_cpu_config(max_live_executions=4))
    world.record_environment()
    rest = world.rest

    class Crowded(CloudRunRest):
        """The response to :run is lost, and five other runs start before the host lists."""

        def __call__(self, method: str, url: str, *, body, headers):
            try:
                return rest(method, url, body=body, headers=headers)
            except ConnectionResetError:
                for index in range(5):
                    _run_directly(world, f"other-{index}", "succeed")
                raise

    world.runtime.cloud_run._transport = Crowded(world.jobs)
    world.jobs.script("lost_response")
    descriptor = world.descriptor(label="prep-lost")
    lists_before = world.jobs.list_calls
    result = world.run("dispatch", descriptor=descriptor)
    assert (result["status"], result["success"]) == ("dispatched", True)
    [ours] = _attempt_executions(world, descriptor["attempt_id"])
    # Newest first, two to a page: the lost run sits on the last of three pages.
    assert result["reconciled"]["listing_pages"] == 3 and world.jobs.list_calls - lists_before == 3
    assert result["worker_identity"].endswith("/executions/" + ours["name"].rsplit("/", 1)[1])
    assert world.lease(descriptor)["state"] == "dispatched"
    assert len(_runs(world, descriptor["attempt_id"])) == 1

    # Neither a repeated dispatch nor a reconcile re-issues the run.
    assert world.run("dispatch", descriptor=descriptor)["status"] == "already_dispatched"
    assert world.run("reconcile", descriptor=descriptor)["status"] == "nothing_to_reconcile"
    assert len(_runs(world, descriptor["attempt_id"])) == 1
    assert len(_attempt_executions(world, descriptor["attempt_id"])) == 1

    # A request lost before Cloud Run saw it: a complete listing finds nothing, so the attempt is
    # abandoned - its transport deleted first - and never run again.
    def dropped(method: str, url: str, *, body, headers):
        if url.endswith(":run") and not json.loads(body).get("validateOnly"):
            rest.requests.append(type(rest.requests[0])(method, url, dict(headers), body))
            raise ConnectionResetError("reset before the request reached Cloud Run")
        return rest(method, url, body=body, headers=headers)

    world.runtime.cloud_run._transport = dropped
    lost = world.descriptor(label="prep-dropped")
    result = world.run("dispatch", descriptor=lost)
    assert result["status"] == "teardown_pending" and result["success"] is False
    assert {"remote_cpu_dispatch_lost", "remote_cpu_provider_zero_unproven"} <= set(result["blockers"])
    lease = world.lease(lost)
    assert (lease["state"], lease["worker_identity"], lease["compute_zero_proven"]) == ("dispatching", None, True)
    transport = lease["transport_object"].split("/", 3)[3]
    assert not world.bucket.exists(transport, generation=lease["transport_generation"])
    assert leases.slots_in_use(world.root) == 2  # the abandoned attempt keeps its slot until provider zero

    # An unreadable listing leaves the attempt unresolved rather than guessing.
    world.clock.now = lease["write_urls_expire_at_epoch"]
    world.runtime.cloud_run._transport = lambda method, url, *, body, headers: (503, b"{}")
    unresolved = world.run("reconcile", descriptor=lost)
    assert unresolved["status"] == "ambiguous_dispatch_unresolved"
    assert "remote_cpu_ambiguous_dispatch_unresolved" in unresolved["blockers"]

    world.runtime.cloud_run._transport = rest
    abandoned = world.run("reconcile", descriptor=lost)
    assert (abandoned["status"], abandoned["success"]) == ("abandoned_dispatch", True)
    assert world.lease(lost)["provider_zero_proven"] is True and leases.slots_in_use(world.root) == 1
    assert _attempt_executions(world, lost["attempt_id"]) == [] and len(_runs(world, lost["attempt_id"])) == 1
    teardown = records.validate_teardown(json.loads(
        (world.root / "teardowns" / f"{lost['attempt_id']}.json").read_text(encoding="utf-8")))
    assert (teardown["worker_identity"], teardown["compute_zero"]["executions_for_attempt"]) == (None, 0)
    settled = json.loads((world.spend / "remote-cpu-settled" / f"{_consumption(world, lost).name[11:]}")
                         .read_text(encoding="utf-8"))
    assert (settled["settled_usd"], settled["basis"]) == (0.0, "no_execution")


def test_preflight_probe_is_a_granted_leased_and_torn_down_attempt(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    worker = environment(cpu_class="sha256:" + "e" * 64)
    FakeWorker(world, environment=worker)
    assert "remote_cpu_environment_unrecorded" in world.run("dispatch", descriptor=world.descriptor())["blockers"]
    required: list[dict] = []
    require = allocator.require_paid_resource_admission
    monkeypatch.setattr(allocator, "require_paid_resource_admission",
                        lambda admission, **kwargs: required.append(dict(admission)) or require(admission, **kwargs))

    source = world.runtime.stage_release_source
    world.runtime.stage_release_source = None
    unstaged = world.run("preflight")
    assert unstaged["blockers"] == ["remote_cpu_release_source_staging_unavailable"]
    world.assert_untouched()
    world.runtime.stage_release_source = source

    result = world.run("preflight")
    assert (result["status"], result["success"]) == ("completed", True), result["blockers"]
    probe = result["probe"]
    descriptor = json.loads((world.root / "descriptors" / f"{probe['attempt_id']}.json").read_text(encoding="utf-8"))
    assert descriptor["stage"] == "environment_probe" and descriptor["execution"]["job"] == JOB.rsplit("/", 1)[1]
    assert descriptor["code"]["environment_digest"] == HOST_RECORD["environment_digest"]

    # Granted: one bound admission, validate_only before the only run.
    [admission] = required
    assert admission["allocation_binding"]["attempt_id"] == probe["attempt_id"] == result["admission"][
        "allocation_binding"]["attempt_id"]
    assert [request.json()["validateOnly"] for request in world.rest.runs()] == [True, False]
    [execution] = world.jobs.executions[JOB]
    assert world.consumed() == [_consumption(world, descriptor)]
    assert result["sentinels"]["transport"]["status"] == "passed"
    assert result["sentinels"]["object_store"]["status"] == "passed"
    assert result["sentinels"]["validate_only"]["status"] == "passed"

    # Leased and torn down: provider zero only once the recorded write URLs expired.
    lease = json.loads((world.root / "leases" / f"{probe['job_id']}.json").read_text(encoding="utf-8"))
    assert (lease["stage"], lease["state"], lease["dispatch_started"]) == ("environment_probe", "completed", True)
    assert lease["worker_identity"].endswith("/executions/" + execution["name"].rsplit("/", 1)[1])
    assert lease["heartbeat"]["sequence"] >= 1 and leases.slots_in_use(world.root) == 0
    teardown = records.validate_teardown(json.loads(
        (world.root / "teardowns" / f"{probe['attempt_id']}.json").read_text(encoding="utf-8")))
    assert teardown["compute_zero_proven"] and teardown["provider_zero_proven"]
    assert teardown["observed_at_epoch"] >= lease["write_urls_expire_at_epoch"]
    assert result["teardown"]["teardown_digest"] == teardown["teardown_digest"] == lease["teardown_digest"]
    settled = json.loads((world.spend / "remote-cpu-settled" / _consumption(world, descriptor).name[11:])
                         .read_text(encoding="utf-8"))
    assert settled["basis"] == "execution_runtime" and 0 < settled["settled_usd"] < WORST_CASE_USD
    assert world.bucket._objects == {}
    staging = descriptor["outputs"]["staging_prefix"].removeprefix(f"s3://{B2_BUCKET}/")
    listing = world.store.list_object_versions(Bucket=B2_BUCKET, Prefix=staging)
    assert listing["Versions"] == [] and listing["DeleteMarkers"] == []

    # The worker environment is recorded with its parity against the host, and dispatch accepts it.
    recorded = json.loads((world.root / "environment" / "episode_compilation.json").read_text(encoding="utf-8"))
    assert (recorded["environment_digest"], recorded["image"], recorded["probe_attempt_id"]) == (
        worker["environment_digest"], IMAGE, probe["attempt_id"])
    assert recorded["parity"] == {"python_version_info": True, "golden_deflate": True, "golden_simd": True,
                                  "distributions": True, "cpu_class": False}
    dispatched = world.run("dispatch", descriptor=world.descriptor(environment_digest=worker["environment_digest"]))
    assert dispatched["status"] == "dispatched"

    # A preflight whose process dies mid-poll leaves its probe running; reconcile closes it only once
    # a live preflight would have finished, and frees its slot.
    polls = []

    def dies(seconds: float) -> None:
        world.clock.advance(seconds)
        polls.append(seconds)
        if len(polls) == 3:
            raise KeyboardInterrupt("the preflight process was killed")

    world.runtime.sleep = dies
    with pytest.raises(KeyboardInterrupt):
        world.run("preflight")
    world.runtime.sleep = world.sleep
    [interrupted] = [path for path in (world.root / "descriptors").iterdir() if probe["attempt_id"] not in path.name]
    orphan = json.loads(interrupted.read_text(encoding="utf-8"))
    assert world.lease(orphan)["state"] == "dispatched" and leases.slots_in_use(world.root) == 2
    world.clock.advance(600)
    assert world.run("reconcile", descriptor=orphan)["status"] == "nothing_to_reconcile"
    world.clock.now = world.lease(orphan)["write_urls_expire_at_epoch"] + 120
    closed = world.run("reconcile", descriptor=orphan)
    assert (closed["status"], closed["success"]) == ("blocked", False) and closed["teardown"]["provider_zero_proven"]
    assert world.lease(orphan)["state"] == "blocked" and leases.slots_in_use(world.root) == 1


def test_cancel_and_sweep_are_termination_only(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    world.jobs.script("hang")
    descriptor = world.descriptor()
    assert world.run("dispatch", descriptor=descriptor)["status"] == "dispatched"
    orphan = _run_directly(world, "rcj-ec-" + "f" * 24 + "-a1-" + "0" * 32)
    before = (len(world.rest.runs()), len(world.store.presigned), len(world.store.operations), world.consumed())
    admissions: list[dict] = []
    monkeypatch.setattr(allocator, "build_paid_lane_admission", lambda **kwargs: admissions.append(kwargs))

    dry = world.run("cancel", descriptor=descriptor, execute=False)
    assert (dry["status"], len(dry["would_cancel"])) == ("dry_run_ready", 1)
    assert all(not row.get("completionTime") for rows in world.jobs.executions.values()
               for row in map(world.jobs.get_execution, [r["name"] for r in rows]))
    cancelled = world.run("cancel", descriptor=descriptor)
    assert cancelled["status"] == "cancelled" and len(cancelled["cancelled"]) == 1
    [ours] = _attempt_executions(world, descriptor["attempt_id"])
    assert ours["cancelledCount"] == 1
    swept = world.run("sweep")
    assert swept["status"] == "swept" and swept["cancelled"] == [orphan["name"].rsplit("/", 1)[1]]

    assert admissions == []
    assert (len(world.rest.runs()), len(world.store.presigned), len(world.store.operations), world.consumed()) == before
    assert {request.url.rsplit(":", 1)[-1] for request in world.rest.mutations if "/executions/" in request.url} == {
        "cancel"}
    assert len(world.jobs.executions[JOB]) == 2

    # Structurally: nothing cancel or sweep can reach admits, mints, consumes or runs.
    tree = ast.parse(Path(allocator.__file__).read_text(encoding="utf-8"))
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    def reachable(name: str, seen: set[str]) -> set[str]:
        calls = {getattr(call.func, "id", getattr(call.func, "attr", "")) for call in ast.walk(functions[name])
                 if isinstance(call, ast.Call)}
        for callee in calls & set(functions) - seen:
            seen.add(callee)
            calls |= reachable(callee, seen)
        return calls

    for entry in ("cancel_remote_cpu_attempt", "sweep_remote_cpu_stage"):
        assert not reachable(entry, {entry}) & {"run_job", "mint_transport", "create", "admit_remote_cpu_job",
                                                 "consume_remote_cpu_authority_once", "presign_remote_cpu_put"}


def test_sweep_cancels_running_executions_without_a_live_lease(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch, config=remote_cpu_config(max_live_executions=4))
    world.record_environment()
    finished = _run_directly(world, "rcj-ec-" + "1" * 24 + "-a1-" + "1" * 32, "succeed")
    world.jobs.script("hang")
    stale = world.descriptor(label="prep-stale")
    assert world.run("dispatch", descriptor=stale)["status"] == "dispatched"
    world.clock.advance(700)  # no heartbeat: the stale attempt's lease expired at its start allowance
    world.jobs.script("hang", "duplicate")
    live = world.descriptor(label="prep-live")
    duplicated = world.descriptor(label="prep-duplicated")
    assert world.run("dispatch", descriptor=live)["status"] == "dispatched"
    assert world.run("dispatch", descriptor=duplicated)["status"] == "dispatched"
    foreign = _run_directly(world, "not-a-remote-cpu-attempt")
    world.clock.advance(10)

    def short(row: dict) -> str:
        return row["name"].rsplit("/", 1)[1]

    [stale_run] = _attempt_executions(world, stale["attempt_id"])
    [live_run] = _attempt_executions(world, live["attempt_id"])
    first, second = sorted(_attempt_executions(world, duplicated["attempt_id"]), key=lambda row: row["createTime"])
    recorded = world.lease(duplicated)["worker_identity"].rsplit("/", 1)[1]
    kept, extra = (first, second) if recorded == short(first) else (second, first)
    assert {row["attempt_id"] for row in leases.live_leases(world.root, now=world.clock.now)} == {
        live["attempt_id"], duplicated["attempt_id"]}

    dry = world.run("sweep", execute=False)
    assert sorted(dry["would_cancel"]) == sorted(map(short, (stale_run, extra, foreign)))
    swept = world.run("sweep")
    assert sorted(swept["cancelled"]) == sorted(map(short, (stale_run, extra, foreign)))
    assert swept["listing_pages"] == 3 and swept["executions_listed"] == 6
    views = {short(row): world.jobs.get_execution(row["name"]) for row in world.jobs.executions[JOB]}
    assert {name for name, view in views.items() if view["cancelledCount"]} == set(map(short, (stale_run, extra, foreign)))
    assert views[short(live_run)]["runningCount"] == 1 and views[short(kept)]["runningCount"] == 1
    assert views[short(finished)]["succeededCount"] == 1 and len(world.rest.runs(validate_only=False)) == 3


def test_remote_cpu_job_stdout_is_success_only(tmp_path: Path, monkeypatch, capsys) -> None:
    from blueprint_pipeline import paid_resource_allocator

    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    monkeypatch.setattr(allocator, "RemoteCpuRuntime", lambda: world.runtime)
    descriptor = world.descriptor()
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")

    def main(*arguments: str) -> tuple[int, str, str]:
        code = paid_resource_allocator.main(["remote-cpu-job", "--stage", "episode_compilation",
                                             "--lease", str(world.root), "--out", str(tmp_path / "out.json"),
                                             *arguments])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    assert main("--action", "dispatch", "--descriptor", str(path), "--execute") == (0, '{"success": true}\n', "")
    record = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert record["status"] == "dispatched" and record["success"] is True
    assert main("--action", "dispatch", "--execute") == (2, '{"success": false}\n', "")
    assert main("--action", "sweep", "--execute") == (0, '{"success": true}\n', "")
    world.write_config(remote_cpu_config(region="europe-west1"))
    assert main("--action", "cancel", "--descriptor", str(path), "--execute") == (2, '{"success": false}\n', "")
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["blockers"] == [
        "remote_cpu_config_region_not_us"]


def test_allocator_and_remote_cpu_modules_stay_within_their_line_budgets() -> None:
    root = Path(allocator.__file__).resolve().parents[2]
    policy = json.loads((root / "docs/source_governance_policy.json").read_text(encoding="utf-8"))
    canonical = "src/blueprint_pipeline/paid_resource_allocator.py"
    budgets = {canonical: policy["grandfathered_module_line_limits"][canonical],
               "src/blueprint_pipeline/remote_cpu_job_allocator.py": 1000,
               "src/blueprint_pipeline/cloud_run_jobs_client.py": 500}
    for relative, budget in budgets.items():
        assert len((root / relative).read_text(encoding="utf-8").splitlines()) <= budget, relative


def _signed_until(url: str) -> float:
    """When a presigned URL stops working: its signing time plus its lifetime."""

    query = parse_qs(urlsplit(url).query)
    return float(query["X-Amz-Date"][0]) + float(query["X-Amz-Expires"][0])


def _refusing_runs(world: RemoteCpuWorld, *, get_job_seconds: float = 0.0, runs: list | None = None):
    """A Cloud Run that refuses every :run (403) after a ``jobs.get`` that takes ``get_job_seconds``."""

    def transport(method: str, url: str, *, body, headers):
        if method == "GET" and url.endswith(JOB):
            world.clock.advance(get_job_seconds)  # a token refresh, a lock wait or a slow jobs.get
        if url.endswith(":run"):
            if runs is not None:
                runs.append(world.clock.now)
            return 403, b'{"error": {"code": 403, "status": "PERMISSION_DENIED"}}'
        return world.rest(method, url, body=body, headers=headers)

    return transport


def test_recorded_url_expiry_outlasts_every_minted_url_and_dispatch_is_stamped_at_run(tmp_path: Path,
                                                                                      monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    runs: list[float] = []
    minted: dict = {}
    world.runtime.cloud_run._transport = _refusing_runs(world, get_job_seconds=90, runs=runs)
    create = world.bucket.create

    def slow_upload(name, data, *, if_generation_match):
        world.clock.advance(30)
        minted.update(json.loads(data))
        return create(name, data, if_generation_match=if_generation_match)

    world.bucket.create = slow_upload
    descriptor = world.descriptor()
    assert world.run("dispatch", descriptor=descriptor)["status"] == "teardown_pending"
    lease = world.lease(descriptor)
    # The receipt GET lives as long as the PUTs; the recorded read expiry is the data GETs' (plan 14 §11).
    puts, data_gets = [*minted["outputs"].values(), minted["receipt_url"]], [
        minted["source_archive"]["url"], *(row["url"] for row in minted["inputs"])]
    # botocore signs at the real clock: the recorded bounds come from after signing, plus a margin.
    assert lease["write_urls_expire_at_epoch"] >= max(map(_signed_until, puts)) + 300
    assert lease["read_urls_expire_at_epoch"] >= max(map(_signed_until, data_gets)) + 300
    # The dispatch clock starts when :run is sent, not when the action began.
    assert runs == [lease["deadlines"]["dispatch_started_at_epoch"]] and runs[0] >= T0 + 120

    world.clock.now = lease["write_urls_expire_at_epoch"] - 1
    assert world.run("reconcile", descriptor=descriptor)["status"] == "teardown_pending"
    world.clock.now = lease["write_urls_expire_at_epoch"]
    closed = world.run("reconcile", descriptor=descriptor)
    assert closed["status"] == "abandoned_dispatch" and closed["teardown"]["provider_zero_proven"] is True
    # Provider zero is sealed only once no minted write URL, nor the receipt GET, still works.
    assert {world.store.request("PUT", url, body=b"late").status for url in minted["outputs"].values()} == {403}
    assert world.store.request("GET", minted["receipt_url"]).status == 403


def test_teardown_resumes_from_its_sealed_record_after_a_failed_transition(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch, config=remote_cpu_config(max_live_executions=4))
    world.record_environment()
    world.runtime.cloud_run._transport = _refusing_runs(world)
    locked, unsettled = world.descriptor(label="prep-locked"), world.descriptor(label="prep-unsettled")
    for descriptor in (locked, unsettled):
        assert world.run("dispatch", descriptor=descriptor)["status"] == "teardown_pending"
    world.clock.now = max(world.lease(descriptor)["write_urls_expire_at_epoch"] for descriptor in (locked, unsettled))

    # Another holder of the lease lock makes the terminal transition fail after the teardown is sealed.
    held = os.open(world.root / "leases" / f"{locked['job_id']}.lock", os.O_RDWR)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        refused = world.run("reconcile", descriptor=locked)
    finally:
        os.close(held)
    assert refused["blockers"] == [f"remote_cpu_lease_locked:{locked['job_id']}"]
    path = world.root / "teardowns" / f"{locked['attempt_id']}.json"
    sealed = path.read_bytes()
    world.clock.advance(60)
    resumed = world.run("reconcile", descriptor=locked)
    assert (resumed["status"], resumed["success"]) == ("abandoned_dispatch", True)
    assert path.read_bytes() == sealed
    assert world.lease(locked)["teardown_digest"] == json.loads(sealed)["teardown_digest"]

    # A settlement that fails after the lease turned terminal is finished by the next reconcile.
    settle = allocator.settle_remote_cpu_attempt

    def fails_once(**kwargs):
        monkeypatch.setattr(allocator, "settle_remote_cpu_attempt", settle)
        raise OSError("no space left on device")

    monkeypatch.setattr(allocator, "settle_remote_cpu_attempt", fails_once)
    interrupted = world.run("reconcile", descriptor=unsettled)
    assert interrupted["status"] == "blocked" and world.lease(unsettled)["state"] == "abandoned_dispatch"
    finished = world.run("reconcile", descriptor=unsettled)
    assert (finished["status"], finished["success"]) == ("abandoned_dispatch", True)
    assert len(list((world.spend / "remote-cpu-settled").glob("*.json"))) == 2
    assert leases.slots_in_use(world.root) == 0
    assert world.run("reconcile", descriptor=locked)["status"] == "nothing_to_reconcile"


def test_mint_failure_after_consumption_discards_the_transport_and_settles_at_zero(tmp_path: Path,
                                                                                   monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    create, delete = world.bucket.create, world.bucket.delete

    def written_then_lost(name, data, *, if_generation_match):
        create(name, data, if_generation_match=if_generation_match)
        raise TimeoutError("the response was lost after the object was written")

    world.bucket.create = written_then_lost
    first = world.descriptor(label="prep-mint")
    result = world.run("dispatch", descriptor=first)
    assert result["status"] == "blocked" and "remote_cpu_transport_mint_failed:TimeoutError" in result["blockers"]
    assert world.bucket._objects == {} and world.rest.runs() == []  # its live PUT URLs went with it
    lease = world.lease(first)
    assert (lease["state"], lease["dispatch_started"], lease["transport_object"]) == ("fallback_host", False, None)
    assert [(row["attempt_id"], row["usd"]) for row in allocator.spend_ledger()] == [(first["attempt_id"], 0.0)]
    assert world.run("dispatch", descriptor=first)["status"] == "already_dispatched"

    # When even the discard fails, the transport was named with the consumption before it was created,
    # so reconcile deletes it, verifies it is gone and settles once the bucket answers again.
    def unavailable(name, *, generation=None):
        raise ConnectionError("the bucket did not answer")

    world.bucket.delete = unavailable
    second = world.descriptor(label="prep-stranded")
    stranded = world.run("dispatch", descriptor=second)
    assert "remote_cpu_transport_discard_unproven" in stranded["blockers"]
    assert world.lease(second)["state"] == "claimed" and len(world.bucket._objects) == 1
    assert sorted(row["usd"] for row in allocator.spend_ledger()) == [0.0, WORST_CASE_USD]
    world.bucket.create, world.bucket.delete = create, delete
    recovered = world.run("reconcile", descriptor=second)
    assert (recovered["status"], recovered["success"]) == ("fallback_host", True)
    assert world.bucket._objects == {} and world.lease(second)["state"] == "fallback_host"
    assert sorted(row["usd"] for row in allocator.spend_ledger()) == [0.0, 0.0]


def test_dispatch_refuses_a_job_whose_cpu_or_memory_differs_from_the_descriptor(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    world.jobs.jobs[JOB]["template"]["template"]["containers"][0]["resources"] = {"limits": {"cpu": "8", "memory": "32Gi"}}
    result = world.run("dispatch", descriptor=world.descriptor())
    assert {"remote_cpu_job_definition_invalid:cpu", "remote_cpu_job_definition_invalid:memory"} <= set(
        result["admission"]["blockers"])
    world.assert_untouched()


def test_remote_cpu_put_presigns_require_the_attempts_grant(tmp_path: Path, monkeypatch) -> None:
    store = RecordingArtifactStore(clock=FakeClock(T0), bucket=B2_BUCKET)
    job_id = "rcj-ec-" + "a" * 24
    staging = f"{OBJECT_PREFIX}/remote-cpu/staging/{job_id}/{job_id}-a1-{'b' * 32}/"
    binding = {"schema_version": "remote_cpu_job_allocation_binding.v1", "attempt_id": f"{job_id}-a1-{'b' * 32}"}
    digest = canonical_digest(binding)
    grant = allocator.admit_remote_cpu_job(blockers=[], binding=binding, execute=True)[1]
    other = allocator.admit_remote_cpu_job(blockers=[], binding={**binding, "attempt_id": "other"}, execute=True)[1]

    def put(**changes):
        arguments = {"grant": grant, "binding_digest": digest, "staging_uri": staging + "receipt.json",
                     "expires_in_seconds": 2520, "client": store, "bucket": B2_BUCKET, **changes}
        return object_store.presign_remote_cpu_put(**arguments)

    for refused in ({"grant": None}, {"grant": other}, {"binding_digest": "sha256:" + "0" * 64}):
        with pytest.raises(PaidResourceAdmissionBlocked):
            put(**refused)
    with pytest.raises(PaidResourceAdmissionBlocked):
        object_store.remote_cpu_object_store_sentinel(grant=None, binding_digest=digest, staging_prefix=staging,
                                                      attempt_id=binding["attempt_id"], client=store, bucket=B2_BUCKET)
    assert store.presigned == [] and store.operations == []
    with pytest.raises(ObjectStoreError) as too_long:
        put(expires_in_seconds=4 * 3600 + 1)  # a PUT never outlives the longest attempt the contract allows
    assert str(too_long.value) == "remote_cpu_presign_expiration_invalid"
    assert "X-Amz-Expires=2520" in put() and [method for method, _, _ in store.presigned] == ["put_object"]


def test_a_run_response_without_an_execution_name_is_reconciled_as_ambiguous(tmp_path: Path, monkeypatch) -> None:
    world = RemoteCpuWorld(tmp_path, monkeypatch)
    world.record_environment()
    rest = world.rest

    def nameless(method: str, url: str, *, body, headers):
        status, payload = rest(method, url, body=body, headers=headers)
        if url.endswith(":run") and status == 200:  # the run was accepted, but the operation names nothing
            return 200, json.dumps({"name": "operations/op-1", "done": False}).encode()
        return status, payload

    world.runtime.cloud_run._transport = nameless
    descriptor = world.descriptor()
    result = world.run("dispatch", descriptor=descriptor)
    [execution] = _attempt_executions(world, descriptor["attempt_id"])
    assert result["status"] == "dispatched" and result["reconciled"]["status"] == "found"
    assert world.lease(descriptor)["worker_identity"].endswith("/executions/" + execution["name"].rsplit("/", 1)[1])
    assert len(_runs(world, descriptor["attempt_id"])) == 1
