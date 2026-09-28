# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_job_allocator.py
#   src/blueprint_pipeline/cloud_run_jobs_client.py
#   src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py
#   src/blueprint_pipeline/remote_cpu_job_lease.py
#   tests/remote_cpu_allocator_fakes.py
"""ADP-009D/day-28, plan 14 PR 2: remote CPU jobs are admitted, consumed, leased and torn down by one seam."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from blueprint_pipeline import remote_cpu_job_allocator as allocator
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_job_lease as leases
from blueprint_pipeline import remote_cpu_job_records as records
from blueprint_pipeline.cloud_run_jobs_client import allocation_binding
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import PAID_LANE_ADMISSION_SCHEMA_VERSION
from tests.remote_cpu_allocator_fakes import (
    B2_BUCKET,
    JOB,
    T0,
    TRANSPORT_BUCKET,
    WORST_CASE_USD,
    RemoteCpuWorld,
    remote_cpu_config,
    standing_authority,
)


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
