"""Exclusive cross-process lease for paid provider lane mutations."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import blueprint_pipeline.isaac_g1_kitchen_parity_job as J
import blueprint_pipeline.paid_provider_lane_lease as lease_module
from blueprint_pipeline.paid_provider_lane_lease import (
    BLOCKER_ALREADY_OWNED,
    BLOCKER_STALE_REQUIRES_RECONCILIATION,
    PaidProviderLaneLeaseSet,
    STALE_RECLAIM_REQUIRED_EVIDENCE,
    accept_paid_provider_lane_lease_handoff,
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    lease_path,
    read_lease,
    release_paid_provider_lane_lease,
    transfer_paid_provider_lane_lease_to_watchdog,
)
from blueprint_pipeline.paid_lane_guard import (
    bind_pending_teardown_instance,
    open_pending_teardown,
)

_SCENARIOS = [
    {
        "scenario_id": "s1",
        "spawn_position_xyz": [0, 0, 0],
        "target_position_xyz": [1, 0, 0],
    }
]


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _reconciliation(
    provider: str = "runpod",
    lane: str = "lane",
    *,
    live_resource_count: int = 0,
    pending: list[dict] | None = None,
) -> dict:
    return build_paid_provider_lane_reconciliation(
        provider=provider,
        lane=lane,
        provider_inventory={
            "status": "observed",
            "api_confirmed": True,
            "live_resource_count": live_resource_count,
            "resources": [
                {"instance_id": "live-1", "name": "blueprint-isaac-g1-kitchen-parity"}
            ]
            if live_resource_count
            else [],
        },
        open_pending_teardowns=list(pending or []),
    )


def test_acquire_records_owner_identity_and_scope(tmp_path: Path) -> None:
    acquired = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="isaac_g1_kitchen_parity",
        job_dir=str(tmp_path / "job"),
        lease_dir=tmp_path,
        reconciliation=_reconciliation(lane="isaac_g1_kitchen_parity"),
    )

    assert acquired["status"] == "acquired"
    lease = acquired["lease"]
    assert lease["schema_version"] == "paid_provider_lane_lease.v1"
    assert lease["owner_pid"] == os.getpid()
    assert lease["hostname"]
    assert lease["lane"] == "isaac_g1_kitchen_parity"
    assert lease["provider"] == "runpod"
    assert lease["intended_provider"] == "runpod"
    assert lease["job_dir"] == str(tmp_path / "job")
    assert lease["started_at_epoch"] > 0
    assert lease["expires_at_epoch"] > lease["started_at_epoch"]
    assert "billing" in lease["claim_boundary"]
    on_disk = read_lease("runpod", "isaac_g1_kitchen_parity", tmp_path)
    assert on_disk == lease


def test_second_acquire_blocks_while_owner_is_alive(tmp_path: Path) -> None:
    first = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="a", lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )
    assert first["status"] == "acquired"

    second = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="b", lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )

    assert second["status"] == "blocked"
    assert second["blockers"] == [BLOCKER_ALREADY_OWNED]
    assert second["holder"]["job_dir"] == "a"
    # The live owner's lease is untouched.
    assert read_lease("runpod", "lane", tmp_path)["job_dir"] == "a"


def test_watchdog_handoff_is_one_time_bound_and_has_no_unowned_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_pids = {os.getpid(), 111, 222}
    monkeypatch.setattr(lease_module, "_pid_is_alive", lambda pid: pid in live_pids)
    acquired = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="groot_oscar_model_volume",
        job_dir=str(tmp_path),
        lease_dir=tmp_path,
        reconciliation=_reconciliation(lane="groot_oscar_model_volume"),
    )
    pending = open_pending_teardown(
        provider="runpod",
        lane="groot_oscar_model_volume",
        run_id="handoff-test",
        resource_kind="network_volume",
        resource_name="model-cache-volume",
        provider_location="US-WA-1",
        registry_dir=tmp_path / "pending",
    )
    bind_pending_teardown_instance(pending["path"], "volume-1")
    binding = {
        "provider": "runpod",
        "lane": "groot_oscar_model_volume",
        "volume_id": "volume-1",
        "pending_teardown_record": pending["path"],
        "watchdog_nonce": "nonce-1",
        "watchdog_deadline_epoch": time.time() + 3600,
    }
    capability = tmp_path / "provider_lane_handoff.capability"
    handoff = transfer_paid_provider_lane_lease_to_watchdog(
        acquired,
        watchdog_pid=111,
        capability_path=capability,
        binding=binding,
    )
    assert handoff["status"] == "pending_canary_acceptance"
    assert capability.stat().st_mode & 0o077 == 0
    retained = read_lease("runpod", "groot_oscar_model_volume", tmp_path)
    assert retained["owner_pid"] == 111
    assert retained["retained_teardown_owner_pid"] == 111

    unrelated = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="groot_oscar_model_volume",
        job_dir="unrelated",
        lease_dir=tmp_path,
        reconciliation=_reconciliation(lane="groot_oscar_model_volume"),
    )
    assert unrelated["status"] == "blocked"
    assert unrelated["blockers"] == [BLOCKER_ALREADY_OWNED]

    forged = accept_paid_provider_lane_lease_handoff(
        handoff,
        canary_watchdog={
            "watchdog_pid": 222,
            "watchdog_pod_name_prefix": "blueprint-groot-oscar-canary-test-",
            "watchdog_deadline_epoch": time.time() + 1800,
            "watchdog_process_identity_verified": True,
            "independent_teardown_watchdog": True,
        },
        expected_binding={**binding, "volume_id": "volume-forged"},
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--pod-name-prefix",
            "blueprint-groot-oscar-canary-test-",
            "--deadline-epoch",
            str(time.time() + 1800),
        ),
    )
    assert forged["status"] == "blocked"
    assert capability.exists()

    canary_deadline = time.time() + 1800
    canary_watchdog = {
        "watchdog_pid": 222,
        "watchdog_pod_name_prefix": "blueprint-groot-oscar-canary-test-",
        "watchdog_deadline_epoch": canary_deadline,
        "watchdog_process_identity_verified": True,
        "independent_teardown_watchdog": True,
    }
    accepted = accept_paid_provider_lane_lease_handoff(
        handoff,
        canary_watchdog=canary_watchdog,
        expected_binding=binding,
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--pod-name-prefix",
            "blueprint-groot-oscar-canary-test-",
            "--deadline-epoch",
            str(canary_deadline),
        ),
    )
    assert accepted["status"] == "accepted"
    assert accepted["capability_consumed"] is True
    assert not capability.exists()
    current = read_lease("runpod", "groot_oscar_model_volume", tmp_path)
    assert current["owner_pid"] == 222
    assert current["retained_teardown_owner_pid"] == 111

    reused = accept_paid_provider_lane_lease_handoff(
        handoff,
        canary_watchdog=canary_watchdog,
        expected_binding=binding,
        process_argv_probe=lambda _pid: (
            "python",
            "-m",
            "blueprint_pipeline.groot_oscar_runpod_watchdog",
            "--pod-name-prefix",
            "blueprint-groot-oscar-canary-test-",
            "--deadline-epoch",
            str(canary_deadline),
        ),
    )
    assert reused["status"] == "blocked"


def test_live_retained_watchdog_prevents_parent_exit_stale_reclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dead_parent = _dead_pid()
    monkeypatch.setattr(
        lease_module,
        "_pid_is_alive",
        lambda pid: pid == 111,
    )
    path = lease_path("runpod", "groot_oscar_model_volume", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "paid_provider_lane_lease.v1",
                "provider": "runpod",
                "lane": "groot_oscar_model_volume",
                "owner_pid": dead_parent,
                "retained_teardown_owner_pid": 111,
                "hostname": "",
                "job_dir": "exited-storage-supervisor",
                "started_at_epoch": time.time() - 60,
                "expires_at_epoch": time.time() + 3600,
            }
        ),
        encoding="utf-8",
    )
    blocked = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="groot_oscar_model_volume",
        job_dir="unrelated",
        lease_dir=tmp_path,
        reconciliation=_reconciliation(lane="groot_oscar_model_volume"),
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [BLOCKER_ALREADY_OWNED]


def test_concurrent_process_blocks_before_paid_lane_mutation(tmp_path: Path) -> None:
    script = r'''
import json, sys
from blueprint_pipeline.paid_provider_lane_lease import (
    acquire_paid_provider_lane_lease,
    build_paid_provider_lane_reconciliation,
    release_paid_provider_lane_lease,
)
lease_dir = sys.argv[1]
reconciliation = build_paid_provider_lane_reconciliation(
    provider="runpod",
    lane="lane",
    provider_inventory={"api_confirmed": True, "live_resource_count": 0, "resources": []},
    open_pending_teardowns=[],
)
acquired = acquire_paid_provider_lane_lease(
    provider="runpod", lane="lane", job_dir="child", lease_dir=lease_dir,
    reconciliation=reconciliation,
)
print(json.dumps({"status": acquired["status"], "owner_pid": acquired.get("lease", {}).get("owner_pid")}), flush=True)
sys.stdin.readline()
release_paid_provider_lane_lease(
    acquired, reason="test_child_done", provider_mutation_started=False,
    lease_dir=lease_dir,
)
'''
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    [
                        str(Path(__file__).resolve().parents[1] / "src"),
                        os.environ.get("PYTHONPATH", ""),
                    ],
                )
            ),
        },
    )
    try:
        assert child.stdout is not None
        output_line = child.stdout.readline()
        if not output_line:
            assert child.stderr is not None
            pytest.fail(child.stderr.read() or "child lease process exited silently")
        child_result = json.loads(output_line)
        assert child_result["status"] == "acquired"
        assert child_result["owner_pid"] == child.pid

        blocked = acquire_paid_provider_lane_lease(
            provider="runpod",
            lane="lane",
            job_dir="parent",
            lease_dir=tmp_path,
            reconciliation=_reconciliation(),
        )
        assert blocked["status"] == "blocked"
        assert blocked["blockers"] == [BLOCKER_ALREADY_OWNED]
        assert blocked["holder"]["owner_pid"] == child.pid
    finally:
        if child.poll() is None and child.stdin is not None:
            try:
                child.stdin.write("\n")
                child.stdin.flush()
            except BrokenPipeError:
                pass
        child.wait(timeout=10)


def test_distinct_lanes_and_providers_do_not_contend(tmp_path: Path) -> None:
    a = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="a", lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )
    b = acquire_paid_provider_lane_lease(
        provider="digitalocean", lane="lane", job_dir="b", lease_dir=tmp_path,
        reconciliation=_reconciliation(provider="digitalocean"),
    )
    c = acquire_paid_provider_lane_lease(
        provider="runpod", lane="other-lane", job_dir="c", lease_dir=tmp_path,
        reconciliation=_reconciliation(lane="other-lane"),
    )
    assert {r["status"] for r in (a, b, c)} == {"acquired"}


def test_stale_dead_owner_requires_reconciliation_evidence(tmp_path: Path) -> None:
    path = lease_path("runpod", "lane", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    path.write_text(
        json.dumps(
            {
                "schema_version": "paid_provider_lane_lease.v1",
                "provider": "runpod",
                "lane": "lane",
                "owner_pid": _dead_pid(),
                "hostname": "",
                "job_dir": "crashed",
                "started_at_epoch": now - 60,
                "expires_at_epoch": now + 3600,
            }
        ),
        encoding="utf-8",
    )

    blocked = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="new", lease_dir=tmp_path,
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [BLOCKER_STALE_REQUIRES_RECONCILIATION]
    assert blocked["stale_reason"] == "owner_pid_not_alive"
    assert list(STALE_RECLAIM_REQUIRED_EVIDENCE) == blocked[
        "required_reconciliation_evidence"
    ]
    # The stale lease is preserved: a crashed owner may have left an
    # allocation behind, so the lane stays closed.
    assert read_lease("runpod", "lane", tmp_path)["job_dir"] == "crashed"

    partial = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="lane",
        job_dir="new",
        lease_dir=tmp_path,
        reconciliation={"provider_inventory_checked": True},
    )
    assert partial["status"] == "blocked"

    reclaimed = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="lane",
        job_dir="new",
        lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )
    assert reclaimed["status"] == "acquired"
    assert read_lease("runpod", "lane", tmp_path)["job_dir"] == "new"


def test_live_same_host_owner_is_never_stale_even_past_expiry(tmp_path: Path) -> None:
    path = lease_path("runpod", "lane", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    path.write_text(
        json.dumps(
            {
                "schema_version": "paid_provider_lane_lease.v1",
                "provider": "runpod",
                "lane": "lane",
                "owner_pid": os.getpid(),
                "hostname": "",
                "job_dir": "long-running-owner",
                "started_at_epoch": now - 7200,
                "expires_at_epoch": now - 1,
            }
        ),
        encoding="utf-8",
    )

    blocked = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="lane",
        job_dir="new",
        lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )
    # Even complete reconciliation evidence never deletes a LIVE owner's lease.
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [BLOCKER_ALREADY_OWNED]
    assert read_lease("runpod", "lane", tmp_path)["job_dir"] == "long-running-owner"


def test_cross_host_lease_is_stale_only_after_expiry(tmp_path: Path) -> None:
    path = lease_path("runpod", "lane", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    record = {
        "schema_version": "paid_provider_lane_lease.v1",
        "provider": "runpod",
        "lane": "lane",
        "owner_pid": 1,
        "hostname": "some-other-host",
        "job_dir": "remote",
        "started_at_epoch": now - 7200,
        "expires_at_epoch": now + 3600,
    }
    path.write_text(json.dumps(record), encoding="utf-8")

    unexpired = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="new", lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )
    assert unexpired["status"] == "blocked"
    assert unexpired["blockers"] == [BLOCKER_ALREADY_OWNED]

    record["expires_at_epoch"] = now - 1
    path.write_text(json.dumps(record), encoding="utf-8")
    expired = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="new", lease_dir=tmp_path,
    )
    assert expired["status"] == "blocked"
    assert expired["stale_reason"] == "lease_expired"
    assert expired["blockers"] == [BLOCKER_STALE_REQUIRES_RECONCILIATION]


def test_release_after_verified_teardown_and_owner_check(tmp_path: Path) -> None:
    acquired = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="a", lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )
    released = release_paid_provider_lane_lease(
        acquired,
        reason="watch_and_collect_finished",
        provider_mutation_started=True,
        terminal_reconciliation=_reconciliation(),
        lease_dir=tmp_path,
    )
    assert released["status"] == "released"
    assert released["teardown_verified"] is True
    assert read_lease("runpod", "lane", tmp_path) is None

    # A second release is a no-op, and a foreign holder is never deleted.
    again = release_paid_provider_lane_lease(
        acquired, reason="again", provider_mutation_started=False, lease_dir=tmp_path
    )
    assert again["status"] == "already_released"

    other = acquire_paid_provider_lane_lease(
        provider="runpod", lane="lane", job_dir="other", lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )
    assert other["status"] == "acquired"
    refused = release_paid_provider_lane_lease(
        acquired,
        reason="stale-first-owner",
        provider_mutation_started=False,
        lease_dir=tmp_path,
    )
    assert refused["status"] == "refused_not_owner"
    assert read_lease("runpod", "lane", tmp_path)["job_dir"] == "other"


def test_initial_acquire_blocks_for_active_allocation_without_legacy_lease(
    tmp_path: Path,
) -> None:
    reconciliation = _reconciliation(live_resource_count=1)

    blocked = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="lane",
        job_dir="new",
        lease_dir=tmp_path,
        reconciliation=reconciliation,
    )

    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [BLOCKER_ALREADY_OWNED]
    assert blocked["reconciliation"]["provider_live_resource_count"] == 1
    assert read_lease("runpod", "lane", tmp_path) is None


def test_initial_acquire_blocks_for_open_pending_teardown_without_legacy_lease(
    tmp_path: Path,
) -> None:
    reconciliation = _reconciliation(
        pending=[
            {
                "schema_version": "pending_teardown.v1",
                "provider": "runpod",
                "lane": "lane",
                "status": "open",
                "run_id": "legacy-run",
                "instance_id": "legacy-pod",
                "job_dir": "legacy-job",
                "path": str(tmp_path / "pending.json"),
            }
        ]
    )

    blocked = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="lane",
        job_dir="new",
        lease_dir=tmp_path,
        reconciliation=reconciliation,
    )

    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == [BLOCKER_ALREADY_OWNED]
    assert blocked["reconciliation"]["open_pending_teardown_count"] == 1


def test_reconciliation_blocks_open_supervised_descendant_record() -> None:
    reconciliation = _reconciliation(
        lane=J.ISAAC_G1_KITCHEN_PARITY_LANE,
        pending=[
            {
                "schema_version": "pending_teardown.v1",
                "provider": "runpod",
                "lane": "isaac_startup_supervisor",
                "status": "open",
                "run_id": "supervised-attempt",
                "instance_id": "child-pod",
                "job_dir": "child-job",
                "path": "/tmp/child-pending.json",
            }
        ],
    )

    assert reconciliation["status"] == "blocked"
    assert reconciliation["open_pending_teardown_count"] == 1
    assert reconciliation["blockers"] == [BLOCKER_ALREADY_OWNED]


def test_supervised_descendant_allocation_is_inside_parity_lease_scope(
    tmp_path: Path,
) -> None:
    class _Provider:
        name = "runpod"

        def billable_inventory(self, *, name_prefix: str) -> dict:
            resource_name = "blueprint-isaac-g1-supervised-attempt-001"
            resources = (
                [{"instance_id": "supervised-pod", "name": resource_name}]
                if resource_name.startswith(name_prefix)
                else []
            )
            return {
                "api_confirmed": True,
                "live_resource_count": len(resources),
                "resources": resources,
            }

    lease_set = PaidProviderLaneLeaseSet(
        providers={"runpod": _Provider()},
        lane=J.ISAAC_G1_KITCHEN_PARITY_LANE,
        job_dir=str(tmp_path / "job"),
        resource_name_prefix=J.ISAAC_G1_KITCHEN_PARITY_RESOURCE_PREFIX,
    )

    summary = lease_set.acquire()

    assert summary["status"] == "blocked"
    assert summary["blockers"] == [BLOCKER_ALREADY_OWNED]
    assert summary["leases"][0]["reconciliation"][
        "provider_live_resource_count"
    ] == 1


def test_release_retains_lease_until_terminal_reconciliation_passes(
    tmp_path: Path,
) -> None:
    acquired = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane="lane",
        job_dir="job",
        lease_dir=tmp_path,
        reconciliation=_reconciliation(),
    )

    retained = release_paid_provider_lane_lease(
        acquired,
        reason="watch_finished_but_pod_still_billable",
        provider_mutation_started=True,
        terminal_reconciliation=_reconciliation(live_resource_count=1),
        lease_dir=tmp_path,
    )

    assert retained["status"] == "retained_unverified_teardown"
    assert retained["released"] is False
    assert read_lease("runpod", "lane", tmp_path) is not None


def test_paid_job_blocks_before_provider_mutation_when_lane_owned(
    tmp_path: Path, monkeypatch
) -> None:
    """A second local agent must stop with a stable blocker before create."""
    lease_dir = Path(os.environ["BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR"])
    other_owner = acquire_paid_provider_lane_lease(
        provider="runpod",
        lane=J.ISAAC_G1_KITCHEN_PARITY_LANE,
        job_dir="other-agent-job",
        lease_dir=lease_dir,
        reconciliation=_reconciliation(lane=J.ISAAC_G1_KITCHEN_PARITY_LANE),
    )
    assert other_owner["status"] == "acquired"

    monkeypatch.setenv(J.ISAAC_G1_MAX_SPEND_USD_ENV, "50.0")
    monkeypatch.setenv(
        J.ISAAC_WORKER_IMAGE_REF_ENV, "registry.example/worker:20260711"
    )
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/b?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/o?sig=B")
        (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/o?sig=C")
        return {"status": "completed", "manifest": {}}

    class _FakeProvider:
        name = "runpod"

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"env": dict(spec.env)}

    def _launch_must_not_run(*_args, **_kwargs):
        raise AssertionError("owned lane must block before any provider mutation")

    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): _FakeProvider())
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "launch_with_marker_retry", _launch_must_not_run)
    monkeypatch.setattr(J, "race_launch", _launch_must_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        cold_race_contenders=1,
    )

    assert m["status"] == "blocked"
    assert BLOCKER_ALREADY_OWNED in m["blockers"]
    assert m["paid_provider_lane_lease"]["status"] == "blocked"
    assert m["paid_provider_lane_lease"]["leases"][0]["holder"]["job_dir"] == (
        "other-agent-job"
    )
    # The prior owner's lease is untouched.
    assert read_lease(
        "runpod", J.ISAAC_G1_KITCHEN_PARITY_LANE, lease_dir
    )["job_dir"] == "other-agent-job"


def _stage_for_paid_job(_bundle_zip, job_dir, *, key_prefix):
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "provider_bundle_url.txt").write_text(
        "https://spaces.example/b?sig=A", encoding="utf-8"
    )
    (job_dir / "provider_output_put_url.txt").write_text(
        "https://spaces.example/o?sig=B", encoding="utf-8"
    )
    (job_dir / "provider_output_get_url.txt").write_text(
        "https://spaces.example/o?sig=C", encoding="utf-8"
    )
    return {"status": "completed", "manifest": {"key_prefix": key_prefix}}


class _InventoryProvider:
    name = "runpod"

    def __init__(self) -> None:
        self.live = False

    def available(self) -> dict:
        return {"provider": self.name, "available": True}

    def build_request(self, spec, job_dir) -> dict:
        return {"env": dict(spec.env), "image": spec.image}

    def billable_inventory(self, *, name_prefix: str) -> dict:
        return {
            "status": "observed",
            "api_confirmed": True,
            "live_resource_count": 1 if self.live else 0,
            "resources": (
                [{"instance_id": "pod-live", "name": name_prefix}]
                if self.live
                else []
            ),
        }


def _configure_paid_job(monkeypatch, provider: _InventoryProvider) -> None:
    monkeypatch.setenv(J.ISAAC_G1_MAX_SPEND_USD_ENV, "50.0")
    monkeypatch.setenv(
        J.ISAAC_WORKER_IMAGE_REF_ENV, "registry.example/worker:20260711"
    )
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )
    monkeypatch.setattr(
        J, "get_render_provider", lambda _name, warm_candidates=(): provider
    )
    monkeypatch.setattr(J, "stage_bundle", _stage_for_paid_job)


def test_paid_job_blocks_on_legacy_open_teardown_before_provider_create(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _InventoryProvider()
    _configure_paid_job(monkeypatch, provider)
    open_pending_teardown(
        provider="runpod",
        lane=J.ISAAC_G1_KITCHEN_PARITY_LANE,
        run_id="legacy-active-run",
        job_dir=tmp_path / "legacy-job",
        max_age_seconds=3600,
    )
    monkeypatch.setattr(
        J,
        "launch_with_marker_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy active lane must block before create")
        ),
    )

    manifest = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        cold_race_contenders=1,
    )

    assert manifest["status"] == "blocked"
    assert BLOCKER_ALREADY_OWNED in manifest["blockers"]
    reconciliation = manifest["paid_provider_lane_lease"]["leases"][0][
        "reconciliation"
    ]
    assert reconciliation["open_pending_teardown_count"] == 1


def test_launch_exception_releases_lease_when_inventory_proves_no_allocation(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _InventoryProvider()
    _configure_paid_job(monkeypatch, provider)
    monkeypatch.setattr(
        J,
        "launch_with_marker_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("provider call failed before allocation")
        ),
    )

    with pytest.raises(RuntimeError, match="before allocation"):
        J.run_isaac_g1_kitchen_parity_job(
            scenarios=_SCENARIOS,
            out_dir=tmp_path / "job",
            provider="runpod",
            allow_paid=True,
            allow_dirty_paid_launch=True,
            cold_race_contenders=1,
        )

    lease_dir = Path(os.environ["BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR"])
    assert read_lease(
        "runpod", J.ISAAC_G1_KITCHEN_PARITY_LANE, lease_dir
    ) is None


def test_stopped_billable_allocation_retains_lane_lease(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _InventoryProvider()
    _configure_paid_job(monkeypatch, provider)

    def _launch(*_args, **_kwargs):
        provider.live = True
        return {
            "status": "launched",
            "instance_id": "pod-live",
            "mode": "cold_create_marker_verified",
        }

    monkeypatch.setattr(J, "launch_with_marker_retry", _launch)
    monkeypatch.setattr(
        J,
        "watch_and_collect",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "elapsed_seconds": 1,
            "teardown": {"status": "stopped"},
            "teardown_reason": "runner_done_preserved_for_warm_reuse",
            "runner_result": {
                "status": "completed",
                "policy_id": "blueprint_default_walk_to_target_smoke_policy",
                "scenarios": [],
            },
        },
    )

    manifest = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        cold_race_contenders=1,
    )

    release = manifest["paid_provider_lane_lease"]["release"]
    assert release["all_providers_terminal"] is False
    assert release["results"][0]["status"] == "retained_unverified_teardown"
    lease_dir = Path(os.environ["BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR"])
    assert read_lease(
        "runpod", J.ISAAC_G1_KITCHEN_PARITY_LANE, lease_dir
    ) is not None
