"""Hermetic tests for the cold-start reliability wiring.

Covers the three levers wired on 2026-07-02 (no network, no GPU, no secrets):

1. same-provider cold-create racing (``resolve_cold_race_contenders`` +
   ``_ColdCreateContender`` + ``race_launch`` with same-name contenders),
2. the image-aware no-runtime guard defaulting on (900s before a manifest floor),
3. the warm serve worker control plane (``scripts/run_warm_render_worker.py``)
   and gpu_spend_guard's expected-serve-pod tagging.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import isaac_g1_kitchen_parity_job as J
from blueprint_pipeline.paid_lane_guard import (
    bind_pending_teardown_instance,
    load_pending_teardowns,
    open_pending_teardown,
)
from blueprint_pipeline.paid_provider_lane_lease import lease_path, read_lease
from blueprint_pipeline.production_gpu_campaign_budget import (
    AUTHORIZED_GPU_WALL_CAP_SECONDS,
)
from blueprint_pipeline.provider_race import race_launch
from scripts import gpu_spend_guard as guard
from scripts import run_warm_render_worker as worker


# --------------------------- cold race contender resolution ---------------------------


def test_resolve_cold_race_contenders_default_env_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(J.COLD_RACE_CONTENDERS_ENV, raising=False)
    assert J.resolve_cold_race_contenders() == J.DEFAULT_COLD_RACE_CONTENDERS
    assert J.DEFAULT_COLD_RACE_CONTENDERS == 1

    assert J.resolve_cold_race_contenders(1) == 1        # explicit disable
    assert J.resolve_cold_race_contenders(0) == 1        # floor
    assert J.resolve_cold_race_contenders(99) == 4       # ceiling

    monkeypatch.setenv(J.COLD_RACE_CONTENDERS_ENV, "3")
    assert J.resolve_cold_race_contenders() == 3
    assert J.resolve_cold_race_contenders(2) == 2        # explicit beats env

    monkeypatch.setenv(J.COLD_RACE_CONTENDERS_ENV, "not-a-number")
    assert J.resolve_cold_race_contenders() == J.DEFAULT_COLD_RACE_CONTENDERS


def test_cold_create_contender_forces_cold_and_delegates() -> None:
    calls: list[dict] = []

    class FakeProvider:
        name = "runpod"

        def launch(self, job_dir, request, *, cold=False, **kwargs):
            calls.append({"cold": cold, "kwargs": kwargs})
            return {"status": "launched", "instance_id": "pod-x", "mode": "cold_create"}

        def terminate(self, iid):
            return {"status": "terminated", "id": iid}

    proxy = J._ColdCreateContender(FakeProvider())
    assert proxy.name == "runpod"
    result = proxy.launch(Path("."), {}, cold=False, allow_cold_fallback=True)
    assert result["status"] == "launched"
    assert calls == [{"cold": True, "kwargs": {"allow_cold_fallback": True}}]
    # everything else delegates to the wrapped provider
    assert proxy.terminate("pod-x") == {"status": "terminated", "id": "pod-x"}


def test_same_provider_race_keeps_first_boot_and_terminates_loser(tmp_path: Path) -> None:
    """Two contenders on ONE provider: the dud never boots, the cold proxy wins."""
    terminated: list[str] = []

    class FakeProvider:
        name = "runpod"

        def launch(self, job_dir, request, *, cold=False, **kwargs):
            # contender 0 launches the dud (warm-then-cold lane); the proxy's forced
            # cold launch gets the healthy pod.
            iid = "pod-healthy" if cold else "pod-dud"
            return {"status": "launched", "instance_id": iid, "mode": "cold_create"}

        def terminate(self, iid):
            terminated.append(iid)
            return {"status": "terminated", "id": iid}

        def inspect(self, iid):
            return {"status": "unavailable", "http": 404, "instance_id": iid}

    prov = FakeProvider()
    contenders = [prov, J._ColdCreateContender(prov)]

    def marker_check(_provider, launch_result):
        return launch_result.get("instance_id") == "pod-healthy"

    race = race_launch(
        contenders,
        {"body": True},
        marker_check,
        marker_timeout=10,
        job_dir=tmp_path,
        cold=False,
        poll_interval=0.01,
        sleep=lambda _s: None,
    )
    assert race["status"] == "launched"
    assert race["instance_id"] == "pod-healthy"
    assert terminated == ["pod-dud"]
    outcomes = {r["instance_id"]: r["outcome"] for r in race["contenders"]}
    assert outcomes["pod-healthy"] == "won"
    assert outcomes["pod-dud"] in ("no_boot", "aborted")


def test_paid_single_provider_job_can_opt_in_to_two_cold_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through run_isaac_g1_kitchen_parity_job: explicit cold_race_contenders=2
    launches two contenders on the one provider, keeps the marker winner, terminates the
    dud, and collects from the winner's contender dir."""
    monkeypatch.delenv(J.COLD_RACE_CONTENDERS_ENV, raising=False)
    monkeypatch.setenv(
        J.ISAAC_WORKER_IMAGE_REF_ENV,
        "registry.example/blueprint/isaac-eval-worker:test",
    )
    monkeypatch.setenv(J.ISAAC_G1_MAX_SPEND_USD_ENV, "10.0")
    launched: list[dict] = []
    terminated: list[str] = []

    def _fake_stage(bundle_zip, job_dir, key_prefix="blueprint/isaac-g1-parity"):
        job_dir = Path(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        for name in ("provider_bundle_url.txt", "provider_output_put_url.txt",
                     "provider_output_get_url.txt"):
            (job_dir / name).write_text(f"https://spaces.example/{name}?sig=x")
        return {"status": "completed", "manifest": {}}

    class _FakeProvider:
        name = "runpod"

        def available(self):
            return {"provider": "runpod", "available": True}

        def build_request(self, spec, job_dir):
            return {"provider": "runpod", "env": dict(spec.env)}

        def billable_inventory(self, *, name_prefix: str):
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
                "name_prefix": name_prefix,
            }

        def launch(self, job_dir, request, *, cold=False, **kwargs):
            # contender threads launch concurrently: key the fake pod id off the
            # contender dir, not a shared counter
            contender = "1" if "contender-1-" in str(job_dir) else "0"
            iid = f"pod-{contender}"
            launched.append({"iid": iid, "cold": cold, "job_dir": str(job_dir)})
            return {"status": "launched", "instance_id": iid, "mode": "cold_create"}

        def terminate(self, iid):
            terminated.append(iid)
            return {"status": "terminated", "id": iid}

        def inspect(self, iid):
            return {"status": "unavailable", "http": 404, "instance_id": iid}

    # the second contender (index 1) is the one that "boots"
    def _fake_marker(job_dir, *, expected_launch_session_id=None, urlopen=None, **_k):
        return "contender-1-" in str(job_dir)

    captured: dict = {}

    def _fake_watch(job_dir, render_out, instance_id, *, provider=None, **_kwargs):
        captured["collect_job_dir"] = str(job_dir)
        captured["collect_instance_id"] = instance_id
        return {
            "status": "completed",
            "elapsed_seconds": 1,
            "teardown": {
                "status": "terminated",
                "verification": {
                    "api_confirmed": True,
                    "provider_status": "not_found",
                },
            },
            "runner_result": {
                "status": "completed",
                "policy_id": "blueprint_default_walk_to_target_smoke_policy",
                "scenarios": [],
                "scenarios_executed": 0,
                "scenarios_passed": 0,
            },
        }

    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): _FakeProvider())
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "boot_marker_present", _fake_marker)
    monkeypatch.setattr(J, "watch_and_collect", _fake_watch)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=[{"scenario_id": "s1", "spawn_position_xyz": [0, 0, 0],
                    "target_position_xyz": [1, 0, 0]}],
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        max_spend_usd=40.0,
        cold_race_contenders=2,
        # keeps race_launch's poll_interval (min(15, marker_timeout)) at 1s so the losing
        # contender's abort check fires immediately — hermetic-fast, same code path
        marker_timeout=1,
    )

    assert m["cold_race_contenders"] == 2
    assert m["race_rounds"] == 1
    assert len(launched) == 2
    # one contender keeps the caller's warm-then-cold lane; the other is forced cold
    # (contender threads run concurrently, so launch ORDER is nondeterministic)
    assert sorted(rec["cold"] for rec in launched) == [False, True]
    assert m["launch"]["status"] == "launched"
    winner_iid = m["launch"]["instance_id"]
    loser_iid = next(rec["iid"] for rec in launched if rec["iid"] != winner_iid)
    assert terminated == [loser_iid]
    assert captured["collect_instance_id"] == winner_iid
    assert "contender-1-runpod" in captured["collect_job_dir"]
    assert m["status"] == "evidence_collected_closure_required"
    assert "g1_kitchen_attempt_closure_missing" in m["blockers"]


def test_job_signature_defaults_enable_race_and_no_runtime_guard() -> None:
    sig = inspect.signature(J.run_isaac_g1_kitchen_parity_job)
    assert (
        sig.parameters["startup_no_runtime_timeout"].default
        == J.DEFAULT_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS
        == 900
    )
    # None -> resolve_cold_race_contenders() -> one-resource default (env-overridable).
    assert sig.parameters["cold_race_contenders"].default is None


def test_audit_launcher_exposes_reliability_flags() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_g1_render_noise_audit.py"
    ).read_text()
    assert "--startup-no-runtime-timeout" in source
    assert "--cold-race-contenders" in source
    assert "startup_no_runtime_timeout=args.startup_no_runtime_timeout" in source
    assert "cold_race_contenders=args.cold_race_contenders" in source


# --------------------------- warm serve worker control plane ---------------------------


def _serving_manifest(pod_id: str = "pod-serve") -> dict:
    return {
        "status": "serving",
        "blockers": [],
        "warm_serve": {
            "instance_id": pod_id,
            "ready": True,
            "inbox_put_url_file": "/tmp/warm_inbox_put_url.txt",
            "output_get_url_file": "/tmp/provider_output_get_url.txt",
            "pending_teardown_record": "/tmp/pending-warm-teardown.json",
        },
    }


def test_start_warm_worker_writes_expected_serve_marker(tmp_path: Path) -> None:
    captured: dict = {}

    def fake_job(**kwargs):
        captured.update(kwargs)
        return _serving_manifest()

    result = worker.start_warm_worker(
        out_dir=tmp_path,
        kitchen_asset_dir=None,
        kitchen_url="https://signed/kitchen.zip",
        provider="runpod",
        allow_paid=True,
        warm_candidates=(),
        marker_timeout=900,
        serve_idle_timeout_s=1800.0,
        serve_max_jobs=None,
        serve_ready_timeout=1800,
        worker_image_manifest_diagnostic=tmp_path / "worker-image-diagnostic.json",
        job_fn=fake_job,
    )
    assert captured["serve"] is True
    assert captured["scenarios"] == []
    assert captured["worker_image_manifest_diagnostic"] == (
        tmp_path / "worker-image-diagnostic.json"
    )
    assert result["status"] == "serving"
    assert result["pod_id"] == "pod-serve"

    marker = json.loads((tmp_path / worker.WARM_SERVE_MARKER_FILENAME).read_text())
    assert marker["status"] == "serving"
    assert marker["pod_id"] == "pod-serve"
    assert marker["manifest_path"].endswith(J.JOB_MANIFEST_FILENAME)
    assert marker["heartbeat_at"]
    assert marker["lease_expires_at"] > marker["heartbeat_at"]
    assert marker["pending_teardown_record"] == "/tmp/pending-warm-teardown.json"
    # the manifest itself is persisted for submit's URL-file lookup
    assert (tmp_path / J.JOB_MANIFEST_FILENAME).is_file()


def test_production_warm_worker_forwards_probe_scenario_gpu_and_supervisor(
    tmp_path: Path,
) -> None:
    captured: dict = {}
    image = "docker.io/blueprint/worker@sha256:" + "a" * 64
    evidence_dir = tmp_path / "production_registration_evidence"
    evidence_dir.mkdir()
    records = {
        "host": {
            "schema_version": "production_gpu_host_boot_evidence.v1",
            "host_image_id": "runpod-secure-l40s-active-worker-v1",
            "actual_gpu_model": "NVIDIA L40S",
            "checks": {
                "host_image_booted": True,
                "nvidia_driver_ready": True,
                "container_runtime_ready": True,
            },
        },
        "cache": {
            "schema_version": "production_gpu_cache_evidence.v1",
            "worker_image_ref": image,
            "model_manifest_digest": "sha256:" + "b" * 64,
            "checks": {"worker_image_cached": True, "models_cached_offline": True},
        },
        "warm": {
            "schema_version": "production_gpu_warm_serve_ready.v2",
            "status": "serving",
            "launch_session_id": "session-1",
            "worker_image_ref": image,
            "checks": {
                "isaac_renderer_warm": True,
                "kitchen_scene_loaded": True,
                "policy_endpoint_ready": True,
                "worker_healthcheck_passed": True,
            },
        },
    }
    filenames = {
        "host": "production_host_boot_evidence.json",
        "cache": "production_cache_evidence.json",
        "warm": "warm_serve_ready.json",
    }
    paths: dict[str, str] = {}
    for label, filename in filenames.items():
        path = evidence_dir / filename
        path.write_text(json.dumps(records[label]), encoding="utf-8")
        paths[label] = str(path)
    manifest = _serving_manifest()
    manifest["warm_serve"]["ready_detail"] = {
        "registration_evidence_paths": paths,
    }
    token = tmp_path / "pool-token"
    token.write_text("x" * 32, encoding="utf-8")
    token.chmod(0o600)
    supervisor = {
        "schema_version": "production_gpu_warm_watchdog.v1",
        "status": "armed",
        "independent_process": True,
        "pid": 123,
        "deadline_epoch": 9_999_999_999,
        "evidence_path": str(tmp_path / "watchdog.json"),
    }

    result = worker.start_warm_worker(
        out_dir=tmp_path,
        kitchen_asset_dir=None,
        kitchen_url="https://signed/kitchen.zip",
        provider="runpod",
        allow_paid=True,
        warm_candidates=(),
        marker_timeout=900,
        serve_idle_timeout_s=1800,
        serve_max_jobs=3,
        serve_ready_timeout=1800,
        scenarios=[{"scenario_id": "warmup", "task": "open drawer"}],
        production_warmup_before_ready=True,
        teardown_supervisor=supervisor,
        pool_base_url="https://pool.example.internal",
        pool_token_file=token,
        worker_endpoint_ref="https://broker.example.internal/workers/{worker_id}",
        registration_sender=lambda *_args: {"ready_for_customer_binding": True},
        heartbeat_launcher=lambda **_kwargs: {"status": "monitoring", "pid": 456},
        job_fn=lambda **kwargs: captured.update(kwargs) or manifest,
    )

    assert result["status"] == "serving"
    assert captured["scenarios"][0]["scenario_id"] == "warmup"
    assert captured["serve_production_warmup_before_ready"] is True
    assert captured["runpod_gpu_types"] == ("NVIDIA L40S",)
    assert captured["serve_teardown_supervisor"] == supervisor
    assert result["production_registration"]["status"] == "registered"
    assert result["production_registration"]["registration_payload"]["endpoint_ref"] == (
        "https://broker.example.internal/workers/pod-serve"
    )
    assert result["heartbeat_agent"] == {"status": "monitoring", "pid": 456}


def test_campaign_budget_is_reserved_before_paid_warm_launch(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign-budget.json"
    reservation = worker.reserve_campaign_budget(
        ledger_path=ledger_path,
        hard_ttl_seconds=2_000,
        max_hourly_rate_usd=1.0,
        initial_spent_usd=3.0,
        initial_used_gpu_seconds=8_815,
        reservation_id="qualification-reservation-one",
    )

    assert reservation["status"] == "open"
    assert reservation["ledger_snapshot"]["committed_gpu_seconds"] == 10_815
    assert reservation["ledger_snapshot"]["remaining_gpu_seconds"] == (
        AUTHORIZED_GPU_WALL_CAP_SECONDS - 10_815
    )


def test_new_campaign_budget_requires_reconciled_baseline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires_reconciled_baseline"):
        worker.reserve_campaign_budget(
            ledger_path=tmp_path / "campaign-budget.json",
            hard_ttl_seconds=300,
            max_hourly_rate_usd=1.0,
            initial_spent_usd=None,
            initial_used_gpu_seconds=None,
        )


def test_start_warm_worker_blocked_job_writes_no_marker(tmp_path: Path) -> None:
    result = worker.start_warm_worker(
        out_dir=tmp_path,
        kitchen_asset_dir=None,
        kitchen_url=None,
        provider="runpod",
        allow_paid=False,
        warm_candidates=(),
        marker_timeout=900,
        serve_idle_timeout_s=1800.0,
        serve_max_jobs=None,
        serve_ready_timeout=1800,
        job_fn=lambda **_k: {"status": "blocked", "blockers": ["kitchen_url_missing"]},
    )
    assert result["status"] == "blocked"
    assert not (tmp_path / worker.WARM_SERVE_MARKER_FILENAME).exists()


def test_submit_tasks_builds_scenarios_and_uses_marker_manifest(tmp_path: Path) -> None:
    worker.start_warm_worker(
        out_dir=tmp_path,
        kitchen_asset_dir=None,
        kitchen_url="https://signed/kitchen.zip",
        provider="runpod",
        allow_paid=True,
        warm_candidates=(),
        marker_timeout=900,
        serve_idle_timeout_s=1800.0,
        serve_max_jobs=None,
        serve_ready_timeout=1800,
        job_fn=lambda **_k: _serving_manifest(),
    )
    captured: dict = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        scenarios = json.loads(Path(kwargs["scenarios_path"]).read_text())["scenarios"]
        captured["scenarios"] = scenarios
        return {"status": "completed", "results_collected": len(scenarios)}

    result = worker.submit_tasks(
        out_dir=tmp_path,
        tasks=["open the fridge door", "turn on the faucet"],
        scenarios_json=None,
        timeout_s=60.0,
        interval_s=1.0,
        submit_fn=fake_submit,
    )
    assert result["status"] == "completed"
    assert str(captured["manifest_path"]).endswith(J.JOB_MANIFEST_FILENAME)
    assert [s["task"] for s in captured["scenarios"]] == [
        "open the fridge door",
        "turn on the faucet",
    ]
    assert all(s["scenario_id"] for s in captured["scenarios"])


def test_submit_tasks_requires_serving_marker(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        worker.submit_tasks(
            out_dir=tmp_path,
            tasks=["x"],
            scenarios_json=None,
            timeout_s=1.0,
            interval_s=1.0,
            submit_fn=lambda **_k: {"status": "completed"},
        )


def test_stop_warm_worker_terminates_and_marks_marker(tmp_path: Path) -> None:
    worker.write_marker(tmp_path, {
        "schema_version": worker.MARKER_SCHEMA_VERSION,
        "status": "serving",
        "provider": "runpod",
        "pod_id": "pod-serve",
        "manifest_path": str(tmp_path / J.JOB_MANIFEST_FILENAME),
    })
    terminated: list[str] = []

    class FakeProvider:
        name = "runpod"

        def terminate(self, iid):
            terminated.append(iid)
            return {"status": "terminated", "http": 200}

        def inspect(self, iid):
            return {"status": "unavailable", "http": 404, "instance_id": iid}

        def billable_inventory(self, *, name_prefix: str):
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
                "name_prefix": name_prefix,
            }

    result = worker.stop_warm_worker(
        out_dir=tmp_path, provider_factory=lambda _name: FakeProvider()
    )
    assert result["status"] == "terminated"
    assert terminated == ["pod-serve"]
    marker = worker.read_marker(tmp_path)
    assert marker["status"] == "terminated"
    assert marker["teardown"] == {"status": "terminated", "http": 200}
    assert marker["teardown_proof"]["status"] == "PASS"
    assert marker["pending_teardown_close"]["status"] == "not_applicable"
    assert marker["paid_provider_lane_lease"]["release"]["all_providers_terminal"] is True


def test_stop_warm_worker_closes_pending_and_reclaims_stale_start_lease(
    tmp_path: Path,
) -> None:
    pending = open_pending_teardown(
        provider="runpod",
        lane=J.ISAAC_G1_KITCHEN_PARITY_LANE,
        run_id="warm-start",
        job_dir=tmp_path,
    )
    bind_pending_teardown_instance(pending["path"], "pod-serve")
    lease_dir = Path(os.environ["BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR"])
    stale_path = lease_path(
        "runpod", J.ISAAC_G1_KITCHEN_PARITY_LANE, lease_dir
    )
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text(
        json.dumps(
            {
                "schema_version": "paid_provider_lane_lease.v1",
                "provider": "runpod",
                "lane": J.ISAAC_G1_KITCHEN_PARITY_LANE,
                "owner_pid": 999_999_999,
                "hostname": "",
                "job_dir": "crashed-warm-start",
                "started_at_epoch": 1,
                "expires_at_epoch": 2,
            }
        ),
        encoding="utf-8",
    )
    worker.write_marker(
        tmp_path,
        {
            "schema_version": worker.MARKER_SCHEMA_VERSION,
            "status": "serving",
            "provider": "runpod",
            "pod_id": "pod-serve",
            "manifest_path": str(tmp_path / J.JOB_MANIFEST_FILENAME),
            "pending_teardown_record": pending["path"],
        },
    )

    class FakeProvider:
        name = "runpod"

        def terminate(self, iid):
            return {"status": "terminated", "http": 204, "instance_id": iid}

        def inspect(self, iid):
            return {"status": "unavailable", "http": 404, "instance_id": iid}

        def billable_inventory(self, *, name_prefix: str):
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
                "name_prefix": name_prefix,
            }

    result = worker.stop_warm_worker(
        out_dir=tmp_path, provider_factory=lambda _name: FakeProvider()
    )

    assert result["status"] == "terminated"
    assert result["pending_teardown_close"]["status"] == "closed"
    assert load_pending_teardowns() == []
    assert read_lease("runpod", J.ISAAC_G1_KITCHEN_PARITY_LANE, lease_dir) is None


def test_stop_warm_worker_retains_blocked_state_when_provider_is_still_live(
    tmp_path: Path,
) -> None:
    worker.write_marker(
        tmp_path,
        {
            "schema_version": worker.MARKER_SCHEMA_VERSION,
            "status": "serving",
            "provider": "runpod",
            "pod_id": "pod-live",
            "manifest_path": str(tmp_path / J.JOB_MANIFEST_FILENAME),
        },
    )

    class FakeProvider:
        name = "runpod"

        def terminate(self, iid):
            return {"status": "terminate_failed", "http": 500, "instance_id": iid}

        def inspect(self, iid):
            return {
                "status": "observed",
                "http": 200,
                "instance_id": iid,
                "desiredStatus": "RUNNING",
                "runtime_present": True,
            }

        def billable_inventory(self, *, name_prefix: str):
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 1,
                "resources": [{"instance_id": "pod-live", "name": name_prefix}],
                "name_prefix": name_prefix,
            }

    result = worker.stop_warm_worker(
        out_dir=tmp_path, provider_factory=lambda _name: FakeProvider()
    )

    assert result["status"] == "teardown_blocked"
    assert result["teardown_proof"]["status"] == "FAIL"
    marker = worker.read_marker(tmp_path)
    assert marker["status"] == "teardown_blocked"
    assert marker["terminated_at"] is None


# --------------------------- spend guard serve-pod tagging ---------------------------


def test_spend_guard_tags_expected_serve_pods_and_protects_them(tmp_path: Path) -> None:
    serve_dir = tmp_path / "output" / "warm_worker_kitchen"
    serve_dir.mkdir(parents=True)
    (serve_dir / guard.WARM_SERVE_MARKER_FILENAME).write_text(
        json.dumps(
            {
                "status": "serving",
                "pod_id": "pod-serve",
                "provider": "runpod",
                "lease_expires_at": "2027-02-01T00:00:00Z",
            }
        )
    )
    stale_dir = tmp_path / "output" / "old_worker"
    stale_dir.mkdir(parents=True)
    (stale_dir / guard.WARM_SERVE_MARKER_FILENAME).write_text(
        json.dumps({"status": "terminated", "pod_id": "pod-old", "provider": "runpod"})
    )

    serve_ids = guard.find_expected_serve_pod_ids(
        [tmp_path / "output"],
        now=1_800_000_000.0,
    )
    assert serve_ids == {"pod-serve"}

    instances = [
        guard.GpuInstance(
            provider="runpod", id="pod-serve", name="warm-serve", state="running",
            booted=True, live=True, cost_per_hr=0.69, age_seconds=7200.0,
        ),
        guard.GpuInstance(
            provider="runpod", id="pod-other", name="parity", state="running",
            booted=True, live=True, cost_per_hr=0.69, age_seconds=600.0,
        ),
    ]
    report = guard.build_report(
        instances,
        protected_ids={"pod-serve"},
        max_boot_seconds=480,
        serve_pod_ids=serve_ids,
    )
    assert "warm-serve worker (expected)" in report
    # the tag names the serve pod's row, not the ordinary one
    serve_row = next(line for line in report.splitlines() if "pod-serve" in line)
    other_row = next(line for line in report.splitlines() if "pod-other" in line)
    assert "expected" in serve_row
    assert "expected" not in other_row
