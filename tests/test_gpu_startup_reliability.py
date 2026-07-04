"""Hermetic tests for the cold-start reliability wiring.

Covers the three levers wired on 2026-07-02 (no network, no GPU, no secrets):

1. same-provider cold-create racing (``resolve_cold_race_contenders`` +
   ``_ColdCreateContender`` + ``race_launch`` with same-name contenders),
2. the earlier no-runtime dud guard defaulting on (600s),
3. the warm serve worker control plane (``scripts/run_warm_render_worker.py``)
   and gpu_spend_guard's expected-serve-pod tagging.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from blueprint_pipeline import isaac_g1_kitchen_parity_job as J
from blueprint_pipeline.provider_race import race_launch
from scripts import gpu_spend_guard as guard
from scripts import run_warm_render_worker as worker


# --------------------------- cold race contender resolution ---------------------------


def test_resolve_cold_race_contenders_default_env_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(J.COLD_RACE_CONTENDERS_ENV, raising=False)
    assert J.resolve_cold_race_contenders() == J.DEFAULT_COLD_RACE_CONTENDERS
    assert J.DEFAULT_COLD_RACE_CONTENDERS == 2

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


def test_paid_single_provider_job_races_two_cold_creates_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through run_isaac_g1_kitchen_parity_job: default cold_race_contenders=2
    launches two contenders on the one provider, keeps the marker winner, terminates the
    dud, and collects from the winner's contender dir."""
    monkeypatch.delenv(J.COLD_RACE_CONTENDERS_ENV, raising=False)
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
            "teardown": {"status": "terminated"},
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
    assert m["status"] == "completed"


def test_job_signature_defaults_enable_race_and_no_runtime_guard() -> None:
    sig = inspect.signature(J.run_isaac_g1_kitchen_parity_job)
    assert (
        sig.parameters["startup_no_runtime_timeout"].default
        == J.DEFAULT_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS
        == 600
    )
    # None -> resolve_cold_race_contenders() -> default 2 (env-overridable)
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
        job_fn=fake_job,
    )
    assert captured["serve"] is True
    assert captured["scenarios"] == []
    assert result["status"] == "serving"
    assert result["pod_id"] == "pod-serve"

    marker = json.loads((tmp_path / worker.WARM_SERVE_MARKER_FILENAME).read_text())
    assert marker["status"] == "serving"
    assert marker["pod_id"] == "pod-serve"
    assert marker["manifest_path"].endswith(J.JOB_MANIFEST_FILENAME)
    # the manifest itself is persisted for submit's URL-file lookup
    assert (tmp_path / J.JOB_MANIFEST_FILENAME).is_file()


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
        def terminate(self, iid):
            terminated.append(iid)
            return {"status": "terminated", "http": 200}

    result = worker.stop_warm_worker(
        out_dir=tmp_path, provider_factory=lambda _name: FakeProvider()
    )
    assert result["status"] == "terminated"
    assert terminated == ["pod-serve"]
    marker = worker.read_marker(tmp_path)
    assert marker["status"] == "terminated"
    assert marker["teardown"] == {"status": "terminated", "http": 200}


# --------------------------- spend guard serve-pod tagging ---------------------------


def test_spend_guard_tags_expected_serve_pods_and_protects_them(tmp_path: Path) -> None:
    serve_dir = tmp_path / "output" / "warm_worker_kitchen"
    serve_dir.mkdir(parents=True)
    (serve_dir / guard.WARM_SERVE_MARKER_FILENAME).write_text(
        json.dumps({"status": "serving", "pod_id": "pod-serve", "provider": "runpod"})
    )
    stale_dir = tmp_path / "output" / "old_worker"
    stale_dir.mkdir(parents=True)
    (stale_dir / guard.WARM_SERVE_MARKER_FILENAME).write_text(
        json.dumps({"status": "terminated", "pod_id": "pod-old", "provider": "runpod"})
    )

    serve_ids = guard.find_expected_serve_pod_ids([tmp_path / "output"])
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
