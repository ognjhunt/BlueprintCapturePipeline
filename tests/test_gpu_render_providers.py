"""Hermetic tests for the provider-agnostic GPU render launch layer (no GPU spend, no net).

Covers: the neutral RenderLaunchSpec, the registry, per-provider request translation
(RunPod pod body vs Vast offer-search/create-instance), credential availability, the
fail-closed no-spend guards, and provider-parameterized teardown.
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.gpu_render_providers import (
    RenderLaunchSpec,
    RunPodRenderProvider,
    VastRenderProvider,
    get_render_provider,
    list_render_providers,
)


def _spec(**over) -> RenderLaunchSpec:
    base = dict(
        name="blueprint-isaac-splat-render",
        image="img:tag",
        env={
            "BLUEPRINT_EVAL_MANIFEST_URI": "https://spaces.example/bundle.zip?sig=A",
            "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": "https://spaces.example/out.zip?sig=B",
            "CAMERAS_FILE": "cameras_canary.json",
        },
        bootstrap_argv=["-lc", "echo container_bash_started; run render"],
    )
    base.update(over)
    return RenderLaunchSpec(**base)


# ----------------------------- spec + registry -----------------------------

def test_render_launch_spec_bootstrap_script_is_last_argv() -> None:
    spec = _spec(bootstrap_argv=["-lc", "the-script-body"])
    assert spec.bootstrap_script == "the-script-body"
    assert spec.entrypoint == ["bash"]
    assert spec.container_disk_gb >= 120  # must hold the 10.7GB image + outputs


def test_registry_returns_known_providers_and_rejects_unknown() -> None:
    assert isinstance(get_render_provider("runpod"), RunPodRenderProvider)
    assert isinstance(get_render_provider("vast"), VastRenderProvider)
    assert isinstance(get_render_provider(None), RunPodRenderProvider)  # default
    assert isinstance(get_render_provider("VAST"), VastRenderProvider)  # case-insensitive
    with pytest.raises(ValueError):
        get_render_provider("lambda-labs")


def test_list_render_providers_reports_both_with_availability() -> None:
    listed = list_render_providers()
    names = {p["provider"] for p in listed}
    assert names == {"runpod", "vast", "digitalocean"}
    for entry in listed:
        assert "available" in entry  # bool reflecting credential presence


# ----------------------------- RunPod translation -----------------------------

def test_runpod_build_request_is_pod_body(tmp_path: Path) -> None:
    body = RunPodRenderProvider().build_request(_spec(), tmp_path)
    assert body["imageName"] == "img:tag"
    assert body["dockerEntrypoint"] == ["bash"]
    assert body["dockerStartCmd"] == ["-lc", "echo container_bash_started; run render"]
    assert "NVIDIA L40S" in body["gpuTypeIds"]
    assert body["containerDiskInGb"] >= 120
    assert body["env"]["BLUEPRINT_EVAL_MANIFEST_URI"].endswith("sig=A")
    assert body["cloudType"] == "SECURE"


def test_runpod_launch_fail_closed_without_key(tmp_path: Path, monkeypatch) -> None:
    # point secret lookups at an empty dir so no key is found and no network call happens
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = RunPodRenderProvider().launch(tmp_path, {"imageName": "x"}, cold=True)
    assert res["status"] == "blocked"
    assert "runpod_api_key_missing" in res["blockers"]


def test_runpod_warm_start_rejection_is_recorded_before_cold_fallback(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        assert key == "rp-key"
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/start" and method == "POST":
            return 409, {"error": "pod is not startable from EXITED"}
        if path == "/pods" and method == "POST":
            return 201, {"id": "cold-1"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)
    res = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path,
        {"imageName": "img:tag", "env": {}, "dockerStartCmd": ["-lc", "run"]},
        cold=False,
    )

    assert res["status"] == "launched"
    assert res["instance_id"] == "cold-1"
    assert res["mode"] == "cold_create"
    assert res["attempts"][0] == {
        "pod_id": "warm-1",
        "get_status": 200,
        "desiredStatus": "EXITED",
        "update_status": 200,
        "start_status": 409,
        "start_error": "pod is not startable from EXITED",
    }
    assert res["attempts"][1]["cold_create_status"] == 201
    assert calls == [
        ("GET", "/pods/warm-1"),
        ("POST", "/pods/warm-1/update"),
        ("POST", "/pods/warm-1/start"),
        ("POST", "/pods"),
    ]


def test_runpod_warm_only_blocks_without_cold_create(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 200, {"id": "warm-1", "desiredStatus": "EXITED"}
        if path == "/pods/warm-1/start" and method == "POST":
            return 409, {"error": "pod is not startable from EXITED"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)
    res = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path,
        {"imageName": "img:tag", "env": {}, "dockerStartCmd": ["-lc", "run"]},
        cold=False,
        allow_cold_fallback=False,
    )

    assert res["status"] == "blocked"
    assert "warm_restart_failed_cold_fallback_disabled" in res["blockers"]
    assert res["attempts"][0]["start_status"] == 409
    assert ("POST", "/pods") not in calls


def test_runpod_warm_update_failure_does_not_start_stale_command(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        if path == "/pods/warm-1" and method == "GET":
            return 200, {"id": "warm-1", "desiredStatus": "STOPPED"}
        if path == "/pods/warm-1/update" and method == "POST":
            return 400, {"error": "invalid update"}
        if path == "/pods" and method == "POST":
            return 201, {"id": "cold-1"}
        raise AssertionError((method, path, body))

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)
    res = RunPodRenderProvider(warm_candidates=("warm-1",)).launch(
        tmp_path,
        {"imageName": "img:tag", "env": {}, "dockerStartCmd": ["-lc", "run"]},
        cold=False,
    )

    assert res["status"] == "launched"
    assert res["instance_id"] == "cold-1"
    assert res["attempts"][0]["update_status"] == 400
    assert res["attempts"][0]["update_error"] == "invalid update"
    assert ("POST", "/pods/warm-1/start") not in calls


def test_runpod_teardown_404_is_already_gone_success(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        calls.append((method, path))
        assert key == "rp-key"
        return 404, {"error": "pod not found"}

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)

    stop = RunPodRenderProvider().stop("pod-missing")
    terminate = RunPodRenderProvider().terminate("pod-missing")

    assert stop == {"status": "stopped", "http": 404, "already_gone": True}
    assert terminate == {"status": "terminated", "http": 404, "already_gone": True}
    assert calls == [
        ("POST", "/pods/pod-missing/stop"),
        ("DELETE", "/pods/pod-missing"),
    ]


def test_runpod_inspect_redacts_and_marks_pre_runtime(monkeypatch) -> None:
    def fake_key(_self):
        return "rp-key"

    def fake_call(method, path, body, *, key, timeout=90):
        assert method == "GET"
        assert path == "/pods/pod-1"
        assert body is None
        assert key == "rp-key"
        return 200, {
            "id": "pod-1",
            "desiredStatus": "RUNNING",
            "publicIp": "",
            "machineId": "machine-a",
            "costPerHr": 0.69,
            "createdAt": "2026-07-01 21:42:02.335 +0000 UTC",
            "lastStartedAt": "2026-07-01 21:42:02.33 +0000 UTC",
            "lastStatusChange": "Rented by User",
            "imageName": "img:tag",
        }

    monkeypatch.setattr(RunPodRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers._runpod_call", fake_call)

    res = RunPodRenderProvider().inspect("pod-1")

    assert res["status"] == "observed"
    assert res["runtime_present"] is False
    assert res["public_ip_present"] is False
    assert res["machineId"] == "machine-a"
    assert res["raw_provider_response_recorded"] is False
    assert "env" not in res and "dockerStartCmd" not in res


# ----------------------------- Vast translation -----------------------------

def test_vast_build_request_offer_search_and_create(tmp_path: Path) -> None:
    req = VastRenderProvider().build_request(_spec(), tmp_path)
    # offer search filters to a single rentable on-demand GPU under the hourly rate
    sp = req["search_payload"]
    assert sp["type"] == "on-demand"
    assert sp["rentable"] == {"eq": True}
    assert sp["num_gpus"] == {"eq": 1}
    assert sp["dph_total"]["lte"] == pytest.approx(5.0)
    # create-instance body: args mode runs our bootstrap via bash, env carries the signed urls
    cp = req["create_payload"]
    assert cp["image"] == "img:tag"
    assert cp["disk"] == req["disk"] >= 120
    assert cp["runtype"] == "args"
    assert cp["target_state"] == "running"
    assert cp["args_str"].startswith("bash -lc")
    assert "container_bash_started" in cp["args_str"]
    assert cp["env"]["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"].endswith("sig=B")
    assert req["create_endpoint"] == "PUT /asks/{ask_contract_id}/"


def test_vast_launch_fail_closed_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = VastRenderProvider().launch(tmp_path, {"search_payload": {}}, cold=False)
    assert res["status"] == "blocked"
    assert "vast_api_key_missing" in res["blockers"]


def test_vast_stop_fail_closed_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = VastRenderProvider().stop("12345")
    assert res["status"] == "blocked"
    assert "vast_api_key_missing" in res["blockers"]


def test_vast_launch_writes_started_instance_id(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=45):
        calls.append((method, path))
        assert api_key == "vast-key"
        if method == "POST" and path == "/bundles/":
            return 200, {"offers": [{"id": "raw-offer"}]}
        if method == "PUT" and path == "/asks/ask-1/":
            return 200, {"new_contract": 12345}
        raise AssertionError((method, path, payload, timeout_seconds))

    offer = {
        "ask_contract_id": "ask-1",
        "gpu_name": "RTX 4090",
        "hourly_rate_usd": 0.44,
    }

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._offers_from_response", lambda _resp: [offer])
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._select_offer", lambda offers, **_kw: offers[0])

    req = VastRenderProvider().build_request(_spec(), tmp_path)
    res = VastRenderProvider().launch(tmp_path, req)

    assert res["status"] == "launched"
    assert res["instance_id"] == "12345"
    assert res["mode"] == "vast_on_demand"
    assert (tmp_path / "started_vast_instance_id.txt").read_text() == "12345"
    assert calls == [("POST", "/bundles/"), ("PUT", "/asks/ask-1/")]


def test_vast_terminate_delegates_to_destroy_instance_delete(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=30):
        calls.append((method, path))
        assert api_key == "vast-key"
        assert payload is None
        return 204, {}

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)

    res = VastRenderProvider().terminate("inst-123")

    assert res["status"] == "stopped"
    assert res["http"] == 204
    assert calls == [("DELETE", "/instances/inst-123/")]


def test_vast_teardown_404_is_already_gone_success(monkeypatch) -> None:
    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=30):
        assert method == "DELETE"
        assert path == "/instances/inst-missing/"
        assert api_key == "vast-key"
        raise urllib.error.HTTPError(
            url="https://console.vast.ai/api/v0/instances/inst-missing/",
            code=404,
            msg="not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)

    assert VastRenderProvider().stop("inst-missing") == {
        "status": "stopped",
        "http": 404,
        "already_gone": True,
    }


# ----------------------------- availability reflects secrets -----------------------------

def test_availability_reflects_secret_presence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    assert RunPodRenderProvider().available()["available"] is False
    assert VastRenderProvider().available()["available"] is False
    (tmp_path / "runpod_api_key").write_text("rp-key")
    (tmp_path / "vast_api_key").write_text("vast-key")
    assert RunPodRenderProvider().available()["available"] is True
    assert VastRenderProvider().available()["available"] is True


# ----------------------------- teardown is provider-parameterized -----------------------------

def test_watch_and_collect_tears_down_via_provider(tmp_path: Path) -> None:
    from blueprint_pipeline.isaac_particlefield_render_job import watch_and_collect

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.terminated: str | None = None

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()
    # max_seconds=0 -> skip the poll loop entirely (no network), go straight to teardown
    res = watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=0)
    assert fake.terminated == "inst-9"  # blocked/no-result pod is DELETED
    assert res["status"] == "blocked"  # nothing rendered
    assert res["teardown"]["status"] == "terminated"


def test_watch_and_collect_terminates_no_output_pod_even_when_preserve_requested(tmp_path: Path) -> None:
    from blueprint_pipeline.isaac_particlefield_render_job import watch_and_collect

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = watch_and_collect(
        job_dir,
        tmp_path / "out",
        "inst-9",
        provider=fake,
        max_seconds=0,
        preserve_instance=True,
    )

    assert res["status"] == "blocked"
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["teardown"]["status"] == "terminated"
    assert res["teardown_reason"] == "timeout_without_runner_done_terminated"
    assert res["timed_out_without_runner_done"] is True


def test_watch_and_collect_stops_successful_pod_for_warm_reuse(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_done", "rc": 0}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({"status": "completed"}))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "completed"
    assert fake.stopped == "inst-9"
    assert fake.terminated is None
    assert res["teardown_reason"] == "runner_done_preserved_for_warm_reuse"
    assert res["teardown"]["status"] == "stopped"


def test_watch_and_collect_terminates_digitalocean_runner_done(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "digitalocean"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 201}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_done", "rc": 0}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({"status": "completed"}))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "completed"
    assert fake.terminated == "inst-9"
    assert fake.stopped is None
    assert res["teardown_reason"] == "runner_done_terminated_no_warm_reuse"
    assert res["teardown"]["status"] == "terminated"


def test_watch_and_collect_stops_blocked_runner_pod_for_warm_reuse(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_done", "rc": 0}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({
            "status": "blocked",
            "blockers": ["placement_validation_failed"],
        }))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "blocked"
    assert fake.stopped == "inst-9"
    assert fake.terminated is None
    assert res["teardown"]["status"] == "stopped"


def test_watch_and_collect_terminates_runner_timeout(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({
            "phase": "runner_timeout",
            "timeout_seconds": 840,
        }))
        zf.writestr("runner_console.log", "SimulationApp boot did not finish\n")
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "blocked"
    assert res["runner_timeout_observed"] is True
    assert res["timed_out_without_runner_done"] is False
    assert res["teardown_reason"] == "runner_timeout_terminated"
    assert fake.terminated == "inst-9"
    assert fake.stopped is None


def test_watch_and_collect_ignores_stale_result_before_runner_done(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped", "http": 204}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated", "http": 204}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "kitchen_fetching"}))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({
            "status": "blocked",
            "blockers": ["stale_previous_run"],
        }))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(job.time, "time", lambda: next(clock, 2.0))
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(job_dir, tmp_path / "out", "inst-9", provider=fake, max_seconds=1, poll=1)

    assert res["status"] == "blocked"
    assert res["last_bootstrap"]["phase"] == "kitchen_fetching"
    assert res["runner_result"]["blockers"] == ["stale_previous_run"]
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["teardown_reason"] == "timeout_without_runner_done_terminated"


def test_watch_and_collect_terminates_current_final_result_without_runner_done(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped"}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated"}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({
            "phase": "runner_starting",
            "launch_session_id": "launch-123",
        }))
        zf.writestr("isaac_g1_kitchen_parity_result.json", json.dumps({
            "status": "blocked",
            "scenarios_executed": 0,
            "blockers": ["isaac_runner_exception_before_scenario_outcome"],
        }))
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    (job_dir / "launch_session_nonce.txt").write_text("launch-123")
    fake = _FakeProvider()

    res = job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "inst-9",
        provider=fake,
        max_seconds=1,
        poll=1,
        preserve_instance=True,
    )

    assert res["status"] == "blocked"
    assert res["runner_result"]["scenarios_executed"] == 0
    assert res["timed_out_without_runner_done"] is False
    assert res["runner_done_observed"] is False
    assert res["final_result_without_runner_done"] is True
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["teardown_reason"] == "final_result_without_runner_done_terminated"


def test_watch_and_collect_terminates_heartbeat_timeout_despite_preserve(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from blueprint_pipeline import isaac_particlefield_render_job as job

    class _FakeProvider:
        name = "fake"

        def __init__(self) -> None:
            self.stopped: str | None = None
            self.terminated: str | None = None

        def stop(self, instance_id: str) -> dict:
            self.stopped = instance_id
            return {"status": "stopped"}

        def terminate(self, instance_id: str) -> dict:
            self.terminated = instance_id
            return {"status": "terminated"}

    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bootstrap.json", json.dumps({"phase": "runner_starting"}))
        zf.writestr("runner_console.log", "Isaac is still starting")
    payload_bytes = payload.getvalue()

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(job.time, "time", lambda: next(clock, 2.0))
    monkeypatch.setattr(job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(job.urllib.request, "urlopen", lambda _url, timeout=60: _Response())
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
    fake = _FakeProvider()

    res = job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "inst-9",
        provider=fake,
        max_seconds=1,
        poll=1,
        preserve_instance=True,
    )

    assert res["status"] == "blocked"
    assert fake.stopped is None
    assert fake.terminated == "inst-9"
    assert res["last_bootstrap"]["phase"] == "runner_starting"
    assert res["timed_out_without_runner_done"] is True


def test_runpod_terminate_is_delete_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = RunPodRenderProvider().terminate("podabc")
    assert res["status"] == "blocked"  # no key -> no network, fail closed
    assert "runpod_api_key_missing" in res["blockers"]
    # terminate is distinct from stop (DELETE vs POST /stop) — both exist on the provider
    assert hasattr(RunPodRenderProvider(), "terminate") and hasattr(RunPodRenderProvider(), "stop")


def test_vast_launch_retries_next_offer_on_create_400(tmp_path: Path, monkeypatch) -> None:
    """A stale ask 400s at create; the launch must record the error body and
    fall through to the next candidate offer instead of blocking the race."""
    import io
    import urllib.error

    def fake_key(_self):
        return "vast-key"

    def fake_api_json(*, method, path, api_key, payload=None, timeout_seconds=45):
        if method == "POST" and path == "/bundles/":
            return 200, {"offers": ["raw"]}
        if method == "PUT" and path == "/asks/ask-stale/":
            raise urllib.error.HTTPError(
                "https://vast/asks/ask-stale/", 400, "Bad Request", None,
                io.BytesIO(b'{"success": false, "msg": "ask expired"}'))
        if method == "PUT" and path == "/asks/ask-fresh/":
            return 200, {"new_contract": 777}
        raise AssertionError((method, path))

    stale = {"ask_contract_id": "ask-stale", "gpu_name": "RTX 4090", "hourly_rate_usd": 0.4}
    fresh = {"ask_contract_id": "ask-fresh", "gpu_name": "RTX 4090", "hourly_rate_usd": 0.5}

    def fake_select(offers, **_kw):
        return offers[0] if offers else None

    monkeypatch.setattr(VastRenderProvider, "_key", fake_key)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._api_json", fake_api_json)
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._offers_from_response",
                        lambda _resp: [stale, fresh])
    monkeypatch.setattr("blueprint_pipeline.vast_provider_adapter._select_offer", fake_select)

    req = VastRenderProvider().build_request(_spec(), tmp_path)
    res = VastRenderProvider().launch(tmp_path, req)

    assert res["status"] == "launched"
    assert res["instance_id"] == "777"
    create_errors = [a for a in res.get("attempts", []) if a.get("create_http_status") == 400]
    assert create_errors and "ask expired" in str(create_errors[0].get("create_error_body"))


def test_default_runpod_gpu_types_exclude_consumer_4090_pool(monkeypatch) -> None:
    """The GeForce 4090 pool produced ~10 dud nodes on 2026-07-02 (never-started
    containers, driver segfaults, wedged workers). Default to the datacenter RTX
    tier; BLUEPRINT_RUNPOD_GPU_TYPES re-adds types for deliberate experiments."""
    monkeypatch.delenv("BLUEPRINT_RUNPOD_GPU_TYPES", raising=False)
    spec = _spec()
    assert "NVIDIA GeForce RTX 4090" not in spec.gpu_types
    assert spec.gpu_types[0] == "NVIDIA L40S"
    assert all(("GeForce" not in g) for g in spec.gpu_types)

    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_GPU_TYPES",
        "NVIDIA GeForce RTX 4090, NVIDIA L40S",
    )
    spec2 = _spec()
    assert spec2.gpu_types == ("NVIDIA GeForce RTX 4090", "NVIDIA L40S")


# ----------------------------- DigitalOcean GPU Droplets -----------------------------

def test_digitalocean_provider_is_registered() -> None:
    from blueprint_pipeline.gpu_render_providers import DigitalOceanRenderProvider

    assert "digitalocean" in {p["provider"] for p in list_render_providers()}
    assert isinstance(get_render_provider("digitalocean"), DigitalOceanRenderProvider)


def test_digitalocean_build_request_wraps_worker_in_user_data(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline.gpu_render_providers import DigitalOceanRenderProvider

    monkeypatch.delenv("BLUEPRINT_DO_GPU_SIZE", raising=False)
    monkeypatch.delenv("BLUEPRINT_DO_GPU_REGION", raising=False)
    spec = _spec()
    body = DigitalOceanRenderProvider().build_request(spec, tmp_path)
    assert body["size"] == "gpu-6000adax1-48gb"   # RT cores + 48GB default
    assert body["region"] == "atl1"
    assert body["image"] == "gpu-h100x1-base"     # NVIDIA AI/ML-ready (drivers+docker)
    ud = body["user_data"]
    assert "set -x" not in ud
    assert "set -euo pipefail" in ud
    assert "mkdir -p /root/blueprint-workspace/out" in ud
    assert '"docker", "run", "-d"' in ud
    assert '"--gpus", "all"' in ud
    assert '"--user", "0:0"' in ud
    assert '"-v", "/root/blueprint-workspace:/workspace"' in ud
    assert '"--workdir", "/workspace"' in ud
    assert spec.image in ud
    # env + bootstrap ride base64 so presigned URLs / scripts survive shell quoting
    assert "base64 -d" in ud
    assert "$(cat /root/blueprint_run.sh)" not in ud
    assert "subprocess.check_call(cmd)" in ud
    assert "blueprint_argv_decoded.json" in ud
    assert body["tags"] == ["blueprint-isaac-render"]


def test_digitalocean_launch_fail_closed_without_token(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline.gpu_render_providers import DigitalOceanRenderProvider

    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tmp_path / "missing"))
    p = DigitalOceanRenderProvider()
    res = p.launch(tmp_path, {"name": "x"})
    assert res["status"] == "blocked"
    assert "digitalocean_token_missing" in res["blockers"]
    assert p.available()["available"] is False


def test_digitalocean_launch_creates_droplet_and_writes_id(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.delenv("BLUEPRINT_DO_SSH_KEY_IDS", raising=False)
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS_FILE", str(tmp_path / "missing_do_ssh_keys"))
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        assert token == "t-redacted"
        if method == "GET" and path == "/account/keys?per_page=200":
            return 200, {"ssh_keys": [{"id": 98765, "name": "worker-key"}]}
        if method == "POST" and path == "/droplets":
            assert body["ssh_keys"] == [98765]
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)
    p = G.DigitalOceanRenderProvider()
    res = p.launch(tmp_path, p.build_request(_spec(), tmp_path))
    assert res["status"] == "launched"
    assert res["instance_id"] == "4242"
    assert res["mode"] == "do_gpu_droplet"
    assert res["ssh_key_configuration"]["source"] == "account_keys_api_first_available"
    assert (tmp_path / "started_do_droplet_id.txt").read_text() == "4242"


def test_digitalocean_launch_regenerates_user_data_after_nonce_injection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    created: dict = {}

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert token == "t-redacted"
        if method == "POST" and path == "/droplets":
            created["body"] = body
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)
    provider = G.DigitalOceanRenderProvider()
    request = provider.build_request(_spec(), tmp_path)
    request["env"]["BLUEPRINT_LAUNCH_SESSION_ID"] = "nonce-123"

    res = provider.launch(tmp_path, request)

    assert res["status"] == "launched"
    body = created["body"]
    assert "env" not in body
    assert "_blueprint_worker_image" not in body
    match = re.search(
        r"echo ([A-Za-z0-9+/=]+) \| base64 -d > /root/blueprint_worker.env",
        body["user_data"],
    )
    assert match is not None
    env_text = base64.b64decode(match.group(1)).decode()
    assert "BLUEPRINT_LAUNCH_SESSION_ID=nonce-123" in env_text


def test_digitalocean_launch_uses_configured_ssh_keys_without_account_lookup(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123, fingerprint-abc")
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        assert path != "/account/keys?per_page=200"
        if method == "POST" and path == "/droplets":
            assert body["ssh_keys"] == [123, "fingerprint-abc"]
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        raise AssertionError((method, path))

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path),
    )

    assert res["status"] == "launched"
    assert res["ssh_key_configuration"]["source"] == "BLUEPRINT_DO_SSH_KEY_IDS"
    assert calls == [("POST", "/droplets")]


def test_digitalocean_launch_retries_gpu_size_region_unavailable(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-6000adax1-48gb,gpu-l40sx1-48gb")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_REGIONS", "atl1,nyc2")
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert method == "POST"
        assert path == "/droplets"
        calls.append((body["size"], body["region"]))
        if body["size"] == "gpu-l40sx1-48gb" and body["region"] == "nyc2":
            return 202, {"droplet": {"id": 4242, "status": "new"}}
        return 422, {
            "error": '{"id":"unprocessable_entity","message":"Size is not available in this region."}\n'
        }

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path),
    )

    assert res["status"] == "launched"
    assert res["instance_id"] == "4242"
    assert calls == [
        ("gpu-6000adax1-48gb", "atl1"),
        ("gpu-6000adax1-48gb", "nyc2"),
        ("gpu-l40sx1-48gb", "atl1"),
        ("gpu-l40sx1-48gb", "nyc2"),
    ]
    assert res["attempts"][-1]["size"] == "gpu-l40sx1-48gb"
    assert res["attempts"][-1]["region"] == "nyc2"
    assert res["budget_policy"]["max_hourly_rate_usd"] == pytest.approx(1.75)


def test_digitalocean_launch_blocks_h200_without_hourly_budget_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS", "123")
    monkeypatch.setenv("BLUEPRINT_DO_GPU_SIZES", "gpu-h200x1-141gb")
    monkeypatch.delenv("BLUEPRINT_DO_MAX_HOURLY_RATE_USD", raising=False)

    def fake_call(method, path, body=None, *, token, timeout=90):
        raise AssertionError("must not create an over-budget DigitalOcean droplet")

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path),
    )

    assert res["status"] == "blocked"
    assert "digitalocean_gpu_size_over_budget" in res["blockers"]
    assert res["budget_policy"]["rejected_size_candidates"] == [
        {
            "size": "gpu-h200x1-141gb",
            "hourly_rate_usd": 3.44,
            "reason": "over_max_hourly_rate",
        }
    ]


def test_digitalocean_launch_blocks_without_ssh_key(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t-redacted")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    monkeypatch.delenv("BLUEPRINT_DO_SSH_KEY_IDS", raising=False)
    monkeypatch.setenv("BLUEPRINT_DO_SSH_KEY_IDS_FILE", str(tmp_path / "missing_do_ssh_keys"))

    def fake_call(method, path, body=None, *, token, timeout=90):
        assert method == "GET"
        assert path == "/account/keys?per_page=200"
        return 200, {"ssh_keys": []}

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().launch(
        tmp_path,
        G.DigitalOceanRenderProvider().build_request(_spec(), tmp_path),
    )

    assert res["status"] == "blocked"
    assert "digitalocean_ssh_key_missing" in res["blockers"]
    assert res["ssh_key_configuration"]["raw_provider_response_recorded"] is False


def test_digitalocean_terminate_deletes_droplet(monkeypatch, tmp_path: Path) -> None:
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))
    calls = []

    def fake_call(method, path, body=None, *, token, timeout=90):
        calls.append((method, path))
        return 204, {}

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().terminate("4242")
    assert res["status"] == "terminated"
    assert ("DELETE", "/droplets/4242") in calls


def test_digitalocean_stop_warns_droplets_bill_while_off(monkeypatch, tmp_path: Path) -> None:
    """Powered-off droplets still bill full price; stop() must say so instead of
    silently pretending it saved money."""
    from blueprint_pipeline import gpu_render_providers as G

    tok = tmp_path / "do_token"
    tok.write_text("t")
    monkeypatch.setenv("DIGITALOCEAN_TOKEN_FILE", str(tok))

    def fake_call(method, path, body=None, *, token, timeout=90):
        return 201, {"action": {"id": 1, "status": "in-progress"}}

    monkeypatch.setattr(G, "_do_call", fake_call)
    res = G.DigitalOceanRenderProvider().stop("4242")
    assert res["status"] == "stopped"
    assert "billing" in json.dumps(res).lower()
