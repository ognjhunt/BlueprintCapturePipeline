"""Hermetic tests for the provider-agnostic GPU render launch layer (no GPU spend, no net).

Covers: the neutral RenderLaunchSpec, the registry, per-provider request translation
(RunPod pod body vs Vast offer-search/create-instance), credential availability, the
fail-closed no-spend guards, and provider-parameterized teardown.
"""
from __future__ import annotations

import io
import json
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
    assert names == {"runpod", "vast"}
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


# ----------------------------- Vast translation -----------------------------

def test_vast_build_request_offer_search_and_create(tmp_path: Path) -> None:
    req = VastRenderProvider().build_request(_spec(), tmp_path)
    # offer search filters to a single rentable on-demand GPU under the hourly rate
    sp = req["search_payload"]
    assert sp["type"] == "on-demand"
    assert sp["rentable"] == {"eq": True}
    assert sp["num_gpus"] == {"eq": 1}
    assert sp["dph_total"]["lte"] == pytest.approx(2.0)
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


def test_watch_and_collect_can_preserve_no_output_pod_for_warm_reuse(tmp_path: Path) -> None:
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
    assert fake.stopped == "inst-9"
    assert fake.terminated is None
    assert res["teardown"]["status"] == "stopped"


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
    assert res["teardown"]["status"] == "stopped"


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


def test_runpod_terminate_is_delete_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = RunPodRenderProvider().terminate("podabc")
    assert res["status"] == "blocked"  # no key -> no network, fail closed
    assert "runpod_api_key_missing" in res["blockers"]
    # terminate is distinct from stop (DELETE vs POST /stop) — both exist on the provider
    assert hasattr(RunPodRenderProvider(), "terminate") and hasattr(RunPodRenderProvider(), "stop")
