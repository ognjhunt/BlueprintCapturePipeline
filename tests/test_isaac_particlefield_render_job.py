"""Hermetic tests for the productionized Isaac ParticleField render job (no GPU spend).

The deterministic build/plan steps are pinned here; staging + launch + render are the
network/GPU integration steps gated behind allow_paid.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import blueprint_pipeline.isaac_particlefield_render_job as render_job
from blueprint_pipeline import paid_lane_guard
from blueprint_pipeline.isaac_particlefield_render_job import (
    build_launch_request,
    build_render_bundle,
    default_image,
    docker_start_cmd,
    ensure_particlefield_usd,
)

_CAMS = [
    {"id": "third_person", "spec": {"pos": [0, 0, 5], "target": [0, 0, 0], "fov": 60, "up": [0, 0, 1]}},
    {"id": "overhead", "spec": {"pos": [0, 0, 9], "target": [0, 0, 0], "fov": 70, "up": [0, 1, 0]}},
]


def test_docker_start_cmd_is_robust_and_invokes_runner() -> None:
    dsc = docker_start_cmd()
    assert dsc[0] == "-lc"
    body = dsc[1]
    # writes scripts to files (not stdin tricks), has the early marker, runs the runner with --usdc
    assert "cat > /workspace/boot.py" in body
    assert "container_bash_started" in body  # early diagnostic marker
    assert "run_isaac_splat_nurec_render.py" in body
    assert "--usdc" in body
    assert "/isaac-sim/python.sh /workspace/boot.py" in body
    assert 'mark("runner_done", rc=rc, idle_ttl_seconds=IDLE_TTL)' in body
    assert render_job.RENDER_POD_HARD_TTL_ENV in body
    assert render_job.RENDER_POD_IDLE_TTL_ENV in body
    assert "pod_hard_ttl_exceeded" in body
    assert "pod_idle_ttl_exceeded" in body
    assert "pkill -TERM -P $$" in body
    assert "while True:" in body and "putout()" in body


def test_build_render_bundle_contains_all_inputs(tmp_path: Path) -> None:
    usdc = tmp_path / "scene_particlefield.usdc"
    usdc.write_bytes(b"#usdc-fake")
    zip_path = build_render_bundle(usdc_path=usdc, cameras=_CAMS, out_dir=tmp_path / "job")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert {"scene_particlefield.usdc", "cameras.json", "cameras_canary.json",
            "run_isaac_splat_nurec_render.py"} <= names
    # canary cameras = just the establishing view
    with zipfile.ZipFile(zip_path) as zf:
        canary = json.loads(zf.read("cameras_canary.json"))
    assert [c["id"] for c in canary] == ["third_person"]


def test_build_launch_request_shape(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    req = build_launch_request(jd, image="img:tag", cameras_file="cameras_canary.json")
    assert req["imageName"] == "img:tag"
    assert req["dockerEntrypoint"] == ["bash"]
    assert req["env"]["BLUEPRINT_EVAL_MANIFEST_URI"].endswith("sig=A")
    assert req["env"]["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"].endswith("sig=B")
    assert req["env"]["CAMERAS_FILE"] == "cameras_canary.json"
    assert req["env"][render_job.RENDER_POD_HARD_TTL_ENV] == str(
        render_job.DEFAULT_RENDER_POD_HARD_TTL_SECONDS
    )
    assert req["env"][render_job.RENDER_POD_IDLE_TTL_ENV] == str(
        render_job.DEFAULT_RENDER_POD_IDLE_TTL_SECONDS
    )
    assert req["containerDiskInGb"] >= 120  # must hold the 10.7GB image + outputs
    assert "NVIDIA L40S" in req["gpuTypeIds"]


def test_ensure_particlefield_usd_passthrough_for_usdc(tmp_path: Path) -> None:
    usdc = tmp_path / "scene.usdc"
    usdc.write_bytes(b"#usdc")
    res = ensure_particlefield_usd(usdc, tmp_path / "asset")
    assert res["status"] == "completed"
    assert res["usdc"] == str(usdc)
    assert res["source_kind"] == "particlefield_usd"


def test_default_image_is_isaac_worker() -> None:
    img = default_image()
    assert "isaac" in img and ":" in img


class _CollectProvider:
    def __init__(self) -> None:
        self.terminated: list[str] = []

    def terminate(self, instance_id: str) -> dict:
        self.terminated.append(instance_id)
        return {"status": "terminated"}

    def inspect(self, instance_id: str) -> dict:
        if instance_id in self.terminated:
            return {"status": "unavailable", "http": 404, "instance_id": instance_id}
        return {
            "status": "observed",
            "http": 200,
            "instance_id": instance_id,
            "desiredStatus": "RUNNING",
        }


def _zip_bytes(files: dict[str, str]) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return buf.getvalue()


def test_watch_and_collect_accepts_g1_parity_result(tmp_path: Path, monkeypatch) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip")
    payload = {
        "bootstrap.json": json.dumps({"phase": "runner_done", "rc": 0}),
        "isaac_g1_kitchen_parity_result.json": json.dumps({"status": "completed"}),
        "runner_console.log": "parity complete\n",
    }

    class _Response:
        def read(self) -> bytes:
            return _zip_bytes(payload)

    monkeypatch.setattr(render_job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(render_job.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())
    provider = _CollectProvider()
    result = render_job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "pod-1",
        provider=provider,
        max_seconds=10,
        poll=1,
    )
    assert result["status"] == "completed"
    assert result["runner_result_source"] == "isaac_g1_kitchen_parity_result.json"
    assert result["runner_result"]["status"] == "completed"
    assert result["last_bootstrap"]["phase"] == "runner_done"
    assert provider.terminated == ["pod-1"]


def test_watch_and_collect_preserves_blocked_bootstrap_and_console(tmp_path: Path, monkeypatch) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip")
    payload = {
        "bootstrap.json": json.dumps({"phase": "runner_done", "rc": 1}),
        "runner_console.log": "Isaac failed after render-product creation\n",
    }

    class _Response:
        def read(self) -> bytes:
            return _zip_bytes(payload)

    monkeypatch.setattr(render_job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(render_job.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())
    result = render_job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "pod-2",
        provider=_CollectProvider(),
        max_seconds=10,
        poll=1,
    )
    assert result["status"] == "blocked"
    assert result["runner_result"] == {}
    assert result["last_bootstrap"] == {"phase": "runner_done", "rc": 1}
    assert "render-product creation" in result["runner_console_tail"]


def test_watch_and_collect_terminates_post_marker_progress_stall(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip")
    payload = {
        "bootstrap.json": json.dumps({"phase": "container_bash_started"}),
    }
    payload_bytes = _zip_bytes(payload)

    class _Response:
        def read(self) -> bytes:
            return payload_bytes

    now = 0.0

    def _time() -> float:
        return now

    def _sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(render_job.time, "time", _time)
    monkeypatch.setattr(render_job.time, "sleep", _sleep)
    monkeypatch.setattr(render_job.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())
    provider = _CollectProvider()

    result = render_job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "pod-stall",
        provider=provider,
        max_seconds=10,
        poll=1,
        progress_timeout_seconds=2,
    )

    assert result["status"] == "blocked"
    assert provider.terminated == ["pod-stall"]
    assert result["teardown_reason"] == "post_marker_progress_timeout_terminated"
    assert result["post_marker_progress_timeout_observed"] is True
    assert result["post_marker_progress_timeout"]["phase"] == "container_bash_started"
    assert result["last_bootstrap"]["phase"] == "container_bash_started"


def test_watch_and_collect_treats_same_phase_marker_changes_as_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip")
    payloads = [
        _zip_bytes({"bootstrap.json": json.dumps({"phase": "bootstrap_fetch_progress", "bytes_read": 1})}),
        _zip_bytes({"bootstrap.json": json.dumps({"phase": "bootstrap_fetch_progress", "bytes_read": 2})}),
        _zip_bytes({"bootstrap.json": json.dumps({"phase": "bootstrap_fetch_progress", "bytes_read": 3})}),
        _zip_bytes(
            {
                "bootstrap.json": json.dumps({"phase": "runner_done", "rc": 0}),
                "isaac_runtime_result.json": json.dumps({"status": "completed"}),
            }
        ),
    ]
    read_count = {"value": 0}

    class _Response:
        def read(self) -> bytes:
            index = min(read_count["value"], len(payloads) - 1)
            read_count["value"] += 1
            return payloads[index]

    now = 0.0

    def _time() -> float:
        return now

    def _sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    monkeypatch.setattr(render_job.time, "time", _time)
    monkeypatch.setattr(render_job.time, "sleep", _sleep)
    monkeypatch.setattr(render_job.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())

    result = render_job.watch_and_collect(
        job_dir,
        tmp_path / "out",
        "pod-active-progress",
        provider=_CollectProvider(),
        max_seconds=6,
        poll=1,
        progress_timeout_seconds=2,
    )

    assert result["status"] == "completed"
    assert result["post_marker_progress_timeout_observed"] is False
    assert result["teardown_reason"] == "runner_done_terminated_no_warm_reuse"


# ---------------------------------------------------------------------------
# Provider reliability manifest integration (paid path, hermetic fakes).
# ---------------------------------------------------------------------------


class _FakePaidProvider:
    name = "runpod"

    def __init__(
        self,
        *,
        available: bool = True,
        launch_status: str = "launched",
        availability_payload: dict | None = None,
    ) -> None:
        self._available = available
        self._launch_status = launch_status
        self._availability_payload = availability_payload
        self.terminated: list[str] = []
        self.launch_calls = 0

    def available(self) -> dict:
        if self._availability_payload is not None:
            return dict(self._availability_payload)
        if self._available:
            return {"available": True}
        return {"available": False, "reason": "runpod_api_key_missing"}

    def build_request(self, spec, job_dir) -> dict:  # noqa: ANN001
        return {"imageName": "img:tag", "gpuTypeIds": ["NVIDIA L40S"]}

    def launch(self, job_dir, request, *, cold: bool = False) -> dict:  # noqa: ANN001
        self.launch_calls += 1
        if self._launch_status != "launched":
            return {"status": "blocked", "blockers": ["no_pod_started"]}
        return {"status": "launched", "instance_id": "pod-1", "mode": "cold_create"}

    def stop(self, instance_id: str) -> dict:
        return {"status": "stopped", "http": 200}

    def inspect(self, instance_id: str) -> dict:
        if instance_id in self.terminated:
            return {"status": "unavailable", "http": 404, "instance_id": instance_id}
        return {
            "status": "observed",
            "http": 200,
            "instance_id": instance_id,
            "desiredStatus": "RUNNING",
        }

    def terminate(self, instance_id: str) -> dict:
        self.terminated.append(instance_id)
        return {"status": "terminated", "http": 200}


def _patch_paid_job_pipeline(monkeypatch, tmp_path: Path, provider: _FakePaidProvider) -> None:
    monkeypatch.setattr(
        render_job, "ensure_particlefield_usd",
        lambda source, out_dir: {"status": "completed", "usdc": str(tmp_path / "s.usdc"),
                                 "standard_ply": str(tmp_path / "s.ply")},
    )
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"zip")
    monkeypatch.setattr(
        render_job, "build_render_bundle",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        render_job, "stage_bundle",
        lambda bundle_zip, job_dir, key_prefix="": {"status": "completed"},
    )
    monkeypatch.setattr(render_job, "build_render_launch_spec", lambda *a, **k: object())
    monkeypatch.setattr(render_job, "get_render_provider", lambda *a, **k: provider)


def _successful_watch_result() -> dict:
    return {
        "status": "completed",
        "runner_result": {"status": "completed"},
        "runner_result_source": "isaac_runtime_result.json",
        "last_bootstrap": {"phase": "runner_done"},
        "runner_console_tail": "",
        "teardown": {
            "status": "terminated",
            "http": 200,
            "verification": {
                "provider_status": "not_found",
                "api_confirmed": True,
                "http": 404,
            },
        },
        "teardown_reason": "runner_done_terminated_no_warm_reuse",
        "timed_out_without_runner_done": False,
        "post_marker_progress_timeout_observed": False,
        "post_marker_progress_timeout": {},
        "runner_done_observed": True,
        "runner_timeout_observed": False,
        "final_result_without_runner_done": False,
        "elapsed_seconds": 10.0,
    }


def test_launch_runpod_blocks_without_budget_before_provider_lookup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        render_job,
        "get_render_provider",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("provider lookup must not happen before budget guard")
        ),
    )

    result = render_job.launch_runpod(tmp_path, {"imageName": "img:tag"})

    assert result["status"] == "blocked"
    assert result["reason"] == "prelaunch_spend_guard_blocked"
    assert "isaac_particlefield_prelaunch_spend_guard_not_passed" in result["blockers"]
    assert "max_spend_usd_missing" in result["blockers"]
    assert result["prelaunch_spend_guard"]["can_launch"] is False
    assert result["prelaunch_spend_guard"]["budget_source"] == "missing"


def test_paid_run_missing_budget_fails_before_provider_launch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(
        render_job,
        "watch_and_collect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("watch must not run when prelaunch budget is missing")
        ),
    )

    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=tmp_path / "out",
        cameras=_CAMS,
        allow_paid=True,
    )

    assert manifest["status"] == "blocked"
    assert provider.launch_calls == 0
    assert "isaac_particlefield_prelaunch_spend_guard_not_passed" in manifest["blockers"]
    assert "max_spend_usd_missing" in manifest["blockers"]
    assert manifest["prelaunch_spend_guard"]["can_launch"] is False
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["failed_phase"] == "pre_spend_preflight"
    assert "max_spend_usd_missing" in reliability["failure_blockers"]


def test_paid_run_enables_stall_watchdog_by_default_and_writes_reliability_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)
    monkeypatch.delenv(render_job.POST_MARKER_NO_PROGRESS_TIMEOUT_ENV, raising=False)
    captured_kwargs: dict = {}

    def fake_watch(job_dir, out_dir, instance_id, **kwargs):  # noqa: ANN001
        captured_kwargs.update(kwargs)
        return _successful_watch_result()

    monkeypatch.setattr(render_job, "watch_and_collect", fake_watch)
    out_dir = tmp_path / "out"
    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=out_dir,
        cameras=_CAMS,
        allow_paid=True,
        max_spend_usd=10.0,
    )
    assert manifest["status"] == "completed"
    # Stall watchdog is on by default — a booted-but-silent pod cannot bill forever.
    assert captured_kwargs["progress_timeout_seconds"] == (
        render_job.DEFAULT_POST_MARKER_NO_PROGRESS_TIMEOUT_SECONDS
    )
    path = Path(manifest["provider_reliability_manifest"])
    assert path.name == render_job.RELIABILITY_MANIFEST_NAME
    reliability = json.loads(path.read_text())
    assert reliability["schema_version"] == "provider_reliability_manifest.v1"
    assert reliability["run_id"] == "pod-1"
    assert reliability["failed_phase"] is None
    assert reliability["teardown_proven"] is True
    assert reliability["open_billing_risk"] is False
    assert reliability["run_completed"] is True
    assert reliability["not_applicable_phases"] == ["artifact_quality", "task_evaluation"]


def test_paid_run_stall_watchdog_env_override(monkeypatch, tmp_path: Path) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)
    monkeypatch.setenv(render_job.POST_MARKER_NO_PROGRESS_TIMEOUT_ENV, "123")
    captured_kwargs: dict = {}

    def fake_watch(job_dir, out_dir, instance_id, **kwargs):  # noqa: ANN001
        captured_kwargs.update(kwargs)
        return _successful_watch_result()

    monkeypatch.setattr(render_job, "watch_and_collect", fake_watch)
    render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=tmp_path / "out",
        cameras=_CAMS,
        allow_paid=True,
        max_spend_usd=10.0,
    )
    assert captured_kwargs["progress_timeout_seconds"] == 123


def test_paid_run_capacity_unavailable_fails_before_spend_with_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider(available=False)
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    def fail_watch(*a, **k):  # noqa: ANN001
        raise AssertionError("watch_and_collect must not run when capacity is unavailable")

    monkeypatch.setattr(render_job, "watch_and_collect", fail_watch)
    out_dir = tmp_path / "out"
    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=out_dir,
        cameras=_CAMS,
        allow_paid=True,
        max_spend_usd=10.0,
    )
    assert manifest["status"] == "blocked"
    assert "runpod_api_key_missing" in manifest["blockers"]
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["failed_phase"] == "pre_spend_preflight"
    assert any(
        b.startswith("capacity_unavailable:") for b in reliability["failure_blockers"]
    )
    # No launch happened, so no phase past preflight was recorded.
    assert reliability["furthest_phase_reached"] == "pre_spend_preflight"
    assert provider.launch_calls == 0


def test_paid_run_strict_preflight_rejects_non_boolean_capacity_before_launch(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider(availability_payload={"available": "yes"})
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=tmp_path / "out",
        cameras=_CAMS,
        allow_paid=True,
        max_spend_usd=10.0,
    )

    assert manifest["status"] == "blocked"
    assert provider.launch_calls == 0
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["failed_phase"] == "pre_spend_preflight"
    assert any(
        blocker.startswith("capacity_unavailable:capacity_evidence_missing")
        for blocker in reliability["failure_blockers"]
    )


def test_paid_run_strict_preflight_rejects_unpinned_image_before_launch(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=tmp_path / "out",
        cameras=_CAMS,
        allow_paid=True,
        max_spend_usd=10.0,
        image="docker.io/example/isaac-worker:latest",
    )

    assert manifest["status"] == "blocked"
    assert provider.launch_calls == 0
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["failed_phase"] == "pre_spend_preflight"
    assert any(
        blocker.startswith("worker_image_contract_invalid:image_not_pinned")
        for blocker in reliability["failure_blockers"]
    )


def test_paid_run_strict_preflight_rejects_disabled_watchdog_before_launch(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=tmp_path / "out",
        cameras=_CAMS,
        allow_paid=True,
        max_spend_usd=10.0,
        post_marker_progress_timeout_seconds=0,
    )

    assert manifest["status"] == "blocked"
    assert provider.launch_calls == 0
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["failed_phase"] == "pre_spend_preflight"
    assert any(
        blocker == "runtime_contract_invalid:startup_timeout_seconds_not_positive"
        for blocker in reliability["failure_blockers"]
    )


def test_paid_run_post_marker_stall_recorded_in_runtime_phase(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    def stalled_watch(job_dir, out_dir, instance_id, **kwargs):  # noqa: ANN001
        result = _successful_watch_result()
        result.update({
            "status": "blocked",
            "runner_result": {},
            "runner_result_source": None,
            "last_bootstrap": {"phase": "container_bash_started"},
            "post_marker_progress_timeout_observed": True,
            "post_marker_progress_timeout": {
                "phase": "container_bash_started",
                "timeout_seconds": 900,
                "elapsed_since_phase_seconds": 901.0,
            },
            "teardown": {
                "status": "terminated",
                "http": 200,
                "verification": {
                    "provider_status": "not_found",
                    "api_confirmed": True,
                    "http": 404,
                },
            },
            "teardown_reason": "post_marker_progress_timeout_terminated",
            "runner_done_observed": False,
        })
        return result

    monkeypatch.setattr(render_job, "watch_and_collect", stalled_watch)
    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply",
        out_dir=tmp_path / "out",
        cameras=_CAMS,
        allow_paid=True,
        max_spend_usd=10.0,
    )
    assert manifest["status"] == "blocked"
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["failed_phase"] == "runtime_execution"
    assert any(
        b.startswith("post_marker_no_progress:") for b in reliability["failure_blockers"]
    )
    # Container startup succeeded — the stall is attributed to runtime, not boot.
    assert reliability["phases"]["container_startup"]["passed"] is True
    assert reliability["teardown_proven"] is True


def test_paid_run_keep_running_records_open_billing_risk(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    def kept_watch(job_dir, out_dir, instance_id, **kwargs):  # noqa: ANN001
        result = _successful_watch_result()
        result.update({
            "teardown": {"status": "skipped", "note": "pod_left_running_by_request"},
            "teardown_reason": "left_running_by_request",
        })
        return result

    monkeypatch.setattr(render_job, "watch_and_collect", kept_watch)
    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply", out_dir=tmp_path / "out", cameras=_CAMS,
        allow_paid=True, preserve_instance=True, max_spend_usd=10.0,
    )
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["teardown_proven"] is False
    assert reliability["open_billing_risk"] is True
    assert reliability["run_completed"] is False
    teardown = reliability["phase_contracts"]["teardown"]
    assert teardown["keep_alive_requested"] is True


def test_paid_run_self_reported_teardown_is_open_billing_risk(
    monkeypatch, tmp_path: Path
) -> None:
    # A terminate DELETE that was never API-verified terminal must not prove teardown.
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    def unverified_watch(job_dir, out_dir, instance_id, **kwargs):  # noqa: ANN001
        result = _successful_watch_result()
        result["teardown"] = {"status": "terminated", "http": 200}
        return result

    monkeypatch.setattr(render_job, "watch_and_collect", unverified_watch)
    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply", out_dir=tmp_path / "out", cameras=_CAMS,
        allow_paid=True, max_spend_usd=10.0,
    )
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["teardown_proven"] is False
    assert reliability["open_billing_risk"] is True
    assert any(
        "terminal_status_not_api_confirmed" in b
        for b in reliability["phase_contracts"]["teardown"]["blockers"]
    )
    # The pending-teardown record stays open: an unproven teardown cannot close it.
    records = paid_lane_guard.load_pending_teardowns()
    assert len(records) == 1
    assert records[0]["instance_id"] == "pod-1"


def test_paid_run_preflight_routes_through_shared_chokepoint(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(render_job, "watch_and_collect", lambda *a, **k: _successful_watch_result())
    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply", out_dir=tmp_path / "out", cameras=_CAMS,
        allow_paid=True, max_spend_usd=10.0,
    )
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    preflight = reliability["phase_contracts"]["pre_spend_preflight"]
    assert preflight["schema_version"] == "pre_spend_preflight.v1"
    assert preflight["lane"] == "isaac_particlefield_render"


def test_paid_run_opens_pending_teardown_and_closes_on_proven_teardown(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)
    monkeypatch.setattr(render_job, "watch_and_collect", lambda *a, **k: _successful_watch_result())
    render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply", out_dir=tmp_path / "out", cameras=_CAMS,
        allow_paid=True, max_spend_usd=10.0,
    )
    assert paid_lane_guard.load_pending_teardowns() == []
    all_records = paid_lane_guard.load_pending_teardowns(include_closed=True)
    assert len(all_records) == 1
    record = all_records[0]
    assert record["status"] == "closed"
    assert record["provider"] == "runpod"
    assert record["lane"] == "isaac_particlefield_render"
    assert record["instance_id"] == "pod-1"
    assert record["teardown_proof"]["status"] == "PASS"


def test_crash_after_launch_leaves_record_that_reap_orphans_cleans(
    monkeypatch, tmp_path: Path
) -> None:
    provider = _FakePaidProvider()
    _patch_paid_job_pipeline(monkeypatch, tmp_path, provider)

    def crashing_watch(*_args, **_kwargs):
        raise RuntimeError("simulated crash between launch and collect")

    monkeypatch.setattr(render_job, "watch_and_collect", crashing_watch)
    with pytest.raises(RuntimeError):
        render_job.run_isaac_particlefield_render_job(
            source=tmp_path / "s.ply", out_dir=tmp_path / "out", cameras=_CAMS,
            allow_paid=True, max_spend_usd=10.0,
        )
    records = paid_lane_guard.load_pending_teardowns()
    assert len(records) == 1
    assert records[0]["status"] == "open"
    assert records[0]["instance_id"] == "pod-1"

    report = paid_lane_guard.reap_orphans(
        provider_clients={"runpod": provider},
        max_age_override_seconds=0,
    )
    assert report["reaped_count"] == 1
    assert report["open_billing_risk_count"] == 0
    entry = report["records"][0]
    assert entry["teardown_proof"]["status"] == "PASS"
    assert provider.terminated == ["pod-1"]
    assert paid_lane_guard.load_pending_teardowns() == []


def test_watch_and_collect_verifies_terminate_via_provider_api(
    tmp_path: Path, monkeypatch
) -> None:
    provider = _CollectProvider()
    (tmp_path / "provider_output_get_url.txt").write_text("https://example.test/out.zip")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    payload = _zip_bytes({"bootstrap.json": json.dumps({"phase": "runner_timeout"})})

    class _Response:
        def read(self) -> bytes:
            return payload

    monkeypatch.setattr(render_job.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(render_job.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response())
    result = render_job.watch_and_collect(
        tmp_path, out_dir, "pod-9", provider=provider, max_seconds=1, poll=0,
    )
    verification = result["teardown"].get("verification")
    assert verification is not None
    assert verification["api_confirmed"] is True
    assert verification["provider_status"] == "not_found"


def test_main_forwards_paid_budget(monkeypatch, tmp_path: Path, capsys) -> None:
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared", "prelaunch_spend_guard": {"can_launch": False}}

    monkeypatch.setattr(render_job, "run_isaac_particlefield_render_job", fake_run)

    exit_code = render_job.main(
        [
            "--source",
            str(tmp_path / "scene.ply"),
            "--out-dir",
            str(tmp_path / "out"),
            "--allow-paid",
            "--max-spend-usd",
            "4.5",
        ]
    )

    assert exit_code == 0
    assert captured["allow_paid"] is True
    assert captured["max_spend_usd"] == 4.5
    assert "prelaunch_spend_guard" in capsys.readouterr().out
