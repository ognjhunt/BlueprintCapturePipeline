"""Hermetic tests for the productionized Isaac ParticleField render job (no GPU spend).

The deterministic build/plan steps are pinned here; staging + launch + render are the
network/GPU integration steps gated behind allow_paid.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import blueprint_pipeline.isaac_particlefield_render_job as render_job
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
    assert 'mark("runner_done", rc=rc)' in body
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


# ---------------------------------------------------------------------------
# Provider reliability manifest integration (paid path, hermetic fakes).
# ---------------------------------------------------------------------------


class _FakePaidProvider:
    name = "runpod"

    def __init__(self, *, available: bool = True, launch_status: str = "launched") -> None:
        self._available = available
        self._launch_status = launch_status
        self.terminated: list[str] = []

    def available(self) -> dict:
        if self._available:
            return {"available": True}
        return {"available": False, "reason": "runpod_api_key_missing"}

    def build_request(self, spec, job_dir) -> dict:  # noqa: ANN001
        return {"imageName": "img:tag", "gpuTypeIds": ["NVIDIA L40S"]}

    def launch(self, job_dir, request, *, cold: bool = False) -> dict:  # noqa: ANN001
        if self._launch_status != "launched":
            return {"status": "blocked", "blockers": ["no_pod_started"]}
        return {"status": "launched", "instance_id": "pod-1", "mode": "cold_create"}

    def stop(self, instance_id: str) -> dict:
        return {"status": "stopped", "http": 200}

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
        "teardown": {"status": "terminated", "http": 200},
        "teardown_reason": "runner_done_terminated_no_warm_reuse",
        "timed_out_without_runner_done": False,
        "post_marker_progress_timeout_observed": False,
        "post_marker_progress_timeout": {},
        "runner_done_observed": True,
        "runner_timeout_observed": False,
        "final_result_without_runner_done": False,
        "elapsed_seconds": 10.0,
    }


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
        source=tmp_path / "s.ply", out_dir=out_dir, cameras=_CAMS, allow_paid=True,
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
        source=tmp_path / "s.ply", out_dir=tmp_path / "out", cameras=_CAMS, allow_paid=True,
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
        source=tmp_path / "s.ply", out_dir=out_dir, cameras=_CAMS, allow_paid=True,
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
            "teardown": {"status": "terminated", "http": 200},
            "teardown_reason": "post_marker_progress_timeout_terminated",
            "runner_done_observed": False,
        })
        return result

    monkeypatch.setattr(render_job, "watch_and_collect", stalled_watch)
    manifest = render_job.run_isaac_particlefield_render_job(
        source=tmp_path / "s.ply", out_dir=tmp_path / "out", cameras=_CAMS, allow_paid=True,
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
        allow_paid=True, preserve_instance=True,
    )
    reliability = json.loads(Path(manifest["provider_reliability_manifest"]).read_text())
    assert reliability["teardown_proven"] is False
    assert reliability["open_billing_risk"] is True
    assert reliability["run_completed"] is False
    teardown = reliability["phase_contracts"]["teardown"]
    assert teardown["keep_alive_requested"] is True
