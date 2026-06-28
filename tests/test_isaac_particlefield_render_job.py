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
