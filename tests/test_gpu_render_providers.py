"""Hermetic tests for the provider-agnostic GPU render launch layer (no GPU spend, no net).

Covers: the neutral RenderLaunchSpec, the registry, per-provider request translation
(RunPod pod body vs Vast offer-search/create-instance), credential availability, the
fail-closed no-spend guards, and provider-parameterized teardown.
"""
from __future__ import annotations

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
    assert fake.terminated == "inst-9"  # pod is DELETED, not merely stopped
    assert res["status"] == "blocked"  # nothing rendered
    assert res["teardown"]["status"] == "terminated"


def test_runpod_terminate_is_delete_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("blueprint_pipeline.gpu_render_providers.SECRETS", tmp_path)
    res = RunPodRenderProvider().terminate("podabc")
    assert res["status"] == "blocked"  # no key -> no network, fail closed
    assert "runpod_api_key_missing" in res["blockers"]
    # terminate is distinct from stop (DELETE vs POST /stop) — both exist on the provider
    assert hasattr(RunPodRenderProvider(), "terminate") and hasattr(RunPodRenderProvider(), "stop")
