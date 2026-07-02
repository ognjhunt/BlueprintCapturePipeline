"""Hermetic tests for the render-noise-audit wiring in the Isaac G1 parity job (no GPU/network)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline import isaac_g1_kitchen_parity_job as J
from blueprint_pipeline.g1_render_noise_audit import build_variant_plan

_AUDIT_SCENARIO = [{
    "scenario_id": "render_noise_audit",
    "instruction": "open the fridge door",
    "task": "open the fridge door",
}]


def test_build_request_carries_render_noise_audit_plan() -> None:
    plan = build_variant_plan()
    req = J.build_request(
        scenarios=_AUDIT_SCENARIO, policy_id="blueprint_default_walk_to_target_smoke_policy",
        steps=1, render_noise_audit_plan=plan,
    )
    assert [v["variant_id"] for v in req["render_noise_audit"]["variants"]] == list("ABCDEFG")
    assert req["render_noise_audit"]["execution_order"][-1] == "A"

    plain = J.build_request(
        scenarios=_AUDIT_SCENARIO, policy_id="blueprint_default_walk_to_target_smoke_policy", steps=1,
    )
    assert "render_noise_audit" not in plain


def test_bundle_ships_audit_module_and_plan(tmp_path: Path) -> None:
    zip_path = J.build_parity_bundle(
        scenarios=_AUDIT_SCENARIO, out_dir=tmp_path,
        render_noise_audit_plan=build_variant_plan(),
    )
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "g1_render_noise_audit.py" in names
        request = json.loads(zf.read("request.json").decode())
    assert "render_noise_audit" in request
    assert "g1_render_noise_audit.py" in J.PARITY_BUNDLE_REQUIRED_FILES


def test_launch_spec_env_carries_audit_flags(tmp_path: Path) -> None:
    (tmp_path / "provider_bundle_url.txt").write_text("https://objects.example/bundle.zip?sig=1")
    (tmp_path / "provider_output_put_url.txt").write_text("https://objects.example/out.zip?sig=2")
    spec = J.build_launch_spec(
        tmp_path, image="example/isaac-worker:1", policy_id="p", steps=1,
        render_noise_audit=True, audit_high_spp=256, audit_warmup_frames=5,
        audit_boost_light_intensity=5000.0,
    )
    assert spec.env["PARITY_RENDER_NOISE_AUDIT"] == "1"
    assert spec.env["PARITY_AUDIT_HIGH_SPP"] == "256"
    assert spec.env["PARITY_AUDIT_WARMUP_FRAMES"] == "5"
    assert spec.env["PARITY_AUDIT_BOOST_LIGHT_INTENSITY"] == "5000.0"

    plain = J.build_launch_spec(tmp_path, image="example/isaac-worker:1", policy_id="p", steps=1)
    assert "PARITY_RENDER_NOISE_AUDIT" not in plain.env


def test_bootstrap_forwards_audit_env_to_runner_flags() -> None:
    assert '--render-noise-audit' in J.BOOTSTRAP
    assert 'PARITY_RENDER_NOISE_AUDIT' in J.BOOTSTRAP
    assert 'PARITY_AUDIT_HIGH_SPP' in J.BOOTSTRAP
    assert '--audit-high-spp' in J.BOOTSTRAP
    assert 'PARITY_AUDIT_WARMUP_FRAMES' in J.BOOTSTRAP
    assert 'PARITY_AUDIT_BOOST_LIGHT_INTENSITY' in J.BOOTSTRAP


def test_job_cli_forwards_audit_flags(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_AUDIT_SCENARIO), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--render-noise-audit",
        "--audit-high-spp", "256",
    ])
    assert rc == 0
    assert captured["render_noise_audit"] is True
    assert captured["audit_high_spp"] == 256


def test_audit_cli_launch_forwards_task_scenario(monkeypatch, tmp_path: Path) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_g1_render_noise_audit.py"
    spec = importlib.util.spec_from_file_location("run_g1_render_noise_audit_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared", "blockers": []}

    monkeypatch.setattr(cli, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = cli.main([
        "launch",
        "--task", "open the fridge door",
        "--out-dir", str(tmp_path / "out"),
        "--kitchen-url", "https://objects.example/kitchen.zip?sig=1",
        "--warm-candidate", "qzafypkkad8dcm",
    ])
    assert rc == 0
    assert captured["render_noise_audit"] is True
    assert captured["warm_candidates"] == ("qzafypkkad8dcm",)
    scenario = captured["scenarios"][0]
    assert scenario["task"] == "open the fridge door"
    assert scenario["instruction"] == "open the fridge door"
    assert "spawn_position_xyz" not in scenario  # dynamic resolution, no hardcoded coordinates
    manifest_path = tmp_path / "out" / "g1_render_noise_audit_job_manifest.json"
    assert manifest_path.is_file()
