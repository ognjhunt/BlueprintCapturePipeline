"""Hermetic tests for the Isaac G1 kitchen MuJoCo-parity job (no GPU spend, no network)."""
from __future__ import annotations

import json
import urllib.error
import zipfile
from pathlib import Path

from blueprint_pipeline import isaac_g1_kitchen_parity_job as J

_SCENARIOS = [
    {"scenario_id": "entry_to_sink", "spawn_position_xyz": [-4.25, -3.35, 0.05],
     "target_position_xyz": [1.75, 1.25, 0.05], "description": "Navigate to the sink work area."},
    {"scenario_id": "narrow_passage_to_sink", "spawn_position_xyz": [-3.0, 2.0, 0.05],
     "target_position_xyz": [1.6, 1.0, 0.05]},
]


def test_build_request_shapes_worker_paths() -> None:
    req = J.build_request(scenarios=_SCENARIOS, policy_id="blueprint_default_walk_to_target_smoke_policy", steps=48)
    assert req["kitchen_usd"] == "/workspace/bundle/kitchen/Collected_KitchenRoom/KitchenRoom.usd"
    assert req["g1_usd"] == "Isaac/Robots/Unitree/G1/g1.usd"  # relative -> resolved on worker
    assert req["steps"] == 48
    assert [s["scenario_id"] for s in req["scenarios"]] == ["entry_to_sink", "narrow_passage_to_sink"]


def test_cli_forwards_reused_kitchen_url(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--kitchen-url", "https://objects.example/kitchen.zip?sig=1",
    ])

    assert rc == 0
    assert captured["kitchen_asset_dir"] is None
    assert captured["kitchen_url"] == "https://objects.example/kitchen.zip?sig=1"


def test_cli_forwards_warm_candidates_without_source_hardcoding(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--warm-candidate", "pod-a",
        "--warm-candidate", "pod-b",
    ])

    assert rc == 0
    assert captured["warm_candidates"] == ("pod-a", "pod-b")
    assert captured["warm_only"] is False


def test_cli_forwards_warm_only(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--warm-candidate", "pod-a",
        "--warm-only",
        "--container-disk-gb", "240",
        "--volume-gb", "120",
    ])

    assert rc == 0
    assert captured["warm_candidates"] == ("pod-a",)
    assert captured["warm_only"] is True
    assert captured["container_disk_gb"] == 240
    assert captured["volume_gb"] == 120


def test_cli_forwards_provider_race_list(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--provider", "runpod,vast",
    ])

    assert rc == 0
    assert captured["provider"] == "runpod,vast"


def test_cli_forwards_vast_max_hourly_rate(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--provider", "vast",
        "--vast-max-hourly-rate", "4.75",
    ])

    assert rc == 0
    assert captured["provider"] == "vast"
    assert captured["vast_max_hourly_rate_usd"] == 4.75


def test_build_parity_bundle_contains_runner_policy_request_and_assets(tmp_path: Path) -> None:
    # fake kitchen asset tree
    kdir = tmp_path / "kitchen_src"
    (kdir / "Collected_KitchenRoom").mkdir(parents=True)
    (kdir / "Collected_KitchenRoom" / "KitchenRoom.usd").write_text("#usda fake")
    (kdir / "Collected_KitchenRoom" / "Sink054").mkdir()
    (kdir / "Collected_KitchenRoom" / "Sink054" / "Sink054.usd").write_text("#usda sink")
    zip_path = J.build_parity_bundle(scenarios=_SCENARIOS, out_dir=tmp_path / "job",
                                     kitchen_asset_dir=kdir, steps=32)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        req = json.loads(zf.read("request.json"))
        manifest = json.loads(zf.read("bundle_manifest.json"))
    assert "run_isaac_g1_kitchen_parity_eval.py" in names
    assert "isaac_g1_policy.py" in names  # policy module shipped for the worker import
    assert "render_visual_qc.py" in names  # Gemini placement QC module shipped for worker import
    assert "request.json" in names
    assert "kitchen/Collected_KitchenRoom/KitchenRoom.usd" in names
    assert "kitchen/Collected_KitchenRoom/Sink054/Sink054.usd" in names
    # scene_placement package shipped so the worker's dynamic task->object resolution works
    # (without it the runner has no blueprint_pipeline on its path and falls back to the literal
    # scenario target — the exact gap that broke the first dynamic render).
    assert "scene_placement/__init__.py" in names
    assert "scene_placement/usd_index.py" in names
    assert "scene_placement/target_resolver.py" in names
    assert "bundle_manifest.json" in names
    for required in J.PARITY_BUNDLE_REQUIRED_FILES:
        assert required in names
        assert required in manifest["required_files"]
    assert not any(n.endswith(".pyc") or "__pycache__" in n for n in names if n.startswith("scene_placement/"))
    assert req["steps"] == 32 and len(req["scenarios"]) == 2


def test_marker_timeout_default_covers_large_image_pull() -> None:
    import inspect
    sig = inspect.signature(J.run_isaac_g1_kitchen_parity_job)
    # The worker image is ~10.7 GB; 420s reaped slow nodes mid-pull before they could boot.
    # The boot window must comfortably exceed the pull time on a slow (~150 Mbps) node.
    assert sig.parameters["marker_timeout"].default >= 900
    assert sig.parameters["max_attempts"].default >= 2


def test_docker_start_cmd_runs_parity_runner() -> None:
    dsc = J.docker_start_cmd()
    assert dsc[0] == "-lc"
    body = dsc[1]
    assert "container_bash_started" in body  # early marker
    assert "run_isaac_g1_kitchen_parity_eval.py" in body
    assert "google-genai" in body
    assert "--request" in body
    assert "/isaac-sim/python.sh /workspace/boot.py" in body
    assert "pathlib.Path(OUT).iterdir()" in body
    assert "shutil.rmtree(p)" in body
    assert 'mark("runner_done", rc=rc)' in body
    assert "while True:" in body and "putout()" in body


def test_build_launch_spec_carries_policy_and_signed_urls(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    spec = J.build_launch_spec(jd, image="img:tag", policy_id="groot_sonic", steps=80)
    assert spec.image == "img:tag"
    assert spec.env["PARITY_POLICY"] == "groot_sonic"
    assert spec.env["PARITY_STEPS"] == "80"
    assert spec.env["BLUEPRINT_EVAL_MANIFEST_URI"].endswith("sig=A")
    assert spec.env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"].endswith("sig=B")
    assert spec.bootstrap_argv[0] == "-lc"
    assert spec.container_disk_gb >= 120
    assert spec.max_hourly_rate_usd == 5.0


def test_build_launch_spec_allows_vast_rate_override(monkeypatch, tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")

    direct = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        vast_max_hourly_rate_usd=4.25,
    )
    assert direct.max_hourly_rate_usd == 4.25

    monkeypatch.setenv("BLUEPRINT_ISAAC_G1_PARITY_VAST_MAX_HOURLY_RATE", "3.75")
    from_env = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert from_env.max_hourly_rate_usd == 3.75


def test_build_launch_spec_threads_gemini_key_only_when_supplied(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    off = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert "GOOGLE_GENAI_API_KEY" not in off.env
    assert "GEMINI_API_KEY" not in off.env
    on = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        gemini_api_key="secret-value",
    )
    assert on.env["GOOGLE_GENAI_API_KEY"] == "secret-value"
    assert on.env["GEMINI_API_KEY"] == "secret-value"


def test_build_launch_spec_threads_nonsecret_render_quality_env(monkeypatch, tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    monkeypatch.setenv("PARITY_RENDER_QUALITY_MODE", "pathtraced")
    monkeypatch.setenv("PARITY_PATH_TRACING_SAMPLES_PER_PIXEL", "128")
    monkeypatch.setenv("PARITY_PATH_TRACED_RT_SUBFRAMES", "2")

    spec = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)

    assert spec.env["PARITY_RENDER_QUALITY_MODE"] == "pathtraced"
    assert spec.env["PARITY_PATH_TRACING_SAMPLES_PER_PIXEL"] == "128"
    assert spec.env["PARITY_PATH_TRACED_RT_SUBFRAMES"] == "2"


def test_manipulation_cam_flag_threads_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    # off by default -> no env, no flag baked into the runner cmd
    off = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert "PARITY_MANIPULATION_CAM" not in off.env
    # on -> env set; the bootstrap only appends --manipulation-cam when that env == "1"
    on = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8, manipulation_cam=True)
    assert on.env["PARITY_MANIPULATION_CAM"] == "1"
    body = J.docker_start_cmd()[1]
    assert 'PARITY_MANIPULATION_CAM' in body and '--manipulation-cam' in body

    reach = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        manipulation_reach=True,
    )
    assert reach.env["PARITY_MANIPULATION_REACH"] == "1"
    assert reach.env["PARITY_MANIPULATION_REACH_ARM"] == "both"
    assert 'PARITY_MANIPULATION_REACH_ARM' in body and '--manipulation-reach-arm' in body


def test_dynamic_standing_contact_threads_physics_articulation_env_and_bootstrap(
    tmp_path: Path,
) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    spec = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        dynamic_standing_contact_steps=24,
    )
    assert spec.env["PARITY_ARTICULATED"] == "1"
    assert spec.env["PARITY_PHYSICS_ARTICULATION_DRIVE"] == "1"
    assert spec.env["PARITY_DYNAMIC_STANDING_CONTACT_STEPS"] == "24"
    body = J.docker_start_cmd()[1]
    assert "PARITY_PHYSICS_ARTICULATION_DRIVE" in body
    assert "--physics-articulation-drive" in body
    assert "PARITY_DYNAMIC_STANDING_CONTACT_STEPS" in body
    assert "--dynamic-standing-contact-steps" in body


def test_neutral_environment_flag_threads_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    off = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert "PARITY_NEUTRAL_ENVIRONMENT" not in off.env
    on = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8, neutral_environment=True)
    assert on.env["PARITY_NEUTRAL_ENVIRONMENT"] == "1"
    body = J.docker_start_cmd()[1]
    assert "PARITY_NEUTRAL_ENVIRONMENT" in body and "--neutral-environment" in body


def test_robot_review_material_override_threads_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    off = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert "PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE" not in off.env
    on = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        robot_review_material_override=True,
    )
    assert on.env["PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE"] == "1"
    body = J.docker_start_cmd()[1]
    assert "PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE" in body
    assert "--robot-review-material-override" in body


def test_collision_approximation_and_verify_cam_thread_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    off = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert "PARITY_COLLISION_APPROXIMATION" not in off.env and "PARITY_VERIFY_CAM" not in off.env
    on = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8,
                             collision_approximation="convexHull", verify_cam=True)
    assert on.env["PARITY_COLLISION_APPROXIMATION"] == "convexHull"
    assert on.env["PARITY_VERIFY_CAM"] == "1"
    body = J.docker_start_cmd()[1]
    assert "PARITY_COLLISION_APPROXIMATION" in body and "--collision-approximation" in body
    assert "PARITY_VERIFY_CAM" in body and "--verify-cam" in body


def test_focus_prune_threads_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    spec = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        focus_radius=2.5,
        keep_objects="room,floor,wall,sink,counter,cabinet,light",
    )
    assert spec.env["PARITY_FOCUS_RADIUS"] == "2.5"
    assert spec.env["PARITY_KEEP_OBJECTS"] == "room,floor,wall,sink,counter,cabinet,light"
    body = J.docker_start_cmd()[1]
    assert "PARITY_FOCUS_RADIUS" in body and "--focus-radius" in body
    assert "PARITY_KEEP_OBJECTS" in body and "--keep-objects" in body


def test_manipulation_stand_flag_threads_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    off = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert "PARITY_MANIPULATION_STAND" not in off.env
    on = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8, manipulation_stand=True)
    assert on.env["PARITY_MANIPULATION_STAND"] == "1"
    body = J.docker_start_cmd()[1]
    assert "PARITY_MANIPULATION_STAND" in body and "--manipulation-stand" in body


def test_build_harness_package_is_wam_ready_and_honest(tmp_path: Path) -> None:
    result = {
        "policy_id": "blueprint_default_walk_to_target_smoke_policy",
        "scenarios_executed": 2, "scenarios_passed": 1,
        "scenarios": [{"scenario_id": "entry_to_sink", "task_success": True},
                      {"scenario_id": "narrow_passage_to_sink", "task_success": False}],
    }
    pkg = J.build_harness_package(result=result, render_out_dir=tmp_path / "render", out_dir=tmp_path / "out")
    assert pkg["wam_evaluator"]["evaluates"] == "video_rollout_fidelity_not_task_success"
    assert pkg["wam_evaluator"]["status"] == "inputs_ready_pending_model_run"
    assert len(pkg["wam_evaluator"]["inputs"]) == 2
    assert pkg["wam_evaluator"]["inputs"][0]["overview_mp4"].endswith("entry_to_sink/overview.mp4")
    assert "not task success" in pkg["claim_boundary"].lower()
    assert (tmp_path / "out" / "isaac_g1_kitchen_parity_harness.json").is_file()


def test_job_prepared_plan_without_spend(tmp_path: Path, monkeypatch) -> None:
    # mock staging so no network/creds are needed; assert the no-spend plan path
    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
        return {"status": "completed", "manifest": {}}

    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    m = J.run_isaac_g1_kitchen_parity_job(scenarios=_SCENARIOS, out_dir=tmp_path / "job",
                                          provider="vast", allow_paid=False,
                                          robot_review_material_override=True)
    assert m["status"] == "prepared"
    assert m["provider"] == "vast"
    assert m["launch_request_shape"]["provider"] == "vast"
    assert m["launch_request_shape"]["vast_max_hourly_rate_usd"] == 5.0
    assert m["launch_request_shape"]["container_disk_gb"] == 140
    assert m["launch_request_shape"]["volume_gb"] == 80
    assert m["launch_request_shape"]["robot_review_material_override"] is True
    assert m["scenario_ids"] == ["entry_to_sink", "narrow_passage_to_sink"]
    assert "git_evidence" in m


def test_job_prepared_multi_provider_plan_without_spend(tmp_path: Path, monkeypatch) -> None:
    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
        return {"status": "completed", "manifest": {}}

    monkeypatch.setattr(J, "stage_bundle", _fake_stage)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod,vast",
        allow_paid=False,
    )

    assert m["status"] == "prepared"
    assert m["provider"] == "runpod,vast"
    assert m["providers"] == ["runpod", "vast"]
    assert m["launch_request_shape"]["provider"] == "runpod"


def test_paid_multi_provider_uses_race_winner_for_collect(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text(
            f"https://spaces.example/{job_dir.name}/bundle.zip?sig=A"
        )
        (job_dir / "provider_output_put_url.txt").write_text(
            f"https://spaces.example/{job_dir.name}/out.zip?sig=B"
        )
        (job_dir / "provider_output_get_url.txt").write_text(
            f"https://spaces.example/{job_dir.name}/out.zip?sig=C"
        )
        return {"status": "completed", "manifest": {}}

    class _FakeProvider:
        def __init__(self, name: str) -> None:
            self.name = name
            self.requests: list[dict] = []

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            body = {"env": dict(spec.env), "provider": self.name, "job_dir": str(job_dir)}
            self.requests.append(body)
            return body

    fake_providers = {"runpod": _FakeProvider("runpod"), "vast": _FakeProvider("vast")}
    monkeypatch.setattr(
        J,
        "get_render_provider",
        lambda name, warm_candidates=(): fake_providers[name],
    )
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    captured: dict = {}

    def _fake_race(providers, request, marker_check, marker_timeout, *, job_dir, cold=False,
                   poll_interval=10.0, circuit_breaker=None, terminate_losers=True,
                   launch_kwargs=None, sleep=None, monotonic=None):
        bodies = []
        for i, provider_obj in enumerate(providers):
            contender_dir = Path(job_dir) / f"contender-{i}-{provider_obj.name}"
            body = request(provider_obj, contender_dir)
            bodies.append(body)
            assert body["env"]["BLUEPRINT_LAUNCH_SESSION_ID"]
            assert launch_kwargs(provider_obj) == {"allow_cold_fallback": True}
        captured["race_bodies"] = bodies
        winner_dir = Path(job_dir) / "contender-1-vast"
        return {
            "schema": "provider_race.v1",
            "status": "launched",
            "provider": "vast",
            "instance_id": "vast-iid",
            "mode": "vast_on_demand",
            "winner_provider": fake_providers["vast"],
            "winner_launch": {"status": "launched", "instance_id": "vast-iid",
                              "mode": "vast_on_demand", "job_dir": str(winner_dir)},
            "contenders": [],
            "skipped": [],
            "terminated_losers": 1,
            "reason": None,
        }

    def _fake_watch(job_dir, render_out, instance_id, *, provider=None, max_seconds=0,
                    preserve_instance=False):
        captured["collect_job_dir"] = Path(job_dir)
        captured["collect_provider"] = provider.name
        captured["collect_instance_id"] = instance_id
        return {
            "status": "completed",
            "elapsed_seconds": 1,
            "teardown": {"status": "preserved"},
            "runner_result": {
                "status": "completed",
                "policy_id": "blueprint_default_walk_to_target_smoke_policy",
                "scenarios": [],
                "scenarios_executed": 0,
                "scenarios_passed": 0,
            },
        }

    monkeypatch.setattr(J, "race_launch", _fake_race)
    monkeypatch.setattr(J, "watch_and_collect", _fake_watch)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod,vast",
        allow_paid=True,
        allow_dirty_paid_launch=True,
    )

    assert m["status"] == "completed"
    assert m["launch"]["provider"] == "vast"
    assert m["launch"]["terminated_losers"] == 1
    assert captured["collect_provider"] == "vast"
    assert captured["collect_instance_id"] == "vast-iid"
    assert captured["collect_job_dir"].name == "contender-1-vast"


def test_paid_job_surfaces_blocked_parity_result_without_runtime_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
        (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
        return {"status": "completed", "manifest": {}}

    class _FakeProvider:
        name = "runpod"

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"env": dict(spec.env), "image": spec.image}

    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): _FakeProvider())
    monkeypatch.setattr(
        J,
        "launch_with_marker_retry",
        lambda *_args, **_kwargs: {
            "status": "launched",
            "instance_id": "pod1",
            "mode": "cold_create_marker_verified",
        },
    )
    monkeypatch.setattr(
        J,
        "watch_and_collect",
        lambda *_args, **_kwargs: {
            "status": "blocked",
            "elapsed_seconds": 1,
            "teardown": {"status": "stopped"},
            "runner_result_source": "isaac_g1_kitchen_parity_result.json",
            "last_bootstrap": {"phase": "runner_done", "rc": 0},
            "timed_out_without_runner_done": False,
            "runner_result": {
                "status": "blocked",
                "blockers": [
                    "manipulation_pov_geometry_failed",
                    "placement_validation_failed",
                ],
                "scenarios": [],
                "scenarios_executed": 1,
                "scenarios_passed": 0,
            },
        },
    )

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
    )

    assert m["status"] == "blocked"
    assert m["runner_completed"] is True
    assert m["parity_result_status"] == "blocked"
    assert "manipulation_pov_geometry_failed" in m["blockers"]
    assert "placement_validation_failed" in m["blockers"]
    assert "isaac_parity_result_blocked" in m["blockers"]
    assert "isaac_runtime_did_not_complete" not in m["blockers"]
    assert "harness" not in m


def test_paid_launch_blocks_dirty_worktree_before_staging(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {
            "status": "available",
            "git_sha": "abc123",
            "dirty": True,
            "dirty_entries_count": 1,
            "dirty_entries": [" M scripts/run_isaac_g1_kitchen_parity_eval.py"],
            "dirty_entries_truncated": False,
        },
    )

    def _stage_should_not_run(*_args, **_kwargs):
        raise AssertionError("paid dirty-tree guard must run before staging")

    monkeypatch.setattr(J, "stage_bundle", _stage_should_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="vast",
        allow_paid=True,
    )

    assert m["status"] == "blocked"
    assert "dirty_worktree_paid_launch_blocked" in m["blockers"]
    assert m["git_evidence"]["dirty"] is True


def test_cli_forwards_dirty_paid_launch_override(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--allow-paid",
        "--allow-dirty-paid-launch",
    ])

    assert rc == 0
    assert captured["allow_paid"] is True
    assert captured["allow_dirty_paid_launch"] is True


def test_job_blocks_on_no_scenarios(tmp_path: Path) -> None:
    m = J.run_isaac_g1_kitchen_parity_job(scenarios=[], out_dir=tmp_path / "job", allow_paid=False)
    assert m["status"] == "blocked" and "no_scenarios" in m["blockers"]


def test_job_blocks_on_unknown_provider(tmp_path: Path) -> None:
    m = J.run_isaac_g1_kitchen_parity_job(scenarios=_SCENARIOS, out_dir=tmp_path / "job",
                                          provider="lambda", allow_paid=False)
    assert m["status"] == "blocked" and "unknown_render_provider" in m["blockers"]


def _make_fake_provider():
    import io as _io
    import zipfile as _zip

    class _FakeProv:
        def __init__(self, marker: bool, marker_mode: str = "matching") -> None:
            self.launched: list[str] = []
            self.terminated: list[str] = []
            self.stopped: list[str] = []
            self._marker = marker
            self._marker_mode = marker_mode
            self._launch_session_id = ""

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            iid = f"pod{len(self.launched)}"
            self.launched.append(iid)
            self._launch_session_id = str((request.get("env") or {}).get("BLUEPRINT_LAUNCH_SESSION_ID") or "")
            return {
                "status": "launched",
                "instance_id": iid,
                "mode": "cold_create" if cold else "warm_restart",
            }

        def terminate(self, iid):
            self.terminated.append(iid)
            return {"status": "terminated"}

        def stop(self, iid):
            self.stopped.append(iid)
            return {"status": "stopped"}

        def urlopen(self, url, timeout=60):
            buf = _io.BytesIO()
            with _zip.ZipFile(buf, "w") as z:
                if self._marker:
                    marker_payload = {}
                    if self._marker_mode == "matching":
                        marker_payload = {"launch_session_id": self._launch_session_id}
                    elif self._marker_mode == "wrong":
                        marker_payload = {"launch_session_id": "old-session"}
                    z.writestr(
                        "bootstrap.json",
                        json.dumps(marker_payload),
                    )

            class _R:
                def read(self_inner):
                    return buf.getvalue()
            return _R()

    return _FakeProv


def test_launch_with_marker_retry_keeps_pod_that_heartbeats(tmp_path: Path, monkeypatch) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")
    fp = _make_fake_provider()(marker=True)  # pod emits the early marker
    monkeypatch.setattr(J.time, "sleep", lambda s: None)
    monkeypatch.setattr(J.urllib.request, "urlopen", fp.urlopen)
    res = J.launch_with_marker_retry(fp, jd, {"img": "x"}, max_attempts=3, marker_timeout=5, poll=1)
    assert res["status"] == "launched"
    assert res["instance_id"] == "pod0"     # first pod heartbeats -> kept
    assert fp.terminated == []              # not terminated (it's the live render pod)
    assert res["mode"] == "cold_create_marker_verified"


def test_launch_with_marker_retry_terminates_all_flaky_pods(tmp_path: Path, monkeypatch) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")
    fp = _make_fake_provider()(marker=False)  # pods never emit the marker (flaky)
    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", fp.urlopen)
    res = J.launch_with_marker_retry(fp, jd, {"img": "x"}, max_attempts=3, marker_timeout=2, poll=1)
    assert res["status"] == "blocked"
    assert "all_launch_attempts_flaky" in res["blockers"]
    # every flaky pod was DELETED — none left billing
    assert fp.terminated == ["pod0", "pod1", "pod2"]


def test_launch_with_marker_retry_ignores_stale_bootstrap_marker(tmp_path: Path, monkeypatch) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")
    fp = _make_fake_provider()(marker=True, marker_mode="wrong")
    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", fp.urlopen)

    res = J.launch_with_marker_retry(
        fp,
        jd,
        {"env": {"EXISTING": "1"}},
        max_attempts=1,
        marker_timeout=2,
        poll=1,
    )

    assert res["status"] == "blocked"
    assert res["attempts"][0]["marker_seen"] is False
    assert fp.terminated == ["pod0"]


def test_launch_with_marker_retry_stops_flaky_warm_restart(tmp_path: Path, monkeypatch) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")
    fp = _make_fake_provider()(marker=False)
    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", fp.urlopen)

    res = J.launch_with_marker_retry(
        fp,
        jd,
        {"img": "x"},
        max_attempts=1,
        marker_timeout=2,
        poll=1,
        cold=False,
    )

    assert res["status"] == "blocked"
    assert fp.terminated == []
    assert fp.stopped == ["pod0"]


def test_launch_with_marker_retry_blocks_warm_only_without_marker_poll(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    class _WarmOnlyBlockedProvider:
        def __init__(self) -> None:
            self.launch_calls: list[tuple[bool, bool]] = []
            self.stopped: list[str] = []
            self.terminated: list[str] = []

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            self.launch_calls.append((cold, allow_cold_fallback))
            assert allow_cold_fallback is False
            return {
                "status": "blocked",
                "blockers": ["warm_restart_failed_cold_fallback_disabled"],
                "attempts": [{"pod_id": "stale-warm"}],
            }

        def stop(self, iid):
            self.stopped.append(iid)
            return {"status": "stopped"}

        def terminate(self, iid):
            self.terminated.append(iid)
            return {"status": "terminated"}

    fp = _WarmOnlyBlockedProvider()
    monkeypatch.setattr(
        J.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("marker polled")),
    )

    res = J.launch_with_marker_retry(
        fp,
        jd,
        {"img": "x"},
        max_attempts=3,
        marker_timeout=2,
        poll=1,
        cold=False,
        allow_cold_fallback=False,
    )

    assert res["status"] == "blocked"
    assert res["blockers"] == ["warm_restart_failed_cold_fallback_disabled"]
    assert res["attempts"][0]["result"] == "launch_call_failed"
    assert fp.launch_calls == [(False, False)]
    assert fp.stopped == []
    assert fp.terminated == []


def test_job_warm_only_blocks_without_cold_spend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
        (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
        return {"status": "completed", "manifest": {}}

    class _WarmOnlyProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.launch_calls: list[tuple[bool, bool]] = []

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"env": dict(spec.env), "image": spec.image}

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            self.launch_calls.append((cold, allow_cold_fallback))
            return {
                "status": "blocked",
                "blockers": ["warm_restart_failed_cold_fallback_disabled"],
            }

    provider = _WarmOnlyProvider()
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): provider)
    monkeypatch.setattr(
        J,
        "watch_and_collect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("collected")),
    )

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        warm_candidates=("stale-warm",),
        warm_only=True,
    )

    assert m["status"] == "blocked"
    assert "warm_restart_failed_cold_fallback_disabled" in m["blockers"]
    assert "launch_failed_all_attempts_flaky" in m["blockers"]
    assert provider.launch_calls == [(False, False)]


def test_await_warm_serve_ready_requires_matching_launch_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import io as _io
    import zipfile as _zip

    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")
    (jd / "launch_session_nonce.txt").write_text("fresh-session", encoding="utf-8")
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as z:
        z.writestr("bootstrap.json", json.dumps({
            "phase": "runner_starting",
            "launch_session_id": "fresh-session",
        }))
        z.writestr("warm_serve_ready.json", json.dumps({
            "status": "serving",
            "launch_session_id": "old-session",
        }))
    data = buf.getvalue()

    class _R:
        def read(self) -> bytes:
            return data

    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", lambda _url, timeout=60: _R())

    res = J._await_warm_serve_ready(jd, instance_id="pod1", timeout_s=2, poll_interval_s=1)

    assert res["ready"] is False
    assert res["reason"] == "serve_ready_timeout"


def test_await_warm_serve_ready_accepts_matching_launch_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import io as _io
    import zipfile as _zip

    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")
    (jd / "launch_session_nonce.txt").write_text("fresh-session", encoding="utf-8")
    buf = _io.BytesIO()
    with _zip.ZipFile(buf, "w") as z:
        z.writestr("bootstrap.json", json.dumps({
            "phase": "runner_starting",
            "launch_session_id": "fresh-session",
        }))
        z.writestr("warm_serve_ready.json", json.dumps({
            "status": "serving",
            "launch_session_id": "fresh-session",
        }))
    data = buf.getvalue()

    class _R:
        def read(self) -> bytes:
            return data

    monkeypatch.setattr(J.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(J.time, "sleep", lambda _s: None)
    monkeypatch.setattr(J.urllib.request, "urlopen", lambda _url, timeout=60: _R())

    res = J._await_warm_serve_ready(jd, instance_id="pod1", timeout_s=2, poll_interval_s=1)

    assert res["ready"] is True
    assert res["serve_detail"]["launch_session_id"] == "fresh-session"


def test_await_warm_serve_ready_surfaces_expired_output_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    monkeypatch.setattr(J.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        J.urllib.request,
        "urlopen",
        lambda url, timeout=60: (_ for _ in ()).throw(
            urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        ),
    )

    res = J._await_warm_serve_ready(jd, instance_id="pod1", timeout_s=2, poll_interval_s=1)

    assert res["ready"] is False
    assert res["reason"] == "presigned_url_expired_or_forbidden"
    assert res["http_status"] == 403
