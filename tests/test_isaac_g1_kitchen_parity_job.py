"""Hermetic tests for the Isaac G1 kitchen MuJoCo-parity job (no GPU spend, no network)."""
from __future__ import annotations

import json
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
    assert "run_isaac_g1_kitchen_parity_eval.py" in names
    assert "isaac_g1_policy.py" in names  # policy module shipped for the worker import
    assert "request.json" in names
    assert "kitchen/Collected_KitchenRoom/KitchenRoom.usd" in names
    assert "kitchen/Collected_KitchenRoom/Sink054/Sink054.usd" in names
    assert req["steps"] == 32 and len(req["scenarios"]) == 2


def test_docker_start_cmd_runs_parity_runner() -> None:
    dsc = J.docker_start_cmd()
    assert dsc[0] == "-lc"
    body = dsc[1]
    assert "container_bash_started" in body  # early marker
    assert "run_isaac_g1_kitchen_parity_eval.py" in body
    assert "--request" in body
    assert "/isaac-sim/python.sh /workspace/boot.py" in body
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
                                          provider="vast", allow_paid=False)
    assert m["status"] == "prepared"
    assert m["provider"] == "vast"
    assert m["launch_request_shape"]["provider"] == "vast"
    assert m["scenario_ids"] == ["entry_to_sink", "narrow_passage_to_sink"]


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
        def __init__(self, marker: bool) -> None:
            self.launched: list[str] = []
            self.terminated: list[str] = []
            self._marker = marker

        def launch(self, job_dir, request, *, cold=False):
            iid = f"pod{len(self.launched)}"
            self.launched.append(iid)
            return {"status": "launched", "instance_id": iid}

        def terminate(self, iid):
            self.terminated.append(iid)
            return {"status": "terminated"}

        def urlopen(self, url, timeout=60):
            buf = _io.BytesIO()
            with _zip.ZipFile(buf, "w") as z:
                if self._marker:
                    z.writestr("bootstrap.json", "{}")

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
    assert "all_cold_launch_attempts_flaky" in res["blockers"]
    # every flaky pod was DELETED — none left billing
    assert fp.terminated == ["pod0", "pod1", "pod2"]
