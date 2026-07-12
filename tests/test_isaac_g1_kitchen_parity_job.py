"""Hermetic tests for the Isaac G1 kitchen MuJoCo-parity job (no GPU spend, no network)."""
from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path

from blueprint_pipeline import isaac_g1_kitchen_parity_job as J
from blueprint_pipeline import isaac_review_media as review_media
from blueprint_pipeline import paid_lane_guard

import pytest

pytestmark = pytest.mark.slow

_SCENARIOS = [
    {"scenario_id": "entry_to_sink", "spawn_position_xyz": [-4.25, -3.35, 0.05],
     "target_position_xyz": [1.75, 1.25, 0.05], "description": "Navigate to the sink work area."},
    {"scenario_id": "narrow_passage_to_sink", "spawn_position_xyz": [-3.0, 2.0, 0.05],
     "target_position_xyz": [1.6, 1.0, 0.05]},
]


def _set_test_worker_image(monkeypatch) -> None:
    monkeypatch.setenv(
        J.ISAAC_WORKER_IMAGE_REF_ENV,
        "registry.example/blueprint/isaac-eval-worker:test",
    )
    monkeypatch.setenv(J.ISAAC_G1_MAX_SPEND_USD_ENV, "10.0")


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
    assert captured["provider"] == J.DEFAULT_ISAAC_REVIEW_PROVIDER


def test_default_provider_is_digitalocean_for_isaac_review_lane_only() -> None:
    assert J.DEFAULT_ISAAC_REVIEW_PROVIDER == "digitalocean"
    assert J._provider_names(None) == ["digitalocean"]
    assert J._provider_names("") == ["digitalocean"]
    assert J._provider_names("runpod") == ["runpod"]
    assert J._provider_names("runpod,vast") == ["runpod", "vast"]


def test_image_manifest_floor_prevents_premature_large_pull_teardown() -> None:
    policy = J._effective_startup_no_runtime_timeout(
        600,
        {"recommended_startup_no_runtime_timeout_seconds": 1446},
    )
    assert policy["effective_seconds"] == 1446
    assert policy["raised_to_image_manifest_floor"] is True

    disabled = J._effective_startup_no_runtime_timeout(
        0,
        {"recommended_startup_no_runtime_timeout_seconds": 1446},
    )
    assert disabled["effective_seconds"] == 0
    assert disabled["disabled"] is True


def test_prelaunch_spend_guard_budgets_all_sequential_startup_attempts() -> None:
    guard = J._isaac_g1_prelaunch_spend_guard(
        allow_paid=True,
        provider_name="runpod",
        max_spend_usd=2.0,
        max_seconds=3600,
        max_hourly_rate_usd=1.0,
        contender_count=1,
        marker_timeout_seconds=1500,
        startup_no_runtime_timeout_seconds=1200,
        max_attempts=3,
    )
    assert guard["startup_budget_seconds"] == 3600
    assert guard["render_budget_seconds"] == 3600
    assert guard["billable_budget_seconds"] == 7200
    assert guard["estimated_max_spend_usd"] == 2.0
    assert guard["can_launch"] is True


def test_capacity_preflight_rate_drives_spend_estimate_not_marketplace_ceiling() -> None:
    capacity = {
        "status": "available",
        "viable_size_regions": [
            {"size": "gpu-6000adax1-48gb", "price_hourly": 1.57, "matching_regions": ["tor1"]},
        ],
    }
    rate = J._capacity_preflight_hourly_rate(capacity)
    assert rate == 1.57

    guard = J._isaac_g1_prelaunch_spend_guard(
        allow_paid=True,
        provider_name="digitalocean",
        max_spend_usd=1.5,
        max_seconds=1800,
        max_hourly_rate_usd=rate,
        max_hourly_rate_source="provider_capacity_preflight_viable_inventory",
        contender_count=1,
        marker_timeout_seconds=1566,
        startup_no_runtime_timeout_seconds=1446,
        max_attempts=1,
    )
    assert guard["can_launch"] is True
    assert guard["estimated_max_spend_usd"] == 1.4156
    assert guard["max_hourly_rate_source"] == (
        "provider_capacity_preflight_viable_inventory"
    )

    runpod_rate = J._capacity_preflight_hourly_rate(
        {
            "status": "available",
            "viable_gpu_types": [
                {"gpu_type_id": "NVIDIA RTX A6000", "on_demand_price_usd_per_hour": 0.49},
                {"gpu_type_id": "NVIDIA A40", "on_demand_price_usd_per_hour": 0.44},
            ],
        }
    )
    assert runpod_rate == 0.49


def test_cli_persists_manifest_even_when_blocked(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    out_dir = tmp_path / "out"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")

    def fake_run(**_kwargs):
        return {"schema_version": J.SCHEMA_VERSION, "status": "blocked", "blockers": ["pod_did_not_boot"]}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(out_dir),
        "--allow-paid",
    ])

    assert rc == 1
    persisted = json.loads((out_dir / J.JOB_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert persisted["status"] == "blocked"
    assert persisted["blockers"] == ["pod_did_not_boot"]


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


def test_cli_forwards_warm_serve_flags(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "serving"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main([
        "--scenarios", str(scenarios_path),
        "--out-dir", str(tmp_path / "out"),
        "--serve",
        "--serve-idle-timeout", "900",
        "--serve-max-jobs", "3",
        "--serve-ready-timeout", "1200",
        "--startup-no-runtime-timeout", "600",
    ])

    assert rc == 0
    assert captured["serve"] is True
    assert captured["serve_idle_timeout_s"] == 900.0
    assert captured["serve_max_jobs"] == 3
    assert captured["serve_ready_timeout"] == 1200
    assert captured["startup_no_runtime_timeout"] == 600


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


def test_cli_forwards_supervised_startup(monkeypatch, tmp_path: Path) -> None:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(json.dumps(_SCENARIOS), encoding="utf-8")
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared"}

    monkeypatch.setattr(J, "run_isaac_g1_kitchen_parity_job", fake_run)
    rc = J.main(
        [
            "--scenarios",
            str(scenarios_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--supervised-startup",
        ]
    )
    assert rc == 0
    assert captured["supervised_startup"] is True


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
    # feedback-driven stance search agent shipped flat for the runner's bundle-first import
    assert "stance_configuration_agent.py" in names
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


def test_supervised_bundle_and_worker_bootstrap_wire_all_startup_gates(tmp_path: Path) -> None:
    inventory = {
        "schema_version": "kitchen_asset_inventory_checksums.v1",
        "main_usd": "Collected_KitchenRoom/KitchenRoom.usd",
        "file_count": 1,
        "total_bytes": 10,
        "archive_sha256": None,
        "files": [
            {
                "path": "Collected_KitchenRoom/KitchenRoom.usd",
                "sha256": "a" * 64,
                "bytes": 10,
            }
        ],
    }
    zip_path = J.build_parity_bundle(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        kitchen_asset_inventory=inventory,
    )
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        persisted = json.loads(archive.read("kitchen_asset_inventory_checksums.json"))
    assert persisted == inventory
    assert "blueprint_pipeline/isaac_review_renderer_canary.py" in names
    assert "blueprint_pipeline/kitchen_asset_startup_gate.py" in names
    assert "blueprint_pipeline/adaptive_task_stance_configurator.py" in names
    assert "safe_extract_zip" in J.BOOTSTRAP
    assert "blueprint_pipeline.isaac_worker_runtime_preflight" in J.BOOTSTRAP
    assert "blueprint_pipeline.kitchen_asset_startup_gate" in J.BOOTSTRAP
    assert "blueprint_pipeline.isaac_review_renderer_canary" in J.BOOTSTRAP
    assert "supervised_startup_gates.json" in J.BOOTSTRAP
    assert '"--archive","/workspace/kitchen_assets.zip"' in J.BOOTSTRAP
    assert J.BOOTSTRAP.index("kitchen_passed=run_startup_gate") < J.BOOTSTRAP.index(
        '"fast_startup_canary"'
    )
    assert J.BOOTSTRAP.index('"fast_startup_canary"') < J.BOOTSTRAP.index(
        '"review_renderer_canary"'
    )


def test_supervised_launch_spec_binds_worker_gate_mode_and_digest(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "provider_bundle_url.txt").write_text("https://example.test/in")
    (job_dir / "provider_output_put_url.txt").write_text("https://example.test/out")
    digest = "sha256:" + "b" * 64
    spec = J.build_launch_spec(
        job_dir,
        image=f"registry.example/worker@{digest}",
        policy_id="p",
        steps=8,
        supervised_startup=True,
    )
    assert spec.env["PARITY_SUPERVISED_STARTUP"] == "1"
    assert spec.env["BLUEPRINT_WORKER_IMAGE_DIGEST"] == digest
    assert spec.env["PYTHONPATH"] == J.WORKER_BUNDLE_DIR


def test_kitchen_asset_layout_validation_selects_root_kitchen_room() -> None:
    detail = J._inspect_kitchen_asset_namelist(
        ["KitchenRoom.usd", "Sink054/Sink054.usd"],
        source="unit",
        byte_size=1234,
    )

    assert detail["status"] == "PASS"
    assert detail["selected_kitchen_main_usd_relative"] == "KitchenRoom.usd"
    assert detail["expected_worker_kitchen_usd"] == "/workspace/bundle/kitchen/KitchenRoom.usd"
    assert detail["layout"] == "root_kitchen_room"


def test_kitchen_asset_layout_validation_prefers_collected_layout() -> None:
    detail = J._inspect_kitchen_asset_namelist(
        ["KitchenRoom.usd", "Collected_KitchenRoom/KitchenRoom.usd"],
        source="unit",
    )

    assert detail["status"] == "PASS"
    assert detail["selected_kitchen_main_usd_relative"] == "Collected_KitchenRoom/KitchenRoom.usd"
    assert detail["layout"] == "collected_kitchen_room"


def test_kitchen_asset_layout_validation_blocks_missing_main_usd() -> None:
    detail = J._inspect_kitchen_asset_namelist(["Sink054/Sink054.usd"], source="unit")

    assert detail["status"] == "FAIL"
    assert "kitchen_main_usd_missing" in detail["blockers"]
    assert detail["selected_kitchen_main_usd_relative"] is None


def test_reused_kitchen_url_layout_updates_worker_request_without_gpu_spend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    kitchen_buf = io.BytesIO()
    with zipfile.ZipFile(kitchen_buf, "w") as zf:
        zf.writestr("KitchenRoom.usd", "#usda root")
        zf.writestr("Sink054/Sink054.usd", "#usda sink")
    kitchen_bytes = kitchen_buf.getvalue()
    captured: dict = {}

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        with zipfile.ZipFile(bundle_zip) as zf:
            captured["request"] = json.loads(zf.read("request.json"))
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
        return {"status": "completed", "manifest": {}}

    monkeypatch.setattr(
        J,
        "_fetch_provider_artifact_bytes",
        lambda _url, **_kwargs: kitchen_bytes,
    )
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        kitchen_url="https://spaces.example/kitchen.zip?sig=redacted",
        allow_paid=False,
    )

    assert m["status"] == "prepared"
    assert m["kitchen_layout_validation"]["status"] == "PASS"
    assert m["kitchen_layout_validation"]["raw_url_values_recorded"] is False
    assert m["kitchen_staging"]["selected_kitchen_main_usd_relative"] == "KitchenRoom.usd"
    assert captured["request"]["kitchen_usd"] == "/workspace/bundle/kitchen/KitchenRoom.usd"


def test_reused_kitchen_url_layout_failure_blocks_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    kitchen_buf = io.BytesIO()
    with zipfile.ZipFile(kitchen_buf, "w") as zf:
        zf.writestr("Sink054/Sink054.usd", "#usda sink")
    kitchen_bytes = kitchen_buf.getvalue()

    monkeypatch.setattr(
        J,
        "_fetch_provider_artifact_bytes",
        lambda _url, **_kwargs: kitchen_bytes,
    )
    monkeypatch.setattr(
        J,
        "stage_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("staged")),
    )

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        kitchen_url="https://spaces.example/kitchen.zip?sig=redacted",
        allow_paid=False,
    )

    assert m["status"] == "blocked"
    assert "kitchen_asset_layout_validation_failed" in m["blockers"]
    assert "kitchen_main_usd_missing" in m["kitchen_layout_validation"]["blockers"]


@pytest.mark.parametrize(
    "url",
    (
        "file:///etc/passwd",
        "http://169.254.169.254/latest/meta-data",
        "http://127.0.0.1/private.zip",
        "gopher://example.com/archive.zip",
    ),
)
def test_provider_artifact_fetch_rejects_non_https_urls(url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        J._fetch_provider_artifact_bytes(url, timeout=1, max_bytes=1024)


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
    assert "PARITY_RUNNER_TIMEOUT_SECONDS" in body
    assert "subprocess.Popen(cmd, start_new_session=True)" in body
    assert "os.killpg(proc.pid, signal.SIGTERM)" in body
    assert 'mark("runner_done", rc=rc' in body
    assert 'mark("runner_timeout"' in body
    assert "while True:" in body and "putout()" in body
    # tee opens runner_console.log before boot.py's OUT cleanup runs; the cleanup must skip
    # it or the console writes to an unlinked inode and never reaches the output zip.
    assert 'if p.name == "runner_console.log": continue' in body
    cleanup_skip = body.index('if p.name == "runner_console.log": continue')
    assert body.index("pathlib.Path(OUT).iterdir()") < cleanup_skip < body.index("shutil.rmtree(p)")


def test_docker_start_cmd_can_run_image_startup_canary() -> None:
    dsc = J.docker_start_cmd(image_startup_canary=True)
    assert dsc[0] == "-lc"
    body = dsc[1]
    assert "container_bash_started" in body
    assert "parity_image_startup_canary.py" in body
    assert "isaac_g1_parity_image_startup_canary.v2" in body
    assert "blueprint_pipeline.isaac_worker_runtime_preflight" in body
    assert "start_new_session=True" in body
    assert "os.killpg(preflight_process.pid, signal.SIGKILL)" in body
    assert "isaac_runtime_preflight_process_group_timeout" in body
    assert "preflight_deadline_done.wait(930)" in body
    assert 'mark("runner_done", rc=124' in body
    assert "os._exit(124)" in body
    assert "split_isaac_carrier_plus_signed_blueprint_bundle" in body
    assert "BLUEPRINT_EVAL_MANIFEST_URI" in body
    assert "canary_bundle" in body
    assert "--require-nvidia-smi" in body
    assert "--require-rtx-render" in body
    assert "python3 /workspace/parity_image_startup_canary.py" in body
    assert "python /workspace/parity_image_startup_canary.py" in body
    assert "/isaac-sim/python.sh /workspace/parity_image_startup_canary.py" in body
    assert "run_isaac_g1_kitchen_parity_eval.py" not in body
    assert 'mark("runner_done", rc=preflight_rc, image_startup_canary=True' in body


def test_build_launch_spec_carries_policy_and_signed_urls(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    spec = J.build_launch_spec(jd, image="img:tag", policy_id="groot_sonic", steps=80)
    assert spec.image == "img:tag"
    assert spec.env["PARITY_POLICY"] == "groot_sonic"
    assert spec.env["PARITY_STEPS"] == "80"
    assert spec.env["RENDER_WIDTH"] == "1280"
    assert spec.env["RENDER_HEIGHT"] == "960"
    assert spec.env["RENDER_FPS"] == "20"
    assert "PARITY_RUNNER_TIMEOUT_SECONDS" not in spec.env
    assert spec.env["BLUEPRINT_EVAL_MANIFEST_URI"].endswith("sig=A")
    assert spec.env["BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL"].endswith("sig=B")
    assert spec.bootstrap_argv[0] == "-lc"
    assert spec.container_disk_gb >= 120
    assert spec.max_hourly_rate_usd == 5.0


def test_build_launch_spec_threads_runner_timeout(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")

    spec = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        runner_timeout_seconds=840,
    )

    assert spec.env["PARITY_RUNNER_TIMEOUT_SECONDS"] == "840"


def test_build_launch_spec_threads_groot_policy_command(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")

    spec = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="groot_sonic",
        steps=8,
        groot_policy_command=(
            "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
        ),
        groot_policy_command_timeout_seconds=15,
    )

    assert spec.env["PARITY_GROOT_POLICY_COMMAND"] == (
        "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
    )
    assert spec.env["PARITY_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS"] == "15.0"
    assert "--groot-policy-command" in spec.bootstrap_argv[1]
    assert "--groot-policy-command-timeout-seconds" in spec.bootstrap_argv[1]


def test_build_launch_spec_canary_uses_canary_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    spec = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="groot_sonic",
        steps=80,
        image_startup_canary=True,
    )
    assert "parity_image_startup_canary.py" in spec.bootstrap_argv[1]
    assert "run_isaac_g1_kitchen_parity_eval.py" not in spec.bootstrap_argv[1]


def test_parity_bundle_ships_groot_policy_command_modules(tmp_path: Path) -> None:
    bundle_zip = J.build_parity_bundle(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        policy_id="groot_sonic",
        steps=4,
    )

    with zipfile.ZipFile(bundle_zip) as zf:
        names = set(zf.namelist())

    assert "blueprint_pipeline/__init__.py" in names
    assert "blueprint_pipeline/common.py" in names
    assert "blueprint_pipeline/unitree_groot_n17_sonic_policy_runtime.py" in names
    assert "blueprint_pipeline/unitree_groot_n17_sonic_policy_server_command.py" in names


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
    assert reach.env["PARITY_MANIPULATION_REACH_ARM"] == "auto"
    assert 'PARITY_MANIPULATION_REACH_ARM' in body and '--manipulation-reach-arm' in body


def test_negative_manipulation_look_at_is_serialized_as_attached_option(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
    spec = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        manipulation_look_at="-1.591312,1.471274,1.241574",
    )

    assert spec.env["PARITY_MANIPULATION_LOOK_AT"] == "-1.591312,1.471274,1.241574"
    body = J.docker_start_cmd()[1]
    assert '"--manipulation-look-at=" + os.environ["PARITY_MANIPULATION_LOOK_AT"]' in body
    assert '["--manipulation-look-at", os.environ["PARITY_MANIPULATION_LOOK_AT"]]' not in body


def test_dynamic_episode_termination_threads_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")

    default = J.build_launch_spec(jd, image="img:tag", policy_id="p", steps=8)
    assert default.env["PARITY_DYNAMIC_EPISODE_TERMINATION"] == "1"

    off = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        dynamic_episode_termination=False,
    )
    assert off.env["PARITY_DYNAMIC_EPISODE_TERMINATION"] == "0"
    assert "PARITY_EPISODE_MAX_STEPS" not in off.env

    on = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        dynamic_episode_termination=True,
        episode_max_steps=24,
        dynamic_episode_check_every=1,
        capture_every=4,
        placement_topdown_capture=False,
    )

    assert on.env["PARITY_DYNAMIC_EPISODE_TERMINATION"] == "1"
    assert on.env["PARITY_EPISODE_MAX_STEPS"] == "24"
    assert "PARITY_DYNAMIC_EPISODE_CHECK_EVERY" not in on.env
    assert on.env["PARITY_CAPTURE_EVERY"] == "4"
    assert on.env["PARITY_NO_PLACEMENT_TOPDOWN_CAPTURE"] == "1"
    body = J.docker_start_cmd()[1]
    assert 'PARITY_DYNAMIC_EPISODE_TERMINATION' in body
    assert "--dynamic-episode-termination" in body
    assert "--no-dynamic-episode-termination" in body
    assert 'PARITY_EPISODE_MAX_STEPS' in body
    assert "--episode-max-steps" in body
    assert 'PARITY_DYNAMIC_EPISODE_CHECK_EVERY' in body
    assert "--dynamic-episode-check-every" in body
    assert 'PARITY_CAPTURE_EVERY' in body
    assert "--capture-every" in body
    assert 'PARITY_NO_PLACEMENT_TOPDOWN_CAPTURE' in body
    assert "--no-placement-topdown-capture" in body


def test_worker_bootstrap_python_is_syntax_valid() -> None:
    compile(J.BOOTSTRAP, "isaac_g1_kitchen_parity_bootstrap.py", "exec")


def test_local_mp4_repair_assembles_missing_videos_without_topdown_layout_mix(
    monkeypatch,
    tmp_path: Path,
) -> None:
    render_out = tmp_path / "render_output"
    scenario_dir = render_out / "microwave_reach"
    frames_dir = scenario_dir / "frames"
    frames_dir.mkdir(parents=True)
    for prefix in ("overview", "robot_pov", "placement_topdown"):
        for idx in range(2):
            (frames_dir / f"{prefix}_{idx:04d}.png").write_bytes(b"fake-png")
    (frames_dir / "placement_topdown_layout_0000.png").write_bytes(b"layout")
    calls: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"fake-mp4")

        class Proc:
            returncode = 0
            stderr = ""

        return Proc()

    monkeypatch.setattr(review_media.shutil, "which", lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(review_media.subprocess, "run", fake_run)

    repair = J._repair_collected_review_mp4s(
        render_out_dir=render_out,
        result={"scenarios": [{"scenario_id": "microwave_reach"}]},
        fps=20,
    )

    assert repair["status"] == "PASS"
    assert {rec["status"] for rec in repair["repairs"]} == {"repaired"}
    assert (scenario_dir / "overview.mp4").is_file()
    assert (scenario_dir / "robot_pov.mp4").is_file()
    assert (scenario_dir / "placement_topdown.mp4").is_file()
    placement_cmd = next(cmd for cmd in calls if cmd[-1].endswith("placement_topdown.mp4"))
    placement_cmd_text = " ".join(placement_cmd)
    assert "placement_topdown_%04d.png" in placement_cmd_text
    assert "placement_topdown_layout" not in placement_cmd_text


def test_local_mp4_repair_skips_optional_missing_topdown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    render_out = tmp_path / "render_output"
    scenario_dir = render_out / "microwave_reach"
    frames_dir = scenario_dir / "frames"
    frames_dir.mkdir(parents=True)
    for prefix in ("overview", "robot_pov"):
        for idx in range(2):
            (frames_dir / f"{prefix}_{idx:04d}.png").write_bytes(b"fake-png")

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        Path(cmd[-1]).write_bytes(b"fake-mp4")

        class Proc:
            returncode = 0
            stderr = ""

        return Proc()

    monkeypatch.setattr(review_media.shutil, "which", lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(review_media.subprocess, "run", fake_run)

    repair = J._repair_collected_review_mp4s(
        render_out_dir=render_out,
        result={"scenarios": [{"scenario_id": "microwave_reach"}]},
        fps=10,
        optional_videos=("placement_topdown",),
    )

    topdown = next(rec for rec in repair["repairs"] if rec["video"] == "placement_topdown")
    assert repair["status"] == "PASS"
    assert repair["blockers"] == []
    assert topdown["status"] == "skipped_optional"
    assert topdown["optional"] is True
    assert not (scenario_dir / "placement_topdown.mp4").exists()


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


def test_non_white_robot_review_material_mode_threads_env_and_bootstrap(tmp_path: Path) -> None:
    jd = tmp_path / "object_store_real_run"
    jd.mkdir()
    (jd / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
    (jd / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")

    spec = J.build_launch_spec(
        jd,
        image="img:tag",
        policy_id="p",
        steps=8,
        robot_review_material_override=True,
        robot_review_material_mode="non_white_matte",
    )

    assert spec.env["PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE"] == "1"
    assert spec.env["PARITY_ROBOT_REVIEW_MATERIAL_MODE"] == "non_white_matte"
    body = J.docker_start_cmd()[1]
    assert "PARITY_ROBOT_REVIEW_MATERIAL_MODE" in body
    assert "--robot-review-material-mode" in body


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
    def _fake_probe(path: Path) -> dict:
        return {
            "status": "ready",
            "path": str(path),
            "width": 640,
            "height": 480,
            "frame_count": 81,
            "fps": 20.0,
        }

    original_probe = J._probe_video_file
    J._probe_video_file = _fake_probe
    result = {
        "policy_id": "blueprint_default_walk_to_target_smoke_policy",
        "scenarios_executed": 2, "scenarios_passed": 1,
        "scenarios": [{"scenario_id": "entry_to_sink", "task_success": True},
                      {"scenario_id": "narrow_passage_to_sink", "task_success": False}],
    }
    try:
        pkg = J.build_harness_package(
            result=result,
            render_out_dir=tmp_path / "render",
            out_dir=tmp_path / "out",
            requested_render_settings={
                "width": 640,
                "height": 480,
                "fps": 20,
                "expected_frame_count_per_scenario": 81,
            },
        )
    finally:
        J._probe_video_file = original_probe
    assert pkg["wam_evaluator"]["evaluates"] == "video_rollout_fidelity_not_task_success"
    assert pkg["wam_evaluator"]["status"] == "inputs_ready_pending_model_run"
    assert pkg["requested_render_settings"]["width"] == 640
    assert pkg["requested_render_settings"]["expected_frame_count_per_scenario"] == 81
    assert len(pkg["wam_evaluator"]["inputs"]) == 2
    assert pkg["wam_evaluator"]["inputs"][0]["overview_mp4"].endswith("entry_to_sink/overview.mp4")
    assert pkg["wam_evaluator"]["inputs"][0]["media_metadata"]["overview_mp4"]["width"] == 640
    assert pkg["wam_evaluator"]["inputs"][0]["media_metadata"]["overview_mp4"]["frame_count"] == 81
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
                                          robot_review_material_override=True,
                                          width=640, height=480, fps=20, steps=81)
    assert m["status"] == "prepared"
    assert m["provider"] == "vast"
    assert m["launch_request_shape"]["provider"] == "vast"
    assert m["requested_render_settings"] == {
        "steps": 81,
        "width": 640,
        "height": 480,
        "fps": 20,
        "warmup_frames": 6,
        "per_scenario_seconds": 420,
        "expected_frame_count_per_scenario": 81,
    }
    assert m["launch_request_shape"]["steps"] == 81
    assert m["launch_request_shape"]["width"] == 640
    assert m["launch_request_shape"]["height"] == 480
    assert m["launch_request_shape"]["fps"] == 20
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
    monkeypatch.setenv(J.ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV, "1")
    _set_test_worker_image(monkeypatch)
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
                   launch_kwargs=None, prelaunch_guard=None,
                   pending_teardown_lane=None, pending_teardown_max_age_seconds=0,
                   sleep=None, monotonic=None):
        assert prelaunch_guard["can_launch"] is True
        assert prelaunch_guard["required_before_provider_launch"] is True
        assert pending_teardown_lane == J.ISAAC_G1_KITCHEN_PARITY_LANE
        assert pending_teardown_max_age_seconds >= 300
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
                    preserve_instance=False, preserve_blocked_instance=True,
                    progress_timeout_seconds=0):
        captured["collect_job_dir"] = Path(job_dir)
        captured["collect_provider"] = provider.name
        captured["collect_instance_id"] = instance_id
        captured["progress_timeout_seconds"] = progress_timeout_seconds
        captured["preserve_blocked_instance"] = preserve_blocked_instance
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
        max_spend_usd=20.0,
    )

    assert m["status"] == "evidence_collected_closure_required"
    assert "g1_kitchen_attempt_closure_missing" in m["blockers"]
    assert m["launch"]["provider"] == "vast"
    assert m["launch"]["terminated_losers"] == 1
    assert captured["collect_provider"] == "vast"
    assert captured["collect_instance_id"] == "vast-iid"
    assert captured["collect_job_dir"].name == "contender-1-vast"
    assert captured["progress_timeout_seconds"] == 360


def test_paid_launch_blocks_before_provider_call_without_max_spend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_test_worker_image(monkeypatch)
    monkeypatch.delenv(J.ISAAC_G1_MAX_SPEND_USD_ENV, raising=False)
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

    class _NoLaunchProvider:
        name = "digitalocean"

        def __init__(self) -> None:
            self.launch_calls = 0

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"image": spec.image}

        def launch(self, *_args, **_kwargs):
            self.launch_calls += 1
            raise AssertionError("provider launch must be prelaunch-blocked")

    provider = _NoLaunchProvider()
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): provider)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="digitalocean",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        max_attempts=1,
    )

    assert m["status"] == "blocked"
    assert "isaac_g1_prelaunch_spend_guard_not_passed" in m["blockers"]
    assert "isaac_g1_max_spend_usd_missing" in m["blockers"]
    assert m["prelaunch_spend_guard"]["can_launch"] is False
    assert m["prelaunch_spend_guard"]["required_before_provider_launch"] is True
    assert m["prelaunch_spend_guard"]["budget_source"] == "missing"
    assert m["prelaunch_spend_guard"]["claim_boundary"][
        "no_provider_api_call_before_can_launch"
    ] is True
    assert provider.launch_calls == 0


def test_paid_vast_launch_blocks_before_staging_without_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(J.ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV, raising=False)
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _stage_should_not_run(*_args, **_kwargs):
        raise AssertionError("vast-only paid launch should block before staging")

    monkeypatch.setattr(J, "stage_bundle", _stage_should_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="vast",
        allow_paid=True,
        allow_dirty_paid_launch=True,
    )

    assert m["status"] == "blocked"
    assert "vast_provider_disabled_for_paid_isaac_review" in m["blockers"]
    assert m["provider_policy"]["status"] == "blocked"
    assert m["provider_policy"]["override_env"] == J.ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV
    assert "staging" not in m


def test_paid_runpod_launch_blocks_before_staging_without_prebuilt_worker_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    for name in (
        J.ISAAC_WORKER_IMAGE_REF_ENV,
        J.ROBOT_EVAL_WORKER_IMAGE_REF_ENV,
        J.ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_REF_FILE_ENV, str(tmp_path / "missing-image-ref"))
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _stage_should_not_run(*_args, **_kwargs):
        raise AssertionError("missing prebuilt worker image should block before staging")

    monkeypatch.setattr(J, "stage_bundle", _stage_should_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
    )

    assert m["status"] == "blocked"
    assert "prebuilt_isaac_eval_worker_image_ref_missing" in m["blockers"]
    assert m["worker_image_policy"]["status"] == "blocked"
    assert m["worker_image_policy"]["worker_image_ref_file_present"] is False
    assert m["worker_image_policy"]["direct_isaac_base_image_runpod_allowed"] is False
    assert "staging" not in m


def test_paid_digitalocean_capacity_preflight_blocks_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_test_worker_image(monkeypatch)
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    class _CapacityBlockedDigitalOceanProvider:
        name = "digitalocean"

        def available(self):
            return {"provider": self.name, "available": True, "reason": None}

        def capacity_preflight(self, request=None):
            assert request == {"min_gpu_ram_mb": 48000, "requires_rtx": True}
            return {
                "status": "blocked",
                "provider": self.name,
                "blockers": ["digitalocean_gpu_size_region_unavailable"],
                "size_candidates": ["gpu-6000adax1-48gb", "gpu-l40sx1-48gb"],
                "region_candidates": ["atl1", "nyc2"],
                "raw_provider_response_recorded": False,
            }

    def _stage_should_not_run(*_args, **_kwargs):
        raise AssertionError("DigitalOcean no-capacity preflight should block before staging")

    monkeypatch.setattr(J, "stage_bundle", _stage_should_not_run)
    monkeypatch.setattr(
        J,
        "get_render_provider",
        lambda name, warm_candidates=(): _CapacityBlockedDigitalOceanProvider(),
    )

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="digitalocean",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        max_spend_usd=4.0,
    )

    assert m["status"] == "blocked"
    assert "digitalocean_gpu_size_region_unavailable" in m["blockers"]
    assert "provider_capacity_unavailable_before_staging" in m["blockers"]
    assert m["provider_capacity_preflight"]["status"] == "blocked"
    assert "kitchen_layout_validation" not in m
    assert "staging" not in m
    assert "launch_request_shape" not in m


def test_paid_groot_sonic_isaac_parity_blocks_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_test_worker_image(monkeypatch)
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _stage_should_not_run(*_args, **_kwargs):
        raise AssertionError("unwired groot_sonic parity policy should block before staging")

    monkeypatch.setattr(J, "stage_bundle", _stage_should_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        policy_id="groot_sonic",
        allow_paid=True,
        allow_dirty_paid_launch=True,
    )

    assert m["status"] == "blocked"
    assert "groot_sonic_policy_not_connected_to_isaac_parity_runner" in m["blockers"]
    assert (
        "groot_sonic_policy_runtime_presence_not_proven_for_selected_image"
        in m["blockers"]
    )
    assert m["policy_runtime_policy"]["status"] == "blocked"
    assert m["policy_runtime_policy"]["policy_command_configured"] is False
    assert m["policy_runtime_policy"]["runtime_location_proven"] is False
    assert "prior Unitree GR00T/SONIC provider action-command evidence" in m[
        "policy_runtime_policy"
    ]["claim_boundary"]
    assert "staging" not in m


def test_paid_groot_sonic_policy_command_without_runtime_presence_blocks_before_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_test_worker_image(monkeypatch)
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _stage_should_not_run(*_args, **_kwargs):
        raise AssertionError("command-only groot_sonic paid run should block before staging")

    monkeypatch.setattr(J, "stage_bundle", _stage_should_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        policy_id="groot_sonic",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        groot_policy_command=(
            "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
        ),
    )

    assert m["status"] == "blocked"
    assert (
        "groot_sonic_policy_runtime_presence_not_proven_for_selected_image"
        in m["blockers"]
    )
    assert m["policy_runtime_policy"]["status"] == "blocked"
    assert m["policy_runtime_policy"]["policy_command_configured"] is True
    assert m["policy_runtime_policy"]["runtime_location_proven"] is False
    assert m["policy_runtime_policy"][
        "runtime_dependency_install_disallowed_for_paid_launch"
    ] is True
    assert "command string alone is not enough" in m["policy_runtime_policy"]["reason"]
    assert "staging" not in m


def test_paid_groot_sonic_runtime_policy_allows_external_policy_server(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        J.UNITREE_GROOT_POLICY_SERVER_URL_ENV,
        "tcp://policy-server.example:5555",
    )

    policy = J._groot_sonic_policy_runtime_policy(
        policy_id="groot_sonic",
        selected_image="registry.example/blueprint/isaac-eval-worker:test",
        allow_paid=True,
        image_startup_canary=False,
        effective_groot_policy_command=(
            "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
        ),
        effective_groot_policy_command_timeout_seconds=30.0,
    )

    assert policy["status"] == "configured"
    assert policy["runtime_location_proven"] is True
    assert policy["runtime_location_source"] == "external_policy_server_url"
    assert policy["blockers"] == []


def test_paid_groot_sonic_runtime_policy_allows_confirmed_prebaked_image(
    monkeypatch,
) -> None:
    monkeypatch.setenv(J.ISAAC_G1_GROOT_POLICY_RUNTIME_MODE_ENV, "prebaked_worker_image")
    monkeypatch.setenv(J.ISAAC_G1_GROOT_POLICY_PREBAKED_IMAGE_CONFIRMED_ENV, "true")

    policy = J._groot_sonic_policy_runtime_policy(
        policy_id="groot_sonic",
        selected_image="registry.example/blueprint/isaac-groot-sonic-worker:test",
        allow_paid=True,
        image_startup_canary=False,
        effective_groot_policy_command=(
            "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
        ),
        effective_groot_policy_command_timeout_seconds=30.0,
    )

    assert policy["status"] == "configured"
    assert policy["runtime_location_proven"] is True
    assert policy["runtime_location_source"] == "prebaked_worker_image_contract"
    assert policy["blockers"] == []


def test_groot_sonic_policy_command_allows_no_spend_prepared_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_test_worker_image(monkeypatch)
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )
    captured: dict = {}

    class _FakeProvider:
        name = "runpod"

        def available(self):
            return {"available": True}

        def build_request(self, spec, job_dir):
            captured["request"] = {"env": spec.env, "image": spec.image, "job_dir": str(job_dir)}
            return captured["request"]

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/bundle.zip?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/out.zip?sig=B")
        (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=C")
        return {"status": "completed", "manifest": {"key_prefix": key_prefix}}

    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): _FakeProvider())
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        policy_id="groot_sonic",
        allow_paid=False,
        groot_policy_command=(
            "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
        ),
    )

    assert m["status"] == "prepared"
    assert m["policy_runtime_policy"]["status"] == "configured_unproven_no_spend_plan"
    assert m["policy_runtime_policy"]["policy_command_configured"] is True
    assert m["policy_runtime_policy"]["runtime_location_proven"] is False
    assert (
        "groot_sonic_policy_runtime_presence_not_proven_for_selected_image"
        in m["policy_runtime_policy"]["blockers"]
    )
    assert m["launch_request_shape"]["groot_policy_command_configured"] is True
    assert captured["request"]["env"]["PARITY_GROOT_POLICY_COMMAND"] == (
        "python -m blueprint_pipeline.unitree_groot_n17_sonic_policy_server_command"
    )


def test_paid_runpod_large_worker_image_requires_canary_or_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_ref = "registry.example/blueprint/isaac-eval-worker:2026-07-01"
    diagnostic_path = tmp_path / "isaac_worker_image_manifest_diagnostic.json"
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema_version": "isaac_worker_image_manifest_diagnostic.v1",
                "status": "completed",
                "image_ref": image_ref,
                "layer_count": 2,
                "total_compressed_size_bytes": 10_900_000_000,
                "largest_layer_size_bytes": 10_600_000_000,
                "large_image_pull_risk": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_REF_ENV, image_ref)
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV, str(diagnostic_path))
    monkeypatch.delenv(J.ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV, raising=False)
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _stage_should_not_run(*_args, **_kwargs):
        raise AssertionError("large cold RunPod image should require canary before staging")

    monkeypatch.setattr(J, "stage_bundle", _stage_should_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
    )

    assert m["status"] == "blocked"
    assert "large_worker_image_requires_canary_or_warm_provider" in m["blockers"]
    assert m["worker_image_policy"]["status"] == "blocked"
    assert m["worker_image_policy"]["worker_image_manifest_diagnostic"][
        "large_image_pull_risk"
    ] is True
    assert m["worker_image_policy"]["runpod_cold_start_possible"] is True
    assert "staging" not in m


def test_paid_runpod_split_layer_worker_image_allows_bounded_cold_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_ref = "registry.example/blueprint/isaac-eval-worker:split"
    diagnostic_path = tmp_path / "isaac_worker_image_manifest_diagnostic.json"
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema_version": "isaac_worker_image_manifest_diagnostic.v1",
                "status": "completed",
                "image_ref": image_ref,
                "layer_count": 24,
                "total_compressed_size_bytes": 10_600_000_000,
                "largest_layer_size_bytes": 2_420_000_000,
                "large_image_pull_risk": True,
                "split_layer_layout_suitable": True,
                "recommended_startup_no_runtime_timeout_seconds": 1446,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_REF_ENV, image_ref)
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV, str(diagnostic_path))
    monkeypatch.delenv(J.ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV, raising=False)

    selected, policy = J._paid_worker_image_policy(
        image=None,
        allow_paid=True,
        provider_names=["runpod"],
        cold=True,
        warm_only=False,
        image_startup_canary=False,
    )
    assert selected == image_ref
    assert policy["status"] == "allowed"
    assert policy["split_layer_cold_start_suitable"] is True
    assert policy["large_runpod_image_fresh_start_allowed"] is True


def test_paid_image_startup_canary_bypasses_large_image_block_without_harness(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_ref = "registry.example/blueprint/isaac-eval-worker:2026-07-01"
    diagnostic_path = tmp_path / "isaac_worker_image_manifest_diagnostic.json"
    diagnostic_path.write_text(
        json.dumps(
            {
                "schema_version": "isaac_worker_image_manifest_diagnostic.v1",
                "status": "completed",
                "image_ref": image_ref,
                "layer_count": 2,
                "total_compressed_size_bytes": 10_900_000_000,
                "largest_layer_size_bytes": 10_600_000_000,
                "large_image_pull_risk": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_REF_ENV, image_ref)
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV, str(diagnostic_path))
    monkeypatch.setenv(J.ISAAC_G1_MAX_SPEND_USD_ENV, "10.0")
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text(
            "https://spaces.example/bundle.zip?sig=A"
        )
        (job_dir / "provider_output_put_url.txt").write_text(
            "https://spaces.example/out.zip?sig=B"
        )
        (job_dir / "provider_output_get_url.txt").write_text(
            "https://spaces.example/out.zip?sig=C"
        )
        return {"status": "completed", "manifest": {}}

    class _FakeProvider:
        name = "runpod"

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"env": dict(spec.env), "dockerStartCmd": spec.bootstrap_argv}

    captured: dict = {}

    def _fake_launch(provider_obj, job_dir, request, **_kwargs):
        captured["bootstrap"] = request["dockerStartCmd"][1]
        return {
            "status": "launched",
            "instance_id": "runpod-canary",
            "mode": "cold_create_marker_verified",
        }

    def _fake_watch(job_dir, render_out, instance_id, *, provider=None, **_kwargs):
        return {
            "status": "completed",
            "elapsed_seconds": 1,
            "teardown": {"status": "stopped"},
            "runner_result_source": "isaac_g1_kitchen_parity_result.json",
            "last_bootstrap": {"phase": "runner_done", "image_startup_canary": True},
            "timed_out_without_runner_done": False,
            "runner_result": {
                "schema_version": "isaac_g1_parity_image_startup_canary.v2",
                "status": "completed",
                "image_startup_canary": True,
            },
        }

    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): _FakeProvider())
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "launch_with_marker_retry", _fake_launch)
    monkeypatch.setattr(J, "watch_and_collect", _fake_watch)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=[],
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        image_startup_canary=True,
        cold_race_contenders=1,
    )

    assert m["image_startup_canary"] is True
    assert m["status"] == "completed"
    assert m["worker_image_policy"]["status"] == "allowed"
    assert m["worker_image_policy"]["image_startup_canary"] is True
    assert m["worker_image_policy"]["large_runpod_image_fresh_start_allowed"] is True
    assert m["image_startup_canary_result"]["image_startup_canary"] is True
    assert "harness" not in m
    assert "parity_image_startup_canary.py" in captured["bootstrap"]
    assert "run_isaac_g1_kitchen_parity_eval.py" not in captured["bootstrap"]


def test_paid_multi_provider_drops_vast_without_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv(J.ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV, raising=False)
    _set_test_worker_image(monkeypatch)
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text(
            "https://spaces.example/bundle.zip?sig=A"
        )
        (job_dir / "provider_output_put_url.txt").write_text(
            "https://spaces.example/out.zip?sig=B"
        )
        (job_dir / "provider_output_get_url.txt").write_text(
            "https://spaces.example/out.zip?sig=C"
        )
        return {"status": "completed", "manifest": {}}

    class _FakeProvider:
        def __init__(self, name: str) -> None:
            self.name = name

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"env": dict(spec.env), "provider": self.name}

    providers = {"runpod": _FakeProvider("runpod"), "vast": _FakeProvider("vast")}
    captured: dict = {}

    def _fake_launch(provider_obj, job_dir, request, **_kwargs):
        captured["launch_provider"] = provider_obj.name
        return {"status": "launched", "instance_id": "runpod-iid", "mode": "cold_create_marker_verified"}

    def _fake_watch(job_dir, render_out, instance_id, *, provider=None, **_kwargs):
        captured["collect_provider"] = provider.name
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

    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): providers[name])
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "launch_with_marker_retry", _fake_launch)
    monkeypatch.setattr(J, "watch_and_collect", _fake_watch)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod,vast",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        cold_race_contenders=1,
    )

    assert m["status"] == "evidence_collected_closure_required"
    assert "g1_kitchen_attempt_closure_missing" in m["blockers"]
    assert m["provider"] == "runpod"
    assert m["providers"] == ["runpod"]
    assert m["provider_policy"]["status"] == "degraded"
    assert m["provider_policy"]["disabled_paid_providers"] == ["vast"]
    assert captured["launch_provider"] == "runpod"
    assert captured["collect_provider"] == "runpod"
    assert captured["collect_instance_id"] == "runpod-iid"


def test_paid_job_surfaces_blocked_parity_result_without_runtime_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_test_worker_image(monkeypatch)
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
        cold_race_contenders=1,
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
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["status"] == "blocked"
    assert trace["blockers"] == ["all_launch_attempts_flaky"]
    assert trace["attempts"][-1]["result"] == "marker_timeout_terminated"


def test_launch_with_marker_retry_reports_provider_capacity_before_instance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    class _CapacityBlockedProvider:
        name = "digitalocean"

        def __init__(self) -> None:
            self.launch_calls = 0

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            self.launch_calls += 1
            return {
                "status": "blocked",
                "blockers": ["digitalocean_gpu_size_region_unavailable"],
                "attempts": [
                    {
                        "create_status": 422,
                        "size": "gpu-l40sx1-48gb",
                        "region": "tor1",
                        "retryable_region_capacity_error": True,
                    }
                ],
            }

    provider = _CapacityBlockedProvider()
    monkeypatch.setattr(
        J.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("marker polled")),
    )

    res = J.launch_with_marker_retry(provider, jd, {"img": "x"}, max_attempts=2)

    assert res["status"] == "blocked"
    assert res["blockers"] == [
        "digitalocean_gpu_size_region_unavailable",
        "provider_capacity_unavailable_before_instance_created",
    ]
    assert provider.launch_calls == 2
    assert all(item["result"] == "launch_call_failed" for item in res["attempts"])
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["status"] == "blocked"
    assert trace["blockers"] == res["blockers"]


def test_launch_with_marker_retry_reports_runpod_create_capacity_before_instance(
    tmp_path: Path,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()

    class _CapacityBlockedProvider:
        name = "runpod"

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {
                "status": "blocked",
                "blockers": [
                    "runpod_secure_cloud_create_capacity_unavailable",
                    "no_pod_started",
                ],
            }

    res = J.launch_with_marker_retry(
        _CapacityBlockedProvider(),
        jd,
        {"img": "x"},
        max_attempts=1,
    )

    assert res["status"] == "blocked"
    assert res["blockers"] == [
        "runpod_secure_cloud_create_capacity_unavailable",
        "provider_capacity_unavailable_before_instance_created",
    ]


def test_launch_with_marker_retry_blocks_prelaunch_guard_before_launch(
    tmp_path: Path,
) -> None:
    class _NoLaunchProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.launch_calls = 0

        def launch(self, *_args, **_kwargs):
            self.launch_calls += 1
            raise AssertionError("prelaunch guard should block before launch")

    provider = _NoLaunchProvider()
    jd = tmp_path / "job"
    jd.mkdir()
    guard = {
        "schema_version": "isaac_g1_kitchen_parity_prelaunch_spend_guard.v1",
        "status": "blocked",
        "can_launch": False,
        "blockers": ["isaac_g1_max_spend_usd_missing"],
    }

    res = J.launch_with_marker_retry(
        provider,
        jd,
        {"img": "x"},
        max_attempts=2,
        prelaunch_guard=guard,
    )

    assert res["status"] == "blocked"
    assert "isaac_g1_prelaunch_spend_guard_not_passed" in res["blockers"]
    assert "isaac_g1_max_spend_usd_missing" in res["blockers"]
    assert res["attempts"][0]["result"] == "prelaunch_blocked"
    assert provider.launch_calls == 0


def test_launch_with_marker_retry_opens_pending_teardown_after_guard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")
    fp = _make_fake_provider()(marker=True)
    guard = {
        "schema_version": "isaac_g1_kitchen_parity_prelaunch_spend_guard.v1",
        "status": "ready",
        "can_launch": True,
        "blockers": [],
    }

    monkeypatch.setattr(J.time, "sleep", lambda s: None)
    monkeypatch.setattr(J.urllib.request, "urlopen", fp.urlopen)

    res = J.launch_with_marker_retry(
        fp,
        jd,
        {"img": "x"},
        max_attempts=1,
        marker_timeout=5,
        poll=1,
        prelaunch_guard=guard,
    )

    assert res["status"] == "launched"
    assert res["pending_teardown_record"]
    records = paid_lane_guard.load_pending_teardowns()
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "open"
    assert record["lane"] == J.ISAAC_G1_KITCHEN_PARITY_LANE
    assert record["provider"] == "unknown"
    assert record["instance_id"] == "pod0"
    assert record["path"] == res["pending_teardown_record"]


def test_launch_with_marker_retry_cancels_pending_teardown_without_allocation(
    tmp_path: Path,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()

    class _BlockedBeforeAllocationProvider:
        name = "runpod"

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "blocked", "blockers": ["no_capacity"]}

    guard = {
        "schema_version": "isaac_g1_kitchen_parity_prelaunch_spend_guard.v1",
        "status": "ready",
        "can_launch": True,
        "blockers": [],
    }

    res = J.launch_with_marker_retry(
        _BlockedBeforeAllocationProvider(),
        jd,
        {"img": "x"},
        max_attempts=1,
        prelaunch_guard=guard,
    )

    assert res["status"] == "blocked"
    assert paid_lane_guard.load_pending_teardowns() == []
    records = paid_lane_guard.load_pending_teardowns(include_closed=True)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "cancelled_no_allocation"
    assert record["cancel_reason"] == "launch_returned_no_allocation"
    assert record["cancel_evidence"]["blockers"] == ["no_capacity"]


def test_launch_with_marker_retry_closes_pending_teardown_on_api_proof(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    class _ApiConfirmedTerminateProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.terminated: list[str] = []

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "launched", "instance_id": "pod0", "mode": "cold_create"}

        def terminate(self, instance_id):
            self.terminated.append(instance_id)
            return {"status": "terminated", "http": 204}

        def inspect(self, instance_id):
            assert instance_id == "pod0"
            return {"status": "unavailable", "http": 404}

    provider = _ApiConfirmedTerminateProvider()
    guard = {
        "schema_version": "isaac_g1_kitchen_parity_prelaunch_spend_guard.v1",
        "status": "ready",
        "can_launch": True,
        "blockers": [],
    }
    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", _make_fake_provider()(marker=False).urlopen)

    res = J.launch_with_marker_retry(
        provider,
        jd,
        {"img": "x"},
        max_attempts=1,
        marker_timeout=2,
        poll=1,
        prelaunch_guard=guard,
    )

    assert res["status"] == "blocked"
    assert provider.terminated == ["pod0"]
    attempt = res["attempts"][0]
    assert attempt["pending_teardown_status"] == "closed"
    assert attempt["teardown_proof"]["status"] == "PASS"
    assert paid_lane_guard.load_pending_teardowns() == []
    records = paid_lane_guard.load_pending_teardowns(include_closed=True)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "closed"
    assert record["lane"] == J.ISAAC_G1_KITCHEN_PARITY_LANE
    assert record["instance_id"] == "pod0"
    assert record["teardown_proof"]["status"] == "PASS"


def test_launch_with_marker_retry_terminates_pre_runtime_stall(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    class _PreRuntimeProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.terminated: list[str] = []

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "launched", "instance_id": "pod0", "mode": "cold_create"}

        def inspect(self, instance_id):
            return {
                "status": "observed",
                "http": 200,
                "instance_id": instance_id,
                "desiredStatus": "RUNNING",
                "runtime_present": False,
                "public_ip_present": False,
                "machineId": "machine-a",
                "costPerHr": 0.69,
                "raw_provider_response_recorded": False,
            }

        def terminate(self, instance_id):
            self.terminated.append(instance_id)
            return {"status": "terminated", "http": 204}

    provider = _PreRuntimeProvider()
    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", _make_fake_provider()(marker=False).urlopen)

    res = J.launch_with_marker_retry(
        provider,
        jd,
        {"env": {}},
        max_attempts=1,
        marker_timeout=100,
        startup_no_runtime_timeout=3,
        poll=1,
    )

    assert res["status"] == "blocked"
    assert "provider_startup_no_runtime_timeout" in res["blockers"]
    assert provider.terminated == ["pod0"]
    attempt = res["attempts"][0]
    assert attempt["result"] == "startup_no_runtime_timeout_terminated"
    assert attempt["elapsed_seconds"] == 3.0
    assert attempt["startup_no_runtime_snapshot"]["machineId"] == "machine-a"
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert "provider_startup_no_runtime_timeout" in trace["blockers"]
    assert trace["attempts"][0]["result"] == "startup_no_runtime_timeout_terminated"


def test_launch_with_marker_retry_quarantines_repeated_bad_machine_without_second_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    class _RepeatedMachineProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.launch_count = 0
            self.terminated: list[str] = []

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            instance_id = f"pod{self.launch_count}"
            self.launch_count += 1
            return {"status": "launched", "instance_id": instance_id, "mode": "cold_create"}

        def inspect(self, instance_id):
            return {
                "status": "observed",
                "http": 200,
                "instance_id": instance_id,
                "desiredStatus": "RUNNING",
                "runtime_present": False,
                "public_ip_present": False,
                "machineId": "machine-repeated",
                "raw_provider_response_recorded": False,
            }

        def terminate(self, instance_id):
            self.terminated.append(instance_id)
            return {"status": "terminated", "http": 204}

    provider = _RepeatedMachineProvider()
    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", _make_fake_provider()(marker=False).urlopen)

    res = J.launch_with_marker_retry(
        provider,
        jd,
        {"env": {}},
        max_attempts=2,
        marker_timeout=100,
        startup_no_runtime_timeout=3,
        poll=1,
    )

    assert res["status"] == "blocked"
    assert provider.terminated == ["pod0", "pod1"]
    assert res["attempts"][0]["elapsed_seconds"] == 3.0
    assert res["attempts"][1]["elapsed_seconds"] == 1.0
    assert res["attempts"][1]["result"] == "quarantined_machine_terminated"
    assert res["attempts"][1]["quarantined_machine_snapshot"]["machineId"] == "machine-repeated"
    assert "provider_repeated_quarantined_machine" in res["blockers"]
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["quarantined_machine_ids"] == ["machine-repeated"]


_PINNED_TEST_IMAGE = "docker.io/example/worker@sha256:" + "d" * 64


def test_launch_with_marker_retry_records_durable_machine_quarantine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """P0-2: a pre-runtime stall must leave a durable cross-run quarantine
    entry keyed by provider + machine + image digest + Isaac version."""
    from blueprint_pipeline import machine_quarantine_registry as Q

    registry_dir = tmp_path / "quarantine"
    monkeypatch.setenv(Q.REGISTRY_DIR_ENV, str(registry_dir))
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    class _PreRuntimeProvider:
        name = "runpod"

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "launched", "instance_id": "pod0", "mode": "cold_create"}

        def inspect(self, instance_id):
            return {
                "status": "observed",
                "http": 200,
                "instance_id": instance_id,
                "desiredStatus": "RUNNING",
                "runtime_present": False,
                "public_ip_present": False,
                "machineId": "machine-dead",
                "raw_provider_response_recorded": False,
            }

        def terminate(self, instance_id):
            return {"status": "terminated", "http": 204}

    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", _make_fake_provider()(marker=False).urlopen)

    res = J.launch_with_marker_retry(
        _PreRuntimeProvider(),
        jd,
        {"env": {}, "imageName": _PINNED_TEST_IMAGE},
        max_attempts=1,
        marker_timeout=100,
        startup_no_runtime_timeout=3,
        poll=1,
    )

    assert res["status"] == "blocked"
    assert res["attempts"][0]["durable_quarantine_path"]
    entry = Q.find_active_quarantine(
        provider="runpod",
        machine_id="machine-dead",
        image_digest="sha256:" + "d" * 64,
        isaac_version=Q.DEFAULT_ISAAC_VERSION,
        registry_dir=registry_dir,
    )
    assert entry is not None
    assert entry["failure_class"] == "container_never_started"
    assert entry["phase"] == Q.PHASE_PRE_RUNTIME


def test_launch_with_marker_retry_seeds_quarantine_from_durable_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A machine quarantined by a PREVIOUS run is terminated on first
    re-allocation without waiting out a second startup timeout."""
    from blueprint_pipeline import machine_quarantine_registry as Q

    registry_dir = tmp_path / "quarantine"
    monkeypatch.setenv(Q.REGISTRY_DIR_ENV, str(registry_dir))
    Q.record_machine_quarantine(
        provider="runpod",
        machine_id="machine-dead",
        image_digest="sha256:" + "d" * 64,
        isaac_version=Q.DEFAULT_ISAAC_VERSION,
        failure_class="container_never_started",
        phase=Q.PHASE_PRE_RUNTIME,
        registry_dir=registry_dir,
    )
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    class _PreRuntimeProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.terminated: list[str] = []

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {"status": "launched", "instance_id": "pod0", "mode": "cold_create"}

        def inspect(self, instance_id):
            return {
                "status": "observed",
                "http": 200,
                "instance_id": instance_id,
                "desiredStatus": "RUNNING",
                "runtime_present": False,
                "public_ip_present": False,
                "machineId": "machine-dead",
                "raw_provider_response_recorded": False,
            }

        def terminate(self, instance_id):
            self.terminated.append(instance_id)
            return {"status": "terminated", "http": 204}

    provider = _PreRuntimeProvider()
    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "time", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(J.urllib.request, "urlopen", _make_fake_provider()(marker=False).urlopen)

    res = J.launch_with_marker_retry(
        provider,
        jd,
        {"env": {}, "imageName": _PINNED_TEST_IMAGE},
        max_attempts=1,
        marker_timeout=100,
        startup_no_runtime_timeout=30,
        poll=1,
    )

    assert res["status"] == "blocked"
    assert provider.terminated == ["pod0"]
    attempt = res["attempts"][0]
    # Quarantine fired on the first poll, far before the 30s startup timeout.
    assert attempt["result"] == "quarantined_machine_terminated"
    assert attempt["elapsed_seconds"] == 1.0
    assert "provider_repeated_quarantined_machine" in res["blockers"]
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["durable_quarantine_machine_ids"] == ["machine-dead"]


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
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["attempts"][0]["result"] == "marker_timeout_terminated"


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
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["attempts"][0]["result"] == "marker_timeout_stopped"


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
    trace = json.loads((jd / J.LAUNCH_ATTEMPT_TRACE_FILENAME).read_text(encoding="utf-8"))
    assert trace["status"] == "launch_call_failed"


def test_job_warm_only_blocks_without_cold_spend(tmp_path: Path, monkeypatch) -> None:
    _set_test_worker_image(monkeypatch)
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


def test_job_reports_provider_capacity_without_flaky_launch_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _set_test_worker_image(monkeypatch)
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
        return {"status": "completed"}

    class _CapacityBlockedProvider:
        name = "digitalocean"

        def available(self):
            return {"provider": self.name, "available": True, "reason": None}

        def build_request(self, spec, job_dir):
            return {"image": spec.image}

        def launch(self, job_dir, request, *, cold=False, allow_cold_fallback=True):
            return {
                "status": "blocked",
                "blockers": ["digitalocean_gpu_size_region_unavailable"],
            }

    provider = _CapacityBlockedProvider()
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): provider)
    monkeypatch.setattr(
        J.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("marker polled")),
    )

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="digitalocean",
        allow_paid=True,
        max_attempts=1,
    )

    assert m["status"] == "blocked"
    assert "digitalocean_gpu_size_region_unavailable" in m["blockers"]
    assert "provider_capacity_unavailable_before_instance_created" in m["blockers"]
    assert "launch_failed_provider_capacity_unavailable" in m["blockers"]
    assert "launch_failed_all_attempts_flaky" not in m["blockers"]


def _install_fake_warm_serve_stack(monkeypatch, tmp_path: Path, *, ready: bool):
    _set_test_worker_image(monkeypatch)
    monkeypatch.setenv(J.ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV, "true")
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

    class _WarmServeProvider:
        name = "runpod"

        def __init__(self) -> None:
            self.terminated: list[str] = []

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"env": dict(spec.env), "image": spec.image}

        def terminate(self, instance_id):
            self.terminated.append(instance_id)
            return {"status": "terminated", "instance_id": instance_id}

    provider = _WarmServeProvider()

    def _fake_inbox(job_dir, *, key_prefix):
        job_dir = Path(job_dir)
        (job_dir / "warm_broker_base_url.txt").write_text(
            "https://warm-broker.example"
        )
        (job_dir / "warm_broker_token.txt").write_text("t" * 64)
        return {
            "status": "completed",
            "blockers": [],
            "transport": "durable_warm_render_broker",
            "single_object_transport_enabled": False,
            "broker_base_url_file": str(job_dir / "warm_broker_base_url.txt"),
            "broker_token_file": str(job_dir / "warm_broker_token.txt"),
        }

    import blueprint_pipeline.wam_provider_object_store as object_store

    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): provider)
    monkeypatch.setattr(object_store, "presign_warm_inbox_channel", _fake_inbox)
    monkeypatch.setattr(
        J,
        "launch_with_marker_retry",
        lambda *_args, **_kwargs: {
            "status": "launched",
            "instance_id": "serve-pod-1",
            "mode": "cold_create_marker_verified",
        },
    )
    monkeypatch.setattr(
        J,
        "_await_warm_serve_ready",
        lambda *_args, **_kwargs: {
            "ready": ready,
            "instance_id": "serve-pod-1",
            "reason": None if ready else "serve_ready_timeout",
        },
    )
    monkeypatch.setattr(
        J,
        "watch_and_collect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("collected")),
    )
    return provider


def test_job_warm_serve_ready_leaves_pod_running_for_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = _install_fake_warm_serve_stack(monkeypatch, tmp_path, ready=True)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=[],
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        serve=True,
        serve_max_jobs=3,
    )

    assert m["status"] == "serving"
    assert m["warm_serve"]["ready"] is True
    assert provider.terminated == []


def test_job_warm_serve_not_ready_terminates_pod(
    tmp_path: Path,
    monkeypatch,
) -> None:
    provider = _install_fake_warm_serve_stack(monkeypatch, tmp_path, ready=False)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=[],
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        serve=True,
    )

    assert m["status"] == "blocked"
    assert "warm_serve_not_ready" in m["blockers"]
    assert provider.terminated == ["serve-pod-1"]
    assert m["warm_serve"]["not_ready_teardown"]["status"] == "terminated"


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

    clock = {"t": 0.0}
    monkeypatch.setattr(J.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(J.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(
        J,
        "_fetch_provider_artifact_bytes",
        lambda _url, **_kwargs: data,
    )

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

    monkeypatch.setattr(J.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(J.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        J,
        "_fetch_provider_artifact_bytes",
        lambda _url, **_kwargs: data,
    )

    res = J._await_warm_serve_ready(jd, instance_id="pod1", timeout_s=2, poll_interval_s=1)

    assert res["ready"] is True
    assert res["serve_detail"]["launch_session_id"] == "fresh-session"


def test_await_warm_serve_ready_fails_fast_when_runner_done_without_ready(
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
            "phase": "runner_done",
            "launch_session_id": "fresh-session",
            "rc": 0,
        }))
        z.writestr("isaac_runner_exception.json", json.dumps({
            "exception_type": "RuntimeError",
        }))
    data = buf.getvalue()

    slept = {"called": False}
    monkeypatch.setattr(J.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(J.time, "sleep", lambda _s: slept.__setitem__("called", True))
    monkeypatch.setattr(
        J,
        "_fetch_provider_artifact_bytes",
        lambda _url, **_kwargs: data,
    )

    res = J._await_warm_serve_ready(jd, instance_id="pod1", timeout_s=120, poll_interval_s=30)

    assert res["ready"] is False
    assert res["reason"] == "runner_completed_without_warm_serve_ready"
    assert res["last_phase"] == "runner_done"
    assert "isaac_runner_exception.json" in res["zip_entries"]
    assert slept["called"] is False


def test_await_warm_serve_ready_fails_fast_when_runner_timeout_without_ready(
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
            "phase": "runner_timeout",
            "launch_session_id": "fresh-session",
            "timeout_seconds": 840,
        }))
        z.writestr("runner_console.log", "SimulationApp boot did not finish\n")
    data = buf.getvalue()

    slept = {"called": False}
    monkeypatch.setattr(J.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(J.time, "sleep", lambda _s: slept.__setitem__("called", True))
    monkeypatch.setattr(
        J,
        "_fetch_provider_artifact_bytes",
        lambda _url, **_kwargs: data,
    )

    res = J._await_warm_serve_ready(jd, instance_id="pod1", timeout_s=120, poll_interval_s=30)

    assert res["ready"] is False
    assert res["reason"] == "runner_timeout_without_warm_serve_ready"
    assert res["last_phase"] == "runner_timeout"
    assert "runner_console.log" in res["zip_entries"]
    assert slept["called"] is False


def test_await_warm_serve_ready_surfaces_expired_output_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jd = tmp_path / "job"
    jd.mkdir()
    (jd / "provider_output_get_url.txt").write_text("https://spaces.example/out.zip?sig=A")

    monkeypatch.setattr(J.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        J,
        "_fetch_provider_artifact_bytes",
        lambda url, **_kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
        ),
    )

    res = J._await_warm_serve_ready(jd, instance_id="pod1", timeout_s=2, poll_interval_s=1)

    assert res["ready"] is False
    assert res["reason"] == "presigned_url_expired_or_forbidden"
    assert res["http_status"] == 403


def test_local_mp4_repair_labels_truncated_frame_sequences(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A repair over fewer frames than the run's step budget must not read as complete.

    Partial provider uploads (pod died mid-render) previously produced a clean
    "repaired" MP4 that masked the truncation.
    """
    render_out = tmp_path / "render_output"
    scenario_dir = render_out / "microwave_reach"
    frames_dir = scenario_dir / "frames"
    frames_dir.mkdir(parents=True)
    # Only 2 of the expected 5 frames arrived before the provider died.
    for prefix in ("overview", "robot_pov", "placement_topdown"):
        for idx in range(2):
            (frames_dir / f"{prefix}_{idx:04d}.png").write_bytes(b"fake-png")

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        Path(cmd[-1]).write_bytes(b"fake-mp4")

        class Proc:
            returncode = 0
            stderr = ""

        return Proc()

    monkeypatch.setattr(review_media.shutil, "which", lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(review_media.subprocess, "run", fake_run)

    repair = J._repair_collected_review_mp4s(
        render_out_dir=render_out,
        result={"scenarios": [{"scenario_id": "microwave_reach"}]},
        fps=20,
        expected_frame_count=5,
    )

    assert repair["status"] == "FAIL"
    assert {rec["status"] for rec in repair["repairs"]} == {"repaired_truncated"}
    assert any(
        blocker.startswith("mp4_repair_truncated_frames:overview:2<5")
        for blocker in repair["blockers"]
    )
    # Evidence is preserved for human review — the video still exists.
    assert (scenario_dir / "overview.mp4").is_file()
    for rec in repair["repairs"]:
        assert rec["expected_frame_count"] == 5


def test_local_mp4_repair_full_frame_count_still_reads_repaired(
    monkeypatch,
    tmp_path: Path,
) -> None:
    render_out = tmp_path / "render_output"
    scenario_dir = render_out / "microwave_reach"
    frames_dir = scenario_dir / "frames"
    frames_dir.mkdir(parents=True)
    for prefix in ("overview", "robot_pov", "placement_topdown"):
        for idx in range(5):
            (frames_dir / f"{prefix}_{idx:04d}.png").write_bytes(b"fake-png")

    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001
        Path(cmd[-1]).write_bytes(b"fake-mp4")

        class Proc:
            returncode = 0
            stderr = ""

        return Proc()

    monkeypatch.setattr(review_media.shutil, "which", lambda name: "/usr/local/bin/ffmpeg" if name == "ffmpeg" else None)
    monkeypatch.setattr(review_media.subprocess, "run", fake_run)

    repair = J._repair_collected_review_mp4s(
        render_out_dir=render_out,
        result={"scenarios": [{"scenario_id": "microwave_reach"}]},
        fps=20,
        expected_frame_count=5,
    )

    assert repair["status"] == "PASS"
    assert {rec["status"] for rec in repair["repairs"]} == {"repaired"}
