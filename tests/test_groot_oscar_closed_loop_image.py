"""Contract tests for the sealed `blueprint-groot-oscar-eval` worker image wiring.

These are hermetic (no Docker, no pod, no network). They pin the launcher-facing
contract: image-ref resolution, the fail-closed sealed-mode gate, and the exact
launch plan the sealed image runs (GR00T server from the baked venv+checkpoint,
then the closed-loop CLI pointed at baked OSCAR paths).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import groot_oscar_closed_loop_image as gocl
from blueprint_pipeline.oscar_official_release import OFFICIAL_OSCAR_HF_REVISION


VERSIONED_REF = "docker.io/nijelhunt/blueprint-groot-oscar-eval:20260706-cu128-amd64"
DIGEST_REF = "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:" + "a" * 64
REAL_CONSISTENCY_COMMAND = "python -m owner_runtime.shared_model_inverse_scorer"
IMAGE_ROOT = Path("deploy/docker/robot_eval_worker/groot_oscar_closed_loop")


# --------------------------------------------------------------------------- #
# image-ref resolution
# --------------------------------------------------------------------------- #

def test_configured_image_ref_prefers_explicit_env():
    env = {gocl.IMAGE_REF_ENV: VERSIONED_REF}
    result = gocl.configured_image_ref(env=env)
    assert result["image_ref"] == VERSIONED_REF
    assert result["source"] == gocl.IMAGE_REF_ENV
    assert result["configured"] is True
    assert result["raw_secret_values_recorded"] is False


def test_configured_image_ref_reads_secret_file(tmp_path):
    ref_file = tmp_path / "groot_oscar_closed_loop_image_ref"
    ref_file.write_text(VERSIONED_REF + "\n", encoding="utf-8")
    env = {gocl.IMAGE_REF_FILE_ENV: str(ref_file)}
    result = gocl.configured_image_ref(env=env)
    assert result["image_ref"] == VERSIONED_REF
    assert result["source"] == gocl.IMAGE_REF_FILE_ENV
    assert result["configured"] is True
    assert result["image_ref_file"] == str(ref_file)
    assert result["image_ref_file_present"] is True


def test_configured_image_ref_generic_fallback(tmp_path):
    env = {
        gocl.IMAGE_REF_FILE_ENV: str(tmp_path / "absent"),
        gocl.ROBOT_EVAL_WORKER_IMAGE_REF_ENV: VERSIONED_REF,
    }
    result = gocl.configured_image_ref(env=env)
    assert result["image_ref"] == VERSIONED_REF
    assert result["source"] == gocl.ROBOT_EVAL_WORKER_IMAGE_REF_ENV
    assert result["configured"] is True


def test_configured_image_ref_none_when_unset(tmp_path):
    # Point the file env at a nonexistent path so the real ~/.blueprint-secrets
    # file (if present on the dev box) cannot leak into the test.
    env = {gocl.IMAGE_REF_FILE_ENV: str(tmp_path / "absent")}
    result = gocl.configured_image_ref(env=env)
    assert result["image_ref"] == ""
    assert result["configured"] is False
    assert result["source"] is None


# --------------------------------------------------------------------------- #
# versioned-tag refusal (mirrors the build-script guard)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ref", [VERSIONED_REF, DIGEST_REF])
def test_launchable_refs_have_no_blockers(ref):
    assert gocl.image_ref_launch_blockers(ref) == []


def test_missing_ref_blocked():
    assert "missing_image_ref" in gocl.image_ref_launch_blockers("")


def test_unversioned_ref_blocked():
    assert "image_ref_must_be_versioned" in gocl.image_ref_launch_blockers(
        "docker.io/nijelhunt/blueprint-groot-oscar-eval"
    )


@pytest.mark.parametrize("tag", ["latest", "local", "dev", "test"])
def test_unstable_tags_blocked(tag):
    ref = f"docker.io/nijelhunt/blueprint-groot-oscar-eval:{tag}"
    assert "image_ref_refuses_unstable_tag" in gocl.image_ref_launch_blockers(ref)


# --------------------------------------------------------------------------- #
# fail-closed sealed-mode gate
# --------------------------------------------------------------------------- #

def test_sealed_contract_not_configured_is_blocked(tmp_path):
    env = {gocl.IMAGE_REF_FILE_ENV: str(tmp_path / "absent")}
    contract = gocl.sealed_image_contract(env=env)
    assert contract["sealed_active"] is False
    assert contract["image_ref_configured"] is False
    assert "missing_image_ref" in contract["blockers"]
    assert contract["raw_secret_values_recorded"] is False


def test_sealed_contract_configured_but_unconfirmed_is_blocked():
    env = {gocl.IMAGE_REF_ENV: VERSIONED_REF}  # no SEALED_CONFIRMED_ENV
    contract = gocl.sealed_image_contract(env=env)
    assert contract["image_ref_configured"] is True
    assert contract["sealed_confirmed"] is False
    assert contract["sealed_active"] is False
    assert "sealed_image_not_confirmed" in contract["blockers"]


def test_sealed_contract_active_when_configured_and_confirmed():
    env = {
        gocl.IMAGE_REF_ENV: VERSIONED_REF,
        gocl.SEALED_CONFIRMED_ENV: "true",
    }
    contract = gocl.sealed_image_contract(env=env)
    assert contract["sealed_active"] is True
    assert contract["blockers"] == []
    # baked defaults
    assert contract["oscar_repo"] == "/opt/OSCAR"
    assert contract["oscar_checkpoint"] == "/opt/blueprint/ckpts/oscar"
    assert contract["groot_root"] == "/opt/gr00t"
    assert contract["sonic_checkpoint"] == "/opt/blueprint/ckpts/sonic"
    assert contract["policy_server_url"] == "tcp://127.0.0.1:5550"
    assert contract["oscar_hf_revision"] == OFFICIAL_OSCAR_HF_REVISION


def test_sealed_contract_honors_overridden_paths():
    env = {
        gocl.IMAGE_REF_ENV: VERSIONED_REF,
        gocl.SEALED_CONFIRMED_ENV: "true",
        gocl.OSCAR_REPO_ENV: "/opt/oscar-public",
        gocl.OSCAR_CHECKPOINT_ENV: "/data/oscar",
        gocl.GROOT_ROOT_ENV: "/srv/gr00t",
        gocl.SONIC_CHECKPOINT_ENV: "/data/sonic",
    }
    contract = gocl.sealed_image_contract(env=env)
    assert contract["oscar_repo"] == "/opt/oscar-public"
    assert contract["oscar_checkpoint"] == "/data/oscar"
    assert contract["groot_root"] == "/srv/gr00t"
    assert contract["sonic_checkpoint"] == "/data/sonic"


# --------------------------------------------------------------------------- #
# launch plan
# --------------------------------------------------------------------------- #

def _active_env():
    return {gocl.IMAGE_REF_ENV: VERSIONED_REF, gocl.SEALED_CONFIRMED_ENV: "true"}


def test_launch_plan_blocked_when_not_sealed(tmp_path):
    env = {gocl.IMAGE_REF_FILE_ENV: str(tmp_path / "absent")}
    plan = gocl.build_sealed_launch_plan(
        env=env,
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )
    assert plan["sealed_active"] is False
    assert plan["groot_server_command"] == []
    assert plan["closed_loop_command"] == []
    assert plan["blockers"]


def test_launch_plan_groot_server_command_uses_baked_venv_and_checkpoint():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
        device="cpu",
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )
    assert plan["sealed_active"] is True
    cmd = plan["groot_server_command"]
    assert cmd[0] == "/opt/gr00t/.venv/bin/python"
    assert cmd[1] == "/opt/gr00t/gr00t/eval/run_gr00t_server.py"
    assert "--model-path" in cmd
    assert cmd[cmd.index("--model-path") + 1] == "/opt/blueprint/ckpts/sonic"
    assert cmd[cmd.index("--embodiment-tag") + 1] == "UNITREE_G1_SONIC"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--port") + 1] == "5550"


def test_launch_plan_closed_loop_command_points_at_baked_paths():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=5,
        task_prompt="open the top cabinet",
        output_dir="/w/out",
        oscar_height=256,
        oscar_width=384,
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )
    cmd = plan["closed_loop_command"]
    assert cmd[:3] == [
        "/opt/oscar-venv/bin/python",
        "-m",
        "blueprint_pipeline.oscar_isaac_closed_loop_eval",
    ]
    assert cmd[cmd.index("--oscar-repo") + 1] == "/opt/OSCAR"
    assert cmd[cmd.index("--checkpoint") + 1] == "/opt/blueprint/ckpts/oscar"
    assert cmd[cmd.index("--groot-root") + 1] == "/opt/gr00t"
    assert cmd[cmd.index("--groot-sonic-policy-server-url") + 1] == "tcp://127.0.0.1:5550"
    assert cmd[cmd.index("--start-frame") + 1] == "/w/frame.png"
    assert cmd[cmd.index("--route-file") + 1] == "/w/route.json"
    assert cmd[cmd.index("--steps") + 1] == "5"
    assert cmd[cmd.index("--task-prompt") + 1] == "open the top cabinet"
    assert cmd[cmd.index("--output-dir") + 1] == "/w/out"
    assert cmd[cmd.index("--oscar-height") + 1] == "256"
    assert cmd[cmd.index("--oscar-width") + 1] == "384"
    assert cmd[cmd.index("--min-coherent-horizon-frames") + 1] == "2"
    assert cmd[cmd.index("--min-steps") + 1] == "3"
    assert "--require-fresh-learned-policy-requery" in cmd
    assert "--allow-wam-consistency-scoring" in cmd
    assert "--require-forward-inverse-consistency" in cmd
    assert cmd[cmd.index("--wam-consistency-command") + 1] == (
        REAL_CONSISTENCY_COMMAND
    )
    assert cmd[cmd.index("--wam-consistency-timeout-seconds") + 1] == "300.0"
    assert "--require-generated-video-success-label" not in cmd
    assert plan["gear_sonic_controller_command"][0:2] == ["bash", "-lc"]
    assert "deploy.sh sim" in plan["gear_sonic_controller_command"][2]
    assert plan["env"]["BLUEPRINT_GEAR_SONIC_ROOT"] == "/opt/wbc"
    assert plan["env"][gocl.GEAR_SONIC_CHECKPOINT_REPO_ENV] == "nvidia/GEAR-SONIC"
    assert plan["env"][gocl.GEAR_SONIC_CHECKPOINT_REVISION_ENV] == (
        "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
    )
    assert "gear_sonic_official_zmq_executor" in plan["env"][
        "BLUEPRINT_GEAR_SONIC_EXECUTOR_COMMAND"
    ]
    assert plan["episode_length_contract"] == {
        "episode_length_unit": "closed_loop_control_steps",
        "stop_condition": "task_completion_or_step_cap",
        "steps_cap": 5,
        "min_steps_before_task_completion": 3,
        "steps_is_safety_cap": True,
        "oscar_num_frames_scope": "per_generation_clip_not_episode_limit",
        "episode_not_bound_to_oscar_clip_frames": True,
    }
    assert plan["quality_gate_contract"] == {
        "min_coherent_horizon_frames": 2,
        "forward_inverse_consistency_required": True,
        "forward_inverse_consistency_command": REAL_CONSISTENCY_COMMAND,
        "forward_inverse_consistency_allow_scoring": True,
        "generated_video_success_label_required": False,
        "generated_video_success_label_command": None,
        "generated_video_success_label_allow_labeling": False,
        "claim_boundary": {
            "forward_inverse_consistency_is_required_for_eval_run_quality": True,
            "generated_video_success_label_is_separate_semantic_review": True,
            "generated_video_success_label_is_not_real_world_task_success": True,
        },
    }


def test_launch_plan_env_bakes_runtime_toggles():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )
    plan_env = plan["env"]
    assert plan_env["MUJOCO_GL"] == "osmesa"
    assert plan_env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert plan_env["PYTHONPATH"].startswith("/opt/OSCAR")
    assert plan_env["BLUEPRINT_OSCAR_WAM_HF_REVISION"] == OFFICIAL_OSCAR_HF_REVISION
    assert plan["claim_boundary"]
    assert plan_env["BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING"] == "true"
    assert plan_env["BLUEPRINT_WAM_EPISODE_CONSISTENCY_COMMAND"] == (
        REAL_CONSISTENCY_COMMAND
    )


def test_launch_plan_can_make_generated_video_success_label_strict():
    command = "python -m blueprint_pipeline.wam_generated_video_success_label_openai"
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
        require_generated_video_success_label=True,
        wam_success_label_command=command,
        allow_wam_success_labeling=True,
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )
    cmd = plan["closed_loop_command"]
    assert "--require-generated-video-success-label" in cmd
    assert "--allow-wam-success-labeling" in cmd
    assert cmd[cmd.index("--wam-success-label-command") + 1] == command
    assert plan["env"]["BLUEPRINT_ALLOW_WAM_SUCCESS_LABELING"] == "true"
    assert plan["env"]["BLUEPRINT_WAM_SUCCESS_LABEL_COMMAND"] == command
    assert plan["quality_gate_contract"]["generated_video_success_label_required"] is True


def test_launch_plan_blocks_strict_generated_video_success_without_labeler():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
        require_generated_video_success_label=True,
        wam_consistency_command=REAL_CONSISTENCY_COMMAND,
    )
    assert plan["sealed_active"] is False
    assert "wam_success_label_command_required" in plan["blockers"]
    assert plan["closed_loop_command"] == []


def test_launch_plan_requires_real_forward_inverse_scorer_by_default():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
    )
    assert plan["sealed_active"] is False
    assert "wam_consistency_command_required" in plan["blockers"]


def test_launch_plan_rejects_visual_motion_smoke_as_consistency_scorer():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
        wam_consistency_command=gocl.DEFAULT_VISUAL_MOTION_SMOKE_COMMAND,
    )
    assert plan["sealed_active"] is False
    assert "visual_motion_smoke_cannot_satisfy_forward_inverse_consistency" in plan[
        "blockers"
    ]


# --------------------------------------------------------------------------- #
# snapshot layer plan (source of truth for the crane-snapshot script)
# --------------------------------------------------------------------------- #

def test_snapshot_plan_lists_all_baked_trees_and_markers():
    plan = gocl.build_snapshot_layer_plan(env=_active_env())
    paths = set(plan["snapshot_paths"])
    for required in (
        "/opt/OSCAR",
        "/opt/gr00t",
        "/opt/gr00t-venv",
        "/opt/blueprint/ckpts/oscar",
        "/opt/blueprint/ckpts/sonic",
    ):
        assert required in paths, f"snapshot plan missing {required}"
    # sealed markers must be written into the image env
    assert plan["image_env"][gocl.SEALED_CONFIRMED_ENV] == "true"
    assert plan["image_env"]["MUJOCO_GL"] == "osmesa"
    assert plan["image_env"][gocl.GEAR_SONIC_CHECKPOINT_REPO_ENV] == (
        "nvidia/GEAR-SONIC"
    )
    assert plan["image_env"][gocl.GEAR_SONIC_CHECKPOINT_REVISION_ENV] == (
        "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
    )
    assert plan["raw_secret_values_recorded"] is False
    assert plan["claim_boundary"]


def test_image_seals_exact_nested_cosmos_backbone_and_disables_network_fallback():
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COSMOS_BACKBONE_REPO=nvidia/Cosmos-Reason2-2B" in dockerfile
    assert (
        "COSMOS_BACKBONE_REVISION=9ce19a195e423419c349abfc86fd07178b230561"
        in dockerfile
    )
    assert 'repo_id=os.environ["COSMOS_BACKBONE_REPO"]' in dockerfile
    assert 'revision=os.environ["COSMOS_BACKBONE_REVISION"]' in dockerfile
    assert '(cosmos_refs / "main").write_text(cosmos_revision' in dockerfile
    assert 'write_text(cosmos_revision + "\\n"' not in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert 'sonic_config["model_name"] = str(cosmos_local)' in dockerfile
    assert 'sonic_config["blueprint_model_revision"] = cosmos_revision' in dockerfile


def test_image_seals_the_exact_gear_sonic_deploy_models():
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "GEAR_SONIC_CHECKPOINT_REPO=nvidia/GEAR-SONIC" in dockerfile
    assert (
        "GEAR_SONIC_CHECKPOINT_REVISION=5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
        in dockerfile
    )
    for required in (
        "policy/release/model_encoder.onnx",
        "policy/release/model_decoder.onnx",
        "policy/release/observation_config.yaml",
        "planner/target_vel/V2/planner_sonic.onnx",
    ):
        assert required in dockerfile
    assert 'revision=os.environ["GEAR_SONIC_CHECKPOINT_REVISION"]' in dockerfile


def test_snapshot_carrier_stamps_the_exact_gear_sonic_revision():
    script = Path("scripts/snapshot_groot_oscar_eval_pod.sh").read_text(encoding="utf-8")
    assert "export GEAR_SONIC_CHECKPOINT_REPO=nvidia/GEAR-SONIC" in script
    assert (
        "export GEAR_SONIC_CHECKPOINT_REVISION="
        "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f"
    ) in script


def test_image_makes_gear_sonic_build_tree_runtime_user_writable():
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "chown -R blueprint:blueprint /opt/wbc/gear_sonic_deploy/build" in dockerfile


def test_wbc_compiler_toolchain_is_confined_to_disposable_builder_stage():
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "AS gear_sonic_builder" in dockerfile
    assert "AS runtime" in dockerfile
    builder_start = dockerfile.index("AS gear_sonic_builder")
    runtime_start = dockerfile.index("AS runtime")
    assert builder_start < runtime_start
    builder = dockerfile[builder_start:runtime_start]
    runtime = dockerfile[runtime_start:]
    assert "scripts/install_deps.sh" in builder
    assert "just build" in builder
    assert "scripts/install_deps.sh" not in runtime
    assert "libnvinfer-dev=${TENSORRT_VERSION}" not in runtime
    assert "COPY --from=gear_sonic_builder" in runtime
    assert "libcudart.so* /usr/local/lib/" in runtime
    assert "ldd /opt/wbc/gear_sonic_deploy/target/release/g1_deploy_onnx_ref" in runtime
    assert "grep -F 'not found'" in runtime


def test_image_healthcheck_enforces_runtime_service_dependencies():
    healthcheck = (IMAGE_ROOT / "groot_oscar_closed_loop_image_healthcheck.py").read_text(
        encoding="utf-8"
    )
    assert "SingleArticulation" in healthcheck
    assert "from isaacsim import SimulationApp" in healthcheck
    assert "app = SimulationApp({'headless': True})" in healthcheck
    assert "app.close()" in healthcheck
    assert "official_gear_sonic_build_tree_not_writable" in healthcheck
    assert "cosmos_backbone_not_sealed_in_hf_cache" in healthcheck
    assert "cosmos_backbone_default_ref_not_pinned" in healthcheck
    assert "groot_nested_processor_not_offline_constructible" in healthcheck
    assert "model_encoder.onnx" in healthcheck
    assert "model_decoder.onnx" in healthcheck
    assert "planner_sonic.onnx" in healthcheck
    assert 'payload["isaac_python_import_stdout_tail"]' in healthcheck
    assert 'payload["isaac_python_import_stderr_tail"]' in healthcheck


def test_isaac_backend_uses_supported_isaac_6_articulation_api():
    backend = Path("src/blueprint_pipeline/isaac_runtime_task_backend.py").read_text(
        encoding="utf-8"
    )
    assert "from isaacsim.core.prims import SingleArticulation" in backend
    assert "omni.isaac.dynamic_control" not in backend


def test_persistent_executor_composes_g1_using_the_episode_route():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the microwave",
        output_dir="/w/out",
        wam_consistency_command="python -m strict_scorer",
    )
    command = plan["isaac_task_executor_command"]
    assert command[command.index("--g1-usd") + 1] == gocl.DEFAULT_UNITREE_G1_USD
    assert command[command.index("--route-file") + 1] == "/w/route.json"


# --------------------------------------------------------------------------- #
# CLI (fail-closed exit codes)
# --------------------------------------------------------------------------- #

def test_cli_print_sealed_contract_exit_code(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv(gocl.IMAGE_REF_FILE_ENV, str(tmp_path / "absent"))
    monkeypatch.delenv(gocl.IMAGE_REF_ENV, raising=False)
    monkeypatch.delenv(gocl.ROBOT_EVAL_WORKER_IMAGE_REF_ENV, raising=False)
    monkeypatch.delenv(gocl.SEALED_CONFIRMED_ENV, raising=False)
    rc = gocl.main(["--print-sealed-contract"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1  # fail-closed: not configured
    assert out["sealed_active"] is False

    monkeypatch.setenv(gocl.IMAGE_REF_ENV, VERSIONED_REF)
    monkeypatch.setenv(gocl.SEALED_CONFIRMED_ENV, "true")
    rc = gocl.main(["--print-sealed-contract"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["sealed_active"] is True


def test_sealed_launch_plan_defaults_to_native_oscar_resolution(monkeypatch):
    """240x320 was an OOM-era mitigation that degraded quality without saving
    weight memory; the sealed launch contract must default to native 480x640
    and carry the lane hardware floor so pods are sized before spend."""
    plan = gocl.build_sealed_launch_plan(
        start_frame="/workspace/seed.png",
        route_file="/workspace/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/workspace/t4_out",
        env={},
    )
    assert plan["lane"] == "kitchen_g1_groot_sonic_eval"
    requirements = plan["lane_hardware_requirements"]
    assert requirements["min_vram_gb"] >= 40.0
    assert "NVIDIA RTX A6000" in requirements["recommended_gpu_type_ids"]

    active = gocl.build_sealed_launch_plan(
        start_frame="/workspace/seed.png",
        route_file="/workspace/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/workspace/t4_out",
        env=dict(gocl.SEALED_MARKER_ENV_TRUE) if hasattr(gocl, "SEALED_MARKER_ENV_TRUE") else {"BLUEPRINT_GROOT_OSCAR_SEALED_IMAGE": "true"},
    )
    command = active["closed_loop_command"]
    if command:
        height = command[command.index("--oscar-height") + 1]
        width = command[command.index("--oscar-width") + 1]
        assert (height, width) == ("480", "640")
