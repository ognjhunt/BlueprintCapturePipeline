"""Contract tests for the sealed `blueprint-groot-oscar-eval` worker image wiring.

These are hermetic (no Docker, no pod, no network). They pin the launcher-facing
contract: image-ref resolution, the fail-closed sealed-mode gate, and the exact
launch plan the sealed image runs (GR00T server from the baked venv+checkpoint,
then the closed-loop CLI pointed at baked OSCAR paths).
"""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline import groot_oscar_closed_loop_image as gocl
from blueprint_pipeline.oscar_official_release import OFFICIAL_OSCAR_HF_REVISION


VERSIONED_REF = "docker.io/nijelhunt/blueprint-groot-oscar-eval:20260706-cu128-amd64"
DIGEST_REF = "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:" + "a" * 64


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


def test_configured_image_ref_generic_fallback():
    env = {gocl.ROBOT_EVAL_WORKER_IMAGE_REF_ENV: VERSIONED_REF}
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
    )
    cmd = plan["closed_loop_command"]
    assert cmd[:3] == ["python", "-m", "blueprint_pipeline.oscar_isaac_closed_loop_eval"]
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
    assert "--require-fresh-learned-policy-requery" in cmd


def test_launch_plan_env_bakes_runtime_toggles():
    plan = gocl.build_sealed_launch_plan(
        env=_active_env(),
        start_frame="/w/frame.png",
        route_file="/w/route.json",
        steps=3,
        task_prompt="open the fridge",
        output_dir="/w/out",
    )
    plan_env = plan["env"]
    assert plan_env["MUJOCO_GL"] == "osmesa"
    assert plan_env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert plan_env["PYTHONPATH"].startswith("/opt/OSCAR")
    assert plan_env["BLUEPRINT_OSCAR_WAM_HF_REVISION"] == OFFICIAL_OSCAR_HF_REVISION
    assert plan["claim_boundary"]


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
    assert plan["raw_secret_values_recorded"] is False
    assert plan["claim_boundary"]


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
