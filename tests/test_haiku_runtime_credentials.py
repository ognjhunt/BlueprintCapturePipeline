"""Native keys use existing private-file staging without wider worker access."""
from __future__ import annotations

import base64
import os
import subprocess

import pytest

from blueprint_pipeline import single_g1_kitchen_episode_runpod as single
from blueprint_pipeline import task_evaluation_scene_configuration_warm_remote_protocol as warm
from blueprint_pipeline import vast_provider_adapter as vast
from blueprint_pipeline import wam_compute_providers as providers


@pytest.mark.parametrize("unsafe", ["missing", "symlink", "public", "relative"])
def test_native_staging_refuses_unsafe_file_without_mutating_source(tmp_path, unsafe):
    source = tmp_path / "native"
    source.write_text("offline-native-fixture\n")
    source.chmod(0o600)
    if unsafe == "missing":
        selected = tmp_path / "absent"
    elif unsafe == "symlink":
        selected = tmp_path / "link"
        selected.symlink_to(source)
    elif unsafe == "public":
        source.chmod(0o644)
        selected = source
    else:
        selected = "relative-native-key"
    with pytest.raises(ValueError, match="invalid_vast_runtime_secret_file:ANTHROPIC_API_KEY_FILE"):
        vast._runtime_secret_file_values({"ANTHROPIC_API_KEY_FILE": selected})
    assert source.read_text() == "offline-native-fixture\n"


def test_native_value_is_only_in_private_startup_partition(tmp_path):
    source = tmp_path / "native"
    source.write_text("offline-native-fixture\n")
    source.chmod(0o600)
    values = vast._runtime_secret_file_values({"ANTHROPIC_API_KEY_FILE": source})
    env = vast._probe_env(job_dir=tmp_path, enable_isaac_smoke=False,
                         forward_hf_token=False, runtime_secret_file_values=values)
    public, private = vast._scene_configuration_startup_environments(env)
    name = vast.VAST_RUNTIME_SECRET_BOOTSTRAP_PREFIX + "ANTHROPIC_API_KEY_FILE"
    assert name not in public and "ANTHROPIC_API_KEY_FILE" not in public
    assert base64.b64decode(private[name]).decode() == "offline-native-fixture"
    assert "offline-native-fixture" not in str(public)


def test_warm_child_cannot_inherit_native_credentials():
    env = {**os.environ, "ANTHROPIC_API_KEY": "offline-native-fixture",
           "ANTHROPIC_API_KEY_FILE": "/private/fixture",
           vast.VAST_RUNTIME_SECRET_BOOTSTRAP_PREFIX + "ANTHROPIC_API_KEY_FILE": "fixture"}
    command = warm._warm_no_secret_shell_command(
        "python3 -c 'import os; assert not any(k.startswith(\"ANTHROPIC_API_KEY\") or k.startswith(\"BLUEPRINT_VAST_RUNTIME_SECRET_B64_\") for k in os.environ)'"
    )
    result = subprocess.run(["bash", "-c", command], env=env, capture_output=True,
                            text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert "offline-native-fixture" not in result.stdout + result.stderr


@pytest.mark.parametrize("model", [None, "gpt-6-luna", "gpt-6-luna-2026-09-01", "claude-haiku-5-5"])
def test_direct_single_episode_holds_native_judge_transport(monkeypatch, model):
    monkeypatch.delenv("BLUEPRINT_OPENAI_WAM_SUCCESS_LABEL_MODEL", raising=False)
    judge = "python -m blueprint_pipeline.wam_generated_video_success_label_openai"
    if model:
        judge += " --model " + model
    assert single._direct_native_judge_transport_blockers({
        "closed_loop_command": ["python", "--wam-success-label-command", judge],
    }) == ["single_episode_native_haiku_secret_transport_unqualified"]


def test_direct_single_episode_preserves_explicit_other_model():
    assert single._direct_native_judge_transport_blockers({"closed_loop_command": [
        "python", "--wam-consistency-command",
        "python -m blueprint_pipeline.wam_episode_consistency_label_openai --model gpt-6-sol",
    ]}) == []


@pytest.mark.parametrize(("worker_model", "host_model", "args", "held"), [
    ("gpt-6-sol", "", "", False),
    ("claude-haiku-5-5", "gpt-6-sol", "", True),
    ("", "", "--model=gpt-6-sol", False),
    ("", "", "--model gpt-6-sol --model claude-haiku-5-5", True),
    ("", "", "--model claude-haiku-5-5 --model=gpt-6-sol", False),
    ("   ", "gpt-6-sol", "", True),
    ("gpt-6-sol", "", "--model '   '", True),
])
def test_direct_native_hold_matches_worker_environment_and_cli(monkeypatch, worker_model, host_model, args, held):
    name = "BLUEPRINT_OPENAI_WAM_SUCCESS_LABEL_MODEL"
    monkeypatch.setenv(name, host_model)
    plan = {"env": {name: worker_model}, "closed_loop_command": [
        "python", "--wam-success-label-command",
        "python -m blueprint_pipeline.wam_generated_video_success_label_openai " + args,
    ]}
    assert bool(single._direct_native_judge_transport_blockers(plan)) is held


def test_runpod_refuses_unqualified_private_file_transport_before_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(providers, "create_runpod_wam_async_run",
                        lambda **kwargs: pytest.fail("provider allocation must not be attempted"))
    spec = providers.WamComputeLaunchSpec(name="fixture", bundle_path=tmp_path / "bundle",
        runtime_secret_file_paths={"ANTHROPIC_API_KEY_FILE": "/private/fixture"})
    result = providers.RunPodWamComputeProvider().create(spec, tmp_path / "job", allow_paid_launch=True)
    assert "runpod_private_runtime_secret_file_transport_unqualified" in result.blockers


@pytest.mark.parametrize("options", [
    ["--wam-success-label-command=python -m blueprint_pipeline.wam_generated_video_success_label_openai"],
    ["--wam-success-label-command", "python -m blueprint_pipeline.wam_generated_video_success_label_openai --model gpt-6-sol",
     "--wam-success-label-command=python -m blueprint_pipeline.wam_generated_video_success_label_openai"],
])
def test_direct_native_hold_matches_outer_cli(options):
    assert single._direct_native_judge_transport_blockers({"closed_loop_command": ["python", *options]}) == [
        "single_episode_native_haiku_secret_transport_unqualified",
    ]
