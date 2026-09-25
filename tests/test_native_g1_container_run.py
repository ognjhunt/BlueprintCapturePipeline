from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_container_run as container_run
from blueprint_pipeline.native_g1_container_run import prepare_g1_container_run
from blueprint_pipeline.native_g1_development_worker import (
    REQUEST_SCHEMA,
    RIGHTS_SCHEMA,
    PINNED_SOURCE_REVISION,
)
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


@pytest.fixture(autouse=True)
def pinned_source_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(container_run, "_source_revision", lambda *args, **kwargs: "pinned")
    monkeypatch.setattr(
        container_run,
        "_verify_packet",
        lambda _root: {
            "arena_scene_plan_digest": SCENE_DIGEST,
            "receipt_digest": "sha256:" + "f" * 64,
        },
    )
    monkeypatch.setattr(
        container_run,
        "verify_native_task_runtime_source_packet",
        lambda *_args, **_kwargs: {
            "receipt_digest": "sha256:" + "e" * 64,
        },
    )
    monkeypatch.setattr(
        container_run,
        "verify_g1_host_asset_identities",
        lambda **_kwargs: {
            "policy_role": "manipulation",
            "inventory_file_sha256": INVENTORY_DIGEST,
            "candidate_inventory_digest": "sha256:" + "d" * 64,
        },
    )


SCENE = {"task_kind": "rigid_pick_place", "robot": {"robot_id": "unitree_g1"}}
SCENE_DIGEST = canonical_digest(SCENE, digest_field="plan_digest")
INVENTORY_DIGEST = "sha256:" + "b" * 64


def _inputs(root: Path) -> dict[str, Path]:
    repo = root / "blueprint"
    (repo / "src/blueprint_pipeline").mkdir(parents=True)
    packet = root / "packet"
    packet.mkdir()
    scene = {**SCENE, "plan_digest": SCENE_DIGEST}
    (packet / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(scene))
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    official = root / "official"
    (official / "lerobot/scripts").mkdir(parents=True)
    (official / "lerobot/src/lerobot").mkdir(parents=True)
    (official / "isaaclab_twist2_g1/action_provider").mkdir(parents=True)
    (official / "lerobot/scripts/serve_lerobot_vla_http.py").write_text("server")
    (official / "isaaclab_twist2_g1/action_provider/action_provider_sonic.py").write_text("sonic")
    subprocess.run(["git", "init", "-q", str(official)], check=True)
    policy_runtime = root / "policy-runtime"
    (policy_runtime / "bin").mkdir(parents=True)
    policy_python = policy_runtime / "bin/python"
    policy_python.write_text("#!/bin/sh\n")
    policy_python.chmod(0o755)
    for name in ("inventory.json", "encoder.onnx", "decoder.onnx", "source.json", "source.zip"):
        (root / name).write_text(name)
    rights = {
        "schema_version": RIGHTS_SCHEMA,
        "status": "approved_for_development_simulation",
        "candidate_id": "humanoidarena_dp_g1_dex3_sonic",
        "scene_plan_digest": scene["plan_digest"],
        "inventory_file_sha256": INVENTORY_DIGEST,
        "source_revision": PINNED_SOURCE_REVISION,
        "human_reviewer": "test reviewer",
        "checkpoint_terms_reviewed": True,
        "source_and_sonic_terms_reviewed": True,
    }
    rights["rights_review_digest"] = canonical_digest(rights, digest_field="rights_review_digest")
    request = {
        "schema_version": REQUEST_SCHEMA,
        "candidate_id": "humanoidarena_dp_g1_dex3_sonic",
        "bundle_root": str(packet),
        "inventory_path": str(root / "inventory.json"),
        "checkpoint_root": str(checkpoints),
        "policy_server_source": str(official / "lerobot/scripts/serve_lerobot_vla_http.py"),
        "sonic_provider_source": str(
            official / "isaaclab_twist2_g1/action_provider/action_provider_sonic.py"
        ),
        "sonic_encoder": str(root / "encoder.onnx"),
        "sonic_encoder_sha256": "sha256:" + "a" * 64,
        "sonic_decoder": str(root / "decoder.onnx"),
        "sonic_decoder_sha256": "sha256:" + "b" * 64,
        "python_executable": str(policy_python),
        "runtime_provisioning_receipt_path": "/local/provisioning.json",
        "port": 8443,
        "max_steps": 100,
        "device": "cuda:0",
        "rights_review": rights,
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    (root / "request.json").write_text(json.dumps(request))
    return {
        "request_path": root / "request.json",
        "source_receipt_path": root / "source.json",
        "source_packet_path": root / "source.zip",
        "policy_runtime_root": policy_runtime,
        "output_dir": root / "output",
        "repo_root": repo,
    }


def test_container_plan_provisions_then_runs_same_sealed_worker(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    plan = prepare_g1_container_run(**paths)
    request = json.loads(
        (paths["output_dir"] / "work/native_g1_development_episode_request.v1.json").read_text()
    )
    assert plan["status"] == "staged_not_executed"
    assert plan["image"] == NATIVE_TASK_ARENA_IMAGE
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    assert request["request_digest"] == canonical_digest(request, digest_field="request_digest")
    assert request["python_executable"] == str(paths["policy_runtime_root"] / "bin/python")
    assert request["runtime_provisioning_receipt_path"].startswith("/blueprint-g1-output/runtime/")
    assert request["rights_review"]["status"] == "approved_for_development_simulation"
    assert plan["host_preflight"]["status"] == "host_inputs_verified"
    assert (
        plan["host_preflight"]["rights_review_digest"]
        == request["rights_review"]["rights_review_digest"]
    )
    command = plan["command"]
    assert command[:5] == ["docker", "run", "--rm", "--pull", "never"]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--gpus") + 1] == "device=0"
    assert command[command.index("--entrypoint") + 2] == NATIVE_TASK_ARENA_IMAGE
    shell = command[-1]
    compile(shlex.split(shell)[2], "g1_policy_probe", "exec")
    assert shell.index("policy-runtime/bin/python") < shell.index(
        "native_task_runtime_source_provision"
    )
    assert shell.index("native_task_runtime_source_provision") < shell.index(
        "native_g1_development_worker"
    )
    assert "native_g1_development_worker" in shell
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert any("dst=/blueprint-g1-output" in mount and "readonly" not in mount for mount in mounts)
    assert any("dst=/blueprint-src,readonly" in mount for mount in mounts)
    assert any(f"dst={tmp_path / 'official'},readonly" in mount for mount in mounts)
    assert any(f"dst={tmp_path / 'policy-runtime'},readonly" in mount for mount in mounts)


def test_container_plan_refuses_missing_rights_before_output(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    request = json.loads(paths["request_path"].read_text())
    request["rights_review"]["checkpoint_terms_reviewed"] = False
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    paths["request_path"].write_text(json.dumps(request))
    with pytest.raises(ValueError, match="g1_worker_rights_review_invalid"):
        prepare_g1_container_run(**paths)
    assert not paths["output_dir"].exists()


def test_container_plan_refuses_model_identity_failure_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)

    def changed_checkpoint(**_kwargs):
        raise ValueError("g1_preflight_checkpoint_identity_mismatch")

    monkeypatch.setattr(container_run, "verify_g1_host_asset_identities", changed_checkpoint)
    with pytest.raises(ValueError, match="g1_preflight_checkpoint_identity_mismatch"):
        prepare_g1_container_run(**paths)
    assert not paths["output_dir"].exists()


def test_container_plan_refuses_source_packet_failure_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)

    def changed_source(*_args, **_kwargs):
        raise ValueError("native_task_runtime_source_packet_identity_mismatch")

    monkeypatch.setattr(container_run, "verify_native_task_runtime_source_packet", changed_source)
    with pytest.raises(ValueError, match="native_task_runtime_source_packet_identity_mismatch"):
        prepare_g1_container_run(**paths)
    assert not paths["output_dir"].exists()


def test_container_plan_refuses_navigation_without_scene_goal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    request = json.loads(paths["request_path"].read_text())
    request["candidate_id"] = "humanoidarena_dp_g1_dex3_sonic_vision_navi"
    request["rights_review"]["candidate_id"] = request["candidate_id"]
    request["rights_review"]["rights_review_digest"] = canonical_digest(
        request["rights_review"], digest_field="rights_review_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    paths["request_path"].write_text(json.dumps(request))
    monkeypatch.setattr(
        container_run,
        "verify_g1_host_asset_identities",
        lambda **_kwargs: {
            "policy_role": "movement_navigation",
            "inventory_file_sha256": INVENTORY_DIGEST,
            "candidate_inventory_digest": "sha256:" + "d" * 64,
        },
    )
    with pytest.raises(ValueError, match="g1_navigation_goal_or_visible_marker_missing"):
        prepare_g1_container_run(**paths)
    assert not paths["output_dir"].exists()


def test_container_plan_refuses_output_inside_sealed_packet(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    paths["output_dir"] = tmp_path / "packet" / "episode"
    with pytest.raises(ValueError, match="g1_container_output_directory_invalid"):
        prepare_g1_container_run(**paths)
    assert not paths["output_dir"].exists()


def test_container_plan_refuses_symlinked_checkpoint_root(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    request = json.loads(paths["request_path"].read_text())
    link = tmp_path / "checkpoint-link"
    link.symlink_to(tmp_path / "checkpoints")
    request["checkpoint_root"] = str(link)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    paths["request_path"].write_text(json.dumps(request))
    with pytest.raises(ValueError, match="g1_container_input_symlink_forbidden"):
        prepare_g1_container_run(**paths)
    assert not paths["output_dir"].exists()


def test_container_plan_refuses_policy_python_outside_mounted_runtime(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    request = json.loads(paths["request_path"].read_text())
    outside = tmp_path / "other-python"
    outside.write_text("#!/bin/sh\n")
    outside.chmod(0o755)
    request["python_executable"] = str(outside)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    paths["request_path"].write_text(json.dumps(request))
    with pytest.raises(ValueError, match="g1_container_policy_python_invalid"):
        prepare_g1_container_run(**paths)
    assert not paths["output_dir"].exists()


def test_execute_refuses_unready_host_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    def prepare(**kwargs: object) -> dict:
        kwargs["output_dir"].mkdir()
        return {"command": ["docker", "run", "fixture"]}
    def fail_host(**_kwargs: object) -> dict:
        raise ValueError("g1_container_pinned_image_unavailable")
    def forbidden_docker(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("docker must not start")
    monkeypatch.setattr(container_run.sys, "platform", "linux")
    monkeypatch.setattr(container_run, "prepare_g1_container_run", prepare)
    monkeypatch.setattr(container_run, "record_g1_container_host", fail_host)
    monkeypatch.setattr(container_run.subprocess, "run", forbidden_docker)
    with pytest.raises(ValueError, match="g1_container_pinned_image_unavailable"):
        container_run.main([
            "--request", str(paths["request_path"]),
            "--source-receipt", str(paths["source_receipt_path"]),
            "--source-packet", str(paths["source_packet_path"]),
            "--policy-runtime-root", str(paths["policy_runtime_root"]),
            "--output-dir", str(paths["output_dir"]),
            "--execute",
        ])
