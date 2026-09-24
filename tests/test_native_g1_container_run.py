from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_container_run import prepare_g1_container_run
from blueprint_pipeline.native_g1_development_worker import REQUEST_SCHEMA
from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


def _inputs(root: Path) -> dict[str, Path]:
    repo = root / "blueprint"
    (repo / "src/blueprint_pipeline").mkdir(parents=True)
    packet = root / "packet"
    packet.mkdir()
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    official = root / "official"
    (official / ".git").mkdir(parents=True)
    (official / "src").mkdir()
    (official / "scripts").mkdir()
    (official / "action_provider").mkdir()
    (official / "scripts/policy_server.py").write_text("server")
    (official / "action_provider/action_provider_sonic.py").write_text("sonic")
    for name in ("inventory.json", "encoder.onnx", "decoder.onnx", "source.json", "source.zip"):
        (root / name).write_text(name)
    request = {
        "schema_version": REQUEST_SCHEMA,
        "candidate_id": "humanoidarena_dp_g1_dex3_sonic",
        "bundle_root": str(packet),
        "inventory_path": str(root / "inventory.json"),
        "checkpoint_root": str(checkpoints),
        "policy_server_source": str(official / "scripts/policy_server.py"),
        "sonic_provider_source": str(official / "action_provider/action_provider_sonic.py"),
        "sonic_encoder": str(root / "encoder.onnx"),
        "sonic_encoder_sha256": "sha256:" + "a" * 64,
        "sonic_decoder": str(root / "decoder.onnx"),
        "sonic_decoder_sha256": "sha256:" + "b" * 64,
        "python_executable": "/local/python",
        "runtime_provisioning_receipt_path": "/local/provisioning.json",
        "port": 8443,
        "max_steps": 100,
        "device": "cuda:0",
        "rights_review": {"status": "approved_for_development_simulation"},
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    (root / "request.json").write_text(json.dumps(request))
    return {
        "request_path": root / "request.json",
        "source_receipt_path": root / "source.json",
        "source_packet_path": root / "source.zip",
        "output_dir": root / "output",
        "repo_root": repo,
    }


def test_container_plan_provisions_then_runs_same_sealed_worker(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    plan = prepare_g1_container_run(**paths)
    request = json.loads((paths["output_dir"] / "work/native_g1_development_episode_request.v1.json").read_text())
    assert plan["status"] == "staged_not_executed"
    assert plan["image"] == NATIVE_TASK_ARENA_IMAGE
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    assert request["request_digest"] == canonical_digest(request, digest_field="request_digest")
    assert request["python_executable"] == "/isaac-sim/python.sh"
    assert request["runtime_provisioning_receipt_path"].startswith("/blueprint-g1-output/runtime/")
    assert request["rights_review"] == {"status": "approved_for_development_simulation"}
    command = plan["command"]
    assert command[:5] == ["docker", "run", "--rm", "--pull", "never"]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--gpus") + 1] == "device=0"
    assert command[command.index("--entrypoint") + 2] == NATIVE_TASK_ARENA_IMAGE
    shell = command[-1]
    assert shell.index("native_task_runtime_source_provision") < shell.index("native_g1_development_worker")
    assert "native_g1_development_worker" in shell
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert any("dst=/blueprint-g1-output" in mount and "readonly" not in mount for mount in mounts)
    assert any("dst=/blueprint-src,readonly" in mount for mount in mounts)
    assert any(f"dst={tmp_path / 'official'},readonly" in mount for mount in mounts)


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
