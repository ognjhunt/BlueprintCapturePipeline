"""The private team worker seals scene, episode, and simulator close evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_policy_execution_packet import (
    prepare_g1_team_policy_execution_packet,
)
from blueprint_pipeline import native_g1_team_policy_worker as worker
from tests.test_native_g1_team_policy_authority import _authority
from tests.test_native_g1_team_policy_run_request import NOW


COMMIT = "a" * 40


def _inputs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_policy_run_request.time.time", lambda: NOW)
    authority, _ = _authority(tmp_path, monkeypatch)
    packet = prepare_g1_team_policy_execution_packet(
        **authority,
        output_dir=tmp_path / "execution-packet",
        implementation_commit=COMMIT,
    )
    setup = packet["trusted_setup"]
    scene = tmp_path / "scene"
    scene.mkdir()
    plan = {
        "scene_id": setup["scene_id"],
        "task_id": setup["task_id"],
        "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "unitree_g1"},
        "task_spec": {
            "task_success_contract_digest": setup["task_success_contract_digest"],
        },
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    (scene / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(plan))
    monkeypatch.setattr(
        worker, "_verify_packet",
        lambda root: {
            "arena_scene_plan_digest": plan["plan_digest"],
            "receipt_digest": "sha256:" + "f" * 64,
        },
    )
    monkeypatch.setattr(worker, "require_pinned_sonic_source", lambda source: None)
    monkeypatch.setattr(worker, "_require_artifact", lambda path, digest: None)
    args = {
        "execution_packet_path": Path(packet["packet_path"]),
        "expected_implementation_commit": COMMIT,
        "scene_packet_root": scene,
        "runtime_provisioning_receipt_path": tmp_path / "runtime.json",
        "sonic_provider_source": tmp_path / "sonic.py",
        "sonic_encoder": tmp_path / "encoder.onnx",
        "sonic_encoder_sha256": "sha256:" + "c" * 64,
        "sonic_decoder": tmp_path / "decoder.onnx",
        "sonic_decoder_sha256": "sha256:" + "d" * 64,
        "output_dir": tmp_path / "worker-result",
        "max_steps": 2,
    }
    worker._execution_packet(Path(packet["packet_path"]), COMMIT)
    return args, plan, packet


def test_approved_worker_runs_one_episode_and_seals_close(tmp_path: Path, monkeypatch) -> None:
    args, plan, packet = _inputs(tmp_path, monkeypatch)
    closed = []
    app = SimpleNamespace(close=lambda: closed.append("simulator"))
    built = SimpleNamespace(env=SimpleNamespace(close=lambda: closed.append("environment")))
    monkeypatch.setattr(worker, "_launch_scene", lambda **kwargs: (app, {"status": "launched"}))
    monkeypatch.setattr(worker, "_build_scene", lambda **kwargs: (built, {"passed": True}))
    monkeypatch.setattr(worker, "build_pinned_g1_team_sonic_bridge", lambda **kwargs: object())

    def episode(**kwargs):
        assert kwargs["built"] is built
        assert kwargs["profile"] == packet["request"]["policy_profile"]
        assert kwargs["credential"] is None
        return {
            "status": "completed_development_only",
            "result_digest": "sha256:" + "e" * 64,
            "policy_query_count": 2,
        }

    monkeypatch.setattr(worker, "run_g1_team_supervised_episode", episode)
    result = worker.run_g1_team_policy_worker(**args)
    assert result["status"] == "completed_development_only", (result["phase_reached"], result["blocker_type"])
    assert result["execution_packet_digest"] == packet["packet_digest"]
    assert result["policy_query_count"] == 2
    assert result["teardown"] == {"environment": "closed", "simulator": "closed"}
    assert closed == ["environment", "simulator"]
    assert json.loads((args["output_dir"] / worker.FILENAME).read_text()) == result
    preclose = json.loads((args["output_dir"] / worker.PRECLOSE_FILENAME).read_text())
    assert preclose["status"] == "awaiting_simulator_close"
    assert preclose["teardown"]["simulator"] == "close_requested"
    assert plan["plan_digest"] == worker.canonical_digest(plan, digest_field="plan_digest")


def test_scene_failure_retains_preclose_before_isaac_exit(tmp_path: Path, monkeypatch) -> None:
    args, _, _ = _inputs(tmp_path, monkeypatch)

    def close() -> None:
        preclose = json.loads((args["output_dir"] / worker.PRECLOSE_FILENAME).read_text())
        assert preclose["phase_reached"] == "scene_build"
        assert preclose["blocker_type"] == "ValueError"
        raise SystemExit(0)

    monkeypatch.setattr(
        worker, "_launch_scene",
        lambda **kwargs: (SimpleNamespace(close=close), {"status": "launched"}),
    )
    monkeypatch.setattr(
        worker,
        "_build_scene",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("private provider detail")),
    )
    result = worker.run_g1_team_policy_worker(**args)
    assert result["status"] == "blocked"
    assert result["blocker_type"] == "ValueError"
    assert result["teardown"] == {"environment": "not_started", "simulator": "closed"}, (result["phase_reached"], result["blocker_type"])
    assert "private provider detail" not in (args["output_dir"] / worker.FILENAME).read_text()


def test_tampered_execution_packet_blocks_before_launch(tmp_path: Path, monkeypatch) -> None:
    args, _, _ = _inputs(tmp_path, monkeypatch)
    packet = json.loads(args["execution_packet_path"].read_text())
    packet["implementation_commit"] = "b" * 40
    tampered_path = tmp_path / "tampered-execution-packet.json"
    tampered_path.write_text(json.dumps(packet))
    args["execution_packet_path"] = tampered_path

    def must_not_launch(**kwargs):
        raise AssertionError("untrusted packet reached simulator")

    monkeypatch.setattr(worker, "_launch_scene", must_not_launch)
    result = worker.run_g1_team_policy_worker(**args)
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "execution_packet"
    assert result["teardown"] == {"environment": "not_started", "simulator": "not_started"}
