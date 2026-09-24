from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_development_worker as worker


SCENE = "sha256:" + "a" * 64
INVENTORY = "sha256:" + "b" * 64
CANDIDATE = "humanoidarena_dp_g1_dex3_sonic"


def _request(tmp_path: Path) -> dict:
    bundle = tmp_path / "packet"
    bundle.mkdir()
    (bundle / "native_task_arena_scene_plan.v1.json").write_text(
        json.dumps({"plan_digest": SCENE, "robot": {"robot_id": "unitree_g1"}})
    )
    rights = {
        "schema_version": worker.RIGHTS_SCHEMA,
        "status": "approved_for_development_simulation",
        "candidate_id": CANDIDATE,
        "scene_plan_digest": SCENE,
        "inventory_file_sha256": INVENTORY,
        "source_revision": worker.PINNED_SOURCE_REVISION,
        "human_reviewer": "test reviewer",
        "checkpoint_terms_reviewed": True,
        "source_and_sonic_terms_reviewed": True,
    }
    rights["rights_review_digest"] = canonical_digest(rights, digest_field="rights_review_digest")
    request = {
        "schema_version": worker.REQUEST_SCHEMA,
        "candidate_id": CANDIDATE,
        "bundle_root": str(bundle),
        "inventory_path": str(tmp_path / "inventory.json"),
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "policy_server_source": str(tmp_path / "server.py"),
        "sonic_provider_source": str(tmp_path / "sonic.py"),
        "sonic_encoder": str(tmp_path / "encoder.onnx"),
        "sonic_encoder_sha256": "sha256:" + "c" * 64,
        "sonic_decoder": str(tmp_path / "decoder.onnx"),
        "sonic_decoder_sha256": "sha256:" + "d" * 64,
        "python_executable": str(tmp_path / "python"),
        "runtime_provisioning_receipt_path": str(tmp_path / "provisioning.json"),
        "port": 8443,
        "max_steps": 2,
        "device": "cuda:0",
        "rights_review": rights,
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def _preflight(**kwargs):
    return {
        "status": "staged_inputs_verified",
        "scene_plan_digest": SCENE,
        "candidate_id": CANDIDATE,
        "inventory_file_sha256": INVENTORY,
    }


def _packet(root):
    return {"arena_scene_plan_digest": SCENE, "receipt_digest": "sha256:" + "f" * 64}


def test_rights_review_refuses_launch_before_simulator(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    request["rights_review"]["checkpoint_terms_reviewed"] = False
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", _preflight)
    monkeypatch.setattr(worker, "_verify_packet", _packet)

    def must_not_launch(**kwargs):
        raise AssertionError("simulator launched without rights")

    monkeypatch.setattr(worker, "_launch_scene", must_not_launch)
    output = tmp_path / "blocked"
    result = worker.run_g1_development_worker(request=request, output_dir=output)
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "rights_review"
    assert result["teardown"] == {"environment": "not_started", "simulator": "not_started"}
    assert json.loads((output / worker.RESULT_FILENAME).read_text()) == result


def test_episode_uses_shared_scene_and_closes_all_resources(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", _preflight)
    monkeypatch.setattr(worker, "_verify_packet", _packet)
    closed = []
    app = SimpleNamespace(close=lambda: closed.append("simulator"))
    env = SimpleNamespace(close=lambda: closed.append("environment"))
    built = SimpleNamespace(env=env, plan={"plan_digest": SCENE})
    monkeypatch.setattr(worker, "_launch_scene", lambda **kwargs: (app, {"status": "launched"}))
    monkeypatch.setattr(worker, "_build_scene", lambda **kwargs: (built, {"passed": True}))

    def run_episode(**kwargs):
        assert kwargs["built"] is built
        assert kwargs["preflight_inputs"]["bundle_root"] == Path(request["bundle_root"])
        return {"status": "completed_development_only", "result_digest": "sha256:" + "e" * 64}

    monkeypatch.setattr(worker, "run_g1_supervised_built_scene_episode", run_episode)
    output = tmp_path / "completed"
    result = worker.run_g1_development_worker(request=request, output_dir=output)
    assert result["status"] == "completed_development_only"
    assert result["rights_review_digest"] == request["rights_review"]["rights_review_digest"]
    assert result["supervised_episode"]["result_digest"] == "sha256:" + "e" * 64
    assert closed == ["environment", "simulator"]
    assert json.loads((output / worker.RESULT_FILENAME).read_text()) == result


def test_failed_episode_retains_blocker_and_closes_resources(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", _preflight)
    monkeypatch.setattr(worker, "_verify_packet", _packet)
    closed = []
    app = SimpleNamespace(close=lambda: closed.append("simulator"))
    env = SimpleNamespace(close=lambda: closed.append("environment"))
    monkeypatch.setattr(worker, "_launch_scene", lambda **kwargs: (app, {}))
    monkeypatch.setattr(worker, "_build_scene", lambda **kwargs: (SimpleNamespace(env=env), {}))

    def fail(**kwargs):
        raise RuntimeError("controller_inference_failed")

    monkeypatch.setattr(worker, "run_g1_supervised_built_scene_episode", fail)
    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "failed"
    )
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "episode"
    assert result["blocker"] == {
        "type": "RuntimeError", "message": "controller_inference_failed"
    }
    assert closed == ["environment", "simulator"]


def test_preflight_failure_still_writes_terminal_receipt(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(worker, "_verify_packet", _packet)

    def fail(**kwargs):
        raise ValueError("model_bytes_mismatch")

    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", fail)
    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "preflight-failed"
    )
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "preflight"
    assert result["blocker"]["message"] == "model_bytes_mismatch"


def test_changed_packet_binding_refuses_simulator_launch(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(
        worker, "_verify_packet",
        lambda root: {"arena_scene_plan_digest": "sha256:" + "0" * 64},
    )
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", _preflight)

    def must_not_launch(**kwargs):
        raise AssertionError("simulator launched for mismatched packet")

    monkeypatch.setattr(worker, "_launch_scene", must_not_launch)
    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "packet-mismatch"
    )
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "preflight"
    assert result["blocker"]["message"] == "g1_worker_preflight_incomplete"


def test_unverified_episode_receipt_cannot_complete_worker(tmp_path: Path, monkeypatch) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(worker, "_verify_packet", _packet)
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", _preflight)
    app = SimpleNamespace(close=lambda: None)
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(worker, "_launch_scene", lambda **kwargs: (app, {}))
    monkeypatch.setattr(worker, "_build_scene", lambda **kwargs: (SimpleNamespace(env=env), {}))
    monkeypatch.setattr(
        worker, "run_g1_supervised_built_scene_episode",
        lambda **kwargs: {"status": "blocked"},
    )
    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "unverified"
    )
    assert result["status"] == "blocked"
    assert result["blocker"]["message"] == "g1_worker_supervised_episode_incomplete"
