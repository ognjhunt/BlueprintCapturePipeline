from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_development_worker as worker
from blueprint_pipeline.native_g1_navigation_goal import seal_g1_navigation_goal_authority
from tests.test_native_g1_navigation_goal import _authority_plan


SCENE = "sha256:" + "a" * 64
INVENTORY = "sha256:" + "b" * 64
CANDIDATE = "humanoidarena_dp_g1_dex3_sonic"


def _request(tmp_path: Path) -> dict:
    bundle = tmp_path / "packet"
    bundle.mkdir()
    (bundle / "native_task_arena_scene_plan.v1.json").write_text(
        json.dumps({
            "plan_digest": SCENE, "task_kind": "rigid_pick_place",
            "robot": {"robot_id": "unitree_g1"},
        })
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
        "robot_id": "unitree_g1",
        "policy_role": "manipulation",
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


def test_scene_failure_is_sealed_before_isaac_close_can_exit(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", _preflight)
    monkeypatch.setattr(worker, "_verify_packet", _packet)

    def close_with_exit() -> None:
        preclose_path = tmp_path / "scene-failed" / worker.PRECLOSE_FILENAME
        preclose = json.loads(preclose_path.read_text(encoding="utf-8"))
        assert preclose["phase_reached"] == "scene_build"
        assert preclose["blocker"]["message"] == "scene_builder_refused"
        assert preclose["teardown"]["simulator"] == "close_requested"
        assert preclose["preclose_digest"] == canonical_digest(
            preclose, digest_field="preclose_digest"
        )
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_launch_scene", lambda **_kwargs: (
        SimpleNamespace(close=close_with_exit), {"status": "launched"}
    ))
    monkeypatch.setattr(worker, "_build_scene", lambda **_kwargs: (
        (_ for _ in ()).throw(ValueError("scene_builder_refused"))
    ))

    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "scene-failed"
    )
    assert result["status"] == "blocked"
    assert result["blocker"]["message"] == "scene_builder_refused"
    assert result["teardown"]["simulator"] == "closed"


def test_g1_scene_dependency_preflight_retains_import_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_task_arena_construction_worker as construction

    matrix = {
        "schema_version": "native_task_dependency_matrix.v1",
        "all_required_available": False,
        "blockers": ["native_task_arena_embodiment_scope_failed:unitree_g1"],
        "embodiment_scope": {
            "error_type": "ModuleNotFoundError",
            "error": "No module named 'isaaclab_arena_g1'",
            "traceback": "exact import stack",
        },
    }
    monkeypatch.setattr(
        construction, "preflight_native_dependency_matrix", lambda **_kwargs: matrix
    )
    path = tmp_path / "native_task_dependency_matrix.v1.json"
    with pytest.raises(ValueError, match="g1_worker_dependency_preflight_failed"):
        worker._build_scene(
            plan={}, bundle_root=tmp_path, device="cuda:0",
            dependency_receipt_path=path,
        )
    assert json.loads(path.read_text(encoding="utf-8")) == matrix


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


def test_usd_preflight_waits_for_isaac_runtime_and_keeps_rights_gate(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(worker, "_verify_packet", _packet)
    events: list[str] = []
    assets = {
        "candidate_id": CANDIDATE,
        "policy_role": "manipulation",
        "inventory_file_sha256": INVENTORY,
        "checkpoint_files": [],
        "policy_server_source": {},
        "sonic_provider_source": {},
        "sonic_encoder": {},
        "sonic_decoder": {},
    }
    monkeypatch.setattr(worker, "verify_g1_host_asset_identities", lambda **kwargs: (
        events.append("model_bytes_verified") or assets
    ))

    def preflight(**kwargs):
        events.append("usd_preflight")
        if events.count("usd_preflight") == 1:
            raise ValueError("g1_preflight_pxr_unavailable")
        return {**_preflight(**kwargs), **assets}

    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", preflight)
    original_rights = worker._rights_review

    def rights(value, *, preflight):
        result = original_rights(value, preflight=preflight)
        events.append("rights_verified")
        return result

    monkeypatch.setattr(worker, "_rights_review", rights)
    app = SimpleNamespace(close=lambda: events.append("simulator_closed"))

    def launch(**kwargs):
        assert events[-2:] == ["model_bytes_verified", "rights_verified"]
        events.append("simulator_launched")
        return app, {"status": "launched"}

    monkeypatch.setattr(worker, "_launch_scene", launch)
    monkeypatch.setattr(worker, "_build_scene", lambda **kwargs: (
        (_ for _ in ()).throw(AssertionError("scene must not build after test stop"))
    ))
    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "usd-runtime"
    )
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "scene_build"
    assert events == [
        "usd_preflight", "model_bytes_verified", "rights_verified", "simulator_launched",
        "usd_preflight", "simulator_closed",
    ]


def test_usd_runtime_fallback_rejects_rights_before_launch(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    request["rights_review"]["checkpoint_terms_reviewed"] = False
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    monkeypatch.setattr(worker, "_verify_packet", _packet)
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", lambda **kwargs: (
        (_ for _ in ()).throw(ValueError("g1_preflight_pxr_unavailable"))
    ))
    monkeypatch.setattr(worker, "verify_g1_host_asset_identities", lambda **kwargs: {
        "candidate_id": CANDIDATE,
        "policy_role": "manipulation",
        "inventory_file_sha256": INVENTORY,
    })
    monkeypatch.setattr(worker, "_launch_scene", lambda **kwargs: (
        (_ for _ in ()).throw(AssertionError("launched without rights"))
    ))
    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "usd-rights-blocked"
    )
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "rights_review"
    assert result["blocker"]["message"] == "g1_worker_rights_review_invalid"
    assert result["teardown"]["simulator"] == "not_started"


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


def test_output_cannot_mutate_sealed_packet(tmp_path: Path) -> None:
    request = _request(tmp_path)
    packet = Path(request["bundle_root"])
    try:
        worker.run_g1_development_worker(
            request=request, output_dir=packet / "episode-output"
        )
    except ValueError as exc:
        assert str(exc) == "g1_worker_output_directory_exists"
    else:
        raise AssertionError("output was allowed inside the sealed packet")
    assert not (packet / "episode-output").exists()


def test_navigation_candidate_requires_visible_goal_before_launch(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    candidate = "humanoidarena_dp_g1_dex3_sonic_vision_navi"
    request["candidate_id"] = candidate
    request["rights_review"]["candidate_id"] = candidate
    request["rights_review"]["rights_review_digest"] = canonical_digest(
        request["rights_review"], digest_field="rights_review_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    monkeypatch.setattr(worker, "_verify_packet", _packet)
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", lambda **kwargs: {
        **_preflight(**kwargs), "candidate_id": candidate,
        "policy_role": "movement_navigation",
    })

    def must_not_launch(**kwargs):
        raise AssertionError("launched without visible navigation goal")

    monkeypatch.setattr(worker, "_launch_scene", must_not_launch)
    result = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "no-navigation-goal"
    )
    assert result["status"] == "blocked"
    assert result["phase_reached"] == "navigation_goal_validation"
    assert result["blocker"]["message"] == "g1_navigation_goal_or_visible_marker_missing"


def test_navigation_candidate_requires_team_goal_authority_before_launch(
    tmp_path: Path, monkeypatch
) -> None:
    request = _request(tmp_path)
    plan = _authority_plan()
    candidate = "humanoidarena_dp_g1_dex3_sonic_vision_navi"
    Path(request["bundle_root"], "native_task_arena_scene_plan.v1.json").write_text(
        json.dumps(plan)
    )
    request["candidate_id"] = candidate
    request["rights_review"].update(
        candidate_id=candidate, scene_plan_digest=plan["plan_digest"]
    )
    request["rights_review"]["rights_review_digest"] = canonical_digest(
        request["rights_review"], digest_field="rights_review_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    monkeypatch.setattr(worker, "_verify_packet", lambda _root: {
        "arena_scene_plan_digest": plan["plan_digest"],
        "receipt_digest": "sha256:" + "f" * 64,
    })
    monkeypatch.setattr(worker, "preflight_g1_shared_scene_run", lambda **_kwargs: {
        "status": "staged_inputs_verified",
        "scene_plan_digest": plan["plan_digest"],
        "candidate_id": candidate,
        "inventory_file_sha256": INVENTORY,
        "robot_id": "unitree_g1",
        "policy_role": "movement_navigation",
    })

    def must_not_launch(**_kwargs):
        raise AssertionError("simulator launched without confirmed movement goal")

    monkeypatch.setattr(worker, "_launch_scene", must_not_launch)
    blocked = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "missing-goal-authority"
    )
    assert blocked["status"] == "blocked"
    assert blocked["phase_reached"] == "navigation_goal_authority_validation"
    assert blocked["teardown"] == {"environment": "not_started", "simulator": "not_started"}

    authority = seal_g1_navigation_goal_authority(
        plan=plan, confirmed_by_team_id="team-a", human_reviewer="owner-a"
    )
    request["navigation_goal_authority"] = authority
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    app = SimpleNamespace(close=lambda: None)
    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(worker, "_launch_scene", lambda **_kwargs: (app, {"status": "launched"}))
    monkeypatch.setattr(worker, "_build_scene", lambda **_kwargs: (
        SimpleNamespace(env=env, plan=plan), {"passed": True}
    ))
    monkeypatch.setattr(worker, "run_g1_supervised_built_scene_episode", lambda **_kwargs: {
        "status": "completed_development_only",
    })
    completed = worker.run_g1_development_worker(
        request=request, output_dir=tmp_path / "confirmed-goal-authority"
    )
    assert completed["status"] == "completed_development_only"
    assert completed["navigation_goal_authority_digest"] == authority["authority_digest"]
