from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import cosmos3_wam_command_adapter as adapter
from blueprint_pipeline import wam_backend_strategy as backend_strategy


ADAPTER_ENV_VARS = (
    "BLUEPRINT_COSMOS3_WAM_SOURCE_ROOT",
    "BLUEPRINT_COSMOS3_NANO_SOURCE_ROOT",
    "BLUEPRINT_COSMOS3_SOURCE_ROOT",
    "BLUEPRINT_COSMOS3_WAM_CHECKPOINT",
    "BLUEPRINT_COSMOS3_NANO_CHECKPOINT",
    "BLUEPRINT_WAM_MODEL_CHECKPOINT",
    "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL",
    "BLUEPRINT_COSMOS3_WAM_ENTRYPOINT",
    "BLUEPRINT_WAM_ROLLOUT_INPUT",
    "BLUEPRINT_WAM_ROLLOUT_OUTPUT",
    "BLUEPRINT_WAM_MODEL_CANDIDATE",
)


def _clear_adapter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ADAPTER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "cosmos3-source"
    (source_root / "examples").mkdir(parents=True)
    (source_root / "examples" / "action_conditioned.py").write_text(
        "# cosmos3\n", encoding="utf-8"
    )
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "cosmos3-nano-oss"\n', encoding="utf-8"
    )
    return source_root


def _checkpoint(tmp_path: Path, *, base_model: str | None) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    if base_model is not None:
        _write_json(checkpoint / "config.json", {"base_model": base_model})
    return checkpoint


def _output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output_path = tmp_path / "out" / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))
    return output_path


def test_adapter_blocks_and_writes_typed_payload_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    output_path = _output_path(tmp_path, monkeypatch)

    payload = adapter.run([])

    assert payload["schema_version"] == "cosmos3_wam_command_adapter.v1"
    assert payload["status"] == "blocked"
    assert "blocked_missing_cosmos3_source_root" in payload["blockers"]
    assert "blocked_missing_cosmos3_checkpoint" in payload["blockers"]
    assert payload["learned_wam_model_ran"] is False
    assert payload["fresh_model_run_claimed"] is False
    assert payload["evaluation_substrate"] == "cosmos3_wam"
    assert payload["raw_credentials_written_to_artifacts"] is False
    assert payload["run_gates"]["auto_run_allowed_without_gate"] is False
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "blocked"


def test_adapter_fails_closed_on_wrong_family_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    _output_path(tmp_path, monkeypatch)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model="Cosmos-Predict2.5-2B")

    def _must_not_run(**kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("wrong-family checkpoint must never launch the model")

    monkeypatch.setattr(adapter, "_run_cosmos3", _must_not_run)
    monkeypatch.setattr(adapter, "_run_import_probe", _must_not_run)

    payload = adapter.run(
        ["--source-root", str(source_root), "--checkpoint", str(checkpoint)]
    )

    assert payload["status"] == "blocked"
    assert payload["blockers"] == [
        "blocked_wrong_model_family_checkpoint_for_cosmos3_wam"
    ]
    assert payload["learned_wam_model_ran"] is False
    probe = payload["checkpoint_identity_probe"]
    assert probe["wrong_model_family_detected"] is True
    assert "Cosmos-Predict2.5-2B" in probe["declared_identity_values"]


def test_adapter_fails_closed_on_unverified_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    _output_path(tmp_path, monkeypatch)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model=None)

    payload = adapter.run(
        ["--source-root", str(source_root), "--checkpoint", str(checkpoint)]
    )

    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["blocked_cosmos3_checkpoint_identity_unverified"]
    assert payload["learned_wam_model_ran"] is False


def test_adapter_respects_local_model_run_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    _output_path(tmp_path, monkeypatch)
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model="nvidia/Cosmos3-Nano")
    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "completed", "blockers": []},
    )

    def _must_not_run(**kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("model must never run without the explicit gate")

    monkeypatch.setattr(adapter, "_run_cosmos3", _must_not_run)

    payload = adapter.run(
        ["--source-root", str(source_root), "--checkpoint", str(checkpoint)]
    )

    assert payload["status"] == "blocked"
    assert payload["blockers"] == [
        "blocked_BLUEPRINT_ALLOW_LOCAL_WAM_MODEL_not_enabled"
    ]
    assert payload["run_gates"]["local_model_gate_enabled"] is False
    assert payload["learned_wam_model_ran"] is False


def test_adapter_emits_trusted_schema_with_verified_identity_and_declared_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    output_path = _output_path(tmp_path, monkeypatch)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model="Cosmos3-Nano")
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"task_prompts": []})
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))

    save_root = tmp_path / "generated"
    save_root.mkdir()
    package_manifest = {
        "schema_version": "blueprint_cosmos_rollout_input_package.v1",
        "save_root": str(save_root),
        "inference_params_path": str(tmp_path / "params.json"),
        "source_review_video_path": str(tmp_path / "review.mp4"),
        "source_camera": "robot_pov",
        "scenario_eval_run_id": "run-1",
        "task_id": "task-1",
        "spawn_id": "spawn-1",
    }
    monkeypatch.setattr(
        adapter,
        "_materialize_cosmos_input_package",
        lambda **kwargs: dict(package_manifest),
    )
    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "completed", "blockers": []},
    )

    def _fake_run_cosmos3(**kwargs: Any) -> dict[str, Any]:
        (save_root / "rollout_0001.mp4").write_bytes(b"generated mp4")
        return {
            "schema_version": "cosmos3_subprocess_result.v1",
            "status": "completed",
            "returncode": 0,
            "blockers": [],
        }

    monkeypatch.setattr(adapter, "_run_cosmos3", _fake_run_cosmos3)
    monkeypatch.setattr(
        adapter,
        "validate_generated_mp4_for_review",
        lambda path: {"status": "completed", "blockers": []},
    )

    payload = adapter.run(
        ["--source-root", str(source_root), "--checkpoint", str(checkpoint)]
    )

    assert payload["schema_version"] == "cosmos3_wam_command_adapter.v1"
    assert payload["status"] == "completed"
    assert payload["base_model"] == "Cosmos3-Nano"
    assert payload["learned_wam_model_ran"] is True
    assert payload["fresh_model_run_claimed"] is True
    assert payload["rollouts"][0]["base_model"] == "Cosmos3-Nano"
    assert payload["rollouts"][0]["success_label_source"] == (
        "generated_video_requires_review"
    )
    truth = payload["truth_boundary"]
    assert truth["checkpoint_identity_verified_as_cosmos3_nano"] is True
    assert truth["sc3_recipe_metadata_is_declared_config_not_proof"] is True
    assert truth["generated_world_rank_fidelity_result_proven"] is False
    recipe = payload["sc3_recipe_declared_config"]
    assert recipe["training_mixture"] == {
        "forward_dynamics": 0.8,
        "cross_view": 0.1,
        "inverse_dynamics": 0.1,
    }
    assert recipe["horizon_decoupling"] == {
        "predict_horizon_frames": 24,
        "execute_horizon_frames": 16,
    }
    assert recipe["claim_boundary"]["recipe_metadata_is_operator_declared_config"] is True
    assert recipe["claim_boundary"]["recipe_metadata_is_execution_or_training_proof"] is False
    assert payload["raw_credentials_written_to_artifacts"] is False
    assert payload["secret_hashes_written_to_artifacts"] is False
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "completed"

    from blueprint_pipeline.oscar_cosmos_wam_evaluator import (
        TRUSTED_WAM_MODEL_PAYLOAD_SCHEMAS,
    )

    assert "cosmos3_wam_command_adapter.v1" in TRUSTED_WAM_MODEL_PAYLOAD_SCHEMAS


def test_rollout_payload_never_claims_learned_run_without_verified_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_root = tmp_path / "generated"
    save_root.mkdir()
    (save_root / "rollout_0001.mp4").write_bytes(b"generated mp4")
    monkeypatch.setattr(
        adapter,
        "validate_generated_mp4_for_review",
        lambda path: {"status": "completed", "blockers": []},
    )

    payload = adapter._rollout_payload(
        package_manifest={"save_root": str(save_root)},
        checkpoint=tmp_path / "checkpoint",
        source_root=tmp_path,
        subprocess_detail={"status": "completed", "blockers": []},
        model="cosmos3/nano/action-cond",
        checkpoint_identity={"checkpoint_identity_verified": False},
        source_identity={"source_identity_verified": True},
    )

    assert payload["status"] == "completed"
    assert payload["fresh_model_command_executed_this_invocation"] is True
    assert payload["learned_wam_model_ran"] is False
    assert payload["fresh_model_run_claimed"] is False


def test_checkpoint_identity_probe_recognizes_cosmos3_and_wrong_families(
    tmp_path: Path,
) -> None:
    verified = _checkpoint(tmp_path / "good", base_model="nvidia/Cosmos3-Nano")
    probe = adapter.checkpoint_identity_probe(verified)
    assert probe["checkpoint_identity_verified"] is True
    assert probe["wrong_model_family_detected"] is False

    wrong = _checkpoint(tmp_path / "bad", base_model="Cosmos3-Super")
    probe = adapter.checkpoint_identity_probe(wrong)
    assert probe["checkpoint_identity_verified"] is False
    assert probe["wrong_model_family_detected"] is True

    oscar = _checkpoint(tmp_path / "oscar", base_model="OSCAR-2B")
    probe = adapter.checkpoint_identity_probe(oscar)
    assert probe["wrong_model_family_detected"] is True

    weights_file = tmp_path / "flat" / "weights.bin"
    weights_file.parent.mkdir(parents=True)
    weights_file.write_bytes(b"weights")
    _write_json(weights_file.parent / "config.json", {"model_name": "Cosmos3-Nano"})
    probe = adapter.checkpoint_identity_probe(weights_file)
    assert probe["checkpoint_identity_verified"] is True


def test_cosmos3_strategy_state_is_derived_from_machine_checked_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    monkeypatch.delenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_COSMOS3_CALIBRATION_ANCHORS_PATH", raising=False)

    row = backend_strategy.get_wam_backend_strategy("cosmos3_wam")
    assert row["aspirational"] is True
    assert row["preferred_candidate_state"] == "aspirational"
    preconditions = row["preconditions"]
    assert preconditions["state_derived_from_machine_checks"] is True
    assert preconditions["state_asserted_manually"] is False
    assert preconditions["checks"]["adapter_module_present"]["passed"] is True
    assert preconditions["checks"]["consistency_scorer_available"]["passed"] is False
    assert preconditions["checks"]["calibration_anchors_present"]["passed"] is False

    manifest = backend_strategy.build_wam_backend_strategy_manifest(
        generated_at="now"
    )
    assert manifest["cosmos3_wam_aspirational"] is True
    assert manifest["cosmos3_wam_preferred_candidate_state"] == "aspirational"
    assert manifest["claim_boundary"][
        "cosmos3_preferred_candidate_state_is_machine_derived_not_asserted"
    ] is True
    by_id = {r["backend_id"]: r for r in manifest["backend_strategies"]}
    assert by_id["cosmos3_wam"]["aspirational"] is True

    anchors_path = tmp_path / "accepted_anchors.json"
    _write_json(
        anchors_path,
        {
            "schema_version": "accepted_real_world_anchor.v1",
            "anchors": [
                {
                    "scenario_eval_run_id": "run-1",
                    "policy_id": "policy-1",
                    "task_id": "task-1",
                    "scenario_variation_instance_id": "var-1",
                    "actual_success": True,
                }
            ],
        },
    )
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", "/opt/cosmos3-adapter")
    derived = backend_strategy.evaluate_cosmos3_wam_preconditions(
        calibration_anchors_path=anchors_path,
        consistency_scorer_available=True,
    )
    assert derived["preconditions_met"] is True
    assert derived["aspirational"] is False
    assert derived["preferred_candidate_state"] == "preferred_configured_candidate"
