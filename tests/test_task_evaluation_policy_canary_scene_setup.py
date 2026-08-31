from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import (
    task_evaluation_policy_canary_scene_setup as canary_setup_module,
)
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import (
    QUICK_FAMILY_COUNTS,
    _quick_cells,
    materialize_policy_canary_presubmission_setup,
    materialize_scene839873_policy_canary_setup,
    materialize_scene839873_policy_canary_setup_from_template,
    materialize_setup_preflight_decision,
)
from blueprint_pipeline.task_evaluation_policy_canary_setup import (
    validate_policy_canary_setup,
)
from blueprint_pipeline.task_evaluation_policy_run_contract import (
    validate_policy_run_setup,
)
from scripts.attach_internal_policy_canary_setup import (
    attach_internal_policy_canary_setup,
)
from tests.test_task_evaluation_policy_run_contract import _template as policy_template


COMMIT = "c" * 40
REVISION = "sha256:" + "9" * 64
LIVE_SCENE_REVISION = "sha256:9bd46da8b103a0ba5faa8a5c910652a3f35787c4bb0ebf8c5afa5cb2352d267d"
ACTIVATION = "sha256:" + "a" * 64
REQUEST = "sha256:" + "b" * 64


def test_cli_routes_both_public_materializers(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    parameters = tmp_path / "parameters.json"
    parameters.write_text(json.dumps({"marker": "bound"}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        canary_setup_module,
        "materialize_setup_preflight_decision",
        lambda **kwargs: calls.append(("preflight", kwargs)) or {"status": "ready"},
    )
    monkeypatch.setattr(
        canary_setup_module,
        "materialize_policy_canary_presubmission_setup",
        lambda **kwargs: calls.append(("presubmission", kwargs))
        or {"status": "selectable"},
    )

    assert canary_setup_module.main(
        ["preflight", "--parameters", str(parameters)]
    ) == 0
    assert canary_setup_module.main(
        ["presubmission", "--parameters", str(parameters)]
    ) == 0

    assert calls == [
        ("preflight", {"marker": "bound"}),
        ("presubmission", {"marker": "bound"}),
    ]
    assert '"status": "ready"' in capsys.readouterr().out


def _write(path: Path, value: dict) -> Path:
    write_json(path, value)
    return path


def _inputs(tmp_path: Path) -> dict[str, Path]:
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "source_commit": COMMIT,
        "request_digest": REQUEST,
    }
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "source_commit": COMMIT,
        "profile_id": "scene839873-current",
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    scenario = {
        "context_kind": "evaluation_cell",
        "cell_id": "configured_scene_canonical",
        "seed": 839873104,
        "parameter_bindings": [],
        "parameter_applications": [],
    }
    scene = {
        "schema_version": "native_task_arena_scene_plan.v1",
        "scene_id": "interiorgs-839873",
        "task_id": "scene-839873-mug-planar-push",
        "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "franka_panda"},
        "scenario": scenario,
        "task_spec": {
            "prompt": "Move the configured rigid object by planar push.",
            "maximum_action_steps": 240,
            "manipulation_strategy": "planar_push",
        },
        "plan_digest": "",
    }
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    packet = {
        "schema_version": "native_task_arena_packet_receipt.v1",
        "scene_id": scene["scene_id"],
        "task_id": scene["task_id"],
        "arena_scene_plan_digest": scene["plan_digest"],
    }
    runtime = {
        "schema_version": "native_task_runtime_source_packet.v1",
        "packet_sha256": "sha256:" + "d" * 64,
        "packet_size_bytes": 4_287_162_924,
        "redistribution_permitted": True,
    }
    template = policy_template()
    configured_preparation = {
        field: template[field]
        for field in (
            "scene",
            "construction",
            "robot",
            "controller",
            "task",
            "sensors",
            "runtime",
            "execution_adapter",
            "spend",
        )
    }
    progression = {
        "configured_scene_revision_digest": REVISION,
        "episode_preparation_request": configured_preparation,
        "episode_preparation_request_digest": canonical_digest(configured_preparation),
    }
    return {
        "request": _write(tmp_path / "request.json", request),
        "profile": _write(tmp_path / "profile.json", profile),
        "scene": _write(tmp_path / "scene.json", scene),
        "packet": _write(tmp_path / "packet.json", packet),
        "runtime": _write(tmp_path / "runtime.json", runtime),
        "progression": _write(tmp_path / "progression.json", progression),
    }


def _kwargs(tmp_path: Path) -> dict:
    inputs = _inputs(tmp_path)
    root = Path(__file__).resolve().parents[1]
    return {
        "source_commit": COMMIT,
        "configured_source_launch_id": "scene839873-configured-source",
        "scene_revision_digest": REVISION,
        "activation_digest": ACTIVATION,
        "capture_session_id": "capture-839873",
        "intake_id": "intake-839873",
        "request_digest": REQUEST,
        "launch_request_path": inputs["request"],
        "launch_profile_path": inputs["profile"],
        "configured_progression_path": inputs["progression"],
        "scene_plan_path": inputs["scene"],
        "packet_receipt_path": inputs["packet"],
        "runtime_source_receipt_path": inputs["runtime"],
        "historical_policy_readiness_path": root
        / "docs/arm_decision_proof_v1/manifests/adp009d_scene_840920_policy_readiness.v1.json",
        "pi05_checkpoint_inventory_path": root
        / "docs/experiments/policy_ranking_thesis_20260726/openpi_polaris_checkpoint_inventory.json",
        "output_dir": tmp_path / "output",
    }


def test_setup_binds_current_scene_pair_quick10_and_unqualified_boundary(
    tmp_path: Path,
) -> None:
    setup = materialize_scene839873_policy_canary_setup(**_kwargs(tmp_path))

    assert setup["status"] == "verified_runnable"
    assert setup["candidate_ids"] == ["pi05_droid", "groot_n17_droid"]
    assert setup["quick_10"]["learned_policy_rollout_count"] == 20
    assert {
        family: sum(cell["family"] == family for cell in setup["quick_10"]["cells"])
        for family in QUICK_FAMILY_COUNTS
    } == QUICK_FAMILY_COUNTS
    assert setup["historical_runtime_smoke"]["current_runtime_proof"] is False
    assert setup["scene_promotion_authorized"] is False
    assert setup["official_ranking_authorized"] is False
    for record in setup["records"].values():
        assert Path(record["path"]).is_file()
        assert record["sha256"].startswith("sha256:")


def test_missing_activation_and_terminal_lineage_emit_typed_blockers(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    kwargs.update(activation_digest="", capture_session_id="", intake_id="")
    decision = materialize_setup_preflight_decision(
        output_path=tmp_path / "decision.json", **kwargs
    )

    assert decision["status"] == "blocked"
    assert decision["blockers"] == [
        "policy_canary_activation_digest_invalid",
        "policy_canary_capture_session_id_missing",
        "policy_canary_intake_id_missing",
    ]
    assert not (tmp_path / "output/task_evaluation_policy_canary_execution_setup.v1.json").exists()


def test_checkpoint_registry_drift_fails_before_specs_are_written(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    readiness_path = Path(kwargs["historical_policy_readiness_path"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["candidates"][0]["checkpoint"]["inventory_digest"] = "sha256:" + "f" * 64
    readiness["readiness_digest"] = canonical_digest(readiness, digest_field="readiness_digest")
    changed = _write(tmp_path / "changed-readiness.json", readiness)
    kwargs["historical_policy_readiness_path"] = changed
    decision = materialize_setup_preflight_decision(
        output_path=tmp_path / "decision.json", **kwargs
    )

    assert decision["status"] == "blocked"
    assert decision["blockers"] == ["policy_canary_pi05_droid_registry_or_rights_invalid"]


def test_presubmission_setup_is_activation_independent_and_profile_ready(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    for field in ("activation_digest", "capture_session_id", "intake_id"):
        kwargs.pop(field)
    kwargs["profile_id"] = "scene839873-internal-policy-canary-c412"
    kwargs["configured_offering_configuration_run_id"] = "scene839873-configuration"
    kwargs["offering_digest"] = "sha256:" + "f" * 64
    kwargs["policy_controller_configuration"] = {
        "uri": "s3://blueprint/policy-canary/controller.json",
        "digest": "sha256:" + "2" * 64,
        "size_bytes": 512,
    }
    kwargs["model_rights"] = {
        "uri": "s3://blueprint/policy-canary/model-rights.json",
        "digest": "sha256:" + "3" * 64,
        "size_bytes": 768,
    }
    emitted = materialize_policy_canary_presubmission_setup(**kwargs)
    setup = emitted["setup"]
    wrapper = emitted["profile_materialization_input"]

    assert setup["schema_version"] == "task_evaluation_policy_canary_setup.v1"
    assert validate_policy_canary_setup(setup) == setup
    assert setup["source_launch_id"] == "scene839873-configured-source"
    assert setup["offering_digest"] == "sha256:" + "f" * 64
    assert setup["episode_presets"][0]["episodes_per_policy"] == 10
    assert len(setup["episode_presets"][0]["matrix"]["cells"]) == 10
    assert setup["episode_presets"][1]["availability"] == "coming_later"
    assert setup["episode_presets"][2]["availability"] == "coming_later"
    assert "activation_digest" not in setup
    assert "capture_session_id" not in setup
    assert "intake_id" not in setup
    assert wrapper["profile_id"] == "scene839873-internal-policy-canary-c412"
    assert wrapper["configured_source_launch_id"] != wrapper["profile_id"]
    assert wrapper["internal_policy_canary_setup"] == setup
    plan = wrapper["internal_policy_canary_execution_plan"]
    assert validate_policy_run_setup(plan["legacy_policy_run_setup"])
    assert (
        plan["preparation_template"]["controller"]["configuration"]
        == plan["policy_controller_configuration"]
    )
    assert (
        plan["preparation_template"]["controller"]["model_or_asset_rights"] == plan["model_rights"]
    )
    assert len(plan["resolved_scenarios"]) == 10
    assert all(isinstance(row["seed"], int) for row in plan["resolved_scenarios"])
    assert plan["configured_offering_configuration_run_id"] == ("scene839873-configuration")
    assert plan["lineage_aliases"] == {
        "capture_session_id": "scene839873-configured-source",
        "capture_session_id_semantics": (
            "configured_scene_offering_source_launch_id_no_capture_upload_session"
        ),
        "intake_id": "scene839873-configuration",
        "intake_id_semantics": "configured_scene_offering_configuration_run_id",
    }
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    assert Path(emitted["setup_path"]).is_file()
    assert Path(emitted["profile_materialization_input_path"]).is_file()
    assert Path(emitted["execution_setup_template_path"]).is_file()
    assert emitted["execution_setup_template"]["provider_mutation_performed"] is False
    shape = json.loads(
        (
            Path(__file__).resolve().parent
            / "fixtures/webapp_internal_policy_canary_setup_shape.v1.json"
        ).read_text(encoding="utf-8")
    )
    robot = setup["robot_presets"][0]
    assert sorted(setup) == shape["top_level_keys"]
    assert sorted(robot) == shape["robot_keys"]
    assert sorted(robot["policy_candidates"][0]) == shape["candidate_keys"]
    assert sorted(setup["episode_presets"][0]) == shape["preset_keys"]
    assert sorted(wrapper) == shape["profile_materialization_input_keys"]
    assert sorted(plan) == shape["execution_plan_keys"]
    attached = attach_internal_policy_canary_setup(
        profile={
            "profile_id": wrapper["profile_id"],
            "configured_source_launch_id": setup["source_launch_id"],
            "profile_digest": "sha256:" + "1" * 64,
        },
        setup=setup,
        profile_validator=lambda _value: [],
    )
    assert attached["internal_policy_canary_setup"] == setup


def test_post_activation_template_separates_configured_and_canary_requests(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    for field in ("activation_digest", "capture_session_id", "intake_id"):
        kwargs.pop(field)
    kwargs["profile_id"] = "scene839873-internal-policy-canary-current"
    kwargs["configured_offering_configuration_run_id"] = "scene839873-configuration"
    kwargs["offering_digest"] = "sha256:" + "f" * 64
    kwargs["policy_controller_configuration"] = {
        "uri": "s3://blueprint/policy-canary/controller.json",
        "digest": "sha256:" + "2" * 64,
        "size_bytes": 512,
    }
    kwargs["model_rights"] = {
        "uri": "s3://blueprint/policy-canary/model-rights.json",
        "digest": "sha256:" + "3" * 64,
        "size_bytes": 768,
    }
    emitted = materialize_policy_canary_presubmission_setup(**kwargs)
    activation_result = {
        "schema_version": "task_evaluation_launch_activation_result.v1",
        "status": "policy_campaign_queue_materialized_no_execution",
        "policy_campaign_activation_digest": ACTIVATION,
    }
    activation_path = _write(tmp_path / "activation-result.json", activation_result)
    envelope = {
        "schema_version": "task_evaluation_policy_canary_dispatch_envelope.v1",
        "activation_id": "activation-839873",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "source_commit": COMMIT,
        "activation_result": {
            "path": str(activation_path),
            "size_bytes": activation_path.stat().st_size,
            "sha256": "sha256:"
            + __import__("hashlib").sha256(activation_path.read_bytes()).hexdigest(),
        },
        "capture_session_id": "capture-new-canary",
        "intake_id": "intake-new-canary",
        "request_digest": "sha256:" + "e" * 64,
        "maximum_provider_allocations": 1,
        "retry_cap": 0,
        "automatic_retry_authorized": False,
        "provider_mutation_performed": False,
        "paid_execution_requested": False,
        "envelope_digest": "",
    }
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")

    setup = materialize_scene839873_policy_canary_setup_from_template(
        template_path=emitted["execution_setup_template_path"],
        activation_envelope=envelope,
        output_dir=tmp_path / "post-activation",
    )

    assert setup["configured_request_digest"] == REQUEST
    assert setup["request_digest"] == "sha256:" + "e" * 64
    assert setup["capture_session_id"] == "capture-new-canary"
    assert setup["activation_digest"] == ACTIVATION


def test_controller_and_rights_source_artifacts_are_exact_and_self_digesting() -> None:
    root = Path(__file__).resolve().parents[1]
    controller = json.loads(
        (
            root
            / "docs/arm_decision_proof_v1/manifests/scene839873_policy_canary_controller_configuration.v1.json"
        ).read_text(encoding="utf-8")
    )
    rights = json.loads(
        (
            root
            / "docs/arm_decision_proof_v1/manifests/scene839873_policy_canary_model_rights.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert controller["configuration_digest"] == canonical_digest(
        controller, digest_field="configuration_digest"
    )
    assert rights["rights_digest"] == canonical_digest(rights, digest_field="rights_digest")
    assert controller["candidate_order"] == ["pi05_droid", "groot_n17_droid"]
    assert [row["port"] for row in controller["candidates"]] == [8000, 5555]
    assert controller["maximum_provider_allocations"] == 1
    assert controller["retry_cap"] == 0
    assert controller["ranking_permitted"] is False
    quick = _quick_cells(LIVE_SCENE_REVISION)
    assert controller["quick_10"]["matrix_digest"] == canonical_digest({"ordered_cells": quick})
    assert [
        (row["cell_id"], row["seed"], row["cell_digest"]) for row in controller["quick_10"]["cells"]
    ] == [(row["cell_id"], row["seed"], row["cell_spec_digest"]) for row in quick]
    assert [row["candidate_id"] for row in rights["candidates"]] == [
        "pi05_droid",
        "groot_n17_droid",
    ]
    assert rights["historical_runtime_smoke"]["input_evidence_only"] is True
    assert rights["historical_runtime_smoke"]["current_scene_runtime_proof"] is False
    for module in rights["blueprint_adapter_code"]["modules"]:
        path = root / module["path"]
        observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == module["sha256"]
