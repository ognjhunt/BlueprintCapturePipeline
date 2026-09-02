from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import (
    task_evaluation_policy_canary_scene_setup as canary_setup_module,
)
from blueprint_pipeline.task_evaluation_policy_canary_scene_setup import (
    QUICK_FAMILY_COUNTS,
    _quick_cells,
    _seal_presubmission_service_access,
    materialize_policy_canary_presubmission_setup,
    materialize_scene839873_policy_canary_setup,
    materialize_scene839873_policy_canary_setup_from_template,
    materialize_setup_preflight_decision,
)
from blueprint_pipeline.task_evaluation_policy_canary_setup import (
    validate_policy_canary_setup,
)
from blueprint_pipeline.task_evaluation_policy_run_contract import (
    compile_policy_run_configuration,
    validate_policy_run_setup,
)
from scripts.attach_internal_policy_canary_setup import (
    attach_internal_policy_canary_setup,
)
from tests.test_task_evaluation_policy_run_contract import _template as policy_template


COMMIT = "c" * 40
REVISION = "sha256:" + "9" * 64
LIVE_SCENE_REVISION = "sha256:20d8e011291d8ba31d20a502ef4800e9260e83a686df7770c331b94af3b6b3f4"
ACTIVATION = "sha256:" + "a" * 64
REQUEST = "sha256:" + "b" * 64


def test_cli_routes_both_public_materializers(tmp_path: Path, monkeypatch, capsys) -> None:
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
        lambda **kwargs: calls.append(("presubmission", kwargs)) or {"status": "selectable"},
    )

    assert canary_setup_module.main(["preflight", "--parameters", str(parameters)]) == 0
    assert canary_setup_module.main(["presubmission", "--parameters", str(parameters)]) == 0

    assert calls == [
        ("preflight", {"marker": "bound"}),
        ("presubmission", {"marker": "bound"}),
    ]
    assert '"status": "ready"' in capsys.readouterr().out


def _write(path: Path, value: dict) -> Path:
    write_json(path, value)
    return path


def _root_materialization_stat_reader(
    *,
    destination: Path,
    artifacts: dict[str, Path],
    service_uid: int,
    installed_groups: dict[Path, int],
):
    owned = {destination.resolve(), *(path.resolve() for path in artifacts.values())}

    def read(path: Path):
        resolved = Path(path).resolve()
        observed = os.stat(resolved)
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_uid=0 if resolved in owned else service_uid,
            st_gid=installed_groups.get(resolved, 0),
        )

    return read


def test_root_presubmission_materialization_is_sealed_for_blueprint(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "root-authored"
    destination.mkdir(mode=0o700)
    artifacts = {
        "setup": _write(destination / "setup.json", {"role": "setup"}),
        "profile_materialization_input": _write(
            destination / "wrapper.json", {"role": "wrapper"}
        ),
        "execution_setup_template": _write(
            destination / "template.json", {"role": "template"}
        ),
    }
    for path in artifacts.values():
        path.chmod(0o600)
    account = {
        "user": "blueprint",
        "uid": 4242,
        "group": "blueprint",
        "gid": 4343,
        "group_ids": [4343],
    }
    installed_groups: dict[Path, int] = {}

    def install_group(path: Path, uid: int, gid: int) -> None:
        assert uid == -1
        installed_groups[Path(path).resolve()] = gid

    stat_reader = _root_materialization_stat_reader(
        destination=destination,
        artifacts=artifacts,
        service_uid=account["uid"],
        installed_groups=installed_groups,
    )

    receipt = _seal_presubmission_service_access(
        destination=destination,
        artifacts=artifacts,
        account_resolver=lambda: account,
        chown=install_group,
        stat_reader=stat_reader,
        digest_reader=lambda path, *, account: "sha256:"
        + hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )

    assert receipt["status"] == "readable_by_service_account"
    assert receipt["service_user"] == "blueprint"
    assert {row["role"] for row in receipt["verified_roles"]} == set(artifacts)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o750
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o440 for path in artifacts.values())
    assert installed_groups == {
        destination.resolve(): account["gid"],
        **{path.resolve(): account["gid"] for path in artifacts.values()},
    }


def test_presubmission_materialization_fails_closed_when_wrapper_stays_unreadable(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "root-authored"
    destination.mkdir(mode=0o700)
    artifacts = {
        "setup": _write(destination / "setup.json", {"role": "setup"}),
        "profile_materialization_input": _write(
            destination / "wrapper.json", {"role": "wrapper"}
        ),
        "execution_setup_template": _write(
            destination / "template.json", {"role": "template"}
        ),
    }
    account = {
        "user": "blueprint",
        "uid": 4242,
        "group": "blueprint",
        "gid": 4343,
        "group_ids": [4343],
    }
    installed_groups: dict[Path, int] = {}
    unreadable = artifacts["profile_materialization_input"].resolve()

    def partially_install_group(path: Path, uid: int, gid: int) -> None:
        assert uid == -1
        resolved = Path(path).resolve()
        if resolved != unreadable:
            installed_groups[resolved] = gid

    stat_reader = _root_materialization_stat_reader(
        destination=destination,
        artifacts=artifacts,
        service_uid=account["uid"],
        installed_groups=installed_groups,
    )

    with pytest.raises(
        canary_setup_module.PolicyCanarySetupError,
        match=(
            "^policy_canary_materialization_service_access_denied:"
            "profile_materialization_input$"
        ),
    ) as denied:
        _seal_presubmission_service_access(
            destination=destination,
            artifacts=artifacts,
            account_resolver=lambda: account,
            chown=partially_install_group,
            stat_reader=stat_reader,
            digest_reader=lambda path, *, account: "sha256:"
            + hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        )

    assert str(unreadable) not in str(denied.value)


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
    def activation_ref(name: str, character: str) -> dict:
        return {
            "uri": f"s3://blueprint/policy-canary/activation/{name}.json",
            "digest": "sha256:" + character * 64,
            "size_bytes": 256,
        }
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
        "activation_release_window_template": activation_ref("release-window-template", "1"),
        "activation_lineage": {
            "kind": "predecessor",
            "prior_authority": activation_ref("prior-authority", "2"),
            "prior_result": activation_ref("prior-result", "3"),
            "prior_launch_receipt": activation_ref("prior-launch-receipt", "4"),
            "prior_webapp_sync": activation_ref("prior-webapp-sync", "5"),
            "prior_provider_zero": activation_ref("prior-provider-zero", "6"),
            "prior_spend_reconciliation": activation_ref("prior-spend", "7"),
            "construction_result": activation_ref("construction-result", "8"),
        },
        "activation_authorization": {
            "reference": "automatic policy-canary activation",
            "authorized_by": "blueprint-policy-lead",
            "profile_revision": "policy-canary-v1",
            "valid_for_seconds": 3600,
        },
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


def test_setup_keeps_execution_commit_distinct_from_configured_scene_commit(
    tmp_path: Path,
) -> None:
    kwargs = _kwargs(tmp_path)
    configured_commit = "d" * 40
    launch_request_path = Path(kwargs["launch_request_path"])
    launch_request = json.loads(launch_request_path.read_text(encoding="utf-8"))
    launch_request["source_commit"] = configured_commit
    _write(launch_request_path, launch_request)
    launch_profile_path = Path(kwargs["launch_profile_path"])
    launch_profile = json.loads(launch_profile_path.read_text(encoding="utf-8"))
    launch_profile["source_commit"] = configured_commit
    launch_profile["profile_digest"] = canonical_digest(
        launch_profile, digest_field="profile_digest"
    )
    _write(launch_profile_path, launch_profile)
    kwargs["configured_source_commit"] = configured_commit

    setup = materialize_scene839873_policy_canary_setup(**kwargs)

    assert setup["source_commit"] == COMMIT
    assert setup["records"]["pi05_execution_spec"]["path"]


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


@pytest.mark.parametrize(
    "input_name,role",
    (
        ("request", "launch_request"),
        ("profile", "launch_profile"),
        ("progression", "configured_progression"),
        ("scene", "scene_plan"),
        ("packet", "packet_receipt"),
    ),
)
def test_known_corrupt_scene_digest_fails_before_launch_materialization(
    tmp_path: Path,
    input_name: str,
    role: str,
) -> None:
    kwargs = _kwargs(tmp_path)
    field_by_input = {
        "request": "launch_request_path",
        "profile": "launch_profile_path",
        "progression": "configured_progression_path",
        "scene": "scene_plan_path",
        "packet": "packet_receipt_path",
    }
    source = Path(kwargs[field_by_input[input_name]])
    value = json.loads(source.read_text(encoding="utf-8"))
    value["superseded_appearance_digest"] = "sha256:d6c3cd3e" + "0" * 56
    _write(source, value)

    decision = materialize_setup_preflight_decision(
        output_path=tmp_path / "decision.json", **kwargs
    )

    assert decision["status"] == "blocked"
    assert (
        f"policy_canary_forbidden_scene_digest_present:{role}:d6c3cd3e"
        in decision["blockers"]
    )
    assert not (
        tmp_path / "output/task_evaluation_policy_canary_execution_setup.v1.json"
    ).exists()


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
    kwargs["native_controller_configuration"] = {
        "uri": "s3://blueprint/policy-canary/native-controller.json",
        "digest": "sha256:" + "4" * 64,
        "size_bytes": 640,
    }
    kwargs["runtime_source_bundle"] = {
        "uri": "s3://blueprint/policy-canary/runtime-source.zip",
        "digest": "sha256:" + "5" * 64,
        "size_bytes": 4_272_421_731,
    }
    kwargs["runtime_source_implementation_commit"] = "b" * 40
    kwargs["model_rights"] = {
        "uri": "s3://blueprint/policy-canary/model-rights.json",
        "digest": "sha256:" + "3" * 64,
        "size_bytes": 768,
    }
    kwargs["policy_observation_setup"] = {
        "schema_version": "task_evaluation_policy_observation_setup.v1",
        "appearance_asset": {
            "uri": "s3://blueprint/policy-canary/appearance.usdc",
            "digest": "sha256:" + "6" * 64,
            "size_bytes": 239_852_301,
        },
        "appearance_authoring_receipt": {
            "uri": "s3://blueprint/policy-canary/appearance-receipt.json",
            "digest": "sha256:" + "7" * 64,
            "size_bytes": 2_048,
        },
        "wrist_camera_mount_registry": {
            "uri": "s3://blueprint/policy-canary/camera-registry.json",
            "digest": "sha256:" + "8" * 64,
            "size_bytes": 4_096,
        },
        "fresh_native_mount_sweep_required": True,
        "policy_master_resolution_wh": [640, 360],
        "overview_review_resolution_wh": [1280, 720],
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
    assert wrapper["configured_base_profile_id"] == "scene839873-current"
    assert wrapper["configured_base_profile_digest"].startswith("sha256:")
    assert wrapper["configured_base_profile_id"] != wrapper[
        "configured_source_launch_id"
    ]
    assert wrapper["internal_policy_canary_execution_plan"][
        "preparation_template"
    ]["execution_adapter"]["runtime_source_implementation_commit"] == "b" * 40
    assert wrapper["configured_source_launch_id"] != wrapper["profile_id"]
    assert wrapper["internal_policy_canary_setup"] == setup
    plan = wrapper["internal_policy_canary_execution_plan"]
    assert validate_policy_run_setup(plan["legacy_policy_run_setup"])
    assert (
        plan["preparation_template"]["controller"]["configuration"]
        == kwargs["native_controller_configuration"]
    )
    assert (
        plan["preparation_template"]["controller"]["configuration"]
        != plan["policy_controller_configuration"]
    )
    assert (
        plan["preparation_template"]["execution_adapter"]["runtime_source_bundle"]
        == kwargs["runtime_source_bundle"]
    )
    assert (
        plan["preparation_template"]["execution_adapter"][
            "policy_observation_setup"
        ]
        == kwargs["policy_observation_setup"]
    )
    assert (
        plan["preparation_template"]["controller"]["model_or_asset_rights"] == plan["model_rights"]
    )
    assert len(plan["resolved_scenarios"]) == 10
    assert all(isinstance(row["seed"], int) for row in plan["resolved_scenarios"])
    assert plan["configured_offering_configuration_run_id"] == ("scene839873-configuration")
    assert plan["activation_automation"]["mode"] == (
        "automatic_after_no_spend_compilation"
    )
    assert plan["lineage_aliases"] == {
        "capture_session_id": "scene839873-configured-source",
        "capture_session_id_semantics": (
            "configured_scene_offering_source_launch_id_no_capture_upload_session"
        ),
        "intake_id": "scene839873-configuration",
        "intake_id_semantics": "configured_scene_offering_configuration_run_id",
    }
    assert plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
    legacy_setup = plan["legacy_policy_run_setup"]
    configuration = compile_policy_run_configuration(
        {
            "schema_version": "task_evaluation_policy_run_selection.v1",
            "run_id": "scene839873-fixed-seed-canary",
            "source_launch_id": legacy_setup["source_launch_id"],
            "offering_digest": legacy_setup["offering_digest"],
            "setup_digest": legacy_setup["setup_digest"],
            "preset_id": "quick_10",
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "scene_revision_digest": REVISION,
            "scene_controls_status_at_submission": "configured_controls_pending",
            "robot_preset_id": "franka_panda_robotiq_2f85_v1",
            "policy_candidate_ids": ["pi05_droid", "groot_n17_droid"],
            "notification": {
                "email": "robotics@example.com",
                "notify_on": ["completed", "blocked", "cancelled"],
            },
            "website_request_digest": "sha256:" + "7" * 64,
        },
        setup=legacy_setup,
    )
    public_cells = setup["episode_presets"][0]["matrix"]["cells"]
    legacy_cells = legacy_setup["presets"][0]["cells"]
    compiled_cells = configuration["matrix"]["cells"]
    assert (
        [(row["cell_id"], row["seed"], row["cell_digest"]) for row in public_cells]
        == [(row["cell_id"], row["seed"], row["cell_spec_digest"]) for row in legacy_cells]
        == [(row["cell_id"], row["seed"], row["cell_spec_digest"]) for row in compiled_cells]
    )
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
    kwargs["configured_source_commit"] = COMMIT
    kwargs["configured_offering_configuration_run_id"] = "scene839873-configuration"
    kwargs["offering_digest"] = "sha256:" + "f" * 64
    kwargs["policy_controller_configuration"] = {
        "uri": "s3://blueprint/policy-canary/controller.json",
        "digest": "sha256:" + "2" * 64,
        "size_bytes": 512,
    }
    kwargs["native_controller_configuration"] = {
        "uri": "s3://blueprint/policy-canary/native-controller.json",
        "digest": "sha256:" + "4" * 64,
        "size_bytes": 640,
    }
    kwargs["runtime_source_bundle"] = {
        "uri": "s3://blueprint/policy-canary/runtime-source.zip",
        "digest": "sha256:" + "5" * 64,
        "size_bytes": 4_272_421_731,
    }
    kwargs["model_rights"] = {
        "uri": "s3://blueprint/policy-canary/model-rights.json",
        "digest": "sha256:" + "3" * 64,
        "size_bytes": 768,
    }
    emitted = materialize_policy_canary_presubmission_setup(**kwargs)
    assert emitted["execution_setup_template"]["configured_source_commit"] == COMMIT
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
    assert controller["quick_10"]["matrix_digest"] == canonical_digest(
        {"ordered_cells": quick}
    )
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
