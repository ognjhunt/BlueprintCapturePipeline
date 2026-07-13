"""Evaluation Run spec: scene bundle + robot adapter + task/scenario pack +
policy adapter + runtime/provider profile + proof contract, composed fail-closed.

The g1_kitchen pack must reproduce the historical kitchen job values exactly so
existing evidence schemas stay byte-compatible; the warehouse pack proves the
same engine accepts a second site as pure configuration.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from blueprint_pipeline import evaluation_run as er
from blueprint_pipeline import isaac_g1_kitchen_parity_job as kitchen_job


# ----------------------------- pack registry -----------------------------


def test_builtin_packs_registered():
    known = er.known_evaluation_pack_ids()
    assert "g1_kitchen" in known
    assert "g1_warehouse" in known


def test_get_unknown_pack_raises():
    with pytest.raises(er.EvaluationRunSpecError):
        er.get_evaluation_pack("no_such_pack")


def test_register_duplicate_pack_rejected():
    spec = er.get_evaluation_pack("g1_kitchen")
    with pytest.raises(er.EvaluationRunSpecError):
        er.register_evaluation_pack(spec)


def test_register_requires_valid_spec():
    spec = er.get_evaluation_pack("g1_kitchen")
    broken = dataclasses.replace(spec, spec_id="broken_pack", policy=dataclasses.replace(spec.policy, policy_id=""))
    with pytest.raises(er.EvaluationRunSpecError):
        er.register_evaluation_pack(broken)


# ----------------------------- g1_kitchen pack -----------------------------


def test_g1_kitchen_pack_valid():
    spec = er.get_evaluation_pack("g1_kitchen")
    assert spec.validation_blockers() == []


def test_g1_kitchen_pack_is_single_source_for_kitchen_job_constants():
    """The kitchen job must consume these values from the pack, not re-declare them."""
    spec = er.get_evaluation_pack("g1_kitchen")
    assert kitchen_job.SCHEMA_VERSION == spec.proof.job_schema == "isaac_g1_kitchen_parity_job.v1"
    assert kitchen_job.JOB_MANIFEST_FILENAME == spec.proof.job_manifest_filename
    assert kitchen_job.LAUNCH_ATTEMPT_TRACE_FILENAME == spec.proof.launch_attempt_trace_filename
    assert kitchen_job.WORKER_BUNDLE_DIR == spec.scene.worker_bundle_dir == "/workspace/bundle"
    assert kitchen_job.ISAAC_G1_KITCHEN_PARITY_LANE == spec.runtime.lane_id == "isaac_g1_kitchen_parity"
    assert kitchen_job.DEFAULT_KITCHEN_MAIN_USD == spec.scene.main_usd_relative
    assert kitchen_job.DEFAULT_G1_USD_RELATIVE == spec.robot.robot_usd_relative
    assert kitchen_job.MAX_KITCHEN_ASSET_ARCHIVE_BYTES == spec.scene.max_asset_archive_bytes
    assert kitchen_job.ISAAC_WORKER_IMAGE_REF_ENV == spec.runtime.image_ref_env
    assert kitchen_job.ISAAC_WORKER_IMAGE_REF_FILE_ENV == spec.runtime.image_ref_file_env
    assert kitchen_job.ROBOT_EVAL_WORKER_IMAGE_REF_ENV == spec.runtime.fallback_image_ref_env
    assert kitchen_job.DEFAULT_ISAAC_WORKER_IMAGE_REF_FILE == spec.runtime.default_image_ref_file
    assert kitchen_job.DEFAULT_PARITY_IMAGE_REF == spec.runtime.default_image_ref
    assert kitchen_job.ISAAC_REVIEW_MIN_GPU_RAM_MB == spec.runtime.min_gpu_ram_mb == 48000
    assert kitchen_job.DEFAULT_ISAAC_REVIEW_PROVIDER == spec.runtime.default_providers[0] == "digitalocean"
    assert kitchen_job.ISAAC_G1_MAX_SPEND_USD_ENV == spec.runtime.max_spend_env
    assert (
        kitchen_job.DEFAULT_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS
        == spec.runtime.default_startup_no_runtime_timeout_seconds
    )


def test_g1_kitchen_pack_pins_legacy_evidence_identifiers():
    spec = er.get_evaluation_pack("g1_kitchen")
    assert spec.scene.evidence_key_prefix == "kitchen"
    assert spec.scene.layout_validation_schema == "kitchen_asset_layout_validation.v1"
    assert spec.scene.inventory_schema == "kitchen_asset_inventory_checksums.v1"
    assert spec.scene.request_usd_key == "kitchen_usd"
    assert spec.robot.request_usd_key == "g1_usd"
    assert spec.robot.robot_profile_id == "unitree_g1"
    assert spec.proof.request_schema == "isaac_g1_kitchen_parity_request.v1"
    assert spec.proof.bundle_schema == "isaac_g1_kitchen_parity_bundle.v2"
    assert spec.proof.harness_schema == "isaac_g1_kitchen_parity_harness.v1"
    assert spec.proof.closure_schema == "g1_kitchen_attempt_closure.v1"
    assert spec.proof.closure_required_blocker == "g1_kitchen_attempt_closure_missing"
    assert spec.proof.result_filename == "isaac_g1_kitchen_parity_result.json"
    assert spec.policy.policy_id == "blueprint_default_walk_to_target_smoke_policy"
    assert set(spec.policy.remote_runtime_policy_ids) == {
        "groot_sonic",
        "groot",
        "groot_n17_sonic",
        "unitree_groot_n17_sonic_policy",
    }


def test_g1_kitchen_robot_adapter_resolves_registered_profile():
    spec = er.get_evaluation_pack("g1_kitchen")
    profile = spec.robot.resolve_profile()
    assert profile.robot_id == "unitree_g1"
    assert profile.usd_prim_path == "/World/G1"


# ----------------------------- warehouse pack -----------------------------


def test_g1_warehouse_pack_valid_and_scene_neutral():
    spec = er.get_evaluation_pack("g1_warehouse")
    assert spec.validation_blockers() == []
    assert spec.scene.evidence_key_prefix == "scene"
    assert spec.scene.request_usd_key == "scene_usd"
    assert spec.robot.request_usd_key == "robot_usd"
    assert spec.runtime.lane_id != "isaac_g1_kitchen_parity"
    assert len(spec.tasks.scenarios) >= 1


def test_g1_warehouse_pack_declares_definition_only_claim_boundary():
    spec = er.get_evaluation_pack("g1_warehouse")
    manifest = spec.to_manifest()
    assert "pack_definition_only" in json.dumps(manifest.get("claim_boundary", ""))


# ----------------------------- fail-closed validation -----------------------------


def test_task_pack_requires_scenarios_or_task_file():
    spec = er.get_evaluation_pack("g1_kitchen")
    empty = dataclasses.replace(spec.tasks, scenarios=(), task_file=None)
    assert any("scenario" in blocker for blocker in empty.validation_blockers())
    respec = dataclasses.replace(spec, tasks=empty)
    assert respec.validation_blockers() != []


def test_scenario_rows_require_scenario_id():
    spec = er.get_evaluation_pack("g1_kitchen")
    bad = dataclasses.replace(spec.tasks, scenarios=({"spawn_position_xyz": [0, 0, 0]},))
    assert any("scenario_id" in blocker for blocker in bad.validation_blockers())


def test_blank_policy_id_blocked():
    spec = er.get_evaluation_pack("g1_kitchen")
    bad = dataclasses.replace(spec.policy, policy_id=" ")
    assert bad.validation_blockers() != []


def test_unknown_robot_profile_blocked():
    spec = er.get_evaluation_pack("g1_kitchen")
    bad = dataclasses.replace(spec.robot, robot_profile_id="not_a_registered_robot")
    assert any("robot_profile" in blocker for blocker in bad.validation_blockers())


def test_proof_schema_versions_must_be_versioned():
    spec = er.get_evaluation_pack("g1_kitchen")
    bad = dataclasses.replace(spec.proof, request_schema="isaac_request_unversioned")
    assert any("request_schema" in blocker for blocker in bad.validation_blockers())


def test_scene_bundle_requires_main_usd_and_mount():
    spec = er.get_evaluation_pack("g1_kitchen")
    bad = dataclasses.replace(spec.scene, main_usd_relative="")
    assert bad.validation_blockers() != []
    bad_mount = dataclasses.replace(spec.scene, bundle_mount_name="")
    assert bad_mount.validation_blockers() != []


def test_spec_validation_blockers_are_component_prefixed():
    spec = er.get_evaluation_pack("g1_kitchen")
    broken = dataclasses.replace(
        spec,
        policy=dataclasses.replace(spec.policy, policy_id=""),
        scene=dataclasses.replace(spec.scene, main_usd_relative=""),
    )
    blockers = broken.validation_blockers()
    assert any(blocker.startswith("policy:") for blocker in blockers)
    assert any(blocker.startswith("scene:") for blocker in blockers)


def test_assert_valid_raises_with_blockers():
    spec = er.get_evaluation_pack("g1_kitchen")
    broken = dataclasses.replace(spec, policy=dataclasses.replace(spec.policy, policy_id=""))
    with pytest.raises(er.EvaluationRunSpecError) as excinfo:
        broken.assert_valid()
    assert "policy:" in str(excinfo.value)


# ----------------------------- manifest round-trip -----------------------------


def test_manifest_round_trip():
    spec = er.get_evaluation_pack("g1_kitchen")
    manifest = spec.to_manifest()
    assert manifest["schema_version"] == "evaluation_run_spec.v1"
    assert manifest["validation"]["status"] == "PASS"
    restored = er.evaluation_run_spec_from_dict(manifest)
    assert restored == spec


def test_manifest_round_trip_survives_json():
    spec = er.get_evaluation_pack("g1_warehouse")
    restored = er.evaluation_run_spec_from_dict(json.loads(json.dumps(spec.to_manifest())))
    assert restored == spec


def test_from_dict_rejects_unknown_top_level_keys():
    manifest = er.get_evaluation_pack("g1_kitchen").to_manifest()
    manifest["surprise_field"] = True
    with pytest.raises(er.EvaluationRunSpecError):
        er.evaluation_run_spec_from_dict(manifest)


def test_from_dict_rejects_unknown_component_keys():
    manifest = er.get_evaluation_pack("g1_kitchen").to_manifest()
    manifest["scene_bundle"]["surprise"] = 1
    with pytest.raises(er.EvaluationRunSpecError):
        er.evaluation_run_spec_from_dict(manifest)


def test_from_dict_missing_component_fails_closed():
    manifest = er.get_evaluation_pack("g1_kitchen").to_manifest()
    del manifest["robot_adapter"]
    with pytest.raises(er.EvaluationRunSpecError):
        er.evaluation_run_spec_from_dict(manifest)


def test_from_json_file(tmp_path):
    spec = er.get_evaluation_pack("g1_kitchen")
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_manifest()), encoding="utf-8")
    assert er.evaluation_run_spec_from_json_file(path) == spec


# ----------------------------- generic scene asset inspection -----------------------------


KITCHEN_COLLECTED_NAMES = [
    "Collected_KitchenRoom/KitchenRoom.usd",
    "Collected_KitchenRoom/Materials/wood.mdl",
]
KITCHEN_ROOT_NAMES = ["KitchenRoom.usd", "textures/tile.png"]
KITCHEN_NESTED_NAMES = ["deep/nested/KitchenRoom.usd"]


@pytest.mark.parametrize(
    "names",
    [KITCHEN_COLLECTED_NAMES, KITCHEN_ROOT_NAMES, KITCHEN_NESTED_NAMES, [], ["other.usd"]],
)
def test_generic_inspector_matches_legacy_kitchen_inspector(names):
    scene = er.get_evaluation_pack("g1_kitchen").scene
    generic = er.inspect_scene_asset_namelist(names, scene=scene, source="reused_existing_url", byte_size=123)
    legacy = kitchen_job._inspect_kitchen_asset_namelist(names, source="reused_existing_url", byte_size=123)
    assert generic == legacy


def test_generic_inspector_scene_neutral_for_warehouse():
    scene = er.get_evaluation_pack("g1_warehouse").scene
    main = scene.main_usd_relative
    detail = er.inspect_scene_asset_namelist([main, "props/tote.usd"], scene=scene, source="local_asset_dir")
    assert detail["status"] == "PASS"
    assert detail["schema_version"] == scene.layout_validation_schema
    assert detail["selected_scene_main_usd_relative"] == main
    assert detail["expected_worker_scene_usd"] == (
        f"{scene.worker_bundle_dir}/{scene.bundle_mount_name}/{main}"
    )
    assert "kitchen" not in json.dumps(detail).lower()


def test_generic_inspector_fails_closed_on_missing_main_usd():
    scene = er.get_evaluation_pack("g1_warehouse").scene
    detail = er.inspect_scene_asset_namelist(["unrelated.txt"], scene=scene, source="local_asset_dir")
    assert detail["status"] == "FAIL"
    assert any(blocker.endswith("_main_usd_missing") for blocker in detail["blockers"])


def test_generic_dir_inspector_matches_legacy(tmp_path):
    asset_dir = tmp_path / "asset"
    (asset_dir / "Collected_KitchenRoom").mkdir(parents=True)
    (asset_dir / "Collected_KitchenRoom" / "KitchenRoom.usd").write_text("#usda 1.0", encoding="utf-8")
    scene = er.get_evaluation_pack("g1_kitchen").scene
    generic = er.inspect_scene_asset_dir_layout(asset_dir, scene=scene)
    legacy = kitchen_job._inspect_kitchen_asset_dir_layout(asset_dir)
    assert generic == legacy
    missing_generic = er.inspect_scene_asset_dir_layout(tmp_path / "nope", scene=scene)
    missing_legacy = kitchen_job._inspect_kitchen_asset_dir_layout(tmp_path / "nope")
    assert missing_generic == missing_legacy


# ----------------------------- generic runner request -----------------------------


SCENARIOS = [
    {
        "scenario_id": "walk_to_sink",
        "spawn_position_xyz": [0.0, 0.0, 0.8],
        "target_position_xyz": [2.49, 1.15, 1.02],
    }
]


def test_build_runner_request_matches_legacy_kitchen_request():
    spec = er.get_evaluation_pack("g1_kitchen")
    generic = er.build_runner_request(spec, scenarios=SCENARIOS, steps=64)
    legacy = kitchen_job.build_request(
        scenarios=SCENARIOS,
        policy_id=spec.policy.policy_id,
        steps=64,
    )
    assert generic == legacy


def test_build_runner_request_honors_overrides_and_audit_plan():
    spec = er.get_evaluation_pack("g1_kitchen")
    plan = {"mode": "spp_sweep"}
    generic = er.build_runner_request(
        spec,
        scenarios=SCENARIOS,
        steps=32,
        policy_id="groot_sonic",
        render_noise_audit_plan=plan,
    )
    legacy = kitchen_job.build_request(
        scenarios=SCENARIOS,
        policy_id="groot_sonic",
        steps=32,
        render_noise_audit_plan=plan,
    )
    assert generic == legacy
    assert generic["render_noise_audit"] == plan


def test_build_runner_request_scene_neutral_keys_for_warehouse():
    spec = er.get_evaluation_pack("g1_warehouse")
    request = er.build_runner_request(spec, scenarios=SCENARIOS, steps=16)
    assert request["schema_version"] == spec.proof.request_schema
    assert request["scene_usd"] == (
        f"{spec.scene.worker_bundle_dir}/{spec.scene.bundle_mount_name}/{spec.scene.main_usd_relative}"
    )
    assert request["robot_usd"] == spec.robot.robot_usd_relative
    assert "kitchen_usd" not in request
    assert "g1_usd" not in request


# ----------------------------- CLI -----------------------------


def test_cli_emits_pack_manifest(tmp_path, capsys):
    out = tmp_path / "spec_manifest.json"
    rc = er.main(["--pack", "g1_kitchen", "--out", str(out)])
    assert rc == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "evaluation_run_spec.v1"
    assert manifest["spec_id"] == "g1_kitchen"


def test_cli_lists_packs(capsys):
    rc = er.main(["--list-packs"])
    assert rc == 0
    listing = capsys.readouterr().out
    assert "g1_kitchen" in listing
    assert "g1_warehouse" in listing


def test_cli_validates_external_spec_file(tmp_path):
    spec = er.get_evaluation_pack("g1_warehouse")
    path = tmp_path / "external.json"
    path.write_text(json.dumps(spec.to_manifest()), encoding="utf-8")
    assert er.main(["--spec-json", str(path)]) == 0
    broken = spec.to_manifest()
    broken["policy_adapter"]["policy_id"] = ""
    path.write_text(json.dumps(broken), encoding="utf-8")
    assert er.main(["--spec-json", str(path)]) != 0
