from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_native_continuation_controls_input import (
    NativeContinuationControlsInputError,
    main,
    materialize_native_continuation_controls_input,
)


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return {"digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def _case(tmp_path, *, publication=False, continuing_spend=False):
    assets = tmp_path / "assets"
    assets.mkdir()
    asset_rows = []
    for role, name in [("appearance", "appearance.usdc"), ("collision", "collision.usda"), ("replacement", "replacement.usdz")]:
        p = assets / name
        p.write_bytes(("unit-fixture-" + role).encode())
        asset_rows.append({"role": role, "relative_path": name, "digest": "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size})
    scene = {"schema_version": "task_evaluation_configured_scene_bundle_candidate.v1", "robot_neutral": True,
             "robot_specific_base_registration_included": False, "assets": asset_rows, "manifest_digest": ""}
    scene["manifest_digest"] = canonical_digest(scene, digest_field="manifest_digest")
    scene_ref = _write(assets / "configured_scene_bundle_candidate.v1.json", scene)
    static, native_import = tmp_path / "static.json", tmp_path / "native-import.json"
    static_ref, import_ref = _write(static, {"unit_fixture": "static"}), _write(native_import, {"unit_fixture": "native-import"})
    references, overrides = [], {}
    sources = ["task.definition", "task.success_criteria", "task.execution", "task.subject.source_object",
               "scene.registration.support_plane", "scene.registration.workspace_clearance"]
    for contract in sources:
        p = tmp_path / (contract + ".json")
        reference = _write(p, {"unit_fixture": contract})
        references.append({"contract_path": contract, "materialized_path": str(p), "uri": "unit-fixture://" + contract,
                           "full_byte_service_account_readback_passed": True, **reference})
        overrides[contract] = p
    stages, configs = [], {}
    for index in range(1, 7):
        stage_id = "stage-" + str(index)
        configs[stage_id] = {"sha256": "sha256:" + str(index) * 64}
        artifacts = []
        if index == 4:
            artifacts = [{"role": "static_qualification_receipt", **static_ref}]
        if index == 5:
            artifacts = [{"role": "native_import_qualification_receipt", **import_ref},
                         {**asset_rows[-1], "role": "native_qualified_replacement_asset"}]
        if index == 6:
            artifacts = [{"role": "configured_scene_bundle_candidate_manifest", **scene_ref}]
        stage = {"schema_version": "task_evaluation_scene_configuration_stage_result.v1", "stage_id": stage_id,
                 "status": "completed", "canonical_allocator": None, "provider_mutations_performed": 0,
                 "paid_execution_requested": False, "executed_inside_parent_configuration_run": True,
                 "raw_secret_values_recorded": False, "configuration_digest": configs[stage_id]["sha256"],
                 "output_artifacts": artifacts, "stage_result_digest": ""}
        stage["stage_result_digest"] = canonical_digest(stage, digest_field="stage_result_digest")
        stages.append(stage)
    envelope = {"run_id": "unit-fixture-run", "materialized_references": references,
                "recipe": {"stage_sequence": [{"stage_id": x["stage_id"]} for x in stages]}, "envelope_digest": ""}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    manifest = {"schema_version": "scene_configuration_native_continuation.v1", "diagnostic_kind": "scene_configuration_native_continuation",
                "qualification_eligible": False, "configured_revision_publication_permitted": publication,
                "implementation_commit": "a" * 40, "construction_envelope_digest": envelope["envelope_digest"],
                "configurations": configs, "input_digest": ""}
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    bundle = tmp_path / "source.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("provider_runtime/native_continuation_manifest.v1.json", json.dumps(manifest))
        archive.writestr("provider_runtime/construction_envelope.v1.json", json.dumps(envelope))
    chain = {"schema_version": "task_evaluation_scene_configuration_provider_stage_chain.v1", "status": "completed",
             "run_id": envelope["run_id"], "stage_results": stages, "stage_count": 6,
             "stage_result_digests": [x["stage_result_digest"] for x in stages],
             "executed_inside_one_parent_provider_run": True, "nested_provider_mutations_performed": 0,
             "nested_paid_execution_requested": False, "evaluation_episode_executed": False, "retry_cap": 0, "result_digest": ""}
    chain["result_digest"] = canonical_digest(chain, digest_field="result_digest")
    native = {"schema_version": "scene_configuration_native_continuation_result.v1", "status": "completed",
              "claim_ceiling": "development_only", "qualification_eligible": False, "candidate_policy_queried": False,
              "provider_zero_required_after_return": True,
              "implementation_commit": manifest["implementation_commit"], "stage_chain": chain, "result_digest": ""}
    native["result_digest"] = canonical_digest(native, digest_field="result_digest")
    native_path = tmp_path / "native-result.json"
    _write(native_path, native)
    allocator = {"status": "completed", "native_control_result_digest": native["result_digest"],
                 "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
                 "continuing_spend_from_this_run": continuing_spend, "all_staged_objects_absent": True,
                 "provider_closeout": {"provider_zero_confirmed": True}, "blockers": []}
    allocator_path = tmp_path / "allocator.json"
    _write(allocator_path, allocator)
    return dict(continuation_result_path=native_path, allocator_result_path=allocator_path,
                continuation_bundle_path=bundle, configured_assets_root=assets, static_qualification_path=static,
                native_import_qualification_path=native_import, output_dir=tmp_path / "out",
                materialized_reference_overrides=overrides)


def test_closed_native_outputs_reuse_bytes_without_inheriting_authority(tmp_path):
    value = materialize_native_continuation_controls_input(**_case(tmp_path))
    assert value["status"] == "materialized"
    assert value["prior_spend_authority_inherited"] is False
    assert value["prior_deadline_extended"] is False
    assert value["qualification_eligible"] is False
    assert value["configured_revision_publication_permitted"] is False
    assert value["evaluation_ready_promotion_permitted"] is False
    assert value["provider_mutation_performed"] is False
    assert "revision_digest" not in value
    assert value["receipt_digest"] == canonical_digest(value, digest_field="receipt_digest")


@pytest.mark.parametrize("changed", ["task", "asset", "static"])
def test_reuse_refuses_changed_source_bytes_before_creating_output(tmp_path, changed):
    args = _case(tmp_path)
    path = {"task": args["materialized_reference_overrides"]["task.definition"],
            "asset": args["configured_assets_root"] / "replacement.usdz", "static": args["static_qualification_path"]}[changed]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(NativeContinuationControlsInputError, match="bound_input_changed"):
        materialize_native_continuation_controls_input(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize("flags,reason", [({"publication": True}, "source_binding_invalid"),
                                         ({"continuing_spend": True}, "completed_closed_source_required")])
def test_reuse_refuses_wider_authority_or_unclosed_source(tmp_path, flags, reason):
    args = _case(tmp_path, **flags)
    with pytest.raises(NativeContinuationControlsInputError, match=reason):
        materialize_native_continuation_controls_input(**args)
    assert not args["output_dir"].exists()


def _argv(args):
    argv = ["--continuation-result", str(args["continuation_result_path"]),
            "--allocator-result", str(args["allocator_result_path"]),
            "--continuation-bundle", str(args["continuation_bundle_path"]),
            "--configured-assets-root", str(args["configured_assets_root"]),
            "--static-qualification", str(args["static_qualification_path"]),
            "--native-import-qualification", str(args["native_import_qualification_path"]),
            "--output-dir", str(args["output_dir"])]
    for contract_path, path in args["materialized_reference_overrides"].items():
        argv += ["--reference-override", f"{contract_path}={path}"]
    return argv


def test_command_line_writes_the_same_receipt(tmp_path, capsys):
    args = _case(tmp_path)
    assert main(_argv(args)) == 0
    receipt = json.loads((args["output_dir"] / "diagnostic_controls_input.v1.json").read_text())
    assert json.loads(capsys.readouterr().out) == {
        "status": "materialized", "receipt_digest": receipt["receipt_digest"]}
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert receipt["prior_spend_authority_inherited"] is False


def test_command_line_refuses_changed_bytes_without_output(tmp_path, capsys):
    args = _case(tmp_path)
    static = args["static_qualification_path"]
    static.write_bytes(static.read_bytes() + b"changed")
    assert main(_argv(args)) == 2
    assert "bound_input_changed" in json.loads(capsys.readouterr().out)["reason"]
    assert not args["output_dir"].exists()
