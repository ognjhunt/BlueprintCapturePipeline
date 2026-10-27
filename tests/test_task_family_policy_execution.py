"""Family dispatch seals runtime selection without admitting request-supplied code."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_family_policy_execution import (
    FAMILY_BACKEND,
    FAMILY_POLICY_INTERFACE,
    task_family_policy_execution_binding,
    validate_task_family_policy_execution_spec,
    validate_task_family_runtime_selection,
)
from tests.test_task_evaluation_task_family_native_adapter import SHA, _runtime_contract


def _scene():
    native = _runtime_contract()
    return {"native_runtime_contract": native, "scene_id": native["scene_id"],
            "task_id": native["task_id"], "task_kind": native["task_kind"],
            "task_spec": native["task_spec"],
            "scenario": {"cell_id": native["scenario_cell_id"], "seed": native["seed"],
                         "scenario_instance_digest": native["scenario_instance_digest"]},
            "task_family_runtime_selection": {"runtime_id": "qualified-button-engine", "runtime_catalog_digest": SHA}}


def _spec(scene):
    task = scene["task_spec"]["task_success_contract"]
    spec = {"schema_version": "native_task_arena_policy_canary_execution_spec.v1",
            "evaluation_backend": FAMILY_BACKEND, "policy_interface_id": FAMILY_POLICY_INTERFACE,
            "candidate_id": "qualified-policy", "task_success_contract": task,
            "task_success_contract_digest": task["contract_digest"],
            **task_family_policy_execution_binding(scene), "checkpoint_digest": SHA,
            "runtime_identity_digest": SHA,
            "policy_spec": {"policy_id": "button-policy", "endpoint_url": "https://policy.example/infer",
                "metadata_url": "https://policy.example/metadata", "allowed_origin": "https://policy.example",
                "model_identity_digest": SHA, "embodiment_id": "humanoid1", "controller": "joint_control",
                "action_schema_id": "joint_action1", "observation_schema_id": "rgb_proprio1",
                "action_bounds": [[-1, 1]]},
            "policy_endpoint": {"credential_env": "BLUEPRINT_FAMILY_POLICY_FILE"}, "candidate_rights_binding": {},
            "prompt": "Press the captured button", "max_policy_queries": 2, "open_loop_horizon": 1,
            "maximum_action_steps": 2, "control_frequency_hz": 1}
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec


def test_family_execution_spec_seals_the_native_scene_and_operator_runtime_selection():
    scene = _scene()
    spec = _spec(scene)
    assert validate_task_family_policy_execution_spec(spec, candidate_id="qualified-policy", scene_plan=scene) == spec
    changed = copy.deepcopy(scene)
    changed["task_family_runtime_selection"]["runtime_id"] = "another-qualified-engine"
    with pytest.raises(ValueError, match="execution_scene_mismatch"):
        validate_task_family_policy_execution_spec(spec, scene_plan=changed)


def test_resolved_cell_preserves_sealed_source_except_its_scenario_identity():
    scene = _scene()
    source = copy.deepcopy(scene["native_runtime_contract"])
    native = scene["native_runtime_contract"]
    native.update(seed=7, scenario_cell_id="variation-1", scenario_instance_digest="sha256:" + "b" * 64)
    native["contract_digest"] = canonical_digest(native, digest_field="contract_digest")
    scene.update(source_native_runtime_contract=source, source_native_runtime_contract_digest=source["contract_digest"])
    scene["scenario"] = {"seed": native["seed"], "cell_id": native["scenario_cell_id"],
                         "scenario_instance_digest": native["scenario_instance_digest"]}
    assert task_family_policy_execution_binding(scene)["native_runtime_contract_digest"] == source["contract_digest"]
    native["assets"][0]["reset_state"] = {"unsealed_reset": 42}
    native["contract_digest"] = canonical_digest(native, digest_field="contract_digest")
    with pytest.raises(ValueError, match="resolved_contract_changed"):
        task_family_policy_execution_binding(scene)


@pytest.mark.parametrize("extra", [{"factory": "untrusted.module:build"}, {"runtime_catalog_path": "/capture/catalog.json"}])
def test_runtime_selection_never_accepts_a_job_supplied_factory_or_catalog_path(extra):
    with pytest.raises(ValueError, match="selection_invalid"):
        validate_task_family_runtime_selection({**_scene()["task_family_runtime_selection"], **extra})


def test_family_execution_spec_refuses_legacy_droid_interface_and_unsealed_mutation():
    spec = _spec(_scene())
    spec["policy_interface_id"] = "openpi_droid.v1"
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_task_family_policy_execution_spec(spec)
    spec = _spec(_scene())
    spec["max_policy_queries"] = 500
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_task_family_policy_execution_spec(spec)


def test_provider_bundle_stages_only_operator_catalog_and_sealed_qualification(tmp_path, monkeypatch):
    from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint, _family_operator_inputs

    qualification = tmp_path / "qualification.json"
    qualification.write_text('{"decision":"qualified_for_bounded_simulation"}\n')
    qualification_sha = "sha256:" + hashlib.sha256(qualification.read_bytes()).hexdigest()
    catalog = {"schema_version": "task_family_native_runtime_catalog.v1", "runtimes": [
        {"runtime_id": "qualified-button-engine", "qualification": {
            "path": "qualification.json", "sha256": qualification_sha}}]}
    catalog["catalog_digest"] = canonical_digest(catalog, digest_field="catalog_digest")
    path = tmp_path / "operator-catalog.json"
    path.write_text(json.dumps(catalog))
    scene = _scene()
    scene["task_family_runtime_selection"]["runtime_catalog_digest"] = catalog["catalog_digest"]
    spec = _spec(scene)
    monkeypatch.delenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", raising=False)
    with pytest.raises(ValueError, match="catalog_not_configured"):
        _family_operator_inputs({"qualified-policy": spec})
    monkeypatch.setenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", str(path))
    files, receipt = _family_operator_inputs({"qualified-policy": spec})
    assert set(files) == {"runtime_operator_inputs/catalog.json", "runtime_operator_inputs/qualification.json"}
    assert receipt["runtime_catalog_digest"] == catalog["catalog_digest"]
    assert 'export BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH="$RUNTIME_DIR/runtime_operator_inputs/catalog.json"' in _entrypoint(
        local_candidates=(), family_runtime_catalog_staged=True)
    qualification.write_text('{"decision":"tampered"}\n')
    with pytest.raises(ValueError, match="qualification_changed"):
        _family_operator_inputs({"qualified-policy": spec})


def test_family_beta_admission_requires_pending_qualification_instead_of_invented_probe():
    from blueprint_pipeline.native_task_arena_policy_canary_session import (
        BETA_ADMISSION_SCHEMA_VERSION, BETA_RUN_KIND, PolicyCanarySessionError, _beta_admission_binding,
    )

    payload = {"run_kind": BETA_RUN_KIND, "candidate_ids": ["qualified-policy"],
               "task_success_contract": _scene()["task_spec"]["task_success_contract"],
               "pre_policy_site_qualification_required": True, "family_static_readiness_digest": SHA,
               **{key: SHA for key in ("scene_revision_digest", "task_success_contract_digest",
                                      "plan_digest", "configuration_digest")}}
    payload["beta_screening_admission"] = {
        "schema_version": BETA_ADMISSION_SCHEMA_VERSION, "candidate_id": "qualified-policy",
        "evaluation_purpose": "screening", "backend": FAMILY_BACKEND, "admission_digest": SHA,
        **{key: payload[key] for key in ("scene_revision_digest", "task_success_contract_digest",
            "plan_digest", "configuration_digest", "pre_policy_site_qualification_required", "family_static_readiness_digest")}}
    _beta_admission_binding(payload)
    payload["pre_policy_site_qualification_required"] = False
    with pytest.raises(PolicyCanarySessionError, match="beta_admission_binding_invalid"):
        _beta_admission_binding(payload)


def test_family_session_keeps_one_allocation_authority_before_native_qualification(tmp_path):
    from blueprint_pipeline.native_task_arena_policy_canary_session import (
        BETA_ADMISSION_SCHEMA_VERSION, BETA_RUN_KIND, build_session_authority, validate_runtime_input_manifest,
    )
    from tests.test_native_task_arena_policy_canary_session import _activation, _record, _runtime_inputs

    activation = _activation()
    inputs = _runtime_inputs(tmp_path, activation)
    task = _scene()["task_spec"]["task_success_contract"]
    inputs.update(run_kind=BETA_RUN_KIND, candidate_ids=["qualified-policy"], task_success_contract=task,
        task_success_contract_digest=task["contract_digest"], scene_revision_digest=SHA,
        pre_policy_site_qualification_required=True, family_static_readiness_digest=SHA)
    inputs.pop("construction_result")
    for cell in inputs["cells"]:
        cell["control_diagnostic"] = {"mode": "native_family_prepolicy_required",
            "typed_gap": "native_site_qualification_pending", "policy_execution_blocked": True}
    inputs["beta_screening_admission"] = {
        "schema_version": BETA_ADMISSION_SCHEMA_VERSION, "candidate_id": "qualified-policy",
        "evaluation_purpose": "screening", "backend": FAMILY_BACKEND, "admission_digest": SHA,
        **{key: inputs[key] for key in ("scene_revision_digest", "task_success_contract_digest",
            "plan_digest", "configuration_digest", "pre_policy_site_qualification_required", "family_static_readiness_digest")}}
    activation.update(run_kind=BETA_RUN_KIND, candidate_ids=["qualified-policy"], task_success_contract=task,
                      task_success_contract_digest=task["contract_digest"])
    activation["activation_digest"] = canonical_digest(activation, digest_field="activation_digest")
    inputs["activation_digest"] = activation["activation_digest"]
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    assert "construction_result" not in validate_runtime_input_manifest(inputs)
    activation_path, inputs_path = tmp_path / "activation.json", tmp_path / "inputs.json"
    activation_path.write_text(json.dumps(activation))
    inputs_path.write_text(json.dumps(inputs))
    authority = build_session_authority(activation_manifest=activation, activation_record=_record(activation_path),
        runtime_inputs=inputs, runtime_input_record=_record(inputs_path),
        resource_name="blueprint-native-task-policy-canary-family", hard_cap_usd=2, hard_ttl_seconds=60)
    assert authority["maximum_provider_allocations"] == 1 and authority["retry_cap"] == 0
    assert authority["pre_policy_site_qualification_required"] is True
    assert authority["family_static_readiness_digest"] == SHA
    assert "robot_free_probe_receipt_digest" not in authority["beta_screening_admission"]


def test_family_provider_transport_requires_native_runtime_and_operator_bytes():
    from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint
    from blueprint_pipeline.policy_canary_session_bundle_identity import (
        ENTRYPOINT, RESULT_FILENAME, SCHEMA, run_kind_entrypoint_blockers,
        run_kind_required_entries, validate_policy_canary_provider_manifest,
    )

    manifest = {"schema_version": SCHEMA, "status": "ready", "blockers": [],
        "execution_mode": "blueprint_beta_screening_session", "run_kind": "blueprint_beta_screening",
        "candidate_ids": ["qualified-policy"], "claim_ceiling": "diagnostic_policy_execution",
        "episodes_per_policy": 1, "learned_policy_rollout_count": 1, "maximum_provider_allocations": 1,
        "retry_cap": 0, "expected_output_filename": RESULT_FILENAME, "runtime_entrypoint": ENTRYPOINT,
        "candidate_policy_queried": False, "provider_zero_required_after_return": True,
        "evaluation_backend": FAMILY_BACKEND, "team_policy_interface_id": FAMILY_POLICY_INTERFACE,
        "team_credential_env": "BLUEPRINT_FAMILY_POLICY_FILE", "secret_material_bundled": False,
        "pre_policy_site_qualification_required": True, "family_static_readiness_digest": SHA,
        "task_family_operator_inputs": {"catalog_relative_path": "runtime_operator_inputs/catalog.json",
            "runtime_catalog_digest": SHA, "files": [{"relative_path": "runtime_operator_inputs/catalog.json",
                                                       "sha256": SHA, "size_bytes": 1}]}}
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    validate_policy_canary_provider_manifest(manifest)
    required = run_kind_required_entries(manifest)
    assert "provider_runtime/blueprint_pipeline/task_family_native_controls.py" in required
    assert "provider_runtime/runtime_operator_inputs/catalog.json" in required
    assert not any("robolab/eval" in name or "construction_result" in name for name in required)
    assert run_kind_entrypoint_blockers(manifest, _entrypoint(local_candidates=(),
        run_kind="blueprint_beta_screening", family_runtime_catalog_staged=True)) == []
    manifest["task_family_operator_inputs"]["files"][0]["relative_path"] = "runtime_operator_inputs/../capture.py"
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    with pytest.raises(ValueError, match="manifest_invalid"):
        validate_policy_canary_provider_manifest(manifest)


def test_family_bundle_preserves_native_camera_identity_without_droid_external_role(tmp_path, monkeypatch):
    from blueprint_pipeline import native_task_camera_start_configuration as camera
    from blueprint_pipeline.native_task_arena_bundle import _rendered_external_views

    scene = _scene()
    assert "role" not in scene["native_runtime_contract"]["cameras"][0]
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "native_task_arena_scene_plan.v1.json").write_text(json.dumps(scene))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"cells": [{"cell_id": "cell1"}]}))
    def droid_camera_cannot_apply(_plan):
        raise AssertionError("family camera must retain its native SDK contract")
    monkeypatch.setattr(camera, "runtime_external_camera_row", droid_camera_cannot_apply)
    assert _rendered_external_views(packet, {"policy_canary_runtime_inputs.json": inputs}) == []
