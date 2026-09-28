# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_preparation_activation_references.py
"""Finite supported references and conservative exact joins."""
import json
from dataclasses import replace

import pytest

from tests.test_preparation_activation_reference_records import (
    C, D, activation_request, digest, observe, preparation_request, record, reference, sealed,
)


def materialized(path="scene.configured_revision", **changes):
    return {"contract_path": path, **reference(), "materialized_path": "/inputs/file",
            "content_addressed_reuse": False, "full_byte_service_account_readback_passed": True, **changes}


def preparation_set(request=None, **changes):
    request = request or preparation_request()
    paths = ["scene.configured_revision", "sensors.configuration", "runtime.health_protocol", "execution_adapter.runtime_source_bundle"]
    refs = [materialized(p, materialized_path="/inputs/" + str(i)) for i, p in enumerate(paths)]
    result = record(role="result", request=request,
                    status="native_arena_inputs_verified_awaiting_profile_authority",
                    run_id=request["run_id"], team_namespace=request["team_namespace"], source_commit=C,
                    references=refs, reference_count=len(refs), full_byte_service_account_readback_passed=True,
                    provider_mutation_performed=False, catalog_mutation_performed=False, paid_execution_requested=False,
                    **changes)
    return [record(request=request), record(role="identity", request=request), result]


def activation_set(prep=None, **changes):
    prep = prep or preparation_set()
    request = activation_request(preparation={"preparation_id": "prep", "request_digest": digest(json.loads(prep[0].raw_bytes)["request"]),
                                              "result_digest": json.loads(prep[-1].raw_bytes)["result_digest"]})
    request.update(changes)
    result = record("activation", "result", request=request,
                    status="profile_authority_materialized_no_execution", preparation_id="prep", team_namespace="team",
                    lane=request["lane"], source_commit=C, preparation_result_digest=request["preparation"]["result_digest"],
                    release_window_digest="sha256:" + "2" * 64, profile_id="profile", profile_digest=D,
                    profile_publication_receipt_digest=D, standing_authorization_digest=D,
                    full_byte_activation_reference_readback_passed=True, profile_publication_performed=True,
                    catalog_mutation_performed=True, standing_authorization_published=True,
                    provider_mutation_performed=False, paid_execution_requested=False)
    return [record("activation", request=request, state="prepared"), record("activation", "identity", request=request), result]


def rewrite(row, **changes):
    value = json.loads(row.raw_bytes)
    value.update(changes)
    field = "envelope_digest" if row.role == "envelope" else "result_digest"
    return replace(row, raw_bytes=json.dumps(sealed(value, field)).encode())


def test_complete_available_family_join_is_narrow_and_all_authority_flags_stay_false():
    prep = preparation_set()
    result = observe(*prep, *activation_set(prep))
    assert result.complete_supplied_supported_projection
    assert len(result.local_path_protections) == 4
    assert all(row.binding_status == "request_bound" for row in result.local_path_protections)
    assert all(len(row.related_sources) == 1 for row in result.local_path_protections)
    assert not result.producer_authorization_verified and not result.request_admission_verified
    assert not result.general_reference_inventory_complete and not result.references_clear


@pytest.mark.parametrize("status", ["blocked", "unknown", "native_arena_inputs_verified_awaiting_profile_authority"])
def test_receipt_only_paths_survive_without_parent_and_under_unknown_status(status):
    row = record(role="result", status=status, references=[materialized()], reference_count=1,
                 full_byte_service_account_readback_passed=True)
    result = observe(row)
    assert result.local_path_protections[0].path == "/inputs/file"
    assert result.local_path_protections[0].binding_status == "receipt_only"
    assert not result.complete_supplied_supported_projection


@pytest.mark.parametrize("path", ["execution_adapter.runtime_source_bundle.external_layers.0",
                                   "construction.recipe.stage_sequence.0.configuration", "scene.configured_revision.appearance", "unknown.nested.path"])
def test_transitive_and_unknown_result_rows_never_upgrade_by_prefix(path):
    row = record(role="result", references=[materialized(path)], reference_count=1)
    result = observe(row)
    assert result.local_path_protections[0].reason == "deferred_parent_reference_proof"
    assert result.local_path_protections[0].binding_status == "receipt_only"
    assert result.missing_edge_obligations


@pytest.mark.parametrize("change", [{"uri": "gs://bucket/other"}, {"digest": "sha256:" + "2" * 64}, {"size_bytes": 2}])
def test_binding_mismatch_preserves_receipt_protection(change):
    prep = preparation_set()
    refs = json.loads(prep[-1].raw_bytes)["references"]
    refs[0].update(change)
    prep[-1] = rewrite(prep[-1], references=refs)
    result = observe(*prep)
    assert len(result.local_path_protections) == 4
    assert "materialized_reference_binding_mismatch" in result.blockers
    assert any(row.binding_status == "receipt_only" for row in result.local_path_protections)


@pytest.mark.parametrize("change", [{"size_bytes": True}, {"size_bytes": 0}, {"uri": "file:///wrong"},
                                   {"digest": "sha256:" + "A" * 64}, {"full_byte_service_account_readback_passed": False},
                                   {"content_addressed_reuse": 1}, {"materialized_path": "../wrong"},
                                   {"host_source_readback": {"network_fetch_performed": True}}])
def test_malformed_materialized_reference_never_becomes_valid_protection(change):
    result = observe(record(role="result", references=[materialized(**change)], reference_count=1))
    assert not result.local_path_protections
    assert not result.complete_supplied_supported_projection


def test_wrong_generic_path_sha_shape_cannot_substitute_for_materialized_fields():
    ref = {"contract_path": "scene.configured_revision", "path": "/wrong", "sha256": D, "size_bytes": 1}
    result = observe(record(role="result", references=[ref], reference_count=1))
    assert not result.local_path_protections


def test_reference_count_mismatch_and_duplicate_contract_path_are_explicit():
    result = observe(record(role="result", references=[materialized(), materialized()], reference_count=1))
    assert "materialized_reference_count_invalid" in result.blockers
    assert len(result.local_path_protections) == 2
    assert not result.complete_supplied_supported_projection


@pytest.mark.parametrize("state", ["completed", "blocked", "pending"])
def test_activation_cannot_reopen_nonmaterialized_preparation(state):
    prep = preparation_set()
    activation = activation_set(prep)
    prep[0] = replace(prep[0], row_path=prep[0].row_path.replace("materialized", state))
    result = observe(*prep, *activation)
    assert "activation_preparation_unresolved" in result.blockers
    assert not result.complete_supplied_supported_projection


def test_exact_older_historical_result_is_selected_without_dropping_newer_protection():
    prep = preparation_set()
    activation = activation_set(prep)
    newer = rewrite(prep[-1], observed_at_iso="2026-09-29T00:00:00Z")
    result = observe(*prep, newer, *activation)
    assert len(result.local_path_protections) == 8
    assert "activation_preparation_unresolved" not in result.blockers
    missing = observe(prep[0], prep[1], newer, *activation)
    assert "activation_preparation_unresolved" in missing.blockers
    assert result == observe(*reversed([*prep, newer, *activation]))


def test_same_canonical_historical_versions_keep_each_raw_provenance():
    prep = preparation_set()
    formatted = replace(prep[-1], raw_bytes=json.dumps(json.loads(prep[-1].raw_bytes), indent=2).encode())
    result = observe(*prep, formatted, *activation_set(prep))
    assert result.complete_supplied_supported_projection
    assert len(result.local_path_protections) == 8
    assert len({row.source.raw_sha256 for row in result.local_path_protections}) == 2


def test_profile_and_window_digest_selectors_never_fabricate_local_tuple():
    prep = preparation_set()
    result = observe(*prep, *activation_set(prep))
    raws = result.raw_digest_selector_obligations
    assert {row.contract_path for row in raws} >= {"profile_publication_receipt_digest", "standing_authorization_digest"}
    assert all(row.path is None and row.size_bytes is None for row in raws)
    windows = [row for row in result.canonical_document_selector_obligations if row.contract_path == "release_window_digest"]
    assert windows[0].digest != reference()["digest"]
    assert all(not row.path for row in windows)


def test_campaign_canary_raw_path_has_no_fabricated_size_or_authority():
    row = record("activation", "result", status="policy_campaign_queue_materialized_no_execution",
                 policy_campaign_activation_digest=D, policy_campaign_activation_sha256=D,
                 policy_canary_runtime_inputs_path="/canary/runtime.json", policy_canary_runtime_inputs_sha256=D,
                 policy_canary_runtime_inputs_digest=D, provider_mutation_performed=False, paid_execution_requested=False)
    result = observe(row)
    protection = result.local_path_protections[0]
    assert protection.path == "/canary/runtime.json" and protection.size_bytes is None
    assert protection.digest_meaning == "raw_digest_only"
    assert not result.complete_supplied_supported_projection


@pytest.mark.parametrize("missing", ["policy_canary_runtime_inputs_sha256", "policy_canary_runtime_inputs_digest"])
def test_partial_canary_companions_refuse_that_projection(missing):
    data = {"policy_canary_runtime_inputs_path": "/canary/runtime.json", "policy_canary_runtime_inputs_sha256": D,
            "policy_canary_runtime_inputs_digest": D}
    data.pop(missing)
    result = observe(record("activation", "result", status="policy_campaign_queue_materialized_no_execution", **data))
    assert not result.local_path_protections


@pytest.mark.parametrize("value", [None, {"uri": "file:///bad", "digest": D, "size_bytes": 1},
                                  {"uri": "gs://bucket/object", "digest": D, "size_bytes": True}])
def test_required_remote_leaf_malformed_is_not_optional_absence(value):
    result = observe(record(request=preparation_request(scene={"mode": "reuse_configured_revision", "configured_revision": value})))
    assert "supported_reference_invalid" in result.blockers
    assert not result.remote_raw_references or all(row.contract_path != "scene.configured_revision" for row in result.remote_raw_references)


@pytest.mark.parametrize("field", ["policy_run_setup", "policy_run_selection", "policy_run_configuration", "policy_canary_activation"])
def test_opaque_policy_is_not_recursively_searched(field):
    result = observe(record(request=preparation_request(**{field: {"path": "/never-inferred", "nested": reference()}})))
    assert not result.local_path_protections
    assert any(row.contract_path == field and row.reason == "deferred_semantic_object" for row in result.missing_edge_obligations)


def test_unknown_object_metadata_retains_raw_but_cannot_complete_zero_projection():
    result = observe(record(request=preparation_request(scene={"mode": "future", "nested": reference()})))
    assert "unsupported_metadata_shape" in result.blockers
    assert not result.complete_supplied_supported_projection


@pytest.mark.parametrize("family,role", [(f, r) for f in ("preparation", "activation")
                                        for r in ("envelope", "identity", "result_conflict")]
                         + [("activation", "result")])
@pytest.mark.parametrize("raw", [b"{}", b'{"schema_version":"future"}'])
def test_invalid_raw_sibling_contests_immutable_valid_selector(family, role, raw):
    first = record(family, role, state="pending")
    second = replace(first, raw_bytes=raw)
    result = observe(first, second)
    assert "immutable_record_conflict" in result.blockers
    assert next(row for row in result.records if row.source.raw_size_bytes == len(first.raw_bytes)).disposition == "conflict"
    assert result == observe(second, first)


def test_invalid_historical_preparation_result_does_not_contest_valid_history():
    row = record(role="result")
    result = observe(row, replace(row, raw_bytes=b"{}"))
    assert "immutable_record_conflict" not in result.blockers
    assert len(result.records) == 2 and not result.complete_supplied_supported_projection


def test_unknown_result_field_prevents_apparently_complete_projection():
    prep = preparation_set()
    prep[-1] = rewrite(prep[-1], hidden_reference={"path": "/never-inferred"})
    result = observe(*prep)
    assert "unsupported_metadata_shape" in result.blockers
    assert len(result.local_path_protections) == 4


def configured_request():
    ref = reference()
    return preparation_request(
        scene={"mode": "configure_source_scene", "source_manifest": ref,
               "appearance": {"kind": "textured_usd", "representation": ref, "renderer_qualification": ref},
               "geometry": {"kind": "observed_mesh", "collision": ref, "validation": ref, "source_derivation": ref},
               "registration": {key: ref for key in ("metric_registration", "support_plane", "robot_mount_interface", "workspace_clearance", "camera_calibration")},
               "rights": {"admission": ref, "evidence": [{"role": "publisher_terms", "artifact": ref}, {"role": "human_authority_record", "artifact": ref}]},
               "website_native_inputs": {"runtime_inputs": ref, "appearance": ref, "observations": ref, "candidate": ref, "frames": [ref]}},
        robot={key: ref for key in ("configuration", "kinematics", "joint_bounds", "base_registration", "controller_configuration")},
        controller={"kind": "policy_container", "configuration": ref, "model_or_asset_rights": ref},
        construction={"mode": "production_recipe", "recipe": ref},
        task={"binding_mode": "define_configuration_template", "definition": ref, "success_criteria": ref, "execution": ref,
              "subject": {"mode": "supplied_qualified_asset", "asset": ref, "physics_validation": ref, "rights_admission": ref}},
        runtime={"health_protocol": ref, "mounts": [{"mode": "read_only", "source": ref, "container_path": "/input"}, {"mode": "output", "container_path": "/out"}]},
        execution_adapter={"runtime_source_bundle": ref, "policy_observation_setup": {"appearance_asset": ref, "appearance_authoring_receipt": ref, "wrist_camera_mount_registry": ref}})


def test_all_configure_scene_supported_reference_paths_are_finite_and_literal():
    result = observe(record(request=configured_request()))
    paths = {row.contract_path for row in result.remote_raw_references}
    assert paths == {"scene.source_manifest", "scene.appearance.representation", "scene.appearance.renderer_qualification",
                     "scene.geometry.collision", "scene.geometry.validation", "scene.geometry.source_derivation",
                     "scene.registration.metric_registration", "scene.registration.support_plane", "scene.registration.robot_mount_interface",
                     "scene.registration.workspace_clearance", "scene.registration.camera_calibration", "scene.rights.admission",
                     "scene.rights.evidence.0.artifact", "scene.rights.evidence.1.artifact", "scene.website_native_inputs.runtime_inputs",
                     "scene.website_native_inputs.appearance", "scene.website_native_inputs.observations", "scene.website_native_inputs.candidate",
                     "scene.website_native_inputs.frames.0", "construction.recipe", "robot.configuration", "robot.kinematics",
                     "robot.joint_bounds", "robot.base_registration", "robot.controller_configuration", "controller.configuration",
                     "controller.model_or_asset_rights", "task.definition", "task.success_criteria", "task.execution",
                     "task.subject.asset", "task.subject.physics_validation", "task.subject.rights_admission", "sensors.configuration",
                     "runtime.health_protocol", "runtime.mounts.0.source", "execution_adapter.runtime_source_bundle",
                     "execution_adapter.policy_observation_setup.appearance_asset", "execution_adapter.policy_observation_setup.appearance_authoring_receipt",
                     "execution_adapter.policy_observation_setup.wrist_camera_mount_registry"}


@pytest.mark.parametrize("mode,extra", [("scene_configuration", {"native_probe": {}}),
                                      ("destination_qualification", {"native_probe": {}, "native_import_qualification": reference(), "geometry": reference()}),
                                      ("episode_evaluation", {"native_import_qualification": reference(), "geometry": reference(), "placement_qualification": reference()})])
def test_destination_reference_branches(mode, extra):
    request = preparation_request(run_mode=mode)
    request["task"]["destination"] = {"asset": reference(), "rights_admission": reference(), "static_qualification": reference(), **extra}
    result = observe(record(request=request))
    assert {row.contract_path for row in result.remote_raw_references if row.contract_path.startswith("task.destination")} == {
        "task.destination." + key for key in ("asset", "rights_admission", "static_qualification", *[key for key in extra if key != "native_probe"])}


@pytest.mark.parametrize("lane", ["native_task_arena_construction_after_destination", "native_task_arena_controls", "native_task_arena_zero_action", "native_task_arena_scripted_positive", "native_task_arena_policy_evaluation"])
def test_actual_predecessor_lanes_and_required_source_leaves(lane):
    leaves = {key: reference() for key in ("prior_authority", "prior_result", "prior_launch_receipt", "prior_webapp_sync", "prior_provider_zero", "prior_spend_reconciliation", "construction_result", "zero_action_result", "controls_qualification_manifest")}
    request = activation_request(lane=lane, lineage={"kind": "predecessor", **leaves})
    result = observe(record("activation", request=request, state="pending"))
    assert {row.contract_path for row in result.remote_raw_references} == {"release_window", *["lineage." + key for key in leaves]}
