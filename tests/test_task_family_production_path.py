"""CPU acceptance of the capture-to-native-family production path.

Reconstruction and HTTP, installed SDK, process and provider transports are
hermetic doubles. Intake, source registration, progression,
planning, bundle sealing/preflight, native factory dispatch, controls, scoring,
media and customer-safe publication execute their production implementations.
The fixture remains development-only and establishes no physics performance.
"""
from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest, cross_runtime_canonical_digest, cross_runtime_canonical_json,
)
from blueprint_pipeline.task_evaluation_scene_attempt_evidence import record
from blueprint_pipeline.task_evaluation_scene_intake import stage_scene_intent
from blueprint_pipeline.team_candidate_quick10_registry import REGISTRY_ENV, REGISTRY_SCHEMA
from tests.test_native_task_arena_bundle import _runtime_source_packet
from tests.test_task_evaluation_scene_configuration_submission import SHA, production_fixture
from tests.test_task_family_native_runtime import installed_sdk as sdk_fixture
from tests.test_task_family_preparation import _evaluation_inputs
from tests.test_task_family_policy_execution import _spec
from tests.test_website_task_family_packet import family_preparation


@pytest.fixture
def installed_sdk(tmp_path, monkeypatch):
    root = tmp_path / "sdk"
    root.mkdir()
    yield from sdk_fixture.__wrapped__(root, monkeypatch)


def production_source(tmp_path, monkeypatch, sdk):
    from blueprint_pipeline import task_evaluation_scene_progression as progression
    from blueprint_pipeline.task_evaluation_public_scene_attempt_factory import RELEASE_SCHEMA
    from blueprint_pipeline.task_evaluation_scene_intake import _read
    from blueprint_pipeline.team_scene_packet_worker import CAPTURE_PARTITION_ENV
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff

    capture = tmp_path / "partition/scenes/site-req1/captures/walkthrough-req1"
    capture.mkdir(parents=True)
    args, _old_preparation, _runtime_path = family_preparation(capture)
    context = args["task_context"]
    context["purpose"] = "scene_preparation"
    context["capture_rights"].update(consent_status="granted", consent_revoked=False,
        consent_scope=["robot_evaluation"], derived_scene_generation_allowed=True)
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    # Bind all scientific inputs to this source afresh, before the first
    # production handoff. Existing fixture compiler output is never adopted.
    original = args["evaluation_inputs"]
    evaluation = _evaluation_inputs(args)
    assets = {asset["asset_id"]: asset for asset in original["native_runtime_inputs"]["assets"]}
    evaluation["native_runtime_inputs"]["assets"] = [copy.deepcopy(assets[asset["asset_id"]])
        for asset in evaluation["native_runtime_inputs"]["assets"]]
    evaluation["asset_sources"] = original["asset_sources"]
    evaluation["task_family_runtime_selection"] = sdk.args["qualified_runtime_selection"]
    args["evaluation_inputs"] = evaluation
    descriptor = {"scene_id": context["scene_id"], "capture_id": context["capture_id"],
        "metadata": {"site_task_context": context, "native_task_evaluation": evaluation,
            "website_scene_execution_authority": {**args["spend"], "schema_version": "website_scene_sponsorship.v1"}}}
    write_json(capture / "capture_descriptor.json", descriptor)
    write_json(capture / "pipeline/website_task_context.json", context)
    removal_path = capture / "pipeline/removal.json"
    write_json(removal_path, args["removal_manifest"])
    atlas = {"status": "ready", "base_scene": args["base_scene"], "source_geometry": args["source_geometry"],
        "source_pose_digest": "sha256:" + "b" * 64, "background_pose_digest": "sha256:" + "c" * 64,
        "alignment": "anchored"}
    monkeypatch.setattr("blueprint_pipeline.website_atlas_scene_source.atlas_scene_source", lambda **_: atlas)
    monkeypatch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **kw: kw["task_masks"])
    bindings, intents = tmp_path / "bindings", tmp_path / "intents"
    monkeypatch.setenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", str(bindings))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(intents))
    monkeypatch.setenv(CAPTURE_PARTITION_ENV, str(tmp_path / "partition"))
    accepted = {}
    http = []

    def website_transport(*, capture_id, operation, payload):
        assert capture_id == context["capture_id"]
        http.append(operation)
        if operation == "task-context":
            return copy.deepcopy(context)
        assert operation == "prepared-scene"
        assert "task_success_contract" not in payload["request"]["task"]
        receipt = stage_scene_intent(value=json.loads(cross_runtime_canonical_json(payload["request"])),
            queue_root=intents, authenticated_client="blueprint-webapp", trusted_clients={"blueprint-webapp"}, now=args["now"])
        accepted.update(receipt)
        return {"id": receipt["intent_id"], "state": "forward_pending",
            "request_digest": cross_runtime_canonical_digest(payload["request"])}

    monkeypatch.setattr("blueprint_pipeline.website_task_context.website_webapp_request", website_transport)
    source_receipt = _runtime_source_packet(tmp_path / "operator-sdk-source")
    registry = {"schema_version": REGISTRY_SCHEMA, "profiles": [], "derived_profiles": {
        "root": str(tmp_path / "derived-profiles"), "defaults": {"allowed_founder_approvers": ["owner-1"],
            "max_hourly_rate_usd": "1.00", "open_loop_horizon": 1,
            "native_runtime_source_receipt": record(source_receipt)}}}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path = tmp_path / "operator-registry.json"
    write_json(registry_path, registry)
    registry_path.chmod(0o600)
    monkeypatch.setenv(REGISTRY_ENV, str(registry_path))
    monkeypatch.setenv("BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH", str(sdk.args["runtime_catalog_path"]))
    handoff = prepare_website_scene_handoff(descriptor=descriptor,
        clean_plate={"privacy_verified": True, "status": "objects_removed", "removal_manifest_path": str(removal_path),
            "task_masks": args["task_masks"]}, capture_root=capture, now=args["now"])
    assert handoff["status"] == "intake_ready", handoff
    assert accepted and handoff["source_registration"] and handoff["simulator_ready"] is False
    directory = intents / accepted["intent_id"]
    intent = _read(directory / "intent.json", "intent_digest")
    release_fixture = production_fixture(tmp_path / "release-fixture")
    machinery = {"schema_version": "task_evaluation_website_scene_machinery.v1", "maximum_preparation_spend_usd": 0,
        "provider": "vast"}
    machinery["machinery_digest"] = canonical_digest(machinery, digest_field="machinery_digest")
    machinery_path = tmp_path / "machinery.json"
    write_json(machinery_path, machinery)
    release = {"schema_version": RELEASE_SCHEMA, "source_commit": SHA, "runtime_digest": "sha256:" + "f" * 64,
        "repo_root": str(tmp_path / "repo"), "runtime_publication_root": str(release_fixture["runtime_publication_root"]),
        "namespace_timestamp": "20261027T120000Z", "release_admission_mode": "promoted",
        **{key: record(release_fixture[key]) for key in ("deploy_receipt", "release_provenance", "release_environment")}}
    release["release_digest"] = canonical_digest(release, digest_field="release_digest")
    config = {"intent_root": str(intents), "website_source_binding_root": str(bindings),
        "website_source_machinery_path": str(machinery_path), "factory_output_root": str(tmp_path / "factories")}

    def advance(*, paused=False):
        return progression._advance_intent(directory, intent,
            {**config, **({"paused_intent_ids": [accepted["intent_id"]]} if paused else {})}, release,
            resolver=None, publisher=lambda **_: pytest.fail("family entered rigid publication"),
            submitter=lambda **_: pytest.fail("family entered rigid submission"), status_reader=None,
            activation_provisioner=lambda **_: pytest.fail("CPU progression allocated resources"), now=args["now"])

    return SimpleNamespace(capture=capture, args=args, context=context, descriptor=descriptor,
        registry_path=registry_path, source_receipt=source_receipt, handoff=handoff, directory=directory,
        intent=intent, http=http, advance=advance)


def configure_candidate(world, sdk, tmp_path, monkeypatch):
    """Supply operator catalog/secret facts; production code decides admission."""
    from blueprint_pipeline.beta_policy_candidate_catalog import CATALOG_ENV, seal_catalog
    from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture, scene_facts

    profile = profile_for_capture(world.capture)
    facts = scene_facts(profile)
    scene = json.loads((Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json").read_text())
    spec = _spec(scene)
    spec.update(execution_authority="internal_policy_canary_unqualified", claim_ceiling="diagnostic_policy_execution",
        ranking_permitted=False, qualification_permitted=False, scene_promotion_permitted=False)
    spec["policy_spec"].update(camera_id="camera1", image_height=16, image_width=16)
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    spec_path = tmp_path / "operator-policy.json"
    write_json(spec_path, spec)
    secret = tmp_path / "operator-policy-token"
    secret.write_text("hermetic-native-family-token")
    secret.chmod(0o600)
    policy = spec["policy_spec"]
    configured = {"candidate_id": spec["candidate_id"], "display_name": "Hermetic qualified family policy",
        "policy": {"kind": "hosted", "source": policy["allowed_origin"], "revision": "hermetic-release",
            "digest": spec["checkpoint_digest"], "digest_kind": "provider_metadata"},
        "interface": {"policy_interface_id": spec["policy_interface_id"],
            "robot_profile_id": policy["embodiment_id"], "observation_profile_id": policy["observation_schema_id"],
            "action_schema_id": policy["action_schema_id"], "controller": policy["controller"],
            "action_bounds": policy["action_bounds"], "control_hz": spec["control_frequency_hz"],
            "action_horizon": 1, "execution_horizon": 1, "camera_roles": ["camera1"],
            "image_height": 16, "image_width": 16, "image_encoding": "rgb_uint8", "image_preprocessing": "identity",
            "history_frames": 1, "normalization_digest": None},
        "runtime": {"backend": spec["evaluation_backend"], "runtime_id": sdk.registration["runtime_id"],
            "minimum_gpu_memory_gb": 1}, "supported_task_kinds": ["button_press"],
        "eligibility": {"enabled": True, "commercial_use": "allowed", "license_id": "hermetic-test-only",
            "license_note": "CPU software acceptance fixture; no physics qualification", "requires_credentials": [],
            "blockers": []}}
    catalog = seal_catalog({"schema_version": "beta_policy_candidate_catalog.v1", "catalog_id": "family-cpu-acceptance",
        "revision": "test", "candidates": [configured]})
    catalog_path = tmp_path / "operator-policy-catalog.json"
    write_json(catalog_path, catalog)
    catalog_path.chmod(0o600)
    monkeypatch.setenv(CATALOG_ENV, str(catalog_path))
    registry = json.loads(world.registry_path.read_text())
    registry["derived_profiles"]["defaults"]["screening_candidates"] = {spec["candidate_id"]: {
        "policy_execution_spec": record(spec_path), "checkpoint_inventory": None, "credential_file": str(secret),
        "credential_origin": policy["allowed_origin"], "provider_hard_cap_usd": "4.00",
        "hosted_inference_maximum_usd": "0.00", "hosted_inference_payment": "free", "hosted_max_cost_per_request_usd": "0"}}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    write_json(world.registry_path, registry)
    world.registry_path.chmod(0o600)
    return catalog, spec, profile_for_capture(world.capture)


@pytest.mark.slow
@pytest.mark.parametrize("refusal", ["owner_pause", "owner_revocation", "capture_withdrawal"])
def test_actual_family_source_controls_refuse_before_native_factory_policy_or_allocator(
    installed_sdk, tmp_path, monkeypatch, refusal,
):
    from blueprint_pipeline.beta_policy_screening_family_sources import enqueue_registered_family_scenes

    sdk = installed_sdk
    monkeypatch.setattr("blueprint_pipeline.team_candidate_quick10_launch._start_allocator",
        lambda *_, **__: pytest.fail("source refusal reached allocator"))
    monkeypatch.setattr("blueprint_pipeline.team_candidate_policy_client.canary_policy_client",
        lambda *_, **__: pytest.fail("source refusal loaded candidate"))
    world = production_source(tmp_path, monkeypatch, sdk)
    if refusal == "owner_pause":
        held = world.advance(paused=True)
        assert held["phase"] == "paused"
        assert not list(world.directory.glob("preparation-attempts/*.json"))
    else:
        prepared = world.advance()
        assert prepared["status"] == "completed" and prepared["phase"] == "family_native_prepared", prepared
        if refusal == "owner_revocation":
            write_json(world.directory / "revoked.json", {"revoked": True})
            expected = "scene_configuration_submission_scene_owner_authority_revoked"
        else:
            tombstone = {"schema_version": "website_capture_withdrawal_tombstone.v1",
                "scene_id": world.context["scene_id"], "request_id": world.context["request_id"]}
            tombstone["digest"] = canonical_digest(tombstone, digest_field="digest")
            write_json(world.capture.parent.parent / "website_withdrawal/tombstone.json", tombstone)
            expected = "team_scene_capture_withdrawn"
        result = enqueue_registered_family_scenes(queue_root=tmp_path / "screening")
        assert len(result["sources"]) == 1 and result["sources"][0]["status"] == "blocked", result
        assert result["sources"][0]["blockers"] == [expected]
    assert not sdk.calls


@pytest.mark.slow
@pytest.mark.parametrize("policy_action,expected_success", [(1.0, True), (0.0, False)])
@pytest.mark.parametrize("controls_paused", [False, True])
def test_actual_family_capture_to_native_screening_publication(
    installed_sdk, tmp_path, monkeypatch, policy_action, expected_success, controls_paused,
):
    from blueprint_pipeline import beta_policy_screening_launch as launch
    from blueprint_pipeline.beta_policy_screening_family_sources import enqueue_registered_family_scenes
    from blueprint_pipeline.task_evaluation_scene_intake import scene_intent_status
    from blueprint_pipeline import task_evaluation_control_stage_policy

    sdk = installed_sdk
    if controls_paused:
        assert task_evaluation_control_stage_policy.CONTROLS_PAUSED is True
        engine = sdk.module().Environment
        monkeypatch.delattr(engine, "scripted_control_action")
        unactuated_step = engine.step_without_robot_action

        def robot_free_only(self, **kwargs):
            assert self.robot_present is False, "paused zero-action control ran"
            return unactuated_step(self, **kwargs)

        monkeypatch.setattr(engine, "step_without_robot_action", robot_free_only)
    else:
        monkeypatch.setattr(task_evaluation_control_stage_policy, "CONTROLS_PAUSED", False)
    world = production_source(tmp_path, monkeypatch, sdk)
    prepared_source = world.advance()
    assert prepared_source["status"] == "completed" and prepared_source["phase"] == "family_native_prepared", prepared_source
    catalog, _specification, profile = configure_candidate(world, sdk, tmp_path, monkeypatch)
    status = scene_intent_status(queue_root=world.directory.parent, intent_id=world.intent["intent_id"], now=world.args["now"])
    source_inputs = json.loads(Path(profile["family_scene_inputs_path"]).read_text())
    assert source_inputs["source"]["scene_revision_digest"] == status["result_reference"]["digest"]
    queue = tmp_path / "screening"
    adopted = enqueue_registered_family_scenes(queue_root=queue)
    assert len(adopted["sources"]) == 1 and adopted["sources"][0]["status"] != "blocked", adopted
    plans = list((queue / "plans").glob("*.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text())
    assert plan["catalog_digest"] == catalog["catalog_digest"]
    assert plan["pre_policy_site_qualification_required"] is True and plan["scenario_scope"] == "native_frozen_reset_seeds"
    assert len(plan["candidates"]) == 1 and plan["candidates"][0]["status"] == "selected", plan["candidates"]
    candidate = plan["candidates"][0]
    assert len(candidate["scenario_cells"]) == candidate["episode_count"] == 10
    assert all(cell["resolved_scenario"]["parameters"] == {} for cell in candidate["scenario_cells"])
    prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    assert not sdk.calls
    bundle = json.loads(Path(prepared["bundle_receipt"]["path"]).read_text())
    assert json.loads((tmp_path / "attempt/static_preflight.json").read_text())["status"] == "passed"
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
        assert manifest["evaluation_backend"] == "task_family_native.v1"
        assert all(b"hermetic-native-family-token" not in archive.read(name) for name in archive.namelist())
        archive.extractall(tmp_path / "extracted")
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()

    # Retain the genuine client/secret loader. Replace only its HTTP wire with
    # an authenticated bounded policy service that returns native actions.
    from blueprint_pipeline.task_family_policy_https_client import TaskFamilyPolicyHttpsClient
    from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
    native_spec = json.loads(Path(candidate["execution_spec"]["policy_execution_spec"]["path"]).read_text())
    monkeypatch.setenv(native_spec["policy_endpoint"]["credential_env"], candidate["execution_spec"]["credential_file"])
    requests = []

    def policy_http(client, *, url, method, body, headers):
        assert headers["Authorization"] == "Bearer hermetic-native-family-token"
        assert url in {client._spec.endpoint_url, client._spec.metadata_url}
        requests.append({"url": url, "method": method, "body": json.loads(body) if body else None})
        if method == "GET":
            return json.dumps(client._spec.expected_metadata()).encode()
        assert set(requests[-1]["body"]["observation"]) == {"rgb", "proprioception", "prompt"}
        assert "asset_states" not in json.dumps(requests[-1]["body"])
        return json.dumps({"actions": [[policy_action]]}).encode()

    monkeypatch.setattr(TaskFamilyPolicyHttpsClient, "_bounded_https", policy_http)
    runtime = tmp_path / "extracted/provider_runtime"
    output = tmp_path / "native-session"

    # The real beta worker also requires the controller's exact current-source
    # observation. Deliver it through the production GPU-local receiver; the
    # policy/engine transport doubles do not replace this authority check.
    from blueprint_pipeline import beta_screening_runtime_control as control
    from tests.test_beta_policy_screening_control_contracts import response

    authority = json.loads(Path(prepared["authority"]["path"]).read_text())
    invocation = {
        "prepared_digest": prepared["prepared_digest"], "plan_digest": plan["plan_digest"],
        "candidate_id": candidate["candidate_id"], "configuration_digest": candidate["configuration_digest"],
        "authority_digest": authority["authority_digest"], "resource_name": authority["resource_name"],
        "bundle_sha256": bundle["bundle_sha256"], "provider": "vast", "attempt_id": "attempt",
    }
    binding = {key: invocation[key] for key in (
        "prepared_digest", "plan_digest", "candidate_id", "configuration_digest", "authority_digest",
        "resource_name", "attempt_id",
    )}
    binding.update(campaign_id=plan["campaign_id"], site_id=plan["site_id"],
        source=plan["source"], invocation_id=canonical_digest(invocation))
    now = int(world.args["now"])
    monkeypatch.setattr(control.time, "time", lambda: now)
    observation = response(scope=binding, now=now,
        deadline=min(now + authority["hard_ttl_seconds"], int(world.args["spend"]["expires_at_epoch"])))
    delivery = control.make_delivery(authority=authority, binding=binding, control=observation)
    assert control.install_runtime_control(runtime_root=runtime, delivery=delivery) == delivery

    def run_cpu_cell(*, index, runtime_root, output_root, child_root):
        # Process isolation itself is a transport seam: run the unchanged cell
        # worker in this CPU interpreter so the installed SDK double is shared.
        return worker._run_selected_cell(index, runtime_root=runtime_root, output_root=child_root,
            provider_output_root=output_root)

    assert worker._run_isolated_cell_processes(runtime_root=runtime, output_root=output,
        run_cell_process=run_cpu_cell) == 0
    inner = json.loads((output / worker.PROVIDER_RESULT_FILENAME).read_text())
    runtimes_per_cell = 2 if controls_paused else 4
    assert len(inner["episodes"]) == 10 and len(sdk.calls) == 10 * runtimes_per_cell
    assert len([request for request in requests if request["method"] == "POST"]) == 20
    assert all(engine.close_count == 1 for engine in sdk.module().instances)
    source_native = json.loads((runtime / "native_task_packet/native_task_runtime_contract.v1.json").read_text())
    assert native_spec["native_runtime_contract_digest"] == source_native["contract_digest"]
    scenario_fields = {"seed", "scenario_cell_id", "scenario_instance_digest", "contract_digest"}
    source_science = {key: value for key, value in source_native.items() if key not in scenario_fields}
    for engine in sdk.module().instances:
        assert {key: value for key, value in engine.contract.items() if key not in scenario_fields} == source_science
        assert engine.contract["task_state_binding"]["source_scene_binding"]["task_context_digest"] == world.context["context_digest"]
        for asset in engine.contract["assets"]:
            path = Path(engine.asset_paths[asset["asset_id"]])
            assert path.resolve().is_relative_to((runtime / "native_task_packet").resolve())
            assert record(path)["sha256"] == asset["content_digest"]
    expected_cells = {cell["cell_id"]: cell for cell in candidate["scenario_cells"]}
    for row in inner["episodes"]:
        cell = expected_cells[row["cell_id"]]
        assert row["seed"] == cell["seed"] and row["resolved_scenario_digest"] == cell["resolved_scenario_digest"]
        receipt = row["episode"]["native_family_receipt"]
        assert receipt["task_succeeded"] is expected_success and receipt["physical_validation"] is False
        assert receipt["native_provenance"]["scenario_cell_id"] == cell["cell_id"]
        assert receipt["native_provenance"]["scenario_instance_digest"] == cell["resolved_scenario_digest"]
        assert receipt["native_provenance"]["seed"] == cell["seed"]
        assert receipt["telemetry"]["samples"][-1]["measurements"]["stage1"]["button_travel_m"] == .04 * policy_action
        assert receipt["visual_evidence"]["status"] == "complete"
        assert row["arm_moved"] is False and row["actions_reached_robot"] is True
        assert row["episode"]["motion_evidence"]["policy_state_changed"] is expected_success
        qualification = row["episode"]["native_site_qualification"]
        assert qualification["controls_qualified"] is not controls_paused
        assert qualification["qualified_comparison_permitted"] is False
        assert set(qualification["controls"]) == ({"robot_free_probe"} if controls_paused else
            {"robot_free_probe", "zero_action_negative", "deterministic_scripted_positive"})
        if controls_paused:
            assert qualification["control_omission_authority"]["qualified_comparison_permitted"] is False
            assert qualification["control_omission_authority"]["source_task_success_contract_digest"] == plan["task_success_contract_digest"]
    assert all(sdk.module().instances[index].actions == [[policy_action], [policy_action]]
               for index in range(runtimes_per_cell - 1, 10 * runtimes_per_cell, runtimes_per_cell))
    assert all(sdk.module().instances[index].actions == []
               for index in range(0, 10 * runtimes_per_cell, runtimes_per_cell))

    # The provider transport is also hermetic. Exercise the actual scoped
    # closeout join against that double, without inventing a closed native
    # session or contacting a provider.
    from blueprint_pipeline import beta_policy_screening_closeout as closeout
    from blueprint_pipeline.native_task_arena_policy_canary_session import consume_session_authority_once
    from tests.test_beta_policy_screening_closeout import Provider
    authority = json.loads(Path(prepared["authority"]["path"]).read_text())
    attempt = tmp_path / "attempt"
    raw_adapter = attempt / "hermetic_provider_adapter.json"
    teardown = attempt / "hermetic_provider_teardown.json"
    write_json(raw_adapter, {"vast_instance_ids": [7001]})
    write_json(teardown, {"vast_instance_ids": [7001]})
    adapter = {"vast_instance_ids": [7001], "bundle_sha256": bundle["bundle_sha256"],
        "authorization_consumption": consume_session_authority_once(authority, consumption_path=attempt / "consumption.json"),
        "continuing_spend_from_this_run": False, "native_control_result_digest": inner["result_digest"],
        "provider_closeout": {"provider_zero_confirmed": True, "warm_session_retained": False,
            "all_staged_objects_absent": True, "adapter_result": record(raw_adapter), "teardown_manifest": record(teardown)}}
    closeout_args = {"adapter": adapter, "prepared": prepared, "adapter_path": attempt / "allocator_result.json"}
    absence = closeout.collect_scoped_closeout(**closeout_args, provider=Provider(resources=[]))
    result = closeout.join_scoped_session_closeout(inner=inner, receipt=absence, **closeout_args)
    from blueprint_pipeline.native_task_arena_policy_canary_session import validate_session_result
    for forgery in ("missing_site_qualification", "forged_success_telemetry"):
        invalid = copy.deepcopy(result)
        episode = invalid["episodes"][0]["episode"]
        if forgery == "missing_site_qualification":
            episode.pop("native_site_qualification")
        else:
            native_receipt = episode["native_family_receipt"]
            terminal = native_receipt["telemetry"]["samples"][-1]["measurements"]["stage1"]
            terminal.update(button_travel_m=1000.0, button_activated=True)
            native_receipt["task_succeeded"] = True
            episode["score"]["task_succeeded"] = True
        # A new outer seal cannot replace native receipts, independent score
        # replay or the required site controls; no additional SDK run is used.
        invalid["result_digest"] = canonical_digest(invalid, digest_field="result_digest")
        with pytest.raises(ValueError, match="episode"):
            validate_session_result(invalid)
    from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary, screening_publication
    summary = candidate_result_summary(result, plan=plan, candidate=candidate)
    assert summary["episodes_scored"] == 10 and summary["episodes_void"] == 0
    assert summary["episodes_succeeded"] == (10 if expected_success else 0)
    assert summary["episodes_failed"] == (0 if expected_success else 10)
    state = {"revision": 2, "updated_at": "2026-10-27T12:00:00Z", "candidates": {
        candidate["candidate_id"]: {"status": "completed", "summary": summary, "blockers": []}}}
    publication = screening_publication(plan=plan, state=state,
        budget={"currency": "USD", "limit_usd": "15", "committed_upper_bound_usd": "4",
            "spent_upper_bound_usd": "4", "reserved_usd": "0"})
    assert publication["candidates"][0]["episodes_scored"] == 10
    assert publication["candidates"][0]["success_rate"] == (1.0 if expected_success else 0.0)
    assert publication["source"] == plan["source"] and "recommendation_eligible" not in publication
    assert publication["claim_ceiling"] == "simulation_estimate" and publication["purpose"] == "job_evaluation"
    from jsonschema import Draft202012Validator
    schema_path = Path(__file__).resolve().parents[1] / "docs/schemas/beta_policy_screening_publication.v2.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text())).validate(publication)
    assert publication["publication_digest"] == cross_runtime_canonical_digest(publication, digest_field="publication_digest")
    wire = json.dumps(publication)
    assert all(secret not in wire for secret in ("hermetic-native-family-token", "policy.example", str(tmp_path), "task_success_contract"))
