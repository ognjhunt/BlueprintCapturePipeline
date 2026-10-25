from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import zipfile

import pytest

from blueprint_pipeline import adp009d_franka_vast as transport
from blueprint_pipeline import task_evaluation_scene_configuration_native_continuation as continuation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import execute_scene_configuration_stage_chain
from blueprint_pipeline.provider_runtime_bundle_contract import provider_runtime_contract_blockers
from tests.test_task_evaluation_scene_configuration_provider_runtime import _astra_inputs, _registry, _producers


def prefix(tmp_path):
    envelope, configurations = _astra_inputs(tmp_path)
    envelope["expected_production_commit"] = "a" * 40
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    output = tmp_path / "original" / "runtime_output"
    (output / "stages").mkdir(parents=True)
    deadline = time.time() + 36000
    chain = execute_scene_configuration_stage_chain(
        envelope=envelope, configurations=configurations, output_root=output / "stages",
        registry=_registry([], real_artifacts=True), producer_registry=_producers(),
        stage_limit="stage-4", parent_deadline_epoch=deadline,
    )
    capsule = tmp_path / "prefix.zip"
    sealed = continuation.seal_existing_native_prefix(output_root=output, envelope=envelope,
        configurations=configurations, capsule_path=capsule)
    return envelope, configurations, output, capsule, sealed, chain, deadline


def test_seal_existing_prefix_is_read_only_and_preserves_its_deadline(tmp_path):
    envelope, _, output, capsule, sealed, _, deadline = prefix(tmp_path)
    before = (output / "stages/astra_same_run_resume_binding.json").read_bytes()
    assert continuation.validate_native_prefix(capsule, envelope) == sealed["transport"]
    assert sealed["transport"]["prefix_deadline_epoch"] == deadline
    assert (output / "stages/astra_same_run_resume_binding.json").read_bytes() == before
    assert not (output / "stages/astra_prefix_continuation_binding.json").exists()
    assert sealed["transport"]["new_paid_allocation_authorized"] is False


def test_completed_prefix_can_reuse_read_only_input_bytes_without_breaking_its_checkpoints(tmp_path):
    envelope, configurations, output, _, _, chain, _ = prefix(tmp_path)
    cas = tmp_path / "cas"
    cas.mkdir()
    target = output / "stages/stage-1/adapter/artifact.json"
    donor = cas / continuation._sha(target).removeprefix("sha256:")
    shutil.copyfile(target, donor)
    donor.chmod(0o440)
    before = target.read_bytes()
    reused = continuation.share_completed_prefix_inputs(output_root=output, envelope=envelope,
        configurations=configurations, content_addressed_root=cas)
    assert reused["logical_bytes_preserved"] is True
    assert target.samefile(donor) and target.read_bytes() == before
    sealed = continuation.seal_existing_native_prefix(output_root=output, envelope=envelope,
        configurations=configurations, capsule_path=tmp_path / "shared-prefix.zip")
    assert sealed["transport"]["completed_stage_ids"] == [f"stage-{i}" for i in range(1, 5)]
    assert len(chain["stage_result_digests"]) == 4


@pytest.mark.parametrize("fault", ["artifact", "checkpoint", "binding", "envelope"])
def test_changed_prefix_is_rejected_before_a_native_runtime_can_start(tmp_path, fault):
    envelope, _, _, capsule, _, _, _ = prefix(tmp_path)
    with zipfile.ZipFile(capsule) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if fault == "artifact":
        members["stages/stage-1/adapter/artifact.json"] = b"changed bytes"
    elif fault == "checkpoint":
        name = "stages/stage-4/completed_stage_checkpoint.json"
        value = json.loads(members[name])
        value["stage_result"]["paid_execution_requested"] = True
        members[name] = json.dumps(value).encode()
    elif fault == "binding":
        name = "stages/astra_same_run_resume_binding.json"
        value = json.loads(members[name])
        value["parent_deadline_epoch"] += 1
        members[name] = json.dumps(value).encode()
    else:
        envelope["run_id"] = "another run"
    capsule.chmod(0o640)
    with zipfile.ZipFile(capsule, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError):
        continuation.validate_native_prefix(capsule, envelope)


def test_exact_bundle_rehearsal_never_restores_over_the_controller_prefix(tmp_path, monkeypatch):
    envelope, configurations, output, capsule, _, chain, _ = prefix(tmp_path)
    source = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(continuation, "_source_root", lambda commit: source)
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(json.dumps(envelope))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"schema_version": continuation.INPUT_SCHEMA,
        "capsule_path": str(capsule), "envelope_path": str(envelope_path),
        "configuration_paths": {name: str(path) for name, (_, path) in configurations.items()},
        "provider_disclosure_authorized": True, "claim_ceiling": "development_only"}))
    receipt = continuation.build_native_continuation_bundle(job_dir=tmp_path / "bundle",
        inputs_path=inputs, implementation_commit="b" * 40)
    assert receipt["exact_bundle_entrypoint_rehearsal"]["status"] == "passed"
    assert receipt["private_data_uploaded"] is True
    assert receipt["model_calls_performed"] == 0
    assert not (tmp_path / "bundle/provider_runtime").exists()
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(tmp_path / "extracted")
    runtime = tmp_path / "extracted/provider_runtime"
    assert provider_runtime_contract_blockers(provider_bundle_kind=continuation.BUNDLE_KIND,
        entrypoint_text=(runtime / "run_adp_arena_provider_runtime.sh").read_text(),
        runner_text=Path(continuation.__file__).read_text()) == []
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
    assert _blueprint_bundle_preflight(job_dir=tmp_path / "preflight", generated_at="fixed",
        enable_blueprint_bundle=True, enable_isaac_smoke=True, provider_bundle_kind=continuation.BUNDLE_KIND,
        bundle_path=Path(receipt["bundle_path"]), provider_bundle_url="https://example.com/bundle.zip",
        provider_output_put_url="https://example.com/output.zip")["blockers"] == []
    assert continuation.build_native_continuation_bundle(job_dir=tmp_path / "bundle",
        inputs_path=inputs, implementation_commit="b" * 40) == receipt
    assert not (output / "stages/stage-5").exists()
    assert not (output / "stages/astra_prefix_continuation_binding.json").exists()

    # Rehearse the container restore and real checkpoint adoption at the exact
    # original paths, with only the two remaining adapters stubbed (no Isaac).
    shutil.move(output, tmp_path / "retained-controller-prefix")
    from blueprint_pipeline import task_evaluation_scene_configuration_provider_runtime as core
    observed = ["stage-1", "stage-2", "stage-3", "stage-4"]
    original = core.execute_scene_configuration_stage_chain
    def execute(**kwargs):
        kwargs["registry"] = _registry(observed)
        kwargs["producer_registry"] = _producers()
        return original(**kwargs)
    monkeypatch.setattr(core, "execute_scene_configuration_stage_chain", execute)
    assert continuation.main(["--runtime-root", str(runtime), "--output-root", str(tmp_path / "returned")]) == 0
    result = json.loads((tmp_path / "returned" / continuation.OUTPUT).read_text())
    assert result["status"] == "completed"
    assert result["stage_chain"]["stage_result_digests"][:4] == chain["stage_result_digests"]
    assert observed == [f"stage-{i}" for i in range(1, 7)]
    with zipfile.ZipFile(tmp_path / "returned/native_continuation_stage_output.zip") as archive:
        assert all(not name.startswith("stages/stage-1/") for name in archive.namelist())


def test_fixed_producer_drives_the_real_native_component_and_verifies_its_bytes(tmp_path, monkeypatch):
    from tests.test_task_evaluation_scene_configuration_native_import_driver import _environment, _native_runner, _observed
    from blueprint_pipeline import task_evaluation_scene_configuration_native_import_driver as driver
    environment = _environment(tmp_path)
    stage_input = json.loads(Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"]).read_text())
    stage = stage_input["stage"]
    stage.update(capability="replacement_native_import_qualification", execution_class="gpu_canary")
    config = tmp_path / "native-configuration.json"
    config.write_text(json.dumps(stage_input["configuration"]))
    deps = json.loads(Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"]).read_text())
    deps[0].update(schema_version="task_evaluation_scene_configuration_stage_result.v1", status="completed")
    deps[0]["stage_result_digest"] = canonical_digest(deps[0], digest_field="stage_result_digest")
    envelope = {"run_id": "retained-run", "expected_production_commit": "a" * 40}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    original = driver.execute_native_import_component
    monkeypatch.setattr(driver, "execute_native_import_component", lambda **kwargs:
        original(**kwargs, native_runner=_native_runner(_observed())))
    root = tmp_path / "native-producer"
    root.mkdir()
    artifacts = continuation.native_import_producer_registry(implementation_commit="b" * 40).execute(
        stage=stage, envelope=envelope, configuration=stage_input["configuration"], configuration_path=config,
        dependency_results=deps, output_root=root)
    assert {row["role"] for row in artifacts} == {"native_import_runtime_result"}
    actual = json.loads(Path(artifacts[0]["path"]).read_text())
    assert actual["deterministic_reset_state_digest_repeat_count"] == 3


def test_native_continuation_uses_the_pinned_runtime_driver_and_normal_paid_transport(tmp_path, monkeypatch):
    from blueprint_pipeline.public_scene_simready_isaac_bundle import DEFAULT_IMAGE
    observed = {}
    def observe(**kwargs):
        from blueprint_pipeline.isaac_driver_support import resolve_max_isaac_driver
        observed.update(kwargs)
        observed["driver_ceiling_inside_call"] = resolve_max_isaac_driver()
        return {}
    monkeypatch.setattr(transport, "run_arena_native_control_vast", observe)
    receipt = {"diagnostic_kind": continuation.BUNDLE_KIND, "container_image": DEFAULT_IMAGE,
        "minimum_driver_version": continuation.MINIMUM_DRIVER_VERSION,
        "maximum_driver_exclusive": continuation.MAXIMUM_DRIVER_EXCLUSIVE, "candidate_policy_queried": False,
        "policy_candidate_id": None, "model_calls_performed": 0, "generated_tool_runtime_used": False,
        "provider_disclosure_authorized": True, "claim_ceiling": "development_only"}
    transport.run_adp009d_native_microcheck_vast(job_dir=tmp_path, prepared_bundle=receipt,
        paid_resource_admission_grant=None, execute=False)
    assert observed["container_image"] == DEFAULT_IMAGE
    assert observed["minimum_driver_version"] == "595.58.03"
    assert observed["min_compute_cap"] == 890
    assert observed["driver_ceiling_inside_call"] == (596, 0, 0)
    from blueprint_pipeline.isaac_driver_support import resolve_max_isaac_driver
    assert resolve_max_isaac_driver() == (581, 0, 0)
    assert observed["provider_bundle_kind"] == continuation.BUNDLE_KIND
    assert observed["require_independent_watchdog"] is True
    assert observed["candidate_policy_query_expected"] is False
    with pytest.raises(ValueError, match="transport_contract_invalid"):
        transport.run_adp009d_native_microcheck_vast(job_dir=tmp_path,
            prepared_bundle={**receipt, "candidate_policy_queried": True},
            paid_resource_admission_grant=None, execute=True)


def test_native_continuation_has_complete_vast_script_environment_and_recovery(tmp_path):
    from blueprint_pipeline import vast_provider_adapter as vast
    from blueprint_pipeline.vast_provider_output_recovery import _REMOTE_ARCHIVE_BY_BUNDLE_KIND
    kind = continuation.BUNDLE_KIND
    assert vast._is_isaac_provider_bundle(kind)
    assert vast._provider_expected_video_count(kind) == 0
    assert vast._resolve_launch_mode(requested="auto", enable_isaac_smoke=False,
        enable_blueprint_bundle=True, provider_bundle_kind=kind) == "ssh_direct"
    assert vast._probe_env(job_dir=tmp_path, enable_isaac_smoke=False,
        provider_bundle_kind=kind, forward_hf_token=False)["ACCEPT_EULA"] == "Y"
    script = vast._probe_shell_script("https://example.com", enable_isaac_smoke=True,
        enable_blueprint_bundle=True, provider_bundle_kind=kind)
    assert "run_adp_arena_provider_runtime.sh" in script
    assert "adp_arena_provider_runtime_output.zip" in script
    assert "BLUEPRINT_VAST_CUDA_RUNTIME_DEFERRED_TO_ISAAC_SIMULATION_APP" in script
    assert _REMOTE_ARCHIVE_BY_BUNDLE_KIND[kind] == "/workspace/adp_arena_provider_runtime_output.zip"


def test_the_new_runtime_admits_its_supported_driver_without_widening_the_older_runtime():
    from blueprint_pipeline.vast_provider_adapter import _select_offer
    offer = {"ask_contract_id": 7, "machine_id": 1, "gpu_name": "RTX 4000 Ada", "gpu_ram": 20480,
        "compute_cap": 890, "num_gpus": 1, "rentable": True, "dph_total": .20,
        "driver_version": "595.91.07", "storage_cost": .20, "disk_space": 224}
    kwargs = {"max_hourly_rate": .70, "min_gpu_ram_mb": 16000, "min_compute_cap": 890,
        "max_compute_cap": 900, "minimum_driver_version": "595.58.03",
        "require_known_supported_isaac_driver": True}
    assert _select_offer([offer], **kwargs) is None
    with transport._native_continuation_driver_scope(True):
        assert _select_offer([offer], **kwargs) is not None
        assert _select_offer([{**offer, "driver_version": "596.01.00"}], **kwargs) is None
        assert _select_offer([{**offer, "driver_version": "580.159.04"}], **kwargs) is None
    assert _select_offer([offer], **kwargs) is None


@pytest.mark.parametrize("execute", [False, True])
def test_native_continuation_cli_preserves_the_canonical_paid_admission(tmp_path, monkeypatch, execute):
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
    from tests.test_adp009d_native_microcheck_bundle import _allocator_args
    observed = {}
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers", lambda:
        ([], {"orchestrator_source_commit": "b" * 40, "checkout_clean": True}))
    monkeypatch.setattr(continuation, "build_native_continuation_bundle", lambda **kwargs:
        {"status": "ready", "bundle_sha256": "sha256:" + "c" * 64, "input_digest": "sha256:" + "d" * 64,
         "diagnostic_kind": continuation.BUNDLE_KIND})
    monkeypatch.setattr(allocator, "run_adp009d_native_microcheck_vast", lambda **kwargs:
        observed.update(kwargs) or {"status": "completed" if execute else "dry_run_ready"})
    argv = _allocator_args(tmp_path, execute=execute)
    for key in ("--adp009d-approved-can", "--adp009d-sage-collision", "--adp009d-harness-manifest"):
        i = argv.index(key)
        del argv[i:i+2]
    argv.extend(["--adp009d-diagnostic-only", "--adp009d-native-continuation-inputs", str(tmp_path / "inputs.json")])
    assert allocator.main(argv) == 0
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is execute
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["private_data_uploaded"] is True
    assert admission["allocation_binding"]["native_continuation_requested"] is True
    assert admission["allocation_binding"]["retry_cap"] == 0
