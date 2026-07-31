from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_worker_contracts import (
    FAILURE_CODES,
    PINNED_MODEL_ASSETS,
    PINNED_WORKER_COMPONENTS,
    ReconstructionWorkerContractError,
    build_checkpoint_manifest,
    build_pose_estimation_request,
    build_pose_estimation_result,
    build_training_request,
    build_training_result,
    build_worker_build_receipt,
    build_worker_smoke_receipt,
    build_worker_stack_manifest,
)
from blueprint_pipeline.reconstruction_worker_build_packet import (
    ALLOCATOR_ENTRYPOINT,
    prepare_reconstruction_worker_build_packet,
)
from blueprint_pipeline.task_evaluation_supervisor.capabilities import SupervisorContext
from blueprint_pipeline.task_evaluation_supervisor.contracts import AutonomyMode
from blueprint_pipeline.task_evaluation_supervisor.supervisor import (
    default_authority_envelope,
)
from blueprint_pipeline.task_evaluation_supervisor.tools import (
    ToolRegistry,
    non_spend_tool_bindings,
    validate_tool_observation_binding,
)


D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
D3 = "sha256:" + "3" * 64
D4 = "sha256:" + "4" * 64
D5 = "sha256:" + "5" * 64
SHA = "a" * 40
IMAGE = "registry.example/blueprint/reconstruction@sha256:" + "b" * 64
RECORDED_PROXY_PATH = (
    Path(__file__).parents[1]
    / "docs/evidence/arkitscenes_raw_proxy_40958756_b2d7297f.json"
)
RECORDED_WORKER_STACK_PATH = (
    Path(__file__).parents[1]
    / "docs/evidence/reconstruction_worker_stack_manifest_ff9deb59.json"
)
RECORDED_WORKER_ADMISSION_PATH = (
    Path(__file__).parents[1]
    / "docs/evidence/reconstruction_worker_build_admission_ff9deb59.json"
)


def _common(**overrides):
    value = {
        "stable_run_identity": "reconstruction-run-001",
        "source_capture_identity": "capture-001",
        "source_capture_digest": D1,
        "original_file_references": [{"artifact_id": "capture.mov", "digest": D2}],
        "producing_method": "blueprint-worker-contract-test",
        "implementation_version": "1.0.0",
        "container_image_digest": IMAGE,
        "source_commit_sha": SHA,
        "deterministic_configuration_digest": D3,
        "input_digests": [{"artifact_id": "dataset", "digest": D4}],
        "output_digests": [],
        "train_heldout_split_digest": D5,
        "camera_calibration_binding": {"calibration_digest": D2},
        "coordinate_frame_declaration": {"frame": "world", "handedness": "right"},
        "units": "meters",
        "metric_scale_status": "sensor_metric_unvalidated",
        "provider_runtime_identity": {"provider": "local", "runtime": "fixture"},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {"mode": "execute_non_spend", "authority_id": "fixture"},
        "warnings": [],
        "blockers": [],
        "proof_effect": "none",
        "claim_ceiling": "request_only",
        "parent_artifact_or_event": {"capture_build_digest": D1},
        "timestamp": "2026-07-30T12:00:00Z",
    }
    value.update(overrides)
    return value


def _worker_manifest():
    return build_worker_stack_manifest(
        {
            "worker_family": "blueprint-reconstruction-worker",
            "runnable_platform": "linux/amd64",
            "headless_required": True,
            "display_required": False,
            "source_commit_sha": SHA,
            "qualification_status": "candidate_unbuilt",
            "minimum_vram_gb": 24,
            "supported_compute_capabilities": [75, 80, 86, 89],
            "tested_driver_range": {"status": "not_yet_tested"},
            "model_assets": list(PINNED_MODEL_ASSETS),
            "hidden_heldout_access": False,
            "trainer_self_grading": False,
        }
    )


def _pose_request(**overrides):
    value = _common(
        method_profile_id="colmap_aliked_lightglue_v1",
        feature_extractor="ALIKED_N16ROT",
        feature_matcher="ALIKED_LIGHTGLUE",
        camera_model="OPENCV",
        reconstruction_dataset_digest=D1,
        camera_rig_digest=D2,
        calibration_digest=D3,
        model_asset_digest=D4,
        matcher_model_asset_digest=D5,
        deterministic_matching=True,
        random_seed=17,
        resource_request={"gpu_count": 1, "minimum_vram_gb": 16},
        timeout_seconds=1800,
        spend_cap_usd=0,
        candidate_dataset_contains_hidden_heldout_pixels=False,
        candidate_can_change_split=False,
        candidate_may_read_hidden_heldout=False,
    )
    value.update(overrides)
    return build_pose_estimation_request(value)


def _training_request(**overrides):
    value = _common(
        method_profile_id="gsplat_3dgut_mcmc_v1",
        reconstruction_dataset_digest=D1,
        calibration_digest=D2,
        initialization_geometry_digest=D3,
        pose_result_digest=D4,
        worker_stack_manifest_digest=D5,
        evaluation_contract_digest=D1,
        camera_model="OPENCV",
        densification_configuration={"strategy": "mcmc", "max_gaussians": 1_000_000},
        random_seed=23,
        iteration_budget=30_000,
        resource_request={"gpu_count": 1, "minimum_vram_gb": 24},
        timeout_seconds=7200,
        spend_cap_usd=0,
        output_contract={"appearance_asset": "standard_3dgs_ply"},
        candidate_dataset_contains_hidden_heldout_pixels=False,
        candidate_can_change_split=False,
        candidate_may_read_hidden_heldout=False,
        trainer_may_grade_heldout=False,
    )
    value.update(overrides)
    return build_training_request(value)


def test_worker_manifest_pins_headless_cuda_onnx_colmap_and_gaussian_stacks():
    manifest = _worker_manifest()
    components = {row["component_id"]: row for row in manifest["components"]}
    assert components["linux_base"]["linux_amd64_digest"] == (
        "sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749"
    )
    assert components["colmap"]["version"] == "4.1.1"
    assert components["colmap"]["build_options"] == {
        "CUDA_ENABLED": True,
        "ONNX_ENABLED": True,
        "FETCH_ONNX": True,
        "GUI_ENABLED": False,
        "CMAKE_CUDA_ARCHITECTURES": [75, 80, 86, 89],
    }
    assert components["onnxruntime"]["version"] == "1.24.4"
    assert components["gsplat"]["version"] == "1.5.3"
    assert components["threedgrut"]["version"] == "1.1.0"
    assert manifest["qualification_status"] == "candidate_unbuilt"
    assert manifest["hidden_heldout_access"] is False
    assert manifest["trainer_self_grading"] is False
    assert {row["component_id"] for row in PINNED_WORKER_COMPONENTS} >= {
        "linux_base",
        "nvidia_driver_contract",
        "compiler_toolchain",
        "ffmpeg",
        "colmap",
        "onnxruntime",
        "gsplat",
        "python_ml_runtime",
        "openusd",
        "threedgrut",
        "deterministic_qa",
    }
    assert len(PINNED_MODEL_ASSETS) == 4


def test_worker_manifest_cannot_claim_driver_test_before_build():
    with pytest.raises(
        ReconstructionWorkerContractError, match="unbuilt_manifest_cannot_claim_driver_test"
    ):
        build_worker_stack_manifest(
            {
                **{key: value for key, value in _worker_manifest().items() if key not in {"schema_version", "worker_stack_manifest_digest", "components"}},
                "tested_driver_range": {"minimum": "550.54"},
            }
        )


def test_build_and_smoke_receipts_do_not_imply_scientific_qualification():
    build = build_worker_build_receipt(
        {
            "worker_stack_manifest_digest": _worker_manifest()["worker_stack_manifest_digest"],
            "status": "built",
            "resolved_image_digest": IMAGE,
            "source_commit_sha": SHA,
            "build_context_digest": D1,
            "duration_seconds": 400,
            "cost_usd": 0,
            "logs": [{"artifact_id": "build.log", "digest": D2}],
            "blockers": [],
            "scientific_qualification_inferred": False,
        }
    )
    smoke = build_worker_smoke_receipt(
        {
            "build_receipt_digest": build["build_receipt_digest"],
            "resolved_image_digest": IMAGE,
            "status": "passed",
            "checks": [
                {"check_id": "colmap_cuda_onnx_headless", "status": "passed", "output_digest": D1},
                {"check_id": "gsplat_import", "status": "passed", "output_digest": D2},
                {"check_id": "3dgut_mcmc_tiny_fixture", "status": "passed", "output_digest": D3},
            ],
            "display_attached": False,
            "scientific_qualification_inferred": False,
        }
    )
    assert smoke["status"] == "passed"
    assert smoke["scientific_qualification_inferred"] is False


def test_pose_request_is_digest_bound_and_pairing_is_frozen():
    first = _pose_request()
    second = _pose_request()
    assert first == second
    assert first["candidate_may_read_hidden_heldout"] is False
    with pytest.raises(ReconstructionWorkerContractError, match="pose_method_pairing_invalid"):
        _pose_request(feature_matcher="SIFT_BRUTEFORCE")


def test_sift_request_does_not_require_learned_model_but_lightglue_does():
    sift = _pose_request(
        method_profile_id="colmap_sift_bruteforce_v1",
        feature_extractor="SIFT",
        feature_matcher="SIFT_BRUTEFORCE",
        model_asset_digest=None,
        matcher_model_asset_digest=None,
    )
    assert sift["feature_extractor"] == "SIFT"
    with pytest.raises(ReconstructionWorkerContractError, match="lightglue_model_digest_missing"):
        _pose_request(matcher_model_asset_digest=None)


def test_pose_result_preserves_rejected_observations_and_cannot_self_grade():
    request = _pose_request()
    result = build_pose_estimation_result(
        _common(
            pose_estimation_request_digest=request["pose_estimation_request_digest"],
            status="succeeded",
            failure_code=None,
            registered_observation_ids=["frame-001", "frame-002"],
            rejected_observation_ids=["frame-003"],
            output_digests=[
                {"artifact_id": "database.db", "digest": D1},
                {"artifact_id": "pose_graph", "digest": D2},
            ],
            heldout_labels_included=False,
            candidate_self_graded=False,
            proof_effect="calibrated_trajectory_candidate_only",
            claim_ceiling="calibrated_camera_trajectory",
        )
    )
    assert result["rejected_observation_ids"] == ["frame-003"]
    tainted = dict(result)
    tainted.pop("pose_estimation_result_digest")
    tainted["heldout_metrics"] = {"psnr": 99}
    with pytest.raises(ReconstructionWorkerContractError, match="heldout_metrics_forbidden"):
        build_pose_estimation_result(tainted)


@pytest.mark.parametrize(
    "status,failure_code",
    [
        ("failed", "pose_estimation_failure"),
        ("failed", "weak_registration"),
        ("timed_out", "ttl_expiration"),
        ("interrupted", "provider_interruption"),
    ],
)
def test_pose_failures_are_typed(status, failure_code):
    result = build_pose_estimation_result(
        _common(
            pose_estimation_request_digest=_pose_request()["pose_estimation_request_digest"],
            status=status,
            failure_code=failure_code,
            registered_observation_ids=[],
            rejected_observation_ids=["frame-001"],
            heldout_labels_included=False,
            candidate_self_graded=False,
            proof_effect="calibrated_trajectory_candidate_only",
            claim_ceiling="calibrated_camera_trajectory",
            blockers=[failure_code],
        )
    )
    assert result["failure_code"] in FAILURE_CODES


def test_training_request_requires_mcmc_and_blocks_hidden_views_and_self_grading():
    request = _training_request()
    assert request["densification_configuration"]["strategy"] == "mcmc"
    with pytest.raises(ReconstructionWorkerContractError, match="hidden_heldout_access_forbidden"):
        _training_request(candidate_may_read_hidden_heldout=True)
    with pytest.raises(ReconstructionWorkerContractError, match="trainer_self_grading_forbidden"):
        _training_request(trainer_may_grade_heldout=True)


def test_checkpoint_is_exact_request_bound_and_has_no_hidden_state():
    request = _training_request()
    checkpoint = build_checkpoint_manifest(
        _common(
            reconstruction_training_request_digest=request[
                "reconstruction_training_request_digest"
            ],
            iteration=10_000,
            checkpoint_digest=D1,
            random_state={"seed": 23, "digest": D2},
            resume_requires_exact_request_binding=True,
            hidden_heldout_state_included=False,
            output_digests=[{"artifact_id": "checkpoint", "digest": D1}],
            proof_effect="none",
            claim_ceiling="checkpoint_only",
        )
    )
    assert checkpoint["resume_requires_exact_request_binding"] is True
    assert checkpoint["hidden_heldout_state_included"] is False


def test_successful_training_result_is_candidate_only_and_keeps_frame_ledger():
    request = _training_request()
    result = build_training_result(
        _common(
            reconstruction_training_request_digest=request[
                "reconstruction_training_request_digest"
            ],
            status="succeeded",
            failure_code=None,
            output_digests=[{"artifact_id": "appearance.ply", "digest": D1}],
            checkpoint_references=[{"artifact_id": "checkpoint-30000", "digest": D2}],
            training_metrics={"loss": 0.01, "iterations_completed": 30_000},
            heldout_labels_included=False,
            candidate_self_graded=False,
            registered_observation_ids=["frame-001"],
            rejected_observation_ids=["frame-002"],
            peak_resource_use={"gpu_memory_gb": 12.5},
            legal_next_actions=["preserve_evidence_and_stop"],
            proof_effect="appearance_asset_candidate_only",
            claim_ceiling="appearance_reconstruction",
        )
    )
    assert result["proof_effect"] == "appearance_asset_candidate_only"
    assert result["heldout_labels_included"] is False
    assert result["rejected_observation_ids"] == ["frame-002"]


@pytest.mark.parametrize(
    "failure_code",
    [
        "training_divergence",
        "nan_output",
        "gpu_out_of_memory",
        "checkpoint_acquisition_failure",
        "malformed_output",
        "provider_interruption",
        "budget_exhaustion",
        "ttl_expiration",
    ],
)
def test_training_failures_are_typed_and_replayable(failure_code):
    result = build_training_result(
        _common(
            reconstruction_training_request_digest=_training_request()[
                "reconstruction_training_request_digest"
            ],
            status="failed",
            failure_code=failure_code,
            checkpoint_references=[],
            training_metrics={},
            heldout_labels_included=False,
            candidate_self_graded=False,
            registered_observation_ids=[],
            rejected_observation_ids=["frame-001"],
            peak_resource_use={},
            legal_next_actions=["preserve_evidence_and_stop", "abstain"],
            proof_effect="appearance_asset_candidate_only",
            claim_ceiling="appearance_reconstruction",
            blockers=[failure_code],
        )
    )
    assert result["failure_code"] == failure_code
    assert result["output_digests"] == []


def test_unknown_failure_and_nonfinite_cost_fail_closed():
    with pytest.raises(ReconstructionWorkerContractError, match="training_failure_code_invalid"):
        build_training_result(
            _common(
                reconstruction_training_request_digest=D1,
                status="failed",
                failure_code="try_again_later",
                checkpoint_references=[],
                training_metrics={},
                heldout_labels_included=False,
                candidate_self_graded=False,
                registered_observation_ids=[],
                rejected_observation_ids=[],
                peak_resource_use={},
                legal_next_actions=["abstain"],
                proof_effect="appearance_asset_candidate_only",
                claim_ceiling="appearance_reconstruction",
            )
        )
    with pytest.raises(ReconstructionWorkerContractError, match="cost_usd_invalid"):
        _training_request(cost_usd=float("nan"))


def test_schema_files_accept_representative_contracts():
    root = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    examples = {
        "reconstruction_worker_stack_manifest.v1.schema.json": _worker_manifest(),
        "pose_estimation_request.v1.schema.json": _pose_request(),
        "reconstruction_training_request.v1.schema.json": _training_request(),
    }
    for filename, artifact in examples.items():
        schema = json.loads((root / filename).read_text(encoding="utf-8"))
        jsonschema.validate(artifact, schema)


def _bindings(context: SupervisorContext):
    registry = ToolRegistry.default()
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[D1],
    ).to_mapping()
    bindings = {
        binding.tool_id: binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
    }
    return registry, authority, bindings


def test_registered_pose_tool_is_digest_only_and_preserves_claim_boundary(tmp_path: Path):
    request = _pose_request()

    def estimator(*, request, output_root):
        assert output_root.name == "pose_estimation"
        return build_pose_estimation_result(
            _common(
                pose_estimation_request_digest=request["pose_estimation_request_digest"],
                status="succeeded",
                failure_code=None,
                registered_observation_ids=["frame-001"],
                rejected_observation_ids=["frame-002"],
                output_digests=[{"artifact_id": "database.db", "digest": D1}],
                heldout_labels_included=False,
                candidate_self_graded=False,
                proof_effect="calibrated_trajectory_candidate_only",
                claim_ceiling="calibrated_camera_trajectory",
            )
        )

    context = SupervisorContext(
        run_id="pose-tool-run",
        customer_question="Estimate candidate poses",
        supervisor_output_dir=str(tmp_path),
        pose_estimation_request=request,
        pose_estimator=estimator,
    )
    registry, authority, bindings = _bindings(context)
    assert set(bindings["run_pose_estimation"].input_schema["properties"]) == {
        "pose_estimation_request_digest"
    }
    observation = bindings["run_pose_estimation"].invoke(
        {"pose_estimation_request_digest": request["pose_estimation_request_digest"]}
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["registered_observation_count"] == 1
    assert observation["typed_result"]["rejected_observation_count"] == 1
    assert observation["typed_result"]["proof_state_changed"] is False
    assert observation["proof_effect"] == "none"
    validate_tool_observation_binding(
        observation,
        run_id=context.run_id,
        capability="capture_testbed_supervisor",
        registry=registry,
        authority=authority,
    )
    refused = bindings["run_pose_estimation"].invoke(
        {"pose_estimation_request_digest": D2}
    )
    assert refused["status"] == "refused"
    assert "pose_request_binding_mismatch" in refused["typed_failure"]["reason"]


def test_registered_training_tool_is_digest_only_and_cannot_self_grade(tmp_path: Path):
    request = _training_request()

    def trainer(*, request, output_root):
        assert output_root.name == "gaussian_reconstruction"
        return build_training_result(
            _common(
                reconstruction_training_request_digest=request[
                    "reconstruction_training_request_digest"
                ],
                status="succeeded",
                failure_code=None,
                output_digests=[{"artifact_id": "appearance.ply", "digest": D1}],
                checkpoint_references=[{"artifact_id": "checkpoint", "digest": D2}],
                training_metrics={"loss": 0.1},
                heldout_labels_included=False,
                candidate_self_graded=False,
                registered_observation_ids=["frame-001"],
                rejected_observation_ids=[],
                peak_resource_use={"gpu_memory_gb": 10},
                legal_next_actions=["preserve_evidence_and_stop"],
                proof_effect="appearance_asset_candidate_only",
                claim_ceiling="appearance_reconstruction",
            )
        )

    context = SupervisorContext(
        run_id="training-tool-run",
        customer_question="Train a candidate appearance reconstruction",
        supervisor_output_dir=str(tmp_path),
        reconstruction_training_request=request,
        gaussian_reconstruction_trainer=trainer,
    )
    registry, authority, bindings = _bindings(context)
    assert set(bindings["train_gaussian_reconstruction"].input_schema["properties"]) == {
        "reconstruction_training_request_digest"
    }
    observation = bindings["train_gaussian_reconstruction"].invoke(
        {
            "reconstruction_training_request_digest": request[
                "reconstruction_training_request_digest"
            ]
        }
    )
    assert observation["status"] == "completed"
    assert observation["typed_result"]["checkpoint_count"] == 1
    assert observation["typed_result"]["heldout_labels_included"] is False
    assert observation["typed_result"]["candidate_self_graded"] is False
    assert observation["typed_result"]["proof_state_changed"] is False
    assert observation["proof_effect"] == "none"
    validate_tool_observation_binding(
        observation,
        run_id=context.run_id,
        capability="capture_testbed_supervisor",
        registry=registry,
        authority=authority,
    )


def test_registered_training_tool_refuses_malicious_runtime_output(tmp_path: Path):
    request = _training_request()

    def malicious_trainer(*, request, output_root):
        return _common(
            reconstruction_training_request_digest=request[
                "reconstruction_training_request_digest"
            ],
            status="succeeded",
            failure_code=None,
            output_digests=[{"artifact_id": "appearance.ply", "digest": D1}],
            checkpoint_references=[{"artifact_id": "checkpoint", "digest": D2}],
            training_metrics={},
            heldout_metrics={"psnr": 99},
            heldout_labels_included=True,
            candidate_self_graded=True,
            registered_observation_ids=[],
            rejected_observation_ids=[],
            peak_resource_use={},
            legal_next_actions=[],
            proof_effect="appearance_asset_candidate_only",
            claim_ceiling="appearance_reconstruction",
        )

    context = SupervisorContext(
        run_id="malicious-training-tool-run",
        customer_question="Reject malicious output",
        supervisor_output_dir=str(tmp_path),
        reconstruction_training_request=request,
        gaussian_reconstruction_trainer=malicious_trainer,
    )
    _, _, bindings = _bindings(context)
    observation = bindings["train_gaussian_reconstruction"].invoke(
        {
            "reconstruction_training_request_digest": request[
                "reconstruction_training_request_digest"
            ]
        }
    )
    assert observation["status"] == "refused"
    assert "trainer_result_contract_invalid" in observation["typed_failure"]["reason"]


def test_worker_build_packet_fails_closed_without_clean_sha_locks_and_paid_authority():
    packet = prepare_reconstruction_worker_build_packet(
        worker_stack_manifest=_worker_manifest(),
        image_ref="docker.io/blueprint/reconstruction-worker:20260730",
        source_commit_sha=SHA,
        source_tree_digest=D1,
        source_worktree_dirty=True,
        build_recipe_digest=None,
        dependency_lock_digest=None,
        license_review_receipt_digest=None,
        max_spend_usd=None,
        ttl_seconds=None,
        retry_cap=None,
        authority_id=None,
        timestamp="2026-07-30T12:00:00Z",
    )
    assert packet["status"] == "blocked"
    assert packet["allocator_entrypoint"] == ALLOCATOR_ENTRYPOINT
    assert packet["direct_provider_launcher_allowed"] is False
    assert packet["paid_execution_started"] is False
    assert {
        "worker_build_requires_clean_immutable_commit",
        "worker_build_recipe_digest_missing",
        "worker_dependency_lock_digest_missing",
        "worker_license_review_receipt_missing",
        "worker_build_explicit_budget_missing",
        "worker_build_explicit_ttl_missing",
        "worker_build_explicit_retry_cap_missing",
        "worker_build_paid_authority_missing",
    } <= set(packet["blockers"])


def test_worker_build_packet_can_become_ready_but_never_launches_or_selects_provider():
    packet = prepare_reconstruction_worker_build_packet(
        worker_stack_manifest=_worker_manifest(),
        image_ref="docker.io/blueprint/reconstruction-worker:20260730",
        source_commit_sha=SHA,
        source_tree_digest=D1,
        source_worktree_dirty=False,
        build_recipe_digest=D2,
        dependency_lock_digest=D3,
        license_review_receipt_digest=D4,
        max_spend_usd=3.0,
        ttl_seconds=3600,
        retry_cap=1,
        authority_id="authority-fixture",
        timestamp="2026-07-30T12:00:00Z",
    )
    assert packet["status"] == "ready"
    assert packet["blockers"] == []
    assert packet["provider_identity"] is None
    assert packet["paid_execution_started"] is False
    assert packet["allocation_success_is_scientific_success"] is False
    assert packet["build_success_is_scientific_success"] is False
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/schemas/reconstruction_worker_build_packet.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validate(packet, schema)


def test_recorded_proxy_cannot_enter_trainer_without_resolved_worker_image() -> None:
    proxy = json.loads(RECORDED_PROXY_PATH.read_text(encoding="utf-8"))
    stack = json.loads(RECORDED_WORKER_STACK_PATH.read_text(encoding="utf-8"))
    admission = json.loads(RECORDED_WORKER_ADMISSION_PATH.read_text(encoding="utf-8"))
    schema_root = Path(__file__).parents[1] / "docs/schemas"

    jsonschema.validate(
        stack,
        json.loads(
            (schema_root / "reconstruction_worker_stack_manifest.v1.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    jsonschema.validate(
        admission,
        json.loads(
            (schema_root / "reconstruction_worker_build_packet.v1.schema.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    assert stack["worker_stack_manifest_digest"] == canonical_digest(
        stack, digest_field="worker_stack_manifest_digest"
    )
    assert admission["build_packet_digest"] == canonical_digest(
        admission, digest_field="build_packet_digest"
    )
    assert admission["status"] == "blocked"
    assert set(admission["blockers"]) == {
        "worker_build_explicit_budget_missing",
        "worker_build_explicit_retry_cap_missing",
        "worker_build_explicit_ttl_missing",
        "worker_build_paid_authority_missing",
        "worker_license_review_receipt_missing",
    }

    outputs = proxy["output_digests"]
    value = {
        "stable_run_identity": "arkitscenes-40958756-training-admission",
        "source_capture_identity": proxy["source_capture_identity"],
        "source_capture_digest": proxy["source_capture_digest"],
        "original_file_references": [
            {
                "artifact_id": row["relative_path"],
                "digest": row["digest"],
            }
            for row in proxy["original_file_references"]
        ],
        "producing_method": "blueprint-training-request-compiler",
        "implementation_version": "reconstruction-training-admission.v1",
        "container_image_digest": None,
        "source_commit_sha": stack["source_commit_sha"],
        "deterministic_configuration_digest": proxy[
            "deterministic_configuration_digest"
        ],
        "input_digests": [
            {"artifact_id": key, "digest": digest}
            for key, digest in sorted(outputs.items())
        ],
        "output_digests": [],
        "train_heldout_split_digest": proxy["train_heldout_split_digest"],
        "camera_calibration_binding": {
            "calibration_digest": outputs["camera_observation_digest"]
        },
        "coordinate_frame_declaration": proxy["coordinate_frame_declaration"],
        "units": "meters",
        "metric_scale_status": "sensor_metric_unvalidated",
        "provider_runtime_identity": {"provider": None, "runtime": None},
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {
            "local_processing_authorized": True,
            "paid_compute_authorized": False,
        },
        "warnings": ["public_dataset_proxy_not_blueprint_raw_contract_3_2"],
        "blockers": ["resolved_worker_image_missing"],
        "proof_effect": "none",
        "claim_ceiling": "request_only",
        "parent_artifact_or_event": {
            "arkitscenes_proxy_compilation_digest": proxy[
                "arkitscenes_proxy_compilation_digest"
            ]
        },
        "timestamp": "2026-07-30T21:04:07Z",
        "method_profile_id": "gsplat_3dgut_mcmc_v1",
        "reconstruction_dataset_digest": outputs["dataset_manifest_digest"],
        "calibration_digest": outputs["camera_observation_digest"],
        "initialization_geometry_digest": outputs[
            "candidate_metric_scaffold_digest"
        ],
        "pose_result_digest": outputs["camera_observation_digest"],
        "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
        "evaluation_contract_digest": outputs["evaluator_metric_scaffold_digest"],
        "camera_model": "PINHOLE",
        "densification_configuration": {
            "strategy": "mcmc",
            "max_gaussians": 1000000,
        },
        "random_seed": 23,
        "iteration_budget": 30000,
        "resource_request": {"gpu_count": 1, "minimum_vram_gb": 24},
        "timeout_seconds": 7200,
        "spend_cap_usd": 0,
        "output_contract": {"appearance_asset": "standard_3dgs_ply"},
        "candidate_dataset_contains_hidden_heldout_pixels": False,
        "candidate_can_change_split": False,
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
    }
    with pytest.raises(
        ReconstructionWorkerContractError,
        match="request_requires_resolved_worker_image",
    ):
        build_training_request(value)
