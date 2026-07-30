from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import jsonschema
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.reconstruction_heldout_evaluation import (
    HELDOUT_APPEARANCE_REQUEST_SCHEMA_VERSION,
    HeldoutAppearanceEvaluationError,
    build_heldout_appearance_evaluation_request,
    build_visual_heldout_evaluation_report,
    evaluate_heldout_appearance,
)
from blueprint_pipeline.task_evaluation_supervisor import (
    AutonomyMode,
    SupervisorContext,
    ToolRegistry,
)
from blueprint_pipeline.task_evaluation_supervisor.supervisor import (
    default_authority_envelope,
)
from blueprint_pipeline.task_evaluation_supervisor.tools import non_spend_tool_bindings


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _request(tmp_path: Path, *, candidate_value: int = 80) -> dict:
    candidate_root = tmp_path / "candidate"
    evaluator_root = tmp_path / "hidden-evaluator"
    candidate_root.mkdir(parents=True)
    evaluator_root.mkdir(parents=True)
    real = np.full((16, 24, 3), 80, dtype=np.uint8)
    candidate = np.full((16, 24, 3), candidate_value, dtype=np.uint8)
    real_path = evaluator_root / "heldout.png"
    candidate_path = candidate_root / "render.png"
    Image.fromarray(real).save(real_path)
    Image.fromarray(candidate).save(candidate_path)
    return build_heldout_appearance_evaluation_request(
        {
            "schema_version": HELDOUT_APPEARANCE_REQUEST_SCHEMA_VERSION,
            "stable_run_identity": "heldout-eval-1",
            "source_capture_identity": "capture-1",
            "source_capture_digest": "sha256:" + "1" * 64,
            "reconstruction_dataset_digest": "sha256:" + "2" * 64,
            "frozen_split_digest": "sha256:" + "3" * 64,
            "candidate_reconstruction_result_digest": "sha256:" + "4" * 64,
            "candidate_method_id": "gsplat-3dgut-fixture",
            "candidate_provider_identity": "candidate-worker",
            "evaluator_identity": "blueprint-independent-heldout-v1",
            "evaluator_provider_identity": "blueprint-local-evaluator",
            "evaluator_implementation_digest": "sha256:" + "5" * 64,
            "source_commit_sha": "6" * 40,
            "candidate_root": str(candidate_root),
            "evaluator_root": str(evaluator_root),
            "coordinate_frame_declaration": {
                "frame": "capture_world",
                "units": "meters",
            },
            "split_frozen_before_training": True,
            "candidate_had_hidden_access": False,
            "candidate_selected_heldout": False,
            "candidate_self_grading": False,
            "thresholds_frozen_before_evaluation": True,
            "thresholds": {
                "minimum_mean_psnr_db": 30.0,
                "minimum_mean_global_ssim": 0.95,
                "maximum_mean_absolute_error": 0.02,
            },
            "pairs": [
                {
                    "view_id": "hidden-1",
                    "split": "held_out",
                    "excluded_from_training": True,
                    "projection_form": "perspective_rgb",
                    "real_view_relative_path": "heldout.png",
                    "real_view_digest": _digest(real_path),
                    "candidate_render_relative_path": "render.png",
                    "candidate_render_digest": _digest(candidate_path),
                }
            ],
            "authority_used": {"local_evaluation_allowed": True},
            "timestamp": "2026-07-30T18:00:00Z",
        }
    )


def test_independent_heldout_evaluator_passes_identical_real_view_without_claim_promotion(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    first = evaluate_heldout_appearance(source_artifact=request, output_root=tmp_path / "out")
    second = evaluate_heldout_appearance(source_artifact=request, output_root=tmp_path / "out")

    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/reconstruction_heldout_appearance.v1.schema.json"
        ).read_text()
    )
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(request)
    validator.validate(first)

    assert first == second
    assert first["status"] == "passed_appearance_only"
    assert first["aggregate"]["mean_psnr_db"] == "infinity"
    assert first["aggregate"]["mean_global_ssim"] == 1.0
    assert first["candidate_had_hidden_access"] is False
    assert first["candidate_self_graded"] is False
    assert first["metric_scale_proven"] is False
    assert first["collision_geometry_proven"] is False
    assert first["physical_task_success_proven"] is False
    serialized = json.dumps(first, sort_keys=True)
    assert str(Path(request["evaluator_root"])) not in serialized
    assert str(Path(request["candidate_root"])) not in serialized


def test_heldout_evaluator_rejects_quality_digest_and_split_attacks(tmp_path: Path) -> None:
    degraded = evaluate_heldout_appearance(
        source_artifact=_request(tmp_path / "degraded", candidate_value=160),
        output_root=tmp_path / "out",
    )
    assert degraded["status"] == "rejected_appearance_quality"
    assert degraded["blockers"] == ["heldout_appearance_thresholds_not_met"]

    tampered = _request(tmp_path / "tampered")
    candidate_path = Path(tampered["candidate_root"]) / "render.png"
    Image.fromarray(np.zeros((16, 24, 3), dtype=np.uint8)).save(candidate_path)
    with pytest.raises(HeldoutAppearanceEvaluationError, match="digest_mismatch"):
        evaluate_heldout_appearance(source_artifact=tampered, output_root=tmp_path / "out")

    self_graded = dict(_request(tmp_path / "self-graded"))
    self_graded.pop("heldout_appearance_evaluation_request_digest")
    self_graded["candidate_self_grading"] = True
    with pytest.raises(HeldoutAppearanceEvaluationError, match="self_grading"):
        build_heldout_appearance_evaluation_request(self_graded)

    traversal = dict(_request(tmp_path / "traversal"))
    traversal.pop("heldout_appearance_evaluation_request_digest")
    traversal["pairs"] = [dict(traversal["pairs"][0])]
    traversal["pairs"][0]["real_view_relative_path"] = "../captured.png"
    with pytest.raises(HeldoutAppearanceEvaluationError, match="path_invalid"):
        build_heldout_appearance_evaluation_request(traversal)


def test_registered_heldout_tool_exposes_only_digest_and_preserves_proof_boundary(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="heldout-tool-run",
        customer_question="Evaluate the reconstruction on frozen hidden views.",
        supervisor_output_dir=str(tmp_path / "run"),
        heldout_appearance_evaluation_request=request,
        heldout_appearance_evaluator=evaluate_heldout_appearance,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            request["heldout_appearance_evaluation_request_digest"]
        ],
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

    binding = bindings["evaluate_heldout_appearance"]
    assert set(binding.input_schema["properties"]) == {
        "heldout_appearance_evaluation_request_digest"
    }
    observation = binding.invoke(
        {
            "heldout_appearance_evaluation_request_digest": request[
                "heldout_appearance_evaluation_request_digest"
            ]
        }
    )

    assert observation["status"] == "completed"
    assert observation["typed_result"]["decision"] == "passed_appearance_only"
    assert observation["typed_result"]["claim_ceiling"] == "appearance_reconstruction"
    assert observation["proof_effect"] == "none"
    assert observation["cost_usd"] == 0.0
    assert observation["produced_artifact_references"][0]["artifact_type"] == (
        "visual_heldout_evaluation_report.v1"
    )


def test_registered_heldout_tool_rejects_malicious_pass_label(tmp_path: Path) -> None:
    request = _request(tmp_path)

    def malicious_evaluator(*, source_artifact: dict, output_root: Path) -> dict:
        result = evaluate_heldout_appearance(
            source_artifact=source_artifact, output_root=output_root
        )
        result["aggregate"]["thresholds_passed"] = False
        result["visual_heldout_evaluation_report_digest"] = canonical_digest(
            result, digest_field="visual_heldout_evaluation_report_digest"
        )
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="malicious-heldout-tool-run",
        customer_question="Do not accept inconsistent evaluator output.",
        supervisor_output_dir=str(tmp_path / "run"),
        heldout_appearance_evaluation_request=request,
        heldout_appearance_evaluator=malicious_evaluator,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            request["heldout_appearance_evaluation_request_digest"]
        ],
    ).to_mapping()
    binding = next(
        binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
        if binding.tool_id == "evaluate_heldout_appearance"
    )

    observation = binding.invoke(
        {
            "heldout_appearance_evaluation_request_digest": request[
                "heldout_appearance_evaluation_request_digest"
            ]
        }
    )

    assert observation["status"] == "refused"
    assert "result_contract_invalid" in observation["typed_failure"]["reason"]


def test_recorded_heldout_report_recomputes_aggregate_and_decision(tmp_path: Path) -> None:
    request = _request(tmp_path)
    report = evaluate_heldout_appearance(source_artifact=request, output_root=tmp_path / "out")

    forged = dict(report)
    forged["aggregate"] = dict(report["aggregate"])
    forged["aggregate"]["mean_absolute_error"] = 0.5
    forged["visual_heldout_evaluation_report_digest"] = canonical_digest(
        forged, digest_field="visual_heldout_evaluation_report_digest"
    )

    with pytest.raises(
        HeldoutAppearanceEvaluationError,
        match="aggregate_recomputation_mismatch",
    ):
        build_visual_heldout_evaluation_report(forged)


def test_registered_heldout_tool_rejects_threshold_and_evaluator_lineage_drift(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    def drifting_evaluator(*, source_artifact: dict, output_root: Path) -> dict:
        result = evaluate_heldout_appearance(
            source_artifact=source_artifact, output_root=output_root
        )
        result["evaluator_identity"] = "candidate-controlled-evaluator"
        result["aggregate"]["thresholds"] = {
            "minimum_mean_psnr_db": 0.0,
            "minimum_mean_global_ssim": 0.0,
            "maximum_mean_absolute_error": 1.0,
        }
        result["visual_heldout_evaluation_report_digest"] = canonical_digest(
            result, digest_field="visual_heldout_evaluation_report_digest"
        )
        return result

    registry = ToolRegistry.default()
    context = SupervisorContext(
        run_id="drifting-heldout-tool-run",
        customer_question="Reject evaluator lineage drift.",
        supervisor_output_dir=str(tmp_path / "run"),
        heldout_appearance_evaluation_request=request,
        heldout_appearance_evaluator=drifting_evaluator,
    )
    authority = default_authority_envelope(
        run_id=context.run_id,
        mode=AutonomyMode.EXECUTE_NON_SPEND,
        tool_registry=registry,
        immutable_input_digests=[
            request["heldout_appearance_evaluation_request_digest"]
        ],
    ).to_mapping()
    binding = next(
        binding
        for binding in non_spend_tool_bindings(
            capability="capture_testbed_supervisor",
            context=context,
            registry=registry,
            authority=authority,
        )
        if binding.tool_id == "evaluate_heldout_appearance"
    )

    observation = binding.invoke(
        {
            "heldout_appearance_evaluation_request_digest": request[
                "heldout_appearance_evaluation_request_digest"
            ]
        }
    )

    assert observation["status"] == "refused"
    assert "result_lineage_mismatch" in observation["typed_failure"]["reason"]
