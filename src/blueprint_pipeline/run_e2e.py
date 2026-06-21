"""Run local preflight, current package pipeline, and agent review end to end."""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

from .agent_runtime.orchestrator import run_agent_review
from .agent_runtime.openai_phase2 import OpenAIPhase2Config
from .capture_orchestrator import (
    PipelineConfig,
    run_capture_pipeline,
)
from .common import PipelineError
from .evaluation_prep_stage import run_evaluation_prep_stage
from .logging_utils import log_event
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .preflight_capture import build_capture_preflight_report
from .synthesis.cosmos_benchmark import run_cosmos_zero_shot_validation_lane


logger = logging.getLogger(__name__)


def run_end_to_end(
    *,
    capture_root: str,
    provider: str,
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
    pipeline_lane: str = "current",
    run_evaluation_prep: bool = False,
    evaluation_prep_provider: str = "manual",
    run_cosmos_validation: bool = False,
) -> dict:
    log_event(
        logger,
        logging.INFO,
        "run_e2e.started",
        capture_root=capture_root,
        provider=provider,
        pipeline_lane=pipeline_lane,
        run_evaluation_prep=run_evaluation_prep,
        run_cosmos_validation=run_cosmos_validation,
    )
    context = resolve_local_capture_context(capture_root)
    preflight = build_capture_preflight_report(context.capture_root)
    if preflight.get("missing_required_inputs"):
        missing_inputs = [str(item) for item in preflight["missing_required_inputs"]]
        log_event(
            logger,
            logging.WARNING,
            "run_e2e.preflight_failed",
            capture_root=str(context.capture_root),
            provider=provider,
            missing_required_input_count=len(missing_inputs),
            missing_required_inputs=missing_inputs,
        )
        missing = ",".join(str(item) for item in preflight["missing_required_inputs"])
        raise PipelineError(f"Preflight failed; missing required inputs: {missing}")
    log_event(
        logger,
        logging.INFO,
        "run_e2e.preflight_completed",
        capture_root=str(context.capture_root),
        provider=provider,
        preflight_status=preflight.get("status"),
    )

    if context.raw_complete_path.is_file():
        log_event(
            logger,
            logging.INFO,
            "run_e2e.materialization_started",
            capture_root=str(context.capture_root),
            raw_prefix_uri=context.raw_prefix_uri,
        )
        materialize_capture_bundle(
            bucket=context.bucket,
            scene_id=context.scene_id,
            capture_id=context.capture_id,
            gcs_root=context.storage_root,
            raw_prefix_uri=context.raw_prefix_uri,
        )
        log_event(
            logger,
            logging.INFO,
            "run_e2e.materialization_completed",
            capture_root=str(context.capture_root),
            raw_prefix_uri=context.raw_prefix_uri,
        )
    elif not context.descriptor_path.is_file():
        log_event(
            logger,
            logging.WARNING,
            "run_e2e.descriptor_missing",
            capture_root=str(context.capture_root),
            raw_complete_path=str(context.raw_complete_path),
            descriptor_path=str(context.descriptor_path),
        )
        raise PipelineError(
            "Descriptor is missing and raw/capture_upload_complete.json was not found."
        )

    pipeline = run_capture_pipeline(
        descriptor_gcs_uri=context.descriptor_uri,
        lane=pipeline_lane,
        config=PipelineConfig(gcs_root=context.storage_root),
    )
    review = run_agent_review(
        capture_root=context.capture_root,
        provider_name=provider,
        mode="qualification",
        openai_phase2_config=openai_phase2_config,
    )
    evaluation_prep_result = (
        run_evaluation_prep_stage(
            capture_root=context.capture_root,
            provider_name=evaluation_prep_provider,
        )
        if run_evaluation_prep
        else None
    )
    cosmos_validation = (
        run_cosmos_zero_shot_validation_lane(
            capture_root=context.capture_root,
            descriptor_gcs_uri=context.descriptor_uri,
            cfg=PipelineConfig(gcs_root=context.storage_root),
        )
        if run_cosmos_validation
        else None
    )
    result = {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "provider": provider,
        "preflight_status": preflight.get("status"),
        "pipeline_status": pipeline.get("status"),
        "pipeline_lanes": pipeline.get("lanes"),
        "pipeline_summary": review.get("artifacts", {}).get("readiness_report"),
        "final_memo_path": review.get("final_memo_path"),
        "final_bundle_path": review.get("final_bundle_path"),
        "evaluation_prep": evaluation_prep_result,
        "webapp_sync_result": (
            evaluation_prep_result.get("webapp_sync_result")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "site_package_manifest": (
            evaluation_prep_result.get("site_package_manifest")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "hosted_review_readiness": (
            evaluation_prep_result.get("hosted_review_readiness")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "proof_pack_manifest": (
            evaluation_prep_result.get("proof_pack_manifest")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "proof_path_status": (
            evaluation_prep_result.get("proof_path_status")
            if isinstance(evaluation_prep_result, dict)
            else None
        ),
        "cosmos_validation": cosmos_validation,
    }
    log_event(
        logger,
        logging.INFO,
        "run_e2e.completed",
        capture_root=str(context.capture_root),
        provider=provider,
        preflight_status=result.get("preflight_status"),
        pipeline_status=result.get("pipeline_status"),
        pipeline_lanes=result.get("pipeline_lanes"),
        evaluation_prep_enabled=run_evaluation_prep,
        cosmos_validation_enabled=run_cosmos_validation,
    )
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a local capture through the current capture-to-package review path"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", required=True, choices=("claude", "openai"))
    parser.add_argument(
        "--pipeline-lane",
        default="current",
        choices=(
            "current",
            "qualification",
            "evaluation_prep",
            "simulation_automation",
            "scene_memory",
            "retrieval_index",
            "frame_alignment",
            "synthesis_coverage_validation",
            "cosmos_single_capture_smoke",
            "all",
        ),
    )
    parser.add_argument("--openai-phase2-mode", choices=("disabled", "codex_cli"))
    parser.add_argument("--openai-phase2-model")
    parser.add_argument("--openai-phase2-codex-bin")
    parser.add_argument("--openai-phase2-timeout-seconds", type=int)
    parser.add_argument("--openai-phase2-reasoning-effort")
    parser.add_argument("--run-evaluation-prep", action="store_true")
    parser.add_argument("--evaluation-prep-provider", default="manual")
    parser.add_argument(
        "--run-cosmos-validation",
        action="store_true",
        help="Legacy optional Cosmos validation path; not part of the current default pipeline.",
    )
    args = parser.parse_args(argv)

    openai_phase2_config = None
    if any(
        [
            args.openai_phase2_mode,
            args.openai_phase2_model,
            args.openai_phase2_codex_bin,
            args.openai_phase2_timeout_seconds,
            args.openai_phase2_reasoning_effort,
        ]
    ):
        env_default = OpenAIPhase2Config.from_env()
        openai_phase2_config = OpenAIPhase2Config(
            mode=args.openai_phase2_mode or env_default.mode,
            model=args.openai_phase2_model or env_default.model,
            codex_bin=args.openai_phase2_codex_bin or env_default.codex_bin,
            timeout_seconds=int(args.openai_phase2_timeout_seconds or env_default.timeout_seconds),
            reasoning_effort=args.openai_phase2_reasoning_effort or env_default.reasoning_effort,
        )

    try:
        result = run_end_to_end(
            capture_root=args.capture_root,
            provider=args.provider,
            openai_phase2_config=openai_phase2_config,
            pipeline_lane=args.pipeline_lane,
            run_evaluation_prep=bool(args.run_evaluation_prep),
            evaluation_prep_provider=args.evaluation_prep_provider,
            run_cosmos_validation=bool(args.run_cosmos_validation),
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "run_e2e.failed",
            capture_root=args.capture_root,
            provider=args.provider,
            reason=str(exc),
        )
        print(f"[run-e2e] FAILED: {exc}")
        return 1

    print(f"[run-e2e] preflight_status={result['preflight_status']}")
    print(f"[run-e2e] pipeline_status={result['pipeline_status']}")
    print(f"[run-e2e] pipeline_lanes={result.get('pipeline_lanes')}")
    print(f"[run-e2e] final_memo={result['final_memo_path']}")
    print(f"[run-e2e] final_bundle={result['final_bundle_path']}")
    if result.get("evaluation_prep"):
        print(f"[run-e2e] evaluation_prep={result['evaluation_prep']['manifest_path']}")
    if result.get("cosmos_validation"):
        print(f"[run-e2e] cosmos_validation={result['cosmos_validation']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
