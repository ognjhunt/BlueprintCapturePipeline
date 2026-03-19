"""Run local preflight, qualification, and agent review end to end, with downstream evaluation optional."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .agent_runtime.orchestrator import run_agent_review
from .agent_runtime.openai_phase2 import OpenAIPhase2Config
from .capture_orchestrator import (
    PipelineConfig,
    run_capture_pipeline,
)
from .common import PipelineError
from .evaluation_prep_stage import run_evaluation_prep_stage
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .preflight_capture import build_capture_preflight_report
from .synthesis.cosmos_benchmark import run_cosmos_zero_shot_validation_lane


def run_end_to_end(
    *,
    capture_root: str,
    provider: str,
    openai_phase2_config: Optional[OpenAIPhase2Config] = None,
    pipeline_lane: str = "qualification",
    run_evaluation_prep: bool = False,
    evaluation_prep_provider: str = "manual",
    run_cosmos_validation: bool = False,
) -> dict:
    context = resolve_local_capture_context(capture_root)
    preflight = build_capture_preflight_report(context.capture_root)
    if preflight.get("missing_required_inputs"):
        missing = ",".join(str(item) for item in preflight["missing_required_inputs"])
        raise PipelineError(f"Preflight failed; missing required inputs: {missing}")

    if context.raw_complete_path.is_file():
        materialize_capture_bundle(
            bucket=context.bucket,
            scene_id=context.scene_id,
            capture_id=context.capture_id,
            gcs_root=context.storage_root,
            raw_prefix_uri=context.raw_prefix_uri,
        )
    elif not context.descriptor_path.is_file():
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
    return {
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
        "cosmos_validation": cosmos_validation,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local capture through qualification-first review")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", required=True, choices=("claude", "openai"))
    parser.add_argument(
        "--pipeline-lane",
        default="qualification",
        choices=("qualification", "scene_memory", "evaluation_prep", "retrieval_index", "frame_alignment", "synthesis_coverage_validation", "cosmos_single_capture_smoke", "all"),
    )
    parser.add_argument("--openai-phase2-mode", choices=("disabled", "codex_cli"))
    parser.add_argument("--openai-phase2-model")
    parser.add_argument("--openai-phase2-codex-bin")
    parser.add_argument("--openai-phase2-timeout-seconds", type=int)
    parser.add_argument("--openai-phase2-reasoning-effort")
    parser.add_argument("--run-evaluation-prep", action="store_true")
    parser.add_argument("--evaluation-prep-provider", default="manual")
    parser.add_argument("--run-cosmos-validation", action="store_true")
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
