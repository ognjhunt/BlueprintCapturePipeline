"""Run local preflight, qualification, and agent review end to end."""

from __future__ import annotations

import argparse
from typing import List, Optional

from .agent_runtime.orchestrator import run_agent_review
from .capture_orchestrator import run_capture_pipeline
from .common import PipelineError
from .local_capture import resolve_local_capture_context
from .materialization import materialize_capture_bundle
from .preflight_capture import build_capture_preflight_report
from .swap_orchestrator import OrchestratorConfig


def run_end_to_end(
    *,
    capture_root: str,
    provider: str,
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
        lane="qualification",
        config=OrchestratorConfig(gcs_root=context.storage_root),
    )
    review = run_agent_review(
        capture_root=context.capture_root,
        provider_name=provider,
        mode="qualification",
    )
    return {
        "schema_version": "v1",
        "capture_root": str(context.capture_root),
        "provider": provider,
        "preflight_status": preflight.get("status"),
        "pipeline_status": pipeline.get("status"),
        "pipeline_summary": review.get("artifacts", {}).get("readiness_report"),
        "final_memo_path": review.get("final_memo_path"),
        "final_bundle_path": review.get("final_bundle_path"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run a local capture end to end")
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument("--provider", required=True, choices=("claude", "openai"))
    args = parser.parse_args(argv)

    try:
        result = run_end_to_end(capture_root=args.capture_root, provider=args.provider)
    except Exception as exc:
        print(f"[run-e2e] FAILED: {exc}")
        return 1

    print(f"[run-e2e] preflight_status={result['preflight_status']}")
    print(f"[run-e2e] pipeline_status={result['pipeline_status']}")
    print(f"[run-e2e] final_memo={result['final_memo_path']}")
    print(f"[run-e2e] final_bundle={result['final_bundle_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
