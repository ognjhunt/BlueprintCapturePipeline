"""Job runner entry point for Cloud Run Jobs execution.

This module provides the entry point for running pipeline jobs in Cloud Run.
It reads job configuration from environment variables and executes the appropriate job.

Usage:
    # As Cloud Run Job
    JOB_PAYLOAD='{"job_name": "frame-extraction", ...}' python -m blueprint_pipeline.runner

    # CLI
    python -m blueprint_pipeline.runner --manifest session.yaml --stage frame-extraction
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .models import ArtifactPaths, Clip, ScaleAnchor, SessionManifest
from .jobs.base import BaseJob, JobResult, JobStatus
from .jobs.frame_extraction import FrameExtractionJob
from .jobs.reconstruction import ReconstructionJob
from .jobs.mesh import MeshExtractionJob
from .jobs.object_assetization import ObjectAssetizationJob
from .jobs.usd_authoring import USDAuthoringJob
from .pipeline import default_artifact_paths
from .orchestrator import PipelineOrchestrator, PipelineStage
from .utils.logging import setup_logging, get_logger
from .utils.io import load_json


# Job registry
JOB_REGISTRY: Dict[str, type] = {
    "frame-extraction": FrameExtractionJob,
    "reconstruction": ReconstructionJob,
    "mesh-extraction": MeshExtractionJob,
    "object-assetization": ObjectAssetizationJob,
    "usd-authoring": USDAuthoringJob,
}


def load_manifest(manifest_path: Path) -> SessionManifest:
    """Load session manifest from file.

    Supports JSON and YAML formats.
    """
    if manifest_path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(manifest_path, "r") as f:
                data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("pyyaml is required for YAML manifests")
    else:
        data = load_json(manifest_path)

    return SessionManifest(
        session_id=data["session_id"],
        capture_start=data["capture_start"],
        device=data["device"],
        scale_anchors=[ScaleAnchor(**a) for a in data.get("scale_anchors", [])],
        clips=[Clip(**c) for c in data.get("clips", [])],
        user_notes=data.get("user_notes"),
    )


def load_payload_from_env() -> Optional[Dict[str, Any]]:
    """Load job payload from JOB_PAYLOAD environment variable."""
    payload_json = os.environ.get("JOB_PAYLOAD")
    if payload_json:
        return json.loads(payload_json)
    return None


def run_from_payload(payload: Dict[str, Any]) -> JobResult:
    """Run a job from a payload dictionary.

    This is the main entry point for Cloud Run Jobs execution.
    """
    logger = setup_logging("runner", cloud_logging=True)

    job_name = payload.get("job_name")
    session_id = payload.get("session_id")
    inputs = payload.get("inputs", {})
    outputs = payload.get("outputs", {})
    parameters = payload.get("parameters", {})

    logger.info(f"Running job: {job_name}")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Parameters: {parameters}")

    # Get job class
    job_class = JOB_REGISTRY.get(job_name)
    if job_class is None:
        logger.error(f"Unknown job: {job_name}")
        return JobResult(
            status=JobStatus.FAILED,
            errors=[f"Unknown job: {job_name}"],
        )

    # Create minimal session manifest
    # In production, this would be loaded from GCS
    session = SessionManifest(
        session_id=session_id,
        capture_start="",
        device={},
        scale_anchors=[],
        clips=[],
    )

    # Infer artifact paths from outputs
    # This is a simplified approach - real implementation would parse properly
    session_root = ""
    for output_uri in outputs.values():
        if output_uri.startswith("gs://"):
            parts = output_uri.split("/sessions/")
            if len(parts) > 1:
                session_root = f"{parts[0]}/sessions/{session_id}"
                break

    if not session_root:
        session_root = f"gs://blueprint-pipeline/sessions/{session_id}"

    artifacts = ArtifactPaths(
        session_root=session_root,
        frames=f"{session_root}/frames",
        masks=f"{session_root}/masks",
        reconstruction=f"{session_root}/reconstruction",
        meshes=f"{session_root}/meshes",
        objects=f"{session_root}/objects",
        reports=f"{session_root}/reports",
    )

    # Run job
    job = job_class()
    result = job.run(
        session=session,
        artifacts=artifacts,
        parameters=parameters,
    )

    # Log result
    if result.status == JobStatus.COMPLETED:
        logger.info(f"Job completed successfully in {result.duration_seconds:.1f}s")
    else:
        logger.error(f"Job failed: {result.errors}")

    return result


def run_from_cli(args: argparse.Namespace) -> JobResult:
    """Run a job from CLI arguments."""
    logger = setup_logging("runner")

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return JobResult(
            status=JobStatus.FAILED,
            errors=[f"Manifest not found: {manifest_path}"],
        )

    session = load_manifest(manifest_path)
    logger.info(f"Loaded session: {session.session_id}")

    # Setup artifact paths
    if args.gcs_bucket:
        base = f"gs://{args.gcs_bucket}/sessions/{{session_id}}"
    else:
        base = f"file://{args.workspace}/sessions/{{session_id}}"

    artifacts = default_artifact_paths(session.session_id, base)

    # Parse parameters
    parameters = {}
    if args.parameters:
        parameters = json.loads(args.parameters)

    # Run single stage or full pipeline
    if args.stage:
        stage = PipelineStage(args.stage)
        job_class = JOB_REGISTRY.get(args.stage)
        if job_class is None:
            logger.error(f"Unknown stage: {args.stage}")
            return JobResult(
                status=JobStatus.FAILED,
                errors=[f"Unknown stage: {args.stage}"],
            )

        job = job_class()
        workspace = Path(args.workspace) / session.session_id / args.stage
        workspace.mkdir(parents=True, exist_ok=True)

        result = job.run(
            session=session,
            artifacts=artifacts,
            parameters=parameters,
            workspace=workspace,
        )
    else:
        # Run full pipeline
        orchestrator = PipelineOrchestrator(
            gcs_bucket=args.gcs_bucket or "",
            workspace_base=Path(args.workspace),
        )

        pipeline_result = orchestrator.run_full_pipeline(
            session=session,
            artifacts=artifacts,
            parameters=parameters,
        )

        # Convert to JobResult
        if pipeline_result.success:
            result = JobResult(
                status=JobStatus.COMPLETED,
                metrics=pipeline_result.to_dict(),
                duration_seconds=pipeline_result.total_duration_seconds,
            )
        else:
            result = JobResult(
                status=JobStatus.FAILED,
                errors=[pipeline_result.error_message or "Pipeline failed"],
                metrics=pipeline_result.to_dict(),
                duration_seconds=pipeline_result.total_duration_seconds,
            )

    return result


def main():
    """Main entry point."""
    # Check for Cloud Run Job payload first
    payload = load_payload_from_env()
    if payload:
        result = run_from_payload(payload)
        sys.exit(0 if result.status == JobStatus.COMPLETED else 1)

    # Otherwise, use CLI
    parser = argparse.ArgumentParser(
        description="Blueprint Capture Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python -m blueprint_pipeline.runner --manifest session.yaml

    # Run single stage
    python -m blueprint_pipeline.runner --manifest session.yaml --stage frame-extraction

    # Run with GCS bucket
    python -m blueprint_pipeline.runner --manifest session.yaml --gcs-bucket my-bucket

    # Run with custom parameters
    python -m blueprint_pipeline.runner --manifest session.yaml --parameters '{"target_fps": 2.0}'
        """,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to session manifest (JSON or YAML)",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=list(JOB_REGISTRY.keys()),
        help="Run specific stage (runs full pipeline if not specified)",
    )

    parser.add_argument(
        "--gcs-bucket",
        type=str,
        help="GCS bucket for artifacts (uses local storage if not specified)",
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default="/tmp/blueprint_pipeline",
        help="Local workspace directory",
    )

    parser.add_argument(
        "--parameters",
        type=str,
        help="Job parameters as JSON string",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    result = run_from_cli(args)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Status: {result.status.value}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.outputs:
        print(f"Outputs: {json.dumps(result.outputs, indent=2)}")
    print(f"{'='*60}")

    sys.exit(0 if result.status == JobStatus.COMPLETED else 1)


if __name__ == "__main__":
    main()
