"""Job runner entry point for Cloud Run Jobs execution.

This module provides the entry point for running pipeline jobs in Cloud Run.
It reads job configuration from environment variables and executes the appropriate job.

Usage:
    # As Cloud Run Job (triggered by Cloud Function)
    JOB_PAYLOAD='{"job_name": "full-pipeline", "session_id": "scene_123", ...}' python -m blueprint_pipeline.runner

    # CLI with manifest file
    python -m blueprint_pipeline.runner --manifest session.yaml --stage frame-extraction

    # CLI with iOS upload (auto-discovers from GCS)
    python -m blueprint_pipeline.runner --ios-upload gs://bucket/scenes/scene_id/iphone/timestamp/raw
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
from .utils.gcs import GCSClient, GCSPath


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


def load_session_from_ios_upload(gcs_uri: str) -> SessionManifest:
    """Load session manifest from an iOS upload in GCS.

    Args:
        gcs_uri: GCS URI to the raw/ directory of an iOS upload
                 e.g., gs://bucket/scenes/scene_id/iphone/timestamp/raw

    Returns:
        SessionManifest converted from iOS format
    """
    from .ios_manifest import load_extended_session

    # Parse GCS URI
    parsed = GCSPath.from_uri(gcs_uri)

    # Load extended session data
    extended = load_extended_session(parsed.bucket, parsed.blob)

    return extended.manifest


def run_from_ios_trigger(payload: Dict[str, Any]) -> JobResult:
    """Run pipeline from iOS Cloud Function trigger.

    Expected payload format (from storage_trigger.py):
    {
        "job_name": "full-pipeline",
        "session_id": "scene_123",
        "inputs": {
            "manifest_uri": "gs://bucket/sessions/scene_123/session_manifest.json"
        },
        "outputs": {
            "base": "gs://bucket/sessions/scene_123"
        },
        "parameters": {}
    }
    """
    logger = setup_logging("runner", cloud_logging=True)

    job_name = payload.get("job_name", "full-pipeline")
    session_id = payload.get("session_id")
    inputs = payload.get("inputs", {})
    outputs = payload.get("outputs", {})
    parameters = payload.get("parameters", {})

    logger.info(f"Running from iOS trigger: {job_name} for session {session_id}")

    # Load session manifest from GCS
    manifest_uri = inputs.get("manifest_uri")
    if manifest_uri:
        logger.info(f"Loading session manifest from {manifest_uri}")
        gcs = GCSClient()
        parsed = GCSPath.from_uri(manifest_uri)
        bucket = gcs._get_bucket(parsed.bucket)
        blob = bucket.blob(parsed.blob)
        manifest_data = json.loads(blob.download_as_text())

        # Build session manifest from the converted format
        session = SessionManifest(
            session_id=manifest_data["session_id"],
            capture_start=manifest_data["capture_start"],
            device=manifest_data["device"],
            scale_anchors=[ScaleAnchor(**a) for a in manifest_data.get("scale_anchors", [])],
            clips=[Clip(**c) for c in manifest_data.get("clips", [])],
            user_notes=manifest_data.get("user_notes"),
        )
    else:
        # Minimal session for direct runs
        session = SessionManifest(
            session_id=session_id,
            capture_start="",
            device={},
            scale_anchors=[],
            clips=[],
        )

    # Setup artifact paths
    output_base = outputs.get("base", f"gs://blueprint-8c1ca.appspot.com/sessions/{session_id}")
    artifacts = ArtifactPaths(
        session_root=output_base,
        frames=f"{output_base}/frames",
        masks=f"{output_base}/masks",
        reconstruction=f"{output_base}/reconstruction",
        meshes=f"{output_base}/meshes",
        objects=f"{output_base}/objects",
        reports=f"{output_base}/reports",
    )

    # Check if this is a single job or full pipeline
    if job_name == "full-pipeline":
        # Extract GCS bucket from output_base
        bucket_match = GCSPath.from_uri(output_base)
        gcs_bucket = bucket_match.bucket

        # Check for ARKit poses to potentially skip SLAM
        raw_data_prefix = manifest_data.get("raw_data_prefix") if manifest_uri else None
        if raw_data_prefix:
            from .arkit_loader import load_arkit_data_from_gcs, can_skip_slam
            from pathlib import Path
            import tempfile

            try:
                parsed_raw = GCSPath.from_uri(raw_data_prefix)
                with tempfile.TemporaryDirectory() as tmpdir:
                    arkit_data = load_arkit_data_from_gcs(
                        parsed_raw.bucket,
                        parsed_raw.blob,
                        Path(tmpdir),
                    )
                    if can_skip_slam(arkit_data):
                        logger.info("ARKit poses available - will use direct pose import instead of SLAM")
                        parameters["use_arkit_poses"] = True
                        parameters["arkit_data_uri"] = raw_data_prefix
            except Exception as e:
                logger.warning(f"Could not check ARKit data: {e}")

        orchestrator = PipelineOrchestrator(
            gcs_bucket=gcs_bucket,
            workspace_base=Path("/tmp/blueprint_pipeline"),
        )

        pipeline_result = orchestrator.run_full_pipeline(
            session=session,
            artifacts=artifacts,
            parameters=parameters,
        )

        if pipeline_result.success:
            return JobResult(
                status=JobStatus.COMPLETED,
                metrics=pipeline_result.to_dict(),
                duration_seconds=pipeline_result.total_duration_seconds,
            )
        else:
            return JobResult(
                status=JobStatus.FAILED,
                errors=[pipeline_result.error_message or "Pipeline failed"],
                metrics=pipeline_result.to_dict(),
                duration_seconds=pipeline_result.total_duration_seconds,
            )
    else:
        # Single job execution
        job_class = JOB_REGISTRY.get(job_name)
        if job_class is None:
            return JobResult(
                status=JobStatus.FAILED,
                errors=[f"Unknown job: {job_name}"],
            )

        job = job_class()
        return job.run(
            session=session,
            artifacts=artifacts,
            parameters=parameters,
        )


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


def run_from_ios_upload(args: argparse.Namespace) -> JobResult:
    """Run pipeline from iOS upload in GCS.

    Args:
        args: CLI arguments with ios_upload GCS URI
    """
    logger = setup_logging("runner")

    gcs_uri = args.ios_upload
    logger.info(f"Loading session from iOS upload: {gcs_uri}")

    # Parse GCS URI to extract bucket and path
    parsed = GCSPath.from_uri(gcs_uri)
    bucket_name = parsed.bucket
    raw_prefix = parsed.blob

    # Load extended session data
    from .ios_manifest import load_extended_session

    try:
        extended = load_extended_session(bucket_name, raw_prefix)
    except Exception as e:
        logger.error(f"Failed to load iOS upload: {e}")
        return JobResult(
            status=JobStatus.FAILED,
            errors=[f"Failed to load iOS upload: {e}"],
        )

    session = extended.manifest
    logger.info(f"Loaded session: {session.session_id}")

    # Check for ARKit poses
    from .arkit_loader import load_arkit_data_from_gcs, can_skip_slam
    import tempfile

    parameters = {}
    if args.parameters:
        parameters = json.loads(args.parameters)

    # Try to load ARKit data and check if we can skip SLAM
    if extended.upload_info.has_arkit_poses:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                arkit_data = load_arkit_data_from_gcs(
                    bucket_name,
                    raw_prefix,
                    Path(tmpdir),
                )
                if can_skip_slam(arkit_data):
                    logger.info("ARKit poses available - will use direct pose import instead of SLAM")
                    parameters["use_arkit_poses"] = True
                    parameters["arkit_data_uri"] = gcs_uri
        except Exception as e:
            logger.warning(f"Could not load ARKit data: {e}")

    # Setup artifact paths
    # Use same bucket as upload, but under sessions/ prefix
    gcs_bucket = args.gcs_bucket or bucket_name
    base = f"gs://{gcs_bucket}/sessions/{{session_id}}"
    artifacts = default_artifact_paths(session.session_id, base)

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
            gcs_bucket=gcs_bucket,
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
        # Check if this is an iOS trigger payload (has manifest_uri in inputs)
        if payload.get("inputs", {}).get("manifest_uri"):
            result = run_from_ios_trigger(payload)
        else:
            result = run_from_payload(payload)
        sys.exit(0 if result.status == JobStatus.COMPLETED else 1)

    # Otherwise, use CLI
    parser = argparse.ArgumentParser(
        description="Blueprint Capture Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with manifest file
    python -m blueprint_pipeline.runner --manifest session.yaml

    # Run from iOS upload in GCS (auto-discovers manifest and ARKit data)
    python -m blueprint_pipeline.runner --ios-upload gs://bucket/scenes/scene_id/iphone/timestamp/raw

    # Run single stage
    python -m blueprint_pipeline.runner --manifest session.yaml --stage frame-extraction

    # Run with GCS bucket
    python -m blueprint_pipeline.runner --manifest session.yaml --gcs-bucket my-bucket

    # Run with custom parameters
    python -m blueprint_pipeline.runner --manifest session.yaml --parameters '{"target_fps": 2.0}'
        """,
    )

    # Input source (one of these is required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--manifest",
        type=str,
        help="Path to session manifest (JSON or YAML)",
    )
    input_group.add_argument(
        "--ios-upload",
        type=str,
        help="GCS URI to iOS upload raw/ directory (gs://bucket/scenes/scene_id/source/folder/raw)",
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

    # Handle iOS upload source
    if args.ios_upload:
        result = run_from_ios_upload(args)
    else:
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
