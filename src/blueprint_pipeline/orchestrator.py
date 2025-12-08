"""Pipeline orchestrator for Cloud Run Jobs execution."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .models import ArtifactPaths, JobPayload, SessionManifest
from .jobs.base import BaseJob, JobResult, JobStatus
from .jobs.frame_extraction import FrameExtractionJob
from .jobs.reconstruction import ReconstructionJob
from .jobs.mesh import MeshExtractionJob
from .jobs.object_assetization import ObjectAssetizationJob
from .jobs.usd_authoring import USDAuthoringJob
from .jobs.articulation import ArticulationJob
from .pipeline import default_artifact_paths
from .utils.logging import get_logger, setup_logging


class PipelineStage(Enum):
    """Pipeline execution stages."""
    FRAME_EXTRACTION = "frame-extraction"
    RECONSTRUCTION = "reconstruction"
    MESH_EXTRACTION = "mesh-extraction"
    OBJECT_ASSETIZATION = "object-assetization"
    USD_AUTHORING = "usd-authoring"
    ARTICULATION = "articulation"


# Mapping of stage names to job classes
STAGE_JOBS: Dict[PipelineStage, Type[BaseJob]] = {
    PipelineStage.FRAME_EXTRACTION: FrameExtractionJob,
    PipelineStage.RECONSTRUCTION: ReconstructionJob,
    PipelineStage.MESH_EXTRACTION: MeshExtractionJob,
    PipelineStage.OBJECT_ASSETIZATION: ObjectAssetizationJob,
    PipelineStage.USD_AUTHORING: USDAuthoringJob,
    PipelineStage.ARTICULATION: ArticulationJob,
}

# Default execution order (DAG-like dependencies)
# Note: Articulation runs after USD_AUTHORING to process scene objects
DEFAULT_PIPELINE_ORDER = [
    PipelineStage.FRAME_EXTRACTION,
    PipelineStage.RECONSTRUCTION,
    PipelineStage.MESH_EXTRACTION,
    PipelineStage.OBJECT_ASSETIZATION,
    PipelineStage.USD_AUTHORING,
    PipelineStage.ARTICULATION,
]


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    stage: PipelineStage
    job_result: JobResult
    start_time: float
    end_time: float

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def success(self) -> bool:
        return self.job_result.status == JobStatus.COMPLETED


@dataclass
class PipelineResult:
    """Result of full pipeline execution."""
    session_id: str
    stages: List[StageResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    success: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "success": self.success,
            "total_duration_seconds": self.total_duration_seconds,
            "error_message": self.error_message,
            "stages": [
                {
                    "stage": sr.stage.value,
                    "success": sr.success,
                    "duration_seconds": sr.duration_seconds,
                    "outputs": sr.job_result.outputs,
                    "errors": sr.job_result.errors,
                }
                for sr in self.stages
            ],
        }


@dataclass
class PipelineOrchestrator:
    """Orchestrates execution of the Blueprint pipeline.

    Can run locally for testing or dispatch to Cloud Run Jobs for production.
    """

    gcs_bucket: str = ""
    workspace_base: Path = field(default_factory=lambda: Path("/tmp/blueprint_pipeline"))
    stop_on_failure: bool = True

    def __post_init__(self):
        self.logger = setup_logging("orchestrator")

    def run_full_pipeline(
        self,
        session: SessionManifest,
        artifacts: Optional[ArtifactPaths] = None,
        parameters: Optional[Dict[str, Any]] = None,
        stages: Optional[List[PipelineStage]] = None,
    ) -> PipelineResult:
        """Run the complete pipeline for a capture session.

        Args:
            session: Session manifest with capture metadata.
            artifacts: Optional artifact paths (auto-generated if not provided).
            parameters: Optional parameters to pass to all jobs.
            stages: Optional list of stages to run (defaults to all).

        Returns:
            PipelineResult with status of all stages.
        """
        start_time = time.time()
        result = PipelineResult(session_id=session.session_id)

        # Setup artifact paths
        if artifacts is None:
            if self.gcs_bucket:
                base = f"gs://{self.gcs_bucket}/sessions/{{session_id}}"
            else:
                base = f"file://{self.workspace_base}/sessions/{{session_id}}"
            artifacts = default_artifact_paths(session.session_id, base)

        # Determine stages to run
        stages_to_run = stages or DEFAULT_PIPELINE_ORDER

        self.logger.info(f"Starting pipeline for session: {session.session_id}")
        self.logger.info(f"Stages to run: {[s.value for s in stages_to_run]}")

        # Create session workspace
        session_workspace = self.workspace_base / session.session_id
        session_workspace.mkdir(parents=True, exist_ok=True)

        # Run each stage
        for stage in stages_to_run:
            self.logger.info(f"Running stage: {stage.value}")
            stage_start = time.time()

            try:
                job_result = self._run_stage(
                    stage=stage,
                    session=session,
                    artifacts=artifacts,
                    parameters=parameters,
                    workspace=session_workspace / stage.value,
                )

                stage_result = StageResult(
                    stage=stage,
                    job_result=job_result,
                    start_time=stage_start,
                    end_time=time.time(),
                )
                result.stages.append(stage_result)

                if job_result.status == JobStatus.COMPLETED:
                    self.logger.info(
                        f"Stage {stage.value} completed in {stage_result.duration_seconds:.1f}s"
                    )
                else:
                    self.logger.error(
                        f"Stage {stage.value} failed: {job_result.errors}"
                    )
                    if self.stop_on_failure:
                        result.error_message = f"Stage {stage.value} failed"
                        break

            except Exception as e:
                self.logger.error(f"Stage {stage.value} raised exception: {e}")
                stage_result = StageResult(
                    stage=stage,
                    job_result=JobResult(
                        status=JobStatus.FAILED,
                        errors=[str(e)],
                    ),
                    start_time=stage_start,
                    end_time=time.time(),
                )
                result.stages.append(stage_result)

                if self.stop_on_failure:
                    result.error_message = f"Stage {stage.value} raised exception: {e}"
                    break

        # Compute overall result
        result.total_duration_seconds = time.time() - start_time
        result.success = all(sr.success for sr in result.stages) and len(result.stages) == len(stages_to_run)

        self.logger.info(
            f"Pipeline {'completed successfully' if result.success else 'failed'} "
            f"in {result.total_duration_seconds:.1f}s"
        )

        return result

    def _run_stage(
        self,
        stage: PipelineStage,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, Any]],
        workspace: Path,
    ) -> JobResult:
        """Run a single pipeline stage."""
        job_class = STAGE_JOBS.get(stage)
        if job_class is None:
            raise ValueError(f"Unknown stage: {stage}")

        job = job_class()
        return job.run(
            session=session,
            artifacts=artifacts,
            parameters=parameters,
            workspace=workspace,
        )

    def run_single_stage(
        self,
        stage: PipelineStage,
        session: SessionManifest,
        artifacts: Optional[ArtifactPaths] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> JobResult:
        """Run a single pipeline stage.

        Useful for retrying failed stages or running stages independently.
        """
        if artifacts is None:
            if self.gcs_bucket:
                base = f"gs://{self.gcs_bucket}/sessions/{{session_id}}"
            else:
                base = f"file://{self.workspace_base}/sessions/{{session_id}}"
            artifacts = default_artifact_paths(session.session_id, base)

        workspace = self.workspace_base / session.session_id / stage.value
        workspace.mkdir(parents=True, exist_ok=True)

        return self._run_stage(
            stage=stage,
            session=session,
            artifacts=artifacts,
            parameters=parameters,
            workspace=workspace,
        )


def create_cloud_run_job_config(
    job_name: str,
    image: str,
    payload: JobPayload,
    gpu_enabled: bool = True,
    timeout_minutes: int = 60,
    memory: str = "16Gi",
    cpu: str = "4",
) -> Dict[str, Any]:
    """Create Cloud Run Job configuration for a pipeline stage.

    Args:
        job_name: Name for the Cloud Run Job.
        image: Docker image URI.
        payload: Job payload with inputs/outputs.
        gpu_enabled: Whether to attach GPU.
        timeout_minutes: Job timeout in minutes.
        memory: Memory allocation (e.g., "16Gi").
        cpu: CPU allocation (e.g., "4").

    Returns:
        Cloud Run Job configuration dictionary.
    """
    config = {
        "apiVersion": "run.googleapis.com/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "annotations": {
                "run.googleapis.com/launch-stage": "BETA",
            },
        },
        "spec": {
            "template": {
                "spec": {
                    "taskCount": 1,
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "image": image,
                                    "resources": {
                                        "limits": {
                                            "memory": memory,
                                            "cpu": cpu,
                                        },
                                    },
                                    "env": [
                                        {
                                            "name": "JOB_PAYLOAD",
                                            "value": json.dumps(payload.as_json()),
                                        },
                                    ],
                                },
                            ],
                            "timeoutSeconds": timeout_minutes * 60,
                            "maxRetries": 1,
                        },
                    },
                },
            },
        },
    }

    # Add GPU if enabled
    if gpu_enabled:
        config["spec"]["template"]["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "1"
        config["metadata"]["annotations"]["run.googleapis.com/gpu-type"] = "nvidia-l4"

    return config


def submit_cloud_run_job(
    project_id: str,
    region: str,
    job_config: Dict[str, Any],
) -> str:
    """Submit a Cloud Run Job.

    Args:
        project_id: GCP project ID.
        region: GCP region.
        job_config: Job configuration from create_cloud_run_job_config.

    Returns:
        Job execution name.
    """
    try:
        from google.cloud import run_v2

        client = run_v2.JobsClient()

        # Create or update job
        job_name = job_config["metadata"]["name"]
        parent = f"projects/{project_id}/locations/{region}"

        # Check if job exists
        try:
            existing = client.get_job(name=f"{parent}/jobs/{job_name}")
            # Update existing job
            client.update_job(job=job_config)
        except Exception:
            # Create new job
            client.create_job(parent=parent, job=job_config, job_id=job_name)

        # Execute the job
        execution = client.run_job(name=f"{parent}/jobs/{job_name}")

        return execution.name

    except ImportError:
        raise ImportError(
            "google-cloud-run is required for Cloud Run Job submission. "
            "Install with: pip install google-cloud-run"
        )
