"""Base classes for pipeline jobs."""
from __future__ import annotations

import os
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import ArtifactPaths, JobPayload, SessionManifest
from ..utils.gcs import GCSClient, GCSPath
from ..utils.gpu import GPUContext, GPUInfo, check_gpu_memory, get_available_gpu
from ..utils.io import ensure_local_dir, load_json, save_json, temp_workspace
from ..utils.logging import ProgressTracker, get_logger, setup_logging


class JobStatus(Enum):
    """Status of a job execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result of a job execution."""
    status: JobStatus
    outputs: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    artifacts_uploaded: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "artifacts_uploaded": self.artifacts_uploaded,
        }


@dataclass
class JobContext:
    """Runtime context for job execution."""
    session: SessionManifest
    artifacts: ArtifactPaths
    parameters: Dict[str, Any]
    workspace: Path
    gcs: GCSClient
    tracker: ProgressTracker
    logger: Any  # logging.Logger

    @classmethod
    def create(
        cls,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Dict[str, Any],
        workspace: Path,
        job_name: str,
    ) -> "JobContext":
        """Create a job context with initialized utilities."""
        logger = setup_logging(
            job_name=job_name,
            session_id=session.session_id,
            cloud_logging=os.environ.get("K_SERVICE") is not None,
        )
        tracker = ProgressTracker(
            job_name=job_name,
            session_id=session.session_id,
            logger=logger,
        )
        return cls(
            session=session,
            artifacts=artifacts,
            parameters=parameters,
            workspace=workspace,
            gcs=GCSClient(),
            tracker=tracker,
            logger=logger,
        )


@dataclass
class BaseJob:
    """Base class for pipeline jobs.

    Each job declares the inputs it consumes and outputs it produces. These
    declarations can be serialized into Cloud Run Job payloads or used by
    an orchestrator to wire Pub/Sub topics.
    """

    name: str
    description: str
    uses_gpu: bool = False
    timeout_minutes: int = 30

    def build_payload(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, object]] = None,
    ) -> JobPayload:
        """Build a Cloud Run Job payload for this job.

        Args:
            session: Session manifest with capture metadata.
            artifacts: Artifact paths configuration.
            parameters: Optional parameter overrides.

        Returns:
            JobPayload ready for Cloud Run execution.
        """
        raise NotImplementedError

    def run(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, Any]] = None,
        workspace: Optional[Path] = None,
    ) -> JobResult:
        """Execute the job.

        This is the main entry point for job execution. It handles:
        - Workspace setup/cleanup
        - Logging initialization
        - Progress tracking
        - Error handling

        Args:
            session: Session manifest with capture metadata.
            artifacts: Artifact paths configuration.
            parameters: Optional parameter overrides.
            workspace: Optional workspace directory (uses temp if not provided).

        Returns:
            JobResult with status, outputs, and metrics.
        """
        start_time = time.time()
        result = JobResult(status=JobStatus.RUNNING)

        # Use provided workspace or create temporary one
        cleanup_workspace = workspace is None
        if workspace is None:
            workspace = Path(f"/tmp/blueprint_{self.name}_{session.session_id}")
            workspace.mkdir(parents=True, exist_ok=True)

        try:
            # Create job context
            merged_params = self._merge_default_parameters(parameters)
            ctx = JobContext.create(
                session=session,
                artifacts=artifacts,
                parameters=merged_params,
                workspace=workspace,
                job_name=self.name,
            )

            ctx.logger.info(f"Starting job: {self.name}")
            ctx.logger.info(f"Session ID: {session.session_id}")
            ctx.logger.info(f"Workspace: {workspace}")
            ctx.logger.info(f"Parameters: {merged_params}")

            # Validate prerequisites
            self._validate_prerequisites(ctx)

            # Execute job-specific logic
            result = self._execute(ctx)
            result.status = JobStatus.COMPLETED

            # Generate and save report
            report = ctx.tracker.generate_report()
            report_path = workspace / "job_report.json"
            save_json(report, report_path)
            result.metrics = report

            ctx.logger.info(f"Job completed successfully: {self.name}")

        except Exception as e:
            result.status = JobStatus.FAILED
            result.errors.append(str(e))
            get_logger(self.name).error(f"Job failed: {e}", exc_info=True)

        finally:
            result.duration_seconds = time.time() - start_time

            # Cleanup workspace if temporary
            if cleanup_workspace and workspace.exists():
                import shutil
                shutil.rmtree(workspace, ignore_errors=True)

        return result

    @abstractmethod
    def _execute(self, ctx: JobContext) -> JobResult:
        """Execute job-specific logic.

        Subclasses must implement this method with their processing logic.

        Args:
            ctx: Job context with utilities and configuration.

        Returns:
            JobResult with outputs and metrics.
        """
        raise NotImplementedError

    def _validate_prerequisites(self, ctx: JobContext) -> None:
        """Validate job prerequisites.

        Override in subclasses to add custom validation.

        Args:
            ctx: Job context.

        Raises:
            ValueError: If prerequisites are not met.
        """
        pass

    def _merge_default_parameters(
        self,
        parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge provided parameters with defaults.

        Args:
            parameters: User-provided parameters.

        Returns:
            Merged parameters dictionary.
        """
        defaults = self._get_default_parameters()
        if parameters:
            defaults.update(parameters)
        return defaults

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this job.

        Override in subclasses to provide defaults.

        Returns:
            Dictionary of default parameters.
        """
        return {}


@dataclass
class GPUJob(BaseJob):
    """Base class for GPU-accelerated pipeline jobs.

    Extends BaseJob with:
    - GPU resource management
    - CUDA memory tracking
    - GPU-specific validation
    """

    uses_gpu: bool = True
    gpu_type: str = "L4"  # Cloud Run GPU type
    gpu_memory_gb: int = 24  # Expected GPU memory
    concurrency: int = 1  # Max concurrent instances
    min_gpu_memory_gb: float = 16.0  # Minimum required GPU memory

    def base_parameters(self) -> Dict[str, object]:
        """Get base GPU-related parameters."""
        return {
            "gpu_type": self.gpu_type,
            "gpu_memory_gb": self.gpu_memory_gb,
            "concurrency": self.concurrency,
        }

    def _get_default_parameters(self) -> Dict[str, Any]:
        """Include GPU parameters in defaults."""
        return dict(self.base_parameters())

    def _validate_prerequisites(self, ctx: JobContext) -> None:
        """Validate GPU availability."""
        super()._validate_prerequisites(ctx)

        gpu = get_available_gpu()
        if gpu is None:
            ctx.logger.warning("No GPU detected - job will run on CPU (slower)")
        else:
            ctx.logger.info(f"GPU detected: {gpu.name} ({gpu.memory_total_gb:.1f}GB)")

            if gpu.memory_available_gb < self.min_gpu_memory_gb:
                ctx.logger.warning(
                    f"Low GPU memory: {gpu.memory_available_gb:.1f}GB available, "
                    f"{self.min_gpu_memory_gb}GB recommended"
                )

    def run(
        self,
        session: SessionManifest,
        artifacts: ArtifactPaths,
        parameters: Optional[Dict[str, Any]] = None,
        workspace: Optional[Path] = None,
    ) -> JobResult:
        """Execute GPU job with resource management."""
        # Wrap execution in GPU context for resource management
        gpu = get_available_gpu()
        device_index = gpu.index if gpu else 0

        with GPUContext(device_index=device_index):
            return super().run(session, artifacts, parameters, workspace)


@dataclass
class FrameSet:
    """Common input/output references for frame-based stages."""

    frames_uri: str
    masks_uri: str

    @classmethod
    def from_artifacts(cls, artifacts: ArtifactPaths) -> "FrameSet":
        """Create from artifact paths."""
        return cls(frames_uri=artifacts.frames, masks_uri=artifacts.masks)


def merge_parameters(
    base: Dict[str, object], extra: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Merge base parameters with extra overrides.

    Args:
        base: Base parameter dictionary.
        extra: Optional extra parameters to merge.

    Returns:
        Merged dictionary with extra values overriding base.
    """
    if not extra:
        return base
    merged = dict(base)
    merged.update(extra)
    return merged


# ==============================================================================
# Utility functions for common job operations
# ==============================================================================

def download_inputs(
    ctx: JobContext,
    input_uris: Dict[str, str],
) -> Dict[str, Path]:
    """Download job inputs from GCS to local workspace.

    Args:
        ctx: Job context.
        input_uris: Mapping of input names to GCS URIs.

    Returns:
        Mapping of input names to local paths.
    """
    local_paths = {}
    inputs_dir = ensure_local_dir(ctx.workspace / "inputs")

    with ctx.tracker.stage("download_inputs", len(input_uris)):
        for name, uri in input_uris.items():
            ctx.logger.info(f"Downloading {name}: {uri}")

            if uri.endswith("/"):
                # Directory download
                local_dir = inputs_dir / name
                ctx.gcs.download_directory(uri, local_dir)
                local_paths[name] = local_dir
            else:
                # Single file download
                parsed = GCSPath.from_uri(uri)
                filename = Path(parsed.blob).name
                local_path = inputs_dir / name / filename
                ctx.gcs.download(uri, local_path)
                local_paths[name] = local_path

            ctx.tracker.update(1)

    return local_paths


def upload_outputs(
    ctx: JobContext,
    local_paths: Dict[str, Path],
    output_uris: Dict[str, str],
) -> Dict[str, str]:
    """Upload job outputs from local workspace to GCS.

    Args:
        ctx: Job context.
        local_paths: Mapping of output names to local paths.
        output_uris: Mapping of output names to destination GCS URIs.

    Returns:
        Mapping of output names to uploaded GCS URIs.
    """
    uploaded = {}

    with ctx.tracker.stage("upload_outputs", len(local_paths)):
        for name, local_path in local_paths.items():
            if name not in output_uris:
                ctx.logger.warning(f"No output URI configured for: {name}")
                continue

            uri = output_uris[name]
            ctx.logger.info(f"Uploading {name}: {local_path} -> {uri}")

            if local_path.is_dir():
                # Directory upload
                uris = ctx.gcs.upload_directory(local_path, uri)
                uploaded[name] = uri
                ctx.tracker.log_metric(f"{name}_files_uploaded", len(uris))
            else:
                # Single file upload
                ctx.gcs.upload(local_path, uri)
                uploaded[name] = uri

            ctx.tracker.update(1)

    return uploaded


def load_session_manifest(manifest_path: Path) -> SessionManifest:
    """Load and validate a session manifest from JSON/YAML.

    Args:
        manifest_path: Path to manifest file.

    Returns:
        Parsed SessionManifest.
    """
    from ..models import Clip, ScaleAnchor

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
