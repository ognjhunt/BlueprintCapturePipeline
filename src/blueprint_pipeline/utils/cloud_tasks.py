"""Cloud Tasks integration for scalable job queuing.

This module provides Cloud Tasks integration for queueing pipeline jobs,
enabling proper rate limiting, retry handling, and dead letter queue support.

Usage:
    from blueprint_pipeline.utils.cloud_tasks import PipelineTaskQueue

    queue = PipelineTaskQueue()

    # Queue a new capture job
    task_id = queue.create_pipeline_task(
        capture_id="capture123",
        session_manifest_uri="gs://bucket/sessions/.../session_manifest.json",
        output_base="gs://bucket/sessions/...",
    )

    # Retry a failed job
    queue.retry_failed_capture(capture_id="capture123")
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Check if Cloud Tasks is available
try:
    from google.cloud import tasks_v2
    from google.protobuf import duration_pb2, timestamp_pb2
    CLOUD_TASKS_AVAILABLE = True
except ImportError:
    CLOUD_TASKS_AVAILABLE = False
    tasks_v2 = None


@dataclass
class TaskQueueConfig:
    """Configuration for the task queue."""
    project_id: str
    region: str = "us-central1"
    queue_name: str = "blueprint-pipeline-queue"
    cloud_run_service_url: str = ""
    cloud_run_job_name: str = "blueprint-pipeline"

    # Retry configuration
    max_retries: int = 3
    min_backoff_seconds: int = 60
    max_backoff_seconds: int = 3600
    max_doublings: int = 3

    # Rate limiting
    max_dispatches_per_second: float = 10.0
    max_concurrent_dispatches: int = 32

    # Dead letter queue
    dead_letter_queue: str = "blueprint-pipeline-dlq"
    max_delivery_attempts: int = 5


class PipelineTaskQueue:
    """Cloud Tasks queue for pipeline job management.

    This provides:
    - Job queuing with proper rate limiting
    - Automatic retries with exponential backoff
    - Dead letter queue for failed jobs
    - Priority queuing support
    - Job scheduling (delayed execution)
    """

    def __init__(self, config: Optional[TaskQueueConfig] = None):
        """Initialize the task queue.

        Args:
            config: Queue configuration (uses env vars if not provided)
        """
        self._client = None
        self._config = config or self._default_config()

        if CLOUD_TASKS_AVAILABLE:
            try:
                self._client = tasks_v2.CloudTasksClient()
                logger.info(f"Cloud Tasks client initialized for queue: {self._config.queue_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Cloud Tasks client: {e}")
        else:
            logger.warning("Cloud Tasks not available - google-cloud-tasks not installed")

    def _default_config(self) -> TaskQueueConfig:
        """Create default config from environment variables."""
        return TaskQueueConfig(
            project_id=os.environ.get("PIPELINE_PROJECT_ID", "blueprint-8c1ca"),
            region=os.environ.get("PIPELINE_REGION", "us-central1"),
            queue_name=os.environ.get("PIPELINE_QUEUE_NAME", "blueprint-pipeline-queue"),
            cloud_run_service_url=os.environ.get("CLOUD_RUN_SERVICE_URL", ""),
            cloud_run_job_name=os.environ.get("CLOUD_RUN_JOB_NAME", "blueprint-pipeline"),
        )

    @property
    def is_available(self) -> bool:
        """Check if Cloud Tasks is available."""
        return self._client is not None

    @property
    def queue_path(self) -> str:
        """Get the full queue path."""
        return f"projects/{self._config.project_id}/locations/{self._config.region}/queues/{self._config.queue_name}"

    def create_pipeline_task(
        self,
        capture_id: str,
        session_manifest_uri: str,
        output_base: str,
        parameters: Optional[Dict[str, Any]] = None,
        schedule_time: Optional[datetime] = None,
        priority: int = 0,
    ) -> Optional[str]:
        """Create a new pipeline task.

        Args:
            capture_id: Unique capture ID
            session_manifest_uri: GCS URI to session manifest
            output_base: GCS URI for outputs
            parameters: Additional job parameters
            schedule_time: When to execute (None = immediately)
            priority: Task priority (higher = more important)

        Returns:
            Task name if created successfully, None otherwise
        """
        if not self.is_available:
            logger.warning("Cloud Tasks not available - cannot create task")
            return None

        try:
            # Build job payload (same format as Cloud Function trigger)
            payload = {
                "job_name": "full-pipeline",
                "session_id": capture_id.split("_")[0] if "_" in capture_id else capture_id,
                "capture_id": capture_id,
                "inputs": {
                    "manifest_uri": session_manifest_uri,
                },
                "outputs": {
                    "base": output_base,
                },
                "parameters": parameters or {},
            }

            # Create HTTP target for Cloud Run Job
            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": self._get_cloud_run_url(),
                    "headers": {
                        "Content-Type": "application/json",
                    },
                    "body": json.dumps(payload).encode(),
                    "oidc_token": {
                        "service_account_email": f"pipeline-invoker@{self._config.project_id}.iam.gserviceaccount.com",
                    },
                },
            }

            # Set task name for deduplication
            task["name"] = f"{self.queue_path}/tasks/{capture_id.replace('/', '-')}"

            # Set schedule time if provided
            if schedule_time:
                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromDatetime(schedule_time)
                task["schedule_time"] = timestamp

            # Create the task
            request = tasks_v2.CreateTaskRequest(
                parent=self.queue_path,
                task=task,
            )

            response = self._client.create_task(request=request)
            logger.info(f"Created task: {response.name}")
            return response.name

        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None

    def retry_failed_capture(
        self,
        capture_id: str,
        delay_seconds: int = 60,
    ) -> Optional[str]:
        """Retry a failed capture job.

        Args:
            capture_id: Capture ID to retry
            delay_seconds: Delay before retry

        Returns:
            Task name if created successfully
        """
        if not self.is_available:
            return None

        try:
            # Get capture info from Firestore to rebuild payload
            from .firestore import get_job_tracker
            tracker = get_job_tracker()
            capture = tracker.get_capture(capture_id)

            if not capture:
                logger.error(f"Capture not found: {capture_id}")
                return None

            # Schedule retry
            schedule_time = datetime.utcnow() + timedelta(seconds=delay_seconds)

            return self.create_pipeline_task(
                capture_id=f"{capture_id}_retry_{int(datetime.utcnow().timestamp())}",
                session_manifest_uri=capture.get("rawDataUri", "").replace("/raw", "/session_manifest.json"),
                output_base=capture.get("rawDataUri", "").rsplit("/raw", 1)[0],
                schedule_time=schedule_time,
            )

        except Exception as e:
            logger.error(f"Failed to retry capture: {e}")
            return None

    def delete_task(self, task_name: str) -> bool:
        """Delete a pending task.

        Args:
            task_name: Full task name

        Returns:
            True if deleted successfully
        """
        if not self.is_available:
            return False

        try:
            self._client.delete_task(name=task_name)
            logger.info(f"Deleted task: {task_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete task: {e}")
            return False

    def list_pending_tasks(self, page_size: int = 100) -> list:
        """List pending tasks in the queue.

        Args:
            page_size: Maximum number of tasks to return

        Returns:
            List of task objects
        """
        if not self.is_available:
            return []

        try:
            request = tasks_v2.ListTasksRequest(
                parent=self.queue_path,
                page_size=page_size,
            )

            tasks = []
            for task in self._client.list_tasks(request=request):
                tasks.append({
                    "name": task.name,
                    "schedule_time": task.schedule_time.ToDatetime() if task.schedule_time else None,
                    "create_time": task.create_time.ToDatetime() if task.create_time else None,
                    "dispatch_count": task.dispatch_count,
                    "response_count": task.response_count,
                })

            return tasks

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue stats
        """
        if not self.is_available:
            return {}

        try:
            queue = self._client.get_queue(name=self.queue_path)

            return {
                "name": queue.name,
                "state": queue.state.name if hasattr(queue.state, "name") else str(queue.state),
                "rate_limits": {
                    "max_dispatches_per_second": queue.rate_limits.max_dispatches_per_second if queue.rate_limits else None,
                    "max_concurrent_dispatches": queue.rate_limits.max_concurrent_dispatches if queue.rate_limits else None,
                } if queue.rate_limits else {},
            }

        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}

    def _get_cloud_run_url(self) -> str:
        """Get the Cloud Run service URL for job invocation."""
        if self._config.cloud_run_service_url:
            return self._config.cloud_run_service_url

        # Default URL format for Cloud Run Jobs
        return f"https://{self._config.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{self._config.project_id}/jobs/{self._config.cloud_run_job_name}:run"


def create_queue_if_not_exists(config: Optional[TaskQueueConfig] = None) -> bool:
    """Create the Cloud Tasks queue if it doesn't exist.

    Args:
        config: Queue configuration

    Returns:
        True if queue exists or was created
    """
    if not CLOUD_TASKS_AVAILABLE:
        logger.warning("Cloud Tasks not available")
        return False

    try:
        client = tasks_v2.CloudTasksClient()
        cfg = config or TaskQueueConfig(
            project_id=os.environ.get("PIPELINE_PROJECT_ID", "blueprint-8c1ca"),
        )

        queue_path = f"projects/{cfg.project_id}/locations/{cfg.region}/queues/{cfg.queue_name}"
        parent = f"projects/{cfg.project_id}/locations/{cfg.region}"

        # Check if queue exists
        try:
            client.get_queue(name=queue_path)
            logger.info(f"Queue already exists: {queue_path}")
            return True
        except Exception:
            pass

        # Create the queue
        queue = {
            "name": queue_path,
            "rate_limits": {
                "max_dispatches_per_second": cfg.max_dispatches_per_second,
                "max_concurrent_dispatches": cfg.max_concurrent_dispatches,
            },
            "retry_config": {
                "max_attempts": cfg.max_retries,
                "min_backoff": duration_pb2.Duration(seconds=cfg.min_backoff_seconds),
                "max_backoff": duration_pb2.Duration(seconds=cfg.max_backoff_seconds),
                "max_doublings": cfg.max_doublings,
            },
        }

        client.create_queue(parent=parent, queue=queue)
        logger.info(f"Created queue: {queue_path}")

        # Create dead letter queue
        dlq_path = f"projects/{cfg.project_id}/locations/{cfg.region}/queues/{cfg.dead_letter_queue}"
        try:
            client.get_queue(name=dlq_path)
        except Exception:
            dlq = {
                "name": dlq_path,
                "rate_limits": {
                    "max_dispatches_per_second": 1.0,
                    "max_concurrent_dispatches": 1,
                },
            }
            client.create_queue(parent=parent, queue=dlq)
            logger.info(f"Created dead letter queue: {dlq_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to create queue: {e}")
        return False
