"""Cloud Tasks integration for scalable multi-region job queuing.

This module provides Cloud Tasks integration for queueing pipeline jobs with:
- Multi-region GPU deployment
- Dynamic region selection based on queue depth
- Priority queuing (free/pro/enterprise tiers)
- Automatic retries with exponential backoff
- Dead letter queue support
- Cost controls and quota management

Usage:
    from blueprint_pipeline.utils.cloud_tasks import PipelineTaskQueue, TaskPriority

    queue = PipelineTaskQueue()

    # Queue a new capture job with priority
    task_id = queue.create_pipeline_task(
        capture_id="capture123",
        session_manifest_uri="gs://bucket/sessions/.../session_manifest.json",
        output_base="gs://bucket/sessions/...",
        priority=TaskPriority.HIGH,
    )

    # Retry a failed job
    queue.retry_failed_capture(capture_id="capture123")
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if Cloud Tasks is available
try:
    from google.cloud import tasks_v2
    from google.protobuf import duration_pb2, timestamp_pb2
    CLOUD_TASKS_AVAILABLE = True
except ImportError:
    CLOUD_TASKS_AVAILABLE = False
    tasks_v2 = None


class TaskPriority(int, Enum):
    """Task priority levels for different user tiers."""
    LOW = 0       # Free tier - queued after others
    NORMAL = 1    # Standard - default priority
    HIGH = 2      # Premium - prioritized processing
    URGENT = 3    # Enterprise SLA - immediate processing


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: str
    queue_name: str = "blueprint-pipeline-queue"
    cloud_run_job_name: str = "blueprint-pipeline"
    max_queue_depth: int = 50  # Consider region "busy" above this
    weight: float = 1.0  # For weighted random selection
    enabled: bool = True

    @property
    def queue_path(self) -> str:
        return f"projects/{{}}/locations/{self.region}/queues/{self.queue_name}"


@dataclass
class TaskQueueConfig:
    """Configuration for the task queue."""
    project_id: str
    primary_region: str = "us-central1"
    secondary_regions: List[str] = field(default_factory=lambda: ["us-east1", "europe-west1"])

    # Queue configuration
    queue_name: str = "blueprint-pipeline-queue"
    cloud_run_job_name: str = "blueprint-pipeline"

    # Retry configuration
    max_retries: int = 3
    min_backoff_seconds: int = 60
    max_backoff_seconds: int = 3600
    max_doublings: int = 3

    # Rate limiting (per region)
    max_dispatches_per_second: float = 10.0
    max_concurrent_dispatches: int = 32

    # Multi-region settings
    max_queue_depth_threshold: int = 50  # Route to secondary if primary exceeds this
    enable_multi_region: bool = True

    # Dead letter queue
    dead_letter_queue: str = "blueprint-pipeline-dlq"
    max_delivery_attempts: int = 5

    # Priority queue settings (separate queues per priority)
    enable_priority_queues: bool = True

    @property
    def all_regions(self) -> List[str]:
        return [self.primary_region] + self.secondary_regions


class PipelineTaskQueue:
    """Cloud Tasks queue for pipeline job management with multi-region support.

    This provides:
    - Multi-region job distribution with automatic failover
    - Priority queuing for different user tiers
    - Automatic retries with exponential backoff
    - Dead letter queue for failed jobs
    - Queue depth monitoring for load balancing
    """

    def __init__(self, config: Optional[TaskQueueConfig] = None):
        """Initialize the task queue.

        Args:
            config: Queue configuration (uses env vars if not provided)
        """
        self._client = None
        self._config = config or self._default_config()
        self._region_configs: Dict[str, RegionConfig] = {}
        self._queue_depth_cache: Dict[str, Tuple[int, float]] = {}  # (depth, timestamp)
        self._cache_ttl = 30  # seconds

        if CLOUD_TASKS_AVAILABLE:
            try:
                self._client = tasks_v2.CloudTasksClient()
                self._init_region_configs()
                logger.info(f"Cloud Tasks client initialized for regions: {self._config.all_regions}")
            except Exception as e:
                logger.warning(f"Failed to initialize Cloud Tasks client: {e}")
        else:
            logger.warning("Cloud Tasks not available - google-cloud-tasks not installed")

    def _default_config(self) -> TaskQueueConfig:
        """Create default config from environment variables."""
        secondary = os.environ.get("PIPELINE_SECONDARY_REGIONS", "us-east1,europe-west1")
        secondary_regions = [r.strip() for r in secondary.split(",") if r.strip()]

        return TaskQueueConfig(
            project_id=os.environ.get("PIPELINE_PROJECT_ID", "blueprint-8c1ca"),
            primary_region=os.environ.get("PIPELINE_REGION", "us-central1"),
            secondary_regions=secondary_regions,
            queue_name=os.environ.get("PIPELINE_QUEUE_NAME", "blueprint-pipeline-queue"),
            cloud_run_job_name=os.environ.get("CLOUD_RUN_JOB_NAME", "blueprint-pipeline"),
            enable_multi_region=os.environ.get("ENABLE_MULTI_REGION", "true").lower() == "true",
            enable_priority_queues=os.environ.get("ENABLE_PRIORITY_QUEUES", "true").lower() == "true",
        )

    def _init_region_configs(self):
        """Initialize region configurations."""
        for region in self._config.all_regions:
            weight = 1.0 if region == self._config.primary_region else 0.5
            self._region_configs[region] = RegionConfig(
                region=region,
                queue_name=self._config.queue_name,
                cloud_run_job_name=self._config.cloud_run_job_name,
                max_queue_depth=self._config.max_queue_depth_threshold,
                weight=weight,
            )

    @property
    def is_available(self) -> bool:
        """Check if Cloud Tasks is available."""
        return self._client is not None

    def _get_queue_path(self, region: str, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Get the full queue path for a region and priority."""
        queue_name = self._config.queue_name
        if self._config.enable_priority_queues and priority != TaskPriority.NORMAL:
            queue_name = f"{queue_name}-{priority.name.lower()}"
        return f"projects/{self._config.project_id}/locations/{region}/queues/{queue_name}"

    def _get_queue_depth(self, region: str, priority: TaskPriority = TaskPriority.NORMAL) -> int:
        """Get the current queue depth for a region.

        Uses caching to avoid excessive API calls.
        """
        cache_key = f"{region}:{priority.value}"
        now = time.time()

        # Check cache
        if cache_key in self._queue_depth_cache:
            depth, timestamp = self._queue_depth_cache[cache_key]
            if now - timestamp < self._cache_ttl:
                return depth

        # Query queue
        try:
            queue_path = self._get_queue_path(region, priority)
            request = tasks_v2.ListTasksRequest(
                parent=queue_path,
                page_size=1000,  # Max page size
            )
            tasks = list(self._client.list_tasks(request=request))
            depth = len(tasks)

            # Update cache
            self._queue_depth_cache[cache_key] = (depth, now)
            return depth

        except Exception as e:
            logger.warning(f"Failed to get queue depth for {region}: {e}")
            return 0  # Assume empty on error

    def _select_best_region(self, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Select the best region for a new task based on queue depth.

        Uses weighted random selection with bias towards less loaded regions.
        """
        if not self._config.enable_multi_region:
            return self._config.primary_region

        # Get queue depths for all regions
        region_depths: Dict[str, int] = {}
        for region in self._config.all_regions:
            config = self._region_configs.get(region)
            if config and config.enabled:
                depth = self._get_queue_depth(region, priority)
                region_depths[region] = depth

        if not region_depths:
            return self._config.primary_region

        # Find regions under threshold
        available_regions = [
            region for region, depth in region_depths.items()
            if depth < self._config.max_queue_depth_threshold
        ]

        if not available_regions:
            # All regions are busy, use the one with lowest depth
            return min(region_depths, key=region_depths.get)

        # Weighted random selection (inverse of queue depth)
        weights = []
        for region in available_regions:
            depth = region_depths[region]
            base_weight = self._region_configs[region].weight
            # Higher weight for lower queue depth
            weight = base_weight * (1.0 / (depth + 1))
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Select region
        selected = random.choices(available_regions, weights=weights, k=1)[0]
        logger.debug(f"Selected region {selected} (depths: {region_depths})")
        return selected

    def create_pipeline_task(
        self,
        capture_id: str,
        session_manifest_uri: str,
        output_base: str,
        user_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        schedule_time: Optional[datetime] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        target_region: Optional[str] = None,
    ) -> Optional[str]:
        """Create a new pipeline task with priority and region selection.

        Args:
            capture_id: Unique capture ID
            session_manifest_uri: GCS URI to session manifest
            output_base: GCS URI for outputs
            user_id: Optional user ID for quota tracking
            parameters: Additional job parameters
            schedule_time: When to execute (None = immediately)
            priority: Task priority level
            target_region: Force specific region (auto-select if None)

        Returns:
            Task name if created successfully, None otherwise
        """
        if not self.is_available:
            logger.warning("Cloud Tasks not available - cannot create task")
            return None

        # Select region
        region = target_region or self._select_best_region(priority)

        try:
            # Build job payload
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
                "metadata": {
                    "priority": priority.value,
                    "priority_name": priority.name,
                    "user_id": user_id or "unknown",
                    "region": region,
                    "queued_at": datetime.utcnow().isoformat() + "Z",
                },
            }

            # Get queue path
            queue_path = self._get_queue_path(region, priority)

            # Create HTTP target for Cloud Run Job
            task = {
                "http_request": {
                    "http_method": tasks_v2.HttpMethod.POST,
                    "url": self._get_cloud_run_url(region),
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
            task_id = f"{capture_id.replace('/', '-')}-{int(time.time())}"
            task["name"] = f"{queue_path}/tasks/{task_id}"

            # Set schedule time if provided
            if schedule_time:
                timestamp = timestamp_pb2.Timestamp()
                timestamp.FromDatetime(schedule_time)
                task["schedule_time"] = timestamp

            # Set dispatch deadline based on priority
            if priority == TaskPriority.URGENT:
                # Urgent tasks get dispatched within 10 seconds
                task["dispatch_deadline"] = duration_pb2.Duration(seconds=10)
            elif priority == TaskPriority.HIGH:
                task["dispatch_deadline"] = duration_pb2.Duration(seconds=30)

            # Create the task
            request = tasks_v2.CreateTaskRequest(
                parent=queue_path,
                task=task,
            )

            response = self._client.create_task(request=request)

            # Invalidate queue depth cache
            cache_key = f"{region}:{priority.value}"
            if cache_key in self._queue_depth_cache:
                del self._queue_depth_cache[cache_key]

            logger.info(f"Created task in {region}: {response.name}")
            return response.name

        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None

    def dispatch_to_available_region(
        self,
        capture_id: str,
        session_manifest_uri: str,
        output_base: str,
        user_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Optional[str]:
        """Route to region with available GPU capacity.

        Implements intelligent load balancing across regions:
        1. Check queue depths for all enabled regions
        2. Filter regions under threshold
        3. Select best available region
        4. Fallback to primary if all are busy

        Args:
            capture_id: Unique capture ID
            session_manifest_uri: GCS URI to session manifest
            output_base: GCS URI for outputs
            user_id: Optional user ID for quota tracking
            parameters: Additional job parameters
            priority: Task priority level

        Returns:
            Task name if created successfully, None otherwise
        """
        # Get queue depths for all regions
        region_loads: Dict[str, Dict] = {}

        for region in self._config.all_regions:
            config = self._region_configs.get(region)
            if config and config.enabled:
                depth = self._get_queue_depth(region, priority)
                region_loads[region] = {
                    "depth": depth,
                    "available": depth < config.max_queue_depth,
                    "load_percent": min(100, int(depth / config.max_queue_depth * 100)),
                }

        logger.info(f"Region loads: {region_loads}")

        # Find best available region
        available_regions = [
            region for region, info in region_loads.items()
            if info["available"]
        ]

        if not available_regions:
            logger.warning("All regions are at capacity, using primary region")
            target_region = self._config.primary_region
        else:
            # Sort by load and select least loaded
            available_regions.sort(key=lambda r: region_loads[r]["depth"])
            target_region = available_regions[0]

        return self.create_pipeline_task(
            capture_id=capture_id,
            session_manifest_uri=session_manifest_uri,
            output_base=output_base,
            user_id=user_id,
            parameters=parameters,
            priority=priority,
            target_region=target_region,
        )

    def retry_failed_capture(
        self,
        capture_id: str,
        delay_seconds: int = 60,
        increment_retry_count: bool = True,
    ) -> Optional[str]:
        """Retry a failed capture job.

        Args:
            capture_id: Capture ID to retry
            delay_seconds: Delay before retry
            increment_retry_count: Whether to increment retry counter

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

            # Check retry count
            retry_count = capture.get("retryCount", 0)
            if retry_count >= self._config.max_retries:
                logger.warning(f"Capture {capture_id} has exceeded max retries ({retry_count})")
                return None

            # Update retry count if requested
            if increment_retry_count:
                tracker.update_status(capture_id=capture_id, error=None)

            # Schedule retry
            schedule_time = datetime.utcnow() + timedelta(seconds=delay_seconds)

            # Use same priority as original or bump to HIGH for retries
            original_priority = capture.get("metadata", {}).get("priority", TaskPriority.NORMAL.value)
            retry_priority = max(original_priority, TaskPriority.HIGH.value)

            raw_data_uri = capture.get("rawDataUri", "")
            return self.create_pipeline_task(
                capture_id=f"{capture_id}_retry_{retry_count + 1}",
                session_manifest_uri=raw_data_uri.replace("/raw", "/session_manifest.json"),
                output_base=raw_data_uri.rsplit("/raw", 1)[0],
                user_id=capture.get("creatorId"),
                schedule_time=schedule_time,
                priority=TaskPriority(retry_priority),
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

    def list_pending_tasks(
        self,
        region: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        page_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """List pending tasks in a queue.

        Args:
            region: Region to list (all regions if None)
            priority: Priority queue to list
            page_size: Maximum number of tasks per region

        Returns:
            List of task objects
        """
        if not self.is_available:
            return []

        tasks = []
        regions = [region] if region else self._config.all_regions

        for r in regions:
            try:
                queue_path = self._get_queue_path(r, priority)
                request = tasks_v2.ListTasksRequest(
                    parent=queue_path,
                    page_size=page_size,
                )

                for task in self._client.list_tasks(request=request):
                    tasks.append({
                        "name": task.name,
                        "region": r,
                        "priority": priority.name,
                        "schedule_time": task.schedule_time.ToDatetime() if task.schedule_time else None,
                        "create_time": task.create_time.ToDatetime() if task.create_time else None,
                        "dispatch_count": task.dispatch_count,
                        "response_count": task.response_count,
                    })

            except Exception as e:
                logger.error(f"Failed to list tasks for {r}: {e}")

        return tasks

    def get_queue_stats(self, include_all_regions: bool = True) -> Dict[str, Any]:
        """Get queue statistics across all regions.

        Returns:
            Dictionary with queue stats
        """
        if not self.is_available:
            return {}

        stats = {
            "regions": {},
            "total_pending": 0,
            "total_capacity": 0,
        }

        regions = self._config.all_regions if include_all_regions else [self._config.primary_region]

        for region in regions:
            try:
                region_stats = {
                    "queues": {},
                    "total_depth": 0,
                }

                # Check each priority queue
                for priority in TaskPriority:
                    queue_path = self._get_queue_path(region, priority)

                    try:
                        queue = self._client.get_queue(name=queue_path)
                        depth = self._get_queue_depth(region, priority)

                        region_stats["queues"][priority.name] = {
                            "depth": depth,
                            "state": queue.state.name if hasattr(queue.state, "name") else str(queue.state),
                            "rate_limit": queue.rate_limits.max_dispatches_per_second if queue.rate_limits else None,
                        }
                        region_stats["total_depth"] += depth

                    except Exception:
                        # Queue might not exist for this priority
                        pass

                config = self._region_configs.get(region)
                region_stats["max_capacity"] = config.max_queue_depth if config else self._config.max_queue_depth_threshold
                region_stats["available"] = region_stats["total_depth"] < region_stats["max_capacity"]
                region_stats["load_percent"] = min(100, int(region_stats["total_depth"] / region_stats["max_capacity"] * 100))

                stats["regions"][region] = region_stats
                stats["total_pending"] += region_stats["total_depth"]
                stats["total_capacity"] += region_stats["max_capacity"]

            except Exception as e:
                logger.error(f"Failed to get queue stats for {region}: {e}")
                stats["regions"][region] = {"error": str(e)}

        return stats

    def _get_cloud_run_url(self, region: str) -> str:
        """Get the Cloud Run service URL for job invocation in a region."""
        # Cloud Run Jobs invocation URL
        return f"https://{region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/{self._config.project_id}/jobs/{self._config.cloud_run_job_name}:run"


# =============================================================================
# Queue Setup Functions
# =============================================================================

def create_queues_for_region(
    project_id: str,
    region: str,
    config: Optional[TaskQueueConfig] = None,
    create_priority_queues: bool = True,
) -> bool:
    """Create all necessary Cloud Tasks queues for a region.

    Args:
        project_id: GCP project ID
        region: Region to create queues in
        config: Queue configuration
        create_priority_queues: Whether to create separate priority queues

    Returns:
        True if all queues created successfully
    """
    if not CLOUD_TASKS_AVAILABLE:
        logger.warning("Cloud Tasks not available")
        return False

    try:
        client = tasks_v2.CloudTasksClient()
        cfg = config or TaskQueueConfig(project_id=project_id)
        parent = f"projects/{project_id}/locations/{region}"

        queues_to_create = [cfg.queue_name]

        # Add priority queues if enabled
        if create_priority_queues:
            for priority in TaskPriority:
                if priority != TaskPriority.NORMAL:
                    queues_to_create.append(f"{cfg.queue_name}-{priority.name.lower()}")

        # Add dead letter queue
        queues_to_create.append(cfg.dead_letter_queue)

        for queue_name in queues_to_create:
            queue_path = f"{parent}/queues/{queue_name}"

            # Check if queue exists
            try:
                client.get_queue(name=queue_path)
                logger.info(f"Queue already exists: {queue_path}")
                continue
            except Exception:
                pass

            # Create the queue
            is_dlq = queue_name == cfg.dead_letter_queue
            is_high_priority = "high" in queue_name.lower() or "urgent" in queue_name.lower()

            queue = {
                "name": queue_path,
                "rate_limits": {
                    "max_dispatches_per_second": 1.0 if is_dlq else (20.0 if is_high_priority else cfg.max_dispatches_per_second),
                    "max_concurrent_dispatches": 1 if is_dlq else (50 if is_high_priority else cfg.max_concurrent_dispatches),
                },
            }

            if not is_dlq:
                queue["retry_config"] = {
                    "max_attempts": cfg.max_retries,
                    "min_backoff": duration_pb2.Duration(seconds=cfg.min_backoff_seconds),
                    "max_backoff": duration_pb2.Duration(seconds=cfg.max_backoff_seconds),
                    "max_doublings": cfg.max_doublings,
                }

            client.create_queue(parent=parent, queue=queue)
            logger.info(f"Created queue: {queue_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to create queues for {region}: {e}")
        return False


def create_all_queues(config: Optional[TaskQueueConfig] = None) -> bool:
    """Create Cloud Tasks queues in all configured regions.

    Args:
        config: Queue configuration

    Returns:
        True if all queues created successfully
    """
    cfg = config or TaskQueueConfig(
        project_id=os.environ.get("PIPELINE_PROJECT_ID", "blueprint-8c1ca"),
    )

    success = True
    for region in cfg.all_regions:
        if not create_queues_for_region(cfg.project_id, region, cfg):
            success = False

    return success


# =============================================================================
# Singleton Instance
# =============================================================================

_task_queue: Optional[PipelineTaskQueue] = None


def get_task_queue() -> PipelineTaskQueue:
    """Get or create the singleton task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = PipelineTaskQueue()
    return _task_queue
