"""Logging and progress tracking utilities."""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager


# Cloud Logging structured format
class CloudLoggingFormatter(logging.Formatter):
    """Formatter that outputs JSON for Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "logger": record.name,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, "extra"):
            log_entry.update(record.extra)

        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")

        msg = f"{color}[{timestamp}] {record.levelname:8s}{self.RESET} {record.getMessage()}"

        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return msg


def setup_logging(
    level: int = logging.INFO,
    job_name: Optional[str] = None,
    session_id: Optional[str] = None,
    cloud_logging: bool = False,
) -> logging.Logger:
    """Set up logging for pipeline jobs.

    Args:
        level: Logging level.
        job_name: Name of the current job for context.
        session_id: Session ID for context.
        cloud_logging: If True, use JSON format for Cloud Logging.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("blueprint_pipeline")
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if cloud_logging:
        handler.setFormatter(CloudLoggingFormatter())
    else:
        handler.setFormatter(ConsoleFormatter())

    logger.addHandler(handler)

    # Add context if provided
    if job_name or session_id:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra = {}
            if job_name:
                record.extra["job_name"] = job_name
            if session_id:
                record.extra["session_id"] = session_id
            return record

        logging.setLogRecordFactory(record_factory)

    return logger


def get_logger(name: str = "blueprint_pipeline") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (will be prefixed with 'blueprint_pipeline.').

    Returns:
        Logger instance.
    """
    if not name.startswith("blueprint_pipeline"):
        name = f"blueprint_pipeline.{name}"
    return logging.getLogger(name)


@dataclass
class StageMetrics:
    """Metrics for a single processing stage."""
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    items_total: int = 0
    items_processed: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def progress_percent(self) -> float:
        if self.items_total == 0:
            return 0.0
        return (self.items_processed / self.items_total) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "duration_seconds": self.duration_seconds,
            "items_total": self.items_total,
            "items_processed": self.items_processed,
            "progress_percent": self.progress_percent,
            "errors": self.errors,
            "metadata": self.metadata,
        }


class ProgressTracker:
    """Track progress across multiple stages of a job.

    Provides:
    - Stage-level progress tracking
    - Time estimation
    - Structured logging
    - Report generation
    """

    def __init__(
        self,
        job_name: str,
        session_id: str,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize progress tracker.

        Args:
            job_name: Name of the job being tracked.
            session_id: Session ID for context.
            logger: Logger instance (creates one if not provided).
        """
        self.job_name = job_name
        self.session_id = session_id
        self.logger = logger or get_logger(job_name)

        self.stages: List[StageMetrics] = []
        self.current_stage: Optional[StageMetrics] = None
        self.job_start_time = time.time()
        self.job_metadata: Dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str, total_items: int = 0):
        """Context manager for tracking a processing stage.

        Args:
            name: Stage name for logging.
            total_items: Expected number of items to process.

        Yields:
            StageMetrics object for the stage.
        """
        # Close any existing stage
        if self.current_stage is not None:
            self.current_stage.end_time = time.time()
            self.stages.append(self.current_stage)

        # Start new stage
        stage_metrics = StageMetrics(
            stage_name=name,
            start_time=time.time(),
            items_total=total_items,
        )
        self.current_stage = stage_metrics

        self.logger.info(f"Starting stage: {name}")
        if total_items > 0:
            self.logger.info(f"  Items to process: {total_items}")

        try:
            yield stage_metrics
        except Exception as e:
            stage_metrics.errors.append(str(e))
            self.logger.error(f"Stage {name} failed: {e}")
            raise
        finally:
            stage_metrics.end_time = time.time()
            duration = stage_metrics.duration_seconds
            self.logger.info(
                f"Completed stage: {name} ({stage_metrics.items_processed}/{total_items} items in {duration:.1f}s)"
            )
            self.stages.append(stage_metrics)
            self.current_stage = None

    def update(self, items_processed: int = 1, **metadata):
        """Update progress for the current stage.

        Args:
            items_processed: Number of items just processed.
            **metadata: Additional metadata to record.
        """
        if self.current_stage is None:
            return

        self.current_stage.items_processed += items_processed
        self.current_stage.metadata.update(metadata)

        # Log progress periodically
        if self.current_stage.items_total > 0:
            progress = self.current_stage.progress_percent
            if (
                self.current_stage.items_processed % max(1, self.current_stage.items_total // 10)
                == 0
            ):
                self.logger.info(
                    f"  Progress: {self.current_stage.items_processed}/{self.current_stage.items_total} ({progress:.1f}%)"
                )

    def log_metric(self, name: str, value: Any):
        """Log a named metric for the current stage or job.

        Args:
            name: Metric name.
            value: Metric value.
        """
        if self.current_stage is not None:
            self.current_stage.metadata[name] = value
        else:
            self.job_metadata[name] = value

        self.logger.info(f"Metric: {name} = {value}")

    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log an error for the current stage.

        Args:
            message: Error message.
            exception: Optional exception object.
        """
        full_message = message
        if exception:
            full_message = f"{message}: {exception}"

        if self.current_stage is not None:
            self.current_stage.errors.append(full_message)

        self.logger.error(full_message)

    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report of job execution.

        Returns:
            Dictionary containing job metrics and stage summaries.
        """
        total_duration = time.time() - self.job_start_time

        return {
            "job_name": self.job_name,
            "session_id": self.session_id,
            "total_duration_seconds": total_duration,
            "stages": [s.to_dict() for s in self.stages],
            "metadata": self.job_metadata,
            "success": all(len(s.errors) == 0 for s in self.stages),
            "total_errors": sum(len(s.errors) for s in self.stages),
        }

    def save_report(self, path: Path):
        """Save the progress report to a JSON file.

        Args:
            path: Output path for the report.
        """
        report = self.generate_report()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"Saved progress report to: {path}")
