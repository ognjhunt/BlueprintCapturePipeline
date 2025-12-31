"""Firestore integration for job status tracking.

This module provides utilities for tracking capture/job status in Firestore,
enabling visibility into the pipeline state from the iOS app.

Collection: captures
Document schema:
{
    "id": str,                    # Capture/session ID
    "sceneId": str,               # Target scene ID
    "creatorId": str,             # User who created the capture
    "status": str,                # queued | processing | completed | failed
    "stage": str,                 # Current processing stage
    "progress": float,            # 0.0 - 1.0
    "createdAt": Timestamp,
    "updatedAt": Timestamp,
    "startedAt": Timestamp | None,
    "completedAt": Timestamp | None,
    "error": str | None,
    "retryCount": int,
    "outputs": {
        "gaussiansUri": str | None,
        "posesUri": str | None,
        "framesUri": str | None,
    },
    "metrics": {
        "totalFrames": int,
        "processingTimeSeconds": float,
    },
    "source": str,                # "iphone" | "meta_glasses"
    "rawDataUri": str,            # GCS URI to raw upload
}
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if Firestore is available
try:
    from google.cloud import firestore
    from google.cloud.firestore_v1 import SERVER_TIMESTAMP
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    firestore = None
    SERVER_TIMESTAMP = None


class CaptureStatus(str, Enum):
    """Status of a capture in the pipeline."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingStage(str, Enum):
    """Current processing stage."""
    UPLOAD_COMPLETE = "upload_complete"
    FRAME_EXTRACTION = "frame_extraction"
    RECONSTRUCTION = "reconstruction"
    EXPORT = "export"
    DONE = "done"


@dataclass
class CaptureOutputs:
    """Output URIs from pipeline processing."""
    gaussians_uri: Optional[str] = None
    poses_uri: Optional[str] = None
    frames_uri: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gaussiansUri": self.gaussians_uri,
            "posesUri": self.poses_uri,
            "framesUri": self.frames_uri,
        }


@dataclass
class CaptureMetrics:
    """Metrics from pipeline processing."""
    total_frames: int = 0
    processing_time_seconds: float = 0.0
    registration_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "totalFrames": self.total_frames,
            "processingTimeSeconds": self.processing_time_seconds,
            "registrationRate": self.registration_rate,
        }


@dataclass
class CaptureDocument:
    """Firestore document for a capture."""
    id: str
    scene_id: str
    creator_id: str
    status: CaptureStatus = CaptureStatus.QUEUED
    stage: ProcessingStage = ProcessingStage.UPLOAD_COMPLETE
    progress: float = 0.0
    error: Optional[str] = None
    retry_count: int = 0
    source: str = "iphone"
    raw_data_uri: str = ""
    outputs: CaptureOutputs = field(default_factory=CaptureOutputs)
    metrics: CaptureMetrics = field(default_factory=CaptureMetrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Firestore document dict."""
        return {
            "id": self.id,
            "sceneId": self.scene_id,
            "creatorId": self.creator_id,
            "status": self.status.value,
            "stage": self.stage.value,
            "progress": self.progress,
            "error": self.error,
            "retryCount": self.retry_count,
            "source": self.source,
            "rawDataUri": self.raw_data_uri,
            "outputs": self.outputs.to_dict(),
            "metrics": self.metrics.to_dict(),
            "updatedAt": SERVER_TIMESTAMP if FIRESTORE_AVAILABLE else datetime.utcnow().isoformat(),
        }


class FirestoreJobTracker:
    """Track job status in Firestore.

    Usage:
        tracker = FirestoreJobTracker()

        # Create initial status when upload completes
        tracker.create_capture(
            capture_id="abc123",
            scene_id="scene456",
            creator_id="user789",
            source="iphone",
            raw_data_uri="gs://bucket/scenes/...",
        )

        # Update status during processing
        tracker.update_status(
            capture_id="abc123",
            status=CaptureStatus.PROCESSING,
            stage=ProcessingStage.FRAME_EXTRACTION,
            progress=0.25,
        )

        # Mark complete
        tracker.mark_completed(
            capture_id="abc123",
            outputs=CaptureOutputs(
                gaussians_uri="gs://bucket/sessions/.../gaussians.ply",
                poses_uri="gs://bucket/sessions/.../poses.json",
            ),
            metrics=CaptureMetrics(total_frames=150, processing_time_seconds=600),
        )

        # Mark failed
        tracker.mark_failed(
            capture_id="abc123",
            error="SLAM reconstruction failed: insufficient parallax",
        )
    """

    COLLECTION = "captures"

    def __init__(self, project_id: Optional[str] = None):
        """Initialize Firestore client.

        Args:
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        """
        self._db = None
        self._project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PIPELINE_PROJECT_ID")

        if FIRESTORE_AVAILABLE:
            try:
                self._db = firestore.Client(project=self._project_id)
                logger.info(f"Firestore client initialized for project: {self._project_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize Firestore client: {e}")
        else:
            logger.warning("Firestore not available - google-cloud-firestore not installed")

    @property
    def is_available(self) -> bool:
        """Check if Firestore is available."""
        return self._db is not None

    def create_capture(
        self,
        capture_id: str,
        scene_id: str,
        creator_id: str,
        source: str = "iphone",
        raw_data_uri: str = "",
    ) -> bool:
        """Create initial capture document when upload completes.

        Args:
            capture_id: Unique capture/session ID
            scene_id: Target scene ID
            creator_id: User who created the capture
            source: "iphone" or "meta_glasses"
            raw_data_uri: GCS URI to raw upload directory

        Returns:
            True if created successfully
        """
        if not self.is_available:
            logger.warning("Firestore not available - skipping create_capture")
            return False

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(capture_id)

            doc_data = {
                "id": capture_id,
                "sceneId": scene_id,
                "creatorId": creator_id,
                "status": CaptureStatus.QUEUED.value,
                "stage": ProcessingStage.UPLOAD_COMPLETE.value,
                "progress": 0.0,
                "error": None,
                "retryCount": 0,
                "source": source,
                "rawDataUri": raw_data_uri,
                "outputs": {
                    "gaussiansUri": None,
                    "posesUri": None,
                    "framesUri": None,
                },
                "metrics": {
                    "totalFrames": 0,
                    "processingTimeSeconds": 0.0,
                    "registrationRate": 0.0,
                },
                "createdAt": SERVER_TIMESTAMP,
                "updatedAt": SERVER_TIMESTAMP,
                "startedAt": None,
                "completedAt": None,
            }

            doc_ref.set(doc_data)
            logger.info(f"Created capture document: {capture_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to create capture document: {e}")
            return False

    def update_status(
        self,
        capture_id: str,
        status: Optional[CaptureStatus] = None,
        stage: Optional[ProcessingStage] = None,
        progress: Optional[float] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update capture status.

        Args:
            capture_id: Capture/session ID
            status: New status (optional)
            stage: New processing stage (optional)
            progress: Progress 0.0-1.0 (optional)
            error: Error message (optional)

        Returns:
            True if updated successfully
        """
        if not self.is_available:
            logger.warning("Firestore not available - skipping update_status")
            return False

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(capture_id)

            update_data = {"updatedAt": SERVER_TIMESTAMP}

            if status is not None:
                update_data["status"] = status.value
                if status == CaptureStatus.PROCESSING:
                    update_data["startedAt"] = SERVER_TIMESTAMP

            if stage is not None:
                update_data["stage"] = stage.value

            if progress is not None:
                update_data["progress"] = min(1.0, max(0.0, progress))

            if error is not None:
                update_data["error"] = error

            doc_ref.update(update_data)
            logger.debug(f"Updated capture {capture_id}: {update_data}")
            return True

        except Exception as e:
            logger.error(f"Failed to update capture status: {e}")
            return False

    def mark_completed(
        self,
        capture_id: str,
        outputs: Optional[CaptureOutputs] = None,
        metrics: Optional[CaptureMetrics] = None,
        send_notification: bool = True,
    ) -> bool:
        """Mark capture as completed.

        Args:
            capture_id: Capture/session ID
            outputs: Output URIs
            metrics: Processing metrics
            send_notification: Whether to send push notification

        Returns:
            True if updated successfully
        """
        if not self.is_available:
            logger.warning("Firestore not available - skipping mark_completed")
            return False

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(capture_id)

            # Get current document to retrieve creator_id and scene_id for notification
            doc = doc_ref.get()
            doc_data = doc.to_dict() if doc.exists else {}

            update_data = {
                "status": CaptureStatus.COMPLETED.value,
                "stage": ProcessingStage.DONE.value,
                "progress": 1.0,
                "error": None,
                "updatedAt": SERVER_TIMESTAMP,
                "completedAt": SERVER_TIMESTAMP,
            }

            if outputs:
                update_data["outputs"] = outputs.to_dict()

            if metrics:
                update_data["metrics"] = metrics.to_dict()

            doc_ref.update(update_data)
            logger.info(f"Marked capture {capture_id} as completed")

            # Send push notification
            if send_notification and doc_data:
                try:
                    from .notifications import send_completion_notification
                    send_completion_notification(
                        user_id=doc_data.get("creatorId", ""),
                        capture_id=capture_id,
                        scene_id=doc_data.get("sceneId", ""),
                        success=True,
                        processing_time_seconds=metrics.processing_time_seconds if metrics else None,
                    )
                except Exception as notif_error:
                    logger.warning(f"Failed to send completion notification: {notif_error}")

            return True

        except Exception as e:
            logger.error(f"Failed to mark capture completed: {e}")
            return False

    def mark_failed(
        self,
        capture_id: str,
        error: str,
        increment_retry: bool = True,
        send_notification: bool = True,
    ) -> bool:
        """Mark capture as failed.

        Args:
            capture_id: Capture/session ID
            error: Error message
            increment_retry: Whether to increment retry count
            send_notification: Whether to send push notification

        Returns:
            True if updated successfully
        """
        if not self.is_available:
            logger.warning("Firestore not available - skipping mark_failed")
            return False

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(capture_id)

            # Get current document for notification
            doc = doc_ref.get()
            doc_data = doc.to_dict() if doc.exists else {}

            update_data = {
                "status": CaptureStatus.FAILED.value,
                "error": error,
                "updatedAt": SERVER_TIMESTAMP,
                "completedAt": SERVER_TIMESTAMP,
            }

            if increment_retry:
                update_data["retryCount"] = firestore.Increment(1)

            doc_ref.update(update_data)
            logger.info(f"Marked capture {capture_id} as failed: {error}")

            # Send push notification for failure
            if send_notification and doc_data:
                try:
                    from .notifications import send_completion_notification
                    send_completion_notification(
                        user_id=doc_data.get("creatorId", ""),
                        capture_id=capture_id,
                        scene_id=doc_data.get("sceneId", ""),
                        success=False,
                        error_message=error,
                    )
                except Exception as notif_error:
                    logger.warning(f"Failed to send failure notification: {notif_error}")

            return True

        except Exception as e:
            logger.error(f"Failed to mark capture failed: {e}")
            return False

    def get_capture(self, capture_id: str) -> Optional[Dict[str, Any]]:
        """Get capture document.

        Args:
            capture_id: Capture/session ID

        Returns:
            Document data or None if not found
        """
        if not self.is_available:
            return None

        try:
            doc_ref = self._db.collection(self.COLLECTION).document(capture_id)
            doc = doc_ref.get()

            if doc.exists:
                return doc.to_dict()
            return None

        except Exception as e:
            logger.error(f"Failed to get capture: {e}")
            return None

    def list_pending_captures(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List captures that are queued or processing.

        Args:
            limit: Maximum number of captures to return

        Returns:
            List of capture documents
        """
        if not self.is_available:
            return []

        try:
            query = (
                self._db.collection(self.COLLECTION)
                .where("status", "in", [CaptureStatus.QUEUED.value, CaptureStatus.PROCESSING.value])
                .order_by("createdAt")
                .limit(limit)
            )

            return [doc.to_dict() for doc in query.stream()]

        except Exception as e:
            logger.error(f"Failed to list pending captures: {e}")
            return []


# Singleton instance for convenience
_tracker: Optional[FirestoreJobTracker] = None


def get_job_tracker() -> FirestoreJobTracker:
    """Get or create the singleton job tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = FirestoreJobTracker()
    return _tracker
