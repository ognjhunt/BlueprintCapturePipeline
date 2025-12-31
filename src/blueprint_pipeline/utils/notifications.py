"""Push notification support for capture completion.

This module provides FCM (Firebase Cloud Messaging) integration
to notify users when their capture processing is complete.

Features:
- Topic-based notifications (user_{user_id}_captures)
- Device token notifications
- Batch notifications
- Notification history tracking in Firestore
- User preference support

Usage:
    from blueprint_pipeline.utils.notifications import send_completion_notification

    # On capture completion
    send_completion_notification(
        user_id="user123",
        capture_id="capture456",
        scene_id="scene789",
        success=True,
    )
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if Firebase Admin is available
try:
    import firebase_admin
    from firebase_admin import credentials, messaging, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    messaging = None
    firestore = None


class NotificationType(str, Enum):
    """Types of notifications."""
    CAPTURE_COMPLETE = "capture_complete"
    CAPTURE_FAILED = "capture_failed"
    CAPTURE_PROGRESS = "capture_progress"
    PAYMENT_READY = "payment_ready"
    SYSTEM_ALERT = "system_alert"


@dataclass
class NotificationPayload:
    """Notification payload for FCM."""
    title: str
    body: str
    notification_type: NotificationType
    capture_id: Optional[str] = None
    scene_id: Optional[str] = None
    data: Dict[str, str] = field(default_factory=dict)

    # Delivery options
    priority: str = "high"  # high or normal
    ttl_seconds: int = 86400  # 24 hours
    collapse_key: Optional[str] = None

    def to_data_dict(self) -> Dict[str, str]:
        """Convert to FCM data dictionary (all values must be strings)."""
        result = {
            "type": self.notification_type.value,
            "timestamp": str(int(time.time())),
        }
        if self.capture_id:
            result["capture_id"] = self.capture_id
        if self.scene_id:
            result["scene_id"] = self.scene_id
        result.update(self.data)
        return result


@dataclass
class NotificationResult:
    """Result of sending a notification."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    recipient: Optional[str] = None  # topic or token


# Firebase app singleton
_firebase_app = None


def _init_firebase() -> bool:
    """Initialize Firebase Admin SDK.

    Returns:
        True if initialized successfully
    """
    global _firebase_app

    if _firebase_app is not None:
        return True

    if not FIREBASE_AVAILABLE:
        logger.warning("firebase-admin not installed - notifications disabled")
        return False

    try:
        # Check if already initialized
        try:
            _firebase_app = firebase_admin.get_app()
            return True
        except ValueError:
            pass

        # Try to initialize with credentials file if specified
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            cred = credentials.Certificate(creds_path)
            _firebase_app = firebase_admin.initialize_app(cred)
        else:
            # Initialize with default credentials (from metadata server on GCP)
            _firebase_app = firebase_admin.initialize_app()

        logger.info("Firebase Admin SDK initialized")
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize Firebase Admin: {e}")
        return False


def _get_firestore_client():
    """Get Firestore client for notification history."""
    if not _init_firebase():
        return None

    try:
        return firestore.client()
    except Exception as e:
        logger.warning(f"Failed to get Firestore client: {e}")
        return None


def _record_notification(
    user_id: str,
    payload: NotificationPayload,
    result: NotificationResult,
) -> None:
    """Record notification in Firestore for history/debugging."""
    db = _get_firestore_client()
    if db is None:
        return

    try:
        doc_ref = db.collection("notifications").document()
        doc_ref.set({
            "userId": user_id,
            "type": payload.notification_type.value,
            "title": payload.title,
            "body": payload.body,
            "captureId": payload.capture_id,
            "sceneId": payload.scene_id,
            "success": result.success,
            "messageId": result.message_id,
            "error": result.error,
            "createdAt": firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        logger.debug(f"Failed to record notification: {e}")


def _check_user_preferences(user_id: str) -> Dict[str, bool]:
    """Check user notification preferences from Firestore.

    Returns:
        Dictionary of preferences, defaults to all enabled
    """
    defaults = {
        "capture_complete": True,
        "capture_failed": True,
        "capture_progress": False,
        "payment_ready": True,
        "system_alert": True,
    }

    db = _get_firestore_client()
    if db is None:
        return defaults

    try:
        doc = db.collection("users").document(user_id).get()
        if doc.exists:
            prefs = doc.to_dict().get("notificationPreferences", {})
            defaults.update(prefs)
    except Exception as e:
        logger.debug(f"Failed to check user preferences: {e}")

    return defaults


def send_notification(
    user_id: str,
    payload: NotificationPayload,
    record_history: bool = True,
    check_preferences: bool = True,
) -> NotificationResult:
    """Send a notification to a user via their topic.

    Args:
        user_id: User ID to send notification to
        payload: Notification payload
        record_history: Whether to record in Firestore
        check_preferences: Whether to check user preferences

    Returns:
        NotificationResult with success status
    """
    if not _init_firebase():
        return NotificationResult(
            success=False,
            error="Firebase not available",
            recipient=f"user_{user_id}_captures",
        )

    # Check user preferences
    if check_preferences:
        prefs = _check_user_preferences(user_id)
        pref_key = payload.notification_type.value
        if not prefs.get(pref_key, True):
            logger.info(f"User {user_id} has disabled {pref_key} notifications")
            return NotificationResult(
                success=True,
                error="Notification disabled by user preference",
                recipient=f"user_{user_id}_captures",
            )

    topic = f"user_{user_id}_captures"

    try:
        # Build FCM message
        message = messaging.Message(
            notification=messaging.Notification(
                title=payload.title,
                body=payload.body,
            ),
            data=payload.to_data_dict(),
            topic=topic,
            # iOS specific options
            apns=messaging.APNSConfig(
                headers={
                    "apns-priority": "10" if payload.priority == "high" else "5",
                    "apns-expiration": str(int(time.time()) + payload.ttl_seconds),
                },
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        alert=messaging.ApsAlert(
                            title=payload.title,
                            body=payload.body,
                        ),
                        badge=1,
                        sound="default",
                        mutable_content=True,
                    ),
                ),
            ),
            # Android specific options
            android=messaging.AndroidConfig(
                priority=payload.priority,
                ttl=f"{payload.ttl_seconds}s",
                collapse_key=payload.collapse_key,
                notification=messaging.AndroidNotification(
                    icon="notification_icon",
                    color="#4285F4",
                    sound="default",
                    channel_id="captures",
                ),
            ),
            # Web push options
            webpush=messaging.WebpushConfig(
                headers={
                    "TTL": str(payload.ttl_seconds),
                    "Urgency": "high" if payload.priority == "high" else "normal",
                },
                notification=messaging.WebpushNotification(
                    title=payload.title,
                    body=payload.body,
                    icon="/icons/notification-icon.png",
                ),
            ),
        )

        response = messaging.send(message)
        logger.info(f"Notification sent to topic {topic}: {response}")

        result = NotificationResult(
            success=True,
            message_id=response,
            recipient=topic,
        )

    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        result = NotificationResult(
            success=False,
            error=str(e),
            recipient=topic,
        )

    # Record in history
    if record_history:
        _record_notification(user_id, payload, result)

    return result


def send_completion_notification(
    user_id: str,
    capture_id: str,
    scene_id: str,
    success: bool,
    error_message: Optional[str] = None,
    processing_time_seconds: Optional[float] = None,
    gaussians_uri: Optional[str] = None,
) -> bool:
    """Send push notification when capture processing completes.

    This sends a notification via FCM to the user's subscribed topic.
    Users are subscribed to topic: user_{user_id}_captures

    Args:
        user_id: User who created the capture
        capture_id: Capture/session ID
        scene_id: Target scene ID
        success: Whether processing succeeded
        error_message: Error message if failed
        processing_time_seconds: Processing duration
        gaussians_uri: URI to Gaussian splat output

    Returns:
        True if notification sent successfully
    """
    if success:
        # Format processing time nicely
        time_str = ""
        if processing_time_seconds:
            minutes = int(processing_time_seconds / 60)
            if minutes > 0:
                time_str = f" in {minutes} min"
            else:
                time_str = f" in {int(processing_time_seconds)}s"

        payload = NotificationPayload(
            title="Scan Complete!",
            body=f"Your 3D capture is ready to view{time_str}.",
            notification_type=NotificationType.CAPTURE_COMPLETE,
            capture_id=capture_id,
            scene_id=scene_id,
            data={
                "status": "completed",
                "gaussians_uri": gaussians_uri or "",
                "processing_time": str(int(processing_time_seconds or 0)),
            },
            collapse_key=f"capture_{capture_id}",
        )
    else:
        payload = NotificationPayload(
            title="Capture Processing Failed",
            body=error_message or "There was an issue processing your scan. Tap to retry.",
            notification_type=NotificationType.CAPTURE_FAILED,
            capture_id=capture_id,
            scene_id=scene_id,
            data={
                "status": "failed",
                "error": error_message or "Unknown error",
            },
            collapse_key=f"capture_{capture_id}",
        )

    result = send_notification(user_id, payload)
    return result.success


def send_progress_notification(
    user_id: str,
    capture_id: str,
    scene_id: str,
    stage: str,
    progress: float,
) -> bool:
    """Send progress update notification (silent on iOS).

    Args:
        user_id: User ID
        capture_id: Capture ID
        scene_id: Scene ID
        stage: Current processing stage
        progress: Progress 0.0-1.0

    Returns:
        True if sent successfully
    """
    payload = NotificationPayload(
        title="Processing...",
        body=f"{stage}: {int(progress * 100)}%",
        notification_type=NotificationType.CAPTURE_PROGRESS,
        capture_id=capture_id,
        scene_id=scene_id,
        data={
            "stage": stage,
            "progress": str(progress),
        },
        priority="normal",  # Lower priority for progress updates
        ttl_seconds=300,  # 5 minutes
        collapse_key=f"progress_{capture_id}",  # Replace previous progress
    )

    result = send_notification(
        user_id,
        payload,
        record_history=False,  # Don't record progress updates
    )
    return result.success


def send_to_device_token(
    device_token: str,
    title: str,
    body: str,
    data: Optional[Dict[str, str]] = None,
    notification_type: NotificationType = NotificationType.SYSTEM_ALERT,
) -> NotificationResult:
    """Send push notification to a specific device.

    Args:
        device_token: FCM device token
        title: Notification title
        body: Notification body
        data: Optional data payload
        notification_type: Type of notification

    Returns:
        NotificationResult with success status
    """
    if not _init_firebase():
        return NotificationResult(
            success=False,
            error="Firebase not available",
            recipient=device_token[:20] + "...",
        )

    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data={
                "type": notification_type.value,
                **(data or {}),
            },
            token=device_token,
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        alert=messaging.ApsAlert(title=title, body=body),
                        sound="default",
                    ),
                ),
            ),
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    sound="default",
                ),
            ),
        )

        response = messaging.send(message)
        logger.info(f"Notification sent to device: {response}")
        return NotificationResult(
            success=True,
            message_id=response,
            recipient=device_token[:20] + "...",
        )

    except Exception as e:
        logger.error(f"Failed to send notification to device: {e}")
        return NotificationResult(
            success=False,
            error=str(e),
            recipient=device_token[:20] + "...",
        )


def send_batch_notifications(
    notifications: List[tuple],
) -> List[NotificationResult]:
    """Send multiple notifications in a batch.

    Args:
        notifications: List of (user_id, payload) tuples

    Returns:
        List of NotificationResult for each notification
    """
    if not _init_firebase():
        return [
            NotificationResult(success=False, error="Firebase not available")
            for _ in notifications
        ]

    if not notifications:
        return []

    # FCM supports up to 500 messages per batch
    results = []
    batch_size = 500

    for i in range(0, len(notifications), batch_size):
        batch = notifications[i:i + batch_size]

        messages = []
        for user_id, payload in batch:
            topic = f"user_{user_id}_captures"
            message = messaging.Message(
                notification=messaging.Notification(
                    title=payload.title,
                    body=payload.body,
                ),
                data=payload.to_data_dict(),
                topic=topic,
            )
            messages.append(message)

        try:
            response = messaging.send_all(messages)

            for j, send_response in enumerate(response.responses):
                if send_response.success:
                    results.append(NotificationResult(
                        success=True,
                        message_id=send_response.message_id,
                        recipient=f"user_{batch[j][0]}_captures",
                    ))
                else:
                    results.append(NotificationResult(
                        success=False,
                        error=str(send_response.exception),
                        recipient=f"user_{batch[j][0]}_captures",
                    ))

            logger.info(f"Batch sent: {response.success_count} success, {response.failure_count} failed")

        except Exception as e:
            logger.error(f"Batch send failed: {e}")
            results.extend([
                NotificationResult(success=False, error=str(e))
                for _ in batch
            ])

    return results


def subscribe_to_topic(device_token: str, topic: str) -> bool:
    """Subscribe a device to a topic.

    Args:
        device_token: FCM device token
        topic: Topic name to subscribe to

    Returns:
        True if subscribed successfully
    """
    if not _init_firebase():
        return False

    try:
        response = messaging.subscribe_to_topic([device_token], topic)
        logger.info(f"Subscribed to topic {topic}: {response.success_count} success")
        return response.success_count > 0

    except Exception as e:
        logger.error(f"Failed to subscribe to topic: {e}")
        return False


def unsubscribe_from_topic(device_token: str, topic: str) -> bool:
    """Unsubscribe a device from a topic.

    Args:
        device_token: FCM device token
        topic: Topic name to unsubscribe from

    Returns:
        True if unsubscribed successfully
    """
    if not _init_firebase():
        return False

    try:
        response = messaging.unsubscribe_from_topic([device_token], topic)
        logger.info(f"Unsubscribed from topic {topic}: {response.success_count} success")
        return response.success_count > 0

    except Exception as e:
        logger.error(f"Failed to unsubscribe from topic: {e}")
        return False


def subscribe_user_device(user_id: str, device_token: str) -> bool:
    """Subscribe a user's device to their capture notifications topic.

    Args:
        user_id: User ID
        device_token: FCM device token

    Returns:
        True if subscribed successfully
    """
    topic = f"user_{user_id}_captures"
    return subscribe_to_topic(device_token, topic)


def unsubscribe_user_device(user_id: str, device_token: str) -> bool:
    """Unsubscribe a user's device from their capture notifications topic.

    Args:
        user_id: User ID
        device_token: FCM device token

    Returns:
        True if unsubscribed successfully
    """
    topic = f"user_{user_id}_captures"
    return unsubscribe_from_topic(device_token, topic)


# =============================================================================
# Admin/Debug Functions
# =============================================================================

def send_test_notification(user_id: str) -> NotificationResult:
    """Send a test notification to a user.

    Args:
        user_id: User ID to send test notification to

    Returns:
        NotificationResult
    """
    payload = NotificationPayload(
        title="Test Notification",
        body="This is a test notification from Blueprint Pipeline.",
        notification_type=NotificationType.SYSTEM_ALERT,
        data={"test": "true"},
    )
    return send_notification(user_id, payload, record_history=True)


def get_notification_history(
    user_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get notification history for a user.

    Args:
        user_id: User ID
        limit: Maximum number of notifications to return

    Returns:
        List of notification records
    """
    db = _get_firestore_client()
    if db is None:
        return []

    try:
        query = (
            db.collection("notifications")
            .where("userId", "==", user_id)
            .order_by("createdAt", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )

        return [doc.to_dict() for doc in query.stream()]

    except Exception as e:
        logger.error(f"Failed to get notification history: {e}")
        return []
