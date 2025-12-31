"""Push notification support for capture completion.

This module provides FCM (Firebase Cloud Messaging) integration
to notify users when their capture processing is complete.

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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Check if Firebase Admin is available
try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    messaging = None

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

        # Initialize with default credentials (from GOOGLE_APPLICATION_CREDENTIALS or metadata)
        _firebase_app = firebase_admin.initialize_app()
        logger.info("Firebase Admin SDK initialized")
        return True

    except Exception as e:
        logger.warning(f"Failed to initialize Firebase Admin: {e}")
        return False


def send_completion_notification(
    user_id: str,
    capture_id: str,
    scene_id: str,
    success: bool,
    error_message: Optional[str] = None,
    processing_time_seconds: Optional[float] = None,
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

    Returns:
        True if notification sent successfully
    """
    if not _init_firebase():
        logger.info("Firebase not available - skipping notification")
        return False

    try:
        # Build notification content
        if success:
            title = "Capture Complete! 🎉"
            body = f"Your scan is ready to view."
            data = {
                "type": "capture_complete",
                "capture_id": capture_id,
                "scene_id": scene_id,
                "status": "completed",
            }
        else:
            title = "Capture Processing Failed"
            body = error_message or "There was an issue processing your scan."
            data = {
                "type": "capture_failed",
                "capture_id": capture_id,
                "scene_id": scene_id,
                "status": "failed",
                "error": error_message or "Unknown error",
            }

        if processing_time_seconds is not None:
            data["processing_time_seconds"] = str(int(processing_time_seconds))

        # Send to user's topic
        topic = f"user_{user_id}_captures"

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data=data,
            topic=topic,
            # iOS specific options
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(
                        alert=messaging.ApsAlert(
                            title=title,
                            body=body,
                        ),
                        badge=1,
                        sound="default",
                    ),
                ),
            ),
            # Android specific options
            android=messaging.AndroidConfig(
                priority="high",
                notification=messaging.AndroidNotification(
                    icon="notification_icon",
                    color="#4285F4",
                    sound="default",
                ),
            ),
        )

        response = messaging.send(message)
        logger.info(f"Notification sent to topic {topic}: {response}")
        return True

    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


def send_to_device_token(
    device_token: str,
    title: str,
    body: str,
    data: Optional[Dict[str, str]] = None,
) -> bool:
    """Send push notification to a specific device.

    Args:
        device_token: FCM device token
        title: Notification title
        body: Notification body
        data: Optional data payload

    Returns:
        True if notification sent successfully
    """
    if not _init_firebase():
        return False

    try:
        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
            ),
            data=data or {},
            token=device_token,
        )

        response = messaging.send(message)
        logger.info(f"Notification sent to device: {response}")
        return True

    except Exception as e:
        logger.error(f"Failed to send notification to device: {e}")
        return False


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
