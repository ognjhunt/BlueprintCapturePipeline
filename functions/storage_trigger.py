"""Cloud Function triggered by Firebase Storage uploads.

This function monitors the scenes/ prefix in the Firebase Storage bucket
and triggers the GPU pipeline when a complete capture session is uploaded.

Deployment:
    gcloud functions deploy storage_trigger \
        --runtime python311 \
        --trigger-resource blueprint-8c1ca.appspot.com \
        --trigger-event google.storage.object.finalize \
        --entry-point on_storage_finalize \
        --region us-central1 \
        --memory 512MB \
        --timeout 60s \
        --set-env-vars PIPELINE_PROJECT_ID=blueprint-8c1ca,PIPELINE_REGION=us-central1
"""
from __future__ import annotations

import json
import os
import re
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Expected file markers that indicate a complete upload
COMPLETION_MARKERS = [
    "manifest.json",  # Required - capture metadata
    "walkthrough.mov",  # Required - main video file
]

# Optional files that enhance processing
OPTIONAL_FILES = [
    "motion.jsonl",
    "arkit/frames.jsonl",
    "arkit/poses.jsonl",
    "arkit/intrinsics.json",
]


def parse_upload_path(object_name: str) -> Optional[Dict[str, str]]:
    """Parse the GCS object path to extract scene metadata.

    Expected format:
        scenes/{scene_id}/{source}/{timestamp}-{uuid}/raw/{filename}

    Examples:
        scenes/ChIJ9_QNuFHkrIkR3YlZInIh5Ow/iphone/2024-12-09T15:30:45-abc123/raw/manifest.json
        scenes/scene_123/glasses/2024-12-09T10:00:00-def456/raw/walkthrough.mov

    Returns:
        Dictionary with scene_id, source, capture_folder, raw_prefix, filename
        or None if path doesn't match expected format.
    """
    # Pattern: scenes/{scene_id}/{source}/{capture_folder}/raw/{filename}
    pattern = r'^scenes/([^/]+)/([^/]+)/([^/]+)/raw/(.+)$'
    match = re.match(pattern, object_name)

    if not match:
        logger.debug(f"Path doesn't match expected pattern: {object_name}")
        return None

    scene_id, source, capture_folder, filename = match.groups()

    return {
        "scene_id": scene_id,
        "source": source,  # "iphone" or "glasses"
        "capture_folder": capture_folder,  # "{timestamp}-{uuid}"
        "raw_prefix": f"scenes/{scene_id}/{source}/{capture_folder}/raw",
        "filename": filename,
    }


def check_upload_completeness(
    bucket_name: str,
    raw_prefix: str,
) -> Tuple[bool, Dict[str, bool]]:
    """Check if all required files have been uploaded.

    Args:
        bucket_name: GCS bucket name
        raw_prefix: Prefix path to the raw/ folder

    Returns:
        Tuple of (is_complete, file_status_dict)
    """
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List all blobs under the raw prefix
        blobs = list(bucket.list_blobs(prefix=raw_prefix))
        existing_files = {blob.name.replace(f"{raw_prefix}/", ""): True for blob in blobs}

        # Check required markers
        file_status = {}
        for marker in COMPLETION_MARKERS:
            file_status[marker] = marker in existing_files

        # Check optional files
        for optional in OPTIONAL_FILES:
            file_status[optional] = optional in existing_files

        # Upload is complete if all required markers exist
        is_complete = all(file_status.get(m, False) for m in COMPLETION_MARKERS)

        logger.info(f"Upload completeness check for {raw_prefix}: {is_complete}")
        logger.info(f"File status: {file_status}")

        return is_complete, file_status

    except Exception as e:
        logger.error(f"Error checking upload completeness: {e}")
        return False, {}


def load_ios_manifest(bucket_name: str, manifest_path: str) -> Optional[Dict[str, Any]]:
    """Load and parse the iOS manifest.json file.

    Args:
        bucket_name: GCS bucket name
        manifest_path: Full path to manifest.json

    Returns:
        Parsed manifest dictionary or None on error
    """
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(manifest_path)

        content = blob.download_as_text()
        manifest = json.loads(content)

        logger.info(f"Loaded iOS manifest: {manifest.get('device_model', 'unknown device')}")
        return manifest

    except Exception as e:
        logger.error(f"Error loading iOS manifest: {e}")
        return None


def convert_ios_manifest_to_session(
    ios_manifest: Dict[str, Any],
    scene_id: str,
    source: str,
    raw_prefix: str,
    bucket_name: str,
    file_status: Dict[str, bool],
) -> Dict[str, Any]:
    """Convert iOS manifest format to pipeline SessionManifest format.

    iOS manifest fields:
        - scene_id, video_uri, device_model, os_version
        - fps_source, width, height, capture_start_epoch_ms
        - has_lidar, scale_hint_m_per_unit, intended_space_type
        - exposure_samples[], object_point_cloud_index, object_point_cloud_count

    Pipeline SessionManifest fields:
        - session_id, capture_start, device{}
        - scale_anchors[], clips[], user_notes
    """
    # Build GCS URI for video
    video_uri = f"gs://{bucket_name}/{raw_prefix}/walkthrough.mov"

    # Convert capture timestamp
    capture_epoch_ms = ios_manifest.get("capture_start_epoch_ms", 0)
    if capture_epoch_ms:
        capture_start = datetime.utcfromtimestamp(capture_epoch_ms / 1000).isoformat() + "Z"
    else:
        capture_start = datetime.utcnow().isoformat() + "Z"

    # Build device info
    device = {
        "platform": "iOS",
        "model": ios_manifest.get("device_model", "iPhone"),
        "os_version": ios_manifest.get("os_version", "unknown"),
        "resolution": f"{ios_manifest.get('width', 1920)}x{ios_manifest.get('height', 1080)}",
        "fps": ios_manifest.get("fps_source", 30),
        "has_lidar": ios_manifest.get("has_lidar", False),
        "capture_source": source,  # "iphone" or "glasses"
    }

    # Build scale anchors from iOS hints
    scale_anchors = []
    scale_hint = ios_manifest.get("scale_hint_m_per_unit")
    if scale_hint and scale_hint != 1.0:
        scale_anchors.append({
            "anchor_type": "ios_scale_hint",
            "size_meters": scale_hint,
            "notes": "Scale hint from iOS ARKit session",
        })

    # Check for ARKit data that provides better scale
    if file_status.get("arkit/intrinsics.json"):
        scale_anchors.append({
            "anchor_type": "arkit_intrinsics",
            "size_meters": 1.0,  # ARKit provides metric scale
            "notes": "ARKit camera intrinsics available for metric reconstruction",
        })

    # Build clips list
    clips = [
        {
            "uri": video_uri,
            "fps": ios_manifest.get("fps_source", 30),
            "notes": f"Main capture from {source}",
        }
    ]

    # Build extended metadata for pipeline
    extended_metadata = {
        "ios_manifest": ios_manifest,
        "has_motion_data": file_status.get("motion.jsonl", False),
        "has_arkit_frames": file_status.get("arkit/frames.jsonl", False),
        "has_arkit_poses": file_status.get("arkit/poses.jsonl", False),
        "has_arkit_depth": any(k.startswith("arkit/depth/") for k in file_status.keys()),
        "intended_space_type": ios_manifest.get("intended_space_type", "unknown"),
        "exposure_samples_count": len(ios_manifest.get("exposure_samples", [])),
        "object_point_cloud_count": ios_manifest.get("object_point_cloud_count", 0),
    }

    # Construct session manifest
    session_manifest = {
        "session_id": scene_id,
        "capture_start": capture_start,
        "device": device,
        "scale_anchors": scale_anchors,
        "clips": clips,
        "user_notes": f"iOS capture from {device.get('model')} - {ios_manifest.get('intended_space_type', 'unknown')} space",
        # Extended fields for enhanced processing
        "raw_data_prefix": f"gs://{bucket_name}/{raw_prefix}",
        "extended_metadata": extended_metadata,
    }

    return session_manifest


def trigger_pipeline(
    session_manifest: Dict[str, Any],
    bucket_name: str,
) -> str:
    """Trigger the GPU pipeline via Cloud Run Jobs or Pub/Sub.

    Args:
        session_manifest: Converted session manifest
        bucket_name: GCS bucket name

    Returns:
        Job execution ID or Pub/Sub message ID
    """
    project_id = os.environ.get("PIPELINE_PROJECT_ID", "blueprint-8c1ca")
    region = os.environ.get("PIPELINE_REGION", "us-central1")

    # First, save the converted manifest to GCS for the pipeline to read
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        session_id = session_manifest["session_id"]
        manifest_path = f"sessions/{session_id}/session_manifest.json"
        blob = bucket.blob(manifest_path)
        blob.upload_from_string(
            json.dumps(session_manifest, indent=2),
            content_type="application/json"
        )
        logger.info(f"Saved session manifest to gs://{bucket_name}/{manifest_path}")

    except Exception as e:
        logger.error(f"Failed to save session manifest: {e}")
        raise

    # Option 1: Trigger via Pub/Sub (recommended for decoupled architecture)
    try:
        from google.cloud import pubsub_v1

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, "pipeline-trigger")

        message_data = json.dumps({
            "session_id": session_manifest["session_id"],
            "manifest_uri": f"gs://{bucket_name}/{manifest_path}",
            "bucket": bucket_name,
            "trigger_time": datetime.utcnow().isoformat() + "Z",
        }).encode("utf-8")

        future = publisher.publish(topic_path, message_data)
        message_id = future.result()

        logger.info(f"Published pipeline trigger message: {message_id}")
        return f"pubsub:{message_id}"

    except Exception as e:
        logger.warning(f"Pub/Sub trigger failed, trying Cloud Run Jobs: {e}")

    # Option 2: Directly invoke Cloud Run Job
    try:
        from google.cloud import run_v2

        client = run_v2.JobsClient()
        job_name = f"projects/{project_id}/locations/{region}/jobs/blueprint-pipeline"

        # Execute the job with the session manifest
        request = run_v2.RunJobRequest(
            name=job_name,
            overrides=run_v2.RunJobRequest.Overrides(
                container_overrides=[
                    run_v2.RunJobRequest.Overrides.ContainerOverride(
                        env=[
                            run_v2.EnvVar(
                                name="JOB_PAYLOAD",
                                value=json.dumps({
                                    "job_name": "full-pipeline",
                                    "session_id": session_manifest["session_id"],
                                    "inputs": {
                                        "manifest_uri": f"gs://{bucket_name}/{manifest_path}",
                                    },
                                    "outputs": {
                                        "base": f"gs://{bucket_name}/sessions/{session_manifest['session_id']}",
                                    },
                                    "parameters": {},
                                })
                            )
                        ]
                    )
                ]
            )
        )

        operation = client.run_job(request=request)
        execution_name = operation.metadata.name if hasattr(operation, 'metadata') else "submitted"

        logger.info(f"Triggered Cloud Run Job: {execution_name}")
        return f"cloud-run:{execution_name}"

    except Exception as e:
        logger.error(f"Failed to trigger pipeline: {e}")
        raise


def record_trigger_event(
    bucket_name: str,
    scene_id: str,
    trigger_result: str,
    session_manifest: Dict[str, Any],
):
    """Record the trigger event for auditing and debugging.

    Creates a record in GCS at triggers/{scene_id}/{timestamp}.json
    """
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        record_path = f"triggers/{scene_id}/{timestamp}.json"

        record = {
            "scene_id": scene_id,
            "trigger_time": datetime.utcnow().isoformat() + "Z",
            "trigger_result": trigger_result,
            "session_manifest_preview": {
                "session_id": session_manifest.get("session_id"),
                "capture_start": session_manifest.get("capture_start"),
                "device": session_manifest.get("device"),
                "clips_count": len(session_manifest.get("clips", [])),
            },
        }

        blob = bucket.blob(record_path)
        blob.upload_from_string(
            json.dumps(record, indent=2),
            content_type="application/json"
        )

        logger.info(f"Recorded trigger event: gs://{bucket_name}/{record_path}")

    except Exception as e:
        logger.warning(f"Failed to record trigger event: {e}")


def on_storage_finalize(event: Dict[str, Any], context: Any) -> None:
    """Cloud Function entry point - triggered when a file is uploaded to GCS.

    This function is called for every file upload. It:
    1. Parses the upload path to identify scene metadata
    2. Checks if the upload is complete (all required files present)
    3. Converts the iOS manifest to pipeline format
    4. Triggers the GPU pipeline

    Args:
        event: GCS event data containing bucket and object info
        context: Cloud Function context (not used)
    """
    bucket_name = event.get("bucket")
    object_name = event.get("name")

    logger.info(f"Storage finalize event: gs://{bucket_name}/{object_name}")

    # Parse the upload path
    path_info = parse_upload_path(object_name)
    if not path_info:
        logger.debug(f"Ignoring non-scene upload: {object_name}")
        return

    scene_id = path_info["scene_id"]
    source = path_info["source"]
    raw_prefix = path_info["raw_prefix"]
    filename = path_info["filename"]

    logger.info(f"Scene upload detected: {scene_id} from {source}")

    # Only proceed if this is a completion marker file
    if filename not in COMPLETION_MARKERS:
        logger.debug(f"Not a completion marker file: {filename}")
        return

    # Check if upload is complete
    is_complete, file_status = check_upload_completeness(bucket_name, raw_prefix)

    if not is_complete:
        logger.info(f"Upload not yet complete for scene {scene_id}")
        return

    logger.info(f"Upload complete for scene {scene_id}! Triggering pipeline...")

    # Load iOS manifest
    manifest_path = f"{raw_prefix}/manifest.json"
    ios_manifest = load_ios_manifest(bucket_name, manifest_path)

    if not ios_manifest:
        logger.error(f"Failed to load iOS manifest for scene {scene_id}")
        return

    # Convert to pipeline format
    session_manifest = convert_ios_manifest_to_session(
        ios_manifest=ios_manifest,
        scene_id=scene_id,
        source=source,
        raw_prefix=raw_prefix,
        bucket_name=bucket_name,
        file_status=file_status,
    )

    # Trigger the pipeline
    try:
        trigger_result = trigger_pipeline(session_manifest, bucket_name)
        logger.info(f"Pipeline triggered successfully: {trigger_result}")

        # Record the trigger event
        record_trigger_event(bucket_name, scene_id, trigger_result, session_manifest)

    except Exception as e:
        logger.error(f"Failed to trigger pipeline for scene {scene_id}: {e}")
        # Record failure
        record_trigger_event(
            bucket_name,
            scene_id,
            f"FAILED: {str(e)}",
            session_manifest
        )


# For local testing
if __name__ == "__main__":
    # Simulate an upload event
    test_event = {
        "bucket": "blueprint-8c1ca.appspot.com",
        "name": "scenes/test_scene_001/iphone/2024-12-09T15:30:45-abc123/raw/manifest.json",
    }
    on_storage_finalize(test_event, None)
