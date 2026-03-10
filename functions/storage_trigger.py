"""Storage trigger for qualification-first capture orchestration."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from blueprint_pipeline.capture_orchestrator import run_capture_pipeline
from blueprint_pipeline.materialization import materialize_capture_bundle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DESCRIPTOR_PATTERN = re.compile(
    r"^scenes/(?P<scene_id>[^/]+)/captures/(?P<capture_id>[^/]+)/capture_descriptor\.json$"
)
_RAW_COMPLETE_PATTERN = re.compile(
    r"^scenes/(?P<scene_id>[^/]+)/captures/(?P<capture_id>[^/]+)/raw/capture_upload_complete\.json$"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_descriptor_path(object_name: str) -> Optional[Dict[str, str]]:
    match = _DESCRIPTOR_PATTERN.match(object_name)
    if not match:
        return None
    data = match.groupdict()
    data["object_name"] = object_name
    return data


def parse_raw_upload_complete_path(object_name: str) -> Optional[Dict[str, str]]:
    match = _RAW_COMPLETE_PATTERN.match(object_name)
    if not match:
        return None
    data = match.groupdict()
    data["object_name"] = object_name
    return data


def _project_id() -> str:
    project_id = (
        os.getenv("PIPELINE_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or ""
    ).strip()
    if not project_id:
        raise RuntimeError("Missing project id (set PIPELINE_PROJECT_ID or GOOGLE_CLOUD_PROJECT)")
    return project_id


def _build_dispatch_payload(
    *,
    bucket: str,
    object_name: str,
    scene_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    descriptor_uri = f"gs://{bucket}/{object_name}"
    return {
        "schema_version": "v1",
        "descriptor_gcs_uri": descriptor_uri,
        "bucket": bucket,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "triggered_at": _utc_now_iso(),
        "source": "storage_finalize",
    }


def _dispatch_pubsub(payload: Dict[str, Any]) -> str:
    from google.cloud import pubsub_v1

    topic_name = (os.getenv("SWAP_TRIGGER_PUBSUB_TOPIC") or "").strip()
    if not topic_name:
        raise RuntimeError("SWAP_TRIGGER_PUBSUB_TOPIC is required for pubsub dispatch mode")

    publisher = pubsub_v1.PublisherClient()
    if topic_name.startswith("projects/"):
        topic_path = topic_name
    else:
        topic_path = publisher.topic_path(_project_id(), topic_name)

    data = json.dumps(payload).encode("utf-8")
    future = publisher.publish(
        topic_path,
        data,
        scene_id=str(payload.get("scene_id", "")),
        capture_id=str(payload.get("capture_id", "")),
    )
    message_id = future.result(timeout=30)
    return f"pubsub:{message_id}"


def _dispatch_cloud_tasks(payload: Dict[str, Any]) -> str:
    from google.cloud import tasks_v2

    queue_name = (os.getenv("SWAP_TRIGGER_TASK_QUEUE") or "").strip()
    location = (os.getenv("SWAP_TRIGGER_TASK_LOCATION") or "").strip()
    target_url = (os.getenv("SWAP_TRIGGER_TASK_URL") or "").strip()

    if not queue_name or not location or not target_url:
        raise RuntimeError(
            "SWAP_TRIGGER_TASK_QUEUE, SWAP_TRIGGER_TASK_LOCATION, and "
            "SWAP_TRIGGER_TASK_URL are required for cloud_tasks mode"
        )

    client = tasks_v2.CloudTasksClient()
    parent = client.queue_path(_project_id(), location, queue_name)

    body = json.dumps(payload).encode("utf-8")
    task: Dict[str, Any] = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": target_url,
            "headers": {"Content-Type": "application/json"},
            "body": body,
        }
    }

    service_account = (os.getenv("SWAP_TRIGGER_TASK_SERVICE_ACCOUNT") or "").strip()
    if service_account:
        task["http_request"]["oidc_token"] = {"service_account_email": service_account}

    response = client.create_task(parent=parent, task=task)
    return f"cloud_tasks:{response.name}"


def _dispatch_payload(payload: Dict[str, Any], *, mode: Optional[str] = None) -> str:
    dispatch_mode = (mode or os.getenv("SWAP_TRIGGER_DISPATCH_MODE") or "pubsub").strip().lower()
    if dispatch_mode == "pubsub":
        return _dispatch_pubsub(payload)
    if dispatch_mode == "cloud_tasks":
        return _dispatch_cloud_tasks(payload)
    if dispatch_mode == "direct":
        allow_direct = (os.getenv("SWAP_TRIGGER_ALLOW_DIRECT") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not allow_direct:
            raise RuntimeError(
                "SWAP_TRIGGER_DISPATCH_MODE=direct is blocked by default. "
                "Set SWAP_TRIGGER_ALLOW_DIRECT=true for local/dev only."
            )
        run_capture_pipeline(descriptor_gcs_uri=str(payload["descriptor_gcs_uri"]))
        return "direct:completed"
    raise RuntimeError(f"Unsupported SWAP_TRIGGER_DISPATCH_MODE: {dispatch_mode}")


def on_storage_finalize(event: Dict[str, Any], context: Any) -> None:  # noqa: ARG001
    bucket = str(event.get("bucket") or "")
    object_name = str(event.get("name") or "")

    if not bucket or not object_name:
        logger.warning("Storage event missing bucket/name: %s", event)
        return

    parsed = parse_descriptor_path(object_name)
    if parsed is not None:
        payload = _build_dispatch_payload(
            bucket=bucket,
            object_name=object_name,
            scene_id=parsed["scene_id"],
            capture_id=parsed["capture_id"],
        )
        logger.info(
            "Queueing capture pipeline for scene=%s capture=%s descriptor=%s",
            parsed["scene_id"],
            parsed["capture_id"],
            payload["descriptor_gcs_uri"],
        )
        dispatch_result = _dispatch_payload(payload)
        logger.info("Dispatch success: %s", dispatch_result)
        return

    raw_complete = parse_raw_upload_complete_path(object_name)
    if raw_complete is None:
        logger.debug("Ignoring non-pipeline object: gs://%s/%s", bucket, object_name)
        return

    logger.info(
        "Materializing capture descriptor from raw upload completion for scene=%s capture=%s",
        raw_complete["scene_id"],
        raw_complete["capture_id"],
    )
    materialized = materialize_capture_bundle(
        bucket=bucket,
        scene_id=raw_complete["scene_id"],
        capture_id=raw_complete["capture_id"],
        gcs_root=Path(os.getenv("GCS_ROOT", "/mnt/gcs")),
    )
    payload = _build_dispatch_payload(
        bucket=bucket,
        object_name=f"scenes/{raw_complete['scene_id']}/captures/{raw_complete['capture_id']}/capture_descriptor.json",
        scene_id=raw_complete["scene_id"],
        capture_id=raw_complete["capture_id"],
    )
    payload["descriptor_gcs_uri"] = str(materialized["descriptor_uri"])
    dispatch_result = _dispatch_payload(payload)
    logger.info("Dispatch success after materialization: %s", dispatch_result)


def on_swap_dispatch(event: Dict[str, Any], context: Any) -> None:  # noqa: ARG001
    """Pub/Sub-triggered worker entrypoint that runs orchestration."""

    data_b64 = event.get("data")
    if not data_b64:
        raise RuntimeError("Pub/Sub event missing data")

    raw = base64.b64decode(data_b64)
    payload = json.loads(raw.decode("utf-8"))
    descriptor_uri = str(payload.get("descriptor_gcs_uri") or "").strip()
    if not descriptor_uri:
        raise RuntimeError("Dispatch payload missing descriptor_gcs_uri")

    logger.info("Running capture pipeline from dispatch message: %s", descriptor_uri)
    run_capture_pipeline(descriptor_gcs_uri=descriptor_uri)


def on_swap_dispatch_http(request: Any):  # type: ignore[no-untyped-def]
    """HTTP worker entrypoint for Cloud Tasks targets."""

    payload = request.get_json(silent=True) if hasattr(request, "get_json") else None
    if not isinstance(payload, dict):
        return ("Invalid payload", 400)

    descriptor_uri = str(payload.get("descriptor_gcs_uri") or "").strip()
    if not descriptor_uri:
        return ("Missing descriptor_gcs_uri", 400)

    logger.info("Running capture pipeline from HTTP dispatch: %s", descriptor_uri)
    run_capture_pipeline(descriptor_gcs_uri=descriptor_uri)
    return ("ok", 200)
