"""Storage trigger for capture-first upload orchestration into packaging and review lanes."""

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
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from blueprint_pipeline.capture_orchestrator import run_capture_pipeline  # noqa: E402
from blueprint_pipeline.materialization import (  # noqa: E402
    capture_materialization_readiness,
    materialize_capture_bundle,
)


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


def _dispatch_mode(raw: Optional[str] = None) -> str:
    return (raw or os.getenv("SWAP_TRIGGER_DISPATCH_MODE") or "pubsub").strip().lower()


def _pipeline_execution_mode() -> str:
    return (os.getenv("PIPELINE_EXECUTION_MODE") or "inline").strip().lower()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _capture_bridge_handoff_primary() -> bool:
    return _bool_env("SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF", default=False)


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
    dispatch_mode = _dispatch_mode(mode)
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
        descriptor_uri = str(
            payload.get("descriptor_gcs_uri") or payload.get("capture_descriptor_uri") or ""
        ).strip()
        if not descriptor_uri:
            raise RuntimeError("Dispatch payload missing descriptor_gcs_uri")
        run_capture_pipeline(**_pipeline_kwargs_from_payload(payload, descriptor_uri=descriptor_uri))
        return "direct:completed"
    raise RuntimeError(f"Unsupported SWAP_TRIGGER_DISPATCH_MODE: {dispatch_mode}")


def _payload_requested_lanes(payload: Dict[str, Any]) -> Optional[list[str]]:
    raw = payload.get("requested_lanes") or payload.get("requestedLanes")
    if raw is None:
        return None
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(value) for value in raw]
    else:
        values = [str(raw)]
    requested_lanes = [value.strip() for value in values if value.strip()]
    return requested_lanes or None


def _pipeline_kwargs_from_payload(payload: Dict[str, Any], *, descriptor_uri: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"descriptor_gcs_uri": descriptor_uri}
    requested_lanes = _payload_requested_lanes(payload)
    if requested_lanes is not None:
        kwargs["requested_lanes"] = requested_lanes
    return kwargs


def _descriptor_uri_for_capture(*, bucket: str, scene_id: str, capture_id: str) -> str:
    return f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/capture_descriptor.json"


def _pipeline_job_target() -> tuple[str, str]:
    job_name = (os.getenv("PIPELINE_RUN_JOB_NAME") or "blueprint-pipeline").strip()
    region = (os.getenv("PIPELINE_RUN_JOB_REGION") or os.getenv("PIPELINE_REGION") or "").strip()
    if not job_name:
        raise RuntimeError("PIPELINE_RUN_JOB_NAME is required for cloud_run_job execution mode")
    if not region:
        raise RuntimeError("PIPELINE_RUN_JOB_REGION or PIPELINE_REGION is required for cloud_run_job execution mode")
    return job_name, region


def _launch_cloud_run_job(payload: Dict[str, Any]) -> str:
    from google.auth import default as google_auth_default
    from google.auth.transport.requests import Request as GoogleAuthRequest

    bucket = str(payload.get("bucket") or "").strip()
    scene_id = str(payload.get("scene_id") or "").strip()
    capture_id = str(payload.get("capture_id") or "").strip()
    # Accept capture_descriptor_uri (extractFrames handoff format) as alias for descriptor_gcs_uri
    descriptor_uri = str(
        payload.get("descriptor_gcs_uri") or payload.get("capture_descriptor_uri") or ""
    ).strip()
    # Derive bucket from a gs:// URI if not explicitly provided (extractFrames handoff omits bucket)
    if not bucket:
        for _uri_field in ("descriptor_gcs_uri", "capture_descriptor_uri", "raw_prefix_uri"):
            _uri = str(payload.get(_uri_field) or "").strip()
            if _uri.startswith("gs://"):
                bucket = _uri[5:].split("/", 1)[0]
                break
    if not bucket or not scene_id or not capture_id:
        raise RuntimeError("Dispatch payload missing bucket/scene_id/capture_id")

    job_name, region = _pipeline_job_target()
    credentials, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(GoogleAuthRequest())

    url = (
        f"https://run.googleapis.com/v2/projects/{_project_id()}/locations/{region}/jobs/{job_name}:run"
    )
    env_overrides = [
        {"name": "PIPELINE_BUCKET", "value": bucket},
        {"name": "PIPELINE_SCENE_ID", "value": scene_id},
        {"name": "PIPELINE_CAPTURE_ID", "value": capture_id},
    ]
    if descriptor_uri:
        env_overrides.append({"name": "PIPELINE_DESCRIPTOR_GCS_URI", "value": descriptor_uri})
    request = urllib_request.Request(
        url,
        data=json.dumps(
            {
                "overrides": {
                    "containerOverrides": [
                        {
                            "env": env_overrides,
                        }
                    ]
                }
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw) if raw else {}
    operation_name = parsed.get("name") if isinstance(parsed, dict) else None
    return f"cloud_run_job:{operation_name or job_name}"


def _run_pipeline_inline(payload: Dict[str, Any]) -> str:
    descriptor_uri = str(
        payload.get("descriptor_gcs_uri") or payload.get("capture_descriptor_uri") or ""
    ).strip()
    bucket = str(payload.get("bucket") or "").strip()
    scene_id = str(payload.get("scene_id") or "").strip()
    capture_id = str(payload.get("capture_id") or "").strip()
    if not descriptor_uri and bucket and scene_id and capture_id:
        readiness = capture_materialization_readiness(
            bucket=bucket,
            scene_id=scene_id,
            capture_id=capture_id,
            gcs_root=Path(os.getenv("GCS_ROOT", "/mnt/gcs")),
        )
        if not readiness["ready"]:
            raise RuntimeError(
                "Capture bundle not ready for inline execution: "
                + ",".join(str(item) for item in readiness["issues"])
            )
        materialized = materialize_capture_bundle(
            bucket=bucket,
            scene_id=scene_id,
            capture_id=capture_id,
            gcs_root=Path(os.getenv("GCS_ROOT", "/mnt/gcs")),
        )
        descriptor_uri = str(materialized["descriptor_uri"])
    if not descriptor_uri:
        raise RuntimeError("Dispatch payload missing descriptor_gcs_uri")
    run_capture_pipeline(**_pipeline_kwargs_from_payload(payload, descriptor_uri=descriptor_uri))
    return "inline:completed"


def _execute_pipeline_payload(payload: Dict[str, Any]) -> str:
    execution_mode = _pipeline_execution_mode()
    if execution_mode == "cloud_run_job":
        return _launch_cloud_run_job(payload)
    if execution_mode == "inline":
        return _run_pipeline_inline(payload)
    raise RuntimeError(f"Unsupported PIPELINE_EXECUTION_MODE: {execution_mode}")


def on_storage_finalize(event: Dict[str, Any], context: Any) -> None:  # noqa: ARG001
    bucket = str(event.get("bucket") or "")
    object_name = str(event.get("name") or "")

    if not bucket or not object_name:
        logger.warning("Storage event missing bucket/name: %s", event)
        return

    parsed = parse_descriptor_path(object_name)
    if parsed is not None:
        if _capture_bridge_handoff_primary():
            logger.info(
                "Ignoring descriptor finalize for scene=%s capture=%s because capture bridge handoff is primary",
                parsed["scene_id"],
                parsed["capture_id"],
            )
            return
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
        "Queueing capture pipeline from raw upload completion for scene=%s capture=%s",
        raw_complete["scene_id"],
        raw_complete["capture_id"],
    )
    if _capture_bridge_handoff_primary() and _dispatch_mode() != "direct":
        logger.info(
            "Ignoring raw upload completion for scene=%s capture=%s because capture bridge handoff is primary",
            raw_complete["scene_id"],
            raw_complete["capture_id"],
        )
        return
    readiness = capture_materialization_readiness(
        bucket=bucket,
        scene_id=raw_complete["scene_id"],
        capture_id=raw_complete["capture_id"],
        gcs_root=Path(os.getenv("GCS_ROOT", "/mnt/gcs")),
    )
    if not readiness["ready"]:
        raise RuntimeError(
            "Capture bundle not ready for dispatch: "
            + ",".join(str(item) for item in readiness["issues"])
        )
    payload = _build_dispatch_payload(
        bucket=bucket,
        object_name=f"scenes/{raw_complete['scene_id']}/captures/{raw_complete['capture_id']}/capture_descriptor.json",
        scene_id=raw_complete["scene_id"],
        capture_id=raw_complete["capture_id"],
    )
    payload["descriptor_gcs_uri"] = _descriptor_uri_for_capture(
        bucket=bucket,
        scene_id=raw_complete["scene_id"],
        capture_id=raw_complete["capture_id"],
    )
    if _dispatch_mode() == "direct":
        materialized = materialize_capture_bundle(
            bucket=bucket,
            scene_id=raw_complete["scene_id"],
            capture_id=raw_complete["capture_id"],
            gcs_root=Path(os.getenv("GCS_ROOT", "/mnt/gcs")),
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
    descriptor_uri = str(
        payload.get("descriptor_gcs_uri") or payload.get("capture_descriptor_uri") or ""
    ).strip()
    logger.info("Dispatch worker executing payload for descriptor=%s", descriptor_uri or "<materialize-first>")
    execution_result = _execute_pipeline_payload(payload)
    logger.info("Dispatch worker result: %s", execution_result)


def on_swap_dispatch_http(request: Any):  # type: ignore[no-untyped-def]
    """HTTP worker entrypoint for Cloud Tasks targets."""

    payload = request.get_json(silent=True) if hasattr(request, "get_json") else None
    if not isinstance(payload, dict):
        return ("Invalid payload", 400)

    descriptor_uri = str(
        payload.get("descriptor_gcs_uri") or payload.get("capture_descriptor_uri") or ""
    ).strip()
    if not descriptor_uri and not (
        str(payload.get("bucket") or "").strip()
        and str(payload.get("scene_id") or "").strip()
        and str(payload.get("capture_id") or "").strip()
    ):
        return ("Missing descriptor or capture identity", 400)

    execution_result = _execute_pipeline_payload(payload)
    logger.info("HTTP dispatch worker result: %s", execution_result)
    return ("ok", 200)
