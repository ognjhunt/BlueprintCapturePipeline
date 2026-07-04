"""Pull BlueprintCapture bridge handoffs from Pub/Sub and run the pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from google.cloud import storage

from .common import PipelineError
from .run_e2e import run_end_to_end

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandoffMessage:
    bucket: str
    scene_id: str
    capture_id: str
    raw_prefix_uri: str
    pipeline_handoff_uri: str | None

    @property
    def capture_prefix(self) -> str:
        return f"scenes/{self.scene_id}/captures/{self.capture_id}"


def parse_handoff_payload(payload: bytes | str | Mapping[str, Any]) -> HandoffMessage:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Pub/Sub handoff payload is not valid JSON: {exc}") from exc
    else:
        data = dict(payload)

    bucket = _required_string(data, "bucket")
    scene_id = _required_string(data, "scene_id")
    capture_id = _required_string(data, "capture_id")
    raw_prefix_uri = _required_string(data, "raw_prefix_uri")
    if raw_prefix_uri != f"gs://{bucket}/scenes/{scene_id}/captures/{capture_id}/raw":
        raise PipelineError(
            "Pub/Sub handoff raw_prefix_uri does not match bucket/scene/capture identity: "
            f"{raw_prefix_uri}"
        )

    pipeline_handoff_uri = data.get("pipeline_handoff_uri")
    if pipeline_handoff_uri is not None and not isinstance(pipeline_handoff_uri, str):
        raise PipelineError("Pub/Sub handoff pipeline_handoff_uri must be a string when present.")

    return HandoffMessage(
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
        raw_prefix_uri=raw_prefix_uri,
        pipeline_handoff_uri=pipeline_handoff_uri,
    )


def stage_handoff_capture(
    handoff: HandoffMessage,
    *,
    storage_root: Path,
    storage_client: storage.Client | None = None,
) -> Path:
    client = storage_client or storage.Client()
    capture_root = storage_root / handoff.bucket / handoff.capture_prefix
    capture_root.mkdir(parents=True, exist_ok=True)

    blobs = list(client.list_blobs(handoff.bucket, prefix=f"{handoff.capture_prefix}/"))
    if not blobs:
        raise PipelineError(f"No objects found for handoff prefix: {handoff.capture_prefix}/")

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        destination = storage_root / handoff.bucket / blob.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))

    if not (capture_root / "raw" / "capture_upload_complete.json").is_file():
        raise PipelineError(
            "Staged handoff capture is missing raw/capture_upload_complete.json; "
            f"capture_root={capture_root}"
        )
    if not (capture_root / "pipeline_handoff.json").is_file():
        # Real iOS bundles never upload pipeline_handoff.json (XR-03); synthesize it from the
        # provenance already carried by raw/manifest.json + raw/capture_context.json so the
        # capture_job_id / site_submission_id / buyer_request_id data contract stays intact.
        _synthesize_pipeline_handoff(handoff, capture_root=capture_root)
    return capture_root


def _read_optional_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _first_non_empty(*sources: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _synthesize_pipeline_handoff(handoff: HandoffMessage, *, capture_root: Path) -> Path:
    """Materialize pipeline_handoff.json from raw sidecars when the iOS bundle omits it.

    We never invent provenance: values come only from raw/manifest.json and
    raw/capture_context.json, both of which the iOS app already writes.
    """

    raw_root = capture_root / "raw"
    manifest = _read_optional_json_object(raw_root / "manifest.json")
    context = _read_optional_json_object(raw_root / "capture_context.json")

    site_submission_id = _first_non_empty(
        manifest, context, keys=("site_submission_id", "siteSubmissionId")
    )
    buyer_request_id = _first_non_empty(
        manifest, context, keys=("buyer_request_id", "buyerRequestId")
    )
    capture_job_id = _first_non_empty(manifest, context, keys=("capture_job_id", "captureJobId"))
    request_id = buyer_request_id or capture_job_id

    requested_outputs: list[str] = []
    for source in (manifest, context):
        for key in ("requested_outputs", "requestedOutputs", "requested_lanes", "requestedLanes"):
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    text = str(item).strip()
                    if text and text not in requested_outputs:
                        requested_outputs.append(text)

    payload: dict[str, Any] = {
        "schema_version": "pipeline_handoff.v1",
        "synthesized": True,
        "synthesized_from": ["raw/manifest.json", "raw/capture_context.json"],
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "bucket": handoff.bucket,
        "raw_prefix_uri": handoff.raw_prefix_uri,
        "site_submission_id": site_submission_id,
        "buyer_request_id": buyer_request_id,
        "capture_job_id": capture_job_id,
        "owner_system": {
            "owner_system": "blueprint_capture",
            "request_id": request_id,
            "site_submission_id": site_submission_id,
            "buyer_request_id": buyer_request_id,
            "capture_job_id": capture_job_id,
        },
    }
    if requested_outputs:
        payload["requested_outputs"] = requested_outputs

    destination = capture_root / "pipeline_handoff.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "pubsub_handoff.synthesized_pipeline_handoff",
        extra={
            "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id,
            "capture_job_id": capture_job_id,
        },
    )
    return destination


def process_handoff_payload(
    payload: bytes | str | Mapping[str, Any],
    *,
    storage_root: Path,
    provider: str,
    run_e2e: Callable[..., dict[str, Any]] = run_end_to_end,
    storage_client: storage.Client | None = None,
    run_evaluation_prep: bool = True,
) -> dict[str, Any]:
    handoff = parse_handoff_payload(payload)
    capture_root = stage_handoff_capture(
        handoff,
        storage_root=storage_root,
        storage_client=storage_client,
    )
    result = run_e2e(
        capture_root=str(capture_root),
        provider=provider,
        run_evaluation_prep=run_evaluation_prep,
    )
    return {
        "schema_version": "v1",
        "status": "processed",
        "bucket": handoff.bucket,
        "scene_id": handoff.scene_id,
        "capture_id": handoff.capture_id,
        "capture_root": str(capture_root),
        "run_e2e": result,
    }


def pull_and_process(
    *,
    subscription: str,
    storage_root: Path,
    provider: str,
    max_messages: int,
    run_evaluation_prep: bool = True,
) -> int:
    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    response = subscriber.pull(
        request={
            "subscription": subscription,
            "max_messages": max_messages,
        },
        timeout=30,
    )
    ack_ids: list[str] = []
    for received in response.received_messages:
        message = received.message
        logger.info(
            "pubsub_handoff.received",
            extra={
                "message_id": message.message_id,
                "attributes": dict(message.attributes),
            },
        )
        process_handoff_payload(
            message.data,
            storage_root=storage_root,
            provider=provider,
            run_evaluation_prep=run_evaluation_prep,
        )
        ack_ids.append(received.ack_id)

    if ack_ids:
        subscriber.acknowledge(request={"subscription": subscription, "ack_ids": ack_ids})
    return len(ack_ids)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"Pub/Sub handoff missing required string: {key}")
    return value.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull BlueprintCapture Pub/Sub handoffs and run blueprint_pipeline.run_e2e."
    )
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--provider", default="openai", choices=("claude", "openai"))
    parser.add_argument("--max-messages", type=int, default=1)
    parser.add_argument("--skip-evaluation-prep", action="store_true")
    args = parser.parse_args(argv)

    storage_root = args.storage_root or Path(tempfile.gettempdir()) / "blueprint-pubsub-handoffs"
    processed = pull_and_process(
        subscription=args.subscription,
        storage_root=storage_root,
        provider=args.provider,
        max_messages=max(1, args.max_messages),
        run_evaluation_prep=not args.skip_evaluation_prep,
    )
    print(json.dumps({"processed": processed, "storage_root": str(storage_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
