#!/usr/bin/env python3
"""Publish, pull, acknowledge, and clean up against a local Pub/Sub emulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


SOURCE_NAME = "pubsub_emulator_round_trip_source.json"
SOURCE_SCHEMA = "blueprint.pubsub_emulator_round_trip_source.v1"
MAX_SOURCE_SIZE = 64 * 1024


def _write_json(path: Path, payload: dict[str, Any]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_SOURCE_SIZE:
        raise ValueError("pubsub_emulator_source_oversize")
    path.write_bytes(encoded)
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", len(encoded)


def _repository_sha(root: Path) -> str | None:
    configured = str(os.environ.get("GITHUB_SHA") or "").strip().lower()
    if configured:
        return configured
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip().lower() if completed.returncode == 0 else None


def _validated_emulator_host(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("PUBSUB_EMULATOR_HOST is required")
    parsed = urlsplit(text if "://" in text else f"http://{text}")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("PUBSUB_EMULATOR_HOST scheme must be http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("PUBSUB_EMULATOR_HOST may not include credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("PUBSUB_EMULATOR_HOST may not include a path")
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("PUBSUB_EMULATOR_HOST must resolve to the local emulator only")
    if parsed.port is None or not (1 <= parsed.port <= 65535):
        raise ValueError("PUBSUB_EMULATOR_HOST must include a valid port")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port)}
    except OSError as exc:
        raise ValueError("PUBSUB_EMULATOR_HOST could not be resolved") from exc
    if not addresses or not addresses <= {"127.0.0.1", "::1"}:
        raise ValueError("PUBSUB_EMULATOR_HOST resolved outside loopback")
    return text.removeprefix("http://").removeprefix("https://")


def run_integration(
    *,
    root: Path,
    source_path: Path | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    message_id: str | None = None
    received_payload: dict[str, object] | None = None
    acknowledged = False
    cleanup_succeeded = False
    expected_payload: dict[str, object] | None = None
    configured = str(os.environ.get("PUBSUB_EMULATOR_HOST") or "")
    try:
        emulator_host = _validated_emulator_host(configured)
    except ValueError as exc:
        emulator_host = None
        blockers.append(f"pubsub_emulator_host_invalid:{exc}")
    if emulator_host is not None:
        os.environ["PUBSUB_EMULATOR_HOST"] = emulator_host
        try:
            from google.auth.credentials import AnonymousCredentials
            from google.cloud import pubsub_v1
        except Exception as exc:  # noqa: BLE001 - missing integration runtime blocks
            blockers.append(f"pubsub_emulator_dependency_missing:{type(exc).__name__}")
        else:
            project = f"blueprint-ci-{uuid.uuid4().hex[:12]}"
            topic_id = f"handoff-{uuid.uuid4().hex[:12]}"
            subscription_id = f"worker-{uuid.uuid4().hex[:12]}"
            credentials = AnonymousCredentials()
            publisher = pubsub_v1.PublisherClient(credentials=credentials)
            subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
            topic_path = publisher.topic_path(project, topic_id)
            subscription_path = subscriber.subscription_path(project, subscription_id)
            try:
                publisher.create_topic(request={"name": topic_path})
                subscriber.create_subscription(
                    request={"name": subscription_path, "topic": topic_path}
                )
                expected: dict[str, object] = {
                    "probe_id": uuid.uuid4().hex,
                    "kind": "pipeline_handoff",
                }
                expected_payload = expected
                message_id = publisher.publish(
                    topic_path,
                    json.dumps(expected, sort_keys=True).encode("utf-8"),
                ).result(timeout=timeout_seconds)
                if not str(message_id or "").strip():
                    blockers.append("pubsub_emulator_publish_message_id_missing")
                deadline = time.monotonic() + timeout_seconds
                while time.monotonic() < deadline and received_payload is None:
                    response = subscriber.pull(
                        request={"subscription": subscription_path, "max_messages": 1},
                        timeout=min(2.0, max(0.1, deadline - time.monotonic())),
                    )
                    for received in response.received_messages:
                        received_payload = json.loads(received.message.data.decode("utf-8"))
                        subscriber.acknowledge(
                            request={
                                "subscription": subscription_path,
                                "ack_ids": [received.ack_id],
                            }
                        )
                        acknowledged = True
                if received_payload != expected:
                    blockers.append("pubsub_emulator_round_trip_payload_mismatch")
                if not acknowledged:
                    blockers.append("pubsub_emulator_message_not_acknowledged")
            except Exception as exc:  # noqa: BLE001 - any emulator failure blocks the lane
                blockers.append(f"pubsub_emulator_round_trip_failed:{type(exc).__name__}")
            finally:
                cleanup_errors: list[str] = []
                try:
                    subscriber.delete_subscription(request={"subscription": subscription_path})
                except Exception as exc:  # noqa: BLE001 - cleanup is part of the lane contract
                    cleanup_errors.append(f"subscription_{type(exc).__name__}")
                try:
                    publisher.delete_topic(request={"topic": topic_path})
                except Exception as exc:  # noqa: BLE001 - cleanup is part of the lane contract
                    cleanup_errors.append(f"topic_{type(exc).__name__}")
                for label, transport in (
                    ("publisher", publisher.transport),
                    ("subscriber", subscriber.transport),
                ):
                    try:
                        transport.close()
                    except Exception as exc:  # noqa: BLE001 - resource cleanup must be explicit
                        cleanup_errors.append(f"{label}_transport_{type(exc).__name__}")
                if cleanup_errors:
                    blockers.extend(
                        f"pubsub_emulator_cleanup_failed:{item}" for item in cleanup_errors
                    )
                else:
                    cleanup_succeeded = True
    repository_sha = _repository_sha(root)
    generated_at = datetime.now(timezone.utc).isoformat()
    canonical_payload = (
        json.dumps(received_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if received_payload is not None
        else b""
    )
    payload_digest = (
        f"sha256:{hashlib.sha256(canonical_payload).hexdigest()}" if canonical_payload else None
    )
    source_digest: str | None = None
    source_size: int | None = None
    if source_path is None:
        blockers.append("pubsub_emulator_source_path_required")
    else:
        source = {
            "schema_version": SOURCE_SCHEMA,
            "generated_at": generated_at,
            "repository_sha": repository_sha,
            "emulator_loopback_only": emulator_host is not None,
            "expected_payload": expected_payload,
            "received_payload": received_payload,
            "published_message_id": message_id,
            "message_acknowledged": acknowledged,
            "cleanup_succeeded": cleanup_succeeded,
            "claim_boundary": {
                "local_emulator_transcript_only": True,
                "not_deployed_pubsub_proof": True,
            },
        }
        try:
            source_digest, source_size = _write_json(source_path, source)
        except (OSError, ValueError) as exc:
            blockers.append(f"pubsub_emulator_source_write_failed:{type(exc).__name__}")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "blueprint.critical_capability_lane_evidence.v1",
        "lane_id": "pubsub_emulator_integration",
        "evidence_schema_version": "blueprint.pubsub_emulator_integration.v1",
        "generated_at": generated_at,
        "repository_sha": repository_sha,
        "status": "passed" if not blockers else "blocked",
        "executed": emulator_host is not None,
        "skipped_count": 0,
        "emulator_loopback_only": emulator_host is not None,
        "published_message_id": message_id,
        "round_trip_payload_received": received_payload is not None,
        "message_acknowledged": acknowledged,
        "cleanup_succeeded": cleanup_succeeded,
        "round_trip_payload_sha256": payload_digest,
        "artifact_digests": {SOURCE_NAME: source_digest} if source_digest else {},
        "artifact_sizes": {SOURCE_NAME: source_size} if source_size is not None else {},
        "blockers": blockers,
        "claim_boundary": {
            "emulator_proof_is_not_deployed_pubsub_proof": True,
            "no_live_google_cloud_project_was_contacted": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    output = args.output.expanduser().absolute()
    result = run_integration(
        root=args.root.resolve(),
        source_path=output.parent / SOURCE_NAME,
        timeout_seconds=args.timeout_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[pubsub-emulator] status={result['status']} output={output}")
    for blocker in result["blockers"]:
        print(f"[pubsub-emulator] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
