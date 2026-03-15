"""Thin provider abstraction for optional third-party preview generation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from .common import utc_now_iso, write_json


class PreviewProvider(Protocol):
    def submit(self, *, descriptor: Mapping[str, Any], capture_root: Path) -> Dict[str, Any]: ...

    def poll(self, *, run_id: str) -> Dict[str, Any]: ...

    def normalize(self, payload: Mapping[str, Any]) -> Dict[str, Any]: ...

    def emit_preview_manifest(self, *, normalized: Mapping[str, Any], output_path: Path) -> Dict[str, Any]: ...

    def emit_provenance(self, *, descriptor: Mapping[str, Any], normalized: Mapping[str, Any]) -> Dict[str, Any]: ...


@dataclass
class StubPreviewProvider:
    provider_name: str = "stub_preview"
    provider_model: str = "stub-v1"

    def submit(self, *, descriptor: Mapping[str, Any], capture_root: Path) -> Dict[str, Any]:
        run_id = f"stub-{descriptor.get('capture_id')}"
        return {
            "provider_name": self.provider_name,
            "provider_model": self.provider_model,
            "provider_run_id": run_id,
            "status": "succeeded",
            "artifact_uris": {
                "preview_still_uri": f"file://{capture_root / 'pipeline' / 'preview_simulation' / 'preview-still.png'}",
            },
            "cost_usd": 0.0,
            "latency_ms": 0,
        }

    def poll(self, *, run_id: str) -> Dict[str, Any]:
        return {"provider_run_id": run_id, "status": "succeeded"}

    def normalize(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(payload)

    def emit_preview_manifest(self, *, normalized: Mapping[str, Any], output_path: Path) -> Dict[str, Any]:
        manifest = {
            "schema_version": "v1",
            "provider_name": normalized.get("provider_name"),
            "provider_model": normalized.get("provider_model"),
            "provider_run_id": normalized.get("provider_run_id"),
            "status": normalized.get("status"),
            "artifact_uris": dict(normalized.get("artifact_uris") or {}),
            "generated_at": utc_now_iso(),
        }
        write_json(output_path, manifest)
        return manifest

    def emit_provenance(self, *, descriptor: Mapping[str, Any], normalized: Mapping[str, Any]) -> Dict[str, Any]:
        digest = sha256(json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "canonical": False,
            "derived": True,
            "input_manifest_hash": digest,
            "provider_name": normalized.get("provider_name"),
            "provider_model": normalized.get("provider_model"),
            "provider_run_id": normalized.get("provider_run_id"),
        }


@dataclass
class WorldLabsPreviewProvider(StubPreviewProvider):
    provider_name: str = "world_labs"
    provider_model: str = "world-api"

    def submit(self, *, descriptor: Mapping[str, Any], capture_root: Path) -> Dict[str, Any]:
        api_key = str(os.getenv("WORLDLABS_API_KEY") or "").strip()
        endpoint = str(os.getenv("WORLDLABS_API_URL") or "").strip()
        if not api_key or not endpoint:
            raise RuntimeError("WORLDLABS_API_KEY and WORLDLABS_API_URL are required")

        started_at = time.time()
        payload = {
            "scene_id": descriptor.get("scene_id"),
            "capture_id": descriptor.get("capture_id"),
            "video_uri": descriptor.get("raw_video_uri"),
            "arkit_poses_uri": descriptor.get("arkit_poses_uri"),
        }
        request = urllib_request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib_error.HTTPError, urllib_error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"world_labs_submit_failed:{exc}") from exc
        latency_ms = int((time.time() - started_at) * 1000)
        return {
            "provider_name": self.provider_name,
            "provider_model": self.provider_model,
            "provider_run_id": str(body.get("id") or body.get("run_id") or ""),
            "status": str(body.get("status") or "submitted"),
            "artifact_uris": body.get("artifact_uris") or {},
            "cost_usd": body.get("cost_usd"),
            "latency_ms": latency_ms,
            "input_manifest_uri": descriptor.get("raw_prefix_uri"),
            "raw_response": body,
        }


def resolve_preview_provider(name: str) -> PreviewProvider:
    normalized = str(name or "stub").strip().lower()
    if normalized in {"world_labs", "worldlabs"}:
        return WorldLabsPreviewProvider()
    return StubPreviewProvider()


def run_preview_provider(
    *,
    provider_name: str,
    descriptor: Mapping[str, Any],
    capture_root: Path,
    pipeline_dir: Path,
) -> Dict[str, Any]:
    provider = resolve_preview_provider(provider_name)
    manifest_path = pipeline_dir / "preview_manifest.json"
    try:
      submitted = provider.submit(descriptor=descriptor, capture_root=capture_root)
      normalized = provider.normalize(submitted)
      manifest = provider.emit_preview_manifest(normalized=normalized, output_path=manifest_path)
      provenance = provider.emit_provenance(descriptor=descriptor, normalized=normalized)
      run_manifest = {
          "schema_version": "v1",
          "provider_name": normalized.get("provider_name"),
          "provider_model": normalized.get("provider_model"),
          "provider_run_id": normalized.get("provider_run_id"),
          "status": normalized.get("status"),
          "input_manifest_uri": normalized.get("input_manifest_uri"),
          "preview_manifest_uri": str(manifest_path),
          "artifact_uris": normalized.get("artifact_uris") or {},
          "cost_usd": normalized.get("cost_usd"),
          "latency_ms": normalized.get("latency_ms"),
          "failure_reason": None,
          "provenance": provenance,
      }
    except Exception as exc:
      run_manifest = {
          "schema_version": "v1",
          "provider_name": getattr(provider, "provider_name", provider_name),
          "provider_model": getattr(provider, "provider_model", "unknown"),
          "provider_run_id": "",
          "status": "failed",
          "input_manifest_uri": descriptor.get("raw_prefix_uri"),
          "preview_manifest_uri": str(manifest_path),
          "artifact_uris": {},
          "cost_usd": None,
          "latency_ms": None,
          "failure_reason": str(exc),
          "provenance": {"canonical": False, "derived": True},
      }
      write_json(manifest_path, {
          "schema_version": "v1",
          "status": "failed",
          "provider_name": run_manifest["provider_name"],
          "failure_reason": str(exc),
          "generated_at": utc_now_iso(),
      })

    write_json(pipeline_dir / "provider_run_manifest.json", run_manifest)
    return run_manifest
