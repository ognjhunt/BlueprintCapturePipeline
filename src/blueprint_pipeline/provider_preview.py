"""Thin provider abstraction for optional third-party preview generation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol
from .common import utc_now_iso, write_json

_DEFAULT_WORLDLABS_TEXT_PROMPT = """Create a grounded, explorable Marble world from this walkthrough video of a real indoor media-room / office-like environment.

Requirements:
- Preserve the real room layout, scale, walkable floor area, and major object placement from the video.
- Treat this as a practical, cluttered working room, not a stylized showroom.
- Keep desks, office chairs, printers, shelving, boxes, monitors, TVs, fireplace, doorways, walls, and window-blind surfaces where they appear in the walkthrough.
- Respect the captured camera path and inferred spatial relationships from the video.
- Favor physical plausibility over visual embellishment.
- Do not invent extra rooms, extra corridors, or dramatic architectural changes.
- Do not clean up clutter unless the video clearly shows open space.
- Avoid fantasy, cinematic, game-like, or exaggerated design choices.
- Maintain a neutral, realistic material palette and ordinary indoor lighting.
- Output should feel like a believable reconstruction of the same site for interactive review and navigation.
- Prioritize navigability, stable geometry, and faithful scene structure over decorative detail.
- If any region is ambiguous, infer the simplest continuation consistent with the walkthrough instead of hallucinating new features."""


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
    provider_model: str = "Marble 0.1-mini"

    @staticmethod
    def _string(value: Any) -> str:
        return str(value or "").strip()

    def _privacy_processing(self, descriptor: Mapping[str, Any]) -> Mapping[str, Any]:
        metadata = descriptor.get("metadata")
        if isinstance(metadata, Mapping):
            payload = metadata.get("privacy_processing")
            if isinstance(payload, Mapping):
                return payload
        return {}

    def _world_prompt_candidates(self, descriptor: Mapping[str, Any]) -> Dict[str, Any]:
        privacy_processing = self._privacy_processing(descriptor)
        metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
        worldlabs_input_video_uri = self._string(metadata.get("worldlabs_input_video_uri"))
        world_model_video_uri = self._string(descriptor.get("world_model_video_uri"))
        privacy_processed_video_uri = self._string(descriptor.get("privacy_processed_video_uri"))
        raw_video_uri = self._string(descriptor.get("raw_video_uri"))
        privacy_status = self._string(
            descriptor.get("privacy_status") or privacy_processing.get("status")
        ).lower()
        raw_retained = bool(privacy_processing.get("raw_retained"))
        raw_allowed = raw_retained and privacy_status == "no_people_detected"

        candidates: List[Dict[str, Any]] = []
        if worldlabs_input_video_uri:
            candidates.append(
                {
                    "source_id": "worldlabs_input_video_uri",
                    "uri": worldlabs_input_video_uri,
                    "eligible": True,
                    "reason": "dedicated World Labs compliant input video",
                }
            )
        if world_model_video_uri:
            candidates.append(
                {
                    "source_id": "world_model_video_uri",
                    "uri": world_model_video_uri,
                    "eligible": True,
                    "reason": "preferred privacy-safe world-model video",
                }
            )
        if privacy_processed_video_uri:
            candidates.append(
                {
                    "source_id": "privacy_processed_video_uri",
                    "uri": privacy_processed_video_uri,
                    "eligible": True,
                    "reason": "privacy-processed walkthrough video",
                }
            )
        if raw_video_uri:
            candidates.append(
                {
                    "source_id": "raw_video_uri",
                    "uri": raw_video_uri,
                    "eligible": raw_allowed,
                    "reason": (
                        "raw walkthrough explicitly allowed by privacy policy"
                        if raw_allowed
                        else "raw walkthrough retained but not approved for direct World Labs generation"
                    ),
                }
            )

        selected = next((item for item in candidates if item.get("eligible")), None)
        return {
            "privacy_status": privacy_status or None,
            "raw_allowed": raw_allowed,
            "selected": selected,
            "candidates": candidates,
        }

    def _build_request_manifest(
        self,
        *,
        descriptor: Mapping[str, Any],
        capture_root: Path,
    ) -> Dict[str, Any]:
        del capture_root
        video_candidates = self._world_prompt_candidates(descriptor)
        metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
        scene_summary = self._string(
            metadata.get("scene_summary")
            or metadata.get("site_summary")
            or metadata.get("environment_summary")
            or metadata.get("operator_notes")
        )
        site_name = self._string(metadata.get("site_name"))
        industry = self._string(metadata.get("industry"))
        task_lane = self._string(metadata.get("task_lane") or metadata.get("task_statement"))
        display_name = self._string(
            metadata.get("display_name")
            or site_name
            or descriptor.get("capture_id")
            or descriptor.get("scene_id")
        )
        prompt_text = scene_summary or _DEFAULT_WORLDLABS_TEXT_PROMPT
        tags = [
            value
            for value in [
                self._string(descriptor.get("scene_id")),
                self._string(descriptor.get("capture_id")),
                self._string(descriptor.get("site_submission_id")),
                site_name,
                industry,
                task_lane,
            ]
            if value
        ]
        keyframe_uri = self._string(descriptor.get("keyframe_uri"))
        frames_index_uri = self._string(descriptor.get("frames_index_uri"))
        arkit_poses_uri = self._string(descriptor.get("arkit_poses_uri"))

        selected_video = video_candidates.get("selected") if isinstance(video_candidates, Mapping) else None
        selected_video_uri = self._string(selected_video.get("uri")) if isinstance(selected_video, Mapping) else ""
        source_id = self._string(selected_video.get("source_id")) if isinstance(selected_video, Mapping) else ""
        generation_source_type = (
            "video_uri"
            if selected_video_uri.startswith(("http://", "https://"))
            else "video_media_asset"
            if selected_video_uri
            else None
        )
        generation_request = {
            "display_name": display_name or f"Blueprint {self._string(descriptor.get('capture_id'))}",
            "model": self._string(os.getenv("WORLDLABS_DEFAULT_MODEL")) or self.provider_model,
            "permission": "private",
            "tags": tags,
            "world_prompt": {
                "type": "video",
                "video_prompt": {
                    "source": "uri" if generation_source_type == "video_uri" else "media_asset",
                    **({"uri": selected_video_uri} if generation_source_type == "video_uri" else {}),
                },
            },
        }
        if prompt_text:
            generation_request["world_prompt"]["text_prompt"] = prompt_text

        return {
            "schema_version": "v1",
            "provider_name": self.provider_name,
            "provider_model": generation_request["model"],
            "scene_id": descriptor.get("scene_id"),
            "capture_id": descriptor.get("capture_id"),
            "site_submission_id": descriptor.get("site_submission_id"),
            "buyer_request_id": descriptor.get("buyer_request_id"),
            "generated_at": utc_now_iso(),
            "status": "ready_for_manual_generation" if selected_video_uri else "blocked",
            "display_name": generation_request["display_name"],
            "generation_source_type": generation_source_type,
            "generation_request": generation_request,
            "selected_video_source_id": source_id or None,
            "selected_video_uri": selected_video_uri or None,
            "video_candidates": video_candidates.get("candidates") if isinstance(video_candidates, Mapping) else [],
            "fallback_inputs": {
                "text_prompt": prompt_text,
                "keyframe_uri": keyframe_uri or None,
                "frames_index_uri": frames_index_uri or None,
                "arkit_poses_uri": arkit_poses_uri or None,
            },
            "privacy": {
                "status": video_candidates.get("privacy_status") if isinstance(video_candidates, Mapping) else None,
                "raw_allowed": bool(video_candidates.get("raw_allowed")) if isinstance(video_candidates, Mapping) else False,
            },
            "notes": [
                "BlueprintCapturePipeline emits the normalized World Labs input bundle only.",
                "Blueprint-WebApp owns API submission, media upload, polling, and world persistence.",
            ],
        }

    def submit(self, *, descriptor: Mapping[str, Any], capture_root: Path) -> Dict[str, Any]:
        started_at = time.time()
        request_manifest = self._build_request_manifest(descriptor=descriptor, capture_root=capture_root)
        latency_ms = int((time.time() - started_at) * 1000)
        return {
            "provider_name": self.provider_name,
            "provider_model": str(request_manifest.get("provider_model") or self.provider_model),
            "provider_run_id": "",
            "status": "queued" if request_manifest.get("selected_video_uri") else "failed",
            "artifact_uris": {},
            "cost_usd": 0.0,
            "latency_ms": latency_ms,
            "input_manifest_uri": descriptor.get("raw_prefix_uri"),
            "worldlabs_request_manifest": request_manifest,
            "raw_response": request_manifest,
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
    worldlabs_request_manifest_path = pipeline_dir / "worldlabs_request_manifest.json"
    try:
      submitted = provider.submit(descriptor=descriptor, capture_root=capture_root)
      normalized = provider.normalize(submitted)
      request_manifest = normalized.get("worldlabs_request_manifest") if isinstance(normalized, Mapping) else None
      if isinstance(request_manifest, Mapping):
          write_json(worldlabs_request_manifest_path, dict(request_manifest))
          artifact_uris = dict(normalized.get("artifact_uris") or {})
          artifact_uris["worldlabs_request_manifest_uri"] = str(worldlabs_request_manifest_path)
          normalized = dict(normalized)
          normalized["artifact_uris"] = artifact_uris
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
