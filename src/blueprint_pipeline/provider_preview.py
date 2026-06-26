"""Thin provider abstraction for optional third-party preview generation."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request as _urllib_request
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol
from .common import resolve_gs_uri_to_path, utc_now_iso, write_json
from .launch_proof_policy import production_launch_mode
from .local_capture import resolve_local_capture_context

# ---------------------------------------------------------------------------
# World Labs API helpers
# ---------------------------------------------------------------------------

_WORLDLABS_BASE_URL = "https://api.worldlabs.ai"
_WORLDLABS_POLL_INTERVAL_SECONDS = 15
_WORLDLABS_POLL_MAX_ATTEMPTS = 80  # up to ~20 minutes
_WORLDLABS_TAG_MAX_CHARS = 32


def _worldlabs_api_key() -> str:
    value = str(os.getenv("WORLDLABS_API_KEY") or "").strip()
    if not value:
        raise EnvironmentError("WORLDLABS_API_KEY environment variable is not set")
    return value


def _worldlabs_api_request(
    path: str,
    *,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{_WORLDLABS_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    headers: Dict[str, str] = {
        "WLT-Api-Key": _worldlabs_api_key(),
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = _urllib_request.Request(url, data=data, headers=headers, method=method)
    try:
        with _urllib_request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_err = (exc.read().decode("utf-8") if exc.fp else "") or "request_failed"
        raise RuntimeError(f"worldlabs_api_{exc.code}:{raw_err}") from exc
    parsed = json.loads(raw) if raw else {}
    return parsed if isinstance(parsed, dict) else {}


def _presigned_upload(
    upload_url: str,
    *,
    method: str = "PUT",
    content_type: str,
    data: bytes,
    required_headers: Optional[Dict[str, str]] = None,
) -> None:
    parsed_url = urllib.parse.urlsplit(upload_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise RuntimeError("worldlabs_upload_url_invalid")
    headers: Dict[str, str] = {"Content-Type": content_type}
    if required_headers:
        headers.update({str(key): str(value) for key, value in required_headers.items()})
    upload_method = str(method or "PUT").strip().upper()
    req = _urllib_request.Request(upload_url, data=data, headers=headers, method=upload_method)
    try:
        with _urllib_request.urlopen(req, timeout=600) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        raw_err = (exc.read().decode("utf-8") if exc.fp else "") or ""
        raise RuntimeError(f"worldlabs_upload_failed:{exc.code}:{raw_err}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"worldlabs_upload_failed:url_error:{exc.reason}") from exc


def _read_uri_bytes(uri: str, *, capture_root: Path | None = None) -> bytes:
    if uri.startswith("gs://"):
        if capture_root is not None:
            try:
                context = resolve_local_capture_context(capture_root)
                local_path = resolve_gs_uri_to_path(uri, context.storage_root)
                if local_path.is_file():
                    return local_path.read_bytes()
            except Exception:
                pass
        try:
            from google.cloud import storage as _gcs  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("google-cloud-storage is required to read gs:// URIs") from exc
        without_prefix = uri[5:]
        bucket_name, _, object_path = without_prefix.partition("/")
        client = _gcs.Client()
        return client.bucket(bucket_name).blob(object_path).download_as_bytes()
    if uri.startswith(("https://", "http://")):
        req = _urllib_request.Request(uri, headers={"User-Agent": "BlueprintCapturePipeline/1.0"})
        with _urllib_request.urlopen(req, timeout=600) as resp:
            return resp.read()
    with open(uri, "rb") as fh:
        return fh.read()


def _extension_from_uri(uri: str, fallback: str = "mp4") -> str:
    try:
        basename = uri.rstrip("/").rsplit("/", 1)[-1]
        ext = basename.rsplit(".", 1)[-1] if "." in basename else ""
        return ext.lower() or fallback
    except Exception:
        return fallback


def _filename_from_uri(uri: str, fallback: str = "capture-video.mp4") -> str:
    try:
        return uri.rstrip("/").rsplit("/", 1)[-1] or fallback
    except Exception:
        return fallback


def _mime_for_extension(ext: str) -> str:
    return {
        "mov": "video/quicktime",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
        "avi": "video/x-msvideo",
    }.get(ext.lower(), "video/mp4")


def _worldlabs_tag(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) <= _WORLDLABS_TAG_MAX_CHARS:
        return text
    shortened = text[:_WORLDLABS_TAG_MAX_CHARS].rstrip(" -_/")
    return shortened or text[:_WORLDLABS_TAG_MAX_CHARS]


def _normalize_permission(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    normalized = str(value or "").strip().lower()
    if normalized == "public":
        return {"public": True, "allow_id_access": True, "allowed_readers": [], "allowed_writers": []}
    return {"public": False, "allow_id_access": False, "allowed_readers": [], "allowed_writers": []}


def _poll_worldlabs_until_terminal(
    *,
    provider: "WorldLabsPreviewProvider",
    operation_id: str,
) -> Dict[str, Any]:
    for attempt in range(_WORLDLABS_POLL_MAX_ATTEMPTS):
        if attempt > 0:
            time.sleep(_WORLDLABS_POLL_INTERVAL_SECONDS)
        result = provider.poll(run_id=operation_id)
        if str(result.get("status") or "").lower() in {"ready", "failed"}:
            return result
    return {
        "provider_run_id": operation_id,
        "status": "failed",
        "failure_reason": f"polling_timeout_after_{_WORLDLABS_POLL_MAX_ATTEMPTS}_attempts",
        "operation_done": False,
        "worldlabs_operation": None,
        "worldlabs_world": None,
    }

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
    def submit(
        self,
        *,
        descriptor: Mapping[str, Any],
        capture_root: Path,
        provider_adapter_input: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]: ...

    def poll(self, *, run_id: str) -> Dict[str, Any]: ...

    def normalize(self, payload: Mapping[str, Any]) -> Dict[str, Any]: ...

    def emit_preview_manifest(self, *, normalized: Mapping[str, Any], output_path: Path) -> Dict[str, Any]: ...

    def emit_provenance(self, *, descriptor: Mapping[str, Any], normalized: Mapping[str, Any]) -> Dict[str, Any]: ...


@dataclass
class StubPreviewProvider:
    provider_name: str = "stub_preview"
    provider_model: str = "stub-v1"

    def submit(
        self,
        *,
        descriptor: Mapping[str, Any],
        capture_root: Path,
        provider_adapter_input: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del provider_adapter_input
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
            "world_id": normalized.get("world_id"),
            "launch_url": normalized.get("launch_url"),
            "worldlabs_launch_url": normalized.get("worldlabs_launch_url") or normalized.get("launch_url"),
            "preview_launch_url": normalized.get("preview_launch_url") or normalized.get("launch_url"),
            "labeling": dict(normalized.get("labeling") or {}),
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
    provider_model: str = "marble-1.1"

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
        privacy_status = self._string(
            descriptor.get("privacy_status") or privacy_processing.get("status")
        ).lower()

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

        selected = next((item for item in candidates if item.get("eligible")), None)
        return {
            "privacy_status": privacy_status or None,
            "selected": selected,
            "candidates": candidates,
        }

    def _adapter_prompt_candidates(
        self,
        provider_adapter_input: Mapping[str, Any],
    ) -> Dict[str, Any]:
        conditioning = (
            provider_adapter_input.get("conditioning_inputs")
            if isinstance(provider_adapter_input.get("conditioning_inputs"), Mapping)
            else {}
        )
        rgb_video = (
            conditioning.get("rgb_video") if isinstance(conditioning.get("rgb_video"), Mapping) else {}
        )
        uri = self._string(rgb_video.get("uri"))
        selected = (
            {
                "source_id": self._string(rgb_video.get("source_id"))
                or "privacy_safe_world_model_input",
                "uri": uri,
                "eligible": True,
                "reason": "canonical ProviderAdapterInput privacy-safe RGB video",
            }
            if uri
            else None
        )
        return {
            "privacy_status": "verified" if bool(rgb_video.get("privacy_safe")) else None,
            "selected": selected,
            "candidates": [selected] if selected else [],
        }

    def _build_request_manifest(
        self,
        *,
        descriptor: Mapping[str, Any],
        capture_root: Path,
        provider_adapter_input: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        del capture_root
        adapter_input = (
            dict(provider_adapter_input)
            if isinstance(provider_adapter_input, Mapping)
            else {}
        )
        adapter_status = self._string(adapter_input.get("status")).lower()
        adapter_blockers = [
            str(item).strip()
            for item in (adapter_input.get("blockers") or [])
            if str(item).strip()
        ]
        adapter_source = (
            adapter_input.get("source") if isinstance(adapter_input.get("source"), Mapping) else {}
        )
        canonical_site_package_uri = self._string(
            adapter_source.get("canonical_site_package_uri")
            or adapter_input.get("canonical_site_package_uri")
        )
        provider_adapter_input_uri = self._string(
            adapter_source.get("provider_adapter_input_uri")
            or adapter_input.get("provider_adapter_input_uri")
        )
        blocked_adapter_selected_video = bool(
            adapter_input
            and adapter_status == "blocked"
            and self._adapter_prompt_candidates(adapter_input).get("selected")
        )
        if adapter_input and adapter_status == "blocked" and not blocked_adapter_selected_video:
            return {
                "schema_version": "v1",
                "provider_name": self.provider_name,
                "provider_model": self.provider_model,
                "scene_id": descriptor.get("scene_id"),
                "capture_id": descriptor.get("capture_id"),
                "site_submission_id": descriptor.get("site_submission_id"),
                "buyer_request_id": descriptor.get("buyer_request_id"),
                "generated_at": utc_now_iso(),
                "status": "blocked",
                "display_name": None,
                "generation_source_type": None,
                "generation_request": {},
                "selected_video_source_id": None,
                "selected_video_uri": None,
                "selected_input_checksum_sha256": None,
                "source_input_checksum_sha256": None,
                "source_manifest_uri": None,
                "worldlabs_input_audit_uri": None,
                "privacy_safe_input": False,
                "video_candidates": [],
                "input_labeling": {},
                "input_audit": {},
                "fallback_inputs": {},
                "privacy": {"status": None, "raw_allowed": False},
                "canonical_site_package_uri": canonical_site_package_uri or None,
                "provider_adapter_input_uri": provider_adapter_input_uri or None,
                "adapter_input_status": adapter_status,
                "blockers": adapter_blockers,
            }

        video_candidates = (
            self._adapter_prompt_candidates(adapter_input)
            if adapter_input
            else self._world_prompt_candidates(descriptor)
        )
        metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
        adapter_conditioning = (
            adapter_input.get("conditioning_inputs")
            if isinstance(adapter_input.get("conditioning_inputs"), Mapping)
            else {}
        )
        adapter_rgb_video = (
            adapter_conditioning.get("rgb_video")
            if isinstance(adapter_conditioning.get("rgb_video"), Mapping)
            else {}
        )
        adapter_generation = (
            adapter_input.get("generation")
            if isinstance(adapter_input.get("generation"), Mapping)
            else {}
        )
        input_labeling = (
            dict(adapter_input.get("labeling") or {})
            if adapter_input
            else dict(metadata.get("worldlabs_input_labeling"))
            if isinstance(metadata.get("worldlabs_input_labeling"), Mapping)
            else {}
        )
        if adapter_input and isinstance(adapter_rgb_video.get("labeling"), Mapping):
            input_labeling.update(dict(adapter_rgb_video.get("labeling") or {}))
        input_audit = (
            {
                "privacy_safe_input": bool(adapter_rgb_video.get("privacy_safe")),
                "raw_video_bypass_used": bool(input_labeling.get("raw_video_bypass_used")),
                "source_manifest_uri": adapter_rgb_video.get("source_manifest_uri"),
                "output_video_uri": adapter_rgb_video.get("uri"),
                "output_checksum_sha256": adapter_rgb_video.get("checksum_sha256"),
                "source_checksum_sha256": adapter_rgb_video.get("source_checksum_sha256"),
            }
            if adapter_input
            else dict(metadata.get("worldlabs_input_audit"))
            if isinstance(metadata.get("worldlabs_input_audit"), Mapping)
            else {}
        )
        input_audit_uri = self._string(
            adapter_rgb_video.get("input_audit_uri")
            or metadata.get("worldlabs_input_audit_uri")
        )
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
            adapter_generation.get("display_name")
            or metadata.get("display_name")
            or site_name
            or descriptor.get("capture_id")
            or descriptor.get("scene_id")
        )
        prompt_text = self._string(adapter_generation.get("text_prompt")) or scene_summary or _DEFAULT_WORLDLABS_TEXT_PROMPT
        tags = [
            tag
            for tag in [
                _worldlabs_tag(value)
                for value in [
                    self._string(descriptor.get("scene_id")),
                    self._string(descriptor.get("capture_id")),
                    self._string(descriptor.get("site_submission_id")),
                    site_name,
                    industry,
                    task_lane,
                ]
            ]
            if tag
        ]
        adapter_tags = [
            _worldlabs_tag(item)
            for item in (adapter_generation.get("tags") or [])
            if str(item).strip()
        ]
        tags.extend(item for item in adapter_tags if item not in tags)
        if bool(input_labeling.get("non_production")):
            tags.append(_worldlabs_tag("non-production-preview"))
        if bool(input_labeling.get("unredacted_input")):
            tags.append(_worldlabs_tag("unredacted-raw-input"))
        keyframe_uri = self._string(descriptor.get("keyframe_uri"))
        frames_index_uri = self._string(descriptor.get("frames_index_uri"))
        arkit_poses_uri = self._string(descriptor.get("arkit_poses_uri"))

        selected_video = video_candidates.get("selected") if isinstance(video_candidates, Mapping) else None
        selected_video_uri = self._string(selected_video.get("uri")) if isinstance(selected_video, Mapping) else ""
        source_id = self._string(selected_video.get("source_id")) if isinstance(selected_video, Mapping) else ""
        audit_output_uri = self._string(input_audit.get("output_video_uri"))
        privacy_safe_input = bool(input_audit.get("privacy_safe_input"))
        raw_bypass_used = bool(input_audit.get("raw_video_bypass_used") or input_labeling.get("raw_video_bypass_used"))
        selected_input_checksum = self._string(input_audit.get("output_checksum_sha256"))
        source_input_checksum = self._string(input_audit.get("source_checksum_sha256"))
        source_manifest_uri = self._string(
            input_audit.get("source_manifest_uri")
            or input_audit.get("source_manifest")
            or metadata.get("worldlabs_input_manifest_uri")
        )
        if production_launch_mode():
            audit_matches_selected = bool(audit_output_uri and selected_video_uri and audit_output_uri == selected_video_uri)
            if not input_audit_uri or not privacy_safe_input or raw_bypass_used or not audit_matches_selected:
                raise RuntimeError("production_worldlabs_input_audit_missing_or_invalid")
            if not source_manifest_uri or not selected_input_checksum:
                raise RuntimeError("production_worldlabs_input_audit_incomplete")
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

        request_status = (
            "blocked"
            if adapter_input and adapter_status == "blocked"
            else "ready_for_generation"
            if selected_video_uri
            else "blocked"
        )
        return {
            "schema_version": "v1",
            "provider_name": self.provider_name,
            "provider_model": generation_request["model"],
            "scene_id": descriptor.get("scene_id"),
            "capture_id": descriptor.get("capture_id"),
            "site_submission_id": descriptor.get("site_submission_id"),
            "buyer_request_id": descriptor.get("buyer_request_id"),
            "generated_at": utc_now_iso(),
            "status": request_status,
            "display_name": generation_request["display_name"],
            "generation_source_type": generation_source_type,
            "generation_request": generation_request,
            "selected_video_source_id": source_id or None,
            "selected_video_uri": selected_video_uri or None,
            "selected_input_checksum_sha256": selected_input_checksum or None,
            "source_input_checksum_sha256": source_input_checksum or None,
            "source_manifest_uri": source_manifest_uri or None,
            "worldlabs_input_audit_uri": input_audit_uri or None,
            "privacy_safe_input": privacy_safe_input,
            "video_candidates": video_candidates.get("candidates") if isinstance(video_candidates, Mapping) else [],
            "input_labeling": input_labeling,
            "input_audit": input_audit,
            "fallback_inputs": {
                "text_prompt": prompt_text,
                "keyframe_uri": keyframe_uri or None,
                "frames_index_uri": frames_index_uri or None,
                "arkit_poses_uri": arkit_poses_uri or None,
            },
            "privacy": {
                "status": video_candidates.get("privacy_status") if isinstance(video_candidates, Mapping) else None,
                "raw_allowed": False,
            },
            "canonical_site_package_uri": canonical_site_package_uri or None,
            "provider_adapter_input_uri": provider_adapter_input_uri or None,
            "adapter_input_status": adapter_status or None,
            "blockers": adapter_blockers,
        }

    def _upload_video_as_media_asset(
        self,
        video_uri: str,
        *,
        descriptor: Mapping[str, Any],
        capture_root: Path,
    ) -> Dict[str, Any]:
        """Upload video to World Labs media assets. Returns upload proof fields."""
        ext = _extension_from_uri(video_uri)
        filename = _filename_from_uri(video_uri, f"capture-video.{ext}")
        upload_payload = _worldlabs_api_request(
            "/marble/v1/media-assets:prepare_upload",
            method="POST",
            body={
                "file_name": filename,
                "extension": ext,
                "kind": "video",
                "metadata": {
                    "scene_id": descriptor.get("scene_id"),
                    "capture_id": descriptor.get("capture_id"),
                    "site_submission_id": descriptor.get("site_submission_id"),
                },
            },
        )
        media_asset = upload_payload.get("media_asset") or {}
        media_asset_id = ""
        if isinstance(media_asset, dict):
            media_asset_id = str(media_asset.get("media_asset_id") or media_asset.get("id") or "").strip()
        upload_info = upload_payload.get("upload_info") or {}
        if not isinstance(upload_info, dict):
            upload_info = {}
        upload_url = str(upload_info.get("upload_url") or "").strip()
        if not upload_url:
            raise RuntimeError("worldlabs_upload_url_missing")
        video_bytes = _read_uri_bytes(video_uri, capture_root=capture_root)
        _presigned_upload(
            upload_url,
            method=str(upload_info.get("upload_method") or "PUT"),
            content_type=_mime_for_extension(ext),
            data=video_bytes,
            required_headers=upload_info.get("required_headers") or {},
        )
        if not media_asset_id:
            raise RuntimeError("worldlabs_media_asset_id_missing")
        return {
            "media_asset_id": media_asset_id,
            "upload_id": str(upload_payload.get("upload_id") or upload_payload.get("id") or media_asset_id).strip(),
            "upload_payload": upload_payload,
            "filename": filename,
            "content_type": _mime_for_extension(ext),
        }

    def submit(
        self,
        *,
        descriptor: Mapping[str, Any],
        capture_root: Path,
        provider_adapter_input: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        started_at = time.time()
        request_manifest = self._build_request_manifest(
            descriptor=descriptor,
            capture_root=capture_root,
            provider_adapter_input=provider_adapter_input,
        )
        selected_video_uri = str(request_manifest.get("selected_video_uri") or "").strip()
        generation_source_type = str(request_manifest.get("generation_source_type") or "").strip()

        if not selected_video_uri:
            failure_reason = "no_eligible_video"
            if str(request_manifest.get("adapter_input_status") or "").lower() == "blocked":
                blockers = [
                    str(item).strip()
                    for item in (request_manifest.get("blockers") or [])
                    if str(item).strip()
                ]
                failure_reason = "provider_adapter_input_blocked:" + ",".join(blockers or ["unknown"])
            return {
                "provider_name": self.provider_name,
                "provider_model": self.provider_model,
                "provider_run_id": "",
                "status": "failed",
                "artifact_uris": {},
                "cost_usd": 0.0,
                "latency_ms": int((time.time() - started_at) * 1000),
                "failure_reason": failure_reason,
                "worldlabs_request_manifest": request_manifest,
                "canonical_site_package_uri": request_manifest.get("canonical_site_package_uri"),
                "provider_adapter_input_uri": request_manifest.get("provider_adapter_input_uri"),
                "adapter_input_status": request_manifest.get("adapter_input_status"),
                "adapter_input_blockers": request_manifest.get("blockers") or [],
                "raw_response": None,
            }

        try:
            generation_request: Dict[str, Any] = json.loads(
                json.dumps(request_manifest.get("generation_request") or {})
            )
            generation_request["permission"] = _normalize_permission(generation_request.get("permission"))
            world_prompt: Dict[str, Any] = dict(generation_request.get("world_prompt") or {})
            if not world_prompt.get("text_prompt"):
                world_prompt["text_prompt"] = _DEFAULT_WORLDLABS_TEXT_PROMPT
            video_prompt: Dict[str, Any] = dict(world_prompt.get("video_prompt") or {})

            media_asset_id = ""
            upload_id = ""
            upload_payload: Dict[str, Any] = {}
            if generation_source_type == "video_uri":
                video_prompt["source"] = "uri"
                video_prompt["uri"] = selected_video_uri
                video_prompt.pop("media_asset_id", None)
            else:
                upload_proof = self._upload_video_as_media_asset(
                    selected_video_uri,
                    descriptor=descriptor,
                    capture_root=capture_root,
                )
                media_asset_id = str(upload_proof.get("media_asset_id") or "").strip()
                upload_id = str(upload_proof.get("upload_id") or media_asset_id).strip()
                upload_payload = dict(upload_proof.get("upload_payload") or {})
                video_prompt["source"] = "media_asset"
                video_prompt["media_asset_id"] = media_asset_id
                video_prompt.pop("uri", None)
                generation_source_type = "video_media_asset"

            world_prompt["video_prompt"] = video_prompt
            generation_request["world_prompt"] = world_prompt

            operation = _worldlabs_api_request(
                "/marble/v1/worlds:generate",
                method="POST",
                body=generation_request,
            )
            operation_id = str(operation.get("id") or operation.get("operation_id") or "").strip()
            latency_ms = int((time.time() - started_at) * 1000)
            return {
                "provider_name": self.provider_name,
                "provider_model": str(request_manifest.get("provider_model") or self.provider_model),
                "provider_run_id": operation_id,
                "status": "processing" if operation_id else "failed",
                "artifact_uris": {},
                "cost_usd": 0.0,
                "latency_ms": latency_ms,
                "failure_reason": None if operation_id else "worldlabs_generate_returned_no_operation_id",
                "input_manifest_uri": descriptor.get("raw_prefix_uri"),
                "worldlabs_request_manifest": request_manifest,
                "worldlabs_operation": operation,
                "worldlabs_operation_id": operation_id or None,
                "worldlabs_media_asset_id": media_asset_id or None,
                "worldlabs_upload_id": upload_id or None,
                "worldlabs_upload": upload_payload or None,
                "generation_source_type": generation_source_type,
                "selected_input_checksum_sha256": request_manifest.get("selected_input_checksum_sha256"),
                "source_input_checksum_sha256": request_manifest.get("source_input_checksum_sha256"),
                "source_manifest_uri": request_manifest.get("source_manifest_uri"),
                "worldlabs_input_audit_uri": request_manifest.get("worldlabs_input_audit_uri"),
                "privacy_safe_input": request_manifest.get("privacy_safe_input"),
                "canonical_site_package_uri": request_manifest.get("canonical_site_package_uri"),
                "provider_adapter_input_uri": request_manifest.get("provider_adapter_input_uri"),
                "adapter_input_status": request_manifest.get("adapter_input_status"),
                "adapter_input_blockers": request_manifest.get("blockers") or [],
                "raw_response": operation,
                "labeling": request_manifest.get("input_labeling") or {},
            }
        except Exception as exc:
            return {
                "provider_name": self.provider_name,
                "provider_model": str(request_manifest.get("provider_model") or self.provider_model),
                "provider_run_id": "",
                "status": "failed",
                "artifact_uris": {},
                "cost_usd": 0.0,
                "latency_ms": int((time.time() - started_at) * 1000),
                "failure_reason": str(exc),
                "input_manifest_uri": descriptor.get("raw_prefix_uri"),
                "worldlabs_request_manifest": request_manifest,
                "worldlabs_operation": None,
                "worldlabs_operation_id": None,
                "worldlabs_media_asset_id": None,
                "worldlabs_upload_id": None,
                "generation_source_type": generation_source_type or None,
                "selected_input_checksum_sha256": request_manifest.get("selected_input_checksum_sha256"),
                "source_input_checksum_sha256": request_manifest.get("source_input_checksum_sha256"),
                "source_manifest_uri": request_manifest.get("source_manifest_uri"),
                "worldlabs_input_audit_uri": request_manifest.get("worldlabs_input_audit_uri"),
                "privacy_safe_input": request_manifest.get("privacy_safe_input"),
                "canonical_site_package_uri": request_manifest.get("canonical_site_package_uri"),
                "provider_adapter_input_uri": request_manifest.get("provider_adapter_input_uri"),
                "adapter_input_status": request_manifest.get("adapter_input_status"),
                "adapter_input_blockers": request_manifest.get("blockers") or [],
                "raw_response": None,
                "labeling": request_manifest.get("input_labeling") or {},
            }

    def poll(self, *, run_id: str) -> Dict[str, Any]:
        operation = _worldlabs_api_request(f"/marble/v1/operations/{run_id}")
        done = bool(operation.get("done"))
        metadata = operation.get("metadata") if isinstance(operation.get("metadata"), Mapping) else {}
        response = operation.get("response") if isinstance(operation.get("response"), Mapping) else {}
        world_id = str(
            response.get("world_id")
            or metadata.get("world_id")
            or operation.get("world_id")
            or ""
        ).strip()
        error = operation.get("error")
        failure_reason = (
            str(error.get("message") or "") if isinstance(error, dict) else
            str(operation.get("failure_reason") or "")
        ).strip() or None

        if done:
            world = dict(response) if response else {}
            if not world and world_id:
                world = _worldlabs_api_request(f"/marble/v1/worlds/{world_id}")
            launch_url = str(world.get("world_marble_url") or "").strip()
            return {
                "provider_run_id": run_id,
                "status": "ready" if launch_url else "failed",
                "operation_done": True,
                "world_id": str(world.get("world_id") or world_id or "").strip() or None,
                "launch_url": launch_url or None,
                "failure_reason": failure_reason if not launch_url else None,
                "operation_terminal_status": "ready" if launch_url else "failed",
                "worldlabs_operation": operation,
                "worldlabs_world": world or None,
            }
        raw_status = str(operation.get("status") or "").lower()
        return {
            "provider_run_id": run_id,
            "status": "queued" if raw_status in {"queued", "pending"} else "processing",
            "operation_done": False,
            "operation_terminal_status": "processing",
            "worldlabs_operation": operation,
        }


def resolve_preview_provider(name: str) -> PreviewProvider:
    normalized = str(name or "").strip().lower()
    if normalized in {"world_labs", "worldlabs"}:
        return WorldLabsPreviewProvider()
    if normalized in {"stub", "stub_preview"}:
        return StubPreviewProvider()
    if not normalized:
        raise ValueError("preview_provider_not_configured")
    raise ValueError(f"unsupported_preview_provider:{normalized}")


def run_preview_provider(
    *,
    provider_name: str,
    descriptor: Mapping[str, Any],
    capture_root: Path,
    pipeline_dir: Path,
    provider_adapter_input: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    manifest_path = pipeline_dir / "preview_manifest.json"
    worldlabs_request_manifest_path = pipeline_dir / "worldlabs_request_manifest.json"
    worldlabs_operation_manifest_path = pipeline_dir / "worldlabs_operation_manifest.json"
    worldlabs_world_manifest_path = pipeline_dir / "worldlabs_world_manifest.json"
    provider: PreviewProvider | None = None
    try:
        provider = resolve_preview_provider(provider_name)
        if provider_adapter_input is not None:
            submitted = provider.submit(
                descriptor=descriptor,
                capture_root=capture_root,
                provider_adapter_input=provider_adapter_input,
            )
        else:
            submitted = provider.submit(descriptor=descriptor, capture_root=capture_root)
        normalized: Dict[str, Any] = dict(provider.normalize(submitted))

        # Write request manifest
        request_manifest = normalized.get("worldlabs_request_manifest")
        if isinstance(request_manifest, Mapping):
            write_json(worldlabs_request_manifest_path, dict(request_manifest))
            artifact_uris = dict(normalized.get("artifact_uris") or {})
            artifact_uris["worldlabs_request_manifest_uri"] = str(worldlabs_request_manifest_path)
            normalized["artifact_uris"] = artifact_uris
            normalized["worldlabs_request_manifest_uri"] = str(worldlabs_request_manifest_path)

        # Write initial operation manifest if generate returned one
        worldlabs_operation = normalized.get("worldlabs_operation")
        if isinstance(worldlabs_operation, Mapping):
            write_json(worldlabs_operation_manifest_path, dict(worldlabs_operation))
            artifact_uris = dict(normalized.get("artifact_uris") or {})
            artifact_uris["worldlabs_operation_manifest_uri"] = str(worldlabs_operation_manifest_path)
            normalized["artifact_uris"] = artifact_uris

        # Poll until terminal state for World Labs providers
        operation_id = str(normalized.get("provider_run_id") or "").strip()
        if isinstance(provider, WorldLabsPreviewProvider) and operation_id:
            poll_result = _poll_worldlabs_until_terminal(provider=provider, operation_id=operation_id)
            normalized["status"] = poll_result.get("status", "failed")
            normalized["failure_reason"] = poll_result.get("failure_reason")
            if poll_result.get("world_id"):
                normalized["world_id"] = poll_result["world_id"]
            if poll_result.get("launch_url"):
                normalized["launch_url"] = poll_result["launch_url"]
                normalized["worldlabs_launch_url"] = poll_result["launch_url"]
                normalized["preview_launch_url"] = poll_result["launch_url"]
            # Write final operation manifest
            final_operation = poll_result.get("worldlabs_operation")
            if isinstance(final_operation, Mapping):
                write_json(worldlabs_operation_manifest_path, dict(final_operation))
                artifact_uris = dict(normalized.get("artifact_uris") or {})
                artifact_uris["worldlabs_operation_manifest_uri"] = str(worldlabs_operation_manifest_path)
                normalized["artifact_uris"] = artifact_uris
                normalized["worldlabs_operation_manifest_uri"] = str(worldlabs_operation_manifest_path)
            # Write world manifest
            worldlabs_world = poll_result.get("worldlabs_world")
            if isinstance(worldlabs_world, Mapping):
                write_json(worldlabs_world_manifest_path, dict(worldlabs_world))
                artifact_uris = dict(normalized.get("artifact_uris") or {})
                artifact_uris["worldlabs_world_manifest_uri"] = str(worldlabs_world_manifest_path)
                normalized["artifact_uris"] = artifact_uris
                normalized["worldlabs_world_manifest_uri"] = str(worldlabs_world_manifest_path)
            normalized["operation_terminal_status"] = poll_result.get("operation_terminal_status") or poll_result.get("status")

        worldlabs_asset_materialization: Dict[str, Any] | None = None
        if (
            isinstance(provider, WorldLabsPreviewProvider)
            and worldlabs_world_manifest_path.is_file()
            and normalized.get("world_id")
        ):
            from .worldlabs_asset_materialization import materialize_worldlabs_assets

            try:
                worldlabs_asset_materialization = materialize_worldlabs_assets(
                    capture_root=capture_root,
                    world_manifest=worldlabs_world_manifest_path,
                )
                artifact_uris = dict(normalized.get("artifact_uris") or {})
                artifact_uris.update(
                    {
                        "worldlabs_asset_materialization_manifest_uri": (
                            worldlabs_asset_materialization.get("manifest_path")
                        ),
                        "worldlabs_export_manifest_uri": (
                            worldlabs_asset_materialization.get("export_manifest_path")
                        ),
                    }
                )
                normalized["artifact_uris"] = artifact_uris
                normalized["worldlabs_asset_materialization"] = worldlabs_asset_materialization
            except Exception as exc:
                normalized["worldlabs_asset_materialization"] = {
                    "schema_version": "worldlabs_asset_materialization_result.v1",
                    "status": "failed",
                    "failure_reason": str(exc),
                    "claim_boundary": {
                        "artifact_purpose": "worldlabs_remote_asset_materialization_for_pre_gpu_handoff",
                        "simulator_execution_proven": False,
                        "rank_fidelity_result_proven": False,
                    },
                }

        marble_sim_asset_handoff: Dict[str, Any] | None = None
        if (
            isinstance(provider, WorldLabsPreviewProvider)
            and worldlabs_world_manifest_path.is_file()
            and normalized.get("world_id")
        ):
            from .marble_sim_assets import build_marble_sim_assets

            try:
                marble_sim_asset_handoff = build_marble_sim_assets(
                    capture_root=capture_root,
                    world_manifest=worldlabs_world_manifest_path,
                )
                artifact_uris = dict(normalized.get("artifact_uris") or {})
                artifact_uris.update(
                    {
                        "marble_asset_manifest_uri": marble_sim_asset_handoff.get("manifest_path"),
                        "marble_asset_validation_uri": marble_sim_asset_handoff.get("validation_path"),
                        "marble_simready_bridge_uri": marble_sim_asset_handoff.get("bridge_path"),
                    }
                )
                for framework, path in dict(
                    marble_sim_asset_handoff.get("simulator_review_manifests") or {}
                ).items():
                    artifact_uris[f"marble_{framework}_review_manifest_uri"] = path
                normalized["artifact_uris"] = artifact_uris
                normalized["marble_sim_asset_handoff"] = marble_sim_asset_handoff
            except Exception as exc:
                normalized["marble_sim_asset_handoff"] = {
                    "schema_version": "marble_sim_assets_result.v1",
                    "status": "failed",
                    "failure_reason": str(exc),
                    "claim_boundary": {
                        "artifact_purpose": "marble_sim_asset_review_packaging_only",
                        "simulator_execution_proven": False,
                        "rank_fidelity_result_proven": False,
                    },
                }

        provider.emit_preview_manifest(normalized=normalized, output_path=manifest_path)
        provenance = provider.emit_provenance(descriptor=descriptor, normalized=normalized)
        run_manifest: Dict[str, Any] = {
            "schema_version": "v1",
            "provider_name": normalized.get("provider_name"),
            "provider_model": normalized.get("provider_model"),
            "provider_run_id": normalized.get("provider_run_id"),
            "worldlabs_operation_id": normalized.get("worldlabs_operation_id") or normalized.get("provider_run_id"),
            "worldlabs_media_asset_id": normalized.get("worldlabs_media_asset_id"),
            "worldlabs_upload_id": normalized.get("worldlabs_upload_id"),
            "operation_terminal_status": normalized.get("operation_terminal_status") or normalized.get("status"),
            "status": normalized.get("status"),
            "input_manifest_uri": normalized.get("input_manifest_uri"),
            "preview_manifest_uri": str(manifest_path),
            "worldlabs_request_manifest_uri": normalized.get("worldlabs_request_manifest_uri"),
            "worldlabs_operation_manifest_uri": normalized.get("worldlabs_operation_manifest_uri"),
            "worldlabs_world_manifest_uri": normalized.get("worldlabs_world_manifest_uri"),
            "request_manifest_uri": normalized.get("worldlabs_request_manifest_uri"),
            "selected_input_checksum_sha256": normalized.get("selected_input_checksum_sha256"),
            "source_input_checksum_sha256": normalized.get("source_input_checksum_sha256"),
            "source_manifest_uri": normalized.get("source_manifest_uri"),
            "worldlabs_input_audit_uri": normalized.get("worldlabs_input_audit_uri"),
            "privacy_safe_input": normalized.get("privacy_safe_input"),
            "canonical_site_package_uri": normalized.get("canonical_site_package_uri"),
            "provider_adapter_input_uri": normalized.get("provider_adapter_input_uri"),
            "adapter_input_status": normalized.get("adapter_input_status"),
            "adapter_input_blockers": normalized.get("adapter_input_blockers") or [],
            "artifact_uris": normalized.get("artifact_uris") or {},
            "cost_usd": normalized.get("cost_usd"),
            "latency_ms": normalized.get("latency_ms"),
            "failure_reason": normalized.get("failure_reason"),
            "world_id": normalized.get("world_id"),
            "launch_url": normalized.get("launch_url"),
            "worldlabs_launch_url": normalized.get("worldlabs_launch_url") or normalized.get("launch_url"),
            "preview_launch_url": normalized.get("preview_launch_url") or normalized.get("launch_url"),
            "worldlabs_asset_materialization": (
                normalized.get("worldlabs_asset_materialization") or {}
            ),
            "marble_sim_asset_handoff": normalized.get("marble_sim_asset_handoff") or {},
            "labeling": dict(normalized.get("labeling") or {}),
            "provenance": provenance,
        }
    except Exception as exc:
        run_manifest = {
            "schema_version": "v1",
            "provider_name": getattr(provider, "provider_name", str(provider_name or "").strip() or None),
            "provider_model": getattr(provider, "provider_model", "unknown"),
            "provider_run_id": "",
            "status": "failed",
            "input_manifest_uri": descriptor.get("raw_prefix_uri"),
            "preview_manifest_uri": str(manifest_path),
            "artifact_uris": {},
            "cost_usd": None,
            "latency_ms": None,
            "failure_reason": str(exc),
            "world_id": None,
            "launch_url": None,
            "worldlabs_launch_url": None,
            "preview_launch_url": None,
            "labeling": {},
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
