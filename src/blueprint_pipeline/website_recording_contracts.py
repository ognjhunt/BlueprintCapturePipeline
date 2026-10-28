"""Pure validation of same-job recording identities; no media or runtime imports."""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "website_capture_continuation.v2"


def recording_sources(binding: Mapping[str, Any], *, request_id: str, scene_id: str,
                      capture_id: str, bucket: str | None = None) -> list[dict[str, Any]]:
    """Validate the exact active set; older replacements remain historical."""
    def require(condition):
        if not condition:
            raise ValueError("website_recording_binding_invalid")

    original = f"walkthrough-{request_id}"
    require(set(binding) == {"schema_version", "original_capture_id", "capture_id", "lineage",
                             "coordinate_frames_independent", "recording_capture_ids"})
    require(scene_id == f"site-{request_id}" and binding["schema_version"] == SCHEMA
            and binding["original_capture_id"] == original and binding["capture_id"] == capture_id
            and binding["coordinate_frames_independent"] is True
            and isinstance(binding["lineage"], list) and 1 <= len(binding["lineage"]) <= 8)
    expected, seen, sources, active, taking, previous = capture_id, set(), {}, [capture_id], True, None
    for entry in binding["lineage"]:
        require(isinstance(entry, Mapping) and set(entry) == {"child", "parent", "supplement", "recording_scope"})
        child, parent, link = entry["child"], entry["parent"], entry["supplement"]
        require(all(isinstance(value, Mapping) for value in (child, parent, link)))
        require(child.get("capture_id") == expected and expected not in seen and expected != original)
        require(previous is None or dict(child) == dict(previous))
        seen.add(expected)
        scope = entry["recording_scope"]
        require(isinstance(scope, str) and scope in {"complete_walkthrough", "additional_views"}
                and scope == link.get("recording_scope", "complete_walkthrough"))
        require(set(link) == {"parent_capture_id", "parent_bundle_digest", "parent_manifest_uri", "coverage_digest"}
                | ({"recording_scope"} if scope == "additional_views" else set()))
        require(link.get("parent_capture_id") == parent.get("capture_id")
                and link.get("parent_bundle_digest") == parent.get("raw_bundle_digest")
                and link.get("parent_manifest_uri") == parent.get("raw_manifest_uri")
                and re.fullmatch(r"[a-f0-9]{64}", str(link.get("coverage_digest", ""))))
        for source in (child, parent):
            identifier = source.get("capture_id")
            require(isinstance(identifier, str) and (identifier == original or re.fullmatch(r"supplement-[a-f0-9]{40}", identifier)))
            parsed = urlsplit(str(source.get("raw_manifest_uri", "")))
            source_bucket = parsed.netloc
            require(parsed.scheme == "gs" and source_bucket and not parsed.query and not parsed.fragment
                    and (bucket is None or source_bucket == bucket)
                    and parsed.path == f"/scenes/{scene_id}/captures/{identifier}/raw/manifest.json"
                    and re.fullmatch(r"sha256:[a-f0-9]{64}", str(source.get("raw_bundle_digest", ""))))
            bucket = source_bucket
            require(identifier not in sources or dict(source) == sources[identifier])
            sources[identifier] = dict(source)
        if taking:
            if scope == "additional_views":
                active.append(parent["capture_id"])
            else:
                taking = False
        expected, previous = parent["capture_id"], parent
    require(expected == original and binding["recording_capture_ids"] == list(reversed(active)))
    result = []
    for identifier in reversed(active):
        source = sources[identifier]
        video = source.get("raw_video")
        require(isinstance(video, Mapping)
                and set(video) == {"object_name", "generation", "size_bytes", "crc32c"} | ({"sha256"} if "sha256" in video else set()))
        prefix = f"scenes/{scene_id}/captures/{identifier}/raw/"
        require(isinstance(video["object_name"], str) and video["object_name"] in {prefix + "walkthrough.mov", prefix + "walkthrough.mp4"}
                and re.fullmatch(r"[1-9][0-9]{0,19}", str(video["generation"]))
                and type(video["size_bytes"]) is int and video["size_bytes"] > 0
                and re.fullmatch(r"[A-Za-z0-9+/]{6}==", str(video["crc32c"]))
                and ("sha256" not in video or re.fullmatch(r"sha256:[a-f0-9]{64}", str(video["sha256"]))))
        # Verification/readback annotations may be enriched after publication.
        # They do not change the actual recordings or invalidate method reuse.
        immutable = {key: source.get(key) for key in ("capture_id", "raw_bundle_digest", "raw_manifest_uri", "upload_completion_digest")}
        immutable["raw_video"] = dict(video)
        result.append({"capture_id": identifier, "bucket": bucket, "raw_video": dict(video), "source_identity": immutable})
    return result
