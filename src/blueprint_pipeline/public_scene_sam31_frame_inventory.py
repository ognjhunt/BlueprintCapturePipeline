"""Represent omitted thresholded observations against the exact retained input frames."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
from .scene_placement.semantic_source_track_import import MASK_ENCODING, _validate_frame_registry, _valid_digest

FRAME_REGISTRY_ERROR = 'sam31_review_retained_frame_registry_unproven'
FRAME_BINDING_ERROR = 'sam31_review_frame_observation_binding_mismatch'


def complete_sparse_frame_inventory(source_tracks: Mapping, frames: dict, packet_path: str | Path) -> dict:
    """Add only explicit no-observation rows; never add a track, detection, or positive pixel.

    Source results stay untouched. Missing rows mean the sparse result retained
    no above-threshold observation, not that the visible subject was absent or
    that segmentation coverage was qualified.
    """
    try:
        path = Path(packet_path).expanduser()
        if path.is_symlink() or not path.is_file():
            raise ValueError(FRAME_REGISTRY_ERROR)
        packet = json.loads(path.read_text())
        if (packet.get('schema_version') != 'public_scene_sam31_task_input_packet.v1'
                or packet.get('status') != 'prepared_no_upload_no_execution'
                or packet.get('receipt_digest') != canonical_digest(packet, digest_field='receipt_digest')):
            raise ValueError(FRAME_REGISTRY_ERROR)
        declared = list((packet.get('camera_frame_map') or {}).values())
        if len(declared) == len(set(declared)) and set(declared) == set(frames):
            return dict(frames)  # Existing complete inventories need no new normalization authority.
        record = packet.get('run_request', {})
        relative, absolute = record.get('relative_path'), record.get('path')
        if bool(relative) == bool(absolute):
            raise ValueError(FRAME_REGISTRY_ERROR)
        request_path = path.parent/str(relative) if relative else Path(str(absolute))
        if (request_path.is_symlink() or not request_path.is_file()
                or (relative and not request_path.resolve().is_relative_to(path.parent.resolve()))):
            raise ValueError(FRAME_REGISTRY_ERROR)
        raw = request_path.read_bytes()
        request = json.loads(raw)
        if (request.get('schema_version') != 'semantic_sam31_source_track_run_request.v1'
                or len(raw) != record.get('size_bytes') or 'sha256:'+hashlib.sha256(raw).hexdigest() != record.get('sha256')
                or canonical_json_digest(request) != record.get('request_digest')):
            raise ValueError(FRAME_REGISTRY_ERROR)
        blockers = []
        registry = _validate_frame_registry(request, blockers)
        frame_ids = list((packet.get('camera_frame_map') or {}).values())
        if (blockers or packet.get('camera_count') != len(registry)
                or len(frame_ids) != len(set(frame_ids)) or set(frame_ids) != set(registry)
                or not set(frames).issubset(registry)
                or any(not _valid_digest(request.get('bindings', {}).get(key))
                       or request.get('bindings', {}).get(key) != source_tracks.get('bindings', {}).get(key)
                       for key in ('capture_digest', 'retained_video_digest', 'camera_solution_digest', 'frame_registry_digest'))
                or not _valid_digest(request.get('provider_profile', {}).get('profile_digest'))
                or request['provider_profile']['profile_digest'] != source_tracks.get('bindings', {}).get('provider_profile_digest')
                or request['provider_profile']['profile_digest'] != source_tracks.get('provider_profile', {}).get('profile_digest')):
            raise ValueError(FRAME_REGISTRY_ERROR)
    except (OSError, TypeError, KeyError, AttributeError, ValueError) as exc:
        raise ValueError(FRAME_REGISTRY_ERROR) from exc
    complete = dict(frames)
    identity_fields = ('source_frame_id', 'source_frame_digest', 'decoded_pts_seconds',
                       'camera_record_digest', 'width', 'height')
    for frame_id, source in registry.items():
        if frame_id in complete:
            if any(complete[frame_id].get(key) != source.get(key) for key in identity_fields):
                raise ValueError(FRAME_BINDING_ERROR)
            continue
        complete[frame_id] = {key: source[key] for key in identity_fields}
        complete[frame_id].update(mask_encoding=MASK_ENCODING, track_masks=[],
                                  mask_artifact_digest=canonical_json_digest([]),
                                  normalization_basis='retained_input_without_above_threshold_track_observation')
    return complete
