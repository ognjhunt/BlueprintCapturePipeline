"""Generation-pinned capture staging from a finite producer membership."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path

from .common import PipelineError
from .capture_original_owner_observer import CaptureOwnerObservationError, OWNER_OBSERVATION_REASON_CODES
from .capture_delivery_membership import load_selected_capture_membership
from .task_evaluation_scene_retirement_access import SceneRetirementAccessError
from .task_evaluation_scene_retirement_generations import (
    _prepare_capture_parent, birth_capture_member,
)


STAGING_REASON_CODES = frozenset({
    'capture_staging_owner_observation_unavailable',
    'capture_staging_membership_unavailable',
    'scene_capture_birth_policy_unavailable',
    'capture_original_birth_refused',
    'capture_original_birth_unavailable',
    'capture_staging_unavailable',
}) | OWNER_OBSERVATION_REASON_CODES


class CaptureStagingError(PipelineError):
    """A bounded refusal reason, never source, owner or provider exception text."""


def _local_matches(path: Path, row: dict) -> bool:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != row['size_bytes']:
            return False
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        after = path.lstat()
        return ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                and 'sha256:' + digest.hexdigest() == row['sha256'])
    except OSError:
        return False


def stage_selected_capture(listener, handoff, *, storage_root, storage_client, expected_purpose=None):
    """Acquire signed owner and exact historical source before capture birth."""
    from .capture_original_owner_observer import load_original_owner_observation

    if handoff.source_membership_selector is None:
        raise CaptureStagingError('capture_original_birth_unavailable')
    try:
        observation = load_original_owner_observation(
            bucket=handoff.bucket, scene_id=handoff.scene_id, capture_id=handoff.capture_id,
            marker_generation=handoff.source_finalize['generation'],
            **({'expected_purpose': expected_purpose} if expected_purpose else {}))
    except Exception as error:
        if isinstance(error, CaptureOwnerObservationError) and str(error) in OWNER_OBSERVATION_REASON_CODES:
            raise
        raise CaptureStagingError('capture_staging_owner_observation_unavailable') from error
    try:
        client = storage_client or listener.storage.Client()
        membership_raw, membership, selected = load_selected_capture_membership(
            storage_client=client, handoff=handoff, observation=observation)
    except Exception as error:
        raise CaptureStagingError('capture_staging_membership_unavailable') from error
    capture_root = listener._handoff_capture_root(handoff, storage_root=storage_root)
    try:
        born = birth_capture_member(
            capture_root, observation=observation,
            membership_selector=dict(handoff.source_membership_selector),
            membership_raw=membership_raw, require_policy=True,
            **({'expected_purpose': expected_purpose} if expected_purpose else {}))
    except Exception as error:
        if isinstance(error, CaptureOwnerObservationError) and str(error) in OWNER_OBSERVATION_REASON_CODES:
            raise
        reason = ('scene_capture_birth_policy_unavailable'
                  if isinstance(error, SceneRetirementAccessError)
                  and str(error) == 'scene_capture_birth_policy_unavailable'
                  else 'capture_original_birth_refused')
        raise CaptureStagingError(reason) from error
    if born is None:
        raise CaptureStagingError('capture_original_birth_unavailable')
    selector = handoff.source_membership_selector
    previous = listener._read_optional_json_object(
        capture_root / listener.STAGING_MANIFEST_FILENAME)
    previous_rows = {}
    if (previous.get('source_membership_selector') == selector
            and previous.get('delivery_key') == membership['delivery_key']
            and type(previous.get('objects')) is list):
        previous_rows = {row.get('name'): row for row in previous['objects']
                         if type(row) is dict and type(row.get('name')) is str}
    manifest_rows = []
    downloads = []
    temporary = []
    destinations = []
    root_device = capture_root.stat().st_dev
    try:
        for row, blob in selected:
            relative = row['relative_path']
            destination = listener.contained_path(
                capture_root, *relative.split('/'), field='selected capture source destination')
            if destination != capture_root.joinpath(*relative.split('/')):
                raise listener.PipelineError('capture_staging_destination_alias')
            _prepare_capture_parent(destination, [{'root': str(capture_root), 'device': root_device}])
            manifest_row = {
                'name': row['object_name'], 'relative_path': relative,
                'size': row['size_bytes'], 'generation': row['generation'],
                'md5_hash': None, 'crc32c': row['crc32c'], 'sha256': row['sha256'],
                'delivery_key': membership['delivery_key'],
                'source_membership_sha256': selector['sha256'],
            }
            manifest_rows.append(manifest_row)
            destinations.append((row, destination))
            if previous_rows.get(row['object_name']) == manifest_row and _local_matches(destination, row):
                continue
            if destination.is_symlink():
                raise listener.PipelineError('capture_staging_destination_alias')
            pending = destination.with_name('.' + secrets.token_hex(16) + '.pending')
            temporary.append((pending, destination, row))
            downloads.append((blob, pending))
        listener.download_with_reservation(
            downloads=downloads, manifest_rows=manifest_rows,
            storage_root=storage_root, capture_root=capture_root,
            selected_generations={row['object_name']: row['generation'] for row, _ in selected})
        for pending, destination, row in temporary:
            if not _local_matches(pending, row):
                raise listener.PipelineError('capture_staging_content_mismatch')
            if destination.is_symlink():
                raise listener.PipelineError('capture_staging_destination_alias')
            os.replace(pending, destination)
        if any(not _local_matches(destination, row) for row, destination in destinations):
            raise listener.PipelineError('capture_staging_content_mismatch')
        manifest_path = capture_root / listener.STAGING_MANIFEST_FILENAME
        if manifest_path.is_symlink():
            raise listener.PipelineError('capture_staging_manifest_alias')
        listener.write_json(manifest_path, {
            'schema_version': listener.STAGING_MANIFEST_SCHEMA_VERSION,
            'bucket': handoff.bucket, 'prefix': f'{handoff.capture_prefix}/',
            'staged_at': listener.utc_now_iso(), 'objects': manifest_rows,
            'delivery_key': membership['delivery_key'],
            'source_membership_selector': selector,
            'local_generation_id': born['generation_id'],
            'raw_video': observation['producer_delivery']['raw_video'],
        })
        if not (capture_root / 'pipeline_handoff.json').is_file():
            listener._synthesize_pipeline_handoff(handoff, capture_root=capture_root)
        return capture_root
    finally:
        for pending, _, _ in temporary:
            try:
                if pending.is_file():
                    pending.unlink()
            except OSError:
                pass
