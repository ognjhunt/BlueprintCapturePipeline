"""Verify received native inventories without qualifying their scientific claims."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Mapping

from .decision_evidence_contracts import canonical_digest


class ProviderOutputInventoryError(ValueError):
    pass


def safe_member_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or any(ord(char) < 32 for char in name) or len(name.encode('utf-8')) > 4096
            or PurePosixPath(name).is_absolute()
            or any(part in ('', '.', '..') or len(part.encode('utf-8')) > 255 for part in name.split('/'))):
        raise ProviderOutputInventoryError('provider_output_archive_path_invalid')
    return name


def _read(root, relative, maximum_json_bytes):
    path = root / safe_member_name(relative)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_json_bytes:
        raise ProviderOutputInventoryError('provider_output_native_document_missing_or_oversize')
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        raise ProviderOutputInventoryError('provider_output_native_document_invalid') from None
    if not isinstance(value, dict):
        raise ProviderOutputInventoryError('provider_output_native_document_invalid')
    return value


def verify_native_inventory(root: Path, binding: Mapping, members: Mapping,
                            *, maximum_json_bytes=128 * 1024**2,
                            maximum_result_json_bytes=256 * 1024**2):
    identity = _read(root, binding['identity_document'], maximum_json_bytes)
    if (identity.get('schema_version') != 'policy_canary_static_startup_preflight.v1'
            or identity.get('run_id') != binding['run_id']
            or identity.get('runtime_inputs_digest') != binding['runtime_inputs_digest']
            or identity.get('result_digest') != canonical_digest(identity, digest_field='result_digest')):
        raise ProviderOutputInventoryError('provider_output_native_identity_mismatch')
    # Quick-10 retains all 20 episode receipts; the observed V27 aggregate is
    # 190,573,875 bytes. Keep its bound separate from individual JSON evidence.
    result = _read(root, binding['result_document'], maximum_result_json_bytes)
    if (result.get('schema_version') != 'native_task_arena_policy_canary_session_result.v1'
            or result.get('run_kind') != 'internal_policy_canary'
            or result.get('claim_ceiling') != 'diagnostic_policy_execution'
            or result.get('result_digest') != canonical_digest(result, digest_field='result_digest')):
        raise ProviderOutputInventoryError('provider_output_native_result_digest_invalid')
    base = PurePosixPath(binding['result_document']).parent
    inventory = result.get('artifact_inventory')
    if (not isinstance(inventory, list)
            or result.get('artifact_inventory_digest') != canonical_digest({'value': inventory})):
        raise ProviderOutputInventoryError('provider_output_native_inventory_digest_invalid')
    checked, manifests = set(), []

    def verify(row, directory):
        if not isinstance(row, Mapping):
            raise ProviderOutputInventoryError('provider_output_native_artifact_invalid')
        relative = safe_member_name(str(row.get('relative_path') or ''))
        full = (directory / relative).as_posix()
        record = members.get(full)
        if (record is None or row.get('sha256', row.get('png_sha256')) != record['sha256']
                or type(row.get('size_bytes')) is not int or row['size_bytes'] != record['size_bytes']):
            raise ProviderOutputInventoryError('provider_output_native_artifact_digest_mismatch')
        checked.add(full)
        if 'frame_manifest' in str(row.get('role') or ''):
            manifests.append((full, row.get('media_root_relative')))

    for row in inventory:
        verify(row, base)
    for episode in result.get('episodes') or []:
        for row in (episode.get('evidence_artifacts') or {}).values():
            if isinstance(row, Mapping) and 'relative_path' in row:
                verify(row, base)
    for relative, media_root in manifests:
        manifest = _read(root, relative, maximum_json_bytes)
        if manifest.get('frame_manifest_digest') != canonical_digest(manifest, digest_field='frame_manifest_digest'):
            raise ProviderOutputInventoryError('provider_output_native_frame_manifest_digest_invalid')
        if media_root:
            media_base = base / safe_member_name(media_root)
        else:
            parts = PurePosixPath(relative).parts
            if 'episodes' not in parts:
                raise ProviderOutputInventoryError('provider_output_native_media_root_unbound')
            media_base = PurePosixPath(*parts[:parts.index('episodes') + 1])
        # Reuse the producer's structural validator without its whole-file hash
        # helper; every PNG byte digest is compared to our streamed member hash.
        from .episode_visual_evidence import MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION, validate_multicamera_frame_manifest
        if manifest.get('schema_version') == MULTICAMERA_FRAME_MANIFEST_SCHEMA_VERSION:
            try:
                validate_multicamera_frame_manifest(manifest, output_dir=root / media_base, verify_files=False)
            except ValueError:
                raise ProviderOutputInventoryError('provider_output_native_frame_manifest_invalid') from None

        def frames(value):
            if isinstance(value, dict):
                if 'relative_path' in value and 'size_bytes' in value and ('sha256' in value or 'png_sha256' in value):
                    verify(value, media_base)
                for nested in value.values():
                    frames(nested)
            elif isinstance(value, list):
                for nested in value:
                    frames(nested)
        frames(manifest)
    return {'identity_document_digest': identity['result_digest'], 'result_document_digest': result['result_digest'],
            'verified_native_file_count': len(checked), 'episode_qualification_performed': False,
            'image_qualification_performed': False, 'scientific_finalization_pending': True}
