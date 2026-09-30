"""Validate recorded restore completion without creating execution authority.

ADP-009D/day28. Snapshot, new member births, durable final and owner access must
agree. The caller authenticates the protected chain and performs fresh readback.
"""
from __future__ import annotations

import stat

from . import control_plane_lane_historical_generation as generation
from .control_plane_lane_historical_restore_snapshot import validate_private, after_reopen


def validate_restore_final(selected, events, snapshot):
    """Pure facts only; never interpret a completed label as a physical receipt."""
    _, decision, manifest, _ = selected
    finals = [event for event in events if event['kind'] == 'restore_final']
    generation._require(len(finals) == 1, 'restore_incomplete')
    final, receipt = finals[0], finals[0]['body']
    files = [row for row in manifest['members'] if row['kind'] == 'file']
    generation._require(receipt.get('status') == 'completed' and receipt.get('action') == 'restore'
        and receipt.get('action_id') == decision['action_id'] and receipt.get('owner') == decision['owner']
        and receipt.get('generation_digest') == manifest['generation_digest']
        and receipt.get('original_manifest') == decision['manifest']
        and receipt.get('original_final_event_digest') == decision['final_event_digest']
        and receipt.get('archive_sha256') == decision['archive']['sha256']
        and receipt.get('archive_size_bytes') == decision['archive']['size_bytes']
        and type(receipt.get('restored_files')) is int and receipt['restored_files'] == len(files)
        and type(receipt.get('restored_logical_bytes')) is int
        and receipt['restored_logical_bytes'] == sum(row['size_bytes'] for row in files)
        and receipt.get('fresh_disk_reservation') is True
        and receipt.get('root_directory_retained') is True
        and receipt.get('owner_access_reopened') is False
        and all(decision['issued_at_epoch'] <= event['observed_at_epoch'] < decision['expires_at_epoch']
                for event in events), 'restore_final_invalid')
    validate_private(manifest, snapshot)
    generation._require(snapshot['members'][0]['version'] == receipt.get('protected_root_version'),
                        'restore_snapshot_changed')
    members = [event for event in events if event['kind'] == 'restore_member']
    by_path = {event['body']['path']: event['body'] for event in members}
    generation._require(len(members) == len(files)
        and set(by_path) == {row['path'] for row in files}
        and all(event['sequence'] < final['sequence'] for event in members)
        and all(by_path[row['path']]['sha256'] == row['sha256']
                    and by_path[row['path']]['size_bytes'] == row['size_bytes'] for row in files),
        'restore_final_invalid')
    generation._require(snapshot['root_version'] == decision['parent_version']
        and all(row['version'][:2] == by_path[row['path']]['version'][:2]
                for row in snapshot['members'] if row['kind'] == 'file'), 'restore_snapshot_changed')
    return final


def validate_restored_receipt(selected, events, snapshot):
    """Return the exact expected current tree after durable owner access."""
    final = validate_restore_final(selected, events, snapshot)
    accesses = [event for event in events if event['kind'] == 'access_reopened']
    generation._require(len(accesses) == 1 and accesses[0] == events[-1], 'restore_incomplete')
    access = accesses[0]
    generation._require(final['sequence'] < access['sequence']
        and access['body'].get('phase') == 'owner_rights_observed'
        and access['body'].get('path') == '', 'restore_final_invalid')
    return final, after_reopen(selected[2], snapshot, access['body'].get('version'))


def validate_pending_owner_access(selected, events, snapshot, final):
    """Only exact root permission intents after the final; no physical inference."""
    generation._require(not any(event['kind'] == 'access_reopened' for event in events),
                        'restore_incomplete')
    owner = selected[2]['members'][0]['version']
    expected = dict(phase='owner_rights', path='', version=snapshot['members'][0]['version'],
                    uid=owner[3], gid=owner[4], mode=stat.S_IMODE(owner[2]))
    pending = [event for event in events if event['sequence'] > final['sequence']]
    generation._require(all(event['kind'] == 'restore_intent' and event['body'] == expected
                            for event in pending), 'restore_incomplete')
