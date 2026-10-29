"""CPU-only truth for an observed room whose original task remains unresolved."""
import json
from pathlib import Path
import re

from .decision_evidence_contracts import canonical_digest


def valid_task_limit(value, context_digest, source_digest=None):
    return (isinstance(value, dict) and value.get('schema_version') == 'atlas_task_evidence_limitation.v1'
        and value.get('digest') == canonical_digest(value, digest_field='digest')
        and value.get('status') == 'abstained' and value.get('reason') == 'atlas_scene_planner_abstained'
        and value.get('original_task_context_digest') == context_digest
        and isinstance(value.get('source_video_digest'), str)
        and re.fullmatch(r'sha256:[a-f0-9]{64}', value['source_video_digest']) is not None
        and (source_digest is None or value['source_video_digest'] == source_digest)
        and isinstance(value.get('planner_receipt_digest'), str)
        and re.fullmatch(r'sha256:[a-f0-9]{64}', value['planner_receipt_digest']) is not None)


def read_room_limit(capture_root, context):
    """Only a privately archived source-bound room is a visual-ready outcome."""
    root = Path(capture_root) / 'pipeline/atlas'
    try:
        def read(name):
            path = root / name
            if path.is_symlink():
                raise ValueError('observed_room_path_unsafe')
            return json.loads(path.read_text())
        job, result = read('preview_job.json'), read('result.json')
        limit = result.get('task_evidence_limitation')
        archive, download = job.get('archive') or {}, result.get('download') or {}
        if (job.get('state') != 'ready' or result.get('observed_scene_only') is not True
                or result.get('digest') != canonical_digest(result, digest_field='digest')
                or result['digest'] != job.get('result_digest')
                or result.get('task_context_digest') != context.get('context_digest')
                or result.get('task_and_surrounding_scene_covered') is not False
                or result.get('simulation_qualified') is not False or result.get('physical_evidence') is not False
                or result.get('native_preparation_eligible') is not False
                or result.get('execution_authority_granted') is not False
                or not isinstance(result.get('source_video_digest'), str)
                or re.fullmatch(r'sha256:[a-f0-9]{64}', result['source_video_digest']) is None
                or not valid_task_limit(limit, context['context_digest'], result.get('source_video_digest'))
                or not archive.get('uri') or not archive.get('remote_verified_at')
                or not isinstance(archive.get('digest'), str)
                or re.fullmatch(r'sha256:[a-f0-9]{64}', archive['digest']) is None
                or archive.get('digest') != download.get('zip_sha256')):
            return None
        return {'source_result_digest': result['digest'], 'source_video_digest': result['source_video_digest'],
                'task_evidence_limitation': limit, 'archive_digest': archive['digest']}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def retains_observed_room_limit(handoff, context):
    value = handoff.get('observed_room') if isinstance(handoff, dict) else None
    return (isinstance(value, dict) and handoff.get('schema_version') == 'website_scene_handoff.v1'
        and handoff.get('digest') == canonical_digest(handoff, digest_field='digest')
        and all(handoff.get(key) == context.get(key) for key in ('scene_id', 'capture_id'))
        and handoff.get('status') == 'needs_input' and handoff.get('simulator_ready') is False
        and handoff.get('visual_reconstruction_ready') is True
        and handoff.get('provider_mutation_performed') is False
        and isinstance(value.get('source_result_digest'), str)
        and re.fullmatch(r'sha256:[a-f0-9]{64}', value['source_result_digest']) is not None
        and isinstance(value.get('archive_digest'), str)
        and re.fullmatch(r'sha256:[a-f0-9]{64}', value['archive_digest']) is not None
        and isinstance(value.get('source_video_digest'), str)
        and re.fullmatch(r'sha256:[a-f0-9]{64}', value['source_video_digest']) is not None
        and valid_task_limit(value.get('task_evidence_limitation'), context.get('context_digest'), value.get('source_video_digest')))


