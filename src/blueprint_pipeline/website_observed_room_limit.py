"""Leased publication of a task the video cannot serve, beside its archived observed room.

Only a final scene-review abstention reaches here (``observed_room_contracts.read_room_limit``):
no frame shows the task object or its region, or no frame is usable. The same video gives the
same answer, so the preparation ends as a hold that asks the site for a new video, naming what
the video lacked (``website_recapture_required``). It is never a wait on Blueprint.
"""
import json
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .observed_room_contracts import read_room_limit


def retain_observed_room_limit(*, capture_root, task_context, source_video):
    from .core.handoff_lease import guarded_handoff_write_json
    from .local_reconstruction_adapters import _sha256_file
    from .website_recapture_required import ABSTENTION_CODES, FIELD, record
    from .observed_room_contracts import read_final_abstention
    value = read_room_limit(capture_root, task_context) or read_final_abstention(capture_root, task_context)
    if value is None:
        return None
    if _sha256_file(Path(source_video)) != value['source_video_digest']:
        raise ValueError('observed_room_source_video_changed')
    code = next(ABSTENTION_CODES[reason] for reason in ABSTENTION_CODES if reason in value['abstention_reasons'])
    result = {'schema_version': 'website_scene_handoff.v1', 'scene_id': task_context['scene_id'],
        'capture_id': task_context['capture_id'], 'status': 'needs_input', 'blockers': [code],
        FIELD: record(code, task_context_digest=task_context['context_digest']),
        # The room the video did show stays archived; it is evidence, never the job's scene.
        'observed_room_evidence': value, 'simulator_ready': False, 'visual_reconstruction_ready': True,
        'provider_mutation_performed': False, 'claim_ceiling': 'development_only'}
    result['digest'] = canonical_digest(result, digest_field='digest')
    root = Path(capture_root) / 'pipeline/website_scene_preparation'
    retained = root / 'observed_room_limits' / result['digest'][7:] / 'handoff.json'
    if retained.is_file():
        if retained.is_symlink() or json.loads(retained.read_text()) != result:
            raise ValueError('observed_room_limit_changed')
    else:
        guarded_handoff_write_json(retained, result)
    guarded_handoff_write_json(root / 'handoff.json', result)
    return result
