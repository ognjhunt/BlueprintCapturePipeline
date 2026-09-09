"""Owner-approved, single-use visual-review correction; GPU retries stay zero."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_unstarted_controls_reservations import _read

SCHEMA = 'task_evaluation_visual_review_correction_authority.v1'
DIRECTORY = 'visual-review-corrections'


def validate(*, directory: Path, value: Mapping[str, Any], admission: bool = False, now: float | None = None) -> dict[str, Any]:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_visual_review_continuation import require
    require(intake._identifier(value.get('source_attempt_id')), 'source_attempt_id_invalid')
    require(intake._DIGEST.fullmatch(str(value.get('source_placement_receipt_digest'))) is not None,
            'source_receipt_digest_invalid')
    owner = intake._read(directory/'intent.json','intent_digest')
    source = intake._read(directory/'attempts'/(str(value.get('source_attempt_id'))+'.json'),'attempt_digest')
    require(value.get('schema_version') == SCHEMA
        and value.get('authority_digest') == canonical_digest(value,digest_field='authority_digest')
        and value.get('owner_intent_digest') == owner['intent_digest']
        and value.get('owner') == owner['request']['owner']
        and value.get('scope') == 'placement_visual_review_only'
        and value.get('maximum_calls') == 1 and value.get('maximum_cost_usd') == .15
        and value.get('native_phase_hard_cap_usd') == .40
        and value.get('gpu_retries') == 0 and value.get('maximum_total_spend_usd') == owner['request']['execution']['max_total_spend_usd']
        and value.get('source_attempt_digest') == source['attempt_digest'] and source['provider'] == 'openai'
        and bool(str(value.get('authorization_reference') or '').strip())
        and value.get('provider_mutation_performed') is False, 'owner_correction_authority_invalid')
    if admission:
        moment=time.time() if now is None else now
        require(not (directory/'revoked.json').exists() and moment < intake.effective_execution_expiry(directory,owner)
            and moment < value['expires_at_epoch'], 'owner_correction_authority_expired_or_revoked')
    return dict(value)


def read_authority(*, directory: Path, source_attempt_id: str, admission: bool = False,
                   now: float | None = None) -> dict[str, Any] | None:
    path=directory/DIRECTORY/(source_attempt_id+'.json')
    if not path.exists() and not path.is_symlink():
        return None
    return validate(directory=directory,value=_read(path),admission=admission,now=now)


def authorize(*, scene_root: str | Path, intent_id: str, source_attempt_id: str,
              source_placement_receipt_digest: str, authorization_reference: str,
              now: float | None = None) -> dict[str, Any]:
    """Call only with an explicit owner approval for this bounded correction."""
    from . import task_evaluation_scene_intake as intake
    directory=Path(scene_root)/intent_id
    with intake._lock(Path(scene_root)):
        owner=intake._read(directory/'intent.json','intent_digest')
        source=intake._read(directory/'attempts'/(source_attempt_id+'.json'),'attempt_digest')
        value={'schema_version':SCHEMA,'scope':'placement_visual_review_only','owner':owner['request']['owner'],
            'owner_intent_digest':owner['intent_digest'],'source_attempt_id':source_attempt_id,
            'source_attempt_digest':source['attempt_digest'],'source_placement_receipt_digest':source_placement_receipt_digest,
            'maximum_calls':1,'maximum_cost_usd':.15,'native_phase_hard_cap_usd':.40,'gpu_retries':0,
            'maximum_total_spend_usd':owner['request']['execution']['max_total_spend_usd'],
            'authorization_reference':authorization_reference,'issued_at_epoch':time.time() if now is None else now,
            'expires_at_epoch':intake.effective_execution_expiry(directory,owner),'provider_mutation_performed':False}
        value['authority_digest']=canonical_digest(value,digest_field='authority_digest')
        validate(directory=directory,value=value,admission=True,now=now)
        target=directory/DIRECTORY/(source_attempt_id+'.json')
        target.parent.mkdir(mode=0o750,exist_ok=True)
        intake.write_exclusive(target,value)
        return {'status':'authorized_one_visual_review_correction','path':str(target),**value}
