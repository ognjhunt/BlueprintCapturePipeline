"""Immutable corrected derivatives of one capture; never a fresh spending grant."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import re

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

FIELD = 'website_preparation_revision'
PREFIX = 'website-correction-'


def revision(request):
    value = (request.get('task', {}).get('subject') or {}).get(FIELD)
    if value is None:
        if str(request.get('submission_id', '')).startswith(PREFIX):
            raise ValueError('website_preparation_revision_invalid')
        return None
    body = {key: value for key, value in request.items() if key != 'submission_id'}
    if (not isinstance(value, dict) or set(value) != {'capture_id', 'corrections_digest'}
            or not isinstance(value['capture_id'], str)
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', value['capture_id']) is None
            or re.fullmatch(r'sha256:[a-f0-9]{64}', str(value['corrections_digest'])) is None
            or request.get('execution', {}).get('purpose') != 'scene_preparation'
            or request['execution'].get('policy_candidates') != []
            or request.get('submission_id') != PREFIX + cross_runtime_canonical_digest(body)[7:]):
        raise ValueError('website_preparation_revision_invalid')
    return value


def capture_id(request):
    value = revision(request)
    return value['capture_id'] if value else request['submission_id']


def requires_retained_authority(request):
    return revision(request) is not None or request.get('consent', {}).get('task_confirmed') is False


def workspace(capture_root):
    from .website_task_target_choice import correction_binding_digest
    root = Path(capture_root) / 'pipeline' / 'website_scene_preparation'
    digest = correction_binding_digest(Path(capture_root) / 'pipeline')
    return root / 'revisions' / digest[7:] if digest else root


def bind_preparation(preparation, *, output_root, capture_root, authority=None):
    """Called before runtime compilation, so every reference binds the same revision."""
    if Path(output_root) == Path(capture_root) / 'pipeline' / 'website_scene_preparation':
        return preparation
    from .website_task_target_choice import correction_binding_digest
    from .core.handoff_lease import guarded_handoff_write_json
    digest = correction_binding_digest(Path(capture_root) / 'pipeline')
    if digest is None or Path(output_root) != workspace(capture_root):
        raise ValueError('website_preparation_revision_workspace_changed')
    value = deepcopy(preparation)
    if authority is not None:
        value['website_preparation_authority'] = deepcopy(authority)
    request = value.get('intake_request')
    if request is None or preparation.get('status') != 'intake_ready':
        return preparation
    request['task']['subject'][FIELD] = {'capture_id': request['submission_id'], 'corrections_digest': digest}
    request['submission_id'] = PREFIX + cross_runtime_canonical_digest(
        {key: item for key, item in request.items() if key != 'submission_id'})[7:]
    revision(request)
    value['digest'] = canonical_digest(value, digest_field='digest')
    guarded_handoff_write_json(Path(output_root) / 'preparation.json', value)
    return value


def capture_root_for_preparation(path):
    path = Path(path)
    parent = path.parent
    if (parent.parent.name == 'revisions' and re.fullmatch(r'[a-f0-9]{64}', parent.name)
            and parent.parent.parent.name == 'website_scene_preparation'):
        parent = parent.parent.parent
    if parent.name == 'website_scene_preparation' and parent.parent.name == 'pipeline':
        return parent.parent.parent
    return None


def shared_reserved_budget(root, directory, intent):
    """Under intake's global lock, count other derivatives' original holds too.

    No allowance is copied, increased or refunded here. A derivative gets its
    current signed grant checked separately before it can reserve anything.
    """
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_terminal_scene_attempt_settlement import budget_retained_hold
    from .task_evaluation_retained_controls_evidence import validated_cancellation
    request = intent['request']
    if request.get('execution', {}).get('purpose') != 'scene_preparation':
        return Decimal(0), 0
    key = (request['owner'], capture_id(request))
    exposure, count = Decimal(0), 0
    for sibling in Path(root).glob('scene-*'):
        if sibling == directory:
            continue
        intake._require(not sibling.is_symlink(), 'record_unsafe')
        previous = intake._read(sibling / 'intent.json', 'intent_digest')
        other = previous['request']
        if (other.get('execution', {}).get('purpose') != 'scene_preparation'
                or (other['owner'], capture_id(other)) != key):
            continue
        attempts = sibling / 'attempts'
        intake._require(not attempts.is_symlink(), 'record_unsafe')
        for path in attempts.glob('*.json'):
            row = intake._read(path, 'attempt_digest')
            intake._require(row.get('intent_digest') == previous['intent_digest'], 'attempt_binding_invalid')
            cancellation = validated_cancellation(sibling, row)
            if cancellation is not None:
                if cancellation.get('schema_version') == intake.TERMINAL_SETTLEMENT_SCHEMA:
                    hold = budget_retained_hold(cancellation)
                    exposure += Decimal(str(hold['retained_spend_usd']))
                    count += int(hold['counts_as_attempt'])
                continue
            exposure += Decimal(str(row['maximum_spend_usd']))
            if not intake._dependent_row_id(row['attempt_id']) and not row.get('visual_review_correction'):
                count += 1
    return exposure, count
