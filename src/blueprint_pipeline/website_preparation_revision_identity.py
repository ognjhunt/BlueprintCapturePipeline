"""Identity of a corrected capture derivative; pure, so read-only closures can use it."""
import re

from .decision_evidence_contracts import cross_runtime_canonical_digest

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
