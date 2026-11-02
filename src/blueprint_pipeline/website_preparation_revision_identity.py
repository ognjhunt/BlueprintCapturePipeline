"""Identity of a capture's prepared derivatives; pure, so read-only closures can use it.

A correction revision re-derives a capture's request after the site corrected
its targets. A renewal re-issues a completed preparation after the sponsor
renewed its expired grant: same capture, same task, a new consent window. Both
keep the capture's identity, so every sibling shares its original native limits.
"""
import re

from .decision_evidence_contracts import cross_runtime_canonical_digest

FIELD = 'website_preparation_revision'
PREFIX = 'website-correction-'
RENEWAL_FIELD = 'website_preparation_renewal'
RENEWAL_PREFIX = 'website-renewal-'
_IDENTIFIER = r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}'


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


def renewal(request):
    """``{capture_id, parent_submission_id}`` of a request re-issued under a renewed grant, or None."""
    value = (request.get('task', {}).get('subject') or {}).get(RENEWAL_FIELD)
    if value is None:
        if str(request.get('submission_id', '')).startswith(RENEWAL_PREFIX):
            raise ValueError('website_preparation_renewal_invalid')
        return None
    body = {key: value for key, value in request.items() if key != 'submission_id'}
    parent = value.get('parent_submission_id') if isinstance(value, dict) else None
    if (not isinstance(value, dict) or set(value) != {'capture_id', 'parent_submission_id'}
            or not isinstance(value['capture_id'], str) or re.fullmatch(_IDENTIFIER, value['capture_id']) is None
            or not isinstance(parent, str) or re.fullmatch(_IDENTIFIER, parent) is None
            or not (parent == value['capture_id'] or parent.startswith(RENEWAL_PREFIX))
            or FIELD in (request['task'].get('subject') or {})
            or request.get('execution', {}).get('purpose') != 'scene_preparation'
            or request['execution'].get('policy_candidates') != []
            or request.get('submission_id') != RENEWAL_PREFIX + cross_runtime_canonical_digest(body)[7:]):
        raise ValueError('website_preparation_renewal_invalid')
    return value


def capture_id(request):
    value = revision(request) or renewal(request)
    return value['capture_id'] if value else request['submission_id']


def requires_retained_authority(request):
    return (revision(request) is not None or renewal(request) is not None
            or request.get('consent', {}).get('task_confirmed') is False)
