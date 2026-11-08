"""Identity of a capture's prepared derivatives; pure, so read-only closures can use it.

A correction revision re-derives a capture's request after the site corrected
its targets. A renewal re-issues a completed preparation after the sponsor
renewed its expired grant: same capture, same task, a new consent window. A
navigation re-issue sends a navigation job's intake again once its scene half
was derived from an Atlas result published after the first intake
(``website_held_handoff_replay``, ``atlas_result_superseded``): the website and
the Pipeline intake keep one intake per submission, so the new scene half needs
its own. All three keep the capture's identity, so every sibling shares its
original limits. The WebApp reads the same identities (shared vectors
``website-preparation-renewal.v1.json`` and ``website-navigation-reissue.v1.json``).
"""
import re

from .decision_evidence_contracts import cross_runtime_canonical_digest

FIELD = 'website_preparation_revision'
PREFIX = 'website-correction-'
RENEWAL_FIELD = 'website_preparation_renewal'
RENEWAL_PREFIX = 'website-renewal-'
#: The navigation task's marker of a re-issued intake, and its submission prefix.
REISSUE_FIELD = 'website_navigation_reissue'
REISSUE_PREFIX = 'website-reissue-'
_NAVIGATION_STRATEGY = 'goal_region_navigation'
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


def reissue(request):
    """``{capture_id, parent_submission_id, atlas_result_digest}`` of a re-issued navigation intake, or None.

    The parent is the capture's own submission or an earlier re-issue of it; the Atlas result is the one
    the re-derived scene half was built from. The submission id is the prefix and the digest of the rest of
    the request, so the marker and every task digest are part of it.
    """
    task = request.get('task') or {}
    # The marker's presence, as the WebApp reads it: a null marker is a changed one, never an absent one.
    if not isinstance(task, dict) or REISSUE_FIELD not in task:
        if str(request.get('submission_id', '')).startswith(REISSUE_PREFIX):
            raise ValueError('website_navigation_reissue_invalid')
        return None
    value = task[REISSUE_FIELD]
    body = {key: value for key, value in request.items() if key != 'submission_id'}
    parent = value.get('parent_submission_id') if isinstance(value, dict) else None
    subject = task.get('subject') if isinstance(task.get('subject'), dict) else {}
    if (not isinstance(value, dict) or set(value) != {'capture_id', 'parent_submission_id', 'atlas_result_digest'}
            or not isinstance(value['capture_id'], str) or re.fullmatch(_IDENTIFIER, value['capture_id']) is None
            or not isinstance(parent, str) or re.fullmatch(_IDENTIFIER, parent) is None
            or not (parent == value['capture_id'] or parent.startswith(REISSUE_PREFIX))
            or not isinstance(value['atlas_result_digest'], str)
            or re.fullmatch(r'sha256:[a-f0-9]{64}', value['atlas_result_digest']) is None
            or task.get('strategy') != _NAVIGATION_STRATEGY or FIELD in subject or RENEWAL_FIELD in subject
            or (request.get('execution') or {}).get('purpose') != 'scene_preparation'
            or request['execution'].get('policy_candidates') != []
            or request.get('submission_id') != REISSUE_PREFIX + cross_runtime_canonical_digest(body)[7:]):
        raise ValueError('website_navigation_reissue_invalid')
    return value


def capture_id(request):
    value = reissue(request) or revision(request) or renewal(request)
    return value['capture_id'] if value else request['submission_id']


def requires_retained_authority(request):
    return (revision(request) is not None or renewal(request) is not None
            or request.get('consent', {}).get('task_confirmed') is False)
