"""A bounded, credential-free description of the failure behind a fixed code.

A fixed reason code says which boundary stopped a job; callers and the WebApp
match on it, so it never changes. On its own it hides why. These helpers keep
the exception's type and a redacted, trimmed message beside the code, in
operator records and logs. They never belong in a payload a site owner sees.
"""

from __future__ import annotations

import errno
import re
from collections.abc import Iterator

ERROR_TEXT_LIMIT = 200
_ERROR_SCAN_LIMIT = 4096
_ERROR_TYPE = re.compile(r'[A-Za-z_][A-Za-z0-9_.<>]{0,199}')
# Applied in order. A URL keeps its scheme, host and path; its query or
# fragment can sign or authorize it. A credential key may be quoted, as in
# JSON or a printed dictionary, and its quotes escaped when that JSON is
# itself a string. The last rule removes runs that mix upper case, lower
# case and digits, as keys and tokens do, while hex digests, generations
# and capture identifiers stay readable.
_ERROR_REDACTIONS = (
    (re.compile(r'(?i)\b([a-z][a-z0-9+.-]*://[^\s?#"\'<>]*)[?#][^\s"\'<>]+'), r'\1?<redacted>'),
    (re.compile(r'(?is)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)'),
     '<redacted>'),
    (re.compile(r'(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]+'), r'\1 <redacted>'),
    (re.compile(r'(?i)\b((?:[a-z0-9]+[_-])*(?:token|key|secret|password|passwd|signature|sig'
                r'|credentials?|authorization|cookie|session)\\?["\']?)(\s*[:=]\s*)'
                r'("[^"]*"|\'[^\']*\'|[^\s,;&)}]+)'),
     r'\1\2<redacted>'),
    (re.compile(r'[?&][^\s=&?]+=[^\s&]+'), '?<redacted>'),
    (re.compile(r'\b(?:AIza[0-9A-Za-z_-]{20,}|ya29\.[0-9A-Za-z._-]+|eyJ[0-9A-Za-z_-]+\.[0-9A-Za-z._-]+'
                r'|(?:sk|pk|rk|ghp|gho|ghs|ghu|ghr|glpat|xox[abprs])[-_][0-9A-Za-z_-]{8,})'), '<redacted>'),
    (re.compile(r'(?<![A-Za-z0-9_+/=-])(?=[A-Za-z0-9_+/=-]*[a-z])(?=[A-Za-z0-9_+/=-]*[A-Z])'
                r'(?=[A-Za-z0-9_+/=-]*[0-9])[A-Za-z0-9_+/=-]{24,}'), '<redacted>'),
)
_CHAIN_LIMIT = 8
_DISK_FULL = frozenset({errno.ENOSPC, errno.EDQUOT})
_HTTP_429_CODE = re.compile(r'(?:^|[^a-z0-9])http_429(?![0-9])')


def bounded_error_text(value) -> str:
    """Redact credential-like text, then trim to ``ERROR_TEXT_LIMIT``; idempotent."""
    text = str(value)
    if len(text) > _ERROR_SCAN_LIMIT:
        # Never keep the fragment of a word the scan limit cut through.
        text = text[:_ERROR_SCAN_LIMIT].rsplit(None, 1)[0]
    text = ' '.join(re.sub(r'[\x00-\x1f\x7f]', ' ', text).split())
    for pattern, replacement in _ERROR_REDACTIONS:
        text = pattern.sub(replacement, text)
    # Redaction precedes trimming, and both are idempotent, so a repeated
    # bound returns the same text.
    return text if len(text) <= ERROR_TEXT_LIMIT else text[:ERROR_TEXT_LIMIT - 3] + '...'


def error_description(error: BaseException) -> dict:
    """``{'type': module.qualname, 'message': redacted and trimmed text}`` of one failure."""
    kind = type(error)
    name = f'{kind.__module__}.{kind.__qualname__}'
    try:
        message = bounded_error_text(error)
    except Exception:  # noqa: BLE001 - an unprintable error still has a type
        message = ''
    return {'type': name if _ERROR_TYPE.fullmatch(name) else 'unrepresentable', 'message': message}


def cause_description(error: BaseException) -> dict | None:
    """The description of the failure a summary code was raised from, if any."""
    cause = error.__cause__
    return None if cause is None else error_description(cause)


def bounded_error(value) -> dict | None:
    """Re-bound a retained description before another result repeats it; None if malformed."""
    if (type(value) is not dict or set(value) != {'type', 'message'}
            or type(value['type']) is not str or not _ERROR_TYPE.fullmatch(value['type'])
            or type(value['message']) is not str):
        return None
    return {'type': value['type'], 'message': bounded_error_text(value['message'])}


def error_chain(error: BaseException) -> Iterator[BaseException]:
    """The error and what it was raised from, in the order a traceback prints them."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(seen) < _CHAIN_LIMIT:
        seen.add(id(current))
        yield current
        current = (current.__cause__ if current.__cause__ is not None
                   else None if current.__suppress_context__ else current.__context__)


def disk_capacity_refusal(error: BaseException) -> bool:
    """Whether the volume refused a write because it is full or over quota."""
    return any(isinstance(item, OSError) and item.errno in _DISK_FULL for item in error_chain(error))


def rate_limited(error: BaseException) -> bool:
    """Whether a provider or the WebApp answered 429 (rate limit or exhausted quota)."""
    for item in error_chain(error):
        try:
            if any(type(value) is not bool and isinstance(value, int) and value == 429
                   for value in (getattr(item, field, None) for field in ('code', 'status', 'status_code'))):
                return True
            if _HTTP_429_CODE.search(str(item)):
                return True
        except Exception:  # noqa: BLE001 - an unreadable attribute or text proves nothing
            continue
    return False
