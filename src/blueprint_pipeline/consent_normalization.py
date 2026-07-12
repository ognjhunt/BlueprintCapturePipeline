"""Shared fail-closed normalization for capture-rights and consent values.

Every surface that reads consent/rights values (materialization, qualification,
rights-provenance review, the consent-takedown delivery gate, PTDP packaging,
buyer readout) must agree on how malformed, missing, or contradictory values
resolve. The contract is monotone toward denial:

- An allow-flag grants only on an explicit boolean/text ``true``; any other
  type or token (including ``"false"``, ``"no"``, lists, dicts) is a deny.
- A revocation signal fires on any recognizable revocation expression, in
  either snake_case or camelCase, nested or top-level; contradictions between
  duplicate spellings resolve to the revoked/blocked side.
- A consent status is "active" only when every observed status token is in the
  explicit active allow-list. Unknown, wrong-typed, or contradictory statuses
  resolve to ``"unknown"`` (blocked), never to ``"active"``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .common import parse_bool

CONSENT_REVOKED_STATUSES = frozenset({"revoked", "withdrawn", "rescinded"})
# Statuses that may count as a positive, live consent grant. Packaging surfaces
# may apply a stricter subset (e.g. PTDP excludes policy_only), but nothing may
# treat a token outside this set as active.
CONSENT_ACTIVE_STATUSES = frozenset(
    {"active", "approved", "documented", "granted", "policy_only"}
)

_NESTED_RIGHTS_KEYS = ("capture_rights", "rights_consent", "rights")
_STATUS_KEYS = ("consent_status", "consentStatus")
_REVOKED_FLAG_KEYS = ("consent_revoked", "consentRevoked")
_REVOKED_AT_KEYS = ("consent_revoked_at", "consentRevokedAt")

# Sentinel status emitted when duplicate spellings disagree; it is outside both
# the active and revoked sets, so every consumer treats it as blocked/unknown.
CONTRADICTORY_CONSENT_STATUS = "contradictory"


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_consent_status_token(value: Any) -> str | None:
    """Lowercased status token, or None when absent/wrong-typed.

    A present-but-non-string status is malformed and maps to ``""`` (blocked),
    never to a valid token; absence maps to None so callers can distinguish
    "no consent field" from "malformed consent field".
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def strict_allow_bool(value: Any) -> bool:
    """Fail-closed allow-flag: True only on an explicit boolean/text true.

    ``"false"``, ``"no"``, ``"denied"``, lists, dicts, and every unrecognized
    token deny. This is the only permitted coercion for permission-granting
    flags such as ``derived_scene_generation_allowed``.
    """
    if isinstance(value, (list, tuple, set, dict)):
        return False
    return parse_bool(value, default=False)


def revocation_signal(value: Any) -> bool | None:
    """Tri-state revocation flag, lenient toward revocation.

    Returns True for any recognizable revocation expression (bool True,
    nonzero number, truthy token, or a revoked-status word), False for an
    explicit non-revoked expression, and None when the value is absent or
    unintelligible. Unintelligible values never mean "not revoked" on their
    own — callers must combine this with the status/timestamp signals.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value != value:  # NaN: malformed, fail toward revocation
            return True
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"} | CONSENT_REVOKED_STATUSES:
            return True
        if token in {"0", "false", "no", "n", "off", "active", "documented"}:
            return False
        return None
    # Wrong-typed containers on a revocation flag are malformed: fail closed
    # toward revocation rather than silently reading as "not revoked".
    return True


def _revoked_at_text(value: Any) -> str | None:
    """Revocation timestamp text; any non-string truthy value is a signal.

    A wrong-typed but present ``consent_revoked_at`` still indicates that a
    revocation was recorded, so it returns a sentinel rather than None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return "malformed_consent_revoked_at"


def resolve_consent_signals(payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Resolve one payload's consent signals most-restrictively.

    Scans the top level plus every nested rights block (``capture_rights``,
    ``rights_consent``, ``rights``) and both key spellings, so a revocation can
    never be shadowed by nesting or naming. Contradictory status tokens
    resolve to ``"contradictory"`` (blocked); statuses outside the active
    allow-list resolve the state to ``"unknown"`` (blocked), never "active".
    """
    root = _mapping(payload)
    sources: List[Dict[str, Any]] = [root]
    for key in _NESTED_RIGHTS_KEYS:
        nested = _mapping(root.get(key))
        if nested:
            sources.append(nested)

    status_tokens: List[str] = []
    status_present = False
    revoked = False
    revoked_at: str | None = None
    malformed: List[str] = []

    for source in sources:
        for key in _STATUS_KEYS:
            value = source.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                # A blank status is an absent status, not a malformed one.
                continue
            status_present = True
            token = normalize_consent_status_token(value)
            if token == "":
                malformed.append(f"consent_status_malformed:{key}")
            elif token is not None and token not in status_tokens:
                status_tokens.append(token)
        for key in _REVOKED_FLAG_KEYS:
            if key in source:
                signal = revocation_signal(source.get(key))
                if signal is True:
                    revoked = True
                elif signal is None and source.get(key) is not None:
                    malformed.append(f"consent_revoked_malformed:{key}")
        for key in _REVOKED_AT_KEYS:
            if key in source:
                text = _revoked_at_text(source.get(key))
                if text:
                    revoked = True
                    if revoked_at is None:
                        revoked_at = text

    if any(token in CONSENT_REVOKED_STATUSES for token in status_tokens):
        revoked = True

    if not status_tokens:
        consent_status: str | None = None
    elif len(status_tokens) == 1:
        consent_status = status_tokens[0]
    else:
        revoked_tokens = [
            token for token in status_tokens if token in CONSENT_REVOKED_STATUSES
        ]
        consent_status = (
            revoked_tokens[0] if revoked_tokens else CONTRADICTORY_CONSENT_STATUS
        )

    has_consent_fields = bool(
        status_present
        or revoked
        or malformed
        or any(
            key in source
            for source in sources
            for key in (*_REVOKED_FLAG_KEYS, *_REVOKED_AT_KEYS)
        )
    )

    if revoked:
        state = "revoked"
    elif (
        consent_status in CONSENT_ACTIVE_STATUSES
        and not malformed
        and len(status_tokens) == 1
    ):
        state = "active"
    else:
        state = "unknown"

    return {
        "state": state,
        "consent_status": consent_status,
        "consent_revoked": revoked,
        "consent_revoked_at": revoked_at,
        "has_consent_fields": has_consent_fields,
        "malformed_fields": sorted(set(malformed)),
    }


def restrictive_scope_list(*values: Any) -> List[str]:
    """Most-restrictive merge of duplicate consent-scope spellings.

    Each value is a scope grant list; when more than one non-empty list is
    present (e.g. ``consent_scope`` and ``consentScope`` disagree) the result
    is their intersection — a scope is granted only if every spelling grants
    it. Non-list scalars are malformed and grant nothing.
    """
    normalized: List[List[str]] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            items = [item for item in value if isinstance(item, str)]
        else:
            # Malformed scope container: present but grants nothing.
            normalized.append([])
            continue
        tokens = [item.strip().lower() for item in items if item.strip()]
        normalized.append(tokens)
    if not normalized:
        return []
    result = [token for token in normalized[0]]
    for tokens in normalized[1:]:
        token_set = set(tokens)
        result = [token for token in result if token in token_set]
    return sorted(set(result))
