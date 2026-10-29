"""Check the actual Atlas key's project against signed owned-source authority.

The worker supplies the authenticated noncreating WebApp configuration. This
module never learns expected authority from a key, creates a grant, or writes
provider metadata or credentials. Project identity proves no retention setting.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import math
import re
import time
from typing import TYPE_CHECKING, Any

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .owned_capture_atlas_context import SCHEMA, validate_owned_capture_atlas_context

if TYPE_CHECKING:
    from .website_atlas_pose import AtlasClient

_ID = r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}"
_OPERATION = re.compile(rf"accounts/({_ID})/projects/({_ID})/operations/({_ID})")
_SHA = re.compile(r"sha256:[0-9a-f]{64}")
_AUTHORITY_FIELDS = {
    "schema_version", "authorization_basis", "request_id", "capture_id", "owner_user_id",
    "task_context_digest", "source_capture", "provider_identity", "source_video_digest",
    "company_authorization_digest", "original_allowance_digest", "approval_reference", "accepted_by",
    "private_processing_authorized", "provider_terms_reference", "provider_training_authorized",
    "provider_retention", "planner", "maximum_cost_usd", "maximum_requests", "expires_at_epoch",
    "claim_scope", "authorization_digest",
}


def _positive(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_owned_atlas_configuration(
    task_context: Mapping[str, Any], configuration: Mapping[str, Any], *,
    expected_owner_uid: str | None = None, now: float | None = None,
) -> dict[str, Any]:
    """Validate signed transport data, not a client self-issued grant.

    The caller must obtain configuration from the authenticated WebApp transport.
    An integrity digest alone is never authentication or human acceptance.
    """
    context = validate_owned_capture_atlas_context(task_context)
    if not isinstance(configuration, Mapping):
        raise ValueError("owned_capture_atlas_configuration_invalid")
    authority = configuration.get("authority")
    if (configuration.get("schema_version") != "owned_capture_atlas_configuration.v1"
            or configuration.get("dispatch_authorized") is not False
            or configuration.get("grant_creation_allowed") is not False
            or configuration.get("task_context_digest") != context["context_digest"]
            or configuration.get("source_video_digest") != context["source_capture"]["capture_digest"]
            or not isinstance(authority, Mapping) or set(authority) != _AUTHORITY_FIELDS):
        raise ValueError("owned_capture_atlas_configuration_invalid")
    try:
        authority_digest = cross_runtime_canonical_digest(authority, digest_field="authorization_digest")
    except (TypeError, ValueError):
        raise ValueError("owned_capture_atlas_authority_binding_invalid") from None
    if (authority.get("authorization_digest") != authority_digest
            or authority.get("schema_version") != "owned_capture_atlas_authorization.v1"
            or authority.get("authorization_basis") != "explicit_company_source_approval"
            or authority.get("request_id") != context["request_id"]
            or authority.get("capture_id") != context["capture_id"]
            or authority.get("task_context_digest") != context["context_digest"]
            or authority.get("source_capture") != context["source_capture"]
            or authority.get("source_video_digest") != context["source_capture"]["capture_digest"]
            or not _text(authority.get("owner_user_id"))
            or authority.get("accepted_by") != authority["owner_user_id"]
            or expected_owner_uid is not None and authority["owner_user_id"] != expected_owner_uid
            or authority.get("private_processing_authorized") is not True
            or authority.get("provider_training_authorized") is not False
            or authority.get("claim_scope") != "development_only"
            or not isinstance(authority.get("provider_retention"), str)
            or authority.get("provider_retention") not in {
                "account_does_not_retain", "project_library_retention_approved"}
            or not _text(authority.get("approval_reference"))
            or any(not isinstance(authority.get(key), str) or not _SHA.fullmatch(authority[key])
                   for key in ("provider_terms_reference", "company_authorization_digest", "original_allowance_digest"))
            or not _positive(authority.get("maximum_cost_usd")) or authority["maximum_cost_usd"] > 1000
            or type(authority.get("maximum_requests")) is not int
            or not 1 <= authority["maximum_requests"] <= 1000
            or not _positive(authority.get("expires_at_epoch"))):
        raise ValueError("owned_capture_atlas_authority_binding_invalid")
    if authority["expires_at_epoch"] <= (time.time() if now is None else now):
        raise ValueError("owned_capture_atlas_authority_expired")
    identity, planner = authority.get("provider_identity"), authority.get("planner")
    if (not isinstance(identity, Mapping) or set(identity) != {"account_id", "project_id"}
            or any(not isinstance(identity[key], str) or not re.fullmatch(_ID, identity[key]) for key in identity)
            or not isinstance(planner, Mapping) or set(planner) != {
                "provider", "provider_terms_reference", "private_processing_authorized",
                "provider_training_authorized", "approval_reference"}
            or planner.get("provider") != "openai" or planner.get("private_processing_authorized") is not True
            or planner.get("provider_training_authorized") is not False
            or not isinstance(planner.get("provider_terms_reference"), str)
            or not _SHA.fullmatch(planner["provider_terms_reference"])
            or not _text(planner.get("approval_reference"))):
        raise ValueError("owned_capture_atlas_authority_binding_invalid")
    return deepcopy(dict(configuration))


@dataclass(frozen=True)
class _ProjectFence:
    context_digest: str
    check: Callable[[AtlasClient], None]


_PROJECT: ContextVar[_ProjectFence | None] = ContextVar("owned_capture_atlas_project", default=None)


def owned_atlas_project_checkpoint(task_context: Mapping[str, Any]) -> Callable[[AtlasClient], None] | None:
    """Capture a matching fence now; later threads need no ContextVar lookup."""
    if task_context.get("schema_version") != SCHEMA:
        return None
    context = validate_owned_capture_atlas_context(task_context)
    fence = _PROJECT.get()
    if fence is not None and fence.context_digest == context["context_digest"]:
        return fence.check

    def missing(_client: AtlasClient) -> None:
        raise ValueError("owned_capture_atlas_project_fence_required")

    return missing


@contextmanager
def bind_owned_atlas_project(
    *, task_context: Mapping[str, Any], configuration: Mapping[str, Any],
    expected_owner_uid: str | None = None,
) -> Iterator[None]:
    """Pin expected identity from signed configuration; check the key on mutations."""
    value = validate_owned_atlas_configuration(task_context, configuration, expected_owner_uid=expected_owner_uid)
    authority = value["authority"]
    expected = (authority["provider_identity"]["account_id"], authority["provider_identity"]["project_id"])
    expires = authority["expires_at_epoch"]

    def check(client: AtlasClient) -> None:
        if time.time() >= expires:
            raise ValueError("owned_capture_atlas_authority_expired")
        try:
            evidence = client.call("GET", "/operations?origin=DEVELOPER_API&pageSize=1")
        except Exception:
            # Provider error bodies and transport exceptions can contain IDs or
            # credential-bearing URLs; expose only a bounded refusal.
            raise ValueError("owned_capture_atlas_project_identity_unavailable") from None
        operations = evidence.get("operations") if isinstance(evidence, Mapping) else None
        if not isinstance(operations, list) or not operations:
            raise ValueError("owned_capture_atlas_project_identity_unavailable")
        for operation in operations:
            name = operation.get("name") if isinstance(operation, Mapping) else None
            match = _OPERATION.fullmatch(name) if isinstance(name, str) else None
            if match is None:
                raise ValueError("owned_capture_atlas_project_identity_invalid")
            if match.groups()[:2] != expected:
                raise ValueError("owned_capture_atlas_project_identity_mismatch")
        if time.time() >= expires:
            raise ValueError("owned_capture_atlas_authority_expired")

    token = _PROJECT.set(_ProjectFence(value["task_context_digest"], check))
    try:
        yield
    finally:
        _PROJECT.reset(token)
