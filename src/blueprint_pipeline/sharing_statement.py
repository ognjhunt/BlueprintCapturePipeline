"""May this capture's video be shared with approved deployment companies? The one answer.

Owner decision 2026-11-12: recording statement ``2026-11-12.v4`` shares the job, its video
included, with approved deployment companies as the service itself. The website's sharing rule is
the one authority: the Pipeline keeps no list of Blueprint's own captures (C-10). Every reader of
that permission (the company clip, the job packet tool) asks ``sharing_basis`` here, a leaf with
no Pipeline imports, so live code never imports the offline packet tool.
``capture_original_owner_observer.RECORDING_RIGHTS_STATEMENT_VERSIONS`` takes each version listed
here, so a new sharing version is added here only. A version is added only in a reviewed change.

This decides the basis only. Consent (granted, not revoked), withdrawal and the takedown gate are
the caller's checks and stop sharing whatever the basis.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SHARING_STATEMENT_VERSIONS: tuple[str, ...] = ("2026-11-12.v4",)
#: An internal test job's recording grant (WebApp ``internalTestJob.ts``): never a sharing version. With the job's
#: ``internal_test`` marker it is the ``internal_test`` basis: the clip and zip are built and published to the website,
#: which shows a test job to no company.
INTERNAL_TEST_VERSION = "internal_test.v1"


def covers_sharing(version: Any) -> bool:
    """True only for a statement version listed above, matched exactly."""
    return isinstance(version, str) and version in SHARING_STATEMENT_VERSIONS


def internal_test_label(version: Any, marker: Any) -> str | None:
    """The test job's label when the version is exactly ``internal_test.v1`` and the marker names a label and a
    footage id; else None (one without the other is no basis)."""
    if version != INTERNAL_TEST_VERSION or not isinstance(marker, Mapping):
        return None
    label, footage = marker.get("label"), marker.get("footage_id")
    return label if isinstance(label, str) and label and isinstance(footage, str) and footage else None


def sharing_basis(task_context: Mapping[str, Any]) -> dict[str, Any] | None:
    """The rule that lets this capture's video be shared, or None.

    ``{"permission_basis": "site_sharing_statement", "statement_version": ...}`` when the task
    context's ``capture_rights.statement_version`` covers sharing; else
    ``{"permission_basis": "internal_test", ...}`` for an internal test job
    (``internal_test_label``); else None.
    """
    rights = task_context.get("capture_rights") if isinstance(task_context, Mapping) else None
    version = rights.get("statement_version") if isinstance(rights, Mapping) else None
    if covers_sharing(version):
        return {"permission_basis": "site_sharing_statement", "statement_version": version}
    label = internal_test_label(version, task_context.get("internal_test"))
    if label is not None:
        return {"permission_basis": "internal_test", "statement_version": version, "label": label}
    return None
