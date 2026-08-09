"""Classify paid provider failures without relabeling them as science."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "provider_attempt_classification.v1"


def classify_provider_attempt(
    *,
    provider_command: Mapping[str, Any],
    blockers: Sequence[str],
) -> dict[str, Any]:
    """Separate pre-entrypoint provider nulls from executed bundle attempts.

    This intentionally does not authorize a retry. Existing paid-lane doctrine
    remains zero automatic retries; a fresh or explicitly amended authority is
    required before another provider mutation.
    """

    provider_started = provider_command.get("provider_bundle_started") is True
    entrypoint_started = provider_command.get("provider_entrypoint_started") is True
    output_returned = (
        provider_command.get("provider_runtime_output_zip_received") is True
    )
    pre_execution = not provider_started and not entrypoint_started and not output_returned
    typed_blockers = sorted(set(str(item) for item in blockers if str(item)))
    provider_null = pre_execution and bool(typed_blockers)
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": (
            "pre_execution_provider_null"
            if provider_null
            else "provider_bundle_attempt_started"
            if provider_started or entrypoint_started
            else "no_paid_attempt_evidence"
        ),
        "provider_bundle_started": provider_started,
        "provider_entrypoint_started": entrypoint_started,
        "provider_output_returned": output_returned,
        "scientific_attempt_consumed": provider_started or entrypoint_started,
        "pre_execution_requeue_eligible_in_principle": provider_null,
        "automatic_requeue_authorized": False,
        "automatic_requeue_executed": False,
        "maximum_automatic_requeues": 0,
        "authority_required_for_next_provider_mutation": True,
        "blockers": typed_blockers,
    }


__all__ = ["classify_provider_attempt"]
