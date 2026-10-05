"""Pure validation of retained paid-provider-zero evidence; no spend authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .vast_evidence_contracts import valid_vast_provider_zero_api_call

ADP_PAID_PROVIDER_ZERO_SCHEMA_VERSION = "adp_paid_provider_zero.v1"


def valid_adp_paid_provider_zero(value: Mapping[str, Any]) -> bool:
    """Validate the canonical Vast-only provider-zero receipt."""

    return bool(
        value.get("schema_version") == ADP_PAID_PROVIDER_ZERO_SCHEMA_VERSION
        and value.get("provider") == "vast"
        and value.get("api_confirmed") is True
        and value.get("provider_zero") is True
        and value.get("global_live_resource_count") == 0
        and value.get("inventory") == []
        and valid_vast_provider_zero_api_call(value.get("api_command"))
        and isinstance(value.get("stderr_present"), bool)
        and value.get("raw_secret_values_recorded") is False
        and value.get("provider_zero_digest")
        == canonical_digest(value, digest_field="provider_zero_digest")
    )
