"""Provider billing source identifiers shared by the reconciler and receipt readers.

Kept apart from ``provider_billing_reconciler``, whose CLI lists provider VMs,
so CPU control-plane readers can parse billing receipts without that import.
"""
from __future__ import annotations

BILLING_SOURCE_SCHEMA_VERSION = "blueprint.provider_billing_source_receipt.v1"
VAST_CHARGES_URL = "https://console.vast.ai/api/v0/charges/"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
