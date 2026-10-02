"""Retained RunPod S3 volume region contract, shared without build-packet imports.

The provider's admitted network-volume locations were recorded on 2026-07-14.
This contract does not query, create, or authorize a provider resource.
"""

RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS = frozenset(
    {
        "AP-IN-2",
        "AP-JP-1",
        "CA-MTL-3",
        "CA-MTL-4",
        "EU-CZ-1",
        "EU-FR-1",
        "EU-NL-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-IS-3",
        "EUR-NO-1",
        "EUR-NO-2",
        "US-CA-2",
        "US-IL-1",
        "US-MO-2",
        "US-NC-1",
        "US-NE-1",
        "US-TX-3",
        "US-WA-1",
    }
)

RUNPOD_S3_DATA_CENTER_IDS = frozenset(
    {
        "EU-CZ-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-NO-1",
        "US-CA-2",
        "US-GA-2",
        "US-IL-1",
        "US-KS-2",
        "US-MD-1",
        "US-MO-1",
        "US-MO-2",
        "US-NC-1",
        "US-NC-2",
        "US-NE-1",
        "US-WA-1",
    }
)

RUNPOD_S3_VOLUME_DATA_CENTER_IDS = RUNPOD_S3_DATA_CENTER_IDS & RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS
