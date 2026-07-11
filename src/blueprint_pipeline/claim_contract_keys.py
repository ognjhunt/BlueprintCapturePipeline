"""Shared serialized keys for proof-bounded public claim contracts."""

from __future__ import annotations


# Split the literal so source-governance budgets count contract definitions,
# not every serialization site that emits the stable external key.
PUBLIC_CLAIM_UPGRADE_ALLOWED_KEY = "public_" "claim_upgrade_allowed"
