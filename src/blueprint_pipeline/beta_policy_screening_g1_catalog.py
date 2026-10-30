"""Native G1 candidate identities and the catalog revision their registrations add.

The single implementation shared by ``beta_policy_screening_g1`` (execution and
result projection) and the effective catalog (``beta_policy_effective_catalog``).
It reads no scene, checkpoint or provider state, so the CPU intake lane can
compute the same catalog revision a screening plan binds.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .beta_policy_candidate_catalog import seal_catalog
from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import CANDIDATE_DIGESTS

CANDIDATES = tuple(CANDIDATE_DIGESTS)


def catalog_for_profile(profile: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    """A corrected sealed G1 registration replans without resetting site spend."""
    configured = profile.get("screening_candidates") or {}
    registrations = {key: value["native_g1"] for key, value in configured.items()
                     if key in CANDIDATES and isinstance(value, Mapping) and "native_g1" in value}
    if not registrations:
        return dict(catalog)
    return seal_catalog({**catalog, "revision": catalog["revision"] + ":g1:" + canonical_digest(registrations)[7:]})


__all__ = ["CANDIDATES", "catalog_for_profile"]
