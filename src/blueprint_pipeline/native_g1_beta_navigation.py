"""Preserve a current goal decision across a frozen beta reset derivation."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .beta_policy_screening_g1 import validate_cell_binding
from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import CELL_SCHEMA
from .native_g1_navigation_goal import validate_g1_navigation_goal_authority

SCHEMA = "native_g1_beta_navigation_authority.v1"


def validate_navigation_authority(value: Mapping[str, Any], *, plan: Mapping[str, Any],
                                  cell_authority: Mapping[str, Any]) -> dict[str, Any]:
    try:
        row = deepcopy(dict(value))
        if (set(row) != {"schema_version", "base_authority", "base_plan", "base_contract", "cell_authority",
                         "scene_plan_digest", "new_human_review_claimed", "authority_digest"}
                or row["schema_version"] != SCHEMA or row["new_human_review_claimed"] is not False
                or row["cell_authority"] != cell_authority
                or cell_authority.get("schema_version") != CELL_SCHEMA
                or not cell_authority["base_authority"]["candidate_id"].endswith("_vision_navi")
                or row["scene_plan_digest"] != plan["plan_digest"]
                or row["authority_digest"] != canonical_digest(row, digest_field="authority_digest")):
            raise ValueError("invalid")
        validate_g1_navigation_goal_authority(row["base_authority"], plan=row["base_plan"])
        validate_cell_binding(row["base_plan"], plan, cell_authority["cell"], cell_authority,
                              base_contract=row["base_contract"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("g1_beta_navigation_authority_invalid") from exc
    return row


def derive_navigation_authority(base_authority: Mapping[str, Any], base_plan: Mapping[str, Any],
                                derived_plan: Mapping[str, Any], cell_authority: Mapping[str, Any], *,
                                base_contract: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the unchanged goal decision; no new scene or human approval claim."""
    row = {"schema_version": SCHEMA, "base_authority": dict(base_authority), "base_plan": dict(base_plan),
           "base_contract": dict(base_contract), "cell_authority": dict(cell_authority),
           "scene_plan_digest": derived_plan["plan_digest"], "new_human_review_claimed": False}
    row["authority_digest"] = canonical_digest(row, digest_field="authority_digest")
    return validate_navigation_authority(row, plan=derived_plan, cell_authority=cell_authority)
