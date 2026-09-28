"""Operator-installed, intent-scoped amendments to a lifetime supervision cap."""
from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from .contracts import AgentExecutionError, DIGEST, IDENTIFIER, digest


class SupervisionAllowance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intent_id: str = Field(pattern=IDENTIFIER)
    intent_digest: str = Field(pattern=DIGEST)
    maximum_revisions: int = Field(ge=1, le=100)
    maximum_reserved_inference_usd: float = Field(gt=0, le=100, allow_inf_nan=False)
    expires_at: float = Field(gt=0, allow_inf_nan=False)
    authorization_reference: str = Field(min_length=1, max_length=1024)

    @property
    def allowance_digest(self):
        return digest(self.model_dump(mode="json"))


def current_allowance(service, intent_id, intent_digest):
    rows = [row for row in service.config.automatic_supervision_allowances if row.intent_id == intent_id]
    if len(rows) > 1 or (rows and rows[0].intent_digest != intent_digest):
        raise AgentExecutionError("automatic_supervision_allowance_scope_invalid")
    return rows[0] if rows else None


def allowance_is_current(service, plan):
    if not plan.automatic_intent_digest:
        return True
    allowance = current_allowance(service, plan.run_id, plan.automatic_intent_digest)
    if allowance is None:
        return plan.automatic_allowance_digest is None
    return (allowance.allowance_digest == plan.automatic_allowance_digest
        and time.time() < allowance.expires_at
        and plan.maximum_revisions == allowance.maximum_revisions
        and plan.maximum_reserved_inference_usd == allowance.maximum_reserved_inference_usd
        and plan.expires_at <= allowance.expires_at)
