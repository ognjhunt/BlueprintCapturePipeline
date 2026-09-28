"""Whether a policy canary attempt's staged objects are gone, for closeout and billing (review C2).

The arena lane seals ``provider_closeout.all_staged_objects_absent`` once, when
it runs. In stream mode a promotion that failed leaves the staged output in
place -- the cleanup gate defers it -- and the sealed lane result says
``false`` for good: ``provider_output_promotion resume`` later promotes and
cleans up but never rewrites that result. Resume writes instead a digest-bound
``staged_object_absence_proof.v1.json`` in the attempt's staging directory
(``provider_output_promotion_records``), which closeout and billing accept in
place of the sealed flag. A lane result that already proves absence (every
download-mode run) never reads a proof.

Policy. Billing needs only the absence: a valid proof for a staging manifest
that required promotion answers the spend question, whatever promotion did.
Closeout also asks whether the output is durable. A proof whose promotion
succeeded (``promoted``), or confirmed that no output was ever staged
(``absent_confirmed``) for a run that has no provider result, closes cleanly.
Any other proof -- ``failed``, or ``absent_confirmed`` beside a provider
result -- still seals, so the run leaves the queue, but carries the blocker
``policy_canary_provider_output_not_durable``: its staged objects are gone and
its output was not kept. An invalid proof proves nothing and is named.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .provider_output_promotion_records import (
    ABSENCE_PROOF_FILENAME,
    ProviderOutputPromotionRecordError,
    load_staged_object_absence_proof,
)

# The arena lane's staging directory under its attempt root.
STAGING_DIRNAME = "object_store_staging"
NOT_DURABLE = "policy_canary_provider_output_not_durable"


def _sealed_absent(lane_result: Mapping[str, Any]) -> bool:
    closeout = lane_result.get("provider_closeout")
    return isinstance(closeout, Mapping) and closeout.get("all_staged_objects_absent") is True


def staged_object_absence_proof(lane_result: Mapping[str, Any]) -> tuple[dict, Path] | None:
    """The attempt's validated absence proof and its path, or None when there is none.

    Only a proof for a staging manifest that required promotion counts. A
    proof file that does not validate raises
    ``ProviderOutputPromotionRecordError``.
    """
    attempt = str(lane_result.get("attempt_root") or "").strip()
    if not attempt:
        return None
    staging = Path(attempt) / STAGING_DIRNAME
    proof = load_staged_object_absence_proof(staging)
    if proof is None or proof.get("output_promotion_required") is not True:
        return None
    return proof, staging / ABSENCE_PROOF_FILENAME


def billing_staged_objects_absent(lane_result: Mapping[str, Any]) -> tuple[bool, Path | None]:
    """Billing's answer: absent, and the proof path when the sealed flag did not say so."""
    if _sealed_absent(lane_result):
        return True, None
    try:
        found = staged_object_absence_proof(lane_result)
    except ProviderOutputPromotionRecordError:
        return False, None
    return (True, found[1]) if found is not None else (False, None)


def closeout_staged_objects(lane_result: Mapping[str, Any]) -> dict[str, Any]:
    """Closeout's answer: ``{"absent": bool, "blockers": [...]}`` under the module policy."""
    if _sealed_absent(lane_result):
        return {"absent": True, "blockers": []}
    try:
        found = staged_object_absence_proof(lane_result)
    except ProviderOutputPromotionRecordError as exc:
        return {"absent": False, "blockers": [f"policy_canary_staged_object_absence_proof_invalid:{exc}"]}
    if found is None:
        return {"absent": False, "blockers": []}
    status = found[0].get("promotion_status")
    produced_output = bool(str(lane_result.get("native_control_result_path") or "").strip())
    durable = status == "promoted" or (status == "absent_confirmed" and not produced_output)
    return {"absent": True, "blockers": [] if durable else [NOT_DURABLE]}


__all__ = [
    "NOT_DURABLE",
    "STAGING_DIRNAME",
    "billing_staged_objects_absent",
    "closeout_staged_objects",
    "staged_object_absence_proof",
]
