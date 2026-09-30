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

Not yet final. A promotion checkpoints its receipt, witness ``pending``, before
its witness step. While the staging dir's current receipt is such a
checkpoint, a promotion is running or was cut short, so neither billing nor
closeout accepts the proof yet (``policy_canary_provider_output_promotion_not_final``
names why): the run waits rather than sealing on a promotion that has not
finished.

Bound to the current receipt (review critical 1). A proof answers only for the
promotion receipt it was sealed with (``promotion_receipt_digest``). A later
promotion that wrote the staging dir's receipt -- say one that found an output
staged after the proof and failed to make it durable -- leaves that proof
saying nothing about what it found, so a proof whose receipt is not the current
one proves nothing (``staged_object_absence_proof_receipt_not_current``). The
next cleanup that proves absence seals a fresh proof for its own receipt and
sets the stale one aside.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .provider_output_promotion_records import (
    ABSENCE_PROOF_FILENAME,
    ProviderOutputPromotionRecordError,
    load_promotion_receipt,
    load_staged_object_absence_proof,
)

# The arena lane's staging directory under its attempt root.
STAGING_DIRNAME = "object_store_staging"
NOT_DURABLE = "policy_canary_provider_output_not_durable"
NOT_FINAL = "policy_canary_provider_output_promotion_not_final"


def _sealed_absent(lane_result: Mapping[str, Any]) -> bool:
    closeout = lane_result.get("provider_closeout")
    return isinstance(closeout, Mapping) and closeout.get("all_staged_objects_absent") is True


def _checkpoint(receipt: Mapping[str, Any]) -> bool:
    """Whether a promotion receipt is a checkpoint (witness pending): a promotion mid-run."""
    section = (receipt.get("staged_objects") or {}).get("paired_witness") or {}
    return (receipt.get("witness") or {}).get("disposition") == "pending" or section.get("state") == "pending"


def staged_object_absence_proof(lane_result: Mapping[str, Any]) -> tuple[dict, Path, bool] | None:
    """The attempt's validated absence proof, its path and whether promotion is final; else None.

    Only a proof for a staging manifest that required promotion counts. A
    proof file that does not validate, or that is bound to a promotion receipt
    other than the staging dir's current, final one, raises
    ``ProviderOutputPromotionRecordError``.
    """
    attempt = str(lane_result.get("attempt_root") or "").strip()
    if not attempt:
        return None
    staging = Path(attempt) / STAGING_DIRNAME
    proof = load_staged_object_absence_proof(staging)
    if proof is None or proof.get("output_promotion_required") is not True:
        return None
    path = staging / ABSENCE_PROOF_FILENAME
    receipt = load_promotion_receipt(staging, staging_manifest_sha256=proof.get("staging_manifest_sha256"))
    if receipt is not None and _checkpoint(receipt):
        return proof, path, False
    if receipt is None or receipt.get("receipt_digest") != proof.get("promotion_receipt_digest"):
        raise ProviderOutputPromotionRecordError("staged_object_absence_proof_receipt_not_current")
    return proof, path, True


def billing_staged_objects_absent(lane_result: Mapping[str, Any]) -> tuple[bool, Path | None]:
    """Billing's answer: absent, and the proof path when the sealed flag did not say so."""
    if _sealed_absent(lane_result):
        return True, None
    try:
        found = staged_object_absence_proof(lane_result)
    except ProviderOutputPromotionRecordError:
        return False, None
    return (True, found[1]) if found is not None and found[2] else (False, None)


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
    if not found[2]:
        return {"absent": False, "blockers": [NOT_FINAL]}
    status = found[0].get("promotion_status")
    produced_output = bool(str(lane_result.get("native_control_result_path") or "").strip())
    durable = status == "promoted" or (status == "absent_confirmed" and not produced_output)
    return {"absent": True, "blockers": [] if durable else [NOT_DURABLE]}


__all__ = [
    "NOT_DURABLE",
    "NOT_FINAL",
    "STAGING_DIRNAME",
    "billing_staged_objects_absent",
    "closeout_staged_objects",
    "staged_object_absence_proof",
]
