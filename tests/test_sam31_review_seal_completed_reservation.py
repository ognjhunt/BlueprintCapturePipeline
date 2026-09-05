"""The review seal must accept the reservation manifest the ledger really writes after a completed call."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from blueprint_pipeline import public_scene_sam31_track_selection_review as review
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit

GOLDEN = Path(__file__).parent / "fixtures" / "sam31_review_r16_20260905"


def _golden(name: str) -> dict:
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def test_the_golden_manifest_is_what_the_ledger_writes_for_a_completed_call(tmp_path: Path) -> None:
    """Producer side of the seam: rebuild the manifest from the retained reservation and
    completion with the ledger itself and compare it with the retained manifest."""

    manifest = _golden("inference_reservations/manifest.json")
    reservation = _golden("inference_reservations/reserved.json")
    completion = _golden("inference_reservations/completed.json")
    token = reservation["reservation_id"].removeprefix("sha256:")
    run_root = tmp_path / "sdk-review"
    (run_root / "inference_reservations" / "reserved").mkdir(parents=True)
    (run_root / "inference_reservations" / "completed").mkdir(parents=True)
    shutil.copy(GOLDEN / "inference_reservations" / "reserved.json", run_root / "inference_reservations" / "reserved" / f"{token}.json")
    shutil.copy(GOLDEN / "inference_reservations" / "completed.json", run_root / "inference_reservations" / "completed" / f"{token}.json")

    rebuilt = InferenceReservationAudit(run_root=run_root, run_id=manifest["run_id"]).manifest()

    assert rebuilt["reserved_max_cost_usd"] == pytest.approx(manifest["reserved_max_cost_usd"])
    assert rebuilt["reserved_max_cost_usd"] == pytest.approx(completion["reconciled_actual_cost_usd"])
    assert rebuilt["reserved_max_cost_usd"] > 0.0  # a completed call keeps its reconciled cost reserved
    assert rebuilt["in_flight_unknown_count"] == 0 and rebuilt["reservations"][0]["status"] == "completed"
    assert {k: v for k, v in rebuilt.items() if k != "reservations"} == {k: v for k, v in manifest.items() if k != "reservations"}


def test_the_seal_accepts_the_completed_manifest_the_ledger_writes() -> None:
    """Consumer side: R16 (2026-09-05 21:54Z) — the SDK reviewer accepted, the ledger wrote
    reserved_max_cost_usd = 0.077398 (the reconciled cost of the completed call), and the
    seal refused the execution because it demanded 0.0. Nothing had run this seam since
    the ledger was hardened on 2026-08-31."""

    manifest = _golden("inference_reservations/manifest.json")
    execution = _golden("execution_receipt.json")

    review._validate_completed_reservation_manifest(manifest, run_id=execution["run_id"])


@pytest.mark.parametrize(
    "mutate, reason",
    [
        (lambda m: m.update(in_flight_unknown_count=1) or m["reservations"][0].update(status="in_flight_unknown"), "in_flight"),
        (lambda m: m.update(reserved_max_cost_usd=m["reserved_max_cost_usd"] + 0.01), "reserved_not_reconciled"),
        (lambda m: m.update(projected_max_cost_usd_total=review.AI_REVIEW_MAX_COST_USD + 0.5), "over_cap"),
        (lambda m: m.update(reservation_count=2), "count"),
        (lambda m: m.update(run_id="other-run"), "run"),
    ],
)
def test_the_seal_still_refuses_an_open_or_inconsistent_manifest(mutate, reason) -> None:
    manifest = _golden("inference_reservations/manifest.json")
    execution = _golden("execution_receipt.json")
    mutate(manifest)
    manifest["inference_reservation_manifest_digest"] = canonical_digest(
        manifest, digest_field="inference_reservation_manifest_digest"
    )

    with pytest.raises(review.Sam31TrackSelectionReviewError, match="sam31_review_execution_receipt_invalid"):
        review._validate_completed_reservation_manifest(manifest, run_id=execution["run_id"])
