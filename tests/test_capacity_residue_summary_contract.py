"""Actual merged residue projection and remaining-byte forecast contract.

Rows are tiny synthetic metadata; no archive, payload, provider or cleanup runs.
The byte counters describe logical local eviction, not observed disk recovery.
"""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_capacity_controller.py
#   src/blueprint_pipeline/control_plane_storage_gc_reasons.py
#   src/blueprint_pipeline/task_evaluation_result_residue_offload.py

import pytest

from blueprint_pipeline import control_plane_capacity_controller as capacity
from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary
from blueprint_pipeline.task_evaluation_result_residue_offload import residue_phase

NOW = 1000


def _row(*, candidate=100, evicted=40, **extra):
    return {
        "status": "applied",
        "candidate_count": 1,
        "candidate_bytes": candidate,
        "offloaded_count": 1,
        "offloaded_bytes": evicted,
        "skipped_by_reason": {},
        **extra,
    }


def _summary(rows, *, enabled=True, applying=True, opt_in=None, phase_extra=None, **extra):
    phase = residue_phase(rows, enabled=enabled, applying=applying)
    phase.update(phase_extra or {})
    return build_storage_gc_summary(
        {
            "status": "applied",
            "observed_at_epoch": NOW,
            "report_digest": "sha256:" + "a" * 64,
            "opt_in": (
                {"evidence_offload": True, "result_residue_offload": True}
                if opt_in is None
                else opt_in
            ),
            "result_residue_offload": phase,
            "skipped_roots": [],
            "phase_errors": [],
            **extra,
        }
    )


def _outlook(summary):
    return capacity._reclaim_outlook(summary, now=NOW, volume_growth="blocked")


@pytest.mark.parametrize("candidate,evicted,remaining", [(100, 40, 60), (100, 100, 0)])
def test_actual_residue_summary_forecasts_only_remaining_logical_bytes(
    candidate, evicted, remaining
):
    summary = _summary([_row(candidate=candidate, evicted=evicted)])
    phase = summary["phases"]["result_residue_offload"]
    assert summary["opt_in"]["result_residue_offload"] is True
    assert phase["enabled"] is True and phase["status"] == "applied"
    assert phase["candidate_bytes"] == candidate
    assert phase["removed_or_offloaded_bytes"] == evicted
    assert phase["retained_by_reason"] == {}
    outlook, reasons, ineffective = _outlook(summary)
    assert outlook["reclaimable_bytes"] == remaining
    assert outlook["sources"]["result_residue_offload"] == remaining
    assert outlook["next_reclaim_epoch"] == NOW + capacity.GC_SUMMARY_INTERVAL_SECONDS
    assert reasons == [] and ineffective is False
    eta = capacity.capacity_eta(50, summary={"reclaim_outlook": outlook}, now=NOW)
    assert eta["eta_basis"] == (
        "reclaim_scheduled" if remaining >= 50 else "operator_action_required"
    )
    assert eta["eta_epoch"] == (float(outlook["next_reclaim_epoch"]) if remaining >= 50 else None)


def test_actual_empty_residue_summary_proves_zero_eligible_and_zero_evicted():
    summary = _summary([])
    outlook, reasons, ineffective = _outlook(summary)
    assert outlook["reclaimable_bytes"] == 0
    assert outlook["sources"]["result_residue_offload"] == 0
    assert reasons == [] and ineffective is True


@pytest.mark.parametrize(
    "options",
    [
        {"applying": False},
        {"enabled": False},
        {
            "enabled": False,
            "applying": False,
            "opt_in": {"evidence_offload": True, "result_residue_offload": False},
        },
        {"opt_in": {"evidence_offload": True}},
        {"opt_in": {"result_residue_offload": True}},
    ],
    ids=[
        "dry-phase",
        "disabled-phase",
        "disabled-switch",
        "missing-residue-switch",
        "missing-evidence-switch",
    ],
)
def test_actual_residue_summary_needs_explicit_applying_enabled_proof(options):
    outlook, reasons, ineffective = _outlook(_summary([_row()], **options))
    assert outlook["reclaimable_bytes"] is None
    assert reasons == [] and ineffective is False
    assert capacity.capacity_eta(1, summary={"reclaim_outlook": outlook}, now=NOW) == {
        "eta_epoch": None,
        "eta_basis": "unknown",
    }


@pytest.mark.parametrize(
    "rows",
    [
        [_row(status="retained", retained_reason="pointer_changed")],
        [_row(skipped_by_reason={"reader_reopens": {"count": 1, "bytes": 3}})],
        [_row(status="retained", retained_reason="deferred_tick_cap", evicted=0)],
        [{"status": "error", "error_type": "OSError", "stage": "residue", "errno": 5}],
    ],
    ids=["retained-run", "skipped-member", "deferred-publication-cap", "failed-run"],
)
def test_actual_kept_or_failed_residue_never_supplies_a_forecast(rows):
    summary = _summary(rows)
    assert summary["phases"]["result_residue_offload"]["retained_by_reason"]
    outlook, reasons, ineffective = _outlook(summary)
    assert outlook["reclaimable_bytes"] is None
    assert reasons == [] and ineffective is False


_DERIVED = {"status": "applied", "removed_bytes": 0, "retained_by_reason": {}}


def test_kept_residue_supplies_no_bytes_but_leaves_the_other_phases_forecast():
    """Residue offload is on by default, so its steady-state retained rows (a missing dispatch receipt, a hot
    run, one already offloaded) must not blank every other phase's forecast or silence the reclaim page."""

    summary = _summary([_row(status="retained", retained_reason="pointer_changed")],
                       derived_directories={**_DERIVED, "candidate_bytes": 5 * 1024**3})
    outlook, _, ineffective = _outlook(summary)
    assert outlook["sources"]["result_residue_offload"] is None
    assert outlook["sources"]["derived_directories"] == 5 * 1024**3
    assert outlook["reclaimable_bytes"] == 5 * 1024**3 and ineffective is False
    eta = capacity.capacity_eta(1024**3, summary={"reclaim_outlook": outlook}, now=NOW)
    assert eta["eta_basis"] == "reclaim_scheduled"


def test_kept_residue_that_still_holds_or_moved_bytes_never_pages_ineffective():
    for row in (_row(status="retained", retained_reason="deferred_tick_cap", evicted=0),
                _row(status="retained", retained_reason="pointer_changed")):
        summary = _summary([row], derived_directories={**_DERIVED, "candidate_bytes": 0})
        outlook, _, ineffective = _outlook(summary)
        assert outlook["reclaimable_bytes"] == 0 and ineffective is False


def test_reclaim_ineffective_pages_once_every_phase_kept_residue_included_is_empty():
    summary = _summary([_row(status="retained", retained_reason="pointer_changed", candidate=0, evicted=0)],
                       derived_directories={**_DERIVED, "candidate_bytes": 0})
    assert summary["phases"]["result_residue_offload"]["retained_by_reason"]
    outlook, _, ineffective = _outlook(summary)
    assert outlook["reclaimable_bytes"] == 0 and ineffective is True


def test_omitted_residue_errors_make_actual_projected_candidates_unknown():
    summary = _summary([_row()], phase_extra={"omitted_errors_count": 1})
    assert summary["phases"]["result_residue_offload"]["candidate_bytes"] is None
    assert _outlook(summary)[0]["reclaimable_bytes"] is None


@pytest.mark.parametrize("field", ["candidate_bytes", "offloaded_bytes"])
@pytest.mark.parametrize("value", [True, "100", None, -1, 1.5])
def test_malformed_phase_counters_do_not_become_forecast_bytes(field, value):
    # Exercise actual summary type validation before the consumer: do not
    # bypass the projection with a hand-constructed summary document.
    summary = _summary([_row()], phase_extra={field: value})
    assert _outlook(summary)[0]["reclaimable_bytes"] is None
    assert _outlook(summary)[2] is False


def test_actual_pin_release_counts_add_no_bytes_to_derived_reclaim():
    summary = _summary(
        [],
        derived_directories={
            "status": "applied",
            "candidate_bytes": 128,
            "removed_bytes": 32,
            "retained_by_reason": {},
        },
        terminal_cache_pins={
            "status": "applied",
            "enabled": True,
            "candidate_count": 50,
            "released_count": 50,
            "retained_by_reason": {},
        },
    )
    assert summary["phases"]["terminal_cache_pins"]["released_count"] == 50
    outlook, _, ineffective = _outlook(summary)
    assert outlook["reclaimable_bytes"] == 96
    assert outlook["sources"]["derived_directories"] == 96
    assert "terminal_cache_pins" not in outlook["sources"]
    assert ineffective is False
