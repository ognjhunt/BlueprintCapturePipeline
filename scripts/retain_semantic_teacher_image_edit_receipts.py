#!/usr/bin/env python3
"""Seal the terminal receipts of a semantic-teacher image-edit paid run.

`semantic_teacher_image_edit_paid_lane` produces the three receipts that make a
paid run terminal, and none of them could be produced by any script or by any
module carrying a `main()`. So the lane could be *started* from a production
path and only *closed* from a Python session -- which is the state that leaves
an attempt with no terminal artifact and a provider bill nobody reconciled.

Same defect as #512 (lanes), #520 (bundle modules), #523 (authority
materializers) and the ArtiFixer3D input chain, in a fifth scope. It is also
what pushed `tests/test_materializer_reachability.py` from 73 to 76.

Which step closes a run depends on how far it got:

    allocation never became possible
        -> no-allocation-closeout   (dry run + watchdog closeout, and a reason)

    the run allocated and finished
        -> provider-zero            (API-confirmed resource and staged-object
                                     zero, watchdog, cost, and both logs)
        -> result                   (every generated PNG under an exact
                                     allowlist, bound to the provider-zero proof)

`result` is sealed last because it binds the provider-zero receipt: the images
are only retainable once the run is proven to be renting nothing.

The flag table below *is* the call -- the parser and the keyword arguments are
both built from it, and `tests/test_semantic_teacher_image_edit_lane_cli.py`
derives the left column from each function's own signature. All 30 keyword-only
parameters across the three are required upstream and required here; on a
closeout a dropped path is a piece of evidence the receipt claims to bind and
does not.

Reads retained bytes only; performs no provider mutation and rents nothing. It
seals what a run already did, and refuses when the evidence is not there.
"""

from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.materializer_cli import Param as _Param, Step, run
from blueprint_pipeline.semantic_teacher_image_edit_paid_lane import (
    materialize_semantic_teacher_image_edit_result,
    materialize_semantic_teacher_no_allocation_closeout,
    materialize_semantic_teacher_provider_zero_receipt,
)


def Param(flag: str, help: str = "", *, type=None) -> _Param:
    """One required flag. Every keyword this script supplies is evidence.

    The shared `Param` defaults to optional because most lanes have selectors
    that may be omitted. This one has none: on a closeout a dropped path is a
    piece of evidence the receipt claims to bind and does not, so requiring
    them is stated here rather than inherited from a default that could change.
    `tests/test_semantic_teacher_image_edit_lane_cli.py` holds that down.
    """

    return _Param(flag, help, required=True, type=type)


OUTPUT = Param("--output", "Where to write the sealed receipt.")

STEPS: dict[str, Step] = {
    "result": Step(
        "Retain every generated PNG under an exact allowlist, once zero is proven.",
        materialize_semantic_teacher_image_edit_result,
        {
            "runtime_output_root": Param(
                "--runtime-output-root", "Root the runtime wrote its images under."
            ),
            "runtime_request_path": Param("--runtime-request", "The request the runtime ran."),
            "bundle_receipt_path": Param("--bundle-receipt", "The sealed bundle it ran."),
            "authority_path": Param("--authority", "The authority that admitted the attempt."),
            "billing_receipt_path": Param("--billing-receipt", "What the provider charged."),
            "scoped_inventory_path": Param(
                "--scoped-inventory", "Provider inventory scoped to this attempt."
            ),
            "global_inventory_path": Param(
                "--global-inventory", "Whole-account inventory, which is what proves zero."
            ),
            "object_store_cleanup_path": Param(
                "--object-store-cleanup", "Proof the staged objects are gone."
            ),
            "watchdog_receipt_path": Param(
                "--watchdog-receipt", "The independent watchdog's observation."
            ),
            "secret_redaction_path": Param(
                "--secret-redaction", "Proof no credential reached a retained byte."
            ),
            "provider_zero_path": Param(
                "--provider-zero",
                "The provider-zero receipt from the step below. Images are only "
                "retainable once the run is proven to be renting nothing.",
            ),
            "expected_task_count": Param(
                "--expected-task-count",
                "Refused unless the retained set matches exactly.",
                type=int,
            ),
            "expected_camera_count": Param(
                "--expected-camera-count",
                "Refused unless the retained set matches exactly.",
                type=int,
            ),
            "output_path": OUTPUT,
        },
    ),
    "provider-zero": Step(
        "Bind API-confirmed resource zero, staged-object zero, watchdog, and cost.",
        materialize_semantic_teacher_provider_zero_receipt,
        {
            "authority_path": Param("--authority", "The authority that admitted the attempt."),
            "bundle_receipt_path": Param("--bundle-receipt", "The sealed bundle it ran."),
            "terminal_result_path": Param(
                "--terminal-result", "The runtime's own terminal result."
            ),
            "billing_receipt_path": Param("--billing-receipt", "What the provider charged."),
            "scoped_inventory_path": Param(
                "--scoped-inventory", "Provider inventory scoped to this attempt."
            ),
            "global_inventory_path": Param(
                "--global-inventory", "Whole-account inventory, which is what proves zero."
            ),
            "object_store_cleanup_path": Param(
                "--object-store-cleanup", "Proof the staged objects are gone."
            ),
            "independent_watchdog_path": Param(
                "--independent-watchdog",
                "An observer the lane does not control; self-reported zero is not zero.",
            ),
            "secret_redaction_path": Param(
                "--secret-redaction", "Proof no credential reached a retained byte."
            ),
            "stdout_log_path": Param("--stdout-log", "The run's retained stdout."),
            "stderr_log_path": Param("--stderr-log", "The run's retained stderr."),
            "output_path": OUTPUT,
        },
    ),
    "no-allocation-closeout": Step(
        "Terminal teardown for a run where allocation never became possible.",
        materialize_semantic_teacher_no_allocation_closeout,
        {
            "dry_run_path": Param("--dry-run", "The dry run that stood in for the attempt."),
            "watchdog_closeout_path": Param(
                "--watchdog-closeout", "The watchdog's closing observation."
            ),
            "reason": Param("--reason", "Why allocation never became possible."),
            "output_path": OUTPUT,
        },
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
