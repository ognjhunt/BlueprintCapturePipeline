"""Bound parent work and retain failed acceptance artifacts without fabricated usage."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from contextlib import contextmanager

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from scripts.control_plane_concurrency_load_test import allocated_tree_bytes, build_summary
from scripts.control_plane_concurrency_protocol import seal_document


@contextmanager
def whole_run_deadline(deadline):
    """A main-thread POSIX watchdog covers synchronous building and retirement too."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("global_timeout")
    if signal.getitimer(signal.ITIMER_REAL) != (0, 0):
        raise ValueError("harness_existing_alarm_not_owned")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("global_timeout")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, remaining)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _error(exc):
    return getattr(exc, "errors", None) or [type(exc).__name__ + ":" + str(exc)]


def failed_child_scenes(children, blockers):
    """Retain verified partial receipts; malformed receipts are separate blockers."""
    scenes = []
    for row in children:
        path = row["result_path"]
        try:
            if not path.exists():
                blockers.append("child_receipt_not_ingested:" + row["scene_key"])
                continue
            if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
                raise ValueError("child_receipt_path_or_size_invalid")
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or value.get("receipt_digest") != canonical_digest(
                value, digest_field="receipt_digest"
            ):
                raise ValueError("child_receipt_changed")
            if not isinstance(value.get("scene_id"), str) or not isinstance(
                value.get("stages"), list
            ):
                raise ValueError("child_receipt_shape_invalid")
            scenes.append(value)
        except TimeoutError:
            raise
        except Exception as exc:
            blockers.extend(_error(exc))
    return scenes


def finish_report(
    *,
    args,
    source,
    release,
    confinement,
    roots,
    baseline,
    children,
    scenes,
    sampler,
    blockers,
    retirement,
    peak,
    held,
    measure=allocated_tree_bytes,
):
    """Reporting has a 30-second recovery bound and always keeps unavailable values null."""
    measurements = {"control_plane": None, "objects": None, "workers": None}
    try:
        with whole_run_deadline(time.monotonic() + 30):
            if blockers:
                scenes = failed_child_scenes(children, blockers)
            for name, root in roots.items():
                try:
                    measurements[name] = measure(root)
                except TimeoutError:
                    raise
                except Exception as exc:
                    blockers.extend(_error(exc))
    except Exception as exc:
        blockers.extend(_error(exc))
    arguments = dict(
        source_commit=source,
        expected_beta_concurrency=args.expected_beta_concurrency,
        owner_confirmed=args.owner_confirmed_concurrency,
        scenes=scenes,
        measured_peak_concurrency=peak,
        concurrency_hold_seconds=held,
        baseline_allocated_bytes=baseline,
        final_allocated_bytes=measurements["control_plane"],
        maximum_retained_bytes=int(args.maximum_retained_gib * 1024**3),
        maximum_delta_bytes=int(args.maximum_delta_gib * 1024**3),
        external_calls=sum(s.get("actual_provider_calls", 0) for s in scenes),
    )
    try:
        summary = build_summary(**arguments)
    except Exception as exc:
        blockers.extend(_error(exc))
        # Invalid stage metrics must fail while retaining the original stage receipts.
        summary = build_summary(**{**arguments, "scenes": []})
        summary["scenes"] = scenes
    uname = os.uname()
    summary.update(
        host={
            "hostname": socket.gethostname(),
            "system": uname.sysname,
            "release": uname.release,
            "machine": uname.machine,
            "python_version": sys.version.split()[0],
            "parent_pid": os.getpid(),
        },
        release_binding=release,
        kernel_confinement=confinement,
        retirement=retirement,
        peak_allocated_bytes=sampler.peak_bytes if sampler.sample_count else None,
        allocation_sample_count=sampler.sample_count,
        allocation_incomplete_scan_count=sampler.incomplete_scan_count,
        filesystem_and_allocation_samples=sampler.samples,
        allocation_interval_seconds=sampler.interval_seconds,
        fixture_object_allocated_bytes=measurements["objects"],
        worker_allocated_bytes=measurements["workers"],
        measurement_states={
            name: "observed" if value is not None else "not_ingested"
            for name, value in measurements.items()
        },
        report_recovery_timeout_seconds=30,
        owned_child_shutdown_timeout_seconds=20,
        maximum_heavy_stage_parallelism_configured=1,
        concurrency_boundary="positive policy-output storage reservations",
        automatic_terminal_reconciliation_proven=False,
        normal_six_hour_retention_proven=False,
    )
    summary["blockers"] = sorted(set(summary["blockers"] + blockers))
    summary["status"] = "failed" if summary["blockers"] else "passed"
    summary["owner_sized_acceptance_complete"] = (
        args.owner_confirmed_concurrency and summary["status"] == "passed"
    )
    seal_document(args.report, summary)
    return 0 if summary["status"] == "passed" else 1
