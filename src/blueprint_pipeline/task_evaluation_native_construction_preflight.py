"""Run the control-plane half of a native construction lane without a provider.

A retained feedback round does most of its work on the control plane: it reads
the sealed packet, binds the admitted candidate universe, and builds the four
planner documents.  Only after all of that does anything reach a GPU.  When one
of those control-plane steps carries a stale key or a stale path, the defect is
invisible until an allocation has already been paid for and torn down, so a
one-line mismatch costs a full rent-crash-fix-redeploy cycle.

This preflight executes that same chain against the same sealed inputs a real
launch would consume, and it allocates nothing.  Every refusal is a named
predicate, so an operator reading the receipt learns which contract failed
rather than that "something" was invalid.

The claim boundary is narrow and deliberate: passing here proves the
control-plane chain binds these exact bytes.  It proves nothing about what the
robot would do, and it is not a substitute for the native gates.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from blueprint_pipeline.task_evaluation_curobo_context import (
    CuroboContextError,
    materialize_remote_curobo_context,
)

PREFLIGHT_SCHEMA_VERSION = "native_construction_downstream_preflight.v1"
CONTEXT_SCHEMA_VERSION = "native_task_arena_launch_preparation_context.v2"

CLAIM_BOUNDARY = (
    "This preflight proves only that the control-plane chain binds these exact "
    "sealed bytes without a provider. It proves nothing about robot behaviour "
    "and does not stand in for any native construction or controls gate."
)


class NativeConstructionPreflightError(RuntimeError):
    """One named control-plane contract refused this prepared lane."""


def _read_json(path: Path, *, blocker: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeConstructionPreflightError(f"{blocker}:{path.name}") from exc
    if not isinstance(value, Mapping):
        raise NativeConstructionPreflightError(f"{blocker}:{path.name}")
    return dict(value)


def _resolve_binding(context: Mapping[str, Any]) -> tuple[Path, str]:
    """Take the packet directory and commit a real launch would consume."""

    if context.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise NativeConstructionPreflightError(
            "preflight_context_schema_invalid"
        )
    operations = context.get("operations")
    references = context.get("references")
    if not isinstance(operations, Mapping) or not isinstance(references, Mapping):
        raise NativeConstructionPreflightError("preflight_context_sections_missing")
    scene = references.get("scene")
    if not isinstance(scene, Mapping):
        raise NativeConstructionPreflightError("preflight_context_scene_missing")
    packet_dir = str(scene.get("packet_dir") or "")
    commit = str(operations.get("source_commit") or "")
    if not packet_dir:
        raise NativeConstructionPreflightError("preflight_context_packet_dir_missing")
    if not commit:
        raise NativeConstructionPreflightError("preflight_context_source_commit_missing")
    packet = Path(packet_dir).expanduser()
    if packet.is_symlink() or not packet.is_dir():
        raise NativeConstructionPreflightError("preflight_packet_dir_unreadable")
    return packet.resolve(), commit


def _candidate_universe(path: Path) -> dict[str, Any]:
    universe = _read_json(path, blocker="preflight_candidate_universe_unreadable")
    rows = universe.get("candidates")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise NativeConstructionPreflightError("preflight_candidate_universe_empty")
    return universe


def run_preflight(
    *,
    context_file: str | Path,
    candidate_universe: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Bind the sealed packet and universe exactly as a retained round would."""

    context = _read_json(
        Path(context_file).expanduser(), blocker="preflight_context_unreadable"
    )
    packet, commit = _resolve_binding(context)
    universe_path = Path(candidate_universe).expanduser()
    universe = _candidate_universe(universe_path)

    scratch = Path(output_root).expanduser() if output_root else Path(
        tempfile.mkdtemp(prefix="native-construction-preflight-")
    )
    context_root = scratch / "curobo-context"
    try:
        materialized, remote_root = materialize_remote_curobo_context(
            packet_dir=packet,
            universe=universe,
            output_root=context_root,
            commit=commit,
            warm_session=None,
        )
    except (CuroboContextError, OSError, ValueError) as exc:
        raise NativeConstructionPreflightError(
            f"preflight_curobo_context_failed:{exc}"
        ) from exc
    finally:
        if output_root is None:
            shutil.rmtree(scratch, ignore_errors=True)

    inventory = materialized.analytic_candidate_inventory
    bound = inventory.get("candidates") if isinstance(inventory, Mapping) else None
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "completed",
        "blockers": [],
        "claim_boundary": CLAIM_BOUNDARY,
        "provider_allocation_performed": False,
        "paid_execution_requested": False,
        "expected_production_commit": commit,
        "packet_dir": str(packet),
        "candidate_universe": str(universe_path),
        "admitted_candidate_count": len(universe.get("candidates") or []),
        "bound_candidate_count": len(bound) if isinstance(bound, Sequence) else 0,
        "remote_package_root": remote_root,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-file", required=True)
    parser.add_argument("--candidate-universe", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--receipt-out")
    args = parser.parse_args(argv)
    try:
        receipt = run_preflight(
            context_file=args.context_file,
            candidate_universe=args.candidate_universe,
            output_root=args.output_root,
        )
    except NativeConstructionPreflightError as exc:
        blocked = {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "blocked",
            "blockers": [str(exc)],
            "claim_boundary": CLAIM_BOUNDARY,
            "provider_allocation_performed": False,
            "paid_execution_requested": False,
        }
        print(json.dumps(blocked, indent=1, sort_keys=True))
        if args.receipt_out:
            Path(args.receipt_out).write_text(
                json.dumps(blocked, indent=1, sort_keys=True), encoding="utf-8"
            )
        return 2
    print(json.dumps(receipt, indent=1, sort_keys=True))
    if args.receipt_out:
        Path(args.receipt_out).write_text(
            json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
