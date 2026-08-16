#!/usr/bin/env python3
"""Ask a lane's launch question without renting anything.

Two of the defects that cost a paid GPU run on 2026-08-13 were not GPU defects
at all. SimReady rented a card, ran the Isaac probe, tore the instance down
with a 200, and reported `completed, blockers: []` -- and the launch was
blocked, because the lane sealed its terminal artifacts under the job root
while its provider run lives one directory deeper. The Content Agents texture
key was the same shape: a binding that could never resolve, discovered after
the money was spent.

Neither needed a provider to find. Both needed the launch's own question asked
against the lane's own sealing path:

    given a result shaped like one this lane produces, and evidence where this
    lane actually writes it, would `_terminal_evidence` pass?

That costs nothing and takes a second. It is the question the dispatcher asks
*after* the GPU is paid for. This asks it first.

Deliberately not a mock of the lane. It calls the real
`seal_lane_terminal_artifacts` against a real directory tree, and the real
dispatcher verifier against the lane's real published profile. The only
synthetic part is the provider evidence -- the bytes a GPU would have left --
because that is the one thing a rehearsal cannot produce.

Rents nothing, contacts no provider, and reads no credential.
"""

from __future__ import annotations

import argparse
import ast
import json
import tempfile
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.task_evaluation_artifact_manifest import (
    ADAPTER_RESULT_NAME,
    PROVIDER_EVIDENCE_DIRNAME,
    PROVIDER_RUN_DIRNAME,
    TEARDOWN_MANIFEST_NAME,
    seal_lane_terminal_artifacts,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import _terminal_evidence

SCHEMA_VERSION = "lane_terminal_contract_rehearsal.v1"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"

#: A result shaped like one a lane produces after a provider actually ran. The
#: cost is what tells the sealer this was not a dry run -- the distinction the
#: SimReady defect turned on.
REHEARSED_RESULT: dict[str, Any] = {
    "status": "completed",
    "blockers": [],
    "estimated_cost_usd": 0.0812,
    "continuing_spend_from_this_run": False,
    "retry_cap": 0,
}


class LaneRehearsalError(ValueError):
    """The lane cannot be rehearsed as configured."""


def lane_seals_under_a_nested_attempt(module: str) -> bool:
    """Does this lane write its provider run under a numbered attempt?

    Read from the lane's own source rather than guessed, so a lane that changes
    its layout changes what gets rehearsed. This is exactly the difference that
    made SimReady's evidence invisible to its own sealer.
    """

    path = SOURCE_ROOT / module
    if not path.is_file():
        raise LaneRehearsalError(f"lane_module_missing:{module}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == "provider_run" for target in node.targets)
            and isinstance(node.value, ast.BinOp)
            and (
                (
                    isinstance(node.value.right, ast.Constant)
                    and node.value.right.value == PROVIDER_RUN_DIRNAME
                )
                or (
                    isinstance(node.value.right, ast.Name)
                    and node.value.right.id == "PROVIDER_RUN_DIRNAME"
                )
            )
        ):
            # ``root`` is the conventional normalized ``job_dir`` alias used
            # by lanes that share the terminal-artifact helper.  It is not a
            # numbered provider attempt.  Treating the imported constant as
            # unreadable (or treating ``root`` as nested) made the no-spend
            # rehearsal reject the semantic-teacher lane before launch.
            return ast.unparse(node.value.left) not in {"job", "root"}
    raise LaneRehearsalError(f"lane_provider_run_root_not_found:{module}")


def _write_provider_evidence(attempt_root: Path) -> None:
    """The bytes a GPU would have left behind, in the conventional roles."""

    provider_run = attempt_root / PROVIDER_RUN_DIRNAME
    provider_run.mkdir(parents=True, exist_ok=True)
    (provider_run / TEARDOWN_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": "vast_teardown_manifest.v1",
                "generated_at": "2026-08-13T00:00:00+00:00",
                "status": "completed",
                "vast_instance_ids": [1],
                "teardown_actions_performed": [
                    {"instance_id": 1, "action": "destroy_instance", "status": "completed"}
                ],
                "continuing_spend_from_this_run": False,
            }
        ),
        encoding="utf-8",
    )
    (provider_run / ADAPTER_RESULT_NAME).write_text(
        json.dumps({"status": "completed", "vast_instance_ids": [1]}), encoding="utf-8"
    )
    evidence = attempt_root / PROVIDER_EVIDENCE_DIRNAME
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "rehearsed_execution.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )


def rehearse_lane_terminal_contract(
    *,
    profile_path: str | Path,
    lane_module: str,
    lane: str | None = None,
) -> dict[str, Any]:
    """Would this lane's launch pass its own terminal contract?"""

    profile_file = Path(profile_path).expanduser().resolve()
    if not profile_file.is_file():
        raise LaneRehearsalError("rehearsal_profile_missing")
    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise LaneRehearsalError("rehearsal_profile_not_object")
    terminal = profile.get("terminal_contract")
    if not isinstance(terminal, dict) or not terminal.get("result_path"):
        raise LaneRehearsalError("rehearsal_profile_has_no_terminal_contract")

    nested = lane_seals_under_a_nested_attempt(lane_module)

    with tempfile.TemporaryDirectory(prefix="lane-rehearsal-") as raw:
        run_root = Path(raw)
        job = run_root / "allocator" / "rehearsal-job"
        attempt_root = job / "attempts" / "attempt_001" if nested else job
        attempt_root.mkdir(parents=True, exist_ok=True)
        _write_provider_evidence(attempt_root)

        sealed = seal_lane_terminal_artifacts(
            dict(REHEARSED_RESULT),
            attempt_root=attempt_root,
            lane=lane or lane_module.removesuffix(".py"),
            binding={"provider": "vast", "rehearsal": True},
        )

        # The dispatcher reads the result from the path the profile names, so
        # the rehearsal has to put it exactly there.
        result_path = Path(
            str(terminal["result_path"]).replace("{launch_run_root}", str(run_root))
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(sealed, sort_keys=True), encoding="utf-8")

        evidence = _terminal_evidence(profile, execute=True, run_root=run_root)

    blockers = list(evidence.get("blockers") or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "would_pass" if not blockers else "would_block",
        "profile_id": profile.get("profile_id"),
        "lane_module": lane_module,
        "seals_under_nested_attempt": nested,
        "terminal_evidence_status": evidence.get("status"),
        "blockers": blockers,
        "sealed_result_status": sealed.get("status"),
        "sealed_result_blockers": list(sealed.get("blockers") or []),
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "claim_boundary": (
            "This rehearsal proves only that a result shaped like this lane's, "
            "with evidence where this lane writes it, satisfies the profile's "
            "terminal contract. It runs no provider and proves nothing about "
            "what the workload would compute."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="A published launch profile.")
    parser.add_argument(
        "--lane-module",
        required=True,
        help="The lane's module filename, e.g. public_scene_simready_isaac_vast.py",
    )
    parser.add_argument("--lane", help="Lane name recorded in the artifact manifest.")
    parser.add_argument("--receipt-out")
    args = parser.parse_args(argv)

    try:
        receipt = rehearse_lane_terminal_contract(
            profile_path=args.profile, lane_module=args.lane_module, lane=args.lane
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    if args.receipt_out:
        out = Path(args.receipt_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=1, sort_keys=True))
    return 0 if receipt["status"] == "would_pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
