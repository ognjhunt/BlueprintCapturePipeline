#!/usr/bin/env python3
"""Materialize the ArtiFixer3D input chain from a command line.

The appearance path's head needs a candidate-inputs receipt before anything
downstream exists, and the four functions that produce one could be called from
no script and from no module carrying a `main()`. So the chain dead-ended at its
first step: the bundle CLI asks for a candidate receipt, and nothing in a
production path could make one.

That is the same defect as #512 (lanes), #520 (bundle modules) and #523
(authority materializers), in a fourth scope. It is also not rare -- 77 of the
161 public `materialize_*` functions in `src/blueprint_pipeline` are reachable
from nothing, which is why `tests/test_materializer_reachability.py` now
rediscovers that set rather than trusting a list.

The order the steps run in:

    calibrated residual preflight
        -> object-absent reference receipt   (one per task, from edited frames)
        -> candidate inputs                  (the receipt the bundle consumes)
        -> whole-frame semantic teacher      (one per task, from teacher frames)
        -> dual-target inputs                (the paired receipt)

`editor-identity` is a JSON file rather than flags because it records *which
model produced the frames*, and that provenance belongs in one signed object
rather than being reassembled from arguments at each call site.

Reads retained bytes only; performs no provider mutation and rents nothing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from blueprint_pipeline.public_scene_artifixer3d_candidate_inputs import (
    materialize_artifixer3d_candidate_inputs,
    materialize_object_absent_reference_candidate_receipt,
)
from blueprint_pipeline.public_scene_artifixer3d_dual_target_inputs import (
    materialize_dual_target_artifixer3d_inputs,
    materialize_whole_frame_semantic_teacher_receipt,
)


@dataclass(frozen=True)
class Param:
    """One materializer keyword and the flag that supplies it."""

    flag: str
    help: str = ""
    required: bool = False
    type: Callable[[str], Any] | None = None
    default: Any = None
    accumulate: bool = False
    #: Read the flag's value as JSON from a file rather than as a string.
    json_file: bool = False


@dataclass(frozen=True)
class Step:
    summary: str
    materialize: Callable[..., Any]
    params: Mapping[str, Param] = field(default_factory=dict)


EDITOR_IDENTITY = Param(
    "--editor-identity",
    "JSON file naming the model that produced these frames.",
    required=True,
    json_file=True,
)

STEPS: dict[str, Step] = {
    "object-absent-reference": Step(
        "One task's object-absent reference receipt, from edited frames.",
        materialize_object_absent_reference_candidate_receipt,
        {
            "source_candidate_inputs_receipt_path": Param("--source-candidate-inputs", required=True),
            "task_id": Param("--task-id", required=True),
            "object_absent_frames_root": Param("--object-absent-frames-root", required=True),
            "editor_identity": EDITOR_IDENTITY,
            "prompt_policy": Param("--prompt-policy", required=True),
            "output_path": Param("--output", required=True),
        },
    ),
    "candidate-inputs": Step(
        "The candidate-inputs receipt the ArtiFixer3D bundle consumes.",
        materialize_artifixer3d_candidate_inputs,
        {
            "calibrated_residual_preflight_path": Param(
                "--calibrated-residual-preflight", required=True
            ),
            "output_root": Param("--output-root", required=True),
            "selected_task_ids": Param(
                "--task-id", "Repeatable; omit for every task.", accumulate=True
            ),
            "object_absent_reference_receipt_paths": Param(
                "--object-absent-reference", "Repeatable.", accumulate=True, default=()
            ),
        },
    ),
    "semantic-teacher": Step(
        "One task's whole-frame semantic teacher receipt.",
        materialize_whole_frame_semantic_teacher_receipt,
        {
            "source_candidate_inputs_receipt_path": Param("--source-candidate-inputs", required=True),
            "task_id": Param("--task-id", required=True),
            "semantic_teacher_frames_root": Param("--semantic-teacher-frames-root", required=True),
            "editor_identity": EDITOR_IDENTITY,
            "prompt_policy": Param("--prompt-policy", required=True),
            "output_path": Param("--output", required=True),
        },
    ),
    "dual-target": Step(
        "The paired receipt, from a candidate receipt plus teacher receipts.",
        materialize_dual_target_artifixer3d_inputs,
        {
            "source_candidate_inputs_receipt_path": Param("--source-candidate-inputs", required=True),
            "semantic_teacher_receipt_paths": Param(
                "--semantic-teacher-receipt", "Repeatable; one per task.", accumulate=True
            ),
            "output_root": Param("--output-root", required=True),
            "transition_radius_pixels": Param(
                "--transition-radius-pixels",
                "Width of the repair support band, in pixels.",
                required=True,
                type=int,
            ),
            "selected_task_ids": Param(
                "--task-id", "Repeatable; omit for every task.", accumulate=True
            ),
        },
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="step", required=True)
    for name, step in STEPS.items():
        target = sub.add_parser(name, help=step.summary)
        for keyword, param in step.params.items():
            options: dict[str, Any] = {"dest": keyword, "help": param.help or None}
            if param.accumulate:
                # `action="append"` appends to whatever default it is given, so
                # a tuple default raises `AttributeError` the first time the
                # flag is actually passed. Start from `None` and let
                # `call_arguments` restore the declared default.
                options["action"] = "append"
                options["default"] = None
            elif param.required:
                options["required"] = True
            else:
                options["default"] = param.default
            if param.type is not None:
                options["type"] = param.type
            target.add_argument(param.flag, **options)
    return parser


def call_arguments(step: Step, namespace: argparse.Namespace) -> dict[str, Any]:
    supplied = vars(namespace)
    arguments: dict[str, Any] = {}
    for keyword, param in step.params.items():
        value = supplied.get(keyword, param.default)
        if param.json_file and value is not None:
            value = json.loads(Path(str(value)).expanduser().read_text(encoding="utf-8"))
        elif param.accumulate:
            collected = tuple(value or ())
            # `None` means "every task" for the selectors; an empty tuple would
            # mean "no tasks", which silently produces an empty receipt.
            value = collected or (() if param.default == () else None)
        arguments[keyword] = value
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    step = STEPS[args.step]

    try:
        receipt = step.materialize(**call_arguments(step, args))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=1,
                sort_keys=True,
            )
        )
        return 2

    summary: dict[str, Any] = {
        "status": "materialized",
        "step": args.step,
        "provider_mutation_performed": False,
    }
    if isinstance(receipt, Mapping):
        for key in ("schema_version", "status", "receipt_digest", "receipt_path", "output_root"):
            if key in receipt:
                summary[f"receipt_{key}" if key == "status" else key] = receipt[key]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
