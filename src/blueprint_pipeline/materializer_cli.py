"""One flag-table-to-materializer skeleton, shared by the entry-point scripts.

`tests/test_materializer_reachability.py` counts the `materialize_*` functions
no command line can reach, and the fix for each is the same shape: a table
mapping the materializer's keyword-only parameters to flags, so the table *is*
the call and a keyword with no flag fails a test instead of failing an operator.

The first such table was written inline in `scripts/prepare_artifixer3d_inputs`.
Copying it per lane would reproduce the defect
`tests/test_live_profile_builder_contract.py` exists to prevent -- a control
surface with one definition per caller drifts, and the drift is only found when
somebody runs the lane. So the skeleton lives here and the scripts declare only
their steps.

Nothing here reads a provider, allocates, or mutates: `run` calls one
materializer with retained bytes and reports what it wrote.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable


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


#: Receipt keys worth echoing so an operator can chain steps without reopening
#: the file. `status` is renamed because the summary carries one of its own.
SUMMARY_KEYS = (
    "schema_version",
    "status",
    "receipt_digest",
    "receipt_path",
    "output_root",
)


def build_parser(
    steps: Mapping[str, Step], *, description: str | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    sub = parser.add_subparsers(dest="step", required=True)
    for name, step in steps.items():
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


def run(
    steps: Mapping[str, Step],
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> int:
    """Call one step's materializer; report `blocked` rather than raising."""

    args = build_parser(steps, description=description).parse_args(argv)
    step = steps[args.step]

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
        for key in SUMMARY_KEYS:
            if key in receipt:
                summary[f"receipt_{key}" if key == "status" else key] = receipt[key]
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0
