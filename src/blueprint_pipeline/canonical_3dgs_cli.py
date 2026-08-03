"""Unified module CLI for the canonical 3DGS production lane."""

from __future__ import annotations

import sys
from typing import Callable, Sequence

from .canonical_3dgs_admission import main as admission_main
from .canonical_3dgs_evaluation import main as evaluation_main
from .canonical_3dgs_execution_request import main as execution_request_main
from .canonical_3dgs_pipeline import finalize_main, main as prepare_main
from .canonical_3dgs_registration import main as registration_main
from .canonical_3dgs_transport import main as transport_main
from .canonical_3dgs_worker import main as worker_main


Command = Callable[[Sequence[str] | None], int]
_COMMANDS: dict[str, Command] = {
    "admit-worker": admission_main,
    "evaluate": evaluation_main,
    "finalize": finalize_main,
    "prepare": prepare_main,
    "register": registration_main,
    "request-execution": execution_request_main,
    "run-arm": worker_main,
    "transport": transport_main,
}


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv if argv is not None else sys.argv[1:])
    if not arguments or arguments[0] in {"-h", "--help"}:
        print("usage: python -m blueprint_pipeline.canonical_3dgs_cli <command> [arguments]")
        print("commands: " + ", ".join(sorted(_COMMANDS)))
        return 0 if arguments else 2
    command = arguments.pop(0)
    target = _COMMANDS.get(command)
    if target is None:
        print(f"unknown canonical 3DGS command: {command}", file=sys.stderr)
        return 2
    return target(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
