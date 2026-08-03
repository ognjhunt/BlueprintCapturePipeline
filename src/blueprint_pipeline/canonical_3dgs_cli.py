"""Single module command surface for canonical 3DGS operations."""

from __future__ import annotations

import importlib
import sys
from typing import Callable, Sequence


COMMANDS: dict[str, tuple[str, str]] = {
    "prepare": ("blueprint_pipeline.canonical_3dgs_pipeline", "main"),
    "run-arm": ("blueprint_pipeline.canonical_3dgs_worker", "main"),
    "finalize": ("blueprint_pipeline.canonical_3dgs_pipeline", "finalize_main"),
    "transport": ("blueprint_pipeline.canonical_3dgs_transport", "main"),
    "admit-worker": ("blueprint_pipeline.canonical_3dgs_admission", "main"),
    "request-execution": (
        "blueprint_pipeline.canonical_3dgs_execution_request",
        "main",
    ),
    "evaluate": ("blueprint_pipeline.canonical_3dgs_evaluation", "main"),
    "register": ("blueprint_pipeline.canonical_3dgs_registration", "main"),
}


def _usage() -> str:
    commands = "|".join(COMMANDS)
    return f"usage: python -m blueprint_pipeline.canonical_3dgs_cli <{commands}> [arguments]"


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    command = arguments.pop(0)
    target = COMMANDS.get(command)
    if target is None:
        print(_usage(), file=sys.stderr)
        return 64
    module = importlib.import_module(target[0])
    handler: Callable[[Sequence[str] | None], int] = getattr(module, target[1])
    return handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["COMMANDS", "main"]
