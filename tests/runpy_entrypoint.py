from __future__ import annotations

import runpy
import sys
from typing import Any


_MISSING = object()


def run_module_as_main(module_name: str) -> dict[str, Any]:
    """Run an already-imported module as ``__main__`` without runpy re-exec warnings."""

    previous = sys.modules.pop(module_name, _MISSING)
    try:
        return runpy.run_module(module_name, run_name="__main__")
    finally:
        if previous is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
