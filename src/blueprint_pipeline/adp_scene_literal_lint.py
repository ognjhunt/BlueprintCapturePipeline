"""Prevent historical scene literals from leaking into reusable ADP modules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


FORBIDDEN_HISTORICAL_LITERALS = re.compile(r"840313|ins160|canned_beverage")
# These modules are explicitly the retained first-rehearsal implementation or
# compatibility adapters. Adding a new path here requires conscious review;
# all other production modules must receive scene/task identity as data.
HISTORICAL_FIRST_FIXTURE_IMPLEMENTATIONS = frozenset(
    {
        "adp009d_isaac_runtime.py",
        "adp009d_native_microcheck_bundle.py",
        "adp009d_sage_franka_placement.py",
        "adp009d_840313_runtime_bundle.py",
        "adp009d_live_readiness.py",
        "adp_content_agents_bundle_preflight.py",
        "adp_content_agents_vast.py",
        "adp_inpaint360_interiorgs_vast.py",
        "public_scene_hybrid_replacement_seal.py",
        "public_scene_inpaint360_adapter.py",
        "public_scene_simready_control.py",
        "public_scene_simready_isaac_bundle.py",
        "public_scene_simready_native.py",
        "public_scene_simready_replacement.py",
        "public_scene_suite_materializer.py",
        "vast_provider_adapter.py",
    }
)


def scan_scene_literal_violations(source_root: str | Path) -> list[dict[str, Any]]:
    """Return forbidden literal locations outside explicit fixture modules."""

    root = Path(source_root).expanduser().resolve()
    violations = []
    for path in sorted(root.rglob("*.py")):
        if path.name in HISTORICAL_FIRST_FIXTURE_IMPLEMENTATIONS or path.name == Path(
            __file__
        ).name:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            matches = sorted(set(FORBIDDEN_HISTORICAL_LITERALS.findall(line)))
            if matches:
                violations.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "line_number": line_number,
                        "literals": matches,
                    }
                )
    return violations


__all__ = [
    "HISTORICAL_FIRST_FIXTURE_IMPLEMENTATIONS",
    "scan_scene_literal_violations",
]
