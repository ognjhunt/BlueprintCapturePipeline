#!/usr/bin/env python3
"""Ask the container what it actually has, before building against a guess.

Two routes to putting a robot in front of the articulated twin, and which one
is viable depends on facts about the image that cannot be checked from a
laptop: whether a Franka USD ships on disk, and whether isaaclab and Arena are
importable. Building either route speculatively risks a substantial amount of
untestable code against the wrong assumption.

So this asks. It deliberately does not start Isaac - no SimulationApp, no
four-minute boot - because everything it needs is a filesystem walk and a
handful of imports. That makes it the cheapest launch in the lane, and its
whole output is a map of what is present.

Nothing here fails the run. A missing Franka and an absent Arena are both
valid answers, and reporting them is the point; a probe that raised on the
first absence would answer half the question and cost a second launch for the
other half.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


RESULT_SCHEMA_VERSION = "adp009d_isaac_capability_probe.v1"
CANDIDATE_MODULES = (
    "isaacsim",
    "isaaclab",
    "isaaclab_arena",
    "isaacsim.core.prims",
    "isaacsim.storage.native",
)
ROBOT_NAME_HINTS = ("franka", "panda")
SEARCH_ROOTS = (
    "/isaac-sim/assets",
    "/isaac-sim/data",
    "/isaac-sim/exts",
    "/root/.local/share/ov",
    "/isaac-sim",
)
MAX_MATCHES = 60
MAX_WALK_ENTRIES = 400_000


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _persist(path: Path, value: dict[str, Any]) -> None:
    value["result_digest"] = _canonical_digest(value, field="result_digest")
    value["_canonical_digest"] = value["result_digest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _module_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in CANDIDATE_MODULES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError) as exc:
            report[name] = {"importable": False, "note": type(exc).__name__}
            continue
        report[name] = {
            "importable": spec is not None,
            "origin": getattr(spec, "origin", None) if spec else None,
        }
    return report


def _robot_assets() -> dict[str, Any]:
    """Walk for USD files whose path mentions a Franka, bounded."""

    matches: list[str] = []
    visited = 0
    truncated = False
    for root in SEARCH_ROOTS:
        base = Path(root)
        if not base.is_dir():
            continue
        for current, directories, files in os.walk(base, followlinks=False):
            visited += 1
            if visited > MAX_WALK_ENTRIES:
                truncated = True
                break
            # Asset trees are enormous; skip the obviously irrelevant.
            directories[:] = [
                d for d in directories if d not in {".git", "__pycache__", "cache"}
            ]
            lowered = current.lower()
            for name in files:
                if not name.endswith((".usd", ".usda", ".usdc", ".usdz")):
                    continue
                candidate = f"{lowered}/{name.lower()}"
                if any(hint in candidate for hint in ROBOT_NAME_HINTS) and not any(
                    marker in candidate
                    for marker in ("/data/tests/", "/tests/", "/unittests/", "/extscache/")
                ):
                    matches.append(str(Path(current) / name))
                    if len(matches) >= MAX_MATCHES:
                        truncated = True
                        break
            if len(matches) >= MAX_MATCHES:
                break
        if len(matches) >= MAX_MATCHES:
            break
    return {
        "match_count": len(matches),
        "matches": sorted(matches),
        "search_roots": list(SEARCH_ROOTS),
        "truncated": truncated,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=False)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(arguments.output).expanduser().resolve()

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "blockers": [],
        # This probe never starts Isaac, so it cannot claim to have executed it.
        "native_isaac_executed": False,
        "isaac_started": False,
        "physical_success_established": False,
        "provider_zero_required_after_return": True,
        "probe_results": [],
    }
    try:
        modules = _module_report()
        assets = _robot_assets()
        result["modules"] = modules
        result["robot_assets"] = assets
        result["arena_available"] = bool(
            modules.get("isaaclab_arena", {}).get("importable")
        ) and bool(modules.get("isaaclab", {}).get("importable"))
        result["robot_asset_available"] = assets["match_count"] > 0
        # Both answers are informative; neither is a failure of the probe.
        result["viable_route"] = (
            "arena_composition"
            if result["arena_available"]
            else "raw_usd_composition"
            if result["robot_asset_available"]
            else "neither_on_this_image"
        )
        result["probe_results"] = [
            {"name": "module_inventory", "passed": True},
            {"name": "robot_asset_inventory", "passed": True},
        ]
        result["claim_boundary"] = {
            "inventory_only_nothing_was_simulated": True,
            "importability_is_not_a_working_arena": True,
            "no_network_asset_root_consulted": True,
        }
    except BaseException as exc:  # noqa: BLE001
        result["status"] = "blocked"
        result["blockers"].append(
            f"isaac_capability_probe_failed:{type(exc).__name__}:{exc}"
        )

    _persist(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
