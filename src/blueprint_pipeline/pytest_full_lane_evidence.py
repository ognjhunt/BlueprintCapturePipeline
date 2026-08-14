"""Pytest plugin that records the exact collected node IDs for full-lane evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "blueprint_full_lane_collection.v1"
OUTPUT_ENV = "BLUEPRINT_FULL_LANE_COLLECTION_MANIFEST"
PHASE_ENV = "BLUEPRINT_FULL_LANE_EVIDENCE_PHASE"
NODEID_PROPERTY = "blueprint_nodeid"


def nodeids_sha256(nodeids: Iterable[str]) -> str:
    canonical = "\n".join(nodeids).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_manifest(items: Iterable[Any], *, phase: str) -> dict[str, Any]:
    # xdist completes tests out of collection order.  Collection identity is a
    # set property, so canonicalize it rather than binding release evidence to
    # worker scheduling order.
    nodeids = sorted(str(item.nodeid) for item in items)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "test_count": len(nodeids),
        "nodeids_sha256": nodeids_sha256(nodeids),
        "nodeids": nodeids,
    }


def pytest_collection_modifyitems(items: Iterable[Any]) -> None:
    """Bind each JUnit testcase to its exact pytest node ID.

    Pytest's JUnit classname/name projection is not reversible for every class,
    parametrization, or custom collector. A dedicated property prevents a
    same-count, different-testcase artifact from qualifying as canonical.
    """

    for item in items:
        properties = list(getattr(item, "user_properties", []))
        if not any(name == NODEID_PROPERTY for name, _value in properties):
            item.user_properties.append((NODEID_PROPERTY, str(item.nodeid)))


def pytest_collection_finish(session: Any) -> None:
    worker_input = getattr(session.config, "workerinput", None)
    if isinstance(worker_input, dict) and worker_input.get("workerid") != "gw0":
        # Every xdist worker collects the same tests.  One designated worker
        # writes the executed manifest so concurrent writers cannot corrupt it.
        return
    output = str(os.environ.get(OUTPUT_ENV) or "").strip()
    if not output:
        return
    phase = str(os.environ.get(PHASE_ENV) or "").strip()
    if phase not in {"planned", "executed"}:
        raise RuntimeError(f"{PHASE_ENV} must be 'planned' or 'executed'")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_manifest(session.items, phase=phase), indent=2) + "\n",
        encoding="utf-8",
    )
