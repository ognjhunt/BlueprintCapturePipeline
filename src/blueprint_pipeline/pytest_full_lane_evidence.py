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
SHARD_PLAN_ENV = "BLUEPRINT_FULL_LANE_SHARD_PLAN"
SHARD_INDEX_ENV = "BLUEPRINT_FULL_LANE_SHARD_INDEX"
SHARD_PLAN_SCHEMA = "blueprint.full_lane_shard_plan.v2"


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


def _foreign_split_nodeids() -> set[str] | None:
    """Node IDs the shard plan assigns to other shards, or ``None`` outside a shard run.

    A file too heavy for one shard is planned by node ID, so each shard's pytest
    collects that whole file and drops only the tests another shard owns. Only
    planned node IDs are ever dropped: a test the plan never saw still runs and
    fails the shard's exact-collection verification.
    """
    plan_path = str(os.environ.get(SHARD_PLAN_ENV) or "").strip()
    raw_index = str(os.environ.get(SHARD_INDEX_ENV) or "").strip()
    if not plan_path and not raw_index:
        return None
    try:
        index = int(raw_index)
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        if not isinstance(plan, dict) or plan.get("schema_version") != SHARD_PLAN_SCHEMA:
            raise ValueError("schema")
        shards = plan["shards"]
        if [shard.get("index") for shard in shards].count(index) != 1:
            raise ValueError("index")
        foreign: set[str] = set()
        for shard in shards:
            if shard.get("index") != index:
                foreign.update(str(nodeid) for nodeid in shard["nodeids"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        import pytest

        raise pytest.UsageError(
            f"full-lane shard selection is unusable ({SHARD_PLAN_ENV}={plan_path!r}, "
            f"{SHARD_INDEX_ENV}={raw_index!r}): {exc}"
        ) from exc
    return foreign


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Bind each JUnit testcase to its exact pytest node ID.

    Pytest's JUnit classname/name projection is not reversible for every class,
    parametrization, or custom collector. A dedicated property prevents a
    same-count, different-testcase artifact from qualifying as canonical.
    """

    foreign = _foreign_split_nodeids()
    if foreign:
        kept = [item for item in items if str(item.nodeid) not in foreign]
        deselected = [item for item in items if str(item.nodeid) in foreign]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = kept
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
