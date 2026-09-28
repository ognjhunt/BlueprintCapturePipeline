"""Publish the facts a self-serve robot-team run needs from this Pipeline.

A robot team buys a screening run in Blueprint-WebApp. Before it can pay, the
WebApp prepares an exact Task Evaluation Run request (the "payment setup
record"), and three of its facts are known only here:

- the capture root on the executor's partition (``agent_run_executor`` refuses
  any other root);
- the one scenario the scene's episode specs run;
- how many episodes those specs hold (the executor refuses a quote that
  differs).

This module reads those facts from ``pipeline/simulation_automation/
episode_specs.json`` once it exists and posts them over the existing signed
website control transport. It publishes facts, never authority: the WebApp
still requires a verified buyer account, cleared rights and a funded hold, and
the executor re-checks the root and the count before it claims a run. A capture
whose specs are missing, empty, or span more than one scenario publishes
nothing, so no self-serve run can be bought against it.

Publishing is best-effort and never fails the simulation lane: a missing offer
fails closed on the WebApp side (the plan stays unpayable).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .website_capture_entry import is_website_entry_source

OFFER_SCHEMA_VERSION = "blueprint.agent_execution_offer.v1"
EPISODE_SPECS_RELATIVE = Path("pipeline") / "simulation_automation" / "episode_specs.json"
PUBLICATION_RELATIVE = Path("pipeline") / "agent_execution_offer_publication.json"


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def build_agent_execution_offer(capture_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The offer for this capture, or ``(None, reason)`` when it cannot be made."""
    root = Path(capture_root).resolve()
    descriptor = _read_object(root / "capture_descriptor.json")
    if descriptor is None:
        return None, "capture_descriptor_missing"
    scene_id = str(descriptor.get("scene_id") or "").strip()
    capture_id = str(descriptor.get("capture_id") or "").strip()
    if not scene_id or not capture_id or root.name != capture_id or root.parent.parent.name != scene_id:
        return None, "capture_root_identity_mismatch"
    specs_path = root / EPISODE_SPECS_RELATIVE
    if not specs_path.is_file():
        return None, "episode_specs_missing"
    raw = specs_path.read_bytes()
    try:
        specs = json.loads(raw)
    except ValueError:
        return None, "episode_specs_invalid"
    episodes = specs.get("episodes") if isinstance(specs, Mapping) else None
    count = specs.get("episode_count") if isinstance(specs, Mapping) else None
    if not isinstance(episodes, list) or not isinstance(count, int) or isinstance(count, bool):
        return None, "episode_specs_invalid"
    if count <= 0 or count != len(episodes):
        return None, "episode_specs_count_invalid"
    scenarios = {
        str(episode.get("scenario_id") or "").strip()
        for episode in episodes
        if isinstance(episode, Mapping)
    }
    scenarios.discard("")
    if len(scenarios) != 1:
        # The executor admits exactly one scenario per run.
        return None, "episode_specs_scenario_not_single"
    return {
        "schema_version": OFFER_SCHEMA_VERSION,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "capture_root": str(root),
        "scenario_id": scenarios.pop(),
        "episode_count": count,
        "episode_specs_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }, None


def publish_agent_execution_offer(capture_root: Path, *, transport=None) -> dict[str, Any]:
    """Post the offer for a website capture. Never raises; records the outcome."""
    root = Path(capture_root).resolve()
    descriptor = _read_object(root / "capture_descriptor.json") or {}
    metadata = descriptor.get("metadata") if isinstance(descriptor.get("metadata"), Mapping) else {}
    request_id = str(descriptor.get("site_submission_id") or metadata.get("site_submission_id") or "").strip()
    if not is_website_entry_source(metadata.get("capture_entry_source")) or not request_id:
        result: dict[str, Any] = {"status": "skipped", "reason": "not_a_website_capture"}
    else:
        offer, reason = build_agent_execution_offer(root)
        if offer is None:
            result = {"status": "skipped", "reason": reason}
        else:
            if transport is None:
                from .website_task_context import website_webapp_request as transport
            try:
                transport(
                    capture_id=offer["capture_id"],
                    operation="agent-execution-offer",
                    payload={"request_id": request_id, "scene_id": offer["scene_id"], "offer": offer},
                )
                result = {"status": "published", "offer": offer}
            except Exception as exc:  # noqa: BLE001 - publishing must not fail the lane
                result = {"status": "failed", "reason": str(exc)[:500], "offer": offer}
    try:
        (root / PUBLICATION_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
        (root / PUBLICATION_RELATIVE).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass
    return result
