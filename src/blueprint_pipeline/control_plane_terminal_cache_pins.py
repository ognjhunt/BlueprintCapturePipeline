"""Release obsolete cache pins from verified archived-run evidence.

The collector previously retained already-archived runs for the pin's full
30-day TTL. This reconciliation changes only the cache ledger; the normal
collector separately rechecks references and removes reproducible directories.
"""
from __future__ import annotations

import json
from pathlib import Path

from .control_plane_storage_pins import load_storage_pins, release_storage_pin
from .decision_evidence_contracts import canonical_digest
from .completed_replay_cache_retention import active_reference
from .control_plane_storage_roots import require_storage_class

def _read(path):
    if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
            or path.stat().st_size > 16 * 1024**2):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _closed_proof(pin, evidence_roots):
    owner, kind = pin["owner_id"], pin["kind"]
    if kind != "activation":
        return None
    for root in evidence_roots:
        root = Path(root)
        path = root / (owner + ".offloaded.v1.json")
        value = _read(path)
        if (value is None or (root / owner).exists()
                or value.get("schema_version") != "control_plane_evidence_offload_pointer.v1"
                or value.get("pointer_digest") != canonical_digest(value, digest_field="pointer_digest")
                or value.get("status") != "offloaded" or value.get("directory") != owner
                or value.get("evidence_deleted") is not False
                or not str(value.get("uri", "")).startswith("s3://blueprint-task-evaluation-artifacts-prod/")
                or value.get("terminal_receipt") not in {"dispatch_receipt.json", "launch_receipt.json", "abandoned_idle"}
                or type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0):
            continue
        return {"kind": "archived_run", "path": str(path), "pointer_digest": value["pointer_digest"],
                "terminal_receipt": value["terminal_receipt"], "archive_digest": value["digest"]}
    return None


def reconcile_terminal_cache_pins(*, pins_root, queue_roots, evidence_roots, now, apply=False,
                                  reference_checker=active_reference, classifier=require_storage_class):
    from .control_plane_storage_gc import _queue_reference_text
    pins_root = Path(pins_root)
    for root in evidence_roots:
        classifier(str(root), expected="evidence_cold", code="terminal_cache_pin_evidence_root_invalid")
    pins = {(p["kind"], p["owner_id"]): p for p in load_storage_pins(pins_root, now=lambda: now) if p["status"] == "live"}
    queue_text = _queue_reference_text(queue_roots)
    candidates, kept, released = [], [], []
    for identity, pin in pins.items():
        proof = _closed_proof(pin, evidence_roots)
        if proof is None or now - pin["created_at_epoch"] < 6 * 3600:
            continue
        for path in pin["paths"]:
            classifier(str(path), expected="cache", code="terminal_cache_pin_cache_root_invalid")
        # Check the entire dependency closure before releasing a parent pin.
        closure, pending = {}, [identity]
        while pending:
            key = pending.pop()
            if key in closure or key not in pins:
                continue
            closure[key] = pins[key]
            pending.extend((d["kind"], d["owner_id"]) for d in pins[key].get("depends_on", []))
        if any(p["owner_id"] in queue_text or any(reference_checker(Path(path)) for path in p["paths"])
               for p in closure.values()):
            kept.append({"kind": pin["kind"], "owner_id": pin["owner_id"], "reason": "active_reference"})
            continue
        if any(any((d["kind"], d["owner_id"]) == identity for d in other.get("depends_on", []))
               for key, other in pins.items() if key not in closure):
            continue
        candidate = {"kind": pin["kind"], "owner_id": pin["owner_id"], "proof": proof}
        candidates.append(candidate)
        if apply:
            # Re-read live queue references and the proof at the mutation edge.
            fresh = _queue_reference_text(queue_roots)
            if (proof != _closed_proof(pin, evidence_roots)
                    or any(p["owner_id"] in fresh or any(reference_checker(Path(path)) for path in p["paths"])
                           for p in closure.values())):
                kept.append({**candidate, "reason": "reference_changed"})
                continue
            released.append(release_storage_pin(pins_root=pins_root, kind=pin["kind"],
                                                 owner_id=pin["owner_id"], now=lambda: now))
    return {"schema_version": "control_plane_terminal_cache_pin_reconciliation.v1",
        "status": "applied" if apply else "dry_run", "candidates": candidates, "released": released,
        "kept": kept, "cache_or_evidence_bytes_removed": False}
