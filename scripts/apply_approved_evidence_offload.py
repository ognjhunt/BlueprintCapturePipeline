#!/usr/bin/env python3
"""Apply only the exact approved offload manifest, preserving newly protected evidence."""
import argparse
import json
import sys
import time
from pathlib import Path

# EnvironmentFile can override a transient unit's PYTHONPATH. This destructive
# entrypoint must import the implementation belonging to its pinned checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blueprint_pipeline.common import write_json
from blueprint_pipeline.control_plane_evidence_offload import (
    EXECUTE_ACK, _tree_snapshot, apply_evidence_offload,
)
from blueprint_pipeline.control_plane_storage_gc import _queue_reference_text
from blueprint_pipeline.control_plane_storage_pins import live_pinned_paths
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-manifest-digest", required=True)
    parser.add_argument("--pins-root", required=True)
    parser.add_argument("--queue-root", action="append", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--ack", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    manifest = report["evidence_offload"]
    if (report.get("apply") is not False
            or report.get("report_digest") != canonical_digest(report, digest_field="report_digest")
            or manifest.get("manifest_digest") != args.expected_manifest_digest
            or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")
            or args.ack != EXECUTE_ACK):
        raise ValueError("approved_offload_manifest_mismatch")
    def protected(path):
        pinned = live_pinned_paths(args.pins_root)
        if any(Path(p) == path or path in Path(p).parents or Path(p) in path.parents for p in pinned):
            return True
        if path.name in _queue_reference_text(args.queue_root):
            return True
        # A later write invalidates this approval even if a terminal filename survived.
        return _tree_snapshot(path)[0] > report["observed_at_epoch"]
    selected = dict(manifest)
    selected["approved_source_manifest_digest"] = manifest["manifest_digest"]
    selected["candidates"] = sorted(manifest["candidates"], key=lambda row: row["size_bytes"])
    selected["manifest_digest"] = canonical_digest(selected, digest_field="manifest_digest")
    result = apply_evidence_offload(selected, ack=args.ack, protection_checker=protected)
    result["approved_source_manifest_digest"] = manifest["manifest_digest"]
    result["completed_at_epoch"] = time.time()
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    write_json(Path(args.receipt_out), result)
    print(json.dumps({k: result[k] for k in ("status", "offloaded_count", "offloaded_bytes", "skipped")}))


if __name__ == "__main__":
    main()
