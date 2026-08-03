#!/usr/bin/env python3
"""Backfill the append-only Postshot attempt ledger for attempts 1-4.

Reads only the immutable evidence already present under
``provider_packets/postshot/<run_id>/`` plus the frozen execution packet, and
writes ``postshot_attempt_ledger.v1.json`` beside the attempt roots.  Each
entry separates exact observed evidence from inference and never invents
timestamps, progress, GPU activity, or billing totals.  Attempt 5+ entries are
appended by the launcher, never by rewriting history here: on rerun this
script refuses to change an existing ledger whose attempts disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint_pipeline.postshot_worker_contracts import (  # noqa: E402
    build_attempt_ledger,
    sha256_file,
    utc_now_iso,
)

HISTORICAL_BAKEOFF_BUDGET_USD = 250.0
HISTORICAL_POSTSHOT_SPEND_ESTIMATE_USD = 3.80


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_inventory(run_dir: Path) -> list[dict]:
    inventory = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        entry = {"relative_path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size}
        # presigned_urls.json holds live-credential URLs; reference it without a
        # digest so the ledger never tempts anyone to reproduce its contents.
        if path.name != "presigned_urls.json":
            entry["digest"] = sha256_file(path)
        inventory.append(entry)
    return inventory


def build_attempts(postshot_root: Path) -> list[dict]:
    attempts: list[dict] = []
    common = {
        "evidence_root_note": "immutable attempt root; append-only, never rewritten",
        "cost_status": "unreconciled_live_estimate_only",
        "billing_reconciliation": "required_not_observed",
    }

    def base(number: int, run_id: str, classification: str) -> dict:
        run_dir = postshot_root / run_id
        record: dict = {
            "attempt": number,
            "run_id": run_id,
            "evidence_root": f"provider_packets/postshot/{run_id}",
            "classification": classification,
            "evidence_files": _evidence_inventory(run_dir),
            **common,
        }
        staging_path = run_dir / "staging.json"
        if staging_path.exists():
            staging = _load(staging_path)
            record["staging_digest"] = staging.get("staging_digest")
            record["dataset_digest"] = staging.get("dataset_digest")
            record["installer_digest"] = staging.get("installer_digest")
        launch_path = run_dir / "launch.json"
        if launch_path.exists():
            launch = _load(launch_path)
            record["instance_id"] = launch.get("instance_id")
            record["instance_type"] = launch.get("instance_type")
            record["launched_at_epoch"] = launch.get("launched_at_epoch")
            record["ttl_deadline_epoch"] = launch.get("ttl_deadline_epoch")
        teardown_path = run_dir / "teardown_proof.json"
        if teardown_path.exists():
            proof = _load(teardown_path)
            record["teardown_proof"] = {
                "schema_version": proof.get("schema_version"),
                "checked_at_epoch": proof.get("checked_at_epoch"),
                "provider_zero": proof.get("provider_zero"),
                "non_terminated_instances": proof.get("non_terminated_instances"),
            }
        else:
            record["teardown_proof"] = None
            record["teardown_proof_gap"] = (
                "no per-run teardown proof was recorded at attempt time; "
                "EC2 zero for this account/region was independently confirmed on 2026-08-01 "
                "during phase A verification (see phase_a_state_verification.json)"
            )
        receipt_path = run_dir / "deletion_receipt.json"
        if receipt_path.exists():
            receipt = _load(receipt_path)
            record["staged_object_deletion"] = {
                "receipt": "deletion_receipt.json",
                "all_absent_verified": receipt.get("all_absent_verified"),
                "checked_at_utc": receipt.get("checked_at_utc"),
            }
        return record

    a1 = base(1, "postshot-20260801T195240Z", "infrastructure_bootstrap_failure")
    a1["observed"] = [
        "NVIDIA driver install succeeded (status.log: DRIVER ok: NVIDIA A10G, 566.03 at 2026-08-01T20:07:46Z)",
        "last status line is 'POSTSHOT downloading installer' (2026-08-01T20:07:46Z); no later status was uploaded",
        "staged installer was the vendor bundle exe (sha256 99fc687c...)",
        "teardown check at epoch 1785618204 still saw i-0dda3d1290611be90 non-terminated (provider_zero=false at that check)",
    ]
    a1["inference"] = [
        "vendor Postshot bootstrapper became silent after driver install (inferred from missing status lines and reproduced by attempt 2)",
        "local watcher/session did not provide durable monitoring; instance was manually terminated",
    ]
    a1["not_evidence_of"] = "reconstruction quality or training behavior"

    a2 = base(2, "postshot-20260801T210457Z", "vendor_wrapper_incompatibility")
    a2["observed"] = [
        "driver ok (NVIDIA L4, 566.03) at 2026-08-01T21:14:51Z",
        "bundle exe downloaded with sha256_ok=True",
        "WiX Burn v6.0.0 (ReactionsBA managed bootstrapper) initialized but never began package installation (install log tail embedded in status.log)",
        "FATAL postshot_install_timed_out_after_900s at 2026-08-01T21:30:29Z; worker self-terminated via Fail-And-Stop",
    ]
    a2["inference"] = [
        "the vendor bundle's managed bootstrapper hangs in /quiet on bare Windows Server 2022 (reproduces attempt 1)",
    ]
    a2["not_evidence_of"] = "reconstruction quality or training behavior"

    a3 = base(3, "postshot-20260801T213413Z", "cli_invocation_failure")
    a3["observed"] = [
        "carved MSI (sha256 70d4c35d...) installed via msiexec with exit=0 at 2026-08-01T21:45:00Z",
        "postshot-cli.exe discovered at the expected path",
        "P1 exited 109 in 0 seconds; P2 exited 109 in 0 seconds (status.log + per-arm receipt.txt)",
        "captured --help proves --login/--password are global options that must precede the train subcommand",
        "results.zip (110290 bytes) retained; local and remote digests matched during phase A verification",
    ]
    a3["inference"] = [
        "exit 109 was caused by placing login flags after the train subcommand (fixed in commit 72d3942ae)",
    ]
    a3["not_evidence_of"] = "model or training behavior; no training occurred"

    a4 = base(4, "postshot-20260801T215521Z", "monitoring_evidence_unavailable")
    a4["observed"] = [
        "carved MSI installed, exit=0 at 2026-08-01T22:05:32Z on NVIDIA L4 driver 566.03",
        "dataset downloaded; 'TRAIN P1_splat3 starting (profile=Splat3)' at 2026-08-01T22:06:17Z is the final status line",
        "heartbeat contained only a PowerShell transcript header for the rest of the run (last upload 2026-08-01T23:14:52Z)",
        "no Postshot log growth, process state, GPU utilization/memory, progress counter, or output growth was externally visible",
        "run manually terminated; teardown proof recorded non_terminated=[] (provider_zero=true) at epoch 1785626659",
        "instance i-037c684e4a5384bd1 terminated and volume deleted; EC2 API re-verified zero during phase A on 2026-08-01",
    ]
    a4["inference"] = [
        "corrected global login placement allowed P1 to start; nothing observable distinguishes healthy training from a hang",
    ]
    a4["not_evidence_of"] = "training success OR training failure; the run was killed for unobservability"

    attempts.extend([a1, a2, a3, a4])
    return attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", required=True, type=Path)
    arguments = parser.parse_args()
    postshot_root = arguments.proxy_root.resolve() / "provider_packets" / "postshot"
    attempts = build_attempts(postshot_root)
    ledger = build_attempt_ledger(
        attempts=attempts,
        historical_bakeoff_budget_usd=HISTORICAL_BAKEOFF_BUDGET_USD,
        historical_postshot_spend_estimate_usd=HISTORICAL_POSTSHOT_SPEND_ESTIMATE_USD,
        generated_at_utc=utc_now_iso(),
    )
    out_path = postshot_root / "postshot_attempt_ledger.v1.json"
    if out_path.exists():
        existing = _load(out_path)
        if existing.get("attempts") != ledger["attempts"]:
            raise SystemExit(
                "refusing to rewrite ledger with different attempt content; append via the launcher instead"
            )
        print(f"ledger already current: {out_path}")
        return 0
    out_path.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ledger": str(out_path), "attempt_count": len(attempts), "ledger_digest": ledger["ledger_digest"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
