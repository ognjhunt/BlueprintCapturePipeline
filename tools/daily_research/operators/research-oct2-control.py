"""Scoped control migration using the installed, reviewed research adapters.

plan: Firestore GETs and private local files only. apply-disabled: root control
configuration only, under the existing fenced lease. No provider access.
"""
import argparse
import copy
import hashlib
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from tools.daily_research import render, search
from tools.daily_research.firestore import (
    Bridge,
    FirestoreLedger,
    control_configuration,
)
from tools.daily_research.runner import (
    Refusal,
    canonical,
    configuration,
    digest,
    read_json,
)

SOURCE = "35f5c9ad43f84aa053aa7616a63a9aa4f6e32a61"
ARCHIVE = "1aa932767fe9ec73c06ece6b5ba1e573027a636a3249363d62df7bf6415a3651"
INSTRUCTIONS = "84aeea7cec9eb6e20d0d2fba10dcb269a615174e48ed91a60ff7a6a83ca37938"
DAY = "2026-10-01"
FIRST = "2026-10-02"
SESSION = "sess_03449d612c6384f3006abe4b46ab20819aa8405df2844574d9"
ENVIRONMENT = "ccarenv_b64_Y2NhcmVudl82YWJlNGI0NmFlMmM4MTkxYmNkODEwMzYxNjdlNWQ0MA"
RAW = "ca1a1a1d341ea7954018645cd9a1126de7ac080200cb688ad4f0e4f0adbd4f3d"
BUDGET_AUTHORITY = "Sentinel_c30352f247c88191bbd695cc2bd99de1"


def normalized(control):
    return {k: v for k, v in control.items() if k != "lease"}


def verify_failed(row, raw):
    if (not row or row.get("date") != DAY or row.get("state") != "failed"
            or row.get("error") != "knowledge_schema_invalid" or row.get("session_id") != SESSION
            or row.get("environment_id") != ENVIRONMENT or row.get("raw_output_digest") != RAW
            or hashlib.sha256(raw).hexdigest() != RAW or row.get("qa") or row.get("delivery")):
        raise Refusal("oct1_failed_history_or_raw_changed")


def package_receipt(root, archive):
    root = Path(root).resolve()
    archive = Path(archive)
    if archive.stat().st_size != 409600 or hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE:
        raise Refusal("oct2_installed_archive_binding_mismatch")
    manifest = read_json(root / "manifest.json")
    with tarfile.open(archive) as bundle:
        if bundle.extractfile("manifest.json").read() != (root / "manifest.json").read_bytes():
            raise Refusal("oct2_installed_manifest_binding_mismatch")
    files = manifest.get("files", {})
    if manifest.get("source_commit") != SOURCE or len(files) != 41:
        raise Refusal("oct2_installed_manifest_binding_mismatch")
    for name, expected in files.items():
        original = root / name
        path = original.resolve()
        if (not path.is_relative_to(root) or not path.is_file() or original.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise Refusal("oct2_installed_file_binding_mismatch")
    # Verify the actual imported implementation, not just an unrelated directory.
    if Path(render.__file__).resolve() != root / "tools/daily_research/render.py":
        raise Refusal("oct2_imported_package_mismatch")
    return {"source_commit": SOURCE, "archive_sha256_reference": ARCHIVE,
            "manifest_digest": digest(manifest), "files_verified": len(files),
            "tool_definitions_digest": digest(search.tools()),
            "application_instructions_sha256": hashlib.sha256(search.instructions().encode()).hexdigest()}


def disabled_candidate(control):
    configuration(control_configuration(control))
    candidate = copy.deepcopy(normalized(control))
    candidate["enabled"] = False
    candidate["source_commit"] = SOURCE
    candidate["config"].update(
        enabled=False, research_contract_version=3,
        search_provider=search.PROFILE, discovery_profile="adaptive-sites-v1",
        max_runtime_seconds=1800, qa_reserved_seconds=600,
        soft_target_usd=5, recurring_budget_authority_reference=BUDGET_AUTHORITY,
        expected_agent_instructions_sha256=INSTRUCTIONS)
    configuration(control_configuration(candidate))
    return candidate


def snapshot(bridge):
    ledger = FirestoreLedger(bridge)
    control = bridge.call("control")
    row = ledger.get(DAY)
    raw = ledger.read_bytes(DAY + "-artifact.json")
    verify_failed(row, raw)
    summary = bridge.call("summary")
    if summary.get("unfinished") or bridge.call("active_qa"):
        raise Refusal("oct2_active_research_or_qa_unreconciled")
    return control, row, summary


def plan(bridge, receipt, now=None):
    control, row, summary = snapshot(bridge)
    candidate = disabled_candidate(control)
    result = {"schema_version": "blueprint.oct2-disabled-migration.v1", "state": "prepared_disabled",
              "observed_at": (now or datetime.now(timezone.utc)).isoformat(),
              "package": receipt, "expected_control_digest": digest(normalized(control)),
              "oct1_row_digest": digest(row), "oct1_raw_sha256": RAW,
              "oct1_cleanup_required": row.get("cleanup_required"), "summary": summary,
              "candidate": candidate, "candidate_digest": digest(candidate),
              "research_seconds": 1200, "qa_reserved_seconds": 600, "total_seconds": 1800,
              "oct2_wake_utc": "2026-10-02T12:00:00+00:00", "provider_calls": 0,
              "firestore_writes": 0, "activation_performed": False}
    result["plan_digest"] = digest(result)
    return result


def apply_disabled(bridge, migration, receipt):
    expected = copy.deepcopy(migration)
    pinned = expected.pop("plan_digest", None)
    if (pinned != digest(expected) or migration.get("package") != receipt
            or migration.get("schema_version") != "blueprint.oct2-disabled-migration.v1"):
        raise Refusal("oct2_migration_receipt_invalid")
    ledger = FirestoreLedger(bridge)
    with ledger.lock():
        control, row, _ = snapshot(bridge)
        candidate = disabled_candidate(control)
        if (digest(normalized(control)) != migration["expected_control_digest"]
                or digest(row) != migration["oct1_row_digest"]
                or candidate != migration["candidate"] or digest(candidate) != migration["candidate_digest"]):
            raise Refusal("oct2_migration_state_changed_replan_required")
        bridge.call("assert_lease")
        # The existing configure transaction fences this writer. No run/file op.
        bridge.call("configure", value=candidate)
        after, retained, _ = snapshot(bridge)
        if normalized(after) != candidate or digest(retained) != migration["oct1_row_digest"]:
            raise Refusal("oct2_disabled_migration_readback_failed")
    return {"state": "configured_disabled", "control_digest": digest(candidate),
            "oct1_row_digest": digest(retained), "oct1_raw_sha256": RAW,
            "provider_calls": 0, "activation_performed": False}


def write_private(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "apply-disabled"])
    parser.add_argument("--package", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    receipt = package_receipt(args.package, args.archive)
    if args.command == "plan" and not args.output or args.command == "apply-disabled" and not args.input:
        raise Refusal("oct2_required_argument_missing")
    bridge = Bridge()
    try:
        result = plan(bridge, receipt) if args.command == "plan" else apply_disabled(bridge, read_json(args.input), receipt)
        if args.command == "plan":
            write_private(args.output, result)
        print(canonical({k: v for k, v in result.items() if k not in {"candidate", "package", "summary"}}))
    finally:
        bridge.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - stable codes; never expose provider/account exceptions
        print(canonical({"state": "blocked", "error": str(error) if isinstance(error, Refusal) else "oct2_migration_unavailable"}))
        raise SystemExit(1) from None
