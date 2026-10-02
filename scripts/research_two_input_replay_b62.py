"""Offline replay of the exact native backup plus original-CRM fixture."""
import argparse
import base64
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "b62c27d2aa9a4c455a05b0e9b598f39e8e8d68b6"
ARCHIVE = "3a21335a9654fd259fe491b56f85b59fc7f02021e4335273e5c686b03b8c5447"
FIXTURE = "5013b4b468e6a9f080609e7df7ecb3075018c42a23e864a634590d460708aefb"
MODULE = "d89d01def1206680423c6b2780454a13b1f4da9ae8e9ab5d54199eed37aff9f6"
RAW = "011c09c6e5900e910c65c71a7852a8a8aefb4b394a98abd91e8c4c9498085a36"
SESSION = "sess_06ea8f997fa27202006abf0b37b9f4819aacfaa2cb1414eb14"


def require(condition, code):
    if not condition:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked_file(path, size, expected):
    require(path.stat().st_size == size, "input_size_mismatch")
    raw = path.read_bytes()
    require(sha(raw) == expected, "input_hash_mismatch")
    return raw


def decode_entries(backup):
    entries = backup["entries"]
    require(isinstance(entries, list) and len(entries) == 64, "backup_entry_inventory_mismatch")
    files = {}
    for entry in entries:
        require(isinstance(entry, dict) and isinstance(entry.get("name"), str)
                and entry["name"] not in files and entry.get("encoding") == "base64"
                and type(entry.get("bytes")) is int and entry["bytes"] >= 0
                and isinstance(entry.get("sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]) is not None
                and isinstance(entry.get("content"), str), "backup_entry_shape_or_identity_mismatch")
        raw = base64.b64decode(entry["content"], validate=True)
        require(len(raw) == entry["bytes"] and sha(raw) == entry["sha256"], "backup_entry_bytes_mismatch")
        files[entry["name"]] = raw
    return files


def materialize(backup, fixture):
    files = decode_entries(backup)
    row_bytes = files["ledger/status.json"]
    require(len(row_bytes) == 183926 and sha(row_bytes).startswith("470c04bf"), "retained_row_receipt_mismatch")
    row = json.loads(row_bytes)
    require(row["session_id"] == SESSION and row["date"] == fixture["run_date"] == "2026-10-01"
            and row["turn_id"] == fixture["root_turn_id"] and row["raw_output_digest"] == RAW,
            "retained_row_session_turn_or_date_mismatch")
    raw = files["ledger/2026-10-01-artifact.json"]
    require(len(raw) == 29190 and sha(raw) == fixture["raw_output_sha256"] == RAW
            and raw == fixture["raw_output_utf8"].encode("utf-8"), "retained_artifact_binding_mismatch")
    for field in ("knowledge_context", "refresh_policy"):
        require(sha(encoded(row[field])) == row[field + "_digest"] == fixture[field + "_digest"]
                and row[field] == fixture[field], "retained_context_binding_mismatch")
    tools = {}
    calls = row["application_tool_calls"]
    require(isinstance(calls, dict) and len(calls) == 44, "retained_tool_inventory_mismatch")
    for call in calls.values():
        name = call["result_file"]
        value = files["ledger/" + name]
        require(sha(value) == call["result_sha256"] and len(value) == call["result_bytes"]
                and sha(encoded(json.loads(value))) == call["result_digest"], "retained_tool_binding_mismatch")
        tools[name] = value
    keys = fixture["crm_identity_keys"]
    require(isinstance(keys, list) and len(keys) == 22
            and all(isinstance(key, str) and re.fullmatch(r"[a-f0-9]{64}", key) for key in keys),
            "original_crm_key_inventory_mismatch")
    return row, raw, tools, set(keys), sha(row_bytes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--backup-sha256", required=True, help="Full authenticated backup receipt SHA; never compute expected from this input")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-f0-9]{64}", args.backup_sha256) is not None
            and args.backup_sha256.startswith("12d6b7"), "authenticated_backup_checksum_required")
    backup_bytes = checked_file(Path(args.backup), 2498003, args.backup_sha256)
    fixture_bytes = checked_file(Path(args.fixture), 184856, FIXTURE)
    archive = Path(args.archive)
    checked_file(archive, 573440, ARCHIVE)
    release = Path(args.release).resolve()
    manifest_bytes = (release / "manifest.json").read_bytes()
    with tarfile.open(archive) as bundle:
        require(bundle.extractfile("manifest.json").read() == manifest_bytes, "reviewed_manifest_mismatch")
    manifest = json.loads(manifest_bytes)
    require(manifest["source_commit"] == SOURCE and len(manifest["files"]) == 46, "reviewed_package_mismatch")
    for name, expected in manifest["files"].items():
        path = release / name
        require(not path.is_symlink() and path.resolve().is_relative_to(release)
                and sha(path.read_bytes()) == expected, "reviewed_package_file_mismatch")
    require(sha((release / "tools/daily_research/recovery.py").read_bytes()) == MODULE, "reviewed_module_mismatch")
    row, artifact, tools, known, row_sha = materialize(json.loads(backup_bytes), json.loads(fixture_bytes))
    sys.path.insert(0, str(release))
    from tools.daily_research.recovery import replay_saved_artifact
    from tools.daily_research.runner import canonical
    result = replay_saved_artifact(row, artifact, tools, known, datetime.now(timezone.utc))
    result.update(source_commit=SOURCE, recovery_module_sha256=MODULE,
                  backup_sha256=args.backup_sha256, backup_generation="1790907375070436",
                  fixture_sha256=FIXTURE, retained_row_sha256=row_sha,
                  original_crm_keys_pointer="/crm_identity_keys", original_crm_key_count=len(known),
                  retained_tool_file_count=len(tools))
    print(canonical(result), flush=True)
    require(result["valid"] and len(result["date_normalizations"]) == 7
            and result["quarantined_proposal_count"] == 1
            and result["provider_calls"] == result["database_writes"] == result["publication_writes"] == 0
            and result["qa_and_publication_verified"] is False, "actual_replay_acceptance_failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        code = str(error) if isinstance(error, ValueError) and re.fullmatch(r"[a-z_]+", str(error)) else type(error).__name__
        print(json.dumps({"schema_version": "blueprint.two-input-replay.v1", "complete": False, "error": code}), flush=True)
        raise SystemExit(2) from None
