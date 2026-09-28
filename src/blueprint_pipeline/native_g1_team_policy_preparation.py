"""Prepare one approved team-selected G1 profile without allocating compute.

Scene and runtime paths come from the operator's owner-scoped registry. This
module only seals the existing selected execution packet and provider bundle;
it cannot dispatch, spend, settle or publish the result.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_campaign_intake import _read
from .native_g1_team_policy_authority import verify_g1_team_policy_authority
from .native_g1_team_policy_execution_packet import prepare_g1_team_policy_execution_packet
from .native_g1_team_provider_bundle import (
    SCHEMA as BUNDLE_SCHEMA, build_g1_team_provider_bundle,
    load_verified_g1_team_provider_bundle,
)
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)


SCHEMA = "native_g1_team_policy_preparation.v1"


def _operator_directory(path: Path, *, required: bool) -> None:
    if (not path.is_absolute() or ".." in path.parts
            or any(parent.is_symlink() for parent in (path, *path.parents))
            or (required and not path.is_dir())
            or (path.exists() and not path.is_dir())):
        raise ValueError("g1_team_policy_operator_directory_invalid")


def prepare_g1_team_policy(
    *, authority_arguments: Mapping[str, Any], work_root: Path,
    sonic_asset_dir: Path, implementation_commit: str,
) -> dict[str, Any]:
    """Seal a selected registered scene; cached preparation is immutable."""
    if not isinstance(implementation_commit, str) or re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise ValueError("g1_team_policy_preparation_commit_invalid")
    root, sonic = Path(work_root), Path(sonic_asset_dir)
    _operator_directory(root, required=False)
    _operator_directory(sonic, required=True)
    live = {**authority_arguments, "now_epoch": time.time()}
    authority = verify_g1_team_policy_authority(**live)
    intent, binding = authority["intent"], authority["registry_binding"]
    request = intent["request"]
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    descriptor = os.open(root / ".preparation.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        live["now_epoch"] = time.time()
        if verify_g1_team_policy_authority(**live) != authority:
            raise ValueError("g1_team_policy_preparation_authority_changed")
        directory = root / intent["intent_id"]
        _operator_directory(directory, required=False)
        directory.mkdir(mode=0o750, exist_ok=True)
        receipt_path = directory / "preparation.json"
        bundle_path = directory / "bundle" / (BUNDLE_SCHEMA + ".json")
        expected = {
            "schema_version": SCHEMA, "status": "bundle_prepared_not_executed",
            "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
            "implementation_commit": implementation_commit,
            "policy_profile_digest": request["policy_profile"]["profile_digest"],
            "operator_approval_digest": authority["operator_approval"]["approval_digest"],
            "objective_id": request["objective_id"], "bundle_receipt_path": str(bundle_path),
            "authorization": request["authorization"], "claim_ceiling": "development_only",
            "provider_mutation_performed": False,
        }
        if receipt_path.exists() or receipt_path.is_symlink():
            previous = _read(receipt_path, field="preparation_digest")
            if any(previous.get(key) != value for key, value in expected.items()):
                raise ValueError("g1_team_policy_preparation_conflict")
            live["now_epoch"] = time.time()
            bundle = load_verified_g1_team_provider_bundle(
                bundle_path, expected_implementation_commit=implementation_commit,
                authority_arguments=live,
            )
            if any(previous.get(key) != bundle[key] for key in (
                "bundle_sha256", "manifest_digest", "execution_packet_digest",
            )):
                raise ValueError("g1_team_policy_preparation_bundle_changed")
            return previous
        packet = prepare_g1_team_policy_execution_packet(
            **live, output_dir=directory / "execution", implementation_commit=implementation_commit,
        )
        scene_field = ("movement_packet_dir" if request["objective_id"] == "g1_navigation_goal"
                       else "manipulation_packet_dir")
        bundle = build_g1_team_provider_bundle(
            job_dir=directory / "bundle", execution_packet_path=Path(packet["packet_path"]),
            authority_arguments=live, scene_packet_root=Path(binding[scene_field]),
            publisher_source=Path(binding["publisher_source_dir"]),
            runtime_source_receipt=Path(binding["runtime_source_receipt_path"]),
            sonic_asset_dir=sonic, expected_implementation_commit=implementation_commit,
        )
        live["now_epoch"] = time.time()
        verified = load_verified_g1_team_provider_bundle(
            bundle_path, expected_implementation_commit=implementation_commit, authority_arguments=live,
        )
        if verified != bundle:
            raise ValueError("g1_team_policy_preparation_bundle_changed")
        receipt = {**expected, **{key: bundle[key] for key in (
            "bundle_sha256", "manifest_digest", "execution_packet_digest",
        )}}
        receipt["preparation_digest"] = digest(receipt, digest_field="preparation_digest")
        write_exclusive(receipt_path, receipt)
        return receipt
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intent-path", "registry-path", "approval-path", "work-root", "sonic-asset-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trusted-client", action="append", required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args(argv)
    result = prepare_g1_team_policy(
        authority_arguments={"intent_path": args.intent_path, "registry_path": args.registry_path,
                             "approval_path": args.approval_path, "trusted_clients": set(args.trusted_client)},
        work_root=args.work_root, sonic_asset_dir=args.sonic_asset_dir,
        implementation_commit=args.implementation_commit,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "intent_id", "preparation_digest", "provider_mutation_performed",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
