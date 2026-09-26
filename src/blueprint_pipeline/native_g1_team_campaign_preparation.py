"""Prepare an accepted G1 team choice for the canonical paid controller.

This step rechecks the immutable intake record and operator registry, then
builds the existing G1 provider bundle. It never calls a provider adapter.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_development_pair import PAIR_ORDER
from .native_g1_provider_bundle import (
    build_g1_provider_bundle,
    load_verified_g1_provider_bundle,
)
from .native_g1_team_campaign_intake import (
    INTENT_SCHEMA,
    REGISTRY_SCHEMA,
    _read,
    _selected_binding,
)
from .native_g1_team_campaign_request import validate_g1_team_campaign_request
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)
from .task_evaluation_packet_planning_setup import make_packet_planning_setup


SCHEMA = "native_g1_team_campaign_preparation.v1"


def _verified_intent(intent_path: Path, registry_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    intent = _read(intent_path, field="intent_digest")
    if (
        intent.get("schema_version") != INTENT_SCHEMA
        or intent.get("status") != "accepted_not_dispatched"
        or intent.get("claim_ceiling") != "development_only"
        or intent.get("provider_mutation_performed") is not False
        or not isinstance(intent.get("request"), dict)
        or not isinstance(intent.get("binding"), dict)
    ):
        raise ValueError("g1_team_campaign_intent_invalid")
    expected_id = "g1-" + digest({
        "owner": intent["request"].get("owner"),
        "run_id": intent["request"].get("run_id"),
    }).removeprefix("sha256:")
    if intent.get("intent_id") != expected_id or intent_path.parent.name != expected_id:
        raise ValueError("g1_team_campaign_intent_id_invalid")
    registry = _read(registry_path, field="registry_digest")
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("g1_team_campaign_registry_invalid")
    if intent.get("registry_digest") != registry["registry_digest"]:
        raise ValueError("g1_team_campaign_registry_changed")
    binding = _selected_binding(registry, intent["request"])
    if (
        binding != intent["binding"]
        or intent.get("binding_digest") != digest(binding)
    ):
        raise ValueError("g1_team_campaign_binding_changed")
    setup = make_packet_planning_setup(source_packet_dir=Path(binding["source_packet_dir"]))
    if (
        binding["scene_id"] != setup["scene_id"]
        or binding["task_id"] != setup["task_id"]
        or binding["source_packet_receipt_digest"] != setup["source_packet_receipt_digest"]
    ):
        raise ValueError("g1_team_campaign_binding_packet_mismatch")
    validate_g1_team_campaign_request(
        intent["request"], trusted_setup=setup, authenticated_owner=binding["owner"]
    )
    return intent, binding


def _ensure_handoff(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if _read(path, field="handoff_digest") != value:
            raise ValueError("g1_team_campaign_handoff_conflict")
    else:
        write_exclusive(path, value)


def prepare_g1_team_campaign(
    *,
    intent_path: Path,
    registry_path: Path,
    work_root: Path,
    implementation_commit: str,
) -> dict[str, Any]:
    """Build one no-spend G1 bundle from an immutable signed team intake."""

    if re.fullmatch(r"[0-9a-f]{40}", implementation_commit) is None:
        raise ValueError("g1_team_campaign_implementation_commit_invalid")
    intent, binding = _verified_intent(Path(intent_path), Path(registry_path))
    root = Path(work_root)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("g1_team_campaign_work_root_invalid")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    lock = os.open(root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        directory = root / intent["intent_id"]
        if directory.is_symlink():
            raise ValueError("g1_team_campaign_work_directory_unsafe")
        directory.mkdir(mode=0o750, exist_ok=True)
        receipt_path = directory / "preparation.json"
        bundle_receipt_path = directory / "bundle" / "native_g1_provider_bundle.v1.json"
        if receipt_path.exists() or receipt_path.is_symlink():
            previous = _read(receipt_path, field="preparation_digest")
            if (
                previous.get("intent_digest") != intent["intent_digest"]
                or previous.get("implementation_commit") != implementation_commit
                or previous.get("bundle_receipt_path") != str(bundle_receipt_path)
            ):
                raise ValueError("g1_team_campaign_preparation_conflict")
            verified_bundle = load_verified_g1_provider_bundle(
                bundle_receipt_path,
                expected_implementation_commit=implementation_commit,
            )
            if verified_bundle["bundle_sha256"] != previous["bundle_sha256"]:
                raise ValueError("g1_team_campaign_prepared_bundle_changed")
            return previous
        book_path = directory / "book_handoff.json"
        movement_path = directory / "movement_handoff.json"
        _ensure_handoff(book_path, intent["request"]["book_handoff"])
        _ensure_handoff(movement_path, intent["request"]["movement_handoff"])
        bundle = build_g1_provider_bundle(
            job_dir=directory / "bundle",
            manipulation_packet=Path(binding["manipulation_packet_dir"]),
            movement_packet=Path(binding["movement_packet_dir"]),
            book_handoff=book_path,
            movement_handoff=movement_path,
            rights_review_paths={
                candidate: Path(binding["rights_review_paths"][candidate])
                for candidate in PAIR_ORDER
            },
            navigation_authority=Path(binding["navigation_authority_path"]),
            publisher_source=Path(binding["publisher_source_dir"]),
            runtime_source_receipt=Path(binding["runtime_source_receipt_path"]),
            implementation_commit=implementation_commit,
        )
        if not bundle_receipt_path.is_file() or bundle_receipt_path.is_symlink():
            raise ValueError("g1_team_campaign_bundle_receipt_missing")
        receipt = {
            "schema_version": SCHEMA,
            "status": "bundle_prepared_not_executed",
            "intent_id": intent["intent_id"],
            "intent_digest": intent["intent_digest"],
            "implementation_commit": implementation_commit,
            "bundle_receipt_path": str(bundle_receipt_path),
            "bundle_sha256": bundle["bundle_sha256"],
            "authorization": intent["request"]["authorization"],
            "claim_ceiling": "development_only",
            "provider_mutation_performed": False,
        }
        receipt["preparation_digest"] = digest(receipt, digest_field="preparation_digest")
        write_exclusive(receipt_path, receipt)
        return receipt
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent-path", type=Path, required=True)
    parser.add_argument("--registry-path", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args(argv)
    receipt = prepare_g1_team_campaign(
        intent_path=args.intent_path,
        registry_path=args.registry_path,
        work_root=args.work_root,
        implementation_commit=args.implementation_commit,
    )
    print(json.dumps({
        "status": receipt["status"],
        "intent_id": receipt["intent_id"],
        "preparation_digest": receipt["preparation_digest"],
        "provider_mutation_performed": False,
    }, sort_keys=True))
    return 0


__all__ = ["SCHEMA", "prepare_g1_team_campaign"]


if __name__ == "__main__":
    raise SystemExit(main())
