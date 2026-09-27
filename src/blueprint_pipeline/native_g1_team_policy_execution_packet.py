"""Seal one selected and approved G1 team policy for private provider staging.

This packet carries exact owner, task, interface, delivery, and operator rights
bindings. It contains no credential value, artifact bytes, or provider result.
The paid dispatcher must recheck live authority after spend admission and bind
this exact packet digest into a private provider bundle.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_campaign_intake import _read
from .native_g1_team_policy_authority import verify_g1_team_policy_authority
from .task_evaluation_launch_preparation_queue import (
    _write_launch_preparation_record_exclusive_locked as write_exclusive,
)


SCHEMA = "native_g1_team_policy_execution_packet.v1"
FILENAME = SCHEMA + ".json"
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def prepare_g1_team_policy_execution_packet(
    *,
    intent_path: Path,
    registry_path: Path,
    approval_path: Path,
    trusted_clients: set[str],
    output_dir: Path,
    implementation_commit: str,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Write an immutable, credential-free packet without allocating compute."""

    if not isinstance(implementation_commit, str) or _COMMIT.fullmatch(implementation_commit) is None:
        raise ValueError("g1_team_policy_execution_commit_invalid")
    root = Path(output_dir)
    if not root.is_absolute() or root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("g1_team_policy_execution_path_invalid")
    authority = verify_g1_team_policy_authority(
        intent_path=intent_path,
        registry_path=registry_path,
        approval_path=approval_path,
        trusted_clients=trusted_clients,
        now_epoch=now_epoch,
    )
    intent = authority["intent"]
    request = intent["request"]
    approval = authority["operator_approval"]
    setup = authority["trusted_setup"]
    packet = {
        "schema_version": SCHEMA,
        "status": "approved_input_not_executed",
        "implementation_commit": implementation_commit,
        "intent_id": intent["intent_id"],
        "intent_digest": intent["intent_digest"],
        "request": request,
        "trusted_setup": setup,
        "operator_approval": approval,
        "policy_profile_digest": request["policy_profile"]["profile_digest"],
        "source_packet_receipt_digest": setup["source_packet_receipt_digest"],
        "objective_id": request["objective_id"],
        "delivery_mode": request["policy_profile"]["delivery"]["mode"],
        "credential_value_included": False,
        "artifact_bytes_included": False,
        "provider_mutation_performed": False,
        "claim_ceiling": "development_only",
    }
    packet["packet_digest"] = digest(packet, digest_field="packet_digest")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    path = root / FILENAME
    if path.exists() or path.is_symlink():
        prior = _read(path, field="packet_digest")
        if prior != packet:
            raise ValueError("g1_team_policy_execution_packet_conflict")
    else:
        write_exclusive(path, packet)
    return {"packet_path": str(path), **packet}


__all__ = ["FILENAME", "SCHEMA", "prepare_g1_team_policy_execution_packet"]
