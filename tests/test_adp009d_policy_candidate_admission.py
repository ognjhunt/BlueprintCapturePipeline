from __future__ import annotations

import json
from pathlib import Path


from blueprint_pipeline.decision_evidence_contracts import canonical_digest


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    REPO_ROOT
    / "docs/arm_decision_proof_v1/manifests/adp009d_policy_candidate_inventory.v1.json"
)


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _candidate(inventory: dict, candidate_id: str) -> dict:
    return next(
        row for row in inventory["candidates"] if row["candidate_id"] == candidate_id
    )


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _runtime_admission(candidate: dict, character: str) -> dict:
    receipt = {
        "schema_version": "adp009d_policy_candidate_runtime_admission.v1",
        "candidate_id": candidate["candidate_id"],
        "candidate_digest": candidate["candidate_digest"],
        "checkpoint_materialization_digest": _sha(character),
        "checkpoint_tree_sha256": _sha(chr(ord(character) + 1)),
        "observation_adapter_digest": _sha(chr(ord(character) + 2)),
        "action_adapter_digest": _sha(chr(ord(character) + 3)),
        "camera_calibration_digest": _sha(chr(ord(character) + 4)),
        "immutable_smoke_input_digest": _sha(chr(ord(character) + 5)),
        "immutable_smoke_output_digest": _sha(chr(ord(character) + 6)),
        "runtime_environment_digest": _sha(chr(ord(character) + 7)),
        "checkpoint_all_files_verified": True,
        "live_policy_frames_used": True,
        "action_adapter_native_probe_passed": True,
        "immutable_smoke_passed": True,
        "task_outcomes_observed": False,
        "blockers": [],
        "admitted": True,
        "admission_digest": "",
    }
    receipt["admission_digest"] = canonical_digest(
        receipt, digest_field="admission_digest"
    )
    return receipt


_FROZEN_INVENTORY = (
    Path(__file__).resolve().parents[1]
    / "config/trial/openpi_polaris_checkpoint_inventory.json"
)

