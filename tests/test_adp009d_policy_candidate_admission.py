from __future__ import annotations

import json
from pathlib import Path


from blueprint_pipeline.adp009d_policy_candidate_admission import (
    EXPECTED_CANDIDATES,
)
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


def test_pi05_checkpoint_identity_satisfies_the_spec_that_refused_the_stock_one(
    tmp_path: Path,
) -> None:
    """The whole point of ratifying: the provisioned URI must now be describable.

    The stock checkpoint failed `checkpoint_uri_not_frozen_openpi_polaris`, so
    no execution spec could name what the lane fetched and still satisfy the
    episode client.
    """

    import json

    from blueprint_pipeline.openpi_droid_policy_runtime import (
        load_policy_spec_from_execution_spec,
    )

    record = EXPECTED_CANDIDATES["pi05_droid"]
    inventory = json.loads(_FROZEN_INVENTORY.read_text(encoding="utf-8"))
    entry = next(
        row
        for row in inventory["entries"]
        if row["policy_id"] == "pi05_droid_jointpos_polaris"
    )
    path = tmp_path / "execution_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "native_task_arena_policy_execution_spec.v1",
                "candidate_id": "pi05_droid",
                "policy_spec": {
                    "policy_id": "pi05_droid_jointpos_polaris",
                    "config_name": "pi05_droid_jointpos_polaris",
                    # The URI this lane actually provisions.
                    "checkpoint_uri": record["checkpoint_repository"],
                    "checkpoint_object_manifest_sha256": entry[
                        "legacy_object_manifest_sha256"
                    ],
                    "checkpoint_generation_manifest_sha256": entry[
                        "generation_manifest_sha256"
                    ],
                    "checkpoint_inventory_sha256": inventory["inventory_sha256"],
                    "checkpoint_object_count": entry["object_count"],
                    "checkpoint_size_bytes": entry["size_bytes"],
                    "action_space": "joint_position",
                    "action_chunk_rows": 15,
                    "open_loop_horizon": 8,
                    "openpi_revision": inventory["openpi_revision"],
                },
            }
        ),
        encoding="utf-8",
    )

    spec = load_policy_spec_from_execution_spec(path)

    assert spec.policy_id == entry["policy_id"]
    assert spec.config_name == "pi05_droid_jointpos_polaris"
    assert spec.checkpoint_uri == record["checkpoint_repository"]
    assert spec.checkpoint_size_bytes == record["checkpoint_total_bytes"]
    assert len(spec.server_metadata()) == 14

