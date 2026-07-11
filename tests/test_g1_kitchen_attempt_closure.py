from __future__ import annotations

import json

import pytest

from blueprint_pipeline.g1_kitchen_attempt_closure import (
    PROOF_ROW_IDS,
    append_attempt_closure,
    build_attempt_closure,
    buyer_readout_projection,
)


def _identity() -> dict:
    digest = "a" * 64
    return {
        "run_id": "run-1",
        "attempt_id": "run-1-attempt-001",
        "launch_nonce": "nonce-1",
        "source_commit": "d1220f788",
        "source_dirty_patch_sha256": digest,
        "image_digest": f"sha256:{digest}",
        "bundle_digest": digest,
        "kitchen_asset_digest": digest,
        "active_selection_sha256": digest,
        "task_contract_sha256": digest,
        "provider_allocation_id": "pod-1",
    }


def _passing_rows(identity: dict) -> dict:
    rows = {
        row_id: {
            "status": "passed",
            "identity_binding": dict(identity),
            "evidence": {"proof": True},
        }
        for row_id in PROOF_ROW_IDS
    }
    rows["teardown"]["evidence"] = {
        "api_confirmed": True,
        "terminal_state": "not_found",
    }
    rows["final_inventory"]["evidence"] = {
        "api_confirmed": True,
        "live_resource_count": 0,
    }
    return rows


@pytest.mark.parametrize(
    ("row_id", "leaf_status"),
    [
        ("persistent_simulator_transition", "renderer_completed"),
        ("controller_fk", "structural_loop_completed"),
        ("startup", "marker_verified"),
    ],
)
def test_leaf_completion_cannot_close_attempt(row_id: str, leaf_status: str) -> None:
    identity = _identity()
    rows = _passing_rows(identity)
    rows[row_id] = {
        "status": "blocked",
        "identity_binding": identity,
        "evidence": {"leaf_status": leaf_status},
        "blockers": [f"{row_id}_proof_missing"],
    }
    closure = build_attempt_closure(identity=identity, proof_rows=rows)
    assert closure["status"] == "blocked"
    assert buyer_readout_projection(closure)["task_success_proven"] is False


def test_full_closure_requires_api_teardown_and_zero_inventory() -> None:
    identity = _identity()
    rows = _passing_rows(identity)
    closure = build_attempt_closure(identity=identity, proof_rows=rows)
    assert closure["status"] == "completed"
    projection = buyer_readout_projection(closure)
    assert projection == {
        "identity": closure["identity"],
        "verified_leaf_artifact_sha256s": projection[
            "verified_leaf_artifact_sha256s"
        ],
        "source_schema_version": "g1_kitchen_attempt_closure.v1",
        "source_closure_sha256": projection["source_closure_sha256"],
        "status": "ready",
        "task_success_proven": True,
        "semantic_review_passed": True,
        "forward_consistency_passed": True,
        "inverse_consistency_passed": True,
        "teardown_and_zero_inventory_proven": True,
        "blockers": [],
    }

    rows["teardown"]["evidence"]["api_confirmed"] = False
    blocked = build_attempt_closure(identity=identity, proof_rows=rows)
    assert blocked["status"] == "blocked"
    assert "teardown:provider_api_confirmation_missing" in blocked["blockers"]


def test_identity_mismatch_and_duplicate_attempt_fail(tmp_path) -> None:
    identity = _identity()
    rows = _passing_rows(identity)
    rows["scene_load"]["identity_binding"]["launch_nonce"] = "stale-nonce"
    blocked = build_attempt_closure(identity=identity, proof_rows=rows)
    assert blocked["status"] == "blocked"
    assert "scene_load:identity_binding_mismatch:launch_nonce" in blocked["blockers"]

    rows = _passing_rows(identity)
    closure = build_attempt_closure(identity=identity, proof_rows=rows)
    registry = tmp_path / "closures.jsonl"
    append_attempt_closure(registry, closure)
    with pytest.raises(ValueError, match="duplicate run/attempt"):
        append_attempt_closure(registry, closure)
    stored = json.loads(registry.read_text().strip())
    assert stored["identity"]["attempt_id"] == identity["attempt_id"]


def test_passed_row_requires_complete_attempt_identity_binding() -> None:
    identity = _identity()
    rows = _passing_rows(identity)
    rows["fast_canary"]["identity_binding"] = {"launch_nonce": identity["launch_nonce"]}
    closure = build_attempt_closure(identity=identity, proof_rows=rows)
    assert closure["status"] == "blocked"
    assert "fast_canary:passed_row_identity_binding_missing:image_digest" in closure[
        "blockers"
    ]


def test_superseded_attempt_is_terminal_blocked() -> None:
    identity = _identity()
    rows = {
        row_id: {
            "status": "not_requested",
            "identity_binding": {},
            "evidence": {},
        }
        for row_id in PROOF_ROW_IDS
    }
    closure = build_attempt_closure(
        identity=identity,
        proof_rows=rows,
        terminal_reason="superseded_by_attempt:run-1-attempt-002",
    )
    assert closure["terminal"] is True
    assert closure["status"] == "blocked"
    assert closure["terminal_reason"].startswith("superseded_by_attempt")


def test_buyer_projection_preserves_verified_leaf_digests() -> None:
    from blueprint_pipeline.g1_kitchen_attempt_closure import buyer_readout_projection

    closure = {
        "schema_version": "g1_kitchen_attempt_closure.v1",
        "status": "completed",
        "identity": {"run_id": "run-1", "attempt_id": "attempt-1"},
        "proof_rows": [
            {
                "row_id": "persistent_simulator_transition",
                "status": "passed",
                "evidence": {
                    "verified_leaf_artifacts": [
                        {"path": "a.json", "sha256": "a" * 64},
                        {"path": "b.json", "sha256": "b" * 64},
                    ]
                },
            },
            {"row_id": "semantic_review", "status": "passed", "evidence": {}},
        ],
    }
    projection = buyer_readout_projection(closure)
    assert projection["verified_leaf_artifact_sha256s"][
        "persistent_simulator_transition"
    ] == ["a" * 64, "b" * 64]
    assert projection["identity"] == closure["identity"]
