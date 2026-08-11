from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_abstention_blocked_markdown import (
    AbstentionBlockedMarkdownError,
    materialize_abstention_blocked_markdown,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/arm_decision_proof_v1/third_scene_dual_task_evidence"


def _abstention(task_root: Path) -> Path:
    payload: dict[str, object] = {
        "schema_version": "adp_task_evaluation_run_abstention.v1",
        "status": "typed_evidence_backed_abstention",
        "scene_id": "fixture_scene",
        "task_id": "fixture_task",
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "controls_executed": False,
        "learned_candidate_episodes_executed": False,
        "smallest_missing_capability": "fresh_paid_authority_missing",
        "task_freeze_digest": "sha256:" + "1" * 64,
        "gaussian_excision_freeze_digest": "sha256:" + "2" * 64,
        "historical_attempt_blockers": ["old_runtime_error"],
        "receipt_digest": "",
    }
    payload["receipt_digest"] = canonical_digest(
        payload, digest_field="receipt_digest"
    )
    path = task_root / "typed_abstention.v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_materializes_controls_and_candidate_blockers_from_abstention(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "task"
    abstention = _abstention(task_root)

    receipt = materialize_abstention_blocked_markdown(
        task_root=task_root,
        abstention_receipt_path=abstention,
        task_label="Fixture task",
    )

    assert receipt["status"] == "materialized"
    assert len(receipt["outputs"]) == 3
    controls = (task_root / "controls/BLOCKED.md").read_text()
    policy = (task_root / "pi05_droid/BLOCKED.md").read_text()
    assert "Current smallest blocker: `fresh_paid_authority_missing`." in controls
    assert "Historical attempt blockers retained as historical only" in controls
    assert "This is not a policy failure" in policy


def test_rejects_tampered_abstention_digest(tmp_path: Path) -> None:
    task_root = tmp_path / "task"
    abstention = _abstention(task_root)
    payload = json.loads(abstention.read_text())
    payload["smallest_missing_capability"] = "changed"
    abstention.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AbstentionBlockedMarkdownError, match="abstention_digest_invalid"):
        materialize_abstention_blocked_markdown(
            task_root=task_root,
            abstention_receipt_path=abstention,
            task_label="Fixture task",
        )


def test_checked_in_blockers_are_bound_to_current_abstentions() -> None:
    for task_dir in ("task_a", "task_b"):
        root = EVIDENCE / task_dir
        abstention = json.loads((root / "typed_abstention.v1.json").read_text())
        assert abstention["receipt_digest"] == canonical_digest(
            abstention, digest_field="receipt_digest"
        )
        current = abstention["smallest_missing_capability"]
        for relative in (
            "controls/BLOCKED.md",
            "pi05_droid/BLOCKED.md",
            "groot_n17_droid/BLOCKED.md",
        ):
            text = (root / relative).read_text()
            assert f"Current smallest blocker: `{current}`." in text
            assert f"Typed abstention digest: `{abstention['receipt_digest']}`." in text
            if relative != "controls/BLOCKED.md":
                assert "This is not a policy failure" in text

