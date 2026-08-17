"""The blast-radius report must find bindings that hide behind a file hash.

Scene 840920's CAD receipts bind the task freeze by the freeze *file's* sha256,
not by its ``task_freeze_digest``. A digest-only trace reported them as
unaffected -- and they are the bindings that turn an amendment into a paid
decision, because re-deriving them means re-running CAD authoring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.seal_blast_radius import (
    SealBlastRadiusError,
    compute_seal_blast_radius,
)

DIGEST = "sha256:" + "a" * 64


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True), encoding="utf-8")
    return path


def _seal(tmp_path: Path) -> Path:
    return _write(tmp_path / "freeze.v1.json", {"schema_version": "dual_task_task_freeze.v1", "task_freeze_digest": DIGEST})


def test_digest_binding_is_forward(tmp_path: Path) -> None:
    _write(tmp_path / "spec.json", {"schema_version": "graph_spec.v1", "task_freeze_digest": DIGEST})
    report = compute_seal_blast_radius(digest=DIGEST, roots=[tmp_path])
    assert report["forward_binding_count"] == 1
    assert report["status"] == "amendable"


def test_history_is_not_rebound(tmp_path: Path) -> None:
    _write(tmp_path / "episode_evidence_index.v1.json", {"task_freeze_digest": DIGEST})
    report = compute_seal_blast_radius(digest=DIGEST, roots=[tmp_path])
    assert report["historical_binding_count"] == 1
    assert report["forward_binding_count"] == 0


def test_sealed_binding_makes_it_a_decision(tmp_path: Path) -> None:
    _write(tmp_path / "cad.json", {"schema_version": "simready_cad_agent_output.v1", "task_freeze_digest": DIGEST})
    report = compute_seal_blast_radius(digest=DIGEST, roots=[tmp_path])
    assert report["sealed_binding_count"] == 1
    assert report["status"] == "amendment_is_a_decision"


def test_file_hash_binding_is_found_only_with_the_seal_file(tmp_path: Path) -> None:
    """The regression that cost hours: CAD binds the freeze by file sha256."""

    seal = _seal(tmp_path)
    file_sha = "sha256:" + hashlib.sha256(seal.read_bytes()).hexdigest()
    _write(
        tmp_path / "cad_agent_request.v1.json",
        {
            "schema_version": "simready_cad_agent_request.v1",
            "inputs": {"task_freeze": {"path": str(seal), "sha256": file_sha}},
        },
    )
    blind = compute_seal_blast_radius(digest=DIGEST, roots=[tmp_path])
    assert blind["sealed_binding_count"] == 1, "only the freeze itself"

    seeing = compute_seal_blast_radius(digest=DIGEST, roots=[tmp_path], seal_file=seal)
    assert seeing["sealed_binding_count"] == 2, "freeze plus the CAD receipt"
    assert seeing["status"] == "amendment_is_a_decision"


def test_report_never_mutates(tmp_path: Path) -> None:
    _write(tmp_path / "spec.json", {"task_freeze_digest": DIGEST})
    report = compute_seal_blast_radius(digest=DIGEST, roots=[tmp_path])
    assert report["provider_mutation_performed"] is False
    assert report["spend_incurred_usd"] == 0.0


def test_malformed_digest_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SealBlastRadiusError):
        compute_seal_blast_radius(digest="nope", roots=[tmp_path])


def test_missing_seal_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(SealBlastRadiusError):
        compute_seal_blast_radius(digest=DIGEST, roots=[tmp_path], seal_file=tmp_path / "absent.json")
