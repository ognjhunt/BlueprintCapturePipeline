"""Selected team policy bytes are sealed before any paid provider call."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.native_g1_team_policy_execution_packet import (
    FILENAME,
    prepare_g1_team_policy_execution_packet,
)
from tests.test_native_g1_team_policy_authority import _authority


COMMIT = "a" * 40


def test_prepares_immutable_no_spend_team_provider_input(tmp_path: Path, monkeypatch) -> None:
    authority_args, approval = _authority(tmp_path, monkeypatch)
    args = {
        **authority_args,
        "output_dir": tmp_path / "execution-packet",
        "implementation_commit": COMMIT,
    }
    packet = prepare_g1_team_policy_execution_packet(**args)
    path = args["output_dir"] / FILENAME
    embedded = json.loads(path.read_text(encoding="utf-8"))
    assert packet == {"packet_path": str(path), **embedded}
    assert embedded["packet_digest"] == cross_runtime_canonical_digest(
        embedded, digest_field="packet_digest"
    )
    assert embedded["operator_approval"] == approval
    assert embedded["credential_value_included"] is False
    assert embedded["artifact_bytes_included"] is False
    assert embedded["provider_mutation_performed"] is False
    assert embedded["request"]["policy_profile"]["delivery"]["mode"] == "container"
    assert prepare_g1_team_policy_execution_packet(**args) == packet

    changed = {**approval, "operator_reviewer": "Second reviewer"}
    changed["approval_digest"] = canonical_digest(changed, digest_field="approval_digest")
    authority_args["approval_path"].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="packet_conflict"):
        prepare_g1_team_policy_execution_packet(**args)


def test_requires_exact_code_identity_before_creating_packet(tmp_path: Path, monkeypatch) -> None:
    authority_args, _ = _authority(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="commit_invalid"):
        prepare_g1_team_policy_execution_packet(
            **authority_args,
            output_dir=tmp_path / "execution-packet",
            implementation_commit="main",
        )
    assert not (tmp_path / "execution-packet").exists()
