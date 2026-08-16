from __future__ import annotations

import subprocess
import json
from pathlib import Path

from blueprint_pipeline import paid_resource_allocator as allocator


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout=stdout, stderr="")


def test_ancestor_promotion_receipt_must_bind_the_exact_full_lane(
    tmp_path: Path, monkeypatch
) -> None:
    commit = "a" * 40
    monkeypatch.setattr(allocator, "CONTROL_PLANE_RELEASE_STATE_ROOT", tmp_path)
    path = tmp_path / commit / allocator.DEPLOY_RELEASE_PROVENANCE_NAME
    path.parent.mkdir(parents=True)
    receipt = {
        "schema_version": "blueprint.deploy_release_provenance.v1",
        "status": "verified",
        "git_sha": commit,
        "run_id": 123,
        "workflow_name": "Full Test Lane",
        "workflow_path": ".github/workflows/full-test-lane.yml",
        "job_name": "Full pytest lane on CPU runner",
        "collection": {"test_count": 100},
        "claim_boundary": {"canonical_full_lane_verified": True},
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    path.chmod(0o440)

    assert allocator._commit_has_verified_production_promotion(commit) is True

    receipt["git_sha"] = "b" * 40
    path.chmod(0o640)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    path.chmod(0o440)
    assert allocator._commit_has_verified_production_promotion(commit) is False


def test_identity_probe_failure_is_not_reported_as_a_dirty_checkout(monkeypatch) -> None:
    """A real launch was failed closed with checkout_not_clean while the checkout
    was clean: git's post-deploy auto-repack outran a short probe. A dirty tree
    still resolves HEAD, so an unresolvable commit AND a non-clean tree together
    can only mean the probe failed, and claiming dirty asserts something untrue."""
    monkeypatch.setattr(allocator, "_run_checkout_probe", lambda argv: None)

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert "gpu_canary_orchestrator_identity_probe_failed" in blockers
    assert "gpu_canary_orchestrator_checkout_not_clean" not in blockers
    assert "gpu_canary_orchestrator_source_commit_unavailable" not in blockers
    assert identity["identity_probe_ran"] is False
    assert identity["orchestrator_source_commit"] is None


def test_checkout_identity_probes_trust_the_resolved_immutable_checkout(
    monkeypatch,
) -> None:
    """A symlink-safe systemd setting alone does not trust ``ROOT.resolve()``."""
    commit = "a" * 40
    commands: list[list[str]] = []

    def fake(argv):
        commands.append(list(argv))
        return _completed(commit + "\n") if "rev-parse" in argv else _completed("")

    monkeypatch.setattr(allocator, "_run_checkout_probe", fake)

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert blockers == []
    assert identity["orchestrator_source_commit"] == commit
    assert commands == [
        [
            "git",
            "-c",
            f"safe.directory={allocator.ROOT}",
            "-C",
            str(allocator.ROOT),
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
        ],
        [
            "git",
            "-c",
            f"safe.directory={allocator.ROOT}",
            "-C",
            str(allocator.ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
    ]


def test_a_genuinely_dirty_checkout_still_blocks(monkeypatch) -> None:
    commit = "a" * 40

    def fake(argv):
        if "rev-parse" in argv:
            return _completed(commit + "\n")
        return _completed(" M src/blueprint_pipeline/x.py\n")

    monkeypatch.setattr(allocator, "_run_checkout_probe", fake)

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert "gpu_canary_orchestrator_checkout_not_clean" in blockers
    assert "gpu_canary_orchestrator_identity_probe_failed" not in blockers
    assert identity["identity_probe_ran"] is True
    assert identity["checkout_clean"] is False


def test_transient_probe_failure_is_retried_before_blocking(monkeypatch) -> None:
    """The repack race is transient, so one failed attempt must not reject a
    paid launch that would succeed a moment later."""
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd="git", timeout=1)
        return _completed("b" * 40 + "\n")

    monkeypatch.setattr(allocator.subprocess, "run", flaky)
    monkeypatch.setattr(allocator.time, "sleep", lambda seconds: None)

    result = allocator._run_checkout_probe(["git", "rev-parse"])

    assert result is not None
    assert result.returncode == 0
    assert calls["n"] == 2


def test_clean_checkout_reports_no_blockers(monkeypatch) -> None:
    commit = "c" * 40
    monkeypatch.setattr(
        allocator,
        "_run_checkout_probe",
        lambda argv: _completed(commit + "\n") if "rev-parse" in argv else _completed(""),
    )

    blockers, identity = allocator._control_plane_checkout_blockers()

    assert blockers == []
    assert identity["identity_probe_ran"] is True
    assert identity["checkout_clean"] is True
    assert identity["orchestrator_source_commit"] == commit


def _prepare_main(
    monkeypatch, *, checkout: str, tip: str, merged: bool, promoted: bool = False
) -> None:
    monkeypatch.setattr(
        allocator, "_current_checkout_source_state", lambda: (checkout, True, True)
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: tip)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: tip)
    monkeypatch.setattr(
        allocator, "_commit_is_merged_into", lambda commit, ref: merged
    )
    monkeypatch.setattr(
        allocator,
        "_commit_has_verified_production_promotion",
        lambda commit: promoted,
    )


def test_a_commit_already_merged_into_main_may_still_launch(monkeypatch) -> None:
    """Production regression: merging a later fix invalidated a ready release.

    The gate needs the launched code to be public and reviewed, which every
    ancestor of main satisfies. Demanding the tip meant any merge landing after
    a release was prepared discarded a promoted deploy and its whole
    commit-bound artifact rebuild.
    """

    _prepare_main(
        monkeypatch,
        checkout="a" * 40,
        tip="b" * 40,
        merged=True,
        promoted=True,
    )

    blockers, commit = allocator._source_checkout_blockers("a" * 40)

    assert blockers == []
    assert commit == "a" * 40


def test_a_commit_outside_main_history_is_still_refused(monkeypatch) -> None:
    _prepare_main(monkeypatch, checkout="a" * 40, tip="b" * 40, merged=False)

    blockers, _ = allocator._source_checkout_blockers("a" * 40)

    assert "gpu_canary_checkout_not_origin_main" in blockers
    assert "gpu_canary_checkout_not_remote_main" in blockers


def test_a_merged_but_unpromoted_commit_is_still_refused(monkeypatch) -> None:
    _prepare_main(
        monkeypatch,
        checkout="a" * 40,
        tip="b" * 40,
        merged=True,
        promoted=False,
    )

    blockers, _ = allocator._source_checkout_blockers("a" * 40)

    assert blockers == ["gpu_canary_checkout_promotion_provenance_invalid"]


def test_the_main_tip_is_accepted_without_consulting_ancestry(monkeypatch) -> None:
    """The tip must not depend on a merge-base probe that could fail closed."""

    tip = "b" * 40
    _prepare_main(monkeypatch, checkout=tip, tip=tip, merged=False)

    blockers, commit = allocator._source_checkout_blockers(tip)

    assert blockers == []
    assert commit == tip


def test_a_dirty_ancestor_checkout_still_blocks(monkeypatch) -> None:
    """Relaxing which commit may launch must not relax whether it is intact."""

    monkeypatch.setattr(
        allocator, "_current_checkout_source_state", lambda: ("a" * 40, False, True)
    )
    monkeypatch.setattr(allocator, "_current_origin_main_commit", lambda: "b" * 40)
    monkeypatch.setattr(allocator, "_current_remote_main_commit", lambda: "b" * 40)
    monkeypatch.setattr(allocator, "_commit_is_merged_into", lambda commit, ref: True)
    monkeypatch.setattr(
        allocator,
        "_commit_has_verified_production_promotion",
        lambda commit: True,
    )

    blockers, _ = allocator._source_checkout_blockers("a" * 40)

    assert "gpu_canary_checkout_not_clean" in blockers


def test_source_checkout_blockers_also_separate_probe_failure_from_dirty(monkeypatch) -> None:
    """The expected-source-commit gate reads the same probe, so it must make the
    same distinction or it will report an unobserved checkout state too."""
    monkeypatch.setattr(
        allocator, "_current_checkout_source_state", lambda: ("", False, False)
    )

    blockers, commit = allocator._source_checkout_blockers("d" * 40)

    assert blockers == ["gpu_canary_checkout_identity_probe_failed"]
    assert "gpu_canary_checkout_not_clean" not in blockers
    assert commit == ""

    monkeypatch.setattr(
        allocator, "_current_checkout_source_state", lambda: ("d" * 40, False, True)
    )
    blockers, commit = allocator._source_checkout_blockers("d" * 40)
    assert "gpu_canary_checkout_not_clean" in blockers
    assert commit == "d" * 40
