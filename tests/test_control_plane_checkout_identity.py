from __future__ import annotations

import subprocess

from blueprint_pipeline import paid_resource_allocator as allocator


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout=stdout, stderr="")


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
