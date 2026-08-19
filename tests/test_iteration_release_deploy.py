from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import blueprint_pipeline.paid_resource_allocator as pra

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "deploy_control_plane_commit", REPO / "scripts" / "deploy_control_plane_commit.py"
)
assert _SPEC and _SPEC.loader
deploy_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deploy_mod)

COMMIT = "a" * 40


def _write_provenance(root: Path, commit: str, payload: dict) -> Path:
    target = root / commit
    target.mkdir(parents=True, exist_ok=True)
    path = target / pra.DEPLOY_RELEASE_PROVENANCE_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o640)
    return path


def _verified_payload(commit: str) -> dict:
    return {
        "schema_version": "blueprint.deploy_release_provenance.v1",
        "status": "verified",
        "git_sha": commit,
        "workflow_name": "Full Test Lane",
        "workflow_path": ".github/workflows/full-test-lane.yml",
        "job_name": "Full pytest lane on CPU runner",
        "run_id": 32287449971,
        "collection": {"test_count": 13736},
        "claim_boundary": {"canonical_full_lane_verified": True},
    }


def _iteration_payload(commit: str) -> dict:
    return {
        "schema_version": "blueprint.deploy_release_provenance.v1",
        "status": "iteration",
        "git_sha": commit,
        "promotion_eligible": False,
        "claim_boundary": {
            "canonical_full_lane_verified": False,
            "promotion_eligible": False,
            "evidence_grade": "development_only",
        },
    }


def test_an_iteration_release_is_never_promotion_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Iteration trades lane verification for cycle time, not for evidence.

    The full lane costs ~15 minutes, so a fix-and-fire loop that waits for it
    spends ~18 minutes per attempt -- tens of hours across a campaign of GPU
    runs. Skipping it is a deliberate trade, but nothing produced from an
    unverified release may later be presented as adjudication-grade.
    """

    monkeypatch.setattr(pra, "CONTROL_PLANE_RELEASE_STATE_ROOT", tmp_path)

    _write_provenance(tmp_path, COMMIT, _iteration_payload(COMMIT))
    assert pra.release_promotion_eligible(COMMIT) is False

    verified_commit = "b" * 40
    _write_provenance(tmp_path, verified_commit, _verified_payload(verified_commit))
    assert pra.release_promotion_eligible(verified_commit) is True


def test_promotion_eligibility_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent or damaged evidence is not evidence of promotion."""

    monkeypatch.setattr(pra, "CONTROL_PLANE_RELEASE_STATE_ROOT", tmp_path)

    # nothing installed at all
    assert pra.release_promotion_eligible(COMMIT) is False

    # a verified receipt for a DIFFERENT commit must not vouch for this one
    _write_provenance(tmp_path, COMMIT, _verified_payload("c" * 40))
    assert pra.release_promotion_eligible(COMMIT) is False

    # the lane claim itself must be present and true
    payload = _verified_payload(COMMIT)
    payload["claim_boundary"] = {"canonical_full_lane_verified": False}
    _write_provenance(tmp_path, COMMIT, payload)
    assert pra.release_promotion_eligible(COMMIT) is False


def _disjoint_roots(tmp_path: Path) -> dict[str, str]:
    """The deploy refuses overlapping roots before it reads any provenance."""

    for name in ("repo", "releases", "state"):
        (tmp_path / name).mkdir()
    return {
        "source_repo": str(tmp_path / "repo"),
        "release_root": str(tmp_path / "releases"),
        "state_root": str(tmp_path / "state"),
        "active_link": str(tmp_path / "active"),
    }


def test_iteration_deploy_refuses_a_provenance_receipt(tmp_path: Path) -> None:
    """Passing both says two different things about the same release."""

    with pytest.raises(deploy_mod.ControlPlaneDeployError) as excinfo:
        deploy_mod.deploy_control_plane_commit(
            source_commit=COMMIT,
            release_provenance=str(tmp_path / "provenance.json"),
            iteration=True,
            **_disjoint_roots(tmp_path),
        )
    assert "deploy_iteration_provenance_conflict" in str(excinfo.value)


def test_a_normal_deploy_still_demands_lane_provenance(tmp_path: Path) -> None:
    """Relaxing the gate for iteration must not relax it by default."""

    with pytest.raises(deploy_mod.ControlPlaneDeployError) as excinfo:
        deploy_mod.deploy_control_plane_commit(
            source_commit=COMMIT,
            release_provenance=None,
            iteration=False,
            **_disjoint_roots(tmp_path),
        )
    assert "deploy_release_provenance_missing" in str(excinfo.value)


def _repo_with_main(tmp_path: Path) -> tuple[Path, str, str]:
    """A real repo with an origin/main ref, a merged commit, and a stray one."""

    import subprocess

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    run = lambda *a, **kw: subprocess.run(  # noqa: E731
        list(a), cwd=kw.get("cwd", upstream), check=True,
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "GIT_AUTHOR_NAME": "t",
             "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path)},
    )
    run("git", "init", "-q", "-b", "main")
    (upstream / "f").write_text("1")
    run("git", "add", "f")
    run("git", "commit", "-qm", "one")

    clone = tmp_path / "clone"
    run("git", "clone", "-q", str(upstream), str(clone), cwd=tmp_path)
    merged = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    # a commit that exists locally but was never merged to main
    (clone / "f").write_text("2")
    subprocess.run(["git", "-C", str(clone), "add", "f"], check=True,
                   capture_output=True)
    subprocess.run(
        ["git", "-C", str(clone), "commit", "-qm", "local only"], check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "GIT_AUTHOR_NAME": "t",
             "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
             "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path)},
    )
    unmerged = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return clone, merged, unmerged


def test_iteration_refuses_a_commit_that_is_not_on_origin_main(
    tmp_path: Path,
) -> None:
    """The guard must live in the tool, not only in the wrapper.

    Within an hour of the wrapper being written its `git fetch` hit a
    permission error, and the obvious workaround was to call this tool
    directly -- which then had no ancestry check at all. Iteration exists to
    skip the lane, never to skip main, so an unmerged commit must be refused
    however the tool is invoked.
    """

    clone, _merged, unmerged = _repo_with_main(tmp_path)
    roots = _disjoint_roots(tmp_path)
    roots["source_repo"] = str(clone)

    with pytest.raises(deploy_mod.ControlPlaneDeployError) as excinfo:
        deploy_mod.deploy_control_plane_commit(
            source_commit=unmerged,
            release_provenance=None,
            iteration=True,
            **roots,
        )
    assert "deploy_iteration_commit_not_on_origin_main" in str(excinfo.value)


def test_iteration_accepts_a_commit_that_is_on_origin_main(
    tmp_path: Path,
) -> None:
    """A merged commit clears the ancestry guard and proceeds past it."""

    clone, merged, _unmerged = _repo_with_main(tmp_path)
    roots = _disjoint_roots(tmp_path)
    roots["source_repo"] = str(clone)

    with pytest.raises(deploy_mod.ControlPlaneDeployError) as excinfo:
        deploy_mod.deploy_control_plane_commit(
            source_commit=merged,
            release_provenance=None,
            iteration=True,
            **roots,
        )
    # It fails later on this synthetic host, but NOT on ancestry.
    assert "deploy_iteration_commit_not_on_origin_main" not in str(excinfo.value)


def test_the_wrapper_fetches_main_specifically(tmp_path: Path) -> None:
    """A bare `git fetch origin` cannot write new remote ref locks.

    The service account gets "Permission denied" creating
    .git/refs/remotes/origin/<branch>.lock for unrelated branches, which is
    what made the wrapper look broken and invited bypassing it.
    """

    text = (REPO / "scripts" / "deploy_control_plane_iteration.sh").read_text()
    assert "git fetch -q origin main" in text
    assert "git fetch -q origin\n" not in text
    assert "merge-base --is-ancestor" in text
