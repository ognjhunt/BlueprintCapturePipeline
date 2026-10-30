"""G1's automatic launcher rebuilds current-release bundles without a second CLI."""
import hashlib
from pathlib import Path
import subprocess

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline import beta_policy_screening_g1_launch as g1_launch
from blueprint_pipeline import native_g1_beta_bundle as bundle
from blueprint_pipeline.native_g1_beta_session import validate_session_authority, PROBE_KIND
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_g1_beta_session import inputs


def setup(tmp_path, monkeypatch):
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(launch.__file__).resolve().parents[2], text=True).strip()
    candidate["execution_spec"]["implementation_commit"] = current
    candidate["admission_digest"] = canonical_digest(candidate["execution_spec"])
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    reopened, builds = [], []
    def reopen(actual_plan, actual_candidate):
        assert actual_plan == plan and actual_candidate == candidate
        reopened.append(actual_candidate["candidate_id"])
        return spec
    monkeypatch.setattr(g1_launch, "reopen_candidate", reopen)
    def build(*, job_dir, authority, spec, implementation_commit):
        # Exact Git/source archive verification has real archive tests. This
        # boundary records the production caller's inputs, without staging
        # model/runtime archives or masking identity, quote or authority checks.
        validate_session_authority(authority)
        assert implementation_commit == current == authority["implementation_commit"]
        assert authority["runtime_inputs_digest"] == spec["execution_spec_digest"]
        assert len(reopened) == len(builds) + 1
        builds.append(job_dir)
        job_dir.mkdir()
        archive = job_dir / "hermetic-bundle.zip"
        archive.write_bytes(b"development-only builder-boundary fixture")
        return {"bundle_path": str(archive), "bundle_size_bytes": archive.stat().st_size,
            "bundle_sha256": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
            "implementation_commit": implementation_commit}
    monkeypatch.setattr(bundle, "build_bundle", build)
    return plan, candidate, reopened, builds


def test_public_beta_launcher_rebuilds_each_new_attempt_at_actual_current_commit(tmp_path, monkeypatch):
    plan, candidate, reopened, builds = setup(tmp_path, monkeypatch)
    for name in ("attempt-one", "attempt-two"):
        root = tmp_path / name
        prepared = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=root)
        assert prepared["argv"][:5] == ["gpu-canary", "--provider", "vast", "--probe-kind", PROBE_KIND]
        assert Path(prepared["bundle_receipt"]["path"]).parent == root
        assert launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=root) == prepared
        assert not (root / "allocator_launch_intent.json").exists()
        assert not (root / "allocator/authorization_consumption.json").exists()
    assert builds == [tmp_path / "attempt-one/bundle", tmp_path / "attempt-two/bundle"]
    assert reopened == [candidate["candidate_id"]] * 2


def test_old_release_plan_never_rebuilds_or_relabels_a_current_bundle(tmp_path, monkeypatch):
    plan, candidate, reopened, builds = setup(tmp_path, monkeypatch)
    candidate["execution_spec"]["implementation_commit"] = "0" * 40
    candidate["admission_digest"] = canonical_digest(candidate["execution_spec"])
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    with pytest.raises(ValueError, match="g1_beta_implementation_commit_changed"):
        launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "stale-attempt")
    assert not reopened and not builds
    assert not (tmp_path / "stale-attempt").exists()


def test_automatic_rebuild_still_reopens_bound_source_before_sealing(tmp_path, monkeypatch):
    plan, candidate, _, builds = setup(tmp_path, monkeypatch)
    def changed(*args):
        raise ValueError("g1_beta_prepared_execution_changed")
    monkeypatch.setattr(g1_launch, "reopen_candidate", changed)
    with pytest.raises(ValueError, match="g1_beta_prepared_execution_changed"):
        launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "changed-source")
    assert not builds and not (tmp_path / "changed-source").exists()
