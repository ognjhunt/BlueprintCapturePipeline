"""Fail-closed registry-diagnostic gate for paid worker-image launches.

The 2026-07-11 live canary retained a 900s cutoff because the runner read the
stale generic default diagnostic instead of the diagnostic for the exact
selected digest. These tests pin the corrected contract: a paid digest-pinned
launch or paid image-startup canary must consume a completed, digest-matching
isaac_worker_image_manifest_diagnostic.v2 artifact, and the evidence-derived
startup timeout floor must actually reach the launch request and spend guard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import blueprint_pipeline.isaac_g1_kitchen_parity_job as J

DIGEST = "sha256:" + "b" * 64
DIGEST_IMAGE = f"docker.io/example/blueprint-groot-oscar-eval@{DIGEST}"
_SCENARIOS = [
    {
        "scenario_id": "s1",
        "spawn_position_xyz": [0, 0, 0],
        "target_position_xyz": [1, 0, 0],
    }
]


def _valid_diagnostic_payload(**overrides) -> dict:
    payload = {
        "schema_version": "isaac_worker_image_manifest_diagnostic.v2",
        "status": "completed",
        "image_ref": "docker.io/example/blueprint-groot-oscar-eval:20260711",
        "resolved_digest": DIGEST,
        "resolved_digest_ref": DIGEST_IMAGE,
        "runnable_platform": "linux/amd64",
        "layer_count": 35,
        "total_compressed_size_bytes": 50_455_509_186,
        "largest_layer_size_bytes": 10_585_790_213,
        "layers_over_1gb": 4,
        "large_image_pull_risk": True,
        "split_layer_layout_suitable": False,
        "recommended_startup_no_runtime_timeout_seconds": 1800,
    }
    payload.update(overrides)
    return payload


def _write_diagnostic(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _policy(**kwargs) -> tuple[str, dict]:
    defaults = dict(
        image=DIGEST_IMAGE,
        allow_paid=True,
        provider_names=["runpod"],
        cold=True,
        warm_only=False,
        image_startup_canary=True,
    )
    defaults.update(kwargs)
    return J._paid_worker_image_policy(**defaults)


def test_stale_generic_default_diagnostic_blocks_exact_digest_launch(
    tmp_path: Path, monkeypatch
) -> None:
    stale = _write_diagnostic(
        tmp_path / "stale_generic.json",
        _valid_diagnostic_payload(
            image_ref="docker.io/example/old-generic-worker:v1",
            resolved_digest="sha256:" + "e" * 64,
            resolved_digest_ref=(
                "docker.io/example/old-generic-worker@sha256:" + "e" * 64
            ),
        ),
    )
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV, str(stale))

    _selected, policy = _policy()

    assert policy["status"] == "blocked"
    assert "worker_image_diagnostic_image_ref_mismatch" in policy["blockers"]
    diagnostic = policy["worker_image_manifest_diagnostic"]
    assert diagnostic["metadata_available_for_selected_image"] is False


def test_explicitly_supplied_mismatched_diagnostic_blocks(tmp_path: Path) -> None:
    mismatched = _write_diagnostic(
        tmp_path / "mismatched.json",
        _valid_diagnostic_payload(
            resolved_digest="sha256:" + "f" * 64,
            resolved_digest_ref=(
                "docker.io/example/blueprint-groot-oscar-eval@sha256:" + "f" * 64
            ),
        ),
    )

    _selected, policy = _policy(worker_image_manifest_diagnostic=mismatched)

    assert policy["status"] == "blocked"
    assert "worker_image_diagnostic_image_ref_mismatch" in policy["blockers"]
    assert (
        policy["worker_image_manifest_diagnostic"]["path_source"] == "cli_argument"
    )


def test_missing_diagnostic_blocks_paid_large_image_canary(tmp_path: Path) -> None:
    _selected, policy = _policy(
        worker_image_manifest_diagnostic=tmp_path / "does_not_exist.json"
    )

    assert policy["status"] == "blocked"
    assert "worker_image_diagnostic_missing_for_paid_launch" in policy["blockers"]


def test_matching_mutable_default_still_requires_explicit_path(
    tmp_path: Path, monkeypatch
) -> None:
    diagnostic = _write_diagnostic(
        tmp_path / "matching_default.json", _valid_diagnostic_payload()
    )
    monkeypatch.setenv(J.ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV, str(diagnostic))

    _selected, policy = _policy()

    assert policy["status"] == "blocked"
    assert "worker_image_diagnostic_explicit_path_required" in policy["blockers"]


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "recommended_startup_no_runtime_timeout_seconds": 1,
            "large_image_pull_risk": False,
            "split_layer_layout_suitable": True,
        },
        {"layers_over_1gb": 36},
        {"largest_layer_size_bytes": 60_000_000_000},
    ],
)
def test_size_derived_policy_mismatch_blocks_paid_launch(
    tmp_path: Path, overrides: dict
) -> None:
    diagnostic = _write_diagnostic(
        tmp_path / "inconsistent.json", _valid_diagnostic_payload(**overrides)
    )

    _selected, policy = _policy(worker_image_manifest_diagnostic=diagnostic)

    assert policy["status"] == "blocked"
    assert "worker_image_diagnostic_size_policy_inconsistent" in policy["blockers"]


def test_effective_timeout_is_rederived_from_registry_sizes() -> None:
    diagnostic = _valid_diagnostic_payload(
        recommended_startup_no_runtime_timeout_seconds=1,
        large_image_pull_risk=False,
        split_layer_layout_suitable=True,
    )

    effective = J._effective_startup_no_runtime_timeout(900, diagnostic)

    assert effective["image_manifest_recommended_seconds"] == 1800
    assert effective["effective_seconds"] == 1800


@pytest.mark.parametrize(
    ("payload_text", "expected_blocker"),
    [
        ("{not json", "worker_image_diagnostic_unreadable"),
        (
            json.dumps(
                _valid_diagnostic_payload(
                    schema_version="isaac_worker_image_manifest_diagnostic.v99"
                )
            ),
            "worker_image_diagnostic_schema_unsupported",
        ),
        (
            json.dumps(_valid_diagnostic_payload(status="blocked")),
            "worker_image_diagnostic_status_not_completed",
        ),
        (
            json.dumps(_valid_diagnostic_payload(status=None)),
            "worker_image_diagnostic_status_not_completed",
        ),
        (
            json.dumps(
                _valid_diagnostic_payload(
                    total_compressed_size_bytes=None, largest_layer_size_bytes=None
                )
            ),
            "worker_image_diagnostic_size_metadata_missing",
        ),
        (
            json.dumps(
                _valid_diagnostic_payload(
                    recommended_startup_no_runtime_timeout_seconds=None
                )
            ),
            "worker_image_diagnostic_timeout_recommendation_invalid",
        ),
        (
            json.dumps(
                _valid_diagnostic_payload(
                    recommended_startup_no_runtime_timeout_seconds=10**9
                )
            ),
            "worker_image_diagnostic_timeout_recommendation_invalid",
        ),
        (
            json.dumps(
                _valid_diagnostic_payload(
                    recommended_startup_no_runtime_timeout_seconds=1800.5
                )
            ),
            "worker_image_diagnostic_timeout_recommendation_invalid",
        ),
        (
            json.dumps(
                _valid_diagnostic_payload(resolved_digest="sha256:" + "a" * 64)
            ),
            "worker_image_diagnostic_resolved_digest_mismatch",
        ),
        (
            json.dumps(_valid_diagnostic_payload(runnable_platform=None)),
            "worker_image_diagnostic_runnable_platform_unverified",
        ),
        (
            json.dumps({"image_ref": "", "resolved_digest_ref": ""}),
            "worker_image_diagnostic_image_ref_mismatch",
        ),
    ],
)
def test_malformed_or_incomplete_diagnostics_block(
    tmp_path: Path, payload_text: str, expected_blocker: str
) -> None:
    path = tmp_path / "diagnostic.json"
    path.write_text(payload_text, encoding="utf-8")

    _selected, policy = _policy(worker_image_manifest_diagnostic=path)

    assert policy["status"] == "blocked"
    assert expected_blocker in policy["blockers"]


def test_tag_only_paid_launch_without_canary_does_not_require_diagnostic() -> None:
    _selected, policy = _policy(
        image="docker.io/example/small-worker:v1",
        image_startup_canary=False,
        worker_image_manifest_diagnostic=None,
    )

    validation = policy["worker_image_diagnostic_validation"]
    assert validation["required"] is False
    assert validation["status"] == "not_required"


def test_paid_canary_refuses_tag_selected_image_even_with_tag_matching_diagnostic(
    tmp_path: Path,
) -> None:
    tag = "docker.io/example/blueprint-groot-oscar-eval:mutable"
    diagnostic = _write_diagnostic(
        tmp_path / "tag.json", _valid_diagnostic_payload(image_ref=tag)
    )

    _selected, policy = _policy(
        image=tag, worker_image_manifest_diagnostic=diagnostic
    )

    assert policy["status"] == "blocked"
    assert "paid_image_startup_canary_requires_digest_pinned_image" in policy["blockers"]
    assert policy["worker_image_diagnostic_validation"]["debug_override_supported"] is False


def test_zero_requested_timeout_still_uses_matching_diagnostic_floor() -> None:
    policy = J._effective_startup_no_runtime_timeout(
        0, _valid_diagnostic_payload()
    )
    assert policy["effective_seconds"] == 1800
    assert policy["raised_to_image_manifest_floor"] is True
    assert policy["disabled"] is False


def _paid_canary_run(tmp_path: Path, monkeypatch, *, diagnostic_path: Path, **run_kwargs):
    """Full paid canary run with fakes; returns (manifest, captured)."""
    monkeypatch.setenv(J.ISAAC_G1_MAX_SPEND_USD_ENV, "50.0")
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _fake_stage(bundle_zip, job_dir, *, key_prefix):
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "provider_bundle_url.txt").write_text("https://spaces.example/b?sig=A")
        (job_dir / "provider_output_put_url.txt").write_text("https://spaces.example/o?sig=B")
        (job_dir / "provider_output_get_url.txt").write_text("https://spaces.example/o?sig=C")
        return {"status": "completed", "manifest": {}}

    class _FakeProvider:
        name = "runpod"

        def available(self) -> dict:
            return {"provider": self.name, "available": True}

        def build_request(self, spec, job_dir):
            return {"env": dict(spec.env), "image": spec.image}

        def billable_inventory(self, *, name_prefix: str) -> dict:
            return {
                "status": "observed",
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
                "name_prefix": name_prefix,
            }

    captured: dict = {}

    def _fake_launch(provider_obj, job_dir, request, **kwargs):
        captured["request"] = request
        captured["launch_kwargs"] = kwargs
        return {
            "status": "launched",
            "instance_id": "runpod-canary",
            "mode": "cold_create_marker_verified",
        }

    def _fake_watch(job_dir, render_out, instance_id, *, provider=None, **_kwargs):
        return {
            "status": "completed",
            "elapsed_seconds": 1,
            "teardown": {"status": "terminated"},
            "runner_result_source": "isaac_g1_kitchen_parity_result.json",
            "last_bootstrap": {"phase": "runner_done"},
            "timed_out_without_runner_done": False,
            "runner_result": {
                "schema_version": "isaac_g1_parity_image_startup_canary.v2",
                "status": "completed",
                "image_startup_canary": True,
            },
        }

    monkeypatch.setattr(J, "get_render_provider", lambda name, warm_candidates=(): _FakeProvider())
    monkeypatch.setattr(J, "stage_bundle", _fake_stage)
    monkeypatch.setattr(J, "launch_with_marker_retry", _fake_launch)
    monkeypatch.setattr(J, "watch_and_collect", _fake_watch)

    kwargs = dict(
        scenarios=[],
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        image=DIGEST_IMAGE,
        image_startup_canary=True,
        cold_race_contenders=1,
        marker_timeout=900,
        startup_no_runtime_timeout=900,
        max_attempts=1,
        worker_image_manifest_diagnostic=diagnostic_path,
    )
    kwargs.update(run_kwargs)
    return J.run_isaac_g1_kitchen_parity_job(**kwargs), captured


def test_matching_diagnostic_raises_timeout_floor_and_reaches_launch(
    tmp_path: Path, monkeypatch
) -> None:
    diagnostic = _write_diagnostic(
        tmp_path / "diag.json", _valid_diagnostic_payload()
    )

    m, captured = _paid_canary_run(tmp_path, monkeypatch, diagnostic_path=diagnostic)

    assert m["status"] == "completed"
    validation = m["worker_image_policy"]["worker_image_diagnostic_validation"]
    assert validation["status"] == "passed"
    assert validation["diagnostic_sha256"]
    # Requested 900 / recommended 1800 -> effective 1800; the floor was raised.
    timeout_policy = m["startup_no_runtime_timeout_policy"]
    assert timeout_policy["requested_seconds"] == 900
    assert timeout_policy["image_manifest_recommended_seconds"] == 1800
    assert timeout_policy["effective_seconds"] == 1800
    assert timeout_policy["raised_to_image_manifest_floor"] is True
    assert timeout_policy["image_manifest_resolved_digest"] == DIGEST
    assert timeout_policy["image_manifest_diagnostic_sha256"]
    # Marker timeout must exceed the effective pre-runtime timeout by >=120s.
    marker_policy = m["startup_marker_timeout_policy"]
    assert marker_policy["effective_seconds"] >= 1920
    # The effective values reach the launch request shape (provider boundary).
    assert m["launch_request_shape"]["startup_no_runtime_timeout_seconds"] == 1800
    assert m["launch_request_shape"]["marker_timeout_seconds"] >= 1920
    assert captured["launch_kwargs"]["startup_no_runtime_timeout"] == 1800
    assert captured["launch_kwargs"]["marker_timeout"] >= 1920
    # Spend guard is computed from the effective values, not the discarded 900s.
    guard = m["prelaunch_spend_guard"]
    assert guard["startup_no_runtime_timeout_seconds"] == 1800
    assert guard["marker_timeout_seconds"] >= 1920
    assert guard["startup_budget_seconds"] == 3840
    # Evidence, never secrets.
    assert m["worker_image_policy"]["worker_image_manifest_diagnostic"][
        "raw_secret_values_recorded"
    ] is False
    assert "hf_token" not in json.dumps(m).lower()
    # The lane lease was held for the mutation window and released after the
    # watch/teardown section completed.
    lease = m["paid_provider_lane_lease"]
    assert lease["status"] == "acquired"
    assert lease["release"]["reason"] == "watch_and_collect_finished"
    assert lease["release"]["results"][0]["status"] == "released"


def test_blocked_diagnostic_prevents_staging_and_provider_launch(
    tmp_path: Path, monkeypatch
) -> None:
    stale = _write_diagnostic(
        tmp_path / "stale.json",
        _valid_diagnostic_payload(
            resolved_digest="sha256:" + "e" * 64,
            resolved_digest_ref=(
                "docker.io/example/blueprint-groot-oscar-eval@sha256:" + "e" * 64
            ),
        ),
    )
    monkeypatch.setattr(
        J,
        "_git_worktree_evidence",
        lambda: {"status": "available", "git_sha": "abc123", "dirty": False},
    )

    def _stage_must_not_run(*_args, **_kwargs):
        raise AssertionError("blocked diagnostic must stop the job before staging")

    def _launch_must_not_run(*_args, **_kwargs):
        raise AssertionError("blocked diagnostic must stop the job before provider launch")

    monkeypatch.setattr(J, "stage_bundle", _stage_must_not_run)
    monkeypatch.setattr(J, "launch_with_marker_retry", _launch_must_not_run)
    monkeypatch.setattr(J, "race_launch", _launch_must_not_run)

    m = J.run_isaac_g1_kitchen_parity_job(
        scenarios=_SCENARIOS,
        out_dir=tmp_path / "job",
        provider="runpod",
        allow_paid=True,
        allow_dirty_paid_launch=True,
        image=DIGEST_IMAGE,
        image_startup_canary=True,
        cold_race_contenders=1,
        worker_image_manifest_diagnostic=stale,
    )

    assert m["status"] == "blocked"
    assert "worker_image_diagnostic_image_ref_mismatch" in m["blockers"]
    assert "staging" not in m
    assert "launch" not in m
