from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.adp_founder_sim_protocol import (
    build_founder_approval_receipt,
    expected_founder_approval_statement,
)
from blueprint_pipeline.adp_isaac_lab_arena_vast import (
    DEFAULT_IMAGE,
    PROBE_KIND,
    _next_attempt_root,
    _remaining_session_live_minutes,
    build_arena_native_control_bundle,
)
from blueprint_pipeline.common import write_json
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
    _resolve_probe_image,
)


def _approval_path(tmp_path: Path) -> Path:
    receipt = build_founder_approval_receipt(
        statement=expected_founder_approval_statement(),
        evidence_ref="codex-task://test/user-message",
    )
    path = tmp_path / "approval.json"
    write_json(path, receipt)
    return path


def test_bundle_is_deterministic_approved_and_native_control_only(tmp_path: Path) -> None:
    approval = _approval_path(tmp_path)
    first = build_arena_native_control_bundle(
        approval_path=approval, job_dir=tmp_path / "first", generated_at="fixed"
    )
    second = build_arena_native_control_bundle(
        approval_path=approval, job_dir=tmp_path / "second", generated_at="fixed"
    )

    assert first["status"] == "ready"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["candidate_policy_queried"] is False
    assert first["candidate_outcomes_accessed"] is False
    assert first["container_image"] == DEFAULT_IMAGE
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        names = set(archive.namelist())
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
        runner = archive.read("provider_runtime/adp_arena_provider_runner.py").decode()
    assert "provider_runtime/founder_sim_approval_receipt.json" in names
    assert "provider_runtime/arena_worker_request.json" in names
    assert "GR00T-N1.6-DROID" not in runner
    assert "pi05_droid_jointpos_polaris" not in runner
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="adp_arena",
            entrypoint_text=entrypoint,
            runner_text=runner,
        )
        == []
    )
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp_arena",
        bundle_path=Path(first["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?sig=redacted",
        provider_output_put_url="https://example.com/output.zip?sig=redacted",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_arena_bundle_uses_isaac_image_terms_and_ssh_bundle_path(tmp_path: Path) -> None:
    assert (
        _resolve_probe_image(
            public_image="public",
            isaac_image=DEFAULT_IMAGE,
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_arena",
        )
        == DEFAULT_IMAGE
    )
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=True,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_arena",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=True,
        provider_bundle_kind="adp_arena",
        forward_hf_token=False,
    )
    assert env["ACCEPT_EULA"] == "Y"
    assert env["PRIVACY_CONSENT"] == "Y"
    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_arena",
    )
    assert "run_adp_arena_provider_runtime.sh" in script
    assert "adp_arena_provider_runtime_output.zip" in script
    assert "BLUEPRINT_VAST_CUDA_RUNTIME_DEFERRED_TO_ISAAC_SIMULATION_APP" in script
    assert "wp.get_devices()" in script
    assert "isaac_simulation_app_warp" in script
    assert script.rindex("BLUEPRINT_VAST_GPU_SANITY_OK") > script.index(
        "BLUEPRINT_VAST_PROVIDER_BUNDLE_STARTED"
    )


def test_paid_attempt_roots_are_fresh_and_preserve_prior_evidence(tmp_path: Path) -> None:
    write_json(tmp_path / "adp_arena_vast_session_budget.json", {"attempt_count": 1})
    prior = tmp_path / "attempts" / "attempt_002"
    prior.mkdir(parents=True)
    (prior / "prior_evidence.txt").write_text("preserve", encoding="utf-8")

    number, root = _next_attempt_root(tmp_path)

    assert number == 3
    assert root == tmp_path / "attempts" / "attempt_003"
    assert (prior / "prior_evidence.txt").read_text(encoding="utf-8") == "preserve"


def test_successor_ttl_reserves_only_remaining_cumulative_budget(tmp_path: Path) -> None:
    write_json(
        tmp_path / "adp_arena_vast_session_budget.json",
        {
            "attempts": [
                {
                    "actual_live_runtime_seconds_observed_by_adapter": 560.9,
                    "estimated_cost_usd": 0.083795,
                }
            ]
        },
    )

    assert (
        _remaining_session_live_minutes(
            job=tmp_path,
            hard_cap_usd=4.0,
            hard_ttl_seconds=14_400,
            max_hourly_rate_usd=1.0,
        )
        == 230
    )


def test_paid_attempt_roots_are_fresh_and_preserve_prior_evidence(tmp_path: Path) -> None:
    write_json(tmp_path / "adp_arena_vast_session_budget.json", {"attempt_count": 1})
    prior = tmp_path / "attempts" / "attempt_002"
    prior.mkdir(parents=True)
    (prior / "prior_evidence.txt").write_text("preserve", encoding="utf-8")

    number, root = _next_attempt_root(tmp_path)

    assert number == 3
    assert root == tmp_path / "attempts" / "attempt_003"
    assert (prior / "prior_evidence.txt").read_text(encoding="utf-8") == "preserve"


def _allocator_args(tmp_path: Path, approval: Path, *, execute: bool) -> list[str]:
    values = [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp-arena",
        "--adp-arena-approval",
        str(approval),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "4.0",
        "--adp-hard-ttl-seconds",
        "14400",
    ]
    if execute:
        values.append("--execute")
    return values


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_arena_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    approval = _approval_path(tmp_path)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_arena_native_control_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, approval, execute=execute)) == 0
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["candidate_policy_queried"] is False
    assert admission["hard_cap_usd"] == 4.0
    assert admission["hard_ttl_seconds"] == 14400
