from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.adp009d_native_microcheck_bundle import PROBE_KIND
from blueprint_pipeline.adp009d_policy_provisioning import (
    build_provisioning_script,
)
from blueprint_pipeline.adp009d_policy_runtime_smoke_bundle import (
    build_policy_runtime_smoke_bundle,
)
from blueprint_pipeline.adp009d_policy_runtime_smoke_worker import (
    seal_policy_runtime_smoke,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_shell_script,
)
from blueprint_pipeline.wam_compute_providers import WamComputeLaunchSpec


COMMIT = "1" * 40


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_bundle_is_one_query_outcome_blind_and_stops_server(
    tmp_path: Path, candidate_id: str
) -> None:
    receipt = build_policy_runtime_smoke_bundle(
        job_dir=tmp_path / candidate_id,
        candidate_id=candidate_id,
        implementation_commit=COMMIT,
        generated_at="2026-08-24T00:00:00Z",
    )

    assert receipt["status"] == "ready"
    assert receipt["execution_mode"] == "outcome_blind_policy_runtime_smoke"
    assert receipt["policy_candidate_id"] == candidate_id
    assert receipt["identity_binding"]["synthetic_query_count"] == 1
    assert receipt["identity_binding"]["actions_executed"] is False
    assert receipt["identity_binding"]["task_scene_loaded"] is False
    assert receipt["controls_requested"] is False
    assert receipt["retry_cap"] == 0
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        provisioning = archive.read(
            f"provider_runtime/adp009d_policy_provisioning.{candidate_id}.sh"
        ).decode()
        entrypoint = archive.read(
            "provider_runtime/run_adp_arena_provider_runtime.sh"
        ).decode()
    assert "--stop-after-round-trip" in provisioning
    assert "provider_runtime/adp009d_policy_runtime_smoke_worker.py" in names
    assert "adp009d_native_microcheck.json" in entrypoint
    assert "task_success" not in entrypoint
    assert "adp_arena_provider_runner.py" not in names
    if candidate_id == "pi05_droid":
        assert "provider_runtime/adp009d_policy_execution_spec.json" in names
        assert "provider_runtime/adp009d_openpi_checkpoint_inventory.json" in names


def test_normal_episode_provisioning_keeps_server_running() -> None:
    assert "--stop-after-round-trip" not in build_provisioning_script("pi05_droid")


def test_vast_provider_accepts_and_routes_policy_runtime_smoke_bundle(
    tmp_path: Path,
) -> None:
    receipt = build_policy_runtime_smoke_bundle(
        job_dir=tmp_path / "bundle",
        candidate_id="pi05_droid",
        implementation_commit=COMMIT,
        generated_at="2026-08-24T00:00:00Z",
    )

    spec = WamComputeLaunchSpec(
        name="policy-smoke-pi05",
        bundle_path=receipt["bundle_path"],
        provider_bundle_kind="adp009d_policy_runtime_smoke",
        expected_video_count=0,
    )
    assert spec.provider_bundle_kind == "adp009d_policy_runtime_smoke"
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="2026-08-24T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind=spec.provider_bundle_kind,
        bundle_path=Path(spec.bundle_path),
        provider_bundle_url="https://staging.example/bundle.zip",
        provider_output_put_url="https://staging.example/output.zip",
    )
    assert preflight["status"] == "passed"
    assert preflight["missing_zip_entries"] == []

    script = _probe_shell_script(
        "https://heartbeat.example",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind=spec.provider_bundle_kind,
    )
    assert "run_adp_arena_provider_runtime.sh" in script
    assert "adp_arena_provider_runtime_output.zip" in script


def _server_receipt(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_id": "pi05_droid",
        "status": "ready",
        "round_trip_completed": True,
        "action_chunk_rows": 15,
        "action_chunk_width": 8,
        "server_stopped_after_round_trip": True,
        "server_pid": 999_999_999,
        "transport": "openpi_websocket",
    }
    value.update(updates)
    return value


def test_worker_seals_one_runtime_query_without_task_claim(tmp_path: Path) -> None:
    server = tmp_path / "server.json"
    server.write_text(json.dumps(_server_receipt()), encoding="utf-8")

    result = seal_policy_runtime_smoke(
        candidate_id="pi05_droid",
        server_receipt_path=server,
        provisioning_exit_code=0,
    )

    assert result["status"] == "completed"
    assert result["synthetic_observation_query_count"] == 1
    assert result["candidate_policy_queried"] is True
    assert result["candidate_outcomes_accessed"] is False
    assert result["claim_boundary"]["actions_executed"] is False
    assert result["claim_boundary"]["task_scene_loaded"] is False
    assert result["claim_boundary"]["task_success_claimed"] is False


def test_worker_refuses_outcome_fields(tmp_path: Path) -> None:
    server = tmp_path / "server.json"
    server.write_text(
        json.dumps(_server_receipt(task_success=False)), encoding="utf-8"
    )

    result = seal_policy_runtime_smoke(
        candidate_id="pi05_droid",
        server_receipt_path=server,
        provisioning_exit_code=0,
    )

    assert result["status"] == "blocked"
    assert "policy_runtime_smoke_outcome_field_forbidden" in result["blockers"]


def _allocator_args(tmp_path: Path, candidate_id: str) -> list[str]:
    return [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "request.json"),
        "--release-evidence",
        str(tmp_path / "release.json"),
        "--model-cache-evidence",
        str(tmp_path / "model.json"),
        "--preflight-bundle",
        str(tmp_path / "preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        f"policy-smoke-{candidate_id}",
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp009d-policy-candidate",
        candidate_id,
        "--adp009d-policy-runtime-smoke",
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "1.5",
        "--adp-hard-ttl-seconds",
        "5400",
    ]


@pytest.mark.parametrize("candidate_id", ["pi05_droid", "groot_n17_droid"])
def test_allocator_dry_run_routes_runtime_smoke_without_task_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate_id: str
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": COMMIT, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "build_policy_runtime_smoke_bundle",
        lambda **kwargs: observed.update(kwargs)
        or {
            "status": "ready",
            "bundle_sha256": "sha256:" + "b" * 64,
            "input_digest": "sha256:" + "c" * 64,
            "policy_candidate_id": candidate_id,
            "execution_mode": "outcome_blind_policy_runtime_smoke",
        },
    )
    monkeypatch.setattr(
        allocator,
        "run_adp009d_native_microcheck_vast",
        lambda **kwargs: observed.update({"run": kwargs}) or {"status": "dry_run_ready"},
    )
    args = _allocator_args(tmp_path, candidate_id)
    if candidate_id == "groot_n17_droid":
        monkeypatch.setattr(allocator, "normalize_model_access_env", lambda: None)
        monkeypatch.setattr(
            allocator,
            "model_access_secret_status",
            lambda: {"huggingface": {"auth_ready": True}},
        )
        monkeypatch.setattr(
            allocator,
            "probe_gated_backbone_access",
            lambda: {
                "status": "authorized",
                "receipt_digest": "sha256:" + "d" * 64,
                "blockers": [],
            },
        )
        args.append("--adp009d-authorize-gated-backbone")

    assert allocator.main(args) == 0
    assert observed["candidate_id"] == candidate_id
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["candidate_policy_queried"] is True
    assert admission["allocation_binding"]["outcome_blind_policy_runtime_smoke"] is True
    assert admission["allocation_binding"]["controls_requested"] is False
