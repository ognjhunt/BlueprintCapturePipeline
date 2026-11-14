from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import policy_ranking_successor_gpu_admission as admission


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "docs/experiments/policy_ranking_cosmos3_followup_20260728"
HISTORICAL_EXPERIMENT = ROOT / "docs/experiments/policy_ranking_successor_experiment_20260727"
BUNDLE_NAME = "cosmos3_followup_provider_bundle.zip"
BUNDLE_RECEIPT_NAME = "cosmos3_followup_provider_bundle_receipt.json"


def test_ctrl_world_profile_reservation_stays_within_target_spend() -> None:
    profile = admission.CTRL_WORLD_REPLAY_PROFILE
    projected_max_spend = profile.max_hourly_rate_usd * profile.hard_ttl_seconds / 3600.0

    assert profile.hard_ttl_seconds == 4_800
    assert projected_max_spend <= profile.target_spend_usd
    assert profile.target_spend_usd <= profile.max_compute_cap_usd


def _load(name: str) -> dict[str, Any]:
    path = EXPERIMENT / name
    if not path.is_file():
        path = HISTORICAL_EXPERIMENT / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if name == "vast_compute_preflight.json":
        value["experiment_id"] = "policy_ranking_cosmos3_followup_20260728"
    return value


def _inspect_bundle(path: Path | None = None) -> dict[str, Any]:
    return admission.inspect_successor_bundle(
        path or EXPERIMENT / BUNDLE_NAME,
        receipt=_load(BUNDLE_RECEIPT_NAME),
        smoke_inventory=_load("smoke_request_inventory.json"),
    )


def _preflight_path(tmp_path: Path) -> Path:
    path = tmp_path / "vast_compute_preflight.json"
    path.write_text(json.dumps(_load("vast_compute_preflight.json")), encoding="utf-8")
    return path


def _oscar_replay_fixture(
    tmp_path: Path,
    *,
    public_script_guard: bool = True,
) -> tuple[Path, dict[str, Any], admission.SuccessorGPUProfile]:
    bundle = tmp_path / "oscar-replay.zip"
    entrypoint = (
        "write_missing_result wam_runner_process_exited_without_runtime_result "
        "blocked_wam_process_exited_without_result"
    ).encode()
    guard = (
        "if official_case_smoke and not official_case_use_script:"
        if public_script_guard
        else "if official_case_smoke:"
    )
    runner = (
        "OSCAR-2B wam_runtime_result.json action_conditioned_video_rollout_generated\n" + guard
    ).encode()
    runtime_manifest = {
        "schema_version": "wam_provider_runtime_manifest.v1",
        "oscar_hf_repo": "zywu2115/OSCAR-2B",
        "oscar_hf_revision": "c9781ffa7dd8556d862d7d9f338a2ea008a58ca6",
        "oscar_source_ref": "4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb",
        "official_case_smoke": "droid_TRI",
        "official_case_use_script": "true",
        "official_case_rgb_video": "0",
        "fps": 14.0,
        "num_frames": 81,
        "height": 480,
        "width": 640,
        "num_steps": 35,
        "guidance": 6.0,
        "seed": 44,
    }
    rollout_manifest = {
        "schema_version": "wam_rollout_input_manifest.v1",
        "experiment_id": "policy_ranking_cosmos3_edge_closed_loop_20260729",
        "arm_id": "oscar_public_replay",
        "case_id": "droid_TRI",
        "physical_future_rgb_provided_to_model": False,
        "physical_outcome_labels_accessed": False,
    }
    payloads = {
        "provider_runtime/wam_provider_runtime_manifest.json": json.dumps(
            runtime_manifest, sort_keys=True
        ).encode(),
        "provider_runtime/wam_rollout_input_manifest.json": json.dumps(
            rollout_manifest, sort_keys=True
        ).encode(),
        "provider_runtime/oscar_input/first_frame.png": b"frozen-first-frame",
        "provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4": (
            b"frozen-skeleton"
        ),
        "provider_runtime/wam_provider_runtime_runner.py": runner,
        "provider_runtime/run_wam_provider_runtime.sh": entrypoint,
    }
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
    embedded = {
        "runtime_manifest_file_sha256": hashlib.sha256(
            payloads["provider_runtime/wam_provider_runtime_manifest.json"]
        ).hexdigest(),
        "rollout_manifest_file_sha256": hashlib.sha256(
            payloads["provider_runtime/wam_rollout_input_manifest.json"]
        ).hexdigest(),
        "first_frame_sha256": hashlib.sha256(
            payloads["provider_runtime/oscar_input/first_frame.png"]
        ).hexdigest(),
        "skeleton_video_sha256": hashlib.sha256(
            payloads["provider_runtime/oscar_input/blueprint_proxy_skeleton_conditioning.mp4"]
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(runner).hexdigest(),
        "entrypoint_sha256": hashlib.sha256(entrypoint).hexdigest(),
    }
    bundle_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    profile = replace(
        admission.OSCAR_PUBLIC_REPLAY_PROFILE,
        expected_bundle_sha256=bundle_sha256,
        expected_bundle_size_bytes=bundle.stat().st_size,
        expected_embedded_input_hashes=embedded,
    )
    receipt = {
        "schema_version": profile.receipt_schema,
        "experiment_id": profile.experiment_id,
        "bundle_sha256": bundle_sha256,
        "bundle_size_bytes": bundle.stat().st_size,
        **embedded,
    }
    return bundle, receipt, profile


def test_oscar_public_replay_profile_binds_exact_v4_bundle() -> None:
    profile = admission.OSCAR_PUBLIC_REPLAY_PROFILE

    assert profile.authorization_ids_by_allocation_index == {
        7: "policy-ranking-cosmos3-edge-closed-loop-20260729-allocation-7"
    }
    assert profile.expected_bundle_sha256 == (
        "d6447c8432eb9d484c64f61244eaec40739f9115d778390f6b1fef18c9564752"
    )
    assert profile.expected_bundle_size_bytes == 1_135_718
    assert profile.target_spend_usd == profile.max_compute_cap_usd == 5.0


def test_oscar_public_replay_bundle_passes_exact_contract(tmp_path: Path) -> None:
    bundle, receipt, profile = _oscar_replay_fixture(tmp_path)

    result = admission.inspect_successor_bundle(
        bundle,
        receipt=receipt,
        smoke_inventory={},
        profile=profile,
    )

    assert result["status"] == "passed", result["blockers"]
    assert result["rollout_manifest"]["physical_future_rgb_provided_to_model"] is False


def test_oscar_public_replay_rejects_stale_pre_script_preparation(tmp_path: Path) -> None:
    bundle, receipt, profile = _oscar_replay_fixture(tmp_path, public_script_guard=False)

    result = admission.inspect_successor_bundle(
        bundle,
        receipt=receipt,
        smoke_inventory={},
        profile=profile,
    )

    assert result["status"] == "blocked"
    assert "successor_oscar_bundle_public_script_ownership_guard_missing" in result["blockers"]


def test_phase_b_vast_preflight_explicitly_admits_blackwell_compute_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Provider:
        def capacity_preflight(self, request: dict[str, Any]) -> dict[str, Any]:
            captured.update(request)
            return {
                "status": "available",
                "viable_gpu_types": [
                    {
                        "ask_contract_id": 1,
                        "gpu_name": "RTX PRO 6000 WS",
                        "num_gpus": 1,
                        "gpu_ram_mb": 97_887,
                        "hourly_rate_usd": 0.88,
                        "reliability": 0.99,
                    }
                ],
            }

        def billable_inventory(self, *, name_prefix: str) -> dict[str, Any]:
            return {"api_confirmed": True, "live_resource_count": 0}

    monkeypatch.setattr(admission, "get_render_provider", lambda provider: Provider())
    result = admission.collect_successor_vast_preflight(name_prefix="phase-b-")

    assert result["status"] == "verified", result["blockers"]
    assert captured["min_compute_cap"] == 1200
    assert captured["max_compute_cap"] == 0
    assert captured["prefer_isaac_rt"] is False


@pytest.mark.parametrize("allocation_index", [1, 2, 3, 4])
def test_droid_reference_admission_accepts_runpod_without_smoke_inventory(
    allocation_index: int,
) -> None:
    profile = admission.DROID_REFERENCE_PROFILE
    environment = {
        "experiment_id": profile.experiment_id,
        "upstream_source": {
            "cosmos": {"revision": profile.cosmos_revision},
            "cosmos_framework": {"revision": profile.cosmos_framework_revision},
            "vllm_omni": {
                "revision": profile.vllm_omni_revision,
                "runtime_image": admission.PUBLIC_IMAGE,
            },
        },
        "checkpoint": {
            "repository": admission.CHECKPOINT_REPOSITORY,
            "revision": admission.CHECKPOINT_REVISION,
            "remote_code_policy": "no_unpinned_remote_code_and_trust_remote_code_false",
        },
    }
    authorization = {
        "schema_version": profile.authorization_schema,
        "experiment_id": profile.experiment_id,
        "authorization_id": profile.authorization_ids_by_allocation_index[allocation_index],
        "allocation_index": allocation_index,
        "maximum_provider_allocations": 1,
        "single_use_consumption_required": True,
        "paid_mutation_authorized": True,
        "authorized_compute_cap_usd": profile.max_compute_cap_usd,
        "per_allocation_maximum_spend_required": True,
        "prior_cumulative_compute_cap_superseded": True,
        "goal_cost_authorization_amendment_sha256": (profile.cost_authorization_binding_sha256),
        "hard_ttl_seconds": profile.hard_ttl_seconds,
        "one_resource_limit": True,
        "independent_teardown_watchdog": True,
        "watchdog_armed_before_allocation": True,
        "automatic_spend_cutoff": True,
        "teardown_required": True,
        "provider_zero_verification_required": True,
        "physical_robot_endpoint_access_allowed": False,
    }
    observed = 1234.0
    preflight = {
        "schema_version": profile.preflight_schema,
        "experiment_id": profile.experiment_id,
        "status": "verified",
        "provider": "runpod",
        "provider_inventory_verified_zero": True,
        "provider_mutations_performed": 0,
        "observed_at_epoch": observed,
        "selected_offer": {
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "gpu_ram_mb": 96_000,
            "hourly_rate_usd": 1.99,
            "cloud_type": "SECURE",
            "capacity_confidence": "advisory",
        },
    }

    result = admission.build_successor_gpu_admission(
        authorization=authorization,
        environment=environment,
        smoke_inventory={},
        provider_preflight=preflight,
        bundle_inspection={"status": "passed", "blockers": [], "bundle_sha256": "a" * 64},
        expected_source_commit="b" * 40,
        execute=True,
        observed_now_epoch=observed + 1,
        profile=profile,
    )

    assert result["status"] == "admitted", result["blockers"]
    assert result["provider"] == "runpod"
    assert result["smoke_inventory_validation"]["status"] == "not_applicable"
    assert result["shared_paid_lane_admission"]["resource_class"] == "runpod_wam_async"


def test_paid_resource_allocator_dispatches_successor_lane_only_through_probe_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda _commit, **_kwargs: ([], "d" * 40),
    )

    def fake_lane(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "dry_run_ready", "provider_mutations_performed": 0}

    monkeypatch.setattr(allocator, "run_successor_gpu_lane", fake_lane)
    code = allocator.main(
        [
            "gpu-canary",
            "--probe-kind",
            allocator.POLICY_RANKING_SUCCESSOR_COSMOS_PROBE_KIND,
            "--provider-launch-request",
            "authorization.json",
            "--release-evidence",
            "environment.json",
            "--model-cache-evidence",
            "inventory.json",
            "--preflight-bundle",
            "preflight.json",
            "--episode-bundle",
            "bundle.zip",
            "--successor-bundle-receipt",
            "receipt.json",
            "--provider-bundle-url-file",
            "bundle-url.txt",
            "--provider-output-put-url-file",
            "output-put-url.txt",
            "--provider-output-get-url-file",
            "output-get-url.txt",
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--pod-name",
            str(tmp_path / "job"),
            "--expected-source-commit",
            "d" * 40,
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}
    assert captured["execute"] is False
    assert captured["provider_bundle_path"] == "bundle.zip"
    assert captured["provider_bundle_receipt_path"] == "receipt.json"
    assert captured["provider_bundle_url_file"] == "bundle-url.txt"
    assert captured["provider_output_put_url_file"] == "output-put-url.txt"
    assert captured["provider_output_get_url_file"] == "output-get-url.txt"
    assert captured["expected_source_commit"] == "d" * 40
