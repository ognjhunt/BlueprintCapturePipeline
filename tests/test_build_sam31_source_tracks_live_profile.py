from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.sam31_gpu_admission import (
    CHECKPOINT_DIGEST,
    CHECKPOINT_REPOSITORY_REVISION,
    LICENSE_TERMS_DIGEST,
    OFFICIAL_CODE_REVISION,
    REQUEST_SCHEMA_VERSION,
)
from blueprint_pipeline.sam31_paid_attempt_authority import (
    materialize_sam31_paid_attempt_authority,
)
from blueprint_pipeline.sam31_paid_resource_allocator_lane import (
    run_sam31_paid_resource_allocator_lane,
)
from blueprint_pipeline.sam31_source_track_canary_worker import BUNDLE_RECEIPT_SCHEMA_VERSION
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

pytestmark = pytest.mark.usefixtures(
    "_materialize_generated_manifest_publication_fixture"
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_sam31_source_tracks_live_profile",
    REPO_ROOT / "scripts" / "build_sam31_source_tracks_live_profile.py",
)
builder = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(builder)

COMMIT = "a" * 40
URI = f"https://raw.githubusercontent.com/example/repo/{COMMIT}/sam31-request.json"


def _fixture(tmp_path: Path) -> dict[str, Path]:
    import hashlib

    tmp_path.mkdir(parents=True, exist_ok=True)
    bundle = tmp_path / "sam31-input.zip"
    bundle.write_bytes(b"sam31-bundle")
    bundle_digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    evidence_digest = "sha256:" + "a" * 64
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "operation": "source_track_canary",
        "source_profile": "monocular_video",
        "source_commit_sha": COMMIT,
        "worker_image_digest": "registry.example/sam31@sha256:" + "b" * 64,
        "worker_stack_manifest_digest": evidence_digest,
        "input_bundle_digest": bundle_digest,
        "input_bundle_size_bytes": bundle.stat().st_size,
        "source_track_run_request_digest": evidence_digest,
        "capture_digest": evidence_digest,
        "retained_video_digest": evidence_digest,
        "camera_solution_digest": evidence_digest,
        "frame_registry_digest": evidence_digest,
        "frame_count": 8,
        "checkpoint_family": "facebook/sam3.1",
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "checkpoint_repository_revision": CHECKPOINT_REPOSITORY_REVISION,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "license_terms_digest": LICENSE_TERMS_DIGEST,
        "license_use_authorization_digest": evidence_digest,
        "privacy_use_authorization_digest": evidence_digest,
        "trade_controls_review_digest": evidence_digest,
        "execution_authorization_digest": evidence_digest,
        "checkpoint_access_authorized": True,
        "commercial_evidence_use_authorized": True,
        "rights_cleared_for_external_processing": True,
        "privacy_safe_for_external_processing": True,
        "trade_controls_reviewed": True,
        "model_self_grading_forbidden": True,
        "metric_claim_upgrade_forbidden": True,
        "physics_claim_upgrade_forbidden": True,
        "physical_claim_upgrade_forbidden": True,
        "network_access_during_inference_forbidden": True,
        "customer_data_training_allowed": False,
        "allowed_evidence_uses": ["semantic_analysis"],
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 600,
        "retry_cap": 0,
        "authority_id": "fresh-scene-sam31-1",
        "proof_effect": "none",
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = tmp_path / "sam31-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    receipt = {
        "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle": {
            "filename": bundle.name,
            "sha256": bundle_digest,
            "size_bytes": bundle.stat().st_size,
        },
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = tmp_path / "sam31-input-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    authority_path = tmp_path / "sam31-authority.json"
    materialize_sam31_paid_attempt_authority(
        request_path=request_path,
        bundle_path=bundle,
        bundle_receipt_path=receipt_path,
        authorization_reference="goal thread",
        authorized_by="fixture-user",
        authorized_on="2026-08-13",
        blueprint_commit=COMMIT,
        max_hourly_rate_usd=0.5,
        hard_cap_usd=1.0,
        hard_ttl_seconds=600,
        aggregate_goal_spend_before_attempt_usd=0.0,
        aggregate_goal_spend_cap_usd=12.0,
        output_path=authority_path,
    )
    token = tmp_path / "hf-token.txt"
    token.write_text("fixture-secret", encoding="utf-8")
    token.chmod(0o600)
    return {
        "request": request_path,
        "bundle": bundle,
        "receipt": receipt_path,
        "authority": authority_path,
        "token": token,
    }


def _build(paths: dict[str, Path], **overrides):
    return builder.build_sam31_source_tracks_live_profile(
        request_path=paths["request"],
        input_bundle_path=paths["bundle"],
        input_bundle_receipt_path=paths["receipt"],
        attempt_authority_path=paths["authority"],
        hf_token_file=paths["token"],
        source_commit=COMMIT,
        raw_manifest_uri=URI,
        **overrides,
    )


def test_builds_publishable_zero_retry_sam_profile(tmp_path: Path) -> None:
    profile = _build(_fixture(tmp_path))
    argv = profile["allocator"]["argv"]
    assert profile["execution_admission"]["live_enabled"] is True
    assert profile["allocator"]["retry_cap"] == 0
    assert "--sam31-attempt-authority" in argv
    assert "--sam31-input-bundle" in argv
    assert "--sam31-max-hourly-rate-usd" in argv
    assert "--provider-bundle-url-file" not in argv
    assert profile["terminal_contract"]["required_values"] == {
        "continuing_spend_from_this_run": False,
        "retry_cap": 0,
    }
    assert "source_track_import_result_path" in profile["terminal_contract"][
        "required_path_fields"
    ]


def test_published_profile_reaches_real_dry_run_with_fresh_preflight(
    tmp_path: Path,
) -> None:
    """Join the published argv to the allocator that consumes its preflight.

    The profile used to name a file beneath the fresh launch root that no
    producer wrote. A fixture that merely creates that file cannot reproduce
    the production contract: the allocator requires a fresh, digest-bound Vast
    capacity/provider-zero snapshot collected under an armed watchdog.
    """

    profile = _build(_fixture(tmp_path / "inputs"))
    run_root = (tmp_path / "launch-run").resolve()
    (run_root / "allocator").mkdir(parents=True)
    argv = [str(value).replace("{launch_run_root}", str(run_root)) for value in profile["allocator"]["argv"]]

    def argument(name: str) -> str:
        return argv[argv.index(name) + 1]

    args = Namespace(
        provider_launch_request=argument("--provider-launch-request"),
        preflight_bundle=argument("--preflight-bundle"),
        admission_out=argument("--admission-out"),
        bound_request_out=argument("--bound-request-out"),
        adapter_output=argument("--adapter-output"),
        provider=argument("--provider"),
        expected_source_commit=argument("--expected-source-commit"),
        sam31_max_spend_usd=float(argument("--sam31-max-spend-usd")),
        sam31_max_hourly_rate_usd=float(argument("--sam31-max-hourly-rate-usd")),
        sam31_hard_ttl_seconds=int(argument("--sam31-hard-ttl-seconds")),
        sam31_retry_cap=int(argument("--sam31-retry-cap")),
        sam31_authority_id=argument("--sam31-authority-id"),
        sam31_input_bundle=argument("--sam31-input-bundle"),
        sam31_input_bundle_receipt=argument("--sam31-input-bundle-receipt"),
        sam31_attempt_authority=argument("--sam31-attempt-authority"),
        sam31_allowed_active_vast_instance_id=[],
        sam31_hf_token_file=argument("--sam31-hf-token-file"),
        execute=False,
    )
    # A profile build cannot freeze a five-minute capacity snapshot. Even if a
    # stale/structurally incomplete file is already present, the allocator must
    # replace it with bytes from the canonical live producer before admission.
    Path(args.preflight_bundle).write_text(
        json.dumps({"schema_version": "semantic_sam31_gpu_provider_preflight.v1"}),
        encoding="utf-8",
    )

    class ReadOnlyProvider:
        def capacity_preflight(self, _request):
            return {
                "status": "available",
                "selected_offer": {
                    "gpu_name": "L40S",
                    "gpu_ram_mb": 48_000,
                    "on_demand_price_usd_per_hour": 0.5,
                },
            }

        def billable_inventory(self, *, name_prefix: str):
            return {
                "api_confirmed": True,
                "live_resource_count": 0,
                "resources": [],
                "name_prefix": name_prefix,
            }

    started = run_root / "allocator" / "started-vast-instance-id.txt"
    closed: list[bool] = []
    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit=COMMIT,
        provider_factory=lambda _name: ReadOnlyProvider(),
        arm_watchdog=lambda **_kwargs: (
            {
                "watchdog_pid": 123,
                "watchdog_started_epoch": 1_000,
                "watchdog_deadline_epoch": 9_999_999_999,
                "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-",
            },
            SimpleNamespace(started_instance_id_path=started),
        ),
        close_watchdog_without_allocation=lambda **_kwargs: (
            closed.append(True) or {"status": "provider_terminal"}
        ),
    )

    preflight = json.loads(Path(args.preflight_bundle).read_text(encoding="utf-8"))
    assert result["status"] == "dry_run_ready"
    assert result["blockers"] == []
    assert preflight["schema_version"] == "semantic_sam31_gpu_provider_preflight.v1"
    assert preflight["status"] == "verified"
    assert preflight["provider_inventory_verified_zero"] is True
    assert preflight["preflight_digest"] == canonical_digest(
        preflight, digest_field="preflight_digest"
    )
    assert closed == [True]


def test_profile_binds_all_nonsecret_immutable_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = {row["name"]: row for row in _build(paths)["immutable_inputs"]}
    assert set(inputs) == {
        "source_bundle_manifest",
        "evaluation_run_spec",
        "sam31_input_bundle",
        "sam31_paid_attempt_authority",
        "manifest_publication_receipt",
    }
    assert all("token" not in row["name"] for row in inputs.values())


def test_profile_refuses_nonprivate_hf_token(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["token"].chmod(0o644)
    with pytest.raises(TaskEvaluationLaunchError, match="permissions_not_0600"):
        _build(paths)


def test_profile_accepts_canonical_root_service_group_secret_mode(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["token"].chmod(0o640)

    profile = _build(paths)

    assert profile["execution_admission"]["live_enabled"] is True


def test_profile_refuses_bundle_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["bundle"].write_bytes(b"tampered")
    with pytest.raises(TaskEvaluationLaunchError, match="sam31_paid_authority_invalid"):
        _build(paths)
