from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_gpu_admission import (
    CHECKPOINT_DIGEST,
    CHECKPOINT_REPOSITORY_REVISION,
    LICENSE_TERMS_DIGEST,
    OFFICIAL_CODE_REVISION,
    PREFLIGHT_SCHEMA_VERSION,
    build_sam31_gpu_canary_admission,
)
from blueprint_pipeline.sam31_provider_launch_packet import (
    EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
    LICENSE_AUTHORIZATION_SCHEMA_VERSION,
    PRIVACY_AUTHORIZATION_SCHEMA_VERSION,
    RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_VERSION,
    TRADE_CONTROLS_SCHEMA_VERSION,
    WORKER_STACK_SCHEMA_VERSION,
    Sam31ProviderLaunchPacketError,
    main,
    materialize_sam31_execution_authorization,
    materialize_sam31_gpu_canary_request,
    materialize_sam31_provider_profile,
    materialize_sam31_worker_stack_manifest,
)
from blueprint_pipeline.sam31_source_track_canary_worker import (
    build_sam31_source_track_input_bundle,
)
from blueprint_pipeline.scene_placement.sam31_source_track_provider import (
    FRAME_INPUT_MODE,
    RUN_REQUEST_SCHEMA_VERSION,
    RUNTIME_API,
    _validate_profile,
)
from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)


COMMIT = "a" * 40
IMAGE = "registry.example/blueprint/sam31@sha256:" + "b" * 64
RUNTIME_DIGEST = "sha256:" + "b" * 64


def _write_receipt(path: Path, value: dict, *, field: str = "receipt_digest") -> Path:
    value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _profile_sources(tmp_path: Path) -> dict[str, Path]:
    stack = _write_receipt(
        tmp_path / "worker-stack.json",
        {
            "schema_version": WORKER_STACK_SCHEMA_VERSION,
            "source_commit_sha": COMMIT,
            "runtime_image_identity": IMAGE,
            "runtime_digest": RUNTIME_DIGEST,
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "checkpoint_repository_revision": CHECKPOINT_REPOSITORY_REVISION,
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "license_terms_digest": LICENSE_TERMS_DIGEST,
            "manifest_digest": "",
        },
        field="manifest_digest",
    )
    image_build = _write_receipt(
        tmp_path / "runtime-image-build.json",
        {
            "schema_version": RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_VERSION,
            "status": "published",
            "source_commit_sha": COMMIT,
            "runtime_image_identity": IMAGE,
            "runtime_digest": RUNTIME_DIGEST,
            "official_code_revision": OFFICIAL_CODE_REVISION,
            "registry_api_digest_verified": True,
            "dockerfile_sha256": "sha256:" + "4" * 64,
            "source_tree_digest": "sha256:" + "5" * 64,
            "build_provenance_digest": "sha256:" + "6" * 64,
            "receipt_digest": "",
        },
    )
    human_authority = {
        "authorized_by": "fixture-human",
        "authorized_on": "2026-08-13",
        "authority_reference": "fixture retained human authority",
        "authority_issued_by_agent": False,
    }
    license_use = _write_receipt(
        tmp_path / "license.json",
        {
            "schema_version": LICENSE_AUTHORIZATION_SCHEMA_VERSION,
            "status": "accepted",
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "license_terms_digest": LICENSE_TERMS_DIGEST,
            "checkpoint_access_authorized": True,
            "commercial_evidence_use_authorized": True,
            "customer_data_training_allowed": False,
            "allowed_evidence_uses": ["semantic_analysis"],
            **human_authority,
            "receipt_digest": "",
        },
    )
    privacy = _write_receipt(
        tmp_path / "privacy.json",
        {
            "schema_version": PRIVACY_AUTHORIZATION_SCHEMA_VERSION,
            "status": "accepted",
            "rights_cleared_for_external_processing": True,
            "privacy_safe_for_external_processing": True,
            "customer_data_training_allowed": False,
            **human_authority,
            "receipt_digest": "",
        },
    )
    trade = _write_receipt(
        tmp_path / "trade.json",
        {
            "schema_version": TRADE_CONTROLS_SCHEMA_VERSION,
            "status": "reviewed",
            "checkpoint_digest": CHECKPOINT_DIGEST,
            "trade_controls_reviewed": True,
            **human_authority,
            "receipt_digest": "",
        },
    )
    execution = _write_receipt(
        tmp_path / "execution.json",
        {
            "schema_version": EXECUTION_AUTHORIZATION_SCHEMA_VERSION,
            "status": "authorized",
            "source_commit_sha": COMMIT,
            "runtime_image_identity": IMAGE,
            "external_execution_authorized": True,
            "network_access_during_inference_forbidden": True,
            "model_self_grading_forbidden": True,
            "metric_claim_upgrade_forbidden": True,
            "physics_claim_upgrade_forbidden": True,
            "physical_claim_upgrade_forbidden": True,
            **human_authority,
            "receipt_digest": "",
        },
    )
    return {
        "stack": stack,
        "image_build": image_build,
        "license": license_use,
        "privacy": privacy,
        "trade": trade,
        "execution": execution,
    }


def _profile(tmp_path: Path) -> tuple[dict, dict[str, Path], Path]:
    sources = _profile_sources(tmp_path)
    output = tmp_path / "provider-profile-packet.json"
    packet = materialize_sam31_provider_profile(
        worker_stack_manifest_path=sources["stack"],
        runtime_image_build_receipt_path=sources["image_build"],
        license_use_authorization_path=sources["license"],
        privacy_use_authorization_path=sources["privacy"],
        trade_controls_review_path=sources["trade"],
        execution_authorization_path=sources["execution"],
        source_commit_sha=COMMIT,
        runtime_image_identity=IMAGE,
        method_version="sam3.1-96914d24",
        output_probability_threshold=0.5,
        max_num_objects=5,
        multiplex_count=16,
        use_fa3=False,
        compile_model=False,
        warm_up=False,
        async_loading_frames=False,
        output_path=output,
    )
    return packet, sources, output


def _run_request(tmp_path: Path, profile: dict) -> Path:
    frames = []
    artifacts = []
    for index in range(2):
        image_path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (8, 6), (20 + index, 30, 40)).save(
            image_path, format="JPEG"
        )
        import hashlib

        digest = "sha256:" + hashlib.sha256(image_path.read_bytes()).hexdigest()
        frame_id = f"task-a:camera-{index}"
        frames.append(
            {
                "source_frame_id": frame_id,
                "model_frame_index": index,
                "source_frame_digest": "sha256:" + str(index + 1) * 64,
                "retained_video_digest": "sha256:" + "d" * 64,
                "decoded_pts_seconds": float(index),
                "sync_map_row_digest": "sha256:" + "e" * 64,
                "camera_record_digest": "sha256:" + str(index + 3) * 64,
                "encoder_retained": True,
                "width": 8,
                "height": 6,
                "analysis_jpeg_digest": digest,
            }
        )
        artifacts.append(
            {
                "source_frame_id": frame_id,
                "path": str(image_path),
                "media_type": "image/jpeg",
                "sha256": digest,
                "size_bytes": image_path.stat().st_size,
            }
        )
    request = {
        "schema_version": RUN_REQUEST_SCHEMA_VERSION,
        "bindings": {
            "capture_digest": "sha256:" + "f" * 64,
            "retained_video_digest": "sha256:" + "d" * 64,
            "camera_solution_digest": "sha256:" + "1" * 64,
            "frame_registry_digest": canonical_json_digest(frames),
        },
        "frame_registry": frames,
        "frame_artifacts": artifacts,
        "provider_profile": profile,
        "prompts": [
            {"prompt_id": "washer", "text": "washer", "output_label": "washer"}
        ],
        "allowed_evidence_uses": ["semantic_analysis"],
    }
    path = tmp_path / "source-track-run-request.json"
    path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _launch_fixture(tmp_path: Path) -> dict[str, object]:
    profile, sources, profile_path = _profile(tmp_path)
    run_request = _run_request(tmp_path, profile)
    bundle = tmp_path / "sam31-input.zip"
    receipt = tmp_path / "sam31-input-receipt.json"
    build_sam31_source_track_input_bundle(
        request_path=run_request,
        bundle_path=bundle,
        receipt_path=receipt,
    )
    return {
        "profile": profile,
        "profile_path": profile_path,
        "sources": sources,
        "run_request": run_request,
        "bundle": bundle,
        "receipt": receipt,
    }


def _materialize_request(tmp_path: Path, fixture: dict[str, object]) -> dict:
    return materialize_sam31_gpu_canary_request(
        provider_profile_path=fixture["profile_path"],
        source_track_run_request_path=fixture["run_request"],
        input_bundle_path=fixture["bundle"],
        input_bundle_receipt_path=fixture["receipt"],
        source_profile="monocular_video",
        source_commit_sha=COMMIT,
        expected_camera_count=2,
        expected_frame_count=2,
        max_spend_usd=1.0,
        hard_ttl_seconds=600,
        retry_cap=0,
        authority_id="scene-840920-sam31",
        output_path=tmp_path / "gpu-canary-request-packet.json",
    )


def test_materializes_exact_provider_profile_from_source_bytes(tmp_path: Path) -> None:
    packet, sources, _ = _profile(tmp_path)
    profile = packet
    assert profile["runtime_api"] == RUNTIME_API
    assert profile["frame_input_mode"] == FRAME_INPUT_MODE
    assert profile["official_code_revision"] == OFFICIAL_CODE_REVISION
    assert profile["runtime_image_identity"] == IMAGE
    assert profile["checkpoint_digest"] == CHECKPOINT_DIGEST
    assert profile["max_num_objects"] == 5
    assert profile["multiplex_count"] == 16
    assert profile["profile_digest"] == canonical_json_digest(
        {key: value for key, value in profile.items() if key != "profile_digest"}
    )
    assert profile["license_use_authorization_digest"] == profile[
        "authorization_sources"
    ]["license_use"]["sha256"]
    assert profile["provider_mutations_performed"] == 0
    assert profile["runtime_image_build_receipt"]["receipt_digest"]
    assert all(path.is_file() for path in sources.values())
    blockers: list[str] = []
    _validate_profile(
        {
            "provider_profile": profile,
            "allowed_evidence_uses": ["semantic_analysis"],
        },
        blockers,
    )
    assert blockers == []


def test_cli_materializes_missing_worker_stack_and_execution_authority(
    tmp_path: Path,
) -> None:
    stack_path = tmp_path / "worker-stack-cli.json"
    execution_path = tmp_path / "execution-authority-cli.json"
    assert main(
        [
            "worker-stack",
            "--source-commit",
            COMMIT,
            "--runtime-image-identity",
            IMAGE,
            "--output",
            str(stack_path),
        ]
    ) == 0
    assert main(
        [
            "execution-authorization",
            "--source-commit",
            COMMIT,
            "--runtime-image-identity",
            IMAGE,
            "--authorized-by",
            "fixture-human",
            "--authorized-on",
            "2026-08-14T00:00:00Z",
            "--authority-reference",
            "retained user authorization",
            "--output",
            str(execution_path),
        ]
    ) == 0
    stack = json.loads(stack_path.read_text())
    execution = json.loads(execution_path.read_text())
    assert stack == materialize_sam31_worker_stack_manifest(
        source_commit_sha=COMMIT,
        runtime_image_identity=IMAGE,
        output_path=tmp_path / "worker-stack-function.json",
    )
    assert execution == materialize_sam31_execution_authorization(
        source_commit_sha=COMMIT,
        runtime_image_identity=IMAGE,
        authorized_by="fixture-human",
        authorized_on="2026-08-14T00:00:00Z",
        authority_reference="retained user authorization",
        output_path=tmp_path / "execution-authority-function.json",
    )
    assert stack["manifest_digest"] == canonical_digest(
        stack, digest_field="manifest_digest"
    )
    assert execution["receipt_digest"] == canonical_digest(
        execution, digest_field="receipt_digest"
    )
    assert execution["authority_issued_by_agent"] is False


def test_materialized_gpu_request_passes_existing_dry_admission(tmp_path: Path) -> None:
    packet = _materialize_request(tmp_path, _launch_fixture(tmp_path))
    request = packet
    assert request["worker_image_digest"] == IMAGE
    assert request["camera_count"] == request["frame_count"] == 2
    assert request["retry_cap"] == 0
    assert request["provider_mutations_performed"] == 0
    assert request["source_records"]["runtime_image_build_receipt"][
        "receipt_digest"
    ]
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "verified",
        "provider": "vast",
        "observed_at_epoch": 1_000.0,
        "provider_api_verified": True,
        "provider_inventory_verified_zero": True,
        "conflicting_owner_present": False,
        "watchdog": {"status": "armed", "independent_process": True},
        "single_gpu_available": True,
        "gpu_memory_bytes": 48 * 1024**3,
        "container_disk_bytes": 80 * 1024**3,
        "on_demand_price_usd_per_hour": 0.5,
    }
    admission, bound = build_sam31_gpu_canary_admission(
        request=request,
        preflight=preflight,
        provider="vast",
        expected_source_commit=COMMIT,
        checkout_source_commit=COMMIT,
        checkout_clean=True,
        max_spend_usd=1.0,
        hard_ttl_seconds=600,
        retry_cap=0,
        authority_id="scene-840920-sam31",
        execute=False,
        observed_now_epoch=1_001.0,
    )
    assert admission["status"] == "dry_run_ready"
    assert admission["blockers"] == []
    assert bound["provider_mutation_authorized"] is False


def test_cli_materializes_raw_profile_and_raw_gpu_request(tmp_path: Path) -> None:
    sources = _profile_sources(tmp_path)
    profile_path = tmp_path / "cli-provider-profile.json"
    assert main(
        [
            "profile",
            "--worker-stack-manifest",
            str(sources["stack"]),
            "--runtime-image-build-receipt",
            str(sources["image_build"]),
            "--license-use-authorization",
            str(sources["license"]),
            "--privacy-use-authorization",
            str(sources["privacy"]),
            "--trade-controls-review",
            str(sources["trade"]),
            "--execution-authorization",
            str(sources["execution"]),
            "--source-commit",
            COMMIT,
            "--runtime-image-identity",
            IMAGE,
            "--method-version",
            "sam3.1-96914d24",
            "--output-probability-threshold",
            "0.5",
            "--max-num-objects",
            "5",
            "--multiplex-count",
            "16",
            "--use-fa3",
            "false",
            "--compile-model",
            "false",
            "--warm-up",
            "false",
            "--async-loading-frames",
            "false",
            "--output",
            str(profile_path),
        ]
    ) == 0
    profile = json.loads(profile_path.read_text())
    assert profile["method_id"] == "meta.sam3.1.object_multiplex"
    assert "profile" not in profile
    run_request = _run_request(tmp_path, profile)
    bundle = tmp_path / "cli-input.zip"
    receipt = tmp_path / "cli-input-receipt.json"
    build_sam31_source_track_input_bundle(
        request_path=run_request,
        bundle_path=bundle,
        receipt_path=receipt,
    )
    gpu_request_path = tmp_path / "cli-gpu-request.json"
    assert main(
        [
            "gpu-request",
            "--provider-profile",
            str(profile_path),
            "--source-track-run-request",
            str(run_request),
            "--input-bundle",
            str(bundle),
            "--input-bundle-receipt",
            str(receipt),
            "--source-profile",
            "monocular_video",
            "--source-commit",
            COMMIT,
            "--expected-camera-count",
            "2",
            "--expected-frame-count",
            "2",
            "--max-spend-usd",
            "1.0",
            "--hard-ttl-seconds",
            "600",
            "--retry-cap",
            "0",
            "--authority-id",
            "scene-840920-sam31",
            "--output",
            str(gpu_request_path),
        ]
    ) == 0
    request = json.loads(gpu_request_path.read_text())
    assert request["schema_version"] == "semantic_sam31_gpu_canary_request.v1"
    assert "gpu_canary_request" not in request


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("rights_bytes", "authorization_bytes_changed"),
        ("bundle_bytes", "configuration_invalid"),
        ("camera_count", "configuration_invalid"),
        ("retry", "configuration_invalid"),
    ],
)
def test_request_materializer_fails_closed_on_changed_or_unbound_inputs(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    fixture = _launch_fixture(tmp_path)
    kwargs = {
        "provider_profile_path": fixture["profile_path"],
        "source_track_run_request_path": fixture["run_request"],
        "input_bundle_path": fixture["bundle"],
        "input_bundle_receipt_path": fixture["receipt"],
        "source_profile": "monocular_video",
        "source_commit_sha": COMMIT,
        "expected_camera_count": 2,
        "expected_frame_count": 2,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 600,
        "retry_cap": 0,
        "authority_id": "scene-840920-sam31",
        "output_path": tmp_path / "gpu-canary-request-packet.json",
    }
    if mutation == "rights_bytes":
        sources = fixture["sources"]
        assert isinstance(sources, dict)
        Path(sources["privacy"]).write_text("{}\n", encoding="utf-8")
    elif mutation == "bundle_bytes":
        Path(fixture["bundle"]).write_bytes(b"changed")
    elif mutation == "camera_count":
        kwargs["expected_camera_count"] = 1
    elif mutation == "retry":
        kwargs["retry_cap"] = 1
    with pytest.raises(Sam31ProviderLaunchPacketError, match=expected):
        materialize_sam31_gpu_canary_request(**kwargs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("commercial_evidence_use_authorized", False),
        ("authority_issued_by_agent", True),
        ("authorized_by", ""),
    ],
)
def test_profile_refuses_unreviewed_or_unattributed_rights(
    tmp_path: Path, field: str, replacement: object
) -> None:
    sources = _profile_sources(tmp_path)
    license_value = json.loads(sources["license"].read_text())
    license_value[field] = replacement
    _write_receipt(sources["license"], license_value)
    with pytest.raises(Sam31ProviderLaunchPacketError, match="authorization_source_invalid"):
        materialize_sam31_provider_profile(
            worker_stack_manifest_path=sources["stack"],
            runtime_image_build_receipt_path=sources["image_build"],
            license_use_authorization_path=sources["license"],
            privacy_use_authorization_path=sources["privacy"],
            trade_controls_review_path=sources["trade"],
            execution_authorization_path=sources["execution"],
            source_commit_sha=COMMIT,
            runtime_image_identity=IMAGE,
            method_version="sam3.1-96914d24",
            output_probability_threshold=0.5,
            max_num_objects=5,
            multiplex_count=16,
            use_fa3=False,
            compile_model=False,
            warm_up=False,
            async_loading_frames=False,
            output_path=tmp_path / "profile.json",
        )


def test_profile_refuses_checkpoint_incompatible_multiplex_count(tmp_path: Path) -> None:
    sources = _profile_sources(tmp_path)
    with pytest.raises(Sam31ProviderLaunchPacketError, match="configuration_invalid"):
        materialize_sam31_provider_profile(
            worker_stack_manifest_path=sources["stack"],
            runtime_image_build_receipt_path=sources["image_build"],
            license_use_authorization_path=sources["license"],
            privacy_use_authorization_path=sources["privacy"],
            trade_controls_review_path=sources["trade"],
            execution_authorization_path=sources["execution"],
            source_commit_sha=COMMIT,
            runtime_image_identity=IMAGE,
            method_version="sam3.1-96914d24",
            output_probability_threshold=0.5,
            max_num_objects=5,
            multiplex_count=5,
            use_fa3=False,
            compile_model=False,
            warm_up=False,
            async_loading_frames=False,
            output_path=tmp_path / "profile.json",
        )


def test_profile_refuses_unpinned_runtime_image(tmp_path: Path) -> None:
    sources = _profile_sources(tmp_path)
    with pytest.raises(Sam31ProviderLaunchPacketError, match="configuration_invalid"):
        materialize_sam31_provider_profile(
            worker_stack_manifest_path=sources["stack"],
            runtime_image_build_receipt_path=sources["image_build"],
            license_use_authorization_path=sources["license"],
            privacy_use_authorization_path=sources["privacy"],
            trade_controls_review_path=sources["trade"],
            execution_authorization_path=sources["execution"],
            source_commit_sha=COMMIT,
            runtime_image_identity="registry.example/sam31:latest",
            method_version="sam3.1-96914d24",
            output_probability_threshold=0.5,
            max_num_objects=5,
            multiplex_count=16,
            use_fa3=False,
            compile_model=False,
            warm_up=False,
            async_loading_frames=False,
            output_path=tmp_path / "profile.json",
        )
