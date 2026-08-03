from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from urllib.parse import urlsplit
import zipfile

import numpy as np
import pytest
import jsonschema

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.reconstruction_provider_contracts import (
    build_reconstruction_provider_admission,
    build_reconstruction_provider_execution_request,
)
from blueprint_pipeline.task_evaluation_supervisor.phase2_artifacts import (
    authorization_receipt,
    authorization_request,
)
from blueprint_pipeline.teleport_reconstruction_adapter import (
    TELEPORT_RESOURCE_CLASS,
    TeleportAdapterError,
    TeleportCredentials,
    TeleportHttpTransport,
    TeleportTransportError,
    load_teleport_credentials,
    run_teleport_reconstruction,
    validate_teleport_upload_packet,
    validate_uploaded_parts,
)


D = ["sha256:" + character * 64 for character in "abcdef"]
REQUIRED_ACTIONS = [
    "confidential_capture_upload",
    "provider_deletion",
    "provider_output_download",
    "provider_reconstruction_execution",
]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ply(path: Path, *, corrupt: bool = False) -> None:
    properties = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 2\n"
        + "".join(f"property float {name}\n" for name in properties)
        + "end_header\n"
    )
    if corrupt:
        path.write_bytes(b"not-a-ply")
        return
    rows = []
    for x in (0.0, 1.0):
        rows.append(
            struct.pack(
                "<14f", x, 0.0, 1.0, 0.2, 0.3, 0.4, 0.5,
                -4.0, -4.0, -4.0, 1.0, 0.0, 0.0, 0.0,
            )
        )
    path.write_bytes(header.encode("ascii") + b"".join(rows))


def _packet(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = tmp_path / "frame_00000.jpg"
    image.write_bytes(b"synthetic-public-candidate-image")
    archive = tmp_path / "candidate.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        for index in range(8):
            stream.writestr(f"frame_{index:05d}.jpg", image.read_bytes() + bytes([index]))
    archive_digest = _sha(archive)
    packet = {
        "schema_version": "teleport_ready_to_upload_packet.v1",
        "status": "ready_to_upload_authorization_required",
        "source_capture_digest": D[0],
        "dataset_class": "rights_cleared_public_dataset",
        "source_license": "synthetic-test-only",
        "customer_or_confidential_data_included": False,
        "frozen_split_digest": D[2],
        "candidate_dataset_digest": D[3],
        "deterministic_configuration_digest": D[1],
        "upload_zip": {
            "relative_path": archive.name,
            "digest": archive_digest,
            "size_bytes": archive.stat().st_size,
            "image_count": 8,
        },
        "upload_name_to_observation_id": {
            f"frame_{index:05d}.jpg": f"frame_{index:05d}" for index in range(8)
        },
        "hidden_images_included": False,
        "hidden_filenames_included": False,
        "provider_upload_authorized": False,
        "provider_mutations_performed": False,
        "proof_effect": "immutable_ready_to_upload_packet_only",
        "claim_ceiling": "none",
    }
    packet["teleport_ready_to_upload_packet_digest"] = canonical_digest(
        packet, digest_field="teleport_ready_to_upload_packet_digest"
    )
    path = tmp_path / "teleport_ready_to_upload_packet.v1.json"
    path.write_text(json.dumps(packet), encoding="utf-8")
    return path, archive_digest


def _request(
    archive_digest: str,
    *,
    quote: float = 4.0,
    ttl: int = 60,
    terms_digest: str = D[3],
) -> dict:
    inputs = sorted([D[0], D[1], D[2], archive_digest])
    auth_request = authorization_request(
        run_id="teleport-public-fixture",
        tool_id="invoke_authorized_reconstruction_provider",
        reason="Synthetic public-data lifecycle fixture",
        requested_max_cost_usd=8.0,
        requested_ttl_seconds=ttl,
        requested_retry_count=1,
        immutable_input_digests=inputs,
        requested_provider_ids=["teleport"],
        requested_action_ids=REQUIRED_ACTIONS,
    )
    issued = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
    expires = issued + timedelta(seconds=ttl)
    expires_text = expires.isoformat().replace("+00:00", "Z")
    auth = authorization_receipt(
        request=auth_request,
        operator_id="fixture-human-operator",
        approved=True,
        granted_max_cost_usd=8.0,
        granted_ttl_seconds=ttl,
        granted_retry_count=1,
        issued_at="2026-08-03T12:00:00Z",
        expires_at=expires_text,
        granted_provider_ids=["teleport"],
        granted_action_ids=REQUIRED_ACTIONS,
    )
    admission = build_reconstruction_provider_admission(
        {
            "stable_run_identity": "teleport-public-fixture",
            "provider_identity": "teleport",
            "provider_product": "teleport-api",
            "product_tier": "professional-pay-as-you-go",
            "terms_version": "fixture-reviewed-2026-08-03",
            "terms_digest": terms_digest,
            "legal_reviewer_identity": "fixture-human-legal-reviewer",
            "legal_reviewer_is_agent": False,
            "legal_review_receipt_digest": D[4],
            "provider_capability_review_digest": D[5],
            "source_capture_digest": D[0],
            "reviewed_at": "2026-08-03T12:00:00Z",
            "review_expires_at": "2026-08-10T12:00:00Z",
            "commercial_use_authorized": True,
            "confidential_capture_upload_authorized_by_terms": True,
            "retention_terms_acceptable": True,
            "deletion_process_verified": True,
            "model_training_terms_acceptable": True,
            "competitive_use_terms_acceptable": True,
            "resale_terms_acceptable": True,
            "benchmarking_terms_acceptable": True,
            "programmatic_upload_job_download_api_qualified": True,
            "canonical_paid_allocator_route_qualified": True,
            "trusted_legal_review_accepted": True,
            "provider_credentials_available": True,
            "provider_mutations_performed": False,
            "proof_effect": "none",
            "claim_ceiling": "none",
        }
    )
    return build_reconstruction_provider_execution_request(
        {
            "stable_run_identity": "teleport-public-fixture",
            "source_capture_identity": "synthetic-public-capture",
            "source_capture_digest": D[0],
            "original_file_references": [{"artifact_id": "capture", "digest": D[0]}],
            "producing_method": "teleport_execution_request_compiler",
            "implementation_version": "1",
            "source_commit_sha": "1" * 40,
            "deterministic_configuration_digest": D[1],
            "train_heldout_split_digest": D[2],
            "input_digests": [
                {"artifact_id": f"input-{index}", "digest": digest}
                for index, digest in enumerate(inputs)
            ],
            "output_digests": [],
            "camera_calibration_binding": {"status": "candidate_only_not_uploaded"},
            "coordinate_frame_declaration": {"status": "provider_scale_unverified"},
            "units": "unverified",
            "metric_scale_status": "unverified",
            "container_image_digest": None,
            "provider_runtime_identity": {
                "provider_identity": "teleport",
                "runtime_identity": "teleport-api-v1-v2",
                "provider_quote": {"quoted_cost_usd": quote, "quote_receipt_digest": D[5]},
                "training_parameters": {
                    "modelv3": {
                        "splat_count": 3_000_000,
                        "training_resolution": 3200,
                        "spherical_harmonics_degrees": 3,
                        "level_of_detail": False,
                    }
                },
                "alignment_thresholds": {
                    "maximum_rms_residual": 0.01,
                    "maximum_max_residual": 0.02,
                },
            },
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "provider_identity": "teleport",
            "provider_admission": admission,
            "provider_admission_digest": admission["provider_admission_digest"],
            "authorization_receipt": auth,
            "authorization_receipt_digest": auth["authorization_receipt_digest"],
            "authority_used": {"authorization_receipt_digest": auth["authorization_receipt_digest"]},
            "immutable_input_digests": inputs,
            "authorized_actions": REQUIRED_ACTIONS,
            "max_cost_usd": 8.0,
            "ttl_seconds": ttl,
            "retry_cap": 1,
            "timestamp": "2026-08-03T12:00:01Z",
            "authority_expires_at": expires_text,
            "warnings": [],
            "blockers": [],
            "parent_artifact_or_event": {"digest": admission["provider_admission_digest"]},
            "operator_upload_authorized": True,
            "confidential_capture_processing_authorized": True,
            "spending_authorized": True,
            "post_job_deletion_required": True,
            "authorization_issued_by_agent": False,
            "candidate_may_read_hidden_heldout": False,
            "proof_effect": "provider_execution_request_only",
            "claim_ceiling": "none",
        }
    )


def _observations() -> list[dict]:
    rng = np.random.default_rng(4)
    rows = []
    for index in range(8):
        matrix = np.eye(4)
        matrix[:3, 3] = rng.uniform(-1.0, 1.0, size=3)
        rows.append(
            {
                "observation_id": f"frame_{index:05d}",
                "camera": {
                    "T_world_camera": matrix.tolist(),
                    "rgb_intrinsics": {
                        "width": 64, "height": 48, "fx": 50.0, "fy": 50.0,
                        "cx": 32.0, "cy": 24.0,
                    },
                },
            }
        )
    return rows


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    @staticmethod
    def wall() -> str:
        return "2026-08-03T12:00:02Z"


class FakeTransport:
    def __init__(self, tmp_path: Path, observations: list[dict], *, mode: str = "success") -> None:
        self.mode = mode
        self.auth_count = 0
        self.create_count = 0
        self.complete_count = 0
        self.delete_count = 0
        self.state = "ABSENT"
        self.deleted = False
        self.poll_count = 0
        self.first_api = True
        self.ply = tmp_path / "provider.ply"
        _write_ply(self.ply, corrupt=mode == "corrupt_ply")
        rng = np.random.default_rng(9)
        cameras = []
        for index, observation in enumerate(observations):
            position = np.asarray(observation["camera"]["T_world_camera"])[:3, 3]
            if mode == "residual_failure":
                position = position + rng.normal(0.0, 0.3, 3)
            cameras.append(
                {"image_name": f"frame_{index:05d}.jpg", "position": position.tolist()}
            )
        self.camera_bytes = json.dumps({"cameras": cameras}).encode()

    def authenticate(self, credentials: TeleportCredentials) -> Mapping[str, Any]:
        assert credentials.client_id and credentials.client_secret
        self.auth_count += 1
        return {"access_token": f"token-{self.auth_count}", "expires_in": 3600, "token_type": "bearer"}

    def api_json(self, method, path, *, access_token, payload=None, query=None):
        if self.mode == "auth_expiry" and self.first_api:
            self.first_api = False
            return 401, {}, {"detail": "expired"}
        if method == "GET" and path == "/api/v1/captures":
            if self.deleted or self.state == "ABSENT":
                return 200, {}, []
            if self.state == "PROCESSING":
                self.poll_count += 1
                if self.mode == "failed_training":
                    self.state = "FAILED"
                elif self.mode != "timeout" and self.poll_count >= 2:
                    self.state = "READY"
            return 200, {}, [
                {
                    "eid": "capture-1", "sid": "sid-1" if self.state == "READY" else None,
                    "name": self.capture_name, "state": self.state,
                    "error_reason": "training failed" if self.state == "FAILED" else None,
                    "error_reason_slug": "training-failed" if self.state == "FAILED" else None,
                    "state_description": self.state.lower(),
                }
            ]
        if method == "POST" and path == "/api/v1/captures":
            self.create_count += 1
            self.state = "CREATED"
            self.capture_name = payload["name"]
            return 200, {}, {"eid": "capture-1", "num_parts": 1, "chunk_size": payload["bytesize"]}
        if method == "POST" and "create-upload-url" in path:
            return 200, {}, {"eid": "capture-1", "chunk_size": payload["bytesize"], "upload_url": "https://upload.example/part-1"}
        if method == "POST" and path.endswith("/uploaded"):
            self.complete_count += 1
            self.state = "PROCESSING"
            return 200, {}, {"state": "PROCESSING"}
        if method == "GET" and path.startswith("/api/v2/captures/"):
            if self.mode == "malformed_metadata":
                return 200, {}, {"sid": "sid-1", "content_profile": "ply"}
            return 200, {}, {
                "name": "fixture", "sid": "sid-1", "content_profile": "ply",
                "coord_system": "colmap", "model_url": "https://cdn.example/model.ply",
                "cameras_url": "https://cdn.example/cameras.json",
            }
        if method == "DELETE" and path == "/api/v1/captures/capture-1":
            self.delete_count += 1
            if self.mode == "deletion_failure":
                return 500, {}, {"detail": "failed"}
            self.deleted = True
            return 204, {}, None
        raise AssertionError((method, path, payload, query))

    def upload_part(self, url, *, source_path, offset, length):
        assert source_path.is_file() and offset == 0 and length == source_path.stat().st_size
        if self.mode == "failed_upload":
            return 400, {}
        etag = "broken" if self.mode == "corrupt_etag" else "0123456789abcdef0123456789abcdef"
        return 200, {"etag": etag}

    def download_bytes(self, url, *, maximum_bytes):
        assert len(self.camera_bytes) <= maximum_bytes
        return self.camera_bytes

    def download_file(self, url, *, destination, maximum_bytes):
        if self.mode == "oversized_ply":
            raise TeleportTransportError("teleport_download_size_out_of_bounds")
        data = self.ply.read_bytes()
        destination.write_bytes(data)
        return len(data)


def _grant():
    return require_paid_resource_admission(
        build_paid_lane_admission(resource_class=TELEPORT_RESOURCE_CLASS),
        resource_class=TELEPORT_RESOURCE_CLASS,
        expected_schema_version="paid_lane_admission.v1",
    )


def _runner(**_kwargs):
    return {
        "schema_version": "visual_heldout_evaluation_report.v2",
        "status": "passed_appearance_only",
        "candidate_had_hidden_access": False,
        "visual_heldout_evaluation_report_digest": "sha256:" + "9" * 64,
    }


def _run(tmp_path: Path, *, mode: str = "success", quote: float = 4.0, ttl: int = 60):
    packet, archive_digest = _packet(tmp_path)
    request = _request(archive_digest, quote=quote, ttl=ttl)
    observations = _observations()
    clock = Clock()
    transport = FakeTransport(tmp_path, observations, mode=mode)
    result = run_teleport_reconstruction(
        upload_packet_path=packet,
        execution_request=request,
        candidate_observations=observations,
        output_root=tmp_path / "output",
        paid_resource_admission_grant=_grant(),
        credentials=TeleportCredentials("fixture-id", "fixture-secret"),
        transport=transport,
        sealed_evaluation_runner=_runner,
        maximum_ply_bytes=10_000_000,
        poll_interval_seconds=5,
        monotonic=clock.monotonic,
        wall_time=clock.wall,
        sleep=clock.sleep,
    )
    return result, transport


def test_mocked_success_auth_refresh_and_exact_once_completion(tmp_path: Path) -> None:
    result, transport = _run(tmp_path / "success")
    assert result["status"] == "succeeded_unqualified"
    assert result["provider_zero_proven"] is False
    assert transport.create_count == 1
    assert transport.complete_count == 1
    assert transport.delete_count == 1
    native = tmp_path / "success/output/provider_native/reconstruction.ply"
    assert native.read_bytes() == transport.ply.read_bytes()
    schema_root = Path(__file__).parents[1] / "docs/schemas"
    for artifact_name, schema_name in (
        ("teleport_provider_progress.v1.json", "teleport_provider_progress.v1.schema.json"),
        ("teleport_provider_cost_receipt.v1.json", "teleport_provider_cost_receipt.v1.schema.json"),
        ("teleport_provider_run_receipt.v1.json", "teleport_provider_run_receipt.v1.schema.json"),
    ):
        jsonschema.validate(
            json.loads((tmp_path / "success/output" / artifact_name).read_text()),
            json.loads((schema_root / schema_name).read_text()),
        )

    refreshed, refresh_transport = _run(tmp_path / "auth", mode="auth_expiry")
    assert refreshed["status"] == "succeeded_unqualified"
    assert refresh_transport.auth_count == 2
    assert refresh_transport.complete_count == 1


def test_cross_origin_redirect_does_not_forward_bearer_token() -> None:
    requests: list[tuple[str, str, dict[str, str]]] = []

    class Response:
        def __init__(self, status: int, headers: list[tuple[str, str]], body: bytes) -> None:
            self.status = status
            self._headers = headers
            self._body = body

        def getheaders(self):
            return self._headers

        def read(self, _maximum: int):
            return self._body

    class Connection:
        def __init__(self, response: Response) -> None:
            self.response = response

        def request(self, method, target, *, body, headers):
            requests.append((method, target, dict(headers)))

        def getresponse(self):
            return self.response

        def close(self):
            return None

    responses = iter(
        [
            Response(307, [("Location", "https://cdn.example/metadata.json")], b""),
            Response(200, [], b'{"sid":"safe"}'),
        ]
    )
    transport = TeleportHttpTransport()
    transport._connection = lambda url: (Connection(next(responses)), urlsplit(url).path)  # type: ignore[method-assign]
    status, _headers, payload = transport._small_request(
        "GET",
        "https://teleport.varjo.com/api/v2/captures/safe/metadata",
        headers={"Authorization": "Bearer secret-token", "Accept": "application/json"},
    )
    assert status == 200
    assert payload == b'{"sid":"safe"}'
    assert requests[0][2]["Authorization"] == "Bearer secret-token"
    assert "Authorization" not in requests[1][2]


def test_canonical_allocator_dry_run_never_reads_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    packet_path, archive_digest = _packet(tmp_path / "packet")
    terms_path = (
        Path(__file__).parents[1]
        / "docs/evidence/teleport_provider_terms_review_2026-08-03.json"
    )
    terms = json.loads(terms_path.read_text())
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            _request(
                archive_digest,
                terms_digest=terms["teleport_provider_terms_review_digest"],
            )
        )
    )
    observations_path = tmp_path / "observations.json"
    observations_path.write_text(json.dumps({"observations": _observations()}))
    output = tmp_path / "preflight"
    monkeypatch.setattr(allocator, "_source_checkout_blockers", lambda *_args, **_kwargs: ([], "1" * 40))
    monkeypatch.setattr(
        allocator,
        "load_teleport_credentials",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("credentials read")),
    )
    exit_code = allocator.main(
        [
            "provider-reconstruction",
            "--upload-packet",
            str(packet_path),
            "--execution-request",
            str(request_path),
            "--candidate-observations",
            str(observations_path),
            "--terms-review",
            str(terms_path),
            "--output-dir",
            str(output),
            "--observed-at",
            "2026-08-03T12:00:02Z",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"success": True}
    preflight = json.loads((output / "teleport_provider_preflight.v1.json").read_text())
    assert preflight["status"] == "ready"
    assert preflight["execute_requested"] is False
    assert preflight["provider_mutations_performed"] == 0
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/teleport_provider_preflight.v1.schema.json"
        ).read_text()
    )
    jsonschema.validate(preflight, schema)
    monkeypatch.delenv("TELEPORT_PUBLIC_DATA_UPLOAD_AUTHORIZED", raising=False)
    monkeypatch.delenv("TELEPORT_PUBLIC_DATA_SPEND_CAP_USD", raising=False)
    blocked_output = tmp_path / "blocked-execute"
    exit_code = allocator.main(
        [
            "provider-reconstruction",
            "--upload-packet",
            str(packet_path),
            "--execution-request",
            str(request_path),
            "--candidate-observations",
            str(observations_path),
            "--terms-review",
            str(terms_path),
            "--output-dir",
            str(blocked_output),
            "--observed-at",
            "2026-08-03T12:00:02Z",
            "--execute",
        ]
    )
    assert exit_code == 2
    assert json.loads(capsys.readouterr().out) == {"success": False}
    blocked = json.loads(
        (blocked_output / "teleport_provider_preflight.v1.json").read_text()
    )
    assert {
        "teleport_public_data_upload_interlock_missing",
        "teleport_public_data_spend_cap_interlock_mismatch",
        "teleport_sealed_evaluation_request_missing",
    } <= set(blocked["blockers"])
    jsonschema.validate(blocked, schema)


def test_committed_teleport_evidence_validates_against_schemas() -> None:
    root = Path(__file__).parents[1]
    for artifact_name, schema_name in (
        (
            "teleport_provider_terms_review_2026-08-03.json",
            "teleport_provider_terms_review.v1.schema.json",
        ),
        (
            "teleport_future_customer_use_constraints_2026-08-03.json",
            "teleport_future_customer_use_constraints.v1.schema.json",
        ),
    ):
        artifact = json.loads((root / "docs/evidence" / artifact_name).read_text())
        schema = json.loads((root / "docs/schemas" / schema_name).read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(artifact, schema)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("failed_upload", "teleport_part_upload_http_status_invalid"),
        ("corrupt_etag", "teleport_upload_etag_invalid"),
        ("timeout", "teleport_poll_timeout"),
        ("failed_training", "teleport_training_failed"),
        ("malformed_metadata", "teleport_v2_metadata"),
        ("oversized_ply", "teleport_download_size_out_of_bounds"),
        ("corrupt_ply", "provider_splat_ply_not_standard_3dgs"),
        ("residual_failure", "provider_alignment_residual_threshold_exceeded"),
        ("deletion_failure", "teleport_deletion_failed"),
    ],
)
def test_mocked_lifecycle_failures_delete_and_emit_typed_receipts(
    tmp_path: Path, mode: str, expected: str
) -> None:
    with pytest.raises(TeleportAdapterError) as caught:
        _run(tmp_path / mode, mode=mode, ttl=10)
    assert any(expected in code for code in caught.value.codes)
    output = tmp_path / mode / "output"
    assert (output / "reconstruction_provider_execution_receipt.v1.json").is_file()
    assert (output / "reconstruction_provider_deletion_receipt.v1.json").is_file()
    if mode == "deletion_failure":
        deletion = json.loads(
            (output / "reconstruction_provider_deletion_receipt.v1.json").read_text()
        )
        assert deletion["status"] == "failed"
        assert deletion["provider_zero_proven"] is False


def test_spend_ceiling_blocks_before_any_provider_call(tmp_path: Path) -> None:
    packet, archive_digest = _packet(tmp_path)
    request = _request(archive_digest, quote=9.0)
    transport = FakeTransport(tmp_path, _observations())
    with pytest.raises(TeleportAdapterError, match="spend_ceiling"):
        run_teleport_reconstruction(
            upload_packet_path=packet,
            execution_request=request,
            candidate_observations=_observations(),
            output_root=tmp_path / "output",
            paid_resource_admission_grant=_grant(),
            credentials=TeleportCredentials("fixture-id", "fixture-secret"),
            transport=transport,
            sealed_evaluation_runner=_runner,
            wall_time=Clock.wall,
        )
    assert transport.auth_count == 0
    assert transport.create_count == 0


def test_customer_or_confidential_packet_is_rejected_before_provider_call(
    tmp_path: Path,
) -> None:
    packet_path, _archive_digest = _packet(tmp_path)
    packet = json.loads(packet_path.read_text())
    packet["dataset_class"] = "customer_capture"
    packet["customer_or_confidential_data_included"] = True
    packet["teleport_ready_to_upload_packet_digest"] = canonical_digest(
        packet, digest_field="teleport_ready_to_upload_packet_digest"
    )
    packet_path.write_text(json.dumps(packet))
    with pytest.raises(TeleportAdapterError) as caught:
        validate_teleport_upload_packet(packet, packet_root=packet_path.parent)
    assert {
        "teleport_upload_packet_not_rights_cleared_public_dataset",
        "teleport_upload_packet_customer_or_confidential_data_not_false",
    } <= set(caught.value.codes)


def test_exact_part_numbers_etags_and_file_backed_secret_permissions(tmp_path: Path) -> None:
    assert validate_uploaded_parts(
        [{"number": 1, "etag": '"0123456789abcdef0123456789abcdef"'}],
        expected_count=1,
    ) == [{"number": 1, "etag": "0123456789abcdef0123456789abcdef"}]
    with pytest.raises(TeleportAdapterError, match="numbers_not_exact"):
        validate_uploaded_parts([{"number": 2, "etag": "0" * 32}], expected_count=1)

    client_id = tmp_path / "client-id"
    client_secret = tmp_path / "client-secret"
    client_id.write_text("unique-client-value-123")
    client_secret.write_text("unique-secret-value-456")
    os.chmod(client_id, 0o600)
    os.chmod(client_secret, 0o600)
    credentials = load_teleport_credentials(
        {}, client_id_file=client_id, client_secret_file=client_secret
    )
    assert "unique-client-value-123" not in repr(credentials)
    assert "unique-secret-value-456" not in repr(credentials)
    os.chmod(client_secret, 0o644)
    with pytest.raises(TeleportAdapterError, match="client_secret_file_invalid"):
        load_teleport_credentials({}, client_id_file=client_id, client_secret_file=client_secret)
