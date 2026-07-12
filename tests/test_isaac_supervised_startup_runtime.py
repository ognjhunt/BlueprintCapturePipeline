from __future__ import annotations

import hashlib
import io
import json
import zipfile

from blueprint_pipeline import isaac_supervised_startup_runtime as R


DIGEST = "sha256:" + "a" * 64


def test_manifest_checksum_rejects_file_replaced_after_validation(tmp_path) -> None:
    path = tmp_path / "diagnostic.json"
    original = b'{"status":"completed","version":1}'
    path.write_bytes(original)
    diagnostic = {
        "path": str(path),
        "metadata_available_for_selected_image": True,
        "diagnostic_sha256": hashlib.sha256(original).hexdigest(),
        "diagnostic_bytes": len(original),
    }
    path.write_bytes(b'{"status":"completed","version":2}')

    result = R.resolve_image_manifest_checksum(diagnostic)

    assert result["status"] == "blocked"
    assert "startup_image_manifest_diagnostic_identity_changed" in result["blockers"]


def test_manifest_checksum_accepts_exact_validated_bytes(tmp_path) -> None:
    path = tmp_path / "diagnostic.json"
    raw = b'{"status":"completed"}'
    path.write_bytes(raw)

    result = R.resolve_image_manifest_checksum(
        {
            "path": str(path),
            "metadata_available_for_selected_image": True,
            "diagnostic_sha256": hashlib.sha256(raw).hexdigest(),
            "diagnostic_bytes": len(raw),
        }
    )

    assert result["status"] == "passed"
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()


def _archive(*, nonce: str = "nonce-1", review_status: str = "passed") -> bytes:
    values = {
        "bootstrap.json": {"phase": "startup_gates_passed", "launch_session_id": nonce},
        R.STARTUP_GATE_SUMMARY_FILENAME: {
            "status": "passed",
            "blockers": [],
            "launch_session_id": nonce,
            "image_digest": DIGEST,
        },
        R.FAST_PREFLIGHT_FILENAME: {
            "status": "passed",
            "blockers": [],
            "launch_session_id": nonce,
            "image_digest": DIGEST,
        },
        R.KITCHEN_GATE_FILENAME: {
            "status": "completed",
            "blockers": [],
            "launch_session_id": nonce,
            "image_digest": DIGEST,
        },
        R.REVIEW_CANARY_FILENAME: {
            "status": review_status,
            "blockers": [] if review_status == "passed" else ["blank_frame"],
            "launch_session_id": nonce,
            "image_digest": DIGEST,
            "isaac_review_renderer_operational": review_status == "passed",
        },
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, value in values.items():
            archive.writestr(name, json.dumps(value))
    return out.getvalue()


def test_archive_requires_all_three_attempt_bound_worker_gates() -> None:
    passed = R.validate_startup_gate_archive(
        _archive(),
        expected_launch_session_id="nonce-1",
        expected_image_digest=DIGEST,
    )
    assert passed["status"] == "passed"
    assert passed["checks"]["kitchen_asset_startup_gate"]["status"] == "completed"

    blocked = R.validate_startup_gate_archive(
        _archive(review_status="blocked"),
        expected_launch_session_id="nonce-1",
        expected_image_digest=DIGEST,
    )
    assert blocked["status"] == "blocked"
    assert "review_renderer_canary_not_passed" in blocked["blockers"]


def test_archive_rejects_stale_attempt_and_wrong_digest() -> None:
    stale = R.validate_startup_gate_archive(
        _archive(nonce="old"),
        expected_launch_session_id="new",
        expected_image_digest=DIGEST,
    )
    assert stale == {"status": "waiting", "reason": "stale_launch_session_archive"}

    wrong_digest = R.validate_startup_gate_archive(
        _archive(),
        expected_launch_session_id="nonce-1",
        expected_image_digest="sha256:" + "b" * 64,
    )
    assert wrong_digest["status"] == "blocked"
    assert "startup_gate_image_digest_mismatch" in wrong_digest["blockers"]


def test_waiter_moves_from_inflight_archive_to_pass_without_sleeping() -> None:
    values = iter([b"not-a-zip", _archive()])
    ticks = iter([0.0, 0.0, 1.0, 1.0, 2.0])
    result = R.wait_for_startup_gates(
        fetch_archive=lambda: next(values),
        expected_launch_session_id="nonce-1",
        expected_image_digest=DIGEST,
        timeout_seconds=10,
        poll_seconds=0.01,
        clock=lambda: next(ticks),
        sleeper=lambda _: None,
    )
    assert result["status"] == "passed"


class _Provider:
    name = "runpod"

    def __init__(self, *, inventory_confirmed: bool = True) -> None:
        self.inventory_confirmed = inventory_confirmed
        self.launch_request = None
        self.capacity_available = True

    def launch(self, job_dir, request, **kwargs):
        self.launch_request = request
        return {"status": "launched", "instance_id": "pod-1"}

    def capacity_preflight(self, request):
        return {
            "status": "available" if self.capacity_available else "blocked",
            "blockers": [] if self.capacity_available else ["no_capacity"],
            "reservation_proven": False,
            "capacity_confidence": "advisory",
            "request_gpu_types": request.get("gpuTypeIds"),
        }

    def inspect(self, instance_id):
        return {"instance_id": instance_id}

    def terminate(self, instance_id):
        return {"status": "terminated", "instance_id": instance_id}

    def billable_inventory(self, *, name_prefix):
        return {
            "api_confirmed": self.inventory_confirmed,
            "live_resource_count": 0 if self.inventory_confirmed else None,
            "resources": [],
            "name_prefix": name_prefix,
        }


def test_provider_adapter_binds_nonce_name_gpu_and_fails_closed_inventory(tmp_path) -> None:
    provider = _Provider()
    adapter = R.SupervisedProviderAdapter(
        provider=provider,
        job_dir=tmp_path,
        base_request={"name": "old", "gpuTypeIds": ["old"], "env": {}},
        resource_name_prefix="blueprint-supervised",
    )
    launched = adapter.allocate(
        gpu_type="NVIDIA A40", attempt_id="run/attempt 1", launch_nonce="nonce-1"
    )
    assert launched["instance_id"] == "pod-1"
    assert provider.launch_request["name"].startswith("blueprint-supervised-run-attempt-1")
    assert provider.launch_request["gpuTypeIds"] == ["NVIDIA A40"]
    assert provider.launch_request["env"]["BLUEPRINT_LAUNCH_SESSION_ID"] == "nonce-1"
    assert (tmp_path / "launch_session_nonce.txt").read_text() == "nonce-1"
    assert adapter.inventory()["live_resource_count"] == 0

    unavailable = R.SupervisedProviderAdapter(
        provider=_Provider(inventory_confirmed=False),
        job_dir=tmp_path,
        base_request={},
        resource_name_prefix="blueprint-supervised",
    ).inventory()
    assert unavailable["live_resource_count"] == 1
    assert "provider_billable_inventory_not_api_confirmed" in unavailable["blockers"]


def test_provider_adapter_treats_catalog_probe_as_advisory_before_create(tmp_path) -> None:
    provider = _Provider()
    provider.capacity_available = False
    adapter = R.SupervisedProviderAdapter(
        provider=provider,
        job_dir=tmp_path,
        base_request={"name": "old", "gpuTypeIds": ["old"], "env": {}},
        resource_name_prefix="blueprint-supervised",
    )
    result = adapter.allocate(
        gpu_type="NVIDIA A40", attempt_id="attempt-1", launch_nonce="nonce"
    )
    assert result["status"] == "launched"
    assert result["instance_id"] == "pod-1"
    assert result["capacity_preflight"]["status"] == "blocked"
    assert result["catalog_capacity_was_advisory"] is True
    assert result["authoritative_capacity_source"] == "provider_create_response"
    assert provider.launch_request is not None
