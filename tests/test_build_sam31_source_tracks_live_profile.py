from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline.sam31_gpu_admission import REQUEST_SCHEMA_VERSION
from blueprint_pipeline.sam31_paid_attempt_authority import (
    materialize_sam31_paid_attempt_authority,
)
from blueprint_pipeline.sam31_source_track_canary_worker import BUNDLE_RECEIPT_SCHEMA_VERSION
from blueprint_pipeline.task_evaluation_launch_dispatcher import TaskEvaluationLaunchError
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


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

    bundle = tmp_path / "sam31-input.zip"
    bundle.write_bytes(b"sam31-bundle")
    bundle_digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "source_commit_sha": COMMIT,
        "worker_image_digest": "registry.example/sam31@sha256:" + "b" * 64,
        "input_bundle_digest": bundle_digest,
        "input_bundle_size_bytes": bundle.stat().st_size,
        "max_spend_usd": 1.0,
        "hard_ttl_seconds": 600,
        "retry_cap": 0,
        "authority_id": "fresh-scene-sam31-1",
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


def test_profile_binds_all_nonsecret_immutable_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = {row["name"]: row for row in _build(paths)["immutable_inputs"]}
    assert set(inputs) == {
        "source_bundle_manifest",
        "evaluation_run_spec",
        "sam31_input_bundle",
        "sam31_paid_attempt_authority",
    }
    assert all("token" not in row["name"] for row in inputs.values())


def test_profile_refuses_nonprivate_hf_token(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["token"].chmod(0o644)
    with pytest.raises(TaskEvaluationLaunchError, match="permissions_not_0600"):
        _build(paths)


def test_profile_refuses_bundle_tamper(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["bundle"].write_bytes(b"tampered")
    with pytest.raises(TaskEvaluationLaunchError, match="sam31_paid_authority_invalid"):
        _build(paths)
