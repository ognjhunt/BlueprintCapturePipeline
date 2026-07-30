from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.policy_ranking_evaluator_diagnostic import (
    complete_graph_diagnostic_protocol,
)
from blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_transport_canary import (
    GeminiTransportCanaryError,
    _ledger_age_seconds,
    build_transport_canary_paid_admission,
    stage_transport_canary_media,
    submit_transport_canary,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


SOURCE_COMMIT = "2" * 40


def _inventory() -> dict:
    pairs = [
        {
            "pair_id": "pair-0",
            "task_instruction": "press the button",
            "episode_a": {"source_request_id": "request-0"},
            "episode_b": {"source_request_id": "request-1"},
        }
    ] + [{"pair_id": f"pair-{index}"} for index in range(1, 882)]
    result = {
        "status": "ready",
        "pair_count": 882,
        "protocol_sha256": complete_graph_diagnostic_protocol()["protocol_sha256"],
        "pairs": pairs,
        "outcome_labels_accessed_to_build_pairs": False,
    }
    result["inventory_sha256"] = canonical_sha256(result)
    return result


def _manifest() -> dict:
    result = {
        "status": "passed",
        "video_count": 441,
        "all_physical_right_half_pixels_excluded": True,
        "receipts": [
            {
                "request_id": f"request-{index}",
                "output_path": f"/tmp/request-{index}.mp4",
                "output_sha256": f"{index:064x}"[-64:],
                "output_size_bytes": index + 1,
            }
            for index in range(441)
        ],
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def _admission() -> dict:
    return build_transport_canary_paid_admission(
        _inventory(),
        _manifest(),
        inventory_file_sha256="a" * 64,
        manifest_file_sha256="b" * 64,
        source_commit=SOURCE_COMMIT,
        realized_api_spend_usd=8.418512,
        realized_missing_graph_spend_usd=0.0,
        projected_canary_cost_usd=0.01,
        canary_hard_cap_usd=0.05,
        missing_graph_hard_cap_usd=9.0,
        campaign_api_hard_cap_usd=25.0,
        credential_ready=True,
    )


def _key(tmp_path: Path) -> Path:
    path = tmp_path / "gemini-key"
    path.write_text("test-key")
    path.chmod(0o600)
    return path


def test_transport_canary_admission_binds_exact_first_pair_and_budget() -> None:
    admission = _admission()

    assert admission["status"] == "admitted"
    assert admission["pair_index"] == 0
    assert admission["pair_id"] == "pair-0"
    assert admission["request_ids"] == ["request-0", "request-1"]
    assert admission["request_count"] == 1
    assert admission["video_count"] == 2
    assert admission["shared_paid_lane_admission"]["resource_class"] == "evaluator_api"
    assert admission["execution_contract"]["ranking_or_confirmation_credit"] is False
    payload = {key: value for key, value in admission.items() if key != "admission_sha256"}
    assert canonical_sha256(payload) == admission["admission_sha256"]


def test_transport_canary_admission_fails_closed_above_canary_cap() -> None:
    admission = build_transport_canary_paid_admission(
        _inventory(),
        _manifest(),
        inventory_file_sha256="a" * 64,
        manifest_file_sha256="b" * 64,
        source_commit=SOURCE_COMMIT,
        realized_api_spend_usd=8.418512,
        realized_missing_graph_spend_usd=0.0,
        projected_canary_cost_usd=0.051,
        canary_hard_cap_usd=0.05,
        missing_graph_hard_cap_usd=9.0,
        campaign_api_hard_cap_usd=25.0,
        credential_ready=True,
    )

    assert admission["status"] == "blocked"
    assert "projected_canary_cost_exceeds_cap" in admission["blockers"]


def test_transport_canary_admission_fails_closed_above_remaining_graph_cap() -> None:
    admission = build_transport_canary_paid_admission(
        _inventory(),
        _manifest(),
        inventory_file_sha256="a" * 64,
        manifest_file_sha256="b" * 64,
        source_commit=SOURCE_COMMIT,
        realized_api_spend_usd=8.418512,
        realized_missing_graph_spend_usd=8.995,
        projected_canary_cost_usd=0.01,
        canary_hard_cap_usd=0.05,
        missing_graph_hard_cap_usd=9.0,
        campaign_api_hard_cap_usd=25.0,
        credential_ready=True,
    )

    assert admission["status"] == "blocked"
    assert "projected_missing_graph_cost_exceeds_cap" in admission["blockers"]


def test_transport_canary_stages_exact_two_bound_videos(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    uploaded_ids: list[str] = []

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"

    def fake_upload(_client, receipt):
        request_id = receipt["request_id"]
        uploaded_ids.append(request_id)
        uploaded = types.SimpleNamespace(name=f"files/{request_id}")
        return uploaded, {
            "request_id": request_id,
            "provider_file_name": uploaded.name,
            "provider_file_uri": f"https://example.invalid/{request_id}",
            "provider_mime_type": "video/mp4",
            "provider_file_state": "FileState.ACTIVE",
        }

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setattr(
        "blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_transport_canary._upload_video",
        fake_upload,
    )
    ledger = stage_transport_canary_media(
        _inventory(),
        _manifest(),
        api_key_file=_key(tmp_path),
        ledger_path=tmp_path / "ledger.json",
        source_commit=SOURCE_COMMIT,
        paid_admission=_admission(),
    )

    assert uploaded_ids == ["request-0", "request-1"]
    assert ledger["uploaded_video_count"] == 2
    assert ledger["physical_ground_truth_pixels_uploaded"] is False
    assert Path(tmp_path / "ledger.json").is_file()


def test_transport_canary_invalid_upload_receipt_forces_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    deleted: list[str] = []

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.files = types.SimpleNamespace(delete=lambda *, name: deleted.append(name))

    def fake_upload(_client, receipt):
        request_id = receipt["request_id"]
        uploaded = types.SimpleNamespace(name=f"files/{request_id}")
        return uploaded, {
            "request_id": request_id,
            "provider_file_name": uploaded.name,
            "provider_file_uri": "",
            "provider_mime_type": "video/mp4",
        }

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setattr(
        "blueprint_pipeline.policy_ranking_evaluator_diagnostic_gemini_transport_canary._upload_video",
        fake_upload,
    )

    with pytest.raises(
        GeminiTransportCanaryError, match="transport_canary_upload_receipts_invalid"
    ):
        stage_transport_canary_media(
            _inventory(),
            _manifest(),
            api_key_file=_key(tmp_path),
            ledger_path=tmp_path / "ledger.json",
            source_commit=SOURCE_COMMIT,
            paid_admission=_admission(),
        )

    assert deleted == ["files/request-0", "files/request-1"]
    assert not (tmp_path / "ledger.json").exists()


def test_transport_canary_submit_refuses_before_90_second_grace(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    inventory = _inventory()
    manifest = _manifest()
    admission = _admission()
    ledger = {
        "schema_version": "policy_ranking_gemini_transport_canary_media_ledger.v1",
        "status": "ready",
        "arm_id": "gemini36_flash_complete_graph",
        "pair_id": "pair-0",
        "inventory_sha256": inventory["inventory_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "paid_admission_sha256": admission["admission_sha256"],
        "uploaded_video_count": 2,
        "uploads": [
            {
                "request_id": f"request-{index}",
                "provider_file_name": f"files/request-{index}",
                "provider_file_uri": f"https://example.invalid/request-{index}",
                "provider_mime_type": "video/mp4",
            }
            for index in range(2)
        ],
        "source_commit": SOURCE_COMMIT,
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_uploaded": False,
        "credential_path_or_value_persisted": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)

    with pytest.raises(
        GeminiTransportCanaryError,
        match="transport_canary_media_propagation_grace_not_elapsed",
    ):
        submit_transport_canary(
            inventory,
            manifest,
            ledger,
            api_key_file=tmp_path / "unused",
            receipt_path=tmp_path / "receipt.json",
            source_commit=SOURCE_COMMIT,
            paid_admission=admission,
        )


def test_transport_canary_ledger_age_uses_frozen_utc_time() -> None:
    ready = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)
    assert (
        _ledger_age_seconds(
            {"updated_at_utc": ready.isoformat().replace("+00:00", "Z")},
            now=ready + timedelta(seconds=91),
        )
        == 91
    )


def test_transport_canary_provider_failure_is_preserved_and_media_deleted(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_ROBOARENA_DIAGNOSTIC_GEMINI", "1")
    inventory = _inventory()
    manifest = _manifest()
    admission = _admission()
    deleted: list[str] = []

    class ExampleClientError(Exception):
        code = 400
        status = "FAILED_PRECONDITION"
        message = "Precondition check failed."
        details = {"error": {"status": "FAILED_PRECONDITION"}}

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.batches = types.SimpleNamespace(
                create=lambda **_kwargs: (_ for _ in ()).throw(ExampleClientError())
            )
            self.files = types.SimpleNamespace(delete=lambda *, name: deleted.append(name))

    class FakeFile:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    fake_types = types.ModuleType("google.genai.types")
    fake_types.File = FakeFile
    fake_types.InlinedRequest = lambda **kwargs: kwargs
    fake_types.GenerateContentConfig = lambda **kwargs: kwargs
    fake_types.ThinkingConfig = lambda **kwargs: kwargs
    fake_types.CreateBatchJobConfig = lambda **kwargs: kwargs
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient
    fake_genai.types = fake_types
    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    uploads = [
        {
            "request_id": f"request-{index}",
            "provider_file_name": f"files/request-{index}",
            "provider_file_uri": f"https://example.invalid/request-{index}",
            "provider_mime_type": "video/mp4",
        }
        for index in range(2)
    ]
    ledger = {
        "schema_version": "policy_ranking_gemini_transport_canary_media_ledger.v1",
        "status": "ready",
        "arm_id": "gemini36_flash_complete_graph",
        "pair_id": "pair-0",
        "inventory_sha256": inventory["inventory_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "paid_admission_sha256": admission["admission_sha256"],
        "uploaded_video_count": 2,
        "uploads": uploads,
        "source_commit": SOURCE_COMMIT,
        "policy_identity_sent_to_provider": False,
        "physical_outcome_sent_to_provider": False,
        "physical_ground_truth_pixels_uploaded": False,
        "credential_path_or_value_persisted": False,
        "updated_at_utc": "2026-07-30T07:00:00Z",
    }
    ledger["ledger_sha256"] = canonical_sha256(ledger)
    result = submit_transport_canary(
        inventory,
        manifest,
        ledger,
        api_key_file=_key(tmp_path),
        receipt_path=tmp_path / "receipt.json",
        source_commit=SOURCE_COMMIT,
        paid_admission=admission,
    )

    assert result["status"] == "failed_before_batch_creation"
    assert result["provider_error"]["provider_status"] == "FAILED_PRECONDITION"
    assert result["provider_generation_rows_created"] == 0
    assert result["all_task_media_deleted"] is True
    assert deleted == ["files/request-0", "files/request-1"]
