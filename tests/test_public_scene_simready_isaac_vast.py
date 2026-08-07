from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

import blueprint_pipeline.paid_resource_allocator as allocator
import blueprint_pipeline.public_scene_simready_isaac_vast as runtime
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_simready_isaac_bundle import DEFAULT_IMAGE


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path: Path, *, commit: str = "a" * 40) -> dict:
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("provider_runtime/example.txt", "bound")
    return {
        "status": "ready",
        "source_commit_sha": commit,
        "container_image": DEFAULT_IMAGE,
        "retry_cap": 0,
        "blockers": [],
        "probe_spec_sha256": "sha256:" + "b" * 64,
        "bundle_path": str(bundle),
        "bundle_sha256": _sha256(bundle),
    }


def _completed_execution() -> dict:
    value = {
        "schema_version": "adp009b_simready_isaac_result.v1",
        "status": "completed",
        "blockers": [],
        "native_isaac_executed": True,
        "physical_success_established": False,
        "source_target_collider_active": False,
        "replacement_count": 1,
        "probe_results": [
            {"probe": name, "passed": True}
            for name in ("drop", "slide", "tip", "gripper")
        ],
    }
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def test_dry_run_never_stages_or_mutates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime,
        "stage_wam_provider_bundle_object_store",
        lambda **kwargs: pytest.fail("dry run staged provider bytes"),
    )

    result = runtime.run_simready_isaac_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=_bundle(tmp_path),
        paid_resource_admission_grant=None,
        execute=False,
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0


def test_live_run_requires_all_four_native_probes_and_provider_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_stage(**kwargs):
        staging = Path(kwargs["job_dir"])
        staging.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text("https://example.invalid/bound", encoding="utf-8")
        return {"status": "completed", "blockers": []}

    def fake_adapter(**kwargs):
        output = Path(kwargs["provider_runtime_output_zip"])
        output.parent.mkdir(parents=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(
                "isaac_runtime_result.json",
                json.dumps(_completed_execution(), sort_keys=True),
            )
        write_json(
            Path(kwargs["job_dir"]) / "vast_teardown_manifest.json",
            {"continuing_spend_from_this_run": False},
        )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.12}

    monkeypatch.setattr(runtime, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(runtime, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        runtime,
        "cleanup_staged_wam_provider_objects",
        lambda path: {"all_objects_absent": True},
    )

    result = runtime.run_simready_isaac_vast(
        job_dir=tmp_path / "job",
        prepared_bundle=_bundle(tmp_path),
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
    )

    assert result["status"] == "completed"
    assert result["retry_cap"] == 0
    assert result["continuing_spend_from_this_run"] is False
    assert result["all_staged_objects_absent"] is True


def _allocator_args(tmp_path: Path, receipt: Path) -> list[str]:
    return [
        "gpu-canary",
        "--probe-kind",
        runtime.PROBE_KIND,
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
        "adp009b-simready",
        "--expected-source-commit",
        "a" * 40,
        "--adp-simready-isaac-bundle-receipt",
        str(receipt),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "1.0",
        "--adp-max-spend-usd",
        "3.0",
        "--adp-hard-ttl-seconds",
        "10800",
    ]


def test_canonical_allocator_binds_exact_bundle_and_withholds_dry_run_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "bundle_receipt.json"
    bundle_receipt = _bundle(tmp_path)
    write_json(receipt, bundle_receipt)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_simready_isaac_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, receipt)) == 0
    assert observed["execute"] is False
    assert observed["paid_resource_admission_grant"] is None
    admission = json.loads((tmp_path / "admission.json").read_text(encoding="utf-8"))
    assert admission["status"] == "admitted"
    assert admission["retry_cap"] == 0
    assert admission["allocation_binding"]["bundle_sha256"] == bundle_receipt[
        "bundle_sha256"
    ]
