from __future__ import annotations

import json
import tarfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import blueprint_pipeline.post_training_data_package as package_module
from blueprint_pipeline.capture_orchestrator import PipelineConfig, run_capture_pipeline
from blueprint_pipeline.post_training_data_package import (
    build_post_training_data_package_export,
)
from tests.test_post_training_data_package import (
    _seed_ready_job,
    _seed_valid_canonical_quality_chain,
    _write_valid_mp4_or_placeholder,
)
from tests.test_site_world_packaging import (
    _HealthyRuntimeClient,
    _build_staged_capture,
    _successful_capture_review,
    _successful_privacy_processing,
    _use_offline_webapp_sync,
    _write_backend_script,
)


def _write_signing_key(path: Path) -> None:
    path.write_bytes(
        Ed25519PrivateKey.generate().private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    path.chmod(0o600)


def _write_test_native_export(path: Path, label: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(label, encoding="utf-8")
    return True


def test_capture_bundle_to_cards_package_archive_and_webapp_projection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """One CPU-only regression crosses the complete sellable product spine."""

    capture_root, descriptor_uri = _build_staged_capture(tmp_path)
    success_backend = tmp_path / "success_backend.py"
    sam3_backend = tmp_path / "sam3_backend.py"
    _write_backend_script(success_backend, mode="success")
    _write_backend_script(sam3_backend, mode="sam3-skip")

    _use_offline_webapp_sync(monkeypatch)
    monkeypatch.setenv(
        "OBJECT_INDEX_YOLO_WORLD_COMMAND",
        f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}",
    )
    monkeypatch.setenv(
        "OBJECT_INDEX_GROUNDING_DINO_COMMAND",
        f"python3 {success_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}",
    )
    monkeypatch.setenv(
        "OBJECT_INDEX_SAM3_COMMAND",
        f"python3 {sam3_backend} {{INPUT_JSON}} {{OUTPUT_JSON}}",
    )
    monkeypatch.setenv("SITE_WORLD_RUNTIME_SERVICE_URL", "http://runtime.test")
    monkeypatch.setattr(
        "blueprint_pipeline.evaluation_prep_stage.SiteWorldRuntimeServiceClient",
        _HealthyRuntimeClient,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.site_package_orchestrator.infer_capture_fidelity_review",
        lambda **_kwargs: _successful_capture_review(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.site_package_orchestrator.run_privacy_postprocess",
        lambda **_kwargs: _successful_privacy_processing(),
    )

    result = run_capture_pipeline(
        descriptor_gcs_uri=descriptor_uri,
        lane="current",
        config=PipelineConfig(gcs_root=tmp_path),
    )

    dataset_root = capture_root / "pipeline" / "robot_eval_dataset"
    for name in ("site_card.json", "task_cards.json", "scenario_cards.json", "eval_cards.json"):
        assert (dataset_root / name).is_file(), name
    webapp_projection = capture_root / "pipeline" / "webapp_sync_result.json"
    assert webapp_projection.is_file()
    webapp_payload = json.loads(webapp_projection.read_text(encoding="utf-8"))
    assert webapp_payload["latest_stage"] == "evaluation_prep"
    evaluation_result = next(
        row for row in result["results"] if row.get("lane") == "evaluation_prep"
    )
    assert evaluation_result["manifest_path"].endswith("evaluation_prep_manifest.json")

    signing_key = tmp_path / "ptdp-signing-key.pem"
    _write_signing_key(signing_key)
    monkeypatch.setenv(package_module.PTDP_SIGNING_KEY_FILE_ENV, str(signing_key))
    monkeypatch.setenv(package_module.PTDP_SIGNING_KEY_ID_ENV, "product-spine-e2e")
    # This tiny fixture needs only the measured archive workspace, not the 1 GiB
    # production reserve. Dedicated resource-preflight tests bind the production
    # default; keeping that host-capacity concern here would make the product-spine
    # regression nondeterministic on otherwise healthy, space-constrained CI runners.
    monkeypatch.setenv(package_module.PTDP_MIN_FREE_HEADROOM_BYTES_ENV, str(1024**2))

    job_dir = tmp_path / "robot-eval-job"
    _seed_ready_job(job_dir)
    clips_manifest_path = job_dir / "clips_manifest.json"
    clips_manifest = json.loads(clips_manifest_path.read_text(encoding="utf-8"))
    clips_manifest["clips"][0].update(
        {
            "consent_revoked": False,
            "delivery_blocked_by_consent_revocation": False,
            "signed_access_revoked_by_consent": False,
            "manual_rights_review_recommended": False,
            "commercial_use_claim_allowed": True,
            "external_licensing_claim_allowed": True,
        }
    )
    clips_manifest_path.write_text(
        json.dumps(clips_manifest, indent=2), encoding="utf-8"
    )
    _write_valid_mp4_or_placeholder(job_dir / "clip-1.mp4")
    _seed_valid_canonical_quality_chain(job_dir)

    monkeypatch.setattr(
        package_module,
        "_write_native_hdf5",
        lambda path, _rows: _write_test_native_export(path, "e2e-hdf5"),
    )
    monkeypatch.setattr(
        package_module,
        "_write_native_parquet",
        lambda path, _rows: _write_test_native_export(path, "e2e-parquet"),
    )
    package_root = capture_root / "pipeline" / "post_training_data_package"
    package = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=package_root,
    )

    assert package["schema_version"] == "post_training_data_package_export.v1"
    archive_manifest = json.loads(
        (package_root / "archive_manifest.json").read_text(encoding="utf-8")
    )
    archive_path = package_root / archive_manifest["archive"]["path"]
    assert archive_manifest["status"] == "created_and_verified"
    assert archive_path.is_file()
    with tarfile.open(archive_path, "r:gz") as archive:
        assert "post_training_data_package_export_manifest.json" in {
            member.name for member in archive.getmembers()
        }
