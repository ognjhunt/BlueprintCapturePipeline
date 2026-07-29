from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

from PIL import Image

from blueprint_pipeline.policy_ranking_droid_reference_bundle import (
    EXPERIMENT_ID,
    RECEIPT_SCHEMA,
    build_droid_reference_bundle,
)
from blueprint_pipeline.policy_ranking_successor_cosmos import canonical_sha256
from blueprint_pipeline.policy_ranking_successor_cosmos_bundle import _sha256_file
from blueprint_pipeline import policy_ranking_successor_gpu_admission as admission


def _reference_fixture(root: Path) -> Path:
    root.mkdir()
    observation = root / "initial_observation.png"
    Image.new("RGB", (640, 540), color=(12, 34, 56)).save(observation)
    actions = {
        "recorded": {"actions": [[0.0] * 10 for _ in range(16)]},
        "no_motion": {"actions": [[0.0] * 10 for _ in range(16)]},
    }
    (root / "action_streams.json").write_text(
        json.dumps(actions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "policy_ranking_cosmos3_official_droid_reference_canary.v1",
        "provider_inputs": {
            "initial_observation_sha256": _sha256_file(observation),
            "action_streams_sha256": canonical_sha256(actions),
        },
        "runtime": {"paid_execution_admitted": False, "provider_called": False},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "canary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def test_reference_bundle_is_deterministic_and_contains_only_registered_path(
    tmp_path: Path,
) -> None:
    reference = _reference_fixture(tmp_path / "reference")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_receipt = build_droid_reference_bundle(
        reference_dir=reference,
        output_bundle=first,
        receipt_path=tmp_path / "first.json",
    )
    second_receipt = build_droid_reference_bundle(
        reference_dir=reference,
        output_bundle=second,
        receipt_path=tmp_path / "second.json",
    )

    assert first_receipt["schema_version"] == RECEIPT_SCHEMA
    assert first_receipt["experiment_id"] == EXPERIMENT_ID
    assert first_receipt["bundle_sha256"] == second_receipt["bundle_sha256"]
    assert first_receipt["paid_resources_used"] is False
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert "provider_runtime/cosmos3_droid_reference/canary_manifest.json" in names
        assert "provider_runtime/cosmos3_droid_reference/action_streams.json" in names
        assert "provider_runtime/cosmos3_droid_reference/initial_observation.png" in names
        assert not any(name.startswith("provider_runtime/cosmos3_input/") for name in names)
        assert "provider_runtime/wam_provider_runtime_runner.py" in names


def test_reference_bundle_rejects_mutated_observation(tmp_path: Path) -> None:
    reference = _reference_fixture(tmp_path / "reference")
    (reference / "initial_observation.png").write_bytes(b"mutated")

    try:
        build_droid_reference_bundle(
            reference_dir=reference,
            output_bundle=tmp_path / "bundle.zip",
            receipt_path=tmp_path / "receipt.json",
        )
    except ValueError as exc:
        assert str(exc) == "official_droid_reference_initial_sha256_mismatch"
    else:
        raise AssertionError("mutated observation must fail closed")


def test_reference_bundle_passes_profile_specific_paid_path_inspection(
    tmp_path: Path,
) -> None:
    reference = _reference_fixture(tmp_path / "reference")
    bundle = tmp_path / "bundle.zip"
    receipt = build_droid_reference_bundle(
        reference_dir=reference,
        output_bundle=bundle,
        receipt_path=tmp_path / "receipt.json",
    )
    expected_hashes = {
        key: receipt[key]
        for key in (
            "reference_manifest_sha256",
            "initial_observation_sha256",
            "action_streams_sha256",
            "provider_runtime_runner_sha256",
        )
    }
    profile = replace(
        admission.DROID_REFERENCE_PROFILE,
        expected_bundle_sha256=receipt["bundle_sha256"],
        expected_bundle_size_bytes=receipt["bundle_size_bytes"],
        expected_embedded_input_hashes=expected_hashes,
    )

    result = admission.inspect_successor_bundle(
        bundle,
        receipt=receipt,
        smoke_inventory={},
        profile=profile,
    )

    assert result["status"] == "passed"
    assert result["blockers"] == []
    assert result["required_entry_count"] == 8
    assert result["embedded_input_hashes"] == expected_hashes
