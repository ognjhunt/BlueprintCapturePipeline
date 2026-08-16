from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _find_contracts_src_dir() -> Path | None:
    for parent in REPO_ROOT.parents:
        candidate = parent / "BlueprintContracts" / "src"
        if candidate.is_dir():
            return candidate
    return None

contract_src_dir = _find_contracts_src_dir()

for candidate in (REPO_ROOT, SRC_DIR, contract_src_dir):
    if candidate is None:
        continue
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


@pytest.fixture(autouse=True)
def _isolated_pending_teardown_registry(tmp_path_factory, monkeypatch):
    """Paid lanes persist pending_teardown.v1 records; keep tests out of ~/."""
    registry = tmp_path_factory.mktemp("pending-teardowns")
    monkeypatch.setenv("BLUEPRINT_PENDING_TEARDOWN_DIR", str(registry))
    return registry


@pytest.fixture(autouse=True)
def _isolated_paid_provider_lane_lease_dir(tmp_path_factory, monkeypatch):
    """Paid lanes acquire an exclusive lane lease; keep tests out of ~/."""
    lease_dir = tmp_path_factory.mktemp("paid-lane-leases")
    monkeypatch.setenv("BLUEPRINT_PAID_PROVIDER_LANE_LEASE_DIR", str(lease_dir))
    return lease_dir


@pytest.fixture
def _materialize_generated_manifest_publication_fixture(
    monkeypatch, tmp_path_factory
):
    """Turn legacy fake manifest URLs into strict publication receipts.

    Existing live-profile tests use exact-commit ``example/repo`` URLs as
    network-free fixtures. Generated production builders no longer accept that
    shortcut. This adapter creates the same self-digesting receipt the real GCS
    publisher returns, then calls the unmodified production validator.
    """

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline import task_evaluation_live_profile as live_profile
    from blueprint_pipeline.robot_eval_provider_input_setup import (
        LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS,
    )

    original = live_profile.bind_live_profile_manifest_publication
    pytest_tmp_root = tmp_path_factory.getbasetemp().resolve()

    def bind_fixture(**kwargs):
        reference = str(kwargs.get("reference") or "")
        builder = str(kwargs.get("profile_builder") or "")
        if (
            "raw.githubusercontent.com/example/repo/" not in reference
            or LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS.get(builder)
            != "content_addressed_full_readback"
        ):
            return original(**kwargs)
        digest = str(kwargs["run_spec_digest"])
        matches = [
            item
            for item in kwargs["immutable_inputs"]
            if item.get("name") == "source_bundle_manifest"
            and item.get("digest") == digest
            and Path(str(item.get("path") or "")).expanduser().is_file()
        ]
        assert len(matches) == 1
        source = Path(str(matches[0]["path"])).expanduser().resolve()
        assert source.is_relative_to(pytest_tmp_root), (
            "publication fixture may write only beside pytest-owned inputs"
        )
        identity = digest.removeprefix("sha256:")
        receipt = {
            "schema_version": "task_evaluation_immutable_manifest_publication.v1",
            "status": "published",
            "source": {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
            },
            "profile_builder": builder,
            "publication_seam": "content_addressed_full_readback",
            "published_uri": f"gs://fixture/sha256/{identity[:2]}/{identity}.json",
            "storage_scheme": "gs",
            "remote_size_bytes": source.stat().st_size,
            "remote_sha256": digest,
            "provider_full_byte_readback_verified": True,
            "content_addressed_key": True,
            "exclusive_create": True,
            "upload_receipt_digest": "sha256:" + "a" * 64,
            "provider_compute_mutation_performed": False,
            "paid_resource_allocated": False,
            "raw_secret_values_recorded": False,
            "blockers": [],
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        receipt_path = source.parent / (
            f".{builder}-{identity}.publication.json"
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return original(**{**kwargs, "reference": str(receipt_path)})

    monkeypatch.setattr(
        live_profile, "bind_live_profile_manifest_publication", bind_fixture
    )
