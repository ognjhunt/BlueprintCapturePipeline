"""Real archive/static transport checks; native SDK/source retrieval are fixture inputs."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import zipfile

import pytest

from blueprint_pipeline import native_g1_beta_bundle as bundle
from blueprint_pipeline import native_g1_beta_session as session
from blueprint_pipeline import beta_policy_screening_g1 as g1
from blueprint_pipeline.beta_policy_screening_launch import _record
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_g1_beta_session import inputs


def fixture(tmp_path, monkeypatch):
    plan, candidate, spec = inputs(tmp_path, monkeypatch)
    publisher = tmp_path / "source-stage"
    source = publisher / "source"
    source.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "LICENSE").write_text("Publisher source fixture license\n")
    (source / "lerobot/scripts").mkdir(parents=True)
    (source / "lerobot/src").mkdir()
    (source / "lerobot/scripts/serve_lerobot_vla_http.py").write_text("# Hermetic publisher source fixture\n")
    (source / "lerobot/src/__init__.py").write_text("")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "source fixture"], check=True)
    with (source / ".git/config").open("a") as stream:
        stream.write('[http]\n extraheader = private-test-credential\n')
    publisher_receipt = publisher / "native_g1_publisher_source_stage.v1.json"
    publisher_receipt.write_text("{}")
    runtime_archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(runtime_archive, "w") as archive:
        archive.writestr("source.txt", b"sealed source runtime fixture")
    runtime = {"runtime_profile": "unitree_g1", "redistribution_permitted": True,
        "receipt_digest": "sha256:" + "a" * 64, "packet_sha256": bundle._sha(runtime_archive),
        "packet_size_bytes": runtime_archive.stat().st_size, "verified_packet_path": str(runtime_archive)}
    runtime_receipt = tmp_path / "runtime.json"
    runtime_receipt.write_text(json.dumps(runtime))
    # Rebind the actual records while retaining the same exact native checkpoint
    # and current source; no GPU/source-verification claim is made by this fixture.
    spec = deepcopy(spec)
    spec["native_g1"]["publisher_source"] = _record(publisher_receipt)
    spec["native_g1"]["runtime_source"] = _record(runtime_receipt)
    rights = spec["rights_authority"]
    rights["registration_digest"] = canonical_digest({"candidate_id": spec["candidate_id"], "native_g1": spec["native_g1"]})
    rights["authority_digest"] = canonical_digest(rights, digest_field="authority_digest")
    spec["runtime_identity"] = g1.runtime_identity(spec)
    spec["runtime_identity_digest"] = canonical_digest(spec["runtime_identity"])
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    g1.validate_execution_spec(spec)
    # Build the session first with the original plan, then rebind exact runtime
    # digest in the sealed authority for the byte-oriented archive fixture.
    # Session validation independently checks the real rights and budget fields.
    authority = {"schema_version": session.SCHEMA, "run_kind": session.RUN_KIND,
        "run_id": "c" * 32, "candidate_ids": [spec["candidate_id"]], "episodes_per_policy": candidate["episode_count"],
        "cells": candidate["scenario_cells"], "beta_screening_admission": {
            "plan_digest": plan["plan_digest"], "candidate_id": candidate["candidate_id"],
            "configuration_digest": candidate["configuration_digest"], "admission_digest": candidate["admission_digest"],
            "scene_revision_digest": plan["scene_revision_digest"], "source": spec["source"], "site_id": spec["site_id"],
            "runtime_identity_digest": spec["runtime_identity_digest"], "scenario_plan_digest": plan["scenario_plan_digest"],
            "maximum_cost_usd": "4.00", "cost_quote": candidate["cost_quote"]},
        "runtime_inputs_digest": spec["execution_spec_digest"], "task_success_contract_digest": spec["task_success_contract_digest"],
        "rights_authority": rights, "implementation_commit": "a" * 40,
        "resource_name": session.RESOURCE_PREFIX + "c" * 32, "hard_cap_usd": 4.0, "max_hourly_rate_usd": "1.00",
        "hard_ttl_seconds": 1800, "maximum_provider_allocations": 1, "retry_cap": 0, "automatic_retry_authorized": False,
        "caller_surviving_watchdog_required": True, "provider_wide_launch_lock_required": True,
        "canonical_allocator": "blueprint_pipeline.paid_resource_allocator", "claim_ceiling": "development_only",
        "public_media_publication_authorized": False, "model_redistribution_authorized": False,
        "official_ranking_authorized": False, "scene_promotion_authorized": False, "physical_outcome_claimed": False}
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    session.validate_session_authority(authority)
    packet = tmp_path / "native"
    receipt = json.loads((packet / "native_task_arena_packet_receipt.v1.json").read_text())
    receipt["arena_scene_plan_digest"] = spec["scene_plan_digest"]
    # The packet builder's exact-byte verification is separately exercised by
    # test_native_g1_beta_packet; this fixture provides its verified return shape.
    monkeypatch.setattr(bundle, "verify_native_task_arena_packet", lambda path: (packet, receipt,
        [{"relative_path": p.name} for p in packet.iterdir() if p.is_file()]))
    monkeypatch.setattr(bundle, "verify_g1_publisher_source", lambda path: {})
    monkeypatch.setattr(bundle, "verify_native_task_runtime_source_packet", lambda path: runtime)
    monkeypatch.setattr(bundle, "_review_g1_runtime_wheels", lambda *a: {"status": "exact_g1_wheels_approved"})
    return authority, spec


def test_selected_dp_real_zip_reopens_and_passes_actual_vast_preflight(tmp_path, monkeypatch):
    authority, spec = fixture(tmp_path, monkeypatch)
    receipt = bundle.build_bundle(job_dir=tmp_path / "sealed", authority=authority, spec=spec, implementation_commit="a" * 40)
    assert receipt["candidate_ids"] == authority["candidate_ids"]
    assert receipt["cell_ids"] == [c["cell_id"] for c in authority["cells"]]
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        assert not any("/pi_tokenizer/" in name for name in archive.namelist())
        for name in ("native_g1_beta_authority", "native_g1_beta_authority_composer"):
            path = "provider_runtime/blueprint_pipeline/" + name + ".py"
            assert path in session.REQUIRED_ENTRIES
            assert archive.read(path) == (Path(session.__file__).parent / (name + ".py")).read_bytes()
        assert b"private-test-credential" not in archive.read("provider_runtime/publisher-source/source/.git/config")
        assert archive.read("provider_runtime/publisher-source/source/LICENSE").startswith(b"Publisher source")
        # Exercise the actual source verifier after extraction. Omitting the Git
        # index would falsely mark all tracked files deleted on the provider.
        from blueprint_pipeline import native_g1_policy_server_supervisor as supervisor
        extracted = tmp_path / "extracted"
        for name in archive.namelist():
            if name.startswith("provider_runtime/publisher-source/"):
                archive.extract(name, extracted)
        source = extracted / "provider_runtime/publisher-source/source"
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
        monkeypatch.setattr(supervisor, "PINNED_SOURCE_REVISION", revision)
        assert supervisor._source_revision(source / "lerobot/scripts/serve_lerobot_vla_http.py") == revision
    assert bundle.load_verified_bundle(tmp_path / "sealed" / (session.BUNDLE_SCHEMA + ".json"),
        expected_implementation_commit="a" * 40) == receipt
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
    # Exercise the real immutable archive adapter, including required runner and
    # static entrypoint contract; no name validator or provider operation is mocked.
    import inspect
    assert "provider_bundle_kind" in inspect.signature(_blueprint_bundle_preflight).parameters
    result = _blueprint_bundle_preflight(job_dir=tmp_path, generated_at="2026-10-30T00:00:00Z",
        enable_blueprint_bundle=True, enable_isaac_smoke=True, provider_bundle_kind=session.PROVIDER_KIND,
        bundle_path=Path(receipt["bundle_path"]), provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/output.zip")
    assert result["status"] == "passed"


def test_resealed_manifest_cannot_change_selected_candidate(tmp_path, monkeypatch):
    authority, spec = fixture(tmp_path, monkeypatch)
    receipt = bundle.build_bundle(job_dir=tmp_path / "sealed", authority=authority, spec=spec, implementation_commit="a" * 40)
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = json.loads(archive.read(session.MANIFEST))
        manifest["candidate_ids"] = ["humanoidarena_pi05_g1_dex3_sonic"]
        manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
        with pytest.raises(ValueError, match="selected_binding_changed"):
            bundle.validate_bundle_archive(archive, manifest)
