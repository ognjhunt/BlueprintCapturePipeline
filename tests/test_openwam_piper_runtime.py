"""Development-only full beta binding/packaging; never a provider or GPU claim."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from blueprint_pipeline import beta_policy_screening_launch as launch
from blueprint_pipeline.beta_policy_candidate_catalog import load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import build_planning_inputs
from blueprint_pipeline.beta_policy_screening_plan import build_screening_plan
from blueprint_pipeline.beta_policy_screening_team import catalog_for_profile
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.openwam_piper_runtime import (
    CANDIDATE_ID, EMBODIMENT_ID, MINIMUM_GPU_MEMORY_GB, build_policy_spec, policy_binding,
)
from blueprint_pipeline.team_candidate_quick10_registry import profile_for_capture
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from tests.test_beta_policy_screening_launch import _plan
from tests.test_team_arm_kinematics import named_inputs


def prepared_registration(tmp_path, monkeypatch):
    base, _ = _plan(tmp_path, monkeypatch, scene_half="team_scene")
    profile = profile_for_capture(str(tmp_path / "capture"))
    model, manifest = named_inputs(tmp_path)
    manifest.update(embodiment_id=EMBODIMENT_ID, action_chunk_rows=1, control_hz=60)
    for camera, name in zip(manifest["cameras"], ("overhead", "left_wrist", "right_wrist"), strict=True):
        camera["name"] = name
    manifest = validate_embodiment_manifest(manifest)
    manifest_path = launch._write(tmp_path / "piper-manifest.json", manifest)
    rights = {"schema_version": "blueprint.team_policy_candidate_rights.v1", "status": "admitted",
              "team_id": "development-only", "checkpoint_id": CANDIDATE_ID, "candidate_id": CANDIDATE_ID,
              "license_to_run_granted": True, "policy_self_grading_permitted": False,
              "official_ranking_permitted": False, "secret_material_recorded": False,
              "adp_qualified_evidence_permitted": False}
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    registration = {"manifest": launch._record(manifest_path), "model": launch._record(model),
        "team_id": rights["team_id"], "checkpoint_id": CANDIDATE_ID, "rights_binding": rights,
        "policy_spec": build_policy_spec(manifest), "minimum_gpu_memory_gb": MINIMUM_GPU_MEMORY_GB,
        "deployment": {"configuration_id": EMBODIMENT_ID, "policy_route_id": "customer_controller",
            "companies": [], "relationship": "public_controller_integration",
            "evidence_url": "https://huggingface.co/OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper"}}
    profile.update(screening_candidates={CANDIDATE_ID: {"robot_policy": registration}},
                   screening_hard_ttl_seconds=1800, screening_candidate_cap_usd="1.00")
    registry_path = Path(__import__("os").environ["BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY"])
    registry = launch._read(registry_path)
    registry["profiles"] = [profile]
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path.write_text(json.dumps(registry))
    return base, profile, registration


def test_registered_local_checkpoint_retains_all_routes_and_builds_real_sealed_bundle(tmp_path, monkeypatch):
    from blueprint_pipeline.beta_robolab_policy_execution import validate_robolab_policy_execution_spec
    from blueprint_pipeline.policy_canary_session_bundle_identity import validate_policy_canary_provider_manifest
    from blueprint_pipeline.openwam_piper_bundle import transfer_byte_budget
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
    from blueprint_pipeline import openwam_piper_client_dependencies as client_dependencies
    from blueprint_pipeline import native_task_arena_vast as transport
    from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight

    base, profile, _ = prepared_registration(tmp_path, monkeypatch)
    catalog = catalog_for_profile(profile)
    assert len(catalog["candidates"]) == len(load_default_catalog()["candidates"])
    row = next(row for row in catalog["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert row["eligibility"]["enabled"] and row["eligibility"]["requires_credentials"] == []
    assert row["deployment"]["policy_route_id"] == "customer_controller"
    # The default screening lane cannot rent this 80 GB floor (RT-core parts at or below the
    # TensorRT ceiling top out at 48 GB); lifting the ceiling admits the RTX PRO 6000.
    from blueprint_pipeline.beta_policy_screening_gpu import gpu_infeasible_blockers
    monkeypatch.delenv("BLUEPRINT_VAST_MAX_COMPUTE_CAP", raising=False)
    assert gpu_infeasible_blockers(row["runtime"]) == ["screening_gpu_memory_floor_unrentable"]
    monkeypatch.setenv("BLUEPRINT_VAST_MAX_COMPUTE_CAP", "0")
    planning = build_planning_inputs(catalog=catalog, profile=profile, capture_root=tmp_path / "capture",
                                    source=base["source"], inputs_dir=tmp_path / "openwam-plan")
    plan = build_screening_plan(catalog, planning["scene"], capabilities=planning["capabilities"],
                               cost_quotes=planning["cost_quotes"], campaign_id="openwam-fixture")
    candidate = next(row for row in plan["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert candidate["status"] == "selected", candidate["blockers"]
    spec = launch._read(launch._bound_path(candidate["execution_spec"]["policy_execution_spec"]))
    assert validate_robolab_policy_execution_spec(spec, candidate_id=CANDIDATE_ID) == spec
    altered = deepcopy(spec)
    altered["policy_spec"]["source_revision"] = "0" * 40
    altered["execution_spec_digest"] = canonical_digest(altered, digest_field="execution_spec_digest")
    with pytest.raises(ValueError, match="openwam_runtime_policy_identity_invalid"):
        validate_robolab_policy_execution_spec(altered, candidate_id=CANDIDATE_ID)
    dependency_activations = []
    monkeypatch.setattr(client_dependencies, "activate_client_dependencies", lambda: dependency_activations.append(True))
    client = canary_policy_client(spec)
    assert dependency_activations == [True]
    assert client.require_named_arm_state and client.action_chunk_rows == 1
    assert client.binding.execution_horizon == 8
    assert client.evidence_summary()["execution_binding_digest"] == spec["execution_spec_digest"]
    result = launch.prepare_candidate(plan=plan, candidate=candidate, attempt_dir=tmp_path / "attempt")
    bundle = launch._read(Path(result["bundle_receipt"]["path"]))
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        manifest = validate_policy_canary_provider_manifest(json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json")))
        assert manifest["team_policy_interface_id"] == "openwam_piper_eef20.v1"
        assert "team_credential_env" not in manifest and manifest["minimum_gpu_memory_gb"] == 80
        assert "provider_runtime/native_task_packet/team_embodiment/robot.usd" in archive.namelist()
        assert "provider_runtime/openwam-piper-source.v1.json" in archive.namelist()
        assert b"openwam_piper_provisioning" in archive.read(f"provider_runtime/adp009d_policy_provisioning.{CANDIDATE_ID}.sh")
    download, upload = transfer_byte_budget(bundle)
    assert download > 24_835_000_000 + 64 * 1024**3 and upload == 8_000_000_000
    preflight = _blueprint_bundle_preflight(job_dir=tmp_path / "transport-preflight", generated_at="fixture",
        enable_blueprint_bundle=True, enable_isaac_smoke=True,
        provider_bundle_kind="native_task_arena_policy_canary_session", bundle_path=Path(bundle["bundle_path"]),
        provider_bundle_url="https://storage.example.com/bundle.zip", provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    observed = []
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY", "download")
    monkeypatch.setattr(transport, "run_arena_native_control_vast", lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready"})
    authority = launch._read(Path(result["authority"]["path"]))
    transport.run_native_task_arena_policy_canary_session_vast(job_dir=tmp_path / "transport",
        prepared_bundle=bundle, session_authority=authority, paid_resource_admission_grant=None, execute=False,
        hard_cap_usd=float(authority["hard_cap_usd"]), hard_ttl_seconds=authority["hard_ttl_seconds"])
    assert len(observed) == 1 and observed[0]["expected_provider_download_bytes"] == download
    assert observed[0]["expected_provider_upload_bytes"] == upload
    assert observed[0]["min_gpu_ram_mb"] == 80_000
    assert observed[0]["forward_hf_token"] is False and observed[0]["stale_offer_create_retry_limit"] == 0
    assert observed[0]["require_independent_watchdog"] is True
    assert not (tmp_path / "attempt/allocator_launch_intent.json").exists()


@pytest.mark.parametrize("fault", ["body", "cadence", "frames", "checkpoint", "chunk", "rights", "memory"])
def test_invalid_local_binding_keeps_individual_reason_and_frozen_candidates(tmp_path, monkeypatch, fault):
    _, profile, registration = prepared_registration(tmp_path, monkeypatch)
    policy = registration["policy_spec"]
    if fault == "body":
        policy["embodiment_id"] = "unrelated_body"
    elif fault == "cadence":
        policy["action_hz"] = 15
    elif fault == "frames":
        policy["binding"]["native_tcp_from_policy_tcp"] = {}
    elif fault == "checkpoint":
        policy["model_identity_digest"] = "sha256:" + "e" * 64
    elif fault == "chunk":
        policy["action_chunk_rows"] = 32
    elif fault == "memory":
        registration["minimum_gpu_memory_gb"] = True
    else:
        registration["rights_binding"]["license_to_run_granted"] = False
    catalog = catalog_for_profile(profile)
    row = next(row for row in catalog["candidates"] if row["candidate_id"] == CANDIDATE_ID)
    assert not row["eligibility"]["enabled"] and row["eligibility"]["blockers"]
    for original in load_default_catalog()["candidates"]:
        if original["candidate_id"] == CANDIDATE_ID:
            assert row["eligibility"]["blockers"] != original["eligibility"]["blockers"]
            continue
        assert next(row for row in catalog["candidates"] if row["candidate_id"] == original["candidate_id"]) == original


def test_explicit_frame_identity_is_labeled_assumption_and_caller_mutations_do_not_change_client(tmp_path, monkeypatch):
    _, _, registration = prepared_registration(tmp_path, monkeypatch)
    binding = policy_binding(registration["policy_spec"])
    assert "recording_tcp_and_firmware_unknown" in binding.frame_binding_reference
    assert binding.control_hz == 30


def test_source_archive_verifies_without_git_and_rejects_tamper(tmp_path):
    from blueprint_pipeline import openwam_piper_source as source

    source.stage_source(tmp_path)
    extracted = source.extract_source(tmp_path / source.SOURCE_ARCHIVE, tmp_path / source.SOURCE_MANIFEST,
                                      tmp_path / "source")
    assert not (extracted / ".git").exists()
    assert (extracted / "openwam/deploy/server.py").is_file()
    assert source.verify_source_directory(extracted, tmp_path / source.SOURCE_MANIFEST) == extracted
    (extracted / "openwam/deploy/server.py").write_text("changed")
    with pytest.raises(ValueError, match="source_file_changed"):
        source.verify_source_directory(extracted, tmp_path / source.SOURCE_MANIFEST)


def test_retained_source_fits_privileged_release_leaf_budget():
    from blueprint_pipeline import openwam_piper_source as source
    from scripts import release_source_manifest

    vendor = Path(source.__file__).parent / "_vendor"
    retained = list(vendor.glob("openwam-source-*"))
    assert retained
    assert all(path.stat().st_size <= release_source_manifest.MAX_LEAF_BYTES for path in retained)
    assert sum(path.stat().st_size for path in retained) == source.SOURCE_ARCHIVE_BYTES


@pytest.mark.parametrize("damage", ["missing", "changed", "oversized"])
def test_retained_source_damage_is_rejected_before_bundle_publication(tmp_path, monkeypatch, damage):
    import shutil
    from blueprint_pipeline import openwam_piper_source as source

    package = tmp_path / "package"
    vendor = package / "_vendor"
    vendor.mkdir(parents=True)
    original = Path(source.__file__).parent / "_vendor"
    for name in (source.SOURCE_MANIFEST, *source.SOURCE_PARTS):
        shutil.copyfile(original / name, vendor / name)
    part = vendor / source.SOURCE_PARTS[0]
    if damage == "missing":
        part.unlink()
    else:
        raw = part.read_bytes()
        part.write_bytes(raw + b"extra" if damage == "oversized" else bytes([raw[0] ^ 1]) + raw[1:])
    monkeypatch.setattr(source, "__file__", str(package / "openwam_piper_source.py"))
    output = tmp_path / "runtime"
    output.mkdir()
    with pytest.raises(ValueError, match="openwam_source_archive_changed"):
        source.stage_source(output)
    assert list(output.iterdir()) == []


def test_download_is_exact_bounded_and_does_not_retry_or_publish_bad_bytes(tmp_path):
    import time
    from blueprint_pipeline.openwam_piper_provisioning import fetch_exact, TransferBudget

    data, calls = b"exact-fixture", []
    row = {"url": "https://huggingface.co/fixed/revision/file", "size_bytes": len(data),
           "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
    def fetch(url, **kwargs):
        calls.append(kwargs)
        kwargs["output_path"].write_bytes(data)
    output = tmp_path / "output"
    result = fetch_exact(row, output, deadline=time.time() + 60, fetch=fetch)
    assert result["status"] == "verified" and len(calls) == 1
    assert calls[0]["max_bytes"] == len(data) and calls[0]["timeout_seconds"] <= 60
    assert fetch_exact(row, output, deadline=time.time() + 60, fetch=fetch)["cache_hit"]
    changed = deepcopy(row)
    changed["sha256"] = "sha256:" + "e" * 64
    with pytest.raises(ValueError, match="download_bytes_changed"):
        fetch_exact(changed, tmp_path / "bad", deadline=time.time() + 60, fetch=fetch)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.partial").exists()
    previous = len(calls)
    with pytest.raises(ValueError, match="total_download_bound_exceeded"):
        fetch_exact(row, tmp_path / "excess", deadline=time.time() + 60, fetch=fetch,
                    budget=TransferBudget(len(data) - 1))
    assert len(calls) == previous
    with pytest.raises(ValueError, match="deadline_exceeded"):
        fetch_exact(row, tmp_path / "late", deadline=time.time() - 1, fetch=fetch)
    assert len(calls) == previous


def test_pinned_client_replaces_preloaded_version_before_native_connection(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from blueprint_pipeline import openwam_piper_client_dependencies as client

    # Synthetic wheel exercises precedence and exact installed-file comparison;
    # the dependency inventory tests verify the real published wheel identity.
    wheel_root = tmp_path / "openwam_client_wheels"
    wheel_root.mkdir()
    wheel = wheel_root / "fixture-websockets.whl"
    code = b"__version__ = '16.0'\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("websockets/__init__.py", code)
    package = tmp_path / "openwam_client/websockets"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(code)
    row = {"filename": wheel.name, "size_bytes": wheel.stat().st_size,
           "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    monkeypatch.setattr(client, "verified_dependency_manifest", lambda path: {"thin_client_wheels": [row]})
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(__version__="10.4", __file__="old-runtime"))
    monkeypatch.setitem(sys.modules, "websockets.sync.client", SimpleNamespace(old=True))
    client.activate_client_dependencies(tmp_path)
    import websockets
    assert websockets.__version__ == "16.0" and Path(websockets.__file__).parent == package
    assert "websockets.sync.client" not in sys.modules
    (package / "__init__.py").write_text("__version__='16.0'\nraise RuntimeError('changed')")
    with pytest.raises(ValueError, match="client_installation_changed"):
        client.activate_client_dependencies(tmp_path)
