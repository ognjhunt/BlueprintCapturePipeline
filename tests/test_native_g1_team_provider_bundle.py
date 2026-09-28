"""A selected G1 bundle binds live approval, scene lineage and exact bytes."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
import zipfile

import pytest

from blueprint_pipeline import native_g1_team_provider_bundle as bundle
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_policy_run_intake import stage_g1_team_policy_run
from blueprint_pipeline.native_g1_team_policy_execution_packet import prepare_g1_team_policy_execution_packet
from tests.test_native_g1_team_policy_authority import _authority
from tests.test_native_g1_team_policy_run_request import NOW
from tests.test_native_g1_team_policy_run_request import _request
from tests.test_native_g1_team_policy_approval import _approval
from tests.test_team_policy_delivery_profile import _profile
from tests.test_native_g1_shared_scene_episode import _Scene
from tests.test_native_rigid_episode_telemetry import _spec


COMMIT = "a" * 40


def _inputs(tmp_path: Path, monkeypatch, *, endpoint=False):
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_policy_run_request.time.time", lambda: NOW)
    authority, _ = _authority(tmp_path, monkeypatch)
    if endpoint:
        # A new immutable queue intent uses the same owner/source registry.
        original = json.loads(authority["intent_path"].read_text())
        current = bundle.verify_g1_team_policy_authority(**authority)
        setup = current["trusted_setup"]
        profile = _profile(setup, {
            "mode": "authenticated_endpoint", "endpoint_url": "https://policy.example.org/v1/action",
            "auth_secret_ref": "secretref:team/policy", "timeout_ms": 5000,
        })
        request = _request(setup, profile)
        queue = tmp_path / "endpoint-queue"
        staged = stage_g1_team_policy_run(
            value=request, registry_path=authority["registry_path"], queue_root=queue,
            authenticated_client=original["authenticated_issuer"],
            trusted_clients=authority["trusted_clients"], now_epoch=NOW,
        )
        authority["intent_path"] = queue / staged["intent_id"] / "intent.json"
        approval = _approval(setup, profile, {
            "mode": "authenticated_endpoint", "profile_digest": profile["profile_digest"],
            "approved_origin": "https://policy.example.org", "resolved_secret_ref": "secretref:team/policy",
        })
        authority["approval_path"].write_text(json.dumps(approval))
    packet = prepare_g1_team_policy_execution_packet(
        **authority, output_dir=tmp_path / "execution", implementation_commit=COMMIT,
    )
    setup = packet["trusted_setup"]
    scene = tmp_path / "scene"
    scene.mkdir()
    request = {
        "scene_id": setup["scene_id"], "task_id": setup["task_id"],
        "task_spec": {"task_success_contract_digest": setup["task_success_contract_digest"]},
        "g1_scene_derivation": {
            "schema_version": "native_g1_scene_packet_derivation.v1",
            "claim_ceiling": "development_only",
            **{key: setup[key] for key in (
                "source_packet_receipt_digest", "source_scene_plan_digest",
                "source_declared_task_success_contract_digest", "setup_digest",
            )},
        },
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    plan = {
        **_Scene.plan,
        "scene_id": setup["scene_id"], "task_id": setup["task_id"],
        "task_kind": "rigid_pick_place", "robot": {"robot_id": "unitree_g1"},
        "task_spec": {**_spec(), "prompt": "pick the box", "task_kind": "rigid_pick_place",
                      "task_success_contract": setup["task_success_contract"],
                      "task_success_contract_digest": setup["task_success_contract_digest"]},
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    for name, value in ((bundle.REQUEST_FILENAME, request), (bundle.PLAN_FILENAME, plan)):
        (scene / name).write_text(json.dumps(value))
    receipt = {"receipt_digest": "sha256:" + "f" * 64,
               "request_digest": request["request_digest"],
               "arena_scene_plan_digest": plan["plan_digest"]}
    rows = [{"relative_path": name} for name in (bundle.REQUEST_FILENAME, bundle.PLAN_FILENAME)]
    monkeypatch.setattr(bundle, "verify_native_task_arena_packet", lambda root: (root, receipt, rows))
    publisher = tmp_path / "publisher"
    (publisher / "source/.git").mkdir(parents=True)
    (publisher / "source/.git/config").write_text("[remote \"origin\"]\n extraheader = private credential\n")
    (publisher / "source/LICENSE").write_text("test source license")
    publisher_receipt = {"receipt_digest": "sha256:" + "d" * 64}
    monkeypatch.setattr(bundle, "verify_g1_publisher_source", lambda root: publisher_receipt)
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    runtime_archive = tmp_path / "runtime.zip"
    runtime_archive.write_bytes(b"test external runtime")
    runtime_value = {
        "runtime_profile": "unitree_g1", "redistribution_permitted": True,
        "receipt_digest": "sha256:" + "c" * 64, "packet_sha256": bundle._sha256(runtime_archive),
        "packet_size_bytes": runtime_archive.stat().st_size, "verified_packet_path": str(runtime_archive),
    }
    runtime.write_text(json.dumps(runtime_value))
    monkeypatch.setattr(bundle, "verify_native_task_runtime_source_packet", lambda path: runtime_value)
    monkeypatch.setattr(bundle, "_review_g1_runtime_wheels", lambda *args: {"status": "exact_g1_wheels_approved"})
    sonic = tmp_path / "sonic"
    sonic.mkdir()
    models = []
    for role in ("encoder", "decoder"):
        path = sonic / ("model_" + role + ".onnx")
        path.write_bytes(role.encode())
        models.append({"role": role, "path": str(path), "sha256": bundle._sha256(path),
                       "size_bytes": path.stat().st_size})
    monkeypatch.setattr(bundle, "_sonic_inventory", lambda: {"files": [
        {"role": row["role"], "path": Path(row["path"]).name,
         "sha256": row["sha256"].removeprefix("sha256:"), "size_bytes": row["size_bytes"]}
        for row in models
    ]})
    args = {
        "job_dir": tmp_path / "bundle", "execution_packet_path": Path(packet["packet_path"]),
        "authority_arguments": authority, "scene_packet_root": scene,
        "publisher_source": publisher, "runtime_source_receipt": runtime,
        "sonic_asset_dir": sonic, "expected_implementation_commit": COMMIT,
    }
    return args, receipt


def test_seals_selected_scene_and_transport_without_credentials_or_paid_mutation(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    assert receipt["status"] == "sealed_not_admitted"
    assert receipt["claim_ceiling"] == "development_only"
    assert receipt["credential_value_included"] is False
    assert receipt["provider_mutation_performed"] is False
    assert receipt["policy_runtime_required"] is True
    loaded = bundle.load_verified_g1_team_provider_bundle(
        Path(receipt["receipt_path"]),
        expected_implementation_commit=COMMIT, authority_arguments=args["authority_arguments"],
    )
    assert loaded == receipt
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        config = archive.read("provider_runtime/publisher-source/source/.git/config").decode()
        assert "private credential" not in config
        assert "filemode = false" in config
        assert b"native_g1_team_provider_runtime" in archive.read(bundle.ENTRYPOINT)
        assert not any("pi_tokenizer/" in name for name in archive.namelist())


def test_reopened_approval_change_blocks_existing_bundle_before_admission(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    approval_path = args["authority_arguments"]["approval_path"]
    value = json.loads(approval_path.read_text())
    value["operator_reviewer"] = "changed approval"
    value["approval_digest"] = canonical_digest(value, digest_field="approval_digest")
    approval_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="authority_changed"):
        bundle.load_verified_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
            authority_arguments=args["authority_arguments"],
        )


def test_rejects_foreign_g1_scene_lineage_before_output_is_created(tmp_path, monkeypatch):
    args, receipt = _inputs(tmp_path, monkeypatch)
    path = args["scene_packet_root"] / bundle.REQUEST_FILENAME
    request = json.loads(path.read_text())
    request["g1_scene_derivation"]["source_packet_receipt_digest"] = "sha256:" + "0" * 64
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    receipt["request_digest"] = request["request_digest"]
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="scene_lineage_invalid"):
        bundle.build_g1_team_provider_bundle(**args)
    assert not args["job_dir"].exists()


def test_bundle_byte_change_is_rejected_without_rebuilding(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    with Path(receipt["bundle_path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="bundle_bytes_invalid"):
        bundle.load_verified_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
            authority_arguments=args["authority_arguments"],
        )


def _rewrite_bundle(receipt, *, change_manifest=None, extra=None):
    path = Path(receipt["bundle_path"])
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    manifest = {key: value for key, value in receipt.items() if key not in bundle._RECEIPT_ONLY_FIELDS}
    if change_manifest:
        change_manifest(manifest)
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    with zipfile.ZipFile(path, "w") as archive:
        for info, value in entries:
            archive.writestr(info, json.dumps(manifest).encode() if info.filename == bundle.MANIFEST else value)
        if extra:
            info = zipfile.ZipInfo(extra)
            info.create_system = 3
            info.external_attr = (stat.S_IFDIR | 0o700) << 16
            archive.writestr(info, b"")
    receipt.update(manifest)
    receipt["bundle_size_bytes"] = path.stat().st_size
    receipt["bundle_sha256"] = bundle._sha256(path)
    Path(receipt["receipt_path"]).write_text(json.dumps(receipt))


@pytest.mark.parametrize("field,value", [
    ("intent_id", "g1-team-foreign"), ("objective_id", "g1_navigation_goal"),
    ("container_image", "foreign:latest"), ("status", "ready"),
])
def test_consistently_resealed_manifest_cannot_change_selected_authority(tmp_path, monkeypatch, field, value):
    args, _ = _inputs(tmp_path, monkeypatch)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    _rewrite_bundle(receipt, change_manifest=lambda manifest: manifest.update({field: value}))
    with pytest.raises(ValueError, match="bundle_manifest_binding_invalid"):
        bundle.load_verified_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
            authority_arguments=args["authority_arguments"],
        )


def test_directory_traversal_and_changed_external_runtime_block_before_admission(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    runtime = Path(receipt["runtime_source_packet"]["packet_path"])
    runtime.write_bytes(b"altered source layer")
    with pytest.raises(ValueError, match="external_runtime_bytes_invalid"):
        bundle.load_verified_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
            authority_arguments=args["authority_arguments"],
        )
    _rewrite_bundle(receipt, extra="../../foreign/")
    with pytest.raises(ValueError, match="archive_path_invalid"):
        bundle.load_verified_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
            authority_arguments=args["authority_arguments"],
        )


@pytest.mark.slow
def test_selected_bundle_entrypoints_import_in_isolated_provider_interpreter(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    root = tmp_path / "extracted"
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        archive.extractall(root)
    runtime_root = root / "provider_runtime"
    script = textwrap.dedent('''
        import pathlib, runpy, sys
        runtime_root = pathlib.Path(sys.argv[1])
        sys.path.insert(0, sys.argv[1])
        module = sys.argv[2]
        if module.endswith(('native_g1_team_policy_worker', 'native_g1_team_vm_host')):
            import importlib
            importlib.import_module('blueprint_pipeline.native_g1_team_relay_runtime_session')
            importlib.import_module('blueprint_pipeline.native_g1_team_vm_output')
        sys.argv = [module, '--help']
        try:
            runpy.run_module(module, run_name='__main__')
        except SystemExit as exc:
            assert exc.code == 0
        for name, loaded in sys.modules.items():
            if name.startswith('blueprint_pipeline') or name.startswith('rfc8785'):
                assert pathlib.Path(loaded.__file__).is_relative_to(runtime_root), name
    ''')
    for module in ("native_g1_team_provider_runtime", "native_g1_team_worker_supervisor",
                   "native_g1_team_policy_worker", "native_task_runtime_source_provision",
                   "native_g1_team_vm_host"):
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(runtime_root), "blueprint_pipeline." + module],
            cwd=tmp_path, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        if module == "native_g1_team_policy_worker":
            assert "--policy-relay-config" in result.stdout
    shell = root / bundle.ENTRYPOINT
    assert subprocess.run(["bash", "-n", str(shell)], capture_output=True).returncode == 0


def test_retained_bundle_evidence_survives_expired_launch_approval_without_admitting_spend(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=True)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    authority = {**args["authority_arguments"], "now_epoch": NOW + 7200}
    with pytest.raises(ValueError):
        bundle.load_verified_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
            authority_arguments=authority,
        )
    retained = bundle.read_retained_g1_team_provider_bundle(
        Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
    )
    assert retained.bundle == receipt
    assert retained.execution_packet["packet_digest"] == receipt["execution_packet_digest"]
    assert retained.bundle["status"] == "sealed_not_admitted"
    assert retained.spend_admitted is False


@pytest.mark.parametrize("change", ["manifest", "external_runtime", "archive_bytes"])
def test_retained_bundle_read_refuses_changed_input_bytes(tmp_path, monkeypatch, change):
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=True)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    if change == "manifest":
        _rewrite_bundle(receipt, change_manifest=lambda value: value.update({"intent_id": "foreign-intent"}))
    elif change == "external_runtime":
        Path(receipt["runtime_source_packet"]["packet_path"]).write_bytes(b"changed")
    else:
        Path(receipt["bundle_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="g1_team_bundle"):
        bundle.read_retained_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
        )


@pytest.mark.parametrize("commit", [None, "", "main", "a" * 39, "g" * 40])
def test_retained_reader_requires_exact_commit_before_opening_inputs(tmp_path, commit):
    with pytest.raises(ValueError, match="g1_team_bundle_implementation_commit_invalid"):
        bundle.read_retained_g1_team_provider_bundle(
            tmp_path / "absent.json", expected_implementation_commit=commit,
        )
