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
import hashlib
import io
import tarfile

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
from blueprint_pipeline import native_g1_team_vm_bootstrap as host_bootstrap


COMMIT = "a" * 40


def _inputs(tmp_path: Path, monkeypatch, *, endpoint=False, archive=False, artifact_fault=None):
    monkeypatch.setattr("blueprint_pipeline.native_g1_team_policy_run_request.time.time", lambda: NOW)
    authority, _ = _authority(tmp_path, monkeypatch)
    staged_archive = None
    if endpoint or archive:
        # A new immutable queue intent uses the same owner/source registry.
        original = json.loads(authority["intent_path"].read_text())
        current = bundle.verify_g1_team_policy_authority(**authority)
        setup = current["trusted_setup"]
        delivery = {
            "mode": "authenticated_endpoint", "endpoint_url": "https://policy.example.org/v1/action",
            "auth_secret_ref": "secretref:team/policy", "timeout_ms": 5000,
        }
        if archive:
            staged_archive = tmp_path / "operator-policy.tar"
            with tarfile.open(staged_archive, "w") as stream:
                member = tarfile.TarInfo("../outside" if artifact_fault == "traversal" else
                                         "policy/other.py" if artifact_fault == "entrypoint" else "policy/run.py")
                content = b"#!/usr/bin/python3\n"
                member.size = len(content)
                if artifact_fault == "link":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/host-private-fixture"
                stream.addfile(member, io.BytesIO(content))
            delivery = {"mode": "noncontainer_artifact", "artifact_uri": "https://files.example.org/policy.tar",
                        "artifact_sha256": bundle._sha256(staged_archive), "entrypoint": "policy/run.py",
                        "protocol": "jsonl_observation_action_v1"}
        profile = _profile(setup, delivery)
        request = _request(setup, profile)
        queue = tmp_path / "endpoint-queue"
        staged = stage_g1_team_policy_run(
            value=request, registry_path=authority["registry_path"], queue_root=queue,
            authenticated_client=original["authenticated_issuer"],
            trusted_clients=authority["trusted_clients"], now_epoch=NOW,
        )
        authority["intent_path"] = queue / staged["intent_id"] / "intent.json"
        binding = {
            "mode": "authenticated_endpoint", "profile_digest": profile["profile_digest"],
            "approved_origin": "https://policy.example.org", "resolved_secret_ref": "secretref:team/policy",
        }
        if archive:
            binding = {"mode": "noncontainer_artifact", "profile_digest": profile["profile_digest"],
                       "artifact_sha256": delivery["artifact_sha256"],
                       "staged_artifact_path": str(tmp_path / "approved-operator-origin.tar")}
        approval = _approval(setup, profile, binding)
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
    if not endpoint:
        paths, catalogue = {}, {}
        for role, row in host_bootstrap.ASSETS.items():
            path = tmp_path / row["filename"]
            path.write_bytes(("fixture-host-asset-" + role).encode())
            paths[role] = path
            catalogue[role] = {**row, "size_bytes": path.stat().st_size,
                              "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
        monkeypatch.setattr(host_bootstrap, "ASSETS", catalogue)
        host_root = tmp_path / "host-package"
        host_bootstrap.seal_g1_vm_host_python(asset_paths=paths, output_root=host_root,
                                             implementation_commit=COMMIT)
        args["host_python_package_root"] = host_root
    if archive:
        args["staged_policy_artifact_path"] = staged_archive
    return args, receipt


@pytest.mark.parametrize("archive", [False, True])
def test_paired_bundle_ships_bound_host_assets_and_separate_vm_entrypoint(tmp_path, monkeypatch, archive):
    args, _ = _inputs(tmp_path, monkeypatch, archive=archive)
    original = json.loads(args["execution_packet_path"].read_text())
    receipt = bundle.build_g1_team_provider_bundle(**args)
    assert receipt["runtime_entrypoint"] == bundle.VM_ENTRYPOINT
    assert receipt["host_python_package"]["relative_root"] == bundle.HOST_PACKAGE_ROOT
    with zipfile.ZipFile(receipt["bundle_path"]) as sealed:
        assert json.loads(sealed.read(bundle.PACKET_RELATIVE_PATH)) == original
        assert b"native_g1_team_provider_runtime" in sealed.read(bundle.ENTRYPOINT)
        host_shell = sealed.read(bundle.VM_ENTRYPOINT)
        assert b"native_g1_team_vm_bootstrap.py" in host_shell
        assert b"native_g1_team_vm_host" in host_shell and b"-I -B" in host_shell
        for row in host_bootstrap.ASSETS.values():
            assert sealed.read(bundle.HOST_PACKAGE_ROOT + "/" + row["filename"]) == (
                args["host_python_package_root"] / row["filename"]).read_bytes()
        if archive:
            assert sealed.read(bundle.POLICY_ARTIFACT_PATH) == args["staged_policy_artifact_path"].read_bytes()
            assert receipt["policy_artifact"]["sha256"] == original["request"]["policy_profile"]["delivery"]["artifact_sha256"]
        else:
            assert receipt["policy_artifact"] is None
            assert bundle.POLICY_ARTIFACT_PATH not in sealed.namelist()


@pytest.mark.parametrize("fault", ["absent", "changed", "commit"])
def test_paired_host_package_is_required_and_exact_before_output(tmp_path, monkeypatch, fault):
    args, _ = _inputs(tmp_path, monkeypatch)
    root = args["host_python_package_root"]
    if fault == "absent":
        args.pop("host_python_package_root")
    elif fault == "changed":
        path = root / host_bootstrap.ASSETS["numpy"]["filename"]
        path.chmod(0o600)
        path.write_bytes(b"changed")
    else:
        path = root / host_bootstrap.MANIFEST_NAME
        value = json.loads(path.read_text())
        value["implementation_commit"] = "b" * 40
        value["manifest_digest"] = host_bootstrap._digest(value)
        path.chmod(0o600)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        bundle.build_g1_team_provider_bundle(**args)
    assert not args["job_dir"].exists()


def test_changed_archive_bytes_block_before_bundle_creation(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch, archive=True)
    args["staged_policy_artifact_path"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="policy_artifact"):
        bundle.build_g1_team_provider_bundle(**args)
    assert not args["job_dir"].exists()


@pytest.mark.parametrize("fault", ["traversal", "link", "entrypoint"])
def test_approved_hash_does_not_admit_unsafe_or_unrunnable_archive(tmp_path, monkeypatch, fault):
    args, _ = _inputs(tmp_path, monkeypatch, archive=True, artifact_fault=fault)
    with pytest.raises(ValueError, match="policy_artifact|member_unsafe"):
        bundle.build_g1_team_provider_bundle(**args)
    assert not args["job_dir"].exists()


@pytest.mark.parametrize("endpoint", [False, True])
def test_artifact_path_argument_is_refused_for_other_modes(tmp_path, monkeypatch, endpoint):
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=endpoint)
    args["staged_policy_artifact_path"] = tmp_path / "foreign-artifact"
    with pytest.raises(ValueError, match="policy_artifact"):
        bundle.build_g1_team_provider_bundle(**args)
    assert not args["job_dir"].exists()


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


def _rewrite_bundle(receipt, *, change_manifest=None, extra=None, remove=None):
    path = Path(receipt["bundle_path"])
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    manifest = {key: value for key, value in receipt.items() if key not in bundle._RECEIPT_ONLY_FIELDS}
    if change_manifest:
        change_manifest(manifest)
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    with zipfile.ZipFile(path, "w") as archive:
        for info, value in entries:
            if info.filename == remove:
                continue
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


@pytest.mark.parametrize("missing", [bundle.VM_ENTRYPOINT,
    "provider_runtime/blueprint_pipeline/native_g1_team_vm_bootstrap.py",
    "provider_runtime/blueprint_pipeline/native_g1_team_vm_host.py"])
def test_resealed_paired_bundle_requires_actual_host_launcher_and_modules(tmp_path, monkeypatch, missing):
    args, _ = _inputs(tmp_path, monkeypatch)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    _rewrite_bundle(receipt, remove=missing, change_manifest=lambda value: value.update(
        artifacts=[row for row in value["artifacts"] if row["relative_path"] != missing]))
    with pytest.raises(ValueError, match="host_package_runtime_missing"):
        bundle.read_retained_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT)


@pytest.mark.parametrize("fault", ["host_root", "host_digest", "host_row", "entrypoint", "archive_hash"])
def test_resealed_transport_cannot_change_host_or_approved_archive_binding(tmp_path, monkeypatch, fault):
    args, _ = _inputs(tmp_path, monkeypatch, archive=fault == "archive_hash")
    receipt = bundle.build_g1_team_provider_bundle(**args)

    def change(value):
        if fault == "host_root":
            value["host_python_package"]["relative_root"] = "provider_runtime/foreign-package"
        elif fault == "host_digest":
            value["host_python_package"]["manifest_digest"] = "sha256:" + "f" * 64
        elif fault == "host_row":
            value["artifacts"] = [row for row in value["artifacts"] if not row["relative_path"].endswith(
                host_bootstrap.ASSETS["numpy"]["filename"])]
        elif fault == "entrypoint":
            value["runtime_entrypoint"] = bundle.ENTRYPOINT
        else:
            value["policy_artifact"]["sha256"] = "sha256:" + "f" * 64

    _rewrite_bundle(receipt, change_manifest=change)
    with pytest.raises(ValueError):
        bundle.load_verified_g1_team_provider_bundle(
            Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT,
            authority_arguments=args["authority_arguments"],
        )


def test_endpoint_never_includes_host_package_and_rejects_foreign_metadata(tmp_path, monkeypatch):
    args, _ = _inputs(tmp_path, monkeypatch, endpoint=True)
    receipt = bundle.build_g1_team_provider_bundle(**args)
    assert receipt["runtime_entrypoint"] == bundle.ENTRYPOINT
    assert receipt["host_python_package"] is None and receipt["policy_artifact"] is None
    _rewrite_bundle(receipt, change_manifest=lambda value: value.update(host_python_package={"foreign": True}))
    with pytest.raises(ValueError, match="host_package_mode"):
        bundle.read_retained_g1_team_provider_bundle(Path(receipt["receipt_path"]), expected_implementation_commit=COMMIT)


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
                   "native_g1_team_vm_host", "native_g1_team_vm_bootstrap"):
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(runtime_root), "blueprint_pipeline." + module],
            cwd=tmp_path, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
        if module == "native_g1_team_policy_worker":
            assert "--policy-relay-config" in result.stdout
    for relative in (bundle.ENTRYPOINT, bundle.VM_ENTRYPOINT):
        assert subprocess.run(["bash", "-n", str(root / relative)], capture_output=True).returncode == 0


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
