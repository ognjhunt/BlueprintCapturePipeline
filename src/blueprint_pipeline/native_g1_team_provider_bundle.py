"""Seal a selected G1 policy and its exact scene for canonical paid admission.

This is transport preparation, not permission to allocate or run. HTTPS needs
a privately resolved credential at launch. OCI/archive delivery also needs an
independently admitted policy runtime; the Isaac image cannot run nested Docker.
The large reviewed Isaac source archive remains an external content-addressed
layer. No policy credential, HF token or separately hosted policy weights are
included here.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .native_g1_provider_bundle import (
    _contract_dependency, _files, _review_g1_runtime_wheels, _runtime_code_files, _sha256,
)
from .native_g1_publisher_source_stage import SOURCE_REPOSITORY, verify_g1_publisher_source
from .native_g1_team_policy_authority import verify_g1_team_policy_authority
from .native_g1_team_policy_execution_packet import FIELDS as PACKET_FIELDS, SCHEMA as PACKET_SCHEMA
from .native_g1_team_policy_worker import _execution_packet
from .native_task_arena_bundle import _write_zip_file, verify_native_task_arena_packet
from .native_task_arena_packet import REQUEST_SCHEMA_VERSION
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .native_task_runtime_source_packet import verify_native_task_runtime_source_packet


SCHEMA = "native_g1_team_provider_bundle.v1"
RESULT_SCHEMA = "native_g1_team_provider_result.v1"
RESULT_FILENAME = RESULT_SCHEMA + ".json"
PROVIDER_BUNDLE_KIND = "native_g1_team_policy"
MANIFEST = "provider_runtime/native_g1_team_provider_manifest.json"
ENTRYPOINT = "provider_runtime/run_adp_arena_provider_runtime.sh"
REQUEST_FILENAME = REQUEST_SCHEMA_VERSION + ".json"
PLAN_FILENAME = "native_task_arena_scene_plan.v1.json"
PACKET_RELATIVE_PATH = "provider_runtime/inputs/execution_packet.json"
SCENE_RELATIVE_ROOT = "provider_runtime/inputs/scene_packet"
_REPOSITORY = Path(__file__).resolve().parents[2]
_SONIC_INVENTORY = _REPOSITORY / "configs/g1_sonic_default_asset_inventory.v1.json"
_RECEIPT_ONLY_FIELDS = {"bundle_path", "bundle_size_bytes", "bundle_sha256", "receipt_path"}


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("g1_team_bundle_json_unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_team_bundle_json_invalid")
    return value


def _authority_matches(packet: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    intent = authority["intent"]
    if (
        packet["intent_id"] != intent["intent_id"]
        or packet["intent_digest"] != intent["intent_digest"]
        or packet["request"] != intent["request"]
        or packet["trusted_setup"] != authority["trusted_setup"]
        or packet["operator_approval"] != authority["operator_approval"]
    ):
        raise ValueError("g1_team_bundle_authority_changed")


def verify_g1_team_scene_packet(root: Path, packet: Mapping[str, Any]):
    """Reject a same-named task with foreign source geometry or task truth."""

    scene_root, receipt, rows = verify_native_task_arena_packet(root)
    request = _json(scene_root / REQUEST_FILENAME)
    plan = _json(scene_root / PLAN_FILENAME)
    setup = packet["trusted_setup"]
    derivation = request.get("g1_scene_derivation")
    task = plan.get("task_spec") or {}
    if (
        not isinstance(derivation, dict)
        or derivation.get("schema_version") != "native_g1_scene_packet_derivation.v1"
        or derivation.get("claim_ceiling") != "development_only"
        or any(derivation.get(key) != setup[key] for key in (
            "source_packet_receipt_digest", "source_scene_plan_digest",
            "source_declared_task_success_contract_digest", "setup_digest",
        ))
        or request.get("request_digest") != receipt["request_digest"]
        or request["request_digest"] != canonical_digest(request, digest_field="request_digest")
        or plan.get("plan_digest") != receipt["arena_scene_plan_digest"]
        or plan["plan_digest"] != canonical_digest(plan, digest_field="plan_digest")
        or any(value.get("scene_id") != setup["scene_id"]
               or value.get("task_id") != setup["task_id"] for value in (request, plan))
        or plan.get("task_kind") != "rigid_pick_place"
        or (plan.get("robot") or {}).get("robot_id") != "unitree_g1"
        or task.get("task_success_contract") != setup["task_success_contract"]
        or task.get("task_success_contract_digest") != setup["task_success_contract_digest"]
        or (request.get("task_spec") or {}).get("task_success_contract_digest")
        != setup["task_success_contract_digest"]
    ):
        raise ValueError("g1_team_bundle_scene_lineage_invalid")
    if packet["objective_id"] == "g1_navigation_goal":
        from .native_g1_navigation_goal import validate_g1_navigation_goal

        validate_g1_navigation_goal(task)
    return scene_root, receipt, rows


def _sonic_inventory() -> dict[str, Any]:
    return _json(_SONIC_INVENTORY)


def _sonic_assets(root: Path, inventory: Path) -> list[dict[str, Any]]:
    # The parameter is an operator-known inventory path, never a team input.
    if inventory != _SONIC_INVENTORY or not root.is_absolute() or root.is_symlink():
        raise ValueError("g1_team_bundle_sonic_path_invalid")
    rows = []
    for row in _sonic_inventory()["files"]:
        path = root / row["path"]
        if (
            path.is_symlink() or not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or _sha256(path) != "sha256:" + row["sha256"]
        ):
            raise ValueError("g1_team_bundle_sonic_bytes_invalid")
        rows.append({"role": row["role"], "path": str(path), "sha256": _sha256(path),
                     "size_bytes": path.stat().st_size})
    return rows


def _entrypoint() -> str:
    return '''#!/usr/bin/env bash
set -u
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${BLUEPRINT_ADP_ARENA_OUTPUT_DIR:-$RUNTIME_DIR/../runtime_output}"
mkdir -p "$OUT_DIR"
cd "$RUNTIME_DIR"
export BLUEPRINT_G1_PINNED_ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim:6.0.1@sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb"
phase=media-toolchain
rc=0
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq >"$OUT_DIR/media_toolchain_install.log" 2>&1 && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >>"$OUT_DIR/media_toolchain_install.log" 2>&1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  rc=2
else
  phase=runtime-source-provisioning
  /isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision \\
    --source-receipt "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_source_packet.v1.json" \\
    --source-packet "$RUNTIME_DIR/native_task_runtime_sources/native_task_runtime_sources.zip" \\
    --extraction-dir "$RUNTIME_DIR/provisioned_runtime_sources" \\
    --output "$OUT_DIR/native_task_runtime_source_provisioning.v1.json" --simulator-root /isaac-sim
  rc=$?
fi
if [ "$rc" -eq 0 ]; then
  phase=selected-worker
  /isaac-sim/python.sh -m blueprint_pipeline.native_g1_team_provider_runtime \\
    --runtime-root "$RUNTIME_DIR" --output-dir "$OUT_DIR"
  rc=$?
fi
if [ ! -f "$OUT_DIR/native_g1_team_provider_result.v1.json" ]; then
  /isaac-sim/python.sh - "$OUT_DIR" "$phase" "$rc" <<'PY'
import json, sys
from pathlib import Path
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
value = {"schema_version": "native_g1_team_provider_result.v1", "status": "blocked",
         "claim_ceiling": "development_only", "stage_reached": sys.argv[2],
         "runner_exit_code": int(sys.argv[3]), "verified_output": None,
         "provider_teardown_verified": False, "official_billing_reconciled": False,
         "public_redistribution_authorized": False}
value["result_digest"] = canonical_digest(value, digest_field="result_digest")
(Path(sys.argv[1]) / "native_g1_team_provider_result.v1.json").write_text(json.dumps(value) + "\\n")
PY
  rc=2
fi
exit "$rc"
'''


def _verify_artifacts(manifest: Mapping[str, Any], read, names: set[str]) -> None:
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("g1_team_bundle_artifact_manifest_invalid")
    expected = {row.get("relative_path") for row in rows}
    if len(expected) != len(rows) or expected | {MANIFEST} != names:
        raise ValueError("g1_team_bundle_artifact_set_invalid")
    for row in rows:
        relative = row.get("relative_path")
        if (not isinstance(relative, str) or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts):
            raise ValueError("g1_team_bundle_artifact_path_invalid")
        digest = hashlib.sha256()
        size = 0
        with read(relative) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if size != row.get("size_bytes") or "sha256:" + digest.hexdigest() != row.get("sha256"):
            raise ValueError("g1_team_bundle_artifact_bytes_invalid")


def verify_g1_team_manifest_binding(manifest: Mapping[str, Any], packet: Mapping[str, Any]) -> None:
    """Every launch-relevant manifest field must describe the approved packet."""

    mode = packet["delivery_mode"]
    expected = {
        "schema_version": SCHEMA, "status": "sealed_not_admitted",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND, "container_image": NATIVE_TASK_ARENA_IMAGE,
        "implementation_commit": packet["implementation_commit"],
        "execution_packet_digest": packet["packet_digest"], "intent_id": packet["intent_id"],
        "policy_profile_digest": packet["policy_profile_digest"],
        "source_packet_receipt_digest": packet["source_packet_receipt_digest"],
        "operator_approval_digest": packet["operator_approval"]["approval_digest"],
        "scene_id": packet["trusted_setup"]["scene_id"], "task_id": packet["trusted_setup"]["task_id"],
        "objective_id": packet["objective_id"], "delivery_mode": mode,
        "policy_runtime_required": mode != "authenticated_endpoint",
        "private_credential_staging_required": mode == "authenticated_endpoint",
        "expected_output_filename": RESULT_FILENAME, "runtime_entrypoint": ENTRYPOINT,
        "claim_ceiling": "development_only", "credential_value_included": False,
        "provider_mutation_performed": False,
    }
    if any(manifest.get(key) != value or type(manifest.get(key)) is not type(value)
           for key, value in expected.items()):
        raise ValueError("g1_team_bundle_manifest_binding_invalid")


def _verify_archive_paths(archive: zipfile.ZipFile) -> None:
    names = set()
    for info in archive.infolist():
        name = info.filename.rstrip("/") if info.is_dir() else info.filename
        path = PurePosixPath(name)
        mode = stat.S_IFMT(info.external_attr >> 16)
        if (
            not name or "\\" in name or "\x00" in name or path.is_absolute()
            or ".." in path.parts or path.as_posix() != name or name in names
            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            or (mode == stat.S_IFDIR and not info.is_dir())
            or (info.is_dir() and not info.filename.startswith("provider_runtime/publisher-source/source/"))
        ):
            raise ValueError("g1_team_bundle_archive_path_invalid")
        names.add(name)


def build_g1_team_provider_bundle(
    *, job_dir: Path, execution_packet_path: Path, authority_arguments: Mapping[str, Any],
    scene_packet_root: Path, publisher_source: Path, runtime_source_receipt: Path,
    sonic_asset_dir: Path, expected_implementation_commit: str,
) -> dict[str, Any]:
    """Create one immutable selected-policy transport; never allocate compute."""

    packet = _execution_packet(execution_packet_path, expected_implementation_commit)
    _authority_matches(packet, verify_g1_team_policy_authority(**authority_arguments))
    scene, scene_receipt, scene_rows = verify_g1_team_scene_packet(scene_packet_root, packet)
    publisher = verify_g1_publisher_source(publisher_source)
    runtime = verify_native_task_runtime_source_packet(runtime_source_receipt)
    if runtime.get("runtime_profile") != "unitree_g1" or runtime.get("redistribution_permitted") is not True:
        raise ValueError("g1_team_bundle_runtime_source_invalid")
    dependency_policy = _REPOSITORY / "docs/runtime_dependency_license_policy.json"
    review = _review_g1_runtime_wheels(runtime, dependency_policy)
    sonic = _sonic_assets(sonic_asset_dir, _SONIC_INVENTORY)
    job = Path(job_dir)
    if not job.is_absolute() or job.exists() or job.is_symlink() or not job.parent.is_dir():
        raise ValueError("g1_team_bundle_output_invalid")
    sources = [(SCENE_RELATIVE_ROOT + "/" + row["relative_path"], scene / row["relative_path"])
               for row in scene_rows]
    source_root = publisher_source / "source"
    sources += [("provider_runtime/publisher-source/source/" + path.relative_to(source_root).as_posix(), path)
                for path in _files(source_root) if path.relative_to(source_root).as_posix() != ".git/config"]
    package = Path(__file__).resolve().parent
    sources += [("provider_runtime/blueprint_pipeline/" + path.relative_to(package).as_posix(), path)
                for path in _runtime_code_files(package)]
    dependency, dependency_sources = _contract_dependency()
    sources += [("provider_runtime/" + relative, path) for relative, path in dependency_sources]
    sources += [("provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json", runtime_source_receipt),
                ("provider_runtime/inputs/rights/runtime_dependency_license_policy.json", dependency_policy),
                ("configs/" + _SONIC_INVENTORY.name, _SONIC_INVENTORY),
                ("configs/g1_humanoidarena_checkpoint_inventory.v1.json",
                 _REPOSITORY / "configs/g1_humanoidarena_checkpoint_inventory.v1.json")]
    sources += [("provider_runtime/inputs/sonic/" + Path(row["path"]).name, Path(row["path"])) for row in sonic]
    # Git must see an origin for publisher verification, but host credential
    # helpers, extraheaders, includes and file modes must never cross this seam.
    generated = {
        PACKET_RELATIVE_PATH: json.dumps(packet, sort_keys=True, indent=2) + "\n",
        "provider_runtime/publisher-source/source/.git/config":
            "[core]\n repositoryformatversion = 0\n bare = false\n filemode = false\n"
            + '[remote "origin"]\n url = ' + SOURCE_REPOSITORY + "\n",
        ENTRYPOINT: _entrypoint(),
    }
    artifacts = [{"relative_path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
                 for relative, path in sources]
    artifacts += [{"relative_path": relative, "sha256": "sha256:" + hashlib.sha256(value.encode()).hexdigest(),
                   "size_bytes": len(value.encode())} for relative, value in generated.items()]
    mode = packet["delivery_mode"]
    manifest = {
        "schema_version": SCHEMA, "status": "sealed_not_admitted",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND, "container_image": NATIVE_TASK_ARENA_IMAGE,
        "implementation_commit": expected_implementation_commit,
        "execution_packet_digest": packet["packet_digest"], "intent_id": packet["intent_id"],
        "policy_profile_digest": packet["policy_profile_digest"],
        "source_packet_receipt_digest": packet["source_packet_receipt_digest"],
        "operator_approval_digest": packet["operator_approval"]["approval_digest"],
        "scene_packet_receipt_digest": scene_receipt["receipt_digest"],
        "scene_plan_digest": scene_receipt["arena_scene_plan_digest"],
        "scene_id": packet["trusted_setup"]["scene_id"], "task_id": packet["trusted_setup"]["task_id"],
        "objective_id": packet["objective_id"], "delivery_mode": mode,
        "policy_runtime_required": mode != "authenticated_endpoint",
        "private_credential_staging_required": mode == "authenticated_endpoint",
        "publisher_source_receipt_digest": publisher["receipt_digest"],
        "publisher_source_identity": {key: value for key, value in publisher.items()
                                      if key not in {"source_root", "receipt_digest"}},
        "sonic_assets": [{key: value for key, value in row.items() if key != "path"} for row in sonic],
        "runtime_source_packet": {
            "runtime_profile": runtime["runtime_profile"], "receipt_digest": runtime["receipt_digest"],
            "packet_sha256": runtime["packet_sha256"], "packet_size_bytes": runtime["packet_size_bytes"],
            "packet_path": runtime["verified_packet_path"], "embedded_in_provider_bundle": False,
            "transport": "content_addressed_external_layer.v1",
        },
        "runtime_dependency_license_review": review,
        "contract_python_dependencies": [dependency], "expected_output_filename": RESULT_FILENAME,
        "runtime_entrypoint": ENTRYPOINT, "claim_ceiling": "development_only",
        "credential_value_included": False, "provider_mutation_performed": False,
        "artifacts": sorted(artifacts, key=lambda row: row["relative_path"]),
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    job.mkdir(mode=0o700)
    path = job / "native_g1_team_provider_bundle.zip"
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for directory in sorted(source_root.rglob("*")):
            if directory.is_dir():
                archive.writestr("provider_runtime/publisher-source/source/" + directory.relative_to(source_root).as_posix() + "/", b"")
        for relative, source in sources:
            _write_zip_file(archive, source=source, archive_path=relative)
        for relative, value in generated.items():
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | (0o755 if relative == ENTRYPOINT else 0o600)) << 16
            archive.writestr(info, value)
        archive.writestr(MANIFEST, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    receipt = {**manifest, "bundle_path": str(path), "bundle_size_bytes": path.stat().st_size,
               "bundle_sha256": _sha256(path), "receipt_path": str(job / (SCHEMA + ".json"))}
    Path(receipt["receipt_path"]).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return load_verified_g1_team_provider_bundle(
        Path(receipt["receipt_path"]), expected_implementation_commit=expected_implementation_commit,
        authority_arguments=authority_arguments,
    )


def load_verified_g1_team_provider_bundle(
    receipt_path: Path, *, expected_implementation_commit: str, authority_arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen current authority and exact dry-run bytes before admission."""

    receipt = _json(receipt_path)
    path = Path(str(receipt.get("bundle_path") or ""))
    manifest = {key: value for key, value in receipt.items() if key not in _RECEIPT_ONLY_FIELDS}
    if (
        not path.is_absolute() or path.is_symlink() or not path.is_file()
        or receipt.get("bundle_size_bytes") != path.stat().st_size
        or receipt.get("bundle_sha256") != _sha256(path)
        or manifest.get("schema_version") != SCHEMA
        or manifest.get("implementation_commit") != expected_implementation_commit
        or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise ValueError("g1_team_bundle_bytes_invalid")
    with zipfile.ZipFile(path) as archive:
        _verify_archive_paths(archive)
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(set(names)) != len(names) or json.loads(archive.read(MANIFEST)) != manifest:
            raise ValueError("g1_team_bundle_manifest_invalid")
        _verify_artifacts(manifest, archive.open, set(names))
        packet = json.loads(archive.read(PACKET_RELATIVE_PATH))
        source = json.loads(archive.read("provider_runtime/native_task_runtime_sources/native_task_runtime_source_packet.v1.json"))
    if (
        not isinstance(packet, dict) or set(packet) != PACKET_FIELDS
        or packet.get("schema_version") != PACKET_SCHEMA
        or packet.get("status") != "approved_input_not_executed"
        or packet.get("implementation_commit") != expected_implementation_commit
        or packet.get("claim_ceiling") != "development_only"
        or packet.get("credential_value_included") is not False
        or packet.get("artifact_bytes_included") is not False
        or packet.get("provider_mutation_performed") is not False
        or packet.get("packet_digest") != cross_runtime_canonical_digest(packet, digest_field="packet_digest")
    ):
        raise ValueError("g1_team_bundle_packet_binding_invalid")
    _authority_matches(packet, verify_g1_team_policy_authority(**authority_arguments))
    verify_g1_team_manifest_binding(manifest, packet)
    layer = manifest["runtime_source_packet"]
    runtime_path = Path(str(layer.get("packet_path") or ""))
    if (
        layer.get("runtime_profile") != "unitree_g1"
        or layer.get("embedded_in_provider_bundle") is not False
        or layer.get("transport") != "content_addressed_external_layer.v1"
        or layer.get("receipt_digest") != source.get("receipt_digest")
        or layer.get("packet_sha256") != source.get("packet_sha256")
        or layer.get("packet_size_bytes") != source.get("packet_size_bytes")
        or not runtime_path.is_absolute() or runtime_path.is_symlink() or not runtime_path.is_file()
        or runtime_path.stat().st_size != layer.get("packet_size_bytes")
        or _sha256(runtime_path) != layer.get("packet_sha256")
    ):
        raise ValueError("g1_team_bundle_external_runtime_bytes_invalid")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("job-dir", "execution-packet", "intent", "registry", "approval",
                 "scene-packet-root", "publisher-source", "runtime-source-receipt", "sonic-asset-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--trusted-client", action="append", required=True)
    args = parser.parse_args(argv)
    receipt = build_g1_team_provider_bundle(
        job_dir=args.job_dir, execution_packet_path=args.execution_packet,
        authority_arguments={"intent_path": args.intent, "registry_path": args.registry,
                             "approval_path": args.approval, "trusted_clients": set(args.trusted_client)},
        scene_packet_root=args.scene_packet_root, publisher_source=args.publisher_source,
        runtime_source_receipt=args.runtime_source_receipt, sonic_asset_dir=args.sonic_asset_dir,
        expected_implementation_commit=args.implementation_commit,
    )
    print(json.dumps({"status": receipt["status"], "bundle_sha256": receipt["bundle_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
