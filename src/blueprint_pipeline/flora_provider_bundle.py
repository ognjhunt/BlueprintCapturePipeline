"""Immutable input/source packaging for the bounded FLORA diagnostic lane."""

from __future__ import annotations

import hashlib
import argparse
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .common import write_json
from .provider_runtime_bundle_contract import provider_runtime_contract_blockers

MODEL_ID = "nvidia/Flora-Rewarder"
MODEL_REVISION = "7de68d22c4a9decec5ddb7f86ddcaee8f82d3367"
CLAIM_CLASS = "development_only_episode_diagnostic"
BUNDLE_SCHEMA = "flora_provider_bundle.v1"


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return "sha256:" + h.hexdigest()


def read_object(path: str | Path) -> dict[str, Any]:
    result = json.loads(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("flora_json_object_required")
    return result


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or any(p in {".", ".."} for p in path.parts):
        raise ValueError("flora_bundle_relative_path_invalid")
    return path.as_posix()


def validate_flora_benchmark_inputs(root: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen source-bound references, media and exact external rights before spend."""
    from .flora_episode_benchmark import run_flora_benchmark
    profile = read_object(root / _relative(inputs["profile_relative_path"]))
    config = profile.get("runtime_config") or {}
    if (profile.get("execution_site") != "external_provider" or profile.get("compute_provider_id") != "vast"
            or config.get("model_path") != "/opt/blueprint/flora/model"
            or config.get("source_root") != "/opt/blueprint/flora/source"
            or config.get("tensor_parallel_size") != 2):
        raise ValueError("flora_provider_profile_not_portable_dual_gpu")
    with tempfile.TemporaryDirectory(prefix="flora-input-preflight-") as output:
        preflight = run_flora_benchmark(manifest_path=root / _relative(inputs["manifest_relative_path"]),
            profile=profile, rights_root=root / _relative(inputs["rights_relative_path"]),
            output_dir=Path(output), execute=False)
    if (preflight.get("status") != "inputs_verified" or preflight.get("eligible_episode_count", 0) <= 0
            or preflight.get("eligible_episode_count") != preflight.get("selected_episode_count")):
        raise ValueError("flora_benchmark_inputs_or_rights_blocked:" + json.dumps(preflight.get("blockers", []), sort_keys=True))
    return preflight


RUN_SCRIPT = '''#!/usr/bin/env bash
set -u
RUNTIME_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUTPUT_DIR=${BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR:-"$RUNTIME_DIR/../runtime_output"}
mkdir -p "$OUTPUT_DIR"
export BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR="$OUTPUT_DIR"
export BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR=${BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR:-"$RUNTIME_DIR/.."}
PYTHON_BIN=${BLUEPRINT_EVALUATOR_PROVIDER_PYTHON:-python3}
"$PYTHON_BIN" "$RUNTIME_DIR/evaluator_provider_runtime_runner.py"
runner_rc=$?
write_missing_result() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/evaluator_runtime_result.json" "$runner_rc" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":"flora_provider_runtime.v1", "status":"blocked", "blockers":["evaluator_runner_process_exited_without_runtime_result"], "blocked_evaluator_process_exited_without_result":True, "runner_exit_code":int(sys.argv[2]), "candidate_policy_queried":False, "claim_class":"development_only_episode_diagnostic"}))
PY
}
if [ ! -f "$OUTPUT_DIR/evaluator_runtime_result.json" ]; then write_missing_result; fi
exit "$runner_rc"
'''


def build_flora_provider_bundle(
    *, benchmark_root: str | Path, output_bundle: str | Path, source_commit: str,
    container_image: str, manifest_relative_path: str = "manifest.json",
    profile_relative_path: str = "profile.json", rights_relative_path: str = "rights",
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Package committed source and a portable, symlink-free benchmark tree."""
    root = Path(benchmark_root).resolve()
    output = Path(output_bundle).resolve()
    if output.is_relative_to(root):
        raise ValueError("flora_bundle_output_inside_input_root")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("flora_source_commit_invalid")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", container_image):
        raise ValueError("flora_container_image_must_be_digest_pinned")
    for item in (manifest_relative_path, profile_relative_path, rights_relative_path):
        if not (root / _relative(item)).exists():
            raise ValueError("flora_bundle_input_missing:" + item)
    repo = Path(__file__).resolve().parents[2]
    raw = subprocess.run(["git", "archive", source_commit, "src/blueprint_pipeline"], cwd=repo, check=True, stdout=subprocess.PIPE).stdout
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                members["provider_runtime/source/" + member.name] = stream.read()
    runner_name = "provider_runtime/source/src/blueprint_pipeline/flora_provider_runtime.py"
    if runner_name not in members:
        raise ValueError("flora_provider_runner_not_in_source_commit")
    members["provider_runtime/evaluator_provider_runtime_runner.py"] = members[runner_name]
    members["provider_runtime/run_evaluator_provider_runtime.sh"] = RUN_SCRIPT.encode()
    disk_members: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("flora_bundle_symlink_forbidden")
        if path.is_file():
            disk_members["benchmark/" + path.relative_to(root).as_posix()] = path
    inputs = {
        "schema_version": "flora_provider_input.v1", "manifest_relative_path": _relative(manifest_relative_path),
        "profile_relative_path": _relative(profile_relative_path), "rights_relative_path": _relative(rights_relative_path),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "claim_class": CLAIM_CLASS,
    }
    preflight = validate_flora_benchmark_inputs(root, inputs)
    members["provider_runtime/evaluator_input_manifest.json"] = json.dumps(inputs, sort_keys=True).encode()
    files = {name: "sha256:" + hashlib.sha256(data).hexdigest() for name, data in sorted(members.items())}
    files.update({name: file_digest(path) for name, path in disk_members.items()})
    manifest = {
        "schema_version": BUNDLE_SCHEMA, "source_commit": source_commit, "container_image": container_image,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "claim_class": CLAIM_CLASS,
        "files": files, "input_digest": digest(inputs), "local_bundle_ready_for_remote_staging": True,
        "blockers": [], "candidate_policy_queried": False,
        "input_preflight": preflight,
    }
    members["provider_runtime/evaluator_provider_runtime_manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            archive.writestr(info, data)
        for name, path in sorted(disk_members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with archive.open(info, "w", force_zip64=True) as target, path.open("rb") as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    receipt = {**manifest, "status": "ready", "provider_bundle_kind": "evaluator", "bundle_path": str(output),
               "bundle_sha256": file_digest(output), "bundle_size_bytes": output.stat().st_size,
               "retry_cap": 0, "maximum_provider_allocations": 1, "provider_zero_required_after_return": True}
    receipt["receipt_digest"] = digest(receipt)
    validate_flora_provider_bundle(receipt, bundle_path=output)
    write_json(Path(receipt_path) if receipt_path else output.with_suffix(".receipt.json"), receipt)
    return receipt


def validate_flora_provider_bundle(receipt: Mapping[str, Any], *, bundle_path: str | Path) -> dict[str, Any]:
    result = dict(receipt)
    path = Path(bundle_path).resolve()
    if (result.get("schema_version") != BUNDLE_SCHEMA or result.get("status") != "ready"
        or result.get("receipt_digest") != digest({k: v for k, v in result.items() if k != "receipt_digest"})
        or not path.is_file() or file_digest(path) != result.get("bundle_sha256")
        or result.get("model_id") != MODEL_ID or result.get("model_revision") != MODEL_REVISION
        or result.get("claim_class") != CLAIM_CLASS or result.get("provider_bundle_kind") != "evaluator"):
        raise ValueError("flora_bundle_receipt_binding_invalid")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or any(_relative(name) != name for name in names):
            raise ValueError("flora_bundle_members_invalid")
        manifest = json.loads(archive.read("provider_runtime/evaluator_provider_runtime_manifest.json"))
        files = manifest.get("files") or {}
        if files != result.get("files") or set(names) != set(files) | {"provider_runtime/evaluator_provider_runtime_manifest.json"}:
            raise ValueError("flora_bundle_manifest_binding_invalid")
        for key in ("source_commit", "container_image", "model_id", "model_revision", "input_digest"):
            if manifest.get(key) != result.get(key):
                raise ValueError("flora_bundle_manifest_binding_invalid")
        for name, expected in files.items():
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for part in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(part)
            if "sha256:" + h.hexdigest() != expected:
                raise ValueError("flora_bundle_member_digest_invalid:" + name)
        blockers = provider_runtime_contract_blockers(provider_bundle_kind="evaluator",
            entrypoint_text=archive.read("provider_runtime/run_evaluator_provider_runtime.sh").decode(),
            runner_text=archive.read("provider_runtime/evaluator_provider_runtime_runner.py").decode())
        if blockers:
            raise ValueError(";".join(blockers))
        inputs = json.loads(archive.read("provider_runtime/evaluator_input_manifest.json"))
        if digest(inputs) != result["input_digest"]:
            raise ValueError("flora_input_manifest_digest_invalid")
        with tempfile.TemporaryDirectory(prefix="flora-bundle-preflight-") as temporary:
            temporary_root = Path(temporary)
            for name in names:
                if name.startswith("benchmark/"):
                    target = temporary_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as source, target.open("xb") as dest:
                        shutil.copyfileobj(source, dest, length=1024 * 1024)
            preflight = validate_flora_benchmark_inputs(temporary_root / "benchmark", inputs)
            if preflight != manifest.get("input_preflight") or preflight != result.get("input_preflight"):
                raise ValueError("flora_input_preflight_binding_invalid")
    result["bundle_path"] = str(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    for name in ("benchmark-root", "output-bundle", "source-commit", "container-image"):
        build.add_argument("--" + name, required=True)
    build.add_argument("--manifest-relative-path", default="manifest.json")
    build.add_argument("--profile-relative-path", default="profile.json")
    build.add_argument("--rights-relative-path", default="rights")
    build.add_argument("--receipt-path")
    args = vars(parser.parse_args())
    args.pop("command")
    result = build_flora_provider_bundle(**args)
    print(json.dumps({key:result[key] for key in ("status", "bundle_path", "bundle_sha256", "receipt_digest")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
