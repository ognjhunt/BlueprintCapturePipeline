"""Continue verified CPU stages with the existing native driver and stage chain.

The capsule retains original logical paths and checkpoints. No authoring tool,
model credential, generated code runner, or model runtime is installed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

from .decision_evidence_contracts import canonical_digest, canonical_json


SCHEMA = "scene_configuration_native_continuation.v1"
BUNDLE_KIND = "scene_configuration_native_continuation"
MANIFEST = "native_continuation_manifest.v1.json"
CAPSULE = "completed_cpu_prefix.zip"
OUTPUT = "adp009d_native_microcheck.json"
MINIMUM_DRIVER_VERSION = "595.58.03"
MAXIMUM_DRIVER_EXCLUSIVE = "596.0.0"
INPUT_SCHEMA = "scene_configuration_native_continuation_inputs.v1"


def add_allocator_arguments(parser):
    parser.add_argument("--adp009d-native-continuation-inputs",
        help="Admitted retained CPU prefix, envelope and configurations for native stages 5-6 only.")


def allocator_mode_conflict(args):
    return (not args.adp009d_diagnostic_only or args.adp009d_policy_candidate or args.adp009d_controls
            or args.adp009d_policy_runtime_smoke or args.adp009d_articulated_diagnostic_asset
            or args.adp009d_articulated_diagnostic_request or args.adp009d_articulated_diagnostic_static_receipt
            or args.adp009d_physics_backend != "physx" or args.adp009d_authorize_gated_backbone
            or args.adp009d_scenario_instance or args.adp009d_approved_can or args.adp009d_sage_collision)


def _require(condition, reason):
    if not condition:
        raise ValueError("native_continuation_" + reason)


def _json(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "record_missing")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    return value


def _sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def _record(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "input_missing")
    return {"path": str(path.resolve()), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _binding_matches(binding, envelope, configurations=None):
    _require(binding.get("binding_digest") == canonical_digest(binding, digest_field="binding_digest")
             and binding.get("envelope_digest") == canonical_digest(envelope)
             and binding.get("run_id") == envelope["run_id"], "binding_changed")
    if configurations is not None:
        _require(binding.get("configuration_digests") == {
            name: _sha(path) for name, (_, path) in configurations.items()}, "configuration_changed")


def _prefix_binding(output):
    paths = [output / "stages" / name for name in (
        "artisan_same_run_resume_binding.json", "astra_same_run_resume_binding.json")]
    found = [path for path in paths if path.is_file()]
    _require(len(found) == 1, "binding_missing")
    return _json(found[0])


def share_completed_prefix_inputs(*, output_root, envelope, configurations, content_addressed_root):
    """Seal closed input copies and reuse their verified, read-only CAS inode.

    Caller owns the run's exclusive controller lock and has verified no writer
    remains. This is immutable artifact materialization, not writable-file GC.
    """
    from .task_evaluation_astra_stage_resume import load_completed_stage
    output, cas = Path(output_root).resolve(), Path(content_addressed_root).resolve()
    binding = _prefix_binding(output)
    _binding_matches(binding, envelope, configurations)
    completed, plan = [], []
    for stage in envelope["recipe"]["stage_sequence"][:4]:
        result = load_completed_stage(output / "stages" / stage["stage_id"], stage, binding, completed)
        _require(result is not None, "cpu_prefix_incomplete")
        completed.append(result)
        for record in result["output_artifacts"]:
            target = Path(record["path"])
            donor = cas / record["digest"].removeprefix("sha256:")
            if not donor.is_file() or donor.is_symlink() or donor.samefile(target):
                continue
            left, right = donor.stat(), target.stat()
            _require(donor.resolve() == donor and _sha(donor) == record["digest"]
                     and left.st_size == record["size_bytes"] and left.st_mode & 0o222 == 0
                     and (left.st_dev, left.st_uid, left.st_gid) == (right.st_dev, right.st_uid, right.st_gid)
                     and right.st_nlink == 1, "immutable_input_reuse_invalid")
            plan.append((donor, target, record, right.st_mode & 0o777))
    for donor, target, record, old_mode in plan:
        temporary = target.with_name("." + target.name + ".immutable-reuse")
        _require(not temporary.exists() and _sha(target) == record["digest"], "immutable_input_reuse_changed")
        os.link(donor, temporary, follow_symlinks=False)
        os.replace(temporary, target)
        _require(_sha(target) == record["digest"], "immutable_input_reuse_changed")
    return {"schema_version": "completed_prefix_immutable_input_reuse.v1", "paths_preserved": True,
        "logical_bytes_preserved": True, "changes": [{"path": str(target), "donor": str(donor),
            "sha256": record["digest"], "size_bytes": record["size_bytes"], "old_mode": old_mode,
            "sealed_mode": target.stat().st_mode & 0o777} for donor, target, record, old_mode in plan]}


def seal_existing_native_prefix(*, output_root, envelope, configurations, capsule_path):
    """Verify and archive real completed stages, without running them again."""
    from .task_evaluation_astra_stage_resume import load_completed_stage
    from .task_evaluation_scene_configuration_cpu_prestage import TRANSPORT_SCHEMA, validate_stage_prefix_capsule
    from .task_evaluation_scene_configuration_output_archive import write_output_archive
    output = Path(output_root).resolve()
    stages = envelope["recipe"]["stage_sequence"]
    _require(len(stages) == 6 and stages[4]["adapter"]["id"] == "simready_native_import_qualification"
             and stages[5]["adapter"]["id"] == "native_task_scene_assembly", "remaining_stages_invalid")
    # This is read-only validation of an existing phase, never a new deadline.
    binding = _prefix_binding(output)
    _binding_matches(binding, envelope, configurations)
    _require(Path(binding["output_root"]) == output / "stages", "output_root_changed")
    completed = []
    for stage in stages[:4]:
        row = load_completed_stage(output / "stages" / stage["stage_id"], stage, binding, completed)
        _require(row is not None, "cpu_prefix_incomplete")
        completed.append(row)
    capsule = Path(capsule_path)
    _require(not capsule.exists(), "capsule_already_exists")
    ids = [row["stage_id"] for row in completed]
    write_output_archive(output, capsule, completed_stages=ids)
    transport = {"schema_version": TRANSPORT_SCHEMA, "output_root": str(output),
        "run_id": envelope["run_id"], "construction_source_commit": envelope["expected_production_commit"],
        "completed_stage_ids": ids, "stage_limit": ids[-1], "prefix_binding_digest": binding["binding_digest"],
        "prefix_deadline_epoch": binding["parent_deadline_epoch"],
        "source_envelope_digest": envelope["envelope_digest"],
        "execution_site": "control_plane", "gpu_execution_performed": False,
        "new_paid_allocation_authorized": False, "source": "verified_existing_completed_stage_checkpoints"}
    transport["transport_digest"] = canonical_digest(transport, digest_field="transport_digest")
    with zipfile.ZipFile(capsule, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cpu_prestage_transport.json", canonical_json(transport) + "\n")
    validate_stage_prefix_capsule(capsule, expected_stage_ids=ids)
    capsule.chmod(0o440)
    return {**_record(capsule), "transport": transport}


def validate_native_prefix(capsule, envelope):
    """Check the exact checkpoint chain and artifact bytes inside its archive."""
    from .task_evaluation_astra_stage_resume import _checkpoint
    from .task_evaluation_scene_configuration_cpu_prestage import validate_stage_prefix_capsule
    stages = envelope["recipe"]["stage_sequence"]
    _require(len(stages) == 6 and stages[4]["adapter"]["id"] == "simready_native_import_qualification"
             and stages[5]["adapter"]["id"] == "native_task_scene_assembly", "remaining_stages_invalid")
    marker = validate_stage_prefix_capsule(Path(capsule), expected_stage_ids=[row["stage_id"] for row in stages[:4]])
    transport = marker["transport"]
    _require(transport["run_id"] == envelope["run_id"] and
             transport["source_envelope_digest"] == envelope["envelope_digest"], "capsule_envelope_changed")
    with zipfile.ZipFile(capsule) as archive:
        names = archive.namelist()
        candidates = [name for name in ("stages/artisan_same_run_resume_binding.json",
                      "stages/astra_same_run_resume_binding.json") if name in names]
        _require(len(candidates) == 1, "binding_missing")
        binding = json.loads(archive.read(candidates[0]))
        _binding_matches(binding, envelope)
        _require(binding["binding_digest"] == transport["prefix_binding_digest"]
                 and Path(binding["output_root"]) == Path(transport["output_root"]) / "stages", "binding_changed")
        _require(binding["parent_deadline_epoch"] == transport["prefix_deadline_epoch"], "deadline_changed")
        previous = []
        for stage in stages[:4]:
            saved = json.loads(archive.read(f"stages/{stage['stage_id']}/completed_stage_checkpoint.json"))
            result = saved.get("stage_result") or {}
            _require(saved == _checkpoint(stage, binding, previous, result)
                     and result.get("schema_version") == "task_evaluation_scene_configuration_stage_result.v1"
                     and result.get("status") == "completed"
                     and result.get("stage_id") == stage["stage_id"]
                     and result.get("configuration_digest") == binding["configuration_digests"][stage["stage_id"]]
                     and result.get("provider_mutations_performed") == 0
                     and result.get("paid_execution_requested") is False
                     and result.get("executed_inside_parent_configuration_run") is True
                     and result.get("canonical_allocator") is None and result.get("retry_cap") == 0
                     and result.get("diagnostic_only") is not True
                     and result.get("qualification_eligible") is not False
                     and result.get("stage_result_digest") == canonical_digest(result, digest_field="stage_result_digest"),
                     "checkpoint_changed")
            for record in result["output_artifacts"]:
                relative = Path(record["path"]).relative_to(Path(binding["output_root"]) / stage["stage_id"])
                member = f"stages/{stage['stage_id']}/{relative.as_posix()}"
                info = archive.getinfo(member)
                with archive.open(member) as stream:
                    digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
                _require(info.file_size == record["size_bytes"] and digest == record["digest"], "prefix_artifact_changed")
            previous.append(result)
    return transport


def _validated_configurations(runtime, manifest, envelope):
    records = manifest["configurations"]
    _require(set(records) == {row["stage_id"] for row in envelope["recipe"]["stage_sequence"]}, "configuration_set_changed")
    with zipfile.ZipFile(runtime / CAPSULE) as archive:
        candidates = [name for name in archive.namelist() if name in {
            "stages/artisan_same_run_resume_binding.json", "stages/astra_same_run_resume_binding.json"}]
        binding = json.loads(archive.read(candidates[0]))
    for name, record in records.items():
        _require(record["member"] == f"configuration/{name}.json", "configuration_member_changed")
        path = runtime / record["member"]
        logical = Path(record["path"])
        _require(logical.is_absolute() and ".." not in logical.parts, "logical_configuration_path_invalid")
        _require(_sha(path) == record["sha256"] == binding["configuration_digests"][name]
                 and path.stat().st_size == record["size_bytes"], "configuration_changed")
    return records


def native_import_producer_registry(*, implementation_commit):
    """One fixed, repository-owned native producer; no generated entrypoint."""
    from .task_evaluation_scene_configuration_stage_producers import (
        ADMITTED_PRODUCER_IDENTITIES, PRODUCTION_RESULT_SCHEMA_VERSION,
        SceneConfigurationStageProducerRegistry,
    )
    identities = [row for row in ADMITTED_PRODUCER_IDENTITIES if row.adapter_id == "simready_native_import_qualification"]
    _require(len(identities) == 1 and re.fullmatch(r"[0-9a-f]{40}", implementation_commit), "producer_identity_invalid")
    identity = identities[0]

    def produce(*, stage, envelope, configuration, configuration_path, dependency_results, output_root):
        from .task_evaluation_scene_configuration_native_import_driver import execute_native_import_component
        from .task_evaluation_scene_configuration_stage_tool import _validate_input, _validate_dependencies, _validate_component_result
        root = Path(output_root)
        source_binding = canonical_digest({"implementation_commit": implementation_commit,
            "runtime": "repository_owned_native_import", "generated_tool_runtime_used": False})
        value = {"schema_version": "task_evaluation_scene_configuration_stage_production_input.v1",
            "run_id": envelope["run_id"], "stage": stage, "configuration": configuration,
            "configuration_sha256": _sha(configuration_path), "source_commit": implementation_commit,
            "construction_source_commit": envelope["expected_production_commit"], "execution_mode": "diagnostic_only",
            "toolchain_digest": source_binding, "construction_envelope": envelope}
        _validate_input(value, adapter_id=identity.adapter_id, diagnostic_only=True)
        _validate_dependencies(list(dependency_results))
        input_path, dependencies = root / "stage_production_input.v1.json", root / "dependency_results.v1.json"
        input_path.write_text(canonical_json(value) + "\n")
        dependencies.write_text(canonical_json(list(dependency_results)) + "\n")
        component = execute_native_import_component(environment={
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(input_path),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(dependencies),
            "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(root),
            "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(root / "component_result.v1.json")})
        artifacts = _validate_component_result(component, adapter_id=identity.adapter_id,
            stage_id=stage["stage_id"], output_root=root, diagnostic_only=False)
        result = {"schema_version": PRODUCTION_RESULT_SCHEMA_VERSION, "status": "completed",
            "stage_id": stage["stage_id"], "capability": stage["capability"], "adapter_id": identity.adapter_id,
            "source_commit": implementation_commit, "construction_source_commit": envelope["expected_production_commit"],
            "toolchain_digest": source_binding, "runtime_kind": "builtin_native_after_completed_cpu_prefix",
            "provider_mutations_performed": 0, "paid_execution_requested": False,
            "executed_inside_parent_configuration_run": True, "artifacts": artifacts}
        result["production_result_digest"] = canonical_digest(result, digest_field="production_result_digest")
        (root / f"{PRODUCTION_RESULT_SCHEMA_VERSION}.json").write_text(canonical_json(result) + "\n")
        return result

    return SceneConfigurationStageProducerRegistry({identity: produce})


ENTRYPOINT = """#!/usr/bin/env bash
set -eo pipefail
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$RUNTIME_DIR/pipeline/src"
export PYTHONDONTWRITEBYTECODE=1
OUT_DIR="$RUNTIME_DIR/../runtime_output"
if [ -n "$BLUEPRINT_ADP_ARENA_OUTPUT_DIR" ]; then OUT_DIR="$BLUEPRINT_ADP_ARENA_OUTPUT_DIR"; fi
if [ "$BLUEPRINT_PROVIDER_BUNDLE_REHEARSAL" = "1" ]; then
  exec python3 -m blueprint_pipeline.task_evaluation_scene_configuration_native_continuation --runtime-root "$RUNTIME_DIR" --output-root "$OUT_DIR" --preflight-only
fi
set +e
/isaac-sim/python.sh -m blueprint_pipeline.task_evaluation_scene_configuration_native_continuation --runtime-root "$RUNTIME_DIR" --output-root "$OUT_DIR"
runtime_rc=$?
if [ ! -f "$OUT_DIR/adp009d_native_microcheck.json" ]; then
  python3 - "$OUT_DIR" "$runtime_rc" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
p.mkdir(parents=True, exist_ok=True)
(p / "adp009d_native_microcheck.json").write_text(json.dumps({
    "schema_version": "scene_configuration_native_continuation_failure.v1",
    "status": "failed", "blockers": ["native_continuation_process_exited_without_result:" + sys.argv[2]],
    "candidate_policy_queried": False, "claim_ceiling": "development_only",
    "provider_zero_required_after_return": True,
}) + "\\n")
PY
fi
exit "$runtime_rc"
"""


def _source_root(commit):
    root = Path(__file__).resolve().parents[2]
    git = ["git", "-c", "safe.directory=" + str(root)]
    observed = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output([*git, "status", "--porcelain"], cwd=root, text=True).strip()
    _require(observed == commit and not dirty, "source_not_clean_exact_commit")
    return root


def build_native_continuation_bundle(*, job_dir, inputs_path, implementation_commit):
    from .public_scene_simready_isaac_bundle import DEFAULT_IMAGE
    from .provider_bundle_rehearsal import rehearse_provider_bundle_entrypoint
    source = _source_root(implementation_commit)
    inputs = _json(inputs_path)
    _require(inputs.get("schema_version") == INPUT_SCHEMA
             and inputs.get("provider_disclosure_authorized") is True
             and inputs.get("claim_ceiling") == "development_only", "inputs_not_admitted")
    capsule_path, envelope_path = inputs["capsule_path"], inputs["envelope_path"]
    configuration_paths = inputs["configuration_paths"]
    envelope = _json(envelope_path)
    _require(envelope["envelope_digest"] == canonical_digest(envelope, digest_field="envelope_digest"), "envelope_changed")
    transport = validate_native_prefix(capsule_path, envelope)
    job = Path(job_dir).resolve()
    if job.exists():
        prior = _json(job / "native_continuation_bundle_receipt.json")
        _require(prior["implementation_commit"] == implementation_commit
                 and prior["operator_inputs"] == _record(inputs_path)
                 and prior["bundle_sha256"] == _sha(prior["bundle_path"])
                 and prior["prefix"]["sha256"] == _sha(capsule_path), "retained_bundle_changed")
        return prior
    runtime = job / "provider_runtime"
    runtime.mkdir(parents=True)
    # A sealed read-only capsule may share storage with its staging alias.
    # Writable caller input is copied, preserving independent file identities.
    capsule = Path(capsule_path)
    if capsule.stat().st_mode & 0o222 == 0:
        try:
            os.link(capsule, runtime / CAPSULE, follow_symlinks=False)
        except OSError:
            shutil.copyfile(capsule, runtime / CAPSULE)
    else:
        shutil.copyfile(capsule, runtime / CAPSULE)
    (runtime / "construction_envelope.v1.json").write_text(canonical_json(envelope) + "\n")
    configs = {}
    for stage in envelope["recipe"]["stage_sequence"]:
        name = stage["stage_id"]
        path = Path(configuration_paths[name]).resolve()
        member = f"configuration/{name}.json"
        target = runtime / member
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        configs[name] = {**_record(path), "member": member}
    package = source / "src/blueprint_pipeline"
    for path in package.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            _require(not path.is_symlink(), "source_symlink")
            target = runtime / "pipeline/src/blueprint_pipeline" / path.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    manifest = {"schema_version": SCHEMA, "diagnostic_kind": BUNDLE_KIND, "implementation_commit": implementation_commit,
        "container_image": DEFAULT_IMAGE, "minimum_driver_version": MINIMUM_DRIVER_VERSION,
        "maximum_driver_exclusive": MAXIMUM_DRIVER_EXCLUSIVE,
        "construction_envelope_digest": envelope["envelope_digest"], "run_id": envelope["run_id"],
        "prefix": {**_record(Path(capsule_path)), "member": CAPSULE}, "prefix_transport": transport,
        "configurations": configs, "candidate_policy_queried": False, "policy_candidate_id": None,
        "model_calls_performed": 0, "generated_tool_runtime_used": False,
        "private_data_uploaded": True, "captured_frame_derivatives_in_bundle": True,
        "provider_disclosure_authorized": True, "operator_inputs": _record(inputs_path),
        "qualification_eligible": False, "configured_revision_publication_permitted": False,
        "claim_ceiling": "development_only", "provider_zero_required_after_return": True}
    manifest["input_digest"] = canonical_digest(manifest, digest_field="input_digest")
    _validated_configurations(runtime, manifest, envelope)
    (runtime / MANIFEST).write_text(canonical_json(manifest) + "\n")
    entrypoint = runtime / "run_adp_arena_provider_runtime.sh"
    entrypoint.write_text(ENTRYPOINT)
    entrypoint.chmod(0o750)
    archive_path = job / "native_continuation_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sorted(runtime.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(job).as_posix(),
                    compress_type=zipfile.ZIP_STORED if path.name == CAPSULE else zipfile.ZIP_DEFLATED)
    rehearsal = rehearse_provider_bundle_entrypoint(bundle_path=archive_path,
        entrypoint_relative_path="provider_runtime/run_adp_arena_provider_runtime.sh",
        evidence_path=job / "native_continuation_rehearsal.json")
    receipt = {**manifest, "status": "ready", "bundle_path": str(archive_path),
        "bundle_sha256": _sha(archive_path), "bundle_size_bytes": archive_path.stat().st_size,
        "expected_output_filename": OUTPUT, "exact_bundle_entrypoint_rehearsal": rehearsal}
    (job / "native_continuation_bundle_receipt.json").write_text(canonical_json(receipt) + "\n")
    # The archive and rehearsal are the retained result. This private staging
    # copy is completely reproducible and can otherwise double the disk peak.
    shutil.rmtree(runtime)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    runtime, output = Path(args.runtime_root).resolve(), Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = _json(runtime / MANIFEST)
    envelope = _json(runtime / "construction_envelope.v1.json")
    _require(manifest["input_digest"] == canonical_digest(manifest, digest_field="input_digest")
             and envelope["envelope_digest"] == canonical_digest(envelope, digest_field="envelope_digest")
             and envelope["envelope_digest"] == manifest["construction_envelope_digest"], "manifest_changed")
    capsule = runtime / CAPSULE
    _require(_sha(capsule) == manifest["prefix"]["sha256"]
             and capsule.stat().st_size == manifest["prefix"]["size_bytes"], "capsule_changed")
    transport = validate_native_prefix(capsule, envelope)
    _validated_configurations(runtime, manifest, envelope)
    if args.preflight_only:
        # Import the actual driver closure now; SimulationApp remains lazy.
        from .task_evaluation_scene_configuration_native_import_driver import execute_native_import_component
        _require(callable(execute_native_import_component), "native_driver_missing")
        value = {"schema_version": "provider_bundle_entrypoint_rehearsal.v1", "status": "passed",
            "archive_extraction_executed": True, "gpu_runtime_started": False,
            "paid_inference_performed": False, "provider_mutations_performed": 0,
            "stopped_before": "isaac_sim_startup", "completed_stage_ids": transport["completed_stage_ids"]}
        (output / "provider_bundle_rehearsal.json").write_text(canonical_json(value) + "\n")
        return 0
    from .task_evaluation_scene_configuration_cpu_prestage import (
        PREFIX_URL_ENV, PREFIX_SHA_ENV, PREFIX_BYTES_ENV, consume_stage_prefix_capsule,
    )
    from .task_evaluation_scene_configuration_provider_runtime import execute_scene_configuration_stage_chain
    logical_output = Path(transport["output_root"])
    logical_output.mkdir(parents=True, exist_ok=True)
    _require(not any(logical_output.iterdir()), "restore_target_not_empty")
    consume_stage_prefix_capsule(environment={PREFIX_URL_ENV: "local-sealed-capsule",
        PREFIX_SHA_ENV: manifest["prefix"]["sha256"], PREFIX_BYTES_ENV: str(capsule.stat().st_size)},
        output_root=logical_output, expected_run_id=envelope["run_id"],
        opener=lambda url, timeout: capsule.open("rb"))
    configurations = {}
    for name, record in manifest["configurations"].items():
        target = Path(record["path"])
        _require(target.is_absolute() and target.resolve() == target and not target.is_symlink(), "logical_configuration_path_invalid")
        target.parent.mkdir(parents=True, exist_ok=True)
        _require(not target.exists() or _sha(target) == record["sha256"], "logical_configuration_conflict")
        if not target.exists():
            shutil.copyfile(runtime / record["member"], target)
        configurations[name] = (_json(target), target)
    result = {"schema_version": "scene_configuration_native_continuation_result.v1", "status": "failed",
        "implementation_commit": manifest["implementation_commit"], "candidate_policy_queried": False,
        "claim_ceiling": "development_only", "qualification_eligible": False, "provider_zero_required_after_return": True}
    try:
        _require(transport["prefix_deadline_epoch"] > time.time(), "original_parent_deadline_expired")
        chain = execute_scene_configuration_stage_chain(envelope=envelope, configurations=configurations,
            output_root=logical_output / "stages", parent_deadline_epoch=transport["prefix_deadline_epoch"],
            producer_registry=native_import_producer_registry(implementation_commit=manifest["implementation_commit"]))
        result.update(status="completed", stage_chain=chain)
    except Exception as exc:
        result.update(blockers=[type(exc).__name__ + ":" + str(exc)[:500]])
        raise
    finally:
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        (output / OUTPUT).write_text(canonical_json(result) + "\n")
        # The verified stages 1-4 remain retained at the controller. Return only
        # new stage outputs, not another copy of their immutable 900MB prefix.
        with zipfile.ZipFile(output / "native_continuation_stage_output.zip", "w",
                             compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for stage in envelope["recipe"]["stage_sequence"][4:]:
                for path in sorted((logical_output / "stages" / stage["stage_id"]).rglob("*")):
                    _require(not path.is_symlink(), "output_symlink")
                    if path.is_file():
                        archive.write(path, path.relative_to(logical_output).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
