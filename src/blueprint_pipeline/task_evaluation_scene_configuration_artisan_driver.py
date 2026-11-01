"""Thin ARTISAN stage adapter under existing rights and paid-parent authority."""
from __future__ import annotations

import fcntl
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import shutil
import time

from .decision_evidence_contracts import canonical_digest
from .task_object_authoring_backend import ARTISAN_BACKEND, frozen_authoring_cap
from .task_object_authoring_contracts import (
    require, read_json, write_json, file_record, source_frames,
    build_construction_request, verify_construction_request, artisan_prompt, verified_file,
)
from .task_object_artisan_exports import upstream_exports, adapt_usd
from .task_object_artisan_process import run_controller
from .task_evaluation_scene_configuration_artisan_runtime import validate_artisan_runtime
from .task_evaluation_scene_configuration_openai_gate import scene_configuration_openai_stage_gate, scene_configuration_openai_stage_scope
from .task_evaluation_scene_configuration_runtime_budget import STAGE_DEADLINE_EPOCH_ENV, PARENT_DEADLINE_EPOCH_ENV

INPUT = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"
DEPENDENCIES = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"
OUTPUT = "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"
RESULT = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"
PACKAGE = "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_ROOT"
# The confined import check before inference is a correctness check, not a
# speed gate. The sealed ARTISAN runtime ships no bytecode and its tree is
# read-only, so every import compiles from source; the release builder
# measured these cold imports past 60 s on the 4-core control plane under
# load (2026-10-13), the host the CPU prestage runs this stage on.
CONFINED_PREFLIGHT_TIMEOUT_SECONDS = 300


def _component_result(stage_input, request, paths):
    from .task_evaluation_scene_configuration_stage_tool import COMPONENT_RESULT_SCHEMA_VERSION
    artifacts = []
    for role, path in paths:
        record = file_record(path)
        artifacts.append({"role": role, "path": record["path"], "digest": record["sha256"],
                          "size_bytes": record["size_bytes"]})
    result = {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, "status": "completed",
        "adapter_id": "content_agents_rigid_replacement", "stage_id": stage_input["stage"]["stage_id"], "backend": ARTISAN_BACKEND,
        "request_digest": request["request_digest"], "provider_mutations_performed": 0, "nested_paid_execution_requested": False,
        "artifacts": artifacts}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def execute_artisan_component(*, environment=None, runner=run_controller,
                              cost_gate_factory=scene_configuration_openai_stage_gate):
    values = dict(os.environ if environment is None else environment)
    output = Path(values[OUTPUT]).resolve()
    require(output.is_dir() and not output.is_symlink(), "artisan_output_root_invalid")
    with (output / "artisan_component.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute(values, output, runner, cost_gate_factory)


def _execute(values, output, runner, cost_gate_factory):
    from .task_evaluation_scene_configuration_stage_tool import _validate_input, _validate_dependencies, _validate_component_result
    input_path = Path(values[INPUT])
    stage_input = _validate_input(read_json(input_path), adapter_id="content_agents_rigid_replacement",
        diagnostic_only=values.get("BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY") == "1")
    dependencies = _validate_dependencies(json.loads(Path(values[DEPENDENCIES]).read_text()))
    configuration, envelope = stage_input["configuration"], stage_input["construction_envelope"]
    manifest, frames = source_frames(stage_input, dependencies)
    request = build_construction_request(stage_input, dependencies, frames, manifest)
    rights_rows = [r for r in envelope["materialized_references"] if r["contract_path"] == "scene.rights.admission"]
    require(len(rights_rows) == 1, "artisan_source_rights_missing")
    rights_record = {**rights_rows[0], "path": rights_rows[0]["materialized_path"]}
    rights = read_json(verified_file(rights_record))
    if configuration.get("source_observation_kind") == "website_capture_frames":
        from .website_native_inputs import validate_website_authoring_disclosure
        validate_website_authoring_disclosure(envelope=envelope, configuration=configuration, rights=rights)
    else:
        require(rights.get("status") == "admitted_for_internal_development"
                and rights.get("private_provider_processing_allowed") is True
                and rights.get("provider_training_allowed") is False
                and rights.get("public_redistribution_allowed") is False, "artisan_disclosure_not_admitted")
    disclosure = configuration.get("provider_disclosure") or {}
    require(disclosure.get("derived_views_and_metric_envelope") is True
            and disclosure.get("provider_training") is False
            and disclosure.get("public_redistribution") is False, "artisan_disclosure_not_admitted")
    runtime = output / "artisan_runtime"
    runtime.mkdir(exist_ok=True)
    request_path = runtime / "blueprint_request.v1.json"
    if request_path.exists():
        require(read_json(request_path) == request, "artisan_issued_request_changed_on_resume")
    else:
        write_json(request_path, request)
    result_path = Path(values[RESULT])
    require(result_path.resolve().is_relative_to(output), "artisan_result_path_outside_output")
    if result_path.exists():
        result = read_json(result_path)
        require(result.get("request_digest") == request["request_digest"]
                and result.get("result_digest") == canonical_digest(result, digest_field="result_digest"), "artisan_completed_result_changed")
        for artifact in result["artifacts"]:
            verified_file(artifact, root=output)
        from .task_object_artisan_candidate import verify_artisan_candidate
        artifacts = {row["role"]: Path(row["path"]) for row in result["artifacts"]}
        verify_artisan_candidate(receipt=read_json(artifacts["replacement_authoring_receipt"]),
            graph=read_json(artifacts["replacement_graph_spec"]), asset=artifacts["replacement_asset"],
            configuration=configuration, source_candidate_digest=request["source_candidate"]["digest"])
        canonical = _component_result(stage_input, request, ((row["role"], Path(row["path"])) for row in result["artifacts"]))
        _validate_component_result(canonical, adapter_id=stage_input["stage"]["adapter"]["id"],
            stage_id=stage_input["stage"]["stage_id"], output_root=output, diagnostic_only=False)
        if result != canonical:
            # Repair only the historical producer's exact hash-key mismatch.
            # Preserve its original bytes before publishing the corrected result.
            legacy = {**canonical, "artifacts": [
                {"role": row["role"], "path": row["path"], "sha256": row["digest"], "size_bytes": row["size_bytes"]}
                for row in canonical["artifacts"]]}
            legacy["result_digest"] = canonical_digest(legacy, digest_field="result_digest")
            require(result == legacy, "artisan_completed_result_changed")
            history = result_path.with_name(f"{result_path.stem}.legacy-sha256-{result['result_digest'][7:]}.json")
            original = result_path.read_bytes()
            require(not history.is_symlink(), "artisan_legacy_result_history_invalid")
            if history.exists():
                require(history.read_bytes() == original, "artisan_legacy_result_history_changed")
            else:
                with history.open("xb") as stream:
                    stream.write(original)
                    stream.flush()
                    os.fsync(stream.fileno())
            write_json(result_path, canonical)
        return canonical
    candidates = output / "candidates"
    retained_receipts = sorted(candidates.glob("*/replacement_authoring_receipt.v1.json")) if candidates.exists() else []
    if retained_receipts:
        from .task_object_artisan_candidate import verify_artisan_candidate
        retained_receipt = retained_receipts[-1]
        candidate_root = retained_receipt.parent
        graph_path, asset = candidate_root / "replacement_graph_spec.v1.json", candidate_root / "replacement_asset.usdz"
        receipt = read_json(retained_receipt)
        require(receipt["graph_sha256"] == file_record(graph_path)["sha256"], "artisan_retained_graph_changed")
        verify_artisan_candidate(receipt=receipt, graph=read_json(graph_path), asset=asset,
            configuration=configuration, source_candidate_digest=request["source_candidate"]["digest"])
        result = _component_result(stage_input, request, (("replacement_asset", asset),
            ("replacement_authoring_receipt", retained_receipt), ("replacement_graph_spec", graph_path)))
        write_json(result_path, result)
        return result
    package = Path(values[PACKAGE]).resolve()
    packaged = validate_artisan_runtime(package)
    from .task_evaluation_scene_configuration_builtin_producers import _validate_toolchain, TOOLCHAIN_ROOT_ENV
    toolchain = Path(values[TOOLCHAIN_ROOT_ENV]).resolve()
    parent_manifest, _ = _validate_toolchain(root=toolchain, expected_source_commit=stage_input["source_commit"])
    require(parent_manifest["toolchain_digest"] == stage_input["toolchain_digest"] and package.is_relative_to(toolchain),
            "artisan_parent_toolchain_binding_changed")
    effective_cap = min(frozen_authoring_cap(configuration),
        float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD"]),
        float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"]))
    require(effective_cap > 0, "artisan_signed_stage_allocation_missing")
    deadline = min(float(values[STAGE_DEADLINE_EPOCH_ENV]),
                   float(values.get(PARENT_DEADLINE_EPOCH_ENV) or values[STAGE_DEADLINE_EPOCH_ENV])) - 600
    scope = scene_configuration_openai_stage_scope(values, stage="content_agents")
    source, libraries, bridge = package / "artisan_source", package / "runtime/site_packages", package / "bridge"
    workspace = runtime / "workspace"
    workspace.mkdir(exist_ok=True)
    blender = package / "runtime/blender/blender"
    import_roots = [str(bridge), str(source), str(libraries)]
    read_roots = [*import_roots, str(Path(sys.base_prefix).resolve()), str(blender.parent)]
    # Credentials must be outside every root visible to generated code.
    key_path = Path(scope["api_key_file"]).resolve()
    system_reads = ("/usr", "/bin", "/lib", "/lib64", "/etc/fonts", "/etc/ld.so.cache", "/etc/localtime", "/proc/self")
    require(not any(key_path.is_relative_to(Path(p).resolve()) for p in [str(workspace), *read_roots, *system_reads]),
            "artisan_credential_inside_generated_read_root")
    upstream_run_id = "pl_" + request["request_digest"][7:39]
    generated_workspace = workspace / "data/cache/runs" / upstream_run_id / "workspace"
    terminal_file = runtime / "terminal.v1.json"
    context = {"workspace": str(workspace), "upstream_run_id": upstream_run_id,
        "usage_file": str(runtime / "blueprint_usage.json"),
        "maximum_cost_usd": effective_cap, "maximum_requests": int(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS"]),
        "deadline_epoch": deadline, "api_key_file": scope["api_key_file"], "terminal_file": str(terminal_file),
        "project_id": values.get("OPENAI_PROJECT_ID"),
        "object_kind": "articulated_assembly" if configuration["schema_version"].startswith("articulated_") else "rigid_body",
        "model_configuration": request["authoring_models"],
        "sandbox": {"write_root": str(generated_workspace), "read_roots": read_roots, "import_roots": import_roots,
                    "trusted_asset_roots": [str(source / "sdk/assets")],
                    "write_paths": [str(generated_workspace / name) for name in ("assets", "inputs", "blender", "renders", "tmp", "xdg", "cache", "blueprint_sandbox_ipc")],
                    "blender": str(blender)}}
    context_path = runtime / "bridge_context.v1.json"
    if context_path.exists():
        require(read_json(context_path) == context, "artisan_authority_or_workspace_changed_on_resume")
    else:
        write_json(context_path, context)
    from .task_object_artisan_hooks import sandbox
    preflight = runtime / "sandbox_preflight"
    preflight.mkdir(exist_ok=True)
    preflight_runner = sandbox({**context["sandbox"], "write_root": str(preflight), "write_paths": None})
    preflight_runner.preflight()
    preflight_result = preflight_runner([str(Path(sys.executable).resolve()), "-S", "-c",
        "import sdk, cadquery, mujoco, coacd, fcl; s=cadquery.Workplane('XY').box(0.1,0.1,0.1); assert s.val().isValid()"],
        cwd=preflight, env={"PYTHONPATH": os.pathsep.join(import_roots)},
        timeout=CONFINED_PREFLIGHT_TIMEOUT_SECONDS, check=False)
    require(preflight_result.returncode == 0, "artisan_confined_runtime_preflight_failed")
    prompt_path = runtime / "prompt.txt"
    prompt = artisan_prompt(request)
    if prompt_path.exists():
        require(prompt_path.read_text() == prompt, "artisan_retained_prompt_changed")
    else:
        prompt_path.write_text(prompt)
    checkpoint = workspace / "data/cache/runs" / upstream_run_id / "pipeline_state.json"
    terminal = read_json(terminal_file) if terminal_file.exists() else None
    completed_upstream = terminal and terminal.get("phase") == "DONE" and terminal.get("registered") is True
    if not completed_upstream:
        require(time.time() < deadline, "artisan_absolute_stage_deadline_exhausted")
        if (runtime / "blueprint_usage.json").exists():
            usage = read_json(runtime / "blueprint_usage.json")
            require(all(row["status"] == "priced" for row in usage["calls"]), "artisan_unknown_charge_requires_reconciliation")
        # Always use original absolute workspace/run identity on upstream resume.
        argv = [str(Path(sys.executable).resolve()), "-S", "-m", "blueprint_pipeline.task_object_artisan_bridge"]
        if checkpoint.exists():
            argv += ["resume", upstream_run_id]
        else:
            argv += ["run", "--prompt-file", str(prompt_path), "--run-id", upstream_run_id]
            for frame in frames:
                argv += ["--image", str(frame)]
        argv += ["--workspace", str(workspace), "--max-cost-usd", str(effective_cap), "--provider", "openai"]
        gate = cost_gate_factory(environment=values, stage="content_agents", run_id=stage_input["run_id"],
            request_digest=file_record(input_path)["sha256"], candidate_digest=request["source_candidate"]["digest"],
            output_root=runtime / "official_openai_cost", max_cost_usd=effective_cap)
        if gate.reservation_path.exists():
            gate._validated_reservation()
        else:
            gate.reserve()
        # Inference controller gets only its scoped key path, proxy/CA trust and
        # explicit runtime paths. Ambient provider credentials/config are excluded.
        child_env = {name: values[name] for name in ("PATH", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                     "http_proxy", "https_proxy", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE") if name in values}
        child_env.update(PYTHONPATH=os.pathsep.join(import_roots), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                         BLUEPRINT_ARTISAN_BRIDGE_CONTEXT_FILE=str(context_path), ARTISAN_CONFIG=str(runtime / "artisan.toml"))
        config_text = '[artisan]\nrender_device = "cpu"\ncpu_threads = 1\n' + ''.join(
            key + ' = ' + json.dumps(value) + '\n' for key, value in context["model_configuration"].items())
        (runtime / "artisan.toml").write_text(config_text)
        attempted, failure = False, None
        try:
            attempted = True
            finished = runner(argv, cwd=workspace, env=child_env, timeout=deadline - time.time(), check=False, capture_output=True, text=True)
            # Log exact upstream progress with credential redaction, never argv keys.
            secret = key_path.read_text().strip()
            log = (finished.stdout or "") + "\n" + (finished.stderr or "")
            if secret:
                log = log.replace(secret, "[redacted]")
            (runtime / ("controller-" + str(len(list(runtime.glob("controller-*.log")))) + ".log")).write_text(log)
            require(terminal_file.is_file(), "artisan_terminal_record_missing")
            terminal = read_json(terminal_file)
            require(finished.returncode == 0, "artisan_framework_terminal_failure")
        except Exception as exc:
            failure = type(exc).__name__
            write_json(runtime / "last_failure.json", {"status": "failed_or_uncertain", "exception_type": failure,
                "request_digest": request["request_digest"], "upstream_run_id": upstream_run_id,
                "completed_generation_adoptable": terminal_file.is_file()})
            raise
        finally:
            meter_path = runtime / "blueprint_usage.json"
            calls = read_json(meter_path)["calls"] if meter_path.exists() else []
            snapshot_gate = gate
            if failure:
                # One original official reservation/baseline governs all
                # attempts. Failure snapshots are immutable; they do not close
                # or reset its shared pool before a same-workspace resume.
                attempts = runtime / "official_openai_cost/attempts"
                attempts.mkdir(exist_ok=True)
                attempt_root = attempts / str(len(list(attempts.iterdir())))
                snapshot_gate = replace(gate, output_root=attempt_root)
                shutil.copyfile(gate.reservation_path, snapshot_gate.reservation_path)
            snapshot_gate.complete(provider_call_performed=attempted and bool(calls), runtime_result_digest=canonical_digest(terminal) if terminal else None,
                          runtime_exception_type=failure)
    verify_construction_request(request)
    exports = upstream_exports(terminal, workspace)
    candidates.mkdir(exist_ok=True)
    candidate_root = candidates / f"{len(list(candidates.iterdir())):04d}"
    candidate_root.mkdir()
    asset = candidate_root / "replacement_asset.usdz"
    try:
        adaptation = adapt_usd(Path(exports["usd"]["path"]), asset, request,
            annotations_path=Path(terminal["paths"]["root"]) / "assets/blueprint_candidate_annotations.v1.json")
    except Exception as exc:
        write_json(candidate_root / "refusal.v1.json", {"schema_version": "artisan_candidate_refusal.v1",
            "status": "candidate_refused", "request_digest": request["request_digest"],
            "upstream_candidate_id": terminal["candidate_id"], "exception_type": type(exc).__name__,
            "reason": str(exc)[:2000], "framework_generation_repeated": False})
        raise
    observed = adaptation["final_readback"]
    graph = {"schema_version": "task_evaluation_articulated_replacement_graph.v1" if context["object_kind"] == "articulated_assembly"
             else "task_evaluation_rigid_replacement_graph.v1", "backend": ARTISAN_BACKEND,
             "asset_id": configuration["replacement_identity"]["id"], "asset_version": configuration["replacement_identity"]["version"],
             "articulation_graph": observed["articulation_graph"], "final_readback": observed,
             "construction_request": file_record(request_path), "physics_authority_granted": False,
             "single_rigid_candidate": context["object_kind"] == "rigid_body"}
    graph_path = candidate_root / "replacement_graph_spec.v1.json"
    write_json(graph_path, graph)
    receipt = {"schema_version": "task_object_artisan_candidate.v1", "backend": ARTISAN_BACKEND,
        "status": "authored_candidate_pending_qualification", "asset_kind": context["object_kind"],
        "replacement_identity": configuration["replacement_identity"], "source_candidate_digest": request["source_candidate"]["digest"],
        "source_candidate_claim": "source_geometry_not_observed_truth_or_physics_authority",
        "source_object_identity": request["source_object_identity"], "request_digest": request["request_digest"],
        "construction_request": file_record(request_path), "upstream_runtime": packaged,
        "upstream_terminal": terminal, "upstream_artifacts": exports, "export_adaptation": adaptation,
        "output_usd": file_record(asset), "graph_sha256": file_record(graph_path)["sha256"],
        "physics_authority_granted": False, "claim_ceiling": "development_only", "native_import_qualified": False,
        "scene_placement_qualified": False, "physical_equivalence_proven": False}
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    receipt_path = candidate_root / "replacement_authoring_receipt.v1.json"
    write_json(receipt_path, receipt)
    result = _component_result(stage_input, request, (("replacement_asset", asset),
              ("replacement_authoring_receipt", receipt_path), ("replacement_graph_spec", graph_path)))
    write_json(result_path, result)
    return result
