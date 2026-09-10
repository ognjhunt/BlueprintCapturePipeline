"""Stage-3 Astra CAD/Blender candidate authoring behind existing parent admission."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import fcntl
from functools import wraps
import importlib.util
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import unquote, urlparse

from .agent_operator_runtime import LIVE_AGENTS_SDK_ENV
from .asset_authoring_sandbox import SandboxedAssetRunner
from .astra_cad_skill_runtime import execute_mac_candidate, verify_cad_sources
from .decision_evidence_contracts import canonical_digest, canonical_json
from .production_blender_runtime import validate_runtime
from .task_evaluation_scene_configuration_builtin_producers import TOOLCHAIN_ROOT_ENV, _validate_toolchain
from .task_evaluation_scene_configuration_content_agents_driver import (
    _ADAPTER_ID, _DEPENDENCIES_ENV, _INPUT_ENV, _OUTPUT_ENV, _PACKAGE_ENV, _RESULT_ENV,
    _dependency_candidate, _file_record, _materialize_cad_skill_runtime, _metric_envelope_spec,
    _physics_bounds, _read, _reference_frames, _required_path, _sha256,
    _validate_metric_envelope_dimensions,
)
from .task_evaluation_scene_configuration_openai_gate import (
    scene_configuration_openai_stage_gate, scene_configuration_openai_stage_scope,
)
from .task_evaluation_scene_configuration_astra_phase_adoption import (
    materialize_automatic_phase_adoption, prepare_phase_adoption,
)
from .task_evaluation_scene_configuration_render_inputs import _materialized
from .task_evaluation_scene_configuration_stage_tool import (
    COMPONENT_RESULT_SCHEMA_VERSION, _validate_dependencies, _validate_input,
)
from .task_object_astra_authoring import (
    AuthoringRequest, budgeted_invoker, execute_asset_authoring, file_record, validate_request,
)
from .task_object_physical_property_review import EvidenceReference

BACKEND = "astra_cad_blender_v1"
BLENDER_ROOT_ENV = "BLUEPRINT_BLENDER_RUNTIME_ROOT"
_CAD_PACKAGE_FILES = ("text_to_cad_skills_source.zip", "multi_agent_cad_source.zip",
                      "cad_skill_source_receipt.json", "multi_agent_cad_skill.md")


class AstraStageError(RuntimeError):
    """The parent stage's evidence, runtime, or inference admission refused."""


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _stage_source_binding(request, stage_input, source_record, rights_path):
    semantic = request.model_dump(mode="json")
    for key in ("request_digest", "expected_production_commit"):
        semantic.pop(key)
    value = {"schema_version": "astra_same_run_source_binding.v1", "run_id": request.run_id,
        "authoring_input_digest": canonical_digest(semantic), "source_candidate": dict(source_record),
        "rights_admission": file_record(rights_path), "configuration_sha256": stage_input["configuration_sha256"]}
    value["binding_digest"] = canonical_digest(value, digest_field="binding_digest")
    return value


def _verify_physical_evidence(values: Any, envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        raise AstraStageError("astra_physical_evidence_invalid")
    verified = []
    for value in values:
        reference = EvidenceReference.model_validate(value)
        parsed = urlparse(reference.uri)
        if parsed.scheme in ("", "file") and parsed.netloc in ("", "localhost"):
            path = Path(unquote(parsed.path))
        else:
            rows = [row for row in envelope.get("materialized_references", [])
                    if row.get("digest") == "sha256:" + reference.sha256
                    and row.get("full_byte_service_account_readback_passed") is True]
            if len(rows) != 1:
                raise AstraStageError("astra_physical_evidence_requires_retained_bytes")
            path = Path(str(rows[0].get("materialized_path") or ""))
        if not path.is_absolute() or file_record(path)["sha256"] != "sha256:" + reference.sha256:
            raise AstraStageError("astra_physical_evidence_digest_mismatch")
        verified.append(reference.model_dump(mode="json"))
    return verified


def build_authoring_request(stage_input: Mapping[str, Any], source_record: Mapping[str, Any],
                            references: list[Path], rights: Mapping[str, Any]) -> AuthoringRequest:
    """Translate retained data without inventing physical measurements or rounding geometry."""
    configuration = stage_input["configuration"]
    disclosure = configuration.get("provider_disclosure") or {}
    if (configuration.get("authoring_backend") != BACKEND
            or disclosure.get("derived_views_and_metric_envelope") is not True
            or disclosure.get("provider_training") is not False
            or disclosure.get("public_redistribution") is not False
            or rights.get("status") != "admitted_for_internal_development"
            or rights.get("private_provider_processing_allowed") is not True
            or rights.get("provider_training_allowed") is not False
            or rights.get("public_redistribution_allowed") is not False):
        raise AstraStageError("astra_derived_disclosure_not_admitted")
    envelope = _metric_envelope_spec(configuration)
    dimensions = envelope["expected_dimensions_m"]
    uncertainty = configuration.get("dimension_uncertainty_m")
    uncertainty_note = "Supplied source-geometry uncertainty; not physical measurement."
    if uncertainty is None:
        uncertainty = configuration["metric_envelope"].get("dimension_uncertainty_m")
    if uncertainty is None:
        uncertainty = [d * envelope["maximum_dimension_relative_error"] for d in dimensions]
        uncertainty_note = ("Source-envelope relative tolerance used as an explicit uncertainty proxy; "
                            "it is not measured accuracy or a permission to round nominal dimensions.")
    if (not isinstance(uncertainty, (tuple, list)) or len(uncertainty) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0 < v < dimensions[i] for i, v in enumerate(uncertainty))):
        raise AstraStageError("astra_source_geometry_uncertainty_missing_or_invalid")
    identity = configuration["replacement_identity"]
    owner = str(configuration.get("authoring_target") or "").strip()
    source_identity = configuration.get("source_object_identity")
    if not owner or not source_identity or not references:
        raise AstraStageError("astra_owner_identity_or_reference_missing")
    geometry_id = "retained_source_geometry"
    evidence = [{"evidence_id": geometry_id, "uri": str(source_record["path"]),
                 "sha256": str(source_record["digest"]).removeprefix("sha256:"), "kind": "source_geometry",
                 "excerpt": "Retained source object identity and metric envelope: " + canonical_json({
                     "source_object_identity": source_identity, "metric_envelope": envelope})}]
    frames = []
    for index, reference in enumerate(references):
        record = file_record(reference)
        frames.append({"path": record["path"], "sha256": record["sha256"], "role": "observed_source",
                       "description": f"Digest-bound stage-1 appearance view {index}; derived source render, not physical truth."})
        evidence.append({"evidence_id": f"retained_source_view_{index}", "uri": record["path"],
                         "sha256": record["sha256"].removeprefix("sha256:"), "kind": "material_observation",
                         "excerpt": "Owner-described object shown in the retained source-derived appearance view."})
    evidence.extend(_verify_physical_evidence(configuration.get("physical_evidence", []),
                                              stage_input["construction_envelope"]))
    if len({row["evidence_id"] for row in evidence}) != len(evidence):
        raise AstraStageError("astra_duplicate_physical_evidence")
    material = str(configuration.get("material_description") or
                   f"Infer material conservatively from the retained owner description and reference views: {owner}")
    appearance = configuration.get("appearance", "unknown")
    physical = {
        "object_id": identity["id"], "object_description": owner, "material_description": material,
        "appearance": appearance, "dimensions": {},
        "measured": configuration.get("measured_physical_properties") or dict.fromkeys(
            ("mass_kg", "static_friction", "dynamic_friction", "restitution")),
        "proposed": None,
        "optical_material": {"name": material, "transmission": 0.0, "opacity": 1.0},
        "admitted_restitution": dict(zip(("lower", "upper"), _physics_bounds(configuration)["restitution"])),
        "evidence": evidence,
    }
    for index, axis in enumerate(("x_m", "y_m", "z_m")):
        physical["dimensions"][axis] = {"value": dimensions[index], "basis": "estimated",
            "interval": {"lower": dimensions[index] - uncertainty[index], "upper": dimensions[index] + uncertainty[index]},
            "rationale": "Exact nominal source-geometry construction constraint.",
            "uncertainty": uncertainty_note, "evidence_ids": [geometry_id]}
    value = {
        "schema_version": "task_object_astra_authoring_request.v1", "run_id": stage_input["run_id"],
        "object_id": identity["id"], "owner_description": owner,
        "role": configuration.get("role", "task_object"), "dimensions_m": dimensions,
        "dimension_authority": "source_geometry", "dimension_source_digest": source_record["digest"],
        "dimension_uncertainty_m": uncertainty, "coordinate_frame": "object_center_xy_bottom_z_z_up_meters",
        "maximum_export_error_m": configuration.get("maximum_export_error_m", 0.00001),
        "source_frames": frames, "physical_review_input": physical,
        "construction_constraints": canonical_json({"owner_description": owner,
            "source_object_identity": source_identity, "replacement_identity": identity,
            "exact_nominal_dimensions_m": dimensions, "source_uncertainty_note": uncertainty_note,
            "required_output": configuration["required_output"],
            "additional_constraints": configuration.get("construction_constraints", "")}),
        "private_provider_processing_allowed": True, "provider_training_allowed": False,
        "public_redistribution_allowed": False, "expected_production_commit": stage_input["source_commit"],
        "request_digest": "",
    }
    value["request_digest"] = "sha256:" + "0" * 64
    value = AuthoringRequest.model_validate(value).model_dump(mode="json")
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    request = validate_request(value)
    if request.maximum_export_error_m * 1000 > 0.1:
        raise AstraStageError("astra_cad_export_tolerance_unsupported")
    evidence_by_id = {row.evidence_id: row for row in request.physical_review_input.evidence}
    for measured in request.physical_review_input.measured:
        prop = measured[1]
        if prop is not None and (prop.basis != "measured" or any(
            evidence_id not in evidence_by_id for evidence_id in prop.evidence_ids
        ) or not any(evidence_by_id[evidence_id].kind in {"physical_measurement", "capture_measurement"}
                     for evidence_id in prop.evidence_ids)):
            raise AstraStageError("astra_measured_property_evidence_invalid")
    return request


class _StageInvoker:
    def __init__(self, invoker, run_id: str, maximum_calls: int, prior_calls: int = 0):
        self.invoker, self.run_id, self.maximum_calls, self.calls = invoker, run_id, maximum_calls, 0
        self.prior_calls = prior_calls

    def invoke(self, spec, input_value):
        if (self.calls + self.prior_calls >= self.maximum_calls or spec.run_id != self.run_id or spec.model != "gpt-6-astra"
                or spec.max_turns != 1 or spec.tool_bindings or spec.max_output_tokens > 12000
                or spec.max_input_tokens is None or spec.max_input_tokens > 80000
                or spec.reasoning_effort not in {"medium", "high"}):
            raise AstraStageError("astra_stage_inference_boundary_refused")
        self.calls += 1
        return self.invoker.invoke(spec, input_value)


@contextmanager
def _stage_sdk_environment(key_path: Path):
    from agents import set_default_openai_client
    from agents.models import _openai_shared
    from openai import AsyncOpenAI
    names = ("OPENAI_API_KEY", "OPENAI_API_KEY_FILE", LIVE_AGENTS_SDK_ENV)
    previous = {name: os.environ.get(name) for name in names}
    previous_client = _openai_shared.get_default_openai_client()
    try:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["OPENAI_API_KEY_FILE"] = str(key_path)
        os.environ[LIVE_AGENTS_SDK_ENV] = "1"
        set_default_openai_client(AsyncOpenAI(api_key=key_path.read_text().strip(),
            base_url='https://api.openai.com/v1', max_retries=0, timeout=600), use_for_tracing=False)
        yield
    finally:
        _openai_shared.set_default_openai_client(previous_client)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _locked_component(function):
    @wraps(function)
    def execute(*args, **kwargs):
        values = os.environ if kwargs.get("environment") is None else kwargs["environment"]
        root = _required_path(values, _OUTPUT_ENV)
        path = root / ".astra-component.lock"
        if path.is_symlink():
            raise AstraStageError("astra_component_lock_unsafe")
        with path.open("a+b") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AstraStageError("astra_component_already_running") from exc
            return function(*args, **kwargs)
    return execute


@_locked_component
def execute_astra_component(*, environment=None, runner=subprocess.run,
                            cost_gate_factory=scene_configuration_openai_stage_gate,
                            authoring_executor=execute_asset_authoring, invoker_factory=budgeted_invoker,
                            sandbox_factory=SandboxedAssetRunner, blender_validator=validate_runtime,
                            package_candidate=None, no_cost_replay=False, retained_runtime=None) -> dict[str, Any]:
    values = dict(os.environ if environment is None else environment)
    input_path = _required_path(values, _INPUT_ENV)
    stage_input = _validate_input(_read(input_path, code="astra_stage_input_invalid"),
        adapter_id=_ADAPTER_ID, diagnostic_only=values.get("BLUEPRINT_SCENE_CONFIGURATION_DIAGNOSTIC_ONLY") == "1")
    dependencies = _validate_dependencies(json.loads(_required_path(values, _DEPENDENCIES_ENV).read_text()))
    toolchain_root = _required_path(values, TOOLCHAIN_ROOT_ENV)
    manifest, _ = _validate_toolchain(root=toolchain_root, expected_source_commit=stage_input["source_commit"])
    if manifest["toolchain_digest"] != stage_input["toolchain_digest"]:
        raise AstraStageError("astra_parent_toolchain_digest_mismatch")
    configuration, envelope = stage_input["configuration"], stage_input["construction_envelope"]
    source_record, _ = _dependency_candidate(dependencies)
    references = _reference_frames(stage_input, dependencies)
    rights_record, rights_path = _materialized(envelope, contract_path="scene.rights.admission")
    request = build_authoring_request(stage_input, source_record, references,
                                     _read(rights_path, code="astra_rights_invalid"))
    physics_bounds = _physics_bounds(configuration)
    output = _required_path(values, _OUTPUT_ENV)
    result_path = _required_path(values, _RESULT_ENV)
    package = _required_path(values, _PACKAGE_ENV)
    if not package.is_relative_to(toolchain_root) or not result_path.is_relative_to(output) or result_path.exists():
        raise AstraStageError("astra_component_path_invalid")
    primary = output / "astra_cad_blender_runtime"
    attempts = output / "astra_resume_attempts"
    prior_roots = ([primary] if primary.exists() else []) + sorted(attempts.glob("attempt-????"))
    descriptor = configuration.get("astra_phase_adoption")
    if retained_runtime is not None:
        if not no_cost_replay or descriptor is not None:
            raise AstraStageError("astra_operational_replay_scope_invalid")
        descriptor = materialize_automatic_phase_adoption(prior_runtime=Path(retained_runtime))
    if prior_roots:
        if descriptor is not None and Path(descriptor["prior_runtime"]) != prior_roots[-1]:
            raise AstraStageError("astra_resume_cannot_skip_latest_budget_journal")
        descriptor = descriptor or materialize_automatic_phase_adoption(prior_runtime=prior_roots[-1])
    if no_cost_replay and (descriptor is None or "authoring_result" not in descriptor.get("completed_artifacts", [])):
        raise AstraStageError("astra_no_cost_replay_requires_completed_authoring")
    source_binding = _stage_source_binding(request, stage_input, source_record, rights_path)
    if descriptor is not None and descriptor.get("schema_version") == "task_evaluation_retained_astra_artifacts.v1":
        prior_binding = Path(descriptor["prior_runtime"]) / "stage_source_binding.json"
        if not prior_binding.is_file() or _read(prior_binding, code="astra_retained_source_binding_invalid") != source_binding:
            raise AstraStageError("astra_retained_source_or_rights_binding_changed")
    if len(prior_roots) >= 16:
        raise AstraStageError("astra_same_run_resume_limit_reached")
    runtime = primary if not prior_roots else attempts / f"attempt-{len(prior_roots):04d}"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(mode=0o700)
    _write(runtime / "stage_source_binding.json", source_binding)
    delivery_output = output if not prior_roots else runtime / "delivery"
    for name in _CAD_PACKAGE_FILES:
        source = package / name
        if source.is_symlink() or not source.is_file():
            raise AstraStageError("astra_cad_package_incomplete")
        shutil.copyfile(source, runtime / name)
    adoption = prepare_phase_adoption(value=descriptor, request_value=request.model_dump(mode="json"),
        package=package, budget_root=runtime / "inference")
    if descriptor is not None:
        _write(runtime / "retained_artifact_contract.json", descriptor)
    if package_candidate is None:
        from .task_object_simready_packaging import package_astra_candidate
        package_candidate = package_astra_candidate
    authored_root = runtime / "authoring"
    authored_root.mkdir()
    if adoption.get("completed_authoring_result") is not None:
        authored = adoption["completed_authoring_result"]
        _write(authored_root / "request.json", request.model_dump(mode="json"))
        _write(authored_root / "result.json", authored)
        _write(runtime / "no_cost_authoring_adoption.json", {"status": "completed_authoring_adopted",
            "adoption_digest": adoption["adoption_digest"], "retained_inference_cost_usd": adoption["retained_inference_cost_usd"],
            "prior_call_count": adoption["prior_call_count"], "new_provider_calls": 0,
            "cad_execution_repeated": False, "blender_execution_repeated": False})
        retained_runtime = {"status": "retained_completed_artifacts", "adoption_digest": adoption["adoption_digest"],
                            "runtime_execution_repeated": False}
        return _finish_component(request=request, authored=authored, package_candidate=package_candidate,
            output=delivery_output, physics_bounds=physics_bounds, configuration=configuration,
            source_record=source_record, stage_input=stage_input, rights_record=rights_record,
            cad_runtime=retained_runtime, blender=retained_runtime, authored_root=authored_root, result_path=result_path)
    cad_runtime = _materialize_cad_skill_runtime(runtime)
    cad_root = Path(cad_runtime["root"])
    verified_sources = {}
    for name, folder, source_id in (("mac", "Multi-Agent-CAD", "multi-agent-cad"),
                                    ("cad", "text-to-cad", "text-to-cad")):
        source_root = cad_root / folder
        archive = runtime / ("multi_agent_cad_source.zip" if name == "mac" else "text_to_cad_skills_source.zip")
        verified_sources[name] = {"root": str(source_root), "commit": cad_runtime["source_commits"][source_id],
            "tracked_file_sha256": {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in source_root.rglob("*") if path.is_file()},
            "source_diff": "", "source_receipt_digest": cad_runtime["receipt_digest"],
            "source_receipt_path": str(runtime / "cad_skill_source_receipt.json"),
            "archive_path": str(archive), "archive_sha256": _sha256(archive)}
    if "verified_sources" not in inspect.signature(execute_mac_candidate).parameters:
        raise AstraStageError("astra_mac_archive_admission_unavailable")
    verify_cad_sources(cad_root / "Multi-Agent-CAD", cad_root / "text-to-cad", verified_sources)
    if str(values.get(BLENDER_ROOT_ENV) or "").strip():
        blender_root = _required_path(values, BLENDER_ROOT_ENV)
        blender = blender_validator(blender_root, runner=runner)
    else:
        from .task_evaluation_scene_configuration_astra_runtime import materialize_packaged_blender_runtime
        blender_root = runtime / "packaged_blender"
        blender = materialize_packaged_blender_runtime(package, blender_root, runner=runner)
    for dependency in ("build123d", "langgraph", "agents"):
        if importlib.util.find_spec(dependency) is None:
            raise AstraStageError(f"astra_runtime_dependency_missing:{dependency}")
    runtime_loader = [Path(value).resolve() for value in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                      if value and Path(value).is_dir()]
    roots = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), cad_root,
             blender_root, Path(__file__).resolve().parent.parent, *runtime_loader]
    roots = list(dict.fromkeys(roots))
    sandbox = sandbox_factory(read_roots=roots, write_root=authored_root,
        executable_roots=[Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), blender_root])
    sandbox.preflight()
    cad_probe = sandbox([sys.executable, "-c", "import build123d; from langgraph.graph import StateGraph; "
                         "from cadpy.generation import run_script_generator; "
                         "assert abs(build123d.Box(1,2,3).volume - 6) < 1e-8"],
        cwd=authored_root, env={"PYTHONPATH": os.pathsep.join(dict.fromkeys([
            str(cad_root / "Multi-Agent-CAD/packages/cadpy/src"),
            str(cad_root / "text-to-cad/packages/cadpy/src"), *map(str, runtime_loader)]))},
        capture_output=True, text=True, check=False, timeout=60)
    if cad_probe.returncode:
        _write(runtime / "cad_runtime_preflight_failure.json", {"returncode": cad_probe.returncode,
            "stdout": cad_probe.stdout[-4000:], "stderr": cad_probe.stderr[-4000:]})
        raise AstraStageError("astra_sandboxed_cad_runtime_preflight_failed")
    scope = scene_configuration_openai_stage_scope(values, stage="content_agents")
    key_path = Path(scope["api_key_file"]).expanduser()
    if key_path.is_symlink() or not key_path.is_file() or key_path.stat().st_mode & 0o077 or not key_path.read_text().strip():
        raise AstraStageError("astra_stage_key_file_invalid")
    try:
        stage_cap = float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD"])
        total_cap = float(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"])
        maximum_cost = min(15.0, stage_cap, total_cap)
        maximum_calls = min(15, int(values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS"]))
    except (KeyError, ValueError, TypeError) as exc:
        raise AstraStageError("astra_parent_budget_invalid") from exc
    if any(not math.isfinite(v) or v <= 0 for v in (stage_cap, total_cap)) or maximum_calls <= 0:
        raise AstraStageError("astra_parent_budget_invalid")
    if adoption.get("retained_inference_cost_usd", 0) > maximum_cost:
        raise AstraStageError("astra_phase_adoption_budget_exhausted")
    base_invoker, audit = invoker_factory(root=runtime / "inference", run_id=request.run_id, maximum_cost_usd=maximum_cost)
    invoker = _StageInvoker(base_invoker, request.run_id, maximum_calls, adoption["prior_call_count"])
    if adoption.get("adoption_digest"):
        _write(runtime / "phase_adoption.json", {key: item for key, item in adoption.items()
                                               if key not in {"authoring_kwargs", "cad_kwargs"}})

    def mac_executor(*, brief, output_root, dimensions_m):
        result = execute_mac_candidate(brief, output_root, cad_root / "Multi-Agent-CAD", cad_root / "text-to-cad",
            invoker, expected_dimensions_mm=tuple(value * 1000 for value in dimensions_m),
            subprocess_runner=sandbox, run_id=request.run_id, object_label=request.object_id,
            max_input_tokens=80000, max_output_tokens=12000, max_calls=maximum_calls, repair_budget=2,
            dimension_tolerance_mm=request.maximum_export_error_m * 1000,
            verified_sources=verified_sources, **adoption["cad_kwargs"])
        return {**result, "stl": file_record(Path(result["stl_path"])),
                "step": file_record(Path(result["step_path"])),
                "measured_dimensions_m": [v / 1000 for v in result["readback"]["measured_dimensions_mm"]]}

    gate = cost_gate_factory(environment=values, stage="content_agents", run_id=request.run_id,
        request_digest=_sha256(input_path), candidate_digest=source_record["digest"],
        output_root=runtime / "official_openai_cost", max_cost_usd=maximum_cost)
    gate.reserve()
    authored, failure = None, None
    try:
        with _stage_sdk_environment(key_path.resolve()):
            authored = authoring_executor(request_value=request.model_dump(mode="json"), output_root=authored_root,
                invoker=invoker, mac_executor=mac_executor, blender_runner=sandbox, blender_executable=blender["executable"],
                authoring_instructions=(cad_root / "text-to-cad/skills/cad/SKILL.md").read_text(),
                **adoption["authoring_kwargs"])
    except Exception as exc:
        failure = type(exc).__name__
        raise
    finally:
        try:
            _write(runtime / "inference_audit.json", audit.manifest())
        finally:
            gate.complete(provider_call_performed=invoker.calls > 0,
                          runtime_result_digest=(authored or {}).get("result_digest"), runtime_exception_type=failure)
    return _finish_component(request=request, authored=authored, package_candidate=package_candidate,
        output=delivery_output, physics_bounds=physics_bounds, configuration=configuration,
        source_record=source_record, stage_input=stage_input, rights_record=rights_record,
        cad_runtime=cad_runtime, blender=blender, authored_root=authored_root, result_path=result_path)


def _finish_component(*, request, authored, package_candidate, output, physics_bounds, configuration,
                      source_record, stage_input, rights_record, cad_runtime, blender, authored_root, result_path):
    output.mkdir(parents=True, exist_ok=True)
    packaged = package_candidate(request=request, authoring_result=authored,
                                  output_root=output, physics_bounds=physics_bounds)
    asset = Path(packaged["asset"]["path"])
    asset_record = {"path": packaged["asset"]["path"],
                    "digest": packaged["asset"].get("digest") or packaged["asset"].get("sha256"),
                    "size_bytes": packaged["asset"]["size_bytes"]}
    if asset.is_symlink() or not asset.resolve().is_relative_to(output) or _file_record(asset) != asset_record:
        raise AstraStageError("astra_packaged_asset_binding_invalid")
    completion = packaged["physics_completion"]
    completion["metric_envelope_validation"] = _validate_metric_envelope_dimensions(
        envelope=_metric_envelope_spec(configuration), observed_dimensions=completion["collision_dimensions_m"])
    completion["completion_digest"] = canonical_digest(completion, digest_field="completion_digest")
    identity = configuration["replacement_identity"]
    graph = {"schema_version": "task_evaluation_rigid_replacement_graph.v1", "asset_id": identity["id"],
        "asset_version": identity["version"], "articulation_graph": {"joints": []}, "single_rigid_candidate": True,
        "physics_bounds": physics_bounds, "physics_authority_granted": False, "authoring_backend": BACKEND}
    graph_path = output / "replacement_graph_spec.v1.json"
    _write(graph_path, graph)
    receipt = {"schema_version": "task_evaluation_rigid_replacement_authoring_result.v1",
        "status": "authored_candidate_pending_qualification", "authoring_backend": BACKEND, "model": "gpt-6-astra",
        "replacement_identity": identity, "source_candidate_digest": source_record["digest"],
        "source_candidate_claim": "source_geometry_not_observed_truth_or_physics_authority",
        "source_commit": stage_input["source_commit"], "toolchain_digest": stage_input["toolchain_digest"],
        "source_rights_admission": dict(rights_record), "cad_skill_runtime": cad_runtime, "blender_runtime": blender,
        "astra_authoring_result": _file_record(authored_root / "result.json"),
        "output_usd": {"sha256": _sha256(asset), "size_bytes": asset.stat().st_size},
        "candidate_physics_completion": completion, "physics_authority_granted": False, "result_digest": ""}
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    receipt_path = output / "replacement_authoring_receipt.v1.json"
    _write(receipt_path, receipt)
    result = {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, "status": "completed", "adapter_id": _ADAPTER_ID,
        "stage_id": stage_input["stage"]["stage_id"], "provider_mutations_performed": 0,
        "nested_paid_execution_requested": False, "authoring_backend": BACKEND, "model": "gpt-6-astra",
        "artifacts": [{"role": role, **_file_record(path)} for role, path in (
            ("replacement_asset", asset), ("replacement_authoring_receipt", receipt_path), ("replacement_graph_spec", graph_path))],
        "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _write(result_path, result)
    return result
