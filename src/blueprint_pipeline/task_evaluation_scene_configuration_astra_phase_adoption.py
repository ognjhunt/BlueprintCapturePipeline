"""Digest-bound same-run Astra phase adoption without resetting paid inference.

This deliberately retains the existing exact source-path and run identity
predicates. Relocating evidence or adopting another run needs a separate contract.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
from .task_object_astra_authoring import AssetAuthoringError, file_record, validate_request
from .task_object_astra_worker import (
    verify_blender_program_adoption,
    verify_physical_review_adoption,
    verify_source_analysis_adoption,
)

SCHEMA_VERSION = "task_evaluation_astra_phase_adoption.v1"
AUTOMATIC_SCHEMA_VERSION = "task_evaluation_retained_astra_artifacts.v1"
PHASES = {"source_analysis", "cad_state", "coder", "physical_review", "blender_program"}
PHASE_ORDER = ("source_analysis", "cad_state", "coder", "physical_review", "blender_program")
SOURCE_ARCHIVES = ("text_to_cad_skills_source.zip", "multi_agent_cad_source.zip",
                   "cad_skill_source_receipt.json", "multi_agent_cad_skill.md")


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssetAuthoringError("astra_phase_adoption_symlink_forbidden")
        if path.is_file():
            relative = path.relative_to(root)
            if (relative.parts[0] in {"authoring", "retained_astra_phases"} or str(relative) in SOURCE_ARCHIVES
                    or str(relative) in {"no_cost_authoring_adoption.json", "stage_source_binding.json"}
                    or relative.parts[:2] == ("inference", "inference_reservations")):
                with path.open("rb") as stream:
                    digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
                records.append({"relative_path": str(relative), "sha256": digest,
                                "size_bytes": path.stat().st_size})
    return records


def materialize_automatic_phase_adoption(*, prior_runtime: Path) -> dict[str, Any]:
    """Select recorded completed boundaries, never infer success from a filename."""
    if not prior_runtime.is_absolute() or prior_runtime.is_symlink() or not prior_runtime.is_dir():
        raise AssetAuthoringError("astra_automatic_retained_root_invalid")
    authored = prior_runtime / "authoring"
    if not (authored / "source_analysis.json").exists() and (prior_runtime / "no_cost_authoring_adoption.json").exists():
        marker = json.loads((prior_runtime / "no_cost_authoring_adoption.json").read_text())
        derived = json.loads((authored / "result.json").read_text())
        source = derived.get("source_authoring_result") or {}
        original = Path(source.get("path", ""))
        if (marker.get("new_provider_calls") != 0 or marker.get("status") != "completed_authoring_adopted"
                or not original.is_relative_to(prior_runtime / "retained_astra_phases")
                or original.name != "result.json" or original.parent.name != "authoring"
                or file_record(original) != source):
            raise AssetAuthoringError("astra_automatic_result_lineage_invalid")
        authored = original.parent
    prior_request = json.loads((authored / "request.json").read_text())
    request = validate_request(prior_request)
    phases, artifacts = [], []
    if (authored / "source_analysis.json").exists():
        phases.append("source_analysis")
    if (authored / "cad_result.json").exists():
        artifacts.append("cad_result")
    elif (authored / "cad/node-node_geometric_architect.json").exists():
        phases.append("cad_state")
        if (authored / "cad/node-node_python_coder.json").exists():
            phases.append("coder")
    elif (authored / "cad").exists():
        raise AssetAuthoringError("astra_automatic_partial_cad_requires_verified_boundary")
    if (authored / "physical_property_review.json").exists():
        if "cad_result" not in artifacts:
            raise AssetAuthoringError("astra_automatic_physical_review_cad_missing")
        phases.append("physical_review")
    selected_round = 0
    for index in (0, 1):
        if (authored / f"appearance-{index:02d}/blender_author_{index}.json").exists():
            selected_round = index
    attempt = authored / f"appearance-{selected_round:02d}"
    if (attempt / f"blender_author_{selected_round}.json").exists():
        if "physical_review" not in phases:
            raise AssetAuthoringError("astra_automatic_blender_physical_review_missing")
        phases.append("blender_program")
        if (attempt / "geometry_readback.json").exists() and (attempt / "final_visual_mesh_receipt.json").exists():
            artifacts.append("blender_execution")
        if (attempt / f"independent_visual_review_{selected_round}.json").exists():
            if "blender_execution" not in artifacts:
                raise AssetAuthoringError("astra_automatic_visual_review_render_missing")
            artifacts.append("visual_review")
    if (authored / "result.json").exists():
        if "visual_review" not in artifacts:
            raise AssetAuthoringError("astra_automatic_authoring_review_missing")
        artifacts.append("authoring_result")
    if not phases or phases[0] != "source_analysis":
        raise AssetAuthoringError("astra_automatic_source_analysis_missing")
    value = {"schema_version": AUTOMATIC_SCHEMA_VERSION, "prior_runtime": str(prior_runtime),
        "phase_root_relative": str(authored.relative_to(prior_runtime)),
        "run_id": request.run_id, "source_request_digest": request.request_digest,
        "phases": phases, "completed_artifacts": artifacts, "blender_round": selected_round,
        "retained_files": _inventory(prior_runtime), "new_spend_authorized": False,
        "scientific_identity_relocated": False, "adoption_digest": ""}
    value["adoption_digest"] = canonical_digest(value, digest_field="adoption_digest")
    return value


def materialize_phase_adoption(*, prior_runtime: Path, phases: list[str],
                               blender_round: int = 0) -> dict[str, Any]:
    """Describe retained bytes for an admitted new stage configuration; no calls."""
    if (not prior_runtime.is_absolute() or prior_runtime.is_symlink()
            or not prior_runtime.is_dir() or not phases or len(phases) != len(set(phases))
            or not set(phases) <= PHASES or type(blender_round) is not int or blender_round not in (0, 1)
            or set(phases) != set(PHASE_ORDER[:len(phases)])):
        raise AssetAuthoringError("astra_phase_adoption_descriptor_invalid")
    prior_request = json.loads((prior_runtime / "authoring/request.json").read_text())
    request = validate_request(prior_request)
    value = {"schema_version": SCHEMA_VERSION, "prior_runtime": str(prior_runtime),
        "run_id": request.run_id, "source_request_digest": request.request_digest,
        "phases": sorted(phases), "blender_round": blender_round,
        "retained_files": _inventory(prior_runtime), "adoption_digest": ""}
    value["adoption_digest"] = canonical_digest(value, digest_field="adoption_digest")
    return value


def prepare_phase_adoption(*, value: Mapping[str, Any] | None, request_value: dict,
                           package: Path, budget_root: Path) -> dict[str, Any]:
    """Verify all old bytes, validate chosen outputs, then restore the full ledger."""
    if value is None:
        return {"authoring_kwargs": {}, "cad_kwargs": {}, "prior_call_count": 0}
    try:
        prior = Path(value["prior_runtime"])
        automatic = value.get("schema_version") == AUTOMATIC_SCHEMA_VERSION
        regenerated = (materialize_automatic_phase_adoption(prior_runtime=prior) if automatic
            else materialize_phase_adoption(prior_runtime=prior, phases=value["phases"], blender_round=value["blender_round"]))
        if dict(value) != regenerated or value["run_id"] != request_value["run_id"]:
            raise AssetAuthoringError("astra_phase_adoption_retained_bytes_changed")
        authoring_root = prior / value.get("phase_root_relative", "authoring")
        prior_request = json.loads((authoring_root / "request.json").read_text())
        def relevant(request):
            return {key: item for key, item in request.items()
                    if key not in {"request_digest", "expected_production_commit"}}
        if relevant(prior_request) != relevant(request_value):
            raise AssetAuthoringError("astra_phase_adoption_source_inputs_changed")
        for name in SOURCE_ARCHIVES:
            if file_record(prior / name)["sha256"] != file_record(package / name)["sha256"]:
                raise AssetAuthoringError("astra_phase_adoption_cad_sources_changed")
        old_budget = prior / "inference"
        old_audit = InferenceReservationAudit(run_root=old_budget, run_id=value["run_id"])
        manifest = old_audit.manifest()
        if manifest["in_flight_unknown_count"]:
            raise AssetAuthoringError("astra_phase_adoption_unresolved_inference")
        kwargs, cad_kwargs = {}, {}
        phases = value["phases"]
        for phase, verifier, output_key, record_key in (
            ("source_analysis", verify_source_analysis_adoption, "adopted_source_analysis", "adoption_record"),
            ("physical_review", verify_physical_review_adoption, "adopted_physical_review", "physical_adoption_record"),
            ("blender_program", verify_blender_program_adoption, "adopted_blender_program", "blender_adoption_record"),
        ):
            if phase in phases:
                extra = {"round_index": value["blender_round"]} if phase == "blender_program" else {}
                output, record = verifier(prior_root=authoring_root, request_value=request_value,
                                          budget_root=old_budget, **extra)
                kwargs[output_key], kwargs[record_key] = output, record
        if "cad_state" in phases:
            cad_kwargs = {"adopt_state_from": authoring_root / "cad", "adoption_budget_root": old_budget}
        if "coder" in phases:
            cad_kwargs["adopt_coder_from"] = authoring_root / "cad"
        retained_result = None
        if automatic:
            from .task_object_astra_retained_artifacts import (
                completed_authoring, completed_blender, completed_cad, completed_visual_review,
            )
            completed = value["completed_artifacts"]
            if "cad_result" in completed:
                cad, record = completed_cad(authoring_root, validate_request(prior_request))
                kwargs.update(adopted_cad_result=cad, cad_adoption_record=record)
                cad_kwargs = {}
            if "blender_execution" in completed:
                kwargs["adopted_blender_execution"] = completed_blender(authoring_root, validate_request(prior_request),
                    round_index=value["blender_round"], program=kwargs["adopted_blender_program"], cad=kwargs["adopted_cad_result"])
            if "visual_review" in completed:
                review, record = completed_visual_review(authoring_root, prior_request, old_budget,
                    round_index=value["blender_round"], execution=kwargs["adopted_blender_execution"])
                kwargs.update(adopted_visual_review=review, visual_adoption_record=record)
            if "authoring_result" in completed:
                from .task_object_astra_authoring import appearance_passed
                if not appearance_passed(kwargs["adopted_visual_review"]):
                    raise AssetAuthoringError("astra_automatic_completed_review_not_passed")
                retained_result = completed_authoring(authoring_root, prior_request)
        if "adoption_record" in kwargs:
            phase = kwargs["adoption_record"]["source_phase"]
            uri = Path(phase["path"]).resolve().as_uri()
            evidence_sha = phase["sha256"]
            previous_adoption = authoring_root / "source_analysis_adoption.json"
            if previous_adoption.exists():
                previous = json.loads(previous_adoption.read_text())
                alias = previous.get("source_evidence_identity")
                if alias is not None:
                    if previous.get("output_digest") != kwargs["adoption_record"]["output_digest"]:
                        raise AssetAuthoringError("astra_source_evidence_alias_output_mismatch")
                    uri, evidence_sha = alias["uri"], alias["sha256"]
            physical_path = authoring_root / "physical_review_input.json"
            if physical_path.exists():
                evidence = json.loads(physical_path.read_text()).get("evidence", [])
                matched = [row for row in evidence if row.get("evidence_id") == "source-appearance-analysis"]
                if matched:
                    if len(matched) != 1 or matched[0].get("sha256") != evidence_sha.removeprefix("sha256:"):
                        raise AssetAuthoringError("astra_source_evidence_alias_digest_mismatch")
                    uri = matched[0]["uri"]
            parsed = urlparse(uri)
            alias_path = Path(unquote(parsed.path))
            if (parsed.scheme != "file" or parsed.netloc not in ("", "localhost")
                    or file_record(alias_path)["sha256"] != evidence_sha):
                raise AssetAuthoringError("astra_source_evidence_alias_bytes_changed")
            kwargs["adoption_record"]["source_evidence_identity"] = {"uri": uri, "sha256": evidence_sha}
        if budget_root.exists():
            raise AssetAuthoringError("astra_phase_adoption_new_budget_root_required")
        # Replay every record through the production validator. Keeping every
        # previous reservation preserves costs and duplicate-call refusal.
        restored = InferenceReservationAudit(run_root=budget_root, run_id=value["run_id"])
        for record in manifest["reservations"]:
            restored.record_reservation(json.loads((old_budget / record["reservation_path"]).read_text()))
            restored.record_completion(json.loads((old_budget / record["completion_path"]).read_text()))
        current = restored.manifest()
        if current["reserved_max_cost_usd"] != manifest["reserved_max_cost_usd"]:
            raise AssetAuthoringError("astra_phase_adoption_inference_balance_changed")
        # Ship adopted evidence with this stage's outputs. The original request
        # bytes and paths remain provenance; no scientific identity is rewritten.
        snapshot = budget_root.parent / "retained_astra_phases"
        snapshot.mkdir(mode=0o700)
        for record in value["retained_files"]:
            source = prior / record["relative_path"]
            target = snapshot / record["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            with target.open("rb") as stream:
                copied_digest = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
            if copied_digest != record["sha256"]:
                raise AssetAuthoringError("astra_phase_adoption_copy_changed")
        for output_key in ("adoption_record", "physical_adoption_record", "blender_adoption_record"):
            if output_key in kwargs:
                for key in ("source_phase", "completed_provider_response"):
                    if key in kwargs[output_key]:
                        original = Path(kwargs[output_key][key]["path"])
                        kwargs[output_key][key] = file_record(snapshot / original.relative_to(prior))
        if cad_kwargs:
            cad_kwargs["adopt_state_from"] = snapshot / authoring_root.relative_to(prior) / "cad"
            cad_kwargs["adoption_budget_root"] = snapshot / "inference"
            if "adopt_coder_from" in cad_kwargs:
                cad_kwargs["adopt_coder_from"] = snapshot / authoring_root.relative_to(prior) / "cad"
        def remap_records(item):
            if isinstance(item, dict):
                if {"path", "sha256", "size_bytes"} <= set(item):
                    path = Path(item["path"])
                    if path.is_relative_to(prior):
                        return {**item, **file_record(snapshot / path.relative_to(prior))}
                return {key: remap_records(nested) for key, nested in item.items()}
            if isinstance(item, list):
                return [remap_records(nested) for nested in item]
            return item
        kwargs = remap_records(kwargs)
        if retained_result is not None:
            retained_result = remap_records(retained_result)
            retained_result.update(request_digest=request_value["request_digest"],
                source_authoring_result=file_record(snapshot / authoring_root.relative_to(prior) / "result.json"),
                retained_artifact_adoption_digest=value["adoption_digest"])
            retained_result["result_digest"] = canonical_digest(retained_result, digest_field="result_digest")
        return {"authoring_kwargs": kwargs, "cad_kwargs": cad_kwargs,
                "prior_call_count": manifest["reservation_count"],
                "adoption_digest": value["adoption_digest"],
                "retained_phase_root": str(snapshot),
                "completed_authoring_result": retained_result,
                "retained_inference_cost_usd": manifest["reserved_max_cost_usd"]}
    except (KeyError, TypeError, OSError, ValueError) as exc:
        if isinstance(exc, AssetAuthoringError):
            raise
        raise AssetAuthoringError("astra_phase_adoption_invalid") from exc
