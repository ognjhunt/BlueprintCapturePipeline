"""Digest-bound same-run Astra phase adoption without resetting paid inference.

This deliberately retains the existing exact source-path and run identity
predicates. Relocating evidence or adopting another run needs a separate contract.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
from .task_object_astra_authoring import AssetAuthoringError, file_record, validate_request
from .task_object_astra_worker import (
    verify_blender_program_adoption,
    verify_physical_review_adoption,
    verify_source_analysis_adoption,
)

SCHEMA_VERSION = "task_evaluation_astra_phase_adoption.v1"
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
            if (relative.parts[0] == "authoring" or str(relative) in SOURCE_ARCHIVES
                    or relative.parts[:2] == ("inference", "inference_reservations")):
                record = file_record(path)
                records.append({"relative_path": str(relative), "sha256": record["sha256"],
                                "size_bytes": record["size_bytes"]})
    return records


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
        regenerated = materialize_phase_adoption(prior_runtime=prior, phases=value["phases"],
                                                blender_round=value["blender_round"])
        if dict(value) != regenerated or value["run_id"] != request_value["run_id"]:
            raise AssetAuthoringError("astra_phase_adoption_retained_bytes_changed")
        prior_request = json.loads((prior / "authoring/request.json").read_text())
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
                output, record = verifier(prior_root=prior / "authoring", request_value=request_value,
                                          budget_root=old_budget, **extra)
                kwargs[output_key], kwargs[record_key] = output, record
        if "cad_state" in phases:
            cad_kwargs = {"adopt_state_from": prior / "authoring/cad", "adoption_budget_root": old_budget}
        if "coder" in phases:
            cad_kwargs["adopt_coder_from"] = prior / "authoring/cad"
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
            if file_record(target)["sha256"] != record["sha256"]:
                raise AssetAuthoringError("astra_phase_adoption_copy_changed")
        for output_key in ("adoption_record", "physical_adoption_record", "blender_adoption_record"):
            if output_key in kwargs:
                for key in ("source_phase", "completed_provider_response"):
                    if key in kwargs[output_key]:
                        original = Path(kwargs[output_key][key]["path"])
                        kwargs[output_key][key] = file_record(snapshot / original.relative_to(prior))
        if cad_kwargs:
            cad_kwargs["adopt_state_from"] = snapshot / "authoring/cad"
            cad_kwargs["adoption_budget_root"] = snapshot / "inference"
            if "adopt_coder_from" in cad_kwargs:
                cad_kwargs["adopt_coder_from"] = snapshot / "authoring/cad"
        return {"authoring_kwargs": kwargs, "cad_kwargs": cad_kwargs,
                "prior_call_count": manifest["reservation_count"],
                "adoption_digest": value["adoption_digest"],
                "retained_phase_root": str(snapshot),
                "retained_inference_cost_usd": manifest["reserved_max_cost_usd"]}
    except (KeyError, TypeError, OSError, ValueError) as exc:
        if isinstance(exc, AssetAuthoringError):
            raise
        raise AssetAuthoringError("astra_phase_adoption_invalid") from exc
