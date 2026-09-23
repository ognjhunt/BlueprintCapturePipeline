"""CPU tools shared by a persistent asset-authoring agent.

The author proposes geometry and appearance; deterministic readback and a
separate reviewer own acceptance. No tool chooses a robot or starts a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_object_astra_authoring import (
    AppearanceReview, AssetAuthoringError, BlenderProgram, SourceFrame, VisualBrief,
    appearance_passed, file_record, invoke_vision, save_json, validate_blender_program,
    validate_geometry_readback, validate_request,
)
from .task_object_physical_property_review import (
    EvidenceReference, PhysicalPropertyReviewProposal, build_physical_property_review_prompt,
    review_physical_properties,
)


APPEARANCE_SCOPE = "observable_v2"


class AssetTools:
    def __init__(self, *, request_value, output_root, cad_executor, blender_runner, blender_executable):
        self.request = validate_request(request_value)
        self.root = Path(output_root)
        self.root.mkdir(parents=True, exist_ok=True)
        if any(p.name not in {"tmp", "xdg", "cache"} for p in self.root.iterdir()):
            raise AssetAuthoringError("authoring_output_already_used")
        self.cad_executor, self.blender_runner, self.blender_executable = cad_executor, blender_runner, blender_executable
        blender_runner.preflight()
        save_json(self.root / "request.json", request_value)
        self.brief = self.cad = self.candidate = None
        self.retained_physics = self.source_evidence_identity = None
        self.retained_visual_review = None
        self.cad_attempts = self.render_attempts = 0

    def observe_object(self, brief):
        """Record the author's interpretation; original image bytes remain authoritative."""
        self.brief = VisualBrief.model_validate(brief)
        save_json(self.root / "source_analysis.json", {"model": "gpt-6-sol", "provider": "openai",
            "request_digest": self.request.request_digest,
            "references": [f.model_dump(mode="json") for f in self.request.source_frames],
            "output": self.brief.model_dump(mode="json"), "origin": "asset_authoring_session"})
        (self.root / "CAD_BRIEF.md").write_text(self.brief.cad_brief_markdown + "\n")
        # A revised interpretation invalidates previously reviewed candidates.
        self.cad = self.candidate = None
        self.retained_physics = self.source_evidence_identity = None
        self.retained_visual_review = None
        return {"status": "recorded", "dimension_authority": self.request.dimension_authority}

    def build_cad(self, program):
        if self.brief is None:
            raise AssetAuthoringError("source_interpretation_required")
        if self.cad_attempts >= 6:
            raise AssetAuthoringError("cad_tool_attempt_limit")
        attempt = self.root / f"cad-{self.cad_attempts:02d}"
        self.cad_attempts += 1
        self.cad = self.candidate = None
        self.retained_physics = None
        self.retained_visual_review = None
        self.cad = self.cad_executor(program=program, output_root=attempt, request=self.request)
        save_json(self.root / "cad_result.json", self.cad)
        return {"status": "built", "readback": self.cad["readback"]}

    def render_candidate(self, program):
        if self.cad is None or self.brief is None:
            raise AssetAuthoringError("validated_cad_required")
        if self.render_attempts >= 6:
            raise AssetAuthoringError("render_tool_attempt_limit")
        parsed = BlenderProgram.model_validate(program)
        validate_blender_program(parsed.program)
        attempt = self.root / f"appearance-{self.render_attempts:02d}"
        self.render_attempts += 1
        self.candidate = None
        self.retained_physics = None
        self.retained_visual_review = None
        attempt.mkdir()
        save_json(attempt / "blender_program.json", parsed.model_dump(mode="json"))
        (attempt / "asset_program.py").write_text(parsed.program)
        stl = Path(self.cad["stl"]["path"])
        if file_record(stl) != self.cad["stl"]:
            raise AssetAuthoringError("authoring_cad_export_changed")
        shutil.copyfile(stl, attempt / "candidate.stl")
        references = []
        for index, frame in enumerate(self.request.source_frames):
            if file_record(Path(frame.path))["sha256"] != frame.sha256:
                raise AssetAuthoringError("authoring_source_image_changed")
            target = attempt / f"reference_{index:02d}.png"
            shutil.copyfile(frame.path, target)
            references.append(target.name)
        save_json(attempt / "render_inputs.json", {"dimensions_m": self.request.dimensions_m,
            "reference_files": references, "cad_units": "millimetres"})
        from . import task_object_blender_runtime
        completed = self.blender_runner([self.blender_executable, "--background", "--factory-startup",
            "--python-exit-code", "23", "--python", str(Path(task_object_blender_runtime.__file__).resolve()),
            "--", str(attempt)], cwd=attempt, timeout=600, check=False, capture_output=True, text=True)
        (attempt / "blender.stdout.txt").write_text((completed.stdout or "")[-100000:])
        (attempt / "blender.stderr.txt").write_text((completed.stderr or "")[-100000:])
        if completed.returncode:
            raise AssetAuthoringError("Blender execution failed:\n" + (completed.stderr or "")[-6000:]
                                     + (completed.stdout or "")[-2000:])
        measurement = json.loads((attempt / "geometry_readback.json").read_text())
        validate_geometry_readback(self.request, measurement, self.brief.proposed_appearance)
        # Check all outputs before allowing inspection/review or acceptance.
        artifacts = {name: file_record(attempt / name) for name in (
            "candidate.usdc", "candidate.blend", "final_visual_mesh.json", "final_visual_mesh_receipt.json",
            "geometry_readback.json", "perspective.png", "top.png", "side.png")}
        self.candidate = {"directory": attempt, "artifacts": artifacts,
            "cad_digest": canonical_digest(self.cad), "brief_digest": canonical_digest(self.brief.model_dump(mode="json"))}
        return {"status": "rendered_pending_independent_review", "measurement": measurement,
                "views": ["perspective", "top", "side"], "artifacts": artifacts}

    def candidate_frames(self):
        if self.candidate is None:
            raise AssetAuthoringError("rendered_candidate_required")
        result = []
        for view in ("perspective", "top", "side"):
            record = self.candidate["artifacts"][view + ".png"]
            if file_record(Path(record["path"])) != record:
                raise AssetAuthoringError("candidate_changed_before_review")
            result.append(SourceFrame(path=record["path"], sha256=record["sha256"],
                role="prior_candidate", description=f"Generated candidate {view} studio render"))
        return result

    def independent_review(self, invoker):
        """Called by the controller adapter, never replaceable with an author's verdict."""
        frames = self.candidate_frames()
        self.validate_candidate()
        physical = self.request.physical_review_input.model_copy(deep=True)
        if physical.appearance == "unknown":
            physical.appearance = self.brief.proposed_appearance
            physical.material_description = self.brief.proposed_material
            record = file_record(self.root / "source_analysis.json")
            identity = self.source_evidence_identity or dict(uri=Path(record["path"]).as_uri(),
                sha256=record["sha256"].removeprefix("sha256:"))
            physical.evidence.append(EvidenceReference(evidence_id="source-appearance-analysis",
                uri=identity['uri'], sha256=identity['sha256'],
                kind="material_observation", excerpt="Candidate interpretation: " + canonical_json({
                    "material": self.brief.proposed_material, "appearance": self.brief.proposed_appearance})))
        save_json(self.root / "physical_review_input.json", physical.model_dump(mode="json"))
        if self.retained_physics is not None:
            retained_input, retained_proposal = self.retained_physics
            if physical.model_dump(mode='json') != retained_input:
                raise AssetAuthoringError('agent_resume_physics_inputs_changed')
            proposal = PhysicalPropertyReviewProposal.model_validate(retained_proposal)
        else:
            proposal = invoke_vision(invoker, self.request, capability=f"physical_property_review_{self.render_attempts}",
                prompt=build_physical_property_review_prompt(physical) + "\nConstruction constraints: "
                    + self.request.construction_constraints + "\nCAD readback (mm, mm3): " + canonical_json(self.cad["readback"]),
                output_type=PhysicalPropertyReviewProposal, frames=self.request.source_frames, root=self.root)
        self.retained_physics = (physical.model_dump(mode="json"), proposal.model_dump(mode="json"))
        physics = review_physical_properties(physical, proposal)
        save_json(self.root / "physical_property_review_result.json", physics.model_dump(mode="json"))
        if physics.accepted is None:
            return {"accepted": False, "blockers": list(physics.blockers)}
        attempt = self.candidate["directory"]
        context = self.request.model_dump(mode="json")
        context.pop("source_frames")
        review = (AppearanceReview.model_validate(self.retained_visual_review) if self.retained_visual_review is not None else
            invoke_vision(invoker, self.request, capability=f"independent_visual_review_{self.render_attempts}_{APPEARANCE_SCOPE}",
            prompt="Independently compare these studio renders with the ORIGINAL source images and task specification. "
            "Check required parts, shape, color, opacity and texture. Generated variants may differ only as specified; "
            "set requested_specification_satisfied for generated objects. Report actionable corrections. "
            "Assess observable appearance only: native USD import, physics, exact dimensions and scene placement "
            "are checked independently and missing proof of them is not an appearance defect. "
            "These are uncalibrated studio views, so perspective alone does not establish a shape mismatch. "
            "Reject visible contradictions; identify their source-image evidence. Record occluded or blurred "
            "surfaces as limitations, not observed defects. Never infer physical truth from plausible renders.\n" + canonical_json(context),
            output_type=AppearanceReview, frames=self.request.source_frames + frames, root=attempt))
        self.retained_visual_review = review.model_dump(mode="json")
        if not appearance_passed(review, generated=self.request.generated_specification is not None):
            return {"accepted": False, "review": review.model_dump(mode="json")}
        self.validate_candidate()
        generated = self.request.generated_specification
        result = {"schema_version": "task_object_astra_authoring_result.v1",
            "status": "candidate_authored_pending_native_qualification", "model": "gpt-6-sol",
            "request_digest": self.request.request_digest, "object_id": self.request.object_id,
            "claim_ceiling": "development_only", "asset": file_record(attempt / "candidate.usdc"),
            "blend": file_record(attempt / "candidate.blend"), "cad": self.cad,
            "geometry_readback": file_record(attempt / "geometry_readback.json"),
            "final_visual_mesh": file_record(attempt / "final_visual_mesh.json"),
            "final_visual_mesh_receipt": file_record(attempt / "final_visual_mesh_receipt.json"),
            "physical_review": file_record(self.root / "physical_property_review_result.json"),
            "physical_review_input": file_record(self.root / "physical_review_input.json"),
            "review_images": [file_record(Path(f.path)) for f in frames],
            "native_import_qualified": False, "scene_placement_qualified": False, "physical_equivalence_proven": False,
            "asset_origin": ("generated_variant" if generated.variant_of else "generated_task_object")
                            if generated else "captured_object_reconstruction",
            "generated_specification": generated.model_dump(mode="json") if generated else None}
        result["result_digest"] = canonical_digest(result)
        save_json(self.root / "result.json", result)
        return {"accepted": True, "result": result}

    def validate_candidate(self):
        if (self.candidate is None or self.candidate["cad_digest"] != canonical_digest(self.cad)
                or self.candidate["brief_digest"] != canonical_digest(self.brief.model_dump(mode="json"))):
            raise AssetAuthoringError("candidate_inputs_changed")
        for record in [*self.candidate["artifacts"].values(), self.cad["step"], self.cad["stl"]]:
            if file_record(Path(record["path"])) != record:
                raise AssetAuthoringError("candidate_changed_during_review")
