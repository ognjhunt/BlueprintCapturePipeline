# Covers (for impacted-test selection):
#   src/blueprint_pipeline/atlas_scene_planner.py
#   src/blueprint_pipeline/atlas_automation_plan.py
#   src/blueprint_pipeline/website_image_completion.py
#   src/blueprint_pipeline/website_image_repair_agent.py
#   src/blueprint_pipeline/clean_plate_removal_analysis_gemini.py
#   src/blueprint_pipeline/website_task_semantics.py
#   src/blueprint_pipeline/website_operation_coverage.py
#   src/blueprint_pipeline/atlas_source_gap_evidence.py
#   src/blueprint_pipeline/atlas_scene_task_regions.py
#   src/blueprint_pipeline/atlas_scene_surface_masks.py
#   src/blueprint_pipeline/atlas_recorded_source_recovery.py
#   src/blueprint_pipeline/task_evaluation_robot_placement_agent.py
#   src/blueprint_pipeline/atlas_scene_chisel.py
#   src/blueprint_pipeline/atlas_scene_generate.py
#   src/blueprint_pipeline/atlas_scene_preservation.py
#   src/blueprint_pipeline/atlas_removal_visibility.py
#   src/blueprint_pipeline/atlas_whole_view_review.py
#   src/blueprint_pipeline/website_candidate_ranking.py
#   src/blueprint_pipeline/website_task_masks.py
#   src/blueprint_pipeline/website_task_grounding.py
"""Owner rule 2026-10-29: one general harness for every site, task and object.

The model prompts on the website capture, Atlas and scene preparation path name
roles and relations (task object, support, destination, fixture, obstacle,
person, floor, opening), never one kind of room or object. A named example
primes the model: on capture-4b60431e the image editor erased the dishwasher
behind a tray because its standing prompt used a dishwasher as its example of a
built-in task object.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src/blueprint_pipeline"
HARNESS_PROMPT_MODULES = (
    "atlas_scene_planner", "atlas_automation_plan", "website_image_completion", "website_image_repair_agent",
    "clean_plate_removal_analysis_gemini", "website_task_semantics", "website_operation_coverage",
    "atlas_source_gap_evidence", "atlas_scene_task_regions", "atlas_scene_surface_masks",
    "atlas_recorded_source_recovery", "task_evaluation_robot_placement_agent", "atlas_scene_chisel",
    "atlas_scene_generate", "atlas_scene_preservation", "atlas_removal_visibility", "atlas_whole_view_review",
    "website_candidate_ranking",
    # Writes no prompt of its own: every SAM concept it sends comes from the
    # plan or the grounding model. Scanned so a fixed concept cannot return; it
    # held one ("under-desk cabinet", chosen from task words) until 2026-10-31.
    "website_task_masks",
    # Its segmentation-concept instruction gave "blue case" and "white book" as
    # examples until 2026-10-31; it now names the role and the kind of attribute.
    "website_task_grounding",
)
# Not scanned yet, because their strings include lookup data or classifier
# vocabulary rather than only instructions: atlas_scene_measurements (standard
# furniture heights for scale), website_object_spec_research (articulated-part
# words used to read specs) and website_assembly_coverage (part names its
# articulated-assembly classifier matches). Each needs its own redesign.
PARTICULAR = re.compile(
    r"\b(?:dishwashers?|ovens?|stoves?|refrigerators?|fridges?|freezers?|microwaves?|sinks?|faucets?|"
    r"kitchens?|bathrooms?|bedrooms?|cupboards?|cabinets?|cabinetry|countertops?|worktops?|drawers?|desks?|"
    r"backpacks?|chairs?|sofas?|couch(?:es)?|beds?|totes?|trays?|vases?|mugs?|cups?|bottles?|dish(?:es)?|"
    r"balls?|cartons?|cardboard|tables?|counters?|lids?|doors?|shelf|shelves|appliances?|"
    r"books?|(?:brief|suit)?cases?)\b", re.IGNORECASE)
# "box" is left out on purpose: every grounding prompt asks for a bounding box.

# The one deliberate list: standardized dimensions that any kind of site may
# show, so a scale estimate never has to lean on the task object itself.
SCALE_CATALOGUE = ("doorway or door leaf, a stair riser, a ceiling tile, a work surface height,\n"
                   "a standard pallet, a rack or shelving upright, a\nparking bay, or a built-in appliance.")


def _prompt_strings(module):
    tree = ast.parse((ROOT / f"{module}.py").read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
    return [node.value.replace(SCALE_CATALOGUE, "") for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings]


@pytest.mark.parametrize("module", HARNESS_PROMPT_MODULES)
def test_harness_prompts_name_roles_never_a_particular_room_or_object(module):
    found = sorted({match.group(0).lower() for text in _prompt_strings(module) for match in PARTICULAR.finditer(text)})
    assert found == []


def test_the_scale_catalogue_is_the_only_exception_and_spans_kinds_of_site():
    from blueprint_pipeline.atlas_automation_plan import AUTOMATION_INSTRUCTIONS

    assert SCALE_CATALOGUE in AUTOMATION_INSTRUCTIONS
    assert "homes, offices, shops, warehouses, factories" in " ".join(AUTOMATION_INSTRUCTIONS.split())
    assert PARTICULAR.findall(AUTOMATION_INSTRUCTIONS.replace(SCALE_CATALOGUE, "")) == []
