"""Pure native task-profile metadata; no robot or policy inventory imports.

Profiles describe existing adapter/scorer contracts, never an intake allowlist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .task_family_scoring import FAMILY_ALIASES, canonical_task_family


@dataclass(frozen=True)
class TaskProfile:
    profile_id: str
    native_task_kind: str
    catalog_task_family_id: str
    runtime_adapter: str
    success_predicate: str
    aliases: tuple[str, ...]
    required_inputs: tuple[str, ...]
    #: What a robot must be able to do (``task_spec_compiler.requirements``); robot
    #: and policy choice read this, never the profile id.
    requires_locomotion: bool = False
    requires_manipulation: bool = True


# First supported profiles. Each names the native adapter that compiles it and
# the success predicate the independent scorer reads.
TASK_PROFILES: tuple[TaskProfile, ...] = (
    TaskProfile(
        profile_id="rigid_pick_place",
        native_task_kind="rigid_pick_place",
        catalog_task_family_id="rigid_relocation",
        runtime_adapter="task_evaluation_rigid_relocation_native_adapter",
        success_predicate="object_pose_within_target_region",
        aliases=("rigid_pick_place", "rigid_pick_and_place", "rigid_object_pick_place", "rigid_object_relocation",
                 "rigid_relocation"),
        required_inputs=("metric_object_geometry", "target_region"),
    ),
    TaskProfile(
        profile_id="articulated_open_close",
        native_task_kind="articulated_open_close",
        catalog_task_family_id="articulated_open_close",
        runtime_adapter="task_evaluation_articulated_open_close_native_adapter",
        success_predicate="target_joint_within_qualified_interval",
        aliases=("articulated_open_close",),
        required_inputs=("qualified_articulation_graph",),
    ),
    *(TaskProfile(
        profile_id=family,
        native_task_kind=family,
        catalog_task_family_id=family,
        runtime_adapter="task_evaluation_task_family_native_adapter",
        success_predicate="independent_native_state_stage_predicates",
        aliases=aliases,
        required_inputs=("task_success_contract", "native_scene_assets", "native_metric_bindings",
                         "reset_contract", "native_runtime_contract"),
    ) for family, aliases in FAMILY_ALIASES.items()),
    # Moving the robot itself to a goal region, clear of the scene: the scene is kept as
    # filmed (no manipulated subject) and success is the compiled spec's predicates
    # (``task_spec_compiler``: the root reaches and holds the goal, no obstacle contact).
    TaskProfile(
        profile_id="goal_region_navigation",
        native_task_kind="goal_region_navigation",
        catalog_task_family_id="goal_region_navigation",
        runtime_adapter="task_spec_compiler",
        success_predicate="compiled_task_predicates",
        aliases=("goal_region_navigation", "navigation", "navigate", "point_to_point_navigation",
                 "mobile_navigation", "locomotion", "walk_to_goal"),
        required_inputs=("scene_collider", "start_and_goal_anchor_or_filmed_path"),
        requires_locomotion=True,
        requires_manipulation=False,
    ),
)


def task_profile_for(task_family: str | None) -> TaskProfile | None:
    value = (task_family or "").strip().lower()
    return next((row for row in TASK_PROFILES if value in row.aliases), None)


def required_task_families(family: str, task_semantics: Mapping[str, Any] | None) -> list[str]:
    """The families one runtime must register: a multi-step job's stages too.

    Here, not in ``website_early_task_hold``, so the hold contract that re-derives it
    (``website_task_context_contracts``) stays off the scene-preparation import graph.
    """
    families = {family}
    if family == "multi_step" and task_semantics is not None:
        clarification = task_semantics.get("operation_coverage_clarification")
        reading = clarification["operation_coverage"] if clarification else task_semantics["semantics"]
        for step in reading["steps"]:
            profile = task_profile_for(step["operation"])
            stage = canonical_task_family(profile.profile_id) if profile is not None else None
            if stage is not None:
                families.add(stage)
    return sorted(families)


#: Soft objects a rigid body cannot stand in for: the backstop when a task object carries no
#: rigidity classification (``task_object_rigidity``). Matched only as the head noun of the
#: manipulated object's own label ("folded bath towel", "stack of towels"), never anywhere
#: in the task text: "put the mug on the towel" moves a mug, and a "towel rack" is a rack.
DEFORMABLE_SUBJECT_NOUNS = frozenset({
    "towel", "cloth", "clothes", "clothing", "garment", "shirt", "t-shirt", "tshirt", "blouse", "sweater",
    "hoodie", "jacket", "coat", "pants", "trousers", "jeans", "shorts", "skirt", "dress", "sock", "scarf",
    "glove", "apron", "uniform", "napkin", "rag", "washcloth", "dishcloth", "blanket", "duvet", "comforter",
    "quilt", "bedsheet", "pillowcase", "linen", "linens", "fabric", "textile", "pillow", "cushion", "curtain",
    "tarp", "tarpaulin", "bag", "sack", "pouch", "rope", "cord", "cable", "wire", "hose", "string", "twine",
    "yarn", "ribbon", "strap", "belt", "sponge", "dough", "laundry",
})
#: A label stops naming its object at the first relation ("towel on the rack", "bag of rice").
_LABEL_RELATION = frozenset({"on", "in", "into", "inside", "onto", "at", "from", "near", "under", "beside", "by",
                             "next", "with", "behind", "above", "below", "against", "over", "to", "for", "that",
                             "which", "of", "and", "or"})
#: "A stack of towels" names the towels.
_COLLECTIVE = frozenset({"stack", "pile", "heap", "bundle", "armful", "load", "handful", "bunch"})


def _deformable_noun(word: str) -> str | None:
    for candidate in (word, word[:-2] if word.endswith("es") else None, word[:-1] if word.endswith("s") else None):
        if candidate in DEFORMABLE_SUBJECT_NOUNS:
            return candidate
    return None


def deformable_subject_noun(label: object) -> str | None:
    """The soft-object head noun of a task object's label, or None. A text screen, never a measurement."""
    if not isinstance(label, str):
        return None
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", re.split(r"[,;:(]", label.lower(), maxsplit=1)[0])
    end = next((index for index, word in enumerate(words) if word in _LABEL_RELATION), len(words))
    head = words[end - 1] if end else None
    if head in _COLLECTIVE and end < len(words) and words[end] == "of":
        rest = words[end + 1:]
        stop = next((index for index, word in enumerate(rest) if word in _LABEL_RELATION), len(rest))
        head = rest[stop - 1] if stop else None
    return _deformable_noun(head) if head else None


#: The clean-plate object analysis's own reading of each target
#: (``clean_plate_removal_response_schema``): whether it keeps its shape when handled.
RIGIDITY_CLASSES = ("rigid", "deformable", "unknown")
RIGIDITY_UNKNOWN_FLAG = "task_object_rigidity_unknown"
#: Which source decided, as the deformable hold records it.
RIGIDITY_BASIS = {"object_analysis": "manipulated_target_rigidity_classification",
                  "label_word_backstop": "manipulated_target_label_text_screen"}


def task_object_rigidity(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a manipulated object is rigid, deformable or unknown, and which source decided.

    The object analysis decides: its ``rigidity`` class and its stated ``rigidity_reason``.
    An analysis bought before it was asked carries neither, and only then does the label's
    head noun decide (``deformable_subject_noun``): a soft noun reads deformable, anything
    else unknown. Unknown never holds a job; it is flagged. The word screen is recorded
    either way. A model's reading, never a measurement.
    """
    label = entry.get("semantic_label")
    label = label if isinstance(label, str) and label.strip() else None
    noun = deformable_subject_noun(label)
    stated = entry.get("rigidity")
    if stated in RIGIDITY_CLASSES:
        reason = entry.get("rigidity_reason")
        reading = {"rigidity": stated, "source": "object_analysis", "field": "rigidity",
                   "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None}
    else:
        reading = {"rigidity": "deformable" if noun else "unknown", "source": "label_word_backstop",
                   "field": "semantic_label", "reason": None}
    return {"target_id": str(entry.get("target_id")), "label": label, **reading, "label_word_screen": noun,
            **({"flag": RIGIDITY_UNKNOWN_FLAG} if reading["rigidity"] == "unknown" else {})}
