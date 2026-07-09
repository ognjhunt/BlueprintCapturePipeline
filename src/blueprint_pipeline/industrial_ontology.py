"""Industrial readiness ontology helpers for qualification artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


_ENTITY_RULES: Dict[str, tuple[str, ...]] = {
    "aisle": ("aisle", "lane", "corridor", "passage"),
    "rack": ("rack", "shelf", "racking"),
    "tote": ("tote", "bin", "crate", "carton", "package", "container", "box"),
    "pallet_zone": ("pallet", "pallet zone", "staging pallet"),
    "forklift_lane": ("forklift", "fork truck", "lift lane"),
    "threshold": ("threshold", "curb", "lip", "step", "ramp"),
    "charger_candidate": ("charger", "charging", "dock"),
    "human_interaction_zone": ("operator", "human", "person", "workstation", "handover"),
    "handoff_point": ("handoff", "dropoff", "drop-off", "pick point", "staging"),
    "door_type": ("door", "gate", "rolling door", "dock door"),
    "floor_hazard": ("spill", "hazard", "debris", "cable", "uneven floor", "wet floor"),
    "traffic_zone": ("traffic", "crossing", "intersection", "shared zone"),
    "barrier": ("barrier", "bollard", "fence", "guardrail"),
    "workcell": ("workcell", "cell", "station", "bench"),
}

_ROUTE_RELEVANT = {
    "aisle",
    "forklift_lane",
    "threshold",
    "door_type",
    "traffic_zone",
    "barrier",
}

_TASK_RELEVANT = {
    "tote",
    "rack",
    "pallet_zone",
    "handoff_point",
    "charger_candidate",
    "human_interaction_zone",
    "workcell",
}

_HAZARD_RELEVANT = {
    "forklift_lane",
    "threshold",
    "floor_hazard",
    "traffic_zone",
    "barrier",
    "human_interaction_zone",
}


def _normalized_entity_text(text: Any) -> str:
    return str(text or "").strip().lower().replace("-", " ").replace("_", " ")


@dataclass(frozen=True)
class IndustrialEntity:
    entity_type: str
    matched_tokens: List[str]
    route_relevant: bool
    task_relevant: bool
    hazard_relevant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "matched_tokens": list(self.matched_tokens),
            "route_relevant": self.route_relevant,
            "task_relevant": self.task_relevant,
            "hazard_relevant": self.hazard_relevant,
        }


def _normalized_tokens(text: str) -> List[str]:
    lowered = _normalized_entity_text(text)
    if not lowered:
        return []
    return [token for token in lowered.split() if token]


def classify_industrial_entity(label: Any) -> IndustrialEntity:
    normalized = _normalized_entity_text(label)
    tokens = _normalized_tokens(normalized)
    matched_type = "object"
    matched_tokens: List[str] = []

    for entity_type, candidates in _ENTITY_RULES.items():
        found = [candidate for candidate in candidates if candidate in normalized]
        if found:
            matched_type = entity_type
            matched_tokens = found
            break

    return IndustrialEntity(
        entity_type=matched_type,
        matched_tokens=matched_tokens or tokens[:2],
        route_relevant=matched_type in _ROUTE_RELEVANT,
        task_relevant=matched_type in _TASK_RELEVANT,
        hazard_relevant=matched_type in _HAZARD_RELEVANT,
    )


def classify_industrial_entities(values: Iterable[Any]) -> List[IndustrialEntity]:
    """Return every ontology entity matched across labels or ids.

    ``classify_industrial_entity`` intentionally returns one best label for
    legacy qualification call sites. Scenario and task-target lanes need the
    full static scene structure context, such as rack + tote + pallet zone.
    """

    entities: List[IndustrialEntity] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized_entity_text(value)
        if not normalized:
            continue
        for entity_type, candidates in _ENTITY_RULES.items():
            found = [candidate for candidate in candidates if candidate in normalized]
            if not found or entity_type in seen:
                continue
            seen.add(entity_type)
            entities.append(
                IndustrialEntity(
                    entity_type=entity_type,
                    matched_tokens=found,
                    route_relevant=entity_type in _ROUTE_RELEVANT,
                    task_relevant=entity_type in _TASK_RELEVANT,
                    hazard_relevant=entity_type in _HAZARD_RELEVANT,
                )
            )
    return entities


def industrial_tags_for_label(label: Any) -> List[str]:
    entity = classify_industrial_entity(label)
    tags = [entity.entity_type]
    if entity.route_relevant:
        tags.append("route_relevant")
    if entity.task_relevant:
        tags.append("task_relevant")
    if entity.hazard_relevant:
        tags.append("hazard_relevant")
    return tags


def derive_capture_plan_tags(values: Iterable[Any]) -> List[str]:
    tags: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        for entity in classify_industrial_entities([text]) or [classify_industrial_entity(text)]:
            if entity.entity_type not in tags:
                tags.append(entity.entity_type)
    return tags
