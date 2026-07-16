"""Canonical, versioned site-type taxonomy shared across the pipeline.

Single source of truth for recognizing and classifying captured location types
(warehouse / manufacturing / cold storage / stockroom ... vs kitchen / home), so
that capture, ``episode_spec``, and ``robot_eval_dataset`` stop maintaining
separate brittle keyword lists, and so an unrecognized site type becomes an
explicit, surfaced state rather than a silent review-only fallback.

Design notes:
- The synonym map is deliberately substring-based (site-type text is short and
  human-authored), but resolution prefers the LONGEST matching synonym so that
  e.g. "distribution center" beats a stray token and "cold storage" beats a bare
  "storage"-adjacent match.
- ``INDUSTRIAL_CATEGORIES`` marks the humanoid-first deployment targets
  (warehouses/factories) that the beta prioritizes.
- Bump ``SITE_TAXONOMY_VERSION`` when categories/synonyms change so downstream
  artifacts can record which taxonomy produced a classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

SITE_TAXONOMY_VERSION = "v1"

UNKNOWN_SITE_CATEGORY = "unknown"

# Canonical category -> ordered synonym substrings (lowercase). Declaration order
# is the tie-breaker when two categories match synonyms of equal length.
SITE_CATEGORY_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "warehouse": (
        "distribution center",
        "distribution_center",
        "fulfillment center",
        "fulfillment",
        "fulfilment",
        "loading dock",
        "dock door",
        "cross dock",
        "cross-dock",
        "cross_dock",
        "sortation",
        "warehouse",
        "logistics",
        "forklift",
        "racking",
        "pallet",
        "rack",
        "aisle",
    ),
    "manufacturing": (
        "manufacturing",
        "production line",
        "assembly line",
        "line-side",
        "line side",
        "machine shop",
        "shop floor",
        "plant floor",
        "fabrication",
        "workcell",
        "work cell",
        "conveyor",
        "assembly",
        "factory",
    ),
    "cold_storage": (
        "cold storage",
        "cold_storage",
        "cold chain",
        "refrigerated",
        "freezer",
    ),
    "stockroom": (
        "stockroom",
        "stock room",
        "back of house",
        "backroom",
        "back room",
        "storeroom",
        "grocery",
    ),
    "lab": (
        "laboratory",
        "cleanroom",
        "clean room",
        "lab",
    ),
    "hospital": (
        "hospital",
        "healthcare",
        "patient",
        "nurse",
        "clinic",
        "ward",
    ),
    "retail": (
        "sales floor",
        "showroom",
        "storefront",
        "retail",
    ),
    "office": (
        "office",
        "cubicle",
        "workspace",
    ),
    "kitchen": (
        "kitchenette",
        "kitchen",
        "galley",
    ),
    "residential": (
        "living room",
        "residential",
        "apartment",
        "bedroom",
        "home",
        "house",
    ),
}

# Humanoid-first deployment targets prioritized for the beta.
INDUSTRIAL_CATEGORIES = frozenset(
    {"warehouse", "manufacturing", "cold_storage", "stockroom"}
)

# Default deterministic task-hint template per canonical category, used to
# propose a starter task when a capture's site type is recognized.
CATEGORY_DEFAULT_TASK_HINT: Dict[str, Tuple[str, str, str]] = {
    "warehouse": (
        "warehouse_tote_transfer",
        "Move a tote between staging and shelf zones",
        "tote_transfer",
    ),
    "manufacturing": (
        "factory_line_side_delivery",
        "Deliver an item to a line-side fixture",
        "line_side_delivery",
    ),
    "cold_storage": (
        "cold_storage_tote_transfer",
        "Move a tote through a cold-storage staging route",
        "tote_transfer",
    ),
    "stockroom": (
        "stockroom_bin_inspection",
        "Inspect labeled bins and staging shelves",
        "inspection_route",
    ),
    "kitchen": (
        "kitchen_counter_navigation",
        "Navigate around counter, sink, and appliance zones",
        "navigation",
    ),
    "lab": (
        "lab_bench_inspection",
        "Inspect a bench-side target zone",
        "inspection_route",
    ),
    "hospital": (
        "hospital_supply_delivery",
        "Deliver supplies through a constrained service route",
        "line_side_delivery",
    ),
}


@dataclass(frozen=True)
class SiteTypeResolution:
    """Result of resolving free-text site-type into the canonical taxonomy."""

    category: str
    matched_token: Optional[str]
    recognized: bool
    is_industrial: bool
    source_text: str
    taxonomy_version: str = SITE_TAXONOMY_VERSION


def canonical_site_categories() -> Tuple[str, ...]:
    """All canonical category ids, in declaration order."""

    return tuple(SITE_CATEGORY_SYNONYMS.keys())


def industrial_site_categories() -> Tuple[str, ...]:
    """Canonical categories considered humanoid-first industrial targets."""

    return tuple(c for c in SITE_CATEGORY_SYNONYMS if c in INDUSTRIAL_CATEGORIES)


def synonyms_for(category: str) -> Tuple[str, ...]:
    """Synonym tokens for a canonical category (empty tuple if unknown)."""

    return SITE_CATEGORY_SYNONYMS.get(category, ())


def default_task_hint_for_category(category: str) -> Optional[Dict[str, str]]:
    """Starter task-hint template for a recognized category, if defined."""

    hint = CATEGORY_DEFAULT_TASK_HINT.get(category)
    if hint is None:
        return None
    task_id, task_text, task_category = hint
    return {
        "task_id": task_id,
        "task_text": task_text,
        "task_category": task_category,
    }


def resolve_site_type(text: Optional[str]) -> SiteTypeResolution:
    """Resolve free-text site-type into a canonical category.

    Prefers the longest matching synonym across all categories; ties break on
    category declaration order. Returns an explicit unrecognized resolution when
    nothing matches (never guesses).
    """

    source = (text or "").strip()
    lowered = source.lower()
    if lowered:
        best: Optional[Tuple[Tuple[int, int], str, str]] = None
        for idx, (category, tokens) in enumerate(SITE_CATEGORY_SYNONYMS.items()):
            for token in tokens:
                if token in lowered:
                    key = (-len(token), idx)
                    if best is None or key < best[0]:
                        best = (key, category, token)
        if best is not None:
            _, category, token = best
            return SiteTypeResolution(
                category=category,
                matched_token=token,
                recognized=True,
                is_industrial=category in INDUSTRIAL_CATEGORIES,
                source_text=source,
            )
    return SiteTypeResolution(
        category=UNKNOWN_SITE_CATEGORY,
        matched_token=None,
        recognized=False,
        is_industrial=False,
        source_text=source,
    )
