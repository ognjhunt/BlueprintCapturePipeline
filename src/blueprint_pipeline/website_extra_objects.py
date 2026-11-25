"""The free source each object the build plan would build after its first could use: a record, and nothing more.

A job builds one object today, the build plan's most important one it can build (``website_build_plan.built_object``).
The plan ranks up to eight, and ``built_objects`` lists every one it would build, top-ranked first. For each object
after the first, ``extra_object_sources`` works out the free source it could use, in the owner's order (owner
direction 2026-11-23): an existing licensed library model first, then a free parametric shape. Anything else needs the
paid generated model (ARTISAN, at its per-object cap: $12 today). It reserves nothing, spends nothing, asks no model
and fetches nothing: it reads the library pick sealed for that object in a directory of its own target
(``object_root``: several objects' picks never share one record) and the plan's row. Nothing on the live path calls
it yet; every job still builds its first object alone.

An extra object counts as free only when it has either

- a sealed, fetched library model the deployed catalog still allows, with a parametric shape to fall back on: compile
  refuses a model whose proportions miss the object's estimated bounds (``website_library_asset.fit``) and builds the
  shape instead; or
- a parametric shape that is not a cylinder. A carton, tote, open bin or tray is made to the estimated bounds
  themselves; a can's diameter is its footprint's mean side, which misses the size tolerance on an oblong footprint
  (``website_supplied_mesh``), and then the object goes to ARTISAN.

A library model whose fallback is a cylinder counts as free by that rule, flagged (``CYLINDER_FALLBACK``): the two
can miss together. An object that opens is never free: supplied geometry is one rigid body. Every other object is
marked as needing ARTISAN at its cap, never reserved here: an extra object's reservation is
``website_build_budget.admit_extra_objects``'s, which reads compile's own configuration of the object, not this record.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .website_build_plan import built_objects
from .website_parametric_object import SHAPES

SCHEMA_VERSION = "website_extra_object_sources.v1"
REVISION = "website_extra_objects.r1"
#: Where each extra object's library pick is sealed, one directory per target, under the job's library root.
EXTRA_OBJECTS = "extra_objects"
LIBRARY, PARAMETRIC, ARTISAN = "library", "parametric", "artisan"
#: The one parametric shape that can miss the object's estimated bounds (``website_parametric_object.design``).
CYLINDER = "cylinder"
#: Why a library model is not counted; the library's own reason (``not_used``, ``rights_now``) is kept as it reads.
NOT_PICKED = "website_extra_object_library_not_picked"
READING_CHANGED = "website_extra_object_library_reading_changed"
NOT_FETCHED = "website_extra_object_library_not_fetched"
FILES_CHANGED = "library_file_changed"
#: Why an object needs ARTISAN.
OPENS = "website_extra_object_opens"
LIBRARY_WITHOUT_FALLBACK = "website_extra_object_library_without_fallback_shape"
CYLINDER_ONLY = "website_extra_object_cylinder_can_miss_size_tolerance"
NO_FREE_SOURCE = "website_extra_object_no_free_source"
#: A free library model whose fallback is a cylinder: both can miss the object's bounds together.
CYLINDER_FALLBACK = "website_extra_object_cylinder_fallback_can_miss_size_tolerance"


def object_root(library_root: Path, target_id: str) -> Path:
    """The directory an extra object's library pick is sealed in: its own, keyed by its target."""
    return Path(library_root) / EXTRA_OBJECTS / ("target-" + hashlib.sha256(str(target_id).encode()).hexdigest()[:16])


def sealed_library_model(row: Mapping[str, Any], *, root: Path, task_context: Mapping[str, Any],
                         catalog_entries: Sequence[Mapping[str, Any]] | None = None
                         ) -> tuple[dict[str, Any] | None, str | None]:
    """The library model sealed for this reading of the object at ``root`` with its files fetched, or None and why:
    no pick sealed, one for another reading of the object, the library's own reason not to use one, files never
    fetched or changed since, or the deployed catalog (``catalog_entries``, by default the checked-in one) no longer
    allowing it under the license it was chosen under. Rights are a hard stop: what cannot be confirmed is refused."""
    from .website_library_asset import _unchanged
    from .website_object_library import SUBJECT_RECORD, SUBJECT_SCHEMA, _read_sealed, reading_digest, rights_now
    subject = _read_sealed(Path(root) / SUBJECT_RECORD, SUBJECT_SCHEMA, str(row["target_id"]))
    if subject is None:
        return None, NOT_PICKED
    if subject.get("reading_digest") != reading_digest(row, task_context):
        return None, READING_CHANGED
    if subject.get("not_used"):
        return None, str(subject["not_used"])
    files = subject.get("files")
    if not isinstance(subject.get("attribution"), Mapping) or not isinstance(files, list) or not files:
        return None, NOT_FETCHED
    try:
        if not all(_unchanged(item) for item in files):
            return None, FILES_CHANGED
    except (KeyError, TypeError, OSError):
        return None, NOT_FETCHED
    refused = rights_now(subject["attribution"], catalog_entries)
    return (None, refused) if refused else (subject, None)


def _opens(row: Mapping[str, Any], entry: Mapping[str, Any] | None) -> bool:
    """Whether the object has a part that opens, by the plan's row or the analysis's reading of it."""
    entry = entry or {}
    return (row.get("rigidity") != "rigid" or bool(entry.get("articulated_part"))
            or entry.get("articulation_kind") in ("prismatic", "revolute"))


def free_source(row: Mapping[str, Any], *, entry: Mapping[str, Any] | None, root: Path,
                task_context: Mapping[str, Any], artisan_cap_usd: float,
                catalog_entries: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """One extra object's record: its library model and parametric shape as they stand, the free source it could use
    (``library`` or ``parametric``), or ``artisan`` with why and the cap it would need, unreserved."""
    model, why = sealed_library_model(row, root=root, task_context=task_context, catalog_entries=catalog_entries)
    library = ({"status": "sealed_and_fetched", "entry_id": model["attribution"].get("entry_id"),
                "license": model["attribution"].get("license"), "subject_digest": model["digest"]}
               if model is not None else {"status": "not_used", "reason": why})
    shape = row.get("parametric_shape") if row.get("parametric_shape") in SHAPES else None
    flags: list[str] = []
    if _opens(row, entry):
        source, reason = ARTISAN, OPENS
    elif model is not None and shape is not None:
        source, reason = LIBRARY, None
        if shape == CYLINDER:
            flags.append(CYLINDER_FALLBACK)
    elif shape is not None and shape != CYLINDER:
        source, reason = PARAMETRIC, None
    else:
        source, reason = ARTISAN, (LIBRARY_WITHOUT_FALLBACK if model is not None
                                   else CYLINDER_ONLY if shape == CYLINDER else NO_FREE_SOURCE)
    return {"target_id": str(row["target_id"]), "name": row.get("name"), "plan_sourcing": row.get("sourcing"),
            "library": library, "parametric_shape": shape, "source": source, "free": source != ARTISAN,
            **({"fallback_shape": shape} if source == LIBRARY else {}),
            **({"needs_artisan": {"reason": reason, "maximum_cost_usd": artisan_cap_usd, "reserved": False}}
               if source == ARTISAN else {}),
            **({"flags": flags} if flags else {})}


def extra_object_sources(plan: Mapping[str, Any] | None, *, entries: Sequence[Mapping[str, Any]],
                         editor_objects: Sequence[Mapping[str, Any]], task_context: Mapping[str, Any],
                         library_root: Path, catalog_entries: Sequence[Mapping[str, Any]] | None = None
                         ) -> dict[str, Any]:
    """The record of each object the plan would build after its first (``built_objects``), in the plan's order: the
    free source it could use, or ARTISAN at its cap (``new_authoring_budget``, as its reservation would freeze it),
    with nothing reserved. ``entries`` are the removal manifest's rows and ``editor_objects`` the scene editor's, as
    ``built_objects`` reads them; ``library_root`` holds each extra object's sealed pick (``object_root``). Reads only;
    writes, reserves, spends and asks nothing. A plan with one object it would build, or none, has no extra object."""
    from .task_object_authoring_backend import new_authoring_budget
    built = built_objects(plan, entries=entries, editor_objects=editor_objects)
    rows = {row["target_id"]: row for row in (plan or {}).get("objects") or []}
    analysis = {str(row.get("target_id")): row for row in entries if row.get("task_effect") == "manipulated"}
    cap = new_authoring_budget()["maximum_cost_usd"]
    objects = [{"rank": rank, **free_source(rows[item["target_id"]], entry=analysis.get(item["target_id"]),
                                           root=object_root(library_root, item["target_id"]),
                                           task_context=task_context, artisan_cap_usd=cap,
                                           catalog_entries=catalog_entries)}
               for rank, item in enumerate(built[1:], start=1)]
    paid = [row for row in objects if not row["free"]]
    record = {"schema_version": SCHEMA_VERSION, "revision": REVISION,
              "task_context_digest": task_context.get("context_digest"),
              "first_object": ({"target_id": built[0]["target_id"], "name": built[0]["name"]} if built else None),
              "objects": objects, "free_count": len(objects) - len(paid), "artisan_count": len(paid),
              "artisan_maximum_cost_usd": round(sum(row["needs_artisan"]["maximum_cost_usd"] for row in paid), 6),
              "claim_boundary": "Which free source each further object the plan would build could use, read from "
                                "what is already sealed; nothing is reserved, spent or built by this record, and "
                                "compile's own configuration of each object decides what it costs."}
    record["digest"] = canonical_digest(record, digest_field="digest")
    return record


__all__ = ["ARTISAN", "CYLINDER_FALLBACK", "CYLINDER_ONLY", "LIBRARY", "LIBRARY_WITHOUT_FALLBACK", "NO_FREE_SOURCE",
           "NOT_PICKED", "OPENS", "PARAMETRIC", "extra_object_sources", "free_source", "object_root",
           "sealed_library_model"]
