"""The preparation a compiled half was built from, and what the company package says of it, from that one alone.

A half is compiled from a published revision, and a website revision names the exact bytes of the preparation it was
built from: its ``source.manifest`` is the copy its submission staged (``STAGED_NAME``, ``website_native_submission``).
The half's sealed record keeps that digest (``FIELD``; ``team_scene_packet`` and ``scene_packet`` seal it, ``binding``),
and the company package (``website_company_simulation``) reads every disclosure a preparation gives from the one under
the capture whose bytes have it (``half_disclosures``), never from the one the job's handoff names now: a retry that
prepared the job again (a correction, a renewal, a new Atlas result) changes nothing in a half compiled before it. So
are the library model's credit (``website_library_notes``), the object's mass and friction estimates, its size notes
and where the rule's numbers came from, and the base scene's notes (generated geometry, generated views, an estimated
surface): from the base scene written beside that preparation, else the handoff's. The compile reads no preparation itself: the remote compile lane is never sent the site's records.

A half sealed before the binding existed, or compiled from another kind of revision, reads the preparation the handoff
names, else the default one, as before.

A half whose own preparation cannot be read (not found by its bytes; or, unbound, missing, not JSON, not an object, or
with authoring inputs that are not one) says none of it, and its object may not leave
(``website_library_notes.CREDIT_UNREADABLE``): whether it holds a library model's geometry cannot be told, and that
geometry never ships without its credit. Its room floor ships instead, flagged (``website_company_simulation``). Reads
only: never decides, never fetches, never buys.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import website_library_notes as library_notes
from .atlas_automation_plan import GENERATED_ESTIMATES_BASIS, GENERATED_IN_COLLISION_ONLY, GENERATED_VIEWS_IN_LOOK
from .decision_evidence_contracts import canonical_digest

#: The half record's field: the sha256 of the preparation its revision was built from.
FIELD = "preparation_sha256"
#: Where the website submission stages the job's preparation (``website_native_submission``): a revision whose source
#: manifest is this file is a website one, and its digest is that preparation's bytes.
STAGED_NAME = "website/preparation.json"
#: The two places ``generated_patch`` names.
COLLISION_ONLY, LOOK_AND_COLLISION = GENERATED_IN_COLLISION_ONLY, "look_and_collision"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
#: How deep under ``website_scene_preparation`` a preparation is looked for (``superseded_atlas_results/<result>/
#: revisions/<hex>/renewals/<hex>/preparation.json`` is the deepest layout written).
_DEPTH = 6
_UNREADABLE = object()


def binding(revision: Mapping[str, Any]) -> dict[str, str]:
    """``{FIELD: digest}`` for a website revision (its source manifest is the staged preparation), else empty."""
    manifest = (revision.get("source") or {}).get("manifest") if isinstance(revision.get("source"), Mapping) else None
    if not isinstance(manifest, Mapping) or not str(manifest.get("uri") or "").endswith("/" + STAGED_NAME):
        return {}
    digest = manifest.get("digest")
    return {FIELD: digest} if isinstance(digest, str) and _SHA256.fullmatch(digest) else {}


def _inside(base: Path, path: Path) -> Path | None:
    """``path`` resolved, when it is a regular file under ``base`` reached through no link."""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(base)) \
            or not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        return None
    return resolved


def _load(path: Path | None) -> Any:
    """The JSON at ``path``, or ``_UNREADABLE``."""
    try:
        return json.loads(Path(str(path)).read_text()) if path is not None else _UNREADABLE
    except (OSError, ValueError):
        return _UNREADABLE


def _folder(capture_root: Path) -> Path:
    return Path(capture_root) / "pipeline" / "website_scene_preparation"


def _handoff(capture_root: Path) -> dict[str, Any]:
    value = _load(_folder(capture_root) / "handoff.json")
    return value if isinstance(value, dict) else {}


def named_path(capture_root: Path) -> Path:
    """The preparation the job's handoff names (inside the capture), else the default preparation record."""
    root = Path(capture_root)
    named = _handoff(root).get("preparation_path")
    if isinstance(named, str):
        candidate = _inside(root, Path(named) if Path(named).is_absolute() else root / named)
        if candidate is not None:
            return candidate
    return _folder(root) / "preparation.json"


def _candidates(capture_root: Path) -> list[Path]:
    """Every preparation record under the capture: the one the handoff names first, then the default, then the rest."""
    root, folder = Path(capture_root), _folder(capture_root)
    found: list[Path] = []
    for directory, names, files in os.walk(folder):
        depth = len(Path(directory).relative_to(folder).parts)
        names[:] = sorted(name for name in names if depth < _DEPTH and not (Path(directory) / name).is_symlink())
        if "preparation.json" in files:
            found.append((Path(directory) / "preparation.json").resolve())
    first = [path.resolve() for path in (named_path(root), folder / "preparation.json")]
    return [*dict.fromkeys(path for path in first if path in found), *sorted(set(found) - set(first))]


def _sha256(path: Path) -> str | None:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def half_preparation(capture_root: Path, record: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    """The preparation the half whose sealed ``record`` this is was built from, and where it is; ``(None, path)``
    when it cannot be read. Bound (``FIELD``): the one under the capture with those bytes. Unbound: the one the
    handoff names, else the default one."""
    root = Path(capture_root)
    digest = record.get(FIELD)
    if isinstance(digest, str):
        path = next((path for path in _candidates(root) if _inside(root, path) is not None and _sha256(path) == digest),
                    None)
    else:
        path = named_path(root)
    value = _load(path)
    if not isinstance(value, dict):
        return None, path
    inputs = value.get("authoring_inputs", {})
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get("configuration", {}), Mapping):
        return None, path
    return value, path


def object_estimates(preparation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The object's mass and friction ranges as the preparation estimated them from the video, when recorded."""
    bounds = (preparation.get("physics") or {}).get("bounds") if isinstance(preparation.get("physics"), Mapping) else None
    return bounds if isinstance(bounds, Mapping) else None


def success_basis(preparation: Mapping[str, Any]) -> dict[str, str] | None:
    """For each number of the rule the preparation set, where it came from: ``site_stated`` (the site's own words),
    ``destination_area`` or ``assumed``, from the preparation's ``success_rule_basis`` when the build choice read the
    site's words (``website_task_preparation._rule_success``); None otherwise, so its package says what it said
    before. Never the quotes: the site's words can name the site."""
    record = preparation.get("success_rule_basis")
    fields = record.get("fields") if isinstance(record, Mapping) and record.get("applied") is True else None
    if not isinstance(fields, Mapping):
        return None
    # ``site_words`` is ``website_build_choice.SUCCESS_RULE_SITE``, named here: the package lane stays off its imports.
    words = {"site_words": "site_stated", "assumed": "assumed", "destination_area": "destination_area"}
    found = {str(key): words[row["basis"]] for key, row in fields.items()
             if isinstance(row, Mapping) and row.get("basis") in words}
    return found or None


def size_notes(preparation: Mapping[str, Any]) -> dict[str, bool]:
    """What the preparation says of the object's size (``subject_extent``) and where it rests: read from its own
    points, an exact model's published size that disagrees, and a destination it overhangs (``destination_overhang``).
    Empty when none, so its package says what it said before."""
    extent = preparation.get("subject_extent")
    extent = extent if isinstance(extent, Mapping) else {}
    check = extent.get("published_cross_check") if isinstance(extent.get("published_cross_check"), Mapping) else {}
    overhang = preparation.get("destination_overhang")
    return {**({"size_from_its_points": True} if extent.get("status") == "from_its_own_points" else {}),
            **({"size_disagrees_with_published": True} if check.get("status") == "disagrees" else {}),
            **({"destination_overhang": True} if isinstance(overhang, Mapping) and overhang.get("flag") else {})}


def handoff_base_scene(capture_root: Path) -> dict[str, Any]:
    """The base scene the job's sealed scene handoff names (inside the capture), else empty."""
    root = Path(capture_root)
    handoff = _handoff(root)
    path = handoff.get("base_scene_path")
    if not isinstance(path, str) or handoff.get("digest") != canonical_digest(handoff, digest_field="digest"):
        return {}
    base = _load(_inside(root, Path(path)))
    return base if isinstance(base, dict) else {}


def _beside(capture_root: Path, preparation_path: Path) -> dict[str, Any]:
    """The base scene written beside a preparation (a development test's sits one folder up), else empty."""
    folder = preparation_path.parent
    folder = folder.parent if folder.name == "development_test" else folder
    base = _load(_inside(Path(capture_root), folder / "base_scene.json"))
    return base if isinstance(base, dict) else {}


def base_scene_notes(base: Mapping[str, Any]) -> dict[str, Any]:
    """What a base scene says the package must tell: where it keeps admitted generated geometry (``generated_patch``:
    ``collision_only``, only in the scan its collision was made from beside a refined look; ``look_and_collision``,
    any other; jobs before 2026-11-17 record no marker), generated views in its look (``look_views``, owner
    2026-11-18: checked against the filmed frames and their depth) and a destination surface the scan did not show,
    estimated from its visible edges (``surface_estimated``). Empty for a base scene with none of them."""
    from .atlas_scene_estimates import DESTINATION_SURFACE_ESTIMATED
    content, look = base.get("content_provenance"), base.get("look_provenance")
    rows = base.get("scene_estimates") if isinstance(base.get("scene_estimates"), list) else []
    patch = None
    if isinstance(content, Mapping) and content.get("basis") == GENERATED_ESTIMATES_BASIS:
        patch = COLLISION_ONLY if content.get("generated_geometry_in") == COLLISION_ONLY else LOOK_AND_COLLISION
    return {**({"generated_patch": patch} if patch else {}),
            **({"look_views": True} if isinstance(look, Mapping)
               and look.get("generated_views_in") == GENERATED_VIEWS_IN_LOOK else {}),
            **({"surface_estimated": True} if any(isinstance(row, Mapping)
                                                  and row.get("code") == DESTINATION_SURFACE_ESTIMATED
                                                  for row in rows) else {})}


def half_disclosures(capture_root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the half's own preparation and base scene say the package must tell, as the source's keys; plus
    ``estimates`` and ``basis`` for its rule. Only ``library_refusal`` when that preparation cannot be read."""
    root = Path(capture_root)
    preparation, path = half_preparation(root, record)
    if preparation is None:
        return {"library_refusal": library_notes.CREDIT_UNREADABLE}
    configuration = (preparation.get("authoring_inputs") or {}).get("configuration") or {}
    base = (_beside(root, path) if isinstance(record.get(FIELD), str) and path is not None else {}) \
        or handoff_base_scene(root)
    return {**base_scene_notes(base), **size_notes(preparation), **library_notes.credit(configuration, root), "estimates": object_estimates(preparation),
            "basis": success_basis(preparation)}


# --- the same lookups by capture, as the package read them before the binding (and for a room) -------------------

def preparation(capture_root: Path) -> dict[str, Any]:
    """The preparation the job's scene handoff names (inside the capture), else the default one; empty if unreadable."""
    value = _load(named_path(capture_root))
    return value if isinstance(value, dict) else {}


def generated_patch(capture_root: Path) -> str | None:
    """Where the base scene the job's sealed scene handoff names keeps admitted generated geometry, or None."""
    return base_scene_notes(handoff_base_scene(capture_root)).get("generated_patch")


def look_views(capture_root: Path) -> bool:
    """Whether the base scene the job's sealed scene handoff names has generated views in its look."""
    return bool(base_scene_notes(handoff_base_scene(capture_root)).get("look_views"))


def surface_estimated(capture_root: Path) -> bool:
    """Whether the base scene the job's sealed scene handoff names holds an estimated destination surface."""
    return bool(base_scene_notes(handoff_base_scene(capture_root)).get("surface_estimated"))


__all__ = ["COLLISION_ONLY", "FIELD", "LOOK_AND_COLLISION", "STAGED_NAME", "base_scene_notes", "binding",
           "generated_patch", "half_disclosures", "half_preparation", "handoff_base_scene", "look_views", "named_path",
           "object_estimates", "preparation", "size_notes", "success_basis", "surface_estimated"]
