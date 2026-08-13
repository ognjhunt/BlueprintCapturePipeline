"""Execute the preregistered ADP-009A scene shortlist screening.

`adp009a_scene_shortlist_extension_preregistration.v1.json` freezes an ordering
rule and a five-step screening sequence, and records
`method_outcomes_may_not_influence_selection`. Nothing executed it: the sequence
existed only as strings in a manifest, so every scene was screened by an
operator and the outcome was a one-off. The cost is visible in the repository --
840313 is the only scene that ever reached a sealed source bundle, and later
work anchored on it rather than on the rank the frozen order actually points to.

This module performs the part of the sequence that decides *which* scene a run
may use: it walks the frozen order, records why each scene is passed over, and
selects the first whose retained bytes are all present. Steps that require
geometry or rendering stay with the maintained modules that already implement
them; what was missing was the ordering, the recorded reasons, and a digest a
later reader can replay.

It reads retained bytes only. It never mutates a provider and never chooses an
order of its own.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "adp009a_scene_shortlist_screening.v1"
_DIGEST_PREFIX = "sha256:"

# The retained bytes a scene needs before it can become a source bundle. A
# scene missing any of these cannot be screened further, however high it ranks.
REQUIRED_INPUTS = (
    ("appearance_3dgs", "3dgs_compressed.ply"),
    ("semantic_metadata", "labels.json"),
    ("scene_structure", "structure.json"),
)
SAGE_INPUT = "sage_usdz"


class SceneShortlistScreeningError(ValueError):
    """A preregistered shortlist failed a fail-closed contract."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: Mapping[str, Any], *, digest_field: str) -> str:
    payload = dict(value)
    payload.pop(digest_field, None)
    return _DIGEST_PREFIX + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return _DIGEST_PREFIX + digest.hexdigest()


def load_preregistered_shortlist(preregistration_path: str | Path) -> list[dict[str, Any]]:
    """Return the frozen shortlist, in rank order, only if its digest binds.

    A shortlist that can be edited without detection is not a preregistration,
    so a digest mismatch is refused rather than repaired.
    """
    path = Path(preregistration_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SceneShortlistScreeningError(
            "scene_shortlist_screening_preregistration_unreadable"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SceneShortlistScreeningError("scene_shortlist_screening_preregistration_invalid")

    recorded = str(payload.get("record_digest") or "")
    if recorded != canonical_digest(payload, digest_field="record_digest"):
        raise SceneShortlistScreeningError(
            "scene_shortlist_screening_preregistration_record_digest_mismatch"
        )

    shortlist = payload.get("shortlist")
    if not isinstance(shortlist, Sequence) or not shortlist:
        raise SceneShortlistScreeningError("scene_shortlist_screening_shortlist_missing")

    entries = [dict(entry) for entry in shortlist if isinstance(entry, Mapping)]
    if len(entries) != len(shortlist):
        raise SceneShortlistScreeningError("scene_shortlist_screening_shortlist_invalid")
    # Rank is the frozen order; sorting by it keeps the file's own ordering
    # authoritative rather than trusting array position.
    entries.sort(key=lambda entry: int(entry.get("rank", 0)))
    return entries


def _screen_scene(entry: Mapping[str, Any], source_root: Path) -> dict[str, Any]:
    scene_id = str(entry.get("scene_id") or "")
    folder = str(entry.get("interiorgs_folder") or "")
    blockers: list[str] = []
    inputs: dict[str, dict[str, Any]] = {}

    scene_dir = source_root / "InteriorGS" / folder
    for name, filename in REQUIRED_INPUTS:
        candidate = scene_dir / filename
        if not candidate.is_file():
            blockers.append(f"scene_shortlist_screening_missing:{name}")
            continue
        inputs[name] = {
            "path": str(candidate),
            "digest": _file_digest(candidate),
            "size_bytes": candidate.stat().st_size,
        }

    usdz = source_root / "SAGE-3D_InteriorGS_usdz" / "InteriorGS_usdz" / f"{scene_id}.usdz"
    if usdz.is_file():
        inputs[SAGE_INPUT] = {
            "path": str(usdz),
            "digest": _file_digest(usdz),
            "size_bytes": usdz.stat().st_size,
        }
    else:
        blockers.append(f"scene_shortlist_screening_missing:{SAGE_INPUT}")

    return {
        "scene_id": scene_id,
        "rank": entry.get("rank"),
        "interiorgs_folder": folder,
        "eligible": not blockers,
        "blockers": blockers,
        "inputs": inputs,
    }


def screen_shortlist(
    preregistration_path: str | Path, *, source_root: str | Path
) -> dict[str, Any]:
    """Walk the frozen order and select the first scene with all retained bytes.

    Every scene passed over carries its own recorded reason, so a later reader
    can tell a scene that was screened and rejected from one that was never
    reached. When no scene is complete this blocks rather than falling back to
    whichever scene happens to be available, because availability is exactly
    what the preregistration forbids from influencing selection.
    """
    entries = load_preregistered_shortlist(preregistration_path)
    root = Path(source_root).expanduser().resolve()

    screened: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for entry in entries:
        row = _screen_scene(entry, root)
        screened.append(row)
        if row["eligible"] and selected is None:
            selected = row
            # Stop at the first eligible scene: screening later ranks could not
            # change the choice, and reading them invites reordering by outcome.
            break

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "selected" if selected else "blocked",
        "ordering_authority": str(Path(preregistration_path).name),
        "selected": selected,
        "screened": screened,
        "provider_mutation_performed": False,
    }
    result["screening_digest"] = canonical_digest(result, digest_field="screening_digest")
    return result


__all__ = [
    "SCHEMA_VERSION",
    "SceneShortlistScreeningError",
    "canonical_digest",
    "load_preregistered_shortlist",
    "screen_shortlist",
]
