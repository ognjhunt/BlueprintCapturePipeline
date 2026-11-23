"""A library model in the company's package: its own file, its credit and its license, in plain words.

Kept apart from the build path so the company package (``website_company_simulation``, the control-plane lane) reads
the preparation's record without importing it. Reads and copies only: never decides, never fetches, never buys.

The model Blueprint used for the job's object (``website_library_asset``: converted, and scaled where it was) ships as
its own clearly named file under ``scene/files/``. The README credits it under a "Library models" heading with the
wording the owner settled, and says the package's terms do not restrict it: CC BY 4.0 allows no further restriction on
the licensed material (owner, 2026-11-23: "yes add the CC BY carve-out to the README"). A package with no library model
has no such section.
"""
from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

#: The package's name for the one library model of the job's object.
MODEL_FILE = "scene/files/library_model.usda"
HEADING = "Library models"
#: One bullet per library file. Owner confirmed the CC BY carve-out and this wording on 2026-11-23 ("yes add the CC BY
#: carve-out to the README"); ``{changes}`` says what Blueprint did to the file, as ``changes`` words it.
CREDIT = ("- {file}: {title} by {author}, from {library} ({link}), under {license} ({license_url}). "
          "Blueprint {changes} to fit this job.")
#: Once, after the bullets (owner, 2026-11-23).
CARVE_OUT = ("These files stay under their own license. Blueprint does not own them, and the terms Blueprint shares "
             "this package under do not restrict them.")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def changes(design: Mapping[str, Any]) -> str:
    """What Blueprint did to the model, in the credit's words: converted it from its own format, scaled it when it
    was scaled."""
    scaled = (design.get("sizing") or {}).get("status") != "as_is"
    return f"converted it from {design['converted_from']}" + (" and scaled it" if scaled else "")


def library_models(capture_root: Path) -> dict[str, Any]:
    """The library model the preparation the job's handoff names was authored from, when it was and its file is
    still the one recorded: ``{"library_models": [...]}``, else empty, so the package says what it said before."""
    root = Path(capture_root)
    folder = root / "pipeline" / "website_scene_preparation"
    named = _read(folder / "handoff.json").get("preparation_path")
    path = folder / "preparation.json"
    if isinstance(named, str):
        candidate = (Path(named) if Path(named).is_absolute() else root / named).resolve()
        path = candidate if candidate.is_relative_to(root.resolve()) else path
    design = ((_read(path).get("authoring_inputs") or {}).get("configuration") or {}).get("library_design")
    if not isinstance(design, Mapping):
        return {}
    try:
        model, credit = design["model_file"], design["attribution"]
        row = {"file": MODEL_FILE, "path": str(model["path"]), "sha256": str(model["sha256"]),
               "title": str(credit["title"]), "author": str(credit["author"]), "library": str(credit["library_title"]),
               "link": str(credit["item_url"]), "license": str(credit["license_name"]),
               "license_url": str(credit["license_url"]), "changes": changes(design)}
    except (KeyError, TypeError):
        return {}
    model_path = Path(row["path"]).resolve()
    if not model_path.is_relative_to(root.resolve()) or _sha256(model_path) != row["sha256"]:
        return {}
    return {"library_models": [row]}


def shipped(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The library models this package ships: only with the job's object in it."""
    files = source.get("files") or {}
    if not any(isinstance(row, Mapping) and row.get("role") == "task_object" for row in files.values()):
        return []
    return [row for row in source.get("library_models") or [] if isinstance(row, Mapping)]


def place(source: Mapping[str, Any], staging: Path, write: Callable[[Path, Path, str], None]) -> list[str]:
    """Writes each shipped library model into the package under its own name (``write``: the package's own writer,
    which never copies a file byte for byte); a model file changed since it was read refuses through ``write``'s
    caller. Returns the names written."""
    names = []
    for row in shipped(source):
        if _sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("website_company_simulation_library_model_changed")
        target = Path(staging) / row["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        write(Path(row["path"]), target, "library model")
        names.append(row["file"])
    return names


def readme_lines(source: Mapping[str, Any], *, width: int = 100) -> list[str]:
    """The README's "Library models" section, or nothing when the package ships none."""
    rows: Sequence[Mapping[str, Any]] = shipped(source)
    if not rows:
        return []
    lines = ["", HEADING, ""]
    for row in rows:
        lines += textwrap.wrap(CREDIT.format(**row), width=width, subsequent_indent="  ", break_long_words=False,
                               break_on_hyphens=False)
    return [*lines, "", *textwrap.wrap(CARVE_OUT, width=width)]


__all__ = ["CARVE_OUT", "CREDIT", "HEADING", "MODEL_FILE", "changes", "library_models", "place", "readme_lines",
           "shipped"]
