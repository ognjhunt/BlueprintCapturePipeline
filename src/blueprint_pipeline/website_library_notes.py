"""A library model in the company's package: its own file, its credit and its license, in plain words.

Kept apart from the build path so the company package (``website_company_simulation``, the control-plane lane) reads
the preparation's record without importing it. Reads and copies only: never decides, never fetches, never buys.

The model Blueprint used for the job's object (``website_library_asset``: converted, and scaled where it was) ships as
its own clearly named file under ``scene/files/``. The README credits it under a "Library models" heading with the
wording the owner settled, and says the package's terms do not restrict it: CC BY 4.0 allows no further restriction on
the licensed material (owner, 2026-11-23: "yes add the CC BY carve-out to the README"; the website Terms of
2026-11-23.4 say the same: a model from a public library stays under that library's license, named in the README).

The credit names every shipped file that holds the model's geometry (``credited_files``): the model's own file, the
job's object part and every USD layer it uses (the object is built from the model, whatever later stages did to it),
and any other shipped USD file with a mesh of the model's own topology (a collision or physics copy). It keeps the
copyright notice and the disclaimer of warranties the library supplies with the model, as written (CC BY 4.0
3(a)(1)(A)(ii) and (iv)), and says nothing where none was supplied. A package with no library model has no such
section. The credit never depends on the model's own file: when that file is gone or changed it is not shipped, and
the credit names the files that are (the object, and any copy found by the topology recorded when the model was
written). The credit is read from the preparation the shipped half was built from (``website_half_preparation``),
never from a newer one. When the object was built from a library model whose credit cannot be read, or that
preparation cannot be read at all, the package may not leave with the object (``refusal``): library geometry never
ships without its credit, and the room floor ships instead. So too when the deployed catalog no longer allows the
model under the license it was built under (``LICENSE_REFUSED``: a newer index, a dataset refused since), whatever
the preparation recorded: rights are read as they stand when the package is built
(``website_library_catalog.rights_now``), never as they stood when the job chose the model.
"""
from __future__ import annotations

import hashlib
import textwrap
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

#: The package's name for the one library model of the job's object.
MODEL_FILE = "scene/files/library_model.usda"
HEADING = "Library models"
#: One bullet per library model. Owner confirmed the CC BY carve-out and this wording on 2026-11-23 ("yes add the CC BY
#: carve-out to the README"). ``{files}`` names every shipped file holding the model's geometry; ``{copyright}`` is the
#: library's own copyright notice in brackets, or nothing; ``{changes}`` says what Blueprint did to the model, as
#: ``changes`` words it; ``{disclaimer}`` is the library's own disclaimer of warranties, or nothing.
CREDIT = ("- {files}: {title} by {author}{copyright}, from {library} ({link}), under {license} ({license_url}). "
          "Blueprint {changes} to fit this job.{disclaimer}")
#: Once, after the bullets (owner, 2026-11-23), word for word.
CARVE_OUT = ("These files stay under their own license. Blueprint does not own them, and the terms Blueprint shares "
             "this package under do not restrict them.")
USD_SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
#: The layer of the job's objects' simulated variations (``website_object_variations.LAYER_FILE``, named here: that
#: module imports this one). It holds no geometry, but its variants change the object built from the model, so a
#: package that ships it names it in the model's credit.
VARIATIONS_FILE = "variations.usda"


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


#: The sizing that scaled the model (``website_library_asset.SCALED``, named here: this lane stays off its imports).
SCALED = "scaled_to_video_estimate"


def changes(design: Mapping[str, Any]) -> str:
    """What Blueprint did to the model, in the credit's words: converted it from its own format, and scaled it only
    when its sizing record says so (``SCALED``); no record, no claim."""
    sizing = design.get("sizing") if isinstance(design.get("sizing"), Mapping) else {}
    return f"converted it from {design['converted_from']}" + (" and scaled it" if sizing.get("status") == SCALED else "")


def credit_text(row: Mapping[str, Any], files: Sequence[str]) -> str:
    """One model's credit: ``files`` named in order, its own copyright notice and disclaimer only where supplied."""
    names = list(files) or ([str(row["file"])] if row.get("file") else ["the job's object"])
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    notice, disclaimer = row.get("copyright"), row.get("disclaimer")
    return CREDIT.format(**{**row, "files": joined, "copyright": f" ({notice})" if notice else "",
                            "disclaimer": f" Its own notice: {disclaimer}" if disclaimer else ""})


#: The preparation authored the job's object from a library model but its credit cannot be read, or the preparation
#: itself cannot be read: the package may not leave with that object (rights is a hard stop), since a model's geometry
#: could ship without its credit.
CREDIT_UNREADABLE = "website_company_simulation_library_credit_unreadable"
#: The preparation authored the job's object from a library model the deployed catalog no longer allows under the
#: license it was chosen under (a newer index, a dataset refused since: ``website_library_catalog.rights_now``): the
#: package may not leave with that object, and the room floor ships instead, flagged.
LICENSE_REFUSED = "website_company_simulation_library_license_refused"
_CREDIT_FIELDS = (("title", "title"), ("author", "author"), ("library", "library_title"), ("link", "item_url"),
                  ("license", "license_name"), ("license_url", "license_url"))


def topology_digest(counts: Any, indices: Any) -> tuple[int, str]:
    """A mesh's topology, as ``mesh_topologies`` finds it: its face-vertex index count and a digest of its face counts
    and indices."""
    import numpy as np
    return len(indices), hashlib.sha256(np.asarray(counts, dtype=np.int64).tobytes() + b"|"
                                        + np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def credit(configuration: Mapping[str, Any], capture_root: Path) -> dict[str, Any]:
    """The library model the job's object was authored from, by the stage-3 ``configuration`` of the half's own
    preparation (``website_half_preparation``): ``{"library_models": [row]}`` whenever it was and its credit reads,
    so the credit never depends on the model's own file (a file missing or changed since it was written, or outside
    ``capture_root``, is not shipped, and the credit names the files that are); ``{"library_refusal":
    CREDIT_UNREADABLE}`` when it was but the credit cannot be read, ``LICENSE_REFUSED`` when the deployed catalog no
    longer allows the model under that license; empty when no library model was used."""
    root = Path(capture_root)
    if not isinstance(configuration, Mapping) or "library_design" not in configuration:
        return {}
    design = configuration["library_design"]
    try:
        credit = design["attribution"]
        row = {key: str(credit[field]).strip() for key, field in _CREDIT_FIELDS}
        if not all(row.values()):
            raise ValueError("credit_field_empty")
        row.update(changes=changes(design),
                   # The library's own words, as supplied; absent when it supplied none.
                   copyright=str(credit["copyright"]) if credit.get("copyright") else None,
                   disclaimer=str(credit["disclaimer"]) if credit.get("disclaimer") else None)
    except (KeyError, TypeError, ValueError, AttributeError):
        return {"library_refusal": CREDIT_UNREADABLE}
    from .website_library_catalog import rights_now
    if rights_now(credit):
        # The deployed catalog no longer allows the model under the license it was built under: it never ships.
        return {"library_refusal": LICENSE_REFUSED}
    model = design.get("model_file") if isinstance(design.get("model_file"), Mapping) else {}
    topology = design.get("model_topology")
    row["topology"] = list(topology) if isinstance(topology, (list, tuple)) and len(topology) == 2 else None
    model_path = Path(str(model.get("path") or ""))
    if (model.get("path") and model_path.resolve().is_relative_to(root.resolve())
            and _sha256(model_path) == model.get("sha256")):
        row.update(file=MODEL_FILE, path=str(model_path), sha256=str(model["sha256"]))
    else:
        row.update(file=None, path=None, sha256=None, model_file_not_shipped=True)
    return {"library_models": [row]}


#: A further object of the job's in the package that the preparation it was built from does not account for, by its
#: identity (``further_parts``): whether its file holds a library model cannot be told, so it may not leave uncredited.
FURTHER_UNACCOUNTED = "website_company_simulation_further_object_unaccounted"


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """A further object's stable identity, as its preparation recorded it (``website_further_objects``): its rank,
    its replacement id, the runtime asset id its scene half's row carries, and the digest of the file it ships as. Never
    its prim: a half that leaves an earlier further object out renumbers the ones after it."""
    asset = row.get("asset") if isinstance(row.get("asset"), Mapping) else {}
    return {"rank": row.get("rank"), "replacement_asset_id": row.get("replacement_asset_id"),
            "runtime_asset_id": row.get("runtime_asset_id"), "asset_sha256": asset.get("sha256")}


def further_credits(preparation: Mapping[str, Any], capture_root: Path) -> dict[str, Any]:
    """Each further object of the job the preparation composed (its ``further_objects``, ``website_further_objects``),
    by its stable identity (``further_parts``: what the package may place), and the credit of each library model one
    was authored from, as ``credit`` reads one, marked with that identity (``further_object``):
    ``{"library_models": rows, "further_parts": identities}``, ``{"library_refusal": code}`` when a model cannot be
    credited (its geometry never ships without its credit), empty without any. A credit is matched to the part that
    ships by that identity, at its final prim (``credited``). Its model's own file is not shipped apart: its credit names
    the shipped files that hold its geometry."""
    record = preparation.get("further_objects") if isinstance(preparation, Mapping) else None
    rows: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for row in (record.get("objects") or []) if isinstance(record, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        identity = _identity(row)
        parts.append({**identity, "source": row.get("source")})
        if row.get("source") != "library":
            continue
        found = credit({"library_design": row.get("library_design")}, capture_root)
        if "library_refusal" in found:
            return {"library_refusal": found["library_refusal"]}
        rows += [{**item, "file": None, "path": None, "sha256": None, "further_object": identity}
                 for item in found["library_models"]]
    return {**({"library_models": rows} if rows else {}), **({"further_parts": parts} if parts else {})}


def merged(*credits: Mapping[str, Any]) -> dict[str, Any]:
    """Several ``credit`` results as one: a refusal when any refuses, else every model in order and every further part
    accounted for; empty without any."""
    refused = next((item["library_refusal"] for item in credits if item.get("library_refusal")), None)
    if refused:
        return {"library_refusal": refused}
    rows = [row for item in credits for row in item.get("library_models") or []]
    parts = [row for item in credits for row in item.get("further_parts") or []]
    return {**({"library_models": rows} if rows else {}), **({"further_parts": parts} if parts else {})}


def further_prims(source: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Each further object's part the package places, by its final prim (``Object_02``...: the plan's order, after any
    renumbering), with the identity its plan row and file give: its runtime asset id and its file's digest."""
    from .scene_package_stage import FURTHER_ROLE, StageRefused, placements
    plan, files = source.get("plan") or {}, source.get("files") or {}
    try:
        rows = placements(plan)
    except StageRefused:
        return {}
    return {placed["prim"]: {"runtime_asset_id": str(raw.get("asset_id") or ""),
                             "asset_sha256": str(files[placed["prim"]].get("sha256") or "")}
            for placed, raw in zip(rows, plan.get("objects") or [])
            if placed["role"] == FURTHER_ROLE and placed["prim"] in files}


def _matches(identity: Mapping[str, Any], shipped_as: Mapping[str, str]) -> bool:
    """Whether a part shipped as ``shipped_as`` is the further object ``identity`` names: the same runtime asset id and
    the same file bytes, both, never its prim."""
    return (bool(identity.get("runtime_asset_id")) and identity.get("runtime_asset_id") == shipped_as["runtime_asset_id"]
            and bool(identity.get("asset_sha256")) and identity.get("asset_sha256") == shipped_as["asset_sha256"])


def refusal(source: Mapping[str, Any]) -> str | None:
    """Why the package may not leave with the job's object in it: the object was built from a library model whose
    credit cannot be read, or from a preparation that cannot be read at all (``CREDIT_UNREADABLE``); or a further
    object's part ships that its preparation does not account for by its identity (``FURTHER_UNACCOUNTED``), so
    whether it holds a library model, and whose, cannot be told. A package without the object (the room floor) is not
    refused: ``website_company_simulation`` ships it instead."""
    files = source.get("files") or {}
    with_object = any(isinstance(row, Mapping) and row.get("role") == "task_object" for row in files.values())
    if not with_object:
        return None
    if source.get("library_refusal"):
        return str(source["library_refusal"])
    parts = [row for row in source.get("further_parts") or [] if isinstance(row, Mapping)]
    for shipped_as in further_prims(source).values():
        if not any(_matches(row, shipped_as) for row in parts):
            return FURTHER_UNACCOUNTED
    return None


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
        if not row.get("file"):
            continue  # its own file is gone or changed: the credit still names every file that ships
        if _sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("website_company_simulation_library_model_changed")
        target = Path(staging) / row["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        write(Path(row["path"]), target, "library model")
        names.append(row["file"])
    return names


def _layers(path: Path) -> list[str]:
    """The layer identifiers a shipped USD file holds: itself, or each USD member of a usdz."""
    if path.suffix.lower() != ".usdz":
        return [str(path)]
    with zipfile.ZipFile(path) as archive:
        return [f"{path}[{name}]" for name in archive.namelist()
                if PurePosixPath(name).suffix.lower() in USD_SUFFIXES[:3]]


def mesh_topologies(path: Path, lengths: set[int] | None = None) -> set[tuple[int, str]]:
    """Each mesh spec's topology in a USD file (or the layers of a usdz): its face-vertex index count and a digest of
    its face counts and indices. A placement, scale or turn leaves it unchanged, so a copy of a model is found by it.
    ``lengths``: only meshes with one of these index counts are digested (a room's large mesh is skipped cheaply)."""
    import numpy as np
    from pxr import Sdf
    found: set[tuple[int, str]] = set()
    for identifier in _layers(Path(path)):
        layer = Sdf.Layer.FindOrOpen(identifier)
        if layer is None:
            continue
        meshes: list[Any] = []
        layer.Traverse(Sdf.Path.absoluteRootPath,
                       lambda spec_path: meshes.append(spec_path) if spec_path.IsPrimPath() else None)
        for spec_path in meshes:
            prim = layer.GetPrimAtPath(spec_path)
            if prim is None or prim.typeName != "Mesh":
                continue
            counts, indices = prim.attributes.get("faceVertexCounts"), prim.attributes.get("faceVertexIndices")
            counts = counts.default if counts is not None else None
            indices = indices.default if indices is not None else None
            if counts is None or indices is None or not len(indices) or (lengths is not None
                                                                          and len(indices) not in lengths):
                continue
            found.add(topology_digest(np.asarray(counts), np.asarray(indices)))
    return found


def _uses(staging: Path, path: Path) -> set[str]:
    """The shipped USD files a part's file is made of: itself and every layer it brings in, inside the package."""
    from pxr import UsdUtils
    names = {path.relative_to(staging).as_posix()}
    layers, assets, _ = UsdUtils.ComputeAllDependencies(str(path))
    for item in [*(layer.realPath for layer in layers), *assets]:
        candidate = Path(str(item).split("[", 1)[0])
        if (candidate.suffix.lower() in USD_SUFFIXES and candidate.resolve().is_relative_to(staging.resolve())
                and candidate.is_file()):
            names.add(candidate.resolve().relative_to(staging.resolve()).as_posix())
    return names


def credited_files(source: Mapping[str, Any], staging: Path, placed: Mapping[str, str]) -> list[str]:
    """Every shipped file holding the library model's geometry, its own file first when it ships: the job's object
    part and the USD layers it uses, any other shipped USD file with a mesh of the model's topology (read from its
    own file, or as recorded when it was written), and the layer of its simulated variations (``VARIATIONS_FILE``).
    Empty without a model."""
    rows = [row for row in shipped(source) if not row.get("further_object")]  # a further object's: ``credited``
    if not rows:
        return []
    staging = Path(staging)
    first = [str(row["file"]) for row in rows if row.get("file") and (staging / str(row["file"])).is_file()]
    names = set(first) | ({VARIATIONS_FILE} if (staging / VARIATIONS_FILE).is_file() else set())
    for prim, row in (source.get("files") or {}).items():
        if isinstance(row, Mapping) and row.get("role") == "task_object" and prim in placed:
            names |= _uses(staging, staging / placed[prim])
    model = set().union(*(mesh_topologies(staging / name) for name in first))
    model |= {(int(row["topology"][0]), str(row["topology"][1])) for row in rows if row.get("topology")}
    lengths = {length for length, _ in model}
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        relative = path.relative_to(staging).as_posix()
        if relative in names or path.suffix.lower() not in USD_SUFFIXES or not lengths:
            continue
        if mesh_topologies(path, lengths) & model:
            names.add(relative)
    return [*first, *sorted(names - set(first))]


def _part_files(row: Mapping[str, Any], staging: Path, placed: Mapping[str, str], prims: Sequence[str]) -> list[str]:
    """The shipped files that hold a further object's library model: the parts it ships as (``prims``, its final prims)
    and the USD layers they use, any other shipped USD file with a mesh of its model's topology, and the layer of its
    simulated variations."""
    own = [placed[prim] for prim in prims if prim in placed]
    names = set().union(*(_uses(staging, staging / name) for name in own)) | (
        {VARIATIONS_FILE} if (staging / VARIATIONS_FILE).is_file() else set())
    topology = {(int(row["topology"][0]), str(row["topology"][1]))} if row.get("topology") else set()
    for path in sorted(item for item in staging.rglob("*") if item.is_file()) if topology else []:
        relative = path.relative_to(staging).as_posix()
        if relative not in names and path.suffix.lower() in USD_SUFFIXES and mesh_topologies(
                path, {length for length, _ in topology}) & topology:
            names.add(relative)
    return [*own, *sorted(names - set(own))]


def credited(source: Mapping[str, Any], staging: Path, placed: Mapping[str, str]) -> dict[str, Any]:
    """What the README's credits name: ``library_files``, every file of the job's object's model
    (``credited_files``), and, when a further object's model is credited, each model's own files on its credit row
    (``credited_files`` on ``library_models``). A further object's credit follows it by its identity to the part it
    ships as (``further_prims``), whatever prim that is after renumbering (``part``); a model whose object does not ship
    is not credited, and a part no credit or record accounts for never leaves (``refusal``)."""
    found: dict[str, Any] = {"library_files": credited_files(source, staging, placed)}
    rows = shipped(source)
    if any(row.get("further_object") for row in rows):
        prims = further_prims(source)
        kept = []
        for row in rows:
            if not row.get("further_object"):
                kept.append(row)
                continue
            matched = sorted(prim for prim, shipped_as in prims.items()
                             if prim in placed and _matches(row["further_object"], shipped_as))
            if matched:
                kept.append({**row, "part": matched[0], "credited_files": _part_files(row, Path(staging), placed,
                                                                                       matched)})
        found["library_models"] = kept
    return found


def readme_lines(source: Mapping[str, Any], *, width: int = 100) -> list[str]:
    """The README's "Library models" section, or nothing when the package ships none. ``source["library_files"]``
    (``credited_files``) names the files each credit covers."""
    rows: Sequence[Mapping[str, Any]] = shipped(source)
    if not rows:
        return []
    files = list(source.get("library_files") or [])
    lines = ["", HEADING, ""]
    for row in rows:
        lines += textwrap.wrap(credit_text(row, row["credited_files"] if "credited_files" in row else files),
                               width=width, subsequent_indent="  ", break_long_words=False, break_on_hyphens=False)
    return [*lines, "", *textwrap.wrap(CARVE_OUT, width=width)]


__all__ = ["CARVE_OUT", "CREDIT", "CREDIT_UNREADABLE", "HEADING", "LICENSE_REFUSED", "MODEL_FILE", "SCALED",
           "VARIATIONS_FILE",
           "changes", "credit", "credited",
           "FURTHER_UNACCOUNTED", "credit_text", "credited_files", "further_credits", "further_prims", "merged", "mesh_topologies", "place", "readme_lines", "refusal", "shipped",
           "topology_digest"]
