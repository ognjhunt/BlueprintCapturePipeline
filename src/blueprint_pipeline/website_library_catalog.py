"""The object libraries' checked-in catalog and its rights, read the same way wherever a library model is used.

Split out of ``website_object_library`` (stress round 2, 2026-11-24) so the company package (the control-plane lane,
which never imports the build path) can read the same catalog: rights are a hard stop, and a model a newer index no
longer allows must not ship under the license an older one gave it.

The catalog is checked in (``_catalog_data/object_library_*.json``, built by ``scripts/build_*_object_index.py``):
Google Scanned Objects, real scanned household objects Google published on Gazebo Fuel under CC BY 4.0, and NVIDIA
SimReady packages, read only from the pinned URLs NVIDIA reviewed (SimReady Central's submission lists). An entry is a
candidate only when every license statement its source makes is, word for word, a spelling of a license on
``ALLOWED_LICENSES`` (for SimReady both the package definition's ``license`` and the USD's own ``asset_license``) and it
names its author; anything else, unknown or missing is refused and never shown. So is a model whose library asks to be
credited in a way the README cannot give (``attribution_refusal``: a link or a name of its own), and, for now, the
whole M-XR dataset, whose attribution files ask for a web link (``DATASETS_REFUSED``).

The deployed catalog is the one this release ships (``INDEXES``). A job's sealed choice of a library model was made
under the catalog of its day: before the model is used again (the handoff), built (compile) or shipped (the company
package), its entry is looked up in the deployed catalog now (``rights_now``). It must still be there, under the same
allowed license and from a dataset that is not refused; otherwise the model is not used and the reason is recorded.
Reads only: never decides, never fetches, never buys.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

DATA = Path(__file__).parent / "_catalog_data"
#: Each library's checked-in index, by its key: the catalog this release ships (the deployed catalog).
SHIPPED_INDEXES = {"google_scanned_objects": DATA / "object_library_gso.v1.json",
                   "nvidia_simready": DATA / "object_library_simready.v1.json"}
INDEXES: Mapping[str, Path] = SHIPPED_INDEXES
#: The only licenses a library model may be used under (rights are a hard stop), with their names and deeds.
ALLOWED_LICENSES = {"CC-BY-4.0": {"name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"},
                    "CC0-1.0": {"name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"}}
#: Every spelling a source uses for an allowed license, matched exactly after trimming: nothing is read loosely.
LICENSE_SPELLINGS = {
    **dict.fromkeys(("CC-BY-4.0", "cc-by-4.0", "CC BY 4.0", "CC-BY 4.0", "Creative Commons Attribution 4.0 International",
                     "http://creativecommons.org/licenses/by/4.0/", "https://creativecommons.org/licenses/by/4.0/"),
                    "CC-BY-4.0"),
    **dict.fromkeys(("CC0-1.0", "cc0-1.0", "CC0 1.0", "CC0 1.0 Universal",
                     "http://creativecommons.org/publicdomain/zero/1.0/",
                     "https://creativecommons.org/publicdomain/zero/1.0/"), "CC0-1.0")}
#: Datasets left out whole for now, listed but never used (rights is a hard stop). M-XR's attribution files ask for
#: credit as "marso by M-XR" with a web link (CC BY 4.0 3(a)(1)(A)(i): "in any reasonable manner requested by the
#: Licensor"), and the package scan refuses an unlisted link, so the README cannot give it (coordinator, 2026-11-23).
DATASETS_REFUSED = {"M-XR/marso-simready": "attribution_link_not_carried"}
#: Why a sealed choice's model is no longer used (``rights_now``): its entry is gone from the deployed catalog, or that
#: catalog now refuses it (its license, its dataset, its credit), or names another license for it.
NOT_IN_CATALOG = "website_object_library_entry_not_in_catalog"
REFUSED_NOW = "website_object_library_license_refused_now"
LICENSE_CHANGED = "website_object_library_license_changed"


def license_of(statements: Sequence[Sequence[Any]]) -> tuple[str | None, str | None]:
    """The one allowed license every statement names, or None with why it is refused (``license_missing``,
    ``license_not_allowed:<where>``, ``licenses_disagree``). Rights are a hard stop: never read loosely."""
    if not statements:
        return None, "license_missing"
    found = set()
    for where, text in statements:
        spdx = LICENSE_SPELLINGS.get(text.strip()) if isinstance(text, str) else None
        if spdx is None:
            return None, f"license_not_allowed:{where}"
        found.add(spdx)
    return (found.pop(), None) if len(found) == 1 else (None, "licenses_disagree")


def _gso_entries(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Google Scanned Objects records as catalog entries; their links follow the index's own templates."""
    rows = []
    for record in index["records"]:
        rights = index["rights"][record["rights"]]
        license = rights.get("license") or {}
        name, version = record["name"], record["version"]
        # Fuel's description: the product's name on its first line, sometimes the packaging's text below it.
        title = (str(record.get("title") or "").split("\n")[0].strip() or name.replace("_", " "))

        def link(template: str) -> str:
            return template.format(name=urllib.parse.quote(name), version=version)
        rows.append({
            "id": f"gso:{name}@{version}", "library": index["library"], "library_title": index["library_title"],
            "title": title, "author": ", ".join(rights.get("authors") or []) or None,
            "copyright": rights.get("copyright") or None, "disclaimer": rights.get("disclaimer") or None,
            "words": [title, name.replace("_", " "), record.get("brand") or ""],
            "categories": list(record.get("categories") or []), "size_m": record.get("size_m"), "mass_kg": None,
            "license_statements": [("Gazebo Fuel's license", license.get("fuel_license_name")),
                                   ("Gazebo Fuel's license link", license.get("fuel_license_url")),
                                   ("the model's own metadata", license.get("metadata_license"))],
            "item_url": link(index["item_url"]), "format": "obj",
            "files": [{"role": "mesh", "url": link(index["mesh_url"]), "sha256": record["mesh"]["sha256"],
                       "size_bytes": record["mesh"]["size_bytes"], "suffix": ".obj"}],
            "thumbnail": {"url": link(index["thumbnail_url"]), "sha256": record["thumbnail"]["sha256"],
                          "size_bytes": record["thumbnail"]["size_bytes"], "suffix": ".jpg"}})
    return rows


SIMREADY_BOM = ".metadata/com.nvidia.simready.packaging.bom.json"


def _simready_entries(index: Mapping[str, Any]) -> list[dict[str, Any]]:
    """SimReady package records as catalog entries, every link at the commit NVIDIA reviewed: the package definition
    (the item's link), its bill of materials (which names every file and its sha256) and its 256 px thumbnail."""
    rows = []
    for record in index["records"]:
        source, rights = index["sources"][record["source"]], index["rights"][record["rights"]]
        repo, commit, asset = source["repo"], source["commit"], record["asset"]

        def link(path: str) -> str:
            return index["file_url"].format(repo=repo, commit=commit, asset=urllib.parse.quote(asset),
                                            path=urllib.parse.quote(path))
        root = PurePosixPath(record["root"])
        thumbnail = record.get("thumbnail")
        named = str(record.get("asset_name") or "")
        rows.append({
            "id": f"simready:{repo}/{asset}@{commit[:12]}", "library": index["library"], "dataset": repo,
            "library_title": f"the SimReady dataset {repo}",
            "attribution_request": [str(line) for line in record.get("attribution_request") or []],
            # The package's own name; the USD's asset_name beside it when that is a name in words.
            "title": asset + (f" ({named})" if " " in named.strip() else ""), "author": rights.get("author"),
            "copyright": rights.get("copyright") or None, "disclaimer": rights.get("disclaimer") or None,
            "words": [asset.replace("_", " "), str(record.get("asset_name") or "")],
            "categories": [str(record["category"])] if record.get("category") else [],
            "size_m": record.get("size_m"), "mass_kg": record.get("mass_kg"),
            "license_statements": [("the package definition", rights.get("package_license")),
                                   ("the USD's own SimReady_Metadata", rights.get("asset_license"))],
            "item_url": index["item_url"].format(repo=repo, commit=commit, asset=urllib.parse.quote(asset)),
            "format": "simready_usd", "package": {"base_url": link(""), "root": str(root)},
            "files": [{"role": "bom", "url": link(SIMREADY_BOM), "sha256": record["bom"]["sha256"],
                       "size_bytes": record["bom"]["size_bytes"], "suffix": ".json"}],
            **({"thumbnail": {"url": link(str(root.parent / ".thumbs" / "256x256" / (root.name + ".png"))),
                              "sha256": thumbnail["sha256"], "size_bytes": thumbnail["size_bytes"], "suffix": ".png"}}
               if thumbnail else {})})
    return rows


#: How each library's index becomes catalog entries.
READERS: dict[str, Callable[[Mapping[str, Any]], list[dict[str, Any]]]] = {"google_scanned_objects": _gso_entries,
                                                                           "nvidia_simready": _simready_entries}


#: Why a library offers nothing (``catalog_status``): its index, or one of the parts it lists, is unreadable, missing
#: or not the bytes or record count the index names. Never a partial library.
INDEX_UNREADABLE, PART_MISSING, PART_MISMATCH = ("library_index_unreadable", "library_index_part_missing",
                                                 "library_index_part_mismatch")


def read_index(path: Path) -> dict[str, Any]:
    """A library's index with its records: in the file itself, or in the parts it lists beside it (``parts``: each
    file's name, sha256 and record count), each read and checked. Raises with ``INDEX_UNREADABLE``, ``PART_MISSING``
    or ``PART_MISMATCH`` and the part's name: a library is read whole or not at all."""
    path = Path(path)
    try:
        index = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(INDEX_UNREADABLE) from exc
    if "parts" not in index:
        return index
    records: list[Any] = []
    for part in index["parts"]:
        name = str(part.get("file") or "")
        target = path.parent / name
        if not name or "/" in name or target.is_symlink() or not target.is_file():
            raise ValueError(f"{PART_MISSING}:{name}")
        data = target.read_bytes()
        try:
            rows = json.loads(data)["records"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{PART_MISMATCH}:{name}") from exc
        if (hashlib.sha256(data).hexdigest() != part.get("sha256") or not isinstance(rows, list)
                or len(rows) != part.get("record_count")):
            raise ValueError(f"{PART_MISMATCH}:{name}")
        records += rows
    return {**index, "records": records}


@lru_cache(maxsize=8)
def _catalog(paths: tuple[tuple[str, str], ...], refused: tuple[tuple[str, str], ...] = ()
             ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    datasets_refused = dict(refused)
    entries: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    for library, path in paths:
        try:
            index = read_index(Path(path))
        except ValueError as exc:
            # A library whose index or any part of it cannot be read offers nothing, said why; the others still do.
            status.append({"library": library, "status": "unavailable", "reason": str(exc)[:200]})
            continue
        status.append({"library": library, "status": "read", "records": len(index["records"])})
        for entry in READERS[library](index):
            spdx, why = license_of(entry["license_statements"])
            if spdx is not None and not entry.get("author"):
                spdx, why = None, "author_not_stated"  # the credit names who made it, or the model is not used
            elif spdx is not None and entry.get("dataset") in datasets_refused:
                spdx, why = None, datasets_refused[entry["dataset"]]
            elif spdx is not None and attribution_refusal(entry, spdx):
                spdx, why = None, attribution_refusal(entry, spdx)
            elif spdx is not None and not credit_shippable(entry, spdx):
                spdx, why = None, "credit_not_shippable"  # a credit the package scan would refuse is never owed
            entries.append({**entry, "license": spdx, **({"license_refused": why} if why else {})})
    return tuple(entries), tuple(status)


_LINK = re.compile(r"https?://[^\s)\]>,]+")
_LABEL = re.compile(r"(?i)^\s*(attribution|credit|source|licen[cs]e|licensor)\s*:\s*")


def _credit(entry: Mapping[str, Any], spdx: str) -> str:
    """The entry's credit as the company README would word it (``website_library_notes.credit_text``)."""
    from .website_library_notes import MODEL_FILE, credit_text
    license = ALLOWED_LICENSES[spdx]
    return credit_text({"title": entry["title"], "author": entry.get("author"), "copyright": entry.get("copyright"),
                        "disclaimer": entry.get("disclaimer"), "library": entry["library_title"],
                        "link": entry["item_url"], "license": license["name"], "license_url": license["url"],
                        "changes": "converted it and scaled it"}, [MODEL_FILE, "object.usdz"])


def attribution_refusal(entry: Mapping[str, Any], spdx: str) -> str | None:
    """Why the README cannot credit the entry the way its library asks (``attribution_request``, read from its notice
    or attribution file), or None when it can or nothing is asked: a link the credit does not give (only the license's
    deed and the item's own link are given) is ``attribution_link_not_carried``; a name or wording it does not give is
    ``attribution_manner_not_carried``. A request is never dropped: the model is refused instead."""
    request = [str(line) for line in entry.get("attribution_request") or []]
    if not request:
        return None
    credit = _credit(entry, spdx)
    deed = ALLOWED_LICENSES[spdx]["url"]
    for line in request:
        for link in _LINK.findall(line):
            link = link.rstrip(".")
            if link not in (deed, deed.replace("https://", "http://", 1)) and link not in credit:
                return "attribution_link_not_carried"
    for line in request:
        wanted = _LINK.sub("", _LABEL.sub("", line)).strip(" .()")
        if wanted and wanted not in credit:
            return "attribution_manner_not_carried"
    return None


def credit_shippable(entry: Mapping[str, Any], spdx: str) -> bool:
    """Whether the entry's credit, as the company README would word it, passes the package's outbound scan (an
    author named by a web address would refuse the whole package): a model whose credit cannot ship is not used."""
    import io
    from .outbound_file_scan import scan_stream
    return scan_stream(io.BytesIO(_credit(entry, spdx).encode()), [], text=True) is None


def _paths(indexes: Mapping[str, Path] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(path)) for key, path in (indexes or INDEXES).items()))


def _refused() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(DATASETS_REFUSED.items()))


def catalog(indexes: Mapping[str, Path] | None = None) -> list[dict[str, Any]]:
    """Every entry of every library (the deployed catalog unless ``indexes`` names others), each with its license read
    (``license``) or why it is refused."""
    return list(_catalog(_paths(indexes), _refused())[0])


def catalog_status(indexes: Mapping[str, Path] | None = None) -> list[dict[str, Any]]:
    """Each library's state: ``read`` with its record count, or ``unavailable`` with why (``read_index``)."""
    return list(_catalog(_paths(indexes), _refused())[1])


def rights_now(attribution: Any, entries: Sequence[Mapping[str, Any]] | None = None) -> str | None:
    """Why a model chosen under an earlier catalog may no longer be used, or None when it still may: its entry
    (``attribution["entry_id"]``) is looked up in ``entries`` (by default the deployed catalog, read now), and must
    still be there (``NOT_IN_CATALOG``, which a library that cannot be read whole also gives), allowed by the catalog
    (``REFUSED_NOW`` with its reason: its license, its dataset or its credit) and under the very license the choice
    named (``LICENSE_CHANGED``). Rights are a hard stop: anything that cannot be confirmed is refused."""
    if not isinstance(attribution, Mapping) or not isinstance(attribution.get("entry_id"), str):
        return NOT_IN_CATALOG
    every = catalog() if entries is None else entries
    entry = next((row for row in every if isinstance(row, Mapping) and row.get("id") == attribution["entry_id"]), None)
    if entry is None:
        return NOT_IN_CATALOG
    if entry.get("license") not in ALLOWED_LICENSES:
        return f"{REFUSED_NOW}:{entry.get('license_refused') or 'license_missing'}"[:300]
    if entry["license"] != attribution.get("license"):
        return f"{LICENSE_CHANGED}:{attribution.get('license')}->{entry['license']}"[:300]
    return None


__all__ = ["ALLOWED_LICENSES", "DATA", "DATASETS_REFUSED", "INDEXES", "INDEX_UNREADABLE", "LICENSE_CHANGED",
           "LICENSE_SPELLINGS", "NOT_IN_CATALOG", "PART_MISMATCH", "PART_MISSING", "READERS", "REFUSED_NOW",
           "SHIPPED_INDEXES", "SIMREADY_BOM", "attribution_refusal", "catalog", "catalog_status", "credit_shippable",
           "license_of", "read_index", "rights_now"]
