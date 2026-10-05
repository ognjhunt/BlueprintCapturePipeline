"""Capability taxonomy: load, validate and classify records against ``taxonomy.json``.

The taxonomy is data. Rows map a robot capability to task families and site
types; site types carry the per-source selectors (NAICS and SIC prefixes,
OpenStreetMap tag filters, FSIS directory membership). The builder enumerates
the union of the site types of ``active`` rows. Category mapping comes from the
site type, never from code branches.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tools.site_universe import normalize

TAXONOMY_SCHEMA = "blueprint.site_universe.taxonomy.v1"
TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")
ROW_STATUSES = ("active", "candidate")
MAX_OVERPASS_QUERIES_PER_STATE = 4

# OpenStreetMap keys a selector may use. Anything else is refused.
OSM_SELECTOR_KEYS = frozenset(
    {"aeroway", "amenity", "building", "craft", "healthcare", "industrial", "landuse", "man_made",
     "power", "shop", "tourism"}
)
# Values that describe homes, personal practices or non-business features.
OSM_REFUSED_VALUES = {
    "building": {
        "apartments", "bungalow", "cabin", "carport", "detached", "dormitory", "farm", "garage",
        "garages", "house", "houseboat", "hut", "residential", "semidetached_house", "shed",
        "static_caravan", "terrace", "yes",
    },
    "landuse": {"allotments", "cemetery", "farmyard", "garages", "religious", "residential"},
    "amenity": {
        "bench", "dentist", "doctors", "grave_yard", "parking", "place_of_worship", "recycling",
        "shelter", "social_facility", "toilets",
    },
    "healthcare": {
        "alternative", "counselling", "dentist", "doctor", "midwife", "nurse", "physiotherapist",
        "psychotherapist",
    },
}
_OSM_TOKEN = re.compile(r"^[a-z][a-z0-9_:]*$")
_DIGITS = re.compile(r"^[0-9]{2,6}$")
_GROUP_ORDER = ("buildings", "industrial_areas", "retail", "amenities")


class TaxonomyError(ValueError):
    """taxonomy.json does not satisfy the schema."""


@dataclass(frozen=True)
class SiteType:
    id: str
    category: str
    label: str
    naics_prefixes: tuple[str, ...]
    naics_exclude_prefixes: tuple[str, ...]
    sic_prefixes: tuple[str, ...]
    sic_exclude_prefixes: tuple[str, ...]
    osm: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    name_excludes: tuple[str, ...]
    fsis_mpi: bool
    generic: bool
    exclusion: bool


@dataclass(frozen=True)
class Classification:
    site_types: tuple[str, ...] = ()
    primary: str | None = None
    matched_by: dict = field(default_factory=dict)
    drop_reason: str | None = None


def file_sha256(path: Path = TAXONOMY_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path | str = TAXONOMY_PATH) -> Taxonomy:
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    return Taxonomy(document, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _strings(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise TaxonomyError(f"{where}: expected a list of non-empty strings")
    if len(set(value)) != len(value):
        raise TaxonomyError(f"{where}: duplicate entries")
    return tuple(value)


def _codes(value, where: str) -> tuple[str, ...]:
    codes = _strings(value, where)
    for code in codes:
        if not _DIGITS.match(code):
            raise TaxonomyError(f"{where}: {code!r} is not a 2-6 digit code prefix")
    return codes


def _osm_selector(selector, where: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(selector, dict) or not selector:
        raise TaxonomyError(f"{where}: an OSM selector is a non-empty tag map")
    out = []
    for key in sorted(selector):
        raw = selector[key]
        values = (raw,) if isinstance(raw, str) else tuple(raw) if isinstance(raw, list) else None
        if not values or not all(isinstance(value, str) for value in values):
            raise TaxonomyError(f"{where}: tag {key!r} needs a value or a list of values")
        if key not in OSM_SELECTOR_KEYS:
            raise TaxonomyError(f"{where}: OSM key {key!r} is not allowed as a selector")
        for value in values:
            if not _OSM_TOKEN.match(value) or value == "yes":
                raise TaxonomyError(f"{where}: OSM value {value!r} must pin a specific tag value")
            if value in OSM_REFUSED_VALUES.get(key, ()):
                raise TaxonomyError(
                    f"{where}: {key}={value} selects homes or personal features and is refused"
                )
        out.append((key, tuple(sorted(set(values)))))
    return tuple(out)


def _evidence(items, where: str, *, required: bool) -> None:
    if not isinstance(items, list):
        raise TaxonomyError(f"{where}: evidence must be a list")
    if required and not items:
        raise TaxonomyError(f"{where}: an active row needs at least one evidence item")
    for index, item in enumerate(items):
        spot = f"{where}.evidence[{index}]"
        if not isinstance(item, dict):
            raise TaxonomyError(f"{spot}: evidence must be an object")
        for key in ("claim", "url", "publisher", "verified_at"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise TaxonomyError(f"{spot}: missing {key}")
        if not item["url"].startswith("https://"):
            raise TaxonomyError(f"{spot}: evidence URL must be https")
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", item["verified_at"]):
            raise TaxonomyError(f"{spot}: verified_at must be YYYY-MM-DD")


class Taxonomy:
    def __init__(self, document: dict, *, sha256: str | None = None):
        self.document = document
        self.sha256 = sha256 or hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._validate()

    # -- validation ---------------------------------------------------------------
    def _validate(self) -> None:
        doc = self.document
        if doc.get("schema") != TAXONOMY_SCHEMA:
            raise TaxonomyError(f"schema must be {TAXONOMY_SCHEMA}")
        for key in ("version", "updated_at"):
            if not isinstance(doc.get(key), str) or not doc[key]:
                raise TaxonomyError(f"missing {key}")
        categories = doc.get("categories")
        if not isinstance(categories, dict) or not categories:
            raise TaxonomyError("categories must be a non-empty object")
        self.version = doc["version"]
        self.categories = {key: value.get("label") for key, value in categories.items()}

        self.site_types: dict[str, SiteType] = {}
        self.site_type_order: list[str] = []
        for index, raw in enumerate(doc.get("site_types") or []):
            where = f"site_types[{index}]"
            if not isinstance(raw, dict):
                raise TaxonomyError(f"{where}: must be an object")
            type_id = raw.get("id")
            if not isinstance(type_id, str) or not re.match(r"^[a-z][a-z0-9_]*$", type_id):
                raise TaxonomyError(f"{where}: invalid id")
            if type_id in self.site_types:
                raise TaxonomyError(f"{where}: duplicate site type {type_id}")
            if raw.get("category") not in categories:
                raise TaxonomyError(f"{where}: unknown category {raw.get('category')!r}")
            selectors = raw.get("selectors")
            if not isinstance(selectors, dict):
                raise TaxonomyError(f"{where}: selectors must be an object")
            known = {
                "naics_prefixes", "naics_exclude_prefixes", "sic_prefixes", "sic_exclude_prefixes",
                "osm", "name_excludes", "fsis_mpi",
            }
            unknown = set(selectors) - known
            if unknown:
                raise TaxonomyError(f"{where}: unknown selector keys {sorted(unknown)}")
            osm = tuple(
                _osm_selector(item, f"{where}.osm[{position}]")
                for position, item in enumerate(selectors.get("osm") or [])
            )
            fsis = selectors.get("fsis_mpi")
            if fsis not in (None, "all_establishments"):
                raise TaxonomyError(f"{where}: fsis_mpi must be 'all_establishments' when present")
            site_type = SiteType(
                id=type_id,
                category=raw["category"],
                label=str(raw.get("label") or type_id),
                naics_prefixes=_codes(selectors.get("naics_prefixes", []), f"{where}.naics_prefixes"),
                naics_exclude_prefixes=_codes(
                    selectors.get("naics_exclude_prefixes", []), f"{where}.naics_exclude_prefixes"
                ),
                sic_prefixes=_codes(selectors.get("sic_prefixes", []), f"{where}.sic_prefixes"),
                sic_exclude_prefixes=_codes(
                    selectors.get("sic_exclude_prefixes", []), f"{where}.sic_exclude_prefixes"
                ),
                osm=osm,
                name_excludes=tuple(
                    normalize.ascii_upper(item)
                    for item in _strings(selectors.get("name_excludes", []), f"{where}.name_excludes")
                ),
                fsis_mpi=fsis == "all_establishments",
                generic=bool(raw.get("generic", False)),
                exclusion=bool(raw.get("exclusion", False)),
            )
            if site_type.exclusion and (site_type.generic or site_type.naics_prefixes):
                raise TaxonomyError(f"{where}: an exclusion site type takes OSM selectors only")
            if not (site_type.naics_prefixes or site_type.sic_prefixes or site_type.osm or site_type.fsis_mpi):
                raise TaxonomyError(f"{where}: a site type needs at least one selector")
            self.site_types[type_id] = site_type
            self.site_type_order.append(type_id)
        if not self.site_types:
            raise TaxonomyError("site_types must be a non-empty list")
        for system in ("naics", "sic"):
            seen: dict[str, str] = {}
            for type_id in self.site_type_order:
                for prefix in getattr(self.site_types[type_id], f"{system}_prefixes"):
                    if prefix in seen:
                        raise TaxonomyError(
                            f"{system} prefix {prefix} appears in {seen[prefix]} and {type_id}"
                        )
                    seen[prefix] = type_id

        self.rows: list[dict] = []
        row_ids: set[str] = set()
        referenced: set[str] = set()
        for index, row in enumerate(doc.get("rows") or []):
            where = f"rows[{index}]"
            if not isinstance(row, dict):
                raise TaxonomyError(f"{where}: must be an object")
            for key in ("row_id", "capability", "label", "status", "added_at"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise TaxonomyError(f"{where}: missing {key}")
            if row["row_id"] in row_ids:
                raise TaxonomyError(f"{where}: duplicate row_id {row['row_id']}")
            if not re.match(r"^[a-z][a-z0-9_.]*$", row["row_id"]):
                raise TaxonomyError(f"{where}: invalid row_id")
            if row["status"] not in ROW_STATUSES:
                raise TaxonomyError(f"{where}: status must be one of {ROW_STATUSES}")
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", row["added_at"]):
                raise TaxonomyError(f"{where}: added_at must be YYYY-MM-DD")
            _strings(row.get("task_families"), f"{where}.task_families")
            for type_id in _strings(row.get("site_types"), f"{where}.site_types"):
                if type_id not in self.site_types:
                    raise TaxonomyError(f"{where}: unknown site type {type_id}")
                if self.site_types[type_id].exclusion:
                    raise TaxonomyError(f"{where}: {type_id} is an exclusion site type")
                referenced.add(type_id)
            _evidence(row.get("evidence"), where, required=row["status"] == "active")
            row_ids.add(row["row_id"])
            self.rows.append(row)
        if not self.rows:
            raise TaxonomyError("rows must be a non-empty list")
        orphans = [
            type_id for type_id in self.site_type_order
            if type_id not in referenced and not self.site_types[type_id].exclusion
        ]
        if orphans:
            raise TaxonomyError(f"site types not used by any row: {orphans}")

        self.active_rows = [row for row in self.rows if row["status"] == "active"]
        self.active_site_types = frozenset(
            type_id for row in self.active_rows for type_id in row["site_types"]
        )
        self.rows_by_site_type: dict[str, tuple[str, ...]] = {}
        self.capabilities_by_site_type: dict[str, tuple[str, ...]] = {}
        for type_id in self.active_site_types:
            rows = [row for row in self.active_rows if type_id in row["site_types"]]
            self.rows_by_site_type[type_id] = tuple(sorted(row["row_id"] for row in rows))
            self.capabilities_by_site_type[type_id] = tuple(sorted({row["capability"] for row in rows}))
        self._order = {type_id: index for index, type_id in enumerate(self.site_type_order)}

    # -- classification -------------------------------------------------------------
    def _code_site_type(self, code: str, system: str) -> str | None:
        best: tuple[int, str] | None = None
        for type_id in self.site_type_order:
            site_type = self.site_types[type_id]
            if site_type.exclusion:
                continue
            prefixes = getattr(site_type, f"{system}_prefixes")
            excludes = getattr(site_type, f"{system}_exclude_prefixes")
            for prefix in prefixes:
                if (
                    code.startswith(prefix)
                    and not any(code.startswith(item) for item in excludes)
                    and (best is None or len(prefix) > best[0])
                ):
                    best = (len(prefix), type_id)
        return best[1] if best else None

    def classify_codes(self, codes, *, system: str = "naics", primary: str | None = None) -> Classification:
        """Classify industry codes; each code maps to its longest-prefix site type."""
        matched: dict[str, list[str]] = {}
        primary_type = None
        for code in sorted({c for c in codes if c}):
            type_id = self._code_site_type(code, system)
            if type_id is None:
                continue
            matched.setdefault(type_id, []).append(f"{system}:{code}")
            if code == primary:
                primary_type = type_id
        return self._finish(matched, primary_type, "no_matching_code")

    def classify_fsis(self) -> Classification:
        matched = {
            type_id: ["fsis_mpi:directory"]
            for type_id in self.site_type_order
            if self.site_types[type_id].fsis_mpi
        }
        return self._finish(matched, None, "no_fsis_site_type")

    def classify_osm(self, tags: dict, name: str | None) -> Classification:
        matched: dict[str, list[str]] = {}
        for type_id in self.site_type_order:
            site_type = self.site_types[type_id]
            for selector in site_type.osm:
                if all(tags.get(key) in values for key, values in selector):
                    label = ",".join(f"{key}={tags[key]}" for key, _ in selector)
                    matched.setdefault(type_id, []).append(f"osm:{label}")
        if any(self.site_types[type_id].exclusion for type_id in matched):
            return Classification(drop_reason="excluded_tag")
        if any(not self.site_types[type_id].generic for type_id in matched):
            matched = {k: v for k, v in matched.items() if not self.site_types[k].generic}
        upper = " " + " ".join(normalize.ascii_upper(name).replace("-", " ").split()) + " "
        for type_id in list(matched):
            for phrase in self.site_types[type_id].name_excludes:
                if f" {phrase} " in upper:
                    matched.pop(type_id)
                    break
        if not matched:
            return Classification(drop_reason="excluded_name")
        return self._finish(matched, None, "no_matching_tag")

    def _finish(self, matched: dict, primary_type: str | None, empty_reason: str) -> Classification:
        if not matched:
            return Classification(drop_reason=empty_reason)
        active = {k: tuple(v) for k, v in matched.items() if k in self.active_site_types}
        if not active:
            return Classification(drop_reason="inactive_site_type")
        ordered = tuple(sorted(active, key=lambda type_id: self._order[type_id]))
        if primary_type not in active:
            primary_type = ordered[0]
        return Classification(site_types=ordered, primary=primary_type, matched_by=active)

    def category(self, site_type: str | None) -> str | None:
        return self.site_types[site_type].category if site_type else None

    def matches(self, site_types) -> dict:
        """taxonomy_matches block for a set of active site types."""
        types = sorted(set(site_types), key=lambda type_id: self._order[type_id])
        rows = sorted({row for type_id in types for row in self.rows_by_site_type.get(type_id, ())})
        capabilities = sorted(
            {cap for type_id in types for cap in self.capabilities_by_site_type.get(type_id, ())}
        )
        return {"capabilities": capabilities, "rows": rows, "site_types": types}

    # -- Overpass ---------------------------------------------------------------------
    def osm_selectors(self) -> list[tuple[tuple[str, tuple[str, ...]], ...]]:
        """Distinct OSM selectors of active, non-exclusion site types, in a stable order."""
        seen = []
        for type_id in self.site_type_order:
            if type_id not in self.active_site_types:
                continue
            for selector in self.site_types[type_id].osm:
                if selector not in seen:
                    seen.append(selector)
        return sorted(seen)

    def overpass_queries(self, state: str, *, timeout_s: int = 300) -> list[tuple[str, str]]:
        """Bounded Overpass QL queries for one state: ``[(group, query), ...]``."""
        state = normalize.normalize_state(state)
        if not state or state not in normalize.STATE_NAMES:
            raise TaxonomyError(f"unknown state code {state!r}")
        groups: dict[str, list[str]] = {}
        for selector in self.osm_selectors():
            keys = {key for key, _ in selector}
            if "shop" in keys:
                group = "retail"
            elif keys & {"building", "man_made"}:
                group = "buildings"
            elif keys & {"landuse", "industrial", "power"}:
                group = "industrial_areas"
            else:
                group = "amenities"
            filters = "".join(
                f'["{key}"="{values[0]}"]' if len(values) == 1
                else f'["{key}"~"^({"|".join(values)})$"]'
                for key, values in selector
            )
            groups.setdefault(group, []).append(f'  nwr{filters}["name"](area.state);')
        if len(groups) > MAX_OVERPASS_QUERIES_PER_STATE:
            raise TaxonomyError(
                f"{len(groups)} Overpass queries per state exceeds the bound of "
                f"{MAX_OVERPASS_QUERIES_PER_STATE}"
            )
        header = (
            f"[out:json][timeout:{int(timeout_s)}];\n"
            f'area["ISO3166-2"="US-{state}"]["admin_level"="4"]->.state;\n'
        )
        queries = []
        for group in _GROUP_ORDER:
            if group in groups:
                body = "(\n" + "\n".join(sorted(groups[group])) + "\n);\nout geom;"
                queries.append((group, header + body))
        return queries

    def summary(self) -> dict:
        return {
            "active_rows": sorted(row["row_id"] for row in self.active_rows),
            "active_site_types": sorted(self.active_site_types),
            "candidate_rows": sorted(row["row_id"] for row in self.rows if row["status"] == "candidate"),
            "sha256": self.sha256,
            "version": self.version,
        }
