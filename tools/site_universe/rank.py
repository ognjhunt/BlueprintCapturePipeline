"""Ranking v1: transparent, deterministic scoring of a site universe snapshot.

``score(site, config)`` adds weighted components. Each component has a value
in [-1, 1] and a weight in points; the weights in ``rank_config.json`` sum to
100, so a score reads as points out of 100. The components measure fit
evidence only (capability and site type fit, NAICS task evidence, size,
operator scale, activity, ownership, source corroboration and location). No
component measures or claims buying interest.

Exclusion rules come from ``rank_config.json`` and from an optional exclusions
input file (for example CRM rows and recent rejections). An excluded site keeps
its score and components and is written after the ranked sites with the ids of
the rules that matched. Nothing is dropped silently.

``rank(snapshot_dir, config)`` checks the snapshot's SHA-256, scores every site
in scope and orders the ranked sites by score, highest first, with ties broken
by ``site_id``. Nothing reads the clock, so the same snapshot, config and
inputs give byte-identical outputs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from tools.site_universe import SCHEMA_VERSION, normalize
from tools.site_universe import build as build_module
from tools.site_universe import taxonomy as taxonomy_module

RANK_CONFIG_PATH = Path(__file__).with_name("rank_config.json")
RANK_CONFIG_SCHEMA = "blueprint.site_universe.rank_config.v1"
RANK_MANIFEST_SCHEMA = "blueprint.site_universe.rank_manifest.v1"
RANKER_VERSION = "1"
RANKED_FILE = "ranked.jsonl.gz"
RANK_MANIFEST_FILE = "rank-manifest.json"
REVIEW_FILE = "review-top.md"
REVIEW_TOP = 25
REVIEW_EXCLUDED_PER_RULE = 5
PERCENTILES = (0, 10, 25, 50, 75, 90, 99, 100)
COMPONENTS = (
    "capability_fit", "task_evidence", "size_fit", "category_fit", "operator_scale",
    "activity_evidence", "ownership", "source_corroboration", "location_quality",
)
PHRASE_FIELDS = ("name", "names", "operator")
MATCHERS = ("phrases", "naics", "frs_activity_status", "missing_street_and_coordinates")
ACTIVITY_KEYS = frozenset(
    {"osha_recent_filing", "fsis_listed", "frs_active", "frs_unknown", "frs_inactive", "none"}
)
LOCATION_KEYS = frozenset({"street_address", "precise_coordinates", "other_coordinates"})
CONFIG_KEYS = (
    "activity_evidence", "capability_scope", "capability_weights", "category_weights", "exclusions",
    "location_quality", "operator_scale", "ownership", "schema", "score_meaning",
    "size", "source_corroboration", "task_evidence", "updated_at", "version", "weights",
)
CONFIG_OPTIONAL_KEYS = ("description", "site_type_weight_notes", "site_type_weights")
SECTION_KEYS = {  # section -> (required keys, optional keys)
    "activity_evidence": (("osha_filing_year_at_least", "values"), ("note",)),
    "capability_scope": (("secondary_site_type_factor", "unspecific_site_types"), ("note",)),
    "location_quality": (tuple(sorted(LOCATION_KEYS)), ("note",)),
    "operator_scale": (("interpolation", "numbered_unit_assumed_sites", "site_count_curve", "unknown"),
                       ("note", "numbered_unit_note")),
    "ownership": (("phrase_fields", "public_naics_prefixes", "public_phrases", "values"), ("note",)),
    "size": (("building_area_confidence", "building_area_m2_curve", "employees_curve", "interpolation",
              "unknown"), ("note",)),
    "source_corroboration": (("values_by_source_count",), ("note",)),
    "task_evidence": (("default", "naics_by_capability"), ("note",)),
}
RULE_KEYS = ("evidence", "id", "label", "match", "reason")
EVIDENCE_KEYS = ("note", "url", "verified_at")
INPUT_KEYS = ("site_id", "name", "operator")
INPUT_FIELDS = frozenset(
    INPUT_KEYS + ("reason", "source", "city", "state", "postal_code", "added_at", "note")
)
_PHRASE = re.compile(r"^[A-Z0-9]+( [A-Z0-9]+)*$")
_ACRONYM = re.compile(r"^[A-Z][A-Z0-9]{1,5}$")
_NON_WORD = re.compile(r"[^A-Z0-9]+")
_RULE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_INPUT_SOURCE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PREFIX = re.compile(r"^[0-9]{2,6}$")
_SITE_ID = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_COUNT = re.compile(r"^[1-9][0-9]*$")
# Unit numbers in names: "0418 HARDWARE MART OF LARKSTONE", "GROCER #962", "Market Store 718 / 2046".
_LEADING_UNIT = re.compile(r"^\s*(\d{3,6})\s*(?:[-/:]\s*)?([A-Za-z].*)$")
_HASH_UNIT = re.compile(r"#\s*\d{2,}")
_STORE_UNIT = re.compile(r"\bSTORE\s*#?\s*\d{2,}\b", re.IGNORECASE)
# Thoroughfare words: "2001 Example Road" is an address used as a name, not a unit number.
_STREET_WORDS = frozenset({
    "AVE", "AVENUE", "BLVD", "BOULEVARD", "CIR", "CIRCLE", "CT", "COURT", "DR", "DRIVE", "EXPY",
    "EXPRESSWAY", "FWY", "FREEWAY", "HWY", "HIGHWAY", "LN", "LANE", "LOOP", "PIKE", "PKWY",
    "PARKWAY", "PL", "PLACE", "RD", "ROAD", "ST", "STREET", "TPKE", "TURNPIKE", "TRL", "TRAIL",
    "WAY",
})
# Words that end the company part of an operator string, as in OSHA ITA's
# "Pecan Healthcare PH WEXMOOR COUNTY LARKSTONE" or "Hardware Mart Companies, INC HARDWARE MART OF LARKSTONE".
# Legal forms that may follow a company acronym in an operator ('RTX Corporation').
LEGAL_FORMS = frozenset({
    "CO", "COMPANY", "CORP", "CORPORATION", "INC", "INCORPORATED", "LIMITED", "LLC", "LLP", "LP", "LTD",
    "PLLC",
})
STEM_WORDS = frozenset({
    "CO", "COMPANIES", "COMPANY", "CORP", "CORPORATION", "DBA", "ENTERPRISES", "GROUP", "HEALTH",
    "HEALTHCARE", "HOLDING", "HOLDINGS", "INC", "INCORPORATED", "INDUSTRIES", "LIMITED", "LLC",
    "LLP", "LP", "LTD", "PLLC", "SYSTEM", "SYSTEMS",
})


class RankError(ValueError):
    """The rank config, the snapshot or the exclusions input breaks the ranking contract."""


def _require(condition, message: str) -> None:
    if not condition:
        raise RankError(message)


# --- text and curves ------------------------------------------------------------------------
def phrase_text(value) -> str:
    """Uppercase ASCII words with single spaces, padded for whole-word search ('' when empty)."""
    words = _NON_WORD.sub(" ", normalize.ascii_upper(value)).split()
    return f" {' '.join(words)} " if words else ""


def interpolate(curve, x: float) -> float:
    """Piecewise-linear value in log10(x) between ``(x, value)`` points, clamped at both ends."""
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for (x0, v0), (x1, v1) in pairwise(curve):
        if x <= x1:
            fraction = (math.log10(x) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
            return v0 + (v1 - v0) * fraction
    return curve[-1][1]


# --- config ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    id: str
    label: str
    reason: str
    evidence: tuple
    phrase_fields: tuple = ()
    phrase_entries: tuple = ()  # ((label, (phrase, ...)), ...)
    acronym_entries: tuple = ()  # ((label, (acronym, ...)), ...): company acronyms such as UPS
    unless_phrases: tuple = ()
    naics_scope: str | None = None
    naics_prefixes: tuple = ()
    frs_values: tuple = ()
    frs_unless_osha_year: int | None = None
    frs_unless_fsis: bool = False
    missing_location: bool = False


@dataclass(frozen=True)
class RankConfig:
    document: dict
    sha256: str
    version: str
    weights: dict
    capability_weights: dict
    category_weights: dict
    site_type_weights: dict
    employees_curve: tuple
    building_area_curve: tuple
    building_area_confidence: float
    size_unknown: float
    unspecific_site_types: frozenset
    secondary_factor: float
    task_default: float
    task_map: dict  # capability -> ((prefix, value, label), ...), longest prefix first
    operator_curve: tuple
    operator_unknown: float
    numbered_unit_sites: int
    activity_year: int
    activity: dict
    ownership: dict
    public_naics_prefixes: tuple
    public_phrase_fields: tuple
    public_phrases: tuple
    source_values: tuple  # ((source_count, value), ...) ascending
    location: dict
    rules: tuple

    def unverified_evidence(self) -> list[dict]:
        """Evidence URLs in exclusion rules that no one has checked on the web yet."""
        out = []
        for rule in self.document["exclusions"]:
            for item in rule["evidence"]:
                if item.get("verified_at") is None:
                    out.append({"rule": rule["id"], "url": item["url"]})
            for entry in (rule["match"].get("phrases") or {}).get("entries") or []:
                if entry.get("evidence_url") and entry.get("verified_at") is None:
                    out.append({"label": entry["label"], "rule": rule["id"],
                                "url": entry["evidence_url"]})
        return out


def _number(value, where: str, low: float = -1.0, high: float = 1.0) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        and low <= value <= high,
        f"{where}: expected a number in [{low}, {high}]",
    )
    return float(value)


def _year(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 2000


def _text(value, where: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{where}: expected a non-empty string")
    return value


def _verified_at(value, where: str) -> None:
    _require(value is None or (isinstance(value, str) and _DATE.match(value)),
             f"{where}: verified_at must be null or YYYY-MM-DD")


def _https(value, where: str) -> str:
    _require(isinstance(value, str) and value.startswith("https://"), f"{where}: URL must be https")
    return value


def _curve(points, where: str) -> tuple:
    _require(isinstance(points, list) and len(points) >= 2, f"{where}: needs at least two points")
    out = []
    for index, point in enumerate(points):
        spot = f"{where}[{index}]"
        _require(isinstance(point, list) and len(point) == 2, f"{spot}: expected [x, value]")
        x = _number(point[0], f"{spot}[0]", 0.0, 1e12)
        _require(x > 0, f"{spot}[0]: x must be above 0 for log10 interpolation")
        out.append((x, _number(point[1], f"{spot}[1]")))
    _require(all(a[0] < b[0] for a, b in pairwise(out)),
             f"{where}: x values must increase strictly")
    return tuple(out)


def _phrases(value, where: str) -> tuple:
    _require(isinstance(value, list) and value, f"{where}: expected a non-empty list of phrases")
    for phrase in value:
        _require(isinstance(phrase, str) and _PHRASE.match(phrase),
                 f"{where}: {phrase!r} must be uppercase letters and digits with single spaces")
    _require(len(set(value)) == len(value), f"{where}: duplicate phrases")
    return tuple(value)


def _acronyms(value, where: str) -> tuple:
    _require(isinstance(value, list) and value, f"{where}: expected a non-empty list of acronyms")
    for acronym in value:
        _require(isinstance(acronym, str) and _ACRONYM.match(acronym),
                 f"{where}: {acronym!r} must be one uppercase word of 2-6 letters and digits")
    _require(len(set(value)) == len(value), f"{where}: duplicate acronyms")
    return tuple(value)


def _prefixes(value, where: str) -> tuple:
    _require(isinstance(value, list) and value, f"{where}: expected a non-empty list of prefixes")
    for prefix in value:
        _require(isinstance(prefix, str) and _PREFIX.match(prefix),
                 f"{where}: {prefix!r} is not 2-6 digits")
    _require(len(set(value)) == len(value), f"{where}: duplicate prefixes")
    return tuple(value)


def _keys(value, where: str, required, optional=()) -> dict:
    """Return ``value`` when it is an object with every required key and no unknown key.

    A misspelt key (for example ``unless_phrase`` for ``unless_phrases``) would
    otherwise switch a rule or a weight off without any error, so every section
    and rule must name its keys exactly. Optional note keys must be text.
    """
    _require(isinstance(value, dict), f"{where}: must be an object")
    missing = sorted(set(required) - set(value))
    unknown = sorted(set(value) - set(required) - set(optional))
    problems = ([f"missing keys {missing}"] if missing else []) + (
        [f"unknown keys {unknown}"] if unknown else [])
    _require(not problems, f"{where}: {'; '.join(problems)}; allowed {sorted({*required, *optional})}")
    for key in sorted(set(value) & set(optional)):
        if key in ("description", "note") or key.endswith("_note"):
            _text(value[key], f"{where}.{key}")
    return value


def _evidence(items, where: str) -> tuple:
    _require(isinstance(items, list) and items, f"{where}: a rule needs at least one evidence item")
    for index, item in enumerate(items):
        spot = f"{where}[{index}]"
        _require(isinstance(item, dict), f"{spot}: evidence must be an object")
        _keys(item, spot, EVIDENCE_KEYS)
        _https(item["url"], spot)
        _text(item["note"], f"{spot}.note")
        _verified_at(item["verified_at"], spot)
    return tuple(dict(item) for item in items)


def _phrase_matcher(spec, where: str) -> dict:
    _keys(spec, where, ("fields", "entries"), ("note", "unless_phrases"))
    fields = spec["fields"]
    _require(isinstance(fields, list) and fields and all(item in PHRASE_FIELDS for item in fields),
             f"{where}.fields: choose from {list(PHRASE_FIELDS)}")
    entries = spec["entries"]
    _require(isinstance(entries, list) and entries, f"{where}.entries: expected a non-empty list")
    parsed, acronyms = [], []
    for index, entry in enumerate(entries):
        spot = f"{where}.entries[{index}]"
        _keys(entry, spot, ("label", "phrases"), ("acronyms", "evidence_url", "verified_at"))
        if "evidence_url" in entry:
            _https(entry["evidence_url"], spot)
            _require("verified_at" in entry,
                     f"{spot}: an evidence_url needs verified_at (null until someone checks it)")
        else:
            _require("verified_at" not in entry, f"{spot}: verified_at needs an evidence_url")
        _verified_at(entry.get("verified_at"), spot)
        label = _text(entry["label"], f"{spot}.label")
        parsed.append((label, _phrases(entry["phrases"], f"{spot}.phrases")))
        if "acronyms" in entry:
            acronyms.append((label, _acronyms(entry["acronyms"], f"{spot}.acronyms")))
    unless = spec.get("unless_phrases")
    return {
        "acronym_entries": tuple(acronyms),
        "phrase_fields": tuple(fields),
        "phrase_entries": tuple(parsed),
        "unless_phrases": _phrases(unless, f"{where}.unless_phrases") if unless is not None else (),
    }


def _rule(raw, where: str) -> Rule:
    _require(isinstance(raw, dict), f"{where}: a rule must be an object")
    _keys(raw, where, RULE_KEYS)
    rule_id = raw["id"]
    _require(isinstance(rule_id, str) and _RULE_ID.match(rule_id), f"{where}: invalid rule id")
    match = raw["match"]
    _require(isinstance(match, dict) and match, f"{where}: a rule needs a match object")
    unknown = set(match) - set(MATCHERS)
    _require(not unknown, f"{where}: unknown matchers {sorted(unknown)}")
    fields = {
        "id": rule_id,
        "label": _text(raw["label"], f"{where}.label"),
        "reason": _text(raw["reason"], f"{where}.reason"),
        "evidence": _evidence(raw["evidence"], f"{where}.evidence"),
    }
    if "phrases" in match:
        fields.update(_phrase_matcher(match["phrases"], f"{where}.match.phrases"))
    if "naics" in match:
        spot = f"{where}.match.naics"
        spec = _keys(match["naics"], spot, ("prefixes", "scope"), ("note",))
        _require(spec["scope"] in ("any", "primary"), f"{spot}.scope: must be 'any' or 'primary'")
        fields.update(naics_scope=spec["scope"],
                      naics_prefixes=_prefixes(spec["prefixes"], f"{spot}.prefixes"))
    if "frs_activity_status" in match:
        spot = f"{where}.match.frs_activity_status"
        spec = _keys(match["frs_activity_status"], spot, ("values",),
                     ("unless_fsis_listed", "unless_osha_filing_year_at_least"))
        values = spec["values"]
        _require(isinstance(values, list) and values
                 and all(item in ("active", "inactive", "unknown") for item in values),
                 f"{spot}.values: choose from active, inactive, unknown")
        year = spec.get("unless_osha_filing_year_at_least")
        _require(year is None or _year(year),
                 f"{spot}.unless_osha_filing_year_at_least: expected a year or null")
        fsis = spec.get("unless_fsis_listed", False)
        _require(isinstance(fsis, bool), f"{spot}.unless_fsis_listed: expected true or false")
        fields.update(frs_values=tuple(values), frs_unless_osha_year=year, frs_unless_fsis=fsis)
    if "missing_street_and_coordinates" in match:
        _require(match["missing_street_and_coordinates"] is True,
                 f"{where}.match.missing_street_and_coordinates: must be true")
        fields.update(missing_location=True)
    return Rule(**fields)


def _weights_map(value, where: str) -> dict:
    _require(isinstance(value, dict), f"{where}: expected an object")
    return {key: _number(item, f"{where}.{key}", 0.0, 1.0) for key, item in sorted(value.items())}


def _section(document, key: str) -> dict:
    required, optional = SECTION_KEYS[key]
    return _keys(document[key], key, required, optional)


def _task_map(task, capability_weights) -> dict:
    by_capability = task["naics_by_capability"]
    _require(isinstance(by_capability, dict), "task_evidence.naics_by_capability must be an object")
    task_map = {}
    for capability, entries in sorted(by_capability.items()):
        spot = f"task_evidence.naics_by_capability.{capability}"
        _require(capability in capability_weights, f"{spot}: unknown capability")
        _require(isinstance(entries, list) and entries, f"{spot}: expected a non-empty list")
        parsed = []
        for index, entry in enumerate(entries):
            _keys(entry, f"{spot}[{index}]", ("label", "prefix", "value"))
            prefix = _prefixes([entry["prefix"]], f"{spot}[{index}].prefix")[0]
            parsed.append((prefix, _number(entry["value"], f"{spot}[{index}].value", 0.0, 1.0),
                           _text(entry["label"], f"{spot}[{index}].label")))
        _require(len({item[0] for item in parsed}) == len(parsed), f"{spot}: duplicate prefixes")
        task_map[capability] = tuple(sorted(parsed, key=lambda item: (-len(item[0]), item[0])))
    return task_map


def _source_values(spec) -> tuple:
    where = "source_corroboration.values_by_source_count"
    _require(isinstance(spec, dict) and spec, f"{where} must be a non-empty object")
    values = []
    for key, value in spec.items():
        _require(isinstance(key, str) and _SOURCE_COUNT.match(key),
                 f"{where}: {key!r} is not a source count (1, 2, 3, ...)")
        values.append((int(key), _number(value, f"{where}.{key}")))
    _require("1" in spec, f"{where} must define '1': every site has at least one source")
    return tuple(sorted(values))


def check_taxonomy(config: RankConfig, taxonomy) -> None:
    """Fail closed when a site type, capability or category id in the config is not in the taxonomy.

    Without this check a misspelt id (for example ``plant_genral``) would leave
    the real id without its weight, and the site would be scored with the
    category weight instead.
    """
    where = f"not in taxonomy {taxonomy.version}"
    site_types = {key for key, value in taxonomy.site_types.items() if not value.exclusion}
    unknown = sorted(set(config.site_type_weights) - site_types)
    _require(not unknown, f"site_type_weights: unknown site types {unknown} ({where})")
    unknown = sorted(config.unspecific_site_types - site_types)
    _require(not unknown,
             f"capability_scope.unspecific_site_types: unknown site types {unknown} ({where})")
    unknown = sorted(set(config.capability_weights) - {row["capability"] for row in taxonomy.rows})
    _require(not unknown, f"capability_weights: unknown capabilities {unknown} ({where})")
    unknown = sorted(set(config.category_weights) - set(taxonomy.categories))
    _require(not unknown, f"category_weights: unknown categories {unknown} ({where})")


def config_from_document(document: dict, *, sha256: str | None = None, taxonomy=None) -> RankConfig:
    """Validate a rank config document. Raises RankError on any defect (fail closed).

    Every section and rule must name its keys exactly, and every site type,
    capability and category id must exist in ``taxonomy`` (default:
    ``taxonomy.json``).
    """
    _keys(document, "rank config", CONFIG_KEYS, CONFIG_OPTIONAL_KEYS)
    _require(document["schema"] == RANK_CONFIG_SCHEMA, f"schema must be {RANK_CONFIG_SCHEMA}")
    version = _text(document["version"], "version")
    _require(isinstance(document["updated_at"], str) and _DATE.match(document["updated_at"]),
             "updated_at must be YYYY-MM-DD")
    _text(document["score_meaning"], "score_meaning")
    weights = document["weights"]
    _require(isinstance(weights, dict) and set(weights) == set(COMPONENTS),
             f"weights must name exactly the components {list(COMPONENTS)}")
    weights = {name: _number(weights[name], f"weights.{name}", 0.0, 100.0) for name in COMPONENTS}
    _require(abs(sum(weights.values()) - 100.0) < 1e-6, "weights must sum to 100 points")

    capabilities = document["capability_weights"]
    _require(isinstance(capabilities, dict) and capabilities,
             "capability_weights must be a non-empty object")
    capability_weights = {}
    for capability, entry in sorted(capabilities.items()):
        spot = f"capability_weights.{capability}"
        _keys(entry, spot, ("weight",), ("note",))
        capability_weights[capability] = _number(entry["weight"], f"{spot}.weight", 0.0, 1.0)
    site_type_weights = _weights_map(document.get("site_type_weights", {}), "site_type_weights")
    site_type_notes = document.get("site_type_weight_notes", {})
    _require(isinstance(site_type_notes, dict), "site_type_weight_notes: must be an object")
    for site_type, note in sorted(site_type_notes.items()):
        _text(note, f"site_type_weight_notes.{site_type}")

    size = _section(document, "size")
    _require(size["interpolation"] == "log10", "size.interpolation must be log10")
    operator = _section(document, "operator_scale")
    _require(operator["interpolation"] == "log10", "operator_scale.interpolation must be log10")
    scope = _section(document, "capability_scope")
    unspecific = scope["unspecific_site_types"]
    _require(isinstance(unspecific, list)
             and all(isinstance(item, str) and item for item in unspecific),
             "capability_scope.unspecific_site_types must be a list of site type ids")
    task = _section(document, "task_evidence")

    activity = _section(document, "activity_evidence")
    _require(_year(activity["osha_filing_year_at_least"]),
             "activity_evidence.osha_filing_year_at_least must be a year")
    activity_values = activity["values"]
    _require(isinstance(activity_values, dict) and set(activity_values) == ACTIVITY_KEYS,
             f"activity_evidence.values must name exactly {sorted(ACTIVITY_KEYS)}")

    ownership = _section(document, "ownership")
    ownership_values = ownership["values"]
    _require(isinstance(ownership_values, dict)
             and set(ownership_values) == {"private", "public", "unknown"},
             "ownership.values must name exactly private, public and unknown")
    public_fields = ownership["phrase_fields"]
    _require(isinstance(public_fields, list) and public_fields
             and all(item in PHRASE_FIELDS for item in public_fields),
             f"ownership.phrase_fields: choose from {list(PHRASE_FIELDS)}")

    source_values = _source_values(_section(document, "source_corroboration")["values_by_source_count"])
    location = _section(document, "location_quality")
    location = {key: _number(location[key], f"location_quality.{key}", 0.0, 1.0)
                for key in sorted(LOCATION_KEYS)}
    _require(location["street_address"] + location["precise_coordinates"] <= 1.0 + 1e-9
             and location["other_coordinates"] <= location["precise_coordinates"],
             "location_quality: street plus precise coordinates must be at most 1")

    raw_rules = document["exclusions"]
    _require(isinstance(raw_rules, list) and raw_rules, "exclusions must be a non-empty list")
    rules = tuple(_rule(raw, f"exclusions[{index}]") for index, raw in enumerate(raw_rules))
    _require(len({rule.id for rule in rules}) == len(rules), "exclusions: duplicate rule ids")

    canonical = build_module.canonical_json(document).encode("utf-8")
    config = RankConfig(
        document=document,
        sha256=sha256 or hashlib.sha256(canonical).hexdigest(),
        version=version,
        weights=weights,
        capability_weights=capability_weights,
        category_weights=_weights_map(document["category_weights"], "category_weights"),
        site_type_weights=site_type_weights,
        employees_curve=_curve(size["employees_curve"], "size.employees_curve"),
        building_area_curve=_curve(size["building_area_m2_curve"], "size.building_area_m2_curve"),
        building_area_confidence=_number(size["building_area_confidence"],
                                         "size.building_area_confidence", 0.0, 1.0),
        size_unknown=_number(size["unknown"], "size.unknown"),
        unspecific_site_types=frozenset(unspecific),
        secondary_factor=_number(scope["secondary_site_type_factor"],
                                 "capability_scope.secondary_site_type_factor", 0.0, 1.0),
        task_default=_number(task["default"], "task_evidence.default", 0.0, 1.0),
        task_map=_task_map(task, capability_weights),
        operator_curve=_curve(operator["site_count_curve"], "operator_scale.site_count_curve"),
        operator_unknown=_number(operator["unknown"], "operator_scale.unknown"),
        numbered_unit_sites=int(_number(operator["numbered_unit_assumed_sites"],
                                        "operator_scale.numbered_unit_assumed_sites", 1, 100000)),
        activity_year=activity["osha_filing_year_at_least"],
        activity={key: _number(value, f"activity_evidence.values.{key}")
                  for key, value in sorted(activity_values.items())},
        ownership={key: _number(value, f"ownership.values.{key}")
                   for key, value in sorted(ownership_values.items())},
        public_naics_prefixes=_prefixes(ownership["public_naics_prefixes"],
                                        "ownership.public_naics_prefixes"),
        public_phrase_fields=tuple(public_fields),
        public_phrases=_phrases(ownership["public_phrases"], "ownership.public_phrases"),
        source_values=source_values,
        location=location,
        rules=rules,
    )
    check_taxonomy(config, taxonomy if taxonomy is not None else taxonomy_module.load())
    stray = sorted(set(site_type_notes) - set(site_type_weights))
    _require(not stray, f"site_type_weight_notes: {stray} have no entry in site_type_weights")
    return config


def load_config(path: Path | str = RANK_CONFIG_PATH, *, taxonomy=None) -> RankConfig:
    """Load and validate a rank config; its id is the SHA-256 of the file bytes."""
    data = Path(path).read_bytes()
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankError(f"{path}: rank config is not JSON: {error}") from error
    return config_from_document(document, sha256=hashlib.sha256(data).hexdigest(),
                                taxonomy=taxonomy)


def as_config(config, taxonomy=None) -> RankConfig:
    """A RankConfig from a RankConfig, a document, a path, or None (the committed file).

    With ``taxonomy``, a RankConfig built earlier is checked against it again.
    """
    if config is None:
        return load_config(taxonomy=taxonomy)
    if isinstance(config, RankConfig):
        if taxonomy is not None:
            check_taxonomy(config, taxonomy)
        return config
    if isinstance(config, dict):
        return config_from_document(config, taxonomy=taxonomy)
    return load_config(config, taxonomy=taxonomy)


# --- site fields ----------------------------------------------------------------------------
def _features(site) -> dict:
    return site.get("features") or {}


def _matches(site) -> dict:
    return site.get("taxonomy_matches") or {}


def _records(site):
    return site.get("records") or ()


def site_naics(site, scope: str = "primary") -> list[str]:
    """The site's NAICS codes: site and record codes, plus every FRS code when scope is 'any'."""
    codes = {site["naics"]} if site.get("naics") else set()
    for record in _records(site):
        if record.get("naics"):
            codes.add(record["naics"])
        if scope == "any":
            codes.update((record.get("attributes") or {}).get("naics_codes") or ())
    return sorted(codes)


def _osha_attributes(site) -> list[dict]:
    return [record.get("attributes") or {} for record in _records(site)
            if record.get("source_id") == "osha_ita"]


def _osha_year(site) -> int | None:
    years = [item["year_filing_for"] for item in _osha_attributes(site)
             if item.get("year_filing_for")]
    return max(years) if years else None


def _field_texts(site, field: str) -> list[str]:
    values = site.get("names") or () if field == "names" else (site.get(field),)
    return [text for text in (phrase_text(value) for value in values if value) if text]


def _code_like(token: str) -> bool:
    """Store numbers and site codes such as 1234, ABC2 or XYZ7."""
    return any(ch.isdigit() for ch in token) and (token.isdigit() or len(token) <= 5)


def _group_key(text, city) -> str | None:
    tokens = [token for token in normalize.name_tokens(text, city=city) if not _code_like(token)]
    if not tokens or all(token in normalize.WEAK_NAME_TOKENS for token in tokens):
        return None
    return " ".join(tokens)


def operator_stem(operator: str | None) -> str | None:
    """The company part of an operator string: the words before its first legal or descriptor
    word, so 'Pecan Healthcare PH WEXMOOR COUNTY' gives 'Pecan'."""
    words = (operator or "").split()
    for index, word in enumerate(words):
        if phrase_text(word).strip() in STEM_WORDS:
            return " ".join(words[:index]) or None
    return " ".join(words) or None


def operator_keys(site) -> tuple[str | None, str | None]:
    """Grouping keys from the operator's company stem and from the full name (no unit codes)."""
    city = site.get("city")
    operator = None
    if site.get("operator"):
        stem = operator_stem(site["operator"])
        operator = (_group_key(stem, city) if stem else None) or _group_key(site["operator"], city)
    name = _group_key(site["name"], city) if site.get("name") else None
    return operator, name


def numbered_unit(name: str | None, street: str | None = None) -> str | None:
    """The unit number in a chain-style name ('0418 HARDWARE MART OF LARKSTONE', 'GROCER #962'), or None.

    A leading number is an address used as a name, not a unit number, when a
    thoroughfare word follows it anywhere in the name ('2001 Example Road',
    '4400 Commerce St Plant') or when it is the house number of the site's own
    ``street``. 'ST' right after the number reads as Saint ('0952 ST QUILL
    HOSPITAL'): an address puts the street name before its suffix.
    """
    if not name:
        return None
    match = _LEADING_UNIT.match(name)
    if match:
        number, words = match.group(1), phrase_text(match.group(2)).split()
        thoroughfare = any(word in _STREET_WORDS and not (position == 0 and word == "ST")
                           for position, word in enumerate(words))
        house = normalize.house_number(street) or ""
        if words and not thoroughfare and number.lstrip("0") != house.lstrip("0"):
            return number
    match = _HASH_UNIT.search(name) or _STORE_UNIT.search(name)
    return match.group(0) if match else None


def build_context(sites, taxonomy=None) -> dict:
    """Snapshot-wide context: sites per operator key and per name key, plus the taxonomy maps."""
    by_operator, by_name = Counter(), Counter()
    for site in sites:
        operator, name = operator_keys(site)
        if operator:
            by_operator[operator] += 1
        if name:
            by_name[name] += 1
    context = {"name_key_counts": by_name, "operator_key_counts": by_operator}
    if taxonomy is not None:
        context["capabilities_by_site_type"] = dict(taxonomy.capabilities_by_site_type)
        context["site_type_category"] = {key: value.category
                                         for key, value in taxonomy.site_types.items()}
        context["taxonomy"] = {"sha256": taxonomy.sha256, "version": taxonomy.version}
    return context


# --- components -----------------------------------------------------------------------------
def capability_support(site, config, context=None) -> dict:
    """Capability -> the site types that support it, in capability order.

    With the taxonomy maps in ``context``, an unspecific site type (config
    ``capability_scope``) supports capabilities only when the site has no
    specific site type. Without them, the primary site type supports every
    capability the snapshot lists for the site.
    """
    matches = _matches(site)
    capabilities = set(matches.get("capabilities") or ())
    mapping = (context or {}).get("capabilities_by_site_type")
    if mapping is None:
        return {cap: (matches.get("primary_site_type"),) for cap in sorted(capabilities)}
    types = list(matches.get("site_types") or ())
    specific = [site_type for site_type in types if site_type not in config.unspecific_site_types]
    support: dict = {}
    for site_type in specific or types:
        for cap in mapping.get(site_type, ()):
            if cap in capabilities:
                support.setdefault(cap, []).append(site_type)
    return {cap: tuple(support[cap]) for cap in sorted(support)}


def _scope(site, config, capabilities, context) -> dict:
    support = capability_support(site, config, context)
    return {cap: types for cap, types in support.items()
            if capabilities is None or cap in capabilities}


def _capability_fit(site, config, scope, context):
    if not scope:
        return 0.0, "no capability in scope"
    missing = [cap for cap in scope if cap not in config.capability_weights]
    _require(not missing, f"rank config has no capability weight for {missing}")
    primary = _matches(site).get("primary_site_type")
    if not any(primary in types for types in capability_support(site, config, context).values()):
        primary = None  # the primary type is unspecific; every specific type counts as primary
    best = None
    for cap, types in scope.items():
        secondary = primary is not None and primary not in types
        value = config.capability_weights[cap] * (config.secondary_factor if secondary else 1.0)
        if best is None or value > best[0]:
            best = (value, f"{cap} via secondary site type {types[0]}" if secondary else cap)
    return best


def lead_capability(row, config=None) -> str:
    """The capability that sets a ranked row's capability fit (used by the backlog export).

    It is the capability named by ``components.capability_fit.basis``: ``cap``, or ``cap via
    secondary site type X``. It must be one of the row's capabilities and have a weight in
    ``config``. Raises RankError otherwise. Reads the row only; ranking output is unchanged.
    """
    config = as_config(config)
    _require(isinstance(row, dict), "a ranked row must be an object")
    fit = (row.get("components") or {}).get("capability_fit") or {}
    basis = fit.get("basis") if isinstance(fit, dict) else None
    capability = basis.split(" via secondary site type ", 1)[0] if isinstance(basis, str) else None
    _require(capability in (row.get("capabilities") or ()) and capability in config.capability_weights,
             f"{row.get('site_id')}: capability fit basis {basis!r} names no capability of the row")
    return capability


def _task_evidence(site, config, scope, context):
    codes = site_naics(site, "primary")
    best = None
    for capability in scope:
        found = None
        for prefix, value, label in config.task_map.get(capability, ()):
            code = next((code for code in codes if code.startswith(prefix)), None)
            if code:
                found = (value, f"{capability}: NAICS {code} ({label})")
                break
        candidate = found or (config.task_default, "no NAICS task evidence")
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best or (config.task_default, "no capability in scope")


def _size_fit(site, config, scope, context):
    employees = site.get("employees")
    if employees is not None:
        return interpolate(config.employees_curve, employees), f"{employees} employees (OSHA ITA)"
    area = site.get("building_area_m2")
    if area:
        value = config.building_area_confidence * interpolate(config.building_area_curve, area)
        return value, f"{round(area)} m2 building footprint (OpenStreetMap); employees unknown"
    return config.size_unknown, "size unknown"


def _type_weight(site_type, category, config):
    if site_type in config.site_type_weights:
        return config.site_type_weights[site_type], f"site type {site_type}"
    _require(category in config.category_weights,
             f"rank config has no weight for category {category!r}")
    return config.category_weights[category], f"category {category}"


def _category_fit(site, config, scope, context):
    categories = (context or {}).get("site_type_category")
    if categories is not None and scope:
        best = None
        for site_type in sorted({site_type for types in scope.values() for site_type in types}):
            candidate = _type_weight(site_type, categories.get(site_type), config)
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best
    return _type_weight(_matches(site).get("primary_site_type"), site.get("category"), config)


def _operator_scale(site, config, scope, context):
    if context is None:
        return config.operator_unknown, "no snapshot context"
    operator, name = operator_keys(site)
    by_operator = context["operator_key_counts"].get(operator, 0) if operator else 0
    by_name = context["name_key_counts"].get(name, 0) if name else 0
    unit = numbered_unit(site.get("name"), site.get("street"))
    if unit and max(by_operator, by_name) < config.numbered_unit_sites:
        count = config.numbered_unit_sites
        return (interpolate(config.operator_curve, count),
                f"unit number '{unit}' in the name: a chain of at least {count} sites")
    if not by_operator and not by_name:
        return config.operator_unknown, "no operator or name key"
    if by_operator >= by_name:
        count, basis = by_operator, f"operator key '{operator}'"
    else:
        count, basis = by_name, f"name key '{name}'"
    noun = "site shares" if count == 1 else "sites share"
    return interpolate(config.operator_curve, count), f"{count} {noun} the {basis}"


def _activity_evidence(site, config, scope, context):
    values, features = config.activity, _features(site)
    options = []
    year = _osha_year(site)
    if year and year >= config.activity_year:
        options.append((values["osha_recent_filing"], f"OSHA ITA filing for {year}"))
    if "fsis_mpi" in (features.get("sources") or ()):
        options.append((values["fsis_listed"], "listed in the USDA FSIS directory"))
    status = features.get("frs_activity_status")
    if status == "active":
        options.append((values["frs_active"], "active EPA FRS program record"))
    elif status == "unknown":
        options.append((values["frs_unknown"], "EPA FRS record without an active status"))
    elif status == "inactive":
        options.append((values["frs_inactive"], "every EPA FRS program record closed"))
    if not options:
        return values["none"], "no activity record (map or directory listing only)"
    return max(options, key=lambda option: option[0])


def _ownership(site, config, scope, context):
    types = sorted({item.get("establishment_type") for item in _osha_attributes(site)} - {None})
    public = [kind for kind in types if kind.endswith("_government")]
    if public:
        return config.ownership["public"], f"OSHA ITA establishment type {public[0]}"
    codes = site_naics(site, "primary")
    for prefix in config.public_naics_prefixes:
        code = next((code for code in codes if code.startswith(prefix)), None)
        if code:
            return config.ownership["public"], f"public-sector NAICS {code}"
    for field in config.public_phrase_fields:
        for text in _field_texts(site, field):
            for phrase in config.public_phrases:
                if f" {phrase} " in text:
                    return config.ownership["public"], f"{field} contains '{phrase}'"
    if "private" in types:
        return config.ownership["private"], "OSHA ITA establishment type private"
    return config.ownership["unknown"], "ownership unknown"


def _source_corroboration(site, config, scope, context):
    features = _features(site)
    count = features.get("source_count") or len(features.get("sources") or ())
    value = config.source_values[0][1]
    for threshold, threshold_value in config.source_values:
        if count >= threshold:
            value = threshold_value
    return value, f"{count} source{'s' if count != 1 else ''}"


def _location_quality(site, config, scope, context):
    features, points, parts = _features(site), 0.0, []
    if features.get("has_street_address"):
        points += config.location["street_address"]
        parts.append("street address")
    if features.get("has_coordinates"):
        precision = features.get("coordinate_precision")
        if precision == "precise":
            points += config.location["precise_coordinates"]
            parts.append("precise coordinates")
        else:
            points += config.location["other_coordinates"]
            parts.append(f"{precision or 'unknown'}-precision coordinates")
    return points, " and ".join(parts) or "no street address or coordinates"


COMPONENT_FUNCTIONS = {
    "capability_fit": _capability_fit,
    "task_evidence": _task_evidence,
    "size_fit": _size_fit,
    "category_fit": _category_fit,
    "operator_scale": _operator_scale,
    "activity_evidence": _activity_evidence,
    "ownership": _ownership,
    "source_corroboration": _source_corroboration,
    "location_quality": _location_quality,
}


# --- exclusions -----------------------------------------------------------------------------
def _acronym_position(text: str, field: str) -> str | None:
    """The word of a field text that can be a company acronym, or None.

    That is the first word of a name (after 'THE'), or the whole operator once
    its legal form is removed: 'UPS' and 'RTX Corporation' qualify, 'UPS
    Holdings LLC' and 'Roll Ups Packaging' do not.
    """
    words = text.split()
    if words and words[0] == "THE":
        words = words[1:]
    if field == "operator":
        words = [word for word in words if word not in LEGAL_FORMS]
        return words[0] if len(words) == 1 else None
    return words[0] if words else None


def _phrase_match(rule: Rule, site) -> str | None:
    """The longest phrase of the rule in the first field text that has one (whole words only).

    An acronym of the rule (for example UPS) matches only in a company-name
    position (see :func:`_acronym_position`).
    """
    for field in rule.phrase_fields:
        for text in _field_texts(site, field):
            if any(f" {phrase} " in text for phrase in rule.unless_phrases):
                continue
            found = [(len(phrase), phrase, label) for label, phrases in rule.phrase_entries
                     for phrase in phrases if f" {phrase} " in text]
            if found:
                _, phrase, label = max(found)
                return f"{field} contains '{phrase}' ({label})"
            word = _acronym_position(text, field) if rule.acronym_entries else None
            for label, acronyms in rule.acronym_entries:
                if word in acronyms:
                    verb = "is" if field == "operator" else "starts with"
                    return f"{field} {verb} '{word}' ({label})"
    return None


def rule_match(rule: Rule, site) -> str | None:
    """The first reason the rule matches the site, or None."""
    if rule.phrase_entries:
        detail = _phrase_match(rule, site)
        if detail:
            return detail
    for code in site_naics(site, rule.naics_scope) if rule.naics_prefixes else ():
        if any(code.startswith(prefix) for prefix in rule.naics_prefixes):
            return f"{rule.naics_scope} NAICS {code}"
    features = _features(site)
    if rule.frs_values and features.get("frs_activity_status") in rule.frs_values:
        year = _osha_year(site)
        osha = bool(rule.frs_unless_osha_year and year and year >= rule.frs_unless_osha_year)
        fsis = rule.frs_unless_fsis and "fsis_mpi" in (features.get("sources") or ())
        if not (osha or fsis):
            return f"EPA FRS activity status {features['frs_activity_status']}"
    located = features.get("has_street_address") or features.get("has_coordinates")
    if rule.missing_location and not located:
        return "no street address and no coordinates"
    return None


@dataclass(frozen=True)
class ExclusionsInput:
    """Rows from an exclusions input JSONL file, indexed for lookup."""

    sha256: str
    rows: tuple
    by_site_id: dict
    by_name: dict
    by_operator: dict

    def matches(self, site) -> list[dict]:
        hits = list(self.by_site_id.get(site.get("site_id"), ()))
        values = [site.get("name"), *(site.get("names") or ())]
        names = {phrase_text(name).strip() for name in values if name} - {""}
        for name in sorted(names):
            hits.extend(row for row in self.by_name.get(name, ()) if _filters_match(row, site))
        operator = phrase_text(site.get("operator")).strip()
        if operator:
            hits.extend(self.by_operator.get(operator, ()))
        unique = {row["line"]: row for row in hits}
        return [unique[line] for line in sorted(unique)]


def _filters_match(row: dict, site) -> bool:
    return all(site.get(field) == value for field, value in row["filters"].items())


_INPUT_FILTERS = (
    ("city", normalize.normalize_city),
    ("state", normalize.normalize_state),
    ("postal_code", normalize.normalize_postal),
)


def _input_row(raw, where: str, number: int) -> dict:
    _require(isinstance(raw, dict), f"{where}: not a JSON object")
    unknown = set(raw) - INPUT_FIELDS
    _require(not unknown, f"{where}: unknown fields {sorted(unknown)}")
    keys = [key for key in INPUT_KEYS if raw.get(key) not in (None, "")]
    _require(len(keys) == 1, f"{where}: give exactly one of site_id, name or operator")
    key = keys[0]
    reason = _text(raw.get("reason"), f"{where}: reason")
    source = raw.get("source", "manual")
    _require(isinstance(source, str) and _INPUT_SOURCE.match(source),
             f"{where}: source must be lowercase letters, digits and underscores")
    added_at = raw.get("added_at")
    _require(added_at is None or (isinstance(added_at, str) and _DATE.match(added_at)),
             f"{where}: added_at must be YYYY-MM-DD")
    if key == "site_id":
        value = str(raw["site_id"]).strip().lower()
        _require(_SITE_ID.match(value), f"{where}: site_id must be 64 hex characters")
    else:
        value = phrase_text(str(raw[key])).strip()
        _require(value, f"{where}: {key} has no letters or digits")
    filters = {}
    for field, normalizer in _INPUT_FILTERS:
        if raw.get(field) not in (None, ""):
            _require(key == "name", f"{where}: {field} narrows a name row only")
            normalized = normalizer(str(raw[field]))
            _require(normalized, f"{where}: {field} {raw[field]!r} does not normalize")
            filters[field] = normalized
    return {"filters": filters, "key": key, "line": number, "reason": reason,
            "rule": f"input:{source}", "value": value}


def load_exclusions_input(path) -> ExclusionsInput:
    """Parse an exclusions input JSONL file (fail closed on any malformed row).

    Each line is an object with exactly one of ``site_id``, ``name`` or
    ``operator``; a non-empty ``reason``; and an optional ``source`` (for
    example ``crm`` or ``rejection``; default ``manual``) that names the rule
    id ``input:<source>``. A ``name`` row may narrow the match with ``city``,
    ``state`` or ``postal_code``. ``added_at`` (YYYY-MM-DD) and ``note`` are
    kept for the record. Names and operators match after the same whole-word
    normalization as the exclusion phrases. Blank lines are skipped.
    """
    path = Path(path)
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RankError(f"{path}: exclusions input is not UTF-8") from error
    rows = []
    index = {"name": {}, "operator": {}, "site_id": {}}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise RankError(f"{where}: not a JSON object: {error.msg}") from error
        row = _input_row(raw, where, number)
        rows.append(row)
        index[row["key"]].setdefault(row["value"], []).append(row)
    return ExclusionsInput(hashlib.sha256(data).hexdigest(), tuple(rows), index["site_id"],
                           index["name"], index["operator"])


# --- score ----------------------------------------------------------------------------------
def _top(components: dict, count: int = 3) -> list[tuple[str, dict]]:
    return sorted(components.items(), key=lambda item: (-abs(item[1]["points"]), item[0]))[:count]


def score(site, config=None, *, capabilities=None, context=None, exclusions_input=None) -> dict:
    """Score one site (pure): ``{score, components, excluded, exclusion_details, explanation}``.

    ``capabilities`` limits capability fit, task evidence and category fit to
    those capabilities (default: all of the site's capabilities). ``context``
    comes from :func:`build_context` over the whole snapshot; without it the
    operator scale is unknown and the snapshot's capability list is used as
    is. ``exclusions_input`` is an :class:`ExclusionsInput`. Components
    measure fit evidence only.
    """
    config = as_config(config)
    scope = _scope(site, config, capabilities, context)
    components = {}
    for name in COMPONENTS:
        value, basis = COMPONENT_FUNCTIONS[name](site, config, scope, context)
        value = round(value, 4)
        weight = config.weights[name]
        components[name] = {"basis": basis, "points": round(weight * value, 3), "value": value,
                            "weight": weight}
    total = round(sum(item["points"] for item in components.values()), 3)
    details = []
    for rule in config.rules:
        detail = rule_match(rule, site)
        if detail:
            details.append({"detail": detail, "rule": rule.id})
    for row in exclusions_input.matches(site) if exclusions_input is not None else ():
        detail = (f"exclusions input line {row['line']} ({row['key']} '{row['value']}'): "
                  f"{row['reason']}")
        details.append({"detail": detail, "rule": row["rule"]})
    excluded = list(dict.fromkeys(item["rule"] for item in details))
    summary = "; ".join(f"{name} {item['points']:+.1f} ({item['basis']})"
                        for name, item in _top(components))
    explanation = f"{total:.1f} points of fit evidence. Largest components: {summary}."
    if details:
        reasons = "; ".join(f"{item['rule']} ({item['detail']})" for item in details)
        explanation = f"Excluded: {reasons}. {explanation}"
    return {"components": components, "excluded": excluded, "exclusion_details": details,
            "explanation": explanation, "score": total}


# --- snapshot and ranking -------------------------------------------------------------------
@dataclass(frozen=True)
class LoadedSnapshot:
    manifest: dict
    sites: list


_RECORD_ATTRIBUTES = ("activity_status", "establishment_type", "naics_codes", "year_filing_for")
_MATCH_FIELDS = ("capabilities", "primary_site_type", "rows", "site_types")
_SITE_FIELDS = (
    "attribution_required", "building_area_m2", "category", "city", "employees", "features", "lat",
    "lon", "naics", "name", "names", "operator", "postal_code", "site_id", "state", "street",
)


def _slim(site: dict) -> dict:
    """The fields that scoring and output rows read; records keep only their scoring fields."""
    slim = {key: site.get(key) for key in _SITE_FIELDS}
    slim["taxonomy_matches"] = {key: _matches(site).get(key) for key in _MATCH_FIELDS}
    slim["records"] = [
        {
            "attributes": {key: (record.get("attributes") or {}).get(key)
                           for key in _RECORD_ATTRIBUTES},
            "naics": record.get("naics"),
            "source_id": record.get("source_id"),
        }
        for record in _records(site)
    ]
    return slim


def load_snapshot(snapshot_dir) -> LoadedSnapshot:
    """Read a snapshot after checking that sites.jsonl.gz matches the manifest's snapshot_id."""
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / build_module.MANIFEST_FILE
    sites_path = snapshot_dir / build_module.SITES_FILE
    _require(manifest_path.is_file() and sites_path.is_file(),
             f"{snapshot_dir}: needs {build_module.MANIFEST_FILE} and {build_module.SITES_FILE}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankError(f"{manifest_path}: not JSON: {error}") from error
    _require(isinstance(manifest, dict) and manifest.get("schema") == build_module.MANIFEST_SCHEMA,
             f"{manifest_path}: schema must be {build_module.MANIFEST_SCHEMA}")
    _require(manifest.get("site_schema") == SCHEMA_VERSION,
             f"{manifest_path}: site_schema must be {SCHEMA_VERSION}")
    data = sites_path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    _require(actual == manifest.get("snapshot_id"),
             f"{sites_path}: SHA-256 {actual} does not match snapshot_id "
             f"{manifest.get('snapshot_id')}")
    lines = gzip.decompress(data).decode("utf-8").splitlines()
    return LoadedSnapshot(manifest, [_slim(json.loads(line)) for line in lines if line])


def _requested(capabilities, config) -> frozenset | None:
    if capabilities is None:
        return None
    requested = [capabilities] if isinstance(capabilities, str) else list(capabilities)
    _require(requested and all(isinstance(cap, str) for cap in requested),
             "capabilities must be strings")
    unknown = sorted(set(requested) - set(config.capability_weights))
    _require(not unknown,
             f"unknown capabilities {unknown}; known: {sorted(config.capability_weights)}")
    return frozenset(requested)


def _check_limit(limit) -> None:
    valid = isinstance(limit, int) and not isinstance(limit, bool) and limit >= 1
    _require(limit is None or valid, "limit must be a positive integer or None")


def _check_coverage(sites, config, context) -> None:
    """Fail closed when the config or the taxonomy cannot score every site of the snapshot."""
    capabilities = {cap for site in sites for cap in _matches(site).get("capabilities") or ()}
    missing = sorted(capabilities - set(config.capability_weights))
    _require(not missing, f"rank config has no capability weight for {missing}")
    types = {site_type for site in sites for site_type in _matches(site).get("site_types") or ()}
    unknown = sorted(types - set(context["capabilities_by_site_type"]))
    _require(not unknown,
             f"snapshot site types {unknown} are not active in taxonomy "
             f"{context['taxonomy']['version']}; rank with the taxonomy the snapshot was "
             "built with")
    categories = {context["site_type_category"][site_type] for site_type in types}
    categories |= {site.get("category") for site in sites} - {None}
    missing = sorted(categories - set(config.category_weights))
    _require(not missing, f"rank config has no category weight for {missing}")


def _prepare(snapshot_dir, config, capabilities, limit, exclusions_input, taxonomy):
    taxonomy = taxonomy or taxonomy_module.load()
    config = as_config(config, taxonomy)
    _check_limit(limit)
    requested = _requested(capabilities, config)
    snapshot = load_snapshot(snapshot_dir)
    context = build_context(snapshot.sites, taxonomy)
    _check_coverage(snapshot.sites, config, context)
    inputs = load_exclusions_input(exclusions_input) if exclusions_input is not None else None
    return config, requested, snapshot, context, inputs


def _row(site, result, scope) -> dict:
    matches = _matches(site)
    return {
        "attribution_required": bool(site.get("attribution_required")),
        "building_area_m2": site.get("building_area_m2"),
        "capabilities": scope,
        "category": site.get("category"),
        "city": site.get("city"),
        "components": result["components"],
        "employees": site.get("employees"),
        "excluded": result["excluded"],
        "exclusion_details": result["exclusion_details"],
        "explanation": result["explanation"],
        "lat": site.get("lat"),
        "lon": site.get("lon"),
        "naics": site.get("naics"),
        "name": site.get("name"),
        "operator": site.get("operator"),
        "postal_code": site.get("postal_code"),
        "primary_site_type": matches.get("primary_site_type"),
        "rank": None,
        "score": result["score"],
        "site_id": site["site_id"],
        "site_types": list(matches.get("site_types") or ()),
        "sources": list(_features(site).get("sources") or ()),
        "state": site.get("state"),
        "status": "excluded" if result["excluded"] else "ranked",
        "street": site.get("street"),
    }


def _rank_sites(sites, config, requested, context, exclusions_input) -> tuple[list, list]:
    """(ranked rows in rank order, excluded rows by site_id) for the sites in scope."""
    ranked, excluded = [], []
    for site in sites:
        scope = _scope(site, config, requested, context)
        if not scope:
            continue
        result = score(site, config, capabilities=requested, context=context,
                       exclusions_input=exclusions_input)
        (excluded if result["excluded"] else ranked).append(_row(site, result, list(scope)))
    ranked.sort(key=lambda row: (-row["score"], row["site_id"]))
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    excluded.sort(key=lambda row: row["site_id"])
    return ranked, excluded


def rank(snapshot_dir, config=None, *, capabilities=None, limit=None, exclusions_input=None,
         taxonomy=None) -> list[dict]:
    """Ranked rows for a snapshot: ranked sites first, then every excluded site.

    Ranked rows carry ``rank`` 1..N in order of score, highest first, ties
    broken by ``site_id``; ``limit`` keeps the first N of them. Excluded rows
    have ``rank`` None and the matched rule ids in ``excluded``; they are never
    limited, so no exclusion is dropped silently. ``exclusions_input`` is a
    path to a JSONL file (see :func:`load_exclusions_input`). ``taxonomy``
    defaults to ``taxonomy.json``; it maps site types to capabilities.
    """
    config, requested, snapshot, context, inputs = _prepare(
        snapshot_dir, config, capabilities, limit, exclusions_input, taxonomy)
    ranked, excluded = _rank_sites(snapshot.sites, config, requested, context, inputs)
    return ranked[:limit] + excluded if limit else ranked + excluded


# --- outputs --------------------------------------------------------------------------------
@dataclass(frozen=True)
class RankRun:
    out_dir: Path
    ranked_path: Path
    manifest_path: Path
    review_path: Path
    manifest: dict


def _percentiles(scores) -> dict:
    """Nearest-rank percentiles of the ranked scores."""
    ordered = sorted(scores)
    if not ordered:
        return {}
    return {f"p{p}": ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)] for p in PERCENTILES}


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _review_components(row) -> str:
    return ", ".join(f"{name} {item['points']:+.1f}" for name, item in _top(row["components"]))


def _review(snapshot, config, review_capabilities, per_capability, counts, excluded, run_scope):
    manifest = snapshot.manifest
    states = ", ".join(manifest.get("states") or [])
    lines = [
        "# Site ranking review: top 25 per capability",
        "",
        ("Internal owner review. Do not commit this file: it lists sites and holds "
         "OpenStreetMap-derived data (ODbL, share-alike)."),
        "Scores add fit evidence only. A score is not a measure of buying interest.",
        "",
        (f"- Snapshot `{manifest['snapshot_id']}` (states {states}; "
         f"taxonomy {(manifest.get('taxonomy') or {}).get('version')})"),
        f"- Rank config {config.version}, SHA-256 `{config.sha256}`",
        f"- Scope: {', '.join(run_scope) if run_scope else 'all capabilities'}",
        (f"- Sites in scope {counts['sites_in_scope']}; ranked {counts['sites_ranked']}; "
         f"excluded {counts['sites_excluded']}"),
        "",
        "## Counts per capability",
        "",
        "| Capability | Weight | In scope | Excluded | Ranked |",
        "|---|---:|---:|---:|---:|",
    ]
    for cap in review_capabilities:
        row = counts["by_capability"].get(cap, {"excluded": 0, "in_scope": 0, "ranked": 0})
        lines.append(f"| `{cap}` | {config.capability_weights[cap]:.2f} | {row['in_scope']} | "
                     f"{row['excluded']} | {row['ranked']} |")
    lines += ["", "## Exclusions by rule", "", "| Rule | Sites | Reason |", "|---|---:|---|"]
    reasons = {rule.id: rule.reason for rule in config.rules}
    for rule_id, number in counts["exclusions_by_rule"].items():
        reason = _cell(reasons.get(rule_id, "exclusions input"))
        lines.append(f"| `{rule_id}` | {number} | {reason} |")
    for cap in review_capabilities:
        lines += [
            "",
            f"## `{cap}` (capability weight {config.capability_weights[cap]:.2f})",
            "",
            "Capability fit, task evidence and category fit are scored for this capability only.",
            "",
            "| # | Name | City | Primary site type | NAICS | Employees | Score | Top 3 components |",
            "|---:|---|---|---|---|---:|---:|---|",
        ]
        if not per_capability[cap]:
            lines.append("| | (no ranked sites) | | | | | | |")
        for row in per_capability[cap]:
            lines.append(
                f"| {row['rank']} | {_cell(row['name'])} | {_cell(row['city'])} | "
                f"{_cell(row['primary_site_type'])} | {_cell(row['naics'])} | "
                f"{_cell(row['employees'])} | {row['score']:.1f} | {_review_components(row)} |"
            )
    lines += ["", (f"## Highest-scoring excluded sites by rule (top {REVIEW_EXCLUDED_PER_RULE}, "
                   "to check the rules)")]
    by_score = sorted(excluded, key=lambda row: (-row["score"], row["site_id"]))
    for rule_id in counts["exclusions_by_rule"]:
        lines += [
            "",
            f"### `{rule_id}`",
            "",
            "| Name | City | Primary site type | Employees | Score | Matched |",
            "|---|---|---|---:|---:|---|",
        ]
        picked = [row for row in by_score if rule_id in row["excluded"]][:REVIEW_EXCLUDED_PER_RULE]
        for row in picked:
            detail = next(item["detail"] for item in row["exclusion_details"]
                          if item["rule"] == rule_id)
            lines.append(f"| {_cell(row['name'])} | {_cell(row['city'])} | "
                         f"{_cell(row['primary_site_type'])} | {_cell(row['employees'])} | "
                         f"{row['score']:.1f} | {_cell(detail)} |")
    return "\n".join(lines) + "\n"


def _counts(snapshot, config, ranked, excluded, limit) -> dict:
    ranked_by_cap, excluded_by_cap = Counter(), Counter()
    for row in ranked:
        ranked_by_cap.update(row["capabilities"])
    for row in excluded:
        excluded_by_cap.update(row["capabilities"])
    in_scope = ranked_by_cap + excluded_by_cap
    by_rule = Counter(rule_id for row in excluded for rule_id in row["excluded"])
    rule_order = [rule.id for rule in config.rules]
    ordered = [rule_id for rule_id in rule_order if by_rule[rule_id]]
    ordered += sorted(rule_id for rule_id in by_rule if rule_id not in rule_order)
    return {
        "by_capability": {
            cap: {"excluded": excluded_by_cap[cap], "in_scope": in_scope[cap],
                  "ranked": ranked_by_cap[cap]}
            for cap in sorted(in_scope)
        },
        "exclusions_by_rule": {rule_id: by_rule[rule_id] for rule_id in ordered},
        "ranked_rows_written": len(ranked[:limit] if limit else ranked),
        "sites_excluded": len(excluded),
        "sites_in_scope": len(ranked) + len(excluded),
        "sites_in_snapshot": len(snapshot.sites),
        "sites_ranked": len(ranked),
    }


def _input_summary(inputs, sites) -> dict | None:
    if inputs is None:
        return None
    matched = {row["line"] for site in sites for row in inputs.matches(site)}
    return {
        "rows": len(inputs.rows),
        "rows_matched": len(matched),
        "sha256": inputs.sha256,
        "unmatched_lines": sorted(row["line"] for row in inputs.rows if row["line"] not in matched),
    }


def write_ranking(snapshot_dir, out_dir, config=None, *, capabilities=None, limit=None,
                  exclusions_input=None, taxonomy=None, log=None) -> RankRun:
    """Rank a snapshot and write ranked.jsonl.gz, review-top.md and rank-manifest.json."""
    log = log or (lambda message: None)
    out_dir = build_module.check_output_dir(out_dir)
    config, requested, snapshot, context, inputs = _prepare(
        snapshot_dir, config, capabilities, limit, exclusions_input, taxonomy)
    log(f"rank {len(snapshot.sites)} sites from snapshot {snapshot.manifest['snapshot_id']}")
    ranked, excluded = _rank_sites(snapshot.sites, config, requested, context, inputs)
    counts = _counts(snapshot, config, ranked, excluded, limit)

    review_capabilities = sorted(requested if requested is not None else counts["by_capability"],
                                 key=lambda cap: (-config.capability_weights[cap], cap))
    per_capability = {}
    for cap in review_capabilities:
        log(f"review {cap}")
        rows, _ = _rank_sites(snapshot.sites, config, frozenset([cap]), context, inputs)
        per_capability[cap] = rows[:REVIEW_TOP]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = (ranked[:limit] if limit else ranked) + excluded
    lines = [build_module.canonical_json(row) for row in written]
    payload = build_module._gzip_lines(lines)
    ranked_path = out_dir / RANKED_FILE
    build_module._write_atomic(ranked_path, payload)
    run_scope = sorted(requested) if requested is not None else None
    review = _review(snapshot, config, review_capabilities, per_capability, counts, excluded,
                     run_scope).encode("utf-8")
    review_path = out_dir / REVIEW_FILE
    build_module._write_atomic(review_path, review)

    unverified = config.unverified_evidence()
    snapshot_taxonomy = (snapshot.manifest.get("taxonomy") or {}).get("sha256")
    manifest = {
        "attribution_required": snapshot.manifest.get("attribution_required", []),
        "counts": counts,
        "distribution": build_module.DISTRIBUTION,
        "exclusion_evidence_unverified": {"count": len(unverified), "items": unverified},
        "exclusion_rules": [{"id": rule.id, "label": rule.label, "reason": rule.reason}
                            for rule in config.rules],
        "exclusions_input": _input_summary(inputs, snapshot.sites),
        "files": {
            RANKED_FILE: {"bytes": len(payload), "lines": len(lines),
                          "sha256": hashlib.sha256(payload).hexdigest()},
            REVIEW_FILE: {"bytes": len(review), "sha256": hashlib.sha256(review).hexdigest()},
        },
        "license_union": snapshot.manifest.get("license_union", []),
        "rank_config": {"sha256": config.sha256, "updated_at": config.document["updated_at"],
                        "version": config.version, "weights": config.weights},
        "ranker": {"code_sha256": build_module.code_sha256(), "version": RANKER_VERSION},
        "schema": RANK_MANIFEST_SCHEMA,
        "scope": {"capabilities": run_scope, "limit": limit},
        "score_meaning": config.document.get("score_meaning"),
        "score_percentiles": _percentiles(row["score"] for row in ranked),
        "snapshot": {
            "site_schema": snapshot.manifest.get("site_schema"),
            "sites": len(snapshot.sites),
            "snapshot_id": snapshot.manifest["snapshot_id"],
            "states": snapshot.manifest.get("states"),
            "taxonomy": snapshot.manifest.get("taxonomy"),
        },
        "taxonomy_used": dict(context["taxonomy"],
                              matches_snapshot=context["taxonomy"]["sha256"] == snapshot_taxonomy),
    }
    manifest_path = out_dir / RANK_MANIFEST_FILE
    text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    build_module._write_atomic(manifest_path, text.encode("utf-8"))
    log(f"ranked {len(ranked)}, excluded {len(excluded)}")
    return RankRun(out_dir, ranked_path, manifest_path, review_path, manifest)
