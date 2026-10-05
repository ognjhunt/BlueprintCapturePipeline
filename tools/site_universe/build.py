"""Build a deterministic site universe snapshot from the raw cache.

``build(state_codes, sources, out_dir=...)`` writes:

- ``sites.jsonl.gz``: one canonical JSON site per line, sorted by ``site_id``,
  gzip with mtime 0 and no file name;
- ``manifest.json``: schema version, raw input SHA-256s, the registry entries
  used, counts, merge statistics and the license union.

The snapshot id is the SHA-256 of ``sites.jsonl.gz``. The output depends only
on the raw cache bytes, the code and ``taxonomy.json``; nothing reads the clock.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tools.site_universe import BUILDER_VERSION, SCHEMA_VERSION, dedupe, geo, normalize, sources
from tools.site_universe import taxonomy as taxonomy_module
from tools.site_universe.adapters import AdapterError, epa_frs, fsis_mpi, osha_ita, osm_overpass
from tools.site_universe.fetch import FetchError, RawCache

MANIFEST_SCHEMA = "blueprint.site_universe.manifest.v1"
# Snapshots hold site-level records and ODbL (share-alike) data: never publish or commit them.
DISTRIBUTION = "internal_only"
SITES_FILE = "sites.jsonl.gz"
MANIFEST_FILE = "manifest.json"
DEFAULT_SOURCES = ("osha_ita", "epa_frs", "osm_overpass", "fsis_mpi")
OVERPASS_MIN_INTERVAL_S = 15.0
ADAPTERS = {
    "epa_frs": epa_frs,
    "fsis_mpi": fsis_mpi,
    "osha_ita": osha_ita,
    "osm_overpass": osm_overpass,
}


class BuildError(RuntimeError):
    """The build cannot produce a snapshot."""


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    out_dir: Path
    sites_path: Path
    manifest_path: Path
    manifest: dict


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def code_sha256() -> str:
    """Digest of the producer's own source files, for provenance."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


# --- raw inputs -------------------------------------------------------------------------
def _raw_entries(source_id: str, entry: dict, state: str, cache: RawCache, taxonomy, log):
    """Return ``[(label, RawEntry)]`` for one source and state, or raise FetchError."""
    if entry["status"] == "manual_import_only":
        imported = [item for item in cache.entries() if item.source_id == source_id]
        if not imported:
            raise FetchError(
                f"{source_id} needs a manual import: download the file in a browser from "
                f"{entry['landing_url']} and run 'python -m tools.site_universe import-raw'"
            )
        latest = max(imported, key=lambda item: (item.retrieved_at, item.url))
        return [("manual_import", latest)]
    if source_id == "epa_frs":
        url = epa_frs.download_url(state)
        return [("state_zip", cache.get(url, source_id=source_id, timeout_s=900))]
    if source_id == "osm_overpass":
        out = []
        for group, query in taxonomy.overpass_queries(state):
            url = osm_overpass.query_url(query)
            out.append(
                (
                    group,
                    cache.get(
                        url,
                        source_id=source_id,
                        suffix=".json",
                        timeout_s=360,
                        min_interval_s=OVERPASS_MIN_INTERVAL_S,
                        validate=osm_overpass.validate_payload,
                    ),
                )
            )
        return out
    raise BuildError(f"no acquisition rule for {source_id}")


# --- classification --------------------------------------------------------------------
PLACEHOLDER_NAMES = frozenset(
    {"N A", "NA", "NO NAME", "NONE", "NOT AVAILABLE", "NULL", "TBD", "TEST", "UNK", "UNKNOWN",
     "UNNAMED", "VACANT"}
)


def _classify(record, taxonomy):
    if " ".join(normalize.ascii_upper(record.name).replace("/", " ").split()) in PLACEHOLDER_NAMES:
        return None, "dropped_placeholder_name"
    if record.source_id == "osm_overpass":
        if not normalize.name_tokens(record.name):
            return None, "dropped_generic_name"
        result = taxonomy.classify_osm(record.attributes.get("osm_tags", {}), record.name)
    elif record.source_id == "fsis_mpi":
        result = taxonomy.classify_fsis()
    else:
        primary = record.attributes.get("primary_code") or record.naics
        result = taxonomy.classify_codes(record.codes, system=record.code_system, primary=primary)
    if result.drop_reason:
        return None, f"dropped_{result.drop_reason}"
    return result, None


# --- site assembly -------------------------------------------------------------------------
def _pick(records, field_key, predicate):
    order = sources.FIELD_PRIORITY[field_key]
    ranked = sorted(
        (record for record in records if record.source_id in order and predicate(record)),
        key=lambda record: (order.index(record.source_id), record.source_id, record.source_record_id),
    )
    return ranked[0] if ranked else None


def _address_key(record):
    return normalize.address_key(record.street, record.city, record.state, record.postal_code)


def _assemble(cluster, taxonomy, registry) -> dict:
    records = sorted(cluster.members, key=lambda r: (r.source_id, r.source_record_id))
    address = (
        _pick(records, "address", lambda r: _address_key(r) is not None)
        or _pick(records, "address", lambda r: bool(r.street))
        or _pick(records, "address", lambda r: bool(r.city or r.postal_code))
    )
    coordinates = _pick(
        records,
        "coordinates",
        lambda r: r.lat is not None and r.attributes.get("coordinate_precision") != "approximate",
    ) or _pick(records, "coordinates", lambda r: r.lat is not None)
    named = _pick(records, "name", lambda r: bool(r.name))
    operator = _pick(records, "operator", lambda r: bool(r.operator))
    naics = _pick(records, "naics", lambda r: bool(r.naics))
    categorized = _pick(records, "category", lambda r: bool(r.category))
    employees = [r.employees for r in records if r.source_id == "osha_ita" and r.employees is not None]
    buildings = {
        r.source_record_id: r.building_area_m2
        for r in records
        if r.source_id == "osm_overpass" and r.building_area_m2
    }
    site_areas = [r.attributes["site_area_m2"] for r in records if r.attributes.get("site_area_m2")]
    site_types = sorted({t for r in records for t in r.site_types})
    matches = taxonomy.matches(site_types)
    matched_by = {
        type_id: sorted(
            {f"{r.source_id}:{m}" for r in records for m in r.site_type_matches.get(type_id, ())}
        )
        for type_id in matches["site_types"]
    }
    primary_type = categorized.attributes.get("primary_site_type") if categorized else None
    matches["primary_site_type"] = primary_type
    matches["matched_by"] = matched_by
    frs = [r for r in records if r.source_id == "epa_frs"]
    activity = None
    if frs:
        states = {r.attributes.get("activity_status") for r in frs}
        activity = "active" if "active" in states else "unknown" if "unknown" in states else "inactive"
    years = [r.attributes.get("last_activity_year") for r in frs if r.attributes.get("last_activity_year")]
    bands = sorted(
        {
            band
            for r in records
            for band in (r.attributes.get("osha_size_band"), r.attributes.get("fsis_size"))
            if band
        }
    )
    source_ids = sorted({r.source_id for r in records})
    licenses = [sources.license_entry(registry[source_id]) for source_id in source_ids]
    attributions = sorted({item["attribution"] for item in licenses if item["attribution_required"]})
    building_area = round(sum(buildings.values()), 1) if buildings else None
    lat = coordinates.lat if coordinates else None
    lon = coordinates.lon if coordinates else None
    category = categorized.category if categorized else None
    site = {
        "address_source": address.source_id if address else None,
        "attribution_required": bool(attributions),
        "attributions": attributions,
        "building_area_m2": building_area,
        "building_area_source": "osm_overpass" if buildings else None,
        "category": category,
        "city": address.city if address else None,
        "coordinate_source": coordinates.source_id if coordinates else None,
        "country": "US",
        "employees": max(employees) if employees else None,
        "employees_source": "osha_ita" if employees else None,
        "features": {
            "building_area_m2": building_area,
            "building_count": len(buildings),
            "capability_count": len(matches["capabilities"]),
            "category": category,
            "coordinate_precision": (
                coordinates.attributes.get("coordinate_precision", "precise") if coordinates else None
            ),
            "employee_bands": bands,
            "employees": max(employees) if employees else None,
            "frs_activity_status": activity,
            "frs_last_activity_year": max(years) if years else None,
            "has_coordinates": lat is not None,
            "has_street_address": bool(address and _address_key(address)),
            "primary_site_type": primary_type,
            "record_count": len(records),
            "site_area_m2": max(site_areas) if site_areas else None,
            "site_type_count": len(matches["site_types"]),
            "source_count": len(source_ids),
            "sources": source_ids,
            "taxonomy_row_count": len(matches["rows"]),
        },
        "lat": lat,
        "licenses": [
            {key: item[key] for key in ("id", "share_alike", "source_id", "url")} for item in licenses
        ],
        "lon": lon,
        "merges": sorted(cluster.merges, key=lambda e: (e["a"], e["b"], e["reason"])),
        "name": named.name if named else None,
        "names": sorted({r.name for r in records if r.name}),
        "naics": naics.naics if naics else None,
        "operator": operator.operator if operator else None,
        "postal_code": address.postal_code if address else None,
        "records": [r.to_dict() for r in records],
        "state": address.state if address else records[0].state,
        "street": address.street if address else None,
        "taxonomy_matches": matches,
        "unit": address.unit if address else None,
    }
    # The id anchors on the earliest record (by source id and record id), so it depends on this
    # site's own records only, never on which other sites share its address.
    anchor = next((r for r in records if r.name), None)
    site["_name_tokens"] = " ".join(dedupe.matching_tokens(anchor)) if anchor else ""
    site["_address_key"] = _address_key(address) if address else None
    site["_anchor_ref"] = records[0].ref
    return site


def _assign_ids(sites: list[dict]) -> dict:
    """Set ``site_id`` (the SHA-256 of ``id_key``) on every site; return id statistics.

    ``id_key`` uses the site's own records only: the address key of its address
    record plus the normalized name of its earliest record (by source id and
    record id); without an address key, a geohash-7 cell plus that name; without
    coordinates, that name plus city, state and ZIP. So a site whose records do
    not change keeps its id when other sites appear or go. When several sites
    share a key, the site with the earliest anchor record keeps it and the
    others add their anchor record reference.
    """
    stats = Counter()
    for site in sites:
        name_key = site.pop("_name_tokens")
        address_key = site.pop("_address_key")
        if address_key:
            basis, key = "address_name", f"{address_key}|{name_key}"
        elif site["lat"] is not None:
            basis, key = "geohash7_name", f"{geo.geohash(site['lat'], site['lon'], 7)}|{name_key}"
        else:
            basis = "name_locality"
            key = f"{name_key}|{site['city'] or ''}|{site['state'] or ''}|{site['postal_code'] or ''}"
        site["id_basis"], site["id_key"] = basis, key
    shared = defaultdict(list)
    for site in sites:
        shared[site["id_key"]].append(site)
    for key, group in shared.items():
        if len(group) > 1:
            stats["id_key_collisions"] += len(group)
            for site in sorted(group, key=lambda site: site["_anchor_ref"])[1:]:
                site["id_key"] = f"{key}|{site['_anchor_ref']}"
    for site in sites:
        del site["_anchor_ref"]
        site["site_id"] = hashlib.sha256(site["id_key"].encode("utf-8")).hexdigest()
        stats[f"id_basis_{site['id_basis']}"] += 1
    if len({site["site_id"] for site in sites}) != len(sites):
        raise BuildError("site_id collision after disambiguation")
    return dict(sorted(stats.items()))


def _gzip_lines(lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        for line in lines:
            handle.write(line.encode("utf-8"))
            handle.write(b"\n")
    return buffer.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


# --- repository guard --------------------------------------------------------------------
def _git_common_dir(root: Path) -> Path | None:
    """The git common directory of the worktree at ``root``, or None when it is not one."""
    marker = root / ".git"
    if marker.is_dir():
        return marker.resolve()
    if not marker.is_file():
        return None
    text = marker.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("gitdir:"):
        return None
    gitdir = (root / text[len("gitdir:"):].strip()).resolve()
    commondir = gitdir / "commondir"
    if commondir.is_file():
        return (gitdir / commondir.read_text(encoding="utf-8").strip()).resolve()
    return gitdir


def _worktree_roots(path: Path):
    """``path`` and its parents that hold a ``.git`` entry, nearest first."""
    return [candidate for candidate in (path, *path.parents) if (candidate / ".git").exists()]


def _code_worktree() -> Path | None:
    """The git worktree that holds this package, or None when it is installed elsewhere."""
    roots = _worktree_roots(Path(__file__).resolve().parent)
    return roots[0] if roots else None


def check_output_dir(path) -> Path:
    """Refuse an output directory inside a worktree of this repository (fail closed).

    Snapshots, rankings and the raw cache hold real site records and ODbL data,
    and the repository is public. A worktree counts as this repository's when it
    is the worktree of this code, shares its git common directory, or holds
    ``tools/site_universe``. Returns the resolved path.
    """
    target = Path(path).expanduser().resolve()
    code_root = _code_worktree()
    code_common = _git_common_dir(code_root) if code_root else None
    for root in _worktree_roots(target):
        if (
            root == code_root
            or (code_common is not None and _git_common_dir(root) == code_common)
            or (root / "tools" / "site_universe").is_dir()
        ):
            raise BuildError(
                f"refusing to write site universe data under {target}: it is inside the repository "
                f"worktree {root}, and the data holds real site records and ODbL data. Write it "
                "outside every checkout of this repository."
            )
    return target


# --- build ----------------------------------------------------------------------------------
def build(
    state_codes,
    source_ids=DEFAULT_SOURCES,
    *,
    out_dir,
    raw_dir=None,
    refresh: bool = False,
    allow_network: bool = True,
    strict: bool = False,
    taxonomy=None,
    cache: RawCache | None = None,
    log=None,
) -> Snapshot:
    log = log or (lambda message: None)
    states = sorted({normalize.normalize_state(code) or "" for code in state_codes})
    if not states or any(state not in normalize.STATE_NAMES for state in states):
        raise BuildError(f"unknown state codes: {list(state_codes)}")
    registry = sources.registry()
    requested = sorted(set(source_ids))
    unknown = [source_id for source_id in requested if source_id not in registry]
    if unknown:
        raise sources.SourceRefused(f"unknown sources: {unknown}")
    for source_id in requested:
        sources.check_source(registry[source_id])
    taxonomy = taxonomy or taxonomy_module.load()
    out_dir = check_output_dir(out_dir)
    if raw_dir is not None:
        raw_dir = check_output_dir(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = cache or RawCache(
        Path(raw_dir) if raw_dir else out_dir / "raw",
        allow_network=allow_network,
        refresh=refresh,
        log=log,
    )

    inputs, skipped, parse_stats = [], [], {}
    records_by_ref: dict[str, object] = {}
    per_source = defaultdict(Counter)
    for state in states:
        for source_id in requested:
            entry = registry[source_id]
            try:
                raws = _raw_entries(source_id, entry, state, cache, taxonomy, log)
            except FetchError as error:
                if strict:
                    raise
                skipped.append({"reason": str(error), "source_id": source_id, "state": state})
                log(f"skip {source_id} {state}: {error}")
                continue
            for label, raw in raws:
                try:
                    result = ADAPTERS[source_id].parse(
                        raw.read_bytes(), state=state, raw_sha256=raw.sha256, retrieved_at=raw.retrieved_at
                    )
                except AdapterError as error:
                    if strict:
                        raise
                    skipped.append({"reason": f"adapter refused raw input: {error}", "source_id": source_id,
                                    "state": state, "url": raw.url})
                    log(f"skip {source_id} {state} {label}: {error}")
                    continue
                row = raw.manifest_row()
                row.update({"label": label, "state": state})
                inputs.append(row)
                parse_stats[f"{source_id}:{state}:{label}"] = dict(sorted(result.stats.items()))
                for record in result.records:
                    per_source[source_id]["parsed"] += 1
                    if record.ref in records_by_ref:
                        per_source[source_id]["duplicate_across_raw_inputs"] += 1
                        continue
                    classification, drop = _classify(record, taxonomy)
                    if drop:
                        per_source[source_id][drop] += 1
                        continue
                    record.site_types = classification.site_types
                    record.site_type_matches = classification.matched_by
                    record.category = taxonomy.category(classification.primary)
                    record.attributes["primary_site_type"] = classification.primary
                    records_by_ref[record.ref] = record
                    per_source[source_id]["kept"] += 1
    records = [records_by_ref[ref] for ref in sorted(records_by_ref)]
    if not records:
        raise BuildError(f"no records survived for states {states}; skipped: {skipped}")
    log(f"dedupe {len(records)} records")
    clusters, merge_stats = dedupe.cluster(records)
    sites = [_assemble(cluster, taxonomy, registry) for cluster in clusters]
    id_stats = _assign_ids(sites)
    sites.sort(key=lambda site: site["site_id"])
    lines = [canonical_json(site) for site in sites]
    payload = _gzip_lines(lines)
    snapshot_id = hashlib.sha256(payload).hexdigest()

    contributing = sorted({source for site in sites for source in site["features"]["sources"]})
    counts = _counts(sites, taxonomy)
    counts["records_by_source"] = {key: dict(sorted(value.items())) for key, value in sorted(per_source.items())}
    combos = Counter("+".join(site["features"]["sources"]) for site in sites)
    merge_stats = dict(merge_stats)
    merge_stats.update(id_stats)
    merge_stats["multi_source_sites"] = sum(1 for site in sites if site["features"]["source_count"] > 1)
    merge_stats["sites_by_source_combination"] = dict(sorted(combos.items()))
    manifest = {
        "attribution_required": sorted(
            {registry[s]["attribution"] for s in contributing if registry[s]["attribution_required"]}
        ),
        "builder": {"code_sha256": code_sha256(), "version": BUILDER_VERSION},
        "counts": counts,
        "distribution": DISTRIBUTION,
        "files": {
            SITES_FILE: {"bytes": len(payload), "lines": len(lines), "sha256": snapshot_id},
        },
        "inputs": sorted(inputs, key=lambda row: (row["source_id"], row["state"], row["url"])),
        "license_union": [sources.license_entry(registry[s]) for s in contributing],
        "merge_stats": merge_stats,
        "parse_stats": dict(sorted(parse_stats.items())),
        "schema": MANIFEST_SCHEMA,
        "site_schema": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "sources_requested": requested,
        "sources_skipped": sorted(skipped, key=lambda row: (row["source_id"], row["state"])),
        "sources_used": [registry[s] for s in contributing],
        "states": states,
        "taxonomy": taxonomy.summary(),
    }
    sites_path = out_dir / SITES_FILE
    manifest_path = out_dir / MANIFEST_FILE
    _write_atomic(sites_path, payload)
    _write_atomic(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())
    log(f"snapshot {snapshot_id}: {len(sites)} sites")
    return Snapshot(snapshot_id, out_dir, sites_path, manifest_path, manifest)


def _counts(sites: list[dict], taxonomy) -> dict:
    by_category, by_type, by_row, by_capability = Counter(), Counter(), Counter(), Counter()
    by_primary_type = Counter()
    for site in sites:
        by_category[site["category"] or "none"] += 1
        matches = site["taxonomy_matches"]
        by_primary_type[matches["primary_site_type"] or "none"] += 1
        by_type.update(matches["site_types"])
        by_row.update(matches["rows"])
        by_capability.update(matches["capabilities"])
    features = [site["features"] for site in sites]
    by_source = Counter(source for f in features for source in f["sources"])
    return {
        "sites": len(sites),
        "sites_by_source": dict(sorted(by_source.items())),
        "sites_by_capability": dict(sorted(by_capability.items())),
        "sites_by_category": dict(sorted(by_category.items())),
        "sites_by_frs_activity": dict(
            sorted(Counter(f["frs_activity_status"] or "not_in_frs" for f in features).items())
        ),
        "sites_by_primary_site_type": dict(sorted(by_primary_type.items())),
        "sites_by_site_type": dict(sorted(by_type.items())),
        "sites_by_taxonomy_row": dict(sorted(by_row.items())),
        "sites_with_building_area": sum(1 for f in features if f["building_area_m2"]),
        "sites_with_coordinates": sum(1 for f in features if f["has_coordinates"]),
        "sites_with_employees": sum(1 for f in features if f["employees"] is not None),
        "sites_with_street_address": sum(1 for f in features if f["has_street_address"]),
        "taxonomy_rows_with_no_sites": sorted(
            row["row_id"] for row in taxonomy.active_rows if row["row_id"] not in by_row
        ),
    }


def read_sites(out_dir):
    with gzip.open(Path(out_dir) / SITES_FILE, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
