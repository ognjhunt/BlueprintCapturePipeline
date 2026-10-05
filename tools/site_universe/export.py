"""Export one ranked backlog for the daily research run's site universe slice.

``python -m tools.site_universe export --ranking DIR --snapshot DIR --out DIR --approval-reference REF``
writes ``backlog.v1.json.gz``: one gzip member (mtime 0) of the canonical JSON document
``{schema_version, manifest, rows}`` that ``tools/daily_research/site_universe.load_export``
accepts. The runtime loader is the contract: the export runs it on its own bytes and refuses
anything it would refuse, before writing.

Rows are ranked rows only: the global top ``--top`` plus the top ``--per-capability`` rows of
each lead capability, in global rank order. No coordinates or footprints. The export checks
the rank manifest's file SHA-256s, the rank config SHA-256 and the snapshot, needs a complete
ranking (no ``--top`` or ``--capabilities``), allows only the runtime's reviewed licenses and
refuses an ``--out`` inside the repository. It holds ODbL data, so it stays internal.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from tools.daily_research import site_universe as runtime
from tools.site_universe import build as build_module
from tools.site_universe import rank as rank_module
from tools.site_universe import taxonomy as taxonomy_module

DEFAULT_TOP = 3000
DEFAULT_PER_CAPABILITY = 100
SEED_WEIGHT = 0.5  # Default selection policy: one seed per capability weighted at least this, in weight order.
FIT_CHARACTERS = 240
MAX_ALIASES = 20
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_SPACE = re.compile(r"\s+")


class ExportError(RuntimeError):
    """The ranking cannot be exported."""


def _require(condition, message: str) -> None:
    if not condition:
        raise ExportError(message)


def _clean(value):
    """Display text without control characters; empty becomes None. Never changes other text."""
    if value is None:
        return None
    text = _SPACE.sub(" ", _CONTROL.sub(" ", str(value))).strip()
    return text or None


def _fit(components: dict) -> str:
    top = sorted(components.items(), key=lambda item: (-abs(item[1]["points"]), item[0]))[:3]
    text = _clean("; ".join(f"{name} {item['points']:+.1f} ({item['basis']})" for name, item in top)) or "no fit evidence"
    return text if len(text) <= FIT_CHARACTERS else text[:FIT_CHARACTERS - 1] + "…"


def _row(row: dict, site: dict, lead: str) -> dict:
    name = _clean(row.get("name"))
    aliases = sorted({_clean(item) for item in site.get("names") or ()} - {None, name})[:MAX_ALIASES]
    operator_key, name_key = rank_module.operator_keys(site)
    return {
        "aliases": aliases,
        "attribution_required": bool(row.get("attribution_required")),
        "capabilities": sorted(set(row["capabilities"])),
        "category": row.get("category"),
        "city": _clean(row.get("city")),
        "employees": row.get("employees"),
        "fit": _fit(row["components"]),
        "group_key": operator_key or name_key,
        "lead_capability": lead,
        "naics": row.get("naics"),
        "name": name,
        "operator": _clean(row.get("operator")),
        "postal_code": row.get("postal_code"),
        "primary_site_type": row.get("primary_site_type"),
        "rank": row["rank"],
        "score": row["score"],
        "site_id": row["site_id"],
        "sources": sorted(set(row.get("sources") or ())),
        "state": row.get("state"),
        "street": _clean(row.get("street")),
    }


def _approval(value) -> str:
    _require(isinstance(value, str) and runtime.ASCII.fullmatch(value) and value.strip()
             and not value.strip().upper().startswith("PENDING"),
             "--approval-reference must be the owner's printable, non-PENDING approval reference")
    return value


def _verified_files(ranking_dir: Path, manifest: dict) -> dict:
    files = manifest.get("files")
    _require(isinstance(files, dict) and rank_module.RANKED_FILE in files,
             f"{ranking_dir}: the rank manifest lists no {rank_module.RANKED_FILE}")
    data = {}
    for name, entry in sorted(files.items()):
        path = ranking_dir / name
        _require(path.is_file(), f"{path}: missing")
        raw = path.read_bytes()
        _require(isinstance(entry, dict) and hashlib.sha256(raw).hexdigest() == entry.get("sha256")
                 and len(raw) == entry.get("bytes"), f"{path}: SHA-256 or size does not match the rank manifest")
        data[name] = raw
    return data


def export(ranking_dir, snapshot_dir, out_dir, *, approval_reference, config=None, previous_snapshot=None,
           top=DEFAULT_TOP, per_capability=DEFAULT_PER_CAPABILITY, max_rows=runtime.MAX_ROWS, taxonomy=None,
           log=None) -> dict:
    """Write ``backlog.v1.json.gz`` under ``out_dir``; return a summary with counts and ids only."""
    log = log or (lambda message: None)
    out_dir = build_module.check_output_dir(out_dir)
    _approval(approval_reference)
    for value, low, high, flag in ((top, 1, None, "--top"), (per_capability, 0, None, "--per-capability"),
                                   (max_rows, 1, runtime.MAX_ROWS, "--max-rows")):
        _require(type(value) is int and value >= low and (high is None or value <= high),
                 f"{flag} must be an integer from {low}" + (f" to {high}" if high else ""))
    ranking_dir = Path(ranking_dir)
    manifest_path = ranking_dir / rank_module.RANK_MANIFEST_FILE
    _require(manifest_path.is_file(), f"{ranking_dir}: needs {rank_module.RANK_MANIFEST_FILE}")
    manifest_raw = manifest_path.read_bytes()
    try:
        rank_manifest = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"{manifest_path}: not JSON: {error}") from error
    _require(isinstance(rank_manifest, dict) and rank_manifest.get("schema") == rank_module.RANK_MANIFEST_SCHEMA,
             f"{manifest_path}: schema must be {rank_module.RANK_MANIFEST_SCHEMA}")
    _require(rank_manifest.get("distribution") == build_module.DISTRIBUTION,
             f"{manifest_path}: distribution must be {build_module.DISTRIBUTION}")
    _require(rank_manifest.get("scope") == {"capabilities": None, "limit": None},
             "export needs a complete ranking: rank without --top or --capabilities")
    files = _verified_files(ranking_dir, rank_manifest)
    taxonomy = taxonomy or taxonomy_module.load()
    config = rank_module.as_config(config, taxonomy)  # A RankConfig, a path, or None (the committed file).
    _require(config.sha256 == (rank_manifest.get("rank_config") or {}).get("sha256"),
             "the rank config SHA-256 does not match the one the ranking used; pass that --config")
    snapshot = rank_module.load_snapshot(snapshot_dir)
    _require(snapshot.manifest.get("distribution") == build_module.DISTRIBUTION,
             f"{snapshot_dir}: distribution must be {build_module.DISTRIBUTION}")
    snapshot_id = snapshot.manifest["snapshot_id"]
    _require((rank_manifest.get("snapshot") or {}).get("snapshot_id") == snapshot_id,
             "the ranking was made from another snapshot")
    licenses = sorted({entry.get("id") if isinstance(entry, dict) else entry
                       for entry in snapshot.manifest.get("license_union") or ()})
    _require(licenses and set(licenses) <= runtime.LICENSES,
             f"licenses {sorted(set(licenses) - runtime.LICENSES) or licenses} are not among the reviewed "
             f"{sorted(runtime.LICENSES)}; the export fails closed")

    lines = gzip.decompress(files[rank_module.RANKED_FILE]).decode("utf-8").splitlines()
    ranked = [row for row in (json.loads(line) for line in lines if line) if row.get("status") == "ranked"]
    counts = rank_manifest.get("counts") or {}
    _require(ranked and len(ranked) == counts.get("sites_ranked")
             and [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1)),
             "the ranked file must hold every ranked site in rank order")
    leads = {row["site_id"]: rank_module.lead_capability(row, config) for row in ranked}
    chosen, per_lead = set(), Counter()
    for row in ranked:
        lead = leads[row["site_id"]]
        if row["rank"] <= top or per_lead[lead] < per_capability:
            chosen.add(row["site_id"])
        per_lead[lead] += 1
    _require(len(chosen) <= max_rows, f"the export would hold {len(chosen)} rows; the limit is {max_rows}")
    sites = {site["site_id"]: site for site in snapshot.sites}
    rows = [_row(row, sites[row["site_id"]], leads[row["site_id"]]) for row in ranked if row["site_id"] in chosen]

    previous_id = new_sites = None
    if previous_snapshot is not None:
        previous = rank_module.load_snapshot(previous_snapshot)
        previous_id = previous.manifest["snapshot_id"]
        _require(previous_id != snapshot_id, "--previous-snapshot is the same snapshot")
        new_sites = len(set(sites) - {site["site_id"] for site in previous.sites})
    weights = dict(sorted(config.capability_weights.items()))
    manifest = {
        "approval_reference": approval_reference,
        "attribution": sorted(set(snapshot.manifest.get("attribution_required") or ())),
        "capability_weights": weights,
        "counts": {"excluded": counts.get("sites_excluded"), "ranked": len(ranked), "rows": len(rows),
                   "sites": counts.get("sites_in_snapshot")},
        "distribution": build_module.DISTRIBUTION,
        "exclusion_counts": dict(sorted((counts.get("exclusions_by_rule") or {}).items())),
        "lead_capability_counts": dict(sorted(Counter(leads.values()).items())),
        "license_union": licenses,
        "new_sites": new_sites,
        "previous_snapshot_id": previous_id,
        "rank_config_sha256": config.sha256,
        "rank_config_version": (rank_manifest.get("rank_config") or {}).get("version"),
        "rank_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "ranked_file_sha256": rank_manifest["files"][rank_module.RANKED_FILE]["sha256"],
        "ranker_sha256": (rank_manifest.get("ranker") or {}).get("code_sha256"),
        "rows_sha256": runtime.rows_digest(rows),
        "selection_policy": {"seed_capabilities": [cap for cap, weight in sorted(weights.items(),
                                                   key=lambda item: (-item[1], item[0])) if weight >= SEED_WEIGHT]},
        "snapshot_id": snapshot_id,
        "states": sorted(set(snapshot.manifest.get("states") or ())),
        "taxonomy_sha256": (rank_manifest.get("taxonomy_used") or {}).get("sha256"),
    }
    data = build_module.canonical_json({"manifest": manifest, "rows": rows,
                                        "schema_version": runtime.EXPORT}).encode("utf-8")
    _require(len(data) <= runtime.MAX_RAW_BYTES, f"the export is {len(data)} bytes raw; the limit is {runtime.MAX_RAW_BYTES}")
    raw = gzip.compress(data, compresslevel=9, mtime=0)
    _require(len(raw) <= runtime.MAX_OBJECT_BYTES,
             f"the export is {len(raw)} bytes gzip; the limit is {runtime.MAX_OBJECT_BYTES}")
    try:
        runtime.load_export(raw)
    except runtime.SiteUniverseError as error:
        raise ExportError(f"the export does not meet the runtime contract ({error})") from error
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / runtime.OBJECT_NAME
    build_module._write_atomic(path, raw)
    sha = hashlib.sha256(raw).hexdigest()
    log(f"exported {len(rows)} of {len(ranked)} ranked sites from snapshot {snapshot_id}")
    return {"bytes": len(raw), "export": str(path), "license_union": licenses, "rank_config_sha256": config.sha256,
            "rows": len(rows), "rows_by_lead_capability": dict(sorted(Counter(row["lead_capability"] for row in rows).items())),
            "sha256": sha, "snapshot_id": snapshot_id, "uri": runtime.object_uri(sha)}
