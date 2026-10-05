"""Command line: build, stats, import-raw, rank and export."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools.site_universe import build as build_module
from tools.site_universe import export as export_module
from tools.site_universe import rank as rank_module
from tools.site_universe import sources
from tools.site_universe.adapters import AdapterError, fsis_mpi, osha_ita
from tools.site_universe.fetch import FetchError, RawCache

MANUAL_VALIDATORS = {"fsis_mpi": fsis_mpi, "osha_ita": osha_ita}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _build(args) -> int:
    snapshot = build_module.build(
        [state.strip() for state in args.states.split(",") if state.strip()],
        [source.strip() for source in args.sources.split(",") if source.strip()],
        out_dir=args.out,
        raw_dir=args.raw_dir,
        refresh=args.refresh,
        allow_network=not args.offline,
        strict=args.strict,
        log=_log,
    )
    manifest = snapshot.manifest
    print(json.dumps({
        "manifest": str(snapshot.manifest_path),
        "sites": manifest["counts"]["sites"],
        "sites_file": str(snapshot.sites_path),
        "snapshot_id": snapshot.snapshot_id,
        "sources_skipped": manifest["sources_skipped"],
    }, indent=2, sort_keys=True))
    return 0


def _sample(out_dir: Path, size: int) -> list[dict]:
    sites = list(build_module.read_sites(out_dir))
    if not sites or size <= 0:
        return []
    step = max(1, len(sites) // size)
    picked = sites[::step][:size]
    return [{"category": s["category"], "city": s["city"], "name": s["name"]} for s in picked]


def _stats(args) -> int:
    out_dir = Path(args.dir)
    manifest = json.loads((out_dir / build_module.MANIFEST_FILE).read_text(encoding="utf-8"))
    data = (out_dir / build_module.SITES_FILE).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != manifest["snapshot_id"]:
        print(f"sites file SHA-256 {actual} does not match snapshot_id {manifest['snapshot_id']}",
              file=sys.stderr)
        return 2
    summary = {
        "counts": manifest["counts"],
        "inputs": [
            {key: row[key] for key in ("acquisition", "bytes", "label", "retrieved_at", "sha256",
                                       "source_id", "state")}
            for row in manifest["inputs"]
        ],
        "license_union": manifest["license_union"],
        "merge_stats": manifest["merge_stats"],
        "snapshot_id": manifest["snapshot_id"],
        "sources_skipped": manifest["sources_skipped"],
        "states": manifest["states"],
        "taxonomy": manifest["taxonomy"],
        "verified_sites_sha256": True,
    }
    if args.sample:
        summary["sample"] = _sample(out_dir, args.sample)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _import_raw(args) -> int:
    registry = sources.registry()
    entry = registry.get(args.source)
    if entry is None:
        raise SystemExit(f"unknown source {args.source}")
    sources.check_source(entry)
    if entry["status"] != "manual_import_only":
        raise SystemExit(f"{args.source} is fetched automatically; manual import is for blocked sources")
    url = args.url or entry.get("download_url")
    if not url or not url.startswith("https://"):
        raise SystemExit("pass --url with the https address the file was downloaded from")
    raw_dir = build_module.check_output_dir(Path(args.raw_dir) if args.raw_dir else Path(args.out) / "raw")
    adapter = MANUAL_VALIDATORS[args.source]

    def validate(path: Path) -> None:
        try:
            adapter.parse(path.read_bytes(), state=args.check_state, raw_sha256="0" * 64,
                          retrieved_at=args.retrieved_at or "2000-01-01T00:00:00Z")
        except AdapterError as error:
            raise FetchError(f"{args.source} adapter refused the file: {error}") from error

    cache = RawCache(raw_dir, allow_network=False)
    row = cache.import_file(url, args.file, source_id=args.source, retrieved_at=args.retrieved_at,
                            validate=validate)
    print(json.dumps(row.manifest_row(), indent=2, sort_keys=True))
    return 0


def _rank(args) -> int:
    if args.top is not None and args.top < 1:
        raise rank_module.RankError("--top must be at least 1")
    capabilities = None
    if args.capabilities:
        capabilities = [item.strip() for item in args.capabilities.split(",") if item.strip()]
    run = rank_module.write_ranking(
        args.snapshot,
        args.out,
        args.config or rank_module.RANK_CONFIG_PATH,
        capabilities=capabilities,
        limit=args.top,
        exclusions_input=args.exclusions_input,
        log=_log,
    )
    manifest = run.manifest
    print(json.dumps({
        "exclusions_by_rule": manifest["counts"]["exclusions_by_rule"],
        "manifest": str(run.manifest_path),
        "rank_config_sha256": manifest["rank_config"]["sha256"],
        "ranked_file": str(run.ranked_path),
        "review": str(run.review_path),
        "sites_excluded": manifest["counts"]["sites_excluded"],
        "sites_ranked": manifest["counts"]["sites_ranked"],
        "snapshot_id": manifest["snapshot"]["snapshot_id"],
    }, indent=2, sort_keys=True))
    return 0


def _export(args) -> int:
    result = export_module.export(
        args.ranking, args.snapshot, args.out, approval_reference=args.approval_reference,
        config=args.config, previous_snapshot=args.previous_snapshot, top=args.top,
        per_capability=args.per_capability, max_rows=args.max_rows, log=_log,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.site_universe")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build a snapshot for one or more states")
    build.add_argument("--states", required=True, help="comma-separated state codes, e.g. TX")
    build.add_argument("--out", required=True, help="output directory")
    build.add_argument("--raw-dir", help="raw cache directory (default: <out>/raw)")
    build.add_argument("--sources", default=",".join(build_module.DEFAULT_SOURCES))
    build.add_argument("--refresh", action="store_true", help="download cached raw files again")
    build.add_argument("--offline", action="store_true", help="use the raw cache only")
    build.add_argument("--strict", action="store_true", help="fail when any source is unavailable")
    build.set_defaults(handler=_build)

    stats = commands.add_parser("stats", help="summarize a snapshot directory")
    stats.add_argument("dir")
    stats.add_argument("--sample", type=int, default=0, help="include N evenly spaced sites")
    stats.set_defaults(handler=_stats)

    imported = commands.add_parser("import-raw", help="record a file downloaded in a browser")
    imported.add_argument("--source", required=True, choices=sorted(MANUAL_VALIDATORS))
    imported.add_argument("--file", required=True)
    imported.add_argument("--url", help="the https URL the file was downloaded from")
    imported.add_argument("--out", help="snapshot directory whose raw cache receives the file")
    imported.add_argument("--raw-dir", help="raw cache directory")
    imported.add_argument("--retrieved-at", help="download time, YYYY-MM-DDTHH:MM:SSZ (default: now)")
    imported.add_argument("--check-state", default="TX", help="state used to check the file parses")
    imported.set_defaults(handler=_import_raw)

    ranked = commands.add_parser("rank", help="rank a snapshot with rank_config.json")
    ranked.add_argument("--snapshot", required=True,
                        help="snapshot directory (sites.jsonl.gz, manifest.json)")
    ranked.add_argument("--out", required=True, help="output directory for the ranking files")
    ranked.add_argument("--capabilities", help="comma-separated capability ids (default: all)")
    ranked.add_argument("--top", type=int,
                        help="write only the top N ranked sites; excluded sites are always written")
    ranked.add_argument("--config",
                        help="rank config path (default: tools/site_universe/rank_config.json)")
    ranked.add_argument("--exclusions-input",
                        help="JSONL of site ids, names or operators to exclude, each with a reason")
    ranked.set_defaults(handler=_rank)

    exported = commands.add_parser(
        "export", help="export the ranked backlog that the daily research run's slice reads")
    exported.add_argument("--ranking", required=True, help="rank output directory (complete ranking)")
    exported.add_argument("--snapshot", required=True, help="the snapshot the ranking was made from")
    exported.add_argument("--out", required=True, help="output directory, outside the repository")
    exported.add_argument("--approval-reference", required=True,
                          help="the owner's approval of this ranking review")
    exported.add_argument("--config", help="the rank config the ranking used (default: rank_config.json)")
    exported.add_argument("--previous-snapshot", help="an earlier snapshot, to count new sites")
    exported.add_argument("--top", type=int, default=export_module.DEFAULT_TOP,
                          help="global top N ranked sites")
    exported.add_argument("--per-capability", type=int, default=export_module.DEFAULT_PER_CAPABILITY,
                          help="top N ranked sites of each lead capability")
    exported.add_argument("--max-rows", type=int, default=export_module.runtime.MAX_ROWS,
                          help="refuse an export with more rows than this")
    exported.set_defaults(handler=_export)

    args = parser.parse_args(argv)
    if args.command == "import-raw" and not (args.out or args.raw_dir):
        parser.error("import-raw needs --out or --raw-dir")
    try:
        return args.handler(args)
    except (sources.SourceRefused, build_module.BuildError, FetchError,
            rank_module.RankError, export_module.ExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
