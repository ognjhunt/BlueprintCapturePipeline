"""Owner command for the robot-team universe: ``python -m tools.team_universe <command>`` from the repository root.

plan reads only the reviewed query set and states the worst-case spend; it calls nothing. discover and screen are dry
runs unless --apply; with --apply they create Parallel Task runs, each admitted first over both stages against the
owner ceiling that the first --apply pins, the spend journal and the ledgers. collect reads run status and results
(reads are not billed; a team screen's cited pages are read then, so its contact is decided before it is stored),
verify reads the cited public pages, and rank and summary read the out dir. The out dir must be private (mode 700),
on durable storage outside every repository and outside /tmp. discover and screen refuse on the daily worker. Output
is counts and stable codes only, never team names, domains or addresses; names stay in the private out dir. The key
comes from --key-file (KEY=VALUE lines holding PARALLEL_API_KEY) or the environment, and is never printed or
written. No CRM write, draft or send.
"""
import argparse
import time
from pathlib import Path

from tools.daily_research import site_screen as ss
from tools.daily_research.site_screen import ScreenError
from tools.team_universe import rank as team_rank
from tools.team_universe import universe as tu

OK_STATES = frozenset({"planned", "complete", "pending"})
MAX_FILE_BYTES = 4 * 1024 * 1024


def read_file(path, code):
    """The bytes of one small input file (family weights or a rank config), bounded."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError:
        raise tu.TeamError(code) from None
    if len(raw) > MAX_FILE_BYTES:
        raise tu.TeamError(code)
    return raw


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m tools.team_universe", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="Count the reviewed queries and state the worst-case spend; no call")
    planner.add_argument("--processor", default=tu.DEFAULT_PROCESSOR)
    planner.add_argument("--ceiling-usd", help="Cap the worst case at this spend ceiling")
    planner.add_argument("--max-runs", type=int, help="Cap the worst case at this run limit")
    planner.add_argument("--screen-batch", type=int, help="Plan a screen of at most this many teams")
    for name, text in (("discover", "Run each reviewed discovery query that has no run yet"),
                       ("screen", "Screen each discovered team that has no run yet, in priority order")):
        spender = commands.add_parser(name, help=text + "; a dry run unless --apply")
        if name == "discover":
            spender.add_argument("--limit", type=int, help="Only the first N queries; the canary is --limit 1")
        else:
            spender.add_argument("--batch-size", type=int, help="Only the first N teams (at most --max-runs)")
            spender.add_argument("--include-unproven", action="store_true",
                                 help="Also screen teams whose discovery quote is not proven")
            spender.add_argument("--family-weights", type=Path,
                                 help="Private per-family weights: teams in the best-weighted families go first")
        spender.add_argument("--out", required=True, type=Path,
                             help="Private durable directory (mode 700) outside every repository and outside /tmp")
        spender.add_argument("--owner-reference", required=True, help="The owner decision the first --apply pins")
        spender.add_argument("--ceiling-usd", required=True,
                             help="Spend ceiling for both stages of this out dir; never above the pin")
        spender.add_argument("--max-runs", required=True, type=int,
                             help="Run limit for both stages of this out dir; never above the pin")
        spender.add_argument("--processor", default=tu.DEFAULT_PROCESSOR)
        spender.add_argument("--key-file", type=Path, help="KEY=VALUE file holding PARALLEL_API_KEY")
        spender.add_argument("--apply", action="store_true")
    collector = commands.add_parser("collect", help="Store each run's terminal result")
    collector.add_argument("--out", required=True, type=Path)
    collector.add_argument("--wait-seconds", type=int, default=ss.COLLECT_WAIT_SECONDS)
    collector.add_argument("--key-file", type=Path)
    for name, text in (("verify", "Check every quote against its cited page; write the records and the team list"),
                       ("summary", "Counts by stage, field, route and tier, and cost")):
        commands.add_parser(name, help=text).add_argument("--out", required=True, type=Path)
    ranker = commands.add_parser("rank", help="Qualify teams using a snapshot-bound private audit, then rank")
    ranker.add_argument("--out", required=True, type=Path)
    ranker.add_argument("--family-weights", required=True, type=Path,
                        help="Private per-family weights file, or a site screen summary.json")
    ranker.add_argument("--config", type=Path, help="A blueprint.team-rank.v1 config (default: the reviewed one)")
    ranker.add_argument("--audit", type=Path,
                        help="Private blueprint.team-manual-eligibility-audit.v1; absent => no qualified teams")
    exporter = commands.add_parser("export", help="Project the current reviewed qualification into private daily-agent evidence; offline")
    exporter.add_argument("--out", required=True, type=Path)
    exporter.add_argument("--audit", required=True, type=Path)
    exporter.add_argument("--destination", required=True, type=Path)
    exporter.add_argument("--config", type=Path)
    return parser


def main(argv=None, *, environ=None, transport=None, reader=None, monotonic=time.monotonic, sleep=time.sleep,
         today=None):
    args = build_parser().parse_args(argv)

    def client():
        key = ss.read_api_key(args.key_file, environ)
        return ss.TaskClient(key, **({"transport": transport} if transport is not None else {}))

    def spend():
        return {"owner_reference": args.owner_reference, "ceiling_usd": args.ceiling_usd, "max_runs": args.max_runs,
                "processor": args.processor, "apply": args.apply}

    if args.command in ("discover", "screen"):
        ss.refuse_on_worker()  # Before any file is made or the key is read.
    if args.command == "plan":
        _, query_set = tu.reviewed_query_set()
        result = tu.plan(query_set, processor=args.processor, ceiling_usd=args.ceiling_usd, max_runs=args.max_runs,
                         screen_batch=args.screen_batch)
    elif args.command == "discover":
        # The out dir guard and the query set check run before the key is read; the out dir is made only after both.
        ss.guard_out_dir(args.out)
        _, query_set = tu.reviewed_query_set()
        task_client = client()
        result = tu.discover(tu.TeamWorkspace(args.out, create=True), query_set, client=task_client, limit=args.limit,
                             **spend())
    elif args.command == "screen":
        weights = team_rank.load_weights(read_file(args.family_weights, "team_universe_family_weights_unreadable")) \
            if args.family_weights else None
        workspace = tu.TeamWorkspace(args.out)
        result = tu.screen(workspace, client=client(), batch_size=args.batch_size,
                           include_unproven=args.include_unproven, weights=weights, **spend())
    elif args.command == "collect":
        workspace = tu.TeamWorkspace(args.out)
        result = ss.collect(workspace, client=client(), reader=reader, today=today, wait_seconds=args.wait_seconds,
                            monotonic=monotonic, sleep=sleep)
    elif args.command == "verify":
        result = tu.verify(tu.TeamWorkspace(args.out), reader=reader, today=today)
    elif args.command == "export":
        from tools.team_universe import evidence_export
        raw = evidence_export.build(tu.TeamWorkspace(args.out), read_file(args.audit, "team_universe_audit_unreadable"),
            today=today, config_raw=read_file(args.config, "team_universe_rank_config_unreadable") if args.config else None)
        # Exclusive private output, never overwrite raw/audit or an earlier export.
        import os
        ss.guard_out_dir(args.destination.parent)
        descriptor = os.open(args.destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
        result = {"command": "export", "state": "complete", "sha256": ss._sha256(raw), "bytes": len(raw)}
    elif args.command == "rank":
        config = read_file(args.config, "team_universe_rank_config_unreadable") if args.config else None
        weights_raw = read_file(args.family_weights, "team_universe_family_weights_unreadable")
        audit = read_file(args.audit, "team_universe_audit_unreadable") if args.audit else None
        result = team_rank.rank(tu.TeamWorkspace(args.out), weights_raw, config_raw=config, audit_raw=audit, today=today)
    else:
        result = tu.summary(tu.TeamWorkspace(args.out), today=today)
    print(ss.canonical(result))
    return result


def run(argv=None):
    """The command's exit status: 0 when it planned or completed, or when runs are still pending; else 1, with one
    stable code printed and never provider text, a key or team data."""
    try:
        outcome = main(argv)
    except Exception as error:  # noqa: BLE001 - stable codes only; never provider, key or team text
        code = str(error) if isinstance(error, ScreenError) else "team_universe_operation_unavailable"
        print(ss.canonical({"state": "blocked", "error": code}))
        return 1
    return 0 if outcome.get("state") in OK_STATES else 1
