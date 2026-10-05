"""Owner command for the per-site research line (site screen) and its contact stage.

plan reads only the input file. run and contact are dry runs unless --apply; with --apply they create
Parallel Task runs, each admitted first over both stages against the owner ceiling the first --apply
pins, the spend journal and the ledgers. collect reads run status and results (reads are not billed),
verify reads the cited public pages and summary reads the out dir. The out dir must be on durable storage
outside every repository and outside /tmp. run and contact refuse on the daily worker. Output is counts
and stable codes only, never site names, people or addresses. The key comes from PARALLEL_API_KEY, or
from --key-file (KEY=VALUE lines); it is never printed or written. No CRM write, draft or send.
"""
import argparse
import os
import time
from pathlib import Path

from tools.daily_research import site_screen
from tools.daily_research.site_screen import ScreenError

OK_STATES = frozenset({"planned", "complete", "pending"})


def api_key(key_file=None, environ=None):
    """PARALLEL_API_KEY from --key-file when one is given, else from the environment."""
    if key_file is None:
        value = (os.environ if environ is None else environ).get(site_screen.API_KEY_ENV, "")
    else:
        try:
            lines = Path(key_file).read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            raise ScreenError("site_screen_key_file_unreadable") from None
        value = ""
        for line in lines:
            line = line.strip()
            name, separator, item = line.removeprefix("export ").partition("=")
            if separator and not line.startswith("#") and name.strip() == site_screen.API_KEY_ENV:
                value = item.strip().strip("\"'")
    if not value:
        raise ScreenError("site_screen_api_key_missing")
    return value


def main(argv=None, *, environ=None, transport=None, reader=None, monotonic=time.monotonic, sleep=time.sleep,
         today=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="Count sites, refusals and cost in one input file")
    planner.add_argument("--input", required=True, type=Path)
    planner.add_argument("--processor", default=site_screen.DEFAULT_PROCESSOR)
    planner.add_argument("--batch-size", type=int, help="Plan this batch: rank order plus about one calibration site in ten")
    planner.add_argument("--seed", default=site_screen.DEFAULT_SEED, help="Calibration seed; the same seed gives the same batch")
    for name, text in (("run", "Screen each site of --input that has no run yet"),
                       ("contact", "Find a contact route for each outreach-ready site")):
        spender = commands.add_parser(name, help=text + "; a dry run unless --apply")
        if name == "run":
            spender.add_argument("--input", required=True, type=Path)
            spender.add_argument("--batch-size", type=int, help="Screen this batch, at most --max-runs (see plan)")
            spender.add_argument("--seed", default=site_screen.DEFAULT_SEED)
        spender.add_argument("--out", required=True, type=Path,
                             help="Private durable directory outside every repository and outside /tmp")
        spender.add_argument("--owner-reference", required=True, help="The owner decision the first --apply pins")
        spender.add_argument("--ceiling-usd", required=True,
                             help="Spend ceiling for all stages in this out dir; never above the pin")
        spender.add_argument("--max-runs", required=True, type=int,
                             help="Run limit for all stages in this out dir; never above the pin")
        spender.add_argument("--processor", default=site_screen.DEFAULT_PROCESSOR)
        spender.add_argument("--key-file", type=Path, help="KEY=VALUE file holding PARALLEL_API_KEY")
        spender.add_argument("--apply", action="store_true")
    collector = commands.add_parser("collect", help="Store each run's terminal result")
    collector.add_argument("--out", required=True, type=Path)
    collector.add_argument("--wait-seconds", type=int, default=site_screen.COLLECT_WAIT_SECONDS)
    collector.add_argument("--key-file", type=Path)
    for name, text in (("verify", "Check every quote against its cited page"),
                       ("summary", "Counts by field, check, tier and recipient, and cost")):
        commands.add_parser(name, help=text).add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    def client():
        key = api_key(args.key_file, environ)
        return site_screen.TaskClient(key, **({"transport": transport} if transport is not None else {}))

    if args.command in ("run", "contact"):
        site_screen.refuse_on_worker()  # Before any file is made or the key is read.
    if args.command == "plan":
        result = site_screen.plan(site_screen.read_input(args.input), processor=args.processor,
                                  batch_size=args.batch_size, seed=args.seed)
    elif args.command == "run":
        # The out dir guard runs before the key is read; the out dir is made only once both pass.
        site_screen.guard_out_dir(args.out)
        task_client, raw = client(), site_screen.read_input(args.input)
        result = site_screen.run(raw, site_screen.Workspace(args.out, create=True), client=task_client,
                                 owner_reference=args.owner_reference, ceiling_usd=args.ceiling_usd,
                                 max_runs=args.max_runs, processor=args.processor, apply=args.apply,
                                 batch_size=args.batch_size, seed=args.seed)
    elif args.command == "contact":
        workspace = site_screen.Workspace(args.out)
        result = site_screen.contact(workspace, client=client(), owner_reference=args.owner_reference,
                                     ceiling_usd=args.ceiling_usd, max_runs=args.max_runs, processor=args.processor,
                                     apply=args.apply)
    elif args.command == "collect":
        workspace = site_screen.Workspace(args.out)
        result = site_screen.collect(workspace, client=client(), reader=reader, today=today,
                                     wait_seconds=args.wait_seconds, monotonic=monotonic, sleep=sleep)
    elif args.command == "verify":
        result = site_screen.verify(site_screen.Workspace(args.out), reader=reader, today=today)
    else:
        result = site_screen.summary(site_screen.Workspace(args.out))
    print(site_screen.canonical(result))
    return result


if __name__ == "__main__":
    try:
        outcome = main()
    except Exception as error:  # noqa: BLE001 - stable codes only; never provider, key or site text
        code = str(error) if isinstance(error, ScreenError) else "site_screen_operation_unavailable"
        print(site_screen.canonical({"state": "blocked", "error": code}))
        raise SystemExit(1) from None
    raise SystemExit(0 if outcome.get("state") in OK_STATES else 1)
