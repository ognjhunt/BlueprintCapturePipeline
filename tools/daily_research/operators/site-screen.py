"""Owner command for the per-site research line (site screen) and its contact stage.

plan reads only the input file. run and contact are dry runs unless --apply; with --apply they create
Parallel Task runs, each admitted first over both stages against the owner ceiling the first --apply
pins, the spend journal and the ledgers. collect reads run status and results (reads are not billed),
verify reads the cited public pages and summary reads the out dir. The out dir must be on durable storage
outside every repository and outside /tmp. run and contact refuse on the daily worker. Output is counts
and stable codes only, never site names, people or addresses. The key comes from PARALLEL_API_KEY, or
from --key-file (KEY=VALUE lines); it is never printed or written. None of these writes a CRM, drafts or sends.

The host-owned admission (tools/daily_research/screen_admission.py) moves outreach-ready records into the CRM as
Hypothesis rows and on to WebApp drafting; nothing sends. admit runs on the owner's machine: a dry run unless
--apply, which keeps the bundle in the out dir, uploads it create-only with the owner's existing gcloud login and
reads it back. admission-show, admission-pin and admission-disable run in the Render worker shell with the existing
worker identity: show only reads; pin and disable are dry runs unless --apply and write only control.screen_admission
through the fenced compare-and-swap. The worker writes the rows itself, only while the daily work is idle.
"""
import argparse
import re
import time
from pathlib import Path

from tools.daily_research import screen_admission, site_screen
from tools.daily_research.runner import Refusal
from tools.daily_research.site_screen import ScreenError

OK_STATES = frozenset({"planned", "complete", "pending", "uploaded", "pinned", "disabled", "already_disabled", "unset",
                       "enabled"})
ADMISSION_COMMANDS = frozenset({"admission-show", "admission-pin", "admission-disable"})


api_key = site_screen.read_api_key  # PARALLEL_API_KEY from --key-file, else from the environment.


def main(argv=None, *, environ=None, transport=None, reader=None, monotonic=time.monotonic, sleep=time.sleep,
         today=None, bridge_factory=None, objects=None, clock=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="Count sites, refusals and cost in one input file")
    planner.add_argument("--input", required=True, type=Path)
    planner.add_argument("--processor", default=site_screen.DEFAULT_PROCESSOR)
    planner.add_argument("--batch-size", type=int, help="Plan this batch: rank order plus about one calibration site in ten")
    planner.add_argument("--seed", default=site_screen.DEFAULT_SEED, help="Calibration seed; the same seed gives the same batch")
    planner.add_argument("--task-focus", help="Aim every site's question at this capability (rows must include it)")
    for name, text in (("run", "Screen each site of --input that has no run yet"),
                       ("contact", "Find a contact route for each outreach-ready site")):
        spender = commands.add_parser(name, help=text + "; a dry run unless --apply")
        if name == "run":
            spender.add_argument("--input", required=True, type=Path)
            spender.add_argument("--batch-size", type=int, help="Screen this batch, at most --max-runs (see plan)")
            spender.add_argument("--seed", default=site_screen.DEFAULT_SEED)
            spender.add_argument("--task-focus", help="Aim every site's question at this capability (rows must include it)")
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
    admitter = commands.add_parser("admit", help="Build an admission bundle of outreach-ready sites; upload it with --apply")
    admitter.add_argument("--out", required=True, type=Path)
    admitter.add_argument("--direction-sha256", required=True, help="control.outreach_ready sha256 (outreach-ready-direction.py show)")
    admitter.add_argument("--direction-generation", required=True, help="Its object generation (the same show)")
    admitter.add_argument("--per-focus", type=int, default=screen_admission.DEFAULT_PER_FOCUS,
                          help="At most this many sites per task family (default 2)")
    admitter.add_argument("--keys", help="Exactly these comma-separated site keys instead of --per-focus")
    admitter.add_argument("--max-records", type=int, default=screen_admission.MAX_RECORDS)
    admitter.add_argument("--apply", action="store_true")
    commands.add_parser("admission-show", help="The pinned admission, its object, direction and state (worker shell)")
    pinner = commands.add_parser("admission-pin", help="Pin one uploaded bundle (worker shell); a dry run unless --apply")
    pinner.add_argument("--admission-id", required=True)
    pinner.add_argument("--generation", required=True)
    pinner.add_argument("--approval-reference", required=True, help="The owner decision record")
    pinner.add_argument("--expect-current", help="Current admission id from admission-show, or none")
    pinner.add_argument("--supersede-uncertain", action="store_true",
                        help="Replace a pinned admission whose claimed write was never read back")
    pinner.add_argument("--during-active-run", action="store_true")
    pinner.add_argument("--apply", action="store_true")
    commands.add_parser("admission-disable", help="The brake (worker shell); a dry run unless --apply").add_argument(
        "--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command in ADMISSION_COMMANDS:
        return admission_command(args, bridge_factory=bridge_factory, clock=clock, sleep=sleep, monotonic=monotonic)

    def client():
        key = api_key(args.key_file, environ)
        return site_screen.TaskClient(key, **({"transport": transport} if transport is not None else {}))

    if args.command in ("run", "contact"):
        site_screen.refuse_on_worker()  # Before any file is made or the key is read.
    if args.command == "plan":
        result = site_screen.plan(site_screen.read_input(args.input), processor=args.processor,
                                  batch_size=args.batch_size, seed=args.seed, focus=args.task_focus)
    elif args.command == "run":
        # The out dir guard runs before the key is read; the out dir is made only once both pass.
        site_screen.guard_out_dir(args.out)
        task_client, raw = client(), site_screen.read_input(args.input)
        result = site_screen.run(raw, site_screen.Workspace(args.out, create=True), client=task_client,
                                 owner_reference=args.owner_reference, ceiling_usd=args.ceiling_usd,
                                 max_runs=args.max_runs, processor=args.processor, apply=args.apply,
                                 batch_size=args.batch_size, seed=args.seed, focus=args.task_focus)
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
    elif args.command == "admit":
        keys = screen_admission.parse_keys(args.keys) if args.keys is not None else None
        result = screen_admission.admit(site_screen.Workspace(args.out), direction_sha256=args.direction_sha256,
                                        direction_generation=args.direction_generation, per_focus=args.per_focus,
                                        keys=keys, max_records=args.max_records, apply=args.apply, objects=objects)
    else:
        result = site_screen.summary(site_screen.Workspace(args.out))
    print(site_screen.canonical(result))
    return result


def admission_command(args, *, bridge_factory=None, clock=None, sleep=time.sleep, monotonic=time.monotonic):
    """The worker-shell admission commands, through the existing worker identity (the bridge)."""
    from datetime import datetime, timezone

    # The worker's bridge, only for these commands.
    from tools.daily_research.firestore import Bridge

    now = (clock or (lambda: datetime.now(timezone.utc)))()
    bridge = (bridge_factory or Bridge)()
    try:
        if args.command == "admission-show":
            result = screen_admission.show(bridge, now=now)
        elif args.command == "admission-pin":
            result = screen_admission.pin(bridge, admission_id=args.admission_id, generation=args.generation,
                                          approval_reference=args.approval_reference, expect=args.expect_current,
                                          supersede_uncertain=args.supersede_uncertain, apply=args.apply,
                                          during_active_run=args.during_active_run, now=now, sleep=sleep,
                                          monotonic=monotonic)
        else:
            result = screen_admission.disable(bridge, apply=args.apply, sleep=sleep, monotonic=monotonic)
        print(site_screen.canonical(result))
        return result
    finally:
        bridge.close()


def error_code(error):
    """A stable code for the command output: a site-screen or admission code, or a bridge refusal code."""
    code = str(error)
    if isinstance(error, (ScreenError, Refusal)) and re.fullmatch(r"[a-z][a-z_]{2,99}", code):
        return code
    return "site_screen_operation_unavailable"


if __name__ == "__main__":
    try:
        outcome = main()
    except Exception as error:  # noqa: BLE001 - stable codes only; never provider, key or site text
        code = error_code(error)
        print(site_screen.canonical({"state": "blocked", "error": code}))
        raise SystemExit(1) from None
    raise SystemExit(0 if outcome.get("state") in OK_STATES else 1)
