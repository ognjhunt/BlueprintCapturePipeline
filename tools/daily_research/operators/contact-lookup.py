"""Owner command for the contact lookup (FullEnrich) of a site-screen out dir.

lookup is a dry run unless --apply. With --apply it pins the owner reference, a FullEnrich credit ceiling and a call
limit on first use (a later run may only lower them), journals every call before sending it, starts one work-email
enrichment per named, current person the contact stage proved by a quote, and reads the results (reads are not billed)
for up to --wait-seconds; a later run reads the rest and never sends a lookup twice. A people search on the operator's
domain for the deciding roles runs only with --person-search naming the owner decision
owner-decision-provider-sourced-person-20261005; its person is labelled provider_sourced. summary recomputes every
record from the journal. The key comes only from --key-file (FULLENRICH_API_KEY=... lines); it is never printed or
written. Output is counts and stable codes only, never names or addresses. Nothing sends, drafts or writes a CRM.
"""
import argparse
import time
from pathlib import Path

from tools.daily_research import contact_lookup, site_screen
from tools.daily_research.site_screen import ScreenError

OK_STATES = frozenset({"planned", "complete", "pending"})


def api_key(key_file):
    """FULLENRICH_API_KEY from a KEY=VALUE file: ``export`` and quotes are allowed and # lines are comments."""
    try:
        lines = Path(key_file).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        raise ScreenError("contact_lookup_key_file_unreadable") from None
    value = ""
    for line in lines:
        line = line.strip()
        name, separator, item = line.removeprefix("export ").partition("=")
        if separator and not line.startswith("#") and name.strip() == contact_lookup.KEY_ENV:
            value = item.strip().strip("\"'")
    if not value:
        raise ScreenError("contact_lookup_api_key_missing")
    return value


def main(argv=None, *, environ=None, transport=None, monotonic=time.monotonic, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    spender = commands.add_parser("lookup", help="Look up work emails; a dry run unless --apply")
    spender.add_argument("--out", required=True, type=Path, help="The site-screen out dir (durable and private)")
    spender.add_argument("--owner-reference", required=True, help="The owner decision the first --apply pins")
    spender.add_argument("--max-credits", required=True, help="FullEnrich credit ceiling for this out dir; never above the pin")
    spender.add_argument("--max-calls", required=True, type=int, help="Paid call limit for this out dir; never above the pin")
    spender.add_argument("--key-file", required=True, type=Path, help="KEY=VALUE file holding FULLENRICH_API_KEY")
    spender.add_argument("--person-search", metavar="OWNER_DECISION",
                         help="Allow a provider_sourced person; must be " + contact_lookup.PERSON_SEARCH_DECISION)
    spender.add_argument("--wait-seconds", type=int, default=contact_lookup.WAIT_SECONDS,
                         help="How long to read started enrichments (reads are not billed)")
    spender.add_argument("--apply", action="store_true")
    commands.add_parser("summary", help="Counts only, recomputed from the journal").add_argument(
        "--out", required=True, type=Path)
    commands.add_parser("balance", help="Read remaining credits only; no billed call or artifact write").add_argument(
        "--key-file", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "balance":
        contact_lookup.refuse_on_worker(environ)
        client = contact_lookup.FullEnrichClient(api_key(args.key_file),
                                                 **({"transport": transport} if transport is not None else {}))
        result = client.balance()
    elif args.command == "lookup":
        contact_lookup.refuse_on_worker(environ)  # Before any file is read or the key is read.
        contact_lookup.person_search_on(args.person_search)
        workspace = site_screen.Workspace(args.out)  # The out dir guard runs before the key is read.
        client = contact_lookup.FullEnrichClient(api_key(args.key_file),
                                                 **({"transport": transport} if transport is not None else {}))
        result = contact_lookup.lookup(workspace, client=client, owner_reference=args.owner_reference,
                                       max_credits=args.max_credits, max_calls=args.max_calls,
                                       person_search=args.person_search, apply=args.apply,
                                       wait_seconds=args.wait_seconds, monotonic=monotonic, sleep=sleep,
                                       environ=environ)
    else:
        result = contact_lookup.summary(site_screen.Workspace(args.out))
    print(site_screen.canonical(result))
    return result


if __name__ == "__main__":
    try:
        outcome = main()
    except Exception as error:  # noqa: BLE001 - stable codes only; never provider, key, person or site text
        code = str(error) if isinstance(error, ScreenError) else "contact_lookup_operation_unavailable"
        print(site_screen.canonical({"state": "blocked", "error": code}))
        raise SystemExit(1) from None
    raise SystemExit(0 if outcome.get("state") in OK_STATES else 1)
