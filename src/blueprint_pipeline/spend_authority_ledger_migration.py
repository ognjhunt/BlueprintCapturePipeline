"""Adopt a spend-authority ledger left behind when its root moved.

PR #453 made the consumption ledger's location configurable so a hardened host
could place it inside its unit's ``ReadWritePaths`` instead of an unwritable
``$HOME``. Binding that root on a host that had already been running moves the
ledger; the records written under the previous root do not follow. The new root
reads empty, so every authorization already spent there looks unspent again --
and the ledger's single job is to make one signed authorization fund exactly one
provider allocation.

This was observed in production: two consumption records from real paid runs
stayed at the previous root while the newly bound root reported zero. Nothing
alerted, because an empty ledger and a ledger with no matching record are
indistinguishable at the point of use.

So an unadopted legacy ledger is treated as a blocker, not as an absence.
Reconciliation runs before the service accepts work: it copies records forward,
refuses on anything it cannot prove it adopted, and never deletes the legacy
tree -- a failed migration must leave the original ledger intact.

Reads and copies retained bytes only; performs no provider mutation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .spend_authority_consumption_root import (
    SPEND_AUTHORITY_ROOT_ENV,
    SpendAuthorityRootError,
    spend_authority_root,
)

#: Colon-separated absolute paths to spend-authority trees this deployment used
#: before the current one. Needed when a root moved more than once, since only
#: the deployment knows where it pointed in between.
LEGACY_ROOTS_ENV = "BLUEPRINT_SPEND_AUTHORITY_LEGACY_ROOTS"

#: The pre-#453 root name, relative to the account's home directory.
LEGACY_DIRECTORY_NAME = ".blueprint-spend-authority"

_CONSUMED_DIRECTORY_NAME = "consumed"
_AUTHORIZATIONS_DIRECTORY_NAME = "authorizations"
_LEDGER_SUBDIRECTORIES = (_CONSUMED_DIRECTORY_NAME, _AUTHORIZATIONS_DIRECTORY_NAME)

#: Depth of the search under each base. Production leaves the legacy tree one
#: level down (``/var/lib/blueprint/spend-authority-home/.blueprint-...``), and
#: an unbounded walk over a data root would be both slow and surprising.
_SEARCH_DEPTH = 3

SCHEMA_VERSION = "spend_authority_ledger_reconciliation.v1"


class SpendAuthorityLedgerError(RuntimeError):
    """A legacy ledger exists and cannot be proven adopted."""


def _resolved_root() -> Path:
    try:
        return spend_authority_root().resolve()
    except SpendAuthorityRootError as exc:
        raise SpendAuthorityLedgerError(str(exc)) from exc


def _is_ledger(path: Path) -> bool:
    try:
        return path.is_dir() and any(
            (path / name).is_dir() for name in _LEDGER_SUBDIRECTORIES
        )
    except OSError:
        # A directory we cannot even stat through is not itself a ledger. If it
        # sits inside one, the enclosing tree is still discovered and adoption
        # fails closed there, where the error names the unreadable record.
        return False


def _named_legacy_roots() -> list[Path]:
    raw = str(os.environ.get(LEGACY_ROOTS_ENV) or "")
    roots: list[Path] = []
    for item in raw.split(os.pathsep):
        candidate = item.strip()
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            raise SpendAuthorityLedgerError(
                f"legacy_ledger_root_must_be_absolute:{LEGACY_ROOTS_ENV}:{candidate}"
            )
        roots.append(path)
    return roots


def _default_search_bases(root: Path) -> list[Path]:
    # Only a deployment that bound a root has moved one, so only it gets the
    # search. Otherwise the root *is* the pre-#453 default and the base would be
    # the account's home directory -- a slow walk with nothing to find.
    if not str(os.environ.get(SPEND_AUTHORITY_ROOT_ENV) or "").strip():
        return []
    # The legacy tree is left beside the new root, which is where the installer
    # places both.
    return [root.parent] if root.parent != root else []


def discover_legacy_ledgers(
    *,
    search_bases: Sequence[Path] | None = None,
    root: Path | None = None,
) -> list[Path]:
    """Return ledger trees that are not the configured root, newest path order.

    Looks in three places, in order of how much the deployment told us: roots it
    named explicitly, the pre-#453 ``$HOME`` default, and a bounded search of
    each base -- the installer leaves the previous tree beside the new one.
    """
    resolved_root = root.resolve() if root is not None else _resolved_root()
    bases = (
        _default_search_bases(resolved_root)
        if search_bases is None
        else [Path(base) for base in search_bases]
    )

    candidates: list[Path] = list(_named_legacy_roots())

    home = str(os.environ.get("HOME") or "").strip()
    if home:
        candidates.append(Path(home).expanduser() / LEGACY_DIRECTORY_NAME)

    for base in bases:
        if not base.is_dir():
            continue
        for depth in range(1, _SEARCH_DEPTH + 1):
            pattern = "/".join(["*"] * depth)
            try:
                candidates.extend(base.glob(pattern))
            except OSError:
                # An unreadable branch of the search space is not evidence of a
                # ledger; a ledger we can see but not read is handled below.
                continue

    discovered: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved == resolved_root or resolved in seen:
            continue
        if not _is_ledger(resolved):
            continue
        seen.add(resolved)
        discovered.append(resolved)
    return sorted(discovered)


def _adopt_directory(source: Path, target: Path) -> tuple[int, int]:
    """Copy every record forward. Returns (adopted, already_present)."""
    if not source.is_dir():
        return (0, 0)
    try:
        entries = sorted(source.iterdir())
    except OSError as exc:
        raise SpendAuthorityLedgerError(f"legacy_ledger_unreadable:{source}") from exc

    adopted = 0
    already_present = 0
    for entry in entries:
        if not entry.is_file():
            continue
        try:
            payload = entry.read_bytes()
        except OSError as exc:
            raise SpendAuthorityLedgerError(f"legacy_ledger_unreadable:{entry}") from exc
        destination = target / entry.name
        if destination.exists():
            try:
                existing = destination.read_bytes()
            except OSError as exc:
                raise SpendAuthorityLedgerError(
                    f"legacy_ledger_unreadable:{destination}"
                ) from exc
            if existing != payload:
                # Two different records claim one authorization. Picking either
                # would assert a spend history we cannot substantiate.
                raise SpendAuthorityLedgerError(
                    f"legacy_ledger_record_conflict:{entry.name}"
                )
            already_present += 1
            continue
        # 0o700, because every paid lane refuses a consumption root with a
        # group or other bit set. Creating it under the process umask made
        # the reconciler and the lanes disagree about the same directory.
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Exclusive create, so a concurrent writer claiming the same
        # authorization wins rather than being silently overwritten.
        try:
            with destination.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise SpendAuthorityLedgerError(
                    f"legacy_ledger_record_conflict:{entry.name}"
                ) from None
            already_present += 1
            continue
        except OSError as exc:
            raise SpendAuthorityLedgerError(
                f"legacy_ledger_adoption_failed:{destination}"
            ) from exc
        adopted += 1
    return (adopted, already_present)


def reconcile_spend_authority_ledger(
    *,
    search_bases: Sequence[Path] | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Adopt any legacy ledger into the configured root, or fail closed.

    Copies rather than moves: a migration that fails partway must leave the
    original ledger readable, since it is the only record of what was spent.
    """
    root = _resolved_root()
    legacy_roots = discover_legacy_ledgers(search_bases=search_bases, root=root)

    adopted = 0
    already_present = 0
    for legacy in legacy_roots:
        for name in _LEDGER_SUBDIRECTORIES:
            moved, present = _adopt_directory(legacy / name, root / name)
            adopted += moved
            already_present += present

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "reconciled" if legacy_roots else "no_legacy_ledger",
        "root": str(root),
        "root_env": SPEND_AUTHORITY_ROOT_ENV,
        "legacy_roots_discovered": [str(path) for path in legacy_roots],
        "records_adopted": adopted,
        "records_already_present": already_present,
        "legacy_ledgers_retained": True,
        "blockers": [],
        "provider_mutation_performed": False,
    }
    if receipt_path is not None:
        destination = Path(receipt_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-base",
        action="append",
        default=None,
        help="Additional directory to search for a legacy ledger.",
    )
    parser.add_argument("--receipt-out")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        receipt = reconcile_spend_authority_ledger(
            search_bases=(
                [Path(base) for base in args.search_base]
                if args.search_base is not None
                else None
            ),
            receipt_path=args.receipt_out,
        )
    except SpendAuthorityLedgerError as exc:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


__all__ = [
    "LEGACY_DIRECTORY_NAME",
    "LEGACY_ROOTS_ENV",
    "SCHEMA_VERSION",
    "SpendAuthorityLedgerError",
    "discover_legacy_ledgers",
    "reconcile_spend_authority_ledger",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
