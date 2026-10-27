"""Keep a service account's Git checkout its own when root-run deploy tools use Git in it.

The control-plane deploy tool runs as root, and it fetched and checked out the service account's
source clone as root. Every deploy left root-owned object folders, refs and working files behind.
On 2026-10-27, 1,665 of 1,887 entries under /opt/blueprint/BlueprintCapturePipeline/.git and
3,631 tracked files belonged to root, so the deploy wrapper's `sudo -u blueprint git fetch` failed
before changing anything, and the workaround (fetching as root) added more.

Git in such a checkout now runs as the checkout's owner, and what root left in its Git data and
tracked files is given back first. A checkout root owns, or a process that is not root, is left
exactly as it was.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 - fixed git argv over a validated checkout
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def checkout_owner(repo: str | Path) -> tuple[int, int] | None:
    """The checkout's owner and group when this process is root and they are not; otherwise None."""
    if os.geteuid() != 0:
        return None
    root = Path(repo)
    try:
        metadata = (root / ".git").stat() if (root / ".git").exists() else root.stat()
    except OSError:
        return None
    return None if metadata.st_uid == 0 else (metadata.st_uid, metadata.st_gid)


def git_run_options(repo: str | Path) -> dict[str, Any]:
    """``subprocess.run`` options that run Git as the checkout's owner, as ``sudo -u <owner>`` does."""
    owner = checkout_owner(repo)
    if owner is None:
        return {}
    import pwd

    try:
        account = pwd.getpwuid(owner[0])
    except KeyError:
        return {"user": owner[0], "group": owner[1], "extra_groups": []}
    return {"user": owner[0], "group": owner[1], "extra_groups": os.getgrouplist(account.pw_name, owner[1]),
            "env": {**os.environ, "HOME": account.pw_dir, "USER": account.pw_name, "LOGNAME": account.pw_name}}


def repair_checkout_ownership(repo: str | Path, *, owner_uid: int, owner_gid: int,
                              chown: Callable[[str, int, int], None] = os.lchown) -> dict[str, Any]:
    """Give the checkout's Git data, its tracked files and their folders back to its owner.

    Untracked and ignored paths, such as a virtual environment, are left alone.
    """
    root = Path(repo)
    listed = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["git", "-c", f"safe.directory={root.resolve()}", "-C", str(root), "ls-files", "-z"],
        capture_output=True, check=False)
    if listed.returncode != 0:
        raise ValueError("git_checkout_ownership_tracked_files_unlisted")
    paths: set[Path] = set()
    for directory, folders, files in os.walk(root / ".git"):
        paths.add(Path(directory))
        paths.update(Path(directory) / name for name in (*folders, *files))
    for raw in listed.stdout.split(b"\0"):
        if raw:
            path = root / os.fsdecode(raw)
            paths.add(path)
            paths.update(parent for parent in path.parents if root in parent.parents)
    repaired = 0
    for path in sorted(paths):
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        if (metadata.st_uid, metadata.st_gid) != (owner_uid, owner_gid):
            chown(str(path), owner_uid, owner_gid)
            repaired += 1
    return {"repaired_count": repaired, "checked_count": len(paths), "owner_uid": owner_uid, "owner_gid": owner_gid}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Give a service checkout back what root-run Git left in it.")
    parser.add_argument("command", choices=["repair"])
    parser.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args(argv)
    owner = checkout_owner(args.repo)
    if owner is None:
        print(json.dumps({"status": "not_needed"}))
        return 0
    print(json.dumps({"status": "repaired", **repair_checkout_ownership(args.repo, owner_uid=owner[0],
                                                                        owner_gid=owner[1])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
