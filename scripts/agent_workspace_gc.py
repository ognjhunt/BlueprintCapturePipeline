#!/usr/bin/env python3
"""Reap stale agent-session scratch directories (dry-run by default).

Claude Code, Codex, Paperclip, and superpowers sessions leave full repo
clones and scratch checkouts under ``~/workspace``, ``/private/tmp``, and
``~/.config/superpowers/worktrees`` and never delete them. The 2026-08-02
disk audit measured ~40 GB of such clones created in six days, which is what
kept filling the disk. This tool deletes only directories that are provably
safe to lose and reports everything else with the reason it was kept.

Safety contract (fail-closed):

- Primary checkouts are allowlisted by name and never touched.
- Git directories with tracked modifications, stashes, unpushed commits, no
  remote, or any git error are kept.
- Directories containing any file modified within ``--age-days`` are kept.
- Names matching evidence/inputs/dataset patterns are kept unless
  ``--include-evidence`` is passed.
- Non-git data directories under workspace-kind roots are never deleted
  automatically (reported for manual review); under tmp-kind roots they are
  eligible by age alone.
- Deletion requires BOTH ``--apply`` and ``--ack reap-agent-scratch``.

Scope boundaries: repo ``output/`` and ``robot_eval_jobs/`` artifacts are
governed by ``scripts/manage_output_artifact_retention.py``; ``~/.claude`` is
bounded by Claude Code's built-in ``cleanupPeriodDays`` cleanup. ``--codex``
additionally prunes old data files (never ``*.sqlite*``) under
``~/.codex/{sessions,archived_sessions,visualizations,generated_images}``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat as stat_mod
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = "agent_workspace_gc_manifest.v1"
EXECUTE_ACK = "reap-agent-scratch"
DEFAULT_AGE_DAYS = 7
DEFAULT_CODEX_AGE_DAYS = 45

DEFAULT_KEEP_NAMES = frozenset(
    {
        "BlueprintCapturePipeline",
        "Blueprint-WebApp",
        "BlueprintCapture",
        "BlueprintPipeline",
        "BlueprintValidation",
        "paperclip",
        ".paperclip-blueprint",
    }
)

# Session scratchpad roots managed by the agent harnesses themselves.
HARNESS_SCRATCH_RE = re.compile(r"^(claude|codex)-\d+$")
EVIDENCE_NAME_RE = re.compile(r"(?i)(evidence|inputs|dataset)")
CODEX_PRUNE_SUBDIRS = ("sessions", "archived_sessions", "visualizations", "generated_images")

GIT_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class RootSpec:
    path: Path
    kind: str  # "workspace" | "tmp"
    depth: int = 1


@dataclass
class Entry:
    path: str
    action: str  # "reap" | "keep"
    reason: str
    size_bytes: int | None = None
    newest_mtime: float | None = None
    deleted: bool = False
    delete_error: str | None = None


@dataclass
class CodexPruneResult:
    subdir: str
    files: int = 0
    bytes: int = 0
    errors: list[str] = field(default_factory=list)


def default_roots() -> list[RootSpec]:
    home = Path.home()
    roots = [
        RootSpec(home / "workspace", "workspace", 1),
        RootSpec(Path("/private/tmp"), "tmp", 1),
        RootSpec(home / ".config" / "superpowers" / "worktrees", "workspace", 2),
    ]
    return [r for r in roots if r.path.is_dir()]


def _run_git(repo: Path, *args: str) -> tuple[bool, str]:
    """Run git in ``repo``; return (ok, stdout). Any failure -> (False, msg)."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"git {' '.join(args)}: {exc}"
    if proc.returncode != 0:
        return False, f"git {' '.join(args)}: rc={proc.returncode} {proc.stderr.strip()[:200]}"
    return True, proc.stdout


def classify_git_dir(path: Path) -> tuple[bool, str]:
    """Return (eligible, reason). Fail closed: any doubt -> not eligible."""
    ok, out = _run_git(path, "status", "--porcelain", "--untracked-files=no")
    if not ok:
        return False, f"git-error ({out})"
    if out.strip():
        return False, "tracked modifications present"
    ok, out = _run_git(path, "stash", "list")
    if not ok:
        return False, f"git-error ({out})"
    if out.strip():
        return False, "stash entries present"
    ok, out = _run_git(path, "remote")
    if not ok:
        return False, f"git-error ({out})"
    if not out.strip():
        return False, "no remote configured (cannot prove pushed)"
    ok, out = _run_git(path, "rev-list", "--branches", "--not", "--remotes", "--max-count=1")
    if not ok:
        return False, f"git-error ({out})"
    if out.strip():
        return False, "unpushed branch commits"
    ok, out = _run_git(path, "rev-list", "HEAD", "--not", "--remotes", "--max-count=1")
    if not ok:
        return False, f"git-error ({out})"
    if out.strip():
        return False, "HEAD not reachable from any remote ref"
    return True, "clean, pushed"


def scan_tree(path: Path, cutoff: float) -> tuple[bool, int, float]:
    """Walk ``path`` without following symlinks.

    Returns (recent, total_bytes, newest_mtime). Stops early once a file or
    directory newer than ``cutoff`` is seen; unreadable entries count as
    recent so errors fail closed toward keeping the directory.
    """
    total = 0
    newest = 0.0
    try:
        root_st = path.lstat()
    except OSError:
        return True, 0, time.time()
    newest = max(newest, root_st.st_mtime)
    if root_st.st_mtime >= cutoff:
        return True, total, newest
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for dirent in it:
                    try:
                        st = dirent.stat(follow_symlinks=False)
                    except OSError:
                        return True, total, time.time()
                    newest = max(newest, st.st_mtime)
                    if st.st_mtime >= cutoff:
                        return True, total, newest
                    if stat_mod.S_ISREG(st.st_mode):
                        total += st.st_size
                    elif stat_mod.S_ISDIR(st.st_mode):
                        stack.append(Path(dirent.path))
        except OSError:
            return True, total, time.time()
    return False, total, newest


def _candidate_dirs(root: RootSpec) -> list[Path]:
    levels = [root.path]
    for _ in range(root.depth):
        next_level: list[Path] = []
        for parent in levels:
            try:
                with os.scandir(parent) as it:
                    for dirent in it:
                        if dirent.is_dir(follow_symlinks=False):
                            next_level.append(Path(dirent.path))
            except OSError:
                continue
        levels = next_level
    return sorted(levels)


def _contained(candidate: Path, root: RootSpec) -> bool:
    try:
        resolved = candidate.resolve(strict=True)
        root_resolved = root.path.resolve(strict=True)
    except OSError:
        return False
    rel_parent = candidate.parents[root.depth - 1] if root.depth >= 1 else candidate
    return rel_parent == root.path and str(resolved).startswith(str(root_resolved) + os.sep)


def classify_entry(
    candidate: Path,
    root: RootSpec,
    cutoff: float,
    keep_names: frozenset[str],
    include_evidence: bool,
) -> Entry:
    name = candidate.name
    path_str = str(candidate)
    if candidate.is_symlink():
        return Entry(path_str, "keep", "symlink (never followed)")
    try:
        st = candidate.lstat()
    except OSError as exc:
        return Entry(path_str, "keep", f"stat failed ({exc})")
    if st.st_uid != os.geteuid():
        return Entry(path_str, "keep", "not owned by current user")
    if name in keep_names:
        return Entry(path_str, "keep", "primary checkout (allowlist)")
    if HARNESS_SCRATCH_RE.match(name):
        return Entry(path_str, "keep", "harness-managed scratchpad root")
    if EVIDENCE_NAME_RE.search(name) and not include_evidence:
        return Entry(path_str, "keep", "evidence/inputs/dataset name (use --include-evidence)")
    cwd = os.getcwd()
    if cwd == path_str or cwd.startswith(path_str + os.sep):
        return Entry(path_str, "keep", "current working directory")
    recent, size_bytes, newest = scan_tree(candidate, cutoff)
    if recent:
        return Entry(path_str, "keep", "modified within age window", None, newest)
    if (candidate / ".git").exists():
        eligible, reason = classify_git_dir(candidate)
        if not eligible:
            return Entry(path_str, "keep", reason, size_bytes, newest)
        return Entry(path_str, "reap", f"idle git scratch clone ({reason})", size_bytes, newest)
    if root.kind == "tmp":
        return Entry(path_str, "reap", "idle non-git tmp scratch dir", size_bytes, newest)
    return Entry(
        path_str,
        "keep",
        "non-git data dir outside tmp (manual review; never auto-deleted)",
        size_bytes,
        newest,
    )


def _rmtree(path: Path) -> None:
    def _onerror(func, p, _exc_info):  # pragma: no cover - exercised via chmod path
        try:
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            raise

    shutil.rmtree(path, onerror=_onerror)


def prune_codex(codex_root: Path, cutoff: float, apply: bool) -> list[CodexPruneResult]:
    results: list[CodexPruneResult] = []
    for subdir in CODEX_PRUNE_SUBDIRS:
        base = codex_root / subdir
        result = CodexPruneResult(subdir=str(base))
        if not base.is_dir():
            results.append(result)
            continue
        doomed: list[Path] = []
        for dirpath, _dirnames, filenames in os.walk(base, followlinks=False):
            for filename in filenames:
                if ".sqlite" in filename:
                    continue
                file_path = Path(dirpath) / filename
                try:
                    st = file_path.lstat()
                except OSError:
                    continue
                if not stat_mod.S_ISREG(st.st_mode):
                    continue
                if st.st_mtime < cutoff:
                    doomed.append(file_path)
                    result.files += 1
                    result.bytes += st.st_size
        if apply:
            for file_path in doomed:
                try:
                    file_path.unlink()
                except OSError as exc:
                    result.errors.append(f"{file_path}: {exc}")
            for dirpath, dirnames, filenames in os.walk(base, topdown=False, followlinks=False):
                if Path(dirpath) == base or dirnames or filenames:
                    continue
                try:
                    os.rmdir(dirpath)
                except OSError:
                    pass
        results.append(result)
    return results


def _human(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "      ?"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:7.1f} {unit}"
        value /= 1024
    return f"{value:7.1f} TiB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--age-days", type=float, default=DEFAULT_AGE_DAYS)
    parser.add_argument(
        "--workspace-root",
        action="append",
        default=None,
        help="Workspace-kind root to scan (replaces defaults when given).",
    )
    parser.add_argument(
        "--tmp-root",
        action="append",
        default=None,
        help="Tmp-kind root to scan (replaces defaults when given).",
    )
    parser.add_argument("--keep", action="append", default=[], help="Extra allowlisted dir name.")
    parser.add_argument("--include-evidence", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Delete eligible dirs (needs --ack).")
    parser.add_argument("--ack", default=None, help=f"Must equal '{EXECUTE_ACK}' with --apply.")
    parser.add_argument("--codex", action="store_true", help="Also prune old ~/.codex data files.")
    parser.add_argument("--codex-age-days", type=float, default=DEFAULT_CODEX_AGE_DAYS)
    parser.add_argument("--codex-root", default=str(Path.home() / ".codex"))
    parser.add_argument("--json", dest="json_path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.apply and args.ack != EXECUTE_ACK:
        print(
            f"refusing to delete: --apply requires --ack {EXECUTE_ACK}",
            file=sys.stderr,
        )
        return 2

    if args.workspace_root is None and args.tmp_root is None:
        roots = default_roots()
    else:
        roots = [RootSpec(Path(p).expanduser(), "workspace", 1) for p in args.workspace_root or []]
        roots += [RootSpec(Path(p).expanduser(), "tmp", 1) for p in args.tmp_root or []]

    keep_names = frozenset(set(DEFAULT_KEEP_NAMES) | set(args.keep))
    now = time.time()
    cutoff = now - args.age_days * 86400.0
    mode = "APPLY" if args.apply else "dry-run"
    print(f"agent-workspace-gc {mode}  age-days={args.age_days:g}  ack={EXECUTE_ACK!r} required to delete")

    entries: list[Entry] = []
    for root in roots:
        if not root.path.is_dir():
            continue
        print(f"\nROOT {root.path} (kind={root.kind}, depth={root.depth})")
        for candidate in _candidate_dirs(root):
            entry = classify_entry(candidate, root, cutoff, keep_names, args.include_evidence)
            if entry.action == "reap" and not _contained(candidate, root):
                entry = Entry(entry.path, "keep", "containment check failed")
            if entry.action == "reap" and args.apply:
                try:
                    _rmtree(candidate)
                    entry.deleted = True
                except OSError as exc:
                    entry.delete_error = str(exc)
            tag = "REAP" if entry.action == "reap" else "keep"
            detail = entry.reason + (" [DELETED]" if entry.deleted else "")
            if entry.delete_error:
                detail += f" [DELETE FAILED: {entry.delete_error}]"
            print(f"  {tag}  {_human(entry.size_bytes)}  {Path(entry.path).name}  ({detail})")
            entries.append(entry)

    reaped = [e for e in entries if e.action == "reap"]
    reap_bytes = sum(e.size_bytes or 0 for e in reaped)
    print(f"\neligible: {len(reaped)} dirs, {_human(reap_bytes).strip()}")
    if not args.apply:
        print(f"dry-run only; rerun with --apply --ack {EXECUTE_ACK} to delete")

    codex_results: list[CodexPruneResult] = []
    if args.codex:
        codex_cutoff = now - args.codex_age_days * 86400.0
        codex_results = prune_codex(Path(args.codex_root).expanduser(), codex_cutoff, args.apply)
        print(f"\ncodex prune (>{args.codex_age_days:g}d, never *.sqlite*):")
        for result in codex_results:
            print(f"  {_human(result.bytes)}  {result.files:6d} files  {result.subdir}")
            for err in result.errors:
                print(f"    ERROR {err}")

    if args.json_path:
        manifest = {
            "schema": SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "mode": mode,
            "age_days": args.age_days,
            "roots": [{"path": str(r.path), "kind": r.kind, "depth": r.depth} for r in roots],
            "entries": [vars(e) for e in entries],
            "eligible_bytes": reap_bytes,
            "codex": [vars(r) for r in codex_results],
        }
        Path(args.json_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nmanifest written: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
