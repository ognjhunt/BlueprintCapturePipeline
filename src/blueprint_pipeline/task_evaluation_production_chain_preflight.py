"""Look ahead at the production task-evaluation chain before anything is submitted or paid for.

Four consecutive production submissions on 2026-09-05 were blocked by facts that
were knowable before the first one: an operator binding the preparation worker
reads was unset; the preparation unit's sandbox had no write path for the SAM
child queue its driver enqueues into; the SAM execution unit's sandbox had no
write path for the spend-authority ledger the paid allocator consumes; and free
disk sat below the floor every admission reservation requires.  Each cost a
redeploy-resubmit cycle to learn one fact.

This asks those questions first, for every unit in the chain, under each unit's
own sandbox and as its own user.  It rents nothing, submits nothing, and mutates
nothing beyond a probe file it creates and removes in each directory it tests.

Two roles:

``run`` (root, on the control-plane host) reads each deployed unit's effective
properties from systemd, replays the unit's sandbox in a transient unit around
the ``probe`` role, and adds the host-level chain checks: release identity,
owner intents, publisher-source and SAM bindings, hand-off configuration,
credential files, disk admission.

``probe`` (inside the sandbox, as the unit user) computes the import closure of
the unit's entry module from the release tree, resolves every production root
that closure names -- literal roots, environment-derived roots, the code's own
root resolvers -- and reports whether the unit can actually write there, plus
the environment names the closure reads that the unit leaves unset.

Findings are complete, not first-failure: the value is in learning all of it at
once.  Passing here proves the chain's plumbing binds; it proves nothing about
what a GPU run would produce.
"""

from __future__ import annotations

import argparse
import ast
import grp
import importlib
import json
import os
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
import textwrap
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "task_evaluation_production_chain_preflight.v1"
RELEASE_LINK = Path("/opt/blueprint/task-evaluation-control-plane")
RELEASES_ROOT = Path("/opt/blueprint/task-evaluation-control-plane-releases")
DEFAULT_PYTHON = "/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python"
SERVICE_ACCOUNT = "blueprint"
PACKAGE = "blueprint_pipeline"
INTAKE_CATALOG_URL = "http://127.0.0.1:8765/api/live-pipeline/task-evaluation-launch-profiles"
ACTIVATION_INTENT_ROOT = Path(  # R4: canonical producer-writable registry (see installer/unit/automation)
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-configuration-activation-intents")
CONTROLS_INTENT_ROOT = Path("/etc/blueprint/task-evaluation-configured-controls-intents")
DISK_RESERVATION_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/disk-reservations")
DISK_TARGET_ROOT = Path("/var/lib/blueprint")
GIB = 1024**3

CHAIN_UNITS: tuple[str, ...] = (
    "blueprint-pipeline-intake.service",
    "blueprint-task-evaluation-launch-preparation.service",
    "blueprint-task-evaluation-sam31-preparation-execution.service",
    "blueprint-task-evaluation-episode-compilation.service",
    "blueprint-task-evaluation-launch-activation.service",
    "blueprint-task-evaluation-configured-controls-progression.service",
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-launch-reconciler.service",
    "blueprint-task-evaluation-launch-supervisor.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-task-evaluation-terminal-resource-release.service",
    "blueprint-gpu-spend-guard.service",
    "blueprint-provider-billing-reconciler.service",
    "blueprint-control-plane-storage-gc.service",
)

# The persistent-owner authority consumers: both dispatchers and controls
# progression.  Selection scope, owner-store resolution and (for controls) the
# autoprovision config + assets are checked here so an owner-mode gap is named
# before a dispatch rather than after a wrong or empty selection.
OWNER_AUTHORITY_UNITS: tuple[str, ...] = (
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-task-evaluation-configured-controls-progression.service",
)
CONTROLS_PROGRESSION_UNIT = "blueprint-task-evaluation-configured-controls-progression.service"
# R8: the launch reconciler tick files owner terminal receipts (launch bridge +
# canary terminal set) for the scene-progression reconciler. Without these roots
# the duty is explicitly ``not_configured`` and a completed owner run never
# closes out, so an owner-mode chain names the gap here.
TERMINAL_INDEX_UNIT = "blueprint-task-evaluation-launch-reconciler.service"
TERMINAL_INDEX_ENV: tuple[str, ...] = (
    "BLUEPRINT_TASK_EVALUATION_POLICY_CANARY_DISPATCH_ROOT",
    "BLUEPRINT_TASK_EVALUATION_TERMINAL_RESULT_ROOT",
    "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT",
)

# Directives replayed onto the transient probe unit when the deployed unit (or
# one of its drop-ins) sets them.  Values come from the merged ``systemctl show``
# view so drop-ins are included.
SANDBOX_DIRECTIVES: tuple[str, ...] = (
    "User", "Group", "SupplementaryGroups", "WorkingDirectory", "UMask",
    "ProtectSystem", "ProtectHome", "PrivateTmp", "PrivateDevices", "PrivateUsers",
    "PrivateNetwork", "NoNewPrivileges", "ProtectProc", "ProcSubset",
    "RestrictAddressFamilies", "SystemCallFilter", "SystemCallArchitectures",
    "RestrictNamespaces", "LockPersonality", "MemoryDenyWriteExecute",
    "RestrictRealtime", "RestrictSUIDSGID", "ProtectKernelTunables",
    "ProtectKernelModules", "ProtectKernelLogs", "ProtectControlGroups",
    "ProtectClock", "ProtectHostname", "CapabilityBoundingSet", "ReadWritePaths",
    "ReadOnlyPaths", "InaccessiblePaths", "BindPaths", "BindReadOnlyPaths",
    "TemporaryFileSystem", "ExecPaths", "NoExecPaths",
)
ROOT_LITERAL = re.compile(r'"(/(?:var/lib|var/log|etc|run|opt|srv)/blueprint[^"\s]*)"')
ENV_NAME = re.compile(r'"((?:BLUEPRINT|VAST|PIPELINE|OPENAI|DOCKER|GOOGLE)_[A-Z0-9_]+)"')
SHA_IN_PATH = re.compile(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])")
SHA8_IN_INPUT = re.compile(r"-([0-9a-f]{8})-20[0-9]{6}")
WRITTEN_STORAGE_CLASSES = frozenset({"work", "ledger", "cache", "evidence_hot", "evidence_cold", "staging"})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _finding(severity: str, code: str, **detail: Any) -> dict[str, Any]:
    row = {"severity": severity, "code": code}
    row.update({key: value for key, value in detail.items() if value is not None})
    return row


# --------------------------------------------------------------------------- #
# Source analysis (runs inside the probe, against the release tree)
# --------------------------------------------------------------------------- #


def _module_path(src: Path, name: str) -> Path | None:
    relative = Path(*name.split("."))
    for candidate in (src / relative.with_suffix(".py"), src / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def import_closure(src: Path, entry: str, entry_path: Path | None = None) -> dict[str, Path]:
    """Every in-package module reachable from ``entry`` by any import statement.

    Lazy imports inside functions count: a root the code writes on one branch
    is still a root the unit must be able to write.
    """

    found: dict[str, Path] = {}
    pending: list[tuple[str, Path | None]] = [(entry, entry_path)]
    while pending:
        name, path = pending.pop()
        if name in found:
            continue
        path = path or _module_path(src, name)
        if path is None:
            continue
        found[name] = path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        is_package = path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == PACKAGE:
                        pending.append((alias.name, None))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if not name.startswith(PACKAGE):
                        continue
                    base = name if is_package else name.rpartition(".")[0]
                    for _ in range(node.level - 1):
                        base = base.rpartition(".")[0]
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                if target.split(".")[0] != PACKAGE:
                    continue
                pending.append((target, None))
                for alias in node.names:
                    pending.append((f"{target}.{alias.name}", None))
    return found


def literal_roots(sources: Mapping[str, str]) -> dict[str, list[str]]:
    roots: dict[str, set[str]] = {}
    for name, text in sources.items():
        for match in ROOT_LITERAL.finditer(text):
            roots.setdefault(match.group(1).rstrip("/"), set()).add(name)
    return {path: sorted(modules) for path, modules in sorted(roots.items())}


def _reads_environment(text: str, start: int) -> bool:
    """True when the quoted name at ``start`` is the argument of an environment read.

    Same line: ``os.environ.get("NAME"``, ``getenv("NAME"``, ``environ["NAME"]`` or a
    ``*_ENV = "NAME"`` constant.  A read split over two lines has only whitespace
    before the name, so the previous line is consulted then.
    """

    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start]
    if not prefix.strip():
        previous_start = text.rfind("\n", 0, max(0, line_start - 1)) + 1
        prefix = text[previous_start:line_start]
    return "environ" in prefix or "getenv" in prefix or re.search(r"_ENV\s*=\s*$", prefix.rstrip()) is not None


def environment_names_read(sources: Mapping[str, str]) -> dict[str, list[str]]:
    """Environment names the closure reads, keyed to the modules that read them."""

    names: dict[str, set[str]] = {}
    for name, text in sources.items():
        for match in ENV_NAME.finditer(text):
            if _reads_environment(text, match.start()):
                names.setdefault(match.group(1), set()).add(name)
    return {key: sorted(value) for key, value in sorted(names.items())}


# --------------------------------------------------------------------------- #
# Filesystem probes
# --------------------------------------------------------------------------- #


def _owner(st: os.stat_result) -> dict[str, Any]:
    try:
        user = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        user = str(st.st_uid)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)
    return {"owner": user, "group": group, "mode": f"{stat.S_IMODE(st.st_mode):04o}"}


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def write_probe(directory: Path) -> dict[str, Any]:
    """Create and remove one exclusive file; report the errno when that fails."""

    marker = directory / f".production-chain-preflight-{os.getpid()}"
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        status = {30: "read_only", 13: "permission_denied"}.get(exc.errno or 0, "error")
        return {"status": status, "errno": exc.errno, "strerror": exc.strerror}
    os.close(descriptor)
    try:
        os.unlink(marker)
    except OSError as exc:
        return {"status": "writable_unlink_failed", "errno": exc.errno, "strerror": exc.strerror}
    return {"status": "writable"}


def path_probe(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    try:
        st = path.lstat()
    except FileNotFoundError:
        ancestor = _existing_ancestor(path)
        verdict: dict[str, Any] = {"exists": False, "nearest_ancestor": str(ancestor)}
        if ancestor.is_dir():
            parent = write_probe(ancestor)
            verdict["status"] = (
                "missing_creatable" if parent["status"] == "writable" else f"missing_not_creatable:{parent['status']}"
            )
        else:
            verdict["status"] = "missing_not_creatable:ancestor_not_directory"
        return verdict
    except OSError as exc:
        return {"exists": None, "status": "stat_failed", "errno": exc.errno, "strerror": exc.strerror}
    verdict = {"exists": True, **_owner(st)}
    if stat.S_ISLNK(st.st_mode):
        verdict["symlink"] = True
        try:
            st = path.stat()
        except OSError as exc:
            verdict["status"] = "dangling_symlink"
            verdict["errno"] = exc.errno
            return verdict
    if stat.S_ISDIR(st.st_mode):
        verdict["kind"] = "directory"
        try:
            with os.scandir(path) as entries:
                next(entries, None)
            verdict["directory_readable"] = True
        except OSError as exc:
            verdict.update(status="unreadable", directory_readable=False, errno=exc.errno)
            return verdict
        verdict.update(write_probe(path))
        return verdict
    verdict["kind"] = "file"
    try:
        with path.open("rb") as stream:
            stream.read(1)
    except OSError as exc:
        verdict["status"] = "unreadable"
        verdict["errno"] = exc.errno
        verdict["strerror"] = exc.strerror
        return verdict
    verdict["status"] = "readable"
    return verdict


# --------------------------------------------------------------------------- #
# Probe role (inside the unit's sandbox)
# --------------------------------------------------------------------------- #


def _classify(path_text: str) -> str | None:
    try:
        module = importlib.import_module(f"{PACKAGE}.control_plane_storage_roots")
        root = module.classify_path(path_text)
    except Exception:  # noqa: BLE001 - classification is advisory
        return None
    return None if root is None else str(root.storage_class)


def _json_paths(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except ValueError:
        return []
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str) and node.startswith("/"):
            found.append(node)
        elif isinstance(node, Mapping):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(parsed)
    return found


def _severity_for_directory(
    verdict: Mapping[str, Any], *, storage_class: str | None, in_rw: bool, declared_ro: bool = False
) -> str:
    status = str(verdict.get("status") or "")
    if status == "writable" or status == "readable":
        return "ok"
    if status in {"read_only", "permission_denied", "error", "writable_unlink_failed"}:
        if declared_ro:
            # ReadOnlyPaths is the more specific declaration and systemd honours it
            # over an enclosing ReadWritePaths entry: read-only is the operator's intent.
            return "info"
        if storage_class in {"container", "release"}:
            # A container or release tree is never written by a unit; the code
            # names it only to derive children.  Root ownership there is by design.
            return "info"
        if in_rw or storage_class in WRITTEN_STORAGE_CLASSES:
            return "blocker"
        return "warning"
    if status.startswith("missing_not_creatable"):
        return "warning" if storage_class in WRITTEN_STORAGE_CLASSES or in_rw else "info"
    if status == "missing_creatable":
        return "info"
    if status in {"unreadable", "dangling_symlink", "stat_failed"}:
        return "blocker"
    return "info"


def _in_read_write_paths(path_text: str, read_write: Sequence[str]) -> bool:
    candidate = Path(path_text)
    for entry in read_write:
        root = Path(entry.lstrip("-+"))
        if candidate == root or root in candidate.parents:
            return True
    return False


ADMISSION_IDENTITY_REPLAY = textwrap.dedent(
    """
    import json, sys
    from blueprint_pipeline import paid_resource_allocator as allocator
    blockers, observed = allocator._source_checkout_blockers(sys.argv[1])
    _, identity = allocator._control_plane_checkout_blockers()
    print(json.dumps({
        "blockers": list(blockers), "observed_commit": observed,
        "release_promotion_eligible": identity.get("release_promotion_eligible"),
        "evidence_grade_ceiling": identity.get("evidence_grade_ceiling"),
    }))
    """
)


def paid_admission_identity(src: Path, release: Path, active_sha: str, *, timeout: float = 120.0) -> dict[str, Any]:
    """Replay the paid allocator's checkout-identity gate from the release, as the unit will.

    Submission #8 (2026-09-05) passed every static check here and was refused at
    paid admission with ``gpu_canary_checkout_not_remote_main``: main had moved
    after the deploy and the sandboxed release could not show ancestry.  That gate
    reads the remote and the deploy receipts at admission time, so only running it
    from the release tree, inside the unit's sandbox, before a submission predicts
    it.  A fresh interpreter keeps the replay on the release's own package.
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-c", ADMISSION_IDENTITY_REPLAY, active_sha],
            cwd=str(release), env={**os.environ, "PYTHONPATH": str(src)},
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    if completed.returncode != 0:
        return {"error": f"exit {completed.returncode}: {completed.stderr.strip()[-300:]}"}
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return {"error": f"unparseable replay output: {type(exc).__name__}"}
    return value if isinstance(value, dict) else {"error": "replay output is not an object"}


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    release = Path(args.release)
    src = release / "src"
    # Run as a script, Python puts this file's own directory (the package
    # directory) first on sys.path.  A bare-name import anywhere in the closure
    # then resolves a package module as a top-level one and its relative imports
    # fail with "no known parent package", which is a probe artefact, not a
    # unit defect.  Import the way the units do: from the release's src root.
    package_dir = Path(__file__).resolve().parent
    sys.path[:] = [
        entry for entry in sys.path
        if entry and Path(entry).resolve() != package_dir
    ]
    sys.path.insert(0, str(src))
    read_write = shlex.split(args.read_write_paths or "")
    read_only = shlex.split(args.read_only_paths or "")
    active_sha = args.active_sha or ""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        user = str(os.getuid())
    umask = os.umask(0)
    os.umask(umask)
    report: dict[str, Any] = {
        "unit": args.unit,
        "entry": args.module,
        "user": user,
        "uid": os.getuid(),
        "home": os.environ.get("HOME"),
        "umask": f"{umask:04o}",
        "python": sys.executable,
        "findings": [],
        "paths": {},
    }
    findings: list[dict[str, Any]] = report["findings"]

    entry_path = None
    if args.module.endswith(".py"):
        entry_path = release / args.module
    closure = import_closure(src, args.module, entry_path)
    report["closure_size"] = len(closure)
    sources = {}
    for name, path in closure.items():
        try:
            sources[name] = path.read_text(encoding="utf-8")
        except OSError:
            continue

    if args.module.startswith(f"{PACKAGE}."):
        try:
            importlib.import_module(args.module)
            report["import"] = "ok"
        except BaseException as exc:  # noqa: BLE001 - any import-time failure is the finding
            report["import"] = f"{type(exc).__name__}: {exc}"[:400]
            findings.append(_finding("blocker", "entry_module_import_failed", module=args.module, error=report["import"]))

    seen: dict[str, dict[str, Any]] = report["paths"]

    def examine(path_text: str, *, source: str, modules: Sequence[str] = (), git_safe_directory: bool = False) -> None:
        path_text = path_text.rstrip("/") or "/"
        row = seen.get(path_text)
        if row is None:
            verdict = path_probe(path_text)
            storage_class = _classify(path_text)
            in_rw = _in_read_write_paths(path_text, read_write)
            declared_ro = _in_read_write_paths(path_text, read_only)
            row = {**verdict, "storage_class": storage_class, "in_read_write_paths": in_rw, "declared_read_only": declared_ro, "sources": []}
            row["severity"] = _severity_for_directory(verdict, storage_class=storage_class, in_rw=in_rw, declared_ro=declared_ro)
            seen[path_text] = row
        row["sources"].append({"source": source, "modules": list(modules)} if modules else {"source": source})
        sha = SHA_IN_PATH.search(path_text)
        if active_sha and sha and sha.group(1) != active_sha and "task-evaluation-control-plane-releases" in path_text:
            if git_safe_directory:
                # A git ``safe.directory`` waiver names another release only to
                # trust that checkout for git, not because this unit's runtime is
                # bound to it.  Keep the finding visible, but as information.
                findings.append(_finding("info", "git_safe_directory_names_other_release", path=path_text, source=source, bound_sha=sha.group(1)))
            else:
                findings.append(_finding("blocker", "path_bound_to_other_release", path=path_text, source=source, bound_sha=sha.group(1)))
        sha8 = SHA8_IN_INPUT.search(path_text)
        if active_sha and sha8 and sha8.group(1) != active_sha[:8] and "/task-evaluation-inputs/" in path_text:
            # Per-release input directories are reused across releases by content
            # identity; the prefix is where the bytes were installed, not a binding.
            findings.append(_finding("info", "input_installed_under_other_release_prefix", path=path_text, source=source, installed_sha8=sha8.group(1)))

    for path_text, modules in literal_roots(sources).items():
        examine(path_text, source="code_literal", modules=modules)

    git_safe_directory_names = _git_safe_directory_value_env_names(os.environ)
    for name, value in sorted(os.environ.items()):
        git_safe_directory = name in git_safe_directory_names
        if value.startswith("/"):
            examine(value, source=f"env:{name}", git_safe_directory=git_safe_directory)
        elif value[:1] in "[{":
            for nested in _json_paths(value):
                examine(nested, source=f"env_json:{name}", git_safe_directory=git_safe_directory)

    resolvers = (
        (f"{PACKAGE}.spend_authority_consumption_root", "consumption_root", ()),
        (f"{PACKAGE}.spend_authority_consumption_root", "authorizations_root", ()),
        (f"{PACKAGE}.control_plane_storage_pins", "pins_root_from_environment", ()),
    )
    for module_name, function_name, call_args in resolvers:
        if module_name not in closure:
            continue
        try:
            module = importlib.import_module(module_name)
            resolved = getattr(module, function_name)(*call_args)
        except BaseException as exc:  # noqa: BLE001 - the resolver's refusal is the finding
            findings.append(_finding("blocker", "root_resolver_failed", resolver=f"{module_name}.{function_name}", error=f"{type(exc).__name__}: {exc}"[:300]))
            continue
        if resolved is None:
            findings.append(_finding("warning", "root_resolver_unset", resolver=f"{module_name}.{function_name}"))
            continue
        examine(str(resolved), source=f"resolver:{module_name.rsplit('.', 1)[-1]}.{function_name}", modules=[module_name])
        if str(resolved).startswith("/nonexistent") or str(resolved).startswith("/root/"):
            findings.append(_finding("blocker", "root_resolves_under_protected_home", resolver=f"{module_name}.{function_name}", path=str(resolved)))

    driver = f"{PACKAGE}.task_evaluation_scene_configuration_sam31_preparation_driver"
    if driver in closure:
        try:
            module = importlib.import_module(driver)
            examine(os.environ.get(module.CHILD_QUEUE_ENV, str(module.DEFAULT_CHILD_QUEUE)), source="resolver:sam31_driver.child_queue", modules=[driver])
        except BaseException as exc:  # noqa: BLE001
            findings.append(_finding("warning", "root_resolver_failed", resolver=f"{driver}.child_queue", error=f"{type(exc).__name__}: {exc}"[:300]))

    allocator_name = f"{PACKAGE}.paid_resource_allocator"
    if active_sha and (allocator_name in closure or any(allocator_name in text for text in sources.values())):
        admission = paid_admission_identity(src, release, active_sha)
        report["paid_admission_identity"] = admission
        if admission.get("error"):
            findings.append(_finding("warning", "paid_admission_identity_probe_failed", error=admission["error"]))
        else:
            for code in admission.get("blockers") or []:
                findings.append(_finding("blocker", "paid_admission_checkout_identity_refused", blocker=str(code), release_commit=active_sha))
            if admission.get("release_promotion_eligible") is False:
                findings.append(_finding("warning", "release_evidence_grade_development_only", release_commit=active_sha,
                                         evidence_grade_ceiling=admission.get("evidence_grade_ceiling")))

    home_users = sorted(name for name, text in sources.items() if "Path.home()" in text or "expanduser(" in text)
    if home_users:
        report["home_users"] = home_users
        home = os.environ.get("HOME") or ""
        if not home or home.startswith("/nonexistent") or not Path(home).is_dir():
            findings.append(_finding("warning", "closure_reads_home_under_protect_home", home=home or None, modules=home_users[:12]))

    unset = {name: modules for name, modules in environment_names_read(sources).items() if not str(os.environ.get(name) or "").strip()}
    report["environment_names_read_but_unset"] = unset

    for path_text, row in seen.items():
        if row["severity"] in {"blocker", "warning"}:
            findings.append(
                _finding(
                    row["severity"],
                    f"path_{row.get('status')}",
                    path=path_text,
                    storage_class=row.get("storage_class"),
                    in_read_write_paths=row.get("in_read_write_paths"),
                    owner=row.get("owner"),
                    mode=row.get("mode"),
                    errno=row.get("errno"),
                    sources=[entry["source"] for entry in row["sources"]][:6],
                    modules=sorted({m for entry in row["sources"] for m in entry.get("modules", [])})[:8] or None,
                )
            )
    return report


# --------------------------------------------------------------------------- #
# Run role (root on the host)
# --------------------------------------------------------------------------- #


def _systemctl(*args: str) -> str:
    completed = subprocess.run(["systemctl", "--no-pager", *args], capture_output=True, text=True, check=False)
    return completed.stdout


def unit_properties(unit: str) -> dict[str, list[str]]:
    props: dict[str, list[str]] = {}
    for line in _systemctl("show", unit).splitlines():
        key, sep, value = line.partition("=")
        if sep:
            props.setdefault(key, []).append(value)
    return props


def configured_directives(unit: str) -> set[str]:
    names: set[str] = set()
    for line in _systemctl("cat", unit).splitlines():
        stripped = line.strip()
        key, sep, _ = stripped.partition("=")
        if sep and key and not key.startswith("#") and not key.startswith("["):
            names.add(key)
    return names


def _first(props: Mapping[str, list[str]], key: str) -> str:
    values = props.get(key) or [""]
    return values[0]


def _git_safe_directory_value_env_names(environ: Mapping[str, str]) -> set[str]:
    """Environment names that carry a git ``safe.directory`` value.

    A unit hands git its recent release checkouts through the
    ``GIT_CONFIG_COUNT``/``GIT_CONFIG_KEY_<i>``/``GIT_CONFIG_VALUE_<i>`` protocol
    so git waives its ownership refusal in those trees.  A ``safe.directory``
    value that names another release only trusts that checkout for git; it does
    not bind the unit's runtime to that release the way a code literal or a root
    resolver would.  A path finding whose source is one of these should stay
    visible but be informational, not a binding blocker.
    """

    names: set[str] = set()
    for key, value in environ.items():
        match = re.fullmatch(r"GIT_CONFIG_KEY_(\d+)", key)
        if match and value.strip() == "safe.directory":
            names.add(f"GIT_CONFIG_VALUE_{match.group(1)}")
    return names


def entry_module(props: Mapping[str, list[str]]) -> str | None:
    for text in props.get("ExecStart", []):
        match = re.search(r"-m (blueprint_pipeline\.[a-z0-9_]+)|(scripts/[a-z0-9_]+\.py)", text)
        if match:
            return match.group(1) or match.group(2)
    return None


def environment_files(props: Mapping[str, list[str]]) -> list[tuple[str, bool]]:
    files: list[tuple[str, bool]] = []
    for text in props.get("EnvironmentFiles", []):
        match = re.match(r"(\S+) \(ignore_errors=(yes|no)\)", text.strip())
        if match:
            files.append((match.group(1), match.group(2) == "yes"))
    return files


def unit_environment(props: Mapping[str, list[str]]) -> dict[str, str]:
    for text in props.get("Environment", []):
        pairs = shlex.split(text)
        env = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if sep:
                env[key] = value
        return env
    return {}


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return env
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key.strip()] = value
    return env


def effective_environment(props: Mapping[str, list[str]]) -> dict[str, str]:
    """The unit's environment as systemd assembles it: files override Environment=."""

    env = dict(unit_environment(props))
    for path_text, _ignore in environment_files(props):
        env.update(_parse_env_file(Path(path_text)))
    return env


def systemd_run_command(
    *,
    unit: str,
    props: Mapping[str, list[str]],
    directives: set[str],
    python: str,
    script: Path,
    module: str,
    release: Path,
    active_sha: str,
) -> list[str]:
    slug = re.sub(r"[^a-z0-9]+", "-", unit.removesuffix(".service"))[:40]
    command = [
        "systemd-run", "--wait", "--pipe", "--collect", "--quiet", "--service-type=exec",
        f"--unit=production-chain-preflight-{slug}-{os.getpid()}", "-p", "TimeoutStartSec=300",
    ]
    for name in SANDBOX_DIRECTIVES:
        if name not in directives and name != "User":
            continue
        value = _first(props, name)
        if not value:
            continue
        command.extend(["-p", f"{name}={value}"])
    for path_text, ignore in environment_files(props):
        command.extend(["-p", f"EnvironmentFile={'-' if ignore else ''}{path_text}"])
    for key, value in unit_environment(props).items():
        command.append(f"--setenv={key}={value}")
    command.append(f"--setenv=BLUEPRINT_PRODUCTION_CHAIN_PREFLIGHT_UNIT={unit}")
    # systemd path directives may begin with ``-`` (an optional path that is
    # silently skipped when absent).  Passed as a separate ``--read-only-paths``
    # ``-/x`` pair, argparse reads the leading-dash value as another option and
    # fails the probe with ``expected one argument`` (this blocked the live
    # intake and gpu-spend-guard units, whose ReadOnlyPaths is optional).  The
    # ``--opt=value`` form keeps each value one argv element, so empty,
    # space-joined multi-path and space-containing values all survive intact.
    command.extend(
        [
            "--", python, str(script), "probe", "--unit", unit, "--module", module, "--release", str(release),
            "--active-sha", active_sha,
            f"--read-write-paths={_first(props, 'ReadWritePaths')}",
            f"--read-only-paths={_first(props, 'ReadOnlyPaths')}",
        ]
    )
    return command


def _service_ids(account: str) -> tuple[int, int] | None:
    try:
        record = pwd.getpwnam(account)
    except KeyError:
        return None
    return record.pw_uid, record.pw_gid


def readable_by(path: Path, uid: int, gid: int) -> bool:
    """Discretionary read access as the kernel grants it; root bypasses mode bits."""

    try:
        st = path.stat()
    except OSError:
        return False
    if uid == 0:
        return True
    mode = stat.S_IMODE(st.st_mode)
    if st.st_uid == uid:
        return bool(mode & stat.S_IRUSR)
    if st.st_gid == gid:
        return bool(mode & stat.S_IRGRP)
    return bool(mode & stat.S_IROTH)


def writable_by(path: Path, uid: int, gid: int) -> bool:
    """Discretionary write access for a non-root service account by mode bits.

    Callers pass the service uid/gid (never 0), so the root bypass in
    ``readable_by`` does not apply and this reports what the ``blueprint`` account
    can actually do to the directory under the real sandbox.
    """

    try:
        st = path.stat()
    except OSError:
        return False
    if uid == 0:
        return True
    mode = stat.S_IMODE(st.st_mode)
    if st.st_uid == uid:
        return bool(mode & stat.S_IWUSR)
    if st.st_gid == gid:
        return bool(mode & stat.S_IWGRP)
    return bool(mode & stat.S_IWOTH)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def active_release() -> tuple[Path | None, str, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    try:
        release = RELEASE_LINK.resolve(strict=True)
    except OSError:
        findings.append(_finding("blocker", "active_release_link_unresolvable", link=str(RELEASE_LINK)))
        return None, "", findings
    sha = release.name
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        findings.append(_finding("blocker", "active_release_name_not_commit", release=str(release)))
    git_pointer = release / ".git"
    head = ""
    try:
        text = git_pointer.read_text(encoding="utf-8").strip() if git_pointer.is_file() else ""
        gitdir = text.partition("gitdir:")[2].strip()
        if gitdir:
            head = (Path(gitdir) / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        head = ""
    if head and head != sha:
        findings.append(_finding("blocker", "release_worktree_identity_drift", release=str(release), head=head))
    if not head:
        findings.append(_finding("warning", "release_worktree_head_unreadable", release=str(release)))
    return release, sha, findings


def intent_checks(active_sha: str, ids: tuple[int, int]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    uid, gid = ids
    for root, label in ((ACTIVATION_INTENT_ROOT, "activation"), (CONTROLS_INTENT_ROOT, "controls")):
        if not root.is_dir():
            findings.append(_finding("blocker", f"{label}_intent_root_missing", path=str(root)))
            continue
        live = [p for p in sorted(root.glob("*.json")) if ".superseded-" not in p.name]
        at_active = []
        for path in live:
            document = _read_json(path) or {}
            bound = str(document.get("expected_production_commit") or "")
            if not readable_by(path, uid, gid):
                findings.append(_finding("blocker", f"{label}_intent_unreadable_by_service", path=str(path)))
            if bound == active_sha:
                at_active.append(path.name)
            elif bound:
                findings.append(_finding("warning", f"{label}_intent_bound_to_other_release", path=str(path), bound_sha=bound))
        findings.append(_finding("info", f"{label}_intents_at_active_release", count=len(at_active), names=at_active[:6]))
        if label == "activation" and not at_active:
            findings.append(_finding("blocker", "activation_intent_missing_for_active_release", active_sha=active_sha))
    return findings


def binding_checks(units: Mapping[str, dict[str, Any]], active_sha: str, ids: tuple[int, int]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    uid, gid = ids
    preparation = units.get("blueprint-task-evaluation-launch-preparation.service", {})
    env = preparation.get("effective_environment", {})
    raw = env.get("BLUEPRINT_TASK_EVALUATION_INSTALLED_SOURCE_BINDINGS_JSON", "")
    if not raw.strip():
        findings.append(_finding("blocker", "installed_source_bindings_unset", unit="blueprint-task-evaluation-launch-preparation.service"))
    else:
        try:
            rows = json.loads(raw)
        except ValueError:
            rows = None
            findings.append(_finding("blocker", "installed_source_bindings_invalid_json"))
        for row in rows or []:
            if not isinstance(row, Mapping):
                continue
            receipt_path = Path(str(row.get("installation_receipt_path") or ""))
            receipt = _read_json(receipt_path)
            if receipt is None:
                findings.append(_finding("blocker", "installation_receipt_unreadable", path=str(receipt_path)))
                continue
            if not readable_by(receipt_path, uid, gid):
                findings.append(_finding("blocker", "installation_receipt_unreadable_by_service", path=str(receipt_path)))
            bound = str(receipt.get("source_commit_sha") or "")
            if not re.fullmatch(r"[0-9a-f]{40}", bound):
                findings.append(_finding("blocker", "installation_receipt_commit_invalid", path=str(receipt_path), bound_sha=bound))
            elif bound != active_sha:
                # Installed sources are bound by content identity (#1653): every member is
                # digest-checked against the publisher inventory and read back, so the
                # installing release is provenance, not a precondition.
                findings.append(_finding("info", "installation_receipt_installed_by_other_release", path=str(receipt_path), installed_by=bound[:12], active_sha=active_sha[:12]))
            intake_path = Path(str(row.get("publisher_intake_path") or ""))
            if not intake_path.is_file() or not readable_by(intake_path, uid, gid):
                findings.append(_finding("blocker", "publisher_intake_unreadable_by_service", path=str(intake_path)))
            else:
                import hashlib

                digest = "sha256:" + hashlib.sha256(intake_path.read_bytes()).hexdigest()
                if digest != str(row.get("publisher_intake_sha256") or ""):
                    findings.append(_finding("blocker", "publisher_intake_digest_mismatch", path=str(intake_path)))
            findings.append(_finding("info", "installed_source_binding", receipt=str(receipt_path), bound_sha=bound[:12]))
    # The look-ahead admission replay re-runs the launch-preparation worker
    # inside the progression service, so that unit must resolve the profile too;
    # before the content registry it never received a per-scene PROFILE_ENV
    # drop-in and refused with sam31_server_profile_missing.
    for unit_name in ("blueprint-task-evaluation-launch-preparation.service",
                      "blueprint-task-evaluation-sam31-preparation-execution.service",
                      "blueprint-task-evaluation-configured-controls-progression.service"):
        unit = units.get(unit_name, {})
        registry = unit.get("effective_environment", {}).get("BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_DIR")
        if registry:
            path = Path(str(registry))
            if not path.is_absolute() or path.is_symlink() or not path.is_dir() or not readable_by(path, uid, gid):
                findings.append(_finding("blocker", "sam31_profile_registry_unreadable_by_service", unit=unit_name))
            else:
                findings.append(_finding("info", "sam31_profile_registry_content_bound", unit=unit_name, path=str(path)))
            for env_file, _ignore in unit.get("environment_files", []):
                if not Path(env_file).is_file():
                    findings.append(_finding("blocker", "environment_file_missing", unit=unit_name, path=env_file))
            continue
        profile = str(unit.get("effective_environment", {}).get("BLUEPRINT_TASK_EVALUATION_SAM31_PREPARATION_PROFILE_FILE") or "")
        if not profile:
            findings.append(_finding("blocker", "sam31_profile_unbound", unit=unit_name))
            continue
        path = Path(profile)
        if not path.is_file() or not readable_by(path, uid, gid):
            findings.append(_finding("blocker", "sam31_profile_unreadable_by_service", unit=unit_name, path=profile))
        sha8 = SHA8_IN_INPUT.search(profile)
        if sha8 and sha8.group(1) != active_sha[:8]:
            findings.append(_finding("blocker", "sam31_profile_bound_to_other_release", unit=unit_name, path=profile, bound_sha8=sha8.group(1), active_sha8=active_sha[:8]))
        findings.extend(_sam31_provider_profile_findings(path, unit_name, active_sha))
        for env_file, _ignore in unit.get("environment_files", []):
            if not Path(env_file).is_file():
                findings.append(_finding("blocker", "environment_file_missing", unit=unit_name, path=env_file))
    return findings


def _sam31_provider_profile_findings(hardware_profile: Path, unit_name: str, active_sha: str) -> list[dict[str, Any]]:
    """The provider profile a SAM hardware profile references must be bound to the active release.

    The SAM launch-packet validator refuses a GPU request whose provider profile carries
    another commit (since #1669 the worker stack manifest and runtime image build receipt
    only need a valid commit, so those are recorded, not refused).
    On 2026-09-05 the hardware profile for scene 841757 pointed at scene 840920's
    provider profile from three weeks earlier; the refusal surfaced only after a
    paid calibration render, as one anonymous blocker among forty predicates.
    """

    findings: list[dict[str, Any]] = []
    document = _read_json(hardware_profile) or {}
    references = document.get("artifact_references")
    reference = references.get("sam31_provider_profile") if isinstance(references, Mapping) else None
    provider_path = Path(str(reference.get("path") or "")) if isinstance(reference, Mapping) else None
    if provider_path is None or not str(provider_path).startswith("/"):
        findings.append(_finding("warning", "sam31_provider_profile_reference_missing", unit=unit_name, hardware_profile=str(hardware_profile)))
        return findings
    provider = _read_json(provider_path)
    if provider is None:
        findings.append(_finding("blocker", "sam31_provider_profile_unreadable", unit=unit_name, path=str(provider_path)))
        return findings
    bound = str(provider.get("source_commit_sha") or "")
    records = {"provider_profile": bound}
    for role in ("worker_stack_manifest", "runtime_image_build_receipt"):
        record = provider.get(role)
        record_path = Path(str(record.get("path") or "")) if isinstance(record, Mapping) else None
        if record_path is not None and record_path.is_file():
            records[role] = str((_read_json(record_path) or {}).get("source_commit_sha") or "")
    # #1669: only the provider profile itself must carry the active release; the worker stack
    # manifest and runtime image build receipt need a valid commit (no image-build producer
    # exists in the repo), so another commit there is provenance, not a refusal.
    stale = {"provider_profile": bound} if bound and bound != active_sha else {}
    if stale:
        findings.append(
            _finding(
                "blocker",
                "sam31_provider_profile_bound_to_other_release",
                unit=unit_name,
                path=str(provider_path),
                bound={role: sha[:12] for role, sha in stale.items()},
                records={role: sha[:12] for role, sha in records.items() if sha},
                active_sha=active_sha[:12],
                consequence="sam31_gpu_canary_request_configuration_invalid after the paid calibration render",
            )
        )
    else:
        findings.append(_finding("info", "sam31_provider_profile_bound_to_active_release", path=str(provider_path)))
    return findings


def handoff_checks(units: Mapping[str, dict[str, Any]], ids: tuple[int, int]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    uid, gid = ids
    unit_name = "blueprint-task-evaluation-configured-controls-progression.service"
    env = units.get(unit_name, {}).get("effective_environment", {})
    email = str(env.get("BLUEPRINT_POLICY_CANARY_NOTIFICATION_EMAIL") or "").strip()
    if not email or "@" not in email:
        findings.append(_finding("blocker", "policy_canary_notification_email_unset", unit=unit_name, env="BLUEPRINT_POLICY_CANARY_NOTIFICATION_EMAIL", consequence="hand-off returns policy_canary_handoff_not_configured after controls"))
    secret = str(env.get("BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_SECRET_FILE") or "")
    if not secret or not Path(secret).is_file() or not readable_by(Path(secret), uid, gid):
        findings.append(_finding("blocker", "webapp_submission_secret_file_unreadable_by_service", unit=unit_name, path=secret or None))
    if not str(env.get("BLUEPRINT_TASK_EVALUATION_WEBAPP_SUBMISSION_URL") or "").startswith("https://"):
        findings.append(_finding("blocker", "webapp_submission_url_unset", unit=unit_name))
    catalog = str(env.get("BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_CATALOG") or "")
    if not catalog:
        findings.append(_finding("blocker", "launch_profile_catalog_path_unset", unit=unit_name))
    public = str(units.get("blueprint-pipeline-intake.service", {}).get("effective_environment", {}).get("BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH") or "")
    if catalog and public and catalog != public:
        findings.append(_finding("blocker", "launch_profile_catalog_path_disagrees_with_intake", progression=catalog, intake=public))
    dispatcher_unit = "blueprint-task-evaluation-launch-dispatcher.service"
    dispatcher = units.get(dispatcher_unit, {}).get("effective_environment", {})
    if str(dispatcher.get("BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE") or "").strip() not in {"1", "true", "yes"}:
        # The unit file ships EXECUTE=true; a false value can only come from an
        # operator hold in an EnvironmentFile or drop-in. Name where it lives so
        # lifting it is one edit, not a search. Hands-off runs carry no hold: the
        # bounded, expiring standing authorization is the control.
        findings.append(_finding("blocker", "launch_dispatcher_execution_hold_present", unit=dispatcher_unit,
                                 held_by=environment_hold_sources(units.get(dispatcher_unit, {}), "BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE") or None))
    if not str(dispatcher.get("BLUEPRINT_TASK_EVALUATION_LAUNCH_EXECUTE_ID") or "").strip():
        # No per-launch id is the hands-off state: the standing authorization the
        # activation publishes admits the launch. Recorded, not a blocker.
        findings.append(_finding("info", "launch_dispatcher_execute_id_unset", unit=dispatcher_unit))
    return findings


def environment_hold_sources(unit: Mapping[str, Any], name: str) -> list[str]:
    """The EnvironmentFile paths that set ``name`` to a non-truthy value for this unit."""

    sources: list[str] = []
    for path_text, _ignore in environment_files(unit.get("properties", {})):
        value = _parse_env_file(Path(path_text)).get(name)
        if value is not None and value.strip() not in {"1", "true", "yes"}:
            sources.append(path_text)
    return sources


def credential_file_checks(units: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for unit_name, unit in units.items():
        ids = _service_ids(unit.get("user") or "root")
        if ids is None:
            continue
        uid, gid = ids
        for name, value in sorted(unit.get("effective_environment", {}).items()):
            if not (name.endswith("_FILE") or name.endswith("_PATH")) or not value.startswith("/"):
                continue
            if (unit_name, value) in seen:
                continue
            seen.add((unit_name, value))
            path = Path(value)
            if not path.exists():
                severity = "blocker" if name.endswith("_FILE") and ("SECRET" in name or "KEY" in name or "IDENTITY" in name or "CREDENTIAL" in name or "PAT" in name) else "warning"
                findings.append(_finding(severity, "configured_file_missing", unit=unit_name, env=name, path=value))
                continue
            if path.is_file() and not readable_by(path, uid, gid):
                findings.append(_finding("blocker", "configured_file_unreadable_by_service", unit=unit_name, env=name, path=value, **_owner(path.stat())))
            if "SSH_IDENTITY" in name and path.is_file():
                mode = stat.S_IMODE(path.stat().st_mode)
                if mode & 0o077 or path.stat().st_uid != uid:
                    findings.append(_finding("blocker", "ssh_identity_permissions_refused_by_ssh", unit=unit_name, path=value, **_owner(path.stat())))
    return findings


def _scope_override_sources(unit: Mapping[str, Any], name: str, expected: str) -> list[str]:
    """EnvironmentFile paths that set ``name`` to something other than ``expected``."""

    sources: list[str] = []
    for path_text, _ignore in environment_files(unit.get("properties", {})):
        value = _parse_env_file(Path(path_text)).get(name)
        if value is not None and value.strip() != expected:
            sources.append(path_text)
    return sources


def _owner_store_findings(unit_name: str, unit: Mapping[str, Any], ids: tuple[int, int]) -> list[dict[str, Any]]:
    """The scene-intake root or the autoprovision config must resolve the owner."""

    uid, gid = ids
    env = unit.get("effective_environment", {})
    findings: list[dict[str, Any]] = []
    root = str(env.get("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT") or "").strip()
    config = str(env.get("BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG") or "").strip()
    clients = str(env.get("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS") or "").strip()
    trusted = [c for c in clients.split(",") if c]
    resolved = False
    if root:
        path = Path(root)
        if path.is_absolute() and path.is_dir() and not path.is_symlink() and readable_by(path, uid, gid):
            resolved = True
        else:
            findings.append(_finding("blocker", "owner_scene_intake_root_unreadable_by_service", unit=unit_name, path=root))
    if config:
        cfg = _read_json(Path(config))
        if cfg is None or not readable_by(Path(config), uid, gid):
            findings.append(_finding("blocker", "owner_store_config_unreadable_by_service", unit=unit_name, path=config))
        else:
            scene_root = str(cfg.get("scene_root") or "")
            if scene_root and Path(scene_root).is_dir() and readable_by(Path(scene_root), uid, gid):
                resolved = True
            if not trusted and isinstance(cfg.get("trusted_clients"), list):
                trusted = [c for c in cfg["trusted_clients"] if c]
    if not resolved:
        findings.append(_finding(
            "blocker", "owner_store_unresolvable", unit=unit_name,
            scene_intake_root=root or None, controls_autoprovision_config=config or None,
            consequence="scene_policy_binding.scene_store() raises owner_store_missing before allocator entry"))
    if not trusted:
        findings.append(_finding("blocker", "owner_store_trusted_clients_unset", unit=unit_name))
    return findings


def _controls_autoprovision_findings(units: Mapping[str, dict[str, Any]], ids: tuple[int, int]) -> list[dict[str, Any]]:
    """Report a missing or broken controls-autoprovision config or its assets.

    A config env that names an unreadable/invalid file fails
    ``progression_owner_scope`` closed and stops every scene, so that case is a
    blocker; in owner mode an unset config is also a blocker because the hands-off
    construction->controls path never advances without it.
    """

    uid, gid = ids
    unit = units.get(CONTROLS_PROGRESSION_UNIT)
    if unit is None:
        return []
    env = unit.get("effective_environment", {})
    owner_mode = str(env.get("BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE") or "").strip() == "persistent_owner_only"
    config = str(env.get("BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG") or "").strip()
    findings: list[dict[str, Any]] = []
    if not config:
        findings.append(_finding(
            "blocker" if owner_mode else "warning", "controls_autoprovision_config_unset",
            unit=CONTROLS_PROGRESSION_UNIT,
            consequence="progression_owner_scope skips autoprovisioning; construction->controls never advances"))
        return findings
    cfg_path = Path(config)
    cfg = _read_json(cfg_path)
    if cfg is None or not readable_by(cfg_path, uid, gid):
        findings.append(_finding("blocker", "controls_autoprovision_config_unreadable_by_service",
            unit=CONTROLS_PROGRESSION_UNIT, path=config,
            consequence="progression_owner_scope fails closed (unresolved) and stops every scene"))
        return findings
    missing = [key for key in ("scene_root", "preparation_queue_root", "controls_root", "intent_root",
                               "profile_dir", "robot_catalog_path", "trusted_clients") if not cfg.get(key)]
    if missing:
        findings.append(_finding("blocker", "controls_autoprovision_config_incomplete",
            unit=CONTROLS_PROGRESSION_UNIT, missing=missing))
        return findings
    for key in ("controls_root", "intent_root"):
        path = Path(str(cfg[key]))
        if not path.is_dir() or path.is_symlink():
            findings.append(_finding("blocker", "controls_autoprovision_root_missing",
                unit=CONTROLS_PROGRESSION_UNIT, root=key, path=str(path)))
        elif not writable_by(path, uid, gid):
            findings.append(_finding("blocker", "controls_autoprovision_root_not_writable_by_service",
                unit=CONTROLS_PROGRESSION_UNIT, root=key, path=str(path), **_owner(path.stat())))
    catalog_path = Path(str(cfg["robot_catalog_path"]))
    catalog = _read_json(catalog_path)
    if catalog is None or not readable_by(catalog_path, uid, gid):
        findings.append(_finding("blocker", "controls_autoprovision_robot_catalog_unreadable",
            unit=CONTROLS_PROGRESSION_UNIT, path=str(catalog_path)))
        return findings
    # A9/R7: bind the catalog with the SAME resolver the real consumer uses, at the
    # active release -- do NOT invent a stricter schema rule. resolve_robot_catalog
    # accepts the sealed content catalog the installer actually writes
    # (task_evaluation_controls_robot_content_catalog.v1) as well as a resolved
    # catalog, verifies the catalog seal, and binds each asset by actual sha256 +
    # the runtime payload digest. First confirm the asset/runtime files are readable
    # BY THE SERVICE USER (the resolver reads as this process, which may be root),
    # then let the resolver refuse any unsealed/tampered/wrong-digest/invalid catalog.
    from .task_evaluation_controls_autoprovision import resolve_robot_catalog
    bindings = catalog.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        findings.append(_finding("blocker", "controls_autoprovision_robot_catalog_invalid",
            unit=CONTROLS_PROGRESSION_UNIT, path=str(catalog_path)))
        return findings
    for name, row in bindings.items():
        row = row if isinstance(row, Mapping) else {}
        for asset in ("robot_asset_usd", "embodiment_camera_template"):
            reference = row.get(asset) if isinstance(row.get(asset), Mapping) else {}
            asset_path = Path(str(reference.get("path") or ""))
            if not (str(asset_path).startswith("/") and asset_path.is_file() and not asset_path.is_symlink()
                    and readable_by(asset_path, uid, gid)):
                findings.append(_finding("blocker", "controls_autoprovision_asset_missing",
                    unit=CONTROLS_PROGRESSION_UNIT, binding=str(name), asset=asset, path=str(asset_path)))
        runtime = Path(str(row.get("runtime_source_payload_dir") or ""))
        if not (str(runtime).startswith("/") and runtime.is_dir() and not runtime.is_symlink()
                and readable_by(runtime, uid, gid)):
            findings.append(_finding("blocker", "controls_autoprovision_runtime_missing",
                unit=CONTROLS_PROGRESSION_UNIT, binding=str(name), path=str(runtime)))
    if any(finding["severity"] == "blocker" for finding in findings):
        return findings
    _, release_commit, _ = active_release()
    if not re.fullmatch(r"[0-9a-f]{40}", release_commit or ""):
        findings.append(_finding("blocker", "controls_autoprovision_active_release_unresolved",
            unit=CONTROLS_PROGRESSION_UNIT, path=str(catalog_path)))
        return findings
    try:
        resolve_robot_catalog(catalog, source_commit=release_commit)
    except ValueError as exc:
        findings.append(_finding("blocker", "controls_autoprovision_robot_catalog_unbindable",
            unit=CONTROLS_PROGRESSION_UNIT, path=str(catalog_path), reason=str(exc),
            consequence="resolve_robot_catalog refuses this catalog; construction->controls never advances"))
    return findings


def owner_scope_checks(units: Mapping[str, dict[str, Any]], ids: tuple[int, int]) -> list[dict[str, Any]]:
    """Owner-mode preflight: selection scope, owner store, and autoprovision assets.

    Selection is not authority; this only proves the persistent-owner posture is
    installed and resolvable so a dispatch selects the correctly bound owner row
    (and refuses when the store, config or assets are absent) before the
    allocator ever runs.  It changes nothing about the execute/dry-run holds.
    """

    findings: list[dict[str, Any]] = []
    for unit_name in OWNER_AUTHORITY_UNITS:
        unit = units.get(unit_name)
        if unit is None:
            continue
        env = unit.get("effective_environment", {})
        scope = str(env.get("BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE") or "").strip()
        if scope != "persistent_owner_only":
            findings.append(_finding(
                "blocker", "dispatch_owner_scope_not_persistent_owner_only", unit=unit_name,
                value=scope or None,
                overridden_by=_scope_override_sources(unit, "BLUEPRINT_TASK_EVALUATION_DISPATCH_OWNER_SCOPE",
                                                      "persistent_owner_only") or None,
                consequence="dispatcher defaults to all_authorized and can select legacy or unowned rows"))
        findings.extend(_owner_store_findings(unit_name, unit, ids))
    findings.extend(_controls_autoprovision_findings(units, ids))
    findings.extend(_terminal_index_findings(units))
    return findings


def _terminal_index_findings(units: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    unit = units.get(TERMINAL_INDEX_UNIT)
    if unit is None:
        return []
    env = unit.get("effective_environment", {})
    return [
        _finding(
            "blocker", "terminal_index_root_unset", unit=TERMINAL_INDEX_UNIT, variable=name,
            consequence="owner terminal receipts are never filed; a completed owner run never closes out")
        for name in TERMINAL_INDEX_ENV
        if not str(env.get(name) or "").strip()
    ]


PAID_ALLOCATION_UNITS: tuple[str, ...] = (
    "blueprint-task-evaluation-launch-dispatcher.service",
    "blueprint-task-evaluation-policy-canary-dispatcher.service",
    "blueprint-task-evaluation-sam31-preparation-execution.service",
)


def _env_usd(name: str, default: float, *environments: Mapping[str, Any]) -> float:
    for environment in (*environments, os.environ):
        raw = str(environment.get(name) or "").strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                return default
            return value if value >= 0 else default
    return default


def provider_credit_check(
    units: Mapping[str, dict[str, Any]], *, observer: Callable[..., Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Refuse a paid unit that leaves the per-attempt credit guard off; read the credit once ($0).

    Vast stops every instance when the account balance crosses its threshold, so an attempt
    that starts on thin credit is torn down mid-episode and loses its evidence.  The guard in
    ``provider_credit_admission`` is opt-in by environment; production must turn it on in every
    unit that reaches the adapter.  The read here is the same GET the guard performs, done before
    a submission so a thin account is named here rather than at the first paid attempt.
    """

    from .provider_credit_admission import ENABLED_ENV, RESERVE_ENV, WARNING_ENV, observe_vast_credit

    findings: list[dict[str, Any]] = []
    environments = [dict(unit.get("effective_environment", {})) for _, unit in sorted(units.items())]
    for unit_name in PAID_ALLOCATION_UNITS:
        unit = units.get(unit_name)
        if unit is None:
            continue
        enabled = str(unit.get("effective_environment", {}).get(ENABLED_ENV) or "").strip().lower()
        if enabled not in {"true", "1"}:
            findings.append(
                _finding("blocker", "provider_credit_guard_disabled", unit=unit_name, env=ENABLED_ENV, value=enabled or None)
            )
    key_file = next((str(e.get("VAST_API_KEY_FILE") or "").strip() for e in environments if e.get("VAST_API_KEY_FILE")), "")
    if not key_file:
        return [*findings, _finding("warning", "provider_credit_key_file_unset")]
    try:
        api_key = Path(key_file).read_text(encoding="utf-8").strip()
    except OSError:
        return [*findings, _finding("warning", "provider_credit_key_file_unreadable", path=key_file)]
    observation = (observer or observe_vast_credit)(api_key=api_key)
    credit = observation.get("credit_usd")
    if observation.get("status") != "observed" or not isinstance(credit, (int, float)) or isinstance(credit, bool):
        return [*findings, _finding(
            "warning", "provider_credit_unverifiable",
            http_status=observation.get("http_status"), blockers=list(observation.get("blockers") or []),
        )]
    reserve = _env_usd(RESERVE_ENV, 1.0, *environments)
    warning = _env_usd(WARNING_ENV, 5.0, *environments)
    if float(credit) < reserve:
        return [*findings, _finding("blocker", "provider_credit_exhausted", credit_usd=float(credit), reserve_usd=reserve)]
    severity, code = ("warning", "provider_credit_low") if float(credit) < warning else ("info", "provider_credit_available")
    return [*findings, _finding(severity, code, credit_usd=float(credit), warning_usd=warning)]


def disk_admission_check(units: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        usage = shutil.disk_usage(DISK_TARGET_ROOT)
    except OSError as exc:
        return [_finding("blocker", "disk_usage_unreadable", path=str(DISK_TARGET_ROOT), errno=exc.errno)]
    floor = max(8 * GIB, int(usage.total * 0.05))
    reserved = 0
    live = 0
    if DISK_RESERVATION_ROOT.is_dir():
        now = time.time()
        for path in DISK_RESERVATION_ROOT.iterdir():
            if path.name.startswith("."):
                continue
            document = _read_json(path) or {}
            expires = document.get("expires_at_epoch") or document.get("expires_at")
            if isinstance(expires, (int, float)) and expires < now:
                continue
            try:
                reserved += int(document.get("expected_bytes") or document.get("reserved_bytes") or 0)
                live += 1
            except (TypeError, ValueError):
                continue
    available = max(0, int(usage.free) - floor - reserved)
    footprints = {
        "launch_preparation": 2 * GIB, "episode_compilation": 2 * GIB, "launch_activation": 2 * GIB,
        "launch_dispatch": 2 * GIB, "policy_canary_dispatch": 2 * GIB,
    }
    refused = sorted(role for role, need in footprints.items() if need > available)
    chain_need = sum(footprints.values())
    findings.append(
        _finding(
            "blocker" if refused else ("warning" if chain_need > available else "info"),
            "disk_admission_projection",
            free_gib=round(usage.free / GIB, 2),
            floor_gib=round(floor / GIB, 2),
            reserved_gib=round(reserved / GIB, 2),
            live_reservations=live,
            available_gib=round(available / GIB, 2),
            refused_roles=refused or None,
            free_needed_for_one_role_gib=round((floor + 2 * GIB) / GIB, 2),
            free_needed_for_whole_chain_gib=round((floor + chain_need) / GIB, 2),
        )
    )
    return findings


def unit_health_checks(units: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for unit_name, unit in units.items():
        props = unit["properties"]
        state = _first(props, "ActiveState")
        result = _first(props, "Result")
        if state == "failed" or result not in {"success", ""}:
            findings.append(_finding("warning", "unit_failed_state", unit=unit_name, active_state=state, result=result, exec_main_status=_first(props, "ExecMainStatus")))
        if _first(props, "LoadState") != "loaded":
            findings.append(_finding("blocker", "unit_not_loaded", unit=unit_name, load_state=_first(props, "LoadState")))
        for condition in props.get("ExecCondition", []):
            if "/usr/bin/false" in condition or "/bin/false" in condition:
                # An ExecCondition that always fails is an operator hold on the
                # unit; the chain stalls there without any queue evidence.
                drop_ins = [p for p in (_first(props, "DropInPaths") or "").split() if p]
                findings.append(_finding("blocker", "unit_execution_hold_present", unit=unit_name, exec_condition=condition[:160], drop_ins=drop_ins or None))
        for trigger in (_first(props, "TriggeredBy") or "").split():
            trigger_state = _first(unit_properties(trigger), "ActiveState")
            if trigger_state != "active":
                findings.append(_finding("blocker", "unit_trigger_inactive", unit=unit_name, trigger=trigger, active_state=trigger_state))
    return findings


def intake_check(units: Mapping[str, dict[str, Any]], ids: tuple[int, int]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(  # nosec B310 - fixed loopback http literal
            INTAKE_CATALOG_URL, timeout=10
        ) as response:
            status = response.status
    except Exception as exc:  # noqa: BLE001
        findings.append(_finding("blocker", "intake_catalog_route_unreachable", url=INTAKE_CATALOG_URL, error=str(exc)[:200]))
        return findings
    if status != 200:
        findings.append(_finding("blocker", "intake_catalog_route_not_ok", url=INTAKE_CATALOG_URL, status=status))
    public = str(units.get("blueprint-pipeline-intake.service", {}).get("effective_environment", {}).get("BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH") or "")
    if public and not (Path(public).is_file() and readable_by(Path(public), *ids)):
        findings.append(_finding("warning", "public_catalog_file_unreadable_by_service", path=public))
    return findings


def append_history(path: Path, report: Mapping[str, Any], *, blockers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Append one bounded line per run so blocker drift is visible over time."""

    row = {
        "generated_at": report.get("generated_at"),
        "active_sha": report.get("active_sha"),
        "blocker_count": len(blockers),
        "warning_count": report.get("warning_count"),
        "blocker_codes": sorted({str(f.get("code")) for f in blockers}),
    }
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def run_chain(args: argparse.Namespace) -> int:
    if os.geteuid() != 0:
        print("run must execute as root on the control-plane host", file=sys.stderr)
        return 3
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "generated_at": _now(), "units": {}, "host_findings": []}
    release, active_sha, findings = active_release()
    report["active_release"] = str(release) if release else None
    report["active_sha"] = active_sha
    report["host_findings"].extend(findings)
    ids = _service_ids(SERVICE_ACCOUNT)
    if ids is None:
        report["host_findings"].append(_finding("blocker", "service_account_missing", account=SERVICE_ACCOUNT))
        ids = (0, 0)
    script = Path(args.probe_script or __file__).resolve()
    units: dict[str, dict[str, Any]] = report["units"]
    selected = list(args.unit) if args.unit else list(CHAIN_UNITS)
    for unit_name in selected:
        props = unit_properties(unit_name)
        if not props or _first(props, "LoadState") == "not-found":
            units[unit_name] = {"properties": props, "findings": [_finding("blocker", "unit_not_found", unit=unit_name)], "user": None}
            continue
        module = entry_module(props)
        env = effective_environment(props)
        entry: dict[str, Any] = {
            "properties": props,
            "user": _first(props, "User") or "root",
            "entry_module": module,
            "environment_files": environment_files(props),
            "effective_environment": env,
            "read_write_paths": _first(props, "ReadWritePaths").split(),
            "findings": [],
        }
        units[unit_name] = entry
        if module is None:
            entry["findings"].append(_finding("warning", "entry_module_unrecognised", unit=unit_name))
            continue
        if release is None or args.skip_sandbox:
            continue
        python = env.get("BLUEPRINT_TASK_EVALUATION_CONTROL_PLANE_PYTHON") or DEFAULT_PYTHON
        command = systemd_run_command(
            unit=unit_name, props=props, directives=configured_directives(unit_name), python=python, script=script,
            module=module, release=release, active_sha=active_sha,
        )
        entry["sandbox_command_directives"] = [item for item in command if item.startswith(tuple(f"{d}=" for d in SANDBOX_DIRECTIVES))]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
        stdout = completed.stdout.strip()
        try:
            probe = json.loads(stdout[stdout.index("{") :]) if "{" in stdout else None
        except ValueError:
            probe = None
        if probe is None:
            entry["findings"].append(
                _finding("blocker", "sandbox_probe_failed", unit=unit_name, returncode=completed.returncode, stderr=completed.stderr.strip()[-600:], stdout=stdout[-300:])
            )
            continue
        entry["probe"] = probe
        entry["findings"].extend(probe.get("findings", []))

    report["host_findings"].extend(intent_checks(active_sha, ids))
    report["host_findings"].extend(binding_checks(units, active_sha, ids))
    report["host_findings"].extend(handoff_checks(units, ids))
    report["host_findings"].extend(owner_scope_checks(units, ids))
    report["host_findings"].extend(credential_file_checks(units))
    report["host_findings"].extend(provider_credit_check(units))
    report["host_findings"].extend(disk_admission_check(units))
    report["host_findings"].extend(unit_health_checks(units))
    report["host_findings"].extend(intake_check(units, ids))

    # Effective environments carry secrets by reference only, but strip values
    # whose names say secret before anything is written or printed.
    for entry in units.values():
        entry["effective_environment"] = {
            key: ("<redacted>" if re.search(r"SECRET|TOKEN|PASSWORD|API_KEY(?!_FILE|_ID)", key) else value)
            for key, value in entry.get("effective_environment", {}).items()
        }
        entry.pop("properties", None)

    blockers = [(None, f) for f in report["host_findings"] if f["severity"] == "blocker"]
    warnings = [(None, f) for f in report["host_findings"] if f["severity"] == "warning"]
    for unit_name, entry in units.items():
        blockers.extend((unit_name, f) for f in entry.get("findings", []) if f["severity"] == "blocker")
        warnings.extend((unit_name, f) for f in entry.get("findings", []) if f["severity"] == "warning")
    report["blocker_count"] = len(blockers)
    report["warning_count"] = len(warnings)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        Path(args.json_out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if getattr(args, "history_out", None):
        append_history(Path(args.history_out), report, blockers=[f for _s, f in blockers])

    def line(scope: str | None, finding: Mapping[str, Any]) -> str:
        detail = {k: v for k, v in finding.items() if k not in {"severity", "code"}}
        text = json.dumps(detail, sort_keys=True, default=str)
        return f"  [{scope or 'host'}] {finding['code']} {text[:420]}"

    print(f"production chain preflight @ {report['generated_at']} active={active_sha[:12]} units={len(units)}")
    print(f"BLOCKERS ({len(blockers)}):")
    for scope, finding in blockers:
        print(line(scope, finding))
    print(f"WARNINGS ({len(warnings)}):")
    for scope, finding in warnings:
        print(line(scope, finding))
    if args.show_unset:
        for unit_name, entry in units.items():
            unset = entry.get("probe", {}).get("environment_names_read_but_unset", {})
            if unset:
                print(f"  [{unit_name}] read-but-unset: {', '.join(sorted(unset))[:900]}")
    return 2 if blockers else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="role", required=True)
    run = sub.add_parser("run", help="root: probe every chain unit under its own sandbox and run the host checks")
    run.add_argument("--unit", action="append", help="restrict to these units (repeatable)")
    run.add_argument("--json-out")
    run.add_argument("--history-out")
    run.add_argument("--probe-script", help="path of this file as visible inside the sandboxes")
    run.add_argument("--skip-sandbox", action="store_true")
    run.add_argument("--show-unset", action="store_true")
    run.set_defaults(func=run_chain)
    probe = sub.add_parser("probe", help="inside a sandbox: report what this unit can reach")
    probe.add_argument("--unit", required=True)
    probe.add_argument("--module", required=True)
    probe.add_argument("--release", required=True)
    probe.add_argument("--active-sha", default="")
    probe.add_argument("--read-write-paths", default="")
    probe.add_argument("--read-only-paths", default="")
    probe.set_defaults(func=lambda ns: print(json.dumps(run_probe(ns), sort_keys=True, default=str)) or 0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
