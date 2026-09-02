"""Signed, development-only source overlays for policy-canary iteration.

The default fast path is deliberately narrower than a production deployment:
it may replace only provider-runtime Python modules that already exist inside a
new policy-canary bundle.  It never edits the active control-plane checkout.
Any dependency, systemd, Website, schema, or otherwise unsupported change is
routed to the normal exact-main deployment instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import grp
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import stat
import subprocess
import time
from typing import Any
import zipfile

from .common import write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_canary_hotfix_overlay.v1"
SERVICE_ACCESS_SCHEMA_VERSION = "task_evaluation_canary_hotfix_service_access.v1"
# The hardened dispatcher unit runs as this identity with ProtectHome and
# ProtectSystem=strict; anything it must read has to be readable by it.
DEFAULT_SERVICE_USER = "blueprint"
DEFAULT_SERVICE_GROUP = "blueprint"
OVERLAY_DIRECTORY_MODE = 0o750
OVERLAY_FILE_MODE = 0o640
PLAN_SCHEMA_VERSION = "task_evaluation_canary_hotfix_overlay_plan.v1"
TEST_RECEIPT_SCHEMA_VERSION = "task_evaluation_canary_hotfix_test_receipt.v1"
APPLICATION_SCHEMA_VERSION = "task_evaluation_canary_hotfix_application.v1"
INSTALLATION_SCHEMA_VERSION = "task_evaluation_canary_hotfix_installation.v1"
DEFAULT_STRATEGY = "signed_hotfix_overlay"
FALLBACK_STRATEGY = "exact_main_deploy"
TARGET_SCOPE = "policy_canary_provider_runtime"
EVIDENCE_GRADE_CEILING = "development_only"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROVIDER_DESTINATION_BY_MODULE = {
    "adp009d_policy_server_worker.py": "adp009d_policy_server_worker.py",
    "groot_n17_droid_policy_runtime.py": "groot_n17_droid_policy_runtime.py",
    "groot_n17_wire_client.py": "groot_n17_wire_client.py",
    "openpi_droid_policy_runtime.py": "openpi_droid_policy_runtime.py",
    "native_task_arena_policy_canary_session.py": (
        "blueprint_pipeline/native_task_arena_policy_canary_session.py"
    ),
    "native_task_arena_policy_canary_worker.py": "adp_arena_provider_runner.py",
    "native_task_arena_policy_worker.py": (
        "blueprint_pipeline/native_task_arena_policy_worker.py"
    ),
}
_ALLOWED_RUNTIME_MODULES = frozenset(_PROVIDER_DESTINATION_BY_MODULE)
_ALLOWED_PROVIDER_DESTINATIONS = frozenset(_PROVIDER_DESTINATION_BY_MODULE.values())


class CanaryHotfixOverlayError(ValueError):
    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = tuple(sorted(set(str(item) for item in blockers if item)))
        super().__init__(";".join(self.blockers))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CanaryHotfixOverlayError(["canary_hotfix_git_query_failed"]) from exc


def resolve_service_account(
    *,
    user: str = DEFAULT_SERVICE_USER,
    group: str = DEFAULT_SERVICE_GROUP,
    getpwnam: Callable[[str], Any] = pwd.getpwnam,
    getgrnam: Callable[[str], Any] = grp.getgrnam,
    getgrall: Callable[[], Sequence[Any]] = grp.getgrall,
) -> dict[str, Any]:
    """Resolve the dispatcher's runtime identity into uid and every effective gid."""

    try:
        passwd = getpwnam(str(user))
        group_entry = getgrnam(str(group))
    except KeyError as exc:
        raise CanaryHotfixOverlayError(["canary_hotfix_service_account_unknown"]) from exc
    group_ids = {int(passwd.pw_gid), int(group_entry.gr_gid)}
    for entry in getgrall():
        if str(user) in tuple(entry.gr_mem or ()):
            group_ids.add(int(entry.gr_gid))
    return {
        "user": str(user),
        "uid": int(passwd.pw_uid),
        "group": str(group),
        "gid": int(group_entry.gr_gid),
        "group_ids": sorted(group_ids),
    }


def _service_account_has(
    metadata: os.stat_result, *, account: Mapping[str, Any], read: bool, traverse: bool
) -> bool:
    mode = stat.S_IMODE(metadata.st_mode)
    if metadata.st_uid == int(account["uid"]):
        read_bit, traverse_bit = stat.S_IRUSR, stat.S_IXUSR
    elif metadata.st_gid in set(int(value) for value in account["group_ids"]):
        read_bit, traverse_bit = stat.S_IRGRP, stat.S_IXGRP
    else:
        read_bit, traverse_bit = stat.S_IROTH, stat.S_IXOTH
    return (not read or bool(mode & read_bit)) and (
        not traverse or bool(mode & traverse_bit)
    )


def service_account_access_blockers(
    path: str | Path,
    *,
    account: Mapping[str, Any],
    stat_reader: Callable[[Path], os.stat_result] = os.stat,
) -> list[str]:
    """Return why the service account could not open ``path`` for reading.

    Every ancestor directory must be traversable and the target readable (and
    traversable when it is a directory).  Blockers name only the last path
    component, never a host path.  The paid run that motivated this check was
    refused because the overlay's parent directory was ``0750 root:root``.
    """

    target = Path(path).expanduser().resolve()
    blockers: list[str] = []
    for ancestor in reversed(target.parents):
        try:
            metadata = stat_reader(ancestor)
        except OSError:
            blockers.append(f"canary_hotfix_service_account_cannot_stat:{ancestor.name or '/'}")
            return blockers
        if not _service_account_has(metadata, account=account, read=False, traverse=True):
            blockers.append(
                f"canary_hotfix_service_account_cannot_traverse:{ancestor.name or '/'}"
            )
    try:
        metadata = stat_reader(target)
    except OSError:
        blockers.append(f"canary_hotfix_service_account_cannot_stat:{target.name}")
        return blockers
    is_directory = stat.S_ISDIR(metadata.st_mode)
    if not _service_account_has(
        metadata, account=account, read=True, traverse=is_directory
    ):
        blockers.append(f"canary_hotfix_service_account_cannot_read:{target.name}")
    return blockers


def bind_service_group(
    paths: Sequence[Path],
    *,
    account: Mapping[str, Any],
    chown: Callable[[Path, int, int], None] = os.chown,
) -> list[dict[str, Any]]:
    """Give the service group read (and directory traverse) access to ``paths``.

    Ownership stays with the operator that prepared the overlay; only the
    group changes, so the service account can read without gaining write.
    """

    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_symlink():
            raise CanaryHotfixOverlayError(["canary_hotfix_service_binding_symlink"])
        mode = OVERLAY_DIRECTORY_MODE if path.is_dir() else OVERLAY_FILE_MODE
        try:
            chown(path, -1, int(account["gid"]))
            path.chmod(mode)
        except OSError as exc:
            raise CanaryHotfixOverlayError(
                ["canary_hotfix_service_group_binding_failed"]
            ) from exc
        rows.append({"name": path.name, "group": account["group"], "mode": f"{mode:04o}"})
    return rows


def _service_access_receipt(
    paths: Sequence[Path],
    *,
    account: Mapping[str, Any],
    bound: Sequence[Mapping[str, Any]] | None = None,
    stat_reader: Callable[[Path], os.stat_result] = os.stat,
) -> dict[str, Any]:
    blockers: list[str] = []
    for path in paths:
        blockers.extend(
            service_account_access_blockers(path, account=account, stat_reader=stat_reader)
        )
    if blockers:
        raise CanaryHotfixOverlayError(
            ["canary_hotfix_overlay_unreadable_by_service_account", *blockers]
        )
    return {
        "schema_version": SERVICE_ACCESS_SCHEMA_VERSION,
        "status": "readable_by_service_account",
        "service_user": account["user"],
        "service_group": account["group"],
        "verified_names": [Path(path).name for path in paths],
        "group_bindings": [dict(row) for row in bound or ()],
    }


def _changed_paths(root: Path, *, base_commit: str, patch_commit: str) -> list[str]:
    raw = _git(
        root,
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        "-z",
        f"{base_commit}...{patch_commit}",
    )
    return sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)


def choose_canary_iteration_strategy(
    *, repo_root: str | Path, base_commit: str, patch_commit: str
) -> dict[str, Any]:
    """Prefer an overlay only when the exact Git delta is provider-runtime-only."""

    root = Path(repo_root).expanduser().resolve()
    blockers: list[str] = []
    if not _COMMIT.fullmatch(base_commit) or not _COMMIT.fullmatch(patch_commit):
        blockers.append("canary_hotfix_commit_invalid")
        changed: list[str] = []
    else:
        try:
            _git(root, "merge-base", "--is-ancestor", base_commit, patch_commit)
            changed = _changed_paths(
                root, base_commit=base_commit, patch_commit=patch_commit
            )
            remote_refs = _git(
                root,
                "for-each-ref",
                "--format=%(refname)",
                "--contains",
                patch_commit,
                "refs/remotes/origin",
            ).decode("utf-8")
            if not remote_refs.strip():
                blockers.append("canary_hotfix_patch_not_pushed")
        except CanaryHotfixOverlayError:
            blockers.append("canary_hotfix_commit_lineage_invalid")
            changed = []
    runtime_paths: list[str] = []
    test_paths: list[str] = []
    unsupported: list[str] = []
    for raw in changed:
        path = PurePosixPath(raw)
        if (
            len(path.parts) == 3
            and path.parts[:2] == ("src", "blueprint_pipeline")
            and path.name in _ALLOWED_RUNTIME_MODULES
        ):
            runtime_paths.append(raw)
        elif path.parts and path.parts[0] == "tests" and path.name.startswith("test_"):
            test_paths.append(raw)
        else:
            unsupported.append(raw)
    if not changed:
        blockers.append("canary_hotfix_change_set_empty")
    if not runtime_paths:
        blockers.append("canary_hotfix_runtime_change_missing")
    if not test_paths:
        blockers.append("canary_hotfix_focused_test_change_missing")
    if unsupported:
        blockers.append("canary_hotfix_unsupported_surface_changed")
    return {
        "schema_version": "task_evaluation_canary_iteration_strategy.v1",
        "strategy": DEFAULT_STRATEGY if not blockers else FALLBACK_STRATEGY,
        "base_commit": base_commit,
        "patch_commit": patch_commit,
        "changed_paths": changed,
        "runtime_paths": runtime_paths,
        "test_paths": test_paths,
        "unsupported_paths": unsupported,
        "blockers": sorted(set(blockers)),
        "evidence_grade_ceiling": EVIDENCE_GRADE_CEILING,
        "normal_deployment_required_for_promotion": True,
    }


def run_focused_hotfix_tests(
    *,
    repo_root: str | Path,
    base_commit: str,
    patch_commit: str,
    commands: Sequence[Sequence[str]],
    exact_failure_input: str | Path,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    failure_input = Path(exact_failure_input).expanduser().resolve()
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if head != patch_commit or dirty:
        raise CanaryHotfixOverlayError(
            ["canary_hotfix_tests_require_clean_patch_checkout"]
        )
    if not failure_input.is_file() or failure_input.is_symlink():
        raise CanaryHotfixOverlayError(["canary_hotfix_failure_input_invalid"])
    if not commands:
        raise CanaryHotfixOverlayError(["canary_hotfix_test_command_missing"])
    results: list[dict[str, Any]] = []
    for raw in commands:
        argv = [str(item) for item in raw]
        if not argv or any(not item for item in argv):
            raise CanaryHotfixOverlayError(["canary_hotfix_test_command_invalid"])
        started = time.time_ns()
        try:
            completed = subprocess.run(
                argv,
                cwd=root,
                capture_output=True,
                timeout=max(1, min(int(timeout_seconds), 120)),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CanaryHotfixOverlayError(["canary_hotfix_focused_test_failed"]) from exc
        result = {
            "argv": argv,
            "exit_code": completed.returncode,
            "started_at_unix_ns": started,
            "completed_at_unix_ns": time.time_ns(),
            "stdout_sha256": _sha256_bytes(completed.stdout),
            "stderr_sha256": _sha256_bytes(completed.stderr),
        }
        results.append(result)
        if completed.returncode != 0:
            raise CanaryHotfixOverlayError(["canary_hotfix_focused_test_failed"])
    receipt: dict[str, Any] = {
        "schema_version": TEST_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "base_commit": base_commit,
        "patch_commit": patch_commit,
        "exact_failure_input": {
            "path": str(failure_input),
            "size_bytes": failure_input.stat().st_size,
            "sha256": _sha256(failure_input),
        },
        "commands": results,
        "timeout_seconds": max(1, min(int(timeout_seconds), 120)),
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


def _validate_test_receipt(
    value: Mapping[str, Any], *, base_commit: str, patch_commit: str
) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value), allow_nan=False))
    failure_input = receipt.get("exact_failure_input")
    if (
        receipt.get("schema_version") != TEST_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "passed"
        or receipt.get("base_commit") != base_commit
        or receipt.get("patch_commit") != patch_commit
        or not isinstance(failure_input, Mapping)
        or not _DIGEST.fullmatch(str(failure_input.get("sha256") or ""))
        or not isinstance(receipt.get("commands"), list)
        or not receipt["commands"]
        or any(row.get("exit_code") != 0 for row in receipt["commands"])
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise CanaryHotfixOverlayError(["canary_hotfix_test_receipt_invalid"])
    return receipt


def prepare_canary_hotfix_overlay(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    base_commit: str,
    patch_commit: str,
    test_receipt: Mapping[str, Any],
    service_user: str = DEFAULT_SERVICE_USER,
    service_group: str = DEFAULT_SERVICE_GROUP,
    account_resolver: Callable[..., Mapping[str, Any]] = resolve_service_account,
    chown: Callable[[Path, int, int], None] = os.chown,
    stat_reader: Callable[[Path], os.stat_result] = os.stat,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    # Resolve the runtime identity before writing anything: an unknown service
    # account is a deployment defect, not something to discover after sealing.
    account = dict(account_resolver(user=service_user, group=service_group))
    strategy = choose_canary_iteration_strategy(
        repo_root=root, base_commit=base_commit, patch_commit=patch_commit
    )
    if strategy["strategy"] != DEFAULT_STRATEGY:
        raise CanaryHotfixOverlayError(strategy["blockers"])
    tests = _validate_test_receipt(
        test_receipt, base_commit=base_commit, patch_commit=patch_commit
    )
    if output.exists() or output.is_symlink():
        raise CanaryHotfixOverlayError(["canary_hotfix_output_exists"])
    output.mkdir(parents=True, mode=0o750)
    inventory: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for source_path in strategy["runtime_paths"]:
        payload = _git(root, "show", f"{patch_commit}:{source_path}")
        destination = _PROVIDER_DESTINATION_BY_MODULE[PurePosixPath(source_path).name]
        payloads[destination] = payload
        inventory.append(
            {
                "source_path": source_path,
                "destination": destination,
                "sha256": _sha256_bytes(payload),
                "size_bytes": len(payload),
                "mode": "0644",
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified_for_application",
        "default_strategy": DEFAULT_STRATEGY,
        "target_scope": TARGET_SCOPE,
        "base_release_commit": base_commit,
        "patch_commit": patch_commit,
        "source_inventory": inventory,
        "test_receipt_digest": tests["receipt_digest"],
        "exact_failure_input_digest": tests["exact_failure_input"]["sha256"],
        "evidence_grade_ceiling": EVIDENCE_GRADE_CEILING,
        "qualification_authorized": False,
        "official_ranking_authorized": False,
        "scene_promotion_authorized": False,
        "merge_back_required": True,
        "normal_deployment_required_for_promotion": True,
        "overlay_digest": "",
    }
    manifest["overlay_digest"] = canonical_digest(
        manifest, digest_field="overlay_digest"
    )
    archive_path = output / "task_evaluation_canary_hotfix_overlay.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for destination, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(f"provider_runtime/{destination}")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
        info = zipfile.ZipInfo("signed_hotfix_overlay.v1.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        archive.writestr(
            info,
            (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
    # The dispatcher runs as the service account with a read-only view of this
    # directory; bind the group now and prove readability before the plan is
    # sealed, so a root-only overlay can never reach the paid queue.
    bound = bind_service_group([output, archive_path], account=account, chown=chown)
    service_access = _service_access_receipt(
        [archive_path], account=account, bound=bound, stat_reader=stat_reader
    )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "status": "ready",
        "strategy": DEFAULT_STRATEGY,
        "manifest": manifest,
        "test_receipt": tests,
        "archive_path": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "service_access": service_access,
        "provider_mutation_performed": False,
        "active_release_mutation_performed": False,
        "plan_digest": "",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    plan_path = output / "task_evaluation_canary_hotfix_overlay_plan.v1.json"
    receipt_path = output / "task_evaluation_canary_hotfix_test_receipt.v1.json"
    write_json(plan_path, plan)
    write_json(receipt_path, tests)
    bind_service_group([plan_path, receipt_path], account=account, chown=chown)
    _service_access_receipt(
        [plan_path, receipt_path], account=account, stat_reader=stat_reader
    )
    return plan


def verify_canary_hotfix_overlay(path: str | Path) -> dict[str, Any]:
    archive_path = Path(path).expanduser().resolve()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = sorted(archive.namelist())
            manifest = json.loads(archive.read("signed_hotfix_overlay.v1.json"))
            expected = sorted(
                ["signed_hotfix_overlay.v1.json"]
                + [
                    f"provider_runtime/{row['destination']}"
                    for row in manifest.get("source_inventory") or []
                ]
            )
            blockers: list[str] = []
            if names != expected:
                blockers.append("canary_hotfix_archive_inventory_invalid")
            for row in manifest.get("source_inventory") or []:
                name = f"provider_runtime/{row['destination']}"
                payload = archive.read(name)
                if (
                    row.get("destination") not in _ALLOWED_PROVIDER_DESTINATIONS
                    or row.get("sha256") != _sha256_bytes(payload)
                    or row.get("size_bytes") != len(payload)
                ):
                    blockers.append("canary_hotfix_archive_file_invalid")
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise CanaryHotfixOverlayError(["canary_hotfix_archive_unreadable"]) from exc
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "verified_for_application"
        or manifest.get("default_strategy") != DEFAULT_STRATEGY
        or manifest.get("target_scope") != TARGET_SCOPE
        or manifest.get("evidence_grade_ceiling") != EVIDENCE_GRADE_CEILING
        or manifest.get("qualification_authorized") is not False
        or manifest.get("official_ranking_authorized") is not False
        or manifest.get("scene_promotion_authorized") is not False
        or manifest.get("merge_back_required") is not True
        or manifest.get("overlay_digest")
        != canonical_digest(manifest, digest_field="overlay_digest")
    ):
        blockers.append("canary_hotfix_manifest_invalid")
    if blockers:
        raise CanaryHotfixOverlayError(blockers)
    return manifest


def canary_hotfix_execution_release(manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = json.loads(json.dumps(dict(manifest), allow_nan=False))
    if (
        verified.get("schema_version") != SCHEMA_VERSION
        or verified.get("overlay_digest")
        != canonical_digest(verified, digest_field="overlay_digest")
    ):
        raise CanaryHotfixOverlayError(["canary_hotfix_manifest_invalid"])
    release: dict[str, Any] = {
        "mode": DEFAULT_STRATEGY,
        "base_release_commit": verified["base_release_commit"],
        "patch_commit": verified["patch_commit"],
        "overlay_digest": verified["overlay_digest"],
        "test_receipt_digest": verified["test_receipt_digest"],
        "exact_failure_input_digest": verified["exact_failure_input_digest"],
        "evidence_grade_ceiling": EVIDENCE_GRADE_CEILING,
        "qualification_authorized": False,
        "official_ranking_authorized": False,
        "scene_promotion_authorized": False,
        "normal_deployment_required_for_promotion": True,
        "release_digest": "",
    }
    release["release_digest"] = canonical_digest(
        release, digest_field="release_digest"
    )
    return release


def apply_canary_hotfix_overlay(
    *, archive_path: str | Path, provider_runtime_root: str | Path
) -> dict[str, Any]:
    manifest = verify_canary_hotfix_overlay(archive_path)
    runtime = Path(provider_runtime_root).expanduser().resolve()
    applications: list[dict[str, Any]] = []
    with zipfile.ZipFile(Path(archive_path).expanduser().resolve()) as archive:
        for row in manifest["source_inventory"]:
            destination = runtime / row["destination"]
            if destination.is_symlink() or not destination.is_file():
                raise CanaryHotfixOverlayError(
                    ["canary_hotfix_destination_not_preexisting"]
                )
            before = _sha256(destination)
            mode = stat.S_IMODE(destination.stat().st_mode)
            payload = archive.read(f"provider_runtime/{row['destination']}")
            temporary = destination.with_name(destination.name + ".hotfix-tmp")
            if temporary.exists() or temporary.is_symlink():
                raise CanaryHotfixOverlayError(["canary_hotfix_temporary_exists"])
            try:
                with temporary.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.chmod(mode)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            applications.append(
                {
                    "destination": row["destination"],
                    "before_sha256": before,
                    "after_sha256": _sha256(destination),
                }
            )
        write_json(runtime / "signed_hotfix_overlay.v1.json", manifest)
    receipt: dict[str, Any] = {
        "schema_version": APPLICATION_SCHEMA_VERSION,
        "status": "applied_to_staging_bundle",
        "target_scope": TARGET_SCOPE,
        "base_release_commit": manifest["base_release_commit"],
        "patch_commit": manifest["patch_commit"],
        "overlay_digest": manifest["overlay_digest"],
        "evidence_grade_ceiling": EVIDENCE_GRADE_CEILING,
        "active_release_mutation_performed": False,
        "provider_mutation_performed": False,
        "applications": applications,
        "application_digest": "",
    }
    receipt["application_digest"] = canonical_digest(
        receipt, digest_field="application_digest"
    )
    return receipt


def install_policy_canary_dispatcher_overlay(
    *,
    plan_path: str | Path,
    drop_in_path: str | Path,
    receipt_path: str | Path,
    service_user: str = DEFAULT_SERVICE_USER,
    service_group: str = DEFAULT_SERVICE_GROUP,
    account_resolver: Callable[..., Mapping[str, Any]] = resolve_service_account,
    stat_reader: Callable[[Path], os.stat_result] = os.stat,
) -> dict[str, Any]:
    plan_file = Path(plan_path).expanduser().resolve()
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    archive = Path(str(plan.get("archive_path") or "")).expanduser().resolve()
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or plan.get("status") != "ready"
        or plan.get("strategy") != DEFAULT_STRATEGY
        or plan.get("plan_digest") != canonical_digest(plan, digest_field="plan_digest")
        or not archive.is_file()
        or archive.is_symlink()
        or plan.get("archive_sha256") != _sha256(archive)
    ):
        raise CanaryHotfixOverlayError(["canary_hotfix_installation_plan_invalid"])
    verify_canary_hotfix_overlay(archive)
    # The drop-in points the hardened dispatcher at this archive.  Prove the
    # service account can actually open it (every ancestor traversable, the
    # file readable) before the unit is told about it.
    account = dict(account_resolver(user=service_user, group=service_group))
    service_access = _service_access_receipt(
        [archive, plan_file], account=account, stat_reader=stat_reader
    )
    destination = Path(drop_in_path).expanduser().resolve()
    if destination.suffix != ".conf" or not destination.is_absolute():
        raise CanaryHotfixOverlayError(["canary_hotfix_drop_in_path_invalid"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[Service]\n"
        "Environment=BLUEPRINT_TASK_EVALUATION_CANARY_HOTFIX_OVERLAY="
        f"{archive}\n"
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt: dict[str, Any] = {
        "schema_version": INSTALLATION_SCHEMA_VERSION,
        "status": "installed_for_next_policy_canary_dispatch",
        "plan_digest": plan["plan_digest"],
        "overlay_digest": plan["manifest"]["overlay_digest"],
        "archive_sha256": plan["archive_sha256"],
        "drop_in_path": str(destination),
        "drop_in_sha256": _sha256(destination),
        "service_access": service_access,
        "active_release_mutation_performed": False,
        "provider_mutation_performed": False,
        "normal_deployment_required_for_promotion": True,
        "installation_digest": "",
    }
    receipt["installation_digest"] = canonical_digest(
        receipt, digest_field="installation_digest"
    )
    write_json(Path(receipt_path).expanduser().resolve(), receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    route = sub.add_parser("route")
    route.add_argument("--repo-root", required=True)
    route.add_argument("--base-commit", required=True)
    route.add_argument("--patch-commit", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--repo-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--base-commit", required=True)
    prepare.add_argument("--patch-commit", required=True)
    prepare.add_argument("--exact-failure-input", required=True)
    prepare.add_argument("--test-command-json", action="append", required=True)
    prepare.add_argument("--service-user", default=DEFAULT_SERVICE_USER)
    prepare.add_argument("--service-group", default=DEFAULT_SERVICE_GROUP)
    install = sub.add_parser("install")
    install.add_argument("--plan", required=True)
    install.add_argument("--drop-in", required=True)
    install.add_argument("--receipt", required=True)
    install.add_argument("--service-user", default=DEFAULT_SERVICE_USER)
    install.add_argument("--service-group", default=DEFAULT_SERVICE_GROUP)
    args = parser.parse_args(argv)
    if args.command == "route":
        print(
            json.dumps(
                choose_canary_iteration_strategy(
                    repo_root=args.repo_root,
                    base_commit=args.base_commit,
                    patch_commit=args.patch_commit,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare":
        commands = [json.loads(value) for value in args.test_command_json]
        tests = run_focused_hotfix_tests(
            repo_root=args.repo_root,
            base_commit=args.base_commit,
            patch_commit=args.patch_commit,
            commands=commands,
            exact_failure_input=args.exact_failure_input,
        )
        plan = prepare_canary_hotfix_overlay(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            base_commit=args.base_commit,
            patch_commit=args.patch_commit,
            test_receipt=tests,
            service_user=args.service_user,
            service_group=args.service_group,
        )
        print(json.dumps(plan, sort_keys=True))
        return 0
    if args.command == "install":
        print(
            json.dumps(
                install_policy_canary_dispatcher_overlay(
                    plan_path=args.plan,
                    drop_in_path=args.drop_in,
                    receipt_path=args.receipt,
                    service_user=args.service_user,
                    service_group=args.service_group,
                ),
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


__all__ = [
    "CanaryHotfixOverlayError",
    "DEFAULT_SERVICE_GROUP",
    "DEFAULT_SERVICE_USER",
    "DEFAULT_STRATEGY",
    "EVIDENCE_GRADE_CEILING",
    "FALLBACK_STRATEGY",
    "apply_canary_hotfix_overlay",
    "bind_service_group",
    "canary_hotfix_execution_release",
    "choose_canary_iteration_strategy",
    "prepare_canary_hotfix_overlay",
    "install_policy_canary_dispatcher_overlay",
    "resolve_service_account",
    "run_focused_hotfix_tests",
    "service_account_access_blockers",
    "verify_canary_hotfix_overlay",
]


if __name__ == "__main__":
    raise SystemExit(main())
