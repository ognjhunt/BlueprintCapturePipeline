"""Canonical paid allocator for GPU canaries and remote CPU jobs.

No other public command may allocate these resource classes.  Provider-specific
modules are adapters behind this interface and their mutation CLIs are disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, write_json
from .paid_resource_admission import (
    RUNPOD_ONLY_ALLOCATOR_COMMANDS,
    RUNPOD_RETIRED_BLOCKER,
    PaidResourceAdmissionBlocked,
    runpod_allocator_request_blockers,
)
from .paid_resource_cli_arguments import (
    add_adp_resource_arguments as _add_adp_resource_arguments,
)
from . import policy_canary_allocator_lane as policy_canary_lane
from .native_task_arena_allocator_dispatch import (
    RIGID_ROBOT_FREE_PROBE_KIND, isolated_native_task_arena_lane,
)
from .remote_cpu_job_allocator import add_remote_cpu_job_arguments, run_remote_cpu_job
from .task_evaluation_scene_configuration_allocator import (
    PROBE_KIND as TASK_EVALUATION_SCENE_CONFIGURATION_PROBE_KIND,
    run_scene_configuration_allocator_probe,
)
from .task_evaluation_terminal_resource_release import dispatch_terminal_resource_release


ROOT = Path(__file__).resolve().parents[2]
CONTROL_PLANE_RELEASE_STATE_ROOT = Path(
    os.environ.get(
        "BLUEPRINT_CONTROL_PLANE_RELEASE_STATE_ROOT",
        "/var/lib/blueprint/pipeline-control-plane/"
        "task-evaluation-control-plane-releases",
    )
).expanduser()
DEPLOY_RELEASE_PROVENANCE_NAME = "deploy-release-provenance.json"
DETACHED_GPU_CANARY_SUPERVISOR_ENV = "BLUEPRINT_DETACHED_GPU_CANARY_SUPERVISOR"
LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR_ENV = (
    "BLUEPRINT_LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR"
)
DETACHED_GPU_CANARY_MANIFEST = "detached_gpu_canary_supervisor.json"
DETACHED_GPU_CANARY_LOG = "detached_gpu_canary_supervisor.log"
DETACHED_GPU_CANARY_LOCK = "detached_gpu_canary_supervisor.lock"
TERMINAL_RESOURCE_RELEASE_WORKER_ENV = (
    "BLUEPRINT_TASK_EVALUATION_TERMINAL_RESOURCE_RELEASE_WORKER"
)
CHECKOUT_IDENTITY_PROBE_TIMEOUT_SECONDS = 20
CHECKOUT_IDENTITY_PROBE_ATTEMPTS = 3
CHECKOUT_IDENTITY_PROBE_BACKOFF_SECONDS = 2.0


def _checkout_git_command(*arguments: str) -> list[str]:
    """Trust only the physical immutable checkout used by this allocator."""
    return ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), *arguments]


def _run_checkout_probe(argv: Sequence[str]) -> subprocess.CompletedProcess | None:
    """Probe immutable checkout identity, retrying a transient Git lock."""
    result: subprocess.CompletedProcess | None = None
    for attempt in range(CHECKOUT_IDENTITY_PROBE_ATTEMPTS):
        try:
            result = subprocess.run(
                list(argv), check=False, capture_output=True, text=True,
                timeout=CHECKOUT_IDENTITY_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            return result
        if attempt + 1 < CHECKOUT_IDENTITY_PROBE_ATTEMPTS:
            time.sleep(CHECKOUT_IDENTITY_PROBE_BACKOFF_SECONDS)
    return result


def _current_checkout_source_state() -> tuple[str, bool, bool]:
    """Separate a failed Git observation from a genuinely dirty checkout."""

    commit_result = _run_checkout_probe(
        _checkout_git_command("rev-parse", "--verify", "HEAD^{commit}")
    )
    status_result = _run_checkout_probe(
        _checkout_git_command("status", "--porcelain", "--untracked-files=no")
    )
    if commit_result is None or status_result is None:
        return "", False, False
    if commit_result.returncode != 0 or status_result.returncode != 0:
        return "", False, False
    commit = commit_result.stdout.strip().lower()
    commit_valid = bool(
        len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    )
    clean = not status_result.stdout.strip()
    return (commit if commit_valid else ""), clean, True


def _commit_is_merged_into(commit: str, ref: str) -> bool:
    """Is this commit already part of ``ref``'s history?

    A launch needs its code to be public and reviewed, which merging into main
    establishes and which every ancestor of main satisfies equally. It does not
    need to be main's *newest* commit. Requiring the tip means merging any later
    fix silently invalidates an already prepared release: on 2026-08-16 that
    discarded a promoted deploy plus a full commit-bound artifact rebuild, and
    the only way to launch again was to redo both for a commit whose worker code
    had not changed. Ancestry keeps the property that matters and drops the one
    that only served to make fixes and launches race each other.
    """

    if not commit or not ref:
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, ref],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def release_promotion_eligible(commit: str) -> bool:
    """Whether this release's evidence may climb above ``development_only``.

    An iteration deploy skips the Full Test Lane to keep a fix-and-fire loop
    at minutes instead of tens of minutes, and stamps its receipt
    ``status: "iteration"`` with ``promotion_eligible: false``. Those runs are
    real and paid, so their evidence is real -- but nothing produced from an
    unverified release may be presented as adjudication-grade.

    Fail closed: a missing, unreadable, or non-verified receipt is not
    eligible. Callers record this on the run so the evidence says which kind
    of release produced it rather than leaving it to be inferred later.
    """

    return _commit_has_verified_production_promotion(commit)


def _commit_has_verified_production_promotion(commit: str) -> bool:
    """Require the exact canonical Full Test Lane receipt for an old release.

    Ancestry proves that the bytes were merged; it does not prove that those
    exact bytes passed the authoritative production-promotion lane.  The
    canonical deploy installs the already-live-verified GitHub receipt beside
    the release state, outside the clean Git checkout.  This local check keeps
    paid admission offline and prevents a merely merged ancestor from gaining
    the stronger status of a promoted release.
    """

    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return False
    # Two deploy surfaces install the same receipt under different roots: the
    # task-evaluation release-state root this module has always read, and the
    # control-plane root one directory above it, where the 2026-08-30 deploy
    # chain installed every release's receipt (8f5b2313 admitted as
    # development_only until an operator hard-linked the verified receipt
    # across).  Accept the identical fail-closed content proof at either exact
    # location; nothing about what the receipt must say is weakened.
    candidates = (
        CONTROL_PLANE_RELEASE_STATE_ROOT / commit / DEPLOY_RELEASE_PROVENANCE_NAME,
        CONTROL_PLANE_RELEASE_STATE_ROOT.parent
        / commit
        / DEPLOY_RELEASE_PROVENANCE_NAME,
    )
    return any(
        _verified_release_provenance_at(path, commit=commit)
        for path in candidates
    )


def _verified_release_provenance_at(path: Path, *, commit: str) -> bool:
    """One exact location's receipt proves the canonical Full Test Lane."""

    try:
        if path.is_symlink() or not path.is_file():
            return False
        metadata = path.stat()
        if metadata.st_mode & 0o022:
            return False
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(value, Mapping):
        return False
    collection = value.get("collection")
    claim_boundary = value.get("claim_boundary")
    return bool(
        value.get("schema_version") == "blueprint.deploy_release_provenance.v1"
        and value.get("status") == "verified"
        and value.get("git_sha") == commit
        and value.get("workflow_name") == "Full Test Lane"
        and value.get("workflow_path") == ".github/workflows/full-test-lane.yml"
        and value.get("job_name") == "Full pytest lane on CPU runner"
        and type(value.get("run_id")) is int
        and value.get("run_id", 0) > 0
        and isinstance(collection, Mapping)
        and type(collection.get("test_count")) is int
        and collection.get("test_count", 0) > 0
        and isinstance(claim_boundary, Mapping)
        and claim_boundary.get("canonical_full_lane_verified") is True
    )


def _current_origin_main_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "rev-parse",
                "--verify",
                "origin/main^{commit}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    commit = result.stdout.strip().lower()
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return ""
    return commit


def _current_remote_main_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "ls-remote",
                "--exit-code",
                "origin",
                "refs/heads/main",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    fields = result.stdout.strip().split()
    commit = fields[0].lower() if len(fields) == 2 else ""
    if (
        result.returncode != 0
        or fields[1:] != ["refs/heads/main"]
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return ""
    return commit


def _current_branch_name() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "branch", "--show-current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _current_remote_branch_commit(branch: str) -> str:
    if not branch or not branch.startswith("codex/"):
        return ""
    reference = f"refs/heads/{branch}"
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-remote", "--exit-code", "origin", reference],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    fields = result.stdout.strip().split()
    commit = fields[0].lower() if len(fields) == 2 else ""
    if (
        result.returncode != 0
        or fields[1:] != [reference]
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return ""
    return commit


def _remote_pull_request_head_has_commit(commit: str) -> bool:
    """Whether origin serves ``commit`` as a pull request's head.

    Merging deletes the experiment branch, but GitHub keeps every
    ``refs/pull/<n>/head`` and refuses pushes to it.
    """
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-remote", "origin", "refs/pull/*/head"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and any(
        len(fields) == 2 and fields[0].lower() == commit and re.fullmatch(r"refs/pull/[0-9]+/head", fields[1])
        for fields in (line.split() for line in result.stdout.splitlines())
    )


def _website_atlas_release_blockers(experimental: bool) -> tuple[str, list[str]]:
    commit, _, _ = _current_checkout_source_state()
    return commit, _source_checkout_blockers(commit, allow_pushed_branch_diagnostic=experimental)[0]


def run_sponsored_website_atlas_pose(*, output_root: Path, capture_id: str, task_context: Mapping[str, Any],
                                     experimental: bool = False,
                                     **options: Any) -> dict[str, Any]:
    """Canonical allocator entry for one admitted Atlas images2PosedRGBD operation.

    Only an explicit controller decision calls this; see website_atlas_pose.
    """
    from .website_atlas_pose import run_admitted_atlas_pose

    return run_admitted_atlas_pose(output_root=output_root, capture_id=capture_id, task_context=task_context,
        release_blockers=lambda: _website_atlas_release_blockers(experimental), **options)


def run_sponsored_website_atlas_task(*, experimental: bool = False, **options: Any) -> dict[str, Any]:
    """Canonical allocator for Atlas view/edit/mesh tasks under the same source cap."""
    from .website_atlas_tasks import run_admitted_atlas_task

    return run_admitted_atlas_task(release_blockers=lambda: _website_atlas_release_blockers(experimental), **options)


def run_sponsored_website_atlas_automation(*, experimental: bool = False, **options: Any) -> dict[str, Any]:
    """The existing source gate protects SAM, editor and Atlas automation alike."""
    from .atlas_scene_automation import run_automation
    preserved_capture = options.pop('preserved_capture', False)

    try:
        arguments = dict(release_blockers=lambda: _website_atlas_release_blockers(experimental),
            task_runner=lambda **kw: run_sponsored_website_atlas_task(experimental=experimental, **kw),
            planner_runner=lambda **kw: run_sponsored_website_atlas_plan(experimental=experimental, **kw),
            pose_runner=lambda **kw: run_sponsored_website_atlas_pose(experimental=experimental, **kw), **options)
        if preserved_capture:
            from .atlas_preserved_completion import run_preserved_completion
            return run_preserved_completion(**arguments)
        return run_automation(**arguments)
    except PaidResourceAdmissionBlocked as exc:
        from .gcp_monthly_budget_admission import stale_report_wait
        waiting = (stale_report_wait() if exc.blockers == ["gcp_monthly_budget_admission_blocked"] else None)
        if waiting is None:
            raise
        # Completed stages and their reservation bindings remain in the same
        # output root. The next controller tick must pass admission again.
        return {"status": "processing", "stage": "paid_admission_wait", **waiting}


def run_sponsored_website_atlas_plan(*, experimental: bool = False, **options: Any) -> dict[str, Any]:
    """Source review uses the same allocator release gate and WebApp test cap."""
    from .atlas_scene_planner import review_survey

    return review_survey(release_blockers=lambda: _website_atlas_release_blockers(experimental), **options)


def _source_checkout_blockers(
    expected_source_commit: str, *, allow_pushed_branch_diagnostic: bool = False
) -> tuple[list[str], str]:
    checkout_commit, checkout_clean, probe_ran = _current_checkout_source_state()
    blockers: list[str] = []
    if not probe_ran:
        return ["gpu_canary_checkout_identity_probe_failed"], ""
    if not checkout_commit:
        blockers.append("gpu_canary_checkout_source_commit_unavailable")
    elif expected_source_commit.strip().lower() != checkout_commit:
        blockers.append("gpu_canary_expected_source_commit_not_current_checkout")
    from .active_deployed_release_admission import inspect_active_deployed_release
    deployed = inspect_active_deployed_release(ROOT, checkout_commit)
    if deployed is not None:
        blockers.extend(deployed["blockers"] + ([] if checkout_clean else ["gpu_canary_checkout_not_clean"]))
        return blockers, checkout_commit
    if allow_pushed_branch_diagnostic:
        branch = _current_branch_name()
        remote_branch_commit = _current_remote_branch_commit(branch)
        if not branch.startswith("codex/"):
            blockers.append("gpu_canary_experimental_branch_not_codex_namespaced")
        if not remote_branch_commit:
            # A merge mid-run deleted the branch three times on 2026-10-24 and
            # stopped live Atlas runs. Its pushed head stays served as a PR ref.
            if not (branch.startswith("codex/") and _remote_pull_request_head_has_commit(checkout_commit)):
                blockers.append("gpu_canary_remote_experimental_branch_commit_unavailable")
        elif checkout_commit != remote_branch_commit:
            blockers.append("gpu_canary_checkout_not_pushed_experimental_branch")
    else:
        origin_main_commit = _current_origin_main_commit()
        remote_main_commit = _current_remote_main_commit()
        accepted_by_origin = False
        accepted_by_remote = False
        if not origin_main_commit:
            blockers.append("gpu_canary_origin_main_commit_unavailable")
        elif checkout_commit == origin_main_commit:
            accepted_by_origin = True
        else:
            accepted_by_origin = _commit_is_merged_into(checkout_commit, "origin/main")
            if not accepted_by_origin:
                blockers.append("gpu_canary_checkout_not_origin_main")
        if not remote_main_commit:
            blockers.append("gpu_canary_remote_main_commit_unavailable")
        elif checkout_commit == remote_main_commit:
            accepted_by_remote = True
        else:
            accepted_by_remote = _commit_is_merged_into(checkout_commit, remote_main_commit)
            if not accepted_by_remote:
                blockers.append("gpu_canary_checkout_not_remote_main")
        ancestor_release = bool(
            checkout_commit
            and origin_main_commit
            and remote_main_commit
            and (
                checkout_commit != origin_main_commit
                or checkout_commit != remote_main_commit
            )
        )
        if (
            ancestor_release
            and accepted_by_origin
            and accepted_by_remote
            and not _commit_has_verified_production_promotion(checkout_commit)
        ):
            blockers.append("gpu_canary_checkout_promotion_provenance_invalid")
    if not checkout_clean:
        blockers.append("gpu_canary_checkout_not_clean")
    return blockers, checkout_commit


def _control_plane_checkout_blockers() -> tuple[list[str], dict[str, object]]:
    """Pin the clean allocator identity without coupling it to the runtime image.

    Main pointers remain recorded diagnostics.  A clean detached commit or
    reviewed branch is a valid orchestrator identity and does not force an
    immutable runtime-image rebuild.
    """

    checkout_commit, checkout_clean, probe_ran = _current_checkout_source_state()
    origin_main_commit = _current_origin_main_commit()
    remote_main_commit = _current_remote_main_commit()
    blockers: list[str] = []
    if not probe_ran:
        blockers.append("gpu_canary_orchestrator_identity_probe_failed")
    else:
        if not checkout_commit:
            blockers.append("gpu_canary_orchestrator_source_commit_unavailable")
        if not checkout_clean:
            blockers.append("gpu_canary_orchestrator_checkout_not_clean")
    identity = {
        "schema_version": "blueprint.gpu_canary_control_plane_identity.v1",
        "orchestrator_source_commit": checkout_commit or None,
        "checkout_clean": checkout_clean,
        "identity_probe_ran": probe_ran,
        "origin_main_commit": origin_main_commit or None,
        "remote_main_commit": remote_main_commit or None,
        "orchestrator_equals_origin_main": bool(
            checkout_commit and checkout_commit == origin_main_commit
        ),
        "orchestrator_equals_remote_main": bool(
            checkout_commit and checkout_commit == remote_main_commit
        ),
        "main_parity_is_diagnostic_not_runtime_identity": True,
        # An iteration deploy skips the Full Test Lane to keep a fix-and-fire
        # loop at minutes rather than tens of minutes. Those runs are real and
        # paid, so their evidence is real -- but it must never be presented as
        # adjudication-grade. Record which kind of release produced the run
        # here, where every paid admission already reads, instead of leaving
        # it to be inferred from the release directory later.
        "release_promotion_eligible": bool(
            checkout_commit and release_promotion_eligible(checkout_commit)
        ),
        "evidence_grade_ceiling": (
            "promotion_eligible"
            if checkout_commit and release_promotion_eligible(checkout_commit)
            else "development_only"
        ),
        "raw_secret_values_recorded": False,
    }
    return blockers, identity


def _adp_expected_source_commit_blockers(
    expected_source_commit: str, control_identity: Mapping[str, object]
) -> tuple[list[str], str]:
    """Bind an ADP paid run to the exact clean orchestrator commit."""

    expected = expected_source_commit.strip().lower()
    blockers: list[str] = []
    if not expected:
        blockers.append("adp_gpu_canary_expected_source_commit_missing")
    elif len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        blockers.append("adp_gpu_canary_expected_source_commit_invalid")
    elif expected != str(control_identity.get("orchestrator_source_commit") or "").lower():
        blockers.append("adp_gpu_canary_expected_source_commit_not_control_plane_checkout")
    return blockers, expected


def _configure_detached_supervisor_signal_policy(command: str) -> bool:
    """Keep an explicitly detached paid supervisor alive through local SIGINT.

    SIGTERM remains available for an intentional stop. Provider resources also
    remain bounded by their independent deadline watchdogs.
    """

    detached_gpu_canary = (
        command == "gpu-canary" and os.getenv(DETACHED_GPU_CANARY_SUPERVISOR_ENV) == "1"
    )
    if not detached_gpu_canary:
        return False
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        return False
    return True


def maybe_launch_detached_gpu_canary(
    *,
    command: str,
    execute: bool,
    supervisor_dir: str | None,
    argv: Sequence[str],
    repo_root: Path,
) -> dict[str, Any] | None:
    """Detach the canonical allocator without recording raw arguments or secrets."""

    if command != "gpu-canary" or not supervisor_dir:
        return None
    if os.getenv(DETACHED_GPU_CANARY_SUPERVISOR_ENV) == "1":
        return None
    if not execute:
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_requires_execute"],
            "provider_mutations_performed": 0,
        }
    declared = Path(supervisor_dir).expanduser()
    if declared.exists() and declared.is_symlink():
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_supervisor_dir_symlink_forbidden"],
            "provider_mutations_performed": 0,
        }
    root = declared.resolve()
    ensure_dir(root)
    try:
        if stat.S_IMODE(root.stat().st_mode) != 0o700:
            os.chmod(root, 0o700)
        installed_mode = stat.S_IMODE(root.stat().st_mode)
    except OSError:
        return {
            "status": "blocked",
            "blockers": [
                "detached_gpu_canary_supervisor_dir_permission_install_failed"
            ],
            "provider_mutations_performed": 0,
        }
    if not root.is_dir() or installed_mode != 0o700:
        return {
            "status": "blocked",
            "blockers": [
                "detached_gpu_canary_supervisor_dir_permission_install_failed"
            ],
            "provider_mutations_performed": 0,
        }
    manifest_path = root / DETACHED_GPU_CANARY_MANIFEST
    log_path = root / DETACHED_GPU_CANARY_LOG
    lock_path = root / DETACHED_GPU_CANARY_LOCK
    if manifest_path.exists() or log_path.exists() or lock_path.exists():
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_supervisor_artifact_exists"],
            "provider_mutations_performed": 0,
        }
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return {
            "status": "blocked",
            "blockers": ["detached_gpu_canary_supervisor_already_starting"],
            "provider_mutations_performed": 0,
        }
    os.close(lock_fd)
    raw_arguments = [str(value) for value in argv]
    argument_shape = [
        value.split("=", 1)[0]
        for index, value in enumerate(raw_arguments)
        if index == 0 or value.startswith("--")
    ]
    argument_shape_digest = (
        "sha256:" + hashlib.sha256("\0".join(argument_shape).encode("utf-8")).hexdigest()
    )
    pending = {
        "schema_version": "detached_gpu_canary_supervisor.v1",
        "status": "launch_pending",
        "argument_shape_digest": argument_shape_digest,
        "argument_count": len(raw_arguments),
        "raw_arguments_recorded": False,
        "raw_argument_values_hashed": False,
        "raw_secret_values_recorded": False,
        "provider_mutations_performed_by_launcher": 0,
    }
    write_json(manifest_path, pending)
    os.chmod(manifest_path, 0o600)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    child_env = dict(os.environ)
    child_env[DETACHED_GPU_CANARY_SUPERVISOR_ENV] = "1"
    try:
        with os.fdopen(log_fd, "ab", closefd=True) as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "blueprint_pipeline.paid_resource_allocator",
                    *raw_arguments,
                ],
                cwd=str(repo_root),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException:
        write_json(
            manifest_path,
            {
                **pending,
                "status": "launch_failed",
                "blockers": ["detached_gpu_canary_supervisor_launch_failed"],
            },
        )
        os.chmod(manifest_path, 0o600)
        raise
    result = {
        **pending,
        "status": "supervisor_started",
        "pid": process.pid,
        "independent_process": True,
        "start_new_session": True,
        "sigint_ignored_by_child": True,
        "log_path": str(log_path),
    }
    write_json(manifest_path, result)
    os.chmod(manifest_path, 0o600)
    return result


def configure_or_launch_detached_gpu_canary(
    command: str,
    *,
    execute: bool,
    argv: Sequence[str],
    repo_root: Path,
) -> int | None:
    """Detach a requested GPU canary, otherwise configure child signal policy."""

    detached = maybe_launch_detached_gpu_canary(
        command=command,
        execute=execute,
        supervisor_dir=os.getenv(LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR_ENV),
        argv=argv,
        repo_root=repo_root,
    )
    if detached is not None:
        print(json.dumps(detached, sort_keys=True))
        return 0 if detached.get("status") == "supervisor_started" else 2
    _configure_detached_supervisor_signal_policy(command)
    return None


def _load(path: str | Path) -> dict:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    from .core.handoff_lease import handoff_lease_checkpoint
    handoff_lease_checkpoint("canonical_allocator_entry")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_remote_cpu_job_arguments(commands)
    gpu = commands.add_parser("gpu-canary")
    # These are legacy generic-canary inputs.  Specialized paid lanes bind
    # their own immutable bundles and must not need meaningless placeholders.
    # The generic branch below still rejects their absence before any mutation.
    gpu.add_argument("--provider-launch-request")
    gpu.add_argument("--release-evidence")
    gpu.add_argument("--model-cache-evidence")
    gpu.add_argument("--preflight-bundle")
    gpu.add_argument("--admission-out")
    gpu.add_argument("--bound-request-out")
    gpu.add_argument("--adapter-output")
    gpu.add_argument("--pod-name")
    gpu.add_argument("--terminal-resource-release", help="Immutable release-only request; cannot be combined with a launch profile.")
    gpu.add_argument(
        "--terminal-resource-release-output",
        help="Receipt destination for an exact stopped-provider-record release.",
    )
    gpu.add_argument("--expected-source-commit")
    gpu.add_argument(
        "--experimental-branch-diagnostic",
        action="store_true",
        help=(
            "Allow a clean exact commit on its pushed codex/ branch for diagnostic canaries. "
            "Release and production evidence still require exact main."
        ),
    )
    gpu.add_argument(
        "--expected-image-source-commit",
        help=(
            "Exact source commit represented by the runtime image. "
            "--expected-source-commit remains a compatibility alias."
        ),
    )
    gpu.add_argument(
        "--provider",
        choices=("runpod", "vast", "digitalocean", "aws"),
        default="runpod",
    )
    gpu.add_argument(
        "--probe-kind",
        choices=(
            # RunPod lane names the RunPod refusal still recognises; no lane runs them.
            "startup",
            "strict-policy-smoke",
            "persistent-policy-wam-loop",
            "single-kitchen-episode",
            "single-kitchen-finetune",
            "openpi-policy-ranking",
            "new-site-diagnostic-canary",
            policy_canary_lane.PROBE_KIND, RIGID_ROBOT_FREE_PROBE_KIND,
            TASK_EVALUATION_SCENE_CONFIGURATION_PROBE_KIND,
        ),
        default="strict-policy-smoke",
    )
    gpu.add_argument(
        "--openpi-provider",
        choices=("vast", "runpod"),
        default="vast",
        help="Policy-ranking GPU provider; Vast is the frozen default.",
    )
    gpu.add_argument(
        "--native-task-arena-bundle-receipt",
        help=(
            "Previously dry-run native task bundle receipt. Required for execute so "
            "the paid launch cannot rebuild different bytes."
        ),
    )
    gpu.add_argument(
        "--native-task-arena-attempt-authority",
        help="Exact single-use paid-attempt authority; required for execute.",
    )
    policy_canary_lane.add_policy_canary_allocator_arguments(gpu)
    gpu.add_argument("--adp-job-dir")
    _add_adp_resource_arguments(gpu)
    gpu.add_argument("--scene-configuration-bundle-receipt")
    gpu.add_argument("--scene-configuration-attempt-authority")
    gpu.add_argument("--scene-configuration-job-dir")
    gpu.add_argument("--scene-configuration-queue-root")
    gpu.add_argument(
        "--scene-configuration-diagnostic-only",
        action="store_true",
        help=(
            "Resume an immutable scene-configuration checkpoint for blocker "
            "diagnosis only; never qualify or publish the result."
        ),
    )
    gpu.add_argument(
        "--scene-configuration-retain-warm-session",
        action="store_true",
        help=(
            "Retain the one admitted diagnostic Vast worker under its independent "
            "watchdog for immutable checkpoint-resume source overlays."
        ),
    )
    gpu.add_argument(
        "--scene-configuration-allowed-vast-machine-id",
        action="append",
        type=int,
        default=[],
        help=(
            "Restrict a diagnostic scene-configuration allocation to a "
            "receipt-bound set of previously proven Vast machines. Repeat the "
            "option for multiple machines. Production qualification rejects it."
        ),
    )
    gpu.add_argument("--scene-configuration-warm-session-authority")
    gpu.add_argument("--scene-configuration-warm-session-output-root")
    gpu.add_argument(
        "--scene-configuration-warm-action",
        choices=("iterate", "closeout"),
    )
    gpu.add_argument("--scene-configuration-warm-session-root")
    gpu.add_argument("--scene-configuration-warm-iteration-authority")
    gpu.add_argument("--scene-configuration-warm-closeout-receipt")
    gpu.add_argument("--execute", action="store_true")
    for name in sorted(RUNPOD_ONLY_ALLOCATOR_COMMANDS):
        commands.add_parser(name, help=argparse.SUPPRESS)
    # A retired RunPod-only command is registered without its old arguments so it can be
    # refused below; every other command is parsed strictly, as before.
    args, _unrecognized = parser.parse_known_args(argv)
    if args.command not in RUNPOD_ONLY_ALLOCATOR_COMMANDS:
        args = parser.parse_args(argv)
    retired_aws = getattr(args, "provider", None) == "aws"
    if retired := (["aws_provider_integration_removed"] if retired_aws else runpod_allocator_request_blockers(args)):
        result = {"status": "blocked", "blockers": retired, "provider_mutations_performed": 0, "success": False}
        if getattr(args, "admission_out", None):
            write_json(Path(args.admission_out), result)
        print(json.dumps(result, sort_keys=True))
        return 2
    if args.command == "gpu-canary":
        normal_required = (
            "provider_launch_request", "release_evidence", "model_cache_evidence",
            "preflight_bundle", "admission_out", "bound_request_out", "adapter_output", "pod_name",
        )
        if args.terminal_resource_release:
            if not args.terminal_resource_release_output or not args.execute:
                parser.error(
                    "--terminal-resource-release requires --terminal-resource-release-output and --execute"
                )
            if str(os.getenv(TERMINAL_RESOURCE_RELEASE_WORKER_ENV) or "").strip().lower() not in {
                "1", "true", "yes", "on"
            }:
                parser.error("--terminal-resource-release is restricted to the queue worker")
            if any(getattr(args, name, None) for name in normal_required):
                parser.error("--terminal-resource-release cannot be combined with launch arguments")
    detached_exit = configure_or_launch_detached_gpu_canary(
        args.command,
        execute=bool(getattr(args, "execute", False)),
        argv=list(argv) if argv is not None else sys.argv[1:],
        repo_root=ROOT,
    )
    if detached_exit is not None:
        return detached_exit
    if args.command == "remote-cpu-job":
        success = run_remote_cpu_job(args)["success"] is True
    elif args.command == "gpu-canary":
        if args.terminal_resource_release:
            output = Path(args.terminal_resource_release_output).expanduser().resolve()
            result = dispatch_terminal_resource_release(
                request_path=args.terminal_resource_release,
                state_root=output.parent,
            )
            write_json(output, result)
            success = result.get("status") == "completed"
            print(json.dumps({"success": success}, sort_keys=True))
            return 0 if success else 2
        if args.probe_kind == TASK_EVALUATION_SCENE_CONFIGURATION_PROBE_KIND:
            return run_scene_configuration_allocator_probe(
                args,
                control_plane_identity_probe=_control_plane_checkout_blockers,
                expected_source_commit_probe=_adp_expected_source_commit_blockers,
            )
        if args.probe_kind in {policy_canary_lane.PROBE_KIND, RIGID_ROBOT_FREE_PROBE_KIND}:
            return isolated_native_task_arena_lane(args.probe_kind)(args, _control_plane_checkout_blockers())
        result = {
            "status": "blocked",
            "blockers": (
                [RUNPOD_RETIRED_BLOCKER]
                if args.provider == "runpod"
                else ["digitalocean_gpu_canary_requires_persistent_host_bake"]
            ),
            "provider_mutations_performed": 0,
        }
        write_json(Path(args.adapter_output), result)
        print(json.dumps({"success": False}, sort_keys=True))
        return 2
    # Provider results can contain credential-bearing request fields. Persisted
    # evidence remains in the explicitly selected output paths; stdout exposes
    # only the derived success bit so CI and operators cannot leak the payload.
    print(json.dumps({"success": success}, sort_keys=True))
    return 0 if success else 2


def gpu_canary_main(argv: Sequence[str] | None = None) -> int:
    return main(["gpu-canary", *(list(argv) if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
