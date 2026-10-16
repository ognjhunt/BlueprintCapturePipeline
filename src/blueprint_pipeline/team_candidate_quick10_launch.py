"""Drive one team's scored evaluation of a site job, from admission to its terminal.

Beta journey step: L5 team policy evaluation. Two processes share the work, so
a scored evaluation of 200 episodes (hours of GPU time) never holds the
agent-run dispatcher:

- the **dispatcher** admits the scene, policy, single-use credential and funding,
  then seals one session containing all 200 cells;
- the **runner** uses the canonical paid-resource allocator once for that
  session, with one cumulative cap, TTL and independent watchdog. The provider
  worker starts a fresh simulator process for each cell, checkpoints completed
  child results, and stops after its first 25 cells if most fail at the policy
  interface. A task failure is an observed score, not an interface refusal.

Previously sealed multi-shard jobs remain readable and resumable without
rewriting their inputs or repeating finished attempts.

As before, customer JSON chooses nothing: the operator profile binds the scene;
an execution lock and a sealed intent make a second launch impossible after a
crash; a shard is re-staged (fresh authority, never a second allocation) only
when its allocator result proves nothing was allocated, as when every Vast
launch slot was busy; and every failure releases the run only once
provider-zero is observed. The token reaches each GPU host only through the
provider's private runtime-secret channel and is removed once every shard's
allocator has returned.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

from .checkpoint_policy_credentials import bearer_lease_id
from .common import utc_now_iso
from .controlled_policy_configuration import canonical_request_digest
from .decision_evidence_contracts import canonical_digest
from .droid_https_policy_client import INTERFACE_ID, METADATA_SCHEMA, SUPPORTED_ACTION_CHUNK_ROWS
from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
from .native_task_arena_policy_canary_session import PROBE_KIND, TEAM_RUN_KIND
from .team_candidate_quick10_admission import (
    CANDIDATE_SCHEMA,
    FOUNDER_SCHEMA,
    LEASE_SCHEMA,
    admit_team_candidate,
    validate_candidate_rights,
)
from .team_candidate_quick10_registry import offer_eligibility, profile_for_request, rights_record
from .team_embodiment import ROBOT_ID as TEAM_ROBOT_ID
from .team_embodiment import (
    TeamEmbodimentError,
    read_model_structure,
    team_embodiment_binding,
    validate_embodiment_manifest,
    verify_model_file,
)
from .team_evaluation_plan import (
    EPISODES,
    campaign_ttl_seconds,
    profile_purpose,
    runtime_cells,
    scored_cells,
    shard_ttl_seconds,
)
from .team_policy_https_client import INTERFACE_ID as TEAM_ROBOT_INTERFACE
from .team_policy_https_client import METADATA_KEYS as TEAM_METADATA_KEYS
from .team_policy_https_client import METADATA_SCHEMA as TEAM_METADATA_SCHEMA

TERMINAL_FILENAME = "team_candidate_quick10_terminal.json"
TERMINAL_SCHEMA = "blueprint.team_candidate_quick10_terminal.v1"
INTENT_FILENAME = "team_candidate_quick10_intent.json"
FAILURE_FILENAME = "team_candidate_quick10_failure.json"
REQUEST_FILENAME = "team_evaluation_request.json"
SHARD_FILENAME = "team_evaluation_shard.json"
CREDENTIAL_ENV = "BLUEPRINT_TEAM_POLICY_BEARER_FILE"
QUEUE_ENV = "BLUEPRINT_TEAM_EVALUATION_QUEUE"
DEFAULT_QUEUE = "/var/lib/blueprint/pipeline-control-plane/team-evaluations"
CONCURRENCY_ENV = "BLUEPRINT_TEAM_EVALUATION_SHARD_CONCURRENCY"
# Two of the fleet's three Vast launch slots, so scene preparation keeps one.
SHARD_CONCURRENCY = 2
# A shard refused before any allocation (all slots busy) is re-staged at most this often.
MAX_UNALLOCATED_RESTAGES = 48
POLL_SECONDS = 60
CONTROL_FILENAME = "team_evaluation_control.json"
MAX_PRELAUNCH_ATTEMPTS = 3
MONITOR_UNAVAILABLE_SECONDS = 10 * 60
STOP_TEARDOWN_GRACE_SECONDS = 5 * 60
DISPATCH_HOLD_PATH = Path("/var/lib/blueprint-operator-door/requests/holds/blueprint-agent-run-dispatcher.timer.json")
_METADATA_KEYS = {"schema_version", "policy_id", "policy_interface_id", "action_space", "action_chunk_rows",
                  "model_identity_digest"}
EMBODIMENT_BINDING_FILENAME = "team_embodiment_binding.json"
EMBODIMENT_MODEL_PATH = Path("team_embodiment") / "robot.usd"
MAX_ROBOT_MODEL_BYTES = 512 * 1024 * 1024
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class TeamCandidateLaunchError(ValueError):
    """The run may not launch, or its result may not be reported yet."""


def _m(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TeamCandidateLaunchError("team_candidate_record_invalid")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _endpoint(request: Mapping[str, Any]) -> dict[str, Any]:
    package = _m(request.get("policy_package"))
    present = [name for name in ("policy_api_endpoint", "docker_container", "sim_controller_plugin")
               if package.get(name)]
    return _m(package.get("policy_api_endpoint")) if present == ["policy_api_endpoint"] else {}


def request_embodiment(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """The team's own robot (validated manifest) on ``blueprint_policy_https.v1``, None on DROID; raise if invalid."""

    payload = _endpoint(request)
    if payload.get("policy_interface_id") != TEAM_ROBOT_INTERFACE:
        return None
    return validate_embodiment_manifest(_m(payload.get("robot_embodiment")))


def request_purpose(request: Mapping[str, Any]) -> str | None:
    """``scored`` when the request's execution profile is the team evaluation's, else None."""

    return profile_purpose(str(_endpoint(request).get("execution_profile") or ""))


def team_request_blockers(request: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """Executor preflight for a team request; staging re-checks all of it before spend."""

    payload = _endpoint(request)
    purpose = request_purpose(request)
    url = urlsplit(str(payload.get("endpoint_url") or ""))
    blockers = []
    try:
        request_embodiment(request)
    except TeamEmbodimentError as exc:
        blockers.extend(exc.blockers)
    if (purpose is None or payload.get("policy_interface_id") not in (INTERFACE_ID, TEAM_ROBOT_INTERFACE)
            or payload.get("credential_kind") != "bearer"
            or re.fullmatch(r"policy-credential-[0-9a-f-]{36}", str(payload.get("credential_ref") or "")) is None
            or url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment):
        blockers.append("agent_execution_team_candidate_package_invalid")
    episodes = EPISODES.get(purpose or "")
    if _m(request.get("execution_authorization")).get("episodes") != episodes or row.get("quoted_episodes") != episodes:
        blockers.append("agent_execution_team_candidate_episode_scope_invalid")
    try:
        profile = profile_for_request(request)
    except (OSError, ValueError):
        return [*blockers, "agent_execution_team_candidate_profile_invalid"]
    if profile is None:
        return [*blockers, "agent_execution_team_candidate_profile_missing"]
    if purpose is not None and offer_eligibility(profile, purpose)[1]:
        blockers.append("agent_execution_team_candidate_not_offered")
    return blockers


def _team_metadata(endpoint_url: str, origin: str, token: str, transport: Callable[..., bytes] | None,
                   manifest: Mapping[str, Any] | None = None) -> dict:
    headers = {"Accept": "application/json", "Authorization": "Bearer " + token}
    if transport is None:
        from .core.security_controls import fetch_bounded_https
        raw = fetch_bounded_https(endpoint_url, method="GET", headers=headers, timeout_seconds=30, max_bytes=65_536,
                                  allowed_origins=(origin,), allowed_content_types=("application/json",),
                                  max_redirects=0).body
    else:
        raw = transport(url=endpoint_url, method="GET", body=None, headers=headers)
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise TeamCandidateLaunchError("team_candidate_metadata_invalid") from exc
    if manifest is not None:
        # The team's own robot: the served policy must name the declared robot and its chunk.
        if (not isinstance(value, dict) or set(value) != TEAM_METADATA_KEYS
                or value["schema_version"] != TEAM_METADATA_SCHEMA
                or value["policy_interface_id"] != TEAM_ROBOT_INTERFACE or value["action_space"] != "joint_position"
                or value["embodiment_id"] != manifest["embodiment_id"]
                or value["action_chunk_rows"] != manifest["action_chunk_rows"]
                or not _ID.fullmatch(str(value["policy_id"]))
                or not _DIGEST.fullmatch(str(value["model_identity_digest"]))):
            raise TeamCandidateLaunchError("team_candidate_metadata_invalid")
        return value
    if (not isinstance(value, dict) or set(value) != _METADATA_KEYS or value["schema_version"] != METADATA_SCHEMA
            or value["policy_interface_id"] != INTERFACE_ID or value["action_space"] != "joint_position"
            or value["action_chunk_rows"] not in SUPPORTED_ACTION_CHUNK_ROWS
            or not _ID.fullmatch(str(value["policy_id"])) or not _DIGEST.fullmatch(str(value["model_identity_digest"]))):
        raise TeamCandidateLaunchError("team_candidate_metadata_invalid")
    return value


def _funding(row: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    funding = _m(_m(row.get("execution_admission")).get("funding"))
    try:
        expires = datetime.fromisoformat(str(funding.get("expires_at_iso")).replace("Z", "+00:00"))
        current = expires.tzinfo is not None and expires > datetime.now(timezone.utc)
    except ValueError:
        current = False
    if not current or funding.get("approved_by") not in (profile.get("allowed_founder_approvers") or []):
        raise TeamCandidateLaunchError("team_candidate_founder_funding_approval_not_admitted")
    return funding


def _candidate_records(*, row: Mapping[str, Any], request: Mapping[str, Any], profile: Mapping[str, Any],
                       facts: Mapping[str, Any], funding: Mapping[str, Any], access: Mapping[str, Any],
                       metadata: Mapping[str, Any], pipeline_run_id: str, purpose: str) -> dict[str, Any]:
    payload = _endpoint(request)
    endpoint = str(payload["endpoint_url"])
    origin = f"{urlsplit(endpoint).scheme}://{urlsplit(endpoint).netloc}"
    team_id, checkpoint_id = str(row.get("team_id") or ""), str(_m(row.get("checkpoint")).get("checkpoint_id") or "")
    candidate_id = f"team-{team_id}.{checkpoint_id}"
    rows = int(metadata["action_chunk_rows"])
    manifest = request_embodiment(request)
    candidate = {"schema_version": CANDIDATE_SCHEMA, "team_id": team_id, "checkpoint_id": checkpoint_id,
                 "candidate_id": candidate_id,
                 "embodiment_preset_id": DROID_POLICY_CANARY_PRESET_ID if manifest is None else TEAM_ROBOT_ID,
                 "policy_interface_id": INTERFACE_ID if manifest is None else TEAM_ROBOT_INTERFACE,
                 **({} if manifest is None else {"embodiment": manifest}),
                 "delivery": {"mode": "https_policy", "endpoint_url": endpoint, "metadata_url": endpoint,
                              "action_chunk_rows": rows,
                              # A team's own robot executes its whole declared chunk before the next query.
                              "open_loop_horizon": (min(int(facts["open_loop_horizon"]), rows) if manifest is None
                                                    else rows),
                              "model_identity_digest": metadata["model_identity_digest"],
                              "team_policy_id": metadata["policy_id"]}}
    lease = {"schema_version": LEASE_SCHEMA, "kind": "kms_bearer_lease", "status": "available", "single_use": True,
             "lease_id": bearer_lease_id(credential_ref=access["credential_ref"], job_id=request["job_id"],
                                         pipeline_run_id=pipeline_run_id),
             "credential_ref_digest": canonical_digest({"credential_ref": access["credential_ref"]}),
             "credential_env": CREDENTIAL_ENV, "origin": origin, "candidate_id": candidate_id,
             "secret_material_recorded": False, "expires_at_iso": funding["expires_at_iso"]}
    budget = {"payer": funding.get("payer"), "customer_price_usd": funding.get("customer_price_usd"),
              "max_attempts": funding.get("max_attempts"), "cap_usd": funding.get("cap_usd"),
              "episodes": _m(request.get("execution_authorization")).get("episodes"),
              "hard_ttl_seconds": campaign_ttl_seconds()}
    founder = {"schema_version": FOUNDER_SCHEMA, "approved_by": funding["approved_by"],
               "approval_reference": "webapp_free_beta_funding:" + str(funding.get("approval_digest")),
               "expires_at_iso": funding["expires_at_iso"],
               "scope": {"team_id": team_id, "candidate_id": candidate_id,
                         "scene_revision_digest": facts["scene"]["scene_revision_digest"], "run_kind": TEAM_RUN_KIND},
               "approval_digest": ""}
    founder["approval_digest"] = canonical_digest(founder, digest_field="approval_digest")
    return {"candidate": candidate, "rights": rights_record(profile, team_id=team_id, checkpoint_id=checkpoint_id) or {},
            "credential_lease": lease, "budget": budget, "founder_admission": founder}


def _start_allocator(argv: Sequence[str], *, log_path: Path) -> int:
    """Start the canonical allocator for one shard; return its pid without waiting."""

    stream = log_path.open("x")
    log_path.chmod(0o600)
    process = subprocess.Popen(  # nosec B603 - fixed module, operator-bound argv
        [sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator", *argv],
        cwd=Path(__file__).resolve().parents[2], stdout=stream, stderr=subprocess.STDOUT,
        # A founder stop must reach the adapter's finally/teardown path. The
        # local experiment opt-out must never suppress a team's deliberate stop.
        env={**os.environ, "BLUEPRINT_VAST_IGNORE_LOCAL_SIGTERM_DURING_PROVIDER_RUN": "false"},
        start_new_session=True)
    stream.close()
    return int(process.pid)


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True  # existence without authority to inspect is not proof of exit
    except (OSError, TypeError, ValueError):
        return False
    try:  # reap our own exited child so it is not reported alive as a zombie
        return os.waitpid(int(pid), os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


def _default_provider_zero() -> dict[str, Any]:
    from .task_evaluation_policy_canary_dispatcher import collect_policy_canary_vast_provider_zero
    return collect_policy_canary_vast_provider_zero()


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                   text=True).strip()


def _queue_root() -> Path:
    return Path(os.environ.get(QUEUE_ENV) or DEFAULT_QUEUE)


def _session_cells(*, purpose: str, facts: Mapping[str, Any], profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    scene = facts["scene"]
    task_spec = _m(scene.get("task_spec"))
    generated = scored_cells(scene_revision_digest=str(scene["scene_revision_digest"]),
                             scene_id=str(profile.get("scene_id") or "scene"), salt=str(profile["scored_matrix_salt"]),
                             task_spec=task_spec, task_kind=str(task_spec.get("task_kind") or ""),
                             nominal_dynamic_friction=profile.get("nominal_dynamic_friction"))
    return runtime_cells(generated, control_diagnostic=facts["inputs"]["control_diagnostic"])


def _stage_shard(*, job_dir: Path, index: int, cells: Sequence[Mapping[str, Any]], admission: Mapping[str, Any],
                 facts: Mapping[str, Any], profile: Mapping[str, Any], cap_usd: float, base_run_id: str,
                 commit: str, restage: int = 0) -> dict[str, Any]:
    """Stage one GPU session (inputs, authority, spec, sealed bundle) and record its allocator argv."""

    from .team_candidate_quick10_bundle import build_team_candidate_session_bundle
    from .team_candidate_quick10_staging import stage_team_runtime

    root = job_dir / "shards" / f"{index:02d}" / f"stage-{restage:02d}"
    run_id = f"{base_run_id}-s{index:02d}" + (f"r{restage:02d}" if restage else "")
    binding_path = job_dir / EMBODIMENT_BINDING_FILENAME
    binding = _read(binding_path) if binding_path.is_file() else None
    hz = int(binding["manifest"]["control_hz"]) if binding else 15
    ttl = min(shard_ttl_seconds(len(cells), control_hz=hz), int(admission["budget"]["hard_ttl_seconds"]))
    max_queries = facts["max_policy_queries"]
    if binding is not None:
        # The team's chunk runs whole at its own rate over the scene's time budget.
        plan = _read(Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json")
        steps = int(plan["cadence"]["maximum_action_steps"]) * hz // 15
        max_queries = -(-steps // int(admission["delivery"]["open_loop_horizon"]))
    if facts.get("probe") is None:
        raise TeamCandidateLaunchError("team_evaluation_robot_free_probe_receipt_missing")
    paths = stage_team_runtime(
        root=root, run_id=run_id, scene_inputs=facts["inputs"], admission=admission, prompt=facts["prompt"],
        max_policy_queries=max_queries, hard_cap_usd=cap_usd, hard_ttl_seconds=ttl,
        cells=list(cells), team_embodiment=binding, robot_free_probe=None)
    bundle = build_team_candidate_session_bundle(
        job_dir=root / "bundle", packet_dir=facts["packet_dir"],
        runtime_source_packet_receipt=facts["runtime_source_packet_receipt"],
        runtime_input_manifest_path=paths["runtime_inputs"], session_authority_path=paths["session_authority"],
        execution_spec_path=paths["execution_spec"], implementation_commit=commit,
        team_robot_model_path=job_dir / EMBODIMENT_MODEL_PATH if binding else None)
    argv = ["gpu-canary", "--provider", "vast", "--probe-kind", PROBE_KIND,
            "--native-task-arena-policy-canary-session-authority", str(paths["session_authority"]),
            "--native-task-arena-policy-canary-session-bundle-receipt",
            str(root / "bundle" / "team_candidate_quick10_session_bundle_receipt.v1.json"),
            "--adp-job-dir", str(root / "allocator"),
            "--adp-max-hourly-rate-usd", str(profile["max_hourly_rate_usd"]),
            "--adp-max-spend-usd", str(cap_usd), "--adp-hard-ttl-seconds", str(ttl),
            "--admission-out", str(root / "paid_admission.json"),
            "--adapter-output", str(root / "allocator_result.json"),
            "--team-candidate-credential-file", str(job_dir / ".team-credential" / "bearer"), "--execute"]
    if profile.get("machine_avoidlist_path"):
        argv.extend(["--adp-machine-avoidlist", str(profile["machine_avoidlist_path"])])
    shard = {"index": index, "restage": restage, "run_id": run_id, "root": str(root),
             "bundle_path": str(bundle["bundle_path"]), "cell_count": len(cells),
             "cells": [dict(cell) for cell in cells], "cap_usd": cap_usd, "hard_ttl_seconds": ttl, "argv": argv}
    _write(root / SHARD_FILENAME, shard)
    return shard


def _stage(*, row: Mapping[str, Any], request: Mapping[str, Any], job_dir: Path, pipeline_run_id: str,
           hooks: Mapping[str, Any]) -> None:
    profile = profile_for_request(request)
    purpose = request_purpose(request)
    if purpose is None:
        raise TeamCandidateLaunchError("agent_execution_team_candidate_package_invalid")
    facts, blockers = offer_eligibility(profile, purpose)
    if facts is None or profile is None:
        raise TeamCandidateLaunchError(";".join(blockers) or "team_candidate_profile_missing")
    funding = _funding(row, profile)
    payload = _endpoint(request)
    team_id = str(row.get("team_id") or "")
    checkpoint_id = str(_m(row.get("checkpoint")).get("checkpoint_id") or "")
    validate_candidate_rights(rights_record(profile, team_id=team_id, checkpoint_id=checkpoint_id) or {},
                              team_id=team_id, checkpoint_id=checkpoint_id,
                              candidate_id=f"team-{team_id}.{checkpoint_id}")
    manifest = request_embodiment(request)
    if manifest is not None:
        _stage_robot_model(job_dir=job_dir, manifest=manifest, facts=facts, fetch=hooks.get("model_fetch"))
    zero = dict(hooks["provider_zero"]())
    _write(job_dir / "initial_provider_zero.json", zero)
    if zero.get("provider_zero_verified") is not True:
        raise TeamCandidateLaunchError("team_candidate_initial_provider_zero_unproven")
    # All checks that do not need the authenticated team service precede the
    # release. Its token is then retained privately for a bounded prelaunch
    # repair, never released a second time and never bundled.
    access = _credential_access(request=request, payload=payload, pipeline_run_id=pipeline_run_id,
                                job_dir=job_dir, credential_client=hooks["credential_client"])
    credential = _m(access.get("credential"))
    token = credential.get("bearer_token")
    if (access.get("kind") != "bearer" or credential.get("endpoint_url") != payload["endpoint_url"]
            or credential.get("job_id") != request["job_id"] or not isinstance(token, str)
            or not 16 <= len(token.strip()) <= 4096 or any(character in token for character in "\r\n\x00")):
        raise TeamCandidateLaunchError("team_candidate_bearer_credential_invalid")
    token = token.strip()
    endpoint = str(payload["endpoint_url"])
    origin = f"{urlsplit(endpoint).scheme}://{urlsplit(endpoint).netloc}"
    metadata = _team_metadata(endpoint, origin, token, hooks.get("metadata_transport"), manifest)
    records = _candidate_records(row=row, request=request, profile=profile, facts=facts, funding=funding,
                                 access=access, metadata=metadata, pipeline_run_id=pipeline_run_id, purpose=purpose)
    admission = admit_team_candidate(**records, scene=facts["scene"], probe_receipt=facts.get("probe_receipt"),
                                     purpose=purpose)
    _write(job_dir / "team_candidate_admission.json", admission)
    _write(job_dir / REQUEST_FILENAME, dict(request))
    commit = hooks.get("implementation_commit") or _source_commit()
    base_run_id = f"team-{purpose}-" + canonical_digest({"job_id": request["job_id"], "run": pipeline_run_id})[7:31]
    cells = _session_cells(purpose=purpose, facts=facts, profile=profile)
    # One paid session owns the whole matrix, its cumulative cap and watchdog.
    # The provider worker preserves per-cell isolation and the first-25 gate.
    groups = [cells]
    total_cap = float(funding["cap_usd"])
    staged = [_stage_shard(job_dir=job_dir, index=index, cells=group, admission=admission, facts=facts,
                           profile=profile, cap_usd=round(total_cap * len(group) / len(cells), 2),
                           base_run_id=base_run_id, commit=commit) for index, group in enumerate(groups)]
    from .team_candidate_quick10_bundle import assert_secret_absent
    for shard in staged:
        assert_secret_absent(shard["bundle_path"], token)
    _write(job_dir / CONTROL_FILENAME, {"run_id": row["run_id"], "pipeline_run_id": pipeline_run_id,
                                        "execution_admission_digest": row["execution_admission_digest"]})
    _write(job_dir / INTENT_FILENAME, {"job_id": request["job_id"], "canonical_request_digest":
                                        canonical_request_digest(request), "pipeline_run_id": pipeline_run_id,
                                        "admission_digest": admission["admission_digest"], "run_id": base_run_id,
                                        "evaluation_purpose": purpose, "episodes": len(cells),
                                        "shard_count": len(staged), "source_commit": commit,
                                        "observed_at_iso": utc_now_iso()})
    queue = Path(hooks.get("queue_root") or _queue_root()) / "pending"
    _write(queue / f"{request['job_id']}.json", {"job_dir": str(job_dir), "queued_at_iso": utc_now_iso()})


def _credential_access(*, request: Mapping[str, Any], payload: Mapping[str, Any], pipeline_run_id: str,
                       job_dir: Path, credential_client: Callable[..., Any]) -> dict[str, Any]:
    """Persist one secret delivery privately, so prelaunch repair does not consume a new lease."""
    from .team_candidate_quick10_vast import credential_file_blockers

    root = job_dir / ".team-credential"
    root.mkdir(mode=0o700, exist_ok=True)
    path = root / "delivery.json"
    binding = {"job_id": request["job_id"], "pipeline_run_id": pipeline_run_id,
               "canonical_request_digest": canonical_request_digest(request)}
    if path.exists():
        if credential_file_blockers(path):
            raise TeamCandidateLaunchError("team_candidate_private_delivery_invalid")
        retained = _read(path)
        if any(retained.get(key) != value for key, value in binding.items()):
            raise TeamCandidateLaunchError("team_candidate_private_delivery_binding_mismatch")
        access = _m(retained.get("access"))
    else:
        lease_id = bearer_lease_id(credential_ref=str(payload["credential_ref"]), job_id=request["job_id"],
                                   pipeline_run_id=pipeline_run_id)
        access = credential_client(request=request, payload=payload,
                                    pipeline_run_id=pipeline_run_id).release_bearer_lease(lease_id=lease_id)
        _write(path, {**binding, "access": access})
    token = _m(access.get("credential")).get("bearer_token")
    if not isinstance(token, str) or not 16 <= len(token.strip()) <= 4096 or any(c in token for c in "\r\n\x00"):
        raise TeamCandidateLaunchError("team_candidate_bearer_credential_invalid")
    bearer = root / "bearer"
    if not bearer.exists():
        with bearer.open("x", encoding="utf-8") as stream:
            bearer.chmod(0o600)
            stream.write(token.strip() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    elif credential_file_blockers(bearer) or bearer.read_text().strip() != token.strip():
        raise TeamCandidateLaunchError("team_candidate_private_delivery_binding_mismatch")
    return access


def _fetch_model(url: str, target: Path) -> None:
    from .core.security_controls import fetch_bounded_https

    origin = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
    fetch_bounded_https(url, method="GET", headers={"Accept": "*/*"}, timeout_seconds=300,
                        max_bytes=MAX_ROBOT_MODEL_BYTES, allowed_origins=(origin,), max_redirects=0,
                        output_path=target)


def _stage_robot_model(*, job_dir: Path, manifest: Mapping[str, Any], facts: Mapping[str, Any],
                       fetch: Callable[[str, Path], None] | None) -> dict[str, Any]:
    """Fetch the team's robot USD once, verify it is the declared file, and seal its binding to the scene."""

    target = job_dir / EMBODIMENT_MODEL_PATH
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    (fetch or _fetch_model)(str(manifest["model"]["url"]), target)
    record = verify_model_file(manifest, target)
    plan = _read(Path(facts["packet_dir"]) / "native_task_arena_scene_plan.v1.json")
    stance = None
    if manifest["base"]["mode"] == "free":
        from .team_embodiment_stance import free_base_stance

        stance = free_base_stance(manifest, scene_plan=plan, packet_dir=Path(facts["packet_dir"]))
    binding = team_embodiment_binding(manifest, model_record=record, structure=read_model_structure(target),
                                      scene_plan=plan, stance=stance)
    _write(job_dir / EMBODIMENT_BINDING_FILENAME, binding)
    return binding


def _current_shard(job_dir: Path, index: int) -> dict[str, Any]:
    stages = sorted((job_dir / "shards" / f"{index:02d}").glob(f"stage-*/{SHARD_FILENAME}"))
    if not stages:
        raise TeamCandidateLaunchError("team_evaluation_shard_missing")
    return _read(stages[-1])


def _shard_state(shard: Mapping[str, Any]) -> str:
    root = Path(shard["root"])
    if (root / "allocator_result.json").is_file():
        try:
            result = _read(root / "allocator_result.json")
            return "returned" if isinstance(result.get("status"), str) and result["status"] else "unknown"
        except (OSError, ValueError):
            return "unknown"
    process = root / "allocator_process.json"
    if not process.is_file():
        # A crash after spawn but before recording its pid is not permission
        # to allocate again. Provider reconciliation must close that attempt.
        return "unknown" if (root / "allocator_launch_intent.json").is_file() else "staged"
    try:
        recorded = _read(process)
        pid = int(recorded["pid"])
        if pid <= 0:
            return "unknown"
        if not _alive(pid):
            return "lost"
        current = _process_start_ticks(pid)
        if current is None or not recorded.get("start_ticks"):
            return "unknown"
        return "running" if current == recorded["start_ticks"] else "lost"
    except (OSError, ValueError, KeyError, TypeError):
        return "unknown"


def _process_start_ticks(pid: int) -> str | None:
    """Linux process identity prevents a reused pid from receiving a founder stop."""
    try:
        # comm may contain spaces and parentheses; fields after its final ')' start at field 3.
        return Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def _stop_allocator(shard: Mapping[str, Any]) -> None:
    """Ask only this recorded allocator to unwind through its provider teardown."""
    path = Path(shard["root"]) / "allocator_process.json"
    process = _read(path)
    pid = int(process["pid"])
    if not process.get("start_ticks") or _process_start_ticks(pid) != process["start_ticks"]:
        raise TeamCandidateLaunchError("team_evaluation_allocator_stop_identity_unproven")
    os.kill(pid, signal.SIGTERM)


def _execution_monitor(control: Mapping[str, Any]) -> dict[str, Any]:
    """Signed liveness/cancellation check, independent of the dispatcher process."""
    from .agent_run_executor import AgentRunWebAppClient
    from .task_evaluation_launch_webapp_sync import load_pipeline_sync_token

    endpoint = os.environ.get("BLUEPRINT_WEBAPP_URL", "")
    if not endpoint.startswith("https://"):
        raise TeamCandidateLaunchError("team_evaluation_control_service_unconfigured")
    token = load_pipeline_sync_token(token_file_path=Path(os.environ.get("BLUEPRINT_PIPELINE_SYNC_TOKEN_FILE", "")),
                                     require_file=True)
    client = AgentRunWebAppClient(base_url=endpoint, token=token, timeout_seconds=30)
    try:
        return client.heartbeat(str(control["run_id"]), str(control["pipeline_run_id"]),
                                 str(control["execution_admission_digest"]))
    except HTTPError as exc:
        if exc.code == 409:
            return {"cancellation_requested": True, "reason": "team_evaluation_control_heartbeat_refused"}
        raise


def _control_stop_reason(control: Mapping[str, Any], *, monitor: Callable[..., Any],
                         hold_path: Path) -> str | None:
    if hold_path.exists():
        return "team_evaluation_founder_dispatch_hold"
    observed = monitor(control)
    if observed.get("cancellation_requested") is True:
        return "team_evaluation_cancellation_requested"
    deadline = observed.get("execution_deadline_at_ms")
    if type(deadline) not in (int, float) or deadline <= time.time() * 1000:
        return "team_evaluation_execution_deadline_missing_or_expired"
    if observed.get("ok") is not True or observed.get("run_id") != control["run_id"]:
        raise TeamCandidateLaunchError("team_evaluation_control_response_invalid")
    return None


def run_staged_evaluation(*, job_dir: Path, hooks: Mapping[str, Any] | None = None,
                          concurrency: int | None = None, poll_seconds: float = POLL_SECONDS) -> dict[str, Any] | None:
    """Run every staged shard (bounded concurrency), then seal the terminal; idempotent after a crash."""

    from .task_evaluation_scene_execution_authority import proves_no_provider_allocation
    from .team_candidate_quick10_closeout import (
        allocator_running,
        seal_blocked_terminal,
        seal_team_candidate_terminal,
    )

    hooks = {"allocator_starter": _start_allocator, "provider_zero": _default_provider_zero,
             "restage": _restage_shard, "sleep": time.sleep, "execution_monitor": _execution_monitor,
             "allocator_stopper": _stop_allocator, "monotonic": time.monotonic,
             "dispatch_hold_path": DISPATCH_HOLD_PATH, **dict(hooks or {})}
    limit = max(1, min(int(concurrency or os.environ.get(CONCURRENCY_ENV) or SHARD_CONCURRENCY), 3))
    request = _read(job_dir / REQUEST_FILENAME)
    if (job_dir / TERMINAL_FILENAME).exists():
        return _read(job_dir / TERMINAL_FILENAME)
    intent = _read(job_dir / INTENT_FILENAME)
    last_contact = hooks["monotonic"]()
    stopping_since = None
    stop_reason = _read(job_dir / "stop_requested.json").get("reason") if (job_dir / "stop_requested.json").exists() else None
    try:
        if not (job_dir / CONTROL_FILENAME).is_file():
            raise TeamCandidateLaunchError("team_evaluation_control_binding_missing")
        control = _read(job_dir / CONTROL_FILENAME)
        while True:
            control_unavailable = False
            if stop_reason is None:
                try:
                    stop_reason = _control_stop_reason(control, monitor=hooks["execution_monitor"],
                                                       hold_path=Path(hooks["dispatch_hold_path"]))
                    last_contact = hooks["monotonic"]()
                except (OSError, ValueError):
                    control_unavailable = True
                    if hooks["monotonic"]() - last_contact >= MONITOR_UNAVAILABLE_SECONDS:
                        stop_reason = "team_evaluation_control_unavailable_timeout"
                if stop_reason:
                    _write(job_dir / "stop_requested.json", {"reason": stop_reason, "observed_at_iso": utc_now_iso()})
            shard_rows = [_current_shard(job_dir, index) for index in range(int(intent["shard_count"]))]
            states = [_shard_state(row) for row in shard_rows]
            if stop_reason:
                for row, state in zip(shard_rows, states, strict=True):
                    stop_path = Path(row["root"]) / "allocator_stop_requested.json"
                    if state == "running" and not stop_path.exists():
                        hooks["allocator_stopper"](row)
                        _write(stop_path, {"reason": stop_reason, "observed_at_iso": utc_now_iso()})
                if not any(state in {"running", "unknown"} for state in states):
                    raise TeamCandidateLaunchError(stop_reason)
                if stopping_since is None:
                    stopping_since = hooks["monotonic"]()
                if hooks["monotonic"]() - stopping_since >= STOP_TEARDOWN_GRACE_SECONDS:
                    # Never kill the adapter past its teardown handler or claim
                    # provider zero. Independent provider TTL/watchdogs remain
                    # responsible while a later queue pass reconciles it.
                    pending = job_dir / "stop_teardown_pending.json"
                    if not pending.exists():
                        _write(pending, {"status": "pending_provider_teardown", "reason": stop_reason,
                                         "observed_at_iso": utc_now_iso()})
                    return None
                hooks["sleep"](poll_seconds)
                continue
            if "unknown" in states:
                # A spawned allocator may not yet have recorded its PID or
                # allocated its provider instance. A global zero snapshot
                # cannot close that window. Wait for its actual result or
                # verified process reconciliation, without starting new work.
                pending = job_dir / "allocator_reconciliation_pending.json"
                if not pending.exists():
                    _write(pending, {"status": "allocator_identity_unknown",
                        "shard_indices": [row["index"] for row, state in zip(shard_rows, states, strict=True)
                                          if state == "unknown"], "observed_at_iso": utc_now_iso()})
                hooks["sleep"](poll_seconds)
                continue
            for row, state in zip(shard_rows, states, strict=True):
                if state == "lost":
                    raise TeamCandidateLaunchError("team_evaluation_shard_allocator_lost:" + str(row["index"]))
                if state == "returned":
                    result = _read(Path(row["root"]) / "allocator_result.json")
                    if proves_no_provider_allocation(result) and row["restage"] < MAX_UNALLOCATED_RESTAGES:
                        hooks["restage"](job_dir=job_dir, shard=row)
                        states = None  # re-read the new stage next round
                        break
            if states is None:
                continue
            # The first shard runs alone; the rest start only once its gate passed.
            opened = states[0] == "returned" and first_shard_gate(job_dir=job_dir, request=request,
                                                                   shard=shard_rows[0])
            running = states.count("running")
            for row, state in zip(shard_rows, states, strict=True):
                if not control_unavailable and state == "staged" and running < limit and (row["index"] == 0 or opened):
                    root = Path(row["root"])
                    _write(root / "allocator_launch_intent.json", {"observed_at_iso": utc_now_iso()})
                    pid = hooks["allocator_starter"](row["argv"], log_path=root / "allocator.log")
                    _write(root / "allocator_process.json", {"pid": pid, "start_ticks": _process_start_ticks(pid),
                                                             "started_at_iso": utc_now_iso()})
                    running += 1
            if all(_shard_state(row) == "returned" for row in shard_rows):
                # A fast allocator can return during the launch pass. With a
                # single session there is no later shard forcing another poll.
                first_shard_gate(job_dir=job_dir, request=request, shard=shard_rows[0])
                break
            hooks["sleep"](poll_seconds)
    except Exception as exc:  # noqa: BLE001 - sealed blocked, released only at provider zero
        seal_blocked_terminal(request=request, job_dir=job_dir, error=exc, provider_zero=hooks["provider_zero"])
        return None
    finally:
        if not allocator_running(job_dir):
            shutil.rmtree(job_dir / ".team-credential", ignore_errors=True)
    try:
        return seal_team_candidate_terminal(request=request, job_dir=job_dir, provider_zero=hooks["provider_zero"])
    except Exception as exc:  # noqa: BLE001 - every failure is sealed, and released only at provider zero
        seal_blocked_terminal(request=request, job_dir=job_dir, error=exc, provider_zero=hooks["provider_zero"])
        return None


FIRST_SHARD_GATE_FILENAME = "first_shard_gate.json"


def first_shard_gate(*, job_dir: Path, request: Mapping[str, Any], shard: Mapping[str, Any]) -> bool:
    """True once the first shard's result opens the other shards; raise to stop the evaluation.

    It stops when the shard was never allocated, when the scene's robot-free
    check recorded in its first cell did not pass (the probe is retained, so
    the scene is closed to every team), or when most of its episodes failed at
    the interface rather than the task. The team learns why from the blocker.
    """

    from .task_evaluation_scene_execution_authority import proves_no_provider_allocation
    from .team_candidate_quick10_closeout import _retain_probe
    from .team_candidate_quick10_result import episode_outcome
    from .team_evaluation_plan import FIRST_SHARD_INTERFACE_FAILURE_SHARE, INTERFACE_FAILURE_REASONS

    path = job_dir / FIRST_SHARD_GATE_FILENAME
    if path.is_file():
        return True
    adapter = _read(Path(shard["root"]) / "allocator_result.json")
    if proves_no_provider_allocation(adapter):
        raise TeamCandidateLaunchError("team_evaluation_shard_not_allocated:00")
    native = Path(str(adapter.get("native_control_result_path") or ""))
    inner = _read(native) if native.is_file() else {}
    if not inner or inner.get("result_digest") != adapter.get("native_control_result_digest"):
        raise TeamCandidateLaunchError("team_candidate_provider_result_missing")
    if isinstance(inner.get("robot_free_probe_trace"), Mapping):
        _retain_probe(job_dir, request, inner["robot_free_probe_trace"])
        if _read(job_dir / "robot_free_probe_retention.json").get("receipt_status") != "passed":
            raise TeamCandidateLaunchError("team_scene_robot_free_probe_not_passed")
    rows = [row for row in inner.get("episodes") or [] if isinstance(row, Mapping)][:25]
    if not rows:
        raise TeamCandidateLaunchError(";".join(map(str, inner.get("blockers") or [])) or "team_evaluation_shard_blocked")
    reasons: dict[str, int] = {}
    for row in rows:
        _, reason = episode_outcome(row)
        if reason in INTERFACE_FAILURE_REASONS:
            reasons[reason] = reasons.get(reason, 0) + 1
    if sum(reasons.values()) > FIRST_SHARD_INTERFACE_FAILURE_SHARE * len(rows):
        raise TeamCandidateLaunchError(f"team_policy_interface_failures:{len(rows)}_episodes:" + ",".join(
            f"{name}={count}" for name, count in sorted(reasons.items())))
    _write(path, {"status": "passed", "episodes": len(rows), "interface_failures": reasons,
                  "observed_at_iso": utc_now_iso()})
    return True


def _restage_shard(*, job_dir: Path, shard: Mapping[str, Any]) -> dict[str, Any]:
    """A shard refused before any allocation gets a fresh authority; it was never allocated."""

    from .team_candidate_quick10_admission import validate_admission

    admission = validate_admission(_read(job_dir / "team_candidate_admission.json"))
    request = _read(job_dir / REQUEST_FILENAME)
    intent = _read(job_dir / INTENT_FILENAME)
    profile = profile_for_request(request)
    facts, blockers = offer_eligibility(profile, intent["evaluation_purpose"])
    if facts is None or profile is None:
        raise TeamCandidateLaunchError(";".join(blockers) or "team_candidate_profile_missing")
    return _stage_shard(job_dir=job_dir, index=int(shard["index"]), cells=shard["cells"], admission=admission,
                        facts=facts, profile=profile, cap_usd=float(shard["cap_usd"]), base_run_id=intent["run_id"],
                        commit=intent["source_commit"], restage=int(shard["restage"]) + 1)


def execute_staged_team_candidate_request(*, row: Mapping[str, Any], request: Mapping[str, Any], job_dir: Path,
                                          pipeline_run_id: str | None, **overrides: Any) -> None:
    """Stage and queue at most once; the runner owns execution. A failure is released only at provider-zero."""

    from .checkpoint_policy_credentials import CheckpointPolicyCredentialClient
    from .team_candidate_quick10_closeout import seal_blocked_terminal

    hooks = {"credential_client": CheckpointPolicyCredentialClient, "provider_zero": _default_provider_zero,
             **overrides}
    job_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if (job_dir / TERMINAL_FILENAME).exists():
        return
    with (job_dir / "team_candidate_quick10.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        try:
            if (job_dir / INTENT_FILENAME).exists():
                # Recover the local crash window after sealing the intent but
                # before queueing. No new staging or allocation is performed.
                pending = Path(hooks.get("queue_root") or _queue_root()) / "pending" / f"{request['job_id']}.json"
                if not pending.exists():
                    _write(pending, {"job_dir": str(job_dir), "queued_at_iso": utc_now_iso()})
                return
            if not isinstance(pipeline_run_id, str) or not pipeline_run_id:
                raise TeamCandidateLaunchError("team_candidate_dispatch_owner_missing")
            attempt = len(list(job_dir.glob("preflight_attempt_*.json"))) + 1
            # Failed prelaunch artifacts are retained separately. They have no
            # allocation authority and cannot be mistaken for completed work.
            archive = job_dir / "preflight_attempts" / f"{attempt - 1:02d}"
            for name in ("shards", "team_embodiment", EMBODIMENT_BINDING_FILENAME, "team_candidate_admission.json",
                         REQUEST_FILENAME, CONTROL_FILENAME, "initial_provider_zero.json"):
                source = job_dir / name
                if source.exists():
                    archive.mkdir(mode=0o700, parents=True, exist_ok=True)
                    source.rename(archive / name)
            _stage(row=row, request=request, job_dir=job_dir, pipeline_run_id=pipeline_run_id, hooks=hooks)
        except Exception as exc:  # noqa: BLE001 - every failure is sealed, and released only at provider zero
            if ((job_dir / ".team-credential" / "delivery.json").is_file()
                    and not (job_dir / INTENT_FILENAME).exists()
                    and str(exc) not in {"team_candidate_secret_in_bundle", "team_candidate_bearer_credential_invalid",
                                         "team_candidate_private_delivery_binding_mismatch"}):
                # The one paid attempt has not started. A corrected metadata
                # endpoint or packaging input may replay complete preflight
                # using this private delivery, without another token release.
                attempt = len(list(job_dir.glob("preflight_attempt_*.json"))) + 1
                _write(job_dir / f"preflight_attempt_{attempt:02d}.json",
                       {"status": "refused_before_allocation", "error_class": type(exc).__name__,
                        "retry_remaining": max(0, MAX_PRELAUNCH_ATTEMPTS - attempt), "observed_at_iso": utc_now_iso()})
                if attempt < MAX_PRELAUNCH_ATTEMPTS:
                    return
            shutil.rmtree(job_dir / ".team-credential", ignore_errors=True)
            seal_blocked_terminal(request=request, job_dir=job_dir, error=exc, provider_zero=hooks["provider_zero"])


def cancel_unlaunched_team_candidate_request(*, request: Mapping[str, Any], job_dir: Path,
                                            reason: str = "team_evaluation_cancellation_requested") -> None:
    """Remove a retained prelaunch delivery when the claimed run is definitely cancelled."""
    from .team_candidate_quick10_closeout import seal_blocked_terminal

    if not job_dir.is_dir() or (job_dir / INTENT_FILENAME).exists():
        return  # an intent belongs to the independent runner's teardown path
    with (job_dir / "team_candidate_quick10.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        shutil.rmtree(job_dir / ".team-credential", ignore_errors=True)
        seal_blocked_terminal(request=request, job_dir=job_dir,
            error=TeamCandidateLaunchError(reason), provider_zero=_default_provider_zero)


def run_queue(queue_root: Path | None = None, *, hooks: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run every queued team evaluation to its terminal (the path-activated runner's entrypoint)."""

    root = Path(queue_root or _queue_root())
    rows = []
    for item in sorted((root / "pending").glob("*.json")):
        job_dir = Path(_read(item)["job_dir"])
        with (job_dir / "team_evaluation_runner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            terminal = run_staged_evaluation(job_dir=job_dir, hooks=hooks)
            if (job_dir / TERMINAL_FILENAME).exists():
                (root / "done").mkdir(parents=True, exist_ok=True, mode=0o700)
                os.replace(item, root / "done" / item.name)
            rows.append({"job_dir": str(job_dir), "status": (terminal or {}).get("status", "pending")})
    return rows


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run queued team evaluations.")
    parser.add_argument("command", choices=["run-queue"])
    parser.add_argument("--queue-root", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(run_queue(args.queue_root), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CREDENTIAL_ENV",
    "FAILURE_FILENAME",
    "INTENT_FILENAME",
    "QUEUE_ENV",
    "REQUEST_FILENAME",
    "TERMINAL_FILENAME",
    "TERMINAL_SCHEMA",
    "TeamCandidateLaunchError",
    "execute_staged_team_candidate_request",
    "first_shard_gate",
    "request_purpose",
    "run_queue",
    "run_staged_evaluation",
    "team_request_blockers",
]
