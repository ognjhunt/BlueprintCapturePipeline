"""Drive one claimed team-candidate Quick-10 run from the agent-run executor to its terminal.

Beta journey step: L5 team policy evaluation. Mirrors the controlled native
queue: the operator registry binds the scene, customer JSON chooses nothing,
an execution lock and a sealed intent make a second allocation impossible after
a crash, and a failure releases the run only after provider-zero is observed.

The steps, each failing closed with a typed reason:

1. the registry's eligibility rule still holds (robot-free probe passed at the
   current revision, DROID Franka Quick-10 cells, no strict controls);
2. the free-beta funding capsule is current, within the cap, single-attempt,
   and approved by an approver the operator registry names;
3. the job-bound bearer credential is claimed through the signed WebApp
   control plane (``CheckpointPolicyCredentialClient``, as the controlled
   native queue does) for the exact endpoint the request names;
4. the team's declared metadata is read once over the bounded HTTPS boundary
   (one origin, no redirects) to fix the policy id, chunk shape and model
   identity the run will check on every episode;
5. ``admit_team_candidate`` seals the admission (rights, lease, budget,
   founder approval, probe, interface), then the team runtime inputs, session
   authority and execution spec are staged and the bundle is sealed; the token
   must be absent from the bundle bytes;
6. provider-zero is observed, the token is written to a private 0600 file and
   handed to ``paid_resource_allocator gpu-canary`` (the canonical allocator),
   which passes it to the provider only through the runtime-secret channel;
   the file is removed as soon as the allocator returns;
7. closeout joins teardown, staged-object absence and a fresh global
   provider-zero to the session result, and only a closed result becomes the
   ``team_candidate_terminal`` receipt the executor reports, with the scoring
   statement the WebApp admits.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
)
from .team_candidate_quick10_registry import (
    EPISODES,
    PROFILE,
    offer_eligibility,
    profile_for_request,
    rights_record,
)

TERMINAL_FILENAME = "team_candidate_quick10_terminal.json"
TERMINAL_SCHEMA = "blueprint.team_candidate_quick10_terminal.v1"
INTENT_FILENAME = "team_candidate_quick10_intent.json"
FAILURE_FILENAME = "team_candidate_quick10_failure.json"
CREDENTIAL_ENV = "BLUEPRINT_TEAM_POLICY_BEARER_FILE"
_METADATA_KEYS = {"schema_version", "policy_id", "policy_interface_id", "action_space", "action_chunk_rows",
                  "model_identity_digest"}
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


def team_request_blockers(request: Mapping[str, Any], row: Mapping[str, Any]) -> list[str]:
    """Executor preflight for a team request; the launch re-checks all of it before spend."""

    payload = _endpoint(request)
    url = urlsplit(str(payload.get("endpoint_url") or ""))
    blockers = []
    if (payload.get("execution_profile") != PROFILE or payload.get("policy_interface_id") != INTERFACE_ID
            or payload.get("credential_kind") != "bearer"
            or re.fullmatch(r"policy-credential-[0-9a-f-]{36}", str(payload.get("credential_ref") or "")) is None
            or url.scheme != "https" or not url.hostname or url.username or url.password or url.fragment):
        blockers.append("agent_execution_team_candidate_package_invalid")
    if _m(request.get("execution_authorization")).get("episodes") != EPISODES or row.get("quoted_episodes") != EPISODES:
        blockers.append("agent_execution_team_candidate_episode_scope_invalid")
    try:
        profile = profile_for_request(request)
    except (OSError, ValueError):
        return [*blockers, "agent_execution_team_candidate_profile_invalid"]
    if profile is None:
        return [*blockers, "agent_execution_team_candidate_profile_missing"]
    if offer_eligibility(profile)[1]:
        blockers.append("agent_execution_team_candidate_not_offered")
    return blockers


def _team_metadata(endpoint_url: str, origin: str, token: str, transport: Callable[..., bytes] | None) -> dict:
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
                       metadata: Mapping[str, Any], pipeline_run_id: str) -> dict[str, Any]:
    payload = _endpoint(request)
    endpoint = str(payload["endpoint_url"])
    origin = f"{urlsplit(endpoint).scheme}://{urlsplit(endpoint).netloc}"
    team_id, checkpoint_id = str(row.get("team_id") or ""), str(_m(row.get("checkpoint")).get("checkpoint_id") or "")
    candidate_id = f"team-{team_id}.{checkpoint_id}"
    rows = int(metadata["action_chunk_rows"])
    candidate = {"schema_version": CANDIDATE_SCHEMA, "team_id": team_id, "checkpoint_id": checkpoint_id,
                 "candidate_id": candidate_id, "embodiment_preset_id": DROID_POLICY_CANARY_PRESET_ID,
                 "policy_interface_id": INTERFACE_ID,
                 "delivery": {"mode": "https_policy", "endpoint_url": endpoint, "metadata_url": endpoint,
                              "action_chunk_rows": rows, "open_loop_horizon": min(int(facts["open_loop_horizon"]), rows),
                              "model_identity_digest": metadata["model_identity_digest"],
                              "team_policy_id": metadata["policy_id"]}}
    lease = {"schema_version": LEASE_SCHEMA, "kind": "kms_bearer_lease", "status": "available", "single_use": True,
             "lease_id": "policy-access-" + canonical_digest({"credential_ref": access["credential_ref"],
                                                              "job_id": request["job_id"],
                                                              "pipeline_run_id": pipeline_run_id})[7:39],
             "credential_ref_digest": canonical_digest({"credential_ref": access["credential_ref"]}),
             "credential_env": CREDENTIAL_ENV, "origin": origin, "candidate_id": candidate_id,
             "secret_material_recorded": False, "expires_at_iso": funding["expires_at_iso"]}
    budget = {"payer": funding.get("payer"), "customer_price_usd": funding.get("customer_price_usd"),
              "max_attempts": funding.get("max_attempts"), "cap_usd": funding.get("cap_usd"),
              "episodes": _m(request.get("execution_authorization")).get("episodes"),
              "hard_ttl_seconds": profile.get("hard_ttl_seconds")}
    founder = {"schema_version": FOUNDER_SCHEMA, "approved_by": funding["approved_by"],
               "approval_reference": "webapp_free_beta_funding:" + str(funding.get("approval_digest")),
               "expires_at_iso": funding["expires_at_iso"],
               "scope": {"team_id": team_id, "candidate_id": candidate_id,
                         "scene_revision_digest": facts["scene"]["scene_revision_digest"], "run_kind": TEAM_RUN_KIND},
               "approval_digest": ""}
    founder["approval_digest"] = canonical_digest(founder, digest_field="approval_digest")
    return {"candidate": candidate, "rights": rights_record(profile, team_id=team_id, checkpoint_id=checkpoint_id) or {},
            "credential_lease": lease, "budget": budget, "founder_admission": founder}


def _default_allocator_runner(argv: Sequence[str], *, log_path: Path, timeout_seconds: int) -> int:
    with log_path.open("x") as stream:
        log_path.chmod(0o600)
        process = subprocess.Popen(  # nosec B603 - fixed module, operator-bound argv
            [sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator", *argv],
            cwd=Path(__file__).resolve().parents[2], stdout=stream, stderr=subprocess.STDOUT)
        # A crashed caller must not release the run while this process can still allocate.
        _write(log_path.parent / "allocator_process.json", {"pid": process.pid, "started_at_iso": utc_now_iso()})
        try:
            return int(process.wait(timeout=timeout_seconds))
        except subprocess.TimeoutExpired:
            process.terminate()
            return int(process.wait(timeout=60))


def _default_provider_zero() -> dict[str, Any]:
    from .task_evaluation_policy_canary_dispatcher import collect_policy_canary_vast_provider_zero
    return collect_policy_canary_vast_provider_zero()


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                                   text=True).strip()


def _launch(*, row: Mapping[str, Any], request: Mapping[str, Any], job_dir: Path, pipeline_run_id: str,
            hooks: Mapping[str, Any]) -> None:
    from .team_candidate_quick10_bundle import assert_secret_absent, build_team_candidate_session_bundle
    from .team_candidate_quick10_closeout import seal_team_candidate_terminal
    from .team_candidate_quick10_staging import stage_team_runtime

    profile = profile_for_request(request)
    facts, blockers = offer_eligibility(profile)
    if facts is None or profile is None:
        raise TeamCandidateLaunchError(";".join(blockers) or "team_candidate_profile_missing")
    funding = _funding(row, profile)
    payload = _endpoint(request)
    if payload.get("execution_profile") != PROFILE:
        raise TeamCandidateLaunchError("agent_execution_team_candidate_package_invalid")
    access = hooks["credential_client"](request=request, payload=payload, pipeline_run_id=pipeline_run_id).access()
    credential = _m(access.get("credential"))
    token = credential.get("bearer_token")
    if (access.get("kind") != "bearer" or credential.get("endpoint_url") != payload["endpoint_url"]
            or credential.get("job_id") != request["job_id"] or not isinstance(token, str)
            or not 16 <= len(token.strip()) <= 4096 or any(character in token for character in "\r\n\x00")):
        raise TeamCandidateLaunchError("team_candidate_bearer_credential_invalid")
    token = token.strip()
    endpoint = str(payload["endpoint_url"])
    origin = f"{urlsplit(endpoint).scheme}://{urlsplit(endpoint).netloc}"
    metadata = _team_metadata(endpoint, origin, token, hooks.get("metadata_transport"))
    records = _candidate_records(row=row, request=request, profile=profile, facts=facts, funding=funding,
                                 access=access, metadata=metadata, pipeline_run_id=pipeline_run_id)
    admission = admit_team_candidate(**records, scene=facts["scene"], probe_receipt=facts["probe_receipt"])
    _write(job_dir / "team_candidate_admission.json", admission)
    commit = hooks.get("implementation_commit") or _source_commit()
    run_id = "team-quick10-" + canonical_digest({"job_id": request["job_id"], "run": pipeline_run_id})[7:31]
    _write(job_dir / INTENT_FILENAME, {"job_id": request["job_id"], "canonical_request_digest":
                                        canonical_request_digest(request), "pipeline_run_id": pipeline_run_id,
                                        "admission_digest": admission["admission_digest"], "run_id": run_id,
                                        "source_commit": commit, "observed_at_iso": utc_now_iso()})
    paths = stage_team_runtime(
        root=job_dir, run_id=run_id, pair_inputs_path=Path(profile["pair_runtime_inputs_path"]),
        pair_activation_path=Path(profile["pair_activation_path"]), admission=admission, prompt=facts["prompt"],
        max_policy_queries=facts["max_policy_queries"], hard_cap_usd=float(funding["cap_usd"]),
        hard_ttl_seconds=int(profile["hard_ttl_seconds"]))
    bundle = build_team_candidate_session_bundle(
        job_dir=job_dir / "bundle", packet_dir=facts["packet_dir"],
        runtime_source_packet_receipt=facts["runtime_source_packet_receipt"],
        runtime_input_manifest_path=paths["runtime_inputs"], session_authority_path=paths["session_authority"],
        execution_spec_path=paths["execution_spec"], implementation_commit=commit)
    assert_secret_absent(bundle["bundle_path"], token)
    zero = dict(hooks["provider_zero"]())
    _write(job_dir / "initial_provider_zero.json", zero)
    if zero.get("provider_zero_verified") is not True:
        raise TeamCandidateLaunchError("team_candidate_initial_provider_zero_unproven")
    secrets_dir = job_dir / ".team-credential"
    secrets_dir.mkdir(mode=0o700)
    token_file = secrets_dir / "bearer"
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(token + "\n")
    argv = ["gpu-canary", "--provider", "vast", "--probe-kind", PROBE_KIND,
            "--native-task-arena-policy-canary-session-authority", str(paths["session_authority"]),
            "--native-task-arena-policy-canary-session-bundle-receipt",
            str(job_dir / "bundle" / "team_candidate_quick10_session_bundle_receipt.v1.json"),
            "--adp-job-dir", str(job_dir / "allocator"),
            "--adp-max-hourly-rate-usd", str(profile["max_hourly_rate_usd"]),
            "--adp-max-spend-usd", str(float(funding["cap_usd"])),
            "--adp-hard-ttl-seconds", str(int(profile["hard_ttl_seconds"])),
            "--admission-out", str(job_dir / "paid_admission.json"),
            "--adapter-output", str(job_dir / "allocator_result.json"),
            "--team-candidate-credential-file", str(token_file), "--execute"]
    if profile.get("machine_avoidlist_path"):
        argv.extend(["--adp-machine-avoidlist", str(profile["machine_avoidlist_path"])])
    try:
        exit_code = hooks["allocator_runner"](argv, log_path=job_dir / "allocator.log",
                                              timeout_seconds=int(profile["hard_ttl_seconds"]) + 900)
    finally:
        shutil.rmtree(secrets_dir, ignore_errors=True)
    _write(job_dir / "allocator_invocation.json", {"exit_code": exit_code, "argv_digest": canonical_digest(
        {"argv": argv}), "credential_file_removed": not secrets_dir.exists(), "automatic_retry_performed": False})
    seal_team_candidate_terminal(request=request, job_dir=job_dir, provider_zero=hooks["provider_zero"])


def execute_staged_team_candidate_request(*, row: Mapping[str, Any], request: Mapping[str, Any], job_dir: Path,
                                          pipeline_run_id: str | None, **overrides: Any) -> None:
    """Launch at most once; a failure releases the run only after provider-zero is observed."""

    from .checkpoint_policy_credentials import CheckpointPolicyCredentialClient
    from .team_candidate_quick10_closeout import seal_blocked_terminal, seal_team_candidate_terminal

    hooks = {"credential_client": CheckpointPolicyCredentialClient, "allocator_runner": _default_allocator_runner,
             "provider_zero": _default_provider_zero, **overrides}
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
                # Recover a sealed allocator result; never allocate twice after a caller crash.
                if not (job_dir / "allocator_result.json").exists():
                    raise TeamCandidateLaunchError("team_candidate_launch_interrupted_without_allocator_result")
                seal_team_candidate_terminal(request=request, job_dir=job_dir, provider_zero=hooks["provider_zero"])
                return
            if not isinstance(pipeline_run_id, str) or not pipeline_run_id:
                raise TeamCandidateLaunchError("team_candidate_dispatch_owner_missing")
            _launch(row=row, request=request, job_dir=job_dir, pipeline_run_id=pipeline_run_id, hooks=hooks)
        except Exception as exc:  # noqa: BLE001 - every failure is sealed, and released only at provider zero
            seal_blocked_terminal(request=request, job_dir=job_dir, error=exc, provider_zero=hooks["provider_zero"])


__all__ = [
    "CREDENTIAL_ENV",
    "FAILURE_FILENAME",
    "INTENT_FILENAME",
    "TERMINAL_FILENAME",
    "TERMINAL_SCHEMA",
    "TeamCandidateLaunchError",
    "execute_staged_team_candidate_request",
    "team_request_blockers",
]
