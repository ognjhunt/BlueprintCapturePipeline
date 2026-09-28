"""Create managed episode tasks directly from the canonical closeout producer."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from ..common import write_json
from ..decision_evidence_contracts import canonical_digest
from ..episode_interpretation_batch_authority import (
    MANAGED_SCHEMA_VERSION, SCHEMA_VERSION, derive_episode_interpretation_rights,
    validate_episode_interpretation_batch_authority,
)
from .contracts import AgentExecutionError, digest
from .episode_tasks import _IdentityScope, prepare_episode_task, collect_episode_task


def schedule_episode_batch(*, requests, result, evidence_root, profile, authority, service, unavailable=()):
    """Reserve once, derive exact rights, enqueue, and reuse eventual receipts."""
    from .production import _read_private
    from ..policy_canary_episode_interpretation_closeout import _artifact

    runtime = profile.get("runtime")
    if (profile.get("schema_version") != "policy_canary_episode_interpreter_profile.v2"
            or runtime not in {"openai_agents_api", "openai_agents_sdk"} or profile.get("status") != "configured"
            or profile.get("profile_digest") != canonical_digest(profile, digest_field="profile_digest")):
        raise AgentExecutionError("managed_episode_profile_invalid")
    model = profile["model"]
    identity = _IdentityScope(runtime, model)
    authority = validate_episode_interpretation_batch_authority(authority,
        run_id=result["run_id"], interpreter=identity, interpreter_profile_digest=profile["profile_digest"],
        maximum_cost_usd=profile["max_cost_usd"])
    expected_schema = MANAGED_SCHEMA_VERSION if runtime == "openai_agents_api" else SCHEMA_VERSION
    if (authority["schema_version"] != expected_schema
            or type(authority.get("maximum_episodes")) is not int or not 1 <= authority["maximum_episodes"] <= 20
            or type(authority.get("expires_at")) not in {int, float} or not math.isfinite(authority["expires_at"])
            or time.time() >= authority["expires_at"] or len(requests) > authority["maximum_episodes"]):
        raise AgentExecutionError("managed_episode_batch_scope_invalid")
    per_task = float(profile["per_episode_budget_usd"])
    if not 0 < per_task <= service.config.max_task_budget_usd or len(requests) * per_task > authority["maximum_cost_usd"]:
        raise AgentExecutionError("managed_episode_batch_budget_exceeded")
    root = Path(evidence_root) / "managed_episode_interpretation"
    receipts, artifacts = list(unavailable), []
    reused = 0
    batch_key = authority["authority_digest"][7:]
    with service.journal.own_task("managed-episode-batch:" + batch_key):
        for _, request, *_ in requests:
            token = request.input_receipt["input_bundle_digest"][7:]
            task_id = "episode-" + digest({"batch": batch_key, "input": token, "profile": profile["profile_digest"]})[7:]
            reservation_id = "episode_batch_reservation_" + batch_key + "_" + token
            reservation = {"task_id": task_id, "input_bundle_digest": request.input_receipt["input_bundle_digest"],
                           "nominal_inference_budget_usd": per_task, "batch_authority_digest": authority["authority_digest"]}
            if service.journal.event(reservation_id) is None:
                with service.journal._connect() as connection:
                    prior = connection.execute("SELECT payload_json FROM events WHERE event_id LIKE ?",
                        ("episode_batch_reservation_" + batch_key + "_%",)).fetchall()
                if len(prior) >= authority["maximum_episodes"] or sum(
                    json.loads(row["payload_json"])["nominal_inference_budget_usd"] for row in prior) + per_task > authority["maximum_cost_usd"]:
                    raise AgentExecutionError("managed_episode_batch_reservation_exhausted")
                service.journal.record_event(reservation_id, reservation)
            rights_path = service.journal.root / "episode-rights" / batch_key / (token + ".json")
            derive_episode_interpretation_rights(authority=authority, request=request, interpreter=identity, output_path=rights_path)
            rights_path.chmod(0o640)
            record_path = Path(service.config.task_store_root) / (task_id + ".json")
            if record_path.exists():
                reused += 1
                record = service.record(task_id)
                if (record.episode_investigation is None or record.episode_investigation.input_receipt != request.input_receipt
                        or record.task.model != model or record.task.run_id != result["run_id"] or record.task.admission.runtime != runtime):
                    raise AgentExecutionError("managed_episode_existing_task_conflict")
            else:
                record = prepare_episode_task(service, task_id=task_id, run_id=result["run_id"], request=request,
                    rights_path=rights_path, owner_client_id="blueprint-webapp", model=model, runtime=runtime, inference_budget_usd=per_task,
                    ttl_seconds=min(900, max(1, int(authority["expires_at"] - time.time()))))
            service.webapp_outbox.queue(record)
            try:
                state = service.journal.task(task_id)
                try:
                    receipt = collect_episode_task(service, task_id) if state["state"] == "completed" else None
                except (AgentExecutionError, ValueError, OSError):
                    receipt = None
                    state = {**state, "collection_refused": True}
            except AgentExecutionError as exc:
                if str(exc) != "agent_task_missing":
                    raise
                state, receipt = {"state": "admitted"}, None
            expired = time.time() >= record.task.deadline
            if expired and state["state"] not in {"admitted", "completed", "failed", "cancelled"}:
                service.service.cancel(task_id)
            if receipt is None and (state["state"] in {"failed", "cancelled"} or state.get("collection_refused") or expired):
                from ..episode_interpretation import materialize_episode_interpretation_abstention
                destination = root / token / "episode_interpretation_abstention.v1.json"
                receipt = materialize_episode_interpretation_abstention(request=request,
                    reason="managed_investigation_unavailable", output_path=destination)
            row = {"episode_id": request.episode_id, "candidate_policy_id": request.candidate_policy_id,
                "input_bundle_digest": request.input_receipt["input_bundle_digest"], "task_id": task_id,
                "task_digest": record.task.task_digest, "status": receipt["status"] if receipt else "pending",
                "runtime_state": state["state"], "receipt": None}
            if receipt is not None:
                destination = root / token / ("episode_interpretation.v2.json" if receipt["schema_version"] == "episode_interpretation_receipt.v2"
                                              else "episode_interpretation_abstention.v1.json")
                if destination.exists() and json.loads(_read_private(destination)) != receipt:
                    raise AgentExecutionError("managed_episode_receipt_conflict")
                write_json(destination, receipt)
                reference = _artifact(destination, root=Path(evidence_root), role="episode_interpretation_receipt")
                artifacts.append(reference)
                row["receipt"] = reference
                row["deterministic_agreement"] = receipt["deterministic_agreement"]
                for name, role in (("inspection.v1.json", "episode_interpretation_inspection"),
                                   ("agent_task_result.v1.json", "episode_interpretation_agent_result")):
                    if receipt["schema_version"] != "episode_interpretation_receipt.v2":
                        continue
                    source_path = service.journal.root / "episode-interpretations" / task_id / name
                    target = destination.parent / name
                    raw = _read_private(source_path, limit=64_000_000)
                    if target.exists() and target.read_bytes() != raw:
                        raise AgentExecutionError("managed_episode_execution_evidence_conflict")
                    from ..common import write_text
                    write_text(target, raw.decode("utf-8"))
                    artifacts.append(_artifact(target, root=Path(evidence_root), role=role))
            receipts.append(row)
    output = json.loads(json.dumps(result))
    inventory = output.setdefault("artifact_inventory", [])
    known = {row["relative_path"] for row in inventory}
    inventory.extend(row for row in artifacts if row["relative_path"] not in known)
    for episode in output["episodes"]:
        episode_id = (episode.get("episode") or {}).get("episode_id") or f"{result.get('run_id', 'policy-canary')}--{episode.get('cell_id')}--{episode.get('candidate_id')}"
        matching = next((row for row in receipts if row["episode_id"] == episode_id and row.get("receipt")), None)
        if matching:
            episode.setdefault("evidence_artifacts", {})["episode_interpretation"] = matching["receipt"]
    pending = sum(row["status"] == "pending" for row in receipts)
    completed = sum(row["status"] == "completed" for row in receipts)
    abstained = sum(row["status"] == "abstained" for row in receipts)
    summary = {"schema_version": "policy_canary_episode_interpretation_closeout.v2", "runtime": runtime,
        "status": "pending" if pending else "abstained" if abstained == len(receipts) else "partial" if abstained else "completed",
        "episode_count": len(output["episodes"]), "pending_count": pending,
        "completed_count": completed, "abstained_count": abstained,
        "disagreement_count": sum(row.get("deterministic_agreement") == "disagrees" for row in receipts),
        "reused_receipt_count": reused, "provider_call_count": None, "provider_invocation_attempt_count": None,
        "input_bundle_unavailable_count": len(unavailable), "interpreter": identity.identity.__dict__,
        "receipt_count": sum(row.get("receipt") is not None for row in receipts), "receipts": receipts,
        "batch_authority_digest": authority["authority_digest"], "interpreter_profile_digest": profile["profile_digest"],
        "authoritative_deterministic_result_unchanged": True, "score_overwrite_performed": False,
        "ranking_or_promotion_effect": "none", "official_cost_reconciliation": "required"}
    summary["summary_digest"] = canonical_digest(summary)
    output["episode_interpretation"] = summary
    output["artifact_inventory_digest"] = canonical_digest({"value": inventory})
    output["result_digest"] = canonical_digest(output, digest_field="result_digest")
    write_json(root / "managed_batch_status.json", summary)
    return output
