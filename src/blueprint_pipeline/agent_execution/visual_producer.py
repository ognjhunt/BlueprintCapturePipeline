"""Produce optional adaptive investigations from exact final-review inputs."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..common import write_json
from ..decision_evidence_contracts import canonical_digest
from .contracts import AgentExecutionError, digest
from .visual_tasks import VisualTaskBinding, VisualView, prepare_visual_task

AUTHORITY_ENV = "BLUEPRINT_MANAGED_VISUAL_INVESTIGATION_AUTHORITY_FILE"


def schedule_visual_investigation(*, review_kind, run_id, candidate_digest, final_review_input_digest,
                                  frames, source_rights_admission_digest, authority_path=None, service=None):
    """The final reviewer still receives its original unmodified input."""
    from .production import _read_private, configured_service
    path = authority_path or os.environ.get(AUTHORITY_ENV)
    if not path:
        return None
    authority = json.loads(_read_private(Path(path)))
    if (authority.get("schema_version") != "blueprint_managed_visual_batch_authority.v1"
            or authority.get("authority_digest") != canonical_digest(authority, digest_field="authority_digest")
            or authority.get("review_kind") != review_kind
            or authority.get("source_rights_admission_digest") != source_rights_admission_digest
            or not (run_id in authority.get("run_ids", []) or candidate_digest in authority.get("candidate_digests", []))
            or authority.get("external_disclosure_authorized") is not True
            or authority.get("provider_training_authorized") is not False
            or authority.get("public_redistribution_authorized") is not False
            or not isinstance(authority.get("accepted_by"), str) or not authority["accepted_by"].strip()
            or type(authority.get("expires_at")) not in {int, float} or time.time() >= authority["expires_at"]
            or not math.isfinite(authority["expires_at"])
            or type(authority.get("maximum_tasks")) is not int or not 1 <= authority["maximum_tasks"] <= 10):
        raise AgentExecutionError("managed_visual_batch_authority_invalid")
    service = service or configured_service()
    budget = float(authority["per_task_budget_usd"])
    if not 0 < budget <= service.config.max_task_budget_usd:
        raise AgentExecutionError("managed_visual_batch_budget_invalid")
    paths = [Path(frame["path"]).resolve(strict=True) for frame in frames]
    root = Path(os.path.commonpath([str(path.parent) for path in paths]))
    views = []
    for index, (path, frame) in enumerate(zip(paths, frames, strict=True)):
        if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
            raise AgentExecutionError("managed_visual_producer_frame_changed")
        views.append(VisualView(view_id=f"view_{index:03d}", relative_path=str(path.relative_to(root)),
            sha256=frame["sha256"], camera_id=frame["camera_id"], role=frame["role"]))
    batch = authority["authority_digest"][7:]
    task_id = "visual-" + digest({"batch": batch, "candidate": candidate_digest, "input": final_review_input_digest})[7:]
    rights_path = service.journal.root / "visual-rights" / (task_id + ".json")
    binding = VisualTaskBinding(review_kind=review_kind, candidate_digest=candidate_digest, evidence_root=str(root),
        views=tuple(views), mandatory_view_ids=tuple(view.view_id for view in views), final_review_input_digest=final_review_input_digest,
        rights_path=str(rights_path), rights_sha256=digest({"pending_rights": True}))
    with service.journal.own_task("visual-batch:" + batch):
        reservation_id = "visual_batch_reservation_" + batch + "_" + task_id
        if service.journal.event(reservation_id) is None:
            with service.journal._connect() as connection:
                count = connection.execute("SELECT COUNT(*) FROM events WHERE event_id LIKE ?", ("visual_batch_reservation_" + batch + "_%",)).fetchone()[0]
            if count >= authority["maximum_tasks"]:
                raise AgentExecutionError("managed_visual_batch_reservation_exhausted")
            service.journal.record_event(reservation_id, {"task_id": task_id, "input_manifest_digest": binding.input_manifest_digest,
                "nominal_inference_budget_usd": budget, "batch_authority_digest": authority["authority_digest"]})
        rights = {"schema_version": "blueprint_visual_investigation_rights.v1", "input_manifest_digest": binding.input_manifest_digest,
            "candidate_digest": candidate_digest, "runtime": "openai_agents_api", "model": authority["model"],
            "agent_runtime_policy": authority["agent_runtime_policy"], "allowed_image_sha256": sorted({view.sha256 for view in views}),
            "external_disclosure_authorized": True, "provider_training_authorized": False, "public_redistribution_authorized": False,
            "accepted_by": authority["accepted_by"], "source_rights_admission_digest": source_rights_admission_digest,
            "batch_authority_digest": authority["authority_digest"]}
        rights["rights_digest"] = digest(rights)
        if rights_path.exists() and json.loads(_read_private(rights_path)) != rights:
            raise AgentExecutionError("managed_visual_derived_rights_conflict")
        write_json(rights_path, rights)
        rights_path.chmod(0o640)
        binding = VisualTaskBinding.model_validate({**binding.model_dump(),
            "rights_sha256": "sha256:" + hashlib.sha256(rights_path.read_bytes()).hexdigest()})
        record_path = Path(service.config.task_store_root) / (task_id + ".json")
        if record_path.exists():
            record = service.record(task_id)
            if record.visual_investigation != binding or record.task.run_id != run_id:
                raise AgentExecutionError("managed_visual_existing_task_conflict")
        else:
            record = prepare_visual_task(service, binding=binding, task_id=task_id, run_id=run_id,
                owner_client_id="blueprint-webapp", inference_budget_usd=budget, model=authority["model"],
                ttl_seconds=min(900, max(1, int(authority["expires_at"] - time.time()))))
        service.webapp_outbox.queue(record)
        return {"task_id": task_id, "task_digest": record.task.task_digest,
                "final_review_input_digest": final_review_input_digest, "final_acceptance_granted": False}


def best_effort_visual_investigation(**kwargs):
    try:
        return schedule_visual_investigation(**kwargs)
    except (AgentExecutionError, ValueError, OSError, KeyError):
        return {"status": "not_admitted", "final_acceptance_granted": False}
