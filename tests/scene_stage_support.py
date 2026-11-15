"""A website job's completed scene-packet requests, built from kept code only, for the scene stage's tests."""
from copy import deepcopy
import json

from blueprint_pipeline import website_task_context
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.team_scene_packet_worker import CAPTURE_PARTITION_ENV


def source_world(tmp_path, monkeypatch):
    context = {"schema_version": "website_site_task_context.v1", "request_id": "job1", "scene_id": "site-job1",
        "capture_id": "walkthrough-job1", "purpose": "scene_preparation", "confirmed": False,
        "confirmed_at": None, "description": "Move a box", "capture_rights": {
            "consent_status": "granted", "consent_revoked": False,
            "derived_scene_generation_allowed": True, "consent_scope": ["robot_evaluation"]}}
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    capture = tmp_path / "scenes/site-job1/captures/walkthrough-job1"
    (capture / "pipeline").mkdir(parents=True)
    (capture / "capture_descriptor.json").write_text(json.dumps({"scene_id": context["scene_id"],
        "capture_id": context["capture_id"]}))
    (capture / "pipeline/website_task_context.json").write_text(json.dumps(context))
    monkeypatch.setenv(CAPTURE_PARTITION_ENV, str(tmp_path))
    source = {"capture_id": context["capture_id"], "task_context_digest": context["context_digest"],
              "task_id": "website-" + context["context_digest"][7:27]}
    monkeypatch.setattr(website_task_context, "load_current_website_task_context", lambda **_: deepcopy(context))
    return source, context


def prepared_source_queue(tmp_path, monkeypatch, *, revisions=1):
    """Real sealed completion requests; stub only host scene/runtime reads."""
    from blueprint_pipeline import task_evaluation_configured_controls_source_evidence as completed_scene
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from blueprint_pipeline import team_scene_packet_worker as packets

    source, context = source_world(tmp_path / "partition", monkeypatch)
    capture = tmp_path / "partition/scenes/site-job1/captures/walkthrough-job1"
    queue = tmp_path / "packets"
    results = {}
    requests = []
    for index in range(revisions):
        revision = canonical_digest({"configured_scene_revision": index})
        receipt = {"receipt_digest": canonical_digest({"launch_receipt": index})}
        launch_root = tmp_path / "launches" / str(index)
        result = {"configured_scene_revision_digest": revision}
        intent = {"intent_id": f"scene-intent-{index}", "intent_digest": canonical_digest({"intent": index}),
                  "request": {"submission_id": source["capture_id"], "task": {"task_id": source["task_id"]}}}
        queued = packets.enqueue_team_scene_request(intent=intent, run_root=launch_root, receipt=receipt,
            result=result, readiness={"status": "passed", "retained": True}, queue_root=queue)
        assert queued["status"] == "enqueued"
        key = queued["request_key"]
        (queue / "pending" / f"{key}.json").rename(queue / "processed" / f"{key}.json")
        (queue / "status" / f"{key}.json").write_text(json.dumps({"status": "completed"}))
        results[launch_root] = (result, receipt, {"provider_zero_confirmed": True})
        requests.append(json.loads((queue / "processed" / f"{key}.json").read_text()))
    validated = []
    def validate(path):
        validated.append(path)
        return deepcopy(results[path])
    monkeypatch.setattr(completed_scene, "_validate_source", validate)
    profile = {"scene_id": context["scene_id"], "capture_root": str(capture)}
    monkeypatch.setattr(registry, "profile_for_capture", lambda path: profile if path == capture else None)
    return {"packet_queue": queue, "screening_queue": tmp_path / "screening", "requests": requests,
            "results": results, "validated": validated, "context": context}
