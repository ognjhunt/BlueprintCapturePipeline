#!/usr/bin/env python3
"""ADP-010/day7: read-only host replay of parked website wake-up checks.

Run on paperclip-prod-01 as blueprint in the owner's read-only systemd sandbox,
with provider credentials withheld. This never calls reconciliation, sponsorship
creation, the handoff processor, or any ledger writer. Candidate source must be
staged separately by the local operator; cloud sessions cannot launch this.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import sys


DEFAULT_CAPTURE = (
    "/var/lib/blueprint/pubsub-handoffs/blueprint-8c1ca.appspot.com/scenes/"
    "site-capture-89d24b73-70d8-420e-a19c-b068c015bdab/captures/"
    "walkthrough-capture-89d24b73-70d8-420e-a19c-b068c015bdab"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-src", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--capture-root", type=Path, default=Path(DEFAULT_CAPTURE))
    args = parser.parse_args()
    if os.geteuid() != pwd.getpwnam("blueprint").pw_uid or os.geteuid() == 0:
        raise SystemExit("replay_requires_blueprint_service_user")
    if not re.fullmatch(r"[a-f0-9]{40}", args.candidate_commit):
        raise SystemExit("replay_candidate_commit_invalid")
    allowed = {"PIPELINE_SYNC_TOKEN", "BLUEPRINT_PIPELINE_SYNC_TOKEN_FILE", "GOOGLE_APPLICATION_CREDENTIALS"}
    credential_names = sorted(name for name in os.environ if any(
        word in name.upper() for word in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")) and name not in allowed)
    if credential_names:
        print(json.dumps({"code": "replay_provider_credentials_not_withheld", "names": credential_names}))
        return 1
    root = args.capture_root
    if not os.statvfs(root).f_flag & os.ST_RDONLY:
        raise SystemExit("replay_requires_read_only_capture_mount")
    candidate = args.candidate_src.resolve(strict=True)
    sys.path.insert(0, str(candidate))
    from blueprint_pipeline import pubsub_handoff_listener as listener
    from blueprint_pipeline import website_assessment_resume as resume
    from blueprint_pipeline import website_task_context as context_reader
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_source_projection
    from blueprint_pipeline.website_preparation_status import _read, _root, read_preparation_status
    for module in (listener, resume, context_reader):
        if not Path(module.__file__).resolve().is_relative_to(candidate):
            raise SystemExit("replay_candidate_import_mismatch")
    result = {"candidate_commit": args.candidate_commit, "capture_id": root.name,
              "provider_credentials_withheld": True, "capture_mount_read_only": True,
              "sponsorship_creation_called": False, "handoff_processor_called": False, "steps": []}
    before = None
    step = "root"
    try:
        _root(root)
        result["steps"].append(step)
        step = "ledger"
        ledger_path = root / listener.JOB_LEDGER_FILENAME
        before = ledger_path.read_bytes()
        ledger = listener._read_job_ledger(root)
        resume._require(ledger.get("status") == "failed_retryable" and ledger.get("last_error") == resume.PENDING_CODE)
        result["steps"].append(step)
        step = "resume_payload"
        record = ledger["assessment_resume"]
        raw = resume._payload_bytes(listener, base64.b64decode(record["payload_base64"], validate=True), record["source_payload_sha256"])
        result["steps"].append(step)
        step = "handoff"
        handoff = listener.parse_handoff_payload(raw)
        storage = root.parents[4]
        resume._require(root == listener._handoff_capture_root(handoff, storage_root=storage))
        resume._require(handoff.source_finalize is not None and handoff.source_membership_selector is not None
                        and ledger["source_payload_sha256"] == record["source_payload_sha256"]
                        and ledger["producer_delivery_key"] == record["producer_delivery_key"])
        result["steps"].append(step)
        step = "preparation_status_and_owner_check"
        selectors = {"request_id": handoff.scene_id.removeprefix("site-"), "scene_id": handoff.scene_id,
            "capture_id": handoff.capture_id, "completion_marker_generation": handoff.source_finalize["generation"],
            "producer_delivery_key": record["producer_delivery_key"], "source_payload_sha256": record["source_payload_sha256"],
            "task_context_digest": record["task_context_digest"]}
        claims = []
        status = read_preparation_status(capture_root=root, selectors=selectors, claim_evidence_sink=claims)
        resume._require(status["state"] == "failed_retryable" and status["revision"] == ledger["revision"])
        result["owner_check"] = "passed_verified_guest_claim" if claims else "passed_unchanged_birth_owner"
        result["steps"].append(step)
        step = "birth_membership"
        birth = capture_birth_source_projection(root, expected_purpose="scene_preparation")
        resume._require(birth is not None and birth["source_membership_selector"] == handoff.source_membership_selector
            and _read(Path(birth["birth_delivery_raw_ref"]["path"]))["producer_delivery"]["kind"] == "website_browser_capture_delivery")
        result["birth_owner_is_guest"] = birth["capture_owner_user_id"] is None
        result["steps"].append(step)
        step = "stored_context"
        context = context_reader.validate_website_task_context(_read(root / "pipeline/website_task_context.json"),
            request_id=selectors["request_id"], scene_id=handoff.scene_id, capture_id=handoff.capture_id, purpose="scene_preparation")
        resume._require(context["context_digest"] == record["task_context_digest"])
        result["steps"].append(step)
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="blocked", blocked_step=step, code=resume._blocker_code(exc))
    finally:
        if before is not None:
            result["ledger_bytes_unchanged"] = before == (root / listener.JOB_LEDGER_FILENAME).read_bytes()
            result["ledger_sha256"] = hashlib.sha256(before).hexdigest()
            if not result["ledger_bytes_unchanged"]:
                result["status"] = "blocked"
                result["code"] = "replay_ledger_changed_during_read"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
