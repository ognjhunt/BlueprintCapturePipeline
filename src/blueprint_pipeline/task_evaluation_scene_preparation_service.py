"""Run the real preparation worker for the persistent owner's isolated queue."""
from __future__ import annotations

import json
import os
from pathlib import Path


from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read, checked_file
from .task_evaluation_scene_progression_state import atomic_json, require, safe_path


def installed_source_environment(config, results):
    """Bind publisher URLs to the selected owners' already installed source bytes.

    Public-scene factories retain these references in their sealed source binding.
    The owned worker must consume that same binding rather than depending on the
    legacy launch worker's per-scene systemd environment. The normal installed
    source resolver still verifies the publisher inventory and every input byte.
    """
    from .task_evaluation_installed_source_bindings import BINDINGS_ENV
    from .task_evaluation_public_scene_attempt_factory import BINDING_SCHEMA
    from .task_evaluation_scene_intake import _read
    environment = dict(os.environ)
    bindings = json.loads(environment.get(BINDINGS_ENV) or "[]")
    require(isinstance(bindings, list), "installed_source_configuration_invalid")
    for result in results:
        intent = _read(safe_path(Path(config["intent_root"]) / result["intent_id"] / "intent.json"), "intent_digest")
        source = intent["request"]["source"]
        if source["kind"] != "public_scene":
            continue
        binding = read(safe_path(Path(config["public_source_binding_root"]) / (source["binding_id"] + ".json")),
                       digest_field="binding_digest")
        require(binding.get("schema_version") == BINDING_SCHEMA
                and binding.get("source_content_digest") == source["content_digest"]
                and binding.get("owner") == intent["request"]["owner"], "installed_source_owner_mismatch")
        refs = binding["references"]
        installation = checked_file(refs["installation_receipt"]["path"], refs["installation_receipt"])
        publisher = checked_file(refs["publisher_intake"]["path"], refs["publisher_intake"])
        row = {"installation_receipt_path": str(installation), "publisher_intake_path": str(publisher),
               "publisher_intake_sha256": refs["publisher_intake"]["sha256"]}
        if row not in bindings:
            bindings.append(row)
    if bindings:
        environment[BINDINGS_ENV] = json.dumps(bindings)
    return environment


def run_preparation_service(*, config_path, now=None):
    from .task_evaluation_scene_progression import process_scene_intents
    from .task_evaluation_launch_preparation_worker import process_launch_preparation_queue
    config = read(safe_path(config_path), digest_field="config_digest")
    preparation = config.get("preparation_worker")
    # R1: main() routes here whenever preparation_worker is present -- for BOTH the
    # unarmed preparation-only config (activation_enabled False) and the
    # owner-authorized activation config (activation_enabled True). Both are driven
    # by process_scene_intents, whose activation_enabled gate decides whether an
    # intent stops at construction_prepared or is advanced to the scene-configuration
    # activation (no allocation). The owned preparation worker runs in either mode.
    activation_enabled = config.get("activation_enabled")
    require(isinstance(preparation, dict) and config.get("submission_transport") == "local_owned_queue"
            and isinstance(activation_enabled, bool), "preparation_service_scope_invalid")
    before = process_scene_intents(config_path=config_path, now=now)
    worker = process_launch_preparation_queue(queue_root=safe_path(config["preparation_queue_root"]),
        input_root=safe_path(preparation["input_root"]),
        allowed_uri_prefixes=preparation["allowed_uri_prefixes"],
        service_account=config["service_account"], source_commit=before["source_commit"],
        construction_queue_root=safe_path(preparation["construction_queue_root"]),
        disk_reservation_root=safe_path(preparation["disk_reservation_root"]),
        storage_pins_root=safe_path(preparation["storage_pins_root"]),
        max_messages=preparation.get("max_messages", 1),
        installed_source_environment=installed_source_environment(config, before["results"]))
    after = process_scene_intents(config_path=config_path, now=now)
    result = {"schema_version": "task_evaluation_scene_preparation_service.v1",
        "status": "processed" if after["results"] else "idle", "source_commit": after["source_commit"],
        "scene_progression": after, "preparation_worker": worker,
        "execution_activation_enabled": activation_enabled, "provider_allocation_performed": False}
    result["service_digest"] = canonical_digest(result, digest_field="service_digest")
    atomic_json(safe_path(config["service_status_path"]), result)
    return result


def preparation_service_status(config_path):
    """Read the actual worker's last receipt; a configured timer is not success."""
    config = read(safe_path(config_path), digest_field="config_digest")
    path = safe_path(config["service_status_path"])
    if not path.exists():
        return {"status": "not_run", "provider_allocation_performed": False}
    return read(path, digest_field="service_digest")
