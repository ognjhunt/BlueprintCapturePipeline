"""Run the real preparation worker for the persistent owner's isolated queue."""
from __future__ import annotations


from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read
from .task_evaluation_scene_progression_state import atomic_json, require, safe_path


def run_preparation_service(*, config_path, now=None):
    from .task_evaluation_scene_progression import process_scene_intents
    from .task_evaluation_launch_preparation_worker import process_launch_preparation_queue
    config = read(safe_path(config_path), digest_field="config_digest")
    preparation = config.get("preparation_worker")
    require(isinstance(preparation, dict) and config.get("submission_transport") == "local_owned_queue"
            and config.get("activation_enabled") is False, "preparation_service_scope_invalid")
    before = process_scene_intents(config_path=config_path, now=now)
    worker = process_launch_preparation_queue(queue_root=safe_path(config["preparation_queue_root"]),
        input_root=safe_path(preparation["input_root"]),
        allowed_uri_prefixes=preparation["allowed_uri_prefixes"],
        service_account=config["service_account"], source_commit=before["source_commit"],
        construction_queue_root=safe_path(preparation["construction_queue_root"]),
        disk_reservation_root=safe_path(preparation["disk_reservation_root"]),
        storage_pins_root=safe_path(preparation["storage_pins_root"]),
        max_messages=preparation.get("max_messages", 1))
    after = process_scene_intents(config_path=config_path, now=now)
    result = {"schema_version": "task_evaluation_scene_preparation_service.v1",
        "status": "processed" if after["results"] else "idle", "source_commit": after["source_commit"],
        "scene_progression": after, "preparation_worker": worker,
        "execution_activation_enabled": False, "provider_allocation_performed": False}
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
