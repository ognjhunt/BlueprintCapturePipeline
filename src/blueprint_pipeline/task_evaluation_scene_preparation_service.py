"""Run the real preparation worker for the persistent owner's isolated queue."""
from __future__ import annotations

import json
import os
from pathlib import Path


from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read, checked_file
from .task_evaluation_scene_progression_state import atomic_json, require, safe_path


def _hold_code(exc):
    code = str(exc).split(":", 1)[0]
    if code and all(c.islower() or c.isdigit() or c == "_" for c in code):
        return code
    return "installed_source_dependency_unavailable"


def installed_source_environment(config, results, holds=None):
    """Bind publisher URLs to the selected owners' already installed source bytes.

    Public-scene factories retain these references in their sealed source binding.
    The owned worker must consume that same binding rather than depending on the
    legacy launch worker's per-scene systemd environment. The normal installed
    source resolver still verifies the publisher inventory and every input byte.

    One intent whose retained binding, installation receipt or publisher intake
    is missing or fails its check is left out of the environment and named in
    ``holds`` (one row per intent, with the failing file and its code); every
    other intent's source is still bound. A left-out intent has no installed
    source, so the worker cannot materialize its publisher URI from the host:
    that URI is outside the worker's allowed object prefixes and its message is
    blocked with ``launch_preparation_reference_prefix_not_allowed``.
    """
    from .task_evaluation_installed_source_bindings import BINDINGS_ENV
    environment = dict(os.environ)
    bindings = json.loads(environment.get(BINDINGS_ENV) or "[]")
    require(isinstance(bindings, list), "installed_source_configuration_invalid")
    for result in results:
        try:
            row = _installed_source_row(config, result)
        except _InstalledSourceHold as hold:
            if holds is not None:
                holds.append({"intent_id": str(result.get("intent_id")), "status": "held",
                              "blockers": [hold.code], "blocker_detail": [hold.detail]})
            continue
        if row is not None and row not in bindings:
            bindings.append(row)
    if bindings:
        environment[BINDINGS_ENV] = json.dumps(bindings)
    return environment


class _InstalledSourceHold(Exception):
    def __init__(self, code, detail):
        super().__init__(code)
        self.code, self.detail = code, detail


def _checked(code, check):
    try:
        return check()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _InstalledSourceHold(code, _hold_code(exc)) from exc


def _installed_source_row(config, result):
    from .task_evaluation_public_scene_attempt_factory import BINDING_SCHEMA
    from .task_evaluation_scene_intake import _read
    intent = _checked("installed_source_intent_unreadable", lambda: _read(
        safe_path(Path(config["intent_root"]) / result["intent_id"] / "intent.json"), "intent_digest"))
    source = intent["request"]["source"]
    if source["kind"] != "public_scene":
        return None
    retained_binding = result.get("source_binding") or (result.get("state") or {}).get("source_binding")
    if retained_binding is not None:
        binding_path = _checked("installed_source_binding_unavailable",
                                lambda: checked_file(retained_binding["path"], retained_binding))
    else:
        binding_path = safe_path(Path(config["public_source_binding_root"]) / (source["binding_id"] + ".json"))
        if not binding_path.exists():
            if (result.get("state") or {}).get("submission"):
                raise _InstalledSourceHold("installed_source_binding_unavailable", "submitted_source_binding_missing")
            return None  # No child can consume a source until its binding exists.
    binding = _checked("installed_source_binding_unavailable",
                       lambda: read(binding_path, digest_field="binding_digest"))
    if not (binding.get("schema_version") == BINDING_SCHEMA
            and binding.get("binding_id") == source["binding_id"]
            and binding.get("source_content_digest") == source["content_digest"]
            and binding.get("owner") == intent["request"]["owner"]):
        raise _InstalledSourceHold("installed_source_owner_mismatch", "installed_source_owner_mismatch")
    refs = binding.get("references") or {}
    installation = _checked("installed_source_installation_receipt_unavailable", lambda: checked_file(
        refs["installation_receipt"]["path"], refs["installation_receipt"]))
    publisher = _checked("installed_source_publisher_intake_unavailable", lambda: checked_file(
        refs["publisher_intake"]["path"], refs["publisher_intake"]))
    return {"installation_receipt_path": str(installation), "publisher_intake_path": str(publisher),
            "publisher_intake_sha256": refs["publisher_intake"]["sha256"]}


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
    # One intent's missing retained source file holds that intent, never the tick.
    holds = []
    try:
        before = process_scene_intents(config_path=config_path, now=now)
        environment = installed_source_environment(config, before["results"], holds)
        worker = process_launch_preparation_queue(queue_root=safe_path(config["preparation_queue_root"]),
            input_root=safe_path(preparation["input_root"]),
            allowed_uri_prefixes=preparation["allowed_uri_prefixes"],
            service_account=config["service_account"], source_commit=before["source_commit"],
            construction_queue_root=safe_path(preparation["construction_queue_root"]),
            disk_reservation_root=safe_path(preparation["disk_reservation_root"]),
            storage_pins_root=safe_path(preparation["storage_pins_root"]),
            max_messages=preparation.get("max_messages", 1),
            installed_source_environment=environment)
        after = process_scene_intents(config_path=config_path, now=now)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A tick that cannot run still says so in its status file, by code.
        code = str(exc).split(":", 1)[0] if isinstance(exc, ValueError) else ""
        if not code or not all(c.islower() or c.isdigit() or c == "_" for c in code):
            code = "preparation_service_dependency_unavailable"
        failed = {"schema_version": "task_evaluation_scene_preparation_service.v1", "status": "blocked",
                  "blockers": [code], "installed_source_holds": holds,
                  "execution_activation_enabled": activation_enabled, "provider_allocation_performed": False}
        failed["service_digest"] = canonical_digest(failed, digest_field="service_digest")
        atomic_json(safe_path(config["service_status_path"]), failed)
        raise
    result = {"schema_version": "task_evaluation_scene_preparation_service.v1",
        "status": "processed" if after["results"] else "idle", "source_commit": after["source_commit"],
        "scene_progression": after, "preparation_worker": worker, "installed_source_holds": holds,
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
