"""Reopen authenticated persistent owner consent for derived execution records."""
from __future__ import annotations

import os
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import checked_file, require


def task_contract_projection(task):
    return {"task_id": task["task_identity"]["id"], "strategy": task["strategy"],
            **{key: task[key] for key in ("subject", "support", "destination", "success")}}


def owner_numeric_task(request_task):
    require(set(request_task).issubset({"task_id", "strategy", "subject", "support", "destination", "success",
                                      "robot_binding_id", "episode_interpretation"}),
            "scene_owner_unknown_task_field")
    return {key: request_task[key] for key in ("task_id", "strategy", "subject", "support", "destination", "success")}


def _words(value):
    words = re.findall(r"[\w]+", str(value).casefold().replace("_", " "))
    return " ".join(word for word in words if word not in {"the", "a", "an"})


def descriptive_task_match(*, owner_task, seed, source_binding):
    """Resolve exact labels/registered descriptions; never infer another target."""
    from .task_evaluation_scene_configuration_submission_inputs import read, source_inputs
    parts = ("subject", "support", "destination", "success")
    if not all(isinstance(owner_task.get(key), dict)
               and set(owner_task[key]) == {"description", "authority"}
               and owner_task[key]["authority"] == "owner_confirmed"
               and isinstance(owner_task[key]["description"], str)
               and _words(owner_task[key]["description"]) for key in parts):
        return None
    if owner_task.get("strategy") != seed.get("strategy"):
        return None
    refs = source_binding["references"]
    paths = {key: checked_file(refs[key]["path"], refs[key]) for key in (
        "installation_receipt", "publisher_intake", "source_preparation_receipt")}
    installation = read(paths["installation_receipt"], digest_field="receipt_digest")
    context = source_inputs(installation_path=paths["installation_receipt"], publisher_path=paths["publisher_intake"],
        preparation_path=paths["source_preparation_receipt"], task=seed,
        commit=seed.get("expected_production_commit", installation["source_commit_sha"]))
    labels = read_labels = json.loads(context["raw"]["semantic_metadata"]["path"].read_text())
    if isinstance(read_labels, dict):
        labels = read_labels.get("objects", read_labels.get("labels", []))
    require(isinstance(labels, list), "scene_owner_source_labels_invalid")
    custom = source_binding.get("owner_description_aliases", {})
    require(isinstance(custom, dict) and set(custom).issubset(parts)
            and all(isinstance(values, list) and len(values) <= 16
                    and all(isinstance(v, str) and 0 < len(v) <= 1000 for v in values)
                    for values in custom.values()), "scene_owner_description_aliases_invalid")
    matches = {}
    for part in ("subject", "support"):
        target = context["identities"][part]["receipt"]["target"]
        label = _words(target["semantic_label"])
        description = _words(owner_task[part]["description"])
        aliases = {_words(seed[part].get("review_label", "")), label}
        aliases.update(_words(v) for v in custom.get(part, []))
        aliases.discard("")
        if description not in aliases:
            return None
        if description == label:
            objects = [row for row in labels if isinstance(row, dict) and _words(row.get("label", "")) == label]
            if len(objects) != 1 or str(objects[0].get("ins_id")) != str(seed[part]["source_instance_id"]):
                return None
        matches[part] = {"description": owner_task[part]["description"],
                         "source_instance_id": str(seed[part]["source_instance_id"]), "match": "exact_normalized_label"}
    destination = _words(seed["destination"]["visible_label"])
    aliases = {destination, destination.split()[-1]}
    aliases.update(_words(v) for v in custom.get("destination", []))
    if _words(owner_task["destination"]["description"]) not in aliases:
        return None
    success = {_words(seed.get("instruction", "")),
               _words("Place the object fully inside the destination, release it, and move the gripper clear.")}
    for target in aliases:
        success.update({"inside " + target, "fully inside " + target})
        for subject in {matches["subject"]["description"], seed["subject"]["review_label"]}:
            for verb in ("place", "put"):
                success.update({_words(f"{verb} {subject} inside {target}"),
                                _words(f"{verb} {subject} fully inside {target}")})
    success.update(_words(v) for v in custom.get("success", []))
    if _words(owner_task["success"]["description"]) not in success:
        return None
    return {"owner_task": owner_numeric_task(owner_task), "matches": matches,
            "original_seed_task_id": seed["task_identity"]["id"],
            "administrative_task_id_rebinding": owner_task["task_id"] != seed["task_identity"]["id"],
            "numeric_parameters_authority": "retained_machine_authored_development_seed",
            "numeric_parameters_owner_measured": False, "physical_truth_claimed": False}


def reopen_scene_intent(reference, *, now=None):
    """Only server-retained intake records may supply owner identity or permission."""
    from .task_evaluation_scene_intake import _read, validate_request
    path = Path(str((reference or {}).get("path", "")))
    root_text = os.getenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", "")
    require(bool(root_text), "scene_owner_intake_root_missing")
    root = Path(root_text)
    require(root.is_absolute() and path.is_absolute() and path.name == "intent.json"
            and path.parent.parent == root and not any(p.is_symlink() for p in (path, *path.parents)),
            "scene_owner_intent_path_invalid")
    checked_file(path, reference)
    intent = _read(path, "intent_digest")
    require(intent.get("schema_version") == "task_evaluation_scene_intent.v1"
            and path.parent.name == intent.get("intent_id"), "scene_owner_intent_invalid")
    trusted = {item.strip() for item in os.getenv(
        "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "blueprint-webapp").split(",") if item.strip()}
    require(intent.get("authenticated_issuer") in trusted, "scene_owner_issuer_not_trusted")
    request = validate_request(intent["request"], now=intent["accepted_at_epoch"])
    moment = time.time() if now is None else now
    require(not (path.parent / "revoked.json").exists(), "scene_owner_authority_revoked")
    require(moment < request["execution"]["expires_at_epoch"], "scene_owner_authority_expired")
    require(request["consent"]["accepted_by"] == request["owner"]["user_id"],
            "scene_owner_actor_mismatch")
    return intent


def validate_task_scene_owner(task, *, provider_terms_path=None, now=None):
    binding = task.get("scene_intent_authority")
    require(isinstance(binding, dict) and set(binding) in ({"intent", "intent_digest"}, {"intent", "intent_digest", "attempt"}),
            "scene_owner_binding_invalid")
    intent = reopen_scene_intent(binding["intent"], now=now)
    require(binding["intent_digest"] == intent["intent_digest"], "scene_owner_binding_changed")
    request, owner = intent["request"], task.get("human_authority", {})
    consent = request["consent"]
    if task.get("owner_description_seed_binding") is not None:
        from .task_evaluation_scene_configuration_submission_inputs import read
        from .task_evaluation_scene_intake import _read
        description = task["owner_description_seed_binding"]
        require("attempt" in binding, "scene_owner_description_attempt_missing")
        attempt_ref = binding["attempt"]
        attempt_path = checked_file(attempt_ref["path"], attempt_ref)
        require(attempt_path.parent == Path(binding["intent"]["path"]).parent / "attempts",
                "scene_owner_attempt_path_invalid")
        attempt = _read(attempt_path, "attempt_digest")
        source_ref = description["source_binding"]
        source_binding = read(checked_file(source_ref["path"], source_ref), digest_field="binding_digest")
        seed_ref = source_binding["accepted_task_seed"]
        seed = read(checked_file(seed_ref["path"], seed_ref))
        numeric = task_contract_projection(task)
        original_numeric = task_contract_projection(seed)
        numeric.pop("task_id")
        original_numeric.pop("task_id")
        require(attempt.get("intent_digest") == intent["intent_digest"]
                and attempt.get("source_commit") == task.get("expected_production_commit")
                and attempt.get("input_digest") == source_binding["binding_digest"]
                and source_binding.get("intent_task_digest") == cross_runtime_canonical_digest(request["task"])
                and task["task_identity"]["id"] == request["task"]["task_id"]
                and numeric == original_numeric, "scene_owner_seed_binding_mismatch")
        match = descriptive_task_match(owner_task=request["task"], seed=seed, source_binding=source_binding)
        require(match is not None and description.get("match") == match, "scene_owner_description_mismatch")
    else:
        require(cross_runtime_canonical_digest(task_contract_projection(task))
                == cross_runtime_canonical_digest(owner_numeric_task(request["task"])), "scene_owner_task_mismatch")
    require(all(task.get(key) == request["task"].get(key) for key in ("robot_binding_id", "episode_interpretation")),
            "scene_owner_execution_preferences_mismatch")
    require(owner.get("accepted_by") == request["owner"]["user_id"]
            and owner.get("authority_reference") == "scene-intent:" + intent["intent_digest"]
            and owner.get("accepted_on") == datetime.fromtimestamp(
                consent["accepted_at_epoch"], timezone.utc).isoformat(),
            "scene_owner_authority_mismatch")
    require(consent["private_processing_authorized"] is True
            and consent["provider_training_authorized"] is False
            and consent["task_confirmed"] is True and consent["spend_authorized"] is True
            and "openai" in request["execution"]["allowed_providers"], "scene_owner_review_not_authorized")
    if provider_terms_path is not None:
        from .task_evaluation_scene_configuration_submission_inputs import sha
        require(consent["provider_terms_reference"] == sha(Path(provider_terms_path)),
                "scene_owner_provider_terms_not_bound")
    return intent
