"""Carry one closed owner's partial Astra work into its next sealed ADP-009D bundle.

Selection and copying are local only. Existing fresh-owner, paid-resource, and
provider-zero gates still authorize the successor; this receipt grants no spend.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import zipfile

from .decision_evidence_contracts import (
    canonical_digest,
    canonical_json,
    cross_runtime_canonical_digest,
)
from .control_plane_disk_budget import reserve_control_plane_disk
from .task_evaluation_scene_attempt_binding import require_scene_execution_binding

SCHEMA = "task_evaluation_partial_astra_transport.v1"
FIELD = "partial_astra_successor"
PREFIX = "stages/stage-3/producer/astra_cad_blender_runtime/"
ORIGINAL_ROOT = "/workspace/task_evaluation_scene_configuration_provider_bundle/runtime_output/" + PREFIX.rstrip("/")
ARCHIVE_RELATIVE = "allocator/scene-configuration-job/vast_provider_run/vast_provider_runtime_output.zip"
CPU_ARCHIVE_RELATIVE = "allocator/scene-configuration-job/cpu_prestage_output.zip"
MAX_SCAN = 4096
MAX_CANDIDATES = 16
MAX_ARCHIVE_BYTES = 1024**3
MAX_EXPANDED_BYTES = 2 * 1024**3
MAX_MEMBERS = 100_000
SOURCE_FILES = {"text_to_cad_skills_source.zip", "multi_agent_cad_source.zip",
                "cad_skill_source_receipt.json", "multi_agent_cad_skill.md", "stage_source_binding.json"}
REQUIRED = {"authoring/request.json", "authoring/source_analysis.json", "authoring/cad_result.json",
            "authoring/physical_property_review.json", "authoring/appearance-00/independent_visual_review_0.json",
            "authoring/appearance-01/blender_author_1.json", "authoring/appearance-01/geometry_readback.json",
            "authoring/appearance-01/final_visual_mesh_receipt.json", "stage_source_binding.json"}


class PartialAstraTransportError(ValueError):
    """A static retained-transport predicate refused before provider admission."""


def _require(condition, reason):
    if not condition:
        raise PartialAstraTransportError("partial_astra_transport_" + reason)


def _safe(path):
    path = Path(path)
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "path_unsafe")
    return path


def _record_digest(value, field):
    # Intake records normalize integral JSON numbers across producer runtimes.
    digest = cross_runtime_canonical_digest if field in {"intent_digest", "attempt_digest"} else canonical_digest
    return digest(value, digest_field=field)


def _read(path, field=None):
    path = _safe(path)
    _require(path.is_file() and path.stat().st_size <= 8 * 1024**2, "record_invalid")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "record_invalid")
    if field:
        _require(value.get(field) == _record_digest(value, field), "record_digest_invalid")
    return value


def _sha(path):
    with _safe(path).open("rb") as source:
        return "sha256:" + hashlib.file_digest(source, "sha256").hexdigest()


def _file(path):
    return {"path": str(path), "sha256": _sha(path), "size_bytes": Path(path).stat().st_size}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(canonical_json(value) + "\n")
    path.chmod(0o440)


def _canonical_owner(value, root):
    """Reopen historical identity without reopening its expired spending authority."""
    from . import task_evaluation_scene_intake as intake
    binding = value["scene_attempt_binding"]
    require_scene_execution_binding(value, source_commit=binding["source_commit"])
    directory = _safe(root) / binding["intent_id"]
    intent = _read(directory / "intent.json", "intent_digest")
    attempt = _read(directory / "attempts" / (binding["attempt_id"] + ".json"), "attempt_digest")
    _require(intent["intent_digest"] == binding["intent_digest"] and
             all(attempt.get(k) == v for k, v in binding.items() if k != "schema_version"), "owner_attempt_changed")
    clients = {v.strip() for v in os.getenv(intake.CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
    _require(intent.get("authenticated_issuer") in clients and attempt.get("provider") == "vast", "owner_issuer_invalid")
    return intent, attempt


def _source_identity(source, current, *, intent_root, successor_run_id):
    profile = _read(source / "launch_profile.json", "profile_digest")
    scope = profile.get("task_evaluation_run") or {}
    _require(scope.get("scene_id") == current["scene_id"] and scope.get("task_id") == current["task_id"]
             and scope.get("run_mode") == "scene_configuration", "source_scope_changed")
    own, new_attempt = _canonical_owner(current, intent_root)
    prior, old_attempt = _canonical_owner(profile, intent_root)
    _require(own == prior and own["request"]["task"]["task_id"] == current["task_id"], "source_owner_changed")
    launch = _read(source / "launch_receipt.json", "receipt_digest")
    zero = _read(source / "post_teardown_provider_zero_receipt.json", "provider_zero_receipt_digest")
    _require(launch.get("status") in {"blocked", "completed"} and launch.get("launch_id") == source.name
             and zero.get("launch_id") == source.name
             and zero.get("status") == "provider_zero_confirmed" and zero.get("provider_zero_verified") is True
             and zero.get("continuing_spend_from_this_run") is False and not zero.get("blockers"), "source_not_closed")
    source_run_id = scope["configuration_run_id"]
    _require(source_run_id != successor_run_id, "same_run_not_successor")
    lineage = {"source_run_id": source_run_id, "successor_run_id": successor_run_id,
               "owner_id": own["request"]["owner"]["user_id"],
               "organization_id": own["request"]["owner"]["organization_id"],
               "stable_intent_id": own["intent_id"], "stable_intent_digest": own["intent_digest"],
               "scene_id": current["scene_id"], "task_id": current["task_id"],
               "source_attempt_digest": old_attempt["attempt_digest"],
               "successor_attempt_digest": new_attempt["attempt_digest"]}
    result = _read(source / "allocator/scene-configuration-job/task_evaluation_scene_configuration_vast_result.v1.json", "result_digest")
    originals = {}
    for name in ("source_bundle_manifest", "scene_configuration_attempt_authority"):
        rows = [r for r in profile["immutable_inputs"] if r.get("name") == name]
        _require(len(rows) == 1 and _sha(Path(rows[0]["path"])) == rows[0]["digest"], "source_input_changed")
        originals[name] = _read(rows[0]["path"])
    proof = {"source_result": result, "source_bundle_manifest": originals["source_bundle_manifest"],
             "source_authority": originals["scene_configuration_attempt_authority"], "source_profile": profile, "source_launch_receipt": launch, "source_provider_zero": zero,
             "successor_owner_attempt": current, "canonical_intent": own,
             "source_attempt": old_attempt, "successor_attempt": new_attempt}
    _validate_source_proof(proof, source)
    return lineage, proof


def _validate_source_proof(proof, source):
    profile, result, zero, launch = (proof[k] for k in ("source_profile", "source_result", "source_provider_zero", "source_launch_receipt"))
    authority = proof["source_authority"]
    remote = result.get("provider_runtime_output_remote_reference") or {}
    _require(result.get("result_digest") == canonical_digest(result, digest_field="result_digest")
             and profile.get("profile_digest") == canonical_digest(profile, digest_field="profile_digest")
             and launch.get("receipt_digest") == canonical_digest(launch, digest_field="receipt_digest")
             and authority.get("authority_digest") == canonical_digest(authority, digest_field="authority_digest")
             and result.get("run_id") == profile["task_evaluation_run"]["configuration_run_id"]
             and result.get("source_commit") == profile["source_commit"] == profile["scene_attempt_binding"]["source_commit"]
             and result.get("authority_digest") == authority["authority_digest"]
             and result.get("bundle_sha256") == authority["bundle_sha256"] == proof["source_bundle_manifest"]["bundle_sha256"]
             and result.get("provider_runtime_output_zip_path") in {str(Path(source) / name)
                 for name in (ARCHIVE_RELATIVE, CPU_ARCHIVE_RELATIVE)}
             and remote.get("status") == "remote_verified"
             and remote.get("digest") == remote.get("readback_digest") == result.get("provider_runtime_output_zip_sha256")
             and remote.get("size_bytes") == remote.get("readback_size_bytes")
             and type(remote.get("size_bytes")) is int and 0 < remote["size_bytes"] <= MAX_ARCHIVE_BYTES
             and remote.get("full_byte_service_account_readback_passed") is True
             and launch.get("launch_profile_digest") == zero.get("launch_profile_digest") == profile["profile_digest"]
             and zero.get("receipt_digest") == launch["receipt_digest"], "source_archive_authority_invalid")


def _allowed(relative):
    parts = PurePosixPath(relative).parts
    return relative in SOURCE_FILES or (parts and parts[0] in {"authoring", "inference", "official_openai_cost"})


def _latest_prefix(archive):
    """Preserve the newest resumed conversation and ledger, never its ancestor."""
    roots = {name.rsplit("/stage_source_binding.json", 1)[0] + "/" for name in archive.namelist()
             if name.endswith("/stage_source_binding.json")}
    resumed = sorted(root for root in roots if re.fullmatch(
        r"stages/stage-3/producer/astra_resume_attempts/attempt-[0-9]{4}/", root))
    return resumed[-1] if resumed else PREFIX


def _members(archive, prefix=PREFIX):
    infos = archive.infolist()
    _require(len(infos) <= MAX_MEMBERS and len({i.filename for i in infos}) == len(infos), "archive_members_invalid")
    selected = []
    for info in infos:
        path = PurePosixPath(info.filename)
        _require(not path.is_absolute() and ".." not in path.parts, "archive_path_invalid")
        if not info.filename.startswith(prefix) or info.is_dir():
            continue
        relative = info.filename[len(prefix):]
        if not _allowed(relative):
            continue
        _require(not any(part in {".env", "secrets", "credentials", "runtime_secrets", "openai_api_key"}
                         or part.endswith((".pem", ".p12", ".key")) for part in PurePosixPath(relative).parts),
                 "runtime_secret_forbidden")
        _require(stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG)
                 and info.file_size >= 0, "archive_entry_invalid")
        selected.append((relative, info))
    names = {name for name, _ in selected}
    sdk = {"authoring/request.json", "authoring/source_analysis.json", "authoring/cad_result.json",
           "stage_source_binding.json", "inference/asset_session/binding.json",
           "inference/asset_session/conversation.sqlite"} <= names
    legacy = REQUIRED <= names and "authoring/appearance-01/independent_visual_review_1.json" not in names
    articulated = "authoring/result.json" in names and {
        "authoring/parts/carcass/request.json", "authoring/parts/carcass/result.json",
        "authoring/parts/drawer/request.json", "authoring/parts/drawer/result.json",
        "inference/agents_api/parts/carcass/agents_api_stage_receipt.json",
        "inference/agents_api/parts/drawer/agents_api_stage_receipt.json",
        "stage_source_binding.json",
    } <= names
    _require(((sdk or legacy) and "authoring/result.json" not in names) or articulated,
             "source_not_pending_final_review_or_completed_articulated")
    _require(sum(i.file_size for _, i in selected) <= MAX_EXPANDED_BYTES, "archive_expansion_exceeded")
    return sorted(selected)


def _archive_json(archive, relative, prefix=PREFIX):
    info = archive.getinfo(prefix + relative)
    _require(info.file_size <= 8 * 1024**2, "archive_json_too_large")
    value = json.loads(archive.read(info))
    _require(isinstance(value, dict), "archive_json_invalid")
    return value


def validate_envelope_owner(envelope, current):
    """Bind the actual prepared construction request to its canonical reserved input."""
    request = envelope.get("request")
    binding = current["scene_attempt_binding"]
    _require(isinstance(request, dict)
             and canonical_digest(request) == binding["input_digest"]
             and request.get("scene_intent_digest") == current["scene_intent_digest"]
             and request.get("scene", {}).get("identity", {}).get("id") == current["scene_id"]
             and request.get("task", {}).get("identity", {}).get("id") == current["task_id"]
             and request.get("team_namespace") == current["team_namespace"] == envelope.get("team_namespace")
             and request.get("run_id") == envelope.get("run_id")
             and request.get("expected_production_commit") == binding["source_commit"] == envelope.get("expected_production_commit"),
             "construction_owner_binding_changed")


def _pin_source_metadata(*, proof, activation_request, activation_root, pins_root, on_pin_created):
    """An actual activation owns the retained roots until its normal terminal release."""
    if activation_request is None:
        return
    from .control_plane_storage_pins import pin_activation_best_effort, pin_status
    current = proof["successor_owner_attempt"]
    owned = _safe(activation_root)
    _require(owned.name == activation_request.get("activation_id")
             and activation_request.get("team_namespace") == current["team_namespace"], "metadata_pin_owner_changed")
    metadata_roots = set()
    for name in ("source_bundle_manifest", "scene_configuration_attempt_authority"):
        references = [row for row in proof["source_profile"]["immutable_inputs"] if row.get("name") == name]
        _require(len(references) == 1, "metadata_pin_source_missing")
        path = _safe(references[0]["path"])
        source_roots = [parent for parent in path.parents if parent.parent == owned.parent]
        _require(len(source_roots) == 1 and source_roots[0] != owned, "metadata_pin_source_root_invalid")
        metadata_roots.add(str(source_roots[0]))
    pin = pin_activation_best_effort(activation_request, owned.parent, pins_root=pins_root,
                                   retained_paths=sorted(metadata_roots), on_created=on_pin_created)
    expected_dependencies = [{"kind": kind, "owner_id": activation_request["preparation"]["preparation_id"]}
                             for kind in ("compilation", "preparation")]
    _require(isinstance(pin, dict) and pin_status(pin, now=time.time()) == "live"
             and pin.get("owner_id") == owned.name and pin.get("kind") == "activation"
             and {str(owned), *metadata_roots} <= set(pin.get("paths", []))
             and pin.get("depends_on") == expected_dependencies, "metadata_pin_missing_or_insufficient")


def select_partial_astra_source(*, owner_attempt_path, envelope, output_root,
                                intent_root=None, launch_root=None, activation_request=None,
                                activation_root=None, pins_root=None, on_pin_created=None):
    """Freeze the newest closed, same-owner/scene/task pending second-review source."""
    from . import task_evaluation_scene_intake as intake
    from .task_object_astra_authoring import AssetAuthoringError
    current = _read(owner_attempt_path, "owner_attempt_digest")
    validate_envelope_owner(envelope, current)
    target = _safe(output_root)
    if target.exists():
        retained = _read(target / "selection.json", "transport_digest")
        _require(retained["authority_evidence"]["successor_owner_attempt"] == current
                 and retained["verified_lineage"]["successor_run_id"] == envelope["run_id"], "retained_selection_changed")
        if activation_request is not None:
            lineage, proof = _source_identity(Path(retained["source_launch_root"]), current,
                intent_root=_safe(intent_root or os.environ[intake.ROOT_ENV]), successor_run_id=envelope["run_id"])
            _require(proof == retained["authority_evidence"] and lineage == retained["verified_lineage"], "retained_source_changed")
            _require(_file(Path(retained["source_archive"]["path"])) == retained["source_archive"],
                     "retained_source_archive_changed")
            _pin_source_metadata(proof=proof, activation_request=activation_request, activation_root=activation_root,
                                 pins_root=pins_root, on_pin_created=on_pin_created)
            _require(_source_identity(Path(retained["source_launch_root"]), current,
                intent_root=_safe(intent_root or os.environ[intake.ROOT_ENV]), successor_run_id=envelope["run_id"]) == (lineage, proof),
                "retained_source_changed_after_pin")
        return target / "selection.json"
    root = _safe(intent_root or os.environ[intake.ROOT_ENV])
    _canonical_owner(current, root)
    # Match public_scene_attempt_factory._prefix_candidates: a fresh-intent flag
    # excludes other intents, not this intent's own interrupted work. Selection
    # below always requires the same canonical intent, including when false.
    launches = _safe(launch_root or os.getenv("BLUEPRINT_TASK_EVALUATION_LAUNCH_STATE_ROOT")
                     or root.parent / "task-evaluation-launch-runs")
    if not launches.exists():
        return None
    sources = []
    for index, source in enumerate(launches.iterdir()):
        if index >= MAX_SCAN:
            break
        if source.is_dir() and not source.is_symlink():
            zero = source / "post_teardown_provider_zero_receipt.json"
            if zero.is_file() and not zero.is_symlink():
                try:
                    profile = _read(source / "launch_profile.json")
                    scope = profile.get("task_evaluation_run") or {}
                    if ((profile.get("scene_attempt_binding") or {}).get("intent_digest") != current["scene_intent_digest"]
                            or scope.get("scene_id") != current["scene_id"] or scope.get("task_id") != current["task_id"]):
                        continue
                    sources.append((zero.stat().st_mtime_ns, source))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
    candidates = sorted(sources, reverse=True)[:MAX_CANDIDATES]
    stages, references = envelope["recipe"]["stage_sequence"], envelope["stage_configuration_references"]
    _require(len(stages) == len(references), "configuration_reference_count_invalid")
    matches = [(index, row) for index, (stage, row) in enumerate(zip(stages, references, strict=True))
               if stage.get("stage_id") == "stage-3"]
    _require(len(matches) == 1, "configuration_stage_invalid")
    index, configuration = matches[0]
    _require(configuration.get("contract_path") == f"construction.recipe.stage_sequence.{index}.configuration",
             "configuration_reference_slot_invalid")
    rejected = []
    unusable_partial = False
    for _, source in candidates:
        created_target = False
        known_partial = False
        authenticated_archive = False
        try:
            lineage, proof = _source_identity(source, current, intent_root=root, successor_run_id=envelope["run_id"])
            authenticated_archive = True
            archive_path = _safe(proof["source_result"]["provider_runtime_output_zip_path"])
            _require(archive_path.is_file() and archive_path.stat().st_size <= MAX_ARCHIVE_BYTES, "source_archive_invalid")
            original_record = _file(archive_path)
            _require(original_record["sha256"] == proof["source_result"]["provider_runtime_output_zip_sha256"]
                     and original_record["size_bytes"] == proof["source_result"]["provider_runtime_output_remote_reference"]["size_bytes"],
                     "source_archive_download_changed")
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                prefix = _latest_prefix(archive)
                known_partial = any(p + "authoring/cad_result.json" in names or
                                    p + "authoring/result.json" in names for p in (prefix, PREFIX))
                from .task_evaluation_partial_astra_successor import (
                    semantic_articulated_requests, semantic_request,
                )
                articulated = prefix + "authoring/result.json" in names
                if articulated:
                    authored = _archive_json(archive, "authoring/result.json", prefix)
                    source_stage = _archive_json(archive, "stage_production_input.v1.json",
                                                 "stages/stage-3/producer/")
                    source_envelope_digest = (source_stage.get("construction_envelope") or {}).get("envelope_digest")
                    _require(authored.get("schema_version") == "task_object_astra_articulated_authoring_result.v1"
                             and authored.get("status") == "parts_authored_pending_native_qualification"
                             and authored.get("agent_runtime") == "openai_agents_api"
                             and authored.get("model") == "gpt-6-sol"
                             and (authored.get("plan") or {}).get("source_geometry_receipt", {}).get(
                                 "construction_envelope_digest") == source_envelope_digest
                             and authored.get("result_digest") == canonical_digest(authored, digest_field="result_digest"),
                             "completed_articulated_result_invalid")
                    requests = {part: _archive_json(archive, f"authoring/parts/{part}/request.json", prefix)
                                for part in sorted(authored.get("parts") or {})}
                    _require(set(requests) == {"carcass", "drawer"}
                             and all(row.get("request_digest") == canonical_digest(row, digest_field="request_digest")
                                     and row.get("run_id") == lineage["source_run_id"] for row in requests.values())
                             and authored.get("part_request_digests") == {
                                 part: row["request_digest"] for part, row in requests.items()},
                             "completed_articulated_requests_invalid")
                    request_digest = canonical_digest({part: row["request_digest"] for part, row in requests.items()})
                    semantic_digest = canonical_digest(semantic_articulated_requests(requests))
                else:
                    request = _archive_json(archive, "authoring/request.json", prefix)
                    request_digest = request["request_digest"]
                    semantic_digest = canonical_digest(semantic_request(request))
                binding = _archive_json(archive, "stage_source_binding.json", prefix)
                _require((articulated or (request.get("request_digest") == canonical_digest(request, digest_field="request_digest")
                         and request.get("run_id") == lineage["source_run_id"]))
                         and binding.get("binding_digest") == canonical_digest(binding, digest_field="binding_digest")
                         and binding.get("run_id") == lineage["source_run_id"]
                         and binding.get("configuration_sha256") == configuration["digest"], "source_configuration_changed")
                members = _members(archive, prefix)
                _pin_source_metadata(proof=proof, activation_request=activation_request, activation_root=activation_root,
                                     pins_root=pins_root, on_pin_created=on_pin_created)
                if activation_request is not None:
                    _require(_source_identity(source, current, intent_root=root, successor_run_id=envelope["run_id"]) == (lineage, proof),
                             "source_changed_after_metadata_pin")
                with reserve_control_plane_disk(
                    "semantic_pretraining", target_root=target.parent,
                    expected_bytes=sum(i.file_size for _, i in members) + 16 * 1024**2,
                ):
                    target.mkdir(parents=True, exist_ok=False)
                    created_target = True
                    packed = target / "retained_runtime.zip"
                    inventory = []
                    with zipfile.ZipFile(packed, "x", compression=zipfile.ZIP_STORED) as output:
                        for relative, info in members:
                            digest = hashlib.sha256()
                            with archive.open(info) as stream, output.open(relative, "w") as destination:
                                while chunk := stream.read(1024 * 1024):
                                    digest.update(chunk)
                                    destination.write(chunk)
                            inventory.append({"relative_path": relative, "sha256": "sha256:" + digest.hexdigest(),
                                              "size_bytes": info.file_size})
                _require(_file(archive_path) == original_record and packed.stat().st_size <= MAX_ARCHIVE_BYTES,
                         "source_archive_changed")
                from .task_evaluation_partial_astra_successor import SCHEMA_VERSION, semantic_request
                descriptor = {"schema_version": SCHEMA_VERSION,
                    **({"adoption_kind": "completed_articulated_agents_api"} if articulated else {}),
                    **({"source_construction_envelope_digest": source_envelope_digest} if articulated else {}),
                    "source_run_id": lineage["source_run_id"],
                    "successor_run_id": envelope["run_id"], "original_runtime_root": ORIGINAL_ROOT.removesuffix(PREFIX.rstrip("/")) + prefix.rstrip("/"),
                    "source_request_digest": request_digest,
                    "semantic_request_digest": semantic_digest,
                    "source_stage_binding_digest": binding["binding_digest"], "owner_intent_lineage": lineage,
                    "retained_runtime_archive": {k: v for k, v in _file(packed).items() if k != "path"},
                    "retained_files": inventory}
                descriptor["adoption_digest"] = canonical_digest(descriptor, digest_field="adoption_digest")
                descriptor_path = target / "descriptor.json"
                _write(descriptor_path, descriptor)
                selection = {"schema_version": SCHEMA, "verified_lineage": lineage, "authority_evidence": proof,
                    "source_launch_root": str(source), "intent_root": str(root),
                    "source_archive": original_record, "descriptor": _file(descriptor_path),
                    "runtime_archive": _file(packed), "rejected_sources": rejected,
                    "new_spend_authorized": False}
                selection["transport_digest"] = canonical_digest(selection, digest_field="transport_digest")
                selection_path = target / "selection.json"
                _write(selection_path, selection)
                return selection_path
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile,
                AssetAuthoringError) as exc:
            reason = str(exc)
            excluded = reason.endswith(("source_configuration_changed", "source_owner_changed", "source_scope_changed",
                                        "owner_attempt_changed", "source_not_closed"))
            unusable_partial |= ((known_partial and not excluded) or reason.endswith("source_archive_download_changed")
                                 or (authenticated_archive and (isinstance(exc, zipfile.BadZipFile)
                                     or reason.endswith("source_archive_invalid"))))
            rejected.append({"source_launch_id": source.name, "reason": reason[:240]})
            if created_target:
                # Only this function's newly created incomplete transport directory.
                shutil.rmtree(output_root)
    if unusable_partial:
        refusal = {"schema_version": SCHEMA, "status": "blocked", "rejected_sources": rejected,
                   "new_spend_authorized": False}
        refusal["transport_digest"] = canonical_digest(refusal, digest_field="transport_digest")
        path = target.parent / "partial_astra_successor_rejections.json"
        if not path.exists():
            _write(path, refusal)
        raise PartialAstraTransportError("partial_astra_transport_known_partial_source_unusable")
    return None


def validate_transport(value, descriptor):
    """Validate the independent owner-record chain sealed by the bundle creator."""
    _require(value.get("schema_version") == SCHEMA and value.get("transport_digest") ==
             canonical_digest(value, digest_field="transport_digest") and value.get("new_spend_authorized") is False,
             "selection_invalid")
    proof = value["authority_evidence"]
    _validate_source_proof(proof, Path(value["source_launch_root"]))
    intent = proof["canonical_intent"]
    current = proof["successor_owner_attempt"]
    profile = proof["source_profile"]
    lineage = value["verified_lineage"]
    for record, field in ((intent, "intent_digest"), (current, "owner_attempt_digest"),
                          (proof["source_attempt"], "attempt_digest"), (proof["successor_attempt"], "attempt_digest"),
                          (proof["source_provider_zero"], "provider_zero_receipt_digest")):
        _require(record.get(field) == _record_digest(record, field), "authority_digest_invalid")
    for record, attempt in ((current, proof["successor_attempt"]), (profile, proof["source_attempt"])):
        binding = record["scene_attempt_binding"]
        require_scene_execution_binding(record, source_commit=binding["source_commit"])
        _require(binding["intent_id"] == intent["intent_id"] and binding["intent_digest"] == intent["intent_digest"]
                 and all(attempt.get(k) == v for k, v in binding.items() if k != "schema_version"), "authority_lineage_invalid")
    zero = proof["source_provider_zero"]
    _require(proof["source_launch_receipt"].get("status") in {"blocked", "completed"}
             and proof["source_launch_receipt"].get("launch_id") == Path(value["source_launch_root"]).name
             and zero.get("launch_id") == Path(value["source_launch_root"]).name
             and zero.get("status") == "provider_zero_confirmed" and zero.get("provider_zero_verified") is True
             and zero.get("continuing_spend_from_this_run") is False and not zero.get("blockers"), "source_not_closed")
    scope = profile["task_evaluation_run"]
    _require(lineage["source_run_id"] == scope["configuration_run_id"]
             and lineage["stable_intent_id"] == intent["intent_id"]
             and lineage["stable_intent_digest"] == intent["intent_digest"]
             and lineage["owner_id"] == intent["request"]["owner"]["user_id"]
             and lineage["organization_id"] == intent["request"]["owner"]["organization_id"]
             and lineage["scene_id"] == current["scene_id"] == scope["scene_id"]
             and lineage["task_id"] == current["task_id"] == scope["task_id"] == intent["request"]["task"]["task_id"]
             and descriptor.get("owner_intent_lineage") == lineage
             and descriptor.get("adoption_digest") == canonical_digest(descriptor, digest_field="adoption_digest")
             and descriptor.get("source_run_id") == lineage["source_run_id"]
             and descriptor.get("successor_run_id") == lineage["successor_run_id"]
             and descriptor.get("retained_runtime_archive") == {k: value["runtime_archive"][k] for k in ("sha256", "size_bytes")},
             "descriptor_lineage_invalid")


def stage_partial_astra_transport(*, selection_path, runtime, successor_run_id, owner_attempt_path, envelope):
    """Reopen authorities and original bytes, then copy only sealed transport data."""
    value = _read(selection_path, "transport_digest")
    descriptor = _read(value["descriptor"]["path"], "adoption_digest")
    validate_transport(value, descriptor)
    actual_owner = _read(owner_attempt_path, "owner_attempt_digest")
    validate_envelope_owner(envelope, actual_owner)
    _require(envelope["run_id"] == successor_run_id, "successor_run_changed")
    _require(actual_owner == value["authority_evidence"]["successor_owner_attempt"], "successor_owner_changed")
    lineage, proof = _source_identity(Path(value["source_launch_root"]), value["authority_evidence"]["successor_owner_attempt"],
                                     intent_root=value["intent_root"], successor_run_id=successor_run_id)
    _require(lineage == value["verified_lineage"] and proof == value["authority_evidence"]
             and _file(Path(value["source_archive"]["path"])) == value["source_archive"], "source_identity_changed")
    result = {"verified_lineage": lineage}
    for name in ("descriptor", "runtime_archive"):
        source = Path(value[name]["path"])
        _require(_file(source) == value[name], "transport_file_changed")
        destination = runtime / "input/partial_astra" / ("descriptor.json" if name == "descriptor" else "retained_runtime.zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied = _file(destination)
        _require(copied["sha256"] == value[name]["sha256"], "transport_copy_changed")
        result[name] = {"path": destination.relative_to(runtime).as_posix(), "digest": copied["sha256"], "size_bytes": copied["size_bytes"]}
    proof_path = runtime / "input/partial_astra/transport.json"
    _write(proof_path, value)
    result["transport"] = {"path": proof_path.relative_to(runtime).as_posix(), "digest": _sha(proof_path), "size_bytes": proof_path.stat().st_size}
    return result
