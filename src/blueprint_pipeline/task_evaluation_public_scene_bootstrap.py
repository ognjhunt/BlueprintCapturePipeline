"""Let the existing scene controller prepare a newly selected public source.

Only publisher reads and the existing CPU installer/source-preparation producer
run here. No provider allocation, model invocation, capture relabeling, or
destination asset fabrication is possible through this source-resolution seam.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_public_scene_catalog import source_choice
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, sha
from .task_evaluation_scene_progression_state import atomic_json, require, safe_path

SCHEMA = "task_evaluation_public_scene_bootstrap.v1"


class _PublisherRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        uri = urlsplit(newurl)
        require(uri.scheme == "https" and not uri.username and not uri.password and uri.hostname
                and any(uri.hostname == host or uri.hostname.endswith("." + host)
                        for host in ("huggingface.co", "hf.co", "xethub.hf.co", "amazonaws.com", "kujiale.com")),
                "public_source_redirect_refused")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def _download(row, path):
    """Exact bounded publisher GET; partial transfers never become installed input."""
    from .model_access_env import normalize_model_access_env
    normalize_model_access_env()
    headers = {}
    if urlsplit(row["publisher_url"]).hostname == "huggingface.co" and os.environ.get("HF_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
    descriptor, temporary = tempfile.mkstemp(prefix=".publisher-", dir=path.parent)
    try:
        digest, count = hashlib.sha256(), 0
        with os.fdopen(descriptor, "wb") as stream:
            with build_opener(_PublisherRedirect()).open(Request(row["publisher_url"], headers=headers), timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    require(count <= row["size_bytes"], "public_source_download_size_exceeded")
                    digest.update(chunk)
                    stream.write(chunk)
            require(count == row["size_bytes"] and "sha256:" + digest.hexdigest() == row["sha256"],
                    "public_source_download_digest_mismatch")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o440)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _ref(path):
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def prepare_registered_public_scene(*, intent, config, release, downloader=None):
    from .control_plane_disk_budget import reserve_control_plane_disk
    choice = source_choice(intent["request"]["source"]["binding_id"], catalog_path=config.get("public_source_catalog_path"))
    # Download + archive + atomic installer copies + collision derivative. This
    # is a source-only reservation; downstream whole-chain admission still runs
    # before any execution attempt can be reserved or submitted.
    expected = 4 * sum(r["size_bytes"] for r in [*choice["files"], *choice["rights"]["evidence"]]) + 64 * 1024**2
    with reserve_control_plane_disk("launch_preparation", target_root=config["factory_output_root"],
            expected_bytes=expected, reservation_root=(config.get("preparation_worker") or {}).get(
                "disk_reservation_root", "/var/lib/blueprint/pipeline-control-plane/disk-reservations")):
        return _prepare_registered_public_scene(intent=intent, config=config, release=release, downloader=downloader)


def _prepare_registered_public_scene(*, intent, config, release, downloader=None):
    """Return replayable source progress to the caller that owns the intent lock."""
    from .public_scene_host_input_intake import (
        _archive_for_request, install_packet_archive, _verified_checkout_head,
    )
    from .public_scene_source_preparation import materialize_public_scene_source_preparation, _cached_result
    from .task_evaluation_scene_progression import SourceResolution

    request = intent["request"]
    choice = source_choice(request["source"]["binding_id"], catalog_path=config.get("public_source_catalog_path"))
    require(request["source"]["kind"] == "public_scene"
            and request["source"]["content_digest"] == choice["source_content_digest"]
            and request["consent"]["rights_reference"] == choice["rights_reference"]
            and request["consent"]["private_processing_authorized"] is True,
            "public_source_choice_or_consent_mismatch")
    require(_verified_checkout_head() == release["source_commit"], "public_source_execution_commit_mismatch")
    for role in ("subject", "support"):
        require(str(request["task"][role].get("source_instance_id"))
                == str(choice["task_proposal"][role]["source_instance_id"]), "public_source_task_selection_mismatch")
    root = safe_path(Path(config["factory_output_root"]) / intent["intent_id"] / "public-source")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    state_path = root / "bootstrap_progress.json"
    binding = {"intent_digest": intent["intent_digest"], "choice_digest": choice["choice_digest"],
               "source_content_digest": choice["source_content_digest"]}

    def progress(phase, **extra):
        value = {"schema_version": SCHEMA, **binding, "phase": phase,
                 "source_commit": release["source_commit"], "provider_allocation_performed": False, **extra}
        value["progress_digest"] = canonical_digest(value, digest_field="progress_digest")
        atomic_json(state_path, value)

    completion = root / "public_source_preparation.json"
    downloads = root / "publisher-downloads"
    downloads.mkdir(exist_ok=True, mode=0o750)
    paths = {}
    for row in [*choice["files"], *choice["rights"]["evidence"]]:
        suffix = Path(urlsplit(row["publisher_url"]).path).suffix
        path = downloads / (row["role"] + suffix)
        if not path.exists():
            progress("publisher_download", current_role=row["role"], verified_roles=sorted(paths))
            (downloader or _download)(row, path)
        checked_file(path, row)
        paths[row["role"]] = path
    task_objects = [{"role": role, "source_instance_id": str(request["task"][key]["source_instance_id"])}
                    for role, key in (("movable_subject", "subject"), ("source_support", "support"))]
    if completion.exists():
        value = read(completion, digest_field="receipt_digest")
        require(all(value.get(k) == v for k, v in binding.items()), "public_source_retained_binding_changed")
        for ref in value["references"].values():
            checked_file(ref["path"], ref)
        prepared = read(value["references"]["source_preparation_receipt"]["path"], digest_field="receipt_digest")
        installation = read(value["references"]["installation_receipt"]["path"], digest_field="receipt_digest")
        installed_files = {r["role"]: Path(installation["destination_root"]) / r["relative_path"]
                           for r in installation["files"] if r.get("role")}
        for row in installation["files"]:
            checked_file(Path(installation["destination_root"]) / row["relative_path"], row)
        _cached_result(Path(value["references"]["source_preparation_receipt"]["path"]).parent,
            source_commit=prepared["source_commit"], installation_digest=installation["receipt_digest"],
            task_digest=canonical_digest({"task_objects": task_objects}), source_files=installed_files)
    else:
        progress("source_installation")
        authority = {"schema_version": "public_scene_rights_authority.v1", "status": "approved_for_internal_use",
            "agent_accepted_terms": False, "authority_source": "authenticated_scene_intent",
            "intent_digest": intent["intent_digest"], "accepted_by": request["consent"]["accepted_by"],
            "accepted_at_epoch": request["consent"]["accepted_at_epoch"], "rights_reference": choice["rights_reference"],
            "use_scope": choice["rights"]["use_scope"], "raw_redistribution_allowed": False,
            "provider_training_authorized": False, "authorized_source_sha256": [r["sha256"] for r in choice["files"]]}
        rights_path = root / "local_import_rights.json"
        if not rights_path.exists():
            atomic_json(rights_path, authority)
        require(read(rights_path) == authority, "public_source_import_authority_changed")
        packet_id = "source-" + choice["source_content_digest"][7:31]
        host_request = {"schema_version": "public_scene_host_input_request.v2", "program_id": "arm-decision-proof-v1",
            "adp_item": "ADP-009D", "scene_id": choice["publisher_scene_id"], "packet_id": packet_id,
            "source_commit_sha": release["source_commit"],
            "rights_receipts": [{"receipt_id": "owner-local-import", **_ref(rights_path)}],
            "files": [{"role": r["role"], "path": str(paths[r["role"]]), "sha256": r["sha256"],
                       "rights_receipt_ids": ["owner-local-import"]} for r in choice["files"]]}
        host_request_path = root / "host_input_request.json"
        installation_path = root / "installed" / packet_id / "public_scene_host_input_installation_receipt.v1.json"
        if not installation_path.exists():
            atomic_json(host_request_path, host_request)
            archive, _packet = _archive_for_request(host_request_path)
            try:
                install_packet_archive(archive, destination_root=root / "installed", allowed_roots=(root,),
                                       service_account=config.get("service_account", "blueprint"))
            finally:
                archive.close()
        installation = read(installation_path, digest_field="receipt_digest")
        publisher = {"schema_version": "public_scene_publisher_source_intake.v1", "scene_id": choice["publisher_scene_id"],
            "status": "publisher_pinned_sources_verified_on_production", "publisher_direct_download": True,
            "source_uploaded_by_blueprint": False, "public_redistribution_allowed": False,
            "artifacts": [{k: v for k, v in row.items() if k != "role"} for row in choice["files"]]}
        publisher_path = root / "publisher_intake.json"
        atomic_json(publisher_path, publisher)
        progress("source_geometry_preparation")
        prepared_root = root / "prepared"
        prepared_path = prepared_root / "public_scene_source_preparation.v1.json"
        if not prepared_path.exists():
            materialize_public_scene_source_preparation(installation_receipt_path=installation_path,
                task_objects=task_objects, expected_source_commit=release["source_commit"],
                output_root=prepared_root, approved_roots=(root,))
        prepared = read(prepared_path, digest_field="receipt_digest")
        require(prepared.get("status") == "source_context_prepared_pending_calibrated_views" and not prepared.get("blockers"),
                "public_source_geometry_preparation_blocked")
        value = {"schema_version": SCHEMA, **binding, "source_commit": release["source_commit"],
            "status": "source_prepared_pending_configuration_binding", "provider_allocation_performed": False,
            "references": {"installation_receipt": _ref(installation_path), "publisher_intake": _ref(publisher_path),
                           "source_preparation_receipt": _ref(prepared_path),
                           **{role: _ref(paths[role]) for role in ("interiorgs_terms", "interiorgs_readme", "sage_readme")}}}
        value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
        atomic_json(completion, value)
    progress("source_prepared", preparation_receipt=_ref(completion))
    # The same resolver will consume the installed binding as soon as the
    # configuration producer can honestly supply it. No invented tray/result.
    return SourceResolution("awaiting_source", blockers=("public_scene_configuration_binding_required",),
                            analysis_reference=_ref(completion))
