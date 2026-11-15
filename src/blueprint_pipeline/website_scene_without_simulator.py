"""Stage 5 of a website rigid job: record, on this host, that no simulator was run.

Owner decision 2026-11-15: Blueprint builds the scene and the package only; it runs no tests of
its own and rents no GPU. Stage 5 (``replacement_native_import_qualification``) used to
import the object into Isaac Sim on a rented GPU and drop it on a support; nothing in the package
a company downloads came from it (the object it passed on was stage 4's, byte for byte). For a
website rigid job it is now this ``no_spend`` adapter:

- it passes stage 4's statically checked object on unchanged, byte for byte;
- its receipt says what is true: no simulator ran (``native_isaac_executed: false``,
  ``native_simulator_import_qualified: false``, ``status: not_run``);
- its stage body (``website_native_background.execute_simulator_not_run``, beside the other website
  stage bodies) is kept apart so the readers here import nothing of the adapter runtime;
- the roles stay the ones stage 6, publication and the scene packet already read
  (``native_qualified_replacement_asset``, ``native_import_qualification_receipt``); publication,
  the last gate, verifies the record against stage 4's files. Every reader that pays for or scores a
  robot run refuses it (the rigid adapter by default); only the website package compile opts in
  (``accept_simulator_not_run``), because it needs the files, never as a qualification.

Articulated jobs (doors and drawers) and any recipe with a supplemental destination keep the Isaac
stage: ``prepare_construction_stages`` selects this adapter only for a rigid website subject, and
the adapter refuses anything else. Nothing here calls a provider or spends.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .immutable_file_digest import sha256_file

ADAPTER_ID = "website_simulator_not_run"
CAPABILITY = "replacement_native_import_qualification"
CONFIGURATION_SCHEMA = "replacement_simulator_not_run_configuration.v1"
RECEIPT_SCHEMA = "task_evaluation_replacement_simulator_not_run.v1"
NOT_RUN = "not_run"
#: The configured revision's ``replacement.physics_authority`` when stage 5 ran no simulator.
PHYSICS_AUTHORITY_NOT_RUN = "static_candidate_no_simulator_run"
QUALIFIED_RESULT_SCHEMA = "task_evaluation_replacement_native_import_result.v1"
REASON = "website_rigid_scene_package_built_without_a_simulator"
#: Robot-free readiness of a scene whose stage 5 ran no simulator: decided as not applicable, so
#: nothing reads it as "undecided, probe required" and no GPU probe is ever queued for it.
READINESS_STATUS = "simulator_not_run"


class SimulatorNotRunError(ValueError):
    """The stage-5 record cannot be written or read as the truth."""


def stage_five_configuration(*, replacement_identity: Mapping[str, Any]) -> dict[str, Any]:
    """The stage-5 configuration of a website rigid job: no simulator, by decision."""
    return {"schema_version": CONFIGURATION_SCHEMA, "replacement_identity": dict(replacement_identity),
            "simulator_run_requested": False, "reason": REASON,
            "physics_authority_on_completion": PHYSICS_AUTHORITY_NOT_RUN}


def configuration_refusal(configuration: Mapping[str, Any], envelope: Mapping[str, Any]) -> str | None:
    """The first predicate the configuration breaks, or None (``validate_immutable_stage_configurations``)."""
    recipe = envelope.get("recipe") or {}
    if configuration.get("schema_version") != CONFIGURATION_SCHEMA:
        return "schema_version"
    if configuration.get("replacement_identity") != recipe.get("subject_identity"):
        return "replacement_identity"
    if (configuration.get("simulator_run_requested") is not False or configuration.get("reason") != REASON
            or configuration.get("physics_authority_on_completion") != PHYSICS_AUTHORITY_NOT_RUN):
        return "simulator_not_run_declaration"
    if recipe.get("supplemental_destination") is not None:
        return "supplemental_destination_requires_native_import"
    if not ((envelope.get("request") or {}).get("scene") or {}).get("website_native_inputs"):
        return "website_job_required"
    return None


def not_run_receipt(*, replacement_identity: Mapping[str, Any], asset_digest: str,
                    static_qualification_digest: str) -> dict[str, Any]:
    value = {"schema_version": RECEIPT_SCHEMA, "status": NOT_RUN, "reason": REASON,
             "replacement_identity": dict(replacement_identity), "asset_digest": asset_digest,
             "static_qualification_digest": static_qualification_digest,
             "asset_is_stage_four_bytes": True, "simulator": None, "native_isaac_executed": False,
             "native_simulator_import_qualified": False, "support_contact_observed": None,
             "physics_authority": PHYSICS_AUTHORITY_NOT_RUN, "provider_mutations_performed": 0,
             "gpu_execution_performed": False, "blockers": [], "result_digest": ""}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def stage_five_receipt_kind(receipt: Any) -> str | None:
    """``qualified`` (a passed Isaac import), ``not_run`` (this record) or None for anything else.

    Readers that only need the files accept both by name; a reader that needs a simulator result
    accepts only ``qualified``.
    """
    if not isinstance(receipt, Mapping) or receipt.get("result_digest") != canonical_digest(
            receipt, digest_field="result_digest"):
        return None
    if (receipt.get("schema_version") == QUALIFIED_RESULT_SCHEMA and receipt.get("status") == "qualified"
            and receipt.get("native_isaac_executed") is True
            and receipt.get("native_simulator_import_qualified") is True and receipt.get("blockers") == []):
        return "qualified"
    if (receipt.get("schema_version") == RECEIPT_SCHEMA and receipt.get("status") == NOT_RUN
            and receipt.get("native_isaac_executed") is False
            and receipt.get("native_simulator_import_qualified") is False
            and receipt.get("gpu_execution_performed") is False
            and receipt.get("asset_is_stage_four_bytes") is True
            and receipt.get("simulator") is None and receipt.get("support_contact_observed") is None
            and receipt.get("physics_authority") == PHYSICS_AUTHORITY_NOT_RUN and receipt.get("blockers") == []):
        return NOT_RUN
    return None


def verify_record_binding(record: Mapping[str, Any], *, stage_four_asset: Path, static_receipt: Path,
                          replacement_asset: Path) -> None:
    """The record names stage 4's object and static receipt by digest, and the object passed on is those bytes.

    Stage 5 writes these fields; publication verifies them against the files it publishes
    (``asset_is_stage_four_bytes`` is required true by ``stage_five_receipt_kind``, and is true here
    only when the bytes are equal). Any mismatch is a typed refusal.
    """
    four = sha256_file(Path(stage_four_asset))
    if record.get("asset_digest") != four:
        raise SimulatorNotRunError("simulator_not_run_asset_digest_mismatch")
    if record.get("static_qualification_digest") != sha256_file(Path(static_receipt)):
        raise SimulatorNotRunError("simulator_not_run_static_qualification_digest_mismatch")
    if sha256_file(Path(replacement_asset)) != four:
        raise SimulatorNotRunError("simulator_not_run_asset_not_stage_four_bytes")


def _read_record(receipt_path: Path) -> Any:
    """The stage-5 receipt's JSON, or None when the file cannot be read as JSON."""
    try:
        return json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None


def _is_not_run(value: Any) -> bool:
    """This module's record (True), anything else (False); a forged record of its schema is refused."""
    if not isinstance(value, Mapping) or value.get("schema_version") != RECEIPT_SCHEMA:
        return False
    if stage_five_receipt_kind(value) != NOT_RUN:
        raise SimulatorNotRunError("simulator_not_run_record_invalid")
    return True


def is_not_run_record(receipt_path: Path) -> bool:
    """Whether a stage-5 receipt is this module's record that no simulator ran.

    False for anything else, which keeps the reader's existing Isaac path exactly as it was; a
    file that names this record's schema but is not a valid record is refused, never read as
    either.
    """
    return _is_not_run(_read_record(receipt_path))


def packet_ran_no_simulator(packet_dir: Path | None) -> bool:
    """Whether a compiled scene packet froze this module's record as its stage 5.

    Read from the packet's own request (``configured_task_template_adapter.source_documents``), so
    the answer does not depend on any receipt retained beside the launch. A packet without it is
    False, exactly as before.
    """
    try:
        frozen = json.loads((Path(packet_dir) / "native_task_arena_packet_request.v1.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, UnicodeError, ValueError):
        return False
    documents = (((frozen if isinstance(frozen, Mapping) else {}).get("configured_task_template_adapter") or {})
                 .get("source_documents") or {}).get("documents") or {}
    return stage_five_receipt_kind(documents.get("native_import_qualification")) == NOT_RUN


def physics_authority(receipt_path: Path, *, stage_four_asset: Path, static_receipt: Path,
                      replacement_asset: Path) -> str:
    """The configured revision's ``replacement.physics_authority`` for this stage-5 receipt.

    A receipt of the Isaac import schema stays ``qualified_replacement_asset`` as before (stage 5
    itself checked it). This module's record is verified against stage 4's files. A file that
    cannot be read, or of any other schema, is refused by name, never labelled "qualified".
    """
    value = _read_record(receipt_path)
    if value is None:
        raise SimulatorNotRunError("stage_five_receipt_unreadable")
    if _is_not_run(value):
        verify_record_binding(value, stage_four_asset=stage_four_asset, static_receipt=static_receipt,
                              replacement_asset=replacement_asset)
        return PHYSICS_AUTHORITY_NOT_RUN
    if isinstance(value, Mapping) and value.get("schema_version") == QUALIFIED_RESULT_SCHEMA:
        return "qualified_replacement_asset"
    raise SimulatorNotRunError("stage_five_receipt_schema_unknown")


__all__ = ["ADAPTER_ID", "CAPABILITY", "CONFIGURATION_SCHEMA", "NOT_RUN", "PHYSICS_AUTHORITY_NOT_RUN",
           "READINESS_STATUS", "RECEIPT_SCHEMA", "SimulatorNotRunError", "configuration_refusal",
           "is_not_run_record", "not_run_receipt", "packet_ran_no_simulator", "physics_authority",
           "stage_five_configuration", "stage_five_receipt_kind", "verify_record_binding"]
