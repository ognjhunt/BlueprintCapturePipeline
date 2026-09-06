"""Provision a rights-admitted public-scene persistent intent from a scene's
ALREADY-RETAINED artifacts, so the autonomous scene-progression worker can
re-prepare that scene (Spec A: the legacy public-scene path for scenes such as
841757).

This module ONLY transforms retained, already-validated artifacts into the
scene-progression persistent-intent shape:

  * a public-source binding (``task_evaluation_public_source_binding.v1``) that
    references the retained installation / publisher / source-preparation /
    destination / standard-splat-conversion receipts and the retained rights
    evidence, written into ``public_source_binding_root/<binding_id>.json``;
  * public-scene machinery (``task_evaluation_public_scene_machinery.v1``)
    assembled from the retained SAM provider profile + preparation config +
    review terms, written to ``machinery_path``;
  * a persistent ``public_scene`` intent staged through the authenticated intake
    (``stage_scene_intent``) whose source content-digest, task and consent are
    derived from the retained evidence.

It NEVER fabricates owner consent, source evidence, rights, or authority, never
installs a publisher asset, never queries a model, never uploads a source, and
never allocates a provider. Every authority/rights/task/source value traces to a
retained input; an absent or invalid required artifact fails closed with a typed
reason. The real ``materialize_public_scene_attempt`` factory (driven by the
scene-progression worker) is the sole consumer that turns this intent into a
publication-ready configuration submission, with no provider allocation.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import time
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_public_scene_attempt_factory import (
    BINDING_SCHEMA, MACHINERY_SCHEMA, PROVIDER_OPTIONS, PROVIDER_REFS, RELEASE_SCHEMA,
    SOURCE_REFS, public_source_content_digest, record,
)
from .task_evaluation_scene_configuration_submission_inputs import (
    SceneConfigurationSubmissionError, checked_file, read, sha,
)
from .task_evaluation_scene_owner_authority import task_contract_projection
from . import task_evaluation_scene_intake as intake

PROVISION_SCHEMA = "task_evaluation_public_scene_intent_provision.v1"
BINDING_STATUS = "admitted_for_private_processing"

#: Retained-artifact keys the provisioner requires and the typed refusal each
#: category raises when a member is absent or does not admit the intent.
SOURCE_EVIDENCE = ("installation_receipt", "publisher_intake", "source_preparation_receipt",
                   "destination_simready_result", "standard_splat_conversion_receipt")
RIGHTS_EVIDENCE = ("interiorgs_terms", "interiorgs_readme", "sage_readme")
PROVIDER_MACHINERY = ("sam_provider_profile", "sam_preparation_config", "review_terms")
_ABSENT_REASON = {**{key: "public_scene_source_evidence_absent" for key in SOURCE_EVIDENCE},
                  "accepted_task_request": "public_scene_task_authority_absent",
                  **{key: "public_scene_rights_absent" for key in RIGHTS_EVIDENCE},
                  **{key: "public_scene_provider_machinery_absent" for key in PROVIDER_MACHINERY},
                  "release_binding": "public_scene_release_absent"}
#: Preparation-config keys the factory rebinds from the release; refuse them here
#: so a retained config cannot smuggle a stale commit/profile into the machinery.
_PREPARATION_RESERVED = ("source_commit", "repo_root", "sam31_provider_profile_path",
                         "sam31_review_rights_attestation_path", "completed_prefix_adoption_path")


class PublicSceneIntentProvisionError(ValueError):
    """A required retained artifact is absent or does not admit the intent."""


def _fail(code: str) -> None:
    raise PublicSceneIntentProvisionError(code)


def _abs(path: str | Path) -> Path:
    item = Path(path)
    return item if item.is_absolute() else item.resolve()


def _reference(path: str | Path, *, reason: str) -> dict[str, Any]:
    """Build a {path, sha256, size_bytes} record for a retained file, failing
    closed with ``reason`` when the file is absent or unreadable."""
    item = _abs(path)
    if item.is_symlink() or not item.is_file():
        _fail(reason)
    try:
        return record(item)
    except (OSError, ValueError):
        _fail(reason)


def _load(path: str | Path, *, reason: str, digest_field: str | None = None) -> dict[str, Any]:
    """Read + digest-verify a retained JSON record, failing closed with ``reason``."""
    item = _abs(path)
    if item.is_symlink() or not item.is_file():
        _fail(reason)
    try:
        return read(item, digest_field=digest_field)
    except (SceneConfigurationSubmissionError, OSError, ValueError):
        _fail(reason)


def resolve_source_preparation_receipt(*, source_preparation_receipt_path: str | Path | None = None,
                                       installation_receipt_path: str | Path | None = None,
                                       task_objects: Sequence[Mapping[str, Any]] | None = None,
                                       expected_source_commit: str | None = None,
                                       approved_roots: Sequence[str | Path] | None = None,
                                       output_root: str | Path | None = None) -> Path:
    """Obtain the scene's source-preparation receipt.

    Reuses the canonical ``materialize_public_scene_source_preparation`` producer:
    with a retained receipt path the receipt is adopted verbatim after validating
    it is an unblocked ``public_scene_source_preparation.v1`` record; otherwise the
    producer derives it from the installed scene bytes (real USD collision
    inspection). Never fabricates identity.
    """
    from .public_scene_source_preparation import (
        SCHEMA_VERSION, PublicSceneSourcePreparationError,
        materialize_public_scene_source_preparation,
    )
    if source_preparation_receipt_path is not None:
        path = _abs(source_preparation_receipt_path)
        value = _load(path, reason="public_scene_source_evidence_absent", digest_field="receipt_digest")
        if value.get("schema_version") != SCHEMA_VERSION or value.get("blockers"):
            _fail("public_scene_source_evidence_absent")
        if value.get("status") != "source_context_prepared_pending_calibrated_views":
            _fail("public_scene_source_evidence_absent")
        return path
    if installation_receipt_path is None or task_objects is None or output_root is None:
        _fail("public_scene_source_evidence_absent")
    try:
        materialize_public_scene_source_preparation(
            installation_receipt_path=installation_receipt_path, task_objects=list(task_objects),
            expected_source_commit=expected_source_commit,
            approved_roots=tuple(approved_roots) if approved_roots else (Path(output_root).parent,),
            output_root=output_root)
    except PublicSceneSourcePreparationError:
        _fail("public_scene_source_evidence_absent")
    return _abs(Path(output_root) / (SCHEMA_VERSION + ".json"))


def _binding(*, retained: Mapping[str, Any], seed: Mapping[str, Any], installation: Mapping[str, Any],
             content_digest: str, binding_id: str, owner: Mapping[str, Any],
             rights_reference: str, source_preparation_path: Path) -> dict[str, Any]:
    """Assemble the public-source binding wrapper from retained references only."""
    references = {"source_preparation_receipt": _reference(
        source_preparation_path, reason="public_scene_source_evidence_absent")}
    for key in ("installation_receipt", "publisher_intake", "destination_simready_result",
                "standard_splat_conversion_receipt"):
        references[key] = _reference(retained[key], reason="public_scene_source_evidence_absent")
    for key in RIGHTS_EVIDENCE:
        references[key] = _reference(retained[key], reason="public_scene_rights_absent")
    if set(references) != SOURCE_REFS:  # defensive: the factory enforces this exactly.
        _fail("public_scene_source_evidence_absent")
    task = task_contract_projection(seed)
    binding = {"schema_version": BINDING_SCHEMA, "status": BINDING_STATUS, "binding_id": binding_id,
               "source_content_digest": content_digest, "publisher_scene_id": str(installation["scene_id"]),
               "owner": dict(owner), "rights_reference": rights_reference,
               "intent_task_digest": cross_runtime_canonical_digest(task),
               "accepted_task_seed": _reference(retained["accepted_task_request"],
                                                reason="public_scene_task_authority_absent"),
               "references": references}
    binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
    return binding


def _machinery(*, retained: Mapping[str, Any], profile_registry_root: str | Path,
               maximum_preparation_spend_usd: float) -> dict[str, Any]:
    """Assemble the release/provider-bound machinery from the retained SAM profile."""
    provider = _load(retained["sam_provider_profile"], reason="public_scene_provider_machinery_absent")
    try:
        provider_references = {
            "worker_stack_manifest": {k: provider["worker_stack_manifest"][k]
                                      for k in ("path", "sha256", "size_bytes")},
            "runtime_image_build_receipt": {k: provider["runtime_image_build_receipt"][k]
                                            for k in ("path", "sha256", "size_bytes")},
            **{new: {k: provider["authorization_sources"][old][k] for k in ("path", "sha256", "size_bytes")}
               for new, old in (("license_use_authorization", "license_use"),
                                ("privacy_use_authorization", "privacy_use"),
                                ("trade_controls_review", "trade_controls"))}}
        provider_options = {key: provider["compile" if key == "compile_model" else key]
                            for key in PROVIDER_OPTIONS}
    except (KeyError, TypeError):
        _fail("public_scene_provider_machinery_absent")
    if set(provider_references) != PROVIDER_REFS or set(provider_options) != PROVIDER_OPTIONS:
        _fail("public_scene_provider_machinery_absent")
    preparation = _load(retained["sam_preparation_config"], reason="public_scene_provider_machinery_absent")
    if any(key in preparation for key in _PREPARATION_RESERVED):
        _fail("public_scene_provider_machinery_absent")
    if not isinstance(preparation.get("approved_roots"), list) or "server_data_root" not in preparation:
        _fail("public_scene_provider_machinery_absent")
    if not (isinstance(maximum_preparation_spend_usd, (int, float))
            and not isinstance(maximum_preparation_spend_usd, bool)
            and maximum_preparation_spend_usd >= 4.5):
        _fail("public_scene_provider_machinery_absent")
    # A2: the factory's completed-prefix adoption path reads these host roots
    # directly from the machinery. Emit the canonical roots so a re-attempt of the
    # same owner intent can adopt already-completed GPU stages (never repeat paid
    # work) rather than KeyError on a missing machinery key. The fresh provider-zero
    # that gates a real adoption is produced live at paid time, never baked here.
    from .task_evaluation_sam31_prefix_adoption import DEFAULT_QUEUE, DEFAULT_PARENT_QUEUE, DEFAULT_EXECUTION
    from .task_evaluation_release_retention import DEFAULT_EVIDENCE_BINDING_ROOT
    machinery = {"schema_version": MACHINERY_SCHEMA,
                 "maximum_preparation_spend_usd": maximum_preparation_spend_usd,
                 "provider_references": provider_references, "provider_options": provider_options,
                 "preparation": preparation,
                 "review_terms": _reference(retained["review_terms"],
                                            reason="public_scene_provider_machinery_absent"),
                 "child_queue_root": str(DEFAULT_QUEUE), "parent_queue_root": str(DEFAULT_PARENT_QUEUE),
                 "execution_root": str(DEFAULT_EXECUTION),
                 "release_retention_binding_root": str(DEFAULT_EVIDENCE_BINDING_ROOT),
                 "profile_registry_root": str(profile_registry_root)}
    machinery["machinery_digest"] = canonical_digest(machinery, digest_field="machinery_digest")
    return machinery


def _validate_release(retained: Mapping[str, Any]) -> dict[str, Any]:
    release = _load(retained["release_binding"], reason="public_scene_release_absent",
                    digest_field="release_digest")
    if release.get("schema_version") != RELEASE_SCHEMA:
        _fail("public_scene_release_absent")
    commit = release.get("source_commit")
    if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
        _fail("public_scene_release_absent")
    return release


def _validate_conversion(retained: Mapping[str, Any], commit: str) -> None:
    """The retained standard-splat conversion must be a structurally valid receipt.

    Content identity, not commit identity. The standard-splat converter needs the
    3DGS decoder, which exists only in the decoder-equipped SAM preparation, never
    on the control plane where provisioning runs. The preparation (the public-scene
    factory) rebinds the conversion at ``release.source_commit`` when it drifts, so
    requiring the retained receipt to already name this release is both
    unsatisfiable at provisioning time and contrary to the content-identity goal --
    a deploy must not invalidate a retained per-scene document (piece 1). Validate
    the receipt's structure and that it names a real commit; leave the exact-release
    rebind to the decoder-equipped preparation. ``commit`` is accepted for context
    but no longer required to equal the recorded conversion commit.
    """
    conversion = _load(retained["standard_splat_conversion_receipt"],
                       reason="public_scene_source_evidence_absent", digest_field="receipt_digest")
    recorded = conversion.get("repository", {}).get("commit")
    if (conversion.get("schema_version") != "standard_splat_conversion_receipt.v1"
            or not (isinstance(recorded, str) and len(recorded) == 40
                    and all(c in "0123456789abcdef" for c in recorded))
            or not isinstance(conversion.get("output"), Mapping)
            or not isinstance(conversion.get("source"), Mapping)):
        _fail("public_scene_source_evidence_absent")


def _validate_seed_and_disclosures(retained: Mapping[str, Any], installation: Mapping[str, Any],
                                   commit: str) -> dict[str, Any]:
    seed = _load(retained["accepted_task_request"], reason="public_scene_task_authority_absent")
    identity = seed.get("task_identity")
    authority = seed.get("human_authority")
    if not (isinstance(identity, Mapping) and intake._identifier(str(identity.get("id", "")))
            and seed.get("strategy") == "pick_and_place"
            and all(isinstance(seed.get(part), Mapping) and bool(seed.get(part))
                    for part in ("subject", "support", "destination", "success"))
            and isinstance(authority, Mapping)):
        _fail("public_scene_task_authority_absent")
    if (seed.get("appearance_removal_method") != "sam31"
            or str(seed.get("publisher_scene_id")) != str(installation.get("scene_id"))):
        _fail("public_scene_task_authority_absent")
    disclosures = authority.get("full_source_provider_disclosure_authorities")
    if not (isinstance(disclosures, Mapping) and disclosures):
        _fail("public_scene_rights_absent")
    for reference in disclosures.values():
        if not isinstance(reference, Mapping):
            _fail("public_scene_rights_absent")
        try:
            checked_file(reference.get("path", ""), reference)
        except (SceneConfigurationSubmissionError, OSError, ValueError, TypeError):
            _fail("public_scene_rights_absent")
    return seed


def _put(path: Path, value: Mapping[str, Any]) -> Path:
    path = _abs(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        _fail("public_scene_provision_output_unsafe")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            _fail("public_scene_provision_immutable_conflict")
        return path
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return path


def provision_public_scene_intent(*, retained: Mapping[str, Any], owner_authority: Mapping[str, Any],
                                  binding_id: str, profile_registry_root: str | Path,
                                  maximum_preparation_spend_usd: float,
                                  intent_root: str | Path, public_source_binding_root: str | Path,
                                  machinery_output_path: str | Path,
                                  authenticated_client: str = "blueprint-webapp",
                                  trusted_clients: Sequence[str] = ("blueprint-webapp",),
                                  now: float | None = None) -> dict[str, Any]:
    """Transform a scene's retained artifacts into a persistent public_scene intent.

    ``retained`` maps every required retained-artifact key (see ``_ABSENT_REASON``)
    to its path. ``owner_authority`` carries the owner's standing intent authority
    ({owner, submission_id, execution, consent}); the provisioner binds its
    source content-digest, task and provider-terms reference to the retained
    evidence and never invents them. Fails closed with a typed reason when any
    required retained artifact is absent or invalid.
    """
    moment = time.time() if now is None else now
    if not intake._identifier(binding_id):
        _fail("public_scene_binding_id_invalid")
    for key, reason in _ABSENT_REASON.items():
        if key not in retained:
            _fail(reason)

    installation = _load(retained["installation_receipt"], reason="public_scene_source_evidence_absent",
                         digest_field="receipt_digest")
    if (installation.get("schema_version") != "public_scene_host_input_installation_receipt.v1"
            or installation.get("status") != "installed" or installation.get("service_readable") is not True):
        _fail("public_scene_source_evidence_absent")
    try:
        content_digest = public_source_content_digest(installation)
    except (ValueError, KeyError, TypeError):
        _fail("public_scene_source_evidence_absent")

    release = _validate_release(retained)
    _validate_conversion(retained, release["source_commit"])
    seed = _validate_seed_and_disclosures(retained, installation, release["source_commit"])

    # Rights evidence + review terms must exist before any owner authority binds.
    for key in RIGHTS_EVIDENCE:
        _reference(retained[key], reason="public_scene_rights_absent")
    review_path = _abs(retained["review_terms"])
    if review_path.is_symlink() or not review_path.is_file():
        _fail("public_scene_provider_machinery_absent")
    review_terms_digest = sha(review_path)

    source_preparation_path = resolve_source_preparation_receipt(
        source_preparation_receipt_path=retained["source_preparation_receipt"])

    owner = owner_authority.get("owner")
    consent = dict(owner_authority.get("consent") or {})
    execution = owner_authority.get("execution")
    submission_id = owner_authority.get("submission_id")
    if not (isinstance(owner, Mapping) and isinstance(execution, Mapping) and consent):
        _fail("public_scene_owner_authority_invalid")
    rights_reference = consent.get("rights_reference")
    if not (isinstance(rights_reference, str) and rights_reference):
        _fail("public_scene_owner_authority_invalid")
    # The owner's accepted provider-terms reference is VALIDATED against the
    # retained review terms, never manufactured. If the owner's recorded consent
    # does not already reference exactly the retained review-terms bytes, we have
    # no evidence they accepted these provider terms and fail closed rather than
    # rewriting consent to fabricate acceptance (the downstream owner-authority
    # check would otherwise be vacuous, since we would have set the very value it
    # compares against).
    accepted_terms = consent.get("provider_terms_reference")
    if not (isinstance(accepted_terms, str) and accepted_terms):
        _fail("public_scene_owner_authority_invalid")
    if accepted_terms != review_terms_digest:
        _fail("public_scene_provider_terms_not_accepted")

    binding = _binding(retained=retained, seed=seed, installation=installation,
                       content_digest=content_digest, binding_id=binding_id, owner=owner,
                       rights_reference=rights_reference, source_preparation_path=source_preparation_path)
    machinery = _machinery(retained=retained, profile_registry_root=profile_registry_root,
                           maximum_preparation_spend_usd=maximum_preparation_spend_usd)

    request = {"schema_version": intake.REQUEST_SCHEMA, "submission_id": submission_id, "owner": dict(owner),
               "source": {"kind": "public_scene", "binding_id": binding_id, "content_digest": content_digest},
               "task": task_contract_projection(seed), "execution": dict(execution), "consent": consent}

    # A11: publish the immutable binding + machinery the worker resolves BEFORE
    # staging the persistent intent that makes the scene visible to the worker.
    # An immutable conflict or I/O failure on a dependency then leaves no dangling
    # intent; re-running the provisioner is idempotent (byte-identical _put/stage).
    binding_path = _put(_abs(public_source_binding_root) / (binding_id + ".json"), binding)
    machinery_path = _put(_abs(machinery_output_path), machinery)
    intent = intake.stage_scene_intent(value=request, queue_root=intent_root,
                                       authenticated_client=authenticated_client,
                                       trusted_clients=set(trusted_clients), now=moment)
    return {"schema_version": PROVISION_SCHEMA, "status": "public_scene_intent_provisioned",
            "intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"],
            "binding_id": binding_id, "binding_digest": binding["binding_digest"],
            "source_content_digest": content_digest, "source_commit": release["source_commit"],
            "binding_path": str(binding_path), "machinery_path": str(machinery_path),
            "provider_mutation_performed": False}


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision a public_scene persistent intent.")
    parser.add_argument("--retained-manifest", required=True,
                        help="JSON mapping every retained-artifact key to its absolute path.")
    parser.add_argument("--owner-authority", required=True,
                        help="JSON owner standing authority ({owner, submission_id, execution, consent}).")
    parser.add_argument("--binding-id", required=True)
    parser.add_argument("--profile-registry-root", required=True)
    parser.add_argument("--maximum-preparation-spend-usd", type=float, default=4.5)
    parser.add_argument("--intent-root", required=True)
    parser.add_argument("--public-source-binding-root", required=True)
    parser.add_argument("--machinery-output-path", required=True)
    parser.add_argument("--authenticated-client", default="blueprint-webapp")
    parser.add_argument("--trusted-client", action="append", dest="trusted_clients",
                        default=None, help="Repeatable; defaults to blueprint-webapp.")
    args = parser.parse_args(argv)
    result = provision_public_scene_intent(
        retained=_load_json(args.retained_manifest), owner_authority=_load_json(args.owner_authority),
        binding_id=args.binding_id, profile_registry_root=args.profile_registry_root,
        maximum_preparation_spend_usd=args.maximum_preparation_spend_usd, intent_root=args.intent_root,
        public_source_binding_root=args.public_source_binding_root,
        machinery_output_path=args.machinery_output_path, authenticated_client=args.authenticated_client,
        trusted_clients=tuple(args.trusted_clients or ("blueprint-webapp",)))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
