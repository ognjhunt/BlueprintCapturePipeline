"""A completed preparation is re-issued as a new job only under the sponsor's renewal of its grant.

capture-447ae4f4 (2026-11-02): the preparation completed, its grant expired and
its intent ended; the sponsor renewed the grant. The re-issue reopens retained
inputs, recomputes only the digest-bound native inputs and hands the renewed
request to the normal intake. It buys, reserves and reconstructs nothing, and
every fixture here is synthetic.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import website_preparation_authority as authority
from blueprint_pipeline import website_preparation_renewal as renewals
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.task_evaluation_retained_controls_evidence import DIRECTORY, UNSTARTED_SCHEMA
from blueprint_pipeline.task_evaluation_scene_intake import (
    SceneIntakeError, reserve_scene_attempt, revoke_scene_intent, stage_scene_intent)
from blueprint_pipeline.website_preparation_revision_identity import RENEWAL_FIELD, capture_id, renewal
from blueprint_pipeline.website_task_family_packet import RUNTIME_SCHEMA

NOW = 1_800_000_000.0  # the prior handoff
RENEWED = NOW + 10_000  # after the prior grant expired at NOW + 3600
OWNER = {"user_id": "owner-1", "organization_id": "org-1"}
VECTOR = json.loads((Path(__file__).parent / "fixtures/website-preparation-renewal.v1.json").read_text())
ATLAS = "blueprint_pipeline.website_atlas_scene_source.atlas_scene_source"


def _seal(value, field):
    value[field] = canonical_digest(value, digest_field=field)
    return value


def _grant(context, *, accepted, parent, renewed=None):
    value = {"schema_version": "website_scene_sponsorship.v1", "sponsor": "blueprint", "purpose": "scene_preparation",
             "request_id": context["request_id"], "scene_id": context["scene_id"], "capture_id": context["capture_id"],
             "task_context_digest": context["context_digest"], "owner": dict(OWNER),
             "consent": {"accepted_by": OWNER["user_id"], "accepted_at_epoch": accepted,
                         "rights_reference": cross_runtime_canonical_digest(context["capture_rights"]),
                         "provider_terms_reference": "sha256:" + "7" * 64, "private_processing_authorized": True,
                         "provider_training_authorized": False, "task_confirmed": True, "spend_authorized": True},
             "preparation_max_total_spend_usd": 30, "upstream_max_spend_usd": 5, "max_total_spend_usd": 20,
             "max_paid_attempts": 2, "expires_at_epoch": accepted + 3600, "policy_digest": "sha256:" + "8" * 64,
             "continuation_parent_authority_digest": parent}
    if renewed is not None:
        value["renewal"] = renewed
    return _seal(value, "authority_digest")


def renewed_grant(old, *, accepted=RENEWED - 50, renewed=None, **changes):
    """The continuation of the sponsor's renewal of ``old``'s stored grant."""
    block = {"parent_authority_digest": old["continuation_parent_authority_digest"],
             "parent_expires_at_epoch": old["expires_at_epoch"], "approved_by": OWNER["user_id"],
             "approval_reference": "synthetic owner approval"}
    value = {key: item for key, item in old.items() if key != "authority_digest"}
    value.update(consent={**old["consent"], "accepted_at_epoch": accepted}, expires_at_epoch=accepted + 3600,
                 continuation_parent_authority_digest="sha256:" + "b" * 64,
                 renewal=block if renewed is None else renewed)
    value.update(changes)
    if value["renewal"] is None:
        del value["renewal"]
    return _seal(value, "authority_digest")


def cancel_unstarted(directory, row):
    original = {"schema_version": "task_evaluation_launch_receipt.v1", "status": "blocked",
                "source_commit": row["source_commit"]}
    original["receipt_digest"] = cross_runtime_canonical_digest(original, digest_field="receipt_digest")
    receipt = _seal({"schema_version": UNSTARTED_SCHEMA, "status": "cancelled_before_controls_eligibility",
                     "attempt_digest": row["attempt_digest"], "intent_digest": row["intent_digest"],
                     "attempt_id": row["attempt_id"], "maximum_spend_usd": row["maximum_spend_usd"],
                     "provider": row["provider"], "original_blocked_launch_receipt": original,
                     "downstream_execution_eligible": False, "provider_mutation_performed": False}, "receipt_digest")
    path = directory / DIRECTORY / (row["attempt_id"] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))
    return path


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """One real handoff, registered and enqueued under a grant that has since expired; built once."""
    from blueprint_pipeline.gaussian_splat_decode import write_standard_3dgs_ply
    from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
    from blueprint_pipeline.website_scene_handoff import prepare_website_scene_handoff
    from tests.test_particlefield_usd import _splat
    from tests.test_website_task_preparation import _arguments
    root = tmp_path_factory.mktemp("renewal")
    capture, intents = root / "capture", root / "intents"
    pipeline = capture / "pipeline"
    pipeline.mkdir(parents=True)
    args = _arguments(pipeline)
    data, _ = _splat(32)
    data.scales[:] = -5
    splat = Path(args["base_scene"]["splat_path"])
    write_standard_3dgs_ply(data, splat)
    digest = _sha256_file(splat)
    base = {**args["base_scene"], "splat_digest": digest, "splat_binding_id": "website-splat-" + digest[7:39],
            "provider": "world_labs_atlas"}
    context = args["task_context"]
    context["capture_rights"] = {"derived_scene_generation_allowed": True, "consent_revoked": False}
    _seal(context, "context_digest")
    old = _grant(context, accepted=NOW - 100, parent="sha256:" + "a" * 64)
    geometry = _seal({**args["source_geometry"], "geometry_backend": "world_labs_atlas"}, "digest")
    masks = _seal({**args["task_masks"], "source_geometry_digest": geometry["digest"]}, "digest")
    atlas = {"status": "ready", "base_scene": base, "source_geometry": geometry, "alignment": "anchored",
             "source_pose_digest": "sha256:" + "1" * 64, "background_pose_digest": "sha256:" + "2" * 64}
    removal = pipeline / "removal.json"
    write_json(removal, args["removal_manifest"])
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", raising=False)
        patch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(intents))
        patch.setattr(ATLAS, lambda **_: atlas)
        patch.setattr("blueprint_pipeline.website_task_masks.bind_task_masks_to_geometry", lambda **_: masks)
        patch.setattr("blueprint_pipeline.website_task_context.enqueue_website_prepared_scene", lambda **kw: {
            "id": "scene-outbox-prior", "state": "forward_pending",
            "request_digest": cross_runtime_canonical_digest(kw["request"])})
        handoff = prepare_website_scene_handoff(descriptor={"capture_id": context["capture_id"],
            "scene_id": context["scene_id"], "metadata": {"site_task_context": context,
                                                          "website_scene_execution_authority": old}},
            clean_plate={"privacy_verified": True, "status": "objects_removed", "task_masks": args["task_masks"],
                         "source_geometry": None, "removal_manifest_path": str(removal)},
            capture_root=capture, now=NOW)
    assert handoff["status"] == "intake_ready" and handoff["website_intake_outbox"], handoff
    write_json(pipeline / "website_scene_sponsorship.json", old)
    prior_path = Path(handoff["preparation_path"])
    prior = json.loads(prior_path.read_text())
    request = prior["intake_request"]
    staged = stage_scene_intent(value=request, queue_root=intents, authenticated_client="webapp",
                                trusted_clients={"webapp"}, now=NOW + 1)
    row = reserve_scene_attempt(queue_root=intents, intent_id=staged["intent_id"], attempt_id="a1",
        source_commit="d" * 40, runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
        provider="vast", maximum_spend_usd=3, now=NOW + 2)
    directory = intents / staged["intent_id"]
    cancel_unstarted(directory, row)
    revoke_scene_intent(queue_root=intents, intent_id=staged["intent_id"], intent_digest=staged["intent_digest"],
                        owner=request["owner"], now=NOW + 3)
    return SimpleNamespace(root=root, capture=capture, intents=intents, bindings=root / "website-source-bindings",
                           context=context, old=old, atlas=atlas, handoff=handoff, prior_path=prior_path,
                           prior=prior, request=request, intent=staged, row=row, intent_dir=directory)


PAID = ("blueprint_pipeline.website_task_preparation.compile_website_scene_preparation",
        "blueprint_pipeline.website_scene_handoff.compile_website_scene_preparation",
        "blueprint_pipeline.website_scene_handoff.prepare_website_scene_handoff",
        "blueprint_pipeline.website_target_background_worker.queue_correction",
        "blueprint_pipeline.website_assembly_coverage.attach_assembly_coverage",
        "blueprint_pipeline.website_object_spec_research.attach_object_specs",
        "blueprint_pipeline.website_task_masks.complete_retained_static_task_masks",
        "blueprint_pipeline.website_operation_coverage_review.clarify_operation_coverage",
        "blueprint_pipeline.website_task_context.reserve_website_preparation_spend",
        "blueprint_pipeline.website_task_context.reserve_website_sam_spend",
        "blueprint_pipeline.website_task_context.website_webapp_request",
        "blueprint_pipeline.website_atlas_pipeline.run_atlas_pipeline",
        "blueprint_pipeline.website_atlas_preview.advance_job",
        "blueprint_pipeline.pubsub_handoff_listener.process_handoff_payload",
        "blueprint_pipeline.paid_resource_admission.require_paid_resource_admission")


@pytest.fixture
def site(template, tmp_path, monkeypatch):
    """Per-test queue and registrations; the shared capture is restored after each test."""
    from blueprint_pipeline import website_scene_dispatch as dispatch
    queue = tmp_path / "intents"
    shutil.copytree(template.intents, queue)
    shutil.copytree(template.bindings, tmp_path / "website-source-bindings")
    monkeypatch.delenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", raising=False)
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT", str(queue))
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "webapp")
    monkeypatch.setattr(ATLAS, lambda **_: template.atlas)

    def paid(*_args, **_kwargs):
        raise AssertionError("a re-issue never reaches a paid or reconstructing helper")

    for target in PAID:
        monkeypatch.setattr(target, paid)
    registered, enqueued, saved = [], [], {}
    real_register = dispatch.register_website_preparation

    def register(**kwargs):
        registered.append(kwargs)
        return real_register(**kwargs)

    monkeypatch.setattr(dispatch, "register_website_preparation", register)
    state = SimpleNamespace(**vars(template), queue=queue, bindings_root=tmp_path / "website-source-bindings",
                            fresh=renewed_grant(template.old), registered=registered, enqueued=enqueued,
                            prior_intent_dir=queue / template.intent["intent_id"])

    def enqueue(**kwargs):
        enqueued.append(copy.deepcopy(kwargs))
        return {"id": "scene-outbox-renewal", "state": "forward_pending",
                "request_digest": cross_runtime_canonical_digest(kwargs["request"])}

    def replace(path, value):
        saved.setdefault(path, path.read_bytes())
        path.write_text(json.dumps(value))

    state.enqueue, state.replace = enqueue, replace
    yield state
    for path, raw in saved.items():
        path.write_bytes(raw)
    shutil.rmtree(template.prior_path.parent / "renewals", ignore_errors=True)


def run(site, *, grant=None, context=None, now=RENEWED, enqueue=None):
    calls = []

    def load_context(**kwargs):
        calls.append(("task-context", kwargs))
        return copy.deepcopy(context or site.context)

    def load_sponsorship(**kwargs):
        calls.append(("scene-sponsorship", {key: kwargs[key] for key in ("now", "create")}))
        return copy.deepcopy(grant or site.fresh)

    result = renewals.reissue_renewed_preparation(capture_root=site.capture, queue_root=site.queue, now=now,
        load_context=load_context, load_sponsorship=load_sponsorship, enqueue=enqueue or site.enqueue)
    assert ("scene-sponsorship", {"now": now, "create": False}) in calls
    return result


def retained(site):
    paths = [site.capture / "pipeline/website_scene_preparation/handoff.json", site.prior_path,
             Path(site.handoff["runtime_inputs"]["path"]),
             site.bindings_root / Path(site.handoff["source_registration"]["path"]).name,
             *sorted(path for path in site.prior_intent_dir.rglob("*") if path.is_file())]
    return {path: path.read_bytes() for path in paths}


def test_the_renewed_preparation_is_rebound_registered_and_enqueued_once(site, monkeypatch):
    before = retained(site)
    result = run(site)
    assert result["status"] == "reissued" and len(site.registered) == 1 and len(site.enqueued) == 1
    new = site.enqueued[0]["request"]
    prior = site.request
    assert site.enqueued[0]["task_context"] == site.context
    # Only the consent window, expiry, renewal field and identity differ.
    assert new["consent"] == site.fresh["consent"] != prior["consent"]
    assert new["execution"] == {**prior["execution"], "expires_at_epoch": site.fresh["expires_at_epoch"]}
    assert renewal(new) == {"capture_id": prior["submission_id"], "parent_submission_id": prior["submission_id"]}
    assert capture_id(new) == prior["submission_id"] != new["submission_id"]

    def strip(value):
        value = copy.deepcopy(value)
        value["task"]["subject"].pop(RENEWAL_FIELD, None)
        return {key: item for key, item in value.items() if key not in {"submission_id", "consent", "execution"}}

    assert strip(new) == strip(prior)
    # The preparation keeps every retained artifact and binds the renewed grant.
    target = site.prior_path.parent / "renewals" / site.fresh["authority_digest"][7:]
    renewed = json.loads((target / "preparation.json").read_text())
    assert renewed["intake_request"] == new and renewed["website_preparation_authority"] == site.fresh
    assert renewed["digest"] == canonical_digest(renewed, digest_field="digest") == result["preparation_digest"]
    changed = {"intake_request", "website_preparation_authority", renewals.FIELD, "digest"}
    assert {k: v for k, v in renewed.items() if k not in changed} == {
        k: v for k, v in site.prior.items() if k not in changed}
    lineage = renewed[renewals.FIELD]
    assert lineage["parent_preparation_digest"] == site.prior["digest"]
    assert lineage["parent_request_digest"] == cross_runtime_canonical_digest(prior)
    assert lineage["parent_intent_id"] == site.intent["intent_id"]
    assert lineage["parent_authority_digest"] == site.old["authority_digest"]
    assert lineage["authority_digest"] == site.fresh["authority_digest"]
    assert lineage["renewal"] == site.fresh["renewal"] and lineage["provider_mutation_performed"] is False
    assert (target / "task_context.json").read_bytes() == (site.prior_path.parent / "task_context.json").read_bytes()
    # Every input bound to the preparation digest was recomputed, on the CPU, for the new one.
    runtime = json.loads((target / "native/runtime_inputs.json").read_text())
    rights = json.loads((target / "native/rights_admission.json").read_text())
    assert runtime["preparation_digest"] == renewed["digest"] == rights["preparation_digest"]
    assert rights["consent"] == site.fresh["consent"]
    assert json.loads((target / "native/capability_verdict.json").read_text())["status"] != "abstained"
    registration = json.loads(Path(result["source_registration"]["path"]).read_text())
    assert registration["request_digest"] == cross_runtime_canonical_digest(new)
    assert registration["references"]["preparation"]["path"] == str(target / "preparation.json")
    assert result["website_intake_outbox"]["id"] == "scene-outbox-renewal"
    assert json.loads((target / "reissue.json").read_text()) == result
    assert retained(site) == before
    # The intake accepts the renewed intent only by reopening the retained authority.
    from blueprint_pipeline import website_task_context as webapp
    monkeypatch.setattr(webapp, "load_current_website_task_context", lambda **_: copy.deepcopy(site.context))
    monkeypatch.setattr(webapp, "load_website_scene_sponsorship", lambda **_: copy.deepcopy(site.fresh))
    monkeypatch.setattr(authority, "current_time", lambda: RENEWED + 1)
    receipt = stage_scene_intent(value=new, queue_root=site.queue, authenticated_client="webapp",
                                 trusted_clients={"webapp"}, now=RENEWED)
    assert receipt["intent_id"] == result["intent_id"] != site.intent["intent_id"]


def test_a_rerun_is_idempotent_and_rewrites_nothing(site, monkeypatch):
    first = run(site)
    target = site.prior_path.parent / "renewals" / site.fresh["authority_digest"][7:]
    files = {path: path.read_bytes() for path in target.rglob("*") if path.is_file() and path.name != "reissue.json"}
    # The WebApp forwarded it: the renewal's own live intent is not a competing sibling.
    monkeypatch.setattr(authority, "require_retained_preparation_authority", lambda **kw: kw["now"])
    stage_scene_intent(value=site.enqueued[0]["request"], queue_root=site.queue, authenticated_client="webapp",
                       trusted_clients={"webapp"}, now=RENEWED)
    second = run(site, now=RENEWED + 5)
    same = ("status", "submission_id", "intent_id", "request_digest", "preparation", "preparation_digest",
            "source_registration", "website_intake_outbox")
    assert {key: second[key] for key in same} == {key: first[key] for key in same}
    assert second["status"] == "reissued" and len(site.registered) == 1
    assert len(site.enqueued) == 2 and site.enqueued[0] == site.enqueued[1]
    assert {path: path.read_bytes() for path in files} == files


def test_a_webapp_ceiling_refusal_is_a_recorded_outcome(site):
    def refuse(**_):
        raise ValueError("website_control_prepared-scene_http_409:website_scene_preparation_budget_exhausted")

    result = run(site, enqueue=refuse)
    assert result["status"] == "enqueue_refused" and len(site.registered) == 1
    assert result["webapp_refusal"] == {"code": "website_scene_preparation_budget_exhausted", "http_status": 409}
    target = site.prior_path.parent / "renewals" / site.fresh["authority_digest"][7:]
    assert json.loads((target / "reissue.json").read_text()) == result
    # Re-running after the owner raises the ceiling enqueues the same registered request.
    assert run(site)["status"] == "reissued" and len(site.registered) == 1


def _without_revocation(site):
    (site.prior_intent_dir / "revoked.json").unlink()


@pytest.mark.parametrize("case,code", [
    ("live_intent", "prior_intent_live"),
    ("unsettled_attempt", "prior_attempt_unsettled"),
    ("missing_intent", "prior_intent_missing"),
    ("grant_not_renewed", "grant_not_renewed"),
    ("parent_digest", "parent_mismatch"),
    ("parent_expiry", "parent_mismatch"),
    ("approver", "approval_invalid"),
    ("caps", "scope_changed"),
    ("attempts", "scope_changed"),
    ("provider", "scope_changed"),
    ("policy", "scope_changed"),
    ("context", "task_context_changed"),
    ("old_consent", "request_invalid"),
])
def test_refusals_before_any_write(site, case, code):
    grant, context, now = None, None, RENEWED
    if case == "live_intent":
        _without_revocation(site)
        now = NOW + 100
    elif case == "unsettled_attempt":
        shutil.rmtree(site.prior_intent_dir / DIRECTORY)
    elif case == "missing_intent":
        shutil.rmtree(site.prior_intent_dir)
    elif case == "grant_not_renewed":
        grant = renewed_grant(site.old, renewal=None)
    elif case == "parent_digest":
        grant = renewed_grant(site.old, renewed={**site.fresh["renewal"], "parent_authority_digest": "sha256:" + "c" * 64})
    elif case == "parent_expiry":
        grant = renewed_grant(site.old, renewed={**site.fresh["renewal"], "parent_expires_at_epoch": NOW})
    elif case == "approver":
        grant = renewed_grant(site.old, renewed={**site.fresh["renewal"], "approved_by": "someone-else"})
    elif case == "caps":
        grant = renewed_grant(site.old, max_total_spend_usd=21)
    elif case == "attempts":
        grant = renewed_grant(site.old, max_paid_attempts=3)
    elif case == "provider":
        grant = renewed_grant(site.old, authoring_provider="anthropic",
                              anthropic_provider_terms_reference="sha256:" + "6" * 64)
    elif case == "policy":
        grant = renewed_grant(site.old, policy_digest="sha256:" + "5" * 64)
    elif case == "context":
        context = _seal({**site.context, "description": "Move the mug onto the tray."}, "context_digest")
    elif case == "old_consent":
        grant = renewed_grant(site.old, accepted=RENEWED - 2 * 86400)
        now = RENEWED
    with pytest.raises(ValueError, match="^website_preparation_renewal_" + code):
        run(site, grant=grant, context=context, now=now)
    assert not (site.prior_path.parent / "renewals").exists()
    assert not site.registered and not site.enqueued


def test_a_withdrawn_capture_is_never_reissued(site, monkeypatch):
    monkeypatch.setattr("blueprint_pipeline.website_atlas_preview_store.capture_withdrawn", lambda root: True)
    with pytest.raises(ValueError, match="^website_preparation_renewal_capture_withdrawn$"):
        run(site)
    assert not site.registered and not site.enqueued


def test_a_task_family_preparation_is_not_reissued(site):
    prior = copy.deepcopy(site.prior)
    prior["task_evaluation_inputs"] = {"native_runtime_contract": {"schema_version": RUNTIME_SCHEMA}}
    _seal(prior, "digest")
    handoff_path = site.capture / "pipeline/website_scene_preparation/handoff.json"
    site.replace(site.prior_path, prior)
    site.replace(handoff_path, _seal({**site.handoff, "preparation_digest": prior["digest"]}, "digest"))
    with pytest.raises(ValueError, match="^website_preparation_renewal_task_family_unsupported$"):
        run(site)
    assert not site.registered and not site.enqueued


def test_a_renewal_of_a_renewed_grant_is_not_proven_in_v1(site):
    old = {key: value for key, value in site.old.items() if key != "authority_digest"}
    old["renewal"] = {"parent_authority_digest": "sha256:" + "0" * 64, "parent_expires_at_epoch": NOW - 3600,
                      "approved_by": OWNER["user_id"], "approval_reference": "an earlier synthetic renewal"}
    site.replace(site.capture / "pipeline/website_scene_sponsorship.json", _seal(old, "authority_digest"))
    with pytest.raises(ValueError, match="^website_preparation_renewal_lineage_unproven$"):
        run(site, grant=renewed_grant(old))
    assert not site.registered and not site.enqueued


def test_a_live_sibling_of_the_capture_blocks_the_reissue(site, monkeypatch):
    monkeypatch.setattr(authority, "require_retained_preparation_authority", lambda **kw: kw["now"])
    other = renewals.renewal_request(site.request, renewed_grant(site.old, accepted=RENEWED - 70))
    stage_scene_intent(value=other, queue_root=site.queue, authenticated_client="webapp",
                       trusted_clients={"webapp"}, now=RENEWED)
    with pytest.raises(ValueError, match="^website_preparation_renewal_live_sibling$"):
        run(site)
    assert not site.registered and not site.enqueued


def test_an_abstained_verdict_stops_before_registration(site, monkeypatch):
    def abstain(**_):
        return {"status": "abstained", "blockers": ["synthetic_no_enabled_embodiment"],
                "verdict_digest": "sha256:" + "4" * 64}

    monkeypatch.setattr("blueprint_pipeline.website_rigid_feasibility.rigid_task_verdict", abstain)
    with pytest.raises(ValueError, match="^website_preparation_renewal_capability_abstained$"):
        run(site)
    assert not site.registered and not site.enqueued
    target = site.prior_path.parent / "renewals" / site.fresh["authority_digest"][7:]
    held = json.loads((target / "reissue.json").read_text())
    assert held["status"] == "held" and held["blockers"] == ["website_preparation_renewal_capability_abstained"]
    assert held["capability_verdict"]["status"] == "abstained"


def test_the_identity_builder_reproduces_the_shared_vector():
    original, renewed, again = VECTOR["original"], VECTOR["renewed"], VECTOR["renewed_again"]

    def window(value):
        execution = value["execution"]
        return {"consent": value["consent"], **{key: execution[key] for key in (
            "max_total_spend_usd", "max_paid_attempts", "expires_at_epoch")}}

    assert renewals.renewal_request(original, window(renewed)) == renewed
    assert renewals.renewal_request(renewed, window(again)) == again


def test_a_correction_revision_is_reissued_as_a_new_revision_identity():
    from blueprint_pipeline.website_preparation_revision_identity import PREFIX, revision
    vector = json.loads((Path(__file__).parent / "fixtures/website-preparation-revision.v1.json").read_text())
    corrected = vector["corrected"]
    window = {"consent": {**corrected["consent"], "accepted_at_epoch": 199}, "max_total_spend_usd": 4,
              "max_paid_attempts": 2, "expires_at_epoch": 2000}
    reissued = renewals.renewal_request(corrected, window)
    assert reissued["submission_id"].startswith(PREFIX) and reissued["submission_id"] != corrected["submission_id"]
    assert revision(reissued) == revision(corrected) and renewal(reissued) is None
    assert capture_id(reissued) == capture_id(corrected)


def test_the_renewal_intent_reopens_retained_authority_and_shares_the_revoked_holds(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS", "webapp")
    queue = tmp_path / "queue"
    original, renewed = VECTOR["original"], VECTOR["renewed"]
    calls = []
    monkeypatch.setattr(authority, "require_retained_preparation_authority",
                        lambda **kw: calls.append(kw["request"]) or kw["now"])

    def stage(value, now):
        return stage_scene_intent(value=value, queue_root=queue, authenticated_client="webapp",
                                  trusted_clients={"webapp"}, now=now)

    def reserve(intent, name, cost, now):
        return reserve_scene_attempt(queue_root=queue, intent_id=intent["intent_id"], attempt_id=name,
            source_commit="d" * 40, runtime_digest="sha256:" + "e" * 64, input_digest="sha256:" + "f" * 64,
            provider="vast", maximum_spend_usd=cost, now=now)

    first = stage(original, 100)
    reserve(first, "a1", 3, 101)
    revoke_scene_intent(queue_root=queue, intent_id=first["intent_id"], intent_digest=first["intent_digest"],
                        owner=original["owner"], now=500)

    def refuse(**_):
        raise ValueError("website_preparation_source_missing")

    monkeypatch.setattr(authority, "require_retained_preparation_authority", refuse)
    with pytest.raises(SceneIntakeError, match="scene_intake_preparation_authority_invalid"):
        stage(renewed, 1500)
    monkeypatch.setattr(authority, "require_retained_preparation_authority",
                        lambda **kw: calls.append(kw["request"]) or kw["now"])
    second = stage(renewed, 1500)
    assert second["intent_id"] != first["intent_id"] and renewed in calls
    # The revoked sibling's hold still counts against the capture's original caps.
    with pytest.raises(SceneIntakeError, match="scene_intake_spend_cap_exhausted"):
        reserve(second, "r1", 2, 1501)
    reserve(second, "r1", 1, 1501)
    with pytest.raises(SceneIntakeError, match="scene_intake_attempt_cap_exhausted"):
        reserve(second, "r2", 0.01, 1502)
