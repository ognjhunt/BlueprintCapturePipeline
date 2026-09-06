"""Decide, fail closed, where one scene's appearance render may execute.

Rendering the object-present views is the single expensive step of a scene
configuration.  It can run in two places, and the choice is a *rights* decision
rather than a performance one:

``control_plane``
    The historical default.  Publisher bytes never leave the control plane; the
    provider receives only derived PNGs.  This is mandatory for a dataset whose
    admission does not permit uploading source bytes, and it is why the render
    was originally local.

``provider_gpu``
    The source appearance bytes travel to the already-rented configuration GPU,
    which renders the same exact cameras.  This is admissible only when the
    scene's own rights admission says so, and it costs no extra provider
    allocation because stage one already runs on that host.

The decision is never inferred from convenience, a caller argument, or the
absence of a field.  Both the scene's rights admission *and* the stage
configuration must independently authorize the upload, and a human authority
record must accept the provider's retention terms.  Anything missing, malformed,
or merely silent resolves to ``control_plane``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "task_evaluation_scene_configuration_disclosure_decision.v1"
CONTROL_PLANE = "control_plane"
PROVIDER_GPU = "provider_gpu"
RENDER_EXECUTION_SITES = frozenset({CONTROL_PLANE, PROVIDER_GPU})

# The admission packet for a gated third-party dataset names the publisher; a
# Blueprint-captured scene has no publisher and uses the neutral key.  Both are
# read so one seam serves owned captures and licensed datasets alike.
_ADMISSION_UPLOAD_KEYS = (
    "source_appearance_downloaded_bytes_may_be_uploaded",
    "raw_interiorgs_downloaded_bytes_may_be_uploaded",
)
_STAGE_UPLOAD_KEYS = (
    "source_appearance_bytes",
    "raw_interiorgs_bytes",
)


class TaskEvaluationSceneConfigurationDisclosureError(ValueError):
    """The requested disclosure scope could not be established safely."""


def _first_present(value: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _admission_permits_upload(rights_admission: Mapping[str, Any]) -> bool:
    disclosure = rights_admission.get("provider_disclosure")
    if not isinstance(disclosure, Mapping):
        return False
    if _first_present(disclosure, _ADMISSION_UPLOAD_KEYS) is not True:
        return False
    # An upload permission is only meaningful alongside the retention and
    # training boundary the run actually relies on.
    return (
        disclosure.get("provider_training_allowed") is not True
        and disclosure.get("public_redistribution_allowed") is not True
        and bool(str(disclosure.get("provider_retention_rule") or "").strip())
    )


def stage_requests_upload(stage_one_configuration: Mapping[str, Any]) -> bool:
    disclosure = stage_one_configuration.get("provider_disclosure")
    if not isinstance(disclosure, Mapping):
        return False
    return (
        _first_present(disclosure, _STAGE_UPLOAD_KEYS) is True
        and disclosure.get("provider_training") is not True
        and disclosure.get("public_redistribution") is not True
    )


def _human_authority_accepts_provider(
    stage_one_configuration: Mapping[str, Any],
) -> bool:
    authority = stage_one_configuration.get("human_authority")
    if not isinstance(authority, Mapping):
        return False
    return (
        authority.get("provider_retention_terms_accepted") is True
        and authority.get("provider_training_authorized") is not True
        and bool(str(authority.get("authority_reference") or "").strip())
    )


def resolve_scene_configuration_disclosure(
    *,
    stage_one_configuration: Mapping[str, Any],
    rights_admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve where the appearance render runs and what the provider receives.

    Returns a digest-bound decision.  ``control_plane`` is not an error state --
    it is the safe answer whenever the upload is not unambiguously admitted by
    both the rights packet and the stage configuration.
    """

    if not isinstance(stage_one_configuration, Mapping) or not isinstance(
        rights_admission, Mapping
    ):
        raise TaskEvaluationSceneConfigurationDisclosureError(
            "scene_configuration_disclosure_inputs_invalid"
        )
    admitted = _admission_permits_upload(rights_admission)
    requested = stage_requests_upload(stage_one_configuration)
    accepted = _human_authority_accepts_provider(stage_one_configuration)
    source_bytes_to_provider = bool(admitted and requested and accepted)
    refusals: list[str] = []
    if requested and not admitted:
        refusals.append("scene_configuration_source_upload_not_rights_admitted")
    if requested and not accepted:
        refusals.append("scene_configuration_source_upload_human_authority_missing")
    decision: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "render_execution_site": PROVIDER_GPU if source_bytes_to_provider else CONTROL_PLANE,
        "source_appearance_bytes_to_provider": source_bytes_to_provider,
        "rights_admission_permits_upload": admitted,
        "stage_configuration_requests_upload": requested,
        "human_authority_accepts_provider_terms": accepted,
        # A refusal is recorded rather than raised: the run continues on the
        # control plane, which is always a permitted way to render.
        "refusals": sorted(refusals),
        "provider_training_authorized": False,
        "public_redistribution_authorized": False,
        "decision_digest": "",
    }
    decision["decision_digest"] = canonical_digest(
        decision, digest_field="decision_digest"
    )
    return decision


def renders_on_provider(decision: Mapping[str, Any]) -> bool:
    """True only for a well-formed decision that admits the provider render."""

    if not isinstance(decision, Mapping):
        return False
    if decision.get("schema_version") != SCHEMA_VERSION:
        return False
    if decision.get("decision_digest") != canonical_digest(
        decision, digest_field="decision_digest"
    ):
        return False
    return (
        decision.get("render_execution_site") == PROVIDER_GPU
        and decision.get("source_appearance_bytes_to_provider") is True
    )


#: Retained so existing private imports keep working.
_stage_requests_upload = stage_requests_upload


MATERIALIZED_STATUS = "derived_method_inputs_materialized"
PENDING_PROVIDER_RENDER_STATUS = "derived_method_inputs_pending_provider_render"
MESH_INPUT_STATUS = "explicit_visual_geometry_prepared"
RENDER_INPUT_STATUSES = frozenset({MATERIALIZED_STATUS, PENDING_PROVIDER_RENDER_STATUS, MESH_INPUT_STATUS})


def render_inputs_disclosure_is_coherent(render_inputs: Mapping[str, Any]) -> bool:
    """True when a render-inputs result's status and disclosure agree.

    This is the single place the old "no source bytes ever reach the provider"
    assertion now lives.  It is not weakened -- it is made conditional on a
    digest-bound decision, so a result claiming provider disclosure must carry
    the decision that authorized it, and a result claiming the control plane
    must still show no source bytes crossing.
    """

    if not isinstance(render_inputs, Mapping):
        return False
    status = render_inputs.get("status")
    crossed = render_inputs.get("raw_interiorgs_bytes_in_provider_packet")
    if status == MESH_INPUT_STATUS:
        geometry = render_inputs.get("derived_visual_geometry")
        return (crossed is False and render_inputs.get("input_kind") == "provided_mesh"
                and render_inputs.get("derived_frames") == []
                and render_inputs.get("derived_frame_count") == 0
                and render_inputs.get("renderer_qualified") is False
                and render_inputs.get("physical_truth_claimed") is False
                and render_inputs.get("provider_render_required") is False
                and isinstance(geometry, Mapping)
                and isinstance(geometry.get("size_bytes"), int) and geometry["size_bytes"] > 0
                and str(geometry.get("digest", "")).startswith("sha256:"))
    if status == MATERIALIZED_STATUS:
        return crossed is False
    if status != PENDING_PROVIDER_RENDER_STATUS:
        return False
    return crossed is True and renders_on_provider(
        render_inputs.get("disclosure_decision") or {}
    )


__all__ = [
    "CONTROL_PLANE",
    "stage_requests_upload",
    "MATERIALIZED_STATUS",
    "PENDING_PROVIDER_RENDER_STATUS",
    "RENDER_INPUT_STATUSES",
    "PROVIDER_GPU",
    "RENDER_EXECUTION_SITES",
    "SCHEMA_VERSION",
    "TaskEvaluationSceneConfigurationDisclosureError",
    "render_inputs_disclosure_is_coherent",
    "renders_on_provider",
    "resolve_scene_configuration_disclosure",
]
