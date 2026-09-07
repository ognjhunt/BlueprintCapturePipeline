"""Observe account-wide Vast inventory before reuse; never allocate or grant spend."""
from __future__ import annotations

from pathlib import Path
import time

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_submission_inputs import read, require
from .task_evaluation_sam31_prefix_adoption import (
    PREFIX_ZERO_DIGEST_FIELD, PREFIX_ZERO_SCHEMA, _zero, record,
)
from .task_evaluation_scene_intake import write_exclusive


def observe_prefix_zero(output_root):
    """Retain the canonical sanitized GET result, independent of static release identity.

    The provider adapter owns credentials and account enumeration. An empty prefix
    observes the whole account, so neither unrelated live resources nor uncertain
    API responses can be mistaken for zero. This is evidence, never launch authority.
    """
    from .gpu_render_providers import get_render_provider
    value = get_render_provider("vast").billable_inventory(name_prefix="")
    at = time.time()
    require(isinstance(value, dict), "public_factory_prefix_inventory_invalid")
    # The canonical adapter already sanitizes resources. Do not persist arbitrary
    # response/error strings or credentials supplied by a failing backend.
    value = {key: value[key] for key in (
        "provider", "status", "name_prefix", "live_resource_count", "resources",
        "api_confirmed", "observed_at_epoch", "http", "raw_provider_response_recorded", "blockers"
    ) if key in value}
    require(value.get("raw_provider_response_recorded") is False,
            "public_factory_prefix_inventory_unsanitized")
    root = Path(output_root)
    require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)),
            "public_factory_prefix_inventory_output_invalid")
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    value = {"schema_version": PREFIX_ZERO_SCHEMA, **value,
             PREFIX_ZERO_DIGEST_FIELD: ""}
    value[PREFIX_ZERO_DIGEST_FIELD] = canonical_digest(
        value, digest_field=PREFIX_ZERO_DIGEST_FIELD)
    path = root / (canonical_digest(value)[7:] + ".json")
    if not path.exists():
        write_exclusive(path, value)
    require(read(path) == value, "public_factory_prefix_inventory_changed")
    _zero(path, at=at)
    return path, at


def selection_observation(output):
    """Freshly check current zero, preserving a committed selection's original witness.

    Interrupted selections may retry with a fresh observation. Once adoption or
    selection is committed, its original timestamp/reference stays immutable;
    the additional current observation is retained without rewriting that proof.
    """
    output = Path(output)
    fresh_path, fresh_at = observe_prefix_zero(output / "prefix-provider-observations")
    selection = output / "prefix_selection.json"
    adoption = output / "completed_prefix_adoption.json"
    if selection.exists():
        value = read(selection, digest_field="selection_digest")
        # A no-reuse selection is only a snapshot of the candidates seen by
        # that attempt.  It must not pin the next retry to its old witness;
        # newly completed work may be eligible and the current zero is the
        # authority for a new adoption.
        if value.get("status") != "reusable_prefix_selected":
            return fresh_path, fresh_at
        witness = value.get("provider_zero_observation")
        at = value.get("provider_zero_checked_at_epoch")
    elif adoption.exists():
        value = read(adoption, digest_field="adoption_digest")
        witness, at = value["provider_zero_at_adoption"], value["created_at_epoch"]
    else:
        return fresh_path, fresh_at
    require(isinstance(witness, dict) and record(witness["path"]) == witness,
            "public_factory_prefix_observation_changed")
    _zero(witness["path"], at=at)
    # The historical witness is checked for immutable-record integrity, but a
    # new adoption must be bound to the fresh account observation above.  The
    # caller may still use the historical record when replaying an already
    # published adoption; it must never be substituted for this current gate.
    return fresh_path, fresh_at
