"""Decide whether a sealed receipt survives a freeze amendment, by evidence.

Scene 840920's washer door was authored hinging the wrong way: positive
rotation drove its far edge into the cabinet, so two paid runs read back
6.01 degrees for every commanded angle.  The axis lives in a prospectively
sealed task freeze, so fixing it meant amending that freeze -- and every CAD
agent receipt then refused with ``cad_agent_request_task_freeze_invalid``.

Those receipts bind the freeze by the sha256 of the **whole file**.  That is
the strictest possible binding, and it is wrong in a specific and expensive
way: the CAD agent reads exactly two fields out of that file -- ``task_id`` and
``removal_plan.replacement_asset_id``.  The amendment changed neither.  It
changed a joint axis, which the CAD agent never sees, because its job is shape
and the axis is articulation.  Under a whole-file hash, an edit that provably
cannot change what the agent was asked still forces paying to ask it again.

So this module replaces "the bytes differ" with the question actually worth
asking: *did the amendment touch anything this receipt consumed?*  A receipt's
consumed fields are declared per schema, here, in the open -- not inferred, and
not supplied by the receipt itself, which could otherwise understate what it
read.  If the changed paths and the consumed paths are disjoint, the receipt
carries forward, and the carry-forward receipt records both digests and the
exact list of changed paths so the reasoning is auditable rather than asserted.

Everything else fails closed.  An unknown schema carries nothing.  A schema
that declares no consumed fields carries nothing, because "reads nothing" is
far more likely to mean "nobody wrote it down" than to be true.  Any overlap at
all, however small, means re-derivation -- which is a decision about evidence
and belongs to an operator, not to this module.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


SCHEMA_VERSION = "freeze_amendment_carry_forward.v1"

CARRIES_FORWARD = "carries_forward"
REQUIRES_REDERIVATION = "requires_rederivation"

#: Fields each sealed schema actually reads out of a task freeze.
#:
#: Declared here rather than in the receipts, so a receipt cannot understate
#: what it consumed in order to survive an amendment.  Adding a schema is a
#: deliberate act: it means someone has read the consuming validator and
#: written down every freeze field it touches.
CONSUMED_FREEZE_FIELDS: Mapping[str, tuple[str, ...]] = {
    # simready_cad_agent_contract.validate_cad_agent_request reads exactly
    # these two, and compares them to the request's own task_id / asset_id.
    "simready_cad_agent_request.v1": (
        "task_id",
        "removal_plan.replacement_asset_id",
    ),
    # Outputs and reference manifests are joined to their request by digest and
    # never re-read the freeze themselves; they inherit the request's verdict.
    "simready_cad_agent_output.v1": (
        "task_id",
        "removal_plan.replacement_asset_id",
    ),
    "simready_cad_agent_reference_manifest.v1": (
        "task_id",
        "removal_plan.replacement_asset_id",
    ),
    # The visual binding joins CAD-side evidence to the graph asset. From the
    # freeze itself it consumes only task/asset identity -- link identity comes
    # from the graph receipt and geometry from the CAD output, so a joint-axis
    # amendment is invisible to it.
    "simready_agent_cad_visual_binding.v2": (
        "task_id",
        "removal_plan.replacement_asset_id",
    ),
}

#: Bookkeeping the amendment itself writes.  Counting these as changes would
#: make every amendment collide with every receipt, including its own record of
#: why it was safe.
AMENDMENT_BOOKKEEPING_FIELDS = ("task_freeze_digest", "freeze_amendments")


class FreezeAmendmentCarryForwardError(ValueError):
    """Fail-closed refusal to rule on an unreadable amendment."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _changed_paths(
    superseded: Any, amended: Any, prefix: str = ""
) -> list[str]:
    """Every dotted path whose value differs between the two documents.

    A changed container reports the leaf that moved, not the container, so a
    consumed-field check compares like with like.
    """

    if isinstance(superseded, Mapping) and isinstance(amended, Mapping):
        changed: list[str] = []
        for key in sorted(set(superseded) | set(amended)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in superseded or key not in amended:
                changed.append(path)
                continue
            changed.extend(_changed_paths(superseded[key], amended[key], path))
        return changed
    if (
        isinstance(superseded, list)
        and isinstance(amended, list)
        and not isinstance(superseded, (str, bytes))
    ):
        if len(superseded) != len(amended):
            return [prefix or "."]
        changed = []
        for index, (before, after) in enumerate(zip(superseded, amended)):
            changed.extend(_changed_paths(before, after, f"{prefix}[{index}]"))
        return changed
    return [] if superseded == amended else [prefix or "."]


def _consumes(changed_path: str, consumed_field: str) -> bool:
    """Does a changed path fall inside a consumed field?

    ``removal_plan.replacement_asset_id`` is consumed, so a change at exactly
    that path collides.  So does a change *beneath* it, if it were a container.
    A change to ``removal_plan.source_collider_prim_path`` does not, and neither
    does a change to ``removal_plan`` as a whole reported at a deeper leaf --
    which is why leaves, not containers, are compared.
    """

    normalized = changed_path.split("[", 1)[0]
    return normalized == consumed_field or normalized.startswith(
        consumed_field + "."
    )


def evaluate_freeze_amendment_carry_forward(
    *,
    superseded_freeze: Mapping[str, Any],
    amended_freeze: Mapping[str, Any],
    sealed_schema: str,
    superseded_file_sha256: str = "",
    amended_file_sha256: str = "",
) -> dict[str, Any]:
    """Rule on one sealed schema against one amendment.

    Both identities are recorded because both are load-bearing and they are not
    interchangeable.  The content digest says which freeze this is; the file
    sha256 says which bytes a sealed receipt actually pinned.  Receipts pin
    bytes, so a proof that named only the content digest could not be matched
    to the record it is meant to rescue.
    """

    if not isinstance(superseded_freeze, Mapping) or not isinstance(
        amended_freeze, Mapping
    ):
        raise FreezeAmendmentCarryForwardError(["freeze_amendment_input_invalid"])

    superseded_digest = superseded_freeze.get("task_freeze_digest")
    amended_digest = amended_freeze.get("task_freeze_digest")
    if not isinstance(superseded_digest, str) or not isinstance(amended_digest, str):
        raise FreezeAmendmentCarryForwardError(["freeze_amendment_digest_missing"])
    if superseded_digest == amended_digest:
        # Same freeze: there is nothing to rule on, and saying "carries forward"
        # would manufacture a proof that no amendment happened.
        raise FreezeAmendmentCarryForwardError(["freeze_amendment_absent"])

    changed = [
        path
        for path in _changed_paths(superseded_freeze, amended_freeze)
        if path.split("[", 1)[0].split(".", 1)[0] not in AMENDMENT_BOOKKEEPING_FIELDS
    ]
    consumed = CONSUMED_FREEZE_FIELDS.get(sealed_schema)
    collisions = (
        sorted(
            {
                path
                for path in changed
                for field in consumed
                if _consumes(path, field)
            }
        )
        if consumed
        else []
    )

    if consumed is None:
        status = REQUIRES_REDERIVATION
        reason = "sealed_schema_declares_no_consumed_freeze_fields"
    elif not consumed:
        status = REQUIRES_REDERIVATION
        reason = "sealed_schema_consumed_field_list_empty"
    elif collisions:
        status = REQUIRES_REDERIVATION
        reason = "amendment_changed_a_consumed_freeze_field"
    elif not changed:
        status = REQUIRES_REDERIVATION
        reason = "amendment_changed_no_freeze_field_outside_bookkeeping"
    else:
        status = CARRIES_FORWARD
        reason = "amendment_changed_no_field_this_schema_consumes"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "sealed_schema": sealed_schema,
        "superseded_task_freeze_digest": superseded_digest,
        "amended_task_freeze_digest": amended_digest,
        "superseded_freeze_file_sha256": superseded_file_sha256,
        "amended_freeze_file_sha256": amended_file_sha256,
        "changed_freeze_paths": sorted(changed),
        "consumed_freeze_fields": list(consumed or ()),
        "colliding_freeze_paths": collisions,
        "task_semantics_changed": bool(collisions),
        "provider_mutation_performed": False,
        "spend_incurred_usd": 0.0,
        "carry_forward_digest": "",
    }
    payload["carry_forward_digest"] = canonical_digest(
        payload, digest_field="carry_forward_digest"
    )
    return payload


def validate_freeze_amendment_carry_forward(
    value: Mapping[str, Any],
    *,
    sealed_schema: str,
    superseded_file_sha256: str,
    amended_file_sha256: str,
) -> dict[str, Any]:
    """Accept a carry-forward proof only for the exact amendment it names.

    A proof is not a token.  It rules on one schema moving between two named
    freeze digests, and presenting it for any other schema or any other pair of
    digests is refused -- otherwise one cheap ruling would launder every sealed
    receipt in the tree.
    """

    if not isinstance(value, Mapping):
        raise FreezeAmendmentCarryForwardError(["freeze_carry_forward_invalid"])
    proof = json.loads(json.dumps(value))
    errors: list[str] = []
    if proof.get("schema_version") != SCHEMA_VERSION:
        errors.append("freeze_carry_forward_schema_invalid")
    if proof.get("carry_forward_digest") != canonical_digest(
        proof, digest_field="carry_forward_digest"
    ):
        errors.append("freeze_carry_forward_digest_invalid")
    if proof.get("status") != CARRIES_FORWARD:
        errors.append("freeze_carry_forward_status_invalid")
    if proof.get("sealed_schema") != sealed_schema:
        errors.append("freeze_carry_forward_schema_mismatch")
    if (
        not proof.get("superseded_freeze_file_sha256")
        or proof.get("superseded_freeze_file_sha256") != superseded_file_sha256
    ):
        errors.append("freeze_carry_forward_superseded_mismatch")
    if (
        not proof.get("amended_freeze_file_sha256")
        or proof.get("amended_freeze_file_sha256") != amended_file_sha256
    ):
        errors.append("freeze_carry_forward_amended_mismatch")
    if errors:
        raise FreezeAmendmentCarryForwardError(errors)
    return proof


def validate_freeze_amendment_carry_forward_content(
    value: Mapping[str, Any],
    *,
    sealed_schema: str,
    superseded_digest: str,
    amended_digest: str,
) -> dict[str, Any]:
    """Accept a proof pinned by freeze *content* digests rather than file bytes.

    CAD receipts pin the freeze by file sha256, so their acceptance validates
    against file hashes. The visual-composition join compares the freeze's own
    ``task_freeze_digest`` across documents, so its acceptance must pin the same
    two content digests the join actually observes. Same proof either way --
    it records both identities -- but each acceptance checks the pair it uses,
    so a proof for one amendment can never speak for another.
    """

    if not isinstance(value, Mapping):
        raise FreezeAmendmentCarryForwardError(["freeze_carry_forward_invalid"])
    proof = json.loads(json.dumps(value))
    errors: list[str] = []
    if proof.get("schema_version") != SCHEMA_VERSION:
        errors.append("freeze_carry_forward_schema_invalid")
    if proof.get("carry_forward_digest") != canonical_digest(
        proof, digest_field="carry_forward_digest"
    ):
        errors.append("freeze_carry_forward_digest_invalid")
    if proof.get("status") != CARRIES_FORWARD:
        errors.append("freeze_carry_forward_status_invalid")
    if proof.get("sealed_schema") != sealed_schema:
        errors.append("freeze_carry_forward_schema_mismatch")
    if (
        not superseded_digest
        or proof.get("superseded_task_freeze_digest") != superseded_digest
    ):
        errors.append("freeze_carry_forward_superseded_mismatch")
    if not amended_digest or proof.get("amended_task_freeze_digest") != amended_digest:
        errors.append("freeze_carry_forward_amended_mismatch")
    if errors:
        raise FreezeAmendmentCarryForwardError(errors)
    return proof


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--superseded-freeze", required=True)
    parser.add_argument("--amended-freeze", required=True)
    parser.add_argument(
        "--sealed-schema",
        required=True,
        help="Schema of the sealed receipts being ruled on. Repeatable via --sealed-schema.",
        action="append",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)

    try:
        superseded_path = Path(args.superseded_freeze).expanduser()
        amended_path = Path(args.amended_freeze).expanduser()
        superseded = json.loads(superseded_path.read_text(encoding="utf-8"))
        amended = json.loads(amended_path.read_text(encoding="utf-8"))
        superseded_file_sha256 = _file_sha256(superseded_path)
        amended_file_sha256 = _file_sha256(amended_path)
    except (OSError, json.JSONDecodeError):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["freeze_amendment_input_invalid"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2

    results = []
    for schema in args.sealed_schema:
        try:
            report = evaluate_freeze_amendment_carry_forward(
                superseded_freeze=superseded,
                amended_freeze=amended,
                sealed_schema=schema,
                superseded_file_sha256=superseded_file_sha256,
                amended_file_sha256=amended_file_sha256,
            )
        except FreezeAmendmentCarryForwardError as exc:
            print(
                json.dumps(
                    {
                        "status": "blocked",
                        "blockers": list(exc.errors),
                        "provider_mutation_performed": False,
                    },
                    sort_keys=True,
                )
            )
            return 2
        results.append(report)
        if args.output_dir:
            destination = Path(args.output_dir).expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"carry_forward_{schema.replace('.', '_')}.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    print(json.dumps(results, indent=2, sort_keys=True))
    # Any receipt needing re-derivation must not read as a green light.
    return 0 if all(r["status"] == CARRIES_FORWARD for r in results) else 3


__all__ = [
    "AMENDMENT_BOOKKEEPING_FIELDS",
    "CARRIES_FORWARD",
    "CONSUMED_FREEZE_FIELDS",
    "REQUIRES_REDERIVATION",
    "SCHEMA_VERSION",
    "FreezeAmendmentCarryForwardError",
    "evaluate_freeze_amendment_carry_forward",
    "main",
    "validate_freeze_amendment_carry_forward",
    "validate_freeze_amendment_carry_forward_content",
]


if __name__ == "__main__":
    raise SystemExit(main())
