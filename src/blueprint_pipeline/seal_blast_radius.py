"""Answer "what breaks if I amend this seal?" before anyone amends it.

On 2026-08-17 Scene 840920's washer door was found to open into its own
cabinet: ``door_hinge`` was authored ``[0, 0, +1]`` against limits ``[0, 1.2]``,
so the commanded direction drove the door's far edge toward the cabinet face.
Two paid runs read back 6.01 degrees for every commanded angle.

The axis lives inside a prospectively-sealed task freeze, so fixing it meant
amending that freeze.  The cost of that amendment was discovered the expensive
way -- one refusal at a time, over hours:

    spec rebound            -> receipt stale
    receipt regenerated     -> static qualification stale
    ... three more bindings -> CI red on two unrelated suites
    staged host freeze      -> diverged from the repo copy
    staged freeze synced    -> CAD receipts refuse: sealed to the old digest

The last one is the point.  ``cad_agent_request_task_freeze_invalid`` cannot be
resolved by re-deriving anything; the CAD receipts are sealed against the old
freeze and re-deriving them means paying for CAD authoring again *and*
re-opening a two-backend comparison.  That is a decision about evidence, not a
repair -- and it was reachable only after five other repairs had already landed.

This module computes that whole picture in one pass, from digests, before the
first edit.  It reads only JSON and answers three questions:

  * which artifacts bind this seal at all,
  * which of them are forward-looking (must be rebound) versus history (must
    keep the superseded digest, because they record what happened under it),
  * which are themselves sealed, and so make the amendment a decision rather
    than a repair.

It changes nothing.  It is a report, and the honest output is frequently
"do not amend this".
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "seal_blast_radius_report.v1"

#: Artifacts whose whole purpose is to record a past run.  They cite the digest
#: that was current when they were sealed, and rebinding them would rewrite
#: history rather than repair anything.
HISTORICAL_NAME_MARKERS = (
    "episode_evidence_index",
    "typed_abstention",
    "_RESULTS",
    "BLOCKED",
    "supporting_evidence_inventory",
    "evidence_package_index",
    "superseded",
)

#: Schema fragments that mark an artifact as itself sealed against the thing
#: being amended.  These cannot be re-derived by rebinding a digest: producing a
#: new one means re-running whatever authored it, which usually costs money and
#: may re-open a frozen comparison.
SEALED_SCHEMA_MARKERS = (
    "cad_agent_request",
    "cad_agent_output",
    "cad_agent_execution",
    "cad_agent_reference_manifest",
    "task_freeze",
    "scene_freeze",
    "paid_attempt_authority",
    "standing_launch_authorization",
)


class SealBlastRadiusError(ValueError):
    """Fail-closed refusal to report on an unreadable seal."""


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cites(value: Any, digest: str) -> int:
    """How many times this document cites the digest, at any depth."""

    if isinstance(value, str):
        return 1 if value == digest else 0
    if isinstance(value, Mapping):
        return sum(_cites(item, digest) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_cites(item, digest) for item in value)
    return 0


def _is_historical(path: Path) -> bool:
    return any(marker in path.name for marker in HISTORICAL_NAME_MARKERS)


def _sealed_schema(document: Any) -> str | None:
    if not isinstance(document, Mapping):
        return None
    schema = str(document.get("schema_version") or "")
    for marker in SEALED_SCHEMA_MARKERS:
        if marker in schema:
            return schema
    return None


def _self_digest_field(document: Any) -> str | None:
    """The field this artifact seals *itself* with, if any.

    An artifact carrying its own digest must be resealed after rebinding, which
    is what makes a cascade a cascade rather than a single edit.
    """

    if not isinstance(document, Mapping):
        return None
    for key in document:
        if key.endswith("_digest") and isinstance(document.get(key), str):
            value = str(document[key])
            if value.startswith("sha256:") and len(value) == 71:
                return key
    return None


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def compute_seal_blast_radius(
    *,
    digest: str,
    roots: Iterable[str | Path],
    exclude: Iterable[str] = (),
    seal_file: str | Path | None = None,
) -> dict[str, Any]:
    """Report every artifact bound to ``digest`` and what amending it costs.

    ``seal_file`` matters more than it looks.  An artifact can bind a seal two
    ways: by citing its ``*_digest``, or by carrying a file record holding the
    sealing file's raw ``sha256``.  Scene 840920's CAD receipts use the second
    form, so a digest-only trace reports them as unaffected -- and they are the
    single binding that makes the amendment a paid decision.  Pass the sealing
    file and both forms are traced.
    """

    if not digest.startswith("sha256:") or len(digest) != 71:
        raise SealBlastRadiusError("seal_blast_radius_digest_invalid")

    tracked = [digest]
    seal_file_sha256: str | None = None
    if seal_file is not None:
        source = Path(seal_file).expanduser().resolve()
        if not source.is_file():
            raise SealBlastRadiusError("seal_blast_radius_seal_file_missing")
        seal_file_sha256 = _file_sha256(source)
        tracked.append(seal_file_sha256)

    excluded = tuple(exclude)
    forward: list[dict[str, Any]] = []
    historical: list[dict[str, Any]] = []
    sealed: list[dict[str, Any]] = []

    for root in roots:
        base = Path(root).expanduser().resolve()
        if not base.exists():
            continue
        candidates = base.rglob("*.json") if base.is_dir() else [base]
        for path in sorted(candidates):
            if any(fragment in str(path) for fragment in excluded):
                continue
            document = _load(path)
            citations = sum(_cites(document, value) for value in tracked)
            if not citations:
                continue
            entry: dict[str, Any] = {
                "path": str(path),
                "citations": citations,
                "self_digest_field": _self_digest_field(document),
            }
            sealed_schema = _sealed_schema(document)
            if _is_historical(path):
                historical.append(entry)
            elif sealed_schema is not None:
                entry["sealed_schema"] = sealed_schema
                sealed.append(entry)
            else:
                forward.append(entry)

    amendable = not sealed
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "digest": digest,
        "seal_file_sha256": seal_file_sha256,
        "traced_digests": tracked,
        "status": "amendable" if amendable else "amendment_is_a_decision",
        "forward_binding_count": len(forward),
        "historical_binding_count": len(historical),
        "sealed_binding_count": len(sealed),
        "forward_bindings": forward,
        "historical_bindings": historical,
        "sealed_bindings": sealed,
        "provider_mutation_performed": False,
        "spend_incurred_usd": 0.0,
        "guidance": (
            "Rebind and reseal every forward binding, in dependency order. "
            "Leave historical bindings citing the superseded digest -- they "
            "record what happened under it."
            if amendable
            else (
                "Do not amend yet. The sealed bindings below cannot be repaired "
                "by rebinding: producing new ones means re-running whatever "
                "authored them, which usually costs money and can re-open a "
                "frozen comparison. That is a decision about evidence, not a "
                "repair, and it belongs to an operator."
            )
        ),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest", required=True, help="The seal digest to trace.")
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Directory or file to search. Repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path fragment to skip. Repeatable.",
    )
    parser.add_argument(
        "--seal-file",
        help=(
            "The sealing file itself. Also traces artifacts that bind it by raw "
            "file sha256 rather than by its *_digest -- which is how sealed "
            "CAD receipts bind a task freeze."
        ),
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    report = compute_seal_blast_radius(
        digest=args.digest,
        roots=args.root,
        exclude=args.exclude,
        seal_file=args.seal_file,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    # A sealed binding is not an error, but it must not read as a green light.
    return 0 if report["status"] == "amendable" else 3


__all__ = [
    "HISTORICAL_NAME_MARKERS",
    "SCHEMA_VERSION",
    "SEALED_SCHEMA_MARKERS",
    "SealBlastRadiusError",
    "compute_seal_blast_radius",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
