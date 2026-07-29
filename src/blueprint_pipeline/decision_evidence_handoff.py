"""Verify the versioned Pipeline-to-WebApp Decision/Evidence handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import jsonschema


def verify(root: Path) -> dict[str, object]:
    blockers: list[str] = []
    manifest = json.loads((root / "artifact-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "decision_evidence_webapp_handoff_manifest.v1":
        blockers.append("manifest_schema_mismatch")
    expected_paths: set[str] = set()
    for row in manifest.get("artifacts") or []:
        relative = str(row.get("path") or "")
        expected_paths.add(relative)
        path = root / relative
        if not path.is_file():
            blockers.append(f"artifact_missing:{relative}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("sha256"):
            blockers.append(f"artifact_digest_mismatch:{relative}")
    actual_paths = {
        str(path.relative_to(root))
        for path in root.iterdir()
        if path.is_file() and path.name != "artifact-manifest.json"
    }
    if actual_paths != expected_paths:
        blockers.append("artifact_inventory_mismatch")
    request_schema = json.loads((root / "request.schema.json").read_text(encoding="utf-8"))
    result_schema = json.loads((root / "result.schema.json").read_text(encoding="utf-8"))
    normalized_schema = json.loads(
        (root / "normalized-evidence-result.schema.json").read_text(encoding="utf-8")
    )
    for schema in (request_schema, result_schema, normalized_schema):
        jsonschema.Draft202012Validator.check_schema(schema)
    examples = json.loads((root / "examples.json").read_text(encoding="utf-8"))
    for example_id, example in sorted((examples.get("examples") or {}).items()):
        try:
            jsonschema.Draft202012Validator(result_schema).validate(example)
        except jsonschema.ValidationError:
            blockers.append(f"decision_example_invalid:{example_id}")
    return {
        "schema_version": "decision_evidence_webapp_handoff_verification.v1",
        "status": "passed" if not blockers else "failed",
        "handoff_version": manifest.get("handoff_version"),
        "artifact_count": len(expected_paths),
        "example_count": len(examples.get("examples") or {}),
        "blockers": sorted(blockers),
        "provider_selection_recomputed_by_webapp": False,
        "scientific_verdict_recomputed_by_webapp": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            Path(__file__).parents[2]
            / "docs/webapp_handoff/decision-evidence-router.v1"
        ),
    )
    args = parser.parse_args(argv)
    result = verify(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
