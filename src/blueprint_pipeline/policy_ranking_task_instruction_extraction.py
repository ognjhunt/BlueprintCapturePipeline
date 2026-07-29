"""Extract one allowlisted task instruction without deserializing outcome metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .common import write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "policy_ranking_task_instruction_receipt.v1"
FIELD_PREFIX = "language_instruction:"
FORBIDDEN_PRECEDING_KEYS = frozenset(
    {"outcome", "outcomes", "success", "success_rate", "score", "rank", "ranking"}
)
MAX_SCANNED_LINES = 100
MAX_INSTRUCTION_CHARACTERS = 1000


def _parse_scalar(raw: str) -> str:
    value = raw.strip()
    if not value or value in {"|", ">"}:
        raise ValueError("task_instruction_scalar_missing_or_multiline")
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("task_instruction_double_quoted_scalar_invalid") from exc
        if not isinstance(parsed, str):
            raise ValueError("task_instruction_scalar_not_string")
        value = parsed
    elif value.startswith("'"):
        if len(value) < 2 or not value.endswith("'"):
            raise ValueError("task_instruction_single_quoted_scalar_invalid")
        value = value[1:-1].replace("''", "'")
    if not value.strip() or len(value) > MAX_INSTRUCTION_CHARACTERS:
        raise ValueError("task_instruction_length_invalid")
    return value.strip()


def extract_task_instruction(metadata_path: str | Path, *, session_id: str) -> dict[str, Any]:
    """Return only the first top-level language instruction and its provenance.

    The file is streamed only until the allowlisted field is found.  Other YAML
    values are never deserialized.  The whole-file SHA-256 is computed as an
    opaque byte digest so the receipt remains bound to the frozen source.
    """

    path = Path(metadata_path).resolve()
    instruction: str | None = None
    instruction_line = 0
    scanned_bytes = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if line_number > MAX_SCANNED_LINES:
                break
            scanned_bytes += len(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("metadata_utf8_invalid_before_instruction") from exc
            if line[:1].isspace() or not line.strip() or line.lstrip().startswith("#"):
                continue
            key = line.split(":", 1)[0].strip().lower()
            if key in FORBIDDEN_PRECEDING_KEYS:
                raise ValueError(f"outcome_like_field_precedes_instruction:{key}")
            if line.startswith(FIELD_PREFIX):
                instruction = _parse_scalar(line[len(FIELD_PREFIX) :])
                instruction_line = line_number
                break
    if instruction is None:
        raise ValueError("task_instruction_not_found_before_scan_limit")
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "session_id_internal_only": str(session_id),
        "task_instruction": instruction,
        "task_instruction_sha256": canonical_sha256(instruction),
        "metadata_file_sha256": file_sha256(path),
        "metadata_file_bytes": path.stat().st_size,
        "instruction_line_number": instruction_line,
        "bytes_streamed_before_instruction_found": scanned_bytes,
        "access_contract": {
            "allowlisted_field": "language_instruction",
            "yaml_document_deserialized": False,
            "outcome_fields_parsed": False,
            "outcome_values_returned": False,
            "opaque_whole_file_hash_computed": True,
        },
        "claim_boundary": "task_prompt_extraction_only; no outcome or ranking evidence",
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = extract_task_instruction(args.metadata, session_id=args.session_id)
    write_json(Path(args.output), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "extract_task_instruction"]
