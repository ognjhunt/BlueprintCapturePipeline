"""Seal a local compressed-to-standard 3DGS conversion without ownership claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .gaussian_splat_decode import (
    convert_to_standard_ply,
    find_splat_transform_cli,
    read_compressed_ply_chunk_bounds,
    read_standard_3dgs_ply,
)


REQUEST_SCHEMA = "standard_splat_conversion_request.v1"
RECEIPT_SCHEMA = "standard_splat_conversion_receipt.v1"


class StandardSplatConversionError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(root: Path, value: str | Path, *, code: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.expanduser().resolve()
    if candidate != root and root not in candidate.parents:
        raise StandardSplatConversionError([code])
    return candidate


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _repository_identity(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise StandardSplatConversionError(
                ["standard_splat_repository_identity_unavailable"]
            ) from exc

    if run("status", "--porcelain", "--untracked-files=no"):
        raise StandardSplatConversionError(
            ["standard_splat_repository_tracked_files_dirty"]
        )
    return {
        "commit": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "tracked_files_clean": True,
    }


def build_standard_splat_conversion_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StandardSplatConversionError(
            ["standard_splat_request_not_json"]
        ) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("standard_splat_request_schema_invalid")
    if request.get("program_id") != "arm-decision-proof-v1":
        errors.append("standard_splat_program_identity_invalid")
    if request.get("frozen_before_conversion") is not True:
        errors.append("standard_splat_request_not_frozen")
    if request.get("learned_policy_outcomes_observed") is not False:
        errors.append("standard_splat_policy_outcome_leakage")
    source = request.get("source")
    if not isinstance(source, Mapping):
        errors.append("standard_splat_source_missing")
    else:
        for key in ("relative_path", "dataset", "revision", "license"):
            if not str(source.get(key) or ""):
                errors.append(f"standard_splat_source_{key}_missing")
        if not _is_digest(source.get("sha256")):
            errors.append("standard_splat_source_sha256_invalid")
        size = source.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            errors.append("standard_splat_source_size_invalid")
    rights = request.get("rights")
    if not isinstance(rights, Mapping):
        errors.append("standard_splat_rights_missing")
    elif (
        rights.get("conversion_execution_location") != "local_only"
        or rights.get("raw_private_upload_authorized") is not False
        or rights.get("training_authorized") is not False
        or not _is_digest(rights.get("terms_digest"))
    ):
        errors.append("standard_splat_rights_invalid")
    output_filename = str(request.get("output_filename") or "")
    if (
        not output_filename.endswith(".ply")
        or "/" in output_filename
        or "\\" in output_filename
        or output_filename.startswith(".")
    ):
        errors.append("standard_splat_output_filename_invalid")
    if errors:
        raise StandardSplatConversionError(errors)
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        raise StandardSplatConversionError(
            ["standard_splat_request_digest_mismatch"]
        )
    request["request_digest"] = expected
    return request


def materialize_standard_splat_conversion(
    *,
    request_path: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    request_file = _under(
        repo, request_path, code="standard_splat_request_outside_repo"
    )
    output = _under(
        data, output_root, code="standard_splat_output_outside_data_root"
    )
    retained_receipt = (
        _under(repo, receipt_output, code="standard_splat_receipt_outside_repo")
        if receipt_output is not None
        else None
    )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise StandardSplatConversionError(
            ["standard_splat_output_not_empty"]
        )
    try:
        parsed = json.loads(request_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandardSplatConversionError(
            ["standard_splat_request_unreadable"]
        ) from exc
    if not isinstance(parsed, Mapping):
        raise StandardSplatConversionError(
            ["standard_splat_request_unreadable"]
        )
    request = build_standard_splat_conversion_request(parsed)
    source_record = request["source"]
    source = _under(
        data,
        str(source_record["relative_path"]),
        code="standard_splat_source_outside_data_root",
    )
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size != source_record["size_bytes"]
        or _sha256(source) != source_record["sha256"]
    ):
        raise StandardSplatConversionError(
            ["standard_splat_source_bytes_changed"]
        )
    repository = _repository_identity(repo)
    cli = find_splat_transform_cli(repo)
    if cli is None or not cli.is_file() or cli.is_symlink():
        raise StandardSplatConversionError(
            ["standard_splat_decoder_missing"]
        )
    package_path = cli.parents[1] / "package.json"
    if not package_path.is_file() or package_path.is_symlink():
        raise StandardSplatConversionError(
            ["standard_splat_decoder_package_missing"]
        )
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StandardSplatConversionError(
            ["standard_splat_decoder_package_invalid"]
        ) from exc
    decoder_version = str(package.get("version") or "")
    if not decoder_version:
        raise StandardSplatConversionError(
            ["standard_splat_decoder_version_missing"]
        )
    source_count = read_compressed_ply_chunk_bounds(source).vertex_count
    output.mkdir(parents=True, exist_ok=True)
    standard = output / str(request["output_filename"])
    conversion = convert_to_standard_ply(
        source,
        standard,
        repo_root=repo,
        timeout_seconds=1800,
    )
    if conversion.get("status") != "completed" or not standard.is_file():
        raise StandardSplatConversionError(
            [
                "standard_splat_conversion_failed",
                *[str(code) for code in conversion.get("blockers", ())],
            ]
        )
    decoded = read_standard_3dgs_ply(standard)
    if decoded.count != source_count:
        raise StandardSplatConversionError(
            ["standard_splat_gaussian_count_changed"]
        )
    if _sha256(source) != source_record["sha256"]:
        raise StandardSplatConversionError(
            ["standard_splat_source_bytes_changed"]
        )
    command = [
        "node",
        str(cli),
        "-w",
        "-q",
        str(source),
        str(standard),
    ]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "standard_splat_conversion_materialized",
        "program_id": "arm-decision-proof-v1",
        "request_digest": request["request_digest"],
        "repository": repository,
        "source": {
            **dict(source_record),
            "source_bytes_unchanged": True,
            "source_gaussian_count": source_count,
        },
        "output": {
            **_record(standard, output),
            "standard_3dgs_schema_validated": True,
            "gaussian_count": decoded.count,
            "gaussian_count_preserved": True,
        },
        "decoder": {
            "name": "@playcanvas/splat-transform",
            "version": decoder_version,
            "cli_sha256": _sha256(cli),
            "package_manifest_sha256": _sha256(package_path),
            "command": command,
            "conversion_result": conversion,
        },
        "rights": dict(request["rights"]),
        "raw_source_uploaded": False,
        "learned_policy_outcomes_used": False,
        "gaussian_ownership_claimed": False,
        "render_qualification_claimed": False,
        "claim_ceiling": "local_format_conversion_only",
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = output / f"{RECEIPT_SCHEMA}.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    if retained_receipt is not None:
        retained_receipt.parent.mkdir(parents=True, exist_ok=True)
        retained_receipt.write_text(
            canonical_json(receipt) + "\n", encoding="utf-8"
        )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt-output")
    args = parser.parse_args(argv)
    receipt = materialize_standard_splat_conversion(
        request_path=args.request,
        repo_root=args.repo_root,
        data_root=args.data_root,
        output_root=args.output_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json(receipt))
    return 0


__all__ = [
    "REQUEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "StandardSplatConversionError",
    "build_standard_splat_conversion_request",
    "materialize_standard_splat_conversion",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
