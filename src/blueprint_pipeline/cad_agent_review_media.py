"""Human-review media for manifest-bound CAD-agent candidates.

This module packages agent-authored CAD candidates for side-by-side review
without selecting a winner or upgrading any candidate into SimReady/native
evidence.  The input is the scene-level CAD matrix: up to five replacement
objects, each with the admitted CAD backends.  The output is a digest-bound
contact sheet plus a small HTML page for Finder/browser review.
"""

from __future__ import annotations

import hashlib
import html
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

try:  # flat provider-bundle layout
    from decision_evidence_contracts import canonical_digest
    from simready_cad_agent_contract import (
        ADMITTED_BACKENDS,
        MAX_REPLACEMENT_OBJECTS,
        SimReadyCadAgentContractError,
        validate_cad_agent_matrix,
    )
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import canonical_digest
    from .simready_cad_agent_contract import (
        ADMITTED_BACKENDS,
        MAX_REPLACEMENT_OBJECTS,
        SimReadyCadAgentContractError,
        validate_cad_agent_matrix,
    )


SCHEMA_VERSION = "scene_replacement_cad_agent_visual_comparison.v1"
CONTACT_SHEET_FILENAME = "cad_agent_visual_comparison.png"
HTML_FILENAME = "OPEN_ME_cad_agent_visual_comparison.html"
RECEIPT_FILENAME = "cad_agent_visual_comparison.v1.json"


class CadAgentReviewMediaError(ValueError):
    """Fail-closed CAD review media error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CadAgentReviewMediaError("cad_review_matrix_unreadable") from exc
    if not isinstance(value, dict):
        raise CadAgentReviewMediaError("cad_review_matrix_not_mapping")
    return value


def _verified_file_record(record: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    path_value = str(record.get("path") or "")
    if not path_value:
        raise CadAgentReviewMediaError(f"cad_review_file_path_missing:{role}")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise CadAgentReviewMediaError(f"cad_review_file_path_not_absolute:{role}")
    if path.is_symlink() or not path.is_file():
        raise CadAgentReviewMediaError(f"cad_review_file_missing:{role}")
    observed_sha = _sha256(path)
    if record.get("sha256") != observed_sha:
        raise CadAgentReviewMediaError(f"cad_review_file_digest_mismatch:{role}")
    if int(record.get("size_bytes", -1)) != path.stat().st_size:
        raise CadAgentReviewMediaError(f"cad_review_file_size_mismatch:{role}")
    return {
        "path": str(path),
        "sha256": observed_sha,
        "size_bytes": path.stat().st_size,
    }


def _output_file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CadAgentReviewMediaError("cad_review_output_missing")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _reference_signature(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest(
        {
            "reference_images": [
                {
                    "sha256": str(row.get("sha256") or ""),
                    "size_bytes": int(row.get("size_bytes", -1)),
                }
                for row in records
            ]
        }
    )


def _file_identity(record: Mapping[str, Any]) -> tuple[int, str]:
    size = record.get("size_bytes")
    digest = str(record.get("sha256") or "")
    if not isinstance(size, int) or size <= 0 or not digest.startswith("sha256:"):
        raise CadAgentReviewMediaError("cad_review_file_identity_invalid")
    return (int(size), digest)


def _choose_snapshot(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records:
        raise CadAgentReviewMediaError("cad_review_snapshot_missing")
    for token in ("iso", "front", "open"):
        for record in records:
            if token in Path(str(record.get("path") or "")).name.lower():
                return record
    return records[0]


def _rows_from_matrix(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for object_row in matrix["objects"]:
        candidates = object_row["candidates"]
        reference_signature: str | None = None
        reference_images: list[dict[str, Any]] | None = None
        reference_manifest: dict[str, Any] | None = None
        reference_manifest_identity: tuple[int, str] | None = None
        reference_manifest_object_digest: str | None = None
        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            request = candidate["request"]
            inputs = request["inputs"]
            backend_id = request["backend"]["backend_id"]
            refs = inputs["reference_images"]
            candidate_reference_signature = _reference_signature(refs)
            candidate_reference_manifest = _verified_file_record(
                inputs["reference_manifest"],
                role=f"reference_manifest:{backend_id}",
            )
            candidate_reference_manifest_identity = _file_identity(
                candidate_reference_manifest
            )
            candidate_reference_manifest_object_digest = str(
                inputs.get("reference_manifest_object_digest") or ""
            )
            if reference_signature is None:
                reference_signature = candidate_reference_signature
                reference_manifest = candidate_reference_manifest
                reference_manifest_identity = candidate_reference_manifest_identity
                reference_manifest_object_digest = (
                    candidate_reference_manifest_object_digest
                )
                reference_images = [
                    _verified_file_record(ref, role=f"reference:{backend_id}")
                    for ref in refs
                ]
            elif reference_signature != candidate_reference_signature:
                raise CadAgentReviewMediaError(
                    "cad_review_candidate_reference_mismatch"
                )
            elif (
                reference_manifest_identity
                != candidate_reference_manifest_identity
                or reference_manifest_object_digest
                != candidate_reference_manifest_object_digest
            ):
                raise CadAgentReviewMediaError(
                    "cad_review_candidate_reference_manifest_mismatch"
                )
            artifacts = candidate["artifacts"]
            snapshot = _choose_snapshot(artifacts.get("snapshots") or [])
            candidate_rows.append(
                {
                    "backend_id": backend_id,
                    "request_digest": candidate["request_digest"],
                    "output_receipt_digest": candidate["receipt_digest"],
                    "status": candidate["status"],
                    "measured_envelope_mm": candidate["measured_envelope_mm"],
                    "reference_manifest_object_digest": (
                        candidate_reference_manifest_object_digest
                    ),
                    "reference_binding_source": inputs["reference_binding_source"],
                    "snapshot": _verified_file_record(
                        snapshot, role=f"snapshot:{backend_id}"
                    ),
                    "step": _verified_file_record(
                        artifacts["step"], role=f"step:{backend_id}"
                    ),
                }
            )
        backend_ids = [row["backend_id"] for row in candidate_rows]
        if backend_ids != sorted(ADMITTED_BACKENDS):
            raise CadAgentReviewMediaError("cad_review_candidate_backend_order_invalid")
        if (
            reference_images is None
            or reference_signature is None
            or reference_manifest is None
            or reference_manifest_object_digest is None
        ):
            raise CadAgentReviewMediaError("cad_review_reference_missing")
        rows.append(
            {
                "replacement_slot": object_row["replacement_slot"],
                "task_id": object_row["task_id"],
                "asset_id": object_row["asset_id"],
                "reference_signature": reference_signature,
                "reference_manifest": reference_manifest,
                "reference_manifest_object_digest": reference_manifest_object_digest,
                "reference_binding_source": "manifest_derived",
                "reference_thumbnail": reference_images[0],
                "reference_images": reference_images,
                "candidates": candidate_rows,
            }
        )
    return rows


def _open_thumbnail(record: Mapping[str, Any], *, size: tuple[int, int]) -> Image.Image:
    try:
        image = Image.open(str(record["path"]))
        image.load()
    except Exception as exc:  # pragma: no cover - exact PIL exception varies
        raise CadAgentReviewMediaError("cad_review_image_unreadable") from exc
    image = image.convert("RGB")
    return ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    max_chars: int,
    line_height: int,
) -> None:
    x, y = position
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if len(candidate) > max_chars and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for index, rendered in enumerate(lines[:4]):
        draw.text((x, y + index * line_height), rendered, fill=fill)


def _render_contact_sheet(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    backend_ids = sorted(ADMITTED_BACKENDS)
    columns = ["observed_reference", *backend_ids]
    cell_w = 460
    cell_h = 390
    header_h = 90
    left_w = 220
    width = left_w + cell_w * len(columns)
    height = header_h + cell_h * len(rows)
    sheet = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, width, header_h), fill=(35, 38, 42))
    draw.text((20, 18), title, fill=(255, 255, 255))
    draw.text(
        (20, 48),
        "Observed reference + agent-authored CAD candidates; not SimReady/native evidence",
        fill=(220, 224, 228),
    )
    for col_index, column in enumerate(columns):
        x = left_w + col_index * cell_w
        draw.text((x + 18, header_h - 30), column, fill=(255, 255, 255))
    for row_index, row in enumerate(rows):
        y0 = header_h + row_index * cell_h
        draw.rectangle((0, y0, width, y0 + cell_h), outline=(210, 210, 210))
        _draw_wrapped(
            draw,
            (16, y0 + 24),
            f"slot {row['replacement_slot']} {row['task_id']} {row['asset_id']}",
            fill=(25, 25, 25),
            max_chars=24,
            line_height=18,
        )
        images: list[tuple[str, Mapping[str, Any], str]] = [
            ("observed_reference", row["reference_thumbnail"], "observed source frame")
        ]
        candidates_by_backend = {
            candidate["backend_id"]: candidate for candidate in row["candidates"]
        }
        for backend_id in backend_ids:
            candidate = candidates_by_backend[backend_id]
            envelope = " × ".join(
                f"{float(value):.1f}" for value in candidate["measured_envelope_mm"]
            )
            images.append((backend_id, candidate["snapshot"], f"{envelope} mm"))
        for col_index, (_label, record, caption) in enumerate(images):
            x0 = left_w + col_index * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), outline=(210, 210, 210))
            thumbnail = _open_thumbnail(record, size=(cell_w - 40, cell_h - 95))
            tx = x0 + (cell_w - thumbnail.width) // 2
            ty = y0 + 20
            sheet.paste(thumbnail, (tx, ty))
            draw.text((x0 + 18, y0 + cell_h - 62), caption, fill=(25, 25, 25))
            draw.text(
                (x0 + 18, y0 + cell_h - 38),
                str(record["sha256"])[:20] + "...",
                fill=(90, 90, 90),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
    ) as temporary:
        tmp_path = Path(temporary.name)
    try:
        sheet.save(tmp_path, format="PNG")
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _render_html(
    *,
    rows: Sequence[Mapping[str, Any]],
    contact_sheet_path: Path,
    output_path: Path,
    title: str,
) -> None:
    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        f"<title>{esc(title)}</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px}"
        "img{max-width:100%;border:1px solid #ddd}.row{margin:24px 0}"
        "code{font-size:12px}</style>",
        f"<h1>{esc(title)}</h1>",
        "<p>Claim boundary: review media only. Agent-authored CAD candidates are not "
        "SimReady, native-import, appearance, physics, or physical-equivalence evidence.</p>",
        f"<p><a href='{esc(contact_sheet_path.name)}'>Open contact sheet PNG</a></p>",
        f"<img src='{esc(contact_sheet_path.name)}' alt='CAD visual comparison contact sheet'>",
    ]
    for row in rows:
        lines.append("<div class='row'>")
        lines.append(
            f"<h2>Slot {esc(row['replacement_slot'])}: {esc(row['task_id'])}</h2>"
        )
        lines.append(f"<p>Asset: <code>{esc(row['asset_id'])}</code></p>")
        lines.append("<ul>")
        lines.append(
            f"<li>Reference digest: <code>{esc(row['reference_signature'])}</code></li>"
        )
        for candidate in row["candidates"]:
            lines.append(
                "<li>"
                f"{esc(candidate['backend_id'])}: output "
                f"<code>{esc(candidate['output_receipt_digest'])}</code>, STEP "
                f"<code>{esc(candidate['step']['sha256'])}</code>"
                "</li>"
            )
        lines.append("</ul></div>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_cad_agent_visual_comparison(
    *,
    matrix_path: str | Path,
    output_dir: str | Path,
    title: str = "CAD-agent visual comparison",
) -> dict[str, Any]:
    """Create digest-bound review media from a CAD-agent matrix."""

    matrix_file = Path(matrix_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    try:
        matrix = validate_cad_agent_matrix(_load_json(matrix_file))
    except SimReadyCadAgentContractError as exc:
        raise CadAgentReviewMediaError("cad_review_matrix_invalid") from exc
    object_count = len(matrix["objects"])
    if object_count < 1 or object_count > MAX_REPLACEMENT_OBJECTS:
        raise CadAgentReviewMediaError("cad_review_object_count_invalid")
    rows = _rows_from_matrix(matrix)
    output.mkdir(parents=True, exist_ok=True)
    contact_sheet_path = output / CONTACT_SHEET_FILENAME
    html_path = output / HTML_FILENAME
    _render_contact_sheet(rows=rows, output_path=contact_sheet_path, title=title)
    _render_html(
        rows=rows,
        contact_sheet_path=contact_sheet_path,
        output_path=html_path,
        title=title,
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_media_materialized",
        "cad_matrix": _output_file_record(matrix_file),
        "cad_matrix_digest": matrix["matrix_digest"],
        "object_count": object_count,
        "maximum_replacement_objects": MAX_REPLACEMENT_OBJECTS,
        "replacement_object_capacity": {
            "minimum": 1,
            "maximum": MAX_REPLACEMENT_OBJECTS,
            "sealed_slots": object_count,
        },
        "backend_ids": sorted(ADMITTED_BACKENDS),
        "rows": rows,
        "contact_sheet": _output_file_record(contact_sheet_path),
        "html": _output_file_record(html_path),
        "claim_boundary": {
            "human_review_media_only": True,
            "agent_authored_cad_candidate": True,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "appearance_qualified": False,
            "physics_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = output / RECEIPT_FILENAME
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt
