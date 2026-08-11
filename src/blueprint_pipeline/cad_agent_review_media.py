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
VISUAL_REVIEW_SCHEMA_VERSION = "scene_replacement_cad_agent_visual_reference_review.v1"
CONTACT_SHEET_FILENAME = "cad_agent_visual_comparison.png"
HTML_FILENAME = "OPEN_ME_cad_agent_visual_comparison.html"
RECEIPT_FILENAME = "cad_agent_visual_comparison.v1.json"

_REVIEWER_KINDS = frozenset(
    {
        "human_visual_review",
        "codex_visual_review",
        "external_vision_model_review",
    }
)
_REVIEW_STATUSES = frozenset(
    {
        "conditionally_admitted_for_construction",
        "rejected_visible_mismatch",
    }
)


class CadAgentReviewMediaError(ValueError):
    """Fail-closed CAD review media error."""


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CadAgentReviewMediaError("cad_review_payload_invalid") from exc
    if not isinstance(result, dict):
        raise CadAgentReviewMediaError("cad_review_payload_invalid")
    return result


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


def _verified_visual_image_record(
    record: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    """Verify that a purported reference or snapshot is an actual decodable image."""

    verified = _verified_file_record(record, role=role)
    try:
        with Image.open(verified["path"]) as image:
            image.verify()
        with Image.open(verified["path"]) as image:
            width, height = image.size
            image_format = str(image.format or "")
    except Exception as exc:  # pragma: no cover - Pillow has format-specific errors
        raise CadAgentReviewMediaError(f"cad_review_image_unreadable:{role}") from exc
    if width <= 0 or height <= 0 or not image_format:
        raise CadAgentReviewMediaError(f"cad_review_image_dimensions_invalid:{role}")
    return {
        **verified,
        "image_format": image_format,
        "width_px": int(width),
        "height_px": int(height),
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
        scene_id: str | None = None
        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            request = candidate["request"]
            inputs = request["inputs"]
            backend_id = request["backend"]["backend_id"]
            candidate_scene_id = str(request.get("scene_id") or "")
            if not candidate_scene_id:
                raise CadAgentReviewMediaError("cad_review_candidate_scene_missing")
            if scene_id is None:
                scene_id = candidate_scene_id
            elif scene_id != candidate_scene_id:
                raise CadAgentReviewMediaError("cad_review_candidate_scene_mismatch")
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
                    _verified_visual_image_record(ref, role=f"reference:{backend_id}")
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
                    "snapshot": _verified_visual_image_record(
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
            or scene_id is None
        ):
            raise CadAgentReviewMediaError("cad_review_reference_missing")
        rows.append(
            {
                "replacement_slot": object_row["replacement_slot"],
                "scene_id": scene_id,
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


def _open_reference_strip(
    records: Sequence[Mapping[str, Any]], *, size: tuple[int, int]
) -> Image.Image:
    """Show every manifest-bound observed frame rather than one convenient thumbnail."""

    if not records:
        raise CadAgentReviewMediaError("cad_review_reference_missing")
    columns = 1 if len(records) == 1 else 2
    rows = (len(records) + columns - 1) // columns
    gap = 8
    width = max(1, (size[0] - gap * (columns - 1)) // columns)
    height = max(1, (size[1] - gap * (rows - 1)) // rows)
    strip = Image.new("RGB", size, (240, 240, 238))
    for index, record in enumerate(records):
        image = _open_thumbnail(record, size=(width, height))
        x0 = (index % columns) * (width + gap) + (width - image.width) // 2
        y0 = (index // columns) * (height + gap) + (height - image.height) // 2
        strip.paste(image, (x0, y0))
    return strip


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
            (
                "observed_reference",
                row["reference_thumbnail"],
                f"{len(row['reference_images'])} observed source frame(s)",
            )
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
            thumbnail = (
                _open_reference_strip(
                    row["reference_images"], size=(cell_w - 40, cell_h - 95)
                )
                if _label == "observed_reference"
                else _open_thumbnail(record, size=(cell_w - 40, cell_h - 95))
            )
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
    scene_ids = {str(row["scene_id"]) for row in rows}
    if len(scene_ids) != 1:
        raise CadAgentReviewMediaError("cad_review_candidate_scene_mismatch")
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
        "scene_id": scene_ids.pop(),
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


def _review_media_catalog(
    value: Mapping[str, Any], *, verify_files: bool
) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    """Return the exact candidate/reference rows available to a visual reviewer."""

    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "review_media_materialized"
        or value.get("receipt_digest")
        != canonical_digest(value, digest_field="receipt_digest")
        or value.get("maximum_replacement_objects") != MAX_REPLACEMENT_OBJECTS
        or not isinstance(value.get("rows"), list)
        or not value["rows"]
        or len(value["rows"]) > MAX_REPLACEMENT_OBJECTS
        or value.get("object_count") != len(value["rows"])
        or not str(value.get("scene_id") or "")
    ):
        raise CadAgentReviewMediaError("cad_review_media_receipt_invalid")
    try:
        matrix_record = _verified_file_record(value.get("cad_matrix") or {}, role="cad_matrix")
        matrix = validate_cad_agent_matrix(_load_json(Path(matrix_record["path"])))
    except (CadAgentReviewMediaError, SimReadyCadAgentContractError) as exc:
        raise CadAgentReviewMediaError("cad_review_media_receipt_invalid") from exc
    if (
        value.get("cad_matrix_digest") != matrix.get("matrix_digest")
        or len(matrix.get("objects") or []) != len(value["rows"])
    ):
        raise CadAgentReviewMediaError("cad_review_media_receipt_invalid")
    catalog: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row_index, row in enumerate(value["rows"]):
        if not isinstance(row, Mapping):
            raise CadAgentReviewMediaError("cad_review_media_receipt_invalid")
        slot = row.get("replacement_slot")
        scene_id = str(row.get("scene_id") or "")
        task_id = str(row.get("task_id") or "")
        asset_id = str(row.get("asset_id") or "")
        references = row.get("reference_images")
        candidates = row.get("candidates")
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or slot < 1
            or slot > MAX_REPLACEMENT_OBJECTS
            or scene_id != value.get("scene_id")
            or not task_id
            or not asset_id
            or not isinstance(references, list)
            or not references
            or not isinstance(candidates, list)
            or not candidates
        ):
            raise CadAgentReviewMediaError("cad_review_media_receipt_invalid")
        verified_references = [
            _verified_visual_image_record(
                reference, role=f"review_reference:{row_index}:{reference_index}"
            )
            for reference_index, reference in enumerate(references)
        ]
        reference_digests = [record["sha256"] for record in verified_references]
        if (
            row.get("reference_signature") != _reference_signature(verified_references)
            or len(reference_digests) != len(set(reference_digests))
        ):
            raise CadAgentReviewMediaError("cad_review_media_receipt_invalid")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise CadAgentReviewMediaError("cad_review_media_receipt_invalid")
            backend_id = str(candidate.get("backend_id") or "")
            output_digest = str(candidate.get("output_receipt_digest") or "")
            key = (slot, task_id, asset_id, backend_id)
            if (
                not backend_id
                or not output_digest.startswith("sha256:")
                or key in catalog
            ):
                raise CadAgentReviewMediaError("cad_review_media_receipt_invalid")
            _verified_visual_image_record(
                candidate.get("snapshot") or {},
                role=f"review_snapshot:{row_index}:{candidate_index}",
            )
            catalog[key] = {
                "replacement_slot": slot,
                "scene_id": scene_id,
                "task_id": task_id,
                "asset_id": asset_id,
                "backend_id": backend_id,
                "cad_agent_output_receipt_digest": output_digest,
                "reference_signature": row["reference_signature"],
                "reference_image_digests": reference_digests,
            }
    return catalog


def _reviewer(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CadAgentReviewMediaError("cad_review_reviewer_invalid")
    kind = str(value.get("reviewer_kind") or "")
    identity = str(value.get("reviewer_id") or "").strip()
    input_mode = str(value.get("visual_input_mode") or "")
    if (
        kind not in _REVIEWER_KINDS
        or not identity
        or input_mode != "all_manifest_bound_reference_frames_and_candidate_snapshots"
    ):
        raise CadAgentReviewMediaError("cad_review_reviewer_invalid")
    return {
        "reviewer_kind": kind,
        "reviewer_id": identity,
        "visual_input_mode": input_mode,
    }


def _canonical_review_decision(
    value: Any, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CadAgentReviewMediaError("cad_review_decision_invalid")
    identity = {
        field: value.get(field)
        for field in ("replacement_slot", "task_id", "asset_id", "backend_id")
    }
    if identity != {
        field: expected[field]
        for field in ("replacement_slot", "task_id", "asset_id", "backend_id")
    }:
        raise CadAgentReviewMediaError("cad_review_decision_identity_mismatch")
    if (
        value.get("cad_agent_output_receipt_digest")
        != expected["cad_agent_output_receipt_digest"]
        or value.get("reference_signature") != expected["reference_signature"]
    ):
        raise CadAgentReviewMediaError("cad_review_decision_identity_mismatch")
    reviewed = value.get("reviewed_reference_image_digests")
    expected_digests = expected["reference_image_digests"]
    if reviewed != expected_digests:
        raise CadAgentReviewMediaError("cad_review_decision_reference_coverage_invalid")
    status = str(value.get("review_status") or "")
    if status not in _REVIEW_STATUSES:
        raise CadAgentReviewMediaError("cad_review_decision_status_invalid")
    findings = value.get("observed_feature_findings")
    if not isinstance(findings, list) or not findings:
        raise CadAgentReviewMediaError("cad_review_decision_findings_invalid")
    canonical_findings: list[dict[str, Any]] = []
    seen_reference_digests: set[str] = set()
    seen_feature_ids: set[str] = set()
    for index, finding in enumerate(findings):
        if not isinstance(finding, Mapping):
            raise CadAgentReviewMediaError("cad_review_decision_findings_invalid")
        feature_id = str(finding.get("feature_id") or "").strip()
        finding_status = str(finding.get("status") or "")
        evidence = finding.get("evidence_reference_image_digests")
        if (
            not feature_id
            or feature_id in seen_feature_ids
            or finding_status not in {"matched", "mismatch", "unresolved"}
            or not isinstance(evidence, list)
            or not evidence
            or any(digest not in expected_digests for digest in evidence)
            or len(evidence) != len(set(evidence))
        ):
            raise CadAgentReviewMediaError("cad_review_decision_findings_invalid")
        seen_feature_ids.add(feature_id)
        seen_reference_digests.update(evidence)
        canonical_findings.append(
            {
                "feature_id": feature_id,
                "status": finding_status,
                "evidence_reference_image_digests": list(evidence),
            }
        )
    if seen_reference_digests != set(expected_digests):
        raise CadAgentReviewMediaError("cad_review_decision_reference_coverage_invalid")
    mismatch_codes = value.get("visible_mismatch_codes")
    generated_labels = value.get("generated_candidate_content_labels")
    if (
        not isinstance(mismatch_codes, list)
        or any(not str(code).strip() for code in mismatch_codes)
        or len(mismatch_codes) != len(set(str(code) for code in mismatch_codes))
        or not isinstance(generated_labels, list)
        or not generated_labels
        or any(not str(label).strip() for label in generated_labels)
        or len(generated_labels) != len(set(str(label) for label in generated_labels))
    ):
        raise CadAgentReviewMediaError("cad_review_decision_labels_invalid")
    return {
        **identity,
        "cad_agent_output_receipt_digest": expected[
            "cad_agent_output_receipt_digest"
        ],
        "reference_signature": expected["reference_signature"],
        "reviewed_reference_image_digests": list(expected_digests),
        "review_status": status,
        "observed_feature_findings": sorted(
            canonical_findings, key=lambda finding: finding["feature_id"]
        ),
        "visible_mismatch_codes": sorted(str(code) for code in mismatch_codes),
        "generated_candidate_content_labels": sorted(
            str(label) for label in generated_labels
        ),
    }


def validate_cad_agent_visual_reference_review(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    """Validate an image-grounded, claim-bounded review of every CAD candidate."""

    review = _clone(value)
    errors: list[str] = []
    if review.get("schema_version") != VISUAL_REVIEW_SCHEMA_VERSION:
        errors.append("cad_review_schema_invalid")
    if review.get("status") != "all_candidates_visually_reviewed":
        errors.append("cad_review_status_invalid")
    try:
        reviewer = _reviewer(review.get("reviewer"))
    except CadAgentReviewMediaError as exc:
        errors.append(str(exc))
        reviewer = {}
    media_record = review.get("review_media")
    catalog: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    if not isinstance(media_record, Mapping):
        errors.append("cad_review_media_invalid")
    else:
        try:
            verified_media = _verified_file_record(media_record, role="review_media")
            media = _load_json(Path(verified_media["path"]))
            catalog = _review_media_catalog(media, verify_files=verify_files)
            if (
                review.get("review_media_digest") != media.get("receipt_digest")
                or review.get("scene_id") != media.get("scene_id")
            ):
                errors.append("cad_review_media_join_invalid")
        except CadAgentReviewMediaError as exc:
            errors.append(str(exc))
        except (OSError, json.JSONDecodeError):
            errors.append("cad_review_media_invalid")
    decisions = review.get("candidate_decisions")
    canonical_decisions: list[dict[str, Any]] = []
    if not isinstance(decisions, list) or not decisions:
        errors.append("cad_review_decisions_invalid")
    elif catalog:
        expected_keys = set(catalog)
        received_keys: set[tuple[int, str, str, str]] = set()
        for decision in decisions:
            if not isinstance(decision, Mapping):
                errors.append("cad_review_decision_invalid")
                continue
            key = (
                decision.get("replacement_slot"),
                str(decision.get("task_id") or ""),
                str(decision.get("asset_id") or ""),
                str(decision.get("backend_id") or ""),
            )
            if key not in catalog or key in received_keys:
                errors.append("cad_review_decision_identity_mismatch")
                continue
            received_keys.add(key)
            try:
                canonical_decisions.append(
                    _canonical_review_decision(decision, expected=catalog[key])
                )
            except CadAgentReviewMediaError as exc:
                errors.append(str(exc))
        if received_keys != expected_keys:
            errors.append("cad_review_decisions_incomplete")
    if review.get("candidate_count") != len(catalog):
        errors.append("cad_review_candidate_count_invalid")
    reference_sets = {
        (key[0], key[1], key[2]): row["reference_image_digests"]
        for key, row in catalog.items()
    }
    if review.get("reviewed_reference_image_count") != sum(
        len(reference_digests) for reference_digests in reference_sets.values()
    ):
        errors.append("cad_review_reference_count_invalid")
    expected_boundary = {
        "all_manifest_bound_reference_images_reviewed": True,
        "candidate_visual_similarity_automatically_proven": False,
        "appearance_materially_qualified": False,
        "simready_qualified": False,
        "physical_equivalence": False,
    }
    if review.get("claim_boundary") != expected_boundary:
        errors.append("cad_review_claim_boundary_invalid")
    if review.get("review_digest") != canonical_digest(review, digest_field="review_digest"):
        errors.append("cad_review_digest_invalid")
    if errors:
        raise CadAgentReviewMediaError(";".join(sorted(set(errors))))
    canonical = {
        **review,
        "reviewer": reviewer,
        "candidate_decisions": sorted(
            canonical_decisions,
            key=lambda decision: (
                decision["replacement_slot"],
                decision["task_id"],
                decision["asset_id"],
                decision["backend_id"],
            ),
        ),
    }
    if canonical != review:
        raise CadAgentReviewMediaError("cad_review_not_canonical")
    return review


def seal_cad_agent_visual_reference_review(
    *,
    review_media_receipt_path: str | Path,
    reviewer: Mapping[str, Any],
    candidate_decisions: Sequence[Mapping[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Seal an exhaustive image-grounded review before candidate construction."""

    media_path = Path(review_media_receipt_path).expanduser().resolve()
    media_record = _verified_file_record(
        {
            "path": str(media_path),
            "sha256": _sha256(media_path),
            "size_bytes": media_path.stat().st_size,
        },
        role="review_media",
    )
    media = _load_json(Path(media_record["path"]))
    catalog = _review_media_catalog(media, verify_files=True)
    canonical_decisions: list[dict[str, Any]] = []
    received_keys: set[tuple[int, str, str, str]] = set()
    for decision in candidate_decisions:
        if not isinstance(decision, Mapping):
            raise CadAgentReviewMediaError("cad_review_decision_invalid")
        key = (
            decision.get("replacement_slot"),
            str(decision.get("task_id") or ""),
            str(decision.get("asset_id") or ""),
            str(decision.get("backend_id") or ""),
        )
        if key not in catalog or key in received_keys:
            raise CadAgentReviewMediaError("cad_review_decision_identity_mismatch")
        received_keys.add(key)
        canonical_decisions.append(
            _canonical_review_decision(decision, expected=catalog[key])
        )
    if received_keys != set(catalog):
        raise CadAgentReviewMediaError("cad_review_decisions_incomplete")
    reference_sets = {
        (key[0], key[1], key[2]): row["reference_image_digests"]
        for key, row in catalog.items()
    }
    payload: dict[str, Any] = {
        "schema_version": VISUAL_REVIEW_SCHEMA_VERSION,
        "status": "all_candidates_visually_reviewed",
        "scene_id": str(media.get("scene_id") or ""),
        "review_media": media_record,
        "review_media_digest": media["receipt_digest"],
        "reviewer": _reviewer(reviewer),
        "candidate_count": len(catalog),
        "reviewed_reference_image_count": sum(
            len(reference_digests) for reference_digests in reference_sets.values()
        ),
        "candidate_decisions": sorted(
            canonical_decisions,
            key=lambda decision: (
                decision["replacement_slot"],
                decision["task_id"],
                decision["asset_id"],
                decision["backend_id"],
            ),
        ),
        "claim_boundary": {
            "all_manifest_bound_reference_images_reviewed": True,
            "candidate_visual_similarity_automatically_proven": False,
            "appearance_materially_qualified": False,
            "simready_qualified": False,
            "physical_equivalence": False,
        },
        "review_digest": "",
    }
    payload["review_digest"] = canonical_digest(payload, digest_field="review_digest")
    admitted = validate_cad_agent_visual_reference_review(payload)
    if output_path is not None:
        target = Path(output_path).expanduser().resolve()
        if target.exists() or target.is_symlink():
            raise CadAgentReviewMediaError("cad_review_destination_exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(admitted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return admitted


def selected_cad_agent_visual_review(
    review_record: Mapping[str, Any],
    *,
    scene_id: str,
    task_id: str,
    asset_id: str,
    backend_id: str,
    cad_agent_output_receipt_digest: str,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Return a selected candidate's review decision, failing closed if rejected."""

    verified = _verified_file_record(review_record, role="visual_review")
    review = _load_json(Path(verified["path"]))
    admitted = validate_cad_agent_visual_reference_review(
        review, verify_files=verify_files
    )
    if admitted.get("scene_id") != scene_id:
        raise CadAgentReviewMediaError("cad_review_selection_scene_mismatch")
    matches = [
        decision
        for decision in admitted["candidate_decisions"]
        if decision.get("task_id") == task_id
        and decision.get("asset_id") == asset_id
        and decision.get("backend_id") == backend_id
        and decision.get("cad_agent_output_receipt_digest")
        == cad_agent_output_receipt_digest
    ]
    if len(matches) != 1:
        raise CadAgentReviewMediaError("cad_review_selection_missing")
    decision = matches[0]
    if decision.get("review_status") != "conditionally_admitted_for_construction":
        raise CadAgentReviewMediaError("cad_review_selection_rejected")
    return decision
