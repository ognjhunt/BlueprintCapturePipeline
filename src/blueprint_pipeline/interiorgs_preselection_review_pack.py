"""Build one visual, digest-bound InteriorGS scene-selection review pack.

This joins the existing publisher-label room survey and local Spark renderer.
It is intentionally reconnaissance only: rendered previews and label boxes help
a person choose what to inspect, but they do not qualify appearance fidelity,
collision identity, task physics, robot reachability, or evaluation readiness.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .decision_evidence_contracts import canonical_digest
from .public_scene_viewpoint_survey import build_room_viewpoint_survey
from .splat_scene_render import render_splat_scene
from .task_evaluation_splat_render_runtime import validate_splat_render_runtime


SCHEMA_VERSION = "interiorgs_preselection_review_pack.v1"
MAX_TARGETS = 5
MAX_OVERLAY_SUPPORTS = 16
MAX_RENDER_CAMERAS = 96
_SUPPORT_LABEL_PARTS = (
    "table",
    "desk",
    "counter",
    "cabinet",
    "shelf",
    "tray",
    "plate",
    "bowl",
    "basket",
    "bin",
)
_STRUCTURAL_LABEL_PARTS = (
    "wall",
    "floor",
    "ceiling",
    "window",
    "door",
    "curtain",
    "rug",
    "carpet",
    "sofa",
    "chair",
    "bed",
)


class InteriorGSPreselectionReviewPackError(ValueError):
    """The requested pack or retained render evidence is not admissible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolved_under(path: str | Path, roots: Sequence[str | Path]) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = [Path(root).expanduser().resolve() for root in roots]
    if not allowed or not any(
        resolved == root or root in resolved.parents for root in allowed
    ):
        raise InteriorGSPreselectionReviewPackError(
            f"interiorgs_preselection_path_outside_approved_roots:{resolved}"
        )
    return resolved


def _source_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise InteriorGSPreselectionReviewPackError(
            "interiorgs_preselection_source_invalid"
        )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _all_objects(survey: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for room in survey.get("rooms") or []:
        if not isinstance(room, Mapping):
            continue
        room_id = str(room.get("room_id") or "")
        for item in room.get("publisher_objects") or []:
            if isinstance(item, Mapping):
                rows.append({"room_id": room_id, **dict(item)})
    return sorted(rows, key=lambda row: (row["room_id"], str(row["ins_id"])))


def _contains_any(label: str, parts: Sequence[str]) -> bool:
    normalized = "_".join(label.strip().lower().split())
    return any(part in normalized for part in parts)


def _classify_candidates(
    objects: Sequence[Mapping[str, Any]], *, target_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    movable: list[dict[str, Any]] = []
    supports: list[dict[str, Any]] = []
    for item in objects:
        label = str(item.get("label") or "")
        size = item.get("aabb_size_m")
        if (
            not isinstance(size, list)
            or len(size) != 3
            or any(not isinstance(value, (int, float)) or value <= 0 for value in size)
        ):
            continue
        row = {
            "ins_id": str(item["ins_id"]),
            "label": label,
            "room_id": item["room_id"],
            "centroid_world_m": item["centroid_world_m"],
            "aabb_min_world_m": item["aabb_min_world_m"],
            "aabb_max_world_m": item["aabb_max_world_m"],
            "aabb_size_m": size,
            "candidate_is_advisory_only": True,
        }
        if _contains_any(label, _SUPPORT_LABEL_PARTS):
            supports.append(row)
            continue
        volume = float(size[0]) * float(size[1]) * float(size[2])
        if (
            not _contains_any(label, _STRUCTURAL_LABEL_PARTS)
            and max(float(value) for value in size) <= 0.6
            and min(float(value) for value in size) >= 0.005
            and volume <= 0.06
        ):
            movable.append(
                {
                    **row,
                    "explicit_review_target": str(item["ins_id"]) in target_ids,
                }
            )
    movable.sort(
        key=lambda row: (
            not row["explicit_review_target"],
            row["room_id"],
            row["label"],
            row["ins_id"],
        )
    )
    supports.sort(key=lambda row: (row["room_id"], row["label"], row["ins_id"]))
    return movable, supports


def _overlay_items(
    objects: Sequence[Mapping[str, Any]],
    *,
    target_ids: set[str],
    support_ids: set[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in objects:
        ins_id = str(item.get("ins_id") or "")
        if ins_id not in target_ids and ins_id not in support_ids:
            continue
        items.append(
            {
                "id": f"{item.get('label')}_{ins_id}",
                "shape": "box",
                "position": list(item["centroid_world_m"]),
                "scale": list(item["aabb_size_m"]),
                "color": 0xFF6B35 if ins_id in target_ids else 0x22CCFF,
                "opacity": 0.16 if ins_id in target_ids else 0.08,
                "role": "review_target" if ins_id in target_ids else "support_candidate",
            }
        )
    return items


def _runtime_identity(node: str, repo_root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [node, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        node_version = completed.stdout.strip() if completed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        node_version = None
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("splat_scene_render.py"),
        repo_root / "tools/splat_render/render_splat.mjs",
        repo_root / "tools/splat_render/src/render_entry.mjs",
        repo_root / "tools/splat_render/package-lock.json",
    )
    return {
        "node_version": node_version,
        "renderer_sources": [_source_record(path) for path in paths],
    }


def _verified_frames(
    render: Mapping[str, Any], *, output_root: Path
) -> list[dict[str, Any]]:
    rows = render.get("cameras")
    if render.get("status") != "completed" or not isinstance(rows, list) or not rows:
        return []
    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            return []
        camera_id = str(row.get("id") or "")
        path = Path(str(row.get("path") or "")).expanduser().resolve()
        if (
            not camera_id
            or camera_id in seen
            or path.is_symlink()
            or not path.is_file()
            or output_root not in path.parents
        ):
            return []
        seen.add(camera_id)
        frames.append(
            {
                "camera_id": camera_id,
                "path": str(path.relative_to(output_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return frames


def _write_contact_sheet(frames: Sequence[Mapping[str, Any]], root: Path) -> Path:
    columns = 4
    cell_width, cell_height, label_height = 480, 360, 32
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "#111515")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(frames):
        source = root / str(record["path"])
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            image.thumbnail((cell_width, cell_height))
        x = (index % columns) * cell_width
        y = (index // columns) * (cell_height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 9), str(record["camera_id"]), fill="#f2f4ef", font=font)
    destination = root / "contact_sheet.png"
    sheet.save(destination)
    return destination


def _write_html(
    *,
    scene_id: str,
    frames: Sequence[Mapping[str, Any]],
    movable: Sequence[Mapping[str, Any]],
    supports: Sequence[Mapping[str, Any]],
    root: Path,
) -> Path:
    def table(rows: Sequence[Mapping[str, Any]]) -> str:
        body = "".join(
            "<tr>"
            f"<td>{escape(str(row['ins_id']))}</td>"
            f"<td>{escape(str(row['label']))}</td>"
            f"<td>{escape(str(row['room_id']))}</td>"
            f"<td>{escape(json.dumps(row['aabb_size_m']))}</td>"
            "</tr>"
            for row in rows
        )
        return "<table><tr><th>ID</th><th>Label</th><th>Room</th><th>Size m</th></tr>" + body + "</table>"

    gallery = "".join(
        "<figure>"
        f"<img loading='lazy' src='{escape(str(frame['path']))}' alt='{escape(str(frame['camera_id']))}'>"
        f"<figcaption>{escape(str(frame['camera_id']))}</figcaption>"
        "</figure>"
        for frame in frames
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>InteriorGS {escape(scene_id)} review</title>
<style>body{{font:15px system-ui;background:#101414;color:#eef1ed;margin:24px}} .warn{{color:#ffbf47}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}} figure{{margin:0;background:#181d1d;padding:10px}} img{{width:100%;height:auto}} table{{border-collapse:collapse;width:100%;margin-bottom:28px}} td,th{{border:1px solid #3b4444;padding:7px;text-align:left}} th{{background:#202727}}</style>
</head><body><h1>InteriorGS {escape(scene_id)} preselection review</h1>
<p class="warn">Reconnaissance only. These previews do not qualify rendering, collision, physics, reachability, or evaluation readiness.</p>
<h2>Rendered views</h2><div class="grid">{gallery}</div>
<h2>Advisory movable rigid candidates</h2>{table(movable)}
<h2>Advisory support/receptacle candidates</h2>{table(supports)}
</body></html>"""
    destination = root / "index.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def build_interiorgs_preselection_review_pack(
    *,
    scene_id: str,
    splat_path: str | Path,
    labels_path: str | Path,
    structure_path: str | Path,
    output_root: str | Path,
    approved_roots: Sequence[str | Path],
    target_ins_ids: Sequence[str] = (),
    width: int = 1280,
    height: int = 960,
    graphics_backend: str = "swiftshader",
    node: str = "node",
    browser_executable: str | Path | None = None,
    apply_overlay: bool = False,
    repo_root: str | Path | None = None,
    production_runtime_root: str | Path | None = None,
    renderer: Callable[..., Mapping[str, Any]] = render_splat_scene,
) -> dict[str, Any]:
    """Create room and target renders, labeled tables, and one sealed manifest."""

    if not scene_id.strip() or not 1 <= len(set(target_ins_ids)) <= MAX_TARGETS:
        raise InteriorGSPreselectionReviewPackError(
            "interiorgs_preselection_scene_or_target_invalid"
        )
    if isinstance(width, bool) or isinstance(height, bool) or not (
        320 <= width <= 4096 and 240 <= height <= 4096
    ):
        raise InteriorGSPreselectionReviewPackError(
            "interiorgs_preselection_dimensions_invalid"
        )
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    production_identity = None
    if production_runtime_root is not None:
        runtime = validate_splat_render_runtime(
            runtime_root=production_runtime_root,
            repo_root=Path(__file__).resolve().parents[2],
        )
        root = Path(runtime["renderer_root"])
        node = runtime["node"]
        browser_executable = runtime["browser_executable"]
        production_identity = runtime["identity"]
    # Validate provenance before rendering; Python and JavaScript are shipped
    # in separate trees in the sealed production runtime.
    runtime_identity = _runtime_identity(node, root)
    if production_identity is not None:
        runtime_identity["production_runtime"] = production_identity
    output = _resolved_under(output_root, approved_roots)
    if output.exists() or output.is_symlink():
        raise InteriorGSPreselectionReviewPackError(
            "interiorgs_preselection_output_exists"
        )
    splat = _resolved_under(splat_path, approved_roots)
    labels = _resolved_under(labels_path, approved_roots)
    structure = _resolved_under(structure_path, approved_roots)
    source_files = {
        "splat": _source_record(splat),
        "labels": _source_record(labels),
        "structure": _source_record(structure),
    }
    output.mkdir(parents=True, mode=0o750)
    base_survey = build_room_viewpoint_survey(
        structure_path=structure,
        labels_path=labels,
        scene_id=scene_id,
        approved_roots=approved_roots,
    )
    cameras = [dict(row) for row in base_survey["cameras"]]
    target_surveys: list[dict[str, Any]] = []
    for target_id in dict.fromkeys(str(value) for value in target_ins_ids):
        target_survey = build_room_viewpoint_survey(
            structure_path=structure,
            labels_path=labels,
            scene_id=scene_id,
            approved_roots=approved_roots,
            target_ins_id=target_id,
        )
        closeup = dict(target_survey["target_closeup"])
        target_surveys.append(closeup)
        cameras.extend(dict(row) for row in closeup["cameras"])
    if len(cameras) > MAX_RENDER_CAMERAS or len({row["id"] for row in cameras}) != len(cameras):
        raise InteriorGSPreselectionReviewPackError(
            "interiorgs_preselection_camera_inventory_invalid"
        )
    objects = _all_objects(base_survey)
    target_ids = {str(value) for value in target_ins_ids}
    movable, supports = _classify_candidates(objects, target_ids=target_ids)
    target_rooms = {
        str(row["room_id"])
        for row in objects
        if str(row["ins_id"]) in target_ids
    }
    support_rows = [row for row in supports if row["room_id"] in target_rooms]
    support_ids = {
        str(row["ins_id"]) for row in support_rows[:MAX_OVERLAY_SUPPORTS]
    }
    overlay = _overlay_items(
        objects, target_ids=target_ids, support_ids=support_ids
    )
    overlay_path = output / "review_overlay.json"
    overlay_path.write_text(json.dumps(overlay, indent=2) + "\n", encoding="utf-8")
    cameras_path = output / "cameras.json"
    cameras_path.write_text(
        json.dumps(cameras, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_output = output / "render"
    render = dict(
        renderer(
            splat,
            render_output,
            camera_specs=cameras,
            width=width,
            height=height,
            decimate=0,
            settle_frames=12,
            warmup_ms=3500,
            graphics_backend=graphics_backend,
            browser_executable=browser_executable,
            overlay_json=overlay_path if apply_overlay else None,
            repo_root=root,
            node=node,
        )
    )
    frames = _verified_frames(render, output_root=output)
    status = "completed" if frames and render.get("status") == "completed" else "blocked"
    contact_sheet = _write_contact_sheet(frames, output) if frames else None
    html_path = (
        _write_html(
            scene_id=scene_id,
            frames=frames,
            movable=movable,
            supports=supports,
            root=output,
        )
        if frames
        else None
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "blockers": [] if status == "completed" else ["interiorgs_preselection_render_incomplete"],
        "scene_id": scene_id,
        "purpose": "selection_reconnaissance_only",
        "source_files": source_files,
        "survey": base_survey,
        "target_closeups": target_surveys,
        "cameras": cameras,
        "camera_file": _source_record(cameras_path),
        "overlay": {
            "items": overlay,
            "file": _source_record(overlay_path),
            "applied_to_render": apply_overlay,
            "default_is_clean_render_to_avoid_occluding_source_appearance": True,
        },
        "advisory_rigid_movable_candidates": movable,
        "advisory_support_candidates": supports,
        "render": render,
        "frames": frames,
        "contact_sheet": _source_record(contact_sheet) if contact_sheet else None,
        "html_review": _source_record(html_path) if html_path else None,
        "runtime_identity": runtime_identity,
        "claim_boundary": {
            "selection_reconnaissance_only": True,
            "method_input": False,
            "evaluation_authorized": False,
            "appearance_fidelity_qualified": False,
            "collision_identity_established": False,
            "robot_reachability_established": False,
            "camera_motion_recovers_uncaptured_regions": False,
        },
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = output / f"{SCHEMA_VERSION}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--splat", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--approved-root", action="append", required=True)
    parser.add_argument("--target-ins-id", action="append", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--graphics-backend", choices=("swiftshader", "metal"), default="swiftshader"
    )
    parser.add_argument("--node", default="node")
    parser.add_argument(
        "--browser-executable",
        help="optional existing Chrome/Chromium executable for local rendering",
    )
    parser.add_argument(
        "--apply-overlay",
        action="store_true",
        help="render translucent advisory boxes; clean renders are the default",
    )
    parser.add_argument(
        "--repo-root",
        help="optional checkout containing the pinned local Spark renderer runtime",
    )
    parser.add_argument(
        "--production-runtime-root",
        help="validate and use the sealed production Node/browser/renderer runtime",
    )
    args = parser.parse_args(argv)
    result = build_interiorgs_preselection_review_pack(
        scene_id=args.scene_id,
        splat_path=args.splat,
        labels_path=args.labels,
        structure_path=args.structure,
        output_root=args.out,
        approved_roots=args.approved_root,
        target_ins_ids=args.target_ins_id,
        width=args.width,
        height=args.height,
        graphics_backend=args.graphics_backend,
        node=args.node,
        browser_executable=args.browser_executable,
        apply_overlay=args.apply_overlay,
        repo_root=args.repo_root,
        production_runtime_root=args.production_runtime_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_path": result["manifest_path"],
                "manifest_digest": result["manifest_digest"],
                "html_review": result["html_review"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "InteriorGSPreselectionReviewPackError",
    "SCHEMA_VERSION",
    "build_interiorgs_preselection_review_pack",
    "main",
]
