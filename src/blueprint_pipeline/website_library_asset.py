"""A library model's files, fetched once and checked, and the model sized to the job's object (``website_object_library``).

Fetching: only from an allow-listed host (``ALLOWED_HOSTS``) through the pipeline's outbound boundary
(``safe_outbound_http.download_file``), every file size-capped and checked against the sha256 the catalog pinned, and
cached by that digest, so a rerun downloads nothing and a changed file is never used.

Sizing (``library_design``, from compile's estimated bounds of the object): the model is used at its own size when
every side is within the metric envelope's tolerance of the estimate (the authoring step checks the same), turned a
quarter turn about its up axis when its footprint lies the other way. Otherwise it is scaled uniformly so its largest
side is the estimate's, and the record says "scaled to the video's estimate": the estimate comes from the video and is
never a measurement. A model whose proportions are clearly not the object's (a side still outside the tolerance after
scaling) is refused with its ratios (``library_aspect_mismatch``); the next source is then tried.

The model file (``library_model.usda``, the model as Blueprint converted and, when it did, scaled it, in metres with Z
up) is the one file of library material: compile writes it once by content, the observation step places it where the
object stood as construction's source candidate (``write_candidate``, as ``website_parametric_object`` does a shape),
and the company package ships it under its own name with its credit (``website_library_notes``).
"""
from __future__ import annotations

import hashlib
import io
import math
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest

GENERATOR = "website_library_asset.v1"
SOURCE = "library"
#: The only hosts a library file is fetched from.
ALLOWED_HOSTS = frozenset({"fuel.gazebosim.org"})
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_THUMBNAIL_BYTES = 4 * 1024 * 1024
THUMBNAIL_SIDE = 256
TIMEOUT_SECONDS = 120.0
MODEL_FILE = "library_model.usda"
AS_IS, SCALED = "as_is", "scaled_to_video_estimate"
ASPECT_MISMATCH = "library_aspect_mismatch"
CLAIM = ("A library model of the same kind of object, sized from the video's estimated dimensions where it was scaled; "
         "never a measurement of the filmed object.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, *, sha256: str, max_bytes: int) -> None:
    """One pinned file through the outbound boundary: https, an allowed host, same-origin redirects only."""
    from .safe_outbound_http import OutboundHttpPolicy, download_file
    download_file(url, output_path=target, expected_sha256="sha256:" + sha256, max_bytes=max_bytes,
                  timeout_seconds=TIMEOUT_SECONDS, headers={"User-Agent": "blueprint-object-library/1"},
                  policy=OutboundHttpPolicy(allowed_hosts=ALLOWED_HOSTS, follow_same_origin_redirects=True,
                                            max_response_bytes=max_bytes))


def fetch_pinned(row: Mapping[str, Any], cache_root: Path, *, fetch: Callable[..., Any] | None = None,
                 max_bytes: int = MAX_FILE_BYTES) -> Path:
    """The file ``row`` names (``url``, ``sha256``, ``size_bytes``, ``suffix``), from the cache or fetched; raises
    ``library_host_not_allowed``, ``library_file_too_large`` or ``library_file_digest_mismatch``."""
    parsed = urllib.parse.urlparse(str(row.get("url") or ""))
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in ALLOWED_HOSTS:
        raise ValueError("library_host_not_allowed")
    size, sha256 = row.get("size_bytes"), str(row.get("sha256") or "")
    if (isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= max_bytes
            or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256)):
        raise ValueError("library_file_too_large" if isinstance(size, int) and size > max_bytes
                         else "library_file_record_invalid")
    target = Path(cache_root) / f"{sha256}{row.get('suffix') or ''}"
    if target.is_file() and not target.is_symlink() and _sha256(target) == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    (fetch or download)(row["url"], target, sha256=sha256, max_bytes=size)
    if not target.is_file() or target.is_symlink() or _sha256(target) != sha256:
        target.unlink(missing_ok=True)
        raise ValueError("library_file_digest_mismatch")
    return target


def fetch_files(chosen: Mapping[str, Any], cache_root: Path, *, fetch: Callable[..., Any] | None = None
                ) -> list[dict[str, Any]]:
    """Every file of the chosen model, fetched and checked: its role, path, sha256 and size."""
    return [{"role": row["role"], "path": str(fetch_pinned(row, cache_root, fetch=fetch)), "sha256": row["sha256"],
             "size_bytes": row["size_bytes"], "suffix": row.get("suffix") or ""}
            for row in chosen["files"]]


def thumbnail_jpeg(entry: Mapping[str, Any], cache_root: Path, *, fetch: Callable[..., Any] | None = None
                   ) -> bytes | None:
    """The entry's thumbnail as a small JPEG for the model to see, or None when it cannot be fetched or read: the
    candidate is then shown by its words alone."""
    try:
        path = fetch_pinned(entry["thumbnail"], cache_root, fetch=fetch, max_bytes=MAX_THUMBNAIL_BYTES)
        from PIL import Image
        with Image.open(path) as image:
            image.load()
            picture = image.convert("RGBA")
        flat = Image.new("RGB", picture.size, (255, 255, 255))
        flat.paste(picture, mask=picture.getchannel("A"))
        flat.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
        out = io.BytesIO()
        flat.save(out, format="JPEG", quality=85)
        return out.getvalue()
    except Exception:  # noqa: BLE001 - a missing picture never stops the choice
        return None


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """An OBJ's vertices and its faces as triangles (polygons fanned); texture and normal indices are dropped."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("rb") as stream:
        for line in stream:
            if line.startswith(b"v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith(b"f "):
                corners = [int(item.split(b"/")[0]) for item in line.split()[1:]]
                corners = [index - 1 if index > 0 else len(vertices) + index for index in corners]
                faces += [[corners[0], corners[i], corners[i + 1]] for i in range(1, len(corners) - 1)]
    points, triangles = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    if (points.ndim != 2 or not len(points) or triangles.ndim != 2 or not len(triangles) or not np.isfinite(points).all()
            or triangles.min() < 0 or triangles.max() >= len(points)):
        raise ValueError("library_mesh_unreadable")
    return points, triangles


#: How each model format becomes vertices (metres, Z up) and triangles.
READERS: dict[str, Callable[[Sequence[Mapping[str, Any]]], tuple[np.ndarray, np.ndarray]]] = {
    "obj": lambda files: read_obj(Path(next(row["path"] for row in files if row["role"] == "mesh")))}
#: What converting each format to USD is called in the record and the package's credit.
CONVERTED_FROM = {"obj": "OBJ"}


def fit(model_size_m: Sequence[float], estimated_m: Sequence[float], tolerance: float) -> dict[str, Any]:
    """How the model meets the object's estimated size: ``as_is`` when every side is within ``tolerance``, else
    ``scaled_to_video_estimate`` (its largest side made the estimate's, every side by the same factor) when that
    brings every side within it; either way a quarter turn about up when the footprint lies the other way. Refused
    (``library_aspect_mismatch``) otherwise, with the closest it came."""
    size = [float(v) for v in model_size_m]
    seen = [float(v) for v in estimated_m]
    if not (len(size) == len(seen) == 3 and all(math.isfinite(v) and v > 0 for v in (*size, *seen))):
        raise ValueError("library_size_invalid")
    tried = []
    for status in (AS_IS, SCALED):
        for turns in (0, 1):
            turned = size if turns == 0 else [size[1], size[0], size[2]]
            scale = 1.0 if status == AS_IS else max(seen) / max(turned)
            made = [v * scale for v in turned]
            errors = [abs(m - s) / s for m, s in zip(made, seen, strict=True)]
            tried.append({"status": status, "scale": round(scale, 6), "quarter_turns_about_up": turns,
                          "dimensions_m": [round(v, 6) for v in made], "relative_errors": [round(v, 6) for v in errors]})
        passing = [row for row in tried[-2:] if max(row["relative_errors"]) <= tolerance + 1e-9]
        if passing:
            return {**min(passing, key=lambda row: max(row["relative_errors"])), "tolerance": tolerance,
                    "model_size_m": size, "estimated_size_m": seen}
    best = min(tried, key=lambda row: max(row["relative_errors"]))
    return {"status": ASPECT_MISMATCH, "closest": best, "tolerance": tolerance, "model_size_m": size,
            "estimated_size_m": seen}


def _write_mesh(path: Path, points: np.ndarray, triangles: np.ndarray, *, prims: Sequence[str], mesh: str) -> None:
    """One mesh under ``prims`` (Xforms, the first the default prim), in metres with Z up."""
    from pxr import Usd, UsdGeom
    path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for prim_path in prims:
        UsdGeom.Xform.Define(stage, prim_path)
    stage.SetDefaultPrim(stage.GetPrimAtPath(prims[0]))
    prim = UsdGeom.Mesh.Define(stage, f"{prims[-1]}/{mesh}")
    prim.CreatePointsAttr(points.astype(np.float32).tolist())
    prim.CreateFaceVertexCountsAttr([3] * len(triangles))
    prim.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()


def library_design(subject: Mapping[str, Any], *, estimated_m: Sequence[float], tolerance: float,
                   output_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The library model sized to the object, with its model file written once by content, or None and why: the
    files changed (``library_file_changed``), the mesh unreadable, or the proportions off (``library_aspect_mismatch``
    with the ratios). Never a measurement."""
    attribution, files = subject.get("attribution") or {}, subject.get("files") or []
    try:
        for row in files:
            if _sha256(Path(row["path"])) != row["sha256"]:
                return None, "library_file_changed"
        points, triangles = READERS[attribution["format"]](files)
    except (OSError, KeyError, StopIteration, ValueError, IndexError, TypeError):
        return None, "library_mesh_unreadable"
    low, high = points.min(axis=0), points.max(axis=0)
    sizing = fit((high - low).tolist(), estimated_m, tolerance)
    if sizing["status"] == ASPECT_MISMATCH:
        ratios = [round(v, 3) for v in sizing["closest"]["relative_errors"]]
        return None, f"{ASPECT_MISMATCH}:{ratios}"
    key = canonical_digest({"generator": GENERATOR, "files": [row["sha256"] for row in files],
                            "scale": sizing["scale"]})[7:23]
    path = Path(output_root) / key / MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    # The model as converted and scaled about its own origin: its own placement and orientation are kept.
    _write_mesh(path, points * sizing["scale"], triangles, prims=["/LibraryModel"], mesh="Mesh")
    centre = ((low + high) / 2 * sizing["scale"]).tolist()
    design = {"source": SOURCE, "generator": GENERATOR, "entry_id": attribution["entry_id"],
              "attribution": dict(attribution), "pick": dict(subject.get("pick") or {}),
              "converted_from": CONVERTED_FROM[attribution["format"]],
              "sizing": sizing, "model_centre_m": [round(v, 6) for v in centre],
              "model_file": {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size},
              "source_files": [{key: row[key] for key in ("role", "sha256", "size_bytes")} for row in files],
              "face_count": int(len(triangles)), "authoring_cost_usd": 0.0, "model_called_for_geometry": False,
              "physical_measurement": False, "claim": CLAIM}
    return design, None


def write_candidate(design: Mapping[str, Any], *, center_xyz_m: Sequence[float], path: Path) -> dict[str, Any]:
    """The sized model as construction's source candidate, centred where the object stood (Z up, metres) and turned
    as ``fit`` said; its model file is checked first."""
    from pxr import Usd, UsdGeom
    model = design["model_file"]
    if _sha256(Path(model["path"])) != model["sha256"]:
        raise ValueError("library_model_file_changed")
    source = Usd.Stage.Open(str(model["path"]))
    mesh = UsdGeom.Mesh(source.GetPrimAtPath("/LibraryModel/Mesh"))
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float) - np.asarray(design["model_centre_m"], dtype=float)
    if design["sizing"]["quarter_turns_about_up"] % 2:
        points = np.stack([-points[:, 1], points[:, 0], points[:, 2]], axis=1)
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    stage_path = Path(path)
    _write_mesh(stage_path, points + np.asarray(center_xyz_m, dtype=float), triangles,
                prims=["/Root", "/Root/SourceObjectCandidate"], mesh="Library")
    return {"source": SOURCE, "generator": GENERATOR, "design": dict(design), "face_count": int(len(triangles)),
            "authoring_cost_usd": 0.0}


__all__ = ["ALLOWED_HOSTS", "AS_IS", "ASPECT_MISMATCH", "CLAIM", "MODEL_FILE", "SCALED", "SOURCE", "fetch_files",
           "fetch_pinned", "fit", "library_design", "read_obj", "thumbnail_jpeg", "write_candidate"]
