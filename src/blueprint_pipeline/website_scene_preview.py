"""A web-sized, private preview of a website job's scene, for the site's own job page.

Owner ask 2026-11-07: a site sees the 3D scene Blueprint built from its video on its
job page. Scene preparation already holds a viewable scene for many jobs and published
nothing a browser could open: the Atlas base scene's splat
(``pipeline/website_scene_preparation/base_scene.json``) or, before preparation reaches
it, the capture's finished Atlas reconstruction (``pipeline/atlas``). This turns
whichever exists into a down-sampled splat the website's vendored viewer reads, and
publishes it over the channels website previews already use:

- the bytes go to the private artifact store under a key named for the capture, as the
  Atlas preview bundle does, so withdrawal deletion finds them after workspace retirement;
- the WebApp is told the preview's identity through the signed ``scene-preview`` control
  operation (``website_task_context.website_webapp_request``, the transport that carries
  ``atlas-preview``), and the WebApp fetches the bytes through the signed
  ``/api/live-pipeline/website-scene-previews`` read; nothing here is public;
- ``scene_preview.json`` beside the base scene keys the preview by the scene revision, so
  a pass that finds the same revision builds, uploads and sends nothing.

Privacy and rights are hard stops: nothing is built, archived or delivered unless the
capture is unwithdrawn and the WebApp's current task context says derived scene
generation is allowed. A withdrawal that lands while publishing takes the archived copy
back, and so does a WebApp refusal for rights. It costs nothing: it reads retained files,
calls no model or reconstruction provider and buys nothing.

The preview is for looking at, never a measurement: it keeps the most visible Gaussians
(opacity times projected size), band-0 colour only, in the source's own coordinates.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .website_atlas_preview_store import capture_name_matches_request, capture_withdrawn

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "website_scene_preview_manifest.v1"
PREVIEW_SCHEMA = "website_scene_preview.v1"
ACCEPTANCE_SCHEMA = "website_scene_preview_acceptance.v1"
#: Beside the base scene, whichever revision workspace the scene was built in.
DIRECTORY = ("pipeline", "website_scene_preparation")
MANIFEST_FILE = "scene_preview.json"
PREVIEW_FILE = "scene_preview.ply"
#: Withdrawal deletion holds this lock, so no pass publishes between its listing and deletion.
LOCK_FILE = "scene_preview.lock"
ARTIFACT_KIND = "website-scene-preview"
#: A standard binary 3DGS PLY: position, band-0 colour, opacity, scale, rotation (14 floats).
FORMAT = "ply_3dgs_sh0"
RECORD_BYTES = 14 * 4
#: Atlas worlds and their levelled copies are right-up-back with +Y up; the viewer route
#: turns exactly this frame for the vendored viewer.
COORDINATE_FRAME = "right_up_back_y_up"
BUILDER_REVISION = "website_scene_preview_builder.v1"
#: The size cap: a phone opens 32 MiB in seconds; a 2 M Gaussian Atlas splat is ~120 MiB.
MAX_PREVIEW_BYTES = 32 * 1024 * 1024
_HEADER_ALLOWANCE = 1024
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SPLAT_MEMBER = re.compile(r"splat/[^/]+\.ply")
#: A WebApp or store refusal that means the site's rights no longer allow the preview.
_RIGHTS_REFUSALS = re.compile(r"(source_revoked|consent_revoked|consent_expired|website_capture_withdrawn"
                              r"|capture_revoked_or_expired)\b")


class ScenePreviewUnavailable(ValueError):
    """No archived scene preview this caller may read; never says why."""


class ScenePreviewBusy(RuntimeError):
    """Another pass (or withdrawal deletion) holds this capture's preview lock."""


def preview_filename(capture_id: str, request_id: str) -> str:
    """The archive name: the walkthrough's own, or a continuation scoped to its job."""
    if not capture_name_matches_request(request_id, capture_id):
        raise ScenePreviewUnavailable("website_scene_preview_unavailable")
    return f"{capture_id}.ply" if capture_id.startswith("walkthrough-") else f"{request_id}-{capture_id}.ply"


def _read(path: Path) -> dict[str, Any]:
    try:
        value = {} if path.is_symlink() else json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _inside(value: Any, root: Path) -> Path | None:
    """A regular file inside ``root`` (the capture), or None."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file():
        return None
    resolved = path.resolve()
    return resolved if resolved.is_relative_to(root.resolve()) else None


def _sha256_file(path: Path) -> str:
    from .local_reconstruction_adapters import _sha256_file as sha
    return sha(path)


def _identity(capture_root: Path) -> dict[str, str]:
    root = Path(capture_root)
    scene_id, capture_id = root.parent.parent.name, root.name
    request_id = scene_id.removeprefix("site-")
    if (root.parent.name != "captures" or not scene_id.startswith("site-")
            or not capture_name_matches_request(request_id, capture_id)):
        raise ValueError("website_scene_preview_capture_identity_invalid")
    return {"request_id": request_id, "scene_id": scene_id, "capture_id": capture_id}


def _base_scene_source(capture_root: Path) -> dict[str, Any] | None:
    from .website_preparation_revision import workspace
    root = Path(capture_root).joinpath(*DIRECTORY)
    candidates = [workspace(capture_root) / "base_scene.json", root / "base_scene.json"]
    for path in dict.fromkeys(candidates):
        base = _read(path)
        if not base:
            continue
        # The levelled scene first (its floor is level), else the provider splat it was levelled from.
        levelling = base.get("levelling") if isinstance(base.get("levelling"), Mapping) else {}
        for splat, digest in ((base.get("splat_path"), base.get("splat_digest")),
                              (levelling.get("provider_splat_path"), levelling.get("provider_splat_digest"))):
            found = _inside(splat, capture_root)
            if found is not None and isinstance(digest, str) and _DIGEST.fullmatch(digest):
                return {"kind": "atlas_base_scene", "path": found, "digest": digest,
                        "record": str(path.relative_to(capture_root)), "record_digest": canonical_digest(base)}
    return None


def _reconstruction_source(capture_root: Path) -> dict[str, Any] | None:
    """The capture's finished Atlas reconstruction: its sealed result and download bundle's splat."""
    job_dir = Path(capture_root) / "pipeline" / "atlas"
    job, result = _read(job_dir / "preview_job.json"), _read(job_dir / "result.json")
    if (job.get("state") != "ready" or not result.get("digest")
            or result["digest"] != canonical_digest(result, digest_field="digest")
            or result["digest"] != job.get("result_digest")):
        return None
    download = result.get("download") or ((result.get("automation") or {}).get("background") or {}).get("download")
    folder = (download or {}).get("directory") if isinstance(download, Mapping) else None
    directory = _inside(str(Path(folder) / "manifest.json"), capture_root) if isinstance(folder, str) and folder else None
    bundle = _read(directory) if directory is not None else {}
    if (bundle.get("schema_version") != "atlas_download_bundle.v1" or bundle.get("splat_available") is not True
            or bundle.get("digest") != canonical_digest(bundle, digest_field="digest")
            or bundle["digest"] != download.get("manifest_digest")):
        return None
    splats = [row for row in bundle.get("assets") or [] if isinstance(row, Mapping)
              and _SPLAT_MEMBER.fullmatch(str(row.get("path") or ""))]
    if len(splats) != 1 or not _DIGEST.fullmatch(str(splats[0].get("sha256") or "")):
        return None
    found = _inside(str(directory.parent / splats[0]["path"]), capture_root)
    if found is None:
        return None
    return {"kind": "atlas_reconstruction", "path": found, "digest": splats[0]["sha256"],
            "record": str((job_dir / "result.json").relative_to(capture_root)),
            "record_digest": canonical_digest({"result": result["digest"], "bundle": bundle["digest"]})}


def scene_source(capture_root: Path) -> dict[str, Any] | None:
    """The capture's viewable scene: the base scene's splat, else its finished reconstruction."""
    return _base_scene_source(Path(capture_root)) or _reconstruction_source(Path(capture_root))


def scene_revision(source: Mapping[str, Any], *, max_bytes: int = MAX_PREVIEW_BYTES) -> str:
    return canonical_digest({"builder": BUILDER_REVISION, "max_bytes": int(max_bytes), "kind": source["kind"],
                             "splat_digest": source["digest"], "record_digest": source["record_digest"]})


def _start_view(xyz: Any) -> dict[str, list[float]] | None:
    """Stand at the scene's middle and look along its longer horizontal side."""
    import numpy as np
    if len(xyz) == 0:
        return None
    centre = np.median(xyz, axis=0)
    low, high = np.percentile(xyz[:, [0, 2]], 10, axis=0), np.percentile(xyz[:, [0, 2]], 90, axis=0)
    extent = high - low
    if not np.all(np.isfinite(centre)) or not np.all(np.isfinite(extent)) or float(extent.max()) <= 0:
        return None
    target = centre.copy()
    target[0 if extent[0] >= extent[1] else 2] += 0.25 * float(extent.max())
    return {"position": [round(float(value), 4) for value in centre],
            "target": [round(float(value), 4) for value in target]}


def build_preview(source: Path, output: Path, *, max_bytes: int = MAX_PREVIEW_BYTES) -> dict[str, Any]:
    """Write the most visible Gaussians of ``source`` that fit ``max_bytes``, deterministically."""
    import numpy as np

    from .gaussian_splat_decode import SplatData, read_standard_3dgs_ply, write_standard_3dgs_ply
    try:
        splat = read_standard_3dgs_ply(source)
    except ValueError:
        raise ValueError("website_scene_preview_source_not_3dgs_ply") from None
    capacity = (int(max_bytes) - _HEADER_ALLOWANCE) // RECORD_BYTES
    columns = np.concatenate([splat.xyz, splat.f_dc, splat.opacity[:, None], splat.scales, splat.quats], axis=1)
    finite = np.isfinite(columns).all(axis=1)
    if splat.count == 0 or capacity <= 0 or not finite.any():
        raise ValueError("website_scene_preview_source_empty")
    # Opacity times projected area: (s0 s1 s2)^(2/3), from log scales.
    area = np.exp(np.clip(splat.scales.astype(np.float64).sum(axis=1) * (2.0 / 3.0), -80.0, 80.0))
    importance = np.where(finite, splat.opacity_sigmoid.astype(np.float64) * area, -1.0)
    keep = min(capacity, int(finite.sum()))
    order = np.lexsort((np.arange(splat.count), -importance))
    kept = np.sort(order[:keep])
    subset = SplatData(count=keep, xyz=splat.xyz[kept], opacity=splat.opacity[kept], f_dc=splat.f_dc[kept],
                       scales=splat.scales[kept], quats=splat.quats[kept], properties=(), sh_rest=None)
    partial = output.with_name(output.name + ".partial")
    write_standard_3dgs_ply(subset, partial)
    size = partial.stat().st_size
    if size > max_bytes:
        partial.unlink()
        raise ValueError("website_scene_preview_over_size_cap")
    os.replace(partial, output)
    return {"format": FORMAT, "coordinate_frame": COORDINATE_FRAME, "digest": _sha256_file(output),
            "size_bytes": size, "gaussian_count": keep, "source_gaussian_count": int(splat.count),
            "max_bytes": int(max_bytes), "start_view": _start_view(subset.xyz.astype(np.float64))}


def preview_record(*, identity: Mapping[str, str], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What the WebApp may store: the preview's identity, never a path, object key or person."""
    built = manifest["preview"]
    value = {"schema_version": PREVIEW_SCHEMA, "capture_id": identity["capture_id"],
             "scene_revision": manifest["scene_revision"], "state": "ready",
             "source_kind": manifest["source"]["kind"], "format": built["format"],
             "coordinate_frame": built["coordinate_frame"], "digest": built["digest"],
             "size_bytes": built["size_bytes"], "gaussian_count": built["gaussian_count"],
             "source_gaussian_count": built["source_gaussian_count"], "start_view": built["start_view"],
             "claim_ceiling": "development_only", "updated_at_epoch": int(manifest["built_at_epoch"])}
    value["preview_digest"] = cross_runtime_canonical_digest(value)
    return value


@contextmanager
def preview_lock(capture_root: Path) -> Iterator[None]:
    directory = Path(capture_root).joinpath(*DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("website_scene_preview_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ScenePreviewBusy("website_scene_preview_busy") from None
        yield


def _current_rights(identity: Mapping[str, str]) -> None:
    """The WebApp's current task context must allow derived scene generation; else refuse."""
    from .website_task_context import load_current_website_task_context
    context = load_current_website_task_context(request_id=identity["request_id"], scene_id=identity["scene_id"],
                                                capture_id=identity["capture_id"], purpose="scene_preparation")
    rights = context.get("capture_rights") or {}
    if (rights.get("derived_scene_generation_allowed") is not True or rights.get("consent_revoked") is not False
            or rights.get("consent_status") != "granted"):
        raise ValueError("website_scene_preview_rights_refused")


def _archive(path: Path, *, identity: Mapping[str, str]) -> dict[str, Any]:
    """The private artifact store only, never the legacy fallback, with its bytes proven."""
    from .task_evaluation_configured_scene_object_store import (
        publish_configured_scene_artifact, verify_dedicated_artifact_store)
    verify_dedicated_artifact_store()
    filename = preview_filename(identity["capture_id"], identity["request_id"])
    reference = publish_configured_scene_artifact(path=path, artifact_kind=ARTIFACT_KIND, filename=filename)
    return {"uri": reference["uri"], "digest": reference["digest"], "size_bytes": reference["size_bytes"],
            "filename": filename}


def _retract(uri: str) -> None:
    """Delete every version of an archived preview, and prove none is left."""
    from .website_capture_withdrawal_deletion import B2ObjectVersions
    store = B2ObjectVersions.configured()
    for key, version in store.versions(uri):
        store.delete(key, version)
    if store.versions(uri):
        raise ValueError("website_scene_preview_archive_versions_remaining")


def _deliver(record: Mapping[str, Any], *, identity: Mapping[str, str]) -> None:
    from .website_task_context import website_webapp_request
    acceptance = website_webapp_request(capture_id=identity["capture_id"], operation="scene-preview", payload={
        "request_id": identity["request_id"], "scene_id": identity["scene_id"], "preview": dict(record)})
    if (acceptance.get("schema_version") != ACCEPTANCE_SCHEMA or acceptance.get("accepted") is not True
            or acceptance.get("scene_revision") != record["scene_revision"]
            or acceptance.get("preview_digest") != record["preview_digest"]):
        raise ValueError("website_scene_preview_acceptance_invalid")


def _reason(error: BaseException) -> str:
    """A typed code as recorded; anything else (it may carry a path or URL) by its type name."""
    text = str(error)
    return text if re.fullmatch(r"[a-z][a-z0-9_.:-]{0,159}", text) else type(error).__name__


def _current(manifest: Mapping[str, Any], revision: str, preview: Path) -> bool:
    built, archive = manifest.get("preview") or {}, manifest.get("archive") or {}
    return (manifest.get("schema_version") == MANIFEST_SCHEMA and manifest.get("scene_revision") == revision
            and not preview.is_symlink() and preview.is_file() and preview.stat().st_size == built.get("size_bytes")
            and archive.get("digest") == built.get("digest") and archive.get("retracted") is not True
            and (manifest.get("delivered") or {}).get("preview_digest") == manifest.get("preview_digest"))


def publish_scene_preview(capture_root: Path, *, now: Callable[[], float] = time.time,
                          check_rights: Callable[[Mapping[str, str]], None] = _current_rights,
                          archive: Callable[..., dict[str, Any]] = _archive,
                          retract: Callable[[str], None] = _retract,
                          deliver: Callable[..., None] = _deliver,
                          max_bytes: int = MAX_PREVIEW_BYTES) -> dict[str, Any]:
    """One idempotent pass for one capture: build, archive and deliver what is not yet current.

    Returns ``{"status": ...}``: ``current``, ``published``, ``no_viewable_scene``,
    ``withdrawn``, ``refused`` (rights, with a reason), ``busy`` or ``failed`` (retried by
    the next pass). Typed refusals never raise; nothing here pays for anything.
    """
    root = Path(capture_root)
    identity = _identity(root)
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    # Read-only first: a capture with nothing to show gets no lock file or directory.
    if scene_source(root) is None:
        return {"status": "no_viewable_scene"}
    directory = root.joinpath(*DIRECTORY)
    manifest_path, preview_path = directory / MANIFEST_FILE, directory / PREVIEW_FILE
    try:
        with preview_lock(root):
            source = scene_source(root)
            if source is None:
                return {"status": "no_viewable_scene"}
            revision = scene_revision(source, max_bytes=max_bytes)
            manifest = _read(manifest_path)
            if _current(manifest, revision, preview_path):
                return {"status": "current", "scene_revision": revision}
            try:
                check_rights(identity)
            except Exception as error:  # noqa: BLE001 - any doubt about rights is a refusal
                return {"status": "refused", "reason": _reason(error)}
            built = manifest.get("preview") or {}
            if (manifest.get("scene_revision") != revision or preview_path.is_symlink() or not preview_path.is_file()
                    or _sha256_file(preview_path) != built.get("digest")):
                if _sha256_file(Path(source["path"])) != source["digest"]:
                    return {"status": "failed", "reason": "website_scene_preview_source_changed"}
                built = build_preview(Path(source["path"]), preview_path, max_bytes=max_bytes)
                manifest = {"schema_version": MANIFEST_SCHEMA, **identity, "scene_revision": revision,
                            "source": {key: source[key] for key in ("kind", "digest", "record", "record_digest")}
                            | {"path": str(Path(source["path"]).relative_to(root.resolve()))},
                            "preview": {**built, "path": PREVIEW_FILE}, "built_at_epoch": int(now()),
                            "archive": None, "delivered": None, "claim_ceiling": "development_only"}
            record = preview_record(identity=identity, manifest=manifest)
            manifest["preview_digest"] = record["preview_digest"]
            write_json(manifest_path, manifest)
            if capture_withdrawn(root):
                return {"status": "withdrawn"}
            stored = manifest.get("archive") or {}
            if stored.get("digest") != built["digest"] or stored.get("retracted") is True:
                manifest["archive"] = archive(preview_path, identity=identity)
                if manifest["archive"]["digest"] != built["digest"]:
                    raise ValueError("website_scene_preview_archive_mismatch")
                write_json(manifest_path, manifest)
            # A withdrawal that landed during the upload takes the copy back at once.
            if capture_withdrawn(root):
                retract(manifest["archive"]["uri"])
                manifest["archive"] = {**manifest["archive"], "retracted": True}
                write_json(manifest_path, manifest)
                return {"status": "withdrawn"}
            try:
                deliver(record, identity=identity)
            except Exception as error:  # noqa: BLE001 - classified below; the archive stands only with rights
                reason = _reason(error)
                if _RIGHTS_REFUSALS.search(str(error)):
                    retract(manifest["archive"]["uri"])
                    manifest["archive"] = {**manifest["archive"], "retracted": True}
                    write_json(manifest_path, manifest)
                    return {"status": "refused", "reason": reason}
                manifest["last_delivery_error"] = reason
                write_json(manifest_path, manifest)
                return {"status": "failed", "reason": reason}
            manifest["delivered"] = {"preview_digest": record["preview_digest"], "at_epoch": int(now())}
            manifest.pop("last_delivery_error", None)
            write_json(manifest_path, manifest)
            return {"status": "published", "scene_revision": revision, "preview_digest": record["preview_digest"]}
    except ScenePreviewBusy:
        return {"status": "busy"}
    except Exception as error:  # noqa: BLE001 - one capture never stops a pass; retried next time
        logger.warning("website_scene_preview_failed capture=%s type=%s", identity["capture_id"],
                       type(error).__name__)
        return {"status": "failed", "reason": _reason(error)}


def capture_roots(storage_root: Path) -> list[Path]:
    """Every website capture holding a base scene or an Atlas reconstruction."""
    found: set[Path] = set()
    for pattern in ("*/scenes/site-*/captures/*/pipeline/website_scene_preparation/base_scene.json",
                    "*/scenes/site-*/captures/*/pipeline/website_scene_preparation/revisions/*/base_scene.json",
                    "*/scenes/site-*/captures/*/pipeline/atlas/result.json"):
        for path in Path(storage_root).glob(pattern):
            capture = next(parent for parent in path.parents if parent.parent.name == "captures")
            if capture_name_matches_request(capture.parent.parent.name.removeprefix("site-"), capture.name):
                found.add(capture)
    return sorted(found)


def tick(storage_root: Path, **options: Any) -> dict[str, int]:
    """One pass over every capture; counts of each outcome."""
    counts: dict[str, int] = {}
    for capture in capture_roots(storage_root):
        try:
            status = publish_scene_preview(capture, **options)["status"]
        except Exception as error:  # noqa: BLE001 - an unreadable capture never stops the others
            logger.warning("website_scene_preview_capture_unreadable type=%s", type(error).__name__)
            status = "unreadable"
        counts[status] = counts.get(status, 0) + 1
    return counts


def open_scene_preview(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
                       client: Any = None, bucket: str | None = None) -> tuple[Any, int]:
    """The archived preview body and size, for a WebApp read it already authorized.

    While the workspace exists its manifest must name this digest as archived and the
    capture must be unwithdrawn; after retirement only the object named for this
    capture and digest exists to read.
    """
    if not capture_name_matches_request(request_id, capture_id) or not _DIGEST.fullmatch(str(digest)):
        raise ScenePreviewUnavailable("website_scene_preview_unavailable")
    if storage_root is not None:
        roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
                 if root.is_dir()]
        if len(roots) > 1:
            raise ScenePreviewUnavailable("website_scene_preview_unavailable")
        if roots:
            manifest = _read(roots[0].joinpath(*DIRECTORY) / MANIFEST_FILE)
            archive = manifest.get("archive") or {}
            if (capture_withdrawn(roots[0]) or manifest.get("schema_version") != MANIFEST_SCHEMA
                    or archive.get("digest") != digest or archive.get("retracted") is True):
                raise ScenePreviewUnavailable("website_scene_preview_unavailable")
    from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    from .task_evaluation_configured_scene_object_store import _artifact_object_store_client, _object_missing
    if client is None or bucket is None:
        client, bucket = _artifact_object_store_client()
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/{ARTIFACT_KIND}/sha256/{digest[7:]}/{preview_filename(capture_id, request_id)}"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(error):
            raise ScenePreviewUnavailable("website_scene_preview_unavailable") from None
        raise
    if (head.get("Metadata") or {}).get("sha256") != digest[7:]:
        raise ScenePreviewUnavailable("website_scene_preview_unavailable")
    return client.get_object(Bucket=bucket, Key=key)["Body"], int(head["ContentLength"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("publish", help="publish one capture's scene preview if it is not current")
    one.add_argument("--capture-root", type=Path, required=True)
    every = commands.add_parser("tick", help="publish every capture's scene preview that is not current")
    every.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = publish_scene_preview(args.capture_root) if args.command == "publish" else tick(args.storage_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
