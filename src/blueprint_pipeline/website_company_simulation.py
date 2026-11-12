"""The simulation package a deployment company downloads: one zip per job, made when the job's simulation exists.

Owner decision 2026-11-12: the simulation is core, because a deployment company cannot say "can do" without
testing on the twin; it downloads the simulation first and tests in its own tools (connecting its software to
Blueprint's simulation comes later and is not built here). This makes the package and publishes it exactly as the
company clip is published (``website_company_clip``):

- the source is the job's compiled team scene half, the simulation the trial itself runs: the team scene packet
  queue's ``status/<key>.json`` (``completed``, ``outcome: compiled``) names the half, whose
  ``scene/native-task-packet/`` holds ``native_task_arena_scene_plan.v1.json`` and the staged USD files under
  ``assets/``: the scene's look (``scene_appearance``), its collision (``scene_collision``) and the job's object
  (``task_object``) with its start pose, and the plan's ``task_spec.task_success_contract.criteria``;
- the package holds ``scene.usda``, written by ``scene_package_stage`` from the plan: every part referenced by a
  relative path and placed at the plan's pose under a neutral prim name (``/World/Room``, ``/World/Object``, ...),
  in metres with the plan's up axis; the parts (``scene/room.*``, ``scene/room_collision.*``, ``object.*``), what
  each needs (renamed neutrally under ``scene/files/``), ``success_rule.json`` (the criteria and run facts, neutral
  only) and ``README.txt``. No record, manifest, ledger or receipt goes in, the plan included; no file name carries
  an id. Every outside or absolute USD reference is rewritten to a path inside the package, or the package is
  refused; a plan row that cannot be placed refuses it, naming the row's role;
- every file is read with the shared outbound scan (``outbound_file_scan``: ids in both forms, account ids,
  keys, host paths, storage, file and web links, bare web addresses, and the text form of every binary USD
  layer) before the zip is made; a finding refuses the package (``refused``), naming the file and the kind;
- the zip is deterministic (names in order, fixed times and modes), so the same inputs give the same digest; it
  goes to the private artifact store by digest under a name for the capture (withdrawal deletion finds every
  version by that name), the WebApp is told its identity through the signed ``company-simulation`` control
  operation and reads it through the signed ``/api/live-pipeline/website-company-simulations`` route.

Rights are the company clip's: not withdrawn, the delivery-time takedown gate allows it, the WebApp's live task
context grants consent, and ``sharing_statement.sharing_basis`` gives a basis. The not-permitted live read is
shared with the clip (its ``statement_check.json``), so a capture that may not be shared is read once a window,
not once per package.

It runs in the Atlas preview timer's pass, after the clip (``website_atlas_preview.company_simulation_tick``):
local files and ``pxr`` only, no model, provider or paid resource. Nothing here raises out of the pass or holds a
job: every outcome is a returned status, retried on the next pass.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import stat
import time
import zipfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from . import website_company_clip as clip
from .common import write_json
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .outbound_file_scan import capture_site_words
from .website_atlas_preview_store import capture_name_matches_request, capture_withdrawn

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "website_company_simulation_manifest.v1"
PACKAGE_SCHEMA = "website_company_simulation.v1"
ACCEPTANCE_SCHEMA = "website_company_simulation_acceptance.v1"
DIRECTORY = ("pipeline", "company_simulation")
MANIFEST_FILE = "company_simulation.json"
PACKAGE_FILE = "simulation.zip"
STAGING = ".staging"
#: Withdrawal deletion holds this lock, so no pass publishes between its listing and deletion.
LOCK_FILE = "company_simulation.lock"
ARTIFACT_KIND = "website-company-simulation"
FORMAT = "zip_usd_scene_object_success_rule"
CONTENT_TYPE = "application/zip"
#: The one real compiled half on the host (capture-447, 2026-11-12) stages 211 MiB of USD (appearance 124 MiB,
#: collision 83 MiB, object 3.8 MiB); its configured bundle's text collision alone is 278 MiB. 512 MiB is about
#: 2.4 times the real scene, the size the first-site team runbook already allows for one flattened USD, and a
#: quarter of what the website accepts (2 GiB). Measured on the source files and on the zip.
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
BUILDER_REVISION = "website_company_simulation_builder.v1"
OVER_CAP = "website_company_simulation_over_size_cap"
QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
DEFAULT_QUEUE_ROOT = "/var/lib/blueprint/pipeline-control-plane/team-scene-packets"
PACKET_DIRECTORIES = ("native-task-packet", "native-task-packet-three-view")
PLAN_FILE = "native_task_arena_scene_plan.v1.json"
_USD = frozenset({".usd", ".usda", ".usdc", ".usdz"})
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_RIGHTS_REFUSALS = re.compile(r"(source_revoked|consent_revoked|consent_expired|website_capture_withdrawn"
                              r"|capture_revoked_or_expired|website_company_simulation_not_permitted"
                              r"|website_company_clip_not_permitted)\b")
#: Record fields whose string values are account or owner ids, and those that are keys.
_ACCOUNT_KEYS = frozenset({"accepted_by", "account_id", "user_id", "uid", "owner_id", "owner_uid", "created_by",
                           "submitted_by", "approved_by", "requested_by", "buyer_account_id", "organization_id"})
_KEY_KEYS = frozenset({"delivery_key", "producer_delivery_key", "upload_token", "token", "session_token"})


class CompanySimulationUnavailable(ValueError):
    """No archived package this caller may read; never says why."""


class CompanySimulationBusy(RuntimeError):
    """Another pass (or withdrawal deletion) holds this capture's package lock."""


class PackageRefused(ValueError):
    """The package may not leave: a scan finding, an outside reference or an unusable scene file."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def package_filename(capture_id: str, request_id: str) -> str:
    """The archive name: the walkthrough's own, or a continuation scoped to its job."""
    if not capture_name_matches_request(request_id, capture_id):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    return f"{capture_id}.zip" if capture_id.startswith("walkthrough-") else f"{request_id}-{capture_id}.zip"


def _read(path: Path) -> dict[str, Any]:
    return clip._read(Path(path))


def _sha256_file(path: Path) -> str:
    return clip._sha256_file(Path(path))


# --- where the job's simulation is -------------------------------------------------------------------------------

def _queue_root() -> Path:
    return Path(os.environ.get(QUEUE_ROOT_ENV) or DEFAULT_QUEUE_ROOT)


def _inside(base: Path, path: Path) -> Path | None:
    """``path`` resolved, when it is a regular file under ``base`` reached through no link."""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(base)) \
            or not resolved.is_relative_to(base.resolve()) or not resolved.is_file():
        return None
    return resolved


def simulation_source(capture_root: Path, *, queue_root: Path | None = None) -> dict[str, Any] | None:
    """The newest compiled team scene half of this capture, as its plan and staged files name it, or None.

    None while no half is compiled (the job is never held for it). A half whose plan or files do not read is
    None too, and is tried again on the next pass.
    """
    root, queue = Path(capture_root), Path(queue_root or _queue_root())
    identity = clip._identity(root)
    statuses = []
    try:
        candidates = sorted((queue / "status").glob("*.json"))
    except OSError:
        return None
    for path in candidates:
        status = _read(path)
        if (status.get("status") == "completed" and status.get("outcome") == "compiled"
                and status.get("capture_id") == identity["capture_id"]
                and Path(str(status.get("capture_root") or "")) == root
                and isinstance(status.get("team_scene_inputs_path"), str)):
            statuses.append((path.stat().st_mtime_ns, path.name, status))
    for _mtime, _name, status in sorted(statuses, reverse=True):
        source = _half(Path(status["team_scene_inputs_path"]).parent, status, estimates=object_estimates(root))
        if source is not None:
            return source
    return None


def object_estimates(capture_root: Path) -> Mapping[str, Any] | None:
    """The object's mass and friction ranges as preparation estimated them from the video, when recorded.

    Read from the preparation the scene handoff names (inside the capture), else the default preparation record.
    """
    root = Path(capture_root)
    preparation = root / "pipeline" / "website_scene_preparation" / "preparation.json"
    named = _read(root / "pipeline" / "website_scene_preparation" / "handoff.json").get("preparation_path")
    if isinstance(named, str):
        candidate = Path(named)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if _inside(root, candidate) is not None:
            preparation = candidate
    bounds = (_read(preparation).get("physics") or {}).get("bounds")
    return bounds if isinstance(bounds, Mapping) else None


def _half(scene: Path, status: Mapping[str, Any], *, estimates: Mapping[str, Any] | None = None
          ) -> dict[str, Any] | None:
    """The half's plan and shipped files; a row that cannot be placed is kept as ``refusal`` naming its role."""
    from .scene_package_stage import StageRefused, neutral_rule, placements, up_axis
    if not scene.is_absolute() or scene.is_symlink() or not scene.is_dir():
        return None
    for name in PACKET_DIRECTORIES:
        packet = scene / name
        plan_path = _inside(scene, packet / PLAN_FILE)
        if plan_path is None:
            continue
        plan = _read(plan_path)
        if not plan:
            continue
        source: dict[str, Any] = {"packet_dir": packet, "plan_path": plan_path, "plan": plan,
                                  "plan_digest": plan.get("plan_digest"), "status": dict(status), "files": {},
                                  "refusal": None}
        try:
            rows = placements(plan)
            axis = up_axis(plan)
            rule = neutral_rule(plan, estimates=estimates)
            if rule is None:
                raise StageRefused("scene_stage_success_rule_missing")
            for placement, raw in zip(rows, plan["objects"], strict=True):
                role = placement["role"]
                path = _inside(packet, packet.joinpath(*PurePosixPath(placement["usd_path"]).parts))
                if (path is None or path.stat().st_size != raw.get("size_bytes")
                        or not _DIGEST.fullmatch(str(raw.get("sha256") or ""))):
                    raise StageRefused("scene_stage_asset_not_shipped", role)
                source["files"][placement["prim"]] = {"path": path, "role": role, "sha256": raw["sha256"], "size_bytes": int(raw["size_bytes"]),
                                         "suffix": PurePosixPath(placement["usd_path"]).suffix.lower(),
                                         "placement": {key: value for key, value in placement.items()
                                                       if key != "usd_path"}}
            source.update(axis=axis, rule=rule, assets_dir=packet / str(plan.get("asset_directory") or "assets"),
                          appearance_representation=(plan.get("appearance_frame_alignment") or {}).get(
                              "representation"))
        except StageRefused as error:
            source["refusal"] = "website_company_simulation_" + str(error).removeprefix("scene_stage_")
        return source
    return None


def simulation_revision(source: Mapping[str, Any], *, max_bytes: int = MAX_PACKAGE_BYTES) -> str:
    """Names this package: the files' digests, their placements, the success rule, the builder and the cap."""
    return canonical_digest({
        "builder": BUILDER_REVISION, "format": FORMAT, "max_bytes": int(max_bytes),
        "files": {prim: {key: row[key] for key in ("sha256", "size_bytes", "suffix", "placement")}
                  for prim, row in sorted(source["files"].items())},
        "rule": source.get("rule"), "axis": source.get("axis"), "refusal": source.get("refusal"),
        "plan_digest": source.get("plan_digest") if source.get("refusal") else None})


# --- what must never leave --------------------------------------------------------------------------------------

def _walk_values(value: Any, keys: frozenset[str]) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys and isinstance(item, str):
                yield item
            else:
                yield from _walk_values(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item, keys)


def secret_literals(capture_root: Path, source: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The job's own ids, account ids and keys, each with the kind printed in a finding (never the value)."""
    root = Path(capture_root)
    identity = clip._identity(root)
    status = source.get("status") or {}
    records = [_read(root / name) for name in (
        "capture_descriptor.json", "raw/manifest.json", "raw/capture_upload_complete.json",
        "pipeline_staging_manifest.json", "pipeline_job_ledger.json", "pipeline/website_task_context.json")]
    records.append(_read(Path(source["plan_path"])))
    found = [(identity[key], "a job id") for key in ("request_id", "scene_id", "capture_id")]
    core = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", identity["request_id"])
    if core:
        found += [(core.group(0), "a job id"), (core.group(0).replace("-", ""), "a job id")]
    found += [(str(status.get(key) or ""), "a job id") for key in ("task_id", "request_key")]
    for record in records:
        found += [(value, "a job id") for value in _walk_values(record, frozenset({"site_submission_id", "intent_id",
                                                                                     "scene_intent_id", "site_id"}))]
        found += [(value, "an account id") for value in _walk_values(record, _ACCOUNT_KEYS)]
        found += [(value, "a delivery key") for value in _walk_values(record, _KEY_KEYS)]
    # Short words ("system", "operator") are roles, not ids: an id here is at least eight characters.
    return [(value, kind) for value, kind in found if len(value) >= 8 and not re.fullmatch(r"[a-z_]+", value)]


# --- making the package ----------------------------------------------------------------------------------------

def _open_layer(path: str) -> Any:
    """The layer, or None when ``pxr`` cannot open it (it raises for a broken file)."""
    from pxr import Sdf
    try:
        return Sdf.Layer.OpenAsAnonymous(path)
    except Exception:  # noqa: BLE001 - any failure to open is a refusal by the caller
        return None


def _asset_paths(layer: Any) -> list[str]:
    from pxr import UsdUtils
    found: list[str] = []

    def collect(path: str) -> str:
        found.append(path)
        return path
    UsdUtils.ModifyAssetPaths(layer, collect)
    return found


def _outside(path: str) -> bool:
    pure = PurePosixPath(path.replace("\\", "/"))
    return ("://" in path or path.startswith(("file:", "/", "\\")) or re.match(r"[A-Za-z]:", path) is not None
            or ".." in pure.parts)


def _check_usdz(path: Path, label: str) -> None:
    """A usdz is kept as it is: every reference inside it must stay inside it."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = [info.filename for info in archive.infolist() if Path(info.filename).suffix.lower() in _USD]
    except (OSError, zipfile.BadZipFile):
        raise PackageRefused("website_company_simulation_scene_unreadable", label) from None
    if not names:
        raise PackageRefused("website_company_simulation_scene_unreadable", label)
    for name in names:
        layer = _open_layer(f"{path}[{name}]")
        if layer is None:
            raise PackageRefused("website_company_simulation_scene_unreadable", label)
        for reference in [*_asset_paths(layer), *(value for _spec, value in _file_strings(layer))]:
            if reference and _outside(reference):
                raise PackageRefused("website_company_simulation_outside_reference", label)


#: A string or token value with one of these endings names a file: it is shipped and rewritten like an asset path.
FILE_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz", ".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga",
                           ".tif", ".tiff", ".bmp", ".ktx", ".ktx2", ".dds", ".mdl", ".obj", ".ply", ".glb", ".gltf",
                           ".vdb", ".nvdb", ".mtl", ".wav", ".mp4", ".mov"})
_STRING_TYPES = frozenset({"string", "token", "string[]", "token[]"})


def _names_file(value: Any) -> bool:
    return isinstance(value, str) and PurePosixPath(value.strip().replace("\\", "/")).suffix.lower() in FILE_SUFFIXES


def _file_strings(layer: Any) -> list[tuple[Any, str]]:
    """Every (attribute spec, value) whose string or token value names a file, in defaults and time samples."""
    from pxr import Sdf
    found: list[tuple[Any, str]] = []
    paths: list = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    for path in paths:
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or str(spec.typeName) not in _STRING_TYPES:
            continue
        values = [spec.default] if spec.HasDefaultValue() else []
        values += [layer.QueryTimeSample(path, sample) for sample in layer.ListTimeSamplesForPath(path)]
        for value in values:
            for item in (list(value) if not isinstance(value, str) and value is not None else [value]):
                if _names_file(item):
                    found.append((spec, str(item)))
    return found


def _rewrite_strings(layer: Any, rewrite: Callable[[str], str]) -> None:
    """Rewrite every string or token value that names a file, in defaults and time samples."""
    from pxr import Sdf
    paths: list = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)

    def fixed(value: Any) -> Any:
        if isinstance(value, str):
            return rewrite(value) if _names_file(value) else value
        if value is not None and not isinstance(value, str):
            items = list(value)
            new = [rewrite(item) if _names_file(item) else item for item in items]
            return new if new != items else value
        return value
    for path in paths:
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or str(spec.typeName) not in _STRING_TYPES:
            continue
        if spec.HasDefaultValue():
            new = fixed(spec.default)
            if new is not spec.default and new != spec.default:
                spec.default = new
        for sample in layer.ListTimeSamplesForPath(path):
            value = layer.QueryTimeSample(path, sample)
            new = fixed(value)
            if new is not value and new != value:
                layer.SetTimeSample(path, sample, new)


_UDIM = "<UDIM>"


class _Localizer:
    """Copies each USD file into the package, rewriting every reference to a relative path inside it.

    Asset paths (sublayers, references, payloads, asset-valued attributes) and string or token values that name a
    file are rewritten; sublayer time offsets are kept; every tile of a UDIM texture inside the scene's own folder
    is shipped. A reference that is a link, points outside the scene's files or names nothing refuses the package.
    """

    def __init__(self, assets_dir: Path, staging: Path):
        self.assets, self.staging = assets_dir.resolve(), staging
        self.placed: dict[Path, Path] = {}
        self.count = 0

    def _target(self, suffix: str) -> Path:
        self.count += 1
        return self.staging / "scene" / "files" / f"file-{self.count:03d}{suffix}"

    def place(self, source: Path, target: Path, label: str) -> None:
        source = source.resolve()
        self.placed[source] = target
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() == ".usdz":
            _check_usdz(source, label)
            shutil.copyfile(source, target)
            return
        if source.suffix.lower() not in _USD:
            shutil.copyfile(source, target)
            return
        layer = _open_layer(str(source))
        if layer is None:
            raise PackageRefused("website_company_simulation_scene_unreadable", label)
        changed: list[bool] = []
        failure: list[str] = []

        def inside(candidate: Path) -> Path | None:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                return None
            return resolved if resolved.is_relative_to(self.assets) and resolved.is_file() else None

        def rewrite(reference: str) -> str:
            if not reference or failure:
                return reference
            if "://" in reference or reference.startswith("file:"):
                failure.append("website_company_simulation_outside_reference")
                return reference
            base = Path(reference) if Path(reference).is_absolute() else source.parent / reference
            if _UDIM in reference:
                return udim(reference, base)
            try:
                base.resolve(strict=True)
            except OSError:
                failure.append("website_company_simulation_reference_missing")
                return reference
            resolved = inside(base)
            if resolved is None:
                failure.append("website_company_simulation_outside_reference")
                return reference
            placed = self.placed.get(resolved)
            if placed is None:
                placed = self._target(resolved.suffix.lower())
                self.place(resolved, placed, label)
            relative = "./" + Path(os.path.relpath(placed, target.parent)).as_posix()
            if relative != reference:
                changed.append(True)
            return relative

        def udim(reference: str, base: Path) -> str:
            head, _, tail = base.name.partition(_UDIM)
            pattern = re.compile(re.escape(head) + r"(1[0-9]{3})" + re.escape(tail))
            try:
                tiles = sorted(item for item in base.parent.iterdir() if pattern.fullmatch(item.name))
            except OSError:
                tiles = []
            if not tiles or any(inside(tile) is None for tile in tiles):
                failure.append("website_company_simulation_reference_missing" if not tiles
                               else "website_company_simulation_outside_reference")
                return reference
            stem = self._target("")
            for tile in tiles:
                number = pattern.fullmatch(tile.name).group(1)
                tile_target = stem.with_name(f"{stem.name}.{number}{tail.lower() or tile.suffix.lower()}")
                tile_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(tile.resolve(), tile_target)
                self.placed.setdefault(tile.resolve(), tile_target)
            changed.append(True)
            return "./" + Path(os.path.relpath(stem.with_name(f"{stem.name}.{_UDIM}{tail.lower()}"),
                                                target.parent)).as_posix()
        from pxr import UsdUtils
        offsets = list(layer.subLayerOffsets)
        UsdUtils.ModifyAssetPaths(layer, rewrite)
        for index, offset in enumerate(offsets):
            if index < len(layer.subLayerPaths):
                layer.subLayerOffsets[index] = offset
        if not failure:
            _rewrite_strings(layer, rewrite)
        if failure:
            raise PackageRefused(failure[0], label)
        if changed:
            if not layer.Export(str(target)):
                raise PackageRefused("website_company_simulation_scene_unreadable", label)
        else:
            shutil.copyfile(source, target)


def _layer_units(path: Path) -> tuple[str | None, float | None]:
    layer = _open_layer(str(path))
    root = layer.pseudoRoot if layer is not None else None
    up = root.GetInfo("upAxis") if root is not None and root.HasInfo("upAxis") else None
    metres = root.GetInfo("metersPerUnit") if root is not None and root.HasInfo("metersPerUnit") else None
    return (str(up) if up else None), (float(metres) if metres is not None else None)


_SAYS = {"scene_appearance": "how the site looks", "scene_collision": "the site's surfaces for contact",
         "task_object": "the job's object, at its start position", "task_support": "the surface the object goes on",
         "replacement": "a stand-in part of the site"}


def _readme(source: Mapping[str, Any], placed: Mapping[str, str], names: list[str],
            units: Mapping[str, tuple[str | None, float | None]]) -> str:
    """Only what is true of the package in front of it."""
    lines = ["Simulation of one job", "", "What this is", "",
             "A simulation of the site a deployment company was asked about, with the job's object in it,",
             "and the rule for what counts as the job done. Test in your own tools.", "",
             "How the pieces fit", "",
             "- scene.usda: open this one. It places every part below at its position, in /World."]
    hidden = []
    for prim, relative in placed.items():
        row = source["files"][prim]
        says = _SAYS.get(row["role"], "a part of the site")
        visible = row["placement"]["visible"]
        lines.append(f"- {relative} (/World/{prim}): {says}." + ("" if visible else " Hidden in scene.usda."))
        if not visible:
            hidden.append(prim)
    if any(name.startswith("scene/files/") for name in names):
        lines.append("- scene/files/: files the parts above use.")
    lines += ["- success_rule.json: what counts as done, as numbers, with the run's timing and the object's",
              "  start position and size where known.", "- README.txt: this file.", ""]
    lines += ["Each part's file is referenced on a child prim, /World/<part>/Asset. The part's position and",
              "orientation in the simulation are on /World/<part>, so any transform the file itself holds (its",
              "own scale or rotation) is kept and applied first, as Blueprint's simulation applies it."]
    for prim in hidden:
        lines += [f"/World/{prim} is in the scene but hidden, as in Blueprint's simulation, where it is used for",
                  "contact and not drawn. To see it, set its visibility to inherited (in usdview: select it and",
                  "press Make Visible)."]
    lines += ["", "Units and axes", "", "- scene.usda is in metres (1 unit is 1 metre), radians and seconds.",
              f"- Up is +{source['axis']} in scene.usda."]
    for prim, relative in placed.items():
        up, metres = units[prim]
        stated = [f"up axis {up}" if up else None, f"{metres:g} metres per unit" if metres is not None else None]
        stated = [item for item in stated if item]
        lines.append(f"- {relative} states " + (" and ".join(stated) if stated else "no up axis or unit") + ".")
    if source.get("appearance_representation") == "particlefield_3d_gaussian_splat":
        lines += ["- The room's look is a 3D Gaussian splat stored as USD particle fields. Some tools cannot show",
                  "  it; the hidden collision part holds the room's geometry."]
    lines += ["", "How it was made", "",
              "Built from one phone video of the site. Sizes, distances and positions are estimates from that",
              "video, not measurements. A result in this simulation is not a promise of what a robot does at the",
              "site, and says nothing about safety. No robot is included: bring your own."]
    if (source.get("rule") or {}).get("object_estimates_from_video"):
        lines.append("The object's mass and friction in success_rule.json are ranges estimated from the video.")
    lines += ["", "Terms", "",
              "This package is for judging this one job. Do not pass it on and do not use it to train software.",
              "Blueprint may ask you to delete it; then delete every copy.", ""]
    return "\n".join(lines)


def build_package(source: Mapping[str, Any], output: Path, *, literals: list[tuple[str, str]],
                  words: list[tuple[bytes, str]] | None = None, max_bytes: int = MAX_PACKAGE_BYTES) -> dict[str, Any]:
    """Stage, rewrite, compose, scan and zip the package to ``output``; ``PackageRefused`` or ``OVER_CAP`` otherwise.

    It holds no lock: every file it writes is its own (a staging folder and a partial zip named for this build).
    """
    import tempfile

    from .outbound_file_scan import literal_forms, scan_tree
    from .scene_package_stage import StageRefused, write_stage
    if source.get("refusal"):
        raise PackageRefused(source["refusal"])
    if sum(row["size_bytes"] for row in source["files"].values()) > max_bytes:
        raise ValueError(OVER_CAP)
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=STAGING + "-", dir=output.parent))
    staging, scratch = work / "package", work / "scan"
    partial = work / "simulation.zip.partial"
    staging.mkdir(mode=0o750)
    scratch.mkdir(mode=0o700)
    try:
        localizer = _Localizer(Path(source["assets_dir"]), staging)
        placed: dict[str, str] = {}
        units: dict[str, tuple[str | None, float | None]] = {}
        rows = []
        for prim, row in source["files"].items():
            if _sha256_file(row["path"]) != row["sha256"]:
                raise PackageRefused("website_company_simulation_source_changed", row["role"])
            relative = row["placement"]["name"] + row["suffix"]
            localizer.place(Path(row["path"]), staging / relative, row["role"])
            placed[prim] = relative
            units[prim] = _layer_units(staging / relative)
            rows.append({**row["placement"], "file": relative})
        try:
            write_stage(staging / "scene.usda", rows, axis=source["axis"])
        except StageRefused as error:
            raise PackageRefused("website_company_simulation_" + str(error).removeprefix("scene_stage_")) from None
        (staging / "success_rule.json").write_text(json.dumps(source["rule"], indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        (staging / "README.txt").write_text(_readme(source, placed, [*names, "README.txt"], units), encoding="utf-8")
        _resolves_inside(staging)
        findings = scan_tree(staging, [*literal_forms(literals), *(words or [])], scratch=scratch)
        if findings:
            raise PackageRefused("website_company_simulation_file_names_the_site", findings[0].sentence())
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
                info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_DEFLATED, 3, 0o100644 << 16
                with (staging / name).open("rb") as stream, archive.open(info, "w", force_zip64=True) as member:
                    shutil.copyfileobj(stream, member, 8 << 20)
        size = partial.stat().st_size
        if size > max_bytes:
            raise ValueError(OVER_CAP)
        os.replace(partial, output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"format": FORMAT, "content_type": CONTENT_TYPE, "digest": _sha256_file(output), "size_bytes": size,
            "files": names, "max_bytes": int(max_bytes)}


def _resolves_inside(staging: Path) -> None:
    """``scene.usda`` opens, and every layer and file it uses resolves inside the package."""
    from pxr import Usd, UsdUtils
    scene = staging / "scene.usda"
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene))
        stage = Usd.Stage.Open(str(scene), Usd.Stage.LoadNone)
    except Exception:  # noqa: BLE001 - any failure is a refusal
        raise PackageRefused("website_company_simulation_scene_unreadable", "scene.usda") from None
    if stage is None or unresolved or not stage.GetPrimAtPath("/World/Object").IsValid():
        raise PackageRefused("website_company_simulation_reference_missing", "scene.usda")
    for item in [*(layer.realPath for layer in layers), *assets]:
        path = Path(str(item).split("[", 1)[0])
        if not path.resolve().is_relative_to(staging.resolve()):
            raise PackageRefused("website_company_simulation_outside_reference", "scene.usda")


# --- publishing -------------------------------------------------------------------------------------------------

def package_record(*, identity: Mapping[str, str], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """What the WebApp may store: the package's identity, never a path, object key or person."""
    built, basis = manifest["package"], manifest["permission"]
    value = {"schema_version": PACKAGE_SCHEMA, "capture_id": identity["capture_id"],
             "simulation_revision": manifest["simulation_revision"], "state": "ready", "format": built["format"],
             "content_type": built["content_type"], "digest": built["digest"], "size_bytes": built["size_bytes"],
             "files": list(built["files"]), "permission_basis": basis["permission_basis"],
             "statement_version": basis["statement_version"], "updated_at_epoch": int(manifest["built_at_epoch"])}
    value["simulation_digest"] = cross_runtime_canonical_digest(value)
    return value


@contextmanager
def package_lock(capture_root: Path) -> Iterator[None]:
    directory = Path(capture_root).joinpath(*DIRECTORY)
    if not directory.parent.is_dir() or directory.parent.is_symlink():
        raise ValueError("website_company_simulation_capture_changed")  # a retired capture is never re-created
    directory.mkdir(mode=0o750, exist_ok=True)
    descriptor = os.open(directory / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("website_company_simulation_lock_unsafe")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CompanySimulationBusy("website_company_simulation_busy") from None
        yield


def _archive(path: Path, *, identity: Mapping[str, str]) -> dict[str, Any]:
    """The private artifact store only, never the legacy fallback, with its bytes proven."""
    from .task_evaluation_configured_scene_object_store import (
        publish_configured_scene_artifact, verify_dedicated_artifact_store)
    verify_dedicated_artifact_store()
    filename = package_filename(identity["capture_id"], identity["request_id"])
    reference = publish_configured_scene_artifact(path=path, artifact_kind=ARTIFACT_KIND, filename=filename)
    return {"uri": reference["uri"], "digest": reference["digest"], "size_bytes": reference["size_bytes"],
            "filename": filename}


def _deliver(record: Mapping[str, Any], *, identity: Mapping[str, str]) -> None:
    from .website_task_context import website_webapp_request
    acceptance = website_webapp_request(capture_id=identity["capture_id"], operation="company-simulation", payload={
        "request_id": identity["request_id"], "scene_id": identity["scene_id"], "simulation": dict(record)})
    if (acceptance.get("schema_version") != ACCEPTANCE_SCHEMA or acceptance.get("accepted") is not True
            or acceptance.get("simulation_revision") != record["simulation_revision"]
            or acceptance.get("simulation_digest") != record["simulation_digest"]):
        raise ValueError("website_company_simulation_acceptance_invalid")


def _current(manifest: Mapping[str, Any], revision: str, package: Path) -> bool:
    built, archive = manifest.get("package") or {}, manifest.get("archive") or {}
    return (manifest.get("schema_version") == MANIFEST_SCHEMA and manifest.get("simulation_revision") == revision
            and not package.is_symlink() and package.is_file() and package.stat().st_size == built.get("size_bytes")
            and archive.get("digest") == built.get("digest") and not clip._hidden(archive)
            and (manifest.get("delivered") or {}).get("simulation_digest") == manifest.get("simulation_digest"))


def _take_back_if_archived(root: Path, manifest_path: Path, retract: Callable[[str], None]) -> None:
    if not manifest_path.is_file():
        return
    with package_lock(root):
        manifest = _read(manifest_path)
        if manifest:
            clip._take_back(manifest, manifest_path, retract)


def _record_statement_check(root: Path, now: float, error: BaseException) -> None:
    """Shared with the clip: a capture its files do not permit is read live once a window, for both."""
    directory = root.joinpath(*clip.DIRECTORY)
    if directory.parent.is_dir() and not directory.parent.is_symlink():
        directory.mkdir(mode=0o750, exist_ok=True)
        write_json(directory / clip.STATEMENT_CHECK_FILE, {"checked_at_epoch": int(now), "outcome": clip._reason(error)})


def publish_company_simulation(capture_root: Path, *, now: Callable[[], float] = time.time,
                               check_rights: Callable[[Mapping[str, str]], Mapping[str, Any]] = clip._live_rights,
                               archive: Callable[..., dict[str, Any]] = _archive,
                               retract: Callable[[str], None] = clip._retract,
                               deliver: Callable[..., None] = _deliver, queue_root: Path | None = None,
                               max_bytes: int = MAX_PACKAGE_BYTES) -> dict[str, Any]:
    """One idempotent pass for one capture: make, archive and deliver what is not yet current.

    Returns ``{"status": ...}``: ``current``, ``published``, ``not_permitted``, ``no_source`` (no compiled
    simulation yet), ``over_size_cap``, ``withdrawn``, ``refused`` (rights, the takedown gate, or the package may
    not leave, with a reason), ``busy`` or ``failed`` (retried by the next pass). It never raises.
    """
    try:
        return _publish(Path(capture_root), now, check_rights, archive, retract, deliver, queue_root, max_bytes)
    except CompanySimulationBusy:
        return {"status": "busy"}
    except Exception as error:  # noqa: BLE001 - one capture never stops a pass; retried next time
        logger.warning("website_company_simulation_failed capture=%s type=%s", Path(capture_root).name,
                       type(error).__name__)
        return {"status": "failed", "reason": clip._reason(error)}


def _own_reason(error: BaseException) -> str:
    """The reason in the simulation's own name: the clip's shared rights checks speak in the clip's."""
    reason = clip._reason(error)
    return "website_company_simulation_" + reason.removeprefix("website_company_clip_") \
        if reason.startswith("website_company_clip_") else reason


def _publish(root, now, check_rights, archive, retract, deliver, queue_root, max_bytes):
    identity = clip._identity(root)
    directory = root.joinpath(*DIRECTORY)
    manifest_path, package_path = directory / MANIFEST_FILE, directory / PACKAGE_FILE
    if clip._retraction_pending(_read(manifest_path)):
        with package_lock(root):
            manifest = _read(manifest_path)
            if clip._retraction_pending(manifest) and not clip._take_back(manifest, manifest_path, retract):
                return {"status": "failed", "reason": "website_company_simulation_retraction_pending"}
    if capture_withdrawn(root):
        return {"status": "withdrawn"}
    if not clip.delivery_gate_allows(root):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "refused", "reason": "website_company_simulation_takedown_gate_closed"}
    if not clip._may_be_permitted(root, identity, now()):
        _take_back_if_archived(root, manifest_path, retract)
        return {"status": "not_permitted"}
    if simulation_source(root, queue_root=queue_root) is None:
        return {"status": "no_source"}
    # 1. Under the lock: decide. 2. Without it: build (minutes for a real scene; withdrawal deletion is never held).
    # 3. Under the lock again: check nothing changed, then publish.
    with package_lock(root):
        source = simulation_source(root, queue_root=queue_root)
        if source is None:
            return {"status": "no_source"}
        revision = simulation_revision(source, max_bytes=max_bytes)
        manifest = _read(manifest_path)
        if _current(manifest, revision, package_path):
            return {"status": "current", "simulation_revision": revision}
        if manifest.get("simulation_revision") == revision and manifest.get("refused") == OVER_CAP:
            return {"status": "over_size_cap"}
        if manifest.get("simulation_revision") == revision and isinstance(manifest.get("refused"), str):
            return {"status": "refused", "reason": manifest["refused"]}
        try:
            basis = clip.permission_from_rights(root, check_rights(identity), identity)
        except Exception as error:  # noqa: BLE001 - any doubt about rights is a refusal
            _record_statement_check(root, now(), error)
            if str(error) != clip._NOT_PERMITTED:
                return {"status": "refused", "reason": _own_reason(error)}
            if manifest:
                clip._take_back(manifest, manifest_path, retract)
            return {"status": "not_permitted"}
        built = manifest.get("package") or {}
        needs_build = (manifest.get("simulation_revision") != revision or package_path.is_symlink()
                       or not package_path.is_file() or _sha256_file(package_path) != built.get("digest"))
        if needs_build and manifest.get("archive") and not clip._take_back(manifest, manifest_path, retract):
            return {"status": "failed", "reason": "website_company_simulation_retraction_pending"}
    base = {"schema_version": MANIFEST_SCHEMA, **identity, "simulation_revision": revision,
            "source": {"plan_digest": source.get("plan_digest"),
                       "files": {prim: row["sha256"] for prim, row in sorted(source["files"].items())}},
            "permission": basis, "built_at_epoch": int(now()), "archive": None, "delivered": None}
    candidate, outcome = directory / f".candidate-{revision[7:23]}.zip", None
    if needs_build:
        try:
            built = build_package(source, candidate, literals=secret_literals(root, source),
                                  words=capture_site_words(root), max_bytes=max_bytes)
        except PackageRefused as error:
            outcome = {**base, "package": None, "refused": error.reason, "detail": error.detail}
        except ValueError as error:
            if str(error) != OVER_CAP:
                candidate.unlink(missing_ok=True)
                raise
            outcome = {**base, "package": None, "refused": OVER_CAP}
    try:
        with package_lock(root):
            if capture_withdrawn(root):
                return {"status": "withdrawn"}
            if needs_build:
                fresh = simulation_source(root, queue_root=queue_root)
                if fresh is None or simulation_revision(fresh, max_bytes=max_bytes) != revision \
                        or _read(manifest_path).get("simulation_revision") not in (None, revision, manifest.get(
                            "simulation_revision")):
                    return {"status": "failed", "reason": "website_company_simulation_changed_during_build"}
                if outcome is not None:
                    package_path.unlink(missing_ok=True)
                    write_json(manifest_path, outcome)
                    if outcome["refused"] == OVER_CAP:
                        logger.warning("website_company_simulation_over_size_cap capture=%s", identity["capture_id"])
                        return {"status": "over_size_cap"}
                    logger.warning("website_company_simulation_refused capture=%s reason=%s", identity["capture_id"],
                                   outcome["refused"])
                    return {"status": "refused", "reason": outcome["refused"], "detail": outcome["detail"]}
                os.replace(candidate, package_path)
                manifest = {**base, "package": {**built, "path": PACKAGE_FILE}}
            return _publish_built(root, identity, manifest, manifest_path, package_path, basis, built, revision,
                                  now, archive, retract, deliver)
    finally:
        candidate.unlink(missing_ok=True)


def _publish_built(root, identity, manifest, manifest_path, package_path, basis, built, revision, now, archive,
                   retract, deliver):
    manifest["permission"] = basis
    record = package_record(identity=identity, manifest=manifest)
    manifest["simulation_digest"] = record["simulation_digest"]
    write_json(manifest_path, manifest)
    stored = manifest.get("archive") or {}
    if stored.get("digest") != built["digest"] or clip._hidden(stored):
        manifest["archive"] = archive(package_path, identity=identity)
        if manifest["archive"]["digest"] != built["digest"]:
            raise ValueError("website_company_simulation_archive_mismatch")
        write_json(manifest_path, manifest)
    if capture_withdrawn(root):
        clip._take_back(manifest, manifest_path, retract)
        return {"status": "withdrawn"}
    try:
        deliver(record, identity=identity)
    except Exception as error:  # noqa: BLE001 - classified below; the archive stands only with rights
        reason = _own_reason(error)
        if _RIGHTS_REFUSALS.search(str(error)):
            clip._take_back(manifest, manifest_path, retract)
            return {"status": "refused", "reason": reason}
        manifest["last_delivery_error"] = reason
        write_json(manifest_path, manifest)
        return {"status": "failed", "reason": reason}
    manifest["delivered"] = {"simulation_digest": record["simulation_digest"], "at_epoch": int(now())}
    manifest.pop("last_delivery_error", None)
    write_json(manifest_path, manifest)
    return {"status": "published", "simulation_revision": revision, "simulation_digest": record["simulation_digest"]}


def tick(storage_root: Path, **options: Any) -> dict[str, int]:
    """One pass over every capture the clip pass covers; counts of each outcome."""
    counts: dict[str, int] = {}
    for capture in clip.capture_roots(storage_root):
        status = publish_company_simulation(capture, **options)["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def open_company_simulation(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
                            range_header: str | None = None, client: Any = None,
                            bucket: str | None = None) -> tuple[Any, int, tuple[int, int] | None]:
    """The archived package body, its full size and the range served, for a read the WebApp authorized.

    While the workspace exists the capture must be unwithdrawn, the takedown gate must allow it and its manifest
    must name this digest as archived; after retirement only the object named for this capture and digest exists
    to read (withdrawal deletion deletes it).
    """
    if not capture_name_matches_request(request_id, capture_id) or not _DIGEST.fullmatch(str(digest)):
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    if storage_root is not None:
        roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
                 if root.is_dir()]
        if len(roots) > 1:
            raise CompanySimulationUnavailable("website_company_simulation_unavailable")
        if roots:
            manifest = _read(roots[0].joinpath(*DIRECTORY) / MANIFEST_FILE)
            stored = manifest.get("archive") or {}
            if (not clip.delivery_gate_allows(roots[0]) or manifest.get("schema_version") != MANIFEST_SCHEMA
                    or stored.get("digest") != digest or clip._hidden(stored)):
                raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    from .task_evaluation_configured_scene_object_store import _artifact_object_store_client, _object_missing
    if client is None or bucket is None:
        client, bucket = _artifact_object_store_client()
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/{ARTIFACT_KIND}/sha256/{digest[7:]}/{package_filename(capture_id, request_id)}"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(error):
            raise CompanySimulationUnavailable("website_company_simulation_unavailable") from None
        raise
    size = int(head["ContentLength"])
    if (head.get("Metadata") or {}).get("sha256") != digest[7:] or not 0 < size <= MAX_PACKAGE_BYTES:
        raise CompanySimulationUnavailable("website_company_simulation_unavailable")
    served = clip.parse_range(range_header, size)
    extra = {"Range": f"bytes={served[0]}-{served[1]}"} if served else {}
    return client.get_object(Bucket=bucket, Key=key, **extra)["Body"], size, served


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("publish", help="make and publish one capture's simulation package if it is not current")
    one.add_argument("--capture-root", type=Path, required=True)
    every = commands.add_parser("tick", help="make and publish every capture's simulation package that is not current")
    every.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = (publish_company_simulation(args.capture_root) if args.command == "publish"
              else tick(args.storage_root))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
