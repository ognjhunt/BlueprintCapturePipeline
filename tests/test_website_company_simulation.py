"""The simulation package a deployment company downloads: made from the compiled scene, scanned, zipped, served.

Hermetic: temporary capture roots and team scene halves built here with real ``pxr`` USD files, and fakes for
the WebApp rights read, the private artifact store, the WebApp acceptance and the bucket. No provider, bucket or
host is reached, and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_company_simulation.py
# Covers: src/blueprint_pipeline/outbound_file_scan.py
from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import zipfile
import zlib
from pathlib import Path

import pytest
from fastapi import Request

from blueprint_pipeline import outbound_file_scan as scan
from blueprint_pipeline import website_company_simulation as sim
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils  # noqa: E402

SHARED, OLDER = "2026-11-12.v4", "2026-11-11.v3"
NOW = 1_794_500_000
REQUEST = "capture-1b2c3d4e-5f60-4718-8a9b-0c1d2e3f4a5b"
CAPTURE = f"walkthrough-{REQUEST}"
SITE = f"site-{REQUEST}"
TASK = "website-0f1e2d3c4b5a69788796"
ACCOUNT = "PLANTaccount7Q2w9"
CRITERIA = {"destination_containment": {"position_bounds_world_m": {"minimum": [0.1, 0.2, 0.7],
                                                                    "maximum": [0.4, 0.5, 0.9]}},
            "settling": {"position_tolerance_m": 0.01}, "motion": {"minimum_lift_m": 0.05}}
OBJECT_POSE = {"position_world_m": [-0.43, -0.3, 0.8], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
IDENTITY_POSE = {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _png(text: dict[str, str] | None = None, *, compressed: bool = False) -> bytes:
    """A 1x1 PNG, with text metadata chunks when given."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    for key, value in (text or {}).items():
        data += (chunk(b"zTXt", key.encode() + b"\x00\x00" + zlib.compress(value.encode())) if compressed
                 else chunk(b"tEXt", key.encode() + b"\x00" + value.encode()))
    return data + chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + chunk(b"IEND", b"")


def _stage(path: Path, root: str, *, doc: str | None = None, texture: str | None = None,
           sublayer: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, f"/{root}").GetPrim()
    stage.SetDefaultPrim(prim)
    UsdGeom.Cube.Define(stage, f"/{root}/Box")
    if doc is not None:
        prim.SetCustomDataByKey("note", doc)
        stage.GetRootLayer().documentation = doc
    if texture is not None:
        shader = UsdShade.Shader.Define(stage, f"/{root}/Look/Texture")
        shader.CreateIdAttr("UsdUVTexture")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
    if sublayer is not None:
        stage.GetRootLayer().subLayerPaths.append(sublayer)
    stage.GetRootLayer().Save()
    return path


def _usdz(path: Path, root: str) -> Path:
    layer = _stage(path.with_name(path.stem + "-src.usdc"), root)
    assert UsdUtils.CreateNewUsdzPackage(str(layer), str(path))
    layer.unlink()
    return path


class Job:
    """One website capture on disk and its compiled team scene half, with the queue status naming it."""

    def __init__(self, tmp_path: Path, *, version: str = SHARED, appearance: str = ".usdc"):
        self.tmp = tmp_path
        self.capture = tmp_path / f"storage/source-bucket/scenes/{SITE}/captures/{CAPTURE}"
        _write(self.capture / "raw/manifest.json", {
            "site_submission_id": REQUEST, "scene_id": SITE, "capture_id": CAPTURE,
            "capture_rights": {"consent_status": "granted", "consent_revoked": False, "statement_version": version}})
        _write(self.capture / "capture_descriptor.json", {"metadata": {"website_scene_execution_authority": {
            "owner": {"uid": ACCOUNT}}}})
        _write(self.capture / "pipeline/source_video/normalization.json", {})  # the clip pass's capture list
        self.queue = tmp_path / "queue"
        self.scene = tmp_path / "halves/0123/abcd-ef01/scene"
        self.packet = self.scene / "native-task-packet"
        self.assets = self.packet / "assets"
        _write(self.assets / "textures/wood.png", _png())
        self.appearance = _stage(self.assets / f"appearance{appearance}", "Appearance", texture="./textures/wood.png")
        self.collision = _stage(self.assets / "collision.usdc", "Collision")
        self.object = _usdz(self.assets / "replacement.usdz", "Object")
        self.criteria = json.loads(json.dumps(CRITERIA))
        self.seal()

    def seal(self) -> None:
        rows = []
        for role, path, pose, visible in (("scene_collision", self.collision, IDENTITY_POSE, False),
                                          ("scene_appearance", self.appearance, IDENTITY_POSE, True),
                                          ("task_object", self.object, OBJECT_POSE, True)):
            rows.append({"name": role, "semantic_role": role, "usd_path": f"assets/{path.name}", "sha256": _sha(path),
                         "size_bytes": path.stat().st_size, "visible": visible, "pose_world": pose})
        _write(self.packet / sim.PLAN_FILE, {
            "schema_version": "native_task_arena_scene_plan.v1", "scene_id": SITE, "task_id": TASK,
            "asset_directory": "assets", "objects": rows, "plan_digest": "sha256:" + "7" * 64,
            "cadence": {"physics_frequency_hz": 120.0, "control_frequency_hz": 15.0},
            "appearance_frame_alignment": {"representation": "usd_geometry"},
            "task_spec": {"task_success_contract": {"criteria": self.criteria, "scope": {"site_id": SITE}},
                          "configured_owner_authority": {"accepted_by": ACCOUNT},
                          "interaction_affordance": {"lift_unit_world": [0.0, 0.0, 1.0]}}})
        _write(self.scene / "team_scene_inputs.v1.json", {"task_success_contract": {"criteria": self.criteria}})
        _write(self.queue / "status/31e12dca983b8837295ac911c44dee68.json", {
            "schema_version": "blueprint.team_scene_packet_status.v1", "status": "completed", "outcome": "compiled",
            "capture_id": CAPTURE, "capture_root": str(self.capture), "task_id": TASK,
            "request_key": "31e12dca983b8837295ac911c44dee68",
            "team_scene_inputs_path": str(self.scene / "team_scene_inputs.v1.json")})


class World:
    """The live rights read, private store and WebApp acceptance a pass talks to."""

    def __init__(self, job: Job, *, rights=None, delivery_error: Exception | None = None, on_archive=None):
        self.job = job
        self.rights = {"consent_status": "granted", "consent_revoked": False, "statement_version": SHARED} \
            if rights is None else rights
        self.delivery_error, self.on_archive = delivery_error, on_archive
        self.archived, self.objects, self.delivered, self.retracted, self.rights_reads = [], {}, [], [], 0

    def check_rights(self, identity):
        self.rights_reads += 1
        if isinstance(self.rights, Exception):
            raise self.rights
        return dict(self.rights)

    def archive(self, path, *, identity):
        digest, name = _sha(path), sim.package_filename(identity["capture_id"], identity["request_id"])
        self.archived.append(path)
        self.objects[f"{_prefix()}/website-company-simulation/sha256/{digest[7:]}/{name}"] = path.read_bytes()
        if self.on_archive:
            self.on_archive()
        return {"uri": f"s3://bucket/x/{digest[7:]}/{name}", "digest": digest, "size_bytes": path.stat().st_size,
                "filename": name}

    def retract(self, uri):
        self.retracted.append(uri)
        self.objects.clear()

    def deliver(self, record, *, identity):
        if self.delivery_error:
            raise self.delivery_error
        assert record["simulation_digest"] == cross_runtime_canonical_digest(
            {key: value for key, value in record.items() if key != "simulation_digest"})
        self.delivered.append(record)

    def publish(self, **options):
        return sim.publish_company_simulation(
            self.job.capture, now=lambda: NOW, check_rights=self.check_rights, archive=self.archive,
            retract=self.retract, deliver=self.deliver, queue_root=self.job.queue, **options)


def _prefix() -> str:
    from blueprint_pipeline.task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    return LARGE_ARTIFACT_KEY_PREFIX


def _manifest(job: Job) -> dict:
    return json.loads((job.capture.joinpath(*sim.DIRECTORY) / sim.MANIFEST_FILE).read_text())


def _package(job: Job) -> Path:
    return job.capture.joinpath(*sim.DIRECTORY) / sim.PACKAGE_FILE


def _withdraw(capture: Path) -> None:
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1", "scene_id": SITE,
             "request_id": REQUEST, "consent_revoked_at": "2026-11-12T12:00:00Z"}
    _write(capture.parents[1] / "website_withdrawal/tombstone.json",
           {**value, "digest": canonical_digest(value, digest_field="digest")})


@pytest.fixture
def job(tmp_path):
    return Job(tmp_path)


# --- the shared scan ------------------------------------------------------------------------------------------

def _scan_text(tmp_path: Path, text: str, name: str = "notes.txt", literals=()):
    return scan.scan_file(_write(tmp_path / name, text), name, scan.literal_forms(literals))


@pytest.mark.parametrize(("text", "kind"), [
    (f"see {REQUEST}", "a job id"), (f"see {REQUEST.replace('-', '_')}", "a job id"),
    ("at /var/lib/blueprint/x", "a host path"), ("at /mnt/blueprint-work/x", "a host path"),
    ("at /" + "home/someone/x", "a host path"), ("at /tmp/x", "a host path"), ("at /opt/blueprint/x", "a host path"),
    ("at C:\\Users\\someone\\x", "a host path"), ("at D:/Work/x", "a host path"),
    ("in gs://bucket/x", "a storage link"), ("in S3://bucket/x", "a storage link"),
    ("in file:///x/y", "a local file link"), ("go http://x.invalid/", "a web link"),
    ("go https://example.org/x", "a web link"), ("write to plant.example.com today", "a web address"),
    ("or bakery.co.uk", "a web address"), ("key AKIAABCDEFGHIJKLMNOP", "a key or token"),
    ("key sk-proj-abcdefghijklmnopqrstuvwxyz", "a key or token"), ("at url?X-Amz-Signature=abcdef0123456789",
                                                                    "a key or token"),
    ("-----BEGIN RSA PRIVATE KEY-----", "a key or token"), ("key ghp_" + "a" * 36, "a key or token"),
])
def test_the_scan_finds_ids_paths_links_and_keys_and_never_repeats_them(tmp_path, text, kind):
    finding = _scan_text(tmp_path, text, literals=[(REQUEST, "a job id")])
    assert finding is not None and finding.kind == kind and finding.file == "notes.txt"
    secret = text.split(" ", 1)[1]
    assert secret not in repr(finding) and secret not in finding.sentence()


@pytest.mark.parametrize("text", [
    "open scene.usd and object.usdz", "version 1.2.3 of usd-core 0.26.5", "README.md, notes.txt, rule.json",
    "the namespace http://www.w3.org/2000/svg", "e.g. 0.5 m, 3.14, v2.1", "numpy.random and pxr.Usd",
    "material:binding and xformOp:translate", "a/tmp/b is relative", "file_name.py", "./textures/wood.png",
])
def test_ordinary_file_names_versions_and_namespaces_are_left_alone(tmp_path, text):
    assert _scan_text(tmp_path, text) is None


def test_a_bare_web_address_counts_only_in_text(tmp_path):
    assert _scan_text(tmp_path, "abc example.com", name="x.bin") is None
    assert _scan_text(tmp_path, "abc example.com", name="x.json").kind == "a web address"


def test_ids_inside_a_binary_usd_are_found_through_its_text_form(tmp_path):
    path = _stage(tmp_path / "layer.usdc", "Root", doc=REQUEST.replace("-", "_"))
    finding = scan.scan_file(path, "layer.usdc", scan.literal_forms([(REQUEST, "a job id")]))
    assert finding.kind == "a job id" and finding.file == "layer.usdc"
    clean = _stage(tmp_path / "clean.usdc", "Root")
    assert scan.scan_file(clean, "clean.usdc", scan.literal_forms([(REQUEST, "a job id")])) is None
    # The text form is read as well as the bytes: the exported text alone finds the id, and says where.
    found = scan._usd_text_finding(path, scan.literal_forms([(REQUEST, "a job id")]))
    assert found is not None and found[0] == "a job id" and "text form" in found[1]


def test_a_usd_that_cannot_be_opened_is_refused_not_skipped(tmp_path):
    for name in ("broken.usdc", "broken.usda", "broken.usd"):
        finding = scan.scan_file(_write(tmp_path / name, b"PXR-USDC not really"), name, [])
        assert finding is not None and finding.kind == scan.UNREADABLE_USD
    finding = scan.scan_file(_write(tmp_path / "broken.usdz", b"not a zip"), "broken.usdz", [])
    assert finding.kind == scan.UNREADABLE_USD


def test_a_usdz_member_layer_is_read_through_its_text_form(tmp_path):
    layer = _stage(tmp_path / "member.usdc", "Root", doc=f"gs://bucket/{REQUEST}")
    package = tmp_path / "object.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(layer), str(package))
    finding = scan.scan_file(package, "object.usdz", [])
    # A usdz keeps its members uncompressed, so the bytes alone already show it; either way it is found.
    assert finding.kind.startswith("a storage link") and finding.file == "object.usdz"


@pytest.mark.parametrize("compressed", [False, True])
def test_texture_metadata_is_read(tmp_path, compressed):
    path = _write(tmp_path / "wood.png", _png({"Comment": f"made at https://plant.example.com/{REQUEST}"},
                                              compressed=compressed))
    finding = scan.scan_file(path, "wood.png", [])
    assert finding is not None and finding.kind in {"a web link", "a web address"}
    assert REQUEST not in finding.sentence()
    comment = b"see bakery.example.com"
    jpeg = _write(tmp_path / "wood.jpg", b"\xff\xd8\xff\xfe" + struct.pack(">H", len(comment) + 2) + comment
                  + b"\xff\xda\x00\x02")
    assert scan.scan_file(jpeg, "wood.jpg", []).kind == "a web address"


# --- the package ----------------------------------------------------------------------------------------------

def test_a_clean_scene_makes_a_package_that_passes_the_scan_opens_and_is_stable(job, tmp_path):
    world = World(job)
    result = world.publish()
    assert result["status"] == "published", result
    package = _package(job)
    record = world.delivered[0]
    assert record["schema_version"].startswith("website_company_simulation.") and record["state"] == "ready"
    assert record["digest"] == _sha(package) and record["size_bytes"] == package.stat().st_size
    assert record["capture_id"] == CAPTURE and record["permission_basis"] == "site_sharing_statement"
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        unpacked = tmp_path / "unpacked"
        archive.extractall(unpacked)
    textures = [name for name in names if name.startswith("scene/files/")]
    assert len(textures) == 1 and textures[0].endswith(".png")
    assert set(names) - set(textures) == {"README.txt", "scene.usda", "scene/room.usdc", "scene/room_collision.usdc",
                                          "object.usdz", "success_rule.json"}
    assert record["files"] == names
    # Every file passes the shared scan, with every id of the job looked for.
    literals = scan.literal_forms([(REQUEST, "a job id"), (CAPTURE, "a job id"), (TASK, "a job id"),
                                   (ACCOUNT, "an account id")])
    assert scan.scan_tree(unpacked, literals) == []
    assert scan.scan_file(package, "simulation.zip", literals) is None
    # It opens with pxr and every layer and file it uses resolves inside the package.
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(unpacked / "scene.usda"))
    assert unresolved == [] and len(layers) >= 3
    for item in [*(layer.realPath for layer in layers), *assets]:
        assert Path(item.split("[", 1)[0]).resolve().is_relative_to(unpacked.resolve())
    stage = Usd.Stage.Open(str(unpacked / "scene.usda"))
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    # Each part sits where the plan puts it: the object's world transform is the plan's pose.
    obj = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Object"))
    world = obj.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert list(world.ExtractTranslation()) == pytest.approx(OBJECT_POSE["position_world_m"])
    rotation = world.ExtractRotationQuat()
    assert [*rotation.GetImaginary(), rotation.GetReal()] == pytest.approx(OBJECT_POSE["orientation_xyzw"])
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/RoomCollision")).ComputeVisibility() == "invisible"
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/Room")).ComputeVisibility() == "inherited"
    rule = json.loads((unpacked / "success_rule.json").read_text())
    assert rule["criteria"] == CRITERIA and rule["up_axis"] == "Z"
    assert rule["object_start"] == {"position_m": OBJECT_POSE["position_world_m"],
                                    "orientation_xyzw": OBJECT_POSE["orientation_xyzw"]}
    assert rule["timing"] == {"physics_frequency_hz": 120.0, "control_frequency_hz": 15.0}
    # Nothing from the plan is copied as text.
    scene_text = (unpacked / "scene.usda").read_text()
    for value in (SITE, TASK, ACCOUNT, "sha256:", "assets/", "scene_appearance", "task_object"):
        assert value not in scene_text and value not in json.dumps(rule)
    readme = (unpacked / "README.txt").read_text()
    assert "metres" in readme and "Up is +Z" in readme and "phone video" in readme and "train software" in readme
    assert "delete" in readme and "Gaussian" not in readme
    # The same inputs give the same bytes.
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    again = sim.build_package(source, tmp_path / "again.zip", literals=sim.secret_literals(job.capture, source))
    assert again["digest"] == record["digest"]


def test_published_once_then_current_without_rebuilding_or_resending(job, monkeypatch):
    world = World(job)
    assert world.publish()["status"] == "published"
    monkeypatch.setattr(sim, "build_package", lambda *a, **k: pytest.fail("rebuilt a current package"))
    assert world.publish()["status"] == "current"
    assert len(world.archived) == 1 and len(world.delivered) == 1 and world.rights_reads == 1


def test_a_new_simulation_revision_replaces_the_package(job):
    world = World(job)
    first = world.publish()
    _stage(job.collision, "Collision", doc="a second compile")
    job.seal()
    second = world.publish()
    assert second["status"] == "published" and second["simulation_revision"] != first["simulation_revision"]
    assert len(world.retracted) == 1 and len(world.delivered) == 2


CARRIERS = ("usda", "usdc", "texture", "success_rule")
PLANTS = {"request_id": REQUEST, "request_id_underscore": REQUEST.replace("-", "_"), "capture_id": CAPTURE,
          "account_id": ACCOUNT, "task_id": TASK, "host_path": "/var/lib/blueprint/pubsub-handoffs/x",
          "home_path": "/" + "home/operator/x", "storage": "gs://plant-bucket/x", "link": "https://plant.invalid/x",
          "domain": "plant-site.com", "token": "AKIAPLANT0123456789A"}


def _plant(job: Job, carrier: str, value: str) -> str:
    if carrier in {"usda", "usdc"}:
        job.appearance.unlink()
        job.appearance = _stage(job.assets / f"appearance.{carrier}", "Appearance", doc=f"note {value} end",
                                texture="./textures/wood.png")
        expected = f"scene/room.{carrier}"
    elif carrier == "texture":
        _write(job.assets / "textures/wood.png", _png({"Comment": f"note {value} end"}))
        expected = "scene/files/file-00"
    else:
        job.criteria[f"note {value} end"] = 1  # a value would be dropped as not neutral; a key is kept
        expected = "success_rule.json"
    job.seal()
    return expected


@pytest.mark.parametrize("carrier", CARRIERS)
@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_a_planted_id_path_link_or_key_refuses_the_package_and_names_the_file(tmp_path, carrier, plant):
    job = Job(tmp_path)
    expected = _plant(job, carrier, PLANTS[plant])
    world = World(job)
    result = world.publish()
    assert result["status"] == "refused" and result["reason"] == "website_company_simulation_file_names_the_site"
    assert result["detail"].startswith(expected) and " holds " in result["detail"], result
    assert PLANTS[plant] not in json.dumps(result) and PLANTS[plant] not in _manifest(job)["detail"]
    assert world.archived == [] and world.delivered == [] and not _package(job).exists()
    # Recorded for this revision: the next pass neither rebuilds nor raises.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sim, "build_package", lambda *a, **k: pytest.fail("rebuilt a refused package"))
        assert world.publish()["status"] == "refused"


def test_an_absolute_reference_inside_the_scene_is_rewritten_to_a_relative_one(job, tmp_path):
    job.appearance = _stage(job.appearance, "Appearance", texture=str(job.assets / "textures/wood.png"))
    job.seal()
    world = World(job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        archive.extractall(tmp_path / "out")
    text = Sdf.Layer.FindOrOpen(str(tmp_path / "out/scene/room.usdc")).ExportToString()
    assert str(job.assets) not in text and re.search(r"@\./files/file-00\d\.png@", text)
    assert UsdUtils.ComputeAllDependencies(str(tmp_path / "out/scene.usda"))[2] == []


@pytest.mark.parametrize("where", ["absolute_outside", "relative_escape", "link", "missing"])
def test_an_outside_or_missing_reference_refuses_the_package(job, tmp_path, where):
    outside = _stage(tmp_path / "elsewhere/other.usda", "Other")
    reference = {"absolute_outside": str(outside), "relative_escape": "../../../../elsewhere/other.usda",
                 "link": "https://plant.invalid/other.usda", "missing": "./not-there.usda"}[where]
    job.appearance = _stage(job.appearance, "Appearance", sublayer=reference)
    job.seal()
    world = World(job)
    result = world.publish()
    assert result["status"] == "refused" and result["reason"] in {
        "website_company_simulation_outside_reference", "website_company_simulation_reference_missing"}
    assert world.archived == [] and str(outside) not in json.dumps(result)


def test_an_object_usdz_with_an_outside_reference_is_refused(job, tmp_path):
    layer = _stage(tmp_path / "obj.usda", "Object", sublayer="/srv/elsewhere/x.usda")
    job.object.unlink()
    with zipfile.ZipFile(job.object, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(layer, "obj.usda")
    job.seal()
    result = World(job).publish()
    assert result["status"] == "refused" and result["reason"] == "website_company_simulation_outside_reference"
    assert "elsewhere" not in json.dumps(result)


@pytest.mark.parametrize(("change", "reason"), [
    (lambda row: row.pop("pose_world"), "website_company_simulation_pose_missing:task_object"),
    (lambda row: row["pose_world"].update(orientation_xyzw=[0.0, 0.0, 0.0, 0.0]),
     "website_company_simulation_pose_missing:task_object"),
    (lambda row: row.update(usd_path="../elsewhere/x.usdz"), "website_company_simulation_asset_not_shipped:task_object"),
    (lambda row: row.update(size_bytes=1), "website_company_simulation_asset_not_shipped:task_object"),
])
def test_a_row_that_cannot_be_placed_refuses_naming_its_role(job, change, reason):
    plan_path = job.packet / sim.PLAN_FILE
    plan = json.loads(plan_path.read_text())
    change(next(row for row in plan["objects"] if row["semantic_role"] == "task_object"))
    plan_path.write_text(json.dumps(plan))
    world = World(job)
    assert world.publish() == {"status": "refused", "reason": reason, "detail": ""}
    assert world.archived == [] and world.rights_reads == 1


def test_a_scaled_row_is_scaled_on_the_stage(job, tmp_path):
    from blueprint_pipeline import scene_package_stage as stage_module
    plan = json.loads((job.packet / sim.PLAN_FILE).read_text())
    plan["objects"][2]["scale"] = [2.0, 2.0, 2.0]
    rows = stage_module.placements(plan)
    out = tmp_path / "pkg"
    _usdz(out / "object.usdz", "Object")
    text = stage_module.write_stage(out / "scene.usda", [{**rows[2], "file": "object.usdz"}], axis="Z")
    assert "xformOp:scale" in text
    stage = Usd.Stage.Open(str(out / "scene.usda"))
    matrix = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Object")).ComputeLocalToWorldTransform(0)
    assert matrix.GetRow3(0).GetLength() == pytest.approx(2.0)
    with pytest.raises(stage_module.StageRefused, match="unknown"):
        stage_module.placements({"objects": [{**plan["objects"][0], "semantic_role": "robot"}]})


# --- the gate -------------------------------------------------------------------------------------------------

def test_an_older_statement_gets_nothing(tmp_path):
    job = Job(tmp_path, version=OLDER)
    world = World(job, rights={"consent_status": "granted", "consent_revoked": False, "statement_version": OLDER})
    assert world.publish() == {"status": "not_permitted"}
    assert world.archived == [] and world.rights_reads == 1 and not _package(job).exists()
    # The live read is shared with the clip's window: the next pass does not ask again.
    assert world.publish() == {"status": "not_permitted"} and world.rights_reads == 1


def test_a_withdrawn_capture_makes_nothing(job):
    _withdraw(job.capture)
    world = World(job)
    assert world.publish() == {"status": "withdrawn"}
    assert world.archived == [] and not job.capture.joinpath(*sim.DIRECTORY).exists()


def test_an_open_takedown_refuses_and_takes_the_package_back(job):
    world = World(job)
    assert world.publish()["status"] == "published"
    _write(job.capture / "pipeline/consent_takedown/takedown_manifest.json", {"takedown_open": True})
    assert world.publish()["status"] == "refused"
    assert len(world.retracted) == 1 and _manifest(job)["archive"]["retracted"] is True


def test_a_failed_or_revoked_live_read_is_a_refusal(job):
    for rights in ({"consent_status": "revoked", "consent_revoked": True, "statement_version": SHARED},
                   ValueError("website_task_context_digest_mismatch")):
        world = World(job, rights=rights)
        assert world.publish()["status"] == "refused" and world.archived == [] and world.delivered == []


def test_a_withdrawal_during_the_upload_takes_the_copy_back(job):
    world = World(job, on_archive=lambda: _withdraw(job.capture))
    assert world.publish() == {"status": "withdrawn"}
    assert len(world.retracted) == 1 and world.delivered == []


def test_a_webapp_rights_refusal_takes_the_copy_back(job):
    error = ValueError("website_control_company-simulation_http_409:website_company_simulation_not_permitted")
    world = World(job, delivery_error=error)
    assert world.publish()["status"] == "refused" and len(world.retracted) == 1


# --- never raises, never holds --------------------------------------------------------------------------------

def test_no_compiled_simulation_is_no_source(job):
    for path in (job.queue / "status").glob("*.json"):
        path.unlink()
    assert World(job).publish() == {"status": "no_source"}


def test_a_broken_half_is_no_source_and_a_broken_scene_file_is_refused(job):
    (job.packet / sim.PLAN_FILE).write_text("{not json")
    assert World(job).publish() == {"status": "no_source"}
    job.seal()
    job.collision.write_bytes(b"PXR-USDC broken" + b"\x00" * 64)
    job.seal()
    result = World(job).publish()
    assert result["status"] == "refused" and result["reason"] in {
        "website_company_simulation_scene_unreadable", "website_company_simulation_file_names_the_site"}


def test_a_huge_scene_is_flagged_once_and_not_rebuilt(job, monkeypatch):
    world = World(job)
    assert world.publish(max_bytes=1024) == {"status": "over_size_cap"}
    assert _manifest(job)["refused"] == sim.OVER_CAP and not _package(job).exists()
    monkeypatch.setattr(sim, "build_package", lambda *a, **k: pytest.fail("rebuilt an over-cap package"))
    assert world.publish(max_bytes=1024) == {"status": "over_size_cap"}


def test_nothing_raises_out_of_a_pass_or_the_tick(job, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("https://secret.invalid/x")
    monkeypatch.setattr(sim, "simulation_source", boom)
    result = World(job).publish()
    assert result["status"] == "failed" and "secret" not in json.dumps(result)
    assert sim.tick(job.tmp / "storage", queue_root=job.queue) == {"failed": 1}
    from blueprint_pipeline import website_atlas_preview
    monkeypatch.setattr(sim, "tick", lambda root: (_ for _ in ()).throw(RuntimeError("boom")))
    assert website_atlas_preview.company_simulation_tick(job.tmp) == {"tick_failed": 1}


def test_a_busy_lock_is_busy(job):
    import fcntl
    directory = job.capture.joinpath(*sim.DIRECTORY)
    directory.mkdir(parents=True)
    with (directory / sim.LOCK_FILE).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert World(job).publish() == {"status": "busy"}


def test_delivery_uses_the_signed_company_simulation_operation_and_checks_the_acceptance(monkeypatch):
    from blueprint_pipeline import website_task_context
    calls, sent = [], []
    record = {"simulation_revision": "sha256:" + "1" * 64, "simulation_digest": "sha256:" + "2" * 64}
    acceptance = {"schema_version": "website_company_simulation_acceptance.v1", "accepted": True, **record}
    monkeypatch.setattr(website_task_context, "website_webapp_request",
                        lambda **kw: calls.append(kw) or dict(acceptance))
    identity = {"request_id": REQUEST, "scene_id": SITE, "capture_id": CAPTURE}
    sim._deliver(record, identity=identity)
    assert calls == [{"capture_id": CAPTURE, "operation": "company-simulation",
                      "payload": {"request_id": REQUEST, "scene_id": SITE, "simulation": record}}]
    acceptance["simulation_digest"] = "sha256:" + "3" * 64
    with pytest.raises(ValueError, match="acceptance_invalid"):
        sim._deliver(record, identity=identity)
    monkeypatch.undo()

    class Response:
        body = b'{"accepted": true}'
    monkeypatch.setenv("PIPELINE_SYNC_WEBAPP_URL", "https://webapp.invalid")
    monkeypatch.setattr(website_task_context, "load_pipeline_sync_token", lambda: "fixture-sync-token")
    monkeypatch.setattr(website_task_context, "safe_request", lambda url, **_kw: sent.append(url) or Response())
    website_task_context.website_webapp_request(capture_id=CAPTURE, operation="company-simulation", payload={})
    assert sent == [f"https://webapp.invalid/api/internal/pipeline/creator-captures/{CAPTURE}/company-simulation"]


# --- the route ------------------------------------------------------------------------------------------------

class _Bucket:
    def __init__(self, objects):
        self.objects, self.ranges = objects, []

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise type("NoSuchKey", (Exception,), {"response": {"Error": {"Code": "404"}}})()
        body = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": {"sha256": hashlib.sha256(body).hexdigest()}}

    def get_object(self, *, Bucket, Key, Range=None):
        self.ranges.append(Range)
        body = self.objects[Key]
        if Range:
            first, last = map(int, Range.removeprefix("bytes=").split("-"))
            body = body[first:last + 1]
        return {"Body": _Body(body)}


class _Body(io.BytesIO):
    def iter_chunks(self, size):
        while chunk := self.read(size):
            yield chunk


@pytest.fixture
def served(job, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from blueprint_pipeline import website_atlas_preview_http as http
    world = World(job)
    assert world.publish()["status"] == "published"
    bucket = _Bucket(world.objects)
    digest = _manifest(job)["archive"]["digest"]

    def admission(request: Request):
        request.state.intake_client_id = request.headers.get("x-test-client", "")
    app = FastAPI()
    http.register_website_atlas_preview_routes(app, require_admission=admission)
    monkeypatch.setenv(http.HANDOFF_STORAGE_ROOT_ENV, str(job.capture.parents[4]))
    real = sim.open_company_simulation
    monkeypatch.setattr(sim, "open_company_simulation", lambda **kw: real(**kw, client=bucket, bucket="b"))
    url = f"/api/live-pipeline/website-company-simulations/{CAPTURE}/simulation.zip?request_id={REQUEST}&digest={digest}"
    return {"client": TestClient(app), "url": url, "body": _package(job).read_bytes(), "digest": digest,
            "bucket": bucket, "job": job,
            "signed": {"x-blueprint-pipeline-signature": "sha256=x", "x-test-client": "blueprint-webapp"}}


def test_the_webapp_reads_the_whole_package_and_any_one_range(served):
    client, url, body, signed = served["client"], served["url"], served["body"], served["signed"]
    whole = client.get(url, headers=signed)
    assert whole.status_code == 200, whole.text
    assert whole.content == body and whole.headers["content-length"] == str(len(body))
    assert whole.headers["content-type"] == "application/zip" and whole.headers["accept-ranges"] == "bytes"
    assert whole.headers["cache-control"] == "private, no-store"
    assert whole.headers["x-blueprint-artifact-sha256"] == served["digest"]
    assert whole.headers["content-disposition"] == 'attachment; filename="simulation.zip"'
    part = client.get(url, headers={**signed, "range": "bytes=100-1123"})
    assert part.status_code == 206 and part.content == body[100:1124]
    assert part.headers["content-range"] == f"bytes 100-1123/{len(body)}"
    assert part.headers["x-blueprint-artifact-sha256"] == served["digest"]
    assert part.headers["cache-control"] == "private, no-store"
    past = client.get(url, headers={**signed, "range": f"bytes={len(body)}-"})
    assert past.status_code == 416 and past.headers["content-range"] == f"bytes */{len(body)}"
    assert past.json() == {"detail": "website_company_simulation_range_not_satisfiable"}


def test_only_the_webapp_reads_and_only_the_named_package(served):
    client, url, signed = served["client"], served["url"], served["signed"]
    assert client.get(url, headers={**signed, "x-test-client": "robot-team"}).status_code == 403
    assert client.get(url, headers={"x-test-client": "blueprint-webapp"}).status_code == 403
    missing = client.get(url.replace(served["digest"][7:], "0" * 64), headers=signed)
    assert missing.status_code == 404 and missing.json() == {"detail": "website_company_simulation_unavailable"}
    assert client.get(url.replace(f"request_id={REQUEST}", "request_id=other1"), headers=signed).status_code == 404


def test_withdrawal_and_an_open_takedown_close_the_route(served):
    job = served["job"]
    _write(job.capture / "pipeline/consent_takedown/takedown_manifest.json", {"takedown_open": True})
    assert served["client"].get(served["url"], headers=served["signed"]).status_code == 404
    (job.capture / "pipeline/consent_takedown/takedown_manifest.json").unlink()
    _withdraw(job.capture)
    assert served["client"].get(served["url"], headers=served["signed"]).status_code == 404


def test_a_storage_fault_is_503_and_names_nothing(served, monkeypatch):
    def broken(**_kw):
        raise RuntimeError("https://secret.invalid/signed?token=x")
    monkeypatch.setattr(sim, "open_company_simulation", broken)
    response = served["client"].get(served["url"], headers=served["signed"])
    assert response.status_code == 503 and "secret" not in response.text
    assert response.json() == {"detail": "website_company_simulation_storage_unavailable"}
