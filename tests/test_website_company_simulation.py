"""The simulation package a deployment company downloads: made from the compiled scene, scanned, zipped, served.

Hermetic: temporary capture roots and team scene halves built here with real ``pxr`` USD files, and fakes for
the WebApp rights read, the private artifact store, the WebApp acceptance and the bucket. No provider, bucket or
host is reached, and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_company_simulation.py
# Covers: src/blueprint_pipeline/website_company_look.py
# Covers: src/blueprint_pipeline/outbound_file_scan.py
# Covers: src/blueprint_pipeline/website_company_member_date.py
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
        # The preparation every half is built from (``website_half_preparation``): here one that says nothing.
        self.preparation = self.capture / "pipeline/website_scene_preparation/preparation.json"
        if not self.preparation.exists():
            _write(self.preparation, {})
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

    def seal_record(self, **fields) -> None:
        """The half's own record sealed by its digest, with ``fields``, as a compile seals it: ``bound_to`` (a
        preparation file) names the preparation it was built from by its bytes (``website_half_preparation``)."""
        bound = fields.pop("bound_to", None)
        path = self.scene / "team_scene_inputs.v1.json"
        record = {**json.loads(path.read_text()), "plan_digest": "sha256:" + "7" * 64, **fields,
                  **({"preparation_sha256": _sha(Path(bound))} if bound is not None else {}), "inputs_digest": ""}
        record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
        _write(path, record)


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
        """Every version of the one object the link names (its digest), as the store's deletion does."""
        self.retracted.append(uri)
        for key in [key for key in self.objects if f"/{uri.split('/')[-2]}/" in key]:
            del self.objects[key]

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


def _assert_absent_from_zip(job: Job, values) -> None:
    """Every member of the published zip, and every member of a usdz in it, is free of every value."""
    with zipfile.ZipFile(_package(job)) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            blobs = [name.encode(), data]
            if name.endswith(".usdz"):
                with zipfile.ZipFile(io.BytesIO(data)) as inner:
                    blobs += [inner.comment] + [part for info in inner.infolist()
                                                for part in (info.filename.encode(), info.extra, inner.read(info))]
            for blob in blobs:
                for value in values:
                    assert value not in blob, (name, value)


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
    # Text is decided by content, never by the ending: a file of text is text whatever it is called.
    binary = _write(tmp_path / "x.json", b"\x00\x01abc example.com\xff")
    assert scan.scan_file(binary, "x.json", []) is None
    assert _scan_text(tmp_path, "abc example.com", name="x.bin").kind == "a web address"


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
    # The job's object is in this package (2026-11-24, ``object_included``, read by the website when present).
    assert record["object_included"] is True
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        # This capture's records carry no date: every member keeps the zip format's first one (``ZIP_EPOCH``).
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        # Plain headers that agree (2026-11-25: macOS's Archive Utility refused a package whose local headers named a
        # ZIP64 field its central directory did not have): no member is in ZIP64, nor needs a ZIP64 reader.
        assert all(info.extract_version == 20 and info.create_version == 20 for info in archive.infolist())
        offsets = [info.header_offset for info in archive.infolist()]
    with package.open("rb") as raw:
        for offset in offsets:
            raw.seek(offset)
            header = raw.read(30)
            version, compressed, size, name_length, extra_length = (
                int.from_bytes(header[4:6], "little"), int.from_bytes(header[18:22], "little"),
                int.from_bytes(header[22:26], "little"), int.from_bytes(header[26:28], "little"),
                int.from_bytes(header[28:30], "little"))
            raw.seek(name_length, 1)
            extra = raw.read(extra_length)
            assert header[:4] == b"PK\x03\x04" and version == 20
            assert 0xFFFFFFFF not in (compressed, size) and b"\x01\x00" != extra[:2]
    with zipfile.ZipFile(package) as archive:
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


#: Job move-the-lamp-2's capture (``raw/manifest.json``): 2026-11-19 15:09:20.505 UTC, when the upload was received.
LAMP_CAPTURE_MS = 1795100960505


def test_members_carry_the_capture_s_own_date_and_the_same_inputs_give_the_same_bytes(job, tmp_path, monkeypatch):
    """Finder showed every file of lamp-2's package as made on 1 January 1980 (25 Nov): each member now carries the
    capture's own date, from its manifest, never the build's, so the same inputs still give the same digest."""
    import time

    manifest = json.loads((job.capture / "raw/manifest.json").read_text())
    _write(job.capture / "raw/manifest.json", {**manifest, "capture_start_epoch_ms": LAMP_CAPTURE_MS,
                                               "capture_start_epoch_ms_basis": "server_upload_received"})
    world = World(job)
    assert world.publish()["status"] == "published"
    record = world.delivered[0]
    with zipfile.ZipFile(_package(job)) as archive:
        assert {info.date_time for info in archive.infolist()} == {(2026, 11, 19, 15, 9, 20)}
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert source["member_date"] == (2026, 11, 19, 15, 9, 20)
    literals = sim.secret_literals(job.capture, source)
    # Built again later, with the clock somewhere else entirely: the same bytes.
    monkeypatch.setattr(time, "time", lambda: 4_102_444_800.0)  # 2100-01-01
    again = sim.build_package(source, tmp_path / "again.zip", literals=literals)
    assert again["digest"] == record["digest"]
    # Another capture date is another package: its revision and its bytes.
    _write(job.capture / "raw/manifest.json", {**manifest, "capture_start_epoch_ms": LAMP_CAPTURE_MS + 86_400_000})
    moved = sim.simulation_source(job.capture, queue_root=job.queue)
    assert moved["member_date"] == (2026, 11, 20, 15, 9, 20)
    assert sim.simulation_revision(moved) != sim.simulation_revision(source)
    assert sim.build_package(moved, tmp_path / "moved.zip", literals=literals)["digest"] != record["digest"]


def test_the_member_date_falls_back_through_the_capture_s_inputs_and_never_to_now(tmp_path):
    from blueprint_pipeline import website_company_member_date as dates

    root = tmp_path / "capture"
    assert dates.member_date(None) == dates.member_date(root) == dates.ZIP_EPOCH == (1980, 1, 1, 0, 0, 0)
    # The rights' recorded time, when no record gives the capture's start.
    _write(root / "pipeline/website_task_context.json",
           {"capture_rights": {"recorded_at_iso": "2026-11-19T15:09:16.758Z"}})
    assert dates.member_date(root) == (2026, 11, 19, 15, 9, 16)
    # The delivery's descriptor before it, and the capture's own manifest before both.
    _write(root / "capture_descriptor.json", {"metadata": {"media_metadata": {"capture_start_epoch_ms": 1795000000000}}})
    assert dates.member_date(root) == (2026, 11, 18, 11, 6, 40)
    _write(root / "raw/manifest.json", {"capture_start_epoch_ms": LAMP_CAPTURE_MS})
    assert dates.member_date(root) == (2026, 11, 19, 15, 9, 20)
    # A value a zip cannot date (1 ms after 1970, as a browser fixture sends), or that is no time, is passed over.
    for bad in (1, True, "1795100960505", float("nan"), 10 ** 20):
        _write(root / "raw/manifest.json", {"capture_start_epoch_ms": bad})
        assert dates.member_date(root) == (2026, 11, 18, 11, 6, 40), bad
    # A rights time with no zone is not read as UTC; an unreadable file is passed over.
    (root / "capture_descriptor.json").write_text("{not json")
    _write(root / "pipeline/website_task_context.json", {"capture_rights": {"recorded_at_iso": "2026-11-19T15:09:16"}})
    assert dates.member_date(root) == dates.ZIP_EPOCH


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


CARRIERS = ("usda", "usdc", "texture")
PLANTS = {"request_id": REQUEST, "request_id_underscore": REQUEST.replace("-", "_"), "capture_id": CAPTURE,
          "account_id": ACCOUNT, "task_id": TASK, "host_path": "/var/lib/blueprint/pubsub-handoffs/x",
          "home_path": "/" + "home/operator/x", "storage": "gs://plant-bucket/x", "link": "https://plant.invalid/x",
          "domain": "plant-site.com", "token": "AKIAPLANT0123456789A"}


def _room_as_the_host_writes_it(job: Job, *, custom: str | None = None) -> str:
    """The room's appearance as the episode compiler's task clearance writes it on the host: a layer under the
    host's own storage, edited on the session layer and written with ``Usd.Stage.Export``, which flattens and adds
    the source-file note naming that layer (``task_evaluation_native_arena_episode_compiler``)."""
    source = _stage(job.tmp / "var/lib/blueprint/task-evaluation-inputs/team-scene-halves/0123/abcd/scene/"
                    "configured-scene/appearance.usdc", "Appearance", texture=str(job.assets / "textures/wood.png"))
    stage = Usd.Stage.Open(str(source))
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath("/Appearance/Box")).AddTranslateOp().Set((0.0, 0.0, 0.01))
    if custom is not None:
        stage.GetRootLayer().customLayerData = {"source": custom}
    job.appearance.unlink()
    job.appearance = job.assets / "scene_appearance.usdc"
    assert stage.Export(str(job.appearance))
    documentation = Sdf.Layer.FindOrOpen(str(job.appearance)).documentation
    assert documentation.startswith(sim.FLATTEN_PROVENANCE) and "/var/lib/blueprint/" in documentation
    job.seal()
    return str(source)


def test_object_included_is_true_only_when_the_jobs_object_is_in_the_package():
    half_parts = {"Room": "sha256:" + "1" * 64, "RoomCollision": "sha256:" + "2" * 64, "Object": "sha256:" + "3" * 64}
    assert sim.object_included({"source": {"files": half_parts}}) is True
    room_parts = {key: value for key, value in half_parts.items() if key != "Object"}
    assert sim.object_included({"source": {"files": room_parts}}) is False  # a room package, or a floor
    assert sim.object_included({}) is False


def test_a_website_that_predates_object_included_still_gets_the_package_once(job):
    """The website's reader of ``website_company_simulation.v1`` is strict: until it lists ``object_included`` it
    answers 409 ``website_company_simulation_invalid``. The same package then goes without the field, that record
    is the one recorded as out, and later passes find it current: no deploy, no request every pass."""
    world = World(job)
    shapes = []

    def strict_website(record, *, identity):
        shapes.append("object_included" in record)
        if "object_included" in record:
            raise ValueError("website_control_company-simulation_http_409:website_company_simulation_invalid")
        World.deliver(world, record, identity=identity)
    world.deliver = strict_website
    assert world.publish()["status"] == "published"
    assert shapes == [True, False] and "object_included" not in world.delivered[0]
    manifest = _manifest(job)
    assert manifest["delivered"]["simulation_digest"] == manifest["simulation_digest"] == world.delivered[0][
        "simulation_digest"]
    assert world.publish()["status"] == "current" and shapes == [True, False]


def test_a_refusal_of_anything_but_the_records_shape_is_not_resent_without_the_field(job):
    world = World(job, delivery_error=ValueError("website_control_company-simulation_http_503:"
                                                 "website_company_simulation_unavailable"))
    assert world.publish()["status"] == "failed" and world.delivered == []


def test_a_room_written_the_way_the_host_writes_it_ships_without_the_host_path(tmp_path):
    """capture itest-e3e25932, 2026-11-16: refused for ``scene/room.usdc holds a host path``, the flatten note."""
    job = Job(tmp_path)
    host = _room_as_the_host_writes_it(job)
    result = World(job).publish()
    assert result["status"] == "published", result
    _assert_absent_from_zip(job, [host.encode(), b"/var/lib/blueprint", sim.FLATTEN_PROVENANCE.encode()])


def test_a_package_refused_by_an_earlier_builder_is_built_at_once_by_this_one(tmp_path, monkeypatch):
    """A refusal is retried only after its window under the same builder; a new builder builds it at the first pass."""
    job = Job(tmp_path)
    _room_as_the_host_writes_it(job)
    world = World(job)
    with monkeypatch.context() as patch:  # the builder before the flatten note was dropped, as on 6ad7b73d3
        patch.setattr(sim, "BUILDER_REVISION", "website_company_simulation_builder.v1")
        patch.setattr(sim, "FLATTEN_PROVENANCE", "\x00never")
        refused = world.publish()
        assert refused["status"] == "refused" and refused["detail"] == "scene/room.usdc holds a host path"
        patch.setattr(sim, "build_package", lambda *a, **k: pytest.fail("rebuilt inside the window"))
        assert world.publish()["status"] == "refused"  # same builder, same revision: it waits
    published = world.publish()
    assert published["status"] == "published", published
    assert "refused" not in _manifest(job)


def test_a_host_path_the_writer_does_not_clean_still_refuses_the_package(tmp_path):
    job = Job(tmp_path)
    _room_as_the_host_writes_it(job, custom="/var/lib/blueprint/pubsub-handoffs/elsewhere/appearance.ply")
    result = World(job).publish()
    assert result["status"] == "refused" and result["reason"] == "website_company_simulation_file_names_the_site"
    assert result["detail"] == "scene/room.usdc holds a host path", result


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
    if carrier == "texture":
        # A PNG's text chunks are never written into the package (outbound_file_writer): the value cannot ship.
        assert result["status"] == "published", result
        _assert_absent_from_zip(job, [PLANTS[plant].encode()])
        return
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
    # A role this module does not know gets a neutral numbered prim; a repeated role is numbered.
    named = stage_module.placements({"objects": [{**plan["objects"][0], "semantic_role": "task_drawer"},
                                                 {**plan["objects"][2]}, {**plan["objects"][2]},
                                                 {**plan["objects"][0], "semantic_role": "Jane's cabinet"}]})
    assert [row["prim"] for row in named] == ["Part_01", "Object", "Object_02", "Part_02"]
    assert [row["name"] for row in named] == ["scene/part_01", "object", "scene/object_02", "scene/part_02"]
    assert named[3]["role"] == "unnamed"


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


@pytest.mark.parametrize("atlas", ["long", "raises"])
def test_a_ready_rooms_package_is_delivered_before_the_atlas_steps_start(job, monkeypatch, atlas):
    # A finished room's package waited behind every job's paid Atlas steps and every clip; the unit now runs the
    # package pass first, whatever the Atlas steps then do.
    from blueprint_pipeline import owned_capture_atlas_worker
    from blueprint_pipeline import website_atlas_preview as preview
    from blueprint_pipeline import website_company_clip
    world, real = World(job), sim.publish_company_simulation
    monkeypatch.setattr(sim, "publish_company_simulation", lambda capture, **_: real(
        capture, now=lambda: NOW, check_rights=world.check_rights, archive=world.archive, retract=world.retract,
        deliver=world.deliver, queue_root=job.queue))
    monkeypatch.setattr(website_company_clip, "tick", lambda root, **_: {})
    monkeypatch.setattr(preview, "trigger_stop_reason", lambda: lambda: None)
    monkeypatch.setattr(preview, "scene_preview_tick", lambda root: {})
    monkeypatch.setattr(owned_capture_atlas_worker, "tick_from_environment", lambda: {})
    seen = []

    def atlas_tick(root, **options):
        seen.append(len(world.delivered))
        if atlas == "raises":
            raise RuntimeError("atlas step failed")
        return {"processing": 1}
    monkeypatch.setattr(preview, "tick", atlas_tick)
    if atlas == "raises":
        with pytest.raises(RuntimeError, match="atlas step failed"):
            preview.main(["tick", "--storage-root", str(job.tmp / "storage")])
    else:
        assert preview.main(["tick", "--storage-root", str(job.tmp / "storage")]) == 0
    assert seen == [1] and len(world.delivered) == 1 and len(world.archived) == 1
    assert world.delivered[0]["simulation_digest"] == _manifest(job)["simulation_digest"]
    # The pass after the Atlas steps finds the package current: no second build or delivery.
    assert world.rights_reads == 1


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
    monkeypatch.setattr(sim, "_wall_clock", lambda: NOW + 1)  # the route's own clock, beside the pass's NOW
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


# --- review fixes (part A) -------------------------------------------------------------------------------------

@pytest.mark.parametrize("plant", sorted(PLANTS))
def test_a_value_planted_in_the_success_rule_never_ships(job, tmp_path, plant):
    # The rule is an allow-list: an unknown key, a nested label, a path and free text are left out, not shipped.
    job.criteria[f"note {PLANTS[plant]} end"] = 1
    job.criteria["settling"]["label_text"] = PLANTS[plant]
    job.criteria["support"] = {"contact_mode": "required", "support_prim_path": "/World/Shelf_" + ACCOUNT,
                               "visible_label": "back shelf", "height_interval_m": [0.8, 0.9]}
    job.criteria["motion"]["kind"] = "loading_dock_by_bobs_desk"
    job.seal()
    world = World(job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        rule = json.loads(archive.read("success_rule.json"))
    text = json.dumps(rule)
    for value in (PLANTS[plant], "back shelf", "loading_dock", "Shelf_", "label_text", "note "):
        assert value not in text
    assert rule["criteria"]["support"] == {"contact_mode": "required", "height_interval_m": [0.8, 0.9]}
    assert rule["criteria_fields_left_out"] >= 5


def test_the_success_rule_allow_list_maps_the_support_prim_only_when_the_stage_has_one():
    from blueprint_pipeline import scene_package_stage as stage_module
    criteria = {"support": {"support_prim_path": "/Root/x", "contact_mode": "required"}, "jane_house": {"mode": "x"},
                "surface_target": {"shape": "flat_green_disc", "visible_label": "wooden table", "radius_m": 0.3,
                                   "nested": [{"label": "a"}]}, "safety": {"mode": "free text here"}}
    kept, left_out = stage_module.neutral_criteria(criteria, support_prim="/World/Support")
    assert kept == {"support": {"support_prim": "/World/Support", "contact_mode": "required"},
                    "surface_target": {"shape": "flat_green_disc", "radius_m": 0.3}}
    assert left_out == 4
    assert "support_prim" not in stage_module.neutral_criteria(criteria)[0]["support"]


def test_the_object_estimates_from_the_video_go_in_the_rule_and_are_left_out_when_absent(job):
    _write(job.capture / "pipeline/website_scene_preparation/preparation.json",
           {"physics": {"bounds": {"mass_kg": [0.4, 1.8], "static_friction": [0.2, 0.9], "dynamic_friction": "x"}}})
    world = World(job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        rule = json.loads(archive.read("success_rule.json"))
        readme = archive.read("README.txt").decode()
    assert rule["object_estimates_from_video"] == {"mass_kg": [0.4, 1.8], "static_friction": [0.2, 0.9]}
    assert "estimated from the video" in readme
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    (job.capture / "pipeline/website_scene_preparation/preparation.json").unlink()
    assert "object_estimates_from_video" not in sim.simulation_source(job.capture, queue_root=job.queue)["rule"]
    assert source["rule"]["object_estimates_from_video"]


def test_the_rule_says_which_numbers_the_site_s_words_set_and_never_quotes_them(job):
    """Owner direction 2026-11-23: the success rule from the site's own words; the company reads each number's basis."""
    _write(job.capture / "pipeline/website_scene_preparation/preparation.json", {"success_rule_basis": {
        "build": "rigid_pick_place", "applied": True, "fields": {
            "minimum_lift_m": {"value": 0.05, "basis": "assumed"},
            "maximum_episode_seconds": {"value": 60, "basis": "site_words", "quote": "Jane's Bakery: up to a minute"},
            "maximum_final_planar_target_error_m": {"value": 0.08, "basis": "destination_area"}}}})
    world = World(job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        rule = json.loads(archive.read("success_rule.json"))
        readme = archive.read("README.txt").decode()
        raw = b"".join(archive.read(name) for name in archive.namelist() if name.endswith((".json", ".txt")))
    assert rule["criteria_basis"] == {"minimum_lift_m": "assumed", "maximum_episode_seconds": "site_stated",
                                      "maximum_final_planar_target_error_m": "destination_area"}
    assert "criteria_basis" in readme and b"Jane" not in raw
    (job.capture / "pipeline/website_scene_preparation/preparation.json").unlink()
    assert "criteria_basis" not in sim.simulation_source(job.capture, queue_root=job.queue)["rule"]


def _asset_with_root_transform(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddScaleOp().Set((0.01, 0.01, 0.01))
    root.AddRotateZOp().Set(90.0)
    UsdGeom.Cube.Define(stage, "/Object/Box")
    stage.GetRootLayer().Save()
    return path


def test_the_assets_own_root_transform_is_kept_under_the_plans_pose(job, tmp_path):
    from pxr import Gf
    asset = _asset_with_root_transform(tmp_path / "src/obj.usda")
    job.object.unlink()
    job.object = job.assets / "replacement.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(asset), str(job.object))
    job.seal()
    assert World(job).publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        archive.extractall(tmp_path / "out")
    stage = Usd.Stage.Open(str(tmp_path / "out/scene.usda"))
    asset_world = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Object/Asset")).ComputeLocalToWorldTransform(0)
    source_stage = Usd.Stage.Open(str(asset))
    own = UsdGeom.Xformable(source_stage.GetPrimAtPath("/Object")).ComputeLocalToWorldTransform(0)
    pose = Gf.Matrix4d(1.0)
    x, y, z, w = OBJECT_POSE["orientation_xyzw"]
    pose.SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    pose.SetTranslateOnly(Gf.Vec3d(*OBJECT_POSE["position_world_m"]))
    expected = own * pose  # USD row vectors: the asset's own transform first, then the plan's pose
    for row in range(4):
        assert list(asset_world.GetRow(row)) == pytest.approx(list(expected.GetRow(row)))
    assert asset_world.GetRow3(0).GetLength() == pytest.approx(0.01)
    readme = (tmp_path / "out/README.txt").read_text()
    assert "/World/<part>/Asset" in readme and "keeps its own root transform" in readme
    assert "Blueprint's simulation applies" not in readme
    # The file's root turns 90 degrees: the case where tools that spawn a file AT the posed prim would differ.
    assert "/World/Object: this part's file carries its own offset" in readme
    rule = json.loads((tmp_path / "out/success_rule.json").read_text())
    note = rule["part_notes"]["/World/Object"]["own_root_offset"]
    assert note["translation_m"] == [0.0, 0.0, 0.0]
    assert note["rotation_xyzw"] == pytest.approx([0.0, 0.0, 0.707107, 0.707107], abs=1e-5)
    assert "part_notes" not in json.dumps({k: v for k, v in rule.items() if k != "part_notes"})
    assert "To see it, set its visibility to inherited" in readme


def test_sublayer_time_offsets_survive_the_rewrite(job, tmp_path):
    other = _stage(job.assets / "layers/extra.usda", "Extra")
    stage = Usd.Stage.Open(str(job.appearance))
    layer = stage.GetRootLayer()
    layer.subLayerPaths.append(str(other))  # absolute inside the scene: rewritten to a relative path
    layer.subLayerOffsets[0] = Sdf.LayerOffset(7.0, 2.0)
    layer.Save()
    job.seal()
    assert World(job).publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        archive.extractall(tmp_path / "out")
    shipped = Sdf.Layer.FindOrOpen(str(tmp_path / "out/scene/room.usdc"))
    assert list(shipped.subLayerOffsets) == [Sdf.LayerOffset(7.0, 2.0)]
    assert not Path(shipped.subLayerPaths[0]).is_absolute()


def _string_texture(job: Job, value: str) -> None:
    stage = Usd.Stage.Open(str(job.appearance))
    prim = stage.GetPrimAtPath("/Appearance/Look/Texture")
    prim.CreateAttribute("inputs:mdl_tex", Sdf.ValueTypeNames.String).Set(value)
    stage.GetRootLayer().Save()
    job.seal()


def test_a_string_that_names_a_file_is_shipped_and_rewritten(job, tmp_path):
    _write(job.assets / "textures/rough.png", _png())
    _string_texture(job, "./textures/rough.png")
    assert World(job).publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        archive.extractall(tmp_path / "out")
    stage = Usd.Stage.Open(str(tmp_path / "out/scene/room.usdc"))
    value = stage.GetPrimAtPath("/Appearance/Look/Texture").GetAttribute("inputs:mdl_tex").Get()
    assert value.startswith("./files/file-") and (tmp_path / "out/scene" / value).is_file()


@pytest.mark.parametrize("value", ["./textures/missing.png", "/srv/elsewhere/rough.png"])
def test_a_string_that_names_a_missing_or_outside_file_refuses_naming_the_role(job, value):
    _string_texture(job, value)
    result = World(job).publish()
    assert result["status"] == "refused" and result["detail"] == "scene_appearance"
    assert result["reason"] in {"website_company_simulation_reference_missing",
                                "website_company_simulation_outside_reference"}


def test_udim_tiles_inside_the_scene_are_all_shipped(job, tmp_path):
    for tile in (1001, 1002):
        _write(job.assets / f"textures/wood.{tile}.png", _png())
    job.appearance = _stage(job.appearance, "Appearance", texture="./textures/wood.<UDIM>.png")
    job.seal()
    assert World(job).publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        names = archive.namelist()
        archive.extractall(tmp_path / "out")
    tiles = [name for name in names if re.fullmatch(r"scene/files/file-\d{3}\.100[12]\.png", name)]
    assert len(tiles) == 2
    text = Sdf.Layer.FindOrOpen(str(tmp_path / "out/scene/room.usdc")).ExportToString()
    assert re.search(r"@\./files/file-\d{3}\.<UDIM>\.png@", text)


def test_udim_without_tiles_refuses(job):
    job.appearance = _stage(job.appearance, "Appearance", texture="./textures/none.<UDIM>.png")
    job.seal()
    result = World(job).publish()
    assert result["status"] == "refused" and result["reason"] == "website_company_simulation_reference_missing"


def test_an_unknown_role_gets_a_neutral_part_prim_and_is_placed(job, tmp_path):
    plan_path = job.packet / sim.PLAN_FILE
    plan = json.loads(plan_path.read_text())
    drawer = _stage(job.assets / "drawer.usda", "Drawer")
    plan["objects"].append({"name": "x", "semantic_role": "task_drawer", "usd_path": "assets/drawer.usda",
                            "sha256": _sha(drawer), "size_bytes": drawer.stat().st_size, "visible": True,
                            "pose_world": {"position_world_m": [1.0, 2.0, 0.5], "orientation_xyzw": [0, 0, 0, 1]}})
    plan_path.write_text(json.dumps(plan))
    assert World(job).publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        assert "scene/part_01.usda" in archive.namelist()
        archive.extractall(tmp_path / "out")
    stage = Usd.Stage.Open(str(tmp_path / "out/scene.usda"))
    world = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Part_01")).ComputeLocalToWorldTransform(0)
    assert list(world.ExtractTranslation()) == pytest.approx([1.0, 2.0, 0.5])
    assert "task_drawer" not in (tmp_path / "out/scene.usda").read_text()


def test_the_resolves_inside_proof_is_what_refuses_a_reference_the_localizer_left(job, monkeypatch):
    # Pin the proof: a localizer that copied a layer without rewriting its reference is caught by the proof alone.
    other = _stage(job.assets / "layers/extra.usda", "Extra")
    job.appearance = _stage(job.appearance, "Appearance", sublayer=str(other))
    job.seal()

    def verbatim(self, source, target, label):
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil
        _shutil.copyfile(source, target)
    monkeypatch.setattr(sim._Localizer, "place", verbatim)
    result = World(job).publish()
    assert result["status"] == "refused" and result["reason"] in {
        "website_company_simulation_outside_reference", "website_company_simulation_reference_missing"}
    assert result["detail"] == "scene.usda"


def test_a_rights_refusal_is_reported_in_the_simulations_own_name(job):
    world = World(job, rights={"consent_status": "revoked", "consent_revoked": True, "statement_version": SHARED})
    result = world.publish()
    assert result == {"status": "refused", "reason": "website_company_simulation_rights_refused"}


# --- the site's identity (its description of the job is shared by its own agreement) ---------------------------

SITE_NAME, STREET, EMAIL, PHONE = "Zebra Plant Foods", "17 Quayside Lane, Harbourton, UK", "owner@zebra.example", \
    "+44 20 7946 0958"
SENTENCE = "Move the wooden tray from the counter to the back shelf every morning"
IDENTITY = "the site's name or contact details"


def _site_records(job: Job) -> None:
    _write(job.capture / "pipeline/website_task_context.json", {
        "description": SENTENCE, "operator_answers": {"who": "Jane Placeholder"},
        "capture_rights": {"consent_status": "granted", "consent_revoked": False, "statement_version": SHARED}})
    descriptor = json.loads((job.capture / "capture_descriptor.json").read_text())
    descriptor["metadata"]["site_identity"] = {"name": SITE_NAME, "address": STREET, "email": EMAIL, "phone": PHONE}
    _write(job.capture / "capture_descriptor.json", descriptor)


def _plant_identity(job: Job, where: str, value: str) -> str:
    if where == "prim_name":
        stage = Usd.Stage.Open(str(job.appearance))
        UsdGeom.Xform.Define(stage, f"/Appearance/{value}")
        stage.GetRootLayer().Save()
        expected = "scene/room.usdc"
    elif where == "string_attribute":
        stage = Usd.Stage.Open(str(job.appearance))
        stage.GetPrimAtPath("/Appearance").CreateAttribute("made_for", Sdf.ValueTypeNames.String).Set(value)
        stage.GetRootLayer().Save()
        expected = "scene/room.usdc"
    else:  # a texture whose name carries it: the string naming the file is in the layer
        _write(job.assets / f"textures/{value}.png", _png())
        job.appearance = _stage(job.appearance, "Appearance", texture=f"./textures/{value}.png")
        expected = "scene/room.usdc"
    job.seal()
    return expected


@pytest.mark.parametrize("where", ["prim_name", "string_attribute", "texture_name"])
@pytest.mark.parametrize("spelling", ["ZebraPlant_Foods", "zebra_plant_foods", "zebraplantfoods", "Zebra_Plant_Foods"])
def test_the_sites_name_in_any_spelling_refuses_the_package(job, where, spelling):
    _site_records(job)
    expected = _plant_identity(job, where, spelling)
    result = World(job).publish()
    if where == "texture_name":
        # A texture's own file name never ships: the package renames every file a part uses (scene/files/file-NNN).
        assert result["status"] == "published", result
        with zipfile.ZipFile(_package(job)) as archive:
            assert not any("zebra" in name.lower() for name in archive.namelist())
            assert all(b"zebra" not in archive.read(name).lower() for name in archive.namelist())
        return
    assert result["status"] == "refused", result
    assert result["detail"] in {f"{expected} holds {IDENTITY}", f"{expected} holds {IDENTITY} inside a zip member"}
    assert "zebra" not in json.dumps(result).lower()


def test_the_sites_name_with_hyphens_refuses_in_text(job):
    _site_records(job)
    job.appearance.unlink()
    job.appearance = _stage(job.assets / "appearance.usda", "Appearance", doc="made for zebra-plant-foods",
                            texture="./textures/wood.png")
    job.seal()
    assert World(job).publish()["detail"] == f"scene/room.usda holds {IDENTITY}"


@pytest.mark.parametrize("value", ["17 Quayside Lane", "Harbourton", EMAIL, PHONE, "442079460958"])
def test_the_sites_address_email_and_phone_refuse_the_package(job, value):
    _site_records(job)
    job.appearance.unlink()
    job.appearance = _stage(job.assets / "appearance.usda", "Appearance", doc=f"made at {value} today",
                            texture="./textures/wood.png")
    job.seal()
    result = World(job).publish()
    assert result["status"] == "refused" and result["detail"] == f"scene/room.usda holds {IDENTITY}", result
    assert value.lower() not in json.dumps(result).lower()


def test_the_job_itself_is_not_identity_a_tray_prim_and_the_description_are_published(job):
    _site_records(job)
    job.object.unlink()
    layer = _stage(job.assets / "src/obj.usdc", "Asset")
    stage = Usd.Stage.Open(str(layer))
    UsdGeom.Xform.Define(stage, "/Asset/wooden_tray")
    stage.GetPrimAtPath("/Asset").SetCustomDataByKey("note", SENTENCE)
    stage.GetRootLayer().Save()
    job.object = job.assets / "replacement.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(layer), str(job.object))
    job.seal()
    assert World(job).publish()["status"] == "published"


def test_identity_is_gathered_from_the_records_and_the_job_is_not(job):
    from blueprint_pipeline import outbound_file_scan
    _site_records(job)
    found = {value for value, _kind in outbound_file_scan.capture_site_identity(job.capture)}
    words = {value[3:].decode() for value in found if value.startswith(b"\x00w:")}
    normal = {value[3:].decode() for value in found if value.startswith(b"\x00n:")}
    assert {SITE_NAME.lower(), "17 quayside lane, harbourton, uk", "17 quayside lane", "harbourton", EMAIL,
            PHONE} <= words
    assert {"zebraplantfoods", "442079460958"} <= normal
    assert "uk" not in words  # a part that is only a country
    for job_words in ("move the wooden tray", "wooden tray", "jane placeholder", "back shelf"):
        assert not any(job_words in value for value in words | normal)


def test_short_and_common_identity_values_are_not_matched_and_identity_is_text_only():
    from blueprint_pipeline import outbound_file_scan as o
    assert o.site_identity(["Bob", "York", "Bath", "World", "Station", "North"]) == []
    needles = o.site_identity(["Harbourton"], addresses=["Unit 1, Mill Lane, York"])
    words = {value for value, _ in needles}
    assert b"\x00w:mill lane" in words and b"\x00n:milllane" in words and b"\x00w:harbourton" in words
    assert not words & {b"\x00w:unit 1", b"\x00n:unit1", b"\x00w:york", b"\x00n:york"}
    assert o._match_window(b"def Xform harbourtonian_sink", needles, text=True) is None  # whole words only
    assert o._match_window(b"made in Harbourton today", needles, text=True) is not None
    assert o._match_window(b"metersPerUnit = 1", needles, text=True) is None
    # Identity is never looked for in raw binary bytes.
    assert o._match_window(b"\x00made in Harbourton\x01", needles, text=False) is None


def test_accents_are_folded_on_both_sides():
    from blueprint_pipeline import outbound_file_scan as o
    needles = o.site_identity(["Café Zürich"])
    for text in ("CAFE ZURICH", "cafe_zurich", "CafeZurich", "Café Zürich"):
        assert o._match_window(f"x {text} y".encode(), needles, text=True) is not None, text


def test_a_clean_package_does_not_trip_on_the_sites_words(job):
    _site_records(job)
    assert World(job).publish()["status"] == "published"


# --- the scan's new kinds and the cheaper walk of large layers -------------------------------------------------

@pytest.mark.parametrize(("text", "kind"), [
    ("at /" + "root/x", "a host path"), ("at /" + "Users/someone/x", "a host path"), ("at /srv/x", "a host path"),
    ("at /data/x", "a host path"), ("at ~/x", "a host path"), ("at \\\\server\\share", "a host path"),
    ("at /var//lib/blueprint/x", "a host path"), ("in b2://bucket/x", "a link"), ("in ftp://x", "a link"),
    ("in wasbs://x@y", "a link"), ("in omniverse://x", "a link"), ("host 10.1.2.3 here", "an IP address"),
    ("mail jane@bakery.example.zz now", "an email address"), ("key p_ghp_" + "a" * 36, "a key or token"),
    ("set aws_secret_access_key=", "a key or token"),
])
def test_the_scan_finds_the_new_kinds(tmp_path, text, kind):
    finding = _scan_text(tmp_path, text)
    assert finding is not None and finding.kind == kind, finding


@pytest.mark.parametrize("text", ["usd-core 0.26.5 and 1.2.3", "v10.1.2", "1.2.3.4.5", "0.5, 1.25, 3.0, 4.75",
                                  "scene.usd", "xformOp:translate = (1.0, 2.0, 3.0)", "@./a.png@ and @./b.usd@"])
def test_the_new_kinds_leave_ordinary_numbers_alone(tmp_path, text):
    assert _scan_text(tmp_path, text) is None


@pytest.mark.parametrize("walk", [False, True])
@pytest.mark.parametrize("where", ["doc", "custom_data", "prim_name", "string_attr", "time_sample", "asset_info",
                                   "relationship", "layer_data"])
def test_the_spec_walk_finds_what_the_text_export_finds(tmp_path, monkeypatch, walk, where):
    monkeypatch.setattr(scan, "SPEC_WALK_BYTES", 0 if walk else 1 << 40)
    planted = REQUEST.replace("-", "_")
    path = tmp_path / "layer.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    UsdGeom.Points.Define(stage, "/Root/Cloud").CreatePointsAttr([(float(i), 0.0, 0.0) for i in range(2000)])
    if where == "doc":
        stage.GetRootLayer().documentation = planted
    elif where == "custom_data":
        root.SetCustomDataByKey("k", planted)
    elif where == "prim_name":
        UsdGeom.Xform.Define(stage, f"/Root/{planted}")
    elif where == "string_attr":
        root.CreateAttribute("note", Sdf.ValueTypeNames.String).Set(planted)
    elif where == "time_sample":
        root.CreateAttribute("note", Sdf.ValueTypeNames.String).Set(planted, 3.0)
    elif where == "asset_info":
        root.SetAssetInfoByKey("name", planted)
    elif where == "relationship":
        UsdGeom.Xform.Define(stage, f"/{planted}")
        root.CreateRelationship("link").AddTarget(f"/{planted}")
    else:
        stage.GetRootLayer().customLayerData = {"by": planted}
    stage.GetRootLayer().Save()
    found = scan._usd_text_finding(path, scan.literal_forms([(REQUEST, "a job id")]))
    assert found is not None and found[0] == "a job id"


# --- review fixes (part B): permission that ends, the website's refusal, the route, the pass's budget ------------

UPLOAD = "sha256:" + "ab" * 32


def _at(world: World, seconds: int) -> dict:
    return sim.publish_company_simulation(world.job.capture, now=lambda: NOW + seconds, check_rights=world.check_rights,
                                          archive=world.archive, retract=world.retract, deliver=world.deliver,
                                          queue_root=world.job.queue)


def _read_route(job: Job, world: World, *, now: float, storage_root: Path | None = None, digest: str | None = None):
    try:
        return sim.open_company_simulation(request_id=REQUEST, capture_id=CAPTURE,
                                           digest=digest or _manifest(job)["archive"]["digest"],
                                           storage_root=job.capture.parents[4] if storage_root is None else storage_root,
                                           client=_Bucket(world.objects), bucket="b", now=now)[1]
    except sim.CompanySimulationUnavailable:
        return "unavailable"


def test_blueprints_own_capture_under_an_older_statement_gets_no_package(tmp_path):
    # C-10: the website's sharing rule is the one authority; the Pipeline keeps no list of its own captures.
    job = Job(tmp_path, version=OLDER)
    _write(job.capture.joinpath("pipeline", "company_clip", "company_clip.json"), {"upload_sha256": UPLOAD})
    world = World(job, rights={"consent_status": "granted", "consent_revoked": False, "statement_version": OLDER})
    assert world.publish() == {"status": "not_permitted"} and world.archived == []
    assert not hasattr(sim, "OWNED_CAPTURES_LIST")


@pytest.mark.parametrize("later", [60, 7 * 3600, 30 * 86400])
def test_a_live_revocation_after_publishing_takes_the_package_back(job, later):
    world = World(job)
    assert world.publish()["status"] == "published"
    reads = world.rights_reads
    world.rights = {"consent_status": "revoked", "consent_revoked": True, "statement_version": SHARED}
    result = _at(world, later)
    if later < sim.permission.LIVE_STATEMENT_RECHECK_SECONDS:
        # Within the window nothing is read: the package stays until the window, or a recheck mark, asks again.
        assert result["status"] == "current" and world.rights_reads == reads
        from blueprint_pipeline.permission_recheck import request_recheck
        assert request_recheck(job.capture, now=NOW + later) == "accepted"
        result = _at(world, later)
    assert result == {"status": "refused", "reason": "website_company_simulation_rights_refused"}
    assert len(world.retracted) == 1 and _read_route(job, world, now=NOW + later) == "unavailable"


@pytest.mark.parametrize("later", [60, 7 * 3600, 30 * 86400])
def test_with_no_live_answer_a_grant_is_kept_at_most_the_bound_and_the_route_applies_it(job, later):
    world = World(job)
    assert world.publish()["status"] == "published"
    world.rights = ValueError("website_task_context_http_503")
    result = _at(world, later)
    bound = sim.permission.LIVE_GRANT_LIFETIME_SECONDS
    if later < bound:
        assert result["status"] == "current" and _read_route(job, world, now=NOW + later) != "unavailable"
    else:
        assert result["status"] == "not_permitted" and len(world.retracted) == 1
    # The route applies the bound by itself, whatever the pass did.
    assert _read_route(job, world, now=NOW + bound + 1) == "unavailable"


def test_a_package_archived_but_not_delivered_is_taken_back_on_a_revoked_answer(job):
    world = World(job, delivery_error=ValueError("website_control_company-simulation_http_503:x"))
    assert world.publish()["status"] == "failed" and len(world.archived) == 1
    world.delivery_error = None
    world.rights = {"consent_status": "revoked", "consent_revoked": True, "statement_version": SHARED}
    result = world.publish()
    assert result == {"status": "refused", "reason": "website_company_simulation_rights_refused"}
    assert len(world.retracted) == 1 and world.objects == {} and _manifest(job)["archive"]["retracted"] is True


def test_a_website_refusal_is_remembered_four_passes_one_upload(job):
    error = ValueError("website_control_company-simulation_http_409:website_company_simulation_not_permitted")
    world = World(job, delivery_error=error)
    statuses = [_at(world, seconds)["status"] for seconds in (0, 60, 120, 180)]
    assert statuses == ["refused"] * 4 and len(world.archived) == 1 and len(world.retracted) == 1


def test_a_recheck_alone_does_not_forget_a_website_refusal_but_changed_rights_or_the_window_do(job):
    from blueprint_pipeline.permission_recheck import request_recheck
    error = ValueError("website_control_company-simulation_http_409:website_company_simulation_not_permitted")
    world = World(job, delivery_error=error)
    assert _at(world, 0)["status"] == "refused"
    assert request_recheck(job.capture, now=NOW + 100) == "accepted"
    reads = world.rights_reads
    assert _at(world, 120)["status"] == "refused" and len(world.archived) == 1
    assert world.rights_reads == reads + 1  # the mark made the live read due; the same rights keep the refusal
    world.delivery_error = None
    assert _at(world, sim.permission.LIVE_STATEMENT_RECHECK_SECONDS + 1)["status"] == "published"
    assert len(world.archived) == 2


def test_a_new_revision_is_offered_again_after_a_website_refusal(job):
    error = ValueError("website_control_company-simulation_http_409:website_company_simulation_not_permitted")
    world = World(job, delivery_error=error)
    assert _at(world, 0)["status"] == "refused"
    _stage(job.collision, "Collision", doc="a second compile")
    job.seal()
    world.delivery_error = None
    assert _at(world, 60)["status"] == "published" and len(world.archived) == 2


def test_the_route_fails_closed(job):
    world = World(job)
    assert world.publish()["status"] == "published"
    assert _read_route(job, world, now=NOW + 1) != "unavailable"
    digest = _manifest(job)["archive"]["digest"]
    try:
        served = sim.open_company_simulation(request_id=REQUEST, capture_id=CAPTURE, digest=digest, storage_root=None,
                                             client=_Bucket(world.objects), bucket="b", now=NOW + 1)
    except sim.CompanySimulationUnavailable:
        served = "unavailable"
    assert served == "unavailable"
    _withdraw(job.capture)
    import shutil
    shutil.rmtree(job.capture)  # the workspace is gone; the site's tombstone stays
    assert world.objects  # the archived object is still stored until withdrawal deletion runs
    assert _read_route(job, world, now=NOW + 1, digest=digest) == "unavailable"
    (job.capture.parents[1] / "website_withdrawal/tombstone.json").unlink()
    # No workspace: nothing to check, so nothing is served.
    assert _read_route(job, world, now=NOW + 1, digest=digest) == "unavailable"


def test_the_pass_stops_between_captures_and_keeps_to_its_budget(tmp_path, monkeypatch):
    job = Job(tmp_path)
    calls = []
    monkeypatch.setattr(sim, "publish_company_simulation", lambda capture, **kw: calls.append(capture)
                        or {"status": "current"})
    assert sim.tick(tmp_path / "storage", stop_reason=lambda: "held") == {"deferred": 1, "stopped_held": 1}
    ticks = iter([0.0, sim.TICK_BUDGET_SECONDS + 1])
    assert sim.tick(tmp_path / "storage", clock=lambda: next(ticks)) == {"deferred": 1, "stopped_time_budget": 1}
    assert calls == [] and job.capture.exists()
    from blueprint_pipeline import website_atlas_preview
    seen = []
    monkeypatch.setattr(sim, "tick", lambda root, stop_reason=None: seen.append(stop_reason) or {})
    website_atlas_preview.company_simulation_tick(tmp_path, stop_reason=lambda: None)
    assert seen and callable(seen[0])


def test_a_build_past_its_deadline_is_recorded_and_tried_again_only_after_the_window(job, monkeypatch):
    world = World(job)
    result = sim.publish_company_simulation(job.capture, now=lambda: NOW, check_rights=world.check_rights,
                                            archive=world.archive, retract=world.retract, deliver=world.deliver,
                                            queue_root=job.queue, build_seconds=-1)
    assert result == {"status": "build_timed_out"} and _manifest(job)["refused"] == sim.BUILD_TIMED_OUT
    monkeypatch.setattr(sim, "build_package", lambda *a, **k: pytest.fail("rebuilt within the window"))
    assert _at(world, 3600) == {"status": "build_timed_out"}
    monkeypatch.undo()
    assert _at(world, sim.permission.LIVE_STATEMENT_RECHECK_SECONDS + 1)["status"] == "published"


def test_a_scan_refusal_is_tried_again_after_the_window(job, monkeypatch):
    job.appearance.unlink()
    job.appearance = _stage(job.assets / "appearance.usdc", "Appearance", doc=REQUEST, texture="./textures/wood.png")
    job.seal()
    world = World(job)
    assert _at(world, 0)["reason"] == "website_company_simulation_file_names_the_site"
    calls = []
    real = sim.build_package
    monkeypatch.setattr(sim, "build_package", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert _at(world, 3600)["status"] == "refused" and calls == []
    assert _at(world, sim.permission.LIVE_STATEMENT_RECHECK_SECONDS + 1)["status"] == "refused" and calls == [1]



# --- re-check fixes (N1 to N9) ---------------------------------------------------------------------------------

def test_a_clean_scene_whose_room_is_random_geometry_is_published(job):
    import numpy as np
    from pxr import Vt
    job.appearance.unlink()
    stage = Usd.Stage.CreateNew(str(job.appearance))
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Appearance")
    stage.SetDefaultPrim(root.GetPrim())
    points = UsdGeom.Points.Define(stage, "/Appearance/Splat")
    points.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.random.default_rng(5).random((200000, 3),
                                                                                        dtype=np.float32)))
    stage.GetRootLayer().Save()
    job.seal()
    assert World(job).publish()["status"] == "published"


def test_random_bytes_pass_the_raw_byte_patterns():
    import io as _io
    import random
    data = random.Random(7).randbytes(64 << 20)
    assert scan.scan_stream(_io.BytesIO(data), [], text=False) is None
    # The same short patterns are still found in text.
    for text in (b"at ~/x", b"in gs://b/x", b"at /tmp/x", b"C:\\Users\\x\\", b"b2://x"):
        assert scan.scan_stream(_io.BytesIO(text), [], text=True) is not None


def test_a_site_package_is_taken_back_when_the_clip_records_a_definite_no(job):
    from blueprint_pipeline import company_share_permission as permission
    world = World(job)
    assert world.publish()["status"] == "published"
    revoked = {"consent_status": "revoked", "consent_revoked": True, "statement_version": SHARED}
    permission.record_live(job.capture, permission.capture_identity(job.capture), lambda _i: dict(revoked), NOW + 60)
    assert _read_route(job, world, now=NOW + 90) == "unavailable"
    assert _at(world, 120)["status"] == "refused" and len(world.retracted) == 1


def test_an_ordinary_description_never_trips_on_blueprints_own_readme(job):
    _write(job.capture / "pipeline/website_task_context.json",
           {"description": "Stack the boxes on the pallet and do not pass it on to the next shift",
            "capture_rights": {"consent_status": "granted", "consent_revoked": False, "statement_version": SHARED}})
    world = World(job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        assert "do not pass it on" in archive.read("README.txt").decode().lower()


def test_an_inside_job_keeps_its_relation_and_bounds_and_nothing_of_the_site():
    from blueprint_pipeline import scene_package_stage as stage_module
    criteria = {
        "mode": "jane_placeholder_shelf",
        "surface_target": {"shape": "flat_green_disc", "radius_m": 0.1, "surface_position_world_m": [1, 2, 3],
                           "support_prim_path": "/World/x/jane", "visible_label": "Jane's shelf",
                           "placement_relation": "inside", "success_rule": "subject_centre_within_radius",
                           "interior_bounds_world_m": {"minimum": [0, 0, 0], "maximum": [1, 1, 1]},
                           "maximum_tilt_rad": 0.2, "stable_seconds": 1.0},
        "temporal_invariants": {"forbidden_contact_classes": ["robot_background", "destination_background", "jane"]},
        "support": {"jane_house": 1}, "controls": {"x": 1}}
    kept, left_out = stage_module.neutral_criteria(criteria)
    target = kept["surface_target"]
    assert target["placement_relation"] == "inside" and target["success_rule"] == "subject_centre_within_radius"
    assert target["interior_bounds_world_m"] == {"minimum": [0, 0, 0], "maximum": [1, 1, 1]}
    assert kept["temporal_invariants"]["forbidden_contact_classes"] == ["robot_background", "destination_background"]
    assert "jane" not in json.dumps(kept).lower() and "shelf" not in json.dumps(kept).lower()
    assert left_out >= 6


def test_a_large_layer_with_a_sublayer_is_walked(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "SPEC_WALK_BYTES", 0)
    _stage(tmp_path / "under.usda", "Under")
    path = _stage(tmp_path / "layer.usdc", "Root", sublayer="./under.usda")
    layer = Sdf.Layer.FindOrOpen(str(path))
    layer.subLayerOffsets[0] = Sdf.LayerOffset(3.0, 1.0)
    layer.Save()
    assert scan._usd_text_finding(path, scan.literal_forms([(REQUEST, "a job id")])) is None
    planted = _stage(tmp_path / "planted.usdc", "Root", sublayer=f"./{REQUEST}.usda")
    assert scan._usd_text_finding(planted, scan.literal_forms([(REQUEST, "a job id")]))[0] == "a job id"


@pytest.mark.parametrize("kind", ["string[]", "token[]"])
def test_the_spec_walk_reads_string_and_token_arrays(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(scan, "SPEC_WALK_BYTES", 0)
    path = tmp_path / "layer.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    value_type = Sdf.ValueTypeNames.StringArray if kind == "string[]" else Sdf.ValueTypeNames.TokenArray
    root.CreateAttribute("names", value_type).Set(["a", "b", REQUEST.replace("-", "_")])
    stage.GetRootLayer().Save()
    found = scan._usd_text_finding(path, scan.literal_forms([(REQUEST, "a job id")]))
    assert found is not None and found[0] == "a job id"


@pytest.mark.parametrize("event", ["takedown", "revoked_recorded"])
def test_an_event_during_the_build_stops_the_upload(tmp_path, monkeypatch, event):
    from blueprint_pipeline import company_share_permission as permission
    job = Job(tmp_path)
    world = World(job)
    real = sim.build_package

    def during(*args, **kwargs):
        built = real(*args, **kwargs)
        if event == "takedown":
            _write(job.capture / "pipeline/consent_takedown/takedown_manifest.json", {"takedown_open": True})
        else:
            permission.record_live(job.capture, permission.capture_identity(job.capture),
                                   lambda _i: {"consent_status": "revoked", "consent_revoked": True,
                                               "statement_version": SHARED}, NOW)
        return built
    monkeypatch.setattr(sim, "build_package", during)
    result = world.publish()
    assert result["status"] in {"refused", "not_permitted"}, result
    assert world.archived == [] and world.delivered == []


# --- a reused half: the worker writes ``already_compiled`` for a duplicate request or a retry --------------------

def _set_outcome(job: Job, outcome: str, **changes) -> None:
    path = job.queue / "status/31e12dca983b8837295ac911c44dee68.json"
    status = json.loads(path.read_text())
    status.update(outcome=outcome, **changes)
    path.write_text(json.dumps(status))


def test_an_already_compiled_half_is_published(job):
    _set_outcome(job, "already_compiled")
    world = World(job)
    assert world.publish()["status"] == "published" and len(world.delivered) == 1


@pytest.mark.parametrize("broken", ["missing", "no_plan"])
def test_an_already_compiled_status_naming_a_missing_or_invalid_half_is_no_source(job, broken):
    _set_outcome(job, "already_compiled")
    if broken == "missing":
        _set_outcome(job, "already_compiled", team_scene_inputs_path=str(job.tmp / "nowhere/scene/x.json"))
    else:
        (job.packet / sim.PLAN_FILE).unlink()
    world = World(job)
    assert world.publish() == {"status": "no_source"} and world.archived == []


@pytest.mark.parametrize("outcome", ["compiled", "already_compiled"])
def test_a_reused_half_whose_file_changed_is_refused_exactly_as_a_compiled_one(job, outcome):
    # The half is checked the same way whatever the outcome word: a file that no longer matches its plan row
    # is refused, naming the row's role; nothing is uploaded and nothing raises.
    _set_outcome(job, outcome)
    job.collision.write_bytes(b"changed after the plan was sealed")
    world = World(job)
    assert world.publish() == {"status": "refused", "reason": "website_company_simulation_asset_not_shipped:"
                               "scene_collision", "detail": ""}
    assert world.archived == []


@pytest.mark.parametrize("outcome", ["skipped", "failed", "compiling", "", None])
def test_an_outcome_that_is_not_a_success_is_ignored(job, outcome):
    _set_outcome(job, outcome)
    assert World(job).publish() == {"status": "no_source"}


# --- the closed list of file kinds, and identity in parsed text only (round 4) -------------------------------

PLANTED = (b"// made from gs://bucket-x/capture/scan and /tmp/work/look, contact jane@acmemail.example, "
           b"ftp://files.example.org/x\n")


@pytest.mark.parametrize("suffix", [".mdl", ".gltf", ".exr", ".hdr", ".tga", ".ply", ".mtlx"])
def test_a_scene_depending_on_a_kind_not_on_the_list_is_refused_for_its_kind(job, suffix):
    (job.assets / f"textures/look{suffix}").write_bytes(PLANTED)
    stage = Usd.Stage.Open(str(job.appearance))
    shader = stage.DefinePrim("/Appearance/Look/Mdl", "Shader")
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set(f"./textures/look{suffix}")
    stage.GetRootLayer().Save()
    job.seal()
    world = World(job)
    result = world.publish()
    assert result == {"status": "refused", "reason": "website_company_simulation_file_kind_not_shipped",
                      "detail": f"scene_appearance {suffix}"}
    assert world.archived == [] and not _package(job).exists()


def test_a_png_that_is_not_a_png_is_refused_for_its_kind(job):
    (job.assets / "textures/wood.png").write_bytes(PLANTED)
    job.seal()
    result = World(job).publish()
    assert result["reason"] == "website_company_simulation_file_kind_not_shipped"
    assert result["detail"] == "scene_appearance .png"


def _random_png(size: int) -> bytes:
    """A real PNG whose pixel data is random (as compressed image data is): about ``size`` bytes."""
    import random
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    side = int((size / 3) ** 0.5)
    rng = random.Random(11)
    rows = b"".join(b"\x00" + rng.randbytes(side * 3) for _ in range(side))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 1)) + chunk(b"IEND", b""))


@pytest.mark.parametrize(("name", "address"), [("Bob", None), ("York", None), ("Bob", "Unit 1, Mill Lane, York")])
def test_short_or_common_identity_values_never_refuse_a_clean_scene(job, name, address):
    descriptor = json.loads((job.capture / "capture_descriptor.json").read_text())
    descriptor["metadata"]["site_identity"] = {"name": name, **({"address": address} if address else {})}
    _write(job.capture / "capture_descriptor.json", descriptor)
    _write(job.assets / "textures/wood.png", _random_png(12 << 20))
    job.seal()
    assert World(job).publish()["status"] == "published"


def test_a_text_member_with_an_odd_ending_inside_a_usdz_is_read_and_never_ships(job, tmp_path):
    layer = _stage(tmp_path / "obj.usdc", "Object")
    job.object.unlink()
    with zipfile.ZipFile(job.object, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(layer, "obj.usdc")
        archive.writestr("notes.cfgx", "made at gs://bucket-x/scan\n")
    job.seal()
    world = World(job)
    result = world.publish()
    # A usdz is rebuilt from USD layers and images only: a text member is refused for its kind.
    assert result == {"status": "refused", "reason": "website_company_simulation_file_kind_not_shipped",
                      "detail": "task_object a usdz member that is not a USD layer, a PNG or a JPEG"}
    assert world.archived == []
    # The scan, used as the proof, reads it too.
    assert scan.scan_file(job.object, "object.usdz", [], closed=True).kind == scan.NOT_SHIPPED


def test_a_binary_member_of_another_kind_inside_a_usdz_never_ships(job, tmp_path):
    layer = _stage(tmp_path / "obj.usdc", "Object")
    job.object.unlink()
    with zipfile.ZipFile(job.object, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(layer, "obj.usdc")
        archive.writestr("look.exr", b"v/1\x01\x00\x00\x00" + PLANTED)
    job.seal()
    world = World(job)
    result = world.publish()
    assert result == {"status": "refused", "reason": "website_company_simulation_file_kind_not_shipped",
                      "detail": "task_object a usdz member that is not a USD layer, a PNG or a JPEG"}
    assert world.archived == []


# --- round 4: nothing is copied, everything is written; the tests read the PUBLISHED zip ------------------------

R4_VALUES = (b"gs://bucket-x", b"/tmp/work", b"jane@acmemail.example", b"req-1234")
R4_PLANT = b" gs://bucket-x/obj /tmp/work/x jane@acmemail.example req-1234 "


def _png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _plain_png(extra: bytes = b"", after_iend: bytes = b"") -> bytes:
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + extra
            + _png_chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + _png_chunk(b"IEND", b"") + after_iend)


def _jpeg(*, app1: bytes = b"", after_sos_com: bool = False, after_eoi: bytes = b"") -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (120, 80, 40)).save(buffer, "JPEG", quality=80)
    data = buffer.getvalue()
    if app1:
        segment = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
        data = data[:2] + segment + data[2:]
    if after_sos_com:  # a COM segment between the entropy-coded data and EOI
        com = b"\xff\xfe" + struct.pack(">H", len(R4_PLANT) + 2) + R4_PLANT
        data = data[:-2] + com + data[-2:]
    return data + after_eoi


def _texture_job(job: Job, name: str, data: bytes) -> None:
    _write(job.assets / f"textures/{name}", data)
    job.appearance = _stage(job.appearance, "Appearance", texture=f"./textures/{name}")
    job.seal()


def _published_and_clean(job: Job, *, image: str | None = None) -> None:
    world = World(job)
    result = world.publish()
    assert result["status"] == "published", result
    _assert_absent_from_zip(job, R4_VALUES)
    out = job.tmp / "published"
    with zipfile.ZipFile(_package(job)) as archive:
        archive.extractall(out)
    assert UsdUtils.ComputeAllDependencies(str(out / "scene.usda"))[2] == []
    assert Usd.Stage.Open(str(out / "scene.usda")).GetPrimAtPath("/World/Object").IsValid()
    if image:
        from PIL import Image
        (picture,) = [path for path in (out / "scene/files").iterdir() if path.suffix == image]
        with Image.open(picture) as opened:
            opened.load()


@pytest.mark.parametrize("carrier", ["after_iend", "private_chunk", "iccp", "text_chunk"])
def test_a_png_is_written_from_its_kept_chunks_only(job, carrier):
    extra, after = b"", b""
    if carrier == "after_iend":
        after = R4_PLANT
    elif carrier == "private_chunk":
        extra = _png_chunk(b"prVt", R4_PLANT)
    elif carrier == "iccp":
        extra = _png_chunk(b"iCCP", b"jane@acmemail.example\x00\x00" + zlib.compress(b"\x00" * 64 + R4_PLANT))
    else:
        extra = _png_chunk(b"tEXt", b"Comment\x00" + R4_PLANT)
    _texture_job(job, "wood.png", _plain_png(extra, after))
    _published_and_clean(job, image=".png")


def test_png_magic_then_text_is_refused(job):
    _texture_job(job, "wood.png", b"\x89PNG\r\n\x1a\n" + R4_PLANT)
    result = World(job).publish()
    assert result == {"status": "refused", "reason": "website_company_simulation_file_kind_not_shipped",
                      "detail": "scene_appearance a PNG that cannot be read chunk by chunk"}


@pytest.mark.parametrize("bad", ["crc", "critical", "no_iend", "ihdr_not_first"])
def test_a_png_that_cannot_be_read_chunk_by_chunk_is_refused(job, bad):
    good = _plain_png()
    if bad == "crc":
        data = good[:-5] + bytes([good[-5] ^ 1]) + good[-4:]
    elif bad == "critical":
        data = _plain_png(_png_chunk(b"ABCD", b"x"))
    elif bad == "no_iend":
        data = good[:-12]
    else:
        data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"gAMA", b"\x00\x00\xb1\x8f") + good[8:]
    _texture_job(job, "wood.png", data)
    assert World(job).publish()["detail"] == "scene_appearance a PNG that cannot be read chunk by chunk"


@pytest.mark.parametrize("carrier", ["after_eoi", "com_after_sos", "app1_exif"])
def test_a_jpeg_is_written_from_its_kept_segments_only(job, carrier):
    data = _jpeg(after_eoi=R4_PLANT if carrier == "after_eoi" else b"", after_sos_com=carrier == "com_after_sos",
                 app1=b"Exif\x00\x00" + R4_PLANT if carrier == "app1_exif" else b"")
    _texture_job(job, "wood.jpg", data)
    _published_and_clean(job, image=".jpg")


def test_the_adobe_app14_is_kept_and_other_appn_dropped():
    from blueprint_pipeline import outbound_file_writer as writer
    data = _jpeg()
    adobe = b"\xff\xee" + struct.pack(">H", 14) + b"Adobe\x00\x64\x00\x00\x00\x00\x01"
    other = b"\xff\xed" + struct.pack(">H", len(R4_PLANT) + 2) + R4_PLANT
    written = writer.jpeg_bytes(data[:2] + adobe + other + data[2:])
    assert adobe in written and R4_PLANT not in written


def test_text_in_a_jpeg_coding_table_is_numeric_payload_and_is_not_searched():
    # The stated limit: a DQT body is a coding table, written as read and not searched as text. A table that is not
    # a valid table does not decode, so the JPEG is refused; a valid table holding chosen bytes would ship.
    from blueprint_pipeline import outbound_file_writer as writer
    data = _jpeg()
    dqt = b"\xff\xdb" + struct.pack(">H", 67) + b"\x00" + (b"jane@acme" * 8)[:64]
    written = writer.jpeg_bytes(data[:2] + dqt + data[2:])
    assert b"jane@acme" in written  # written as read: numeric payload is not searched


def test_a_usdc_with_bytes_after_the_crate_is_written_without_them(job):
    job.appearance.write_bytes(job.appearance.read_bytes() + R4_PLANT)
    job.seal()
    _published_and_clean(job)


def test_a_usda_comment_is_not_written(job):
    job.appearance.unlink()
    job.appearance = _stage(job.assets / "appearance.usda", "Appearance", texture="./textures/wood.png")
    text = job.appearance.read_text()
    job.appearance.write_text(text.replace("#usda 1.0\n", "#usda 1.0\n# made from gs://bucket-x /tmp/work "
                                           "jane@acmemail.example req-1234\n", 1))
    job.seal()
    _published_and_clean(job)


def _object_usdz(job: Job, tmp_path: Path, *, members=None, comment: bytes = b"", extra: bytes = b"") -> None:
    layer = _stage(tmp_path / "obj-src.usdc", "Object")
    job.object.unlink()
    with zipfile.ZipFile(job.object, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("obj.usdc", date_time=(1980, 1, 1, 0, 0, 0))
        info.extra = extra
        archive.writestr(info, layer.read_bytes() + (R4_PLANT if members == "trailing" else b""))
        for name, data in (members if isinstance(members, list) else []):
            archive.writestr(name, data)
        archive.comment = comment
    job.seal()


@pytest.mark.parametrize("carrier", ["comment", "extra_field", "trailing"])
def test_a_usdz_is_rebuilt_without_its_comment_extra_fields_or_trailing_bytes(job, tmp_path, carrier):
    _object_usdz(job, tmp_path, comment=R4_PLANT if carrier == "comment" else b"",
                 extra=struct.pack("<HH", 0xCAFE, len(R4_PLANT)) + R4_PLANT if carrier == "extra_field" else b"",
                 members="trailing" if carrier == "trailing" else None)
    _published_and_clean(job)
    with zipfile.ZipFile(_package(job)) as archive:
        data = archive.read("object.usdz")
    with zipfile.ZipFile(io.BytesIO(data)) as inner:
        assert inner.comment == b"" and inner.infolist()[0].filename == "obj.usdc"
        for info in inner.infolist():
            assert info.extra in (b"",) or info.extra[:2] == struct.pack("<H", 0x1986)
            assert set(info.extra[4:]) <= {0}
            start = info.header_offset + 30 + len(info.filename) + len(info.extra)
            assert start % 64 == 0 and info.date_time == (1980, 1, 1, 0, 0, 0)


@pytest.mark.parametrize("name", ["gs:__bucket-x_jane@acmemail.example.png", "/tmp/work/x.png", "../../escape.png",
                                  "C:/x.png", "a\\b.png"])
def test_a_usdz_member_name_that_holds_a_value_or_is_not_relative_is_refused(job, tmp_path, name):
    _object_usdz(job, tmp_path, members=[(name, _plain_png())])
    world = World(job)
    result = world.publish()
    assert result["status"] == "refused" and world.archived == []
    assert result["detail"] in {"task_object a usdz member name that is not a plain relative path",
                                "object.usdz holds an email address inside a zip member",
                                "object.usdz holds a storage link inside a zip member"}, result


def test_a_rebuilt_usdz_opens_and_every_member_resolves(tmp_path):
    from blueprint_pipeline import outbound_file_writer as writer
    root = tmp_path / "src"
    _write(root / "tex.png", _plain_png())
    stage = Usd.Stage.CreateNew(str(root / "obj.usda"))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Object").GetPrim())
    shader = UsdShade.Shader.Define(stage, "/Object/Tex")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./tex.png")
    stage.GetRootLayer().Save()
    source = tmp_path / "in.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(root / "obj.usda"), str(source))
    target = tmp_path / "out.usdz"
    writer.rewrite_usdz(source, target)
    opened = Usd.Stage.Open(str(target))
    assert opened.GetPrimAtPath("/Object").IsValid()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(target))
    assert unresolved == [] and any(str(asset).endswith("tex.png]") for asset in assets)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist()[0] == "obj.usda"
    again = tmp_path / "again.usdz"
    writer.rewrite_usdz(source, again)
    assert again.read_bytes() == target.read_bytes()  # deterministic


def test_the_walk_reads_a_newline_inside_a_dictionary_value(tmp_path, monkeypatch):
    monkeypatch.setattr(scan, "SPEC_WALK_BYTES", 0)
    path = tmp_path / "layer.usdc"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    root.SetCustomDataByKey("note", "made by\nZebra Plant Foods\nlast week")
    stage.GetRootLayer().Save()
    needles = scan.site_identity(["Zebra Plant Foods"])
    walked = scan._usd_text_finding(path, needles)
    monkeypatch.setattr(scan, "SPEC_WALK_BYTES", 1 << 40)
    exported = scan._usd_text_finding(path, needles)
    assert walked is not None and exported is not None and walked[0] == exported[0] == scan.SITE_IDENTITY


def test_the_raw_set_finds_links_and_long_paths_in_binary_bytes():
    import io as _io
    for planted in (b"gs://bucket-x/y", b"https://host.example/x", b"omniverse://srv/x", b"/var/lib/blueprint/x",
                    b"/" + b"home/someone/x"):
        assert scan.scan_stream(_io.BytesIO(b"\x00\x01" + planted + b"\x02"), [], text=False) is not None, planted
    for short in (b"/tmp/work/x", b"~/x", b"jane@acmemail.example", b"10.1.2.3"):
        assert scan.scan_stream(_io.BytesIO(b"\x00\x01" + short + b"\x02"), [], text=False) is None, short


# --- round 5: kept PNG chunks are bounded; fixed link schemes are found whatever byte precedes them ---------------

def _ihdr(colour: int = 2, depth: int = 8, interlace: int = 0, width: int = 1, height: int = 1) -> bytes:
    return _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, interlace))


def _png_with(chunks: bytes, *, colour: int = 2, depth: int = 8, pixel: bytes = b"\x00\x80\x80\x80",
              idat: bytes | None = None) -> bytes:
    data = zlib.compress(pixel) if idat is None else idat
    return b"\x89PNG\r\n\x1a\n" + _ihdr(colour, depth) + chunks + _png_chunk(b"IDAT", data) + _png_chunk(b"IEND", b"")


OVERSIZED = {
    "gAMA": lambda: _png_with(_png_chunk(b"gAMA", b"\x00\x00\xb1\x8f" + R4_PLANT)),
    "cHRM": lambda: _png_with(_png_chunk(b"cHRM", b"\x00" * 32 + R4_PLANT)),
    "sRGB": lambda: _png_with(_png_chunk(b"sRGB", b"\x00" + R4_PLANT)),
    "pHYs": lambda: _png_with(_png_chunk(b"pHYs", b"\x00" * 9 + R4_PLANT)),
    "sBIT": lambda: _png_with(_png_chunk(b"sBIT", b"\x08\x08\x08" + R4_PLANT)),
    "sBIT_short": lambda: _png_with(_png_chunk(b"sBIT", b"\x08")),
    "PLTE_not_multiple_of_3": lambda: _png_with(_png_chunk(b"PLTE", b"\x00" * 3 + R4_PLANT[:4]), colour=3,
                                                pixel=b"\x00\x00"),
    "PLTE_over_bit_depth": lambda: _png_with(_png_chunk(b"PLTE", b"\x00" * 9), colour=3, depth=1, pixel=b"\x00\x00"),
    "PLTE_on_greyscale": lambda: _png_with(_png_chunk(b"PLTE", b"\x00" * 3), colour=0, pixel=b"\x00\x00"),
    "tRNS_over_palette": lambda: _png_with(_png_chunk(b"PLTE", b"\x00" * 6) + _png_chunk(b"tRNS", b"\x00" * 3),
                                           colour=3, pixel=b"\x00\x00"),
    "tRNS_truecolour": lambda: _png_with(_png_chunk(b"tRNS", b"\x00" * 6 + R4_PLANT)),
    "tRNS_with_alpha": lambda: _png_with(_png_chunk(b"tRNS", b"\x00" * 2), colour=6,
                                         pixel=b"\x00\x80\x80\x80\x80"),
    "IHDR_long": lambda: (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
                                                            + R4_PLANT)
                          + _png_chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80")) + _png_chunk(b"IEND", b"")),
    "IEND_long": lambda: (_plain_png()[:-12] + _png_chunk(b"IEND", R4_PLANT)),
    "IDAT_trailing": lambda: _png_with(b"", idat=zlib.compress(b"\x00\x80\x80\x80") + R4_PLANT),
    "IDAT_unfinished": lambda: _png_with(b"", idat=zlib.compress(b"\x00\x80\x80\x80")[:-4]),
}


@pytest.mark.parametrize("case", sorted(OVERSIZED))
def test_a_kept_png_chunk_of_the_wrong_length_or_trailing_image_data_refuses(case):
    from blueprint_pipeline import outbound_file_writer as writer
    with pytest.raises(writer.WriteRefused, match="a PNG that cannot be read chunk by chunk"):
        writer.png_bytes(OVERSIZED[case]())


def test_an_oversized_kept_chunk_refuses_the_package(job):
    _texture_job(job, "wood.png", OVERSIZED["gAMA"]())
    world = World(job)
    assert world.publish() == {"status": "refused", "reason": "website_company_simulation_file_kind_not_shipped",
                               "detail": "scene_appearance a PNG that cannot be read chunk by chunk"}
    assert world.archived == []


def _real_images() -> dict[str, bytes]:
    import numpy as np
    from PIL import Image, ImageCms
    rng = np.random.default_rng(3)
    rgb = Image.fromarray(rng.integers(0, 255, (64, 48, 3), dtype=np.uint8), "RGB")
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    def encode(image, **options):
        buffer = io.BytesIO()
        image.save(buffer, "PNG", **options)
        return buffer.getvalue()
    palette = rgb.convert("P", palette=Image.ADAPTIVE, colors=16)
    interlaced = (b"\x89PNG\r\n\x1a\n" + _ihdr(2, 8, interlace=1) + _png_chunk(b"IDAT", zlib.compress(b"\x00\x10\x20\x30"))
                  + _png_chunk(b"IEND", b""))
    return {"icc": encode(rgb, icc_profile=icc),
            "16-bit": encode(Image.fromarray(rng.integers(0, 65535, (32, 32), dtype=np.uint16))),
            "interlaced": interlaced,
            "paletted_trns": encode(palette, transparency=3),
            "rgba": encode(rgb.convert("RGBA")),
            "with_gamma_and_phys": encode(rgb, dpi=(72, 72))}


@pytest.mark.parametrize("label", ["icc", "16-bit", "interlaced", "paletted_trns", "rgba", "with_gamma_and_phys"])
def test_real_pngs_are_written_with_identical_pixels(label):
    import numpy as np
    from PIL import Image
    from blueprint_pipeline import outbound_file_writer as writer
    data = _real_images()[label]
    written = writer.png_bytes(data)
    with Image.open(io.BytesIO(data)) as before, Image.open(io.BytesIO(written)) as after:
        assert before.mode == after.mode and before.size == after.size
        assert np.array_equal(np.asarray(before), np.asarray(after))
        assert "icc_profile" not in after.info


def test_a_real_paletted_texture_still_publishes(job):
    _texture_job(job, "wood.png", _real_images()["paletted_trns"])
    _published_and_clean(job, image=".png")


@pytest.mark.parametrize("scheme", [b"gs://bucket-x/y", b"s3://bucket-x/y", b"http://host.example/x",
                                    b"https://host.example/x", b"file:///tmp/x"])
@pytest.mark.parametrize("before", [b"A", b"7", b" ", b"\x00", b"\xff"])
def test_a_fixed_link_scheme_is_found_in_raw_bytes_whatever_byte_precedes_it(scheme, before):
    assert scan.scan_stream(io.BytesIO(b"\x01\x02" + before + scheme + b"\x03"), [], text=False) is not None


# --- round 6: the image data inflates to exactly the size its header calls for ---------------------------------

def _zlib_stored(*blocks: bytes) -> bytes:
    body = b"".join(bytes([1 if index == len(blocks) - 1 else 0]) + struct.pack("<HH", len(block), 0xFFFF ^ len(block))
                    + block for index, block in enumerate(blocks))
    return b"\x78\x01" + body + struct.pack(">I", zlib.adler32(b"".join(blocks)))


def test_image_data_with_text_after_the_rows_inside_one_stream_refuses_end_to_end(job):
    data = (b"\x89PNG\r\n\x1a\n" + _ihdr() + _png_chunk(b"IDAT", _zlib_stored(b"\x00\x80\x80\x80", R4_PLANT))
            + _png_chunk(b"IEND", b""))
    _texture_job(job, "wood.png", data)
    world = World(job)
    assert world.publish() == {"status": "refused", "reason": "website_company_simulation_file_kind_not_shipped",
                               "detail": "scene_appearance a PNG that cannot be read chunk by chunk"}
    assert world.archived == [] and not _package(job).exists()


@pytest.mark.parametrize("case", ["one_byte_short", "one_byte_long", "bad_pair", "bad_interlace", "bad_filter",
                                  "bomb"])
def test_image_data_that_is_not_exactly_the_image_size_or_an_undefined_header_refuses(case):
    from blueprint_pipeline import outbound_file_writer as writer
    if case == "one_byte_short":
        data = _png_with(b"", idat=zlib.compress(b"\x00\x80\x80"))
    elif case == "one_byte_long":
        data = _png_with(b"", idat=zlib.compress(b"\x00\x80\x80\x80\x80"))
    elif case == "bomb":
        data = _png_with(b"", idat=zlib.compress(b"\x00" * (64 << 20)))
    else:
        header = {"bad_pair": struct.pack(">IIBBBBB", 1, 1, 4, 2, 0, 0, 0),
                  "bad_interlace": struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 2),
                  "bad_filter": struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 1, 0)}[case]
        data = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(b"\x00\x80\x80\x80"))
                + _png_chunk(b"IEND", b""))
    with pytest.raises(writer.WriteRefused, match="a PNG that cannot be read chunk by chunk"):
        writer.png_bytes(data)


def _adam7_png(width: int, height: int, colour: int = 2, depth: int = 8) -> bytes:
    """A hand-built Adam7 image: each pass's rows (filter byte 0, then packed pixels of a varying value)."""
    from blueprint_pipeline import outbound_file_writer as writer
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colour]
    rows = bytearray()
    for index, (x, y, dx, dy) in enumerate(writer._ADAM7):
        columns = (width - x + dx - 1) // dx if width > x else 0
        count = (height - y + dy - 1) // dy if height > y else 0
        if not columns or not count:
            continue
        for row in range(count):
            rows += b"\x00" + bytes(((index * 37 + row * 11 + byte) % 251)
                                    for byte in range((columns * depth * channels + 7) // 8))
    header = struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, 1)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(bytes(rows)))
            + _png_chunk(b"IEND", b""))


def _round6_images() -> dict[str, bytes]:
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(5)

    def encode(image, **options):
        buffer = io.BytesIO()
        image.save(buffer, "PNG", **options)
        return buffer.getvalue()
    rgb = Image.fromarray(rng.integers(0, 255, (13, 7, 3), dtype=np.uint8), "RGB")
    images = {"16-bit": encode(Image.fromarray(rng.integers(0, 65535, (9, 5), dtype=np.uint16))),
              "rgba": encode(rgb.convert("RGBA")),
              "one_by_one": encode(Image.new("RGB", (1, 1), (1, 2, 3))),
              "grey_trns": encode(Image.fromarray(rng.integers(0, 255, (6, 11), dtype=np.uint8), "L"), transparency=5)}
    for bits, colours in ((1, 2), (2, 4), (4, 16), (8, 200)):
        palette = rgb.convert("P", palette=Image.ADAPTIVE, colors=colours)
        images[f"paletted_{bits}_bit"] = encode(palette, bits=bits, transparency=1)
    for width in range(1, 10):
        for height in (1, 3, 8, 9):
            images[f"adam7_{width}x{height}"] = _adam7_png(width, height)
    images["adam7_grey_1_bit_5x5"] = _adam7_png(5, 5, colour=0, depth=1)
    images["adam7_rgba_16_bit_6x7"] = _adam7_png(6, 7, colour=6, depth=16)
    return images


@pytest.mark.parametrize("label", ["16-bit", "rgba", "one_by_one", "grey_trns", "paletted_1_bit", "paletted_2_bit",
                                   "paletted_4_bit", "paletted_8_bit", "adam7_grey_1_bit_5x5", "adam7_rgba_16_bit_6x7",
                                   *[f"adam7_{w}x{h}" for w in range(1, 10) for h in (1, 3, 8, 9)]])
def test_real_and_interlaced_pngs_are_written_with_identical_pixels(label):
    import numpy as np
    from PIL import Image
    from blueprint_pipeline import outbound_file_writer as writer
    data = _round6_images()[label]
    written = writer.png_bytes(data)
    with Image.open(io.BytesIO(data)) as before, Image.open(io.BytesIO(written)) as after:
        assert before.mode == after.mode and before.size == after.size
        assert np.array_equal(np.asarray(before), np.asarray(after))


WITHHELD = "website_native_task_profile_preparation_required"


def _withhold(job: Job, *, reason: str | None = WITHHELD, sealed: bool = True, plan_rule: bool = False,
              record_rule: bool = False, task_object: bool = True, plan_digest: str | None = None,
              schema: str = "blueprint.team_scene_inputs.v1", support_held: bool = False) -> None:
    """The half the team-scene compile writes for a job no task profile models (``materialize_team_scene``'s
    ``rule_withheld``): no rule in its plan or its record, and its sealed record says why."""
    plan = json.loads((job.packet / sim.PLAN_FILE).read_text())
    if not plan_rule:
        plan["task_spec"].pop("task_success_contract")
    if not task_object:
        plan["objects"] = [row for row in plan["objects"] if row["semantic_role"] != "task_object"]
    _write(job.packet / sim.PLAN_FILE, plan)
    record = {"schema_version": schema, "plan_digest": plan_digest or plan["plan_digest"],
              **({"rule_withheld": {"reason": reason}} if reason is not None else {}),
              **({"task_success_contract": {"criteria": job.criteria}} if record_rule else {}),
              **({"task_support_pad": {"status": "held", "reasons": ["x"]}} if support_held else {})}
    record["inputs_digest"] = (canonical_digest(record, digest_field="inputs_digest") if sealed
                               else "sha256:" + "0" * 64)
    _write(job.scene / "team_scene_inputs.v1.json", record)


def test_a_job_no_profile_models_ships_its_object_with_no_rule_over_releases_and_withdrawal(job):
    """Objects step 1: published, then current; a release recompiles the half (new object digest): rebuilt once,
    then current; a withdrawal after delivery takes it back."""
    _withhold(job, support_held=True)  # its support note speaks of the rule: not said
    world = World(job)
    first = world.publish()
    assert first["status"] == "published", first
    names = zipfile.ZipFile(_package(job)).namelist()
    readme = zipfile.ZipFile(_package(job)).read("README.txt").decode()
    assert "success_rule.json" not in names
    assert "object.usdz" in names
    assert "scene.usda" in names
    assert " ".join(readme.split()).count(sim.NO_RULE) == 1 and "success_rule" not in readme
    assert "the rule for what counts" not in readme
    assert sim.SUPPORT_NOT_HELD.split(",")[0] not in readme
    assert "start position as estimated from the video" in readme
    for word in ("Do the", "destination", "place it"):
        assert word not in readme
    assert world.publish()["status"] == "current" and len(world.archived) == 1
    # A release: the team-scene worker recompiles the half, with new object bytes.
    job.object.unlink()
    job.object = _usdz(job.assets / "replacement.usdz", "ObjectRecompiled")
    job.seal()
    _withhold(job)
    second = world.publish()
    assert second["status"] == "published" and second["simulation_revision"] != first["simulation_revision"]
    assert world.publish()["status"] == "current" and len(world.archived) == 2
    _withdraw(job.capture)
    assert world.publish()["status"] == "withdrawn" and world.retracted


@pytest.mark.parametrize("half, reason", [
    ({"reason": None}, "success_rule_missing"),
    ({"sealed": False}, "success_rule_missing"),
    ({"reason": "website_some_other_gap"}, "success_rule_missing"),
    ({"plan_rule": True}, "rule_withheld_mismatch"),
    ({"record_rule": True}, "rule_withheld_mismatch"),
    ({"task_object": False}, "rule_withheld_mismatch"),
    ({"plan_digest": "sha256:" + "8" * 64}, "rule_withheld_mismatch"),
    ({"schema": sim.ROOM_RECORD_SCHEMA}, "room_record_mismatch"),
])
def test_a_half_without_its_rule_and_without_a_sealed_named_reason_is_refused(job, half, reason):
    """kinds v4's guard holds: a rigid half with no rule ships only with its object, its own plan and the sealed,
    named reason in its own record, and no rule anywhere; any other half with no rule is refused."""
    _withhold(job, **half)
    result = World(job).publish()
    assert (result["status"], result["reason"]) == ("refused", "website_company_simulation_" + reason), result


def test_the_withheld_reasons_are_the_handoffs_room_gaps_and_the_model_s_choice():
    from blueprint_pipeline.website_build_choice import RULE_WITHHELD_CODE
    from blueprint_pipeline.website_scene_handoff import ROOM_GAPS
    assert sim.RULE_WITHHELD_REASONS == ROOM_GAPS | {RULE_WITHHELD_CODE}


# --- the size limit is on the download ----------------------------------------------------------------------------

def _large_look(tmp_path: Path, points: int = 200_000) -> Job:
    """A job whose look is a large, very compressible text layer: far bigger as files than as a zip."""
    job = Job(tmp_path, appearance=".usda")
    stage = Usd.Stage.Open(str(job.appearance))
    UsdGeom.Points.Define(stage, "/Appearance/Splat").CreatePointsAttr([(0.0, 0.0, 0.0)] * points)
    stage.GetRootLayer().Save()
    job.seal()
    return job


def _sizes(job: Job, tmp_path: Path) -> tuple[int, int, int]:
    """The source files' bytes, the zip's with the look and the zip's without it."""
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    literals = sim.secret_literals(job.capture, source)
    whole = sim.build_package(source, tmp_path / "whole.zip", literals=literals, max_bytes=1 << 30)
    bare = sim.build_package(sim.without_look(source), tmp_path / "bare.zip", literals=literals, max_bytes=1 << 30)
    assert "look_left_out" not in whole
    return sum(row["size_bytes"] for row in source["files"].values()), whole["size_bytes"], bare["size_bytes"]


def test_the_limit_is_the_zips_and_the_files_are_refused_unbuilt_only_above_two_gib(tmp_path, monkeypatch):
    job = _large_look(tmp_path)
    raw, whole, _bare = _sizes(job, tmp_path / "sizes")
    assert whole < raw  # the files are over a limit the zip is under
    world = World(job)
    assert world.publish(max_bytes=whole)["status"] == "published"
    assert world.delivered[0]["size_bytes"] == whole and "scene/room.usda" in world.delivered[0]["files"]
    assert "look_left_out" not in _manifest(job)["package"]
    assert sim.MAX_SOURCE_BYTES == 2 * 1024 ** 3
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    monkeypatch.setattr(sim, "MAX_SOURCE_BYTES", raw - 1)
    monkeypatch.setattr(sim, "_build_once", lambda *a, **k: pytest.fail("built files above the source ceiling"))
    with pytest.raises(ValueError, match=sim.OVER_CAP):
        sim.build_package(source, tmp_path / "x.zip", literals=[], max_bytes=1 << 30)


def test_a_zip_over_the_limit_is_built_once_more_without_the_look_and_says_so(tmp_path):
    job = _large_look(tmp_path)
    _raw, whole, bare = _sizes(job, tmp_path / "sizes")
    assert bare < whole
    world = World(job)
    assert world.publish(max_bytes=bare)["status"] == "published"
    record, manifest = world.delivered[0], _manifest(job)
    assert record["size_bytes"] == bare and manifest["package"]["look_left_out"] is True
    assert "scene/room.usda" not in record["files"] and "scene/room_collision.usdc" in record["files"]
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(_package(job)) as archive:
        archive.extractall(unpacked)
    stage = Usd.Stage.Open(str(unpacked / "scene.usda"))
    assert not stage.GetPrimAtPath("/World/Room").IsValid() and stage.GetPrimAtPath("/World/Object").IsValid()
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/RoomCollision")).ComputeVisibility() == "inherited"
    readme = " ".join((unpacked / "README.txt").read_text().split())
    assert sim.LOOK_LEFT_OUT in readme and "Hidden in scene.usda" not in readme
    assert world.publish(max_bytes=bare)["status"] == "current"


def test_a_package_over_the_limit_without_its_look_too_is_over_the_cap(tmp_path, monkeypatch):
    job = _large_look(tmp_path)
    _raw, _whole, bare = _sizes(job, tmp_path / "sizes")
    builds = []
    real = sim._build_once
    monkeypatch.setattr(sim, "_build_once", lambda source, *a, **k: builds.append(bool(source.get("look_left_out")))
                        or real(source, *a, **k))
    world = World(job)
    assert world.publish(max_bytes=bare - 1) == {"status": "over_size_cap"}
    assert builds == [False, True] and _manifest(job)["size_rule"] == sim.SIZE_RULE
    assert world.publish(max_bytes=bare - 1) == {"status": "over_size_cap"} and builds == [False, True]
    assert world.delivered == []


def test_a_room_with_no_collision_to_show_is_not_built_without_its_look(job):
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    files = {prim: row for prim, row in source["files"].items() if row["role"] != "scene_collision"}
    assert sim.without_look({**source, "files": files}) is None
    assert sim.without_look(source)["files"].keys() == {"RoomCollision", "Object"}


def test_a_size_refusal_made_under_the_old_rule_is_built_once_more_at_no_cost(tmp_path):
    job = _large_look(tmp_path)
    _raw, whole, _bare = _sizes(job, tmp_path / "sizes")
    source = sim.package_source(job.capture, queue_root=job.queue)
    revision = sim.simulation_revision(source, max_bytes=whole)
    # Job 1's record as the earlier builder wrote it: the same revision, refused for its files' size an hour ago,
    # with no ``size_rule`` and no ``source_kind``.
    old = {"schema_version": sim.MANIFEST_SCHEMA, "request_id": REQUEST, "scene_id": SITE, "capture_id": CAPTURE,
           "simulation_revision": revision, "archive": None, "delivered": None, "package": None,
           "refused": sim.OVER_CAP, "refused_at_epoch": NOW - 3600}
    _write(job.capture.joinpath(*sim.DIRECTORY) / sim.MANIFEST_FILE, old)
    world = World(job)
    assert world.publish(max_bytes=whole)["status"] == "published"  # at once, not after the window
    assert _manifest(job)["size_rule"] == sim.SIZE_RULE and len(world.delivered) == 1
    assert world.publish(max_bytes=whole)["status"] == "current"


def test_a_half_refused_for_size_under_the_old_rule_is_tried_again_unless_its_floor_was_delivered(job, monkeypatch):
    monkeypatch.setattr(sim, "_floor_source", lambda *a, **k: {"source_kind": "floor"})
    half = sim.simulation_source(job.capture, queue_root=job.queue)
    revision = sim.simulation_revision(half)
    old = {"refused_half": {"simulation_revision": revision, "reason": sim.OVER_CAP}}
    assert sim.package_source(job.capture, queue_root=job.queue, manifest=old)["source_kind"] == "half"
    delivered = {**old, "delivered": {"simulation_digest": "sha256:" + "1" * 64}, "package": {"digest": "x"},
                 "source_kind": "floor"}
    assert sim.package_source(job.capture, queue_root=job.queue, manifest=delivered)["source_kind"] == "floor"
    new = {"refused_half": {**old["refused_half"], "size_rule": sim.SIZE_RULE}}
    assert sim.package_source(job.capture, queue_root=job.queue, manifest=new)["source_kind"] == "floor"


def test_the_limit_is_unchanged_and_a_delivered_package_is_not_rebuilt(job, monkeypatch):
    assert sim.MAX_PACKAGE_BYTES == 512 * 1024 * 1024
    world = World(job)
    assert world.publish()["status"] == "published"
    manifest = _manifest(job)
    manifest.pop("size_rule")  # delivered by the earlier builder, which recorded no size rule
    _write(job.capture.joinpath(*sim.DIRECTORY) / sim.MANIFEST_FILE, manifest)
    monkeypatch.setattr(sim, "build_package", lambda *a, **k: pytest.fail("rebuilt a delivered package"))
    assert world.publish()["status"] == "current" and len(world.delivered) == 1 and world.retracted == []


def test_a_package_without_its_look_says_nothing_of_the_look(tmp_path):
    job = _large_look(tmp_path)
    _raw, _whole, bare = _sizes(job, tmp_path / "sizes")
    source = {**sim.simulation_source(job.capture, queue_root=job.queue), "removed_items": 2,
              "appearance_quality": "not_passed", "appearance_representation": "particlefield_3d_gaussian_splat",
              "generated_patch": "look_and_collision"}
    built = sim.build_package(source, tmp_path / "bare.zip", literals=[], max_bytes=bare + 4096)
    assert built["look_left_out"] is True
    with zipfile.ZipFile(tmp_path / "bare.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
    assert sim.LOOK_LEFT_OUT in readme and sim.FURNITURE_REMOVED_NO_LOOK.format(count=2) in readme
    assert sim.APPEARANCE_QUALITY_NOT_PASSED not in readme and "it still appears in the look file" not in readme
    # Admitted generated geometry is still named, by the collision file alone.
    assert sim.GENERATED_NO_LOOK_NOTE in readme and sim.GENERATED_LOOK_AND_COLLISION_NOTE not in readme
    assert "Gaussian" not in readme and "how the site looks" not in readme and "Hidden in scene.usda" not in readme


def test_a_timeout_during_the_build_without_the_look_is_recorded_and_waits_its_window(tmp_path, monkeypatch):
    job = _large_look(tmp_path)
    _raw, _whole, bare = _sizes(job, tmp_path / "sizes")
    builds = []
    real = sim._build_once

    def build(source, *a, **k):
        builds.append(bool(source.get("look_left_out")))
        if source.get("look_left_out"):
            raise ValueError(sim.BUILD_TIMED_OUT)
        return real(source, *a, **k)
    monkeypatch.setattr(sim, "_build_once", build)
    world = World(job)
    assert world.publish(max_bytes=bare) == {"status": "build_timed_out"} and builds == [False, True]
    manifest = _manifest(job)
    assert manifest["refused"] == sim.BUILD_TIMED_OUT and manifest["refused_half"]["reason"] == sim.BUILD_TIMED_OUT
    assert world.publish(max_bytes=bare) == {"status": "build_timed_out"} and builds == [False, True]
    assert world.delivered == [] and not _package(job).exists()


@pytest.mark.parametrize("where", [None, "collision_only", "look_and_collision", "unsealed", "other_basis"])
def test_admitted_generated_geometry_is_named_where_it_is_and_nothing_else_changes(job, tmp_path, where):
    """Owner, 2026-11-17: the README says where admitted generated geometry is: only in the scan the collision
    file was made from, or in the look as well (a job whose refined splat could not be had, or one recorded
    before the marker). A base scene with none says what it said before, at the same revision."""
    before = sim.simulation_source(job.capture, queue_root=job.queue)
    preparation = job.capture / "pipeline" / "website_scene_preparation"
    if where is not None:
        provenance = {"basis": "captured_scene_with_bounded_generated_estimates" if where != "other_basis"
                      else "something_else", "completion_admission_digest": "sha256:" + "3" * 64,
                      "generated_geometry_is_estimate": True,
                      **({"generated_geometry_in": "collision_only"} if where == "collision_only" else {})}
        _write(preparation / "base_scene.json", {"content_provenance": provenance})
        handoff = {"status": "intake_ready", "base_scene_path": str(preparation / "base_scene.json")}
        handoff["digest"] = canonical_digest(handoff, digest_field="digest")
        if where == "unsealed":
            handoff["digest"] = "sha256:" + "0" * 64
        _write(preparation / "handoff.json", handoff)
    expected = where if where in ("collision_only", "look_and_collision") else None
    assert sim.generated_patch(job.capture) == expected
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert source.get("generated_patch") == expected
    assert (sim.simulation_revision(source) == sim.simulation_revision(before)) is (expected is None)
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
    assert (sim.GENERATED_COLLISION_ONLY_NOTE in readme) is (expected == "collision_only")
    assert (sim.GENERATED_LOOK_AND_COLLISION_NOTE in readme) is (expected == "look_and_collision")
    assert sim.GENERATED_NO_LOOK_NOTE not in readme


@pytest.mark.parametrize("where", [None, "look", "unsealed", "other"])
def test_generated_views_in_the_look_are_named_in_one_sentence_only_when_the_look_holds_them(job, tmp_path, where):
    """Owner, 2026-11-18: the look may hold generated views of areas the video saw from few angles,
    checked against the filmed frames and their depth (``atlas_look_views``). The README says so in one
    sentence only when the base scene the handoff sealed records them; otherwise the package is unchanged."""
    before = sim.simulation_source(job.capture, queue_root=job.queue)
    preparation = job.capture / "pipeline" / "website_scene_preparation"
    if where is not None:
        _write(preparation / "base_scene.json", {"look_provenance": {
            "generated_views_in": "look" if where != "other" else "collision_only",
            "look_views_digest": "sha256:" + "5" * 64, "view_count": 3}})
        handoff = {"status": "intake_ready", "base_scene_path": str(preparation / "base_scene.json")}
        handoff["digest"] = canonical_digest(handoff, digest_field="digest")
        if where == "unsealed":
            handoff["digest"] = "sha256:" + "0" * 64
        _write(preparation / "handoff.json", handoff)
    expected = where == "look"
    assert sim.look_views(job.capture) is expected
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert bool(source.get("look_views")) is expected and source.get("generated_patch") is None
    assert (sim.simulation_revision(source) == sim.simulation_revision(before)) is (not expected)
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
    assert (sim.GENERATED_VIEWS_LOOK_NOTE in readme) is expected
    assert readme.count("generated views") == (1 if expected else 0)
    # A package built without the look says nothing of it.
    sim.build_package(sim.without_look(source), tmp_path / "bare.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "bare.zip") as archive:
        assert sim.GENERATED_VIEWS_LOOK_NOTE not in " ".join(archive.read("README.txt").decode().split())


@pytest.mark.parametrize("where", [None, "estimated", "unsealed", "other_code"])
def test_an_estimated_destination_surface_is_named_in_one_sentence_only_when_the_scan_holds_it(job, tmp_path, where):
    """A destination surface the scan did not show (glass) is estimated from its visible edges and added to the
    collider only (``website_unscanned_surface``). The README says so in one sentence when the base scene the handoff
    sealed records it; otherwise the package is unchanged, at the same revision."""
    from blueprint_pipeline.atlas_scene_estimates import DESTINATION_SURFACE_ESTIMATED
    before = sim.simulation_source(job.capture, queue_root=job.queue)
    preparation = job.capture / "pipeline" / "website_scene_preparation"
    if where is not None:
        code = DESTINATION_SURFACE_ESTIMATED if where != "other_code" else "scene_estimate:floor"
        _write(preparation / "base_scene.json", {"scene_estimates": [{"code": code, "source": "test", "evidence": {}}]})
        handoff = {"status": "intake_ready", "base_scene_path": str(preparation / "base_scene.json")}
        handoff["digest"] = canonical_digest(handoff, digest_field="digest")
        if where == "unsealed":
            handoff["digest"] = "sha256:" + "0" * 64
        _write(preparation / "handoff.json", handoff)
    expected = where == "estimated"
    assert sim.surface_estimated(job.capture) is expected
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert bool(source.get("surface_estimated")) is expected
    assert (sim.simulation_revision(source) == sim.simulation_revision(before)) is (not expected)
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
    assert (sim.SURFACE_ESTIMATED_NOTE in readme) is expected and sim.SURFACE_ESTIMATED_NO_LOOK_NOTE not in readme
    assert readme.count("estimated from its visible edges") == (1 if expected else 0)
    if expected:
        assert "glass" in sim.SURFACE_ESTIMATED_NOTE and "does not show in the scan" not in readme
        assert "not show in the scan" in readme and "not in the look" in readme
    # A package built without the look names the collision file alone.
    sim.build_package(sim.without_look(source), tmp_path / "bare.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "bare.zip") as archive:
        bare = " ".join(archive.read("README.txt").decode().split())
    assert (sim.SURFACE_ESTIMATED_NO_LOOK_NOTE in bare) is expected and sim.SURFACE_ESTIMATED_NOTE not in bare


@pytest.mark.parametrize("where", [None, "not_admitted", "unsealed", "other_flag"])
def test_gap_views_not_admitted_are_named_in_a_package_with_the_job_s_object(job, tmp_path, where):
    """A completion whose generated gap views were not admitted (``website_atlas_pipeline.SCENE_COMPLETION_NOT_ADMITTED``:
    a captured object's job, move-the-lamp-2, or a preserved one that builds an object) leaves the areas the video did
    not show unfilled. The scene source names it on the base scene; the README says so once, with or without the look.
    A room package says it from its record's flags instead (``test_website_scene_completion_floor``)."""
    before = sim.simulation_source(job.capture, queue_root=job.queue)
    preparation = job.capture / "pipeline" / "website_scene_preparation"
    if where is not None:
        flag = "scene_completion_not_admitted" if where != "other_flag" else "something_else"
        _write(preparation / "base_scene.json", {"scene_completion": {
            "generated_geometry_rejected": flag, "not_admitted_reason": "atlas_completion_visual_review_required"}})
        handoff = {"status": "intake_ready", "base_scene_path": str(preparation / "base_scene.json")}
        handoff["digest"] = canonical_digest(handoff, digest_field="digest")
        if where == "unsealed":
            handoff["digest"] = "sha256:" + "0" * 64
        _write(preparation / "handoff.json", handoff)
    expected = where == "not_admitted"
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert source.get("room") is None and bool(source.get("not_filled_in")) is expected
    assert (sim.simulation_revision(source) == sim.simulation_revision(before)) is (not expected)
    for built, name in ((source, "package.zip"), (sim.without_look(source), "bare.zip")):
        sim.build_package(built, tmp_path / name, literals=[])
        with zipfile.ZipFile(tmp_path / name) as archive:
            readme = " ".join(archive.read("README.txt").decode().split())
        assert readme.count(sim.NOT_FILLED_IN) == (1 if expected else 0)


@pytest.mark.parametrize("extent", [None, "from_its_own_points", "kept", "disagrees", "overhang", "box_meets_room"])
def test_an_object_sized_from_its_own_points_is_named_in_one_sentence_only_when_it_was(job, tmp_path, extent):
    """The object's size read again from its own points, its masks' spill left out (``website_object_extent``):
    the README says so in one sentence, and says when the exact model's published size disagrees with it, when
    it overhangs its destination (``website_destination_overhang``), and when its box meets the room wherever it would
    rest there (job move-the-lamp-2, placed all the same). A job whose box fit its masks and whose destination holds it
    records nothing and its package is unchanged, at the same revision."""
    from blueprint_pipeline.website_destination_unheld import BOX_MEETS_ROOM_NOTE
    before = sim.simulation_source(job.capture, queue_root=job.queue)
    preparation = job.capture / "pipeline" / "website_scene_preparation"
    if extent is not None:
        record = {"status": "kept" if extent in ("kept", "overhang", "box_meets_room") else "from_its_own_points",
                  **({"published_cross_check": {"status": "disagrees"}} if extent == "disagrees" else {})}
        _write(preparation / "preparation.json", {"subject_extent": record, **(
            {"destination_overhang": {"flag": "scene_estimate:destination_overhang"}} if extent == "overhang" else {}),
            **({"destination_box_meets_room": {"flag": "scene_estimate:destination_box_meets_room",
                                               "cause": "other_geometry_in_its_box_wherever_it_would_rest"}}
               if extent == "box_meets_room" else {})})
        handoff = {"status": "intake_ready", "preparation_path": str(preparation / "preparation.json")}
        handoff["digest"] = canonical_digest(handoff, digest_field="digest")
        _write(preparation / "handoff.json", handoff)
    expected = {"size_from_its_points": extent in ("from_its_own_points", "disagrees"),
                "size_disagrees_with_published": extent == "disagrees", "destination_overhang": extent == "overhang",
                "destination_box_meets_room": extent == "box_meets_room"}
    assert sim.size_notes(job.capture) == {key: True for key, value in expected.items() if value}
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert (sim.simulation_revision(source) == sim.simulation_revision(before)) is (not any(expected.values()))
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
    assert (sim.SIZE_FROM_ITS_POINTS_NOTE in readme) is expected["size_from_its_points"]
    assert (sim.SIZE_DISAGREES_WITH_PUBLISHED_NOTE in readme) is expected["size_disagrees_with_published"]
    assert readme.count("points of the object") == (1 if expected["size_from_its_points"] else 0)
    assert (sim.DESTINATION_OVERHANG_NOTE in readme) is expected["destination_overhang"]
    assert (" ".join(BOX_MEETS_ROOM_NOTE.split()) in readme) is expected["destination_box_meets_room"]
    assert "could not place" not in readme  # a package with the object never says it could not be placed


# --- a look written again from its source splat, and its PLY ------------------------------------------------------

#: The website look's frame (``website_native_appearance``): a levelled Y-up world, scaled and turned to Z-up, in
#: the writer's row-major form (its transpose is ``runtime_to_simulator``).
LOOK_TRANSFORM = [[0.8, 0.0, 0.0, 0.0], [0.0, 0.0, 0.8, 0.0], [0.0, -0.8, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _turns(count: int):
    """Known turns about oblique axes, never their own scrambled order, at varied norms."""
    import numpy as np

    axes = np.asarray([[1, 2, 3], [-2, 0.5, 1], [0.3, -1, 0.2], [0, 0, 1], [1, 1, 0]], float)
    angles = np.radians([37, 115, 170, 90, 63])
    known = np.column_stack([np.cos(angles / 2), (axes / np.linalg.norm(axes, axis=1, keepdims=True))
                             * np.sin(angles / 2)[:, None]])
    return np.resize(known, (count, 4)) * np.linspace(0.5, 2.0, count)[:, None]


def _splat_job(tmp_path: Path, *, count: int = 3000, old_writer: bool = False, receipt: bool = True) -> Job:
    """A job whose look the website authored from its levelled splat (``website_native_appearance``), the room
    package's way, so its sealed receipt names that source. ``old_writer``: the look and its receipt as the writer
    before 2026-11-18 left them, every rotation stored scrambled. ``receipt=False``: a splat look nothing names."""
    import numpy as np

    from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
    from blueprint_pipeline.particlefield_usd import write_particlefield_usd
    from blueprint_pipeline.website_native_appearance import prepare_native_appearance

    job = Job(tmp_path)
    job.appearance.unlink()
    rng = np.random.default_rng(count)
    splat = SplatData(count=count, xyz=rng.uniform([-4, 0, -4], [4, 3, 4], (count, 3)).astype(np.float32),
                      opacity=rng.normal(0, 3, count).astype(np.float32),
                      f_dc=rng.normal(0, 1, (count, 3)).astype(np.float32),
                      scales=np.tile(np.log([0.02, 0.05, 0.11]), (count, 1)).astype(np.float32),
                      quats=_turns(count).astype(np.float32), properties=())
    preparation_root = job.capture / "pipeline" / "website_scene_preparation"
    job.source = preparation_root / "atlas" / "atlas_depth_splat_levelled.ply"
    job.source.parent.mkdir(parents=True, exist_ok=True)
    write_standard_3dgs_ply(splat, job.source)
    job.splat = splat
    job.appearance = job.assets / "appearance.usdc"
    if not receipt:
        written = write_particlefield_usd(splat, job.appearance, layer_transform_row_major=LOOK_TRANSFORM)
        assert written["status"] == "completed"
    else:
        base = {"splat_path": str(job.source), "splat_digest": _sha(job.source)}
        _write(preparation_root / "base_scene.json", base)
        column = [list(row) for row in zip(*LOOK_TRANSFORM)]
        frame = {"binding": {"splat_digest": base["splat_digest"]},
                 "coordinate_frame": {"declared_meters_per_unit": 0.8, "declared_up_axis": "Y",
                                      "physical_scale_measured": False, "runtime_to_simulator": column}}
        frame["digest"] = canonical_digest(frame, digest_field="digest")
        authored = prepare_native_appearance(preparation=frame, base_scene=base,
                                             output_root=preparation_root / "room_package" / "appearance")
        artifact = Path(authored["artifact"]["path"])
        if old_writer:
            _as_the_old_writer_left_it(artifact, artifact.parent / "appearance.json")
        job.appearance.write_bytes(artifact.read_bytes())
    job.seal()
    plan = json.loads((job.packet / sim.PLAN_FILE).read_text())
    plan["appearance_frame_alignment"] = {"representation": "particlefield_3d_gaussian_splat"}
    _write(job.packet / sim.PLAN_FILE, plan)
    return job


def _as_the_old_writer_left_it(look: Path, receipt_path: Path) -> None:
    """The look's rotations stored as ``FromNumpy`` read (w, x, y, z) rows before 2026-11-18, and its receipt as
    that writer sealed it (no ``orientation_storage``)."""
    import numpy as np
    from pxr import Vt

    stage = Usd.Stage.Open(str(look))
    field = next(prim for prim in stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat")
    attribute = field.GetAttribute("orientations")
    wxyz = np.asarray([(q.GetReal(), *q.GetImaginary()) for q in attribute.Get()], dtype=np.float32)
    attribute.Set(Vt.QuatfArray.FromNumpy(np.ascontiguousarray(wxyz)))
    stage.GetRootLayer().Save()
    receipt = json.loads(receipt_path.read_text())
    authoring = {key: value for key, value in receipt["authoring"].items() if key != "orientation_storage"}
    authoring.update(output_sha256=_sha(look), output_bytes=look.stat().st_size)
    authoring["receipt_digest"] = canonical_digest(authoring, digest_field="receipt_digest")
    receipt.update(authoring=authoring, artifact={**receipt["artifact"], "digest": _sha(look),
                                                  "size_bytes": look.stat().st_size})
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    _write(receipt_path, receipt)


def _unpacked(path: Path, into: Path) -> Path:
    with zipfile.ZipFile(path) as archive:
        archive.extractall(into)
    return into


def _covariance(quats_wxyz, scales, linear=None):
    """The 3D covariance a 3DGS renderer builds (L = R S, Sigma = L L^T), carried by ``linear`` when given."""
    import numpy as np

    w, x, y, z = (np.asarray(quats_wxyz, float) / np.linalg.norm(quats_wxyz, axis=1, keepdims=True)).T
    turn = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
    covariance = turn @ (np.asarray(scales, float)[:, :, None] ** 2 * np.eye(3)) @ np.swapaxes(turn, 1, 2)
    return covariance if linear is None else np.asarray(linear) @ covariance @ np.asarray(linear).T


def _look_covariances(path: Path):
    """The covariances a USD reader renders from a look file (Gf real part first), in the file's world, with the
    field's arrays and world matrix."""
    import numpy as np

    stage = Usd.Stage.Open(str(path))
    field = next(prim for prim in stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat")
    matrix = np.asarray(UsdGeom.Xformable(field).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    gf = [(q.GetReal(), *q.GetImaginary()) for q in field.GetAttribute("orientations").Get()]
    stored = {name: np.asarray(field.GetAttribute(name).Get(), dtype=float)
              for name in ("positions", "scales", "opacities", "radiance:sphericalHarmonicsCoefficients")}
    return _covariance(gf, stored["scales"], matrix[:3, :3]), stored, matrix

#: Float32 scales and rotations: covariance entries (at most 0.008 m^2 here) agree within 1e-8, under 2e-6 of the
#: largest (5.4e-9 seen); a scrambled rotation is off by about 1e-3. Entries near zero make a relative bound
#: meaningless.
SAME = {"rtol": 0, "atol": 1e-8}


def _source_covariances(job: Job):
    import numpy as np

    linear = np.asarray(LOOK_TRANSFORM).T[:3, :3]
    return _covariance(job.splat.quats, np.exp(job.splat.scales.astype(float)), linear)


def test_a_splat_look_is_written_again_and_ships_with_a_viewer_ply_of_the_same_source(tmp_path):
    import numpy as np

    from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply, viewer_3dgs_properties

    job = _splat_job(tmp_path)
    world = World(job)
    assert world.publish()["status"] == "published"
    files = world.delivered[0]["files"]
    assert "scene/room.ply" in files and "scene/room.usdc" in files
    unpacked = _unpacked(_package(job), tmp_path / "unpacked")
    data = (unpacked / "scene/room.ply").read_bytes()
    header = data[:data.index(b"end_header\n")].decode("ascii").splitlines()
    assert header[:3] == ["ply", "format binary_little_endian 1.0", "element vertex 3000"]
    assert [line.removeprefix("property float ") for line in header[3:]] == list(viewer_3dgs_properties(0))
    # The PLY against the shipped USD's own values, where that file places them, and both against the source.
    ply = read_standard_3dgs_ply(unpacked / "scene/room.ply")
    usd, stored, matrix = _look_covariances(unpacked / "scene/room.usdc")
    np.testing.assert_allclose(ply.xyz, stored["positions"] @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-5)
    np.testing.assert_allclose(np.exp(ply.scales.astype(float)), 0.8 * stored["scales"], rtol=1e-6)
    np.testing.assert_allclose(1 / (1 + np.exp(-ply.opacity.astype(float))), stored["opacities"], atol=1e-7)
    np.testing.assert_allclose(ply.f_dc, stored["radiance:sphericalHarmonicsCoefficients"], atol=1e-7)
    np.testing.assert_allclose(_covariance(ply.quats, np.exp(ply.scales.astype(float))), usd, **SAME)
    np.testing.assert_allclose(usd, _source_covariances(job), **SAME)
    # The look's up (Y in the levelled splat) is the file's +Z: the PLY is in the USD look's frame.
    np.testing.assert_allclose(ply.xyz[:, 2], 0.8 * job.splat.xyz[:, 1], atol=1e-5)
    readme = (unpacked / "README.txt").read_text()
    flat = " ".join(readme.split())
    assert "- scene/room.ply: the same look as a standard 3D Gaussian splat PLY" in readme
    assert "scene/room.ply holds the Gaussians where scene/room.usdc places them" in flat
    assert "in its frame: up axis Z and 1 metre per unit." in flat
    assert sim.PLY_LEFT_OUT not in flat and sim.LOOK_AS_FIRST_WRITTEN not in flat
    assert "room.ply" not in (unpacked / "scene.usda").read_text()
    assert "look_not_written" not in _manifest(job)["package"]
    assert world.publish()["status"] == "current"


def test_a_package_the_old_writer_built_comes_out_with_the_sources_covariances_after_the_v3_rebuild(
        tmp_path, monkeypatch):
    """Review, 2026-11-18: nothing authors a finished job's look again, so a v3 rebuild of the half's look kept its
    scrambled rotations (94% of Gaussians off by more than 10%). The rebuild writes it again from its source."""
    import numpy as np

    job = _splat_job(tmp_path, old_writer=True)
    source = _source_covariances(job)
    half, _field, _matrix = _look_covariances(job.appearance)
    off = np.linalg.norm(half - source, axis=(1, 2)) / np.linalg.norm(source, axis=(1, 2))
    assert (off > 0.1).mean() > 0.5  # the half's look as the old writer left it
    # The package the v2 builder delivered: the half's look as it is.
    monkeypatch.setattr(sim, "BUILDER_REVISION", "website_company_simulation_builder.v2")
    monkeypatch.setattr(sim, "look_source", lambda *a, **k: None)
    world = World(job)
    assert world.publish()["status"] == "published"
    old = _manifest(job)["archive"]
    delivered, _f, _m = _look_covariances(_unpacked(_package(job), tmp_path / "v2") / "scene/room.usdc")
    assert np.abs(delivered - source).max() > 1e-4
    monkeypatch.undo()
    assert world.publish()["status"] == "published"
    unpacked = _unpacked(_package(job), tmp_path / "v3")
    usd, _f, _m = _look_covariances(unpacked / "scene/room.usdc")
    np.testing.assert_allclose(usd, source, **SAME)
    from blueprint_pipeline.gaussian_splat_decode import read_standard_3dgs_ply
    ply = read_standard_3dgs_ply(unpacked / "scene/room.ply")
    np.testing.assert_allclose(_covariance(ply.quats, np.exp(ply.scales.astype(float))), source, **SAME)
    readme = " ".join((unpacked / "README.txt").read_text().split())
    assert sim.LOOK_AS_FIRST_WRITTEN not in readme and sim.ROTATIONS_STORED_WRONG not in readme
    # The v2 package was read until the website took the v3 one, then retired.
    assert world.retracted == [old["uri"]]
    assert [row["retracted"] for row in _manifest(job)[sim.permission.SUPERSEDED]] == [True]


@pytest.mark.parametrize("why", ["missing", "over_memory_bound", "changed"])
def test_a_look_whose_source_cannot_be_written_again_ships_as_first_written_and_says_so(tmp_path, monkeypatch, why):
    job = _splat_job(tmp_path, old_writer=True)
    if why == "missing":
        job.source.unlink()
    elif why == "over_memory_bound":
        from blueprint_pipeline import website_company_look
        monkeypatch.setattr(website_company_look, "REAUTHOR_MAX_SOURCE_BYTES", job.source.stat().st_size - 1)
    else:
        job.source.write_bytes(job.source.read_bytes()[:-4] + b"\x00\x00\x80\x3f")
    world = World(job)
    assert world.publish()["status"] == "published"
    files = world.delivered[0]["files"]
    assert "scene/room.usdc" in files and not any(name.endswith(".ply") for name in files)
    reason = {"missing": "missing", "over_memory_bound": "over_memory_bound", "changed": "changed"}[why]
    assert _manifest(job)["package"]["look_not_written"] == {
        "Room": f"website_company_simulation_look_source_{reason}"}
    readme = " ".join(_unpacked(_package(job), tmp_path / "u").joinpath("README.txt").read_text().split())
    assert sim.LOOK_AS_FIRST_WRITTEN in readme and sim.ROTATIONS_STORED_WRONG in readme and "room.ply" not in readme


def test_a_fixed_writers_look_that_cannot_be_written_again_names_no_wrong_rotations(tmp_path):
    job = _splat_job(tmp_path)
    job.source.unlink()
    world = World(job)
    assert world.publish()["status"] == "published"
    readme = " ".join(_unpacked(_package(job), tmp_path / "u").joinpath("README.txt").read_text().split())
    assert sim.LOOK_AS_FIRST_WRITTEN in readme and sim.ROTATIONS_STORED_WRONG not in readme


def test_a_splat_look_nothing_names_ships_as_it_is_with_no_ply_and_no_note(tmp_path):
    job = _splat_job(tmp_path, receipt=False)
    world = World(job)
    assert world.publish()["status"] == "published"
    assert not any(name.endswith(".ply") for name in world.delivered[0]["files"])
    readme = " ".join(_unpacked(_package(job), tmp_path / "u").joinpath("README.txt").read_text().split())
    assert sim.LOOK_AS_FIRST_WRITTEN not in readme and "look_not_written" not in _manifest(job)["package"]


def test_a_look_that_is_not_a_splat_ships_no_ply_and_records_nothing(job):
    world = World(job)
    assert world.publish()["status"] == "published"
    assert not any(name.endswith(".ply") for name in world.delivered[0]["files"])
    assert "look_not_written" not in _manifest(job)["package"]


# --- over the limit: the PLY goes first, and a pass out of time leaves the next where it stopped ------------------

def _look_sizes(job: Job, tmp_path: Path) -> tuple[int, int, int]:
    """The zip's bytes with the look and its PLY, with the look only, and with neither."""
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    sizes = [sim.build_package(value, tmp_path / f"{index}.zip", literals=[], max_bytes=1 << 30)
             for index, value in enumerate((source, {**source, "ply_left_out": True}, sim.without_look(source)))]
    assert "scene/room.ply" in sizes[0]["files"] and "scene/room.ply" not in sizes[1]["files"]
    return sizes[0]["size_bytes"], sizes[1]["size_bytes"], sizes[2]["size_bytes"]


_REAL_BUILD = sim._build_once


def _counting(monkeypatch, *, timeout_on=None) -> list:
    """Each build's (PLY left out, look left out), the real build behind it; ``timeout_on`` runs out of time."""
    builds = []
    real = _REAL_BUILD

    def build(source, *a, **k):
        step = (bool(source.get("ply_left_out")), bool(source.get("look_left_out")))
        builds.append(step)
        if step == timeout_on:
            raise ValueError(sim.BUILD_TIMED_OUT)
        return real(source, *a, **k)
    monkeypatch.setattr(sim, "_build_once", build)
    return builds


@pytest.mark.parametrize("over", ["ply", "look"])
def test_over_the_limit_the_ply_goes_before_the_look_and_the_readme_says_which(tmp_path, monkeypatch, over):
    job = _splat_job(tmp_path)
    whole, usd_only, bare = _look_sizes(job, tmp_path / "sizes")
    assert bare < usd_only < whole
    builds = _counting(monkeypatch)
    world = World(job)
    assert world.publish(max_bytes=usd_only if over == "ply" else bare)["status"] == "published"
    record, package = world.delivered[0], _manifest(job)["package"]
    assert "scene/room.ply" not in record["files"]
    readme = " ".join(_unpacked(_package(job), tmp_path / "unpacked").joinpath("README.txt").read_text().split())
    assert "room.ply" not in readme
    if over == "ply":
        assert builds == [(False, False), (True, False)] and record["size_bytes"] == usd_only
        assert package["ply_left_out"] is True and "look_left_out" not in package
        assert "scene/room.usdc" in record["files"] and sim.PLY_LEFT_OUT in readme and sim.LOOK_LEFT_OUT not in readme
    else:
        assert builds == [(False, False), (True, False), (False, True)] and record["size_bytes"] == bare
        assert package["look_left_out"] is True and "ply_left_out" not in package
        assert "scene/room.usdc" not in record["files"]
        assert sim.LOOK_LEFT_OUT in readme and sim.PLY_LEFT_OUT not in readme and "Gaussian" not in readme


def test_over_the_limit_even_without_the_look_is_over_the_cap_after_three_builds(tmp_path, monkeypatch):
    job = _splat_job(tmp_path)
    _whole, _usd_only, bare = _look_sizes(job, tmp_path / "sizes")
    builds = _counting(monkeypatch)
    world = World(job)
    assert world.publish(max_bytes=bare - 1) == {"status": "over_size_cap"}
    assert builds == [(False, False), (True, False), (False, True)] and world.delivered == []
    assert _manifest(job)["size_over"] == list(sim.SIZE_STEPS)


def test_a_pass_out_of_time_leaves_the_next_where_it_stopped(tmp_path, monkeypatch):
    """Review, 2026-11-18: three builds inside one deadline could miss it on every pass. The steps found over the
    limit are recorded, and the next try of the revision starts after them."""
    job = _splat_job(tmp_path)
    _whole, _usd_only, bare = _look_sizes(job, tmp_path / "sizes")
    builds = _counting(monkeypatch, timeout_on=(True, False))
    world = World(job)
    assert world.publish(max_bytes=bare) == {"status": "build_timed_out"}
    assert builds == [(False, False), (True, False)] and _manifest(job)["size_over"] == ["whole"]
    builds = _counting(monkeypatch, timeout_on=(False, True))
    later = sim.permission.LIVE_STATEMENT_RECHECK_SECONDS + 1
    result = sim.publish_company_simulation(job.capture, now=lambda: NOW + later, check_rights=world.check_rights,
                                            archive=world.archive, retract=world.retract, deliver=world.deliver,
                                            queue_root=job.queue, max_bytes=bare)
    assert result == {"status": "build_timed_out"} and builds == [(True, False), (False, True)]
    assert _manifest(job)["size_over"] == ["whole", "without_ply"]
    builds = _counting(monkeypatch)
    result = sim.publish_company_simulation(job.capture, now=lambda: NOW + 2 * later, check_rights=world.check_rights,
                                            archive=world.archive, retract=world.retract, deliver=world.deliver,
                                            queue_root=job.queue, max_bytes=bare)
    assert result["status"] == "published" and builds == [(False, True)]
    assert _manifest(job)["package"]["look_left_out"] is True


# --- the package out stays readable through a rebuild ----------------------------------------------------------------

def _new_revision(job: Job) -> None:
    job.criteria["settling"]["position_tolerance_m"] = 0.02
    job.seal()


def test_a_rebuild_keeps_the_delivered_package_readable_until_the_website_takes_the_new_one(job, monkeypatch):
    """Review, 2026-11-18: the package out was taken back before each rebuild, so downloads failed for the build's
    length. The new zip is built beside it; the website is moved by its delivery, and only then is the old retired."""
    world = World(job)
    assert world.publish()["status"] == "published"
    old = _manifest(job)["archive"]
    _new_revision(job)
    seen = []
    real = sim._build_once
    monkeypatch.setattr(sim, "_build_once", lambda *a, **k: seen.append(
        _read_route(job, world, now=NOW + 1, digest=old["digest"])) or real(*a, **k))
    delivered = []
    original = world.deliver

    def deliver(record, *, identity):
        delivered.append(_read_route(job, world, now=NOW + 1, digest=old["digest"]))
        original(record, identity=identity)
    world.deliver = deliver
    assert world.publish()["status"] == "published"
    assert seen and all(isinstance(size, int) for size in seen)  # read through the build
    assert isinstance(delivered[0], int)  # and up to the website's taking the new one
    manifest = _manifest(job)
    assert manifest["archive"]["digest"] != old["digest"] and world.retracted == [old["uri"]]
    assert _read_route(job, world, now=NOW + 1, digest=old["digest"]) == "unavailable"
    assert isinstance(_read_route(job, world, now=NOW + 1), int)


def test_a_failed_delivery_of_the_new_package_keeps_the_old_one_read_until_the_next_pass_delivers(job):
    world = World(job)
    assert world.publish()["status"] == "published"
    old = _manifest(job)["archive"]
    _new_revision(job)
    world.delivery_error = RuntimeError("website unreachable")
    assert world.publish()["status"] == "failed"
    assert world.retracted == [] and isinstance(_read_route(job, world, now=NOW + 1, digest=old["digest"]), int)
    world.delivery_error = None
    assert world.publish()["status"] == "published" and world.retracted == [old["uri"]]
    assert _read_route(job, world, now=NOW + 1, digest=old["digest"]) == "unavailable"


def test_a_withdrawal_during_a_rebuild_takes_back_both_copies(job, monkeypatch):
    world = World(job)
    assert world.publish()["status"] == "published"
    old = _manifest(job)["archive"]
    _new_revision(job)
    real = sim._build_once
    monkeypatch.setattr(sim, "_build_once", lambda *a, **k: _withdraw(job.capture) or real(*a, **k))
    assert world.publish()["status"] == "withdrawn"
    assert old["uri"] in world.retracted and world.objects == {}


def _route_reads(job: Job, world: World, *archives) -> list:
    return [_read_route(job, world, now=NOW + 1, digest=archive["digest"]) for archive in archives]


@pytest.mark.parametrize("failure", ["refused", "over_size_cap", "build_timed_out"])
def test_a_failed_rebuild_leaves_the_delivered_package_exactly_as_it_was(job, monkeypatch, failure):
    """Re-review, 2026-11-18: a rebuild refused, out of time or over the limit took the delivered zip back. It now
    leaves it as it was, still read, and keeps the refusal beside it for its revision."""
    world = World(job)
    assert world.publish()["status"] == "published"
    before = _manifest(job)
    if failure == "refused":
        _stage(job.collision, "Collision", doc=f"compiled for {REQUEST}")  # the job's id: the scan refuses it
        job.seal()
        result = world.publish()
    elif failure == "over_size_cap":
        result = world.publish(max_bytes=1024)
    else:
        _new_revision(job)
        result = sim.publish_company_simulation(job.capture, now=lambda: NOW, check_rights=world.check_rights,
                                                archive=world.archive, retract=world.retract, deliver=world.deliver,
                                                queue_root=job.queue, build_seconds=-1)
    assert result["status"] == failure
    after = _manifest(job)
    assert {key: after[key] for key in ("archive", "package", "delivered", "simulation_revision")} == {
        key: before[key] for key in ("archive", "package", "delivered", "simulation_revision")}
    assert after["rebuild_refused"]["refused"] != before.get("refused") and world.retracted == []
    assert isinstance(_read_route(job, world, now=NOW + 1), int) and _package(job).is_file()
    # The refusal stands for its revision as before: no build again within its window, and the package stays. A
    # half refused for its size or time is the floor rule's (``refused_half``): a delivered half stays, no source.
    monkeypatch.setattr(sim, "build_package", lambda *a, **k: pytest.fail("rebuilt within the refusal's window"))
    again = (world.publish(max_bytes=1024) if failure == "over_size_cap" else _at(world, 60))
    assert again["status"] == (failure if failure == "refused" else "no_source")
    assert world.retracted == [] and len(world.delivered) == 1 and isinstance(_read_route(job, world, now=NOW + 1), int)


def test_a_copy_whose_delivery_answer_was_lost_stays_read_until_a_newer_one_is_taken(job):
    """Re-review, 2026-11-18: the website took a package but its answer was lost, then the next revision's upload
    failed: nothing was readable until a later pass. Every copy since the last one known taken stays readable,
    in any order of failures, until a newer one is taken."""
    world = World(job)
    assert world.publish()["status"] == "published"
    first = _manifest(job)["archive"]
    _new_revision(job)
    original = world.deliver

    def taken_but_answer_lost(record, *, identity):
        original(record, identity=identity)
        raise RuntimeError("website answer lost")
    world.deliver = taken_but_answer_lost
    assert world.publish()["status"] == "failed"
    second = _manifest(job)["archive"]
    assert len(world.delivered) == 2  # the website holds the second now
    world.deliver = original
    job.criteria["settling"]["position_tolerance_m"] = 0.03
    job.seal()
    upload = world.archive

    def upload_fails(path, *, identity):
        raise RuntimeError("store unavailable")
    world.archive = upload_fails
    assert world.publish()["status"] == "failed"
    assert world.retracted == [] and all(isinstance(size, int) for size in _route_reads(job, world, first, second))
    world.archive = upload
    assert world.publish()["status"] == "published"
    assert sorted(world.retracted) == sorted([first["uri"], second["uri"]])
    assert _route_reads(job, world, first, second) == ["unavailable", "unavailable"]
    assert isinstance(_read_route(job, world, now=NOW + 1), int)


class _Stopped(BaseException):
    """The process stopping (no ``except Exception`` catches it)."""


def test_a_stop_right_after_the_delivery_is_written_still_retires_the_old_copies(job, monkeypatch):
    """Codex P2, PR #3641: the delivered manifest was written before the old copies' retirement was recorded, so a
    stop between the two left them live for good (the next pass found the package current). The delivery and the
    retirement intent are now one write, and the next pass retires them first."""
    world = World(job)
    assert world.publish()["status"] == "published"
    old = _manifest(job)["archive"]
    _new_revision(job)

    def stop(*a, **k):
        raise _Stopped()
    monkeypatch.setattr(sim.permission, "retire", stop)
    with pytest.raises(_Stopped):
        world.publish()
    manifest = _manifest(job)
    assert manifest["delivered"]["simulation_digest"] == manifest["simulation_digest"] and world.retracted == []
    assert [row.get("retraction") for row in manifest[sim.permission.SUPERSEDED]] == ["pending"]
    assert _read_route(job, world, now=NOW + 1, digest=old["digest"]) == "unavailable"  # the website moved on
    monkeypatch.undo()
    assert world.publish()["status"] == "current" and world.retracted == [old["uri"]]
    assert [row["retracted"] for row in _manifest(job)[sim.permission.SUPERSEDED]] == [True]


# --- money: no price, and none of Blueprint's own spend, reaches the company's package ------------------------------

def test_no_price_and_none_of_blueprints_spend_reaches_the_package(job, tmp_path):
    from tests.money_copy import price_findings

    spend = {"reserved_cost_usd": 37.25, "max_spend_usd": 61.5, "spend": {"usd": 12.75, "note": "Scene budget $40 PLANTSPEND"}}
    for path in (job.packet / sim.PLAN_FILE, job.scene / "team_scene_inputs.v1.json",
                 job.queue / "status/31e12dca983b8837295ac911c44dee68.json"):
        _write(path, {**json.loads(path.read_text()), **spend})
    world = World(job)
    assert world.publish()["status"] == "published"
    with zipfile.ZipFile(_package(job)) as archive:
        texts = {name: archive.read(name).decode("utf-8") for name in archive.namelist()
                 if name.endswith((".txt", ".json", ".usda"))}
    assert {"README.txt", "success_rule.json", "scene.usda"} <= set(texts)
    for name, text in [*texts.items(), ("the record the WebApp shows the company", json.dumps(world.delivered[0]))]:
        assert price_findings(text) == [], name
    _assert_absent_from_zip(job, [b"37.25", b"61.5", b"12.75", b"PLANTSPEND", b"reserved_cost", b"max_spend"])
    assert not {"37.25", "61.5", "12.75", "PLANTSPEND"} & set(re.findall(r"[\w.]+", json.dumps(world.delivered[0])))


@pytest.mark.parametrize("sealed", [None, "v1", "plan", "tampered"])
def test_the_build_plan_is_named_in_plain_words_only_when_its_decision_is_sealed(job, tmp_path, sealed):
    """The build plan (owner direction 2026-11-23) in the README: what the job needs, the one object the plan takes
    out of the scan, the ones it keeps as filmed, what the video does not show. Never that anything was built."""
    from blueprint_pipeline.website_build_plan_notes import DECISION_SCHEMA
    before = sim.simulation_source(job.capture, queue_root=job.queue)
    row = {"variety": None, "rigidity": "rigid", "size_m": None, "mass_kg": None, "reason": "r",
           "editor_object_ids": []}
    plan = {"objects": [
        {**row, "target_id": "tote", "name": "blue tote", "role": "moved_object", "sourcing": "parametric"},
        {**row, "target_id": "carton", "name": "carton", "role": "moved_object", "sourcing": "generate"},
        {**row, "target_id": "not_in_video", "name": "pallet", "role": "fixture", "sourcing": "library"}]}
    decided = {"schema_version": "atlas_job_object_rule.v1" if sealed == "v1" else DECISION_SCHEMA,
               "applied": False, "build_plan": {"plan": plan},
               "built": {"target_id": "tote", "editor_object_ids": ["e"]}}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    if sealed == "tampered":
        decided["built"] = {"target_id": "carton"}
    if sealed is not None:
        _write(job.capture / "pipeline" / "atlas" / "job_object_rule.json", decided)
    expected = sealed == "plan"
    source = sim.simulation_source(job.capture, queue_root=job.queue)
    assert bool(source.get("plan_notes")) is expected
    assert (sim.simulation_revision(source) == sim.simulation_revision(before)) is (not expected)
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
    for sentence in ("What this simulation needs, as Blueprint's model read the job: blue tote, carton and pallet.",
                     "The plan takes this out of the scan to build as its own object: blue tote.",
                     "The plan keeps these in the scan as filmed, not built on their own: carton.",
                     "Not in the video, so not in this simulation: pallet."):
        assert (sentence in readme) is expected
    assert "was built" not in readme
