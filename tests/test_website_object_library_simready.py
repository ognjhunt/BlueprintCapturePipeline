"""NVIDIA SimReady as a second library (owner direction 2026-11-23): only the packages NVIDIA reviewed, at their pinned
commits; both licenses (the package definition's and the USD's own) on the allow-list, and an author, or the package
is never used; its USD composed and converted to the job's model, credited in the company's package.

Offline: a tiny fixture package written here with ``pxr`` (a root layer in centimetres with Y up and a payload that
holds its mesh), served by an injected fetcher; no Gemini key is set and nothing is paid for.
"""
# Covers: src/blueprint_pipeline/website_object_library.py
# Covers: src/blueprint_pipeline/website_library_asset.py
# Covers: scripts/build_simready_object_index.py
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import website_library_asset as asset
from blueprint_pipeline import website_object_library as library
from blueprint_pipeline.website_supplied_mesh import LIBRARY_SUBJECT, supplied_mesh_configuration

pytest.importorskip("pxr")

REPO, COMMIT, ASSET = "nvidia/simready-assets", "6" * 40, "box_cardboard_a01"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{COMMIT}/{ASSET}/"
ROOT = "simready_usd/sm_box_cardboard_a01_01.usd"
PAYLOAD = "simready_usd/payloads/geometries.usda"
THUMB = "simready_usd/.thumbs/256x256/sm_box_cardboard_a01_01.usd.png"
BOM = ".metadata/com.nvidia.simready.packaging.bom.json"
ROW = {"target_id": "box", "name": "cardboard box", "role": "moved_object", "variety": None, "rigidity": "rigid",
       "size_m": {"low": 0.4, "high": 0.6, "basis": "estimated", "from": "the analysis"}, "mass_kg": None,
       "sourcing": "parametric", "parametric_shape": "carton", "reason": "The job moves the box.",
       "editor_object_ids": ["e"]}


def _package(tmp_path: Path) -> dict[str, bytes]:
    """The fixture package's files by their path in it: a 60 x 40 x 30 cm box, stored Y up in centimetres."""
    from PIL import Image
    from pxr import Sdf, Usd, UsdGeom
    folder = tmp_path / "package-source"
    (folder / "simready_usd/payloads").mkdir(parents=True)
    geometry = Usd.Stage.CreateNew(str(folder / PAYLOAD))
    UsdGeom.SetStageUpAxis(geometry, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(geometry, 0.01)
    geometry.SetDefaultPrim(UsdGeom.Xform.Define(geometry, "/RootNode").GetPrim())
    mesh = UsdGeom.Mesh.Define(geometry, "/RootNode/Box")
    corners = [(x, y, z) for z in (-20, 20) for y in (0, 30) for x in (-30, 30)]
    mesh.CreatePointsAttr(corners)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    collider = UsdGeom.Mesh.Define(geometry, "/RootNode/Collider")  # a guide is never part of the model
    collider.CreatePointsAttr([(0, 0, 0), (900, 0, 0), (0, 900, 0)])
    collider.CreateFaceVertexCountsAttr([3])
    collider.CreateFaceVertexIndicesAttr([0, 1, 2])
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
    geometry.GetRootLayer().Save()
    root = Usd.Stage.CreateNew(str(folder / "root.usda"))
    UsdGeom.SetStageUpAxis(root, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(root, 0.01)
    node = root.DefinePrim("/RootNode", "Xform")
    root.SetDefaultPrim(node)
    node.GetPayloads().AddPayload("./payloads/geometries.usda")
    root.GetRootLayer().customLayerData = {"SimReady_Metadata": {
        "asset_license": "CC-BY 4.0", "author": "NVIDIA", "asset_name": "sm_box_cardboard_a01_01", "category": "prop",
        "asset_type": "prop", "mass": 1.0, "asset_extents": "(60.0, 30.0, 40.0)"}}
    root.GetRootLayer().Save()
    layer = Sdf.Layer.FindOrOpen(str(folder / "root.usda"))
    layer.Export(str(folder / ROOT))  # a .usd root, as SimReady names it
    out = io.BytesIO()
    Image.new("RGBA", (256, 256), (200, 160, 100, 255)).save(out, format="PNG")
    return {ROOT: (folder / ROOT).read_bytes(), PAYLOAD: (folder / PAYLOAD).read_bytes(), THUMB: out.getvalue()}


def _bom(files: dict[str, bytes]) -> bytes:
    return json.dumps({"format_version": "1.0", "items": [
        {"relative_path": path, "size": len(data), "hash": {"sha256": hashlib.sha256(data).hexdigest()}}
        for path, data in files.items()]}).encode()


class Fetcher:
    def __init__(self, served: dict[str, bytes]):
        self.served, self.urls = served, []

    def __call__(self, url, target, *, sha256, max_bytes):
        self.urls.append(url)
        Path(target).write_bytes(self.served[url])


def _fixture(tmp_path: Path, *, rights=None) -> tuple[dict[str, Path], Fetcher]:
    files = _package(tmp_path)
    bom = _bom(files)
    sha = {path: hashlib.sha256(data).hexdigest() for path, data in files.items()}
    record = {"asset": ASSET, "root": ROOT, "asset_name": "sm_box_cardboard_a01_01", "category": "prop",
              "asset_type": "prop", "up_axis": "Y", "meters_per_unit": 0.01, "size_m": [0.6, 0.4, 0.3],
              "mass_kg": 1.0, "bom": {"sha256": hashlib.sha256(bom).hexdigest(), "size_bytes": len(bom)},
              "thumbnail": {"sha256": sha[THUMB], "size_bytes": len(files[THUMB])}, "source": 0, "rights": 0}
    allowed = {"package_license": "CC-BY-4.0", "asset_license": "CC-BY 4.0", "author": "NVIDIA"}
    index = {"schema_version": "object_library_simready_index.v1", "library": "nvidia_simready",
             "library_title": "NVIDIA SimReady",
             "item_url": "https://huggingface.co/datasets/{repo}/resolve/{commit}/{asset}/"
                         "com.nvidia.simready.packaging.json",
             "file_url": "https://huggingface.co/datasets/{repo}/resolve/{commit}/{asset}/{path}",
             "sources": [{"repo": REPO, "commit": COMMIT}], "rights": rights or [allowed],
             "records": [record]}
    path = tmp_path / "object_library_simready.v1.json"
    path.write_text(json.dumps(index))
    served = {BASE + path_: data for path_, data in files.items()}
    served[BASE + BOM] = bom
    from tests.test_website_object_library import deploy
    return deploy({"nvidia_simready": path}), Fetcher(served)


def test_both_licenses_must_be_allowed_and_an_author_named(tmp_path):
    indexes, _ = _fixture(tmp_path)
    entry = library.catalog(indexes)[0]
    assert entry["license"] == "CC-BY-4.0" and entry["author"] == "NVIDIA" and entry["title"] == ASSET
    assert entry["item_url"] == BASE + "com.nvidia.simready.packaging.json"
    for case, (rights, why) in enumerate((({"package_license": "CC-BY-4.0", "asset_license": "CC-BY-NC 4.0", "author": "NVIDIA"},
                         "license_not_allowed:the USD's own SimReady_Metadata"),
                        ({"package_license": "other", "asset_license": "CC-BY 4.0", "author": "NVIDIA"},
                         "license_not_allowed:the package definition"),
                        ({"package_license": "CC-BY-4.0", "asset_license": None, "author": "NVIDIA"},
                         "license_not_allowed:the USD's own SimReady_Metadata"),
                        ({"package_license": "CC-BY-4.0", "asset_license": "CC-BY 4.0", "author": None},
                         "author_not_stated"),
                        # The package's own words, as one reviewed package states them (M-XR, 2026-11-23).
                        ({"package_license": "cc-by-4.0", "author": "marso by M-XR", "asset_license":
                          "Not licensed for distribution; private internal evaluation only; public licence undecided"},
                         "license_not_allowed:the USD's own SimReady_Metadata"),
                        # A credit the package's outbound scan would refuse (a web address as the author's name).
                        ({"package_license": "CC-BY-4.0", "asset_license": "CC-BY-4.0",
                          "author": "Freepoly.org (Sketchfab)"}, "credit_not_shippable"))):
        refused = library.catalog(_fixture(tmp_path / f"case-{case}", rights=[rights])[0])[0]
        assert refused["license"] is None and refused["license_refused"] == why
        assert library.candidates(ROW, [refused]) == []


def test_the_reviewed_package_is_fetched_at_its_commit_composed_and_converted(tmp_path):
    from blueprint_pipeline.website_build_plan_notes import DECISION_SCHEMA
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    indexes, fetch = _fixture(tmp_path)
    decided = {"schema_version": DECISION_SCHEMA, "applied": False, "build_plan": {"plan": {"objects": [ROW]}},
               "built": {"target_id": "box", "editor_object_ids": ["e"]}}
    decided["digest"] = canonical_digest(decided, digest_field="digest")
    capture = tmp_path / "capture"
    (capture / "pipeline/atlas").mkdir(parents=True)
    (capture / "pipeline/atlas/job_object_rule.json").write_text(json.dumps(decided))
    asked = []

    def ask(**kwargs):
        asked.append(kwargs)
        return {"answer": {"candidate_id": "c01", "reason": "The same corrugated carton."}}
    removal = library.attach_library_subject({"entries": []}, capture_root=capture, task_context={},
                                             output_root=capture / "library", entries=library.catalog(indexes),
                                             fetch=fetch, ask=ask)
    # The model saw the package's thumbnail beside its words.
    assert [caption for caption, _ in asked[0]["images"]] == ["Candidate c01:"]
    assert asked[0]["images"][0][1][:2] == b"\xff\xd8"  # sent as a small JPEG
    subject = removal[LIBRARY_SUBJECT]
    assert subject["pick"]["source"] == "model" and subject["attribution"]["format"] == "simready_usd"
    # Only reviewed bytes, at the pinned commit: the bill, then every layer it lists, by its sha256; no texture.
    assert all(url.startswith(BASE) for url in fetch.urls) and BASE + BOM in fetch.urls
    assert {row["role"] for row in subject["files"]} == {"bom", "root", "layer"}
    assert {row.get("relative_path") for row in subject["files"]} >= {ROOT, PAYLOAD}
    # Converted from USD (centimetres, Y up) to the job's model (metres, Z up) and scaled to the video's estimate.
    made = supplied_mesh_configuration(
        {"authoring_backend": "artisan_v1", "metric_envelope": {"maximum_dimension_relative_error": 0.25}},
        synthetic=None, synthetic_authoring=None, parametric={"target_id": "box", "shape": "carton"}, subject_id="box",
        sim_min=[0, 0, 0], sim_max=[0.45, 0.3, 0.225], library=subject, library_root=capture / "model")
    design = made["library_design"]
    assert made["authoring_source"] == "library" and "parametric_design" not in made
    assert design["converted_from"] == "USD" and design["sizing"]["model_size_m"] == pytest.approx([0.6, 0.4, 0.3])
    assert design["sizing"]["status"] == "scaled_to_video_estimate" and design["sizing"]["scale"] == pytest.approx(0.75)
    from blueprint_pipeline.website_library_notes import changes
    assert changes(design) == "converted it from USD and scaled it"
    assert design["face_count"] == 12  # the guide collider is not part of the model
    # A changed layer is never used: fetched again at its pinned sha256 when it can be (stress round 2), else the next
    # source.
    payload = Path(next(row["path"] for row in subject["files"] if row.get("relative_path") == PAYLOAD))
    pinned = payload.read_bytes()
    payload.write_text(payload.read_text().replace("30", "31"))

    def down(url, target, *, sha256, max_bytes):
        raise OSError("no network")
    again = supplied_mesh_configuration(
        {"authoring_backend": "artisan_v1", "metric_envelope": {"maximum_dimension_relative_error": 0.25}},
        synthetic=None, synthetic_authoring=None, parametric=None, subject_id="box", sim_min=[0, 0, 0],
        sim_max=[0.45, 0.3, 0.225], library=subject, library_root=capture / "model2", library_fetch=down)
    assert again["library_not_used"] == "library_file_changed"
    healed = supplied_mesh_configuration(
        {"authoring_backend": "artisan_v1", "metric_envelope": {"maximum_dimension_relative_error": 0.25}},
        synthetic=None, synthetic_authoring=None, parametric=None, subject_id="box", sim_min=[0, 0, 0],
        sim_max=[0.45, 0.3, 0.225], library=subject, library_root=capture / "model3", library_fetch=fetch)
    assert healed["authoring_source"] == "library" and payload.read_bytes() == pinned


@pytest.mark.parametrize("repo, request_lines, refused", [
    # As M-XR asks (its ATTRIBUTION.txt and LICENSE.txt, 2026-11-23): a link the README cannot give.
    (REPO, ["marso by M-XR", "Source: https://huggingface.co/datasets/M-XR/marso-simready"],
     "attribution_link_not_carried"),
    (REPO, ["Copyright (c) M-XR Limited. Attribution: marso by M-XR (https://m-xr.com)."],
     "attribution_link_not_carried"),
    # A name the credit does not give is refused too; never dropped.
    (REPO, ["Please credit: Someone Else Studio"], "attribution_manner_not_carried"),
    # What the credit already gives (its author, its license and deed) is carried, and the model is used.
    (REPO, ["NVIDIA", "Licence: CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"], None),
    (REPO, [], None),
    # The M-XR dataset is left out whole for now, whatever a package's files say.
    ("M-XR/marso-simready", [], "attribution_link_not_carried")])
def test_a_credit_the_readme_cannot_give_as_asked_leaves_the_model_out(tmp_path, repo, request_lines, refused):
    indexes, _ = _fixture(tmp_path)
    index = json.loads(indexes["nvidia_simready"].read_text())
    index["sources"][0]["repo"] = repo
    if request_lines:
        index["records"][0]["attribution_request"] = request_lines
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(index))
    entry = library.catalog({"nvidia_simready": path})[0]
    assert entry.get("license_refused") == refused and (entry["license"] is None) is (refused is not None)
    if refused:
        assert library.candidates(ROW, [entry]) == []  # listed, never shown or used


def test_a_layer_path_cannot_leave_its_package_folder(tmp_path):
    for relative in ("../outside.usd", "/etc/x.usd", ""):
        with pytest.raises(ValueError, match="library_file_record_invalid"):
            asset.fetch_pinned({"url": BASE + "x.usd", "sha256": "a" * 64, "size_bytes": 3, "relative_path": relative},
                               tmp_path, fetch=lambda *a, **k: None)


def test_every_checked_in_credit_link_passes_the_outbound_scan():
    """Each usable library model's credit names its pinned item: the scan admits exactly those links, and the
    licenses'; a dataset left out whole (M-XR, whose credit asks for a link the README cannot give) needs none."""
    from blueprint_pipeline import outbound_file_scan as scan
    entries = library.catalog()
    used = [entry for entry in entries if entry["license"]]
    assert {entry["library"] for entry in used} == {"google_scanned_objects", "nvidia_simready"}
    for link in {entry["item_url"] for entry in used} | {row["url"] for row in library.ALLOWED_LICENSES.values()}:
        assert scan._blank_allowed_links(f"({link}).".encode()) == b"(" + b" " * len(link) + b").", link
    mxr = [entry for entry in entries if entry.get("dataset") == "M-XR/marso-simready"]
    assert mxr and not any(entry["license"] for entry in mxr)
    assert {entry["license_refused"] for entry in mxr} <= {"attribution_link_not_carried",
                                                          "license_not_allowed:the USD's own SimReady_Metadata"}
    assert all(entry["license_refused"] == "attribution_link_not_carried" for entry in mxr
               if entry["license_statements"][1][1] == "cc-by-4.0")
    # Every request to be credited in a manner of its own is the M-XR one: no other usable model carries one.
    assert not any(entry.get("attribution_request") for entry in used)
    assert b"M-XR" not in b"".join(scan.ALLOWED_LINK_PREFIXES)


def test_the_index_builder_reads_only_pinned_urls_and_checks_the_bill(tmp_path):
    from scripts import build_simready_object_index as builder
    files = _package(tmp_path)
    bom = _bom(files)
    listing = tmp_path / "simready-assets"
    listing.write_text(f"{BASE}com.nvidia.simready.packaging.json\n")
    packaging = json.dumps({"license": "CC-BY-4.0", "package_id": "p", "metadata": [
        {"name": "com.nvidia.simready.packaging.bom.json", "hash": {"sha256": hashlib.sha256(bom).hexdigest()}}]})
    served = {BASE + "com.nvidia.simready.packaging.json": packaging.encode(), BASE + BOM: bom,
              **{BASE + path: data for path, data in files.items()}}
    record = builder.package_record(builder.submission_urls(listing)[0], lambda url, cap: served[url])
    assert (record["package_license"], record["asset_license"], record["author"]) == ("CC-BY-4.0", "CC-BY 4.0", "NVIDIA")
    assert record["root"] == ROOT and record["up_axis"] == "Y" and record["meters_per_unit"] == 0.01
    assert record["size_m"] == [0.6, 0.4, 0.3]  # Y up in centimetres, as metres with Z up
    assert record["thumbnail"]["sha256"] == hashlib.sha256(files[THUMB]).hexdigest()
    # The package supplies no copyright notice and no disclaimer: none is recorded, and none made up.
    assert (record["copyright"], record["disclaimer"]) == (None, None) and "notice_files" not in record
    # A notice file the bill names is read for its copyright line and its disclaimer, as written; a contact is not.
    attribution = ("marso by M-XR\nCopyright 2026 Example Ltd.\nContact: someone@example.com\n"
                   "Provided as is, without warranty of any kind.\n").encode()
    noticed = {**files, "usd/ATTRIBUTION.txt": attribution}
    bill = _bom(noticed)
    served.update({BASE + "usd/ATTRIBUTION.txt": attribution, BASE + BOM: bill,
                   BASE + "com.nvidia.simready.packaging.json": packaging.replace(
                       hashlib.sha256(bom).hexdigest(), hashlib.sha256(bill).hexdigest()).encode()})
    record = builder.package_record(builder.submission_urls(listing)[0], lambda url, cap: served[url])
    assert record["copyright"] == "Copyright 2026 Example Ltd."
    assert record["disclaimer"] == "Provided as is, without warranty of any kind."
    assert record["notice_files"][0]["path"] == "usd/ATTRIBUTION.txt" and "someone" not in json.dumps(record)
    # The attribution file is a request for how to be credited: kept as written (but its contact line), never dropped.
    assert record["attribution_request"] == ["marso by M-XR", "Copyright 2026 Example Ltd.",
                                             "Provided as is, without warranty of any kind."]
    # A license file's own copyright line is kept (its first sentence); the legal code's words are not a notice, and its
    # disclaimer is the license text itself, not one beyond it.
    licence = ("Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
               "Copyright (c) M-XR Limited. Attribution: marso by M-XR (https://m-xr.com).\n"
               "     copyright and certain other rights. Our licenses are\n"
               "Section 5 -- Disclaimer of Warranties and Limitation of Liability.\n").encode()
    licensed = {**files, "usd/LICENSE.txt": licence}
    bill = _bom(licensed)
    served.update({BASE + "usd/LICENSE.txt": licence, BASE + BOM: bill,
                   BASE + "com.nvidia.simready.packaging.json": packaging.replace(
                       hashlib.sha256(bom).hexdigest(), hashlib.sha256(bill).hexdigest()).encode()})
    record = builder.package_record(builder.submission_urls(listing)[0], lambda url, cap: served[url])
    assert (record["copyright"], record["disclaimer"]) == ("Copyright (c) M-XR Limited.", None)
    assert record["attribution_request"] == ["Copyright (c) M-XR Limited. Attribution: marso by M-XR (https://m-xr.com)."]
    served[BASE + BOM] = bill + b" "
    with pytest.raises(ValueError, match="bom_digest_mismatch"):
        builder.package_record(builder.submission_urls(listing)[0], lambda url, cap: served[url])
    listing.write_text(f"https://huggingface.co/datasets/{REPO}/resolve/main/{ASSET}/com.nvidia.simready.packaging.json\n")
    with pytest.raises(ValueError, match="submission_line_not_pinned"):
        builder.submission_urls(listing)


#: A copyright notice and a disclaimer of warranties a package might supply (none of the reviewed ones does today).
SUPPLIED = {"package_license": "CC-BY-4.0", "asset_license": "CC-BY 4.0", "author": "NVIDIA",
            "copyright": "Copyright (c) 2026 Example Corporation.",
            "disclaimer": "This asset is provided as is, without warranties of any kind."}


@pytest.mark.parametrize("supplied", [False, True])
def test_the_package_credits_a_simready_model_with_the_owners_wording(tmp_path, supplied):
    """The settled credit, every file holding the model named, and the package's own copyright notice and disclaimer
    kept exactly where it supplies them; nothing in their place where it supplies none."""
    from blueprint_pipeline import outbound_file_scan as scan
    from blueprint_pipeline import website_company_simulation as sim
    from tests.test_website_company_simulation import Job
    indexes, fetch = _fixture(tmp_path, rights=[SUPPLIED] if supplied else None)
    job = Job(tmp_path / "job")
    capture = job.capture
    chosen = library._attribution(library.catalog(indexes)[0])
    subject = {"target_id": "box", "attribution": chosen, "pick": {"source": "model"},
               "files": asset.fetch_files(chosen, capture / "files", fetch=fetch)}
    made = supplied_mesh_configuration(
        {"authoring_backend": "artisan_v1", "metric_envelope": {"maximum_dimension_relative_error": 0.25}},
        synthetic=None, synthetic_authoring=None, parametric=None, subject_id="box", sim_min=[0, 0, 0],
        sim_max=[0.45, 0.3, 0.225], library=subject, library_root=capture / "pipeline/website_scene_preparation/m")
    (capture / "pipeline/website_scene_preparation/preparation.json").write_text(
        json.dumps({"authoring_inputs": {"configuration": made}}))
    source = sim.simulation_source(capture, queue_root=job.queue)
    sim.build_package(source, tmp_path / "package.zip", literals=[])
    with zipfile.ZipFile(tmp_path / "package.zip") as archive:
        readme = " ".join(archive.read("README.txt").decode().split())
        archive.extractall(tmp_path / "unpacked")
    notice = " (Copyright (c) 2026 Example Corporation.)" if supplied else ""
    disclaimer = (" Its own notice: This asset is provided as is, without warranties of any kind." if supplied
                  else "")
    assert (f"- scene/files/library_model.usda and object.usdz: {ASSET} by NVIDIA{notice}, from the SimReady dataset "
            f"{REPO} ({BASE}com.nvidia.simready.packaging.json), under CC BY 4.0 "
            "(https://creativecommons.org/licenses/by/4.0/). Blueprint converted it from USD and scaled it to fit "
            f"this job.{disclaimer}") in readme
    assert ("Copyright" in readme) is supplied and ("warrant" in readme) is supplied  # never one made up
    assert ("These files stay under their own license. Blueprint does not own them, and the terms Blueprint shares "
            "this package under do not restrict them.") in readme
    assert scan.scan_tree(tmp_path / "unpacked", []) == []
