"""No provider calls: an Atlas GLB is placed in the splat's frame and trimmed to observed support."""
import json

import numpy as np
import pytest

from blueprint_pipeline import atlas_mesh_registration as registration
from blueprint_pipeline.atlas_depth_splat import PLY_PROPERTIES
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

VOXEL = 0.01


def grid(z, extent=0.5, step=VOXEL, x0=0.0):
    axis = np.arange(-extent, extent + 1e-9, step)
    xs, ys = np.meshgrid(axis + x0, axis)
    vertices = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)], axis=1)
    n = len(axis)
    corner = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    faces = np.concatenate([np.stack([corner, corner + 1, corner + n], 1),
                            np.stack([corner + 1, corner + n + 1, corner + n], 1)])
    return vertices, faces


def splat_root(root, points=None):
    if points is None:
        points, _ = grid(-2.0)
    count = len(points)
    splat = SplatData(count=count, xyz=points.astype(np.float32), opacity=np.full(count, 3.0, np.float32),
                      f_dc=np.tile(np.array([[1.0, -0.5, 0.0]], np.float32), (count, 1)),
                      scales=np.full((count, 3), np.log(VOXEL), np.float32),
                      quats=np.tile(np.array([[1.0, 0, 0, 0]], np.float32), (count, 1)), properties=PLY_PROPERTIES)
    ply = write_standard_3dgs_ply(splat, root / "atlas_depth_splat.ply")
    manifest = {"voxel_size": VOXEL, "units": "provider_world_units",
                "outputs": [{"role": "splat_ply", "path": ply.name, "sha256": _sha256_file(ply)}]}
    manifest["digest"] = canonical_digest(manifest, digest_field="digest")
    (root / "atlas_depth_splat_manifest.json").write_text(json.dumps(manifest))
    return root


def exported_glb(path, *meshes, sign=(1.0, -1.0, -1.0)):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        vertices.append(v * np.array(sign))
        faces.append(f + offset)
        offset += len(v)
    vertices = np.concatenate(vertices)
    return registration.write_glb_triangles(path, vertices, np.concatenate(faces),
                                            np.full((len(vertices), 3), 128, np.uint8))


def test_glb_round_trip_preserves_geometry(tmp_path):
    vertices, faces = grid(-1.0, extent=0.05)
    path = registration.write_glb_triangles(tmp_path / "m.glb", vertices, faces, np.zeros((len(vertices), 3)))
    read_vertices, read_faces = registration.read_glb_triangles(path)
    assert np.allclose(read_vertices, vertices, atol=1e-6) and np.array_equal(read_faces, faces)


def test_rotated_export_is_registered_and_unseen_closing_sheets_are_trimmed(tmp_path):
    root = splat_root(tmp_path / "splat")
    observed, phantom = grid(-2.0, extent=0.3), grid(-1.6, extent=0.15)
    glb = exported_glb(tmp_path / "mesh.glb", observed, phantom)
    record = registration.register_atlas_mesh(glb=glb, splat_root=root, output_root=tmp_path / "registered")
    assert record["glb_frame"] == "atlas_world_rotated_180_x"
    assert record["median_residual_by_frame"]["atlas_world_rotated_180_x"] < 1e-6
    assert record["faces_kept"] == len(observed[1]) and record["faces_in"] == len(observed[1]) + len(phantom[1])
    assert record["digest"] == canonical_digest(record, digest_field="digest")
    vertices, _ = registration.read_glb_triangles(tmp_path / "registered" / "atlas_mesh_supported.glb")
    assert np.allclose(vertices[:, 2], -2.0, atol=1e-6)


def test_mesh_that_fits_no_declared_frame_fails_closed(tmp_path):
    root = splat_root(tmp_path / "splat")
    glb = exported_glb(tmp_path / "mesh.glb", grid(-2.0, x0=0.6), sign=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="atlas_mesh_not_registered_to_splat"):
        registration.register_atlas_mesh(glb=glb, splat_root=root, output_root=tmp_path / "registered")


def test_closing_sheets_can_outnumber_observed_vertices_without_hiding_source_overlap(tmp_path):
    root = splat_root(tmp_path / "splat")
    observed, phantom = grid(-2.0), grid(-1.6, step=VOXEL / 2)
    glb = exported_glb(tmp_path / "mesh.glb", observed, phantom)
    record = registration.register_atlas_mesh(glb=glb, splat_root=root, output_root=tmp_path / "registered")
    support = record["frame_support"]["atlas_world_rotated_180_x"]
    assert support["mesh_to_source_median"] is None
    assert support["mesh_supported_fraction"] < 0.5
    assert support["source_supported_fraction"] == 1.0
    assert record["registration_distance_limit"] == 3 * VOXEL
    assert record["faces_kept"] == len(observed[1])
    assert record["median_residual_by_frame"]["atlas_world"] is None
    json.dumps(record, allow_nan=False)


def test_triangle_with_supported_centroid_but_unseen_vertices_is_removed(tmp_path):
    root = splat_root(tmp_path / "splat")
    observed = grid(-2.0)
    bridge = (np.array([[-10, 0, -2], [5, -8.66, -2], [5, 8.66, -2]]), np.array([[0, 1, 2]]))
    glb = exported_glb(tmp_path / "mesh.glb", observed, bridge)
    record = registration.register_atlas_mesh(glb=glb, splat_root=root, output_root=tmp_path / "registered")
    assert record["faces_kept"] == len(observed[1])


def test_two_supported_declared_frames_are_ambiguous(tmp_path):
    root = splat_root(tmp_path / "splat", np.concatenate([grid(-2.0)[0], grid(2.0)[0]]))
    glb = exported_glb(tmp_path / "mesh.glb", grid(-2.0))
    with pytest.raises(ValueError, match="atlas_mesh_frame_ambiguous"):
        registration.register_atlas_mesh(glb=glb, splat_root=root, output_root=tmp_path / "registered")
