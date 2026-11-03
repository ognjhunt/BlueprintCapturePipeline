"""A compile reads a USDA text collider once, into a binary copy; the published file is never touched.

capture-447's collider is a 292 MB USDA text file, and OpenUSD parsed it 22 times in one team packet,
transiently at about twelve times its size each time. The copy must read exactly as the text does.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")

from blueprint_pipeline import usd_binary_read_cache as cache  # noqa: E402


def _tokens(rng: np.random.Generator, count: int) -> list[str]:
    """Floats as text in every form a writer uses: shortest, 9 and 17 digits, exponents, integers, signed zero."""

    values = np.concatenate([rng.normal(0.0, 3.0, count), rng.uniform(-1e-30, 1e-30, count // 4),
                             rng.uniform(-3e38, 3e38, count // 4), np.round(rng.normal(0.0, 100.0, count // 4)),
                             [0.0, -0.0, 1e-45, 1.17549435e-38, 16777217.0, 123456789012345.0]])
    forms = [repr, "{:.9g}".format, "{:.17g}".format, "{:.3e}".format, "{:.6f}".format,
             lambda v: str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(v)]
    return [forms[index % len(forms)](float(value)) for index, value in enumerate(values)]


def write_text_collider(path: Path, *, seed: int = 1, count: int = 3000, extra: str = "") -> Path:
    """One collision mesh under ``/Root`` as USDA text, with every array type the reader takes."""

    rng = np.random.default_rng(seed)
    tokens = _tokens(rng, count)
    whole = len(tokens) // 3 * 3
    points = ", ".join(f"({tokens[i]}, {tokens[i + 1]}, {tokens[i + 2]})" for i in range(0, whole, 3))
    uv = ", ".join(f"({tokens[i]}, {tokens[i + 1]})" for i in range(0, whole - 1, 2))
    counts = ", ".join("3" for _ in range(whole // 3 // 3))
    indices = ", ".join(str(int(v)) for v in rng.integers(0, whole // 3, whole // 3 // 3 * 3))
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root"
{{
    def Mesh "mesh_0000_GLTF" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        float3[] extent = [(-1, -2, -3), (4, 5, 6)]
        int[] faceVertexCounts = [{counts}]
        int[] faceVertexIndices = [{indices}]
        normal3f[] normals = [{points}] (
            interpolation = "vertex"
        )
        uniform token physics:approximation = "none"
        bool physics:collisionEnabled = 1
        point3f[] points = [{points}]
        texCoord2f[] primvars:st = [{uv}] (
            interpolation = "faceVarying"
        )
        float[] widths = [{", ".join(tokens)}]
        double3 xformOp:translate = (0.25, 0, 1e-3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}{extra}
}}
''')
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("seed", [1, 2])
def test_large_arrays_read_with_numpy_are_bit_for_bit_what_openusd_parses(tmp_path, seed) -> None:
    from pxr import Sdf

    path = write_text_collider(tmp_path / "collision.usda", seed=seed)
    fast = cache._fast_layer(path, minimum_bytes=64)
    assert fast is not None
    parsed = Sdf.Layer.OpenAsAnonymous(str(path))
    assert fast.ExportToString() == parsed.ExportToString()
    for name in ("points", "normals", "primvars:st", "widths", "faceVertexCounts", "faceVertexIndices"):
        mine = np.asarray(fast.GetAttributeAtPath(f"/Root/mesh_0000_GLTF.{name}").default)
        theirs = np.asarray(parsed.GetAttributeAtPath(f"/Root/mesh_0000_GLTF.{name}").default)
        assert mine.dtype == theirs.dtype and mine.tobytes() == theirs.tobytes(), name


@pytest.mark.parametrize("literal", [
    "(1, 2, 3), # a comment\n (4, 5, 6)",   # a comment in the literal
    "(1, 2, 3), (4, 5, 6),",                # a trailing comma
    "(1, 2, 3), (4, 5)",                    # a short tuple
    "(1, 2, 3), (nan, 5, 6)",               # a word
    "(1, 2, 3), (12345678901234567, 5, 6)",  # an integer past double precision
])
def test_a_literal_the_reader_does_not_recognise_is_left_to_openusd(tmp_path, literal) -> None:
    path = tmp_path / "odd.usda"
    path.write_text(f'#usda 1.0\ndef Mesh "m"\n{{\n    point3f[] points = [{literal}]\n}}\n')
    assert cache._fast_layer(path, minimum_bytes=1) is None


def test_a_marker_the_skeleton_does_not_hold_once_declines(tmp_path) -> None:
    # A genuine one-point array that looks like the marker for literal 0: found twice, so the reader declines.
    path = write_text_collider(tmp_path / "collision.usda",
                               extra='\n    def Mesh "decoy"\n    {\n        point3f[] points = [(-1.5e38, 0, 0)]\n    }')
    assert cache._fast_layer(path, minimum_bytes=64) is None


def test_a_compile_reads_the_collider_once_and_never_touches_it(tmp_path, monkeypatch) -> None:
    from pxr import Sdf, Usd, UsdPhysics

    monkeypatch.setattr(cache, "LARGE_ARRAY_BYTES", 64)
    (tmp_path / "configured-scene").mkdir()
    source = write_text_collider(tmp_path / "configured-scene" / "collision.usda")
    digest, status = _sha(source), os.stat(source)
    staged = tmp_path / "packet" / "assets" / "collision.usda"
    staged.parent.mkdir(parents=True)
    os.link(source, staged)  # the packet stages it as a hard link
    assert cache.readable(source) == str(source) and cache.binary_copy(source, tmp_path / "copy") == source
    with cache.read_scope():
        copy = cache.binary_copy(source, tmp_path / "copy")
        assert copy.suffix == ".usdc" and copy.parent == tmp_path / "copy"
        assert cache.copy_method(source) == "numpy_large_arrays"
        assert cache.binary_copy(source, tmp_path / "other") == copy  # once per compile
        assert cache.readable(source) == cache.readable(staged) == str(copy)
        assert Sdf.Layer.FindOrOpen(str(copy)).ExportToString() == Sdf.Layer.OpenAsAnonymous(
            str(source)).ExportToString()
        # A step that may edit gets its own copy: its edit reaches neither the shared copy nor the file.
        own = Usd.Stage.Open(cache.editable(staged))
        UsdPhysics.MeshCollisionAPI(own.GetPrimAtPath("/Root/mesh_0000_GLTF")).GetApproximationAttr().Set("convexHull")
        shared = Usd.Stage.Open(cache.readable(source))
        assert UsdPhysics.MeshCollisionAPI(shared.GetPrimAtPath("/Root/mesh_0000_GLTF")).GetApproximationAttr().Get() == "none"
        del own, shared
    assert not copy.exists() and not copy.parent.exists()
    assert cache.readable(source) == str(source)
    after = os.stat(source)
    assert _sha(source) == digest and (after.st_size, after.st_mtime_ns) == (status.st_size, status.st_mtime_ns)


def test_a_copy_openusd_must_parse_is_still_made_once(tmp_path) -> None:
    # Arrays under the reader's size floor: OpenUSD parses the file, once, and the copy reads the same.
    from pxr import Sdf

    source = write_text_collider(tmp_path / "collision.usda", count=30)
    with cache.read_scope():
        copy = cache.binary_copy(source, tmp_path / "copy")
        assert cache.copy_method(source) == "openusd_text_parse"
        assert Sdf.Layer.FindOrOpen(str(copy)).ExportToString() == Sdf.Layer.OpenAsAnonymous(
            str(source)).ExportToString()


def test_a_collider_that_composes_other_files_is_not_copied(tmp_path) -> None:
    (tmp_path / "part.usda").write_text('#usda 1.0\ndef Mesh "m" {}\n')
    source = tmp_path / "collision.usda"
    source.write_text('#usda 1.0\ndef Xform "Root" (\n    prepend references = @./part.usda@</m>\n)\n{\n}\n')
    with cache.read_scope():
        assert cache.binary_copy(source, tmp_path / "copy") == source
        assert cache.readable(source) == str(source)
    assert not (tmp_path / "copy").exists()


def test_the_gpu_collision_adaptation_writes_the_same_bytes_from_its_own_copy(tmp_path, monkeypatch) -> None:
    # It may convert convex colliders in memory before exporting: from the compile's copy it writes the very
    # file it wrote from the text, and its edit stays its own.
    from pxr import Usd, UsdPhysics

    from blueprint_pipeline.native_task_arena_runtime import author_gpu_compatible_scene_collision

    monkeypatch.setattr(cache, "LARGE_ARRAY_BYTES", 64)
    source = write_text_collider(tmp_path / "collision.usda")
    text = source.read_text().replace('physics:approximation = "none"', 'physics:approximation = "convexHull"')
    source.write_text(text)
    plain = author_gpu_compatible_scene_collision(source, tmp_path / "plain" / "collision.usda")
    with cache.read_scope():
        cache.binary_copy(source, tmp_path / "copy")
        copied = author_gpu_compatible_scene_collision(source, tmp_path / "copied" / "collision.usda",
                                                       opened=cache.editable(source))
        shared = Usd.Stage.Open(cache.readable(source))
        assert UsdPhysics.MeshCollisionAPI(shared.GetPrimAtPath("/Root/mesh_0000_GLTF")).GetApproximationAttr().Get() == (
            "convexHull")
        del shared
    assert plain is not None and plain == copied
    assert (tmp_path / "plain" / "collision.usda").read_bytes() == (tmp_path / "copied" / "collision.usda").read_bytes()
    assert source.read_text() == text


def test_background_triangles_measured_in_blocks_are_the_same_triangles(tmp_path, monkeypatch) -> None:
    # The carry search loads the scene near each stance; a room's whole triangle array held 2.1 GB at once.
    from blueprint_pipeline import scene_physics_preflight as preflight
    from tests.test_team_embodiment_stance_room_mesh import write_scanned_room

    room = write_scanned_room(tmp_path / "collision.usda")
    whole = preflight.load_background_triangles(room, centre=[-0.4, -1.2, 0.8], radius=1.3, world_from_stage=np.eye(4))
    monkeypatch.setattr(preflight, "BACKGROUND_TRIANGLE_BLOCK", 7)
    blocks = preflight.load_background_triangles(room, centre=[-0.4, -1.2, 0.8], radius=1.3,
                                                 world_from_stage=np.eye(4))
    assert len(whole) > 100 and blocks.dtype == whole.dtype and blocks.tobytes() == whole.tobytes()
