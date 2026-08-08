"""Prove the NuRec codec against the real InteriorGS volume before trusting it.

Isaac renders NuRec natively -- an InteriorGS scene in this format has already
been rendered with a full-size robot composited inside it -- while the same
appearance authored as an Omniverse ParticleField has never rendered correctly.
So the encoder is the thing that lets a ghost-removed appearance use the
renderer that works, and it has to be proven before a cent of GPU time is spent
on it.

The strong proof is byte-equality on the payload: decode the shipped volume,
re-encode the unmodified document, and require the original decompressed bytes
back.  Anything weaker would leave the encoder's field order, precision
handling, or container framing unverified, and those are exactly what a
renderer refuses on.  The gzip header is checked separately and held to a
single deliberate difference, the pinned mtime.
"""

from __future__ import annotations

import gzip
import zipfile
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.nurec_volume_codec import (
    GAUSSIAN_ARRAY_WIDTHS,
    NuRecCodecError,
    build_state_dict,
    decode_nurec_bytes,
    describe_volume,
    encode_nurec_bytes,
    gaussian_arrays,
    layer_precision,
)

INTERIORGS_USDZ = Path(
    "/Users/nijelhunt_1/workspace/BlueprintValidation/data/adp009a_tranche1_20260804"
    "/shortlist/SAGE-3D_InteriorGS_usdz/InteriorGS_usdz/840313.usdz"
)
NUREC_MEMBER = "840313.usdz.nurec"


def _shipped_payload() -> bytes:
    if not INTERIORGS_USDZ.is_file():
        pytest.skip("InteriorGS usdz not present in this checkout")
    with zipfile.ZipFile(INTERIORGS_USDZ) as archive:
        return archive.read(NUREC_MEMBER)


def _synthetic_arrays(count: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260808)
    return {
        name: rng.normal(size=(count, width)).astype(np.float16)
        for name, width in GAUSSIAN_ARRAY_WIDTHS.items()
    }


# --- the proof ---------------------------------------------------------------


def test_the_shipped_volume_round_trips_to_identical_payload_bytes() -> None:
    """Decode then re-encode the real volume and require the original payload.

    Byte-equality is the only proof that covers MessagePack framing, field
    order and precision handling at once -- the things a renderer refuses on
    and that an array-level comparison would silently pass.

    The comparison is on the *decompressed* payload rather than the gzip
    stream, because gzip embeds an mtime and the shipped file carries the
    timestamp of whoever built it.  That is container metadata, not content;
    pinning it is what makes our own output reproducible.  The next test holds
    the header itself to exactly that one difference, so this is not a licence
    for the framing to drift.
    """

    payload = _shipped_payload()
    document = decode_nurec_bytes(payload)
    assert gzip.decompress(encode_nurec_bytes(document)) == gzip.decompress(payload)


def test_the_reencoded_header_differs_only_in_the_pinned_mtime() -> None:
    """Bound what the round-trip above is allowed to excuse.

    A gzip header is magic, method, flags, mtime, extra flags and OS.  Only
    mtime (bytes 4..8) may differ, and only because we pin it deliberately.
    """

    payload = _shipped_payload()
    reencoded = encode_nurec_bytes(decode_nurec_bytes(payload))
    assert reencoded[:4] == payload[:4], "magic, method and flags must match"
    assert reencoded[4:8] != payload[4:8], "mtime is the pinned field"
    assert int.from_bytes(reencoded[4:8], "little") == 0


def test_the_shipped_volume_decodes_to_the_arrays_a_gaussian_layer_needs() -> None:
    arrays = gaussian_arrays(decode_nurec_bytes(_shipped_payload()))
    counts = {name: value.shape[0] for name, value in arrays.items()}
    assert len(set(counts.values())) == 1, counts
    for name, width in GAUSSIAN_ARRAY_WIDTHS.items():
        assert arrays[name].shape[1] == width, name
    # 45 is 15 non-DC spherical-harmonic coefficients across 3 channels, which
    # is exactly the shape Aura's sh_rest already has.
    assert arrays["features_specular"].shape[1] == 45


def test_values_are_stored_pre_activation() -> None:
    """Scales are logs and densities are logits; the renderer activates them.

    This is why the encoder writes learned parameters straight through: every
    activation applied on the authoring side is a place to be wrong about
    units, which is how a "structural" scale once became one metre.
    """

    document = decode_nurec_bytes(_shipped_payload())
    described = describe_volume(document)
    assert described["scale_activation"] == "exp"
    assert described["density_activation"] == "sigmoid"
    arrays = gaussian_arrays(document)
    # Log-space scales are mostly negative; activated ones could not be.
    assert float(arrays["scales"].astype(np.float32).mean()) < 0.0
    activated = np.exp(arrays["scales"].astype(np.float32))
    assert 1e-4 < float(np.median(activated)) < 1.0


def test_the_container_declares_a_planar_kernel_flag() -> None:
    """The flag is what lets Aura's 2DGS use this format at all.

    InteriorGS is 3DGS and sets it False; a planar field sets it True, which is
    what Aura's own rasterizer does and what the ParticleField surflet kernel
    was supposed to do.
    """

    described = describe_volume(decode_nurec_bytes(_shipped_payload()))
    assert described["density_kernel_planar"] is False
    assert described["radiance_sph_degree"] == 3
    assert described["renderer"] == "3dgut-nrend"


# --- encoder contracts -------------------------------------------------------


def test_encoding_is_deterministic() -> None:
    """gzip embeds an mtime; unpinned, a re-encode differs and proves nothing."""

    document = {"version": "0.2.576", "model": "nre", "config": {}, "state_dict": {}}
    assert encode_nurec_bytes(document) == encode_nurec_bytes(document)
    # And it really is gzip, not merely equal to itself.
    assert encode_nurec_bytes(document)[:2] == b"\x1f\x8b"


def test_synthetic_arrays_survive_a_full_round_trip() -> None:
    arrays = _synthetic_arrays()
    document = {
        "version": "0.2.576",
        "model": "nre",
        "config": {"layers": {"gaussians": {"precision": 16}}},
        "state_dict": build_state_dict(arrays, precision=16),
    }
    decoded = gaussian_arrays(decode_nurec_bytes(encode_nurec_bytes(document)))
    for name in GAUSSIAN_ARRAY_WIDTHS:
        np.testing.assert_array_equal(decoded[name], arrays[name])


def test_a_missing_array_fails_closed() -> None:
    arrays = _synthetic_arrays()
    del arrays["rotations"]
    with pytest.raises(NuRecCodecError) as excinfo:
        build_state_dict(arrays, precision=16)
    assert "nurec_array_missing:rotations" in excinfo.value.errors


def test_arrays_of_disagreeing_length_fail_closed() -> None:
    """A per-gaussian array shorter than the rest is not a smaller field.

    It is a field whose parameters do not correspond, and the renderer would
    read past the end of one of them.
    """

    arrays = _synthetic_arrays()
    arrays["densities"] = arrays["densities"][:3]
    with pytest.raises(NuRecCodecError) as excinfo:
        build_state_dict(arrays, precision=16)
    assert any("count_disagreement" in e for e in excinfo.value.errors)


def test_a_nonfinite_parameter_fails_closed() -> None:
    """The renderer could not report which array carried it."""

    arrays = _synthetic_arrays()
    arrays["positions"][0, 0] = np.float16("inf")
    with pytest.raises(NuRecCodecError) as excinfo:
        build_state_dict(arrays, precision=16)
    assert "nurec_array_nonfinite:positions" in excinfo.value.errors


def test_an_unsupported_precision_fails_closed() -> None:
    with pytest.raises(NuRecCodecError):
        build_state_dict(_synthetic_arrays(), precision=8)
    with pytest.raises(NuRecCodecError):
        layer_precision({"config": {"layers": {"gaussians": {"precision": 8}}}})


def test_a_payload_that_is_not_gzip_fails_closed() -> None:
    with pytest.raises(NuRecCodecError) as excinfo:
        decode_nurec_bytes(b"not a nurec volume")
    assert "nurec_payload_not_gzip" in excinfo.value.errors


def test_gzip_without_the_container_document_fails_closed() -> None:
    with pytest.raises(NuRecCodecError):
        decode_nurec_bytes(gzip.compress(b"\x80"))


# --- Aura 2DGS -> NuRec ------------------------------------------------------


def _aura_surfels(count: int = 8):
    from blueprint_pipeline.gaussian_splat_decode import GaussianSurfelData

    rng = np.random.default_rng(20260808)
    opacity = np.full(count, 3.0, dtype="float32")
    opacity[0] = np.inf  # the sealed PLY genuinely carries these
    return GaussianSurfelData(
        count=count,
        xyz=rng.normal(size=(count, 3)).astype("float32"),
        opacity=opacity,
        f_dc=rng.normal(size=(count, 3)).astype("float32"),
        scales=np.full((count, 2), -7.0, dtype="float32"),
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (count, 1)),
        sh_rest=rng.normal(size=(count, 45)).astype("float32"),
        mask_logits=np.zeros((count, 3), dtype="float32"),
        properties=(),
    )


def _authored():
    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    return build_aura_nurec_document(_aura_surfels(), template=_shipped_payload())


def test_aura_surfels_author_a_volume_that_round_trips() -> None:
    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    document = build_aura_nurec_document(_aura_surfels(), template=_shipped_payload())
    decoded = gaussian_arrays(decode_nurec_bytes(encode_nurec_bytes(document)))
    assert decoded["positions"].shape == (8, 3)
    assert decoded["features_specular"].shape == (8, 45)
    assert decoded["scales"].shape == (8, 3)


def test_the_planar_kernel_is_the_one_substantive_override() -> None:
    """Aura's field is planar; the template's is not.

    A 2D gaussian rendered by a volumetric kernel is not the same surface.
    Everything else -- projection, culling, render mode -- is NVIDIA's own
    and must be inherited rather than invented, because none of it can be
    tested locally.
    """

    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    template = decode_nurec_bytes(_shipped_payload())
    document = build_aura_nurec_document(_aura_surfels(), template=_shipped_payload())
    got = document["config"]["layers"]["gaussians"]["particle"]
    ref = template["config"]["layers"]["gaussians"]["particle"]
    assert got["density_kernel_planar"] is True
    assert ref["density_kernel_planar"] is False
    for key in ref:
        if key != "density_kernel_planar":
            assert got[key] == ref[key], key
    assert document["config"]["renderer"] == template["config"]["renderer"]


def test_the_structural_scale_is_flat_in_log_space() -> None:
    """NuRec stores scales pre-activation, so flat is a large negative log.

    Authoring a small *linear* value here would decode as exp(small) ~ 1, i.e.
    a one-metre thickness -- the same failure that buried the camera in opaque
    needles, arrived at from the opposite direction.
    """

    from blueprint_pipeline.aura_nurec_volume import describe_authored_volume

    described = describe_authored_volume(_authored())
    assert described["structural_is_flatter_than_planar"] is True
    assert described["activated_structural_median_m"] < described["activated_planar_median_m"]
    # And decisively flat, not merely smaller.
    assert described["activated_structural_median_m"] < 1e-3


def test_infinite_opacity_logits_are_clamped_and_counted() -> None:
    """float16 cannot hold +inf as data, and sigmoid is already 1.0 well before.

    Clamping preserves the decoded opacity exactly; leaving the infinity would
    make the buffer unreadable rather than opaque.  The count is reported so
    the substitution is visible rather than silent.
    """

    from blueprint_pipeline.aura_nurec_volume import FINITE_LOGIT_CLAMP

    document = _authored()
    assert document["_blueprint_authoring"]["infinite_opacity_logits_clamped"] == 1
    densities = gaussian_arrays(document)["densities"].astype(np.float32)
    assert np.isfinite(densities).all()
    assert densities.max() <= FINITE_LOGIT_CLAMP
    # Still saturating: the clamp must not change what the value means.
    assert float(1 / (1 + np.exp(-densities.max()))) > 0.999999


def test_learned_parameters_are_written_unactivated() -> None:
    """Every activation applied here is a chance to be wrong about units."""

    surfels = _aura_surfels()
    arrays = gaussian_arrays(_authored())
    np.testing.assert_allclose(
        arrays["scales"].astype(np.float32)[:, :2], surfels.scales, rtol=1e-2
    )
    np.testing.assert_allclose(
        arrays["features_albedo"].astype(np.float32), surfels.f_dc, atol=1e-2
    )


def test_a_template_that_is_not_a_gaussian_volume_fails_closed() -> None:
    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    with pytest.raises(NuRecCodecError):
        build_aura_nurec_document(_aura_surfels(), template={"config": {"layers": {}}})


# --- USDZ packaging ----------------------------------------------------------


def test_the_packaged_usdz_declares_a_default_prim(tmp_path) -> None:
    """A reference into a layer without one resolves to nothing.

    The previous appearance shipped, composed into no geometry at all, and
    five runs went looking for a render bug that was a missing metadata field.
    """

    from pxr import Usd

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz

    receipt = write_aura_nurec_usdz(_authored(), tmp_path / "aura.usdz")
    assert receipt["default_prim"] == "/World"
    stage = Usd.Stage.Open(str(tmp_path / "aura.usdz"))
    assert stage is not None
    assert str(stage.GetDefaultPrim().GetPath()) == "/World"


def test_the_packaged_usdz_is_a_nurec_volume(tmp_path) -> None:
    """The Volume composes one level deeper than it is authored.

    The root layer's over "gauss" references gauss.usda, whose default prim is
    World, so the Volume lands at /World/gauss/gauss -- exactly where the
    shipped InteriorGS package composes it too.
    """

    from pxr import Usd

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz

    write_aura_nurec_usdz(_authored(), tmp_path / "aura.usdz")
    stage = Usd.Stage.Open(str(tmp_path / "aura.usdz"))
    volume = stage.GetPrimAtPath("/World/gauss/gauss")
    assert volume and volume.IsValid()
    assert volume.GetAttribute("omni:nurec:isNuRecVolume").Get() is True
    fields = [
        p for p in stage.Traverse() if str(p.GetTypeName()) == "OmniNuRecFieldAsset"
    ]
    assert len(fields) == 2, [str(f.GetPath()) for f in fields]
    for field in fields:
        assert field.GetAttribute("filePath").Get() is not None


def test_the_world_rotation_is_identity_not_the_shipped_mirror(tmp_path) -> None:
    """InteriorGS mirrors because its positions are in a NuRec-internal frame.

    Its extent matches its raw positions and the matrix maps those to world.
    Aura's positions are already in the admitted world frame, so copying that
    matrix would mirror and rotate the room while looking entirely plausible.

    Only the translation may be non-trivial, and only to undo recentring.
    """

    from pxr import Usd

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz

    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    document = build_aura_nurec_document(
        _aura_surfels(), template=_shipped_payload(), recentre=False
    )
    receipt = write_aura_nurec_usdz(document, tmp_path / "aura.usdz")
    assert receipt["world_transform"] == "identity"
    stage = Usd.Stage.Open(str(tmp_path / "aura.usdz"))
    matrix = stage.GetPrimAtPath("/World/gauss/gauss").GetAttribute("xformOp:transform").Get()
    for row in range(4):
        for col in range(4):
            assert matrix[row][col] == (1.0 if row == col else 0.0)


def test_the_extent_is_taken_from_the_data_not_the_template(tmp_path) -> None:
    """A copied extent would crop the field to someone else's room."""

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz

    document = _authored()
    positions = gaussian_arrays(document)["positions"].astype(np.float32)
    receipt = write_aura_nurec_usdz(document, tmp_path / "aura.usdz")
    np.testing.assert_allclose(receipt["extent_min"], positions.min(axis=0), rtol=1e-5)
    np.testing.assert_allclose(receipt["extent_max"], positions.max(axis=0), rtol=1e-5)


def test_the_payload_round_trips_out_of_the_package(tmp_path) -> None:
    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz

    receipt = write_aura_nurec_usdz(_authored(), tmp_path / "aura.usdz")
    with zipfile.ZipFile(tmp_path / "aura.usdz") as archive:
        names = archive.namelist()
        assert receipt["payload_name"] in names
        assert "default.usda" in names and "gauss.usda" in names
        # USDZ members must be stored, not deflated, so they can be mapped.
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED, info.filename
        payload = archive.read(receipt["payload_name"])
    assert gaussian_arrays(decode_nurec_bytes(payload))["positions"].shape[0] == 8


def test_the_composed_prim_tree_matches_the_shipped_package(tmp_path) -> None:
    """Shape-for-shape against the package Isaac has actually rendered.

    Divergence here is the cheapest possible thing to catch locally and the
    most expensive to discover on a GPU.
    """

    from pxr import Usd

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz

    write_aura_nurec_usdz(_authored(), tmp_path / "aura.usdz")

    def tree(path: str) -> list[tuple[str, str]]:
        stage = Usd.Stage.Open(path)
        return [
            (str(p.GetPath()).replace("840313", "aura"), str(p.GetTypeName()))
            for p in stage.Traverse()
        ]

    if not INTERIORGS_USDZ.is_file():
        pytest.skip("InteriorGS usdz not present in this checkout")
    assert tree(str(tmp_path / "aura.usdz")) == tree(str(INTERIORGS_USDZ))


# --- Z-order ------------------------------------------------------------------


def test_the_shipped_payload_is_not_z_ordered() -> None:
    """And the trap that made it look like it was.

    ``np.diff(keys) >= 0`` on a uint32 array wraps on subtraction, so every
    difference is non-negative by construction and the answer is 1.000
    whatever the data says.  That reading nearly added a sort that would have
    made our payload differ from the only reference known to render.
    """

    from blueprint_pipeline.aura_nurec_volume import morton_order

    positions = gaussian_arrays(decode_nurec_bytes(_shipped_payload()))["positions"]
    order = morton_order(positions.astype(np.float32))
    identity = float((order == np.arange(order.size)).mean())
    assert identity < 0.01, "shipped payload is in arbitrary order, not Morton"


def test_the_unsigned_diff_trap_is_what_it_looks_like() -> None:
    """Pin the arithmetic directly, so the mistake cannot be made twice."""

    descending = np.array([9, 5, 1], dtype=np.uint32)
    assert float((np.diff(descending) >= 0).mean()) == 1.0, "unsigned wrap"
    assert float((np.diff(descending.astype(np.int64)) >= 0).mean()) == 0.0


def test_authored_volumes_are_not_z_ordered_by_default() -> None:
    """Matching the shipped payload, which is the only known-rendering shape."""

    assert _authored()["_blueprint_authoring"]["z_ordered"] is False


def test_every_array_is_permuted_by_the_same_index() -> None:
    """Otherwise the parameters stop corresponding to each other.

    A positions/colour mismatch renders a plausible-looking scene made of the
    wrong colours, which is far worse than a scene that fails to render.
    """

    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document, morton_order

    surfels = _aura_surfels()
    unsorted_doc = build_aura_nurec_document(surfels, template=_shipped_payload())
    sorted_doc = build_aura_nurec_document(
        surfels, template=_shipped_payload(), z_order=True
    )
    order = morton_order(np.asarray(surfels.xyz, dtype=np.float32))

    a = gaussian_arrays(unsorted_doc)
    b = gaussian_arrays(sorted_doc)
    for name in ("positions", "rotations", "scales", "densities", "features_albedo",
                 "features_specular"):
        np.testing.assert_array_equal(
            b[name], a[name][order], err_msg=f"{name} permuted inconsistently"
        )


def test_z_order_can_be_enabled_for_a_controlled_comparison() -> None:
    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    document = build_aura_nurec_document(
        _aura_surfels(), template=_shipped_payload(), z_order=True
    )
    assert document["_blueprint_authoring"]["z_ordered"] is True


def test_a_degenerate_axis_does_not_divide_by_zero() -> None:
    """A flat scene is still orderable."""

    from blueprint_pipeline.aura_nurec_volume import morton_order

    flat = np.zeros((5, 3), dtype=np.float32)
    flat[:, 0] = np.arange(5)
    assert morton_order(flat).shape == (5,)


def test_precision_can_be_raised_for_a_finer_field() -> None:
    """float16 displaces an Aura surfel by more than its own width.

    Rounding error is 0.93mm at p95 against a 0.81mm median surfel, so the
    field is smeared onto a grid coarser than its detail.  InteriorGS is
    unharmed by the same grid because its gaussians are 6.1mm.
    """

    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    surfels = _aura_surfels()
    doc16 = build_aura_nurec_document(surfels, template=_shipped_payload())
    doc32 = build_aura_nurec_document(
        surfels, template=_shipped_payload(), precision=32
    )
    assert doc16["config"]["layers"]["gaussians"]["precision"] == 16
    assert doc32["config"]["layers"]["gaussians"]["precision"] == 32
    # And the declared precision must match how the bytes were actually laid
    # out, or the decoder reads the array at the wrong stride.
    a16 = gaussian_arrays(decode_nurec_bytes(encode_nurec_bytes(doc16)))
    a32 = gaussian_arrays(decode_nurec_bytes(encode_nurec_bytes(doc32)))
    assert a16["positions"].dtype == np.float16
    assert a32["positions"].dtype == np.float32
    assert a32["positions"].shape == a16["positions"].shape


def test_float32_preserves_positions_that_float16_rounds_away() -> None:
    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    surfels = _aura_surfels()
    exact = np.asarray(surfels.xyz, dtype=np.float32)
    doc32 = build_aura_nurec_document(
        surfels, template=_shipped_payload(), precision=32
    )
    stored = gaussian_arrays(doc32)["positions"]
    # Recentred, so the stored value is the world position minus the offset the
    # volume's translation re-applies.  Comparing without it would assert the
    # payload is wrong precisely because it is right.
    centre = np.asarray(
        doc32["_blueprint_authoring"]["centre_offset_m"], dtype=np.float32
    )
    np.testing.assert_allclose(stored + centre, exact, rtol=0, atol=1e-6)


def test_recentring_halves_the_quantisation_error() -> None:
    """float16 resolution is relative to magnitude.

    The grid is coarse at 8.4m and fine near zero, so centring the field more
    than halves the rounding error -- 1.15x the median surfel width down to
    0.53x -- and costs nothing.  It is arithmetic, not a format feature the
    renderer might not implement.
    """

    from blueprint_pipeline.aura_nurec_volume import build_aura_nurec_document

    import numpy as _np

    rng = _np.random.default_rng(7)
    from blueprint_pipeline.gaussian_splat_decode import GaussianSurfelData

    n = 512
    far = rng.uniform(low=[0.2, -6.8, -0.8], high=[8.4, -1.9, 2.2], size=(n, 3)).astype("float32")
    surfels = GaussianSurfelData(
        count=n, xyz=far, opacity=_np.full(n, 3.0, dtype="float32"),
        f_dc=rng.normal(size=(n, 3)).astype("float32"),
        scales=_np.full((n, 2), -7.1, dtype="float32"),
        quats=_np.tile(_np.array([1, 0, 0, 0], dtype="float32"), (n, 1)),
        sh_rest=rng.normal(size=(n, 45)).astype("float32"),
        mask_logits=_np.zeros((n, 3), dtype="float32"), properties=(),
    )

    def p95(doc):
        stored = gaussian_arrays(doc)["positions"].astype(_np.float32)
        centre = _np.asarray(doc["_blueprint_authoring"]["centre_offset_m"], dtype=_np.float32)
        return float(_np.percentile(_np.abs((stored + centre) - far), 95))

    plain = build_aura_nurec_document(surfels, template=_shipped_payload(), recentre=False)
    centred = build_aura_nurec_document(surfels, template=_shipped_payload())
    assert p95(centred) < p95(plain) / 1.8, (p95(plain), p95(centred))
    assert plain["_blueprint_authoring"]["centre_offset_m"] == [0.0, 0.0, 0.0]
    assert any(centred["_blueprint_authoring"]["centre_offset_m"])


def test_the_offset_is_carried_as_a_translation_on_the_volume(tmp_path) -> None:
    """Or the room renders correctly, sharply, and metres from the arm."""

    from pxr import Usd

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz

    document = _authored()
    receipt = write_aura_nurec_usdz(document, tmp_path / "aura.usdz")
    centre = document["_blueprint_authoring"]["centre_offset_m"]
    assert receipt["world_translation_m"] == centre
    stage = Usd.Stage.Open(str(tmp_path / "aura.usdz"))
    matrix = stage.GetPrimAtPath("/World/gauss/gauss").GetAttribute("xformOp:transform").Get()
    for i in range(3):
        assert abs(matrix[3][i] - centre[i]) < 1e-5, (i, matrix[3][i], centre[i])
    # Rotation stays identity: Aura is already in the admitted world frame.
    for r in range(3):
        for c in range(3):
            assert matrix[r][c] == (1.0 if r == c else 0.0)
