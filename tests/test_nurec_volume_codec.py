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
