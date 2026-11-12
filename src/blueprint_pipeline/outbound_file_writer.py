"""Every file that leaves Blueprint as a simulation file is written by Blueprint, never copied.

The rule: no file reaches the simulation package (``website_company_simulation``) or the job packet's simulation
folders (``job_packet_export``) byte for byte. Each is written here, from values Blueprint read, into a new file:
what the reader skips is not written, and every part that is kept is either a field of a fixed, checked length
(an image header, a PNG gamma or colour chunk, a JPEG frame header) or numeric payload (the one limit below). The
shared scan (``outbound_file_scan``) then runs over the output as the proof.

| Kind | Read | Written | Dropped |
|---|---|---|---|
| USD layer (``.usd``, ``.usda``, ``.usdc``) | ``Sdf.Layer.OpenAsAnonymous`` | ``Sdf.Layer.Export`` to the same format | anything the layer does not hold as data: text comments, bytes after a crate |
| PNG | chunk by chunk, every CRC checked, each kept chunk at exactly the length its type allows (IHDR 13, IEND 0, gAMA 4, cHRM 32, sRGB 1, pHYs 9, sBIT and tRNS by colour type, PLTE a multiple of 3 within the bit depth), the joined IDAT exactly one complete compressed stream; no decoder | the signature, then only IHDR, PLTE, IDAT, tRNS, gAMA, cHRM, sRGB, sBIT, pHYs, IEND, whole and in order; then decoded with Pillow as a check | every other ancillary chunk (tEXt, zTXt, iTXt, eXIf, iCCP, tIME, bKGD, hIST, sPLT, private chunks, animation chunks) and anything after IEND |
| JPEG | segment by segment, no decoder | SOI, DQT, DHT, DAC, DRI, SOF (C0 to CF except C4, C8, CC), SOS with its entropy-coded data, EOI; the 12-byte Adobe APP14 only; then decoded with Pillow as a check | every other APPn and every COM segment, wherever it stands, and anything after EOI |
| USDZ | member by member | a new stored archive: each member written by its kind above, names checked, 64-byte aligned data with zero-filled padding, fixed times, no comment, no other extra field | the archive comment, extra fields, any member that is not a USD layer, a PNG or a JPEG (refused) |

Pixels are unchanged (PNG and JPEG are not re-encoded). The one limit: numeric payload (pixel data, an image's
coding tables, palette and transparency tables, numeric arrays in a layer) is written as read and is not searched as text. These files come from
Blueprint's own tools and from asset tools, never from a person typing; the scan guards against what tools write
by accident (metadata, comments, names, paths, links), not against text deliberately encoded as numbers.
"""
from __future__ import annotations

import io
import struct
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_KEPT = (b"IHDR", b"PLTE", b"IDAT", b"tRNS", b"gAMA", b"cHRM", b"sRGB", b"sBIT", b"pHYs", b"IEND")
#: JPEG markers kept besides SOS: DQT, DHT, DAC, DRI and the SOF markers (C0 to CF except C4, C8, CC).
JPEG_KEPT = frozenset({0xDB, 0xC4, 0xCC, 0xDD} | {code for code in range(0xC0, 0xD0)} - {0xC8})
_JPEG_SOF = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
USD_LAYER_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
#: The extra-field id USD's own zip writer uses for its alignment padding.
_PAD_HEADER = 0x1986


class WriteRefused(ValueError):
    """The file cannot be written from what was read; ``detail`` says what it was."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


# --- PNG ----------------------------------------------------------------------------------------------------------

#: The length each kept fixed-size chunk must have (PNG specification); a longer one could carry free text.
_PNG_FIXED_LENGTHS = {b"IHDR": 13, b"IEND": 0, b"gAMA": 4, b"cHRM": 32, b"sRGB": 1, b"pHYs": 9}
#: sBIT's length and tRNS's by colour type (0 greyscale, 2 truecolour, 3 paletted, 4 grey + alpha, 6 RGBA).
_SBIT_LENGTHS = {0: 1, 2: 3, 3: 3, 4: 2, 6: 4}
_TRNS_LENGTHS = {0: 2, 2: 6}


def _png_chunk_fits(kind: bytes, body: bytes, header: bytes, palette_entries: int | None) -> bool:
    """A kept chunk has exactly the length its type allows, given the image's header."""
    if kind in _PNG_FIXED_LENGTHS:
        return len(body) == _PNG_FIXED_LENGTHS[kind]
    depth, colour = header[8], header[9]
    if kind == b"sBIT":
        return len(body) == _SBIT_LENGTHS.get(colour, -1)
    if kind == b"PLTE":
        limit = min(256, 2 ** depth) if colour == 3 else 256
        return colour in (2, 3, 6) and len(body) % 3 == 0 and 0 < len(body) // 3 <= limit
    if kind == b"tRNS":
        if colour == 3:
            return palette_entries is not None and 0 < len(body) <= palette_entries
        return len(body) == _TRNS_LENGTHS.get(colour, -1)
    return True  # IDAT: checked as one stream below


#: The bit depths each colour type allows, and its channels.
_PNG_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
#: Adam7: each pass's (first column, first row, column step, row step).
_ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def png_filtered_size(header: bytes) -> int | None:
    """The exact length of the inflated image data an IHDR calls for (a filter byte, then the packed row, for
    every row of every pass), or None for a header whose size cannot be computed (an illegal colour type and bit
    depth pair, a compression, filter or interlace method that is not defined, a zero side)."""
    width, height, depth, colour, compression, filtering, interlace = struct.unpack(">IIBBBBB", header)
    if (depth not in _PNG_DEPTHS.get(colour, ()) or compression or filtering or interlace not in (0, 1)
            or not width or not height):
        return None
    bits = depth * _PNG_CHANNELS[colour]

    def rows(columns: int, count: int) -> int:
        return 0 if not columns or not count else count * (1 + (columns * bits + 7) // 8)
    if interlace == 0:
        return rows(width, height)
    return sum(rows((width - x + dx - 1) // dx if width > x else 0, (height - y + dy - 1) // dy if height > y else 0)
               for x, y, dx, dy in _ADAM7)


def png_bytes(data: bytes) -> bytes:
    """The PNG rewritten from its kept chunks, or ``WriteRefused``.

    Each kept chunk must have the length its type allows, and the image data (IDAT, joined) must be exactly one
    complete compressed stream (no byte may follow its end) that inflates to exactly the filtered image size the
    header calls for (``png_filtered_size``). So a kept chunk holds only fixed-length fields and numeric payload
    (pixel rows, palette and transparency tables), never free text after or beside them.
    """
    refuse = WriteRefused("a PNG that cannot be read chunk by chunk")
    if not data.startswith(PNG_SIGNATURE):
        raise refuse
    out, position, ended = [PNG_SIGNATURE], len(PNG_SIGNATURE), False
    header, palette_entries, image_data = b"", None, []
    while position < len(data):
        if position + 12 > len(data):
            raise refuse
        length, kind = struct.unpack(">I4s", data[position:position + 8])
        end = position + 12 + length
        if end > len(data) or not kind.isalpha():
            raise refuse
        body = data[position + 8:position + 8 + length]
        crc = struct.unpack(">I", data[end - 4:end])[0]
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc or (not header and kind != b"IHDR"):
            raise refuse
        if kind == b"IHDR":
            if header or len(body) != 13 or png_filtered_size(body) is None:
                raise refuse
            header = body
        if kind in PNG_KEPT:
            if not _png_chunk_fits(kind, body, header, palette_entries):
                raise refuse
            if kind == b"PLTE":
                palette_entries = len(body) // 3
            if kind == b"IDAT":
                image_data.append(body)
            out.append(data[position:end])
        elif kind[:1].isupper():  # an unknown critical chunk: the image cannot be read without it
            raise refuse
        position = end
        if kind == b"IEND":
            ended = True
            break  # nothing after IEND is written
    if not ended or not image_data:
        raise refuse
    expected = png_filtered_size(header)
    stream, inflated = zlib.decompressobj(), 0
    try:
        for part in image_data:
            while part:
                # Bounded steps, and never more than one byte past the size the header calls for: a stream that
                # inflates to more (rows, then text; or a decompression bomb) is refused as soon as it shows.
                inflated += len(stream.decompress(part, min(1 << 24, expected - inflated + 1)))
                if inflated > expected:
                    raise refuse
                part = stream.unconsumed_tail
    except zlib.error:
        raise refuse from None
    if not stream.eof or stream.unused_data or inflated != expected:
        raise refuse  # a stream that does not end, bytes after its end, or not exactly the image's size
    written = b"".join(out)
    _decodes(written, "a PNG that cannot be read chunk by chunk")
    return written


# --- JPEG ---------------------------------------------------------------------------------------------------------

def _adobe(body: bytes) -> bool:
    return len(body) == 12 and body.startswith(b"Adobe")


def jpeg_bytes(data: bytes) -> bytes:
    """The JPEG rewritten from its kept segments, or ``WriteRefused``."""
    refuse = WriteRefused("a JPEG that cannot be read part by part")
    if not data.startswith(b"\xff\xd8"):
        raise refuse
    out, position, ended, framed = [b"\xff\xd8"], 2, False, False
    while position < len(data):
        if data[position] != 0xFF:
            raise refuse
        while position < len(data) and data[position] == 0xFF:  # fill bytes
            position += 1
        if position >= len(data):
            raise refuse
        marker = data[position]
        position += 1
        if marker == 0xD9:
            out.append(b"\xff\xd9")
            ended = True
            break  # nothing after EOI is written
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            raise refuse
        if position + 2 > len(data):
            raise refuse
        length = struct.unpack(">H", data[position:position + 2])[0]
        if length < 2 or position + length > len(data):
            raise refuse
        segment = data[position - 2:position + length]
        body = data[position + 2:position + length]
        position += length
        if marker == 0xDA:  # SOS, then its entropy-coded data up to the next real marker
            start = position
            while position + 1 < len(data):
                if data[position] == 0xFF and data[position + 1] != 0x00 and not 0xD0 <= data[position + 1] <= 0xD7:
                    break
                position += 1
            else:
                raise refuse
            out.append(segment + data[start:position])
        elif marker in JPEG_KEPT:
            framed |= marker in _JPEG_SOF
            out.append(segment)
        elif marker == 0xEE and _adobe(body):
            out.append(segment)
        elif 0xE0 <= marker <= 0xEF or marker == 0xFE:
            continue  # every APPn (bar the Adobe form) and COM is dropped
        else:
            raise refuse
    if not ended or not framed:
        raise refuse
    written = b"".join(out)
    _decodes(written, "a JPEG that cannot be read part by part")
    return written


def _decodes(data: bytes, detail: str) -> None:
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except Exception:  # noqa: BLE001 - any failure to decode refuses
        raise WriteRefused(detail) from None


# --- USD ----------------------------------------------------------------------------------------------------------

def usd_layer_bytes(data: bytes, suffix: str, *, scratch: Path | None = None) -> bytes:
    """A layer written by ``Sdf.Layer.Export`` from the opened layer, in the format its ending names."""
    with tempfile.TemporaryDirectory(prefix="outbound-write-", dir=scratch) as folder:
        source, target = Path(folder) / f"in{suffix}", Path(folder) / f"out{suffix}"
        source.write_bytes(data)
        write_usd_layer(source, target)
        return target.read_bytes()


def write_usd_layer(source: Path, target: Path, layer: object | None = None) -> None:
    """``Sdf.Layer.Export`` of ``layer`` (or of ``source`` opened) to ``target``; ``WriteRefused`` otherwise."""
    from pxr import Sdf
    try:
        layer = layer if layer is not None else Sdf.Layer.OpenAsAnonymous(str(source))
        if layer is None or not layer.Export(str(target)):
            raise WriteRefused("a USD layer that cannot be read")
    except WriteRefused:
        raise
    except Exception:  # noqa: BLE001 - any failure to open or write refuses
        raise WriteRefused("a USD layer that cannot be read") from None


# --- USDZ ---------------------------------------------------------------------------------------------------------

def member_name_ok(name: str) -> bool:
    """Relative, no ``..``, no leading slash, no backslash, no drive, no empty part."""
    pure = PurePosixPath(name)
    return (bool(name) and "\\" not in name and not name.startswith("/") and ":" not in name
            and ".." not in pure.parts and all(pure.parts) and not name.endswith("/"))


def member_kind(name: str, data: bytes) -> str:
    """``usd``, ``png``, ``jpeg`` or ``other``, by content."""
    if data.startswith(b"PXR-USDC") or data.startswith(b"#usda"):
        return "usd"
    if data.startswith(PNG_SIGNATURE):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return "other"


def usdz_members(source: Path, *, scratch: Path | None = None) -> list[tuple[str, bytes]]:
    """Every member of a usdz, written by its kind, in order; ``WriteRefused`` for anything else."""
    try:
        with zipfile.ZipFile(source) as archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            raw = [(info.filename, archive.read(info)) for info in infos]
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error):
        raise WriteRefused("a usdz that cannot be read") from None
    members = []
    for index, (name, data) in enumerate(raw):
        if not member_name_ok(name):
            raise WriteRefused("a usdz member name that is not a plain relative path")
        kind = member_kind(name, data)
        suffix = PurePosixPath(name).suffix.lower()
        if kind == "usd" and suffix in USD_LAYER_SUFFIXES:
            members.append((name, usd_layer_bytes(data, suffix, scratch=scratch)))
        elif kind == "png" and suffix == ".png":
            members.append((name, png_bytes(data)))
        elif kind == "jpeg" and suffix in (".jpg", ".jpeg"):
            members.append((name, jpeg_bytes(data)))
        else:
            raise WriteRefused("a usdz member that is not a USD layer, a PNG or a JPEG")
        if index == 0 and kind != "usd":
            raise WriteRefused("a usdz whose first member is not a USD layer")
    if not members:
        raise WriteRefused("a usdz with no member")
    return members


def write_usdz(members: list[tuple[str, bytes]], target: Path) -> None:
    """A stored archive, each member's data starting on a 64-byte boundary, padded with USD's own zero-filled
    extra field, with fixed times and no comment. The first member is the default layer."""
    with Path(target).open("wb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_STORED, 3, 0o100644 << 16
            start = stream.tell() + 30 + len(name.encode("utf-8"))
            pad = (-start) % 64
            if pad and pad < 4:
                pad += 64
            info.extra = struct.pack("<HH", _PAD_HEADER, pad - 4) + b"\x00" * (pad - 4) if pad else b""
            archive.writestr(info, data)


def rewrite_usdz(source: Path, target: Path, *, scratch: Path | None = None) -> None:
    write_usdz(usdz_members(source, scratch=scratch), target)


# --- by kind ------------------------------------------------------------------------------------------------------

def write_file(source: Path, target: Path, *, scratch: Path | None = None) -> None:
    """``source`` written into ``target`` by its kind (USD layer, usdz, PNG, JPEG); ``WriteRefused`` otherwise."""
    suffix = Path(source).suffix.lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix in USD_LAYER_SUFFIXES:
        write_usd_layer(source, target)
    elif suffix == ".usdz":
        rewrite_usdz(source, target, scratch=scratch)
    elif suffix == ".png":
        target.write_bytes(png_bytes(Path(source).read_bytes()))
    elif suffix in (".jpg", ".jpeg"):
        target.write_bytes(jpeg_bytes(Path(source).read_bytes()))
    else:
        raise WriteRefused("a kind of file that may not ship")


__all__ = ["JPEG_KEPT", "PNG_KEPT", "WriteRefused", "jpeg_bytes", "member_name_ok", "png_bytes", "rewrite_usdz",
           "usd_layer_bytes", "usdz_members", "write_file", "write_usd_layer", "write_usdz"]
