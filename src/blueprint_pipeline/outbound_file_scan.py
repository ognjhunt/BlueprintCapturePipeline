"""One scan for every file that leaves Blueprint: ids, keys, host paths and links.

The job packet tool (``job_packet_export``) and the simulation package for deployment companies
(``website_company_simulation``) both read every file they are about to hand over with this module
and refuse, naming the file, when one holds something that says who or where the site is, where
Blueprint keeps things, or how to reach it.

What it looks for:

- the caller's own values (``literals``): the job's ids, account ids, delivery keys and the like.
  An id is looked for in its hyphen form and its underscore form (``capture-1`` and ``capture_1``);
- storage links (``gs://``, ``s3://``), local file links (``file://``) and host paths
  (``/var/lib/blueprint``, ``/mnt/blueprint-work``, ``/home/``, ``/tmp/``, ``/opt/``, a Windows
  drive path);
- web links (``http://``, ``https://``), except XML namespace identifiers, which point at no site;
- in text (a text file, the text form of a USD layer, an image's text metadata) also bare web
  addresses: a word with a dot and a known top-level domain. File names such as ``scene.usd`` and
  version numbers are left alone, because their last part is not a top-level domain;
- tokens and keys in the forms secret scanners know (provider API keys, cloud access key ids,
  signed-link query parameters, private key blocks, bearer and web tokens).

Binary USD (``.usdc``, and every USD layer inside a ``.usdz``) is opened with ``pxr`` and exported to
text, and that text is read too, so an id kept as a crate string is found. A USD file that cannot be
opened is a finding, never skipped. Zip members are read one by one.

A finding names the file, the kind and where. It never repeats what matched.
"""
from __future__ import annotations

import re
import struct
import tempfile
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

#: Links a file may carry: XML namespace identifiers (an SVG, XML or USD header names them).
ALLOWED_LINK_PREFIXES = (b"http://www.w3.org/",)
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
TEXT_SUFFIXES = frozenset({".usda", ".json", ".txt", ".md", ".csv", ".yaml", ".yml", ".xml", ".svg", ".mtl",
                           ".obj", ".urdf", ".sdf", ".html", ".toml", ".ini", ".cfg"})
CHUNK_BYTES = 8 << 20
UNREADABLE_ZIP = "a zip that cannot be read"
UNREADABLE_USD = "a USD file that cannot be opened"
UNREADABLE_FILE = "a file that cannot be read"
UNREADABLE = frozenset({UNREADABLE_ZIP, UNREADABLE_USD, UNREADABLE_FILE})

STORAGE_LINK, HOST_PATH, FILE_LINK, WEB_LINK, WEB_ADDRESS, KEY = (
    "a storage link", "a host path", "a local file link", "a web link", "a web address", "a key or token")
OTHER_LINK, IP_ADDRESS, EMAIL = "a link", "an IP address", "an email address"
#: The site's identity: its name, organisation, typed address, contact and submitter names, email and phone. The
#: site's description of its job is shared by its own agreement and is not looked for.
SITE_IDENTITY = "the site's name or contact details"
SITE_WORDS = SITE_IDENTITY  # the earlier name, kept for callers
#: A normalised identity value (lower-case, every non-alphanumeric removed) is looked for in normalised text when it
#: has at least ``NORMALISED_TEXT_MINIMUM`` characters, and in binary data only from ``NORMALISED_BINARY_MINIMUM``
#: (a random 5-letter string would match about twice in 512 MiB of binary; 10 letters about 3e-8 times). A value of
#: digits only (a phone) needs ``NORMALISED_DIGITS_MINIMUM`` digits: the text form of float geometry is mostly
#: digits once its separators are removed.
NORMALISED_TEXT_MINIMUM, NORMALISED_BINARY_MINIMUM, NORMALISED_DIGITS_MINIMUM = 5, 10, 11
_WORD, _NORMALISED = b"\x00w:", b"\x00n:"
_NOT_ALNUM = re.compile(rb"[^a-z0-9]+")
#: A binary USD layer larger than this is read by walking its specs (every name, string, token, asset path and
#: metadata value; numeric arrays skipped) instead of exporting all its geometry to text.
SPEC_WALK_BYTES = 8 << 20

#: Checked on every byte of every file, binary included, in this order (after the caller's own values). Only a
#: pattern with at least six fixed bytes is here: in 512 MiB of random bytes each is expected to match by chance
#: fewer than 1 time in 10,000 (``/data/``, the weakest, about 2e-6; ``file:/`` about 3e-5). Shorter patterns would
#: match ordinary float geometry (``~/`` about 8 times a MiB) and are in ``_TEXT_PATTERNS``.
_BINARY_PATTERNS: tuple[tuple[re.Pattern[bytes], str], ...] = (
    (re.compile(rb"/var/+lib/+blueprint|/mnt/+blueprint-work"
                rb"|(?<![A-Za-z0-9_.])/(?:home|root|Users|data)/"), HOST_PATH),
    (re.compile(rb"(?i)\bfile:/"), FILE_LINK),
    (re.compile(
        rb"(?:(?<![A-Za-z0-9])sk-(?:proj-|live-|svcacct-|ant-)?[A-Za-z0-9_-]{20,}"
        rb"|(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"
        rb"|(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}|(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{30,}"
        rb"|(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}|(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}"
        rb"|(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}"
        rb"|(?<![A-Za-z0-9])(?:ntn|secret)_[A-Za-z0-9]{40,}|(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{20,}"
        rb"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
        rb"|(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\."
        rb"|(?i:(?:aws_)?secret_access_key\s*[=:])"
        rb"|(?i:(?<![A-Za-z0-9])bearer\s+[A-Za-z0-9._~+/-]{20,}))"), KEY),
)
#: Looked for only in text: text files, the text form (or spec walk) of a USD layer, image text metadata. Each
#: would match random bytes by chance in 512 MiB: ``gs://``/``s3://`` about 0.002 times, ``/tmp/``, ``/opt/`` and
#: ``/srv/`` about 0.0015, a signed-link parameter about 0.005, any ``scheme://`` about 1,000, a UNC share about
#: 500, a Windows drive path about 1,600 and ``~/`` about 4,000.
_TEXT_PATTERNS: tuple[tuple[re.Pattern[bytes], str], ...] = (
    (re.compile(rb"(?i)\b(?:gs|s3)://"), STORAGE_LINK),
    (re.compile(rb"(?<![A-Za-z0-9_.])/(?:tmp|opt|srv)/|(?<![A-Za-z0-9_.])~/"), HOST_PATH),
    (re.compile(rb"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'@<>|:*?\\/\x00-\x1f]{1,64}[\\/]"), HOST_PATH),
    (re.compile(rb"(?<![A-Za-z0-9_\\])\\\\[A-Za-z0-9][A-Za-z0-9._$-]{0,62}\\[A-Za-z0-9._$ -]{1,80}"), HOST_PATH),
    (re.compile(rb"(?i:[?&](?:access_token|refresh_token|id_token|token|api_key|apikey|key|secret|password|sig"
                rb"|signature|x-amz-signature|x-goog-signature|x-amz-credential|x-goog-credential)=[^\s&\"'<>]{8,})"),
     KEY),
)
_WEB_LINK = re.compile(rb"(?i)https?://")
#: Any other scheme: Blueprint's own offload (``b2://``), ``ftp://``, ``omniverse://`` and the like.
_ANY_LINK = re.compile(rb"(?i)(?<![A-Za-z0-9+.-])[a-z][a-z0-9+.-]{1,15}://")
#: In text only. Four dotted numbers 0 to 255 standing alone (a version string has three parts or more digits).
_IP_ADDRESS = re.compile(rb"(?<![0-9A-Za-z.])(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
                         rb"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?![0-9A-Za-z]|\.[0-9])")
_EMAIL = re.compile(rb"(?<![A-Za-z0-9._%+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@[A-Za-z0-9-]{1,63}"
                    rb"(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}(?![A-Za-z0-9-])")
#: Common top-level domains, without those that are also common file name endings (md, py, sh, so, in, pl, rs).
_TLDS = ("com", "net", "org", "io", "ai", "co", "dev", "app", "gov", "edu", "info", "biz", "cloud", "xyz", "site",
         "online", "tech", "me", "us", "uk", "de", "fr", "ca", "au", "nl", "eu", "jp", "cn", "br", "es", "it", "se",
         "ch", "ly", "tv", "gg", "int", "mil", "store", "shop", "company", "email", "link", "page", "ws", "be", "at",
         "dk", "no", "fi", "ie", "nz", "za", "kr", "ru", "pt", "mx", "sg", "hk", "tw")
_WEB_ADDRESS = re.compile(
    rb"(?i)(?<![A-Za-z0-9_./\\-])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:"
    + b"|".join(name.encode() for name in _TLDS) + rb")(?![A-Za-z0-9_-]|\.[A-Za-z0-9])")


@dataclass(frozen=True)
class Finding:
    """What was found, never the matched text: the file, the kind, and where in it."""

    file: str
    kind: str
    where: str

    def sentence(self) -> str:
        return f"{self.file}: {self.kind}" if self.kind in UNREADABLE else f"{self.file} holds {self.kind}"


def literal_forms(literals: Iterable[tuple[str | bytes, str]], *, minimum: int = 4) -> list[tuple[bytes, str]]:
    """Each value with its kind, plus its underscore form (and hyphen form) when it has one; order kept."""
    found: list[tuple[bytes, str]] = []
    seen: set[bytes] = set()
    for value, kind in literals:
        raw = value.encode() if isinstance(value, str) else value
        if not isinstance(raw, bytes) or len(raw) < minimum:
            continue
        for form in (raw, raw.replace(b"-", b"_"), raw.replace(b"_", b"-")):
            if form not in seen:
                seen.add(form)
                found.append((form, kind))
    return found


def _blank_allowed_links(window: bytes) -> bytes:
    """The window with every allowed namespace link replaced by spaces, so no other check sees it."""
    if b"://" not in window:
        return window
    out = bytearray(window)
    for prefix in ALLOWED_LINK_PREFIXES:
        start = window.find(prefix)
        while start >= 0:
            end = start + len(prefix)
            while end < len(window) and window[end] not in b" \t\r\n\"'<>@)":
                end += 1
            out[start:end] = b" " * (end - start)
            start = window.find(prefix, end)
    return bytes(out)


def _normalised(text: str) -> bytes:
    return _NOT_ALNUM.sub(b"", text.lower().encode("utf-8", "replace"))


#: An address part that is only a country says nothing about which site it is.
_COUNTRIES = frozenset({"uk", "u.k.", "united kingdom", "england", "scotland", "wales", "northern ireland", "usa",
                        "u.s.a.", "us", "united states", "united states of america", "america", "canada", "ireland",
                        "france", "germany", "deutschland", "spain", "italy", "netherlands", "belgium", "australia",
                        "new zealand", "india", "japan", "china", "mexico", "brazil"})


def site_identity(values: Iterable[str], *, addresses: Iterable[str] = ()) -> list[tuple[bytes, str]]:
    """The site's identity as needles: each value (and each address part) as whole words without regard to case,
    and in its normalised form (lower-case, separators removed) so ``Zebra Plant Foods`` is also found as
    ``ZebraPlant_Foods``, ``zebra-plant-foods`` or ``zebraplantfoods``.

    A value shorter than three characters is not looked for. An address is looked for whole and by each line or
    comma-separated part of three characters or more that is not just a country or a number.
    """
    found: dict[bytes, None] = {}

    def add(text: str) -> None:
        text = " ".join(str(text).split())
        if len(text) < 3:
            return
        found[_WORD + text.lower().encode("utf-8", "replace")] = None
        normal = _normalised(text)
        # An email is looked for as written: its normalised form would also match a link to its domain.
        if "@" not in text and len(normal) >= NORMALISED_TEXT_MINIMUM and (not normal.isdigit() or len(normal) >= NORMALISED_DIGITS_MINIMUM):
            found[_NORMALISED + normal] = None
    for value in values:
        add(value)
    for address in addresses:
        add(address)
        for part in re.split(r"[,\n;]+", str(address)):
            part = " ".join(part.split())
            if (len(part) >= 3 and part.lower() not in _COUNTRIES
                    and not re.fullmatch(r"[0-9 #-]+", part)):
                add(part)
    return [(value, SITE_IDENTITY) for value in found]


def site_words(values: Iterable[str], *, phrases: Iterable[str] = (), minimum: int = 8) -> list[tuple[bytes, str]]:
    """The earlier name: identity values only (``phrases`` are no longer looked for)."""
    return site_identity(values)


#: Record fields that name the site or a person there, or say how to reach them.
_IDENTITY_KEYS = frozenset({"site_name", "organisation", "organization", "company_name", "contact_name",
                            "submitter_name", "owner_name", "email", "contact_email", "submitter_email", "phone",
                            "contact_phone", "submitter_phone"})
_ADDRESS_KEYS = frozenset({"address", "street", "street_address", "typed_address", "site_address"})
#: Branches whose ``name`` is a site's or a person's (elsewhere ``name`` names an object or a part).
_IDENTITY_BRANCHES = frozenset({"site_identity", "contact", "submitter", "site"})
SITE_RECORDS = ("capture_descriptor.json", "raw/manifest.json", "pipeline/website_task_context.json",
                "pipeline/website_scene_preparation/preparation.json")


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def record_site_identity(records: Iterable[object]) -> list[tuple[bytes, str]]:
    """The site's identity from parsed records: its name, organisation, address, contact and submitter names,
    email and phone. Never its description of the job, labels, the item or its free-text answers."""
    values: list[str] = []
    addresses: list[str] = []

    def walk(value: object, in_identity: bool = False) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in _ADDRESS_KEYS:
                    addresses.extend(_strings(item))
                elif key in _IDENTITY_KEYS or (in_identity and key == "name"):
                    values.extend(_strings(item))
                else:
                    walk(item, in_identity or key in _IDENTITY_BRANCHES)
        elif isinstance(value, list):
            for item in value:
                walk(item, in_identity)
    for record in records:
        walk(record)
    return site_identity(values, addresses=addresses)


record_site_words = record_site_identity  # the earlier name, kept for callers


def capture_site_identity(capture_root: Path) -> list[tuple[bytes, str]]:
    """``record_site_identity`` over the capture's records on disk (a record that does not read is skipped)."""
    import json
    records = []
    for relative in SITE_RECORDS:
        path = Path(capture_root) / relative
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")) if not path.is_symlink() else {})
        except (OSError, ValueError):
            continue
    return record_site_identity(records)


capture_site_words = capture_site_identity  # the earlier name, kept for callers


def _whole_word(haystack: bytes, needle: bytes) -> int:
    """The first place ``needle`` stands as whole words in ``haystack`` (no letter or digit either side), or -1."""
    start = haystack.find(needle)
    while start >= 0:
        end = start + len(needle)
        before = haystack[start - 1:start]
        after = haystack[end:end + 1]
        if not (before.isalnum() or after.isalnum()):
            return start
        start = haystack.find(needle, start + 1)
    return -1


def _match_window(window: bytes, literals: Sequence[tuple[bytes, str]], *, text: bool) -> tuple[str, int] | None:
    lowered = normal = None
    for value, kind in literals:
        if value.startswith(_WORD):
            lowered = window.lower() if lowered is None else lowered
            position = _whole_word(lowered, value[len(_WORD):])
        elif value.startswith(_NORMALISED):
            needle = value[len(_NORMALISED):]
            if len(needle) < (NORMALISED_TEXT_MINIMUM if text else NORMALISED_BINARY_MINIMUM):
                continue
            lowered = window.lower() if lowered is None else lowered
            normal = _NOT_ALNUM.sub(b"", lowered) if normal is None else normal
            position = 0 if needle in normal else -1  # the place in the original is not kept
        else:
            position = window.find(value)
        if position >= 0:
            return kind, position
    cleaned = _blank_allowed_links(window)
    for pattern, kind in _BINARY_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return kind, match.start()
    match = _WEB_LINK.search(cleaned)
    if match:
        return WEB_LINK, match.start()
    if text:
        for pattern, kind in (*_TEXT_PATTERNS, (_ANY_LINK, OTHER_LINK), (_EMAIL, EMAIL), (_WEB_ADDRESS, WEB_ADDRESS),
                              (_IP_ADDRESS, IP_ADDRESS)):
            match = pattern.search(cleaned)
            if match:
                return kind, match.start()
    return None


def scan_stream(stream: BinaryIO, literals: Sequence[tuple[bytes, str]], *, text: bool) -> tuple[str, int] | None:
    """The kind and byte offset of the first finding in ``stream``, read in overlapping chunks, or None."""
    overlap = max([len(value) for value, _ in literals] + [512]) + 4096
    tail, consumed = b"", 0
    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
        window = tail + chunk
        found = _match_window(window, literals, text=text)
        if found:
            return found[0], consumed - len(tail) + found[1]
        consumed += len(chunk)
        tail = window[-overlap:]
    return None


def _image_text(data: bytes, suffix: str) -> Iterator[tuple[str, bytes]]:
    """Text metadata of a PNG (tEXt, zTXt, iTXt, eXIf) or a JPEG (APPn and COM segments), decompressed."""
    if suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n"):
        position = 8
        while position + 8 <= len(data):
            length, kind = struct.unpack(">I4s", data[position:position + 8])
            body = data[position + 8:position + 8 + length]
            if kind == b"tEXt" or kind == b"eXIf":
                yield kind.decode(), body
            elif kind == b"zTXt":
                keyword, _, rest = body.partition(b"\x00")
                yield "zTXt", keyword + b" " + _inflate(rest[1:])
            elif kind == b"iTXt":
                keyword, _, rest = body.partition(b"\x00")
                compressed, rest = rest[:1], rest[2:]
                language, _, rest = rest.partition(b"\x00")
                translated, _, value = rest.partition(b"\x00")
                yield "iTXt", b" ".join((keyword, language, translated,
                                         _inflate(value) if compressed == b"\x01" else value))
            if kind == b"IEND":
                return
            position += 12 + length
    elif suffix in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8"):
        position = 2
        while position + 4 <= len(data) and data[position] == 0xFF:
            marker = data[position + 1]
            if marker == 0xDA:  # start of scan: no more metadata
                return
            length = struct.unpack(">H", data[position + 2:position + 4])[0]
            if 0xE0 <= marker <= 0xEF or marker == 0xFE:
                yield f"segment {marker:02X}", data[position + 4:position + 2 + length]
            position += 2 + length


def _inflate(value: bytes) -> bytes:
    try:
        return zlib.decompressobj().decompress(value, 64 << 20)
    except zlib.error:
        return b""


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_IMAGE_METADATA_LIMIT = 256 << 20


_TEXT_ARRAYS = frozenset({"StringArray", "TokenArray", "AssetPathArray"})


def _textual(value: object) -> bytes | None:
    """A field value as text, or None for numbers and numeric arrays (geometry carries no names)."""
    if value is None or isinstance(value, (bool, int, float)):
        return None
    kind = type(value).__name__
    if kind.endswith("Array") and kind not in _TEXT_ARRAYS:
        return None
    if kind.startswith(("Vec", "Matrix", "Quat", "Range", "Interval")):
        return None
    try:
        return str(value).encode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a value that cannot be shown is shown by its type
        return kind.encode()


def _spec_text(layer: object) -> Iterator[tuple[bytes, str]]:
    """Every name, metadata value and non-numeric field of a layer, with the spec path it came from."""
    from pxr import Sdf

    def fields(spec: object) -> Iterator[bytes]:
        for key in spec.ListInfoKeys():
            yield key.encode()
            try:
                value = spec.GetInfo(key)
            except Exception:  # noqa: BLE001 - a field with no Python form (subLayerOffsets) is read below
                continue
            text = _textual(value)
            if text:
                yield text
    root = layer.pseudoRoot
    offsets = [str(offset).encode() for offset in layer.subLayerOffsets]
    yield b"\n".join([*fields(root), *(path.encode() for path in layer.subLayerPaths), *offsets]), "layer metadata"
    paths: list = []
    layer.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    for path in paths:
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            continue
        chunk = [path.pathString.encode(), *fields(spec)]
        if isinstance(spec, Sdf.AttributeSpec):
            for time in layer.ListTimeSamplesForPath(path):
                text = _textual(layer.QueryTimeSample(path, time))
                if text:
                    chunk.append(text)
        yield b"\n".join(chunk), path.pathString


def _usd_text_finding(path: Path, literals: Sequence[tuple[bytes, str]],
                      scratch: Path | None = None) -> tuple[str, str] | None:
    """Open one USD layer with ``pxr`` and read its text: exported whole, or walked spec by spec when large."""
    try:
        from pxr import Sdf
        layer = Sdf.Layer.OpenAsAnonymous(str(path))
    except Exception:  # noqa: BLE001 - any failure to open is a refusal, never a skip
        layer = None
    if layer is None:
        return UNREADABLE_USD, "the file"
    try:
        large = Path(path).stat().st_size > SPEC_WALK_BYTES
    except OSError:
        return UNREADABLE_USD, "the file"
    if large:
        try:
            for text, where in _spec_text(layer):
                found = _match_window(text, literals, text=True)
                if found:
                    return found[0], "a field of its text form"
        except Exception:  # noqa: BLE001 - a layer that cannot be walked is a refusal
            return UNREADABLE_USD, "its text form"
        return None
    with tempfile.TemporaryDirectory(prefix="outbound-scan-", dir=scratch) as folder:
        text = Path(folder) / "layer.usda"
        try:
            if not layer.Export(str(text)):
                return UNREADABLE_USD, "its text form"
        except Exception:  # noqa: BLE001 - as above
            return UNREADABLE_USD, "its text form"
        with text.open("rb") as stream:
            found = scan_stream(stream, literals, text=True)
        if found is None:
            return None
        with text.open("rb") as stream:
            line = stream.read(found[1]).count(b"\n") + 1
        return found[0], f"line {line} of its text form"


def scan_file(path: Path, relative: str, literals: Sequence[tuple[bytes, str]],
              scratch: Path | None = None) -> Finding | None:
    """The first finding in one file (its bytes, zip members, image metadata, USD text form), or None."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        with path.open("rb") as stream:
            head = stream.read(5)
            stream.seek(0)
            found = scan_stream(stream, literals, text=suffix in TEXT_SUFFIXES or head == b"#usda")
        if found:
            return Finding(relative, found[0], f"byte {found[1]}")
        if suffix in _IMAGE_SUFFIXES and path.stat().st_size <= _IMAGE_METADATA_LIMIT:
            for label, value in _image_text(path.read_bytes(), suffix):
                hit = _match_window(value, literals, text=True)
                if hit:
                    return Finding(relative, hit[0], f"image metadata ({label})")
    except OSError:
        return Finding(relative, UNREADABLE_FILE, "the file")
    if suffix == ".usdz" or (suffix != ".usd" and zipfile.is_zipfile(path)):
        return _scan_zip(path, relative, literals, usdz=suffix == ".usdz", scratch=scratch)
    if suffix in USD_SUFFIXES:
        found_text = _usd_text_finding(path, literals, scratch)
        return Finding(relative, *found_text) if found_text else None
    return None


def _scan_zip(path: Path, relative: str, literals: Sequence[tuple[bytes, str]], *, usdz: bool,
              scratch: Path | None = None) -> Finding | None:
    try:
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix="outbound-scan-",
                                                                           dir=scratch) as folder:
            for index, info in enumerate(archive.infolist()):
                if info.is_dir():
                    continue
                member_suffix = Path(info.filename).suffix.lower()
                with archive.open(info) as member:
                    head = member.read(5)
                with archive.open(info) as member:
                    found = scan_stream(member, literals,
                                        text=member_suffix in TEXT_SUFFIXES or head == b"#usda")
                if found:
                    return Finding(relative, found[0] + " inside a zip member", f"member {index + 1}")
                if member_suffix in _IMAGE_SUFFIXES and info.file_size <= _IMAGE_METADATA_LIMIT:
                    for label, value in _image_text(archive.read(info), member_suffix):
                        hit = _match_window(value, literals, text=True)
                        if hit:
                            return Finding(relative, hit[0] + " inside a zip member",
                                           f"member {index + 1}, image metadata ({label})")
                if member_suffix in USD_SUFFIXES - {".usdz"}:
                    extracted = Path(folder) / f"member-{index}{member_suffix}"
                    with archive.open(info) as source, extracted.open("wb") as target:
                        while chunk := source.read(CHUNK_BYTES):
                            target.write(chunk)
                    found_text = _usd_text_finding(extracted, literals, folder)
                    extracted.unlink()
                    if found_text:
                        kind = found_text[0] if found_text[0] in UNREADABLE else found_text[0] + " inside a zip member"
                        return Finding(relative, kind, f"member {index + 1}, {found_text[1]}")
            if usdz and not any(Path(info.filename).suffix.lower() in USD_SUFFIXES for info in archive.infolist()):
                return Finding(relative, UNREADABLE_USD, "no USD layer inside")
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, zlib.error):
        return Finding(relative, UNREADABLE_USD if usdz else UNREADABLE_ZIP, "the file")
    return None


def _needles(relative: str, literals: Sequence[tuple[bytes, str]], words_only_under: str | None,
             words_except: Sequence[str] = ()):
    """The needles for one file: the site's identity only where it could come from the site's scene or job,
    never in a file Blueprint writes with fixed text (``words_except``)."""
    if relative not in words_except and (words_only_under is None or relative.startswith(words_only_under)):
        return literals
    return [row for row in literals if row[1] != SITE_WORDS]


def scan_tree(root: Path, literals: Sequence[tuple[bytes, str]], *, scratch: Path | None = None,
              words_only_under: str | None = None, words_except: Sequence[str] = (),
              before_each: Callable[[], None] | None = None) -> list[Finding]:
    """Every file under ``root`` in name order, at most one finding per file.

    ``words_only_under``: look for the site's identity only in files under this prefix. ``before_each`` is
    called before each file (a caller's deadline check, which may raise).
    """
    root = Path(root)
    findings = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if before_each is not None:
            before_each()
        relative = path.relative_to(root).as_posix()
        finding = scan_file(path, relative, _needles(relative, literals, words_only_under, words_except), scratch)
        if finding:
            findings.append(finding)
    return findings


def first_finding(root: Path, literals: Sequence[tuple[bytes, str]], *, scratch: Path | None = None,
                  words_only_under: str | None = None, words_except: Sequence[str] = ()) -> Finding | None:
    """The first finding under ``root`` in name order, or None; stops at the first."""
    root = Path(root)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        finding = scan_file(path, relative, _needles(relative, literals, words_only_under, words_except), scratch)
        if finding:
            return finding
    return None


__all__ = ["ALLOWED_LINK_PREFIXES", "Finding", "SITE_WORDS", "UNREADABLE", "UNREADABLE_USD", "UNREADABLE_ZIP",
           "capture_site_identity", "capture_site_words", "first_finding", "record_site_identity",
           "record_site_words", "site_identity", "literal_forms", "scan_file", "scan_stream", "scan_tree", "site_words"]
