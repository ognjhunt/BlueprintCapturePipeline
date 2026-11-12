"""One scan for every file that leaves Blueprint: ids, keys, host paths, links and the site's identity.

The job packet tool (``job_packet_export``) and the simulation package for deployment companies
(``website_company_simulation``) both hand over simulation files that Blueprint WROTE, never copied: every USD layer,
usdz, PNG and JPEG is written by ``outbound_file_writer`` from what was read (what the reader does not read cannot
be in the output). This scan then runs over the written output, as the proof, and refuses, naming the file, when
one holds something that says who or where the site is, where Blueprint keeps things, or how to reach it. A finding
names the file, the kind and where; it never repeats what matched.

The one limit: numeric payload (pixel data, an image's coding tables, palette and transparency tables, numeric
arrays in a layer) is written as read
and is not searched as text. These files come from Blueprint's own tools and from asset tools, never from a person
typing; the scan guards against what tools write by accident (metadata, comments, names, paths, links), not against
text deliberately encoded as numbers.

Rule 1: a closed list of file kinds, each with one stated way its text is obtained. A file's kind is decided by
its content (its magic bytes, or valid UTF-8 with no NUL byte for text), never by its ending.

| Kind | How its text is obtained | Pattern classes applied |
|---|---|---|
| USD layer (``.usd``, ``.usda``, ``.usdc``; a usdz member) | ``pxr``: the text export for a layer up to ``SPEC_WALK_BYTES``, the spec walk above (each walked value whole, newlines included); a ``.usda`` file is also text itself | parsed text: all classes; raw bytes of a binary layer: the raw set |
| USDZ archive | each member by its own kind (a USD layer, a PNG or a JPEG; anything else may not ship); each member's name as text; an archive comment may not ship | as each member's kind; names: all classes |
| PNG, JPEG | their metadata chunks (tEXt, zTXt, iTXt, eXIf; APPn and COM), decompressed; the written files hold none | metadata: all classes; raw bytes: the raw set |
| Text (by content): Blueprint's own README, ``success_rule.json``, ``scene.usda``, the packet's sheet and README | the file itself | all classes |
| PLY (the packet's reconstruction splat, ascii or binary) | its header, up to and including ``end_header`` within 1 MiB, read as text (a header that does not end refuses); the body is not rewritten | header: all classes; body: the raw set |
| GLB (the packet's collision mesh) | its JSON chunk, read as text | JSON: all classes; body: the raw set |
| MP4 (the packet's video, written by Blueprint's own ffmpeg step without sound and checked with ffprobe by its own code; the trial clips, Blueprint's own review renders, copied) | none here | raw bytes: the raw set |
| Any other kind | none | refused where only the closed list may ship (``closed``: the package, the packet's simulation folders); elsewhere raw bytes: the raw set |

Rule 2: each pattern class has one place it is applied. The raw set (raw bytes of a binary file) holds only
patterns whose chance of matching random bytes is about 1e-5 per GiB or less (the arithmetic is beside
``_RAW_LINKS_AND_PATHS``).

| Pattern class | In parsed text | In raw bytes of a binary file (the raw set) |
|---|---|---|
| The caller's ids, account ids and keys (hyphen and underscore forms) | yes, any length of 4 or more | only those of ``RAW_LITERAL_MINIMUM`` (16) characters or more |
| Token formats with long fixed prefixes (``AKIA``/``ASIA`` key ids, ``github_pat_``, ``glpat-``, private-key blocks, JWTs, ``aws_secret_access_key=``) | yes | yes (``_RAW_TOKENS``; below 1e-9 each in 512 MiB) |
| Links with three or more host characters: ``gs://``, ``s3://``, ``http(s)://``, ``file://``, any lower-case scheme of four letters or more | yes (any scheme, any case) | yes (``_RAW_LINKS_AND_PATHS``; a fixed scheme whatever byte precedes it, about 1.7e-5 per GiB for ``gs``/``s3``, far less for ``http(s)``/``file``; any other scheme after a non-letter, about 1e-5 per GiB) |
| Long host paths: ``/var/lib/blueprint``, ``/mnt/blueprint-work``, ``/home/<3+>/``, ``/Users/<3+>/`` | yes | yes (below 1e-18 a position) |
| Short host paths (``/tmp/``, ``/opt/``, ``/srv/``, ``/data/``, ``/root/``, ``~/``, drive and UNC paths), other tokens (``sk-``, ``gh*_``, ``xox*-``, ``AIza``, ``hf_``, bearer), signed-link parameters | yes | no |
| Email and IPv4 addresses, bare web addresses | yes | no |
| The site's identity (``site_identity``) | yes, whole words, accents folded | never |

The site's identity, looked for in parsed text only, with accents folded (NFKD, combining marks dropped) and case
ignored on both sides:

- each value as whole words (no letter or digit either side);
- a value of two or more words also in its normalised form (lower-case letters and digits, separators removed),
  matched only where it starts and ends on a word boundary of the text, so ``Zebra Plant Foods`` is found as
  ``ZebraPlantFoods``, ``zebra_plant_foods`` or ``zebra-plant-foods``, and ``Unit 1`` could not match inside
  ``metersPerUnit = 1``; a value of digits only (a phone) needs 11 digits in that form;
- a single-word value is matched only when it has five or more characters and is not one of ``COMMON_WORDS``
  (words scenes and addresses use: World, Floor, Wall, Station, North, ...); shorter single words such as
  ``Bob`` or ``York`` are not matched;
- an address is matched whole, and each line or comma-separated part that has a run of four or more letters, is
  not only a country, and is not a unit or a number (it does not start with Unit, Suite, Flat, Apt, Apartment,
  Floor, Level, Building, Bldg, Block, Room, Office, No, Number, Box or PO Box followed by at most one short
  word); a part that is a single word follows the single-word rule.

Binary USD is read through ``pxr`` (a layer that cannot be opened is a finding, never skipped).
"""
from __future__ import annotations

import codecs
import re
import struct
import tempfile
import unicodedata
import zipfile
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

#: Links a file may carry: XML namespace identifiers (an SVG, XML or USD header names them).
ALLOWED_LINK_PREFIXES = (b"http://www.w3.org/",)
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
#: What a scene may ship (Rule 1). Blueprint's own text files are recognised by their content.
SHIPPED_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz", ".png", ".jpg", ".jpeg"})
CHUNK_BYTES = 8 << 20
UNREADABLE_ZIP = "a zip that cannot be read"
UNREADABLE_USD = "a USD file that cannot be opened"
UNREADABLE_FILE = "a file that cannot be read"
NOT_SHIPPED = "a kind of file that may not ship"
UNREADABLE_HEADER = "a header that cannot be read"
UNREADABLE = frozenset({UNREADABLE_ZIP, UNREADABLE_USD, UNREADABLE_FILE, NOT_SHIPPED, UNREADABLE_HEADER})

STORAGE_LINK, HOST_PATH, FILE_LINK, WEB_LINK, WEB_ADDRESS, KEY = (
    "a storage link", "a host path", "a local file link", "a web link", "a web address", "a key or token")
OTHER_LINK, IP_ADDRESS, EMAIL = "a link", "an IP address", "an email address"
#: The site's identity: its name, organisation, typed address, contact and submitter names, email and phone. The
#: site's description of its job is shared by its own agreement and is not looked for.
SITE_IDENTITY = "the site's name or contact details"
SITE_WORDS = SITE_IDENTITY  # the earlier name, kept for callers
#: A caller's value is looked for in raw binary bytes only from this length (a random match is then impossible).
RAW_LITERAL_MINIMUM = 16
_WORD, _NORMALISED = b"\x00w:", b"\x00n:"
#: A binary USD layer larger than this is read by walking its specs (every name, string, token, asset path and
#: metadata value; numeric arrays skipped) instead of exporting all its geometry to text.
SPEC_WALK_BYTES = 8 << 20

#: Raw bytes of a binary file, besides long-prefix tokens: links and long host paths, each with a chance in random
#: bytes of about 1e-5 per GiB or less (a GiB is about 1.07e9 positions):
#: - ``gs://``, ``s3://`` + 3 host characters, whatever byte comes before: (1/256)^5 x (66/256)^3, about 1.6e-14
#:   a position, 1.7e-5 per GiB each (lower-case schemes only; upper case is found in parsed text). No look-behind:
#:   a fixed scheme is found after a letter, a digit, a space, a NUL or a high byte alike, and dropping it does not
#:   raise the rate, since the five fixed bytes alone set it;
#: - ``http://``, ``https://``, ``file://`` + 3: (1/256)^7 or less x 0.017, below 1e-18 a position;
#: - any other scheme of 4 to 15 lower-case letters + ``://`` + 3: (26/256)^4 x (1/256)^3 x 0.017 x 0.9, about
#:   1e-14 a position, about 1e-5 per GiB. This one keeps its look-behind (no letter before it): it anchors the
#:   match at the start of a run of letters, so one run is counted once, not once for each of its suffixes;
#: - ``/var/lib/blueprint``, ``/mnt/blueprint-work``: below 1e-30 a position;
#: - ``/home/`` or ``/Users/`` + 3 name characters + ``/``: (1/256)^7 x (40/256)^3, below 1e-18 a position.
#: Short paths (``/tmp/``, ``~/``, ``/opt/``, ``/data/``), emails, IPv4 and bare domains stay in parsed text only.
_RAW_LINKS_AND_PATHS = re.compile(
    rb"(?:gs|s3|https?|file)://[A-Za-z0-9._~%/-]{3}"
    rb"|(?<![A-Za-z])[a-z][a-z0-9+.-]{3,15}://[A-Za-z0-9._~%-]{3}"
    rb"|/var/+lib/+blueprint|/mnt/+blueprint-work|/home/[A-Za-z0-9._-]{3,64}/|/Users/[A-Za-z0-9._ -]{3,64}/")
_RAW_LINK_KINDS = ((b"gs:", STORAGE_LINK), (b"s3:", STORAGE_LINK), (b"http", WEB_LINK), (b"file:", FILE_LINK),
                   (b"/", HOST_PATH))
#: Raw bytes of a binary file: token formats with long fixed prefixes.
_RAW_TOKENS = re.compile(
    rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])|(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{30,}"
    rb"|(?<![A-Za-z0-9])glpat-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    rb"|(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.|(?i:aws_secret_access_key\s*[=:])")
#: Parsed text: every class, in this order.
_TEXT_PATTERNS: tuple[tuple[re.Pattern[bytes], str], ...] = (
    (re.compile(rb"(?i)\b(?:gs|s3)://"), STORAGE_LINK),
    (re.compile(rb"/var/+lib/+blueprint|/mnt/+blueprint-work"
                rb"|(?<![A-Za-z0-9_.])/(?:home|root|Users|data|tmp|opt|srv)/|(?<![A-Za-z0-9_.])~/"), HOST_PATH),
    (re.compile(rb"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'@<>|:*?\\/\x00-\x1f]{1,64}[\\/]"), HOST_PATH),
    (re.compile(rb"(?<![A-Za-z0-9_\\])\\\\[A-Za-z0-9][A-Za-z0-9._$-]{0,62}\\[A-Za-z0-9._$ -]{1,80}"), HOST_PATH),
    (re.compile(rb"(?i)\bfile:/"), FILE_LINK),
    (_RAW_TOKENS, KEY),
    (re.compile(
        rb"(?:(?<![A-Za-z0-9])sk-(?:proj-|live-|svcacct-|ant-)?[A-Za-z0-9_-]{20,}"
        rb"|(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}"
        rb"|(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}|(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}"
        rb"|(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}|(?<![A-Za-z0-9])(?:ntn|secret)_[A-Za-z0-9]{40,}"
        rb"|(?i:(?<![A-Za-z0-9])bearer\s+[A-Za-z0-9._~+/-]{20,})"
        rb"|(?i:[?&](?:access_token|refresh_token|id_token|token|api_key|apikey|key|secret|password|sig"
        rb"|signature|x-amz-signature|x-goog-signature|x-amz-credential|x-goog-credential)=[^\s&\"'<>]{8,}))"), KEY),
    (re.compile(rb"(?i)https?://"), WEB_LINK),
    (re.compile(rb"(?i)(?<![A-Za-z0-9+.-])[a-z][a-z0-9+.-]{1,15}://"), OTHER_LINK),
    (re.compile(rb"(?<![A-Za-z0-9._%+-])[A-Za-z0-9][A-Za-z0-9._%+-]{0,63}@[A-Za-z0-9-]{1,63}"
                rb"(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}(?![A-Za-z0-9-])"), EMAIL),
)
_IP_ADDRESS = re.compile(rb"(?<![0-9A-Za-z.])(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])\.){3}"
                         rb"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])(?![0-9A-Za-z]|\.[0-9])")
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


def fold(text: str) -> str:
    """Lower-case with accents folded: NFKD, combining marks dropped."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).lower()


_TOKEN = re.compile(r"[0-9a-z]+")
#: Words scenes and addresses use; a single-word identity value that is one of these is not matched.
COMMON_WORDS = frozenset({
    "world", "scene", "stage", "root", "asset", "object", "objects", "point", "points", "light", "lights", "floor",
    "floors", "wall", "walls", "tile", "tiles", "mesh", "meshes", "room", "rooms", "station", "north", "south",
    "east", "west", "table", "shelf", "shelves", "door", "doors", "window", "ground", "plane", "camera", "cube",
    "material", "materials", "looks", "geometry", "collision", "appearance", "support", "render", "model", "base",
    "center", "centre", "kitchen", "office", "store", "shop", "warehouse", "factory", "plant", "house", "building",
    "street", "road", "lane", "avenue", "court", "place", "square", "park", "estate", "industrial", "unit", "units",
    "suite", "level", "block", "upper", "lower", "inner", "outer", "front", "back", "left", "right", "middle",
    "volume", "field", "gaussians", "capture", "captured", "particle", "transform", "default", "xform", "scope"})
_UNIT_PART = re.compile(r"(?:unit|suite|flat|apt|apartment|floor|level|building|bldg|block|room|office|no|number"
                        r"|box|po box)\.?\s*[0-9a-z-]{0,6}")
#: An address part that is only a country says nothing about which site it is.
_COUNTRIES = frozenset({"uk", "u.k.", "united kingdom", "england", "scotland", "wales", "northern ireland", "usa",
                        "u.s.a.", "us", "united states", "united states of america", "america", "canada", "ireland",
                        "france", "germany", "deutschland", "spain", "italy", "netherlands", "belgium", "australia",
                        "new zealand", "india", "japan", "china", "mexico", "brazil"})


def site_identity(values: Iterable[str], *, addresses: Iterable[str] = ()) -> list[tuple[bytes, str]]:
    """The site's identity as needles (see the module's rules): whole words, and for values of two or more words
    also the normalised form; single words of five or more characters that are not common words; address parts
    that are not a unit, a number or a country."""
    found: dict[bytes, None] = {}

    def add(text: str) -> None:
        folded = " ".join(fold(str(text)).split())
        words = _TOKEN.findall(folded)
        if not words:
            return
        if "@" in folded:  # an email: as written
            found[_WORD + folded.encode()] = None
            return
        if len(words) == 1:
            word = words[0]
            if len(word) >= 5 and word not in COMMON_WORDS and not word.isdigit():
                found[_WORD + folded.encode()] = None
            return
        found[_WORD + folded.encode()] = None
        normal = "".join(words)
        if (normal.isdigit() and len(normal) >= 11) or (not normal.isdigit() and len(normal) >= 5):
            found[_NORMALISED + normal.encode()] = None
    for value in values:
        add(value)
    for address in addresses:
        add(address)
        for part in re.split(r"[,\n;]+", str(address)):
            folded = " ".join(fold(part).split())
            if (re.search(r"[a-z]{4}", folded) and folded not in _COUNTRIES
                    and not _UNIT_PART.fullmatch(folded)):
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


def _whole_word(haystack: str, needle: str) -> int:
    """The first place ``needle`` stands as whole words in ``haystack`` (no letter or digit either side), or -1."""
    start = haystack.find(needle)
    while start >= 0:
        end = start + len(needle)
        if not (haystack[start - 1:start].isalnum() or haystack[end:end + 1].isalnum()):
            return start
        start = haystack.find(needle, start + 1)
    return -1


def _normalised_at_boundaries(tokens: list[str], needle: str) -> bool:
    """``needle`` equals the joined text of consecutive whole tokens (it starts and ends on token boundaries)."""
    for index, first in enumerate(tokens):
        if not needle.startswith(first):
            continue
        joined = first
        for token in tokens[index + 1:]:
            if len(joined) >= len(needle):
                break
            joined += token
        if joined == needle:
            return True
    return False


def _identity_found(window: bytes, needles: Sequence[tuple[bytes, str]]) -> str | None:
    if not needles:
        return None
    text = fold(window.decode("utf-8", "replace"))
    tokens: list[str] | None = None
    for value, kind in needles:
        needle = value[3:].decode("utf-8", "replace")
        if value.startswith(_WORD):
            if _whole_word(text, needle) >= 0:
                return kind
        else:
            tokens = _TOKEN.findall(text) if tokens is None else tokens
            if _normalised_at_boundaries(tokens, needle):
                return kind
    return None


def _match_window(window: bytes, literals: Sequence[tuple[bytes, str]], *, text: bool) -> tuple[str, int] | None:
    """The kind and offset of the first finding in one window. ``text``: parsed text, every class (Rule 2);
    otherwise raw bytes of a binary file: long caller values and long-prefix tokens only."""
    identity = [row for row in literals if row[0].startswith((_WORD, _NORMALISED))]
    for value, kind in literals:
        if value.startswith((_WORD, _NORMALISED)) or (not text and len(value) < RAW_LITERAL_MINIMUM):
            continue
        position = window.find(value)
        if position >= 0:
            return kind, position
    if not text:
        match = _RAW_TOKENS.search(window)
        if match:
            return KEY, match.start()
        match = _RAW_LINKS_AND_PATHS.search(_blank_allowed_links(window))
        if match:
            value = match.group(0)
            kind = next((name for prefix, name in _RAW_LINK_KINDS if value.startswith(prefix)), OTHER_LINK)
            return kind, match.start()
        return None
    found = _identity_found(window, identity)
    if found:
        return found, 0
    cleaned = _blank_allowed_links(window)
    for pattern, kind in (*_TEXT_PATTERNS, (_WEB_ADDRESS, WEB_ADDRESS), (_IP_ADDRESS, IP_ADDRESS)):
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
    if isinstance(value, str):
        return value.encode("utf-8", "replace")
    if isinstance(value, Mapping) or hasattr(value, "items"):
        # A dictionary (customData, assetInfo): its keys and values as they are, so a newline inside a value is
        # read as a newline, not as the two characters a printed form would show.
        try:
            parts = [part for key, item in value.items()
                     for part in (str(key).encode("utf-8", "replace"), _textual(item)) if part]
            return b"\n".join(parts)
        except Exception:  # noqa: BLE001 - fall back to its printed form
            pass
    if isinstance(value, (list, tuple)) or kind in _TEXT_ARRAYS:
        try:
            return b"\n".join(part for item in value if (part := _textual(item)))
        except Exception:  # noqa: BLE001 - fall back to its printed form
            pass
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


def content_kind(path: Path) -> str:
    """``usdc``, ``usda``, ``usdz``, ``png``, ``jpeg``, ``text`` or ``other``, by content (Rule 1)."""
    with Path(path).open("rb") as stream:
        head = stream.read(8)
        if head.startswith(b"PXR-USDC"):
            return "usdc"
        if head.startswith(b"#usda"):
            return "usda"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if head.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if head.startswith(b"PK\x03\x04") and Path(path).suffix.lower() == ".usdz":
            return "usdz"
        if head.startswith((b"ply\n", b"ply\r\n")):
            return "ply"
        if head.startswith(b"glTF"):
            return "glb"
        stream.seek(0)
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                if b"\x00" in chunk:
                    return "other"
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            return "other"
    return "zip" if zipfile.is_zipfile(path) else "text"


PLY_HEADER_LIMIT = 1 << 20


def _header_text(path: Path, kind: str) -> bytes | None:
    """A PLY's header (up to and including ``end_header``, within 1 MiB) or a GLB's JSON chunk, as text; None when
    it cannot be read."""
    with path.open("rb") as stream:
        if kind == "ply":
            head = stream.read(PLY_HEADER_LIMIT)
            end = head.find(b"end_header")
            return head[:end + len(b"end_header")] if end >= 0 else None
        head = stream.read(20)
        if len(head) < 20:
            return None
        length, chunk = struct.unpack("<I4s", head[12:20])
        if chunk != b"JSON" or length > 64 << 20:
            return None
        body = stream.read(length)
        return body if len(body) == length else None


def _scan_kind(path: Path, relative: str, kind: str, literals: Sequence[tuple[bytes, str]], *, closed: bool,
               scratch: Path | None, member: str = "") -> Finding | None:
    suffix = " inside a zip member" if member else ""
    where = f"{member}, " if member else ""
    if kind == "other" and closed:
        return Finding(relative, NOT_SHIPPED, f"{where}the file")
    if kind in ("ply", "glb"):
        header = _header_text(Path(path), kind)
        if header is None:
            return Finding(relative, UNREADABLE_HEADER, f"{where}the {kind.upper()} header")
        hit = _match_window(header, literals, text=True)
        if hit:
            return Finding(relative, hit[0] + suffix, f"{where}its {kind.upper()} header")
    try:
        with Path(path).open("rb") as stream:
            found = scan_stream(stream, literals, text=kind in ("text", "usda"))
        if found:
            return Finding(relative, found[0] + suffix, f"{where}byte {found[1]}")
        if kind in ("png", "jpeg") and Path(path).stat().st_size <= _IMAGE_METADATA_LIMIT:
            for label, value in _image_text(Path(path).read_bytes(), ".png" if kind == "png" else ".jpg"):
                hit = _match_window(value, literals, text=True)
                if hit:
                    return Finding(relative, hit[0] + suffix, f"{where}image metadata ({label})")
    except OSError:
        return Finding(relative, UNREADABLE_FILE, "the file")
    if kind in ("usda", "usdc"):
        found_text = _usd_text_finding(path, literals, scratch)
        if found_text:
            name = found_text[0] if found_text[0] in UNREADABLE else found_text[0] + suffix
            return Finding(relative, name, f"{where}{found_text[1]}")
    return None


def scan_file(path: Path, relative: str, literals: Sequence[tuple[bytes, str]],
              scratch: Path | None = None, *, closed: bool = False) -> Finding | None:
    """The first finding in one file, read by its kind (Rule 1), or None. ``closed``: a kind not on the list is
    itself a finding."""
    path = Path(path)
    try:
        kind = content_kind(path)
    except OSError:
        return Finding(relative, UNREADABLE_FILE, "the file")
    if path.suffix.lower() == ".usdz" and kind not in ("usdz",):
        return Finding(relative, UNREADABLE_USD, "the file")
    if path.suffix.lower() in (".usd", ".usdc", ".usda") and kind not in ("usdc", "usda"):
        return Finding(relative, UNREADABLE_USD, "the file")
    if kind in ("usdz", "zip"):
        return _scan_zip(path, relative, literals, usdz=kind == "usdz", scratch=scratch, closed=closed)
    return _scan_kind(path, relative, kind, literals, closed=closed, scratch=scratch)


def _scan_zip(path: Path, relative: str, literals: Sequence[tuple[bytes, str]], *, usdz: bool,
              scratch: Path | None = None, closed: bool = False) -> Finding | None:
    """Each member by its own kind (a usdz's members are its layers, textures and any text)."""
    try:
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix="outbound-scan-",
                                                                           dir=scratch) as folder:
            layers = 0
            if usdz and archive.comment:
                return Finding(relative, NOT_SHIPPED, "the archive comment")
            for index, info in enumerate(archive.infolist()):
                if info.is_dir():
                    continue
                if usdz:
                    from .outbound_file_writer import member_name_ok
                    if not member_name_ok(info.filename):
                        return Finding(relative, NOT_SHIPPED, f"member {index + 1}, its name")
                    hit = _match_window(info.filename.encode("utf-8", "replace"), literals, text=True)
                    if hit:
                        return Finding(relative, hit[0] + " inside a zip member", f"member {index + 1}, its name")
                extracted = Path(folder) / f"member-{index}{Path(info.filename).suffix.lower()}"
                with archive.open(info) as source, extracted.open("wb") as target:
                    while chunk := source.read(CHUNK_BYTES):
                        target.write(chunk)
                kind = content_kind(extracted)
                layers += kind in ("usda", "usdc")
                if kind in ("usdz", "zip") or (usdz and kind not in ("usda", "usdc", "png", "jpeg")):
                    kind = "other"  # an archive inside an archive is not read; a usdz holds layers and images only
                found = _scan_kind(extracted, relative, kind, literals, closed=closed or usdz, scratch=folder,
                                   member=f"member {index + 1}")
                extracted.unlink()
                if found:
                    return found
            if usdz and not layers:
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
    return [row for row in literals if row[1] != SITE_IDENTITY]


def _closed(relative: str, closed: bool | Sequence[str]) -> bool:
    return closed if isinstance(closed, bool) else relative.startswith(tuple(closed))


def scan_tree(root: Path, literals: Sequence[tuple[bytes, str]], *, scratch: Path | None = None,
              words_only_under: str | None = None, words_except: Sequence[str] = (),
              before_each: Callable[[], None] | None = None, closed: bool | Sequence[str] = False) -> list[Finding]:
    """Every file under ``root`` in name order, at most one finding per file.

    ``words_only_under``: look for the site's identity only in files under this prefix. ``before_each`` is
    called before each file (a caller's deadline check, which may raise). ``closed``: True, or the prefixes under
    which, only the closed list of kinds may ship.
    """
    root = Path(root)
    findings = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if before_each is not None:
            before_each()
        relative = path.relative_to(root).as_posix()
        finding = scan_file(path, relative, _needles(relative, literals, words_only_under, words_except), scratch,
                            closed=_closed(relative, closed))
        if finding:
            findings.append(finding)
    return findings


def first_finding(root: Path, literals: Sequence[tuple[bytes, str]], *, scratch: Path | None = None,
                  words_only_under: str | None = None, words_except: Sequence[str] = (),
                  closed: bool | Sequence[str] = False) -> Finding | None:
    """The first finding under ``root`` in name order, or None; stops at the first."""
    root = Path(root)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        finding = scan_file(path, relative, _needles(relative, literals, words_only_under, words_except), scratch,
                            closed=_closed(relative, closed))
        if finding:
            return finding
    return None


__all__ = ["ALLOWED_LINK_PREFIXES", "COMMON_WORDS", "Finding", "NOT_SHIPPED", "RAW_LITERAL_MINIMUM",
           "SHIPPED_SUFFIXES", "SITE_IDENTITY", "SITE_WORDS", "UNREADABLE", "UNREADABLE_USD", "UNREADABLE_ZIP",
           "capture_site_identity", "capture_site_words", "content_kind", "first_finding", "fold", "literal_forms",
           "record_site_identity", "record_site_words", "scan_file", "scan_stream", "scan_tree", "site_identity",
           "site_words"]
