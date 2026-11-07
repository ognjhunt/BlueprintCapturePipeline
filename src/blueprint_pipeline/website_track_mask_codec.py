"""Run-length track masks: decoded without the segmentation runtime, held compactly, written exactly.

Kept apart from ``website_task_masks`` so CPU-only readers (preparation status,
the site's object question) can decode a mask without importing the SAM and
GPU provider modules that the mask producer needs.

A track observation's ``runs`` is the row-major list of ``{"start", "length"}``
pixel runs over the whole frame. A large surface has about one run per image
row, so the tracks of a long 4K video reach millions of runs: as Python dicts
that is hundreds of bytes a run, and ``json.dumps(..., indent=2)`` holds a list
of several string pieces a run before joining them. ``RunLengths`` holds the
runs as one integer array that reads, iterates and compares like the list of
dicts, and the writers here produce the bytes ``common.write_json`` and
``canonical_digest`` produce for the list form, chunk by chunk, without it.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json_chunks

_RUN_KEYS = ("start", "length")
_INT32 = np.iinfo(np.int32)
#: Runs rendered per text chunk: a few MB of JSON at most.
_RENDER_BATCH = 1 << 15


def _compact_dtype(*arrays: np.ndarray) -> type:
    if all(not len(array) or (_INT32.min <= int(array.min()) and int(array.max()) <= _INT32.max)
           for array in arrays):
        return np.int32
    return np.int64


class RunLengths(Sequence):
    """An immutable list of ``{"start": int, "length": int}`` runs held as one (n, 2) integer array.

    Indexing and iteration give fresh dicts, equality with a list compares as that list of dicts
    would, and ``json``-style encoding (``__json_value__``) is the list itself. It is not a
    ``list``: an encoder that knows neither type refuses it loudly instead of writing ``[]``.
    """

    __slots__ = ("_pairs",)
    __hash__ = None  # type: ignore[assignment]  # unhashable, like the list it stands for

    def __init__(self, starts: Any = (), lengths: Any = ()) -> None:
        starts, lengths = np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64)
        if starts.ndim != 1 or starts.shape != lengths.shape:
            raise ValueError("run_lengths_shape_invalid")
        pairs = np.empty((len(starts), 2), dtype=_compact_dtype(starts, lengths))
        pairs[:, 0], pairs[:, 1] = starts, lengths
        pairs.flags.writeable = False
        self._pairs = pairs

    @classmethod
    def from_runs(cls, runs: Sequence[Any]) -> RunLengths | None:
        """The compact form of ``runs`` when it re-encodes to the same JSON bytes, else None.

        Only dicts with exactly the keys ``start`` then ``length``, both plain ``int``, qualify;
        any other key, order, type or value outside int64 keeps the list as it is.
        """
        if not all(type(run) is dict and len(run) == 2 and tuple(run) == _RUN_KEYS for run in runs):
            return None
        starts = [run["start"] for run in runs]
        lengths = [run["length"] for run in runs]
        if not all(type(value) is int for value in itertools.chain(starts, lengths)):
            return None
        try:
            return cls(starts, lengths)
        except OverflowError:
            return None

    @property
    def starts(self) -> np.ndarray:
        return self._pairs[:, 0]

    @property
    def lengths(self) -> np.ndarray:
        return self._pairs[:, 1]

    @property
    def nbytes(self) -> int:
        return int(self._pairs.nbytes)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [{"start": start, "length": length} for start, length in self._pairs[index].tolist()]
        start, length = self._pairs[index].tolist()
        return {"start": start, "length": length}

    def __iter__(self) -> Iterator[dict[str, int]]:
        for begin in range(0, len(self._pairs), _RENDER_BATCH):
            for start, length in self._pairs[begin:begin + _RENDER_BATCH].tolist():
                yield {"start": start, "length": length}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RunLengths):
            return bool(np.array_equal(self._pairs, other._pairs))
        if isinstance(other, list):
            return len(other) == len(self) and all(item == run for item, run in zip(other, self))
        return NotImplemented

    def __repr__(self) -> str:
        return f"RunLengths({len(self)} runs)"

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.starts.copy(), self.lengths.copy()))

    def __copy__(self) -> RunLengths:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> RunLengths:
        return self

    def to_list(self) -> list[dict[str, int]]:
        return self[:]

    __json_value__ = to_list

    def _render_indented(self, level: int) -> Iterator[str]:
        """``json.dumps(self.to_list(), indent=2)`` as nested ``level`` deep, in pieces."""
        if not len(self):
            yield "[]"
            return
        outer, item, inner = "  " * level, "  " * (level + 1), "  " * (level + 2)
        render = ("{{\n" + inner + '"start": {},\n' + inner + '"length": {}\n' + item + "}}").format
        separator = ",\n" + item
        yield "[\n" + item
        for begin in range(0, len(self._pairs), _RENDER_BATCH):
            text = separator.join(itertools.starmap(render, self._pairs[begin:begin + _RENDER_BATCH].tolist()))
            yield text if not begin else separator + text
        yield "\n" + outer + "]"

    def _render_canonical(self) -> Iterator[str]:
        """``canonical_json`` (sorted keys, compact separators) of ``self.to_list()``, in pieces."""
        if not len(self):
            yield "[]"
            return
        render = '{{"length":{1},"start":{0}}}'.format
        yield "["
        for begin in range(0, len(self._pairs), _RENDER_BATCH):
            text = ",".join(itertools.starmap(render, self._pairs[begin:begin + _RENDER_BATCH].tolist()))
            yield text if not begin else "," + text
        yield "]"


def raster_runs(raster: np.ndarray, *, left: int, top: int, width: int) -> RunLengths:
    """The runs of ``raster`` pasted at (``left``, ``top``) into an all-zero frame ``width`` pixels wide.

    Equal to pasting it into a frame-sized uint8 array and differencing that array flattened as
    int8, with the k-th +1 step paired with the k-th -1 step, but only the raster's own rows are
    ever allocated. A raster narrower than the frame has zeros on both sides of every row; one
    as wide as the frame is contiguous in the flattened frame, so its runs may span rows.
    """
    raster = np.asarray(raster).astype(np.uint8, copy=False)
    rows, columns = raster.shape
    if not raster.size:
        return RunLengths()
    # Each row followed by one zero (none needed when rows meet end to end in the frame), as int8.
    stride = columns if columns == width else columns + 1
    padded = np.zeros((rows, stride), dtype=np.int8)
    padded[:, :columns] = raster
    flat = padded.reshape(-1)
    # edges[i] = flat[i] - flat[i - 1], with zero before the first pixel and after the last.
    edges = np.empty(len(flat) + 1, dtype=np.int8)
    edges[0] = flat[0]
    np.negative(flat[-1:], out=edges[-1:])
    np.subtract(flat[1:], flat[:-1], out=edges[1:-1])
    steps = np.flatnonzero(edges)
    kinds = edges[steps]
    offsets = steps // stride * width + steps % stride
    base = top * width + left
    starts, ends = offsets[kinds == 1] + base, offsets[kinds == -1] + base
    count = min(len(starts), len(ends))
    return RunLengths(starts[:count], ends[:count] - starts[:count])


def mask_runs(mask: np.ndarray) -> RunLengths:
    """The runs of a whole-frame 2-D mask (nonzero pixels of a 0/1 or boolean mask)."""
    return raster_runs(mask, left=0, top=0, width=mask.shape[1])


def decode_track_mask(observation: Mapping[str, Any]) -> np.ndarray:
    height, width = int(observation["height"]), int(observation["width"])
    if height <= 0 or width <= 0 or height * width > 100_000_000:
        raise ValueError("website_track_mask_dimensions_invalid")
    mask = np.zeros(height * width, dtype=bool)
    runs = observation["runs"]
    if isinstance(runs, RunLengths):
        starts, lengths = runs.starts.astype(np.int64), runs.lengths.astype(np.int64)
        if len(starts) and ((starts < 0) | (lengths <= 0) | (starts + lengths > mask.size)).any():
            raise ValueError("website_track_mask_run_invalid")
        for start, stop in zip(starts.tolist(), (starts + lengths).tolist()):
            mask[start:stop] = True
        return mask.reshape(height, width)
    for run in runs:
        start, length = int(run["start"]), int(run["length"])
        if start < 0 or length <= 0 or start + length > mask.size:
            raise ValueError("website_track_mask_run_invalid")
        mask[start:start + length] = True
    return mask.reshape(height, width)


def plain_runs(value: Any) -> Any:
    """``value`` with every ``RunLengths`` written out as its list of dicts (today's in-memory form).

    Containers holding none are returned as they are; the rest are copied, never mutated.
    """
    kind = type(value)
    if kind is RunLengths:
        return value.to_list()
    if kind is dict:
        items = {key: plain_runs(item) for key, item in value.items()}
        return items if any(items[key] is not item for key, item in value.items()) else value
    if kind is list or kind is tuple:
        items = [plain_runs(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        return items if kind is list else tuple(items)
    return value


def _compact_observation(row: dict[str, Any]) -> dict[str, Any]:
    runs = row.get("runs")
    if type(runs) is list and runs:
        compact = RunLengths.from_runs(runs)
        if compact is not None:
            row["runs"] = compact
    return row


def load_track_json(path: Path) -> Any:
    """``json.loads(path.read_text())`` with each observation's runs compacted as soon as it is read.

    The text is read about a MB at a time and each ``"runs"`` list of ``{"start", "length"}`` dicts
    goes straight into its ``RunLengths``: neither the file's text nor one dict per run is ever
    held whole. (A retained 4K ``tracks.json`` is about 200 MB of text; ``read_text`` held it as
    bytes and as a string at once.) Anything this reader does not follow, malformed JSON
    included, is read again by exactly ``json.loads(path.read_text(), ...)``, which then gives
    the same value or raises the same error it always did.
    """
    try:
        with open(path, encoding=io.text_encoding(None)) as handle:  # read_text's own decoding
            return _StreamingTrackReader(handle).document()
    except (OSError, MemoryError):
        raise
    except Exception:  # noqa: BLE001 - the exact reader below decides; nothing is guessed
        pass
    return json.loads(Path(path).read_text(), object_hook=_compact_observation)


_WHITESPACE = re.compile(r"[ \t\n\r]*")
#: ``json``'s C scanner: ASCII digits only; a fraction or exponent makes a float.
_NUMBER = re.compile(r"(-?(?:0|[1-9][0-9]*))(\.[0-9]+)?([eE][-+]?[0-9]+)?")
_NUMBER_CHARACTERS = re.compile(r"[-+0-9.eE]*")
_RUN_TEXT = (r"[ \t\n\r]*\{[ \t\n\r]*\"start\"[ \t\n\r]*:[ \t\n\r]*-?(?:0|[1-9][0-9]*)[ \t\n\r]*,"
             r"[ \t\n\r]*\"length\"[ \t\n\r]*:[ \t\n\r]*-?(?:0|[1-9][0-9]*)[ \t\n\r]*\}[ \t\n\r]*")
#: The inside of a non-empty JSON list of ``{"start": int, "length": int}`` objects, keys in order.
_RUN_LIST = re.compile(_RUN_TEXT + "(?:," + _RUN_TEXT + ")*")
_INTEGER = re.compile(r"-?[0-9]+")
_READ_CHARACTERS = 1 << 20
_CONSTANTS = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


class _StreamingTrackReader:
    """A JSON reader over a text stream with ``json.loads(..., object_hook=_compact_observation)``'s result.

    Strings are read by ``json``'s own ``scanstring``; numbers and literals as its C scanner reads
    them; objects are built key by key (a repeated key keeps its first place and last value) and
    passed through ``_compact_observation``. A ``"runs"`` value that is a list of exactly
    ``{"start": int, "length": int}`` objects becomes ``RunLengths`` without building its dicts,
    which is what ``_compact_observation`` makes of those dicts. Any departure raises, and the
    caller falls back to ``json.loads``.
    """

    __slots__ = ("_handle", "_text", "_at", "_ended")

    def __init__(self, handle: Any) -> None:
        self._handle, self._text, self._at, self._ended = handle, "", 0, False

    def _more(self) -> bool:
        """Read more text after what is unread; False at the end. Read text before ``_at`` is dropped."""
        if self._ended:
            return False
        text = self._handle.read(_READ_CHARACTERS)
        if not text:
            self._ended = True
            return False
        self._text = self._text[self._at:] + text if self._at < len(self._text) else text
        self._at = 0
        return True

    def _next(self) -> str:
        """The next character after whitespace, not consumed; ``""`` at the end."""
        while True:
            self._at = _WHITESPACE.match(self._text, self._at).end()
            if self._at < len(self._text) or not self._more():
                return self._text[self._at:self._at + 1]

    def document(self) -> Any:
        value = self._value()
        if self._next():
            raise ValueError("track_json_extra_data")
        return value

    def _value(self) -> Any:
        character = self._next()
        if character == '"':
            return self._string()
        if character == "{":
            return self._object()
        if character == "[":
            return self._array()
        return self._scalar()

    def _string(self) -> str:
        while True:
            try:
                value, end = json.decoder.scanstring(self._text, self._at + 1, True)
            except json.JSONDecodeError as exc:
                # Only a string the read text cuts short is read again with more text.
                if (exc.msg.startswith("Unterminated string") or exc.pos >= len(self._text) - 6) and self._more():
                    continue
                raise
            self._at = end
            return value

    def _scalar(self) -> Any:
        while (_NUMBER_CHARACTERS.match(self._text, self._at).end() == len(self._text)
               or len(self._text) - self._at < 9) and self._more():
            pass
        text, at = self._text, self._at
        for literal, value in (("null", None), ("true", True), ("false", False)):
            if text.startswith(literal, at):
                self._at = at + len(literal)
                return value
        match = _NUMBER.match(text, at)
        if match is not None:
            self._at = match.end()
            integer, fraction, exponent = match.groups()
            if fraction or exponent:
                return float(integer + (fraction or "") + (exponent or ""))
            return int(integer)
        for literal, value in _CONSTANTS.items():
            if text.startswith(literal, at):
                self._at = at + len(literal)
                return value
        raise ValueError("track_json_value_expected")

    def _object(self) -> Any:
        self._at += 1
        row: dict[str, Any] = {}
        character = self._next()
        if character == "}":
            self._at += 1
            return _compact_observation(row)
        while True:
            if character != '"':
                raise ValueError("track_json_key_expected")
            key = self._string()
            if self._next() != ":":
                raise ValueError("track_json_colon_expected")
            self._at += 1
            value = self._runs() if key == "runs" and self._next() == "[" else None
            row[key] = self._value() if value is None else value
            character = self._next()
            self._at += 1
            if character == "}":
                return _compact_observation(row)
            if character != ",":
                raise ValueError("track_json_comma_expected")
            character = self._next()

    def _array(self) -> list[Any]:
        self._at += 1
        items: list[Any] = []
        if self._next() == "]":
            self._at += 1
            return items
        while True:
            items.append(self._value())
            character = self._next()
            self._at += 1
            if character == "]":
                return items
            if character != ",":
                raise ValueError("track_json_comma_expected")

    def _runs(self) -> RunLengths | None:
        """The list at ``_at`` as ``RunLengths`` when it is a non-empty list of exact runs, else None.

        Run objects hold no bracket and no other string, so such a list ends at the first ``]``.
        Nothing is consumed unless it is one.
        """
        while (end := self._text.find("]", self._at)) < 0:
            if not self._more():
                return None
        if _RUN_LIST.fullmatch(self._text, self._at + 1, end) is None:
            return None
        values = [int(value) for value in _INTEGER.findall(self._text, self._at + 1, end)]
        try:
            runs = RunLengths(values[0::2], values[1::2])
        except OverflowError:  # beyond int64: kept as its list of dicts, as ``from_runs`` keeps it
            return None
        self._at = end + 1
        return runs


def detached_track_json(value: Any) -> Any:
    """``json.loads(json.dumps(value, allow_nan=False))`` with every ``RunLengths`` kept as it is.

    ``RunLengths`` is immutable and equals its list of dicts, so sharing it is a detached copy,
    and the round trip never renders it as text. Exact dicts with ``str`` keys, lists and tuples
    are copied here; ``str``, ``int``, ``bool``, ``None`` and finite ``float`` of exact type are
    the round trip's own result; every other value takes the round trip itself, so it gives
    what it gave and refuses what it refused.
    """
    return _detached(value, set())


def _detached(value: Any, active: set[int]) -> Any:
    kind = type(value)
    if kind is RunLengths or kind is str or kind is bool or value is None:
        return value
    if (kind is int and value.bit_length() < 4096) or (kind is float and math.isfinite(value)):
        return value  # (an int of more than 4300 digits is one ``json.dumps`` refuses)
    if (kind is dict and all(type(key) is str for key in value)) or kind is list or kind is tuple:
        if id(value) in active:
            raise ValueError("Circular reference detected")
        active.add(id(value))
        try:
            if kind is dict:
                return {key: _detached(item, active) for key, item in value.items()}
            return [_detached(item, active) for item in value]
        finally:
            active.discard(id(value))
    return json.loads(json.dumps(plain_runs(value), allow_nan=False))


# ---- exact streaming JSON -------------------------------------------------------------------
#
# Exact ``dict`` (with ``str`` keys), ``list``, ``tuple`` and scalars of exact type are encoded
# here, and ``RunLengths`` from its array. Anything else, and any plain run list (one
# observation's worth), is encoded by the very ``json`` encoder ``write_json`` or
# ``canonical_json`` would use, re-indented to its depth: JSON text has no raw newline inside a
# string, so prefixing every line of a nested value with its depth's indent is exact.

def _json_value(value: Any) -> Any:
    encode = getattr(type(value), "__json_value__", None)
    if encode is None:
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
    return encode(value)


_INDENT_ENCODER = json.JSONEncoder(indent=2, default=_json_value)
_CANONICAL_ENCODER = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                      default=_json_value)
_ascii = json.encoder.encode_basestring_ascii
_unicode = json.encoder.encode_basestring
_INFINITY = float("inf")


def _float_text(value: float) -> str:
    if value != value:
        return "NaN"
    if value == _INFINITY:
        return "Infinity"
    if value == -_INFINITY:
        return "-Infinity"
    return float.__repr__(value)


def _scalar(value: Any, quote: Any) -> str | None:
    kind = type(value)
    if kind is str:
        return quote(value)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if kind is int:
        return int.__repr__(value)
    if kind is float:
        return _float_text(value)
    return None


def _plain_run_list(value: Any) -> bool:
    return (type(value) is list and bool(value) and type(value[0]) is dict
            and tuple(value[0])[:2] == _RUN_KEYS)


def _walked(value: Any) -> bool:
    kind = type(value)
    return ((kind is dict and all(type(key) is str for key in value)) or kind is list or kind is tuple) \
        and not _plain_run_list(value)


def _indent_chunks(value: Any, level: int, active: set[int]) -> Iterator[str]:
    text = _scalar(value, _ascii)
    if text is not None:
        yield text
    elif type(value) is RunLengths:
        yield from value._render_indented(level)
    elif not _walked(value):
        text = _INDENT_ENCODER.encode(value)
        yield text.replace("\n", "\n" + "  " * level) if level else text
    elif not value:
        yield "{}" if type(value) is dict else "[]"
    else:
        if id(value) in active:
            raise ValueError("Circular reference detected")
        active.add(id(value))
        inner = "\n" + "  " * (level + 1)
        if type(value) is dict:
            separator = "{" + inner
            for key, item in value.items():
                yield separator + _ascii(key) + ": "
                separator = "," + inner
                yield from _indent_chunks(item, level + 1, active)
            yield "\n" + "  " * level + "}"
        else:
            separator = "[" + inner
            for item in value:
                yield separator
                separator = "," + inner
                yield from _indent_chunks(item, level + 1, active)
            yield "\n" + "  " * level + "]"
        active.discard(id(value))


def _canonical_chunks(value: Any, active: set[int]) -> Iterator[str]:
    text = _scalar(value, _unicode)
    if text is not None:
        yield text
    elif type(value) is RunLengths:
        yield from value._render_canonical()
    elif not _walked(value):
        yield _CANONICAL_ENCODER.encode(value)
    elif not value:
        yield "{}" if type(value) is dict else "[]"
    else:
        if id(value) in active:
            raise ValueError("Circular reference detected")
        active.add(id(value))
        if type(value) is dict:
            separator = "{"
            for key in sorted(value):
                yield separator + _unicode(key) + ":"
                separator = ","
                yield from _canonical_chunks(value[key], active)
            yield "}"
        else:
            separator = "["
            for item in value:
                yield separator
                separator = ","
                yield from _canonical_chunks(item, active)
            yield "]"
        active.discard(id(value))


def write_track_json(path: Path, payload: Mapping[str, Any]) -> None:
    """``common.write_json(path, payload)``, byte for byte, holding no more than a chunk of text at once."""
    write_json_chunks(path, _indent_chunks(dict(payload), 0, set()))


def canonical_track_digest(value: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    """``canonical_digest(value, digest_field=...)``, hashed chunk by chunk instead of as one string."""
    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    digest = hashlib.sha256()
    pending: list[str] = []
    size = 0
    for chunk in _canonical_chunks(normalized, set()):
        pending.append(chunk)
        size += len(chunk)
        if size >= 1 << 20:
            digest.update("".join(pending).encode("utf-8"))
            pending, size = [], 0
    digest.update("".join(pending).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"
