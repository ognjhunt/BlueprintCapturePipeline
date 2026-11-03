"""A compile's binary copy of a USDA text collider, read once; the published file is never touched.

Beta journey step: L5 team policy evaluation (``team_scene_packet``). A website
collider is a USDA text file (capture-447's: one 292 MB mesh of 3.25 M points),
and OpenUSD parses the whole text again at every open that finds no stage of it
alive, transiently at about twelve times the file's size (3.4 GB for a 280 MB
synthetic room). A team packet opened it 22 times, and the host's OOM killer
ended capture-447's compile twice on 2026-11-03 with the unit at 2.4 and 3.7 GB,
under its 8 GB cap.

Inside ``compile_scoped``, ``binary_copy`` writes the collider once as a USDC
file under the compile's own root, holds it open, and deletes it when the
compile ends. Readers open ``readable(path)``, which is that copy for the source
or any hard link of it (the packet stages the collider as one); a step that may
edit what it opens takes ``editable(path)``, its own in-memory copy, so an edit
never reaches another reader. Receipts, digests and staged bytes keep naming the
published file; the copy is only read.

The copy is written without OpenUSD's text parse of the large arrays: each
numeric array literal of at least ``LARGE_ARRAY_BYTES`` is read with numpy
(text to double, correctly rounded, then to float, as the text parser converts
it), and OpenUSD parses the rest with those literals replaced by markers it then
finds by value. Anything it does not recognise (an unknown type, a comment or a
word in a literal, a marker not found exactly once, a 16-digit integer) falls
back to OpenUSD's own parse of the whole file, still once per compile. A
collider with composition dependencies (sublayers, references, payloads) is not
copied: its relative paths would resolve from the copy's directory.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

#: Array literals at least this long are read with numpy; smaller ones are left to OpenUSD.
LARGE_ARRAY_BYTES = 1 << 20
#: Marker values for a replaced literal: never a real coordinate or index next to its ordinal.
_MARK_FLOAT = -1.5e38
_MARK_INT = -2147483647
_TYPES = {b"point3f": 3, b"normal3f": 3, b"vector3f": 3, b"float3": 3, b"color3f": 3,
          b"texCoord2f": 2, b"float2": 2, b"float": 1, b"int": 0}
_DECLARATION = re.compile(rb"(?<![\w:.])(" + b"|".join(_TYPES) + rb")\[\][ \t]+[A-Za-z_][\w:]*[ \t]*=[ \t]*\[")
_NUMERIC = b" \t\r\n0123456789+-.eE,()"
_LONG_INTEGER = re.compile(rb"(?<![\w.+-])[+-]?\d{16,}(?![\w.])")

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class _Copy:
    source: Path
    version: tuple[int, int]  # the source's size and mtime when copied, checked again at every read
    path: Path
    stage: Any
    method: str


#: Keyed by the source's (device, inode), so a hard link of it (a staged packet asset) reads the copy too.
_COPIES: dict[tuple[int, int], _Copy] = {}
_SCOPES: list[list[tuple[int, int]]] = []


def _identity(path: str | Path) -> tuple[tuple[int, int], tuple[int, int]] | None:
    try:
        status = os.stat(path)
    except OSError:
        return None
    return (status.st_dev, status.st_ino), (status.st_size, status.st_mtime_ns)


def _copy_for(path: str | Path) -> _Copy | None:
    identity = _identity(path)
    copy = None if identity is None else _COPIES.get(identity[0])
    return copy if copy is not None and identity is not None and copy.version == identity[1] else None


def readable(path: str | Path) -> str:
    """What to open to read ``path``: the compile's binary copy of it (or of the file it hard-links), else itself."""

    copy = _copy_for(path)
    return str(copy.path) if copy is not None else str(path)


def editable(path: str | Path) -> Any:
    """What to open to read ``path`` and maybe edit it in memory: an anonymous copy no other reader shares."""

    copy = _copy_for(path)
    if copy is None:
        return str(path)
    from pxr import Sdf

    return Sdf.Layer.OpenAsAnonymous(str(copy.path))


@contextmanager
def read_scope() -> Iterator[None]:
    """The copies ``binary_copy`` makes inside are released and deleted when the block ends."""

    made: list[tuple[int, int]] = []
    _SCOPES.append(made)
    try:
        yield
    finally:
        _SCOPES.remove(made)
        for key in made:
            copy = _COPIES.pop(key, None)
            if copy is None:
                continue
            copy.stage = None
            copy.path.unlink(missing_ok=True)
            try:
                copy.path.parent.rmdir()
            except OSError:
                pass


def compile_scoped(function: F) -> F:
    """``function`` runs inside its own ``read_scope``."""

    @functools.wraps(function)
    def scoped(*args: Any, **kwargs: Any) -> Any:
        with read_scope():
            return function(*args, **kwargs)

    return scoped  # type: ignore[return-value]


def binary_copy(source: str | Path, directory: str | Path) -> Path:
    """The binary copy of text collider ``source`` in ``directory``, made on the first call inside a scope.

    Outside a scope, for a file that is not USDA text, or for one with composition dependencies, ``source``
    itself.
    """

    source = Path(source)
    existing = _copy_for(source)
    if existing is not None:
        return existing.path
    identity = _identity(source)
    if not _SCOPES or identity is None:
        return source
    with source.open("rb") as handle:
        if handle.read(5) != b"#usda":
            return source  # binary already: every open of it is cheap
    from pxr import Sdf, Tf, Usd

    try:
        layer, method = _fast_layer(source, minimum_bytes=LARGE_ARRAY_BYTES), "numpy_large_arrays"
    except (Tf.ErrorException, ValueError, TypeError, OverflowError):
        layer = None
    if layer is None:
        try:
            layer, method = Sdf.Layer.OpenAsAnonymous(str(source)), "openusd_text_parse"
        except Tf.ErrorException:
            return source  # the reader that needs it refuses by its own name
    if layer is None or layer.GetCompositionAssetDependencies():
        return source
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / (source.stem + ".usdc")
    try:
        exported = layer.Export(str(path))
    except Tf.ErrorException:
        exported = False
    del layer
    if not exported:
        shutil.rmtree(target, ignore_errors=True)
        return source
    _COPIES[identity[0]] = _Copy(source=source, version=identity[1], path=path,
                                 stage=Usd.Stage.Open(str(path)), method=method)
    _SCOPES[-1].append(identity[0])
    return path


def copy_method(path: str | Path) -> str | None:
    """How the compile's copy of ``path`` was read (``numpy_large_arrays`` or ``openusd_text_parse``), or None."""

    copy = _copy_for(path)
    return None if copy is None else copy.method


def _numbers(literal: bytes, arity: int) -> Any:
    """A numeric array literal's values as numpy, exactly as the text parser converts them; None to decline."""

    import numpy as np

    if literal.translate(None, _NUMERIC) or _LONG_INTEGER.search(literal):
        return None
    tuples = literal.count(b"(")
    if arity > 1:
        if tuples != literal.count(b")") or not tuples:
            return None
        flat = literal.translate(bytes.maketrans(b"()", b"  "))
        expected_commas = tuples * (arity - 1) + tuples - 1
    else:
        if tuples or literal.count(b")"):
            return None
        flat = literal
        expected_commas = None
    if arity == 0 and (b"." in flat or b"e" in flat or b"E" in flat):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            values = np.fromstring(flat, dtype=np.int64 if arity == 0 else np.float64, sep=",")
        except (ValueError, DeprecationWarning, RuntimeWarning):
            return None
        commas = flat.count(b",")
        if expected_commas is None:
            expected_commas = len(values) - 1
            if commas != expected_commas or not len(values):
                return None
        elif commas != expected_commas or len(values) != tuples * arity:
            return None
        if arity == 0:
            if values.min() < -(1 << 31) or values.max() >= 1 << 31:
                return None
            return values.astype(np.int32)
        try:
            narrowed = values.astype(np.float32)
        except (FloatingPointError, RuntimeWarning):
            return None
    return narrowed.reshape(-1, arity) if arity > 1 else narrowed


def _fast_layer(source: Path, *, minimum_bytes: int) -> Any:
    """``source`` as an anonymous layer, its large numeric arrays read with numpy; None to decline."""

    import numpy as np
    from pxr import Sdf, Tf, Vt

    data = source.read_bytes()
    if not data.startswith(b"#usda"):
        return None
    pieces, arrays, cursor = [], [], 0
    for match in _DECLARATION.finditer(data):
        start = match.end()
        if start < cursor:
            continue
        end = data.find(b"]", start)
        if end < 0:
            return None
        if end - start < minimum_bytes:
            continue
        kind = match.group(1)
        arity = _TYPES[kind]
        values = _numbers(data[start:end], arity)
        if values is None:
            return None
        ordinal = len(arrays)
        if arity > 1:
            marker = "(" + ", ".join([repr(_MARK_FLOAT), str(ordinal)] + ["0"] * (arity - 2)) + ")"
        else:
            marker = f"{_MARK_INT if arity == 0 else repr(_MARK_FLOAT)}, {ordinal}"
        pieces += [data[cursor:start], marker.encode()]
        arrays.append((arity, values))
        cursor = end
    if not arrays:
        return None
    pieces.append(data[cursor:])
    del data
    layer = Sdf.Layer.CreateAnonymous(".usda")
    try:
        if not layer.ImportFromString(b"".join(pieces).decode("utf-8")):
            return None
    except (Tf.ErrorException, UnicodeDecodeError):
        return None
    del pieces
    found: dict[int, list[Any]] = {}
    array_types = {0: Vt.IntArray, 1: Vt.FloatArray, 2: Vt.Vec2fArray, 3: Vt.Vec3fArray}

    def visit(path: Any) -> None:
        spec = layer.GetAttributeAtPath(path) if path.IsPropertyPath() else None
        if spec is None or not spec.HasDefaultValue() or type(spec.default) not in array_types.values():
            return
        value = spec.default
        if len(value) == 1 and isinstance(value, (Vt.Vec2fArray, Vt.Vec3fArray)):
            first, ordinal = value[0][0], value[0][1]
        elif len(value) == 2 and isinstance(value, (Vt.IntArray, Vt.FloatArray)):
            first, ordinal = value[0], value[1]
        else:
            return
        if first in (_MARK_INT, np.float32(_MARK_FLOAT)) and float(ordinal).is_integer():
            found.setdefault(int(ordinal), []).append(spec)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    if sorted(found) != list(range(len(arrays))) or any(len(specs) != 1 for specs in found.values()):
        return None
    for ordinal, (arity, values) in enumerate(arrays):
        spec = found[ordinal][0]
        if type(spec.default) is not array_types[arity]:
            return None
        spec.default = array_types[arity].FromNumpy(values)
    return layer


__all__ = ["LARGE_ARRAY_BYTES", "binary_copy", "compile_scoped", "copy_method", "editable", "read_scope",
           "readable"]
