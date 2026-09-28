"""Search what stays in a sealed result run for every file of the run it names.

The residue offload (``task_evaluation_result_residue_offload``) moves a file only
when no reader can reach it from what stays. Readers reopen files that a kept
receipt, manifest, log or interpretation names by path, so every file that stays
is searched here, whatever its format or size, and every file of the run it
names stays too, and is searched in turn.

A file is read as a stream, a chunk at a time, and every run of name characters
(``NAME_CHARACTERS``) in it is a candidate path. A token cut by a chunk boundary
is carried whole into the next chunk (only its last ``_MAX_TOKEN_CHARS``, since
a path it holds must end there). ``\\/`` reads as ``/``, while any other JSON
escape (``\\n``, ``\\t``, ``\\uXXXX``) separates tokens, so an escape never glues
itself to a path; a token is also read without the ``.``, ``-`` or ``~`` a
sentence can end it with; and a relative token with a ``/`` names every file of
the run whose path ends with it, since it may be relative to any directory. A
binary (a NUL in its first 64 KiB) names nothing. So a residue member may only
be named with name characters (``name_supported``): no document can then name it
in a form this search does not read.
"""

from __future__ import annotations

import os
import posixpath
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

#: The characters a residue member's name is made of. The search reads any run of them
#: in a text as a candidate path, so a member named with anything else stays (``name_unsupported``).
NAME_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-+@%~/"
_PATH_TOKEN = re.compile("[" + re.escape(NAME_CHARACTERS) + "]+")
#: A token that can continue into the next chunk, including a ``\\`` that may escape a ``/``.
_CARRIED_CHARACTERS = NAME_CHARACTERS + "\\"
_SCAN_CHUNK_BYTES = 1024 * 1024
#: A token longer than any path keeps its tail, where a path it holds must end.
_MAX_TOKEN_CHARS = 16 * 1024
#: A file with a NUL in its first 64 KiB is binary and names nothing.
_BINARY_SNIFF_BYTES = 64 * 1024
#: A JSON escape: ``\/`` is a slash of a path, any other (``\n``, ``\t``, ``\uXXXX``...) separates.
_ESCAPE = re.compile(r"\\(u[0-9A-Fa-f]{4}|.)", re.DOTALL)
#: Punctuation a sentence can end a path with; a token is also read without it.
_TRAILING_PUNCTUATION = ".-~"


def _unescaped(text: str) -> str:
    return _ESCAPE.sub(lambda match: "/" if match.group(1) == "/" else " ", text)


class ResultResidueOffloadError(RuntimeError):
    """A residue could not be offloaded or restored safely."""


def name_supported(relative: str) -> bool:
    """Whether the search can recognize the name wherever a text writes it.

    Only such a name can be proven unnamed by every kept document; ``_pack_stream``
    and the pointer carry it as well.
    """

    return _PATH_TOKEN.fullmatch(relative) is not None


def stream_tokens(stream, *, chunk_bytes: int = _SCAN_CHUNK_BYTES) -> Iterator[set[str]]:
    """The path-like tokens of an open file, one chunk's worth at a time; nothing for a binary.

    Neither memory nor a chunk's work grows with the file: a token cut by a chunk
    boundary is carried into the next (only its last ``_MAX_TOKEN_CHARS``, since a
    path it holds must end there), with any escape it ends in. ``\\/`` reads as
    ``/``; any other escape (``\\n``, ``\\t``, ``\\uXXXX``) and every byte outside the
    name characters only separate tokens.
    """

    carry, first = "", True
    while True:
        chunk = stream.read(chunk_bytes)
        if first:
            first = False
            if b"\x00" in chunk[:_BINARY_SNIFF_BYTES]:
                return
        if not chunk:
            break
        text = carry + chunk.decode("latin-1")
        cut = len(text.rstrip(_CARRIED_CHARACTERS))
        carry = text[cut:][-_MAX_TOKEN_CHARS:]
        yield set(_PATH_TOKEN.findall(_unescaped(text[:cut])))
    if carry:
        yield {token[-_MAX_TOKEN_CHARS:] for token in _PATH_TOKEN.findall(_unescaped(carry))}


def document_tokens(root: Path, relative: str) -> Iterator[set[str]]:
    """``stream_tokens`` of a file of the run.

    It must still be the regular file the walk listed. One that is gone names
    nothing; one that cannot be opened or read, or is no longer a regular file,
    raises, since what it names is unknown.
    """

    try:
        descriptor = os.open(root / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return  # gone since it was listed: it names nothing
    except OSError as exc:
        raise ResultResidueOffloadError("result_residue_reference_document_unreadable") from exc
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ResultResidueOffloadError("result_residue_reference_document_changed")
        try:
            yield from stream_tokens(stream, chunk_bytes=_SCAN_CHUNK_BYTES)
        except OSError as exc:
            raise ResultResidueOffloadError("result_residue_reference_document_unreadable") from exc


def named_paths(strings, document: str, run_name: str, basenames: set[str] | None = None,
                suffixes: Mapping[str, Sequence[str]] | None = None):
    """The run-relative paths a document's strings may name.

    That is what follows each ``/<run>/`` (the run's name may recur deeper in the
    path, so every occurrence counts), what follows a leading ``<run>/`` (a path
    relative to the evidence root), and a relative string joined to every
    directory from the document's own up to the run root: with ``basenames``,
    only a string whose last component is the name of a file of the run. A
    relative string with a ``/`` may be relative to any directory, so with
    ``suffixes`` (``suffix_index``) every file of the run it ends is named too.
    Each string is also read without the punctuation a sentence can end it with.
    """

    marker = f"/{run_name}/"
    bases = [str(parent) for parent in PurePosixPath(document).parents]
    for raw in strings:
        for string in {raw, raw.rstrip(_TRAILING_PUNCTUATION)} - {""}:
            text = string if string.startswith("/") else "/" + string
            start = text.find(marker)
            while start != -1:
                yield text[start + len(marker):]
                start = text.find(marker, start + 1)
            if string.startswith("/"):
                continue
            if basenames is None or string.rsplit("/", 1)[-1] in basenames:
                for base in bases:
                    yield string if base == "." else f"{base}/{string}"
            if suffixes is not None and "/" in string:
                tail = posixpath.normpath(string)
                while tail.startswith("../"):
                    tail = tail[3:]
                yield from suffixes.get(tail, ())


def suffix_index(files) -> dict[str, list[str]]:
    """Every file of the run under each tail of its path that starts a component: ``a/b/c`` is
    under ``a/b/c``, ``b/c`` and ``c``."""

    index: dict[str, list[str]] = {}
    for path in files:
        parts = path.split("/")
        for start in range(len(parts)):
            index.setdefault("/".join(parts[start:]), []).append(path)
    return index


def receipt_references(root: Path, documents: Sequence[str], files: Mapping[str, Any]) -> set[str]:
    """Every file of the run that a searched document names, closed over what those name in turn.

    Billing re-validation, spend ledgers and rescoring reopen the files a sealed
    receipt binds by path (the terminal result, and the adapter result and
    manifests it names), so a file any kept document names is kept too, and is
    searched in turn whether or not it is residue. ``files`` is every regular file
    the walk listed.
    """

    named: set[str] = set()
    basenames = {PurePosixPath(path).name for path in files}
    suffixes = suffix_index(files)
    queue, searched = list(documents), set()
    while queue:
        document = queue.pop()
        if document in searched:
            continue
        searched.add(document)
        for tokens in document_tokens(root, document):
            for name in named_paths(tokens, document, root.name, basenames, suffixes):
                path = posixpath.normpath(name)
                if path in files and path not in named:
                    named.add(path)
                    queue.append(path)
    return named


__all__ = [
    "NAME_CHARACTERS",
    "ResultResidueOffloadError",
    "document_tokens",
    "name_supported",
    "named_paths",
    "receipt_references",
    "stream_tokens",
    "suffix_index",
]
