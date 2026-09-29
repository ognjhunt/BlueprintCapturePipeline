"""Finite unsupported-publisher refusal; this does not clear references."""
from __future__ import annotations

import os
import math
import posixpath
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .control_plane_lane_experiment_errors import OwnerTargetVersionError
from .control_plane_reference_budget import ReferenceCollectionBudget, ReferenceCollectionBudgetError

_RESERVED = re.compile(r'(?:^|/) (?:g1|arena)/registered-', re.X)
_RAW_LIMIT = 65536


def _reference(text, budget):
    # A URI is interpreted by its declared scheme. Remote URI names cannot
    # confer local authority; file URIs must have no remote authority/query.
    parsed = urlsplit(text) if '://' in text else None
    if parsed is not None:
        path = unquote(parsed.path)
        local = parsed.scheme == 'file'
        if local and (parsed.netloc not in ('', 'localhost') or parsed.query or parsed.fragment):
            raise OwnerTargetVersionError('experiment_publisher_input_limit')
    else:
        path, local = text, True
    normalized = posixpath.normpath(path.replace('\\', '/'))
    if _RESERVED.search(normalized):
        raise OwnerTargetVersionError('experiment_external_publisher_unsupported')
    # Only path-shaped local strings have a filesystem interpretation. Walk
    # one finite pathname without following links; never resolve a payload.
    if local and '/' in path:
        selected = Path(path)
        if not selected.is_absolute():
            selected = Path.cwd() / selected
        if len(selected.parts) > 64:
            raise OwnerTargetVersionError('experiment_publisher_input_limit')
        current = Path(selected.anchor)
        for component in selected.parts[1:]:
            budget.charge('values')
            current = current / component
            try:
                info = os.stat(current, follow_symlinks=False)
            except FileNotFoundError:
                break
            if stat.S_ISLNK(info.st_mode):
                # macOS's immutable system /var alias is part of its named
                # root layout, not a caller-selected target alias. Every other
                # link still refuses before payload or publisher mutation.
                system_var = (sys.platform == 'darwin' and current == Path('/var')
                              and info.st_uid == 0 and os.readlink(current) == 'private/var')
                if not system_var:
                    raise OwnerTargetVersionError('experiment_external_publisher_unsupported')


def refuse_registered_references(*values):
    """Observe actual inputs under ONE native 5s/10k/64KiB allowance.

    Recursion is depth-bounded and charged before descent; no pending array is
    expanded before admission. This is an unsupported-family refusal, not a
    complete reference scanner or permission to publish/deallocate a target.
    Relative paths use the calling process's actual current directory. URI
    paths use their scheme; no supplied base or regex creates ownership.
    """
    budget = ReferenceCollectionBudget(values_limit=10000)
    raw = 0

    def charge_raw(amount):
        nonlocal raw
        if amount > _RAW_LIMIT - raw:
            raise OwnerTargetVersionError('experiment_publisher_input_limit')
        budget.charge('raw_bytes', amount)
        raw += amount

    def visit(value, depth):
        budget.charge('values')
        if depth > 64:
            raise OwnerTargetVersionError('experiment_publisher_input_limit')
        if isinstance(value, (str, Path)):
            if isinstance(value, Path) and (len(value.parts) > 64 or sum(len(part) + 1 for part in value.parts) > 4096):
                raise OwnerTargetVersionError('experiment_publisher_input_limit')
            text = str(value)
            charge_raw(2)
            if len(text) > _RAW_LIMIT - raw:
                raise OwnerTargetVersionError('experiment_publisher_input_limit')
            for index, char in enumerate(text):
                if index % 1024 == 0:
                    budget.tick()
                ordinal = ord(char)
                if 0xD800 <= ordinal <= 0xDFFF:
                    raise OwnerTargetVersionError('experiment_publisher_input_limit')
                width = (2 if char in '\\"\b\f\n\r\t' else 6 if ordinal < 32 else
                         1 if ordinal < 128 else 2 if ordinal < 2048 else 3 if ordinal < 65536 else 4)
                charge_raw(width)
            _reference(text, budget)
        elif isinstance(value, Mapping):
            budget.available('values', len(value) * 2)
            charge_raw(2 + len(value) + max(0, len(value) - 1))
            for key, item in value.items():
                visit(key, depth + 1)
                visit(item, depth + 1)
        elif isinstance(value, (tuple, list)):
            budget.available('values', len(value))
            charge_raw(2 + max(0, len(value) - 1))
            for item in value:
                visit(item, depth + 1)
        elif value is None:
            charge_raw(4)
        elif type(value) is bool:
            charge_raw(4 if value else 5)
        elif type(value) is int and value.bit_length() <= 4096:
            charge_raw(len(str(value)))
        elif type(value) is float and math.isfinite(value):
            charge_raw(len(str(value)))
        else:
            raise OwnerTargetVersionError('experiment_publisher_input_limit')

    try:
        for value in values:
            visit(value, 0)
        budget.tick()
    except (ReferenceCollectionBudgetError, OSError, ValueError) as exc:
        if isinstance(exc, OwnerTargetVersionError):
            raise
        raise OwnerTargetVersionError('experiment_publisher_input_limit') from None
    finally:
        budget.close()
