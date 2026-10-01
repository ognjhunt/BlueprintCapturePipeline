"""Stage the finite historical source closure before the operator code swap.

This installer does not start units, enable flags or acquire owner decisions.
The normal installer later seals the staged tree root-owned and read-only to
the service account. No payload, checkout or service venv is copied.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import stat
import sys


def _identity(info):
    return tuple(getattr(info, name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_uid',
        'st_gid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def stage(origin, destination):
    package = destination / 'blueprint_pipeline'
    package.mkdir(parents=True, mode=0o755)
    pending = ['__init__', 'control_plane_lane_historical_action', 'control_plane_lane_historical_gc']
    selected, size = set(), 0
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('historical_generation_stage_unproven')
        path = origin / (name + '.py')
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= 1024**2:
                raise ValueError('historical_generation_stage_unproven')
            raw = os.read(descriptor, before.st_size + 1)
            if len(raw) != before.st_size or _identity(os.fstat(descriptor)) != _identity(before) \
                    or _identity(path.lstat()) != _identity(before):
                raise ValueError('historical_generation_stage_unproven')
        finally:
            os.close(descriptor)
        selected.add(name)
        size += len(raw)
        if len(selected) > 256 or size > 8 * 1024**2:
            raise ValueError('historical_generation_stage_unproven')
        with (package / path.name).open('xb') as output:
            output.write(raw)
        (package / path.name).chmod(0o644)
        for node in ast.walk(ast.parse(raw)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level == 1:
                names = [node.module.split('.')[0]] if node.module else [alias.name for alias in node.names]
            elif node.module == 'blueprint_pipeline':
                names = [alias.name for alias in node.names]
            elif node.module and node.module.startswith('blueprint_pipeline.'):
                names = [node.module.split('.')[1]]
            else:
                names = []
            pending.extend(item for item in names if (origin / (item + '.py')).is_file())


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(2)
    stage(Path(sys.argv[1]), Path(sys.argv[2]))
