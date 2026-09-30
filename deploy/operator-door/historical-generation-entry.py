"""Fixed root historical action entry, invoked by system Python -I -S.

No service checkout, venv, caller command/config/path, user site or environment
import path is admitted. Every application import uses protected installed
source bytes. Optional archive SDK imports use the root system distribution;
absence is a typed worker refusal, never a preservation/readback substitute.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

_ROOT = Path('/opt/blueprint/operator-door')
_CONFIG = '/etc/blueprint-operator-door/door.json'
_SYSTEM_PACKAGES = Path('/usr/lib/python3/dist-packages')
_ERROR = 'historical_generation_entry_unproven'


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _identity(info):
    return tuple(getattr(info, name) for name in ('st_dev', 'st_ino', 'st_mode', 'st_uid',
        'st_gid', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def _protected(info, *, directory=False):
    return (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)) \
        and info.st_uid == 0 and not info.st_mode & 0o022 \
        and (directory or info.st_nlink == 1)


def _open(path, *, directory=False):
    _require(path.is_absolute() and '..' not in path.parts)
    descriptors = []
    try:
        parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        descriptors.append(parent)
        _require(_protected(os.fstat(parent), directory=True))
        for index, name in enumerate(path.parts[1:]):
            is_directory = directory or index < len(path.parts) - 2
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            _require(_protected(before, directory=is_directory))
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                | (os.O_DIRECTORY if is_directory else 0), dir_fd=parent)
            descriptors.append(child)
            _require(_identity(os.fstat(child)) == _identity(before))
            parent = child
        return descriptors
    except (OSError, ValueError):
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise ValueError(_ERROR) from None


def _read_source(path):
    descriptors = _open(path)
    try:
        before = os.fstat(descriptors[-1])
        _require(0 < before.st_size <= 1024**2)
        raw = bytearray()
        while len(raw) <= before.st_size:
            part = os.read(descriptors[-1], min(65536, before.st_size + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        _require(len(raw) == before.st_size
            and _identity(os.fstat(descriptors[-1])) == _identity(before)
            and _identity(os.stat(path.name, dir_fd=descriptors[-2], follow_symlinks=False))
                == _identity(before))
        return bytes(raw)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _directory(path):
    for descriptor in reversed(_open(path, directory=True)):
        os.close(descriptor)


class _Sources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Application source never falls back to a writable checkout or bytecode."""
    def __init__(self):
        self.sources = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'blueprint_pipeline':
            source = _ROOT / 'historical-python/blueprint_pipeline/__init__.py'
        elif fullname.startswith('blueprint_pipeline.'):
            name = fullname.removeprefix('blueprint_pipeline.')
            _require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name))
            source = _ROOT / 'historical-python/blueprint_pipeline' / (name + '.py')
        else:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None:
                for directory in spec.submodule_search_locations or ():
                    _directory(Path(directory))
                if spec.origin and spec.origin not in ('built-in', 'frozen'):
                    descriptors = _open(Path(spec.origin))
                    for descriptor in reversed(descriptors):
                        os.close(descriptor)
            return spec
        self.sources[fullname] = (source, _read_source(source))
        return importlib.util.spec_from_loader(fullname, self, origin=str(source),
                                               is_package=fullname == 'blueprint_pipeline')

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source, raw = self.sources.pop(module.__name__)
        module.__file__ = str(source)
        if module.__name__ == 'blueprint_pipeline':
            module.__path__ = [str(source.parent)]
        exec(compile(raw, str(source), 'exec', dont_inherit=True), module.__dict__)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    _require(type(arguments) is list and len(arguments) == 1
        and type(arguments[0]) is str and re.fullmatch(r'[0-9a-f]{32}', arguments[0])
        and sys.flags.isolated == 1 and sys.flags.no_site == 1
        and os.getuid() == os.geteuid() == 0)
    _require(not any(name == 'blueprint_pipeline' or name.startswith('blueprint_pipeline.')
                     for name in sys.modules))
    _directory(_ROOT / 'historical-python')
    standard = [value for value in sys.path if value and value.startswith('/usr/lib/python')
                and 'site-packages' not in value and 'dist-packages' not in value]
    _require(standard)
    for value in standard:
        if Path(value).is_dir():
            _directory(Path(value))
    sys.path[:] = standard
    if _SYSTEM_PACKAGES.exists():
        _directory(_SYSTEM_PACKAGES)
        sys.path.append(str(_SYSTEM_PACKAGES))
    sys.dont_write_bytecode = True
    os.chdir('/')
    loader = _Sources()
    sys.meta_path.insert(0, loader)
    try:
        from blueprint_pipeline.control_plane_lane_historical_action import run_historical_action
        receipt = run_historical_action(installed_config_path=_CONFIG, action_id=arguments[0], now=time.time())
        print(json.dumps(receipt, sort_keys=True, separators=(',', ':')))
        return 0
    finally:
        sys.meta_path.remove(loader)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(json.dumps(dict(status='kept', code=getattr(error, 'code', _ERROR))), file=sys.stderr)
        raise SystemExit(1) from None
