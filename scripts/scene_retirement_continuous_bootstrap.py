"""Fixed root entrypoint; run with system Python -I -S, never the service venv.

The installed source snapshot is separate from the service-owned checkout.
Only the fixed stdlib-only core runs before the supervisor drops credentials.
No environment path, bytecode cache, or preloaded application module supplies
the root boundary. Installing this entrypoint does not enable retirement.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType


_RUNTIME_ROOT = Path('/mnt/blueprint-work/scene-retirement-runtime')
_OWNER = 0
_CORE = ('__init__.py', 'task_evaluation_scene_retirement_supervisor.py',
         'task_evaluation_scene_retirement_access.py', 'decision_evidence_contracts.py',
         'task_evaluation_scene_retirement_preservation.py',
         'task_evaluation_scene_retirement_generations.py')
_DAEMONS = frozenset(('blueprint_pipeline.live_pipeline_intake_service',
                     'blueprint_pipeline.agent_execution.production'))
_ERROR = 'scene_retirement_bootstrap_unproven'
_MAX_SOURCE = 1024 * 1024


def _require(value):
    if not value:
        raise ValueError(_ERROR)


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _component(info, *, directory):
    _require((stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
             and info.st_uid in {0, _OWNER} and not info.st_mode & 0o022)
    if not directory:
        _require(stat.S_IMODE(info.st_mode) == 0o644 and info.st_nlink == 1
                 and 0 < info.st_size <= _MAX_SOURCE)


def _source(path, *, retain_identity=False):
    """Retain protected ancestry descriptors through exact named source read."""
    _require(path.is_absolute() and '..' not in path.parts)
    descriptors = []
    try:
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        info = os.fstat(fd)
        _component(info, directory=True)
        descriptors.append((fd, _identity(info)))
        for part in path.parts[1:-1]:
            before = os.stat(part, dir_fd=fd, follow_symlinks=False)
            _component(before, directory=True)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            observed = os.fstat(child)
            _require(_identity(observed) == _identity(before))
            descriptors.append((child, _identity(observed)))
            fd = child
        before = os.stat(path.name, dir_fd=fd, follow_symlinks=False)
        _component(before, directory=False)
        child = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
        observed = os.fstat(child)
        _require(_identity(observed) == _identity(before))
        descriptors.append((child, _identity(observed)))
        raw = bytearray()
        while len(raw) <= before.st_size:
            chunk = os.read(child, min(65536, before.st_size + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        _require(len(raw) == before.st_size
                 and _identity(os.fstat(child)) == _identity(before)
                 and _identity(os.stat(path.name, dir_fd=fd, follow_symlinks=False)) == _identity(before))
        return (bytes(raw), _identity(before)) if retain_identity else bytes(raw)
    except OSError as exc:
        raise ValueError(_ERROR) from exc
    finally:
        for owned, version in reversed(descriptors):
            # Never close a reused descriptor token whose acquisition identity
            # has disappeared. This is a refusal, not permission to guess.
            _require(_identity(os.fstat(owned)) == version)
            os.close(owned)


class _RetainedSources(dict):
    """Bytes and acquisition identities retained by the native root loader."""


def _core_sources():
    package = _RUNTIME_ROOT / 'src/blueprint_pipeline'
    values = _RetainedSources()
    identities = {}
    for name in _CORE:
        module = 'blueprint_pipeline' if name == '__init__.py' else 'blueprint_pipeline.' + name[:-3]
        path = package / name
        raw, identity = _source(path, retain_identity=True)
        values[module] = (path, raw)
        identities[module] = identity
    values.identities = MappingProxyType(identities)
    return values


class _SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, values):
        self.values = MappingProxyType(dict(values))
        self.source_identities = MappingProxyType(dict(values.identities))

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.values:
            return None
        source, _ = self.values[fullname]
        return importlib.util.spec_from_loader(fullname, self, origin=str(source),
                                              is_package=fullname == 'blueprint_pipeline')

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        source, raw = self.values[module.__name__]
        module.__file__ = str(source)
        if module.__name__ == 'blueprint_pipeline':
            module.__path__ = [str(source.parent)]
        exec(compile(raw, str(source), 'exec', dont_inherit=True), module.__dict__)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    _require(sys.flags.isolated == 1 and sys.flags.no_site == 1
             and os.getuid() == os.geteuid() == 0
             and len(arguments) >= 2 and arguments[0] == '--continuous-module'
             and arguments[1] in _DAEMONS
             and (len(arguments) == 2 or arguments[2] == '--'))
    _require(not any(name == 'blueprint_pipeline' or name.startswith('blueprint_pipeline.')
                     for name in sys.modules))
    loader = _SourceOnly(_core_sources())
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, loader)
    try:
        module = importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_supervisor')
        return module.main(arguments)
    finally:
        sys.meta_path.remove(loader)


if __name__ == '__main__':
    raise SystemExit(main())
