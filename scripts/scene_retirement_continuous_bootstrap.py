"""Fixed root entrypoint; run with system Python -I -S, never the service venv.

The installed source snapshot is separate from the service-owned checkout.
Only the fixed stdlib-only core runs before the supervisor drops credentials.
No environment path, bytecode cache, or preloaded application module supplies
the root boundary. Installing this entrypoint does not enable retirement.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import MappingProxyType


_RUNTIME_ROOT = Path('/mnt/blueprint-work/scene-retirement-runtime')
_BOOT_ROOT = Path('/usr/lib/blueprint/scene-retirement-runtime')
_OWNER = 0
_CORE = ('__init__.py', 'task_evaluation_scene_retirement_supervisor.py',
         'task_evaluation_scene_retirement_access.py', 'decision_evidence_contracts.py',
         'task_evaluation_scene_retirement_preservation.py',
         'task_evaluation_scene_retirement_generations.py')
_DAEMONS = frozenset(('blueprint_pipeline.live_pipeline_intake_service',
                     'blueprint_pipeline.agent_execution.production'))
_ACTIONS = frozenset(('blueprint_pipeline.control_plane_storage_gc',
                     'blueprint_pipeline.task_evaluation_scene_retirement_cli'))
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


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value)
        value[key] = item
    return value


def _select_runtime():
    """Select one protected source/SDK tuple before application import."""
    current = _BOOT_ROOT / 'CURRENT.json'
    if not os.path.lexists(current):
        return _RUNTIME_ROOT, _RUNTIME_ROOT / 'dependencies'
    raw = _source(current)
    _require(len(raw) <= 4096)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError(_ERROR) from exc
    fields = {'schema', 'runtime_root', 'dependencies_root', 'source_digest',
              'dependency_digest', 'bootstrap', 'previous'}
    _require(type(value) is dict and set(value) == fields
             and value['schema'] == 'scene-retirement-runtime-cohort.v1')
    for key in ('source_digest', 'dependency_digest'):
        digest = value[key]
        _require(type(digest) is str and len(digest) == 64
                 and all(char in '0123456789abcdef' for char in digest))
    source = _RUNTIME_ROOT / 'generations' / value['source_digest']
    sdk = Path(value['dependencies_root']) if type(value['dependencies_root']) is str else None
    _require(value['runtime_root'] == str(source)
             and sdk in {_RUNTIME_ROOT / 'dependencies',
                         _RUNTIME_ROOT / 'sdk-generations' / value['dependency_digest']})
    bootstrap = value['bootstrap']
    _require(type(bootstrap) is dict and set(bootstrap) == {'sha256', 'size', 'mode'}
             and type(bootstrap['size']) is int and bootstrap['size'] > 0
             and type(bootstrap['mode']) is int and bootstrap['mode'] == 0o644)
    installed = _BOOT_ROOT / 'continuous_bootstrap.py'
    _require(Path(__file__) == installed)
    executable = _source(installed)
    _require(len(executable) == bootstrap['size']
             and hashlib.sha256(executable).hexdigest() == bootstrap['sha256'])
    _directory(source / 'src')
    _directory(source / 'scripts')
    _directory(sdk)
    _require(_source(current) == raw)
    return source, sdk


def _core_sources():
    root, dependencies = _select_runtime()
    package = root / 'src/blueprint_pipeline'
    values = _RetainedSources()
    identities = {}
    for name in _CORE:
        module = 'blueprint_pipeline' if name == '__init__.py' else 'blueprint_pipeline.' + name[:-3]
        path = package / name
        raw, identity = _source(path, retain_identity=True)
        values[module] = (path, raw)
        identities[module] = identity
    values.identities = MappingProxyType(identities)
    values.runtime_root = root
    values.dependencies_root = dependencies
    return values


def _action_sources(module):
    _require(module in _ACTIONS)
    values = _core_sources()
    path = values.runtime_root / 'src' / (module.replace('.', '/') + '.py')
    raw, identity = _source(path, retain_identity=True)
    values[module] = (path, raw)
    values.identities = MappingProxyType(dict(values.identities) | {module: identity})
    return values


def _directory(path):
    descriptors = []
    try:
        fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        _component(os.fstat(fd), directory=True)
        descriptors.append(fd)
        for name in path.parts[1:]:
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            _component(before, directory=True)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=fd)
            descriptors.append(child)
            _require(_identity(os.fstat(child)) == _identity(before))
            fd = child
    except OSError as exc:
        raise ValueError(_ERROR) from exc
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _action_main(module, arguments):
    """Root GC and door actions share one installed, isolated entry boundary."""
    values = _action_sources(module)
    root, dependencies = values.runtime_root, values.dependencies_root
    _directory(root / 'src')
    _directory(dependencies)
    _directory(root / 'scripts')
    stdlib = [value for value in sys.path if value and value.startswith('/usr/lib/python')]
    _require(stdlib)
    for path in stdlib:
        if Path(path).is_dir():
            _directory(Path(path))
    sys.path[:] = [str(root / 'src'), str(dependencies), str(root), *stdlib]
    os.chdir('/')
    loader = _SourceOnly(values)
    sys.meta_path.insert(0, loader)
    try:
        supervisor = importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_supervisor')
        with supervisor._installed_policy_binding():
            selected = importlib.import_module(module)
            _require(callable(getattr(selected, 'main', None)))
            return selected.main(arguments)
    finally:
        sys.meta_path.remove(loader)


class _SourceOnly(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, values):
        self.values = MappingProxyType(dict(values))
        self.source_identities = MappingProxyType(dict(values.identities))
        self.runtime_root = values.runtime_root
        self.dependencies_root = values.dependencies_root

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
        if module.__name__ == 'blueprint_pipeline.task_evaluation_scene_retirement_supervisor':
            module._TRUSTED_SOURCE_ROOT = self.runtime_root / 'src'
            module._TRUSTED_DEPENDENCIES = self.dependencies_root


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    _require(sys.flags.isolated == 1 and sys.flags.no_site == 1
             and os.getuid() == os.geteuid() == 0
             and 2 <= len(arguments) <= 64
             and all(type(value) is str and len(value.encode()) <= 4096 for value in arguments))
    _require((arguments[0] == '--continuous-module' and arguments[1] in _DAEMONS
              and (len(arguments) == 2 or arguments[2] == '--'))
             or (arguments[0] == '--action-module' and arguments[1] in _ACTIONS))
    _require(not any(name == 'blueprint_pipeline' or name.startswith('blueprint_pipeline.')
                     for name in sys.modules))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    if arguments[0] == '--action-module':
        return _action_main(arguments[1], arguments[2:])
    loader = _SourceOnly(_core_sources())
    sys.meta_path.insert(0, loader)
    try:
        module = importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_supervisor')
        return module.main(arguments)
    finally:
        sys.meta_path.remove(loader)


if __name__ == '__main__':
    raise SystemExit(main())
