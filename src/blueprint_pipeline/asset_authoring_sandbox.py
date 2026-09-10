"""Run candidate CAD/Blender programs without network or production credentials.

This is an OS isolation boundary, not a Python AST sandbox. The caller supplies
only required runtime/source roots; the sole writable retained root is one new
authoring attempt. Missing OS isolation refuses before model spend.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import pwd
import shutil
import subprocess
from collections.abc import Sequence


class AssetSandboxError(RuntimeError):
    pass


class SandboxedAssetRunner:
    def __init__(self, *, read_roots: Sequence[Path], write_root: Path,
                 executable_roots: Sequence[Path] = ()) -> None:
        self.write_root = Path(write_root).resolve(strict=True)
        self.read_roots = tuple(Path(p).resolve(strict=True) for p in read_roots)
        self.executable_roots = tuple(Path(p).resolve(strict=True) for p in executable_roots)
        forbidden = (Path.home(), Path('/'), Path('/etc'), Path('/Users'), Path('/var'))
        if self.write_root in forbidden or any(p in forbidden for p in self.read_roots):
            raise AssetSandboxError('asset_sandbox_root_too_broad')
        self.system = platform.system()
        self.launcher = shutil.which('sandbox-exec' if self.system == 'Darwin' else 'bwrap')

    def preflight(self) -> None:
        if self.system not in {'Darwin', 'Linux'} or not self.launcher:
            raise AssetSandboxError('asset_sandbox_runtime_missing')
        result = self(['/usr/bin/true'], cwd=self.write_root, timeout=15,
                      capture_output=True, text=True, check=False)
        if result.returncode:
            raise AssetSandboxError('asset_sandbox_unavailable')

    def __call__(self, argv, *, cwd=None, env=None, timeout=300, check=False,
                 capture_output=True, text=True, **kwargs):
        if not self.launcher:
            raise AssetSandboxError('asset_sandbox_runtime_missing')
        directory = Path(cwd or self.write_root).resolve(strict=True)
        if not directory.is_relative_to(self.write_root):
            raise AssetSandboxError('asset_sandbox_cwd_outside_attempt')
        executable = Path(argv[0]).resolve(strict=True)
        roots = (*self.read_roots, *self.executable_roots)
        if executable != Path('/usr/bin/true') and not any(
            executable == p or executable.is_relative_to(p) for p in roots
        ):
            raise AssetSandboxError('asset_sandbox_executable_not_admitted')
        environment = {
            'PATH': '/usr/bin:/bin', 'HOME': str(self.write_root),
            'TMPDIR': str(self.write_root / 'tmp'),
            'XDG_RUNTIME_DIR': str(self.write_root / 'xdg'),
            'XDG_CACHE_HOME': str(self.write_root / 'cache'),
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
        }
        # Only loader settings needed for the explicitly mounted CAD closure.
        for name in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
            if env and name in env:
                paths = str(env[name]).split(os.pathsep)
                if any(not any(Path(p).resolve().is_relative_to(r) for r in roots)
                       for p in paths if p):
                    raise AssetSandboxError('asset_sandbox_loader_path_not_admitted')
                environment[name] = str(env[name])
        (self.write_root / 'tmp').mkdir(exist_ok=True)
        (self.write_root / 'xdg').mkdir(exist_ok=True, mode=0o700)
        (self.write_root / 'cache').mkdir(exist_ok=True)
        if self.system == 'Darwin':
            readable = ['/System', '/usr', '/bin', '/sbin', '/Library/Fonts',
                        '/Library/Apple', '/private/etc/fonts', '/private/etc/localtime',
                        '/private/preboot/Cryptexes', '/dev/null', '/dev/urandom',
                        '/dev/random', '/dev/autofs_nowait', '/dev/dtracehelper',
                        '/private/var/db/timezone', str(self.write_root)]
            readable += [str(p) for p in roots]
            # sysctl/mach access is required by Metal/Accelerate even for CPU
            # render initialization. Neither grants file or network access.
            profile = '\n'.join([
                '(version 1)', '(deny default)', '(allow process*)',
                '(allow sysctl-read)', '(allow mach-lookup)', '(allow ipc-posix*)',
                '(allow iokit-open (iokit-user-client-class "IOSurfaceRootUserClient") '
                '(iokit-user-client-class "AGXDeviceUserClient"))',
                '(allow user-preference-read)',
                '(allow file-map-executable)',
                '(allow file-read-metadata)',
                '(allow file-read-data (literal "/"))',
                '(allow file-read* ' + ' '.join('(subpath ' + json.dumps(p) + ')'
                                               for p in readable) + ')',
                '(allow file-write* (subpath ' + json.dumps(str(self.write_root)) +
                ') (literal "/dev/null"))',
            ])
            command = [self.launcher, '-p', profile, *map(str, argv)]
        else:
            privileged = os.geteuid() == 0
            namespaces = (['--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net']
                          if privileged else ['--unshare-all'])
            command = [self.launcher, *namespaces, '--die-with-parent',
                       '--new-session', '--proc', '/proc', '--dev', '/dev',
                       '--perms', '0755', '--tmpfs', '/tmp']
            system_paths = [Path(path) for path in ['/usr', '/bin', '/lib', '/lib64',
                '/etc/fonts', '/etc/ld.so.cache', '/etc/localtime'] if Path(path).exists()]
            # Deep bind targets otherwise acquire root-owned 0700 ancestors.
            # Create traversable EMPTY namespace directories before mounting
            # anything. This never chmods or exposes host ancestor contents.
            parents = {parent for target in (*system_paths, *roots, self.write_root)
                       for parent in target.parents if str(parent) not in {'/', '/tmp', '/proc', '/dev'}}
            for parent in sorted(parents, key=lambda value: (len(value.parts), str(value))):
                command += ['--perms', '0755', '--dir', str(parent)]
            for path in system_paths:
                command += ['--ro-bind', str(path), str(path)]
            for root in dict.fromkeys(roots):
                command += ['--ro-bind', str(root), str(root)]
            command += ['--bind', str(self.write_root), str(self.write_root),
                        '--chdir', str(directory), '--']
            if privileged:
                # A trusted service creates kernel namespaces where the host
                # forbids unprivileged user namespaces, then drops all host
                # privileges before the candidate interpreter starts.
                try:
                    account = pwd.getpwnam('blueprint')
                except KeyError:
                    account = pwd.getpwnam('nobody')
                for folder in {self.write_root, directory, self.write_root / 'tmp',
                               self.write_root / 'xdg', self.write_root / 'cache',
                               *[p for p in directory.parents if p.is_relative_to(self.write_root)]}:
                    os.chown(folder, account.pw_uid, account.pw_gid)
                command += ['/usr/bin/setpriv', f'--reuid={account.pw_uid}',
                            f'--regid={account.pw_gid}', '--clear-groups',
                            '--no-new-privs', '--']
            command += list(map(str, argv))
        # Never inherit a caller's stdin handles or credentials. Bound outputs
        # on readback; subprocess wall time remains enforced by the parent.
        if kwargs:
            raise AssetSandboxError('asset_sandbox_subprocess_option_not_admitted')
        return subprocess.run(command, cwd=directory, env=environment,
                              timeout=min(float(timeout), 900), check=check,
                              capture_output=capture_output, text=text)
