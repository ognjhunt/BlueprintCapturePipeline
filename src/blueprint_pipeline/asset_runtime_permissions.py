"""Make explicitly admitted runtime code readable without Linux DAC capabilities."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import stat


def prepare_runtime_code_access(roots):
    """Trusted setup only, before generated code runs; never add write access.

    Vendor images may install Python under a foreign UID's private directories.
    Container root can read those with CAP_DAC_OVERRIDE, but the sandbox cannot
    after clearing its capabilities. Preserve bytes, ownership, and write bits;
    add only the read/search bits needed by the current UID without capabilities.
    Callers supply runtime-code roots, never capture data or credential roots.
    """
    report = {'status': 'not_required', 'changed_count': 0, 'changed_paths': []}
    if platform.system() != 'Linux' or os.geteuid() != 0:
        return report
    groups = {*os.getgroups(), os.getegid()}
    uid = os.geteuid()

    def admit(path, *, search_only=False):
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            return  # Never follow an external link to another tree.
        if not (stat.S_ISDIR(before.st_mode) or stat.S_ISREG(before.st_mode)):
            return
        shift = 6 if before.st_uid == uid else 3 if before.st_gid in groups else 0
        needed = 1 if search_only else 5 if stat.S_ISDIR(before.st_mode) else (
            5 if before.st_mode & 0o111 else 4)
        mode = stat.S_IMODE(before.st_mode)
        updated = mode | (needed << shift)
        if updated == mode:
            return
        # Only immutable, trusted runtime directories are normalized here.
        os.chmod(path, updated, follow_symlinks=False)
        after = path.lstat()
        if (after.st_dev, after.st_ino, after.st_uid, after.st_gid, after.st_size) != (
                before.st_dev, before.st_ino, before.st_uid, before.st_gid, before.st_size):
            raise RuntimeError('asset_runtime_identity_changed')
        report['changed_count'] += 1
        report['changed_paths'].append({'path': str(path), 'before_mode': oct(mode),
                                        'after_mode': oct(stat.S_IMODE(after.st_mode))})

    for root in dict.fromkeys(Path(p).resolve(strict=True) for p in roots):
        # Standard system roots are not a vendor-private runtime tree, and may
        # include unrelated private installations. Never broaden them wholesale.
        if root in {Path('/usr'), Path('/usr/local')}:
            continue
        if root in {Path('/'), Path('/etc'), Path('/var'), Path('/home'), Path('/root')}:
            raise ValueError('asset_runtime_code_root_too_broad')
        for parent in reversed(root.parents):
            if parent != Path('/'):
                admit(parent, search_only=True)
        admit(root)
        if root.is_dir():
            for directory, names, files in os.walk(root, followlinks=False):
                for name in [*names, *files]:
                    admit(Path(directory) / name)
    report['status'] = 'ready'
    return report
