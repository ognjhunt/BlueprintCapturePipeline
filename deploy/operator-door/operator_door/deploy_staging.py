"""Install or observe the fixed, guarded work-volume temporary deploy parent."""
from __future__ import annotations

import os
import pwd
from pathlib import Path
import stat
import sys

ROOT = Path("/")
ROOT_UID = 0
WORK = ("mnt", "blueprint-work")
STAGING = (*WORK, "control-plane-deploy-staging")


def _service_groups() -> set[int]:
    account = pwd.getpwnam("blueprint")
    if account.pw_uid == ROOT_UID:
        raise ValueError("deploy_staging_service_account_invalid")
    return set(os.getgrouplist(account.pw_name, account.pw_gid))


def _device(descriptor: int) -> int:
    return os.fstat(descriptor).st_dev


def _check(descriptor: int) -> None:
    info = os.fstat(descriptor)
    permissions = (info.st_mode >> 3) if info.st_gid in _service_groups() else info.st_mode
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != ROOT_UID
            or info.st_mode & 0o022 or permissions & 0o005 != 0o005):
        raise ValueError("deploy_staging_parent_unsafe")


def _open(parts: tuple[str, ...]) -> int:
    descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _check(descriptor)
        for part in parts:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
            _check(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def validated_staging_parent() -> str:
    """Keep admission/builders on one path; preserve service-user traversal."""
    descriptors: list[int] = []
    try:
        for parts in ((), WORK, STAGING):
            descriptors.append(_open(parts))
        root, work, staging = descriptors
        if _device(work) == _device(root) or _device(staging) != _device(work):
            raise ValueError("deploy_staging_parent_wrong_device")
        return str(ROOT.joinpath(*STAGING))
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def install_staging_parent() -> str:
    """Create only the absent fixed leaf; never repair/adopt an unsafe path."""
    if os.geteuid() != ROOT_UID:
        raise ValueError("deploy_staging_installer_requires_root")
    root, work = _open(()), None
    try:
        work = _open(WORK)
        if _device(work) == _device(root):
            raise ValueError("deploy_staging_parent_wrong_device")
        try:
            os.mkdir(STAGING[-1], mode=0o755, dir_fd=work)
        except FileExistsError:
            pass
        else:
            leaf = os.open(STAGING[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=work)
            try:
                # Only this invocation's new directory is adjusted for umask.
                os.fchmod(leaf, 0o755)
            finally:
                os.close(leaf)
    finally:
        if work is not None:
            os.close(work)
        os.close(root)
    return validated_staging_parent()


def main() -> int:
    try:
        if sys.argv[1:] == ["--install"]:
            print(install_staging_parent())
            return 0
        if len(sys.argv) != 1:
            raise ValueError("deploy_staging_parent_arguments_refused")
        print(validated_staging_parent())
        return 0
    except (OSError, ValueError, KeyError):
        print("deploy_staging_parent_unsafe", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
