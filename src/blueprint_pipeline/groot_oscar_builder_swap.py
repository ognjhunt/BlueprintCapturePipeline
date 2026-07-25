"""Ephemeral swap provisioning for the CPU image builder.

The approved builder profile has 16 GiB of RAM.  That is enough to build and
push the thin release, but not to survive the late-stage
``syft registry:<release>`` SPDX scan, whose peak SIGKILLs the scan (exit 137)
*after* the image has already been pushed.  Everything downstream of that scan
-- ``validate-thin-release``, the thin-release contract, tag promotion, and the
build-result evidence -- is then skipped, so the build yields no usable release
evidence.

Every prior successful build carried a hand-created swapfile recorded only as a
``swap_evidence.json`` artifact with no code behind it.  This module encodes
that step so it cannot be forgotten between runs.
"""

from __future__ import annotations

SWAPFILE_PATH = "/swapfile"
# Sized against the reproduced syft registry-scan peak on the 16 GiB builder
# profile, not chosen as a round number.
SWAP_GIB = 16

#: Shell predicate that is true only when the swapfile is actually swapped on.
#: ``swapon --show`` reports active swap, so this cannot pass on a host where
#: the file merely exists.
SWAP_ACTIVE_CHECK = f"swapon --show=NAME --noheadings | grep -qx {SWAPFILE_PATH}"


def provision_runcmd_lines() -> str:
    """Return cloud-init ``runcmd`` lines that make swap active.

    Idempotent: an existing swapfile is reused, and an already-active swap is
    not re-enabled.  ``fallocate`` falls back to ``dd`` for filesystems that do
    not support preallocation.
    """

    return (
        f"  - bash -c 'test -e {SWAPFILE_PATH}"
        f" || fallocate -l {SWAP_GIB}G {SWAPFILE_PATH}"
        f" || dd if=/dev/zero of={SWAPFILE_PATH} bs=1M count={SWAP_GIB * 1024}'\n"
        f"  - chmod 0600 {SWAPFILE_PATH}\n"
        f"  - bash -c '{SWAP_ACTIVE_CHECK}"
        f" || (mkswap {SWAPFILE_PATH} && swapon {SWAPFILE_PATH})'"
    )
