#!/usr/bin/env python3
"""Collapse byte-identical retained control-plane files onto shared inodes via hardlinks.

Deletes nothing. Every path survives, resolving to identical bytes. This is the
canonical copy of the operator tool that ran as ``/root/dedup_apply.py`` on the
control plane (2026-09-13: 10.54 GiB reclaimed across 680 links); the host copy
must match this file.

Why this exists next to ``control_plane_hardlink_dedup.py``: that command now
delegates to kernel copy-on-write extent deduplication (FIDEDUPERANGE), which
the droplet's ext4 root does not support, so it reclaims nothing there. Retained
launch inputs are written once and then only read, so sharing an inode between
byte-identical copies is safe under the two rules below; this remains the only
reclaim lever on ext4.

Hard links share future writes, so a link is only safe between files nobody is
still writing. Equal bytes at one instant are not enough (2026-09-13 audit): a
write landing between hashing and linking would be replaced by the keeper's bytes.
Two rules close that window as far as an unsynchronised tool can:

  * quiescence: a file modified within ``MINIMUM_AGE_SECONDS`` is never touched;
    retained launch inputs are written once and then only read;
  * identity re-proof: keeper and victim are re-stat'ed immediately before the
    link and must carry the exact identity (device, inode, size, mode, owner,
    nanosecond mtime and kernel-owned ctime) they had when their bytes were hashed.

Each duplicate group is partitioned by (mode, uid, gid) and deduped only
WITHIN a partition, so no file's permissions or ownership ever change.

Safety gates before any link:
  * re-hash keeper and victim NOW (scan data is not trusted)
  * identical size / mode / uid / gid / filesystem
  * excluded prefixes (actively-written evidence) never touched
  * atomic replace: link to temp name, rename over victim, so the path is
    never absent even if the process dies mid-flight
"""
import argparse
import hashlib
import json
import os
import stat as statmod
import time

MINIMUM_AGE_SECONDS = 1800

EXCLUDE_PREFIXES = (
    "/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard",
    "/var/lib/blueprint/preserved-scratch-evidence",
    "/var/lib/blueprint/pipeline-control-plane/intake_nonce_store",
    "/var/lib/blueprint/pipeline-control-plane/standing-authorizations",
    "/var/lib/blueprint/pipeline-control-plane/deploy-receipts",
    "/var/lib/blueprint/pipeline-control-plane/retention-receipts",
)


def digest(path):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _identity(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mode, st.st_uid, st.st_gid,
            st.st_mtime_ns, st.st_ctime_ns)


def _unchanged(path, st):
    """True when ``path`` still carries exactly the identity its bytes were hashed under."""
    try:
        return _identity(os.lstat(path)) == _identity(st)
    except OSError:
        return False


def _after_own_link(path, before, expected_links, expected_digest):
    """Accept our link-count/ctime change only after re-proving unchanged bytes."""
    current = os.lstat(path)
    if (_identity(current)[:-1] != _identity(before)[:-1]
            or current.st_nlink != expected_links
            or digest(path) != expected_digest
            or not _unchanged(path, current)):
        raise OSError("changed after own link")
    return current


def dedup_partition(live, apply_, skipped, *, minimum_age_seconds=MINIMUM_AGE_SECONDS, now=None):
    """live: [(path, stat)] all sharing size/mode/uid/gid. Returns (bytes, links)."""
    now = time.time() if now is None else now
    quiet = []
    for p, s in live:
        if now - s.st_mtime < minimum_age_seconds or now - s.st_ctime < minimum_age_seconds:
            skipped.append("%s: modified within the last %ds" % (p, minimum_age_seconds))
            continue
        quiet.append((p, s))
    live = quiet
    if len({(s.st_dev, s.st_ino) for _, s in live}) < 2:
        return 0, 0

    live.sort(key=lambda ps: (-ps[1].st_nlink, ps[0]))
    keeper, kst = live[0]
    try:
        kdig = digest(keeper)
    except OSError as e:
        skipped.append("%s: unreadable (%s)" % (keeper, e))
        return 0, 0

    by_inode = {}
    for p, s in live[1:]:
        if (s.st_dev, s.st_ino) == (kst.st_dev, kst.st_ino):
            continue
        by_inode.setdefault((s.st_dev, s.st_ino), []).append((p, s))

    freed = links = 0
    for (vdev, _vino), entries in by_inode.items():
        vst = entries[0][1]
        if vdev != kst.st_dev:
            skipped.append("%s: different filesystem" % entries[0][0])
            continue
        try:
            if digest(entries[0][0]) != kdig:
                skipped.append("%s: CONTENT CHANGED since scan" % entries[0][0])
                continue
        except OSError as e:
            skipped.append("%s: unreadable (%s)" % (entries[0][0], e))
            continue
        # Re-prove both files are still the bytes that were hashed: any write between
        # hashing and linking moves the nanosecond mtime and the kernel-owned ctime.
        if not _unchanged(keeper, kst):
            skipped.append("%s: keeper changed during verification" % keeper)
            return freed, links
        if not all(_unchanged(p, s) for p, s in entries):
            skipped.append("%s: changed during verification" % entries[0][0])
            continue

        # blocks return only if every link to this inode is in our set
        frees_blocks = len(entries) >= vst.st_nlink

        ok = 0
        for index, (victim, _) in enumerate(entries):
            if not apply_:
                ok += 1
                continue
            tmp = "%s.dedup-tmp-%d" % (victim, os.getpid())
            try:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
                if not _unchanged(keeper, kst) or not _unchanged(victim, vst):
                    raise OSError("changed before link")
                os.link(keeper, tmp)
                os.rename(tmp, victim)  # atomic
                ok += 1
                # link() advances the keeper's ctime; rename() decrements the
                # victim inode's links and advances its remaining aliases' ctime.
                # Reusing the pre-link stats falsely refuses the rest of a group.
                kst = _after_own_link(keeper, kst, kst.st_nlink + 1, kdig)
                if index + 1 < len(entries):
                    vst = _after_own_link(entries[index + 1][0], vst, vst.st_nlink - 1, kdig)
            except OSError as e:
                skipped.append("%s: link failed (%s)" % (victim, e))
                try:
                    if os.path.lexists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
        links += ok
        if frees_blocks and ok == len(entries):
            freed += vst.st_size
    return freed, links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="/root/dedup_groups.json")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--skip-first", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--minimum-age-seconds", type=int, default=MINIMUM_AGE_SECONDS,
                    help="never touch a file modified more recently than this (quiescence window)")
    args = ap.parse_args()

    groups = json.load(open(args.groups))
    groups.sort(key=lambda g: -g["reclaim"])
    if args.skip_first:
        groups = groups[args.skip_first:]
    if args.top:
        groups = groups[: args.top]

    reclaimed = linked = 0
    skipped = []

    for gi, g in enumerate(groups, 1):
        parts = {}
        for p in g["paths"]:
            if p.startswith(EXCLUDE_PREFIXES):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if not statmod.S_ISREG(st.st_mode):
                continue
            key = (st.st_size, statmod.S_IMODE(st.st_mode), st.st_uid, st.st_gid)
            parts.setdefault(key, []).append((p, st))

        for _key, live in parts.items():
            if len(live) < 2:
                continue
            freed_bytes, made_links = dedup_partition(live, args.apply, skipped, minimum_age_seconds=args.minimum_age_seconds)
            reclaimed += freed_bytes
            linked += made_links

        if gi % 50 == 0:
            print("  ...group %d/%d  %.2f GiB  %d links"
                  % (gi, len(groups), reclaimed / 2 ** 30, linked), flush=True)

    print("\n=== %s ===" % ("APPLIED" if args.apply else "DRY RUN"))
    print("links made:      %d" % linked)
    print("space reclaimed: %.2f GiB" % (reclaimed / 2 ** 30))
    print("skipped:         %d" % len(skipped))
    for s in skipped[:15]:
        print("    SKIP %s" % s)
    if len(skipped) > 15:
        print("    ... +%d more" % (len(skipped) - 15))


if __name__ == "__main__":
    main()
