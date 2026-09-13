#!/usr/bin/env python3
"""Compatibility CLI for kernel-verified copy-on-write extent deduplication.

Retains the historical scan, filtering and dry-run arguments, but never links
or replaces pathnames. Unsupported filesystems are skipped. Run from the
installed release so blueprint_pipeline.control_plane_file_dedup is available.
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


def dedup_partition(live, apply_, skipped, *, minimum_age_seconds=MINIMUM_AGE_SECONDS, now=None):
    """live: [(path, stat)] all sharing size/mode/uid/gid. Returns (kernel-deduplicated bytes, extent pairs)."""
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

    from blueprint_pipeline.control_plane_file_dedup import deduplicate_pair
    keeper = live[0][0]
    shared = pairs = 0
    for victim, _ in live[1:]:
        result = deduplicate_pair(keeper, victim, apply=apply_)
        if result["status"] == "deduplicated":
            shared += result["bytes_deduplicated"]
            pairs += 1
        elif result["status"] != "candidate":
            skipped.append("%s: %s" % (victim, result.get("reason", "skipped")))
    return shared, pairs


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
    print("extent pairs:      %d" % linked)
    print("bytes deduplicated: %.2f GiB" % (reclaimed / 2 ** 30))
    print("skipped:         %d" % len(skipped))
    for s in skipped[:15]:
        print("    SKIP %s" % s)
    if len(skipped) > 15:
        print("    ... +%d more" % (len(skipped) - 15))


if __name__ == "__main__":
    main()
