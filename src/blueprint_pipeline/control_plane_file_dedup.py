"""Deduplicate extents without aliasing writable file identities.

Linux FIDEDUPERANGE compares bytes in the kernel and preserves copy-on-write
semantics. Unsupported filesystems are skipped; there is no hard-link fallback.
UAPI: include/uapi/linux/fs.h, struct file_dedupe_range.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
import stat
import struct
import sys

FIDEDUPERANGE = 0xC0189436
CHUNK_BYTES = 16 * 1024 * 1024


def deduplicate_pair(source, destination, *, apply=False):
    result = {"source": str(source), "destination": str(destination), "status": "skipped", "bytes_deduplicated": 0}
    try:
        with ExitStack() as stack:
            src = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            stack.callback(os.close, src)
            dst = os.open(destination, (os.O_RDWR if apply else os.O_RDONLY) | os.O_NOFOLLOW)
            stack.callback(os.close, dst)
            left, right = os.fstat(src), os.fstat(dst)
            if (not stat.S_ISREG(left.st_mode) or not stat.S_ISREG(right.st_mode)
                    or left.st_dev != right.st_dev or left.st_ino == right.st_ino
                    or left.st_size != right.st_size):
                return {**result, "reason": "ineligible_pair"}
            if not apply:
                return {**result, "status": "candidate", "candidate_bytes": left.st_size}
            if not sys.platform.startswith('linux'):
                return {**result, "reason": "unsupported_platform"}
            offset = 0
            while offset < left.st_size:
                size = min(CHUNK_BYTES, left.st_size - offset)
                request = bytearray(struct.pack('=QQHHIqQQiI', offset, size, 1, 0, 0, dst, offset, 0, 0, 0))
                fcntl.ioctl(src, FIDEDUPERANGE, request, True)
                _, _, done, status, _ = struct.unpack_from('=qQQiI', request, 24)
                if status != 0 or done <= 0 or done > size:
                    return {**result, "reason": "different_or_unsupported", "kernel_status": status}
                result['bytes_deduplicated'] += done
                offset += done
            return {**result, "status": "deduplicated"}
    except OSError as exc:
        return {**result, "reason": "os_error", "errno": exc.errno}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', required=True, help='Read-only candidate scan JSON; never trusted as byte equality proof.')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    with open(args.groups) as stream:
        groups = json.load(stream)
    results = []
    for group in groups:
        paths = group.get('paths') or []
        for target in paths[1:]:
            results.append(deduplicate_pair(paths[0], target, apply=args.apply))
    print(json.dumps({'schema_version': 'control_plane_file_dedup.v1', 'applied': args.apply,
                      'bytes_deduplicated': sum(r['bytes_deduplicated'] for r in results),
                      'hardlinks_created': 0, 'results': results}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
