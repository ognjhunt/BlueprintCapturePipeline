"""Which directories of a site root are buckets: the one rule every reader of a site's withdrawal shares.

A site root holds ``scenes/<site>/`` itself and ``<bucket>/scenes/<site>/`` for each bucket beside it. A bucket is
named after its storage bucket, which never starts with a dot. The hidden folders beside the buckets are spools,
evidence and caches (the storage clean-up's root-only ``.scene-workspace-inventory`` among them), so a hidden folder
is never a bucket. ``scenes`` itself is not a bucket either.
"""
from __future__ import annotations


def is_bucket(name: str) -> bool:
    return name != "scenes" and not name.startswith(".")


__all__ = ["is_bucket"]
